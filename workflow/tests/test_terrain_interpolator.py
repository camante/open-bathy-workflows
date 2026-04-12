import numpy as np
import pytest

from support_classes import SupportClass
from provenance_schema import ProvenanceClass
from river_bank_guidance import _weighted_pava_nonincreasing
from terrain_interpolator import (
    TerrainInterpolationConfig,
    TerrainInterpolationInputs,
    interpolate_support_aware_surface,
)


def test_interpolator_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        interpolate_support_aware_surface(
            inputs=TerrainInterpolationInputs(
                candidate=np.zeros((3, 3), dtype=np.float32),
                auth=np.zeros((4, 4), dtype=np.float32),
                sdb_ok=np.zeros((3, 3), dtype=bool),
                river_ok=np.zeros((3, 3), dtype=bool),
            ),
            config=TerrainInterpolationConfig(pixel_size_m=10.0),
        )


def test_interpolator_hard_locks_authoritative_and_reports_low_confidence_backstop():
    candidate = np.array([[np.nan, np.nan, np.nan], [np.nan, 5.0, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    auth = np.array([[1.0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=np.zeros_like(candidate, dtype=bool),
            river_ok=np.zeros_like(candidate, dtype=bool),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=1.0, support_decay_m=10.0, support_density_radius_m=10.0),
    )
    assert np.isfinite(out["conditioned"]).all()
    assert int(out["support"][0, 0]) == int(SupportClass.AUTHORITATIVE_LOCKED)
    assert int(out["provenance"][0, 0]) == int(ProvenanceClass.AUTHORITATIVE_LOCKED)
    assert int(out["support"][2, 2]) == int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)
    assert int(out["provenance"][2, 2]) == int(ProvenanceClass.LOW_CONFIDENCE_FILL)
    assert out["support_note"].startswith("terrain_interpolator")


def test_interpolator_uses_river_support_depth_as_anchor_surface():
    candidate = np.full((5, 5), 10.0, dtype=np.float32)
    auth = np.full((5, 5), np.nan, dtype=np.float32)
    auth[2, 2] = 1.0
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[1:4, 1:4] = True
    river_support = np.zeros((5, 5), dtype=np.uint8)
    river_support[2, 1] = 1
    river_support_depth = np.full((5, 5), np.nan, dtype=np.float32)
    river_support_depth[2, 1] = 3.25
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=np.zeros((5, 5), dtype=bool),
            river_ok=river_ok,
            river_support=river_support,
            river_support_depth=river_support_depth,
            river_gw=np.where(river_ok & ~np.isfinite(auth), 0.8, 0.0).astype(np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert np.isfinite(out["conditioned"][2, 1])
    assert out["conditioned"][2, 1] < candidate[2, 1]


def test_interpolator_can_run_without_prebuilt_candidate():
    from terrain_interpolator import TerrainInterpolationConfig, TerrainInterpolationInputs, interpolate_support_aware_surface
    auth = np.array([[1.0, np.nan, 3.0],[1.5, np.nan, 3.5],[2.0, np.nan, 4.0]], dtype=np.float32)
    sdb_ok = np.array([[False, True, False],[False, True, False],[False, True, False]], dtype=bool)
    river_ok = np.zeros_like(sdb_ok, dtype=bool)
    sdb_depth = np.array([[np.nan, 2.0, np.nan],[np.nan, 2.5, np.nan],[np.nan, 3.0, np.nan]], dtype=np.float32)
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            candidate=np.zeros_like(auth, dtype=np.float32),
            sdb_depth_guidance=sdb_depth,
            sdb_ok=sdb_ok,
            river_ok=river_ok,
            sdb_gw=np.ones_like(sdb_depth, dtype=np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=3.0),
    )
    assert np.isfinite(out["guidance_surface"][1, 1])
    assert np.isfinite(out["conditioned"][1, 1])


def test_interpolator_can_rasterize_native_guide_points_inside_engine(monkeypatch, tmp_path):
    import terrain_interpolator as ti

    def _fake_sdb(path, template_raster, logger=None):
        return np.array([[np.nan, 2.0, np.nan],[np.nan, 2.5, np.nan],[np.nan, 3.0, np.nan]], dtype=np.float32)

    monkeypatch.setattr(ti, "log", ti.log)
    import sdb_guidance
    monkeypatch.setattr(sdb_guidance, "rasterize_sdb_guide_points_to_template", _fake_sdb)

    auth = np.array([[1.0, np.nan, 3.0],[1.5, np.nan, 3.5],[2.0, np.nan, 4.0]], dtype=np.float32)
    sdb_ok = np.array([[False, True, False],[False, True, False],[False, True, False]], dtype=bool)
    river_ok = np.zeros_like(sdb_ok, dtype=bool)
    (tmp_path / "template.tif").write_text("template")
    out = ti.interpolate_support_aware_surface(
        inputs=ti.TerrainInterpolationInputs(
            auth=auth,
            candidate=None,
            sdb_guide_points_path="fake.gpkg",
            guidance_template_raster=str((tmp_path / "template.tif").resolve()),
            sdb_ok=sdb_ok,
            river_ok=river_ok,
            sdb_gw=np.ones_like(auth, dtype=np.float32),
        ),
        config=ti.TerrainInterpolationConfig(pixel_size_m=3.0),
    )
    assert np.isfinite(out["guidance_surface"][1, 1])
    assert np.isfinite(out["conditioned"][1, 1])


def test_interpolator_requires_template_for_native_guide_points():
    auth = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32)
    with pytest.raises(ValueError, match="guidance_template_raster is required"):
        interpolate_support_aware_surface(
            inputs=TerrainInterpolationInputs(
                auth=auth,
                sdb_guide_points_path="fake.gpkg",
                sdb_ok=np.array([[False, True], [False, True]], dtype=bool),
                river_ok=np.zeros((2, 2), dtype=bool),
            ),
            config=TerrainInterpolationConfig(pixel_size_m=3.0),
        )


def test_interpolator_rejects_resolved_native_guidance_shape_mismatch(monkeypatch, tmp_path):
    import terrain_interpolator as ti

    template = tmp_path / "template.tif"
    template.write_text("placeholder", encoding="utf-8")

    def _fake_sdb(path, template_raster, logger=None):
        return np.ones((3, 3), dtype=np.float32)

    import sdb_guidance
    monkeypatch.setattr(sdb_guidance, "rasterize_sdb_guide_points_to_template", _fake_sdb)

    auth = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32)
    with pytest.raises(ValueError, match="resolved shape"):
        ti.interpolate_support_aware_surface(
            inputs=ti.TerrainInterpolationInputs(
                auth=auth,
                sdb_guide_points_path="fake.gpkg",
                guidance_template_raster=str(template),
                sdb_ok=np.array([[False, True], [False, True]], dtype=bool),
                river_ok=np.zeros((2, 2), dtype=bool),
                sdb_gw=np.ones((2, 2), dtype=np.float32),
            ),
            config=ti.TerrainInterpolationConfig(pixel_size_m=3.0),
        )


def test_interpolator_preserves_baseline_background_outside_guidance_domains():
    auth = np.array([[1.0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    background = np.array([[1.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], dtype=np.float32)
    river_ok = np.array([[False, False, False], [False, True, False], [False, False, False]], dtype=bool)
    river_depth = np.full((3, 3), np.nan, dtype=np.float32)
    river_depth[1, 1] = 5.0
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            background_surface=background,
            primary_river_guidance_surface=river_depth,
            river_ok=river_ok,
            sdb_ok=np.zeros((3, 3), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert out["conditioned"][0, 1] == pytest.approx(11.0)
    assert out["conditioned"][2, 2] == pytest.approx(18.0)
    assert int(out["support"][0, 1]) == int(SupportClass.ANCHORED_INTERPOLATION)
    assert int(out["support"][2, 2]) == int(SupportClass.ANCHORED_INTERPOLATION)
    assert int(out["support"][1, 1]) in {int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.SCAFFOLD_INFERRED)}

def test_interpolator_reports_authoritative_first_contract_and_preserves_baseline_exact_domain():
    auth = np.array([[2.0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    background = np.array([[2.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]], dtype=np.float32)
    river_ok = np.array([[False, False, False], [False, True, False], [False, False, False]], dtype=bool)
    river_depth = np.full((3, 3), np.nan, dtype=np.float32)
    river_depth[1, 1] = 5.0
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            background_surface=background,
            primary_river_guidance_surface=river_depth,
            river_ok=river_ok,
            sdb_ok=np.zeros((3, 3), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    contract = out["authoritative_first_contract"]
    assert contract["ok"] is True
    assert contract["locked_changed_count"] == 0
    assert contract["background_changed_count"] == 0
    assert contract["low_confidence_fill_outside_guidance_count"] == 0
    baseline_exact_domain = out["baseline_exact_domain"].astype(bool)
    assert baseline_exact_domain[0, 1]
    assert out["conditioned"][0, 1] == pytest.approx(background[0, 1])
    assert int(out["support"][0, 1]) == int(SupportClass.ANCHORED_INTERPOLATION)
    assert out["guidance_influence"][0, 1] == pytest.approx(0.0)


def test_interpolator_reenforces_locked_cells_even_if_candidate_matches_wrong_value():
    auth = np.array([[7.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    candidate = np.full((2, 2), 50.0, dtype=np.float32)
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            candidate=candidate,
            sdb_ok=np.zeros((2, 2), dtype=bool),
            river_ok=np.zeros((2, 2), dtype=bool),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=5.0),
    )
    assert out["conditioned"][0, 0] == pytest.approx(7.0)
    assert int(out["support"][0, 0]) == int(SupportClass.AUTHORITATIVE_LOCKED)
    assert out["authoritative_first_contract"]["locked_changed_count"] == 0


def test_interpolator_builds_primary_river_surface_from_longitudinal_guidance_before_candidate():
    auth = np.array([[2.0, np.nan, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan]], dtype=np.float32)
    candidate = np.full((5, 5), 100.0, dtype=np.float32)
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[:, 1:4] = True
    centerline_stationing = np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 5))
    longitudinal = np.full((5, 5), np.nan, dtype=np.float32)
    longitudinal[:, 2] = np.array([3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    longitudinal_influence = np.zeros((5, 5), dtype=np.float32)
    longitudinal_influence[:, 1:4] = 0.9
    centerline_influence = np.zeros((5, 5), dtype=np.float32)
    centerline_influence[:, 1:4] = 1.0
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            candidate=candidate,
            river_ok=river_ok,
            sdb_ok=np.zeros((5, 5), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_centerline_stationing=centerline_stationing,
            river_longitudinal_profile_elevation=longitudinal,
            river_longitudinal_profile_influence=longitudinal_influence,
            river_centerline_influence=centerline_influence,
            river_contract_mode="legacy_fallbacks",
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert np.isfinite(out["river_primary_surface"][2, 1])
    assert out["guidance_surface"][2, 1] == pytest.approx(out["river_primary_surface"][2, 1])
    assert out["guidance_surface"][2, 1] < 20.0
    assert out["guidance_surface"][2, 1] != pytest.approx(candidate[2, 1])


def test_interpolator_expands_primary_river_surface_across_channel_interior():
    auth = np.array([[1.0, np.nan, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan],
                     [np.nan, np.nan, np.nan, np.nan, np.nan]], dtype=np.float32)
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[:, 1:4] = True
    centerline_stationing = np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 5))
    longitudinal = np.full((5, 5), np.nan, dtype=np.float32)
    longitudinal[:, 2] = 4.0
    longitudinal_influence = np.zeros((5, 5), dtype=np.float32)
    longitudinal_influence[:, 1:4] = 0.8
    centerline_influence = np.zeros((5, 5), dtype=np.float32)
    centerline_influence[:, 1:4] = 1.0
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            river_ok=river_ok,
            sdb_ok=np.zeros((5, 5), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_centerline_stationing=centerline_stationing,
            river_longitudinal_profile_elevation=longitudinal,
            river_longitudinal_profile_influence=longitudinal_influence,
            river_centerline_influence=centerline_influence,
            river_contract_mode="legacy_fallbacks",
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    primary = out["river_primary_surface"]
    assert np.isfinite(primary[2, 1])
    assert np.isfinite(primary[2, 2])
    assert np.isfinite(primary[2, 3])
    assert out["river_primary_surface_confidence"][2, 1] > 0.0


def test_primary_river_surface_keeps_bank_guidance_as_boundary_control():
    auth = np.full((5, 5), np.nan, dtype=np.float32)
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[:, 1:4] = True
    centerline_stationing = np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 5))
    longitudinal = np.full((5, 5), np.nan, dtype=np.float32)
    longitudinal[:, 2] = 4.0
    longitudinal_influence = np.zeros((5, 5), dtype=np.float32)
    longitudinal_influence[:, 1:4] = 0.9
    centerline_influence = np.zeros((5, 5), dtype=np.float32)
    centerline_influence[:, 1:4] = 1.0
    bank_elev = np.full((5, 5), np.nan, dtype=np.float32)
    bank_elev[:, 1] = 20.0
    bank_elev[:, 3] = 20.0
    bank_infl = np.zeros((5, 5), dtype=np.float32)
    bank_infl[:, 1] = 1.0
    bank_infl[:, 3] = 1.0
    bank_infl[:, 2] = 0.1
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            river_ok=river_ok,
            sdb_ok=np.zeros((5, 5), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_centerline_stationing=centerline_stationing,
            river_longitudinal_profile_elevation=longitudinal,
            river_longitudinal_profile_influence=longitudinal_influence,
            river_centerline_influence=centerline_influence,
            river_bank_elevation=bank_elev,
            river_bank_influence=bank_infl,
            river_contract_mode="legacy_fallbacks",
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    primary = out["river_primary_surface"]
    # Center should stay controlled by longitudinal guidance, not get pulled to bank elevation.
    assert primary[2, 2] == pytest.approx(4.0, abs=1.0e-6)
    # Bank guidance should remain bounded and not behave like a full-domain bank-routed bed field.
    assert primary[2, 1] < 20.0
    assert abs(float(primary[2, 1]) - float(primary[2, 2])) < 2.0


def test_primary_river_surface_limits_xs_influence_to_local_along_channel_neighborhood():
    auth = np.full((9, 5), np.nan, dtype=np.float32)
    river_ok = np.zeros((9, 5), dtype=bool)
    river_ok[:, 1:4] = True
    centerline_stationing = np.tile((np.arange(9, dtype=np.float32) * 100.0)[:, None], (1, 5))
    longitudinal = np.full((9, 5), np.nan, dtype=np.float32)
    longitudinal[:, 2] = 4.0
    longitudinal_influence = np.zeros((9, 5), dtype=np.float32)
    longitudinal_influence[:, 1:4] = 0.9
    centerline_influence = np.zeros((9, 5), dtype=np.float32)
    centerline_influence[:, 1:4] = 1.0
    xs_elev = np.full((9, 5), np.nan, dtype=np.float32)
    xs_elev[4, 1:4] = 10.0
    xs_w = np.zeros((9, 5), dtype=np.float32)
    xs_w[4, 1:4] = 1.0
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            river_ok=river_ok,
            sdb_ok=np.zeros((9, 5), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_centerline_stationing=centerline_stationing,
            river_longitudinal_profile_elevation=longitudinal,
            river_longitudinal_profile_influence=longitudinal_influence,
            river_centerline_influence=centerline_influence,
            river_xs_support_elevation=xs_elev,
            river_xs_support_weight=xs_w,
            river_contract_mode="legacy_fallbacks",
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0, river_aniso_along_scale_m=500.0, river_aniso_cross_scale_m=30.0),
    )
    primary = out["river_primary_surface"]
    # XS guidance should not independently raise the river bed above the
    # longitudinal backbone in unsupported reaches.
    assert primary[4, 2] == pytest.approx(4.0, abs=1.0e-6)
    # Far enough upstream/downstream, the XS should no longer own the surface and
    # the longitudinal backbone should remain dominant.
    assert primary[0, 2] == pytest.approx(4.0, abs=1.0e-6)
    assert primary[8, 2] == pytest.approx(4.0, abs=1.0e-6)


def test_river_primary_surface_contract_rejects_invalid_source_and_support_leakage():
    from river_primary_surface_contract import validate_river_primary_surface_contract

    surface = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    conf = np.array([[0.8, 0.0], [0.0, 0.0]], dtype=np.float32)
    source = np.array([[9, 1], [0, 0]], dtype=np.uint8)
    support = np.array([[9, 0], [1, 0]], dtype=np.uint8)
    domain = np.array([[1, 0], [0, 0]], dtype=bool)

    contract = validate_river_primary_surface_contract(
        primary_surface=surface,
        primary_confidence=conf,
        source_class=source,
        support_count=support,
        domain=domain,
    )

    assert contract["ok"] is False
    assert "invalid_source_class_code_inside_domain" in contract["failures"]
    assert "support_count_exceeds_expected_range" in contract["failures"]
    assert "source_class_outside_domain" in contract["failures"]
    assert "support_count_outside_domain" in contract["failures"]


def test_interpolator_reports_river_primary_surface_contract_bundle():
    auth = np.full((5, 5), np.nan, dtype=np.float32)
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[1:4, 1:4] = True
    centerline = np.full((5, 5), np.nan, dtype=np.float32)
    centerline[2, 1:4] = np.array([5.0, 4.0, 3.0], dtype=np.float32)
    station = np.full((5, 5), np.nan, dtype=np.float32)
    station[2, 1:4] = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    influence = np.zeros((5, 5), dtype=np.float32)
    influence[2, 1:4] = 1.0
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            river_ok=river_ok,
            sdb_ok=np.zeros((5, 5), dtype=bool),
            river_centerline_elevation=centerline,
            river_centerline_stationing=station,
            river_centerline_influence=influence,
            river_longitudinal_profile_elevation=centerline,
            river_longitudinal_profile_influence=influence,
            river_xs_support_elevation=np.full((5, 5), np.nan, dtype=np.float32),
            river_xs_support_weight=np.zeros((5, 5), dtype=np.float32),
            river_bank_elevation=np.full((5, 5), np.nan, dtype=np.float32),
            river_bank_influence=np.zeros((5, 5), dtype=np.float32),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_contract_mode="legacy_fallbacks",
        ),
        config=TerrainInterpolationConfig(pixel_size_m=5.0),
    )
    contract = out["river_primary_surface_contract"]
    assert contract["ok"] is True
    assert out["river_primary_surface_domain"].sum() > 0
    assert np.isfinite(out["river_primary_surface"]).any()
    assert np.all(out["river_primary_surface_source_class"][np.isfinite(out["river_primary_surface"])] > 0)
    assert np.all(out["river_primary_surface_support_count"][np.isfinite(out["river_primary_surface"])] >= 1)
    summary = out["river_primary_guidance_summary"]
    assert summary["active_product_name"] == "river_primary_surface"
    assert summary["active_product_role"] == "primary"
    assert summary["primary_builder_mode"] == "backbone_fallback"
    assert summary["primary_surface_contract_ok"] is True
    assert isinstance(summary["continuity_safeguard_used"], bool)


def test_primary_surface_contract_rejects_nonfinite_surface_metadata_leakage():
    from river_primary_surface_contract import validate_river_primary_surface_contract
    surface = np.array([[np.nan, 1.0]], dtype=np.float32)
    conf = np.array([[0.2, 0.9]], dtype=np.float32)
    source = np.array([[1, 1]], dtype=np.uint8)
    support = np.array([[1, 1]], dtype=np.uint8)
    domain = np.array([[1, 1]], dtype=bool)
    contract = validate_river_primary_surface_contract(
        primary_surface=surface,
        primary_confidence=conf,
        source_class=source,
        support_count=support,
        domain=domain,
    )
    assert not contract["ok"]
    assert "confidence_on_nonfinite_primary_surface" in contract["failures"]
    assert "source_class_on_nonfinite_primary_surface" in contract["failures"]
    assert "support_count_on_nonfinite_primary_surface" in contract["failures"]


def test_interpolator_hard_fails_when_active_river_guidance_reaches_locked_cells():
    auth = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    river_ok = np.array([[True, False], [False, False]], dtype=bool)
    with pytest.raises(RuntimeError, match="subordinate guidance inside authoritative locked cells"):
        interpolate_support_aware_surface(
            inputs=TerrainInterpolationInputs(
                auth=auth,
                sdb_ok=np.zeros((2, 2), dtype=bool),
                river_ok=river_ok,
                river_centerline_influence=np.array([[0.5, 0.0], [0.0, 0.0]], dtype=np.float32),
            ),
            config=TerrainInterpolationConfig(pixel_size_m=3.0),
        )


def test_interpolator_records_locked_guidance_contract_when_arrays_are_clean():
    auth = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            sdb_ok=np.zeros((2, 2), dtype=bool),
            river_ok=np.zeros((2, 2), dtype=bool),
            river_centerline_influence=np.array([[0.0, 0.4], [0.0, 0.0]], dtype=np.float32),
            river_centerline_elevation=np.array([[np.nan, 5.0], [np.nan, np.nan]], dtype=np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=3.0),
    )
    contract = out["guidance_locked_contract"]
    assert contract["ok"] is True
    assert contract["locked_pixels"] == 1
    assert contract["arrays"]["river_centerline_influence"]["policy"] == "zero_on_locked"
    assert contract["arrays"]["river_centerline_elevation"]["policy"] == "nan_on_locked"
    assert contract["arrays"]["river_centerline_influence"]["violating_pixels"] == 0




def test_channel_surface_prediction_confidence_tunes_river_terrain_response():
    auth = np.full((2, 2), np.nan, dtype=np.float32)
    river_ok = np.ones((2, 2), dtype=bool)
    channel_surface = np.array([[-1.0, -1.5], [-2.0, -2.5]], dtype=np.float32)
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            sdb_ok=np.zeros((2, 2), dtype=bool),
            river_ok=river_ok,
            background_surface=np.zeros((2, 2), dtype=np.float32),
            river_channel_surface=channel_surface,
            river_channel_surface_confidence=np.full((2, 2), 0.9, dtype=np.float32),
            river_channel_surface_source_class=np.ones((2, 2), dtype=np.uint8),
            river_channel_surface_support_count=np.ones((2, 2), dtype=np.uint8),
            river_channel_surface_prediction_support_confidence=np.array([[0.95, 0.20], [0.95, 0.20]], dtype=np.float32),
            river_channel_surface_measured_anchor_fraction=np.array([[0.85, 0.05], [0.85, 0.05]], dtype=np.float32),
            river_channel_surface_structure_only_fraction=np.array([[0.10, 0.90], [0.10, 0.90]], dtype=np.float32),
            river_channel_surface_low_support_caution=np.array([[0, 1], [0, 1]], dtype=np.uint8),
            river_channel_surface_prediction_admissibility=np.array([[1, 0], [1, 0]], dtype=np.uint8),
            river_gw=np.ones((2, 2), dtype=np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=3.0),
    )
    assert out["river_primary_surface_confidence"][0, 0] > out["river_primary_surface_confidence"][0, 1]
    assert out["river_primary_surface_terrain_response"][0, 0] > out["river_primary_surface_terrain_response"][0, 1]
    assert out["support"][0, 1] == int(SupportClass.SCAFFOLD_INFERRED)
    assert out["river_primary_surface_cautious_structure"][0, 1] == 1
    assert out["river_primary_surface_channel_core_preserve"][0, 0] == 0
    assert out["river_primary_surface_channel_core_preserve"][0, 1] == 0
    assert out["river_channel_core_preservation_receipt"]["channel_core_zone_pixels"] == 0



def test_inadmissible_channel_surface_is_more_strongly_damped():
    auth = np.full((1, 2), np.nan, dtype=np.float32)
    river_ok = np.ones((1, 2), dtype=bool)
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            sdb_ok=np.zeros((1, 2), dtype=bool),
            river_ok=river_ok,
            background_surface=np.zeros((1, 2), dtype=np.float32),
            river_channel_surface=np.array([[-2.0, -2.0]], dtype=np.float32),
            river_channel_surface_confidence=np.full((1, 2), 0.9, dtype=np.float32),
            river_channel_surface_source_class=np.ones((1, 2), dtype=np.uint8),
            river_channel_surface_support_count=np.ones((1, 2), dtype=np.uint8),
            river_channel_surface_prediction_support_confidence=np.array([[0.7, 0.7]], dtype=np.float32),
            river_channel_surface_measured_anchor_fraction=np.array([[0.0, 0.0]], dtype=np.float32),
            river_channel_surface_structure_only_fraction=np.array([[0.85, 0.85]], dtype=np.float32),
            river_channel_surface_low_support_caution=np.array([[0, 1]], dtype=np.uint8),
            river_channel_surface_prediction_admissibility=np.array([[1, 0]], dtype=np.uint8),
            river_gw=np.ones((1, 2), dtype=np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=3.0),
    )
    assert out["river_primary_surface_confidence"][0, 1] < out["river_primary_surface_confidence"][0, 0]
    assert out["river_primary_surface_terrain_response"][0, 1] < out["river_primary_surface_terrain_response"][0, 0]
    assert out["river_primary_surface_terrain_response"][0, 1] <= 0.26
    assert out["river_primary_surface_cautious_structure"][0, 1] == 1
def test_channel_surface_channel_core_preserve_zone_reports_prepost_delta():
    auth = np.full((1, 2), np.nan, dtype=np.float32)
    river_ok = np.ones((1, 2), dtype=bool)
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            sdb_ok=np.zeros((1, 2), dtype=bool),
            river_ok=river_ok,
            background_surface=np.zeros((1, 2), dtype=np.float32),
            river_channel_surface=np.array([[-2.0, -1.0]], dtype=np.float32),
            river_channel_surface_confidence=np.full((1, 2), 0.9, dtype=np.float32),
            river_channel_surface_source_class=np.ones((1, 2), dtype=np.uint8),
            river_channel_surface_support_count=np.ones((1, 2), dtype=np.uint8),
            river_channel_surface_prediction_support_confidence=np.array([[0.7, 0.2]], dtype=np.float32),
            river_channel_surface_measured_anchor_fraction=np.array([[0.1, 0.1]], dtype=np.float32),
            river_channel_surface_structure_only_fraction=np.array([[0.8, 0.8]], dtype=np.float32),
            river_channel_surface_prediction_admissibility=np.array([[1, 1]], dtype=np.uint8),
            river_channel_surface_low_support_caution=np.array([[0, 1]], dtype=np.uint8),
            river_bank_influence=np.array([[1.0, 0.0]], dtype=np.float32),
            river_gw=np.ones((1, 2), dtype=np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=3.0),
    )
    assert out["river_primary_surface_channel_core_preserve"][0, 0] == 1
    assert out["river_channel_core_preservation_zone"][0, 0] == 1
    assert np.isfinite(out["river_channel_core_prepost_delta"][0, 0])
    assert out["river_channel_core_bank_pull_risk"][0, 0] > 0.0
    assert out["river_channel_core_preservation_receipt"]["channel_core_zone_pixels"] == 1


def test_channel_surface_authoritative_lock_survives_to_support_and_provenance():
    auth = np.full((2, 2), np.nan, dtype=np.float32)
    river_ok = np.ones((2, 2), dtype=bool)
    channel_surface = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    lock_mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            sdb_ok=np.zeros((2, 2), dtype=bool),
            river_ok=river_ok,
            background_surface=np.zeros((2, 2), dtype=np.float32),
            river_channel_surface=channel_surface,
            river_channel_surface_confidence=np.ones((2, 2), dtype=np.float32),
            river_channel_surface_source_class=np.ones((2, 2), dtype=np.uint8),
            river_channel_surface_support_count=np.ones((2, 2), dtype=np.uint8),
            river_channel_surface_authoritative_lock_applied=lock_mask,
            river_gw=np.ones((2, 2), dtype=np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=3.0),
    )
    assert out["support"][0, 0] == int(SupportClass.AUTHORITATIVE_LOCKED)
    assert out["provenance"][0, 0] == int(ProvenanceClass.AUTHORITATIVE_LOCKED)
    assert out["conditioned"][0, 0] == channel_surface[0, 0]
    assert out["river_authoritative_locked"][0, 0] == 1


def test_interpolator_applies_channel_core_preservation_to_pull_final_toward_primary_surface():
    auth = np.array([[np.nan]], dtype=np.float32)
    river_ok = np.array([[True]], dtype=bool)
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            candidate=np.array([[10.0]], dtype=np.float32),
            background_surface=np.array([[10.0]], dtype=np.float32),
            sdb_ok=np.zeros((1, 1), dtype=bool),
            river_ok=river_ok,
            river_channel_surface=np.array([[-3.0]], dtype=np.float32),
            river_channel_surface_confidence=np.array([[0.9]], dtype=np.float32),
            river_channel_surface_source_class=np.ones((1, 1), dtype=np.uint8),
            river_channel_surface_support_count=np.ones((1, 1), dtype=np.uint8),
            river_channel_surface_prediction_support_confidence=np.array([[0.8]], dtype=np.float32),
            river_channel_surface_measured_anchor_fraction=np.array([[0.0]], dtype=np.float32),
            river_channel_surface_structure_only_fraction=np.array([[0.9]], dtype=np.float32),
            river_channel_surface_prediction_admissibility=np.array([[1]], dtype=np.uint8),
            river_channel_surface_low_support_caution=np.array([[0]], dtype=np.uint8),
            river_bank_influence=np.array([[1.0]], dtype=np.float32),
            river_gw=np.ones((1, 1), dtype=np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=3.0),
    )
    pre = float(out["river_primary_surface"][0, 0])
    final = float(out["conditioned"][0, 0])
    assert pre == pytest.approx(-3.0)
    assert final < 1.0
    assert abs(final - pre) < 3.1
    assert out["river_channel_core_preservation_receipt"]["preservation_applied_pixels"] == 1
    assert out["support_note"].find("channel_core_preservation_applied") >= 0


def test_very_inadmissible_channel_surface_is_even_more_strongly_damped():
    auth = np.full((1, 1), np.nan, dtype=np.float32)
    river_ok = np.ones((1, 1), dtype=bool)
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            sdb_ok=np.zeros((1, 1), dtype=bool),
            river_ok=river_ok,
            background_surface=np.zeros((1, 1), dtype=np.float32),
            river_channel_surface=np.array([[-2.0]], dtype=np.float32),
            river_channel_surface_confidence=np.full((1, 1), 0.9, dtype=np.float32),
            river_channel_surface_source_class=np.ones((1, 1), dtype=np.uint8),
            river_channel_surface_support_count=np.ones((1, 1), dtype=np.uint8),
            river_channel_surface_prediction_support_confidence=np.array([[0.20]], dtype=np.float32),
            river_channel_surface_measured_anchor_fraction=np.array([[0.0]], dtype=np.float32),
            river_channel_surface_structure_only_fraction=np.array([[0.90]], dtype=np.float32),
            river_channel_surface_low_support_caution=np.array([[1]], dtype=np.uint8),
            river_channel_surface_prediction_admissibility=np.array([[0]], dtype=np.uint8),
            river_gw=np.ones((1, 1), dtype=np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=3.0),
    )
    assert out["river_primary_surface_terrain_response"][0, 0] <= 0.14
    assert out["river_primary_surface_confidence"][0, 0] <= 0.22
    assert out["river_primary_surface_cautious_structure"][0, 0] == 1


def test_interpolator_accepts_primary_river_guidance_surface_alias():
    auth = np.full((3, 3), np.nan, dtype=np.float32)
    river_ok = np.zeros((3, 3), dtype=bool)
    river_ok[1, 1] = True
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            background_surface=np.full((3, 3), 9.0, dtype=np.float32),
            primary_river_guidance_surface=np.array([[np.nan, np.nan, np.nan],[np.nan, 5.0, np.nan],[np.nan, np.nan, np.nan]], dtype=np.float32),
            river_ok=river_ok,
            sdb_ok=np.zeros((3, 3), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert np.isfinite(out["guidance_surface"][1, 1])
    assert out["river_primary_guidance_summary"]["primary_input_surface_source"] == "primary_river_guidance_surface"




def test_canonical_direct_primary_surface_uses_one_gap_take_domain_even_with_background():
    auth = np.array([[1.0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    background = np.array([[1.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], dtype=np.float32)
    river_ok = np.array([[False, True, True], [False, True, False], [False, False, False]], dtype=bool)
    river_depth = np.full((3, 3), np.nan, dtype=np.float32)
    river_depth[0, 1] = 5.0
    river_depth[0, 2] = 6.0
    river_depth[1, 1] = 7.0

    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            background_surface=background,
            primary_river_guidance_surface=river_depth,
            river_ok=river_ok,
            sdb_ok=np.zeros((3, 3), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_contract_mode="canonical_v322",
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )

    take_domain = out["river_guidance_take_domain"].astype(bool)
    eligible = out["eligible"].astype(bool)
    baseline_exact = out["baseline_exact_domain"].astype(bool)

    assert int(np.count_nonzero(take_domain)) == 3
    assert int(np.count_nonzero(eligible)) == 3
    assert not baseline_exact[0, 1]
    assert not baseline_exact[0, 2]
    assert not baseline_exact[1, 1]
    assert out["conditioned"][0, 1] != pytest.approx(background[0, 1])
    assert out["conditioned"][0, 2] != pytest.approx(background[0, 2])
    assert out["conditioned"][1, 1] != pytest.approx(background[1, 1])
    assert out["guidance_surface"][0, 1] == pytest.approx(5.0)
    assert out["guidance_surface"][0, 2] == pytest.approx(6.0)
    assert out["guidance_surface"][1, 1] == pytest.approx(7.0)
    assert int(out["support"][0, 1]) in {int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.SCAFFOLD_INFERRED)}
    assert int(out["support"][0, 2]) in {int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.SCAFFOLD_INFERRED)}
    assert int(out["support"][1, 1]) in {int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.SCAFFOLD_INFERRED)}


def test_canonical_direct_primary_surface_builder_ignores_legacy_channel_surface_inputs():
    auth = np.full((3, 3), np.nan, dtype=np.float32)
    river_ok = np.array([[False, True, False], [False, True, False], [False, True, False]], dtype=bool)
    river_depth = np.full((3, 3), np.nan, dtype=np.float32)
    river_depth[:, 1] = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    conflicting_channel_surface = np.full((3, 3), np.nan, dtype=np.float32)
    conflicting_channel_surface[:, 1] = np.array([50.0, 60.0, 70.0], dtype=np.float32)

    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            background_surface=np.full((3, 3), 99.0, dtype=np.float32),
            primary_river_guidance_surface=river_depth,
            river_channel_surface=conflicting_channel_surface,
            river_channel_surface_confidence=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_ok=river_ok,
            sdb_ok=np.zeros((3, 3), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_contract_mode="canonical_v322",
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )

    take_domain = out["river_guidance_take_domain"].astype(bool)
    assert int(np.count_nonzero(take_domain)) == 3
    assert out["river_primary_guidance_summary"]["primary_builder_mode"] == "direct_primary_surface_alias"
    assert out["river_primary_guidance_summary"]["primary_builder_inputs_simplified"] is True
    assert np.allclose(out["guidance_surface"][take_domain], river_depth[take_domain], equal_nan=False)


def test_canonical_direct_primary_surface_bypasses_legacy_structured_overrides():
    auth = np.full((3, 3), np.nan, dtype=np.float32)
    river_ok = np.array([[False, True, False], [False, True, False], [False, True, False]], dtype=bool)
    river_depth = np.full((3, 3), np.nan, dtype=np.float32)
    river_depth[:, 1] = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    conflicting_longitudinal = np.full((3, 3), np.nan, dtype=np.float32)
    conflicting_longitudinal[:, 1] = np.array([50.0, 60.0, 70.0], dtype=np.float32)
    conflicting_centerline = np.full((3, 3), np.nan, dtype=np.float32)
    conflicting_centerline[:, 1] = np.array([500.0, 600.0, 700.0], dtype=np.float32)
    conflicting_xs = np.full((3, 3), np.nan, dtype=np.float32)
    conflicting_xs[:, 1] = np.array([5000.0, 6000.0, 7000.0], dtype=np.float32)

    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            background_surface=np.full((3, 3), 99.0, dtype=np.float32),
            primary_river_guidance_surface=river_depth,
            river_ok=river_ok,
            sdb_ok=np.zeros((3, 3), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_longitudinal_profile_elevation=conflicting_longitudinal,
            river_longitudinal_profile_influence=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_centerline_elevation=conflicting_centerline,
            river_centerline_influence=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_xs_support_elevation=conflicting_xs,
            river_xs_support_weight=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_contract_mode="canonical_v322",
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )

    take_domain = out["river_guidance_take_domain"].astype(bool)
    assert int(np.count_nonzero(take_domain)) == 3
    assert out["river_primary_guidance_summary"]["primary_builder_mode"] == "direct_primary_surface_alias"
    assert out["river_primary_guidance_summary"]["legacy_structured_take_bypassed"] is True
    assert out["river_primary_guidance_summary"]["canonical_direct_primary_mode"] is True
    assert np.allclose(out["guidance_surface"][take_domain], river_depth[take_domain], equal_nan=False)


def test_interpolator_marks_legacy_river_depth_guidance_alias_as_deprecated_under_canonical_contract():
    auth = np.full((3, 3), np.nan, dtype=np.float32)
    river_ok = np.zeros((3, 3), dtype=bool)
    river_ok[1, 1] = True
    river_depth = np.full((3, 3), np.nan, dtype=np.float32)
    river_depth[1, 1] = 5.0
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            background_surface=np.full((3, 3), 9.0, dtype=np.float32),
            river_depth_guidance=river_depth,
            river_ok=river_ok,
            sdb_ok=np.zeros((3, 3), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_contract_mode="canonical_v322",
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    summary = out["river_primary_guidance_summary"]
    assert summary["primary_input_surface_source"] == "rejected_legacy_river_depth_guidance_alias"
    assert summary["primary_input_contract_ok"] is False
    assert summary["deprecated_primary_guidance_alias_rejected"] is True
    assert "deprecated_primary_guidance_alias_rejected" in summary["degraded_mode_reasons"]


def test_canonical_mode_does_not_build_primary_river_surface_from_legacy_backbone_fallbacks():
    auth = np.full((5, 5), np.nan, dtype=np.float32)
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[:, 1:4] = True
    centerline_stationing = np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 5))
    longitudinal = np.full((5, 5), np.nan, dtype=np.float32)
    longitudinal[:, 2] = 4.0
    longitudinal_influence = np.zeros((5, 5), dtype=np.float32)
    longitudinal_influence[:, 1:4] = 0.9
    centerline_influence = np.zeros((5, 5), dtype=np.float32)
    centerline_influence[:, 1:4] = 1.0
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            river_ok=river_ok,
            sdb_ok=np.zeros((5, 5), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_centerline_stationing=centerline_stationing,
            river_longitudinal_profile_elevation=longitudinal,
            river_longitudinal_profile_influence=longitudinal_influence,
            river_centerline_influence=centerline_influence,
            river_contract_mode="canonical_v322",
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert not np.isfinite(out["river_primary_surface"]).any()
    summary = out["river_primary_guidance_summary"]
    assert summary["primary_builder_mode"] == "no_primary_surface"
    assert summary["degraded_mode_active"] is True
    assert "missing_channel_surface_primary_path" in summary["degraded_mode_reasons"]


def test_canonical_mode_rejects_native_river_guide_points_inside_interpolator(tmp_path):
    template = tmp_path / "template.tif"
    template.write_text("template", encoding="utf-8")
    auth = np.full((3, 3), np.nan, dtype=np.float32)
    river_ok = np.zeros((3, 3), dtype=bool)
    river_ok[1, 1] = True
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            background_surface=np.full((3, 3), 9.0, dtype=np.float32),
            river_guide_points_path="fake.gpkg",
            guidance_template_raster=str(template),
            river_ok=river_ok,
            sdb_ok=np.zeros((3, 3), dtype=bool),
            river_gw=np.where(river_ok, 1.0, 0.0).astype(np.float32),
            river_contract_mode="canonical_v322",
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    summary = out["river_primary_guidance_summary"]
    assert summary["primary_input_surface_source"] == "rejected_native_river_guide_points_path"
    assert summary["deprecated_native_river_guide_points_rejected"] is True
    assert "deprecated_native_river_guide_points_rejected" in summary["degraded_mode_reasons"]





def test_bank_monotone_qc_enforces_nonincreasing_profile_downstream():
    vals = np.array([10.0, 9.0, 9.5, 7.0], dtype=np.float32)
    out = _weighted_pava_nonincreasing(vals, np.ones_like(vals, dtype=np.float32))
    assert np.all(np.diff(out) <= 1.0e-6)
    assert out[2] <= out[1] + 1.0e-6


def test_canonical_direct_primary_surface_applies_bank_shaping_once_near_edges():
    auth = np.zeros((5, 5), dtype=np.float32)
    auth[:, 1:4] = np.nan
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[:, 1:4] = True
    river_gw = np.where(river_ok, 1.0, 0.0).astype(np.float32)
    primary = np.full((5, 5), np.nan, dtype=np.float32)
    primary[:, 1:4] = 4.0
    bank_elev = np.full((5, 5), np.nan, dtype=np.float32)
    bank_elev[:, 1] = 8.0
    bank_elev[:, 3] = 8.0
    bank_infl = np.zeros((5, 5), dtype=np.float32)
    bank_infl[:, 1] = 1.0
    bank_infl[:, 3] = 1.0
    bank_infl[:, 2] = 0.1
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            river_ok=river_ok,
            sdb_ok=np.zeros((5, 5), dtype=bool),
            river_gw=river_gw,
            primary_river_guidance_surface=primary,
            river_bank_elevation=bank_elev,
            river_bank_influence=bank_infl,
            river_contract_mode='canonical_v322',
            background_surface=np.where(river_ok, 3.5, np.nan).astype(np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=5.0),
    )
    surf = out['river_primary_surface']
    assert surf[2, 1] > 4.0
    assert surf[2, 1] < 8.0
    assert surf[2, 2] == pytest.approx(4.0, abs=0.05)


def test_canonical_direct_primary_uses_only_explicit_bank_inputs():
    auth = np.zeros((5, 5), dtype=np.float32)
    auth[:, 1:4] = np.nan
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[:, 1:4] = True
    river_gw = np.where(river_ok, 1.0, 0.0).astype(np.float32)
    primary = np.full((5, 5), np.nan, dtype=np.float32)
    primary[:, 1:4] = 4.0
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            river_ok=river_ok,
            sdb_ok=np.zeros((5, 5), dtype=bool),
            river_gw=river_gw,
            primary_river_guidance_surface=primary,
            river_contract_mode='canonical_v322',
            background_surface=np.where(river_ok, 3.5, np.nan).astype(np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=5.0),
    )
    surf = out['river_primary_surface']
    assert np.allclose(surf[:, 1:4], 4.0, atol=1.0e-6, equal_nan=False)


def test_canonical_direct_primary_uses_finite_primary_gap_domain_even_if_river_ok_is_empty():
    auth = np.zeros((3, 3), dtype=np.float32)
    auth[1, 1] = np.nan
    river_ok = np.zeros((3, 3), dtype=bool)
    primary = np.full((3, 3), np.nan, dtype=np.float32)
    primary[1, 1] = 4.0
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            river_ok=river_ok,
            sdb_ok=np.zeros((3, 3), dtype=bool),
            river_gw=np.zeros((3, 3), dtype=np.float32),
            primary_river_guidance_surface=primary,
            river_contract_mode='canonical_v322',
            background_surface=np.full((3, 3), 3.5, dtype=np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=5.0),
    )
    assert bool(out['river_guidance_take_domain'][1, 1])
    assert out['guidance_surface'][1, 1] == pytest.approx(4.0)
