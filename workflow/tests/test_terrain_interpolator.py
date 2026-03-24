import numpy as np
import pytest

from support_classes import SupportClass
from provenance_schema import ProvenanceClass
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
            river_gw=np.where(river_ok, 0.8, 0.0).astype(np.float32),
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
            candidate=None,
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
