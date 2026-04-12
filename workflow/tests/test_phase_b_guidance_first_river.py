from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from deterministic_terrain_stage import run_deterministic_terrain_stage
from guidance_assembly_stage import assemble_guidance_inputs
from river_guidance import build_river_guidance_manifest
from terrain_interpolator import _nearest_surface_within_domain


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def _write_raster(path: Path, arr: np.ndarray, *, nodata: float = -9999.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=str(arr.dtype),
        crs="EPSG:4326",
        transform=from_origin(0, arr.shape[0], 1, 1),
        nodata=nodata,
    ) as ds:
        ds.write(arr, 1)
    return path


def test_river_guidance_manifest_includes_centerline_stationing(tmp_path: Path):
    river_dir = tmp_path / "river"
    for name in [
        "river_guidance_weight.tif",
        "river_admissibility.tif",
        "river_corridor_mask.tif",
        "river_guide_points.gpkg",
        "river_centerline_stationing_m.tif",
    ]:
        _touch(river_dir / name)
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report={"river": {"outputs": {}}})
    assert "centerline_stationing" in manifest["final_route_contract"]["allowed_structural_artifacts"]
    assert manifest["artifact_roles"]["centerline_stationing"] == "centerline_stationing_coordinate"


def test_guidance_assembly_loads_centerline_stationing(tmp_path: Path):
    auth = np.array([[1.0, -9999.0, 3.0], [1.5, -9999.0, 3.5], [2.0, -9999.0, 4.0]], dtype=np.float32)
    template = _write_raster(tmp_path / "combined" / "auth.tif", auth)
    _touch(tmp_path / "river" / "river_guide_points.gpkg")
    _write_raster(tmp_path / "river" / "river_centerline_stationing_m.tif", np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32))
    _write_raster(tmp_path / "river" / "river_centerline_elevation.tif", np.array([[5.0, 5.5, 6.0], [5.0, 5.5, 6.0], [5.0, 5.5, 6.0]], dtype=np.float32))
    _write_raster(tmp_path / "river" / "river_centerline_influence.tif", np.ones((3, 3), dtype=np.float32))
    _write_raster(tmp_path / "river" / "river_xs_support_elevation.tif", np.array([[4.0, 4.5, 5.0], [4.0, 4.5, 5.0], [4.0, 4.5, 5.0]], dtype=np.float32))
    _write_raster(tmp_path / "river" / "river_xs_support_weight.tif", np.ones((3, 3), dtype=np.float32))
    _write_raster(tmp_path / "river" / "river_bank_influence.tif", np.ones((3, 3), dtype=np.float32))
    _write_raster(tmp_path / "river" / "river_bank_elevation_xs.tif", np.array([[6.0, 6.5, 7.0], [6.0, 6.5, 7.0], [6.0, 6.5, 7.0]], dtype=np.float32))
    _write_raster(tmp_path / "river" / "river_bank_continuity_weight.tif", np.ones((3, 3), dtype=np.float32))
    _write_raster(tmp_path / "river" / "river_bank_graph_confidence.tif", np.ones((3, 3), dtype=np.float32))
    _write_raster(tmp_path / "river" / "river_bank_confluence_damping.tif", np.ones((3, 3), dtype=np.float32))
    _write_raster(tmp_path / "river" / "river_bank_estuary_side_decay.tif", np.ones((3, 3), dtype=np.float32))
    _write_raster(tmp_path / "river" / "river_admissibility.tif", np.ones((3, 3), dtype=np.uint8), nodata=0)
    _write_raster(tmp_path / "river" / "river_corridor_mask.tif", np.ones((3, 3), dtype=np.uint8), nodata=0)

    paths = SimpleNamespace(
        template_path=template,
        auth_src=template,
        guidance_receipt_path=tmp_path / "combined" / "guidance_receipt.json",
    )
    cfg = SimpleNamespace(out_dir=tmp_path, tile_bbox=None)
    report = {"river": {"outputs": {
        "guide_points": str(tmp_path / "river" / "river_guide_points.gpkg"),
        "centerline_stationing": str(tmp_path / "river" / "river_centerline_stationing_m.tif"),
        "centerline_elevation": str(tmp_path / "river" / "river_centerline_elevation.tif"),
        "centerline_influence": str(tmp_path / "river" / "river_centerline_influence.tif"),
        "xs_support_elevation": str(tmp_path / "river" / "river_xs_support_elevation.tif"),
        "xs_support_weight": str(tmp_path / "river" / "river_xs_support_weight.tif"),
        "bank_influence": str(tmp_path / "river" / "river_bank_influence.tif"),
        "bank_elevation_xs": str(tmp_path / "river" / "river_bank_elevation_xs.tif"),
        "bank_continuity_weight": str(tmp_path / "river" / "river_bank_continuity_weight.tif"),
        "bank_graph_confidence": str(tmp_path / "river" / "river_bank_graph_confidence.tif"),
        "bank_confluence_damping": str(tmp_path / "river" / "river_bank_confluence_damping.tif"),
        "bank_estuary_side_decay": str(tmp_path / "river" / "river_bank_estuary_side_decay.tif"),
        "admissibility": str(tmp_path / "river" / "river_admissibility.tif"),
        "corridor_mask": str(tmp_path / "river" / "river_corridor_mask.tif"),
    }}}
    assembled = assemble_guidance_inputs(cfg=cfg, paths=paths, candidate_path=None, provenance_path=None, report=report)
    assert assembled.arrays["river_centerline_stationing"] is not None
    assert float(assembled.arrays["river_centerline_stationing"][1, 1]) == 1.0


def test_deterministic_terrain_stage_passes_centerline_stationing(monkeypatch, tmp_path: Path):
    captured = {}

    def _fake_support_weighted_condition_arrays(**kwargs):
        captured.update(kwargs)
        auth = kwargs["auth"]
        return {
            "locked": np.isfinite(auth),
            "gap": ~np.isfinite(auth),
            "eligible": np.zeros_like(auth, dtype=bool),
            "conditioned": np.where(np.isfinite(auth), auth, np.float32(0.0)).astype(np.float32),
            "support": np.zeros_like(auth, dtype=np.uint8),
            "provenance": np.zeros_like(auth, dtype=np.uint8),
            "support_note": "ok",
        }

    monkeypatch.setattr("deterministic_terrain_stage.support_weighted_condition_arrays", _fake_support_weighted_condition_arrays)

    guidance = SimpleNamespace(
        arrays={
            "sdb_adm": np.zeros((2, 2), dtype=np.uint8),
            "river_adm": np.zeros((2, 2), dtype=np.uint8),
            "sdb_gw": None,
            "sdb_ti": None,
            "river_gw": None,
            "river_ti": None,
            "river_support": None,
            "river_support_depth": None,
            "river_estuary_transition": None,
            "river_corridor": None,
            "river_bank_influence": None,
            "river_bank_elevation_xs": None,
            "river_bank_pair_weight": None,
            "river_bank_continuity_weight": None,
            "river_bank_graph_confidence": None,
            "river_bank_confluence_damping": None,
            "river_bank_estuary_side_decay": None,
            "river_centerline_elevation": None,
            "river_centerline_influence": None,
            "river_centerline_stationing": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            "river_longitudinal_profile_elevation": None,
            "river_longitudinal_profile_uncertainty": None,
            "river_longitudinal_profile_influence": None,
            "river_xs_support_elevation": None,
            "river_xs_support_weight": None,
        },
        auth=np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32),
        sdb_guide_points_path=None,
        river_guide_points_path=None,
        support_params={
            "pixel_size_m": 1.0,
            "support_decay_m": 10.0,
            "support_density_radius_m": 10.0,
            "coastal_sdb_support_transition_m": 10.0,
            "river_anchor_density_radius_m": 10.0,
            "river_scaffold_transition_m": 10.0,
        },
        source_candidate={},
        candidate_prov=np.zeros((2, 2), dtype=np.uint8),
        baseline_background=None,
        baseline_cudem_path=None,
    )
    run_deterministic_terrain_stage(guidance=guidance, template_path=str(tmp_path / "template.tif"))
    assert "river_centerline_stationing" in captured
    assert captured["river_centerline_stationing"] is not None



def test_deterministic_terrain_stage_allows_direct_primary_surface_handoff_without_legacy_primary_surface(monkeypatch, tmp_path: Path):
    def _fake_support_weighted_condition_arrays(**kwargs):
        auth = kwargs["auth"]
        conditioned = np.where(np.isfinite(auth), auth, np.float32(0.0)).astype(np.float32)
        conditioned[0, 1] = np.float32(5.0)
        conditioned[1, 1] = np.float32(6.0)
        return {
            "locked": np.isfinite(auth),
            "gap": ~np.isfinite(auth),
            "eligible": np.zeros_like(auth, dtype=bool),
            "conditioned": conditioned,
            "support": np.zeros_like(auth, dtype=np.uint8),
            "provenance": np.zeros_like(auth, dtype=np.uint8),
            "support_note": "ok",
            "memory_diagnostics": [{"river_guidance_finite": 4}],
            "river_primary_guidance_summary": {"primary_surface_finite_pixels": 0},
        }

    monkeypatch.setattr("deterministic_terrain_stage.support_weighted_condition_arrays", _fake_support_weighted_condition_arrays)

    guidance = SimpleNamespace(
        arrays={
            "sdb_adm": np.zeros((2, 2), dtype=np.uint8),
            "river_adm": np.zeros((2, 2), dtype=np.uint8),
            "sdb_gw": None,
            "sdb_ti": None,
            "river_gw": None,
            "river_ti": None,
            "river_support": None,
            "river_support_depth": None,
            "river_estuary_transition": None,
            "river_corridor": np.ones((2, 2), dtype=np.uint8),
            "river_bank_influence": None,
            "river_bank_elevation_xs": None,
            "river_bank_pair_weight": None,
            "river_bank_continuity_weight": None,
            "river_bank_graph_confidence": None,
            "river_bank_confluence_damping": None,
            "river_bank_estuary_side_decay": None,
            "river_centerline_elevation": None,
            "river_centerline_influence": None,
            "river_centerline_stationing": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            "river_channel_surface": None,
            "primary_river_guidance_surface": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "river_longitudinal_profile_elevation": None,
            "river_longitudinal_profile_uncertainty": None,
            "river_longitudinal_profile_influence": None,
            "river_xs_support_elevation": None,
            "river_xs_support_weight": None,
        },
        auth=np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32),
        sdb_guide_points_path=None,
        river_guide_points_path=None,
        support_params={
            "pixel_size_m": 1.0,
            "support_decay_m": 10.0,
            "support_density_radius_m": 10.0,
            "coastal_sdb_support_transition_m": 10.0,
            "river_anchor_density_radius_m": 10.0,
            "river_scaffold_transition_m": 10.0,
            "river_contract_mode": "canonical_v322",
        },
        source_candidate={"river_primary_surface": "x"},
        candidate_prov=np.zeros((2, 2), dtype=np.uint8),
        baseline_background=np.zeros((2, 2), dtype=np.float32),
        baseline_cudem_path=None,
    )
    receipt_path = tmp_path / "terrain_receipt.json"
    result = run_deterministic_terrain_stage(guidance=guidance, template_path=str(tmp_path / "template.tif"), receipt_path=str(receipt_path))
    assert result is not None
    uptake = (tmp_path / "river_primary_surface_uptake_receipt.json").read_text(encoding="utf-8")
    assert '"status": "uptake_ok"' in uptake
    effect = (tmp_path / "river_primary_surface_conditioning_effect_receipt.json").read_text(encoding="utf-8")
    assert '"status": "effect_detected"' in effect
    assert '"changed_vs_background_pixels": 4' in effect
    assert '"changed_vs_background_unlocked_pixels": 2' in effect


def test_anisotropic_river_nearest_prefers_along_channel_stationing():
    domain = np.zeros((7, 7), dtype=bool)
    domain[1:6, 1:6] = True
    centerline = np.zeros((7, 7), dtype=bool)
    centerline[3, 1:6] = True
    stationing = np.full((7, 7), np.nan, dtype=np.float32)
    stationing[3, 1:6] = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    valid = np.zeros((7, 7), dtype=bool)
    values = np.full((7, 7), np.nan, dtype=np.float32)
    valid[2, 2] = True   # upper bank-ish, station 10
    values[2, 2] = 10.0
    valid[3, 5] = True   # along-channel farther, station 40
    values[3, 5] = 40.0

    out = _nearest_surface_within_domain(
        valid,
        values,
        domain,
        centerline_mask=centerline,
        stationing_raster=stationing,
        along_scale_m=1000.0,
        cross_scale_m=1.0,
        pixel_size_m=1.0,
    )
    # Query point near centerline station 30 should prefer the along-channel source at station 40,
    # even though the cross-channel source is closer in Euclidean space.
    assert float(out[3, 4]) == 40.0


def test_guidance_assembly_fails_loudly_when_critical_river_artifacts_missing(tmp_path: Path):
    auth = np.full((3, 3), -9999.0, dtype=np.float32)
    template = _write_raster(tmp_path / "combined" / "auth.tif", auth)
    _write_raster(tmp_path / "river" / "river_admissibility.tif", np.ones((3, 3), dtype=np.uint8), nodata=0)
    _write_raster(tmp_path / "river" / "river_corridor_mask.tif", np.ones((3, 3), dtype=np.uint8), nodata=0)

    paths = SimpleNamespace(
        template_path=template,
        auth_src=template,
        guidance_receipt_path=tmp_path / "combined" / "guidance_receipt.json",
    )
    cfg = SimpleNamespace(out_dir=tmp_path, tile_bbox=None)
    report = {"river": {"outputs": {
        "admissibility": str(tmp_path / "river" / "river_admissibility.tif"),
        "corridor_mask": str(tmp_path / "river" / "river_corridor_mask.tif"),
    }}}
    try:
        assemble_guidance_inputs(cfg=cfg, paths=paths, candidate_path=None, provenance_path=None, report=report)
    except RuntimeError as exc:
        msg = str(exc)
        assert "missing_required_river_structural_guidance_artifacts_for_final_route" in msg
        assert "centerline_stationing" in msg
        assert "guide_points" in msg
    else:
        raise AssertionError("Expected assemble_guidance_inputs to fail on missing critical river artifacts")


def test_guidance_assembly_fails_loudly_when_sdb_structural_semantics_are_invalid(tmp_path: Path):
    template = _write_raster(tmp_path / "combined" / "auth.tif", np.full((3, 3), -9999.0, dtype=np.float32))
    sdb_dir = tmp_path / "sdb"
    sdb_dir.mkdir(parents=True, exist_ok=True)
    (sdb_dir / "artifacts_sdb.json").write_text('{"depth_raster":"pred_depth.tif","admissibility_raster":"pred_depth_admissibility.tif","guidance_weight_raster":"pred_depth_guidance_weight.tif","guide_points":"pred_depth_guide_points.gpkg"}', encoding="utf-8")
    _write_raster(sdb_dir / "pred_depth.tif", np.full((3, 3), -5.0, dtype=np.float32))
    _write_raster(sdb_dir / "pred_depth_admissibility.tif", np.ones((3, 3), dtype=np.uint8), nodata=0)
    _write_raster(sdb_dir / "pred_depth_guidance_weight.tif", np.full((3, 3), 1.5, dtype=np.float32))
    _touch(sdb_dir / "pred_depth_guide_points.gpkg")

    paths = SimpleNamespace(
        template_path=template,
        auth_src=template,
        guidance_receipt_path=tmp_path / "combined" / "guidance_receipt.json",
    )
    cfg = SimpleNamespace(out_dir=tmp_path, tile_bbox=None)
    try:
        assemble_guidance_inputs(cfg=cfg, paths=paths, candidate_path=None, provenance_path=None, report={})
    except RuntimeError as exc:
        msg = str(exc)
        assert "missing_required_sdb_structural_guidance_artifacts_for_final_route" in msg
        assert "sdb_guidance_weight_out_of_0_1_range" in msg
    else:
        raise AssertionError("Expected assemble_guidance_inputs to fail on invalid SDB structural semantics")


def test_deterministic_terrain_stage_writes_primary_surface_uptake_receipt(monkeypatch, tmp_path: Path):
    def _fake_support_weighted_condition_arrays(**kwargs):
        auth = kwargs["auth"]
        return {
            "locked": np.isfinite(auth),
            "gap": ~np.isfinite(auth),
            "eligible": np.zeros_like(auth, dtype=bool),
            "conditioned": np.where(np.isfinite(auth), auth, np.float32(0.0)).astype(np.float32),
            "support": np.zeros_like(auth, dtype=np.uint8),
            "provenance": np.zeros_like(auth, dtype=np.uint8),
            "support_note": "ok",
            "memory_diagnostics": [{"river_guidance_finite": 4}],
            "river_primary_guidance_summary": {"primary_surface_finite_pixels": 0},
        }

    monkeypatch.setattr("deterministic_terrain_stage.support_weighted_condition_arrays", _fake_support_weighted_condition_arrays)

    receipt_path = tmp_path / "terrain_receipt.json"
    guidance = SimpleNamespace(
        arrays={
            "sdb_adm": np.zeros((2, 2), dtype=np.uint8),
            "river_adm": np.zeros((2, 2), dtype=np.uint8),
            "sdb_gw": None,
            "sdb_ti": None,
            "river_gw": None,
            "river_ti": None,
            "river_support": None,
            "river_support_depth": None,
            "river_estuary_transition": None,
            "river_corridor": np.ones((2, 2), dtype=np.uint8),
            "river_bank_influence": None,
            "river_bank_elevation_xs": None,
            "river_bank_pair_weight": None,
            "river_bank_continuity_weight": None,
            "river_bank_graph_confidence": None,
            "river_bank_confluence_damping": None,
            "river_bank_estuary_side_decay": None,
            "river_centerline_elevation": None,
            "river_centerline_influence": None,
            "river_centerline_stationing": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            "river_channel_surface": None,
            "primary_river_guidance_surface": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "river_longitudinal_profile_elevation": None,
            "river_longitudinal_profile_uncertainty": None,
            "river_longitudinal_profile_influence": None,
            "river_xs_support_elevation": None,
            "river_xs_support_weight": None,
        },
        auth=np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32),
        sdb_guide_points_path=None,
        river_guide_points_path=None,
        support_params={
            "pixel_size_m": 1.0,
            "support_decay_m": 10.0,
            "support_density_radius_m": 10.0,
            "coastal_sdb_support_transition_m": 10.0,
            "river_anchor_density_radius_m": 10.0,
            "river_scaffold_transition_m": 10.0,
        },
        source_candidate={},
        candidate_prov=np.zeros((2, 2), dtype=np.uint8),
        baseline_background=None,
        baseline_cudem_path=None,
    )

    run_deterministic_terrain_stage(guidance=guidance, template_path=str(tmp_path / "template.tif"), receipt_path=str(receipt_path))
    uptake = (tmp_path / "river_primary_surface_uptake_receipt.json").read_text(encoding="utf-8")
    assert '"status": "uptake_ok"' in uptake
