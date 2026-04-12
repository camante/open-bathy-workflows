from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from final_route_inputs_stage import collect_final_route_inputs
from guidance_assembly_stage import assemble_guidance_inputs, _require_river_structural_artifacts, _exclude_guidance_from_authoritative_locked_cells


def _write_raster(path: Path, arr: np.ndarray, *, nodata=-9999.0):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": str(arr.dtype),
        "crs": "EPSG:26919",
        "transform": from_origin(0.0, float(arr.shape[0]), 1.0, 1.0),
        "nodata": nodata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr, 1)


def test_guidance_assembly_preserves_exact_baseline_grid_and_records_alignment(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    auth = tmp_path / "auth.tif"
    baseline = tmp_path / "cudem_baseline_interpolation.tif"
    auth_arr = np.array([[1.0, -9999.0], [2.0, -9999.0]], dtype=np.float32)
    baseline_arr = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
    _write_raster(auth, auth_arr)
    _write_raster(baseline, baseline_arr)
    cfg = SimpleNamespace(out_dir=str(out_dir), authoritative_base=str(auth), tile_bbox=None)
    report = {"authoritative_base_auto": {"baseline_cudem_interpolation": str(baseline)}, "river": {"outputs": {}}}
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=None, report=report)
    guidance = assemble_guidance_inputs(cfg=cfg, paths=paths, candidate_path=None, provenance_path=None, report=report)
    assert np.allclose(guidance.baseline_background, baseline_arr, equal_nan=True)
    payload = json.loads(paths.guidance_receipt_path.read_text(encoding="utf-8"))
    assert payload["structural_alignment"]["baseline_cudem_interpolation"]["mode"] == "direct_read_exact_grid"
    assert payload["structural_alignment"]["baseline_cudem_interpolation"]["resampling"] == "none"


def test_guidance_assembly_excludes_guidance_arrays_inside_authoritative_locked_cells(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    auth = tmp_path / "auth.tif"
    baseline = tmp_path / "cudem_baseline_interpolation.tif"
    auth_arr = np.array([[1.0, -9999.0], [2.0, -9999.0]], dtype=np.float32)
    _write_raster(auth, auth_arr)
    _write_raster(baseline, np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32))

    river_dir = out_dir / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    (river_dir / "river_guide_points.gpkg").write_text("x", encoding="utf-8")
    _write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32))
    _write_raster(river_dir / "river_centerline_elevation.tif", np.array([[5.0, 5.5], [6.0, 6.5]], dtype=np.float32))
    _write_raster(river_dir / "river_centerline_influence.tif", np.ones((2, 2), dtype=np.float32))
    _write_raster(river_dir / "river_xs_support_elevation.tif", np.array([[4.0, 4.5], [5.0, 5.5]], dtype=np.float32))
    _write_raster(river_dir / "river_xs_support_weight.tif", np.ones((2, 2), dtype=np.float32))
    _write_raster(river_dir / "river_bank_influence.tif", np.ones((2, 2), dtype=np.float32))
    _write_raster(river_dir / "river_bank_elevation_xs.tif", np.array([[6.0, 6.5], [7.0, 7.5]], dtype=np.float32))
    _write_raster(river_dir / "river_bank_continuity_weight.tif", np.ones((2, 2), dtype=np.float32))
    _write_raster(river_dir / "river_bank_graph_confidence.tif", np.ones((2, 2), dtype=np.float32))
    _write_raster(river_dir / "river_bank_confluence_damping.tif", np.ones((2, 2), dtype=np.float32))
    _write_raster(river_dir / "river_bank_estuary_side_decay.tif", np.ones((2, 2), dtype=np.float32))
    _write_raster(river_dir / "river_admissibility.tif", np.ones((2, 2), dtype=np.uint8), nodata=0)
    _write_raster(river_dir / "river_corridor_mask.tif", np.ones((2, 2), dtype=np.uint8), nodata=0)

    cfg = SimpleNamespace(out_dir=str(out_dir), authoritative_base=str(auth), tile_bbox=None)
    report = {"authoritative_base_auto": {"baseline_cudem_interpolation": str(baseline)}, "river": {"outputs": {
        "guide_points": str(river_dir / "river_guide_points.gpkg"),
        "centerline_stationing": str(river_dir / "river_centerline_stationing_m.tif"),
        "centerline_elevation": str(river_dir / "river_centerline_elevation.tif"),
        "centerline_influence": str(river_dir / "river_centerline_influence.tif"),
        "xs_support_elevation": str(river_dir / "river_xs_support_elevation.tif"),
        "xs_support_weight": str(river_dir / "river_xs_support_weight.tif"),
        "bank_influence": str(river_dir / "river_bank_influence.tif"),
        "bank_elevation_xs": str(river_dir / "river_bank_elevation_xs.tif"),
        "bank_continuity_weight": str(river_dir / "river_bank_continuity_weight.tif"),
        "bank_graph_confidence": str(river_dir / "river_bank_graph_confidence.tif"),
        "bank_confluence_damping": str(river_dir / "river_bank_confluence_damping.tif"),
        "bank_estuary_side_decay": str(river_dir / "river_bank_estuary_side_decay.tif"),
        "admissibility": str(river_dir / "river_admissibility.tif"),
        "corridor_mask": str(river_dir / "river_corridor_mask.tif"),
    }}}
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=None, report=report)
    guidance = assemble_guidance_inputs(cfg=cfg, paths=paths, candidate_path=None, provenance_path=None, report=report)
    locked = np.isfinite(guidance.auth)
    assert guidance.arrays["river_adm"][locked].sum() == 0
    assert np.all(guidance.arrays["river_centerline_influence"][locked] == 0.0)
    assert np.all(~np.isfinite(guidance.arrays["river_centerline_elevation"][locked]))
    payload = json.loads(paths.guidance_receipt_path.read_text(encoding="utf-8"))
    assert payload["authoritative_locked_guidance_exclusion"]["locked_pixels"] == int(np.count_nonzero(locked))
    assert payload["authoritative_locked_guidance_exclusion"]["arrays"]["river_centerline_elevation"]["policy"] == "nan_on_locked"


def test_guidance_semantic_validation_runs_before_authoritative_lock_masking(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    guide_points = river_dir / "river_guide_points.gpkg"
    guide_points.write_text("x", encoding="utf-8")
    centerline = river_dir / "river_centerline_elevation.tif"
    stationing = river_dir / "river_centerline_stationing_m.tif"
    bank_infl = river_dir / "river_bank_influence.tif"
    corridor = river_dir / "river_corridor_mask.tif"
    adm = river_dir / "river_admissibility.tif"
    _write_raster(centerline, np.array([[5.0, 5.5], [6.0, 6.5]], dtype=np.float32))
    _write_raster(stationing, np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32))
    _write_raster(bank_infl, np.ones((2, 2), dtype=np.float32))
    _write_raster(corridor, np.ones((2, 2), dtype=np.uint8), nodata=0)
    _write_raster(adm, np.ones((2, 2), dtype=np.uint8), nodata=0)

    river_outputs = {
        "guide_points": str(guide_points),
        "centerline_elevation": str(centerline),
        "centerline_stationing": str(stationing),
        "centerline_influence": str(bank_infl),
        "bank_influence": str(bank_infl),
        "bank_elevation_xs": str(centerline),
        "bank_continuity_weight": str(bank_infl),
        "bank_graph_confidence": str(bank_infl),
        "bank_confluence_damping": str(bank_infl),
        "bank_estuary_side_decay": str(bank_infl),
        "admissibility": str(adm),
        "corridor_mask": str(corridor),
    }
    arrays = {
        "river_corridor": np.ones((2, 2), dtype=np.uint8),
        "river_adm": np.ones((2, 2), dtype=np.uint8),
        "river_centerline_stationing": np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32),
        "river_centerline_elevation": np.array([[5.0, 5.5], [6.0, 6.5]], dtype=np.float32),
        "river_bank_influence": np.ones((2, 2), dtype=np.float32),
        "river_centerline_influence": np.ones((2, 2), dtype=np.float32),
        "river_bank_elevation_xs": np.array([[6.0, 6.5], [7.0, 7.5]], dtype=np.float32),
        "river_bank_continuity_weight": np.ones((2, 2), dtype=np.float32),
        "river_bank_graph_confidence": np.ones((2, 2), dtype=np.float32),
        "river_bank_confluence_damping": np.ones((2, 2), dtype=np.float32),
        "river_bank_estuary_side_decay": np.ones((2, 2), dtype=np.float32),
        "river_xs_support_elevation": None,
        "river_xs_support_weight": None,
    }
    # Upstream semantics are valid before lock masking.
    req = _require_river_structural_artifacts(
        river_outputs=river_outputs,
        outputs_base_dir=tmp_path,
        arrays=arrays,
        river_guide_points_path=guide_points,
    )
    assert req["requested"] is True

    locked = np.ones((2, 2), dtype=bool)
    locked_payload = _exclude_guidance_from_authoritative_locked_cells(arrays, locked)
    masked = locked_payload["arrays"]
    assert np.all(~np.isfinite(masked["river_centerline_elevation"]))



def test_guidance_semantics_audit_does_not_flip_authoritative_base():
    from guidance_assembly_stage import _maybe_harmonize_authoritative_semantics

    auth = -np.arange(121, dtype=np.float32).reshape(11, 11) - 1.0
    primary = -auth
    baseline = auth.copy()
    legacy = auth.copy()
    auth_out, legacy_out, baseline_out, receipt = _maybe_harmonize_authoritative_semantics(
        auth=auth,
        legacy_candidate=legacy,
        baseline_background=baseline,
        river_support_depth=None,
        primary_river_guidance_surface=primary,
    )
    assert np.array_equal(auth_out, auth)
    assert np.array_equal(legacy_out, legacy)
    assert np.array_equal(baseline_out, baseline)
    assert receipt["applied"] is False
    assert receipt["reason"] in {"detected_possible_sign_mismatch_preserved_authoritative_reference", "already_aligned"}
