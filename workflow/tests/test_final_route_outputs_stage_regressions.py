from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from final_route_inputs_stage import collect_final_route_inputs
from final_route_outputs_stage import write_final_route_outputs
from legacy_cleanup_stage import build_legacy_cleanup_summary


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


class _Cfg:
    def __init__(self, out_dir: Path, authoritative_base: Path):
        self.out_dir = out_dir
        self.authoritative_base = authoritative_base
        self.output_vdatum = "NAVD88"
        self.output_vdatum_epsg = "5703"


def test_legacy_cleanup_ignores_non_path_containers():
    report = {
        "sdb": {"artifacts": {"depth_raster": ["/tmp/a.tif", "/tmp/b.tif"]}},
        "river": {"outputs": {"depth_terrain": {"path": "/tmp/river_depth.tif"}}},
        "final_dem_route": {"structural_inputs": {"authoritative_base": "/tmp/auth.tif"}},
    }
    summary = build_legacy_cleanup_summary(report)
    assert "dense_sdb_depth_raster" not in summary["deprecated_artifacts_present"]
    assert "dense_river_depth_raster" not in summary["deprecated_artifacts_present"]



def test_write_final_route_outputs_handles_missing_optional_bank_rasters_and_reports_outputs(tmp_path: Path):
    auth_arr = np.array([
        [1.0, 1.5, 2.0],
        [1.1, 1.6, 2.1],
        [1.2, 1.7, 2.2],
    ], dtype=np.float32)
    auth_path = _write_raster(tmp_path / "auth.tif", auth_arr)
    cfg = _Cfg(out_dir=tmp_path, authoritative_base=auth_path)
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=None, report={})
    assert paths is not None

    with rasterio.open(auth_path) as ds:
        profile = ds.profile.copy()

    guidance = SimpleNamespace(
        auth=auth_arr,
        profile=profile,
        nodata=-9999.0,
        sdb_depth_path=None,
        sdb_guide_points_path=None,
        river_guide_points_path=None,
    )
    terrain = SimpleNamespace(
        result={
            "locked": np.ones((3, 3), dtype=bool),
            "gap": np.zeros((3, 3), dtype=bool),
            "eligible": np.zeros((3, 3), dtype=bool),
            "support": np.ones((3, 3), dtype=np.uint8),
            "regime": np.ones((3, 3), dtype=np.uint8),
            "support_distance_m": np.zeros((3, 3), dtype=np.float32),
            "support_density": np.ones((3, 3), dtype=np.float32),
            "anchor_uncertainty": np.zeros((3, 3), dtype=np.float32),
            "guidance_uncertainty": np.zeros((3, 3), dtype=np.float32),
            "conditioned_uncertainty": np.zeros((3, 3), dtype=np.float32),
            "guidance_influence": np.zeros((3, 3), dtype=np.float32),
            "coastal_sdb_confidence": np.zeros((3, 3), dtype=np.float32),
            "river_anchor_distance_m": np.zeros((3, 3), dtype=np.float32),
            "river_anchor_density": np.zeros((3, 3), dtype=np.float32),
            "river_scaffold_confidence": np.zeros((3, 3), dtype=np.float32),
            "river_bank_distance_m": np.zeros((3, 3), dtype=np.float32),
            "river_bank_influence": np.zeros((3, 3), dtype=np.float32),
            "river_bank_elevation": auth_arr + 0.5,
            "river_primary_surface": auth_arr.copy(),
            "river_primary_surface_confidence": np.full((3, 3), 0.9, dtype=np.float32),
            "river_primary_surface_source_class": np.ones((3, 3), dtype=np.uint8),
            "river_primary_surface_support_count": np.ones((3, 3), dtype=np.uint8),
            "river_primary_surface_domain": np.ones((3, 3), dtype=np.uint8),
            "river_primary_surface_contract": {"ok": True, "failures": [], "metrics": {"domain_pixels": 9, "finite_pixels": 9}},
            "river_bank_continuity_weight": None,
            "river_bank_graph_confidence": None,
            "river_bank_confluence_damping": None,
            "river_bank_estuary_side_decay": None,
            "conditioned": auth_arr.copy(),
            "provenance": np.full((3, 3), 10, dtype=np.uint8),
            "authoritative_first_contract": {"ok": True, "locked_changed_count": 0, "background_changed_count": 0, "low_confidence_fill_outside_guidance_count": 0},
            "support_note": "ok",
            "guidance_surface": auth_arr.copy(),
        },
        source_candidate={"stats": {}, "backstop_policy": {}, "provenance_codes": {}},
        candidate_prov=np.zeros((3, 3), dtype=np.uint8),
    )
    report = {
        "authoritative_base_auto": {},
        "sdb": {"artifacts": {}},
        "river": {"outputs": {}},
    }

    conditioned_path, _, _, _, _, _ = write_final_route_outputs(
        cfg=cfg,
        paths=paths,
        guidance=guidance,
        terrain=terrain,
        candidate_path=None,
        provenance_path=None,
        report=report,
    )
    assert conditioned_path.exists()

    continuity = paths.combined_dir / "river_bank_continuity_weight.tif"
    graph = paths.combined_dir / "river_bank_graph_confidence.tif"
    confluence = paths.combined_dir / "river_bank_confluence_damping.tif"
    estuary = paths.combined_dir / "river_bank_estuary_side_decay.tif"
    for path in (continuity, graph, confluence, estuary):
        assert path.exists()

    with rasterio.open(continuity) as ds:
        arr = ds.read(1)
        assert np.allclose(arr, 0.0)
    with rasterio.open(graph) as ds:
        arr = ds.read(1)
        assert np.allclose(arr, 0.0)
    with rasterio.open(confluence) as ds:
        arr = ds.read(1)
        assert np.allclose(arr, 1.0)
    with rasterio.open(estuary) as ds:
        arr = ds.read(1)
        assert np.allclose(arr, 1.0)

    ab_out = report["authoritative_base"]["outputs"]
    outputs_receipt = json.loads(paths.outputs_receipt_path.read_text(encoding='utf-8'))
    final_route_receipt = json.loads(paths.final_route_receipt_path.read_text(encoding='utf-8'))
    assert outputs_receipt['artifact_roles']['conditioned_depth'] == 'primary'
    assert outputs_receipt['artifact_roles']['river_primary_guidance_summary'] == 'diagnostic_only'
    assert outputs_receipt['artifact_roles']['river_primary_surface'] == 'primary'
    assert 'river_active_product_story' in final_route_receipt['primary_runtime_receipts']
    assert final_route_receipt['primary_runtime_receipts']['active_runtime_products'] == ['aligned_authoritative_base', 'conditioned_depth', 'river_primary_surface']
    assert final_route_receipt['artifact_roles']['final_output_layer_contract'] == 'diagnostic_only'
    assert final_route_receipt['artifact_roles']['river_primary_surface'] == 'primary'
    assert ab_out["river_bank_graph_confidence"] == str(graph)
    assert ab_out["river_bank_confluence_damping"] == str(confluence)
    assert ab_out["river_bank_estuary_side_decay"] == str(estuary)
    debug_dir = paths.combined_dir / "debug_final_route"
    stage_summary = debug_dir / "stage_divergence_summary.csv"
    stage_trace = debug_dir / "stage_divergence_cell_trace.csv"
    stage_receipt = debug_dir / "stage_divergence_receipt.json"
    semantics_receipt = debug_dir / "stage_semantics_receipt.json"
    for path in (debug_dir, stage_receipt, semantics_receipt):
        assert path.exists()

    outputs_receipt = json.loads(paths.outputs_receipt_path.read_text(encoding='utf-8'))
    assert outputs_receipt['written_outputs']['stage_divergence_debug_dir'] == str(debug_dir)
    assert outputs_receipt['written_outputs']['stage_divergence_receipt'] == str(stage_receipt)
    assert outputs_receipt['written_outputs']['stage_semantics_receipt'] == str(semantics_receipt)
    assert outputs_receipt['diagnostic_only_outputs']['stage_divergence_summary_csv'] is None
    assert outputs_receipt['diagnostic_only_outputs']['stage_divergence_cell_trace_csv'] is None

    primary_contract = paths.combined_dir / "river_primary_surface_contract.json"
    primary_surface = paths.combined_dir / "river_primary_surface.tif"
    primary_conf = paths.combined_dir / "river_primary_surface_confidence.tif"
    primary_class = paths.combined_dir / "river_primary_surface_source_class.tif"
    primary_support = paths.combined_dir / "river_primary_surface_support_count.tif"
    primary_domain = paths.combined_dir / "river_primary_surface_domain.tif"
    primary_preserve = paths.combined_dir / "river_primary_surface_channel_core_preserve.tif"
    preserve_zone = paths.combined_dir / "river_channel_core_preservation_zone.tif"
    prepost_delta = paths.combined_dir / "river_channel_core_prepost_delta.tif"
    bank_pull_risk = paths.combined_dir / "river_channel_core_bank_pull_risk.tif"
    preserve_receipt = paths.combined_dir / "river_channel_core_preservation_receipt.json"
    for path in (primary_contract, primary_surface, primary_conf, primary_class, primary_support, primary_domain, primary_preserve, preserve_zone, prepost_delta, bank_pull_risk, preserve_receipt):
        assert path.exists()
    assert ab_out["river_primary_surface"] == str(primary_surface)
    assert ab_out["river_primary_surface_contract"] == str(primary_contract)
    assert ab_out["river_channel_core_preservation_receipt"] == str(preserve_receipt)



def test_write_final_route_outputs_does_not_write_final_dem_when_primary_contract_invalid(tmp_path: Path):
    auth_arr = np.array([[1.0, 1.5],[1.1, 1.6]], dtype=np.float32)
    auth_path = _write_raster(tmp_path / "auth.tif", auth_arr)
    cfg = _Cfg(out_dir=tmp_path, authoritative_base=auth_path)
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=None, report={})
    with rasterio.open(auth_path) as ds:
        profile = ds.profile.copy()
    guidance = SimpleNamespace(auth=auth_arr, profile=profile, nodata=-9999.0, sdb_depth_path=None, sdb_guide_points_path=None, river_guide_points_path=None)
    terrain = SimpleNamespace(
        result={
            "locked": np.ones((2, 2), dtype=bool),
            "gap": np.zeros((2, 2), dtype=bool),
            "eligible": np.zeros((2, 2), dtype=bool),
            "support": np.ones((2, 2), dtype=np.uint8),
            "regime": np.ones((2, 2), dtype=np.uint8),
            "support_distance_m": np.zeros((2, 2), dtype=np.float32),
            "support_density": np.ones((2, 2), dtype=np.float32),
            "anchor_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "guidance_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "conditioned_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "guidance_influence": np.zeros((2, 2), dtype=np.float32),
            "coastal_sdb_confidence": np.zeros((2, 2), dtype=np.float32),
            "river_anchor_distance_m": np.zeros((2, 2), dtype=np.float32),
            "river_anchor_density": np.zeros((2, 2), dtype=np.float32),
            "river_scaffold_confidence": np.zeros((2, 2), dtype=np.float32),
            "river_bank_distance_m": np.zeros((2, 2), dtype=np.float32),
            "river_bank_influence": np.zeros((2, 2), dtype=np.float32),
            "river_bank_elevation": auth_arr + 0.5,
            "river_primary_surface": auth_arr.copy(),
            "river_primary_surface_confidence": np.full((2, 2), 0.9, dtype=np.float32),
            "river_primary_surface_source_class": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_support_count": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_domain": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_contract": {"ok": False, "failures": ["test_failure"], "metrics": {}},
            "river_bank_continuity_weight": None,
            "river_bank_graph_confidence": None,
            "river_bank_confluence_damping": None,
            "river_bank_estuary_side_decay": None,
            "conditioned": auth_arr.copy(),
            "provenance": np.full((2, 2), 10, dtype=np.uint8),
            "authoritative_first_contract": {"ok": True, "locked_changed_count": 0, "background_changed_count": 0, "low_confidence_fill_outside_guidance_count": 0},
            "support_note": "ok",
            "guidance_surface": auth_arr.copy(),
        },
        source_candidate={"stats": {}, "backstop_policy": {}, "provenance_codes": {}},
        candidate_prov=np.zeros((2, 2), dtype=np.uint8),
    )
    report = {"authoritative_base_auto": {}, "sdb": {"artifacts": {}}, "river": {"outputs": {}}}
    import pytest
    with pytest.raises(RuntimeError, match="river primary surface contract failed"):
        write_final_route_outputs(cfg=cfg, paths=paths, guidance=guidance, terrain=terrain, candidate_path=None, provenance_path=None, report=report)
    assert not paths.conditioned_path.exists()
    assert (paths.river_primary_surface_contract_path).exists()


def test_write_final_route_outputs_clears_stale_authoritative_locked_support_before_validation(tmp_path: Path):
    auth_arr = np.array([[10.0, np.nan], [30.0, 40.0]], dtype=np.float32)
    auth_path = _write_raster(tmp_path / "auth.tif", auth_arr)
    cfg = _Cfg(out_dir=tmp_path, authoritative_base=auth_path)
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=None, report={})
    assert paths is not None

    with rasterio.open(auth_path) as ds:
        profile = ds.profile.copy()

    guidance = SimpleNamespace(
        auth=auth_arr,
        profile=profile,
        nodata=-9999.0,
        sdb_depth_path=None,
        sdb_guide_points_path=None,
        river_guide_points_path=None,
    )
    terrain = SimpleNamespace(
        result={
            "locked": np.array([[True, False], [False, False]], dtype=bool),
            "gap": np.array([[False, True], [True, True]], dtype=bool),
            "eligible": np.zeros((2, 2), dtype=bool),
            "support": np.array([[1, 1], [2, 2]], dtype=np.uint8),
            "regime": np.ones((2, 2), dtype=np.uint8),
            "support_distance_m": np.zeros((2, 2), dtype=np.float32),
            "support_density": np.ones((2, 2), dtype=np.float32),
            "anchor_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "guidance_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "conditioned_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "guidance_influence": np.zeros((2, 2), dtype=np.float32),
            "coastal_sdb_confidence": np.zeros((2, 2), dtype=np.float32),
            "river_anchor_distance_m": np.zeros((2, 2), dtype=np.float32),
            "river_anchor_density": np.zeros((2, 2), dtype=np.float32),
            "river_scaffold_confidence": np.zeros((2, 2), dtype=np.float32),
            "river_bank_distance_m": np.zeros((2, 2), dtype=np.float32),
            "river_bank_influence": np.zeros((2, 2), dtype=np.float32),
            "river_bank_elevation": auth_arr + 0.5,
            "river_primary_surface": auth_arr.copy(),
            "river_primary_surface_confidence": np.full((2, 2), 0.9, dtype=np.float32),
            "river_primary_surface_source_class": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_support_count": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_domain": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_contract": {"ok": True, "failures": [], "metrics": {"domain_pixels": 4, "finite_pixels": 4}},
            "river_bank_continuity_weight": None,
            "river_bank_graph_confidence": None,
            "river_bank_confluence_damping": None,
            "river_bank_estuary_side_decay": None,
            "conditioned": np.array([[10.0, 12.5], [30.0, 40.0]], dtype=np.float32),
            "provenance": np.array([[10, 0], [0, 0]], dtype=np.uint8),
            "authoritative_first_contract": {"ok": True, "locked_changed_count": 0, "background_changed_count": 0, "low_confidence_fill_outside_guidance_count": 0},
            "support_note": "ok",
            "guidance_surface": auth_arr.copy(),
        },
        source_candidate={"stats": {}, "backstop_policy": {}, "provenance_codes": {}},
        candidate_prov=np.zeros((2, 2), dtype=np.uint8),
    )
    report = {"authoritative_base_auto": {}, "sdb": {"artifacts": {}}, "river": {"outputs": {}}}

    conditioned_path, _, _, _, _, support_path = write_final_route_outputs(
        cfg=cfg,
        paths=paths,
        guidance=guidance,
        terrain=terrain,
        candidate_path=None,
        provenance_path=None,
        report=report,
    )

    assert conditioned_path.exists()
    with rasterio.open(support_path) as ds:
        support = ds.read(1)
    assert support[0, 0] == 1
    assert support[0, 1] != 1

    validation_path = paths.combined_dir / "final_route_authoritative_lock_validation.json"
    assert validation_path.exists()


def test_write_final_route_outputs_preserves_runtime_authoritative_state_without_semantic_repair(tmp_path: Path):
    auth_arr = np.array([[10.0, 20.0], [30.0, np.nan]], dtype=np.float32)
    auth_path = _write_raster(tmp_path / "auth.tif", auth_arr)
    cfg = _Cfg(out_dir=tmp_path, authoritative_base=auth_path)
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=None, report={})
    with rasterio.open(auth_path) as ds:
        profile = ds.profile.copy()

    guidance = SimpleNamespace(auth=auth_arr, profile=profile, nodata=-9999.0, sdb_depth_path=None, sdb_guide_points_path=None, river_guide_points_path=None)
    terrain = SimpleNamespace(
        result={
            "locked": np.array([[True, True], [True, False]], dtype=bool),
            "gap": np.array([[False, False], [False, True]], dtype=bool),
            "eligible": np.zeros((2, 2), dtype=bool),
            "support": np.array([[1, 1], [1, 4]], dtype=np.uint8),
            "regime": np.ones((2, 2), dtype=np.uint8),
            "support_distance_m": np.zeros((2, 2), dtype=np.float32),
            "support_density": np.ones((2, 2), dtype=np.float32),
            "anchor_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "guidance_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "conditioned_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "guidance_influence": np.array([[0.0, 0.0], [0.0, 0.75]], dtype=np.float32),
            "coastal_sdb_confidence": np.zeros((2, 2), dtype=np.float32),
            "river_anchor_distance_m": np.zeros((2, 2), dtype=np.float32),
            "river_anchor_density": np.zeros((2, 2), dtype=np.float32),
            "river_scaffold_confidence": np.zeros((2, 2), dtype=np.float32),
            "river_bank_distance_m": np.zeros((2, 2), dtype=np.float32),
            "river_bank_influence": np.zeros((2, 2), dtype=np.float32),
            "river_bank_elevation": np.where(np.isfinite(auth_arr), auth_arr + 1.0, 0.0).astype(np.float32),
            "river_primary_surface": np.where(np.isfinite(auth_arr), auth_arr, 0.0).astype(np.float32),
            "river_primary_surface_confidence": np.full((2, 2), 0.9, dtype=np.float32),
            "river_primary_surface_source_class": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_support_count": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_domain": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_contract": {"ok": True, "failures": [], "metrics": {"domain_pixels": 4, "finite_pixels": 3}},
            "river_bank_continuity_weight": None,
            "river_bank_graph_confidence": None,
            "river_bank_confluence_damping": None,
            "river_bank_estuary_side_decay": None,
            "conditioned": np.array([[10.0, 20.0], [30.0, 4.0]], dtype=np.float32),
            "provenance": np.array([[10, 10], [10, 0]], dtype=np.uint8),
            "authoritative_first_contract": {"ok": True, "locked_changed_count": 0, "background_changed_count": 0, "low_confidence_fill_outside_guidance_count": 0},
            "support_note": "ok",
            "guidance_surface": np.array([[10.0, 20.0], [30.0, 4.0]], dtype=np.float32),
        },
        source_candidate={"stats": {}, "backstop_policy": {}, "provenance_codes": {}},
        candidate_prov=np.zeros((2, 2), dtype=np.uint8),
    )
    report = {"authoritative_base_auto": {}, "sdb": {"artifacts": {}}, "river": {"outputs": {}}}

    conditioned_path, conditioned_prov_path, _, _, _, support_path = write_final_route_outputs(
        cfg=cfg,
        paths=paths,
        guidance=guidance,
        terrain=terrain,
        candidate_path=None,
        provenance_path=None,
        report=report,
    )

    with rasterio.open(conditioned_path) as ds:
        conditioned = ds.read(1)
    with rasterio.open(conditioned_prov_path) as ds:
        provenance = ds.read(1)
    with rasterio.open(support_path) as ds:
        support = ds.read(1)
    with rasterio.open(paths.guidance_influence_path) as ds:
        influence = ds.read(1)

    assert np.allclose(conditioned[0, 0], 10.0)
    assert np.allclose(conditioned[0, 1], 20.0)
    assert np.allclose(conditioned[1, 0], 30.0)
    assert support[0, 0] == 1 and support[0, 1] == 1 and support[1, 0] == 1
    assert provenance[0, 0] == 10 and provenance[0, 1] == 10 and provenance[1, 0] == 10
    assert influence[0, 0] == 0.0 and influence[0, 1] == 0.0 and influence[1, 0] == 0.0

    import json
    receipt = json.loads((paths.combined_dir / "final_route_authoritative_lock_validation.json").read_text(encoding="utf-8"))
    assert receipt["runtime_authoritative_output_validation"]["ok"] is True
    assert receipt["runtime_authoritative_output_validation"]["policy"] == "final_route_validates_runtime_authoritative_state_without_semantic_repair"


def test_write_final_route_outputs_raises_on_runtime_authoritative_mismatch_instead_of_repairing(tmp_path: Path):
    auth_arr = np.array([[10.0, 20.0], [30.0, np.nan]], dtype=np.float32)
    auth_path = _write_raster(tmp_path / "auth.tif", auth_arr)
    cfg = _Cfg(out_dir=tmp_path, authoritative_base=auth_path)
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=None, report={})
    with rasterio.open(auth_path) as ds:
        profile = ds.profile.copy()

    guidance = SimpleNamespace(auth=auth_arr, profile=profile, nodata=-9999.0, sdb_depth_path=None, sdb_guide_points_path=None, river_guide_points_path=None)
    terrain = SimpleNamespace(
        result={
            "locked": np.array([[True, True], [True, False]], dtype=bool),
            "gap": np.array([[False, False], [False, True]], dtype=bool),
            "eligible": np.zeros((2, 2), dtype=bool),
            "support": np.full((2, 2), 4, dtype=np.uint8),
            "regime": np.ones((2, 2), dtype=np.uint8),
            "support_distance_m": np.zeros((2, 2), dtype=np.float32),
            "support_density": np.ones((2, 2), dtype=np.float32),
            "anchor_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "guidance_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "conditioned_uncertainty": np.zeros((2, 2), dtype=np.float32),
            "guidance_influence": np.full((2, 2), 0.75, dtype=np.float32),
            "coastal_sdb_confidence": np.zeros((2, 2), dtype=np.float32),
            "river_anchor_distance_m": np.zeros((2, 2), dtype=np.float32),
            "river_anchor_density": np.zeros((2, 2), dtype=np.float32),
            "river_scaffold_confidence": np.zeros((2, 2), dtype=np.float32),
            "river_bank_distance_m": np.zeros((2, 2), dtype=np.float32),
            "river_bank_influence": np.zeros((2, 2), dtype=np.float32),
            "river_bank_elevation": np.where(np.isfinite(auth_arr), auth_arr + 1.0, 0.0).astype(np.float32),
            "river_primary_surface": np.where(np.isfinite(auth_arr), auth_arr, 0.0).astype(np.float32),
            "river_primary_surface_confidence": np.full((2, 2), 0.9, dtype=np.float32),
            "river_primary_surface_source_class": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_support_count": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_domain": np.ones((2, 2), dtype=np.uint8),
            "river_primary_surface_contract": {"ok": True, "failures": [], "metrics": {"domain_pixels": 4, "finite_pixels": 3}},
            "river_bank_continuity_weight": None,
            "river_bank_graph_confidence": None,
            "river_bank_confluence_damping": None,
            "river_bank_estuary_side_decay": None,
            "conditioned": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "provenance": np.zeros((2, 2), dtype=np.uint8),
            "authoritative_first_contract": {"ok": True, "locked_changed_count": 0, "background_changed_count": 0, "low_confidence_fill_outside_guidance_count": 0},
            "support_note": "ok",
            "guidance_surface": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        },
        source_candidate={"stats": {}, "backstop_policy": {}, "provenance_codes": {}},
        candidate_prov=np.zeros((2, 2), dtype=np.uint8),
    )
    report = {"authoritative_base_auto": {}, "sdb": {"artifacts": {}}, "river": {"outputs": {}}}

    import pytest
    with pytest.raises(RuntimeError, match="final-route inputs violate authoritative-lock semantics"):
        write_final_route_outputs(
            cfg=cfg,
            paths=paths,
            guidance=guidance,
            terrain=terrain,
            candidate_path=None,
            provenance_path=None,
            report=report,
        )


def test_collect_final_route_inputs_tolerates_process_utils_stub_without_sdb_resolver(tmp_path: Path, monkeypatch):
    import sys, types
    auth_arr = np.array([[1.0]], dtype=np.float32)
    auth_path = _write_raster(tmp_path / "auth.tif", auth_arr)
    cfg = _Cfg(out_dir=tmp_path, authoritative_base=auth_path)

    stub = types.ModuleType("process_utils")
    monkeypatch.setitem(sys.modules, "process_utils", stub)

    paths = collect_final_route_inputs(cfg=cfg, candidate_path=None, report={"sdb": {"artifacts": {}}, "river": {"outputs": {}}, "authoritative_base_auto": {}})
    assert paths is not None
    assert paths.auth_src == auth_path



def test_final_route_semantics_audit_does_not_negate_conditioned_surface():
    from final_route_outputs_stage import _maybe_harmonize_conditioned_elevation
    from support_classes import SupportClass

    conditioned = -np.arange(121, dtype=np.float32).reshape(11, 11) - 1.0
    auth = conditioned.copy()
    river_primary = -conditioned
    support = np.full((11, 11), int(SupportClass.AUTHORITATIVE_LOCKED), dtype=np.uint8)

    out, receipt = _maybe_harmonize_conditioned_elevation(
        conditioned=conditioned,
        auth=auth,
        river_primary_surface=river_primary,
        support=support,
    )
    assert np.array_equal(out, conditioned)
    assert receipt["applied"] is False
    assert receipt["reason"] in {"detected_possible_sign_mismatch_preserved_conditioned_surface", "already_aligned"}
