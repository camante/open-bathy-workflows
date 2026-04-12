from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from final_route_contract import validate_final_route_contract
from output_products import build_final_output_contract
from river_guidance import build_river_guidance_manifest
from sdb_guidance import build_sdb_guidance_manifest


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path




def _river_output_mapping(river_dir: Path) -> dict:
    mapping = {
        "guidance_manifest": str(river_dir / "river_guidance_manifest.json"),
        "guide_points": str(river_dir / "river_guide_points.gpkg"),
        "guidance_weight": str(river_dir / "river_guidance_weight.tif"),
        "admissibility": str(river_dir / "river_admissibility.tif"),
        "corridor_mask": str(river_dir / "river_corridor_mask.tif"),
        "bank_influence": str(river_dir / "river_bank_influence.tif"),
        "bank_elevation_xs": str(river_dir / "river_bank_elevation_xs.tif"),
        "bank_continuity_weight": str(river_dir / "river_bank_continuity_weight.tif"),
        "bank_graph_confidence": str(river_dir / "river_bank_graph_confidence.tif"),
        "bank_confluence_damping": str(river_dir / "river_bank_confluence_damping.tif"),
        "bank_estuary_side_decay": str(river_dir / "river_bank_estuary_side_decay.tif"),
        "centerline_elevation": str(river_dir / "river_centerline_elevation.tif"),
        "centerline_influence": str(river_dir / "river_centerline_influence.tif"),
        "centerline_stationing": str(river_dir / "river_centerline_stationing_m.tif"),
        "xs_support_elevation": str(river_dir / "river_xs_support_elevation.tif"),
        "xs_support_weight": str(river_dir / "river_xs_support_weight.tif"),
    }
    return mapping


def _sdb_artifact_mapping(sdb_dir: Path) -> dict:
    return {
        "guidance_manifest": str(sdb_dir / "pred_depth_guidance_manifest.json"),
        "guide_points": str(sdb_dir / "pred_depth_guide_points.gpkg"),
        "guidance_weight_raster": str(sdb_dir / "pred_depth_guidance_weight.tif"),
        "admissibility_raster": str(sdb_dir / "pred_depth_admissibility.tif"),
        "trusted_interior_raster": str(sdb_dir / "pred_depth_trusted_interior.tif"),
        "regime_class_raster": str(sdb_dir / "pred_depth_regime_class.tif"),
        "sdb_guidance_active": str(sdb_dir / "pred_depth.tif"),
        "depth_raster": str(sdb_dir / "pred_depth.tif"),
        "raw_prediction_raster": str(sdb_dir / "pred_depth.tif"),
        "lock_diff_before_overwrite_raster": str(sdb_dir / "pred_depth_lock_diff_before_overwrite.tif"),
    }

def test_guidance_manifests_expose_allowed_structural_and_diagnostic_roles(tmp_path: Path):
    depth = _touch(tmp_path / "sdb" / "pred_depth.tif")
    _touch(tmp_path / "sdb" / "pred_depth_guidance_weight.tif")
    _touch(tmp_path / "sdb" / "pred_depth_admissibility.tif")
    args = SimpleNamespace(authoritative_base="")
    sdb_manifest = build_sdb_guidance_manifest(out_root=tmp_path, depth_raster=depth, args=args)
    assert sdb_manifest["final_route_contract"]["diagnostic_only_artifacts"] == ["raw_prediction_raster", "lock_diff_before_overwrite_raster"] or sdb_manifest["final_route_contract"]["diagnostic_only_artifacts"] == ["lock_diff_before_overwrite_raster", "raw_prediction_raster"]
    assert "guide_points" in sdb_manifest["final_route_contract"]["allowed_structural_artifacts"]

    river_dir = tmp_path / "river"
    _touch(river_dir / "river_guidance_weight.tif")
    _touch(river_dir / "river_admissibility.tif")
    _touch(river_dir / "river_corridor_mask.tif")
    _touch(river_dir / "river_guide_points.gpkg")
    report = {"river": {"outputs": {}}}
    river_manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    assert sorted(river_manifest["final_route_contract"]["diagnostic_only_artifacts"]) == ["bottom_elevation", "depth_terrain"]
    assert "centerline_elevation" in river_manifest["final_route_contract"]["allowed_structural_artifacts"]


def test_final_route_contract_validates_manifest_roles(tmp_path: Path):
    depth = _touch(tmp_path / "sdb" / "pred_depth.tif")
    _touch(tmp_path / "sdb" / "pred_depth_guidance_weight.tif")
    _touch(tmp_path / "sdb" / "pred_depth_admissibility.tif")
    _touch(tmp_path / "sdb" / "pred_depth_trusted_interior.tif")
    _touch(tmp_path / "sdb" / "pred_depth_regime_class.tif")
    _touch(tmp_path / "sdb" / "pred_depth_guide_points.gpkg")
    args = SimpleNamespace(authoritative_base="")
    sdb_manifest_path = tmp_path / "sdb" / "pred_depth_guidance_manifest.json"
    sdb_manifest_path.write_text(json.dumps(build_sdb_guidance_manifest(out_root=tmp_path, depth_raster=depth, args=args)), encoding="utf-8")

    river_dir = tmp_path / "river"
    for name in [
        "river_guidance_weight.tif",
        "river_admissibility.tif",
        "river_corridor_mask.tif",
        "river_guide_points.gpkg",
        "river_centerline_elevation.tif",
        "river_centerline_influence.tif",
        "river_xs_support_elevation.tif",
        "river_xs_support_weight.tif",
        "river_bank_influence.tif",
        "river_bank_elevation_xs.tif",
        "river_bank_continuity_weight.tif",
        "river_bank_graph_confidence.tif",
        "river_bank_confluence_damping.tif",
        "river_bank_estuary_side_decay.tif",
        "river_centerline_stationing_m.tif",
    ]:
        _touch(river_dir / name)
    river_manifest_path = river_dir / "river_guidance_manifest.json"
    river_manifest_path.write_text(json.dumps(build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report={"river": {"outputs": {}}})), encoding="utf-8")

    report = {
        "sdb": {"artifacts": _sdb_artifact_mapping(tmp_path / "sdb")},
        "river": {"outputs": _river_output_mapping(river_dir)},
    }
    validated = validate_final_route_contract(report)
    assert validated["guidance_manifests"]["sdb"]["valid"] is True
    assert validated["guidance_manifests"]["river"]["valid"] is True
    assert validated["strict_all_present_and_valid"] is True


def test_output_contract_exposes_final_route_validation(tmp_path: Path):
    final_native = _touch(tmp_path / "combined" / "conditioned_depth.tif")
    final_prov = _touch(tmp_path / "combined" / "conditioned_prov.tif")
    depth = _touch(tmp_path / "sdb" / "pred_depth.tif")
    for name in [
        "pred_depth_guidance_weight.tif",
        "pred_depth_admissibility.tif",
        "pred_depth_trusted_interior.tif",
        "pred_depth_regime_class.tif",
        "pred_depth_guide_points.gpkg",
    ]:
        _touch(tmp_path / "sdb" / name)
    sdb_manifest = build_sdb_guidance_manifest(out_root=tmp_path, depth_raster=depth, args=SimpleNamespace(authoritative_base=""))
    sdb_manifest_path = tmp_path / "sdb" / "pred_depth_guidance_manifest.json"
    sdb_manifest_path.write_text(json.dumps(sdb_manifest), encoding="utf-8")

    river_dir = tmp_path / "river"
    for name in [
        "river_guidance_weight.tif",
        "river_admissibility.tif",
        "river_corridor_mask.tif",
        "river_guide_points.gpkg",
        "river_centerline_elevation.tif",
        "river_centerline_influence.tif",
        "river_xs_support_elevation.tif",
        "river_xs_support_weight.tif",
        "river_bank_influence.tif",
        "river_bank_elevation_xs.tif",
        "river_bank_continuity_weight.tif",
        "river_bank_graph_confidence.tif",
        "river_bank_confluence_damping.tif",
        "river_bank_estuary_side_decay.tif",
        "river_centerline_stationing_m.tif",
    ]:
        _touch(river_dir / name)
    river_manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report={"river": {"outputs": {}}})
    river_manifest_path = river_dir / "river_guidance_manifest.json"
    river_manifest_path.write_text(json.dumps(river_manifest), encoding="utf-8")
    report = {
        "authoritative_base": {"status": "applied", "outputs": {"conditioned_depth": str(final_native)}},
        "sdb": {"artifacts": _sdb_artifact_mapping(tmp_path / "sdb")},
        "river": {"outputs": _river_output_mapping(river_dir)},
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="")
    contract = build_final_output_contract(cfg, report, final_native=final_native, final_for_user=None, final_provenance=final_prov)
    rv = contract["guidance_contract"]["route_validation"]
    assert "authoritative_aligned_base" in rv["allowed_final_route_inputs"]
    assert "legacy_fused_candidate_raster" in rv["forbidden_structural_inputs"]
    assert rv["strict_all_present_and_valid"] is True


def test_final_route_contract_fails_when_one_manifest_is_invalid(tmp_path: Path):
    depth = _touch(tmp_path / "sdb" / "pred_depth.tif")
    for name in [
        "pred_depth_guidance_weight.tif",
        "pred_depth_admissibility.tif",
        "pred_depth_trusted_interior.tif",
        "pred_depth_regime_class.tif",
        "pred_depth_guide_points.gpkg",
    ]:
        _touch(tmp_path / "sdb" / name)
    args = SimpleNamespace(authoritative_base="")
    sdb_manifest = build_sdb_guidance_manifest(out_root=tmp_path, depth_raster=depth, args=args)
    sdb_manifest_path = tmp_path / "sdb" / "pred_depth_guidance_manifest.json"
    sdb_manifest_path.write_text(json.dumps(sdb_manifest), encoding="utf-8")

    river_dir = tmp_path / "river"
    for name in [
        "river_guidance_weight.tif",
        "river_admissibility.tif",
        "river_corridor_mask.tif",
        "river_guide_points.gpkg",
        "river_centerline_elevation.tif",
        "river_centerline_influence.tif",
        "river_xs_support_elevation.tif",
        "river_xs_support_weight.tif",
        "river_bank_influence.tif",
        "river_bank_elevation_xs.tif",
        "river_bank_continuity_weight.tif",
        "river_bank_graph_confidence.tif",
        "river_bank_confluence_damping.tif",
        "river_bank_estuary_side_decay.tif",
        "river_centerline_stationing_m.tif",
    ]:
        _touch(river_dir / name)
    river_manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report={"river": {"outputs": {}}})
    river_manifest["schema_version"] = 1
    river_manifest_path = river_dir / "river_guidance_manifest.json"
    river_manifest_path.write_text(json.dumps(river_manifest), encoding="utf-8")

    report = {
        "sdb": {"artifacts": _sdb_artifact_mapping(tmp_path / "sdb")},
        "river": {"outputs": _river_output_mapping(river_dir)},
    }
    validated = validate_final_route_contract(report)
    assert validated["guidance_manifests"]["sdb"]["valid"] is True
    assert validated["guidance_manifests"]["river"]["valid"] is False
    assert validated["all_manifest_contracts_valid"] is False
    assert validated["strict_all_present_and_valid"] is False
    assert "schema_version_mismatch" in validated["guidance_manifests"]["river"]["errors"]


def test_validate_final_route_contract_flags_missing_river_structural_outputs(tmp_path: Path):
    river_dir = tmp_path / "river"
    _touch(river_dir / "river_guidance_manifest.json")
    manifest = {
        "schema_version": 2,
        "artifact_family": "river_guidance",
        "guidance_only": True,
        "artifact_roles": {"depth_terrain": "diagnostic_only", "centerline_stationing": "centerline_stationing_coordinate"},
        "final_route_contract": {
            "allowed_structural_artifacts": ["centerline_stationing", "corridor_mask"],
            "diagnostic_only_artifacts": ["depth_terrain"],
            "forbidden_structural_inputs": [
                "legacy_fused_candidate_raster",
                "dense_river_depth_raster_as_peer_surface",
                "dense_sdb_depth_raster_as_peer_surface",
                "weighted_overlap_blended_bathymetry_as_structural_input",
            ],
        },
    }
    (river_dir / "river_guidance_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = {"sdb": {"artifacts": {}}, "river": {"outputs": {"guidance_manifest": str(river_dir / "river_guidance_manifest.json")}}}
    validated = validate_final_route_contract(report)
    river = validated["guidance_manifests"]["river"]
    assert river["valid"] is False
    assert "critical_structural_artifact_missing_on_disk" in river["errors"]
    assert "centerline_stationing" in river["missing_critical_artifacts"]



def test_final_route_contract_allows_missing_sdb_manifest_when_sdb_inactive(tmp_path: Path):
    river_dir = tmp_path / "river"
    for name in [
        "river_guidance_weight.tif",
        "river_admissibility.tif",
        "river_corridor_mask.tif",
        "river_guide_points.gpkg",
        "river_centerline_elevation.tif",
        "river_centerline_influence.tif",
        "river_xs_support_elevation.tif",
        "river_xs_support_weight.tif",
        "river_bank_influence.tif",
        "river_bank_elevation_xs.tif",
        "river_bank_continuity_weight.tif",
        "river_bank_graph_confidence.tif",
        "river_bank_confluence_damping.tif",
        "river_bank_estuary_side_decay.tif",
        "river_centerline_stationing_m.tif",
    ]:
        _touch(river_dir / name)
    river_manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report={"river": {"outputs": {}}})
    (river_dir / "river_guidance_manifest.json").write_text(json.dumps(river_manifest), encoding="utf-8")
    report = {
        "domain_inference": {"requested": ["sdb", "river", "fuse"], "effective": ["river", "fuse"], "skipped": {"sdb": "no_ocean_water_detected_by_waffles"}},
        "sdb": {"artifacts": {}},
        "river": {"outputs": _river_output_mapping(river_dir)},
    }
    validated = validate_final_route_contract(report)
    assert validated["guidance_manifests"]["sdb"]["valid"] is True
    assert "manifest_not_required_for_inactive_family" in validated["guidance_manifests"]["sdb"]["warnings"]
    assert validated["guidance_manifests"]["river"]["valid"] is True
    assert validated["strict_all_present_and_valid"] is True


def test_build_river_guidance_manifest_marks_active_structural_outputs_required_even_if_missing(tmp_path: Path):
    river_dir = tmp_path / "river"
    for name in [
        "river_guidance_weight.tif",
        "river_admissibility.tif",
        "river_corridor_mask.tif",
        "river_guide_points.gpkg",
    ]:
        _touch(river_dir / name)
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report={"river": {"outputs": {}}})
    required = set(manifest["final_route_contract"]["required_structural_artifacts"])
    assert "centerline_elevation" in required
    assert "centerline_influence" in required
    assert "centerline_stationing" in required


def test_sdb_manifest_exposes_canonical_active_guidance_key(tmp_path: Path):
    depth = _touch(tmp_path / "sdb" / "pred_depth.tif")
    _touch(tmp_path / "sdb" / "pred_depth_guidance_weight.tif")
    _touch(tmp_path / "sdb" / "pred_depth_admissibility.tif")
    locked = _touch(tmp_path / "sdb" / "pred_depth_authoritative_locked.tif")
    _touch(tmp_path / "sdb" / "pred_depth_lock_diff_before_overwrite.tif")
    _touch(tmp_path / "sdb" / "pred_depth_lock_contract.json")
    args = SimpleNamespace(authoritative_base="")
    sdb_manifest = build_sdb_guidance_manifest(out_root=tmp_path, depth_raster=depth, args=args)
    assert sdb_manifest["artifacts"]["sdb_guidance_active"].endswith(locked.name)
    assert sdb_manifest["artifact_roles"]["sdb_guidance_active"] == "active_guidance"
    assert sdb_manifest["artifact_roles"]["raw_prediction_raster"] == "diagnostic_only"


def test_validate_guidance_manifest_accepts_anchor_policy_artifacts():
    from final_route_contract import validate_guidance_manifest
    manifest = {
        "schema_version": 2,
        "artifact_family": "river_guidance",
        "guidance_only": True,
        "artifact_roles": {
            "guide_points": "structured_scaffold_points",
            "anchor_table": "canonical_anchor_policy_table",
            "anchor_summary": "canonical_anchor_policy_summary",
            "depth_terrain": "diagnostic_only",
            "bottom_elevation": "diagnostic_only",
        },
        "final_route_contract": {
            "allowed_structural_artifacts": ["guide_points", "anchor_table", "anchor_summary"],
            "required_structural_artifacts": ["guide_points"],
            "optional_structural_artifacts": ["anchor_table", "anchor_summary"],
            "diagnostic_only_artifacts": ["depth_terrain", "bottom_elevation"],
            "forbidden_structural_inputs": [
                "legacy_fused_candidate_raster",
                "dense_river_depth_raster_as_peer_surface",
                "dense_sdb_depth_raster_as_peer_surface",
                "weighted_overlap_blended_bathymetry_as_structural_input",
            ],
        },
    }
    result = validate_guidance_manifest(manifest=manifest, family="river_guidance")
    assert result["valid"] is True
