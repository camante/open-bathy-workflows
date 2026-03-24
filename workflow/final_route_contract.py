from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import logging
log = logging.getLogger(__name__)


_ALLOWED_FINAL_ROUTE_INPUTS = [
    "authoritative_aligned_base",
    "authoritative_gap_mask",
    "authoritative_eligible_fill_mask",
    "support_classes",
    "structured_river_guidance_artifacts",
    "sdb_guidance_artifacts",
    "deterministic_terrain_interpolation",
    "plausible_bounds",
    "regime_masks",
    "provenance_support_confidence_outputs",
]

_FORBIDDEN_STRUCTURAL_INPUTS = [
    "legacy_fused_candidate_raster",
    "dense_river_depth_raster_as_peer_surface",
    "dense_sdb_depth_raster_as_peer_surface",
    "weighted_overlap_blended_bathymetry_as_structural_input",
]

_ALLOWED_SDB_STRUCTURAL_ARTIFACTS = {
    "guidance_weight_raster",
    "trusted_interior_raster",
    "admissibility_raster",
    "regime_class_raster",
    "guide_points",
    "lower_bound_raster",
    "upper_bound_raster",
    "confidence_raster",
    "provenance_raster",
}

_ALLOWED_RIVER_STRUCTURAL_ARTIFACTS = {
    "guidance_weight",
    "trusted_interior",
    "soft_guidance_domain",
    "admissibility",
    "regime_class",
    "guide_points",
    "authoritative_support",
    "authoritative_support_depth",
    "corridor_mask",
    "bank_edge_mask",
    "bank_distance",
    "bank_influence",
    "bank_elevation_xs",
    "bank_pair_weight",
    "bank_continuity_weight",
    "bank_graph_confidence",
    "bank_confluence_damping",
    "bank_estuary_side_decay",
    "bank_points",
    "centerline_points",
    "xs_support_points",
    "centerline_elevation",
    "centerline_influence",
    "centerline_stationing",
    "xs_support_elevation",
    "xs_support_weight",
    "retained_network",
    "scaffold_domains",
    "scaffold_manifest",
    "longitudinal_profile_contract",
    "longitudinal_profile",
    "longitudinal_profile_points",
    "longitudinal_profile_summary",
    "longitudinal_profile_elevation",
    "longitudinal_profile_uncertainty",
    "longitudinal_profile_influence",
    "hydraulic_backbone",
    "hydraulic_backbone_nodes",
    "hydraulic_backbone_edges",
}

_DIAGNOSTIC_ONLY_ARTIFACTS = {
    "sdb_guidance": {"depth_raster"},
    "river_guidance": {"depth_terrain", "bottom_elevation"},
}


def allowed_final_route_inputs() -> list[str]:
    return list(_ALLOWED_FINAL_ROUTE_INPUTS)


def forbidden_structural_inputs() -> list[str]:
    return list(_FORBIDDEN_STRUCTURAL_INPUTS)


def _read_json(path: str | Path | None) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.debug("_read_json: suppressed exception", exc_info=True)
        return None
    return obj if isinstance(obj, dict) else None


def _expected_allowed_set(family: str) -> set[str]:
    if family == "sdb_guidance":
        return set(_ALLOWED_SDB_STRUCTURAL_ARTIFACTS)
    if family == "river_guidance":
        return set(_ALLOWED_RIVER_STRUCTURAL_ARTIFACTS)
    return set()




def _artifact_path_mapping(report: Dict[str, Any], family: str) -> Dict[str, Any]:
    if family == "sdb_guidance":
        sdb = report.get("sdb", {}) if isinstance(report.get("sdb", {}), dict) else {}
        return sdb.get("artifacts", {}) if isinstance(sdb.get("artifacts", {}), dict) else {}
    if family == "river_guidance":
        river = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
        return river.get("outputs", {}) if isinstance(river.get("outputs", {}), dict) else {}
    return {}


def _path_exists(value: Any) -> bool:
    if not value:
        return False
    try:
        return Path(str(value)).exists()
    except (TypeError, ValueError, OSError):
        return False


def _critical_required_artifacts(family: str) -> set[str]:
    if family == "river_guidance":
        return {
            "guide_points",
            "admissibility",
            "corridor_mask",
            "bank_influence",
            "bank_elevation_xs",
            "bank_continuity_weight",
            "bank_graph_confidence",
            "bank_confluence_damping",
            "bank_estuary_side_decay",
            "centerline_elevation",
            "centerline_influence",
            "centerline_stationing",
            "xs_support_elevation",
            "xs_support_weight",
        }
    if family == "sdb_guidance":
        return {"guide_points", "guidance_weight_raster", "admissibility_raster"}
    return set()


def _validate_manifest_outputs_present(*, manifest: Dict[str, Any] | None, report: Dict[str, Any], family: str) -> Dict[str, Any]:
    mapping = _artifact_path_mapping(report, family)
    if not isinstance(manifest, dict):
        return {"missing_declared_artifacts": [], "missing_critical_artifacts": []}
    frc = manifest.get("final_route_contract") if isinstance(manifest.get("final_route_contract"), dict) else {}
    allowed = frc.get("allowed_structural_artifacts") if isinstance(frc.get("allowed_structural_artifacts"), list) else []
    missing_declared = sorted(name for name in {str(x) for x in allowed} if not _path_exists(mapping.get(name)))
    critical = _critical_required_artifacts(family)
    missing_critical = sorted(name for name in critical if not _path_exists(mapping.get(name)))
    return {"missing_declared_artifacts": missing_declared, "missing_critical_artifacts": missing_critical}

def validate_guidance_manifest(*, manifest: Dict[str, Any] | None, family: str) -> Dict[str, Any]:
    expected_allowed = _expected_allowed_set(family)
    diagnostic_expected = set(_DIAGNOSTIC_ONLY_ARTIFACTS.get(family, set()))
    if not isinstance(manifest, dict):
        return {
            "present": False,
            "valid": False,
            "errors": ["manifest_missing_or_unreadable"],
            "allowed_structural_artifacts": [],
            "diagnostic_only_artifacts": [],
        }

    errors: list[str] = []
    if manifest.get("schema_version") != 2:
        errors.append("schema_version_mismatch")
    if manifest.get("artifact_family") != family:
        errors.append("artifact_family_mismatch")
    if manifest.get("guidance_only") is not True:
        errors.append("guidance_only_not_true")

    frc = manifest.get("final_route_contract")
    if not isinstance(frc, dict):
        errors.append("final_route_contract_missing")
        frc = {}

    allowed = frc.get("allowed_structural_artifacts")
    diagnostic = frc.get("diagnostic_only_artifacts")
    if not isinstance(allowed, list):
        errors.append("allowed_structural_artifacts_missing")
        allowed = []
    if not isinstance(diagnostic, list):
        errors.append("diagnostic_only_artifacts_missing")
        diagnostic = []

    allowed_set = {str(x) for x in allowed}
    diagnostic_set = {str(x) for x in diagnostic}

    if not allowed_set.issubset(expected_allowed):
        errors.append("unexpected_allowed_structural_artifact")
    if not diagnostic_expected.issubset(diagnostic_set):
        errors.append("expected_diagnostic_artifact_missing")
    if allowed_set & diagnostic_set:
        errors.append("artifact_marked_both_structural_and_diagnostic")

    artifact_roles = manifest.get("artifact_roles", {}) if isinstance(manifest.get("artifact_roles"), dict) else {}
    for name in diagnostic_expected:
        if artifact_roles.get(name) != "diagnostic_only":
            errors.append(f"{name}_not_diagnostic_only")

    for name in allowed_set:
        if artifact_roles.get(name) == "diagnostic_only":
            errors.append(f"{name}_incorrectly_diagnostic_only")

    disallowed = frc.get("forbidden_structural_inputs")
    if list(disallowed) != _FORBIDDEN_STRUCTURAL_INPUTS:
        errors.append("forbidden_structural_inputs_mismatch")

    return {
        "present": True,
        "valid": not errors,
        "errors": errors,
        "allowed_structural_artifacts": sorted(allowed_set),
        "diagnostic_only_artifacts": sorted(diagnostic_set),
    }


def validate_final_route_contract(report: Dict[str, Any]) -> Dict[str, Any]:
    sdb = report.get("sdb", {}) if isinstance(report.get("sdb", {}), dict) else {}
    river = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    sdb_artifacts = sdb.get("artifacts", {}) if isinstance(sdb.get("artifacts", {}), dict) else {}
    river_outputs = river.get("outputs", {}) if isinstance(river.get("outputs", {}), dict) else {}

    sdb_manifest = _read_json(sdb_artifacts.get("guidance_manifest"))
    river_manifest = _read_json(river_outputs.get("guidance_manifest"))

    sdb_validation = validate_guidance_manifest(manifest=sdb_manifest, family="sdb_guidance") if sdb_artifacts.get("guidance_manifest") else {
        "present": False, "valid": False, "errors": ["manifest_missing"], "allowed_structural_artifacts": [], "diagnostic_only_artifacts": []
    }
    river_validation = validate_guidance_manifest(manifest=river_manifest, family="river_guidance") if river_outputs.get("guidance_manifest") else {
        "present": False, "valid": False, "errors": ["manifest_missing"], "allowed_structural_artifacts": [], "diagnostic_only_artifacts": []
    }

    sdb_presence = _validate_manifest_outputs_present(manifest=sdb_manifest, report=report, family="sdb_guidance")
    river_presence = _validate_manifest_outputs_present(manifest=river_manifest, report=report, family="river_guidance")
    if sdb_presence["missing_declared_artifacts"]:
        sdb_validation.setdefault("warnings", []).append("declared_structural_artifact_missing_on_disk")
    if sdb_presence["missing_critical_artifacts"]:
        sdb_validation["valid"] = False
        sdb_validation.setdefault("errors", []).append("critical_structural_artifact_missing_on_disk")
    sdb_validation.update(sdb_presence)

    if river_presence["missing_declared_artifacts"]:
        river_validation.setdefault("warnings", []).append("declared_structural_artifact_missing_on_disk")
    if river_presence["missing_critical_artifacts"]:
        river_validation["valid"] = False
        river_validation.setdefault("errors", []).append("critical_structural_artifact_missing_on_disk")
    river_validation.update(river_presence)

    return {
        "contract_version": 1,
        "allowed_final_route_inputs": allowed_final_route_inputs(),
        "forbidden_structural_inputs": forbidden_structural_inputs(),
        "guidance_manifests": {
            "sdb": sdb_validation,
            "river": river_validation,
        },
        "all_manifest_contracts_valid": bool(sdb_validation.get("valid") and river_validation.get("valid")),
        "strict_all_present_and_valid": bool(sdb_validation.get("valid") and river_validation.get("valid")),
    }


__all__ = [
    "allowed_final_route_inputs",
    "forbidden_structural_inputs",
    "validate_guidance_manifest",
    "validate_final_route_contract",
]
