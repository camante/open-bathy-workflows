from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

from simple_river_stage_contract import (
    ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
    ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
    simple_river_stage_contract_summary,
)

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
    "sdb_guidance_active",
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

LEGACY_ALLOWED_RIVER_STRUCTURAL_ARTIFACTS = {
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
    "xs_bank_qc_points",
    "xs_bank_qc_summary",
    "bank_elevation_xs",
    "bank_pair_weight",
    "bank_continuity_weight",
    "bank_graph_confidence",
    "bank_confluence_damping",
    "bank_estuary_side_decay",
    "bank_points",
    "bank_longitudinal_fit_points",
    "bank_longitudinal_fit_summary",
    "left_bank_fit_elevation",
    "right_bank_fit_elevation",
    "bank_pair_fit_elevation",
    "authoritative_bed_anchor_curve",
    "authoritative_bed_anchor_curve_summary",
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
    "longitudinal_profile_local_authoritative_reconciliation",
    "longitudinal_profile_local_authoritative_reconciliation_influence",
    "active_core_support_elevation",
    "active_core_support_uncertainty",
    "active_core_support_influence",
    "hydraulic_backbone",
    "hydraulic_backbone_nodes",
    "hydraulic_backbone_edges",
    "channel_frame_points",
    "channel_frame_contract",
    "station_targets",
    "station_targets_summary",
    "anchor_table",
    "anchor_summary",
    "authoritative_centerline_anchors",
    "authoritative_xs_anchors",
    "channel_scaffold_nodes",
    "channel_scaffold_contract",
    "channel_surface",
    "channel_surface_confidence",
    "channel_surface_source_class",
    "channel_surface_support_count",
    "channel_surface_contract",
    "xs_participation_contract",
}

SIMPLE_RIVER_PLAN_TARGET_ARTIFACTS = {
    "authoritative_base.tif",
    "river_guidance_domain_mask.tif",
    "river_centerline_points.gpkg",
    "centerline_wse_proxy_points.gpkg",
    "centerline_authoritative_bed_points.gpkg",
    "centerline_observed_offset_points.gpkg",
    "centerline_offset_modeled_points.gpkg",
    "river_centerline_bed_backbone_points.gpkg",
    "river_primary_surface.tif",
    "river_primary_surface_authoritative_locked.tif",
    "combined/conditioned_final_dem_internal.tif",
    "combined/DEM_enhanced.tif",
}

_DIAGNOSTIC_ONLY_ARTIFACTS = {
    "sdb_guidance": {"raw_prediction_raster", "lock_diff_before_overwrite_raster"},
    "river_guidance": {"depth_terrain", "bottom_elevation"},
}


def allowed_final_route_inputs() -> list[str]:
    return list(_ALLOWED_FINAL_ROUTE_INPUTS)


def forbidden_structural_inputs() -> list[str]:
    return list(_FORBIDDEN_STRUCTURAL_INPUTS)


def allowed_river_structural_artifacts(route_mode: str = ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION) -> set[str]:
    if route_mode == ROUTE_MODE_SIMPLE_RIVER_PLAN_V1:
        return set(SIMPLE_RIVER_PLAN_TARGET_ARTIFACTS)
    return set(LEGACY_ALLOWED_RIVER_STRUCTURAL_ARTIFACTS)


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


def _expected_allowed_set(family: str, route_mode: str = ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION) -> set[str]:
    if family == "sdb_guidance":
        return set(_ALLOWED_SDB_STRUCTURAL_ARTIFACTS)
    if family == "river_guidance":
        return allowed_river_structural_artifacts(route_mode)
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


def _critical_required_artifacts(family: str, manifest: Dict[str, Any] | None = None) -> set[str]:
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


def _required_structural_artifacts(family: str, manifest: Dict[str, Any] | None) -> set[str]:
    if not isinstance(manifest, dict):
        return set()
    frc = manifest.get("final_route_contract") if isinstance(manifest.get("final_route_contract"), dict) else {}
    required = frc.get("required_structural_artifacts")
    if isinstance(required, list):
        return {str(x) for x in required}
    allowed = frc.get("allowed_structural_artifacts") if isinstance(frc.get("allowed_structural_artifacts"), list) else []
    return {str(x) for x in allowed}


def _optional_structural_artifacts(manifest: Dict[str, Any] | None) -> set[str]:
    if not isinstance(manifest, dict):
        return set()
    frc = manifest.get("final_route_contract") if isinstance(manifest.get("final_route_contract"), dict) else {}
    optional = frc.get("optional_structural_artifacts")
    if isinstance(optional, list):
        return {str(x) for x in optional}
    return set()


def _validate_manifest_outputs_present(*, manifest: Dict[str, Any] | None, report: Dict[str, Any], family: str) -> Dict[str, Any]:
    mapping = _artifact_path_mapping(report, family)
    if not isinstance(manifest, dict):
        return {"missing_declared_artifacts": [], "missing_critical_artifacts": []}
    required_set = _required_structural_artifacts(family, manifest)
    missing_declared = sorted(name for name in required_set if not _path_exists(mapping.get(name)))
    critical = _critical_required_artifacts(family, manifest=manifest)
    critical = {name for name in critical if name in required_set}
    missing_critical = sorted(name for name in critical if not _path_exists(mapping.get(name)))
    return {"missing_declared_artifacts": missing_declared, "missing_critical_artifacts": missing_critical}


def validate_guidance_manifest(
    *,
    manifest: Dict[str, Any] | None,
    family: str,
    route_mode: str = ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
) -> Dict[str, Any]:
    manifest_route_mode = route_mode
    if isinstance(manifest, dict):
        manifest_route_mode = str(manifest.get("route_mode") or route_mode)
    expected_allowed = _expected_allowed_set(family, manifest_route_mode)
    diagnostic_expected = set(_DIAGNOSTIC_ONLY_ARTIFACTS.get(family, set()))
    if not isinstance(manifest, dict):
        return {
            "present": False,
            "valid": False,
            "errors": ["manifest_missing_or_unreadable"],
            "allowed_structural_artifacts": [],
            "diagnostic_only_artifacts": [],
            "route_mode": manifest_route_mode,
            "target_route_mode": ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
        }

    errors: list[str] = []
    warnings: list[str] = []
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
    required = frc.get("required_structural_artifacts")
    optional = frc.get("optional_structural_artifacts")
    diagnostic = frc.get("diagnostic_only_artifacts")
    if not isinstance(allowed, list):
        errors.append("allowed_structural_artifacts_missing")
        allowed = []
    if required is not None and not isinstance(required, list):
        errors.append("required_structural_artifacts_invalid")
        required = []
    if optional is not None and not isinstance(optional, list):
        errors.append("optional_structural_artifacts_invalid")
        optional = []
    if not isinstance(diagnostic, list):
        errors.append("diagnostic_only_artifacts_missing")
        diagnostic = []

    allowed_set = {str(x).strip() for x in allowed if str(x).strip()}
    required_set = {str(x).strip() for x in required} if isinstance(required, list) else set(allowed_set)
    optional_set = {str(x).strip() for x in optional} if isinstance(optional, list) else set()
    diagnostic_set = {str(x).strip() for x in diagnostic if str(x).strip()}

    unexpected_allowed = sorted(name for name in allowed_set if name not in expected_allowed)
    if unexpected_allowed:
        errors.append("unexpected_allowed_structural_artifact")
        errors.extend([f"unexpected_allowed_structural_artifact:{name}" for name in unexpected_allowed])
    if not required_set.issubset(allowed_set):
        errors.append("required_structural_artifact_not_allowed")
    if not optional_set.issubset(allowed_set):
        errors.append("optional_structural_artifact_not_allowed")
    if required_set & optional_set:
        errors.append("artifact_marked_both_required_and_optional")
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

    if family == "river_guidance" and manifest_route_mode != ROUTE_MODE_SIMPLE_RIVER_PLAN_V1:
        warnings.append("river_guidance_manifest_in_transition_mode")

    return {
        "present": True,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "allowed_structural_artifacts": sorted(allowed_set),
        "required_structural_artifacts": sorted(required_set),
        "optional_structural_artifacts": sorted(optional_set),
        "diagnostic_only_artifacts": sorted(diagnostic_set),
        "route_mode": manifest_route_mode,
        "target_route_mode": ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
    }


def _family_requested_or_active(report: Dict[str, Any], family: str) -> bool:
    domain = report.get("domain_inference", {}) if isinstance(report.get("domain_inference", {}), dict) else {}
    effective = domain.get("effective", []) if isinstance(domain.get("effective", []), list) else []
    if family == "sdb_guidance":
        return "sdb" in effective
    if family == "river_guidance":
        return "river" in effective
    return True


def _missing_manifest_validation(*, family: str, report: Dict[str, Any]) -> Dict[str, Any]:
    active = _family_requested_or_active(report, family)
    if active:
        return {
            "present": False,
            "valid": False,
            "errors": ["manifest_missing"],
            "allowed_structural_artifacts": [],
            "diagnostic_only_artifacts": [],
            "route_mode": ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
            "target_route_mode": ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
        }
    return {
        "present": False,
        "valid": True,
        "errors": [],
        "warnings": ["manifest_not_required_for_inactive_family"],
        "allowed_structural_artifacts": [],
        "diagnostic_only_artifacts": [],
        "missing_declared_artifacts": [],
        "missing_critical_artifacts": [],
        "route_mode": ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
        "target_route_mode": ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
    }


def validate_final_route_contract(report: Dict[str, Any]) -> Dict[str, Any]:
    sdb = report.get("sdb", {}) if isinstance(report.get("sdb", {}), dict) else {}
    river = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    sdb_artifacts = sdb.get("artifacts", {}) if isinstance(sdb.get("artifacts", {}), dict) else {}
    river_outputs = river.get("outputs", {}) if isinstance(river.get("outputs", {}), dict) else {}
    final_route = report.get("final_dem_route", {}) if isinstance(report.get("final_dem_route", {}), dict) else {}
    current_route_mode = str(final_route.get("route_mode") or ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION)

    sdb_manifest = _read_json(sdb_artifacts.get("guidance_manifest"))
    river_manifest = _read_json(river_outputs.get("guidance_manifest"))

    sdb_validation = validate_guidance_manifest(manifest=sdb_manifest, family="sdb_guidance", route_mode=current_route_mode) if sdb_artifacts.get("guidance_manifest") else _missing_manifest_validation(family="sdb_guidance", report=report)
    river_validation = validate_guidance_manifest(manifest=river_manifest, family="river_guidance", route_mode=current_route_mode) if river_outputs.get("guidance_manifest") else _missing_manifest_validation(family="river_guidance", report=report)

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

    missing_simple_river_target_artifacts = []
    if current_route_mode == ROUTE_MODE_SIMPLE_RIVER_PLAN_V1:
        available_paths = set(str(v) for v in river_outputs.values() if isinstance(v, str))
        available_paths.update(str(v) for v in sdb_artifacts.values() if isinstance(v, str))
        for target_name in sorted(SIMPLE_RIVER_PLAN_TARGET_ARTIFACTS):
            if not any(str(p).endswith(target_name) for p in available_paths):
                missing_simple_river_target_artifacts.append(target_name)

    legacy_transitional_artifacts_present = []
    if river_manifest is not None and current_route_mode != ROUTE_MODE_SIMPLE_RIVER_PLAN_V1:
        legacy_transitional_artifacts_present = sorted(allowed_river_structural_artifacts(ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION))

    return {
        "contract_version": 2,
        "current_route_mode": current_route_mode,
        "target_route_mode": ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
        "allowed_final_route_inputs": allowed_final_route_inputs(),
        "forbidden_structural_inputs": forbidden_structural_inputs(),
        "simple_river_stage_contract": simple_river_stage_contract_summary(),
        "simple_river_target_artifacts": sorted(SIMPLE_RIVER_PLAN_TARGET_ARTIFACTS),
        "missing_simple_river_target_artifacts": missing_simple_river_target_artifacts,
        "legacy_transitional_artifacts_present": legacy_transitional_artifacts_present,
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
    "allowed_river_structural_artifacts",
    "validate_guidance_manifest",
    "validate_final_route_contract",
]
