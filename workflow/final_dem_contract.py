from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from final_dem_policy import FinalDemPolicy, default_final_dem_policy
from final_route_contract import allowed_final_route_inputs, forbidden_structural_inputs
from simple_river_stage_contract import simple_river_stage_status_placeholder


def _existing_path(value: Any) -> Optional[str]:
    if not value:
        return None
    try:
        p = Path(value)
    except (TypeError, ValueError):
        return None
    return str(p) if p.exists() else None


def build_final_dem_contract_summary(
    report: Dict[str, Any],
    *,
    policy: Optional[FinalDemPolicy] = None,
    stage_status: Optional[dict] = None,
    legacy_transitional_artifacts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    policy_obj = policy or default_final_dem_policy()
    authoritative_base = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    candidate_generation = authoritative_base.get("candidate_generation", {}) if isinstance(authoritative_base.get("candidate_generation", {}), dict) else {}
    stats = candidate_generation.get("stats", {}) if isinstance(candidate_generation.get("stats", {}), dict) else {}
    backstop_policy = candidate_generation.get("backstop_policy", {}) if isinstance(candidate_generation.get("backstop_policy", {}), dict) else {}
    sdb = report.get("sdb", {}) if isinstance(report.get("sdb", {}), dict) else {}
    river = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    sdb_artifacts = sdb.get("artifacts", {}) if isinstance(sdb.get("artifacts", {}), dict) else {}
    river_outputs = river.get("outputs", {}) if isinstance(river.get("outputs", {}), dict) else {}
    river_v2_pass4 = river.get("v2_pass4", {}) if isinstance(river.get("v2_pass4", {}), dict) else {}

    dense_sdb_role = sdb_artifacts.get("depth_raster_role", "diagnostic_only")
    river_guidance_manifest = river_outputs.get("guidance_manifest")
    active_river_guidance_surface = _existing_path(river_outputs.get("primary_river_guidance_surface"))
    river_v2_active = bool(river_v2_pass4.get("success")) and bool(active_river_guidance_surface)
    dense_river_role = "authoritative_applied_final_route_input" if river_v2_active else "diagnostic_only"

    legacy_fallback_pixels = int(stats.get("legacy_fallback_pixels", 0) or 0)
    blocked_pixels = int(stats.get("legacy_blocked_in_river_corridor_pixels", 0) or 0)
    final_route = report.get("final_dem_route", {}) if isinstance(report.get("final_dem_route", {}), dict) else {}
    legacy_enabled = bool(backstop_policy.get("legacy_candidate_enabled", legacy_fallback_pixels > 0))

    current_route_mode = str(final_route.get("route_mode") or policy_obj.current_route_mode)
    target_route_mode = str(policy_obj.target_route_mode)
    if stage_status is None:
        stage_status = simple_river_stage_status_placeholder()
    if legacy_transitional_artifacts is None:
        legacy_transitional_artifacts = []
        if dense_river_role == "diagnostic_only":
            legacy_transitional_artifacts.append("dense_river_guidance_products")
        if legacy_enabled:
            legacy_transitional_artifacts.append("legacy_candidate_backstop")

    return {
        "contract_version": 2,
        "final_dem_model": "authoritative_first_guidance_conditioned_terrain_generation",
        "current_route_mode": current_route_mode,
        "target_route_mode": target_route_mode,
        "final_dem_filename": policy_obj.final_dem_filename,
        "internal_final_dem_filename": policy_obj.internal_final_dem_filename,
        "authoritative_hard_lock_required": bool(policy_obj.authoritative_hard_lock_required),
        "write_final_dem_once": bool(policy_obj.write_final_dem_once),
        "verify_only_postwrite": bool(policy_obj.verify_only_postwrite),
        "simple_river_plan_target_route": bool(policy_obj.simple_river_plan_target_route),
        "simple_river_stage_status": stage_status,
        "legacy_transitional_artifacts_present": legacy_transitional_artifacts,
        "invariants": {
            "continuous_output": True,
            "single_authoritative_route_active": bool(final_route.get("single_authoritative_route_active", False)),
            "legacy_parallel_route_retired": bool(final_route.get("legacy_parallel_route_retired", False)),
            "authoritative_hard_lock": True,
            "guidance_subordinate_to_authoritative": True,
            "dense_inferred_rasters_diagnostic_only": bool(dense_sdb_role == "diagnostic_only" and dense_river_role == "diagnostic_only"),
            "no_legacy_candidate_backstop_in_final_route": not legacy_enabled,
            "river_v2_locked_surface_active": bool(river_v2_active),
        },
        "route_cleanup": {
            "route_mode": current_route_mode,
            "source_candidate_mode": candidate_generation.get("mode"),
            "legacy_candidate_enabled": legacy_enabled,
            "legacy_candidate_role": backstop_policy.get("legacy_candidate_role", "disabled" if not legacy_enabled else "gap_only_backstop"),
            "disallow_legacy_in_river_corridor_outside_estuary": bool(backstop_policy.get("disallow_legacy_in_river_corridor_outside_estuary", True)),
            "legacy_gap_only_backstop_pixels": legacy_fallback_pixels,
            "legacy_blocked_in_river_corridor_pixels": blocked_pixels,
            "legacy_backstop_used": bool(legacy_enabled and legacy_fallback_pixels > 0),
        },
        "guidance_roles": {
            "dense_sdb_depth": dense_sdb_role,
            "dense_river_depth": dense_river_role,
            "active_river_guidance_surface": active_river_guidance_surface,
            "preferred_guidance_inputs": [
                "authoritative_hard_locks",
                "support_classes",
                "river_guide_points",
                "sdb_guide_points",
                "trusted_interior",
                "admissibility",
                "corridor_masks",
                "plausible_bounds",
            ],
        },
        "final_route_contract": {
            "allowed_final_route_inputs": allowed_final_route_inputs(),
            "forbidden_structural_inputs": forbidden_structural_inputs(),
            "river_v2_final_route_contract": {
                "active": bool(river_v2_active),
                "execution_mode": river_v2_pass4.get("execution_mode"),
                "active_river_guidance_surface": active_river_guidance_surface,
                "active_stage": "river_primary_surface_authoritative_applied" if river_v2_active else None,
                "legacy_river_final_route_participation_blocked": bool(river_v2_active),
                "runtime_enforced": bool((river.get("v2_route_contract") if isinstance(river.get("v2_route_contract"), dict) else {}).get("runtime_enforced", False)),
            },
        },
    }


__all__ = ["build_final_dem_contract_summary"]
