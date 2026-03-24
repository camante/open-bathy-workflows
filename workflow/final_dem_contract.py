from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from final_route_contract import allowed_final_route_inputs, forbidden_structural_inputs


def _existing_path(value: Any) -> Optional[str]:
    if not value:
        return None
    try:
        p = Path(value)
    except (TypeError, ValueError):
        return None
    return str(p) if p.exists() else None


def build_final_dem_contract_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    authoritative_base = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    candidate_generation = authoritative_base.get("candidate_generation", {}) if isinstance(authoritative_base.get("candidate_generation", {}), dict) else {}
    stats = candidate_generation.get("stats", {}) if isinstance(candidate_generation.get("stats", {}), dict) else {}
    backstop_policy = candidate_generation.get("backstop_policy", {}) if isinstance(candidate_generation.get("backstop_policy", {}), dict) else {}
    sdb = report.get("sdb", {}) if isinstance(report.get("sdb", {}), dict) else {}
    river = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    sdb_artifacts = sdb.get("artifacts", {}) if isinstance(sdb.get("artifacts", {}), dict) else {}
    river_outputs = river.get("outputs", {}) if isinstance(river.get("outputs", {}), dict) else {}

    dense_sdb_role = sdb_artifacts.get("depth_raster_role", "diagnostic_only")
    dense_river_role = None
    river_guidance_manifest = river_outputs.get("guidance_manifest")
    if river_guidance_manifest and _existing_path(river_guidance_manifest):
        dense_river_role = "diagnostic_only"
    else:
        dense_river_role = "diagnostic_only"

    legacy_fallback_pixels = int(stats.get("legacy_fallback_pixels", 0) or 0)
    blocked_pixels = int(stats.get("legacy_blocked_in_river_corridor_pixels", 0) or 0)
    final_route = report.get("final_dem_route", {}) if isinstance(report.get("final_dem_route", {}), dict) else {}
    legacy_enabled = bool(backstop_policy.get("legacy_candidate_enabled", legacy_fallback_pixels > 0))
    return {
        "contract_version": 1,
        "final_dem_model": "authoritative_first_guidance_conditioned_terrain_generation",
        "invariants": {
            "continuous_output": True,
            "single_authoritative_route_active": bool(final_route.get("single_authoritative_route_active", False)),
            "legacy_parallel_route_retired": bool(final_route.get("legacy_parallel_route_retired", False)),
            "authoritative_hard_lock": True,
            "guidance_subordinate_to_authoritative": True,
            "dense_inferred_rasters_diagnostic_only": True,
            "no_legacy_candidate_backstop_in_final_route": not legacy_enabled,
        },
        "route_cleanup": {
            "route_mode": final_route.get("route_mode"),
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
        },
    }


__all__ = ["build_final_dem_contract_summary"]
