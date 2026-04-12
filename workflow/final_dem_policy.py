from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from provenance_schema import provenance_schema_summary
from simple_river_stage_contract import (
    ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
    ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
    simple_river_stage_contract_summary,
)
from support_classes import REGIME_CLASS_CODE_TO_NAME, support_schema_summary


@dataclass(frozen=True)
class FinalDemPolicy:
    hard_lock_finite_authoritative_cells: bool = True
    fill_only_authoritative_nodata_gaps: bool = True
    continuous_raster_no_nodata_target: bool = True
    unsupported_gaps_do_not_fill: bool = False
    final_dem_filename: str = "DEM_enhanced.tif"
    internal_final_dem_filename: str = "conditioned_final_dem_internal.tif"
    authoritative_hard_lock_required: bool = True
    write_final_dem_once: bool = True
    verify_only_postwrite: bool = True
    allow_dense_river_peer_surface: bool = False
    allow_dense_sdb_peer_surface: bool = False
    current_route_mode: str = ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION
    target_route_mode: str = ROUTE_MODE_SIMPLE_RIVER_PLAN_V1
    simple_river_plan_target_route: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_final_dem_policy() -> FinalDemPolicy:
    return FinalDemPolicy()


def build_final_dem_policy_dict(
    cfg: Any,
    *,
    support_note: str,
    guidance_masks: Dict[str, Any],
    base_policy: Optional[FinalDemPolicy] = None,
) -> Dict[str, Any]:
    policy_obj = base_policy or default_final_dem_policy()
    policy = policy_obj.as_dict()
    policy.update({
        "support_note": support_note,
        "support_decay_m": float(getattr(cfg, "authoritative_support_decay_m", 300.0) or 300.0),
        "support_density_radius_m": float(getattr(cfg, "authoritative_support_density_radius_m", 250.0) or 250.0),
        "coastal_sdb_support_transition_m": float(getattr(cfg, "coastal_sdb_support_transition_m", 600.0) or 600.0),
        "river_anchor_density_radius_m": float(getattr(cfg, "river_anchor_density_radius_m", 200.0) or 200.0),
        "river_scaffold_transition_m": float(getattr(cfg, "river_scaffold_transition_m", 800.0) or 800.0),
        "support_class_codes": support_schema_summary()["codes"],
        "support_class_families": support_schema_summary()["families"],
        "regime_class_codes": {str(k): v for k, v in REGIME_CLASS_CODE_TO_NAME.items()},
        "provenance_class_codes": provenance_schema_summary()["codes"],
        "provenance_class_families": provenance_schema_summary()["families"],
        "provenance_support_family_hints": provenance_schema_summary()["support_family_hints"],
        "support_schema": support_schema_summary(),
        "provenance_schema": provenance_schema_summary(),
        "simple_river_stage_contract": simple_river_stage_contract_summary(),
        "guidance_zero_rules": {
            "authoritative_locked": "guidance influence must be zero on authoritative locked cells",
            "river_outside_trusted_interior": "river guidance influence must be zero outside trusted export region",
            "soft_guidance_vs_anchor": "soft guidance/admissibility excludes authoritative anchor support pixels",
        },
        "guidance_masks": guidance_masks,
    })
    return policy


__all__ = ["FinalDemPolicy", "default_final_dem_policy", "build_final_dem_policy_dict"]
