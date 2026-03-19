from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from provenance_schema import PROVENANCE_CLASS_CODE_TO_NAME, PROVENANCE_CLASS_FAMILY
from support_classes import SUPPORT_CLASS_CODE_TO_NAME, SUPPORT_CLASS_FAMILY, REGIME_CLASS_CODE_TO_NAME


@dataclass(frozen=True)
class FinalDemPolicy:
    hard_lock_finite_authoritative_cells: bool = True
    fill_only_authoritative_nodata_gaps: bool = True
    continuous_raster_no_nodata_target: bool = True
    unsupported_gaps_do_not_fill: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return asdict(self)


def build_final_dem_policy_dict(cfg: Any, *, support_note: str, guidance_masks: Dict[str, Any], base_policy: FinalDemPolicy | None = None) -> Dict[str, Any]:
    policy = (base_policy or FinalDemPolicy()).as_dict()
    policy.update({
        "support_note": support_note,
        "support_decay_m": float(getattr(cfg, "authoritative_support_decay_m", 300.0) or 300.0),
        "support_density_radius_m": float(getattr(cfg, "authoritative_support_density_radius_m", 250.0) or 250.0),
        "coastal_sdb_support_transition_m": float(getattr(cfg, "coastal_sdb_support_transition_m", 600.0) or 600.0),
        "river_anchor_density_radius_m": float(getattr(cfg, "river_anchor_density_radius_m", 200.0) or 200.0),
        "river_scaffold_transition_m": float(getattr(cfg, "river_scaffold_transition_m", 800.0) or 800.0),
        "support_class_codes": {str(k): v for k, v in SUPPORT_CLASS_CODE_TO_NAME.items()},
        "support_class_families": {str(k): v for k, v in SUPPORT_CLASS_FAMILY.items()},
        "regime_class_codes": {str(k): v for k, v in REGIME_CLASS_CODE_TO_NAME.items()},
        "provenance_class_codes": {str(k): v for k, v in PROVENANCE_CLASS_CODE_TO_NAME.items()},
        "provenance_class_families": {str(k): v for k, v in PROVENANCE_CLASS_FAMILY.items()},
        "guidance_zero_rules": {
            "authoritative_locked": "guidance influence must be zero on authoritative locked cells",
            "river_outside_trusted_interior": "river guidance influence must be zero outside trusted export region",
            "soft_guidance_vs_anchor": "soft guidance/admissibility excludes authoritative anchor support pixels",
        },
        "guidance_masks": guidance_masks,
    })
    return policy


__all__ = ["FinalDemPolicy", "build_final_dem_policy_dict"]
