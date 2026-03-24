from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from authoritative_conditioning import support_weighted_condition_arrays
from final_route_receipts import write_json_receipt


@dataclass
class DeterministicTerrainResult:
    result: dict
    source_candidate: dict
    candidate_prov: np.ndarray
    receipt_path: str | None


def run_deterministic_terrain_stage(*, guidance, template_path: str, receipt_path: str | None = None) -> DeterministicTerrainResult:
    arrays = guidance.arrays
    auth = guidance.auth
    sdb_ok = (np.asarray(arrays["sdb_adm"]) > 0) if arrays["sdb_adm"] is not None else np.zeros(auth.shape, dtype=bool)
    river_ok = (np.asarray(arrays["river_adm"]) > 0) if arrays["river_adm"] is not None else np.zeros(auth.shape, dtype=bool)
    result = support_weighted_condition_arrays(
        candidate=None,
        auth=auth,
        sdb_depth_guidance=None,
        river_depth_guidance=None,
        sdb_guide_points_path=str(guidance.sdb_guide_points_path) if guidance.sdb_guide_points_path is not None else None,
        river_guide_points_path=str(guidance.river_guide_points_path) if guidance.river_guide_points_path is not None else None,
        guidance_template_raster=str(template_path),
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_gw=arrays["sdb_gw"],
        sdb_ti=arrays["sdb_ti"],
        river_gw=arrays["river_gw"],
        river_ti=arrays["river_ti"],
        river_support=arrays["river_support"],
        river_support_depth=arrays["river_support_depth"],
        estuary_transition=arrays["river_estuary_transition"],
        river_corridor_mask=arrays["river_corridor"],
        river_bank_influence=arrays["river_bank_influence"],
        river_bank_elevation=arrays["river_bank_elevation_xs"],
        river_bank_pair_weight=arrays["river_bank_pair_weight"],
        river_bank_continuity_weight=arrays["river_bank_continuity_weight"],
        river_bank_graph_confidence=arrays["river_bank_graph_confidence"],
        river_bank_confluence_damping=arrays["river_bank_confluence_damping"],
        river_bank_estuary_side_decay=arrays["river_bank_estuary_side_decay"],
        river_centerline_elevation=arrays["river_centerline_elevation"],
        river_centerline_influence=arrays["river_centerline_influence"],
        river_centerline_stationing=arrays["river_centerline_stationing"],
        river_longitudinal_profile_elevation=arrays["river_longitudinal_profile_elevation"],
        river_longitudinal_profile_uncertainty=arrays["river_longitudinal_profile_uncertainty"],
        river_longitudinal_profile_influence=arrays["river_longitudinal_profile_influence"],
        river_xs_support_elevation=arrays["river_xs_support_elevation"],
        river_xs_support_weight=arrays["river_xs_support_weight"],
        **guidance.support_params,
    )
    if receipt_path:
        write_json_receipt(Path(receipt_path), {
            "stage": "deterministic_terrain",
            "route_mode": "staged_final_route_single_source_of_truth",
            "support_note": result.get("support_note"),
            "result_stats": {
                "locked_pixels": int(np.count_nonzero(result.get("locked", 0) > 0)),
                "gap_pixels": int(np.count_nonzero(result.get("gap", 0) > 0)),
                "eligible_pixels": int(np.count_nonzero(result.get("eligible", 0) > 0)),
                "conditioned_finite_pixels": int(np.count_nonzero(np.isfinite(result.get("conditioned")))),
            },
            "structural_inputs": {
                "sdb_guide_points": str(guidance.sdb_guide_points_path) if guidance.sdb_guide_points_path is not None else None,
                "river_guide_points": str(guidance.river_guide_points_path) if guidance.river_guide_points_path is not None else None,
            },
        })
    return DeterministicTerrainResult(result=result, source_candidate=guidance.source_candidate, candidate_prov=guidance.candidate_prov, receipt_path=receipt_path)
