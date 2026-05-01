from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc

import numpy as np

from authoritative_conditioning import support_weighted_condition_arrays
from pipeline.final_route.final_route_receipts import write_json_receipt




def _compact_array(arr, *, dtype=None, threshold_bytes: int = 1_000_000):
    if arr is None:
        return None
    out = np.asarray(arr if dtype is None else np.asarray(arr, dtype=dtype), dtype=(dtype or np.asarray(arr).dtype))
    if out.nbytes <= threshold_bytes:
        return out
    return np.ascontiguousarray(out)


def _compact_guidance_arrays(arrays: dict) -> dict:
    compact_spec = {
        "sdb_gw": np.float16,
        "sdb_ti": np.uint8,
        "river_gw": np.float16,
        "river_ti": np.uint8,
        "river_support": np.uint8,
        "river_estuary_transition": np.uint8,
        "river_corridor": np.uint8,
        "river_bank_influence": np.float16,
        "river_bank_pair_weight": np.float16,
        "river_bank_continuity_weight": np.float16,
        "river_bank_graph_confidence": np.float16,
        "river_bank_confluence_damping": np.float16,
        "river_bank_estuary_side_decay": np.float16,
        "river_centerline_influence": np.float16,
        "river_channel_surface_confidence": np.float16,
        "river_channel_surface_source_class": np.uint8,
        "river_channel_surface_support_count": np.uint16,
        "river_channel_surface_authoritative_lock_scope": np.uint8,
        "river_channel_surface_authoritative_lock_applied": np.uint8,
        "river_channel_surface_prediction_support_confidence": np.float16,
        "river_channel_surface_measured_anchor_fraction": np.float16,
        "river_channel_surface_structure_only_fraction": np.float16,
        "river_channel_surface_low_support_caution": np.uint8,
        "river_channel_surface_prediction_admissibility": np.uint8,
        "river_longitudinal_profile_influence": np.float16,
        "river_longitudinal_profile_local_authoritative_reconciliation_influence": np.float16,
        "river_xs_support_weight": np.float16,
    }
    compacted = {}
    for key, dtype in compact_spec.items():
        compacted[key] = _compact_array(arrays.get(key), dtype=dtype)
        arrays[key] = None

    # Validation-only SDB diagnostics are not used by the terrain interpolator.
    # Drop them here so they do not survive into the peak-memory stage.
    for key in ("sdb_candidate", "sdb_confidence", "sdb_lower_bound", "sdb_upper_bound"):
        arrays[key] = None

    gc.collect()
    return compacted

@dataclass
class DeterministicTerrainResult:
    result: dict
    source_candidate: dict
    candidate_prov: np.ndarray
    receipt_path: str | None


def run_deterministic_terrain_stage(*, guidance, template_path: str, receipt_path: str | None = None) -> DeterministicTerrainResult:
    arrays = guidance.arrays
    compacted = _compact_guidance_arrays(arrays)
    auth = np.asarray(guidance.auth, dtype=np.float32)
    sdb_ok = (np.asarray(arrays["sdb_adm"]) > 0) if arrays["sdb_adm"] is not None else np.zeros(auth.shape, dtype=bool)
    river_ok = (np.asarray(arrays["river_adm"]) > 0) if arrays["river_adm"] is not None else np.zeros(auth.shape, dtype=bool)
    result = support_weighted_condition_arrays(
        candidate=None,
        background_surface=guidance.baseline_background,
        auth=auth,
        sdb_depth_guidance=None,
        river_depth_guidance=None,
        primary_river_guidance_surface=arrays.get("primary_river_guidance_surface"),
        sdb_guide_points_path=str(guidance.sdb_guide_points_path) if guidance.sdb_guide_points_path is not None else None,
        river_guide_points_path=str(guidance.river_guide_points_path) if guidance.river_guide_points_path is not None else None,
        guidance_template_raster=str(template_path),
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_gw=compacted["sdb_gw"],
        sdb_ti=compacted["sdb_ti"],
        river_gw=compacted["river_gw"],
        river_ti=compacted["river_ti"],
        river_support=compacted["river_support"],
        river_support_depth=arrays["river_support_depth"],
        estuary_transition=compacted["river_estuary_transition"],
        river_corridor_mask=compacted["river_corridor"],
        river_bank_influence=compacted["river_bank_influence"],
        river_bank_elevation=arrays["river_bank_elevation_xs"],
        river_bank_pair_weight=compacted["river_bank_pair_weight"],
        river_bank_continuity_weight=compacted["river_bank_continuity_weight"],
        river_bank_graph_confidence=compacted["river_bank_graph_confidence"],
        river_bank_confluence_damping=compacted["river_bank_confluence_damping"],
        river_bank_estuary_side_decay=compacted["river_bank_estuary_side_decay"],
        river_centerline_elevation=arrays["river_centerline_elevation"],
        river_centerline_influence=compacted["river_centerline_influence"],
        river_centerline_stationing=arrays["river_centerline_stationing"],
        river_channel_surface=arrays.get("river_channel_surface"),
        river_channel_surface_confidence=compacted["river_channel_surface_confidence"],
        river_channel_surface_source_class=compacted["river_channel_surface_source_class"],
        river_channel_surface_support_count=compacted["river_channel_surface_support_count"],
        river_channel_surface_authoritative_lock_scope=compacted["river_channel_surface_authoritative_lock_scope"],
        river_channel_surface_authoritative_lock_applied=compacted["river_channel_surface_authoritative_lock_applied"],
        river_channel_surface_prediction_support_confidence=compacted["river_channel_surface_prediction_support_confidence"],
        river_channel_surface_measured_anchor_fraction=compacted["river_channel_surface_measured_anchor_fraction"],
        river_channel_surface_structure_only_fraction=compacted["river_channel_surface_structure_only_fraction"],
        river_channel_surface_low_support_caution=compacted["river_channel_surface_low_support_caution"],
        river_channel_surface_prediction_admissibility=compacted["river_channel_surface_prediction_admissibility"],
        river_longitudinal_profile_elevation=arrays["river_longitudinal_profile_elevation"],
        river_longitudinal_profile_uncertainty=arrays["river_longitudinal_profile_uncertainty"],
        river_longitudinal_profile_influence=compacted["river_longitudinal_profile_influence"],
        river_longitudinal_profile_local_authoritative_reconciliation=arrays.get("river_longitudinal_profile_local_authoritative_reconciliation"),
        river_longitudinal_profile_local_authoritative_reconciliation_influence=compacted.get("river_longitudinal_profile_local_authoritative_reconciliation_influence"),
        river_xs_support_elevation=arrays["river_xs_support_elevation"],
        river_xs_support_weight=compacted["river_xs_support_weight"],
        **guidance.support_params,
    )
    direct_primary = arrays.get("primary_river_guidance_surface")
    direct_primary_finite = int(np.count_nonzero(np.isfinite(np.asarray(direct_primary, dtype=np.float32)))) if direct_primary is not None else 0
    river_guidance_finite = 0
    for entry in result.get("memory_diagnostics", []):
        if isinstance(entry, dict) and ("river_guidance_finite" in entry):
            river_guidance_finite = int(entry.get("river_guidance_finite") or 0)
    primary_summary = result.get("river_primary_guidance_summary") if isinstance(result.get("river_primary_guidance_summary"), dict) else {}
    primary_surface_finite = int(primary_summary.get("primary_surface_finite_pixels") or 0)
    if direct_primary_finite > 0 and river_guidance_finite <= 0:
        raise RuntimeError(
            f"primary_river_guidance_surface_handoff_failed: direct_primary_finite={direct_primary_finite} river_guidance_finite={river_guidance_finite} primary_surface_finite={primary_surface_finite}"
        )

    if receipt_path:
        terrain_receipt = {
            "stage": "deterministic_terrain",
            "route_mode": "staged_final_route_single_source_of_truth",
            "support_note": result.get("support_note"),
            "result_stats": {
                "locked_pixels": int(np.count_nonzero(result.get("locked", 0) > 0)),
                "gap_pixels": int(np.count_nonzero(result.get("gap", 0) > 0)),
                "eligible_pixels": int(np.count_nonzero(result.get("eligible", 0) > 0)),
                "conditioned_finite_pixels": int(np.count_nonzero(np.isfinite(result.get("conditioned")))),
                "channel_core_preserve_pixels": int(np.count_nonzero(np.asarray(result.get("river_primary_surface_channel_core_preserve", 0)) > 0)),
                "channel_core_zone_pixels": int(np.count_nonzero(np.asarray(result.get("river_channel_core_preservation_zone", 0)) > 0)),
                "channel_core_abs_p95_m": float(((result.get("river_channel_core_preservation_receipt") or {}).get("delta_abs_p95_m", 0.0) or 0.0)),
                "channel_core_bank_pull_risk_p95": float(((result.get("river_channel_core_preservation_receipt") or {}).get("bank_pull_risk_p95", 0.0) or 0.0)),
            },
            "structural_inputs": {
                "baseline_cudem_interpolation": str(guidance.baseline_cudem_path) if guidance.baseline_cudem_path is not None else None,
                "sdb_guide_points": str(guidance.sdb_guide_points_path) if guidance.sdb_guide_points_path is not None else None,
                "river_guide_points": str(guidance.river_guide_points_path) if guidance.river_guide_points_path is not None else None,
                "primary_river_guidance_surface_finite": int(direct_primary_finite),
                "river_guidance_finite_after_handoff": int(river_guidance_finite),
                "primary_surface_finite_after_handoff": int(primary_surface_finite),
            },
        }
        write_json_receipt(Path(receipt_path), terrain_receipt)

        uptake_status = "no_direct_primary_surface"
        if direct_primary_finite > 0 and river_guidance_finite > 0:
            uptake_status = "uptake_ok"
        elif direct_primary_finite > 0 and river_guidance_finite <= 0:
            uptake_status = "handoff_lost"

        uptake_receipt_path = Path(receipt_path).with_name("river_primary_surface_uptake_receipt.json")
        write_json_receipt(uptake_receipt_path, {
            "stage": "deterministic_terrain",
            "receipt_kind": "river_primary_surface_uptake",
            "status": uptake_status,
            "primary_river_guidance_surface_finite": int(direct_primary_finite),
            "river_guidance_finite_after_handoff": int(river_guidance_finite),
            "primary_surface_finite_after_handoff": int(primary_surface_finite),
            "eligible_pixels": int(np.count_nonzero(result.get("eligible", 0) > 0)),
            "gap_pixels": int(np.count_nonzero(result.get("gap", 0) > 0)),
            "conditioned_finite_pixels": int(np.count_nonzero(np.isfinite(result.get("conditioned")))),
        })

        effect_receipt_path = Path(receipt_path).with_name("river_primary_surface_conditioning_effect_receipt.json")
        conditioned = np.asarray(result.get("conditioned"), dtype=np.float32)
        background = None if guidance.baseline_background is None else np.asarray(guidance.baseline_background, dtype=np.float32)
        direct_primary_arr = None if direct_primary is None else np.asarray(direct_primary, dtype=np.float32)
        direct_primary_mask = np.isfinite(direct_primary_arr) if direct_primary_arr is not None else np.zeros(auth.shape, dtype=bool)
        locked_mask = np.asarray(result.get("locked", 0)) > 0
        if direct_primary_finite <= 0:
            effect_status = "no_direct_primary_surface"
            effect_payload = {
                "effect_domain_pixels": 0,
                "changed_vs_background_pixels": 0,
                "changed_vs_background_unlocked_pixels": 0,
                "changed_vs_background_p95_abs_m": 0.0,
                "changed_vs_background_mean_abs_m": 0.0,
            }
        elif background is None:
            effect_status = "no_background_reference"
            effect_payload = {
                "effect_domain_pixels": int(np.count_nonzero(direct_primary_mask)),
                "changed_vs_background_pixels": 0,
                "changed_vs_background_unlocked_pixels": 0,
                "changed_vs_background_p95_abs_m": 0.0,
                "changed_vs_background_mean_abs_m": 0.0,
            }
        else:
            valid_compare = direct_primary_mask & np.isfinite(conditioned) & np.isfinite(background)
            diffs = np.abs(conditioned[valid_compare] - background[valid_compare]) if np.any(valid_compare) else np.zeros((0,), dtype=np.float32)
            changed_mask = np.zeros(auth.shape, dtype=bool)
            if np.any(valid_compare):
                changed_values = diffs > 1.0e-6
                changed_mask_indices = np.where(valid_compare)
                changed_mask[changed_mask_indices] = changed_values
            changed_count = int(np.count_nonzero(changed_mask))
            changed_unlocked_count = int(np.count_nonzero(changed_mask & ~locked_mask))
            effect_status = "effect_detected" if changed_count > 0 else "no_effect_detected"
            effect_payload = {
                "effect_domain_pixels": int(np.count_nonzero(valid_compare)),
                "changed_vs_background_pixels": changed_count,
                "changed_vs_background_unlocked_pixels": changed_unlocked_count,
                "changed_vs_background_p95_abs_m": float(np.nanpercentile(diffs, 95)) if diffs.size else 0.0,
                "changed_vs_background_mean_abs_m": float(np.nanmean(diffs)) if diffs.size else 0.0,
            }
        write_json_receipt(effect_receipt_path, {
            "stage": "deterministic_terrain",
            "receipt_kind": "river_primary_surface_conditioning_effect",
            "status": effect_status,
            "primary_river_guidance_surface_finite": int(direct_primary_finite),
            "river_guidance_finite_after_handoff": int(river_guidance_finite),
            "primary_surface_finite_after_handoff": int(primary_surface_finite),
            "eligible_pixels": int(np.count_nonzero(result.get("eligible", 0) > 0)),
            "gap_pixels": int(np.count_nonzero(result.get("gap", 0) > 0)),
            "conditioned_finite_pixels": int(np.count_nonzero(np.isfinite(result.get("conditioned")))),
            **effect_payload,
        })
    return DeterministicTerrainResult(result=result, source_candidate=guidance.source_candidate, candidate_prov=guidance.candidate_prov, receipt_path=receipt_path)
