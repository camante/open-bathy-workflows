from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from river_longitudinal_tendency import _distribution, _is_authoritative_like, _is_measured_xs
from river_support_semantics import station_semantics
from river_source_semantics import normalize_station_target_source_class, is_generalized_rebuild_target_source
from river_section_tendency import build_section_from_thalweg, classify_section_tendency_family, compute_tendency_depth_fraction
from river_support_roles import canonical_river_support_class
from river_target_contract import ACTIVE_INTERIOR_TARGET_AUTHORITATIVE, ACTIVE_INTERIOR_TARGET_BACKBONE

CHANNEL_CORE_ROLES = {"thalweg", "left_inner", "right_inner"}
BANK_ROLES = {"left_bank", "right_bank"}

log = logging.getLogger(__name__)


def _station_rebuild_weight(row: pd.Series, *, station_authoritative: bool, station_measured_xs: bool, station_channel_protected: bool = False, station_inner_rebuildable: bool = True) -> float:
    if bool(row.get("target_anchor_blocks_rebuild", False)):
        return 0.0
    semantics = station_semantics(
        row,
        station_authoritative=station_authoritative,
        station_measured_xs=station_measured_xs,
    )
    if str(semantics.get("station_rebuild_regime", "eligible")) != "eligible":
        return 0.0
    bank_authoritative_fraction = pd.to_numeric(row.get("station_authoritative_bank_fraction"), errors="coerce")
    bank_authoritative_fraction = float(bank_authoritative_fraction) if np.isfinite(bank_authoritative_fraction) else 0.0
    unsupported_fraction = pd.to_numeric(row.get("unsupported_fraction"), errors="coerce")
    xs_fraction = pd.to_numeric(row.get("xs_support_fraction"), errors="coerce")
    authoritative_fraction = pd.to_numeric(row.get("authoritative_anchor_fraction"), errors="coerce")
    pred_conf = pd.to_numeric(row.get("prediction_support_confidence"), errors="coerce")
    unsupported_fraction = float(unsupported_fraction) if np.isfinite(unsupported_fraction) else 0.0
    xs_fraction = float(xs_fraction) if np.isfinite(xs_fraction) else 0.0
    authoritative_fraction = float(authoritative_fraction) if np.isfinite(authoritative_fraction) else 0.0
    pred_conf = float(pred_conf) if np.isfinite(pred_conf) else np.nan
    template_type = str(semantics.get("station_template_type", row.get("xs_template_type", "measured")) or "measured")
    support_class = str(semantics.get("station_support_regime", row.get("xs_support_template_class", "missing")) or "missing")
    base = 0.20 + 0.50 * max(unsupported_fraction, 0.0) + 0.22 * max(1.0 - xs_fraction, 0.0)
    suppress = 0.20 * min(max(bank_authoritative_fraction, 0.0), 1.0)
    base *= max(0.10, 1.0 - suppress)
    base *= 1.0 - 0.25 * min(max(authoritative_fraction - bank_authoritative_fraction, 0.0), 1.0)
    if np.isfinite(pred_conf):
        base *= 1.0 + 0.45 * max(0.0, 0.65 - pred_conf)
    support_dist = _support_distance_value(row)
    station_far_from_measured = bool(semantics.get("station_far_from_measured", False))
    if station_far_from_measured and (not station_authoritative) and (not station_measured_xs):
        if np.isfinite(support_dist) and support_dist < 300.0:
            base = max(base, 0.45)
        else:
            base = min(base, 0.42)
    if template_type == "generic_symmetric":
        base = max(base, 0.52 if (np.isfinite(support_dist) and support_dist < 300.0) else 0.38)
    elif template_type == "generic_u":
        base = max(base, 0.40 if (np.isfinite(support_dist) and support_dist < 300.0) else 0.30)
    elif template_type == "hybrid_longitudinal":
        base = max(base, 0.30)
    if support_class == "bank_only_low_confidence":
        base = max(base, 0.48 if (np.isfinite(support_dist) and support_dist < 300.0) else 0.34)
    if (not station_authoritative) and (not station_measured_xs):
        if not np.isfinite(support_dist):
            base = min(base, 0.38)
        elif support_dist >= 1000.0:
            base = min(base, 0.24)
        elif support_dist >= 750.0:
            base = min(base, 0.30)
        elif support_dist >= 500.0:
            base = min(base, 0.38)
        elif support_dist >= 300.0:
            base = min(base, 0.48)
    return float(np.clip(base, 0.0, 0.85))


def _support_distance_value(row: pd.Series) -> float:
    for field in (
        "station_authoritative_bed_support_distance_m",
        "authoritative_reconciliation_support_distance_m",
        "profile_authoritative_bed_support_distance_m",
        "authoritative_bed_support_point_distance_m",
    ):
        value = pd.to_numeric(row.get(field), errors="coerce")
        if np.isfinite(value):
            return float(value)
    return float("nan")



def _support_distance_geometry_damping(row: pd.Series) -> tuple[float, float]:
    support_dist = _support_distance_value(row)
    if not np.isfinite(support_dist):
        bed_support_present = bool(row.get("station_authoritative_bed_support_present", False))
        support_regime = str(row.get("station_support_regime", row.get("xs_support_template_class", "")) or "").strip()
        if (not bed_support_present) or support_regime in {"bank_only_low_confidence", "missing", "unsupported", "none", ""}:
            return 0.32, 0.22
        return 0.60, 0.45
    if support_dist >= 1000.0:
        return 0.25, 0.18
    if support_dist >= 750.0:
        return 0.34, 0.24
    if support_dist >= 500.0:
        return 0.46, 0.34
    if support_dist >= 300.0:
        return 0.62, 0.50
    if support_dist >= 150.0:
        return 0.80, 0.72
    return 1.0, 1.0


def _station_has_real_authoritative_interior_bed_support(row: pd.Series) -> bool:
    bed_support_class = str(row.get("station_authoritative_bed_support_class", "no_authoritative_bed_support") or "no_authoritative_bed_support")
    if bed_support_class in {"authoritative_bed_strong", "authoritative_bed_nearby"}:
        return True
    channel_fraction = pd.to_numeric(row.get("station_authoritative_channel_fraction"), errors="coerce")
    bed_core_fraction = pd.to_numeric(row.get("station_authoritative_bed_core_fraction"), errors="coerce")
    return bool(
        (np.isfinite(channel_fraction) and float(channel_fraction) > 0.0)
        or (np.isfinite(bed_core_fraction) and float(bed_core_fraction) > 0.0)
    )


def _station_backbone_reference_value(row: pd.Series) -> float:
    for field in (
        "backbone_bed_reference_z_m",
        "backbone_target_z_m",
        "target_thalweg_z_m",
        "profile_network_backbone_z_m",
        "graph_backbone_z_m",
    ):
        value = pd.to_numeric(row.get(field), errors="coerce")
        if np.isfinite(value):
            return float(value)
    return float("nan")


def _station_thalweg_backbone_guardrail_tolerance_m(row: pd.Series) -> float:
    component_class = str(row.get("component_support_class", "unknown") or "unknown")
    support_regime = str(row.get("station_support_regime", "missing") or "missing")
    real_bed_support = _station_has_real_authoritative_interior_bed_support(row)
    support_dist = _support_distance_value(row)

    if real_bed_support:
        tolerance = 1.10
        if np.isfinite(support_dist):
            if support_dist <= 50.0:
                tolerance = 1.35
            elif support_dist <= 100.0:
                tolerance = 1.20
    elif component_class == "anchored_mainstem":
        tolerance = 0.65
    elif component_class == "unsupported_mainstem":
        tolerance = 0.45
    elif component_class == "unsupported_side_component":
        tolerance = 0.35
    elif component_class == "tiny_detached_component":
        tolerance = 0.28
    else:
        tolerance = 0.50

    if not real_bed_support:
        if support_regime in {"bank_only_low_confidence", "bank_only_authoritative"}:
            tolerance = min(tolerance, 0.40 if component_class != "tiny_detached_component" else 0.28)
        elif support_regime in {"supported_transition", "bank_supported_with_good_longitudinal_context"}:
            tolerance = min(tolerance, 0.55)
        if np.isfinite(support_dist):
            if support_dist >= 1000.0:
                tolerance = min(tolerance, 0.25)
            elif support_dist >= 500.0:
                tolerance = min(tolerance, 0.32)
            elif support_dist >= 300.0:
                tolerance = min(tolerance, 0.40)

    return float(np.clip(tolerance, 0.20, 1.50))


def _station_backbone_guardrail_weight(row: pd.Series, *, tolerance_m: float) -> float:
    if not np.isfinite(tolerance_m) or tolerance_m <= 0.0:
        return 0.0
    component_class = str(row.get("component_support_class", "unknown") or "unknown")
    if _station_has_real_authoritative_interior_bed_support(row):
        base = 1.5
    elif component_class == "unsupported_mainstem":
        base = 5.0
    elif component_class == "unsupported_side_component":
        base = 6.0
    elif component_class == "tiny_detached_component":
        base = 7.0
    elif component_class == "anchored_mainstem":
        base = 3.5
    else:
        base = 4.0
    return float(np.clip(base * (0.45 / float(tolerance_m)), 1.0, 12.0))


def _station_backbone_guardrail_eligible(row: pd.Series, *, rebuild_mode: str) -> bool:
    if rebuild_mode == "skip":
        return False
    current_thalweg = pd.to_numeric(row.get("thalweg_z_m"), errors="coerce")
    return _station_prefers_backbone_bed_reference(row, current_thalweg_z=float(current_thalweg) if np.isfinite(current_thalweg) else float("nan"))


def _station_backbone_bed_reference_plausible(row: pd.Series, *, current_thalweg_z: float) -> bool:
    backbone_ref = pd.to_numeric(row.get("backbone_bed_z_m", row.get("backbone_bed_reference_z_m")), errors="coerce")
    if not np.isfinite(backbone_ref):
        return False
    if bool(row.get("station_target_present", False)) or bool(row.get("station_target_local_authoritative_reconciled", False)):
        return True
    if np.isfinite(current_thalweg_z) and abs(float(backbone_ref) - float(current_thalweg_z)) <= 8.0:
        return True
    return False


def _station_prefers_backbone_bed_reference(row: pd.Series, *, current_thalweg_z: float) -> bool:
    if not _station_backbone_bed_reference_plausible(row, current_thalweg_z=current_thalweg_z):
        return False
    template_type = str(row.get("xs_template_type", "measured") or "measured")
    support_regime = str(row.get("station_support_regime", "missing") or "missing")
    component_class = str(row.get("component_support_class", "unknown") or "unknown")
    target_present = bool(row.get("station_target_present", False))
    if target_present:
        return True
    if template_type in {"generic_symmetric", "generic_u", "hybrid_longitudinal"}:
        return True
    if support_regime in {
        "bank_only_low_confidence",
        "bank_only_authoritative",
        "supported_transition",
        "bank_supported_with_good_longitudinal_context",
        "authoritative_nearby_but_not_measured_xs",
    }:
        return True
    return component_class in {"anchored_mainstem", "unsupported_mainstem", "unsupported_side_component", "tiny_detached_component"}


BACKBONE_SMOOTHING_SIGMA_STATIONS = 2.0
BACKBONE_SMOOTHING_MAX_SHIFT_M = 0.60
TRANSITION_BACKBONE_SMOOTHING_MAX_SHIFT_M = 0.20
UNSUPPORTED_MAINSTEM_BACKBONE_SMOOTHING_MAX_SHIFT_M = 1.15
UNSUPPORTED_MAINSTEM_BACKBONE_SMOOTHING_STRENGTH_FLOOR = 0.92


def _station_backbone_smoothing_max_shift(row: pd.Series) -> float:
    support_regime = str(row.get("station_support_regime", "missing") or "missing")
    component_class = str(row.get("component_support_class", "unknown") or "unknown")
    support_dist = _support_distance_value(row)
    if support_regime == "supported_transition":
        base = TRANSITION_BACKBONE_SMOOTHING_MAX_SHIFT_M
    elif component_class == "unsupported_mainstem":
        base = UNSUPPORTED_MAINSTEM_BACKBONE_SMOOTHING_MAX_SHIFT_M
    elif component_class == "unsupported_side_component":
        base = 0.80
    elif component_class == "tiny_detached_component":
        base = 0.65
    else:
        base = BACKBONE_SMOOTHING_MAX_SHIFT_M
    if np.isfinite(support_dist):
        if support_dist >= 1000.0:
            base = max(base, 1.20)
        elif support_dist >= 500.0:
            base = max(base, 0.90)
        elif support_dist <= 75.0:
            base = min(base, 0.25)
    return float(np.clip(base, 0.15, 1.40))


def _station_backbone_smoothing_eligible(row: pd.Series) -> bool:
    candidate, _, _ = _backbone_action_candidate_status(row)
    return bool(candidate)


def _station_backbone_smoothing_strength(row: pd.Series) -> float:
    if not _station_backbone_smoothing_eligible(row):
        return 0.0
    support_regime = str(row.get("station_support_regime", "missing") or "missing")
    support_dist = _support_distance_value(row)
    component_class = str(row.get("component_support_class", "unknown") or "unknown")
    candidate_group = str(row.get("backbone_candidate_group", "other_excluded") or "other_excluded")
    if support_regime == "supported_transition":
        base = 0.30
    elif candidate_group == "weak_supported_mainstem":
        base = 0.72
    elif component_class == "unsupported_mainstem":
        base = max(0.85, UNSUPPORTED_MAINSTEM_BACKBONE_SMOOTHING_STRENGTH_FLOOR)
    elif component_class == "unsupported_side_component":
        base = 0.90
    elif component_class == "tiny_detached_component":
        base = 0.75
    else:
        base = 0.55
    if np.isfinite(support_dist):
        if support_dist >= 1000.0:
            base = max(base, 1.0)
        elif support_dist >= 500.0:
            base = max(base, 0.85)
        elif support_dist >= 300.0:
            base = max(base, 0.70)
        elif support_dist <= 50.0:
            base = min(base, 0.25)
        elif support_dist <= 100.0:
            base = min(base, 0.40)
    return float(np.clip(base, 0.0, 1.0))


def _backbone_action_candidate_status(row: pd.Series) -> tuple[bool, str, str]:
    support_class = str(canonical_river_support_class(row) or "missing")
    component_class = str(row.get("component_support_class", "unknown") or "unknown")
    if not bool(row.get("profile_inside_fluvial_monotone_domain", True)):
        return False, "other_excluded", "outside_fluvial_monotone_domain"
    if bool(row.get("station_authoritative", False)) or bool(row.get("station_measured_xs", False)) or _station_has_real_authoritative_interior_bed_support(row):
        return False, "authoritative_locked", "authoritative_or_real_interior_support"
    if support_class == "bank_margin_only":
        return False, "bank_only", "bank_margin_only"
    if component_class != "unsupported_mainstem":
        if component_class in {"unsupported_side_component", "tiny_detached_component"}:
            return False, "unsupported_side", "non_mainstem_component"
        return False, "other_excluded", f"component_class_{component_class}"
    if support_class == "unsupported_interior":
        return True, "unsupported_mainstem", "candidate"
    if support_class == "weak_supported_interior":
        return True, "weak_supported_mainstem", "candidate"
    return False, "other_excluded", f"support_class_{support_class}"


def _backbone_smoothing_group_summary(profile_df: pd.DataFrame, group_col: str) -> dict[str, Any]:
    if profile_df.empty or group_col not in profile_df.columns:
        return {}
    summary: dict[str, Any] = {}
    for value, sub in profile_df.groupby(group_col, dropna=False):
        eligible = pd.to_numeric(sub.get('eligible', pd.Series(dtype=float)), errors='coerce').fillna(False).astype(bool)
        candidate = pd.to_numeric(sub.get('candidate_evaluated', pd.Series(dtype=float)), errors='coerce').fillna(False).astype(bool)
        applied = pd.to_numeric(sub.get('smoothing_applied', pd.Series(dtype=float)), errors='coerce').fillna(False).astype(bool)
        requested = pd.to_numeric(sub.get('requested_adjustment_m', pd.Series(dtype=float)), errors='coerce').to_numpy(dtype=float)
        clamped = pd.to_numeric(sub.get('clamped_adjustment_m', pd.Series(dtype=float)), errors='coerce').to_numpy(dtype=float)
        weight = pd.to_numeric(sub.get('smoothing_weight', pd.Series(dtype=float)), errors='coerce').to_numpy(dtype=float)
        delta = pd.to_numeric(sub.get('applied_adjustment_m', pd.Series(dtype=float)), errors='coerce').to_numpy(dtype=float)
        rejections = {str(k): int(v) for k, v in sub.get('rejection_reason', pd.Series(dtype='object')).astype(str).value_counts(dropna=False).items()}
        summary[str(value)] = {
            'station_count': int(len(sub)),
            'eligible_station_count': int(eligible.sum()),
            'candidate_station_count': int(candidate.sum()),
            'requested_nonzero_adjustment_count': int(np.count_nonzero(np.isfinite(requested) & (np.abs(requested) > 1.0e-6))),
            'clamped_nonzero_adjustment_count': int(np.count_nonzero(np.isfinite(clamped) & (np.abs(clamped) > 1.0e-6))),
            'applied_station_count': int(applied.sum()),
            'weight_summary': _distribution(weight),
            'delta_abs_m': _distribution(np.abs(delta[np.isfinite(delta)])),
            'rejection_reason_counts': rejections,
        }
    return summary


def _backbone_smoothing_effectiveness_warning(summary: dict[str, Any]) -> str | None:
    candidate_count = int(summary.get('candidate_station_count', 0) or 0)
    adjusted_count = int(summary.get('adjusted_station_count', 0) or 0)
    eligible_count = int(summary.get('eligible_station_count', 0) or 0)
    weak_support_count = int(summary.get('weak_support_candidate_station_count', 0) or 0)
    unsupported_mainstem_candidates = int(summary.get('unsupported_mainstem_candidate_station_count', 0) or 0)
    unsupported_mainstem_adjusted = int(summary.get('unsupported_mainstem_adjusted_station_count', 0) or 0)
    if unsupported_mainstem_candidates > 0 and unsupported_mainstem_adjusted == 0:
        return 'unsupported_mainstem_candidates_present_but_no_backbone_adjustments_applied'
    if adjusted_count == 0 and weak_support_count > 0:
        return 'weak_support_candidates_present_but_no_backbone_adjustments_applied'
    if adjusted_count == 0 and (candidate_count > 0 or eligible_count > 0):
        return 'eligible_backbone_smoothing_candidates_present_but_no_adjustments_applied'
    return None


def _build_backbone_action_gate_receipt(summary: dict[str, Any]) -> dict[str, Any]:
    by_group = summary.get("by_candidate_group", {}) or {}
    unsupported_bucket = by_group.get("unsupported_mainstem", {}) or {}
    weak_bucket = by_group.get("weak_supported_mainstem", {}) or {}
    unsupported_candidates = int(unsupported_bucket.get("candidate_station_count", 0) or 0)
    unsupported_eligible = int(unsupported_bucket.get("eligible_station_count", 0) or 0)
    unsupported_requested = int(unsupported_bucket.get("requested_nonzero_adjustment_count", 0) or 0)
    unsupported_clamped = int(unsupported_bucket.get("clamped_nonzero_adjustment_count", 0) or 0)
    unsupported_applied = int(unsupported_bucket.get("applied_station_count", 0) or 0)
    should_fail = bool(unsupported_candidates > 0 and unsupported_requested > 0 and unsupported_applied == 0)
    if unsupported_candidates == 0:
        first_blocking_gate = "no_unsupported_mainstem_candidates"
    elif unsupported_eligible == 0:
        first_blocking_gate = "eligibility_gate"
    elif unsupported_requested == 0:
        first_blocking_gate = "requested_adjustment_gate"
    elif unsupported_clamped == 0 and unsupported_applied == 0:
        first_blocking_gate = "clamp_gate"
    elif unsupported_applied == 0:
        first_blocking_gate = "application_gate"
    else:
        first_blocking_gate = "passed"
    return {
        "available": True,
        "candidate_station_count_total": int(summary.get("candidate_station_count", 0) or 0),
        "candidate_station_count_unsupported_mainstem": unsupported_candidates,
        "candidate_station_count_weak_supported_mainstem": int(weak_bucket.get("candidate_station_count", 0) or 0),
        "candidate_selection_exclusion_counts": dict(summary.get("candidate_selection_exclusion_counts", {}) or {}),
        "unsupported_mainstem_gate_counts": {
            "candidate_count": unsupported_candidates,
            "eligible_count": unsupported_eligible,
            "requested_nonzero_count": unsupported_requested,
            "clamped_nonzero_count": unsupported_clamped,
            "applied_nonzero_count": unsupported_applied,
            "rejection_reason_counts": dict(unsupported_bucket.get("rejection_reason_counts", {}) or {}),
        },
        "weak_supported_mainstem_gate_counts": {
            "candidate_count": int(weak_bucket.get("candidate_station_count", 0) or 0),
            "eligible_count": int(weak_bucket.get("eligible_station_count", 0) or 0),
            "requested_nonzero_count": int(weak_bucket.get("requested_nonzero_adjustment_count", 0) or 0),
            "clamped_nonzero_count": int(weak_bucket.get("clamped_nonzero_adjustment_count", 0) or 0),
            "applied_nonzero_count": int(weak_bucket.get("applied_station_count", 0) or 0),
            "rejection_reason_counts": dict(weak_bucket.get("rejection_reason_counts", {}) or {}),
        },
        "first_blocking_gate": first_blocking_gate,
        "should_fail": should_fail,
        "failure_reason": "unsupported_mainstem_candidates_present_with_requested_adjustments_but_no_applied_backbone_action" if should_fail else "none",
    }


def _station_backbone_led_inner_target_scale(row: pd.Series) -> tuple[bool, float, str]:
    support_regime = str(row.get("station_support_regime", "") or "").strip()
    if support_regime in {"", "missing", "nan", "None"}:
        support_regime = str(row.get("xs_support_template_class", "missing") or "missing")
    component_class = str(row.get("component_support_class", "unknown") or "unknown")
    true_measured_fraction = pd.to_numeric(row.get("station_true_measured_xs_fraction"), errors="coerce")
    true_measured_fraction = float(true_measured_fraction) if np.isfinite(true_measured_fraction) else 0.0
    if _station_has_real_authoritative_interior_bed_support(row):
        return False, 1.0, "authoritative_interior_bed_support"
    if true_measured_fraction > 0.20:
        return False, 1.0, "true_measured_xs_available"
    if support_regime in {"bank_only_low_confidence", "bank_only_authoritative"}:
        return True, 0.55, "bank_margin_only"
    if support_regime in {"supported_transition", "authoritative_nearby_but_not_measured_xs", "bank_supported_with_good_longitudinal_context"}:
        return True, 0.68, "transition_margin_guided"
    if component_class == "unsupported_mainstem":
        return True, 0.72, "unsupported_mainstem"
    if component_class in {"unsupported_side_component", "tiny_detached_component"}:
        return True, 0.62, "unsupported_side_component"
    return False, 1.0, "structured_targets_allowed"


def _centerline_width_group_summary(profile_df: pd.DataFrame, group_col: str) -> dict[str, Any]:
    if profile_df.empty or group_col not in profile_df.columns:
        return {}
    summary: dict[str, Any] = {}
    for value, sub in profile_df.groupby(group_col, dropna=False):
        backbone_led = pd.to_numeric(sub.get("backbone_led_inner_targets", pd.Series(dtype=float)), errors="coerce").fillna(False).astype(bool)
        damping = pd.to_numeric(sub.get("bank_margin_damping_active", pd.Series(dtype=float)), errors="coerce").fillna(False).astype(bool)
        adjusted = pd.to_numeric(sub.get("station_adjusted", pd.Series(dtype=float)), errors="coerce").fillna(False).astype(bool)
        channel_core_changed = pd.to_numeric(sub.get("station_changed_channel_core_node_count", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        scale = pd.to_numeric(sub.get("backbone_led_inner_relief_scale", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        inner_weight = pd.to_numeric(sub.get("inner_weight_scale", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        summary[str(value)] = {
            "station_count": int(len(sub)),
            "backbone_led_station_count": int(backbone_led.sum()),
            "bank_margin_damping_station_count": int(damping.sum()),
            "adjusted_station_count": int(adjusted.sum()),
            "changed_channel_core_station_count": int(np.count_nonzero(channel_core_changed.to_numpy(dtype=float) > 0.0)),
            "backbone_led_inner_relief_scale_summary": _distribution(scale),
            "inner_weight_scale_summary": _distribution(inner_weight),
            "backbone_led_reason_counts": {str(k): int(v) for k, v in sub.get("backbone_led_inner_reason", pd.Series(dtype="object")).astype(str).value_counts(dropna=False).items()},
            "target_source_counts": {str(k): int(v) for k, v in sub.get("primary_surface_rebuild_target_source", pd.Series(dtype="object")).astype(str).value_counts(dropna=False).items()},
        }
    return summary


def _weighted_linear_prediction(x: np.ndarray, y: np.ndarray, w: np.ndarray, x0: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    xv = x[valid]
    yv = y[valid]
    wv = w[valid]
    x_center = float(np.average(xv, weights=wv))
    dx = xv - x_center
    denom = float(np.sum(wv * dx * dx))
    if denom <= 0.0:
        return float("nan")
    y_center = float(np.average(yv, weights=wv))
    slope = float(np.sum(wv * dx * (yv - y_center)) / denom)
    return float(y_center + slope * (x0 - x_center))


def _backbone_smoothing_reference_series(work: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def _series_or_nan(name: str) -> pd.Series:
        if name in work.columns:
            return pd.to_numeric(work[name], errors="coerce")
        return pd.Series(np.nan, index=work.index, dtype=float)

    backbone_vals = _series_or_nan("backbone_target_z_m").to_numpy(dtype=float)
    fitted_vals = _series_or_nan("active_core_fit_z_m").to_numpy(dtype=float)
    thalweg_vals = _series_or_nan("thalweg_z_m").to_numpy(dtype=float)
    component_classes = work.get("component_support_class", pd.Series("unknown", index=work.index)).astype(str).to_numpy()
    support_regimes = work.get("station_support_regime", pd.Series("missing", index=work.index)).astype(str).to_numpy()
    source = np.full(len(work), "backbone_target", dtype=object)
    reference = backbone_vals.copy()
    fitted_finite = fitted_vals[np.isfinite(fitted_vals)]
    fitted_span = float(np.nanmax(fitted_finite) - np.nanmin(fitted_finite)) if fitted_finite.size else np.nan
    fit_has_structure = bool(np.isfinite(fitted_span) and fitted_span > 1.0e-6)
    use_fit = fit_has_structure & np.isfinite(fitted_vals) & (
        np.isin(component_classes, ["unsupported_mainstem", "unsupported_side_component", "tiny_detached_component"])
        | np.isin(support_regimes, [
            "unsupported",
            "missing",
            "none",
            "bank_only_low_confidence",
            "bank_only_authoritative",
            "supported_transition",
            "bank_supported_with_good_longitudinal_context",
            "authoritative_nearby_but_not_measured_xs",
        ])
    )
    reference[use_fit] = fitted_vals[use_fit]
    source[use_fit] = "active_core_fit"
    use_thalweg = ~np.isfinite(reference) & np.isfinite(thalweg_vals)
    reference[use_thalweg] = thalweg_vals[use_thalweg]
    source[use_thalweg] = "thalweg_fallback"
    return reference, source


def _apply_support_aware_backbone_smoothing(station_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = station_df.copy()
    work["smoothed_backbone_target_z_m"] = pd.to_numeric(work.get("backbone_target_z_m"), errors="coerce")
    work["backbone_smoothing_applied"] = False
    work["backbone_smoothing_delta_m"] = np.float32(0.0)
    work["backbone_smoothing_weight"] = np.float32(0.0)
    work["backbone_smoothing_max_shift_m"] = np.float32(np.nan)
    work["backbone_smoothing_eligible"] = False
    work["backbone_smoothing_rejection_reason"] = "not_evaluated"
    work["backbone_smoothing_requested_adjustment_m"] = np.float32(0.0)
    work["backbone_smoothing_clamped_adjustment_m"] = np.float32(0.0)
    work["backbone_smoothing_reference_z_m"] = np.float32(np.nan)
    work["backbone_smoothing_reference_source"] = "backbone_target"
    work["backbone_candidate"] = False
    work["backbone_candidate_group"] = "other_excluded"
    work["backbone_candidate_exclusion_reason"] = "not_evaluated"
    if work.empty:
        return work, pd.DataFrame(), {"available": False, "reason": "no_stations"}

    station_vals = pd.to_numeric(work.get("station_m"), errors="coerce").to_numpy(dtype=float)
    backbone_vals = pd.to_numeric(work.get("backbone_target_z_m"), errors="coerce").to_numpy(dtype=float)
    reference_vals, reference_source_arr = _backbone_smoothing_reference_series(work)
    smoothed_vals = backbone_vals.copy()
    finite_station_vals = station_vals[np.isfinite(station_vals)]
    station_step = float(np.nanmedian(np.diff(finite_station_vals))) if finite_station_vals.size >= 2 else 1.0
    if not np.isfinite(station_step) or station_step <= 0.0:
        station_step = 1.0
    sigma_m = float(max(BACKBONE_SMOOTHING_SIGMA_STATIONS * station_step, station_step))
    profile_rows: list[dict[str, Any]] = []
    adjusted = 0
    candidate_count = 0
    proposed_nonzero = 0
    clamped_nonzero = 0
    linear_assisted_count = 0
    rejection_counts: dict[str, int] = {}
    candidate_selection_exclusion_counts: dict[str, int] = {}

    work_rows = list(work.iterrows())
    candidate_meta = [_backbone_action_candidate_status(r) for _, r in work_rows]
    eligibility_arr = np.array([bool(meta[0]) for meta in candidate_meta], dtype=bool)
    candidate_group_arr = np.array([str(meta[1]) for meta in candidate_meta], dtype=object)
    candidate_exclusion_arr = np.array([str(meta[2]) for meta in candidate_meta], dtype=object)
    strength_arr = np.array([
        _station_backbone_smoothing_strength(r) if eligibility_arr[pos] else 0.0
        for pos, (_, r) in enumerate(work_rows)
    ], dtype=float)

    for pos, (idx, row) in enumerate(work_rows):
        eligible = bool(eligibility_arr[pos])
        candidate_group = str(candidate_group_arr[pos])
        candidate_exclusion = str(candidate_exclusion_arr[pos])
        strength = float(strength_arr[pos]) if eligible else 0.0
        work.at[idx, "backbone_candidate"] = bool(eligible)
        work.at[idx, "backbone_candidate_group"] = candidate_group
        work.at[idx, "backbone_candidate_exclusion_reason"] = candidate_exclusion
        work.at[idx, "backbone_smoothing_eligible"] = bool(eligible)
        work.at[idx, "backbone_smoothing_weight"] = np.float32(strength)
        max_shift = _station_backbone_smoothing_max_shift(row)
        work.at[idx, "backbone_smoothing_max_shift_m"] = np.float32(max_shift)
        work.at[idx, "backbone_smoothing_reference_z_m"] = np.float32(reference_vals[pos]) if np.isfinite(reference_vals[pos]) else np.float32(np.nan)
        work.at[idx, "backbone_smoothing_reference_source"] = str(reference_source_arr[pos])
        if not eligible:
            candidate_selection_exclusion_counts[candidate_exclusion] = candidate_selection_exclusion_counts.get(candidate_exclusion, 0) + 1
        current = backbone_vals[pos]
        candidate = current
        requested_shift = 0.0
        clamped_shift = 0.0
        rejection_reason = "not_eligible"
        if eligible and np.isfinite(current) and np.isfinite(station_vals[pos]):
            candidate_count += 1
            dx = station_vals - station_vals[pos]
            valid = np.isfinite(dx) & np.isfinite(reference_vals)
            if np.count_nonzero(valid) < 3:
                rejection_reason = "insufficient_finite_neighbors"
            else:
                kernel = np.exp(-0.5 * (dx[valid] / sigma_m) ** 2)
                neigh_strength = strength_arr[valid]
                neigh_row_idx = np.where(valid)[0]
                anchor_boost = np.array([
                    1.0 if (
                        bool(work.iloc[j].get("station_authoritative", False))
                        or bool(work.iloc[j].get("station_measured_xs", False))
                        or _station_has_real_authoritative_interior_bed_support(work.iloc[j])
                    ) else 0.0
                    for j in neigh_row_idx
                ], dtype=float)
                kernel *= np.clip(np.where(neigh_strength > 0.0, neigh_strength, 0.15) + (0.35 * anchor_boost), 0.15, 1.35)
                exact_mask = np.isclose(dx[valid], 0.0)
                leave_one_out_kernel = kernel.copy()
                leave_one_out_kernel[exact_mask] = 0.0
                denom = float(np.sum(leave_one_out_kernel))
                if denom <= 0.0:
                    kernel[exact_mask] = np.maximum(kernel[exact_mask], 1.0)
                    denom = float(np.sum(kernel))
                    if denom <= 0.0:
                        rejection_reason = "zero_kernel"
                    else:
                        smooth_target = float(np.sum(kernel * reference_vals[valid]) / denom)
                else:
                    smooth_target = float(np.sum(leave_one_out_kernel * reference_vals[valid]) / denom)
                if rejection_reason != "zero_kernel":
                    linear_target = _weighted_linear_prediction(
                        station_vals[valid],
                        reference_vals[valid],
                        leave_one_out_kernel if denom > 0.0 else kernel,
                        float(station_vals[pos]),
                    )
                    if np.isfinite(linear_target):
                        mean_shift = float(smooth_target - current)
                        linear_shift = float(linear_target - current)
                        if abs(linear_shift) > abs(mean_shift) + 1.0e-6:
                            smooth_target = float(linear_target)
                            linear_assisted_count += 1
                    requested_shift = float(smooth_target - current)
                    if abs(requested_shift) > 1.0e-6:
                        proposed_nonzero += 1
                    clamped_shift = float(np.clip(requested_shift, -max_shift, max_shift))
                    if abs(clamped_shift) > 1.0e-6:
                        clamped_nonzero += 1
                    candidate = float(current + strength * clamped_shift)
                    if abs(requested_shift) <= 1.0e-6:
                        rejection_reason = "already_equal"
                    elif abs(clamped_shift) <= 1.0e-6:
                        rejection_reason = "clamped_to_zero"
                    elif abs(candidate - current) <= 1.0e-6:
                        rejection_reason = "strength_zeroed"
                    else:
                        rejection_reason = "applied"
        delta = float(candidate - current) if np.isfinite(candidate) and np.isfinite(current) else 0.0
        work.at[idx, "backbone_smoothing_requested_adjustment_m"] = np.float32(requested_shift)
        work.at[idx, "backbone_smoothing_clamped_adjustment_m"] = np.float32(clamped_shift)
        work.at[idx, "backbone_smoothing_rejection_reason"] = rejection_reason
        rejection_counts[rejection_reason] = rejection_counts.get(rejection_reason, 0) + 1
        if abs(delta) > 1.0e-6:
            adjusted += 1
            work.at[idx, "backbone_smoothing_applied"] = True
            work.at[idx, "backbone_smoothing_delta_m"] = np.float32(delta)
            smoothed_vals[pos] = candidate
        profile_rows.append({
            "component_id": str(row.get("component_id", "unknown") or "unknown"),
            "station_m": float(row.get("station_m", np.nan)),
            "component_support_class": str(row.get("component_support_class", "unknown") or "unknown"),
            "support_class_canonical": str(canonical_river_support_class(row) or "missing"),
            "station_support_regime": str(row.get("station_support_regime", "missing") or "missing"),
            "distance_to_authoritative_bed_m": _support_distance_value(row),
            "original_backbone_target_z_m": float(current) if np.isfinite(current) else np.nan,
            "smoothed_backbone_target_z_m": float(candidate) if np.isfinite(candidate) else np.nan,
            "backbone_smoothing_reference_z_m": float(reference_vals[pos]) if np.isfinite(reference_vals[pos]) else np.nan,
            "backbone_smoothing_reference_source": str(reference_source_arr[pos]),
            "backbone_candidate": bool(eligible),
            "backbone_candidate_group": candidate_group,
            "backbone_candidate_exclusion_reason": candidate_exclusion,
            "candidate_evaluated": bool(eligible and np.isfinite(current) and np.isfinite(station_vals[pos])),
            "requested_adjustment_m": float(requested_shift),
            "clamped_adjustment_m": float(clamped_shift),
            "applied_adjustment_m": float(delta),
            "smoothing_applied": bool(abs(delta) > 1.0e-6),
            "smoothing_weight": float(strength),
            "max_shift_m": float(max_shift),
            "eligible": bool(eligible),
            "rejection_reason": str(rejection_reason),
        })

    work["smoothed_backbone_target_z_m"] = smoothed_vals
    profile_df = pd.DataFrame(profile_rows)
    eligible_count = int(np.count_nonzero(eligibility_arr))
    weak_support_candidate_count = 0
    if not profile_df.empty and 'backbone_candidate_group' in profile_df.columns:
        weak_support_candidate_count = int(np.count_nonzero(profile_df['backbone_candidate_group'].astype(str).eq('weak_supported_mainstem').to_numpy(dtype=bool)))
        profile_df["support_group"] = profile_df.get("component_support_class", pd.Series(dtype="object")).astype(str) + "|" + profile_df.get("station_support_regime", pd.Series(dtype="object")).astype(str)
    unsupported_mainstem_bucket = {}
    weak_supported_mainstem_bucket = {}
    if not profile_df.empty and "backbone_candidate_group" in profile_df.columns:
        candidate_group_summary = _backbone_smoothing_group_summary(profile_df, "backbone_candidate_group")
        unsupported_mainstem_bucket = candidate_group_summary.get("unsupported_mainstem", {})
        weak_supported_mainstem_bucket = candidate_group_summary.get("weak_supported_mainstem", {})
    else:
        candidate_group_summary = {}
    summary = {
        "available": True,
        "reason": "applied" if adjusted > 0 else ("no_adjustments_applied" if candidate_count > 0 or eligible_count > 0 else "no_candidates"),
        "station_count": int(len(work)),
        "eligible_station_count": int(eligible_count),
        "candidate_station_count": int(candidate_count),
        "candidate_station_count_total": int(candidate_count),
        "candidate_station_count_unsupported_mainstem": int(unsupported_mainstem_bucket.get("candidate_station_count", 0) or 0),
        "candidate_station_count_weak_supported_mainstem": int(weak_supported_mainstem_bucket.get("candidate_station_count", 0) or 0),
        "candidate_selection_exclusion_counts": candidate_selection_exclusion_counts,
        "weak_support_candidate_station_count": int(weak_support_candidate_count),
        "proposed_nonzero_adjustment_count": int(proposed_nonzero),
        "clamped_nonzero_adjustment_count": int(clamped_nonzero),
        "linear_assisted_count": int(linear_assisted_count),
        "adjusted_station_count": int(adjusted),
        "delta_abs_m": _distribution(np.abs(pd.to_numeric(work.get("backbone_smoothing_delta_m"), errors="coerce").to_numpy(dtype=float))),
        "weight_summary": _distribution(pd.to_numeric(work.get("backbone_smoothing_weight"), errors="coerce").to_numpy(dtype=float)),
        "rejection_reason_counts": rejection_counts,
        "unsupported_mainstem_candidate_station_count": int(unsupported_mainstem_bucket.get("candidate_station_count", 0) or 0),
        "unsupported_mainstem_adjusted_station_count": int(unsupported_mainstem_bucket.get("applied_station_count", 0) or 0),
        "unsupported_mainstem_rejection_reason_counts": unsupported_mainstem_bucket.get("rejection_reason_counts", {}),
        "reference_source_counts": {str(k): int(v) for k, v in profile_df.get("backbone_smoothing_reference_source", pd.Series(dtype="object")).astype(str).value_counts(dropna=False).items()} if not profile_df.empty else {},
        "by_candidate_group": candidate_group_summary,
        "by_component_support_class": _backbone_smoothing_group_summary(profile_df, 'component_support_class'),
        "by_station_support_regime": _backbone_smoothing_group_summary(profile_df, 'station_support_regime'),
        "by_support_group": _backbone_smoothing_group_summary(profile_df, 'support_group'),
    }
    summary["effectiveness_warning"] = _backbone_smoothing_effectiveness_warning(summary)
    summary["action_gate_receipt"] = _build_backbone_action_gate_receipt(summary)
    return work, profile_df, summary

def _station_core_geometry_controls(row: pd.Series) -> tuple[float, float]:
    support_regime = str(row.get("station_support_regime", "") or "").strip()
    if support_regime in {"", "missing", "nan", "None"}:
        support_regime = str(row.get("xs_support_template_class", "missing") or "missing")
    component_class = str(row.get("component_support_class", "unknown") or "unknown")
    bed_support_class = str(row.get("station_authoritative_bed_support_class", "no_authoritative_bed_support") or "no_authoritative_bed_support")
    bed_support_fraction = float(pd.to_numeric(row.get("station_authoritative_bed_support_fraction"), errors="coerce") if row.get("station_authoritative_bed_support_fraction") is not None else 0.0)
    bank_margin_fraction = float(pd.to_numeric(row.get("station_authoritative_bank_margin_fraction"), errors="coerce") if row.get("station_authoritative_bank_margin_fraction") is not None else 0.0)
    if bed_support_class in {"", "missing", "nan", "None", "no_authoritative_bed_support"} and bank_margin_fraction > 0.0 and bed_support_fraction <= 0.0:
        bed_support_class = "authoritative_bank_margin_only"
    true_measured_fraction = float(pd.to_numeric(row.get("station_true_measured_xs_fraction"), errors="coerce") if row.get("station_true_measured_xs_fraction") is not None else 0.0)
    indirect_fraction = float(pd.to_numeric(row.get("station_indirect_xs_fraction"), errors="coerce") if row.get("station_indirect_xs_fraction") is not None else 0.0)
    residual_fraction = float(pd.to_numeric(row.get("station_residual_xs_fraction"), errors="coerce") if row.get("station_residual_xs_fraction") is not None else 0.0)
    backbone_led, relief_scale, _ = _station_backbone_led_inner_target_scale(row)
    ratio_scale = 1.0
    inner_weight_scale = 1.0
    if bed_support_class == "authoritative_bank_margin_only" or support_regime in {"bank_only_low_confidence", "bank_only_authoritative"}:
        ratio_scale = 0.55
        inner_weight_scale = 0.45
    elif support_regime in {"supported_transition", "bank_supported_with_good_longitudinal_context", "authoritative_nearby_but_not_measured_xs"}:
        ratio_scale = 0.75
        inner_weight_scale = 0.70
    if backbone_led:
        if support_regime in {"bank_only_low_confidence", "bank_only_authoritative"}:
            ratio_scale = min(ratio_scale, 0.36)
            inner_weight_scale = max(inner_weight_scale, 0.78)
        elif support_regime in {"supported_transition", "bank_supported_with_good_longitudinal_context", "authoritative_nearby_but_not_measured_xs"}:
            ratio_scale = min(ratio_scale, 0.48)
            inner_weight_scale = max(inner_weight_scale, 0.72)
        elif component_class == "unsupported_mainstem":
            ratio_scale = min(ratio_scale, 0.56)
            inner_weight_scale = max(inner_weight_scale, 0.68)
        elif component_class in {"unsupported_side_component", "tiny_detached_component"}:
            ratio_scale = min(ratio_scale, 0.46)
            inner_weight_scale = max(inner_weight_scale, 0.62)
    if true_measured_fraction <= 0.10 and max(indirect_fraction, residual_fraction) >= 0.25:
        ratio_scale *= 0.85
        inner_weight_scale *= 0.85
    dist_ratio_scale, dist_inner_scale = _support_distance_geometry_damping(row)
    ratio_scale *= dist_ratio_scale
    inner_weight_scale *= dist_inner_scale
    if backbone_led:
        ratio_scale = min(ratio_scale, max(0.18, 0.72 * relief_scale))
        floor = 0.66
        if support_regime in {"bank_only_low_confidence", "bank_only_authoritative"}:
            floor = 0.58
        elif support_regime in {"supported_transition", "bank_supported_with_good_longitudinal_context", "authoritative_nearby_but_not_measured_xs"}:
            floor = 0.70
        elif component_class == "unsupported_mainstem":
            floor = 0.68
        elif component_class in {"unsupported_side_component", "tiny_detached_component"}:
            floor = 0.62
        inner_weight_scale = max(inner_weight_scale, floor)
    return float(np.clip(ratio_scale, 0.10, 1.0)), float(np.clip(inner_weight_scale, 0.08, 1.0))



def _station_inner_targets_from_tendency(
    row: pd.Series,
    *,
    target_th: float,
    left_bank_target: float,
    right_bank_target: float,
) -> tuple[float, float]:
    backbone_led, relief_scale, _ = _station_backbone_led_inner_target_scale(row)
    target_thalweg = pd.to_numeric(row.get("target_thalweg_z_m"), errors="coerce")
    left_target = pd.to_numeric(row.get("target_left_inner_z_m"), errors="coerce")
    right_target = pd.to_numeric(row.get("target_right_inner_z_m"), errors="coerce")
    if np.isfinite(target_thalweg) and not backbone_led:
        left_offset = float(left_target - target_thalweg) if np.isfinite(left_target) else np.nan
        right_offset = float(right_target - target_thalweg) if np.isfinite(right_target) else np.nan
        if np.isfinite(left_offset) and np.isfinite(right_offset):
            left_inner = target_th + left_offset
            right_inner = target_th + right_offset
            if np.isfinite(left_bank_target):
                left_inner = min(left_inner, left_bank_target - 0.05)
            if np.isfinite(right_bank_target):
                right_inner = min(right_inner, right_bank_target - 0.05)
            left_inner = max(left_inner, target_th)
            right_inner = max(right_inner, target_th)
            return float(left_inner), float(right_inner)
    width = pd.to_numeric(row.get("target_effective_channel_width_m"), errors="coerce")
    family = str(row.get("section_tendency_family", "") or "").strip() or classify_section_tendency_family(
        float(width) if np.isfinite(width) else float("nan"),
        str(row.get("station_authoritative_bed_support_class", "unsupported") or "unsupported"),
    )
    frac = pd.to_numeric(row.get("section_tendency_depth_m"), errors="coerce")
    if np.isfinite(width):
        default_frac = compute_tendency_depth_fraction(
            float(width),
            family,
            str(row.get("station_authoritative_bed_support_class", "unsupported") or "unsupported"),
            component_class=str(row.get("component_support_class", "unknown") or "unknown"),
            reconciliation_confidence=pd.to_numeric(row.get("authoritative_reconciliation_confidence"), errors="coerce"),
            support_distance_m=pd.to_numeric(row.get("component_support_median_distance_m"), errors="coerce"),
        )
        amplitude = pd.to_numeric(row.get("section_tendency_inner_relief_m"), errors="coerce")
        if np.isfinite(amplitude):
            denom = 0.28 + 0.022 * np.sqrt(min(max(float(width), 4.0), 120.0))
            if family == "narrow_v":
                denom += 0.18
            elif family == "compound_lowflow":
                denom += 0.05
            elif family == "authoritative_section_derived":
                denom += 0.10
            if denom > 0.0:
                frac = 0.20 * float(amplitude) / float(denom)
        frac = float(frac) if np.isfinite(frac) else default_frac
    else:
        width = np.nan
        frac = float(frac) if np.isfinite(frac) else 0.18
    left_inner, right_inner, _ = build_section_from_thalweg(
        float(target_th),
        float(width) if np.isfinite(width) else 20.0,
        family,
        frac,
        bank_caps=(float(left_bank_target), float(right_bank_target)),
    )
    if backbone_led:
        left_inner = float(target_th + relief_scale * max(float(left_inner) - float(target_th), 0.0))
        right_inner = float(target_th + relief_scale * max(float(right_inner) - float(target_th), 0.0))
        if np.isfinite(left_bank_target):
            left_inner = min(left_inner, float(left_bank_target) - 0.05)
        if np.isfinite(right_bank_target):
            right_inner = min(right_inner, float(right_bank_target) - 0.05)
        left_inner = max(left_inner, float(target_th))
        right_inner = max(right_inner, float(target_th))
    return float(left_inner), float(right_inner)


def _station_template_ratios(row: pd.Series) -> tuple[float, float]:
    backbone_led, relief_scale, _ = _station_backbone_led_inner_target_scale(row)
    station_target_present = bool(row.get("station_target_present", False))
    if station_target_present and not backbone_led:
        left_target = pd.to_numeric(row.get("target_left_inner_z_m"), errors="coerce")
        right_target = pd.to_numeric(row.get("target_right_inner_z_m"), errors="coerce")
        left_bank_target = pd.to_numeric(row.get("target_left_bank_z_m"), errors="coerce")
        right_bank_target = pd.to_numeric(row.get("target_right_bank_z_m"), errors="coerce")
        thalweg_target = pd.to_numeric(row.get("target_thalweg_z_m"), errors="coerce")
        left_ratio = _ratio(float(left_target), float(left_bank_target), float(thalweg_target)) if np.isfinite(left_target) else float("nan")
        right_ratio = _ratio(float(right_target), float(right_bank_target), float(thalweg_target)) if np.isfinite(right_target) else float("nan")
        if np.isfinite(left_ratio) and np.isfinite(right_ratio):
            return float(np.clip(left_ratio, 0.05, 0.95)), float(np.clip(right_ratio, -0.95, -0.05))
    template_type = str(row.get("xs_template_type", "measured") or "measured")
    left_ratio = pd.to_numeric(row.get("xs_realism_left_ratio_after"), errors="coerce")
    right_ratio = pd.to_numeric(row.get("xs_realism_right_ratio_after"), errors="coerce")
    ratio_scale, _ = _station_core_geometry_controls(row)
    if backbone_led:
        backbone_magnitude = float(np.clip(0.32 * relief_scale, 0.12, 0.26))
        if template_type == "generic_u":
            backbone_magnitude = float(np.clip(0.26 * relief_scale, 0.10, 0.22))
            return 0.90 * backbone_magnitude, -0.90 * backbone_magnitude
        return backbone_magnitude, -backbone_magnitude
    if np.isfinite(left_ratio) and np.isfinite(right_ratio):
        if template_type == "generic_symmetric":
            magnitude = float(np.nanmedian(np.abs([left_ratio, right_ratio])))
            magnitude = float(np.clip(magnitude, 0.25, 0.65))
            magnitude = float(np.clip(magnitude * ratio_scale, 0.10, 0.65))
            return magnitude, -magnitude
        if template_type == "generic_u":
            magnitude = float(np.nanmedian(np.abs([left_ratio, right_ratio])))
            magnitude = float(np.clip(magnitude, 0.18, 0.50))
            magnitude = float(np.clip(magnitude * ratio_scale, 0.08, 0.50))
            return 0.85 * magnitude, -0.85 * magnitude
        return float(left_ratio) * ratio_scale, float(right_ratio) * ratio_scale
    if template_type == "generic_symmetric":
        magnitude = float(np.clip(0.45 * ratio_scale, 0.10, 0.45))
        return magnitude, -magnitude
    if template_type == "generic_u":
        magnitude = float(np.clip(0.30 * ratio_scale, 0.08, 0.30))
        return magnitude, -magnitude
    magnitude = float(np.clip(0.40 * ratio_scale, 0.10, 0.40))
    return magnitude, -magnitude


def _station_canonical_target_section(row: pd.Series, station_group: pd.DataFrame) -> tuple[float, float, float, str]:
    station_target_present = bool(row.get("station_target_present", False))
    if not station_target_present:
        return np.nan, np.nan, np.nan, "missing"
    def _station_or_group(col: str, role: str | None = None) -> float:
        val = pd.to_numeric(row.get(col), errors="coerce")
        if np.isfinite(val):
            return float(val)
        if col in station_group.columns:
            sub = station_group
            if role is not None:
                sub = station_group.loc[station_group["node_role"].astype(str).eq(role)]
            vals = pd.to_numeric(sub.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
            if len(vals):
                return float(vals.iloc[0])
        return float("nan")
    def _source_class() -> str:
        src = str(row.get("station_target_source_class", "") or "").strip()
        if src:
            return src
        if "station_target_source_class" in station_group.columns:
            vals = station_group["station_target_source_class"].astype(str).str.strip()
            vals = vals[vals.ne("")]
            if len(vals):
                return normalize_station_target_source_class(str(vals.iloc[0]))
        return normalize_station_target_source_class(
            row.get("station_target_source_class"),
            local_reconciled=bool(row.get("station_target_local_authoritative_reconciled", False)),
            target_present=bool(row.get("station_target_present", False)),
        )
    thalweg = _station_or_group("target_thalweg_z_m", "thalweg")
    left_bank = _station_or_group("target_left_bank_z_m", "left_bank")
    right_bank = _station_or_group("target_right_bank_z_m", "right_bank")
    if np.isfinite(thalweg) and (np.isfinite(left_bank) or np.isfinite(right_bank)):
        return thalweg, left_bank, right_bank, _source_class()
    return np.nan, np.nan, np.nan, "missing"


def _station_fitted_section_targets(row: pd.Series, station_group: pd.DataFrame) -> tuple[float, float, float, str]:
    canonical_core, canonical_left_bank, canonical_right_bank, canonical_source = _station_canonical_target_section(row, station_group)
    if canonical_source != "missing":
        return canonical_core, canonical_left_bank, canonical_right_bank, canonical_source
    active_core_fit = pd.to_numeric(row.get("active_core_fit_z_m"), errors="coerce")
    if not np.isfinite(active_core_fit):
        active_core_fit = pd.to_numeric(station_group.get("active_core_fit_z_m", pd.Series(dtype=float)), errors="coerce").dropna()
        active_core_fit = float(active_core_fit.iloc[0]) if len(active_core_fit) else np.nan
    left_bank_fit = pd.to_numeric(row.get("left_bank_fit_z_m"), errors="coerce")
    if not np.isfinite(left_bank_fit) and "left_bank_fit_z_m" in station_group.columns:
        left_bank_fit = pd.to_numeric(station_group.loc[station_group["node_role"].astype(str) == "left_bank", "left_bank_fit_z_m"], errors="coerce").dropna()
        left_bank_fit = float(left_bank_fit.iloc[0]) if len(left_bank_fit) else np.nan
    right_bank_fit = pd.to_numeric(row.get("right_bank_fit_z_m"), errors="coerce")
    if not np.isfinite(right_bank_fit) and "right_bank_fit_z_m" in station_group.columns:
        right_bank_fit = pd.to_numeric(station_group.loc[station_group["node_role"].astype(str) == "right_bank", "right_bank_fit_z_m"], errors="coerce").dropna()
        right_bank_fit = float(right_bank_fit.iloc[0]) if len(right_bank_fit) else np.nan
    bank_pair_fit = pd.to_numeric(row.get("bank_pair_fit_z_m"), errors="coerce")
    if not np.isfinite(bank_pair_fit):
        vals = [v for v in [left_bank_fit, right_bank_fit] if np.isfinite(v)]
        bank_pair_fit = float(np.nanmean(vals)) if vals else np.nan
    support_regime = str(row.get("station_support_regime", "") or "").strip()
    if support_regime in {"", "missing", "nan", "None"}:
        support_regime = str(row.get("xs_support_template_class", "missing") or "missing")
    bed_support_class = str(row.get("station_authoritative_bed_support_class", "no_authoritative_bed_support") or "no_authoritative_bed_support")
    true_measured_fraction = pd.to_numeric(row.get("station_true_measured_xs_fraction"), errors="coerce")
    true_measured_fraction = float(true_measured_fraction) if np.isfinite(true_measured_fraction) else 0.0
    template_type = str(row.get("xs_template_type", "measured") or "measured")
    fitted_available = np.isfinite(active_core_fit) and (np.isfinite(left_bank_fit) or np.isfinite(right_bank_fit) or np.isfinite(bank_pair_fit))
    prefer_fitted = False
    if fitted_available:
        if bed_support_class in {"no_authoritative_bed_support", "authoritative_bank_margin_only"} and true_measured_fraction <= 0.10:
            prefer_fitted = True
        if support_regime in {"bank_only_low_confidence", "bank_only_authoritative", "supported_transition", "authoritative_nearby_but_not_measured_xs", "bank_supported_with_good_longitudinal_context"}:
            prefer_fitted = True
        if template_type in {"generic_symmetric", "generic_u", "hybrid_longitudinal"} and true_measured_fraction <= 0.20:
            prefer_fitted = True
    source = normalize_station_target_source_class("generalized_longitudinal_section" if prefer_fitted else "longitudinal_backbone_template")
    return float(active_core_fit) if np.isfinite(active_core_fit) else np.nan, float(left_bank_fit) if np.isfinite(left_bank_fit) else np.nan, float(right_bank_fit) if np.isfinite(right_bank_fit) else np.nan, source

def _node_rebuild_role_weight(role_name: str, *, rebuild_mode: str, station_weight: float, bank_protected: bool, node_is_authoritative: bool = False, inner_weight_scale: float = 1.0) -> float:
    role = str(role_name or "")
    if rebuild_mode == "skip" or node_is_authoritative:
        return 0.0
    if role == "thalweg":
        if rebuild_mode in {"channel_core_generic_rebuild", "thalweg_only_generic_rebuild"}:
            return float(np.clip(max(station_weight, 0.72), 0.0, 0.98))
        return float(np.clip(max(station_weight, 0.55), 0.0, 0.95))
    if role in {"left_inner", "right_inner"}:
        if rebuild_mode in {"thalweg_only_rebuild", "thalweg_only_generic_rebuild"}:
            return 0.0
        if rebuild_mode == "channel_core_generic_rebuild":
            base = float(np.clip(max(0.88 * station_weight, 0.48), 0.0, 0.94))
        else:
            base = float(np.clip(max(0.74 * station_weight, 0.30), 0.0, 0.90))
        return float(np.clip(base * float(np.clip(inner_weight_scale, 0.0, 1.0)), 0.0, 0.90))
    if role in BANK_ROLES:
        if bank_protected:
            return 0.0
        if rebuild_mode == "full_section_rebuild":
            return float(np.clip(0.18 * station_weight, 0.0, 0.20))
        return 0.0
    return 0.0




def _station_rebuild_mode(row: pd.Series, available_roles: set[str], *, station_authoritative: bool, station_measured_xs: bool, station_channel_protected: bool = False, station_inner_rebuildable: bool = True) -> str:
    semantics = station_semantics(
        row,
        station_authoritative=station_authoritative,
        station_measured_xs=station_measured_xs,
    )
    if str(semantics.get("station_rebuild_regime", "eligible")) != "eligible":
        return "skip"
    envelope_class = str(semantics.get("station_measured_support_envelope_class", row.get("station_measured_support_envelope_class", "no_measured_support_anywhere")) or "no_measured_support_anywhere")
    support_regime = str(semantics.get("station_support_regime", row.get("station_support_regime", "missing")) or "missing")
    protection_regime = str(semantics.get("station_protection_regime", row.get("station_protection_regime", "unprotected")) or "unprotected")
    has_thalweg = "thalweg" in available_roles
    has_inner = bool({"left_inner", "right_inner"} & set(available_roles))
    has_banks = bool({"left_bank", "right_bank"} & set(available_roles))
    if not has_thalweg:
        return "skip"
    if protection_regime == "bank_only_protected" or support_regime == "bank_only_authoritative":
        if has_inner:
            return "channel_core_generic_rebuild"
        if has_banks:
            return "thalweg_only_generic_rebuild"
        return "skip"
    if envelope_class == "unsupported_upstream_generic":
        if has_inner:
            return "channel_core_generic_rebuild"
        if has_banks:
            return "thalweg_only_generic_rebuild"
    if envelope_class in {"weak_support_upstream", "bank_margin_only"}:
        if has_inner:
            return "channel_core_generic_rebuild"
        return "thalweg_only_generic_rebuild" if has_banks else "skip"
    if has_inner and has_banks:
        return "full_section_rebuild"
    if has_inner:
        return "inner_nodes_only_rebuild"
    return "thalweg_only_rebuild"

def _node_rebuild_regime(role_name: str, *, rebuild_mode: str, node_is_authoritative: bool, node_is_measured: bool) -> str:
    role = str(role_name or "")
    if node_is_measured:
        return "protected_true_measured"
    if node_is_authoritative and role in {"thalweg", "left_inner", "right_inner", "left_bank", "right_bank"}:
        return "protected_authoritative"
    if rebuild_mode == "skip":
        return "skip"
    if role == "thalweg":
        return "rebuild_channel_core"
    if role in {"left_inner", "right_inner"}:
        return "rebuild_inner_generic" if rebuild_mode == "channel_core_generic_rebuild" else "rebuild_inner"
    if role in BANK_ROLES:
        return "preserve_bank"
    return "skip"


def _pava_nonincreasing(values: np.ndarray, *, weights: np.ndarray | None = None) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    n = vals.size
    if n <= 1:
        return vals.copy()
    work = -vals.copy()
    means = work.copy()
    if weights is None:
        wts = np.ones(n, dtype=float)
    else:
        wts = np.asarray(weights, dtype=float)
        if wts.shape != work.shape:
            raise ValueError("pava_weights_shape_mismatch")
        wts = np.where(np.isfinite(wts) & (wts > 0.0), wts, 1.0)
    starts = np.arange(n, dtype=int)
    ends = np.arange(n, dtype=int)
    m = 0
    for i in range(n):
        starts[m] = i
        ends[m] = i
        means[m] = work[i]
        wts[m] = wts[i]
        while m > 0 and means[m - 1] > means[m]:
            tot_w = wts[m - 1] + wts[m]
            means[m - 1] = ((means[m - 1] * wts[m - 1]) + (means[m] * wts[m])) / tot_w
            wts[m - 1] = tot_w
            ends[m - 1] = ends[m]
            m -= 1
        m += 1
    out = np.empty(n, dtype=float)
    for b in range(m):
        out[starts[b]: ends[b] + 1] = means[b]
    return -out


def _apply_component_channel_core_monotone_projection(component_work: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    if component_work.empty:
        return component_work, {
            "fluvial_station_count": 0,
            "pre_monotone_violation_count": 0,
            "post_monotone_violation_count": 0,
            "adjusted_node_count": 0,
            "hard_anchor_node_count": 0,
            "hard_anchor_moved_count": 0,
            "projection_delta_abs_m": _distribution(np.array([], dtype=float)),
        }, pd.DataFrame()
    work = component_work.copy()
    work["post_rebuild_monotone_applied"] = False
    work["post_rebuild_monotone_delta_m"] = np.float32(0.0)
    work["post_rebuild_monotone_target_z_m"] = np.float32(np.nan)
    work["post_rebuild_monotone_weight"] = np.float32(0.0)
    work["support_class_canonical"] = work.apply(canonical_river_support_class, axis=1).astype(str)
    work["active_interior_target_source"] = "missing"
    work["active_interior_target_z_m"] = np.float32(np.nan)
    work["active_interior_target_reason"] = "missing"
    work["diagnostic_target_source"] = "missing"
    work["active_interior_target_degraded"] = False
    work["primary_surface_contract_mode"] = "canonical_v314"
    role_frames: list[pd.DataFrame] = []
    pre_viol = 0
    post_viol = 0
    adjusted_nodes = 0
    hard_anchor_nodes = 0
    hard_anchor_moved = 0
    deltas: list[float] = []

    for role_name in sorted(CHANNEL_CORE_ROLES):
        role_mask = work["node_role"].astype(str).eq(role_name)
        if not bool(role_mask.any()):
            continue
        role_df = work.loc[role_mask].copy().sort_values("station_m")
        if role_df.empty:
            continue
        fluvial = pd.to_numeric(role_df.get("profile_inside_fluvial_monotone_domain", pd.Series([True] * len(role_df), index=role_df.index)), errors="coerce").fillna(1.0).to_numpy(dtype=float) >= 0.5
        base = pd.to_numeric(role_df.get("bed_z_m"), errors="coerce").to_numpy(dtype=float)
        hard_anchor = (_is_authoritative_like(role_df) | _is_measured_xs(role_df)) & np.isfinite(base)
        hard_anchor_nodes += int(np.count_nonzero(hard_anchor))
        weights = np.ones(len(role_df), dtype=float)
        base_w = pd.to_numeric(role_df.get("primary_surface_rebuild_weight", 0.0), errors="coerce").to_numpy(dtype=float)
        weights *= 1.0 + np.clip(np.nan_to_num(base_w, nan=0.0), 0.0, 1.0)
        if role_name == "thalweg":
            guard_w = pd.to_numeric(role_df.get("primary_surface_backbone_guardrail_weight", 0.0), errors="coerce").to_numpy(dtype=float)
            weights *= 1.0 + np.clip(np.nan_to_num(guard_w, nan=0.0), 0.0, 12.0)
        support_dist = pd.to_numeric(role_df.get("station_authoritative_bed_support_distance_m", pd.Series([np.nan] * len(role_df), index=role_df.index)), errors="coerce").to_numpy(dtype=float)
        near = np.isfinite(support_dist)
        weights = np.where(near, weights * np.clip(1.6 - 0.01 * np.nan_to_num(support_dist, nan=50.0), 0.25, 1.6), weights)
        far_support = near & (support_dist >= 500.0) & (~hard_anchor)
        weights = np.where(far_support, weights * np.clip(1.0 - 0.0006 * np.nan_to_num(support_dist, nan=500.0), 0.15, 0.70), weights)
        weights = np.where(hard_anchor, 1.0e9, weights)
        solved = base.copy()
        role_pre = 0
        role_post = 0
        for start_end in [(None, None)]:
            run_start = None
            flags = list(fluvial) + [False]
            for i, active in enumerate(flags):
                if active and run_start is None:
                    run_start = i
                elif (not active) and (run_start is not None):
                    seg = slice(run_start, i)
                    seg_vals = base[seg].copy()
                    seg_finite = np.isfinite(seg_vals)
                    if np.count_nonzero(seg_finite) >= 2:
                        seg_fill = seg_vals.copy()
                        missing = ~seg_finite
                        if np.any(missing):
                            seg_fill[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(seg_finite), seg_vals[seg_finite])
                        seg_w = weights[seg].copy()
                        seg_hard = hard_anchor[seg].copy()
                        seg_hard_vals = seg_vals.copy()
                        projected = _pava_nonincreasing(seg_fill, weights=seg_w)
                        if np.any(seg_hard):
                            projected = np.where(seg_hard, seg_hard_vals, projected)
                            projected = _pava_nonincreasing(projected, weights=np.where(seg_hard, 1.0e9, seg_w))
                            projected = np.where(seg_hard, seg_hard_vals, projected)
                        solved[seg] = projected
                        ordered_base = seg_fill[np.isfinite(seg_fill)]
                        ordered_solved = projected[np.isfinite(projected)]
                        if ordered_base.size >= 2:
                            role_pre += int(np.count_nonzero(np.diff(ordered_base) > 0.0))
                        if ordered_solved.size >= 2:
                            role_post += int(np.count_nonzero(np.diff(ordered_solved) > 1.0e-6))
                    run_start = None
        delta = solved - base
        applied = np.isfinite(delta) & (np.abs(delta) > 1.0e-6) & fluvial
        adjusted_nodes += int(np.count_nonzero(applied))
        deltas.extend(np.abs(delta[applied]).tolist())
        hard_anchor_moved += int(np.count_nonzero(applied & hard_anchor))
        pre_viol += role_pre
        post_viol += role_post
        work.loc[role_df.index, "bed_z_m"] = np.where(np.isfinite(solved), solved, pd.to_numeric(work.loc[role_df.index, "bed_z_m"], errors="coerce").to_numpy(dtype=float)).astype(np.float32)
        work.loc[role_df.index, "post_rebuild_monotone_applied"] = applied
        work.loc[role_df.index, "post_rebuild_monotone_delta_m"] = np.nan_to_num(delta, nan=0.0).astype(np.float32)
        work.loc[role_df.index, "post_rebuild_monotone_target_z_m"] = np.where(np.isfinite(solved), solved, np.nan).astype(np.float32)
        work.loc[role_df.index, "post_rebuild_monotone_weight"] = np.where(applied, np.clip(np.nan_to_num(weights, nan=1.0), 0.0, 1.0e9), 0.0).astype(np.float32)
        role_out = role_df[["component_id", "station_m", "node_role"]].copy()
        role_out["fluvial_monotone_domain"] = fluvial
        role_out["hard_anchor"] = hard_anchor
        role_out["pre_projection_z_m"] = base
        role_out["post_projection_z_m"] = solved
        role_out["projection_delta_m"] = delta
        role_out["projection_applied"] = applied
        role_frames.append(role_out)

    summary = {
        "fluvial_station_count": int(np.count_nonzero(pd.to_numeric(work.get("profile_inside_fluvial_monotone_domain", pd.Series([True] * len(work), index=work.index)), errors="coerce").fillna(1.0).to_numpy(dtype=float) >= 0.5)),
        "pre_monotone_violation_count": int(pre_viol),
        "post_monotone_violation_count": int(post_viol),
        "adjusted_node_count": int(adjusted_nodes),
        "hard_anchor_node_count": int(hard_anchor_nodes),
        "hard_anchor_moved_count": int(hard_anchor_moved),
        "projection_delta_abs_m": _distribution(np.asarray(deltas, dtype=float)),
    }
    profile = pd.concat(role_frames, ignore_index=True) if role_frames else pd.DataFrame(columns=["component_id", "station_m", "node_role", "fluvial_monotone_domain", "hard_anchor", "pre_projection_z_m", "post_projection_z_m", "projection_delta_m", "projection_applied"])
    return work, summary, profile


def apply_primary_surface_rebuild_to_nodes(
    nodes: pd.DataFrame,
    *,
    river_dir: str | Path,
    disabled: bool = False,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, Dict[str, str], Dict[str, Any]]:
    active_logger = logger or log
    river_dir = Path(river_dir)
    work = nodes.copy()
    if work.empty:
        return work, {}, {"available": False, "component_count": 0, "adjusted_node_count": 0}

    for col in [
        "station_m", "bed_z_m", "prediction_support_confidence", "profile_network_backbone_z_m",
        "graph_backbone_z_m", "unsupported_fraction", "xs_support_fraction", "authoritative_anchor_fraction",
        "xs_realism_left_ratio_after", "xs_realism_right_ratio_after",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "xs_template_type" not in work.columns:
        work["xs_template_type"] = "measured"
    else:
        work["xs_template_type"] = work["xs_template_type"].fillna("measured").astype(str)
    if "xs_support_template_class" not in work.columns:
        work["xs_support_template_class"] = "missing"
    else:
        work["xs_support_template_class"] = work["xs_support_template_class"].fillna("missing").astype(str)

    component_receipts = []
    profile_rows = []
    guardrail_rows = []
    monotone_profile_rows: list[pd.DataFrame] = []
    work["primary_surface_rebuild_applied"] = False
    work["primary_surface_rebuild_delta_m"] = np.float32(0.0)
    work["primary_surface_rebuild_weight"] = np.float32(0.0)
    work["primary_surface_rebuild_target_z_m"] = np.float32(np.nan)
    work["primary_surface_rebuild_mode"] = "skip"
    work["primary_surface_rebuild_skip_reason"] = "unprocessed"
    work["primary_surface_rebuild_node_regime"] = "skip"
    work["primary_surface_rebuild_target_source"] = "none"
    work["primary_surface_rebuild_role_weight"] = np.float32(0.0)
    work["primary_surface_backbone_guardrail_applied"] = False
    work["primary_surface_backbone_guardrail_delta_m"] = np.float32(0.0)
    work["primary_surface_backbone_guardrail_reference_z_m"] = np.float32(np.nan)
    work["primary_surface_backbone_guardrail_tolerance_m"] = np.float32(np.nan)
    work["primary_surface_backbone_guardrail_weight"] = np.float32(0.0)
    work["primary_surface_backbone_guardrail_component_class"] = "unknown"
    work["primary_surface_backbone_led_inner_targets"] = False
    work["primary_surface_backbone_led_inner_relief_scale"] = np.float32(1.0)
    work["primary_surface_backbone_led_inner_reason"] = "structured_targets_allowed"
    work["primary_surface_bank_margin_damped"] = False
    work["post_rebuild_monotone_applied"] = False
    work["post_rebuild_monotone_delta_m"] = np.float32(0.0)
    work["post_rebuild_monotone_target_z_m"] = np.float32(np.nan)
    work["post_rebuild_monotone_weight"] = np.float32(0.0)
    work["support_class_canonical"] = work.apply(canonical_river_support_class, axis=1).astype(str)
    work["active_interior_target_source"] = "missing"
    work["active_interior_target_z_m"] = np.float32(np.nan)
    work["active_interior_target_reason"] = "missing"
    work["active_interior_target_degraded"] = False
    work["primary_surface_contract_mode"] = "canonical_v314"

    if disabled:
        profile_csv = river_dir / "river_primary_surface_rebuild_profile.csv"
        summary_path = river_dir / "river_primary_surface_rebuild_summary.json"
        monotone_profile_csv = river_dir / "river_post_rebuild_monotone_projection_profile.csv"
        monotone_summary_path = river_dir / "river_post_rebuild_monotone_projection_summary.json"
        empty_profile = pd.DataFrame(columns=[
            "component_id", "station_m", "xs_template_type", "xs_support_template_class", "station_support_regime", "station_authoritative",
            "station_measured_xs", "station_channel_protected", "station_inner_rebuildable", "rebuild_weight",
            "backbone_target_z_m", "fitted_core_target_z_m", "primary_surface_rebuild_target_source", "core_geometry_scale",
            "inner_weight_scale", "backbone_led_inner_targets", "backbone_led_inner_relief_scale", "backbone_led_inner_reason",
            "bank_margin_damping_active", "station_adjusted", "station_delta_abs_m", "rebuild_mode", "rebuild_skip_reason",
            "available_roles", "station_changed_node_count", "station_changed_channel_core_node_count",
            "component_support_class", "support_class_canonical", "active_interior_target_source", "active_interior_target_z_m", "active_interior_target_degraded", "backbone_guardrail_reference_z_m", "backbone_guardrail_tolerance_m",
            "backbone_guardrail_weight", "backbone_guardrail_applied", "backbone_guardrail_delta_m",
        ])
        empty_profile.to_csv(profile_csv, index=False)
        guardrail_profile_csv = river_dir / "river_primary_surface_backbone_guardrail_profile.csv"
        pd.DataFrame(columns=[
            "component_id", "station_m", "component_support_class", "station_support_regime",
            "backbone_reference_z_m", "tolerance_m", "guardrail_weight", "pre_guardrail_thalweg_z_m",
            "post_guardrail_thalweg_z_m", "guardrail_delta_m", "guardrail_applied",
        ]).to_csv(guardrail_profile_csv, index=False)
        work.loc[work["node_role"].astype(str).isin(list(CHANNEL_CORE_ROLES)), [
            "component_id", "station_m", "node_role", "bed_z_m", "post_rebuild_monotone_applied",
            "post_rebuild_monotone_delta_m", "post_rebuild_monotone_target_z_m", "post_rebuild_monotone_weight"
        ]].sort_values(["component_id", "station_m", "node_role"]).to_csv(monotone_profile_csv, index=False)
        summary = {
            "available": False,
            "disabled_by_option": True,
            "component_count": 0,
            "adjusted_node_count": 0,
            "changed_channel_core_node_count": 0,
            "rebuild_mode_counts": {},
            "target_source_counts": {},
            "canonical_support_class_counts": {},
            "active_interior_target_source_counts": {},
            "degraded_active_target_station_count": 0,
            "delta_abs_m": _distribution(np.array([], dtype=float)),
            "blend_weight_summary": _distribution(np.array([], dtype=float)),
            "core_geometry_scale_summary": _distribution(np.array([], dtype=float)),
            "inner_weight_scale_summary": _distribution(np.array([], dtype=float)),
            "backbone_guardrail_adjusted_node_count": 0,
            "backbone_guardrail_delta_abs_m": _distribution(np.array([], dtype=float)),
            "backbone_guardrail_by_component_class": {},
            "post_rebuild_monotone_available": False,
            "post_rebuild_monotone_adjusted_node_count": 0,
            "post_rebuild_monotone_delta_abs_m": _distribution(np.array([], dtype=float)),
            "post_rebuild_monotone_pre_violation_count": 0,
            "post_rebuild_monotone_post_violation_count": 0,
            "components": [],
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        guardrail_summary_path = river_dir / "river_primary_surface_backbone_guardrail_summary.json"
        guardrail_summary_path.write_text(json.dumps({
            "available": False,
            "disabled_by_option": True,
            "adjusted_node_count": 0,
            "delta_abs_m": _distribution(np.array([], dtype=float)),
            "by_component_class": {},
        }, indent=2), encoding="utf-8")
        monotone_summary_path.write_text(json.dumps({
            "available": False,
            "disabled_by_option": True,
            "adjusted_node_count": 0,
            "pre_violation_count": 0,
            "post_violation_count": 0,
            "delta_abs_m": _distribution(np.array([], dtype=float)),
            "components": [],
        }, indent=2), encoding="utf-8")
        active_logger.info("[RIVER][PRIMARY] Primary surface rebuild disabled by option; keeping longitudinal/core scaffold solution unchanged")
        outputs = {
            "primary_surface_rebuild_profile": str(profile_csv),
            "primary_surface_rebuild_summary": str(summary_path),
            "primary_surface_backbone_guardrail_profile": str(guardrail_profile_csv),
            "primary_surface_backbone_guardrail_summary": str(guardrail_summary_path),
            "post_rebuild_monotone_projection_profile": str(monotone_profile_csv),
            "post_rebuild_monotone_projection_summary": str(monotone_summary_path),
        }
        return work, outputs, summary

    for component_id, comp_nodes in work.groupby("component_id", sort=False):
        roles = set(comp_nodes["node_role"].astype(str).tolist())
        if "thalweg" not in roles:
            continue
        station_records = []
        stations_seen = []
        for station_m, station_group in comp_nodes.groupby("station_m", sort=True):
            roles_map = {str(r): idx for idx, r in zip(station_group.index, station_group["node_role"].astype(str))}
            available_roles = sorted(roles_map.keys())
            if "thalweg" not in roles_map:
                continue
            left_bank = float(pd.to_numeric(work.at[roles_map["left_bank"], "bed_z_m"], errors="coerce")) if "left_bank" in roles_map else np.nan
            thalweg = float(pd.to_numeric(work.at[roles_map["thalweg"], "bed_z_m"], errors="coerce"))
            right_bank = float(pd.to_numeric(work.at[roles_map["right_bank"], "bed_z_m"], errors="coerce")) if "right_bank" in roles_map else np.nan
            if not np.isfinite(thalweg):
                continue
            station_authoritative = bool(np.any(_is_authoritative_like(station_group)))
            station_measured_xs = bool(np.any(_is_measured_xs(station_group)))
            station_channel_protected = bool(pd.to_numeric(station_group.get("station_channel_protected", pd.Series([False] * len(station_group))), errors="coerce").fillna(False).astype(bool).any())
            station_inner_rebuildable = bool(pd.to_numeric(station_group.get("station_inner_rebuildable", pd.Series([True] * len(station_group))), errors="coerce").fillna(False).astype(bool).any())
            row0 = station_group.iloc[0]
            current_thalweg = float(pd.to_numeric(work.at[roles_map["thalweg"], "bed_z_m"], errors="coerce"))
            backbone_target = np.nan
            if _station_prefers_backbone_bed_reference(row0, current_thalweg_z=current_thalweg):
                backbone_target = pd.to_numeric(row0.get("backbone_bed_z_m"), errors="coerce")
            if not np.isfinite(backbone_target):
                backbone_target = pd.to_numeric(row0.get("profile_network_backbone_z_m"), errors="coerce")
            if not np.isfinite(backbone_target):
                backbone_target = pd.to_numeric(row0.get("graph_backbone_z_m"), errors="coerce")
            station_records.append({
                "station_m": float(station_m),
                "left_bank_z_m": left_bank,
                "thalweg_z_m": thalweg,
                "right_bank_z_m": right_bank,
                "available_roles": available_roles,
                "station_authoritative": station_authoritative,
                "station_measured_xs": station_measured_xs,
                "station_channel_protected": station_channel_protected,
                "station_inner_rebuildable": station_inner_rebuildable,
                "xs_template_type": str(row0.get("xs_template_type", "measured") or "measured"),
                "xs_support_template_class": str(row0.get("xs_support_template_class", "missing") or "missing"),
                "prediction_support_confidence": pd.to_numeric(row0.get("prediction_support_confidence"), errors="coerce"),
                "unsupported_fraction": pd.to_numeric(row0.get("unsupported_fraction"), errors="coerce"),
                "xs_support_fraction": pd.to_numeric(row0.get("xs_support_fraction"), errors="coerce"),
                "authoritative_anchor_fraction": pd.to_numeric(row0.get("authoritative_anchor_fraction"), errors="coerce"),
                "station_true_measured_xs_fraction": pd.to_numeric(row0.get("station_true_measured_xs_fraction"), errors="coerce"),
                "station_indirect_xs_fraction": pd.to_numeric(row0.get("station_indirect_xs_fraction"), errors="coerce"),
                "station_residual_xs_fraction": pd.to_numeric(row0.get("station_residual_xs_fraction"), errors="coerce"),
                "station_authoritative_channel_fraction": pd.to_numeric(row0.get("station_authoritative_channel_fraction"), errors="coerce"),
                "station_authoritative_bank_fraction": pd.to_numeric(row0.get("station_authoritative_bank_fraction"), errors="coerce"),
                "station_authoritative_bed_core_fraction": pd.to_numeric(row0.get("station_authoritative_bed_core_fraction"), errors="coerce"),
                "station_support_regime": str(row0.get("station_support_regime", "missing") or "missing"),
                "station_protection_regime": str(row0.get("station_protection_regime", "unprotected") or "unprotected"),
                "station_rebuild_regime": str(row0.get("station_rebuild_regime", "blocked_not_weak_support") or "blocked_not_weak_support"),
                "station_authoritative_bed_support_class": str(row0.get("station_authoritative_bed_support_class", "no_authoritative_bed_support") or "no_authoritative_bed_support"),
                "station_authoritative_bed_support_fraction": pd.to_numeric(row0.get("station_authoritative_bed_support_fraction"), errors="coerce"),
                "station_authoritative_bank_margin_fraction": pd.to_numeric(row0.get("station_authoritative_bank_margin_fraction"), errors="coerce"),
                "station_authoritative_bed_support_present": bool(row0.get("station_authoritative_bed_support_present", False)),
                "station_authoritative_bed_support_distance_m": pd.to_numeric(row0.get("station_authoritative_bed_support_distance_m"), errors="coerce"),
                "component_support_class": str(row0.get("component_support_class", "unknown") or "unknown"),
                "profile_inside_fluvial_monotone_domain": bool(pd.to_numeric(row0.get("profile_inside_fluvial_monotone_domain", True), errors="coerce") >= 0.5),
                "backbone_bed_reference_z_m": pd.to_numeric(row0.get("backbone_bed_z_m"), errors="coerce"),
                "backbone_target_z_m": backbone_target,
                "active_core_fit_z_m": pd.to_numeric(row0.get("active_core_fit_z_m"), errors="coerce"),
                "left_bank_fit_z_m": pd.to_numeric(row0.get("left_bank_fit_z_m"), errors="coerce"),
                "right_bank_fit_z_m": pd.to_numeric(row0.get("right_bank_fit_z_m"), errors="coerce"),
                "bank_pair_fit_z_m": pd.to_numeric(row0.get("bank_pair_fit_z_m"), errors="coerce"),
                "station_target_present": bool(row0.get("station_target_present", False)),
                "station_target_source_class": str(row0.get("station_target_source_class", "missing") or "missing"),
                "station_target_policy_reason": str(row0.get("station_target_policy_reason", "missing") or "missing"),
                "target_xs_realism_allowed": bool(row0.get("target_xs_realism_allowed", True)),
                "target_rebuild_allowed": bool(row0.get("target_rebuild_allowed", True)),
                "target_anchor_class": str(row0.get("target_anchor_class", "non_anchor") or "non_anchor"),
                "target_anchor_exact": bool(row0.get("target_anchor_exact", False)),
                "target_anchor_locks_core": bool(row0.get("target_anchor_locks_core", False)),
                "target_anchor_blocks_rebuild": bool(row0.get("target_anchor_blocks_rebuild", False)),
                "target_left_bank_z_m": pd.to_numeric(row0.get("target_left_bank_z_m"), errors="coerce"),
                "target_right_bank_z_m": pd.to_numeric(row0.get("target_right_bank_z_m"), errors="coerce"),
                "target_left_inner_z_m": pd.to_numeric(row0.get("target_left_inner_z_m"), errors="coerce"),
                "target_right_inner_z_m": pd.to_numeric(row0.get("target_right_inner_z_m"), errors="coerce"),
                "target_thalweg_z_m": pd.to_numeric(row0.get("target_thalweg_z_m"), errors="coerce"),
                "left_ratio": pd.to_numeric(row0.get("xs_realism_left_ratio_after"), errors="coerce"),
                "right_ratio": pd.to_numeric(row0.get("xs_realism_right_ratio_after"), errors="coerce"),
            })
            stations_seen.append(float(station_m))
        if not station_records:
            continue
        station_df = pd.DataFrame(station_records).sort_values("station_m").reset_index(drop=True)
        station_df, backbone_smoothing_profile_df, backbone_smoothing_summary = _apply_support_aware_backbone_smoothing(station_df)
        adjusted_station_count = 0
        station_delta_abs = []
        for _, row in station_df.iterrows():
            station_m = float(row["station_m"])
            station_mask = comp_nodes["station_m"].to_numpy(dtype=float) == station_m
            station_group = comp_nodes.loc[station_mask].copy()
            station_authoritative = bool(row["station_authoritative"])
            station_measured_xs = bool(row["station_measured_xs"])
            station_channel_protected = bool(row.get("station_channel_protected", False))
            station_inner_rebuildable = bool(row.get("station_inner_rebuildable", True))
            rebuild_weight = _station_rebuild_weight(row, station_authoritative=station_authoritative, station_measured_xs=station_measured_xs, station_channel_protected=station_channel_protected, station_inner_rebuildable=station_inner_rebuildable)
            if rebuild_weight <= 0.0:
                station_delta_abs.append(0.0)
                continue
            roles_map = {str(r): idx for idx, r in zip(station_group.index, station_group["node_role"].astype(str))}
            available_roles = set(roles_map.keys())
            rebuild_mode = _station_rebuild_mode(row, available_roles, station_authoritative=station_authoritative, station_measured_xs=station_measured_xs, station_channel_protected=station_channel_protected, station_inner_rebuildable=station_inner_rebuildable)
            th_idx = roles_map.get("thalweg")
            if th_idx is None or rebuild_mode == "skip":
                station_delta_abs.append(0.0)
                for idx in station_group.index:
                    work.at[idx, "primary_surface_rebuild_mode"] = "skip"
                    work.at[idx, "primary_surface_rebuild_skip_reason"] = "missing_roles_or_ineligible"
                continue
            current_th = float(pd.to_numeric(work.at[th_idx, "bed_z_m"], errors="coerce"))
            backbone_target = float(pd.to_numeric(row.get("smoothed_backbone_target_z_m", row.get("backbone_target_z_m")), errors="coerce"))
            fitted_core_target, fitted_left_bank, fitted_right_bank, target_source = _station_fitted_section_targets(row, station_group)
            if np.isfinite(fitted_core_target) and is_generalized_rebuild_target_source(target_source):
                target_th = (1.0 - rebuild_weight) * current_th + rebuild_weight * fitted_core_target
            elif np.isfinite(backbone_target):
                target_th = (1.0 - rebuild_weight) * current_th + rebuild_weight * backbone_target
            else:
                target_th = current_th
            left_bank = float(pd.to_numeric(work.at[roles_map["left_bank"], "bed_z_m"], errors="coerce")) if "left_bank" in roles_map else np.nan
            right_bank = float(pd.to_numeric(work.at[roles_map["right_bank"], "bed_z_m"], errors="coerce")) if "right_bank" in roles_map else np.nan
            if is_generalized_rebuild_target_source(target_source):
                if np.isfinite(fitted_left_bank):
                    left_bank = fitted_left_bank
                if np.isfinite(fitted_right_bank):
                    right_bank = fitted_right_bank
            if not np.isfinite(left_bank) and np.isfinite(right_bank):
                left_bank = right_bank
            if not np.isfinite(right_bank) and np.isfinite(left_bank):
                right_bank = left_bank
            if not np.isfinite(left_bank) and not np.isfinite(right_bank):
                station_delta_abs.append(0.0)
                for idx in station_group.index:
                    work.at[idx, "primary_surface_rebuild_mode"] = "skip"
                    work.at[idx, "primary_surface_rebuild_skip_reason"] = "missing_banks"
                continue
            left_relief = max(left_bank - target_th, 0.05)
            right_relief = max(right_bank - target_th, 0.05)
            # Symmetrize bank relief in low-support generic reaches: smooth/generic beats noisy/wrong.
            if str(row.get("xs_template_type", "measured")) in {"generic_symmetric", "generic_u", "hybrid_longitudinal"}:
                common_relief = float(np.nanmedian([left_relief, right_relief]))
                common_relief = float(max(common_relief, 0.10))
                left_relief = (1.0 - 0.65 * rebuild_weight) * left_relief + (0.65 * rebuild_weight) * common_relief
                right_relief = (1.0 - 0.65 * rebuild_weight) * right_relief + (0.65 * rebuild_weight) * common_relief
                left_bank_target = target_th + left_relief
                right_bank_target = target_th + right_relief
            else:
                left_bank_target = left_bank
                right_bank_target = right_bank
            left_ratio, right_ratio = _station_template_ratios(row)
            core_geometry_scale, inner_weight_scale = _station_core_geometry_controls(row)
            backbone_led_inner_targets, backbone_led_inner_relief_scale, backbone_led_inner_reason = _station_backbone_led_inner_target_scale(row)
            bank_margin_damping_active = bool(backbone_led_inner_targets and (not _station_has_real_authoritative_interior_bed_support(row)))
            support_class_canonical = canonical_river_support_class(row)
            active_interior_target_z = float(target_th) if np.isfinite(target_th) else (
                float(fitted_core_target) if np.isfinite(fitted_core_target) else (
                    float(backbone_target) if np.isfinite(backbone_target) else (float(current_thalweg) if np.isfinite(current_thalweg) else np.nan)
                )
            )
            if _station_has_real_authoritative_interior_bed_support(row) or station_measured_xs:
                active_interior_target_source = ACTIVE_INTERIOR_TARGET_AUTHORITATIVE
                active_interior_target_reason = ACTIVE_INTERIOR_TARGET_AUTHORITATIVE
            elif np.isfinite(active_interior_target_z):
                active_interior_target_source = ACTIVE_INTERIOR_TARGET_BACKBONE
                active_interior_target_reason = ACTIVE_INTERIOR_TARGET_BACKBONE
            else:
                active_interior_target_source = "missing"
                active_interior_target_reason = "missing_canonical_target"
            diagnostic_target_source = str(target_source or "missing")
            backbone_candidate_group = str(row.get("backbone_candidate_group", "other_excluded") or "other_excluded")
            backbone_action_applied = bool(row.get("backbone_smoothing_applied", False))
            active_interior_target_degraded = bool(support_class_canonical == "unsupported_interior" and not np.isfinite(active_interior_target_z))
            if (
                support_class_canonical == "unsupported_interior"
                and backbone_candidate_group == "unsupported_mainstem"
                and not backbone_action_applied
            ):
                active_interior_target_degraded = True
                if active_interior_target_source == ACTIVE_INTERIOR_TARGET_BACKBONE and np.isfinite(active_interior_target_z):
                    active_interior_target_reason = "backbone_inert_unsupported_mainstem"
            allowed_mode = rebuild_mode
            if allowed_mode == "full_section_rebuild" and not {"left_inner", "right_inner", "left_bank", "right_bank", "thalweg"}.issubset(available_roles):
                allowed_mode = "skip"
            elif allowed_mode == "inner_nodes_only_rebuild" and not ("thalweg" in available_roles and ({"left_inner", "right_inner"} & available_roles)):
                allowed_mode = "skip"
            elif allowed_mode in {"thalweg_only_rebuild", "thalweg_only_generic_rebuild"} and not {"left_bank", "right_bank", "thalweg"}.issubset(available_roles):
                allowed_mode = "skip"
            elif allowed_mode == "channel_core_generic_rebuild" and not ("thalweg" in available_roles and ({"left_inner", "right_inner"} & available_roles)):
                allowed_mode = "thalweg_only_generic_rebuild" if {"left_bank", "right_bank", "thalweg"}.issubset(available_roles) else "skip"

            rebuild_mode = allowed_mode

            if rebuild_mode == "skip":
                station_delta_abs.append(0.0)
                for idx in station_group.index:
                    work.at[idx, "primary_surface_rebuild_mode"] = "skip"
                    work.at[idx, "primary_surface_rebuild_skip_reason"] = "insufficient_roles"
                    work.at[idx, "support_class_canonical"] = str(support_class_canonical)
                    work.at[idx, "active_interior_target_source"] = str(active_interior_target_source)
                    work.at[idx, "active_interior_target_z_m"] = np.float32(active_interior_target_z) if np.isfinite(active_interior_target_z) else np.float32(np.nan)
                    work.at[idx, "active_interior_target_reason"] = str(active_interior_target_reason)
                    work.at[idx, "diagnostic_target_source"] = diagnostic_target_source
                    work.at[idx, "active_interior_target_degraded"] = bool(active_interior_target_degraded)
                profile_rows.append({
                    "component_id": str(component_id),
                    "station_m": station_m,
                    "xs_template_type": str(row.get("xs_template_type", "measured")),
                    "xs_support_template_class": str(row.get("xs_support_template_class", "missing")),
                    "station_support_regime": str(row.get("station_support_regime", "missing")),
                    "station_authoritative": station_authoritative,
                    "station_measured_xs": station_measured_xs,
                    "station_channel_protected": station_channel_protected,
                    "station_inner_rebuildable": station_inner_rebuildable,
                    "rebuild_weight": rebuild_weight,
                    "backbone_target_z_m": backbone_target,
                    "fitted_core_target_z_m": fitted_core_target,
                    "primary_surface_rebuild_target_source": target_source,
                "support_class_canonical": support_class_canonical,
                "active_interior_target_source": active_interior_target_source,
                "active_interior_target_z_m": active_interior_target_z,
                "active_interior_target_degraded": active_interior_target_degraded,
                    "core_geometry_scale": core_geometry_scale,
                    "inner_weight_scale": inner_weight_scale,
                    "backbone_led_inner_targets": backbone_led_inner_targets,
                    "backbone_led_inner_relief_scale": backbone_led_inner_relief_scale,
                    "backbone_led_inner_reason": backbone_led_inner_reason,
                    "bank_margin_damping_active": bank_margin_damping_active,
                    "station_adjusted": False,
                    "station_delta_abs_m": 0.0,
                    "rebuild_mode": rebuild_mode,
                    "rebuild_skip_reason": "insufficient_roles",
                    "available_roles": ",".join(sorted(available_roles)),
                    "station_changed_node_count": 0,
                    "station_changed_channel_core_node_count": 0,
                })
                continue

            node_targets = {"thalweg": target_th}
            if "left_bank" in roles_map:
                node_targets["left_bank"] = left_bank_target
            if "right_bank" in roles_map:
                node_targets["right_bank"] = right_bank_target
            if rebuild_mode in {"full_section_rebuild", "inner_nodes_only_rebuild", "channel_core_generic_rebuild"}:
                tendency_left_inner, tendency_right_inner = _station_inner_targets_from_tendency(
                    row,
                    target_th=target_th,
                    left_bank_target=left_bank_target,
                    right_bank_target=right_bank_target,
                )
                if "left_inner" in roles_map:
                    node_targets["left_inner"] = tendency_left_inner if np.isfinite(tendency_left_inner) else target_th + left_ratio * max(left_bank_target - target_th, 0.05)
                if "right_inner" in roles_map:
                    node_targets["right_inner"] = tendency_right_inner if np.isfinite(tendency_right_inner) else target_th + right_ratio * max(right_bank_target - target_th, 0.05)
            station_max_delta = 0.0
            station_adjusted = False
            channel_core_changed = 0
            station_changed_nodes = 0
            backbone_guardrail_reference = _station_backbone_reference_value(row)
            backbone_guardrail_tolerance = _station_thalweg_backbone_guardrail_tolerance_m(row)
            backbone_guardrail_weight = _station_backbone_guardrail_weight(row, tolerance_m=backbone_guardrail_tolerance)
            backbone_guardrail_applied = False
            backbone_guardrail_delta = 0.0
            pre_guardrail_thalweg = np.nan
            for role_name, target_z in node_targets.items():
                idx = roles_map.get(role_name)
                if idx is None:
                    continue
                node_is_authoritative = bool(_is_authoritative_like(work.loc[[idx]]).item())
                node_is_measured = bool(_is_measured_xs(work.loc[[idx]]).item())
                node_regime = _node_rebuild_regime(
                    role_name,
                    rebuild_mode=rebuild_mode,
                    node_is_authoritative=node_is_authoritative,
                    node_is_measured=node_is_measured,
                )
                role_weight = _node_rebuild_role_weight(
                    role_name,
                    rebuild_mode=rebuild_mode,
                    station_weight=rebuild_weight,
                    bank_protected=bool(node_is_authoritative or role_name in BANK_ROLES),
                    node_is_authoritative=node_is_authoritative,
                    inner_weight_scale=inner_weight_scale,
                )
                work.at[idx, "primary_surface_rebuild_node_regime"] = node_regime
                work.at[idx, "primary_surface_rebuild_target_source"] = target_source
                work.at[idx, "primary_surface_rebuild_role_weight"] = np.float32(role_weight)
                work.at[idx, "primary_surface_backbone_led_inner_targets"] = bool(backbone_led_inner_targets)
                work.at[idx, "primary_surface_backbone_led_inner_relief_scale"] = np.float32(backbone_led_inner_relief_scale)
                work.at[idx, "primary_surface_backbone_led_inner_reason"] = str(backbone_led_inner_reason)
                work.at[idx, "primary_surface_bank_margin_damped"] = bool(bank_margin_damping_active)
                work.at[idx, "support_class_canonical"] = str(support_class_canonical)
                work.at[idx, "active_interior_target_source"] = str(active_interior_target_source)
                work.at[idx, "active_interior_target_z_m"] = np.float32(active_interior_target_z) if np.isfinite(active_interior_target_z) else np.float32(np.nan)
                work.at[idx, "active_interior_target_reason"] = str(active_interior_target_reason)
                work.at[idx, "diagnostic_target_source"] = diagnostic_target_source
                work.at[idx, "active_interior_target_degraded"] = bool(active_interior_target_degraded)
                if node_regime in {"protected_true_measured", "protected_authoritative", "preserve_bank", "skip"} or role_weight <= 0.0:
                    continue
                current_z = float(pd.to_numeric(work.at[idx, "bed_z_m"], errors="coerce"))
                if not np.isfinite(current_z) or not np.isfinite(target_z):
                    continue
                updated = (1.0 - role_weight) * current_z + role_weight * target_z
                delta = float(updated - current_z)
                if abs(delta) > 1.0e-6:
                    station_adjusted = True
                    station_changed_nodes += 1
                    if role_name in CHANNEL_CORE_ROLES:
                        channel_core_changed += 1
                    station_max_delta = max(station_max_delta, abs(delta))
                work.at[idx, "bed_z_m"] = np.float32(updated)
                work.at[idx, "primary_surface_rebuild_applied"] = abs(delta) > 1.0e-6
                work.at[idx, "primary_surface_rebuild_delta_m"] = np.float32(delta)
                work.at[idx, "primary_surface_rebuild_weight"] = np.float32(role_weight)
                work.at[idx, "primary_surface_rebuild_target_z_m"] = np.float32(target_z)
                work.at[idx, "primary_surface_rebuild_mode"] = rebuild_mode
                work.at[idx, "primary_surface_rebuild_skip_reason"] = ""

            if th_idx is not None:
                th_node = work.loc[[th_idx]]
                th_is_authoritative = bool(_is_authoritative_like(th_node).item())
                th_is_measured = bool(_is_measured_xs(th_node).item())
                work.at[th_idx, "primary_surface_backbone_guardrail_reference_z_m"] = np.float32(backbone_guardrail_reference) if np.isfinite(backbone_guardrail_reference) else np.float32(np.nan)
                work.at[th_idx, "primary_surface_backbone_guardrail_tolerance_m"] = np.float32(backbone_guardrail_tolerance) if np.isfinite(backbone_guardrail_tolerance) else np.float32(np.nan)
                work.at[th_idx, "primary_surface_backbone_guardrail_weight"] = np.float32(backbone_guardrail_weight)
                work.at[th_idx, "primary_surface_backbone_guardrail_component_class"] = str(row.get("component_support_class", "unknown") or "unknown")
                if (
                    _station_backbone_guardrail_eligible(row, rebuild_mode=rebuild_mode)
                    and np.isfinite(backbone_guardrail_reference)
                    and np.isfinite(backbone_guardrail_tolerance)
                    and not th_is_authoritative
                    and not th_is_measured
                ):
                    current_thalweg = float(pd.to_numeric(work.at[th_idx, "bed_z_m"], errors="coerce"))
                    pre_guardrail_thalweg = current_thalweg
                    if np.isfinite(current_thalweg):
                        guarded_thalweg = float(np.clip(
                            current_thalweg,
                            backbone_guardrail_reference - backbone_guardrail_tolerance,
                            backbone_guardrail_reference + backbone_guardrail_tolerance,
                        ))
                        backbone_guardrail_delta = float(guarded_thalweg - current_thalweg)
                        if abs(backbone_guardrail_delta) > 1.0e-6:
                            station_adjusted = True
                            station_max_delta = max(station_max_delta, abs(backbone_guardrail_delta))
                            if abs(float(pd.to_numeric(work.at[th_idx, "primary_surface_rebuild_delta_m"], errors="coerce"))) <= 1.0e-6:
                                station_changed_nodes += 1
                                channel_core_changed += 1
                            work.at[th_idx, "bed_z_m"] = np.float32(guarded_thalweg)
                            work.at[th_idx, "primary_surface_backbone_guardrail_applied"] = True
                            work.at[th_idx, "primary_surface_backbone_guardrail_delta_m"] = np.float32(backbone_guardrail_delta)
                            backbone_guardrail_applied = True
                guardrail_rows.append({
                    "component_id": str(component_id),
                    "station_m": station_m,
                    "component_support_class": str(row.get("component_support_class", "unknown") or "unknown"),
                    "station_support_regime": str(row.get("station_support_regime", "missing") or "missing"),
                    "backbone_reference_z_m": backbone_guardrail_reference,
                    "tolerance_m": backbone_guardrail_tolerance,
                    "guardrail_weight": backbone_guardrail_weight,
                    "pre_guardrail_thalweg_z_m": pre_guardrail_thalweg,
                    "post_guardrail_thalweg_z_m": float(pd.to_numeric(work.at[th_idx, "bed_z_m"], errors="coerce")) if th_idx is not None else np.nan,
                    "guardrail_delta_m": backbone_guardrail_delta,
                    "guardrail_applied": backbone_guardrail_applied,
                })
            for idx in station_group.index:
                if str(work.at[idx, "primary_surface_rebuild_mode"]) == "skip":
                    work.at[idx, "primary_surface_rebuild_mode"] = rebuild_mode
                    work.at[idx, "primary_surface_rebuild_skip_reason"] = ""
            profile_rows.append({
                "component_id": str(component_id),
                "station_m": station_m,
                "xs_template_type": str(row.get("xs_template_type", "measured")),
                "xs_support_template_class": str(row.get("xs_support_template_class", "missing")),
                "station_support_regime": str(row.get("station_support_regime", "missing")),
                "station_authoritative": station_authoritative,
                "station_measured_xs": station_measured_xs,
                "station_channel_protected": station_channel_protected,
                "station_inner_rebuildable": station_inner_rebuildable,
                "rebuild_weight": rebuild_weight,
                "backbone_target_z_m": backbone_target,
                "fitted_core_target_z_m": fitted_core_target,
                "primary_surface_rebuild_target_source": target_source,
                "core_geometry_scale": core_geometry_scale,
                "inner_weight_scale": inner_weight_scale,
                "backbone_led_inner_targets": backbone_led_inner_targets,
                "backbone_led_inner_relief_scale": backbone_led_inner_relief_scale,
                "backbone_led_inner_reason": backbone_led_inner_reason,
                "bank_margin_damping_active": bank_margin_damping_active,
                "station_adjusted": station_adjusted,
                "station_delta_abs_m": station_max_delta,
                "rebuild_mode": rebuild_mode,
                "rebuild_skip_reason": "" if rebuild_mode != "skip" else "insufficient_roles",
                "available_roles": ",".join(sorted(available_roles)),
                "station_changed_node_count": station_changed_nodes,
                "station_changed_channel_core_node_count": channel_core_changed,
                "component_support_class": str(row.get("component_support_class", "unknown") or "unknown"),
                "backbone_guardrail_reference_z_m": backbone_guardrail_reference,
                "backbone_guardrail_tolerance_m": backbone_guardrail_tolerance,
                "backbone_guardrail_weight": backbone_guardrail_weight,
                "backbone_guardrail_applied": backbone_guardrail_applied,
                "backbone_guardrail_delta_m": backbone_guardrail_delta,
                "backbone_smoothing_applied": bool(row.get("backbone_smoothing_applied", False)),
                "backbone_smoothing_delta_m": float(pd.to_numeric(row.get("backbone_smoothing_delta_m"), errors="coerce")),
                "backbone_smoothing_weight": float(pd.to_numeric(row.get("backbone_smoothing_weight"), errors="coerce")),
                "smoothed_backbone_target_z_m": float(pd.to_numeric(row.get("smoothed_backbone_target_z_m"), errors="coerce")),
                "backbone_smoothing_reference_z_m": float(pd.to_numeric(row.get("backbone_smoothing_reference_z_m"), errors="coerce")),
                "backbone_smoothing_reference_source": str(row.get("backbone_smoothing_reference_source", "backbone_target") or "backbone_target"),
            })
            station_delta_abs.append(station_max_delta)
            if station_adjusted:
                adjusted_station_count += 1
        component_work = work.loc[comp_nodes.index].copy()
        component_work, monotone_projection_summary, monotone_projection_profile = _apply_component_channel_core_monotone_projection(component_work)
        if not monotone_projection_profile.empty:
            monotone_profile_rows.append(monotone_projection_profile)
        work.loc[component_work.index, :] = component_work
        component_profile_df = pd.DataFrame(profile_rows)
        if not component_profile_df.empty:
            component_profile_df = component_profile_df.loc[component_profile_df["component_id"] == str(component_id)].copy()
        component_receipts.append({
            "component_id": str(component_id),
            "station_count": int(len(station_df)),
            "adjusted_station_count": int(adjusted_station_count),
            "template_type_counts": {str(k): int(v) for k, v in station_df["xs_template_type"].astype(str).value_counts(dropna=False).items()},
            "support_template_class_counts": {str(k): int(v) for k, v in station_df["xs_support_template_class"].astype(str).value_counts(dropna=False).items()},
            "rebuild_mode_counts": {str(k): int(v) for k, v in component_profile_df["rebuild_mode"].astype(str).value_counts(dropna=False).items()} if not component_profile_df.empty else {},
            "target_source_counts": {str(k): int(v) for k, v in component_profile_df["primary_surface_rebuild_target_source"].astype(str).value_counts(dropna=False).items()} if not component_profile_df.empty else {},
            "canonical_support_class_counts": {str(k): int(v) for k, v in component_profile_df.get("support_class_canonical", pd.Series(dtype="object")).astype(str).value_counts(dropna=False).items()} if not component_profile_df.empty else {},
            "active_interior_target_source_counts": {str(k): int(v) for k, v in component_profile_df.get("active_interior_target_source", pd.Series(dtype="object")).astype(str).value_counts(dropna=False).items()} if not component_profile_df.empty else {},
            "degraded_active_target_station_count": int(component_profile_df.get("active_interior_target_degraded", pd.Series(dtype=float)).fillna(False).astype(bool).sum()) if not component_profile_df.empty else 0,
            "changed_channel_core_node_count": int(np.count_nonzero(component_work["primary_surface_rebuild_applied"].to_numpy(dtype=bool) & component_work["node_role"].astype(str).isin(list(CHANNEL_CORE_ROLES)).to_numpy(dtype=bool))),
            "station_delta_abs_m": _distribution(np.asarray(station_delta_abs, dtype=float)),
            "core_geometry_scale_summary": _distribution(pd.to_numeric(component_profile_df.get("core_geometry_scale", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)) if not component_profile_df.empty else _distribution(np.array([], dtype=float)),
            "inner_weight_scale_summary": _distribution(pd.to_numeric(component_profile_df.get("inner_weight_scale", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)) if not component_profile_df.empty else _distribution(np.array([], dtype=float)),
            "post_rebuild_monotone_projection": monotone_projection_summary,
            "backbone_smoothing": backbone_smoothing_summary,
        })

    _active_source = work.get("active_interior_target_source", pd.Series("missing", index=work.index)).astype(str)
    _canonical_support = work.get("support_class_canonical", pd.Series("missing", index=work.index)).astype(str)
    _rebuild_target_z = pd.to_numeric(work.get("primary_surface_rebuild_target_z_m", np.nan), errors="coerce")
    _fill_backbone = _active_source.eq("missing") & _canonical_support.ne("authoritative_interior") & np.isfinite(_rebuild_target_z.to_numpy(dtype=float))
    if bool(np.any(_fill_backbone)):
        work.loc[_fill_backbone, "active_interior_target_source"] = ACTIVE_INTERIOR_TARGET_BACKBONE
        work.loc[_fill_backbone, "active_interior_target_reason"] = ACTIVE_INTERIOR_TARGET_BACKBONE
        work.loc[_fill_backbone, "active_interior_target_z_m"] = _rebuild_target_z.loc[_fill_backbone].to_numpy(dtype=np.float32)
    _fill_authoritative = _active_source.eq("missing") & _canonical_support.eq("authoritative_interior") & np.isfinite(_rebuild_target_z.to_numpy(dtype=float))
    if bool(np.any(_fill_authoritative)):
        work.loc[_fill_authoritative, "active_interior_target_source"] = ACTIVE_INTERIOR_TARGET_AUTHORITATIVE
        work.loc[_fill_authoritative, "active_interior_target_reason"] = ACTIVE_INTERIOR_TARGET_AUTHORITATIVE
        work.loc[_fill_authoritative, "active_interior_target_z_m"] = _rebuild_target_z.loc[_fill_authoritative].to_numpy(dtype=np.float32)

    outputs: Dict[str, str] = {}
    profile_df_all = pd.DataFrame(profile_rows)
    for _col, _default in [("support_class_canonical", "missing"), ("active_interior_target_source", "missing"), ("active_interior_target_reason", "missing"), ("diagnostic_target_source", "missing"), ("active_interior_target_z_m", np.nan), ("active_interior_target_degraded", False)]:
        if _col not in profile_df_all.columns:
            profile_df_all[_col] = _default
    _profile_active_source = profile_df_all.get("active_interior_target_source", pd.Series("missing", index=profile_df_all.index)).astype(str)
    _profile_support = profile_df_all.get("support_class_canonical", pd.Series("missing", index=profile_df_all.index)).astype(str)
    _profile_target = pd.to_numeric(profile_df_all.get("active_interior_target_z_m", np.nan), errors="coerce")
    _profile_backbone = pd.to_numeric(profile_df_all.get("backbone_target_z_m", np.nan), errors="coerce")
    _profile_fitted = pd.to_numeric(profile_df_all.get("fitted_core_target_z_m", np.nan), errors="coerce")
    _profile_fill_target = _profile_target.copy()
    _profile_fill_target = _profile_fill_target.where(np.isfinite(_profile_fill_target), _profile_fitted)
    _profile_fill_target = _profile_fill_target.where(np.isfinite(_profile_fill_target), _profile_backbone)
    _profile_fill_backbone = _profile_active_source.eq("missing") & _profile_support.ne("authoritative_interior") & np.isfinite(_profile_fill_target.to_numpy(dtype=float))
    if bool(np.any(_profile_fill_backbone)):
        profile_df_all.loc[_profile_fill_backbone, "active_interior_target_source"] = ACTIVE_INTERIOR_TARGET_BACKBONE
        profile_df_all.loc[_profile_fill_backbone, "active_interior_target_reason"] = ACTIVE_INTERIOR_TARGET_BACKBONE
        profile_df_all.loc[_profile_fill_backbone, "active_interior_target_z_m"] = _profile_fill_target.loc[_profile_fill_backbone].to_numpy(dtype=np.float32)
    _profile_fill_authoritative = _profile_active_source.eq("missing") & _profile_support.eq("authoritative_interior") & np.isfinite(_profile_fill_target.to_numpy(dtype=float))
    if bool(np.any(_profile_fill_authoritative)):
        profile_df_all.loc[_profile_fill_authoritative, "active_interior_target_source"] = ACTIVE_INTERIOR_TARGET_AUTHORITATIVE
        profile_df_all.loc[_profile_fill_authoritative, "active_interior_target_reason"] = ACTIVE_INTERIOR_TARGET_AUTHORITATIVE
        profile_df_all.loc[_profile_fill_authoritative, "active_interior_target_z_m"] = _profile_fill_target.loc[_profile_fill_authoritative].to_numpy(dtype=np.float32)
    guardrail_df_all = pd.DataFrame(guardrail_rows)
    guardrail_applied = pd.to_numeric(work.get("primary_surface_backbone_guardrail_applied", pd.Series([False] * len(work), index=work.index)), errors="coerce").fillna(False).astype(bool)
    guardrail_delta = np.abs(pd.to_numeric(work.get("primary_surface_backbone_guardrail_delta_m", pd.Series([0.0] * len(work), index=work.index)), errors="coerce").to_numpy(dtype=float))
    guardrail_by_component_class = {}
    if not guardrail_df_all.empty:
        for component_class, sub in guardrail_df_all.groupby("component_support_class", dropna=False):
            applied_mask = sub["guardrail_applied"].fillna(False).astype(bool)
            guardrail_by_component_class[str(component_class)] = {
                "n": int(len(sub)),
                "applied_count": int(applied_mask.sum()),
                "active_fraction": float(applied_mask.mean()) if len(sub) else 0.0,
                "delta_abs_m": _distribution(np.abs(pd.to_numeric(sub.loc[applied_mask, "guardrail_delta_m"], errors="coerce").to_numpy(dtype=float))),
            }
    summary: Dict[str, Any] = {
        "available": bool(component_receipts),
        "component_count": int(len(component_receipts)),
        "adjusted_node_count": int(np.count_nonzero(pd.to_numeric(work.get("primary_surface_rebuild_applied"), errors="coerce").fillna(False).astype(bool))),
        "changed_channel_core_node_count": int(np.count_nonzero(pd.to_numeric(work.get("primary_surface_rebuild_applied"), errors="coerce").fillna(False).astype(bool).to_numpy(dtype=bool) & work["node_role"].astype(str).isin(list(CHANNEL_CORE_ROLES)).to_numpy(dtype=bool))),
        "rebuild_mode_counts": {str(k): int(v) for k, v in profile_df_all["rebuild_mode"].astype(str).value_counts(dropna=False).items()} if not profile_df_all.empty else {},
        "target_source_counts": {str(k): int(v) for k, v in profile_df_all["primary_surface_rebuild_target_source"].astype(str).value_counts(dropna=False).items()} if not profile_df_all.empty else {},
        "canonical_support_class_counts": {str(k): int(v) for k, v in profile_df_all.get("support_class_canonical", pd.Series(dtype="object")).astype(str).value_counts(dropna=False).items()} if not profile_df_all.empty else {},
        "active_interior_target_source_counts": {str(k): int(v) for k, v in profile_df_all.get("active_interior_target_source", pd.Series(dtype="object")).astype(str).value_counts(dropna=False).items()} if not profile_df_all.empty else {},
        "degraded_active_target_station_count": int(profile_df_all.get("active_interior_target_degraded", pd.Series(dtype=float)).fillna(False).astype(bool).sum()) if not profile_df_all.empty else 0,
        "delta_abs_m": _distribution(np.abs(pd.to_numeric(work.get("primary_surface_rebuild_delta_m"), errors="coerce").to_numpy(dtype=float))),
        "blend_weight_summary": _distribution(pd.to_numeric(work.get("primary_surface_rebuild_weight"), errors="coerce").to_numpy(dtype=float)),
        "core_geometry_scale_summary": _distribution(pd.to_numeric(profile_df_all.get("core_geometry_scale", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)) if not profile_df_all.empty else _distribution(np.array([], dtype=float)),
        "inner_weight_scale_summary": _distribution(pd.to_numeric(profile_df_all.get("inner_weight_scale", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)) if not profile_df_all.empty else _distribution(np.array([], dtype=float)),
        "backbone_led_inner_target_station_count": int(np.count_nonzero(pd.to_numeric(profile_df_all.get("backbone_led_inner_targets", pd.Series(dtype=float)), errors="coerce").fillna(False).astype(bool))) if not profile_df_all.empty else 0,
        "bank_margin_damping_station_count": int(np.count_nonzero(pd.to_numeric(profile_df_all.get("bank_margin_damping_active", pd.Series(dtype=float)), errors="coerce").fillna(False).astype(bool))) if not profile_df_all.empty else 0,
        "backbone_led_inner_relief_scale_summary": _distribution(pd.to_numeric(profile_df_all.get("backbone_led_inner_relief_scale", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)) if not profile_df_all.empty else _distribution(np.array([], dtype=float)),
        "backbone_led_reason_counts": {str(k): int(v) for k, v in profile_df_all.get("backbone_led_inner_reason", pd.Series(dtype="object")).astype(str).value_counts(dropna=False).items()} if not profile_df_all.empty else {},
        "backbone_smoothing_adjusted_station_count": int(np.count_nonzero(pd.to_numeric(profile_df_all.get("backbone_smoothing_applied", pd.Series(dtype=float)), errors="coerce").fillna(False).astype(bool))) if not profile_df_all.empty else 0,
        "backbone_smoothing_delta_abs_m": _distribution(np.abs(pd.to_numeric(profile_df_all.get("backbone_smoothing_delta_m", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float))) if not profile_df_all.empty else _distribution(np.array([], dtype=float)),
        "backbone_smoothing_weight_summary": _distribution(pd.to_numeric(profile_df_all.get("backbone_smoothing_weight", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)) if not profile_df_all.empty else _distribution(np.array([], dtype=float)),
        "backbone_smoothing_linear_assisted_count": int((backbone_smoothing_summary or {}).get("linear_assisted_count", 0)),
        "backbone_guardrail_adjusted_node_count": int(np.count_nonzero(guardrail_applied.to_numpy(dtype=bool))),
        "backbone_guardrail_delta_abs_m": _distribution(guardrail_delta[guardrail_applied.to_numpy(dtype=bool)]),
        "backbone_guardrail_by_component_class": guardrail_by_component_class,
        "post_rebuild_monotone_available": bool(component_receipts),
        "post_rebuild_monotone_adjusted_node_count": int(np.count_nonzero(pd.Series(work.get("post_rebuild_monotone_applied", pd.Series([False] * len(work), index=work.index)), index=work.index).fillna(False).astype(bool))),
        "post_rebuild_monotone_delta_abs_m": _distribution(np.abs(pd.to_numeric(work.get("post_rebuild_monotone_delta_m", pd.Series([0.0] * len(work), index=work.index)), errors="coerce").to_numpy(dtype=float))),
        "post_rebuild_monotone_pre_violation_count": int(sum(int((c.get("post_rebuild_monotone_projection") or {}).get("pre_monotone_violation_count", 0)) for c in component_receipts)),
        "post_rebuild_monotone_post_violation_count": int(sum(int((c.get("post_rebuild_monotone_projection") or {}).get("post_monotone_violation_count", 0)) for c in component_receipts)),
        "components": component_receipts,
    }
    if profile_rows:
        profile_csv = river_dir / "river_primary_surface_rebuild_profile.csv"
        pd.DataFrame(profile_rows).to_csv(profile_csv, index=False)
        summary_path = river_dir / "river_primary_surface_rebuild_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        smoothing_profile_csv = river_dir / "river_backbone_smoothing_profile.csv"
        smoothing_df_all = backbone_smoothing_profile_df.copy() if not backbone_smoothing_profile_df.empty else pd.DataFrame()
        if not smoothing_df_all.empty:
            smoothing_df_all = smoothing_df_all.rename(columns={
                "original_backbone_target_z_m": "backbone_target_z_m",
                "requested_adjustment_m": "backbone_smoothing_requested_adjustment_m",
                "clamped_adjustment_m": "backbone_smoothing_clamped_adjustment_m",
                "applied_adjustment_m": "backbone_smoothing_delta_m",
                "smoothing_applied": "backbone_smoothing_applied",
                "smoothing_weight": "backbone_smoothing_weight",
                "max_shift_m": "backbone_smoothing_max_shift_m",
                "eligible": "backbone_smoothing_eligible",
                "rejection_reason": "backbone_smoothing_rejection_reason",
            })
        smoothing_df_all.to_csv(smoothing_profile_csv, index=False)
        smoothing_summary_path = river_dir / "river_backbone_smoothing_summary.json"
        smoothing_summary_payload = dict(backbone_smoothing_summary or {})
        smoothing_summary_payload.update({
            "available": bool(component_receipts),
            "adjusted_station_count": summary.get("backbone_smoothing_adjusted_station_count", 0),
            "delta_abs_m": summary.get("backbone_smoothing_delta_abs_m", {}),
            "weight_summary": summary.get("backbone_smoothing_weight_summary", {}),
            "linear_assisted_count": summary.get("backbone_smoothing_linear_assisted_count", 0),
        })
        smoothing_summary_path.write_text(json.dumps(smoothing_summary_payload, indent=2), encoding="utf-8")
        backbone_action_gate_receipt_path = river_dir / "river_backbone_action_gate_receipt.json"
        backbone_action_gate_payload = dict((backbone_smoothing_summary or {}).get("action_gate_receipt", {}) or {})
        backbone_action_gate_payload.update({
            "available": bool(component_receipts),
            "effectiveness_warning": (backbone_smoothing_summary or {}).get("effectiveness_warning"),
            "backbone_smoothing_summary_path": str(smoothing_summary_path),
        })
        backbone_action_gate_receipt_path.write_text(json.dumps(backbone_action_gate_payload, indent=2), encoding="utf-8")
        guardrail_profile_csv = river_dir / "river_primary_surface_backbone_guardrail_profile.csv"
        guardrail_df_all.to_csv(guardrail_profile_csv, index=False)
        guardrail_summary_path = river_dir / "river_primary_surface_backbone_guardrail_summary.json"
        guardrail_summary_path.write_text(json.dumps({
            "available": bool(component_receipts),
            "adjusted_node_count": summary.get("backbone_guardrail_adjusted_node_count", 0),
            "delta_abs_m": summary.get("backbone_guardrail_delta_abs_m", {}),
            "by_component_class": summary.get("backbone_guardrail_by_component_class", {}),
        }, indent=2), encoding="utf-8")
        monotone_profile_csv = river_dir / "river_post_rebuild_monotone_projection_profile.csv"
        if monotone_profile_rows:
            pd.concat(monotone_profile_rows, ignore_index=True).sort_values(["component_id", "station_m", "node_role"]).to_csv(monotone_profile_csv, index=False)
        else:
            work.loc[work["node_role"].astype(str).isin(list(CHANNEL_CORE_ROLES)), [
                "component_id", "station_m", "node_role", "bed_z_m", "post_rebuild_monotone_applied",
                "post_rebuild_monotone_delta_m", "post_rebuild_monotone_target_z_m", "post_rebuild_monotone_weight"
            ]].sort_values(["component_id", "station_m", "node_role"]).to_csv(monotone_profile_csv, index=False)
        monotone_summary_path = river_dir / "river_post_rebuild_monotone_projection_summary.json"
        monotone_summary_path.write_text(json.dumps({
            "available": bool(component_receipts),
            "adjusted_node_count": summary.get("post_rebuild_monotone_adjusted_node_count", 0),
            "pre_violation_count": summary.get("post_rebuild_monotone_pre_violation_count", 0),
            "post_violation_count": summary.get("post_rebuild_monotone_post_violation_count", 0),
            "delta_abs_m": summary.get("post_rebuild_monotone_delta_abs_m", {}),
            "components": [{"component_id": c.get("component_id"), **(c.get("post_rebuild_monotone_projection") or {})} for c in component_receipts],
        }, indent=2), encoding="utf-8")
        width_profile_csv = river_dir / "river_centerline_width_propagation_profile.csv"
        profile_df_all.loc[:, [
            "component_id", "station_m", "component_support_class", "xs_support_template_class", "station_support_regime", "rebuild_mode",
            "primary_surface_rebuild_target_source", "support_class_canonical", "active_interior_target_source", "active_interior_target_z_m", "active_interior_target_degraded", "core_geometry_scale", "inner_weight_scale",
            "backbone_led_inner_targets", "backbone_led_inner_relief_scale", "backbone_led_inner_reason",
            "bank_margin_damping_active", "station_adjusted", "station_changed_channel_core_node_count"
        ]].to_csv(width_profile_csv, index=False)
        width_summary_path = river_dir / "river_centerline_width_propagation_summary.json"
        width_summary_path.write_text(json.dumps({
            "available": bool(component_receipts),
            "station_count": int(len(profile_df_all)),
            "backbone_led_station_count": summary.get("backbone_led_inner_target_station_count", 0),
            "bank_margin_damping_station_count": summary.get("bank_margin_damping_station_count", 0),
            "backbone_led_inner_relief_scale_summary": summary.get("backbone_led_inner_relief_scale_summary", {}),
            "backbone_led_reason_counts": summary.get("backbone_led_reason_counts", {}),
            "by_component_id": _centerline_width_group_summary(profile_df_all, "component_id"),
            "by_component_support_class": _centerline_width_group_summary(profile_df_all, "component_support_class"),
            "by_station_support_regime": _centerline_width_group_summary(profile_df_all, "station_support_regime"),
        }, indent=2), encoding="utf-8")
        outputs["primary_surface_rebuild_profile"] = str(profile_csv)
        outputs["primary_surface_rebuild_summary"] = str(summary_path)
        outputs["backbone_smoothing_profile"] = str(smoothing_profile_csv)
        outputs["backbone_smoothing_summary"] = str(smoothing_summary_path)
        outputs["backbone_action_gate_receipt"] = str(backbone_action_gate_receipt_path)
        outputs["primary_surface_backbone_guardrail_profile"] = str(guardrail_profile_csv)
        outputs["primary_surface_backbone_guardrail_summary"] = str(guardrail_summary_path)
        outputs["post_rebuild_monotone_projection_profile"] = str(monotone_profile_csv)
        outputs["post_rebuild_monotone_projection_summary"] = str(monotone_summary_path)
        outputs["centerline_width_propagation_profile"] = str(width_profile_csv)
        outputs["centerline_width_propagation_summary"] = str(width_summary_path)
        active_logger.info(
            "[RIVER][PRIMARY] Primary surface rebuild applied: adjusted_nodes=%d channel_core_changed=%d guardrail_adjusted=%d components=%d",
            int(summary["adjusted_node_count"]),
            int(summary.get("changed_channel_core_node_count", 0)),
            int(summary.get("backbone_guardrail_adjusted_node_count", 0)),
            int(summary["component_count"]),
        )
    else:
        active_logger.info("[RIVER][PRIMARY] Primary surface rebuild skipped: insufficient station roles")
    backbone_action_gate_receipt = dict((backbone_smoothing_summary or {}).get("action_gate_receipt", {}) or {}) if component_receipts else {}
    if bool(backbone_action_gate_receipt.get("should_fail", False)):
        receipt_path = outputs.get("backbone_action_gate_receipt", str(river_dir / "river_backbone_action_gate_receipt.json"))
        raise RuntimeError(
            "Unsupported-mainstem backbone smoothing candidates had requested adjustments but no applied backbone action; "
            f"see {receipt_path}"
        )
    return work, outputs, summary


__all__ = ["apply_primary_surface_rebuild_to_nodes"]
