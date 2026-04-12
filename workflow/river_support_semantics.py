from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

WEAK_SUPPORT_CLASSES = {
    "bank_only_low_confidence",
    "bank_supported_with_good_longitudinal_context",
    "supported_transition",
    "bank_only_authoritative",
}

LONGITUDINAL_MEASURED_PROTECTED = "longitudinal_measured_protected"
LONGITUDINAL_CHANNEL_ANCHORED = "longitudinal_channel_anchored"
LONGITUDINAL_WEAK_SUPPORT = "longitudinal_weak_support"

MEASURED_CLASS_STRONG = "true_measured_xs_strong"
MEASURED_CLASS_PARTIAL = "true_measured_xs_partial"
MEASURED_CLASS_BANK_ONLY = "bank_only_authoritative"
MEASURED_CLASS_NEARBY = "authoritative_nearby_but_not_measured_xs"
MEASURED_CLASS_NONE = "no_authoritative_xs_support"

AUTHORITATIVE_BED_CLASS_STRONG = "authoritative_bed_strong"
AUTHORITATIVE_BED_CLASS_NEARBY = "authoritative_bed_nearby"
AUTHORITATIVE_BED_CLASS_BANK_ONLY = "authoritative_bank_margin_only"
AUTHORITATIVE_BED_CLASS_AMBIGUOUS = "authoritative_ambiguous_only"
AUTHORITATIVE_BED_CLASS_NONE = "no_authoritative_bed_support"

MEASURED_SUPPORT_DISTANCE_FAR_M = 500.0
MEASURED_SUPPORT_DISTANCE_VERY_FAR_M = 1000.0
AUTHORITATIVE_BED_SUPPORT_DISTANCE_FAR_M = 500.0
AUTHORITATIVE_BED_SUPPORT_DISTANCE_VERY_FAR_M = 1000.0


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = pd.to_numeric(row.get(key), errors="coerce")
    return float(value) if pd.notna(value) else float(default)


def _flag(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if pd.isna(value):
        return bool(default)
    return bool(value)


def _measured_support_distance_m(row: Mapping[str, Any]) -> float:
    for key in (
        "station_measured_support_distance_m",
        "measured_support_distance_median_m",
        "profile_measured_support_distance_m",
        "longitudinal_tendency_anchor_distance_m",
    ):
        value = pd.to_numeric(row.get(key), errors="coerce")
        if pd.notna(value):
            return float(value)
    return float("nan")


def _authoritative_bed_support_distance_m(row: Mapping[str, Any]) -> float:
    for key in (
        "station_authoritative_bed_support_distance_m",
        "authoritative_bed_support_distance_median_m",
        "profile_authoritative_bed_support_distance_m",
    ):
        value = pd.to_numeric(row.get(key), errors="coerce")
        if pd.notna(value):
            return float(value)
    return float("nan")


def station_semantics(row: Mapping[str, Any], *, station_authoritative: bool | None = None, station_measured_xs: bool | None = None) -> dict[str, Any]:
    true_measured_fraction = _num(row, "station_true_measured_xs_fraction")
    indirect_fraction = _num(row, "station_indirect_xs_fraction")
    residual_fraction = _num(row, "station_residual_xs_fraction")
    authoritative_fraction = _num(row, "station_authoritative_fraction")
    channel_authoritative_fraction = _num(row, "station_authoritative_channel_fraction")
    bank_authoritative_fraction = _num(row, "station_authoritative_bank_fraction")
    unsupported_fraction = _num(row, "unsupported_fraction")
    xs_fraction = _num(row, "xs_support_fraction")
    authoritative_anchor_fraction = _num(row, "authoritative_anchor_fraction")
    prediction_support_confidence = _num(row, "prediction_support_confidence")
    auth_inner_count = _num(row, "auth_xs_support_inner_count")
    auth_bank_count = _num(row, "auth_xs_support_bank_count")
    auth_support_class = str(row.get("auth_xs_support_class", "") or "")
    station_true_measured_qualified = _flag(row, "station_true_measured_xs_qualified")
    station_bank_only_authoritative = _flag(row, "station_bank_only_authoritative")
    station_bank_protected = _flag(row, "station_bank_protected")
    station_channel_protected = _flag(row, "station_channel_protected")
    station_inner_rebuildable = _flag(row, "station_inner_rebuildable", True)

    station_authoritative_bed_support_present = _flag(row, "station_authoritative_bed_support_present")
    station_authoritative_bank_margin_present = _flag(row, "station_authoritative_bank_margin_present")
    station_authoritative_role = str(row.get("station_authoritative_role", row.get("profile_authoritative_role", "no_authoritative_support")) or "no_authoritative_support")
    bed_support_fraction = _num(row, "station_authoritative_bed_support_fraction")
    bank_margin_fraction = _num(row, "station_authoritative_bank_margin_fraction")
    bed_core_fraction = _num(row, "station_authoritative_bed_core_fraction")
    ambiguous_fraction = _num(row, "station_authoritative_ambiguous_fraction")

    measured_support_distance_m = _measured_support_distance_m(row)
    authoritative_bed_support_distance_m = _authoritative_bed_support_distance_m(row)
    far_from_measured = (not np.isfinite(measured_support_distance_m)) or (measured_support_distance_m >= MEASURED_SUPPORT_DISTANCE_FAR_M)
    very_far_from_measured = (not np.isfinite(measured_support_distance_m)) or (measured_support_distance_m >= MEASURED_SUPPORT_DISTANCE_VERY_FAR_M)
    far_from_measured_fraction = _num(row, "far_from_measured_fraction")
    far_from_authoritative_bed_fraction = _num(row, "far_from_authoritative_bed_fraction")
    far_from_authoritative_bed = (not np.isfinite(authoritative_bed_support_distance_m)) or (authoritative_bed_support_distance_m >= AUTHORITATIVE_BED_SUPPORT_DISTANCE_FAR_M)
    very_far_from_authoritative_bed = (not np.isfinite(authoritative_bed_support_distance_m)) or (authoritative_bed_support_distance_m >= AUTHORITATIVE_BED_SUPPORT_DISTANCE_VERY_FAR_M)

    bed_support_present = bool(
        station_authoritative_bed_support_present
        or auth_inner_count > 0.0
        or bed_support_fraction > 0.0
        or station_authoritative_role in {"authoritative_bed_core", "authoritative_bed_inner", "authoritative_bed_mixed"}
    )
    bank_margin_present = bool(
        station_authoritative_bank_margin_present
        or station_bank_only_authoritative
        or auth_bank_count > 0.0
        or bank_margin_fraction > 0.0
        or bank_authoritative_fraction > 0.0
        or station_authoritative_role == "authoritative_bank_margin"
    )
    bed_support_strength = max(bed_support_fraction, min(channel_authoritative_fraction, 1.0))

    station_authoritative = bool(
        authoritative_fraction > 0.0 or bed_support_present or bank_margin_present
    ) if station_authoritative is None else bool(station_authoritative)
    station_measured_xs = bool(true_measured_fraction > 0.0 or station_true_measured_qualified) if station_measured_xs is None else bool(station_measured_xs)

    if station_bank_only_authoritative or (auth_bank_count > 0.0 and auth_inner_count <= 0.0 and true_measured_fraction <= 0.0 and not station_true_measured_qualified):
        measured_support_class = MEASURED_CLASS_BANK_ONLY
    elif station_true_measured_qualified and (auth_support_class == "authoritative_bed_core" or true_measured_fraction >= 0.50 or channel_authoritative_fraction >= 0.45):
        measured_support_class = MEASURED_CLASS_STRONG
    elif station_true_measured_qualified or true_measured_fraction > 0.0:
        measured_support_class = MEASURED_CLASS_PARTIAL
    elif auth_inner_count > 0.0 or auth_bank_count > 0.0 or channel_authoritative_fraction > 0.0 or bank_authoritative_fraction > 0.0:
        measured_support_class = MEASURED_CLASS_NEARBY
    else:
        measured_support_class = MEASURED_CLASS_NONE

    if bed_support_present and (bed_core_fraction >= 0.25 or bed_support_strength >= 0.50 or station_authoritative_role == "authoritative_bed_core") and not far_from_authoritative_bed:
        authoritative_bed_support_class = AUTHORITATIVE_BED_CLASS_STRONG
    elif bed_support_present:
        authoritative_bed_support_class = AUTHORITATIVE_BED_CLASS_NEARBY
    elif bank_margin_present:
        authoritative_bed_support_class = AUTHORITATIVE_BED_CLASS_BANK_ONLY
    elif ambiguous_fraction > 0.0 or (station_authoritative and authoritative_fraction > 0.0):
        authoritative_bed_support_class = AUTHORITATIVE_BED_CLASS_AMBIGUOUS
    else:
        authoritative_bed_support_class = AUTHORITATIVE_BED_CLASS_NONE

    weak_due_to_distance = far_from_measured and measured_support_class not in {MEASURED_CLASS_STRONG, MEASURED_CLASS_PARTIAL}
    weak_due_to_bed_distance = far_from_authoritative_bed and authoritative_bed_support_class not in {AUTHORITATIVE_BED_CLASS_STRONG}
    generic_upstream = (
        weak_due_to_distance
        and weak_due_to_bed_distance
        and authoritative_bed_support_class in {AUTHORITATIVE_BED_CLASS_BANK_ONLY, AUTHORITATIVE_BED_CLASS_AMBIGUOUS, AUTHORITATIVE_BED_CLASS_NONE}
        and bed_support_strength <= 0.15
        and true_measured_fraction <= 0.10
    )

    if measured_support_class == MEASURED_CLASS_STRONG:
        support_regime = MEASURED_CLASS_STRONG
        template_type = "measured"
    elif measured_support_class == MEASURED_CLASS_PARTIAL:
        support_regime = MEASURED_CLASS_PARTIAL
        template_type = "hybrid_xs"
    elif authoritative_bed_support_class == AUTHORITATIVE_BED_CLASS_STRONG:
        support_regime = "authoritative_bed_controlled"
        template_type = "hybrid_longitudinal"
    elif authoritative_bed_support_class == AUTHORITATIVE_BED_CLASS_NEARBY:
        support_regime = "authoritative_bed_transition"
        template_type = "hybrid_longitudinal"
    elif generic_upstream:
        support_regime = "bank_only_low_confidence"
        template_type = "generic_symmetric"
    elif authoritative_bed_support_class == AUTHORITATIVE_BED_CLASS_BANK_ONLY:
        support_regime = MEASURED_CLASS_BANK_ONLY
        template_type = "generic_u"
    elif measured_support_class == MEASURED_CLASS_NEARBY:
        support_regime = MEASURED_CLASS_NEARBY
        template_type = "hybrid_longitudinal"
    elif unsupported_fraction >= 0.80 and bed_support_strength <= 0.10 and true_measured_fraction <= 0.10:
        support_regime = "bank_only_low_confidence"
        template_type = "generic_symmetric"
    elif unsupported_fraction >= 0.55 and bed_support_strength <= 0.15 and true_measured_fraction <= 0.10:
        support_regime = "bank_supported_with_good_longitudinal_context"
        template_type = "generic_u"
    elif max(indirect_fraction, residual_fraction) > 0.0 and true_measured_fraction <= 0.10 and bed_support_strength <= 0.10:
        support_regime = "supported_transition"
        template_type = "hybrid_longitudinal"
    elif authoritative_anchor_fraction <= 0.25 and true_measured_fraction <= 0.10:
        support_regime = "supported_transition"
        template_type = "hybrid_longitudinal"
    else:
        support_regime = "bank_supported_with_good_longitudinal_context"
        template_type = "generic_u"

    if measured_support_class in {MEASURED_CLASS_STRONG, MEASURED_CLASS_PARTIAL}:
        envelope_class = "measured_supported"
    elif authoritative_bed_support_class == AUTHORITATIVE_BED_CLASS_STRONG:
        envelope_class = "authoritative_bed_supported"
    elif generic_upstream and very_far_from_measured and very_far_from_authoritative_bed:
        envelope_class = "unsupported_upstream_generic"
    elif generic_upstream:
        envelope_class = "weak_support_upstream"
    elif authoritative_bed_support_class == AUTHORITATIVE_BED_CLASS_BANK_ONLY:
        envelope_class = "bank_margin_only"
    elif weak_due_to_distance or weak_due_to_bed_distance:
        envelope_class = "distance_limited_transition"
    else:
        envelope_class = "moderate_support_transition"

    if measured_support_class in {MEASURED_CLASS_STRONG, MEASURED_CLASS_PARTIAL}:
        longitudinal_regime = LONGITUDINAL_MEASURED_PROTECTED
        longitudinal_reason = measured_support_class
    elif authoritative_bed_support_class in {AUTHORITATIVE_BED_CLASS_STRONG, AUTHORITATIVE_BED_CLASS_NEARBY} and not far_from_authoritative_bed:
        longitudinal_regime = LONGITUDINAL_CHANNEL_ANCHORED
        longitudinal_reason = "authoritative_bed_support"
    elif authoritative_bed_support_class == AUTHORITATIVE_BED_CLASS_BANK_ONLY:
        longitudinal_regime = LONGITUDINAL_WEAK_SUPPORT
        longitudinal_reason = "bank_margin_only_support"
    elif weak_due_to_distance or weak_due_to_bed_distance:
        longitudinal_regime = LONGITUDINAL_WEAK_SUPPORT
        longitudinal_reason = "far_from_true_bed_support"
    elif unsupported_fraction >= 0.45 or xs_fraction <= 0.20 or max(indirect_fraction, residual_fraction) >= 0.25:
        longitudinal_regime = LONGITUDINAL_WEAK_SUPPORT
        longitudinal_reason = "weak_support"
    else:
        longitudinal_regime = LONGITUDINAL_CHANNEL_ANCHORED
        longitudinal_reason = "moderate_channel_support"

    if measured_support_class in {MEASURED_CLASS_STRONG, MEASURED_CLASS_PARTIAL}:
        protection_regime = "true_measured_protected"
    elif authoritative_bed_support_class in {AUTHORITATIVE_BED_CLASS_STRONG, AUTHORITATIVE_BED_CLASS_NEARBY}:
        protection_regime = "authoritative_bed_protected"
    elif authoritative_bed_support_class == AUTHORITATIVE_BED_CLASS_BANK_ONLY or (station_bank_protected and station_inner_rebuildable):
        protection_regime = "bank_only_protected"
    elif station_channel_protected and not station_inner_rebuildable:
        protection_regime = "channel_protected"
    elif station_inner_rebuildable:
        protection_regime = "inner_rebuildable"
    else:
        protection_regime = "unprotected"

    if measured_support_class in {MEASURED_CLASS_STRONG, MEASURED_CLASS_PARTIAL}:
        rebuild_regime = "blocked_true_measured_xs"
    elif authoritative_bed_support_class in {AUTHORITATIVE_BED_CLASS_STRONG, AUTHORITATIVE_BED_CLASS_NEARBY}:
        rebuild_regime = "blocked_authoritative_bed_support"
    elif station_channel_protected and not station_inner_rebuildable:
        rebuild_regime = "blocked_channel_protected"
    elif not station_inner_rebuildable:
        rebuild_regime = "blocked_no_inner_rebuildable_nodes"
    elif weak_due_to_distance or weak_due_to_bed_distance:
        rebuild_regime = "eligible"
    elif support_regime in {MEASURED_CLASS_NEARBY, MEASURED_CLASS_BANK_ONLY, "bank_only_low_confidence", "bank_supported_with_good_longitudinal_context", "supported_transition"}:
        rebuild_regime = "eligible"
    elif template_type == "measured":
        rebuild_regime = "blocked_measured_or_missing_template"
    elif support_regime not in WEAK_SUPPORT_CLASSES and support_regime != MEASURED_CLASS_NEARBY:
        rebuild_regime = "blocked_not_weak_support"
    else:
        rebuild_regime = "eligible"

    return {
        "station_true_measured_xs_fraction": true_measured_fraction,
        "station_indirect_xs_fraction": indirect_fraction,
        "station_residual_xs_fraction": residual_fraction,
        "station_authoritative_fraction": authoritative_fraction,
        "station_authoritative_channel_fraction": channel_authoritative_fraction,
        "station_authoritative_bank_fraction": bank_authoritative_fraction,
        "station_authoritative_bed_support_fraction": bed_support_fraction,
        "station_authoritative_bank_margin_fraction": bank_margin_fraction,
        "station_authoritative_bed_core_fraction": bed_core_fraction,
        "station_authoritative_ambiguous_fraction": ambiguous_fraction,
        "station_authoritative_bed_support_distance_m": authoritative_bed_support_distance_m,
        "station_far_from_authoritative_bed": far_from_authoritative_bed,
        "station_far_from_authoritative_bed_fraction": far_from_authoritative_bed_fraction,
        "station_measured_support_distance_m": measured_support_distance_m,
        "station_far_from_measured": far_from_measured,
        "station_far_from_measured_fraction": far_from_measured_fraction,
        "station_measured_support_class": measured_support_class,
        "station_authoritative_bed_support_class": authoritative_bed_support_class,
        "station_measured_support_envelope_class": envelope_class,
        "station_support_regime": support_regime,
        "station_template_type": template_type,
        "longitudinal_support_regime": longitudinal_regime,
        "longitudinal_tendency_suppression_reason": longitudinal_reason,
        "station_protection_regime": protection_regime,
        "station_rebuild_regime": rebuild_regime,
        "prediction_support_confidence": prediction_support_confidence,
        "station_bank_protected": station_bank_protected,
        "station_channel_protected": station_channel_protected,
        "station_inner_rebuildable": station_inner_rebuildable,
        "station_authoritative_bed_support_present": bed_support_present,
        "station_authoritative_bank_margin_present": bank_margin_present,
    }


__all__ = [
    "WEAK_SUPPORT_CLASSES",
    "LONGITUDINAL_MEASURED_PROTECTED",
    "LONGITUDINAL_CHANNEL_ANCHORED",
    "LONGITUDINAL_WEAK_SUPPORT",
    "MEASURED_CLASS_STRONG",
    "MEASURED_CLASS_PARTIAL",
    "MEASURED_CLASS_BANK_ONLY",
    "MEASURED_CLASS_NEARBY",
    "MEASURED_CLASS_NONE",
    "MEASURED_SUPPORT_DISTANCE_FAR_M",
    "MEASURED_SUPPORT_DISTANCE_VERY_FAR_M",
    "station_semantics",
]
