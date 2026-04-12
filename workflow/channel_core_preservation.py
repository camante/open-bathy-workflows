from __future__ import annotations

from typing import Any, Optional

from support_classes import SupportClass

import numpy as np


CHANNEL_CORE_STRUCTURE_ONLY_MIN = np.float32(0.55)
CHANNEL_CORE_SUPPORT_CONFIDENCE_MIN = np.float32(0.35)
CHANNEL_CORE_MEASURED_ANCHOR_MAX = np.float32(0.25)


def _fraction(arr: Optional[np.ndarray], shape: tuple[int, ...], *, default: float = 0.0) -> np.ndarray:
    if arr is None:
        return np.full(shape, float(default), dtype=np.float32)
    return np.clip(np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=float(default)), 0.0, 1.0).astype(np.float32)


def _bool_mask(arr: Optional[np.ndarray], shape: tuple[int, ...], *, default: bool = False) -> np.ndarray:
    if arr is None:
        return np.full(shape, bool(default), dtype=bool)
    return np.asarray(np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=1.0 if default else 0.0) > 0.5, dtype=bool)


def compute_channel_core_preserve_mask(
    *,
    prediction_support_confidence: Optional[np.ndarray],
    measured_anchor_fraction: Optional[np.ndarray],
    structure_only_fraction: Optional[np.ndarray],
    low_support_caution: Optional[np.ndarray],
    prediction_admissibility: Optional[np.ndarray],
) -> np.ndarray:
    shape = None
    for arr in (
        prediction_support_confidence,
        measured_anchor_fraction,
        structure_only_fraction,
        low_support_caution,
        prediction_admissibility,
    ):
        if arr is not None:
            shape = np.asarray(arr).shape
            break
    if shape is None:
        raise ValueError("channel-core preserve mask requires at least one diagnostic array")

    support_conf = _fraction(prediction_support_confidence, shape)
    measured = _fraction(measured_anchor_fraction, shape)
    structure = _fraction(structure_only_fraction, shape)
    caution = _bool_mask(low_support_caution, shape, default=False)
    admiss = _bool_mask(prediction_admissibility, shape, default=True)

    has_prediction = np.zeros(shape, dtype=bool)
    for arr in (
        prediction_support_confidence,
        measured_anchor_fraction,
        structure_only_fraction,
        low_support_caution,
        prediction_admissibility,
    ):
        if arr is not None:
            has_prediction |= np.isfinite(np.asarray(arr, dtype=np.float32))

    return (
        has_prediction
        & admiss
        & (~caution)
        & (structure >= CHANNEL_CORE_STRUCTURE_ONLY_MIN)
        & (support_conf >= CHANNEL_CORE_SUPPORT_CONFIDENCE_MIN)
        & (measured < CHANNEL_CORE_MEASURED_ANCHOR_MAX)
    )



def apply_channel_core_preservation(
    *,
    conditioned_surface: np.ndarray,
    primary_surface: np.ndarray,
    primary_domain: np.ndarray,
    channel_core_preserve: np.ndarray,
    authoritative_locked: Optional[np.ndarray] = None,
    bank_influence: Optional[np.ndarray] = None,
    measured_anchor_fraction: Optional[np.ndarray] = None,
    structure_only_fraction: Optional[np.ndarray] = None,
    prediction_support_confidence: Optional[np.ndarray] = None,
    guidance_influence: Optional[np.ndarray] = None,
    support: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    conditioned = np.array(conditioned_surface, dtype=np.float32, copy=True)
    primary = np.asarray(primary_surface, dtype=np.float32)
    domain = np.asarray(primary_domain, dtype=bool)
    preserve = np.asarray(channel_core_preserve, dtype=bool)
    locked = np.zeros(primary.shape, dtype=bool) if authoritative_locked is None else np.asarray(authoritative_locked, dtype=bool)

    zone = domain & preserve & (~locked) & np.isfinite(primary) & np.isfinite(conditioned)
    guidance = None if guidance_influence is None else np.array(guidance_influence, dtype=np.float32, copy=True)
    support_out = None if support is None else np.array(support, copy=True)
    if not np.any(zone):
        return {
            "conditioned": conditioned,
            "guidance_influence": guidance,
            "support": support_out,
            "receipt": {
                "preservation_applied_pixels": 0,
                "preservation_mean_strength": 0.0,
                "preservation_p95_strength": 0.0,
                "post_preservation_abs_delta_mean_m": 0.0,
                "post_preservation_abs_delta_p95_m": 0.0,
            },
        }

    bank = _fraction(bank_influence, primary.shape)
    measured = _fraction(measured_anchor_fraction, primary.shape)
    structure = _fraction(structure_only_fraction, primary.shape)
    support_conf = _fraction(prediction_support_confidence, primary.shape)

    preserve_strength = np.clip(
        0.35
        + (0.30 * bank)
        + (0.18 * structure)
        + (0.12 * support_conf)
        - (0.22 * measured),
        0.25,
        0.92,
    ).astype(np.float32)
    conditioned[zone] = (
        ((1.0 - preserve_strength[zone]) * conditioned[zone])
        + (preserve_strength[zone] * primary[zone])
    ).astype(np.float32)

    if guidance is not None:
        preserve_floor = np.clip(
            0.78
            + (0.10 * bank)
            + (0.08 * structure)
            - (0.06 * measured),
            0.72,
            0.98,
        ).astype(np.float32)
        guidance[zone] = np.maximum(guidance[zone], preserve_floor[zone]).astype(np.float32)

    if support_out is not None:
        structure_scaffold = zone & (structure >= np.float32(0.75))
        river_guided = zone & (~structure_scaffold)
        if np.any(structure_scaffold):
            support_out[structure_scaffold] = int(SupportClass.SCAFFOLD_INFERRED)
        if np.any(river_guided):
            support_out[river_guided] = int(SupportClass.GUIDANCE_CONDITIONED_RIVER)

    post_abs_delta = np.abs(conditioned[zone] - primary[zone]).astype(np.float32)
    zone_strength = preserve_strength[zone]
    return {
        "conditioned": conditioned,
        "guidance_influence": guidance,
        "support": support_out,
        "receipt": {
            "preservation_applied_pixels": int(np.count_nonzero(zone)),
            "preservation_mean_strength": float(np.mean(zone_strength)),
            "preservation_p95_strength": float(np.percentile(zone_strength, 95.0)),
            "post_preservation_abs_delta_mean_m": float(np.mean(post_abs_delta)),
            "post_preservation_abs_delta_p95_m": float(np.percentile(post_abs_delta, 95.0)),
        },
    }


def build_channel_core_preservation_diagnostics(
    *,
    primary_surface: np.ndarray,
    conditioned_surface: np.ndarray,
    primary_domain: np.ndarray,
    channel_core_preserve: np.ndarray,
    authoritative_locked: Optional[np.ndarray] = None,
    bank_influence: Optional[np.ndarray] = None,
    measured_anchor_fraction: Optional[np.ndarray] = None,
    structure_only_fraction: Optional[np.ndarray] = None,
    prediction_support_confidence: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    primary = np.asarray(primary_surface, dtype=np.float32)
    conditioned = np.asarray(conditioned_surface, dtype=np.float32)
    domain = np.asarray(primary_domain, dtype=bool)
    preserve = np.asarray(channel_core_preserve, dtype=bool)
    locked = np.zeros(primary.shape, dtype=bool) if authoritative_locked is None else np.asarray(authoritative_locked, dtype=bool)

    zone = domain & preserve & (~locked) & np.isfinite(primary) & np.isfinite(conditioned)
    delta = np.full(primary.shape, np.nan, dtype=np.float32)
    if np.any(zone):
        delta[zone] = (conditioned[zone] - primary[zone]).astype(np.float32)
    abs_delta = np.abs(delta).astype(np.float32)

    bank = _fraction(bank_influence, primary.shape)
    measured = _fraction(measured_anchor_fraction, primary.shape)
    structure = _fraction(structure_only_fraction, primary.shape)
    support_conf = _fraction(prediction_support_confidence, primary.shape)
    bank_pull_risk = (bank * structure * (1.0 - measured) * (1.0 - support_conf)).astype(np.float32)
    bank_pull_risk[~(domain & preserve & (~locked))] = 0.0

    zone_delta = delta[zone]
    zone_abs = abs_delta[zone]
    if zone_delta.size:
        signed_mean = float(np.mean(zone_delta))
        signed_median = float(np.median(zone_delta))
        abs_mean = float(np.mean(zone_abs))
        abs_p95 = float(np.percentile(zone_abs, 95.0))
        max_abs = float(np.max(zone_abs))
    else:
        signed_mean = signed_median = abs_mean = abs_p95 = max_abs = 0.0

    return {
        "channel_core_preservation_zone": zone.astype(np.uint8),
        "channel_core_prepost_delta": delta.astype(np.float32),
        "channel_core_bank_pull_risk": np.clip(bank_pull_risk, 0.0, 1.0).astype(np.float32),
        "receipt": {
            "channel_core_preserve_pixels": int(np.count_nonzero(preserve & domain)),
            "channel_core_zone_pixels": int(np.count_nonzero(zone)),
            "authoritative_locked_pixels_excluded": int(np.count_nonzero(domain & preserve & locked)),
            "delta_signed_mean_m": signed_mean,
            "delta_signed_median_m": signed_median,
            "delta_abs_mean_m": abs_mean,
            "delta_abs_p95_m": abs_p95,
            "delta_abs_max_m": max_abs,
            "bank_pull_risk_mean": float(np.mean(bank_pull_risk[zone])) if np.any(zone) else 0.0,
            "bank_pull_risk_p95": float(np.percentile(bank_pull_risk[zone], 95.0)) if np.any(zone) else 0.0,
        },
    }
