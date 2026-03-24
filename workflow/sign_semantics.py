from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class NumericSignSummary:
    valid: int
    frac_neg: float
    frac_pos: float
    p01: float
    p50: float
    p99: float


def summarize_numeric_sign(values: Any) -> NumericSignSummary:
    arr = np.asarray(values, dtype=np.float64)
    m = np.isfinite(arr)
    if not np.any(m):
        return NumericSignSummary(0, np.nan, np.nan, np.nan, np.nan, np.nan)
    vals = arr[m]
    return NumericSignSummary(
        valid=int(vals.size),
        frac_neg=float(np.mean(vals < 0.0)),
        frac_pos=float(np.mean(vals > 0.0)),
        p01=float(np.percentile(vals, 1.0)),
        p50=float(np.percentile(vals, 50.0)),
        p99=float(np.percentile(vals, 99.0)),
    )


def raster_value_semantics(tags: Optional[Mapping[str, Any]]) -> str:
    tags = tags or {}
    value_type = str(tags.get("VALUE_TYPE", "") or "").strip().lower()
    depth_sign = str(tags.get("DEPTH_SIGN", "") or tags.get("SIGN_CONVENTION", "") or "").strip().lower()
    if value_type in {"elevation", "bed_elevation", "bathymetric_elevation", "bottom_elevation"}:
        return "absolute_elevation"
    if value_type == "depth":
        if depth_sign in {"positive_down", "positive-below-surface", "positive_below_surface", "positive-down"}:
            return "depth_positive_down"
        if depth_sign in {"negative_down", "negative-below-surface", "negative_below_surface", "negative-below-datum", "negative_below_datum", "negative-down"}:
            return "depth_negative_down"
        return "depth_unknown_sign"
    return "unknown"


def should_expect_negative_depth(tags: Optional[Mapping[str, Any]], default: bool = True) -> bool:
    semantics = raster_value_semantics(tags)
    if semantics == "absolute_elevation":
        return False
    if semantics == "depth_positive_down":
        return False
    if semantics == "depth_negative_down":
        return True
    return bool(default)


def maybe_warn_auto_depth_mode(logger, *, label: str, values: Any, mode: str, bed_elev_hint: bool = False) -> None:
    if logger is None:
        return
    if str(mode or "auto").strip().lower() != "auto":
        return
    stats = summarize_numeric_sign(values)
    if stats.valid <= 0:
        return
    if np.isfinite(stats.frac_neg) and stats.frac_neg >= 0.70:
        logger.info("[%s] Auto depth sign inference selected negative-down input (frac_neg=%.3f p50=%.3f p99=%.3f)", label, stats.frac_neg, stats.p50, stats.p99)
        return
    if np.isfinite(stats.frac_neg) and stats.frac_neg <= 0.05 and np.isfinite(stats.p50) and stats.p50 > 1.0:
        msg = (
            "[%s] Auto depth sign inference selected non-negative values (frac_neg=%.3f p50=%.3f p99=%.3f). "
            "This is fine for positive-down depth magnitudes, but if these are absolute elevations use the explicit bed-elevation path instead."
        )
        if bed_elev_hint:
            msg += " Consider setting soundings-mode=bed_elev explicitly."
        logger.warning(msg, label, stats.frac_neg, stats.p50, stats.p99)
        return
    logger.info("[%s] Auto depth sign inference kept input sign as-is (frac_neg=%.3f p50=%.3f p99=%.3f)", label, stats.frac_neg, stats.p50, stats.p99)


def semantics_from_soundings_mode(mode: str) -> str:
    mode_l = str(mode or "auto").strip().lower()
    if mode_l == "bed_elev":
        return "absolute_elevation"
    if mode_l == "depth_pos":
        return "depth_positive_down"
    if mode_l == "depth_neg":
        return "depth_negative_down"
    return "auto"


def semantics_from_depth_positive_down_flag(flag: Optional[bool]) -> str:
    if flag is True:
        return "depth_positive_down"
    if flag is False:
        return "depth_negative_down"
    return "depth_unknown_sign"
