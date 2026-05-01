"""Small helpers for explicit raster/sign/value semantics.

These helpers are intentionally deterministic and metadata-driven.  They do not
infer a workflow route or mutate products; they only normalize semantic labels
and summarize sampled values for runtime contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class NumericSignSummary:
    valid: int
    frac_neg: float
    frac_pos: float
    frac_zero: float
    p01: float
    p50: float
    p99: float


def _normalize_semantics(value: str | None) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "elevation": "absolute_elevation",
        "absolute": "absolute_elevation",
        "absolute_elevation": "absolute_elevation",
        "bed_elevation": "absolute_elevation",
        "bottom_elevation": "absolute_elevation",
        "navd88_elevation": "absolute_elevation",
        "depth_positive": "depth_positive_down",
        "depth_positive_down": "depth_positive_down",
        "positive_down": "depth_positive_down",
        "down_positive": "depth_positive_down",
        "down_is_positive": "depth_positive_down",
        "depth_pos": "depth_positive_down",
        "depth_negative": "depth_negative_down",
        "depth_negative_down": "depth_negative_down",
        "negative_down": "depth_negative_down",
        "down_negative": "depth_negative_down",
        "down_is_negative": "depth_negative_down",
        "depth_neg": "depth_negative_down",
        "auto": "auto",
        "unknown": "unknown",
    }
    return aliases.get(raw, raw or "unknown")


def raster_value_semantics(tags: Mapping[str, Any] | None) -> str:
    tags = tags or {}
    for key in (
        "VALUE_SEMANTICS",
        "value_semantics",
        "VALUE_TYPE",
        "value_type",
        "SEMANTICS",
        "semantics",
        "DEPTH_SIGN",
        "depth_sign",
        "POSITIVE_DIRECTION",
        "positive_direction",
    ):
        value = tags.get(key)
        if value not in (None, ""):
            norm = _normalize_semantics(str(value))
            if norm in {"absolute_elevation", "depth_positive_down", "depth_negative_down"}:
                return norm
            if norm == "positive_down":
                return "depth_positive_down"
            if norm == "negative_down":
                return "depth_negative_down"
    return "unknown"


def should_expect_negative_depth(tags: Mapping[str, Any] | None, *, default: bool = True) -> bool:
    tags = tags or {}
    semantics = raster_value_semantics(tags)
    if semantics == "depth_negative_down":
        return True
    if semantics == "depth_positive_down":
        return False
    for key in ("DEPTH_SIGN", "depth_sign", "POSITIVE_DIRECTION", "positive_direction"):
        value = _normalize_semantics(str(tags.get(key, "")))
        if value == "depth_negative_down":
            return True
        if value == "depth_positive_down":
            return False
    return bool(default)


def semantics_from_depth_positive_down_flag(flag: bool | None) -> str:
    if flag is True:
        return "depth_positive_down"
    if flag is False:
        return "depth_negative_down"
    return "depth_unknown_sign"


def semantics_from_soundings_mode(mode: str | None) -> str:
    key = str(mode or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"auto", "infer", "inferred", "unknown"}:
        return "auto"
    if key in {"bed", "bed_elev", "bed_elevation", "elevation", "absolute", "absolute_elevation", "z", "navd88"}:
        return "absolute_elevation"
    if key in {"depth_pos", "depth_positive", "depth_positive_down", "positive_down", "down_positive"}:
        return "depth_positive_down"
    if key in {"depth_neg", "depth_negative", "depth_negative_down", "negative_down", "down_negative"}:
        return "depth_negative_down"
    return "unknown"


def summarize_numeric_sign(values: Any) -> NumericSignSummary:
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return NumericSignSummary(0, 0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan"))
    neg = int(np.count_nonzero(arr < 0.0))
    pos = int(np.count_nonzero(arr > 0.0))
    zero = int(np.count_nonzero(arr == 0.0))
    p01, p50, p99 = np.nanpercentile(arr, [1, 50, 99])
    return NumericSignSummary(
        valid=n,
        frac_neg=float(neg / n),
        frac_pos=float(pos / n),
        frac_zero=float(zero / n),
        p01=float(p01),
        p50=float(p50),
        p99=float(p99),
    )


__all__ = [
    "NumericSignSummary",
    "raster_value_semantics",
    "semantics_from_depth_positive_down_flag",
    "semantics_from_soundings_mode",
    "should_expect_negative_depth",
    "summarize_numeric_sign",
]
