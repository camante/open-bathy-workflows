from __future__ import annotations

from typing import Any

import numpy as np

SUPPORT_CLASS_CODES = {
    "missing": 0,
    "unsupported": 1,
    "low_support_scaffolded": 2,
    "stage_controlled": 3,
    "graph_backbone": 4,
    "xs_residual_only": 5,
    "xs_supported": 6,
    "anchored_interpolated": 7,
    "authoritative_bank_margin": 8,
    "authoritative_in_channel": 9,
    "authoritative_locked": 10,
}
SOLUTION_MODE_CODES = {
    "missing": 0,
    "unsupported": 1,
    "scaffolded": 2,
    "interpolated": 3,
    "graph": 4,
    "xs": 5,
    "authoritative": 6,
}
UNCERTAINTY_CLASS_CODES = {
    "missing": 0,
    "very_high": 1,
    "high": 2,
    "moderate": 3,
    "low": 4,
    "very_low": 5,
}
UNSUPPORTED_REGIME_CODES = {
    "missing": 0,
    "unsupported": 1,
    "low_support": 2,
    "scaffolded": 3,
    "supported": 4,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def graph_solution_confidence(row: Any) -> float:
    """Estimate a bounded confidence from commonly available support fields."""
    getter = row.get if hasattr(row, "get") else (lambda k, d=None: d)
    if bool(getter("authoritative_anchor_present", False)) or str(getter("support_class_canonical", "")).startswith("authoritative"):
        return 0.95
    if bool(getter("target_xs_realism_allowed", False)) or "xs" in str(getter("channel_support_class", "")):
        return 0.75
    conf = max(
        _as_float(getter("prediction_support_confidence", 0.0)),
        _as_float(getter("graph_confidence", 0.0)),
        _as_float(getter("bank_influence", 0.0)) * 0.45,
    )
    return float(np.clip(conf, 0.05, 0.95))


def uncertainty_class_from_confidence(confidence: float) -> str:
    c = _as_float(confidence, 0.0)
    if c >= 0.90:
        return "very_low"
    if c >= 0.70:
        return "low"
    if c >= 0.45:
        return "moderate"
    if c >= 0.20:
        return "high"
    return "very_high"


def support_class_code(name: object) -> int:
    return int(SUPPORT_CLASS_CODES.get(str(name), 0))


def solution_mode_code(name: object) -> int:
    return int(SOLUTION_MODE_CODES.get(str(name), 0))


def uncertainty_class_code(name: object) -> int:
    return int(UNCERTAINTY_CLASS_CODES.get(str(name), 0))


def unsupported_regime_code(name: object) -> int:
    return int(UNSUPPORTED_REGIME_CODES.get(str(name), 0))


__all__ = [
    "SUPPORT_CLASS_CODES",
    "SOLUTION_MODE_CODES",
    "UNCERTAINTY_CLASS_CODES",
    "UNSUPPORTED_REGIME_CODES",
    "graph_solution_confidence",
    "uncertainty_class_from_confidence",
    "support_class_code",
    "solution_mode_code",
    "uncertainty_class_code",
    "unsupported_regime_code",
]
