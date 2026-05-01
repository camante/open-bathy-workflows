from __future__ import annotations

"""Small deterministic section-tendency utilities used by legacy channel modules."""

from typing import Any

import numpy as np


def classify_section_tendency_family(width_m: float | None, support_class: str | None = None) -> str:
    support = str(support_class or "unsupported").lower()
    try:
        width = float(width_m)
    except Exception:
        width = float("nan")
    if "authoritative" in support:
        return "authoritative_supported"
    if "xs" in support:
        return "xs_supported"
    if np.isfinite(width) and width >= 80.0:
        return "wide_low_support"
    if np.isfinite(width) and width <= 20.0:
        return "narrow_low_support"
    return "general_low_support"


def compute_tendency_depth_fraction(
    width_m: float | None,
    family: str | None,
    support_class: str | None = None,
    *,
    component_class: str | None = None,
    reconciliation_confidence: float | None = None,
    support_distance_m: float | None = None,
    **_: Any,
) -> float:
    """Return a bounded inner-node fraction from thalweg toward bank elevations."""
    fam = str(family or "general_low_support")
    support = str(support_class or "unsupported").lower()
    if "authoritative" in support or fam == "authoritative_supported":
        base = 0.55
    elif "xs" in support or fam == "xs_supported":
        base = 0.50
    elif fam == "wide_low_support":
        base = 0.42
    elif fam == "narrow_low_support":
        base = 0.62
    else:
        base = 0.50
    try:
        conf = float(reconciliation_confidence)
        if np.isfinite(conf):
            base = (0.70 * base) + (0.30 * np.clip(conf, 0.0, 1.0))
    except Exception:
        pass
    return float(np.clip(base, 0.25, 0.80))


def build_section_from_thalweg(
    thalweg_z: float,
    width_m: float | None,
    family: str | None,
    depth_fraction: float,
    *,
    bank_caps: tuple[float | None, float | None] | None = None,
    **_: Any,
) -> tuple[float, float, dict[str, Any]]:
    """Build left/right inner elevations between thalweg and bank caps."""
    th = float(thalweg_z)
    frac = float(np.clip(depth_fraction, 0.0, 1.0))
    left_bank = right_bank = float("nan")
    if bank_caps is not None:
        try:
            left_bank = float(bank_caps[0])
        except Exception:
            left_bank = float("nan")
        try:
            right_bank = float(bank_caps[1])
        except Exception:
            right_bank = float("nan")
    if not np.isfinite(th):
        return float("nan"), float("nan"), {"family": str(family or "unknown"), "depth_fraction": frac, "valid": False}
    if not np.isfinite(left_bank):
        left_bank = th
    if not np.isfinite(right_bank):
        right_bank = th
    left_inner = th + frac * (left_bank - th)
    right_inner = th + frac * (right_bank - th)
    return float(left_inner), float(right_inner), {"family": str(family or "unknown"), "depth_fraction": frac, "valid": True}


__all__ = ["build_section_from_thalweg", "classify_section_tendency_family", "compute_tendency_depth_fraction"]
