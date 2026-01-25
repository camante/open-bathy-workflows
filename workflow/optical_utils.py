#!/usr/bin/env python3
"""optical_utils.py - Shared optical processing utilities for SDB pipeline."""

import logging
from typing import Dict, Optional, Any
import numpy as np

log = logging.getLogger("sdb.optical_utils")
EPSILON = 1e-6


def estimate_linf_from_arrays(
    b02: np.ndarray, b03: np.ndarray, b04: np.ndarray, b08: np.ndarray,
    *, mask: Optional[np.ndarray] = None,
    nir_max: float = 0.03, bright_max: float = 0.15,
    percentile: float = 1.0, min_pixels: int = 1000,
) -> Dict[str, float]:
    """Estimate L-infinity from band arrays."""
    brightness = (b02 + b03 + b04) / 3.0
    deep_mask = (np.isfinite(b08) & (b08 < nir_max) & 
                 np.isfinite(brightness) & (brightness < bright_max))
    if mask is not None:
        deep_mask &= mask
    
    if np.count_nonzero(deep_mask) < min_pixels:
        return {}
    
    out = {}
    for name, arr in [("B02", b02), ("B03", b03), ("B04", b04), ("B08", b08)]:
        vals = arr[deep_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {}
        out[name] = max(0.0, float(np.percentile(vals, percentile)))
    return out


def normalize_linf_constants(d: Any) -> Dict[str, float]:
    """Normalize L-infinity dict to standard band keys."""
    defaults = {"B02": 0.0, "B03": 0.0, "B04": 0.0, "B08": 0.0}
    if not isinstance(d, dict):
        return defaults.copy()
    
    key_map = {
        "B02": "B02", "B2": "B02", "BLUE": "B02",
        "B03": "B03", "B3": "B03", "GREEN": "B03",
        "B04": "B04", "B4": "B04", "RED": "B04",
        "B08": "B08", "B8": "B08", "NIR": "B08",
    }
    
    out = {}
    for k, v in d.items():
        if k is None:
            continue
        normalized = key_map.get(str(k).strip().upper())
        if normalized:
            try:
                out[normalized] = float(v)
            except (TypeError, ValueError):
                pass
    
    for k, v in defaults.items():
        out.setdefault(k, v)
    return out


def compute_optical_features(
    b02: np.ndarray, b03: np.ndarray, b04: np.ndarray, b08: np.ndarray,
    *, l_inf: Optional[Dict[str, float]] = None, eps: float = EPSILON,
) -> Dict[str, np.ndarray]:
    """Compute optical features for SDB prediction."""
    if l_inf is None:
        l_inf = {"B02": 0.0, "B03": 0.0, "B04": 0.0, "B08": 0.0}
    
    b02_c = np.maximum(b02 - l_inf.get("B02", 0.0), eps)
    b03_c = np.maximum(b03 - l_inf.get("B03", 0.0), eps)
    b04_c = np.maximum(b04 - l_inf.get("B04", 0.0), eps)
    b08_c = np.maximum(b08 - l_inf.get("B08", 0.0), eps)
    
    log_b02 = np.log(b02_c)
    log_b03 = np.log(b03_c)
    log_b04 = np.log(b04_c)
    log_b08 = np.log(b08_c)
    
    # Fixed: use eps directly instead of sign*eps to avoid division by zero
    log_b03_safe = np.where(np.abs(log_b03) < eps, eps, log_b03)
    stumpf_idx = log_b02 / log_b03_safe
    
    return {
        "B02": b02, "B03": b03, "B04": b04, "B08": b08,
        "log_B02": log_b02, "log_B03": log_b03, "log_B04": log_b04, "log_B08": log_b08,
        "brightness": (b02 + b03 + b04) / 3.0,
        "B03_B02": b03_c / b02_c,
        "B04_B03": b04_c / b03_c,
        "nbri": (b03_c - b08_c) / (b03_c + b08_c + eps),
        "stumpf_idx": np.where(np.isfinite(stumpf_idx), stumpf_idx, 0.0),
    }


def normalize_depth_convention(
    depth_array: np.ndarray, source: str = "unknown",
    target: str = "positive_down"
) -> np.ndarray:
    """Normalize depth values to consistent convention (positive_down by default)."""
    arr = np.asarray(depth_array, dtype=np.float64)
    finite = np.isfinite(arr)
    
    if not np.any(finite):
        return arr
    
    median = float(np.nanmedian(arr[finite]))
    
    if target == "positive_down" and median < 0:
        log.debug(f"[{source}] Converting negative-down to positive-down")
        return np.abs(arr)
    elif target == "negative_down" and median > 0:
        log.debug(f"[{source}] Converting positive-down to negative-down")
        return -np.abs(arr)
    
    return arr
