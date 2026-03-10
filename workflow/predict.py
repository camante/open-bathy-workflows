#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py – SDB inference engine (scene-wide prediction).

Supports hybrid RF + Stumpf fallback, full uncertainty quantification,
Domain of Applicability enforcement, and per-pixel confidence/provenance rasters.
"""


import os as _os
# Force a headless-safe Matplotlib backend early (prevents TkAgg/Tkinter crashes under multiprocessing)
_os.environ.setdefault('MPLBACKEND', 'Agg')

import os
import sys
import json
import logging
import argparse
import subprocess
from process_utils import run_cmd
import shlex
from pathlib import Path
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
import joblib

# scipy is optional (keep the module importable even if scipy is absent)
try:
    from scipy.ndimage import median_filter as _scipy_median_filter
except Exception:
    _scipy_median_filter = None

# tqdm is optional
try:
    from tqdm import tqdm
    _TQDM = tqdm
except Exception:
    _TQDM = None

log = logging.getLogger("sdb.predict")


def _fmt_phys_scalar(value) -> str:
    """Format physics metadata values that may be scalar or nested dicts."""
    try:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float, np.floating)):
            v = float(value)
            return f"{v:.4f}" if np.isfinite(v) else "nan"
        if isinstance(value, dict):
            for k in ("value", "mean", "kd", "ku"):
                if k in value:
                    return _fmt_phys_scalar(value.get(k))
            # compact fallback for unexpected dict schema
            return json.dumps(value, sort_keys=True)
        return str(value)
    except Exception:
        return str(value)


# -----------------------------------------------------------------------------
# Optional modules
# -----------------------------------------------------------------------------

# Post-prediction alignment
ALIGNMENT_AVAILABLE: bool = False
_alignment_import_error: Optional[str] = None
try:
    import alignment
    ALIGNMENT_AVAILABLE = True
except ImportError as e:
    _alignment_import_error = f"ImportError: {e}"
except Exception as e:
    _alignment_import_error = f"{type(e).__name__}: {e}"
    log.error("Alignment module failed to load: %s", _alignment_import_error, exc_info=True)

# Physics-based SDB (Kim et al. 2024)
PHYSICS_MODULE_AVAILABLE: bool = False
_physics_import_error: Optional[str] = None
try:
    import physics_integration
    import bottom_physics
    PHYSICS_MODULE_AVAILABLE = True
except ImportError as e:
    _physics_import_error = f"ImportError: {e}"
except Exception as e:
    _physics_import_error = f"{type(e).__name__}: {e}"
    log.error("Physics module failed to load: %s", _physics_import_error, exc_info=True)

# Import standardized constants
try:
    from constants import NODATA_DEPTH, DEFAULT_TILE_SIZE, NUMERICAL_EPS
    NODATA_VAL: float = float(NODATA_DEPTH)
except ImportError:
    NODATA_VAL = -9999.0
    DEFAULT_TILE_SIZE = 1024
    NUMERICAL_EPS = 1e-10

FEATURE_KEYS_IMPLEMENTED: List[str] = [
    "B02", "B03", "B04", "B08",
    "log_B02", "log_B03", "log_B04", "log_B08",
    "brightness", "B03_B02", "B04_B03", "nbri", "stumpf_idx", "stumpf_depth",
]

# --- Import uncertainty module ---
UNCERTAINTY_STATUS_MSG: Optional[str] = None
UNCERTAINTY_MODULE_AVAILABLE: bool = False
try:
    from sdb_uncertainty import (
        UncertaintyParams,
        compute_model_variance,
        compute_optical_variance,
        compute_depth_variance,
        compute_extrapolation_variance,
        compute_input_variance,
        combine_variances,
    )
    UNCERTAINTY_MODULE_AVAILABLE = True
    # Do not log at import time; log once per run from predict_scene().
    UNCERTAINTY_STATUS_MSG = "sdb_uncertainty module available (full uncertainty enabled)"
except ImportError:
    UNCERTAINTY_MODULE_AVAILABLE = False
    UNCERTAINTY_STATUS_MSG = "sdb_uncertainty module not available (tree-std only)"


# -----------------------------------------------------------------------------
# Hybrid RF + Stumpf prediction
# -----------------------------------------------------------------------------

def compute_hybrid_prediction(
    rf_pred: np.ndarray,
    rf_std: np.ndarray,
    stumpf_idx: np.ndarray,
    stumpf_depth_fitted: np.ndarray,
    actual_training_max: float,
    stumpf_lr_coef: float = None,
    stumpf_lr_intercept: float = None,
    rf_std_threshold: float = 0.25,
    physics_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blend RF and Stumpf predictions for extrapolation beyond training depth.

    IMPORTANT: Only blend toward Stumpf when it predicts DEEPER than RF.
    All depth values here are positive magnitudes.
    """
    # Ensure actual_training_max is positive magnitude
    actual_training_max = abs(float(actual_training_max))
    
    kd_corrected = None
    if physics_params:
        kd_corrected = physics_params.get("kd_corrected")
        if kd_corrected:
            log.debug("Using geometry-corrected Kd=%s", _fmt_phys_scalar(kd_corrected))

    if stumpf_lr_coef is not None and stumpf_lr_intercept is not None:
        stumpf_physics = stumpf_lr_intercept + stumpf_lr_coef * stumpf_idx
        stumpf_physics = np.maximum(stumpf_physics, 0.0)
    else:
        stumpf_physics = stumpf_depth_fitted

    rf_uncertain = np.clip(rf_std / rf_std_threshold, 0.0, 1.0)

    rf_at_edge = np.clip(
        (rf_pred - actual_training_max * 0.7) / (actual_training_max * 0.3 + 0.1),
        0.0, 1.0
    )

    physics_deeper = stumpf_physics > rf_pred
    physics_extrapolation_amount = np.where(
        physics_deeper,
        np.clip((stumpf_physics - rf_pred) / 1.0, 0.0, 1.0),
        0.0
    )
    extrap_signal = rf_at_edge * physics_extrapolation_amount

    stumpf_weight = np.where(
        physics_deeper,
        np.maximum(rf_uncertain * 0.3, extrap_signal),
        0.0
    ).astype(np.float32)

    blended_pred = (1 - stumpf_weight) * rf_pred + stumpf_weight * stumpf_physics
    blended_pred = np.maximum(blended_pred, 0.0)

    rf_unc = np.maximum(rf_std, 0.10)
    extrap_distance = np.maximum(blended_pred - actual_training_max, 0)
    extrap_unc = 0.15 + 0.12 * extrap_distance

    blended_unc = np.where(
        stumpf_weight > 0.01,
        np.maximum(rf_unc, extrap_unc),
        rf_unc
    )
    blended_unc = np.maximum(blended_unc, 0.10)

    return blended_pred.astype(np.float32), blended_unc.astype(np.float32), stumpf_weight.astype(np.float32)


def compute_comprehensive_uncertainty(
    y_pred: np.ndarray,
    tree_std: np.ndarray,
    kd_values: np.ndarray,
    feature_arrays: Dict[str, np.ndarray],
    training_bounds: Dict[str, Dict[str, float]],
    max_training_depth: float,
    params: "UncertaintyParams" = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not UNCERTAINTY_MODULE_AVAILABLE:
        return tree_std, {"model_tree_std": tree_std ** 2}

    if params is None:
        params = UncertaintyParams()

    shape = y_pred.shape
    depth_abs = np.abs(y_pred)

    var_model = compute_model_variance(mode="calibrated", rf_tree_std=tree_std, params=params)
    if np.isscalar(var_model):
        var_model = np.full(shape, var_model, dtype=np.float32)

    var_optical = compute_optical_variance(kd_map=kd_values, depth_map=depth_abs, params=params)
    var_depth = compute_depth_variance(depth_map=depth_abs, params=params)
    var_extrap = compute_extrapolation_variance(
        depth_map=depth_abs,
        feature_arrays=feature_arrays,
        training_bounds=training_bounds,
        max_training_depth=max_training_depth,
        params=params,
    )
    var_input = compute_input_variance(s2_bands={"dummy": np.zeros(shape)}, params=params)

    var_total = combine_variances(var_model, var_optical, var_depth, var_extrap, var_input)
    var_total = np.clip(var_total, params.min_uncertainty_m ** 2, params.max_uncertainty_m ** 2)
    uncertainty = np.sqrt(var_total).astype(np.float32)

    components = {
        "model": var_model,
        "optical": var_optical,
        "depth": var_depth,
        "extrapolation": var_extrap,
        "input": var_input,
    }
    return uncertainty, components


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def estimate_linf_from_rasters(
    b02: np.ndarray,
    b03: np.ndarray,
    b04: np.ndarray,
    b08: np.ndarray,
    clear_water: Optional[np.ndarray] = None,
    *,
    cw_min: float = 0.5,
    deepwater_nir_max: float = 0.03,
    deepwater_bright_max: float = 0.15,
    percentile: float = 1.0,
) -> Dict[str, float]:
    brightness = (b02 + b03 + b04) / 3.0
    m = (
        np.isfinite(b08) & (b08 < float(deepwater_nir_max)) &
        np.isfinite(brightness) & (brightness < float(deepwater_bright_max))
    )
    if clear_water is not None:
        m &= np.isfinite(clear_water) & (clear_water >= float(cw_min))

    if np.count_nonzero(m) < 1000:
        return {}

    out = {}
    for name, arr in [("B02", b02), ("B03", b03), ("B04", b04), ("B08", b08)]:
        v = arr[m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {}
        out[name] = float(np.percentile(v, float(percentile)))

    for k in out:
        if out[k] < 0:
            out[k] = 0.0
    return out


def _compute_features_block(
    b02: np.ndarray, b03: np.ndarray, b04: np.ndarray, b08: np.ndarray, brightness: np.ndarray,
    stumpf_lr_model: Any = None, l_inf: Dict[str, float] = None
) -> Dict[str, np.ndarray]:
    eps = 1e-6
    if l_inf is None:
        l_inf = {"B02": 0.0, "B03": 0.0, "B04": 0.0, "B08": 0.0}

    b02_c = np.maximum(b02 - l_inf["B02"], eps)
    b03_c = np.maximum(b03 - l_inf["B03"], eps)
    b04_c = np.maximum(b04 - l_inf["B04"], eps)
    b08_c = np.maximum(b08 - l_inf["B08"], eps)

    log_b02 = np.log(b02_c); log_b03 = np.log(b03_c); log_b04 = np.log(b04_c); log_b08 = np.log(b08_c)

    log_b03_safe = np.where(np.abs(log_b03) < eps, eps, log_b03)
    stumpf_idx = log_b02 / log_b03_safe
    stumpf_idx[~np.isfinite(stumpf_idx)] = 0.0

    features: Dict[str, np.ndarray] = {
        "B02": b02, "B03": b03, "B04": b04, "B08": b08,
        "log_B02": log_b02, "log_B03": log_b03, "log_B04": log_b04, "log_B08": log_b08,
        "brightness": brightness,
        "B03_B02": b03_c / b02_c,
        "B04_B03": b04_c / b03_c,
        "nbri": (b03_c - b08_c) / (b03_c + b08_c + eps),
        "stumpf_idx": stumpf_idx,
    }

    if stumpf_lr_model is not None:
        flat = stumpf_idx.reshape(-1, 1)
        flat = np.nan_to_num(flat, nan=0.0)
        pred_depth = stumpf_lr_model.predict(flat)
        features["stumpf_depth"] = pred_depth.reshape(stumpf_idx.shape)

    return features


def _smooth_features(brightness, stumpf_idx, kernel_size):
    if kernel_size <= 1:
        return brightness, stumpf_idx
    k = int(kernel_size)
    if _scipy_median_filter is None:
        # This smoothing is optional (artifact suppression / QA). If scipy is missing,
        # return inputs unchanged rather than breaking prediction.
        if not getattr(_smooth_features, "_warned_no_scipy", False):
            log.warning("scipy not available; skipping median smoothing filter")
            _smooth_features._warned_no_scipy = True
        return brightness, stumpf_idx
    b_sm = _scipy_median_filter(brightness, size=k, mode="nearest")
    s_sm = _scipy_median_filter(stumpf_idx, size=k, mode="nearest")
    return b_sm, s_sm


def _normalize_linf_constants(d):
    defaults = {"B02": 0.0, "B03": 0.0, "B04": 0.0, "B08": 0.0}
    if not isinstance(d, dict):
        return defaults

    out = {}
    for k, v in d.items():
        if k is None:
            continue
        ku = str(k).strip().upper()
        if ku in {"BLUE", "B2", "BAND2"}: ku = "B02"
        elif ku in {"GREEN", "B3", "BAND3"}: ku = "B03"
        elif ku in {"RED", "B4", "BAND4"}: ku = "B04"
        elif ku in {"NIR", "B8", "BAND8"}: ku = "B08"

        if ku in defaults:
            try:
                out[ku] = float(v)
            except (TypeError, ValueError):
                pass  # non-numeric linf constant; skip

    for k, v in defaults.items():
        out.setdefault(k, v)
    return out


def _get_max_depth_from_meta(meta: Dict[str, Any], default: float = 20.0) -> Tuple[float, str]:
    def _try_float(v) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() == "auto":
            return None
        try:
            fv = float(v)
        except Exception:
            return None
        if not np.isfinite(fv) or fv <= 0:
            return None
        return fv

    for k in ("max_depth_sdb_final", "max_depth_sdb"):
        fv = _try_float(meta.get(k))
        if fv is not None:
            return fv, k

    diag = meta.get("max_depth_sdb_auto_diagnostics")
    if isinstance(diag, dict):
        depths = diag.get("depths_m")
        if isinstance(depths, dict):
            for kk in ("strict", "relaxed"):
                fv = _try_float(depths.get(kk))
                if fv is not None:
                    return fv, f"max_depth_sdb_auto_diagnostics.depths_m.{kk}"
        fv = _try_float(diag.get("max_depth_supported_m"))
        if fv is not None:
            return fv, "max_depth_sdb_auto_diagnostics.max_depth_supported_m"

    for k in ("max_depth_sdb_auto_strict", "max_depth_sdb_auto_relaxed"):
        fv = _try_float(meta.get(k))
        if fv is not None:
            return fv, k

    for k in (
        "max_depth_sdb_auto_m",
        "max_depth_sdb_auto",
        "auto_max_sdb_depth_m",
        "auto_max_depth_sdb_m",
        "max_depth_auto_sdb",
        "max_depth_auto",
        "max_depth_sdb_m",
        "max_depth",
    ):
        fv = _try_float(meta.get(k))
        if fv is not None:
            return fv, k

    return float(default), "default"


# -----------------------------------------------------------------------------
# Main prediction entry
# -----------------------------------------------------------------------------

def predict_scene(
    s2_paths: Dict[str, str],
    land_mask_path: str,
    rf_model_path: str,
    meta_json_path: str,
    out_path: str,
    stumpf_lr_path: Optional[str] = None,
    sdb_mode: str = "all_sdb",
    tile_size: int = DEFAULT_TILE_SIZE,
    s2_smooth_kernel: int = 0,
    enable_doa: bool = True,
    doa_threshold: float = 0.90,
    write_doa_score: bool = False,
    doa_soft_k: float = 3.0,
    linf_estimate_deepwater: bool = False,
    linf_deepwater_nir_max: float = 0.03,
    linf_deepwater_bright_max: float = 0.15,
    linf_percentile: float = 1.0,
    cw_min: Optional[float] = None,
    land_max: Optional[float] = None,
    land_mask_type: str = "auto",
    land_mask_water_val: int = 0,
    land_mask_invert: bool = False,
    land_mask_threshold: float = 0.5,
    depth_limit_mode: str = "optical",      # "optical", "training", "none"
    optical_depth_factor: float = 2.3,
    max_depth_hard_cap: float = 50.0,
    hybrid_mode: bool = True,
    diagnostics_dir: Optional[str] = None,
    high_unc_threshold: float = 2.0,
    # Post-prediction alignment
    align_mode: str = "median",
    align_tie_points_gpkg: Optional[str] = None,
    align_min_points: int = 150,
    align_depth_bins: str = "auto",
    align_source_priority: str = "atl24,atl03,xyz,other",
    align_extra_points: Optional[List[str]] = None,
    align_max_abs_residual_m_for_fit: Optional[float] = 10.0,
    # Guidance raster outputs
    write_confidence: bool = True,
    write_provenance: bool = True,
    min_confidence_threshold: float = 0.0,
):
    # Log optional module status once per run (avoid import-time logging).
    if UNCERTAINTY_STATUS_MSG:
        log.info("%s", UNCERTAINTY_STATUS_MSG)
    log.info("Loading RF model: %s", rf_model_path)
    rf_model = joblib.load(rf_model_path)
    stumpf_lr = joblib.load(stumpf_lr_path) if stumpf_lr_path and os.path.exists(stumpf_lr_path) else None

    stumpf_lr_coef = None
    stumpf_lr_intercept = None
    if stumpf_lr is not None:
        try:
            stumpf_lr_coef = float(stumpf_lr.coef_[0])
            stumpf_lr_intercept = float(stumpf_lr.intercept_)
            log.info("Stumpf LR: depth = %.3f + %.3f * stumpf_idx", stumpf_lr_intercept, stumpf_lr_coef)
        except Exception as e:
            log.warning("Could not extract Stumpf LR coefficients: %s", e)

    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta.get("feature_columns", [])

    physics_params = meta.get("physics", {})
    if physics_params:
        log.info(f"Loaded physics params: SZA={physics_params.get('sun_zenith_deg', 'N/A')}°")
        if physics_params.get("kd_corrected"):
            log.info("Geometry-corrected Kd=%s", _fmt_phys_scalar(physics_params.get('kd_corrected')))
        if physics_params.get("seagrass_detected"):
            log.warning("Seagrass signature detected in training scene")

    # DOA weights
    doa_weights: Dict[str, float] = {}
    meta_doa = meta.get("doa", {}) if isinstance(meta, dict) else {}
    meta_weights = meta_doa.get("weights", {}) if isinstance(meta_doa, dict) else {}
    if isinstance(meta_weights, dict) and meta_weights:
        try:
            doa_weights = {str(k): float(v) for k, v in meta_weights.items()}
            log.info("Loaded DOA weights from metadata (%s features).", len(doa_weights))
        except Exception:
            doa_weights = {}

    if not doa_weights:
        try:
            importances = getattr(rf_model, "feature_importances_", None)
            if importances is not None and len(importances) == len(feature_cols):
                doa_weights = {c: float(w) for c, w in zip(feature_cols, importances)}
            else:
                doa_weights = {c: 1.0 for c in feature_cols}
        except Exception:
            doa_weights = {c: 1.0 for c in feature_cols}

    max_depth_training, max_depth_src = _get_max_depth_from_meta(meta, default=20.0)

    actual_training_depth = meta.get("depth_stats_m", {}).get("max")
    if actual_training_depth is None:
        actual_training_depth = meta.get("training_bounds", {}).get("stumpf_depth", {}).get("max", max_depth_training)

    if depth_limit_mode == "none":
        max_depth = max_depth_hard_cap
        log.info("Depth limit mode=%s: predicting up to %.1fm hard cap", depth_limit_mode, max_depth)
    elif depth_limit_mode == "optical":
        max_depth = max_depth_hard_cap
        log.info("Depth limit mode=%s: using Kd-based per-pixel limit (factor=%s)", depth_limit_mode, optical_depth_factor)
        log.info("Max depth limit from metadata: %.2fm (source=%s)", max_depth_training, max_depth_src)
        if actual_training_depth:
            log.info("Actual training data depth range: 0 - %.2fm", actual_training_depth)
    else:
        max_depth = max_depth_training
        log.info("Depth limit mode=%s: max_depth_sdb = %.2fm (source=%s)", depth_limit_mode, max_depth, max_depth_src)

    linf_enabled = bool(meta.get("linf_enabled", False))
    linf_raw = meta.get("linf") or meta.get("l_inf_constants") or meta.get("l_inf") or meta.get("L_inf") or None

    l_inf_values: Dict[str, float] = {"B02": 0.0, "B03": 0.0, "B04": 0.0, "B08": 0.0}
    if linf_enabled:
        if not isinstance(linf_raw, dict) or not linf_raw:
            raise ValueError(
                "Model metadata has linf_enabled=True but no L∞ constants were found. "
                "Re-train with raster-based L∞ estimation so constants are stored in model_meta.json."
            )
        l_inf_values = _normalize_linf_constants(linf_raw)
        if linf_estimate_deepwater:
            log.warning("--linf-estimate-deepwater is deprecated and ignored (strict L∞ consistency).")
    else:
        l_inf_values = _normalize_linf_constants({})
        if isinstance(linf_raw, dict) and linf_raw:
            try:
                if any(float(v) != 0 for v in linf_raw.values()):
                    log.warning("linf_enabled=False but NON-ZERO L∞ constants are present in metadata; ignoring.")
            except Exception:
                log.debug("ignored", exc_info=True)  # linf_raw check failed
        log.info("L_inf disabled; using zeros.")

    training_bounds = meta.get("training_bounds", {})
    if enable_doa and training_bounds:
        log.info("Enforcing Domain of Applicability using %s feature bounds.", len(training_bounds))
    elif not enable_doa:
        log.warning("Domain of Applicability disabled. Model will extrapolate to unknown areas.")
    else:
        log.warning("No training bounds found. Extrapolation is possible.")

    implemented_set = set(FEATURE_KEYS_IMPLEMENTED)
    missing = [c for c in feature_cols if c not in implemented_set]
    if missing:
        raise RuntimeError(f"predict.py missing implementation for: {missing}")

    if cw_min is None or land_max is None:
        if sdb_mode == "lakes":
            cw_min = 0.8 if cw_min is None else cw_min
            land_max = 0.1 if land_max is None else land_max
        else:
            cw_min = 0.5 if cw_min is None else cw_min
            land_max = 0.5 if land_max is None else land_max

    with rasterio.open(s2_paths["B02"]) as src_ref:
        profile = src_ref.profile.copy()
        height, width = src_ref.height, src_ref.width

    profile.update(
        count=1,
        dtype=rasterio.float32,
        nodata=NODATA_VAL,
        compress="DEFLATE",
        tiled=True,
        blockxsize=tile_size,
        blockysize=tile_size,
    )

    out_unc_path = str(Path(out_path).with_name(Path(out_path).stem + "_uncertainty.tif"))
    doa_path = str(Path(out_path).with_name(Path(out_path).stem + "_doa_score.tif")) if write_doa_score else None
    conf_path = str(Path(out_path).with_name(Path(out_path).stem + "_confidence.tif")) if write_confidence else None
    prov_path = str(Path(out_path).with_name(Path(out_path).stem + "_provenance.tif")) if write_provenance else None

    if min_confidence_threshold > 0.0 and not write_confidence:
        log.warning(
            "[PREDICT] min_confidence_threshold=%.2f has no effect because write_confidence=False; "
            "depth raster will not be masked. Pass write_confidence=True to enable per-pixel masking.",
            min_confidence_threshold,
        )

    srcs: Dict[str, rasterio.DatasetReader] = {}
    try:
        srcs["B02"] = rasterio.open(s2_paths["B02"])
        srcs["B03"] = rasterio.open(s2_paths["B03"])
        srcs["B04"] = rasterio.open(s2_paths["B04"])
        srcs["B08"] = rasterio.open(s2_paths["B08"])

        # Ensure ancillary rasters are on the exact same grid as the S2 bands.
        # This prevents window-read shape mismatches when masks differ in CRS/extent/resolution.
        ref = srcs["B02"]
        ref_kwargs = dict(
            crs=ref.crs,
            transform=ref.transform,
            height=ref.height,
            width=ref.width,
            resampling=Resampling.nearest,
        )

        _cwm_src = rasterio.open(s2_paths["CLEAR_WATER"])
        _brt_src = rasterio.open(s2_paths["BRIGHTNESS"])

        # Robustness: the land mask is an external artifact (often produced by waffles).
        # If it is missing, create a conservative all-water mask aligned to the S2 grid
        # so prediction can proceed (and log loudly so users notice).
        land_mask_path = str(land_mask_path)
        if not os.path.exists(land_mask_path):
            log.warning(
                f"[PREDICT][MASK] land mask missing: {land_mask_path}. "
                "Creating aligned all-water fallback mask (water=0)."
            )
            _dst_dir = os.path.dirname(land_mask_path)
            if _dst_dir:
                os.makedirs(_dst_dir, exist_ok=True)
            prof = ref.profile.copy()
            prof.update(driver="GTiff", dtype="uint8", count=1, nodata=None, compress="DEFLATE", tiled=True)
            fallback = np.zeros((ref.height, ref.width), dtype=np.uint8)
            with rasterio.open(land_mask_path, "w", **prof) as _dst:
                _dst.write(fallback, 1)

        _lnd_src = rasterio.open(land_mask_path)

        srcs["_CWM_SRC"] = _cwm_src
        srcs["_BRT_SRC"] = _brt_src
        srcs["_LND_SRC"] = _lnd_src

        srcs["CWM"] = WarpedVRT(_cwm_src, **ref_kwargs)
        srcs["BRT"] = WarpedVRT(_brt_src, **ref_kwargs)
        srcs["LND"] = WarpedVRT(_lnd_src, **ref_kwargs)


        with ExitStack() as stack:
            dst_depth = stack.enter_context(rasterio.open(out_path, "w", **profile))
            dst_unc = stack.enter_context(rasterio.open(out_unc_path, "w", **profile))
            dst_doa = stack.enter_context(rasterio.open(doa_path, "w", **profile)) if doa_path else None

            dst_conf = stack.enter_context(rasterio.open(conf_path, "w", **profile)) if conf_path else None
            prov_profile = profile.copy()
            prov_profile.update(dtype=rasterio.uint8, nodata=0)
            dst_prov = stack.enter_context(rasterio.open(prov_path, "w", **prov_profile)) if prov_path else None

            dst_depth.update_tags(UNITS="meters", CONVENTION="negative-down", MAX_DEPTH_SDB=str(max_depth))
            dst_unc.update_tags(UNITS="meters", DESC="1-Sigma Uncertainty", MAX_DEPTH_SDB=str(max_depth))
            if dst_doa is not None:
                dst_doa.update_tags(UNITS="unitless", DESC="Weighted Domain of Applicability score (0..1)")
            if dst_conf is not None:
                dst_conf.update_tags(
                    UNITS="unitless",
                    DESC="Per-pixel guidance confidence (0=low/nodata, 1=high): DOA x optical_quality x uncertainty_quality",
                    MIN_CONFIDENCE_THRESHOLD=str(min_confidence_threshold),
                )
            if dst_prov is not None:
                dst_prov.update_tags(
                    DESC="Provenance code: 0=nodata/masked, 1=predicted",
                )

            windows = [
                Window(c, r, min(tile_size, width - c), min(tile_size, height - r))
                for r in range(0, height, tile_size)
                for c in range(0, width, tile_size)
            ]

            funnel = {
                "pixels_total": 0,
                "finite_optical": 0,
                "cw_pass": 0,
                "land_pass": 0,
                "base_valid": 0,
                "doa_pass": 0,
                "predicted": 0,
                "pixels_with_uncertainty": 0,
                "pixels_high_uncertainty": 0,
            }

            doa_valid_total = 0
            doa_pass_total = 0
            doa_reject_total = 0
            doa_reject_penalty_sum: Dict[str, float] = {c: 0.0 for c in feature_cols}

            unc_sample: List[float] = []
            unc_sample_max = 200000

            lt_mode = str(land_mask_type).lower().strip()

            it = windows
            if _TQDM is not None:
                it = _TQDM(windows, desc="Inference", unit="tile")

            for window in it:
                b02 = srcs["B02"].read(1, window=window).astype(np.float32)
                b03 = srcs["B03"].read(1, window=window).astype(np.float32)
                b04 = srcs["B04"].read(1, window=window).astype(np.float32)
                b08 = srcs["B08"].read(1, window=window).astype(np.float32)
                cwm = srcs["CWM"].read(1, window=window)
                brt = srcs["BRT"].read(1, window=window).astype(np.float32)
                lnd = srcs["LND"].read(1, window=window)

                funnel["pixels_total"] += int(b02.size)

                finite_opt = (
                    np.isfinite(b02) & (b02 > 0) &
                    np.isfinite(b03) & (b03 > 0) &
                    np.isfinite(b04) & (b04 > 0) &
                    np.isfinite(b08) & (b08 > 0) &
                    np.isfinite(brt)
                )
                cw_ok = np.isfinite(cwm) & (cwm >= float(cw_min))

                if lt_mode in ("water_only", "land_binary", "binary"):
                    wv = int(land_mask_water_val)
                    land_ok = (lnd != wv) if bool(land_mask_invert) else (lnd == wv)
                elif lt_mode in ("land_probability", "probability"):
                    thr = float(land_mask_threshold) if land_mask_threshold is not None else (float(land_max) if land_max is not None else 0.5)
                    land_ok = (lnd <= thr)
                else:
                    lm = float(land_max) if land_max is not None else 0.5
                    land_ok = ((~np.isfinite(lnd)) | (lnd <= lm))

                funnel["finite_optical"] += int(finite_opt.sum())
                funnel["cw_pass"] += int((finite_opt & cw_ok).sum())
                funnel["land_pass"] += int((finite_opt & cw_ok & land_ok).sum())

                valid_mask = finite_opt & cw_ok & land_ok
                funnel["base_valid"] += int(valid_mask.sum())

                out_block = np.full(b02.shape, NODATA_VAL, dtype=np.float32)
                out_unc_block = np.full(b02.shape, NODATA_VAL, dtype=np.float32)
                doa_score_block = np.full(b02.shape, NODATA_VAL, dtype=np.float32) if dst_doa else None
                confidence_block = np.full(b02.shape, NODATA_VAL, dtype=np.float32) if dst_conf else None
                provenance_block = np.zeros(b02.shape, dtype=np.uint8) if dst_prov else None

                if np.any(valid_mask):
                    try:
                        feats_dict = _compute_features_block(b02, b03, b04, b08, brt, stumpf_lr_model=stumpf_lr, l_inf=l_inf_values)
                        if s2_smooth_kernel > 1:
                            b_sm, s_sm = _smooth_features(feats_dict["brightness"], feats_dict["stumpf_idx"], s2_smooth_kernel)
                            feats_dict["brightness"] = b_sm
                            feats_dict["stumpf_idx"] = s_sm

                        feature_stack = [feats_dict[col][valid_mask] for col in feature_cols]
                        X_block = np.vstack(feature_stack).T.astype(np.float32)

                        domain_mask_local = np.ones(X_block.shape[0], dtype=bool)
                        doa_score_local = np.ones(X_block.shape[0], dtype=np.float32)

                        if enable_doa and training_bounds:
                            num = np.zeros(X_block.shape[0], dtype=np.float32)
                            den = np.float32(0.0)

                            for j, col_name in enumerate(feature_cols):
                                if col_name not in training_bounds:
                                    continue
                                b_min = float(training_bounds[col_name]["min"])
                                b_max = float(training_bounds[col_name]["max"])
                                w = float(doa_weights.get(col_name, 1.0))
                                if w <= 0:
                                    continue

                                x = X_block[:, j]
                                rng = max(b_max - b_min, 1e-12)
                                dist = np.zeros_like(x, dtype=np.float32)

                                lo = x < b_min
                                hi = x > b_max
                                if np.any(lo):
                                    dist[lo] = (b_min - x[lo]).astype(np.float32) / np.float32(rng)
                                if np.any(hi):
                                    dist[hi] = (x[hi] - b_max).astype(np.float32) / np.float32(rng)

                                score_f = np.exp(-np.float32(doa_soft_k) * dist).astype(np.float32)
                                num += np.float32(w) * score_f
                                den += np.float32(w)

                            if den > 0:
                                doa_score_local = num / den
                                domain_mask_local = (doa_score_local >= np.float32(doa_threshold))

                                doa_valid_total += int(X_block.shape[0])
                                doa_pass_total += int(domain_mask_local.sum())
                                doa_reject_total += int((~domain_mask_local).sum())

                        funnel["doa_pass"] += int(domain_mask_local.sum())

                        # Guard sklearn prediction against NaN/Inf feature rows that can survive
                        # earlier masks (e.g., NaNs fail comparisons and slip through DOA logic).
                        if np.any(domain_mask_local):
                            finite_domain_mask = np.all(np.isfinite(X_block[domain_mask_local]), axis=1)
                            if not np.all(finite_domain_mask):
                                bad_n = int((~finite_domain_mask).sum())
                                funnel.setdefault("doa_nonfinite_reject", 0)
                                funnel["doa_nonfinite_reject"] += bad_n
                                idx_local = np.flatnonzero(domain_mask_local)
                                domain_mask_local[idx_local[~finite_domain_mask]] = False
                                funnel["doa_pass"] -= bad_n

                        if doa_score_block is not None:
                            doa_score_block[valid_mask] = doa_score_local.astype(np.float32)

                        if np.any(domain_mask_local):
                            X_domain = X_block[domain_mask_local]
                            y_pred_domain = rf_model.predict(X_domain).astype(np.float32)

                            # tree std
                            tree_preds = np.zeros((len(rf_model.estimators_), len(X_domain)), dtype=np.float32)
                            for i, est in enumerate(rf_model.estimators_):
                                tree_preds[i] = est.predict(X_domain).astype(np.float32)
                            y_std_domain = np.std(tree_preds, axis=0).astype(np.float32)

                            y_pred_domain = np.maximum(y_pred_domain, 0.0)

                            # crude Kd estimate for optical limit / uncertainty
                            b02_domain = b02[valid_mask][domain_mask_local]
                            b03_domain = b03[valid_mask][domain_mask_local]
                            b02_safe = np.maximum(b02_domain, 0.001)
                            ratio = b03_domain / b02_safe
                            kd_est = (0.02 + 0.12 * np.clip(ratio, 0.5, 3.0)).astype(np.float32)

                            stumpf_depth_col = feature_cols.index("stumpf_depth") if "stumpf_depth" in feature_cols else None
                            stumpf_idx_col = feature_cols.index("stumpf_idx") if "stumpf_idx" in feature_cols else None

                            actual_training_max = meta.get("depth_stats_m", {}).get("max", 3.8) or 3.8

                            if hybrid_mode and stumpf_depth_col is not None and stumpf_idx_col is not None:
                                stumpf_depth_domain = np.maximum(X_domain[:, stumpf_depth_col], 0.0)
                                stumpf_idx_domain = X_domain[:, stumpf_idx_col]

                                y_pred_domain, y_unc_hybrid, blend_weight = compute_hybrid_prediction(
                                    rf_pred=y_pred_domain,
                                    rf_std=y_std_domain,
                                    stumpf_idx=stumpf_idx_domain,
                                    stumpf_depth_fitted=stumpf_depth_domain,
                                    actual_training_max=float(actual_training_max),
                                    stumpf_lr_coef=stumpf_lr_coef,
                                    stumpf_lr_intercept=stumpf_lr_intercept,
                                    physics_params=physics_params,
                                )
                            else:
                                y_unc_hybrid = None

                            # apply limits
                            if depth_limit_mode == "optical":
                                optical_max_depth = optical_depth_factor / np.maximum(kd_est, 0.02)
                                optical_max_depth = np.minimum(optical_max_depth, max_depth_hard_cap)
                                beyond = y_pred_domain > optical_max_depth
                                y_pred_domain[beyond] = np.nan
                            elif depth_limit_mode == "training":
                                y_pred_domain[y_pred_domain > float(max_depth)] = np.nan

                            # uncertainty
                            feature_arrays_domain = {col: X_domain[:, j] for j, col in enumerate(feature_cols)}
                            comprehensive_unc, _ = compute_comprehensive_uncertainty(
                                y_pred=y_pred_domain,
                                tree_std=y_std_domain,
                                kd_values=kd_est,
                                feature_arrays=feature_arrays_domain,
                                training_bounds=training_bounds if training_bounds else {},
                                max_training_depth=float(max_depth_training) if max_depth_training else 20.0,
                            )
                            y_unc_domain = np.maximum(comprehensive_unc, y_unc_hybrid) if y_unc_hybrid is not None else comprehensive_unc

                            y_pred_neg = -1.0 * y_pred_domain

                            final_pixels = np.full(int(np.sum(valid_mask)), NODATA_VAL, dtype=np.float32)
                            final_unc = np.full(int(np.sum(valid_mask)), NODATA_VAL, dtype=np.float32)

                            good = np.isfinite(y_pred_neg)
                            funnel["predicted"] += int(good.sum())

                            _unc_good = y_unc_domain[good]
                            funnel["pixels_with_uncertainty"] += int(_unc_good.size)
                            funnel["pixels_high_uncertainty"] += int((_unc_good > float(high_unc_threshold)).sum())

                            # reservoir sample
                            if _unc_good.size:
                                if len(unc_sample) < unc_sample_max:
                                    n_add = min(int(_unc_good.size), int(unc_sample_max - len(unc_sample)))
                                    if n_add > 0:
                                        idx = np.random.choice(_unc_good.size, size=n_add, replace=False) if _unc_good.size > n_add else np.arange(_unc_good.size)
                                        unc_sample.extend([float(x) for x in _unc_good[idx]])

                            final_pixels[domain_mask_local] = np.where(good, y_pred_neg, NODATA_VAL)
                            final_unc[domain_mask_local] = np.where(good, y_unc_domain, NODATA_VAL)

                            out_block[valid_mask] = final_pixels
                            out_unc_block[valid_mask] = final_unc

                            # Confidence: DOA × optical quality × uncertainty quality.
                            # optical_q normalises CWM above cw_min to [0,1].
                            # unc_q decays from 1 toward 0 with uncertainty (half-weight at 1.5 m).
                            if confidence_block is not None:
                                cw_min_f = float(cw_min) if cw_min is not None else 0.0
                                cw_range = max(1.0 - cw_min_f, 0.01)
                                optical_q_valid = np.clip(
                                    (cwm[valid_mask].astype(np.float32) - cw_min_f) / cw_range,
                                    0.0, 1.0,
                                )
                                optical_q_domain = optical_q_valid[domain_mask_local]
                                doa_q_domain = doa_score_local[domain_mask_local]
                                unc_q_domain = (1.0 / (1.0 + y_unc_domain / 1.5)).astype(np.float32)
                                conf_domain = doa_q_domain * optical_q_domain * unc_q_domain

                                conf_pixels = np.full(int(np.sum(valid_mask)), NODATA_VAL, dtype=np.float32)
                                conf_pixels[domain_mask_local] = np.where(good, conf_domain, NODATA_VAL)
                                confidence_block[valid_mask] = conf_pixels

                    except Exception:
                        log.exception("Prediction failed on tile")

                # FINAL hard enforcement (negative-down)
                if max_depth is not None and np.isfinite(max_depth):
                    bad = (np.abs(out_block) > float(max_depth)) | (out_block > 0.0)
                    if np.any(bad):
                        out_block[bad] = NODATA_VAL
                        out_unc_block[bad] = NODATA_VAL
                        if doa_score_block is not None:
                            doa_score_block[bad] = NODATA_VAL
                        if confidence_block is not None:
                            confidence_block[bad] = NODATA_VAL
                        if provenance_block is not None:
                            provenance_block[bad] = 0

                # Confidence threshold: nodata-mask depth cells below min threshold
                if min_confidence_threshold > 0.0 and confidence_block is not None:
                    low_conf = (
                        np.isfinite(confidence_block) &
                        (confidence_block != NODATA_VAL) &
                        (confidence_block < float(min_confidence_threshold))
                    )
                    if np.any(low_conf):
                        out_block[low_conf] = NODATA_VAL
                        out_unc_block[low_conf] = NODATA_VAL
                        if doa_score_block is not None:
                            doa_score_block[low_conf] = NODATA_VAL
                        confidence_block[low_conf] = NODATA_VAL
                        if provenance_block is not None:
                            provenance_block[low_conf] = 0

                # Provenance: 1 = valid predicted cell, 0 = nodata/masked (background)
                if provenance_block is not None:
                    valid_pred = np.isfinite(out_block) & (out_block != NODATA_VAL)
                    provenance_block[valid_pred] = 1

                dst_depth.write(np.ascontiguousarray(out_block, dtype=np.float32), 1, window=window)
                dst_unc.write(np.ascontiguousarray(out_unc_block, dtype=np.float32), 1, window=window)
                if dst_doa is not None and doa_score_block is not None:
                    dst_doa.write(np.ascontiguousarray(doa_score_block, dtype=np.float32), 1, window=window)
                if dst_conf is not None and confidence_block is not None:
                    dst_conf.write(np.ascontiguousarray(confidence_block, dtype=np.float32), 1, window=window)
                if dst_prov is not None and provenance_block is not None:
                    dst_prov.write(np.ascontiguousarray(provenance_block, dtype=np.uint8), 1, window=window)

        # Diagnostic summary
        log.info("Prediction summary:")
        total = max(1, funnel['pixels_total'])
        log.info("  Total pixels:      %10d", funnel['pixels_total'])
        log.info("  Finite optical:    %10d  (%4.1f%%)", funnel['finite_optical'], 100*funnel['finite_optical']/total)
        log.info("  Passed clear-water:%10d  (%4.1f%%)", funnel['cw_pass'], 100*funnel['cw_pass']/total)
        log.info("  Passed land mask:  %10d  (%4.1f%%)", funnel['land_pass'], 100*funnel['land_pass']/total)
        log.info("  Base valid:        %10d  (%4.1f%%)", funnel['base_valid'], 100*funnel['base_valid']/total)
        log.info("  Passed DOA:        %10d  (%4.1f%%)", funnel['doa_pass'], 100*funnel['doa_pass']/total)
        log.info("  Final predicted:   %10d  (%4.1f%%)", funnel['predicted'], 100*funnel['predicted']/total)

        if funnel['predicted'] == 0:
            log.error("Zero pixels predicted. Possible causes: "
                      "cw_min too high; land mask masking water; missing S2 data; DOA rejecting everything (try --no-doa)")
        elif funnel['predicted'] < 0.01 * total:
            log.warning("Very few pixels predicted (%d). Consider reviewing mask settings.", funnel['predicted'])

        if conf_path:
            log.info("Confidence raster: %s", conf_path)
            if min_confidence_threshold > 0:
                log.info("Applied min_confidence_threshold=%.2f — cells below this are nodata in depth raster", min_confidence_threshold)
        if prov_path:
            log.info("Provenance raster: %s  (0=nodata, 1=predicted)", prov_path)

        # report
        try:
            report_dir = Path(diagnostics_dir) if diagnostics_dir else Path(out_path).parent
            report_dir.mkdir(parents=True, exist_ok=True)
            rep = {"predict": {"funnel": funnel, "high_unc_threshold_m": float(high_unc_threshold),
                               "confidence_path": conf_path, "provenance_path": prov_path,
                               "min_confidence_threshold": min_confidence_threshold}}
            with open(report_dir / "predict_report.json", "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
        except Exception:
            log.debug("Failed to write predict_report.json", exc_info=True)

    finally:
        for s in srcs.values():
            try:
                s.close()
            except Exception:
                log.debug("ignored", exc_info=True)  # close error

    log.info("Finished. Depth: %s", out_path)
    return {"status": "ok"}


def reproject_to_nad83(src_path: str, dst_path: str):
    cmd = f"gdalwarp -overwrite -t_srs EPSG:4269 -r bilinear -of GTiff {shlex.quote(src_path)} {shlex.quote(dst_path)}"
    log.info("Reprojecting with: %s", cmd)
    try:
        run_cmd(shlex.split(cmd), check=True)
    except Exception:
        log.warning("gdalwarp failed.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--land-mask-path", required=True)
    parser.add_argument("--out-tif", required=True)
    parser.add_argument("--sdb-mode", default="all_sdb")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--smooth-kernel", type=int, default=0)
    parser.add_argument("--nad83", action="store_true")
    parser.add_argument("--no-doa", action="store_true", help="Disable Domain of Applicability checks")
    parser.add_argument("--doa-threshold", type=float, default=0.90)
    parser.add_argument("--doa-soft-k", type=float, default=3.0)
    parser.add_argument("--write-doa-score", action="store_true")

    parser.add_argument("--land-mask-type", default="auto", help="water_only, land_binary, land_probability, auto")
    parser.add_argument("--land-mask-water-val", type=int, default=0)
    parser.add_argument("--land-mask-invert", action="store_true")
    parser.add_argument("--land-mask-threshold", type=float, default=0.5)

    parser.add_argument("--high-unc-threshold", type=float, default=2.0, help="Meters; counts pixels above this threshold in report")

    args = parser.parse_args(argv)

    # Configure logging ONLY here (no import-time side effects)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    s2_dir = Path(args.s2_dir)
    model_dir = Path(args.model_dir)
    rf_path = model_dir / "rf_model.pkl"
    meta_path = model_dir / "model_meta.json"
    stumpf_path = str(model_dir / "stumpf_lr.pkl") if (model_dir / "stumpf_lr.pkl").exists() else None

    s2_paths = {
        "B02": str(s2_dir / "B02_10m.tif"),
        "B03": str(s2_dir / "B03_10m.tif"),
        "B04": str(s2_dir / "B04_10m.tif"),
        "B08": str(s2_dir / "B08_10m.tif"),
        "CLEAR_WATER": str(s2_dir / "CLEAR_WATER_MASK_10m.tif"),
        "BRIGHTNESS": str(s2_dir / "BRIGHTNESS_10m.tif"),
    }

    predict_scene(
        s2_paths=s2_paths,
        land_mask_path=args.land_mask_path,
        rf_model_path=str(rf_path),
        meta_json_path=str(meta_path),
        out_path=args.out_tif,
        stumpf_lr_path=stumpf_path,
        sdb_mode=args.sdb_mode,
        tile_size=args.tile_size,
        s2_smooth_kernel=args.smooth_kernel,
        enable_doa=not args.no_doa,
        doa_threshold=args.doa_threshold,
        write_doa_score=args.write_doa_score,
        doa_soft_k=args.doa_soft_k,
        land_mask_type=args.land_mask_type,
        land_mask_water_val=args.land_mask_water_val,
        land_mask_invert=args.land_mask_invert,
        land_mask_threshold=args.land_mask_threshold,
        high_unc_threshold=args.high_unc_threshold,
    )

    if args.nad83:
        reproject_to_nad83(args.out_tif, str(Path(args.out_tif).with_name(Path(args.out_tif).stem + "_NAD83.tif")))

    return 0


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sys.exit(main())
