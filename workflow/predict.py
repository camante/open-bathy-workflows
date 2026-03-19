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
from pyproj import Transformer
from source_guidance_contract import build_sdb_source_contract
try:
    from pyproj.exceptions import ProjError
except (ImportError, AttributeError):
    class ProjError(Exception):
        pass

# scipy is optional (keep the module importable even if scipy is absent)
try:
    from scipy.ndimage import median_filter as _scipy_median_filter
except (ImportError, AttributeError, OSError, SyntaxError):
    _scipy_median_filter = None
try:
    from scipy.spatial import cKDTree as _cKDTree
except (ImportError, AttributeError, OSError, SyntaxError):
    _cKDTree = None

# tqdm is optional
try:
    from tqdm import tqdm
    _TQDM = tqdm
except (ImportError, AttributeError, OSError, SyntaxError):
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
    except (TypeError, ValueError):
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
except (AttributeError, OSError, SyntaxError) as e:
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
except (AttributeError, OSError, SyntaxError) as e:
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


def _core_optical_hard_mask(
    X_block: np.ndarray,
    feature_cols: List[str],
    training_bounds: Dict[str, Dict[str, float]],
    core_features: List[str],
    margin_frac: float,
) -> np.ndarray:
    """Require core optical features to remain safely inside the observed manifold."""
    if X_block.size == 0:
        return np.zeros((0,), dtype=bool)
    keep = np.ones(X_block.shape[0], dtype=bool)
    margin_frac = float(max(margin_frac, 0.0))
    feature_index = {name: i for i, name in enumerate(feature_cols)}
    for feat in core_features or []:
        j = feature_index.get(feat)
        bounds = training_bounds.get(feat) if isinstance(training_bounds, dict) else None
        if j is None or not bounds:
            continue
        try:
            lo = float(bounds.get("min"))
            hi = float(bounds.get("max"))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if hi < lo:
            lo, hi = hi, lo
        vals = X_block[:, j]
        rng = max(hi - lo, 1e-12)
        inner_lo = lo + margin_frac * rng
        inner_hi = hi - margin_frac * rng
        if inner_hi < inner_lo:
            mid = 0.5 * (lo + hi)
            inner_lo = mid
            inner_hi = mid
        keep &= np.isfinite(vals) & (vals >= inner_lo) & (vals <= inner_hi)
    return keep


def _stumpf_support_mask(
    X_block: np.ndarray,
    feature_cols: List[str],
    training_bounds: Dict[str, Dict[str, float]],
    guidance_meta: Dict[str, Any],
) -> np.ndarray:
    """Conservative support gate for physics-guided prediction.

    This gate is intentionally based on *stumpf_depth* support, not stumpf_idx support.
    The downstream guidance logic is expressed in depth magnitude space, so falling back
    to index-space bounds would mix incompatible units and can silently over-reject.
    """
    if X_block.size == 0:
        return np.zeros((0,), dtype=bool)
    keep = np.ones(X_block.shape[0], dtype=bool)
    feature_index = {name: i for i, name in enumerate(feature_cols)}

    if "stumpf_depth" not in feature_index:
        return keep

    feat = "stumpf_depth"
    bounds = training_bounds.get(feat, {}) if isinstance(training_bounds, dict) else {}
    support_lo = guidance_meta.get("stumpf_support_min_m", bounds.get("min"))
    support_hi = guidance_meta.get("stumpf_support_max_m", bounds.get("max"))
    try:
        support_lo = float(support_lo)
        support_hi = float(support_hi)
    except (TypeError, ValueError):
        return keep
    if not (np.isfinite(support_lo) and np.isfinite(support_hi)):
        return keep
    if support_hi < support_lo:
        support_lo, support_hi = support_hi, support_lo

    low_support = bool(guidance_meta.get("low_support", True))
    envelope = float(guidance_meta.get("stumpf_envelope_m", 0.5))
    tol = min(envelope, 0.15 if low_support else 0.30)

    vals = X_block[:, feature_index[feat]]
    keep &= np.isfinite(vals) & (vals >= (support_lo - tol)) & (vals <= (support_hi + tol))

    if "stumpf_idx" in feature_index:
        try:
            idx_lo = float(guidance_meta.get("stumpf_idx_support_min", training_bounds.get("stumpf_idx", {}).get("min", -np.inf)))
            idx_hi = float(guidance_meta.get("stumpf_idx_support_max", training_bounds.get("stumpf_idx", {}).get("max", np.inf)))
        except (TypeError, ValueError):
            idx_lo, idx_hi = -np.inf, np.inf
        if np.isfinite(idx_lo) and np.isfinite(idx_hi):
            if idx_hi < idx_lo:
                idx_lo, idx_hi = idx_hi, idx_lo
            idx_vals = X_block[:, feature_index["stumpf_idx"]]
            idx_tol = 0.02 if low_support else 0.05
            idx_rng = max(idx_hi - idx_lo, 1e-12)
            keep &= np.isfinite(idx_vals) & (idx_vals >= (idx_lo - idx_tol * idx_rng)) & (idx_vals <= (idx_hi + idx_tol * idx_rng))

    if "brightness" in feature_index and isinstance(training_bounds, dict) and "brightness" in training_bounds:
        bb = training_bounds.get("brightness", {})
        try:
            blo = float(bb.get("min"))
            bhi = float(bb.get("max"))
        except (TypeError, ValueError):
            blo = bhi = None
        if blo is not None and bhi is not None and np.isfinite(blo) and np.isfinite(bhi):
            vb = X_block[:, feature_index["brightness"]]
            brng = max(bhi - blo, 1e-12)
            bright_margin = 0.02 if low_support else 0.05
            keep &= np.isfinite(vb) & (vb >= (blo - bright_margin * brng)) & (vb <= (bhi + bright_margin * brng))

    return keep


def _build_support_kdtree(guidance_meta: Dict[str, Any], raster_crs) -> tuple[Any, Optional[Transformer], float, bool]:
    pts = guidance_meta.get("support_points_lonlat", []) if isinstance(guidance_meta, dict) else []
    max_dist = float(guidance_meta.get("max_train_point_dist_m", 0.0) or 0.0)
    gate_required = bool(max_dist > 0)
    anchor_support_good = bool(guidance_meta.get("anchor_support_good", False)) if isinstance(guidance_meta, dict) else False
    if max_dist <= 0 or _cKDTree is None:
        return None, None, max_dist, gate_required
    if not pts:
        log.warning("Support-distance DOA gate requested (max_train_point_dist_m=%.1f m) but no support_points_lonlat were provided.", max_dist)
        return None, None, max_dist, gate_required
    try:
        arr = np.asarray(pts, dtype=np.float64)
    except (TypeError, ValueError):
        log.warning("Support-distance DOA gate requested but support_points_lonlat could not be coerced to float; failing closed.")
        return None, None, max_dist, gate_required
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        log.warning("Support-distance DOA gate requested but support_points_lonlat are malformed; failing closed.")
        return None, None, max_dist, gate_required
    try:
        tfm = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        xs, ys = tfm.transform(arr[:, 0], arr[:, 1])
    except (ProjError, TypeError, ValueError):
        log.warning("Failed to transform support points into raster CRS for support-distance DOA gating; failing closed.", exc_info=True)
        return None, None, max_dist, gate_required
    xy = np.column_stack([xs, ys])
    m = np.isfinite(xy).all(axis=1)
    xy = xy[m]
    if xy.shape[0] == 0:
        log.warning("Support-distance DOA gate requested but no finite support points remained after CRS transform; failing closed.")
        return None, None, max_dist, gate_required
    try:
        # Dense authoritative-anchor runs should not fail closed because the
        # support proxy was thinned too aggressively. Widen the support reach
        # when the support cloud is still sparse after thinning.
        if anchor_support_good and xy.shape[0] < 2000:
            max_dist = max(max_dist, 500.0)
        if anchor_support_good and xy.shape[0] < 1000:
            max_dist = max(max_dist, 750.0)
        return _cKDTree(xy), tfm, max_dist, gate_required
    except (TypeError, ValueError):
        log.warning("Failed to build support KDTree for support-distance DOA gating; failing closed.", exc_info=True)
        return None, None, max_dist, gate_required


def _support_distance_metrics(window: Window, valid_mask: np.ndarray, ds_transform, support_tree: Any, max_dist_m: float, gate_required: bool, min_neighbors: int = 1) -> tuple[np.ndarray, np.ndarray]:
    n = int(valid_mask.sum())
    if not np.any(valid_mask):
        return np.ones(n, dtype=bool), np.ones(n, dtype=np.float32)
    if max_dist_m <= 0:
        return np.ones(n, dtype=bool), np.ones(n, dtype=np.float32)
    if support_tree is None:
        if gate_required:
            return np.zeros(n, dtype=bool), np.zeros(n, dtype=np.float32)
        return np.ones(n, dtype=bool), np.ones(n, dtype=np.float32)
    rows, cols = np.where(valid_mask)
    abs_rows = rows + int(window.row_off)
    abs_cols = cols + int(window.col_off)
    if hasattr(rasterio, "transform") and hasattr(rasterio.transform, "xy"):
        xs, ys = rasterio.transform.xy(ds_transform, abs_rows, abs_cols, offset="center")
    else:
        a = float(getattr(ds_transform, "a", 1.0))
        e = float(getattr(ds_transform, "e", -1.0))
        c = float(getattr(ds_transform, "c", 0.0))
        f = float(getattr(ds_transform, "f", 0.0))
        xs = c + (abs_cols.astype(np.float64) + 0.5) * a
        ys = f + (abs_rows.astype(np.float64) + 0.5) * e
    q = np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])
    k = max(int(min_neighbors), 1)
    dists, _ = support_tree.query(q, k=k)
    if k == 1:
        nearest = np.asarray(dists, dtype=np.float64)
        kth = nearest
    else:
        dists = np.asarray(dists, dtype=np.float64)
        nearest = dists[:, 0]
        kth = dists[:, -1]
    keep = np.isfinite(kth) & (kth <= float(max_dist_m))
    # Strong decay: full trust near retained support, rapidly decays to zero toward max distance.
    ratio = np.clip(np.asarray(nearest, dtype=np.float32) / max(float(max_dist_m), 1e-6), 0.0, 1.0)
    weight = (1.0 - ratio) ** 2
    weight[~np.isfinite(weight)] = 0.0
    weight[~keep] = 0.0
    return keep, weight.astype(np.float32)


def _edge_trust_weight(window: Window, valid_mask: np.ndarray, full_shape: tuple[int, int], halo_px: int) -> np.ndarray:
    n = int(valid_mask.sum())
    if halo_px <= 0 or not np.any(valid_mask):
        return np.ones(n, dtype=np.float32)
    rows, cols = np.where(valid_mask)
    abs_rows = rows + int(window.row_off)
    abs_cols = cols + int(window.col_off)
    h, w = int(full_shape[0]), int(full_shape[1])
    d_edge = np.minimum.reduce([
        abs_rows.astype(np.int64),
        abs_cols.astype(np.int64),
        (h - 1 - abs_rows).astype(np.int64),
        (w - 1 - abs_cols).astype(np.int64),
    ]).astype(np.float32)
    return np.clip(d_edge / float(max(halo_px, 1)), 0.0, 1.0).astype(np.float32)


def _export_sdb_guide_points(
    depth_path: str,
    confidence_path: Optional[str],
    guidance_weight_path: Optional[str],
    uncertainty_path: Optional[str],
    out_gpkg: str,
    *,
    min_confidence: float = 0.15,
    max_points: int = 5000,
) -> None:
    """Export spatially thinned SDB pseudo-soundings as a GeoPackage.

    Each point carries depth, confidence, guidance_weight, uncertainty, and
    provenance metadata.  Points are subordinate to authoritative survey data
    and must be treated as confidence-weighted interpolation guidance only.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    NODATA_VAL = -9999.0

    with rasterio.open(depth_path) as ds:
        depth = ds.read(1).astype("float32")
        nodata = ds.nodata if ds.nodata is not None else NODATA_VAL
        valid = np.isfinite(depth) & (depth != nodata)
        transform = ds.transform
        crs = ds.crs

    conf = None
    if confidence_path and os.path.exists(confidence_path):
        with rasterio.open(confidence_path) as ds:
            conf = ds.read(1).astype("float32")
        valid &= np.isfinite(conf) & (conf != NODATA_VAL) & (conf >= min_confidence)

    gw = None
    if guidance_weight_path and os.path.exists(guidance_weight_path):
        with rasterio.open(guidance_weight_path) as ds:
            gw = ds.read(1).astype("float32")
        valid &= np.isfinite(gw) & (gw != NODATA_VAL) & (gw > 0.05)

    unc = None
    if uncertainty_path and os.path.exists(uncertainty_path):
        with rasterio.open(uncertainty_path) as ds:
            unc = ds.read(1).astype("float32")

    rows, cols = np.where(valid)
    if rows.size == 0:
        log.info("No valid SDB pixels above confidence threshold for guide-point export.")
        return

    # Spatial thinning: subsample to max_points using stride
    step = max(int(np.sqrt(rows.size / max(max_points, 1))), 1)
    sel = np.arange(0, rows.size, step, dtype=int)
    rows, cols = rows[sel], cols[sel]

    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    data = {
        "depth_m": depth[rows, cols].astype("float32"),
        "artifact_role": ["sdb_guide_point"] * len(rows),
        "authoritative": [False] * len(rows),
        "provenance": ["SDB"] * len(rows),
    }
    if conf is not None:
        data["confidence"] = conf[rows, cols].astype("float32")
    if gw is not None:
        data["guidance_weight"] = gw[rows, cols].astype("float32")
    if unc is not None:
        data["uncertainty_m"] = unc[rows, cols].astype("float32")

    gdf = gpd.GeoDataFrame(
        data,
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=crs,
    )
    out_p = Path(out_gpkg)
    if out_p.exists():
        out_p.unlink()
    gdf.to_file(out_gpkg, driver="GPKG")
    log.info("Exported %d SDB guide points to %s", len(gdf), out_gpkg)


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


def compute_physics_guided_prediction(
    rf_pred_residual: np.ndarray,
    rf_std: np.ndarray,
    stumpf_idx: np.ndarray,
    stumpf_depth_fitted: np.ndarray,
    correction_alpha: float,
    residual_clip_m: float,
    stumpf_lr_coef: float = None,
    stumpf_lr_intercept: float = None,
    physics_params: Optional[Dict[str, Any]] = None,
    baseline_source: str = "stumpf_depth",
    stumpf_envelope_m: float = 0.60,
    support_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physics-first prediction for guidance-only SDB.

    RF predicts only a bounded residual about a Stumpf/optical baseline so the output
    preserves monotonic optical ordering in low-support regions.
    """
    baseline_source = str(baseline_source or "stumpf_depth").lower()
    if baseline_source == "stumpf_lr" and stumpf_lr_coef is not None and stumpf_lr_intercept is not None:
        stumpf_physics = stumpf_lr_intercept + stumpf_lr_coef * stumpf_idx
        stumpf_physics = np.maximum(stumpf_physics, 0.0)
    else:
        stumpf_physics = np.maximum(stumpf_depth_fitted, 0.0)

    correction_alpha = float(np.clip(correction_alpha, 0.0, 1.0))
    residual_clip_m = float(max(residual_clip_m, 0.05))
    stumpf_envelope_m = float(max(stumpf_envelope_m, residual_clip_m, 0.10))
    rf_pred_residual = np.clip(rf_pred_residual, -residual_clip_m, residual_clip_m)
    if support_weight is None:
        effective_alpha = np.full(rf_pred_residual.shape, correction_alpha, dtype=np.float32)
    else:
        effective_alpha = correction_alpha * np.clip(np.asarray(support_weight, dtype=np.float32), 0.0, 1.0)
    residual_applied = effective_alpha * rf_pred_residual
    guided_pred = np.maximum(stumpf_physics + residual_applied, 0.0)
    guided_pred = np.clip(
        guided_pred,
        np.maximum(stumpf_physics - stumpf_envelope_m, 0.0),
        stumpf_physics + stumpf_envelope_m,
    )

    baseline_weight = (1.0 - effective_alpha).astype(np.float32)
    guided_unc = np.maximum(rf_std, 0.10 + 0.25 * np.abs(residual_applied) + 0.10 * baseline_weight)
    return guided_pred.astype(np.float32), guided_unc.astype(np.float32), baseline_weight.astype(np.float32)



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
        flat = stumpf_idx.ravel()
        flat = np.nan_to_num(flat, nan=0.0)
        # IsotonicRegression expects 1D; LinearRegression expects 2D
        if hasattr(stumpf_lr_model, "increasing"):
            # IsotonicRegression
            pred_depth = stumpf_lr_model.predict(flat)
        else:
            # LinearRegression or similar
            pred_depth = stumpf_lr_model.predict(flat.reshape(-1, 1)).ravel()
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
    def _source_family(name: Any) -> str:
        src = str(name or "").strip().lower()
        if not src:
            return ""
        if "combined" in src:
            return "combined"
        if "physics" in src or "kd" in src:
            return "physics"
        if "rmse" in src or "support" in src:
            return "rmse"
        if "training_p95" in src or "p95" in src:
            return "training_p95"
        return src

    def _try_float(v) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() == "auto":
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(fv) or fv <= 0:
            return None
        return fv

    def _lookup(candidates):
        for k in candidates:
            fv = _try_float(meta.get(k))
            if fv is not None:
                return fv, k
        return None

    for k in ("max_depth_sdb_final", "max_depth_sdb"):
        fv = _try_float(meta.get(k))
        if fv is not None:
            return fv, k

    final_source = meta.get("max_depth_sdb_final_source")
    opts = meta.get("max_depth_options") if isinstance(meta, dict) else {}
    selected_source = opts.get("selected_source") if isinstance(opts, dict) else None
    family = _source_family(final_source or selected_source)

    family_order = {
        "physics": ["max_depth_sdb_auto_physics", "max_depth_sdb_combined", "max_depth_sdb_auto"],
        "combined": ["max_depth_sdb_combined", "max_depth_sdb_auto_physics", "max_depth_sdb_auto"],
        "rmse": ["max_depth_sdb_auto", "max_depth_sdb_combined", "max_depth_sdb_auto_physics"],
    }
    found = _lookup(family_order.get(family, []))
    if found is not None:
        return found

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

    found = _lookup([
        "max_depth_sdb_combined",
        "max_depth_sdb_auto_physics",
        "max_depth_sdb_auto_m",
        "max_depth_sdb_auto",
        "auto_max_sdb_depth_m",
        "auto_max_depth_sdb_m",
        "max_depth_auto_sdb",
        "max_depth_auto",
        "max_depth_sdb_m",
        "max_depth",
    ])
    if found is not None:
        return found

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
            if hasattr(stumpf_lr, "coef_") and hasattr(stumpf_lr, "intercept_"):
                stumpf_lr_coef = float(stumpf_lr.coef_[0])
                stumpf_lr_intercept = float(stumpf_lr.intercept_)
                log.info("Stumpf LR: depth = %.3f + %.3f * stumpf_idx", stumpf_lr_intercept, stumpf_lr_coef)
            elif stumpf_lr.__class__.__name__ == "IsotonicRegression":
                log.info("Using monotonic Stumpf baseline model (IsotonicRegression); linear coefficients are not applicable.")
            else:
                log.info("Using non-linear Stumpf baseline model (%s); linear coefficients are not applicable.", stumpf_lr.__class__.__name__)
        except (AttributeError, TypeError, ValueError, IndexError) as e:
            log.warning("Could not inspect Stumpf baseline model: %s", e)

    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta.get("feature_columns", [])

    physics_params = meta.get("physics", {})
    if physics_params:
        log.info("Loaded physics params: SZA=%s°", physics_params.get('sun_zenith_deg', 'N/A'))
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
        except (TypeError, ValueError):
            doa_weights = {}

    if not doa_weights:
        try:
            importances = getattr(rf_model, "feature_importances_", None)
            if importances is not None and len(importances) == len(feature_cols):
                doa_weights = {c: float(w) for c, w in zip(feature_cols, importances)}
            else:
                doa_weights = {c: 1.0 for c in feature_cols}
        except (AttributeError, TypeError, ValueError):
            doa_weights = {c: 1.0 for c in feature_cols}

    doa_meta = meta.get("doa", {}) if isinstance(meta, dict) else {}
    try:
        if "threshold_default" in doa_meta:
            doa_threshold = float(doa_meta["threshold_default"])
    except (TypeError, ValueError):
        pass
    try:
        if "soft_k_default" in doa_meta:
            doa_soft_k = float(doa_meta["soft_k_default"])
    except (TypeError, ValueError):
        pass

    model_tier = int(meta.get("model_tier", 1)) if isinstance(meta, dict) else 1
    if model_tier >= 2:
        log.info("[Tier %d] DOA relaxed: threshold=%.2f, soft_k=%.1f",
                 model_tier, doa_threshold, doa_soft_k)

    max_depth_training, max_depth_src = _get_max_depth_from_meta(meta, default=20.0)

    depth_stats = meta.get("depth_stats_m", {})
    if not isinstance(depth_stats, dict) or not depth_stats:
        depth_stats = (meta.get("train_report", {}) or {}).get("depth_stats_m", {})
    if not isinstance(depth_stats, dict):
        depth_stats = {}
    actual_training_depth_min = depth_stats.get("min")
    actual_training_depth_max = depth_stats.get("max")
    if actual_training_depth_max is None:
        actual_training_depth_max = meta.get("training_bounds", {}).get("stumpf_depth", {}).get("max", max_depth_training)

    if depth_limit_mode == "none":
        max_depth = max_depth_hard_cap
        log.info("Depth limit mode=%s: predicting up to %.1fm hard cap", depth_limit_mode, max_depth)
    elif depth_limit_mode == "optical":
        max_depth = max_depth_hard_cap
        log.info("Depth limit mode=%s: using Kd-based per-pixel limit (factor=%s)", depth_limit_mode, optical_depth_factor)
        log.info("Max depth limit from metadata: %.2fm (source=%s)", max_depth_training, max_depth_src)
        if actual_training_depth_max is not None:
            if actual_training_depth_min is None:
                actual_training_depth_min = 0.0
            log.info("Actual training data depth range: %.2f - %.2fm", float(actual_training_depth_min), float(actual_training_depth_max))
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
            except (TypeError, ValueError) as _exc:
                log.debug("Suppressed: %s", _exc, exc_info=True)
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
    guidance_weight_path = str(Path(out_path).with_name(Path(out_path).stem + "_guidance_weight.tif"))
    trusted_interior_path = str(Path(out_path).with_name(Path(out_path).stem + "_trusted_interior.tif"))
    admissibility_path = str(Path(out_path).with_name(Path(out_path).stem + "_admissibility.tif"))
    regime_class_path = str(Path(out_path).with_name(Path(out_path).stem + "_regime_class.tif"))

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
                "[PREDICT][MASK] land mask missing: %s. "
                "Creating aligned all-water fallback mask (water=0).",
                land_mask_path,
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
            dst_guidance = stack.enter_context(rasterio.open(guidance_weight_path, "w", **profile))
            trusted_profile = profile.copy()
            trusted_profile.update(dtype=rasterio.uint8, nodata=0)
            dst_trusted = stack.enter_context(rasterio.open(trusted_interior_path, "w", **trusted_profile))
            admiss_profile = profile.copy()
            admiss_profile.update(dtype=rasterio.uint8, nodata=0)
            dst_admiss = stack.enter_context(rasterio.open(admissibility_path, "w", **admiss_profile))
            prov_profile = profile.copy()
            prov_profile.update(dtype=rasterio.uint8, nodata=0)
            dst_prov = stack.enter_context(rasterio.open(prov_path, "w", **prov_profile)) if prov_path else None

            dst_depth.update_tags(UNITS="meters", CONVENTION="negative-down", MAX_DEPTH_SDB=str(max_depth), ROLE="interpolation_guidance", AUTHORITATIVE="false", CUDEM_INTENT="guidance_only", USE_NOTE="Non-authoritative guidance surface for interpolation in unsupported gaps.")
            dst_unc.update_tags(UNITS="meters", DESC="1-Sigma Uncertainty", MAX_DEPTH_SDB=str(max_depth), ROLE="interpolation_guidance", AUTHORITATIVE="false", CUDEM_INTENT="guidance_only")
            if dst_doa is not None:
                dst_doa.update_tags(UNITS="unitless", DESC="Weighted Domain of Applicability score (0..1)", ROLE="interpolation_guidance", CUDEM_INTENT="guidance_only")
            if dst_conf is not None:
                dst_conf.update_tags(
                    UNITS="unitless",
                    DESC="Per-pixel guidance confidence (0=low/nodata, 1=high): DOA x optical_quality x uncertainty_quality",
                    ROLE="interpolation_guidance", AUTHORITATIVE="false", CUDEM_INTENT="guidance_only",
                    MIN_CONFIDENCE_THRESHOLD=str(min_confidence_threshold),
                )
            dst_guidance.update_tags(UNITS="unitless", DESC="Guidance weight for interpolation use (0..1): support_distance x AOI_interior x baseline_weight", ROLE="interpolation_guidance", AUTHORITATIVE="false", CUDEM_INTENT="guidance_only")
            dst_trusted.update_tags(DESC="Trusted interior mask for interpolation use (1=trusted interior, 0=edge/unsupported)", ROLE="interpolation_guidance", AUTHORITATIVE="false", CUDEM_INTENT="guidance_only")
            dst_admiss.update_tags(DESC="SDB admissibility mask: 1=optical domain where SDB guidance is valid (water, not cloud/land/deep), 0=inadmissible", ROLE="interpolation_guidance", AUTHORITATIVE="false", CUDEM_INTENT="guidance_only")
            if dst_prov is not None:
                dst_prov.update_tags(
                    DESC="Provenance code: 0=nodata/masked, 1=predicted",
                    ROLE="interpolation_guidance", AUTHORITATIVE="false", CUDEM_INTENT="guidance_only",
                )

            guidance_meta_global = meta.get("physics_guidance", {}) if isinstance(meta, dict) else {}
            support_tree, _support_tfm, max_support_dist_m, support_gate_required = _build_support_kdtree(guidance_meta_global, srcs["B02"].crs)
            min_support_neighbors = max(int(guidance_meta_global.get("min_support_neighbors", 1) or 1), 1)
            trusted_halo_px = int(guidance_meta_global.get("trusted_halo_px", 128) or 128)
            anchor_support_good = bool(guidance_meta_global.get("anchor_support_good", False))
            if anchor_support_good:
                min_support_neighbors = 1
                trusted_halo_px = min(trusted_halo_px, 96) if trusted_halo_px > 0 else 96
                if cw_min is not None:
                    cw_min = min(float(cw_min), 0.35)
            if support_tree is not None and max_support_dist_m > 0:
                log.info("Using CUDEM guidance support gate: max_train_point_dist_m=%.1f m, min_support_neighbors=%d (%d support pts)", max_support_dist_m, min_support_neighbors, support_tree.n)
            elif support_gate_required:
                log.warning("CUDEM support-distance DOA gate is active but no usable support tree was built; prediction will fail closed for unsupported pixels.")
            log.info("CUDEM alignment: outputs are guidance-only and non-authoritative; use guidance_weight/trusted_interior to control interpolation influence.")

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
                it = _TQDM(windows, desc="Inference", unit="tile", disable=not sys.stderr.isatty(), leave=False)

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
                guidance_block = np.full(b02.shape, NODATA_VAL, dtype=np.float32)
                trusted_block = np.zeros(b02.shape, dtype=np.uint8)
                admiss_block = np.zeros(b02.shape, dtype=np.uint8)
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
                        support_weight_local = np.ones(X_block.shape[0], dtype=np.float32)
                        edge_weight_local = np.ones(X_block.shape[0], dtype=np.float32)
                        guidance_meta = meta.get("physics_guidance", {}) if isinstance(meta, dict) else {}
                        strict_min_doa = float(max(doa_threshold, guidance_meta.get("min_doa", doa_threshold)))
                        core_optical_features = guidance_meta.get("core_optical_features", ["stumpf_idx", "stumpf_depth", "brightness", "B02", "B03", "B04", "B08"])
                        hard_optical_margin_frac = float(guidance_meta.get("hard_optical_margin_frac", 0.05))

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
                                domain_mask_local = (doa_score_local >= np.float32(strict_min_doa))
                                hard_optical_mask = _core_optical_hard_mask(
                                    X_block,
                                    feature_cols,
                                    training_bounds,
                                    list(core_optical_features),
                                    hard_optical_margin_frac,
                                )
                                support_mask = _stumpf_support_mask(
                                    X_block,
                                    feature_cols,
                                    training_bounds,
                                    guidance_meta,
                                )
                                distance_mask, support_weight_local = _support_distance_metrics(
                                    window,
                                    valid_mask,
                                    srcs["B02"].transform,
                                    support_tree,
                                    max_support_dist_m,
                                    support_gate_required,
                                    min_neighbors=min_support_neighbors,
                                )
                                edge_weight_local = _edge_trust_weight(window, valid_mask, (height, width), trusted_halo_px)
                                domain_mask_local &= hard_optical_mask & support_mask & distance_mask

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
                            actual_training_max = meta.get("depth_stats_m", {}).get("max", 3.8) or 3.8
                            prediction_mode = str(meta.get("prediction_mode", "stumpf_residual")).lower()
                            correction_alpha = float(guidance_meta.get("correction_alpha", 0.35))
                            residual_clip_m = float(meta.get("residual_clip_m", guidance_meta.get("residual_clip_m", 0.5)))
                            baseline_source = str(guidance_meta.get("baseline_source", "stumpf_depth"))
                            stumpf_envelope_m = float(guidance_meta.get("stumpf_envelope_m", max(residual_clip_m, 0.3)))
                            if prediction_mode == "stumpf_residual" and hasattr(rf_model, "predict_residual"):
                                y_pred_domain = np.asarray(rf_model.predict_residual(X_domain), dtype=np.float32)
                            else:
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

                            if stumpf_depth_col is not None and stumpf_idx_col is not None:
                                stumpf_depth_domain = np.maximum(X_domain[:, stumpf_depth_col], 0.0)
                                stumpf_idx_domain = X_domain[:, stumpf_idx_col]

                                if prediction_mode == "stumpf_residual":
                                    support_domain = None
                                    try:
                                        support_domain = (support_weight_local[domain_mask_local] * edge_weight_local[domain_mask_local]).astype(np.float32)
                                    except (TypeError, ValueError, AttributeError) as exc:
                                        log.debug("Unable to compute support_domain for stumpf_residual: %s", exc)
                                        support_domain = None
                                    y_pred_domain, y_unc_hybrid, blend_weight = compute_physics_guided_prediction(
                                        rf_pred_residual=y_pred_domain,
                                        rf_std=y_std_domain,
                                        stumpf_idx=stumpf_idx_domain,
                                        stumpf_depth_fitted=stumpf_depth_domain,
                                        correction_alpha=correction_alpha,
                                        residual_clip_m=residual_clip_m,
                                        stumpf_lr_coef=stumpf_lr_coef,
                                        stumpf_lr_intercept=stumpf_lr_intercept,
                                        physics_params=physics_params,
                                        baseline_source=baseline_source,
                                        stumpf_envelope_m=stumpf_envelope_m,
                                        support_weight=support_domain,
                                    )
                                elif hybrid_mode:
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
                            else:
                                y_unc_hybrid = None

                            # apply limits
                            if stumpf_depth_col is not None and model_tier < 2:
                                stumpf_support = np.maximum(X_domain[:, stumpf_depth_col], 0.0)
                                y_pred_domain = np.clip(
                                    y_pred_domain,
                                    np.maximum(stumpf_support - stumpf_envelope_m, 0.0),
                                    stumpf_support + stumpf_envelope_m,
                                )
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
                                support_q_domain = np.ones_like(doa_q_domain, dtype=np.float32)
                                if support_weight_local is not None and edge_weight_local is not None:
                                    try:
                                        support_q_domain = (support_weight_local[domain_mask_local] * edge_weight_local[domain_mask_local]).astype(np.float32)
                                    except (TypeError, ValueError, AttributeError) as _exc:
                                        log.debug("Suppressed exception: %s", _exc)
                                conf_domain = doa_q_domain * optical_q_domain * unc_q_domain * support_q_domain

                                conf_pixels = np.full(int(np.sum(valid_mask)), NODATA_VAL, dtype=np.float32)
                                conf_pixels[domain_mask_local] = np.where(good, conf_domain, NODATA_VAL)
                                confidence_block[valid_mask] = conf_pixels

                            guidance_pixels = np.full(int(np.sum(valid_mask)), NODATA_VAL, dtype=np.float32)
                            trusted_pixels = np.zeros(int(np.sum(valid_mask)), dtype=np.uint8)
                            if support_weight_local is not None and edge_weight_local is not None:
                                try:
                                    support_q_domain = (support_weight_local[domain_mask_local] * edge_weight_local[domain_mask_local]).astype(np.float32)
                                except (TypeError, ValueError, AttributeError):
                                    support_q_domain = np.ones(int(domain_mask_local.sum()), dtype=np.float32)
                            else:
                                support_q_domain = np.ones(int(domain_mask_local.sum()), dtype=np.float32)
                            guidance_domain = np.clip(blend_weight * support_q_domain, 0.0, 1.0).astype(np.float32)
                            guidance_pixels[domain_mask_local] = np.where(good, guidance_domain, NODATA_VAL)
                            trusted_domain = (guidance_domain >= 0.85).astype(np.uint8)
                            trusted_pixels[domain_mask_local] = np.where(good, trusted_domain, 0).astype(np.uint8)
                            guidance_block[valid_mask] = guidance_pixels
                            trusted_block[valid_mask] = trusted_pixels

                            # Admissibility: marks the optical domain where SDB guidance is valid.
                            # A pixel is admissible if it passed all masks (valid_mask) AND DOA.
                            admiss_pixels = np.zeros(int(np.sum(valid_mask)), dtype=np.uint8)
                            admiss_pixels[domain_mask_local] = np.where(good, np.uint8(1), np.uint8(0))
                            admiss_block[valid_mask] = admiss_pixels

                    except (RuntimeError, ValueError, TypeError, OSError, MemoryError) as exc:
                        log.exception("Prediction failed on tile: %s", exc)

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
                        guidance_block[bad] = NODATA_VAL
                        trusted_block[bad] = 0
                        admiss_block[bad] = 0
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
                        guidance_block[low_conf] = NODATA_VAL
                        trusted_block[low_conf] = 0
                        # Note: admissibility is NOT zeroed by confidence threshold —
                        # the pixel is still in the admissible optical domain, just low
                        # confidence.  Downstream can decide whether to use it.
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
                dst_guidance.write(np.ascontiguousarray(guidance_block, dtype=np.float32), 1, window=window)
                dst_trusted.write(np.ascontiguousarray(trusted_block, dtype=np.uint8), 1, window=window)
                dst_admiss.write(np.ascontiguousarray(admiss_block, dtype=np.uint8), 1, window=window)
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

        # --- SDB Prediction Acceptance Ledger ---
        # Machine-readable diagnostic explaining exactly where pixels were
        # accepted or rejected.  Downstream QA tools consume this to identify
        # coverage problems.
        try:
            ledger = {
                "schema": "sdb_acceptance_ledger_v1",
                "funnel": {k: int(v) for k, v in funnel.items()},
                "rejection_gates": {
                    "not_finite": int(funnel["pixels_total"] - funnel["finite_optical"]),
                    "clear_water_mask": int(funnel["finite_optical"] - funnel["cw_pass"]),
                    "land_mask": int(funnel["cw_pass"] - funnel["land_pass"]),
                    "doa_rejection": int(funnel["base_valid"] - funnel["doa_pass"]),
                    "support_distance_or_depth_limit": int(funnel["doa_pass"] - funnel["predicted"]),
                },
                "acceptance_rate": float(funnel["predicted"]) / max(float(funnel["pixels_total"]), 1.0),
                "water_acceptance_rate": float(funnel["predicted"]) / max(float(funnel["land_pass"]), 1.0),
                "support_gate": {
                    "max_train_point_dist_m": float(max_support_dist_m) if max_support_dist_m else 0.0,
                    "min_support_neighbors": int(min_support_neighbors),
                    "n_support_points": int(support_tree.n) if support_tree is not None else 0,
                    "anchor_support_good": bool(anchor_support_good),
                },
                "optical_limits": {
                    "depth_limit_mode": str(depth_limit_mode),
                    "max_depth_training_m": float(max_depth_training) if max_depth_training else None,
                    "max_depth_hard_cap": float(max_depth_hard_cap) if max_depth_hard_cap else None,
                    "cw_min": float(cw_min) if cw_min is not None else None,
                },
            }
            if doa_valid_total > 0:
                ledger["doa_stats"] = {
                    "valid_pixels": int(doa_valid_total),
                    "pass": int(doa_pass_total),
                    "reject": int(doa_reject_total),
                    "reject_rate": float(doa_reject_total) / max(float(doa_valid_total), 1.0),
                }
            ledger_path = Path(str(out_path)).with_name(Path(str(out_path)).stem + "_acceptance_ledger.json")
            import json as _json_ledger
            with open(ledger_path, "w") as _lf:
                _json_ledger.dump(ledger, _lf, indent=2, default=str)
            log.info("Acceptance ledger: %s", ledger_path)
        except (OSError, TypeError, ValueError) as _ledger_e:
            log.warning("Failed to write acceptance ledger: %s", _ledger_e)

        if funnel['predicted'] == 0:
            log.error("Zero pixels predicted. Possible causes: "
                      "cw_min too high; land mask masking water; missing S2 data; DOA rejecting everything (try --no-doa)")
        elif funnel['predicted'] < 0.01 * total:
            log.warning("Very few pixels predicted (%d). Consider reviewing mask settings.", funnel['predicted'])

        try:
            with rasterio.open(out_path, "r+") as ds_depth, \
                 rasterio.open(guidance_weight_path, "r+") as ds_gw, \
                 rasterio.open(trusted_interior_path, "r+") as ds_ti, \
                 rasterio.open(admissibility_path, "r+") as ds_adm:
                depth_arr = ds_depth.read(1).astype(np.float32)
                gw_arr = ds_gw.read(1).astype(np.float32)
                ti_arr = ds_ti.read(1).astype(np.uint8)
                adm_arr = ds_adm.read(1).astype(np.uint8)
                nodata = ds_depth.nodata if ds_depth.nodata is not None else NODATA_VAL
                valid_depth = np.isfinite(depth_arr) & (depth_arr != nodata)
                contract = build_sdb_source_contract(
                    valid_depth=valid_depth,
                    guidance_weight=gw_arr,
                    trusted_interior=ti_arr,
                    admissibility=adm_arr,
                )
                ds_gw.write(contract["guidance_weight"].astype(np.float32), 1)
                ds_ti.write(contract["trusted_interior"].astype(np.uint8), 1)
                ds_adm.write(contract["admissibility"].astype(np.uint8), 1)
                regime_profile = ds_depth.profile.copy()
                regime_profile.update(dtype=rasterio.uint8, nodata=0, count=1, compress="DEFLATE")
            with rasterio.open(regime_class_path, "w", **regime_profile) as ds_reg:
                ds_reg.write(contract["regime"].astype(np.uint8), 1)
                ds_reg.update_tags(DESC="Shared regime contract class: 1=upland, 2=nearshore_water, 3=estuary_transition, 4=river_channel", ROLE="interpolation_guidance", AUTHORITATIVE="false", CUDEM_INTENT="guidance_only")
            regime_summary = contract.get("regime_summary", {})
        except Exception:
            regime_summary = {}
            log.debug("Failed to normalize SDB source guidance contract outputs", exc_info=True)

        if conf_path:
            log.info("Confidence raster: %s", conf_path)
            if min_confidence_threshold > 0:
                log.info("Applied min_confidence_threshold=%.2f — cells below this are nodata in depth raster", min_confidence_threshold)
        log.info("Guidance-weight raster: %s", guidance_weight_path)
        log.info("Trusted-interior raster: %s", trusted_interior_path)
        log.info("Admissibility raster: %s", admissibility_path)
        log.info("Regime-class raster: %s", regime_class_path)
        if prov_path:
            log.info("Provenance raster: %s  (0=nodata, 1=predicted)", prov_path)

        # report
        try:
            report_dir = Path(diagnostics_dir) if diagnostics_dir else Path(out_path).parent
            report_dir.mkdir(parents=True, exist_ok=True)
            rep = {"predict": {"funnel": funnel, "high_unc_threshold_m": float(high_unc_threshold),
                               "confidence_path": conf_path, "provenance_path": prov_path,
                               "guidance_weight_path": guidance_weight_path, "trusted_interior_path": trusted_interior_path,
                               "admissibility_path": admissibility_path,
                               "regime_class_path": regime_class_path,
                               "regime_summary": regime_summary,
                               "cudem_guidance_only": True, "authoritative": False,
                               "min_confidence_threshold": min_confidence_threshold}}
            with open(report_dir / "predict_report.json", "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
        except OSError:
            log.debug("Failed to write predict_report.json", exc_info=True)

        # -------------------------------------------------------------------
        # Sparse guide-point export (guidance-producer alignment)
        #
        # Emit spatially thinned pseudo-soundings from the SDB prediction as a
        # GeoPackage.  Each point carries depth, confidence, guidance_weight,
        # uncertainty, and provenance so downstream DEM interpolation can treat
        # them as confidence-weighted sparse constraints — NOT wall-to-wall
        # raster truth.  Points are subordinate to authoritative survey data.
        # -------------------------------------------------------------------
        guide_points_path = str(Path(out_path).with_name(Path(out_path).stem + "_guide_points.gpkg"))
        try:
            _export_sdb_guide_points(
                depth_path=out_path,
                confidence_path=conf_path,
                guidance_weight_path=guidance_weight_path,
                uncertainty_path=out_unc_path,
                out_gpkg=guide_points_path,
                min_confidence=max(float(min_confidence_threshold), 0.15),
                max_points=5000,
            )
            rep.setdefault("predict", {})["guide_points_path"] = guide_points_path
            log.info("SDB guide points exported: %s", guide_points_path)
        except (ImportError, ModuleNotFoundError, OSError, ValueError, RuntimeError):
            log.debug("SDB guide-point export skipped (optional dependency missing or no valid pixels)", exc_info=True)

        try:
            from sdb_guidance import write_sdb_guidance_bounds
            bounds = write_sdb_guidance_bounds(out_path, out_unc_path, logger=log)
            rep.setdefault("predict", {}).update({k: v for k, v in bounds.items() if v})
        except (ImportError, ModuleNotFoundError, OSError, ValueError, RuntimeError):
            log.debug("SDB guidance bounds export skipped", exc_info=True)

    finally:
        for s in srcs.values():
            try:
                close_fn = getattr(s, "close", None)
                if callable(close_fn):
                    close_fn()
            except (OSError, AttributeError):
                log.debug("ignored", exc_info=True)  # close error

    log.info("Finished. Depth: %s", out_path)
    return {"status": "ok"}


def reproject_to_nad83(src_path: str, dst_path: str):
    cmd = f"gdalwarp -overwrite -t_srs EPSG:4269 -r bilinear -of GTiff {shlex.quote(src_path)} {shlex.quote(dst_path)}"
    log.info("Reprojecting with: %s", cmd)
    try:
        run_cmd(shlex.split(cmd), check=True)
    except (OSError, subprocess.CalledProcessError):
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
    except (ImportError, OSError, ValueError):
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sys.exit(main())
