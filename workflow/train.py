#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py – SDB model training with Feature Importance & Robust Metadata

Updates:
- **PHYSICS INTEGRATION: Bottom endmembers & geometry-corrected Kd (Kim et al. 2024)**
- **FIXED: Auto-Depth on Random Split**: Allows depth-of-support calc even if spatial_split=False.
- **FIXED: Smart Spatial Split**: Now selects a test cluster that actually covers the depth range.
- **FIXED: Stratified Random Split**: Uses quantile binning to ensure deep points appear in Test.
"""

import sys
import os
import argparse
import logging
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from plot_utils import lazy_pyplot
plt = None  # lazy-loaded when plots are enabled

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# --- Physics-based SDB (Kim et al. 2024) ---
PHYSICS_AVAILABLE = False
_physics_import_error = None
try:
    import physics_integration
    PHYSICS_AVAILABLE = True
except ImportError as e:
    _physics_import_error = f"ImportError: {e}"
except Exception as e:
    # Catch any other errors (syntax, missing dependencies, etc.)
    _physics_import_error = f"{type(e).__name__}: {e}"
    import logging
    logging.getLogger(__name__).error(
        f"[train] Physics module failed to load: {_physics_import_error}"
    )

# --- Robust Import Strategy for s2_optics (lazy + dynamic-module aware) ---
S2_OPTICS_AVAILABLE = False


def get_physics_import_error() -> str:
    """Return the physics module import error message, if any."""
    return _physics_import_error or ""


def stratified_train_test_split(df, target_col='depth_m', test_size=0.2, seed=42):
    """
    Splits data while ensuring the distribution of depths is preserved
    in both Train and Test sets (Stratified by Quantile).
    """
    try:
        df = df.copy()
        df['stratify_bin'] = pd.qcut(df[target_col], q=20, labels=False, duplicates='drop')
        
        bin_counts = df['stratify_bin'].value_counts()
        rare_bins = bin_counts[bin_counts < 2].index
        if len(rare_bins) > 0:
            major_bin = bin_counts.idxmax()
            df.loc[df['stratify_bin'].isin(rare_bins), 'stratify_bin'] = major_bin

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df['stratify_bin']
        )
        
        train_df = train_df.drop(columns=['stratify_bin'])
        test_df = test_df.drop(columns=['stratify_bin'])
        return train_df.index.to_numpy(), test_df.index.to_numpy()

    except Exception as e:
        logging.getLogger(__name__).warning(f"Stratification failed ({e}). Falling back to random split.")
        tr, te = train_test_split(df.index.to_numpy(), test_size=test_size, random_state=seed)
        return tr, te


def estimate_linf_from_df(df, *, nir_max=0.03, bright_max=0.15, percentile=1.0, cw_col=None):
    import numpy as _np
    if df is None or df.empty:
        return {}
    need = ["B02", "B03", "B04", "B08"]
    for c in need:
        if c not in df.columns:
            return {}
    m = _np.isfinite(df["B08"].to_numpy()) & (_np.asarray(df["B08"]) < float(nir_max))
    if "brightness" in df.columns:
        m &= _np.isfinite(df["brightness"].to_numpy()) & (_np.asarray(df["brightness"]) < float(bright_max))
    if cw_col and cw_col in df.columns:
        m &= _np.isfinite(df[cw_col].to_numpy()) & (_np.asarray(df[cw_col]) >= 0.5)

    if m.sum() < 50:
        return {}

    out = {}
    for c in need:
        v = _np.asarray(df.loc[m, c], dtype="float64")
        v = v[_np.isfinite(v)]
        if v.size == 0:
            return {}
        out[c] = float(_np.percentile(v, float(percentile)))
    return out


def estimate_linf_from_rasters(
    raster_paths: dict,
    *,
    nir_max: float = 0.03,
    bright_max: float = 0.15,
    percentile: float = 1.0,
    max_samples: int = 2_000_000,
):
    import numpy as _np
    import rasterio as _rio

    need = ["B02", "B03", "B04", "B08"]
    if raster_paths is None:
        return {}
    missing = [k for k in need if k not in raster_paths or raster_paths[k] is None]
    if missing:
        raise ValueError(f"estimate_linf_from_rasters requires raster_paths for: {missing}")

    paths = {k: str(raster_paths[k]) for k in need}

    vals = {k: [] for k in need}

    with _rio.open(paths["B08"]) as ds8, _rio.open(paths["B02"]) as ds2, _rio.open(paths["B03"]) as ds3, _rio.open(paths["B04"]) as ds4:
        if (ds8.width, ds8.height) != (ds2.width, ds2.height) or (ds8.width, ds8.height) != (ds3.width, ds3.height) or (ds8.width, ds8.height) != (ds4.width, ds4.height):
            raise ValueError("S2 band rasters must be co-registered and same shape for raster-based L∞ estimation.")

        for ji, window in ds8.block_windows(1):
            b08 = ds8.read(1, window=window).astype("float32")
            b02 = ds2.read(1, window=window).astype("float32")
            b03 = ds3.read(1, window=window).astype("float32")
            b04 = ds4.read(1, window=window).astype("float32")

            m = _np.isfinite(b08) & _np.isfinite(b02) & _np.isfinite(b03) & _np.isfinite(b04)
            if not m.any():
                continue

            brightness = (b02 + b03 + b04) / 3.0
            deep = m & (b08 < nir_max) & (brightness < bright_max)
            if not deep.any():
                continue

            for k, arr in (("B02", b02), ("B03", b03), ("B04", b04), ("B08", b08)):
                vv = arr[deep].ravel()
                if vv.size:
                    vals[k].append(vv)

            cur = sum(v.size for parts in vals.values() for v in parts)
            if cur >= max_samples:
                break

    out = {}
    for k in need:
        if not vals[k]:
            continue
        v = _np.concatenate(vals[k])
        out[k] = float(_np.percentile(v, percentile))
    return out

# -----------------------------------------------------------------------------
# Local S2 optical feature helpers
# -----------------------------------------------------------------------------

def add_s2_optical_features(df, l_inf: dict = None, eps: float = 1e-6):
    """
    Add derived optical features from Sentinel-2 bands.
    
    Features added:
        - log_B02, log_B03, log_B04, log_B08: Log-transformed bands
        - B03_B02, B04_B03: Band ratios
        - nbri: Normalized Band Ratio Index (B03-B08)/(B03+B08)
        - stumpf_idx: Stumpf log-ratio index for bathymetry
    
    Args:
        df: DataFrame with B02, B03, B04, B08 columns
        l_inf: Optional L-infinity correction values per band
        eps: Small value to prevent log(0) and division by zero
    
    Returns:
        DataFrame with added feature columns
        
    Note:
        FIX: Division-safe calculation of stumpf_idx to prevent inf/nan
        when log_B03 is near zero.
    """
    import numpy as _np
    from constants import NUMERICAL_EPS

    if l_inf is None:
        l_inf = {"B02": 0.0, "B03": 0.0, "B04": 0.0, "B08": 0.0}

    need = ["B02", "B03", "B04", "B08"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot add S2 features; missing bands: {missing}")

    out = df.copy()

    # Apply L-infinity correction and ensure positive values for log transform
    b02 = _np.maximum(out["B02"].to_numpy(_np.float32) - l_inf.get("B02", 0.0), eps)
    b03 = _np.maximum(out["B03"].to_numpy(_np.float32) - l_inf.get("B03", 0.0), eps)
    b04 = _np.maximum(out["B04"].to_numpy(_np.float32) - l_inf.get("B04", 0.0), eps)
    b08 = _np.maximum(out["B08"].to_numpy(_np.float32) - l_inf.get("B08", 0.0), eps)

    # Log-transformed bands
    out["log_B02"] = _np.log(b02)
    out["log_B03"] = _np.log(b03)
    out["log_B04"] = _np.log(b04)
    out["log_B08"] = _np.log(b08)

    # Band ratios with division-by-zero protection
    out["B03_B02"] = b03 / _np.maximum(b02, NUMERICAL_EPS)
    out["B04_B03"] = b04 / _np.maximum(b03, NUMERICAL_EPS)


    # Mean visible-band brightness (use L∞-corrected bands so it is consistent with other features)
    out["brightness"] = (b02 + b03 + b04) / 3.0

    # NBRI: protect against divide-by-zero
    nbri_denom = b03 + b08
    out["nbri"] = _np.where(
        _np.abs(nbri_denom) > NUMERICAL_EPS,
        (b03 - b08) / nbri_denom,
        0.0
    )

    # FIX: Stumpf index with division-by-zero protection
    # stumpf_idx = log(B02) / log(B03)
    # When log(B03) is near zero, this ratio becomes undefined
    log_b03_safe = _np.where(
        _np.abs(out["log_B03"]) > NUMERICAL_EPS,
        out["log_B03"],
        _np.sign(out["log_B03"]) * NUMERICAL_EPS  # Preserve sign
    )
    out["stumpf_idx"] = out["log_B02"] / log_b03_safe
    
    # Clip extreme values that indicate numerical issues
    out["stumpf_idx"] = _np.clip(out["stumpf_idx"], -10.0, 10.0)

    return out


def apply_s2_brightness_depth_filter(df, *args, **kwargs):
    import numpy as _np
    if df is None or df.empty:
        return df

    max_blue = kwargs.get("max_blue", None)
    max_brightness = kwargs.get("max_brightness", None)

    m = _np.ones(len(df), dtype=bool)
    if max_blue is not None and "B02" in df.columns:
        m &= (df["B02"].to_numpy(_np.float32) <= float(max_blue))

    if max_brightness is not None and all(c in df.columns for c in ["B02", "B03", "B04"]):
        rgb_mean = (df["B02"].to_numpy(_np.float32) + df["B03"].to_numpy(_np.float32) + df["B04"].to_numpy(_np.float32)) / 3.0
        m &= (rgb_mean <= float(max_brightness))

    return df.loc[m].copy()


# -----------------------------------------------------------------------------
# Physics guardrail: Stumpf residual outlier filter
# -----------------------------------------------------------------------------
from sklearn.linear_model import RANSACRegressor

def apply_stumpf_residual_filter(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    residual_threshold_std: float = 2.5,
    residual_abs_min_m: float = 0.5,
    min_points: int = 200,
    min_inliers: int = 100,
    min_depth_m: float = 0.0,
    max_depth_m: Optional[float] = None,
    n_clusters: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    if (not enabled) or df is None or df.empty:
        return df
    if "stumpf_idx" not in df.columns or "depth_m" not in df.columns:
        return df

    stumpf = pd.to_numeric(df["stumpf_idx"], errors="coerce").to_numpy(dtype="float64")
    depth = pd.to_numeric(df["depth_m"], errors="coerce").to_numpy(dtype="float64")
    m_valid = np.isfinite(stumpf) & np.isfinite(depth)
    n_valid = int(np.count_nonzero(m_valid))
    if n_valid < int(min_points):
        return df

    d = depth[m_valid]
    flip = -1.0 if (np.nanmedian(d) < 0.0) else 1.0
    depth_pd = np.abs(depth * flip)

    m_fit = m_valid.copy()
    if min_depth_m is not None:
        m_fit &= depth_pd >= float(min_depth_m)
    if max_depth_m is not None:
        m_fit &= depth_pd <= float(max_depth_m)

    if int(np.count_nonzero(m_fit)) < int(min_points):
        return df

    if ("brightness" in df.columns) and np.isfinite(pd.to_numeric(df["brightness"], errors="coerce").to_numpy(dtype="float64")).any():
        bright = pd.to_numeric(df["brightness"], errors="coerce").to_numpy(dtype="float64")
        Z = np.column_stack([stumpf, bright])
        Z_fit = Z[m_fit]
        if not np.all(np.isfinite(Z_fit)):
            med = np.nanmedian(Z_fit[np.isfinite(Z_fit)])
            Z_fit = np.where(np.isfinite(Z_fit), Z_fit, med)
    else:
        Z_fit = stumpf[m_fit].reshape(-1, 1)

    clusters = np.zeros(int(np.count_nonzero(m_fit)), dtype=int)
    k = int(max(1, n_clusters))
    if k > 1 and int(np.count_nonzero(m_fit)) >= max(500, k * 200):
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(Z_fit)
            clusters = km.labels_.astype(int)
        except Exception:
            clusters = np.zeros_like(clusters)

    X_fit = stumpf[m_fit].reshape(-1, 1)
    y_fit = depth_pd[m_fit]
    idx_fit = np.flatnonzero(m_fit)

    keep = np.ones(len(df), dtype=bool)

    dropped_total = 0
    dropped_by_cluster = []

    for ci in range(int(np.max(clusters)) + 1):
        sel = (clusters == ci)
        if int(np.count_nonzero(sel)) < int(min_points):
            continue

        Xc = X_fit[sel]
        yc = y_fit[sel]

        ransac = RANSACRegressor(
            random_state=random_state,
            min_samples=0.6,
            residual_threshold=float(residual_abs_min_m),
        )
        try:
            ransac.fit(Xc, yc)
        except Exception:
            continue

        inliers = getattr(ransac, "inlier_mask_", None)
        if inliers is None or int(np.count_nonzero(inliers)) < int(min_inliers):
            continue

        y_pred = ransac.predict(Xc)
        resid = np.abs(yc - y_pred)
        sigma = float(np.nanstd(resid[inliers])) if np.any(inliers) else np.nan
        if not np.isfinite(sigma):
            sigma = 0.0
        thr = max(float(residual_abs_min_m), float(residual_threshold_std) * sigma)

        drop_c = resid > thr
        n_drop_c = int(np.count_nonzero(drop_c))
        if n_drop_c > 0:
            df_idx = idx_fit[sel][drop_c]
            keep[df_idx] = False
            dropped_total += n_drop_c
            dropped_by_cluster.append((ci, n_drop_c, thr, sigma, int(np.count_nonzero(sel))))

    if dropped_total > 0:
        try:
            d_drop = depth_pd[~keep & m_valid]
            d_keep = depth_pd[keep & m_valid]
            log.info(
                f"[QC] Stumpf residual filter dropped {dropped_total} / {len(df)} "
                f"({100.0*dropped_total/len(df):.1f}%). "
                f"Depth keep p50/p95={np.nanpercentile(d_keep,50):.2f}/{np.nanpercentile(d_keep,95):.2f} m; "
                f"dropped p50/p95={np.nanpercentile(d_drop,50):.2f}/{np.nanpercentile(d_drop,95):.2f} m."
            )
        except Exception:
            log.info(f"[QC] Stumpf residual filter dropped {dropped_total} outliers.")
        return df.loc[keep].reset_index(drop=True)

    return df

def _resolve_s2_optics_module():
    import sys
    import importlib
    if "s2_optics" in sys.modules:
        return sys.modules["s2_optics"]
    try:
        return importlib.import_module("s2_optics")
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    for k, v in list(sys.modules.items()):
        if k.startswith("s2_optics_dyn_"):
            return v
    return None

def _bind_s2_optics_functions():
    global S2_OPTICS_AVAILABLE
    global apply_stumpf_residual_filter, apply_s2_brightness_depth_filter

    mod = _resolve_s2_optics_module()
    if mod is None:
        S2_OPTICS_AVAILABLE = False
        return False

    try:
        apply_s2_brightness_depth_filter = getattr(mod, 'apply_s2_brightness_depth_filter', apply_s2_brightness_depth_filter)
        S2_OPTICS_AVAILABLE = True
        return True
    except Exception:
        S2_OPTICS_AVAILABLE = False
        return False

# Use centralized logging - get logger, don't configure root here
log = logging.getLogger("sdb.train")


# -----------------------------------------------------------------------------
# S2 A/B selection helpers (best-date vs composite)
# -----------------------------------------------------------------------------

def radiometric_coherence_score_from_band(band_path: Path, *, max_pixels: int = 1_500_000, seed: int = 42) -> float:
    """Return a radiometric coherence score in [0,1] (higher is better)."""
    import numpy as _np
    import rasterio as _rio

    band_path = Path(band_path)
    if not band_path.exists():
        return 0.0

    with _rio.open(str(band_path)) as ds:
        a = ds.read(1).astype("float32")
        nod = ds.nodata
        if nod is not None and _np.isfinite(nod):
            a[a == _np.float32(nod)] = _np.nan

    flat = a.ravel()
    m = _np.isfinite(flat)
    n = int(_np.count_nonzero(m))
    if n < 10_000:
        return 0.0

    if n > int(max_pixels):
        rng = _np.random.default_rng(int(seed))
        idx = _np.flatnonzero(m)
        idx_s = rng.choice(idx, size=int(max_pixels), replace=False)
        v = flat[idx_s]
    else:
        v = flat[m]

    med = float(_np.nanmedian(v))
    mad_signal = float(_np.nanmedian(_np.abs(v - med)))
    if not _np.isfinite(mad_signal) or mad_signal <= 0:
        return 0.0

    dv = _np.diff(v)
    mad_grad = float(_np.nanmedian(_np.abs(dv)))
    if not _np.isfinite(mad_grad):
        mad_grad = 0.0

    ratio = mad_grad / mad_signal
    return float(1.0 / (1.0 + ratio))


def _compute_test_rmse_from_df_te(df_te: pd.DataFrame) -> Tuple[Optional[float], int]:
    """Compute RMSE on df_te if depth_pred_m exists; returns (rmse, n)."""
    if df_te is None or df_te.empty:
        return None, 0
    if ("depth_m" not in df_te.columns) or ("depth_pred_m" not in df_te.columns):
        return None, 0
    yt = pd.to_numeric(df_te["depth_m"], errors="coerce").to_numpy(dtype="float64")
    yp = pd.to_numeric(df_te["depth_pred_m"], errors="coerce").to_numpy(dtype="float64")
    m = np.isfinite(yt) & np.isfinite(yp)
    n = int(np.count_nonzero(m))
    if n == 0:
        return None, 0
    yt, yp, _ = _normalize_depth_pair(yt[m], yp[m])
    return float(np.sqrt(np.mean((yp - yt) ** 2))), n


def _resolve_s2_paths(s2_dir: Path, *, suffix: str = "") -> Dict[str, str]:
    """Resolve expected S2 raster filenames in a directory."""
    s2_dir = Path(s2_dir)
    suf = suffix
    return {
        "B02": str(s2_dir / f"B02_10m{suf}.tif"),
        "B03": str(s2_dir / f"B03_10m{suf}.tif"),
        "B04": str(s2_dir / f"B04_10m{suf}.tif"),
        "B08": str(s2_dir / f"B08_10m{suf}.tif"),
        "CLEAR_WATER": str(s2_dir / f"CLEAR_WATER_MASK_10m{suf}.tif"),
        "BRIGHTNESS": str(s2_dir / f"BRIGHTNESS_10m{suf}.tif"),
    }


def _s2_paths_exist(paths: Dict[str, str]) -> bool:
    return all(paths.get(k) and Path(paths[k]).exists() for k in ("B02","B03","B04","B08","CLEAR_WATER","BRIGHTNESS"))


def choose_s2_variant(best_date: dict, composite: dict, *, rmse_margin: float = 0.05) -> Tuple[dict, str]:
    """Choose best-date vs composite using RMSE first, then coherence."""
    a = best_date
    b = composite

    if a.get("rmse_m") is not None and b.get("rmse_m") is not None:
        if a["rmse_m"] + float(rmse_margin) < b["rmse_m"]:
            return a, "lower RMSE"
        if b["rmse_m"] + float(rmse_margin) < a["rmse_m"]:
            return b, "lower RMSE"
        if float(a.get("coherence", 0.0)) >= float(b.get("coherence", 0.0)):
            return a, "RMSE tie → higher coherence"
        return b, "RMSE tie → higher coherence"

    if float(a.get("coherence", 0.0)) >= float(b.get("coherence", 0.0)):
        return a, "RMSE missing → higher coherence"
    return b, "RMSE missing → higher coherence"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _to_python_float(val):
    if isinstance(val, dict):
        return {k: _to_python_float(v) for k, v in val.items()}
    if hasattr(val, "item"):
        return val.item()
    return float(val)

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    return float(np.sqrt(mean_squared_error(a[m], b[m])))

def r2_masked(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 2:
        return np.nan
    return float(r2_score(a[m], b[m]))


def _normalize_depth_pair(y_true: np.ndarray, y_pred: np.ndarray):
    """Normalize depths to a consistent positive-down convention."""
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    m = np.isfinite(yt)
    flip = -1.0 if (np.any(m) and np.nanmedian(yt[m]) < 0) else 1.0
    return yt * flip, yp * flip, {"ref_flip": int(flip), "pred_flip": int(flip)}


def _make_depth_bin_edges(
    depths_pos: np.ndarray,
    *,
    depth_bin_m: float = 0.5,
    min_samples_per_bin: int = 200,
    max_depth_cap_m: Optional[float] = None,
    depth_binning: str = "quantile",
    max_bins: int = 30,
) -> np.ndarray:
    """Return depth bin edges for diagnostics."""
    d = np.asarray(depths_pos, dtype="float64")
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.array([], dtype="float64")

    max_ref = float(np.nanmax(d))
    if max_depth_cap_m is not None:
        try:
            max_ref = min(max_ref, float(max_depth_cap_m))
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    if (not np.isfinite(max_ref)) or max_ref <= 0:
        return np.array([], dtype="float64")

    mode = str(depth_binning or "quantile").strip().lower()
    if mode not in ("fixed", "quantile"):
        mode = "fixed"

    if mode == "fixed":
        bin_w = max(0.1, float(depth_bin_m))
        edges = np.arange(0.0, max_ref + bin_w, bin_w)
        if edges.size and edges[-1] < max_ref:
            edges = np.append(edges, max_ref)
        return edges.astype("float64")

    # quantile bins
    n = int(d.size)
    target_bins = max(2, int(np.floor(n / max(1, int(min_samples_per_bin)))))
    target_bins = max(2, min(int(max_bins), target_bins))

    dc = d[d <= max_ref]
    if dc.size < max(50, int(min_samples_per_bin)):
        bin_w = max(0.1, float(depth_bin_m))
        edges = np.arange(0.0, max_ref + bin_w, bin_w)
        if edges.size and edges[-1] < max_ref:
            edges = np.append(edges, max_ref)
        return edges.astype("float64")

    qs = np.linspace(0.0, 1.0, target_bins + 1)
    edges = np.quantile(dc, qs)
    edges[0] = 0.0
    edges[-1] = max_ref

    edges_u = np.unique(edges.astype("float64"))
    if edges_u.size < 3:
        bin_w = max(0.1, float(depth_bin_m))
        edges_u = np.arange(0.0, max_ref + bin_w, bin_w).astype("float64")
        if edges_u.size and edges_u[-1] < max_ref:
            edges_u = np.append(edges_u, max_ref)

    return edges_u


def estimate_max_depth_from_spatial_validation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    rmse_target_m: float = 0.5,
    depth_bin_m: float = 0.5,
    min_samples_per_bin: int = 200,
    max_depth_cap_m: Optional[float] = None,
depth_binning: str = "quantile",
max_depth_bins: int = 30,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Estimate a conservative 'depth-of-support' using spatially independent validation."""
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    m = np.isfinite(yt) & np.isfinite(yp)
    if np.count_nonzero(m) < max(50, int(min_samples_per_bin)):
        return None, {"reason": "insufficient_samples", "n": int(np.count_nonzero(m))}

    yt, yp, _ = _normalize_depth_pair(yt[m], yp[m])
    yt = np.abs(yt)
    yp = np.abs(yp)

    max_ref = float(np.nanmax(yt)) if yt.size else 0.0
    if max_depth_cap_m is not None:
        try:
            max_ref = min(max_ref, float(max_depth_cap_m))
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    if not np.isfinite(max_ref) or max_ref <= 0:
        return None, {"reason": "invalid_depth_range", "max_ref": max_ref}
    edges = _make_depth_bin_edges(
        yt,
        depth_bin_m=depth_bin_m,
        min_samples_per_bin=min_samples_per_bin,
        max_depth_cap_m=max_depth_cap_m,
        depth_binning=depth_binning,
        max_bins=max_depth_bins,
    )
    if edges.size < 3:
        return None, {
            "reason": "too_few_bins",
            "max_ref": max_ref,
            "depth_bin_m": float(depth_bin_m),
            "depth_binning": str(depth_binning),
        "bin_edges_m": [float(e) for e in np.asarray(edges).ravel().tolist()],
            "n_edges": int(edges.size),
        }

    rmse_by = []
    mae_by = []
    bias_by = []
    n_by = []
    mids = []
    supported_upper_edges = []

    log.info(f"[TRAIN][VAL] --- Depth-of-Support Analysis (Target RMSE <= {rmse_target_m:.2f} m) ---")
    log.info(f"[TRAIN][VAL] {'Bin Range (m)':<15} | {'RMSE (m)':<10} | {'Bias (m)':<10} | {'MAE (m)':<10} | {'Count':<8} | {'Status'}")
    log.info("-" * 85)

    first_fail_msg = None

    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mm = (yt >= lo) & (yt < hi)
        n = int(np.count_nonzero(mm))
        mids.append((lo + hi) / 2.0)
        n_by.append(n)

        status = "SKIP (N)"
        r_val = np.nan
        b_val = np.nan
        m_val = np.nan

        if n >= int(min_samples_per_bin):
            r = yp[mm] - yt[mm]
            r_val = float(np.sqrt(np.mean(r * r)))
            m_val = float(np.mean(np.abs(r)))
            b_val = float(np.mean(r))

            rmse_by.append(r_val)
            mae_by.append(m_val)
            bias_by.append(b_val)

            if np.isfinite(r_val) and (r_val <= float(rmse_target_m)):
                supported_upper_edges.append(hi)
                status = "PASS"
            else:
                status = "FAIL (RMSE)"
                if first_fail_msg is None:
                    first_fail_msg = f"RMSE threshold exceeded at bin {lo:.1f}-{hi:.1f}m (RMSE={r_val:.2f}m > {rmse_target_m:.2f}m)"
        else:
            rmse_by.append(np.nan); mae_by.append(np.nan); bias_by.append(np.nan)

        log.info(f"[TRAIN][VAL] {lo:5.1f} - {hi:5.1f}   | {r_val:10.3f} | {b_val:10.3f} | {m_val:10.3f} | {n:8d} | {status}")

    log.info("-" * 85)

    if first_fail_msg:
        log.info(f"[TRAIN][VAL] {first_fail_msg}")

    max_supported = float(max(supported_upper_edges)) if supported_upper_edges else None
    if max_supported is not None and max_depth_cap_m is not None:
        try:
            max_supported = min(max_supported, float(max_depth_cap_m))
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    diag = {
        "rmse_target_m": float(rmse_target_m),
        "depth_binning": str(depth_binning),
        "depth_bin_m": float(depth_bin_m),
        "min_samples_per_bin": int(min_samples_per_bin),
        "depth_bin_mids": [float(x) for x in mids],
        "n_by_bin": [int(x) for x in n_by],
        "rmse_by_bin_m": [None if not np.isfinite(x) else float(x) for x in rmse_by],
        "mae_by_bin_m": [None if not np.isfinite(x) else float(x) for x in mae_by],
        "bias_by_bin_m": [None if not np.isfinite(x) else float(x) for x in bias_by],
        "max_ref_depth_m": float(max_ref),
        "max_depth_supported_m": None if max_supported is None else float(max_supported),
    }
    if max_supported is None:
        diag["reason"] = "no_bins_meet_threshold"
    return max_supported, diag

def estimate_max_depth_from_spatial_validation_dual(
    y_true_te: np.ndarray,
    y_pred_te: np.ndarray,
    *,
    y_true_tr: Optional[np.ndarray] = None,
    rmse_target_m: float = 0.5,
    depth_bin_m: float = 0.5,
    depth_binning: str = "quantile",
    max_depth_bins: int = 30,
    min_samples_per_bin_te: int = 100,
    min_samples_per_bin_tr: int = 50,
    max_depth_cap_m: Optional[float] = None,
    max_consecutive_empty_bins_relaxed: int = 4,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    """Estimate depth-of-support using spatially independent validation."""
    log = logging.getLogger("sdb.train")

    def _norm_pair(yt: np.ndarray, yp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        yt = np.asarray(yt, dtype="float64")
        yp = np.asarray(yp, dtype="float64")
        m = np.isfinite(yt) & np.isfinite(yp)
        if np.count_nonzero(m) == 0:
            return np.array([], dtype="float64"), np.array([], dtype="float64")
        yt = np.abs(yt[m])
        yp = np.abs(yp[m])
        return yt, yp

    yt_te, yp_te = _norm_pair(y_true_te, y_pred_te)
    if yt_te.size < max(50, int(min_samples_per_bin_te)):
        return {"strict": None, "relaxed": None}, {
            "reason": "insufficient_samples_test_total",
            "n_test_total": int(yt_te.size),
        }

    yt_tr = None
    if y_true_tr is not None:
        yt_tr = np.asarray(y_true_tr, dtype="float64")
        yt_tr = yt_tr[np.isfinite(yt_tr)]
        yt_tr = np.abs(yt_tr)

    max_ref = float(np.nanmax(yt_te)) if yt_te.size else 0.0
    if max_depth_cap_m is not None:
        max_ref = min(max_ref, float(max_depth_cap_m))
    edges = _make_depth_bin_edges(
        yt_te,
        depth_bin_m=depth_bin_m,
        min_samples_per_bin=min_samples_per_bin_te,
        max_depth_cap_m=max_depth_cap_m,
        depth_binning=depth_binning,
        max_bins=max_depth_bins,
    )
    if str(depth_binning).lower() == "quantile":
        edges_list = [round(float(e), 3) for e in np.asarray(edges).ravel().tolist()]
        log.info(f"[TRAIN][VAL] Quantile depth bin edges (m): {edges_list}")


    strict_max: Optional[float] = None
    relaxed_max: Optional[float] = None
    strict_stop = None
    relaxed_stop = None

    consec_gap = 0
    gap_start_idx: Optional[int] = None

    def _n_train(lo: float, hi: float) -> int:
        if yt_tr is None or yt_tr.size == 0:
            return 0
        return int(np.count_nonzero((yt_tr >= lo) & (yt_tr < hi)))

    def _gap_has_train_support(i0: int, i1: int) -> bool:
        if yt_tr is None or yt_tr.size == 0:
            return False
        total = 0
        any_ok = False
        for j in range(i0, i1 + 1):
            lo, hi = float(edges[j]), float(edges[j + 1])
            ntr = _n_train(lo, hi)
            total += ntr
            if ntr >= int(min_samples_per_bin_tr):
                any_ok = True
        return (total >= int(min_samples_per_bin_tr)) and any_ok

    per_bin = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mm = (yt_te >= lo) & (yt_te < hi)
        n = int(np.count_nonzero(mm))

        entry = {
            "lo": lo,
            "hi": hi,
            "n_test": n,
            "n_train": _n_train(lo, hi) if yt_tr is not None else None,
            "rmse": None,
            "mae": None,
            "bias": None,
            "status_strict": None,
            "status_relaxed": None,
        }

        if n >= int(min_samples_per_bin_te):
            r = yp_te[mm] - yt_te[mm]
            rmse = float(np.sqrt(np.mean(r * r))) if r.size else np.nan
            mae = float(np.mean(np.abs(r))) if r.size else np.nan
            bias = float(np.mean(r)) if r.size else np.nan
            entry.update({"rmse": rmse, "mae": mae, "bias": bias})

            consec_gap = 0
            gap_start_idx = None

            if np.isfinite(rmse) and (rmse <= float(rmse_target_m)):
                entry["status_strict"] = "PASS"
                entry["status_relaxed"] = "PASS"
                strict_max = hi if strict_stop is None else strict_max
                relaxed_max = hi if relaxed_stop is None else relaxed_max
            else:
                entry["status_strict"] = "FAIL_RMSE"
                entry["status_relaxed"] = "FAIL_RMSE"
                if strict_stop is None:
                    strict_stop = f"rmse_fail_{lo:.2f}_{hi:.2f}_rmse_{rmse:.3f}"
                if relaxed_stop is None:
                    relaxed_stop = f"rmse_fail_{lo:.2f}_{hi:.2f}_rmse_{rmse:.3f}"
                per_bin.append(entry)
                break

        else:
            if strict_stop is None:
                strict_stop = f"gap_or_sparse_{lo:.2f}_{hi:.2f}_n_{n}"
                entry["status_strict"] = "STOP_GAP"
            else:
                entry["status_strict"] = "IGNORED_AFTER_STOP"

            if relaxed_stop is None:
                consec_gap += 1
                if gap_start_idx is None:
                    gap_start_idx = i

                if consec_gap <= int(max_consecutive_empty_bins_relaxed):
                    entry["status_relaxed"] = f"BRIDGE_GAP_{consec_gap}"
                else:
                    # Avoid `assert` here (assertions can be disabled with -O).
                    # If we somehow lost the gap start index, fail closed with a clear stop reason.
                    if gap_start_idx is None:
                        entry["status_relaxed"] = "STOP_GAP_INTERNAL_STATE"
                        relaxed_stop = "gap_internal_state_missing_start_idx"
                    elif _gap_has_train_support(gap_start_idx, i):
                        extra_allow = 2
                        if consec_gap <= int(max_consecutive_empty_bins_relaxed) + extra_allow:
                            entry["status_relaxed"] = f"BRIDGE_GAP_TRAIN_OK_{consec_gap}"
                        else:
                            entry["status_relaxed"] = "STOP_GAP_TOO_LARGE"
                            relaxed_stop = f"gap_too_large_even_with_train_{edges[gap_start_idx]:.2f}_{hi:.2f}_bins_{consec_gap}"
                    else:
                        entry["status_relaxed"] = "STOP_GAP_NO_TRAIN_SUPPORT"
                        relaxed_stop = f"gap_no_train_support_{edges[gap_start_idx]:.2f}_{hi:.2f}_bins_{consec_gap}"

        per_bin.append(entry)

    depths = {
        "strict": float(strict_max) if strict_max and strict_max > 0 else None,
        "relaxed": float(relaxed_max) if relaxed_max and relaxed_max > 0 else None,
    }
    diag = {
        "rmse_target_m": float(rmse_target_m),
        "depth_binning": str(depth_binning),
        "depth_bin_m": float(depth_bin_m),
        "min_samples_per_bin_te": int(min_samples_per_bin_te),
        "min_samples_per_bin_tr": int(min_samples_per_bin_tr),
        "max_depth_cap_m": float(max_depth_cap_m) if max_depth_cap_m is not None else None,
        "max_consecutive_empty_bins_relaxed": int(max_consecutive_empty_bins_relaxed),
        "stop_reason_strict": strict_stop or "reached_max_depth",
        "stop_reason_relaxed": relaxed_stop or "reached_max_depth",
        "depths_m": depths,
        "per_bin": per_bin,
    }
    return depths, diag

def scatter_plot(
    y_true,
    y_pred,
    title,
    out_png,
    max_depth=None,
    x_label="Reference Depth (m)",
    y_label="Predicted Depth (m)",
    *,
    all_pctl: float = 95.0,
):
    """Scatter plot with robust limits and optional subset view."""
    out_png = Path(out_png)
    global plt
    if plt is None:
        plt = lazy_pyplot()

    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        log.warning(f"[PLOT] {title}: no finite points; skipping {out_png}")
        return

    yt = np.asarray(y_true, dtype="float64")[m]
    yp = np.asarray(y_pred, dtype="float64")[m]
    yt, yp, _ = _normalize_depth_pair(yt, yp)

    def _stats_text(yt_s, yp_s, *, note: Optional[str] = None) -> str:
        yt_s = np.asarray(yt_s, dtype="float64")
        yp_s = np.asarray(yp_s, dtype="float64")
        n_s = int(len(yt_s))
        if n_s == 0:
            return (note + "\n" if note else "") + "N=0"
        rr = rmse(yt_s, yp_s)
        r2v = r2_masked(yt_s, yp_s)
        maev = float(mean_absolute_error(yt_s, yp_s))
        biasv = float(np.nanmean(yp_s - yt_s))
        header = (note + "\n") if note else ""
        return (
            header
            + f"RMSE={rr:.2f} m\n"
            + f"MAE={maev:.2f} m\n"
            + f"Bias={biasv:.2f} m\n"
            + f"R²={r2v:.2f}\n"
            + f"N={n_s}"
        )

    stats_txt_all = _stats_text(yt, yp)

    def _lims_percentile(vals, p=95.0):
        vals = np.asarray(vals, dtype="float64")
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return [-1.0, 1.0]
        vals = np.abs(vals)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return [-1.0, 1.0]
        p = float(p)
        p = min(max(p, 1.0), 100.0)
        hi = float(np.percentile(vals, p))
        if not np.isfinite(hi) or hi <= 0:
            hi = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 1.0
        pad = 0.02 * hi if hi > 0 else 0.5
        return [float(0.0 - pad), float(hi + pad)]

    def _render(yt_sub, yp_sub, suffix, path_out, *, stats_txt, lim=None):
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()

        if yt_sub.size:
            ax.scatter(yt_sub, yp_sub, s=5, c="black", alpha=0.3, edgecolors="none")
        else:
            ax.text(0.5, 0.5, "No Valid Points", ha="center", va="center", transform=ax.transAxes)

        if lim is None:
            lim = _lims_percentile(np.concatenate([yt_sub, yp_sub]) if yt_sub.size else np.concatenate([yt, yp]), p=all_pctl)

        ax.plot(lim, lim, "r--", label="1:1")
        ax.set_xlim(lim)
        ax.set_ylim(lim)

        ax.text(
            0.05, 0.95, stats_txt,
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

        ax.set_title(f"{title}{suffix}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_out, dpi=150)
        plt.close(fig)
        log.info(f"[PLOT] Saved: {path_out}")

    lim_all = _lims_percentile(np.concatenate([yt, yp]), p=all_pctl)
    
    if max_depth is not None:
        md = float(max_depth)
        mlim = (yt >= 0.0) & (yt <= md)
        n_limited = int(np.count_nonzero(mlim))
        n_all = int(yt.size)
        
        # Only create V1_AllData if max_depth actually excludes some data
        if n_limited < n_all:
            v1_path = out_png.parent / f"{out_png.stem}_V1_AllData{out_png.suffix}"
            _render(yt, yp, " (All Data)", v1_path, stats_txt=stats_txt_all, lim=lim_all)
            
            stats_txt_lim = _stats_text(yt[mlim], yp[mlim], note=f"0–{md:g} m subset")
            _render(yt[mlim], yp[mlim], f" (0-{md:g}m)", out_png, stats_txt=stats_txt_lim, lim=lim_all)
        else:
            # max_depth >= all data, just render one plot (no _V1_AllData needed)
            _render(yt, yp, f" (0-{md:g}m)", out_png, stats_txt=stats_txt_all, lim=lim_all)
    else:
        _render(yt, yp, "", out_png, stats_txt=stats_txt_all, lim=lim_all)



def plot_depth_binning_sanity(
    y_true_te: np.ndarray,
    *,
    y_true_tr: Optional[np.ndarray] = None,
    bin_edges: Optional[List[float]] = None,
    out_png: Optional[Path] = None,
    title: str = "Depth binning sanity",
    min_samples_per_bin_te: int = 100,
    strict_max: Optional[float] = None,
    relaxed_max: Optional[float] = None,
) -> None:
    """Save a quick diagnostic plot showing the depth distribution and bins."""
    global plt
    if plt is None:
        plt = lazy_pyplot()
    if out_png is None or bin_edges is None:
        return
    try:
        edges = np.asarray(bin_edges, dtype="float64")
        if edges.size < 3:
            return

        yt_te = np.asarray(y_true_te, dtype="float64")
        yt_te = yt_te[np.isfinite(yt_te)]
        yt_te = np.abs(yt_te)

        yt_tr = None
        if y_true_tr is not None:
            yt_tr = np.asarray(y_true_tr, dtype="float64")
            yt_tr = yt_tr[np.isfinite(yt_tr)]
            yt_tr = np.abs(yt_tr)

        counts_te, _ = np.histogram(yt_te, bins=edges)
        mids = 0.5 * (edges[:-1] + edges[1:])

        plt.figure(figsize=(11, 5.5))
        plt.bar(mids, counts_te, width=np.diff(edges), align="center")
        plt.axhline(float(min_samples_per_bin_te), linestyle="--")

        if yt_tr is not None and yt_tr.size > 0:
            counts_tr, _ = np.histogram(yt_tr, bins=edges)
            if counts_tr.max() > 0 and counts_te.max() > 0:
                counts_tr_scaled = counts_tr * (counts_te.max() / counts_tr.max())
            else:
                counts_tr_scaled = counts_tr
            plt.plot(mids, counts_tr_scaled)

        for e in edges:
            plt.axvline(float(e), linestyle=":", linewidth=0.8)

        if strict_max is not None:
            plt.axvline(float(strict_max), linestyle="-.", linewidth=1.5)
        if relaxed_max is not None:
            plt.axvline(float(relaxed_max), linestyle="--", linewidth=1.5)

        plt.xlabel("Depth magnitude (m)")
        plt.ylabel("Count per bin (TE bars; TR overlay)")
        plt.title(title)
        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception:
        try:
            plt.close()
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


def plot_feature_importance(rf_model, feature_names, out_png):
    """Generates a bar chart of feature importance."""
    global plt
    if plt is None:
        plt = lazy_pyplot()
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)), sorted_vals, align="center",
           color="skyblue", edgecolor="black")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right")
    ax.set_ylabel("Gini Importance")
    ax.set_title("Random Forest Feature Importance")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info(f"[PLOT] Feature importance saved: {out_png}")

def _count(df: pd.DataFrame, label: str):
    log.info(f"[TRAIN][COUNT] {label}: n={len(df)}")

def _nonfinite_report(df: pd.DataFrame, cols: List[str], label: str, max_lines: int = 30):
    if df.empty:
        log.warning(f"[TRAIN][DIAG] {label}: df is empty; cannot compute non-finite fractions.")
        return
    lines = []
    n = len(df)
    for c in cols:
        if c not in df.columns:
            continue
        v = df[c].to_numpy()
        bad = np.sum(~np.isfinite(v))
        if bad > 0:
            lines.append((bad / n, c, int(bad)))
    if not lines:
        log.info(f"[TRAIN][DIAG] {label}: all requested cols are finite.")
        return
    lines.sort(reverse=True)
    log.warning(f"[TRAIN][DIAG] {label}: non-finite values detected (showing up to {max_lines}).")
    for frac, c, bad in lines[:max_lines]:
        log.warning(f"  - {c}: {bad}/{n} non-finite ({frac*100:.1f}%)")


def _funnel_df_stats(df: pd.DataFrame, stage: str, *, rr: Optional[Any] = None, depth_col: str = "depth_m",
                     hist_max_m: float = 40.0, hist_bin_m: float = 1.0) -> None:
    try:
        n = int(len(df)) if df is not None else 0
    except Exception:
        n = 0
    if df is None or n == 0:
        log.info(f"[Funnel][TRAIN] {stage}: n=0")
        if rr is not None:
            try:
                rr.add(f"funnel.train.{stage}.n", 0)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        return

    d = None
    if depth_col in df.columns:
        try:
            d = pd.to_numeric(df[depth_col], errors="coerce").to_numpy(dtype="float64")
        except Exception:
            d = None

    if d is None:
        log.info(f"[Funnel][TRAIN] {stage}: n={n} (no {depth_col} column)")
        if rr is not None:
            try:
                rr.add(f"funnel.train.{stage}.n", n)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        return

    mfin = np.isfinite(d)
    nfin = int(np.count_nonzero(mfin))
    if nfin:
        p50 = float(np.nanpercentile(d[mfin], 50))
        p95 = float(np.nanpercentile(d[mfin], 95))
        dmax = float(np.nanmax(d[mfin]))
    else:
        p50 = p95 = dmax = float("nan")

    log.info(f"[Funnel][TRAIN] {stage}: n={n} finite={nfin} depth_p50/p95/max={p50:.3f}/{p95:.3f}/{dmax:.3f} m")

    hist = None
    edges = None
    try:
        if nfin:
            dabs = np.abs(d[mfin])
            edges = np.arange(0.0, float(hist_max_m) + float(hist_bin_m), float(hist_bin_m))
            hist, _ = np.histogram(dabs, bins=edges)
    except Exception:
        hist = None
        edges = None

    if rr is not None:
        try:
            rr.add(f"funnel.train.{stage}.n", n)
            rr.add(f"funnel.train.{stage}.finite_n", nfin)
            rr.add_dict(f"funnel.train.{stage}.depth_stats_m", {"p50": p50, "p95": p95, "max": dmax})
            if hist is not None and edges is not None:
                rr.add_dict(f"funnel.train.{stage}.depth_hist_0_{int(hist_max_m)}_{int(hist_bin_m)}m",
                            {"bins": edges.tolist(), "counts": hist.tolist()})
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

def sample_s2_bands_at_points(train_df: pd.DataFrame,
                             s2_paths: Dict[str, str],
                             land_mask_path: Optional[str]) -> pd.DataFrame:
    """Samples S2 rasters at training point locations."""
    if train_df.empty:
        return train_df

    log.info(f"[SAMPLE] Sampling S2 features at {len(train_df)} locations...")
    lon_values = train_df["longitude"].to_numpy(np.float64)
    lat_values = train_df["latitude"].to_numpy(np.float64)

    sampling_targets = {
        "B02": s2_paths["B02"], "B03": s2_paths["B03"], "B04": s2_paths["B04"],
        "B08": s2_paths["B08"], "CLEAR_WATER": s2_paths["CLEAR_WATER"],
        "brightness": s2_paths["BRIGHTNESS"],
    }
    if land_mask_path:
        sampling_targets["LAND"] = land_mask_path

    sampled_data: Dict[str, np.ndarray] = {}

    for col, path in sampling_targets.items():
        if not os.path.exists(path):
            log.warning(f"[SAMPLE] Missing raster: {path}. Filling {col} with NaNs.")
            sampled_data[col] = np.full(len(train_df), np.nan, dtype=np.float32)
            continue

        try:
            with rasterio.open(path) as ds:
                tfm = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
                x, y = tfm.transform(lon_values, lat_values)
                XY = np.column_stack([x, y])

                it = ds.sample(XY)
                arr = np.fromiter((v[0] for v in it), dtype=np.float32, count=len(XY))

                nod = ds.nodata
                if col not in ("CLEAR_WATER", "LAND"):
                    if nod is not None and np.isfinite(nod):
                        arr[arr == np.float32(nod)] = np.nan
                else:
                    pass

                if col in ("B02", "B03", "B04", "B08", "brightness"):
                    arr[arr <= 0.0] = np.nan

                sampled_data[col] = arr

        except Exception as exc:
            log.warning(f"[SAMPLE] Failed to sample {col}: {exc}")
            sampled_data[col] = np.full(len(train_df), np.nan, dtype=np.float32)

    df_out = train_df.copy()
    for col, arr in sampled_data.items():
        df_out[col] = arr
    return df_out

def _sanitize_feature_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------

def train_sdb_model(
    train_df: pd.DataFrame,
    max_depth_sdb: float,
    seed: int,
    plots_dir: Path,
    water_class: str,
    use_stumpf_depth: bool,
    min_training_points_for_sdb: int,
    spatial_split: bool = False,
    spatial_cv_enabled: bool = False,
    spatial_cv_strategy: str = "spatial_cluster",
    spatial_cv_folds: int = 5,
    cw_min: float = 0.5,
    land_max: float = 0.0,
    land_mask_type: str = "auto",
    land_mask_water_val: Optional[int] = None,
    land_mask_invert: bool = False,
    land_mask_threshold: float = 0.5,
    land_mask_nodata_is_water: bool = True,
    linf_enabled: bool = False,
    linf_estimate: str = "none",
    linf_deepwater_nir_max: float = 0.03,
    linf_deepwater_bright_max: float = 0.15,
    linf_percentile: float = 1.0,
    raster_paths: dict = None,
    rmse_target_sdb: float = 0.5,
    depth_bin_m: float = 0.5,
    depth_binning: str = "quantile",
    max_depth_bins: int = 30,
    min_samples_per_bin: int = 200,
    max_depth_source: str = "physics",  # physics, rmse, combined, training_p95
    diagnostics_dir: Optional[Path] = None,  # Directory for JSON reports
    rr: Optional[Any] = None,
    model_bank_dir: Optional[Path] = None,
    model_bank_enabled: bool = True,
    model_bank_max_samples: int = 100000,
    model_bank_seed: int = 1337,
    model_bank_retrain_min_new: int = 2000,
) -> Tuple[RandomForestRegressor, Optional[LinearRegression], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    log.info("--- Stage 3: Feature Engineering and Model Training ---")
    if not S2_OPTICS_AVAILABLE:
        _bind_s2_optics_functions()
    if not S2_OPTICS_AVAILABLE:
        log.warning(">> 's2_optics' module not found via lazy import. Using dummy generators.")

    df = train_df.copy()

    if "source" in df.columns:
        df["source"] = df["source"].astype(str)
        df["source_norm"] = df["source"].astype(str).str.lower()
    else:
        df["source_norm"] = ""

    _count(df, "start (input to train_sdb_model)")
    _funnel_df_stats(df, "start", rr=rr)

    if "LAND" not in df.columns or land_max is None:
        m_land = np.ones(len(df), dtype=bool)
    else:
        land_v = pd.to_numeric(df["LAND"], errors="coerce").to_numpy(dtype="float64")
        finite = np.isfinite(land_v)
        n_fin = int(finite.sum())
        
        if land_mask_nodata_is_water:
            nodata_ok = ~finite
        else:
            nodata_ok = np.zeros(len(df), dtype=bool)

        lt = (str(land_mask_type or "auto").lower()).strip()
        u = np.unique(land_v[finite]) if n_fin else np.array([])
        is_binaryish = (u.size > 0 and u.size <= 4 and np.all(np.isin(u, [0.0, 1.0])))
        is_probish = (n_fin > 0 and float(np.nanmin(land_v)) >= 0.0 and float(np.nanmax(land_v)) <= 1.0 and (not is_binaryish))

        if lt in ("land_probability", "probability") or (lt == "auto" and is_probish):
            thr = float(land_mask_threshold if land_mask_threshold is not None else land_max)
            m_land = nodata_ok | (finite & (land_v <= float(thr)))
            log.info(f"[TRAIN][LAND] using land_probability semantics: keep LAND<= {thr} plus nodata_ok={land_mask_nodata_is_water}")
        else:
            if lt == "auto" and is_binaryish:
                if u.size == 1:
                    m_land = nodata_ok | (finite & (land_v <= float(land_max)))
                else:
                    if land_mask_water_val is None:
                        vv = land_v[finite & np.isfinite(land_v)]
                        if vv.size > 0:
                            vals, counts = np.unique(vv.astype(np.int64), return_counts=True)
                            wv = float(vals[int(np.argmax(counts))])
                            log.info(f"[TRAIN][LAND] auto-inferred water_val={wv} from training points distribution: {dict(zip(vals.tolist(), counts.tolist()))}")
                        else:
                            wv = 0.0
                            log.warning("[TRAIN][LAND] Could not infer water_val (no finite LAND samples); falling back to 0.")
                    else:
                        wv = float(land_mask_water_val)
                    keep_eq = (finite & (land_v == wv))
                    m_land = nodata_ok | (~keep_eq if bool(land_mask_invert) else keep_eq)
                    log.info(f"[TRAIN][LAND] using water_val={wv} invert={bool(land_mask_invert)} nodata_ok={land_mask_nodata_is_water}")
            elif lt in ("water_only", "land_binary", "binary", "mask"):
                if land_mask_water_val is None:
                    vv = land_v[finite & np.isfinite(land_v)]
                    if vv.size > 0:
                        vals, counts = np.unique(vv.astype(np.int64), return_counts=True)
                        wv = float(vals[int(np.argmax(counts))])
                        log.info(f"[TRAIN][LAND] auto-inferred water_val={wv} from training points distribution: {dict(zip(vals.tolist(), counts.tolist()))}")
                    else:
                        wv = 0.0
                        log.warning("[TRAIN][LAND] Could not infer water_val (no finite LAND samples); falling back to 0.")
                else:
                    wv = float(land_mask_water_val)
                keep_eq = (finite & (land_v == wv))
                m_land = nodata_ok | (~keep_eq if bool(land_mask_invert) else keep_eq)
                log.info(f"[TRAIN][LAND] using explicit semantics: water_val={wv} invert={bool(land_mask_invert)} nodata_ok={land_mask_nodata_is_water}")
            else:
                m_land = nodata_ok | (finite & (land_v <= float(land_max)))
                log.info(f"[TRAIN][LAND] using threshold semantics: keep LAND<= {land_max} plus nodata_ok={land_mask_nodata_is_water}")

    if cw_min is None or "CLEAR_WATER" not in df.columns:
        m_cw = np.ones(len(df), dtype=bool)
    else:
        cw_v = pd.to_numeric(df["CLEAR_WATER"], errors="coerce").to_numpy(dtype="float64")
        m_cw = np.isfinite(cw_v) & (cw_v >= float(cw_min))

    m_env = m_land & m_cw

    # FIX: extra_xyz points should bypass BOTH land AND clear_water filters
    # because they are typically high-quality surveyed data (hydronos, ehydro, etc.)
    # that may be in areas where S2-derived masks are unreliable (turbid coastal zones)
    if "source_norm" in df.columns:
        m_is_xyz = (df["source_norm"] == "extra_xyz")
        n_xyz_before = int(m_is_xyz.sum())
        m_env |= m_is_xyz
        if n_xyz_before > 0:
            log.info(f"[TRAIN][ENV] Bypassed env filter for {n_xyz_before} extra_xyz points (high-quality survey data)")

    df = df[m_env].reset_index(drop=True)
    _count(df, f"after env filter (LAND<= {land_max}, CLEAR_WATER>= {cw_min})")
    _funnel_df_stats(df, "after_env_filter", rr=rr)

    if df.empty:
        log.error("[TRAIN] No samples remain after environment filter.")
        return RandomForestRegressor(), None, pd.DataFrame(), pd.DataFrame(), {}

    l_inf_constants = {"B02": 0.0, "B03": 0.0, "B04": 0.0, "B08": 0.0}
    if linf_enabled:
        if str(linf_estimate).lower() == "deepwater":
            est = {}
            try:
                if raster_paths:
                    est = estimate_linf_from_rasters(
                        raster_paths,
                        nir_max=linf_deepwater_nir_max,
                        bright_max=linf_deepwater_bright_max,
                        percentile=linf_percentile,
                    )
                    if est:
                        log.info(f"[TRAIN] Estimated L_inf constants from rasters: {est}")
            except Exception as e:
                log.warning(f"[TRAIN] Raster-based L_inf estimation failed; falling back to DF method. Reason: {e}")

            if not est:
                est = estimate_linf_from_df(
                    df,
                    nir_max=linf_deepwater_nir_max,
                    bright_max=linf_deepwater_bright_max,
                    percentile=linf_percentile,
                    cw_col="CLEAR_WATER" if "CLEAR_WATER" in df.columns else None,
                )
                if est:
                    log.info(f"[TRAIN] Estimated L_inf constants from deepwater (DF fallback): {est}")

            if est:
                l_inf_constants = est
            else:
                log.warning("[TRAIN] L_inf estimate failed (no suitable deepwater pixels found). Using zeros.")

    df = add_s2_optical_features(df, l_inf=l_inf_constants)
    _count(df, "after add_s2_optical_features")
    _funnel_df_stats(df, "after_s2_features", rr=rr)

    try:
        df = apply_s2_brightness_depth_filter(df, water_class=water_class)
    except TypeError:
        df = apply_s2_brightness_depth_filter(df, wc=water_class)

    _count(df, "after brightness/depth filter")
    _funnel_df_stats(df, "after_brightness_filter", rr=rr)

    df = apply_stumpf_residual_filter(
        df,
        enabled=True,
        residual_threshold_std=2.5,
        residual_abs_min_m=0.5,
        min_points=200,
        min_inliers=100,
        min_depth_m=0.25,
        max_depth_m=max_depth_sdb,
        n_clusters=2,
        random_state=seed,
    )
    _count(df, "after stumpf residual filter")
    _funnel_df_stats(df, "after_stumpf_residual_filter", rr=rr)

    feat_cols = [
        "B02", "B03", "B04", "B08",
        "log_B02", "log_B03", "log_B04", "log_B08",
        "brightness", "B03_B02", "B04_B03", "nbri", "stumpf_idx",
    ]

    stumpf_lr: Optional[LinearRegression] = None
    if use_stumpf_depth and "stumpf_idx" in df.columns:
        x = pd.to_numeric(df["stumpf_idx"], errors="coerce").to_numpy(np.float32)
        y_raw = pd.to_numeric(df["depth_m"], errors="coerce").to_numpy(np.float32)
        # Use positive magnitude for Stumpf LR (consistent with RF training)
        y = np.abs(y_raw)
        m = np.isfinite(x) & np.isfinite(y)

        if np.sum(m) >= 20:
            try:
                stumpf_lr = LinearRegression()
                stumpf_lr.fit(x[m].reshape(-1, 1), y[m])

                all_x = x.reshape(-1, 1)
                pred = np.full(len(df), np.nan, dtype=np.float32)
                pred[m] = stumpf_lr.predict(all_x[m]).astype(np.float32)
                df["stumpf_depth"] = pred

                feat_cols.append("stumpf_depth")
                log.info("[TRAIN] Fitted auxiliary stumpf_depth LR model (positive magnitudes).")
            except Exception as e:
                log.warning(f"[TRAIN] Stumpf LR failed: {e}")

    df = _sanitize_feature_columns(df, feat_cols + ["depth_m", "sample_weight"])
    _count(df, "after sanitize (inf->nan, coercion)")

    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0

    req_cols = feat_cols + ["depth_m", "sample_weight"]
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        log.error(f"[TRAIN] Missing required columns: {missing_cols}. Training aborted.")
        return RandomForestRegressor(), stumpf_lr, pd.DataFrame(), pd.DataFrame(), {}

    _nonfinite_report(df, req_cols, "pre finite-row drop")
    mat = df[req_cols].to_numpy(np.float32)
    m_all = np.all(np.isfinite(mat), axis=1)
    df = df[m_all].reset_index(drop=True)
    _count(df, "after finite-row drop")
    _funnel_df_stats(df, "after_finite_row_drop", rr=rr)

    training_bounds: Dict[str, Dict[str, float]] = {}
    for col in feat_cols:
        vals = df[col].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            p_min, p_max = np.percentile(vals, [0.5, 99.5])
            buff = (p_max - p_min) * 0.1
            training_bounds[col] = {"min": float(p_min - buff), "max": float(p_max + buff)}
        else:
            training_bounds[col] = {"min": -9999.0, "max": 9999.0}

    log.info(f"[TRAIN] Calculated Domain of Applicability bounds for {len(feat_cols)} features.")

    doa_weights: Dict[str, float] = {}
    try:
        doa_weights = {c: 1.0 / max(len(feat_cols), 1) for c in feat_cols}
    except Exception:
        doa_weights = {c: 1.0 / max(len(feat_cols), 1) for c in feat_cols}

    metadata: Dict[str, Any] = {
        "feature_columns": feat_cols,
        "linf_enabled": bool(linf_enabled),
        "linf_estimate": str(linf_estimate),
        "linf_deepwater_nir_max": float(linf_deepwater_nir_max),
        "linf_deepwater_bright_max": float(linf_deepwater_bright_max),
        "linf_percentile": float(linf_percentile),
        "l_inf_constants": _to_python_float(l_inf_constants) if linf_enabled else None,
        "linf": _to_python_float(l_inf_constants) if linf_enabled else None,
        "training_bounds": _to_python_float(training_bounds),
        "doa": {
            "mode": "soft_exp",
            "soft_k_default": 3.0,
            "threshold_default": 0.90,
            "weights": _to_python_float(doa_weights),
            "bounds_method": {"percentiles": [0.5, 99.5], "buffer_frac": 0.10},
        },
        "water_class": water_class,
        "max_depth_sdb": max_depth_sdb,
        "cw_min": float(cw_min) if cw_min is not None else None,
        "land_max": float(land_max) if land_max is not None else None,
    }

    metadata["rmse_target_sdb"] = float(rmse_target_sdb)
    metadata["depth_bin_m"] = float(depth_bin_m)
    metadata["min_samples_per_bin"] = int(min_samples_per_bin)
    metadata["max_depth_sdb_auto"] = None
    metadata["max_depth_sdb_auto_diagnostics"] = {}


    # --- MODEL BANK (bounded reservoir) ---
    bank_meta = None
    bank_dir = None
    if model_bank_enabled and model_bank_dir is not None:
        try:
            from model_bank import update_bank
            bank_dir = Path(model_bank_dir)

            # Stable, minimal reservoir schema: only what we need to train/predict.
            # This keeps the on-disk reservoir bounded even as upstream feature
            # engineering evolves.
            bank_schema_cols = []
            try:
                bank_schema_cols = list(dict.fromkeys(
                    list(feat_cols) + ["longitude", "latitude", "source", "source_norm", "depth_m"]
                ))
            except Exception:
                bank_schema_cols = ["longitude", "latitude", "source", "source_norm", "depth_m"]

            bank_df, bank_meta = update_bank(
                bank_dir,
                df,
                target_col='depth_m',
                max_samples=int(model_bank_max_samples),
                seed=int(model_bank_seed),
                extra_meta={
                    'water_class': water_class,
                    'max_depth_sdb': float(max_depth_sdb) if max_depth_sdb is not None else None,
                    'linf_enabled': bool(linf_enabled),
                    'linf_estimate': str(linf_estimate),
                },
                schema_cols=bank_schema_cols,
            )
            # Train from bank reservoir for cross-tile consistency
            if isinstance(bank_df, pd.DataFrame) and len(bank_df) > 0:
                df = bank_df
                _count(df, f'after model bank update (n_kept={bank_meta.get("n_kept")}, n_seen={bank_meta.get("n_seen")})')
                _funnel_df_stats(df, 'after_model_bank', rr=rr)
            metadata['model_bank'] = {
                'enabled': True,
                'dir': str(bank_dir),
                'max_samples': int(bank_meta.get('max_samples', model_bank_max_samples)) if bank_meta else int(model_bank_max_samples),
                'n_seen': int(bank_meta.get('n_seen', 0)) if bank_meta else 0,
                'n_kept': int(bank_meta.get('n_kept', 0)) if bank_meta else 0,
                'last_added': int(bank_meta.get('last_added', 0)) if bank_meta else 0,
                'last_replaced': int(bank_meta.get('last_replaced', 0)) if bank_meta else 0,
            }
        except Exception as ex:
            log.warning(f"[MODEL_BANK] Update failed; continuing without bank: {ex}")
            metadata['model_bank'] = {'enabled': False, 'error': str(ex)}
    else:
        metadata['model_bank'] = {'enabled': False}

    # If using model bank, optionally reuse an existing trained model until enough new samples accumulate.
    # This provides stability across adjacent tiles and avoids overfitting to tiny incremental updates.
    if model_bank_enabled and model_bank_dir is not None:
        try:
            bank_dir_p = Path(model_bank_dir)
            rf_p = bank_dir_p / "rf_model.pkl"
            meta_p = bank_dir_p / "bank_meta.json"
            last_trained_n_seen = 0
            n_seen_now = 0
            if meta_p.exists():
                try:
                    with open(meta_p, "r") as _f:
                        _bm = json.load(_f)
                    last_trained_n_seen = int(_bm.get("last_trained_n_seen", 0))
                    n_seen_now = int(_bm.get("n_seen", 0))
                except Exception:
                    last_trained_n_seen = 0
                    n_seen_now = 0
            new_since_train = max(0, n_seen_now - last_trained_n_seen) if n_seen_now else 0

            if rf_p.exists() and (new_since_train < int(model_bank_retrain_min_new)):
                # Reuse the last bank model for this run.
                rf_reuse = joblib.load(rf_p)
                lr_reuse = None
                lr_p = bank_dir_p / "stumpf_lr.pkl"
                if lr_p.exists():
                    try:
                        lr_reuse = joblib.load(lr_p)
                    except Exception:
                        lr_reuse = None
                # Try to load model meta if present; otherwise keep metadata as-is.
                mm_p = bank_dir_p / "model_meta.json"
                if mm_p.exists():
                    try:
                        with open(mm_p, "r") as _f:
                            mm = json.load(_f)
                        metadata.setdefault("model_bank", {})
                        metadata["model_bank"]["reused_model"] = True
                        metadata["model_bank"]["new_since_train"] = int(new_since_train)
                        metadata["model_bank"]["retrain_min_new"] = int(model_bank_retrain_min_new)
                        metadata["model_bank"]["reused_model_meta"] = True
                        # Propagate reuse flags into the returned model_meta so callers can
                        # distinguish "reused" from "failed training".
                        try:
                            if isinstance(mm, dict):
                                mm.setdefault("model_bank", {})
                                mm["model_bank"].update({
                                    "enabled": True,
                                    "reused_model": True,
                                    "new_since_train": int(new_since_train),
                                    "retrain_min_new": int(model_bank_retrain_min_new),
                                })
                        except Exception:
                            pass
                        return rf_reuse, lr_reuse, pd.DataFrame(), pd.DataFrame(), mm
                    except Exception:
                        pass

                metadata.setdefault("model_bank", {})
                metadata["model_bank"]["reused_model"] = True
                metadata["model_bank"]["new_since_train"] = int(new_since_train)
                metadata["model_bank"]["retrain_min_new"] = int(model_bank_retrain_min_new)
                return rf_reuse, lr_reuse, pd.DataFrame(), pd.DataFrame(), metadata
        except Exception as ex:
            log.debug(f"[MODEL_BANK] Reuse check failed; proceeding to retrain: {ex}", exc_info=True)

    if len(df) < min_training_points_for_sdb:
        log.warning(f"[TRAIN] Insufficient samples ({len(df)} < {min_training_points_for_sdb}).")

        # If a model bank exists, prefer reusing its last trained model rather than returning
        # an untrained RF (which can later look like a "successful" run but produce nonsense).
        if model_bank_enabled and model_bank_dir is not None:
            try:
                bank_dir_p = Path(model_bank_dir)
                rf_p = bank_dir_p / "rf_model.pkl"
                if rf_p.exists():
                    rf_reuse = joblib.load(rf_p)
                    lr_reuse = None
                    lr_p = bank_dir_p / "stumpf_lr.pkl"
                    if lr_p.exists():
                        try:
                            lr_reuse = joblib.load(lr_p)
                        except Exception:
                            lr_reuse = None

                    mm = None
                    mm_p = bank_dir_p / "model_meta.json"
                    if mm_p.exists():
                        try:
                            with open(mm_p, "r") as _f:
                                mm = json.load(_f)
                        except Exception:
                            mm = None

                    metadata.setdefault("model_bank", {})
                    metadata["model_bank"].update({
                        "enabled": True,
                        "reused_model": True,
                        "reuse_reason": "insufficient_new_samples",
                        "min_training_points_for_sdb": int(min_training_points_for_sdb),
                        "n_samples": int(len(df)),
                    })

                    if isinstance(mm, dict):
                        mm.setdefault("model_bank", {})
                        mm["model_bank"].update(metadata["model_bank"])
                        return rf_reuse, lr_reuse, pd.DataFrame(), pd.DataFrame(), mm

                    return rf_reuse, lr_reuse, pd.DataFrame(), pd.DataFrame(), metadata
            except Exception as ex:
                log.debug(f"[MODEL_BANK] Insufficient-sample reuse failed; returning empty model: {ex}", exc_info=True)

        metadata.setdefault("train_status", {})
        metadata["train_status"].update({
            "ok": False,
            "reason": "insufficient_samples",
            "min_training_points_for_sdb": int(min_training_points_for_sdb),
            "n_samples": int(len(df)),
        })
        # Return a placeholder model object to satisfy caller contracts, but mark metadata as failed.
        return RandomForestRegressor(), stumpf_lr, pd.DataFrame(), pd.DataFrame(), metadata


    atl_like = {"atl03", "atl24", "atl_agreed", "atl03+atl24_agree", "atl03_atl24_agree"}
    df_atl = df[df["source_norm"].isin(atl_like)].copy()

    if df_atl.empty:
        log.info("[TRAIN] No ATL-like sources found. Training solely on non-ATL data.")
        df_atl = df.copy()

    # --- TRAIN/TEST SPLIT LOGIC ---
    if not spatial_split:
        # UPDATED: Use Quantile-Stratified Split for Random mode
        log.info("[TRAIN] Performing Stratified Random Split (by Depth Quantile)...")
        idx_tr, idx_te = stratified_train_test_split(
            df_atl, 
            target_col='depth_m', 
            test_size=0.2, 
            seed=seed
        )
    else:
        # UPDATED: Smart Spatial Split
        # Pick the spatial cluster that best represents the full depth range (esp. deep water).
        log.info("[TRAIN] Performing Spatial Split (K-Means Clustering)...")
        coords = df_atl[["longitude", "latitude"]].to_numpy()

        # Robustness: KMeans can fail (or behave poorly) when sample counts are small.
        # Also, sklearn versions prior to 1.4 may not accept n_init=10.
        do_spatial = coords.shape[0] >= 200
        if not do_spatial:
            log.warning(
                f"[TRAIN] Spatial split requested but too few samples for stable clustering (n={coords.shape[0]}). "
                "Falling back to stratified random split."
            )
            idx_tr, idx_te = stratified_train_test_split(
                df_atl,
                target_col='depth_m',
                test_size=0.2,
                seed=seed
            )
        else:
            # Choose a conservative cluster count based on sample size.
            # Keep this stable to reduce tile-to-tile variability.
            n_clusters = int(min(5, max(2, coords.shape[0] // 500)))
            try:
                km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(coords)
            except Exception:
                # Last-resort fallback
                km = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(coords)
        
        if do_spatial:
            # Analyze clusters to pick a "good" test set (one that isn't just shallow)
            # We want a test cluster that has a p95 depth similar to the global p95.
            global_p95 = np.percentile(df_atl['depth_m'], 95)
            best_k = 0
            best_score = float('inf')
            
            stats_by_k = {}
            for k in range(int(km.n_clusters)):
                mask = (km.labels_ == k)
                n_k = int(np.sum(mask))
                if n_k < 50:
                    continue  # Skip tiny clusters

                d_k = df_atl.loc[df_atl.index[mask], 'depth_m']
                p95_k = float(np.percentile(d_k, 95))

                # Score: diff in p95 depth (lower is better) so test set spans deep water too
                score = float(abs(p95_k - global_p95))
                stats_by_k[k] = {'n': n_k, 'p95': p95_k, 'score': score}

                if score < best_score:
                    best_score = score
                    best_k = k

            p95_best = float(stats_by_k.get(best_k, {}).get('p95', float('nan')))
            log.info(f"[TRAIN] Spatial Split: Global p95={global_p95:.2f}m. Selected Cluster {best_k} as Test (p95={p95_best:.2f}m).")
            
            idx_te = df_atl.index[km.labels_ == best_k].to_numpy()
            idx_tr = df_atl.index[km.labels_ != best_k].to_numpy()

    idx_xyz = df[df["source_norm"].astype(str).str.lower().str.startswith("extra_xyz")].index.to_numpy()
    final_tr_indices = np.unique(np.concatenate([idx_tr, idx_xyz]))

    df_tr = df.loc[final_tr_indices].copy()
    df_te = df.loc[idx_te].copy()


    # Defensive: ensure depth_m is numeric in BOTH train and test before any np.isfinite checks
    def _coerce_depth_m(_df, _label):
        if _df is None or _df.empty or ('depth_m' not in _df.columns):
            return _df
        _df2 = _df.copy()
        d = pd.to_numeric(_df2['depth_m'], errors='coerce').to_numpy()
        m = np.isfinite(d)
        n_drop = int((~m).sum())
        if n_drop:
            log.warning(f"[TRAIN] Dropping {n_drop} rows with non-finite depth_m in {_label}.")
        _df2 = _df2.loc[m].copy()
        _df2['depth_m'] = d[m]
        return _df2

    df_tr = _coerce_depth_m(df_tr, 'train set')
    df_te = _coerce_depth_m(df_te, 'test set')

    _count(df_tr, "final train set")
    _count(df_te, "final test set")
    _funnel_df_stats(df_tr, "split_train", rr=rr)
    _funnel_df_stats(df_te, "split_test", rr=rr)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1
    )

    df_tr_fit = df_tr
    # Ensure depth_m is numeric (extra XYZ can carry strings/objects depending on ingest path).
    # Log the effect so we don't silently train on a tiny subset.
    if "depth_m" in df_tr_fit.columns:
        _n0 = len(df_tr_fit)
        df_tr_fit = df_tr_fit.copy()
        df_tr_fit["depth_m"] = pd.to_numeric(df_tr_fit["depth_m"], errors="coerce")
        m_depth = np.isfinite(df_tr_fit["depth_m"].to_numpy())
        _dropped = int((~m_depth).sum())
        if _dropped:
            log.warning(f"[TRAIN][QC] Dropping {_dropped:,}/{_n0:,} rows with non-finite depth_m after coercion")
        df_tr_fit = df_tr_fit.loc[m_depth].copy()
    if max_depth_sdb is not None:
        try:
            _md = float(max_depth_sdb)
            m_fit = np.abs(df_tr_fit["depth_m"].to_numpy()) <= _md
            df_tr_fit = df_tr_fit.loc[m_fit].copy()
        except (TypeError, ValueError):
            # Handle "auto" or other non-float strings gracefully
            pass
        
    # Enforce a final source quota at the fit target to prevent one source from dominating
    # the RF (e.g., massive extra_xyz Hydronos/eHydro contributions).
    def _enforce_final_source_quota(_df, _seed, _target_n, _max_source_frac=0.85):
        if _df is None or _df.empty:
            return _df
        source_col_local = 'source_norm' if 'source_norm' in _df.columns else ('source' if 'source' in _df.columns else None)
        if source_col_local is None:
            return _df
        _df = _df.copy()
        _df[source_col_local] = _df[source_col_local].astype(str)
        n0 = int(len(_df))
        target_n = int(max(1, min(int(_target_n), n0)))
        max_source_frac = float(_max_source_frac)
        quota_n = int(max(1, np.floor(max_source_frac * target_n)))
        vc0 = _df[source_col_local].value_counts(dropna=False)
        dom0 = float(vc0.iloc[0] / max(1, n0)) if len(vc0) else 0.0
        needs_quota = bool((vc0 > quota_n).any()) or (n0 > target_n) or (dom0 > max_source_frac + 1e-9)
        if not needs_quota:
            return _df

        rng = np.random.default_rng(int(_seed))

        # If there is only one source, a max-source-fraction quota is impossible by definition.
        # Still downsample to target_n (if needed), but log clearly that the quota cannot be satisfied.
        if len(vc0) <= 1:
            keep_idx = _df.index.to_numpy()
            if keep_idx.size > target_n:
                keep_idx = rng.choice(keep_idx, size=target_n, replace=False)
            keep_idx = np.sort(keep_idx)
            df_out = _df.loc[keep_idx].copy()
            vc1 = df_out[source_col_local].value_counts(dropna=False)
            dom1 = float(vc1.iloc[0] / max(1, len(df_out))) if len(vc1) else 0.0
            log.warning(
                f"[TRAIN][QC] Final source quota cannot be satisfied with a single source. "
                f"Applied target downsampling only: {n0:,} -> {len(df_out):,} rows; dominant source fraction remains {dom1:.3f}."
            )
            log.info(f"[TRAIN][QC] Source counts before quota: {vc0.to_dict()}")
            log.info(f"[TRAIN][QC] Source counts after quota: {vc1.to_dict()}")
            return df_out

        # Build shuffled indices per source (stable seed) so any trimming is deterministic/reproducible.
        src_idx = {}
        init_counts = {}
        for src, grp in _df.groupby(source_col_local, sort=False):
            idx = grp.index.to_numpy()
            if idx.size > 1:
                rng.shuffle(idx)
            src = str(src)
            src_idx[src] = idx
            init_counts[src] = int(min(idx.size, quota_n))

        # First pass enforces quota relative to target_n.
        counts = {k: int(v) for k, v in init_counts.items()}

        # If capped pool exceeds target_n, randomly trim across capped pool (fractions can only improve).
        total_capped = int(sum(counts.values()))
        if total_capped > target_n:
            capped_idx = np.concatenate([src_idx[s][:counts[s]] for s in counts if counts[s] > 0])
            keep_idx = rng.choice(capped_idx, size=target_n, replace=False)
            keep_idx = np.sort(keep_idx)
            df_out = _df.loc[keep_idx].copy()
        else:
            # IMPORTANT: Do NOT refill from leftovers. Leftovers come from groups already at quota_n.
            # Refilling would violate the source-fraction invariant.
            if total_capped < target_n:
                log.warning(
                    f"[TRAIN][QC] Final source quota limits available rows below target: "
                    f"target={target_n:,}, quota-limited rows={total_capped:,}. Applying strict fraction cap on actual rows."
                )

            # Second pass enforces max_source_frac on the ACTUAL retained row count (not target_n),
            # because target-based caps alone can still yield >max_source_frac when the quota-limited
            # pool is much smaller than target_n (e.g., one huge source + tiny minority sources).
            # Iteratively trim any source exceeding floor(f * total) until stable.
            # Convergence can require many iterations in extreme dominance cases (e.g., 99:1).
            # Cost is trivial because the number of sources is small.
            for _ in range(1000):
                total_now = int(sum(counts.values()))
                if total_now <= 0:
                    break
                allowed_now = int(np.floor(max_source_frac * total_now))
                changed = False
                for s in list(counts.keys()):
                    if counts[s] > allowed_now:
                        counts[s] = int(allowed_now)
                        changed = True
                if not changed:
                    break

            total_now = int(sum(counts.values()))
            if total_now <= 0:
                # Fallback: keep the smallest feasible multi-source subset (1 row from up to two sources)
                # to avoid returning an empty frame if the iterative cap collapses due to tiny counts.
                nonempty = [s for s, idx in src_idx.items() if len(idx) > 0]
                seed_rows = []
                for s in nonempty[:2]:
                    seed_rows.append(src_idx[s][0])
                keep_idx = np.sort(np.array(seed_rows, dtype=_df.index.dtype)) if seed_rows else np.array([], dtype=_df.index.dtype)
                df_out = _df.loc[keep_idx].copy()
                log.warning(
                    "[TRAIN][QC] Strict actual-row source quota collapsed sample count severely; "
                    f"falling back to minimal multi-source subset (n={len(df_out)})."
                )
            else:
                keep_parts = [src_idx[s][:counts[s]] for s in counts if counts[s] > 0]
                keep_idx = np.sort(np.concatenate(keep_parts)) if keep_parts else np.array([], dtype=_df.index.dtype)
                df_out = _df.loc[keep_idx].copy()

        vc1 = df_out[source_col_local].value_counts(dropna=False)
        dom1 = float(vc1.iloc[0] / max(1, len(df_out))) if len(vc1) else 0.0
        log.warning(
            f"[TRAIN][QC] Enforced final source quota (target={target_n:,}, max_source_frac={max_source_frac:.2f}, quota={quota_n:,}): "
            f"{n0:,} -> {len(df_out):,} rows. Dominant source fraction {dom0:.3f} -> {dom1:.3f}."
        )
        log.info(f"[TRAIN][QC] Source counts before quota: {vc0.to_dict()}")
        log.info(f"[TRAIN][QC] Source counts after quota: {vc1.to_dict()}")
        return df_out


    try:
        final_target_n = int(max(5000, min(len(df_tr_fit), int(model_bank_max_samples))))
    except Exception:
        final_target_n = int(max(5000, len(df_tr_fit)))
    df_tr_fit = _enforce_final_source_quota(df_tr_fit, seed, final_target_n, _max_source_frac=0.85)

    missing = [c for c in feat_cols if (c not in df_tr_fit.columns) or (c not in df_te.columns)]
    if missing:
        log.warning(f"[TRAIN] Dropping missing feature columns: {missing}")
        feat_cols = [c for c in feat_cols if c not in missing]

    X_train = df_tr_fit[feat_cols].to_numpy()
    y_train_raw = df_tr_fit["depth_m"].to_numpy()
    w_train = df_tr_fit["sample_weight"].to_numpy() if "sample_weight" in df_tr_fit.columns else None

    # Refresh arrays after any optional rebalancing / QC changes to df_tr_fit
    X_train = df_tr_fit[feat_cols].to_numpy()
    y_train_raw = df_tr_fit["depth_m"].to_numpy()
    w_train = df_tr_fit["sample_weight"].to_numpy() if "sample_weight" in df_tr_fit.columns else None

    # === CRITICAL: Validate depth sign convention ===
    depths_finite = y_train_raw[np.isfinite(y_train_raw)]
    
    if len(depths_finite) == 0:
        log.error("[TRAIN] CRITICAL: No finite depth values in training data!")
        return RandomForestRegressor(), stumpf_lr, pd.DataFrame(), pd.DataFrame(), {}
    
    pct_negative = (depths_finite < 0).mean()
    pct_positive = (depths_finite > 0).mean()
    
    log.info(f"[TRAIN] Depth sign distribution: {pct_negative*100:.1f}% negative, {pct_positive*100:.1f}% positive")
    log.info(f"[TRAIN] Depth range: [{depths_finite.min():.2f}, {depths_finite.max():.2f}] m")
    
    # Error if mostly positive (wrong convention)
    if pct_positive > 0.9:
        log.error("[TRAIN] ❌ DEPTH CONVENTION ERROR: 90%+ depths are POSITIVE")
        log.error("[TRAIN]    Bathymetry depths should be NEGATIVE (below surface)")
        log.error("[TRAIN]    Check your XYZ data depth column and sign convention")
        log.error("[TRAIN]    Expected: water depth = negative value")
        raise ValueError("Invalid depth sign convention: depths should be negative (below surface)")
    
    # Warning if mixed or mostly positive
    if pct_negative < 0.8:
        log.warning(f"[TRAIN] ⚠️  DEPTH CONVENTION WARNING: Only {pct_negative*100:.1f}% depths are negative")
        log.warning(f"[TRAIN]    Expected: depths below surface should be negative")
        log.warning(f"[TRAIN]    Verify depth sign convention in source data")

    # === CRITICAL: Convert to positive magnitude for training ===
    # The model should predict POSITIVE depth magnitudes.
    # Input depths are negative (below surface), we convert to positive for training.
    # Prediction outputs will be positive, then negated in predict.py for final output.
    y_train = np.abs(y_train_raw)
    log.info(f"[TRAIN] Converted depths to positive magnitudes for training: [{y_train[np.isfinite(y_train)].min():.2f}, {y_train[np.isfinite(y_train)].max():.2f}] m")

    # === Guardrail: avoid single-source domination (especially extra_xyz hydronos/ehydro) ===
    if 'source' in df_tr_fit.columns or 'source_norm' in df_tr_fit.columns:
        try:
            source_col = 'source_norm' if 'source_norm' in df_tr_fit.columns else 'source'
            vc = df_tr_fit[source_col].value_counts(dropna=False)
            total_n = int(len(df_tr_fit))
            if total_n > 0 and len(vc) >= 2:
                dom_src = vc.index[0]
                dom_n = int(vc.iloc[0])
                dom_frac = dom_n / float(total_n)
                metadata.setdefault('training_qc', {})
                metadata['training_qc']['dominant_source'] = str(dom_src)
                metadata['training_qc']['dominant_source_frac'] = float(dom_frac)
                if dom_frac > 0.90:
                    log.warning(
                        f"[TRAIN][QC] Dominant source '{dom_src}' contributes {dom_frac*100:.1f}% of training rows ({dom_n}/{total_n}). "
                        "Expect weak generalization / source-specific bias."
                    )
                # Conservative deterministic cap: only trim if one source dominates AND others exist.
                max_dom_frac = 0.70
                if dom_frac > 0.90 and total_n >= 1000:
                    allowed_dom = int(max(100, (max_dom_frac / max(1e-9, 1.0 - max_dom_frac)) * (total_n - dom_n)))
                    if allowed_dom < dom_n:
                        dom_idx = df_tr_fit.index[df_tr_fit[source_col] == dom_src].to_numpy()
                        keep_dom = np.random.default_rng(int(seed)).choice(dom_idx, size=allowed_dom, replace=False)
                        keep_other = df_tr_fit.index[df_tr_fit[source_col] != dom_src].to_numpy()
                        keep_idx = np.concatenate([keep_other, keep_dom])
                        df_tr_fit = df_tr_fit.loc[keep_idx].copy()
                        # refresh arrays after rebalance
                        X_train = None
                        y_train_raw = None
                        w_train = None
                        log.warning(
                            f"[TRAIN][QC] Rebalanced dominant source '{dom_src}' to reduce count domination: "
                            f"{total_n} -> {len(df_tr_fit)} rows (dominant kept={allowed_dom})."
                        )
                        metadata['training_qc']['dominant_source_rebalanced'] = True
                        metadata['training_qc']['dominant_source_rebalanced_target_frac'] = float(max_dom_frac)
        except Exception as _ex:
            log.error(f"[TRAIN][QC] Dominance guardrail failed: {_ex}")

    # === NEW: Log training data composition by source ===
    if 'source' in df_tr_fit.columns or 'source_norm' in df_tr_fit.columns:
        source_col = 'source_norm' if 'source_norm' in df_tr_fit.columns else 'source'
        log.info("=" * 60)
        log.info("[TRAIN] TRAINING DATA COMPOSITION:")
        log.info("=" * 60)
        
        source_counts = df_tr_fit[source_col].value_counts()
        total_samples = len(df_tr_fit)
        
        for source in source_counts.index:
            count = source_counts[source]
            pct = 100.0 * count / total_samples
            
            # Get sample weights
            mask = df_tr_fit[source_col] == source
            if w_train is not None:
                weights = df_tr_fit.loc[mask, 'sample_weight']
                mean_weight = weights.mean()
                total_weight = weights.sum()
                log.info(f"  {source:20s}: n={count:6d} ({pct:5.1f}%), weight_mean={mean_weight:5.1f}, weight_total={total_weight:8.0f}")
            else:
                log.info(f"  {source:20s}: n={count:6d} ({pct:5.1f}%)")
        
        if w_train is not None:
            total_weighted = df_tr_fit['sample_weight'].sum()
            log.info(f"\n  Total samples: {total_samples:,}")
            log.info(f"  Total weighted: {total_weighted:,.0f}")
            
            # Effective contribution
            log.info(f"\n  Effective training influence:")
            for source in source_counts.index:
                mask = df_tr_fit[source_col] == source
                weight_contrib = df_tr_fit.loc[mask, 'sample_weight'].sum()
                effective_pct = 100.0 * weight_contrib / total_weighted
                log.info(f"    {source:20s}: {effective_pct:5.1f}%")
        
        try:
            if w_train is not None and total_weighted > 0:
                eff_fracs = []
                for source in source_counts.index:
                    mask = df_tr_fit[source_col] == source
                    weight_contrib = float(df_tr_fit.loc[mask, 'sample_weight'].sum())
                    eff_fracs.append(weight_contrib / float(total_weighted))
                if eff_fracs:
                    metadata.setdefault('training_qc', {})
                    metadata['training_qc']['max_effective_source_influence_frac'] = float(max(eff_fracs))
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        log.info("=" * 60)

    rf.fit(X_train, y_train, sample_weight=w_train)
    log.info(f"[TRAIN] RF trained on {len(df_tr_fit)} samples. (Test set: {len(df_te)})")

    # --- Run Spatial Cross-Validation if enabled ---
    spatial_cv_summary = None
    if spatial_cv_enabled and len(df) >= 200:
        try:
            from spatial_cv import run_spatial_cv, plot_spatial_cv_results
            log.info(f"[TRAIN] Running {spatial_cv_folds}-fold spatial CV ({spatial_cv_strategy})...")
            
            spatial_cv_summary = run_spatial_cv(
                df,
                feature_cols=feat_cols,
                target_col="depth_m",
                cv_strategy=spatial_cv_strategy,
                n_folds=spatial_cv_folds,
                seed=seed,
            )
            
            if plots_dir and spatial_cv_summary.n_folds > 0:
                plot_spatial_cv_results(
                    spatial_cv_summary,
                    plots_dir / "Spatial_CV_Results.png",
                    title=f"Spatial CV ({spatial_cv_strategy}, {spatial_cv_folds} folds)"
                )
                
            log.info(
                f"[TRAIN] Spatial CV complete: RMSE={spatial_cv_summary.rmse_mean:.3f}±{spatial_cv_summary.rmse_std:.3f}m, "
                f"R²={spatial_cv_summary.r2_mean:.3f}±{spatial_cv_summary.r2_std:.3f}"
            )
        except ImportError:
            log.debug("[TRAIN] spatial_cv module not available")
        except Exception as e:
            log.warning(f"[TRAIN] Spatial CV failed: {e}")

    # --- Training Data Diversity Analysis ---
    diversity_report = None
    try:
        from training_diversity import add_diversity_analysis_to_training
        diversity_report = add_diversity_analysis_to_training(df, plots_dir, feat_cols)
        if diversity_report and "overall_score" in diversity_report:
            log.info(f"[TRAIN] Data diversity score: {diversity_report['overall_score']*100:.0f}%")
            if diversity_report.get("recommendations"):
                for rec in diversity_report["recommendations"][:3]:
                    log.info(f"[TRAIN] → {rec}")
    except ImportError:
        log.debug("[TRAIN] training_diversity module not available")
    except Exception as e:
        log.warning(f"[TRAIN] Diversity analysis failed: {e}")

    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_feature_importance(rf, feat_cols, plots_dir / "Feature_Importance.png")

    if not df_te.empty:
        # RF predicts positive magnitudes; convert to negative for comparison with depth_m
        df_te["depth_pred_m"] = -np.abs(rf.predict(df_te[feat_cols].to_numpy()).astype(np.float32))
        split_name = "Spatial" if spatial_split else "Random"
        title = f"SDB Accuracy ({split_name} Split)"
        
        out_png = plots_dir / f"SDB_Accuracy_Assessment_{split_name}Split.png"
        
        scatter_plot(
            df_te["depth_m"].to_numpy(),
            df_te["depth_pred_m"].to_numpy(),
            title,
            out_png,
            max_depth=max_depth_sdb
        )

        # === NEW: Source-specific validation ===
        if 'source' in df_te.columns or 'source_norm' in df_te.columns:
            source_col = 'source_norm' if 'source_norm' in df_te.columns else 'source'
            log.info("=" * 60)
            log.info("[TRAIN] VALIDATION BY DATA SOURCE:")
            log.info("=" * 60)
            
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            source_validation = {}
            for source in df_te[source_col].unique():
                mask = df_te[source_col] == source
                n_source = mask.sum()
                
                if n_source < 3:
                    continue
                
                y_true = df_te.loc[mask, 'depth_m'].values
                y_pred = df_te.loc[mask, 'depth_pred_m'].values
                
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred) if n_source >= 5 else np.nan
                
                source_validation[source] = {
                    'n': int(n_source),
                    'rmse_m': float(rmse),
                    'mae_m': float(mae),
                    'r2': float(r2) if np.isfinite(r2) else None,
                    'depth_range': [float(y_true.min()), float(y_true.max())]
                }
                
                # FIX: Safe format string handling for R²
                r2_str = f"{r2:5.3f}" if np.isfinite(r2) else "  N/A"
                log.info(
                    f"  {source:20s}: n={n_source:5d}, RMSE={rmse:5.2f}m, "
                    f"MAE={mae:5.2f}m, R²={r2_str}"
                )
            
            log.info("=" * 60)
            
            # Save to metadata
            if rr is not None:
                try:
                    if hasattr(rr, "add"):
                        rr.add("validation.by_source", source_validation)
                    elif hasattr(rr, "data") and isinstance(rr.data, dict):
                        rr.data["validation.by_source"] = source_validation
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        max_depth_sdb_auto = None
        max_depth_diag = {}
        # UPDATED: Run auto-depth diagnostics regardless of split type if target is set
        if (spatial_split or max_depth_sdb == "auto") or (rmse_target_sdb is not None):
            try:
                # Use a safe cap for calculation if max_depth_sdb is "auto" string
                calc_cap = 50.0 
                if isinstance(max_depth_sdb, (float, int)):
                    calc_cap = float(max_depth_sdb)

                depths, max_depth_diag = estimate_max_depth_from_spatial_validation_dual(
                    y_true_te=df_te["depth_m"].to_numpy(),
                    y_pred_te=df_te["depth_pred_m"].to_numpy(),
                    y_true_tr=df_tr["depth_m"].to_numpy(),
                    rmse_target_m=rmse_target_sdb,
                    depth_bin_m=depth_bin_m,
                    depth_binning=depth_binning,
                    max_depth_bins=max_depth_bins,
                    min_samples_per_bin_te=min_samples_per_bin,
                    min_samples_per_bin_tr=max(50, int(min_samples_per_bin) // 4),
                    max_depth_cap_m=calc_cap,
                    max_consecutive_empty_bins_relaxed=4,
                )
                max_depth_sdb_auto = depths.get("strict", None)
                max_depth_sdb_auto_relaxed = depths.get("relaxed", None)

                if plots_dir and isinstance(max_depth_diag, dict) and max_depth_diag.get("bin_edges_m"):
                    plot_depth_binning_sanity(
                        df_te["depth_m"].to_numpy(),
                        y_true_tr=(df_tr["depth_m"].to_numpy() if ("df_tr" in locals() and df_tr is not None and not df_tr.empty) else None),
                        bin_edges=max_depth_diag.get("bin_edges_m"),
                        out_png=plots_dir / "Depth_Binning_Sanity.png",
                        title=f"Depth binning sanity ({depth_binning})",
                        min_samples_per_bin_te=min_samples_per_bin,
                        strict_max=max_depth_sdb_auto,
                        relaxed_max=max_depth_sdb_auto_relaxed,
                    )

            except Exception as e:
                log.exception("[TRAIN] Auto max-depth estimation failed")

        metadata["rmse_target_sdb"] = float(rmse_target_sdb)
        metadata["depth_bin_m"] = float(depth_bin_m)
        metadata["min_samples_per_bin"] = int(min_samples_per_bin)

        metadata["max_depth_sdb_auto"] = None if max_depth_sdb_auto is None else float(max_depth_sdb_auto)
        metadata["max_depth_sdb_auto_strict"] = metadata["max_depth_sdb_auto"]
        metadata["max_depth_sdb_auto_relaxed"] = None if (locals().get("max_depth_sdb_auto_relaxed", None) is None) else float(max_depth_sdb_auto_relaxed)
        metadata["max_depth_sdb_auto_diagnostics"] = max_depth_diag

        # --- Physics-based depth limit (Kd estimation) ---
        physics_depth_result = None
        if raster_paths is not None:
            try:
                from kd_estimation import compute_physics_based_max_depth
                physics_depth_result = compute_physics_based_max_depth(
                    raster_paths,
                    rmse_based_max=max_depth_sdb_auto,
                    algorithm="lee2005",
                    confidence_level="moderate",
                )
                
                metadata["kd_estimation"] = physics_depth_result
                
                # Log the results
                if physics_depth_result.get("physics_max_depth_m") is not None:
                    kd_median = physics_depth_result.get("kd_490_median", 0)
                    water_type = physics_depth_result.get("water_type", "unknown")
                    phys_max = physics_depth_result["physics_max_depth_m"]
                    combined = physics_depth_result.get("combined_max_depth_m")
                    limiting = physics_depth_result.get("limiting_factor", "unknown")
                    
                    log.info(
                        f"[TRAIN] Physics-based depth: Kd(490)={kd_median:.3f} m⁻¹ ({water_type}), "
                        f"optical limit={phys_max:.1f}m"
                    )
                    
                    # ALWAYS store physics-based max depth when computed
                    metadata["max_depth_sdb_auto_physics"] = float(phys_max)
                    metadata["kd_490_median"] = float(kd_median)
                    metadata["water_type"] = water_type
                    
                    if combined is not None:
                        log.info(
                            f"[TRAIN] Operational depth cap candidate (combined physics+rmse): {combined:.1f}m "
                            f"(limited by {limiting})"
                        )
                        metadata["max_depth_sdb_combined"] = float(combined)
                        metadata["max_depth_limiting_factor"] = limiting
                        
            except ImportError:
                log.debug("[TRAIN] kd_estimation module not available, skipping physics-based depth")
            except Exception as e:
                log.warning(f"[TRAIN] Physics-based depth estimation failed: {e}")

        # --- Physics-based enhancements (Kim et al. 2024) ---
        # Estimate scene-specific bottom endmembers and geometry-corrected attenuation
        if PHYSICS_AVAILABLE and raster_paths is not None:
            try:
                # Get sun/view angles from S2 metadata
                s2_dir = Path(raster_paths.get("B02", "")).parent if raster_paths else None
                sza, vza = physics_integration.get_sun_view_angles_from_s2_metadata(
                    str(s2_dir) if s2_dir else None
                )
                
                log.info(f"[TRAIN][PHYSICS] Sun zenith: {sza:.1f}°, View zenith: {vza:.1f}°")
                
                # Store angles in metadata
                metadata["physics"] = {
                    "sun_zenith_deg": float(sza),
                    "view_zenith_deg": float(vza),
                }
                
                # Geometry-corrected Kd/Ku if we have Kd estimate
                def _safe_float(x):
                    """Coerce a value to float without raising (handles dict/None)."""
                    try:
                        if isinstance(x, dict):
                            # Common patterns: {'value': v} or {'kd': v} etc.
                            for k in ("value", "kd", "ku", "mean"):
                                if k in x:
                                    return float(x[k])
                            return float("nan")
                        return float(x)
                    except Exception:
                        return float("nan")

                kd_490 = metadata.get("kd_490_median")
                if kd_490 is not None and kd_490 > 0:
                    kd_corr, ku_corr = physics_integration.compute_geometry_corrected_attenuation(
                        kd_490, sza_deg=sza, vza_deg=vza
                    )
                    # physics_integration returns per-band dicts
                    metadata["physics"]["kd_corrected"] = {k: _safe_float(v) for k, v in (kd_corr or {}).items()}
                    metadata["physics"]["ku_corrected"] = {k: _safe_float(v) for k, v in (ku_corr or {}).items()}
                    try:
                        kd_mean = float(np.nanmean(list(metadata["physics"]["kd_corrected"].values()))) if metadata["physics"]["kd_corrected"] else float('nan')
                        ku_mean = float(np.nanmean(list(metadata["physics"]["ku_corrected"].values()))) if metadata["physics"]["ku_corrected"] else float('nan')
                        log.info(f"[TRAIN][PHYSICS] Geometry-corrected (mean over bands): Kd={kd_mean:.4f}, Ku={ku_mean:.4f}")
                    except Exception:
                        log.debug("[TRAIN][PHYSICS] Geometry correction summary failed", exc_info=True)
                
                # Estimate bottom endmembers from scene
                try:
                    endmember_result = physics_integration.estimate_scene_bottom_endmembers(
                        raster_paths,
                        kd_490=kd_490 if kd_490 else 0.1,
                        sza_deg=sza,
                        n_clusters=3,
                        min_depth_m=0.5,
                        max_depth_m=5.0,
                    )
                    
                    if endmember_result.get("status") == "ok":
                        metadata["physics"]["bottom_endmembers"] = endmember_result.get("endmembers", {})
                        metadata["physics"]["endmember_method"] = "eigenanalysis"
                        
                        # Check for seagrass signature
                        sand = endmember_result.get("endmembers", {}).get("sand", {})
                        if sand.get("Green", 0) > sand.get("Blue", 0):
                            metadata["physics"]["seagrass_detected"] = True
                            log.info("[TRAIN][PHYSICS] ⚠️ Seagrass signature detected (Green > Blue in shallow)")
                        else:
                            metadata["physics"]["seagrass_detected"] = False
                        
                        log.info(f"[TRAIN][PHYSICS] Bottom endmembers estimated: {list(endmember_result.get('endmembers', {}).keys())}")
                    else:
                        log.debug(f"[TRAIN][PHYSICS] Endmember estimation skipped: {endmember_result.get('reason', 'unknown')}")
                        
                except Exception as e:
                    log.debug(f"[TRAIN][PHYSICS] Endmember estimation failed: {e}")
                    
            except Exception as e:
                log.warning(f"[TRAIN][PHYSICS] Physics integration failed: {e}")
        elif not PHYSICS_AVAILABLE:
            log.debug("[TRAIN] physics_integration module not available")

        # =========================================================================
        # CANONICAL max_depth_sdb_final - THE SINGLE SOURCE OF TRUTH
        # =========================================================================
        # Priority depends on max_depth_source parameter:
        #   - "physics": Use Kd-based optical limit only
        #   - "rmse": Use RMSE-based depth-of-support only  
        #   - "combined": Use min(physics, rmse)
        #   - "training_p95": Use 95th percentile of training depths
        final_depth = None
        final_depth_source = "default"
        
        physics_max = metadata.get("max_depth_sdb_auto_physics")
        rmse_max = metadata.get("max_depth_sdb_auto")
        combined_max = metadata.get("max_depth_sdb_combined")
        
        # Get training depth stats
        try:
            d_vals = df["depth_m"].to_numpy(dtype=float)
            training_p95 = float(_np.nanpercentile(d_vals[_np.isfinite(d_vals)], 95))
            training_max = float(_np.nanmax(d_vals[_np.isfinite(d_vals)]))
        except Exception:
            training_p95 = None
            training_max = None
        
        # Select based on max_depth_source
        if max_depth_source == "physics":
            if physics_max is not None:
                final_depth = float(physics_max)
                final_depth_source = "physics_kd"
            elif combined_max is not None:
                # Fall back to combined if physics alone not available
                final_depth = float(combined_max)
                final_depth_source = "combined_fallback"
            else:
                log.warning("[TRAIN] Physics-based depth not available, falling back to training_p95")
                if training_p95 is not None:
                    final_depth = training_p95
                    final_depth_source = "training_p95_fallback"
                    
        elif max_depth_source == "rmse":
            if rmse_max is not None:
                final_depth = float(rmse_max)
                final_depth_source = "rmse_depth_of_support"
            else:
                log.warning("[TRAIN] RMSE-based depth not available, falling back to training_p95")
                if training_p95 is not None:
                    final_depth = training_p95
                    final_depth_source = "training_p95_fallback"
                    
        elif max_depth_source == "combined":
            if combined_max is not None:
                final_depth = float(combined_max)
                final_depth_source = "combined_physics_rmse"
            elif physics_max is not None and rmse_max is not None:
                final_depth = min(float(physics_max), float(rmse_max))
                final_depth_source = "combined_computed"
            elif physics_max is not None:
                final_depth = float(physics_max)
                final_depth_source = "physics_only"
            elif rmse_max is not None:
                final_depth = float(rmse_max)
                final_depth_source = "rmse_only"
            else:
                if training_p95 is not None:
                    final_depth = training_p95
                    final_depth_source = "training_p95_fallback"
                    
        elif max_depth_source == "training_p95":
            if training_p95 is not None:
                final_depth = training_p95
                final_depth_source = "training_p95"
        
        # Final fallback if nothing worked
        if final_depth is None:
            if isinstance(max_depth_sdb, (int, float)) and str(max_depth_sdb).lower() != "auto":
                final_depth = float(max_depth_sdb)
                final_depth_source = "input_param"
            elif training_p95 is not None:
                final_depth = training_p95
                final_depth_source = "training_p95_final_fallback"
            else:
                final_depth = 20.0
                final_depth_source = "hardcoded_default"
        
        # Store all computed limits for reference
        metadata["max_depth_options"] = {
            "physics_kd": physics_max,
            "rmse_depth_of_support": rmse_max,
            "combined": combined_max,
            "training_p95": training_p95,
            "training_max": training_max,
            "selected_source": max_depth_source,
        }
        
        metadata["max_depth_sdb_final"] = final_depth
        metadata["max_depth_sdb_final_source"] = final_depth_source
        log.info(
            f"[TRAIN] Canonical requested-source depth candidate (max_depth_sdb_final) = "
            f"{final_depth:.2f}m (source: {final_depth_source}, requested: {max_depth_source})"
        )
        if combined_max is not None and physics_max is not None and str(max_depth_source).lower() == "physics":
            try:
                if np.isfinite(float(combined_max)) and np.isfinite(float(final_depth)) and float(combined_max) < float(final_depth):
                    log.info(
                        f"[TRAIN] Note: RMSE-constrained combined/operational cap is {float(combined_max):.2f}m "
                        f"(< physics candidate {float(final_depth):.2f}m); downstream product code may enforce the stricter cap."
                    )
            except Exception:
                pass

        # Add spatial CV results to metadata
        if spatial_cv_summary is not None and spatial_cv_summary.n_folds > 0:
            metadata["spatial_cv"] = {
                "enabled": True,
                "strategy": spatial_cv_strategy,
                "n_folds": spatial_cv_summary.n_folds,
                "rmse_mean": spatial_cv_summary.rmse_mean,
                "rmse_std": spatial_cv_summary.rmse_std,
                "rmse_per_fold": spatial_cv_summary.rmse_per_fold,
                "r2_mean": spatial_cv_summary.r2_mean,
                "r2_std": spatial_cv_summary.r2_std,
                "r2_per_fold": spatial_cv_summary.r2_per_fold,
                "mae_mean": spatial_cv_summary.mae_mean,
                "bias_mean": spatial_cv_summary.bias_mean,
            }
            
            # Use spatial CV RMSE for more conservative max depth estimation
            if spatial_cv_summary.rmse_mean > 0:
                # Estimate max depth where spatial CV RMSE < target
                # This is a more honest estimate of generalization performance
                metadata["spatial_cv_rmse_for_depth"] = spatial_cv_summary.rmse_mean
        
        # Add diversity analysis to metadata
        if diversity_report and "overall_score" in diversity_report:
            metadata["training_diversity"] = diversity_report
                
    try:
        import numpy as _np
        from pathlib import Path as _Path
        rep = {"train": {}}
        rep["train"]["n_input_samples"] = int(len(train_df)) if train_df is not None else None
        rep["train"]["n_after_feature_engineering"] = int(len(df))
        rep["train"]["n_train"] = int(len(df_tr))
        rep["train"]["n_test"] = int(len(df_te))
        try:
            rep["train"]["max_depth_sdb_auto_m"] = metadata.get("max_depth_sdb_auto")
            rep["train"]["max_depth_sdb_auto_diagnostics"] = metadata.get("max_depth_sdb_auto_diagnostics", {})
            rep["train"]["rmse_target_sdb_m"] = metadata.get("rmse_target_sdb")
            rep["train"]["depth_bin_m"] = metadata.get("depth_bin_m")
            rep["train"]["min_samples_per_bin"] = metadata.get("min_samples_per_bin")
            
            # CANONICAL max depth values - these are what sdb_main.py should use
            rep["train"]["max_depth_sdb_final"] = metadata.get("max_depth_sdb_final")
            rep["train"]["max_depth_sdb_final_source"] = metadata.get("max_depth_sdb_final_source")
            rep["train"]["max_depth_sdb_auto_physics"] = metadata.get("max_depth_sdb_auto_physics")
            rep["train"]["max_depth_options"] = metadata.get("max_depth_options", {})
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        try:
            if df_te is not None and (not df_te.empty) and ("depth_pred_m" in df_te.columns) and ("depth_m" in df_te.columns):
                _bins = _np.array([0, 5, 10, 15, 20, 30, 50], dtype=float)
                _y = df_te["depth_m"].to_numpy(dtype=float)
                _yp = df_te["depth_pred_m"].to_numpy(dtype=float)
                _res = _yp - _y
                _bias = {}
                _rmse = {}
                _counts = {}
                for _i in range(len(_bins) - 1):
                    _lo, _hi = float(_bins[_i]), float(_bins[_i + 1])
                    _m = _np.isfinite(_y) & _np.isfinite(_yp) & (_y >= _lo) & (_y < _hi)
                    _n = int(_m.sum())
                    if _n:
                        _r = _res[_m]
                        _bias[f"{_lo:g}-{_hi:g}m"] = float(_np.mean(_r))
                        _rmse[f"{_lo:g}-{_hi:g}m"] = float(_np.sqrt(_np.mean(_r * _r)))
                        _counts[f"{_lo:g}-{_hi:g}m"] = _n
                rep["train"]["bias_per_depth_bin_m"] = _bias
                rep["train"]["rmse_per_depth_bin_m"] = _rmse
                rep["train"]["n_per_depth_bin"] = _counts
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        if "depth_m" in df.columns and len(df) > 0:
            d = df["depth_m"].to_numpy(dtype=float)
            d_abs = _np.abs(d)  # Use positive magnitudes for stats
            rep["train"]["depth_stats_m"] = {
                "min": float(_np.nanmin(d_abs)),
                "p50": float(_np.nanpercentile(d_abs, 50)),
                "p95": float(_np.nanpercentile(d_abs, 95)),
                "max": float(_np.nanmax(d_abs)),
            }
            bins = _np.array([0,1,2,3,5,7.5,10,15,20,30,50], dtype=float)
            hist, edges = _np.histogram(d_abs[_np.isfinite(d_abs)], bins=bins)
            rep["train"]["depth_hist_bins_m"] = edges.tolist()
            rep["train"]["depth_hist_counts"] = hist.tolist()
        try:
            if "source_norm" in df.columns:
                _src_stats = {}
                _mean_depth = float(_np.nanmean(df["depth_m"].to_numpy(dtype=float))) if "depth_m" in df.columns and len(df) else None
                for _src, _grp in df.groupby("source_norm"):
                    _d = _grp["depth_m"].to_numpy(dtype=float) if "depth_m" in _grp.columns else _np.array([], dtype=float)
                    _d = _d[_np.isfinite(_d)]
                    _src_stats[str(_src)] = {
                        "count": int(len(_grp)),
                        "depth_p50": float(_np.percentile(_d, 50)) if _d.size else None,
                        "depth_p95": float(_np.percentile(_d, 95)) if _d.size else None,
                        "bias_vs_mean": (float(_np.mean(_d) - _mean_depth) if (_d.size and _mean_depth is not None) else None),
                    }
                rep["train"]["source_breakdown"] = _src_stats
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        try:
            imps = rf.feature_importances_
            pairs = list(zip(feat_cols, [float(x) for x in imps]))
            pairs.sort(key=lambda x: x[1], reverse=True)
            rep["train"]["feature_importance"] = [{"feature": k, "importance": v} for k, v in pairs[:15]]
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        try:
            top_feats = [p["feature"] for p in rep["train"].get("feature_importance", [])[:8]]
            fstats = {}
            for c in top_feats:
                if c in df.columns:
                    v = df[c].to_numpy(dtype=float)
                    v = v[_np.isfinite(v)]
                    if v.size:
                        fstats[c] = {
                            "p1": float(_np.percentile(v, 1)),
                            "p50": float(_np.percentile(v, 50)),
                            "p99": float(_np.percentile(v, 99)),
                        }
            rep["train"]["feature_stats_top"] = fstats
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        metadata["train_report"] = rep["train"]
        # Write to diagnostics dir if provided, otherwise fallback to plots_dir parent
        if diagnostics_dir is not None:
            report_dir = _Path(diagnostics_dir)
        else:
            report_dir = _Path(plots_dir).parent if plots_dir is not None else _Path(".")
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "train_report.json", "w") as f:
            import json as _json
            _json.dump(rep, f, indent=2)
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    return rf, stumpf_lr, df_tr, df_te, metadata



def main():
    parser = argparse.ArgumentParser(description="Standalone Model Trainer")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-model-dir", required=True)
    parser.add_argument("--s2-dir", default=None, help="If provided, samples S2 bands.")
    parser.add_argument("--cw-min", type=float, default=None, help="Minimum CLEAR_WATER fraction for training filter (None disables).")
    parser.add_argument("--land-max", type=float, default=None, help="Maximum LAND fraction for training filter (None disables).")
    parser.add_argument("--max-depth", type=float, default=20.0)

    parser.add_argument(
        "--s2-choice",
        choices=["auto", "composite", "best_date"],
        default="auto",
        help="Which S2 raster set to use when --s2-dir is provided. "
             "'auto' trains both (if available) and chooses by RMSE/coherence.",
    )
    parser.add_argument(
        "--s2-ab-rmse-margin",
        type=float,
        default=0.05,
        help="RMSE margin (m) required to prefer one variant; otherwise use coherence as tiebreaker.",
    )

    parser.add_argument("--rmse-target-sdb", type=float, default=0.5,
                        help="Target spatial RMSE (m) used to estimate max supported SDB depth (for metadata/reporting).")
    parser.add_argument("--depth-bin-m", type=float, default=0.5,
                        help="Depth bin width (m) used for spatial error-vs-depth diagnostics (default: 0.5).")
    parser.add_argument("--depth-binning", choices=["fixed","quantile"], default="quantile",
        help="Depth binning mode for spatial error-vs-depth diagnostics. fixed=uniform bins (depth-bin-m). quantile=equal-count bins targeting min-samples-per-bin.")
    parser.add_argument("--max-depth-bins", type=int, default=30,
        help="Maximum number of depth bins when --depth-binning quantile (default: 30).")
    parser.add_argument("--min-samples-per-bin", type=int, default=200,
                        help="Minimum spatial validation samples per depth bin for auto depth-of-support diagnostics.")

    parser.add_argument(
        "--validate-spatial",
        dest="validate_spatial",
        action="store_true",
        default=True,
        help="Use spatial (clustered) train/test split (default: on).",
    )
    parser.add_argument(
        "--no-validate-spatial",
        dest="validate_spatial",
        action="store_false",
        help="Disable spatial validation and use random split.",
    )

    parser.add_argument("--linf-enabled", action="store_true",
                        help="Enable L_inf (dark-object subtraction) correction. Requires constants or estimation.")
    parser.add_argument("--linf-estimate", choices=["none", "deepwater"], default="none",
                        help="If linf-enabled and constants not provided, estimate L_inf from deepwater-like samples.")
    parser.add_argument("--linf-deepwater-nir-max", type=float, default=0.03)
    parser.add_argument("--linf-deepwater-bright-max", type=float, default=0.15)
    parser.add_argument("--linf-percentile", type=float, default=1.0,
                        help="Percentile used to estimate L_inf (e.g., 1.0).")

    args = parser.parse_args()

    if args.validate_spatial:
        log.info("[TRAIN] Spatial validation ENABLED (default)")
    else:
        log.warning("[TRAIN] Spatial validation DISABLED — using random split")

    out_dir = Path(args.out_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    df0 = pd.read_csv(args.train_csv)

    chosen = None
    chosen_reason = "single_run"
    ab_summary = None

    s2_paths = None
    land = None

    if args.s2_dir:
        s2_d = Path(args.s2_dir)
        land = str(s2_d / "LAND_MASK_aligned.tif") if (s2_d / "LAND_MASK_aligned.tif").exists() else None

        comp_paths = _resolve_s2_paths(s2_d, suffix="")
        best_paths = _resolve_s2_paths(s2_d, suffix="_best_date")

        has_comp = _s2_paths_exist(comp_paths)
        has_best = _s2_paths_exist(best_paths)

        if not has_comp:
            log.warning(f"[S2-AB] Composite rasters not found/complete in: {s2_d}. Proceeding without raster sampling.")
        else:
            if args.s2_choice == "composite" or (args.s2_choice == "auto" and not has_best):
                if args.s2_choice == "auto" and not has_best:
                    log.info("[S2-AB] Best-date rasters not found; using composite.")
                df = sample_s2_bands_at_points(df0, comp_paths, land)
                s2_paths = comp_paths

                rf, lr, _, df_te, meta = train_sdb_model(
                    train_df=df,
                    max_depth_sdb=args.max_depth,
                    seed=42,
                    plots_dir=plot_dir,
                    water_class="clear_ocean",
                    use_stumpf_depth=True,
                    min_training_points_for_sdb=10,
                    spatial_split=args.validate_spatial,
                    cw_min=getattr(args, "cw_min", None),
                    land_max=getattr(args, "land_max", None),
                    linf_enabled=args.linf_enabled,
                    linf_estimate=args.linf_estimate,
                    linf_deepwater_nir_max=args.linf_deepwater_nir_max,
                    linf_deepwater_bright_max=args.linf_deepwater_bright_max,
                    linf_percentile=args.linf_percentile,
                    raster_paths=s2_paths,
                    rmse_target_sdb=args.rmse_target_sdb,
                    depth_bin_m=args.depth_bin_m,
                    min_samples_per_bin=args.min_samples_per_bin,
                    depth_binning=args.depth_binning,
                    max_depth_bins=args.max_depth_bins,
                )
                chosen = {"rf": rf, "lr": lr, "meta": meta, "variant": "composite"}

            elif args.s2_choice == "best_date":
                if not has_best:
                    log.warning("[S2-AB] Requested best_date but best-date rasters not found; falling back to composite.")
                    df = sample_s2_bands_at_points(df0, comp_paths, land)
                    s2_paths = comp_paths
                    variant = "composite"
                else:
                    df = sample_s2_bands_at_points(df0, best_paths, land)
                    s2_paths = best_paths
                    variant = "best_date"

                rf, lr, _, df_te, meta = train_sdb_model(
                    train_df=df,
                    max_depth_sdb=args.max_depth,
                    seed=42,
                    plots_dir=plot_dir,
                    water_class="clear_ocean",
                    use_stumpf_depth=True,
                    min_training_points_for_sdb=10,
                    spatial_split=args.validate_spatial,
                    cw_min=getattr(args, "cw_min", None),
                    land_max=getattr(args, "land_max", None),
                    linf_enabled=args.linf_enabled,
                    linf_estimate=args.linf_estimate,
                    linf_deepwater_nir_max=args.linf_deepwater_nir_max,
                    linf_deepwater_bright_max=args.linf_deepwater_bright_max,
                    linf_percentile=args.linf_percentile,
                    raster_paths=s2_paths,
                    rmse_target_sdb=args.rmse_target_sdb,
                    depth_bin_m=args.depth_bin_m,
                    min_samples_per_bin=args.min_samples_per_bin,
                    depth_binning=args.depth_binning,
                    max_depth_bins=args.max_depth_bins,
                )
                chosen = {"rf": rf, "lr": lr, "meta": meta, "variant": variant}

            else:
                log.info("[S2-AB] Running A/B training: composite vs best_date")
                (plot_dir / "composite").mkdir(parents=True, exist_ok=True)
                (plot_dir / "best_date").mkdir(parents=True, exist_ok=True)
                df_comp = sample_s2_bands_at_points(df0, comp_paths, land)
                rf_c, lr_c, _, df_te_c, meta_c = train_sdb_model(
                    train_df=df_comp,
                    max_depth_sdb=args.max_depth,
                    seed=42,
                    plots_dir=(plot_dir / "composite"),
                    water_class="clear_ocean",
                    use_stumpf_depth=True,
                    min_training_points_for_sdb=10,
                    spatial_split=args.validate_spatial,
                    cw_min=getattr(args, "cw_min", None),
                    land_max=getattr(args, "land_max", None),
                    linf_enabled=args.linf_enabled,
                    linf_estimate=args.linf_estimate,
                    linf_deepwater_nir_max=args.linf_deepwater_nir_max,
                    linf_deepwater_bright_max=args.linf_deepwater_bright_max,
                    linf_percentile=args.linf_percentile,
                    raster_paths=comp_paths,
                    rmse_target_sdb=args.rmse_target_sdb,
                    depth_bin_m=args.depth_bin_m,
                    min_samples_per_bin=args.min_samples_per_bin,
                    depth_binning=args.depth_binning,
                    max_depth_bins=args.max_depth_bins,
                )
                rmse_c, nval_c = _compute_test_rmse_from_df_te(df_te_c)
                coh_c = radiometric_coherence_score_from_band(Path(comp_paths["B03"]), seed=42)

                df_best = sample_s2_bands_at_points(df0, best_paths, land)
                rf_b, lr_b, _, df_te_b, meta_b = train_sdb_model(
                    train_df=df_best,
                    max_depth_sdb=args.max_depth,
                    seed=42,
                    plots_dir=(plot_dir / "best_date"),
                    water_class="clear_ocean",
                    use_stumpf_depth=True,
                    min_training_points_for_sdb=10,
                    spatial_split=args.validate_spatial,
                    cw_min=getattr(args, "cw_min", None),
                    land_max=getattr(args, "land_max", None),
                    linf_enabled=args.linf_enabled,
                    linf_estimate=args.linf_estimate,
                    linf_deepwater_nir_max=args.linf_deepwater_nir_max,
                    linf_deepwater_bright_max=args.linf_deepwater_bright_max,
                    linf_percentile=args.linf_percentile,
                    raster_paths=best_paths,
                    rmse_target_sdb=args.rmse_target_sdb,
                    depth_bin_m=args.depth_bin_m,
                    min_samples_per_bin=args.min_samples_per_bin,
                    depth_binning=args.depth_binning,
                    max_depth_bins=args.max_depth_bins,
                )
                rmse_b, nval_b = _compute_test_rmse_from_df_te(df_te_b)
                coh_b = radiometric_coherence_score_from_band(Path(best_paths["B03"]), seed=42)

                best_pack = {
                    "name": "best_date",
                    "rmse_m": rmse_b,
                    "n_val": nval_b,
                    "coherence": coh_b,
                }
                comp_pack = {
                    "name": "composite",
                    "rmse_m": rmse_c,
                    "n_val": nval_c,
                    "coherence": coh_c,
                }

                pick, reason = choose_s2_variant(best_pack, comp_pack, rmse_margin=args.s2_ab_rmse_margin)
                chosen_reason = reason

                if pick["name"] == "best_date":
                    chosen = {"rf": rf_b, "lr": lr_b, "meta": meta_b, "variant": "best_date"}
                    s2_paths = best_paths
                else:
                    chosen = {"rf": rf_c, "lr": lr_c, "meta": meta_c, "variant": "composite"}
                    s2_paths = comp_paths

                ab_summary = {
                    "chosen": chosen["variant"],
                    "reason": chosen_reason,
                    "best_date": best_pack,
                    "composite": comp_pack,
                }
                log.info(f"[S2-AB] Chosen={ab_summary['chosen']} ({ab_summary['reason']}); "
                         f"RMSE best={best_pack.get('rmse_m')} comp={comp_pack.get('rmse_m')}; "
                         f"coh best={best_pack.get('coherence'):.3f} comp={comp_pack.get('coherence'):.3f}")

    if chosen is None:
        df = df0.copy()
        rf, lr, _, df_te, meta = train_sdb_model(
            train_df=df,
            max_depth_sdb=args.max_depth,
            seed=42,
            plots_dir=plot_dir,
            water_class="clear_ocean",
            use_stumpf_depth=True,
            min_training_points_for_sdb=10,
            spatial_split=args.validate_spatial,
            cw_min=getattr(args, "cw_min", None),
            land_max=getattr(args, "land_max", None),
            linf_enabled=args.linf_enabled,
            linf_estimate=args.linf_estimate,
            linf_deepwater_nir_max=args.linf_deepwater_nir_max,
            linf_deepwater_bright_max=args.linf_deepwater_bright_max,
            linf_percentile=args.linf_percentile,
            raster_paths=s2_paths,
            rmse_target_sdb=args.rmse_target_sdb,
            depth_bin_m=args.depth_bin_m,
            min_samples_per_bin=args.min_samples_per_bin,
                    depth_binning=args.depth_binning,
                    max_depth_bins=args.max_depth_bins,
        )
        chosen = {"rf": rf, "lr": lr, "meta": meta, "variant": "no_s2"}

    if ab_summary is not None:
        chosen["meta"]["s2_ab_test"] = ab_summary
        with open(out_dir / "s2_ab_test.json", "w") as f:
            json.dump(ab_summary, f, indent=2)

    joblib.dump(chosen["rf"], out_dir / "rf_model.pkl")
    if chosen["lr"]:
        joblib.dump(chosen["lr"], out_dir / "stumpf_lr.pkl")

    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(chosen["meta"], f, indent=2)

    log.info(f"Model saved to {out_dir} (variant={chosen.get('variant')})")


if __name__ == "__main__":
    main()