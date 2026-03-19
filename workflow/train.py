#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train.py – SDB model training: feature engineering, RF training, spatial CV, and model bank."""

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
try:
    from pyproj.exceptions import ProjError
except (ImportError, AttributeError):
    class ProjError(Exception):
        pass

from plot_utils import lazy_pyplot
plt = None  # lazy-loaded when plots are enabled

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

# --- Physics-based SDB (Kim et al. 2024) ---
PHYSICS_AVAILABLE = False
_physics_import_error = None
try:
    import physics_integration
    PHYSICS_AVAILABLE = True
except ImportError as e:
    _physics_import_error = f"ImportError: {e}"
except (AttributeError, OSError, SyntaxError) as e:
    # Catch common non-ImportError module load failures without masking arbitrary runtime errors.
    _physics_import_error = f"{type(e).__name__}: {e}"

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

    except (ValueError, TypeError) as e:
        log.warning("Stratification failed (%s). Falling back to random split.", e)
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

    # stumpf_idx = log(B02) / log(B03); guard against log(B03) near zero
    log_b03_safe = _np.where(
        _np.abs(out["log_B03"]) > NUMERICAL_EPS,
        out["log_B03"],
        _np.sign(out["log_B03"]) * NUMERICAL_EPS  # Preserve sign
    )
    out["stumpf_idx"] = out["log_B02"] / log_b03_safe
    
    # Clip extreme values that indicate numerical issues
    out["stumpf_idx"] = _np.clip(out["stumpf_idx"], -10.0, 10.0)

    return out


_WATER_CLASS_BRIGHTNESS_PROFILES = {
    "clear_ocean": {"max_blue": 0.25, "max_brightness": 0.18},
    "ocean": {"max_blue": 0.25, "max_brightness": 0.18},
    "clear": {"max_blue": 0.25, "max_brightness": 0.18},
    "mixed": {"max_blue": 0.28, "max_brightness": 0.20},
    "coastal": {"max_blue": 0.28, "max_brightness": 0.20},
    "turbid": {"max_blue": 0.32, "max_brightness": 0.24},
    "inland": {"max_blue": 0.32, "max_brightness": 0.24},
}


def _normalize_water_class_label(wc: Any) -> str:
    label = str(wc or "clear_ocean").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "mixed_water": "mixed",
        "mixed_waters": "mixed",
        "coastal_mixed": "mixed",
        "clearwater": "clear_ocean",
        "clear_water": "clear_ocean",
        "open_ocean": "clear_ocean",
        "turbid_inland": "turbid",
        "turbid_coastal": "turbid",
    }
    return aliases.get(label, label)



def _resolve_brightness_filter_profile(wc: Any) -> Dict[str, float]:
    label = _normalize_water_class_label(wc)
    return dict(_WATER_CLASS_BRIGHTNESS_PROFILES.get(label, _WATER_CLASS_BRIGHTNESS_PROFILES["clear_ocean"]))



def apply_s2_brightness_depth_filter(df, *args, **kwargs):
    import numpy as _np
    if df is None or df.empty:
        return df

    water_class = kwargs.get("water_class", kwargs.get("wc", "clear_ocean"))
    profile = _resolve_brightness_filter_profile(water_class)
    max_blue = kwargs.get("max_blue", profile.get("max_blue"))
    max_brightness = kwargs.get("max_brightness", profile.get("max_brightness"))
    allow_bright_shallow_pixels = bool(kwargs.get("allow_bright_shallow_pixels", False))
    bright_shallow_nir_max = float(kwargs.get("bright_shallow_nir_max", 0.03))

    m = _np.ones(len(df), dtype=bool)

    b08 = None
    if "B08" in df.columns:
        b08 = pd.to_numeric(df["B08"], errors="coerce").to_numpy(_np.float32)

    if max_blue is not None and "B02" in df.columns:
        blue = pd.to_numeric(df["B02"], errors="coerce").to_numpy(_np.float32)
        keep_blue = _np.isfinite(blue) & (blue <= float(max_blue))
        if allow_bright_shallow_pixels and b08 is not None:
            keep_blue |= (_np.isfinite(b08) & (b08 <= bright_shallow_nir_max))
        m &= keep_blue

    if max_brightness is not None and all(c in df.columns for c in ["B02", "B03", "B04"]):
        b02 = pd.to_numeric(df["B02"], errors="coerce").to_numpy(_np.float32)
        b03 = pd.to_numeric(df["B03"], errors="coerce").to_numpy(_np.float32)
        b04 = pd.to_numeric(df["B04"], errors="coerce").to_numpy(_np.float32)
        rgb_mean = (b02 + b03 + b04) / 3.0
        keep_brightness = _np.isfinite(rgb_mean) & (rgb_mean <= float(max_brightness))
        if allow_bright_shallow_pixels and b08 is not None:
            keep_brightness |= (_np.isfinite(b08) & (b08 <= bright_shallow_nir_max))
        m &= keep_brightness

    dropped = int(len(df) - int(_np.count_nonzero(m)))
    if dropped > 0:
        log.info(
            "[QC] S2 brightness/depth filter (%s) dropped %s / %s points (max_blue=%s, max_brightness=%s, bright_shallow_escape=%s).",
            _normalize_water_class_label(water_class),
            dropped,
            len(df),
            max_blue,
            max_brightness,
            allow_bright_shallow_pixels,
        )

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

    # Treat extra_xyz/survey-style soundings as authoritative bathymetric
    # anchors. They may live in turbid or optically weak water where the
    # Stumpf relationship is expected to fail; dropping them because they do
    # not fit a simple optical residual model creates exactly the shallow-bias
    # failure mode we want to avoid.
    protected_mask = np.zeros(len(df), dtype=bool)
    source_col = "source_norm" if "source_norm" in df.columns else ("source" if "source" in df.columns else None)
    if source_col is not None:
        try:
            src = df[source_col].astype(str).str.lower()
            protected_mask = (
                src.str.startswith("extra_xyz") |
                src.str.contains("hydronos|ehydro|survey|sonar|lidar|bag|sound|authoritative_base", regex=True)
            ).to_numpy(dtype=bool)
        except Exception:
            protected_mask = np.zeros(len(df), dtype=bool)

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
        except ValueError:
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
        except ValueError:
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
            if protected_mask.any():
                keepable = ~protected_mask[df_idx]
                protected_n = int(np.count_nonzero(~keepable))
                if protected_n > 0:
                    log.info(
                        "[QC] Preserving %d authoritative extra_xyz/survey points that exceeded the Stumpf residual threshold.",
                        protected_n,
                    )
                df_idx = df_idx[keepable]
            if df_idx.size == 0:
                continue
            keep[df_idx] = False
            dropped_total += int(df_idx.size)
            dropped_by_cluster.append((ci, int(df_idx.size), thr, sigma, int(np.count_nonzero(sel))))

    if dropped_total > 0:
        try:
            d_drop = depth_pd[~keep & m_valid]
            d_keep = depth_pd[keep & m_valid]
            log.info(
                "[QC] Stumpf residual filter dropped %d / %d (%.1f%%). Depth keep p50/p95=%.2f/%.2f m; dropped p50/p95=%.2f/%.2f m.",
                dropped_total, len(df), 100.0*dropped_total/max(len(df),1), np.nanpercentile(d_keep,50), np.nanpercentile(d_keep,95), np.nanpercentile(d_drop,50), np.nanpercentile(d_drop,95),
            )
        except (ValueError, IndexError, FloatingPointError):
            log.info("Stumpf residual filter dropped %s outliers.", dropped_total)
        return df.loc[keep].reset_index(drop=True)

    return df


_LOCAL_APPLY_S2_BRIGHTNESS_DEPTH_FILTER = apply_s2_brightness_depth_filter
_LOCAL_APPLY_STUMPF_RESIDUAL_FILTER = apply_stumpf_residual_filter


def _resolve_s2_optics_module():
    import sys
    import importlib.util
    from pathlib import Path as _Path

    local_path = (_Path(__file__).resolve().parent / "s2_optics.py").resolve()
    if not local_path.exists():
        log.error("Required local s2_optics.py not found at %s", local_path)
        return None

    existing = sys.modules.get("s2_optics")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file and _Path(existing_file).resolve() == local_path:
            return existing

    spec = importlib.util.spec_from_file_location("s2_optics", str(local_path))
    if spec is None or spec.loader is None:
        log.error("Could not create import spec for local s2_optics module at %s", local_path)
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError, SyntaxError, AttributeError, ValueError):
        log.exception("Failed to import local s2_optics module from %s", local_path)
        return None

    sys.modules["s2_optics"] = module
    return module


def _bind_s2_optics_functions():
    global S2_OPTICS_AVAILABLE, apply_stumpf_residual_filter, apply_s2_brightness_depth_filter

    mod = _resolve_s2_optics_module()
    if mod is None:
        S2_OPTICS_AVAILABLE = False
        return False

    missing = []
    brightness_fn = getattr(mod, "apply_s2_brightness_depth_filter", None)
    if brightness_fn is None:
        brightness_fn = _LOCAL_APPLY_S2_BRIGHTNESS_DEPTH_FILTER
        missing.append("apply_s2_brightness_depth_filter")

    stumpf_fn = getattr(mod, "apply_stumpf_residual_filter", None)
    if stumpf_fn is None:
        stumpf_fn = _LOCAL_APPLY_STUMPF_RESIDUAL_FILTER
        missing.append("apply_stumpf_residual_filter")

    apply_s2_brightness_depth_filter = brightness_fn
    apply_stumpf_residual_filter = stumpf_fn
    S2_OPTICS_AVAILABLE = True

    if missing:
        log.warning(
            "Local s2_optics module imported from %s but is missing %s; using train.py fallback implementations for those helpers.",
            getattr(mod, "__file__", "unknown"),
            ", ".join(missing),
        )
    else:
        log.info("Using local s2_optics module: %s", getattr(mod, "__file__", "unknown"))
    return True


def _infer_local_metric_epsg(lon: float, lat: float) -> int:
    zone = int((float(lon) + 180.0) / 6.0) + 1
    zone = max(1, min(zone, 60))
    return (32600 if float(lat) >= 0.0 else 32700) + zone


def _validate_physics_guidance_settings(guidance: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(guidance or {})
    float_fields = [
        "support_score", "depth_span_m", "correction_alpha", "residual_clip_m",
        "stumpf_envelope_m", "min_doa", "hard_optical_margin_frac", "max_train_point_dist_m",
    ]
    int_fields = ["n_points", "unique_tracks", "trusted_halo_px", "min_support_neighbors"]
    for key in float_fields:
        try:
            normalized[key] = float(normalized.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            normalized[key] = 0.0
    for key in int_fields:
        try:
            normalized[key] = int(normalized.get(key, 0) or 0)
        except (TypeError, ValueError):
            normalized[key] = 0
    normalized["low_support"] = bool(normalized.get("low_support", False))
    cof = normalized.get("core_optical_features", [])
    normalized["core_optical_features"] = [str(v) for v in cof] if isinstance(cof, (list, tuple)) else []
    pts = normalized.get("support_points_lonlat", [])
    clean_pts = []
    if isinstance(pts, (list, tuple)):
        for item in pts:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                lon = float(item[0]); lat = float(item[1])
            except (TypeError, ValueError):
                continue
            if np.isfinite(lon) and np.isfinite(lat):
                clean_pts.append([lon, lat])
    normalized["support_points_lonlat"] = clean_pts
    return normalized


def _resolve_depth_source_family(source_name: Optional[str]) -> str:
    src = str(source_name or "").strip().lower()
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


def _compute_physics_guidance_settings(df_tr: pd.DataFrame) -> Dict[str, Any]:
    """Derive conservative guidance settings from the retained training rows."""
    out: Dict[str, Any] = {
        "support_score": 0.0,
        "low_support": True,
        "n_points": 0,
        "unique_tracks": 0,
        "depth_span_m": 0.0,
        "correction_alpha": 0.08,
        "residual_clip_m": 0.12,
        "anchor_support_good": False,
        "anchor_fraction": 0.0,
        "baseline_source": "stumpf_depth",
        "stumpf_envelope_m": 0.18,
        "min_doa": 0.992,
        "hard_optical_margin_frac": 0.012,
        "max_train_point_dist_m": 125.0,
        "min_support_neighbors": 2,
        "trusted_halo_px": 192,
        "core_optical_features": ["stumpf_idx", "stumpf_depth", "brightness", "B02", "B03", "B04", "B08"],
        "support_points_lonlat": [],
    }
    if df_tr is None or df_tr.empty:
        return out

    depth = np.abs(pd.to_numeric(df_tr.get("depth_m"), errors="coerce").to_numpy(dtype=np.float32))
    finite_depth = depth[np.isfinite(depth)]
    out["n_points"] = int(finite_depth.size)
    if finite_depth.size:
        out["depth_span_m"] = float(np.nanpercentile(finite_depth, 95) - np.nanpercentile(finite_depth, 5))

    if "track_id" in df_tr.columns:
        out["unique_tracks"] = int(pd.Series(df_tr["track_id"]).astype(str).nunique(dropna=True))
    elif "granule" in df_tr.columns and "beam" in df_tr.columns:
        tracks = df_tr[["granule", "beam"]].astype(str).agg("|".join, axis=1)
        out["unique_tracks"] = int(tracks.nunique(dropna=True))
    elif "source" in df_tr.columns:
        out["unique_tracks"] = int(pd.Series(df_tr["source"]).astype(str).nunique(dropna=True))

    stumpf_depth = pd.to_numeric(df_tr.get("stumpf_depth"), errors="coerce").to_numpy(dtype=np.float32) if "stumpf_depth" in df_tr.columns else np.full(len(df_tr), np.nan, dtype=np.float32)
    stumpf_idx = pd.to_numeric(df_tr.get("stumpf_idx"), errors="coerce").to_numpy(dtype=np.float32) if "stumpf_idx" in df_tr.columns else np.full(len(df_tr), np.nan, dtype=np.float32)

    paired = np.isfinite(depth) & np.isfinite(stumpf_depth)
    if np.any(paired):
        residual = depth[paired] - stumpf_depth[paired]
        out["residual_clip_m"] = float(np.clip(np.nanpercentile(np.abs(residual), 90), 0.10, 0.50))
        out["stumpf_envelope_m"] = float(np.clip(np.nanpercentile(np.abs(residual), 95), 0.15, 0.60))
        out["stumpf_support_min_m"] = float(np.nanpercentile(stumpf_depth[paired], 1))
        out["stumpf_support_max_m"] = float(np.nanpercentile(stumpf_depth[paired], 99))
    else:
        finite_sd = stumpf_depth[np.isfinite(stumpf_depth)]
        if finite_sd.size:
            out["stumpf_support_min_m"] = float(np.nanpercentile(finite_sd, 1))
            out["stumpf_support_max_m"] = float(np.nanpercentile(finite_sd, 99))

    finite_si = stumpf_idx[np.isfinite(stumpf_idx)]
    if finite_si.size:
        out["stumpf_idx_support_min"] = float(np.nanpercentile(finite_si, 1))
        out["stumpf_idx_support_max"] = float(np.nanpercentile(finite_si, 99))

    source_series = None
    if "source_norm" in df_tr.columns:
        source_series = df_tr["source_norm"].astype(str).str.lower()
    elif "source" in df_tr.columns:
        source_series = df_tr["source"].astype(str).str.lower()

    anchor_support_good = False
    if source_series is not None and len(source_series):
        anchor_mask = source_series.str.startswith("extra_xyz") | source_series.str.contains("hydronos|ehydro|survey|sonar|lidar|bag|sound|authoritative_base", regex=True)
        out["anchor_fraction"] = float(anchor_mask.mean())
        anchor_support_good = bool(out["anchor_fraction"] >= 0.70 and out["n_points"] >= 1200 and out["depth_span_m"] >= 4.0)
        out["anchor_support_good"] = anchor_support_good
        if not anchor_support_good:
            log.info("[ANCHOR] anchor_support_good=False: fraction=%.3f (need>=0.70) n=%d (need>=1200) span=%.1f (need>=4.0) sources=%s",
                     out["anchor_fraction"], out["n_points"], out["depth_span_m"],
                     dict(source_series.value_counts().head(5)))
    else:
        log.info("[ANCHOR] No source_series available (source_norm=%s source=%s cols=%s)",
                 "source_norm" in df_tr.columns, "source" in df_tr.columns,
                 list(df_tr.columns)[:10])

    # Depth-based fallback: if we have very dense, deep data but source
    # labels are missing or mangled, infer anchor_support_good from data
    # characteristics alone.  Dense deep data (>5000 pts, >15m span) is
    # extremely unlikely to come from ATL alone.
    if not anchor_support_good and out["n_points"] >= 5000 and out["depth_span_m"] >= 15.0:
        anchor_support_good = True
        out["anchor_support_good"] = True
        out["anchor_support_good_source"] = "depth_fallback"
        log.info("[ANCHOR] anchor_support_good=True via depth fallback: n=%d span=%.1fm "
                 "(dense deep data unlikely from ATL alone)", out["n_points"], out["depth_span_m"])

    n_score = min(1.0, out["n_points"] / 600.0)
    track_score = min(1.0, out["unique_tracks"] / 6.0)
    depth_score = min(1.0, out["depth_span_m"] / 3.0)
    support_score = 0.45 * n_score + 0.30 * track_score + 0.25 * depth_score
    out["support_score"] = float(support_score)
    low_support = bool((out["n_points"] < 400) or (out["unique_tracks"] < 4) or (out["depth_span_m"] < 1.5) or (support_score < 0.70))
    if anchor_support_good:
        low_support = False
    out["low_support"] = low_support
    if anchor_support_good:
        # Dense authoritative XYZ data (hydronos, ehydro, etc.) is ground truth.
        # The RF should be allowed to fully correct the Stumpf baseline — not
        # limited to ±0.75m corrections.  The Stumpf ratio is still a valuable
        # input feature, but it should not constrain the final prediction when
        # high-quality measured depths are available for training.
        out["correction_alpha"] = 1.0   # Full RF correction (was 0.18)
        out["residual_clip_m"] = 15.0   # Allow corrections up to 15m (was 0.75)
        out["stumpf_envelope_m"] = 20.0 # Wide envelope (was 1.0)
        out["min_doa"] = 0.965 if out["n_points"] >= 5000 else 0.975
        out["hard_optical_margin_frac"] = 0.005
        out["max_train_point_dist_m"] = 2000.0 if out["n_points"] >= 5000 else 1000.0
        out["min_support_neighbors"] = 1
        out["trusted_halo_px"] = 64
    else:
        out["correction_alpha"] = 0.05 if low_support else 0.12
        out["min_doa"] = 0.995 if low_support else 0.985
        out["hard_optical_margin_frac"] = 0.008 if low_support else 0.015
        out["max_train_point_dist_m"] = 100.0 if low_support else 250.0
        out["min_support_neighbors"] = 3 if low_support else 2
        out["trusted_halo_px"] = 256 if low_support else 160

    lon_col = None
    lat_col = None
    if "longitude" in df_tr.columns and "latitude" in df_tr.columns:
        lon_col, lat_col = "longitude", "latitude"
    elif "lon" in df_tr.columns and "lat" in df_tr.columns:
        lon_col, lat_col = "lon", "lat"

    if lon_col is not None and lat_col is not None:
        ll = df_tr[[lon_col, lat_col]].copy().rename(columns={lon_col: "longitude", lat_col: "latitude"})
        ll["longitude"] = pd.to_numeric(ll["longitude"], errors="coerce")
        ll["latitude"] = pd.to_numeric(ll["latitude"], errors="coerce")
        ll = ll[np.isfinite(ll["longitude"]) & np.isfinite(ll["latitude"])]
        if len(ll):
            max_keep = 50000 if anchor_support_good else 10000
            ll = _spatially_thin_lonlat_dataframe(ll, max_keep=max_keep, anchor_support_good=anchor_support_good)
            out["support_points_lonlat"] = ll[["longitude", "latitude"]].to_numpy(dtype=float).tolist()

    return _validate_physics_guidance_settings(out)




def _spatially_thin_lonlat_dataframe(ll: pd.DataFrame, max_keep: int, *, anchor_support_good: bool = False) -> pd.DataFrame:
    """Deterministically thin support points in metric space for tile-stable support gating.

    For dense authoritative-support runs, keep a much richer support cloud so the
    downstream CUDEM guidance gate reflects real survey support rather than a
    sparse, over-thinned proxy.
    """
    if anchor_support_good:
        try:
            max_keep = max(int(max_keep), 50000)
        except Exception:
            max_keep = 50000
    if ll is None or ll.empty or len(ll) <= max_keep:
        return ll
    work = ll[["longitude", "latitude"]].copy()
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work = work[np.isfinite(work["longitude"]) & np.isfinite(work["latitude"])]
    work = work.drop_duplicates().sort_values(["longitude", "latitude"]).reset_index(drop=True)
    if len(work) <= max_keep:
        return work

    lon0 = float(work["longitude"].median())
    lat0 = float(work["latitude"].median())
    epsg = _infer_local_metric_epsg(lon0, lat0)
    try:
        tfm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        xs, ys = tfm.transform(work["longitude"].to_numpy(dtype=float), work["latitude"].to_numpy(dtype=float))
    except ProjError:
        log.warning("Metric thinning projection failed for EPSG:%s; falling back to geographic thinning.", epsg, exc_info=True)
        xs = work["longitude"].to_numpy(dtype=float)
        ys = work["latitude"].to_numpy(dtype=float)

    metric = work.copy()
    metric["_x"] = xs
    metric["_y"] = ys
    metric = metric[np.isfinite(metric["_x"]) & np.isfinite(metric["_y"])].reset_index(drop=True)
    if len(metric) <= max_keep:
        return metric[["longitude", "latitude"]]

    x0 = float(metric["_x"].min())
    y0 = float(metric["_y"].min())
    x_span = max(float(metric["_x"].max() - x0), 1e-6)
    y_span = max(float(metric["_y"].max() - y0), 1e-6)
    area = x_span * y_span
    cell = max((area / float(max_keep)) ** 0.5, 1.0)

    thinned = metric
    for _ in range(8):
        gx = np.floor((metric["_x"].to_numpy(dtype=float) - x0) / cell).astype(np.int64)
        gy = np.floor((metric["_y"].to_numpy(dtype=float) - y0) / cell).astype(np.int64)
        thinned = metric.assign(_gx=gx, _gy=gy).drop_duplicates(subset=["_gx", "_gy"], keep="first")
        if len(thinned) <= max_keep:
            break
        cell *= 1.25

    thinned = thinned.sort_values(["longitude", "latitude"]).reset_index(drop=True)
    if len(thinned) > max_keep:
        thinned = thinned.iloc[:max_keep].reset_index(drop=True)
    return thinned[["longitude", "latitude"]]

def _tighten_guidance_for_unrepresentative_spatial_holdout(guidance_settings: Dict[str, Any], spatial_status: Dict[str, Any]) -> Dict[str, Any]:
    out = _validate_physics_guidance_settings(guidance_settings)
    if not isinstance(spatial_status, dict):
        return out
    ok = bool(spatial_status.get("ok", False))
    reason = str(spatial_status.get("reason", "")).lower()
    if (not ok) or ("unrepresentative" in reason) or ("no_representative" in reason):
        # Dense authoritative-anchor runs can still justify broader prediction
        # support even when a representative spatial holdout cannot be formed.
        if bool(out.get("anchor_support_good", False)) and int(out.get("n_points", 0)) >= 5000:
            out["low_support"] = False
            out["correction_alpha"] = 1.0
            out["residual_clip_m"] = 15.0
            out["stumpf_envelope_m"] = 20.0
            out["min_doa"] = min(float(out.get("min_doa", 0.975)), 0.975)
            out["max_train_point_dist_m"] = max(float(out.get("max_train_point_dist_m", 350.0)), 2000.0)
            out["min_support_neighbors"] = 1
            out["hard_optical_margin_frac"] = min(float(out.get("hard_optical_margin_frac", 0.008)), 0.008)
            out["trusted_halo_px"] = min(int(out.get("trusted_halo_px", 160)), 128)
        else:
            out["low_support"] = True
            out["correction_alpha"] = min(float(out.get("correction_alpha", 0.08)), 0.05)
            out["residual_clip_m"] = min(float(out.get("residual_clip_m", 0.12)), 0.12)
            out["stumpf_envelope_m"] = min(float(out.get("stumpf_envelope_m", 0.18)), 0.18)
            out["min_doa"] = max(float(out.get("min_doa", 0.992)), 0.997)
            out["max_train_point_dist_m"] = min(float(out.get("max_train_point_dist_m", 125.0)), 100.0)
            out["min_support_neighbors"] = max(int(out.get("min_support_neighbors", 2)), 3)
            out["hard_optical_margin_frac"] = min(float(out.get("hard_optical_margin_frac", 0.012)), 0.008)
            out["trusted_halo_px"] = max(int(out.get("trusted_halo_px", 192)), 256)
    return _validate_physics_guidance_settings(out)

class PhysicsGuidedResidualModel:
    """Wrapper that preserves residual-mode internals while exposing magnitude predictions.

    The wrapper is persisted with joblib and later loaded by predict.py. During
    unpickling, Python may probe for ``__setstate__`` before instance state has
    been restored, so attribute delegation must remain safe even when
    ``base_model`` is not yet present on ``self.__dict__``.
    """

    def __init__(self, base_model: RandomForestRegressor, feature_columns: List[str], guidance_settings: Dict[str, Any]) -> None:
        self.base_model = base_model
        self.feature_columns = list(feature_columns)
        self.guidance_settings = _validate_physics_guidance_settings(guidance_settings)
        self._stumpf_depth_idx = self.feature_columns.index("stumpf_depth") if "stumpf_depth" in self.feature_columns else None

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "feature_columns": list(self.feature_columns),
            "guidance_settings": dict(self.guidance_settings),
            "_stumpf_depth_idx": self._stumpf_depth_idx,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        feature_columns = list(state.get("feature_columns", []) or [])
        guidance_settings = _validate_physics_guidance_settings(state.get("guidance_settings", {}))
        self.base_model = state.get("base_model")
        self.feature_columns = feature_columns
        self.guidance_settings = guidance_settings
        stumpf_idx = state.get("_stumpf_depth_idx")
        if stumpf_idx is None:
            stumpf_idx = feature_columns.index("stumpf_depth") if "stumpf_depth" in feature_columns else None
        self._stumpf_depth_idx = stumpf_idx

    @property
    def estimators_(self):
        return self.base_model.estimators_

    @property
    def feature_importances_(self):
        return self.base_model.feature_importances_

    @property
    def n_features_in_(self):
        return self.base_model.n_features_in_

    def predict_residual(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.base_model.predict(X), dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        # When dense authoritative XYZ data trained the model, bypass the
        # Stumpf residual architecture entirely.  The Stumpf blue/green ratio
        # is fundamentally wrong in turbid channels (low ratio = interpreted
        # as shallow, but channels are actually deep).  The RF learned the
        # correct depth-to-spectral relationship from ground truth — let it
        # predict directly without the wrong baseline constraining it.
        if self.guidance_settings.get("anchor_support_good", False):
            return np.maximum(self.predict_residual(X), 0.0).astype(np.float32)
        if self._stumpf_depth_idx is None or X.ndim != 2 or self._stumpf_depth_idx >= X.shape[1]:
            return np.maximum(self.predict_residual(X), 0.0).astype(np.float32)
        stumpf_base = np.maximum(X[:, self._stumpf_depth_idx].astype(np.float32), 0.0)
        residual = self.predict_residual(X)
        correction_alpha = float(np.clip(self.guidance_settings.get("correction_alpha", 0.35), 0.0, 1.0))
        residual_clip_m = float(max(self.guidance_settings.get("residual_clip_m", 0.5), 0.05))
        stumpf_envelope_m = float(max(self.guidance_settings.get("stumpf_envelope_m", residual_clip_m), residual_clip_m, 0.10))
        residual = np.clip(residual, -residual_clip_m, residual_clip_m)
        guided = stumpf_base + (correction_alpha * residual)
        guided = np.clip(guided, np.maximum(stumpf_base - stumpf_envelope_m, 0.0), stumpf_base + stumpf_envelope_m)
        return np.maximum(guided, 0.0).astype(np.float32)

    def __getattr__(self, name: str):
        if name == "base_model":
            raise AttributeError(name)
        base_model = object.__getattribute__(self, "__dict__").get("base_model")
        if base_model is None:
            raise AttributeError(name)
        return getattr(base_model, name)


def _predict_physics_guided_magnitude(
    rf: RandomForestRegressor,
    df_eval: pd.DataFrame,
    feat_cols: List[str],
    guidance_settings: Dict[str, Any],
) -> np.ndarray:
    if df_eval is None or len(df_eval) == 0:
        return np.zeros(0, dtype=np.float32)
    missing = [c for c in feat_cols if c not in df_eval.columns]
    if missing:
        raise ValueError(f"Missing feature columns for physics-guided prediction: {missing}")
    X_eval = df_eval[feat_cols].to_numpy()

    # When anchor_support_good, the RF was trained on absolute depth — predict
    # directly without the Stumpf residual architecture.
    if guidance_settings.get("anchor_support_good", False):
        if hasattr(rf, "predict"):
            # For PhysicsGuidedResidualModel, predict() already handles anchor bypass
            pred = np.asarray(rf.predict(X_eval), dtype=np.float32)
        else:
            pred = np.asarray(rf.predict(X_eval), dtype=np.float32)
        return np.maximum(pred, 0.0).astype(np.float32)

    if "stumpf_depth" not in df_eval.columns:
        raise ValueError("Missing stumpf_depth feature for physics-guided prediction")
    if hasattr(rf, "predict_residual"):
        residual = np.asarray(rf.predict_residual(X_eval), dtype=np.float32)
    else:
        residual = np.asarray(rf.predict(X_eval), dtype=np.float32)
    stumpf_base = np.maximum(pd.to_numeric(df_eval["stumpf_depth"], errors="coerce").to_numpy(dtype=np.float32), 0.0)
    correction_alpha = float(np.clip(guidance_settings.get("correction_alpha", 0.35), 0.0, 1.0))
    residual_clip_m = float(max(guidance_settings.get("residual_clip_m", 0.5), 0.05))
    stumpf_envelope_m = float(max(guidance_settings.get("stumpf_envelope_m", residual_clip_m), residual_clip_m, 0.10))
    residual = np.clip(residual, -residual_clip_m, residual_clip_m)
    guided = stumpf_base + (correction_alpha * residual)
    guided = np.clip(guided, np.maximum(stumpf_base - stumpf_envelope_m, 0.0), stumpf_base + stumpf_envelope_m)
    return np.maximum(guided, 0.0).astype(np.float32)


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
    """Recursively convert metadata payloads to plain Python scalars/containers.

    Historical callers use this helper for numeric metadata, but newer guidance
    metadata also includes strings, booleans, None, lists, and numpy scalars.
    Preserve non-numeric leaves instead of forcing everything through float().
    """
    if isinstance(val, dict):
        return {k: _to_python_float(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_python_float(v) for v in val]
    if isinstance(val, np.ndarray):
        return [_to_python_float(v) for v in val.tolist()]
    if isinstance(val, (str, bool)) or val is None:
        return val
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if hasattr(val, "item"):
        item = val.item()
        return _to_python_float(item)
    try:
        return float(val)
    except (TypeError, ValueError):
        return val

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
        except (RuntimeError, ValueError, OSError):
            log.debug("ignored", exc_info=True)
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
        except (AttributeError, TypeError, ValueError):
            log.debug("ignored", exc_info=True)

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

    log.info("Depth-of-support analysis (target RMSE <= %.2f m)", rmse_target_m)
    log.info("%-15s | %-10s | %-10s | %-10s | %-8s | %s", 'Bin Range (m)', 'RMSE (m)', 'Bias (m)', 'MAE (m)', 'Count', 'Status')

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

        log.info("%5.1f - %5.1f   | %10.3f | %10.3f | %10.3f | %8d | %s", lo, hi, r_val, b_val, m_val, n, status)


    if first_fail_msg:
        log.warning("%s", first_fail_msg)

    max_supported = float(max(supported_upper_edges)) if supported_upper_edges else None
    if max_supported is not None and max_depth_cap_m is not None:
        try:
            max_supported = min(max_supported, float(max_depth_cap_m))
        except (TypeError, ValueError, KeyError):
            log.debug("ignored", exc_info=True)

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
        log.info("Quantile depth bin edges (m): %s", edges_list)


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
        log.warning("%s: no finite points; skipping %s", title, out_png)
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
        log.info("Saved: %s", path_out)

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
        except Exception as _exc:
            log.debug("Suppressed: %s", _exc, exc_info=True)


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
    log.info("Feature importance saved: %s", out_png)

def _count(df: pd.DataFrame, label: str):
    log.info("%s: n=%s", label, len(df))

def _nonfinite_report(df: pd.DataFrame, cols: List[str], label: str, max_lines: int = 30):
    if df.empty:
        log.warning("%s: df is empty; cannot compute non-finite fractions.", label)
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
        log.info("%s: all requested cols are finite.", label)
        return
    lines.sort(reverse=True)
    log.warning("%s: non-finite values detected (showing up to %s).", label, max_lines)
    for frac, c, bad in lines[:max_lines]:
        log.warning("  - %s: %s/%s non-finite (%.1f%%)", c, bad, n, frac*100)


def _funnel_df_stats(df: pd.DataFrame, stage: str, *, rr: Optional[Any] = None, depth_col: str = "depth_m",
                     hist_max_m: float = 40.0, hist_bin_m: float = 1.0) -> None:
    try:
        n = int(len(df)) if df is not None else 0
    except TypeError:
        n = 0
    if df is None or n == 0:
        log.info("%s: n=0", stage)
        if rr is not None:
            try:
                rr.add(f"funnel.train.{stage}.n", 0)
            except (AttributeError, TypeError, ValueError):
                log.debug("ignored", exc_info=True)
        return

    d = None
    if depth_col in df.columns:
        try:
            d = pd.to_numeric(df[depth_col], errors="coerce").to_numpy(dtype="float64")
        except (TypeError, ValueError):
            d = None

    if d is None:
        log.info("%s: n=%s (no %s column)", stage, n, depth_col)
        if rr is not None:
            try:
                rr.add(f"funnel.train.{stage}.n", n)
            except Exception as _exc:
                log.debug("Suppressed: %s", _exc, exc_info=True)
        return

    mfin = np.isfinite(d)
    nfin = int(np.count_nonzero(mfin))
    if nfin:
        p50 = float(np.nanpercentile(d[mfin], 50))
        p95 = float(np.nanpercentile(d[mfin], 95))
        dmax = float(np.nanmax(d[mfin]))
    else:
        p50 = p95 = dmax = float("nan")

    log.info("%s: n=%s finite=%s depth_p50/p95/max=%.3f/%.3f/%.3f m", stage, n, nfin, p50, p95, dmax)

    hist = None
    edges = None
    try:
        if nfin:
            dabs = np.abs(d[mfin])
            edges = np.arange(0.0, float(hist_max_m) + float(hist_bin_m), float(hist_bin_m))
            hist, _ = np.histogram(dabs, bins=edges)
    except (ValueError, FloatingPointError):
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
        except Exception as _exc:
            log.debug("Suppressed: %s", _exc, exc_info=True)

def sample_s2_bands_at_points(train_df: pd.DataFrame,
                             s2_paths: Dict[str, str],
                             land_mask_path: Optional[str]) -> pd.DataFrame:
    """Samples S2 rasters at training point locations."""
    if train_df.empty:
        return train_df

    log.info("Sampling S2 features at %s locations...", len(train_df))
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
            log.warning("Missing raster: %s. Filling %s with NaNs.", path, col)
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
            log.warning("Failed to sample %s: %s", col, exc)
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



def _try_reuse_model_bank(
    model_bank_dir, metadata, reason="periodic_retrain_skip",
    extra_meta=None,
    required_context=None,
):
    """Attempt to load and return a previously-trained model from the model bank.

    Returns (rf, lr, df_train, df_test, meta) if a model was found,
    or None if no model is available.

    Args:
        model_bank_dir: Path to the model bank directory.
        metadata: Metadata dict to update with reuse info.
        reason: Why we are reusing (for logging/metadata).
        extra_meta: Additional key-value pairs to add to model_bank metadata.
    """
    bank_dir_p = Path(model_bank_dir)
    rf_p = bank_dir_p / "rf_model.pkl"
    if not rf_p.exists():
        return None

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
            with open(mm_p, "r", encoding="utf-8") as _f:
                mm = json.load(_f)
        except Exception:
            mm = None

    if required_context:
        mm_ctx = mm.get("model_bank_partition") if isinstance(mm, dict) else None
        if not isinstance(mm_ctx, dict):
            log.warning(
                "[MODEL_BANK] Refusing reuse from %s: saved model metadata has no model_bank_partition context.",
                bank_dir_p,
            )
            return None
        mismatch = []
        for _k, _v in dict(required_context).items():
            if mm_ctx.get(_k) != _v:
                mismatch.append((_k, mm_ctx.get(_k), _v))
        if mismatch:
            msg = ", ".join(f"{k}: saved={sv!r} current={cv!r}" for k, sv, cv in mismatch[:6])
            log.warning(
                "[MODEL_BANK] Refusing reuse from %s due to partition/context mismatch (%s).",
                bank_dir_p,
                msg,
            )
            return None

    bank_meta = {"enabled": True, "reused_model": True, "reuse_reason": reason}
    if extra_meta:
        bank_meta.update(extra_meta)
    metadata.setdefault("model_bank", {}).update(bank_meta)

    if isinstance(mm, dict):
        mm.setdefault("model_bank", {}).update(bank_meta)
        return rf_reuse, lr_reuse, pd.DataFrame(), pd.DataFrame(), mm

    return rf_reuse, lr_reuse, pd.DataFrame(), pd.DataFrame(), metadata


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
    model_bank_context: Optional[Dict[str, Any]] = None,
    fallback_registry: Optional[Any] = None,
) -> Tuple[RandomForestRegressor, Optional[LinearRegression], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    log.info("Stage 3: Feature Engineering and Model Training")
    if not S2_OPTICS_AVAILABLE:
        _bind_s2_optics_functions()
    if not S2_OPTICS_AVAILABLE:
        required_cols = {"B02", "B03", "B04", "B08", "stumpf_idx", "depth_m"}
        fallback_ok = raster_paths is None and required_cols.issubset(set(train_df.columns))
        if fallback_ok:
            log.warning(
                "Could not bind local s2_optics.py; using internal train.py optical fallback filters for this non-production context."
            )
        else:
            raise ImportError("Could not bind the real local s2_optics module for production training.")

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
            log.info("using land_probability semantics: keep LAND<= %s plus nodata_ok=%s", thr, land_mask_nodata_is_water)
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
                            log.info("auto-inferred water_val=%s from training points distribution: %s", wv, dict(zip(vals.tolist(), counts.tolist())))
                        else:
                            wv = 0.0
                            log.warning("Could not infer water_val (no finite LAND samples); falling back to 0.")
                    else:
                        wv = float(land_mask_water_val)
                    keep_eq = (finite & (land_v == wv))
                    m_land = nodata_ok | (~keep_eq if bool(land_mask_invert) else keep_eq)
                    log.info("using water_val=%s invert=%s nodata_ok=%s", wv, bool(land_mask_invert), land_mask_nodata_is_water)
            elif lt in ("water_only", "land_binary", "binary", "mask"):
                if land_mask_water_val is None:
                    vv = land_v[finite & np.isfinite(land_v)]
                    if vv.size > 0:
                        vals, counts = np.unique(vv.astype(np.int64), return_counts=True)
                        wv = float(vals[int(np.argmax(counts))])
                        log.info("auto-inferred water_val=%s from training points distribution: %s", wv, dict(zip(vals.tolist(), counts.tolist())))
                    else:
                        wv = 0.0
                        log.warning("Could not infer water_val (no finite LAND samples); falling back to 0.")
                else:
                    wv = float(land_mask_water_val)
                keep_eq = (finite & (land_v == wv))
                m_land = nodata_ok | (~keep_eq if bool(land_mask_invert) else keep_eq)
                log.info("using explicit semantics: water_val=%s invert=%s nodata_ok=%s", wv, bool(land_mask_invert), land_mask_nodata_is_water)
            else:
                m_land = nodata_ok | (finite & (land_v <= float(land_max)))
                log.info("using threshold semantics: keep LAND<= %s plus nodata_ok=%s", land_max, land_mask_nodata_is_water)

    if cw_min is None or "CLEAR_WATER" not in df.columns:
        m_cw = np.ones(len(df), dtype=bool)
    else:
        cw_v = pd.to_numeric(df["CLEAR_WATER"], errors="coerce").to_numpy(dtype="float64")
        m_cw = np.isfinite(cw_v) & (cw_v >= float(cw_min))

    m_env = m_land & m_cw

    # extra_xyz (hydronos, ehydro, etc.) bypasses land and clear-water filters:
    # these are high-quality soundings that may be in turbid/masked zones.
    if "source_norm" in df.columns:
        m_is_xyz = df["source_norm"].astype(str).str.startswith("extra_xyz")
        n_xyz_before = int(m_is_xyz.sum())
        m_env |= m_is_xyz
        if n_xyz_before > 0:
            log.info("Bypassed env filter for %s extra_xyz points (high-quality survey data)", n_xyz_before)

    df = df[m_env].reset_index(drop=True)
    _count(df, f"after env filter (LAND<= {land_max}, CLEAR_WATER>= {cw_min})")
    _funnel_df_stats(df, "after_env_filter", rr=rr)

    if df.empty:
        log.error("No samples remain after environment filter.")
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
                        log.info("Estimated L_inf constants from rasters: %s", est)
            except Exception as e:
                log.warning("Raster-based L_inf estimation failed; falling back to DF method. Reason: %s", e)

            if not est:
                est = estimate_linf_from_df(
                    df,
                    nir_max=linf_deepwater_nir_max,
                    bright_max=linf_deepwater_bright_max,
                    percentile=linf_percentile,
                    cw_col="CLEAR_WATER" if "CLEAR_WATER" in df.columns else None,
                )
                if est:
                    log.info("Estimated L_inf constants from deepwater (DF fallback): %s", est)

            if est:
                l_inf_constants = est
            else:
                log.warning("L_inf estimate failed (no suitable deepwater pixels found). Using zeros.")

    df = add_s2_optical_features(df, l_inf=l_inf_constants)
    _count(df, "after add_s2_optical_features")
    _funnel_df_stats(df, "after_s2_features", rr=rr)

    try:
        df = apply_s2_brightness_depth_filter(df, water_class=water_class)
    except TypeError:
        df = apply_s2_brightness_depth_filter(df, wc=water_class)

    _count(df, "after brightness/depth filter")
    _funnel_df_stats(df, "after_brightness_filter", rr=rr)

    # ---------------------------------------------------------------
    # Should we skip the Stumpf residual filter?
    # Uses the same 3-criterion check as the model selection:
    # if the data would trigger physics-only Stumpf, don't filter
    # out the deeper points that we'll ignore anyway.
    # ---------------------------------------------------------------
    _skip_stumpf_filter = False
    if "depth_m" in df.columns and "stumpf_idx" in df.columns and len(df) > 50:
        _d_pre = np.abs(pd.to_numeric(df["depth_m"], errors="coerce").dropna().to_numpy())
        _si_pre = pd.to_numeric(df["stumpf_idx"], errors="coerce").dropna().to_numpy()
        if len(_d_pre) > 50 and len(_si_pre) > 50:
            _iqr = float(np.percentile(_d_pre, 75) - np.percentile(_d_pre, 25))
            _full_range = float(np.max(_d_pre) - np.min(_d_pre))
            _median_d = float(np.median(_d_pre))
            _floor = float(np.min(_d_pre))
            _near_floor_frac = float(np.mean(_d_pre < (_floor + 0.5)))

            # Correlation between stumpf_idx and depth
            _both = np.isfinite(_d_pre[:len(_si_pre)]) & np.isfinite(_si_pre[:len(_d_pre)])
            _abs_corr = 0.0
            if np.sum(_both) > 10:
                _c = np.corrcoef(_si_pre[_both], _d_pre[_both])[0, 1]
                _abs_corr = abs(_c) if np.isfinite(_c) else 0.0

            skip_reasons = []
            if _abs_corr < 0.3:
                skip_reasons.append("|corr|=%.2f < 0.30" % _abs_corr)
            if _iqr < 1.0 and _full_range > 0 and (_full_range / max(_iqr, 0.01)) > 5.0:
                skip_reasons.append("IQR=%.2fm, range/IQR=%.1f" % (_iqr, _full_range / max(_iqr, 0.01)))
            if _near_floor_frac > 0.80 and _median_d < 1.5:
                skip_reasons.append("near_floor=%.0f%%, median=%.2fm" % (_near_floor_frac * 100, _median_d))

            if skip_reasons:
                _skip_stumpf_filter = True
                log.info("[QC] Skipping Stumpf residual filter (physics-only path): %s",
                         "; ".join(skip_reasons))
            elif _full_range < 2.0:
                _skip_stumpf_filter = True
                log.info("[QC] Skipping Stumpf residual filter: depth_range=%.1fm "
                         "too narrow to safely remove any points.", _full_range)

    if not _skip_stumpf_filter:
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

    stumpf_lr: Optional[Any] = None
    if use_stumpf_depth and "stumpf_idx" in df.columns:
        x = pd.to_numeric(df["stumpf_idx"], errors="coerce").to_numpy(np.float32)
        y_raw = pd.to_numeric(df["depth_m"], errors="coerce").to_numpy(np.float32)
        y = np.abs(y_raw)
        m = np.isfinite(x) & np.isfinite(y)

        if np.sum(m) >= 20:
            try:
                x_fit = x[m].astype(np.float64)
                y_fit = y[m].astype(np.float64)
                corr = np.corrcoef(x_fit, y_fit)[0, 1] if x_fit.size >= 3 else np.nan
                increasing = bool(not np.isfinite(corr) or corr >= 0.0)
                x_unique = np.unique(np.round(x_fit, 6))

                # ---------------------------------------------------------------
                # Decision: ATL-calibrated vs physics-only Stumpf model
                # ---------------------------------------------------------------
                # Three independent criteria — ANY one triggers physics-only:
                #
                # 1. CORRELATION TEST: |corr(stumpf_idx, depth)| < 0.3
                #    The optical ratio has no meaningful relationship with
                #    the ATL depths — regression would learn noise.
                #
                # 2. DEPTH DISTRIBUTION SKEW: IQR < 1m AND range/IQR > 5
                #    Data is dominated by a shallow cluster with sparse
                #    deeper outliers. Regression would be driven by the
                #    cluster, not the real depth-reflectance relationship.
                #
                # 3. NEAR-FLOOR DOMINANCE: >80% of points within 0.5m of
                #    the shallowest depth AND median depth < 1.5m.
                #    Almost all ATL returns are surface noise.
                # ---------------------------------------------------------------
                depth_range = float(np.max(y_fit) - np.min(y_fit))
                depth_std = float(np.std(y_fit))
                depth_iqr = float(np.percentile(y_fit, 75) - np.percentile(y_fit, 25))
                depth_median = float(np.median(y_fit))
                abs_corr = abs(corr) if np.isfinite(corr) else 0.0

                # Near-floor: fraction of points within 0.5m of the minimum
                depth_floor = float(np.min(y_fit))
                near_floor_frac = float(np.mean(y_fit < (depth_floor + 0.5)))

                use_physics_only = False
                physics_only_reasons = []

                # Criterion 1: weak correlation
                if abs_corr < 0.3:
                    use_physics_only = True
                    physics_only_reasons.append("|corr|=%.2f < 0.30" % abs_corr)

                # Criterion 2: skewed depth distribution
                if depth_iqr < 1.0 and depth_range > 0 and (depth_range / max(depth_iqr, 0.01)) > 5.0:
                    use_physics_only = True
                    physics_only_reasons.append(
                        "IQR=%.2fm, range/IQR=%.1f > 5" % (depth_iqr, depth_range / max(depth_iqr, 0.01)))

                # Criterion 3: near-floor dominance
                if near_floor_frac > 0.80 and depth_median < 1.5:
                    use_physics_only = True
                    physics_only_reasons.append(
                        "near_floor=%.0f%%, median=%.2fm" % (near_floor_frac * 100, depth_median))

                if use_physics_only:
                    log.info("[Stumpf] ATL quality check → PHYSICS-ONLY: %s",
                             "; ".join(physics_only_reasons))
                else:
                    log.info("[Stumpf] ATL quality check → ATL-CALIBRATED "
                             "(|corr|=%.2f, IQR=%.2fm, near_floor=%.0f%%)",
                             abs_corr, depth_iqr, near_floor_frac * 100)

                use_isotonic = (not use_physics_only
                                and x_unique.size >= 10
                                and depth_range >= 2.0
                                and depth_std >= 0.5)

                if use_isotonic:
                    stumpf_lr = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
                    stumpf_lr.fit(x_fit, y_fit)
                    pred = np.full(len(df), np.nan, dtype=np.float32)
                    pred[m] = np.asarray(stumpf_lr.predict(x_fit), dtype=np.float32)
                    df["stumpf_depth"] = pred
                    feat_cols.append("stumpf_depth")
                    log.info("Fitted monotonic stumpf_depth model using IsotonicRegression (increasing=%s).", increasing)
                else:
                    if use_physics_only:
                        # -------------------------------------------------------
                        # PHYSICS-ONLY STUMPF MODEL (no ATL calibration)
                        # -------------------------------------------------------
                        # When training data is shallow-dominated (IQR < 1m),
                        # the ATL points are mostly near-surface noise. Using
                        # them to calibrate produces a flat or inverted slope.
                        #
                        # Instead, use the Stumpf ratio as a pure physics-based
                        # relative depth index. The ratio DECREASES with depth
                        # (blue penetrates deeper than green, so in deeper water
                        # the green signal drops faster → lower ratio).
                        #
                        # depth ≈ physics_max * (si_max - stumpf_idx) / si_range
                        #
                        # where si_max = value at shallowest water (shoreline).
                        # Reference: Stumpf et al. 2003, Lyzenga 1978
                        # -------------------------------------------------------
                        si_min = float(np.percentile(x_fit, 2))
                        si_max = float(np.percentile(x_fit, 98))
                        si_range = max(si_max - si_min, 0.01)

                        physics_max_depth = max_depth_sdb if max_depth_sdb and max_depth_sdb < 100 else 20.0

                        slope = -physics_max_depth / si_range
                        intercept = physics_max_depth * si_max / si_range

                        stumpf_lr = LinearRegression()
                        stumpf_lr.coef_ = np.array([slope])
                        stumpf_lr.intercept_ = intercept

                        pred = np.full(len(df), np.nan, dtype=np.float32)
                        pred[m] = np.clip(
                            intercept + slope * x_fit, 0.0, physics_max_depth
                        ).astype(np.float32)
                        df["stumpf_depth"] = pred
                        feat_cols.append("stumpf_depth")
                        log.info("[Stumpf PHYSICS] Physics-only model (no ATL calibration): "
                                 "depth = %.2f + %.2f * stumpf_idx "
                                 "(si_range=[%.3f, %.3f], physics_max=%.1fm). "
                                 "Shallow-dominated ATL data bypassed.",
                                 intercept, slope, si_min, si_max, physics_max_depth)
                    else:
                        # Standard linear fit (adequate data quality)
                        stumpf_lr = LinearRegression()
                        stumpf_lr.fit(x_fit.reshape(-1, 1), y_fit)
                        pred = np.full(len(df), np.nan, dtype=np.float32)
                        pred[m] = stumpf_lr.predict(x_fit.reshape(-1, 1)).astype(np.float32)
                        df["stumpf_depth"] = pred
                        feat_cols.append("stumpf_depth")
                        log.info("Stumpf LR coefficients: depth = %.3f + %.3f * stumpf_idx",
                                 float(stumpf_lr.intercept_), float(stumpf_lr.coef_[0]))
                        if depth_range < 2.0 or depth_std < 0.5:
                            log.info("Fitted linear stumpf_depth model (depth_range=%.1fm, "
                                     "std=%.2fm too narrow for isotonic).",
                                     depth_range, depth_std)
                        else:
                            log.info("Fitted auxiliary stumpf_depth LR model.")
            except Exception as e:
                log.warning("Stumpf monotonic/LR model failed: %s", e)

    df = _sanitize_feature_columns(df, feat_cols + ["depth_m", "sample_weight"])
    _count(df, "after sanitize (inf->nan, coercion)")

    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0

    req_cols = feat_cols + ["depth_m", "sample_weight"]
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        log.error("Missing required columns: %s. Training aborted.", missing_cols)
        return RandomForestRegressor(), stumpf_lr, pd.DataFrame(), pd.DataFrame(), {}

    _nonfinite_report(df, req_cols, "pre finite-row drop")
    mat = df[req_cols].to_numpy(np.float32)
    m_all = np.all(np.isfinite(mat), axis=1)
    df = df[m_all].reset_index(drop=True)
    _count(df, "after finite-row drop")
    _funnel_df_stats(df, "after_finite_row_drop", rr=rr)

    training_bounds: Dict[str, Dict[str, float]] = {}
    doa_percentiles = (2.0, 98.0)
    doa_buffer_frac = 0.03
    for col in feat_cols:
        vals = df[col].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            p_min, p_max = np.percentile(vals, doa_percentiles)
            buff = max((p_max - p_min) * doa_buffer_frac, 1e-6)
            training_bounds[col] = {"min": float(p_min - buff), "max": float(p_max + buff)}
        else:
            training_bounds[col] = {"min": -9999.0, "max": 9999.0}

    log.info("Calculated conservative Domain of Applicability bounds for %s features.", len(feat_cols))

    doa_priority = {
        "stumpf_depth": 6.0,
        "stumpf_idx": 4.0,
        "B04_B03": 3.0,
        "B03_B02": 3.0,
        "brightness": 2.0,
        "B03": 1.5,
        "B02": 1.5,
        "B04": 1.2,
        "B08": 1.0,
        "log_B03": 1.0,
        "log_B02": 1.0,
        "log_B04": 0.8,
        "log_B08": 0.6,
        "nbri": 1.5,
    }
    weight_sum = float(sum(doa_priority.get(c, 1.0) for c in feat_cols)) or 1.0
    doa_weights: Dict[str, float] = {c: float(doa_priority.get(c, 1.0) / weight_sum) for c in feat_cols}

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
            "soft_k_default": 8.0,
            "threshold_default": 0.97,
            "weights": _to_python_float(doa_weights),
            "bounds_method": {"percentiles": [2.0, 98.0], "buffer_frac": 0.03},
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
    if model_bank_context:
        metadata["model_bank_partition"] = _to_python_float(dict(model_bank_context))


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
                    'partition': dict(model_bank_context or {}),
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
            log.warning("Update failed; continuing without bank: %s", ex)
            metadata['model_bank'] = {'enabled': False, 'error': str(ex)}
    else:
        metadata['model_bank'] = {'enabled': False}

    # If using model bank, optionally reuse an existing trained model until enough new samples accumulate.
    # This provides stability across adjacent tiles and avoids overfitting to tiny incremental updates.
    if model_bank_enabled and model_bank_dir is not None:
        try:
            bank_dir_p = Path(model_bank_dir)
            meta_p = bank_dir_p / "bank_meta.json"
            last_trained_n_seen = 0
            n_seen_now = 0
            if meta_p.exists():
                try:
                    with open(meta_p, "r", encoding="utf-8") as _f:
                        _bm = json.load(_f)
                    last_trained_n_seen = int(_bm.get("last_trained_n_seen", 0))
                    n_seen_now = int(_bm.get("n_seen", 0))
                except Exception:
                    last_trained_n_seen = 0
                    n_seen_now = 0
            new_since_train = max(0, n_seen_now - last_trained_n_seen) if n_seen_now else 0

            if (bank_dir_p / "rf_model.pkl").exists() and (new_since_train < int(model_bank_retrain_min_new)):
                result = _try_reuse_model_bank(
                    model_bank_dir, metadata,
                    reason="periodic_retrain_skip",
                    extra_meta={
                        "new_since_train": int(new_since_train),
                        "retrain_min_new": int(model_bank_retrain_min_new),
                    },
                    required_context=model_bank_context,
                )
                if result is not None:
                    return result
        except Exception as ex:
            log.debug("Reuse check failed; proceeding to retrain: %s", ex, exc_info=True)

    if len(df) < min_training_points_for_sdb:
        log.warning("Insufficient samples (%s < %s).", len(df), min_training_points_for_sdb)

        # If a model bank exists, prefer reusing its last trained model rather than returning
        # an untrained RF (which can later look like a "successful" run but produce nonsense).
        if model_bank_enabled and model_bank_dir is not None:
            try:
                result = _try_reuse_model_bank(
                    model_bank_dir, metadata,
                    reason="insufficient_new_samples",
                    extra_meta={
                        "min_training_points_for_sdb": int(min_training_points_for_sdb),
                        "n_samples": int(len(df)),
                    },
                    required_context=model_bank_context,
                )
                if result is not None:
                    return result
            except Exception as ex:
                log.debug("Insufficient-sample reuse failed; returning empty model: %s", ex, exc_info=True)

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
    df_validation = df.copy()
    if "source_norm" in df_validation.columns:
        src_series = df_validation["source_norm"].astype(str).str.lower()
        val_mask = src_series.str.startswith("extra_xyz") | src_series.str.contains("authoritative_base", regex=False) | src_series.isin(atl_like)
        if bool(val_mask.any()):
            df_validation = df_validation.loc[val_mask].copy()
    if df_validation.empty:
        log.info("Validation candidate set is empty after source filtering. Falling back to full training dataframe.")
        df_validation = df.copy()
    metadata.setdefault("validation_support", {})
    metadata["validation_support"].update({
        "uses_authoritative_extra_xyz": bool("source_norm" in df_validation.columns and df_validation["source_norm"].astype(str).str.lower().str.startswith("extra_xyz").any()),
        "uses_atl": bool("source_norm" in df_validation.columns and df_validation["source_norm"].astype(str).str.lower().isin(atl_like).any()),
        "n_candidates": int(len(df_validation)),
    })

    # --- TRAIN/TEST SPLIT LOGIC ---
    if not spatial_split:
        log.info("Performing Stratified Random Split (by Depth Quantile) on mixed authoritative+ATL support...")
        idx_tr, idx_te = stratified_train_test_split(
            df_validation,
            target_col='depth_m',
            test_size=0.2,
            seed=seed
        )
    else:
        # Pick the spatial cluster that best represents the full depth range.
        log.info("Performing Spatial Split (K-Means Clustering) on mixed authoritative+ATL support...")
        coords = df_validation[["longitude", "latitude"]].to_numpy()

        # Robustness: KMeans can fail (or behave poorly) when sample counts are small.
        # Also, sklearn versions prior to 1.4 may not accept n_init=10.
        do_spatial = coords.shape[0] >= 200
        if not do_spatial:
            log.warning(
                "Spatial split requested but too few samples for stable clustering (n=%d). "
                "Falling back to stratified random split.",
                coords.shape[0],
            )
            idx_tr, idx_te = stratified_train_test_split(
                df_validation,
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
            # Analyze clusters using positive depth magnitudes. Reject shallow, low-variance,
            # non-representative holdouts instead of scoring them as meaningful spatial tests.
            depth_abs_all = np.abs(pd.to_numeric(df_validation['depth_m'], errors='coerce').to_numpy(dtype=float))
            finite_all = np.isfinite(depth_abs_all)
            depth_abs_all = depth_abs_all[finite_all]

            if depth_abs_all.size == 0:
                log.warning(
                    "Spatial split requested but no finite depth values were available. "
                    "Using full ATL set for training and leaving the spatial test set empty."
                )
                idx_tr = df_validation.index.to_numpy()
                idx_te = np.array([], dtype=df_validation.index.dtype)
                metadata.setdefault('spatial_split_status', {})
                metadata['spatial_split_status'].update({
                    'ok': False,
                    'reason': 'no_finite_depth_values',
                })
            else:
                global_p50 = float(np.percentile(depth_abs_all, 50))
                global_p95 = float(np.percentile(depth_abs_all, 95))
                global_span = float(np.nanmax(depth_abs_all) - np.nanmin(depth_abs_all))
                global_std = float(np.nanstd(depth_abs_all))
                n_total = int(depth_abs_all.size)
                min_test_n = max(30, int(round(0.08 * n_total)))
                min_test_frac = 0.10
                max_test_frac = 0.60
                min_depth_span_m = min(max(0.50, 0.25 * global_span), global_span) if global_span > 0 else 0.50
                min_depth_std_m = max(0.10, 0.20 * global_std)
                min_test_p95_ratio = 0.85
                min_train_p95_ratio = 0.85
                min_test_p50_ratio = 0.75
                max_test_p50_ratio = 1.25
                min_train_p50_ratio = 0.75
                max_train_p50_ratio = 1.25
                min_test_std_ratio = 0.50 if global_std > 0 else 0.0
                min_train_std_ratio = 0.50 if global_std > 0 else 0.0
                max_test_p95_abs_diff_m = max(0.30, 0.20 * global_p95)
                max_train_p95_abs_diff_m = max(0.30, 0.20 * global_p95)
                max_test_p50_abs_diff_m = max(0.20, 0.20 * global_p50)
                max_train_p50_abs_diff_m = max(0.20, 0.20 * global_p50)
                global_sources = set(df_validation['source_norm'].dropna().astype(str).unique()) if 'source_norm' in df_validation.columns else set()

                best = None
                candidate_stats = []
                for k in range(int(km.n_clusters)):
                    mask = (km.labels_ == k)
                    n_k = int(np.sum(mask))
                    if n_k < min_test_n:
                        continue

                    test_frac = n_k / max(1, n_total)
                    if test_frac < min_test_frac or test_frac > max_test_frac:
                        continue

                    d_test = np.abs(pd.to_numeric(
                        df_validation.loc[df_validation.index[mask], 'depth_m'], errors='coerce'
                    ).to_numpy(dtype=float))
                    d_test = d_test[np.isfinite(d_test)]
                    if d_test.size < min_test_n:
                        continue

                    d_train = np.abs(pd.to_numeric(
                        df_validation.loc[df_validation.index[~mask], 'depth_m'], errors='coerce'
                    ).to_numpy(dtype=float))
                    d_train = d_train[np.isfinite(d_train)]
                    if d_train.size < min_test_n:
                        continue

                    test_span = float(np.nanmax(d_test) - np.nanmin(d_test)) if d_test.size else 0.0
                    test_std = float(np.nanstd(d_test)) if d_test.size else 0.0
                    train_span = float(np.nanmax(d_train) - np.nanmin(d_train)) if d_train.size else 0.0
                    train_std = float(np.nanstd(d_train)) if d_train.size else 0.0
                    test_p50 = float(np.percentile(d_test, 50))
                    test_p95 = float(np.percentile(d_test, 95))
                    train_p50 = float(np.percentile(d_train, 50))
                    train_p95 = float(np.percentile(d_train, 95))

                    test_p95_ratio = (test_p95 / global_p95) if global_p95 > 0 else 1.0
                    train_p95_ratio = (train_p95 / global_p95) if global_p95 > 0 else 1.0
                    test_p50_ratio = (test_p50 / global_p50) if global_p50 > 0 else 1.0
                    train_p50_ratio = (train_p50 / global_p50) if global_p50 > 0 else 1.0
                    test_std_ratio = (test_std / global_std) if global_std > 0 else 1.0
                    train_std_ratio = (train_std / global_std) if global_std > 0 else 1.0

                    depth_floor_ok = (
                        (test_span >= min_depth_span_m)
                        and (train_span >= min_depth_span_m)
                        and (test_std >= min_depth_std_m)
                        and (train_std >= min_depth_std_m)
                    )
                    depth_ratio_ok = (
                        (test_p95_ratio >= min_test_p95_ratio)
                        and (train_p95_ratio >= min_train_p95_ratio)
                        and (min_test_p50_ratio <= test_p50_ratio <= max_test_p50_ratio)
                        and (min_train_p50_ratio <= train_p50_ratio <= max_train_p50_ratio)
                        and (test_std_ratio >= min_test_std_ratio)
                        and (train_std_ratio >= min_train_std_ratio)
                    )
                    depth_absdiff_ok = (
                        (abs(test_p95 - global_p95) <= max_test_p95_abs_diff_m)
                        and (abs(train_p95 - global_p95) <= max_train_p95_abs_diff_m)
                        and (abs(test_p50 - global_p50) <= max_test_p50_abs_diff_m)
                        and (abs(train_p50 - global_p50) <= max_train_p50_abs_diff_m)
                    )
                    depth_ok = depth_floor_ok and depth_ratio_ok and depth_absdiff_ok

                    reject_reasons = []
                    if test_span < min_depth_span_m:
                        reject_reasons.append('test_span_small')
                    if train_span < min_depth_span_m:
                        reject_reasons.append('train_span_small')
                    if test_std < min_depth_std_m:
                        reject_reasons.append('test_std_small')
                    if train_std < min_depth_std_m:
                        reject_reasons.append('train_std_small')
                    if test_p95_ratio < min_test_p95_ratio:
                        reject_reasons.append('test_p95_too_shallow')
                    if train_p95_ratio < min_train_p95_ratio:
                        reject_reasons.append('train_p95_too_shallow')
                    if not (min_test_p50_ratio <= test_p50_ratio <= max_test_p50_ratio):
                        reject_reasons.append('test_p50_unrepresentative')
                    if not (min_train_p50_ratio <= train_p50_ratio <= max_train_p50_ratio):
                        reject_reasons.append('train_p50_unrepresentative')
                    if test_std_ratio < min_test_std_ratio:
                        reject_reasons.append('test_variance_too_small')
                    if train_std_ratio < min_train_std_ratio:
                        reject_reasons.append('train_variance_too_small')
                    if abs(test_p95 - global_p95) > max_test_p95_abs_diff_m:
                        reject_reasons.append('test_p95_absdiff_large')
                    if abs(train_p95 - global_p95) > max_train_p95_abs_diff_m:
                        reject_reasons.append('train_p95_absdiff_large')
                    if abs(test_p50 - global_p50) > max_test_p50_abs_diff_m:
                        reject_reasons.append('test_p50_absdiff_large')
                    if abs(train_p50 - global_p50) > max_train_p50_abs_diff_m:
                        reject_reasons.append('train_p50_absdiff_large')

                    if 'source_norm' in df_validation.columns:
                        test_sources = set(df_validation.loc[df_validation.index[mask], 'source_norm'].dropna().astype(str).unique())
                        train_sources = set(df_validation.loc[df_validation.index[~mask], 'source_norm'].dropna().astype(str).unique())
                    else:
                        test_sources = set()
                        train_sources = set()
                    source_overlap = len(test_sources & train_sources)
                    source_ok = (len(test_sources) > 0 and source_overlap > 0) or (not global_sources)

                    score = (
                        abs(test_p95 - global_p95)
                        + 0.50 * abs(test_p50 - global_p50)
                        + 0.25 * abs(test_frac - 0.20) * max(global_p95, 1.0)
                    )
                    if test_span > 0:
                        score -= 0.10 * min(test_span, global_span)
                    if test_std > 0:
                        score -= 0.10 * min(test_std, global_std)
                    if source_overlap > 0:
                        score -= 0.05 * min(source_overlap, 3)

                    candidate = {
                        'cluster': int(k),
                        'n_test': n_k,
                        'test_frac': float(test_frac),
                        'test_p50': test_p50,
                        'test_p95': test_p95,
                        'train_p50': train_p50,
                        'train_p95': train_p95,
                        'test_span': test_span,
                        'test_std': test_std,
                        'train_std': train_std,
                        'test_p50_ratio': float(test_p50_ratio),
                        'test_p95_ratio': float(test_p95_ratio),
                        'train_p50_ratio': float(train_p50_ratio),
                        'train_p95_ratio': float(train_p95_ratio),
                        'test_std_ratio': float(test_std_ratio),
                        'train_std_ratio': float(train_std_ratio),
                        'depth_ok': bool(depth_ok),
                        'source_ok': bool(source_ok),
                        'score': float(score),
                        'reject_reasons': list(reject_reasons),
                        'test_sources': sorted(test_sources),
                        'train_sources': sorted(train_sources),
                    }
                    candidate_stats.append(candidate)
                    log.info(
                        "Spatial split candidate cluster=%s n_train=%s n_test=%s train_p95=%.2fm test_p95=%.2fm "
                        "train_p95/global=%.2f test_p95/global=%.2f train_std/global=%.2f test_std/global=%.2f "
                        "train_sources=%s test_sources=%s depth_ok=%s source_ok=%s score=%.3f reject=%s",
                        candidate['cluster'],
                        int(d_train.size),
                        candidate['n_test'],
                        candidate['train_p95'],
                        candidate['test_p95'],
                        candidate['train_p95_ratio'],
                        candidate['test_p95_ratio'],
                        candidate['train_std_ratio'],
                        candidate['test_std_ratio'],
                        candidate['train_sources'],
                        candidate['test_sources'],
                        candidate['depth_ok'],
                        candidate['source_ok'],
                        candidate['score'],
                        ','.join(candidate['reject_reasons']) if candidate['reject_reasons'] else 'none',
                    )

                    if depth_ok and source_ok:
                        if best is None or candidate['score'] < best['score']:
                            best = candidate

                metadata.setdefault('spatial_split_status', {})
                metadata['spatial_split_status'].update({
                    'ok': bool(best is not None),
                    'min_test_n': int(min_test_n),
                    'min_depth_span_m': float(min_depth_span_m),
                    'min_depth_std_m': float(min_depth_std_m),
                    'min_test_p95_ratio': float(min_test_p95_ratio),
                    'min_train_p95_ratio': float(min_train_p95_ratio),
                    'min_test_std_ratio': float(min_test_std_ratio),
                    'min_train_std_ratio': float(min_train_std_ratio),
                    'candidate_count': int(len(candidate_stats)),
                })

                if best is None:
                    log.warning(
                        "Spatial split requested, but no representative cluster holdout passed the minimum "
                        "depth-span/source-overlap tests. Training on the full ATL set and leaving the "
                        "spatial test set empty rather than reporting a misleading spatial score."
                    )
                    idx_tr = df_validation.index.to_numpy()
                    idx_te = np.array([], dtype=df_validation.index.dtype)
                    metadata['spatial_split_status']['reason'] = 'no_representative_cluster_holdout'
                else:
                    log.info(
                        "Spatial Split: Global p50=%.2fm p95=%.2fm. Selected Cluster %s as Test "
                        "(test_p50=%.2fm test_p95=%.2fm; train_p50=%.2fm train_p95=%.2fm).",
                        global_p50,
                        global_p95,
                        best['cluster'],
                        best['test_p50'],
                        best['test_p95'],
                        best['train_p50'],
                        best['train_p95'],
                    )
                    idx_te = df_validation.index[km.labels_ == best['cluster']].to_numpy()
                    idx_tr = df_validation.index[km.labels_ != best['cluster']].to_numpy()
                    metadata['spatial_split_status'].update({
                        'selected_cluster': int(best['cluster']),
                        'selected_test_n': int(best['n_test']),
                        'selected_test_p50_m': float(best['test_p50']),
                        'selected_test_p95_m': float(best['test_p95']),
                        'selected_test_span_m': float(best['test_span']),
                        'selected_test_std_m': float(best['test_std']),
                        'selected_test_sources': list(best['test_sources']),
                        'selected_train_sources': list(best['train_sources']),
                    })

    final_tr_indices = np.unique(np.asarray(idx_tr, dtype=df.index.dtype))

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
            log.warning("Dropping %s rows with non-finite depth_m in %s.", n_drop, _label)
        _df2 = _df2.loc[m].copy()
        _df2['depth_m'] = d[m]
        return _df2

    df_tr = _coerce_depth_m(df_tr, 'train set')
    df_te = _coerce_depth_m(df_te, 'test set')

    _count(df_tr, "final train set")
    _count(df_te, "final test set")
    _funnel_df_stats(df_tr, "split_train", rr=rr)
    _funnel_df_stats(df_te, "split_test", rr=rr)

    guidance_settings = _compute_physics_guidance_settings(df)
    if bool(spatial_split):
        guidance_settings = _tighten_guidance_for_unrepresentative_spatial_holdout(
            guidance_settings,
            metadata.get("spatial_split_status", {}),
        )
    metadata["physics_guidance"] = _to_python_float(guidance_settings)
    log.info(
        "Physics-guided training mode: support_score=%.2f low_support=%s n=%d tracks=%d depth_span=%.2fm correction_alpha=%.2f residual_clip=%.2fm",
        guidance_settings.get("support_score", 0.0),
        guidance_settings.get("low_support", True),
        guidance_settings.get("n_points", 0),
        guidance_settings.get("unique_tracks", 0),
        guidance_settings.get("depth_span_m", 0.0),
        guidance_settings.get("correction_alpha", 0.35),
        guidance_settings.get("residual_clip_m", 0.5),
    )

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
            log.warning("Dropping %s/%s rows with non-finite depth_m after coercion", format(_dropped, ","), format(_n0, ","))
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
                "Final source quota cannot be satisfied with a single source. "
                "Applied target downsampling only: %s -> %s rows; dominant source fraction remains %.3f.",
                f"{n0:,}", f"{len(df_out):,}", dom1,
            )
            log.info("Source counts before quota: %s", vc0.to_dict())
            log.info("Source counts after quota: %s", vc1.to_dict())
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
            # Do not refill from leftovers: those groups are already at quota.
            # Refilling would violate the source-fraction invariant.
            if total_capped < target_n:
                log.warning(
                    "Final source quota limits available rows below target: "
                    "target=%s, quota-limited rows=%s. Applying strict fraction cap on actual rows.",
                    f"{target_n:,}", f"{total_capped:,}",
                )

            # Second pass can enforce max_source_frac on the ACTUAL retained row count, but only
            # when doing so does not destroy the fit sample. In guidance-anchored regimes we still want
            # dense authoritative support to inform optics; collapsing thousands of retained rows down to
            # a few dozen is scientifically worse than tolerating some remaining source dominance.
            pre_second_pass_total = int(sum(counts.values()))
            min_rows_for_strict_actual_cap = int(max(1000, np.floor(0.25 * target_n)))

            sorted_counts = sorted((int(v) for v in counts.values()), reverse=True)
            dominant_count = int(sorted_counts[0]) if sorted_counts else 0
            other_count = int(sum(sorted_counts[1:])) if len(sorted_counts) > 1 else 0
            if max_source_frac >= 1.0:
                feasible_total_under_actual_cap = pre_second_pass_total
            elif other_count <= 0:
                feasible_total_under_actual_cap = 0
            else:
                feasible_total_under_actual_cap = int(min(pre_second_pass_total, np.floor(other_count / max(1.0e-9, (1.0 - max_source_frac)))))

            minority_pool_too_small = other_count < max(50, int(np.ceil(0.10 * target_n)))
            catastrophic_collapse_ratio = (
                (pre_second_pass_total > 0)
                and (feasible_total_under_actual_cap / float(pre_second_pass_total) < 0.50)
            )
            allow_strict_actual_cap = (
                pre_second_pass_total >= min_rows_for_strict_actual_cap
                and feasible_total_under_actual_cap >= min_rows_for_strict_actual_cap
                and not minority_pool_too_small
                and not catastrophic_collapse_ratio
            )
            if allow_strict_actual_cap:
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
            else:
                log.warning(
                    "Final source quota skipped strict actual-row cap to avoid collapsing the fit set "
                    "(quota_limited_rows=%s, feasible_total_under_actual_cap=%s, min_rows_for_strict_actual_cap=%s, "
                    "minority_pool_too_small=%s, catastrophic_collapse_ratio=%s). "
                    "Keeping target-based capped pool.",
                    f"{pre_second_pass_total:,}",
                    f"{feasible_total_under_actual_cap:,}",
                    f"{min_rows_for_strict_actual_cap:,}",
                    minority_pool_too_small,
                    catastrophic_collapse_ratio,
                )

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
                    "falling back to minimal multi-source subset (n=%d).",
                    len(df_out),
                )
            else:
                keep_parts = [src_idx[s][:counts[s]] for s in counts if counts[s] > 0]
                keep_idx = np.sort(np.concatenate(keep_parts)) if keep_parts else np.array([], dtype=_df.index.dtype)
                df_out = _df.loc[keep_idx].copy()

        vc1 = df_out[source_col_local].value_counts(dropna=False)
        dom1 = float(vc1.iloc[0] / max(1, len(df_out))) if len(vc1) else 0.0
        log.warning(
            "Enforced final source quota (target=%s, max_source_frac=%.2f, quota=%s): "
            "%s -> %s rows. Dominant source fraction %.3f -> %.3f.",
            f"{target_n:,}", max_source_frac, f"{quota_n:,}",
            f"{n0:,}", f"{len(df_out):,}", dom0, dom1,
        )
        log.info("Source counts before quota: %s", vc0.to_dict())
        log.info("Source counts after quota: %s", vc1.to_dict())
        return df_out


    try:
        final_target_n = int(max(5000, min(len(df_tr_fit), int(model_bank_max_samples))))
    except Exception:
        final_target_n = int(max(5000, len(df_tr_fit)))
    df_tr_fit = _enforce_final_source_quota(df_tr_fit, seed, final_target_n, _max_source_frac=0.85)

    missing = [c for c in feat_cols if (c not in df_tr_fit.columns) or (c not in df_te.columns)]
    if missing:
        log.warning("Dropping missing feature columns: %s", missing)
        feat_cols = [c for c in feat_cols if c not in missing]

    X_train = df_tr_fit[feat_cols].to_numpy()
    y_train_raw = df_tr_fit["depth_m"].to_numpy()
    w_train = df_tr_fit["sample_weight"].to_numpy() if "sample_weight" in df_tr_fit.columns else None

    # Refresh arrays after any optional rebalancing / QC changes to df_tr_fit
    X_train = df_tr_fit[feat_cols].to_numpy()
    y_train_raw = df_tr_fit["depth_m"].to_numpy()
    w_train = df_tr_fit["sample_weight"].to_numpy() if "sample_weight" in df_tr_fit.columns else None

    depths_finite = y_train_raw[np.isfinite(y_train_raw)]
    
    if len(depths_finite) == 0:
        log.error("No finite depth values in training data")
        return RandomForestRegressor(), stumpf_lr, pd.DataFrame(), pd.DataFrame(), {}
    
    pct_negative = (depths_finite < 0).mean()
    pct_positive = (depths_finite > 0).mean()
    
    log.info("Depth sign distribution: %.1f%% negative, %.1f%% positive", pct_negative*100, pct_positive*100)
    log.info("Depth range: [%.2f, %.2f] m", depths_finite.min(), depths_finite.max())
    
    # Error if mostly positive (wrong convention)
    if pct_positive > 0.9:
        log.error("Depth sign error: >90%% of depths are positive; expected negative-down")
        log.error("Check the depth column and sign convention in your XYZ data")
        raise ValueError("Invalid depth sign convention: depths should be negative (below surface)")
    
    # Warning if mixed or mostly positive
    if pct_negative < 0.8:
        log.warning("Only %.1f%% of depths are negative; verify sign convention in source data", pct_negative*100)

    # Model uses a physics-first baseline (Stumpf depth) plus a bounded RF residual.
    y_train_mag = np.abs(y_train_raw)
    log.info("Converted depths to positive magnitudes for training: [%.2f, %.2f] m", y_train_mag[np.isfinite(y_train_mag)].min(), y_train_mag[np.isfinite(y_train_mag)].max())

    _anchor_good = bool(guidance_settings.get("anchor_support_good", False))

    if _anchor_good:
        # Dense authoritative XYZ: train RF on ABSOLUTE depth, not Stumpf residual.
        # The Stumpf baseline is wrong in turbid channels (low blue/green ratio
        # misinterpreted as shallow).  Training on the residual forces the RF to
        # fight against the wrong baseline.  Training on absolute depth lets the
        # RF learn the correct spectral→depth mapping directly from survey data.
        y_train = y_train_mag.astype(np.float32)
        metadata["prediction_mode"] = "direct_depth"
        metadata["residual_clip_m"] = 0.0
        metadata["residual_target_stats"] = {
            "min": float(np.nanmin(y_train)),
            "max": float(np.nanmax(y_train)),
            "p05": float(np.nanpercentile(y_train, 5)),
            "p95": float(np.nanpercentile(y_train, 95)),
        }
        log.info(
            "Training RF on DIRECT DEPTH (anchor_support_good): range [%.2f, %.2f] m "
            "(bypassing Stumpf residual architecture for turbid-water accuracy)",
            float(np.nanmin(y_train)), float(np.nanmax(y_train)),
        )
    else:
        if "stumpf_depth" not in df_tr_fit.columns:
            log.error("Physics-guided residual mode requires stumpf_depth feature; training aborted.")
            raise ValueError("Missing stumpf_depth feature for physics-guided residual training")
        stumpf_base_train = np.maximum(pd.to_numeric(df_tr_fit["stumpf_depth"], errors="coerce").to_numpy(dtype=np.float32), 0.0)
        residual_clip_m = float(guidance_settings.get("residual_clip_m", 0.5))
        y_train = np.clip(y_train_mag - stumpf_base_train, -residual_clip_m, residual_clip_m).astype(np.float32)
        metadata["prediction_mode"] = "stumpf_residual"
        metadata["residual_clip_m"] = residual_clip_m
        metadata["residual_target_stats"] = {
            "min": float(np.nanmin(y_train)),
            "max": float(np.nanmax(y_train)),
            "p05": float(np.nanpercentile(y_train, 5)),
            "p95": float(np.nanpercentile(y_train, 95)),
        }
        log.info(
            "Training RF on bounded residuals about Stumpf baseline: residual range [%.2f, %.2f] m (clip=±%.2f, alpha=%.2f)",
            float(np.nanmin(y_train)),
            float(np.nanmax(y_train)),
            residual_clip_m,
            float(guidance_settings.get("correction_alpha", 0.35)),
        )

    # Guardrail: avoid single-source domination
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
                        "Dominant source '%s' contributes %.1f%% of training rows (%d/%d). "
                        "Expect weak generalization / source-specific bias.",
                        dom_src, dom_frac * 100, dom_n, total_n,
                    )
                # Conservative deterministic cap: only trim if one source dominates AND others exist,
                # *and* doing so will not catastrophically collapse the training set.  In dense
                # authoritative-support mode we prefer keeping a large, source-dominated but physically
                # grounded fit set over shrinking the sample to a tiny mixed subset that no longer reflects
                # the survey-controlled optics regime.
                max_dom_frac = 0.70
                if dom_frac > 0.90 and total_n >= 1000:
                    allowed_dom = int(max(100, (max_dom_frac / max(1e-9, 1.0 - max_dom_frac)) * (total_n - dom_n)))
                    other_n = int(total_n - dom_n)
                    rebalance_total = int(other_n + min(dom_n, allowed_dom))
                    collapse_ratio = rebalance_total / float(max(1, total_n))
                    minority_pool_too_small = other_n < max(50, int(np.ceil(0.10 * total_n)))
                    catastrophic_collapse = rebalance_total < 1000 or collapse_ratio < 0.50
                    if allowed_dom < dom_n and not minority_pool_too_small and not catastrophic_collapse:
                        dom_idx = df_tr_fit.index[df_tr_fit[source_col] == dom_src].to_numpy()
                        keep_dom = np.random.default_rng(int(seed)).choice(dom_idx, size=allowed_dom, replace=False)
                        keep_other = df_tr_fit.index[df_tr_fit[source_col] != dom_src].to_numpy()
                        keep_idx = np.concatenate([keep_other, keep_dom])
                        df_tr_fit = df_tr_fit.loc[keep_idx].copy()
                        log.warning(
                            "Rebalanced dominant source '%s' to reduce count domination: "
                            "%d -> %d rows (dominant kept=%d).",
                            dom_src, total_n, len(df_tr_fit), allowed_dom,
                        )
                        metadata['training_qc']['dominant_source_rebalanced'] = True
                        metadata['training_qc']['dominant_source_rebalanced_target_frac'] = float(max_dom_frac)
                    elif allowed_dom < dom_n:
                        log.warning(
                            "Skipped dominant-source row rebalance for '%s' to avoid collapsing the fit set "
                            "(total=%d, dominant=%d, minority=%d, proposed_total=%d, collapse_ratio=%.3f, minority_pool_too_small=%s).",
                            dom_src,
                            total_n,
                            dom_n,
                            other_n,
                            rebalance_total,
                            collapse_ratio,
                            minority_pool_too_small,
                        )
                        metadata['training_qc']['dominant_source_rebalanced'] = False
                        metadata['training_qc']['dominant_source_rebalance_skipped'] = True
                        metadata['training_qc']['dominant_source_rebalance_skip_reason'] = 'catastrophic_collapse_or_tiny_minority_pool'
        except (TypeError, ValueError, KeyError, RuntimeError) as _ex:
            log.error("Dominance guardrail failed: %s", _ex, exc_info=True)

    # Refresh arrays after all training-row QC / source-balance adjustments.
    if not feat_cols:
        raise ValueError("No training feature columns remain after QC and schema checks")
    X_train = df_tr_fit.loc[:, feat_cols].to_numpy(dtype=np.float32, copy=True)
    y_train_raw = pd.to_numeric(df_tr_fit["depth_m"], errors="coerce").to_numpy(dtype=np.float32)
    w_train = (
        pd.to_numeric(df_tr_fit["sample_weight"], errors="coerce").to_numpy(dtype=np.float32)
        if "sample_weight" in df_tr_fit.columns
        else None
    )

    if X_train.ndim != 2:
        raise ValueError(f"Training design matrix must be 2D; got shape={getattr(X_train, 'shape', None)}")
    if X_train.shape[0] != len(df_tr_fit):
        raise ValueError(
            f"Training design matrix row mismatch after QC: X_rows={X_train.shape[0]} df_rows={len(df_tr_fit)}"
        )
    if X_train.shape[1] != len(feat_cols):
        raise ValueError(
            f"Training design matrix column mismatch after QC: X_cols={X_train.shape[1]} feat_cols={len(feat_cols)}"
        )
    if not np.isfinite(X_train).all():
        bad = int(np.size(X_train) - np.isfinite(X_train).sum())
        raise ValueError(f"Training design matrix contains {bad} non-finite feature values after QC")
    if not np.isfinite(y_train_raw).all():
        bad = int(y_train_raw.size - np.isfinite(y_train_raw).sum())
        raise ValueError(f"Training target contains {bad} non-finite depth values after QC")
    if w_train is not None:
        if w_train.ndim != 1 or w_train.shape[0] != len(df_tr_fit):
            raise ValueError(
                f"Sample-weight vector mismatch after QC: shape={getattr(w_train, 'shape', None)} df_rows={len(df_tr_fit)}"
            )
        if not np.isfinite(w_train).all():
            bad = int(w_train.size - np.isfinite(w_train).sum())
            raise ValueError(f"Sample-weight vector contains {bad} non-finite values after QC")

    if metadata.get("prediction_mode") == "direct_depth":
        y_train = np.abs(y_train_raw).astype(np.float32)
        metadata["residual_target_stats"] = {
            "min": float(np.nanmin(y_train)),
            "max": float(np.nanmax(y_train)),
            "p05": float(np.nanpercentile(y_train, 5)),
            "p95": float(np.nanpercentile(y_train, 95)),
        }
        log.info(
            "Final direct-depth target after QC: n=%d range=[%.2f, %.2f] m",
            len(y_train),
            float(np.nanmin(y_train)),
            float(np.nanmax(y_train)),
        )
    else:
        if "stumpf_depth" not in df_tr_fit.columns:
            raise ValueError("Missing stumpf_depth feature after final QC for residual training")
        residual_clip_m = float(metadata.get("residual_clip_m", guidance_settings.get("residual_clip_m", 0.5)))
        stumpf_base_train = np.maximum(
            pd.to_numeric(df_tr_fit["stumpf_depth"], errors="coerce").to_numpy(dtype=np.float32),
            0.0,
        )
        if not np.isfinite(stumpf_base_train).all():
            bad = int(stumpf_base_train.size - np.isfinite(stumpf_base_train).sum())
            raise ValueError(f"Stumpf baseline contains {bad} non-finite values after QC")
        y_train = np.clip(np.abs(y_train_raw) - stumpf_base_train, -residual_clip_m, residual_clip_m).astype(np.float32)
        metadata["residual_target_stats"] = {
            "min": float(np.nanmin(y_train)),
            "max": float(np.nanmax(y_train)),
            "p05": float(np.nanpercentile(y_train, 5)),
            "p95": float(np.nanpercentile(y_train, 95)),
        }
        log.info(
            "Final residual target after QC: n=%d residual_range=[%.2f, %.2f] m (clip=±%.2f)",
            len(y_train),
            float(np.nanmin(y_train)),
            float(np.nanmax(y_train)),
            residual_clip_m,
        )

    # Log training data composition by source (single pass)
    if 'source' in df_tr_fit.columns or 'source_norm' in df_tr_fit.columns:
        source_col = 'source_norm' if 'source_norm' in df_tr_fit.columns else 'source'
        source_counts = df_tr_fit[source_col].value_counts()
        total_samples = len(df_tr_fit)
        total_weighted = df_tr_fit['sample_weight'].sum() if w_train is not None else 0.0
        log.info("Training data composition (%d samples):", total_samples)
        eff_fracs = []
        for source in source_counts.index:
            count = int(source_counts[source])
            pct = 100.0 * count / total_samples
            if w_train is not None and total_weighted > 0:
                weight_contrib = float(df_tr_fit.loc[df_tr_fit[source_col] == source, 'sample_weight'].sum())
                eff_pct = 100.0 * weight_contrib / total_weighted
                eff_fracs.append(weight_contrib / total_weighted)
                log.info("  %-20s n=%6d (%5.1f%%)  weight_mean=%5.1f  training_influence=%5.1f%%",
                         source, count, pct, weight_contrib / count if count else 0, eff_pct)
            else:
                log.info("  %-20s n=%6d (%5.1f%%)", source, count, pct)
        try:
            if eff_fracs:
                metadata.setdefault('training_qc', {})
                metadata['training_qc']['max_effective_source_influence_frac'] = float(max(eff_fracs))
        except Exception as _exc:
            log.debug("Suppressed: %s", _exc, exc_info=True)

    rf.fit(X_train, y_train, sample_weight=w_train)
    log.info("RF trained on %s samples. (Test set: %s)", len(df_tr_fit), len(df_te))

    # --- Run Spatial Cross-Validation if enabled ---
    spatial_cv_summary = None
    if spatial_cv_enabled and len(df) >= 200:
        try:
            from spatial_cv import run_spatial_cv, plot_spatial_cv_results
            log.info("Running %s-fold spatial CV (%s)...", spatial_cv_folds, spatial_cv_strategy)
            
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
                "Spatial CV complete: RMSE=%3f±%3fm,  R²=%3f±%3f",
                spatial_cv_summary.rmse_mean, spatial_cv_summary.rmse_std, spatial_cv_summary.r2_mean, spatial_cv_summary.r2_std,
            )
        except ImportError:
            log.debug("spatial_cv module not available")
        except (TypeError, ValueError, KeyError, RuntimeError) as e:
            log.warning("Spatial CV failed: %s", e)
            try:
                from errors_scientific import record_fallback, FallbackClass
                record_fallback(fallback_registry, "spatial_cv_failed", FallbackClass.DEGRADED,
                                stage="training_validation", error=e,
                                detail="Spatial CV failed; random-split validation remains available.")
            except (ImportError, AttributeError):
                log.debug("Fallback registry unavailable for spatial CV failure", exc_info=True)

    # --- Training Data Diversity Analysis ---
    diversity_report = None
    try:
        from training_diversity import add_diversity_analysis_to_training
        aoi_bounds = None
        if isinstance(raster_paths, dict):
            aoi_bounds = raster_paths.get("aoi_bounds") or raster_paths.get("aoi")
        diversity_report = add_diversity_analysis_to_training(df, plots_dir, feat_cols, aoi_bounds=aoi_bounds)
        if diversity_report and "overall_score" in diversity_report:
            log.info("Data diversity score: %.0f%%", diversity_report['overall_score'] * 100)
            if "temporal_diversity_score" in diversity_report:
                log.info("Temporal diversity score: %.0f%%", 100.0 * float(diversity_report["temporal_diversity_score"]))
            if diversity_report.get("recommendations"):
                for rec in diversity_report["recommendations"][:4]:
                    log.info("→ %s", rec)
    except ImportError:
        log.debug("training_diversity module not available")
    except (TypeError, ValueError, KeyError, OSError) as e:
        log.warning("Diversity analysis failed: %s", e)
        try:
            from errors_scientific import record_fallback, FallbackClass
            record_fallback(fallback_registry, "training_diversity_failed", FallbackClass.SAFE,
                            stage="training_diagnostics", error=e,
                            detail="Training diversity diagnostics failed; model training continued.")
        except (ImportError, AttributeError):
            log.debug("Fallback registry unavailable for diversity failure", exc_info=True)

    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_feature_importance(rf, feat_cols, plots_dir / "Feature_Importance.png")

    if not df_te.empty:
        # Physics-guided prediction: Stumpf baseline plus bounded RF residual.
        depth_pred_mag = _predict_physics_guided_magnitude(rf, df_te, feat_cols, guidance_settings)
        df_te["depth_pred_m"] = -depth_pred_mag.astype(np.float32)
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

        # Source-specific validation
        if 'source' in df_te.columns or 'source_norm' in df_te.columns:
            source_col = 'source_norm' if 'source_norm' in df_te.columns else 'source'
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            source_validation = {}
            log.info("Validation by source:")
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
                r2_str = f"{r2:5.3f}" if np.isfinite(r2) else "  N/A"
                log.info("  %-20s n=%5d  RMSE=%5.2fm  MAE=%5.2fm  R²=%s",
                         source, n_source, rmse, mae, r2_str)
            # Save to metadata
            if rr is not None:
                try:
                    if hasattr(rr, "add"):
                        rr.add("validation.by_source", source_validation)
                    elif hasattr(rr, "data") and isinstance(rr.data, dict):
                        rr.data["validation.by_source"] = source_validation
                except Exception as _exc:
                    log.debug("Suppressed: %s", _exc, exc_info=True)

        max_depth_sdb_auto = None
        max_depth_diag = {}
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
                        y_true_tr=(df_tr["depth_m"].to_numpy() if (df_tr is not None and not df_tr.empty) else None),
                        bin_edges=max_depth_diag.get("bin_edges_m"),
                        out_png=plots_dir / "Depth_Binning_Sanity.png",
                        title=f"Depth binning sanity ({depth_binning})",
                        min_samples_per_bin_te=min_samples_per_bin,
                        strict_max=max_depth_sdb_auto,
                        relaxed_max=max_depth_sdb_auto_relaxed,
                    )

            except Exception:
                log.exception("Auto max-depth estimation failed")

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
                        "Physics-based depth: Kd(490)=%3f m⁻¹ (%s),  optical limit=%1fm",
                        kd_median, water_type, phys_max,
                    )
                    
                    # ALWAYS store physics-based max depth when computed
                    metadata["max_depth_sdb_auto_physics"] = float(phys_max)
                    metadata["kd_490_median"] = float(kd_median)
                    metadata["water_type"] = water_type
                    
                    if combined is not None:
                        log.info(
                            "Operational depth cap candidate (combined physics+rmse): %1fm  (limited by %s)",
                            combined, limiting,
                        )
                        metadata["max_depth_sdb_combined"] = float(combined)
                        metadata["max_depth_limiting_factor"] = limiting
                        
            except ImportError:
                log.debug("kd_estimation module not available, skipping physics-based depth")
            except Exception as e:
                log.warning("Physics-based depth estimation failed: %s", e)

        # --- Physics-based enhancements (Kim et al. 2024) ---
        # Estimate scene-specific bottom endmembers and geometry-corrected attenuation
        if PHYSICS_AVAILABLE and raster_paths is not None:
            try:
                # Get sun/view angles from S2 metadata
                s2_dir = Path(raster_paths.get("B02", "")).parent if raster_paths else None
                sza, vza = physics_integration.get_sun_view_angles_from_s2_metadata(
                    str(s2_dir) if s2_dir else None
                )
                
                log.info("Sun zenith: %.1f°, View zenith: %.1f°", sza, vza)
                
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
                        log.info("Geometry-corrected (mean over bands): Kd=%.4f, Ku=%.4f", kd_mean, ku_mean)
                    except Exception:
                        log.debug("Geometry correction summary failed", exc_info=True)
                
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
                            log.info("Seagrass signature detected (Green > Blue in shallow)")
                        else:
                            metadata["physics"]["seagrass_detected"] = False
                        
                        log.info("Bottom endmembers estimated: %s",
                                 list(endmember_result.get('endmembers', {}).keys()))
                    else:
                        log.debug("Endmember estimation skipped: %s", endmember_result.get('reason', 'unknown'))
                        
                except Exception as e:
                    log.debug("Endmember estimation failed: %s", e)
                    
            except Exception as e:
                log.warning("Physics integration failed: %s", e)
        elif not PHYSICS_AVAILABLE:
            log.debug("physics_integration module not available")

        # Resolve max_depth_sdb_final from available estimates per max_depth_source setting
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
                log.warning("Physics-based depth not available, falling back to training_p95")
                if training_p95 is not None:
                    final_depth = training_p95
                    final_depth_source = "training_p95_fallback"
                    
        elif max_depth_source == "rmse":
            if rmse_max is not None:
                final_depth = float(rmse_max)
                final_depth_source = "rmse_depth_of_support"
            else:
                log.warning("RMSE-based depth not available, falling back to training_p95")
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
        
        try:
            guidance_meta = metadata.get("physics_guidance", {}) if isinstance(metadata, dict) else {}
            final_family = _resolve_depth_source_family(final_depth_source or max_depth_source)
            if guidance_meta.get("low_support") and training_p95 is not None and final_family not in {"physics", "combined"}:
                conservative_cap = float(max(training_p95, 0.75))
                if np.isfinite(conservative_cap) and np.isfinite(float(final_depth)) and conservative_cap < float(final_depth):
                    log.info(
                        "Low-support guidance mode: tightening final depth cap from %.2fm to training_p95 %.2fm.",
                        float(final_depth),
                        conservative_cap,
                    )
                    final_depth = conservative_cap
                    final_depth_source = "training_p95_low_support_guardrail"
        except Exception as _exc:
            log.debug("Suppressed: %s", _exc, exc_info=True)

        metadata["max_depth_sdb_final"] = final_depth
        metadata["max_depth_sdb_final_source"] = final_depth_source
        log.info(
            "Canonical requested-source depth candidate (max_depth_sdb_final) =  %2fm (source: %s, requested: %s)",
            final_depth, final_depth_source, max_depth_source,
        )
        if combined_max is not None and physics_max is not None and str(max_depth_source).lower() == "physics":
            try:
                if np.isfinite(float(combined_max)) and np.isfinite(float(final_depth)) and float(combined_max) < float(final_depth):
                    log.info(
                        "Note: RMSE-constrained combined/operational cap is %2fm  (< physics candidate %2fm); downstream product code may enforce the stricter cap.",
                        float(combined_max), float(final_depth),
                    )
            except Exception as _exc:
                log.debug("Suppressed: %s", _exc, exc_info=True)

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
            
            rep["train"]["max_depth_sdb_final"] = metadata.get("max_depth_sdb_final")
            rep["train"]["max_depth_sdb_final_source"] = metadata.get("max_depth_sdb_final_source")
            rep["train"]["max_depth_sdb_auto_physics"] = metadata.get("max_depth_sdb_auto_physics")
            rep["train"]["max_depth_options"] = metadata.get("max_depth_options", {})
        except Exception as _exc:
            log.debug("Suppressed: %s", _exc, exc_info=True)
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
        except Exception as _exc:
            log.debug("Suppressed: %s", _exc, exc_info=True)

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
        except Exception as _exc:
            log.debug("Suppressed: %s", _exc, exc_info=True)

        try:
            imps = rf.feature_importances_
            pairs = list(zip(feat_cols, [float(x) for x in imps]))
            pairs.sort(key=lambda x: x[1], reverse=True)
            rep["train"]["feature_importance"] = [{"feature": k, "importance": v} for k, v in pairs[:15]]
        except Exception as _exc:
            log.debug("Suppressed: %s", _exc, exc_info=True)
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
        except Exception as _exc:
            log.debug("Suppressed: %s", _exc, exc_info=True)

        metadata["train_report"] = rep["train"]
        # Write to diagnostics dir if provided, otherwise fallback to plots_dir parent
        if diagnostics_dir is not None:
            report_dir = _Path(diagnostics_dir)
        else:
            report_dir = _Path(plots_dir).parent if plots_dir is not None else _Path(".")
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "train_report.json", "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(rep, f, indent=2)
    except Exception as _exc:
        log.debug("Suppressed: %s", _exc, exc_info=True)

    rf_out = PhysicsGuidedResidualModel(rf, feat_cols, guidance_settings)
    return rf_out, stumpf_lr, df_tr, df_te, metadata



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
        log.info("Spatial validation enabled")
    else:
        log.warning("Spatial validation disabled — using random split")

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
            log.warning("[S2-AB] Composite rasters not found/complete in: %s. Proceeding without raster sampling.", s2_d)
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
                log.info("[S2-AB] Chosen=%s (%s); RMSE best=%s comp=%s; coh best=%.3f comp=%.3f",
                         ab_summary['chosen'], ab_summary['reason'],
                         best_pack.get('rmse_m'), comp_pack.get('rmse_m'),
                         best_pack.get('coherence', 0.0), comp_pack.get('coherence', 0.0))

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
        with open(out_dir / "s2_ab_test.json", "w", encoding="utf-8") as f:
            json.dump(ab_summary, f, indent=2)

    joblib.dump(chosen["rf"], out_dir / "rf_model.pkl")
    if chosen["lr"]:
        joblib.dump(chosen["lr"], out_dir / "stumpf_lr.pkl")

    with open(out_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(chosen["meta"], f, indent=2)

    log.info("Model saved to %s (variant=%s)", out_dir, chosen.get('variant'))


if __name__ == "__main__":
    main()