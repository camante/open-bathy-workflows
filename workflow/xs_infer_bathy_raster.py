#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xs_infer_bathy_raster.py – Infer river bathymetry from cross-sections and rasterize a DEM-aligned bed patch

Infer approximate river bathymetry from cross-sections (XS) + bank elevations,
optionally calibrated by nearby measured depths (sonar / bathy lidar).

Also supports generating DEM-aligned rasters:
  - bed patch raster (elevation, meters)
  - patch mask raster (1 where patch exists, else 0)
  - uncertainty raster (meters)

NEW: continuous patch interpolation modes
----------------------------------------
--continuous {walid,aidw,median}

- median (legacy): median per template pixel from XS bathy points (discrete, may look stripy)
- aidw: adaptive inverse-distance weighting over a river corridor mask
- walid (default): thalweg-guided IDW intended to produce a smoother, continuous surface
    * uses all bathy points as "query points"
    * adds thalweg control points (deepest point per cross-section) with extra influence
    * interpolates onto a DEM grid inside a buffered corridor derived from the points

Raster-only mode
----------------
If you already have a bathy points layer (e.g., after monotonic adjustment), you can rasterize it:

python xs_infer_bathy_raster.py \
  --raster-from-gpkg river_xs_bathy_monotonic.gpkg \
  --raster-layer xs_bathy_points_monotonic \
  --raster-value-col z_bed_adj_m \
  --template-raster /path/to/cudem_dem.tif \
  --out-bathy-raster river_bed_patch.tif \
  --out-mask-raster river_bed_mask.tif \
  --continuous walid

Notes on CRS & nodata
---------------------
- Works with projected or geographic template rasters.
- If template CRS is geographic (degrees), interpolation distances are computed in a local UTM
  CRS (auto-estimated from data), while outputs remain in the template CRS.
- Nodata is propagated and treated consistently; default nodata=-9999.0 for float rasters.

Optional: SWOT water-surface elevation (WSE) stage anchoring
-----------------------------------------------------------
If you provide SWOT (or other) WSE point observations, the script can blend or replace the per-XS
DEM/topo-derived WSE proxy. This helps constrain bed elevations in backwater/tidal reaches where the
proxy stage can be biased.

Example:
  python xs_infer_bathy_raster.py \
    --xs-gpkg river_xs.gpkg \
    --out-gpkg river_xs_bathy.gpkg \
    --swot-wse swot_wse_points.gpkg \
    --swot-wse-col wse_m \
    --swot-stage-blend 1.0

Use --no-swot-use-for-slope if you want observed stage to shift bed elevations but NOT drive the
longitudinal WSE-profile slope fit.
"""


import argparse
import json as _json
import logging
import os
import time
import contextlib


# Logging is configured by entrypoints (e.g., bathy_main.py / sdb_main.py).
# Standalone scripts configure logging in __main__.

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import math

import numpy as np
import pandas as pd

# Ensure GeoPandas remains usable on pandas>=2.0 even if GeoPandas lags.
import compat_pandas  # noqa: F401

# Longitudinal WSE profile fitting (stabilizes slope for Manning / multivariate priors)
try:
    from river_wse import WSEFitConfig, fit_wse_profile
except Exception:  # pragma: no cover
    WSEFitConfig = None
    fit_wse_profile = None
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.features import rasterize
from shapely.geometry import Point, mapping
from shapely.ops import unary_union, linemerge
from pyproj import CRS, Transformer

log = logging.getLogger("xs_infer_bathy")


def _union_all(geoms):
    """Compatibility helper for Shapely 1.x vs 2.x.

    GeoPandas/Shapely 2 exposes `GeoSeries.union_all()`, while older stacks use `GeoSeries.unary_union`.
    """
    try:
        return geoms.union_all()
    except Exception:
        return geoms.unary_union
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

# Import centralized constants
try:
    from constants import (
        HYDRAULIC_GEOMETRY_A,
        HYDRAULIC_GEOMETRY_B,
        HYDRAULIC_MIN_DEPTH_M,
        HYDRAULIC_MAX_DEPTH_M,
        TRAPEZOID_BOTTOM_FRAC,
        NODATA_DEPTH,
    )
except ImportError:
    # Fallback values with documentation
    log.warning("[xs_infer] constants module not found, using local defaults")

    # Leopold & Maddock (1953) coefficients
    HYDRAULIC_GEOMETRY_A = 0.18
    HYDRAULIC_GEOMETRY_B = 0.50
    HYDRAULIC_MIN_DEPTH_M = 0.50
    HYDRAULIC_MAX_DEPTH_M = 15.0
    TRAPEZOID_BOTTOM_FRAC = 0.30
    NODATA_DEPTH = -9999.0



# Optional module: Manning inversion utilities (added in v0.7.8+)
try:
    from manning_inversion import invert_manning_for_depth, estimate_q2_from_drainage_area
except Exception:
    invert_manning_for_depth = None  # type: ignore
    estimate_q2_from_drainage_area = None  # type: ignore

@dataclass
class InferConfig:
    """
    Configuration for river bathymetry inference.

    Key concepts:
    - A *prior* predicts maximum depth (Dmax) from bank-to-bank width (W) and (optionally)
      additional reach attributes (slope, drainage area).
    - One or more *calibration anchors* can override/refine the prior:
        1) measured soundings near the cross-section
        2) width–stage inversion (at-a-station) if provided
        3) USGS discharge *measurement* records (width+area -> mean depth)
    """

    # bed shape (used to distribute Dmax across the XS)
    bottom_width_frac: float = TRAPEZOID_BOTTOM_FRAC
    xs_profile_shape: str = "cosine_trapezoid"  # linear_trapezoid | cosine_trapezoid

    # WSE estimation (proxy from DEM/topo profile)
    wse_center_frac: float = 0.10
    wse_quantile: float = 0.10
    wse_fallback_drop_m: float = 0.50


    # Optional SWOT (or other) water-surface elevation (stage) observations
    # If provided, these can be used to anchor/blend the per-XS WSE proxy.
    swot_wse: Optional[Path] = None
    swot_wse_col: str = "wse_m"
    swot_x_col: Optional[str] = None  # for CSV inputs
    swot_y_col: Optional[str] = None  # for CSV inputs
    swot_csv_crs: str = "EPSG:4326"  # CRS for CSV lon/lat or x/y columns
    swot_max_dist_m: float = 1000.0
    swot_stage_blend: float = 1.0  # 0 => ignore obs; 1 => replace proxy when available
    swot_vertical_offset_m: float = 0.0
    swot_outlier_mad_z: float = 4.0
    swot_use_for_slope: bool = True  # if True, blended WSE drives WSE-profile slope fit

    # Curvature-driven cross-section asymmetry (optional)
    # Computes a signed curvature proxy from the sequence of XS center points (per component),
    # then shifts the trapezoid "flat bottom" toward the outer bend. This approximates
    # thalweg skew using only planform geometry (a conservative proxy; not a sediment model).
    curv_asymmetry_enabled: bool = True
    curv_window_m: float = 500.0          # half-window for local polynomial fit (meters)
    curv_min_points: int = 7              # minimum XS points per fit window
    curv_kappa_scale_1pm: float = 0.002   # curvature scale (1/m) for tanh mapping (legacy)
    curv_use_dimensionless: bool = True   # if True, use κ* = κ×W (dimensionless) instead of κ/kscale
    curv_kappa_star_scale: float = 1.0    # multiplier on κ* before tanh; tune per river type
    curv_max_offset_frac: float = 0.25    # maximum shift fraction of width (0..0.45)
    curv_lag_m: float = 0.0               # evaluate curvature at s+lag (meters), optional

    # Prior (base width->depth power law; Leopold & Maddock 1953 as default)
    a: float = HYDRAULIC_GEOMETRY_A
    b: float = HYDRAULIC_GEOMETRY_B
    dmin_m: float = HYDRAULIC_MIN_DEPTH_M
    dmax_m: float = HYDRAULIC_MAX_DEPTH_M

    # Prior mode upgrades
    prior_mode: str = "powerlaw"  # powerlaw | multivariate

    # Multivariate prior parameters (only used when prior_mode='multivariate')
    # Dmax = mv_a0 * W^mv_bw * (A_drain + mv_eps_a)^mv_ba * (S + mv_eps_s)^mv_bs
    mv_a0: float = HYDRAULIC_GEOMETRY_A
    mv_bw: float = HYDRAULIC_GEOMETRY_B
    mv_ba: float = 0.0
    mv_bs: float = -0.10
    mv_eps_a: float = 1.0
    mv_eps_s: float = 1e-4

    # Slope proxy stabilization (used when slope attribute is missing)
    slope_proxy_window: int = 9              # rolling median window on WSE along reach
    slope_min: float = 1e-5                  # minimum plausible slope (m/m)
    slope_max: float = 0.05                  # maximum plausible slope (m/m)
    slope_proxy_min_n: int = 7               # minimum XS per river_id to compute slope proxy

    # Longitudinal WSE profile fitting (recommended). When enabled, we compute
    # a stabilized WSE curve per river_id and derive slope from that curve.
    # This is more robust than slope_proxy_mpm derived directly from noisy WSE.
    wse_profile_enabled: bool = True
    wse_profile_window: int = 9
    wse_profile_min_n: int = 7
    wse_profile_monotonic: bool = True

    # Optional: reach-scale 1D energy-consistent depth solver (flag-controlled)
    energy_solver_enabled: bool = False
    energy_solver_only_when_no_soundings: bool = True
    energy_max_weight: float = 0.25     # max weight for 1-D energy solver depths
    energy_min_confidence: float = 0.30 # min solver confidence to apply weight
    force_slope_proxy: bool = False      # override reach slope with WSE proxy even when NHD slope available
    # Safety gate: if the only WSE anchor is a DEM/topo proxy (no observed stage),
    # skip the energy solver by default to avoid creating a false sense of
    # observational hydraulic constraint. Can be overridden explicitly.
    energy_allow_dem_proxy_wse: bool = False
    out_1d_solver_inputs_json: str | None = None
    out_1d_solver_outputs_json: str | None = None
    out_1d_solver_accounting_json: str | None = None

    # smoothing
    smooth_window: int = 7

    # Calibration from local soundings
    calib_max_dist_m: float = 200.0
    calib_stat: str = "p90"

    # Soundings loading guardrails (prevents OOM when extra_xyz is huge)
    soundings_max_points: int = 2_000_000
    soundings_sample_seed: int = 0

    # Optional: write the (possibly downsampled) soundings set to a single file so downstream
    # river steps can reuse the exact same subset (reproducibility + seam consistency).
    # Intended to be passed from bathy_main.py.
    write_soundings_subset: Optional[str] = None
    only_write_soundings_subset: bool = False

    # Pass 2 explicit soundings handoff: when provided, this pre-clipped parquet is loaded
    # INSTEAD of the raw --soundings files. Hard-fail if missing or empty.
    # Set by bathy_main.py after Pass 1 validates the subset row-count > 0.
    soundings_subset: Optional[str] = None

    # Tier-1 calibration anchors (no bed data required)
    usgs_max_dist_m: float = 5000.0          # meters (distance from gage to XS center)
    usgs_mean_to_dmax: float = 1.30          # convert mean depth -> approximate Dmax
    usgs_a_stat: str = "median"              # median | p90 | mean for a_site aggregation
    usgs_q_quantile_lo: float = 0.20          # optional flow-window filter lower quantile
    usgs_q_quantile_hi: float = 0.80          # optional flow-window filter upper quantile
    usgs_a_cv_warn: float = 0.50              # warn if a-values CV exceeds this
    usgs_width_ratio_max: float = 3.0         # skip/blend if bank width / meas width exceeds this
    usgs_width_ratio_blend: bool = True       # blend toward prior when width ratio is large
    gage_snap_max_dist_m: float = 1000.0      # meters (snap gages to network for river_id assignment)


    width_stage_max_dist_m: float = 5000.0   # meters (distance from station to XS center)
    width_stage_min_n: int = 6               # minimum observations to fit width–stage slope
    width_stage_min_r2: float = 0.25          # minimum R^2 for width-stage fit to be trusted
    width_stage_max_weight: float = 0.8       # maximum blend weight for width-stage anchor

    # Optional Manning inversion prior (secondary, blended)
    # Estimate mean depth y from Manning (wide-channel approx) and convert to Dmax using trapezoid factor.
    manning_mode: str = "off"              # off | constant | from_field | q2_regional
    manning_q_cms: Optional[float] = None  # discharge (m^3/s) when mode='constant'
    manning_q_field: Optional[str] = None  # reach attribute field containing Q (m^3/s) when mode='from_field'
    manning_n: float = 0.035
    manning_region: str = "default"
    manning_n_by_region: dict = None  # optional mapping {region: n}; supplied via --manning-n-by-region
    manning_min_confidence: float = 0.30
    manning_max_weight: float = 0.60
    manning_backwater_slope_thresh: float = 1e-4  # disable when slope below this (backwater/tidal risk)
    manning_dist_to_mouth_field: Optional[str] = None
    manning_dist_to_mouth_km_max: float = 10.0

    # If the run has no soundings, optionally auto-enable a conservative Manning prior
    # (requires slope + drainage area). This provides a more stable absolute depth scale
    # in data-sparse reaches.
    auto_manning_when_no_soundings: bool = True



    # Optional Regional Hydraulic Geometry Curves (Drainage Area -> Bankfull Depth)
    # This is *not* a bed observation; use as a soft prior. Coefficients are region-specific.
    # Depth form: D_bkf = c * DA^f
    # DA units controlled by regional_curve_da_units ('km2' or 'mi2').
    # If depth represents mean depth, convert to Dmax using trapezoid factor unless overridden.
    regional_curve_enabled: bool = False
    regional_curve_region: Optional[str] = None  # must be explicitly set to use regional curves
    # Safety valve: allow using built-in regional curve tables (if present) even when
    # regional_curve_region is not explicitly provided. Default is False for reproducibility.
    allow_builtin_regional_curves: bool = False
    regional_curve_c: Optional[float] = None    # override c
    regional_curve_f: Optional[float] = None    # override f
    regional_curve_unc_pct: float = 40.0        # 1-sigma-ish percent uncertainty (used for weighting)
    regional_curve_da_field: Optional[str] = None  # if set, use this drainage area field name
    regional_curve_da_units: str = "km2"        # km2 | mi2
    regional_curve_depth_units: str = "m"      # m | ft (coefficients units)
    regional_curve_depth_type: str = "mean"     # mean | max
    regional_curve_to_dmax: str = "auto"        # auto | factor
    regional_curve_to_dmax_factor: float = 1.25 # used when to_dmax='factor'

    # ---- Geomorphic depth envelope (stabilizes absolute scale in no-sounding areas) ----
    # Uses the regional curve (DA -> bankfull depth) to compute a conservative upper bound on Dmax.
    # This is a *cap* applied after anchors/priors: dmax_raw_m = min(dmax_raw_m, dmax_env_m).
    geomorphic_envelope_enabled: bool = True
    geomorphic_envelope_region: str = "auto"   # auto -> use regional_curve_region
    geomorphic_envelope_inflate_unc: bool = True  # if True, multiply by (1 + regional_curve_unc_pct/100)
    geomorphic_envelope_only_when_no_soundings: bool = True

    regional_curve_max_weight: float = 0.60
    regional_curve_min_da_km2: float = 1.0      # ignore very small DA (unstable curves)

    # control
    only_with_banks: bool = True


# --------------------------------------------------------------------------------------
# Helpers

# --------------------------------------------------------------------------------------

def _read_layer(gpkg: Path, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg, layer=layer)
    if gdf.empty:
        raise RuntimeError(f"Layer '{layer}' is empty in {gpkg}")
    if gdf.crs is None:
        raise RuntimeError(f"Layer '{layer}' has no CRS: {gpkg}")
    return gdf


def _read_layer_with_fallback(gpkg: Path, preferred_layer: str, purpose: str = "rivers") -> Tuple[gpd.GeoDataFrame, str]:
    """Read a layer from a GeoPackage, with deterministic fallback.

    Why: different river-network builders may write different layer names.
    We try the requested layer name first; if missing, we scan layers and pick
    the first plausible candidate.
    """
    try:
        gdf = gpd.read_file(gpkg, layer=preferred_layer)
        if gdf is not None and len(gdf) > 0:
            if gdf.crs is None:
                raise RuntimeError(f"Layer '{preferred_layer}' has no CRS: {gpkg}")
            return gdf, preferred_layer
    except Exception:
        pass

    try:
        import fiona

        layers = list(fiona.listlayers(gpkg))
    except Exception as e:
        raise RuntimeError(f"Failed to list layers in {gpkg} ({e})")

    # Deterministic candidate ordering: prefer explicit names, then substring matches.
    preferred = [
        preferred_layer,
        "rivers_clip",
        "rivers",
        "river",
        "flowlines",
        "flowline",
        "nhd_flowline",
        "nhdflowline",
        "network",
    ]

    ordered = []
    seen = set()
    for name in preferred + layers:
        if name in seen:
            continue
        if name in layers:
            ordered.append(name)
            seen.add(name)

    last_err = None
    for lyr in ordered:
        try:
            gdf = gpd.read_file(gpkg, layer=lyr)
            if gdf is None or len(gdf) == 0:
                continue
            if gdf.crs is None:
                continue
            return gdf, lyr
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"No usable '{purpose}' layer found in {gpkg} (tried {len(ordered)} candidates; last_err={last_err})")


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _rolling_smooth(series: pd.Series, window: int) -> pd.Series:
    w = int(max(1, window))
    if w % 2 == 0:
        w += 1
    s_med = series.rolling(window=w, center=True, min_periods=max(1, w // 3)).median()
    s_mean = s_med.rolling(window=w, center=True, min_periods=max(1, w // 3)).mean()
    return s_mean


def _default_uncertainty_m(calib_n: int) -> float:
    return 0.5 if calib_n and calib_n > 0 else 1.5


def _pick_bank_dists_from_points(xsp: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    left = None
    right = None
    if "is_bank_left" in xsp.columns and xsp["is_bank_left"].any():
        left = float(xsp.loc[xsp["is_bank_left"] == True, "dist_m"].iloc[0])
    if "is_bank_right" in xsp.columns and xsp["is_bank_right"].any():
        right = float(xsp.loc[xsp["is_bank_right"] == True, "dist_m"].iloc[0])
    return left, right


def _estimate_wse_from_profile(
    xsp: pd.DataFrame,
    xs_len_m: float,
    bank_left_z: float,
    bank_right_z: float,
    cfg: InferConfig,
) -> float:
    if xsp.empty or not np.isfinite(xs_len_m) or xs_len_m <= 0:
        return float("nan")

    center = xs_len_m / 2.0
    halfw = max(1.0, cfg.wse_center_frac * xs_len_m / 2.0)
    m = (xsp["dist_m"] >= (center - halfw)) & (xsp["dist_m"] <= (center + halfw))

    vals = xsp.loc[m, "z_dem"].to_numpy(dtype="float64") if "z_dem" in xsp.columns else np.array([], dtype="float64")
    vals = vals[np.isfinite(vals)]
    if vals.size >= 3:
        return float(np.quantile(vals, cfg.wse_quantile))

    vals2 = xsp.loc[m, "z_topo"].to_numpy(dtype="float64") if "z_topo" in xsp.columns else np.array([], dtype="float64")
    vals2 = vals2[np.isfinite(vals2)]
    if vals2.size >= 3:
        return float(np.quantile(vals2, cfg.wse_quantile))

    banks = [bank_left_z, bank_right_z]
    banks = [b for b in banks if np.isfinite(b)]
    if banks:
        return float(min(banks) - cfg.wse_fallback_drop_m)

    return float("nan")



def _trapezoid_depth_profile(
    dist_from_left: np.ndarray,
    W: float,
    Dmax: float,
    bottom_frac: float,
    offset_frac: float = 0.0,
) -> np.ndarray:
    """Trapezoidal depth profile across a cross-section.

    Parameters
    ----------
    dist_from_left:
        Distance from the *left* bank pick (meters), same convention as xs_points.dist_m.
    W:
        Bank-to-bank width (meters).
    Dmax:
        Maximum depth at the thalweg/flat-bottom (meters).
    bottom_frac:
        Flat-bottom width fraction (0..0.95) of W.
    offset_frac:
        Optional shift of the flat-bottom center as a fraction of width W.
        Positive shifts deeper region toward the *right* bank (larger dist_from_left).
        Negative shifts toward the *left* bank.

    Notes
    -----
    For symmetric profiles, offset_frac=0.
    """
    out = np.full_like(dist_from_left, np.nan, dtype="float64")
    if not np.isfinite(W) or W <= 0 or not np.isfinite(Dmax) or Dmax <= 0:
        return out

    Wb = float(np.clip(bottom_frac, 0.0, 0.95) * W)
    Wb = max(Wb, 1e-6)

    # Clamp the flat-bottom center so the plateau stays within [0, W]
    # Clamp based on geometry so the flat-bottom section stays within the channel.
    # Leave a small margin (5%) so that the trapezoid does not "flip" at high offsets.
    max_off = (1.0 - float(bottom_frac)) / 2.0 - 0.05
    max_off = float(np.clip(max_off, 0.0, 0.45))
    off = float(np.clip(offset_frac, -max_off, max_off))
    center = (W / 2.0) + (off * W)
    center = float(np.clip(center, Wb / 2.0, W - (Wb / 2.0)))

    left_edge = center - (Wb / 2.0)
    right_edge = center + (Wb / 2.0)

    # Side-slope run lengths (can be asymmetric)
    left_run = max(left_edge, 1e-6)
    right_run = max(W - right_edge, 1e-6)

    d = dist_from_left
    inside = (d >= 0.0) & (d <= W)

    # Left slope: 0 -> Dmax from x=0 to x=left_edge
    left = inside & (d < left_edge)
    out[left] = (d[left] / left_run) * Dmax

    # Flat bottom
    mid = inside & (d >= left_edge) & (d <= right_edge)
    out[mid] = Dmax

    # Right slope: Dmax -> 0 from x=right_edge to x=W
    right = inside & (d > right_edge)
    out[right] = ((W - d[right]) / right_run) * Dmax

    out[inside] = np.clip(out[inside], 0.0, Dmax)
    return out


def _cosine_trapezoid_depth_profile(
    dist_from_left: np.ndarray,
    W: float,
    Dmax: float,
    bottom_frac: float,
    offset_frac: float = 0.0,
) -> np.ndarray:
    """Trapezoidal profile with *smooth* cosine side slopes.

    This is a small but high-impact physical improvement over the piecewise-linear
    trapezoid: banks tend to have low curvature at the edges, and the cosine ramp
    avoids sharp slope discontinuities that can amplify interpolation artifacts.

    The profile is still strictly width-constrained (0 depth at banks, max depth
    limited to Dmax) and still supports thalweg asymmetry via offset_frac.
    """
    out = np.full_like(dist_from_left, np.nan, dtype="float64")
    if not np.isfinite(W) or W <= 0 or not np.isfinite(Dmax) or Dmax <= 0:
        return out

    Wb = float(np.clip(bottom_frac, 0.0, 0.95) * W)
    Wb = max(Wb, 1e-6)

    max_off = (1.0 - float(bottom_frac)) / 2.0 - 0.05
    max_off = float(np.clip(max_off, 0.0, 0.45))
    off = float(np.clip(offset_frac, -max_off, max_off))
    center = (W / 2.0) + (off * W)
    center = float(np.clip(center, Wb / 2.0, W - (Wb / 2.0)))

    left_edge = center - (Wb / 2.0)
    right_edge = center + (Wb / 2.0)

    left_run = max(left_edge, 1e-6)
    right_run = max(W - right_edge, 1e-6)

    d = dist_from_left
    inside = (d >= 0.0) & (d <= W)

    # Left ramp: 0 -> Dmax with zero slope at both ends
    left = inside & (d < left_edge)
    if left.any():
        x = np.clip(d[left] / left_run, 0.0, 1.0)
        out[left] = 0.5 * (1.0 - np.cos(np.pi * x)) * Dmax

    # Flat bottom
    mid = inside & (d >= left_edge) & (d <= right_edge)
    out[mid] = Dmax

    # Right ramp: Dmax -> 0 with zero slope at both ends
    right = inside & (d > right_edge)
    if right.any():
        x = np.clip((W - d[right]) / right_run, 0.0, 1.0)
        out[right] = 0.5 * (1.0 - np.cos(np.pi * x)) * Dmax

    out[inside] = np.clip(out[inside], 0.0, Dmax)
    return out


def _local_quad_derivatives(
    s: np.ndarray,
    v: np.ndarray,
    i: int,
    half_window_m: float,
    min_pts: int,
) -> Tuple[float, float]:
    """Estimate first and second derivatives v'(s), v''(s) at index i via local quadratic fit."""
    n = int(len(s))
    if n < max(3, int(min_pts)):
        return (float("nan"), float("nan"))

    s0 = float(s[i])
    mask = np.isfinite(s) & np.isfinite(v)
    if not mask.any():
        return (float("nan"), float("nan"))

    # Candidate indices within window
    idx = np.where(mask & (np.abs(s - s0) <= float(half_window_m)))[0]
    if idx.size < int(min_pts):
        # Fall back to nearest points
        good = np.where(mask)[0]
        if good.size < int(min_pts):
            return (float("nan"), float("nan"))
        # Sort by |s-s0| and take min_pts
        order = np.argsort(np.abs(s[good] - s0))
        idx = good[order[: int(min_pts)]]

    ss = (s[idx] - s0).astype("float64")
    vv = v[idx].astype("float64")

    A = np.vstack([np.ones_like(ss), ss, ss ** 2]).T
    try:
        coef, *_ = np.linalg.lstsq(A, vv, rcond=None)
    except Exception:
        return (float("nan"), float("nan"))

    d1 = float(coef[1])
    d2 = float(2.0 * coef[2])
    return (d1, d2)


def _compute_signed_curvature_1pm(
    s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    half_window_m: float,
    min_pts: int,
) -> np.ndarray:
    """Compute signed planform curvature (1/m) from a sequence of (x(s), y(s)) points.

    Uses a Savitzky–Golay-like local quadratic fit per point to estimate derivatives.
    Positive curvature indicates a left-turning trajectory in (x,y) coordinates.
    """
    n = int(len(s))
    kappa = np.full((n,), np.nan, dtype="float64")
    if n < max(5, int(min_pts)):
        return kappa

    # Require monotonic s (if not, re-sort by s)
    order = np.argsort(s)
    s = s[order].astype("float64")
    x = x[order].astype("float64")
    y = y[order].astype("float64")

    for ii in range(n):
        dx, ddx = _local_quad_derivatives(s, x, ii, half_window_m=half_window_m, min_pts=min_pts)
        dy, ddy = _local_quad_derivatives(s, y, ii, half_window_m=half_window_m, min_pts=min_pts)
        if not (np.isfinite(dx) and np.isfinite(dy) and np.isfinite(ddx) and np.isfinite(ddy)):
            continue
        denom = (dx * dx + dy * dy) ** 1.5
        if not np.isfinite(denom) or denom <= 0:
            continue
        kappa[ii] = (dx * ddy - dy * ddx) / denom

    # Reorder back to original
    out = np.full((n,), np.nan, dtype="float64")
    out[order] = kappa
    return out


def _attach_curvature_asymmetry(
    xs_lines: gpd.GeoDataFrame,
    xs_param: pd.DataFrame,
    cfg: InferConfig,
) -> pd.DataFrame:
    """Attach curvature proxy and thalweg offset fraction to xs_param.

    Curvature is estimated per `component_id` from the sequence of cross-section center points,
    ordered by `s_center_m` (if available). We map curvature to a signed thalweg offset fraction,
    shifting the flat-bottom portion of the trapezoid toward the outer bend.

    By default we use a dimensionless curvature κ* = κ×W (a common non-dimensionalization):

        offset_frac = tanh(κ* * curv_kappa_star_scale) * curv_max_offset_frac

    If `curv_use_dimensionless` is False, we fall back to the legacy scaling:

        offset_frac = tanh(κ / curv_kappa_scale_1pm) * curv_max_offset_frac

    Note:
        This is a conservative geometric proxy intended to reduce symmetric-channel artifacts.
        It is not a substitute for hydraulic/sediment-transport modeling.
    """
    xs_param = xs_param.copy()
    xs_param["curv_kappa_1pm"] = np.nan
    xs_param["thalweg_offset_frac"] = 0.0
    xs_param["thalweg_offset_m"] = 0.0
    xs_param["thalweg_side"] = "center"

    if not cfg.curv_asymmetry_enabled:
        return xs_param

    if xs_lines is None or xs_lines.empty:
        return xs_param

    # Build XS center points
    centers = xs_lines[["xs_id", "geometry"]].copy()
    centers["xs_id"] = centers["xs_id"].astype(str)
    try:
        centers["geometry"] = centers.geometry.interpolate(0.5, normalized=True)
    except Exception:
        centers["geometry"] = centers.geometry.centroid
    centers = gpd.GeoDataFrame(centers, geometry="geometry", crs=xs_lines.crs)

    # Merge attributes needed for grouping and ordering
    tmp = xs_param[["xs_id", "component_id", "s_center_m", "width_m"]].copy()
    tmp["xs_id"] = tmp["xs_id"].astype(str)
    g = centers.merge(tmp, on="xs_id", how="inner")
    if g.empty:
        return xs_param

    # Project to a metric CRS for curvature math
    crs_g = CRS.from_user_input(g.crs)
    if not crs_g.is_projected:
        try:
            u = unary_union(list(g.geometry))
            c = u.centroid
            utm = _utm_crs_from_lonlat(float(c.x), float(c.y))
            g_m = g.to_crs(utm)
        except Exception:
            g_m = g
    else:
        g_m = g

    # If s_center_m is missing, build a pseudo-station by cumulative distance
    g_m["_s_m"] = pd.to_numeric(g_m["s_center_m"], errors="coerce")
    need = ~np.isfinite(g_m["_s_m"])
    if need.any():
        g_m = g_m.sort_values(["component_id", "xs_id"]).copy()
        # Compute cumulative distance within each component
        svals = []
        for cid, gg in g_m.groupby("component_id", dropna=False):
            pts = np.vstack([gg.geometry.x.to_numpy(dtype="float64"), gg.geometry.y.to_numpy(dtype="float64")]).T
            if pts.shape[0] < 2:
                s = np.full((pts.shape[0],), np.nan)
            else:
                d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
                s = np.concatenate([[0.0], np.cumsum(d)])
            svals.append(pd.Series(s, index=gg.index))
        g_m["_s_m"] = pd.concat(svals).sort_index()

    half_w = float(max(1.0, cfg.curv_window_m)) / 2.0
    min_pts = int(max(5, cfg.curv_min_points))
    kscale = float(max(1e-9, cfg.curv_kappa_scale_1pm))
    max_off = float(np.clip(cfg.curv_max_offset_frac, 0.0, 0.45))
    lag_m = cfg.curv_lag_m

    out_rows = []
    for cid, gg in g_m.groupby("component_id", dropna=False):
        gg = gg.sort_values("_s_m").copy()
        s = gg["_s_m"].to_numpy(dtype="float64")
        x = gg.geometry.x.to_numpy(dtype="float64")
        y = gg.geometry.y.to_numpy(dtype="float64")
        if len(s) < min_pts:
            continue
        kappa = _compute_signed_curvature_1pm(s, x, y, half_window_m=half_w, min_pts=min_pts)

        if np.isfinite(lag_m) and abs(lag_m) > 0 and np.isfinite(s).any():
            # Evaluate curvature at s+lag via linear interpolation
            kappa = np.interp(s + lag_m, s, kappa, left=np.nan, right=np.nan)

        # Map curvature to offset fraction
        use_dim = cfg.curv_use_dimensionless
        if use_dim:
            # κ* = κ×W (dimensionless); adapt to local width
            scale_star = cfg.curv_kappa_star_scale
            w = pd.to_numeric(gg.get("width_m"), errors="coerce").to_numpy(dtype=float)
            if np.isfinite(w).any():
                w_fill = float(np.nanmedian(w))
            else:
                w_fill = 0.0
            w = np.where(np.isfinite(w), w, w_fill)
            kappa_scaled = kappa * w * scale_star
        else:
            # Legacy: κ/kscale (kscale in 1/m)
            kappa_scaled = kappa / max(kscale, 1e-12)

        off = np.tanh(kappa_scaled) * max_off
        off = np.where(np.isfinite(off), off, 0.0)

        # Determine side label for QA/debug
        side = np.where(off > 1e-6, "right", np.where(off < -1e-6, "left", "center"))

        out_rows.append(
            pd.DataFrame(
                dict(
                    xs_id=gg["xs_id"].astype(str).to_numpy(),
                    curv_kappa_1pm=kappa,
                    thalweg_offset_frac=off,
                    thalweg_offset_m=off * pd.to_numeric(gg["width_m"], errors="coerce").to_numpy(dtype="float64"),
                    thalweg_side=side,
                )
            )
        )

    if out_rows:
        out_df = pd.concat(out_rows, ignore_index=True)
        xs_param = xs_param.merge(out_df, on="xs_id", how="left", suffixes=("", "_curv"))
        # Fill NaNs from merge
        xs_param["curv_kappa_1pm"] = pd.to_numeric(xs_param["curv_kappa_1pm"], errors="coerce")
        xs_param["thalweg_offset_frac"] = pd.to_numeric(xs_param["thalweg_offset_frac"], errors="coerce").fillna(0.0)
        xs_param["thalweg_offset_m"] = pd.to_numeric(xs_param["thalweg_offset_m"], errors="coerce").fillna(0.0)
        xs_param["thalweg_side"] = xs_param["thalweg_side"].fillna("center")
    return xs_param


def _compute_dmax_prior(W: float, cfg: InferConfig, acct: Optional[dict] = None) -> float:
    """Univariate Dmax prior (width-only).

    If acct is provided, updates:
      - n_prior_total
      - n_prior_clipped_min
      - n_prior_clipped_max
    """
    if not np.isfinite(W) or W <= 0:
        return float("nan")
    D0 = float(cfg.a) * (float(W) ** float(cfg.b))
    D = float(np.clip(D0, cfg.dmin_m, cfg.dmax_m))
    if acct is not None:
        acct["n_prior_total"] = int(acct.get("n_prior_total", 0)) + 1
        if np.isfinite(D0):
            if float(D0) < float(cfg.dmin_m):
                acct["n_prior_clipped_min"] = int(acct.get("n_prior_clipped_min", 0)) + 1
            elif float(D0) > float(cfg.dmax_m):
                acct["n_prior_clipped_max"] = int(acct.get("n_prior_clipped_max", 0)) + 1
    return D
def _guess_field(columns, candidates):
    """Return the first candidate present in columns (case-insensitive), else None."""
    if columns is None:
        return None
    try:
        if len(columns) == 0:
            return None
    except Exception:
        return None
    lower = {str(c).lower(): str(c) for c in list(columns)}
    for c in candidates:
        key = str(c).lower()
        if key in lower:
            return lower[key]
    return None



def _compute_slope_proxy(
    xs_param: pd.DataFrame,
    window: int = 9,
    slope_min: float = 1e-5,
    slope_max: float = 0.05,
    min_n: int = 7,
) -> pd.Series:
    """Estimate a stabilized water-surface slope proxy per XS (m/m).

    This is only a fallback when no reach slope attribute exists. It can be noisy because
    WSE is estimated from a DEM-based proxy.

    Stabilizations applied:
    - Require at least `min_n` XS within a river_id group
    - Rolling-median smoothing of WSE along stationing before differencing
    - Central-difference slope with clipping to [slope_min, slope_max]
    """
    if xs_param is None or xs_param.empty:
        return pd.Series([], dtype="float64")

    out = pd.Series(np.nan, index=xs_param.index, dtype="float64")
    req = {"river_id", "s_center_m", "wse_m"}
    if not req.issubset(set(xs_param.columns)):
        return out

    window = int(max(3, window))
    if window % 2 == 0:
        window += 1

    for _, g in xs_param.groupby("river_id", dropna=False):
        gg = g.copy()
        gg["s_center_m"] = pd.to_numeric(gg["s_center_m"], errors="coerce")
        gg["wse_m"] = pd.to_numeric(gg["wse_m"], errors="coerce")
        gg = gg.dropna(subset=["s_center_m", "wse_m"]).sort_values("s_center_m")
        if len(gg) < int(min_n):
            continue

        # Smooth WSE to suppress DEM/edge artifacts
        wse_smooth = gg["wse_m"].rolling(window=window, center=True, min_periods=max(3, window // 2)).median()
        wse = wse_smooth.to_numpy()
        s = gg["s_center_m"].to_numpy()

        # Central differences
        ds = s[2:] - s[:-2]
        dw = wse[2:] - wse[:-2]
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.abs(dw / ds)

        slope = np.concatenate([[np.nan], slope, [np.nan]])
        slope = pd.Series(slope, index=gg.index).astype("float64")

        # Clip and invalidate obviously bad results
        slope = slope.where(np.isfinite(slope))
        slope = slope.clip(lower=float(slope_min), upper=float(slope_max))
        out.loc[gg.index] = slope

    return out



def _compute_dmax_prior_multivariate(row: pd.Series, cfg: InferConfig, acct: Optional[dict] = None) -> float:
    """Multivariate Dmax prior.

    Dmax = mv_a0 * W^mv_bw * (A_drain + mv_eps_a)^mv_ba * (S + mv_eps_s)^mv_bs
    """
    W = float(row.get("width_m", np.nan))
    if not np.isfinite(W) or W <= 0:
        return float("nan")
    A = float(row.get("drain_area_km2", np.nan))
    S = float(row.get("slope_mpm", np.nan))
    if not np.isfinite(A):
        A = 0.0
    if not np.isfinite(S):
        S = 0.0

    D0 = (
        float(cfg.mv_a0)
        * (W ** float(cfg.mv_bw))
        * ((A + float(cfg.mv_eps_a)) ** float(cfg.mv_ba))
        * ((S + float(cfg.mv_eps_s)) ** float(cfg.mv_bs))
    )
    D = float(np.clip(D0, cfg.dmin_m, cfg.dmax_m))
    if acct is not None:
        acct["n_prior_total"] = int(acct.get("n_prior_total", 0)) + 1
        if np.isfinite(D0):
            if float(D0) < float(cfg.dmin_m):
                acct["n_prior_clipped_min"] = int(acct.get("n_prior_clipped_min", 0)) + 1
            elif float(D0) > float(cfg.dmax_m):
                acct["n_prior_clipped_max"] = int(acct.get("n_prior_clipped_max", 0)) + 1

        # Attribute availability accounting (helps diagnose under-constraint)
        if np.isfinite(float(row.get("drain_area_km2", np.nan) or np.nan)):
            acct["n_with_da"] = int(acct.get("n_with_da", 0)) + 1
        if np.isfinite(float(row.get("slope_mpm", np.nan) or np.nan)):
            acct["n_with_slope"] = int(acct.get("n_with_slope", 0)) + 1
    return D


def _manning_n_effective(cfg: InferConfig) -> float:
    """Return Manning's n used for inversion, optionally keyed by region.

    This avoids hardcoded region defaults (user supplies mapping) while still allowing
    deterministic, explicit friction priors.
    """
    n = cfg.manning_n
    region = str(cfg.manning_region or "default")
    by_region = cfg.manning_n_by_region
    if isinstance(by_region, dict) and region in by_region:
        try:
            n_reg = float(by_region[region])
            if np.isfinite(n_reg) and n_reg > 0:
                return n_reg
        except Exception:
            pass
    return n


def _compute_dmax_manning(width_m: float, slope_mpm: float, q_cms: float, cfg: InferConfig) -> float:
    """Estimate Dmax using Manning inversion (wide channel approximation).

    We solve for mean depth y in a wide rectangular channel:

        Q ≈ (1/n) * W * y^(5/3) * S^(1/2)

    => y ≈ ((n*Q)/(W*sqrt(S)))^(3/5)

    Then convert mean depth to an approximate Dmax using the trapezoid relation:
        mean_to_dmax ≈ 2/(1 + bottom_width_frac)

    This is a *soft prior*, not a bed observation.
    """
    W = float(width_m)
    S = float(slope_mpm)
    Q = float(q_cms)
    if not (np.isfinite(W) and W > 0 and np.isfinite(S) and S > 0 and np.isfinite(Q) and Q > 0):
        return float("nan")

    n = float(_manning_n_effective(cfg))
    with np.errstate(divide="ignore", invalid="ignore"):
        y_mean = ((n * Q) / (W * np.sqrt(S))) ** (3.0 / 5.0)

    if not np.isfinite(y_mean) or y_mean <= 0:
        return float("nan")

    mean_to_dmax = float(2.0 / (1.0 + float(cfg.bottom_width_frac)))
    dmax = float(y_mean) * mean_to_dmax
    return float(np.clip(dmax, cfg.dmin_m, cfg.dmax_m))


def _compute_manning_weight(row: pd.Series, cfg: InferConfig) -> float:
    """Compute a 0–1 guard factor for Manning/energy priors (separate from max weight)."""
    if str(cfg.manning_mode).lower().strip() == "off":
        return 0.0

    S = float(row.get("slope_mpm", np.nan))
    if not (np.isfinite(S) and S > 0):
        return 0.0

    # Backwater / tidal guard: very low slopes are risky for local normal-depth assumptions
    if S < float(cfg.manning_backwater_slope_thresh):
        return 0.0

    # Optional near-mouth guard if a distance-to-mouth attribute exists
    if cfg.manning_dist_to_mouth_field:
        dkm = row.get("dist_to_mouth_km", np.nan)
        if np.isfinite(dkm) and float(dkm) <= float(cfg.manning_dist_to_mouth_km_max):
            return 0.0

    return 1.0




# ---- Regional hydraulic geometry curves (Drainage Area -> bankfull depth) ----
# IMPORTANT: coefficients are highly region-specific.
# Built-ins below are *illustrative placeholders* so the workflow runs end-to-end.
# For defensible results, supply published coefficients for your state/region via
# --regional-curve-c/--regional-curve-f and specify the DA units they expect.
REGIONAL_CURVE_DEFAULTS = {
    # region: (c, f, da_units, depth_units, depth_type, unc_pct)
    "default": (0.25, 0.30, "km2", "m", "mean", 40.0),
    # Very rough physiographic placeholders (DO NOT treat as authoritative)
    "coastal_plain": (0.22, 0.32, "km2", "m", "mean", 45.0),
    "piedmont": (0.20, 0.33, "km2", "m", "mean", 40.0),
    "appalachian": (0.18, 0.35, "km2", "m", "mean", 45.0),
    "great_plains": (0.15, 0.30, "km2", "m", "mean", 55.0),
}





def _apply_1d_energy_solver(xs_param: pd.DataFrame, cfg: InferConfig, soundings_path: str | None, acct: dict | None = None) -> pd.DataFrame:
    """Option A: Solve (mean) depth from local WSE energy slope and Manning friction, then convert to Dmax."""
    def _acct_set(reason: str) -> None:
        if acct is None:
            return
        acct["energy_solver_requested"] = cfg.energy_solver_enabled
        acct["energy_solver_reason"] = str(reason)
        acct.setdefault("energy_solver_n_total", 0)
        acct.setdefault("energy_solver_n_applied", 0)

    if not cfg.energy_solver_enabled:
        _acct_set("disabled")
        return xs_param
    if cfg.energy_solver_only_when_no_soundings and soundings_path:
        _acct_set("skipped_soundings_present")
        return xs_param

    required_cols = {"component_id", "s_center_m", "width_m"}
    if not required_cols.issubset(set(xs_param.columns)):
        _acct_set("missing_required_cols")
        return xs_param
    if "manning_q_cms_used" not in xs_param.columns:
        _acct_set("missing_manning_q")
        return xs_param

    has_wse_m = "wse_m" in xs_param.columns
    has_wse_fit = ("wse_fit_m" in xs_param.columns) and xs_param["wse_fit_m"].notna().any()
    if (not has_wse_m) and (not has_wse_fit):
        _acct_set("missing_wse")
        return xs_param

    # Use fitted WSE where available, but fall back to per-XS proxy WSE when fit is missing.
    # The previous implementation selected a single column globally, which could silently drop
    # most stations if wse_fit_m is only partially populated.
    wse_mode = "wse_m"
    if has_wse_fit and has_wse_m:
        wse_mode = "wse_fit_m_fallback_wse_m"
    elif has_wse_fit:
        wse_mode = "wse_fit_m"

    if acct is not None:
        acct["energy_solver_wse_source"] = str(wse_mode)
        acct["energy_solver_wse_fit_used_n"] = 0
        acct["energy_solver_wse_raw_used_n"] = 0
        acct["energy_solver_groups_total"] = 0
        acct["energy_solver_groups_used"] = 0
        acct["energy_solver_groups_skipped_lt2"] = 0
        acct["energy_solver_ok_rows"] = 0
        acct["energy_solver_ok_pairs"] = 0

    n_val = float(_manning_n_effective(cfg))
    if not (np.isfinite(n_val) and n_val > 0):
        _acct_set("missing_manning_n")
        return xs_param

    # Safety gate: if upstream indicates the WSE anchor is DEM/topo proxy only,
    # skip unless explicitly permitted. This prevents "physics" adjustments from
    # being over-interpreted as observation-constrained when they are not.
    if acct is not None:
        wse_anchor = str(acct.get("wse_anchor_source", "unknown"))
        if (wse_anchor == "dem_proxy") and (not cfg.energy_allow_dem_proxy_wse):
            _acct_set("skipped_dem_proxy_wse")
            return xs_param

    mean_to_dmax = float(2.0 / (1.0 + float(cfg.bottom_width_frac)))

    rows_in: list[dict] = []
    rows_out: list[dict] = []

    n_total = 0
    n_applied = 0
    n_capped_min = 0
    n_capped_max = 0
    resid_vals: list[float] = []
    conf_vals: list[float] = []

    xs_param = xs_param.copy()
    xs_param["energy_slope_mpm"] = np.nan
    xs_param["energy_depth_mean_m"] = np.nan
    xs_param["energy_dmax_m"] = np.nan
    xs_param["energy_residual_mpm"] = np.nan
    xs_param["energy_conf"] = np.nan
    xs_param["energy_flags"] = ""

    for comp, g in xs_param.groupby("component_id", dropna=False, sort=False):
        # Robust ordering: xs_id may not exist depending on upstream steps.
        sort_cols = ["s_center_m"]
        if "xs_id" in g.columns:
            sort_cols.append("xs_id")
        gg = g.sort_values(sort_cols).copy()
        s = pd.to_numeric(gg["s_center_m"], errors="coerce").astype("float64").values
        wse_m = (
            pd.to_numeric(gg["wse_m"], errors="coerce").astype("float64").values
            if has_wse_m else np.full(len(gg), np.nan, dtype="float64")
        )
        wse_fit = (
            pd.to_numeric(gg["wse_fit_m"], errors="coerce").astype("float64").values
            if has_wse_fit else np.full(len(gg), np.nan, dtype="float64")
        )
        wse = wse_fit
        if has_wse_m:
            wse = np.where(np.isfinite(wse_fit), wse_fit, wse_m)
        B = pd.to_numeric(gg["width_m"], errors="coerce").astype("float64").values
        Q = pd.to_numeric(gg["manning_q_cms_used"], errors="coerce").astype("float64").values

        ok = np.isfinite(s) & np.isfinite(wse) & np.isfinite(B) & (B > 0) & np.isfinite(Q) & (Q > 0)
        if acct is not None:
            acct["energy_solver_groups_total"] = int(acct.get("energy_solver_groups_total", 0) or 0) + 1
            acct["energy_solver_ok_rows"] = int(acct.get("energy_solver_ok_rows", 0) or 0) + int(np.sum(ok))
            if has_wse_fit:
                acct["energy_solver_wse_fit_used_n"] = int(acct.get("energy_solver_wse_fit_used_n", 0) or 0) + int(np.sum(ok & np.isfinite(wse_fit)))
                if has_wse_m:
                    acct["energy_solver_wse_raw_used_n"] = int(acct.get("energy_solver_wse_raw_used_n", 0) or 0) + int(np.sum(ok & (~np.isfinite(wse_fit)) & np.isfinite(wse_m)))
        if ok.sum() < 2:
            if acct is not None:
                acct["energy_solver_groups_skipped_lt2"] = int(acct.get("energy_solver_groups_skipped_lt2", 0) or 0) + 1
            continue

        slope = np.full_like(s, np.nan, dtype="float64")
        pairs_ok = 0
        for i in range(len(s) - 1):
            if not (ok[i] and ok[i + 1]):
                continue
            dx = float(s[i + 1] - s[i])
            if not (np.isfinite(dx) and abs(dx) > 0):
                continue
            se = float((wse[i] - wse[i + 1]) / dx)
            if not np.isfinite(se):
                continue
            se = abs(se)
            se = float(np.clip(se, float(cfg.slope_min), float(cfg.slope_max)))
            slope[i] = se
            pairs_ok += 1
        if len(slope) >= 2:
            slope[-1] = slope[-2]
        gg["energy_slope_mpm"] = slope

        if acct is not None:
            acct["energy_solver_ok_pairs"] = int(acct.get("energy_solver_ok_pairs", 0) or 0) + int(pairs_ok)
            if pairs_ok > 0:
                acct["energy_solver_groups_used"] = int(acct.get("energy_solver_groups_used", 0) or 0) + 1

        dmin = float(cfg.dmin_m)
        if "dmax_env_m" in gg.columns:
            env = pd.to_numeric(gg["dmax_env_m"], errors="coerce").astype("float64").values
        else:
            env = np.full(len(gg), np.nan, dtype="float64")
        dmax_global = float(cfg.dmax_m)

        depth = np.full_like(s, np.nan, dtype="float64")
        dmax_energy = np.full_like(s, np.nan, dtype="float64")
        residual = np.full_like(s, np.nan, dtype="float64")
        conf = np.full_like(s, np.nan, dtype="float64")
        flags = [""] * len(s)

        for i in range(len(s)):
            if not ok[i] or not np.isfinite(slope[i]) or slope[i] <= 0:
                continue
            n_total += 1
            sf = float(slope[i])
            Bi = float(B[i])
            Qi = float(Q[i])
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                h = float(((n_val * n_val * Qi * Qi) / (Bi * Bi * sf)) ** (3.0 / 10.0))
            if not (np.isfinite(h) and h > 0):
                continue

            dmax_i = dmax_global
            if np.isfinite(env[i]) and env[i] > 0:
                dmax_i = float(min(dmax_i, env[i]))
            h_max = float(max(dmin, dmax_i / mean_to_dmax))

            f: list[str] = []
            if h < dmin:
                h = dmin
                n_capped_min += 1
                f.append("cap_dmin")
            if h > h_max:
                h = h_max
                n_capped_max += 1
                f.append("cap_dmax")

            depth[i] = h
            dm = float(np.clip(h * mean_to_dmax, float(cfg.dmin_m), dmax_global))
            dmax_energy[i] = dm

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                sf_implied = float((n_val * n_val * Qi * Qi) / (Bi * Bi * (h ** (10.0 / 3.0))))
            if np.isfinite(sf_implied):
                residual[i] = float(sf_implied - sf)
                resid_vals.append(residual[i])

                # A simple, parameter-light confidence score based on relative residual.
                # rel=|S_implied - S_est| / S_est. Use exp(-rel) to keep it conservative.
                rel = float(abs(residual[i]) / max(sf, 1e-12))
                conf[i] = float(np.exp(-rel))
                conf_vals.append(conf[i])

            flags[i] = "|".join(f)

            rows_in.append(dict(xs_id=str(gg.iloc[i].get("xs_id")), component_id=int(gg.iloc[i].get("component_id", -1)) if pd.notna(gg.iloc[i].get("component_id", np.nan)) else -1,
                                s_center_m=float(s[i]), width_m=float(Bi), wse_m=float(wse[i]), wse_source=str(wse_mode), manning_n=float(n_val),
                                discharge_cms=float(Qi), energy_slope_mpm=float(sf), dmax_env_m=float(env[i]) if np.isfinite(env[i]) else None))
            rows_out.append(dict(xs_id=str(gg.iloc[i].get("xs_id")), energy_depth_mean_m=float(h), energy_dmax_m=float(dm),
                                 energy_residual_mpm=float(residual[i]) if np.isfinite(residual[i]) else None, flags=flags[i]))
            n_applied += 1

        gg["energy_depth_mean_m"] = depth
        gg["energy_dmax_m"] = dmax_energy
        gg["energy_residual_mpm"] = residual
        gg["energy_conf"] = conf
        gg["energy_flags"] = flags
        xs_param.loc[gg.index, ["energy_slope_mpm", "energy_depth_mean_m", "energy_dmax_m", "energy_residual_mpm", "energy_conf", "energy_flags"]] = gg[
            ["energy_slope_mpm", "energy_depth_mean_m", "energy_dmax_m", "energy_residual_mpm", "energy_conf", "energy_flags"]
        ]

    # Summarize solver outcome for debuggability.
    reason = "ok"
    if n_total == 0:
        reason = "no_valid_stations"
        log.warning("[ENERGY] No valid solver stations (need >=2 XS per component with finite Q/width/WSE).")
    elif n_applied == 0:
        reason = "no_applied"
        log.warning("[ENERGY] Solver had candidate stations but did not apply to any (numerical/filters).")

    if acct is not None:
        acct["energy_solver_enabled"] = True
        acct["energy_solver_reason"] = str(reason)
        acct["energy_solver_wse_source"] = str(wse_mode)
        acct["energy_solver_n_total"] = int(n_total)
        acct["energy_solver_n_applied"] = int(n_applied)
        acct["energy_solver_cap_dmin"] = int(n_capped_min)
        acct["energy_solver_cap_dmax"] = int(n_capped_max)

    try:
        in_p = cfg.out_1d_solver_inputs_json
        out_p = cfg.out_1d_solver_outputs_json
        acct_p = cfg.out_1d_solver_accounting_json
        if in_p:
            Path(str(in_p)).parent.mkdir(parents=True, exist_ok=True)
            Path(str(in_p)).write_text(_json.dumps({"stations": rows_in}, indent=2, sort_keys=True), encoding="utf-8")
        if out_p:
            Path(str(out_p)).parent.mkdir(parents=True, exist_ok=True)
            Path(str(out_p)).write_text(_json.dumps({"stations": rows_out}, indent=2, sort_keys=True), encoding="utf-8")
        if acct_p:
            payload = {"enabled": True, "reason": str(reason), "wse_source": str(wse_mode), "n_total": int(n_total), "n_applied": int(n_applied),
                       "cap_dmin": int(n_capped_min), "cap_dmax": int(n_capped_max)}
            if resid_vals:
                payload["residual_mean_mpm"] = float(np.nanmean(resid_vals))
                payload["residual_abs_p95_mpm"] = float(np.nanpercentile(np.abs(resid_vals), 95))
            if conf_vals:
                payload["confidence_mean"] = float(np.nanmean(conf_vals))
                payload["confidence_p05"] = float(np.nanpercentile(conf_vals, 5))
            Path(str(acct_p)).parent.mkdir(parents=True, exist_ok=True)
            Path(str(acct_p)).write_text(_json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as e:
        log.warning("[ENERGY] Failed to write 1D solver artifacts: %s", e)

    return xs_param

def _convert_da_units(da_km2: float, to_units: str) -> float:
    """Convert DA in km^2 to requested units."""
    if not np.isfinite(da_km2):
        return float("nan")
    u = str(to_units).lower().strip()
    if u == "km2":
        return float(da_km2)
    if u == "mi2":
        return float(da_km2 * 0.3861021585424458)
    return float(da_km2)


def _convert_depth_units(depth: float, from_units: str, to_units: str = "m") -> float:
    if not np.isfinite(depth):
        return float("nan")
    fu = str(from_units).lower().strip()
    tu = str(to_units).lower().strip()
    if fu == tu:
        return float(depth)
    if fu == "ft" and tu == "m":
        return float(depth * 0.3048)
    if fu == "m" and tu == "ft":
        return float(depth / 0.3048)
    return float(depth)


def _compute_dmax_regional_curve(row: pd.Series, cfg: InferConfig) -> Tuple[float, float, str]:
    """Compute a regional-curve-derived Dmax and a blend weight.

    Returns (dmax_m, weight, detail). If unavailable, returns (nan, 0, "").
    """
    if not cfg.regional_curve_enabled:
        return (float("nan"), 0.0, "")

    # Drainage area
    # Try configured field first, then common NHDPlus / StreamStats-style names.
    da = float("nan")
    cand_fields = []
    if cfg.regional_curve_da_field:
        cand_fields.append(str(cfg.regional_curve_da_field))
    cand_fields.extend([
        "drain_area_km2", "drainage_area_km2", "DA_km2", "DA_KM2", "DA_sqkm", "TotDASqKM", "DrainArKm2",
        "drain_area_mi2", "drainage_area_mi2", "DA_mi2", "DA_MI2", "DA_sqmi", "TotDASqMI", "DrainArMi2",
    ])
    for f in cand_fields:
        if f in row.index:
            da = float(row.get(f, np.nan))
            if np.isfinite(da):
                # If we fell back to a mi2 field, update units to mi2 unless user forced otherwise.
                if f.lower().endswith("mi2") or "sqmi" in f.lower() or f.lower().endswith("sqmi") or f.lower().endswith("dasqmi"):
                    if cfg.regional_curve_da_units is None:
                        cfg.regional_curve_da_units = "mi2"  # type: ignore
                break

    if not (np.isfinite(da) and da > 0):
        return (float("nan"), 0.0, "")

    # Ignore tiny basins (curves unstable; also often headwater morphology)
    if float(da) < cfg.regional_curve_min_da_km2:
        return (float("nan"), 0.0, "")

    # Coefficients
    c = cfg.regional_curve_c
    f = cfg.regional_curve_f
    reg_raw = cfg.regional_curve_region

    # Anti-slop guard: built-in coefficients are illustrative placeholders and must be explicitly opted into.
    # If the user did not provide explicit (c,f), require that they *explicitly* provided a region key.
    if (c is None or f is None) and (reg_raw is None):
        return (float("nan"), 0.0, "rc:no_coeffs_no_region")

    reg = str(reg_raw).lower().replace(" ", "_").replace("-", "_") if reg_raw is not None else ""
    if (c is None or f is None) and (reg in ("", "default", "none")):
        return (float("nan"), 0.0, "rc:default_placeholder_blocked")

    used_builtin = False
    if c is None or f is None:
        c0, f0, da_u0, depth_u0, depth_t0, unc0 = REGIONAL_CURVE_DEFAULTS.get(reg, REGIONAL_CURVE_DEFAULTS["default"])
        used_builtin = True
        if not cfg.allow_builtin_regional_curves:
            return (float("nan"), 0.0, "rc:builtin_not_allowed")
        if c is None:
            c = c0
        if f is None:
            f = f0
        da_units = cfg.regional_curve_da_units
        depth_units = cfg.regional_curve_depth_units
        depth_type = cfg.regional_curve_depth_type
        unc_pct = cfg.regional_curve_unc_pct
    else:
        da_units = cfg.regional_curve_da_units
        depth_units = cfg.regional_curve_depth_units
        depth_type = cfg.regional_curve_depth_type
        unc_pct = cfg.regional_curve_unc_pct
    da_use = _convert_da_units(float(da), str(da_units)) if str(da_units).lower().strip() != "km2" else float(da)
    # (If da_units is km2, da_use is km2; if mi2, converted.)
    # Bankfull depth
    with np.errstate(over="ignore", invalid="ignore"):
        d_bkf = float(c) * (float(da_use) ** float(f))

    d_bkf_m = _convert_depth_units(d_bkf, depth_units, "m")
    if not (np.isfinite(d_bkf_m) and d_bkf_m > 0):
        return (float("nan"), 0.0, "")

    # Convert to Dmax if needed
    to_dmax_mode = str(cfg.regional_curve_to_dmax or "auto").lower().strip()
    if str(depth_type).lower().strip() == "max":
        dmax = d_bkf_m
        conv = "bkf=max"
    else:
        if to_dmax_mode == "factor":
            dmax = d_bkf_m * cfg.regional_curve_to_dmax_factor
            conv = f"bkf=mean*{cfg.regional_curve_to_dmax_factor:.2f}"
        else:
            # trapezoid mean-to-dmax conversion
            mean_to_dmax = float(2.0 / (1.0 + float(cfg.bottom_width_frac)))
            dmax = d_bkf_m * mean_to_dmax
            conv = "bkf=mean->dmax(trap)"

    dmax = float(np.clip(dmax, cfg.dmin_m, cfg.dmax_m))

    # Weight: inverse of uncertainty, capped
    # Basic: weight = max_weight * (1 - unc_pct/100) clipped
    max_w = cfg.regional_curve_max_weight
    w = max(0.0, min(1.0, 1.0 - float(unc_pct) / 100.0))
    w = float(np.clip(max_w * w, 0.0, 1.0))

    detail = f"{('builtin:' if used_builtin else '')}{reg}:{conv}:unc{unc_pct:.0f}%"
    return (dmax, w, detail)
def _uncertainty_for_row(row: pd.Series, cfg: InferConfig) -> float:
    """Heuristic uncertainty (1-sigma, meters) by depth source with basic quality modifiers."""
    src = str(row.get("calib_src", "prior") or "prior").lower().strip()
    try:
        n = int(row.get("calib_n", 0) or 0)
    except Exception:
        n = 0

    # Base uncertainties
    if src == "soundings":
        base = float(max(0.35, 0.90 / max(1, np.sqrt(max(n, 1)))))
    elif src == "width_stage":
        base = 1.0
    elif src == "usgs":
        base = 1.2
    else:
        base = 1.5

    # Quality modifiers
    if src == "width_stage":
        r2 = row.get("width_stage_r2", np.nan)
        wt = row.get("width_stage_wt", np.nan)
        if np.isfinite(r2) and float(r2) < 0.85:
            base *= 1.2
        if np.isfinite(wt) and float(wt) < 0.5:
            base *= 1.2

    if src == "usgs":
        a_cv = row.get("usgs_a_cv", np.nan)
        wt = row.get("usgs_wt", np.nan)
        if np.isfinite(a_cv) and float(a_cv) > 0.5:
            base *= 1.3
        if np.isfinite(wt):
            # low weight implies less trust
            base *= float(1.0 + max(0.0, 0.6 - float(wt)))

    return float(np.clip(base, 0.25, 5.0))
def _load_width_stage_csvs(paths) -> pd.DataFrame:
    """Load one or more width-stage CSVs.

    Expected columns (flexible):
      - width_m / width (or width_ft)
      - stage_m / stage (or stage_ft)
      - site_no / site (optional, used to link to a USGS gage)
      - date (optional)

    Returns a long dataframe with standardized columns: site_no, width_m, stage_m
    """
    if not paths:
        return pd.DataFrame(columns=["site_no", "width_m", "stage_m"])

    if isinstance(paths, (str, Path)):
        paths = [paths]

    out_rows = []
    for pth in paths:
        if pth is None:
            continue
        for part in str(pth).split(","):
            part = part.strip()
            if not part:
                continue
            fp = Path(part)
            if not fp.exists():
                continue
            df = pd.read_csv(fp)

            cols = list(df.columns)
            width_col = _guess_field(cols, ["width_m", "width", "wet_width_m", "w_m", "width_ft", "width_feet"])
            stage_col = _guess_field(cols, ["stage_m", "stage", "gage_height_m", "h_m", "stage_ft", "gage_height_ft", "stage_feet"])
            site_col = _guess_field(cols, ["site_no", "site", "usgs_site", "station", "station_id"])

            if width_col is None or stage_col is None:
                continue

            width = pd.to_numeric(df[width_col], errors="coerce")
            stage = pd.to_numeric(df[stage_col], errors="coerce")

            width_is_ft = "ft" in str(width_col).lower() or "feet" in str(width_col).lower()
            stage_is_ft = "ft" in str(stage_col).lower() or "feet" in str(stage_col).lower()

            if width_is_ft:
                width = width * 0.3048
            if stage_is_ft:
                stage = stage * 0.3048

            if site_col is None:
                site = pd.Series([pd.NA] * len(df))
            else:
                site = df[site_col].astype(str)

            m = width.notna() & stage.notna() & (width > 0)
            for s, w, h in zip(site[m], width[m], stage[m]):
                out_rows.append({"site_no": s, "width_m": float(w), "stage_m": float(h)})

    if not out_rows:
        return pd.DataFrame(columns=["site_no", "width_m", "stage_m"])

    out = pd.DataFrame(out_rows)
    out.loc[out["site_no"].astype(str).isin(["nan", "None", "NA", ""]), "site_no"] = pd.NA
    return out


def _compute_dmax_geomorphic_envelope(row: pd.Series, cfg: InferConfig) -> Tuple[float, str]:
    """Compute a conservative upper bound on Dmax using the regional curve.

    This is intended to stabilize absolute depth scale in no-sounding reaches where
    hydraulic inversion can be weak/unstable (e.g., very low slopes).
    Returns (dmax_env_m, detail). If unavailable, returns (nan, reason).
    """
    if not cfg.geomorphic_envelope_enabled:
        return float("nan"), "disabled"
    if cfg.geomorphic_envelope_only_when_no_soundings:
        # if soundings were used anywhere, do not apply envelope as a hard cap
        if str(row.get("calib_src", "")).lower().strip() == "soundings":
            return float("nan"), "soundings_calibrated"
    # Reuse the regional curve computation (DA -> depth), but treat as a cap rather than a blend.
    d_rc, w_rc, det = _compute_dmax_regional_curve(row, cfg)
    if not np.isfinite(d_rc):
        return float("nan"), "no_regional_curve:" + str(det)
    d_env = float(d_rc)
    if cfg.geomorphic_envelope_inflate_unc:
        try:
            unc = cfg.regional_curve_unc_pct
            if np.isfinite(unc) and unc > 0:
                d_env *= (1.0 + unc / 100.0)
        except Exception:
            pass
    d_env = float(np.clip(d_env, float(cfg.dmin_m), float(cfg.dmax_m)))
    return d_env, f"rc_cap({det})"


def _fit_width_stage_beta(ws: pd.DataFrame) -> Tuple[Optional[float], int, float]:
    """Fit stage = alpha + beta * width.

    Returns:
      beta (m per m), n_used, r2

    Notes:
    - We expect beta > 0.
    - r2 is computed on the subset used in the fit.
    """
    if ws is None or ws.empty:
        return None, 0, float("nan")
    w = pd.to_numeric(ws["width_m"], errors="coerce")
    h = pd.to_numeric(ws["stage_m"], errors="coerce")
    m = w.notna() & h.notna() & (w > 0)
    n = int(m.sum())
    if n < 3:
        return None, n, float("nan")

    wv = w[m].to_numpy(dtype="float64")
    hv = h[m].to_numpy(dtype="float64")
    try:
        beta, alpha = np.polyfit(wv, hv, 1)
    except Exception:
        return None, n, float("nan")

    if (not np.isfinite(beta)) or beta <= 0:
        return None, n, float("nan")

    # R^2
    try:
        yhat = alpha + beta * wv
        ss_res = float(((hv - yhat) ** 2).sum())
        ss_tot = float(((hv - hv.mean()) ** 2).sum())
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    except Exception:
        r2 = float("nan")

    return float(beta), n, float(r2)


def _width_stage_weight(n: int, r2: float, cfg: InferConfig) -> float:
    """Compute a conservative blend weight for width–stage calibration.

    Width–stage inversion is treated as a *soft* constraint. Weight increases with:
    - number of observations (n)
    - fit quality (r2)

    The maximum weight is capped by cfg.width_stage_max_weight.
    """
    if n is None:
        return 0.0
    n = int(max(0, n))
    if n < int(cfg.width_stage_min_n):
        return 0.0
    if not np.isfinite(r2) or float(r2) < float(cfg.width_stage_min_r2):
        return 0.0

    n_scale = min(1.0, (n / max(1.0, float(cfg.width_stage_min_n) * 2.0)) ** 0.5)
    r2_scale = max(0.0, min(1.0, float(r2)))
    w = float(cfg.width_stage_max_weight) * n_scale * (0.25 + 0.75 * r2_scale)
    return float(np.clip(w, 0.0, float(cfg.width_stage_max_weight)))


def _dmax_from_width_stage(beta: float, Wtop: float, bottom_frac: float) -> float:
    """Given stage-width slope beta and channel bank width Wtop, infer Dmax for trapezoid."""
    if not np.isfinite(beta) or not np.isfinite(Wtop) or Wtop <= 0:
        return float("nan")
    Wb = float(bottom_frac) * float(Wtop)
    return float(max(0.0, beta * (Wtop - Wb)))


# --------------------------------------------------------------------------------------
# Soundings (optional)
# --------------------------------------------------------------------------------------



def _looks_like_lonlat(x, y) -> bool:
    """Heuristic: do x/y look like lon/lat degrees?"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not (np.isfinite(x).any() and np.isfinite(y).any()):
        return False
    xmax = float(np.nanmax(x))
    xmin = float(np.nanmin(x))
    ymax = float(np.nanmax(y))
    ymin = float(np.nanmin(y))
    # Standard lon/lat ranges
    if xmin >= -180.0 and xmax <= 180.0 and ymin >= -90.0 and ymax <= 90.0:
        return True
    # 0..360 longitudes
    if xmin >= 0.0 and xmax <= 360.0 and ymin >= -90.0 and ymax <= 90.0:
        return True
    return False


def _guess_xy_crs(x, y, target_crs, x_name: str | None = None, y_name: str | None = None) -> CRS:
    """Guess source CRS for x/y point columns.

    If values look like lon/lat degrees, returns EPSG:4326. Otherwise, if the target CRS is
    projected, assumes x/y are already in target CRS (common for pre-projected XYZs in this pipeline).
    """
    # Column-name hint (best-effort)
    if x_name and y_name:
        xn = x_name.lower()
        yn = y_name.lower()
        if ("lon" in xn or "long" in xn) and ("lat" in yn):
            return CRS.from_epsg(4326)

    if _looks_like_lonlat(x, y):
        return CRS.from_epsg(4326)

    t = CRS.from_user_input(target_crs)
    if t.is_projected:
        return t

    # Fall back: we can't safely infer a projected CRS when target is geographic
    return CRS.from_epsg(4326)
def _load_soundings(
    path: Path,
    target_crs,
    depth_col: Optional[str],
    elev_col: Optional[str],
    x_col: Optional[str],
    y_col: Optional[str],
    soundings_crs: Optional[str] = None,
) -> Optional[gpd.GeoDataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    suf = path.suffix.lower()

    if suf == ".parquet":
        df = pd.read_parquet(path)
        if df.empty:
            log.warning("[SOUNDINGS] Parquet soundings file is empty (0 rows): %s", path)
            return None
        # Expect x/y/z columns; allow lon/lat as fallback.
        if x_col is None or y_col is None:
            for xc, yc in [("x", "y"), ("lon", "lat"), ("longitude", "latitude"), ("easting", "northing")]:
                if xc in df.columns and yc in df.columns:
                    x_col, y_col = xc, yc
                    break
        if x_col is None or y_col is None:
            raise RuntimeError("Soundings Parquet requires x/y columns (or specify --soundings-x-col/--soundings-y-col).")
        # CRS is stored as a string column when produced by our subset writer.
        if soundings_crs:
            src_crs = CRS.from_user_input(soundings_crs)
        elif "crs" in df.columns and df["crs"].notna().any() and str(df["crs"].iloc[0]).strip():
            src_crs = CRS.from_user_input(str(df["crs"].iloc[0]))
        else:
            src_crs = _guess_xy_crs(df[x_col].to_numpy(), df[y_col].to_numpy(), target_crs, x_col, y_col)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs=src_crs)

    elif suf == ".csv":
        df = pd.read_csv(path)
        if x_col is None or y_col is None:
            for xc, yc in [("lon", "lat"), ("longitude", "latitude"), ("x", "y"), ("easting", "northing")]:
                if xc in df.columns and yc in df.columns:
                    x_col, y_col = xc, yc
                    break
        if x_col is None or y_col is None:
            raise RuntimeError("Soundings CSV requires --soundings-x-col/--soundings-y-col (or lon/lat columns).")

        if soundings_crs:
            src_crs = CRS.from_user_input(soundings_crs)
        else:
            src_crs = _guess_xy_crs(df[x_col].to_numpy(), df[y_col].to_numpy(), target_crs, x_col, y_col)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs=src_crs)
    elif suf in (".xyz", ".txt", ".dat"):
        # USACE eHydro and similar XYZ soundings are commonly whitespace or comma-delimited
        # with 3 columns: x y z (or lon lat depth). Support optional header.
        # We assume EPSG:4326 for XYZ/CSV unless the user provides a vector file with CRS.
        # If your XYZ is projected (e.g., UTM), convert to GPKG/GeoJSON or extend this loader with a --soundings-crs option.
        first = None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("//"):
                    continue
                first = s
                break
        has_header = False
        if first is not None:
            # If the first non-comment line contains letters, treat as header
            has_header = any(ch.isalpha() for ch in first)
        if has_header:
            df = pd.read_csv(path, sep=r"[\s,]+", engine="python", comment="#")
        else:
            df = pd.read_csv(path, sep=r"[\s,]+", engine="python", comment="#", header=None, names=["x","y","z"])

        # Resolve x/y columns if user didn't specify
        if x_col is None or y_col is None:
            for xc, yc in [("lon", "lat"), ("longitude", "latitude"), ("x", "y"), ("easting", "northing")]:
                if xc in df.columns and yc in df.columns:
                    x_col, y_col = xc, yc
                    break
        if x_col is None or y_col is None:
            # Fall back to first two columns
            x_col, y_col = df.columns[0], df.columns[1]

        if soundings_crs:
            src_crs = CRS.from_user_input(soundings_crs)
        else:
            src_crs = _guess_xy_crs(df[x_col].to_numpy(), df[y_col].to_numpy(), target_crs, x_col, y_col)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs=src_crs)
    else:
        gdf = gpd.read_file(path)
        if gdf.empty:
            return None
        if gdf.crs is None:
            raise RuntimeError(f"Soundings file has no CRS: {path}")

    gdf = gdf.to_crs(target_crs)

    cols_lower = {c.lower(): c for c in gdf.columns}
    if depth_col is None:
        for cand in ["depth_m", "depth", "depthmeter", "depthmeters"]:
            if cand in cols_lower:
                depth_col = cols_lower[cand]
                break
    if elev_col is None:
        for cand in ["z_m", "elev_m", "elevation", "elev", "z"]:
            if cand in cols_lower:
                elev_col = cols_lower[cand]
                break

    gdf["_depth_m"] = pd.to_numeric(gdf[depth_col], errors="coerce") if depth_col and depth_col in gdf.columns else np.nan
    gdf["_z_m"] = pd.to_numeric(gdf[elev_col], errors="coerce") if elev_col and elev_col in gdf.columns else np.nan
    return gdf


def _expand_soundings_inputs(items) -> list[Path]:
    """Expand soundings inputs:
    - repeated args (list)
    - comma-separated lists within an arg
    - directories (glob for supported extensions)
    Returns a de-duplicated list of Paths (order-preserving).
    """
    if not items:
        return []
    out: list[Path] = []
    for item in items:
        if item is None:
            continue
        # allow comma-separated
        parts = [p.strip() for p in str(item).split(",") if p.strip()]
        for part in parts:
            pth = Path(part)
            if pth.is_dir():
                for ext in ("*.xyz", "*.csv", "*.txt", "*.dat", "*.gpkg", "*.shp", "*.geojson", "*.json", "*.parquet"):
                    out.extend(sorted(pth.glob(ext)))
            else:
                out.append(pth)

    seen = set()
    out2: list[Path] = []
    for p in out:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            out2.append(p)
    return out2


def _load_soundings_many(
    items,
    target_crs,
    depth_col: Optional[str],
    elev_col: Optional[str],
    x_col: Optional[str],
    y_col: Optional[str],
    soundings_crs: Optional[str] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Load and merge multiple soundings inputs into a single GeoDataFrame."""
    paths = _expand_soundings_inputs(items)
    if not paths:
        return None
    gdfs = []
    for p in paths:
        try:
            g = _load_soundings(
                path=Path(p),
                target_crs=target_crs,
                depth_col=depth_col,
                elev_col=elev_col,
                x_col=x_col,
                y_col=y_col,
                soundings_crs=soundings_crs,
            )
            if g is not None and not g.empty:
                g = g.copy()
                g["_src_file"] = str(p)
                gdfs.append(g)
        except Exception as e:
            log.warning("[SOUNDINGS] Failed to load '%s': %s", str(p), e)
    if not gdfs:
        return None
    df = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=target_crs)


def _write_soundings_subset(path: Path, soundings: gpd.GeoDataFrame) -> None:
    """Write a unified soundings subset for reuse by downstream river steps.

    Preferred output is **Parquet** (fast + small). If the target path ends with
    .parquet (or has no suffix), a flat table is written with explicit x/y.

    Fallback output is GPKG (geometry-preserving) if parquet isn't available.
    """
    if soundings is None or soundings.empty:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Default to parquet if no suffix.
    if path.suffix == "":
        path = path.with_suffix(".parquet")

    # Prefer a depth column if available; keep elevation too when present.
    out = soundings.copy()
    if "_depth_m" in out.columns:
        out["depth"] = pd.to_numeric(out["_depth_m"], errors="coerce")
        out["depth_m"] = out["depth"]
    if "_z_m" in out.columns:
        out["z"] = pd.to_numeric(out["_z_m"], errors="coerce")
        out["z_m"] = out["z"]

    keep = ["geometry"]
    for c in ["depth", "depth_m", "z", "z_m", "_src_file"]:
        if c in out.columns:
            keep.append(c)
    out = out[keep].copy()
    out = out[out.geometry.notnull() & (~out.geometry.is_empty)].copy()
    if out.empty:
        return

    ext = path.suffix.lower()

    # Remove any pre-existing file to avoid stale layers.
    try:
        if path.exists():
            path.unlink()
    except Exception:
        log.debug("Could not remove existing subset file %s; will overwrite.", str(path), exc_info=True)

    if ext == ".parquet":
        # Parquet: store explicit x/y + z + src + crs.
        try:
            import pyarrow  # noqa: F401
            # Coalesce vertical values: prefer depth (positive-down) when valid, fall back to
            # z_m/z (elevation). This is critical for elevation-only sources like eHydro XYZ
            # where _depth_m is NaN and the elevation lives in _z_m.
            # BUG FIX: the old code did `out["depth"] if "depth" in columns` which always triggered
            # (depth column is always created above, just set to NaN for elevation-only data),
            # resulting in an all-NaN z column and an empty parquet after the isfinite filter.
            _depth_s = pd.to_numeric(out["depth"], errors="coerce") if "depth" in out.columns else pd.Series(np.nan, index=out.index, dtype="float64")
            _z_s = (
                pd.to_numeric(out["z_m"], errors="coerce") if "z_m" in out.columns
                else pd.to_numeric(out["z"], errors="coerce") if "z" in out.columns
                else pd.Series(np.nan, index=out.index, dtype="float64")
            )
            # z = coalesced primary filter column (depth if valid, else elevation)
            _z_coalesced = _depth_s.combine_first(_z_s)
            tbl = pd.DataFrame({
                "x": out.geometry.x.astype("float64"),
                "y": out.geometry.y.astype("float64"),
                # Canonical coalesced column for the isfinite filter (never all-NaN).
                "z": _z_coalesced,
                # Explicit semantic columns so _load_soundings finds the right one by name:
                # depth_m  -> _depth_m (positive-down; NaN when source is elevation-only)
                # z_m      -> _z_m (elevation in vertical datum; NaN when source is depth-only)
                "depth_m": _depth_s,
                "z_m": _z_s,
                "_src_file": out.get("_src_file", "unknown"),
                "crs": str(out.crs) if out.crs is not None else "",
            })
            tbl = tbl[np.isfinite(tbl["x"]) & np.isfinite(tbl["y"]) & np.isfinite(tbl["z"])].copy()
            if tbl.empty:
                log.warning(
                    "[SOUNDINGS] Subset parquet would be empty after finite filter "
                    "(all rows had NaN for both depth and z_m). Check sounding vertical datum. "
                    "Falling back to GPKG to preserve raw geometry."
                )
                raise ValueError("empty after finite filter")
            tbl.to_parquet(path, index=False)
            log.debug("[SOUNDINGS] Subset parquet written: %d rows, columns=%s", len(tbl), list(tbl.columns))
            return
        except Exception as e:
            log.warning("[SOUNDINGS] Parquet write failed (%s); falling back to GPKG.", e)
            path = path.with_suffix(".gpkg")

    # GPKG fallback
    out.to_file(path, driver="GPKG")


def _soundings_one_line(path: Path, n_out: int, n_in: int, by_src: dict[str, int] | None = None) -> str:
    """One-line summary: <file>: n=<out> from <in> (src=...)."""
    parts = []
    if by_src:
        for k, v in sorted(by_src.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            parts.append(f"{k}={int(v):,}")
    src = f" ({', '.join(parts)})" if parts else ""
    return f"{Path(path).name}: n={int(n_out):,} from {int(n_in):,}{src}"




def _load_wse_obs(
    path: Path,
    target_crs,
    wse_col: str = "wse_m",
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    csv_crs: str = "EPSG:4326",
) -> Optional[gpd.GeoDataFrame]:
    """Load water-surface elevation (WSE) observations as points.

    Supported:
      - Vector (gpkg/shp/geojson): point geometries + a WSE column
      - CSV: requires x_col/y_col + a WSE column

    Returns a GeoDataFrame in target_crs with a float column `_wse_m`.
    """
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    suf = path.suffix.lower()
    if suf == ".csv":
        if not (x_col and y_col):
            raise ValueError("CSV WSE requires --swot-x-col and --swot-y-col.")
        df = pd.read_csv(path)
        if wse_col not in df.columns:
            raise ValueError(f"WSE column '{wse_col}' not found in CSV: {path}")
        g = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[x_col].astype(float), df[y_col].astype(float)),
            crs=CRS.from_user_input(csv_crs),
        )
    else:
        g = gpd.read_file(path)
        if g.empty:
            return None
        if g.crs is None:
            raise RuntimeError(f"WSE observation file has no CRS: {path}")
        if wse_col not in g.columns:
            raise ValueError(f"WSE column '{wse_col}' not found in {path}. Available: {list(g.columns)}")

    g = g.copy()
    g["_wse_m"] = pd.to_numeric(g[wse_col], errors="coerce")
    g = g[np.isfinite(g["_wse_m"])].copy()
    if g.empty:
        return None

    # Reproject to target CRS for spatial matching
    if target_crs is not None:
        g = g.to_crs(target_crs)

    return gpd.GeoDataFrame(g, geometry="geometry", crs=g.crs)


def _project_for_distance(
    a: gpd.GeoDataFrame,
    b: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, CRS]:
    """Ensure both GeoDataFrames are in a projected CRS suitable for meter distances."""
    crs_a = CRS.from_user_input(a.crs)
    if crs_a.is_projected:
        return a, b.to_crs(a.crs), crs_a

    # Choose UTM from centroid of a
    try:
        union_geom = a.geometry.union_all()
    except Exception:
        union_geom = unary_union(list(a.geometry))
    c = union_geom.centroid
    utm = _utm_crs_from_lonlat(float(c.x), float(c.y))
    return a.to_crs(utm), b.to_crs(utm), CRS.from_user_input(utm)


def _attach_swot_wse_to_xs(
    xs_lines: gpd.GeoDataFrame,
    xs_param: pd.DataFrame,
    swot: gpd.GeoDataFrame,
    cfg: InferConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Attach WSE observations to cross-sections and blend with the DEM/topo proxy.

    Returns:
      - updated xs_param (adds swot_wse_m, swot_dist_m, wse_proxy_m, wse_src, updates wse_m)
      - updated wse_by_xs mapping (consistent with xs_param['wse_m'])
    """
    xs_param = xs_param.copy()
    xs_param["wse_proxy_m"] = xs_param.get("wse_m", np.nan)
    xs_param["swot_wse_m"] = np.nan
    xs_param["swot_dist_m"] = np.nan
    xs_param["wse_src"] = "dem_proxy"

    if swot is None or swot.empty:
        wse_by_xs = dict(zip(xs_param["xs_id"].astype(str), xs_param["wse_m"].astype("float64")))
        return xs_param, wse_by_xs

    # Build XS center points for matching
    xsc = xs_lines[["xs_id", "geometry"]].copy()
    xsc["xs_id"] = xsc["xs_id"].astype(str)
    try:
        xsc["geometry"] = xsc.geometry.interpolate(0.5, normalized=True)
    except Exception:
        xsc["geometry"] = xsc.geometry.centroid

    xsc = gpd.GeoDataFrame(xsc, geometry="geometry", crs=xs_lines.crs)

    # Project to meters for distance constraints
    xsc_m, swot_m, _ = _project_for_distance(xsc, swot)

    # Attach observations using a metric-space KDTree (more robust than sjoin_nearest,
    # avoids spatial-index backend requirements, and supports robust aggregation when
    # multiple observations fall near the same cross-section).
    sw = pd.DataFrame({"xs_id": xsc_m["xs_id"].astype(str).values})
    sw["swot_wse_m"] = np.nan
    sw["swot_dist_m"] = np.nan

    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None

    max_d = float(cfg.swot_max_dist_m)
    if cKDTree is None:
        # Fallback to geopandas nearest join (requires spatial index backend)
        try:
            joined = gpd.sjoin_nearest(
                xsc_m,
                swot_m[["_wse_m", "geometry"]],
                how="left",
                distance_col="_dist_m",
            )
            joined.loc[joined["_dist_m"] > max_d, "_wse_m"] = np.nan
            joined.loc[joined["_dist_m"] > max_d, "_dist_m"] = np.nan
            sw = joined[["xs_id", "_wse_m", "_dist_m"]].copy()
            sw = sw.rename(columns={"_wse_m": "swot_wse_m", "_dist_m": "swot_dist_m"})
            sw["swot_wse_m"] = pd.to_numeric(sw["swot_wse_m"], errors="coerce")
        except Exception as e:
            raise RuntimeError(
                "SWOT WSE attachment failed. Install SciPy for KDTree support, or install a GeoPandas spatial index backend (rtree/pygeos). "
                f"Original error: {e}"
            )
    else:
        # KDTree-based neighbor aggregation
        sw_xy = np.column_stack([swot_m.geometry.x.values.astype(float), swot_m.geometry.y.values.astype(float)])
        xs_xy = np.column_stack([xsc_m.geometry.x.values.astype(float), xsc_m.geometry.y.values.astype(float)])
        sw_wse = pd.to_numeric(swot_m["_wse_m"], errors="coerce").values.astype(float)

        # Filter finite observations before building the tree
        ok_obs = np.isfinite(sw_xy[:, 0]) & np.isfinite(sw_xy[:, 1]) & np.isfinite(sw_wse)
        sw_xy = sw_xy[ok_obs]
        sw_wse = sw_wse[ok_obs]
        if sw_xy.shape[0] > 0:
            tree = cKDTree(sw_xy)

            def _robust_median_mad(v: np.ndarray, zmax: float) -> float:
                v = np.asarray(v, dtype=float)
                v = v[np.isfinite(v)]
                if v.size == 0:
                    return float("nan")
                if v.size < 3:
                    return float(np.nanmedian(v))
                med = float(np.nanmedian(v))
                mad = float(np.nanmedian(np.abs(v - med)))
                if mad <= 0.0 or (not np.isfinite(mad)):
                    return float(med)
                z = np.abs(v - med) / (1.4826 * mad)
                v2 = v[z <= float(zmax)]
                if v2.size == 0:
                    return float(med)
                return float(np.nanmedian(v2))

            # Query up to k nearest to stabilize aggregation and distance reporting.
            # Use a small k cap for speed; aggregation uses only those within max_d anyway.
            k = int(min(16, max(4, sw_xy.shape[0])))
            dists, idxs = tree.query(xs_xy, k=k, distance_upper_bound=max_d)
            # Ensure 2D arrays
            dists = np.atleast_2d(dists)
            idxs = np.atleast_2d(idxs)

            wse_out = np.full((xs_xy.shape[0],), np.nan, dtype=float)
            dist_out = np.full((xs_xy.shape[0],), np.nan, dtype=float)

            for i in range(xs_xy.shape[0]):
                di = np.asarray(dists[i], dtype=float).ravel()
                ii = np.asarray(idxs[i], dtype=int).ravel()
                good = np.isfinite(di) & (di <= max_d) & (ii >= 0) & (ii < sw_wse.shape[0])
                if not np.any(good):
                    continue
                di = di[good]
                ii = ii[good]
                vals = sw_wse[ii]
                # Robust aggregate (median after MAD rejection) to reduce sensitivity to
                # local outliers or mixed-quality observations near confluences.
                wse_out[i] = _robust_median_mad(vals, float(cfg.swot_outlier_mad_z))
                dist_out[i] = float(np.nanmin(di)) if di.size else np.nan

            sw["swot_wse_m"] = wse_out
            sw["swot_dist_m"] = dist_out
        else:
            # no usable observations
            pass

# Respect distance threshold
    joined.loc[joined["_dist_m"] > float(cfg.swot_max_dist_m), "_wse_m"] = np.nan
    joined.loc[joined["_dist_m"] > float(cfg.swot_max_dist_m), "_dist_m"] = np.nan

    sw = joined[["xs_id", "_wse_m", "_dist_m"]].copy()
    sw = sw.rename(columns={"_wse_m": "swot_wse_m", "_dist_m": "swot_dist_m"})
    sw["swot_wse_m"] = pd.to_numeric(sw["swot_wse_m"], errors="coerce")

    # Optional along-channel outlier filtering (MAD)
    try:
        tmp = xs_param[["xs_id", "component_id", "s_center_m"]].copy()
        tmp["xs_id"] = tmp["xs_id"].astype(str)
        tmp = tmp.merge(sw, on="xs_id", how="left")
        tmp = tmp.sort_values(["component_id", "s_center_m", "xs_id"]).reset_index(drop=True)
        tmp["_wse_smooth"] = tmp.groupby("component_id", dropna=False)["swot_wse_m"].apply(
            lambda s: _rolling_smooth(s, int(max(3, cfg.wse_profile_window)))
        ).reset_index(level=0, drop=True)
        resid = tmp["swot_wse_m"] - tmp["_wse_smooth"]
        med = resid.groupby(tmp["component_id"]).transform("median")
        mad = (resid - med).abs().groupby(tmp["component_id"]).transform("median")
        scale = 1.4826 * mad
        z = (resid - med).abs() / scale.replace(0, np.nan)
        tmp.loc[z > float(cfg.swot_outlier_mad_z), "swot_wse_m"] = np.nan
        sw = tmp[["xs_id", "swot_wse_m", "swot_dist_m"]]
    except Exception:
        log.debug("Optional step failed; continuing.", exc_info=True)

    xs_param = xs_param.merge(sw, on="xs_id", how="left")
    xs_param["swot_wse_m"] = xs_param["swot_wse_m"] + float(cfg.swot_vertical_offset_m)

    # Blend where SWOT WSE is finite
    blend = float(np.clip(cfg.swot_stage_blend, 0.0, 1.0))
    has = np.isfinite(xs_param["swot_wse_m"])
    xs_param.loc[has, "wse_src"] = "swot"
    xs_param.loc[has, "wse_m"] = (1.0 - blend) * xs_param.loc[has, "wse_proxy_m"].astype("float64") + blend * xs_param.loc[has, "swot_wse_m"].astype("float64")

    wse_by_xs = dict(zip(xs_param["xs_id"].astype(str), xs_param["wse_m"].astype("float64")))
    return xs_param, wse_by_xs
def _calibrate_dmax_from_soundings(
    xs_lines: gpd.GeoDataFrame,
    soundings: gpd.GeoDataFrame,
    wse_by_xs: Dict[str, float],
    cfg: InferConfig,
) -> pd.DataFrame:
    if soundings is None or soundings.empty:
        return pd.DataFrame(columns=["xs_id", "calib_n", "calib_depth_stat"])

    try:
        joined = gpd.sjoin_nearest(
            soundings,
            xs_lines[["xs_id", "geometry"]],
            how="inner",
            distance_col="_dist_to_xs",
        )
    except Exception as e:
        raise RuntimeError(
            "Spatial join nearest failed. Install a spatial index backend (rtree) or use shapely>=2. "
            f"Original error: {e}"
        )

    joined = joined[joined["_dist_to_xs"] <= float(cfg.calib_max_dist_m)].copy()
    if joined.empty:
        return pd.DataFrame(columns=["xs_id", "calib_n", "calib_depth_stat"])

    depths = []
    for _, r in joined.iterrows():
        xsid = str(r["xs_id"])
        wse = wse_by_xs.get(xsid, np.nan)
        if not np.isfinite(wse):
            depths.append(np.nan)
            continue

        d = r.get("_depth_m", np.nan)
        z = r.get("_z_m", np.nan)

        if np.isfinite(d):
            depth = float(d)
        elif np.isfinite(z):
            depth = float(wse - float(z))
        else:
            depth = np.nan

        depths.append(depth)

    joined["_depth_from_wse"] = depths
    joined = joined[np.isfinite(joined["_depth_from_wse"]) & (joined["_depth_from_wse"] >= 0.0)].copy()
    if joined.empty:
        return pd.DataFrame(columns=["xs_id", "calib_n", "calib_depth_stat"])

    def _stat(x: pd.Series) -> float:
        arr = x.to_numpy(dtype="float64")
        if arr.size == 0:
            return np.nan
        if cfg.calib_stat == "max":
            return float(np.nanmax(arr))
        if cfg.calib_stat == "median":
            return float(np.nanmedian(arr))
        return float(np.nanquantile(arr, 0.90))  # p90

    out = joined.groupby("xs_id")["_depth_from_wse"].agg(calib_n="count", calib_depth_stat=_stat).reset_index()
    return out


# --------------------------------------------------------------------------------------
# Raster outputs / interpolation
# --------------------------------------------------------------------------------------

def _open_template_raster(path: Path):
    ds = rasterio.open(path)
    if ds.crs is None:
        raise RuntimeError(f"Template raster has no CRS: {path}")
    return ds


def _template_pixel_size_m(template_ds: rasterio.DatasetReader) -> float:
    # best effort: if projected, use meters from transform; else approximate at mid-lat
    t = template_ds.transform
    px = float(abs(t.a))
    py = float(abs(t.e))
    if template_ds.crs and CRS.from_user_input(template_ds.crs).is_projected:
        return float(np.mean([px, py]))
    # degrees -> approximate using latitude at center of raster
    try:
        # transform center pixel to lon/lat
        cx = template_ds.width / 2.0
        cy = template_ds.height / 2.0
        lon, lat = (t * (cx, cy))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat))
        return float(np.mean([px * m_per_deg_lon, py * m_per_deg_lat]))
    except Exception:
        return 30.0


def _utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    """Return a WGS84 UTM CRS for the given lon/lat."""
    # Clamp lon into [-180, 180)
    lon = ((lon + 180.0) % 360.0) - 180.0
    zone = int(np.floor((lon + 180.0) / 6.0)) + 1
    zone = int(min(max(zone, 1), 60))
    epsg = (32600 + zone) if lat >= 0 else (32700 + zone)
    return CRS.from_epsg(epsg)


def _utm_crs_for_point(x: float, y: float, source_crs: CRS) -> CRS:
    """Return a WGS84 UTM CRS for a point, handling both geographic and projected input CRS.

    Args:
        x: X coordinate (longitude if geographic, easting if projected)
        y: Y coordinate (latitude if geographic, northing if projected)
        source_crs: CRS of the input coordinates

    Returns:
        CRS object for the appropriate UTM zone
    """
    source_crs = CRS.from_user_input(source_crs)
    if source_crs.is_geographic:
        lon, lat = x, y
    else:
        # Transform to WGS84 to get lon/lat for UTM zone calculation
        tr = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(x, y)
    return _utm_crs_from_lonlat(float(lon), float(lat))


def _build_corridor_mask_from_points(
    pts: gpd.GeoDataFrame,
    template_ds: rasterio.DatasetReader,
    buffer_m: float,
    all_touched: bool = True,
) -> np.ndarray:
    """
    Corridor mask raster (uint8 0/1) derived from buffered union of points.
    Buffer is applied in a projected CRS if template is geographic.
    """
    if pts.empty:
        return np.zeros((template_ds.height, template_ds.width), dtype="uint8")

    # work in template CRS for final rasterization
    if pts.crs is None:
        raise RuntimeError("Points CRS missing.")
    pts_t = pts.to_crs(template_ds.crs) if CRS.from_user_input(pts.crs) != CRS.from_user_input(template_ds.crs) else pts

    # choose projected CRS for buffering if needed
    crs_t = CRS.from_user_input(template_ds.crs)
    if crs_t.is_projected:
        pts_buf = pts_t.copy()
        geom = unary_union(pts_buf.geometry)
        geom_buf = geom.buffer(float(buffer_m))
        shapes = [(geom_buf, 1)]
    else:
        # estimate utm from centroid in lon/lat
        # Avoid GeoPandas unary_union deprecation (and keep compatibility across versions)
        try:
            union_geom = pts_t.geometry.union_all()
        except Exception:
            union_geom = unary_union(list(pts_t.geometry))
        c = union_geom.centroid
        lon, lat = float(c.x), float(c.y)
        utm = _utm_crs_from_lonlat(lon, lat)
        to_utm = Transformer.from_crs(crs_t, utm, always_xy=True)
        to_tpl = Transformer.from_crs(utm, crs_t, always_xy=True)

        def _tx_geom(g, tr):
            # shapely 2 has transform in shapely.ops, but keep no extra deps:
            from shapely.ops import transform as _transform
            return _transform(lambda x, y, z=None: tr.transform(x, y), g)

        geom = unary_union(pts_t.geometry)
        geom_utm = _tx_geom(geom, to_utm)
        geom_buf_utm = geom_utm.buffer(float(buffer_m))
        geom_buf = _tx_geom(geom_buf_utm, to_tpl)
        shapes = [(geom_buf, 1)]

    mask = rasterize(
        shapes=shapes,
        out_shape=(template_ds.height, template_ds.width),
        transform=template_ds.transform,
        fill=0,
        dtype="uint8",
        all_touched=bool(all_touched),
    )
    return mask


def _build_corridor_mask_from_lines(
    lines: Optional[gpd.GeoDataFrame],
    template_ds: rasterio.DatasetReader,
    buffer_m: float,
    all_touched: bool = True,
) -> Optional[np.ndarray]:
    """Corridor mask raster (uint8 0/1) derived from buffered union of line geometries.

    This is preferred over a *point*-buffer-derived mask when you want interpolation to
    fill *between* cross-sections (longitudinally along the channel). A point-buffer mask
    can become a series of disconnected "bands" when cross-section spacing exceeds the
    buffer distance.

    Returns None if no valid line geometries are provided.
    """
    if lines is None or len(lines) == 0:
        return None

    g = lines.copy()
    g = g[g.geometry.notnull()]
    if len(g) == 0:
        return None

    # Ensure we have a CRS
    if g.crs is None:
        raise ValueError("Line GeoDataFrame has no CRS; cannot build corridor mask")

    # Buffer in meters: project to local UTM, buffer, project back to raster CRS.
    try:
        centroid = _union_all(g.geometry).centroid
        utm_crs = _utm_crs_for_point(float(centroid.x), float(centroid.y), g.crs)
    except Exception:
        # Fall back to buffering in raster CRS units if UTM inference fails.
        utm_crs = None

    if utm_crs is not None:
        g_utm = g.to_crs(utm_crs)
        buffered = g_utm.geometry.buffer(float(buffer_m))
        buffered = gpd.GeoSeries(buffered, crs=utm_crs)
        poly = _union_all(buffered)
        poly = gpd.GeoSeries([poly], crs=utm_crs).to_crs(template_ds.crs).iloc[0]
    else:
        # WARNING: units may not be meters (e.g., degrees). This is a last resort.
        poly = _union_all(g.geometry.buffer(float(buffer_m)))

    mask = rasterize(
        [(mapping(poly), 1)],
        out_shape=(template_ds.height, template_ds.width),
        transform=template_ds.transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )
    return mask




def _load_channel_mask_raster(
    channel_mask_path: Path,
    template_ds: rasterio.DatasetReader,
    inside_value: int = 1,
    invert: bool = False,
) -> np.ndarray:
    """Load a DEM-aligned channel mask raster and convert to uint8 {0,1} on template grid.

    Requirements:
      * Same grid as template (shape/transform/crs). If not, caller should warp first.
    """
    with rasterio.open(channel_mask_path) as ms:
        if (ms.width != template_ds.width) or (ms.height != template_ds.height) or (ms.transform != template_ds.transform) or (ms.crs != template_ds.crs):
            raise ValueError(
                f"Channel mask raster is not aligned to template grid. "
                f"mask={channel_mask_path} "
                f"(crs={ms.crs}, shape={ms.height}x{ms.width}) vs "
                f"template (crs={template_ds.crs}, shape={template_ds.height}x{template_ds.width}). "
                f"Warp/align the mask to the DEM first."
            )
        m = ms.read(1)
        valid = ms.read_masks(1) > 0  # True where not NoData

    inside = (valid & (m == inside_value))
    if invert:
        inside = valid & (~inside)

    return inside.astype("uint8")
def _kd_tree():
    """
    Return a (TreeClass, name) using available libs.
    """
    try:
        return cKDTree, "scipy"
    except Exception:
        try:
            return KDTree, "sklearn"
        except Exception:
            return None, "none"


def _idw_interpolate_on_mask(
    pts_xy: np.ndarray,
    pts_val: np.ndarray,
    q_xy: np.ndarray,
    k: int = 12,
    power: float = 2.0,
    adaptive: bool = False,
    eps: float = 1e-6,
    pts_weight: Optional[np.ndarray] = None,
    pts_group: Optional[np.ndarray] = None,
    pts_priority: Optional[np.ndarray] = None,
    priority_delta: float = 0.0,
    priority_min_spread: float = 1.0,
) -> np.ndarray:
    """IDW / adaptive IDW interpolation for query coordinates.

    Memory-stable implementation: computes kNN + weights in chunks and writes results
    directly into the output vector (avoids allocating full n_query×k arrays).
    """
    pts_xy = np.asarray(pts_xy, dtype="float64")
    pts_val = np.asarray(pts_val, dtype="float64").reshape(-1)
    q_xy = np.asarray(q_xy, dtype="float64")

    if pts_weight is not None:
        pts_weight = np.asarray(pts_weight, dtype="float64").reshape(-1)
        if pts_weight.shape[0] != pts_val.shape[0]:
            raise ValueError("pts_weight must have same length as pts_val")
        pts_weight = np.maximum(pts_weight, 0.0)

    Tree, which = _kd_tree()
    if Tree is None:
        which = "numpy"

    n_pts = int(len(pts_xy))
    n_q = int(len(q_xy))
    if n_pts == 0 or n_q == 0:
        return np.full((n_q,), np.nan, dtype="float64")

    k_eff = int(min(max(1, int(k)), n_pts))
    out = np.full((n_q,), np.nan, dtype="float64")

    def _compute_vals_from_knn(d: np.ndarray, idx: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype="float64")
        idx = np.asarray(idx, dtype="int64")
        if d.ndim == 1:
            d = d[:, None]
            idx = idx[:, None]

        if adaptive:
            d1 = d[:, 0]
            dK = d[:, -1]
            ratio = np.clip(dK / np.maximum(d1, eps), 1.0, 10.0)
            p = 1.5 + (np.log(ratio) / np.log(10.0)) * (4.0 - 1.5)
            p = p[:, None]
            w = 1.0 / (np.power(d + eps, p))
        else:
            w = 1.0 / (np.power(d + eps, float(power)))

        if pts_weight is not None:
            w = w * pts_weight[idx]

        # Confluence guard: when multiple river branches contribute nearby control points,
        # prefer the highest-priority branch (typically main stem) to avoid circular "bullseye" artifacts.
        if pts_group is not None and pts_priority is not None:
            pg = np.asarray(pts_group)
            pp = np.asarray(pts_priority, dtype="float64").reshape(-1)
            if pg.shape[0] == pts_val.shape[0] and pp.shape[0] == pts_val.shape[0]:
                try:
                    g = pg[idx]
                    # Ignore missing/unknown groups (factorize may encode NaN as -1).
                    gv = g.astype("float64", copy=False)
                    gv[gv < 0] = np.nan
                    gmin = np.nanmin(gv, axis=1)
                    gmax = np.nanmax(gv, axis=1)
                    multi = np.isfinite(gmin) & np.isfinite(gmax) & (gmin != gmax)
                    p = pp[idx]
                    pmax = np.nanmax(p, axis=1)
                    pmin = np.nanmin(p, axis=1)
                    spread = pmax - pmin
                    apply = multi & np.isfinite(pmax) & np.isfinite(spread) & (spread >= float(priority_min_spread))
                    if np.any(apply):
                        keep = p >= (pmax[:, None] - float(priority_delta))
                        mask_keep = np.ones_like(w, dtype=bool)
                        mask_keep[apply, :] = keep[apply, :]
                        w_f = w * mask_keep
                        den_f = np.sum(w_f, axis=1)
                        bad = den_f <= eps
                        if np.any(bad):
                            # Fallback to unfiltered weights where filtering would remove all neighbors.
                            w_f[bad, :] = w[bad, :]
                        w = w_f
                except Exception:
                    pass

        v = pts_val[idx]
        sw = np.sum(w, axis=1)
        # avoid divide-by-zero
        good = np.isfinite(sw) & (sw > 0)
        outv = np.full((idx.shape[0],), np.nan, dtype="float64")
        outv[good] = np.sum(w[good] * v[good], axis=1) / sw[good]
        return outv

    # Query neighbors and compute results in chunks (prevents OOM for large n_q).
    chunk_q = 200000 if which in ("scipy", "sklearn") else 20000

    if which == "scipy":
        tree = Tree(pts_xy)
        for i0 in range(0, n_q, chunk_q):
            q = q_xy[i0 : i0 + chunk_q]
            try:
                d_blk, idx_blk = tree.query(q, k=k_eff, workers=-1)
            except TypeError:
                d_blk, idx_blk = tree.query(q, k=k_eff)
            if k_eff == 1:
                d_blk = np.asarray(d_blk).reshape(-1, 1)
                idx_blk = np.asarray(idx_blk).reshape(-1, 1)
            out[i0 : i0 + d_blk.shape[0]] = _compute_vals_from_knn(d_blk, idx_blk)

    elif which == "sklearn":
        tree = Tree(pts_xy)
        for i0 in range(0, n_q, chunk_q):
            q = q_xy[i0 : i0 + chunk_q]
            d_blk, idx_blk = tree.query(q, k=k_eff, return_distance=True)
            if k_eff == 1:
                d_blk = np.asarray(d_blk).reshape(-1, 1)
                idx_blk = np.asarray(idx_blk).reshape(-1, 1)
            out[i0 : i0 + d_blk.shape[0]] = _compute_vals_from_knn(d_blk, idx_blk)

    else:
        # numpy brute-force kNN (chunked)
        for i0 in range(0, n_q, chunk_q):
            q = q_xy[i0 : i0 + chunk_q]
            dx = q[:, None, 0] - pts_xy[None, :, 0]
            dy = q[:, None, 1] - pts_xy[None, :, 1]
            dist2 = dx * dx + dy * dy

            idx_k = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
            row = np.arange(idx_k.shape[0])[:, None]
            dist2_k = dist2[row, idx_k]
            ord_k = np.argsort(dist2_k, axis=1)
            idx_sorted = idx_k[row, ord_k]
            d_sorted = np.sqrt(dist2[row, idx_sorted])

            out[i0 : i0 + idx_sorted.shape[0]] = _compute_vals_from_knn(d_sorted, idx_sorted)

    return out

def _aniso_idw_interpolate_on_mask(
    pts_xy: np.ndarray,
    pts_val: np.ndarray,
    q_xy: np.ndarray,
    centerline: "LineString",
    k: int = 12,
    power: float = 2.0,
    along_scale_m: float = 500.0,
    cross_scale_m: float = 30.0,
    eps: float = 1e-6,
    pts_weight: Optional[np.ndarray] = None,
    pts_group: Optional[np.ndarray] = None,
    pts_priority: Optional[np.ndarray] = None,
    priority_delta: float = 0.0,
    priority_min_spread: float = 1.0,
) -> np.ndarray:
    """Anisotropic IDW using a centerline as the along-channel axis.

    Effective distance:
        d_eff = sqrt( (d_along/along_scale)^2 + (d_cross/cross_scale)^2 )

    pts_weight (optional): per-control-point multiplicative weights (>=0) applied
    to the kernel (useful for emphasizing thalweg points).
    """
    if pts_xy.size == 0 or q_xy.size == 0:
        return np.full((q_xy.shape[0],), np.nan, dtype="float64")

    if pts_weight is not None:
        pts_weight = np.asarray(pts_weight, dtype="float64").reshape(-1)
        if pts_weight.shape[0] != len(pts_val):
            raise ValueError("pts_weight must have same length as pts_val")
        pts_weight = np.maximum(pts_weight, 0.0)

    along_scale_m = float(max(1e-3, along_scale_m))
    cross_scale_m = float(max(1e-3, cross_scale_m))

    # Precompute chainage for points
    try:
        s_pts = np.array([centerline.project(Point(float(x), float(y))) for x, y in pts_xy], dtype="float64")
    except Exception:
        return _idw_interpolate_on_mask(
            pts_xy, pts_val, q_xy, k=int(k), power=float(power), adaptive=False, eps=eps, pts_weight=pts_weight,
            pts_group=pts_group,
            pts_priority=pts_priority,
            priority_delta=0.0,
            priority_min_spread=1.0,
        )

    Tree, which = _kd_tree()
    if Tree is None:
        return _idw_interpolate_on_mask(
            pts_xy, pts_val, q_xy, k=int(k), power=float(power), adaptive=False, eps=eps, pts_weight=pts_weight,
            pts_group=pts_group,
            pts_priority=pts_priority,
            priority_delta=0.0,
            priority_min_spread=1.0,
        )

    tree = Tree(pts_xy)
    n = pts_xy.shape[0]
    k = int(min(max(1, int(k)), n))
    cand = int(min(n, max(k, k * 5)))

    out = np.full((q_xy.shape[0],), np.nan, dtype="float64")

    # Query candidate neighbors in manageable chunks to reduce peak memory.
    chunk = 200000
    for i0 in range(0, q_xy.shape[0], chunk):
        q_blk = q_xy[i0 : i0 + chunk]
        try:
            d_eu, idx = tree.query(q_blk, k=cand, workers=-1)
        except TypeError:
            d_eu, idx = tree.query(q_blk, k=cand)
        if cand == 1:
            d_eu = np.asarray(d_eu).reshape(-1, 1)
            idx = np.asarray(idx).reshape(-1, 1)

        for j in range(q_blk.shape[0]):
            i = i0 + j

            ids = idx[j]
            de = d_eu[j].astype("float64")
            try:
                s_q = centerline.project(Point(float(q_xy[i, 0]), float(q_xy[i, 1])))
            except Exception:
                ids2 = ids[:k]
                de2 = de[:k]
                w = 1.0 / (np.maximum(de2, eps) ** float(power))
                if pts_weight is not None:
                    w = w * pts_weight[ids2]
                vv = pts_val[ids2]
                sw = np.sum(w)
                out[i] = float(np.sum(w * vv) / sw) if np.isfinite(sw) and sw > 0 else float(np.nan)
                continue
        
            s_p = s_pts[ids]
            d_along = np.abs(s_q - s_p)
        
            rad = de * de - d_along * d_along
            valid = rad >= 0
            d_cross = np.zeros_like(de)
            d_cross[valid] = np.sqrt(rad[valid])
            d_cross[~valid] = de[~valid]  # Penalize shortcuts
        
            d_eff = np.sqrt((d_along / along_scale_m) ** 2 + (d_cross / cross_scale_m) ** 2) + eps
        
            order = np.argsort(d_eff)[:k]
            ids2 = ids[order]
            d2 = d_eff[order]
        
            w = 1.0 / (d2 ** float(power))
            if pts_weight is not None:
                w = w * pts_weight[ids2]
            # Confluence guard: prefer highest-priority branch when multiple groups contribute.
            if pts_group is not None and pts_priority is not None:
                try:
                    g = np.asarray(pts_group)[ids2]
                    gv = g.astype('float64', copy=False)
                    gv[gv < 0] = np.nan
                    gmin = np.nanmin(gv)
                    gmax = np.nanmax(gv)
                    if np.isfinite(gmin) and np.isfinite(gmax) and (gmin != gmax):
                        p = np.asarray(pts_priority, dtype='float64').reshape(-1)[ids2]
                        pmax = np.nanmax(p)
                        pmin = np.nanmin(p)
                        if np.isfinite(pmax) and np.isfinite(pmin) and (pmax - pmin) >= float(priority_min_spread):
                            keep = p >= (pmax - float(priority_delta))
                            w_f = w * keep
                            if np.sum(w_f) > 0:
                                w = w_f
                except Exception:
                    log.debug("Optional step failed; continuing.", exc_info=True)
            vv = pts_val[ids2]
            sw = np.sum(w)
            out[i] = float(np.sum(w * vv) / sw) if np.isfinite(sw) and sw > 0 else float(np.nan)
        
    return out


def _rasterize_points_reduce(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    template_ds: rasterio.io.DatasetReader,
    nodata: float = -9999.0,
    reducer: str = "min",
) -> np.ndarray:
    """Rasterize points by taking the *minimum* value per pixel.

    This uses a vectorized implementation (faster than pandas groupby) to reduce
    multiple points falling into the same output pixel.
    """
    h, w = template_ds.height, template_ds.width
    out = np.full((h, w), float(nodata), dtype=np.float32)
    
    if gdf.empty:
        return out
        
    pts = gdf
    if CRS.from_user_input(gdf.crs) != CRS.from_user_input(template_ds.crs):
        pts = gdf.to_crs(template_ds.crs)

    xs = pts.geometry.x.to_numpy()
    ys = pts.geometry.y.to_numpy()
    vals = pd.to_numeric(pts[value_col], errors='coerce').to_numpy(dtype=float)
    
    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(vals)
    if not np.any(valid):
        return out
        
    rows, cols = rasterio.transform.rowcol(template_ds.transform, xs[valid], ys[valid])
    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    vals = vals[valid]
    
    inb = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    rows = rows[inb]
    cols = cols[inb]
    vals = vals[inb]
    
    if len(vals) == 0:
        return out
        
    flat_idx = rows * w + cols
    reducer = str(reducer).lower().strip()
    
    if reducer in ("min", "minimum", "deeper"):
        # Vectorized min using ufunc.at
        temp = np.full(h * w, np.inf, dtype=np.float32)
        np.minimum.at(temp, flat_idx, vals.astype(np.float32))
        temp[temp == np.inf] = float(nodata)
        out = temp.reshape((h, w))
    elif reducer in ("median", "med"):
        # Vectorized median via sort
        sorter = np.argsort(flat_idx)
        flat_idx_s = flat_idx[sorter]
        vals_s = vals[sorter]
        
        # Identify changes in index
        unique_indices, split_idx = np.unique(flat_idx_s, return_index=True)
        # Split values into groups
        grouped = np.split(vals_s, split_idx[1:])
        medians = np.array([np.median(g) for g in grouped], dtype=np.float32)
        
        flat_out = out.flatten()
        flat_out[unique_indices] = medians
        out = flat_out.reshape((h, w))
    else:
        # Default to min
        temp = np.full(h * w, np.inf, dtype=np.float32)
        np.minimum.at(temp, flat_idx, vals.astype(np.float32))
        temp[temp == np.inf] = float(nodata)
        out = temp.reshape((h, w))
        
    return out


def _continuous_surface(
    pts_gdf: gpd.GeoDataFrame,
    value_col: str,
    template_ds: rasterio.DatasetReader,
    method: str,
    buffer_m: float,
    k: int,
    idw_power: float,
    aniso_along_scale_m: float,
    aniso_cross_scale_m: float,
    thalweg_weight: float,
    nodata: float,
    overlap_reducer: str = "min",
    corridor_lines_gdf: Optional[gpd.GeoDataFrame] = None,
    channel_mask_raster: Path | None = None,
    channel_mask_inside_value: int = 1,
    channel_mask_invert: bool = False,
    max_query_dist_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a continuous surface raster (float32) and a mask raster (uint8) on the template grid.
    """
    h, w = template_ds.height, template_ds.width
    
    if pts_gdf.empty:
        arr = np.full((h, w), nodata, dtype="float32")
        m = np.zeros((h, w), dtype="uint8")
        return arr, m

    # Reproject points to template CRS for geometry alignment
    pts_t = pts_gdf.to_crs(template_ds.crs) if CRS.from_user_input(pts_gdf.crs) != CRS.from_user_input(template_ds.crs) else pts_gdf

    vals = pd.to_numeric(pts_t[value_col], errors="coerce").to_numpy(dtype="float64")
    good = np.isfinite(vals)
    pts_t = pts_t.loc[good].copy()
    vals = vals[good]
    if pts_t.empty:
        arr = np.full((h, w), nodata, dtype="float32")
        m = np.zeros((h, w), dtype="uint8")
        return arr, m
    # 1. Optimized Overlap Reduction (avoid allocating full H*W temporaries)
    rr_pts, cc_pts = rasterio.transform.rowcol(
        template_ds.transform,
        pts_t.geometry.x.to_numpy(dtype="float64"),
        pts_t.geometry.y.to_numpy(dtype="float64"),
    )
    rr_pts = np.asarray(rr_pts, dtype=np.int64)
    cc_pts = np.asarray(cc_pts, dtype=np.int64)

    # Filter points outside raster bounds
    in_bounds = (rr_pts >= 0) & (rr_pts < h) & (cc_pts >= 0) & (cc_pts < w)
    if not np.any(in_bounds):
        log.warning("[continuous] No control points fall inside template extent.")
        arr = np.full((h, w), nodata, dtype="float32")
        m = np.zeros((h, w), dtype="uint8")
        return arr, m

    keep_idx = np.where(in_bounds)[0]
    pts_t = pts_t.iloc[keep_idx].copy()
    rr_pts = rr_pts[in_bounds]
    cc_pts = cc_pts[in_bounds]
    vals = vals[in_bounds]

    # WALID expects thalweg emphasis; median reducers destroy that signal.
    if str(method).lower().startswith("walid") and str(overlap_reducer).lower() == "median":
        log.warning("[continuous] overlap_reducer=median is incompatible with WALID thalweg weighting; using 'min'.")
        overlap_reducer = "min"

    reducer = str(overlap_reducer).lower().strip()
    if reducer != "none":
        flat = rr_pts * w + cc_pts
        order = np.argsort(flat)
        flat_s = flat[order]
        vals_s = vals[order]

        uniq, start = np.unique(flat_s, return_index=True)
        ends = np.r_[start[1:], len(flat_s)]

        def _group_stat(fn):
            outv = np.empty((len(start),), dtype="float64")
            for i, (s, e) in enumerate(zip(start, ends)):
                outv[i] = fn(vals_s[s:e])
            return outv

        if reducer == "min":
            agg_vals = np.minimum.reduceat(vals_s, start)
            sel = []
            for s, e in zip(start, ends):
                j = s + int(np.argmin(vals_s[s:e]))
                sel.append(int(order[j]))
            pts_t = pts_t.iloc[sel].copy()

        elif reducer == "max":
            agg_vals = np.maximum.reduceat(vals_s, start)
            sel = []
            for s, e in zip(start, ends):
                j = s + int(np.argmax(vals_s[s:e]))
                sel.append(int(order[j]))
            pts_t = pts_t.iloc[sel].copy()

        elif reducer in ("mean", "avg"):
            agg_vals = _group_stat(np.mean)
            sel = [int(order[s]) for s in start]
            pts_t = pts_t.iloc[sel].copy()

        elif reducer == "median":
            agg_vals = _group_stat(np.median)
            # Median has no meaningful representative row; rebuild a minimal point set.
            agg_r = (uniq // w).astype("int64")
            agg_c = (uniq % w).astype("int64")
            xs, ys = rasterio.transform.xy(template_ds.transform, agg_r, agg_c, offset="center")
            pts_t = gpd.GeoDataFrame({value_col: agg_vals}, geometry=gpd.points_from_xy(xs, ys), crs=template_ds.crs)

        else:
            log.warning("[continuous] Unknown overlap_reducer=%r, using 'min'.", overlap_reducer)
            agg_vals = np.minimum.reduceat(vals_s, start)
            sel = []
            for s, e in zip(start, ends):
                j = s + int(np.argmin(vals_s[s:e]))
                sel.append(int(order[j]))
            pts_t = pts_t.iloc[sel].copy()

        # Stabilize geometry to pixel centers for deterministic results
        agg_r = (uniq // w).astype("int64")
        agg_c = (uniq % w).astype("int64")
        xs, ys = rasterio.transform.xy(template_ds.transform, agg_r, agg_c, offset="center")
        try:
            pts_t = pts_t.copy()
            pts_t.geometry = gpd.points_from_xy(xs, ys)
            pts_t.set_crs(template_ds.crs, inplace=True)
            pts_t[value_col] = agg_vals
        except Exception:
            log.debug("Optional step failed; continuing.", exc_info=True)

        vals = np.asarray(agg_vals, dtype="float64")

    # Corridor mask (where we are allowed to interpolate)
    if channel_mask_raster is not None:
        mask = _load_channel_mask_raster(
            Path(channel_mask_raster),
            template_ds,
            inside_value=int(channel_mask_inside_value),
            invert=bool(channel_mask_invert),
        )
    else:
        # Prefer a corridor derived from cross-section (or channel) linework when available.
        # Building the mask solely from point buffers can leave gaps between cross-sections.
        mask = None
        if corridor_lines_gdf is not None and len(corridor_lines_gdf) > 0:
            try:
                mask = _build_corridor_mask_from_lines(corridor_lines_gdf, template_ds, buffer_m=float(buffer_m), all_touched=True)
            except Exception as e:
                log.warning("[RIVER][MASK] failed to build corridor mask from lines; falling back to point-buffer mask: %s", e)
                mask = None

        if mask is None:
            mask = _build_corridor_mask_from_points(pts_t, template_ds, buffer_m=float(buffer_m), all_touched=True)

    # Query coordinates: pixel centers where mask==1
    rr, cc = np.where(mask == 1)
    if rr.size == 0:
        arr = np.full((h, w), nodata, dtype="float32")
        return arr, mask

    xs_q, ys_q = rasterio.transform.xy(template_ds.transform, rr, cc, offset="center")
    q_xy_tpl = np.vstack([np.asarray(xs_q, dtype="float64"), np.asarray(ys_q, dtype="float64")]).T

    # 3. Robust CRS Handling (Fix CRS Mismatch)
    crs_tpl = CRS.from_user_input(template_ds.crs)
    
    # Variables to hold the UTM-projected data
    pts_xy_utm = None
    q_xy_utm = None
    utm_crs_obj = None

    if crs_tpl.is_projected:
        pts_xy_utm = np.vstack([pts_t.geometry.x.to_numpy(dtype="float64"), pts_t.geometry.y.to_numpy(dtype="float64")]).T
        q_xy_utm = q_xy_tpl
        utm_crs_obj = crs_tpl
    else:
        # Template is geographic; project everything to UTM
        # Avoid GeoPandas unary_union deprecation
        try:
            union_geom = pts_t.geometry.union_all()
        except Exception:
            union_geom = unary_union(list(pts_t.geometry))
        c = union_geom.centroid
        utm_crs_obj = _utm_crs_from_lonlat(float(c.x), float(c.y))
        tr = Transformer.from_crs(crs_tpl, utm_crs_obj, always_xy=True)
        
        px, py = tr.transform(pts_t.geometry.x.to_numpy(dtype="float64"), pts_t.geometry.y.to_numpy(dtype="float64"))
        qx, qy = tr.transform(q_xy_tpl[:, 0], q_xy_tpl[:, 1])
        
        pts_xy_utm = np.vstack([px, py]).T
        q_xy_utm = np.vstack([qx, qy]).T

    pts_val = vals


    # Optional confluence guard inputs
    # - pts_group: integer codes for river branches (river_id)
    # - pts_priority: branch priority (stream order) so main stems dominate at junctions
    pts_group = None
    pts_priority = None

    # Group code for confluence guard. Prefer stable IDs if present.
    try:
        if "river_id" in pts_t.columns:
            s = pts_t["river_id"]
            if pd.notna(s).any():
                pts_group = pd.factorize(s.astype(str), sort=False)[0].astype("int32")
        if pts_group is None or (np.asarray(pts_group) < 0).all():
            if "component_id" in pts_t.columns:
                s = pts_t["component_id"]
                if pd.notna(s).any():
                    pts_group = pd.factorize(s.astype(str), sort=False)[0].astype("int32")
    except Exception:
        pts_group = None

    if "stream_order" in pts_t.columns:
        try:
            pts_priority = pd.to_numeric(pts_t["stream_order"], errors="coerce").to_numpy(dtype="float64")
        except Exception:
            pts_priority = None


    # Interpolate (continuous modes)
    method_l = str(method).lower()

    # Optional WALID thalweg emphasis (per-control-point weights)
    pts_weight = None
    if method_l.startswith("walid") and thalweg_weight is not None and float(thalweg_weight) > 1.0:
        if "xs_id" in pts_t.columns:
            try:
                idx_thalweg = pts_t.groupby("xs_id")[value_col].idxmin()
                pts_weight = np.ones((len(pts_t),), dtype="float64")
                pos = pts_t.index.get_indexer(idx_thalweg.to_numpy())
                pos = pos[pos >= 0]
                pts_weight[pos] = float(thalweg_weight)
            except Exception as e:
                log.warning("[continuous] Thalweg weighting failed; continuing unweighted: %s", e)
                pts_weight = None
        else:
            log.debug("[continuous] WALID requested but xs_id not present; skipping thalweg weighting.")

    if method_l in ("aniso", "walid_aniso"):
        centerline = None
        if corridor_lines_gdf is not None and len(corridor_lines_gdf) > 0:
            try:
                # Ensure centerline is in the same projected CRS (UTM)
                lines_utm = corridor_lines_gdf.to_crs(utm_crs_obj)
                merged = linemerge(unary_union(lines_utm.geometry))
                if merged is not None:
                    if hasattr(merged, "geoms"):
                        # choose longest segment
                        centerline = max(list(merged.geoms), key=lambda g: g.length)
                    else:
                        centerline = merged
            except Exception as e:
                log.warning("[RIVER][ANISO] centerline extraction failed; falling back to isotropic IDW: %s", e)
                centerline = None

        if centerline is not None:
            # We now safe pass UTM coords and UTM centerline
            vals_q = _aniso_idw_interpolate_on_mask(
                pts_xy=pts_xy_utm,
                pts_val=pts_val,
                q_xy=q_xy_utm,
                centerline=centerline,
                k=int(k),
                power=float(idw_power),
                along_scale_m=float(aniso_along_scale_m),
                cross_scale_m=float(aniso_cross_scale_m),
                eps=1e-6,
                pts_weight=pts_weight,
                pts_group=pts_group,
                pts_priority=pts_priority,
                priority_delta=0.0,
                priority_min_spread=1.0,
            )
        else:
            vals_q = _idw_interpolate_on_mask(
                pts_xy=pts_xy_utm,
                pts_val=pts_val,
                q_xy=q_xy_utm,
                k=int(k),
                power=float(idw_power),
                adaptive=False,
                eps=1e-6,
                pts_weight=pts_weight,
                pts_group=pts_group,
                pts_priority=pts_priority,
                priority_delta=0.0,
                priority_min_spread=1.0,
            )
    else:
        adaptive = (method_l == "aidw")
        vals_q = _idw_interpolate_on_mask(
            pts_xy=pts_xy_utm,
            pts_val=pts_val,
            q_xy=q_xy_utm,
            k=int(k),
            power=float(idw_power),
            adaptive=adaptive,
            eps=1e-6,
            pts_weight=pts_weight,
            pts_group=pts_group,
            pts_priority=pts_priority,
            priority_delta=0.0,
            priority_min_spread=1.0,
        )

    # Optionally prune predictions that are too far from any control point
    vals_q = np.asarray(vals_q, dtype="float64").reshape(-1)
    mask_out = mask.astype("uint8").copy()

    if max_query_dist_m is not None:
        try:
            maxd = float(max_query_dist_m)
        except Exception:
            maxd = float('nan')
        if np.isfinite(maxd) and maxd > 0:
            Tree, which = _kd_tree()
            dmin = np.empty((q_xy_utm.shape[0],), dtype="float64")
            if Tree is not None and which == "scipy":
                tree = Tree(pts_xy_utm)
                try:
                    dmin[:] = tree.query(q_xy_utm, k=1, workers=-1)[0]
                except TypeError:
                    dmin[:] = tree.query(q_xy_utm, k=1)[0]
            elif Tree is not None and which == "sklearn":
                tree = Tree(pts_xy_utm)
                d, _ = tree.query(q_xy_utm, k=1, return_distance=True)
                dmin[:] = np.asarray(d).reshape(-1)
            else:
                # Numpy brute-force nearest distance (chunked)
                chunk = 50000
                for s in range(0, q_xy_utm.shape[0], chunk):
                    e = min(q_xy_utm.shape[0], s + chunk)
                    q = q_xy_utm[s:e]
                    # (m,1,2) - (1,n,2) -> (m,n,2)
                    diff = q[:, None, :] - pts_xy_utm[None, :, :]
                    d2 = np.sum(diff * diff, axis=2)
                    dmin[s:e] = np.sqrt(np.min(d2, axis=1))

            far = dmin > maxd
            if np.any(far):
                mask_out[rr[far], cc[far]] = 0
                vals_q[far] = np.nan

    good_q = np.isfinite(vals_q)
    out = np.full((h, w), nodata, dtype="float32")
    if np.any(good_q):
        out[rr[good_q], cc[good_q]] = vals_q[good_q].astype("float32")

    # ---------------------------------------------------------------------
    # Junction/confluence artifact suppression (best-effort)
    #
    # Near tributary mouths, mixing control points from different branches can
    # create circular "bullseye" artifacts on the main stem. When a river
    # network graph is available (graph_nodes with degree >= 3) and per-point
    # stream order is present, overwrite predictions in a small junction zone
    # using an interpolation driven by main-stem-only control points.
    #
    # This is a conservative selection rule in a limited neighborhood; it does
    # not introduce new modeling assumptions.
    # ---------------------------------------------------------------------
    try:
        river_gpkg = getattr(_continuous_surface, "_river_gpkg", None)
    except Exception:
        river_gpkg = None

    do_junction_fix = False
    try:
        do_junction_fix = (
            river_gpkg is not None
            and pts_priority is not None
            and np.isfinite(np.nanmax(np.asarray(pts_priority, dtype="float64")))
            and Path(str(river_gpkg)).exists()
        )
    except Exception:
        do_junction_fix = False

    if do_junction_fix:
        try:
            import geopandas as gpd
            from rasterio.features import rasterize

            rg = Path(str(river_gpkg))
            nodes = None
            for lyr in ("graph_nodes", "nodes", "river_nodes"):
                try:
                    g = gpd.read_file(rg, layer=lyr)
                    if g is not None and len(g) > 0:
                        nodes = g
                        break
                except Exception:
                    continue
            if nodes is None or len(nodes) == 0:
                raise RuntimeError("no nodes")

            keep = None
            for c in ("degree", "deg", "node_degree"):
                if c in nodes.columns:
                    deg = pd.to_numeric(nodes[c], errors="coerce").fillna(0.0)
                    keep = deg >= 3.0
                    break
            if keep is None:
                if "node_type" in nodes.columns:
                    t = nodes["node_type"].astype(str).str.lower()
                    keep = t.str.contains("conflu") | t.str.contains("junction")
                else:
                    keep = pd.Series(False, index=nodes.index)
            nodes = nodes.loc[keep]
            if len(nodes) == 0:
                raise RuntimeError("no junction nodes")

            nodes_utm = nodes.to_crs(utm_crs_obj)

            try:
                px = float(abs(template_ds.transform.a))
            except Exception:
                px = 10.0
            r_m = max(40.0, 4.0 * px)

            geoms = [geom.buffer(r_m) for geom in nodes_utm.geometry if geom is not None and not geom.is_empty]
            if not geoms:
                raise RuntimeError("no buffered geometries")

            jmask = rasterize(
                [(g, 1) for g in geoms],
                out_shape=(h, w),
                transform=template_ds.transform,
                fill=0,
                dtype="uint8",
                all_touched=True,
            )
            jmask = (jmask == 1) & (mask_out == 1)
            if not np.any(jmask):
                raise RuntimeError("junction mask empty")

            if "component_id" in pts_t.columns:
                comp = pd.to_numeric(pts_t["component_id"], errors="coerce")
                comp_code = pd.factorize(comp.astype("Int64").astype(str), sort=False)[0].astype("int32")
            elif pts_group is not None:
                comp_code = np.asarray(pts_group, dtype="int32")
            else:
                comp_code = np.zeros((len(pts_t),), dtype="int32")

            pr = np.asarray(pts_priority, dtype="float64")
            keep_main = np.zeros((len(pr),), dtype=bool)
            for cid in np.unique(comp_code):
                if cid < 0:
                    continue
                m = comp_code == cid
                if not np.any(m):
                    continue
                mx = np.nanmax(pr[m])
                if not np.isfinite(mx):
                    continue
                keep_main |= (m & (pr >= (mx - 0.5)))

            if int(np.sum(keep_main)) < 6:
                raise RuntimeError("mainstem controls too sparse")

            pts_xy_m = pts_xy_utm[keep_main]
            pts_val_m = pts_val[keep_main]
            pts_w_m = pts_weight[keep_main] if pts_weight is not None else None

            vals_q_m = _idw_interpolate_on_mask(
                pts_xy=pts_xy_m,
                pts_val=pts_val_m,
                q_xy=q_xy_utm,
                k=int(min(int(k), 24)),
                power=float(idw_power),
                adaptive=False,
                eps=1e-6,
                pts_weight=pts_w_m,
                pts_group=None,
                pts_priority=None,
            )
            vals_q_m = np.asarray(vals_q_m, dtype="float64").reshape(-1)
            good_m = np.isfinite(vals_q_m)
            if not np.any(good_m):
                raise RuntimeError("no mainstem predictions")

            out_m = np.full((h, w), nodata, dtype="float32")
            out_m[rr[good_m], cc[good_m]] = vals_q_m[good_m].astype("float32")

            overwrite = jmask & np.isfinite(out_m)
            if np.any(overwrite):
                out[overwrite] = out_m[overwrite]
        except Exception:
            log.debug("Optional step failed; continuing.", exc_info=True)


    # ---------------------------------------------------------------------
    # Main-stem corridor overwrite (tributary-first, then mainstem override)
    #
    # The most reliable way to eliminate circular artifacts at tributary mouths
    # is to prevent mixed-branch control points from influencing the *mainstem*
    # surface along its corridor. We therefore:
    #   1) interpolate using all control points (tributaries included) -> out
    #   2) interpolate using mainstem-only control points -> out_m
    #   3) overwrite out with out_m inside a buffered mainstem corridor mask
    #
    # This keeps small-stream structure where it belongs, while ensuring the
    # main stem remains continuous and dominant through confluences.
    # ---------------------------------------------------------------------
    if do_junction_fix:
        try:
            import geopandas as gpd
            from rasterio.features import rasterize

            rg = Path(str(river_gpkg))

            # Try to load linework representing the river network.
            lines = None
            for lyr in ("graph_edges", "rivers_clip", "rivers", "flowlines", "river_network"):
                try:
                    g = gpd.read_file(rg, layer=lyr)
                    if g is not None and len(g) > 0:
                        lines = g
                        break
                except Exception:
                    continue
            if lines is None or len(lines) == 0:
                raise RuntimeError("no river linework")

            lines_utm = lines.to_crs(utm_crs_obj)

            # Estimate pixel size in meters for buffer defaults.
            try:
                px = float(abs(template_ds.transform.a))
            except Exception:
                px = 10.0
            buf_m = max(30.0, 3.0 * px)

            # Choose a stream order / priority field on the linework if present.
            so_field = None
            for c in ("streamorde", "stream_order", "streamorder", "order", "StrmOrdr", "StreamOrde"):
                if c in lines_utm.columns:
                    so_field = c
                    break

            if so_field is not None:
                so = pd.to_numeric(lines_utm[so_field], errors="coerce")
            else:
                so = pd.Series(np.nan, index=lines_utm.index, dtype="float64")

            # Component grouping (optional).
            comp_field = None
            for c in ("component_id", "component", "comp_id"):
                if c in lines_utm.columns:
                    comp_field = c
                    break
            if comp_field is not None:
                comp = pd.to_numeric(lines_utm[comp_field], errors="coerce").fillna(-1).astype("int32")
            else:
                comp = pd.Series(0, index=lines_utm.index, dtype="int32")

            # Select mainstem linework inside each component:
            # - if stream order exists: edges with order >= max(order)-0.5
            # - else: fall back to the longest 20% of segments by length
            seg_len = lines_utm.geometry.length
            keep_edge = np.zeros((len(lines_utm),), dtype=bool)
            comp_arr = np.asarray(comp, dtype="int32")
            so_arr = np.asarray(so, dtype="float64")
            len_arr = np.asarray(seg_len, dtype="float64")
            for cid in np.unique(comp_arr):
                m = comp_arr == cid
                if not np.any(m):
                    continue

                mx = np.nanmax(so_arr[m])
                if np.isfinite(mx):
                    # Prefer the highest stream-order edges within this component.
                    keep_edge |= (m & (so_arr >= (mx - 0.5)))
                    continue

                # No usable stream-order: choose a defensible "mainstem" by graph diameter
                # (longest path) in an undirected graph built from segment endpoints.
                # This is more stable at confluences than "longest segments" heuristics.
                try:
                    idxs = np.where(m)[0].tolist()
                    node_id = {}
                    nodes = []
                    adj = {}  # nid -> list of (nbr, weight, edge_index)

                    def _get_nid(xy):
                        key = (round(float(xy[0]), 3), round(float(xy[1]), 3))
                        if key in node_id:
                            return node_id[key]
                        nid = len(nodes)
                        node_id[key] = nid
                        nodes.append(key)
                        adj[nid] = []
                        return nid

                    for ei in idxs:
                        g = lines_utm.geometry.iloc[ei]
                        if g is None or g.is_empty:
                            continue
                        try:
                            coords = list(g.coords)
                        except Exception:
                            try:
                                coords = list(list(g.geoms[0].coords)) + list(list(g.geoms[-1].coords))
                            except Exception:
                                continue
                        if len(coords) < 2:
                            continue
                        a = coords[0]
                        b = coords[-1]
                        na = _get_nid(a)
                        nb = _get_nid(b)
                        wgt = float(len_arr[ei]) if np.isfinite(len_arr[ei]) else float(g.length)
                        if wgt <= 0:
                            continue
                        adj[na].append((nb, wgt, ei))
                        adj[nb].append((na, wgt, ei))

                    if len(nodes) < 2:
                        raise RuntimeError("graph too small")

                    import heapq

                    def _dijkstra(src):
                        dist = {src: 0.0}
                        prev = {}  # node -> (prev_node, edge_index)
                        pq = [(0.0, src)]
                        while pq:
                            d, u = heapq.heappop(pq)
                            if d != dist.get(u, None):
                                continue
                            for v, w, ei in adj.get(u, []):
                                nd = d + w
                                if nd < dist.get(v, 1e300):
                                    dist[v] = nd
                                    prev[v] = (u, ei)
                                    heapq.heappush(pq, (nd, v))
                        far = max(dist.items(), key=lambda kv: kv[1])
                        return far[0], far[1], prev

                    src0 = next((nid for nid, neis in adj.items() if neis), 0)
                    a_node, _, _ = _dijkstra(src0)
                    b_node, _, prev = _dijkstra(a_node)

                    # reconstruct diameter path edges from b back to a
                    path_edges = []
                    cur = b_node
                    seen = set()
                    while cur != a_node and cur in prev and cur not in seen:
                        seen.add(cur)
                        pu, ei = prev[cur]
                        path_edges.append(ei)
                        cur = pu

                    if path_edges:
                        keep_edge |= np.isin(np.arange(len(lines_utm)), np.array(path_edges, dtype=int))
                    else:
                        q = np.nanquantile(len_arr[m], 0.80)
                        keep_edge |= (m & (len_arr >= q))
                except Exception:
                    q = np.nanquantile(len_arr[m], 0.80)
                    keep_edge |= (m & (len_arr >= q))

            if int(np.sum(keep_edge)) == 0:
                raise RuntimeError("no mainstem edges selected")

                        # Build a mainstem corridor mask.
            #
            # Preferred (robust) approach: rasterize the mainstem *centerline* and
            # construct a corridor using distance transforms, bounded by the local
            # channel half-width estimated from the channel mask. This avoids
            # narrow/overly-wide fixed buffers and reduces confluence "bullseye"
            # artifacts by ensuring the overwrite covers the full mainstem width.
            #
            # Fallback: fixed-width geometric buffer if SciPy distance transforms
            # are unavailable.
            line_geoms = []
            for geom in lines_utm.loc[keep_edge].geometry:
                if geom is None or geom.is_empty:
                    continue
                line_geoms.append(geom)

            if not line_geoms:
                raise RuntimeError("no mainstem line geometries")

            # Rasterize mainstem linework to the template grid
            mline = rasterize(
                [(g, 1) for g in line_geoms],
                out_shape=(h, w),
                transform=template_ds.transform,
                fill=0,
                dtype="uint8",
                all_touched=True,
            )

            mmask = None
            try:
                from scipy.ndimage import distance_transform_edt

                # Pixel sizes (x,y) in meters for sampling
                try:
                    px = float(abs(template_ds.transform.a))
                except Exception:
                    px = 10.0
                try:
                    py = float(abs(template_ds.transform.e))
                except Exception:
                    py = px

                # Distance to mainstem line (meters)
                dist_to_line = distance_transform_edt(mline == 0, sampling=(py, px))

                # Local channel half-width proxy (meters): distance to channel edge
                # for pixels inside the channel mask.
                halfw = distance_transform_edt((mask_out == 1), sampling=(py, px))

                # Corridor radius: at least a few pixels, but bounded by local half-width.
                min_r = max(30.0, 3.0 * px)
                rad = np.maximum(min_r, 0.90 * halfw)

                mmask = (mask_out == 1) & (dist_to_line <= rad)
            except Exception:
                # Fallback: fixed-width geometric buffer
                geoms = []
                for geom in line_geoms:
                    try:
                        geoms.append(geom.buffer(buf_m))
                    except Exception:
                        continue
                if not geoms:
                    raise RuntimeError("no buffered mainstem geometries")
                mm = rasterize(
                    [(g, 1) for g in geoms],
                    out_shape=(h, w),
                    transform=template_ds.transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=True,
                )
                mmask = (mm == 1) & (mask_out == 1)

            if not np.any(mmask):
                raise RuntimeError("mainstem corridor mask empty")

            # Recompute mainstem-only control point selection (same rule as junction fix).
            if "component_id" in pts_t.columns:
                comp_p = pd.to_numeric(pts_t["component_id"], errors="coerce")
                comp_code = pd.factorize(comp_p.astype("Int64").astype(str), sort=False)[0].astype("int32")
            elif pts_group is not None:
                comp_code = np.asarray(pts_group, dtype="int32")
            else:
                comp_code = np.zeros((len(pts_t),), dtype="int32")

            pr = np.asarray(pts_priority, dtype="float64")
            keep_main = np.zeros((len(pr),), dtype=bool)
            for cid in np.unique(comp_code):
                if cid < 0:
                    continue
                m = comp_code == cid
                if not np.any(m):
                    continue
                mx = np.nanmax(pr[m])
                if not np.isfinite(mx):
                    continue
                keep_main |= (m & (pr >= (mx - 0.5)))

            if int(np.sum(keep_main)) < 6:
                raise RuntimeError("mainstem controls too sparse")

            pts_xy_m = pts_xy_utm[keep_main]
            pts_val_m = pts_val[keep_main]
            pts_w_m = pts_weight[keep_main] if pts_weight is not None else None

            vals_q_m = _idw_interpolate_on_mask(
                pts_xy=pts_xy_m,
                pts_val=pts_val_m,
                q_xy=q_xy_utm,
                k=int(min(int(k), 24)),
                power=float(idw_power),
                adaptive=False,
                eps=1e-6,
                pts_weight=pts_w_m,
                pts_group=None,
                pts_priority=None,
            )
            vals_q_m = np.asarray(vals_q_m, dtype="float64").reshape(-1)
            good_m = np.isfinite(vals_q_m)
            if not np.any(good_m):
                raise RuntimeError("no mainstem predictions")

            out_m = np.full((h, w), nodata, dtype="float32")
            out_m[rr[good_m], cc[good_m]] = vals_q_m[good_m].astype("float32")

            overwrite = mmask & np.isfinite(out_m)
            if np.any(overwrite):
                # Feather the overwrite to avoid seams at the corridor boundary.
                # If SciPy is unavailable, fall back to a hard overwrite.
                try:
                    from scipy.ndimage import distance_transform_edt  # type: ignore

                    # Pixel sizes (x,y) in meters
                    try:
                        px = float(abs(template_ds.transform.a))
                    except Exception:
                        px = 10.0
                    try:
                        py = float(abs(template_ds.transform.e))
                    except Exception:
                        py = px

                    feather_m = max(20.0, 2.0 * px)

                    # Distance-to-boundary inside overwrite zone (meters)
                    d_in = distance_transform_edt(overwrite.astype(np.uint8), sampling=(py, px)).astype("float64")

                    wgt = np.clip(d_in / feather_m, 0.0, 1.0).astype("float32")

                    base = out.astype("float64", copy=True)
                    base[base == nodata] = np.nan
                    mainv = out_m.astype("float64", copy=False)
                    mainv[mainv == nodata] = np.nan

                    m = overwrite & np.isfinite(mainv)
                    if np.any(m):
                        out_blend = base
                        # Where base is missing, take mainstem directly
                        miss = m & (~np.isfinite(base))
                        out_blend[miss] = mainv[miss]

                        # Blend where both are present
                        both = m & np.isfinite(base)
                        if np.any(both):
                            ww = wgt[both].astype("float64")
                            out_blend[both] = out_blend[both] * (1.0 - ww) + mainv[both] * ww

                        out = out_blend.astype("float32", copy=False)
                        out[~np.isfinite(out)] = nodata
                except Exception:
                    out[overwrite] = out_m[overwrite]
        except Exception:
            log.debug("Optional step failed; continuing.", exc_info=True)

    return out, mask_out



def _exists_with_retry(path: Path, tries: int = 10, sleep_s: float = 0.2, min_size_bytes: int = 1) -> bool:
    """Best-effort existence check for networked / delayed filesystems."""
    for i in range(max(1, int(tries))):
        try:
            if path.exists():
                if min_size_bytes <= 0:
                    return True
                try:
                    if path.stat().st_size >= min_size_bytes:
                        return True
                except FileNotFoundError:
                    pass
        except Exception:
            log.debug("Optional step failed; continuing.", exc_info=True)
        time.sleep(sleep_s * (1.0 + 0.15 * i))
    return False


def _fsync_dir(path: Path) -> None:
    """Flush directory metadata (helps on some filesystems)."""
    if os.name != "posix":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        return


def _write_geotiff_gdal(path: Path, arr: np.ndarray, tmpl: rasterio.io.DatasetReader, nodata: float, dtype: str) -> None:
    """Write a GeoTIFF using GDAL directly (fallback path)."""
    try:
        from osgeo import gdal  # type: ignore
    except Exception as e:
        raise RuntimeError(f"GDAL python bindings not available for fallback write: {e}") from e

    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("GDAL GTiff driver not available")

    # Map dtype → GDAL type
    dt = np.dtype(dtype)
    gdal_type_map = {
        np.dtype("uint8"): gdal.GDT_Byte,
        np.dtype("int16"): gdal.GDT_Int16,
        np.dtype("uint16"): gdal.GDT_UInt16,
        np.dtype("int32"): gdal.GDT_Int32,
        np.dtype("uint32"): gdal.GDT_UInt32,
        np.dtype("float32"): gdal.GDT_Float32,
        np.dtype("float64"): gdal.GDT_Float64,
    }
    gdal_dt = gdal_type_map.get(dt, gdal.GDT_Float32)

    opts = [
        "COMPRESS=DEFLATE",
        "TILED=YES",
        "BLOCKXSIZE=256",
        "BLOCKYSIZE=256",
        "BIGTIFF=IF_SAFER",
    ]

    ds = driver.Create(str(path), int(tmpl.width), int(tmpl.height), 1, gdal_dt, options=opts)
    if ds is None:
        raise RuntimeError(f"GDAL failed to create output dataset: {path}")

    # GeoTransform / projection
    try:
        ds.SetGeoTransform(tmpl.transform.to_gdal())
    except Exception:
        # As a fallback, try the tuple form
        ds.SetGeoTransform(tuple(tmpl.transform)[:6])

    if tmpl.crs is not None:
        try:
            ds.SetProjection(tmpl.crs.to_wkt())
        except Exception:
            log.debug("Optional step failed; continuing.", exc_info=True)

    band = ds.GetRasterBand(1)
    try:
        band.SetNoDataValue(float(nodata))
    except Exception:
        log.debug("Optional step failed; continuing.", exc_info=True)

    band.WriteArray(arr.astype(dtype, copy=False))
    band.FlushCache()
    ds.FlushCache()
    ds = None


def _write_geotiff(path: Path, arr: np.ndarray, tmpl: rasterio.io.DatasetReader, nodata: float, dtype: str = "float32") -> None:
    """Write a GeoTIFF robustly (atomic write + retry + GDAL fallback)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(arr)
    try:
        arr = arr.astype(dtype, copy=False)
    except Exception:
        arr = arr.astype("float32", copy=False)
        dtype = "float32"

    # Write to a temp path in the same directory, then atomically replace.
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with contextlib.suppress(Exception):
        if tmp.exists():
            tmp.unlink()

    # 1) Try rasterio first
    wrote = False
    last_err: Exception | None = None
    try:
        profile = tmpl.profile.copy()
        profile.update(
            driver="GTiff",
            dtype=dtype,
            count=1,
            compress="deflate",
            nodata=nodata,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(arr, 1)
        # Some small or highly-compressible rasters (e.g., uint8 masks) can be <1KB.
        # Existence is the reliable signal here; size thresholds cause false negatives.
        wrote = _exists_with_retry(tmp, tries=8, sleep_s=0.15, min_size_bytes=1)
    except Exception as e:
        last_err = e
        wrote = False

    # 2) If rasterio didn't produce a file, try GDAL direct
    if not wrote:
        with contextlib.suppress(Exception):
            if tmp.exists():
                tmp.unlink()
        try:
            _write_geotiff_gdal(tmp, arr, tmpl, nodata=nodata, dtype=dtype)
            wrote = _exists_with_retry(tmp, tries=8, sleep_s=0.15, min_size_bytes=1)
        except Exception:
            if last_err is not None:
                log.error("[WRITE] rasterio write failed: %s", last_err)
            raise

    if not wrote:
        # Last resort: dump directory listing for debugging
        try:
            parent_files = ", ".join(sorted([f.name for f in path.parent.iterdir() if f.is_file()])[:50])
        except Exception:
            parent_files = "<unavailable>"
        raise RuntimeError(f"GeoTIFF write produced no file: tmp={tmp} | parent_files={parent_files}")

    # Atomic replace
    os.replace(str(tmp), str(path))
    _fsync_dir(path.parent)

    # Verify final output (network FS can delay metadata visibility)
    if not _exists_with_retry(path, tries=12, sleep_s=0.15, min_size_bytes=1):
        try:
            parent_files = ", ".join(sorted([f.name for f in path.parent.iterdir() if f.is_file()])[:50])
        except Exception:
            parent_files = "<unavailable>"
        raise RuntimeError(f"GeoTIFF missing after atomic replace: {path} | parent_files={parent_files}")



def _make_mask(arr: np.ndarray, nodata: float) -> np.ndarray:
    m = np.isfinite(arr) & (arr != nodata)
    return m.astype("uint8")


# --------------------------------------------------------------------------------------
# Core inference
# --------------------------------------------------------------------------------------

def infer_bathy(
    xs_gpkg: Path,
    out_gpkg: Path,
    dem_path: Optional[Path],
    soundings_path,
    soundings_depth_col: Optional[str],
    soundings_elev_col: Optional[str],
    soundings_x_col: Optional[str],
    soundings_y_col: Optional[str],
    soundings_crs: Optional[str],
    cfg: InferConfig,
    xs_lines_layer: str = "xs_lines",
    xs_points_layer: str = "xs_points",
    # optional reach attributes / calibration anchors
    river_gpkg: Optional[Path] = None,
    rivers_layer: str = "rivers_clip",
    drain_area_field: Optional[str] = None,
    slope_field: Optional[str] = None,
    manning_q_field: Optional[str] = None,
    manning_dist_to_mouth_field: Optional[str] = None,
    usgs_sites: Optional[List[str]] = None,
    usgs_start: Optional[str] = None,
    usgs_end: Optional[str] = None,
    usgs_cache_dir: Optional[Path] = None,
    width_stage_csv: Optional[List[str]] = None,

    # raster outputs
    template_raster: Optional[Path] = None,
    out_bathy_raster: Optional[Path] = None,
    out_mask_raster: Optional[Path] = None,
    out_uncert_raster: Optional[Path] = None,
    # rasterization options
    raster_value_col: str = "z_bed_pred_m",
    raster_uncert_col: str = "uncert_m",
    continuous: str = "walid",
    continuous_buffer_m: Optional[float] = None,
    continuous_k: int = 12,
    idw_power: float = 2.0,
    aniso_along_scale_m: float = 500.0,
    aniso_cross_scale_m: float = 30.0,
    thalweg_weight: float = 6.0,
    nodata: float = -9999.0,
    overlap_reducer: str = "min",
    channel_mask_raster: Optional[Path] = None,
    channel_mask_inside_value: int = 1,
    channel_mask_invert: bool = False,
    max_query_dist_m: Optional[float] = None,
    thalweg_only: bool = False,
    thalweg_densify_step_m: Optional[float] = None,
    out_accounting_json: Optional[Path] = None,
    out_meta_json: Optional[Path] = None,
) -> None:
    xs_lines = _read_layer(xs_gpkg, xs_lines_layer)
    xs_pts = _read_layer(xs_gpkg, xs_points_layer)

    if "xs_id" not in xs_lines.columns or "xs_id" not in xs_pts.columns:
        raise RuntimeError("xs_lines/xs_points must have 'xs_id' column (from xs_builder.py).")

    xs_lines = xs_lines.copy()
    xs_lines["xs_len_m"] = xs_lines.geometry.length

    # ensure bank cols exist
    for col in ["bank_left_z_m", "bank_right_z_m", "bank_left_dist_m", "bank_right_dist_m", "s_center_m", "component_id", "river_id"]:
        if col not in xs_lines.columns:
            xs_lines[col] = np.nan

    pts_df = xs_pts.drop(columns=["geometry"]).copy()
    for c in ["dist_m", "z_dem", "z_topo", "is_bank_left", "is_bank_right", "river_id", "component_id"]:
        if c not in pts_df.columns:
            pts_df[c] = np.nan

    grouped = pts_df.groupby("xs_id", sort=False)

    # Constraint accounting (to expose where the model is under-constrained)
    acct: Dict[str, object] = {
        "prior_mode": str(cfg.prior_mode or "width_power"),
        "dmin_m": cfg.dmin_m,
        "dmax_m": cfg.dmax_m,
        "manning_mode": str(cfg.manning_mode or "off"),
        "n_xs_total": 0,
        "n_prior_total": 0,
        "n_prior_clipped_min": 0,
        "n_prior_clipped_max": 0,
        "n_with_da": 0,
        "n_with_slope": 0,
    }

    xs_records = []
    wse_by_xs: Dict[str, float] = {}

    for _, xsl in xs_lines.iterrows():
        acct["n_xs_total"] = int(acct.get("n_xs_total", 0)) + 1
        xsid = str(xsl["xs_id"])
        if xsid not in grouped.groups:
            continue

        xsp = grouped.get_group(xsid).copy().sort_values("dist_m")
        xs_len = float(xsl["xs_len_m"])

        bank_left_dist = _safe_float(xsl.get("bank_left_dist_m", np.nan))
        bank_right_dist = _safe_float(xsl.get("bank_right_dist_m", np.nan))
        if not np.isfinite(bank_left_dist) or not np.isfinite(bank_right_dist):
            bl, br = _pick_bank_dists_from_points(xsp)
            if not np.isfinite(bank_left_dist):
                bank_left_dist = bl if bl is not None else np.nan
            if not np.isfinite(bank_right_dist):
                bank_right_dist = br if br is not None else np.nan

        bank_left_z = _safe_float(xsl.get("bank_left_z_m", np.nan))
        bank_right_z = _safe_float(xsl.get("bank_right_z_m", np.nan))
        if not np.isfinite(bank_left_z) or not np.isfinite(bank_right_z):
            if "is_bank_left" in xsp.columns and xsp["is_bank_left"].any():
                r = xsp.loc[xsp["is_bank_left"] == True].iloc[0]
                bank_left_z = float(r["z_topo"]) if np.isfinite(r.get("z_topo", np.nan)) else float(r.get("z_dem", np.nan))
            if "is_bank_right" in xsp.columns and xsp["is_bank_right"].any():
                r = xsp.loc[xsp["is_bank_right"] == True].iloc[0]
                bank_right_z = float(r["z_topo"]) if np.isfinite(r.get("z_topo", np.nan)) else float(r.get("z_dem", np.nan))

        if cfg.only_with_banks and (not np.isfinite(bank_left_dist) or not np.isfinite(bank_right_dist)):
            continue

        W = float(abs(bank_right_dist - bank_left_dist)) if (np.isfinite(bank_left_dist) and np.isfinite(bank_right_dist)) else float(np.nan)
        wse = _estimate_wse_from_profile(xsp, xs_len, bank_left_z, bank_right_z, cfg)
        wse_by_xs[xsid] = wse

        dmax_prior = _compute_dmax_prior(W, cfg, acct=acct)

        xs_records.append(
            dict(
                xs_id=xsid,
                river_id=xsl.get("river_id", np.nan),
                component_id=int(xsl.get("component_id", -1)) if pd.notna(xsl.get("component_id", np.nan)) else -1,
                s_center_m=_safe_float(xsl.get("s_center_m", np.nan)),
                xs_len_m=xs_len,
                bank_left_dist_m=bank_left_dist,
                bank_right_dist_m=bank_right_dist,
                bank_left_z_m=bank_left_z,
                bank_right_z_m=bank_right_z,
                width_m=W,
                wse_m=wse,
                dmax_prior_m=dmax_prior,
            )
        )

    xs_param = pd.DataFrame(xs_records)
    if xs_param.empty:
        raise RuntimeError("No valid cross-sections to process (check bank picks and filters).")

    # XS that survive basic filtering are the effective constraint set.
    acct["n_xs_used"] = int(len(xs_param))




    # Optional: curvature-driven trapezoid asymmetry (thalweg skew proxy)
    xs_param = _attach_curvature_asymmetry(xs_lines=xs_lines, xs_param=xs_param, cfg=cfg)
    # Optional reach attributes (for multivariate priors)
    xs_param["drain_area_km2"] = np.nan
    xs_param["slope_mpm"] = np.nan
    if river_gpkg is not None and Path(river_gpkg).exists():
        try:
            rivers, rivers_layer_used = _read_layer_with_fallback(Path(river_gpkg), rivers_layer, purpose="rivers")
            if rivers_layer_used != rivers_layer:
                log.info("[RIVER][ATTR] rivers layer '%s' not found/usable; using '%s'", str(rivers_layer), str(rivers_layer_used))
            if "river_id" not in rivers.columns:
                # fall back to common id fields
                rid_guess = _guess_field(rivers.columns, ["river_id", "RiverID", "RID", "COMID", "comid"])
                if rid_guess is not None:
                    rivers = rivers.rename(columns={rid_guess: "river_id"})
            if "river_id" in rivers.columns and not rivers.empty:
                da_field = drain_area_field or _guess_field(
                    rivers.columns,
                    ["TotDASqKm", "totdasqkm", "TOTDASQKM", "drain_area_km2", "DA_SQ_KM", "DrainArea", "drainarea"],
                )
                sl_field = slope_field or _guess_field(rivers.columns, ["SLOPE", "slope", "slope_mpm", "Slope"])
                q_field = None
                if str(cfg.manning_mode).lower().strip() == "from_field" and manning_q_field:
                    q_field = manning_q_field
                dist_field = None
                if manning_dist_to_mouth_field:
                    dist_field = manning_dist_to_mouth_field

                rdf = pd.DataFrame({"river_id": rivers["river_id"].astype(str)})
                rdf["manning_q_cms"] = np.nan
                rdf["dist_to_mouth_km"] = np.nan
                if da_field is not None:
                    rdf["drain_area_km2"] = pd.to_numeric(rivers[da_field], errors="coerce")
                if sl_field is not None:
                    rdf["slope_mpm"] = pd.to_numeric(rivers[sl_field], errors="coerce")

                if q_field is not None and q_field in rivers.columns:
                    rdf["manning_q_cms"] = pd.to_numeric(rivers[q_field], errors="coerce")
                if dist_field is not None and dist_field in rivers.columns:
                    rdf["dist_to_mouth_km"] = pd.to_numeric(rivers[dist_field], errors="coerce")

                xs_param["_river_id_str"] = xs_param["river_id"].astype(str)
                rdf["_river_id_str"] = rdf["river_id"].astype(str)
                rdf_cols = ["_river_id_str", "drain_area_km2", "slope_mpm", "manning_q_cms", "dist_to_mouth_km"]
                for _c in rdf_cols:
                    if _c not in rdf.columns:
                        rdf[_c] = np.nan

                xs_param = xs_param.merge(
                    rdf[rdf_cols],
                    on="_river_id_str",
                    how="left",
                    suffixes=("", "_r"),
                )
                # NOTE: xs_param already contains placeholder columns (drain_area_km2, slope_mpm, ...).
                # After merge, pandas keeps the left-hand placeholders and writes the attached values
                # to *_r columns. We must explicitly fill placeholders from the attached columns.
                for _base in ["drain_area_km2", "slope_mpm", "manning_q_cms", "dist_to_mouth_km"]:
                    _r = f"{_base}_r"
                    if _base in xs_param.columns and _r in xs_param.columns:
                        _lhs = pd.to_numeric(xs_param[_base], errors="coerce")
                        _rhs = pd.to_numeric(xs_param[_r], errors="coerce")
                        # Treat common placeholder defaults (e.g., 0) as missing so real attached
                        # attributes are not silently ignored.
                        keep = np.isfinite(_lhs)
                        if _base in ("drain_area_km2", "slope_mpm", "manning_q_cms"):
                            keep = keep & (_lhs > 0)
                        elif _base == "dist_to_mouth_km":
                            keep = keep & (_lhs >= 0)
                        xs_param[_base] = np.where(keep, _lhs, _rhs)
                        xs_param = xs_param.drop(columns=[_r])
                xs_param = xs_param.drop(columns=["_river_id_str"])
        except Exception as e:
            log.warning("[RIVER][ATTR] failed to attach river attributes from %s:%s (%s)", river_gpkg, rivers_layer, e)

    # --------------------------------------------------------------------------------------
    # Reach attribute sanity checks
    # --------------------------------------------------------------------------------------
    # Drainage area and slope are optional; some network sources omit these fields. We warn but continue.
    da_ok = True
    sl_ok = True

    if "drain_area_km2" in xs_param.columns:
        # Drainage area should be strictly positive. Treat non-positive or non-finite values as missing
        # so DA-dependent priors do not silently activate on DA=0 headwater rows.
        da_vals = pd.to_numeric(xs_param["drain_area_km2"], errors="coerce")
        da_vals = da_vals.where(np.isfinite(da_vals) & (da_vals > 0), np.nan)
        xs_param["drain_area_km2"] = da_vals
        da_ok = np.isfinite(da_vals).any()
        if not da_ok:
            log.warning("[RIVER][ATTR] No valid drainage area values found; disabling DA-dependent priors.")
            xs_param["drain_area_km2"] = np.nan
    else:
        da_ok = False
    # slope_mpm may be missing in network attributes; we compute a slope proxy below if needed.
    sl_ok = False



# ------------------------
# Optional SWOT (or other) WSE anchoring (stage)
# ------------------------
# If provided, blend/replace the DEM/topo-derived WSE proxy with observed WSE (e.g., SWOT)
    # BEFORE we fit a longitudinal WSE profile and BEFORE any anchor that uses WSE.
    if cfg.swot_wse is not None:
        try:
            swot = _load_wse_obs(
                path=Path(cfg.swot_wse),
                target_crs=xs_lines.crs,
                wse_col=cfg.swot_wse_col,
                x_col=cfg.swot_x_col,
                y_col=cfg.swot_y_col,
                csv_crs=cfg.swot_csv_crs,
            )
            if swot is not None and not swot.empty:
                xs_param, wse_by_xs = _attach_swot_wse_to_xs(xs_lines, xs_param, swot, cfg)
                log.info("[SWOT][WSE] Attached/blended WSE observations: n_xs=%d", int(np.isfinite(xs_param["swot_wse_m"]).sum()))
            else:
                log.info("[SWOT][WSE] WSE observations empty; using DEM/topo proxy.")
        except Exception as e:
            log.warning("[SWOT][WSE] Failed to load/attach WSE observations (%s). Using DEM/topo proxy.", e)

    # Record the *anchoring* WSE source for downstream logic (e.g., energy solver gating).
    # This is intentionally conservative: unless we have actual observed stage values
    # attached, we treat WSE as DEM/topo proxy.
    try:
        wse_anchor_source = "dem_proxy"
        if "swot_wse_m" in xs_param.columns:
            if np.isfinite(pd.to_numeric(xs_param["swot_wse_m"], errors="coerce")).any():
                wse_anchor_source = "swot"
        acct["wse_anchor_source"] = str(wse_anchor_source)
    except Exception:
        acct["wse_anchor_source"] = "unknown"

    # --------------------------------------------------------------------------------------
    # Slope estimation / proxy (stage)
    # --------------------------------------------------------------------------------------
    # If reach slope was not provided by the network source, estimate a stabilized water-surface slope.
    # We prefer a fitted longitudinal WSE profile (if available) and fall back to a robust slope proxy.
    #
    # This is a key constraint for hydraulically consistent depths. Without it, inference collapses to
    # width-only priors + smoothing.
    slope_missing = True
    if "slope_mpm" in xs_param.columns:
        sl_vals = pd.to_numeric(xs_param["slope_mpm"], errors="coerce")
        slope_missing = (not np.isfinite(sl_vals).any()) or cfg.force_slope_proxy
    if not slope_missing:
        n_sl = int(np.isfinite(pd.to_numeric(xs_param.get("slope_mpm", pd.Series([])), errors="coerce")).sum())
        log.info("[RIVER][SLOPE] Using reach slope from network attributes (NHD): n_valid_xs=%d. "
                 "Depth estimates will be AOI-independent.", n_sl)
    else:
        log.warning("[RIVER][SLOPE] No reach slope from network; will estimate from WSE profile. "
                    "Depth estimates near AOI edges may vary by up to ~1 m between runs with different AOI extents.")

    # If requested, keep observed stage for bed elevations but do NOT let it drive slope fitting.
    if (cfg.swot_wse is not None) and (not cfg.swot_use_for_slope):
        xs_param["_wse_blended_m"] = xs_param["wse_m"]
        if "wse_proxy_m" in xs_param.columns:
            xs_param["wse_m"] = xs_param["wse_proxy_m"]

    if slope_missing:
        xs_param["wse_fit_m"] = np.nan
        xs_param["slope_wse_mpm"] = np.nan

        # 1) Fit a longitudinal WSE profile (preferred) if the helper is available.
        if cfg.wse_profile_enabled:
            try:
                wcfg = WSEFitConfig(
                    enabled=True,
                    window=cfg.wse_profile_window,
                    min_n=cfg.wse_profile_min_n,
                    enforce_monotonic=cfg.wse_profile_monotonic,
                    slope_min=float(cfg.slope_min),
                    slope_max=float(cfg.slope_max),
                )
                if fit_wse_profile is None:
                    raise RuntimeError("river_wse module not available")
                wse_fit, slope_fit = fit_wse_profile(xs_param, cfg=wcfg)
                xs_param["wse_fit_m"] = wse_fit
                xs_param["slope_wse_mpm"] = slope_fit
            except Exception as e:
                log.warning("[RIVER][WSE] WSE profile fit failed, falling back to slope proxy (%s)", e)

        # 2) Always compute a robust slope proxy as a fallback.
        xs_param["slope_proxy_mpm"] = _compute_slope_proxy(
            xs_param,
            window=cfg.slope_proxy_window,
            slope_min=cfg.slope_min,
            slope_max=cfg.slope_max,
            min_n=cfg.slope_proxy_min_n,
        )

        # Prefer slope from fitted WSE profile if available; otherwise fallback to slope_proxy.
        if xs_param["slope_wse_mpm"].notna().any():
            xs_param["slope_mpm"] = xs_param["slope_wse_mpm"]
            n_wse = int(xs_param["slope_wse_mpm"].notna().sum())
            log.info("[RIVER][SLOPE] Source: WSE-profile fit (n=%d XS). AOI-boundary dependent.", n_wse)
        else:
            xs_param["slope_mpm"] = xs_param["slope_proxy_mpm"]
            n_prx = int(xs_param["slope_proxy_mpm"].notna().sum())
            log.warning("[RIVER][SLOPE] Source: rolling WSE proxy (n=%d XS). AOI-boundary dependent. "
                        "Add Slope field to NHD fetch or provide SWOT WSE for stable results.", n_prx)


    # Validate / finalize slope availability after proxy computation.
    if "slope_mpm" in xs_param.columns:
        sl_vals = pd.to_numeric(xs_param["slope_mpm"], errors="coerce")
        sl_ok = np.isfinite(sl_vals).any()
        if not sl_ok:
            log.warning("[RIVER][ATTR] No valid slope values found (even after slope proxy); disabling slope-dependent priors.")
            xs_param["slope_mpm"] = np.nan
    else:
        sl_ok = False

    # If multivariate priors were requested but required reach attributes are missing, fall back.
    if cfg.prior_mode.lower() == "multivariate" and not (da_ok and sl_ok):
        log.warning("[PRIOR] multivariate prior requested but reach attributes missing (da_ok=%s slope_ok=%s); falling back to powerlaw.", da_ok, sl_ok)
        cfg.prior_mode = "powerlaw"
    # Restore blended WSE after slope estimation if we suppressed SWOT stage during slope fitting.
    if "_wse_blended_m" in xs_param.columns:
        xs_param["wse_m"] = xs_param["_wse_blended_m"]
        xs_param = xs_param.drop(columns=["_wse_blended_m"])
    # Upgrade prior if requested
    if str(cfg.prior_mode).lower().strip() == "multivariate":
        def _mv_prior_row(r: pd.Series) -> float:
            return _compute_dmax_prior_multivariate(r, cfg, acct=acct)
        xs_param["dmax_prior_m"] = xs_param.apply(_mv_prior_row, axis=1)
        acct["prior_mode_effective"] = "multivariate"
    else:
        acct["prior_mode_effective"] = "powerlaw"

    # ---- Optional soft priors (blended into dmax_prior_m) ----
    # 1) Regional hydraulic geometry curves (DA -> bankfull depth)
    xs_param["dmax_regional_curve_m"] = np.nan
    xs_param["regional_curve_wt"] = 0.0
    xs_param["regional_curve_detail"] = ""
    if cfg.regional_curve_enabled:
        def _rc_apply(r):
            d, w, det = _compute_dmax_regional_curve(r, cfg)
            return pd.Series({"dmax_regional_curve_m": d, "regional_curve_wt": w, "regional_curve_detail": det})
        rc = xs_param.apply(_rc_apply, axis=1)
        xs_param[["dmax_regional_curve_m", "regional_curve_wt", "regional_curve_detail"]] = rc
        # Blend
        w = xs_param["regional_curve_wt"].astype("float64").clip(0.0, 1.0)
        d0 = xs_param["dmax_prior_m"].astype("float64")
        d1 = xs_param["dmax_regional_curve_m"].astype("float64")
        use = np.isfinite(d0) & np.isfinite(d1) & (w > 0)
        xs_param.loc[use, "dmax_prior_m"] = (1.0 - w[use]) * d0[use] + w[use] * d1[use]
        acct["n_regional_curve_applied"] = int(np.sum(use))
        try:
            acct["regional_curve_weight_mean"] = float(np.nanmean(w[use])) if np.any(use) else 0.0
        except Exception:
            acct["regional_curve_weight_mean"] = 0.0

    # 2) Manning inversion prior (requires Q, width, slope)
    xs_param["manning_q_cms_used"] = np.nan
    xs_param["manning_depth_mean_m"] = np.nan
    xs_param["manning_dmax_m"] = np.nan
    xs_param["manning_conf"] = 0.0
    xs_param["manning_wt"] = 0.0
    xs_param["manning_flags"] = ""
    xs_param["manning_q2_equation_id"] = ""
    xs_param["manning_q2_uncertainty_pct"] = np.nan

    # Auto-enable (conservative) Manning prior when no soundings were provided.
    # This is specifically to stabilize absolute depth scale in data-sparse reaches.
    if (
        cfg.auto_manning_when_no_soundings
        and (str(cfg.manning_mode).lower().strip() == "off")
        and (not soundings_path)
        and da_ok
        and sl_ok
        and (estimate_q2_from_drainage_area is not None)
    ):
        cfg.manning_mode = "q2_regional"
        log.info(
            "[PRIOR][MANNING] auto-enabled manning_mode=q2_regional (no soundings; da_ok=%s slope_ok=%s region=%s)",
            str(da_ok), str(sl_ok), cfg.manning_region
        )

    if str(cfg.manning_mode).lower().strip() != "off":
        m_mode = str(cfg.manning_mode).lower().strip()
        def _m_apply(r):
            W = float(r.get("width_m", np.nan))
            q2_eq_id = ""
            q2_unc_pct = np.nan
            S = float(r.get("slope_mpm", np.nan))
            if not (np.isfinite(W) and W > 0 and np.isfinite(S) and S > 0):
                return pd.Series({"manning_q_cms_used": np.nan, "manning_depth_mean_m": np.nan, "manning_dmax_m": np.nan, "manning_conf": 0.0, "manning_wt": 0.0, "manning_flags": "", "manning_q2_equation_id": q2_eq_id, "manning_q2_uncertainty_pct": q2_unc_pct})

            # Guard weight (simple slope/mouth filters)
            w_guard = _compute_manning_weight(r, cfg)
            if w_guard <= 0:
                return pd.Series({"manning_q_cms_used": np.nan, "manning_depth_mean_m": np.nan, "manning_dmax_m": np.nan, "manning_conf": 0.0, "manning_wt": 0.0, "manning_flags": "guard", "manning_q2_equation_id": q2_eq_id, "manning_q2_uncertainty_pct": q2_unc_pct})

            # Discharge
            Q = np.nan
            qsrc = ""
            q2_eq_id = ""
            q2_unc_pct = np.nan
            if m_mode == "constant":
                Q = float(cfg.manning_q_cms) if cfg.manning_q_cms is not None else np.nan
                qsrc = "constant"
            elif m_mode == "from_field":
                Q = float(r.get("manning_q_cms", np.nan))
                qsrc = "field"
            elif m_mode == "q2_regional":
                # Estimate bankfull discharge Q2 from drainage area (requires manning_inversion module)
                da = float("nan")
                for _k in ["drain_area_km2","drainage_area_km2","TotDASqKM","totdasqkm","DA_sqkm","DA_KM2","DrainArKm2","DrainArea","drain_area","DA"]:
                    if _k in r.index:
                        da = float(r.get(_k, np.nan))
                        break
                if estimate_q2_from_drainage_area is None or not (np.isfinite(da) and da > 0):
                    Q = np.nan
                else:
                    q2 = estimate_q2_from_drainage_area(da, region=cfg.manning_region)
                    Q = float(getattr(q2, "q2_m3s", np.nan))
                    q2_eq_id = str(getattr(q2, "equation_id", "") or "")
                    q2_unc_pct = float(getattr(q2, "uncertainty_pct", np.nan))
                    qsrc = f"q2_{cfg.manning_region}"

            if not (np.isfinite(Q) and Q > 0):
                return pd.Series({"manning_q_cms_used": Q, "manning_depth_mean_m": np.nan, "manning_dmax_m": np.nan, "manning_conf": 0.0, "manning_wt": 0.0, "manning_flags": f"noQ_{qsrc}", "manning_q2_equation_id": q2_eq_id, "manning_q2_uncertainty_pct": q2_unc_pct})

            # Depth estimate
            conf = 0.7  # default
            tidal = False
            backwater = False
            if invert_manning_for_depth is not None:
                res = invert_manning_for_depth(discharge_m3s=Q, width_m=W, slope=S, manning_n=float(_manning_n_effective(cfg)), discharge_source=qsrc)
                y = float(getattr(res, "depth_m", np.nan))
                conf = float(getattr(res, "confidence", 0.0) or 0.0)
                tidal = bool(getattr(res, "tidal_flag", False))
                backwater = bool(getattr(res, "backwater_flag", False))
            else:
                # Fallback to simple wide-channel inversion (mean depth)
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = ((float(_manning_n_effective(cfg)) * Q) / (W * np.sqrt(S))) ** (3.0 / 5.0)

            if not (np.isfinite(y) and y > 0):
                return pd.Series({"manning_q_cms_used": Q, "manning_depth_mean_m": y, "manning_dmax_m": np.nan, "manning_conf": 0.0, "manning_wt": 0.0, "manning_flags": f"badY_{qsrc}", "manning_q2_equation_id": q2_eq_id, "manning_q2_uncertainty_pct": q2_unc_pct})

            mean_to_dmax = float(2.0 / (1.0 + float(cfg.bottom_width_frac)))
            dmax = float(np.clip(y * mean_to_dmax, cfg.dmin_m, cfg.dmax_m))

            # Final weight: guard * max_weight * confidence
            min_conf = cfg.manning_min_confidence
            if conf < min_conf:
                w = 0.0
            else:
                w = float(np.clip(w_guard * float(cfg.manning_max_weight) * float(conf), 0.0, 1.0))

            flags = qsrc
            if tidal:
                flags += "|tidal"
            if backwater:
                flags += "|backwater"
            return pd.Series({"manning_q_cms_used": Q, "manning_depth_mean_m": y, "manning_dmax_m": dmax, "manning_conf": conf, "manning_wt": w, "manning_flags": flags, "manning_q2_equation_id": q2_eq_id, "manning_q2_uncertainty_pct": q2_unc_pct})

        mm = xs_param.apply(_m_apply, axis=1)
        xs_param[["manning_q_cms_used", "manning_depth_mean_m", "manning_dmax_m", "manning_conf", "manning_wt", "manning_flags", "manning_q2_equation_id", "manning_q2_uncertainty_pct"]] = mm

        # Blend
        w = xs_param["manning_wt"].astype("float64").clip(0.0, 1.0)
        d0 = xs_param["dmax_prior_m"].astype("float64")
        d1 = xs_param["manning_dmax_m"].astype("float64")
        use = np.isfinite(d0) & np.isfinite(d1) & (w > 0)
        xs_param.loc[use, "dmax_prior_m"] = (1.0 - w[use]) * d0[use] + w[use] * d1[use]



    
    # Receipt: expose Manning prior/Q usage so discharge-driven behavior is debuggable.
    try:
        if cfg.manning_mode.lower().strip() != "off":
            w = pd.to_numeric(xs_param.get("manning_wt", 0.0), errors="coerce").fillna(0.0)
            q = pd.to_numeric(xs_param.get("manning_q_cms_used", np.nan), errors="coerce")
            n_total = int(len(xs_param))
            n_eff = float(_manning_n_effective(cfg))
            n_q = int(np.sum(np.isfinite(q) & (q > 0)))
            n_used = int(np.sum(w > 0))
            w_mean = float(np.nanmean(w.values)) if n_total > 0 else 0.0

            flags = xs_param.get("manning_flags", "").astype(str)
            src = flags.str.split("\\|", n=1, expand=False).str[0].replace("", "unknown")
            src_counts = src.value_counts(dropna=False).to_dict() if n_total > 0 else {}

            eqid = xs_param.get("manning_q2_equation_id", "").astype(str)
            eq_counts = eqid.replace("", "none").value_counts(dropna=False).to_dict() if n_total > 0 else {}

            # Warn if we're relying on coarse built-ins (intentionally non-authoritative).
            n_simplified = int(np.sum(eqid.astype(str).str.contains("_simplified", na=False)))
            n_placeholder = int(np.sum(eqid.astype(str).str.contains("placeholder", case=False, na=False)))
            if n_placeholder > 0:
                log.warning(
                    "[PRIOR][MANNING] Q2 regression metadata indicates placeholders for %d XS (replace with published coefficients in sdb_config.json).",
                    n_placeholder,
                )

            if n_simplified > 0:
                log.warning(
                    "[PRIOR][MANNING] Q2 regression used simplified fallback for %d XS (provide published coefficients via sdb_config.json river.registry.q2_regressions).",
                    n_simplified,
                )

            q50 = float(q[q > 0].median()) if n_q > 0 else float("nan")
            qmin = float(q[q > 0].min()) if n_q > 0 else float("nan")
            qmax = float(q[q > 0].max()) if n_q > 0 else float("nan")

            log.info(
                "[PRIOR][MANNING] mode=%s region=%s q_valid=%d/%d used=%d w_mean=%.3f n=%.4f Q(m3/s) median=%.4g range=[%.4g, %.4g] sources=%s q2_eq=%s",
                cfg.manning_mode,
                cfg.manning_region,
                n_q,
                n_total,
                n_used,
                w_mean, n_eff,
                q50,
                qmin,
                qmax,
                str(src_counts),
                str(eq_counts),
            )
    except Exception:
        pass
    # Receipt: echo energy-solver gating inputs so activation is debuggable from logs.
    try:
        wse_anchor = "unknown"
        if isinstance(acct, dict):
            wse_anchor = str(acct.get("wse_anchor_source", "unknown"))
        log.info(
            "[ENERGY] requested=%s allow_dem_proxy_wse=%s wse_anchor=%s",
            cfg.energy_solver_enabled,
            cfg.energy_allow_dem_proxy_wse,
            wse_anchor,
        )
    except Exception:
        pass

    # ------------------------
    # Calibration anchors
    # ------------------------
    # --soundings-subset: explicit Pass 2 handoff from bathy_main.py.
    # When provided, use the pre-clipped parquet INSTEAD of the raw --soundings files.
    # Hard-fail if the file is missing or empty so the wire-sever bug is caught immediately
    # rather than silently producing a prior-only result that looks like a successful run.
    _subset_path = str(cfg.soundings_subset or "").strip()
    if _subset_path:
        _subset_p = Path(_subset_path)
        if not _subset_p.exists():
            log.error(
                "[CALIB][HARD-FAIL] --soundings-subset was specified (%s) but the file does not exist. "
                "This means Pass 1 (--only-write-soundings-subset) did not complete successfully, "
                "or bathy_main.py passed a wrong path. Aborting to prevent a silent prior-only run.",
                _subset_path,
            )
            raise SystemExit(2)
        try:
            import pandas as _pd_check
            _n_check = len(_pd_check.read_parquet(_subset_p))
        except Exception as _e_check:
            log.error(
                "[CALIB][HARD-FAIL] --soundings-subset (%s) could not be read: %s. Aborting.",
                _subset_path, _e_check,
            )
            raise SystemExit(2)
        if _n_check == 0:
            log.error(
                "[CALIB][HARD-FAIL] --soundings-subset (%s) contains 0 rows after loading. "
                "The parquet was written empty — most likely the z/depth column was all-NaN "
                "after the finite filter in _write_soundings_subset (eHydro elevation-only data). "
                "Fix _write_soundings_subset so depth_m/z_m are coalesced before filtering. "
                "Aborting to prevent a silent prior-only run.",
                _subset_path,
            )
            raise SystemExit(2)
        log.info("[CALIB] --soundings-subset validated: n=%d rows. Using subset instead of raw --soundings.", _n_check)
        # Override soundings_path so the rest of the function uses the validated subset.
        soundings_path = [_subset_path]
        # Subset parquet has a 'crs' column; do not override with caller's soundings_crs
        # (which points at the raw source CRS, not the template-projected subset CRS).
        soundings_crs = None

    # Soundings calibration (optional)
    calib_df = pd.DataFrame(columns=["xs_id", "calib_n", "calib_depth_stat"])
    if soundings_path:
        soundings = _load_soundings_many(
            soundings_path,
            target_crs=xs_lines.crs,
            depth_col=soundings_depth_col,
            elev_col=soundings_elev_col,
            x_col=soundings_x_col,
            y_col=soundings_y_col,
            soundings_crs=soundings_crs,
        )
        if soundings is not None and not soundings.empty:
            n_in_all = int(len(soundings))
            log.info("[CALIB] Loaded soundings: n=%d", n_in_all)
            # Guard against massive point clouds (e.g., Hydronos/eHydro exports).
            max_n = int(cfg.soundings_max_points or 0)
            if max_n > 0 and len(soundings) > max_n:
                seed = int(cfg.soundings_sample_seed or 0)
                rng = np.random.default_rng(seed)
                total = int(len(soundings))
                # Preserve relative source-file composition when possible.
                if "_src_file" in soundings.columns:
                    parts = []
                    for src, gsrc in soundings.groupby("_src_file", sort=False):
                        frac = len(gsrc) / max(total, 1)
                        take = max(1, int(round(frac * max_n)))
                        if len(gsrc) <= take:
                            parts.append(gsrc)
                        else:
                            idx = rng.choice(gsrc.index.values, size=take, replace=False)
                            parts.append(gsrc.loc[idx])
                    soundings = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=soundings.crs)
                else:
                    idx = rng.choice(soundings.index.values, size=max_n, replace=False)
                    soundings = soundings.loc[idx].copy()
                log.warning("[CALIB] Downsampled soundings to n=%d (from %d) to avoid OOM (soundings_max_points=%d).",
                            int(len(soundings)), total, int(max_n))

            # Optional: write the unified (possibly downsampled) set for reuse by downstream steps.
            if cfg.write_soundings_subset:
                try:
                    out_path = Path(str(cfg.write_soundings_subset))
                    _write_soundings_subset(out_path, soundings)
                    by_src = None
                    if "_src_file" in soundings.columns:
                        by_src = {}
                        vc = soundings["_src_file"].astype(str).value_counts()
                        for k, v in vc.items():
                            kk = Path(str(k)).stem if str(k) not in ["", "nan", "None"] else "unknown"
                            by_src[kk] = int(v)
                    log.info("[CALIB] %s", _soundings_one_line(out_path, int(len(soundings)), n_in_all, by_src))

                    if cfg.only_write_soundings_subset:
                        log.info("[CALIB] --only-write-soundings-subset requested; exiting after subset write.")
                        raise SystemExit(0)
                except SystemExit:
                    raise
                except Exception as e:
                    log.warning("[CALIB] Failed to write soundings subset '%s': %s", str(cfg.write_soundings_subset), e)

            calib_df = _calibrate_dmax_from_soundings(xs_lines[["xs_id", "geometry"]].copy(), soundings, wse_by_xs, cfg)
            log.info("[CALIB] Matched XS: %d", len(calib_df))
        else:
            log.info("[CALIB] Soundings empty; will use other anchors / priors.")
    else:
        log.info("[CALIB] No soundings provided; will use other anchors / priors.")

    xs_param = xs_param.merge(calib_df, on="xs_id", how="left")
    xs_param["soundings_n"] = pd.to_numeric(xs_param["calib_n"], errors="coerce").fillna(0).astype("int64")
    xs_param["soundings_dmax_m"] = pd.to_numeric(xs_param["calib_depth_stat"], errors="coerce")
    xs_param = xs_param.drop(columns=["calib_n", "calib_depth_stat"])
    # ---- Optional: 1D energy-consistent depth solver (flag-controlled) ----
    # NOTE: To avoid tile-to-tile discontinuities, treat “soundings present” as
    # “at least one XS has usable soundings after masking/subsetting”, not merely
    # “a soundings file path was provided”.
    if cfg.energy_solver_enabled:
        _soundings_effective = False
        try:
            _sn = pd.to_numeric(xs_param.get("soundings_n", 0), errors="coerce").fillna(0)
            _soundings_effective = bool((_sn > 0).any())
        except Exception:
            _soundings_effective = False
        _soundings_gate = soundings_path if _soundings_effective else None
        try:
            xs_param = _apply_1d_energy_solver(xs_param=xs_param, cfg=cfg, soundings_path=_soundings_gate, acct=acct)
            if "energy_dmax_m" in xs_param.columns:
                # Conservative blend: guardrails * energy confidence * energy max weight.
                w_guard = xs_param.apply(lambda r: _compute_manning_weight(r, cfg), axis=1).astype("float64").clip(0.0, 1.0)
                econf = pd.to_numeric(xs_param.get("energy_conf", np.nan), errors="coerce").astype("float64").clip(0.0, 1.0).fillna(0.0)

                # Default conservative cap for production unless explicitly configured.
                # NOTE: This is intentionally separate from manning_max_weight to avoid
                # over-weighting the energy solver when Q or slope are uncertain.
                emax = cfg.energy_max_weight
                emin = cfg.energy_min_confidence
                w = w_guard * econf * emax
                w = w.where(econf >= emin, 0.0)
                d0 = xs_param["dmax_prior_m"].astype("float64")
                d1 = pd.to_numeric(xs_param["energy_dmax_m"], errors="coerce").astype("float64")
                use = np.isfinite(d0) & np.isfinite(d1) & (w > 0)
                if use.any():
                    # Track magnitude of the physics adjustment (for verifiability).
                    before = d0.copy()
                    xs_param.loc[use, "dmax_prior_m"] = (1.0 - w[use]) * d0[use] + w[use] * d1[use]
                    after = xs_param["dmax_prior_m"].astype("float64")
                    delta = (after - before).abs()
                    # Only summarize where we actually blended.
                    delta_use = pd.to_numeric(delta[use], errors="coerce").astype("float64")
                    if acct is not None:
                        acct["energy_solver_blend_n"] = int(np.sum(use))
                        acct["energy_solver_blend_max_weight"] = float(emax)
                        acct["energy_solver_blend_min_conf"] = float(emin)
                        try:
                            # Robust summary statistics for scientific interpretation.
                            dv = delta_use[np.isfinite(delta_use.values)].values
                            if dv.size:
                                acct["energy_solver_delta_dmax_m_median"] = float(np.nanmedian(dv))
                                acct["energy_solver_delta_dmax_m_p95"] = float(np.nanpercentile(dv, 95))
                            else:
                                acct["energy_solver_delta_dmax_m_median"] = 0.0
                                acct["energy_solver_delta_dmax_m_p95"] = 0.0

                            # Additional sanity metrics: how many rows truly changed, and
                            # the longest contiguous run of changes along-stream.
                            # This avoids a false sense of security when the median is small
                            # but changes are spatially concentrated.
                            changed = np.zeros(len(xs_param), dtype=bool)
                            # Use a strict >0 threshold; values are floats but the blend is deterministic.
                            changed_idx = use.values.copy()
                            # Only treat as changed where delta is finite and > 0.
                            try:
                                dmask = np.isfinite(delta.values) & (delta.values > 0)
                                changed_idx = changed_idx & dmask
                            except Exception:
                                pass
                            changed[changed_idx] = True
                            acct["energy_solver_changed_n"] = int(np.sum(changed))

                            max_run = 0
                            if int(np.sum(changed)) > 0 and ("s_center_m" in xs_param.columns):
                                try:
                                    order = np.argsort(pd.to_numeric(xs_param["s_center_m"], errors="coerce").values)
                                    c = changed[order]
                                    run = 0
                                    for v in c:
                                        if bool(v):
                                            run += 1
                                            if run > max_run:
                                                max_run = run
                                        else:
                                            run = 0
                                except Exception:
                                    max_run = 0
                            acct["energy_solver_changed_max_run"] = int(max_run)
                        except Exception:
                            pass
        except Exception as e:
            log.warning("[ENERGY] Energy solver failed; continuing without it (%s)", e)

        # Always emit a single "receipt" line when the solver is requested so runs are
        # verifiable from logs (and not inferred from side-effects).
        try:
            reason = str(acct.get("energy_solver_reason", "unknown"))
            n_total = int(acct.get("energy_solver_n_total", 0) or 0)
            n_applied = int(acct.get("energy_solver_n_applied", 0) or 0)
            wse_src = str(acct.get("energy_solver_wse_source", "none"))
            wse_anchor = str(acct.get("wse_anchor_source", "unknown"))
            blend_n = int(acct.get("energy_solver_blend_n", 0) or 0)
            changed_n = int(acct.get("energy_solver_changed_n", 0) or 0)
            max_run = int(acct.get("energy_solver_changed_max_run", 0) or 0)
            dmed = float(acct.get("energy_solver_delta_dmax_m_median", 0.0) or 0.0)
            dp95 = float(acct.get("energy_solver_delta_dmax_m_p95", 0.0) or 0.0)
            log.info(
                "[ENERGY] status: enabled=%s reason=%s n_total=%d n_applied=%d blend_n=%d changed_n=%d max_run=%d wse_source=%s wse_anchor=%s |delta_dmax| median=%.4g p95=%.4g",
                cfg.energy_solver_enabled, reason, n_total, n_applied, blend_n, changed_n, max_run, wse_src, wse_anchor, dmed, dp95,
            )
        except Exception:
            pass
    # Write an energy solver receipt alongside the XS constraint meta so the run is
    # inspectable without grepping logs.
    try:
        # NOTE: do not bind the name "Path" inside infer_bathy(); it is already
        # imported at module scope, and rebinding it here makes it a local variable
        # which can trigger UnboundLocalError earlier in the function.
        from pathlib import Path as _Path
        import json as _json
        if out_meta_json:
            _receipt_path = _Path(out_meta_json).with_name("energy_solver_receipt.json")
        else:
            _receipt_path = _Path(out_gpkg).with_name("energy_solver_receipt.json")

        _receipt = {
            "requested": bool(acct.get("energy_solver_requested", cfg.energy_solver_enabled)),
            "enabled": bool(acct.get("energy_solver_enabled", False)),
            "allow_dem_proxy_wse": cfg.energy_allow_dem_proxy_wse,
            "wse_anchor_source": str(acct.get("wse_anchor_source", "unknown")),
            "wse_source": str(acct.get("energy_solver_wse_source", "none")),
            "reason": str(acct.get("energy_solver_reason", "unknown")),
            "n_total": int(acct.get("energy_solver_n_total", 0) or 0),
            "n_applied": int(acct.get("energy_solver_n_applied", 0) or 0),
            "blend_n": int(acct.get("energy_solver_blend_n", 0) or 0),
            "changed_n": int(acct.get("energy_solver_changed_n", 0) or 0),
            "changed_max_run": int(acct.get("energy_solver_changed_max_run", 0) or 0),
            "delta_dmax_m_abs_median": float(acct.get("energy_solver_delta_dmax_m_median", 0.0) or 0.0),
            "delta_dmax_m_abs_p95": float(acct.get("energy_solver_delta_dmax_m_p95", 0.0) or 0.0),
        }
        _receipt_path.write_text(_json.dumps(_receipt, indent=2, sort_keys=True) + "\n")
        log.info("[ENERGY] Receipt written: %s", _receipt_path)
    except Exception as e:
        log.warning("[ENERGY] Failed to write receipt: %s", e)



    # Final selection fields
    xs_param["calib_src"] = "prior"
    xs_param["calib_n"] = 0
    xs_param["dmax_raw_m"] = xs_param["dmax_prior_m"]

    # Apply soundings where available
    m_snd = xs_param["soundings_n"] > 0
    xs_param.loc[m_snd, "dmax_raw_m"] = xs_param.loc[m_snd, "soundings_dmax_m"]
    xs_param.loc[m_snd, "calib_src"] = "soundings"
    xs_param.loc[m_snd, "calib_n"] = xs_param.loc[m_snd, "soundings_n"]

    # Build station/gage assignment (used by width-stage and USGS anchors)
    # IMPORTANT: do not overwrite upstream gage linkage if already present.
    if "gage_site_no" not in xs_param.columns:
        xs_param["gage_site_no"] = pd.NA
    if "gage_dist_m" not in xs_param.columns:
        xs_param["gage_dist_m"] = np.nan

    # Flatten sites list
    usgs_sites_flat = []
    if usgs_sites:
        for s in usgs_sites:
            if s is None:
                continue
            for part in str(s).split(","):
                part = part.strip()
                if part:
                    usgs_sites_flat.append(part)

    # Load width-stage observations (optional)
    ws_df = _load_width_stage_csvs(width_stage_csv)

    # If width-stage CSV has no site_no and there is exactly one USGS site, assume that site
    if (not ws_df.empty) and ws_df["site_no"].isna().all() and len(usgs_sites_flat) == 1:
        ws_df["site_no"] = usgs_sites_flat[0]

    # Determine which station sites we need locations for
    sites_for_loc = set(usgs_sites_flat)
    if not ws_df.empty:
        for s in ws_df["site_no"].dropna().astype(str).unique().tolist():
            sites_for_loc.add(s)

    gage_gdf = None
    cache_dir = None
    if sites_for_loc:
        try:
            import datetime
            from usgs_nwis import fetch_site_locations

            cache_dir = Path(usgs_cache_dir) if usgs_cache_dir is not None else (Path(out_gpkg).parent / "usgs_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)

            site_df = fetch_site_locations(sorted(list(sites_for_loc)), cache_dir=cache_dir)
            if not site_df.empty:
                gage_gdf = gpd.GeoDataFrame(
                    site_df,
                    geometry=gpd.points_from_xy(site_df["dec_long_va"], site_df["dec_lat_va"]),
                    crs="EPSG:4326",
                ).to_crs(xs_lines.crs)

                # OPTIONAL: reduce wrong-gage assignment by snapping gages to the river network and
                # restricting XS->gage matching by river_id when available.
                if river_gpkg is not None and "river_id" in xs_param.columns:
                    try:
                        rivers_net = _read_layer(Path(river_gpkg), layer=rivers_layer).to_crs(xs_lines.crs)
                        if "river_id" in rivers_net.columns and not rivers_net.empty:
                            sidx = rivers_net.sindex
                            gage_rids = []
                            max_snap = float(cfg.gage_snap_max_dist_m)
                            for pt in gage_gdf.geometry:
                                rid_val = pd.NA
                                if pt is None or pt.is_empty:
                                    gage_rids.append(rid_val)
                                    continue
                                cand_idx = list(sidx.intersection(pt.buffer(max_snap).bounds))
                                if cand_idx:
                                    cand = rivers_net.iloc[cand_idx]
                                    d = cand.geometry.distance(pt)
                                    jmin = int(d.idxmin())
                                    if float(d.loc[jmin]) <= max_snap:
                                        rid_val = rivers_net.loc[jmin, "river_id"]
                                gage_rids.append(rid_val)
                            gage_gdf["river_id"] = gage_rids
                        else:
                            log.info("[CALIB][GAGE] rivers layer has no river_id; using nearest-gage matching.")
                    except Exception as e:
                        log.info("[CALIB][GAGE] could not snap gages to river network: %s", e)

                centers = xs_lines.set_index("xs_id").geometry.interpolate(0.5, normalized=True)
                gage_cols = ["site_no", "geometry"] + (["river_id"] if "river_id" in gage_gdf.columns else [])
                gage_sites = gage_gdf[gage_cols].copy()
                for i, r in xs_param.iterrows():
                    xsid = r["xs_id"]
                    if xsid not in centers.index:
                        continue
                    cgeom = centers.loc[xsid]
                    gage_candidates = gage_sites
                    if ('river_id' in xs_param.columns) and ('river_id' in gage_sites.columns):
                        rid = xs_param.at[i, 'river_id']
                        if pd.notna(rid):
                            cand = gage_sites[gage_sites['river_id'].astype(str) == str(rid)]
                            if not cand.empty:
                                gage_candidates = cand
                    dists = gage_candidates.geometry.distance(cgeom)
                    if len(dists) == 0:
                        continue
                    j = int(dists.idxmin())
                    new_site = str(gage_candidates.loc[j, "site_no"])
                    new_dist = float(dists.loc[j])
                    cur_site = xs_param.at[i, "gage_site_no"] if "gage_site_no" in xs_param.columns else pd.NA
                    cur_dist = xs_param.at[i, "gage_dist_m"] if "gage_dist_m" in xs_param.columns else np.nan
                    # Only fill if missing, or if we found a closer site.
                    if (pd.isna(cur_site) or str(cur_site).lower() in ("nan", "none", "")) or (not np.isfinite(cur_dist)) or (new_dist < float(cur_dist)):
                        xs_param.at[i, "gage_site_no"] = new_site
                        xs_param.at[i, "gage_dist_m"] = new_dist
        except Exception as e:
            log.warning("[CALIB][USGS] Failed to fetch station locations: %s", e)

    # Width-stage inversion anchor (optional)
    if (gage_gdf is not None) and (ws_df is not None) and (not ws_df.empty):
        xs_param["width_stage_beta"] = np.nan
        xs_param["width_stage_dmax_m"] = np.nan
        xs_param["width_stage_r2"] = np.nan
        xs_param["width_stage_wt"] = np.nan

        for site_no, gws in ws_df.groupby("site_no", dropna=True):
            site_no = str(site_no)
            m_site = (
                (xs_param["gage_site_no"].astype(str) == site_no)
                & (pd.to_numeric(xs_param["gage_dist_m"], errors="coerce") <= float(cfg.width_stage_max_dist_m))
            )
            if m_site.sum() < 3:
                continue
            Wtop = float(pd.to_numeric(xs_param.loc[m_site, "width_m"], errors="coerce").median())
            if not np.isfinite(Wtop) or Wtop <= 0:
                continue

            gws2 = gws.copy()
            m_in = (gws2["width_m"] > 0) & (gws2["width_m"] <= (1.15 * Wtop))
            gws2 = gws2.loc[m_in]
            if len(gws2) < int(cfg.width_stage_min_n):
                continue

            beta, nfit, r2 = _fit_width_stage_beta(gws2)
            if beta is None:
                continue
            w_ws = _width_stage_weight(int(nfit), float(r2) if r2 is not None else float("nan"), cfg)
            if w_ws <= 0:
                continue
            dmax_ws = _dmax_from_width_stage(beta, Wtop, cfg.bottom_width_frac)
            if not np.isfinite(dmax_ws) or dmax_ws <= 0:
                continue
            dmax_ws = float(np.clip(dmax_ws, cfg.dmin_m, cfg.dmax_m))

            m_apply = m_site & (xs_param["calib_src"] == "prior")
            # Soft blend toward width-stage inferred Dmax
            d_cur = pd.to_numeric(xs_param.loc[m_apply, "dmax_raw_m"], errors="coerce")
            d_new = (1.0 - float(w_ws)) * d_cur + float(w_ws) * float(dmax_ws)
            xs_param.loc[m_apply, "dmax_raw_m"] = pd.to_numeric(d_new, errors="coerce")
            xs_param.loc[m_apply, "calib_src"] = "width_stage"
            xs_param.loc[m_apply, "calib_n"] = int(nfit)
            xs_param.loc[m_apply, "width_stage_beta"] = float(beta)
            xs_param.loc[m_apply, "width_stage_dmax_m"] = float(dmax_ws)
            xs_param.loc[m_apply, "width_stage_r2"] = float(r2)
            xs_param.loc[m_apply, "width_stage_wt"] = float(w_ws)

            log.info(
                "[CALIB][WIDTH_STAGE] site=%s n=%d r2=%.3f beta=%.6f w=%.2f Wtop=%.1f Dmax=%.2f (applied=%d)",
                site_no,
                nfit,
                float(r2) if r2 is not None else float('nan'),
                float(beta),
                float(w_ws),
                float(Wtop),
                float(dmax_ws),
                int(m_apply.sum()),
            )

    # USGS discharge-measurement anchor (Option A)
    # Uses velocity-area discharge measurements (width+area -> mean depth) to fit a local 'a_site'
    # for Dmax ≈ a_site * W^b, then applies as a *soft* constraint to nearby cross-sections.
    if (gage_gdf is not None) and usgs_sites_flat:
        import datetime
        start_dt = usgs_start or "1900-01-01"
        end_dt = usgs_end or datetime.date.today().isoformat()

        try:
            from usgs_nwis import fetch_discharge_measurements, normalize_units_us_to_si, compute_site_a_from_measurements

            cache_dir = cache_dir or (Path(out_gpkg).parent / "usgs_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)

            if "calib_src_detail" not in xs_param.columns:
                xs_param["calib_src_detail"] = xs_param["calib_src"].astype(str)

            xs_param["usgs_a_site"] = np.nan
            xs_param["usgs_n_meas"] = 0
            xs_param["usgs_wt"] = np.nan
            xs_param["usgs_a_cv"] = np.nan
            xs_param["usgs_width_med_m"] = np.nan
            xs_param["usgs_width_ratio"] = np.nan

            q_lo = float(cfg.usgs_q_quantile_lo)
            q_hi = float(cfg.usgs_q_quantile_hi)
            q_range = (q_lo, q_hi) if (0.0 <= q_lo < q_hi <= 1.0) else None

            # If requested, derive mean→max conversion from your trapezoid assumption.
            mean_to_dmax = float(cfg.usgs_mean_to_dmax)
            if (not np.isfinite(mean_to_dmax)) or (mean_to_dmax <= 0):
                mean_to_dmax = 2.0 / (1.0 + float(cfg.bottom_width_frac))
                log.info("[CALIB][USGS] mean_to_dmax=auto -> %.3f (bottom_width_frac=%.2f)", mean_to_dmax, float(cfg.bottom_width_frac))

            for site_no in usgs_sites_flat:
                meas = fetch_discharge_measurements([site_no], start_dt, end_dt, cache_dir=cache_dir)
                if meas is None or meas.empty:
                    continue
                meas_si = normalize_units_us_to_si(meas)

                a_site, n_meas, meta = compute_site_a_from_measurements(
                    meas_si,
                    b=float(cfg.b),
                    mean_to_dmax=float(mean_to_dmax),
                    stat=str(cfg.usgs_a_stat),
                    q_quantile_range=q_range,
                )
                if a_site is None or (not np.isfinite(a_site)) or int(n_meas) < 3:
                    continue

                # Warn on instability in the derived a-values
                a_cv = float(meta.get("a_cv", float("nan")))
                if np.isfinite(a_cv) and a_cv > float(cfg.usgs_a_cv_warn):
                    log.warning(
                        "[CALIB][USGS] site=%s unstable a-values (cv=%.3f > %.3f). Using as soft prior only.",
                        str(site_no), a_cv, float(cfg.usgs_a_cv_warn)
                    )

                # Candidate XS near this site
                m_site = (
                    (xs_param["gage_site_no"].astype(str) == str(site_no))
                    & (pd.to_numeric(xs_param["gage_dist_m"], errors="coerce") <= float(cfg.usgs_max_dist_m))
                )
                if m_site.sum() == 0:
                    continue

                # Apply to prior or width-stage results (soundings remain highest priority)
                m_apply = m_site & (xs_param["calib_src"].isin(["prior", "width_stage"]))
                if m_apply.sum() == 0:
                    continue

                # Width-mismatch guard: bank-to-bank width can be far larger than wet width during USGS measurements.
                w_meas = float(meta.get("width_med_m", float("nan")))
                w_bank = float(pd.to_numeric(xs_param.loc[m_site, "width_m"], errors="coerce").median())
                w_ratio = (w_bank / w_meas) if (np.isfinite(w_bank) and np.isfinite(w_meas) and w_meas > 0) else float("nan")

                w_usgs = 1.0

                if np.isfinite(w_ratio) and (w_ratio > float(cfg.usgs_width_ratio_max)):
                    if bool(cfg.usgs_width_ratio_blend):
                        w_usgs *= max(0.0, float(cfg.usgs_width_ratio_max) / float(w_ratio))
                        log.info(
                            "[CALIB][USGS] site=%s width ratio=%.2f > %.2f; blending weight=%.2f",
                            str(site_no), float(w_ratio), float(cfg.usgs_width_ratio_max), float(w_usgs)
                        )
                    else:
                        log.info(
                            "[CALIB][USGS] site=%s width ratio=%.2f > %.2f; skipping anchor",
                            str(site_no), float(w_ratio), float(cfg.usgs_width_ratio_max)
                        )
                        continue

                # Reduce weight further if a-values are unstable
                if np.isfinite(a_cv) and a_cv > 0:
                    w_usgs *= min(1.0, float(cfg.usgs_a_cv_warn) / float(a_cv))

                # Compute USGS-implied Dmax at each XS using bank width (consistent with current parameterization)
                W = pd.to_numeric(xs_param.loc[m_apply, "width_m"], errors="coerce")
                dmax_usgs = float(a_site) * (W ** float(cfg.b))
                dmax_usgs = dmax_usgs.clip(cfg.dmin_m, cfg.dmax_m)

                # Blend with current estimate (prior or width-stage). Keep soundings untouched.
                d_cur = pd.to_numeric(xs_param.loc[m_apply, "dmax_raw_m"], errors="coerce")
                d_new = (1.0 - w_usgs) * d_cur + w_usgs * dmax_usgs

                xs_param.loc[m_apply, "dmax_raw_m"] = pd.to_numeric(d_new, errors="coerce")
                xs_param.loc[m_apply, "calib_src_detail"] = xs_param.loc[m_apply, "calib_src"].astype(str) + "+usgs"
                xs_param.loc[m_apply, "calib_src"] = "usgs"
                xs_param.loc[m_apply, "calib_n"] = int(n_meas)
                xs_param.loc[m_apply, "usgs_a_site"] = float(a_site)
                xs_param.loc[m_apply, "usgs_n_meas"] = int(n_meas)
                xs_param.loc[m_apply, "usgs_wt"] = float(w_usgs)
                xs_param.loc[m_apply, "usgs_a_cv"] = float(a_cv) if np.isfinite(a_cv) else np.nan
                xs_param.loc[m_apply, "usgs_width_med_m"] = float(w_meas) if np.isfinite(w_meas) else np.nan
                xs_param.loc[m_apply, "usgs_width_ratio"] = float(w_ratio) if np.isfinite(w_ratio) else np.nan

                log.info(
                    "[CALIB][USGS] site=%s n=%d a_site=%.6f w=%.2f applied=%d (q_range=%s)",
                    str(site_no), int(n_meas), float(a_site), float(w_usgs), int(m_apply.sum()), str(q_range)
                )
        except Exception as e:
            log.warning("[CALIB][USGS] failed to apply USGS measurement calibration: %s", e)

    # Clip and proceed

    # ---- Geomorphic envelope cap (stabilizes absolute depth scale) ----
    xs_param["dmax_env_m"] = np.nan
    xs_param["env_detail"] = ""
    if cfg.geomorphic_envelope_enabled:
        try:
            env = xs_param.apply(lambda r: _compute_dmax_geomorphic_envelope(r, cfg), axis=1)
            xs_param["dmax_env_m"] = env.apply(lambda t: float(t[0]) if isinstance(t, tuple) else float("nan"))
            xs_param["env_detail"] = env.apply(lambda t: str(t[1]) if isinstance(t, tuple) else "")
            d = pd.to_numeric(xs_param["dmax_raw_m"], errors="coerce")
            e = pd.to_numeric(xs_param["dmax_env_m"], errors="coerce")
            use = np.isfinite(d) & np.isfinite(e)
            if use.any():
                d_new = d.copy()
                d_new.loc[use] = np.minimum(d.loc[use].astype("float64"), e.loc[use].astype("float64"))
                xs_param["dmax_raw_m"] = pd.to_numeric(d_new, errors="coerce")
                if acct is not None:
                    acct["n_env_total"] = int(np.sum(use))
                    acct["n_env_clipped"] = int(np.sum(use & (d_new < d)))
                    try:
                        acct["env_clip_frac"] = float(acct["n_env_clipped"]) / float(max(1, acct["n_env_total"]))
                    except Exception:
                        pass
        except Exception as e:
            log.warning("[ENVELOPE] failed to apply geomorphic envelope cap: %s", e)

    xs_param["dmax_raw_m"] = pd.to_numeric(xs_param["dmax_raw_m"], errors="coerce").clip(cfg.dmin_m, cfg.dmax_m)

    # Smooth along stationing within component
    xs_param = xs_param.sort_values(["component_id", "s_center_m", "xs_id"]).reset_index(drop=True)
    xs_param["dmax_smooth_m"] = xs_param.groupby("component_id", dropna=False)["dmax_raw_m"].apply(
        lambda s: _rolling_smooth(s, cfg.smooth_window)
    ).reset_index(level=0, drop=True)
    xs_param["dmax_smooth_m"] = xs_param["dmax_smooth_m"].where(np.isfinite(xs_param["dmax_smooth_m"]), xs_param["dmax_raw_m"])

    xs_param["uncert_m"] = xs_param.apply(
        lambda r: _uncertainty_for_row(r, cfg),
        axis=1
    )

    param_map = xs_param.set_index("xs_id").to_dict(orient="index")

    # Build predicted points inside banks only
    keep_cols = ["xs_id", "dist_m", "z_dem", "z_topo", "is_bank_left", "is_bank_right", "river_id", "component_id", "geometry"]
    for c in keep_cols:
        if c not in xs_pts.columns:
            xs_pts[c] = np.nan
    xs_pts_geom = xs_pts[keep_cols].copy()
    xs_pts_geom["xs_id"] = xs_pts_geom["xs_id"].astype(str)

    pred_rows = []
    shape = str(cfg.xs_profile_shape or "linear_trapezoid").strip().lower()
    if shape in ("cosine", "cosine_trapezoid", "smooth", "smooth_trapezoid"):
        profile_fn = _cosine_trapezoid_depth_profile
    else:
        profile_fn = _trapezoid_depth_profile
    if bool(thalweg_only):
        # One control point per XS at the thalweg (deepest point)
        for _, xs in xs_lines.iterrows():
            xsid = str(xs.get('xs_id'))
            p = param_map.get(xsid)
            if p is None:
                continue
            bl = float(p.get('bank_left_dist_m', np.nan))
            br = float(p.get('bank_right_dist_m', np.nan))
            if not (np.isfinite(bl) and np.isfinite(br)):
                continue
            left = float(min(bl, br)); right = float(max(bl, br))
            W = float(right - left)
            if not (np.isfinite(W) and W > 0):
                continue
            off = float(p.get('thalweg_offset_frac', 0.0) or 0.0)
            # Convert offset_frac (fraction of width) into normalized position along XS line
            t = 0.5 + float(np.clip(off, -0.45, 0.45))
            t = float(np.clip(t, 0.05, 0.95))
            try:
                geom = xs.geometry.interpolate(t, normalized=True)
            except Exception:
                continue
            dist_from_left = float(np.clip(t, 0.0, 1.0)) * W
            d = left + dist_from_left
            Dmax = float(p.get('dmax_smooth_m', np.nan))
            wse = float(p.get('wse_m', np.nan))
            if not (np.isfinite(Dmax) and np.isfinite(wse)):
                continue
            depth = profile_fn(
                np.array([dist_from_left], dtype='float64'),
                W=W,
                Dmax=Dmax,
                bottom_frac=cfg.bottom_width_frac,
                offset_frac=off,
            )[0]
            if not np.isfinite(depth):
                continue
            z_bed = wse - float(depth)
            bank_min = np.nanmin([p.get('bank_left_z_m', np.nan), p.get('bank_right_z_m', np.nan)])
            if np.isfinite(bank_min) and np.isfinite(z_bed):
                z_bed = min(float(z_bed), float(bank_min) - 0.05)
            pred_rows.append(dict(
                xs_id=xsid,
                river_id=p.get('river_id', np.nan),
                component_id=int(p.get('component_id', -1)),
                s_center_m=float(p.get('s_center_m', np.nan)),
                curv_kappa_1pm=float(p.get('curv_kappa_1pm', np.nan)),
                thalweg_offset_frac=float(off),
                dist_m=float(d),
                width_m=float(W),
                wse_m=float(wse),
                dmax_raw_m=float(p.get('dmax_raw_m', np.nan)),
                dmax_smooth_m=float(Dmax),
                uncert_m=float(p.get('uncert_m', np.nan)),
                depth_pred_m=float(depth),
                z_bed_pred_m=float(z_bed),
                geometry=geom,
            ))
    else:
        for _, r in xs_pts_geom.iterrows():
            xsid = str(r["xs_id"])
            p = param_map.get(xsid)
            if p is None:
                continue

            bl = p["bank_left_dist_m"]
            br = p["bank_right_dist_m"]
            if not np.isfinite(bl) or not np.isfinite(br):
                continue

            left = float(min(bl, br))
            right = float(max(bl, br))
            W = float(right - left)
            if not np.isfinite(W) or W <= 0:
                continue

            d = float(r["dist_m"])
            if not (left <= d <= right):
                continue

            dist_from_left = d - left
            Dmax = float(p["dmax_smooth_m"])
            wse = float(p["wse_m"])

            depth = profile_fn(
                np.array([dist_from_left], dtype="float64"),
                W=W,
                Dmax=Dmax,
                bottom_frac=cfg.bottom_width_frac,
                offset_frac=float(p.get("thalweg_offset_frac", 0.0)),
            )[0]
            if not np.isfinite(depth):
                continue

            z_bed = wse - depth if np.isfinite(wse) else np.nan

            # guardrail: keep bed below lower bank by 5 cm
            bank_min = np.nanmin([p.get("bank_left_z_m", np.nan), p.get("bank_right_z_m", np.nan)])
            if np.isfinite(bank_min) and np.isfinite(z_bed):
                z_bed = min(z_bed, bank_min - 0.05)

            pred_rows.append(
                dict(
                    xs_id=xsid,
                    river_id=p.get("river_id", np.nan),
                    component_id=int(p.get("component_id", -1)),
                    s_center_m=float(p.get("s_center_m", np.nan)),
                    curv_kappa_1pm=float(p.get("curv_kappa_1pm", np.nan)),
                    thalweg_offset_frac=float(p.get("thalweg_offset_frac", 0.0)),
                    dist_m=d,
                    width_m=W,
                    wse_m=wse,
                    dmax_raw_m=float(p.get("dmax_raw_m", np.nan)),
                    dmax_smooth_m=Dmax,
                    uncert_m=float(p.get("uncert_m", np.nan)),
                    depth_pred_m=float(depth),
                    z_bed_pred_m=float(z_bed) if np.isfinite(z_bed) else np.nan,
                    geometry=r["geometry"],
                )
            )

    pred_gdf = gpd.GeoDataFrame(pred_rows, crs=xs_pts.crs)
    if pred_gdf.empty:
        raise RuntimeError("No predicted points generated (check bank picks and WSE estimation).")

    # XS summary
    xs_summary = xs_param.merge(xs_lines[["xs_id", "geometry"]], on="xs_id", how="left")
    xs_summary_gdf = gpd.GeoDataFrame(xs_summary, geometry="geometry", crs=xs_lines.crs)

    # Write GPKG
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    log.info("[WRITE] %s (xs_bathy_points=%d, xs_bathy_xs=%d)", out_gpkg, len(pred_gdf), len(xs_summary_gdf))
    pred_gdf.to_file(out_gpkg, layer="xs_bathy_points", driver="GPKG")
    xs_summary_gdf.to_file(out_gpkg, layer="xs_bathy_xs", driver="GPKG")

    # Optional rasters
    if out_bathy_raster or out_mask_raster or out_uncert_raster:
        tmpl_path = template_raster or dem_path
        if tmpl_path is None:
            raise RuntimeError("Raster outputs requested but no template raster available. Provide --template-raster or --dem.")
        with _open_template_raster(Path(tmpl_path)) as tmpl:
            if continuous_buffer_m is None:
                # Use a larger buffer to ensure continuity between cross-sections
                # Buffer should be large enough to span gaps between XS transects
                continuous_buffer_m = max(3.0 * _template_pixel_size_m(tmpl), 150.0, 20.0)

            if continuous.lower() == "median":
                bed_arr = _rasterize_points_reduce(
                    pred_gdf, raster_value_col, tmpl, nodata=nodata, reducer=str(overlap_reducer)
                )
                mask_arr = _make_mask(bed_arr, nodata=nodata).astype("uint8")
            else:
                try:
                    corridor_lines = xs_lines
                    if bool(thalweg_only) and str(continuous).lower().startswith("walid"):


                        # Densify thalweg spine(s) to at least 2 vertices per template pixel.
                        # For a 10 m raster, this defaults to 5 m spacing (>=2 vertices per pixel).
                        px = 10.0  # meters/pixel fallback
                        try:
                            import rasterio
                            with rasterio.open(str(template_raster)) as _src:
                                px = float(abs(_src.transform.a))
                                if not (px > 0):
                                    raise ValueError("non-positive pixel size")
                        except Exception as e:
                            log.warning("[THALWEG] Could not read template raster pixel size; defaulting px=10 m: %s", e)

                        if thalweg_densify_step_m is not None and float(thalweg_densify_step_m) > 0:
                            thalweg_densify_step_m_eff = float(thalweg_densify_step_m)
                        else:
                            thalweg_densify_step_m_eff = px * 0.5

                        # Clamp: never coarser than half-pixel; never finer than 1 m to avoid huge vertex counts.
                        thalweg_densify_step_m_eff = max(min(float(thalweg_densify_step_m_eff), px * 0.5), 1.0)
                        # Build a robust thalweg spine from the thalweg control points to define the along-channel axis.
                        # Avoid falling back to XS rungs as the axis (causes cross-channel "vertebrae" artifacts).
                        try:
                            # Use corridor buffer as a proxy for XS spacing if explicit spacing isn't available.
                            spacing_proxy = float(continuous_buffer_m) if continuous_buffer_m is not None else 200.0
                            jump = max(150.0, spacing_proxy * 2.5)
                            thalweg_lines_gdf = _build_thalweg_lines_from_points(
                                pred_gdf,
                                max_jump_m=jump,
                                densify_step_m=thalweg_densify_step_m_eff,
                                wse_col="wse_m",
                            )
                            if thalweg_lines_gdf is not None and (not thalweg_lines_gdf.empty):
                                corridor_lines = thalweg_lines_gdf
                        except Exception as e:
                            log.warning("[THALWEG] Failed to build spine from points (%s); using xs_lines axis.", e)

                    bed_arr, mask_arr = _continuous_surface(
                        pts_gdf=pred_gdf,
                        value_col=raster_value_col,
                        template_ds=tmpl,
                        method=continuous.lower(),
                        buffer_m=float(continuous_buffer_m),
                        k=int(continuous_k),
                        idw_power=float(idw_power),
                        aniso_along_scale_m=float(aniso_along_scale_m),
                        aniso_cross_scale_m=float(aniso_cross_scale_m),
                        thalweg_weight=float(thalweg_weight),
                        nodata=float(nodata),
                        overlap_reducer=str(overlap_reducer),
                        corridor_lines_gdf=corridor_lines,
                        channel_mask_raster=channel_mask_raster,
                        channel_mask_inside_value=int(channel_mask_inside_value),
                        channel_mask_invert=bool(channel_mask_invert),
                        max_query_dist_m=(float(max_query_dist_m) if max_query_dist_m is not None else None),
                    )
                except Exception as e:
                    log.warning("[CONTINUOUS] %s failed (%s); falling back to reducer='%s'", str(continuous), e, str(overlap_reducer))
                    bed_arr = _rasterize_points_reduce(
                        pred_gdf, raster_value_col, tmpl, nodata=nodata, reducer=str(overlap_reducer)
                    )
                    mask_arr = _make_mask(bed_arr, nodata=nodata).astype('uint8')

            if out_bathy_raster:
                out_bathy_raster = Path(out_bathy_raster)
                out_bathy_raster.parent.mkdir(parents=True, exist_ok=True)
                n_valid = int(np.sum(np.isfinite(bed_arr) & (bed_arr != float(nodata))))
                log.info("[WRITE] bathy raster -> %s (shape=%s valid=%d)", str(out_bathy_raster), bed_arr.shape, n_valid)
                _write_geotiff(out_bathy_raster, bed_arr, tmpl, nodata=float(nodata), dtype="float32")
                if not _exists_with_retry(out_bathy_raster):
                    log.warning("[WRITE] bathy raster missing after write: %s", str(out_bathy_raster))
                else:
                    log.info("[WRITE] bathy raster ok (%d bytes)", out_bathy_raster.stat().st_size)

                    # ------------------------------------------------------------------
                    # Explicit constraint metadata sidecar (NO filename guessing)
                    # ------------------------------------------------------------------
                    # Downstream pipeline stages should *not* infer which constraints were used.
                    # Write a deterministic sidecar next to the requested output raster.
                    try:
                        import json as _json

                        # Soundings constraint
                        snd_used = False
                        snd_total = 0
                        if "soundings_n" in xs_param.columns:
                            sn = pd.to_numeric(xs_param["soundings_n"], errors="coerce").fillna(0)
                            snd_total = int(sn.sum())
                            snd_used = bool((sn > 0).any())

                        # Drainage area constraint
                        da_used = False
                        if "drain_area_km2" in xs_param.columns:
                            da_used = bool(np.isfinite(pd.to_numeric(xs_param["drain_area_km2"], errors="coerce")).any())

                        # Slope constraint + source
                        slope_used = False
                        slope_source = "none"
                        if "slope_mpm" in xs_param.columns:
                            slope_used = bool(np.isfinite(pd.to_numeric(xs_param["slope_mpm"], errors="coerce")).any())
                            if slope_used:
                                slope_source = "network"
                                if "slope_proxy_mpm" in xs_param.columns and np.isfinite(pd.to_numeric(xs_param["slope_proxy_mpm"], errors="coerce")).any():
                                    slope_source = "proxy"
                                if "slope_wse_mpm" in xs_param.columns and np.isfinite(pd.to_numeric(xs_param["slope_wse_mpm"], errors="coerce")).any():
                                    slope_source = "wse_fit"

                        # WSE source (anchoring for bed elevations)
                        wse_source = "dem_proxy"
                        if "swot_wse_m" in xs_param.columns and np.isfinite(pd.to_numeric(xs_param["swot_wse_m"], errors="coerce")).any():
                            wse_source = "swot"

                        # Constraint level
                        if snd_used:
                            level = "CALIBRATED"
                        elif slope_used or da_used:
                            level = "PARTIALLY_CONSTRAINED"
                        else:
                            level = "PRIOR_ONLY"

                        meta = {
                            "constraints": {
                                "soundings_used": bool(snd_used),
                                "soundings_total_matched": int(snd_total),
                                "drainage_area_used": bool(da_used),
                                "slope_used": bool(slope_used),
                                "slope_source": str(slope_source),
                                "wse_source": str(wse_source),
                                "level": str(level),
                                "manning": {
                                    "enabled": bool(cfg.manning_mode.lower().strip() != "off"),
                                    "mode": cfg.manning_mode,
                                    "region": cfg.manning_region,
                                    "n_q_valid": int(np.sum(np.isfinite(pd.to_numeric(xs_param.get("manning_q_cms_used", np.nan), errors="coerce")) & (pd.to_numeric(xs_param.get("manning_q_cms_used", np.nan), errors="coerce") > 0))) if ("manning_q_cms_used" in xs_param.columns) else 0,
                                    "n_used": int(np.sum(pd.to_numeric(xs_param.get("manning_wt", 0.0), errors="coerce").fillna(0.0) > 0)) if ("manning_wt" in xs_param.columns) else 0,
                                    "q_source_counts": (xs_param.get("manning_flags", "").astype(str).str.split("\\|", n=1, expand=False).str[0].replace("", "unknown").value_counts(dropna=False).to_dict() if ("manning_flags" in xs_param.columns) else {}),
                                    "q2_equation_id_counts": (xs_param.get("manning_q2_equation_id", "").astype(str).replace("", "none").value_counts(dropna=False).to_dict() if ("manning_q2_equation_id" in xs_param.columns) else {}),
                                },
                            },
                            "energy_solver": {
                                "requested": cfg.energy_solver_enabled,
                                "enabled": bool(acct.get("energy_solver_enabled", False)),
                                "reason": str(acct.get("energy_solver_reason", "unknown")),
                                "wse_source": str(acct.get("energy_solver_wse_source", "none")),
                                "wse_anchor_source": str(acct.get("wse_anchor_source", "unknown")),
                                "n_total": int(acct.get("energy_solver_n_total", 0) or 0),
                                "n_applied": int(acct.get("energy_solver_n_applied", 0) or 0),
                                "blend_n": int(acct.get("energy_solver_blend_n", 0) or 0),
                                "delta_dmax_m_median": float(acct.get("energy_solver_delta_dmax_m_median", 0.0) or 0.0),
                                "delta_dmax_m_p95": float(acct.get("energy_solver_delta_dmax_m_p95", 0.0) or 0.0),
                            },
                            "outputs": {
                                "out_bathy_raster": str(out_bathy_raster),
                                "out_gpkg": str(out_gpkg),
                            },
                        }

                        meta_path = Path(out_meta_json) if out_meta_json else Path(str(out_bathy_raster) + ".meta.json")
                        meta_path.write_text(_json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
                        log.info("[WRITE] constraint meta -> %s", str(meta_path))
                    except Exception as e:
                        log.warning("[WRITE] Failed to write constraint meta sidecar (%s)", e)

            if out_mask_raster:
                out_mask_raster = Path(out_mask_raster)
                out_mask_raster.parent.mkdir(parents=True, exist_ok=True)
                n_valid = int(np.sum(mask_arr.astype(bool)))
                log.info("[WRITE] mask raster -> %s (shape=%s valid=%d)", str(out_mask_raster), mask_arr.shape, n_valid)
                _write_geotiff(out_mask_raster, mask_arr.astype("uint8"), tmpl, nodata=0, dtype="uint8")
                if not _exists_with_retry(out_mask_raster):
                    log.warning("[WRITE] mask raster missing after write: %s", str(out_mask_raster))
                else:
                    log.info("[WRITE] mask raster ok (%d bytes)", out_mask_raster.stat().st_size)

            if out_uncert_raster:
                out_uncert_raster = Path(out_uncert_raster)
                out_uncert_raster.parent.mkdir(parents=True, exist_ok=True)
                # uncertainty: still rasterize median per pixel; continuous uncertainty can come later
                # REUSE the optimized reducer instead of old _rasterize_points_median
                unc_arr = _rasterize_points_reduce(
                    pred_gdf, raster_uncert_col, tmpl, nodata=float(nodata), reducer="median"
                )
                n_valid = int(np.sum(np.isfinite(unc_arr) & (unc_arr != float(nodata))))
                log.info("[WRITE] uncert raster -> %s (shape=%s valid=%d)", str(out_uncert_raster), unc_arr.shape, n_valid)
                _write_geotiff(out_uncert_raster, unc_arr, tmpl, nodata=float(nodata), dtype="float32")
                if not _exists_with_retry(out_uncert_raster):
                    log.warning("[WRITE] uncert raster missing after write: %s", str(out_uncert_raster))
                else:
                    log.info("[WRITE] uncert raster ok (%d bytes)", out_uncert_raster.stat().st_size)

    
    # --------------------------------------------------------------------------------------
    # Strict output validation
    # --------------------------------------------------------------------------------------
    # If raster outputs were requested, ensure they were actually written.

    # Final derived accounting (calibration sources)
    try:
        if "calib_src" in xs_param.columns:
            vc = xs_param["calib_src"].value_counts(dropna=False).to_dict()
            acct["calib_src_counts"] = {str(k): int(v) for k, v in vc.items()}
            acct["n_width_stage_applied"] = int(vc.get("width_stage", 0))
            acct["n_usgs_applied"] = int(vc.get("usgs", 0))
            acct["n_soundings_calib_applied"] = int(vc.get("soundings", 0))
    except Exception:
        pass


    # Attribute availability accounting (helps diagnose under-constraint).
    # Note: prior-mode-specific functions only update these counters in some modes;
    # compute them here for consistency across all runs.
    try:
        da = pd.to_numeric(xs_param.get("drain_area_km2", np.nan), errors="coerce")
        acct["n_with_da"] = int(np.sum(np.isfinite(da) & (da > 0)))
        sl = pd.to_numeric(xs_param.get("slope_mpm", np.nan), errors="coerce")
        acct["n_with_slope"] = int(np.sum(np.isfinite(sl) & (sl > 0)))
    except Exception:
        pass

    # Optional: write constraint accounting JSON (explicit path; no guessing)
    if out_accounting_json is not None:
        out_accounting_json = Path(out_accounting_json)
        out_accounting_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_accounting_json.with_suffix(out_accounting_json.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(acct, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(str(tmp), str(out_accounting_json))
        _fsync_dir(out_accounting_json.parent)

    if out_bathy_raster is not None and not _exists_with_retry(out_bathy_raster):
        raise RuntimeError(f"Requested bathy raster was not written: {out_bathy_raster}. "
                           f"This usually means no valid bathy points survived filtering or the interpolation domain was empty. "
                           f"Check that river_bathy.gpkg exists and contains inferred points.")
    if out_mask_raster is not None and not _exists_with_retry(out_mask_raster):
        raise RuntimeError(f"Requested mask raster was not written: {out_mask_raster}")
    if out_uncert_raster is not None and not _exists_with_retry(out_uncert_raster):
        raise RuntimeError(f"Requested uncertainty raster was not written: {out_uncert_raster}")

    log.info("[DONE] Inference complete.")



    # --------------------------------------------------------------------------------------
    # CLI
    # --------------------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "xs_infer_bathy_raster.py – infer river bathy from XS + banks + optional soundings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Primary inference mode
    p.add_argument("--xs-gpkg", default=None, help="XS GeoPackage produced by xs_builder.py (must contain xs_lines/xs_points)")
    p.add_argument("--out-gpkg", default=None, help="Output GeoPackage with inferred bathy layers")
    p.add_argument("--dem", default=None, help="Optional DEM path used only as default template raster for GeoTIFF outputs")
    p.add_argument("--xs-lines-layer", default="xs_lines")
    p.add_argument("--xs-points-layer", default="xs_points")

    # Optional calibration
    # Optional calibration soundings (measured bathymetry).
    # You may repeat the flag, pass a comma-separated list, or pass a directory of files.
    p.add_argument(
        "--soundings", dest="soundings", action="append", default=[],
        help="Optional measured depth points (GPKG/GeoJSON/SHP/CSV/XYZ). Can repeat, comma-separate, or provide a directory."
    )
    # Preferred unified alias (same destination).
    p.add_argument(
        "--extra-xyz", dest="soundings", action="append",
        help="Alias for --soundings (preferred). Accepts GPKG/GeoJSON/SHP/CSV/XYZ; can repeat, comma-separate, or provide a directory."
    )
    p.add_argument("--soundings-depth-col", default=None, help="Depth column name (positive-down).")
    p.add_argument("--soundings-elev-col", default=None, help="Elevation column name (meters).")
    p.add_argument("--soundings-x-col", default=None, help="CSV X column (lon/easting).")
    p.add_argument("--soundings-y-col", default=None, help="CSV Y column (lat/northing).")
    p.add_argument("--soundings-crs", default=None, help="CRS of soundings XY for CSV/XYZ inputs (e.g., EPSG:26919). If omitted, CRS is guessed (lon/lat→EPSG:4326; otherwise falls back to target CRS when projected).")
    p.add_argument(
        "--write-soundings-subset",
        default=None,
        help="Optional output path to write the (possibly downsampled) unified soundings set for reuse by downstream river steps. Preferred: .parquet; fallback: .gpkg.",
    )
    p.add_argument(
        "--only-write-soundings-subset",
        action="store_true",
        help="If set, load + downsample soundings, write --write-soundings-subset, print a one-line summary, then exit 0 (no XS inference).",
    )
    p.add_argument(
        "--soundings-subset",
        default=None,
        dest="soundings_subset",
        help=(
            "Path to a pre-clipped, validated soundings parquet produced by Pass 1 "
            "(--only-write-soundings-subset). When provided, this file is used INSTEAD of "
            "--soundings for calibration so the two-pass handoff is an explicit, auditable "
            "step. The script will hard-fail (rc=2) if this file is missing or contains 0 "
            "rows, preventing the 'Soundings empty' silent-success bug."
        ),
    )
    p.add_argument("--calib-max-dist-m", type=float, default=200.0, help="Max distance from sounding to XS line to use")
    p.add_argument("--calib-stat", choices=["p90", "max", "median"], default="p90", help="Per-XS depth stat from soundings")

    # Tier-1 anchors (optional)
    p.add_argument("--usgs-sites", default=None, help="Comma-separated USGS site numbers to use as at-a-station anchors.")
    p.add_argument("--usgs-start", default=None, help="Start date (YYYY-MM-DD) for pulling USGS discharge measurements.")
    p.add_argument("--usgs-end", default=None, help="End date (YYYY-MM-DD) for pulling USGS discharge measurements.")
    p.add_argument("--usgs-cache-dir", default=None, help="Optional directory to cache NWIS responses.")
    p.add_argument("--usgs-max-dist-m", type=float, default=5000.0, help="Max distance (m) from gage to XS center for applying USGS anchor.")
    p.add_argument(
        "--usgs-mean-to-dmax",
        default="auto",
        help="Convert mean depth (area/width) to Dmax. Use a number (e.g., 1.3) or 'auto' to use 2/(1+bottom_width_frac) for a trapezoid.",
    )
    p.add_argument("--usgs-a-stat", choices=["median", "p90", "mean"], default="median", help="How to aggregate a_site from multiple measurements.")
    p.add_argument("--usgs-q-quantile-lo", type=float, default=0.20, help="Lower discharge quantile for filtering USGS measurements (0-1).")
    p.add_argument("--usgs-q-quantile-hi", type=float, default=0.80, help="Upper discharge quantile for filtering USGS measurements (0-1).")
    p.add_argument("--usgs-a-cv-warn", type=float, default=0.50, help="Warn/soften anchor when coefficient-of-variation of fitted a-values exceeds this.")
    p.add_argument("--usgs-width-ratio-max", type=float, default=3.0, help="Threshold for (bank width / median measured wet width) to down-weight/skip USGS anchor.")
    p.add_argument("--usgs-width-ratio-blend", action="store_true", help="Blend (down-weight) USGS anchor when width ratio is large (default).")
    p.add_argument("--no-usgs-width-ratio-blend", dest="usgs_width_ratio_blend", action="store_false", help="Disable blending; skip USGS anchor when width ratio is large.")
    p.set_defaults(usgs_width_ratio_blend=True)
    p.add_argument("--gage-snap-max-dist-m", type=float, default=1000.0, help="Max distance (m) to snap gages to the river network for river_id assignment.")

    p.add_argument("--width-stage-csv", action="append", default=[], help="Optional width-stage CSV(s). Repeat the flag and/or pass comma-separated lists.")
    p.add_argument("--width-stage-max-dist-m", type=float, default=5000.0, help="Max distance (m) from station to XS center for applying width-stage anchor.")
    p.add_argument("--width-stage-min-n", type=int, default=6, help="Minimum number of width-stage observations to fit the stage-width slope.")
    p.add_argument("--width-stage-min-r2", type=float, default=0.25, help="Minimum R^2 for the width–stage fit to be trusted.")

    # Optional Manning inversion prior (secondary, blended; requires Q + W + S)
    p.add_argument("--manning-enabled", action="store_true", help="Alias: enable Manning prior using --manning-mode=q2_regional.")
    p.add_argument("--manning-mode", choices=["off","constant","from_field","q2_regional"], default="off",
               help="Blend a Manning-based Dmax prior into the base prior. off=disabled; constant=use one Q; from_field=read Q from river attributes.")
    p.add_argument("--manning-q-cms", type=float, default=None, help="Discharge Q (m^3/s) used when --manning-mode=constant.")
    p.add_argument("--manning-q-field", default=None, help="River attribute field containing Q (m^3/s) used when --manning-mode=from_field.")
    p.add_argument("--manning-n", type=float, default=0.035, help="Manning roughness n (typical 0.03-0.08).")
    p.add_argument("--manning-region", default="default", help="Region key for --manning-mode=q2_regional (used with drainage area). Provide published coefficients in manning_inversion.py or override in code.")
    p.add_argument("--manning-n-by-region", action="append", default=[],
               help="Optional mapping region=n (repeatable), e.g. --manning-n-by-region default=<n> --manning-n-by-region piedmont=<n>. If provided, overrides --manning-n for matching region.")
    p.add_argument("--manning-min-confidence", type=float, default=0.30, help="Minimum confidence required to apply Manning prior (0-1).")
    p.add_argument("--manning-max-weight", type=float, default=0.60, help="Maximum blend weight for Manning prior (0-1).")
    p.add_argument("--manning-backwater-slope-thresh", type=float, default=1e-4, help="Disable Manning prior when slope is below this (backwater/tidal risk).")
    p.add_argument("--manning-dist-to-mouth-field", default=None, help="Optional river attribute field containing distance-to-mouth (km).")
    p.add_argument("--manning-dist-to-mouth-km-max", type=float, default=10.0, help="If dist-to-mouth is provided, disable Manning prior when distance <= this (km).")

    p.add_argument("--no-auto-manning-when-no-soundings", dest="auto_manning_when_no_soundings", action="store_false",
                   help="Disable auto-enabling the q2_regional Manning prior when no soundings are provided.")
    p.set_defaults(auto_manning_when_no_soundings=True)


    # Regional hydraulic geometry curves (Drainage Area -> bankfull depth)
    p.add_argument("--regional-curve-enabled", action="store_true", help="Enable regional hydraulic geometry curve prior (DA -> bankfull depth) as a soft prior.")
    p.add_argument("--regional-curve-region", default=None, help="Region key for built-in coefficients (illustrative placeholders). To use built-ins you must pass this explicitly. Prefer providing --regional-curve-c/--regional-curve-f from published curves.")
    p.add_argument("--allow-builtin-regional-curves", action="store_true", help="Allow using built-in regional curve coefficients (illustrative placeholders). Prefer published coefficients via --regional-curve-c/--regional-curve-f.")
    p.add_argument("--regional-curve-c", type=float, default=None, help="Coefficient c in D_bkf = c * DA^f (units depend on --regional-curve-da-units and --regional-curve-depth-units).")
    p.add_argument("--regional-curve-f", type=float, default=None, help="Exponent f in D_bkf = c * DA^f.")
    p.add_argument("--regional-curve-da-units", choices=["km2","mi2"], default="km2", help="Drainage area units expected by the curve coefficients.")
    p.add_argument("--regional-curve-depth-units", choices=["m","ft"], default="m", help="Depth units of the curve coefficients.")
    p.add_argument("--regional-curve-depth-type", choices=["mean","max"], default="mean", help="Whether curve depth represents mean or max depth at bankfull.")
    p.add_argument("--regional-curve-to-dmax", choices=["auto","factor"], default="auto", help="Conversion from bankfull depth to Dmax. 'auto' uses trapezoid mean->Dmax conversion; 'factor' uses --regional-curve-to-dmax-factor.")
    p.add_argument("--regional-curve-to-dmax-factor", type=float, default=1.25, help="Used when --regional-curve-to-dmax=factor.")
    p.add_argument("--regional-curve-unc-pct", type=float, default=40.0, help="Uncertainty percent (used to downweight).")
    p.add_argument("--regional-curve-max-weight", type=float, default=0.60, help="Maximum blend weight for regional-curve prior (0-1).")
    # Geomorphic envelope cap (DA/region-based upper bound on Dmax)
    p.add_argument("--no-geomorphic-envelope", action="store_true", help="Disable geomorphic envelope cap (regional-curve-based Dmax upper bound).")
    p.add_argument("--geomorphic-envelope-inflate-unc", action="store_true", help="Inflate envelope by (1 + regional_curve_unc_pct/100). Default: on.")
    p.add_argument("--geomorphic-envelope-no-inflate-unc", action="store_true", help="Do not inflate envelope by regional_curve_unc_pct.")
    p.add_argument("--geomorphic-envelope-only-when-no-soundings", action="store_true", help="Apply envelope cap only for XS not calibrated by soundings (default behavior).")
    p.add_argument("--geomorphic-envelope-always", action="store_true", help="Apply envelope cap even when soundings exist (not recommended).")
    p.add_argument("--regional-curve-min-da-km2", type=float, default=1.0, help="Ignore DA smaller than this (km^2) for the curve prior.")

    p.add_argument("--width-stage-max-weight", type=float, default=0.8, help="Maximum blend weight for width–stage anchor.")

    # Optional river attributes (for multivariate priors)
    p.add_argument("--river-gpkg", default=None, help="Optional river network GPKG to attach reach attributes (e.g., drainage area, slope).")
    p.add_argument("--rivers-layer", default="rivers_clip", help="Layer within --river-gpkg containing per-reach attributes.")
    p.add_argument("--drain-area-field", default=None, help="Field name in rivers layer for drainage area (km^2). If omitted, tries common names.")
    p.add_argument("--slope-field", default=None, help="Field name in rivers layer for slope (m/m). If omitted, tries common names.")

    # Bed model + priors
    p.add_argument("--bottom-width-frac", type=float, default=0.30, help="Flat bottom width fraction of channel width")
    p.add_argument(
        "--xs-profile-shape",
        choices=["linear_trapezoid", "cosine_trapezoid"],
        default="cosine_trapezoid",
        help=(
            "Cross-section depth profile family. 'cosine_trapezoid' keeps the same width-constrained "
            "trapezoid concept but uses smooth cosine side slopes (reduces corner artifacts)."
        ),
    )
    p.add_argument("--a", type=float, default=0.18, help="Width→depth prior coefficient (Dmax=a*W^b)")
    p.add_argument("--b", type=float, default=0.50, help="Width→depth prior exponent")
    p.add_argument("--dmin-m", type=float, default=0.50, help="Minimum Dmax (m)")
    p.add_argument("--dmax-m", type=float, default=15.0, help="Maximum Dmax (m)")

    # Prior mode upgrades
    p.add_argument(
        "--prior-mode",
        choices=["powerlaw", "multivariate"],
        default="powerlaw",
        help=(
            "Dmax prior mode. 'powerlaw' uses Dmax=a*W^b. "
            "'multivariate' uses W plus optional reach attributes (drainage area, slope). "
            "Tip: for regional curves DA→Dbkf (Dbkf=c*DA^f), set --prior-mode=multivariate, "
            "--mv-bw=0, --mv-ba=<f>, --mv-a0=<c>."
        ),
    )
    p.add_argument("--mv-a0", type=float, default=0.18, help="Multivariate prior base coefficient (a0)")
    p.add_argument("--mv-bw", type=float, default=0.50, help="Multivariate prior width exponent")
    p.add_argument("--mv-ba", type=float, default=0.0, help="Multivariate prior drainage-area exponent")
    p.add_argument("--mv-bs", type=float, default=-0.10, help="Multivariate prior slope exponent (negative => deeper at lower slope)")
    p.add_argument("--mv-eps-a", type=float, default=1.0, help="Additive epsilon for drainage area (km^2) to avoid zeros")
    p.add_argument("--mv-eps-s", type=float, default=1e-4, help="Additive epsilon for slope (m/m) to avoid zeros")

    # Slope proxy stabilization (used when slope attribute is missing)
    p.add_argument("--slope-proxy-window", type=int, default=9, help="Rolling median window (XS count) for WSE smoothing before slope differencing.")
    p.add_argument("--slope-min", type=float, default=1e-5, help="Minimum plausible slope (m/m) for slope proxy.")
    p.add_argument("--slope-max", type=float, default=0.05, help="Maximum plausible slope (m/m) for slope proxy.")
    p.add_argument("--slope-proxy-min-n", type=int, default=7, help="Minimum XS per river_id to compute slope proxy.")

    # Longitudinal WSE profile fitting (preferred slope when reach slope attribute is missing)
    p.add_argument("--wse-profile-enabled", dest="wse_profile_enabled", action="store_true", help="Enable longitudinal WSE profile fitting (default).")
    p.add_argument("--no-wse-profile", dest="wse_profile_enabled", action="store_false", help="Disable WSE profile fitting; use legacy slope proxy only.")
    p.set_defaults(wse_profile_enabled=True)
    p.add_argument("--wse-profile-window", type=int, default=9, help="Rolling window (XS count) for WSE profile smoothing.")
    p.add_argument("--wse-profile-min-n", type=int, default=7, help="Minimum XS per river_id required to fit a WSE profile.")
    p.add_argument("--wse-profile-monotonic", dest="wse_profile_monotonic", action="store_true", help="Enforce monotonic WSE along stationing (default).")
    p.add_argument("--no-wse-profile-monotonic", dest="wse_profile_monotonic", action="store_false", help="Disable monotonic constraint.")
    p.set_defaults(wse_profile_monotonic=True)
    p.add_argument("--enable-1d-energy-solver", dest="energy_solver_enabled", action="store_true",
               help="Enable reach-scale 1D energy-consistent depth solver (Option A). Default: off.")
    p.add_argument("--no-1d-energy-solver", dest="energy_solver_enabled", action="store_false",
               help="Disable 1D energy solver.")
    p.set_defaults(energy_solver_enabled=False)
    p.add_argument("--energy-allow-dem-proxy-wse", dest="energy_allow_dem_proxy_wse", action="store_true",
               help="Allow energy solver to run even when WSE anchoring is DEM/topo proxy only (no observed stage). Default: off (safety gate).")
    p.set_defaults(energy_allow_dem_proxy_wse=False)
    p.add_argument("--out-1d-solver-inputs-json", dest="out_1d_solver_inputs_json", default=None,
               help="Write 1D solver station inputs JSON to this explicit path.")
    p.add_argument("--out-1d-solver-outputs-json", dest="out_1d_solver_outputs_json", default=None,
               help="Write 1D solver station outputs JSON to this explicit path.")
    p.add_argument("--out-1d-solver-accounting-json", dest="out_1d_solver_accounting_json", default=None,
               help="Write 1D solver accounting JSON to this explicit path.")

    # WSE proxy

    p.add_argument("--wse-center-frac", type=float, default=0.10, help="Center window fraction of XS length")
    p.add_argument("--wse-quantile", type=float, default=0.10, help="Quantile of center window elevations for WSE proxy")
    p.add_argument("--wse-fallback-drop-m", type=float, default=0.50, help="Fallback WSE= min(bank_z)-drop (m)")

    # Optional SWOT (or other) WSE observations (stage anchoring)
    p.add_argument("--swot-wse", default=None, help="Optional point observations of water-surface elevation (WSE). GPKG/SHP/GeoJSON points or CSV.")
    p.add_argument("--swot-wse-col", default="wse_m", help="Column name holding WSE in meters in --swot-wse.")
    p.add_argument("--swot-x-col", default=None, help="For CSV --swot-wse: X column (lon or easting).")
    p.add_argument("--swot-y-col", default=None, help="For CSV --swot-wse: Y column (lat or northing).")
    p.add_argument("--swot-csv-crs", default="EPSG:4326", help="CRS for CSV lon/lat or x/y columns (default EPSG:4326).")
    p.add_argument("--swot-max-dist-m", type=float, default=1000.0, help="Maximum distance (m) from XS center to accept a WSE observation.")
    p.add_argument("--swot-stage-blend", type=float, default=1.0, help="Blend factor for WSE when observations exist: 0 uses DEM proxy; 1 replaces proxy.")
    p.add_argument("--swot-vertical-offset-m", type=float, default=0.0, help="Additive offset (m) applied to observed WSE before blending (for datum tweaks).")
    p.add_argument("--swot-outlier-mad-z", type=float, default=4.0, help="Outlier threshold (robust MAD z-score) for WSE observations along-channel.")
    p.add_argument("--swot-use-for-slope", dest="swot_use_for_slope", action="store_true", help="Use blended WSE to drive WSE-profile slope fitting (default).")
    p.add_argument("--no-swot-use-for-slope", dest="swot_use_for_slope", action="store_false", help="Do NOT use WSE observations for slope fitting; only shift bed elevations.")
    p.set_defaults(swot_use_for_slope=True)

    # Curvature-driven XS asymmetry (optional; thalweg skew proxy)
    p.add_argument(
        "--curv-asymmetry-enabled",
        dest="curv_asymmetry_enabled",
        action="store_true",
        help="Enable curvature-driven skew of the XS depth profile (shift trapezoid flat-bottom toward outer bend).",
    )
    p.add_argument("--curv-window-m", type=float, default=500.0, help="Along-channel window length (m) for local curvature estimation.")
    p.add_argument("--curv-min-points", type=int, default=7, help="Minimum XS points required in the curvature fit window.")
    p.add_argument("--curv-kappa-scale-1pm", type=float, default=0.002, help="Curvature scale (1/m) for tanh mapping to offset.")
    p.add_argument("--curv-max-offset-frac", type=float, default=0.25, help="Max shift as fraction of width (0..0.45).")
    p.add_argument("--curv-lag-m", type=float, default=0.0, help="Evaluate curvature at s+lag (m) to model downstream thalweg response.")

    # Smoothing
    p.add_argument("--smooth-window", type=int, default=7, help="Rolling window (XS count) for Dmax smoothing")
    p.add_argument("--allow-missing-banks", action="store_true", help="Process XS even if banks aren't detected")

    p.add_argument("--thalweg-only", action="store_true",
                   help="Output only one predicted point per cross-section at the thalweg (deepest point). This reduces overlap artifacts and enforces a continuous channel spine for interpolation.")
    p.add_argument("--thalweg-densify-step-m", type=float, default=None,
                   help="Vertex spacing (m) used to densify thalweg spine(s). Default: half the template raster pixel size (>=2 vertices per pixel).")

    # Raster outputs
    p.add_argument("--template-raster", default=None, help="Template raster to align GeoTIFF outputs (usually your CUDEM DEM). If omitted, uses --dem when available.")
    p.add_argument("--channel-mask-raster", default=None, help="Optional DEM-aligned channel mask raster to constrain interpolation (1=inside by default).")
    p.add_argument("--channel-mask-inside-value", type=int, default=1, help="Raster value treated as inside-channel (default 1). For masks where water=0, set this to 0 or use --channel-mask-invert.")
    p.add_argument("--channel-mask-invert", action="store_true", help="Invert channel mask logic (inside becomes outside). Useful if mask uses 1=land and 0=water.")
    p.add_argument("--max-query-dist-m", type=float, default=None,
                   help="Optional max distance (meters) from any control point to allow interpolation. Pixels farther than this are dropped from the river surface/mask.")
    p.add_argument("--out-bathy-raster", default=None, help="Optional output GeoTIFF of predicted bed elevation (z_bed_pred_m) on template grid")
    p.add_argument("--out-mask-raster", default=None, help="Optional output GeoTIFF mask (1 where bathy raster has data)")
    p.add_argument("--out-uncert-raster", default=None, help="Optional output GeoTIFF uncertainty (meters) on template grid")
    p.add_argument(
        "--out-accounting-json",
        default=None,
        help="Optional output JSON with constraint-accounting statistics (e.g., % of XS where priors hit dmin/dmax, and which priors/anchors were applied).",
    )
    p.add_argument(
        "--out-meta-json",
        default=None,
        help="Optional output JSON for constraint metadata sidecar (avoids deriving/guessing a filename from --out-bathy-raster).",
    )

    # Continuous surface options (affects raster outputs only)
    p.add_argument("--continuous", choices=["median", "walid", "aidw", "aniso", "walid_aniso"], default="walid_aniso",
                   help="How to produce the patch raster surface from bathy points.")
    p.add_argument("--continuous-buffer-m", type=float, default=None,
                   help="Buffer (meters) around points used to define interpolation corridor. Default scales with pixel size.")
    p.add_argument("--continuous-k", type=int, default=12, help="K nearest points used for IDW/AIDW.")
    p.add_argument("--idw-power", type=float, default=2.0, help="IDW power for --continuous modes.")
    p.add_argument("--aniso-along-scale-m", type=float, default=500.0, help="Anisotropic IDW along-channel scale (meters). Larger => smoother along-channel influence.")
    p.add_argument("--aniso-cross-scale-m", type=float, default=30.0, help="Anisotropic IDW cross-channel scale (meters). Smaller => stronger bank barrier effect.")
    p.add_argument("--thalweg-weight", type=float, default=6.0, help="Extra influence for thalweg control points in walid mode (approx. duplicates).")
    p.add_argument("--nodata", type=float, default=-9999.0, help="Nodata value for float rasters.")
    p.add_argument(
        "--overlap-reducer",
        choices=["min", "median"],
        default="min",
        help="When multiple points fall in the same output pixel (common where streams overlap), how to collapse them. 'min' keeps the deeper bed / most-negative depth.",
    )

    # Raster-only mode (skip inference)
    p.add_argument("--raster-from-gpkg", default=None,
                   help="If set, skip inference and rasterize values from this GPKG (layer --raster-layer).")
    p.add_argument("--raster-layer", default="xs_bathy_points",
                   help="Layer name inside --raster-from-gpkg to rasterize.")
    p.add_argument("--raster-value-col", default="z_bed_pred_m",
                   help="Column to rasterize for bed GeoTIFF (e.g., z_bed_pred_m or z_bed_adj_m).")
    p.add_argument("--raster-uncert-col", default="uncert_m",
                   help="Column to rasterize for uncertainty GeoTIFF.")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Parse optional Manning n-by-region mapping (explicit; no built-in defaults).
    manning_n_by_region = {}
    try:
        for item in (getattr(args, "manning_n_by_region", None) or []):
            if not item:
                continue
            if "=" not in str(item):
                continue
            k, v = str(item).split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            try:
                fv = float(v)
                if np.isfinite(fv) and fv > 0:
                    manning_n_by_region[k] = fv
            except Exception:
                continue
    except Exception:
        manning_n_by_region = {}

    # Geomorphic envelope toggles (defaults are conservative)
    geomorphic_envelope_enabled = not bool(getattr(args, "no_geomorphic_envelope", False))
    inflate_unc = True
    if bool(getattr(args, "geomorphic_envelope_no_inflate_unc", False)):
        inflate_unc = False
    if bool(getattr(args, "geomorphic_envelope_inflate_unc", False)):
        inflate_unc = True
    only_no_soundings = True
    if bool(getattr(args, "geomorphic_envelope_always", False)):
        only_no_soundings = False
    if bool(getattr(args, "geomorphic_envelope_only_when_no_soundings", False)):
        only_no_soundings = True


    # Provide river network path to the continuous interpolator (best-effort)
    # for junction/confluence artifact suppression.
    try:
        _continuous_surface._river_gpkg = args.river_gpkg
    except Exception:
        log.debug("Optional step failed; continuing.", exc_info=True)

    # Backwards-compatible alias
    if getattr(args, "manning_enabled", False) and str(getattr(args, "manning_mode", "off")) == "off":
        args.manning_mode = "q2_regional"

    # Raster-only mode
    if args.raster_from_gpkg:
        if not args.template_raster:
            raise RuntimeError("Raster-only mode requires --template-raster")

        gpkg = Path(args.raster_from_gpkg)
        layer = str(args.raster_layer)
        value_col = str(args.raster_value_col)
        uncert_col = str(args.raster_uncert_col)

        pred_gdf = gpd.read_file(gpkg, layer=layer)
        if pred_gdf.empty:
            raise RuntimeError(f"Raster-only: layer '{layer}' is empty in {gpkg}")
        if pred_gdf.crs is None:
            raise RuntimeError(f"Raster-only: layer '{layer}' has no CRS in {gpkg}")
        if value_col not in pred_gdf.columns:
            raise RuntimeError(f"Raster-only: missing '{value_col}' in layer '{layer}'")

        with _open_template_raster(Path(args.template_raster)) as tmpl:
            if args.continuous.lower() == "median":
                bed = _rasterize_points_reduce(
                    pred_gdf, value_col, tmpl, nodata=float(args.nodata), reducer=str(args.overlap_reducer)
                )
                mask = _make_mask(bed, nodata=float(args.nodata)).astype("uint8")
            else:
                bed, mask = _continuous_surface(
                    pts_gdf=pred_gdf,
                    value_col=value_col,
                    template_ds=tmpl,
                    method=args.continuous.lower(),
                    buffer_m=float(args.continuous_buffer_m) if args.continuous_buffer_m is not None else max(3.0 * _template_pixel_size_m(tmpl), 150.0, 20.0),
                    k=int(args.continuous_k),
                    idw_power=float(args.idw_power),
                    thalweg_weight=float(args.thalweg_weight),
                    nodata=float(args.nodata),
                    overlap_reducer=str(args.overlap_reducer),
                    channel_mask_raster=Path(args.channel_mask_raster) if args.channel_mask_raster else None,
                    channel_mask_inside_value=int(args.channel_mask_inside_value),
                    channel_mask_invert=bool(args.channel_mask_invert),
                    max_query_dist_m=(float(args.max_query_dist_m) if args.max_query_dist_m is not None else None),
                    aniso_along_scale_m=float(args.aniso_along_scale_m),
                    aniso_cross_scale_m=float(args.aniso_cross_scale_m)
                )

            if args.out_bathy_raster:
                _write_geotiff(Path(args.out_bathy_raster), bed, tmpl, nodata=float(args.nodata), dtype="float32")
                log.info("[WRITE] %s", str(args.out_bathy_raster))

            if args.out_mask_raster:
                _write_geotiff(Path(args.out_mask_raster), mask.astype("uint8"), tmpl, nodata=0.0, dtype="uint8")
                log.info("[WRITE] %s", str(args.out_mask_raster))

            if args.out_uncert_raster:
                if uncert_col in pred_gdf.columns:
                    unc = _rasterize_points_reduce(pred_gdf, uncert_col, tmpl, nodata=float(args.nodata), reducer="median")
                else:
                    unc = np.full((tmpl.height, tmpl.width), float(args.nodata), dtype="float32")
                _write_geotiff(Path(args.out_uncert_raster), unc, tmpl, nodata=float(args.nodata), dtype="float32")
                log.info("[WRITE] %s", str(args.out_uncert_raster))

        log.info("[DONE] Raster-only complete.")
        return

    # Inference mode requires xs-gpkg and out-gpkg
    if not args.xs_gpkg or not args.out_gpkg:
        raise RuntimeError("Inference mode requires --xs-gpkg and --out-gpkg (or use --raster-from-gpkg for raster-only).")

    # Hard requirement: if the 1D energy solver is enabled, the longitudinal WSE fitter
    # must be available. Silent fallbacks make runs look 'successful' while skipping
    # the intended physics stabilization.
    if bool(getattr(args, "enable_1d_energy_solver", False)) and bool(getattr(args, "wse_profile_enabled", True)):
        if fit_wse_profile is None or WSEFitConfig is None:
            raise RuntimeError(
                "1D energy solver requested, but river_wse.fit_wse_profile is unavailable. "
                "Ensure river_wse.py is on PYTHONPATH and imports succeed."
            )

    # Parse mean→max depth conversion. Use 'auto' to derive Dmax/mean for a trapezoid: 2/(1+bottom_width_frac).
    if str(args.usgs_mean_to_dmax).strip().lower() == "auto":
        usgs_mean_to_dmax_val = -1.0
    else:
        usgs_mean_to_dmax_val = float(args.usgs_mean_to_dmax)

    cfg = InferConfig(
        bottom_width_frac=float(args.bottom_width_frac),
        xs_profile_shape=str(getattr(args, "xs_profile_shape", "cosine_trapezoid")),
        wse_center_frac=float(args.wse_center_frac),
        wse_quantile=float(args.wse_quantile),
        wse_fallback_drop_m=float(args.wse_fallback_drop_m),
        swot_wse=Path(args.swot_wse) if args.swot_wse else None,
        swot_wse_col=str(args.swot_wse_col),
        swot_x_col=args.swot_x_col,
        swot_y_col=args.swot_y_col,
        swot_csv_crs=str(args.swot_csv_crs),
        swot_max_dist_m=float(args.swot_max_dist_m),
        swot_stage_blend=float(args.swot_stage_blend),
        swot_vertical_offset_m=float(args.swot_vertical_offset_m),
        swot_outlier_mad_z=float(args.swot_outlier_mad_z),
        swot_use_for_slope=bool(getattr(args, "swot_use_for_slope", True)),
        curv_asymmetry_enabled=bool(getattr(args, "curv_asymmetry_enabled", False)),
        curv_window_m=float(getattr(args, "curv_window_m", 500.0)),
        curv_min_points=int(getattr(args, "curv_min_points", 7)),
        curv_kappa_scale_1pm=float(getattr(args, "curv_kappa_scale_1pm", 0.002)),
        curv_max_offset_frac=float(getattr(args, "curv_max_offset_frac", 0.25)),
        curv_lag_m=float(getattr(args, "curv_lag_m", 0.0)),
        a=float(args.a),
        b=float(args.b),
        dmin_m=float(args.dmin_m),
        dmax_m=float(args.dmax_m),
        prior_mode=str(args.prior_mode),
        mv_a0=float(args.mv_a0),
        mv_bw=float(args.mv_bw),
        mv_ba=float(args.mv_ba),
        mv_bs=float(args.mv_bs),
        mv_eps_a=float(args.mv_eps_a),
        mv_eps_s=float(args.mv_eps_s),
        slope_proxy_window=int(args.slope_proxy_window),
        slope_min=float(args.slope_min),
        slope_max=float(args.slope_max),
        slope_proxy_min_n=int(args.slope_proxy_min_n),
        wse_profile_enabled=bool(getattr(args, "wse_profile_enabled", True)),
        wse_profile_window=int(getattr(args, "wse_profile_window", args.slope_proxy_window)),
        wse_profile_min_n=int(getattr(args, "wse_profile_min_n", args.slope_proxy_min_n)),
        wse_profile_monotonic=bool(getattr(args, "wse_profile_monotonic", True)),
        energy_solver_enabled=bool(getattr(args, "energy_solver_enabled", False)),
        energy_allow_dem_proxy_wse=bool(getattr(args, "energy_allow_dem_proxy_wse", False)),
        out_1d_solver_inputs_json=getattr(args, "out_1d_solver_inputs_json", None),
        out_1d_solver_outputs_json=getattr(args, "out_1d_solver_outputs_json", None),
        out_1d_solver_accounting_json=getattr(args, "out_1d_solver_accounting_json", None),
        smooth_window=int(args.smooth_window),
        calib_max_dist_m=float(args.calib_max_dist_m),
        calib_stat=str(args.calib_stat),
        usgs_max_dist_m=float(args.usgs_max_dist_m),
        usgs_mean_to_dmax=float(usgs_mean_to_dmax_val),
        usgs_a_stat=str(args.usgs_a_stat),
        usgs_q_quantile_lo=float(args.usgs_q_quantile_lo),
        usgs_q_quantile_hi=float(args.usgs_q_quantile_hi),
        usgs_a_cv_warn=float(args.usgs_a_cv_warn),
        usgs_width_ratio_max=float(args.usgs_width_ratio_max),
        usgs_width_ratio_blend=bool(args.usgs_width_ratio_blend),
        gage_snap_max_dist_m=float(args.gage_snap_max_dist_m),
        width_stage_max_dist_m=float(args.width_stage_max_dist_m),
        width_stage_min_n=int(args.width_stage_min_n),
        width_stage_min_r2=float(args.width_stage_min_r2),
        width_stage_max_weight=float(args.width_stage_max_weight),
        manning_mode=str(args.manning_mode),
        manning_q_cms=args.manning_q_cms,
        manning_q_field=str(args.manning_q_field) if args.manning_q_field else None,
        manning_n=float(args.manning_n),
        manning_region=str(args.manning_region),
        manning_n_by_region=manning_n_by_region,
        manning_min_confidence=float(args.manning_min_confidence),
        manning_max_weight=float(args.manning_max_weight),
        manning_backwater_slope_thresh=float(args.manning_backwater_slope_thresh),
        manning_dist_to_mouth_field=str(args.manning_dist_to_mouth_field) if args.manning_dist_to_mouth_field else None,
        manning_dist_to_mouth_km_max=float(args.manning_dist_to_mouth_km_max),
        auto_manning_when_no_soundings=bool(getattr(args, "auto_manning_when_no_soundings", True)),
        regional_curve_enabled=bool(args.regional_curve_enabled),
        regional_curve_region=str(args.regional_curve_region),
        allow_builtin_regional_curves=bool(args.allow_builtin_regional_curves),
        regional_curve_c=(float(args.regional_curve_c) if args.regional_curve_c is not None else None),
        regional_curve_f=(float(args.regional_curve_f) if args.regional_curve_f is not None else None),
        regional_curve_da_units=str(args.regional_curve_da_units),
        regional_curve_depth_units=str(args.regional_curve_depth_units),
        regional_curve_depth_type=str(args.regional_curve_depth_type),
        regional_curve_to_dmax=str(args.regional_curve_to_dmax),
        regional_curve_to_dmax_factor=float(args.regional_curve_to_dmax_factor),
        geomorphic_envelope_enabled=bool(geomorphic_envelope_enabled),
        geomorphic_envelope_inflate_unc=bool(inflate_unc),
        geomorphic_envelope_only_when_no_soundings=bool(only_no_soundings),
        regional_curve_unc_pct=float(args.regional_curve_unc_pct),
        regional_curve_max_weight=float(args.regional_curve_max_weight),
        regional_curve_min_da_km2=float(args.regional_curve_min_da_km2),
        only_with_banks=not bool(args.allow_missing_banks),
        write_soundings_subset=(str(args.write_soundings_subset) if args.write_soundings_subset else None),
        only_write_soundings_subset=bool(getattr(args, "only_write_soundings_subset", False)),
        soundings_subset=(str(args.soundings_subset) if getattr(args, "soundings_subset", None) else None),
    )

    def _split_multi(vals):
        out = []
        if not vals:
            return out
        for v in vals:
            if v is None:
                continue
            for part in str(v).split(','):
                part = part.strip()
                if part:
                    out.append(part)
        return out

    usgs_sites = None
    if args.usgs_sites:
        usgs_sites = [s.strip() for s in str(args.usgs_sites).split(',') if s.strip()]

    width_stage_csv = _split_multi(getattr(args, 'width_stage_csv', []))

    infer_bathy(
        xs_gpkg=Path(args.xs_gpkg),
        out_gpkg=Path(args.out_gpkg),
        dem_path=Path(args.dem) if args.dem else None,
        soundings_path=args.soundings,
        soundings_depth_col=args.soundings_depth_col,
        soundings_elev_col=args.soundings_elev_col,
        soundings_x_col=args.soundings_x_col,
        soundings_y_col=args.soundings_y_col,
        soundings_crs=args.soundings_crs,
        cfg=cfg,
        xs_lines_layer=args.xs_lines_layer,
        xs_points_layer=args.xs_points_layer,
        river_gpkg=Path(args.river_gpkg) if args.river_gpkg else None,
        rivers_layer=str(args.rivers_layer),
        drain_area_field=args.drain_area_field,
        slope_field=args.slope_field,
        manning_q_field=args.manning_q_field,
        manning_dist_to_mouth_field=args.manning_dist_to_mouth_field,
        usgs_sites=usgs_sites,
        usgs_start=args.usgs_start,
        usgs_end=args.usgs_end,
        usgs_cache_dir=Path(args.usgs_cache_dir) if args.usgs_cache_dir else None,
        width_stage_csv=width_stage_csv if width_stage_csv else None,
        template_raster=Path(args.template_raster) if args.template_raster else (Path(args.dem) if args.dem else None),
        out_bathy_raster=Path(args.out_bathy_raster) if args.out_bathy_raster else None,
        out_mask_raster=Path(args.out_mask_raster) if args.out_mask_raster else None,
        out_uncert_raster=Path(args.out_uncert_raster) if args.out_uncert_raster else None,
        raster_value_col=str(args.raster_value_col),
        raster_uncert_col=str(args.raster_uncert_col),
        continuous=str(args.continuous),
        continuous_buffer_m=args.continuous_buffer_m,
        continuous_k=int(args.continuous_k),
        idw_power=float(args.idw_power),
        aniso_along_scale_m=float(args.aniso_along_scale_m),
        aniso_cross_scale_m=float(args.aniso_cross_scale_m),
        thalweg_weight=float(args.thalweg_weight),
        nodata=float(args.nodata),
        overlap_reducer=str(args.overlap_reducer),
        channel_mask_raster=Path(args.channel_mask_raster) if args.channel_mask_raster else None,
        channel_mask_inside_value=int(args.channel_mask_inside_value),
        channel_mask_invert=bool(args.channel_mask_invert),
        max_query_dist_m=(float(args.max_query_dist_m) if args.max_query_dist_m is not None else None),
        out_accounting_json=Path(args.out_accounting_json) if getattr(args, "out_accounting_json", None) else None,
        out_meta_json=Path(args.out_meta_json) if getattr(args, "out_meta_json", None) else None,
    )


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
