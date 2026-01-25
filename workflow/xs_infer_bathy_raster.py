#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xs_infer_bathy_raster.py

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

"""

from __future__ import annotations

import argparse
import logging


# Logging is configured by entrypoints (e.g., bathy_main.py / sdb_main.py).
# Standalone scripts configure logging in __main__.

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# Longitudinal WSE profile fitting (stabilizes slope for Manning / multivariate priors)
from river_wse import WSEFitConfig, fit_wse_profile
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.features import rasterize
from shapely.geometry import Point, mapping
from shapely.ops import unary_union
from pyproj import CRS, Transformer

log = logging.getLogger("xs_infer_bathy")
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

    # WSE estimation (proxy from DEM/topo profile)
    wse_center_frac: float = 0.10
    wse_quantile: float = 0.10
    wse_fallback_drop_m: float = 0.50

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

    # smoothing
    smooth_window: int = 7

    # Calibration from local soundings
    calib_max_dist_m: float = 200.0
    calib_stat: str = "p90"

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
    manning_min_confidence: float = 0.30
    manning_max_weight: float = 0.60
    manning_backwater_slope_thresh: float = 1e-4  # disable when slope below this (backwater/tidal risk)
    manning_dist_to_mouth_field: Optional[str] = None
    manning_dist_to_mouth_km_max: float = 10.0



    # Optional Regional Hydraulic Geometry Curves (Drainage Area -> Bankfull Depth)
    # This is *not* a bed observation; use as a soft prior. Coefficients are region-specific.
    # Depth form: D_bkf = c * DA^f
    # DA units controlled by regional_curve_da_units ('km2' or 'mi2').
    # If depth represents mean depth, convert to Dmax using trapezoid factor unless overridden.
    regional_curve_enabled: bool = False
    regional_curve_region: str = "default"      # selects built-in placeholder coefficients
    regional_curve_c: Optional[float] = None    # override c
    regional_curve_f: Optional[float] = None    # override f
    regional_curve_unc_pct: float = 40.0        # 1-sigma-ish percent uncertainty (used for weighting)
    regional_curve_da_field: Optional[str] = None  # if set, use this drainage area field name
    regional_curve_da_units: str = "km2"        # km2 | mi2
    regional_curve_depth_units: str = "m"      # m | ft (coefficients units)
    regional_curve_depth_type: str = "mean"     # mean | max
    regional_curve_to_dmax: str = "auto"        # auto | factor
    regional_curve_to_dmax_factor: float = 1.25 # used when to_dmax='factor'
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


def _trapezoid_depth_profile(dist_from_left: np.ndarray, W: float, Dmax: float, bottom_frac: float) -> np.ndarray:
    out = np.full_like(dist_from_left, np.nan, dtype="float64")
    if not np.isfinite(W) or W <= 0 or not np.isfinite(Dmax) or Dmax <= 0:
        return out

    Wb = float(np.clip(bottom_frac, 0.0, 0.95) * W)
    side = (W - Wb) / 2.0
    side = max(side, 1e-6)

    d = dist_from_left
    inside = (d >= 0.0) & (d <= W)

    left = inside & (d < side)
    out[left] = (d[left] / side) * Dmax

    mid = inside & (d >= side) & (d <= (side + Wb))
    out[mid] = Dmax

    right = inside & (d > (side + Wb))
    out[right] = ((W - d[right]) / side) * Dmax

    out[inside] = np.clip(out[inside], 0.0, Dmax)
    return out


def _compute_dmax_prior(W: float, cfg: InferConfig) -> float:
    if not np.isfinite(W) or W <= 0:
        return float("nan")
    D = cfg.a * (W ** cfg.b)
    return float(np.clip(D, cfg.dmin_m, cfg.dmax_m))
def _guess_field(columns, candidates):
    """Return the first candidate present in columns (case-insensitive), else None."""
    if not columns:
        return None
    lower = {str(c).lower(): str(c) for c in columns}
    for c in candidates:
        if str(c).lower() in lower:
            return lower[str(c).lower()]
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



def _compute_dmax_prior_multivariate(row: pd.Series, cfg: InferConfig) -> float:
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

    D = (
        float(cfg.mv_a0)
        * (W ** float(cfg.mv_bw))
        * ((A + float(cfg.mv_eps_a)) ** float(cfg.mv_ba))
        * ((S + float(cfg.mv_eps_s)) ** float(cfg.mv_bs))
    )
    return float(np.clip(D, cfg.dmin_m, cfg.dmax_m))


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

    n = float(cfg.manning_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_mean = ((n * Q) / (W * np.sqrt(S))) ** (3.0 / 5.0)

    if not np.isfinite(y_mean) or y_mean <= 0:
        return float("nan")

    mean_to_dmax = float(2.0 / (1.0 + float(cfg.bottom_width_frac)))
    dmax = float(y_mean) * mean_to_dmax
    return float(np.clip(dmax, cfg.dmin_m, cfg.dmax_m))


def _compute_manning_weight(row: pd.Series, cfg: InferConfig) -> float:
    """Compute a blend weight for the Manning prior, with simple backwater/tidal guards."""
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

    return float(np.clip(float(cfg.manning_max_weight), 0.0, 1.0))




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
    if not bool(getattr(cfg, "regional_curve_enabled", False)):
        return (float("nan"), 0.0, "")

    # Drainage area
    # Try configured field first, then common NHDPlus / StreamStats-style names.
    da = float("nan")
    cand_fields = []
    if getattr(cfg, "regional_curve_da_field", None):
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
                    if getattr(cfg, "regional_curve_da_units", None) is None:
                        cfg.regional_curve_da_units = "mi2"  # type: ignore
                break

    if not (np.isfinite(da) and da > 0):
        return (float("nan"), 0.0, "")

    # Ignore tiny basins (curves unstable; also often headwater morphology)
    if float(da) < float(getattr(cfg, "regional_curve_min_da_km2", 1.0)):
        return (float("nan"), 0.0, "")

    # Coefficients
    reg = str(getattr(cfg, "regional_curve_region", "default") or "default").lower().replace(" ", "_").replace("-", "_")
    c = getattr(cfg, "regional_curve_c", None)
    f = getattr(cfg, "regional_curve_f", None)

    if c is None or f is None:
        c0, f0, da_u0, depth_u0, depth_t0, unc0 = REGIONAL_CURVE_DEFAULTS.get(reg, REGIONAL_CURVE_DEFAULTS["default"])
        if c is None:
            c = c0
        if f is None:
            f = f0
        da_units = getattr(cfg, "regional_curve_da_units", da_u0)
        depth_units = getattr(cfg, "regional_curve_depth_units", depth_u0)
        depth_type = getattr(cfg, "regional_curve_depth_type", depth_t0)
        unc_pct = float(getattr(cfg, "regional_curve_unc_pct", unc0))
    else:
        da_units = getattr(cfg, "regional_curve_da_units", "km2")
        depth_units = getattr(cfg, "regional_curve_depth_units", "m")
        depth_type = getattr(cfg, "regional_curve_depth_type", "mean")
        unc_pct = float(getattr(cfg, "regional_curve_unc_pct", 40.0))

    da_use = _convert_da_units(float(da), str(da_units)) if str(da_units).lower().strip() != "km2" else float(da)
    # (If da_units is km2, da_use is km2; if mi2, converted.)
    # Bankfull depth
    with np.errstate(over="ignore", invalid="ignore"):
        d_bkf = float(c) * (float(da_use) ** float(f))

    d_bkf_m = _convert_depth_units(d_bkf, depth_units, "m")
    if not (np.isfinite(d_bkf_m) and d_bkf_m > 0):
        return (float("nan"), 0.0, "")

    # Convert to Dmax if needed
    to_dmax_mode = str(getattr(cfg, "regional_curve_to_dmax", "auto") or "auto").lower().strip()
    if str(depth_type).lower().strip() == "max":
        dmax = d_bkf_m
        conv = "bkf=max"
    else:
        if to_dmax_mode == "factor":
            dmax = d_bkf_m * float(getattr(cfg, "regional_curve_to_dmax_factor", 1.25))
            conv = f"bkf=mean*{float(getattr(cfg, 'regional_curve_to_dmax_factor', 1.25)):.2f}"
        else:
            # trapezoid mean-to-dmax conversion
            mean_to_dmax = float(2.0 / (1.0 + float(cfg.bottom_width_frac)))
            dmax = d_bkf_m * mean_to_dmax
            conv = "bkf=mean->dmax(trap)"

    dmax = float(np.clip(dmax, cfg.dmin_m, cfg.dmax_m))

    # Weight: inverse of uncertainty, capped
    # Basic: weight = max_weight * (1 - unc_pct/100) clipped
    max_w = float(getattr(cfg, "regional_curve_max_weight", 0.6))
    w = max(0.0, min(1.0, 1.0 - float(unc_pct) / 100.0))
    w = float(np.clip(max_w * w, 0.0, 1.0))

    detail = f"{reg}:{conv}:unc{unc_pct:.0f}%"
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

def _load_soundings(
    path: Path,
    target_crs,
    depth_col: Optional[str],
    elev_col: Optional[str],
    x_col: Optional[str],
    y_col: Optional[str],
) -> Optional[gpd.GeoDataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if x_col is None or y_col is None:
            for xc, yc in [("lon", "lat"), ("longitude", "latitude"), ("x", "y"), ("easting", "northing")]:
                if xc in df.columns and yc in df.columns:
                    x_col, y_col = xc, yc
                    break
        if x_col is None or y_col is None:
            raise RuntimeError("Soundings CSV requires --soundings-x-col/--soundings-y-col (or lon/lat columns).")

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs="EPSG:4326")
    elif path.suffix.lower() in (".xyz", ".txt", ".dat"):
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

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs="EPSG:4326")
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
                for ext in ("*.xyz", "*.csv", "*.txt", "*.dat", "*.gpkg", "*.shp", "*.geojson", "*.json"):
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


def _rasterize_points_median(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    template_ds: rasterio.DatasetReader,
    nodata: float,
) -> np.ndarray:
    """
    Rasterize point values onto the template grid by computing median per pixel.
    """
    arr = np.full((template_ds.height, template_ds.width), nodata, dtype="float32")
    if gdf.empty:
        return arr

    if gdf.crs is None:
        raise RuntimeError("Points GeoDataFrame missing CRS.")
    if CRS.from_user_input(gdf.crs) != CRS.from_user_input(template_ds.crs):
        gdf2 = gdf.to_crs(template_ds.crs)
    else:
        gdf2 = gdf

    vals = pd.to_numeric(gdf2[value_col], errors="coerce").to_numpy(dtype="float64")
    good = np.isfinite(vals)
    gdf2 = gdf2.loc[good].copy()
    vals = vals[good]
    if gdf2.empty:
        return arr

    xs = gdf2.geometry.x.to_numpy(dtype="float64")
    ys = gdf2.geometry.y.to_numpy(dtype="float64")
    rows, cols = rowcol(template_ds.transform, xs, ys)

    df = pd.DataFrame({"row": rows, "col": cols, "val": vals})
    df = df[(df["row"] >= 0) & (df["row"] < template_ds.height) & (df["col"] >= 0) & (df["col"] < template_ds.width)]
    if df.empty:
        return arr

    agg = df.groupby(["row", "col"])["val"].median().reset_index()
    arr[agg["row"].to_numpy(), agg["col"].to_numpy()] = agg["val"].to_numpy(dtype="float32")
    return arr


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


def _ensure_projected_for_distance(crs_in: CRS, lonlat_sample: Tuple[float, float]) -> CRS:
    if CRS.from_user_input(crs_in).is_projected:
        return CRS.from_user_input(crs_in)
    # Geographic CRS: estimate a metric CRS for buffering/distances.
    # NOTE: pyproj CRS objects do not universally expose get_utm_crs().
    # We implement a simple UTM chooser (WGS84 UTM zone by lon/lat). This is
    # good enough for short-distance buffering (corridor masks, etc.).
    lon, lat = lonlat_sample
    return _utm_crs_from_lonlat(float(lon), float(lat))


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
        centroid = g.geometry.unary_union.centroid
        utm_crs = _utm_crs_for_point(float(centroid.x), float(centroid.y), g.crs)
    except Exception:
        # Fall back to buffering in raster CRS units if UTM inference fails.
        utm_crs = None

    if utm_crs is not None:
        g_utm = g.to_crs(utm_crs)
        buffered = g_utm.geometry.buffer(float(buffer_m))
        buffered = gpd.GeoSeries(buffered, crs=utm_crs)
        poly = buffered.unary_union
        poly = gpd.GeoSeries([poly], crs=utm_crs).to_crs(template_ds.crs).iloc[0]
    else:
        # WARNING: units may not be meters (e.g., degrees). This is a last resort.
        poly = g.geometry.buffer(float(buffer_m)).unary_union

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
        from scipy.spatial import cKDTree  # type: ignore
        return cKDTree, "scipy"
    except Exception:
        try:
            from sklearn.neighbors import KDTree  # type: ignore
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
) -> np.ndarray:
    """
    IDW / adaptive IDW for query coordinates.
    pts_xy: (n,2), pts_val: (n,), q_xy: (m,2)
    Returns: (m,) float64
    """
    Tree, which = _kd_tree()
    if Tree is None:
        # Fallback: brute-force kNN in chunks (slow, but avoids hard dependency)
        which = "numpy"

    pts_xy = np.asarray(pts_xy, dtype="float64")
    pts_val = np.asarray(pts_val, dtype="float64")
    q_xy = np.asarray(q_xy, dtype="float64")

    if which == "scipy":
        tree = Tree(pts_xy)
        d, idx = tree.query(q_xy, k=min(k, len(pts_xy)))
    elif which == "sklearn":
        tree = Tree(pts_xy)
        d, idx = tree.query(q_xy, k=min(k, len(pts_xy)), return_distance=True)
    else:
        # numpy brute-force kNN (chunked)
        k_eff = int(min(k, len(pts_xy)))
        d_list = []
        idx_list = []
        chunk = 20000
        for i0 in range(0, len(q_xy), chunk):
            q = q_xy[i0 : i0 + chunk]
            dx = q[:, None, 0] - pts_xy[None, :, 0]
            dy = q[:, None, 1] - pts_xy[None, :, 1]
            dist2 = dx * dx + dy * dy

            idx_k = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
            row = np.arange(idx_k.shape[0])[:, None]
            dist2_k = dist2[row, idx_k]
            ord_k = np.argsort(dist2_k, axis=1)
            idx_sorted = idx_k[row, ord_k]
            d_sorted = np.sqrt(dist2[row, idx_sorted])

            d_list.append(d_sorted)
            idx_list.append(idx_sorted)

        d = np.vstack(d_list)
        idx = np.vstack(idx_list)

    d = np.asarray(d, dtype="float64")
    idx = np.asarray(idx, dtype="int64")

    # Ensure 2D
    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]

    # Adaptive power: increase in sparse areas, reduce in dense
    if adaptive:
        # crude AIDW: p in [1.5, 4.0] based on nearest-neighbor spread
        d1 = d[:, 0]
        dK = d[:, -1]
        ratio = np.clip(dK / np.maximum(d1, eps), 1.0, 10.0)
        p = 1.5 + (np.log(ratio) / np.log(10.0)) * (4.0 - 1.5)
        p = p[:, None]
        w = 1.0 / (np.power(d + eps, p))
    else:
        w = 1.0 / (np.power(d + eps, power))

    v = pts_val[idx]
    num = np.sum(w * v, axis=1)
    den = np.sum(w, axis=1)
    out = num / np.maximum(den, eps)

    # exact hits
    hit = d[:, 0] <= eps
    if np.any(hit):
        out[hit] = pts_val[idx[hit, 0]]
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
) -> np.ndarray:
    """Barrier-aware-ish anisotropic IDW using a centerline as the along-channel axis.

    We compute an effective distance:

        d_eff = sqrt( (d_along/along_scale)^2 + (d_cross/cross_scale)^2 )

    where d_along is the absolute difference of projected chainage on `centerline`,
    and d_cross is derived from Euclidean distance by:

        d_cross = sqrt(max(0, d_euclid^2 - d_along^2))

    This strongly discourages cross-channel influence while still allowing smooth
    along-channel interpolation.

    Notes:
    - This is not a full geodesic-in-mask solver (too expensive for large rasters),
      but it removes the most common cross-channel artifacts.
    - Requires a projected CRS in meters for meaningful scales.
    """
    if pts_xy.size == 0 or q_xy.size == 0:
        return np.full((q_xy.shape[0],), np.nan, dtype="float64")

    # Guard scales
    along_scale_m = float(max(1e-3, along_scale_m))
    cross_scale_m = float(max(1e-3, cross_scale_m))

    # Precompute chainage for points
    try:
        s_pts = np.array([centerline.project(Point(float(x), float(y))) for x, y in pts_xy], dtype="float64")
    except Exception:
        # If projection fails for any reason, fall back to isotropic IDW
        return _idw_interpolate_on_mask(pts_xy, pts_val, q_xy, k=int(k), power=float(power), adaptive=False, eps=eps)

    Tree, which = _kd_tree()
    if Tree is None:
        # SciPy missing; fall back
        return _idw_interpolate_on_mask(pts_xy, pts_val, q_xy, k=int(k), power=float(power), adaptive=False, eps=eps)

    tree = Tree(pts_xy)
    n = pts_xy.shape[0]
    k = int(min(max(1, int(k)), n))
    # Candidate pool to re-rank by anisotropic distance
    cand = int(min(n, max(k, k * 5)))

    # Query Euclidean candidates
    d_eu, idx = tree.query(q_xy, k=cand)
    if cand == 1:
        d_eu = d_eu.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    out = np.full((q_xy.shape[0],), np.nan, dtype="float64")

    for i in range(q_xy.shape[0]):
        ids = idx[i]
        de = d_eu[i].astype("float64")
        # Chainage for query
        try:
            s_q = centerline.project(Point(float(q_xy[i, 0]), float(q_xy[i, 1])))
        except Exception:
            # Fallback for this point
            ids = ids[:k]
            de = de[:k]
            w = 1.0 / (np.maximum(de, eps) ** float(power))
            vv = pts_val[ids]
            out[i] = float(np.sum(w * vv) / np.sum(w)) if np.isfinite(np.sum(w)) and np.sum(w) > 0 else float(np.nan)
            continue

        s_p = s_pts[ids]
        d_along = np.abs(s_q - s_p)

        # derive cross distance from euclid + along (Pythagoras in the along/cross frame)
        d_cross2 = np.maximum(0.0, de * de - d_along * d_along)
        d_cross = np.sqrt(d_cross2)

        d_eff = np.sqrt((d_along / along_scale_m) ** 2 + (d_cross / cross_scale_m) ** 2) + eps

        # choose top-k smallest effective distances
        order = np.argsort(d_eff)[:k]
        ids2 = ids[order]
        d2 = d_eff[order]

        w = 1.0 / (d2 ** float(power))
        vv = pts_val[ids2]
        sw = np.sum(w)
        out[i] = float(np.sum(w * vv) / sw) if np.isfinite(sw) and sw > 0 else float(np.nan)

    return out


def _rasterize_points_min(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    template_ds: rasterio.io.DatasetReader,
    nodata: float = -9999.0,
) -> np.ndarray:
    """Rasterize points by taking the *minimum* value per pixel.

    This is the safest reducer for river bathymetry where multiple profiles
    (often from overlapping tributaries) can land in the same output pixel.
    Taking the minimum keeps the deeper bed (more-negative depth / lower bed
    elevation) and avoids artificial shoals created by averaging.
    """
    out = np.full((template_ds.height, template_ds.width), float(nodata), dtype=np.float32)
    if gdf is None or len(gdf) == 0:
        return out

    xs = np.asarray(gdf.geometry.x, dtype=float)
    ys = np.asarray(gdf.geometry.y, dtype=float)
    vals = pd.to_numeric(gdf[value_col], errors='coerce').to_numpy(dtype=float)
    m = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(vals)
    if not np.any(m):
        return out
    rows, cols = rasterio.transform.rowcol(template_ds.transform, xs[m], ys[m])
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    inb = (rows >= 0) & (rows < out.shape[0]) & (cols >= 0) & (cols < out.shape[1])
    if not np.any(inb):
        return out
    rows = rows[inb]
    cols = cols[inb]
    vals = vals[m][inb]

    # Vectorized pixel-wise minimum reduction
    h, w = out.shape
    idx = rows * w + cols
    # Use an intermediate flat array with +inf so np.minimum.at works
    flat = np.full(h * w, np.inf, dtype=np.float32)
    v = vals.astype(np.float32, copy=False)
    # Reduce: for duplicate indices, keep the minimum value
    np.minimum.at(flat, idx, v)
    flat = flat.reshape((h, w))
    # Where nothing was written, set nodata
    flat[~np.isfinite(flat)] = float(nodata)
    out = flat.astype(np.float32, copy=False)
    return out


def _rasterize_points_reduce(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    template_ds: rasterio.io.DatasetReader,
    nodata: float = -9999.0,
    reducer: str = "min",
) -> np.ndarray:
    reducer = (reducer or "min").lower().strip()
    if reducer in ("min", "minimum", "deeper"):
        return _rasterize_points_min(gdf, value_col, template_ds, nodata=nodata)
    if reducer in ("median", "med"):
        return _rasterize_points_median(gdf, value_col, template_ds, nodata=nodata)
    raise ValueError(f"Unknown reducer: {reducer}")


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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a continuous surface raster (float32) and a mask raster (uint8) on the template grid.
    """
    if pts_gdf.empty:
        arr = np.full((template_ds.height, template_ds.width), nodata, dtype="float32")
        m = np.zeros((template_ds.height, template_ds.width), dtype="uint8")
        return arr, m

    # Reproject points to template CRS for geometry alignment
    pts_t = pts_gdf.to_crs(template_ds.crs) if CRS.from_user_input(pts_gdf.crs) != CRS.from_user_input(template_ds.crs) else pts_gdf

    vals = pd.to_numeric(pts_t[value_col], errors="coerce").to_numpy(dtype="float64")
    good = np.isfinite(vals)
    pts_t = pts_t.loc[good].copy()
    vals = vals[good]
    if pts_t.empty:
        arr = np.full((template_ds.height, template_ds.width), nodata, dtype="float32")
        m = np.zeros((template_ds.height, template_ds.width), dtype="uint8")
        return arr, m

    # ------------------------------------------------------------------
    # Handle overlapping profiles / multi-stream overlaps.
    # ------------------------------------------------------------------
    # When multiple points fall into the same output pixel, it is safer to
    # keep the *deepest* (minimum elevation / most-negative depth) rather
    # than averaging (which can artificially shoal the bed at confluences).
    try:
        rr_pts, cc_pts = rasterio.transform.rowcol(
            template_ds.transform,
            pts_t.geometry.x.to_numpy(dtype="float64"),
            pts_t.geometry.y.to_numpy(dtype="float64"),
        )
        tmp = pd.DataFrame({"r": rr_pts, "c": cc_pts, "v": vals})
        if str(overlap_reducer).lower() == "median":
            agg = tmp.groupby(["r", "c"], sort=False)["v"].median().reset_index()
        else:
            agg = tmp.groupby(["r", "c"], sort=False)["v"].min().reset_index()
        if len(agg) < len(tmp):
            xs, ys = rasterio.transform.xy(template_ds.transform, agg["r"].to_numpy(), agg["c"].to_numpy(), offset="center")
            pts_t = gpd.GeoDataFrame(
                {value_col: agg["v"].to_numpy(dtype="float64")},
                geometry=gpd.points_from_xy(xs, ys),
                crs=template_ds.crs,
            )
            vals = pts_t[value_col].to_numpy(dtype="float64")
    except Exception as e:
        log.warning("[RIVER] overlap collapsing failed; continuing without: %s", e)

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
        arr = np.full((template_ds.height, template_ds.width), nodata, dtype="float32")
        return arr, mask

    xs_q, ys_q = rasterio.transform.xy(template_ds.transform, rr, cc, offset="center")
    q_xy_tpl = np.vstack([np.asarray(xs_q, dtype="float64"), np.asarray(ys_q, dtype="float64")]).T

    # Distance CRS selection
    crs_tpl = CRS.from_user_input(template_ds.crs)
    if crs_tpl.is_projected:
        pts_xy = np.vstack([pts_t.geometry.x.to_numpy(dtype="float64"), pts_t.geometry.y.to_numpy(dtype="float64")]).T
        q_xy = q_xy_tpl
    else:
        # geographic: project to UTM for distance computations
        # Avoid GeoPandas unary_union deprecation (and keep compatibility across versions)
        try:
            union_geom = pts_t.geometry.union_all()
        except Exception:
            union_geom = unary_union(list(pts_t.geometry))
        c = union_geom.centroid
        lon, lat = float(c.x), float(c.y)
        utm = _utm_crs_from_lonlat(lon, lat)
        tr = Transformer.from_crs(crs_tpl, utm, always_xy=True)
        px, py = tr.transform(pts_t.geometry.x.to_numpy(dtype="float64"), pts_t.geometry.y.to_numpy(dtype="float64"))
        qx, qy = tr.transform(q_xy_tpl[:, 0], q_xy_tpl[:, 1])
        pts_xy = np.vstack([px, py]).T
        q_xy = np.vstack([qx, qy]).T

    pts_val = vals

    # WALID-style thalweg guidance: deepest point per xs_id
    if method == "walid" and "xs_id" in pts_t.columns:
        # choose thalweg as minimum bed elevation per xs_id (deepest)
        g = pts_t.assign(_val=pts_val).groupby("xs_id", sort=False)["_val"]
        idxmin = g.idxmin()
        thal = pts_t.loc[idxmin].copy()
        thal_val = pd.to_numeric(thal[value_col], errors="coerce").to_numpy(dtype="float64")
        thal_good = np.isfinite(thal_val)
        thal = thal.loc[thal_good]
        thal_val = thal_val[thal_good]
        if not thal.empty and thalweg_weight > 1.0:
            # duplicate thalweg points to increase weight
            reps = int(max(1.0, float(thalweg_weight)))
            thal_rep = pd.concat([thal] * reps, ignore_index=True)
            thal_val_rep = np.tile(thal_val, reps)
            # append
            pts_xy_thal = np.vstack([thal_rep.geometry.x.to_numpy(dtype="float64"), thal_rep.geometry.y.to_numpy(dtype="float64")]).T
            if not crs_tpl.is_projected:
                # project thalweg too (reuse tr)
                tx, ty = tr.transform(pts_xy_thal[:, 0], pts_xy_thal[:, 1])
                pts_xy_thal = np.vstack([tx, ty]).T
            pts_xy = np.vstack([pts_xy, pts_xy_thal])
            pts_val = np.concatenate([pts_val, thal_val_rep])

    
    # Interpolate (continuous modes)
    method_l = str(method).lower()
    if method_l in ("aniso", "walid_aniso"):
        centerline = None
        if corridor_lines_gdf is not None and len(corridor_lines_gdf) > 0:
            try:
                from shapely.ops import linemerge
                lines_t = corridor_lines_gdf.to_crs(template_ds.crs) if CRS.from_user_input(corridor_lines_gdf.crs) != CRS.from_user_input(template_ds.crs) else corridor_lines_gdf
                merged = linemerge(unary_union(lines_t.geometry))
                if merged is not None:
                    if hasattr(merged, "geoms"):
                        # choose longest segment
                        centerline = max(list(merged.geoms), key=lambda g: g.length)
                    else:
                        centerline = merged
            except Exception as e:
                log.warning("[RIVER][ANISO] centerline extraction failed; falling back to isotropic IDW: %s", e)
                centerline = None

        if centerline is not None and crs_tpl.is_projected:
            vals_q = _aniso_idw_interpolate_on_mask(
                pts_xy=pts_xy,
                pts_val=pts_val,
                q_xy=q_xy,
                centerline=centerline,
                k=int(k),
                power=float(idw_power),
                along_scale_m=float(aniso_along_scale_m),
                cross_scale_m=float(aniso_cross_scale_m),
                eps=1e-6,
            )
        else:
            vals_q = _idw_interpolate_on_mask(
                pts_xy=pts_xy,
                pts_val=pts_val,
                q_xy=q_xy,
                k=int(k),
                power=float(idw_power),
                adaptive=False,
                eps=1e-6,
            )
    else:
        adaptive = (method_l == "aidw")
        vals_q = _idw_interpolate_on_mask(
            pts_xy=pts_xy,
            pts_val=pts_val,
            q_xy=q_xy,
            k=int(k),
            power=float(idw_power),
            adaptive=adaptive,
            eps=1e-6,
        )

    out = np.full((template_ds.height, template_ds.width), nodata, dtype="float32")
    out[rr, cc] = vals_q.astype("float32")
    return out, mask.astype("uint8")


def _write_geotiff(
    path: Path,
    arr: np.ndarray,
    template_ds: rasterio.DatasetReader,
    nodata: float,
    dtype: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = template_ds.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


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

    xs_records = []
    wse_by_xs: Dict[str, float] = {}

    for _, xsl in xs_lines.iterrows():
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

        dmax_prior = _compute_dmax_prior(W, cfg)

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


    # Optional reach attributes (for multivariate priors)
    xs_param["drain_area_km2"] = np.nan
    xs_param["slope_mpm"] = np.nan
    if river_gpkg is not None and Path(river_gpkg).exists():
        try:
            rivers = _read_layer(Path(river_gpkg), rivers_layer)
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
                xs_param = xs_param.merge(
                    rdf[["_river_id_str", "drain_area_km2", "slope_mpm", "manning_q_cms", "dist_to_mouth_km"]],
                    on="_river_id_str",
                    how="left",
                    suffixes=("", "_r"),
                )
                xs_param = xs_param.drop(columns=["_river_id_str"])
        except Exception as e:
            log.warning("[RIVER][ATTR] failed to attach river attributes from %s:%s (%s)", river_gpkg, rivers_layer, e)

    # If slope was not provided, estimate a stabilized water-surface slope from a fitted
    # longitudinal WSE profile (preferred) and fall back to the legacy slope proxy.
    xs_param["wse_fit_m"] = np.nan
    xs_param["slope_wse_mpm"] = np.nan
    if bool(getattr(cfg, "wse_profile_enabled", True)):
        try:
            wcfg = WSEFitConfig(
                enabled=True,
                window=int(getattr(cfg, "wse_profile_window", cfg.slope_proxy_window)),
                min_n=int(getattr(cfg, "wse_profile_min_n", cfg.slope_proxy_min_n)),
                enforce_monotonic=bool(getattr(cfg, "wse_profile_monotonic", True)),
                slope_min=float(cfg.slope_min),
                slope_max=float(cfg.slope_max),
            )
            wse_fit, slope_fit = fit_wse_profile(xs_param, cfg=wcfg)
            xs_param["wse_fit_m"] = wse_fit
            xs_param["slope_wse_mpm"] = slope_fit
        except Exception as e:
            log.warning("[RIVER][WSE] WSE profile fit failed, falling back to slope proxy (%s)", e)

    xs_param["slope_proxy_mpm"] = _compute_slope_proxy(
        xs_param,
        window=cfg.slope_proxy_window,
        slope_min=cfg.slope_min,
        slope_max=cfg.slope_max,
        min_n=cfg.slope_proxy_min_n,
    )
    # Prefer slope from WSE profile if available; otherwise fallback to slope_proxy
    if xs_param["slope_mpm"].isna().all():
        if xs_param["slope_wse_mpm"].notna().any():
            xs_param["slope_mpm"] = xs_param["slope_wse_mpm"]
        else:
            xs_param["slope_mpm"] = xs_param["slope_proxy_mpm"]

    # Upgrade prior if requested
    if str(cfg.prior_mode).lower().strip() == "multivariate":
        xs_param["dmax_prior_m"] = xs_param.apply(lambda r: _compute_dmax_prior_multivariate(r, cfg), axis=1)

    # ---- Optional soft priors (blended into dmax_prior_m) ----
    # 1) Regional hydraulic geometry curves (DA -> bankfull depth)
    xs_param["dmax_regional_curve_m"] = np.nan
    xs_param["regional_curve_wt"] = 0.0
    xs_param["regional_curve_detail"] = ""
    if bool(getattr(cfg, "regional_curve_enabled", False)):
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

    # 2) Manning inversion prior (requires Q, width, slope)
    xs_param["manning_q_cms_used"] = np.nan
    xs_param["manning_depth_mean_m"] = np.nan
    xs_param["manning_dmax_m"] = np.nan
    xs_param["manning_conf"] = 0.0
    xs_param["manning_wt"] = 0.0
    xs_param["manning_flags"] = ""

    if str(cfg.manning_mode).lower().strip() != "off":
        m_mode = str(cfg.manning_mode).lower().strip()
        def _m_apply(r):
            W = float(r.get("width_m", np.nan))
            S = float(r.get("slope_mpm", np.nan))
            if not (np.isfinite(W) and W > 0 and np.isfinite(S) and S > 0):
                return pd.Series({"manning_q_cms_used": np.nan, "manning_depth_mean_m": np.nan, "manning_dmax_m": np.nan, "manning_conf": 0.0, "manning_wt": 0.0, "manning_flags": ""})

            # Guard weight (simple slope/mouth filters)
            w_guard = _compute_manning_weight(r, cfg)
            if w_guard <= 0:
                return pd.Series({"manning_q_cms_used": np.nan, "manning_depth_mean_m": np.nan, "manning_dmax_m": np.nan, "manning_conf": 0.0, "manning_wt": 0.0, "manning_flags": "guard"})

            # Discharge
            Q = np.nan
            qsrc = ""
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
                    q2 = estimate_q2_from_drainage_area(da, region=str(getattr(cfg, "manning_region", "default")))
                    Q = float(getattr(q2, "q2_m3s", np.nan))
                    qsrc = f"q2_{str(getattr(cfg, 'manning_region', 'default'))}"

            if not (np.isfinite(Q) and Q > 0):
                return pd.Series({"manning_q_cms_used": Q, "manning_depth_mean_m": np.nan, "manning_dmax_m": np.nan, "manning_conf": 0.0, "manning_wt": 0.0, "manning_flags": f"noQ_{qsrc}"})

            # Depth estimate
            conf = 0.7  # default
            tidal = False
            backwater = False
            if invert_manning_for_depth is not None:
                res = invert_manning_for_depth(discharge_m3s=Q, width_m=W, slope=S, manning_n=float(cfg.manning_n), discharge_source=qsrc)
                y = float(getattr(res, "depth_m", np.nan))
                conf = float(getattr(res, "confidence", 0.0) or 0.0)
                tidal = bool(getattr(res, "tidal_flag", False))
                backwater = bool(getattr(res, "backwater_flag", False))
            else:
                # Fallback to simple wide-channel inversion (mean depth)
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = ((float(cfg.manning_n) * Q) / (W * np.sqrt(S))) ** (3.0 / 5.0)

            if not (np.isfinite(y) and y > 0):
                return pd.Series({"manning_q_cms_used": Q, "manning_depth_mean_m": y, "manning_dmax_m": np.nan, "manning_conf": 0.0, "manning_wt": 0.0, "manning_flags": f"badY_{qsrc}"})

            mean_to_dmax = float(2.0 / (1.0 + float(cfg.bottom_width_frac)))
            dmax = float(np.clip(y * mean_to_dmax, cfg.dmin_m, cfg.dmax_m))

            # Final weight: guard * max_weight * confidence
            min_conf = float(getattr(cfg, "manning_min_confidence", 0.30))
            if conf < min_conf:
                w = 0.0
            else:
                w = float(np.clip(w_guard * float(cfg.manning_max_weight) * float(conf), 0.0, 1.0))

            flags = qsrc
            if tidal:
                flags += "|tidal"
            if backwater:
                flags += "|backwater"
            return pd.Series({"manning_q_cms_used": Q, "manning_depth_mean_m": y, "manning_dmax_m": dmax, "manning_conf": conf, "manning_wt": w, "manning_flags": flags})

        mm = xs_param.apply(_m_apply, axis=1)
        xs_param[["manning_q_cms_used", "manning_depth_mean_m", "manning_dmax_m", "manning_conf", "manning_wt", "manning_flags"]] = mm

        # Blend
        w = xs_param["manning_wt"].astype("float64").clip(0.0, 1.0)
        d0 = xs_param["dmax_prior_m"].astype("float64")
        d1 = xs_param["manning_dmax_m"].astype("float64")
        use = np.isfinite(d0) & np.isfinite(d1) & (w > 0)
        xs_param.loc[use, "dmax_prior_m"] = (1.0 - w[use]) * d0[use] + w[use] * d1[use]

    # ------------------------
    # Calibration anchors
    # ------------------------
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
        )
        if soundings is not None and not soundings.empty:
            log.info("[CALIB] Loaded soundings: n=%d", len(soundings))
            calib_df = _calibrate_dmax_from_soundings(xs_lines[["xs_id", "geometry"]].copy(), soundings, wse_by_xs, cfg)
            log.info("[CALIB] Matched XS: %d", len(calib_df))
        else:
            log.info("[CALIB] Soundings empty; will use other anchors / priors.")
    else:
        log.info("[CALIB] No soundings provided; will use other anchors / priors.")

    xs_param = xs_param.merge(calib_df, on="xs_id", how="left")
    xs_param["soundings_n"] = xs_param["calib_n"].fillna(0).astype(int)
    xs_param["soundings_dmax_m"] = pd.to_numeric(xs_param["calib_depth_stat"], errors="coerce")
    xs_param = xs_param.drop(columns=["calib_n", "calib_depth_stat"])

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
                float(beta),
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

        depth = _trapezoid_depth_profile(
            np.array([dist_from_left], dtype="float64"),
            W=W,
            Dmax=Dmax,
            bottom_frac=cfg.bottom_width_frac,
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
                    corridor_lines_gdf=xs_lines,
                )

            if out_bathy_raster:
                _write_geotiff(Path(out_bathy_raster), bed_arr, tmpl, nodata=float(nodata), dtype="float32")
                log.info("[WRITE] %s", str(out_bathy_raster))

            if out_mask_raster:
                _write_geotiff(Path(out_mask_raster), mask_arr.astype("uint8"), tmpl, nodata=0.0, dtype="uint8")
                log.info("[WRITE] %s", str(out_mask_raster))

            if out_uncert_raster:
                if continuous.lower() == "median":
                    unc_arr = _rasterize_points_median(pred_gdf, raster_uncert_col, tmpl, nodata=float(nodata))
                else:
                    # uncertainty: still rasterize median per pixel; continuous uncertainty can come later
                    unc_arr = _rasterize_points_median(pred_gdf, raster_uncert_col, tmpl, nodata=float(nodata))
                _write_geotiff(Path(out_uncert_raster), unc_arr, tmpl, nodata=float(nodata), dtype="float32")
                log.info("[WRITE] %s", str(out_uncert_raster))

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
    p.add_argument("--manning-min-confidence", type=float, default=0.30, help="Minimum confidence required to apply Manning prior (0-1).")
    p.add_argument("--manning-max-weight", type=float, default=0.60, help="Maximum blend weight for Manning prior (0-1).")
    p.add_argument("--manning-backwater-slope-thresh", type=float, default=1e-4, help="Disable Manning prior when slope is below this (backwater/tidal risk).")
    p.add_argument("--manning-dist-to-mouth-field", default=None, help="Optional river attribute field containing distance-to-mouth (km).")
    p.add_argument("--manning-dist-to-mouth-km-max", type=float, default=10.0, help="If dist-to-mouth is provided, disable Manning prior when distance <= this (km).")


    # Regional hydraulic geometry curves (Drainage Area -> bankfull depth)
    p.add_argument("--regional-curve-enabled", action="store_true", help="Enable regional hydraulic geometry curve prior (DA -> bankfull depth) as a soft prior.")
    p.add_argument("--regional-curve-region", default="default", help="Region key for built-in coefficients (illustrative). Prefer providing --regional-curve-c and --regional-curve-f from published curves.")
    p.add_argument("--regional-curve-c", type=float, default=None, help="Coefficient c in D_bkf = c * DA^f (units depend on --regional-curve-da-units and --regional-curve-depth-units).")
    p.add_argument("--regional-curve-f", type=float, default=None, help="Exponent f in D_bkf = c * DA^f.")
    p.add_argument("--regional-curve-da-units", choices=["km2","mi2"], default="km2", help="Drainage area units expected by the curve coefficients.")
    p.add_argument("--regional-curve-depth-units", choices=["m","ft"], default="m", help="Depth units of the curve coefficients.")
    p.add_argument("--regional-curve-depth-type", choices=["mean","max"], default="mean", help="Whether curve depth represents mean or max depth at bankfull.")
    p.add_argument("--regional-curve-to-dmax", choices=["auto","factor"], default="auto", help="Conversion from bankfull depth to Dmax. 'auto' uses trapezoid mean->Dmax conversion; 'factor' uses --regional-curve-to-dmax-factor.")
    p.add_argument("--regional-curve-to-dmax-factor", type=float, default=1.25, help="Used when --regional-curve-to-dmax=factor.")
    p.add_argument("--regional-curve-unc-pct", type=float, default=40.0, help="Uncertainty percent (used to downweight).")
    p.add_argument("--regional-curve-max-weight", type=float, default=0.60, help="Maximum blend weight for regional-curve prior (0-1).")
    p.add_argument("--regional-curve-min-da-km2", type=float, default=1.0, help="Ignore DA smaller than this (km^2) for the curve prior.")

    p.add_argument("--width-stage-max-weight", type=float, default=0.8, help="Maximum blend weight for width–stage anchor.")

    # Optional river attributes (for multivariate priors)
    p.add_argument("--river-gpkg", default=None, help="Optional river network GPKG to attach reach attributes (e.g., drainage area, slope).")
    p.add_argument("--rivers-layer", default="rivers_clip", help="Layer within --river-gpkg containing per-reach attributes.")
    p.add_argument("--drain-area-field", default=None, help="Field name in rivers layer for drainage area (km^2). If omitted, tries common names.")
    p.add_argument("--slope-field", default=None, help="Field name in rivers layer for slope (m/m). If omitted, tries common names.")

    # Bed model + priors
    p.add_argument("--bottom-width-frac", type=float, default=0.30, help="Flat bottom width fraction of channel width")
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
    p.add_argument("--slope-proxy-min-n", type=int, default=7, help="Minimum XS per river_id required to compute slope proxy.")

    # Longitudinal WSE profile fitting (preferred slope when reach slope attribute is missing)
    p.add_argument("--wse-profile-enabled", dest="wse_profile_enabled", action="store_true", help="Enable longitudinal WSE profile fitting (default).")
    p.add_argument("--no-wse-profile", dest="wse_profile_enabled", action="store_false", help="Disable WSE profile fitting; use legacy slope proxy only.")
    p.set_defaults(wse_profile_enabled=True)
    p.add_argument("--wse-profile-window", type=int, default=9, help="Rolling window (XS count) for WSE profile smoothing.")
    p.add_argument("--wse-profile-min-n", type=int, default=7, help="Minimum XS per river_id required to fit a WSE profile.")
    p.add_argument("--wse-profile-monotonic", dest="wse_profile_monotonic", action="store_true", help="Enforce monotonic WSE along stationing (default).")
    p.add_argument("--no-wse-profile-monotonic", dest="wse_profile_monotonic", action="store_false", help="Disable monotonic constraint.")
    p.set_defaults(wse_profile_monotonic=True)

    # WSE proxy

    p.add_argument("--wse-center-frac", type=float, default=0.10, help="Center window fraction of XS length")
    p.add_argument("--wse-quantile", type=float, default=0.10, help="Quantile of center window elevations for WSE proxy")
    p.add_argument("--wse-fallback-drop-m", type=float, default=0.50, help="Fallback WSE= min(bank_z)-drop (m)")

    # Smoothing
    p.add_argument("--smooth-window", type=int, default=7, help="Rolling window (XS count) for Dmax smoothing")
    p.add_argument("--allow-missing-banks", action="store_true", help="Process XS even if banks aren't detected")

    # Raster outputs
    p.add_argument("--template-raster", default=None, help="Template raster to align GeoTIFF outputs (usually your CUDEM DEM). If omitted, uses --dem when available.")
    p.add_argument("--channel-mask-raster", default=None, help="Optional DEM-aligned channel mask raster to constrain interpolation (1=inside by default).")
    p.add_argument("--channel-mask-inside-value", type=int, default=1, help="Raster value treated as inside-channel (default 1). For masks where water=0, set this to 0 or use --channel-mask-invert.")
    p.add_argument("--channel-mask-invert", action="store_true", help="Invert channel mask logic (inside becomes outside). Useful if mask uses 1=land and 0=water.")
    p.add_argument("--out-bathy-raster", default=None, help="Optional output GeoTIFF of predicted bed elevation (z_bed_pred_m) on template grid")
    p.add_argument("--out-mask-raster", default=None, help="Optional output GeoTIFF mask (1 where bathy raster has data)")
    p.add_argument("--out-uncert-raster", default=None, help="Optional output GeoTIFF uncertainty (meters) on template grid")

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
                )

            if args.out_bathy_raster:
                _write_geotiff(Path(args.out_bathy_raster), bed, tmpl, nodata=float(args.nodata), dtype="float32")
                log.info("[WRITE] %s", str(args.out_bathy_raster))

            if args.out_mask_raster:
                _write_geotiff(Path(args.out_mask_raster), mask.astype("uint8"), tmpl, nodata=0.0, dtype="uint8")
                log.info("[WRITE] %s", str(args.out_mask_raster))

            if args.out_uncert_raster:
                if uncert_col in pred_gdf.columns:
                    unc = _rasterize_points_median(pred_gdf, uncert_col, tmpl, nodata=float(args.nodata))
                else:
                    unc = np.full((tmpl.height, tmpl.width), float(args.nodata), dtype="float32")
                _write_geotiff(Path(args.out_uncert_raster), unc, tmpl, nodata=float(args.nodata), dtype="float32")
                log.info("[WRITE] %s", str(args.out_uncert_raster))

        log.info("[DONE] Raster-only complete.")
        return

    # Inference mode requires xs-gpkg and out-gpkg
    if not args.xs_gpkg or not args.out_gpkg:
        raise RuntimeError("Inference mode requires --xs-gpkg and --out-gpkg (or use --raster-from-gpkg for raster-only).")

    # Parse mean→max depth conversion. Use 'auto' to derive Dmax/mean for a trapezoid: 2/(1+bottom_width_frac).
    if str(args.usgs_mean_to_dmax).strip().lower() == "auto":
        usgs_mean_to_dmax_val = -1.0
    else:
        usgs_mean_to_dmax_val = float(args.usgs_mean_to_dmax)

    cfg = InferConfig(
        bottom_width_frac=float(args.bottom_width_frac),
        wse_center_frac=float(args.wse_center_frac),
        wse_quantile=float(args.wse_quantile),
        wse_fallback_drop_m=float(args.wse_fallback_drop_m),
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
        manning_min_confidence=float(args.manning_min_confidence),
        manning_max_weight=float(args.manning_max_weight),
        manning_backwater_slope_thresh=float(args.manning_backwater_slope_thresh),
        manning_dist_to_mouth_field=str(args.manning_dist_to_mouth_field) if args.manning_dist_to_mouth_field else None,
        manning_dist_to_mouth_km_max=float(args.manning_dist_to_mouth_km_max),
        regional_curve_enabled=bool(args.regional_curve_enabled),
        regional_curve_region=str(args.regional_curve_region),
        regional_curve_c=(float(args.regional_curve_c) if args.regional_curve_c is not None else None),
        regional_curve_f=(float(args.regional_curve_f) if args.regional_curve_f is not None else None),
        regional_curve_da_units=str(args.regional_curve_da_units),
        regional_curve_depth_units=str(args.regional_curve_depth_units),
        regional_curve_depth_type=str(args.regional_curve_depth_type),
        regional_curve_to_dmax=str(args.regional_curve_to_dmax),
        regional_curve_to_dmax_factor=float(args.regional_curve_to_dmax_factor),
        regional_curve_unc_pct=float(args.regional_curve_unc_pct),
        regional_curve_max_weight=float(args.regional_curve_max_weight),
        regional_curve_min_da_km2=float(args.regional_curve_min_da_km2),
        only_with_banks=not bool(args.allow_missing_banks),
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
    )


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
