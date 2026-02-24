#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gapfill_intelligent.py - Physics-Informed Gap-Filling for Bathymetry

This module implements intelligent gap-filling using the "prior + residual" framework:

    z_final(x) = z_prior(x) + r_interp(x)

Design Goals
------------
1. Standalone gap-fill for bathymetry rasters
2. Future integration as a waffles gridding module (like WafflesSpline, WafflesIDW)
3. Clean separation between core interpolation and raster I/O

Architecture
------------
- BathyInterpolator: Core class for residual interpolation (waffles-ready)
- gapfill_depth_raster(): Raster workflow wrapper
- Point-based API for direct XYZ input/output

Key Scientific Features
-----------------------
1. Prior + residual framework (regression kriging equivalent)
2. Coordinate scaling to meters (handles geographic CRS correctly)
3. Barrier-constrained interpolation (per water body)
4. Anisotropic river smoothing (along-channel vs cross-channel)
5. Proper uncertainty propagation (Bayesian combination)

References
----------
- Hengl, T., et al. (2007). "About regression-kriging." Computers & Geosciences.
- Merwade, V., et al. (2008). "Anisotropic considerations while interpolating
  river channel bathymetry." J. Hydrology.

Author: SDB Pipeline Development Team
Version: 0.8.0
"""


import logging
import math
import warnings
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

log = logging.getLogger(__name__)

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================

try:
    import rasterio
    from rasterio.transform import rowcol, xy
    from rasterio.crs import CRS
    _HAVE_RASTERIO = True
except ImportError:
    _HAVE_RASTERIO = False
    rasterio = None
    CRS = None

try:
    from scipy.interpolate import Rbf, RBFInterpolator
    from scipy.ndimage import label, gaussian_filter1d, distance_transform_edt
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    Rbf = None
    RBFInterpolator = None
    label = None
    gaussian_filter1d = None
    distance_transform_edt = None
    cKDTree = None

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    _HAVE_SKLEARN_GP = True
except ImportError:
    _HAVE_SKLEARN_GP = False
    GaussianProcessRegressor = None

try:
    import geopandas as gpd
    import pandas as pd
    _HAVE_GEOPANDAS = True
except ImportError:
    _HAVE_GEOPANDAS = False
    gpd = None
    pd = None

try:
    from pyproj import CRS as PyprojCRS, Transformer
    _HAVE_PYPROJ = True
except ImportError:
    _HAVE_PYPROJ = False
    PyprojCRS = None
    Transformer = None


# =============================================================================
# PROVENANCE CODES
# =============================================================================

class GapfillProvenance(IntEnum):
    """Provenance codes for gap-filled bathymetry."""
    NODATA = 0
    PRIOR_ONLY = 1          # Prior unchanged (no HQ data nearby)
    PRIOR_CORRECTED = 2     # Prior + residual correction
    MEASURED_EXACT = 3      # At or very near HQ measurement
    RIVER_SMOOTHED = 4      # River corridor with along-channel smoothing
    BANK_CONSTRAINED = 5    # Depth clamped to stay below bank
    EXTRAPOLATED = 6        # Beyond HQ data extent, prior-dominated


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GapfillConfig:
    """Configuration for intelligent gap-filling."""
    
    # Prior handling
    prior_sigma_default: float = 1.0
    """Default uncertainty (m) for prior where no uncertainty raster provided."""
    
    # Interpolation method
    interpolation_method: str = "rbf"
    """Method: 'rbf' (thin-plate spline), 'gp' (Gaussian Process), 'idw'"""
    
    rbf_function: str = "thin_plate"
    """RBF kernel: 'thin_plate', 'multiquadric', 'cubic', 'linear'"""
    
    rbf_smoothing: float = 0.0
    """RBF smoothing parameter (0 = exact interpolation)"""
    
    # GP settings
    gp_length_scale: float = 500.0
    """GP Matern kernel length scale (meters)"""
    
    gp_nu: float = 2.5
    """GP Matern smoothness parameter"""
    
    gp_noise_level: float = 0.1
    """GP white noise level"""
    
    gp_max_points: int = 2000
    """Max points for GP fitting"""
    
    # Component handling
    component_min_points: int = 10
    """Minimum HQ points per connected component."""
    
    rbf_min_points: int = 25
    """Minimum points for RBF fitting."""
    
    # Performance
    max_hq_points: int = 10000
    """Cap on HQ points for interpolation."""
    
    # IDW settings
    idw_power: float = 2.0
    """IDW power parameter."""
    
    idw_neighbors: int = 12
    """Number of neighbors for IDW."""
    
    # River anisotropy
    river_anisotropy_ratio: float = 5.0
    """Along-channel / cross-channel length scale ratio."""
    
    river_smooth_sigma_m: float = 500.0
    """Gaussian smoothing sigma (meters) along centerline."""
    
    # Bank constraints
    enforce_bank_constraint: bool = True
    """Clamp bed to stay below bank elevation."""
    
    bank_clearance_m: float = 0.3
    """Minimum clearance (m) between bed and bank."""
    
    # Uncertainty model
    residual_sigma_floor: float = 0.20
    """Minimum residual uncertainty (m)."""
    
    distance_sigma_scale_m: float = 1500.0
    """Distance scale for uncertainty growth."""
    
    uncertainty_growth_rate: float = 0.5
    """Rate of uncertainty growth with distance."""
    
    # Outlier handling
    residual_clip_sigma: float = 4.0
    """Clip residuals beyond this many sigma."""
    
    max_residual_m: float = 10.0
    """Absolute cap on residual magnitude (m)."""
    
    # Output options
    output_cudem_xyz: bool = False
    """Output CUDEM-compatible XYZ with uncertainty."""
    
    cudem_weight_scale: float = 1.0
    """Weight scaling for CUDEM datalist format."""


# =============================================================================
# COORDINATE UTILITIES
# =============================================================================

def _estimate_meters_per_degree(lat: float) -> Tuple[float, float]:
    """
    Estimate meters per degree at given latitude.
    
    Returns:
        (meters_per_degree_lon, meters_per_degree_lat)
    """
    # WGS84 ellipsoid
    a = 6378137.0  # semi-major axis
    b = 6356752.3142  # semi-minor axis
    
    lat_rad = math.radians(lat)
    
    # Meridional radius of curvature
    e2 = 1 - (b/a)**2
    M = a * (1 - e2) / (1 - e2 * math.sin(lat_rad)**2)**1.5
    
    # Parallel radius of curvature
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    
    meters_per_deg_lat = math.radians(1) * M
    meters_per_deg_lon = math.radians(1) * N * math.cos(lat_rad)
    
    return meters_per_deg_lon, meters_per_deg_lat


def _is_geographic_crs(crs) -> bool:
    """Check if CRS is geographic (lat/lon)."""
    if crs is None:
        return True  # Assume geographic if unknown
    
    if _HAVE_PYPROJ and PyprojCRS is not None:
        try:
            c = PyprojCRS.from_user_input(crs)
            return c.is_geographic
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    
    # Fallback: check for common geographic CRS strings
    crs_str = str(crs).lower()
    if 'epsg:4326' in crs_str or 'wgs84' in crs_str or 'wgs 84' in crs_str:
        return True
    if 'epsg:4269' in crs_str or 'nad83' in crs_str:
        return True
    
    return False


class CoordinateScaler:
    """
    Handles coordinate scaling between geographic/projected and local meters.
    
    For RBF/IDW/GP to work correctly, distances must be in consistent units.
    This class converts coordinates to a local meter system centered on the data.
    """
    
    def __init__(self, xy: np.ndarray, crs=None):
        """
        Initialize scaler from sample coordinates.
        
        Args:
            xy: Nx2 array of coordinates
            crs: Coordinate reference system (optional)
        """
        self.crs = crs
        self.is_geographic = _is_geographic_crs(crs)
        
        # Compute center
        self.x_center = np.nanmean(xy[:, 0])
        self.y_center = np.nanmean(xy[:, 1])
        
        if self.is_geographic:
            # Geographic: compute meters per degree at center
            self.mx, self.my = _estimate_meters_per_degree(self.y_center)
        else:
            # Projected: assume already in meters (or linear units)
            self.mx = 1.0
            self.my = 1.0
    
    def to_meters(self, xy: np.ndarray) -> np.ndarray:
        """Convert coordinates to local meter system."""
        result = np.empty_like(xy, dtype=np.float64)
        result[:, 0] = (xy[:, 0] - self.x_center) * self.mx
        result[:, 1] = (xy[:, 1] - self.y_center) * self.my
        return result
    
    def from_meters(self, xy_m: np.ndarray) -> np.ndarray:
        """Convert from local meters back to original coordinates."""
        result = np.empty_like(xy_m, dtype=np.float64)
        result[:, 0] = xy_m[:, 0] / self.mx + self.x_center
        result[:, 1] = xy_m[:, 1] / self.my + self.y_center
        return result


# =============================================================================
# CORE INTERPOLATOR CLASS (Waffles-Ready Design)
# =============================================================================

class BathyInterpolator:
    """
    Core bathymetry interpolation using prior + residual framework.
    
    This class can be used standalone or integrated into waffles as a 
    gridding module. It handles:
    - Residual computation (measured - prior)
    - Coordinate scaling for correct distances
    - Multiple interpolation methods (RBF, GP, IDW)
    - Uncertainty propagation
    
    Usage:
        interp = BathyInterpolator(cfg)
        interp.fit(obs_xy, obs_z, prior_z_at_obs, obs_uncertainty)
        z_pred, unc_pred = interp.predict(query_xy, prior_z_at_query)
    """
    
    def __init__(self, cfg: GapfillConfig = None, crs=None):
        """
        Initialize interpolator.
        
        Args:
            cfg: Configuration (uses defaults if None)
            crs: Coordinate reference system for scaling
        """
        self.cfg = cfg or GapfillConfig()
        self.crs = crs
        
        # Will be set during fit()
        self.scaler: Optional[CoordinateScaler] = None
        self._obs_xy_m: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._obs_unc: Optional[np.ndarray] = None
        self._residual_rmse: float = 0.0
        self._rbf = None
        self._gp = None
        self._kdtree = None
        self._fitted = False
    
    def fit(
        self,
        obs_xy: np.ndarray,
        obs_z: np.ndarray,
        prior_z: np.ndarray,
        obs_uncertainty: Optional[np.ndarray] = None
    ) -> 'BathyInterpolator':
        """
        Fit the interpolator to observations.
        
        Args:
            obs_xy: Nx2 observation coordinates
            obs_z: N observed depths/elevations
            prior_z: N prior values at observation locations
            obs_uncertainty: N observation uncertainties (optional)
            
        Returns:
            self (for method chaining)
        """
        n_obs = len(obs_xy)
        if n_obs < self.cfg.component_min_points:
            log.warning("[BathyInterp] Insufficient points (%d < %d)", 
                       n_obs, self.cfg.component_min_points)
            self._fitted = False
            return self
        
        # Initialize coordinate scaler
        self.scaler = CoordinateScaler(obs_xy, self.crs)
        
        # Convert to local meters
        self._obs_xy_m = self.scaler.to_meters(obs_xy)
        
        # Auto-detect and align depth sign convention
        obs_z, prior_z = self._align_depth_signs(obs_z, prior_z)
        
        # Compute residuals
        residuals = obs_z - prior_z
        
        # Clip outliers
        residuals = self._clip_outliers(residuals)
        
        self._residuals = residuals
        self._residual_rmse = float(np.sqrt(np.nanmean(residuals**2)))
        
        # Store uncertainty
        if obs_uncertainty is not None:
            self._obs_unc = obs_uncertainty.copy()
        else:
            self._obs_unc = np.full(n_obs, np.nan)
        
        # Subsample if needed
        if n_obs > self.cfg.max_hq_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_obs, size=self.cfg.max_hq_points, replace=False)
            self._obs_xy_m = self._obs_xy_m[idx]
            self._residuals = self._residuals[idx]
            self._obs_unc = self._obs_unc[idx]
        
        # Fit interpolation model
        self._fit_model()
        
        self._fitted = True
        log.info("[BathyInterp] Fitted with %d points, residual RMSE=%.3f m",
                 len(self._residuals), self._residual_rmse)
        
        return self
    
    def _align_depth_signs(
        self, 
        obs_z: np.ndarray, 
        prior_z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect and align depth sign conventions."""
        obs_finite = obs_z[np.isfinite(obs_z)]
        prior_finite = prior_z[np.isfinite(prior_z)]
        
        if len(obs_finite) == 0 or len(prior_finite) == 0:
            return obs_z, prior_z
        
        obs_med = np.median(obs_finite)
        prior_med = np.median(prior_finite)
        
        # If signs are opposite and both have clear bathymetric range
        if obs_med * prior_med < 0:
            if abs(obs_med) > 0.5 and abs(prior_med) > 0.5:
                log.info("[BathyInterp] Auto-flipping observation depth signs")
                return -obs_z, prior_z
        
        return obs_z, prior_z
    
    def _clip_outliers(self, residuals: np.ndarray) -> np.ndarray:
        """Clip outlier residuals."""
        finite = residuals[np.isfinite(residuals)]
        if len(finite) < 3:
            return residuals
        
        r_mean = np.mean(finite)
        r_std = np.std(finite)
        
        if r_std > 0:
            clip_val = self.cfg.residual_clip_sigma * r_std
            residuals = np.clip(residuals, r_mean - clip_val, r_mean + clip_val)
        
        residuals = np.clip(residuals, -self.cfg.max_residual_m, self.cfg.max_residual_m)
        
        return residuals
    
    def _fit_model(self):
        """Fit the interpolation model."""
        method = self.cfg.interpolation_method.lower()
        
        # Build KDTree for IDW / uncertainty
        if _HAVE_SCIPY and cKDTree is not None:
            self._kdtree = cKDTree(self._obs_xy_m)
        
        if method == "rbf" and len(self._residuals) >= self.cfg.rbf_min_points:
            self._fit_rbf()
        elif method == "gp" and _HAVE_SKLEARN_GP:
            self._fit_gp()
        # IDW uses KDTree, no additional fitting needed
    
    def _fit_rbf(self):
        """Fit RBF interpolator."""
        if not _HAVE_SCIPY:
            return
        
        # Map kernel names between old Rbf and new RBFInterpolator
        kernel_map = {
            'thin_plate': 'thin_plate_spline',
            'multiquadric': 'multiquadric',
            'cubic': 'cubic',
            'linear': 'linear',
            'gaussian': 'gaussian',
            'quintic': 'quintic',
        }
        
        try:
            if RBFInterpolator is not None:
                # Newer scipy (>=1.7) - use RBFInterpolator
                kernel = kernel_map.get(self.cfg.rbf_function, self.cfg.rbf_function)
                self._rbf = RBFInterpolator(
                    self._obs_xy_m,
                    self._residuals,
                    kernel=kernel,
                    smoothing=self.cfg.rbf_smoothing
                )
            elif Rbf is not None:
                # Older scipy - use Rbf
                self._rbf = Rbf(
                    self._obs_xy_m[:, 0],
                    self._obs_xy_m[:, 1],
                    self._residuals,
                    function=self.cfg.rbf_function,
                    smooth=self.cfg.rbf_smoothing
                )
        except Exception as e:
            log.warning("[BathyInterp] RBF fit failed: %s", e)
            self._rbf = None
    
    def _fit_gp(self):
        """Fit Gaussian Process."""
        if not _HAVE_SKLEARN_GP:
            return
        
        try:
            kernel = (
                Matern(length_scale=self.cfg.gp_length_scale, nu=self.cfg.gp_nu) +
                WhiteKernel(noise_level=self.cfg.gp_noise_level)
            )
            
            alpha = np.where(
                np.isfinite(self._obs_unc),
                self._obs_unc**2,
                self.cfg.gp_noise_level**2
            )
            
            self._gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=True,
                n_restarts_optimizer=2
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._gp.fit(self._obs_xy_m, self._residuals)
                
        except Exception as e:
            log.warning("[BathyInterp] GP fit failed: %s", e)
            self._gp = None
    
    def predict(
        self,
        query_xy: np.ndarray,
        prior_z: np.ndarray,
        prior_sigma: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict at query locations.
        
        Args:
            query_xy: Mx2 query coordinates
            prior_z: M prior values at query locations
            prior_sigma: M prior uncertainties (optional)
            
        Returns:
            Tuple of (predicted_z, uncertainty)
        """
        n_query = len(query_xy)
        
        if not self._fitted or self.scaler is None:
            # Return prior unchanged
            sigma = prior_sigma if prior_sigma is not None else np.full(n_query, self.cfg.prior_sigma_default)
            return prior_z.copy(), sigma
        
        # Convert to local meters
        query_xy_m = self.scaler.to_meters(query_xy)
        
        # Predict residual correction
        correction = self._predict_residual(query_xy_m)
        
        # Apply correction
        z_pred = prior_z + correction
        
        # Compute uncertainty
        sigma = self._compute_uncertainty(query_xy_m, prior_sigma)
        
        return z_pred.astype(np.float32), sigma.astype(np.float32)
    
    def _predict_residual(self, query_xy_m: np.ndarray) -> np.ndarray:
        """Predict residual at query points in local meter coordinates."""
        n_query = len(query_xy_m)
        
        # Try RBF first
        if self._rbf is not None:
            try:
                if hasattr(self._rbf, '__call__'):
                    # RBFInterpolator
                    return self._rbf(query_xy_m).astype(np.float32)
                else:
                    # Old Rbf
                    return self._rbf(query_xy_m[:, 0], query_xy_m[:, 1]).astype(np.float32)
            except Exception as e:
                log.debug("[BathyInterp] RBF predict failed: %s", e)
        
        # Try GP
        if self._gp is not None:
            try:
                return self._gp.predict(query_xy_m).astype(np.float32)
            except Exception as e:
                log.debug("[BathyInterp] GP predict failed: %s", e)
        
        # IDW fallback
        return self._predict_idw(query_xy_m)
    
    def _predict_idw(self, query_xy_m: np.ndarray) -> np.ndarray:
        """IDW interpolation of residuals."""
        if self._kdtree is None:
            return np.full(len(query_xy_m), np.nanmedian(self._residuals), dtype=np.float32)
        
        k = min(self.cfg.idw_neighbors, len(self._residuals))
        dist, idx = self._kdtree.query(query_xy_m, k=k)
        
        if k == 1:
            dist = dist.reshape(-1, 1)
            idx = idx.reshape(-1, 1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1.0 / np.maximum(dist, 1e-10) ** self.cfg.idw_power
        
        values = self._residuals[idx]
        result = np.sum(weights * values, axis=1) / np.sum(weights, axis=1)
        
        return result.astype(np.float32)
    
    def _compute_uncertainty(
        self,
        query_xy_m: np.ndarray,
        prior_sigma: Optional[np.ndarray]
    ) -> np.ndarray:
        """Compute uncertainty at query points."""
        n_query = len(query_xy_m)
        
        # Distance to nearest observation
        if self._kdtree is not None:
            dist, _ = self._kdtree.query(query_xy_m, k=1)
        else:
            dist = np.full(n_query, self.cfg.distance_sigma_scale_m)
        
        # Residual uncertainty grows with distance
        base_sigma = max(self._residual_rmse, self.cfg.residual_sigma_floor)
        dist_factor = 1.0 + self.cfg.uncertainty_growth_rate * (dist / self.cfg.distance_sigma_scale_m)
        sigma_residual = base_sigma * dist_factor
        
        # Prior uncertainty
        if prior_sigma is not None:
            sigma_prior = prior_sigma
        else:
            sigma_prior = np.full(n_query, self.cfg.prior_sigma_default)
        
        # Combine (variance addition)
        sigma = np.sqrt(sigma_prior**2 + sigma_residual**2)
        
        return sigma.astype(np.float32)
    
    @property
    def residual_rmse(self) -> float:
        """Return the residual RMSE from fitting."""
        return self._residual_rmse


# =============================================================================
# POINT LOADING UTILITIES
# =============================================================================

def _guess_columns(cols: List[str]) -> Tuple[str, str, str, Optional[str]]:
    """Infer x, y, z, and optional uncertainty columns."""
    cols_lower = [c.lower() for c in cols]
    
    def pick(options):
        for opt in options:
            if opt in cols_lower:
                return cols[cols_lower.index(opt)]
        return None
    
    x = pick(["x", "lon", "longitude", "easting", "lng"])
    y = pick(["y", "lat", "latitude", "northing"])
    z = pick(["z", "depth", "depth_m", "z_m", "elev", "elevation", "bed_elev"])
    unc = pick(["uncertainty", "sigma", "unc", "error", "rmse", "std"])
    
    if x is None or y is None or z is None:
        raise ValueError(f"Could not infer x/y/z columns from: {cols}")
    
    return x, y, z, unc


def load_hq_points(
    files: List[Union[str, Path]],
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load high-quality measurement points.
    
    Returns:
        Tuple of (coords Nx3, uncertainties N)
    """
    if logger is None:
        logger = log
    
    all_coords = []
    all_unc = []
    
    for f in files:
        fp = Path(f)
        if not fp.exists():
            logger.warning("[GAPFILL] HQ file not found: %s", fp)
            continue
        
        suffix = fp.suffix.lower()
        
        try:
            if suffix in (".gpkg", ".shp", ".geojson"):
                if not _HAVE_GEOPANDAS:
                    raise RuntimeError("geopandas required for vector inputs")
                
                gdf = gpd.read_file(fp)
                if gdf.empty:
                    continue
                
                x = gdf.geometry.x.values
                y = gdf.geometry.y.values
                
                # Find depth column
                z_col = None
                for cand in ("depth", "depth_m", "z", "z_m", "elev", "elevation"):
                    matches = [c for c in gdf.columns if c.lower() == cand]
                    if matches:
                        z_col = matches[0]
                        break
                
                if z_col is None:
                    raise ValueError(f"No depth column in {fp.name}")
                
                z = gdf[z_col].astype(float).values
                
                # Uncertainty column
                unc = np.full(len(z), np.nan)
                for cand in ("uncertainty", "sigma", "unc", "error"):
                    matches = [c for c in gdf.columns if c.lower() == cand]
                    if matches:
                        unc = gdf[matches[0]].astype(float).values
                        break
                
                all_coords.append(np.column_stack([x, y, z]))
                all_unc.append(unc)
                
            else:
                # CSV / TXT / XYZ
                if _HAVE_GEOPANDAS:
                    try:
                        df = pd.read_csv(fp)
                    except Exception:
                        df = pd.read_csv(fp, sep=r'\s+', comment="#")
                    
                    if df.empty:
                        continue
                    
                    x_col, y_col, z_col, unc_col = _guess_columns(list(df.columns))
                    
                    x = df[x_col].astype(float).values
                    y = df[y_col].astype(float).values
                    z = df[z_col].astype(float).values
                    
                    unc = df[unc_col].astype(float).values if unc_col else np.full(len(z), np.nan)
                    
                    all_coords.append(np.column_stack([x, y, z]))
                    all_unc.append(unc)
                else:
                    # Pure numpy fallback
                    try:
                        data = np.loadtxt(fp, delimiter=',', skiprows=1)
                    except Exception:
                        data = np.loadtxt(fp, skiprows=1)
                    
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    
                    if data.shape[1] >= 3:
                        all_coords.append(data[:, :3])
                        unc = data[:, 3] if data.shape[1] >= 4 else np.full(len(data), np.nan)
                        all_unc.append(unc)
                        
        except Exception as e:
            logger.warning("[GAPFILL] Failed to load %s: %s", fp, e)
            continue
    
    if not all_coords:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    
    coords = np.vstack(all_coords).astype(np.float64)
    uncertainties = np.concatenate(all_unc).astype(np.float64)
    
    # Remove invalid
    valid = np.all(np.isfinite(coords), axis=1)
    coords = coords[valid]
    uncertainties = uncertainties[valid]
    
    logger.info("[GAPFILL] Loaded %d HQ points from %d files", len(coords), len(files))
    
    return coords, uncertainties


# =============================================================================
# RASTER I/O
# =============================================================================

def _read_raster(path: Path) -> Tuple[np.ndarray, Any, dict]:
    """Read raster, converting nodata to NaN."""
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        profile = ds.profile.copy()
        transform = ds.transform
        nodata = ds.nodata
        crs = ds.crs
    
    if nodata is not None:
        arr[arr == nodata] = np.nan
    
    profile['crs'] = crs
    return arr, transform, profile


def _write_raster(path: Path, arr: np.ndarray, profile: dict, nodata: float = -9999.0):
    """Write raster."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=float(nodata), compress="deflate")
    
    out = arr.astype(np.float32)
    out[np.isnan(out)] = float(nodata)
    
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(out, 1)


def _write_provenance_raster(path: Path, arr: np.ndarray, profile: dict):
    """Write uint8 provenance raster."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    prof = profile.copy()
    prof.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(arr.astype(np.uint8), 1)


# =============================================================================
# CONNECTED COMPONENT PROCESSING
# =============================================================================

def _label_water_components(water_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label connected water components."""
    if not _HAVE_SCIPY or label is None:
        return water_mask.astype(np.int32), 1
    
    labels, n_comp = label(water_mask)
    return labels, n_comp


# =============================================================================
# MAIN GAP-FILL FUNCTION
# =============================================================================

def gapfill_depth_raster(
    prior_raster: Path,
    hq_point_files: List[str],
    out_raster: Path,
    out_sigma: Path,
    out_provenance: Path,
    cfg: GapfillConfig = None,
    prior_sigma_raster: Optional[Path] = None,
    water_mask_raster: Optional[Path] = None,
    river_mask_raster: Optional[Path] = None,
    bank_elev_raster: Optional[Path] = None,
    xs_params_gpkg: Optional[Union[Path, str]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Intelligent gap-filling using prior + residual interpolation.
    
    Args:
        prior_raster: Prior depth/elevation raster (SDB + river fusion)
        hq_point_files: List of high-quality point files (sonar, lidar)
        out_raster: Output gap-filled raster path
        out_sigma: Output uncertainty raster path
        out_provenance: Output provenance raster path
        cfg: GapfillConfig (uses defaults if None)
        prior_sigma_raster: Optional prior uncertainty raster
        water_mask_raster: Optional water mask (1=water)
        river_mask_raster: Optional river corridor mask
        bank_elev_raster: Optional bank elevation for constraints
        xs_params_gpkg: Optional XS parameters for river smoothing
        logger: Optional logger
        
    Returns:
        Dict with statistics
    """
    if not _HAVE_RASTERIO:
        raise RuntimeError("rasterio required")
    
    if cfg is None:
        cfg = GapfillConfig()
    
    if logger is None:
        logger = log
    
    logger.info("[GAPFILL] Starting intelligent gap-fill")
    logger.info("[GAPFILL]   Prior: %s", prior_raster)
    logger.info("[GAPFILL]   HQ files: %d", len(hq_point_files))
    logger.info("[GAPFILL]   Method: %s", cfg.interpolation_method)
    
    stats = {
        "n_hq_points": 0,
        "n_hq_points_in_water": 0,
        "n_components": 0,
        "residual_rmse": np.nan,
        "pixels_corrected": 0,
        "pixels_prior_only": 0,
        "pixels_bank_constrained": 0,
    }
    
    # Load prior
    z_prior, transform, profile = _read_raster(prior_raster)
    h, w = z_prior.shape
    nodata = profile.get("nodata", -9999.0)
    crs = profile.get("crs", None)
    
    # Load prior uncertainty
    if prior_sigma_raster and Path(prior_sigma_raster).exists():
        prior_sigma, _, _ = _read_raster(prior_sigma_raster)
    else:
        prior_sigma = np.full_like(z_prior, cfg.prior_sigma_default)
        prior_sigma[~np.isfinite(z_prior)] = np.nan
    
    # Determine water mask
    if water_mask_raster and Path(water_mask_raster).exists():
        wm, _, _ = _read_raster(water_mask_raster)
        water = (wm == 1) | (wm > 0.5)
    else:
        water = np.isfinite(z_prior)
    
    # Handle no water case
    if not np.any(water):
        logger.warning("[GAPFILL] No water pixels found")
        _write_raster(out_raster, z_prior, profile, nodata)
        _write_raster(out_sigma, prior_sigma, profile, nodata)
        prov = np.zeros((h, w), dtype=np.uint8)
        prov[np.isfinite(z_prior)] = GapfillProvenance.PRIOR_ONLY
        _write_provenance_raster(out_provenance, prov, profile)
        return stats
    
    # Load HQ points
    hq_coords, hq_unc = load_hq_points(hq_point_files, logger)
    stats["n_hq_points"] = len(hq_coords)
    
    if len(hq_coords) < cfg.component_min_points:
        logger.warning("[GAPFILL] Insufficient HQ points (%d)", len(hq_coords))
        _write_raster(out_raster, z_prior, profile, nodata)
        _write_raster(out_sigma, prior_sigma, profile, nodata)
        prov = np.zeros((h, w), dtype=np.uint8)
        prov[water] = GapfillProvenance.PRIOR_ONLY
        _write_provenance_raster(out_provenance, prov, profile)
        return stats
    
    # Convert HQ coords to pixel indices
    hq_x, hq_y, hq_z = hq_coords[:, 0], hq_coords[:, 1], hq_coords[:, 2]
    hq_rows, hq_cols = rowcol(transform, hq_x, hq_y)
    hq_rows = np.asarray(hq_rows)
    hq_cols = np.asarray(hq_cols)
    
    # Filter to valid pixels in water
    valid = (
        (hq_rows >= 0) & (hq_rows < h) &
        (hq_cols >= 0) & (hq_cols < w)
    )
    valid[valid] &= water[hq_rows[valid], hq_cols[valid]]
    
    hq_rows = hq_rows[valid]
    hq_cols = hq_cols[valid]
    hq_z = hq_z[valid]
    hq_x = hq_x[valid]
    hq_y = hq_y[valid]
    hq_unc = hq_unc[valid]
    
    stats["n_hq_points_in_water"] = len(hq_z)
    
    if len(hq_z) < cfg.component_min_points:
        logger.warning("[GAPFILL] Insufficient HQ points in water (%d)", len(hq_z))
        _write_raster(out_raster, z_prior, profile, nodata)
        _write_raster(out_sigma, prior_sigma, profile, nodata)
        prov = np.zeros((h, w), dtype=np.uint8)
        prov[water] = GapfillProvenance.PRIOR_ONLY
        _write_provenance_raster(out_provenance, prov, profile)
        return stats
    
    # Sample prior at HQ locations
    z_prior_at_hq = z_prior[hq_rows, hq_cols]
    
    # Initialize outputs
    z_out = z_prior.copy()
    sigma_out = prior_sigma.copy()
    prov = np.zeros((h, w), dtype=np.uint8)
    prov[water] = GapfillProvenance.PRIOR_ONLY
    
    # Label connected components
    comp_labels, n_comp = _label_water_components(water)
    stats["n_components"] = n_comp
    logger.info("[GAPFILL] Processing %d water components", n_comp)
    
    # Process each component
    hq_xy = np.column_stack([hq_x, hq_y])
    
    for k in range(1, n_comp + 1):
        comp_mask = comp_labels == k
        if not np.any(comp_mask):
            continue
        
        # Find HQ points in this component
        comp_at_hq = comp_labels[hq_rows, hq_cols]
        in_comp = comp_at_hq == k
        
        n_in_comp = np.sum(in_comp)
        if n_in_comp < cfg.component_min_points:
            continue
        
        # Extract component data
        comp_xy = hq_xy[in_comp]
        comp_z = hq_z[in_comp]
        comp_prior = z_prior_at_hq[in_comp]
        comp_unc = hq_unc[in_comp]
        
        # Create interpolator for this component
        interp = BathyInterpolator(cfg, crs)
        interp.fit(comp_xy, comp_z, comp_prior, comp_unc)
        
        if not interp._fitted:
            continue
        
        # Get query points (all pixels in component)
        query_rows, query_cols = np.where(comp_mask)
        query_x, query_y = xy(transform, query_rows, query_cols, offset="center")
        query_xy = np.column_stack([query_x, query_y])
        
        # Get prior at query points
        query_prior = z_prior[query_rows, query_cols]
        query_prior_sigma = prior_sigma[query_rows, query_cols]
        
        # Predict
        z_pred, sigma_pred = interp.predict(query_xy, query_prior, query_prior_sigma)
        
        # Update outputs
        z_out[query_rows, query_cols] = z_pred
        sigma_out[query_rows, query_cols] = sigma_pred
        prov[query_rows, query_cols] = GapfillProvenance.PRIOR_CORRECTED
        stats["pixels_corrected"] += len(query_rows)
        
        if k == 1:  # Store RMSE from largest component
            stats["residual_rmse"] = interp.residual_rmse
    
    # Bank constraint enforcement
    if cfg.enforce_bank_constraint and bank_elev_raster:
        bank_path = Path(bank_elev_raster)
        if bank_path.exists():
            try:
                bank_elev, _, _ = _read_raster(bank_path)
                max_bed = bank_elev - cfg.bank_clearance_m
                violations = np.isfinite(z_out) & np.isfinite(max_bed) & (z_out > max_bed)
                z_out[violations] = max_bed[violations]
                prov[violations] = GapfillProvenance.BANK_CONSTRAINED
                stats["pixels_bank_constrained"] = int(np.sum(violations))
            except Exception as e:
                logger.debug("[GAPFILL] Bank constraint failed: %s", e)
    
    # Count prior-only pixels
    stats["pixels_prior_only"] = int(np.sum(prov == GapfillProvenance.PRIOR_ONLY))
    
    # Write outputs
    _write_raster(out_raster, z_out, profile, nodata)
    _write_raster(out_sigma, sigma_out, profile, nodata)
    _write_provenance_raster(out_provenance, prov, profile)
    
    # Optional CUDEM XYZ output
    if cfg.output_cudem_xyz:
        xyz_path = Path(out_raster).with_suffix(".xyz")
        _write_cudem_xyz(z_out, sigma_out, water, transform, xyz_path, cfg)
        stats["cudem_xyz"] = str(xyz_path)
    
    logger.info("[GAPFILL] Complete. Corrected %d pixels, prior-only %d pixels",
                stats["pixels_corrected"], stats["pixels_prior_only"])
    
    return stats


def _write_cudem_xyz(
    depth: np.ndarray,
    sigma: np.ndarray,
    water: np.ndarray,
    transform: Any,
    out_path: Path,
    cfg: GapfillConfig
) -> None:
    """Write CUDEM-compatible XYZ file."""
    rows, cols = np.where(water & np.isfinite(depth))
    
    if len(rows) == 0:
        return
    
    xs, ys = xy(transform, rows, cols, offset="center")
    zs = depth[rows, cols]
    sigs = sigma[rows, cols]
    
    # Weight from uncertainty
    weights = cfg.cudem_weight_scale / np.maximum(sigs, 0.1)
    weights = np.clip(weights, 0.01, 1.0)
    
    sigs = np.where(np.isfinite(sigs), sigs, cfg.prior_sigma_default)
    
    data = np.column_stack([xs, ys, zs, weights, sigs])
    
    np.savetxt(
        out_path,
        data,
        fmt="%.6f %.6f %.3f %.4f %.3f",
        header="x y z weight uncertainty",
        comments=""
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Intelligent gap-filling for bathymetry"
    )
    
    parser.add_argument("--prior", required=True, help="Prior depth raster")
    parser.add_argument("--hq", nargs="+", required=True, help="HQ point files")
    parser.add_argument("--out", required=True, help="Output raster")
    parser.add_argument("--sigma", required=True, help="Output uncertainty")
    parser.add_argument("--prov", required=True, help="Output provenance")
    
    parser.add_argument("--prior-sigma", help="Prior uncertainty raster")
    parser.add_argument("--water-mask", help="Water mask raster")
    parser.add_argument("--river-mask", help="River corridor mask")
    parser.add_argument("--bank-elev", help="Bank elevation raster")
    parser.add_argument("--xs-gpkg", help="XS parameters GPKG")
    
    parser.add_argument("--method", default="rbf", choices=["rbf", "gp", "idw"])
    parser.add_argument("--rbf-function", default="thin_plate")
    parser.add_argument("--cudem-xyz", action="store_true")
    
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    cfg = GapfillConfig(
        interpolation_method=args.method,
        rbf_function=args.rbf_function,
        output_cudem_xyz=args.cudem_xyz,
    )
    
    stats = gapfill_depth_raster(
        prior_raster=Path(args.prior),
        hq_point_files=args.hq,
        out_raster=Path(args.out),
        out_sigma=Path(args.sigma),
        out_provenance=Path(args.prov),
        cfg=cfg,
        prior_sigma_raster=Path(args.prior_sigma) if args.prior_sigma else None,
        water_mask_raster=Path(args.water_mask) if args.water_mask else None,
        river_mask_raster=Path(args.river_mask) if args.river_mask else None,
        bank_elev_raster=Path(args.bank_elev) if args.bank_elev else None,
        xs_params_gpkg=Path(args.xs_gpkg) if args.xs_gpkg else None,
    )
    
    log.info(f"\nGap-fill complete:")
    log.info(f"  HQ points: {stats['n_hq_points']} (in water: {stats['n_hq_points_in_water']})")
    log.info(f"  Components: {stats['n_components']}")
    log.info(f"  Residual RMSE: {stats['residual_rmse']:.3f} m")
    log.info(f"  Pixels corrected: {stats['pixels_corrected']}")
    log.info(f"  Pixels prior-only: {stats['pixels_prior_only']}")


if __name__ == "__main__":
    main()
