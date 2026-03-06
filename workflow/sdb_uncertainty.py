#!/usr/bin/env python3
"""
sdb_uncertainty.py - Uncertainty-Aware Satellite Derived Bathymetry

This module enables SDB prediction with comprehensive uncertainty quantification,
supporting scenarios from pure physics-based prediction (no training data) to
fully calibrated ML models with ATL validation.

UNCERTAINTY FRAMEWORK
=====================

All uncertainty sources are combined as VARIANCES (σ²) then converted to 
standard deviation at the end. This is the statistically correct way to 
combine independent error sources:

    σ_total² = σ_model² + σ_optical² + σ_spatial² + σ_extrapolation² + ...
    σ_total = sqrt(σ_total²)

The final uncertainty raster represents ±1 standard deviation (68% confidence).
For 95% confidence intervals, multiply by 1.96.

Uncertainty Sources:
--------------------
1. MODEL UNCERTAINTY (σ_model):
   - For RF: Standard deviation across tree predictions
   - For physics: Parameter uncertainty propagated through model
   - Captures: Model structure, training noise, parameter uncertainty

2. OPTICAL/SIGNAL UNCERTAINTY (σ_optical):
   - Increases with optical depth (Kd × z)
   - Captures: Signal attenuation, bottom signal weakness at depth
   - Formula: σ_optical = σ_base × (1 - exp(-2×Kd×z)) × depth_factor

3. SPATIAL/EPISTEMIC UNCERTAINTY (σ_spatial):
   - Distance from training/validation data
   - Captures: Lack of local calibration, geographic extrapolation

4. EXTRAPOLATION UNCERTAINTY (σ_extrap):
   - Beyond training depth range
   - Outside training feature bounds (Domain of Applicability)
   - Captures: Model behavior in untrained regions

5. INPUT DATA UNCERTAINTY (σ_input):
   - Sentinel-2 radiometric uncertainty (~3-5%)
   - Atmospheric correction residuals
   - Captures: Measurement error in input bands

Output:
-------
- Depth raster (meters, positive down)
- Uncertainty raster (meters, 1-sigma)
- Confidence raster (0-1, derived from uncertainty)

The uncertainty value at each pixel means:
"There is a 68% probability the true depth is within ±uncertainty of predicted depth"
"""

import logging
from typing import Dict, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Uncertainty model parameters
# -----------------------------------------------------------------------------

@dataclass
class UncertaintyParams:
    """
    Parameters controlling uncertainty estimation.
    
    All uncertainty values are in meters (1-sigma standard deviation).
    Variances are computed internally and combined, then sqrt for final σ.
    """
    # --- Model/Base Uncertainty (σ_model) ---
    # Irreducible model error even in ideal conditions
    base_sigma_calibrated: float = 0.10      # Well-trained RF with good ATL data
    base_sigma_transfer: float = 0.25        # Model from different region
    base_sigma_physics: float = 0.40         # Physics model, regional params
    base_sigma_uncalibrated: float = 0.80    # Physics model, default params
    
    # --- Optical/Signal Uncertainty (σ_optical) ---
    # Uncertainty due to light attenuation - increases with optical depth
    optical_sigma_shallow: float = 0.05      # σ at surface (z≈0)
    optical_sigma_scale: float = 0.15        # Additional σ per unit optical depth
    
    # --- Depth-Dependent Uncertainty ---
    # Linear growth with depth (independent of optical effects)
    depth_sigma_per_meter: float = 0.03      # σ grows 3cm per meter depth
    
    # --- Extrapolation Uncertainty (σ_extrap) ---
    # Penalty for predictions outside training domain
    extrap_sigma_per_meter_beyond: float = 0.20  # Per meter beyond max training depth
    extrap_sigma_feature_outside: float = 0.30   # If outside feature bounds (DoA)
    
    # --- Spatial Uncertainty (σ_spatial) ---
    # Distance from nearest training/calibration point
    spatial_sigma_base: float = 0.10         # Base when far from training data
    spatial_sigma_per_km: float = 0.05       # Additional σ per km from nearest ATL
    spatial_max_distance_km: float = 50.0    # Cap distance effect
    
    # --- Input Data Uncertainty (σ_input) ---
    # Sentinel-2 measurement uncertainty propagated through model
    s2_radiometric_cv: float = 0.03          # ~3% coefficient of variation
    input_sigma_base: float = 0.08           # Propagated to depth uncertainty
    
    # --- Limits ---
    max_uncertainty_m: float = 10.0          # Cap total uncertainty
    min_uncertainty_m: float = 0.05          # Floor (can't be more certain than this)


# -----------------------------------------------------------------------------
# Physics-based SDB (no training required)
# -----------------------------------------------------------------------------

def stumpf_ratio_model(
    blue: np.ndarray,
    green: np.ndarray,
    *,
    m0: float = 0.0,
    m1: float = 5.0,
    n: float = 1000.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Stumpf et al. (2003) log-ratio model for bathymetry.
    
    depth = m0 + m1 * ln(n * blue) / ln(n * green)
    
    Parameters
    ----------
    blue, green : np.ndarray
        Reflectance values (typically B02, B03 for Sentinel-2)
    m0 : float
        Offset parameter (intercept)
    m1 : float
        Scaling parameter (slope) - controls depth range
    n : float
        Constant to ensure positive log arguments
        
    Returns
    -------
    np.ndarray
        Estimated depth in meters (positive down)
    """
    blue_safe = np.maximum(blue, eps)
    green_safe = np.maximum(green, eps)
    
    log_blue = np.log(n * blue_safe)
    log_green = np.log(n * green_safe)
    log_green_safe = np.where(np.abs(log_green) < eps, eps, log_green)
    
    ratio = log_blue / log_green_safe
    depth = m0 + m1 * ratio
    
    return depth.astype(np.float32)


def lyzenga_linear_model(
    bands: Dict[str, np.ndarray],
    coefficients: Dict[str, float],
    l_inf: Dict[str, float] = None,
    *,
    intercept: float = 0.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Lyzenga (1978, 1985) linearized depth model.
    
    depth = intercept + sum(coef_i * ln(band_i - L_inf_i))
    
    Parameters
    ----------
    bands : dict
        Band arrays (e.g., {"B02": array, "B03": array})
    coefficients : dict
        Regression coefficients per band
    l_inf : dict
        Deep water radiance per band
    intercept : float
        Model intercept
        
    Returns
    -------
    np.ndarray
        Estimated depth in meters
    """
    if l_inf is None:
        l_inf = {k: 0.0 for k in bands}
    
    depth = np.full_like(list(bands.values())[0], intercept, dtype=np.float32)
    
    for band_name, coef in coefficients.items():
        if band_name in bands:
            band = bands[band_name].astype(np.float32)
            linf = l_inf.get(band_name, 0.0)
            corrected = np.maximum(band - linf, eps)
            depth += coef * np.log(corrected)
    
    return depth


# -----------------------------------------------------------------------------
# Regional parameter lookup
# -----------------------------------------------------------------------------

# Default parameters for different water types
# These are rough estimates - real applications should calibrate locally
REGIONAL_PARAMS = {
    "caribbean_clear": {
        "stumpf": {"m0": -0.5, "m1": 6.5, "n": 1000},
        "kd_490_typical": 0.08,
        "max_reliable_depth": 25.0,
        "description": "Clear Caribbean waters, coral/sand bottom"
    },
    "florida_keys": {
        "stumpf": {"m0": -0.3, "m1": 5.5, "n": 1000},
        "kd_490_typical": 0.12,
        "max_reliable_depth": 18.0,
        "description": "Florida Keys, mixed bottom types"
    },
    "pacific_islands": {
        "stumpf": {"m0": -0.4, "m1": 6.0, "n": 1000},
        "kd_490_typical": 0.06,
        "max_reliable_depth": 30.0,
        "description": "Clear Pacific waters"
    },
    "temperate_coastal": {
        "stumpf": {"m0": -0.2, "m1": 4.0, "n": 1000},
        "kd_490_typical": 0.25,
        "max_reliable_depth": 10.0,
        "description": "Temperate coastal, moderate turbidity"
    },
    "turbid_coastal": {
        "stumpf": {"m0": 0.0, "m1": 2.5, "n": 1000},
        "kd_490_typical": 0.50,
        "max_reliable_depth": 5.0,
        "description": "Turbid coastal/estuarine"
    },
    "default": {
        "stumpf": {"m0": 0.0, "m1": 5.0, "n": 1000},
        "kd_490_typical": 0.15,
        "max_reliable_depth": 15.0,
        "description": "Generic default parameters"
    }
}


def get_regional_params(region: str = "default") -> Dict[str, Any]:
    """Get physics model parameters for a region."""
    return REGIONAL_PARAMS.get(region, REGIONAL_PARAMS["default"])


def estimate_region_from_location(lon: float, lat: float) -> str:
    """
    Rough region estimation based on coordinates.
    
    In practice, you'd want a proper water mass classification.
    """
    # Very rough geographic heuristics
    if 15 < lat < 30 and -90 < lon < -60:
        return "caribbean_clear"
    elif 24 < lat < 27 and -83 < lon < -79:
        return "florida_keys"
    elif -30 < lat < 30 and (lon < -100 or lon > 100):
        return "pacific_islands"
    elif 30 < lat < 60 or -60 < lat < -30:
        return "temperate_coastal"
    else:
        return "default"


# -----------------------------------------------------------------------------
# Uncertainty estimation - VARIANCE-BASED APPROACH
# -----------------------------------------------------------------------------

def compute_model_variance(
    mode: str,
    rf_tree_std: np.ndarray = None,
    params: UncertaintyParams = None,
) -> np.ndarray:
    """
    Compute model uncertainty variance (σ²_model).
    
    For RF models: Uses standard deviation across tree predictions.
    For physics models: Uses base parameter uncertainty.
    
    Returns variance (σ²), not standard deviation.
    """
    if params is None:
        params = UncertaintyParams()
    
    if mode == "calibrated":
        base_sigma = params.base_sigma_calibrated
    elif mode == "transfer":
        base_sigma = params.base_sigma_transfer
    elif mode == "physics":
        base_sigma = params.base_sigma_physics
    else:
        base_sigma = params.base_sigma_uncalibrated
    
    # Base variance
    base_var = base_sigma ** 2
    
    # Add RF tree variance if available
    if rf_tree_std is not None:
        rf_var = rf_tree_std ** 2
        # Combine: total model variance = base² + rf_tree²
        return (base_var + rf_var).astype(np.float32)
    else:
        return np.float32(base_var)


def compute_optical_variance(
    kd_map: np.ndarray,
    depth_map: np.ndarray,
    params: UncertaintyParams = None,
) -> np.ndarray:
    """
    Compute optical/signal uncertainty variance (σ²_optical).
    
    As light attenuates with depth, the bottom signal weakens and 
    depth estimates become less certain.
    
    σ_optical = σ_shallow + σ_scale × optical_depth
    where optical_depth = Kd × z
    
    Returns variance (σ²).
    """
    if params is None:
        params = UncertaintyParams()
    
    # Optical depth (dimensionless)
    # At optical_depth = 2.3, 90% of light is absorbed
    optical_depth = np.abs(kd_map * depth_map)
    
    # Linear growth with optical depth
    sigma_optical = params.optical_sigma_shallow + params.optical_sigma_scale * optical_depth
    
    return (sigma_optical ** 2).astype(np.float32)


def compute_depth_variance(
    depth_map: np.ndarray,
    params: UncertaintyParams = None,
) -> np.ndarray:
    """
    Compute depth-dependent uncertainty variance (σ²_depth).
    
    Uncertainty grows linearly with depth due to accumulated
    error in the water column model.
    
    Returns variance (σ²).
    """
    if params is None:
        params = UncertaintyParams()
    
    depth_abs = np.abs(depth_map)
    sigma_depth = params.depth_sigma_per_meter * depth_abs
    
    return (sigma_depth ** 2).astype(np.float32)


def compute_extrapolation_variance(
    depth_map: np.ndarray,
    feature_arrays: Dict[str, np.ndarray],
    training_bounds: Dict[str, Dict[str, float]],
    max_training_depth: float = None,
    params: UncertaintyParams = None,
) -> np.ndarray:
    """
    Compute extrapolation uncertainty variance (σ²_extrap).
    
    Adds uncertainty when:
    1. Depth exceeds training range
    2. Features are outside Domain of Applicability
    
    Returns variance (σ²).
    """
    if params is None:
        params = UncertaintyParams()
    
    shape = depth_map.shape
    var_extrap = np.zeros(shape, dtype=np.float32)
    
    # Depth extrapolation
    if max_training_depth is not None:
        depth_beyond = np.maximum(np.abs(depth_map) - max_training_depth, 0)
        sigma_depth_extrap = params.extrap_sigma_per_meter_beyond * depth_beyond
        var_extrap += sigma_depth_extrap ** 2
    
    # Feature extrapolation (DoA)
    if training_bounds and feature_arrays:
        for feat, bounds in training_bounds.items():
            if feat in feature_arrays:
                vals = feature_arrays[feat]
                outside_low = vals < bounds.get("min", -np.inf)
                outside_high = vals > bounds.get("max", np.inf)
                outside = outside_low | outside_high
                # Add variance for pixels outside bounds
                var_extrap = np.where(
                    outside,
                    var_extrap + params.extrap_sigma_feature_outside ** 2,
                    var_extrap
                )
    
    return var_extrap.astype(np.float32)


def compute_spatial_variance(
    lon_map: np.ndarray,
    lat_map: np.ndarray,
    training_points: np.ndarray = None,
    params: UncertaintyParams = None,
) -> np.ndarray:
    """
    Compute spatial uncertainty variance (σ²_spatial).
    
    Uncertainty increases with distance from training/calibration data.
    
    Returns variance (σ²).
    """
    if params is None:
        params = UncertaintyParams()
    
    if training_points is None or len(training_points) == 0:
        # No calibration data - return high constant variance
        return np.full(lon_map.shape, params.spatial_sigma_base ** 2 * 9, dtype=np.float32)
    
    from scipy.spatial import cKDTree
    
    # Build tree of calibration points
    tree = cKDTree(training_points)
    
    # Query distance to nearest calibration point for each pixel
    coords = np.column_stack([lon_map.ravel(), lat_map.ravel()])
    
    # Approximate degrees to km
    deg_to_km = 111.0
    distances_deg, _ = tree.query(coords, k=1)
    distances_km = distances_deg.reshape(lon_map.shape) * deg_to_km
    
    # Cap distance effect
    distances_km = np.minimum(distances_km, params.spatial_max_distance_km)
    
    # σ_spatial = base + rate × distance
    sigma_spatial = params.spatial_sigma_base + params.spatial_sigma_per_km * distances_km
    
    return (sigma_spatial ** 2).astype(np.float32)


def compute_input_variance(
    s2_bands: Dict[str, np.ndarray],
    depth_sensitivity: float = 1.0,
    params: UncertaintyParams = None,
) -> np.ndarray:
    """
    Compute input data uncertainty variance (σ²_input).
    
    Propagates Sentinel-2 radiometric uncertainty through the depth model.
    
    Returns variance (σ²).
    """
    if params is None:
        params = UncertaintyParams()
    
    shape = list(s2_bands.values())[0].shape
    
    # Base input uncertainty - propagated through model
    # This accounts for S2 calibration, atmospheric correction, etc.
    sigma_input = params.input_sigma_base * depth_sensitivity
    
    return np.full(shape, sigma_input ** 2, dtype=np.float32)


def combine_variances(*variances) -> np.ndarray:
    """
    Combine independent variance components.
    
    Total variance = sum of individual variances (for independent sources).
    
    Returns total variance (σ²_total).
    """
    total_var = np.zeros_like(variances[0], dtype=np.float32)
    for var in variances:
        if isinstance(var, np.ndarray):
            total_var += var
        elif isinstance(var, (int, float)):
            total_var += float(var)
    return total_var


# -----------------------------------------------------------------------------
# Main prediction with uncertainty
# -----------------------------------------------------------------------------

@dataclass
class SDBPrediction:
    """Result of SDB prediction with uncertainty."""
    depth: np.ndarray
    uncertainty: np.ndarray  # 1-sigma in meters
    confidence: np.ndarray   # 0-1 reliability score
    mode: str               # calibrated, transfer, physics, uncalibrated
    variance_components: Dict[str, np.ndarray]  # Individual variance terms
    metadata: Dict[str, Any]


def predict_sdb_with_uncertainty(
    s2_bands: Dict[str, np.ndarray],
    *,
    mode: str = "physics",
    rf_model: Any = None,
    model_meta: Dict[str, Any] = None,
    kd_map: np.ndarray = None,
    training_points: np.ndarray = None,  # lon, lat of ATL points
    region: str = "default",
    l_inf: Dict[str, float] = None,
    params: UncertaintyParams = None,
    lon_map: np.ndarray = None,
    lat_map: np.ndarray = None,
) -> SDBPrediction:
    """
    Predict bathymetry with comprehensive uncertainty quantification.
    
    UNCERTAINTY COMBINATION:
    All uncertainty sources are computed as VARIANCES (σ²), then combined:
        σ²_total = σ²_model + σ²_optical + σ²_depth + σ²_spatial + σ²_extrap + σ²_input
        σ_total = sqrt(σ²_total)
    
    The output uncertainty represents 1 standard deviation (68% CI).
    For 95% CI, multiply by 1.96.
    
    Parameters
    ----------
    s2_bands : dict
        Sentinel-2 band arrays {"B02": array, "B03": array, ...}
    mode : str
        Prediction mode:
        - "calibrated": Use trained RF model (requires rf_model, model_meta)
        - "transfer": Use model from different region
        - "physics": Use Stumpf/Lyzenga with regional params
        - "uncalibrated": Pure physics with default params
    rf_model : sklearn model, optional
        Trained Random Forest model
    model_meta : dict, optional
        Model metadata including training bounds
    kd_map : np.ndarray, optional
        Per-pixel Kd estimates (from kd_estimation.py)
    training_points : np.ndarray, optional
        Locations of ATL training data (lon, lat)
    region : str
        Regional parameter set for physics modes
    l_inf : dict, optional
        Deep water reflectance per band
    params : UncertaintyParams, optional
        Uncertainty model parameters
        
    Returns
    -------
    SDBPrediction
        Depth, uncertainty (1-sigma), confidence, variance components, and metadata
    """
    if params is None:
        params = UncertaintyParams()
    
    shape = s2_bands["B02"].shape
    regional_params = get_regional_params(region)
    
    # =========================================================================
    # STEP 1: DEPTH PREDICTION
    # =========================================================================
    
    rf_tree_std = None  # Will hold per-pixel std from RF trees if available
    max_training_depth = None
    training_bounds = {}
    
    if mode == "calibrated" and rf_model is not None:
        depth, rf_tree_std = _predict_calibrated_with_tree_variance(
            s2_bands, rf_model, model_meta, l_inf
        )
        max_training_depth = model_meta.get("max_depth_sdb") if model_meta else None
        training_bounds = model_meta.get("training_bounds", {}) if model_meta else {}
        
    elif mode == "transfer" and rf_model is not None:
        depth, rf_tree_std = _predict_calibrated_with_tree_variance(
            s2_bands, rf_model, model_meta, l_inf
        )
        max_training_depth = model_meta.get("max_depth_sdb") if model_meta else None
        training_bounds = model_meta.get("training_bounds", {}) if model_meta else {}
        
    elif mode == "physics":
        stumpf_params = regional_params["stumpf"]
        depth = stumpf_ratio_model(
            s2_bands["B02"], s2_bands["B03"],
            **stumpf_params
        )
        max_training_depth = regional_params["max_reliable_depth"]
        
    else:  # uncalibrated
        depth = stumpf_ratio_model(
            s2_bands["B02"], s2_bands["B03"],
            m0=0.0, m1=5.0, n=1000
        )
        max_training_depth = 15.0
    
    # Clip negative depths
    depth = np.maximum(depth, 0.0)
    
    # =========================================================================
    # STEP 2: COMPUTE VARIANCE COMPONENTS
    # =========================================================================
    
    variance_components = {}
    
    # --- 2a. Model variance (σ²_model) ---
    var_model = compute_model_variance(mode, rf_tree_std, params)
    if isinstance(var_model, (int, float)):
        var_model = np.full(shape, var_model, dtype=np.float32)
    variance_components["model"] = var_model
    
    # --- 2b. Optical variance (σ²_optical) ---
    if kd_map is not None:
        var_optical = compute_optical_variance(kd_map, depth, params)
    else:
        # Use regional typical Kd
        kd_typical = regional_params.get("kd_490_typical", 0.15)
        kd_est = np.full(shape, kd_typical, dtype=np.float32)
        var_optical = compute_optical_variance(kd_est, depth, params)
    variance_components["optical"] = var_optical
    
    # --- 2c. Depth-dependent variance (σ²_depth) ---
    var_depth = compute_depth_variance(depth, params)
    variance_components["depth"] = var_depth
    
    # --- 2d. Extrapolation variance (σ²_extrap) ---
    # Build feature arrays for DoA check
    feature_arrays = {}
    if mode in ("calibrated", "transfer"):
        for band in ["B02", "B03", "B04"]:
            if band in s2_bands:
                feature_arrays[band] = s2_bands[band]
        # Add derived features if bounds exist
        if "log_ratio_B02_B03" in training_bounds and "B02" in s2_bands and "B03" in s2_bands:
            eps = 1e-6
            feature_arrays["log_ratio_B02_B03"] = np.log(np.maximum(s2_bands["B02"], eps)) - \
                                                   np.log(np.maximum(s2_bands["B03"], eps))
    
    var_extrap = compute_extrapolation_variance(
        depth, feature_arrays, training_bounds, max_training_depth, params
    )
    variance_components["extrapolation"] = var_extrap
    
    # --- 2e. Spatial variance (σ²_spatial) ---
    if lon_map is not None and lat_map is not None:
        var_spatial = compute_spatial_variance(
            lon_map, lat_map, training_points, params
        )
    elif training_points is None or len(training_points) == 0:
        # No training data and no coordinate maps - high spatial uncertainty
        var_spatial = np.full(shape, (params.spatial_sigma_base * 3) ** 2, dtype=np.float32)
    else:
        var_spatial = np.full(shape, params.spatial_sigma_base ** 2, dtype=np.float32)
    variance_components["spatial"] = var_spatial
    
    # --- 2f. Input data variance (σ²_input) ---
    var_input = compute_input_variance(s2_bands, depth_sensitivity=1.0, params=params)
    variance_components["input"] = var_input
    
    # =========================================================================
    # STEP 3: COMBINE VARIANCES
    # =========================================================================
    
    # Total variance = sum of independent variance components
    total_variance = combine_variances(
        var_model,
        var_optical,
        var_depth,
        var_extrap,
        var_spatial,
        var_input
    )
    
    # Convert to standard deviation (1-sigma uncertainty)
    uncertainty = np.sqrt(total_variance)
    
    # Apply bounds
    uncertainty = np.clip(uncertainty, params.min_uncertainty_m, params.max_uncertainty_m)
    uncertainty = uncertainty.astype(np.float32)
    
    # =========================================================================
    # STEP 4: COMPUTE CONFIDENCE
    # =========================================================================
    
    # Confidence based on uncertainty relative to depth
    # confidence = 1 / (1 + uncertainty/scale)
    # Scale chosen so uncertainty = 0.5m gives confidence ≈ 0.8
    confidence_scale = 0.3
    confidence = 1.0 / (1.0 + uncertainty / (depth + 0.5) * confidence_scale)
    
    # Additional penalties
    if kd_map is not None:
        optical_depth = kd_map * depth
        # Low confidence where optical depth > 3 (95% attenuation)
        confidence = np.where(optical_depth > 3.0, confidence * 0.5, confidence)
        confidence = np.where(optical_depth > 4.0, confidence * 0.3, confidence)
    
    # Very low confidence where extrapolation is extreme
    if max_training_depth is not None:
        extrap_ratio = depth / max(max_training_depth, 1.0)
        confidence = np.where(extrap_ratio > 1.5, confidence * 0.3, confidence)
        confidence = np.where(extrap_ratio > 2.0, confidence * 0.1, confidence)
    
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    
    # =========================================================================
    # STEP 5: COMPILE METADATA
    # =========================================================================
    
    # Compute summary statistics for each variance component
    var_summary = {}
    for name, var in variance_components.items():
        sigma = np.sqrt(var)
        var_summary[name] = {
            "sigma_mean": float(np.nanmean(sigma)),
            "sigma_median": float(np.nanmedian(sigma)),
            "sigma_max": float(np.nanmax(sigma)),
            "variance_fraction": float(np.nanmean(var) / np.nanmean(total_variance)) if np.nanmean(total_variance) > 0 else 0,
        }
    
    metadata = {
        "mode": mode,
        "region": region,
        "max_training_depth": max_training_depth,
        "depth_stats": {
            "min": float(np.nanmin(depth)),
            "max": float(np.nanmax(depth)),
            "mean": float(np.nanmean(depth)),
            "median": float(np.nanmedian(depth)),
        },
        "uncertainty_stats": {
            "min": float(np.nanmin(uncertainty)),
            "max": float(np.nanmax(uncertainty)),
            "mean": float(np.nanmean(uncertainty)),
            "median": float(np.nanmedian(uncertainty)),
            "p90": float(np.nanpercentile(uncertainty, 90)),
        },
        "confidence_stats": {
            "mean": float(np.nanmean(confidence)),
            "high_confidence_fraction": float(np.mean(confidence > 0.7)),
            "low_confidence_fraction": float(np.mean(confidence < 0.3)),
        },
        "variance_components": var_summary,
        "has_kd_map": kd_map is not None,
        "has_training_points": training_points is not None and len(training_points) > 0,
        "has_rf_tree_variance": rf_tree_std is not None,
    }
    
    log.info(
        f"[SDB] Prediction complete: mode={mode}, "
        f"depth={metadata['depth_stats']['mean']:.1f}m (mean), "
        f"uncertainty={metadata['uncertainty_stats']['median']:.2f}m (median 1-sigma)"
    )
    
    # Log dominant uncertainty sources
    sorted_var = sorted(var_summary.items(), key=lambda x: x[1]["variance_fraction"], reverse=True)
    top_sources = ", ".join([f"{k}:{v['variance_fraction']*100:.0f}%" for k, v in sorted_var[:3]])
    log.info("[SDB] Dominant uncertainty sources: %s", top_sources)
    
    return SDBPrediction(
        depth=depth.astype(np.float32),
        uncertainty=uncertainty,
        confidence=confidence,
        mode=mode,
        variance_components=variance_components,
        metadata=metadata,
    )


def _predict_calibrated_with_tree_variance(
    s2_bands: Dict[str, np.ndarray],
    rf_model: Any,
    model_meta: Dict[str, Any],
    l_inf: Dict[str, float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict using calibrated RF model, returning depth and tree variance.
    """
    
    shape = s2_bands["B02"].shape
    
    # Get feature columns from metadata
    feat_cols = model_meta.get("feature_columns", ["B02", "B03", "B04", "log_ratio_B02_B03"])
    
    # Apply L_inf correction if available
    if l_inf is None:
        l_inf = model_meta.get("l_inf_constants", model_meta.get("linf", {})) or {}
    
    eps = 1e-6
    
    # Build feature array
    features = {}
    for band in ["B02", "B03", "B04", "B08"]:
        if band in s2_bands:
            corrected = np.maximum(s2_bands[band] - l_inf.get(band, 0.0), eps)
            features[band] = corrected
            features[f"log_{band}"] = np.log(corrected)
    
    # Add derived features
    if "B02" in features and "B03" in features:
        features["log_ratio_B02_B03"] = features["log_B02"] - features["log_B03"]
        # Stumpf index
        n = 1000
        log_b02 = np.log(n * features["B02"])
        log_b03 = np.log(n * features["B03"])
        log_b03_safe = np.where(np.abs(log_b03) < eps, eps, log_b03)
        features["stumpf_index"] = log_b02 / log_b03_safe
    
    # Stack features in correct order
    X = np.column_stack([
        features.get(col, np.zeros(shape)).ravel()
        for col in feat_cols if col in features
    ])
    
    # Handle missing features
    if X.shape[1] < len(feat_cols):
        log.warning(f"[SDB] Missing some features, prediction may be less accurate")
    
    # Predict
    depth = rf_model.predict(X).reshape(shape).astype(np.float32)
    
    # Estimate uncertainty from tree variance
    if hasattr(rf_model, 'estimators_'):
        # Get predictions from individual trees
        tree_preds = np.array([
            tree.predict(X).reshape(shape)
            for tree in rf_model.estimators_[:20]  # Sample first 20 trees for speed
        ])
        ml_uncertainty = np.std(tree_preds, axis=0).astype(np.float32)
    else:
        ml_uncertainty = np.full(shape, 0.2, dtype=np.float32)
    
    return depth, ml_uncertainty


# -----------------------------------------------------------------------------
# Output generation
# -----------------------------------------------------------------------------

def write_uncertainty_rasters(
    prediction: SDBPrediction,
    output_dir: Path,
    profile: Dict[str, Any],
    prefix: str = "SDB",
) -> Dict[str, Path]:
    """
    Write depth, uncertainty, and confidence rasters.
    
    Parameters
    ----------
    prediction : SDBPrediction
        Prediction results
    output_dir : Path
        Output directory
    profile : dict
        Rasterio profile for output
    prefix : str
        Filename prefix
        
    Returns
    -------
    dict
        Paths to output files
    """
    import rasterio
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # Update profile for single-band float32
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=np.nan)
    
    # Depth raster
    depth_path = output_dir / f"{prefix}_Depth_{prediction.mode}.tif"
    with rasterio.open(depth_path, "w", **out_profile) as dst:
        dst.write(prediction.depth, 1)
        dst.set_band_description(1, "depth_m")
    outputs["depth"] = depth_path
    
    # Uncertainty raster
    unc_path = output_dir / f"{prefix}_Uncertainty_{prediction.mode}.tif"
    with rasterio.open(unc_path, "w", **out_profile) as dst:
        dst.write(prediction.uncertainty, 1)
        dst.set_band_description(1, "uncertainty_1sigma_m")
    outputs["uncertainty"] = unc_path
    
    # Confidence raster
    conf_path = output_dir / f"{prefix}_Confidence_{prediction.mode}.tif"
    with rasterio.open(conf_path, "w", **out_profile) as dst:
        dst.write(prediction.confidence, 1)
        dst.set_band_description(1, "confidence_0_1")
    outputs["confidence"] = conf_path
    
    # Combined multi-band raster
    combined_profile = out_profile.copy()
    combined_profile.update(count=3)
    combined_path = output_dir / f"{prefix}_Combined_{prediction.mode}.tif"
    with rasterio.open(combined_path, "w", **combined_profile) as dst:
        dst.write(prediction.depth, 1)
        dst.write(prediction.uncertainty, 2)
        dst.write(prediction.confidence, 3)
        dst.set_band_description(1, "depth_m")
        dst.set_band_description(2, "uncertainty_1sigma_m")
        dst.set_band_description(3, "confidence_0_1")
    outputs["combined"] = combined_path
    
    log.info("[SDB] Wrote uncertainty-aware outputs to %s", output_dir)
    
    return outputs


# -----------------------------------------------------------------------------
# Convenience function for physics-only prediction
# -----------------------------------------------------------------------------

def predict_physics_only(
    s2_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    region: str = "default",
    allow_heuristic_region: bool = False,
    kd_algorithm: str = "lee2005",
) -> SDBPrediction:
    """
    Run physics-only SDB prediction with uncertainty.
    
    This is the simplest way to get bathymetry without any training data.
    
    Parameters
    ----------
    s2_dir : Path
        Directory with S2 band GeoTIFFs (B02_10m.tif, B03_10m.tif, etc.)
    output_dir : Path
        Output directory
    region : str
        Regional parameter set. Use "auto" only if allow_heuristic_region=True.
    kd_algorithm : str
        Kd estimation algorithm
        
    Returns
    -------
    SDBPrediction
        Prediction results (also written to output_dir)
    """
    import rasterio
    
    s2_dir = Path(s2_dir)
    output_dir = Path(output_dir)
    
    # Load bands
    bands = {}
    profile = None
    for band in ["B02", "B03", "B04", "B08"]:
        path = s2_dir / f"{band}_10m.tif"
        if path.exists():
            with rasterio.open(path) as ds:
                bands[band] = ds.read(1).astype(np.float32)
                if profile is None:
                    profile = ds.profile.copy()
                    bounds = ds.bounds
                    
    if "B02" not in bands or "B03" not in bands:
        raise ValueError("B02 and B03 bands are required")
    
    # Auto-detect region
    if region == "auto" and not allow_heuristic_region:
        raise ValueError(            "region='auto' relies on very rough geographic heuristics. "            "To use it, pass allow_heuristic_region=True (and record this in your provenance). "            "Otherwise choose an explicit region (e.g., region='default')."        )

    if region == "auto":
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.bottom + bounds.top) / 2
        region = estimate_region_from_location(center_lon, center_lat)
        log.info("[SDB] Auto-detected region: %s", region)
    
    # Estimate Kd
    kd_map = None
    try:
        from kd_estimation import estimate_kd_from_reflectance
        kd_map = estimate_kd_from_reflectance(
            bands["B02"], bands["B03"], bands.get("B04", bands["B03"]),
            algorithm=kd_algorithm
        )
    except ImportError:
        log.warning("[SDB] kd_estimation module not available, using regional typical Kd")
    
    # Predict
    prediction = predict_sdb_with_uncertainty(
        bands,
        mode="physics",
        kd_map=kd_map,
        region=region,
    )
    
    # Write outputs
    write_uncertainty_rasters(prediction, output_dir, profile)
    
    # Save metadata
    import json
    meta_path = output_dir / "sdb_physics_meta.json"
    with open(meta_path, "w") as f:
        json.dump(prediction.metadata, f, indent=2)
    
    return prediction


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import argparse
    
    parser = argparse.ArgumentParser(description="Physics-based SDB with uncertainty")
    parser.add_argument("--s2-dir", required=True, help="Sentinel-2 band directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--region",
        default="default",
        choices=list(REGIONAL_PARAMS.keys()) + ["auto"],
        help="Regional parameter set. Use region='auto' only with --allow-heuristic-region.",
    )
    parser.add_argument(
        "--allow-heuristic-region",
        action="store_true",
        help="Allow region='auto' (uses very rough geographic heuristics; not scientifically defensible without disclosure).",
    )
    parser.add_argument("--mode", default="physics",
                       choices=["physics", "uncalibrated"])
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    prediction = predict_physics_only(
        args.s2_dir,
        args.output_dir,
        region=args.region,
        allow_heuristic_region=bool(args.allow_heuristic_region),
    )
    
    log.info(f"\nResults saved to {args.output_dir}")
    log.info(f"Mode: {prediction.mode}")
    log.info(f"Depth range: {prediction.metadata['depth_stats']['min']:.1f} - {prediction.metadata['depth_stats']['max']:.1f} m")
    log.info(f"Mean uncertainty: {prediction.metadata['uncertainty_stats']['mean']:.2f} m")
    log.info(f"High confidence fraction: {prediction.metadata['confidence_stats']['high_confidence_fraction']*100:.1f}%")
