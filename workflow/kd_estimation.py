#!/usr/bin/env python3
"""
kd_estimation.py - Physics-based water clarity and depth penetration estimation

This module estimates the diffuse attenuation coefficient (Kd) from Sentinel-2 imagery
to determine the maximum reliable depth for satellite-derived bathymetry (SDB).

Theory:
-------
Light attenuates exponentially with depth according to Beer-Lambert law:
    L(z) = L(0) * exp(-Kd * z)

Where:
    L(z) = radiance at depth z
    L(0) = surface radiance  
    Kd = diffuse attenuation coefficient (m⁻¹)
    z = depth (m)

The maximum depth at which bottom reflectance contributes meaningfully to the
surface-leaving signal depends on:
    1. Water clarity (Kd) - lower Kd = clearer water = deeper penetration
    2. Bottom albedo - brighter bottoms are visible deeper
    3. Sensor sensitivity - ability to detect small signals

Empirical relationships:
    Secchi depth ≈ 1.7 / Kd(490)  (classic relationship)
    Max SDB depth ≈ 2.0-3.0 * Secchi depth (depending on bottom type)
    
For conservative SDB estimates:
    Max reliable depth ≈ 1.5 / Kd(blue)  (90% light attenuation)
    Max possible depth ≈ 2.5 / Kd(blue)  (optimistic, bright sand bottom)

References:
-----------
- Lee et al. (2005) - Kd estimation from remote sensing
- Stumpf et al. (2003) - SDB depth limits
- Lyzenga (1978, 1981) - Optical bathymetry theory
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
import numpy as np

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Physical constants and empirical coefficients
# -----------------------------------------------------------------------------

# Sentinel-2 band center wavelengths (nm)
S2_WAVELENGTHS = {
    "B02": 490,   # Blue
    "B03": 560,   # Green
    "B04": 665,   # Red
    "B08": 842,   # NIR
}

# Pure water absorption coefficients (m⁻¹) at S2 wavelengths
# From Pope & Fry (1997), Smith & Baker (1981)
PURE_WATER_ABSORPTION = {
    "B02": 0.0145,   # 490 nm
    "B03": 0.0596,   # 560 nm  
    "B04": 0.429,    # 665 nm
    "B08": 2.87,     # 842 nm (NIR strongly absorbed)
}

# Empirical Kd-to-depth conversion factors
# Conservative: 90% light attenuation (1/Kd * ln(10) ≈ 2.3/Kd)
# Moderate: 95% attenuation (≈ 3.0/Kd)  
# Optimistic: 99% attenuation with bright bottom (≈ 4.6/Kd)
DEPTH_FACTOR_CONSERVATIVE = 1.5  # Safe operational limit
DEPTH_FACTOR_MODERATE = 2.3      # Typical SDB limit
DEPTH_FACTOR_OPTIMISTIC = 3.5    # Clear water, bright sand

# Water type classifications based on Kd(490)
WATER_TYPES = {
    "oceanic_clear": (0.02, 0.06),      # Open ocean, oligotrophic
    "oceanic_moderate": (0.06, 0.12),   # Coastal ocean
    "coastal_clear": (0.12, 0.20),      # Clear coastal
    "coastal_moderate": (0.20, 0.35),   # Typical coastal
    "coastal_turbid": (0.35, 0.60),     # Turbid coastal
    "estuarine": (0.60, 1.50),          # Estuaries, rivers
    "highly_turbid": (1.50, 5.00),      # Very turbid
}


# -----------------------------------------------------------------------------
# Kd estimation algorithms
# -----------------------------------------------------------------------------

def estimate_kd_from_reflectance(
    rrs_blue: np.ndarray,
    rrs_green: np.ndarray,
    rrs_red: np.ndarray,
    *,
    algorithm: str = "lee2005",
) -> np.ndarray:
    """
    Estimate Kd(490) from remote sensing reflectance.
    
    Parameters
    ----------
    rrs_blue : np.ndarray
        Remote sensing reflectance in blue band (~490 nm)
    rrs_green : np.ndarray
        Remote sensing reflectance in green band (~560 nm)
    rrs_red : np.ndarray
        Remote sensing reflectance in red band (~665 nm)
    algorithm : str
        Algorithm to use: 'lee2005', 'mueller2000', 'morel1988', 'ratio_empirical'
        
    Returns
    -------
    np.ndarray
        Kd(490) in m⁻¹
    """
    # Ensure float and handle zeros
    eps = 1e-8
    rrs_b = np.maximum(np.asarray(rrs_blue, dtype=np.float64), eps)
    rrs_g = np.maximum(np.asarray(rrs_green, dtype=np.float64), eps)
    rrs_r = np.maximum(np.asarray(rrs_red, dtype=np.float64), eps)
    
    if algorithm == "lee2005":
        # Lee et al. (2005) semi-analytical algorithm
        # Kd(490) = Kw(490) + χ * Chl^e
        # Using blue/green ratio as chlorophyll proxy
        ratio = np.log10(rrs_b / rrs_g)
        # Empirical fit: Kd = a0 + a1*R + a2*R² + a3*R³ + a4*R⁴
        # Coefficients from NASA Ocean Color
        a = [-0.8813, -2.0584, 2.5878, -3.4885, 1.5061]
        log_kd = a[0] + a[1]*ratio + a[2]*ratio**2 + a[3]*ratio**3 + a[4]*ratio**4
        kd = 10**log_kd
        
    elif algorithm == "mueller2000":
        # Mueller (2000) - simpler empirical relationship
        # Kd(490) = 0.0166 + 0.0773 * (Rrs443/Rrs555)^(-1.0)
        # Adapted for S2 bands
        ratio = rrs_b / rrs_g
        kd = 0.0166 + 0.0773 * (ratio ** -1.0)
        
    elif algorithm == "morel1988":
        # Morel (1988) Case 1 waters - chlorophyll-based
        # First estimate Chl from blue/green ratio
        ratio = np.log10(rrs_b / rrs_g)
        log_chl = 0.283 - 2.753*ratio + 1.457*ratio**2 + 0.659*ratio**3 - 1.403*ratio**4
        chl = 10**log_chl
        # Then Kd from Chl
        kd = 0.0166 + 0.0395 * chl**0.79
        
    elif algorithm == "ratio_empirical":
        # Simple empirical ratio for coastal waters
        # Higher blue/green ratio = clearer water = lower Kd
        ratio = rrs_b / rrs_g
        # Empirical fit for coastal waters
        kd = 0.05 + 0.15 * np.exp(-2.5 * ratio)
        
    else:
        raise ValueError(f"Unknown Kd algorithm: {algorithm}")
    
    # Clip to physically realistic range
    kd = np.clip(kd, 0.01, 5.0)
    
    return kd.astype(np.float32)


def estimate_kd_from_linf_ratio(
    l_inf: Dict[str, float],
    surface_reflectance: Dict[str, float],
) -> Dict[str, float]:
    """
    Estimate Kd for each band from the ratio of deep water to shallow water reflectance.
    
    This uses the principle that in optically deep water, bottom reflectance = 0,
    so the surface signal is purely water column + atmosphere.
    
    Parameters
    ----------
    l_inf : dict
        Deep water (optically infinite depth) reflectance per band
    surface_reflectance : dict
        Average surface reflectance in shallow water per band
        
    Returns
    -------
    dict
        Estimated Kd per band
    """
    kd = {}
    for band in ["B02", "B03", "B04"]:
        if band in l_inf and band in surface_reflectance:
            linf = float(l_inf[band])
            surf = float(surface_reflectance[band])
            if surf > linf > 0:
                # Assume typical shallow depth ~2m for calibration
                assumed_depth = 2.0
                # Beer-Lambert: surf = linf + (bottom - linf) * exp(-2*Kd*z)
                # Rearranging with bottom_contrib = surf - linf
                bottom_contrib = surf - linf
                if bottom_contrib > 0.001:
                    # Assume ~50% bottom albedo for sand
                    bottom_albedo = 0.5
                    # Solve: bottom_contrib = bottom_albedo * exp(-2*Kd*z)
                    kd[band] = -np.log(bottom_contrib / bottom_albedo) / (2 * assumed_depth)
                    kd[band] = max(0.01, min(5.0, kd[band]))
    
    return kd


def estimate_kd_from_rasters(
    raster_paths: Dict[str, str],
    *,
    algorithm: str = "lee2005",
    deep_water_nir_max: float = 0.02,
    sample_fraction: float = 0.1,
    max_samples: int = 500_000,
) -> Dict[str, Any]:
    """
    Estimate Kd statistics from S2 raster imagery.
    
    Parameters
    ----------
    raster_paths : dict
        Paths to B02, B03, B04, B08 rasters
    algorithm : str
        Kd algorithm to use
    deep_water_nir_max : float
        NIR threshold for identifying optically deep water
    sample_fraction : float
        Fraction of pixels to sample (for speed)
    max_samples : int
        Maximum number of samples
        
    Returns
    -------
    dict
        Kd statistics including median, percentiles, and derived depth limits
    """
    import rasterio
    
    required = ["B02", "B03", "B04", "B08"]
    missing = [b for b in required if b not in raster_paths or not raster_paths[b]]
    if missing:
        raise ValueError(f"Missing required bands for Kd estimation: {missing}")
    
    # Read bands
    with rasterio.open(raster_paths["B02"]) as ds:
        b02 = ds.read(1).astype(np.float32)
        profile = ds.profile
    with rasterio.open(raster_paths["B03"]) as ds:
        b03 = ds.read(1).astype(np.float32)
    with rasterio.open(raster_paths["B04"]) as ds:
        b04 = ds.read(1).astype(np.float32)
    with rasterio.open(raster_paths["B08"]) as ds:
        b08 = ds.read(1).astype(np.float32)
    
    # Identify water pixels (low NIR, finite values)
    water_mask = (
        np.isfinite(b02) & np.isfinite(b03) & np.isfinite(b04) & np.isfinite(b08) &
        (b08 < deep_water_nir_max * 3) &  # Allow some shallow water
        (b02 > 0.001) & (b03 > 0.001) & (b04 > 0.001)  # Valid reflectance
    )
    
    n_water = np.count_nonzero(water_mask)
    if n_water < 100:
        log.warning(f"[Kd] Insufficient water pixels ({n_water}) for Kd estimation")
        return {"error": "insufficient_water_pixels", "n_water": n_water}
    
    # Sample pixels for speed
    water_indices = np.where(water_mask.ravel())[0]
    n_sample = min(max_samples, int(n_water * sample_fraction))
    if n_sample < n_water:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(water_indices, size=n_sample, replace=False)
    else:
        sample_idx = water_indices
    
    # Extract sampled values
    flat_shape = b02.shape[0] * b02.shape[1]
    rrs_b = b02.ravel()[sample_idx]
    rrs_g = b03.ravel()[sample_idx]
    rrs_r = b04.ravel()[sample_idx]
    nir = b08.ravel()[sample_idx]
    
    # Estimate Kd
    kd_values = estimate_kd_from_reflectance(rrs_b, rrs_g, rrs_r, algorithm=algorithm)
    
    # Separate deep water (for L∞) and all water
    deep_mask = nir < deep_water_nir_max
    
    # Calculate statistics
    kd_all = kd_values[np.isfinite(kd_values)]
    kd_deep = kd_values[deep_mask & np.isfinite(kd_values)] if deep_mask.any() else kd_all
    
    if kd_all.size == 0:
        log.warning("[Kd] No valid Kd values computed")
        return {"error": "no_valid_kd"}
    
    # Compute percentiles
    pct = [5, 10, 25, 50, 75, 90, 95]
    kd_percentiles = {f"p{p}": float(np.percentile(kd_all, p)) for p in pct}
    
    # Use median Kd for depth limits (robust to outliers)
    kd_median = float(np.median(kd_all))
    kd_p10 = float(np.percentile(kd_all, 10))  # Clearest 10%
    kd_p90 = float(np.percentile(kd_all, 90))  # Turbid 10%
    
    # Classify water type
    water_type = classify_water_type(kd_median)
    
    # Calculate depth limits
    depth_limits = calculate_depth_limits(kd_median, kd_p10, kd_p90)
    
    # Deep water statistics for L∞ estimation
    linf_stats = {}
    if deep_mask.any() and deep_mask.sum() > 50:
        for band, arr in [("B02", rrs_b), ("B03", rrs_g), ("B04", rrs_r)]:
            deep_vals = arr[deep_mask]
            if deep_vals.size > 0:
                linf_stats[band] = {
                    "p01": float(np.percentile(deep_vals, 1)),
                    "p05": float(np.percentile(deep_vals, 5)),
                    "median": float(np.median(deep_vals)),
                }
    
    result = {
        "algorithm": algorithm,
        "n_water_pixels": int(n_water),
        "n_sampled": int(len(sample_idx)),
        "n_deep_water": int(deep_mask.sum()) if deep_mask.any() else 0,
        "kd_490": {
            "mean": float(np.mean(kd_all)),
            "median": kd_median,
            "std": float(np.std(kd_all)),
            "percentiles": kd_percentiles,
        },
        "water_type": water_type,
        "depth_limits_m": depth_limits,
        "linf_deep_water": linf_stats,
    }
    
    log.info(
        f"[Kd] Estimated Kd(490)={kd_median:.3f} m⁻¹ ({water_type}), "
        f"max depth: {depth_limits['conservative']:.1f}-{depth_limits['optimistic']:.1f} m"
    )
    
    return result


def classify_water_type(kd_490: float) -> str:
    """Classify water type based on Kd(490)."""
    for wtype, (lo, hi) in WATER_TYPES.items():
        if lo <= kd_490 < hi:
            return wtype
    if kd_490 < 0.02:
        return "ultra_clear"
    return "highly_turbid"


def calculate_depth_limits(
    kd_median: float,
    kd_clear: Optional[float] = None,
    kd_turbid: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate SDB depth limits from Kd estimates.
    
    Parameters
    ----------
    kd_median : float
        Median Kd(490) for the scene
    kd_clear : float, optional
        Kd for clearest water (e.g., p10)
    kd_turbid : float, optional
        Kd for most turbid water (e.g., p90)
        
    Returns
    -------
    dict
        Depth limits for different confidence levels
    """
    eps = 0.01  # Minimum Kd to avoid division issues
    kd = max(kd_median, eps)
    
    limits = {
        "conservative": DEPTH_FACTOR_CONSERVATIVE / kd,
        "moderate": DEPTH_FACTOR_MODERATE / kd,
        "optimistic": DEPTH_FACTOR_OPTIMISTIC / kd,
    }
    
    # Add limits based on water clarity variation if available
    if kd_clear is not None:
        kd_c = max(kd_clear, eps)
        limits["clear_water_max"] = DEPTH_FACTOR_OPTIMISTIC / kd_c
        
    if kd_turbid is not None:
        kd_t = max(kd_turbid, eps)
        limits["turbid_water_max"] = DEPTH_FACTOR_CONSERVATIVE / kd_t
    
    # Cap at physically reasonable limits
    for key in limits:
        limits[key] = min(limits[key], 50.0)  # Max 50m even in clearest water
        limits[key] = round(limits[key], 1)
    
    return limits


def estimate_pixel_depth_confidence(
    kd_map: np.ndarray,
    depth_map: np.ndarray,
    *,
    confidence_model: str = "exponential",
) -> np.ndarray:
    """
    Estimate per-pixel confidence based on depth relative to optical limit.
    
    Parameters
    ----------
    kd_map : np.ndarray
        Per-pixel Kd estimates
    depth_map : np.ndarray
        Predicted depth values
    confidence_model : str
        'exponential' or 'linear'
        
    Returns
    -------
    np.ndarray
        Confidence values 0-1 (1 = high confidence, 0 = beyond optical limit)
    """
    # Maximum depth where we have any confidence
    max_depth_per_pixel = DEPTH_FACTOR_MODERATE / np.maximum(kd_map, 0.01)
    
    # Ratio of actual depth to max depth
    depth_ratio = np.abs(depth_map) / np.maximum(max_depth_per_pixel, 0.1)
    
    if confidence_model == "exponential":
        # Exponential decay: full confidence at surface, drops off with depth
        confidence = np.exp(-2 * depth_ratio)
    elif confidence_model == "linear":
        # Linear: 1 at surface, 0 at max depth
        confidence = np.clip(1 - depth_ratio, 0, 1)
    else:
        raise ValueError(f"Unknown confidence model: {confidence_model}")
    
    # Zero confidence where depth exceeds optical limit
    confidence = np.where(depth_ratio > 1.5, 0, confidence)
    
    return confidence.astype(np.float32)


# -----------------------------------------------------------------------------
# Integration with training pipeline
# -----------------------------------------------------------------------------

def compute_physics_based_max_depth(
    raster_paths: Dict[str, str],
    training_df: Optional[Any] = None,
    *,
    rmse_based_max: Optional[float] = None,
    algorithm: str = "lee2005",
    confidence_level: str = "moderate",
) -> Dict[str, Any]:
    """
    Compute physics-based maximum SDB depth, optionally combining with RMSE-based estimate.
    
    This is the main entry point for integrating Kd-based depth limits into the pipeline.
    
    Parameters
    ----------
    raster_paths : dict
        Paths to S2 band rasters
    training_df : DataFrame, optional
        Training data for additional validation
    rmse_based_max : float, optional
        RMSE-based max depth from binning analysis
    algorithm : str
        Kd estimation algorithm
    confidence_level : str
        'conservative', 'moderate', or 'optimistic'
        
    Returns
    -------
    dict
        Combined depth limit recommendation with diagnostics
    """
    result = {
        "method": "physics_based_kd",
        "algorithm": algorithm,
        "confidence_level": confidence_level,
    }
    
    try:
        kd_stats = estimate_kd_from_rasters(raster_paths, algorithm=algorithm)
        
        if "error" in kd_stats:
            log.warning(f"[Kd] Estimation failed: {kd_stats['error']}")
            result["error"] = kd_stats["error"]
            result["physics_max_depth_m"] = None
        else:
            depth_limits = kd_stats["depth_limits_m"]
            physics_max = depth_limits.get(confidence_level, depth_limits["moderate"])
            
            result.update({
                "kd_490_median": kd_stats["kd_490"]["median"],
                "kd_490_p10": kd_stats["kd_490"]["percentiles"]["p10"],
                "kd_490_p90": kd_stats["kd_490"]["percentiles"]["p90"],
                "water_type": kd_stats["water_type"],
                "physics_max_depth_m": physics_max,
                "depth_limits": depth_limits,
                "linf_deep_water": kd_stats.get("linf_deep_water", {}),
            })
            
    except Exception as e:
        log.warning(f"[Kd] Physics-based estimation failed: {e}")
        result["error"] = str(e)
        result["physics_max_depth_m"] = None
    
    # Combine with RMSE-based estimate if available
    if rmse_based_max is not None and result.get("physics_max_depth_m") is not None:
        physics_max = result["physics_max_depth_m"]
        
        # Use the more conservative of the two
        combined_max = min(physics_max, rmse_based_max)
        
        result["rmse_based_max_depth_m"] = rmse_based_max
        result["combined_max_depth_m"] = combined_max
        result["limiting_factor"] = "physics" if physics_max < rmse_based_max else "rmse"
        
        log.info(
            f"[Kd] Combined depth limit: {combined_max:.1f} m "
            f"(physics={physics_max:.1f} m, rmse={rmse_based_max:.1f} m, "
            f"limited by {result['limiting_factor']})"
        )
    elif rmse_based_max is not None:
        result["rmse_based_max_depth_m"] = rmse_based_max
        result["combined_max_depth_m"] = rmse_based_max
        result["limiting_factor"] = "rmse_only"
    elif result.get("physics_max_depth_m") is not None:
        result["combined_max_depth_m"] = result["physics_max_depth_m"]
        result["limiting_factor"] = "physics_only"
    else:
        result["combined_max_depth_m"] = None
        result["limiting_factor"] = "unknown"
    
    return result


def generate_kd_qc_layer(
    raster_paths: Dict[str, str],
    output_path: Union[str, Path],
    depth_raster_path: Optional[Union[str, Path]] = None,
    *,
    algorithm: str = "lee2005",
) -> str:
    """
    Generate a Kd-based quality control layer showing optical depth confidence.
    
    Parameters
    ----------
    raster_paths : dict
        Paths to S2 band rasters
    output_path : str or Path
        Output path for the QC raster
    depth_raster_path : str or Path, optional
        Path to depth prediction raster (if available)
    algorithm : str
        Kd estimation algorithm
        
    Returns
    -------
    str
        Path to generated QC raster
    """
    import rasterio
    from rasterio.transform import from_bounds
    
    output_path = Path(output_path)
    
    # Read bands
    with rasterio.open(raster_paths["B02"]) as ds:
        b02 = ds.read(1).astype(np.float32)
        profile = ds.profile.copy()
        bounds = ds.bounds
        crs = ds.crs
    with rasterio.open(raster_paths["B03"]) as ds:
        b03 = ds.read(1).astype(np.float32)
    with rasterio.open(raster_paths["B04"]) as ds:
        b04 = ds.read(1).astype(np.float32)
    
    # Compute per-pixel Kd
    kd_map = estimate_kd_from_reflectance(b02, b03, b04, algorithm=algorithm)
    
    # Compute max depth per pixel
    max_depth_map = DEPTH_FACTOR_MODERATE / np.maximum(kd_map, 0.01)
    max_depth_map = np.clip(max_depth_map, 0, 50)
    
    # If depth raster provided, compute confidence
    if depth_raster_path is not None and Path(depth_raster_path).exists():
        with rasterio.open(depth_raster_path) as ds:
            depth_map = ds.read(1).astype(np.float32)
        
        confidence = estimate_pixel_depth_confidence(kd_map, depth_map)
        
        # Write 2-band output: max_depth, confidence
        profile.update(count=2, dtype='float32', nodata=np.nan)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(max_depth_map, 1)
            dst.write(confidence, 2)
            dst.set_band_description(1, 'max_optical_depth_m')
            dst.set_band_description(2, 'depth_confidence')
    else:
        # Write single-band max depth
        profile.update(count=1, dtype='float32', nodata=np.nan)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(max_depth_map, 1)
            dst.set_band_description(1, 'max_optical_depth_m')
    
    log.info(f"[Kd] Wrote QC layer: {output_path}")
    return str(output_path)


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Estimate Kd and depth limits from S2 imagery")
    parser.add_argument("--b02", required=True, help="Path to B02 raster")
    parser.add_argument("--b03", required=True, help="Path to B03 raster")
    parser.add_argument("--b04", required=True, help="Path to B04 raster")
    parser.add_argument("--b08", required=True, help="Path to B08 raster")
    parser.add_argument("--algorithm", default="lee2005", choices=["lee2005", "mueller2000", "morel1988", "ratio_empirical"])
    parser.add_argument("--output-json", help="Output JSON file for results")
    parser.add_argument("--output-qc-raster", help="Output Kd/confidence raster")
    parser.add_argument("--depth-raster", help="Depth prediction raster (for confidence calc)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    raster_paths = {
        "B02": args.b02,
        "B03": args.b03,
        "B04": args.b04,
        "B08": args.b08,
    }
    
    result = compute_physics_based_max_depth(raster_paths, algorithm=args.algorithm)
    
    print(json.dumps(result, indent=2))
    
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
    
    if args.output_qc_raster:
        generate_kd_qc_layer(
            raster_paths,
            args.output_qc_raster,
            args.depth_raster,
            algorithm=args.algorithm,
        )
