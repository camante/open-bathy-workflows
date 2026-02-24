#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bottom_physics.py - Physics-Based SDB Enhancements

Implements improvements from Kim et al. (2024):
1. Scene-derived bottom endmember estimation via eigenanalysis
2. Separate Kd (downwelling) and Ku (upwelling) attenuation coefficients
3. Full physics-only prediction mode using radiative transfer

References:
-----------
Kim, M., Danielson, J., Storlazzi, C., & Park, S. (2024). 
Physics-Based Satellite-Derived Bathymetry (SDB) Using Landsat OLI Images.
Remote Sensing, 16(5), 843. https://doi.org/10.3390/rs16050843

Theory:
-------
The optically shallow water reflectance equation (Eq. 19 from Kim et al.):

    r_rs(z) = r_rs_inf × (1 - exp(-(Ku + Kd) × z)) + (ρ_b/π) × exp(-(Ku + Kd) × z)

Where:
    r_rs(z)   = subsurface remote sensing reflectance at depth z
    r_rs_inf  = reflectance from optically deep water (volume scattering)
    Ku        = diffuse attenuation for upwelling radiance
    Kd        = diffuse attenuation for downwelling irradiance
    ρ_b       = bottom reflectance (albedo)
    z         = water depth (m)

The first term represents water column volume scattering (increases with depth).
The second term represents bottom reflection (decreases with depth).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
import numpy as np

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Physical Constants - Import from centralized constants module
# -----------------------------------------------------------------------------

try:
    from constants import (
        N_WATER,
        PURE_WATER_ABSORPTION,
        PURE_WATER_BACKSCATTER,
        S2_WAVELENGTHS,
        GORDON_G1,
        GORDON_G2,
        DEFAULT_SAND_SPECTRUM,
        DEFAULT_SEAGRASS_SPECTRUM,
    )
    # Use GORDON_G1/G2 as G1/G2 for backwards compatibility
    G1 = GORDON_G1
    G2 = GORDON_G2
except ImportError:
    # Fallback to local definitions if constants module unavailable
    log.warning("[bottom_physics] constants module not found, using local definitions")
    
    # Refractive index of seawater (typical)
    # Source: Mobley, C.D. (1994). "Light and Water: Radiative Transfer in 
    #         Natural Waters." Academic Press.
    N_WATER = 1.34

    # Pure water IOPs at S2 wavelengths (m⁻¹)
    # Source: Pope, R.M. & Fry, E.S. (1997). "Absorption spectrum (380–700 nm) 
    #         of pure water. II. Integrating cavity measurements."
    #         Applied Optics, 36(33), 8710-8723.
    # And: Morel, A. (1974). "Optical properties of pure water and pure sea water."
    PURE_WATER_ABSORPTION = {
        "B02": 0.0145,   # 490 nm (Blue)
        "B03": 0.0596,   # 560 nm (Green)
        "B04": 0.429,    # 665 nm (Red)
        "B08": 2.87,     # 842 nm (NIR)
    }

    PURE_WATER_BACKSCATTER = {
        "B02": 0.00144,  # 490 nm
        "B03": 0.00093,  # 560 nm
        "B04": 0.00047,  # 665 nm
        "B08": 0.00017,  # 842 nm
    }

    # Sentinel-2 wavelengths (nm)
    S2_WAVELENGTHS = {
        "B02": 490,
        "B03": 560,
        "B04": 665,
        "B08": 842,
    }

    # Gordon (1975) model coefficients for r_rs to u conversion
    # r_rs = g1*u + g2*u²  where u = bb/(a+bb)
    # Source: Gordon, H.R., et al. (1975). "Computed relationships between the 
    #         inherent and apparent optical properties of a flat homogeneous ocean."
    #         Applied Optics, 14(2), 417-427.
    G1 = 0.0949
    G2 = 0.0794

    # Default bottom library spectra (normalized)
    # Source: Lee Stocking Island measurements from Kim et al. (2024)
    # Kim, M., et al. (2024). "Physics-Based Satellite-Derived Bathymetry (SDB) 
    #                         Using Landsat OLI Images." Remote Sensing, 16(5), 843.
    DEFAULT_SAND_SPECTRUM = {
        "B02": 0.35,  # Blue - moderate
        "B03": 0.42,  # Green - higher  
        "B04": 0.45,  # Red - highest
    }

    DEFAULT_SEAGRASS_SPECTRUM = {
        "B02": 0.05,  # Blue - low (absorbed)
        "B03": 0.12,  # Green - moderate (reflected)
        "B04": 0.04,  # Red - very low (absorbed)
    }


# -----------------------------------------------------------------------------
# Kd/Ku Geometry-Corrected Attenuation
# -----------------------------------------------------------------------------

def compute_kd_ku_from_iops(
    a_total: Union[float, np.ndarray],
    b_b: Union[float, np.ndarray],
    sza_deg: float,
    vza_deg: float = 0.0,
    n_water: float = N_WATER,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute direction-specific diffuse attenuation coefficients.
    
    Based on Kim et al. (2024) Equations 17-18:
        Kd = (a + bb) / cos(SZA_water)
        Ku = (a + bb) / cos(VZA_water)
    
    Parameters
    ----------
    a_total : float or ndarray
        Total absorption coefficient (m⁻¹)
    b_b : float or ndarray
        Total backscattering coefficient (m⁻¹)
    sza_deg : float
        Solar zenith angle in degrees
    vza_deg : float
        View (sensor) zenith angle in degrees (typically small for satellites)
    n_water : float
        Refractive index of water (default 1.34)
        
    Returns
    -------
    Kd : float or ndarray
        Downwelling diffuse attenuation coefficient (m⁻¹)
    Ku : float or ndarray
        Upwelling diffuse attenuation coefficient (m⁻¹)
    """
    # Convert to radians
    sza_rad = np.radians(sza_deg)
    vza_rad = np.radians(vza_deg)
    
    # Refract into water using Snell's law
    # sin(θ_water) = sin(θ_air) / n_water
    sin_sza_water = np.sin(sza_rad) / n_water
    sin_vza_water = np.sin(vza_rad) / n_water
    
    # Clip to valid range for arcsin
    sin_sza_water = np.clip(sin_sza_water, -1.0, 1.0)
    sin_vza_water = np.clip(sin_vza_water, -1.0, 1.0)
    
    sza_water = np.arcsin(sin_sza_water)
    vza_water = np.arcsin(sin_vza_water)
    
    # Avoid division by zero for near-horizontal angles
    cos_sza_water = np.maximum(np.cos(sza_water), 0.1)
    cos_vza_water = np.maximum(np.cos(vza_water), 0.1)
    
    # Total beam attenuation
    c_total = a_total + b_b
    
    # Direction-specific Kd and Ku
    Kd = c_total / cos_sza_water
    Ku = c_total / cos_vza_water
    
    return Kd, Ku


def compute_kd_ku_from_bands(
    kd_scalar: float,
    sza_deg: float,
    vza_deg: float = 0.0,
    wavelength_nm: float = 490.0,
    n_water: float = N_WATER,
) -> Tuple[float, float]:
    """
    Convert scalar Kd to direction-corrected Kd and Ku.
    
    This is useful when you have a Kd estimate (e.g., from Lee2005)
    but want to account for sun/view geometry.
    
    Parameters
    ----------
    kd_scalar : float
        Scalar diffuse attenuation coefficient (m⁻¹), typically at nadir
    sza_deg : float
        Solar zenith angle in degrees
    vza_deg : float
        View zenith angle in degrees
    wavelength_nm : float
        Reference wavelength (nm)
    n_water : float
        Refractive index of water
        
    Returns
    -------
    Kd, Ku : tuple of float
        Direction-corrected attenuation coefficients (m⁻¹)
    """
    # For typical satellite nadir viewing, the scalar Kd approximates
    # the attenuation for near-vertical paths. We need to correct for
    # the actual slant paths.
    
    sza_rad = np.radians(sza_deg)
    vza_rad = np.radians(vza_deg)
    
    # In-water angles
    sin_sza_water = np.sin(sza_rad) / n_water
    sin_vza_water = np.sin(vza_rad) / n_water
    
    sza_water = np.arcsin(np.clip(sin_sza_water, -1, 1))
    vza_water = np.arcsin(np.clip(sin_vza_water, -1, 1))
    
    # The scalar Kd is typically defined for nadir (cos=1)
    # For slant paths, multiply by 1/cos(θ)
    cos_sza_water = np.maximum(np.cos(sza_water), 0.1)
    cos_vza_water = np.maximum(np.cos(vza_water), 0.1)
    
    # Corrected coefficients
    Kd = kd_scalar / cos_sza_water
    Ku = kd_scalar / cos_vza_water
    
    return float(Kd), float(Ku)


def compute_total_path_attenuation(
    Kd: Union[float, np.ndarray],
    Ku: Union[float, np.ndarray],
    depth: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute total two-way optical path attenuation.
    
    The round-trip attenuation is exp(-(Kd + Ku) × z), representing:
    - Downward path: light travels from surface to bottom
    - Upward path: reflected light travels from bottom to surface
    
    Parameters
    ----------
    Kd : float or ndarray
        Downwelling attenuation coefficient (m⁻¹)
    Ku : float or ndarray  
        Upwelling attenuation coefficient (m⁻¹)
    depth : float or ndarray
        Water depth (m)
        
    Returns
    -------
    attenuation : float or ndarray
        Two-way attenuation factor (0-1), where 1 = no attenuation
    """
    optical_depth = (Kd + Ku) * depth
    attenuation = np.exp(-optical_depth)
    return attenuation


# -----------------------------------------------------------------------------
# Bottom Endmember Estimation (Kim et al. Section 3.6)
# -----------------------------------------------------------------------------

def estimate_bottom_reflectance(
    rrs: Dict[str, np.ndarray],
    depth: np.ndarray,
    Kd: Dict[str, float],
    Ku: Dict[str, float],
    rrs_deep: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Invert the shallow water reflectance equation to estimate bottom reflectance.
    
    Based on Kim et al. (2024) Equation 27:
        ρ_b = π × r_rs(z) × exp((Ku + Kd) × z)
    
    This assumes water volume reflectance is negligible in very shallow water.
    
    Parameters
    ----------
    rrs : dict
        Remote sensing reflectance per band (B02, B03, B04)
    depth : ndarray
        Known depth values (m)
    Kd : dict
        Downwelling attenuation per band (m⁻¹)
    Ku : dict
        Upwelling attenuation per band (m⁻¹)
    rrs_deep : dict, optional
        Deep water reflectance per band (for volume scattering correction)
        
    Returns
    -------
    rho_b : dict
        Estimated bottom reflectance per band
    """
    rho_b = {}
    
    for band in ["B02", "B03", "B04"]:
        if band not in rrs or band not in Kd or band not in Ku:
            continue
            
        rrs_band = np.asarray(rrs[band], dtype=np.float64)
        kd = float(Kd[band])
        ku = float(Ku[band])
        z = np.asarray(depth, dtype=np.float64)
        
        # Correct for volume scattering if deep water reference available
        if rrs_deep is not None and band in rrs_deep:
            rrs_vol = float(rrs_deep[band])
            # r_rs = r_rs_deep × (1 - exp(-(Ku+Kd)×z)) + (ρ_b/π) × exp(-(Ku+Kd)×z)
            # Rearranging: ρ_b = π × (r_rs - r_rs_deep × (1 - exp(-(Ku+Kd)×z))) / exp(-(Ku+Kd)×z)
            exp_term = np.exp(-(kd + ku) * z)
            vol_contribution = rrs_vol * (1 - exp_term)
            bottom_signal = rrs_band - vol_contribution
            rho_b[band] = np.pi * bottom_signal / np.maximum(exp_term, 1e-6)
        else:
            # Simplified: assume volume scattering negligible in shallow water
            # ρ_b = π × r_rs(z) × exp((Ku + Kd) × z)
            rho_b[band] = np.pi * rrs_band * np.exp((kd + ku) * z)
        
        # Clip to physically valid range [0, 1]
        rho_b[band] = np.clip(rho_b[band], 0.0, 1.0)
    
    return rho_b


def estimate_bottom_endmembers_eigenanalysis(
    rho_b: Dict[str, np.ndarray],
    valid_mask: Optional[np.ndarray] = None,
    sand_percentile: float = 95.0,
    grass_percentile: float = 5.0,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    Extract sand and grass endmember spectra using eigenanalysis.
    
    Based on Kim et al. (2024) Equations 28-30:
    1. Compute covariance matrix of bottom reflectance spectra
    2. Find first eigenvector (direction of maximum variance)
    3. Project all spectra onto this eigenvector
    4. Use percentiles to identify endmembers
    
    Parameters
    ----------
    rho_b : dict
        Bottom reflectance per band, each as 1D array of N samples
    valid_mask : ndarray, optional
        Boolean mask for valid pixels
    sand_percentile : float
        Percentile for sand endmember (high brightness, default 95)
    grass_percentile : float
        Percentile for grass endmember (low brightness, default 5)
        
    Returns
    -------
    rho_sand : dict
        Sand-like endmember spectrum per band
    rho_grass : dict
        Grass-like endmember spectrum per band  
    diagnostics : dict
        Eigenanalysis diagnostics (eigenvalues, eigenvector, projections)
    """
    bands = [b for b in ["B02", "B03", "B04"] if b in rho_b]
    if len(bands) < 2:
        log.warning("[BottomEndmembers] Need at least 2 bands for eigenanalysis")
        return DEFAULT_SAND_SPECTRUM.copy(), DEFAULT_SEAGRASS_SPECTRUM.copy(), {}
    
    # Stack bands into N×M matrix (N samples, M bands)
    arrays = [np.asarray(rho_b[b]).flatten() for b in bands]
    data = np.column_stack(arrays)
    
    # Apply mask if provided
    if valid_mask is not None:
        mask_flat = np.asarray(valid_mask).flatten()
        # Ensure same length
        if len(mask_flat) == len(data):
            data = data[mask_flat]
    
    # Remove invalid values
    valid = np.all(np.isfinite(data), axis=1) & np.all(data >= 0, axis=1) & np.all(data <= 1, axis=1)
    data = data[valid]
    
    n_samples = len(data)
    if n_samples < 10:
        log.warning(f"[BottomEndmembers] Only {n_samples} valid samples, using defaults")
        return DEFAULT_SAND_SPECTRUM.copy(), DEFAULT_SEAGRASS_SPECTRUM.copy(), {"n_samples": n_samples}
    
    log.info(f"[BottomEndmembers] Eigenanalysis on {n_samples} samples, {len(bands)} bands")
    
    # Compute mean and covariance
    mean_rho = np.mean(data, axis=0)
    centered = data - mean_rho
    cov_matrix = np.cov(centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # First eigenvector (maximum variance direction)
    v1 = eigenvectors[:, 0]
    
    # Ensure v1 points toward "brighter" spectra (positive blue correlation)
    # This ensures consistent sand vs grass assignment
    if v1[0] < 0:  # B02 (blue) component
        v1 = -v1
    
    # Project all samples onto first eigenvector
    projections = centered @ v1
    
    # Use percentiles to find endmembers
    k_sand = np.percentile(projections, sand_percentile)
    k_grass = np.percentile(projections, grass_percentile)
    
    # Reconstruct endmember spectra
    rho_sand_vec = k_sand * v1 + mean_rho
    rho_grass_vec = k_grass * v1 + mean_rho
    
    # Clip to valid range
    rho_sand_vec = np.clip(rho_sand_vec, 0.0, 1.0)
    rho_grass_vec = np.clip(rho_grass_vec, 0.0, 1.0)
    
    # Convert to dict
    rho_sand = {b: float(rho_sand_vec[i]) for i, b in enumerate(bands)}
    rho_grass = {b: float(rho_grass_vec[i]) for i, b in enumerate(bands)}
    
    # Diagnostics
    variance_explained = eigenvalues[0] / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 0
    diagnostics = {
        "n_samples": n_samples,
        "bands": bands,
        "mean_spectrum": {b: float(mean_rho[i]) for i, b in enumerate(bands)},
        "eigenvalues": eigenvalues.tolist(),
        "first_eigenvector": {b: float(v1[i]) for i, b in enumerate(bands)},
        "variance_explained_pc1": float(variance_explained),
        "k_sand": float(k_sand),
        "k_grass": float(k_grass),
        "projection_range": (float(projections.min()), float(projections.max())),
    }
    
    log.info(f"[BottomEndmembers] PC1 explains {variance_explained*100:.1f}% of variance")
    log.info(f"[BottomEndmembers] Sand endmember: {rho_sand}")
    log.info(f"[BottomEndmembers] Grass endmember: {rho_grass}")
    
    return rho_sand, rho_grass, diagnostics


def estimate_bottom_endmembers_from_training(
    training_df,
    s2_dir: Union[str, Path],
    Kd: Dict[str, float],
    Ku: Dict[str, float],
    max_depth_for_endmembers: float = 2.0,
    rrs_deep: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    High-level function to estimate scene-specific bottom endmembers from training data.
    
    Parameters
    ----------
    training_df : DataFrame
        Training data with columns: longitude, latitude, depth_m
    s2_dir : str or Path
        Directory containing S2 band rasters
    Kd : dict
        Downwelling attenuation per band (m⁻¹)
    Ku : dict
        Upwelling attenuation per band (m⁻¹)
    max_depth_for_endmembers : float
        Maximum depth to use for endmember estimation (default 2m)
    rrs_deep : dict, optional
        Deep water reflectance for volume scattering correction
        
    Returns
    -------
    rho_sand : dict
        Sand-like endmember spectrum
    rho_grass : dict
        Grass-like endmember spectrum
    diagnostics : dict
        Analysis diagnostics
    """
    import rasterio
    from pyproj import Transformer
    
    s2_dir = Path(s2_dir)
    
    # Load bands
    band_paths = {
        "B02": s2_dir / "B02_10m.tif",
        "B03": s2_dir / "B03_10m.tif",
        "B04": s2_dir / "B04_10m.tif",
    }
    
    # Check files exist
    for band, path in band_paths.items():
        if not path.exists():
            log.warning(f"[BottomEndmembers] Missing {band}: {path}")
            return DEFAULT_SAND_SPECTRUM.copy(), DEFAULT_SEAGRASS_SPECTRUM.copy(), {"error": "missing_bands"}
    
    # Filter to shallow points
    df = training_df.copy()
    depth_col = "depth_m" if "depth_m" in df.columns else "depth"
    if depth_col not in df.columns:
        log.warning("[BottomEndmembers] No depth column found")
        return DEFAULT_SAND_SPECTRUM.copy(), DEFAULT_SEAGRASS_SPECTRUM.copy(), {"error": "no_depth"}
    
    shallow_mask = df[depth_col] <= max_depth_for_endmembers
    df_shallow = df[shallow_mask].copy()
    
    if len(df_shallow) < 10:
        log.warning(f"[BottomEndmembers] Only {len(df_shallow)} shallow points, using defaults")
        return DEFAULT_SAND_SPECTRUM.copy(), DEFAULT_SEAGRASS_SPECTRUM.copy(), {"n_shallow": len(df_shallow)}
    
    log.info(f"[BottomEndmembers] Using {len(df_shallow)} points with depth <= {max_depth_for_endmembers}m")
    
    # Get coordinates
    lon_col = "longitude" if "longitude" in df_shallow.columns else "lon"
    lat_col = "latitude" if "latitude" in df_shallow.columns else "lat"
    
    lons = df_shallow[lon_col].values
    lats = df_shallow[lat_col].values
    depths = df_shallow[depth_col].values
    
    # Sample reflectance at each point
    rrs_samples = {band: [] for band in ["B02", "B03", "B04"]}
    
    with rasterio.open(band_paths["B02"]) as src:
        crs = src.crs
        transform = src.transform
    
    # Transform coordinates
    if crs.to_epsg() != 4326:
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
    else:
        xs, ys = lons, lats
    
    # Sample each band
    for band in ["B02", "B03", "B04"]:
        with rasterio.open(band_paths[band]) as src:
            data = src.read(1)
            for x, y in zip(xs, ys):
                row, col = ~src.transform * (x, y)
                row, col = int(row), int(col)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = data[row, col]
                    if np.isfinite(val) and val > 0:
                        # Convert to Rrs (assuming reflectance values)
                        rrs = val / np.pi if val > 0.01 else val
                        rrs_samples[band].append(rrs)
                    else:
                        rrs_samples[band].append(np.nan)
                else:
                    rrs_samples[band].append(np.nan)
    
    # Convert to arrays
    rrs = {band: np.array(rrs_samples[band]) for band in ["B02", "B03", "B04"]}
    
    # Find valid samples (all bands have data)
    valid = np.all([np.isfinite(rrs[b]) for b in ["B02", "B03", "B04"]], axis=0)
    
    if np.sum(valid) < 10:
        log.warning(f"[BottomEndmembers] Only {np.sum(valid)} valid samples after filtering")
        return DEFAULT_SAND_SPECTRUM.copy(), DEFAULT_SEAGRASS_SPECTRUM.copy(), {"n_valid": int(np.sum(valid))}
    
    # Filter to valid
    rrs_valid = {b: rrs[b][valid] for b in ["B02", "B03", "B04"]}
    depths_valid = depths[valid]
    
    # Estimate bottom reflectance
    rho_b = estimate_bottom_reflectance(rrs_valid, depths_valid, Kd, Ku, rrs_deep)
    
    # Run eigenanalysis
    rho_sand, rho_grass, diag = estimate_bottom_endmembers_eigenanalysis(rho_b)
    
    diag["n_input_points"] = len(df_shallow)
    diag["n_valid_samples"] = int(np.sum(valid))
    diag["max_depth_used"] = max_depth_for_endmembers
    
    return rho_sand, rho_grass, diag


# -----------------------------------------------------------------------------
# Full Physics-Only SDB Prediction (Kim et al. Algorithm)
# -----------------------------------------------------------------------------

def physics_sdb_inversion(
    rrs: Dict[str, np.ndarray],
    rrs_deep: Dict[str, float],
    Kd: Dict[str, float],
    Ku: Dict[str, float],
    rho_sand: Dict[str, float],
    rho_grass: Dict[str, float],
    max_iterations: int = 20,
    tolerance: float = 0.001,
    max_depth: float = 30.0,
    min_depth: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Full physics-based SDB inversion using Levenberg-Marquardt optimization.
    
    Solves for depth and bottom mixture coefficients from the shallow water
    reflectance equation (Kim et al. Eq. 19):
    
        r_rs(z) = r_rs_inf × (1 - exp(-(Ku+Kd)×z)) + (ρ_b/π) × exp(-(Ku+Kd)×z)
        
    Where ρ_b = C_s × ρ_sand + C_g × ρ_grass
    
    Parameters
    ----------
    rrs : dict
        Remote sensing reflectance per band (2D arrays, same shape)
    rrs_deep : dict
        Deep water reflectance per band (scalars)
    Kd : dict
        Downwelling attenuation per band (m⁻¹)
    Ku : dict  
        Upwelling attenuation per band (m⁻¹)
    rho_sand : dict
        Sand endmember reflectance per band
    rho_grass : dict
        Grass endmember reflectance per band
    max_iterations : int
        Maximum LM iterations
    tolerance : float
        Convergence tolerance
    max_depth : float
        Maximum depth to consider (m)
    min_depth : float
        Minimum depth (m)
        
    Returns
    -------
    depth : ndarray
        Estimated water depth (m)
    sand_fraction : ndarray
        Sand bottom fraction (0-1)
    uncertainty : ndarray
        Depth uncertainty estimate (m)
    diagnostics : dict
        Convergence diagnostics
    """
    bands = [b for b in ["B02", "B03", "B04"] if b in rrs and b in Kd and b in Ku]
    if len(bands) < 2:
        raise ValueError("Need at least 2 bands for physics inversion")
    
    # Get shape from first band
    shape = rrs[bands[0]].shape
    n_pixels = np.prod(shape)
    
    # Flatten all inputs
    rrs_flat = {b: np.asarray(rrs[b], dtype=np.float64).flatten() for b in bands}
    
    # Initialize outputs
    depth = np.full(n_pixels, 5.0, dtype=np.float64)  # Initial guess: 5m
    sand_frac = np.full(n_pixels, 0.5, dtype=np.float64)  # Initial guess: 50% sand
    
    # Precompute constants
    k_sum = {b: float(Kd[b]) + float(Ku[b]) for b in bands}
    rrs_inf = {b: float(rrs_deep.get(b, 0.01)) for b in bands}
    rho_s = {b: float(rho_sand.get(b, 0.3)) for b in bands}
    rho_g = {b: float(rho_grass.get(b, 0.08)) for b in bands}
    
    # Convergence tracking
    converged = np.zeros(n_pixels, dtype=bool)
    iterations_used = np.zeros(n_pixels, dtype=np.int32)
    final_residual = np.zeros(n_pixels, dtype=np.float64)
    
    log.info(f"[PhysicsSDB] Starting inversion on {n_pixels} pixels, {len(bands)} bands")
    
    # Levenberg-Marquardt optimization (simplified pixel-by-pixel)
    damping = 0.01
    
    for iteration in range(max_iterations):
        # Forward model: compute predicted reflectance
        exp_term = {b: np.exp(-k_sum[b] * depth) for b in bands}
        
        # Bottom reflectance (linear mixture)
        rho_b = {b: sand_frac * rho_s[b] + (1 - sand_frac) * rho_g[b] for b in bands}
        
        # Predicted reflectance (Eq. 19)
        rrs_pred = {
            b: rrs_inf[b] * (1 - exp_term[b]) + (rho_b[b] / np.pi) * exp_term[b]
            for b in bands
        }
        
        # Compute residuals
        residuals = {b: rrs_flat[b] - rrs_pred[b] for b in bands}
        total_residual = np.sqrt(np.sum([residuals[b]**2 for b in bands], axis=0))
        
        # Check convergence
        newly_converged = (total_residual < tolerance) & ~converged
        converged |= newly_converged
        iterations_used[newly_converged] = iteration
        
        if np.all(converged):
            log.info(f"[PhysicsSDB] All pixels converged at iteration {iteration}")
            break
        
        # Compute Jacobian (partial derivatives)
        # ∂rrs/∂z = (Ku+Kd) × (rrs_inf - ρ_b/π) × exp(-(Ku+Kd)×z)
        # ∂rrs/∂Cs = (ρ_sand - ρ_grass)/π × exp(-(Ku+Kd)×z)
        
        J_z = {b: k_sum[b] * (rrs_inf[b] - rho_b[b]/np.pi) * exp_term[b] for b in bands}
        J_Cs = {b: (rho_s[b] - rho_g[b]) / np.pi * exp_term[b] for b in bands}
        
        # Compute update (simplified Gauss-Newton step)
        # For 2 parameters (z, Cs), we need to solve normal equations
        # Here we use a simplified approach: update each parameter separately
        
        # Update depth
        num_z = np.sum([residuals[b] * J_z[b] for b in bands], axis=0)
        den_z = np.sum([J_z[b]**2 for b in bands], axis=0) + damping
        delta_z = num_z / np.maximum(den_z, 1e-10)
        
        # Update sand fraction
        num_Cs = np.sum([residuals[b] * J_Cs[b] for b in bands], axis=0)
        den_Cs = np.sum([J_Cs[b]**2 for b in bands], axis=0) + damping
        delta_Cs = num_Cs / np.maximum(den_Cs, 1e-10)
        
        # Apply updates (only to non-converged pixels)
        update_mask = ~converged
        depth[update_mask] = np.clip(depth[update_mask] + delta_z[update_mask], min_depth, max_depth)
        sand_frac[update_mask] = np.clip(sand_frac[update_mask] + delta_Cs[update_mask], 0.0, 1.0)
    
    # Final residual for uncertainty estimation
    exp_term_final = {b: np.exp(-k_sum[b] * depth) for b in bands}
    rho_b_final = {b: sand_frac * rho_s[b] + (1 - sand_frac) * rho_g[b] for b in bands}
    rrs_pred_final = {
        b: rrs_inf[b] * (1 - exp_term_final[b]) + (rho_b_final[b] / np.pi) * exp_term_final[b]
        for b in bands
    }
    final_residual = np.sqrt(np.sum([(rrs_flat[b] - rrs_pred_final[b])**2 for b in bands], axis=0))
    
    # Uncertainty: base + residual-based + depth-based
    uncertainty = 0.5 + 2.0 * final_residual / np.maximum(np.mean(list(rrs_inf.values())), 0.01)
    uncertainty += 0.05 * depth  # Grows with depth
    uncertainty = np.clip(uncertainty, 0.3, 5.0)
    
    # Reshape outputs
    depth = depth.reshape(shape)
    sand_frac = sand_frac.reshape(shape)
    uncertainty = uncertainty.reshape(shape)
    
    # Mask invalid pixels
    valid_mask = np.all([np.isfinite(rrs[b]) for b in bands], axis=0)
    depth = np.where(valid_mask, depth, np.nan)
    sand_frac = np.where(valid_mask, sand_frac, np.nan)
    uncertainty = np.where(valid_mask, uncertainty, np.nan)
    
    diagnostics = {
        "n_pixels": n_pixels,
        "n_converged": int(np.sum(converged)),
        "convergence_rate": float(np.sum(converged) / n_pixels),
        "mean_iterations": float(np.mean(iterations_used[converged])) if np.any(converged) else 0,
        "mean_residual": float(np.nanmean(final_residual)),
        "bands_used": bands,
    }
    
    log.info(f"[PhysicsSDB] Converged: {diagnostics['n_converged']}/{n_pixels} "
             f"({diagnostics['convergence_rate']*100:.1f}%)")
    
    return depth.astype(np.float32), sand_frac.astype(np.float32), uncertainty.astype(np.float32), diagnostics


def physics_only_predict(
    s2_dir: Union[str, Path],
    output_dir: Union[str, Path],
    kd_estimate: float = 0.15,
    sza_deg: float = 45.0,
    vza_deg: float = 0.0,
    rho_sand: Optional[Dict[str, float]] = None,
    rho_grass: Optional[Dict[str, float]] = None,
    max_depth: float = 25.0,
) -> Dict[str, Any]:
    """
    Full physics-only SDB prediction without any training data.
    
    This is useful when ICESat-2 data is unavailable or as a fallback.
    
    Parameters
    ----------
    s2_dir : str or Path
        Directory containing S2 band rasters
    output_dir : str or Path
        Output directory for results
    kd_estimate : float
        Estimated Kd(490) for the area (m⁻¹)
    sza_deg : float
        Solar zenith angle (degrees)
    vza_deg : float
        View zenith angle (degrees)
    rho_sand : dict, optional
        Sand endmember spectrum (uses default if None)
    rho_grass : dict, optional
        Grass endmember spectrum (uses default if None)
    max_depth : float
        Maximum depth to predict (m)
        
    Returns
    -------
    result : dict
        Dictionary with output paths and diagnostics
    """
    import rasterio
    
    s2_dir = Path(s2_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use default endmembers if not provided
    if rho_sand is None:
        rho_sand = DEFAULT_SAND_SPECTRUM.copy()
    if rho_grass is None:
        rho_grass = DEFAULT_SEAGRASS_SPECTRUM.copy()
    
    # Compute Kd and Ku for each band
    # Scale Kd by wavelength (approximate)
    Kd = {}
    Ku = {}
    for band, wl in S2_WAVELENGTHS.items():
        if band == "B08":  # NIR not used for depth
            continue
        # Scale Kd by wavelength (blue has lowest attenuation in clear water)
        wl_factor = (wl / 490) ** 0.5  # Approximate wavelength scaling
        kd_band = kd_estimate * wl_factor
        kd_corr, ku_corr = compute_kd_ku_from_bands(kd_band, sza_deg, vza_deg, wl)
        Kd[band] = kd_corr
        Ku[band] = ku_corr
    
    log.info(f"[PhysicsOnly] Kd: {Kd}")
    log.info(f"[PhysicsOnly] Ku: {Ku}")
    
    # Load S2 bands
    band_paths = {
        "B02": s2_dir / "B02_10m.tif",
        "B03": s2_dir / "B03_10m.tif",
        "B04": s2_dir / "B04_10m.tif",
        "B08": s2_dir / "B08_10m.tif",
    }
    
    rrs = {}
    with rasterio.open(band_paths["B02"]) as src:
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        
    for band in ["B02", "B03", "B04"]:
        with rasterio.open(band_paths[band]) as src:
            data = src.read(1).astype(np.float64)
            # Convert to Rrs (assuming surface reflectance values)
            rrs[band] = data / np.pi
    
    # Estimate deep water reflectance from dark pixels
    with rasterio.open(band_paths["B08"]) as src:
        nir = src.read(1)
    
    # Deep water: low NIR, low brightness
    brightness = (rrs["B02"] + rrs["B03"] + rrs["B04"]) / 3
    deep_mask = (nir < 0.02) & (brightness < 0.02) & np.isfinite(brightness)
    
    if np.sum(deep_mask) < 100:
        log.warning("[PhysicsOnly] Few deep water pixels, using defaults")
        rrs_deep = {"B02": 0.005, "B03": 0.003, "B04": 0.001}
    else:
        rrs_deep = {b: float(np.nanmedian(rrs[b][deep_mask])) for b in ["B02", "B03", "B04"]}
    
    log.info(f"[PhysicsOnly] Deep water Rrs: {rrs_deep}")
    
    # Create water mask (exclude land)
    water_mask = (nir < 0.1) & (brightness > 0.001) & np.isfinite(brightness)
    
    # Run physics inversion
    depth, sand_frac, uncertainty, diag = physics_sdb_inversion(
        rrs=rrs,
        rrs_deep=rrs_deep,
        Kd=Kd,
        Ku=Ku,
        rho_sand=rho_sand,
        rho_grass=rho_grass,
        max_depth=max_depth,
    )
    
    # Apply water mask
    depth = np.where(water_mask, depth, np.nan)
    
    # Write outputs
    profile.update(dtype="float32", nodata=-9999)
    
    depth_path = output_dir / "SDB_Physics_Depth_10m.tif"
    with rasterio.open(depth_path, "w", **profile) as dst:
        out = np.where(np.isfinite(depth), depth, -9999).astype(np.float32)
        dst.write(out, 1)
    
    unc_path = output_dir / "SDB_Physics_Uncertainty_10m.tif"
    with rasterio.open(unc_path, "w", **profile) as dst:
        out = np.where(np.isfinite(uncertainty), uncertainty, -9999).astype(np.float32)
        dst.write(out, 1)
    
    sand_path = output_dir / "SDB_Physics_SandFraction_10m.tif"
    with rasterio.open(sand_path, "w", **profile) as dst:
        out = np.where(np.isfinite(sand_frac), sand_frac, -9999).astype(np.float32)
        dst.write(out, 1)
    
    # Compute statistics
    valid_depths = depth[np.isfinite(depth)]
    stats = {
        "n_valid_pixels": int(len(valid_depths)),
        "depth_min": float(np.min(valid_depths)) if len(valid_depths) > 0 else None,
        "depth_max": float(np.max(valid_depths)) if len(valid_depths) > 0 else None,
        "depth_median": float(np.median(valid_depths)) if len(valid_depths) > 0 else None,
        "mean_uncertainty": float(np.nanmean(uncertainty)) if len(valid_depths) > 0 else None,
    }
    
    log.info(f"[PhysicsOnly] Valid pixels: {stats['n_valid_pixels']}")
    log.info(f"[PhysicsOnly] Depth range: {stats['depth_min']:.1f} - {stats['depth_max']:.1f} m")
    
    return {
        "depth_path": str(depth_path),
        "uncertainty_path": str(unc_path),
        "sand_fraction_path": str(sand_path),
        "statistics": stats,
        "diagnostics": diag,
        "parameters": {
            "kd_estimate": kd_estimate,
            "sza_deg": sza_deg,
            "vza_deg": vza_deg,
            "rho_sand": rho_sand,
            "rho_grass": rho_grass,
            "Kd": Kd,
            "Ku": Ku,
            "rrs_deep": rrs_deep,
        },
    }


# -----------------------------------------------------------------------------
# Integration helpers
# -----------------------------------------------------------------------------

def enhance_hybrid_prediction_with_physics(
    rf_pred: np.ndarray,
    rf_std: np.ndarray,
    rrs: Dict[str, np.ndarray],
    rrs_deep: Dict[str, float],
    Kd: Dict[str, float],
    Ku: Dict[str, float],
    rho_sand: Dict[str, float],
    rho_grass: Dict[str, float],
    training_max_depth: float,
    blend_threshold_depth: float = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Enhanced hybrid prediction that uses full physics model for extrapolation.
    
    This replaces the simple Stumpf-based extrapolation with the full
    radiative transfer model from Kim et al.
    
    Parameters
    ----------
    rf_pred : ndarray
        RF predicted depths
    rf_std : ndarray
        RF prediction uncertainty (tree std)
    rrs : dict
        Remote sensing reflectance per band
    rrs_deep : dict
        Deep water reflectance
    Kd, Ku : dict
        Direction-specific attenuation coefficients
    rho_sand, rho_grass : dict
        Bottom endmember spectra
    training_max_depth : float
        Maximum depth in training data
    blend_threshold_depth : float, optional
        Depth at which to start blending (default: 0.8 × training_max)
        
    Returns
    -------
    blended_depth : ndarray
        Hybrid prediction
    blended_uncertainty : ndarray
        Combined uncertainty
    diagnostics : dict
        Blending statistics
    """
    if blend_threshold_depth is None:
        blend_threshold_depth = training_max_depth * 0.8
    
    # Run physics prediction
    physics_depth, _, physics_unc, _ = physics_sdb_inversion(
        rrs=rrs,
        rrs_deep=rrs_deep,
        Kd=Kd,
        Ku=Ku,
        rho_sand=rho_sand,
        rho_grass=rho_grass,
    )
    
    # Compute blend weight: 0 at shallow depths, 1 at/beyond training edge
    blend_weight = np.clip(
        (rf_pred - blend_threshold_depth) / (training_max_depth - blend_threshold_depth + 0.1),
        0.0, 1.0
    )
    
    # Additional weight from RF uncertainty
    rf_uncertain = np.clip(rf_std / 0.3, 0.0, 1.0)
    blend_weight = np.maximum(blend_weight, rf_uncertain * 0.5)
    
    # Only blend where physics predicts deeper (safe extrapolation)
    physics_deeper = physics_depth > rf_pred
    blend_weight = np.where(physics_deeper, blend_weight, 0.0)
    
    # Blend predictions
    blended_depth = (1 - blend_weight) * rf_pred + blend_weight * physics_depth
    
    # Combine uncertainties
    rf_unc = np.maximum(rf_std, 0.1)
    blended_uncertainty = np.sqrt(
        (1 - blend_weight)**2 * rf_unc**2 + 
        blend_weight**2 * physics_unc**2
    )
    
    # Add extrapolation penalty
    extrap_distance = np.maximum(blended_depth - training_max_depth, 0)
    blended_uncertainty += 0.1 * extrap_distance
    
    diagnostics = {
        "n_blended": int(np.sum(blend_weight > 0.01)),
        "n_physics_deeper": int(np.sum(physics_deeper)),
        "mean_blend_weight": float(np.nanmean(blend_weight)),
        "max_physics_depth": float(np.nanmax(physics_depth)),
    }
    
    return blended_depth.astype(np.float32), blended_uncertainty.astype(np.float32), diagnostics
