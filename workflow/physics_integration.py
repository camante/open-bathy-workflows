#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
physics_integration.py - Integrates bottom_physics.py (Kim et al. 2024) into the SDB pipeline.

Handles scene-derived bottom endmember estimation, geometry-corrected Kd/Ku computation,
physics-only prediction, and hybrid RF+physics prediction.

Usage in sdb_main.py:
---------------------
    from physics_integration import (
        estimate_scene_bottom_endmembers,
        compute_geometry_corrected_attenuation,
        run_physics_only_prediction,
        enhance_prediction_with_physics,
    )
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
import numpy as np

log = logging.getLogger(__name__)

# Try to import the new physics module
try:
    from bottom_physics import (
        compute_kd_ku_from_bands,
        estimate_bottom_endmembers_from_training,
        physics_only_predict,
        enhance_hybrid_prediction_with_physics,
        DEFAULT_SAND_SPECTRUM,
        DEFAULT_SEAGRASS_SPECTRUM,
        S2_WAVELENGTHS,
    )
    PHYSICS_MODULE_AVAILABLE = True
except ImportError:
    PHYSICS_MODULE_AVAILABLE = False
    log.warning("bottom_physics module not available")


def estimate_scene_bottom_endmembers(
    training_df,
    s2_dir: Union[str, Path],
    kd_estimate: float,
    sza_deg: float = 45.0,
    vza_deg: float = 0.0,
    max_depth_for_endmembers: float = 2.0,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    Estimate scene-specific bottom endmembers from training data.
    
    This implements Kim et al. (2024) Section 3.6 eigenanalysis approach.
    
    Parameters
    ----------
    training_df : DataFrame
        Training data with depth_m, longitude, latitude columns
    s2_dir : str or Path
        Directory containing Sentinel-2 band rasters
    kd_estimate : float
        Estimated Kd(490) for the scene (m⁻¹)
    sza_deg : float
        Solar zenith angle (degrees)
    vza_deg : float
        View zenith angle (degrees) - typically small for S2
    max_depth_for_endmembers : float
        Maximum depth to use for endmember estimation (default 2m)
    output_dir : str or Path, optional
        Directory to save endmember diagnostics JSON
        
    Returns
    -------
    rho_sand : dict
        Sand-like endmember spectrum {B02: val, B03: val, B04: val}
    rho_grass : dict  
        Grass/seagrass-like endmember spectrum
    diagnostics : dict
        Analysis diagnostics including eigenvalues, variance explained, etc.
    """
    if not PHYSICS_MODULE_AVAILABLE:
        log.warning("Physics module unavailable, using default endmembers")
        return DEFAULT_SAND_SPECTRUM.copy(), DEFAULT_SEAGRASS_SPECTRUM.copy(), {"error": "module_unavailable"}
    
    # Compute Kd and Ku for each band
    Kd = {}
    Ku = {}
    for band, wl in S2_WAVELENGTHS.items():
        if band == "B08":
            continue
        # Scale Kd by wavelength
        wl_factor = (wl / 490) ** 0.5
        kd_band = kd_estimate * wl_factor
        kd_corr, ku_corr = compute_kd_ku_from_bands(kd_band, sza_deg, vza_deg, wl)
        Kd[band] = kd_corr
        Ku[band] = ku_corr
    
    log.info("Computing bottom endmembers with Kd=%s, Ku=%s", Kd, Ku)
    
    # Estimate deep water reflectance from rasters
    import rasterio
    s2_dir = Path(s2_dir)
    
    rrs_deep = None
    try:
        b08_path = s2_dir / "B08_10m.tif"
        if b08_path.exists():
            with rasterio.open(b08_path) as src:
                nir = src.read(1)
            
            # Load visible bands
            rrs = {}
            for band in ["B02", "B03", "B04"]:
                with rasterio.open(s2_dir / f"{band}_10m.tif") as src:
                    rrs[band] = src.read(1) / np.pi  # Convert to Rrs
            
            # Deep water mask
            brightness = (rrs["B02"] + rrs["B03"] + rrs["B04"]) / 3
            deep_mask = (nir < 0.02) & (brightness < 0.02) & np.isfinite(brightness)
            
            if np.sum(deep_mask) > 100:
                rrs_deep = {b: float(np.nanmedian(rrs[b][deep_mask])) for b in ["B02", "B03", "B04"]}
                log.info("Deep water Rrs: %s", rrs_deep)
    except Exception as e:
        log.warning("Could not estimate deep water Rrs: %s", e)
    
    # Run endmember estimation
    try:
        rho_sand, rho_grass, diag = estimate_bottom_endmembers_from_training(
            training_df=training_df,
            s2_dir=s2_dir,
            Kd=Kd,
            Ku=Ku,
            max_depth_for_endmembers=max_depth_for_endmembers,
            rrs_deep=rrs_deep,
        )
        
        diag["Kd"] = Kd
        diag["Ku"] = Ku
        diag["rrs_deep"] = rrs_deep
        diag["sza_deg"] = sza_deg
        diag["vza_deg"] = vza_deg
        
        # Save diagnostics
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            diag_path = output_dir / "bottom_endmembers.json"
            with open(diag_path, "w", encoding="utf-8") as f:
                # Convert numpy types
                diag_json = json.loads(json.dumps(diag, default=lambda x: float(x) if hasattr(x, 'item') else str(x)))
                json.dump({
                    "rho_sand": rho_sand,
                    "rho_grass": rho_grass,
                    "diagnostics": diag_json,
                }, f, indent=2)
            log.info("Saved endmember diagnostics to %s", diag_path)
        
        return rho_sand, rho_grass, diag
        
    except Exception as e:
        log.error("Endmember estimation failed: %s", e, exc_info=True)
        return DEFAULT_SAND_SPECTRUM.copy(), DEFAULT_SEAGRASS_SPECTRUM.copy(), {"error": str(e)}


def compute_geometry_corrected_attenuation(
    kd_scalar: float,
    sza_deg: float,
    vza_deg: float = 0.0,
    bands: Optional[list] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute geometry-corrected Kd and Ku for each band.
    
    This implements Kim et al. (2024) Equations 17-18 for direction-specific
    diffuse attenuation coefficients.
    
    Parameters
    ----------
    kd_scalar : float
        Scalar Kd(490) estimate (m⁻¹)
    sza_deg : float
        Solar zenith angle (degrees)
    vza_deg : float
        View zenith angle (degrees)
    bands : list, optional
        Bands to compute (default: B02, B03, B04)
        
    Returns
    -------
    Kd : dict
        Downwelling attenuation per band {B02: val, ...}
    Ku : dict
        Upwelling attenuation per band
    """
    if not PHYSICS_MODULE_AVAILABLE:
        # Fallback: simple scaling without geometry correction
        Kd = {"B02": kd_scalar, "B03": kd_scalar * 1.1, "B04": kd_scalar * 1.3}
        Ku = Kd.copy()
        return Kd, Ku
    
    if bands is None:
        bands = ["B02", "B03", "B04"]
    
    Kd = {}
    Ku = {}
    
    for band in bands:
        if band not in S2_WAVELENGTHS:
            continue
        wl = S2_WAVELENGTHS[band]
        
        # Scale Kd by wavelength (approximate relationship)
        wl_factor = (wl / 490) ** 0.5
        kd_band = kd_scalar * wl_factor
        
        kd_corr, ku_corr = compute_kd_ku_from_bands(kd_band, sza_deg, vza_deg, wl)
        Kd[band] = kd_corr
        Ku[band] = ku_corr
    
    return Kd, Ku


def run_physics_only_prediction(
    s2_dir: Union[str, Path],
    output_dir: Union[str, Path],
    kd_estimate: float,
    sza_deg: float = 45.0,
    vza_deg: float = 0.0,
    rho_sand: Optional[Dict[str, float]] = None,
    rho_grass: Optional[Dict[str, float]] = None,
    max_depth: float = 25.0,
) -> Dict[str, Any]:
    """
    Run physics-only SDB prediction (no training data required).
    
    This is useful as a fallback when ICESat-2 data is unavailable,
    or for reconnaissance mapping.
    
    Parameters
    ----------
    s2_dir : str or Path
        Directory containing S2 band rasters
    output_dir : str or Path
        Output directory
    kd_estimate : float
        Estimated Kd(490) for the scene (m⁻¹)
    sza_deg : float
        Solar zenith angle (degrees)
    vza_deg : float
        View zenith angle (degrees)
    rho_sand, rho_grass : dict, optional
        Bottom endmember spectra (uses defaults if None)
    max_depth : float
        Maximum depth to predict (m)
        
    Returns
    -------
    result : dict
        Output paths and statistics
    """
    if not PHYSICS_MODULE_AVAILABLE:
        log.error("Cannot run physics-only: module unavailable")
        return {"error": "module_unavailable"}
    
    return physics_only_predict(
        s2_dir=s2_dir,
        output_dir=output_dir,
        kd_estimate=kd_estimate,
        sza_deg=sza_deg,
        vza_deg=vza_deg,
        rho_sand=rho_sand,
        rho_grass=rho_grass,
        max_depth=max_depth,
    )


def enhance_prediction_with_physics(
    rf_pred: np.ndarray,
    rf_std: np.ndarray,
    s2_dir: Union[str, Path],
    kd_estimate: float,
    sza_deg: float,
    vza_deg: float,
    rho_sand: Dict[str, float],
    rho_grass: Dict[str, float],
    training_max_depth: float,
    blend_start_depth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Enhance RF predictions with physics-based extrapolation.
    
    This replaces simple Stumpf-based hybrid mode with full radiative
    transfer model from Kim et al. (2024).
    
    Parameters
    ----------
    rf_pred : ndarray
        RF predicted depths (2D array)
    rf_std : ndarray
        RF prediction uncertainty
    s2_dir : str or Path
        Directory with S2 rasters
    kd_estimate : float
        Kd(490) estimate
    sza_deg, vza_deg : float
        Sun/view geometry
    rho_sand, rho_grass : dict
        Bottom endmember spectra
    training_max_depth : float
        Maximum depth in training data
    blend_start_depth : float, optional
        Depth at which to start blending physics (default: 0.8 × training_max)
        
    Returns
    -------
    blended_depth : ndarray
        Hybrid prediction
    blended_uncertainty : ndarray
        Combined uncertainty
    diagnostics : dict
        Blending statistics
    """
    if not PHYSICS_MODULE_AVAILABLE:
        log.warning("Physics module unavailable, returning RF predictions")
        return rf_pred, rf_std, {"error": "module_unavailable"}
    
    import rasterio
    s2_dir = Path(s2_dir)
    
    # Load Rrs
    rrs = {}
    for band in ["B02", "B03", "B04"]:
        with rasterio.open(s2_dir / f"{band}_10m.tif") as src:
            rrs[band] = src.read(1).astype(np.float64) / np.pi
    
    # Compute Kd/Ku
    Kd, Ku = compute_geometry_corrected_attenuation(kd_estimate, sza_deg, vza_deg)
    
    # Estimate deep water Rrs
    with rasterio.open(s2_dir / "B08_10m.tif") as src:
        nir = src.read(1)
    brightness = (rrs["B02"] + rrs["B03"] + rrs["B04"]) / 3
    deep_mask = (nir < 0.02) & (brightness < 0.02) & np.isfinite(brightness)
    
    if np.sum(deep_mask) > 100:
        rrs_deep = {b: float(np.nanmedian(rrs[b][deep_mask])) for b in ["B02", "B03", "B04"]}
    else:
        rrs_deep = {"B02": 0.005, "B03": 0.003, "B04": 0.001}
    
    return enhance_hybrid_prediction_with_physics(
        rf_pred=rf_pred,
        rf_std=rf_std,
        rrs=rrs,
        rrs_deep=rrs_deep,
        Kd=Kd,
        Ku=Ku,
        rho_sand=rho_sand,
        rho_grass=rho_grass,
        training_max_depth=training_max_depth,
        blend_threshold_depth=blend_start_depth,
    )


def get_sun_view_angles_from_s2_metadata(s2_dir: Union[str, Path]) -> Tuple[float, float]:
    """
    Extract sun/view angles from S2 metadata files or compute from location/date.
    
    Checks multiple sources in order:
    1. S2_DATE_QC.json (created by s2_optics.py with mean angles)
    2. STAC metadata JSON (if saved during download)
    3. GeoTIFF metadata tags
    4. Compute from centroid lat/lon and date (astronomical calculation)
    5. Fall back to reasonable defaults
    
    Parameters
    ----------
    s2_dir : str or Path
        Directory containing S2 rasters and metadata
        
    Returns
    -------
    sza_deg : float
        Mean solar zenith angle (degrees)
    vza_deg : float
        Mean view zenith angle (degrees)
    """
    import rasterio
    from datetime import datetime
    
    s2_dir = Path(s2_dir)
    
    sza_deg = None
    vza_deg = 5.0   # Default for S2 (near-nadir, typical value)
    
    # Source 1: Check S2_DATE_QC.json first (created by s2_optics.py)
    qc_path = s2_dir / "S2_DATE_QC.json"
    if qc_path.exists():
        try:
            with open(qc_path) as f:
                qc = json.load(f)
            
            if qc.get("mean_sun_zenith") is not None:
                sza_deg = float(qc["mean_sun_zenith"])
            if qc.get("mean_view_zenith") is not None:
                vza_deg = float(qc["mean_view_zenith"])
            
            if sza_deg is not None:
                log.info("Found angles from S2_DATE_QC.json: SZA=%.1f°, VZA=%.1f°", sza_deg, vza_deg)
                return float(sza_deg), float(min(abs(vza_deg), 12.0))
        except Exception as e:
            log.debug("Could not read S2_DATE_QC.json: %s", e)
    
    # Source 2: Try other metadata JSON files
    meta_paths = (
        list(s2_dir.glob("*META*.json")) + 
        list(s2_dir.glob("*stac*.json")) +
        list(s2_dir.glob("metadata.json"))
    )
    
    for meta_path in meta_paths:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            
            # STAC properties format (Element84 Earth Search)
            props = meta.get("properties", meta)  # Handle both formats
            
            # Sun zenith - try various property names
            for key in ["s2:mean_solar_zenith", "view:sun_elevation", "sun_zenith", 
                        "mean_sun_zenith", "solar_zenith_angle", "sza",
                        "s2:sun_zenith", "eo:sun_zenith"]:
                if key in props:
                    val = props[key]
                    if "elevation" in key.lower():
                        # Sun elevation -> zenith = 90 - elevation
                        sza_deg = 90.0 - float(val)
                    else:
                        sza_deg = float(val)
                    break
            
            # View zenith - try various property names  
            for key in ["s2:mean_viewing_zenith", "view:off_nadir", "view_zenith",
                        "mean_view_zenith", "viewing_zenith_angle", "vza",
                        "s2:view_zenith", "eo:view_zenith"]:
                if key in props:
                    vza_deg = float(props[key])
                    break
            
            if sza_deg is not None:
                log.info("Found angles from %s: SZA=%.1f°, VZA=%.1f°", meta_path.name, sza_deg, vza_deg)
                break
                
        except Exception:
            log.debug("physics_integration: suppressed exception", exc_info=True)
            continue
    
    # Source 2: Try to compute from raster metadata (date + location)
    if sza_deg is None:
        try:
            # Find a raster to get location and date
            raster_path = None
            for pattern in ["B02*.tif", "B03*.tif", "*_10m.tif"]:
                matches = list(s2_dir.glob(pattern))
                if matches:
                    raster_path = matches[0]
                    break
            
            if raster_path:
                with rasterio.open(raster_path) as src:
                    bounds = src.bounds
                    # Get centroid
                    center_lon = (bounds.left + bounds.right) / 2
                    center_lat = (bounds.top + bounds.bottom) / 2
                    
                    # Try to get date from filename or tags
                    date_str = None
                    
                    # Check raster tags
                    tags = src.tags()
                    for key in ["TIFFTAG_DATETIME", "datetime", "acquisition_date"]:
                        if key in tags:
                            date_str = tags[key]
                            break
                    
                    # Try filename patterns (YYYYMMDD or YYYY-MM-DD)
                    if date_str is None:
                        import re
                        fname = raster_path.stem
                        # Match YYYYMMDD
                        m = re.search(r'(\d{4})(\d{2})(\d{2})', fname)
                        if m:
                            date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    
                    if date_str and center_lat:
                        sza_deg = compute_solar_zenith(center_lat, center_lon, date_str)
                        if sza_deg is not None:
                            log.info("Computed SZA=%.1f° from location (%.2f, %.2f) and date", sza_deg, center_lat, center_lon)
                            
        except Exception as e:
            log.debug("Could not compute SZA from raster: %s", e)
    
    # Source 3: Default based on typical tropical/subtropical conditions
    if sza_deg is None:
        sza_deg = 45.0  # Reasonable default for most SDB applications
        log.info("Using default angles: SZA=%.1f°, VZA=%.1f°", sza_deg, vza_deg)
    
    # Sanity check VZA (Sentinel-2 is typically < 10°)
    vza_deg = min(abs(vza_deg), 12.0)
    
    return float(sza_deg), float(vza_deg)


def compute_solar_zenith(lat: float, lon: float, date_str: str, hour_utc: float = 10.5) -> Optional[float]:
    """
    Compute approximate solar zenith angle for a location and date.
    
    Uses simplified astronomical calculation (accurate to ~1-2°).
    
    Parameters
    ----------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees  
    date_str : str
        Date string (YYYY-MM-DD format)
    hour_utc : float
        Hour of day in UTC (default 10:30 for typical S2 overpass)
        
    Returns
    -------
    sza : float or None
        Solar zenith angle in degrees
    """
    import math
    from datetime import datetime
    
    try:
        # Parse date
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            hour_utc = dt.hour + dt.minute / 60.0
        else:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        
        # Day of year
        doy = dt.timetuple().tm_yday
        
        # Solar declination (simplified)
        declination = 23.45 * math.sin(math.radians(360 * (284 + doy) / 365))
        
        # Hour angle (degrees from solar noon)
        # Solar noon occurs when sun is directly over the longitude
        solar_noon_utc = 12.0 - lon / 15.0  # Approximate
        hour_angle = 15.0 * (hour_utc - solar_noon_utc)
        
        # Solar zenith angle
        lat_rad = math.radians(lat)
        dec_rad = math.radians(declination)
        ha_rad = math.radians(hour_angle)
        
        cos_sza = (
            math.sin(lat_rad) * math.sin(dec_rad) +
            math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad)
        )
        
        # Clip to valid range
        cos_sza = max(-1.0, min(1.0, cos_sza))
        sza = math.degrees(math.acos(cos_sza))
        
        return sza
        
    except Exception as e:
        log.debug("Solar zenith calculation failed: %s", e)
        return None


# -----------------------------------------------------------------------------
# CLI support for standalone physics prediction
# -----------------------------------------------------------------------------

def main():
    """Command-line interface for physics-only SDB prediction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Physics-only SDB prediction (no training data required)"
    )
    parser.add_argument("--s2-dir", required=True, help="Directory with S2 rasters")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--kd", type=float, default=0.15, help="Kd(490) estimate (m⁻¹)")
    parser.add_argument("--sza", type=float, default=45.0, help="Solar zenith angle (degrees)")
    parser.add_argument("--vza", type=float, default=5.0, help="View zenith angle (degrees)")
    parser.add_argument("--max-depth", type=float, default=25.0, help="Maximum depth (m)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    result = run_physics_only_prediction(
        s2_dir=args.s2_dir,
        output_dir=args.output_dir,
        kd_estimate=args.kd,
        sza_deg=args.sza,
        vza_deg=args.vza,
        max_depth=args.max_depth,
    )
    
    log.info(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    try:
        from core.logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
