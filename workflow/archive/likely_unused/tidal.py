#!/usr/bin/env python3
"""
tidal.py - Tidal Correction for SDB Pipeline

This module provides tidal height estimation to correct for water level differences
between ICESat-2 acquisition times and Sentinel-2 acquisition times.

THEORY
======
ICESat-2 and Sentinel-2 capture the water surface at different times. If ICESat-2
measures depth at high tide and S2 imagery is from low tide (or vice versa), there
will be a systematic bias in the trained model.

Correction approach:
1. Estimate tidal height at ICESat-2 acquisition time
2. Estimate tidal height at S2 acquisition time (or reference datum)
3. Apply correction: depth_corrected = depth_measured + (tide_ref - tide_atl)

TIDAL MODELS
============
This module supports multiple tidal prediction approaches:

1. NOAA CO-OPS API (US waters only)
   - Real-time and predicted tide data from NOAA stations
   - Most accurate for US coastal areas
   - Requires internet connection

2. Harmonic prediction (offline, global)
   - Uses pre-computed harmonic constituents
   - Works offline after initial setup
   - Good accuracy for open ocean, less accurate near complex coastlines

3. FES2014 / TPXO (external models)
   - High-accuracy global tidal models
   - Requires separate installation and data files
   - Best accuracy but more complex setup

4. Simple astronomical approximation
   - Basic lunar/solar tidal prediction
   - No external data required
   - Lower accuracy but always available

DATUM CONVENTIONS
=================
- MSL (Mean Sea Level): Standard reference, used as default
- MLLW (Mean Lower Low Water): US chart datum
- LAT (Lowest Astronomical Tide): UK/international chart datum

All corrections normalize to MSL unless otherwise specified.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Major tidal constituents (period in hours)
TIDAL_CONSTITUENTS = {
    "M2": 12.4206,   # Principal lunar semidiurnal
    "S2": 12.0000,   # Principal solar semidiurnal
    "N2": 12.6583,   # Larger lunar elliptic
    "K1": 23.9345,   # Lunar diurnal
    "O1": 25.8193,   # Lunar diurnal
    "P1": 24.0659,   # Solar diurnal
    "K2": 11.9672,   # Lunisolar semidiurnal
    "Q1": 26.8684,   # Larger lunar elliptic diurnal
}

# Approximate amplitudes (meters) for open ocean - very rough!
# Real values vary hugely by location
DEFAULT_AMPLITUDES = {
    "M2": 0.25,
    "S2": 0.10,
    "N2": 0.05,
    "K1": 0.15,
    "O1": 0.10,
    "P1": 0.05,
    "K2": 0.03,
    "Q1": 0.02,
}

# Regional tidal range estimates (meters, approximate)
REGIONAL_TIDAL_RANGES = {
    "atlantic_us_east": {"range": 1.5, "type": "semidiurnal"},
    "atlantic_us_southeast": {"range": 2.0, "type": "semidiurnal"},
    "gulf_mexico": {"range": 0.5, "type": "diurnal"},
    "pacific_us_west": {"range": 1.8, "type": "mixed"},
    "caribbean": {"range": 0.3, "type": "mixed"},
    "hawaii": {"range": 0.6, "type": "mixed"},
    "mediterranean": {"range": 0.3, "type": "semidiurnal"},
    "uk_atlantic": {"range": 5.0, "type": "semidiurnal"},
    "australia_east": {"range": 1.5, "type": "semidiurnal"},
    "australia_north": {"range": 6.0, "type": "semidiurnal"},
    "default": {"range": 1.0, "type": "mixed"},
}


@dataclass
class TidalPrediction:
    """Result of tidal height prediction."""
    height_m: float              # Tidal height relative to datum
    datum: str                   # Reference datum (MSL, MLLW, etc.)
    time_utc: datetime           # Time of prediction
    method: str                  # Prediction method used
    uncertainty_m: float         # Estimated uncertainty
    metadata: Dict[str, Any]     # Additional info


@dataclass  
class TidalCorrection:
    """Tidal correction to apply to depth measurements."""
    correction_m: float          # Add this to measured depth
    tide_at_measurement: float   # Tidal height when depth was measured
    tide_at_reference: float     # Tidal height at reference time
    reference_datum: str         # Datum used
    uncertainty_m: float         # Combined uncertainty
    method: str                  # Method used


# -----------------------------------------------------------------------------
# Region detection
# -----------------------------------------------------------------------------

def estimate_region(lon: float, lat: float) -> str:
    """
    Estimate tidal region from coordinates.
    
    Returns region key for REGIONAL_TIDAL_RANGES lookup.
    """
    # US East Coast
    if -82 < lon < -65 and 25 < lat < 45:
        if lat < 32:
            return "atlantic_us_southeast"
        return "atlantic_us_east"
    
    # Gulf of Mexico
    if -98 < lon < -80 and 18 < lat < 31:
        return "gulf_mexico"
    
    # US West Coast
    if -130 < lon < -115 and 30 < lat < 50:
        return "pacific_us_west"
    
    # Caribbean
    if -90 < lon < -60 and 10 < lat < 25:
        return "caribbean"
    
    # Hawaii
    if -162 < lon < -154 and 18 < lat < 23:
        return "hawaii"
    
    # Mediterranean
    if -6 < lon < 37 and 30 < lat < 46:
        return "mediterranean"
    
    # UK/Ireland
    if -12 < lon < 3 and 49 < lat < 61:
        return "uk_atlantic"
    
    # Australia East
    if 140 < lon < 155 and -40 < lat < -10:
        return "australia_east"
    
    # Australia North
    if 120 < lon < 140 and -20 < lat < -10:
        return "australia_north"
    
    return "default"


def get_regional_tidal_range(lon: float, lat: float) -> Tuple[float, str]:
    """
    Get approximate tidal range for a location.
    
    Returns (range_m, tidal_type)
    """
    region = estimate_region(lon, lat)
    info = REGIONAL_TIDAL_RANGES.get(region, REGIONAL_TIDAL_RANGES["default"])
    return info["range"], info["type"]


# -----------------------------------------------------------------------------
# Simple harmonic tidal prediction
# -----------------------------------------------------------------------------

def predict_tide_harmonic(
    lon: float,
    lat: float,
    time_utc: datetime,
    amplitudes: Dict[str, float] = None,
    phases: Dict[str, float] = None,
) -> TidalPrediction:
    """
    Predict tidal height using harmonic constituents.
    
    This is a simplified harmonic model. For accurate predictions,
    use location-specific harmonic constants from tidal databases.
    
    Parameters
    ----------
    lon, lat : float
        Location coordinates
    time_utc : datetime
        Time for prediction (UTC)
    amplitudes : dict, optional
        Harmonic amplitudes by constituent (meters)
    phases : dict, optional
        Harmonic phases by constituent (degrees)
        
    Returns
    -------
    TidalPrediction
    """
    if amplitudes is None:
        # Scale default amplitudes by regional tidal range
        range_m, tidal_type = get_regional_tidal_range(lon, lat)
        scale = range_m / 1.0  # Default range is ~1m
        amplitudes = {k: v * scale for k, v in DEFAULT_AMPLITUDES.items()}
    
    if phases is None:
        # Approximate phases based on longitude (very rough!)
        # Real phases depend on complex coastal geometry
        base_phase = (lon % 360) * 0.5  # Rough westward propagation
        phases = {k: base_phase + i * 30 for i, k in enumerate(TIDAL_CONSTITUENTS.keys())}
    
    # Reference epoch (J2000.0)
    epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    hours_since_epoch = (time_utc.replace(tzinfo=timezone.utc) - epoch).total_seconds() / 3600.0
    
    # Sum harmonic contributions
    height = 0.0
    for constituent, period_hours in TIDAL_CONSTITUENTS.items():
        amp = amplitudes.get(constituent, 0.0)
        phase = phases.get(constituent, 0.0)
        
        # Angular frequency (radians per hour)
        omega = 2 * np.pi / period_hours
        
        # Height contribution
        height += amp * np.cos(omega * hours_since_epoch - np.radians(phase))
    
    # Estimate uncertainty based on method
    range_m, _ = get_regional_tidal_range(lon, lat)
    uncertainty = range_m * 0.3  # ~30% of tidal range as uncertainty
    
    return TidalPrediction(
        height_m=float(height),
        datum="MSL",
        time_utc=time_utc,
        method="harmonic_simple",
        uncertainty_m=uncertainty,
        metadata={
            "region": estimate_region(lon, lat),
            "tidal_range_m": range_m,
            "n_constituents": len([a for a in amplitudes.values() if a > 0]),
        }
    )


# -----------------------------------------------------------------------------
# NOAA CO-OPS API (US waters)
# -----------------------------------------------------------------------------

def get_nearest_noaa_station(lon: float, lat: float) -> Optional[Dict[str, Any]]:
    """
    Find nearest NOAA tide station.
    
    Returns station info dict or None if no station found.
    """
    # Common NOAA stations (subset - real implementation would use full database)
    NOAA_STATIONS = [
        {"id": "8723214", "name": "Virginia Key", "lon": -80.1617, "lat": 25.7317},
        {"id": "8724580", "name": "Key West", "lon": -81.8067, "lat": 24.5508},
        {"id": "8726520", "name": "St Petersburg", "lon": -82.6267, "lat": 27.7606},
        {"id": "8727520", "name": "Cedar Key", "lon": -83.0317, "lat": 29.1350},
        {"id": "8729108", "name": "Panama City", "lon": -85.6678, "lat": 30.1522},
        {"id": "8761724", "name": "Grand Isle", "lon": -89.9572, "lat": 29.2633},
        {"id": "8770570", "name": "Sabine Pass", "lon": -93.8700, "lat": 29.7283},
        {"id": "8771341", "name": "Galveston Bay", "lon": -94.8433, "lat": 29.3572},
        {"id": "8443970", "name": "Boston", "lon": -71.0503, "lat": 42.3539},
        {"id": "8518750", "name": "NYC Battery", "lon": -74.0142, "lat": 40.7006},
        {"id": "8658120", "name": "Wilmington NC", "lon": -77.9536, "lat": 34.2267},
        {"id": "8665530", "name": "Charleston", "lon": -79.9236, "lat": 32.7817},
        {"id": "9410660", "name": "Los Angeles", "lon": -118.2720, "lat": 33.7200},
        {"id": "9414290", "name": "San Francisco", "lon": -122.4659, "lat": 37.8063},
        {"id": "9447130", "name": "Seattle", "lon": -122.3392, "lat": 47.6025},
    ]
    
    if not (-130 < lon < -60 and 24 < lat < 50):
        return None  # Outside US waters
    
    # Find nearest station
    min_dist = float('inf')
    nearest = None
    
    for station in NOAA_STATIONS:
        dist = np.sqrt((lon - station["lon"])**2 + (lat - station["lat"])**2)
        if dist < min_dist:
            min_dist = dist
            nearest = station
    
    if nearest and min_dist < 3.0:  # Within ~3 degrees
        nearest["distance_deg"] = min_dist
        return nearest
    
    return None


def predict_tide_noaa(
    lon: float,
    lat: float, 
    time_utc: datetime,
    station_id: str = None,
) -> Optional[TidalPrediction]:
    """
    Get tidal prediction from NOAA CO-OPS API.
    
    Requires internet connection. Falls back to harmonic if API fails.
    
    Parameters
    ----------
    lon, lat : float
        Location coordinates
    time_utc : datetime
        Time for prediction (UTC)
    station_id : str, optional
        NOAA station ID. If None, uses nearest station.
        
    Returns
    -------
    TidalPrediction or None if API unavailable
    """
    try:
        import urllib.request
        import json
    except ImportError:
        return None
    
    # Find station
    if station_id is None:
        station = get_nearest_noaa_station(lon, lat)
        if station is None:
            return None
        station_id = station["id"]
    
    # Format time for API
    date_str = time_utc.strftime("%Y%m%d")
    
    # NOAA CO-OPS API endpoint
    url = (
        f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
        f"date={date_str}&station={station_id}&product=predictions"
        f"&datum=MSL&units=metric&time_zone=gmt&format=json&interval=h"
    )
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        predictions = data.get("predictions", [])
        if not predictions:
            return None
        
        # Find closest prediction to requested time
        target_hour = time_utc.hour + time_utc.minute / 60
        best_pred = None
        best_diff = float('inf')
        
        for pred in predictions:
            pred_time = datetime.strptime(pred["t"], "%Y-%m-%d %H:%M")
            pred_hour = pred_time.hour + pred_time.minute / 60
            diff = abs(pred_hour - target_hour)
            if diff < best_diff:
                best_diff = diff
                best_pred = pred
        
        if best_pred is None:
            return None
        
        return TidalPrediction(
            height_m=float(best_pred["v"]),
            datum="MSL",
            time_utc=time_utc,
            method="noaa_api",
            uncertainty_m=0.05,  # NOAA predictions are quite accurate
            metadata={
                "station_id": station_id,
                "station_name": station.get("name", "Unknown") if station else "Unknown",
            }
        )
        
    except Exception as e:
        log.debug(f"NOAA API request failed: {e}")
        return None


# -----------------------------------------------------------------------------
# Main tidal correction interface
# -----------------------------------------------------------------------------

def predict_tide(
    lon: float,
    lat: float,
    time_utc: datetime,
    method: str = "auto",
) -> TidalPrediction:
    """
    Predict tidal height at a location and time.
    
    Parameters
    ----------
    lon, lat : float
        Location coordinates (degrees)
    time_utc : datetime
        Time for prediction (should be UTC)
    method : str
        Prediction method: 'auto', 'noaa', 'harmonic'
        
    Returns
    -------
    TidalPrediction
    """
    # Ensure timezone-aware
    if time_utc.tzinfo is None:
        time_utc = time_utc.replace(tzinfo=timezone.utc)
    
    if method == "auto":
        # Try NOAA first for US waters
        pred = predict_tide_noaa(lon, lat, time_utc)
        if pred is not None:
            return pred
        # Fall back to harmonic
        return predict_tide_harmonic(lon, lat, time_utc)
    
    elif method == "noaa":
        pred = predict_tide_noaa(lon, lat, time_utc)
        if pred is None:
            log.warning("NOAA prediction failed, falling back to harmonic")
            return predict_tide_harmonic(lon, lat, time_utc)
        return pred
    
    else:  # harmonic
        return predict_tide_harmonic(lon, lat, time_utc)


def compute_tidal_correction(
    lon: float,
    lat: float,
    time_measurement: datetime,
    time_reference: datetime = None,
    method: str = "auto",
) -> TidalCorrection:
    """
    Compute tidal correction between measurement time and reference time.
    
    For SDB with S2 composites:
    --------------------------
    S2 median composites aggregate imagery across many dates/tidal states,
    so they effectively represent Mean Sea Level (MSL). Therefore:
    
    - Set time_reference=None (default) to correct ATL depths to MSL
    - This makes ATL depths consistent with S2 composite "average" water level
    
    The correction formula:
        depth_at_MSL = depth_measured + tide_at_measurement
        
    Example: If tide was +0.5m when ICESat-2 measured 3.0m depth,
             the MSL-referenced depth is 3.0 + 0.5 = 3.5m
             (water was higher, so true depth is deeper)
    
    Parameters
    ----------
    lon, lat : float
        Location coordinates
    time_measurement : datetime
        When the depth measurement was taken (e.g., ICESat-2 acquisition)
    time_reference : datetime, optional
        Reference time to normalize to. If None, uses MSL (zero tide).
        For S2 composites, leave as None.
    method : str
        Prediction method: 'auto', 'noaa', 'harmonic'
        
    Returns
    -------
    TidalCorrection
        Correction to ADD to measured depth to get MSL-referenced depth
    """
    # Predict tide at measurement time
    tide_meas = predict_tide(lon, lat, time_measurement, method=method)
    
    # Reference tide
    if time_reference is not None:
        tide_ref = predict_tide(lon, lat, time_reference, method=method)
        ref_height = tide_ref.height_m
    else:
        # Reference to MSL (zero) - appropriate for S2 composites
        ref_height = 0.0
    
    # Correction: positive if measured at high tide (depths appear shallower)
    # depth_MSL = depth_measured + (tide_meas - tide_ref)
    # If tide_meas > 0, water was above MSL, bottom appears shallower, 
    # so true MSL depth is deeper (add positive correction)
    correction = tide_meas.height_m - ref_height
    
    # Combined uncertainty
    unc = tide_meas.uncertainty_m
    if time_reference is not None:
        unc = np.sqrt(tide_meas.uncertainty_m**2 + tide_ref.uncertainty_m**2)
    
    return TidalCorrection(
        correction_m=correction,
        tide_at_measurement=tide_meas.height_m,
        tide_at_reference=ref_height,
        reference_datum=tide_meas.datum,
        uncertainty_m=unc,
        method=tide_meas.method,
    )


def correct_atl_depths_to_msl(
    df: "pd.DataFrame",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    depth_col: str = "depth_m",
    delta_time_col: str = "delta_time",
    method: str = "auto",
) -> "pd.DataFrame":
    """
    Correct ATL depths to Mean Sea Level for use with S2 composites.
    
    S2 median composites aggregate imagery across many dates and tidal states,
    effectively representing Mean Sea Level. This function corrects ICESat-2
    depths (measured at specific tidal states) to MSL for consistency.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ATL points
    lat_col, lon_col : str
        Column names for coordinates
    depth_col : str
        Column name for depth values
    delta_time_col : str
        Column name for ICESat-2 delta_time (seconds since ATLAS epoch)
    method : str
        Tidal prediction method
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - depth_m_msl: MSL-corrected depth
        - tidal_correction_m: Applied correction
        - tide_at_acquisition_m: Tide height when measured
    """
    import pandas as pd
    
    df = df.copy()
    
    # Initialize output columns
    df["depth_m_msl"] = df[depth_col].copy()
    df["tidal_correction_m"] = 0.0
    df["tide_at_acquisition_m"] = np.nan
    
    # Check if delta_time is available
    if delta_time_col not in df.columns or df[delta_time_col].isna().all():
        log.warning("[TIDAL] No delta_time available - skipping tidal correction")
        return df
    
    # Process each point
    n_corrected = 0
    for idx in df.index:
        try:
            lat = float(df.loc[idx, lat_col])
            lon = float(df.loc[idx, lon_col])
            delta_time = df.loc[idx, delta_time_col]
            
            if pd.isna(delta_time):
                continue
            
            # Convert delta_time to datetime
            acq_time = parse_atl_datetime(float(delta_time))
            
            # Compute correction to MSL
            corr = compute_tidal_correction(
                lon=lon,
                lat=lat,
                time_measurement=acq_time,
                time_reference=None,  # MSL
                method=method,
            )
            
            # Apply correction
            df.loc[idx, "depth_m_msl"] = df.loc[idx, depth_col] + corr.correction_m
            df.loc[idx, "tidal_correction_m"] = corr.correction_m
            df.loc[idx, "tide_at_acquisition_m"] = corr.tide_at_measurement
            n_corrected += 1
            
        except Exception as e:
            log.debug(f"Tidal correction failed for point {idx}: {e}")
            continue
    
    log.info(f"[TIDAL] Corrected {n_corrected}/{len(df)} points to MSL")
    
    # Log summary statistics
    if n_corrected > 0:
        corr_vals = df["tidal_correction_m"].dropna()
        log.info(f"[TIDAL] Correction stats: min={corr_vals.min():.3f}m, "
                 f"median={corr_vals.median():.3f}m, max={corr_vals.max():.3f}m")
    
    return df


def correct_depths_for_tide(
    depths_m: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    times: List[datetime],
    reference_time: datetime = None,
    method: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply tidal corrections to an array of depth measurements.
    
    Parameters
    ----------
    depths_m : np.ndarray
        Measured depths (positive down)
    lons, lats : np.ndarray
        Coordinates for each measurement
    times : list of datetime
        Acquisition time for each measurement
    reference_time : datetime, optional
        Reference time to normalize to
    method : str
        Prediction method
        
    Returns
    -------
    corrected_depths : np.ndarray
        Tide-corrected depths
    corrections : np.ndarray
        Applied corrections (for QC)
    """
    n = len(depths_m)
    corrected = np.zeros(n, dtype=np.float32)
    corrections = np.zeros(n, dtype=np.float32)
    
    # Group by similar locations to reduce computation
    # For now, simple loop (could be optimized with spatial binning)
    
    for i in range(n):
        try:
            corr = compute_tidal_correction(
                lon=float(lons[i]),
                lat=float(lats[i]),
                time_measurement=times[i],
                time_reference=reference_time,
                method=method,
            )
            corrections[i] = corr.correction_m
            corrected[i] = depths_m[i] + corr.correction_m
        except Exception as e:
            log.debug(f"Tidal correction failed for point {i}: {e}")
            corrected[i] = depths_m[i]
            corrections[i] = 0.0
    
    return corrected, corrections


def estimate_tidal_uncertainty_for_region(
    lon: float,
    lat: float,
    time_diff_hours: float = 6.0,
) -> float:
    """
    Estimate potential tidal error for a region given time difference.
    
    Parameters
    ----------
    lon, lat : float
        Location
    time_diff_hours : float
        Time difference between measurements (hours)
        
    Returns
    -------
    float
        Estimated maximum tidal error in meters
    """
    range_m, tidal_type = get_regional_tidal_range(lon, lat)
    
    # Maximum error is half the tidal range (high to low)
    max_error = range_m / 2
    
    # For short time differences, error is reduced
    if time_diff_hours < 6:
        # Approximate sinusoidal variation
        max_error *= np.sin(np.pi * time_diff_hours / 12)
    
    return max_error


# -----------------------------------------------------------------------------
# Integration helpers
# -----------------------------------------------------------------------------

def parse_atl_datetime(delta_time: float, atlas_epoch: datetime = None) -> datetime:
    """
    Convert ICESat-2 delta_time to datetime.
    
    Parameters
    ----------
    delta_time : float
        Seconds since ATLAS epoch
    atlas_epoch : datetime, optional
        ATLAS epoch (default: 2018-01-01 00:00:00 UTC)
        
    Returns
    -------
    datetime
    """
    if atlas_epoch is None:
        atlas_epoch = datetime(2018, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    return atlas_epoch + timedelta(seconds=float(delta_time))


def parse_s2_datetime(date_str: str) -> datetime:
    """
    Parse Sentinel-2 acquisition date from filename or metadata.
    
    Parameters
    ----------
    date_str : str
        Date string in various formats
        
    Returns
    -------
    datetime
    """
    # Common S2 formats
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y%m%dT%H%M%S",
        "%Y-%m-%d",
        "%Y%m%d",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str[:len(fmt.replace("%", ""))], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse date: {date_str}")


# -----------------------------------------------------------------------------
# Main entry point for testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tidal prediction utility")
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--time", type=str, help="ISO format datetime (UTC)")
    parser.add_argument("--method", default="auto", choices=["auto", "noaa", "harmonic"])
    
    args = parser.parse_args()
    
    if args.time:
        time_utc = datetime.fromisoformat(args.time).replace(tzinfo=timezone.utc)
    else:
        time_utc = datetime.now(timezone.utc)
    
    print(f"\nLocation: {args.lon:.4f}, {args.lat:.4f}")
    print(f"Time (UTC): {time_utc.isoformat()}")
    print(f"Region: {estimate_region(args.lon, args.lat)}")
    
    range_m, tidal_type = get_regional_tidal_range(args.lon, args.lat)
    print(f"Tidal range: ~{range_m:.1f}m ({tidal_type})")
    
    pred = predict_tide(args.lon, args.lat, time_utc, method=args.method)
    print(f"\nPrediction:")
    print(f"  Height: {pred.height_m:.3f}m ({pred.datum})")
    print(f"  Method: {pred.method}")
    print(f"  Uncertainty: ±{pred.uncertainty_m:.3f}m")
