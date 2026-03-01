#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manning_inversion.py - Manning's Equation Inversion for River Depth Estimation

This module provides depth estimation by inverting Manning's equation using
discharge estimates (from StreamStats, USGS gages, or regional regression),
channel width, and slope.

Scientific Background
---------------------
Manning's equation relates flow velocity to channel geometry and roughness:

    V = (1/n) * R^(2/3) * S^(1/2)

Where:
    V = mean velocity (m/s)
    n = Manning's roughness coefficient
    R = hydraulic radius ≈ D for wide channels (m)
    S = water surface slope (m/m)

Combined with continuity (Q = V * A) and assuming wide rectangular channel:
    Q = (W * D * D^(2/3) * S^(1/2)) / n
    Q = (W * D^(5/3) * S^(1/2)) / n

Solving for depth:
    D = (Q * n / (W * S^(1/2)))^(3/5)

Backwater/Tidal Guard
--------------------
Manning's equation assumes steady, uniform flow. It becomes unreliable in:
- Tidal zones (flow reversal, varying water levels)
- Backwater areas (downstream controls raising water level)
- Very low slopes (S < 0.0001) where flow is not gravity-driven

This module implements a "guard" that reduces confidence in Manning estimates
based on slope magnitude and distance to tidal influence.

StreamStats Integration
----------------------
USGS StreamStats provides regional regression equations for flood quantiles.
The 2-year flood (Q2) approximates bankfull discharge in many regions.

References
----------
- Chow, V.T. (1959). Open-Channel Hydraulics. McGraw-Hill.
- Barnes, H.H. (1967). Roughness characteristics of natural channels.
  USGS Water-Supply Paper 1849.
- Ries, K.G., et al. (2017). StreamStats, version 4. USGS Fact Sheet 2017-3046.
- Dingman, S.L. & Sharma, K.P. (1997). Statistical development and validation
  of discharge equations for natural channels. J. Hydrology, 199, 13-35.
"""


import logging
import math
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import urllib.request
import urllib.parse

import numpy as np

log = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Manning's n by channel type (Barnes, 1967; Chow, 1959)
MANNING_N_BY_TYPE: Dict[str, float] = {
    "clean_straight": 0.030,
    "clean_winding": 0.040,
    "clean_cobble": 0.040,
    "weeds_light": 0.035,
    "weeds_heavy": 0.050,
    "brush_light": 0.050,
    "brush_heavy": 0.100,
    "sand_bed": 0.030,
    "gravel_bed": 0.035,
    "cobble_bed": 0.045,
    "boulder_bed": 0.055,
    "mountain_clean": 0.040,
    "mountain_cobble": 0.050,
    "mountain_boulder": 0.060,
    "floodplain_grass": 0.035,
    "floodplain_brush": 0.070,
    "floodplain_trees": 0.120,
    "default": 0.035,
}

# Regional default Manning's n
MANNING_N_BY_REGION: Dict[str, float] = {
    "coastal_plain": 0.030,
    "piedmont": 0.035,
    "appalachian": 0.040,
    "interior_lowlands": 0.032,
    "great_plains": 0.028,
    "rocky_mountain": 0.050,
    "pacific_coast": 0.040,
    "default": 0.035,
}

# Slope thresholds for backwater/tidal guard
SLOPE_TIDAL_THRESHOLD: float = 0.0001
SLOPE_BACKWATER_THRESHOLD: float = 0.0003
SLOPE_NORMAL_THRESHOLD: float = 0.001

# Distance-to-tide thresholds (meters)
DIST_TO_TIDE_TIDAL: float = 5000.0
DIST_TO_TIDE_TRANSITION: float = 15000.0


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ManningResult:
    """Result of Manning inversion depth estimation."""
    depth_m: float
    uncertainty_m: float
    confidence: float
    discharge_m3s: float
    width_m: float
    slope: float
    manning_n: float
    backwater_flag: bool = False
    tidal_flag: bool = False
    guard_factor: float = 1.0
    discharge_source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth_m": self.depth_m,
            "uncertainty_m": self.uncertainty_m,
            "confidence": self.confidence,
            "discharge_m3s": self.discharge_m3s,
            "width_m": self.width_m,
            "slope": self.slope,
            "manning_n": self.manning_n,
            "backwater_flag": self.backwater_flag,
            "tidal_flag": self.tidal_flag,
            "guard_factor": self.guard_factor,
            "discharge_source": self.discharge_source,
        }


@dataclass
class BackwaterGuardResult:
    """Result of backwater/tidal guard evaluation."""
    is_backwater: bool
    is_tidal: bool
    confidence_factor: float
    reason: str
    slope_factor: float = 1.0
    distance_factor: float = 1.0
    combined_factor: float = 1.0


@dataclass 
class StreamStatsQ2:
    """StreamStats Q2 (2-year flood) estimate."""
    q2_m3s: float
    uncertainty_pct: float
    drainage_area_km2: float
    region: str
    equation_id: Optional[str] = None


# =============================================================================
# LOCAL REGISTRY (sdb_config.json) OVERRIDES FOR Q2 REGRESSIONS
# =============================================================================

_Q2_REGISTRY = None  # type: ignore

def _load_q2_registry() -> dict:
    """Load Q2 regression registry from adjacent sdb_config.json (best-effort)."""
    global _Q2_REGISTRY
    if _Q2_REGISTRY is not None:
        return _Q2_REGISTRY
    _Q2_REGISTRY = {}
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(here, 'sdb_config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            _Q2_REGISTRY = ((cfg.get('river') or {}).get('registry') or {}).get('q2_regressions') or {}
    except Exception:
        _Q2_REGISTRY = {}
    return _Q2_REGISTRY

def _convert_da_to_registry_units(da_km2: float, da_units: str) -> float:
    u = (da_units or 'km2').lower()
    if u in ('km2','sqkm','km^2'):
        return da_km2
    if u in ('mi2','sqmi','mi^2'):
        return da_km2 * 0.386102
    # unknown units -> assume km2
    return da_km2

# =============================================================================
# REGION AUTO-SELECTION (AOI -> USGS "region" key for simplified regressions)
# =============================================================================

_STATE_TO_REGION_DEFAULT = {
    # NOTE: These map US state/territory abbreviations to the *simplified* region keys
    # used by estimate_q2_from_drainage_area(). They are not official USGS "regions".
    # If you have better regionalization for your study area, pass --river-manning-region
    # explicitly (or extend this map via a JSON file; see load_state_region_map()).
    "AL": "coastal_plain",
    "MS": "coastal_plain",
    "LA": "coastal_plain",
    "FL": "coastal_plain",
    "GA": "coastal_plain",
    "SC": "coastal_plain",
    "NC": "piedmont",
    "VA": "piedmont",
    "MD": "piedmont",
    "DE": "coastal_plain",
    "NJ": "piedmont",
    "PA": "appalachian",
    "WV": "appalachian",
    "TN": "appalachian",
    "KY": "appalachian",
    "ME": "new_england",
    "NH": "new_england",
    "VT": "new_england",
    "MA": "new_england",
    "RI": "new_england",
    "CT": "new_england",
    "NY": "new_england",
    # Great Lakes / interior
    "OH": "interior_lowlands",
    "IN": "interior_lowlands",
    "IL": "interior_lowlands",
    "MI": "interior_lowlands",
    "WI": "interior_lowlands",
    "MN": "interior_lowlands",
    # Plains / Rockies / West (very coarse)
    "ND": "great_plains",
    "SD": "great_plains",
    "NE": "great_plains",
    "KS": "great_plains",
    "OK": "great_plains",
    "TX": "great_plains",
    "CO": "rocky_mountain",
    "WY": "rocky_mountain",
    "MT": "rocky_mountain",
    "UT": "rocky_mountain",
    "ID": "rocky_mountain",
    "AZ": "southwest_desert",
    "NM": "southwest_desert",
    "NV": "southwest_desert",
    "CA": "california_coast",
    "OR": "pacific_northwest",
    "WA": "pacific_northwest",
}

def load_state_region_map(json_path: str) -> dict:
    """Load (and validate) a user-supplied state->region map JSON."""
    if not json_path:
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        out = {}
        for k,v in (d or {}).items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            kk = k.strip().upper()
            vv = v.strip().lower().replace(" ", "_").replace("-", "_")
            if len(kk) == 2 and vv:
                out[kk] = vv
        return out
    except Exception:
        return {}

def _census_state_from_latlon(lat: float, lon: float, timeout_s: float = 10.0) -> Optional[str]:
    """Best-effort reverse geocode (US Census Geocoder) to a US state abbreviation.

    This is lightweight and does not require an API key.
    """
    try:
        # https://geocoding.geo.census.gov/geocoder/geographies/coordinates?x={lon}&y={lat}&benchmark=Public_AR_Current&vintage=Current_Current&format=json
        base = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
        q = urllib.parse.urlencode({
            "x": f"{lon:.8f}",
            "y": f"{lat:.8f}",
            "benchmark": "Public_AR_Current",
            "vintage": "Current_Current",
            "format": "json",
        })
        url = f"{base}?{q}"
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            js = json.loads(resp.read().decode("utf-8", "ignore"))
        st = (
            js.get("result", {})
              .get("geographies", {})
              .get("States", [{}])[0]
              .get("STUSAB", None)
        )
        if isinstance(st, str) and len(st.strip()) == 2:
            return st.strip().upper()
    except Exception:
        return None
    return None

def infer_manning_region_from_aoi(
    aoi_bbox: str,
    state_region_map_json: str = None,
    default_region: str = "default",
    timeout_s: float = 10.0,
) -> Tuple[str, Optional[str]]:
    """Infer a coarse Manning region key from an AOI bbox string "lon0/lon1/lat0/lat1".

    Returns:
        (region_key, state_abbrev_or_None)
    """
    try:
        parts = [float(p) for p in str(aoi_bbox).replace(",", "/").split("/") if str(p).strip() != ""]
        if len(parts) != 4:
            return (default_region, None)
        lon0, lon1, lat0, lat1 = parts
        lon_c = 0.5 * (lon0 + lon1)
        lat_c = 0.5 * (lat0 + lat1)
    except Exception:
        return (default_region, None)

    st = _census_state_from_latlon(lat_c, lon_c, timeout_s=timeout_s)
    user_map = load_state_region_map(state_region_map_json)
    region = None
    if st and st in user_map:
        region = user_map.get(st)
    if region is None and st:
        region = _STATE_TO_REGION_DEFAULT.get(st)
    if region is None:
        region = default_region
    region = str(region).lower().replace(" ", "_").replace("-", "_")
    return (region, st)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def estimate_manning_n(
    bed_type: str = None,
    region: str = None,
    stream_order: int = None,
    sinuosity: float = None
) -> Tuple[float, float]:
    """
    Estimate Manning's n roughness coefficient.
    
    Returns:
        Tuple of (manning_n, uncertainty)
    """
    n = 0.035
    
    if bed_type:
        bed_key = bed_type.lower().replace(" ", "_").replace("-", "_")
        if bed_key in MANNING_N_BY_TYPE:
            n = MANNING_N_BY_TYPE[bed_key]
        elif "sand" in bed_key:
            n = 0.030
        elif "gravel" in bed_key:
            n = 0.035
        elif "cobble" in bed_key:
            n = 0.045
        elif "boulder" in bed_key:
            n = 0.055
    
    if region:
        region_key = region.lower().replace(" ", "_").replace("-", "_")
        if region_key in MANNING_N_BY_REGION:
            n_region = MANNING_N_BY_REGION[region_key]
            n = 0.5 * n + 0.5 * n_region
    
    if stream_order is not None:
        if stream_order <= 2:
            n *= 1.15
        elif stream_order >= 6:
            n *= 0.90
    
    if sinuosity is not None and sinuosity > 1.0:
        n *= 1.0 + 0.05 * min(sinuosity - 1.0, 1.0)
    
    uncertainty = 0.25 * n
    return (n, uncertainty)


def compute_backwater_guard(
    slope: float,
    distance_to_tide_m: float = None,
    elevation_m: float = None
) -> BackwaterGuardResult:
    """
    Evaluate backwater/tidal guard to adjust Manning confidence.
    """
    is_backwater = False
    is_tidal = False
    reasons = []
    
    # Slope-based guard
    if not np.isfinite(slope) or slope <= 0:
        slope_factor = 0.3
        reasons.append("invalid_slope")
        is_backwater = True
    elif slope < SLOPE_TIDAL_THRESHOLD:
        slope_factor = 0.2
        reasons.append(f"very_low_slope_{slope:.6f}")
        is_tidal = True
        is_backwater = True
    elif slope < SLOPE_BACKWATER_THRESHOLD:
        slope_factor = 0.5
        reasons.append(f"low_slope_{slope:.6f}")
        is_backwater = True
    elif slope < SLOPE_NORMAL_THRESHOLD:
        slope_factor = 0.5 + 0.5 * (slope - SLOPE_BACKWATER_THRESHOLD) / (SLOPE_NORMAL_THRESHOLD - SLOPE_BACKWATER_THRESHOLD)
        reasons.append(f"moderate_slope_{slope:.6f}")
    else:
        slope_factor = 1.0
    
    # Distance-to-tide guard
    distance_factor = 1.0
    if distance_to_tide_m is not None:
        if distance_to_tide_m < DIST_TO_TIDE_TIDAL:
            distance_factor = 0.3 + 0.4 * (distance_to_tide_m / DIST_TO_TIDE_TIDAL)
            is_tidal = True
            reasons.append(f"near_tide_{distance_to_tide_m:.0f}m")
        elif distance_to_tide_m < DIST_TO_TIDE_TRANSITION:
            progress = (distance_to_tide_m - DIST_TO_TIDE_TIDAL) / (DIST_TO_TIDE_TRANSITION - DIST_TO_TIDE_TIDAL)
            distance_factor = 0.7 + 0.3 * progress
            reasons.append(f"transition_zone_{distance_to_tide_m:.0f}m")
    
    # Elevation guard
    if elevation_m is not None and elevation_m < 3.0:
        elev_factor = max(0.5, elevation_m / 3.0)
        distance_factor = min(distance_factor, elev_factor)
        if elevation_m < 1.0:
            is_tidal = True
            reasons.append(f"low_elevation_{elevation_m:.1f}m")
    
    combined_factor = min(slope_factor, distance_factor)
    combined_factor = max(0.1, min(1.0, combined_factor))
    
    reason_str = "; ".join(reasons) if reasons else "normal_flow"
    
    return BackwaterGuardResult(
        is_backwater=is_backwater,
        is_tidal=is_tidal,
        confidence_factor=combined_factor,
        reason=reason_str,
        slope_factor=slope_factor,
        distance_factor=distance_factor,
        combined_factor=combined_factor
    )


def invert_manning_for_depth(
    discharge_m3s: float,
    width_m: float,
    slope: float,
    manning_n: float = 0.035,
    distance_to_tide_m: float = None,
    elevation_m: float = None,
    discharge_source: str = "unknown",
    min_depth_m: float = 0.3,
    max_depth_m: float = 20.0
) -> ManningResult:
    """
    Invert Manning's equation to estimate channel depth.
    
    D = (Q * n / (W * S^0.5))^(3/5)
    """
    # Validate inputs
    if not np.isfinite(discharge_m3s) or discharge_m3s <= 0:
        return ManningResult(
            depth_m=np.nan, uncertainty_m=np.nan, confidence=0.0,
            discharge_m3s=discharge_m3s, width_m=width_m, slope=slope,
            manning_n=manning_n, backwater_flag=True, discharge_source=discharge_source
        )
    
    if not np.isfinite(width_m) or width_m <= 0:
        return ManningResult(
            depth_m=np.nan, uncertainty_m=np.nan, confidence=0.0,
            discharge_m3s=discharge_m3s, width_m=width_m, slope=slope,
            manning_n=manning_n, discharge_source=discharge_source
        )
    
    if not np.isfinite(slope) or slope <= 0:
        d_fallback = 0.18 * (width_m ** 0.5)
        return ManningResult(
            depth_m=d_fallback, uncertainty_m=d_fallback * 0.5, confidence=0.0,
            discharge_m3s=discharge_m3s, width_m=width_m, slope=slope,
            manning_n=manning_n, backwater_flag=True, guard_factor=0.0,
            discharge_source=discharge_source
        )
    
    # Compute backwater/tidal guard
    guard = compute_backwater_guard(slope=slope, distance_to_tide_m=distance_to_tide_m, elevation_m=elevation_m)
    
    # Invert Manning's equation
    sqrt_slope = math.sqrt(slope)
    numerator = discharge_m3s * manning_n
    denominator = width_m * sqrt_slope
    
    if denominator <= 0:
        return ManningResult(
            depth_m=np.nan, uncertainty_m=np.nan, confidence=0.0,
            discharge_m3s=discharge_m3s, width_m=width_m, slope=slope,
            manning_n=manning_n, backwater_flag=True, discharge_source=discharge_source
        )
    
    depth = (numerator / denominator) ** 0.6
    depth = max(min_depth_m, min(max_depth_m, depth))
    
    base_uncertainty = 0.40 * depth
    base_confidence = 0.7
    confidence = base_confidence * guard.confidence_factor
    uncertainty = base_uncertainty / max(0.3, guard.confidence_factor)
    
    return ManningResult(
        depth_m=depth,
        uncertainty_m=uncertainty,
        confidence=confidence,
        discharge_m3s=discharge_m3s,
        width_m=width_m,
        slope=slope,
        manning_n=manning_n,
        backwater_flag=guard.is_backwater,
        tidal_flag=guard.is_tidal,
        guard_factor=guard.confidence_factor,
        discharge_source=discharge_source
    )


# =============================================================================
# STREAMSTATS INTEGRATION
# =============================================================================

def estimate_q2_from_drainage_area(
    drainage_area_km2: float,
    region: str = "default",
    mean_annual_precip_mm: float = None
) -> StreamStatsQ2:
    """
    Estimate Q2 (2-year flood) from drainage area using regional regression.
    """
    if not np.isfinite(drainage_area_km2) or drainage_area_km2 <= 0:
        return StreamStatsQ2(
            q2_m3s=np.nan, uncertainty_pct=100.0,
            drainage_area_km2=drainage_area_km2, region=region
        )
    
    area_mi2 = drainage_area_km2 * 0.386102

    # 1) Prefer explicit published coefficients from local registry (sdb_config.json),
    # keyed by `region` (e.g., 'AL_default', 'NJ_default').
    reg = _load_q2_registry()
    if isinstance(reg, dict) and region in reg and isinstance(reg.get(region), dict):
        ent = reg.get(region) or {}
        try:
            c = float(ent.get('c'))
            fexp = float(ent.get('f'))
            da_u = str(ent.get('da_units','km2'))
            q_u = str(ent.get('q_units','cms')).lower()
            da_val = _convert_da_to_registry_units(drainage_area_km2, da_u)
            q_val = c * (da_val ** fexp)
            # units: expect cms (m^3/s). If c is in cfs, user should set q_units='cfs'.
            if q_u in ('cfs','ft3/s','ft^3/s'):
                q_val = q_val * 0.0283168
            return StreamStatsQ2(
                q2_m3s=float(q_val),
                uncertainty_pct=float(ent.get('uncertainty_pct', 45.0)),
                drainage_area_km2=drainage_area_km2,
                region=region,
                equation_id=str(ent.get('source','registry'))
            )
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    # 2) Fallback: coarse built-in regressions (intended only as a last resort).
    
    region_equations = {
        "coastal_plain": (15.0, 0.75, 40.0),
        "piedmont": (20.0, 0.78, 35.0),
        "appalachian": (25.0, 0.80, 35.0),
        "new_england": (30.0, 0.82, 40.0),
        "interior_lowlands": (18.0, 0.76, 40.0),
        "great_plains": (8.0, 0.70, 50.0),
        "ozarks": (22.0, 0.78, 35.0),
        "rocky_mountain": (15.0, 0.75, 45.0),
        "pacific_northwest": (35.0, 0.85, 40.0),
        "california_coast": (12.0, 0.72, 50.0),
        "southwest_desert": (5.0, 0.65, 60.0),
        "default": (18.0, 0.77, 45.0),
    }
    
    region_key = region.lower().replace(" ", "_").replace("-", "_")
    if region_key not in region_equations:
        region_key = "default"
    
    a, b, unc = region_equations[region_key]
    
    if mean_annual_precip_mm is not None:
        precip_factor = (mean_annual_precip_mm / 1000.0) ** 0.5
        a *= precip_factor
    
    q2_cfs = a * (area_mi2 ** b)
    q2_m3s = q2_cfs * 0.0283168
    
    return StreamStatsQ2(
        q2_m3s=q2_m3s,
        uncertainty_pct=unc,
        drainage_area_km2=drainage_area_km2,
        region=region,
        equation_id=f"{region_key}_simplified"
    )


# =============================================================================
# BLENDING
# =============================================================================

def blend_manning_with_multivariate(
    manning_result: ManningResult,
    multivariate_depth_m: float,
    multivariate_confidence: float = 0.6,
    min_manning_confidence: float = 0.3
) -> Tuple[float, float, float, str]:
    """
    Blend Manning inversion result with multivariate prior.
    
    Returns:
        Tuple of (blended_depth, uncertainty, confidence, method_string)
    """
    manning_valid = (
        np.isfinite(manning_result.depth_m) and
        manning_result.confidence >= min_manning_confidence
    )
    mv_valid = np.isfinite(multivariate_depth_m) and multivariate_confidence > 0
    
    if not manning_valid and not mv_valid:
        return (np.nan, np.nan, 0.0, "none")
    
    if not manning_valid:
        return (multivariate_depth_m, multivariate_depth_m * 0.3, multivariate_confidence, "multivariate_only")
    
    if not mv_valid:
        return (manning_result.depth_m, manning_result.uncertainty_m, manning_result.confidence,
                f"manning_only_{manning_result.discharge_source}")
    
    # Both valid - blend using inverse-variance weighting
    w_manning = manning_result.confidence ** 2
    w_mv = multivariate_confidence ** 2
    w_total = w_manning + w_mv
    w_manning /= w_total
    w_mv /= w_total
    
    blended_depth = w_manning * manning_result.depth_m + w_mv * multivariate_depth_m
    
    var_manning = manning_result.uncertainty_m ** 2
    var_mv = (multivariate_depth_m * 0.3) ** 2
    blended_var = w_manning ** 2 * var_manning + w_mv ** 2 * var_mv
    blended_unc = math.sqrt(blended_var)
    
    blended_conf = w_manning * manning_result.confidence + w_mv * multivariate_confidence
    
    method = f"blend_manning({w_manning:.2f})_mv({w_mv:.2f})"
    if manning_result.backwater_flag:
        method += "_backwater"
    if manning_result.tidal_flag:
        method += "_tidal"
    
    return (blended_depth, blended_unc, blended_conf, method)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for Manning inversion."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Estimate river depth using Manning's equation inversion")
    parser.add_argument("--discharge", type=float, required=True, help="Discharge in m³/s")
    parser.add_argument("--width", type=float, required=True, help="Channel width in meters")
    parser.add_argument("--slope", type=float, required=True, help="Water surface slope (m/m)")
    parser.add_argument("--manning-n", type=float, default=0.035, help="Manning's roughness coefficient")
    parser.add_argument("--dist-tide", type=float, default=None, help="Distance to tidal datum (meters)")
    parser.add_argument("--elevation", type=float, default=None, help="Water surface elevation (meters)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    result = invert_manning_for_depth(
        discharge_m3s=args.discharge,
        width_m=args.width,
        slope=args.slope,
        manning_n=args.manning_n,
        distance_to_tide_m=args.dist_tide,
        elevation_m=args.elevation,
        discharge_source="user_input"
    )
    
    if args.json:
        log.info(json.dumps(result.to_dict(), indent=2))
    else:
        log.info(f"\n{'='*50}")
        log.info("MANNING INVERSION DEPTH ESTIMATE")
        log.info(f"{'='*50}")
        log.info(f"Depth:        {result.depth_m:.2f} m")
        log.info(f"Uncertainty:  ±{result.uncertainty_m:.2f} m")
        log.info(f"Confidence:   {result.confidence:.0%}")
        log.info(f"Backwater:    {'YES' if result.backwater_flag else 'No'}")
        log.info(f"Tidal:        {'YES' if result.tidal_flag else 'No'}")
        log.info(f"Guard Factor: {result.guard_factor:.2f}")
        log.info(f"{'='*50}\n")


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
