#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
constants.py - Centralized Constants for SDB + River Bathymetry Pipeline

This module provides:
1. Standardized nodata values across all modules
2. Documented physical constants with literature citations
3. Default parameters with scientific justification

All magic numbers are documented with their sources.
"""

from typing import Dict, FrozenSet, Final
import numpy as np

__version__ = "0.8.0"
PIPELINE_VERSION = "sdb_river_unified_v0.8.0"

# =============================================================================
# NODATA VALUES - Standardized across all modules
# =============================================================================

# Primary nodata value for bathymetry rasters (meters, float32)
# Using -9999.0 as it's outside any physically plausible depth range
NODATA_DEPTH: float = -9999.0
NODATA: float = NODATA_DEPTH  # Backwards compatibility alias

# Nodata for uncertainty rasters (meters, float32)
NODATA_UNCERTAINTY: float = -9999.0

# Nodata for integer mask rasters (uint8)
NODATA_MASK: int = 255

# Nodata for provenance rasters (uint8)
NODATA_PROVENANCE: int = 0

# Land/water mask values (backwards compatibility)
LAND_MASK_VAL: int = 1
WATER_MASK_VAL: int = 0


def is_nodata(value: float, nodata: float = NODATA_DEPTH, tolerance: float = 1e-6) -> bool:
    """Check if a value represents nodata."""
    if not np.isfinite(value):
        return True
    return abs(value - nodata) < tolerance


def mask_nodata(arr: np.ndarray, nodata: float = NODATA_DEPTH) -> np.ndarray:
    """Return boolean mask where True = valid data."""
    return np.isfinite(arr) & (arr != nodata)


# =============================================================================
# SENTINEL-2 CONSTANTS
# =============================================================================

# SCL (Scene Classification Layer) codes
SCL_NODATA: int = 0
SCL_SATURATED_DEFECTIVE: int = 1
SCL_DARK_AREA_PIXELS: int = 2
SCL_CLOUD_SHADOWS: int = 3
SCL_VEGETATION: int = 4
SCL_NOT_VEGETATED: int = 5
SCL_WATER: int = 6
SCL_UNCLASSIFIED: int = 7
SCL_CLOUD_MEDIUM_PROBABILITY: int = 8
SCL_CLOUD_HIGH_PROBABILITY: int = 9
SCL_THIN_CIRRUS: int = 10
SCL_SNOW_ICE: int = 11

SCL_BAD: FrozenSet[int] = frozenset({0, 3, 8, 9, 10, 11})

# Sentinel-2 band center wavelengths (nm)
S2_WAVELENGTHS: Dict[str, int] = {
    "B02": 490,   # Blue
    "B03": 560,   # Green
    "B04": 665,   # Red
    "B08": 842,   # NIR
}


# =============================================================================
# OPTICAL PHYSICS CONSTANTS
# =============================================================================

# Gordon (1975) model coefficients for r_rs to u conversion
# r_rs = g1*u + g2*u² where u = bb/(a+bb)
# Source: Gordon, H.R., et al. (1975). "Computed relationships between the 
#         inherent and apparent optical properties of a flat homogeneous ocean."
#         Applied Optics, 14(2), 417-427.
GORDON_G1: float = 0.0949
GORDON_G2: float = 0.0794

# Refractive index of seawater (typical value at visible wavelengths)
# Source: Mobley, C.D. (1994). "Light and Water: Radiative Transfer in 
#         Natural Waters." Academic Press.
N_WATER: float = 1.34

# Pure water inherent optical properties (m⁻¹)
# Source: Pope, R.M. & Fry, E.S. (1997). "Absorption spectrum (380–700 nm) 
#         of pure water. II. Integrating cavity measurements."
#         Applied Optics, 36(33), 8710-8723.
# And: Morel, A. (1974). "Optical properties of pure water and pure sea water."
#      Optical Aspects of Oceanography, 1, 1-24.
PURE_WATER_ABSORPTION: Dict[str, float] = {
    "B02": 0.0145,   # 490 nm (Blue)
    "B03": 0.0596,   # 560 nm (Green)
    "B04": 0.429,    # 665 nm (Red)
    "B08": 2.87,     # 842 nm (NIR)
}

PURE_WATER_BACKSCATTER: Dict[str, float] = {
    "B02": 0.00144,  # 490 nm
    "B03": 0.00093,  # 560 nm
    "B04": 0.00047,  # 665 nm
    "B08": 0.00017,  # 842 nm
}

# Default Kd factor for optical depth limit
# Source: Lee, Z., et al. (2005). "Euphotic zone depth: Its derivation and 
#         implication to ocean-color remote sensing."
#         Journal of Geophysical Research, 110, C02017.
DEFAULT_KD_FACTOR: float = 2.3

# Default bottom reflectance spectra (normalized albedo)
# Source: Lee Stocking Island measurements from Kim et al. (2024)
# Kim, M., et al. (2024). "Physics-Based Satellite-Derived Bathymetry (SDB) 
#                         Using Landsat OLI Images." Remote Sensing, 16(5), 843.
DEFAULT_SAND_SPECTRUM: Dict[str, float] = {
    "B02": 0.35,  # Blue - moderate
    "B03": 0.42,  # Green - higher  
    "B04": 0.45,  # Red - highest
}

DEFAULT_SEAGRASS_SPECTRUM: Dict[str, float] = {
    "B02": 0.05,  # Blue - low (absorbed by chlorophyll)
    "B03": 0.12,  # Green - moderate (green peak)
    "B04": 0.04,  # Red - very low (absorbed)
}


# =============================================================================
# HYDRAULIC GEOMETRY CONSTANTS
# =============================================================================

# Width-Depth Power Law Coefficients: D_max = a * W^b
# Source: Leopold, L.B. & Maddock, T. (1953). "The hydraulic geometry of 
#         stream channels and some physiographic implications." 
#         U.S. Geological Survey Professional Paper 252.
# These are typical values for natural alluvial channels.
# Users should calibrate for specific river systems.

HYDRAULIC_GEOMETRY_A: float = 0.18
"""
Width-depth coefficient 'a' in D = a * W^b.
Typical range: 0.10 - 0.25 depending on bed material and regime.
- Sandy beds: ~0.15
- Gravel beds: ~0.20
- Cobble/boulder: ~0.25
"""

HYDRAULIC_GEOMETRY_B: float = 0.50
"""
Width-depth exponent 'b' in D = a * W^b.
Theoretical value for stable alluvial channels is 0.5 (square root relationship).
Observed range: 0.35 - 0.60
- Low gradient meandering: ~0.40
- Moderate gradient: ~0.50
- High gradient mountain: ~0.55
"""

# Minimum and maximum depth constraints for hydraulic geometry
HYDRAULIC_MIN_DEPTH_M: float = 0.50
"""Minimum channel depth (m). Prevents unrealistically shallow estimates."""

HYDRAULIC_MAX_DEPTH_M: float = 15.0
"""Maximum channel depth (m) from width-depth relationship alone.
Deeper channels require measured data for calibration."""

# Trapezoid cross-section default flat bottom width fraction
# Source: USACE HEC-RAS Hydraulic Reference Manual
TRAPEZOID_BOTTOM_FRAC: float = 0.30
"""
Fraction of channel width that forms the flat bottom.
Range 0.0 (triangular) to 1.0 (rectangular).
0.30 is typical for natural alluvial channels.
"""


# =============================================================================
# SPATIAL PROCESSING CONSTANTS
# =============================================================================

# Default tile size for memory-efficient raster processing
DEFAULT_TILE_SIZE: int = 1024
"""Tile size in pixels for chunked processing. 1024x1024 balances memory and I/O."""

# Maximum pixels before triggering chunked processing
CHUNK_THRESHOLD_PIXELS: int = 50_000_000
"""~50 million pixels (~200MB at float32). Above this, use tiled processing."""

# Default buffer for spatial operations (meters)
DEFAULT_BUFFER_M: float = 50.0

# Epsilon for numerical stability (avoid division by zero)
EPSILON: float = 1e-6
NUMERICAL_EPS: float = 1e-10
"""Small value to prevent division by zero in ratio calculations."""

# Log-transform epsilon (ensures log(x) is defined)
LOG_EPS: float = 1e-6
"""Added to values before log transform to handle zeros."""


# =============================================================================
# L-INFINITY ESTIMATION DEFAULTS
# =============================================================================

LINF_DEEPWATER_NIR_MAX_DEFAULT: float = 0.03
LINF_DEEPWATER_BRIGHT_MAX_DEFAULT: float = 0.15
LINF_PERCENTILE_DEFAULT: float = 1.0
MIN_DEEPWATER_PIXELS_FOR_LINF: int = 1000


# =============================================================================
# ATL PROCESSING DEFAULTS
# =============================================================================

ATL03_CONFIDENCE_MIN_DEFAULT: int = 4
ATL24_CONFIDENCE_MIN_DEFAULT: float = 0.9


# =============================================================================
# DEPTH LIMITS
# =============================================================================

MAX_DEPTH_SDB_DEFAULT: float = 20.0
MIN_DEPTH_DEFAULT: float = 0.5
MAX_VALID_SDB_DEPTH_M: float = 50.0
MAX_VALID_RIVER_DEPTH_M: float = 30.0

# Depth Convention
# NOTE: The pipeline uses NEGATIVE values for positions below the reference surface.
# A value of -5.0m means the seabed is 5m below the reference (MSL or NAVD88).
DEPTH_CONVENTION: str = "negative_below_datum"
"""
Sign convention used throughout the pipeline:
- negative_below_datum: Values below the reference datum are negative
- Example: -5.0m MSL means seabed is 5m below Mean Sea Level
- This is consistent with elevation conventions (positive up)

SDB OUTPUT VALUES:
- SDB outputs are ELEVATION values relative to MSL (not "depth below water surface")
- A pixel value of -5.0m means: seabed is at -5.0m MSL (5m below MSL)
- MSL is the zero reference due to:
  * Multi-temporal S2 compositing (averages tidal variations)
  * ICESat-2 training data uses EGM2008 orthometric heights ≈ MSL
  
- Use --convert-sdb-to-navd88 to convert from MSL to NAVD88 reference
"""

# =============================================================================
# VERTICAL DATUM CONSTANTS
# =============================================================================

# Common compound CRS codes for vertical datum transformations
EPSG_NAD83_MSL: str = "epsg:4269+5714"      # NAD83 + MSL height
EPSG_NAD83_NAVD88: str = "epsg:4269+5703"   # NAD83 + NAVD88 height
EPSG_NAD83_MLLW: str = "epsg:4269+5866"     # NAD83 + MLLW height (tidal)
EPSG_WGS84_MSL: str = "epsg:4326+5714"      # WGS84 + MSL height
EPSG_WGS84_NAVD88: str = "epsg:4326+5703"   # WGS84 + NAVD88 height

# Vertical datum EPSG codes (vertical component only)
EPSG_MSL: int = 5714           # Mean Sea Level height
EPSG_NAVD88: int = 5703        # NAVD88 height
EPSG_MLLW: int = 5866          # Mean Lower Low Water height
EPSG_EGM2008: int = 3855       # EGM2008 geoid height

# Default vertical datums for different data sources
SDB_TRAINING_VDATUM: str = "MSL"
"""
SDB training data vertical datum reference:
- ICESat-2: EGM2008 orthometric heights ≈ MSL
- Extra XYZ: Should be converted to MSL (e.g., via dlim from MLLW)
- S2 composite: Temporal median averages tidal variations toward MSL

Result: SDB outputs are elevations relative to MSL.
"""

SDB_OUTPUT_TYPE: str = "bed_elevation_relative_to_MSL"
"""
SDB output value type.
- Primary output: Seabed elevation relative to MSL (negative = below MSL)
- Values are elevations, NOT "depth below instantaneous water surface"
- Optional: --convert-sdb-to-navd88 converts to NAVD88 vertical datum
"""

RIVER_OUTPUT_VDATUM: str = "NAVD88"
"""
River bathymetry vertical datum reference.
- River bed elevations derived from DEM (typically NAVD88)
- River depths are relative to water surface from DEM
"""


# =============================================================================
# MODEL TRAINING DEFAULTS
# =============================================================================

RANDOM_FOREST_N_ESTIMATORS: int = 100
RANDOM_SEED_DEFAULT: int = 42
TEST_SIZE_DEFAULT: float = 0.2
MIN_TRAINING_POINTS_DEFAULT: int = 50
MIN_POINTS_FOR_RANSAC: int = 50
MIN_INLIERS_RANSAC: int = 100


# =============================================================================
# QC THRESHOLDS
# =============================================================================

RMSE_TARGET_SDB_DEFAULT: float = 1.0
DOA_THRESHOLD_DEFAULT: float = 0.90


# =============================================================================
# PROVENANCE CODES
# =============================================================================

class Provenance:
    """Provenance codes for multi-source fusion tracking."""
    NODATA: int = 0
    SDB: int = 1
    RIVER: int = 2
    MEASURED: int = 3
    DEM: int = 4
    AVERAGED: int = 5
    BLENDED: int = 6
    EXTRAPOLATED: int = 7
    
    # Gap-fill specific codes (10-19 range)
    GAPFILL_PRIOR_ONLY: int = 10
    GAPFILL_CORRECTED: int = 11
    GAPFILL_MEASURED_EXACT: int = 12
    GAPFILL_RIVER_SMOOTHED: int = 13
    GAPFILL_BANK_CONSTRAINED: int = 14
    
    @classmethod
    def to_description(cls, code: int) -> str:
        """Convert provenance code to human-readable description."""
        mapping = {
            0: "NoData",
            1: "SDB (satellite-derived)",
            2: "River (cross-section interpolation)",
            3: "Measured (sonar/lidar/survey)",
            4: "Existing DEM",
            5: "Averaged (multiple sources)",
            6: "Blended (tapered transition)",
            7: "Extrapolated (physics-based)",
            # Gap-fill codes
            10: "Gap-fill: Prior only (no HQ nearby)",
            11: "Gap-fill: Prior + residual correction",
            12: "Gap-fill: At HQ measurement",
            13: "Gap-fill: River smoothed (anisotropic)",
            14: "Gap-fill: Bank constrained",
        }
        return mapping.get(code, f"Unknown ({code})")


# Backwards compatibility aliases
PROV_NODATA = Provenance.NODATA
PROV_SDB = Provenance.SDB
PROV_RIVER = Provenance.RIVER
PROV_MEASURED = Provenance.MEASURED
PROV_DEM = Provenance.DEM
PROV_AVERAGED = Provenance.AVERAGED
PROV_BLENDED = Provenance.BLENDED


# =============================================================================
# UNCERTAINTY CONSTANTS
# =============================================================================

# Base uncertainty floor (meters) - minimum reportable uncertainty
UNCERTAINTY_FLOOR_M: float = 0.10
"""Minimum uncertainty (m) even for ideal conditions."""

# Uncertainty scaling with depth
UNCERTAINTY_DEPTH_SCALE: float = 0.05
"""Uncertainty increases by this fraction per meter of depth."""

# Extrapolation uncertainty penalty
UNCERTAINTY_EXTRAP_BASE: float = 0.15
"""Base uncertainty (m) added when extrapolating beyond training."""

UNCERTAINTY_EXTRAP_SCALE: float = 0.12
"""Additional uncertainty per meter beyond training depth."""
