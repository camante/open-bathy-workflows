#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validation.py - Centralized Input Validation for SDB Pipeline

This module provides comprehensive validation of all pipeline inputs BEFORE
processing begins. Fail fast with clear error messages.

Usage:
    from validation import validate_pipeline_config, ValidationResult
    
    result = validate_pipeline_config(config_dict)
    if not result.is_valid:
        for error in result.errors:
            log.info("ERROR: %s", error)
        sys.exit(1)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

# Import centralized constants
try:
    from constants import (
        MAX_VALID_SDB_DEPTH_M,
        MAX_VALID_RIVER_DEPTH_M,
        MIN_DEPTH_DEFAULT,
        HYDRAULIC_MIN_DEPTH_M,
        HYDRAULIC_MAX_DEPTH_M,
    )
except ImportError:
    MAX_VALID_SDB_DEPTH_M = 50.0
    MAX_VALID_RIVER_DEPTH_M = 30.0
    MIN_DEPTH_DEFAULT = 0.5
    HYDRAULIC_MIN_DEPTH_M = 0.5
    HYDRAULIC_MAX_DEPTH_M = 15.0

# Import custom errors
try:
    from errors import ConfigError, ValidationError, DataError
except ImportError:
    class ConfigError(Exception):
        pass
    class ValidationError(Exception):
        pass
    class DataError(Exception):
        pass


@dataclass
class ValidationResult:
    """Result of validation checks."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str) -> None:
        """Add an error and mark result as invalid."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)
    
    def merge(self, other: "ValidationResult") -> None:
        """Merge another ValidationResult into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.update(other.info)
    
    def raise_if_invalid(self) -> None:
        """Raise ConfigError if validation failed."""
        if not self.is_valid:
            error_msg = "; ".join(self.errors)
            raise ConfigError(f"Validation failed: {error_msg}")
    
    def log_results(self, logger: logging.Logger = None) -> None:
        """Log all validation results."""
        _log = logger or log
        
        if self.errors:
            _log.error("=" * 60)
            _log.error("VALIDATION ERRORS:")
            for err in self.errors:
                _log.error(f"  ✗ {err}")
            _log.error("=" * 60)
        
        if self.warnings:
            _log.warning("-" * 60)
            _log.warning("VALIDATION WARNINGS:")
            for warn in self.warnings:
                _log.warning(f"  ⚠ {warn}")
            _log.warning("-" * 60)


# =============================================================================
# AOI Validation
# =============================================================================

def validate_aoi(aoi: Union[str, List[float], Tuple[float, ...]], 
                 result: ValidationResult = None) -> ValidationResult:
    """
    Validate AOI (Area of Interest) bounds.
    
    Args:
        aoi: AOI as "W/E/S/N" string or [W, E, S, N] list/tuple
        result: Optional existing ValidationResult to append to
        
    Returns:
        ValidationResult with validation outcome
    """
    if result is None:
        result = ValidationResult()
    
    # Parse AOI
    try:
        if isinstance(aoi, str):
            parts = [float(x.strip()) for x in aoi.split("/")]
        elif isinstance(aoi, (list, tuple)):
            parts = [float(x) for x in aoi]
        else:
            result.add_error(f"AOI must be string 'W/E/S/N' or list [W,E,S,N], got {type(aoi)}")
            return result
        
        if len(parts) != 4:
            result.add_error(f"AOI must have exactly 4 values (W/E/S/N), got {len(parts)}")
            return result
        
        w, e, s, n = parts
        
    except ValueError as err:
        result.add_error(f"AOI values must be numeric: {err}")
        return result
    
    # Validate longitude range
    if not (-180 <= w <= 180):
        result.add_error(f"West longitude {w} out of range [-180, 180]")
    if not (-180 <= e <= 180):
        result.add_error(f"East longitude {e} out of range [-180, 180]")
    
    # Validate latitude range
    if not (-90 <= s <= 90):
        result.add_error(f"South latitude {s} out of range [-90, 90]")
    if not (-90 <= n <= 90):
        result.add_error(f"North latitude {n} out of range [-90, 90]")
    
    # Validate ordering
    if w >= e:
        result.add_error(f"West ({w}) must be less than East ({e})")
    if s >= n:
        result.add_error(f"South ({s}) must be less than North ({n})")
    
    # Check for reasonable AOI size
    if result.is_valid:
        width_deg = e - w
        height_deg = n - s
        area_deg2 = width_deg * height_deg
        
        # Warn if AOI is very large (> ~100km x 100km at equator)
        if area_deg2 > 1.0:
            result.add_warning(
                f"Large AOI ({width_deg:.2f}° x {height_deg:.2f}°). "
                "Processing may take significant time and memory."
            )
        
        # Warn if AOI is very small
        if area_deg2 < 0.0001:
            result.add_warning(
                f"Very small AOI ({width_deg:.6f}° x {height_deg:.6f}°). "
                "May not contain sufficient training data."
            )
        
        result.info["aoi_parsed"] = {"west": w, "east": e, "south": s, "north": n}
        result.info["aoi_area_deg2"] = area_deg2
    
    return result


# =============================================================================
# Date Range Validation
# =============================================================================

def validate_date_range(start_date: str, end_date: str,
                        result: ValidationResult = None) -> ValidationResult:
    """
    Validate date range for satellite imagery acquisition.
    
    Args:
        start_date: Start date as ISO format string (YYYY-MM-DD)
        end_date: End date as ISO format string (YYYY-MM-DD)
        result: Optional existing ValidationResult
        
    Returns:
        ValidationResult
    """
    if result is None:
        result = ValidationResult()
    
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    
    # Validate format
    if not date_pattern.match(start_date):
        result.add_error(f"start_date '{start_date}' must be YYYY-MM-DD format")
        return result
    
    if not date_pattern.match(end_date):
        result.add_error(f"end_date '{end_date}' must be YYYY-MM-DD format")
        return result
    
    # Parse dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as err:
        result.add_error(f"Invalid date: {err}")
        return result
    
    # Validate ordering
    if end < start:
        result.add_error(f"end_date ({end_date}) must be after start_date ({start_date})")
        return result
    
    # Calculate date range
    days = (end - start).days
    
    # Warnings for edge cases
    if days == 0:
        result.add_warning("Single-day date range may not yield sufficient imagery")
    elif days > 365 * 3:
        result.add_warning(f"Date range spans {days} days (>3 years). Consider narrowing.")
    
    # Check if dates are in the future
    now = datetime.now()
    if start > now:
        result.add_warning(f"start_date ({start_date}) is in the future")
    if end > now:
        result.add_warning(f"end_date ({end_date}) is in the future")
    
    # Check for Sentinel-2 availability (launched April 2015)
    s2_launch = datetime(2015, 6, 23)
    if start < s2_launch:
        result.add_warning(
            f"start_date ({start_date}) is before Sentinel-2A launch (2015-06-23). "
            "No S2 imagery will be available before this date."
        )
    
    result.info["date_range_days"] = days
    
    return result


# =============================================================================
# Depth Parameter Validation
# =============================================================================

def validate_depth_params(
    max_depth_sdb: Union[str, float, None] = None,
    rmse_target: float = None,
    depth_bin_m: float = None,
    result: ValidationResult = None
) -> ValidationResult:
    """
    Validate depth-related parameters.
    
    Args:
        max_depth_sdb: Maximum SDB depth (float or "auto")
        rmse_target: Target RMSE for auto depth-of-support
        depth_bin_m: Depth bin size for analysis
        result: Optional existing ValidationResult
        
    Returns:
        ValidationResult
    """
    if result is None:
        result = ValidationResult()
    
    # Validate max_depth_sdb
    if max_depth_sdb is not None:
        if isinstance(max_depth_sdb, str):
            if max_depth_sdb.lower().strip() != "auto":
                result.add_error(
                    f"max_depth_sdb must be a number or 'auto', got '{max_depth_sdb}'"
                )
        else:
            try:
                depth = float(max_depth_sdb)
                if depth <= 0:
                    result.add_error(f"max_depth_sdb must be positive, got {depth}")
                elif depth > MAX_VALID_SDB_DEPTH_M:
                    result.add_error(
                        f"max_depth_sdb ({depth}m) exceeds physical limit "
                        f"({MAX_VALID_SDB_DEPTH_M}m) for optical SDB"
                    )
                elif depth > 25:
                    result.add_warning(
                        f"max_depth_sdb ({depth}m) is deep for optical SDB. "
                        "Results may have high uncertainty beyond ~20m."
                    )
            except (TypeError, ValueError):
                result.add_error(f"max_depth_sdb must be numeric, got {type(max_depth_sdb)}")
    
    # Validate rmse_target
    if rmse_target is not None:
        try:
            rmse = float(rmse_target)
            if rmse <= 0:
                result.add_error(f"rmse_target must be positive, got {rmse}")
            elif rmse < 0.1:
                result.add_warning(
                    f"rmse_target ({rmse}m) is very strict. "
                    "May result in very shallow depth-of-support."
                )
            elif rmse > 3.0:
                result.add_warning(
                    f"rmse_target ({rmse}m) is permissive. "
                    "Consider tightening for higher quality outputs."
                )
        except (TypeError, ValueError):
            result.add_error(f"rmse_target must be numeric, got {type(rmse_target)}")
    
    # Validate depth_bin_m
    if depth_bin_m is not None:
        try:
            binsize = float(depth_bin_m)
            if binsize <= 0:
                result.add_error(f"depth_bin_m must be positive, got {binsize}")
            elif binsize < 0.25:
                result.add_warning(
                    f"depth_bin_m ({binsize}m) is very fine. "
                    "May have insufficient samples per bin."
                )
            elif binsize > 5.0:
                result.add_warning(
                    f"depth_bin_m ({binsize}m) is coarse. "
                    "May miss depth-dependent error patterns."
                )
        except (TypeError, ValueError):
            result.add_error(f"depth_bin_m must be numeric, got {type(depth_bin_m)}")
    
    return result


# =============================================================================
# File Path Validation
# =============================================================================

def validate_file_path(
    path: Union[str, Path],
    name: str,
    must_exist: bool = True,
    allowed_extensions: List[str] = None,
    result: ValidationResult = None
) -> ValidationResult:
    """
    Validate a file path.
    
    Args:
        path: Path to validate
        name: Human-readable name for error messages
        must_exist: Whether file must already exist
        allowed_extensions: List of allowed file extensions (e.g., ['.tif', '.tiff'])
        result: Optional existing ValidationResult
        
    Returns:
        ValidationResult
    """
    if result is None:
        result = ValidationResult()
    
    if path is None:
        return result  # None is valid (optional parameter)
    
    try:
        p = Path(path)
    except Exception as err:
        result.add_error(f"{name}: Invalid path '{path}': {err}")
        return result
    
    # Check existence
    if must_exist and not p.exists():
        result.add_error(f"{name}: File not found: {path}")
        return result
    
    # Check extension
    if allowed_extensions:
        ext = p.suffix.lower()
        allowed_lower = [e.lower() if e.startswith('.') else f'.{e.lower()}' 
                        for e in allowed_extensions]
        if ext not in allowed_lower:
            result.add_error(
                f"{name}: Invalid extension '{ext}'. "
                f"Allowed: {', '.join(allowed_extensions)}"
            )
    
    # Check if file is readable (if it exists)
    if p.exists() and not p.is_file():
        result.add_error(f"{name}: Path exists but is not a file: {path}")
    
    return result


def validate_directory(
    path: Union[str, Path],
    name: str,
    must_exist: bool = False,
    create_if_missing: bool = False,
    result: ValidationResult = None
) -> ValidationResult:
    """
    Validate a directory path.
    
    Args:
        path: Directory path to validate
        name: Human-readable name for error messages
        must_exist: Whether directory must already exist
        create_if_missing: Attempt to create if missing
        result: Optional existing ValidationResult
        
    Returns:
        ValidationResult
    """
    if result is None:
        result = ValidationResult()
    
    if path is None:
        return result
    
    try:
        p = Path(path)
    except Exception as err:
        result.add_error(f"{name}: Invalid path '{path}': {err}")
        return result
    
    if p.exists():
        if not p.is_dir():
            result.add_error(f"{name}: Path exists but is not a directory: {path}")
    elif must_exist:
        result.add_error(f"{name}: Directory not found: {path}")
    elif create_if_missing:
        try:
            p.mkdir(parents=True, exist_ok=True)
            result.info[f"{name}_created"] = True
        except Exception as err:
            result.add_error(f"{name}: Could not create directory: {err}")
    
    return result


# =============================================================================
# Raster File Validation
# =============================================================================

def validate_raster(
    path: Union[str, Path],
    name: str,
    expected_crs: str = None,
    expected_bands: int = None,
    result: ValidationResult = None
) -> ValidationResult:
    """
    Validate a raster file.
    
    Args:
        path: Path to raster file
        name: Human-readable name for error messages
        expected_crs: Expected CRS (e.g., "EPSG:4326")
        expected_bands: Expected number of bands
        result: Optional existing ValidationResult
        
    Returns:
        ValidationResult
    """
    if result is None:
        result = ValidationResult()
    
    if path is None:
        return result
    
    # First validate as file
    validate_file_path(
        path, name, must_exist=True, 
        allowed_extensions=['.tif', '.tiff', '.vrt', '.img'],
        result=result
    )
    
    if not result.is_valid:
        return result
    
    # Try to open with rasterio
    try:
        import rasterio
        
        with rasterio.open(path) as ds:
            # Check CRS
            if expected_crs and ds.crs:
                from pyproj import CRS
                actual = CRS.from_user_input(ds.crs)
                expected = CRS.from_user_input(expected_crs)
                if actual != expected:
                    result.add_warning(
                        f"{name}: CRS mismatch. Expected {expected_crs}, "
                        f"got {ds.crs}. Reprojection may be needed."
                    )
            elif ds.crs is None:
                result.add_error(f"{name}: Raster has no CRS defined")
            
            # Check bands
            if expected_bands and ds.count != expected_bands:
                result.add_error(
                    f"{name}: Expected {expected_bands} bands, got {ds.count}"
                )
            
            # Check for valid data
            if ds.width == 0 or ds.height == 0:
                result.add_error(f"{name}: Raster has zero dimension")
            
            result.info[f"{name}_shape"] = (ds.height, ds.width)
            result.info[f"{name}_crs"] = str(ds.crs)
            result.info[f"{name}_bands"] = ds.count
    
    except ImportError:
        result.add_warning(f"{name}: rasterio not available, skipping detailed validation")
    except Exception as err:
        result.add_error(f"{name}: Could not read raster: {err}")
    
    return result


# =============================================================================
# Main Pipeline Config Validation
# =============================================================================

def validate_pipeline_config(config: Dict[str, Any]) -> ValidationResult:
    """
    Comprehensive validation of all pipeline configuration.
    
    This is the main entry point for validation. Call this before
    starting any processing.
    
    Args:
        config: Dictionary containing pipeline configuration
        
    Returns:
        ValidationResult with all validation outcomes
        
    Example:
        config = {
            "aoi": "-76.5/-76.0/38.5/39.0",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "max_depth_sdb": "auto",
            "out_dir": "./output",
        }
        result = validate_pipeline_config(config)
        result.raise_if_invalid()
    """
    result = ValidationResult()
    
    log.info("=" * 60)
    log.info("VALIDATING PIPELINE CONFIGURATION")
    log.info("=" * 60)
    
    # Required: AOI
    if "aoi" not in config:
        result.add_error("Missing required parameter: aoi")
    else:
        validate_aoi(config["aoi"], result)
    
    # Required: Date range
    start = config.get("start_date") or config.get("start")
    end = config.get("end_date") or config.get("end")
    
    if not start or not end:
        result.add_error("Missing required parameters: start_date and end_date")
    else:
        validate_date_range(start, end, result)
    
    # Depth parameters
    validate_depth_params(
        max_depth_sdb=config.get("max_depth_sdb"),
        rmse_target=config.get("rmse_target_sdb"),
        depth_bin_m=config.get("depth_bin_m"),
        result=result
    )
    
    # Output directory
    out_dir = config.get("out_dir") or config.get("output_dir")
    if out_dir:
        validate_directory(
            out_dir, "out_dir", 
            must_exist=False, 
            create_if_missing=True,
            result=result
        )
    
    # Cache directory
    cache_dir = config.get("cache_root") or config.get("cache_dir")
    if cache_dir:
        validate_directory(
            cache_dir, "cache_dir",
            must_exist=False,
            create_if_missing=True,
            result=result
        )
    
    # Optional input files
    if config.get("river_dem"):
        validate_raster(config["river_dem"], "river_dem", result=result)
    
    if config.get("land_mask"):
        validate_raster(config["land_mask"], "land_mask", result=result)
    
    # Validate methods list
    methods = config.get("methods", ["sdb", "river"])
    valid_methods = {"sdb", "river", "measured"}
    for m in methods:
        if m.lower() not in valid_methods:
            result.add_error(f"Unknown method: {m}. Valid: {valid_methods}")
    
    # Validate priority
    priority = config.get("priority", "sdb")
    if priority.lower() not in valid_methods:
        result.add_error(f"Unknown priority: {priority}. Valid: {valid_methods}")
    
    # Validate fusion strategy
    fusion_strategy = config.get("fusion_strategy", "spatial_taper")
    valid_strategies = {"priority", "uncertainty", "average", "blend", 
                       "weighted_overlap", "spatial_taper"}
    if fusion_strategy.lower() not in valid_strategies:
        result.add_error(
            f"Unknown fusion_strategy: {fusion_strategy}. Valid: {valid_strategies}"
        )
    
    # Log results
    result.log_results()
    
    if result.is_valid:
        log.info("✓ Configuration validation PASSED")
    else:
        log.error("✗ Configuration validation FAILED")
    
    log.info("=" * 60)
    
    return result


def validate_sdb_config(args) -> ValidationResult:
    """
    Validate SDB-specific configuration from argparse namespace.
    
    Args:
        args: argparse.Namespace with SDB arguments
        
    Returns:
        ValidationResult
    """
    config = vars(args) if hasattr(args, '__dict__') else dict(args)
    return validate_pipeline_config(config)


def validate_river_config(args) -> ValidationResult:
    """
    Validate river bathymetry configuration from argparse namespace.
    
    Args:
        args: argparse.Namespace with river arguments
        
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    config = vars(args) if hasattr(args, '__dict__') else dict(args)
    
    # River-specific validations
    xs_spacing = config.get("xs_spacing_m", 200)
    if xs_spacing is not None:
        try:
            spacing = float(xs_spacing)
            if spacing <= 0:
                result.add_error(f"xs_spacing_m must be positive, got {spacing}")
            elif spacing < 10:
                result.add_warning(
                    f"xs_spacing_m ({spacing}m) is very fine. "
                    "May create excessive cross-sections."
                )
            elif spacing > 1000:
                result.add_warning(
                    f"xs_spacing_m ({spacing}m) is coarse. "
                    "May miss river features."
                )
        except (TypeError, ValueError):
            result.add_error(f"xs_spacing_m must be numeric, got {type(xs_spacing)}")
    
    xs_length = config.get("xs_length_m", 1000)
    if xs_length is not None:
        try:
            length = float(xs_length)
            if length <= 0:
                result.add_error(f"xs_length_m must be positive, got {length}")
            elif length < 50:
                result.add_warning(
                    f"xs_length_m ({length}m) is short. "
                    "May not capture full channel width."
                )
        except (TypeError, ValueError):
            result.add_error(f"xs_length_m must be numeric, got {type(xs_length)}")
    
    # Merge with general validation
    general = validate_pipeline_config(config)
    result.merge(general)
    
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import sys
    import json
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    if len(sys.argv) < 2:
        log.info("Usage: python validation.py <config.json>")
        log.info("       python validation.py --aoi '-76.5/-76.0/38.5/39.0'")
        sys.exit(1)
    
    if sys.argv[1] == "--aoi":
        # Quick AOI validation
        aoi = sys.argv[2] if len(sys.argv) > 2 else None
        if not aoi:
            log.info("ERROR: --aoi requires an argument")
            sys.exit(1)
        result = validate_aoi(aoi)
        result.log_results()
        sys.exit(0 if result.is_valid else 1)
    
    # Load and validate config file
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        log.info("ERROR: Config file not found: %s", config_path)
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    result = validate_pipeline_config(config)
    sys.exit(0 if result.is_valid else 1)
