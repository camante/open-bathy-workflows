#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
errors.py - Custom exception hierarchy for SDB pipeline

This module provides a structured exception hierarchy for better error
handling, debugging, and user feedback throughout the pipeline.

Usage:
    from errors import DataError, ProcessingError, ConfigError
    
    def load_data(path):
        if not path.exists():
            raise DataError(f"Input file not found: {path}")
        try:
            return gpd.read_file(path)
        except Exception as e:
            raise ProcessingError(f"Failed to read {path}") from e
"""

from typing import Optional, Any


class SDBError(Exception):
    """
    Base exception for all SDB pipeline errors.
    
    All custom exceptions in the pipeline inherit from this class,
    making it easy to catch any pipeline-specific error.
    """
    
    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class ConfigError(SDBError):
    """
    Configuration-related errors.
    
    Raised when:
    - Invalid configuration parameters
    - Missing required configuration
    - Configuration file parsing errors
    - Region configuration mismatches
    
    Example:
        raise ConfigError(
            "Invalid AOI bounds", 
            details={"aoi": aoi_str, "reason": "W > E"}
        )
    """
    pass


class DataError(SDBError):
    """
    Data loading and validation errors.
    
    Raised when:
    - Input files not found
    - Empty datasets
    - Invalid data formats
    - CRS mismatches
    - Missing required fields
    
    Example:
        raise DataError(
            "No ICESat-2 data in AOI",
            details={"aoi": aoi, "date_range": (start, end)}
        )
    """
    pass


class ProcessingError(SDBError):
    """
    General processing failures.
    
    Raised when:
    - Algorithm failures
    - Numerical instabilities
    - Unexpected conditions during processing
    - Resource exhaustion
    
    Example:
        raise ProcessingError(
            "RF training failed",
            details={"n_samples": len(df), "error": str(e)}
        )
    """
    pass


class NetworkError(SDBError):
    """
    Network and API communication errors.
    
    Raised when:
    - API timeouts
    - Connection failures
    - Authentication errors
    - Rate limiting
    
    Example:
        raise NetworkError(
            "Harmony API timeout",
            details={"endpoint": url, "timeout_s": 300}
        )
    """
    pass


class ValidationError(SDBError):
    """
    Data validation errors.
    
    Raised when:
    - Data quality checks fail
    - Inconsistent datasets
    - Out-of-range values
    - Failed sanity checks
    
    Example:
        raise ValidationError(
            "Depth values out of range",
            details={"min": -50, "max": 10, "expected_range": (0, -30)}
        )
    """
    pass


class MemoryError(SDBError):
    """
    Memory-related errors.
    
    Raised when:
    - Insufficient memory for operation
    - Array size exceeds limits
    - Chunked processing required
    
    Example:
        raise MemoryError(
            "AOI too large for memory",
            details={"size_gb": 16.5, "available_gb": 8.0}
        )
    """
    pass


class InterpolationError(ProcessingError):
    """
    Interpolation-specific errors.
    
    Raised when:
    - Insufficient data points
    - Invalid interpolation parameters
    - Convergence failures
    
    Example:
        raise InterpolationError(
            "IDW requires at least 3 points",
            details={"n_points": 2, "method": "walid"}
        )
    """
    pass


class ModelError(ProcessingError):
    """
    Machine learning model errors.
    
    Raised when:
    - Model training fails
    - Model loading fails
    - Prediction errors
    - Feature mismatch
    
    Example:
        raise ModelError(
            "Feature mismatch",
            details={
                "expected": ["B02", "B03", "B04"],
                "got": ["B02", "B03"]
            }
        )
    """
    pass


class GeospatialError(SDBError):
    """
    Geospatial operation errors.
    
    Raised when:
    - CRS transformation fails
    - Geometry operations fail
    - Spatial joins fail
    - Raster alignment issues
    
    Example:
        raise GeospatialError(
            "CRS mismatch",
            details={"source": "EPSG:4326", "target": "EPSG:32617"}
        )
    """
    pass


def validate_aoi(aoi: str) -> tuple:
    """
    Validate AOI string format and bounds.
    
    Args:
        aoi: AOI string in format "W/E/S/N"
    
    Returns:
        Tuple of (west, east, south, north) as floats
    
    Raises:
        ConfigError: If AOI format is invalid
        ValidationError: If bounds are invalid
    """
    try:
        parts = [float(x.strip()) for x in aoi.split("/")]
        if len(parts) != 4:
            raise ConfigError(
                "AOI must have 4 values: W/E/S/N",
                details={"aoi": aoi, "parts": len(parts)}
            )
        
        w, e, s, n = parts
        
        if not (-180 <= w <= 180 and -180 <= e <= 180):
            raise ValidationError(
                "Longitude out of range",
                details={"west": w, "east": e, "valid_range": (-180, 180)}
            )
        
        if not (-90 <= s <= 90 and -90 <= n <= 90):
            raise ValidationError(
                "Latitude out of range",
                details={"south": s, "north": n, "valid_range": (-90, 90)}
            )
        
        if w >= e:
            raise ValidationError(
                "West must be less than East",
                details={"west": w, "east": e}
            )
        
        if s >= n:
            raise ValidationError(
                "South must be less than North",
                details={"south": s, "north": n}
            )
        
        return w, e, s, n
        
    except ValueError as e:
        raise ConfigError(
            "AOI values must be numeric",
            details={"aoi": aoi, "error": str(e)}
        ) from e


def handle_exception(exc: Exception, context: str = "") -> None:
    """
    Log exception with appropriate level based on type.
    
    Args:
        exc: Exception to log
        context: Additional context string
    """
    import logging
    log = logging.getLogger("sdb_pipeline")
    
    if isinstance(exc, SDBError):
        # Our custom errors - log with context
        log.error(f"{context}: {exc}")
        if exc.details:
            log.debug(f"Error details: {exc.details}")
    else:
        # Unexpected errors - log with full traceback
        log.exception(f"{context}: Unexpected error: {exc}")


# Convenience function for wrapping operations
def safe_operation(func, *args, error_class=ProcessingError, 
                   context="Operation failed", **kwargs):
    """
    Wrap a function call with error handling.
    
    Args:
        func: Function to call
        *args: Positional arguments
        error_class: Exception class to raise on failure
        context: Error context message
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    
    Raises:
        error_class: On any exception
    
    Example:
        data = safe_operation(
            gpd.read_file, 
            path,
            error_class=DataError,
            context=f"Loading {path}"
        )
    """
    try:
        return func(*args, **kwargs)
    except SDBError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        # Wrap other exceptions
        raise error_class(
            context,
            details={"function": func.__name__, "error": str(e)}
        ) from e
