#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
river_diagnostics.py - Diagnostic utilities for River Bathymetry Pipeline

This module provides diagnostic functions for analyzing river bathymetry
processing results, including cross-section analysis, calibration metrics,
and spatial coverage assessments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from river_report import RiverReport
    REPORT_AVAILABLE = True
except ImportError:
    RiverReport = None  # type: ignore
    REPORT_AVAILABLE = False

log = logging.getLogger("river_diagnostics")


@dataclass
class XSectionStats:
    """Statistics for a single cross-section."""
    xs_id: str
    width_m: float
    max_depth_m: float
    mean_depth_m: float
    area_m2: float
    n_points: int
    calibrated: bool = False
    calib_rmse_m: Optional[float] = None


@dataclass
class RiverDiagnostics:
    """Container for river bathymetry diagnostics."""
    
    # Cross-section statistics
    n_xs_total: int = 0
    n_xs_calibrated: int = 0
    n_xs_estimated: int = 0
    
    # Depth statistics
    depth_min_m: float = 0.0
    depth_max_m: float = 0.0
    depth_mean_m: float = 0.0
    depth_std_m: float = 0.0
    
    # Coverage statistics
    coverage_km2: float = 0.0
    total_river_length_km: float = 0.0
    
    # Quality metrics
    calibration_rmse_m: Optional[float] = None
    n_soundings_used: int = 0
    
    # Per-XS statistics
    xs_stats: List[XSectionStats] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_xs_total": self.n_xs_total,
            "n_xs_calibrated": self.n_xs_calibrated,
            "n_xs_estimated": self.n_xs_estimated,
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
            "depth_mean_m": self.depth_mean_m,
            "depth_std_m": self.depth_std_m,
            "coverage_km2": self.coverage_km2,
            "total_river_length_km": self.total_river_length_km,
            "calibration_rmse_m": self.calibration_rmse_m,
            "n_soundings_used": self.n_soundings_used,
            "xs_stats": [
                {
                    "xs_id": xs.xs_id,
                    "width_m": xs.width_m,
                    "max_depth_m": xs.max_depth_m,
                    "mean_depth_m": xs.mean_depth_m,
                    "area_m2": xs.area_m2,
                    "n_points": xs.n_points,
                    "calibrated": xs.calibrated,
                    "calib_rmse_m": xs.calib_rmse_m,
                }
                for xs in self.xs_stats
            ],
        }


def compute_xs_diagnostics(
    xs_gpkg: Union[str, Path],
    layer: str = "xs_bathy_points",
) -> RiverDiagnostics:
    """
    Compute diagnostics from cross-section bathymetry GPKG.
    
    Args:
        xs_gpkg: Path to cross-section bathymetry GeoPackage
        layer: Layer name containing bathymetry points
    
    Returns:
        RiverDiagnostics with computed statistics
    """
    if not GEOPANDAS_AVAILABLE:
        log.warning("[DIAG] geopandas not available; returning empty diagnostics")
        return RiverDiagnostics()
    
    xs_gpkg = Path(xs_gpkg)
    if not xs_gpkg.exists():
        log.warning(f"[DIAG] XS GPKG not found: {xs_gpkg}")
        return RiverDiagnostics()
    
    try:
        gdf = gpd.read_file(xs_gpkg, layer=layer)
    except Exception as e:
        log.warning(f"[DIAG] Failed to read {xs_gpkg}/{layer}: {e}")
        return RiverDiagnostics()
    
    if gdf.empty:
        return RiverDiagnostics()
    
    diag = RiverDiagnostics()
    
    # Get depth column (may be 'bed_elev_m' or 'depth_m')
    depth_col = None
    for col in ["depth_m", "bed_depth_m", "bed_elev_m"]:
        if col in gdf.columns:
            depth_col = col
            break
    
    if depth_col is None:
        log.warning("[DIAG] No depth column found in XS data")
        return diag
    
    depths = pd.to_numeric(gdf[depth_col], errors="coerce").dropna()
    
    if len(depths) > 0:
        diag.depth_min_m = float(depths.min())
        diag.depth_max_m = float(depths.max())
        diag.depth_mean_m = float(depths.mean())
        diag.depth_std_m = float(depths.std())
    
    # Count cross-sections
    if "xs_id" in gdf.columns:
        xs_ids = gdf["xs_id"].unique()
        diag.n_xs_total = len(xs_ids)
        
        # Check for calibration info
        if "calib_n" in gdf.columns:
            calib_counts = gdf.groupby("xs_id")["calib_n"].first()
            diag.n_xs_calibrated = int((calib_counts > 0).sum())
            diag.n_xs_estimated = diag.n_xs_total - diag.n_xs_calibrated
    
    return diag


def compute_raster_diagnostics(
    raster_path: Union[str, Path],
    nodata: float = -9999.0,
) -> Dict[str, Any]:
    """
    Compute diagnostics from river bathymetry raster.
    
    Args:
        raster_path: Path to bathymetry raster (bed elevation or depth)
        nodata: Nodata value
    
    Returns:
        Dictionary with raster statistics
    """
    if not RASTERIO_AVAILABLE:
        log.warning("[DIAG] rasterio not available")
        return {}
    
    raster_path = Path(raster_path)
    if not raster_path.exists():
        return {}
    
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            
            # Mask nodata
            valid = np.isfinite(data) & (data != nodata)
            valid_data = data[valid]
            
            if valid_data.size == 0:
                return {"valid_pixels": 0}
            
            # Compute pixel area
            pixel_area_m2 = abs(transform.a * transform.e)
            coverage_km2 = (valid.sum() * pixel_area_m2) / 1e6
            
            return {
                "valid_pixels": int(valid.sum()),
                "total_pixels": int(data.size),
                "valid_fraction": float(valid.sum() / data.size),
                "coverage_km2": float(coverage_km2),
                "min": float(valid_data.min()),
                "max": float(valid_data.max()),
                "mean": float(valid_data.mean()),
                "std": float(valid_data.std()),
                "median": float(np.median(valid_data)),
                "crs": str(crs) if crs else None,
            }
    except Exception as e:
        log.warning(f"[DIAG] Failed to read raster {raster_path}: {e}")
        return {}


def summarize_river_run(
    work_dir: Union[str, Path],
    report: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate summary diagnostics for a river bathymetry run.
    
    Args:
        work_dir: Working directory containing river outputs
        report: Optional RiverReport instance to update
    
    Returns:
        Summary dictionary
    """
    work_dir = Path(work_dir)
    summary: Dict[str, Any] = {"work_dir": str(work_dir)}
    
    # Check for expected outputs
    expected_files = [
        "river_network.gpkg",
        "cross_sections.gpkg",
        "river_bathy.gpkg",
        "river_bed_elev_patch.tif",
    ]
    
    files_found = {}
    for fname in expected_files:
        fpath = work_dir / fname
        files_found[fname] = fpath.exists()
    
    summary["files_found"] = files_found
    summary["all_files_present"] = all(files_found.values())
    
    # Compute raster stats if bed elevation raster exists
    bed_raster = work_dir / "river_bed_elev_patch.tif"
    if bed_raster.exists():
        summary["raster_stats"] = compute_raster_diagnostics(bed_raster)
    
    # Compute XS stats if bathy GPKG exists
    bathy_gpkg = work_dir / "river_bathy.gpkg"
    if bathy_gpkg.exists():
        diag = compute_xs_diagnostics(bathy_gpkg)
        summary["xs_diagnostics"] = diag.to_dict()
    
    # Update report if provided
    if report is not None and REPORT_AVAILABLE and isinstance(report, RiverReport):
        try:
            report.add("diagnostics", summary)
        except Exception:
            pass
    
    return summary

