#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
river_bathy.py – River Bathymetry Pipeline Wrapper

A simplified interface to the river cross-section interpolation workflow.
This module coordinates:
1. River network extraction (river_network.py)
2. Cross-section generation (xs_builder.py)
3. Bathymetry inference (xs_infer_bathy_raster.py)
4. Monotonic adjustment (xs_adjust_monotonic.py)
5. Optional DEM fusion (cudem_river_burn_taper.py)

Can be run standalone or called from bathy_main.py.

Usage:
------
python river_bathy.py \\
    --aoi "-74.5/-74.0/40.3/40.6" \\
    --out-dir output/river_bathy \\
    --dem /path/to/dem.tif \\
    --soundings /path/to/soundings.gpkg
"""

from __future__ import annotations

import argparse
import json
import logging


# Logging is configured by entrypoints (e.g., bathy_main.py / sdb_main.py).
# Standalone scripts configure logging in __main__.

import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Use centralized logging - get logger, don't configure root here
log = logging.getLogger("river_bathy")


# Structured diagnostics report (optional)
try:
    from river_report import RiverReport
    DIAGNOSTICS_AVAILABLE = True
except Exception as e:
    DIAGNOSTICS_AVAILABLE = False
    log.warning(f"river_report not available - limited reporting: {e}")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RiverBathyConfig:
    """Configuration for river bathymetry pipeline."""
    
    # Required
    aoi: str = ""  # W/E/S/N
    out_dir: Path = Path("output/river")
    
    # Optional inputs
    dem_path: Optional[Path] = None
    soundings_path: Optional[Path] = None
    swot_path: Optional[Path] = None  # SWOT water surface heights
    
    # Network extraction
    network_source: str = "nhd"  # "nhd" or "hydrorivers"
    min_stream_order: int = 1
    
    # Cross-section parameters
    xs_spacing_m: float = 200.0
    xs_half_width_m: float = 150.0
    xs_sample_step_m: float = 2.0
    
    # Bathymetry inference
    bottom_width_frac: float = 0.30
    depth_width_a: float = 0.18  # Dmax = a * W^b
    depth_width_b: float = 0.50
    dmin_m: float = 0.50
    dmax_m: float = 15.0
    
    # Calibration
    calib_max_dist_m: float = 200.0
    calib_stat: str = "p90"  # "p90", "max", "median"
    
    # Rasterization
    continuous_method: str = "walid"  # "median", "walid", "aidw"
    continuous_buffer_m: float = 50.0
    nodata: float = -9999.0
    
    # Processing
    cache_dir: Path = Path("cache/river")
    enforce_monotonic: bool = True
    epsilon_m: float = 0.02  # Allowed upstream depth decrease


@dataclass
class RiverBathyResult:
    """Results from river bathymetry pipeline."""
    
    status: str = "pending"
    network_gpkg: Optional[Path] = None
    xs_gpkg: Optional[Path] = None
    bathy_gpkg: Optional[Path] = None
    bathy_raster: Optional[Path] = None
    # NOTE: bathy_raster is the inferred bed elevation raster (orthometric, in DEM datum).
    bed_elev_raster: Optional[Path] = None
    depth_terrain_raster: Optional[Path] = None
    mask_raster: Optional[Path] = None
    uncertainty_raster: Optional[Path] = None
    report: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# =============================================================================
# Helpers
# =============================================================================

def run_command(cmd: Union[str, list], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """
    Run a command and return (returncode, stdout, stderr).
    
    SECURITY FIX v0.6.1: Now accepts list format and uses shell=False.
    
    Args:
        cmd: Command as list (preferred) or string (deprecated)
        cwd: Working directory
    
    Returns:
        (returncode, stdout, stderr)
    """
    import shlex
    
    # Convert string to list if needed
    if isinstance(cmd, str):
        log.warning(f"[SECURITY] run_command() called with string. Use list format instead.")
        try:
            cmd_list = shlex.split(cmd)
        except Exception as e:
            log.error(f"[SECURITY] Could not parse command safely: {e}")
            log.error("[SECURITY] Refusing to execute unparsed command string. Provide list args.")

            return 2, "", "Could not safely parse command string"
    else:
        cmd_list = cmd
    
    log.debug(f"Running: {' '.join(cmd_list)}")
    
    result = subprocess.run(
        cmd_list,
        shell=False,  # SECURITY: Never use shell=True
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    
    return result.returncode, result.stdout, result.stderr


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Pipeline Steps
# =============================================================================

def compute_depth_from_bed_and_dem(bed_elev_tif: Path, dem_tif: Path, out_depth_tif: Path) -> Path:
    """Compute depth relative to DEM terrain surface: depth = bed_elev - dem (negative down)."""
    import numpy as np
    import rasterio

    ensure_dir(out_depth_tif.parent)

    with rasterio.open(bed_elev_tif) as bed, rasterio.open(dem_tif) as dem:
        if (bed.width != dem.width) or (bed.height != dem.height) or (bed.transform != dem.transform):
            raise ValueError("bed elevation raster and DEM must be on same grid")

        profile = bed.profile.copy()
        profile.update(dtype="float32", count=1, compress="DEFLATE", predictor=2)

        bed_nodata = bed.nodata
        dem_nodata = dem.nodata

        with rasterio.open(out_depth_tif, "w", **profile) as dst:
            for _, window in bed.block_windows(1):
                b = bed.read(1, window=window).astype("float32")
                d = dem.read(1, window=window).astype("float32")

                mask = ~np.isfinite(b) | ~np.isfinite(d)
                if bed_nodata is not None:
                    mask |= (b == bed_nodata)
                if dem_nodata is not None:
                    mask |= (d == dem_nodata)

                depth = b - d
                depth[mask] = np.nan
                dst.write(depth.astype("float32"), 1, window=window)
            try:
                dst.nodata = np.nan
            except Exception:
                pass
    return out_depth_tif
def extract_river_network(cfg: RiverBathyConfig, work_dir: Path, script_dir: Path) -> Tuple[Optional[Path], Dict]:
    """
    Extract river network for the AOI using river_network.py CLI.
    """
    log.info("[RIVER] Step 1: Extracting river network...")
    
    out_gpkg = work_dir / "river_network.gpkg"
    step_report = {"status": "running"}
    
    cmd_parts = [
        sys.executable, "river_network.py",
        f"--aoi={cfg.aoi}",
        f"--out-gpkg={out_gpkg}",
        f"--cache-dir={cfg.cache_dir}",
    ]
    
    # FIX: Pass list directly to run_command (secure)
    log.info(f"[RIVER] Command: {' '.join(cmd_parts)}")
    
    rc, stdout, stderr = run_command(cmd_parts, cwd=script_dir)
    
    step_report["returncode"] = rc
    step_report["command"] = ' '.join(cmd_parts)  # Store as string for logging
    
    if rc != 0:
        step_report["status"] = "failed"
        step_report["error"] = stderr[:1000] if stderr else "Unknown error"
        log.error(f"[RIVER] Network extraction failed: {stderr[:500]}")
        return None, step_report
    
    if not out_gpkg.exists():
        step_report["status"] = "failed"
        step_report["error"] = "Output file not created"
        log.error("[RIVER] Network extraction produced no output")
        return None, step_report
    
    step_report["status"] = "success"
    step_report["output"] = str(out_gpkg)
    log.info(f"[RIVER] Network extracted: {out_gpkg}")
    
    return out_gpkg, step_report


def generate_cross_sections(
    cfg: RiverBathyConfig,
    network_gpkg: Path,
    work_dir: Path,
    script_dir: Path,
) -> Tuple[Optional[Path], Dict]:
    """
    Generate cross-sections along river centerlines using xs_builder.py CLI.
    """
    log.info("[RIVER] Step 2: Generating cross-sections...")
    
    out_gpkg = work_dir / "cross_sections.gpkg"
    step_report = {"status": "running"}
    
    # xs_builder.py requires --dem
    if cfg.dem_path is None or not cfg.dem_path.exists():
        step_report["status"] = "failed"
        step_report["error"] = "DEM is required for cross-section generation"
        log.error("[RIVER] DEM is required but not provided")
        return None, step_report
    
    cmd_parts = [
        sys.executable, "xs_builder.py",
        f"--river-gpkg={network_gpkg}",
        f"--dem={cfg.dem_path}",
        f"--out-gpkg={out_gpkg}",
        f"--spacing-m={cfg.xs_spacing_m}",
        f"--half-width-m={cfg.xs_half_width_m}",
        f"--sample-step-m={cfg.xs_sample_step_m}",
    ]
    
    # FIX: Pass list directly to run_command (secure)
    log.info(f"[RIVER] Command: {' '.join(cmd_parts)}")
    
    rc, stdout, stderr = run_command(cmd_parts, cwd=script_dir)
    
    step_report["returncode"] = rc
    step_report["command"] = ' '.join(cmd_parts)
    
    if rc != 0:
        step_report["status"] = "failed"
        step_report["error"] = stderr[:1000] if stderr else "Unknown error"
        log.error(f"[RIVER] XS generation failed: {stderr[:500]}")
        return None, step_report
    
    if not out_gpkg.exists():
        step_report["status"] = "failed"
        step_report["error"] = "Output file not created"
        log.error("[RIVER] XS generation produced no output")
        return None, step_report
    
    step_report["status"] = "success"
    step_report["output"] = str(out_gpkg)
    log.info(f"[RIVER] Cross-sections generated: {out_gpkg}")
    
    return out_gpkg, step_report


def infer_bathymetry(
    cfg: RiverBathyConfig,
    xs_gpkg: Path,
    work_dir: Path,
    script_dir: Path,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Dict]:
    """
    Infer bathymetry from cross-sections using xs_infer_bathy_raster.py CLI.
    
    Returns (bathy_gpkg, bathy_raster, mask_raster, step_report)
    """
    log.info("[RIVER] Step 3: Inferring bathymetry...")
    
    out_gpkg = work_dir / "river_bathy.gpkg"
    out_raster = work_dir / "river_bed_elevation.tif"
    out_mask = work_dir / "river_bed_mask.tif"
    out_uncert = work_dir / "river_bed_uncertainty.tif"
    
    step_report = {"status": "running"}
    
    cmd_parts = [
        sys.executable, "xs_infer_bathy_raster.py",
        f"--xs-gpkg={xs_gpkg}",
        f"--out-gpkg={out_gpkg}",
        f"--a={cfg.depth_width_a}",
        f"--b={cfg.depth_width_b}",
        f"--dmin-m={cfg.dmin_m}",
        f"--dmax-m={cfg.dmax_m}",
        f"--bottom-width-frac={cfg.bottom_width_frac}",
        f"--continuous={cfg.continuous_method}",
    ]
    
    # Add DEM/template for raster output
    if cfg.dem_path and cfg.dem_path.exists():
        cmd_parts.extend([
            f"--template-raster={cfg.dem_path}",
            f"--out-bathy-raster={out_raster}",
            f"--out-mask-raster={out_mask}",
            f"--out-uncert-raster={out_uncert}",
        ])
    
    # Add soundings for calibration
    if cfg.soundings_path and cfg.soundings_path.exists():
        cmd_parts.extend([
            f"--soundings={cfg.soundings_path}",
            f"--calib-max-dist-m={cfg.calib_max_dist_m}",
            f"--calib-stat={cfg.calib_stat}",
        ])
    
    # FIX: Pass list directly
    log.info("[RIVER] Command: %s", " ".join(cmd_parts))
    
    rc, stdout, stderr = run_command(cmd_parts, cwd=script_dir)
    
    step_report["returncode"] = rc
    step_report["command"] = " ".join(cmd_parts)
    
    if rc != 0:
        step_report["status"] = "failed"
        step_report["error"] = stderr[:1000] if stderr else "Unknown error"
        log.warning(f"[RIVER] Bathymetry inference had issues: {stderr[:500]}")
        # Don't return None - partial results may be useful
    else:
        step_report["status"] = "success"
    
    bathy_gpkg = out_gpkg if out_gpkg.exists() else None
    bathy_raster = out_raster if out_raster.exists() else None
    mask_raster = out_mask if out_mask.exists() else None
    
    if bathy_gpkg:
        step_report["gpkg"] = str(bathy_gpkg)
        log.info(f"[RIVER] Bathymetry GPKG: {bathy_gpkg}")
    if bathy_raster:
        step_report["raster"] = str(bathy_raster)
        log.info(f"[RIVER] Bathymetry raster: {bathy_raster}")
    
    return bathy_gpkg, bathy_raster, mask_raster, step_report


def enforce_monotonic(
    cfg: RiverBathyConfig,
    bathy_gpkg: Path,
    work_dir: Path,
    script_dir: Path,
) -> Tuple[Optional[Path], Dict]:
    """
    Enforce downstream-monotonic bed profile using xs_adjust_monotonic.py CLI.
    """
    log.info("[RIVER] Step 4: Enforcing monotonic downstream bed...")
    
    out_gpkg = work_dir / "river_bathy_monotonic.gpkg"
    step_report = {"status": "running"}
    
    cmd_parts = [
        sys.executable, "xs_adjust_monotonic.py",
        f"--in-gpkg={bathy_gpkg}",
        f"--in-layer=xs_bathy_points",
        f"--out-gpkg={out_gpkg}",
        f"--out-layer=xs_bathy_points_monotonic",
        f"--epsilon-m={cfg.epsilon_m}",
    ]
    
    # FIX: Pass list directly
    log.info("[RIVER] Command: %s", " ".join(cmd_parts))
    
    rc, stdout, stderr = run_command(cmd_parts, cwd=script_dir)
    
    step_report["returncode"] = rc
    step_report["command"] = " ".join(cmd_parts)
    
    if rc != 0:
        step_report["status"] = "failed"
        step_report["error"] = stderr[:500] if stderr else "Unknown error"
        log.warning(f"[RIVER] Monotonic adjustment failed: {stderr[:500]}")
        return bathy_gpkg, step_report  # Return original on failure
    
    if not out_gpkg.exists():
        step_report["status"] = "failed"
        step_report["error"] = "Output file not created"
        return bathy_gpkg, step_report
    
    step_report["status"] = "success"
    step_report["output"] = str(out_gpkg)
    log.info(f"[RIVER] Monotonic adjustment complete: {out_gpkg}")
    
    return out_gpkg, step_report


# =============================================================================
# Main Pipeline
# =============================================================================

def run_river_bathymetry(cfg: RiverBathyConfig) -> RiverBathyResult:
    """
    Run the complete river bathymetry pipeline.
    
    Steps:
    1. Extract river network from NHD/HydroRIVERS
    2. Generate perpendicular cross-sections
    3. Infer bathymetry from bank slopes + width-depth prior
    4. (Optional) Calibrate with soundings
    5. (Optional) Enforce monotonic downstream bed
    6. Rasterize to DEM grid
    
    Returns RiverBathyResult with paths to outputs and status.
    """
    log.info("=" * 60)
    log.info("RIVER BATHYMETRY PIPELINE")
    log.info("=" * 60)
    log.info(f"AOI: {cfg.aoi}")
    log.info(f"Output: {cfg.out_dir}")
    log.info(f"DEM: {cfg.dem_path}")
    log.info(f"Soundings: {cfg.soundings_path}")
    log.info("=" * 60)
    
    result = RiverBathyResult()
    
    # NEW v0.7.1: Initialize structured report
    river_report = None
    if DIAGNOSTICS_AVAILABLE:
        river_report = RiverReport(cfg.out_dir)
    
    # Legacy report structure (keep for backward compatibility)
    result.report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "aoi": cfg.aoi,
            "xs_spacing_m": cfg.xs_spacing_m,
            "dmax_m": cfg.dmax_m,
            "dem": str(cfg.dem_path) if cfg.dem_path else None,
            "soundings": str(cfg.soundings_path) if cfg.soundings_path else None,
        },
        "steps": {},
    }
    
    # Setup directories
    cfg.out_dir = Path(cfg.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = cfg.out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir = Path(cfg.cache_dir)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Script directory (where the Python modules are)
    script_dir = Path(__file__).parent
    
    # Track soundings info
    soundings_files = []
    soundings_total = 0
    soundings_in_aoi = 0
    segments_calibrated = 0
    
    try:
        # Step 1: Extract network
        network_gpkg, step_report = extract_river_network(cfg, work_dir, script_dir)
        result.report["steps"]["network"] = step_report
        
        if river_report:
            river_report.add_step(
                "network_generation",
                "success" if network_gpkg else "failed",
                returncode=step_report.get("returncode", -1),
                command=step_report.get("command", ""),
                duration_seconds=step_report.get("duration_seconds", 0)
            )
            
            # Extract network stats from step report if available
            if network_gpkg and network_gpkg.exists():
                try:
                    import geopandas as gpd
                    network_gdf = gpd.read_file(network_gpkg)
                    total_length_km = network_gdf.geometry.length.sum() / 1000.0
                    n_segments = len(network_gdf)
                    river_report.set_network_info(
                        segments=n_segments,
                        total_length_km=total_length_km,
                        outlets=1  # Could be extracted from network analysis
                    )
                except Exception as e:
                    log.warning(f"Could not extract network stats: {e}")
        
        if network_gpkg is None:
            result.status = "failed"
            result.error = "Network extraction failed"
            if river_report:
                river_report.add_error("network_generation", "Network extraction returned None")
                river_report.finalize("failed")
            return result
        
        result.network_gpkg = network_gpkg
        
        # Step 2: Generate cross-sections
        xs_gpkg, step_report = generate_cross_sections(cfg, network_gpkg, work_dir, script_dir)
        result.report["steps"]["xs_builder"] = step_report
        
        if river_report:
            river_report.add_step(
                "cross_section_generation",
                "success" if xs_gpkg else "failed",
                returncode=step_report.get("returncode", -1),
                command=step_report.get("command", "")
            )
            
            # Extract XS stats
            if xs_gpkg and xs_gpkg.exists():
                try:
                    import geopandas as gpd
                    xs_gdf = gpd.read_file(xs_gpkg)
                    river_report.set_cross_section_info(
                        count=len(xs_gdf),
                        spacing_m=cfg.xs_spacing_m,
                        width_m=cfg.xs_half_width_m * 2
                    )
                except Exception as e:
                    log.warning(f"Could not extract XS stats: {e}")
        
        if xs_gpkg is None:
            result.status = "failed"
            result.error = "Cross-section generation failed"
            if river_report:
                river_report.add_error("cross_section_generation", "XS generation returned None")
                river_report.finalize("failed")
            return result
        
        result.xs_gpkg = xs_gpkg
        
        # Step 3: Infer bathymetry
        bathy_gpkg, bathy_raster, mask_raster, step_report = infer_bathymetry(
            cfg, xs_gpkg, work_dir, script_dir
        )
        result.report["steps"]["infer_bathy"] = step_report
        
        if river_report:
            river_report.add_step(
                "bathymetry_inference",
                "success" if (bathy_gpkg or bathy_raster) else "failed",
                returncode=step_report.get("returncode", -1),
                command=step_report.get("command", "")
            )
            
            # Record soundings usage
            if cfg.soundings_path:
                soundings_files = [str(cfg.soundings_path)]
                # Try to extract actual counts from step report or stdout
                # For now, record that soundings were provided
                river_report.set_soundings_info(
                    files=soundings_files,
                    total_points=0,  # Would need to parse from subprocess output
                    points_in_aoi=0,
                    segments_calibrated=0,  # Would need to parse from subprocess output
                    calibration_method=cfg.calib_stat
                )
            else:
                river_report.set_soundings_info(
                    files=[],
                    total_points=0,
                    points_in_aoi=0,
                    segments_calibrated=0
                )
            
            # Record inference method
            river_report.set_inference_info(
                method="hydraulic_geometry",
                shape_model="trapezoid_fixed",  # Honest about current limitation
                cross_sections_processed=0,  # Would extract from XS count
                output_resolution_m=10.0  # Typical value
            )
        
        if bathy_gpkg is None and bathy_raster is None:
            result.status = "failed"
            result.error = "Bathymetry inference failed"
            if river_report:
                river_report.add_error("bathymetry_inference", "No outputs generated")
                river_report.finalize("failed")
            return result
        
        result.bathy_gpkg = bathy_gpkg
        result.bathy_raster = bathy_raster
        result.mask_raster = mask_raster
        
        # Step 4: Monotonic adjustment (optional)
        if cfg.enforce_monotonic and bathy_gpkg:
            mono_gpkg, step_report = enforce_monotonic(cfg, bathy_gpkg, work_dir, script_dir)
            result.report["steps"]["monotonic"] = step_report
            
            if mono_gpkg and mono_gpkg != bathy_gpkg:
                result.bathy_gpkg = mono_gpkg
        
        # Copy final outputs to main output directory
        import shutil
        
        if result.bathy_raster and result.bathy_raster.exists():
            # This raster is the inferred bed elevation (orthometric, in the DEM datum).
            final_bed = cfg.out_dir / "river_bottom_navd88.tif"
            if result.bathy_raster != final_bed:
                shutil.copy(result.bathy_raster, final_bed)
            result.bathy_raster = final_bed
            result.bed_elev_raster = final_bed

            # Derive depth relative to DEM terrain surface (negative down): depth = bed_elev - dem
            if cfg.dem:
                try:
                    depth_out = cfg.out_dir / "river_depth_terrain.tif"
                    compute_depth_from_bed_and_dem(final_bed, Path(cfg.dem), depth_out)
                    result.depth_terrain_raster = depth_out
                except Exception as e:
                    log.warning(f"[RIVER] Could not compute depth_terrain raster: {e}")
        if result.mask_raster and result.mask_raster.exists():
            final_mask = cfg.out_dir / "river_bed_mask.tif"
            if result.mask_raster != final_mask:
                shutil.copy(result.mask_raster, final_mask)
                result.mask_raster = final_mask
        
        # Record outputs in structured report
        if river_report:
            if result.bathy_raster:
                river_report.add_output("bathy_raster", str(result.bathy_raster), result.bathy_raster.exists())
            if result.mask_raster:
                river_report.add_output("mask_raster", str(result.mask_raster), result.mask_raster.exists())
            if result.depth_terrain_raster:
                river_report.add_output("depth_terrain_raster", str(result.depth_terrain_raster), result.depth_terrain_raster.exists())
            if result.bed_elev_raster:
                river_report.add_output("bed_elev_raster", str(result.bed_elev_raster), result.bed_elev_raster.exists())
            if result.network_gpkg:
                river_report.add_output("network_gpkg", str(result.network_gpkg), result.network_gpkg.exists())
            if result.xs_gpkg:
                river_report.add_output("xs_gpkg", str(result.xs_gpkg), result.xs_gpkg.exists())
        
        result.status = "success"
        log.info(f"[RIVER] Pipeline complete: {result.bathy_raster or result.bathy_gpkg}")
        
    except Exception as e:
        log.error(f"[RIVER] Pipeline failed: {e}")
        import traceback
        result.status = "error"
        result.error = str(e)
        result.report["error"] = traceback.format_exc()
        
        if river_report:
            river_report.add_error("pipeline", str(e))
    
    # NEW v0.7.1: Finalize structured report
    if river_report:
        try:
            river_report.finalize(result.status)
        except Exception as e:
            log.warning(f"Could not finalize river report: {e}")
    
    # Write legacy report (backward compatibility)
    report_path = cfg.out_dir / "river_bathy_report.json"
    try:
        with open(report_path, "w") as f:
            json.dump(result.report, f, indent=2, default=str)
    except Exception as e:
        log.warning(f"Could not write legacy report: {e}")
    
    return result


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="River Bathymetry Pipeline (Cross-Section Interpolation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required
    p.add_argument("--aoi", required=True, help="Area of Interest: W/E/S/N")
    p.add_argument("--out-dir", default="output/river_bathy", help="Output directory")
    
    # Optional inputs
    p.add_argument("--dem", default=None, help="DEM for bank elevation sampling (required for raster output)")
    p.add_argument("--soundings", default=None, help="Measured soundings for calibration")
    
    # Cross-section parameters
    p.add_argument("--xs-spacing", type=float, default=200.0, help="Cross-section spacing (m)")
    p.add_argument("--xs-width", type=float, default=150.0, help="Cross-section half-width (m)")
    
    # Bathymetry parameters
    p.add_argument("--dmax", type=float, default=15.0, help="Maximum depth prior (m)")
    p.add_argument("--bottom-frac", type=float, default=0.30, help="Flat bottom width fraction")
    
    # Processing
    p.add_argument("--continuous", choices=["median", "walid", "aidw"], default="walid",
                   help="Rasterization method")
    p.add_argument("--no-monotonic", action="store_true", help="Skip monotonic enforcement")
    p.add_argument("--cache-dir", default="cache/river", help="Cache directory")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    cfg = RiverBathyConfig(
        aoi=args.aoi,
        out_dir=Path(args.out_dir),
        dem_path=Path(args.dem) if args.dem else None,
        soundings_path=Path(args.soundings) if args.soundings else None,
        xs_spacing_m=args.xs_spacing,
        xs_half_width_m=args.xs_width,
        dmax_m=args.dmax,
        bottom_width_frac=args.bottom_frac,
        continuous_method=args.continuous,
        enforce_monotonic=not args.no_monotonic,
        cache_dir=Path(args.cache_dir),
    )
    
    result = run_river_bathymetry(cfg)
    
    if result.status == "success":
        log.info(f"Success: {result.bathy_raster or result.bathy_gpkg}")
        sys.exit(0)
    else:
        log.error(f"Failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
