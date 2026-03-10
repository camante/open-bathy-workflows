#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
river_diagnostics.py - Diagnostic utilities for River Bathymetry Pipeline

This module provides diagnostic functions for analyzing river bathymetry
processing results, including cross-section analysis, calibration metrics,
and spatial coverage assessments.
"""


import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
        log.warning("geopandas not available; returning empty diagnostics")
        return RiverDiagnostics()
    
    xs_gpkg = Path(xs_gpkg)
    if not xs_gpkg.exists():
        log.warning("XS GPKG not found: %s", xs_gpkg)
        return RiverDiagnostics()
    
    try:
        gdf = gpd.read_file(xs_gpkg, layer=layer)
    except Exception as e:
        log.warning("Failed to read %s/%s: %s", xs_gpkg, layer, e)
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
        log.warning("No depth column found in XS data")
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
        log.warning("rasterio not available")
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
        log.warning("Failed to read raster %s: %s", raster_path, e)
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
            log.debug("report.add diagnostics failed", exc_info=True)
    
    return summary


def create_unified_bathy_report(
    output_dir: Union[str, Path],
    sdb_output: Optional[Union[str, Path]] = None,
    river_output: Optional[Union[str, Path]] = None,
    methods: Optional[List[str]] = None,
    priority: Optional[str] = None,
) -> Path:
    """Create a unified, lightweight report that summarizes the *whole* bathy pipeline.

    This is intentionally dependency-light: it should work even when geopandas is
    not installed. It is safe to call at the end of bathy_main.

    Outputs:
      - <output_dir>/unified_bathy_report.json
      - <output_dir>/unified_bathy_report.md
    """
    import json
    from datetime import datetime, timezone

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sdb_out = Path(sdb_output) if sdb_output is not None else None
    river_out = Path(river_output) if river_output is not None else None

    rep: Dict[str, Any] = {
        "pipeline": "unified_bathy",
        "version": "0.1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "methods": list(methods) if methods is not None else [],
        "priority": str(priority) if priority is not None else None,
        "status": {},
        "artifacts": {},
        "notes": "Derived from bathy_report.json + filesystem discovery; safe when optional deps are missing.",
    }

    # Primary report produced by bathy_main
    main_rep_path = out_dir / "bathy_report.json"
    if main_rep_path.exists():
        try:
            main_rep = json.loads(main_rep_path.read_text(errors="ignore"))
            rep["bathy_report_path"] = str(main_rep_path)
            # Extract high-level statuses if present
            try:
                rep["status"]["sdb"] = main_rep.get("sdb", {}).get("status")
                rep["status"]["river"] = main_rep.get("river", {}).get("status")
                rep["status"]["fusion"] = main_rep.get("fusion", {}).get("status")
            except Exception:
                log.debug("ignored", exc_info=True)
            # Keep the main report embedded for traceability (users can inspect one file)
            rep["bathy_report"] = main_rep
        except Exception as e:
            rep["status"]["bathy_report_read_error"] = str(e)

    def _discover_artifacts(base: Path, kinds: tuple[str, ...] = (".tif", ".tiff", ".gpkg", ".json", ".csv", ".pkl")) -> List[str]:
        if base is None or (not base.exists()):
            return []
        out: List[str] = []
        try:
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.suffix.lower() in kinds:
                    out.append(str(p))
        except Exception:
            return []
        # Avoid huge reports: cap
        return out[:200]

    # SDB outputs: include model + key rasters if present
    if sdb_out is not None and sdb_out.exists():
        rep["artifacts"]["sdb_dir"] = str(sdb_out)
        rep["artifacts"]["sdb_files"] = _discover_artifacts(sdb_out)
        try:
            mm = sdb_out / "model" / "model_meta.json"
            if mm.exists():
                rep["artifacts"]["sdb_model_meta"] = json.loads(mm.read_text(errors="ignore"))
        except Exception:
            log.debug("ignored", exc_info=True)

    # River outputs
    if river_out is not None and river_out.exists():
        rep["artifacts"]["river_dir"] = str(river_out)
        rep["artifacts"]["river_files"] = _discover_artifacts(river_out)

    out_json = out_dir / "unified_bathy_report.json"
    out_md = out_dir / "unified_bathy_report.md"

    try:
        out_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("Failed to write unified JSON report: %s", e)

    # Simple markdown mirror for humans
    try:
        lines = []
        lines.append(f"# Unified bathymetry report")
        lines.append("")
        lines.append(f"- Timestamp (UTC): {rep.get('timestamp_utc')}")
        lines.append(f"- Methods: {', '.join(rep.get('methods') or []) or 'n/a'}")
        lines.append(f"- Priority: {rep.get('priority') or 'n/a'}")
        st = rep.get("status", {}) or {}
        lines.append(f"- Status: SDB={st.get('sdb','n/a')} | River={st.get('river','n/a')} | Fusion={st.get('fusion','n/a')}")
        lines.append("")
        if rep.get("bathy_report_path"):
            lines.append(f"- bathy_report.json: {rep.get('bathy_report_path')}")
        if rep.get("artifacts", {}).get("sdb_dir"):
            lines.append(f"- SDB dir: {rep['artifacts']['sdb_dir']}")
        if rep.get("artifacts", {}).get("river_dir"):
            lines.append(f"- River dir: {rep['artifacts']['river_dir']}")
        lines.append("")
        lines.append("## Discovered artifacts (capped)")
        lines.append("")
        for key in ("sdb_files", "river_files"):
            files = rep.get("artifacts", {}).get(key, [])
            if files:
                lines.append(f"### {key}")
                for fp in files[:50]:
                    lines.append(f"- {fp}")
                if len(files) > 50:
                    lines.append(f"- ... ({len(files)-50} more)")
                lines.append("")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        log.warning("Failed to write unified markdown report: %s", e)

    return out_json

