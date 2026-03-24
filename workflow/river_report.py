"""river_report.py – Structured diagnostics and metadata for the river bathymetry pipeline."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

log = logging.getLogger(__name__)


class RiverReport:
    """
    Structured report generator for river bathymetry pipeline.
    Tracks all steps, inputs, outputs, and diagnostics.
    """
    
    def __init__(self, output_dir: Path, version: str = ""):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = {
            "pipeline": "river_bathymetry",
            "version": version,
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_end": None,
            "duration_seconds": None,
            "status": "running",
            "steps": [],
            "inputs": {},
            "outputs": {},
            "diagnostics": {},
            "errors": []
        }
    
    def add_input(self, key: str, value: Any):
        """Add input parameter or file."""
        self.data["inputs"][key] = value
    
    def add_output(self, key: str, value: Any):
        """Add output file or artifact."""
        self.data["outputs"][key] = value
    
    def add_diagnostic(self, key: str, value: Any):
        """Add diagnostic information."""
        self.data["diagnostics"][key] = value
    
    def add_step(self, step_name: str, status: str, **kwargs):
        """
        Add a pipeline step.
        
        Args:
            step_name: Name of the step
            status: 'running', 'success', 'failed'
            **kwargs: Additional step metadata (command, duration, etc.)
        """
        step = {
            "name": step_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.data["steps"].append(step)
    
    def add_error(self, error_msg: str, step: Optional[str] = None):
        """Add error message."""
        self.data["errors"].append({
            "message": error_msg,
            "step": step,
            "timestamp": datetime.now().isoformat()
        })
    
    def finalize(self, status: str = "success"):
        """
        Finalize the report and write to disk.
        
        Args:
            status: Final status ('success', 'failed', 'partial')
        """
        self.data["timestamp_end"] = datetime.now().isoformat()
        
        # Calculate duration
        if self.data["timestamp_start"]:
            try:
                start = datetime.fromisoformat(self.data["timestamp_start"])
                end = datetime.fromisoformat(self.data["timestamp_end"])
                self.data["duration_seconds"] = (end - start).total_seconds()
            except (ValueError, TypeError) as e:
                log.debug("Could not calculate duration: %s", e)
        
        self.data["status"] = status
        
        # Write to file
        report_path = self.output_dir / "river_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)
        
        log.info("[RIVER REPORT] Written to %s", report_path)
        return report_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Return report as dictionary."""
        return self.data.copy()


def track_soundings_usage(
    soundings_df,
    segments_df,
    calibration_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Track how soundings were used in the pipeline.
    
    Args:
        soundings_df: DataFrame of loaded soundings
        segments_df: DataFrame of river segments
        calibration_results: Results from Dmax calibration
    
    Returns:
        Dictionary of soundings usage statistics
    """
    stats = {
        "soundings_loaded": len(soundings_df) if soundings_df is not None else 0,
        "soundings_sources": [],
        "soundings_spatial_extent": None,
        "soundings_depth_range": None,
        "calibration": {
            "method": "percentile_based",
            "segments_total": len(segments_df) if segments_df is not None else 0,
            "segments_calibrated": 0,
            "segments_default": 0,
            "dmax_statistics": {}
        },
        "usage_summary": "soundings_for_dmax_calibration_only",
        "ml_shape_learning": False,
    }
    
    if soundings_df is not None and len(soundings_df) > 0:
        # Spatial extent
        try:
            stats["soundings_spatial_extent"] = {
                "lon_min": float(soundings_df['longitude'].min()),
                "lon_max": float(soundings_df['longitude'].max()),
                "lat_min": float(soundings_df['latitude'].min()),
                "lat_max": float(soundings_df['latitude'].max())
            }
        except (KeyError, ValueError, TypeError):
            pass
        
        # Depth range
        try:
            depths = soundings_df['depth_m'].values
            depths = depths[np.isfinite(depths)]
            if len(depths) > 0:
                stats["soundings_depth_range"] = {
                    "min": float(depths.min()),
                    "max": float(depths.max()),
                    "mean": float(depths.mean()),
                    "median": float(np.median(depths))
                }
        except (KeyError, ValueError, TypeError):
            pass
        
        # Source tracking
        if 'source' in soundings_df.columns:
            stats["soundings_sources"] = soundings_df['source'].unique().tolist()
    
    # Calibration statistics
    if calibration_results:
        stats["calibration"]["segments_calibrated"] = calibration_results.get("n_calibrated", 0)
        stats["calibration"]["segments_default"] = calibration_results.get("n_default", 0)
        
        if "dmax_values" in calibration_results:
            dmax_vals = np.array(calibration_results["dmax_values"])
            dmax_vals = dmax_vals[np.isfinite(dmax_vals)]
            if len(dmax_vals) > 0:
                stats["calibration"]["dmax_statistics"] = {
                    "min": float(dmax_vals.min()),
                    "max": float(dmax_vals.max()),
                    "mean": float(dmax_vals.mean()),
                    "median": float(np.median(dmax_vals))
                }
    
    return stats


def create_network_metadata(
    network_gpkg: Path,
    dem_path: Path,
    aoi_bounds: tuple,
    threshold_km2: float
) -> Dict[str, Any]:
    """
    Create metadata for river network generation.
    
    Args:
        network_gpkg: Path to network GeoPackage
        dem_path: Path to input DEM
        aoi_bounds: (W, E, S, N) bounds
        threshold_km2: Drainage area threshold
    
    Returns:
        Network metadata dictionary
    """
    metadata = {
        "step": "network_generation",
        "inputs": {
            "dem": str(dem_path),
            "aoi": list(aoi_bounds),
            "threshold_km2": threshold_km2
        },
        "outputs": {
            "network_gpkg": str(network_gpkg)
        }
    }
    
    # Try to read network stats
    if network_gpkg.exists():
        try:
            import geopandas as gpd
            try:
                layers = gpd.list_layers(network_gpkg)
                layer_names = [str(v) for v in layers["name"].tolist()] if layers is not None else []
            except Exception:
                log.debug("create_network_metadata: suppressed exception", exc_info=True)
                layer_names = []
            preferred = next((lyr for lyr in ["rivers_clip", "rivers_aoi", "rivers", "graph_edges"] if lyr in layer_names), None)
            network = gpd.read_file(network_gpkg, layer=preferred) if preferred is not None else gpd.read_file(network_gpkg)
            
            metadata["statistics"] = {
                "segments_count": len(network),
                "total_length_km": float(network.geometry.length.sum() / 1000),
                "segments_with_order": (network['stream_order'] > 0).sum() if 'stream_order' in network.columns else 0
            }
            
            if 'stream_order' in network.columns:
                metadata["statistics"]["max_stream_order"] = int(network['stream_order'].max())
        except Exception as e:
            log.warning("[RIVER REPORT] Could not read network stats: %s", e)
    
    return metadata


def create_xs_metadata(
    xs_gpkg: Path,
    network_gpkg: Path,
    spacing_m: float,
    width_m: float
) -> Dict[str, Any]:
    """
    Create metadata for cross-section generation.
    
    Args:
        xs_gpkg: Path to cross-sections GeoPackage
        network_gpkg: Path to network GeoPackage
        spacing_m: Spacing between cross-sections
        width_m: Width of cross-sections
    
    Returns:
        Cross-section metadata dictionary
    """
    metadata = {
        "step": "cross_section_generation",
        "inputs": {
            "network_gpkg": str(network_gpkg),
            "spacing_m": spacing_m,
            "width_m": width_m
        },
        "outputs": {
            "xs_gpkg": str(xs_gpkg)
        }
    }
    
    # Try to read XS stats
    if xs_gpkg.exists():
        try:
            import geopandas as gpd
            try:
                xs = gpd.read_file(xs_gpkg, layer="xs_lines")
            except Exception:
                log.debug("create_xs_metadata: suppressed exception", exc_info=True)
                xs = gpd.read_file(xs_gpkg)
            
            metadata["statistics"] = {
                "cross_sections_count": len(xs),
                "segments_with_xs": xs['segment_id'].nunique() if 'segment_id' in xs.columns else 0
            }
            
            if 'width_m' in xs.columns:
                widths = xs['width_m'].values
                widths = widths[np.isfinite(widths)]
                if len(widths) > 0:
                    metadata["statistics"]["width_range_m"] = {
                        "min": float(widths.min()),
                        "max": float(widths.max()),
                        "mean": float(widths.mean())
                    }
        except Exception as e:
            log.warning("[RIVER REPORT] Could not read XS stats: %s", e)
    
    return metadata


def create_inference_metadata(
    bathy_raster: Path,
    xs_gpkg: Path,
    soundings_stats: Dict[str, Any],
    method: str = "hydraulic_geometry"
) -> Dict[str, Any]:
    """
    Create metadata for bathymetry inference.
    
    Args:
        bathy_raster: Path to output bathymetry raster
        xs_gpkg: Path to cross-sections GeoPackage
        soundings_stats: Soundings usage statistics
        method: Interpolation method used
    
    Returns:
        Inference metadata dictionary
    """
    metadata = {
        "step": "bathymetry_inference",
        "method": method,
        "inputs": {
            "xs_gpkg": str(xs_gpkg)
        },
        "outputs": {
            "bathy_raster": str(bathy_raster)
        },
        "soundings_usage": soundings_stats
    }
    
    # Try to read raster stats
    if bathy_raster.exists():
        try:
            import rasterio
            with rasterio.open(bathy_raster) as src:
                metadata["raster_properties"] = {
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "dtype": str(src.dtypes[0]),
                    "crs": str(src.crs),
                    "bounds": list(src.bounds),
                    "resolution": list(src.res)
                }
                
                # Sample depth statistics
                try:
                    data = src.read(1, masked=True)
                    valid_data = data.compressed()
                    if len(valid_data) > 0:
                        metadata["depth_statistics"] = {
                            "min": float(valid_data.min()),
                            "max": float(valid_data.max()),
                            "mean": float(valid_data.mean()),
                            "median": float(np.median(valid_data)),
                            "valid_pixels": int(len(valid_data)),
                            "total_pixels": int(src.width * src.height)
                        }
                except (ValueError, IndexError) as e:
                    log.debug("Could not compute raster statistics: %s", e)
        except Exception as e:
            log.warning("[RIVER REPORT] Could not read raster stats: %s", e)
    
    return metadata


def generate_contract_tests_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate contract test results - verify the pipeline did what it claimed.
    
    Args:
        report_data: Full river report data
    
    Returns:
        Dictionary of test results
    """
    tests = {
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    # Test 1: Pipeline completed all steps
    tests["tests"].append({
        "name": "pipeline_completion",
        "passed": all(step.get("status") == "success" for step in report_data.get("steps", [])),
        "message": "All pipeline steps completed successfully"
    })
    
    # Test 2: Soundings were loaded if provided
    soundings_diag = report_data.get("diagnostics", {}).get("soundings_usage", {})
    if soundings_diag.get("soundings_loaded", 0) > 0:
        tests["tests"].append({
            "name": "soundings_loaded",
            "passed": True,
            "message": f"Loaded {soundings_diag['soundings_loaded']} soundings",
            "value": soundings_diag['soundings_loaded']
        })
        
        # Test 3: Soundings were used for calibration
        cal_stats = soundings_diag.get("calibration", {})
        segments_calibrated = cal_stats.get("segments_calibrated", 0)
        tests["tests"].append({
            "name": "soundings_used_for_calibration",
            "passed": segments_calibrated > 0,
            "message": f"Soundings used to calibrate {segments_calibrated} segments",
            "value": segments_calibrated
        })
    
    # Test 4: Output raster exists and has data
    output_raster = report_data.get("outputs", {}).get("bathy_raster")
    if output_raster:
        raster_path = Path(output_raster)
        tests["tests"].append({
            "name": "output_raster_exists",
            "passed": raster_path.exists() and raster_path.stat().st_size > 0,
            "message": f"Output raster created: {output_raster}"
        })
    
    # Test 5: Network generation produced segments
    network_stats = next((s for s in report_data.get("steps", []) 
                         if s.get("name") == "network_generation"), {}).get("statistics", {})
    if network_stats:
        segments_count = network_stats.get("segments_count", 0)
        tests["tests"].append({
            "name": "network_segments_generated",
            "passed": segments_count > 0,
            "message": f"Generated {segments_count} river segments",
            "value": segments_count
        })
    
    # Calculate pass rate
    passed = sum(1 for t in tests["tests"] if t.get("passed", False))
    total = len(tests["tests"])
    tests["summary"] = {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0.0
    }
    
    return tests
