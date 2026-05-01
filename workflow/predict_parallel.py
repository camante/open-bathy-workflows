#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_parallel.py - Parallel Tile Processing for SDB Prediction

This module provides parallel processing capability for large-area SDB prediction.
Tiles are processed concurrently using ProcessPoolExecutor for 2-4x speedup on
multi-core systems.

Features:
- Automatic tile partitioning with configurable overlap
- Process-based parallelism (avoids GIL)
- Memory-efficient streaming of results
- Graceful degradation to sequential processing on failure
- Progress tracking with optional tqdm support

Usage:
    from predict_parallel import predict_scene_parallel, ParallelConfig
    
    config = ParallelConfig(
        max_workers=4,
        tile_size=2048,
        overlap=256
    )
    
    result = predict_scene_parallel(
        s2_paths={"B02": "...", "B03": "...", ...},
        model_path="rf_model.pkl",
        meta_path="model_meta.json",
        out_path="sdb_output.tif",
        config=config
    )

Note:
    Models are loaded inside each worker process (sklearn models can't be pickled
    reliably across processes). This adds some overhead but ensures correctness.
"""

import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

# Import constants
try:
    from core.constants import NODATA_DEPTH, DEFAULT_TILE_SIZE
except ImportError:
    NODATA_DEPTH = -9999.0
    DEFAULT_TILE_SIZE = 1024  # Match constants.py value


@dataclass
class ParallelConfig:
    """Configuration for parallel prediction."""
    
    # Number of worker processes (None = auto-detect)
    max_workers: Optional[int] = None
    
    # Tile size in pixels
    tile_size: int = DEFAULT_TILE_SIZE
    
    # Tile overlap in pixels (for seamless blending)
    overlap: int = 256
    
    # Timeout per tile (seconds)
    tile_timeout: float = 300.0
    
    # Whether to show progress bar
    show_progress: bool = True
    
    # Memory limit per worker (GB) - used for auto-sizing
    memory_limit_gb: float = 4.0
    
    # Fall back to sequential on failure
    fallback_sequential: bool = True
    
    # Temporary directory for intermediate files
    temp_dir: Optional[Path] = None
    
    def get_max_workers(self) -> int:
        """Get actual number of workers to use."""
        if self.max_workers is not None:
            return max(1, self.max_workers)
        
        # Auto-detect: use CPU count minus 1, minimum 1
        cpu_count = os.cpu_count() or 4
        return max(1, cpu_count - 1)


@dataclass
class TileSpec:
    """Specification for a single tile."""
    
    tile_id: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    
    # Inner region (without overlap) for final assembly
    inner_row_start: int
    inner_row_end: int
    inner_col_start: int
    inner_col_end: int
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Full tile shape (rows, cols)."""
        return (self.row_end - self.row_start, self.col_end - self.col_start)
    
    @property
    def inner_shape(self) -> Tuple[int, int]:
        """Inner tile shape (rows, cols)."""
        return (
            self.inner_row_end - self.inner_row_start,
            self.inner_col_end - self.inner_col_start
        )


@dataclass
class TileResult:
    """Result from processing a single tile."""
    
    tile_id: int
    success: bool
    data: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


def create_tile_specs(
    height: int,
    width: int,
    tile_size: int,
    overlap: int
) -> List[TileSpec]:
    """
    Create tile specifications for an image.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        tile_size: Tile size in pixels
        overlap: Overlap between tiles in pixels
        
    Returns:
        List of TileSpec objects
    """
    specs = []
    tile_id = 0
    
    # Calculate step size (tile_size - overlap)
    step = tile_size - overlap
    if step <= 0:
        step = tile_size // 2  # Fallback for very large overlap
    
    row = 0
    while row < height:
        col = 0
        while col < width:
            # Full tile bounds (with overlap)
            row_start = row
            row_end = min(row + tile_size, height)
            col_start = col
            col_end = min(col + tile_size, width)
            
            # Inner bounds (without overlap) for assembly
            # First tile in each dimension starts at 0
            # Last tile in each dimension ends at image edge
            inner_row_start = row if row == 0 else row + overlap // 2
            inner_row_end = row_end if row_end == height else row_end - overlap // 2
            inner_col_start = col if col == 0 else col + overlap // 2
            inner_col_end = col_end if col_end == width else col_end - overlap // 2
            
            specs.append(TileSpec(
                tile_id=tile_id,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                inner_row_start=inner_row_start,
                inner_row_end=inner_row_end,
                inner_col_start=inner_col_start,
                inner_col_end=inner_col_end
            ))
            
            tile_id += 1
            col += step
        
        row += step
    
    return specs


def _worker_predict_tile(args: Tuple) -> TileResult:
    """
    Worker function to predict a single tile.
    
    This runs in a separate process. Model is loaded inside the worker
    because sklearn models don't pickle reliably across processes.
    
    Args:
        args: Tuple of (tile_spec, s2_paths, model_path, meta_path, extra_kwargs)
        
    Returns:
        TileResult with prediction data or error
    """
    import time
    start_time = time.time()
    
    tile_spec, s2_paths, model_path, meta_path, extra_kwargs = args
    tile_id = tile_spec.tile_id
    
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import Window
        import joblib
        import json
        
        # Load model
        rf = joblib.load(model_path)
        
        # Load metadata
        with open(meta_path) as f:
            meta = json.load(f)
        
        # Get feature columns from metadata
        feat_cols = meta.get("feature_columns", [
            "log_B02", "log_B03", "log_B04", "log_B08",
            "brightness", "B03_B02", "B04_B03", "nbri", "stumpf_idx"
        ])
        
        # Read tile data from S2 bands
        window = Window(
            tile_spec.col_start,
            tile_spec.row_start,
            tile_spec.col_end - tile_spec.col_start,
            tile_spec.row_end - tile_spec.row_start
        )
        
        bands = {}
        for band_name in ["B02", "B03", "B04", "B08"]:
            if band_name not in s2_paths:
                raise ValueError(f"Missing S2 band path: {band_name}")
            
            with rasterio.open(s2_paths[band_name]) as ds:
                bands[band_name] = ds.read(1, window=window).astype(np.float32)
                nodata = ds.nodata
                if nodata is not None:
                    bands[band_name][bands[band_name] == nodata] = np.nan
        
        # Compute derived features
        eps = 1e-6
        
        # Ensure positive values for log transform
        b02 = np.maximum(bands["B02"], eps)
        b03 = np.maximum(bands["B03"], eps)
        b04 = np.maximum(bands["B04"], eps)
        b08 = np.maximum(bands["B08"], eps)
        
        features = {
            "log_B02": np.log(b02),
            "log_B03": np.log(b03),
            "log_B04": np.log(b04),
            "log_B08": np.log(b08),
            "brightness": (b02 + b03 + b04) / 3.0,
            "B03_B02": b03 / np.maximum(b02, eps),
            "B04_B03": b04 / np.maximum(b03, eps),
        }
        
        # NBRI
        nbri_denom = b03 + b08
        features["nbri"] = np.where(
            np.abs(nbri_denom) > eps,
            (b03 - b08) / nbri_denom,
            0.0
        )
        
        # Stumpf index
        log_b03_safe = np.where(
            np.abs(features["log_B03"]) > eps,
            features["log_B03"],
            np.sign(features["log_B03"]) * eps
        )
        features["stumpf_idx"] = np.clip(
            features["log_B02"] / log_b03_safe,
            -10.0, 10.0
        )
        
        # Create feature array
        h, w = bands["B02"].shape
        n_pixels = h * w
        
        # Build feature matrix
        X = np.zeros((n_pixels, len(feat_cols)), dtype=np.float32)
        for i, col in enumerate(feat_cols):
            if col in features:
                X[:, i] = features[col].ravel()
            elif col in bands:
                X[:, i] = bands[col].ravel()
            else:
                # Unknown feature - fill with zeros
                pass
        
        # Create mask for valid pixels
        valid_mask = np.ones(n_pixels, dtype=bool)
        for band in bands.values():
            valid_mask &= np.isfinite(band.ravel())
        
        # Initialize output
        prediction = np.full(n_pixels, NODATA_DEPTH, dtype=np.float32)
        uncertainty = np.full(n_pixels, NODATA_DEPTH, dtype=np.float32)
        
        # Predict valid pixels
        n_valid = valid_mask.sum()
        if n_valid > 0:
            X_valid = X[valid_mask]
            
            # Get predictions from all trees for uncertainty
            if hasattr(rf, 'estimators_'):
                tree_preds = np.array([
                    tree.predict(X_valid) for tree in rf.estimators_
                ])
                pred_valid = np.mean(tree_preds, axis=0)
                std_valid = np.std(tree_preds, axis=0)
            else:
                pred_valid = rf.predict(X_valid)
                std_valid = np.zeros_like(pred_valid)
            
            # Convert to negative depth
            pred_valid = -np.abs(pred_valid)
            
            prediction[valid_mask] = pred_valid.astype(np.float32)
            uncertainty[valid_mask] = std_valid.astype(np.float32)
        
        # Reshape to 2D
        prediction = prediction.reshape((h, w))
        uncertainty = uncertainty.reshape((h, w))
        
        # Extract inner region (without overlap)
        inner_row_off = tile_spec.inner_row_start - tile_spec.row_start
        inner_col_off = tile_spec.inner_col_start - tile_spec.col_start
        inner_h = tile_spec.inner_row_end - tile_spec.inner_row_start
        inner_w = tile_spec.inner_col_end - tile_spec.inner_col_start
        
        inner_pred = prediction[
            inner_row_off:inner_row_off + inner_h,
            inner_col_off:inner_col_off + inner_w
        ]
        inner_unc = uncertainty[
            inner_row_off:inner_row_off + inner_h,
            inner_col_off:inner_col_off + inner_w
        ]
        
        duration = time.time() - start_time
        
        return TileResult(
            tile_id=tile_id,
            success=True,
            data=inner_pred,
            uncertainty=inner_unc,
            duration_seconds=duration,
            metrics={"valid_pixels": int(n_valid)}
        )
        
    except Exception as e:
        log.debug("predict_parallel: suppressed exception", exc_info=True)
        duration = time.time() - start_time
        return TileResult(
            tile_id=tile_id,
            success=False,
            error=str(e),
            duration_seconds=duration
        )


def predict_scene_parallel(
    s2_paths: Dict[str, str],
    model_path: str,
    meta_path: str,
    out_path: str,
    land_mask_path: Optional[str] = None,
    config: ParallelConfig = None,
    uncertainty_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict SDB for entire scene using parallel tile processing.
    
    Args:
        s2_paths: Dictionary of S2 band paths {"B02": path, "B03": path, ...}
        model_path: Path to trained RF model (.pkl)
        meta_path: Path to model metadata (.json)
        out_path: Output path for SDB raster
        land_mask_path: Optional land mask path
        config: Parallel processing configuration
        uncertainty_path: Optional output path for uncertainty raster
        **kwargs: Additional arguments passed to tile workers
        
    Returns:
        Dictionary with processing statistics
    """
    import rasterio
    
    if config is None:
        config = ParallelConfig()
    
    log.info("Parallel SDB prediction starting.")
    
    # Get image dimensions from first band
    ref_band = s2_paths.get("B02") or list(s2_paths.values())[0]
    with rasterio.open(ref_band) as ds:
        height = ds.height
        width = ds.width
        profile = ds.profile.copy()
        crs = ds.crs
        transform = ds.transform
    
    log.info("Image size: %s x %s pixels", width, height)
    log.info("Tile size: %s, overlap: %s", config.tile_size, config.overlap)
    
    # Create tile specifications
    tiles = create_tile_specs(height, width, config.tile_size, config.overlap)
    n_tiles = len(tiles)
    log.info("Total tiles: %s", n_tiles)
    
    # Get number of workers
    n_workers = config.get_max_workers()
    log.info("Workers: %s", n_workers)
    
    # Initialize output arrays
    output = np.full((height, width), NODATA_DEPTH, dtype=np.float32)
    output_unc = np.full((height, width), NODATA_DEPTH, dtype=np.float32)
    
    # Prepare worker arguments
    worker_args = [
        (tile, s2_paths, model_path, meta_path, kwargs)
        for tile in tiles
    ]
    
    # Process tiles in parallel
    start_time = time.time()
    completed = 0
    failed = 0
    total_valid_pixels = 0
    
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_worker_predict_tile, args): args[0].tile_id
                for args in worker_args
            }
            
            # Create progress bar if available
            if config.show_progress and TQDM_AVAILABLE:
                pbar = tqdm(total=n_tiles, desc="Processing tiles", unit="tile")
            else:
                pbar = None
            
            # Collect results as they complete
            for future in as_completed(futures, timeout=config.tile_timeout * n_tiles):
                tile_id = futures[future]
                
                try:
                    result = future.result(timeout=config.tile_timeout)
                    
                    if result.success:
                        # Write result to output array
                        tile = tiles[result.tile_id]
                        
                        output[
                            tile.inner_row_start:tile.inner_row_end,
                            tile.inner_col_start:tile.inner_col_end
                        ] = result.data
                        
                        if result.uncertainty is not None:
                            output_unc[
                                tile.inner_row_start:tile.inner_row_end,
                                tile.inner_col_start:tile.inner_col_end
                            ] = result.uncertainty
                        
                        completed += 1
                        total_valid_pixels += result.metrics.get("valid_pixels", 0)
                    else:
                        log.warning("Tile %s failed: %s", tile_id, result.error)
                        failed += 1
                        
                except TimeoutError:
                    log.warning("Tile %s timed out", tile_id)
                    failed += 1
                except Exception as e:
                    log.warning("Tile %s exception: %s", tile_id, e)
                    failed += 1
                
                if pbar:
                    pbar.update(1)
            
            if pbar:
                pbar.close()
    
    except Exception as e:
        log.error("Parallel processing failed: %s", e, exc_info=True)
        
        if config.fallback_sequential:
            log.info("Falling back to sequential processing...")
            return _predict_sequential_fallback(
                s2_paths, model_path, meta_path, out_path,
                land_mask_path, uncertainty_path, **kwargs
            )
        else:
            raise
    
    total_duration = time.time() - start_time
    
    # Apply land mask if provided
    if land_mask_path:
        try:
            with rasterio.open(land_mask_path) as mask_ds:
                land_mask = mask_ds.read(1)
                # Assume 0 = water, 1 = land
                output[land_mask == 1] = NODATA_DEPTH
                output_unc[land_mask == 1] = NODATA_DEPTH
        except Exception as e:
            log.warning("Could not apply land mask: %s", e)
    
    # Write output
    profile.update(
        dtype="float32",
        count=1,
        nodata=NODATA_DEPTH,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256
    )
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(output, 1)
        dst.update_tags(
            VALUE_TYPE="depth",
            UNITS="m",
            SIGN_CONVENTION="negative_below_surface",
            PROCESSING="parallel_tile"
        )
    
    log.info("Wrote prediction: %s", out_path)
    
    # Write uncertainty if requested
    if uncertainty_path:
        with rasterio.open(uncertainty_path, "w", **profile) as dst:
            dst.write(output_unc, 1)
            dst.update_tags(
                VALUE_TYPE="uncertainty",
                UNITS="m",
                METHOD="tree_std"
            )
        log.info("Wrote uncertainty: %s", uncertainty_path)
    
    # Summary
    log.info("Completed: %s/%s tiles", completed, n_tiles)
    if failed > 0:
        log.warning("Failed: %s/%s tiles", failed, n_tiles)
    log.info("Total time: %.1fs, %.2fs/tile, %s valid pixels", total_duration, total_duration / max(completed, 1), format(total_valid_pixels, ","))
    
    return {
        "success": failed == 0,
        "output_path": str(out_path),
        "uncertainty_path": uncertainty_path,
        "tiles_total": n_tiles,
        "tiles_completed": completed,
        "tiles_failed": failed,
        "valid_pixels": total_valid_pixels,
        "duration_seconds": total_duration,
        "workers": n_workers
    }


def _predict_sequential_fallback(
    s2_paths: Dict[str, str],
    model_path: str,
    meta_path: str,
    out_path: str,
    land_mask_path: Optional[str] = None,
    uncertainty_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fallback to sequential prediction if parallel fails.
    
    This imports and uses the original predict module.
    """
    log.info("Using sequential prediction fallback...")
    
    try:
        from predict import predict_scene
        
        predict_scene(
            s2_paths=s2_paths,
            land_mask_path=land_mask_path,
            rf_model_path=model_path,
            meta_json_path=meta_path,
            out_path=out_path,
            **kwargs
        )
        
        return {
            "success": True,
            "output_path": str(out_path),
            "method": "sequential_fallback"
        }
        
    except Exception as e:
        log.error("Sequential fallback also failed: %s", e, exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "method": "sequential_fallback"
        }


def estimate_parallel_benefit(
    height: int,
    width: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    n_workers: int = None
) -> Dict[str, Any]:
    """
    Estimate the benefit of parallel processing for given image dimensions.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        tile_size: Tile size to use
        n_workers: Number of workers (None = auto-detect)
        
    Returns:
        Dictionary with estimates
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)
    
    # Estimate number of tiles
    n_tiles_h = (height + tile_size - 1) // tile_size
    n_tiles_w = (width + tile_size - 1) // tile_size
    n_tiles = n_tiles_h * n_tiles_w
    
    # Rough timing estimates (based on typical performance)
    # Assume ~0.5 seconds per tile on average
    time_per_tile = 0.5
    
    sequential_time = n_tiles * time_per_tile
    parallel_time = (n_tiles / n_workers) * time_per_tile + 2.0  # +2s overhead
    
    speedup = sequential_time / max(parallel_time, 0.1)
    
    return {
        "image_size": f"{width} x {height}",
        "total_pixels": width * height,
        "tile_size": tile_size,
        "n_tiles": n_tiles,
        "n_workers": n_workers,
        "estimated_sequential_seconds": round(sequential_time, 1),
        "estimated_parallel_seconds": round(parallel_time, 1),
        "estimated_speedup": round(speedup, 2),
        "recommended": n_tiles >= 4 and speedup > 1.3
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        from core.logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import sys
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="Parallel SDB Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--b02", required=True, help="Path to B02 (Blue) band")
    parser.add_argument("--b03", required=True, help="Path to B03 (Green) band")
    parser.add_argument("--b04", required=True, help="Path to B04 (Red) band")
    parser.add_argument("--b08", required=True, help="Path to B08 (NIR) band")
    parser.add_argument("--model", required=True, help="Path to RF model (.pkl)")
    parser.add_argument("--meta", required=True, help="Path to model metadata (.json)")
    parser.add_argument("--output", required=True, help="Output path for SDB raster")
    parser.add_argument("--uncertainty", default=None, help="Output path for uncertainty")
    parser.add_argument("--land-mask", default=None, help="Land mask raster path")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--tile-size", type=int, default=2048, help="Tile size in pixels")
    parser.add_argument("--overlap", type=int, default=256, help="Tile overlap in pixels")
    parser.add_argument("--estimate", action="store_true", 
                        help="Only estimate benefit, don't run")
    
    args = parser.parse_args()
    
    s2_paths = {
        "B02": args.b02,
        "B03": args.b03,
        "B04": args.b04,
        "B08": args.b08
    }
    
    if args.estimate:
        import rasterio
        with rasterio.open(args.b02) as ds:
            estimate = estimate_parallel_benefit(
                ds.height, ds.width,
                tile_size=args.tile_size,
                n_workers=args.workers
            )
        
        import json
        log.info(json.dumps(estimate, indent=2))
        sys.exit(0)
    
    config = ParallelConfig(
        max_workers=args.workers,
        tile_size=args.tile_size,
        overlap=args.overlap
    )
    
    result = predict_scene_parallel(
        s2_paths=s2_paths,
        model_path=args.model,
        meta_path=args.meta,
        out_path=args.output,
        land_mask_path=args.land_mask,
        config=config,
        uncertainty_path=args.uncertainty
    )
    
    sys.exit(0 if result["success"] else 1)
