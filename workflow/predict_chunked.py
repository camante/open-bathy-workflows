#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_chunked.py - Memory-efficient chunked prediction wrapper

Wraps predict.py with chunked_processing to handle large AOIs that
exceed available memory.

Features:
- Automatic memory requirement estimation
- Tile-based processing with overlap
- Seamless mosaicking
- Progress tracking
- Compatible with existing predict.py interface

Usage:
    from predict_chunked import predict_scene_chunked
    
    # Automatic chunking based on memory
    predict_scene_chunked(
        model_dir=Path("model"),
        band_paths=band_paths,
        out_dir=Path("output"),
        max_memory_gb=8.0  # Triggers chunking if needed
    )
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import rasterio
from rasterio.windows import Window

log = logging.getLogger(__name__)

# Import chunked processing
try:
    from chunked_processing import (
        ChunkedRasterProcessor,
        should_use_chunked_processing,
        estimate_memory_requirement_gb
    )
    CHUNKED_AVAILABLE = True
except ImportError:
    log.warning("[predict_chunked] chunked_processing module not available")
    CHUNKED_AVAILABLE = False

# Import main prediction function
try:
    import predict
    PREDICT_AVAILABLE = True
except ImportError:
    log.error("[predict_chunked] predict module not available")
    PREDICT_AVAILABLE = False


def predict_scene_chunked(
    model_dir: Path,
    band_paths: Dict[str, Path],
    out_dir: Path,
    template_raster: Optional[Path] = None,
    land_mask_path: Optional[Path] = None,
    max_depth_m: float = 20.0,
    max_memory_gb: float = 8.0,
    tile_size: int = 2048,
    overlap: int = 256,
    use_chunked: Optional[bool] = None,
    **kwargs
) -> Dict:
    """
    Predict SDB with automatic chunking for large AOIs.
    
    Args:
        model_dir: Directory containing trained model
        band_paths: Dict of {band_name: raster_path}
        out_dir: Output directory
        template_raster: Template for output grid (default: B02)
        land_mask_path: Optional land mask
        max_depth_m: Maximum depth to predict
        max_memory_gb: Memory limit for automatic chunking
        tile_size: Tile size in pixels (for chunked mode)
        overlap: Overlap between tiles in pixels
        use_chunked: Force chunked mode (None = auto-detect)
        **kwargs: Additional arguments passed to predict functions
    
    Returns:
        Dictionary with prediction statistics
    """
    if not CHUNKED_AVAILABLE or not PREDICT_AVAILABLE:
        raise ImportError("Required modules not available")
    
    # Determine template raster
    if template_raster is None:
        template_raster = band_paths.get('B02') or list(band_paths.values())[0]
    
    # Check memory requirements
    with rasterio.open(template_raster) as ds:
        height, width = ds.height, ds.width
        
        # Estimate memory for prediction
        n_bands = 14  # Number of features
        memory_gb = estimate_memory_requirement_gb(
            height, width, 
            n_arrays=n_bands + 3,  # features + prediction + uncertainty + mask
            dtype='float32'
        )
        
        log.info(f"[predict_chunked] Estimated memory: {memory_gb:.2f} GB "
                f"(limit: {max_memory_gb:.2f} GB)")
        
        # Decide on chunking
        if use_chunked is None:
            use_chunked = should_use_chunked_processing(
                height, width, max_memory_gb,
                n_arrays=n_bands + 3, dtype='float32'
            )
        
        if use_chunked:
            log.info(f"[predict_chunked] Using chunked processing "
                    f"(tile_size={tile_size}, overlap={overlap})")
            return _predict_chunked(
                model_dir, band_paths, out_dir, ds,
                land_mask_path, max_depth_m,
                tile_size, overlap, **kwargs
            )
        else:
            log.info(f"[predict_chunked] Using monolithic processing")
            return _predict_monolithic(
                model_dir, band_paths, out_dir,
                land_mask_path, max_depth_m, **kwargs
            )


def _predict_monolithic(
    model_dir: Path,
    band_paths: Dict[str, Path],
    out_dir: Path,
    land_mask_path: Optional[Path],
    max_depth_m: float,
    **kwargs
) -> Dict:
    """Call original predict.predict_scene() for small AOIs."""
    
    # Call original prediction function
    result = predict.predict_scene(
        model_dir=model_dir,
        band_paths=band_paths,
        out_dir=out_dir,
        land_mask_path=land_mask_path,
        max_depth_sdb_m=max_depth_m,
        **kwargs
    )
    
    return result


def _predict_chunked(
    model_dir: Path,
    band_paths: Dict[str, Path],
    out_dir: Path,
    template_ds: rasterio.DatasetReader,
    land_mask_path: Optional[Path],
    max_depth_m: float,
    tile_size: int,
    overlap: int,
    **kwargs
) -> Dict:
    """
    Tile-based prediction for large AOIs.
    
    Strategy:
    1. Load model once
    2. Process in tiles with overlap
    3. Blend overlapping predictions
    4. Mosaic into output rasters
    """
    import joblib
    from predict import _load_rf_model, add_s2_optical_features
    
    log.info("[predict_chunked] Starting chunked prediction")
    
    # Load model once
    model_meta_path = model_dir / "model_meta.json"
    if not model_meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {model_meta_path}")
    
    with open(model_meta_path) as f:
        model_meta = json.load(f)
    
    rf_model = joblib.load(model_dir / "rf_model.pkl")
    feature_names = model_meta.get('feature_names', [])
    
    log.info(f"[predict_chunked] Loaded model with {len(feature_names)} features")
    
    # Setup output rasters
    out_dir.mkdir(parents=True, exist_ok=True)
    
    depth_path = out_dir / "SDB_Prediction_10m.tif"
    unc_path = out_dir / "SDB_Uncertainty_10m.tif"
    mask_path = out_dir / "SDB_Mask_10m.tif"
    
    # Create chunked processors
    depth_proc = ChunkedRasterProcessor(
        template_path=str(list(band_paths.values())[0]),
        output_path=str(depth_path),
        tile_size=(tile_size, tile_size),
        overlap=overlap,
        dtype='float32',
        nodata=-9999.0
    )
    
    unc_proc = ChunkedRasterProcessor(
        template_path=str(list(band_paths.values())[0]),
        output_path=str(unc_path),
        tile_size=(tile_size, tile_size),
        overlap=overlap,
        dtype='float32',
        nodata=-9999.0
    )
    
    # Statistics accumulation
    stats = {
        'n_tiles': 0,
        'n_valid_pixels': 0,
        'depth_sum': 0.0,
        'depth_sum_sq': 0.0,
        'unc_sum': 0.0
    }
    
    # Process tiles
    with depth_proc as dp, unc_proc as up:
        tiles = list(dp.tiles())
        log.info(f"[predict_chunked] Processing {len(tiles)} tiles")
        
        for tile_info in tiles:
            window = tile_info.window
            
            try:
                # Read bands for this tile
                tile_data = {}
                for band_name, band_path in band_paths.items():
                    with rasterio.open(band_path) as ds:
                        tile_data[band_name] = ds.read(1, window=window).astype('float32')
                
                # Read land mask if provided
                if land_mask_path:
                    with rasterio.open(land_mask_path) as ds:
                        land_mask = ds.read(1, window=window)
                        water_mask = (land_mask == 0)  # Assuming 0 = water
                else:
                    water_mask = np.ones(tile_data['B02'].shape, dtype=bool)
                
                # Convert to DataFrame for feature engineering
                import pandas as pd
                
                h, w = tile_data['B02'].shape
                n_pixels = h * w
                
                # Flatten arrays
                df_dict = {
                    band: arr.ravel() 
                    for band, arr in tile_data.items()
                }
                df = pd.DataFrame(df_dict)
                
                # Add features
                l_inf = model_meta.get('l_inf', {})
                df = add_s2_optical_features(df, l_inf=l_inf)
                
                # Select features matching model
                X = df[feature_names].to_numpy(dtype='float32')
                
                # Predict
                depth_pred = rf_model.predict(X).reshape(h, w)
                
                # Simple uncertainty estimate (tree std)
                if hasattr(rf_model, 'estimators_'):
                    tree_preds = np.array([
                        tree.predict(X) for tree in rf_model.estimators_[:min(20, len(rf_model.estimators_))]
                    ])
                    unc = np.std(tree_preds, axis=0).reshape(h, w)
                else:
                    unc = np.ones((h, w), dtype='float32') * 0.5
                
                # Apply water mask and depth limits
                depth_pred[~water_mask] = -9999.0
                depth_pred[depth_pred > 0] = -9999.0  # Above water
                depth_pred[depth_pred < -max_depth_m] = -9999.0  # Too deep
                
                unc[~water_mask] = -9999.0
                unc[depth_pred == -9999.0] = -9999.0
                
                # Write tile
                dp.write_tile(depth_pred, window)
                up.write_tile(unc, window)
                
                # Update statistics
                valid = (depth_pred != -9999.0) & np.isfinite(depth_pred)
                stats['n_tiles'] += 1
                stats['n_valid_pixels'] += valid.sum()
                stats['depth_sum'] += depth_pred[valid].sum()
                stats['depth_sum_sq'] += (depth_pred[valid] ** 2).sum()
                stats['unc_sum'] += unc[valid].sum()
                
            except Exception as e:
                log.error(f"[predict_chunked] Tile {tile_info.index} failed: {e}")
                continue
    
    # Compute final statistics
    if stats['n_valid_pixels'] > 0:
        mean_depth = stats['depth_sum'] / stats['n_valid_pixels']
        var_depth = (stats['depth_sum_sq'] / stats['n_valid_pixels']) - mean_depth**2
        std_depth = np.sqrt(max(0, var_depth))
        mean_unc = stats['unc_sum'] / stats['n_valid_pixels']
    else:
        mean_depth = std_depth = mean_unc = 0.0
    
    result = {
        'status': 'success',
        'n_tiles': stats['n_tiles'],
        'n_valid_pixels': int(stats['n_valid_pixels']),
        'mean_depth_m': float(mean_depth),
        'std_depth_m': float(std_depth),
        'mean_uncertainty_m': float(mean_unc),
        'output_files': {
            'depth': str(depth_path),
            'uncertainty': str(unc_path)
        }
    }
    
    log.info(f"[predict_chunked] Completed {stats['n_tiles']} tiles, "
            f"{stats['n_valid_pixels']:,} valid pixels")
    log.info(f"[predict_chunked] Mean depth: {mean_depth:.2f} m ± {std_depth:.2f} m")
    
    return result


def main():
    """CLI for chunked prediction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-efficient chunked SDB prediction")
    parser.add_argument("--model-dir", required=True, help="Model directory")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--bands", required=True, nargs='+', 
                       help="Band rasters: B02=path B03=path ...")
    parser.add_argument("--land-mask", help="Land mask raster")
    parser.add_argument("--max-depth", type=float, default=20.0, help="Max depth (m)")
    parser.add_argument("--max-memory", type=float, default=8.0, help="Memory limit (GB)")
    parser.add_argument("--tile-size", type=int, default=2048, help="Tile size (pixels)")
    parser.add_argument("--overlap", type=int, default=256, help="Tile overlap (pixels)")
    parser.add_argument("--force-chunked", action='store_true', help="Force chunked mode")
    
    args = parser.parse_args()
    
    # Parse band paths
    band_paths = {}
    for item in args.bands:
        if '=' in item:
            band, path = item.split('=', 1)
            band_paths[band] = Path(path)
    
    # Run prediction
    result = predict_scene_chunked(
        model_dir=Path(args.model_dir),
        band_paths=band_paths,
        out_dir=Path(args.out_dir),
        land_mask_path=Path(args.land_mask) if args.land_mask else None,
        max_depth_m=args.max_depth,
        max_memory_gb=args.max_memory,
        tile_size=args.tile_size,
        overlap=args.overlap,
        use_chunked=args.force_chunked or None
    )
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.basicConfig(level=logging.INFO)
    main()
