#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
river_skeleton.py – Raster-based River Bathymetry with Automatic Domain Classification

Replaces vector cross-sections with a "Channel Skeleton" distance transform.
Automatically distinguishes "River" from "Open Water" based on channel width and proximity.

Algorithm:
1. Load Water Mask (defining 'wet') and River Centerlines (defining 'spine').
2. Compute Euclidean Distance Transforms (EDT):
   - d_bank: Distance to nearest LAND.
   - d_spine: Distance to nearest CENTERLINE.
3. Automatic Classification (River Domain Mask):
   - Pixel is RIVER if: (d_bank < max_width/2) AND (d_spine < search_radius)
   - Pixel is OPEN WATER if: (d_bank >= max_width/2) OR (d_spine >= search_radius)
4. Interpolation (Only in River Domain):
   - Calculate normalized coordinate r = d_bank / (d_bank + d_spine).
   - Propagate Dmax (a*W^b) from the spine to the banks using Nearest Neighbor.
   - Apply profile shape: Depth = Dmax * (r ^ shape_exp).
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import geopandas as gpd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("river_skeleton")

def _load_raster_mask(path):
    """Load water mask. Returns boolean array (True=Water), profile, transform."""
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata
        
        # Robust mask creation: treat valid water values (usually 1 or 255) as True
        if nodata is not None:
            mask = (arr != nodata) & (arr != 0)
        else:
            mask = (arr != 0)
            
    return mask, profile, transform

def _save_raster(path, data, profile, nodata=-9999.0):
    profile.update(dtype=rasterio.float32, count=1, nodata=nodata, compress='deflate')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)

def main():
    p = argparse.ArgumentParser(description="Generate river bathymetry using skeleton distance transform.")
    p.add_argument("--river-gpkg", required=True, help="River network GPKG (centerlines)")
    p.add_argument("--water-mask", required=True, help="Binary water mask TIF (1=water, 0=land)")
    p.add_argument("--out-depth", required=True, help="Output Depth GeoTIFF path")
    
    # Classification Parameters (The "Automatic" Logic)
    p.add_argument("--max-width", type=float, default=600.0, 
                   help="Maximum channel width (m) to classify as river. Wider areas are treated as open water (NoData).")
    p.add_argument("--search-radius", type=float, default=1000.0,
                   help="Maximum distance (m) from a centerline to interpolate. Water beyond this is open water.")

    # Physics/Interpolation Parameters
    p.add_argument("--a", type=float, default=0.18, help="Depth-Width coefficient (D = a * W^b)")
    p.add_argument("--b", type=float, default=0.50, help="Depth-Width exponent")
    p.add_argument("--shape-exp", type=float, default=0.5, help="Channel shape exponent (0.5=Parabolic, 1.0=V-shape)")
    p.add_argument("--min-depth", type=float, default=0.5, help="Minimum Dmax clamp")
    p.add_argument("--max-depth", type=float, default=30.0, help="Maximum Dmax clamp")
    
    args = p.parse_args()
    
    # 1. Load Inputs
    log.info(f"Loading inputs... (Max Width: {args.max_width}m)")
    rivers = gpd.read_file(args.river_gpkg)
    if rivers.empty:
        log.error("River network is empty.")
        sys.exit(1)
        
    mask_bool, profile, transform = _load_raster_mask(args.water_mask)
    pixel_size = transform[0] # Assuming square pixels
    
    # 2. Rasterize Centerlines (The Skeleton)
    log.info("Rasterizing river centerlines...")
    # Filter to valid geometries
    rivers = rivers[rivers.geometry.notnull() & ~rivers.geometry.is_empty]
    
    # Burn skeleton as boolean (1=Skeleton, 0=Other)
    skeleton_mask = rasterize(
        [(g, 1) for g in rivers.geometry],
        out_shape=mask_bool.shape,
        transform=transform,
        all_touched=True,
        dtype=np.uint8
    ).astype(bool)
    
    # 3. Distance Transforms (The Core Math)
    log.info("Computing distance fields (Bank & Skeleton)...")
    
    # d_bank: Distance TO nearest Land (0). 
    # Input to edt: 1=Water (background), 0=Land (target).
    d_bank = distance_transform_edt(mask_bool.astype(np.float32), sampling=pixel_size)
    
    # d_spine: Distance TO nearest Skeleton (0).
    # Input to edt: 1=NotSkeleton, 0=Skeleton.
    d_spine = distance_transform_edt((~skeleton_mask).astype(np.float32), sampling=pixel_size)

    # 4. Automatic Domain Classification
    log.info("Classifying River vs. Open Water...")
    
    # Condition A: Is it narrow enough? (Local width approx 2 * d_bank)
    is_narrow = d_bank <= (args.max_width / 2.0)
    
    # Condition B: Is it near a mapped river?
    is_near_spine = d_spine <= args.search_radius
    
    # Valid River Domain: Must be Water AND Narrow AND Near Spine
    river_domain = mask_bool & is_narrow & is_near_spine
    
    valid_count = np.sum(river_domain)
    total_water = np.sum(mask_bool)
    if total_water > 0:
        pct = valid_count / total_water * 100
    else:
        pct = 0.0
    log.info(f"Classification: {valid_count} pixels classified as River ({pct:.1f}% of water).")
    
    if valid_count == 0:
        log.warning("No pixels classified as river! Check max-width/search-radius or alignment.")
        # Output empty raster
        final_depth = np.full(mask_bool.shape, -9999.0, dtype=np.float32)
        _save_raster(args.out_depth, final_depth, profile)
        return

    # 5. Interpolation (Only within River Domain)
    log.info("Interpolating river bathymetry...")
    
    # Normalized coordinate r (0=Bank, 1=Center)
    # r = d_bank / (d_bank + d_spine)
    denom = d_bank + d_spine
    denom[denom == 0] = 0.001
    r = d_bank / denom
    r = np.clip(r, 0.0, 1.0)
    
    # 6. Dmax Estimation & Propagation
    # We estimate Dmax ONLY at the skeleton pixels, then spread it.
    
    # Identify valid skeleton pixels (Skeleton inside Water)
    skeleton_valid = skeleton_mask & mask_bool
    
    # Initialize Dmax field
    dmax_field = np.full(mask_bool.shape, np.nan, dtype=np.float32)
    
    if np.sum(skeleton_valid) > 0:
        # At skeleton, Width ~ 2 * d_bank
        w_at_skeleton = 2.0 * d_bank[skeleton_valid]
        d_at_skeleton = args.a * np.power(w_at_skeleton, args.b)
        d_at_skeleton = np.clip(d_at_skeleton, args.min_depth, args.max_depth)
        dmax_field[skeleton_valid] = d_at_skeleton
        
        # Propagate Nearest Neighbor using EDT indices
        # We find the nearest skeleton pixel for every pixel in the river domain
        # Input to EDT: Target=Skeleton(0), Background=Other(1)
        _, indices = distance_transform_edt((~skeleton_mask).astype(np.float32), return_indices=True)
        
        # indices is (2, H, W). Map indices to dmax values.
        # This gives every pixel the Dmax of its nearest centerline point.
        dmax_propagated = dmax_field[indices[0], indices[1]]
    else:
        log.warning("Centerlines do not overlap water mask. Using default depth.")
        dmax_propagated = np.full(mask_bool.shape, args.min_depth, dtype=np.float32)

    # 7. Apply Shape Function
    # Depth = Dmax * r^shape_exp
    depth_model = dmax_propagated * np.power(r, args.shape_exp)
    
    # 8. Masking & Output
    # Apply the strict domain mask (Open Water becomes NoData)
    final_depth = np.where(river_domain, depth_model, -9999.0)
    
    # Extra safety: ensure valid pixels are finite
    final_depth = np.where(np.isfinite(final_depth), final_depth, -9999.0)
    
    log.info(f"Saving to {args.out_depth}")
    _save_raster(args.out_depth, final_depth, profile)
    log.info("Done.")

if __name__ == "__main__":
    main()