"""
Tile Blending for Chunked Predictions

Provides seamless blending in overlap regions to eliminate tile seams.
Addresses feedback: "chunked overlap handling should blend, not just crop"

Author: SDB Pipeline Development Team
Version: 0.7.1
"""

import numpy as np
import logging
from typing import Tuple, Optional
from enum import Enum

log = logging.getLogger(__name__)


class BlendMode(Enum):
    """Blending modes for tile overlaps."""
    NONE = "none"  # Inner window only (fast, possible seams)
    LINEAR = "linear"  # Linear ramp weights
    COSINE = "cosine"  # Cosine ramp weights (smoother)


def create_blend_weights_1d(size: int, overlap: int, mode: BlendMode = BlendMode.COSINE) -> np.ndarray:
    """
    Create 1D blend weights for overlap region.
    
    Args:
        size: Total size of dimension
        overlap: Overlap size on each side
        mode: Blending mode
    
    Returns:
        1D array of weights [0, 1]
    """
    weights = np.ones(size, dtype=np.float32)
    
    if overlap <= 0 or mode == BlendMode.NONE:
        return weights
    
    # Left ramp (0 → 1)
    if mode == BlendMode.LINEAR:
        left_ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    elif mode == BlendMode.COSINE:
        # Cosine ramp: smoother transitions
        t = np.linspace(0, np.pi/2, overlap, dtype=np.float32)
        left_ramp = np.sin(t)  # 0 → 1 with smooth acceleration
    else:
        left_ramp = np.ones(overlap, dtype=np.float32)
    
    weights[:overlap] = left_ramp
    
    # Right ramp (1 → 0)
    if mode == BlendMode.LINEAR:
        right_ramp = np.linspace(1, 0, overlap, dtype=np.float32)
    elif mode == BlendMode.COSINE:
        t = np.linspace(0, np.pi/2, overlap, dtype=np.float32)
        right_ramp = np.cos(t)  # 1 → 0 with smooth deceleration
    else:
        right_ramp = np.ones(overlap, dtype=np.float32)
    
    weights[-overlap:] = right_ramp
    
    return weights


def create_blend_weights_2d(
    height: int,
    width: int,
    overlap_y: int,
    overlap_x: int,
    mode: BlendMode = BlendMode.COSINE
) -> np.ndarray:
    """
    Create 2D blend weights for tile with overlaps.
    
    Args:
        height: Tile height
        width: Tile width
        overlap_y: Overlap in Y direction
        overlap_x: Overlap in X direction
        mode: Blending mode
    
    Returns:
        2D array of weights [0, 1]
    """
    # Create 1D weights for each dimension
    weights_y = create_blend_weights_1d(height, overlap_y, mode)
    weights_x = create_blend_weights_1d(width, overlap_x, mode)
    
    # Outer product for 2D weights
    weights_2d = np.outer(weights_y, weights_x).astype(np.float32)
    
    return weights_2d


class TileBlender:
    """
    Manages weighted accumulation for seamless tile blending.
    """
    
    def __init__(
        self,
        output_shape: Tuple[int, int],
        tile_size: int,
        overlap: int,
        mode: BlendMode = BlendMode.COSINE,
        nodata_value: float = -9999.0
    ):
        """
        Initialize tile blender.
        
        Args:
            output_shape: (height, width) of full output
            tile_size: Size of tiles
            overlap: Overlap between tiles
            mode: Blending mode
            nodata_value: NoData value to ignore
        """
        self.output_shape = output_shape
        self.tile_size = tile_size
        self.overlap = overlap
        self.mode = mode
        self.nodata_value = nodata_value
        
        # Accumulation arrays
        self.value_sum = np.zeros(output_shape, dtype=np.float64)
        self.weight_sum = np.zeros(output_shape, dtype=np.float64)
        
        log.info(f"[BLEND] Initialized TileBlender: mode={mode.value}, "
                f"shape={output_shape}, tile_size={tile_size}, overlap={overlap}")
    
    def add_tile(
        self,
        tile_data: np.ndarray,
        row_start: int,
        col_start: int
    ):
        """
        Add a tile to the accumulation with blending weights.
        
        Args:
            tile_data: Tile data array
            row_start: Starting row in output
            col_start: Starting column in output
        """
        tile_h, tile_w = tile_data.shape
        
        # Determine actual overlap for this tile (edge tiles may have less)
        overlap_top = self.overlap if row_start > 0 else 0
        overlap_bottom = self.overlap if row_start + tile_h < self.output_shape[0] else 0
        overlap_left = self.overlap if col_start > 0 else 0
        overlap_right = self.overlap if col_start + tile_w < self.output_shape[1] else 0
        
        # Create blend weights for this tile
        if self.mode == BlendMode.NONE:
            # No blending - use inner window only
            row_slice = slice(row_start + overlap_top, row_start + tile_h - overlap_bottom)
            col_slice = slice(col_start + overlap_left, col_start + tile_w - overlap_right)
            
            tile_slice_y = slice(overlap_top, tile_h - overlap_bottom)
            tile_slice_x = slice(overlap_left, tile_w - overlap_right)
            
            # Copy inner data
            inner_data = tile_data[tile_slice_y, tile_slice_x]
            valid_mask = inner_data != self.nodata_value
            
            self.value_sum[row_slice, col_slice][valid_mask] = inner_data[valid_mask]
            self.weight_sum[row_slice, col_slice][valid_mask] = 1.0
        
        else:
            # Blended mode - use weights
            weights = create_blend_weights_2d(
                tile_h, tile_w,
                max(overlap_top, overlap_bottom),
                max(overlap_left, overlap_right),
                self.mode
            )
            
            # Valid data mask
            valid_mask = tile_data != self.nodata_value
            
            # Extract the portion that goes into output
            row_end = min(row_start + tile_h, self.output_shape[0])
            col_end = min(col_start + tile_w, self.output_shape[1])
            
            out_h = row_end - row_start
            out_w = col_end - col_start
            
            # Accumulate weighted values
            self.value_sum[row_start:row_end, col_start:col_end] += (
                tile_data[:out_h, :out_w] * weights[:out_h, :out_w] * valid_mask[:out_h, :out_w]
            )
            self.weight_sum[row_start:row_end, col_start:col_end] += (
                weights[:out_h, :out_w] * valid_mask[:out_h, :out_w]
            )
    
    def finalize(self) -> np.ndarray:
        """
        Finalize blending by dividing accumulated values by weights.
        
        Returns:
            Blended output array
        """
        output = np.full(self.output_shape, self.nodata_value, dtype=np.float32)
        
        # Divide by weights where weight > 0
        valid_weights = self.weight_sum > 0
        output[valid_weights] = (
            self.value_sum[valid_weights] / self.weight_sum[valid_weights]
        ).astype(np.float32)
        
        # Calculate coverage statistics
        coverage_pct = 100.0 * valid_weights.sum() / valid_weights.size
        
        log.info(f"[BLEND] Finalized: {coverage_pct:.1f}% coverage")
        
        return output


def compute_seam_diagnostic(
    blended: np.ndarray,
    tile_size: int,
    overlap: int,
    nodata_value: float = -9999.0
) -> np.ndarray:
    """
    Compute seam diagnostic raster showing differences at tile boundaries.
    
    Args:
        blended: Blended output raster
        tile_size: Tile size used
        overlap: Overlap used
        nodata_value: NoData value
    
    Returns:
        Diagnostic raster showing absolute differences at boundaries
    """
    height, width = blended.shape
    diagnostic = np.zeros((height, width), dtype=np.float32)
    
    stride = tile_size - 2 * overlap
    
    # Check vertical seams
    for col in range(stride, width, stride):
        if col - 1 >= 0 and col < width:
            left = blended[:, col - 1]
            right = blended[:, col]
            
            valid = (left != nodata_value) & (right != nodata_value)
            if valid.any():
                diff = np.abs(left - right)
                diagnostic[:, col][valid] = diff[valid]
    
    # Check horizontal seams
    for row in range(stride, height, stride):
        if row - 1 >= 0 and row < height:
            top = blended[row - 1, :]
            bottom = blended[row, :]
            
            valid = (top != nodata_value) & (bottom != nodata_value)
            if valid.any():
                diff = np.abs(top - bottom)
                diagnostic[row, :][valid] = diff[valid]
    
    return diagnostic


def analyze_seam_quality(
    seam_diagnostic: np.ndarray,
    threshold_m: float = 0.5
) -> dict:
    """
    Analyze seam quality from diagnostic raster.
    
    Args:
        seam_diagnostic: Seam diagnostic raster
        threshold_m: Threshold for "significant" seam
    
    Returns:
        Dictionary of seam quality metrics
    """
    seam_pixels = seam_diagnostic > 0
    
    if not seam_pixels.any():
        return {
            "seams_detected": False,
            "message": "No seams detected"
        }
    
    seam_values = seam_diagnostic[seam_pixels]
    
    stats = {
        "seams_detected": True,
        "total_seam_pixels": int(seam_pixels.sum()),
        "max_difference_m": float(seam_values.max()),
        "mean_difference_m": float(seam_values.mean()),
        "median_difference_m": float(np.median(seam_values)),
        "std_difference_m": float(seam_values.std()),
        "pixels_above_threshold": int((seam_values > threshold_m).sum()),
        "quality_assessment": "good"
    }
    
    # Quality assessment
    if stats["max_difference_m"] > 2.0:
        stats["quality_assessment"] = "poor"
        stats["recommendation"] = "Consider using cosine blending or check for systematic errors"
    elif stats["max_difference_m"] > 1.0:
        stats["quality_assessment"] = "fair"
        stats["recommendation"] = "Seams are noticeable, consider cosine blending"
    elif stats["mean_difference_m"] > threshold_m:
        stats["quality_assessment"] = "acceptable"
        stats["recommendation"] = "Minor seams present but within tolerance"
    
    return stats


# Example usage function
def blend_tiles_example():
    """Example of how to use TileBlender."""
    
    # Setup
    output_shape = (5000, 5000)
    tile_size = 1024
    overlap = 128
    
    blender = TileBlender(
        output_shape=output_shape,
        tile_size=tile_size,
        overlap=overlap,
        mode=BlendMode.COSINE
    )
    
    # Process tiles (pseudocode)
    # for tile_row, tile_col in tile_grid:
    #     tile_data = predict_tile(...)
    #     blender.add_tile(tile_data, tile_row * stride, tile_col * stride)
    
    # Finalize
    blended_output = blender.finalize()
    
    # Diagnostic
    seam_diagnostic = compute_seam_diagnostic(
        blended_output, tile_size, overlap
    )
    
    seam_stats = analyze_seam_quality(seam_diagnostic)
    
    return blended_output, seam_diagnostic, seam_stats
