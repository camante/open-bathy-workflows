#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunked_processing.py - Memory-Efficient Tile-Based Raster Processing

This module provides utilities for processing large rasters in chunks/tiles
to avoid memory exhaustion on large AOIs.

Key Features:
- Automatic tiling based on available memory
- Overlapping tiles for seamless interpolation
- Progress tracking with tqdm
- Compatible with rasterio window-based I/O
"""


import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable, Generator, Iterator, List, Optional, Tuple, TypeVar, Union
)

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

log = logging.getLogger(__name__)

# Import constants
try:
    from constants import (
        DEFAULT_TILE_SIZE,
        CHUNK_THRESHOLD_PIXELS,
        NODATA_DEPTH,
    )
except ImportError:
    DEFAULT_TILE_SIZE = 1024
    CHUNK_THRESHOLD_PIXELS = 50_000_000
    NODATA_DEPTH = -9999.0


@dataclass
class TileInfo:
    """Information about a single tile for processing."""
    
    # Global coordinates (in full raster)
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    
    # Tile index
    tile_row: int
    tile_col: int
    tile_idx: int
    
    # Overlap information
    overlap_top: int = 0
    overlap_bottom: int = 0
    overlap_left: int = 0
    overlap_right: int = 0
    
    @property
    def height(self) -> int:
        return self.row_end - self.row_start
    
    @property
    def width(self) -> int:
        return self.col_end - self.col_start
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)
    
    @property
    def window(self) -> "rasterio.windows.Window":
        """Return a rasterio Window object for this tile."""
        from rasterio.windows import Window
        return Window(
            col_off=self.col_start,
            row_off=self.row_start,
            width=self.width,
            height=self.height
        )
    
    @property
    def inner_slice(self) -> Tuple[slice, slice]:
        """
        Return slice for the inner (non-overlapping) portion of the tile.
        Use this when writing results to avoid double-counting overlap zones.
        """
        row_slice = slice(
            self.overlap_top,
            self.height - self.overlap_bottom if self.overlap_bottom > 0 else None
        )
        col_slice = slice(
            self.overlap_left,
            self.width - self.overlap_right if self.overlap_right > 0 else None
        )
        return (row_slice, col_slice)
    
    @property
    def inner_global_bounds(self) -> Tuple[int, int, int, int]:
        """Return (row_start, row_end, col_start, col_end) for inner portion."""
        return (
            self.row_start + self.overlap_top,
            self.row_end - self.overlap_bottom,
            self.col_start + self.overlap_left,
            self.col_end - self.overlap_right,
        )


def estimate_memory_usage(
    height: int,
    width: int,
    n_bands: int = 1,
    dtype: np.dtype = np.float32,
    n_arrays: int = 3,
) -> int:
    """
    Estimate memory usage in bytes for raster processing.
    
    Args:
        height: Raster height in pixels
        width: Raster width in pixels
        n_bands: Number of bands
        dtype: Data type
        n_arrays: Number of arrays to hold in memory simultaneously
        
    Returns:
        Estimated memory usage in bytes
    """
    itemsize = np.dtype(dtype).itemsize
    return height * width * n_bands * itemsize * n_arrays


def estimate_memory_requirement_gb(
    height: int,
    width: int,
    n_bands: int = 1,
    dtype: np.dtype = np.float32,
    n_arrays: int = 3,
) -> float:
    """Estimate memory requirement in gigabytes.

    This is a thin compatibility wrapper over :func:`estimate_memory_usage`.
    """
    bytes_needed = estimate_memory_usage(height=height, width=width, n_bands=n_bands, dtype=dtype, n_arrays=n_arrays)
    return float(bytes_needed) / (1024.0 ** 3)

def should_use_chunked_processing(
    height: int,
    width: int,
    threshold_pixels: int = CHUNK_THRESHOLD_PIXELS,
    *,
    max_memory_gb: float | None = None,
    n_bands: int = 1,
    dtype: np.dtype = np.float32,
    n_arrays: int = 3,
) -> bool:
    """Determine if chunked processing should be used.

    Two heuristics are supported:

    1) **Pixel threshold** (default): if total pixels exceed `threshold_pixels`
    2) **Memory budget** (optional): if estimated memory (GB) exceeds `max_memory_gb`

    The memory-budget mode is useful on shared/limited-memory machines and is
    what the integration tests exercise.
    """
    total_pixels = int(height) * int(width)

    if max_memory_gb is not None:
        mem_gb = estimate_memory_requirement_gb(
            height=int(height),
            width=int(width),
            n_bands=int(n_bands),
            dtype=dtype,
            n_arrays=int(n_arrays),
        )
        return mem_gb > float(max_memory_gb)

    return total_pixels > int(threshold_pixels)

def generate_tiles(
    height: int,
    width: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = 0,
) -> Generator[TileInfo, None, None]:
    """
    Generate tile information for processing a raster in chunks.
    
    Args:
        height: Total raster height
        width: Total raster width
        tile_size: Size of each tile (square)
        overlap: Overlap between adjacent tiles (for interpolation)
        
    Yields:
        TileInfo objects for each tile
    """
    tile_idx = 0
    n_rows = (height + tile_size - 1) // tile_size
    n_cols = (width + tile_size - 1) // tile_size
    
    for tile_row in range(n_rows):
        for tile_col in range(n_cols):
            # Calculate base tile bounds
            row_start = tile_row * tile_size
            col_start = tile_col * tile_size
            
            row_end = min(row_start + tile_size, height)
            col_end = min(col_start + tile_size, width)
            
            # Add overlap (but don't exceed raster bounds)
            overlap_top = min(overlap, row_start)
            overlap_bottom = min(overlap, height - row_end)
            overlap_left = min(overlap, col_start)
            overlap_right = min(overlap, width - col_end)
            
            # Expand tile to include overlap
            row_start_with_overlap = row_start - overlap_top
            row_end_with_overlap = row_end + overlap_bottom
            col_start_with_overlap = col_start - overlap_left
            col_end_with_overlap = col_end + overlap_right
            
            yield TileInfo(
                row_start=row_start_with_overlap,
                row_end=row_end_with_overlap,
                col_start=col_start_with_overlap,
                col_end=col_end_with_overlap,
                tile_row=tile_row,
                tile_col=tile_col,
                tile_idx=tile_idx,
                overlap_top=overlap_top,
                overlap_bottom=overlap_bottom,
                overlap_left=overlap_left,
                overlap_right=overlap_right,
            )
            
            tile_idx += 1


def count_tiles(
    height: int,
    width: int,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> int:
    """Count the number of tiles for a given raster size."""
    n_rows = (height + tile_size - 1) // tile_size
    n_cols = (width + tile_size - 1) // tile_size
    return n_rows * n_cols


T = TypeVar('T')


def process_tiles_with_progress(
    tiles: Iterator[TileInfo],
    total: int,
    process_fn: Callable[[TileInfo], T],
    desc: str = "Processing tiles",
    disable_progress: bool = False,
) -> Generator[T, None, None]:
    """
    Process tiles with progress bar.
    
    Args:
        tiles: Iterator of TileInfo objects
        total: Total number of tiles
        process_fn: Function to process each tile
        desc: Description for progress bar
        disable_progress: Disable progress bar
        
    Yields:
        Results from process_fn for each tile
    """
    if TQDM_AVAILABLE and not disable_progress:
        tiles = tqdm(tiles, total=total, desc=desc)
    
    for tile in tiles:
        yield process_fn(tile)


class ChunkedRasterProcessor:
    """
    Context manager for chunked raster processing.
    
    Example usage:
        with ChunkedRasterProcessor(src_path, dst_path, tile_size=1024, overlap=64) as proc:
            for tile in proc.tiles():
                # Read tile data
                data = proc.read_tile(tile)
                
                # Process
                result = my_processing_function(data)
                
                # Write result (uses inner slice to avoid overlap duplication)
                proc.write_tile(tile, result)
    """
    
    def __init__(
        self,
        src_path: Union[str, Path],
        dst_path: Optional[Union[str, Path]] = None,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = 0,
        dtype: np.dtype = np.float32,
        nodata: float = NODATA_DEPTH,
    ):
        """
        Initialize chunked processor.
        
        Args:
            src_path: Path to source raster
            dst_path: Path to output raster (optional)
            tile_size: Size of processing tiles
            overlap: Overlap between tiles
            dtype: Output data type
            nodata: Nodata value
        """
        self.src_path = Path(src_path)
        self.dst_path = Path(dst_path) if dst_path else None
        self.tile_size = tile_size
        self.overlap = overlap
        self.dtype = dtype
        self.nodata = nodata
        
        self._src_ds = None
        self._dst_ds = None
        self._profile = None
    
    def __enter__(self) -> "ChunkedRasterProcessor":
        import rasterio
        
        self._src_ds = rasterio.open(self.src_path)
        self._profile = self._src_ds.profile.copy()
        
        if self.dst_path:
            self._profile.update(
                dtype=self.dtype,
                nodata=self.nodata,
                compress="deflate",
                tiled=True,
                blockxsize=min(256, self.tile_size),
                blockysize=min(256, self.tile_size),
            )
            self.dst_path.parent.mkdir(parents=True, exist_ok=True)
            self._dst_ds = rasterio.open(self.dst_path, "w", **self._profile)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._src_ds:
            self._src_ds.close()
        if self._dst_ds:
            self._dst_ds.close()
        return False
    
    @property
    def height(self) -> int:
        return self._src_ds.height
    
    @property
    def width(self) -> int:
        return self._src_ds.width
    
    @property
    def profile(self) -> dict:
        return self._profile.copy()
    
    def tiles(self) -> Generator[TileInfo, None, None]:
        """Generate tiles for this raster."""
        return generate_tiles(
            self.height,
            self.width,
            self.tile_size,
            self.overlap
        )
    
    def tile_count(self) -> int:
        """Return total number of tiles."""
        return count_tiles(self.height, self.width, self.tile_size)
    
    def read_tile(self, tile: TileInfo, band: int = 1) -> np.ndarray:
        """Read data for a tile."""
        return self._src_ds.read(band, window=tile.window)
    
    def write_tile(
        self,
        tile: TileInfo,
        data: np.ndarray,
        band: int = 1,
        use_inner: bool = True,
    ):
        """
        Write data for a tile.
        
        Args:
            tile: Tile information
            data: Data array (full tile including overlap)
            band: Band to write
            use_inner: If True, only write the inner (non-overlapping) portion
        """
        if self._dst_ds is None:
            raise RuntimeError("No destination file opened for writing")
        
        if use_inner and self.overlap > 0:
            # Extract inner portion
            inner_slice = tile.inner_slice
            inner_data = data[inner_slice]
            
            # Calculate inner window
            r0, r1, c0, c1 = tile.inner_global_bounds
            from rasterio.windows import Window
            inner_window = Window(col_off=c0, row_off=r0, width=c1-c0, height=r1-r0)
            
            self._dst_ds.write(inner_data.astype(self.dtype), band, window=inner_window)
        else:
            self._dst_ds.write(data.astype(self.dtype), band, window=tile.window)




def chunked_apply_array(
    arr: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = 0,
    nodata: float = NODATA_DEPTH,
    show_progress: bool = True,
) -> np.ndarray:
    """Apply a function to a large in-memory array using tiling.

    This is a lightweight companion to the raster-on-disk chunked pipeline and
    is primarily used for tests and small helper utilities.

    Args:
        arr: 2D array
        func: function mapping tile->tile (same shape)
        tile_size: tile edge size in pixels
        overlap: overlap in pixels (context margin); func sees the full tile including overlap
        nodata: nodata fill value
        show_progress: show tqdm progress if available

    Returns:
        Output array of same shape as arr
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError('chunked_apply_array expects a 2D array')
    H, W = a.shape
    tiles = list(generate_tiles(H, W, tile_size=tile_size, overlap=overlap))
    out = np.full((H, W), nodata, dtype=np.float32)

    it = tiles
    if show_progress and TQDM_AVAILABLE:
        it = tqdm(tiles, desc='chunked_apply_array', total=len(tiles))

    for tinfo in it:
        tile = a[tinfo.row_start:tinfo.row_end, tinfo.col_start:tinfo.col_end]
        res = func(tile)
        if res is None:
            continue
        res = np.asarray(res)
        if res.shape != tile.shape:
            raise ValueError(f'chunked_apply_array func returned {res.shape}, expected {tile.shape}')
        # write only inner (non-overlap) portion
        rs, re, cs, ce = tinfo.inner_global_bounds
        inner = res[tinfo.inner_slice]
        out[rs:re, cs:ce] = inner

    return out

def chunked_apply_raster(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    func: Callable[[np.ndarray], np.ndarray],
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = 0,
    dtype: np.dtype = np.float32,
    nodata: float = NODATA_DEPTH,
    desc: str = "Processing",
    disable_progress: bool = False,
):
    """
    Apply a function to a raster using chunked processing.
    
    This is a high-level convenience function for simple per-tile operations.
    
    Args:
        src_path: Input raster path
        dst_path: Output raster path
        func: Function that takes a numpy array and returns a numpy array
        tile_size: Tile size for processing
        overlap: Overlap between tiles
        dtype: Output data type
        nodata: Nodata value
        desc: Progress bar description
        disable_progress: Disable progress bar
        
    Example:
        # Apply a median filter to a large raster
        def median_filter(arr):
            from scipy.ndimage import median_filter
            return median_filter(arr, size=3)
        
        chunked_apply("large_dem.tif", "filtered_dem.tif", median_filter)
    """
    with ChunkedRasterProcessor(
        src_path, dst_path, tile_size, overlap, dtype, nodata
    ) as proc:
        total = proc.tile_count()
        tiles = proc.tiles()
        
        if TQDM_AVAILABLE and not disable_progress:
            tiles = tqdm(tiles, total=total, desc=desc)
        
        for tile in tiles:
            data = proc.read_tile(tile)
            result = func(data)
            proc.write_tile(tile, result)
    
    log.info("Chunked processing complete: %s", dst_path)



def chunked_apply(*args, **kwargs):
    """Backward-compatible dispatcher for chunked processing.

    Supports two calling conventions:

    1) In-memory arrays (tests & helpers):
        chunked_apply(arr: np.ndarray, func: Callable[[np.ndarray], np.ndarray], *, tile_size=..., overlap=..., ...)

       -> dispatches to chunked_apply_array(...)

    2) Raster-on-disk:
        chunked_apply(src_path, dst_path, func, tile_size=..., overlap=..., ...)

       -> dispatches to chunked_apply_raster(...)
    """
    import numpy as _np

    # In-memory conventions:
    #   (arr, func, ...)  OR  (func, arr, ...)
    if len(args) >= 2 and isinstance(args[0], _np.ndarray) and callable(args[1]):
        return chunked_apply_array(args[0], args[1], **kwargs)

    if len(args) >= 2 and callable(args[0]) and isinstance(args[1], _np.ndarray):
        return chunked_apply_array(args[1], args[0], **kwargs)

    # Raster mode requires at least 3 positional args: (src_path, dst_path, func, ...)
    if len(args) < 3:
        raise TypeError(
            "chunked_apply expected (arr, func, ...), (func, arr, ...), or (src_path, dst_path, func, ...)"
        )

    return chunked_apply_raster(*args, **kwargs)