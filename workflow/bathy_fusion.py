#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bathy_fusion.py – Multi-Source Bathymetry Fusion

Combines bathymetry from multiple sources with configurable priority and blending:
- SDB (Satellite-Derived Bathymetry) - coastal/nearshore clear water
- River cross-section interpolation - turbid rivers
- Measured soundings - sonar, multibeam, lidar
- CUDEM/existing DEMs - baseline

Fusion Strategies:
------------------
1. "priority": Higher-priority source overwrites lower
2. "uncertainty": Use source with lowest uncertainty per pixel
3. "average": Weighted average where sources overlap
4. "blend": Gradual taper at source boundaries

Output:
-------
- Combined bathymetry raster
- Provenance raster (which source contributed each pixel)
- Combined uncertainty raster (if inputs have uncertainty)

Provenance Codes:
-----------------
0 = NoData
1 = SDB (satellite-derived)
2 = River (cross-section interpolation)
3 = Measured (sonar/lidar/survey)
4 = Existing DEM
5 = Averaged (multiple sources)
6 = Blended (tapered transition)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Use centralized logging - get logger, don't configure root here
log = logging.getLogger("bathy_fusion")


# Provenance codes
PROV_NODATA = 0
PROV_SDB = 1
PROV_RIVER = 2
PROV_MEASURED = 3
PROV_DEM = 4
PROV_AVERAGED = 5
PROV_BLENDED = 6


@dataclass
class FusionConfig:
    """Configuration for bathymetry fusion."""
    
    # Input rasters (all optional)
    sdb_raster: Optional[Path] = None
    river_raster: Optional[Path] = None
    measured_raster: Optional[Path] = None
    dem_raster: Optional[Path] = None
    
    # Uncertainty rasters (optional)
    sdb_uncertainty: Optional[Path] = None
    river_uncertainty: Optional[Path] = None
    measured_uncertainty: Optional[Path] = None
    
    # Output
    out_dir: Path = Path("output/fusion")
    
    # Strategy
    #   - priority: take the first valid value in priority_order
    #   - uncertainty: take the valid value with lowest uncertainty
    #   - average: average all valid values
    #   - blend: priority with edge taper (when two sources overlap)
    #   - weighted_overlap: fixed-weight blend in overlap regions (typically 70/30)
    strategy: str = "priority"  # priority, uncertainty, average, blend, weighted_overlap, spatial_taper
    
    # Priority order (highest first)
    priority_order: List[str] = field(default_factory=lambda: ["measured", "sdb", "river", "dem"])
    
    # Blending
    taper_m: float = 50.0
    blend_overlap: bool = True

    # Weighted overlap (only used when strategy == "weighted_overlap")
    # In overlap pixels where both the primary and secondary sources are valid:
    #     out = primary_weight * primary + secondary_weight * secondary
    # Outside overlaps, the first valid value in priority_order is used.
    primary_weight: float = 0.7
    secondary_weight: float = 0.3
    
    # Processing
    nodata: float = -9999.0
    template_raster: Optional[Path] = None  # Use this grid if set


@dataclass
class FusionResult:
    """Results from fusion."""
    
    status: str = "pending"
    combined_raster: Optional[Path] = None
    provenance_raster: Optional[Path] = None
    uncertainty_raster: Optional[Path] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def _read_raster(path: Path) -> Tuple[np.ndarray, dict, Any]:
    """Read a raster and return (data, profile, nodata)."""
    import rasterio
    
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float32)
        profile = ds.profile.copy()
        nodata = ds.nodata
        
        # Mark nodata as NaN for easier handling
        if nodata is not None:
            data[data == nodata] = np.nan
        
        return data, profile, nodata


def _write_raster(path: Path, data: np.ndarray, profile: dict, nodata: float = -9999.0):
    """Write a raster."""
    import rasterio
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Handle NaN → nodata
    out_data = data.copy()
    out_data[~np.isfinite(out_data)] = nodata
    
    profile.update(dtype="float32", nodata=nodata, compress="deflate")
    
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out_data.astype(np.float32), 1)


def _validate_crs_compatibility(src_crs, dst_crs, src_name: str = "source", dst_name: str = "destination") -> bool:
    """
    Validate CRS compatibility between source and destination.
    
    Args:
        src_crs: Source CRS (can be any format rasterio accepts)
        dst_crs: Destination CRS
        src_name: Name for logging
        dst_name: Name for logging
        
    Returns:
        True if reprojection is safe, False if there may be issues
        
    Logs warnings for potential issues:
        - Mismatched geographic vs projected CRS
        - Large coordinate system differences
        - Missing CRS definitions
    """
    from pyproj import CRS
    
    if src_crs is None:
        log.warning(f"[CRS] {src_name} has no CRS defined - assuming EPSG:4326")
        return False
    
    if dst_crs is None:
        log.warning(f"[CRS] {dst_name} has no CRS defined - assuming EPSG:4326")
        return False
    
    try:
        src = CRS.from_user_input(src_crs)
        dst = CRS.from_user_input(dst_crs)
        
        # Check if CRS are equal (no reprojection needed)
        if src == dst:
            return True
        
        # Warn about geographic vs projected mismatches
        if src.is_geographic and not dst.is_geographic:
            log.info(f"[CRS] Reprojecting {src_name} from geographic to projected CRS")
        elif not src.is_geographic and dst.is_geographic:
            log.info(f"[CRS] Reprojecting {src_name} from projected to geographic CRS")
        
        # Warn about different datums
        if src.datum != dst.datum:
            log.warning(f"[CRS] Different datums: {src_name}={src.datum}, {dst_name}={dst.datum}")
        
        # Check for very different coordinate systems (e.g., different hemispheres)
        src_bounds = src.area_of_use
        dst_bounds = dst.area_of_use
        if src_bounds and dst_bounds:
            # Check if there's any overlap in area of use
            if (src_bounds.east < dst_bounds.west or src_bounds.west > dst_bounds.east or
                src_bounds.north < dst_bounds.south or src_bounds.south > dst_bounds.north):
                log.warning(
                    f"[CRS] {src_name} and {dst_name} have non-overlapping areas of use. "
                    "This may indicate a CRS mismatch."
                )
        
        return True
        
    except Exception as e:
        log.error(f"[CRS] Failed to validate CRS compatibility: {e}")
        return False


def _reproject_to_template(
    src_data: np.ndarray,
    src_profile: dict,
    template_profile: dict,
) -> np.ndarray:
    """
    Reproject source data to template grid.
    
    Args:
        src_data: Source raster data array
        src_profile: Source raster profile (including CRS and transform)
        template_profile: Template raster profile to match
        
    Returns:
        Reprojected data array matching template grid
        
    Note:
        CRS validation is performed before reprojection.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    
    # Validate CRS compatibility
    _validate_crs_compatibility(
        src_profile.get("crs"),
        template_profile.get("crs"),
        "source",
        "template"
    )
    
    dst_data = np.full(
        (template_profile["height"], template_profile["width"]),
        np.nan,
        dtype=np.float32,
    )
    
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=template_profile["transform"],
        dst_crs=template_profile["crs"],
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    
    return dst_data


def _compute_distance_from_edge(valid_mask: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Compute distance from edge of valid region."""
    try:
        from scipy.ndimage import distance_transform_edt
        
        # Distance from invalid pixels (edges of valid region)
        dist = distance_transform_edt(valid_mask) * pixel_size_m
        return dist
    except ImportError:
        log.warning("scipy not available, distance transform skipped")
        return np.zeros_like(valid_mask, dtype=np.float32)


def fuse_bathymetry(cfg: FusionConfig) -> FusionResult:
    """
    Fuse bathymetry sources into a combined raster on a common grid.

    Fixes (v0.5.1 style):
      - Uses CRS-aware meters-per-pixel (geodesic for geographic CRS; no 111000 hack)
      - Uses chunked/windowed processing for memory safety (writes outputs incrementally)
      - Treats nodata consistently by reading as masked arrays and converting to NaN
    """
    import rasterio
    from rasterio.windows import Window
    from pyproj import CRS, Geod
    from scipy.ndimage import distance_transform_edt
    import math

    result = FusionResult()

    # -------------------------
    # Resolve input paths
    # -------------------------
    src_paths = {
        "sdb": cfg.sdb_raster,
        "river": cfg.river_raster,
        "measured": cfg.measured_raster,
        "dem": cfg.dem_raster,
    }
    uncert_paths = {
        "sdb": cfg.sdb_uncertainty,
        "river": cfg.river_uncertainty,
        "measured": cfg.measured_uncertainty,
    }

    # Ensure at least one source exists
    if not any(p is not None for p in src_paths.values()):
        result.error = "No input rasters provided."
        return result

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Choose template grid
    # -------------------------
    template_path = cfg.template_raster or cfg.dem_raster or cfg.sdb_raster or cfg.river_raster or cfg.measured_raster
    if template_path is None:
        result.error = "Could not determine template raster."
        return result

    # Open template
    with rasterio.open(template_path) as tmpl:
        template_profile = tmpl.profile.copy()
        template_crs = tmpl.crs
        template_transform = tmpl.transform
        height, width = tmpl.height, tmpl.width
        bounds = tmpl.bounds

    # Force float output
    template_profile.update(dtype="float32", nodata=np.nan, compress="deflate")

    # -------------------------
    # CRS-aware pixel size (meters)
    # -------------------------
    def _pixel_size_m() -> float:
        if template_transform is None:
            return 10.0
        dx = abs(template_transform.a)
        dy = abs(template_transform.e)
        if template_crs is None:
            return float(dx)
        crs_obj = CRS.from_user_input(template_crs)
        if not crs_obj.is_geographic:
            # assume meters
            return float(max(dx, dy))
        # Geographic: compute geodesic meters per pixel at raster center latitude
        geod = Geod(ellps="WGS84")
        lon_c = (bounds.left + bounds.right) / 2.0
        lat_c = (bounds.bottom + bounds.top) / 2.0
        # X distance: (lon,lat) -> (lon+dx,lat)
        _, _, dist_x = geod.inv(lon_c, lat_c, lon_c + dx, lat_c)
        # Y distance: (lon,lat) -> (lon,lat+dy)
        _, _, dist_y = geod.inv(lon_c, lat_c, lon_c, lat_c + dy)
        return float(max(abs(dist_x), abs(dist_y), 1e-6))

    pixel_size_m = _pixel_size_m()

    # -------------------------
    # Align all sources to template grid (on disk, streaming)
    # -------------------------
    aligned_dir = cfg.out_dir / "_aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    def _aligned_path(tag: str, src: Path) -> Path:
        return aligned_dir / f"{tag}_aligned.tif"

    def _needs_alignment(src_profile: dict) -> bool:
        # Compare CRS, transform, shape
        if src_profile.get("crs") != template_profile.get("crs"):
            return True
        if src_profile.get("transform") != template_profile.get("transform"):
            return True
        if src_profile.get("height") != template_profile.get("height") or src_profile.get("width") != template_profile.get("width"):
            return True
        return False

    def _reproject_to_template_file(src_path: Path, dst_path: Path) -> Path:
        # Stream reprojection into a GeoTIFF on the template grid
        from rasterio.warp import reproject, Resampling
        with rasterio.open(src_path) as src:
            src_prof = src.profile.copy()
            if not _needs_alignment(src_prof):
                return src_path

            dst_prof = template_profile.copy()
            dst_prof.update(
                dtype="float32",
                nodata=np.nan,
                count=1,
                compress="deflate",
            )
            # Make sure directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(dst_path, "w", **dst_prof) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=template_transform,
                    dst_crs=template_crs,
                    resampling=Resampling.nearest,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                )
        return dst_path

    # Align data rasters
    aligned_src = {}
    for tag, p in src_paths.items():
        if p is None:
            aligned_src[tag] = None
            continue
        ap = _reproject_to_template_file(Path(p), _aligned_path(tag, Path(p)))
        aligned_src[tag] = ap

    # Align uncertainty rasters (if any)
    aligned_unc = {}
    for tag, p in uncert_paths.items():
        if p is None:
            aligned_unc[tag] = None
            continue
        ap = _reproject_to_template_file(Path(p), _aligned_path(f"{tag}_unc", Path(p)))
        aligned_unc[tag] = ap

    # -------------------------
    # Decide chunking
    # -------------------------
    try:
        from chunked_processing import should_use_chunked_processing, generate_tiles, count_tiles
        use_chunked = should_use_chunked_processing(height, width)
    except Exception:
        # Fall back to chunked always for safety if helper isn't available
        use_chunked = True
        generate_tiles = None
        count_tiles = None

    tile_size = getattr(cfg, "tile_size", 1024)
    taper_m = float(cfg.taper_m or 0.0)
    taper_px = int(max(0, math.ceil(taper_m / max(pixel_size_m, 1e-6))))

    # For blend, we need overlap so distance-to-edge is correct near tile boundaries
    overlap_px = taper_px + 2 if cfg.strategy == "blend" and taper_px > 0 else 0

    # -------------------------
    # Prepare outputs (incremental writes)
    # -------------------------
    combined_path = cfg.out_dir / "BATHY_FUSED.tif"
    provenance_path = cfg.out_dir / "BATHY_FUSED_PROVENANCE.tif"
    uncertainty_path = cfg.out_dir / "BATHY_FUSED_UNCERTAINTY.tif"

    result.combined_raster = combined_path
    result.provenance_raster = provenance_path

    prov_profile = template_profile.copy()
    prov_profile.update(dtype="uint8", nodata=0, compress="deflate")

    # Open sources
    src_ds = {}
    unc_ds = {}
    try:
        for tag, p in aligned_src.items():
            if p is not None:
                src_ds[tag] = rasterio.open(p)
        for tag, p in aligned_unc.items():
            if p is not None:
                unc_ds[tag] = rasterio.open(p)

        with rasterio.open(combined_path, "w", **template_profile) as out_depth, \
             rasterio.open(provenance_path, "w", **prov_profile) as out_prov, \
             rasterio.open(uncertainty_path, "w", **template_profile) as out_unc:

            # Helper to read window as float with NaN nodata
            def _read_win(ds, win: Window) -> np.ndarray:
                arr = ds.read(1, window=win, masked=True).astype(np.float32)
                return arr.filled(np.nan)

            # Helper to write inner portion (when overlap used)
            def _write_inner(win_outer: Window, win_inner: Window, data_outer: np.ndarray, prov_outer: np.ndarray, unc_outer: Optional[np.ndarray]):
                # win_inner is in global coordinates, so we need to slice local array region
                ro = int(win_inner.row_off - win_outer.row_off)
                co = int(win_inner.col_off - win_outer.col_off)
                hh = int(win_inner.height)
                ww = int(win_inner.width)
                d = data_outer[ro:ro+hh, co:co+ww]
                p = prov_outer[ro:ro+hh, co:co+ww]
                out_depth.write(d.astype(np.float32), 1, window=win_inner)
                out_prov.write(p.astype(np.uint8), 1, window=win_inner)
                if unc_outer is not None:
                    u = unc_outer[ro:ro+hh, co:co+ww]
                    out_unc.write(u.astype(np.float32), 1, window=win_inner)

            # Determine tiles
            if use_chunked and generate_tiles is not None:
                tiles = list(generate_tiles(height, width, tile_size=tile_size, overlap=overlap_px))
            else:
                class _SimpleTile:
                    def __init__(self):
                        self.window = Window(0, 0, width, height)
                        self.inner_window = Window(0, 0, width, height)
                        self.inner_global_bounds = (0, height, 0, width)
                tiles = [_SimpleTile()]

            total_tiles = len(tiles)

            if TQDM_AVAILABLE and use_chunked and generate_tiles is not None:
                tiles_iter = tqdm(tiles, total=total_tiles, desc="Fusing bathymetry (chunked)")
            else:
                tiles_iter = tiles

            # Stats counters
            cnt = { "sdb":0, "river":0, "measured":0, "dem":0, "averaged":0, "blended":0, "total":0 }

            for tile in tiles_iter:
                # Build windows
                win_outer = tile.window
                win_inner = tile.inner_window

                # Read sources for this tile
                sdb = _read_win(src_ds["sdb"], win_outer) if "sdb" in src_ds else None
                river = _read_win(src_ds["river"], win_outer) if "river" in src_ds else None
                meas = _read_win(src_ds["measured"], win_outer) if "measured" in src_ds else None
                dem = _read_win(src_ds["dem"], win_outer) if "dem" in src_ds else None

                sdb_u = _read_win(unc_ds["sdb"], win_outer) if "sdb" in unc_ds else None
                river_u = _read_win(unc_ds["river"], win_outer) if "river" in unc_ds else None
                meas_u = _read_win(unc_ds["measured"], win_outer) if "measured" in unc_ds else None

                # Valid masks
                valid = {
                    "sdb": (np.isfinite(sdb) if sdb is not None else None),
                    "river": (np.isfinite(river) if river is not None else None),
                    "measured": (np.isfinite(meas) if meas is not None else None),
                    "dem": (np.isfinite(dem) if dem is not None else None),
                }

                out = np.full(sdb.shape if sdb is not None else (int(win_outer.height), int(win_outer.width)),
                              np.nan, dtype=np.float32)
                prov = np.zeros(out.shape, dtype=np.uint8)
                out_u = np.full(out.shape, np.nan, dtype=np.float32) if (sdb_u is not None or river_u is not None or meas_u is not None) else None

                # -------------------------
                # Strategy implementations (tile-local)
                # -------------------------
                if cfg.strategy == "priority":
                    for src in cfg.priority_order:
                        arr = {"sdb":sdb, "river":river, "measured":meas, "dem":dem}.get(src)
                        m = valid.get(src)
                        if arr is None or m is None:
                            continue
                        take = np.isnan(out) & m
                        if not take.any():
                            continue
                        out[take] = arr[take]
                        if src == "sdb":
                            prov[take] = PROV_SDB
                        elif src == "river":
                            prov[take] = PROV_RIVER
                        elif src == "measured":
                            prov[take] = PROV_MEASURED
                        elif src == "dem":
                            prov[take] = PROV_DEM
                        if out_u is not None:
                            ua = {"sdb":sdb_u, "river":river_u, "measured":meas_u}.get(src)
                            if ua is not None:
                                out_u[take] = ua[take]

                elif cfg.strategy == "weighted_overlap":
                    # Fixed-weight blend in overlap regions for the first two *available* sources
                    # in the priority order (typically: sdb + river). Outside overlaps, this is
                    # identical to strict priority.

                    # Determine primary and secondary sources that exist for this run
                    src_to_arr = {"measured": meas, "sdb": sdb, "river": river, "dem": dem}
                    available = [s for s in cfg.priority_order if src_to_arr.get(s) is not None]
                    primary = available[0] if available else None
                    secondary = available[1] if len(available) > 1 else None

                    # 1) Fill by strict priority
                    for src in cfg.priority_order:
                        arr = src_to_arr.get(src)
                        m = valid.get(src)
                        if arr is None or m is None:
                            continue
                        take = np.isnan(out) & m
                        if not take.any():
                            continue
                        out[take] = arr[take]
                        if src == "sdb":
                            prov[take] = PROV_SDB
                        elif src == "river":
                            prov[take] = PROV_RIVER
                        elif src == "measured":
                            prov[take] = PROV_MEASURED
                        elif src == "dem":
                            prov[take] = PROV_DEM
                        if out_u is not None:
                            ua = {"sdb": sdb_u, "river": river_u, "measured": meas_u}.get(src)
                            if ua is not None:
                                out_u[take] = ua[take]

                    # 2) Overwrite overlap pixels with the requested weights (primary vs secondary)
                    if primary is not None and secondary is not None:
                        a = src_to_arr.get(primary)
                        b = src_to_arr.get(secondary)
                        if a is not None and b is not None:
                            m = np.isfinite(a) & np.isfinite(b)
                            if m.any():
                                w1 = float(cfg.primary_weight)
                                w2 = float(cfg.secondary_weight)
                                s = w1 + w2
                                if s <= 0:
                                    w1, w2, s = 0.7, 0.3, 1.0
                                w1 /= s
                                w2 /= s
                                out[m] = (w1 * a[m]) + (w2 * b[m])
                                prov[m] = PROV_BLENDED
                                if out_u is not None:
                                    ua = {"sdb": sdb_u, "river": river_u, "measured": meas_u}.get(primary)
                                    ub = {"sdb": sdb_u, "river": river_u, "measured": meas_u}.get(secondary)
                                    if ua is not None and ub is not None:
                                        out_u[m] = (w1 * ua[m]) + (w2 * ub[m])

                
                elif cfg.strategy == "spatial_taper":
                    # Spatially-aware blend for SDB+River overlaps.
                    # Start with strict priority to fill gaps, then in overlap pixels blend using
                    # base weights (primary/secondary) modulated by distance-to-river-edge.
                    #
                    # Distance-to-river-edge is computed *inside the river-valid mask*: pixels on
                    # the river boundary have distance 0 and shift toward the base weights; pixels
                    # toward the river center have larger distances and trend toward 100% river.
                    #
                    # Requires SciPy for a true distance transform; if SciPy is unavailable, falls
                    # back to fixed-weight overlap (cfg.primary_weight/cfg.secondary_weight).

                    src_to_arr = {"measured": meas, "sdb": sdb, "river": river, "dem": dem}
                    available = [s for s in cfg.priority_order if src_to_arr.get(s) is not None]
                    primary = available[0] if available else None
                    secondary = available[1] if len(available) > 1 else None

                    # 1) Fill by strict priority
                    for src in cfg.priority_order:
                        arr = src_to_arr.get(src)
                        m = valid.get(src)
                        if arr is None or m is None:
                            continue
                        take = np.isnan(out) & m
                        if not take.any():
                            continue
                        out[take] = arr[take]
                        if src == "sdb":
                            prov[take] = PROV_SDB
                        elif src == "river":
                            prov[take] = PROV_RIVER
                        elif src == "measured":
                            prov[take] = PROV_MEASURED
                        elif src == "dem":
                            prov[take] = PROV_DEM
                        if out_u is not None:
                            ua = {"sdb": sdb_u, "river": river_u, "measured": meas_u}.get(src)
                            if ua is not None:
                                out_u[take] = ua[take]

                    # 2) Blend overlap pixels (river + sdb only)
                    if primary is not None and secondary is not None:
                        a = src_to_arr.get(primary)
                        b = src_to_arr.get(secondary)
                        if a is not None and b is not None:
                            m = np.isfinite(a) & np.isfinite(b)

                            # Normalize base weights
                            w1 = float(cfg.primary_weight)
                            w2 = float(cfg.secondary_weight)
                            s = w1 + w2
                            if s <= 0:
                                w1, w2, s = 0.7, 0.3, 1.0
                            w1 /= s
                            w2 /= s

                            # Only support SDB+River spatial taper (other combinations fallback to fixed overlap)
                            combo = {primary, secondary}
                            if m.any() and combo == {"sdb", "river"} and river is not None and sdb is not None:
                                # Determine which array is river vs sdb
                                river_arr = river if primary == "river" else b if secondary == "river" else river
                                sdb_arr = sdb if primary == "sdb" else b if secondary == "sdb" else sdb

                                # Base river weight from normalized weights
                                base_river = w1 if primary == "river" else w2 if secondary == "river" else 0.5

                                try:
                                    from scipy.ndimage import distance_transform_edt

                                    river_mask = np.isfinite(river_arr)
                                    if river_mask.any():
                                        # Distance to nearest outside-of-river pixel (edge distance inside river)
                                        dist_px = distance_transform_edt(river_mask)
                                        taper_px_local = max(1, int(taper_px)) if taper_px > 0 else 1
                                        alpha = np.clip(dist_px / float(taper_px_local), 0.0, 1.0)

                                        # Modulate: at edge (alpha=0) => base weights; toward center => 100% river
                                        w_r = base_river + alpha * (1.0 - base_river)
                                        w_s = 1.0 - w_r

                                        mm = m & river_mask & np.isfinite(sdb_arr)
                                        if mm.any():
                                            out[mm] = (w_r[mm] * river_arr[mm]) + (w_s[mm] * sdb_arr[mm])
                                            prov[mm] = PROV_BLENDED
                                            if out_u is not None:
                                                # Uncertainty blend if available
                                                ua = river_u if primary == "river" else (river_u if secondary == "river" else None)
                                                ub = sdb_u if primary == "sdb" else (sdb_u if secondary == "sdb" else None)
                                                if ua is not None and ub is not None:
                                                    out_u[mm] = (w_r[mm] * ua[mm]) + (w_s[mm] * ub[mm])
                                except Exception:
                                    # Fallback: fixed overlap blend
                                    if m.any():
                                        out[m] = (w1 * a[m]) + (w2 * b[m])
                                        prov[m] = PROV_BLENDED
                                        if out_u is not None:
                                            ua = {"sdb": sdb_u, "river": river_u, "measured": meas_u}.get(primary)
                                            ub = {"sdb": sdb_u, "river": river_u, "measured": meas_u}.get(secondary)
                                            if ua is not None and ub is not None:
                                                out_u[m] = (w1 * ua[m]) + (w2 * ub[m])
                            else:
                                # Not SDB+River: fixed overlap blend
                                if m.any():
                                    out[m] = (w1 * a[m]) + (w2 * b[m])
                                    prov[m] = PROV_BLENDED
                                    if out_u is not None:
                                        ua = {"sdb": sdb_u, "river": river_u, "measured": meas_u}.get(primary)
                                        ub = {"sdb": sdb_u, "river": river_u, "measured": meas_u}.get(secondary)
                                        if ua is not None and ub is not None:
                                            out_u[m] = (w1 * ua[m]) + (w2 * ub[m])

                elif cfg.strategy == "uncertainty":
                    # Use lowest uncertainty among sources that have both depth and uncertainty
                    candidates = []
                    for src, arr, ua in [
                        ("measured", meas, meas_u),
                        ("sdb", sdb, sdb_u),
                        ("river", river, river_u),
                    ]:
                        if arr is None or ua is None:
                            continue
                        m = np.isfinite(arr) & np.isfinite(ua)
                        if m.any():
                            candidates.append((src, arr, ua, m))
                    if not candidates:
                        # Fallback to priority
                        for src in cfg.priority_order:
                            arr = {"sdb":sdb, "river":river, "measured":meas, "dem":dem}.get(src)
                            m = valid.get(src)
                            if arr is None or m is None:
                                continue
                            take = np.isnan(out) & m
                            out[take] = arr[take]
                            prov[take] = {"sdb":PROV_SDB,"river":PROV_RIVER,"measured":PROV_MEASURED,"dem":PROV_DEM}[src]
                    else:
                        # Initialize with first
                        src0, arr0, u0, m0 = candidates[0]
                        out[m0] = arr0[m0]
                        if out_u is not None:
                            out_u[m0] = u0[m0]
                        prov[m0] = {"sdb":PROV_SDB,"river":PROV_RIVER,"measured":PROV_MEASURED}.get(src0,0)
                        # Compare others
                        for src, arr, ua, m in candidates[1:]:
                            better = m & np.isfinite(out_u) & (ua < out_u)
                            out[better] = arr[better]
                            out_u[better] = ua[better]
                            prov[better] = {"sdb":PROV_SDB,"river":PROV_RIVER,"measured":PROV_MEASURED}.get(src,0)

                elif cfg.strategy == "average":
                    stack = []
                    prov_stack = []
                    for src, arr in [("measured", meas), ("sdb", sdb), ("river", river), ("dem", dem)]:
                        if arr is None:
                            continue
                        stack.append(arr)
                        prov_stack.append(src)
                    if stack:
                        arr3 = np.stack(stack, axis=0)
                        out = np.nanmean(arr3, axis=0).astype(np.float32)
                        prov[np.isfinite(out)] = PROV_AVERAGED
                        if out_u is not None:
                            # uncertainty: take nanmean of available uncertainties
                            ustack = []
                            for src in prov_stack:
                                ua = {"measured":meas_u, "sdb":sdb_u, "river":river_u}.get(src)
                                if ua is not None:
                                    ustack.append(ua)
                            if ustack:
                                out_u = np.nanmean(np.stack(ustack, axis=0), axis=0).astype(np.float32)

                elif cfg.strategy == "blend":
                    # Blend SDB and river near edges of SDB coverage; measured always overrides if present
                    if sdb is None and river is None:
                        pass
                    else:
                        # Start with priority measured > blend(sdb,river) > dem
                        if meas is not None:
                            m_meas = np.isfinite(meas)
                            out[m_meas] = meas[m_meas]
                            prov[m_meas] = PROV_MEASURED
                            if out_u is not None and meas_u is not None:
                                out_u[m_meas] = meas_u[m_meas]

                        m_sdb = np.isfinite(sdb) if sdb is not None else np.zeros(out.shape, dtype=bool)
                        m_riv = np.isfinite(river) if river is not None else np.zeros(out.shape, dtype=bool)

                        # Regions where only one exists (and not already filled by measured)
                        empty = np.isnan(out)
                        only_sdb = empty & m_sdb & ~m_riv
                        only_riv = empty & m_riv & ~m_sdb
                        both = empty & m_sdb & m_riv

                        out[only_sdb] = sdb[only_sdb]
                        prov[only_sdb] = PROV_SDB
                        out[only_riv] = river[only_riv]
                        prov[only_riv] = PROV_RIVER

                        if taper_px > 0 and both.any():
                            # Distance inside SDB coverage to edge (invalid pixels)
                            dist = distance_transform_edt(m_sdb).astype(np.float32) * float(pixel_size_m)
                            w_sdb = np.clip(dist / max(taper_m, 1e-6), 0.0, 1.0).astype(np.float32)
                            w = w_sdb
                            out[both] = (w[both] * sdb[both] + (1.0 - w[both]) * river[both]).astype(np.float32)
                            prov[both] = PROV_BLENDED
                            if out_u is not None:
                                # conservative uncertainty: min of both (if both provided), else whichever exists
                                if sdb_u is not None and river_u is not None:
                                    out_u[both] = np.minimum(sdb_u[both], river_u[both])
                                elif sdb_u is not None:
                                    out_u[both] = sdb_u[both]
                                elif river_u is not None:
                                    out_u[both] = river_u[both]
                        else:
                            # No taper: choose SDB when both, else river
                            out[both] = sdb[both]
                            prov[both] = PROV_SDB

                        # Fill from DEM if still empty
                        if dem is not None:
                            m_dem = np.isfinite(dem)
                            take = np.isnan(out) & m_dem
                            out[take] = dem[take]
                            prov[take] = PROV_DEM

                else:
                    raise ValueError(f"Unknown fusion strategy: {cfg.strategy}")

                # Update per-tile stats on inner region only
                # Use inner window slice
                ro = int(win_inner.row_off - win_outer.row_off)
                co = int(win_inner.col_off - win_outer.col_off)
                hh = int(win_inner.height)
                ww = int(win_inner.width)
                prov_inner = prov[ro:ro+hh, co:co+ww]
                cnt["total"] += int((prov_inner > 0).sum())
                cnt["sdb"] += int((prov_inner == PROV_SDB).sum())
                cnt["river"] += int((prov_inner == PROV_RIVER).sum())
                cnt["measured"] += int((prov_inner == PROV_MEASURED).sum())
                cnt["dem"] += int((prov_inner == PROV_DEM).sum())
                cnt["averaged"] += int((prov_inner == PROV_AVERAGED).sum())
                cnt["blended"] += int((prov_inner == PROV_BLENDED).sum())

                # Write
                _write_inner(win_outer, win_inner, out, prov, out_u)

            # If we never wrote uncertainty (no uncertainty inputs), remove the file later
            wrote_uncert = any(p is not None for p in aligned_unc.values())

    finally:
        for ds in src_ds.values():
            try:
                ds.close()
            except Exception:
                pass
        for ds in unc_ds.values():
            try:
                ds.close()
            except Exception:
                pass

    # Clean up uncertainty output if unused (all NaN)
    try:
        with rasterio.open(uncertainty_path) as ds:
            test = ds.read(1, masked=True)
            if not np.isfinite(test.filled(np.nan)).any():
                # remove and null in result
                try:
                    uncertainty_path.unlink()
                except Exception:
                    pass
                result.uncertainty_raster = None
            else:
                result.uncertainty_raster = uncertainty_path
    except Exception:
        result.uncertainty_raster = None

    result.stats["pixel_counts"] = cnt
    result.stats["pixel_size_m"] = float(pixel_size_m)
    result.stats["taper_m"] = float(taper_m)
    result.stats["tile_size"] = int(tile_size)
    result.stats["overlap_px"] = int(overlap_px)

    return result

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-Source Bathymetry Fusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Inputs
    p.add_argument("--sdb", default=None, help="SDB bathymetry raster")
    p.add_argument("--river", default=None, help="River bathymetry raster")
    p.add_argument("--measured", default=None, help="Measured bathymetry raster")
    p.add_argument("--dem", default=None, help="Existing DEM raster")
    
    # Uncertainties
    p.add_argument("--sdb-uncert", default=None, help="SDB uncertainty raster")
    p.add_argument("--river-uncert", default=None, help="River uncertainty raster")
    
    # Output
    p.add_argument("--out-dir", default="output/fusion", help="Output directory")
    
    # Strategy
    p.add_argument("--strategy", choices=["priority", "uncertainty", "average", "blend"],
                   default="priority", help="Fusion strategy")
    p.add_argument("--priority", default="measured,sdb,river,dem",
                   help="Priority order (comma-separated, highest first)")
    
    # Blending
    p.add_argument("--taper", type=float, default=50.0, help="Taper distance for blending (m)")
    
    # Template
    p.add_argument("--template", default=None, help="Template raster for output grid")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    cfg = FusionConfig(
        sdb_raster=Path(args.sdb) if args.sdb else None,
        river_raster=Path(args.river) if args.river else None,
        measured_raster=Path(args.measured) if args.measured else None,
        dem_raster=Path(args.dem) if args.dem else None,
        sdb_uncertainty=Path(args.sdb_uncert) if args.sdb_uncert else None,
        river_uncertainty=Path(args.river_uncert) if args.river_uncert else None,
        out_dir=Path(args.out_dir),
        strategy=args.strategy,
        priority_order=[p.strip() for p in args.priority.split(",")],
        taper_m=args.taper,
        template_raster=Path(args.template) if args.template else None,
    )
    
    result = fuse_bathymetry(cfg)
    
    if result.status == "success":
        print(f"Combined bathymetry: {result.combined_raster}")
        sys.exit(0)
    else:
        print(f"Fusion failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()