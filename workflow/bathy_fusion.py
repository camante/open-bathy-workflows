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


import logging
log = logging.getLogger(__name__)
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import rasterio
from rasterio.windows import Window
from pyproj import CRS, Geod

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



def _sample_raster_values(path: Path, max_samples: int = 50000) -> np.ndarray:
    """Return a 1D sample of finite values from the first band of a raster.

    Uses windowed reads to avoid loading whole rasters. Returns empty array if no finite values found.
    """
    with rasterio.open(path) as ds:
        if ds.count != 1:
            raise ValueError(f"Expected single-band raster for depth/uncertainty: {path} (bands={ds.count})")
        h, w = ds.height, ds.width
        # Sample a grid of windows (deterministic) rather than random sampling
        step_y = max(1, int(round(h / 10)))
        step_x = max(1, int(round(w / 10)))
        vals = []
        for y0 in range(0, h, step_y):
            for x0 in range(0, w, step_x):
                win = Window(col_off=x0, row_off=y0, width=min(128, w - x0), height=min(128, h - y0))
                arr = ds.read(1, window=win, masked=True).astype("float64")
                if np.ma.isMaskedArray(arr):
                    a = arr.compressed()
                else:
                    a = arr.ravel()
                if a.size:
                    a = a[np.isfinite(a)]
                    if a.size:
                        vals.append(a)
                        if sum(v.size for v in vals) >= max_samples:
                            break
            if sum(v.size for v in vals) >= max_samples:
                break
        if not vals:
            return np.array([], dtype="float64")
        out = np.concatenate(vals)
        if out.size > max_samples:
            out = out[:max_samples]
        return out


def _validate_depth_raster(path: Optional[Path], *, name: str, expect_negative: bool = True) -> None:
    """Validate that a raster looks like a *depth* product (not imagery/mask).

    Hard-fails on multi-band rasters. Emits warnings (not errors) for suspicious ranges/signs.
    """
    if path is None:
        return
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} raster not found: {p}")
    with rasterio.open(p) as ds:
        if ds.count != 1:
            raise ValueError(f"{name} raster must be single-band (got {ds.count} bands): {p}")
        if ds.dtypes and ds.dtypes[0].startswith("uint"):
            log.warning("%s raster is unsigned integer (%s). This is unusual for depth; check units/selection: %s", name, ds.dtypes[0], p)

    vals = _sample_raster_values(p, max_samples=20000)
    if vals.size == 0:
        log.warning("%s raster has no finite samples (all nodata?). Continuing: %s", name, p)
        return

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    frac_neg = float(np.mean(vals < 0.0))
    frac_pos = float(np.mean(vals > 0.0))

    # Strong imagery smell: values in [0,1] or [0,10000] and almost all positive
    if frac_neg < 0.01 and frac_pos > 0.95 and ((0.0 <= vmin and vmax <= 1.5) or (0.0 <= vmin and vmax >= 500 and vmax <= 20000)):
        log.warning(
            "%s raster value range/sign looks imagery-like (min=%.3f max=%.3f frac_neg=%.3f). Verify you're not fusing reflectance/mask: %s",
            name, vmin, vmax, frac_neg, p
        )

    if expect_negative and frac_neg < 0.20:
        log.warning(
            "%s raster has low fraction of negative values (frac_neg=%.3f; min=%.3f max=%.3f). For negative-down depth workflows this is suspicious: %s",
            name, frac_neg, vmin, vmax, p
        )


def _validate_uncertainty_raster(path: Optional[Path], *, name: str) -> None:
    """Validate an uncertainty raster: single-band, mostly non-negative."""
    if path is None:
        return
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} raster not found: {p}")
    with rasterio.open(p) as ds:
        if ds.count != 1:
            raise ValueError(f"{name} raster must be single-band (got {ds.count} bands): {p}")
    vals = _sample_raster_values(p, max_samples=20000)
    if vals.size == 0:
        log.warning("%s uncertainty raster has no finite samples. Continuing: %s", name, p)
        return
    frac_neg = float(np.mean(vals < 0.0))
    if frac_neg > 0.01:
        log.warning("%s uncertainty has negative values (frac_neg=%.3f). Check units/sign: %s", name, frac_neg, p)
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
    


    # Optional river domain mask (1=river domain, 0=outside). When provided and
    # river_overrides_sdb_in_domain is True, river values override SDB wherever
    # mask==1 and river has a finite prediction (measured pixels remain highest authority).
    river_domain_mask: Optional[Path] = None
    river_overrides_sdb_in_domain: bool = False
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
    """Fuse multiple bathymetry sources into a single raster.

    Key scientific/operational policies enforced:
      - **Measured wins**: measured pixels are never overridden or blended.
      - **Tile-safe**: when a river-domain mask is supplied and enabled, river overrides SDB
        wherever (mask==1 and river is finite). This suppresses tile-to-tile SDB differences
        in overlapping corridors that create seam offsets.
      - **Uncertainty propagation**: when blending, uncertainties are combined using weighted
        root-sum-of-squares (RSS): sigma = sqrt((w1*s1)^2 + (w2*s2)^2).

    The function streams processing in tiles when rasters are large.
    """

    log = logging.getLogger("bathy_fusion")
    result = FusionResult(status="failed")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Choose template grid
    # -------------------------
    template_path = cfg.template_raster or cfg.dem_raster or cfg.sdb_raster or cfg.river_raster or cfg.measured_raster
    if template_path is None:
        result.error = "Could not determine template raster."
        return result

    with rasterio.open(template_path) as tmpl:
        template_profile = tmpl.profile.copy()
        template_crs = tmpl.crs
        template_transform = tmpl.transform
        height, width = tmpl.height, tmpl.width
        bounds = tmpl.bounds

    # Output nodata (use numeric nodata; NaN nodata is fragile in GDAL toolchains)
    nodata_out = float(getattr(cfg, "nodata", -9999.0) or -9999.0)

    # -------------------------
    # Input sanity checks (prevents fusing optical imagery/masks as 'depth')
    # -------------------------
    try:
        _validate_depth_raster(cfg.sdb_raster, name='SDB', expect_negative=True)
        _validate_depth_raster(cfg.river_raster, name='River', expect_negative=True)
        _validate_depth_raster(cfg.measured_raster, name='Measured', expect_negative=True)
        _validate_depth_raster(cfg.dem_raster, name='DEM', expect_negative=False)
        _validate_uncertainty_raster(cfg.sdb_uncertainty, name='SDB')
        _validate_uncertainty_raster(cfg.river_uncertainty, name='River')
        _validate_uncertainty_raster(cfg.measured_uncertainty, name='Measured')
        if cfg.river_domain_mask is not None and not Path(cfg.river_domain_mask).exists():
            raise FileNotFoundError(f'River domain mask not found: {cfg.river_domain_mask}')
    except Exception as e:
        # Fail closed: it's better to stop than to silently fuse the wrong product.
        result.error = f'Input validation failed: {e}'
        log.error(result.error)
        return result

    template_profile.update(
        dtype="float32",
        nodata=nodata_out,
        count=1,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    # -------------------------
    # CRS-aware pixel size (meters)
    # -------------------------
    def _pixel_size_m() -> float:
        if template_transform is None:
            return 10.0
        dx = abs(template_transform.a)
        dy = abs(template_transform.e)
        if template_crs is None:
            return float(max(dx, dy))
        crs_obj = CRS.from_user_input(template_crs)
        if not crs_obj.is_geographic:
            return float(max(dx, dy))
        geod = Geod(ellps="WGS84")
        lon_c = (bounds.left + bounds.right) / 2.0
        lat_c = (bounds.bottom + bounds.top) / 2.0
        _, _, dist_x = geod.inv(lon_c, lat_c, lon_c + dx, lat_c)
        _, _, dist_y = geod.inv(lon_c, lat_c, lon_c, lat_c + dy)
        return float(max(abs(dist_x), abs(dist_y), 1e-6))

    pixel_size_m = _pixel_size_m()

    # -------------------------
    # Align sources to template grid
    # -------------------------
    from rasterio.warp import reproject, Resampling

    aligned_dir = cfg.out_dir / "_aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    def _aligned_path(tag: str) -> Path:
        return aligned_dir / f"{tag}_aligned.tif"

    def _needs_alignment(src_profile: dict) -> bool:
        if src_profile.get("crs") != template_profile.get("crs"):
            return True
        if src_profile.get("transform") != template_profile.get("transform"):
            return True
        if src_profile.get("height") != template_profile.get("height") or src_profile.get("width") != template_profile.get("width"):
            return True
        return False

    def _reproject_to_template_file(src_path: Path, dst_path: Path, resampling: Resampling = Resampling.nearest) -> Path:
        with rasterio.open(src_path) as src:
            src_prof = src.profile.copy()
            if not _needs_alignment(src_prof):
                return src_path

            dst_prof = template_profile.copy()
            dst_prof.update(dtype="float32", nodata=nodata_out)
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(dst_path, "w", **dst_prof) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=template_transform,
                    dst_crs=template_crs,
                    resampling=resampling,
                    src_nodata=src.nodata,
                    dst_nodata=nodata_out,
                )
        return dst_path

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

    if not any(p is not None for p in src_paths.values()):
        result.error = "No input rasters provided."
        return result

    aligned_src: dict[str, Path | None] = {}
    for tag, p in src_paths.items():
        if p is None:
            aligned_src[tag] = None
            continue
        ap = _reproject_to_template_file(Path(p), _aligned_path(tag), resampling=Resampling.nearest)
        aligned_src[tag] = ap

    aligned_unc: dict[str, Path | None] = {}
    for tag, p in uncert_paths.items():
        if p is None:
            aligned_unc[tag] = None
            continue
        ap = _reproject_to_template_file(Path(p), _aligned_path(f"{tag}_unc"), resampling=Resampling.nearest)
        aligned_unc[tag] = ap

    # Optional river domain mask (aligned using nearest)
    aligned_domain_mask: Path | None = None
    if getattr(cfg, "river_domain_mask", None):
        mp = Path(getattr(cfg, "river_domain_mask"))
        if mp.exists():
            aligned_domain_mask = _reproject_to_template_file(mp, _aligned_path("river_domain_mask"), resampling=Resampling.nearest)
        else:
            log.warning("[FUSION] river_domain_mask does not exist: %s", str(mp))

    has_unc_inputs = any(p is not None for p in aligned_unc.values())

    # -------------------------
    # Decide chunking
    # -------------------------
    try:
        from chunked_processing import should_use_chunked_processing, generate_tiles, count_tiles, TQDM_AVAILABLE
        use_chunked = should_use_chunked_processing(height, width)
    except Exception:
        use_chunked = False
        generate_tiles = None
        count_tiles = None
        TQDM_AVAILABLE = False

    tile_size = int(getattr(cfg, "tile_size", 1024) or 1024)
    taper_m = float(getattr(cfg, "taper_m", 0.0) or 0.0)
    taper_px = int(max(0, round(taper_m / max(pixel_size_m, 1e-6))))

    strat = str(getattr(cfg, "strategy", "priority") or "priority").strip().lower()
    overlap_px = int(taper_px + 2) if strat in ("blend", "spatial_taper") and taper_px > 0 else 0

    # -------------------------
    # Prepare outputs
    # -------------------------
    combined_path = cfg.out_dir / "BATHY_FUSED.tif"
    provenance_path = cfg.out_dir / "BATHY_FUSED_PROVENANCE.tif"
    uncertainty_path = cfg.out_dir / "BATHY_FUSED_UNCERTAINTY.tif"

    result.combined_raster = combined_path
    result.provenance_raster = provenance_path

    prov_profile = template_profile.copy()
    prov_profile.update(dtype="uint8", nodata=0)

    src_ds: dict[str, rasterio.DatasetReader] = {}
    unc_ds: dict[str, rasterio.DatasetReader] = {}
    domain_ds: rasterio.DatasetReader | None = None

    from contextlib import ExitStack

    def _read_win(ds: rasterio.DatasetReader, win: Window) -> np.ndarray:
        arr = ds.read(1, window=win, masked=True).astype(np.float32)
        return arr.filled(np.nan)

    def _read_mask_win(ds: rasterio.DatasetReader, win: Window) -> np.ndarray:
        arr = ds.read(1, window=win, masked=True)
        a = arr.filled(0).astype(np.float32)
        return a > 0.5

    # Tiles iterator
    if use_chunked and generate_tiles is not None:
        tiles = list(generate_tiles(width, height, tile_size=tile_size, overlap=overlap_px))
        ntiles = len(tiles)
    else:
        class _SimpleTile:
            def __init__(self, window: Window):
                self.window = window
                self.inner_window = window
                self.inner_global_bounds = (int(window.row_off), int(window.row_off + window.height), int(window.col_off), int(window.col_off + window.width))
        tiles = [_SimpleTile(Window(0, 0, width, height))]
        ntiles = 1

    tiles_iter = tiles
    if TQDM_AVAILABLE and ntiles > 1:
        try:
            tiles_iter = tqdm(tiles, total=ntiles, desc="Fusing", unit="tile")
        except Exception:
            tiles_iter = tiles

    # Stats counters
    cnt = {"sdb": 0, "river": 0, "measured": 0, "dem": 0, "averaged": 0, "blended": 0, "total": 0}

    wrote_unc_any = False

    try:
        for tag, p in aligned_src.items():
            if p is not None:
                src_ds[tag] = rasterio.open(p)
        if has_unc_inputs:
            for tag, p in aligned_unc.items():
                if p is not None:
                    unc_ds[tag] = rasterio.open(p)
        if aligned_domain_mask is not None:
            domain_ds = rasterio.open(aligned_domain_mask)

        with ExitStack() as stack:
            out_depth = stack.enter_context(rasterio.open(combined_path, "w", **template_profile))
            out_prov = stack.enter_context(rasterio.open(provenance_path, "w", **prov_profile))
            out_unc = None
            if has_unc_inputs:
                out_unc = stack.enter_context(rasterio.open(uncertainty_path, "w", **template_profile))

            def _write_inner(win_outer: Window, win_inner: Window, data_outer: np.ndarray, prov_outer: np.ndarray, unc_outer: np.ndarray | None):
                nonlocal wrote_unc_any
                ro = int(win_inner.row_off - win_outer.row_off)
                co = int(win_inner.col_off - win_outer.col_off)
                hh = int(win_inner.height)
                ww = int(win_inner.width)

                d = data_outer[ro:ro + hh, co:co + ww]
                p = prov_outer[ro:ro + hh, co:co + ww]

                d_out = d.copy()
                d_out[~np.isfinite(d_out)] = nodata_out
                out_depth.write(d_out.astype(np.float32), 1, window=win_inner)
                out_prov.write(p.astype(np.uint8), 1, window=win_inner)

                if out_unc is not None and unc_outer is not None:
                    u = unc_outer[ro:ro + hh, co:co + ww]
                    if np.isfinite(u).any():
                        wrote_unc_any = True
                    u_out = u.copy()
                    u_out[~np.isfinite(u_out)] = nodata_out
                    out_unc.write(u_out.astype(np.float32), 1, window=win_inner)

            # -------------------------
            # Per-tile processing
            # -------------------------
            for tile in tiles_iter:
                win_outer = tile.window
                if hasattr(tile, "inner_window"):
                    win_inner = tile.inner_window
                else:
                    r0, r1, c0, c1 = tile.inner_global_bounds
                    win_inner = Window(col_off=int(c0), row_off=int(r0), width=int(c1 - c0), height=int(r1 - r0))

                shape = (int(win_outer.height), int(win_outer.width))

                sdb = _read_win(src_ds["sdb"], win_outer) if "sdb" in src_ds else None
                river = _read_win(src_ds["river"], win_outer) if "river" in src_ds else None
                meas = _read_win(src_ds["measured"], win_outer) if "measured" in src_ds else None
                dem = _read_win(src_ds["dem"], win_outer) if "dem" in src_ds else None

                sdb_u = _read_win(unc_ds["sdb"], win_outer) if "sdb" in unc_ds else None
                river_u = _read_win(unc_ds["river"], win_outer) if "river" in unc_ds else None
                meas_u = _read_win(unc_ds["measured"], win_outer) if "measured" in unc_ds else None

                domain_mask = _read_mask_win(domain_ds, win_outer) if domain_ds is not None else None

                # Suppress SDB anywhere the river-domain mask is true.
                # This avoids tile-to-tile seams caused by SDB training differences within the river corridor
                # and ensures river (or measured) is the only contributor inside the river domain.
                if (domain_mask is not None) and bool(getattr(cfg, 'river_overrides_sdb_in_domain', False)) and (sdb is not None):
                    sdb = sdb.copy()
                    sdb[domain_mask] = float('nan')
                    if sdb_u is not None:
                        sdb_u = sdb_u.copy()
                        sdb_u[domain_mask] = float('nan')


                out = np.full(shape, np.nan, dtype=np.float32)
                prov = np.zeros(shape, dtype=np.uint8)
                out_u = np.full(shape, np.nan, dtype=np.float32) if has_unc_inputs else None

                src_to_arr = {"measured": meas, "sdb": sdb, "river": river, "dem": dem}
                src_to_unc = {"measured": meas_u, "sdb": sdb_u, "river": river_u}
                src_to_prov = {"sdb": PROV_SDB, "river": PROV_RIVER, "measured": PROV_MEASURED, "dem": PROV_DEM}

                def _priority_fill(order: list[str]):
                    for src in order:
                        arr = src_to_arr.get(src)
                        if arr is None:
                            continue
                        m = np.isfinite(arr)
                        take = np.isnan(out) & m
                        if not take.any():
                            continue
                        out[take] = arr[take]
                        prov[take] = src_to_prov.get(src, 0)
                        if out_u is not None:
                            ua = src_to_unc.get(src)
                            if ua is not None:
                                out_u[take] = ua[take]

                if strat == "priority":
                    _priority_fill(cfg.priority_order)

                elif strat == "weighted_overlap":
                    _priority_fill(cfg.priority_order)

                    # Only blend SDB+River (never measured)
                    blend_order = [s for s in cfg.priority_order if s in ("sdb", "river") and src_to_arr.get(s) is not None]
                    if len(blend_order) >= 2:
                        primary, secondary = blend_order[0], blend_order[1]
                        a = src_to_arr.get(primary)
                        b = src_to_arr.get(secondary)
                        if a is not None and b is not None:
                            m = np.isfinite(a) & np.isfinite(b)
                            if meas is not None:
                                m &= ~np.isfinite(meas)
                            if m.any():
                                w1 = float(getattr(cfg, "primary_weight", 0.7) or 0.7)
                                w2 = float(getattr(cfg, "secondary_weight", 0.3) or 0.3)
                                s = w1 + w2
                                if s <= 0:
                                    w1, w2, s = 0.7, 0.3, 1.0
                                w1 /= s
                                w2 /= s
                                out[m] = (w1 * a[m]) + (w2 * b[m])
                                prov[m] = PROV_BLENDED
                                if out_u is not None:
                                    ua = src_to_unc.get(primary)
                                    ub = src_to_unc.get(secondary)
                                    if ua is not None and ub is not None:
                                        out_u[m] = np.sqrt((w1 * ua[m]) ** 2 + (w2 * ub[m]) ** 2)
                                    elif ua is not None:
                                        out_u[m] = ua[m]
                                    elif ub is not None:
                                        out_u[m] = ub[m]

                elif strat == "spatial_taper":
                    _priority_fill(cfg.priority_order)

                    # Only taper/blend SDB+River (never measured)
                    if river is not None and sdb is not None:
                        m = np.isfinite(river) & np.isfinite(sdb)
                        if meas is not None:
                            m &= ~np.isfinite(meas)
                        if m.any():
                            w1 = float(getattr(cfg, "primary_weight", 0.7) or 0.7)
                            w2 = float(getattr(cfg, "secondary_weight", 0.3) or 0.3)
                            s = w1 + w2
                            if s <= 0:
                                w1, w2, s = 0.7, 0.3, 1.0
                            w1 /= s
                            w2 /= s

                            blend_order = [ss for ss in cfg.priority_order if ss in ("sdb", "river") and src_to_arr.get(ss) is not None]
                            if len(blend_order) >= 2:
                                primary, secondary = blend_order[0], blend_order[1]
                                base_river = w1 if primary == "river" else (w2 if secondary == "river" else 0.5)
                            else:
                                base_river = 0.7

                            river_mask = np.isfinite(river)

                            w_r = None
                            if taper_px > 0:
                                try:
                                    dist_px = distance_transform_edt(river_mask).astype(np.float32)
                                    alpha = np.clip(dist_px / float(max(1, taper_px)), 0.0, 1.0)
                                    w_r = (base_river + alpha * (1.0 - base_river)).astype(np.float32)
                                except Exception:
                                    w_r = None

                            if w_r is None:
                                w_r = np.full(shape, base_river, dtype=np.float32)

                            w_s = (1.0 - w_r).astype(np.float32)

                            out[m] = (w_r[m] * river[m]) + (w_s[m] * sdb[m])
                            prov[m] = PROV_BLENDED

                            if out_u is not None:
                                if river_u is not None and sdb_u is not None:
                                    out_u[m] = np.sqrt((w_r[m] * river_u[m]) ** 2 + (w_s[m] * sdb_u[m]) ** 2)
                                elif river_u is not None:
                                    out_u[m] = river_u[m]
                                elif sdb_u is not None:
                                    out_u[m] = sdb_u[m]

                elif strat == "uncertainty":
                    # Measured always wins; otherwise choose min-unc among river/sdb
                    if meas is not None:
                        mm = np.isfinite(meas)
                        out[mm] = meas[mm]
                        prov[mm] = PROV_MEASURED
                        if out_u is not None and meas_u is not None:
                            out_u[mm] = meas_u[mm]

                    remaining = np.isnan(out)

                    cand = []
                    if sdb is not None and sdb_u is not None:
                        m_s = remaining & np.isfinite(sdb) & np.isfinite(sdb_u)
                        if m_s.any():
                            cand.append(("sdb", sdb, sdb_u, m_s))
                    if river is not None and river_u is not None:
                        m_r = remaining & np.isfinite(river) & np.isfinite(river_u)
                        if m_r.any():
                            cand.append(("river", river, river_u, m_r))

                    if cand:
                        src0, arr0, u0, m0 = cand[0]
                        out[m0] = arr0[m0]
                        prov[m0] = src_to_prov[src0]
                        if out_u is not None:
                            out_u[m0] = u0[m0]

                        for src, arr, ua, m in cand[1:]:
                            if out_u is None:
                                break
                            better = m & np.isfinite(out_u) & (ua < out_u)
                            out[better] = arr[better]
                            out_u[better] = ua[better]
                            prov[better] = src_to_prov[src]

                    remaining = np.isnan(out)
                    if remaining.any():
                        for src in [s for s in cfg.priority_order if s != "measured"]:
                            arr = src_to_arr.get(src)
                            if arr is None:
                                continue
                            m = remaining & np.isfinite(arr)
                            if not m.any():
                                continue
                            out[m] = arr[m]
                            prov[m] = src_to_prov.get(src, 0)
                            if out_u is not None:
                                ua = src_to_unc.get(src)
                                if ua is not None:
                                    out_u[m] = ua[m]
                            remaining = np.isnan(out)
                            if not remaining.any():
                                break

                elif strat == "average":
                    stacks = []
                    u_stacks = []
                    for src in ("sdb", "river", "dem"):
                        arr = src_to_arr.get(src)
                        if arr is not None:
                            stacks.append(arr)
                            if out_u is not None:
                                ua = src_to_unc.get(src)
                                if ua is not None:
                                    u_stacks.append(ua)

                    if stacks:
                        arr3 = np.stack(stacks, axis=0)
                        out = np.nanmean(arr3, axis=0).astype(np.float32)
                        prov[np.isfinite(out)] = PROV_AVERAGED

                    if out_u is not None and u_stacks:
                        u3 = np.stack(u_stacks, axis=0).astype(np.float32)
                        cnt_u = np.sum(np.isfinite(u3), axis=0).astype(np.float32)
                        rss = np.sqrt(np.nansum(u3 ** 2, axis=0)).astype(np.float32)
                        out_u = rss / np.maximum(cnt_u, 1.0)
                        out_u[cnt_u <= 0] = np.nan

                    if dem is not None:
                        m_dem = np.isnan(out) & np.isfinite(dem)
                        out[m_dem] = dem[m_dem]
                        prov[m_dem] = PROV_DEM

                elif strat == "blend":
                    _priority_fill([s for s in cfg.priority_order if s != "measured"])  # measured enforced later

                    m_sdb = np.isfinite(sdb) if sdb is not None else np.zeros(shape, dtype=bool)
                    m_riv = np.isfinite(river) if river is not None else np.zeros(shape, dtype=bool)

                    empty = np.isnan(out)
                    both = empty & m_sdb & m_riv

                    if both.any():
                        w_sdb = None
                        if taper_px > 0:
                            try:
                                dist_px = distance_transform_edt(m_sdb).astype(np.float32)
                                dist_m = dist_px * float(pixel_size_m)
                                w_sdb = np.clip(dist_m / max(taper_m, 1e-6), 0.0, 1.0).astype(np.float32)
                            except Exception:
                                w_sdb = None

                        if w_sdb is None:
                            w_sdb = np.full(shape, 0.5, dtype=np.float32)

                        w_riv = (1.0 - w_sdb).astype(np.float32)

                        out[both] = (w_sdb[both] * sdb[both]) + (w_riv[both] * river[both])
                        prov[both] = PROV_BLENDED

                        if out_u is not None:
                            if sdb_u is not None and river_u is not None:
                                out_u[both] = np.sqrt((w_sdb[both] * sdb_u[both]) ** 2 + (w_riv[both] * river_u[both]) ** 2)
                            elif sdb_u is not None:
                                out_u[both] = sdb_u[both]
                            elif river_u is not None:
                                out_u[both] = river_u[both]

                else:
                    raise ValueError(f"Unknown fusion strategy: {cfg.strategy}")

                # River dominates inside domain mask (seam prevention)
                if bool(getattr(cfg, "river_overrides_sdb_in_domain", False)) and domain_mask is not None and river is not None:
                    take = domain_mask & np.isfinite(river)
                    if meas is not None:
                        take &= ~np.isfinite(meas)
                    if take.any():
                        out[take] = river[take]
                        prov[take] = PROV_RIVER
                        if out_u is not None:
                            if river_u is not None:
                                out_u[take] = river_u[take]
                            else:
                                # Avoid carrying over unrelated uncertainty where depth is river.
                                out_u[take] = nodata

                # Measured always overrides (highest authority)
                if meas is not None:
                    m_meas = np.isfinite(meas)
                    if m_meas.any():
                        out[m_meas] = meas[m_meas]
                        prov[m_meas] = PROV_MEASURED
                        if out_u is not None:
                            if meas_u is not None:
                                out_u[m_meas] = meas_u[m_meas]
                            else:
                                # Do not leave behind unrelated uncertainty values when measured wins.
                                out_u[m_meas] = nodata

                # Stats on inner
                ro = int(win_inner.row_off - win_outer.row_off)
                co = int(win_inner.col_off - win_outer.col_off)
                hh = int(win_inner.height)
                ww = int(win_inner.width)
                prov_inner = prov[ro:ro + hh, co:co + ww]
                cnt["total"] += int((prov_inner > 0).sum())
                cnt["sdb"] += int((prov_inner == PROV_SDB).sum())
                cnt["river"] += int((prov_inner == PROV_RIVER).sum())
                cnt["measured"] += int((prov_inner == PROV_MEASURED).sum())
                cnt["dem"] += int((prov_inner == PROV_DEM).sum())
                cnt["averaged"] += int((prov_inner == PROV_AVERAGED).sum())
                cnt["blended"] += int((prov_inner == PROV_BLENDED).sum())

                _write_inner(win_outer, win_inner, out, prov, out_u)

    finally:
        for ds in list(src_ds.values()):
            try:
                ds.close()
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        for ds in list(unc_ds.values()):
            try:
                ds.close()
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        if domain_ds is not None:
            try:
                domain_ds.close()
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    if has_unc_inputs and wrote_unc_any and uncertainty_path.exists():
        result.uncertainty_raster = uncertainty_path
    else:
        try:
            if uncertainty_path.exists():
                uncertainty_path.unlink()
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        result.uncertainty_raster = None

    result.stats["pixel_counts"] = cnt
    result.stats["pixel_size_m"] = float(pixel_size_m)
    result.stats["taper_m"] = float(taper_m)
    result.stats["tile_size"] = int(tile_size)
    result.stats["overlap_px"] = int(overlap_px)

    result.status = "success"
    result.error = None
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
    p.add_argument("--strategy", choices=["priority", "uncertainty", "average", "blend", "weighted_overlap", "spatial_taper"],
                   default="priority", help="Fusion strategy")
    p.add_argument("--priority", default="measured,sdb,river,dem",
                   help="Priority order (comma-separated, highest first)")
    
    # Blending
    p.add_argument("--taper", type=float, default=50.0, help="Taper distance for blending (m)")
    p.add_argument("--primary-weight", type=float, default=0.7, help="Primary weight (weighted_overlap/spatial_taper)")
    p.add_argument("--secondary-weight", type=float, default=0.3, help="Secondary weight (weighted_overlap/spatial_taper)")
    p.add_argument("--river-domain-mask", default=None, help="Optional river domain mask raster (1=river)")
    p.add_argument("--river-overrides-sdb-in-domain", action="store_true", help="Inside river domain mask, river overrides SDB wherever river is finite")
    
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
        primary_weight=float(getattr(args, "primary_weight", 0.7)),
        secondary_weight=float(getattr(args, "secondary_weight", 0.3)),
        river_domain_mask=Path(getattr(args, "river_domain_mask")) if getattr(args, "river_domain_mask", None) else None,
        river_overrides_sdb_in_domain=bool(getattr(args, "river_overrides_sdb_in_domain", False)),
        template_raster=Path(args.template) if args.template else None,
    )
    
    result = fuse_bathymetry(cfg)
    
    if result.status == "success":
        log.info(f"Combined bathymetry: {result.combined_raster}")
        sys.exit(0)
    else:
        log.info(f"Fusion failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
