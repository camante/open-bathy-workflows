#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bathy_main.py – Unified Coastal + River Bathymetry Pipeline

Orchestrates:
- SDB pipeline (sdb_main.py)
- River cross-section interpolation (river_network.py -> xs_builder.py -> xs_infer_bathy_raster.py)

Key behavior:
- Streams subprocess stdout live (SDB logs are on stdout)
- Captures stderr tails for debugging
- Robustly discovers SDB depth raster output
- Uses correct river_network.py CLI flags (--out-gpkg)
"""


# IMPORTANT: Configure logging FIRST, before importing other pipeline modules
import os as _os
# Force a headless-safe Matplotlib backend early (prevents TkAgg/Tkinter crashes under multiprocessing)
_os.environ.setdefault('MPLBACKEND', 'Agg')

import logging
import sys

# Set up centralized logging before any other imports
try:
    from logging_config import setup_logging
    setup_logging()
except ImportError:
    # Fallback if logging_config.py is not present
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

import argparse
import hashlib
import math
import json
import os
import re
import shutil
from vdatum_utils import convert_sdb_msl_to_navd88
from process_utils import run_cmd
import subprocess
import shlex
import threading

# Central constants (versioning, nodata)
import constants
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _clip_raster_to_mask(
    raster_path: Path,
    mask_path: Path,
    *,
    inside_value: int = 1,
    invert: bool = False,
    nodata: float = -9999.0,
) -> bool:
    """Hard-clip a float raster so values outside a mask become nodata.

    This is an operational safety net to guarantee final river outputs are only inside
    the river domain. It does not change values inside the domain.

    Mask raster is expected to be on the same grid as raster_path.
    """
    try:
        import rasterio
        import numpy as np
    except Exception:
        return False

    if not Path(raster_path).exists() or not Path(mask_path).exists():
        return False

    tmp = Path(str(raster_path) + ".tmpclip.tif")
    with rasterio.open(raster_path) as src, rasterio.open(mask_path) as msrc:
        if (src.width != msrc.width) or (src.height != msrc.height) or (src.transform != msrc.transform):
            # If grids don't match, do nothing (caller should align first).
            return False

        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=float(nodata), compress="deflate")
        data = src.read(1).astype("float32")
        m = msrc.read(1)

        inside = (m == int(inside_value))
        if invert:
            inside = ~inside

        out = np.where(inside & np.isfinite(data), data, float(nodata)).astype("float32")

        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(out, 1)

    tmp.replace(raster_path)
    return True


def _clip_raster_to_mask_reproject(
    raster_path: Path,
    mask_path: Path,
    *,
    inside_value: int = 1,
    invert: bool = False,
    nodata: float = -9999.0,
) -> bool:
    """Hard-clip raster_path to mask_path, reprojecting mask to raster grid if needed.

    This is used as a final safety net when the mask (e.g., waffles coastline)
    is not on the same grid as the output raster.
    """
    try:
        import rasterio
        import numpy as np
        from rasterio.warp import reproject, Resampling

        nodata = -9999.0
    except Exception:
        return False

    raster_path = Path(raster_path)
    mask_path = Path(mask_path)
    if (not raster_path.exists()) or (not mask_path.exists()):
        return False

    tmp = Path(str(raster_path) + ".tmpclip.tif")
    with rasterio.open(raster_path) as src, rasterio.open(mask_path) as msrc:
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=float(nodata), compress="deflate")
        data = src.read(1).astype("float32")

        # Reproject mask to raster grid
                # Reproject mask to raster grid.
        # IMPORTANT: treat mask nodata / areas outside coverage as OUTSIDE the mask.
        fill_val = np.uint8(255)
        m_aligned = np.full((src.height, src.width), fill_val, dtype=np.uint8)
        src_nodata = msrc.nodata
        reproject(
            source=msrc.read(1),
            destination=m_aligned,
            src_transform=msrc.transform,
            src_crs=msrc.crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            resampling=Resampling.nearest,
        )

        valid = (m_aligned != fill_val)
        inside = valid & (m_aligned == int(inside_value))
        if invert:
            inside = ~inside

        out = np.where(inside & np.isfinite(data), data, float(nodata)).astype("float32")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(out, 1)

    tmp.replace(raster_path)
    return True




def _parse_aoi_bbox(aoi: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """Parse AOI bbox from 'W/E/S/N' string."""
    if not aoi:
        return None
    s = str(aoi).strip()
    # allow commas or slashes
    parts = re.split(r"[,/\s]+", s)
    parts = [p for p in parts if p]
    if len(parts) != 4:
        return None
    try:
        w, e, s_, n = [float(p) for p in parts]
    except Exception:
        return None
    # basic sanity: W < E, S < N
    if not (w < e and s_ < n):
        return None
    return (w, s_, e, n)

def _bbox_to_aoi_str(bbox: Tuple[float, float, float, float]) -> str:
    w, s, e, n = bbox
    return f"{w}/{e}/{s}/{n}"


def _aoi_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    w, s, e, n = bbox
    return ((w + e) / 2.0, (s + n) / 2.0)  # (lon, lat)


def _expand_bbox_km(bbox: Tuple[float, float, float, float], buffer_km: float) -> Tuple[float, float, float, float]:
    """Expand lon/lat bbox by an approximate kilometer buffer.

    Uses 1 deg lat ~ 111.32 km and 1 deg lon scaled by cos(lat_center).
    """
    w, s, e, n = bbox
    if buffer_km <= 0:
        return bbox
    lon_c, lat_c = _aoi_centroid(bbox)
    km_per_deg_lat = 111.32
    km_per_deg_lon = max(1e-6, km_per_deg_lat * math.cos(math.radians(lat_c)))
    dlat = buffer_km / km_per_deg_lat
    dlon = buffer_km / km_per_deg_lon
    return (w - dlon, s - dlat, e + dlon, n + dlat)


def _gaussian_smooth_masked(data: "np.ndarray", valid: "np.ndarray", sigma_px: float) -> "np.ndarray":
    """Gaussian smooth with nodata masking using the standard weighted approach."""
    from scipy.ndimage import gaussian_filter
    import numpy as np

    sigma_px = float(max(0.0, sigma_px))
    if sigma_px == 0.0:
        return data.astype("float32", copy=False)

    w = valid.astype("float32")
    data0 = np.where(valid, data.astype("float32"), 0.0).astype("float32")
    num = gaussian_filter(data0, sigma=sigma_px, mode="nearest")
    den = gaussian_filter(w, sigma=sigma_px, mode="nearest")
    out = np.where(den > 1e-6, num / den, np.nan).astype("float32")
    return out


def _apply_tile_edge_taper_epsg4269(
    raster_path: Path,
    tile_bbox: Tuple[float, float, float, float],
    *,
    taper_km: float,
    smooth_sigma_km: float,
    nodata: float = -9999.0,
) -> Optional[Path]:
    """Blend raster toward a low-frequency smooth field near tile edges.

    This is a *seam stability* measure: in the outer taper band, we downweight
    high-frequency texture (often SDB-driven) which is sensitive to per-tile training noise.
    Adjacent tiles applying the same edge policy converge toward similar low-frequency values.

    Requires raster in EPSG:4269 (lon/lat).
    Returns the tapered raster path (new file), or None if no-op/failure.
    """
    try:
        import rasterio
        import numpy as np
    except Exception:
        return None

    raster_path = Path(raster_path)
    if (not raster_path.exists()) or taper_km <= 0:
        return None

    w, s, e, n = tile_bbox
    out_path = raster_path.with_name(raster_path.stem + "_tapered" + raster_path.suffix)

    with rasterio.open(raster_path) as src:
        if src.count != 1:
            return None
        crs = str(src.crs) if src.crs else ""
        if "4269" not in crs and "4326" not in crs:
            # We only support geographic lon/lat here; caller should warp first.
            return None
        data = src.read(1).astype("float32")
        prof = src.profile.copy()
        t = src.transform

    import numpy as np
    valid = np.isfinite(data) & (data != float(nodata))

    # Build lon/lat coordinate arrays
    ny, nx = data.shape
    # pixel centers
    xs = (np.arange(nx) + 0.5) * t.a + t.c
    ys = (np.arange(ny) + 0.5) * t.e + t.f  # t.e negative
    lon = xs[None, :]
    lat = ys[:, None]

    # Distance to nearest tile edge (km) using local scale
    km_per_deg_lat = 111.32
    km_per_deg_lon = km_per_deg_lat * np.cos(np.deg2rad(lat))
    km_per_deg_lon = np.maximum(km_per_deg_lon, 1e-6)

    dx_km = np.minimum(np.abs(lon - w) * km_per_deg_lon, np.abs(lon - e) * km_per_deg_lon)
    dy_km = np.minimum(np.abs(lat - s) * km_per_deg_lat, np.abs(lat - n) * km_per_deg_lat)
    dist_km = np.minimum(dx_km, dy_km).astype("float32")

    # Compute smooth field sigma in pixels ~ sigma_km / pixel_km
    # Approximate pixel size (km) from transform
    px_km_y = abs(t.e) * km_per_deg_lat
    px_km_x = abs(t.a) * (km_per_deg_lat * math.cos(math.radians((s + n) / 2.0)))
    px_km = float(max(1e-6, (px_km_x + px_km_y) / 2.0))
    sigma_px = float(max(0.0, smooth_sigma_km / px_km))
    smooth = _gaussian_smooth_masked(data, valid, sigma_px=sigma_px)

    # Taper weights: 0 at edge, 1 at/inside taper_km
    wgt = np.clip(dist_km / float(taper_km), 0.0, 1.0).astype("float32")
    out = np.where(valid, (wgt * data + (1.0 - wgt) * smooth).astype("float32"), float(nodata)).astype("float32")

    prof.update(dtype="float32", count=1, nodata=float(nodata), compress="deflate")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)

    return out_path


def _compute_edge_band_metrics_epsg4269(
    raster_path: Path,
    tile_bbox: Tuple[float, float, float, float],
    *,
    band_km: float,
    smooth_sigma_km: float,
    nodata: float = -9999.0,
) -> Dict[str, Any]:
    """Compute seam-stability diagnostics in an edge band (EPSG:4269 rasters only)."""
    out: Dict[str, Any] = {}
    try:
        import rasterio
        import numpy as np
    except Exception:
        return out

    raster_path = Path(raster_path)
    if (not raster_path.exists()) or band_km <= 0:
        return out

    with rasterio.open(raster_path) as src:
        if src.count != 1:
            return out
        crs = str(src.crs) if src.crs else ""
        if "4269" not in crs and "4326" not in crs:
            return out
        data = src.read(1).astype("float32")
        t = src.transform

    import numpy as np
    valid = np.isfinite(data) & (data != float(nodata))
    if valid.sum() < 10:
        return out

    w, s, e, n = tile_bbox
    ny, nx = data.shape
    xs = (np.arange(nx) + 0.5) * t.a + t.c
    ys = (np.arange(ny) + 0.5) * t.e + t.f
    lon = xs[None, :]
    lat = ys[:, None]

    km_per_deg_lat = 111.32
    km_per_deg_lon = km_per_deg_lat * np.cos(np.deg2rad(lat))
    km_per_deg_lon = np.maximum(km_per_deg_lon, 1e-6)

    dx_km = np.minimum(np.abs(lon - w) * km_per_deg_lon, np.abs(lon - e) * km_per_deg_lon)
    dy_km = np.minimum(np.abs(lat - s) * km_per_deg_lat, np.abs(lat - n) * km_per_deg_lat)
    dist_km = np.minimum(dx_km, dy_km).astype("float32")

    edge = (dist_km <= float(band_km)) & valid
    n_edge = int(edge.sum())
    out["edge_band_km"] = float(band_km)
    out["edge_pixels"] = n_edge
    if n_edge < 25:
        return out

    # Low-frequency comparison
    px_km_y = abs(t.e) * km_per_deg_lat
    px_km_x = abs(t.a) * (km_per_deg_lat * math.cos(math.radians((s + n) / 2.0)))
    px_km = float(max(1e-6, (px_km_x + px_km_y) / 2.0))
    sigma_px = float(max(0.0, smooth_sigma_km / px_km))
    smooth = _gaussian_smooth_masked(data, valid, sigma_px=sigma_px)

    diff = (data - smooth).astype("float32")
    d = diff[edge]
    out["edge_diff_to_smooth_mae_m"] = float(np.nanmean(np.abs(d)))
    out["edge_diff_to_smooth_rmse_m"] = float(np.sqrt(np.nanmean(d * d)))
    out["edge_diff_to_smooth_p95_m"] = float(np.nanpercentile(np.abs(d), 95))

    # Gradient magnitude RMS in edge band (proxy for seam instability / texture)
    gy, gx = np.gradient(np.where(valid, data, np.nan))
    gmag = np.sqrt(gx * gx + gy * gy).astype("float32")
    out["edge_grad_rms"] = float(np.nanmean(gmag[edge] * gmag[edge]) ** 0.5)

    return out


def _clip_raster_to_bbox(
    raster_path: Path,
    bbox: Tuple[float, float, float, float],
    *,
    nodata: float = -9999.0,
) -> bool:
    """Hard-crop raster to bbox (in the raster's CRS) by setting outside pixels to nodata.

    This is a conservative safety-net for cases where upstream masks do not cover the
    entire processing grid. It does **not** resample; it only writes nodata outside bbox.
    """
    try:
        import rasterio
        import numpy as np
        from rasterio.windows import from_bounds
    except Exception:
        return False

    raster_path = Path(raster_path)
    if not raster_path.exists():
        return False

    left, bottom, right, top = bbox
    tmp = Path(str(raster_path) + ".tmpbbox.tif")
    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=float(nodata), compress="deflate")
        data = src.read(1).astype("float32")

        # window in raster grid
        try:
            win = from_bounds(left, bottom, right, top, transform=src.transform)
            # clamp window
            row0 = max(0, int(np.floor(win.row_off)))
            col0 = max(0, int(np.floor(win.col_off)))
            row1 = min(src.height, int(np.ceil(win.row_off + win.height)))
            col1 = min(src.width, int(np.ceil(win.col_off + win.width)))
        except Exception:
            return False

        mask = np.zeros((src.height, src.width), dtype=bool)
        if row1 > row0 and col1 > col0:
            mask[row0:row1, col0:col1] = True

        out = np.where(mask & np.isfinite(data), data, float(nodata)).astype("float32")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(out, 1)

    tmp.replace(raster_path)
    return True


def _crop_raster_extent_to_bbox(
    raster_path: Path,
    bbox: Tuple[float, float, float, float],
    *,
    nodata: float = -9999.0,
) -> bool:
    """Crop raster *extent* to bbox by writing a new windowed GeoTIFF.

    Unlike _clip_raster_to_bbox, this reduces the raster bounds to the AOI.
    Only intended for final EPSG:4269 deliverables.
    """
    try:
        import rasterio
        import numpy as np
        from rasterio.windows import from_bounds
    except Exception:
        return False

    raster_path = Path(raster_path)
    if not raster_path.exists():
        return False

    left, bottom, right, top = bbox
    tmp = Path(str(raster_path) + ".tmpaoi.tif")
    with rasterio.open(raster_path) as src:
        # Compute window in raster grid
        win = from_bounds(left, bottom, right, top, transform=src.transform)
        win = win.round_offsets().round_lengths()

        if win.width <= 0 or win.height <= 0:
            return False

        data = src.read(1, window=win).astype("float32")
        prof = src.profile.copy()
        prof.update(
            dtype="float32",
            count=1,
            nodata=float(nodata),
            compress="deflate",
            height=int(win.height),
            width=int(win.width),
            transform=rasterio.windows.transform(win, src.transform),
        )
        # Ensure outside is nodata if any NaNs
        data = np.where(np.isfinite(data), data, float(nodata)).astype("float32")

        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(data, 1)

    tmp.replace(raster_path)
    return True



def _discover_waffles_coastline_ocean_only(out_dir: Path, cache_root: Path) -> Optional[Path]:
    """Best-effort discovery of waffles coastline (ocean-only) mask (water=0, land=1)."""
    pats = [
        "waffles_coastline_ocean_only*.tif",
        "waffles*ocean_only*.tif",
        "*coastline*ocean*only*.tif",
    ]
    bases: List[Path] = []
    for base in [out_dir, cache_root]:
        if base and Path(base).exists():
            base = Path(base)
            for sub in ["masks", "mask", "waffles", "coastline"]:
                cand = base / sub
                if cand.exists():
                    bases.append(cand)
            bases.append(base)

    for base in bases:
        for pat in pats:
            hits = list(Path(base).rglob(pat))
            if hits:
                hits = sorted(hits, key=lambda p: (len(str(p)), str(p)))
                return hits[0]
    return None



def _apply_final_domain_policy(cfg: "BathyConfig", out_dir: Path, cache_root: Path) -> None:
    """Apply final domain clipping rules requested by user:

    - If SDB mode is on (and river is off): clip final combined outputs to waffles coastline (ocean-only) mask.
    - If river mode is on (and SDB is off): clip river outputs to NHDArea-derived channel mask.
    - If both SDB and river are on: clip combined + river outputs to waffles coastline with NHD mask.

    Mask conventions:
      - Waffles masks: water=0, land=1
      - River channel mask: inside=1, outside=0
    """
    methods = [m.strip().lower() for m in (getattr(cfg, "methods", []) or [])]
    sdb_on = "sdb" in methods
    river_on = "river" in methods

    combined_depth = out_dir / "combined" / "bathy_depth_final_epsg4269.tif"
    combined_bottom = out_dir / "combined" / "bathy_bottom_navd88_final_epsg4269.tif"
    river_depth = out_dir / "river" / "river_depth_final_epsg4269.tif"
    river_bottom = out_dir / "river" / "river_bottom_navd88_final_epsg4269.tif"
    root_copy = out_dir / "bathy_depth_final_epsg4269.tif"

    def _clip(path: Path, mask: Path, inside_value: int) -> None:
        if path.exists() and mask.exists():
            _clip_raster_to_mask_reproject(path, mask, inside_value=inside_value, invert=False, nodata=float(getattr(cfg, "final_nodata", -9999.0)))

    if sdb_on and river_on:
        wm = _discover_waffles_coastline_with_nhd(out_dir, cache_root)
        if wm and wm.exists():
            for p in [combined_depth, combined_bottom, river_depth, river_bottom, root_copy]:
                _clip(p, wm, inside_value=0)
    elif sdb_on and (not river_on):
        wm = _discover_waffles_coastline_ocean_only(out_dir, cache_root) or _discover_waffles_coastline_with_nhd(out_dir, cache_root)
        if wm and wm.exists():
            for p in [combined_depth, combined_bottom, root_copy]:
                _clip(p, wm, inside_value=0)
    elif river_on and (not sdb_on):
        rm = _discover_river_channel_mask(out_dir, cache_root)
        if rm and rm.exists():
            for p in [river_depth, river_bottom, combined_bottom]:
                _clip(p, rm, inside_value=1)


def _discover_waffles_coastline_with_nhd(out_dir: Path, cache_root: Path) -> Optional[Path]:
    """Best-effort discovery of waffles coastline+NHD mask (water=0, land=1).

    We deliberately search common subdirectories (e.g., *masks/*) first because the
    pipeline often stores waffles products there.
    """
    pats = [
        "waffles_coastline_with_nhd*.tif",
        "waffles*coastline*nhd*.tif",
        "*coastline*with*nhd*.tif",
        "*waffles*with*nhd*.tif",
    ]
    bases: List[Path] = []
    for base in [out_dir, cache_root]:
        if base and Path(base).exists():
            base = Path(base)
            # prefer explicit masks/ directories
            for sub in ["masks", "mask", "waffles", "coastline"]:
                cand = base / sub
                if cand.exists():
                    bases.append(cand)
            bases.append(base)

    for base in bases:
        for pat in pats:
            hits = list(Path(base).rglob(pat))
            if hits:
                # prefer shorter path (less likely to be an old copy elsewhere)
                hits = sorted(hits, key=lambda p: (len(str(p)), str(p)))
                return hits[0]
    return None




def _discover_river_channel_mask(out_dir: Path, cache_root: Path) -> Optional[Path]:
    """Find river channel mask (NHDArea-derived) produced by river_domain_mask.py (inside=1)."""
    pats = [
        "river_channel_mask.tif",
        "*river_channel_mask*.tif",
        "*channel_mask*.tif",
    ]
    bases: List[Path] = []
    for base in [out_dir / "river", out_dir, cache_root]:
        if base and Path(base).exists():
            bases.append(Path(base))
    for base in bases:
        for pat in pats:
            hits = list(base.rglob(pat))
            if hits:
                hits = sorted(hits, key=lambda p: (len(str(p)), str(p)))
                return hits[0]
    return None


def _ensure_output_contract(out_dir: Path, cache_root: Path) -> None:
    """Ensure expected output filenames exist for downstream debug/metrics.

    This is a pure bookkeeping step: it creates symlinks (or copies as fallback)
    to standard names when alternative names exist.
    """
    river_dir = out_dir / "river"
    combined_dir = out_dir / "combined"
    river_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    def _link(target: Path, source: Path) -> None:
        try:
            if target.exists():
                return
            target.symlink_to(source.resolve())
        except Exception:
            try:
                shutil.copy2(source, target)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    # Standard river depth name
    std_depth = river_dir / "river_depth_final_epsg4269.tif"
    if not std_depth.exists():
        for cand in [
            river_dir / "river_depth_terrain_final_epsg4269.tif",
            river_dir / "river_depth_terrain_patch.tif",
            river_dir / "river_depth_patch.tif",
            river_dir / "river_depth.tif",
        ]:
            if cand.exists():
                _link(std_depth, cand)
                break

    # Standard river bottom name
    std_bottom = river_dir / "river_bottom_navd88_final_epsg4269.tif"
    if not std_bottom.exists():
        for cand in [
            river_dir / "river_bottom_navd88_patch_epsg4269.tif",
            river_dir / "river_bottom_navd88.tif",
            river_dir / "river_bottom.tif",
        ]:
            if cand.exists():
                _link(std_bottom, cand)
                break

    # Combined NAVD88 bottom output should exist if river bottom exists
    comb_bottom = combined_dir / "bathy_bottom_navd88_final_epsg4269.tif"
    if (not comb_bottom.exists()) and std_bottom.exists():
        _link(comb_bottom, std_bottom)

    
    # Final domain clipping is handled by _apply_final_domain_policy(cfg, out_dir, cache_root)
    # after the pipeline completes, so we do not apply unconditional clipping here.
# Import new improvement modules
try:
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    
try:
    CHECKPOINTS_AVAILABLE = True
except ImportError:
    CHECKPOINTS_AVAILABLE = False

try:
    PARALLEL_PREDICT_AVAILABLE = True
except ImportError:
    PARALLEL_PREDICT_AVAILABLE = False


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

log = logging.getLogger("bathy_main")

def _fingerprint_path(p: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Cheap fingerprint for cache invalidation (no content hashing)."""
    if p is None:
        return None
    try:
        st = p.stat()
        return {"path": str(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except FileNotFoundError:
        return {"path": str(p), "missing": True}


def _fingerprint_script(name: str) -> Dict[str, Any]:
    here = Path(__file__).resolve().parent
    fp = _fingerprint_path(here / name)
    return fp or {"path": str(here / name), "missing": True}


def _river_cache_key_and_manifest(cfg: "BathyConfig") -> tuple[str, Dict[str, Any]]:
    """Stable cache key for river interpolation products.

    NOTE: River products are highly sensitive to priors and domain masks. Keep this manifest
    reasonably complete so cached outputs are not silently reused under different settings.
    """
    soundings: List[Path] = []
    if cfg.river_soundings:
        for part in str(cfg.river_soundings).split(","):
            part = part.strip()
            if part:
                soundings.append(Path(part))

    manifest = {
        "aoi": cfg.aoi,
        "river_method": getattr(cfg, "river_method", "xs"),
        "river_dem": _fingerprint_path(cfg.river_dem),
        "soundings": [_fingerprint_path(p) for p in soundings],
        "river_soundings_mode": getattr(cfg, "river_soundings_mode", "auto"),
        "river_soundings_max_dist_m": float(getattr(cfg, "river_soundings_max_dist_m", 1500.0)),
        "river_soundings_min_r": float(getattr(cfg, "river_soundings_min_r", 0.25)),
        "river_soundings_enforce": bool(getattr(cfg, "river_soundings_enforce", True)),
        "snap_m": cfg.snap_m,

        "river_authoritative_bed": _fingerprint_path(getattr(cfg, "river_authoritative_bed", None)),
        "river_authoritative_bed_max_dist_m": float(getattr(cfg, "river_authoritative_bed_max_dist_m", 2000.0)),
        "river_residual_blend_sigma_m": float(getattr(cfg, "river_residual_blend_sigma_m", 120.0)),
        "river_nodata": float(getattr(cfg, "river_nodata", -9999.0)),
        "mask_river_to_waffles": bool(getattr(cfg, "mask_river_to_waffles", True)),
        "river_use_nhdarea": bool(getattr(cfg, "river_use_nhdarea", True)),
        "river_nhdarea_layer": getattr(cfg, "river_nhdarea_layer", "nhdarea_clip"),

        # Legacy XS parameters (only used when river_method == "xs")
        "xs_spacing_m": cfg.xs_spacing_m,
        "xs_length_m": cfg.xs_length_m,
        "river_continuous": cfg.river_continuous,
        "river_continuous_buffer_m": cfg.river_continuous_buffer_m,
        "river_continuous_k": cfg.river_continuous_k,
        "river_idw_power": cfg.river_idw_power,
        "river_aniso_along_scale_m": cfg.river_aniso_along_scale_m,
        "river_aniso_cross_scale_m": cfg.river_aniso_cross_scale_m,
        "river_thalweg_weight": cfg.river_thalweg_weight,
        "river_thalweg_only": getattr(cfg, "river_thalweg_only", False),
        "river_thalweg_densify_factor": getattr(cfg, "river_thalweg_densify_factor", 0.5),
        "river_thalweg_densify_step_m": getattr(cfg, "river_thalweg_densify_step_m", None),
        "river_overlap_reducer": cfg.river_overlap_reducer,

        # XS generation controls (geometry stability / artifact reduction)
        "xs_smoothing_window_m": getattr(cfg, "xs_smoothing_window_m", 0.0),
        "xs_trim_overlaps": bool(getattr(cfg, "xs_trim_overlaps", True)),
        "xs_global_deconflict": bool(getattr(cfg, "xs_global_deconflict", True)),
        "xs_deconflict_tol_m": float(getattr(cfg, "xs_deconflict_tol_m", 2.0)),
        "xs_skip_junctions": bool(getattr(cfg, "xs_skip_junctions", True)),
        "xs_junction_snap_m": float(getattr(cfg, "xs_junction_snap_m", 30.0)),
        "xs_junction_buffer_m": float(getattr(cfg, "xs_junction_buffer_m", 120.0)),
        "xs_densify_step_m": float(getattr(cfg, "xs_densify_step_m", 20.0)),

        # Skeleton parameters (only used when river_method == "skeleton")
        "river_channel_buffer_m": getattr(cfg, "river_channel_buffer_m", 400.0),
        "river_max_channel_width_m": getattr(cfg, "river_max_channel_width_m", 600.0),
        "river_mainstem_min_order": getattr(cfg, "river_mainstem_min_order", 5),
        "river_max_mainstem_width_m": getattr(cfg, "river_max_mainstem_width_m", 2500.0),
        "river_shape_exp": getattr(cfg, "river_shape_exp", 0.5),
        "river_dmax_min_m": getattr(cfg, "river_dmax_min_m", 0.5),
        "river_dmax_max_m": getattr(cfg, "river_dmax_max_m", 30.0),

        # Priors (shared)
        "river_prior_mode": cfg.river_prior_mode,
        "river_mv_a0": cfg.river_mv_a0,
        "river_mv_bw": cfg.river_mv_bw,
        "river_mv_ba": cfg.river_mv_ba,
        "river_mv_bs": cfg.river_mv_bs,
        "river_mv_eps_a": cfg.river_mv_eps_a,
        "river_mv_eps_s": cfg.river_mv_eps_s,

        # USGS anchors / gage-derived priors
        "river_usgs_sites": getattr(cfg, "river_usgs_sites", None),
        "river_usgs_start": getattr(cfg, "river_usgs_start", None),
        "river_usgs_end": getattr(cfg, "river_usgs_end", None),
        "river_usgs_cache_dir": _fingerprint_path(getattr(cfg, "river_usgs_cache_dir", None)),
        "river_usgs_max_dist_m": float(getattr(cfg, "river_usgs_max_dist_m", 5000.0)),
        "river_usgs_mean_to_dmax": getattr(cfg, "river_usgs_mean_to_dmax", "auto"),
        "river_usgs_a_stat": getattr(cfg, "river_usgs_a_stat", "median"),
        "river_usgs_q_quantile_lo": float(getattr(cfg, "river_usgs_q_quantile_lo", 0.20)),
        "river_usgs_q_quantile_hi": float(getattr(cfg, "river_usgs_q_quantile_hi", 0.80)),
        "river_usgs_a_cv_warn": float(getattr(cfg, "river_usgs_a_cv_warn", 0.50)),
        "river_usgs_width_ratio_max": float(getattr(cfg, "river_usgs_width_ratio_max", 3.0)),
        "river_usgs_width_ratio_blend": bool(getattr(cfg, "river_usgs_width_ratio_blend", True)),
        "river_gage_snap_max_dist_m": float(getattr(cfg, "river_gage_snap_max_dist_m", 1000.0)),

        # Width-stage CSV anchors
        "river_width_stage_csv": getattr(cfg, "river_width_stage_csv", None),
        "river_width_stage_max_dist_m": float(getattr(cfg, "river_width_stage_max_dist_m", 5000.0)),
        "river_width_stage_min_n": int(getattr(cfg, "river_width_stage_min_n", 6)),
        "river_width_stage_min_r2": float(getattr(cfg, "river_width_stage_min_r2", 0.25)),
        "river_width_stage_max_weight": float(getattr(cfg, "river_width_stage_max_weight", 0.8)),

        # Slope/WSE profile controls
        "river_slope_proxy_window": int(getattr(cfg, "river_slope_proxy_window", 9)),
        "river_slope_min": float(getattr(cfg, "river_slope_min", 1e-5)),
        "river_slope_max": float(getattr(cfg, "river_slope_max", 0.05)),
        "river_slope_proxy_min_n": int(getattr(cfg, "river_slope_proxy_min_n", 7)),
        "river_wse_profile_enabled": bool(getattr(cfg, "river_wse_profile_enabled", True)),
        "river_wse_profile_window": int(getattr(cfg, "river_wse_profile_window", 9)),
        "river_wse_profile_min_n": int(getattr(cfg, "river_wse_profile_min_n", 7)),
        "river_wse_profile_monotonic": bool(getattr(cfg, "river_wse_profile_monotonic", True)),

        # Skeleton scientific controls
        "river_skeleton_wse_mode": str(getattr(cfg, "river_skeleton_wse_mode", "bank")),
        "river_skeleton_wse_smooth_sigma_m": float(getattr(cfg, "river_skeleton_wse_smooth_sigma_m", 0.0)),
        "river_skeleton_wse_profile_step_m": float(getattr(cfg, "river_skeleton_wse_profile_step_m", 20.0)),
        "river_skeleton_wse_profile_resample_m": float(getattr(cfg, "river_skeleton_wse_profile_resample_m", 20.0)),
        "river_skeleton_wse_profile_smooth_sigma_m": float(getattr(cfg, "river_skeleton_wse_profile_smooth_sigma_m", 200.0)),
        "river_skeleton_wse_profile_max_slope": float(getattr(cfg, "river_skeleton_wse_profile_max_slope", 0.005)),
        "river_skeleton_wse_profile_min_samples": int(getattr(cfg, "river_skeleton_wse_profile_min_samples", 10)),
        "river_skeleton_wse_profile_max_query_dist_m": float(getattr(cfg, "river_skeleton_wse_profile_max_query_dist_m", 250.0)),
        "river_skeleton_junction_mode": str(getattr(cfg, "river_skeleton_junction_mode", "smooth")),
        "river_skeleton_junction_buffer_m": float(getattr(cfg, "river_skeleton_junction_buffer_m", 120.0)),
        "river_skeleton_junction_degree_min": int(getattr(cfg, "river_skeleton_junction_degree_min", 3)),
        "river_skeleton_junction_smooth_sigma_m": float(getattr(cfg, "river_skeleton_junction_smooth_sigma_m", 80.0)),
        "river_skeleton_junction_max_width_m": float(getattr(cfg, "river_skeleton_junction_max_width_m", 300.0)),
        "river_skeleton_asymmetry_mode": str(getattr(cfg, "river_skeleton_asymmetry_mode", "none")),
        "river_skeleton_asymmetry_strength": float(getattr(cfg, "river_skeleton_asymmetry_strength", 0.25)),
        "river_skeleton_asymmetry_curv_ref": float(getattr(cfg, "river_skeleton_asymmetry_curv_ref", 0.002)),
        "river_skeleton_asymmetry_max_shift": float(getattr(cfg, "river_skeleton_asymmetry_max_shift", 0.20)),
        "river_skeleton_asymmetry_min_width_m": float(getattr(cfg, "river_skeleton_asymmetry_min_width_m", 10.0)),
        "river_skeleton_asymmetry_min_curv": float(getattr(cfg, "river_skeleton_asymmetry_min_curv", 0.0005)),
        "river_skeleton_asymmetry_densify_step_m": float(getattr(cfg, "river_skeleton_asymmetry_densify_step_m", 20.0)),

        "tnm_enable": bool(cfg.tnm_enable),
        "tnm_dataset": cfg.tnm_dataset,
        "scripts": {
            "river_network.py": _fingerprint_script("river_network.py"),
            "xs_builder.py": _fingerprint_script("xs_builder.py"),
            "xs_infer_bathy_raster.py": _fingerprint_script("xs_infer_bathy_raster.py"),
            "river_domain_mask.py": _fingerprint_script("river_domain_mask.py"),
            "river_skeleton_bathy.py": _fingerprint_script("river_skeleton_bathy.py"),
        },
            "version": "river_cache_v8",
    }
    s = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return key, manifest


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class BathyConfig:
    aoi: str
    start_date: str
    end_date: str
    # Operational tiling policy:
    # - Run processing on an expanded AOI to reduce edge effects.
    # - Clip final outputs back to the tile AOI.
    aoi_tile: Optional[str] = None  # original tile AOI string (W/E/S/N)
    tile_bbox: Optional[Tuple[float, float, float, float]] = None  # (W,S,E,N) in EPSG:4269
    tile_buffer_km: float = 15.0
    tile_edge_taper_enabled: bool = True
    tile_edge_taper_km: float = 2.0
    tile_edge_smooth_sigma_km: float = 10.0
    tile_edge_metrics_band_km: float = 2.0

    # SDB cross-tile consistency policy (DEFAULT):
    # Use a bounded "model bank" (reservoir sample) across AOIs and periodically retrain
    # the RF only when enough new samples have accumulated.
    # This drives cross-tile consistency without storing unbounded training data.
    sdb_model_bank_enabled: bool = True
    sdb_model_bank: str = "auto"
    sdb_bank_max_samples: int = 100000
    sdb_bank_seed: int = 1337
    sdb_bank_retrain_min_new: int = 2000

    # Deprecated regional model cache (kept for backwards compatibility)
    sdb_model_cache_enabled: bool = False
    sdb_model_cache_key: str = "auto"


    out_dir: Path = Path("output/unified")
    methods: List[str] = field(default_factory=lambda: ["sdb", "river"])
    priority: str = "sdb"  # "sdb" or "river"

    # SDB args passed through
    cloud: int = 70
    icesat: str = "all_atl"
    sdb_mode: str = "all_sdb"
    cache_root: Path = Path("cache")
    align_mode: str = "median"

    # Sun-glint correction (Hedley-style) applied to Sentinel-2 composites (SDB only)
    glint_correct: bool = False
    glint_nir_band: str = "B08"
    glint_vis_bands: str = "B02,B03,B04"
    glint_nir_min_percentile: float = 1.0
    glint_deepwater_b02_max: float = 0.20
    glint_min_samples: int = 5000
    glint_max_samples: int = 2000000
    glint_clip_min: float = 1e-6


    # CRS policy
    # working_srs: CRS used internally for meter-based operations (thinning, river DEM warps).
    # If not provided, we attempt to read it from the Sentinel-2 RGB_10m.tif CRS; otherwise
    # we compute a WGS84 UTM zone from the AOI center.
    working_srs: str = "auto"
    working_vcrs_epsg: int = 5703  # NAVD88 height (EPSG:5703)

    # Final output CRS for rasters (default NAD83 geographic (horizontal-only) height)
    final_out_srs: str = "EPSG:4269"

    # River DEM auto-download (TNM 1/3 arc-sec) when --river-dem is not provided.
    river_dem_auto: bool = True
    river_dem_source: str = "tnm:datasets=3"
    river_dem_res_m: float = 10.0
    extra_xyz_crs: str = "EPSG:4326"

    # River args
    river_dem: Optional[Path] = None
    river_soundings: Optional[str] = None
    # If river soundings are provided, they can refine the skeleton Dmax prior and optionally be enforced.
    river_soundings_mode: str = "auto"            # auto | depth_pos | depth_neg
    river_soundings_max_dist_m: float = 1500.0    # max distance for soundings to influence skeleton prior
    river_soundings_min_r: float = 0.25           # min r when inverting depth->Dmax
    river_soundings_enforce: bool = True          # enforce observed depths at sounding pixels
    # Optional authoritative bed elevation raster blending (NAVD88, etc.)
    river_authoritative_bed: Optional[Path] = None
    river_authoritative_bed_max_dist_m: float = 2000.0   # max distance for authoritative residual influence (m)
    river_residual_blend_sigma_m: float = 120.0          # Gaussian sigma for residual blending (m); 0 disables
    # River bathymetry method:
    # - "skeleton": raster distance-transform "channel skeleton" method (no cross-sections). Recommended for sinuous/tidal channels.
    # - "xs": legacy vector cross-section method (xs_builder.py + xs_infer_bathy_raster.py)
    river_method: str = "hybrid"

    # Skeleton (distance-transform) method parameters
    # These control *where* river bathy is applied (river vs. ocean) and the within-channel depth profile.
    river_channel_buffer_m: float = 400.0           # buffer around NHD flowlines to define candidate river corridor
    river_max_channel_width_m: float = 600.0        # max channel width allowed in corridor (prevents filling open bays)
    river_mainstem_min_order: int = 5               # stream order threshold for allowing larger widths (if available)
    river_max_mainstem_width_m: float = 2500.0      # max width allowed for mainstem corridor (m)
    river_shape_exp: float = 0.5                    # depth profile exponent (0.5 ~ U-shaped; 1.0 ~ V-shaped)
    river_dmax_min_m: float = 0.5                   # clamp Dmax prior (m)
    river_dmax_max_m: float = 30.0                  # clamp Dmax prior (m)
    # Optional: longitudinal bed profile constraints (skeleton method)
    river_bed_profile_max_slope: float = 0.0       # max |dz/ds| along flow (m/m); 0 disables
    river_bed_profile_max_curv: float = 0.0        # max |d2z/ds2| along flow (1/m); 0 disables
    river_bed_profile_step_m: float = 25.0         # sampling step (m) for profile constraints
    river_bed_profile_strength: float = 0.6        # blend strength (0..1)
    river_bed_profile_power: float = 2.0           # distance-decay power when spreading correction
    river_save_skeleton_debug: bool = False         # write debug rasters (r, d_bank, d_center, dmax, wse)

    # Skeleton WSE proxy controls (scientific correctness guardrails)
    river_skeleton_wse_mode: str = "bank"          # 'bank' (recommended) or 'skeleton' (legacy)
    river_skeleton_wse_smooth_sigma_m: float = 0.0 # optional smoothing of WSE field inside channel
    # Confluence/junction handling (degree>=3 graph nodes). Helps suppress artifacts near confluences.
    river_skeleton_junction_mode: str = "smooth"   # smooth | mask | none
    river_skeleton_junction_buffer_m: float = 120.0
    river_skeleton_junction_degree_min: int = 3
    river_skeleton_junction_smooth_sigma_m: float = 80.0
    river_skeleton_junction_max_width_m: float = 300.0
    # Optional: constrain river bathymetry domain using polygonal channel features (NHDArea)
    river_use_nhdarea: bool = True
    river_nhdarea_layer: str = "nhdarea_clip"
    river_ocean_keep_dist_m: float = 0.0  # allow ocean-connected water near flowlines (tidal mouths)
    # River depth inference priors / anchors (passed through to xs_infer_bathy_raster.py)
    river_prior_mode: str = "powerlaw"  # powerlaw | multivariate
    river_mv_a0: float = 0.18
    river_mv_bw: float = 0.50
    river_mv_ba: float = 0.0
    river_mv_bs: float = -0.10
    river_mv_eps_a: float = 1.0
    river_mv_eps_s: float = 1e-4

    river_usgs_sites: Optional[str] = None           # comma-separated site numbers
    river_usgs_start: Optional[str] = None           # YYYY-MM-DD
    river_usgs_end: Optional[str] = None             # YYYY-MM-DD
    river_usgs_cache_dir: Optional[Path] = None
    river_usgs_max_dist_m: float = 5000.0
    river_usgs_mean_to_dmax: str = "auto"
    river_usgs_a_stat: str = "median"                # median | p90 | mean
    river_usgs_q_quantile_lo: float = 0.20
    river_usgs_q_quantile_hi: float = 0.80
    river_usgs_a_cv_warn: float = 0.50
    river_usgs_width_ratio_max: float = 3.0
    river_usgs_width_ratio_blend: bool = True
    river_gage_snap_max_dist_m: float = 1000.0


    river_width_stage_csv: Optional[str] = None      # comma-separated CSV paths
    river_width_stage_max_dist_m: float = 5000.0
    river_width_stage_min_n: int = 6
    river_width_stage_min_r2: float = 0.25
    river_width_stage_max_weight: float = 0.8
    river_slope_proxy_window: int = 9
    river_slope_min: float = 1e-5
    river_slope_max: float = 0.05
    river_slope_proxy_min_n: int = 7

    # Longitudinal WSE profile fit (preferred slope proxy when reach slope attribute is missing)
    river_wse_profile_enabled: bool = True
    river_wse_profile_window: int = 9
    river_wse_profile_min_n: int = 7
    river_wse_profile_monotonic: bool = True

    # Optional: pass through to river_network
    # Hydrography acquisition strategy for river network & polygons
    # - "arcgis": ArcGIS REST only (fast, avoids TNM catalog crawl)
    # - "arcgis_tnm": ArcGIS first, then TNM fallback if ArcGIS fails
    # - "tnm": TNM preferred (will still fall back to ArcGIS as a guardrail)
    river_hydrography_source: str = "arcgis"

    tnm_enable: bool = False
    tnm_dataset: str = "NHDPlusHR"
    snap_m: float = 30.0

    # XS params
    xs_spacing_m: float = 200.0
    xs_length_m: float = 300.0
    xs_smoothing_window_m: float = 0.0   # 0 = auto (use xs_spacing_m)
    xs_trim_overlaps: bool = True        # local/adjacent trimming
    xs_global_deconflict: bool = True    # drop XS that intersect non-adjacent XS within a reach
    xs_deconflict_tol_m: float = 2.0     # treat endpoint "touches" within this tolerance as non-harmful
    xs_skip_junctions: bool = True       # avoid XS too close to confluences/junction nodes
    xs_junction_snap_m: float = 30.0     # snapping scale for junction detection (m)
    xs_junction_buffer_m: float = 75.0  # do not place XS within this distance of junctions (m)
    xs_densify_step_m: float = 20.0      # densify centerlines to this vertex spacing before tangents (m)

    # River patch rasterization / interpolation options (passed to xs_infer_bathy_raster.py)
    # NOTE: xs_infer defaults can be expensive for large networks; expose here for control.
    river_continuous: str = "walid_aniso"  # median | walid | aidw | aniso | walid_aniso
    river_continuous_buffer_m: Optional[float] = None
    river_continuous_k: int = 12
    river_idw_power: float = 2.0
    river_aniso_along_scale_m: float = 500.0
    river_aniso_cross_scale_m: float = 30.0
    river_thalweg_weight: float = 6.0

    river_thalweg_only: bool = False
    river_thalweg_densify_factor: float = 0.5
    river_thalweg_densify_step_m: Optional[float] = None
    river_overlap_reducer: str = "median"  # min | median
    river_nodata: float = -9999.0

    # Fusion controls
    fusion_strategy: str = "spatial_taper"  # weighted_overlap | spatial_taper | priority | blend
    fusion_primary_weight: float = 0.70
    fusion_secondary_weight: float = 0.30
    fusion_taper_m: float = 75.0  # meters for spatial taper inside river corridor

    # Output masking
    mask_river_to_waffles: bool = True  # apply waffles coastline mask to river depth + bed outputs when available

    # Intelligent gap-filling (Tier 1–2): prior + residual interpolation
    gapfill_enabled: bool = False
    gapfill_hq: Optional[List[str]] = None  # list of HQ point files (x,y,depth)
    gapfill_water_mask: Optional[Path] = None  # optional explicit water mask (1=water)
    gapfill_method: str = "rbf"  # rbf | gp | idw
    gapfill_river_smooth_sigma_m: float = 500.0  # along-channel smoothing sigma
    gapfill_prior_sigma_raster: Optional[Path] = None  # optional prior uncertainty raster
    gapfill_bank_elev_raster: Optional[Path] = None  # optional bank elevation for constraints
    gapfill_output_cudem_xyz: bool = False  # output CUDEM-compatible XYZ with uncertainty

    # Run health / contracts
    strict: bool = False  # if True, run contract tests and fail fast on invalid outputs


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_latest_waffles_mask(cache_root: Path) -> Optional[Path]:
    """Find the most recent waffles coastline mask in cache_root/masks/.

    Returns None if no mask is present.
    """
    try:
        masks_dir = Path(cache_root).resolve() / "masks"
        pats = sorted(masks_dir.glob("waffles_coastline_*.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
        return pats[0] if pats else None
    except Exception:
        return None



def _ensure_waffles_coastline_mask(
    cache_masks: Path,
    aoi: str,
    inc_arcsec: float = 1.0,
    want_nhd: bool = False,
    want_lakes: bool = False,
    prefix: str = "waffles_coastline",
    log: Optional[logging.Logger] = None,
) -> Path:
    """Ensure a waffles coastline mask exists and return the .tif path.

    Waffles coastline masks use the convention: land=1, water=0.
    When want_nhd=False and want_lakes=False, this should not depend on TNM and is
    intended to be resilient when TNM is flaky.
    """
    logger = log or logging.getLogger(__name__)
    cache_masks.mkdir(parents=True, exist_ok=True)

    params = f"want_nhd={str(bool(want_nhd)).lower()}:want_lakes={str(bool(want_lakes)).lower()}"
    chash = hashlib.sha1(f"{aoi}|{inc_arcsec:.9f}|{params}".encode()).hexdigest()[:12]
    out_prefix = cache_masks / f"{prefix}_{chash}"
    out_tif = out_prefix.with_suffix(".tif")

    if out_tif.exists() and out_tif.stat().st_size > 0:
        return out_tif

    inc_str = f"{inc_arcsec:.9f}s"
    cmd = [
        "waffles",
        "-M",
        f"coastline:{params}",
        f"-R={aoi}",
        "-E",
        inc_str,
        "-O",
        str(out_prefix),
    ]
    logger.info("[WAFFLES] Running: %s", " ".join(cmd))
    try:
        run_command(cmd, prefix="[WAFFLES] ")
    except Exception as e:
        # leave error handling to caller; waffles failures are expected in flaky network conditions
        raise

    if out_tif.exists() and out_tif.stat().st_size > 0:
        return out_tif

    # Fallback glob in case waffles varied the output name slightly
    cands = sorted(cache_masks.glob(f"{prefix}_{chash}*.tif"))
    if cands:
        return cands[0]

    raise RuntimeError("Waffles did not produce a coastline raster. Ensure 'waffles' is in your PATH.")


def _buffer_aoi(aoi: str, buf_deg: float = 0.0145) -> str:
    """Return AOI string buffered by buf_deg in degrees.

    AOI format: 'w/e/s/n'. Buffer expands outward (w-buf, e+buf, s-buf, n+buf).
    """
    try:
        w, e, s, n = [float(x) for x in aoi.split('/')[:4]]
    except Exception as exc:
        raise ValueError(f"Invalid AOI string: {aoi!r}") from exc
    return f"{w - buf_deg:.8f}/{e + buf_deg:.8f}/{s - buf_deg:.8f}/{n + buf_deg:.8f}"

def _mask_raster_to_waffles(raster_path: Path, waffles_mask_path: Path, nodata: float = -9999.0) -> bool:
    """Mask a raster *in place* to the waffles coastline mask.

    - Reprojects the waffles mask to the raster grid (nearest neighbor).
    - Auto-infers which mask value represents 'keep' by sampling under valid raster pixels.
    - Sets pixels outside the keep region to nodata.

    Returns True on success.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling

    raster_path = Path(raster_path)
    waffles_mask_path = Path(waffles_mask_path)

    if not raster_path.exists() or not waffles_mask_path.exists():
        return False

    # Avoid mutating cached products through a symlink.
    try:
        if raster_path.is_symlink():
            tgt = raster_path.resolve()
            tmp_copy = raster_path.with_suffix(".tmp_copy.tif")
            if tmp_copy.exists():
                tmp_copy.unlink()
            shutil.copy2(str(tgt), str(tmp_copy))
            raster_path.unlink()
            shutil.move(str(tmp_copy), str(raster_path))
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    tmp_out = raster_path.with_suffix(".tmp_masked.tif")

    with rasterio.open(raster_path) as ds:
        arr = ds.read(1).astype(np.float32)
        prof = ds.profile.copy()
        ds_nodata = prof.get("nodata", nodata)
        valid = np.isfinite(arr) & (arr != ds_nodata)

        if not valid.any():
            return False

        # Reproject waffles mask to ds grid
        with rasterio.open(waffles_mask_path) as ms:
            mask_src = ms.read(1)
            # IMPORTANT: initialize destination to LAND(1) so any pixels outside the reprojected
            # mask footprint remain LAND instead of uninitialized garbage (np.empty()).
            fill_val = 1  # waffles coastline mask convention: land=1, water=0
            mask_dst = np.full((ds.height, ds.width), fill_val, dtype=mask_src.dtype)

            # Only honor src_nodata if it is not a semantic (0/1) value.
            src_nodata = ms.nodata
            if src_nodata in (0, 1):
                src_nodata = None

            reproject(
                source=mask_src,
                destination=mask_dst,
                src_transform=ms.transform,
                src_crs=ms.crs,
                dst_transform=ds.transform,
                dst_crs=ds.crs,
                resampling=Resampling.nearest,
            )
        # STRICT keep rule: waffles coastline mask is binary with WATER=0, LAND=1.
        # We keep only WATER pixels (mask == 0) and set everything else to nodata.
        # (No auto-inference; avoids accidentally keeping land when interpolation spills onto banks.)
        keep = (mask_dst == 0)

        out = arr.copy()
        out[~keep] = float(ds_nodata)

        prof.update(nodata=float(ds_nodata), compress=prof.get("compress", "deflate"))

        if tmp_out.exists():
            tmp_out.unlink()
        with rasterio.open(tmp_out, "w", **prof) as out_ds:
            out_ds.write(out.astype(np.float32), 1)
            # Preserve tags
            try:
                out_ds.update_tags(**ds.tags())
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    # Atomic replace
    try:
        shutil.move(str(tmp_out), str(raster_path))
    finally:
        if tmp_out.exists():
            tmp_out.unlink()

    return True


def _mask_raster_to_nhdarea(
    raster_path: Path,
    nhd_gpkg: Path,
    nhd_layer: str = "nhdarea_clip",
    nodata: float = -9999.0,
) -> bool:
    """Mask *outside* NHDArea polygons (best-effort).

    Intended use:
      - Constrain river bathymetry outputs (especially XS interpolation) to polygonal river/channel
        features when those exist (NHDArea).
      - Avoids accidental bathy spill into adjacent water bodies or across banks.

    Returns True if a non-empty NHDArea mask was applied, False otherwise.
    """
    try:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.features import rasterize
    except Exception:
        return False

    raster_path = Path(raster_path)
    nhd_gpkg = Path(nhd_gpkg)

    if (not raster_path.exists()) or (not nhd_gpkg.exists()):
        return False

    try:
        areas = gpd.read_file(nhd_gpkg, layer=nhd_layer)
    except Exception:
        return False

    if areas is None or areas.empty:
        return False

    areas = areas[areas.geometry.notnull() & (~areas.geometry.is_empty)]
    areas = areas[areas.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if areas.empty:
        return False

    # Fix invalid polygons where possible
    if (~areas.is_valid).any():
        try:
            fixed = areas.geometry.buffer(0)
            ok = fixed.geom_type.isin(["Polygon", "MultiPolygon"])
            areas.loc[ok, "geometry"] = fixed[ok].values
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        areas = areas[areas.geometry.notnull() & (~areas.geometry.is_empty)]
        areas = areas[areas.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    if areas.empty:
        return False

    with rasterio.open(raster_path) as src:
        r_crs = src.crs
        transform = src.transform
        shape = (src.height, src.width)
        profile = src.profile.copy()
        arr = src.read(1)

    # Reproject polygons to raster CRS if needed
    try:
        if areas.crs is not None and r_crs is not None and areas.crs != r_crs:
            areas = areas.to_crs(r_crs)
    except Exception:
        # If CRS handling fails, try rasterizing in-place; worst case it yields empty mask and we no-op.
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    try:
        geom = _union_all_geoms(areas.geometry)
    except Exception:
        return False

    mask = rasterize(
        [(geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)

    if not bool(mask.any()):
        return False

    out = arr.copy()
    out[~mask] = nodata

    # Write via temp file then atomic replace
    tmp = raster_path.with_suffix(raster_path.suffix + ".nhdmask.tmp")
    profile.update(nodata=float(nodata))
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(out, 1)

    tmp.replace(raster_path)
    return True

def _utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """Return a WGS84 UTM EPSG code string (EPSG:326## or EPSG:327##) for lon/lat."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    if lat >= 0:
        return f"EPSG:{32600 + zone}"
    return f"EPSG:{32700 + zone}"

def _aoi_center_lonlat(aoi: str) -> tuple[float, float]:
    w, e, s, n = [float(x) for x in aoi.split("/")]
    return (0.5 * (w + e), 0.5 * (s + n))

def detect_working_srs(cfg: 'BathyConfig') -> str:
    """Determine the working CRS.

    Preference order:
      1) User-specified cfg.working_srs (anything other than 'auto')
      2) CRS of an existing Sentinel-2 RGB_10m.tif in the cache
      3) WGS84 UTM zone based on AOI center

    Returns a CRS string usable by GDAL/PROJ (e.g., 'EPSG:32616').
    """
    if cfg.working_srs and str(cfg.working_srs).lower() != "auto":
        return str(cfg.working_srs)

    # Try to detect from cached Sentinel-2 products
    try:
        import rasterio
        s2_root = Path(cfg.cache_root) / "sentinel2"
        if s2_root.exists():
            candidates = sorted(s2_root.glob("S2_*/*RGB_10m.tif")) + sorted(s2_root.glob("S2_*/RGB_10m.tif"))
            for p in candidates:
                try:
                    with rasterio.open(p) as ds:
                        if ds.crs:
                            return ds.crs.to_string()
                except Exception:
                    continue
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    # Fallback: AOI center UTM zone (WGS84)
    lon, lat = _aoi_center_lonlat(cfg.aoi)
    return _utm_epsg_from_lonlat(lon, lat)

def _compound_srs(horizontal_srs: str, v_epsg: int) -> str:
    # Keep whatever CRS string the user provided (EPSG:xxxx or WKT), but append +EPSG:xxxx for vertical.
    # PROJ/GDAL accept 'EPSG:XXXX+YYYY' for compound CRS in many contexts.
    h = str(horizontal_srs)
    v = f"{int(v_epsg)}"
    if "+" in h:
        # If already compound, leave as-is.
        return h
    if h.lower().startswith("epsg:"):
        return f"{h}+{v}"
    return f"{h}+{v}"



def ensure_river_dem_auto(cfg: 'BathyConfig', report: Dict[str, Any]) -> Optional[Path]:
    """Auto-download/build a river DEM using CUDEM `fetches` (TNM 1/3 arc-sec) when cfg.river_dem is not provided.

    Steps (best-effort):
      1) `fetches -R=<AOI> tnm:datasets=3` into <cache_root>/river_dem/tnm/
      2) If multiple versions of a tile exist (historical), keep the newest by date in filename.
      3) Build a clipped, reprojected DEM in the *working* CRS (cfg.working_srs) at cfg.river_dem_res_m.

    Returns a path to a GeoTIFF, or None on failure.
    """
    if cfg.river_dem and Path(cfg.river_dem).exists():
        return Path(cfg.river_dem)

    if not cfg.river_dem_auto:
        report.setdefault("river", {})["status"] = "skipped"
        report["river"]["reason"] = "river_dem_missing_and_auto_disabled"
        return None

    fetches_exe = shutil.which("fetches")
    if fetches_exe is None:
        report.setdefault("river", {})["status"] = "skipped"
        report["river"]["reason"] = "fetches_not_found"
        log.warning("[RIVER][DEM] fetches not found on PATH; cannot auto-download TNM DEM.")
        return None

    # Working CRS (projected, meter units preferred)
    working_srs = detect_working_srs(cfg)
    cfg.working_srs = working_srs

    dem_cache = ensure_dir(Path(cfg.cache_root) / "river_dem")
    tnm_dir = ensure_dir(dem_cache / "tnm")
    manifest = {
        "aoi": cfg.aoi,
        "source": cfg.river_dem_source,
        "working_srs": working_srs,
        "res_m": cfg.river_dem_res_m,
        "tiles_dir": str(tnm_dir),
    }
    report.setdefault("river", {})["dem_auto"] = manifest

    # Download into tnm_dir (fetches writes into subdir in CWD, so we run with cwd=tnm_dir parent)
    try:
        cmd = [fetches_exe, f'-R={cfg.aoi}', cfg.river_dem_source]
        log.info("[RIVER][DEM] Auto-download: %s", " ".join(cmd))
        proc = run_cmd(cmd, cwd=str(dem_cache))
        report["river"]["dem_auto"]["fetches_rc"] = proc.returncode
        report["river"]["dem_auto"]["fetches_stderr_tail"] = (proc.stderr or "")[-4000:]
        if proc.returncode != 0:
            log.warning("[RIVER][DEM] fetches failed (rc=%s). stderr tail: %s", proc.returncode, report["river"]["dem_auto"]["fetches_stderr_tail"])
    except Exception as e:
        log.warning("[RIVER][DEM] fetches exception: %s", e)

    # fetches for tnm typically creates a 'tnm' subdir in cwd; support both.
    tnm_search_dirs = [tnm_dir, dem_cache / "tnm"]
    tifs = []
    for d in tnm_search_dirs:
        if d.exists():
            tifs.extend(sorted(d.glob("*.tif")))
    if not tifs:
        log.warning("[RIVER][DEM] No TNM GeoTIFFs found after fetches.")
        report["river"]["dem_auto"]["status"] = "failed_no_tifs"
        return None

    # De-duplicate by tile id with newest date (USGS_13_n41w075_YYYYMMDD.tif)
    def tile_key(p: Path) -> str:
        m = re.search(r"(n\d{2}w\d{3})", p.name.lower())
        return m.group(1) if m else p.stem.lower()

    def date_key(p: Path) -> int:
        m = re.search(r"(\d{8})", p.name)
        return int(m.group(1)) if m else 0

    best = {}
    for p in tifs:
        k = tile_key(p)
        if k not in best or date_key(p) > date_key(best[k]):
            best[k] = p
    keep = sorted(best.values())
    report["river"]["dem_auto"]["tiles_kept"] = [str(p) for p in keep]

    # Build VRT then warp to working CRS + clip to AOI
    out_dem = dem_cache / f"river_dem_tnm_{_hash_key(cfg.aoi, working_srs, cfg.river_dem_res_m)}.tif"
    if out_dem.exists() and out_dem.stat().st_size > 0:
        log.info("[RIVER][DEM] Using cached river DEM: %s", out_dem)
        return out_dem

    gdalbuildvrt = shutil.which("gdalbuildvrt")
    gdalwarp = shutil.which("gdalwarp")
    if gdalbuildvrt is None or gdalwarp is None:
        log.warning("[RIVER][DEM] gdalbuildvrt/gdalwarp not found; cannot mosaic/warp TNM DEM.")
        report["river"]["dem_auto"]["status"] = "failed_no_gdal"
        return None

    vrt = dem_cache / "tnm_mosaic.vrt"
    try:
        cmd_vrt = [gdalbuildvrt, "-overwrite", str(vrt)] + [str(p) for p in keep]
        log.info("[RIVER][DEM] Build VRT: %s", " ".join(cmd_vrt[:6]) + (" ..." if len(cmd_vrt) > 6 else ""))
        run_cmd(cmd_vrt, check=True)
    except Exception as e:
        log.warning("[RIVER][DEM] gdalbuildvrt failed: %s", e)
        report["river"]["dem_auto"]["status"] = "failed_vrt"
        return None

    # Compute AOI bounds in working CRS for clipping
    try:
        from pyproj import Transformer
        w, e, s, n = [float(x) for x in cfg.aoi.split("/")]
        t = Transformer.from_crs("EPSG:4326", working_srs, always_xy=True)
        xs, ys = t.transform([w, e, w, e], [s, s, n, n])
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
    except Exception:
        minx = miny = maxx = maxy = None

    cmd_warp = [gdalwarp, "-overwrite", "-t_srs", working_srs, "-r", "bilinear",
                "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES"]
    # Force a sensible meter grid
    if cfg.river_dem_res_m and cfg.river_dem_res_m > 0:
        cmd_warp += ["-tr", str(cfg.river_dem_res_m), str(cfg.river_dem_res_m), "-tap"]
    if minx is not None:
        cmd_warp += ["-te", str(minx), str(miny), str(maxx), str(maxy)]
    cmd_warp += [str(vrt), str(out_dem)]

    try:
        log.info("[RIVER][DEM] Warp/clip: %s", " ".join(cmd_warp[:10]) + (" ..." if len(cmd_warp) > 10 else ""))
        run_cmd(cmd_warp, check=True)
        if out_dem.exists() and out_dem.stat().st_size > 0:
            report["river"]["dem_auto"]["status"] = "success"
            return out_dem
    except Exception as e:
        log.warning("[RIVER][DEM] gdalwarp failed: %s", e)
        report["river"]["dem_auto"]["status"] = "failed_warp"

    return None




def apply_depth_metadata(
    raster_path: Path,
    depth_sign: str = "negative_down",
    depth_reference: str = "water_surface",
) -> None:
    """Attach depth semantics metadata to a GeoTIFF (best-effort).

    The CRS is intentionally horizontal-only for depth rasters. Pixel values represent depth
    relative to a stated reference surface (e.g., water_surface, terrain_surface).
    """
    try:
        import rasterio

        depth_reference = (depth_reference or "water_surface").strip().lower()
        if depth_reference == "water_surface":
            vdatum = "N/A (relative depth)"
            note = "Depth values are relative to the water surface; no orthometric vertical datum applies."
        elif depth_reference in ("terrain_surface", "bank_elevation", "dem_surface"):
            vdatum = "DEM-derived (relative depth)"
            note = "Depth values are relative to the terrain/bank surface from the provided DEM; no orthometric vertical datum applies to depth."
        else:
            vdatum = "N/A (relative depth)"
            note = f"Depth values are relative to '{depth_reference}'."

        tags = {
            "VALUE_TYPE": "depth",
            "DEPTH_UNITS": "m",
            "DEPTH_SIGN": depth_sign,
            "DEPTH_REFERENCE": depth_reference,
            "VERTICAL_DATUM": vdatum,
            "VERTICAL_DATUM_NOTE": note,
        }

        with rasterio.open(raster_path, "r+") as dst:
            dst.update_tags(**tags)
    except Exception:
        return


def apply_elevation_metadata(
    raster_path: Path,
    vertical_datum: str = "NAVD88",
    units: str = "m",
) -> None:
    """Attach elevation semantics metadata to a GeoTIFF (best-effort).

    For elevation rasters, CRS is still horizontal-only in this pipeline; vertical datum is
    described via metadata tags.
    """
    try:
        import rasterio
        tags = {
            "VALUE_TYPE": "elevation",
            "ELEV_UNITS": units,
            "VERTICAL_DATUM": vertical_datum,
            "VERTICAL_DATUM_NOTE": "Elevation values are orthometric heights in the stated vertical datum; CRS may be horizontal-only.",
        }
        with rasterio.open(raster_path, "r+") as dst:
            dst.update_tags(**tags)
    except Exception:
        return


def compute_depth_from_bed_and_dem(
    bed_elev_tif: Path,
    dem_tif: Path,
    out_depth_tif: Path,
    depth_sign: str = "negative_down",
) -> None:
    """Compute depth relative to DEM surface: depth = bed_elev - dem (negative when bed below terrain).

    Assumes rasters are co-registered (same grid/extent). Uses chunked IO.
    """
    import numpy as np
    import rasterio

    ensure_dir(out_depth_tif.parent)

    with rasterio.open(bed_elev_tif) as bed, rasterio.open(dem_tif) as dem:
        if (bed.width != dem.width) or (bed.height != dem.height) or (bed.transform != dem.transform):
            raise ValueError("bed_elev_tif and dem_tif must be on the same grid to compute depth")

        profile = bed.profile.copy()
        profile.update(dtype="float32", count=1, compress="DEFLATE", predictor=2)

        bed_nodata = bed.nodata
        dem_nodata = dem.nodata
        nodata_out = -9999.0
        profile.update(nodata=float(nodata_out))

        with rasterio.open(out_depth_tif, "w", **profile) as dst:
            for ji, window in bed.block_windows(1):
                b = bed.read(1, window=window).astype("float32")
                d = dem.read(1, window=window).astype("float32")

                mask = np.zeros(b.shape, dtype=bool)
                if bed_nodata is not None:
                    mask |= (b == bed_nodata)
                if dem_nodata is not None:
                    mask |= (d == dem_nodata)
                mask |= ~np.isfinite(b) | ~np.isfinite(d)

                depth = b - d  # negative when b < d (bed below terrain)
                depth[mask] = float(nodata_out)

                dst.write(depth.astype("float32"), 1, window=window)

    # Tag semantics
    apply_depth_metadata(out_depth_tif, depth_sign=depth_sign, depth_reference="terrain_surface")


def warp_raster_to_srs(in_raster: Path, out_raster: Path, dst_srs: str, write_depth_metadata: bool = True) -> Optional[Path]:
    """Warp a raster to dst_srs (best-effort)."""
    gdalwarp = shutil.which("gdalwarp")
    if gdalwarp is None:
        log.warning("[WARP] gdalwarp not found; cannot reproject %s", in_raster)
        return None
    if out_raster.exists() and out_raster.stat().st_size > 0:
        if write_depth_metadata:
            apply_depth_metadata(out_raster)
        return out_raster
    cmd = [
        gdalwarp, "-overwrite", "-t_srs", str(dst_srs),
        "-r", "bilinear",
        "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
        str(in_raster), str(out_raster)
    ]
    try:
        log.info("[WARP] %s -> %s (%s)", in_raster.name, out_raster.name, dst_srs)
        run_cmd(cmd, check=True)
        if out_raster.exists() and write_depth_metadata:
            apply_depth_metadata(out_raster)
        return out_raster if out_raster.exists() else None
    except Exception as e:
        log.warning("[WARP] gdalwarp failed: %s", e)
        return None

# -----------------------------------------------------------------------------
# CUDEM dlim auto-soundings helpers
# -----------------------------------------------------------------------------

_CUDEM_XYZ_SOURCE_ALIASES = {
    # NOAA/NOS Hydrographic surveys via CUDEM provider name
    "nos": "hydronos",
    "hydronos": "hydronos",
    # USACE eHydro
    "usace": "ehydro",
    "ehydro": "ehydro",
}

def _normalize_cudem_source_name(src: str) -> str:
    s = (src or "").strip().lower()
    if not s:
        return s
    return _CUDEM_XYZ_SOURCE_ALIASES.get(s, s)

def _hash_key(*parts: str, n: int = 12) -> str:
    # Be defensive: callers sometimes pass floats/ints.
    h = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:n]


def _reproject_xyz_file(
    xyz_path: Path,
    src_crs: str,
    dst_crs: str,
    cache_dir: Path,
) -> Path:
    """Reproject an XYZ file from src_crs to dst_crs.
    
    Returns path to reprojected file (cached).
    Only reprojects X,Y coordinates; Z (depth) is unchanged.
    """
    from pyproj import Transformer
    import numpy as np
    
    # Normalize CRS strings
    src_crs_str = str(src_crs).upper()
    dst_crs_str = str(dst_crs).upper()
    
    # If same CRS, return original
    # Strip vertical component for comparison (e.g., EPSG:4269+5703 -> EPSG:4269)
    src_h = src_crs_str.split("+")[0] if "+" in src_crs_str else src_crs_str
    dst_h = dst_crs_str.split("+")[0] if "+" in dst_crs_str else dst_crs_str
    if src_h == dst_h:
        return xyz_path
    
    # Build cache key and output path
    key = _hash_key(str(xyz_path), src_crs_str, dst_crs_str)
    out_path = cache_dir / f"{xyz_path.stem}_{dst_h.replace(':', '')}_{key}.xyz"
    
    # Cache hit
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    
    # Read XYZ (space or comma delimited, 3+ columns: x, y, z, ...)
    data = []
    with open(xyz_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    extra = parts[3:] if len(parts) > 3 else []
                    data.append((x, y, z, extra))
                except ValueError:
                    continue
    
    if not data:
        raise ValueError(f"No valid XYZ data in {xyz_path}")
    
    # Transform coordinates (horizontal only)
    xs = np.array([d[0] for d in data])
    ys = np.array([d[1] for d in data])
    zs = np.array([d[2] for d in data])
    extras = [d[3] for d in data]
    
    # Use horizontal CRS for transformation (strip vertical)
    transformer = Transformer.from_crs(src_h, dst_h, always_xy=True)
    xs_out, ys_out = transformer.transform(xs, ys)
    
    # Write output
    tmp_path = out_path.with_suffix(".xyz.tmp")
    with open(tmp_path, "w") as f:
        for i in range(len(xs_out)):
            extra_str = " ".join(extras[i]) if extras[i] else ""
            if extra_str:
                f.write(f"{xs_out[i]:.6f} {ys_out[i]:.6f} {zs[i]:.4f} {extra_str}\n")
            else:
                f.write(f"{xs_out[i]:.6f} {ys_out[i]:.6f} {zs[i]:.4f}\n")
    
    tmp_path.replace(out_path)
    return out_path


def fetch_cudem_soundings_via_dlim(
    *,
    aoi: str,
    sources: List[str],
    cache_root: Path,
    out_crs: str,
    thin_res_m: Optional[float] = 10.0,
    filter_spec: Optional[str] = None,
    force: bool = False,
    prefix: str = "[XYZ][DLIM] ",
) -> Tuple[List[Path], Dict[str, Any]]:
    """Fetch external soundings using CUDEM `dlim` providers.

    Writes one .xyz per source into: <cache_root>/xyz/
    Returns (paths, report_dict). Never raises (best-effort).
    """
    report: Dict[str, Any] = {
        "requested_sources": list(sources),
        "out_crs": out_crs,
        "thin_res_m": thin_res_m,
        "filter_spec": filter_spec,
        "outputs": [],
        "status": "skipped",
    }

    dlim_exe = shutil.which("dlim")
    if dlim_exe is None:
        report["status"] = "skipped"
        report["reason"] = "dlim_not_found"
        log.warning("%sdlim not found on PATH; skipping --extra-xyz-cudem", prefix)
        return [], report

    # WARNING: If output CRS is geographic, block_thin:res is interpreted in *degrees*.
    # For convenience, when the user passes thin_res_m (meters) and did not provide an
    # explicit filter_spec, we convert meters -> degrees using an AOI-center latitude approximation.
    # NOTE: dlim's block_thin takes a single scalar "res" in coordinate units, so we choose a degree
    # value that is conservative in longitude (accounts for cos(lat)).
    out_crs_lower = str(out_crs).lower()
    is_geographic = any(x in out_crs_lower for x in ["4326", "4269", "4267"])
    thin_res_m_eff = thin_res_m
    if is_geographic and thin_res_m and thin_res_m > 0 and (filter_spec is None or str(filter_spec).strip() == ""):
        try:
            w, e, s, n = [float(x) for x in str(aoi).split("/")]
            lat0 = 0.5 * (s + n)
        except Exception:
            lat0 = 0.0
        import math
        coslat = max(0.2, abs(math.cos(math.radians(lat0))))
        meters_per_deg_lon = 111320.0 * coslat
        thin_res_m_eff = float(thin_res_m) / meters_per_deg_lon
        log.warning(
            "%sGeographic output CRS (%s): interpreting --extra-xyz-cudem-thin-res-m=%s m as ~%.8f degrees for dlim block_thin (lat0=%.4f).",
            prefix, out_crs, thin_res_m, thin_res_m_eff, lat0
        )
    elif is_geographic and thin_res_m and thin_res_m > 0:
        log.info(
            "%sNote: Output CRS is geographic (%s). Your explicit filter_spec will be passed to dlim unchanged.",
            prefix, out_crs
        )

    xyz_cache = ensure_dir(Path(cache_root) / "xyz")
    out_paths: List[Path] = []

    for src_raw in sources:
        src = _normalize_cudem_source_name(src_raw)
        if not src:
            continue

        key = _hash_key(
            "dlim",
            src,
            aoi,
            out_crs,
            filter_spec or (f"block_thin:res={thin_res_m_eff}" if thin_res_m_eff else "no_filter"),
        )
        out_xyz = xyz_cache / f"{src}_{key}.xyz"

        # Build dlim command
        # Basic: dlim -R=W/E/S/N <source>
        # With projection: dlim -R=W/E/S/N <source> -P epsg:XXXX
        # With filter: dlim -R=W/E/S/N <source> -F block_thin:res=10
        cmd = [dlim_exe, f'-R={aoi}', src]
        
        # Add projection if specified (dlim defaults to epsg:4326 output)
        if out_crs:
            # dlim -P expects lowercase 'epsg:' format
            crs_str = str(out_crs).lower()
            if not crs_str.startswith("epsg:"):
                crs_str = f"epsg:{crs_str}" if crs_str.isdigit() else crs_str
            cmd += ["-P", crs_str]
        
        # Optional thinning/filtering
        this_filter = filter_spec or (f"block_thin:res={thin_res_m_eff}" if thin_res_m_eff and thin_res_m_eff > 0 else None)
        if this_filter:
            cmd += ["-F", this_filter]
        
        cmd_str = " ".join(str(c) for c in cmd)

        # Cache hit
        if out_xyz.exists() and out_xyz.stat().st_size > 0 and not force:
            log.info("%sCache hit for %s: %s", prefix, src, str(out_xyz))
            out_paths.append(out_xyz)
            report["outputs"].append({
                "source": src,
                "path": str(out_xyz),
                "status": "cached",
                "command": cmd_str,
            })
            continue

        # (Re)download
        log.info("%sFetching %s via dlim -> %s", prefix, src, str(out_xyz))
        log.info("%sCommand: %s", prefix, cmd_str)
        stderr_tail = ""
        rc = 999
        try:
            tmp_path = out_xyz.with_suffix(".xyz.tmp")
            if tmp_path.exists():
                tmp_path.unlink()

            with open(tmp_path, "w", encoding="utf-8") as f_out:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stderr = proc.stderr.read() if proc.stderr else ""
                rc = proc.wait()
                stderr_tail = (stderr or "")[-4000:]

            # Basic sanity: non-empty file
            file_size = tmp_path.stat().st_size if tmp_path.exists() else 0
            if rc != 0 or file_size == 0:
                if tmp_path.exists():
                    tmp_path.unlink()
                if rc == 0 and file_size == 0:
                    log.warning("%sNo data returned for %s (file is empty, rc=0).", prefix, src)
                    log.warning("%sThis could mean: (1) no data in AOI, (2) filter too aggressive, or (3) projection issue.", prefix)
                    if stderr_tail:
                        log.warning("%sdlim stderr: %s", prefix, stderr_tail.strip()[-500:])
                    status = "no_data"
                else:
                    log.warning("%sFailed fetch for %s (rc=%s).", prefix, src, rc)
                    if stderr_tail:
                        log.warning("%sdlim stderr: %s", prefix, stderr_tail.strip()[-500:])
                    status = "failed"
                report["outputs"].append({
                    "source": src,
                    "path": str(out_xyz),
                    "status": status,
                    "returncode": rc,
                    "command": cmd_str,
                    "stderr_tail": stderr_tail,
                })
                continue

            # Atomic move into cache
            tmp_path.replace(out_xyz)

            out_paths.append(out_xyz)
            report["outputs"].append({
                "source": src,
                "path": str(out_xyz),
                "status": "downloaded",
                "returncode": rc,
                "command": cmd_str,
            })

        except Exception as e:
            try:
                if out_xyz.exists() and out_xyz.stat().st_size == 0:
                    out_xyz.unlink()
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
            log.warning("%sException fetching %s via dlim: %s", prefix, src, str(e))
            report["outputs"].append({
                "source": src,
                "path": str(out_xyz),
                "status": "failed",
                "returncode": rc,
                "command": cmd_str,
                "stderr_tail": stderr_tail,
                "exception": str(e),
            })

    if out_paths:
        report["status"] = "success"
    elif report["outputs"]:
        report["status"] = "failed"
    else:
        report["status"] = "skipped"

    return out_paths, report


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    # Flight recorder breadcrumb (best effort)
    try:
        from flight_recorder import emit_artifact_written
        emit_artifact_written(path, kind="json", role="report_or_metadata")
    except Exception as e:
        logging.getLogger(__name__).debug("Optional flight-recorder emit failed: %s", e)


def run_command(
    cmd: Any,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    prefix: str = "",
    stream_stdout: bool = True,
    stream_stderr: bool = True,
    stdout_log_path: Optional[Path] = None,
    stderr_log_path: Optional[Path] = None,
    max_lines: int = 8000,
    tail_chars: int = 16000,
) -> Tuple[int, str, str]:
    """
    Run a command as a list of arguments (NOT shell=True for security).
    
    SECURITY FIX: Using list-based arguments prevents shell injection attacks.
    
    Args:
        cmd: List of command arguments (e.g., [sys.executable, "script.py", "--arg=value"])
        cwd: Working directory
        env: Environment variables
        prefix: Prefix for log messages
        stream_stdout: Whether to stream stdout to log
        stream_stderr: Whether to stream stderr to log
        max_lines: Maximum lines to keep in memory
        tail_chars: Maximum chars to return
        
    Returns:
        Tuple of (return_code, stdout_tail, stderr_tail)
    """
    env = env or os.environ.copy()

    # Allow cmd to be either a list(argv) or a single command string.
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    cmd = [str(c) for c in cmd]

    stdout_lines = deque(maxlen=max_lines)
    stderr_lines = deque(maxlen=max_lines)
    stdout_fh = None
    stderr_fh = None
    try:
        if stdout_log_path is not None:
            ensure_dir(Path(stdout_log_path).parent)
            stdout_fh = open(stdout_log_path, 'w', buffering=1, encoding='utf-8')
        if stderr_log_path is not None:
            ensure_dir(Path(stderr_log_path).parent)
            stderr_fh = open(stderr_log_path, 'w', buffering=1, encoding='utf-8')
    except Exception:
        stdout_fh = None
        stderr_fh = None

    
    # Log the command being run (safely formatted)
    cmd_str = " ".join(str(c) for c in cmd)
    log.debug(f"Executing: {cmd_str}")

    # Flight recorder: capture subprocess lifecycle
    _fr = None
    try:
        from flight_recorder import FlightRecorder

        _fr = FlightRecorder.global_instance()
    except Exception:
        _fr = None
    if _fr is not None:
        _fr.record_event(
            "subprocess_start",
            cmd=cmd,
            cmd_str=cmd_str,
            cwd=str(cwd) if cwd else None,
            prefix=prefix,
        )

    proc = subprocess.Popen(
        cmd,  # List of arguments - no shell injection possible
        shell=False,  # SECURITY: Never use shell=True with user input
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    def _pump(stream, sink, log_fn, pfx: str, enabled: bool, fh=None):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                sink.append(line)
                if fh is not None:
                    try:
                        fh.write(line)
                    except Exception:
                        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                if enabled:
                    log_fn(f"{pfx}{line.rstrip()}" if pfx else line.rstrip())
        finally:
            try:
                stream.close()
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    threads: List[threading.Thread] = []
    if proc.stdout is not None:
        threads.append(threading.Thread(
            target=_pump,
            args=(proc.stdout, stdout_lines, log.info, prefix, stream_stdout, stdout_fh),
            daemon=True,
        ))
    if proc.stderr is not None:
        err_pfx = f"{prefix}[stderr] " if prefix else "[stderr] "
        threads.append(threading.Thread(
            target=_pump,
            args=(proc.stderr, stderr_lines, log.warning, err_pfx, stream_stderr, stderr_fh),
            daemon=True,
        ))

    for t in threads:
        t.start()

    rc = proc.wait()

    for t in threads:
        t.join(timeout=2.0)

    try:
        if stdout_fh is not None:
            stdout_fh.close()
        if stderr_fh is not None:
            stderr_fh.close()
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    out = "".join(stdout_lines)
    err = "".join(stderr_lines)

    if _fr is not None:
        _fr.record_event(
            "subprocess_end",
            cmd=cmd,
            rc=int(rc),
            stdout_tail=out[-tail_chars:] if isinstance(out, str) else None,
            stderr_tail=err[-tail_chars:] if isinstance(err, str) else None,
        )
    if len(out) > tail_chars:
        out = out[-tail_chars:]
    if len(err) > tail_chars:
        err = err[-tail_chars:]

    # Flight recorder heuristic: many GDAL/tools write to the last argument.
    # Emit artifact_written if the last token looks like a path and exists.
    try:
        from flight_recorder import emit_artifact_written
        if isinstance(cmd, (list, tuple)) and len(cmd) >= 2:
            cand = cmd[-1]
            if isinstance(cand, (str, Path)):
                p = Path(str(cand))
                if p.exists() and p.is_file():
                    ext = p.suffix.lower()
                    kind = "raster" if ext in (".tif", ".tiff", ".vrt") else "file"
                    emit_artifact_written(p, kind=kind, role="subprocess_output")
    except Exception as e:
        logging.getLogger(__name__).debug("Optional flight-recorder emit failed: %s", e)

    return rc, out, err



def _append_river_soundings_args(cmd, cfg, *, include_calib_args=True, include_mode_args=False):
    """Append soundings (extra XYZ) args for river inference scripts.

    cfg.river_soundings is a comma-separated list of files, already in working CRS.
    - xs_infer_bathy_raster.py supports calibration args (--calib-*) and --soundings-crs
    - river_skeleton_bathy.py supports soundings mode/weighting args (--soundings-mode, etc.)
    """
    try:
        snd = getattr(cfg, 'river_soundings', None)
        if not snd:
            return
        snd_list = [s.strip() for s in str(snd).split(',') if s.strip()]
        if not snd_list:
            return

        # Common soundings file args
        # xs_infer_bathy_raster.py defines --soundings as action='append' (one argument per flag),
        # while river_skeleton_bathy.py defines --soundings with nargs='*' (many after one flag).
        # Use the CLI shape that matches the target script to avoid stray positional args.
        if include_calib_args and not include_mode_args:
            for _snd in snd_list:
                cmd.append(f"--soundings={_snd}")
        else:
            cmd.extend(['--soundings'] + snd_list)

        # XY are already in working CRS (reprojected earlier when needed).
        work_srs = str(getattr(cfg, 'working_srs', '') or '').strip()
        if work_srs:
            cmd.append(f"--soundings-crs={work_srs}")

        if include_calib_args:
            try:
                d = float(getattr(cfg, 'river_soundings_calib_max_dist_m', 0.0) or 0.0)
                if d > 0:
                    cmd.append(f"--calib-max-dist-m={d}")
            except Exception:
                pass
            try:
                st = str(getattr(cfg, 'river_soundings_calib_stat', '') or '').strip()
                if st:
                    cmd.append(f"--calib-stat={st}")
            except Exception:
                pass

        if include_mode_args:
            # Skeleton-side sounding assimilation controls
            try:
                sm = str(getattr(cfg, 'river_soundings_mode', 'auto') or 'auto').strip()
                if sm:
                    cmd.append(f"--soundings-mode={sm}")
            except Exception:
                pass
            try:
                sp = float(getattr(cfg, 'river_soundings_cell_percentile', 25.0) or 0.0)
                if sp > 0:
                    cmd.append(f"--soundings-cell-percentile={sp}")
            except Exception:
                pass
            try:
                md = float(getattr(cfg, 'river_soundings_max_dist_m', 150.0) or 0.0)
                if md > 0:
                    cmd.append(f"--soundings-max-dist-m={md}")
            except Exception:
                pass
            try:
                mr = float(getattr(cfg, 'river_soundings_min_r', 0.15) or 0.0)
                if mr > 0:
                    cmd.append(f"--soundings-min-r={mr}")
            except Exception:
                pass
            if bool(getattr(cfg, 'river_no_soundings_enforce', False)):
                cmd.append('--no-soundings-enforce')
    except Exception:
        logging.getLogger(__name__).debug('Failed to append river soundings args; continuing.', exc_info=True)


def _build_cmd(*args) -> List[str]:
    """
    Build a command list from arguments, filtering None values.
    
    This helper ensures all arguments are properly converted to strings
    and None values are excluded.
    """
    return [str(a) for a in args if a is not None]


def _score_sdb_candidate(tif: Path) -> float:
    """
    Heuristic scoring to pick the best SDB depth raster.
    Higher = better.
    """
    name = tif.name.lower()

    # Exclude obvious non-products
    bad_tokens = ["rgb", "mask", "land", "clear", "qa", "scl", "doa", "weights", "uncert", "error", "diff"]
    if any(tok in name for tok in bad_tokens):
        return -1.0

    # Prefer typical product tokens
    good = 0.0
    if "sdb" in name:
        good += 5.0
    if "rf" in name:
        good += 3.0
    if "depth" in name or "bathy" in name:
        good += 2.0
    if "10m" in name:
        good += 1.0
    if "aligned" in name or "final" in name or "product" in name:
        good += 1.0

    # Prefer bigger/newer files (often the actual raster product)
    try:
        size = tif.stat().st_size
        mtime = tif.stat().st_mtime
        good += min(size / 1e8, 5.0)   # cap size influence
        good += min(mtime / 1e10, 5.0) # small nudge for recency
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    return good


def _is_depth_raster_like(path: Path, expect_negative: bool = True) -> tuple[bool, str]:
    """Quick guardrail to prevent mistaking optical imagery/masks for bathymetry.

    Returns (ok, reason). This is intentionally lightweight and uses a small decimated read.
    """
    try:
        import rasterio
        import numpy as np
        from rasterio.enums import Resampling

        if not path.exists():
            return False, "missing"
        with rasterio.open(path) as ds:
            if int(ds.count) != 1:
                return False, f"band_count={ds.count}"
            # Prefer float depth products; integer-only is usually masks/reflectance
            if str(ds.dtypes[0]).startswith(("uint", "int")):
                # Allow int only if values look like small signed depths (rare)
                pass

            h, w = int(ds.height), int(ds.width)
            out_h = min(256, h)
            out_w = min(256, w)
            arr = ds.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.nearest,
            ).astype("float32")

            nod = ds.nodata
            m = np.isfinite(arr)
            if nod is not None:
                m &= (arr != float(nod))
            if not np.any(m):
                # Could be a valid depth raster with no predictions (all nodata)
                return True, "all_nodata"

            vals = arr[m]
            p1, p50, p99 = np.percentile(vals, [1, 50, 99])

            # Depth magnitudes should not be enormous. If they are, this is likely reflectance/mask encoded.
            if np.abs(p99) > 500.0:
                return False, f"p99_abs_too_large={float(np.abs(p99)):.2f}"

            # Optical reflectance products are typically non-negative; depths should be mostly negative-down
            if expect_negative:
                frac_neg = float(np.mean(vals < 0))
                if frac_neg < 0.01:
                    # If everything is small non-negative (0..1 or 0..10000), it's almost certainly imagery/mask
                    if float(p1) >= -1e-6:
                        return False, f"too_few_negative(frac_neg={frac_neg:.3f}, p1={float(p1):.3f}, p99={float(p99):.3f})"

            return True, f"ok(p50={float(p50):.3f}, p99={float(p99):.3f})"
    except Exception as e:
        return False, f"exception:{e}"

def find_sdb_depth_raster(sdb_dir: Path) -> Optional[Path]:
    """
    Robustly locate the SDB depth product raster under sdb_dir.
    """
    if not sdb_dir.exists():
        return None

    tifs = list(sdb_dir.rglob("*.tif"))
    if not tifs:
        return None

    # Hard candidates first (match current sdb_main outputs)
    hard = [
        # Current canonical outputs from sdb_main.py
        sdb_dir / "rasters" / "SDB_Prediction_10m.tif",
        sdb_dir / "rasters" / "SDB_Prediction_10m_NAD83.tif",
        sdb_dir / "rasters" / "SDB_Prediction_10m_NAVD88.tif",
        sdb_dir / "rasters" / "SDB_Prediction_10m_NAD83_NAVD88.tif",
        # Older/alternate naming (back-compat)
        sdb_dir / "rasters" / "SDB_RF_10m.tif",
        sdb_dir / "rasters" / "SDB_RF_10m_ALIGNED.tif",
        sdb_dir / "product" / "SDB_RF_10m.tif",
        sdb_dir / "product" / "SDB_RF_10m_ALIGNED.tif",
        sdb_dir / "output" / "SDB_RF_10m.tif",
        sdb_dir / "output" / "SDB_RF_10m_ALIGNED.tif",
    ]
    for c in hard:
        if c.exists():
            ok, why = _is_depth_raster_like(c, expect_negative=True)
            if ok:
                return c
            try:
                log.warning(f"[SDB] Candidate depth raster rejected ({why}): {c}")
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    # Heuristic selection
    scored = []
    for t in tifs:
        s = _score_sdb_candidate(t)
        if s > 0:
            scored.append((s, t))
    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)

    # Validate candidates to avoid selecting optical imagery (e.g., B02/B03) as "depth".
    for _, cand in scored:
        ok, why = _is_depth_raster_like(cand, expect_negative=True)
        if ok:
            return cand
        try:
            log.warning(f"[SDB] Heuristic candidate rejected ({why}): {cand}")
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    return None


def find_sdb_land_mask(sdb_dir: Path) -> Optional[Path]:
    """Locate the aligned land mask produced by sdb_main.py.

    Convention: sdb_main writes rasters/LAND_MASK_aligned.tif where land=1, water=0.
    """
    try:
        c = sdb_dir / "rasters" / "LAND_MASK_aligned.tif"
        if c.exists():
            return c
        # fallback search
        for t in sdb_dir.rglob("LAND_MASK_aligned.tif"):
            return t
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    return None


def find_sdb_wse_navd88_raster(sdb_dir: Path) -> Optional[Path]:
    """Best-effort locator for an SDB water-surface elevation raster (NAVD88).

    Some workflows optionally output a rasterized/warped water surface (WSE)
    referenced to NAVD88. If present, we can convert SDB depths (positive-down)
    to bottom elevations as: bottom = WSE - depth.
    """
    if not sdb_dir.exists():
        return None
    cand = []
    for t in sdb_dir.rglob("*.tif"):
        n = t.name.lower()
        if ("wse" in n or "water_surface" in n or "waterlevel" in n) and ("navd88" in n or "vert" in n):
            cand.append(t)
    if not cand:
        return None
    # Prefer aligned/final naming
    def _score(p: Path) -> float:
        n = p.name.lower()
        s = 0.0
        if "aligned" in n or "final" in n:
            s += 2.0
        try:
            s += min(p.stat().st_size / 1e8, 5.0)
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        return s
    cand.sort(key=_score, reverse=True)
    return cand[0]



# -----------------------------------------------------------------------------
# SDB
# -----------------------------------------------------------------------------

def run_sdb(cfg: BathyConfig, report: Dict[str, Any]) -> Optional[Path]:
    log.info("=" * 60)
    log.info("RUNNING SDB PIPELINE (Coastal/Nearshore)")
    log.info("=" * 60)

    sdb_dir = ensure_dir(cfg.out_dir / "sdb")
    script_dir = Path(__file__).parent

    cmd = [
        sys.executable, "sdb_main.py",
        f"--aoi={cfg.aoi}",
        f"--start={cfg.start_date}",
        f"--end={cfg.end_date}",
        f"--out-dir={sdb_dir}",
        f"--cloud={cfg.cloud}",
        f"--icesat={cfg.icesat}",
        f"--sdb-mode={cfg.sdb_mode}",
        f"--cache-root={cfg.cache_root}",
        f"--align-mode={cfg.align_mode}",
        f"--working-srs={cfg.working_srs}",
        f"--working-vcrs-epsg={cfg.working_vcrs_epsg}",
        # NOTE: --atl-transform-datum is NOT passed by default.
        # ATL depths are relative (not geodetic heights), so vertical transforms would corrupt them.
        # The horizontal transform (WGS84 -> NAD83) is tiny and usually unnecessary.
    ]

    # Bounded model bank (reservoir) + periodic retrain (DEFAULT)
    try:
        if bool(getattr(cfg, 'sdb_model_bank_enabled', True)):
            cmd.append(f"--model-bank={getattr(cfg, 'sdb_model_bank', 'auto')}")
            cmd.append(f"--bank-max-samples={int(getattr(cfg, 'sdb_bank_max_samples', 100000))}")
            cmd.append(f"--bank-seed={int(getattr(cfg, 'sdb_bank_seed', 1337))}")
            cmd.append(f"--bank-retrain-min-new={int(getattr(cfg, 'sdb_bank_retrain_min_new', 2000))}")
        else:
            cmd.append("--no-model-bank")
    except Exception:
        log.debug('Unexpected exception suppressed (model bank policy).', exc_info=True)

    # Deprecated regional model cache passthrough (kept for compatibility; disabled by default)
    try:
        if bool(getattr(cfg, "sdb_model_cache_enabled", False)):
            cmd.append(f"--model-cache-key={getattr(cfg, 'sdb_model_cache_key', 'auto')}")
        else:
            cmd.append("--no-model-cache")
    except Exception:
        log.debug('Unexpected exception suppressed (model cache policy).', exc_info=True)

    # Forward S2 sun-glint correction flags into SDB, if supported.
    if getattr(cfg, "glint_correct", False):
        sdb_main_path = (Path(__file__).parent / "sdb_main.py")
        supports_glint = False
        try:
            if sdb_main_path.exists():
                txt = sdb_main_path.read_text(encoding="utf-8", errors="ignore")
                supports_glint = ("--glint-correct" in txt) or ("glint_correct" in txt)
        except Exception:
            supports_glint = False

        if not supports_glint:
            log.warning("[SDB][GLINT] --glint-correct requested, but sdb_main.py does not appear to support glint flags; skipping glint passthrough.")
        else:
            cmd.append("--glint-correct")
            cmd.append(f"--glint-nir-band={cfg.glint_nir_band}")
            cmd.append(f"--glint-vis-bands={cfg.glint_vis_bands}")
            cmd.append(f"--glint-nir-min-percentile={cfg.glint_nir_min_percentile}")
            cmd.append(f"--glint-deepwater-b02-max={cfg.glint_deepwater_b02_max}")
            cmd.append(f"--glint-min-samples={cfg.glint_min_samples}")
            cmd.append(f"--glint-max-samples={cfg.glint_max_samples}")
            cmd.append(f"--glint-clip-min={cfg.glint_clip_min}")

    # Forward extra XYZ bathymetry into SDB training/fusion if provided.
    # cfg.river_soundings is a comma-separated list of files produced by --extra-xyz normalization.
    if cfg.river_soundings:
        xyz_list = [p.strip() for p in str(cfg.river_soundings).split(",") if p.strip()]
        if xyz_list:
            cmd += ["--extra-xyz"] + xyz_list
            cmd.append(f"--extra-xyz-crs={cfg.extra_xyz_crs}")
    
    # Forward adaptive sampling parameters (v0.7.0+, enabled by default v0.7.1)
    enable_sampling = getattr(cfg, 'enable_adaptive_sampling', True)
    if not enable_sampling:
        # User explicitly disabled it
        cmd.append("--disable-adaptive-sampling")
    
    # Always pass the tuning parameters (even if disabled, sdb_main will ignore them)
    cmd.append(f"--sampling-target-points={getattr(cfg, 'sampling_target_points', 2000)}")
    cmd.append(f"--sampling-min-threshold={getattr(cfg, 'sampling_min_threshold', 3000)}")
    cmd.append(f"--sampling-max-gap-m={getattr(cfg, 'sampling_max_gap_m', 100.0)}")

    log.info(f"[SDB] Command: {cmd}")
    logs_dir = ensure_dir(cfg.out_dir / "logs")
    rc, out, err = run_command(
        cmd,
        cwd=script_dir,
        prefix="[SDB] ",
        stdout_log_path=logs_dir / "sdb.stdout.log",
        stderr_log_path=logs_dir / "sdb.stderr.log",
    )

    report["sdb"] = {
        "status": "success" if rc == 0 else "failed",
        "returncode": rc,
        "command": cmd,
        "stdout_tail": out,
        "stderr_tail": err,
    }

    if rc != 0:
        log.error(f"[SDB] Failed with code {rc}")
        return None

    depth = find_sdb_depth_raster(sdb_dir)
    if depth is None:
        log.warning("[SDB] Completed but could not find a depth raster in SDB output tree.")
        return None

    log.info(f"[SDB] Depth raster found: {depth}")
    try:
        apply_depth_metadata(Path(depth))
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    return Path(depth).resolve() if not isinstance(depth, Path) else depth.resolve()


# -----------------------------------------------------------------------------
# River
# -----------------------------------------------------------------------------

def run_river(cfg: BathyConfig, report: Dict[str, Any]) -> Optional[Path]:
    log.info("=" * 60)
    log.info("RUNNING RIVER PIPELINE (method=%s)", getattr(cfg, "river_method", "xs"))
    log.info("=" * 60)

    if cfg.river_dem is None:
        auto_dem = ensure_river_dem_auto(cfg, report)
        if auto_dem is None:
            log.error("[RIVER] Missing --river-dem (and auto-build failed)")
            report["river"] = {"status": "failed", "reason": "missing river_dem"}
            return None
        cfg.river_dem = auto_dem

    river_dir = ensure_dir(cfg.out_dir / "river")
    script_dir = Path(__file__).parent

    # River cache (hashed) — avoids stale reuse when DEM/soundings/params change.
    cache_root = Path(cfg.cache_root).resolve() / "river"
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_key, cache_manifest = _river_cache_key_and_manifest(cfg)
    cache_dir = cache_root / cache_key
    work_dir = cache_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "manifest.json"

    # River interpolation produces bed elevation (on the DEM's vertical datum). We store:
    #   - bed_elev raster (orthometric elevation in DEM datum, usually NAVD88)
    #   - depth raster relative to DEM terrain surface: depth = bed_elev - dem (negative when bed below terrain)
    cached_bed_tif = cache_dir / "river_bed_elev_patch.tif"
    cached_depth_tif = cache_dir / "river_depth_terrain_patch.tif"

    report["river_cache"] = {
        "cache_root": str(cache_root),
        "cache_key": cache_key,
        "cache_dir": str(cache_dir),
        "hit": False,
    }

    # Cache hit if manifest matches and both rasters exist.
    if cached_bed_tif.exists() and cached_depth_tif.exists() and manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old == cache_manifest:
                report["river_cache"]["hit"] = True
                log.info("[RIVER][CACHE] Hit: %s", str(cache_dir))

                out_depth = river_dir / "river_depth_terrain_patch.tif"
                out_bed = river_dir / "river_bottom_navd88_patch.tif"

                # Ensure output directory exists (important when out_dir is relative and cwd may change).
                try:
                    out_depth.parent.mkdir(parents=True, exist_ok=True)
                    out_bed.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


                # Materialize into run output folder for convenience.
                # Be careful with broken symlinks: Path.exists() is False for a broken link,
                # so use lexists()/is_symlink() and verify readability after creation.
                for src, dst in [(cached_depth_tif, out_depth), (cached_bed_tif, out_bed)]:
                    src = Path(src)
                    dst = Path(dst)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        src_real = src.resolve(strict=True)
                    except Exception as e:
                        log.warning("[RIVER][CACHE] Cached source missing; cannot materialize %s from %s (%s)", str(dst), str(src), e)
                        raise
                    try:
                        if dst.is_symlink() or os.path.lexists(str(dst)):
                            dst.unlink()
                    except FileNotFoundError:
                        pass
                    except Exception:
                        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                        try:
                            if dst.exists():
                                dst.unlink()
                        except Exception:
                            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                    materialized = False
                    try:
                        os.symlink(str(src_real), str(dst))
                        # Validate symlink resolves to an existing file before trusting it.
                        if dst.exists() and dst.is_file():
                            materialized = True
                            log.info("[RIVER][CACHE] Materialized symlink: %s -> %s", str(dst), str(src_real))
                            # Extra validation: ensure GDAL/rasterio can open the materialized path (symlinks can fail on some mounts).
                            try:
                                import rasterio
                                with rasterio.open(str(dst)) as _ds:
                                    _ = _ds.count
                            except Exception as e_open:
                                try:
                                    if dst.is_symlink() or os.path.lexists(str(dst)):
                                        dst.unlink()
                                except Exception:
                                    logging.getLogger(__name__).debug('Optional step failed; continuing.', exc_info=True)
                                raise OSError(f'symlink exists but is not readable by rasterio/GDAL: {e_open}')
                        else:
                            try:
                                if dst.is_symlink() or os.path.lexists(str(dst)):
                                    dst.unlink()
                            except Exception:
                                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                            raise OSError("symlink created but destination is not readable")
                    except Exception as e:
                        try:
                            if dst.is_symlink() or os.path.lexists(str(dst)):
                                dst.unlink()
                        except Exception:
                            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                        shutil.copy2(str(src_real), str(dst))
                        materialized = True
                        log.warning("[RIVER][CACHE] Symlink materialization failed (%s); copied instead: %s", e, str(dst))
                    if not materialized or (not dst.exists()):
                        raise FileNotFoundError(f"[RIVER][CACHE] Failed to materialize cached raster: {dst}")

                report["river"] = {
                    "status": "success",
                    "returncode": 0,
                    "command": "<cache-hit>",
                    "outputs": {
                        "depth_terrain": str(out_depth),
                        "bottom_elevation": str(out_bed),
                    },
                    "steps": {},
                }

                # Optional: mask river outputs to waffles coastline (if available).
                try:
                    if cfg.mask_river_to_waffles:
                        wm = _find_latest_waffles_mask(cfg.cache_root)
                        if wm and wm.exists():
                            # Do NOT eagerly unlink+copy symlinks here: if copy fails, we can accidentally
                            # delete the materialized cache-hit output and then silently continue. The masking helper
                            # already contains its own symlink-safe copy-on-write logic.
                            md_ok = _mask_raster_to_waffles(out_depth, wm, nodata=-9999.0)
                            mb_ok = _mask_raster_to_waffles(out_bed, wm, nodata=-9999.0)
                            report["river"].setdefault("masking", {})["waffles_mask"] = str(wm)
                            report["river"].setdefault("masking", {})["masked_depth"] = bool(md_ok)
                            report["river"].setdefault("masking", {})["masked_bed"] = bool(mb_ok)
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

                try:
                    apply_depth_metadata(out_depth, depth_reference="terrain_surface")
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                try:
                    apply_elevation_metadata(out_bed, vertical_datum="NAVD88")
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

                # Final cache-hit validation: do not return a path that no longer exists after
                # optional masking/metadata steps. If validation fails, fall through and rebuild.
                for _pth in (out_depth, out_bed):
                    try:
                        if not Path(_pth).exists():
                            raise FileNotFoundError(f"[RIVER][CACHE] Materialized cache output vanished before return: {_pth}")
                    except Exception:
                        raise

                return out_depth.resolve()
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    report["river"] = {"status": "running", "steps": {}}

    # Step 1: river network
    log.info("[RIVER] Step 1: Extracting river network...")
    network_gpkg = work_dir / "river_network.gpkg"

    # SECURITY FIX: Use list-based command construction (no shell injection)
    cmd = [
        sys.executable, "river_network.py",
        f"--aoi={cfg.aoi}",
        f"--out-gpkg={network_gpkg}",
        f"--hydrography-source={cfg.river_hydrography_source}",
        f"--tnm-dataset={cfg.tnm_dataset}",
        f"--snap-m={cfg.snap_m}",
    ]
    if cfg.tnm_enable:
        cmd.append("--tnm-enable")

    rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
    cmd_str = " ".join(str(c) for c in cmd)  # For logging only
    report["river"]["steps"]["network"] = {
        "status": "success" if rc == 0 else "failed",
        "returncode": rc,
        "command": cmd_str,
        "stdout_tail": out,
        "stderr_tail": err,
    }
    if rc != 0 or not network_gpkg.exists():
        log.error("[RIVER] Failed to extract river network.")
        report["river"]["status"] = "failed"
        return None

    log.info(f"[RIVER] Network extracted: {network_gpkg}")


    river_method = str(getattr(cfg, "river_method", "hybrid")).lower().strip()
    

    # Hybrid default: XS only on mainstem (order>=threshold + largest component), skeleton elsewhere.
    # Rationale: XS is most defensible on mainstem and most failure-prone at dense tributary junctions.
    # This preserves continuity along the mainstem while avoiding tributary overlap artifacts.

    def _build_domain_masks(_work_dir: Path, *, strict: bool = False):
        """Build channel/open-water/mainstem masks (DEM-aligned).

        If strict=True, missing channel mask is treated as fatal.
        """
        channel_mask_tif = _work_dir / "river_channel_mask.tif"
        open_water_mask_tif = _work_dir / "open_water_mask.tif"
        mainstem_mask_tif = _work_dir / "mainstem_mask.tif"

        cache_masks = Path(cfg.cache_root) / "masks"
        aoi_buf = _buffer_aoi(str(cfg.aoi), float(getattr(cfg, 'waffles_aoi_buffer_deg', 0.0145)))

        ocean_mask = None
        with_nhd_mask = None
        try:
            ocean_mask = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks,
                aoi=aoi_buf,
                inc_arcsec=1.0,
                want_nhd=False,
                want_lakes=False,
                prefix="waffles_coastline_ocean_only",
                log=log,
            )
        except Exception as e:
            log.warning(f"[WAFFLES] Ocean-only mask unavailable; ocean bleed protection degraded: {e}")
            ocean_mask = None

        try:
            with_nhd_mask = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks,
                aoi=aoi_buf,
                inc_arcsec=1.0,
                want_nhd=True,
                want_lakes=False,
                prefix="waffles_coastline_with_nhd",
                log=log,
            )
        except Exception as e:
            log.warning(f"[WAFFLES] With-NHD mask unavailable (TNM flaky?): {e}. Proceeding with corridor+ArcGIS NHD flowlines + ocean-only mask.")
            with_nhd_mask = None

        cmd = [
            sys.executable, "river_domain_mask.py",
            f"--river-gpkg={network_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-channel-mask={channel_mask_tif}",
            f"--out-open-water-mask={open_water_mask_tif}",
            f"--out-mainstem-mask={mainstem_mask_tif}",
            f"--channel-buffer-m={getattr(cfg, 'river_channel_buffer_m', 400.0)}",
            f"--max-channel-width-m={getattr(cfg, 'river_max_channel_width_m', 600.0)}",
            f"--mainstem-min-order={getattr(cfg, 'river_mainstem_min_order', 5)}",
            f"--max-mainstem-width-m={getattr(cfg, 'river_max_mainstem_width_m', 2500.0)}",
        ]

        chan_src = str(getattr(cfg, 'river_channel_source', 'auto') or 'auto').strip().lower()
        if chan_src not in ('auto', 'nhdarea', 'corridor'):
            log.warning(f"[RIVER] Unknown river_channel_source='{chan_src}', defaulting to 'auto'.")
            chan_src = 'auto'
        cmd.append(f"--channel-source={chan_src}")

        nhd_allow = getattr(cfg, 'river_nhdarea_allow_ftype', None)
        if nhd_allow is None:
            nhd_allow = "460"
        cmd.append(f"--nhdarea-allow-ftype={nhd_allow}")
        nhd_allow_fcode = getattr(cfg, 'river_nhdarea_allow_fcode', None)
        if nhd_allow_fcode:
            cmd.append(f"--nhdarea-allow-fcode={nhd_allow_fcode}")

        if (chan_src in ('auto', 'nhdarea')) and bool(getattr(cfg, "river_use_nhdarea", True)):
            cmd.append(f"--nhdarea-gpkg={network_gpkg}")
            cmd.append(f"--nhdarea-layer={getattr(cfg, 'river_nhdarea_layer', 'nhdarea_clip')}")

        if ocean_mask and Path(ocean_mask).exists():
            cmd.append(f"--ocean-mask={ocean_mask}")
        oke = float(getattr(cfg, 'river_ocean_keep_dist_m', 0.0) or 0.0)
        if (oke > 0.0):
            cmd.append(f"--ocean-keep-dist-m={oke}")

        if with_nhd_mask and Path(with_nhd_mask).exists():
            cmd.append(f"--water-mask={with_nhd_mask}")

        # Persist the waffles mask used (for downstream final clipping/QA).
        try:
            wm = None
            if with_nhd_mask and Path(with_nhd_mask).exists():
                wm = with_nhd_mask
            elif ocean_mask and Path(ocean_mask).exists():
                wm = ocean_mask
            report.setdefault("river", {}).setdefault("outputs", {})["waffles_water_mask"] = str(wm) if wm else None
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["domain_mask"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or (not channel_mask_tif.exists()):
            msg = "[RIVER] Failed to build channel mask"
            if strict:
                log.error("%s (strict mode).", msg)
            else:
                log.warning("%s; continuing without hard domain constraint.", msg)
        else:
            try:
                cfg.river_domain_mask_for_fusion = Path(channel_mask_tif)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        return channel_mask_tif, open_water_mask_tif, mainstem_mask_tif


    def _combine_mainstem_xs_and_skeleton(
        bed_xs: Path,
        bed_skel: Path,
        mainstem_mask: Path,
        out_bed: Path,
        template: Path,
        nodata: float,
    ) -> None:
        """Hybrid combine: use XS where available inside mainstem mask; otherwise skeleton.

        This guarantees a continuous mainstem bed even if XS has gaps (fallback to skeleton).
        Output is aligned to template grid.
        """
        import numpy as np
        import rasterio
        from rasterio.warp import reproject, Resampling

        def _read_align(src_path: Path, band: int = 1, resamp=Resampling.nearest):
            with rasterio.open(template) as tmpl:
                prof = tmpl.profile.copy()
                prof.update(dtype="float32", nodata=nodata, count=1, compress="deflate")
                arr = np.full((tmpl.height, tmpl.width), nodata, dtype=np.float32)
                with rasterio.open(src_path) as src:
                    reproject(
                        source=rasterio.band(src, band),
                        destination=arr,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=tmpl.transform,
                        dst_crs=tmpl.crs,
                        resampling=resamp,
                        src_nodata=src.nodata,
                        dst_nodata=nodata,
                    )
            return arr, prof

        xs_a, prof = _read_align(bed_xs, resamp=Resampling.nearest)
        sk_a, _ = _read_align(bed_skel, resamp=Resampling.nearest)
        ms_a, _ = _read_align(mainstem_mask, resamp=Resampling.nearest)
        ms = ms_a > 0.5

        out = sk_a.copy()
        # Guard against NaNs propagating into the combined raster.
        xs_ok = np.isfinite(xs_a) & (xs_a != nodata)
        take = ms & xs_ok
        out[take] = xs_a[take]

        out_bed.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_bed, "w", **prof) as dst:
            dst.write(out.astype(np.float32), 1)


    if river_method == "hybrid":
        log.info("[RIVER] Using HYBRID method: XS(mainstem) + skeleton(elsewhere)")

        # Step 2: build domain masks (includes mainstem corridor)
        log.info("[RIVER] Step 2: Building river channel/mainstem masks (hybrid)...")
        channel_mask_tif, open_water_mask_tif, mainstem_mask_tif = _build_domain_masks(work_dir, strict=True)

        # Hybrid requires a valid channel mask (skeleton method depends on it). Fail closed if it is missing.
        if channel_mask_tif is None or (not Path(channel_mask_tif).exists()):
            log.error("[RIVER] Hybrid requires a valid channel mask, but it was not created: %s", str(channel_mask_tif))
            report["river"]["status"] = "failed"
            return None

        # If the mainstem mask is empty/missing, we can still run XS builder (largest component),
        # but XS will not be injected into the final hybrid raster; warn loudly.
        if mainstem_mask_tif is None or (not Path(mainstem_mask_tif).exists()):
            log.warning("[RIVER] Mainstem mask not created; hybrid will effectively behave like skeleton-only (XS not injected).")

        # Step 3a: XS on mainstem only
        log.info("[RIVER] Step 3a: Generating cross-sections (mainstem only, conservative defaults)...")
        xs_gpkg = work_dir / "cross_sections_mainstem.gpkg"

        cmd = [
            sys.executable, "xs_builder.py",
            f"--river-gpkg={network_gpkg}",
            f"--dem={cfg.river_dem}",
            f"--out-gpkg={xs_gpkg}",
            f"--spacing-m={cfg.xs_spacing_m}",
            f"--half-width-m={cfg.xs_length_m / 2.0}",
            f"--smoothing-window-m={getattr(cfg, 'xs_smoothing_window_m', 0.0)}",
            f"--deconflict-tol-m={getattr(cfg, 'xs_deconflict_tol_m', 2.0)}",
            f"--junction-snap-m={getattr(cfg, 'xs_junction_snap_m', 30.0)}",
            f"--junction-buffer-m={getattr(cfg, 'xs_junction_buffer_m', 75.0)}",
            f"--densify-step-m={getattr(cfg, 'xs_densify_step_m', 20.0)}",
            f"--min-stream-order={int(getattr(cfg, 'river_mainstem_min_order', 5))}",
            "--keep-top-components=1",
        ]
        if not bool(getattr(cfg, "xs_trim_overlaps", True)):
            cmd.append("--no-trim-overlaps")
        if not bool(getattr(cfg, "xs_global_deconflict", True)):
            cmd.append("--no-global-deconflict")
        if not bool(getattr(cfg, "xs_skip_junctions", True)):
            cmd.append("--no-skip-junctions")

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["xs_builder_mainstem"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not xs_gpkg.exists():
            log.error("[RIVER] Failed to build mainstem cross-sections for hybrid method.")
            report["river"]["status"] = "failed"
            return None

        log.info(f"[RIVER] Mainstem cross-sections generated: {xs_gpkg}")

        log.info("[RIVER] Step 3b: Inferring mainstem bathymetry from XS...")
        bed_xs_tif = work_dir / "river_bed_elev_xs_mainstem.tif"
        bathy_gpkg = work_dir / "river_bathy_xs_mainstem.gpkg"

        cmd = [
            sys.executable, "xs_infer_bathy_raster.py",
            f"--xs-gpkg={xs_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-gpkg={bathy_gpkg}",
            f"--out-bathy-raster={bed_xs_tif}",
            f"--continuous={getattr(cfg, 'river_continuous', 'walid_aniso')}",
            f"--continuous-k={getattr(cfg, 'river_continuous_k', 12)}",
            f"--idw-power={getattr(cfg, 'river_idw_power', 2.0)}",
            f"--aniso-along-scale-m={getattr(cfg, 'river_aniso_along_scale_m', 500.0)}",
            f"--aniso-cross-scale-m={getattr(cfg, 'river_aniso_cross_scale_m', 30.0)}",
            f"--thalweg-weight={getattr(cfg, 'river_thalweg_weight', 6.0)}",
            f"--nodata={getattr(cfg, 'river_nodata', -9999.0)}",
            f"--overlap-reducer={getattr(cfg, 'river_overlap_reducer', 'min')}",
        ]
        if channel_mask_tif is not None and Path(channel_mask_tif).exists():
            cmd.append(f"--channel-mask-raster={channel_mask_tif}")
            cmd.append("--channel-mask-inside-value=1")
        cmd.append(f"--river-gpkg={network_gpkg}")
        cmd.append(f"--prior-mode={cfg.river_prior_mode}")
        cmd.append(f"--mv-a0={cfg.river_mv_a0}")
        cmd.append(f"--mv-bw={cfg.river_mv_bw}")
        cmd.append(f"--mv-ba={cfg.river_mv_ba}")
        cmd.append(f"--mv-bs={cfg.river_mv_bs}")
        cmd.append(f"--mv-eps-a={cfg.river_mv_eps_a}")
        cmd.append(f"--mv-eps-s={cfg.river_mv_eps_s}")
        cmd.append(f"--slope-proxy-window={cfg.river_slope_proxy_window}")
        cmd.append(f"--slope-min={cfg.river_slope_min}")
        cmd.append(f"--slope-max={cfg.river_slope_max}")
        cmd.append(f"--slope-proxy-min-n={cfg.river_slope_proxy_min_n}")
        if bool(getattr(cfg, "river_wse_profile_enabled", True)):
            cmd.append("--wse-profile-enabled")
        else:
            cmd.append("--no-wse-profile")
        cmd.append(f"--wse-profile-window={getattr(cfg, 'river_wse_profile_window', cfg.river_slope_proxy_window)}")
        cmd.append(f"--wse-profile-min-n={getattr(cfg, 'river_wse_profile_min_n', cfg.river_slope_proxy_min_n)}")
        if bool(getattr(cfg, "river_wse_profile_monotonic", True)):
            cmd.append("--wse-profile-monotonic")
        else:
            cmd.append("--no-wse-profile-monotonic")

        # Pass authoritative point soundings (e.g., extra_xyz subsets) into XS inference anchoring when available.
        if (not getattr(cfg, "river_soundings", None)) and getattr(cfg, "extra_xyz_files", None):
            try:
                _fallback_soundings = [str(p) for p in (cfg.extra_xyz_files or []) if p]
                if _fallback_soundings:
                    cfg.river_soundings = _fallback_soundings
                    log.info("[RIVER] XS inference: using extra_xyz fallback as soundings anchors (n=%d)", len(_fallback_soundings))
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        # If soundings are available, generate a single cached subset ONCE and reuse it for BOTH
        # XS + skeleton so they see the exact same points (seam consistency + reproducibility).
        # Default output is parquet (fast/small).
        soundings_subset_path = None
        try:
            if getattr(cfg, "river_soundings", None):
                soundings_subset_path = work_dir / "river_soundings_subset.parquet"
                soundings_subset_meta = work_dir / "river_soundings_subset.meta.json"

                # Build a lightweight signature so we don't accidentally reuse a stale subset when
                # the soundings inputs or sampling config changed between reruns.
                def _soundings_signature(cfg_obj) -> str:
                    import hashlib
                    import json
                    inp = getattr(cfg_obj, "river_soundings", None)
                    if inp is None:
                        files = []
                    elif isinstance(inp, (list, tuple)):
                        files = [str(p) for p in inp]
                    else:
                        files = [str(inp)]

                    info = []
                    for fp in files:
                        try:
                            p = Path(fp)
                            if p.exists() and p.is_file():
                                st = p.stat()
                                info.append({"path": fp, "size": int(st.st_size), "mtime": float(st.st_mtime)})
                            else:
                                info.append({"path": fp, "size": None, "mtime": None})
                        except Exception:
                            info.append({"path": fp, "size": None, "mtime": None})

                    payload = {
                        "files": sorted(info, key=lambda d: d.get("path") or ""),
                        "soundings_max_points": int(getattr(cfg_obj, "soundings_max_points", 0) or 0),
                        "soundings_sample_seed": int(getattr(cfg_obj, "soundings_sample_seed", 0) or 0),
                    }
                    b = json.dumps(payload, sort_keys=True).encode("utf-8")
                    return hashlib.sha1(b).hexdigest()

                sig_now = _soundings_signature(cfg)

                # If subset exists but meta is missing or doesn't match, rebuild.
                if soundings_subset_path.exists():
                    try:
                        import json
                        if not soundings_subset_meta.exists():
                            log.info("[RIVER] Cached soundings subset exists but meta missing; rebuilding for safety.")
                            soundings_subset_path.unlink(missing_ok=True)
                        else:
                            meta = json.loads(soundings_subset_meta.read_text(encoding="utf-8"))
                            if meta.get("signature") != sig_now:
                                log.info("[RIVER] Cached soundings subset signature mismatch; rebuilding.")
                                soundings_subset_path.unlink(missing_ok=True)
                    except Exception:
                        logging.getLogger(__name__).debug("Subset meta check failed; rebuilding subset to be safe.", exc_info=True)
                        try:
                            soundings_subset_path.unlink(missing_ok=True)
                        except Exception:
                            pass

                # Create the subset if missing.
                if not soundings_subset_path.exists():
                    _tmp_out_gpkg = work_dir / "_tmp_soundings_subset_only.gpkg"
                    _tmp_out_tif = work_dir / "_tmp_soundings_subset_only.tif"
                    cmd_subset = [
                        sys.executable, "xs_infer_bathy_raster.py",
                        f"--xs-gpkg={xs_gpkg}",
                        f"--template-raster={cfg.river_dem}",
                        f"--out-gpkg={_tmp_out_gpkg}",
                        f"--out-bathy-raster={_tmp_out_tif}",
                        f"--continuous={getattr(cfg, 'river_continuous', 'walid_aniso')}",
                        f"--nodata={getattr(cfg, 'river_nodata', -9999.0)}",
                        f"--write-soundings-subset={soundings_subset_path}",
                        "--only-write-soundings-subset",
                    ]
                    _append_river_soundings_args(cmd_subset, cfg, include_calib_args=True, include_mode_args=False)
                    rc_s, out_s, err_s = run_command(cmd_subset, cwd=script_dir, prefix="[RIVER] ")
                    report["river"]["steps"]["soundings_subset"] = {
                        "status": "success" if rc_s == 0 else "failed",
                        "returncode": rc_s,
                        "command": " ".join(str(c) for c in cmd_subset),
                        "stdout_tail": out_s,
                        "stderr_tail": err_s,
                    }
                    if rc_s != 0 or not soundings_subset_path.exists():
                        log.warning("[RIVER] Failed to create cached soundings subset (rc=%s); continuing with raw soundings.", rc_s)
                        soundings_subset_path = None

                    # Write a tiny meta sidecar for reproducibility / stale-cache protection.
                    if soundings_subset_path is not None and soundings_subset_path.exists():
                        try:
                            import json
                            import pandas as _pd
                            meta = {
                                "signature": sig_now,
                                "subset_path": str(soundings_subset_path),
                                "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "soundings_max_points": int(getattr(cfg, "soundings_max_points", 0) or 0),
                                "soundings_sample_seed": int(getattr(cfg, "soundings_sample_seed", 0) or 0),
                            }
                            # Record per-source counts if available.
                            try:
                                if soundings_subset_path.suffix.lower() == ".parquet":
                                    df = _pd.read_parquet(soundings_subset_path, columns=["_src_file"])
                                    if "_src_file" in df.columns:
                                        vc = df["_src_file"].astype(str).value_counts()
                                        meta["by_src"] = {Path(str(k)).stem if str(k) not in ["", "nan", "None"] else "unknown": int(v) for k, v in vc.items()}
                                        meta["n_subset"] = int(len(df))
                            except Exception:
                                pass
                            soundings_subset_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
                        except Exception:
                            logging.getLogger(__name__).debug("Failed to write subset meta; continuing.", exc_info=True)

                # If we have a valid subset, point cfg.river_soundings at it so ALL downstream steps share it.
                if soundings_subset_path is not None and soundings_subset_path.exists():
                    cfg.river_soundings = str(soundings_subset_path)
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        _append_river_soundings_args(cmd, cfg, include_calib_args=True, include_mode_args=False)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["infer_xs_mainstem"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not bed_xs_tif.exists():
            if rc != 0 and bathy_gpkg.exists() and not bed_xs_tif.exists():
                log.warning("[RIVER] XS inference wrote GPKG but no raster (rc=%s); falling back to skeleton-only. stderr_tail=%s", rc, (err or '').strip()[-500:])
            elif rc != 0:
                log.warning("[RIVER] XS mainstem inference failed (rc=%s); hybrid will fall back to skeleton-only. stderr_tail=%s", rc, (err or '').strip()[-500:])
            else:
                log.warning("[RIVER] XS mainstem inference completed but raster missing; hybrid will fall back to skeleton-only.")
            bed_xs_tif = None

        # Note: cfg.river_soundings may already point at the cached subset (preferred).

        # Step 3c: Skeleton for full river network
        log.info("[RIVER] Step 3c: Inferring bathymetry from skeleton (full network)...")
        bed_skel_tif = work_dir / "river_bed_elev_skeleton_full.tif"

        cmd = [
            sys.executable, "river_skeleton_bathy.py",
            f"--river-gpkg={network_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--dem={cfg.river_dem}",
            f"--channel-mask={channel_mask_tif}",
            f"--out-bed={bed_skel_tif}",
            f"--dmax-min-m={getattr(cfg, 'river_dmax_min_m', 1.0)}",
            f"--dmax-max-m={getattr(cfg, 'river_dmax_max_m', 20.0)}",
            f"--shape-exp={getattr(cfg, 'river_shape_exp', 0.33)}",
            f"--bed-profile-max-slope={getattr(cfg, 'river_bed_profile_max_slope', 0.015)}",
            f"--bed-profile-max-curv={getattr(cfg, 'river_bed_profile_max_curv', 0.0005)}",
            f"--bed-profile-step-m={getattr(cfg, 'river_bed_profile_step_m', 50.0)}",
            f"--bed-profile-strength={getattr(cfg, 'river_bed_profile_strength', 0.65)}",
            f"--bed-profile-power={getattr(cfg, 'river_bed_profile_power', 2.0)}",
            f"--prior-mode={cfg.river_prior_mode}",
            f"--mv-a0={cfg.river_mv_a0}",
            f"--mv-bw={cfg.river_mv_bw}",
            f"--mv-ba={cfg.river_mv_ba}",
            f"--mv-bs={cfg.river_mv_bs}",
            f"--mv-eps-a={cfg.river_mv_eps_a}",
            f"--mv-eps-s={cfg.river_mv_eps_s}",
            f"--residual-blend-sigma-m={float(getattr(cfg, 'river_residual_blend_sigma_m', 120.0))}",
            f"--authoritative-bed-max-dist-m={float(getattr(cfg, 'river_authoritative_bed_max_dist_m', 2000.0))}",
            f"--wse-mode={getattr(cfg, 'river_skeleton_wse_mode', 'bank_profile')}",
            f"--wse-smooth-sigma-m={getattr(cfg, 'river_skeleton_wse_smooth_sigma_m', 250.0)}",
        ]

        # Optional authoritative bed raster blending
        if getattr(cfg, 'river_authoritative_bed', None):
            try:
                ab = Path(getattr(cfg, 'river_authoritative_bed'))
                if ab.exists():
                    cmd.append(f"--authoritative-bed-raster={ab}")
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        # Junction / confluence handling
        jmode = str(getattr(cfg, "river_skeleton_junction_mode", "smooth")).strip().lower()
        if jmode and jmode != "none":
            cmd.append(f"--junction-mode={jmode}")
            cmd.append(f"--junction-buffer-m={float(getattr(cfg, 'river_skeleton_junction_buffer_m', 120.0))}")
            cmd.append(f"--junction-degree-min={int(getattr(cfg, 'river_skeleton_junction_degree_min', 3))}")
            jsig = float(getattr(cfg, "river_skeleton_junction_smooth_sigma_m", 80.0) or 0.0)
            if jsig > 0.0:
                cmd.append(f"--junction-smooth-sigma-m={jsig}")
            cmd.append(f"--junction-max-width-m={float(getattr(cfg, 'river_skeleton_junction_max_width_m', 300.0))}")

        # Curvature-driven asymmetry
        amode = str(getattr(cfg, "river_skeleton_asymmetry_mode", "none")).strip().lower()
        if amode and amode != "none":
            cmd.append(f"--asymmetry-mode={amode}")
            cmd.append(f"--asymmetry-strength={float(getattr(cfg, 'river_skeleton_asymmetry_strength', 0.25))}")
            cmd.append(f"--asymmetry-curv-ref={float(getattr(cfg, 'river_skeleton_asymmetry_curv_ref', 0.002))}")
            cmd.append(f"--asymmetry-max-shift={float(getattr(cfg, 'river_skeleton_asymmetry_max_shift', 0.20))}")
            cmd.append(f"--asymmetry-min-width-m={float(getattr(cfg, 'river_skeleton_asymmetry_min_width_m', 10.0))}")
            cmd.append(f"--asymmetry-min-curv={float(getattr(cfg, 'river_skeleton_asymmetry_min_curv', 0.0005))}")
            cmd.append(f"--asymmetry-densify-step-m={float(getattr(cfg, 'river_skeleton_asymmetry_densify_step_m', 20.0))}")
        _append_river_soundings_args(cmd, cfg, include_calib_args=False, include_mode_args=True)


        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["skeleton_full"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not Path(bed_skel_tif).exists():
            log.error("[RIVER] Skeleton bathymetry failed (hybrid).")
            report["river"]["status"] = "failed"
            return None

        # Step 3d: Combine
        log.info("[RIVER] Step 3d: Combining XS(mainstem) + skeleton(full) into cached bed raster...")
        bed_tif = cached_bed_tif
        nod = float(getattr(cfg, 'river_nodata', -9999.0))
        if bed_xs_tif is not None and Path(mainstem_mask_tif).exists():
            _combine_mainstem_xs_and_skeleton(Path(bed_xs_tif), Path(bed_skel_tif), Path(mainstem_mask_tif), Path(bed_tif), Path(cfg.river_dem), nod)
        else:
            import shutil
            shutil.copy2(str(bed_skel_tif), str(bed_tif))

        report.setdefault('river', {}).setdefault('outputs', {})['bed_elev_xs_mainstem'] = str(bed_xs_tif) if bed_xs_tif is not None else None
        report.setdefault('river', {}).setdefault('outputs', {})['bed_elev_skeleton_full'] = str(bed_skel_tif)
        report.setdefault('river', {}).setdefault('outputs', {})['mainstem_mask'] = str(mainstem_mask_tif) if Path(mainstem_mask_tif).exists() else None

    if river_method == "hybrid":
        # HYBRID handled above (bed_tif already built)
        pass
    elif river_method == "skeleton":
        # Step 2: Build a river channel mask (river vs. open water) from NHD flowlines + waffles water mask.
        log.info("[RIVER] Step 2: Building river channel mask (skeleton method)...")
        channel_mask_tif = work_dir / "river_channel_mask.tif"
        open_water_mask_tif = work_dir / "open_water_mask.tif"

        # Waffles-derived masks:
        #   1) ocean-only (want_nhd=False) always attempted first to prevent ocean bleed (resilient if TNM is flaky).
        #   2) with-NHD (want_nhd=True) attempted second; if it fails we still proceed using corridor+ArcGIS flowlines.
        cache_masks = Path(cfg.cache_root) / "masks"
        # Buffer AOI slightly for waffles so the mask fully covers the corridor near edges.
        aoi_buf = _buffer_aoi(str(cfg.aoi), float(getattr(cfg, 'waffles_aoi_buffer_deg', 0.0145)))

        ocean_mask = None
        with_nhd_mask = None
        try:
            ocean_mask = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks,
                aoi=aoi_buf,
                inc_arcsec=1.0,
                want_nhd=False,
                want_lakes=False,
                prefix="waffles_coastline_ocean_only",
                log=log,
            )
        except Exception as e:
            log.warning(f"[WAFFLES] Ocean-only mask unavailable; ocean bleed protection degraded: {e}")
            ocean_mask = None

        try:
            with_nhd_mask = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks,
                aoi=aoi_buf,
                inc_arcsec=1.0,
                want_nhd=True,
                want_lakes=False,
                prefix="waffles_coastline_with_nhd",
                log=log,
            )
        except Exception as e:
            log.warning(f"[WAFFLES] With-NHD mask unavailable (TNM flaky?): {e}. Proceeding with corridor+ArcGIS NHD flowlines + ocean-only mask.")
            with_nhd_mask = None

        cmd = [
            sys.executable, "river_domain_mask.py",
            f"--river-gpkg={network_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-channel-mask={channel_mask_tif}",
            f"--out-open-water-mask={open_water_mask_tif}",
            f"--channel-buffer-m={getattr(cfg, 'river_channel_buffer_m', 400.0)}",
            f"--max-channel-width-m={getattr(cfg, 'river_max_channel_width_m', 600.0)}",
            f"--mainstem-min-order={getattr(cfg, 'river_mainstem_min_order', 5)}",
            f"--max-mainstem-width-m={getattr(cfg, 'river_max_mainstem_width_m', 2500.0)}",
        ]

        # Channel domain source policy
        # - auto: prefer NHDArea river polygons when usable, else fall back to corridor
        # - nhdarea: require river polygons (exclude lakes)
        # - corridor: buffered flowline corridor only
        chan_src = str(getattr(cfg, 'river_channel_source', 'auto') or 'auto').strip().lower()
        if chan_src not in ('auto', 'nhdarea', 'corridor'):
            log.warning(f"[RIVER] Unknown river_channel_source='{chan_src}', defaulting to 'auto'.")
            chan_src = 'auto'
        cmd.append(f"--channel-source={chan_src}")

        # NHDArea filtering: keep Stream/River polygons only (exclude lakes/reservoirs).
        # Default is conservative: FType=460 (Stream/River). Users can override via config.
        nhd_allow = getattr(cfg, 'river_nhdarea_allow_ftype', None)
        if nhd_allow is None:
            nhd_allow = "460"
        cmd.append(f"--nhdarea-allow-ftype={nhd_allow}")
        nhd_allow_fcode = getattr(cfg, 'river_nhdarea_allow_fcode', None)
        if nhd_allow_fcode:
            cmd.append(f"--nhdarea-allow-fcode={nhd_allow_fcode}")

        # Optional: NHDArea constraint (if river_network.py wrote polygons into the gpkg).
        # NOTE: river_domain_mask filters NHDArea to river/stream polygons (excluding lakes).
        # Only pass NHDArea inputs when using auto/nhdarea mode.
        if (chan_src in ('auto', 'nhdarea')) and bool(getattr(cfg, "river_use_nhdarea", True)):
            cmd.append(f"--nhdarea-gpkg={network_gpkg}")
            cmd.append(f"--nhdarea-layer={getattr(cfg, 'river_nhdarea_layer', 'nhdarea_clip')}")
        if ocean_mask and Path(ocean_mask).exists():
            cmd.append(f"--ocean-mask={ocean_mask}")
        oke = float(getattr(cfg, 'river_ocean_keep_dist_m', 0.0) or 0.0)
        if (oke > 0.0):
            cmd.append(f"--ocean-keep-dist-m={oke}")

        if with_nhd_mask and Path(with_nhd_mask).exists():
            cmd.append(f"--water-mask={with_nhd_mask}")
        if getattr(cfg, "river_save_skeleton_debug", False):
            cmd.append("--write-debug")

        # For reporting: prefer the more inclusive water mask (with_nhd) if it exists,
        # otherwise fall back to the ocean-only mask (still useful to prevent ocean bleed).
        wm = None
        if with_nhd_mask and Path(with_nhd_mask).exists():
            wm = with_nhd_mask
        elif ocean_mask and Path(ocean_mask).exists():
            wm = ocean_mask

        # Persist the waffles mask used (for downstream final clipping/QA).
        try:
            report.setdefault("river", {}).setdefault("outputs", {})["waffles_water_mask"] = str(wm) if wm else None
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["domain_mask"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
            "waffles_mask": str(wm) if wm else None,
        }
        if rc != 0 or (not channel_mask_tif.exists()):
            log.error("[RIVER] Failed to build river channel mask.")
            report["river"]["status"] = "failed"
            return None

        log.info(f"[RIVER] Channel mask built: {channel_mask_tif}")

        try:
            report.setdefault("river", {}).setdefault("outputs", {})["river_channel_mask"] = str(channel_mask_tif)
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


        # Stash river domain/channel mask for fusion: inside this mask, river should override SDB to avoid tile seams
        try:
            cfg.river_domain_mask_for_fusion = Path(channel_mask_tif)
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


        # Step 3: Skeleton bathymetry (distance-transform, no cross-sections)
        log.info("[RIVER] Step 3: Inferring bathymetry (channel skeleton)...")
        bed_tif = cached_bed_tif

        cmd = [
            sys.executable, "river_skeleton_bathy.py",
            f"--river-gpkg={network_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--dem={cfg.river_dem}",
            f"--channel-mask={channel_mask_tif}",
            f"--out-bed={bed_tif}",
            f"--shape-exp={getattr(cfg, 'river_shape_exp', 0.5)}",
            f"--dmax-min-m={getattr(cfg, 'river_dmax_min_m', 0.5)}",
            f"--dmax-max-m={getattr(cfg, 'river_dmax_max_m', 30.0)}",
            f"--bed-profile-max-slope={float(getattr(cfg,'river_bed_profile_max_slope',0.0) or 0.0)}",
            f"--bed-profile-max-curv={float(getattr(cfg,'river_bed_profile_max_curv',0.0) or 0.0)}",
            f"--bed-profile-step-m={float(getattr(cfg,'river_bed_profile_step_m',25.0) or 25.0)}",
            f"--bed-profile-strength={float(getattr(cfg,'river_bed_profile_strength',0.6) or 0.6)}",
            f"--bed-profile-power={float(getattr(cfg,'river_bed_profile_power',2.0) or 2.0)}",
            f"--prior-mode={cfg.river_prior_mode}",
            f"--mv-a0={cfg.river_mv_a0}",
            f"--mv-bw={cfg.river_mv_bw}",
            f"--mv-ba={cfg.river_mv_ba}",
            f"--mv-bs={cfg.river_mv_bs}",
            f"--mv-eps-a={cfg.river_mv_eps_a}",
            f"--mv-eps-s={cfg.river_mv_eps_s}",
            f"--residual-blend-sigma-m={float(getattr(cfg, 'river_residual_blend_sigma_m', 120.0))}",
            f"--authoritative-bed-max-dist-m={float(getattr(cfg, 'river_authoritative_bed_max_dist_m', 2000.0))}",
        ]

        # WSE proxy controls (bank-derived WSE is usually more robust than in-channel DEM sampling)
        cmd.append(f"--wse-mode={getattr(cfg, 'river_skeleton_wse_mode', 'bank')}")
        wse_sig = float(getattr(cfg, 'river_skeleton_wse_smooth_sigma_m', 0.0) or 0.0)
        if wse_sig > 0.0:
            cmd.append(f"--wse-smooth-sigma-m={wse_sig}")
        if str(getattr(cfg, "river_skeleton_wse_mode", "bank")).strip().lower() == "bank_profile":
            cmd.append(f"--wse-profile-step-m={float(getattr(cfg, 'river_skeleton_wse_profile_step_m', 20.0) or 20.0)}")
            cmd.append(f"--wse-profile-resample-m={float(getattr(cfg, 'river_skeleton_wse_profile_resample_m', 20.0) or 20.0)}")
            cmd.append(f"--wse-profile-smooth-sigma-m={float(getattr(cfg, 'river_skeleton_wse_profile_smooth_sigma_m', 200.0) or 0.0)}")
            cmd.append(f"--wse-profile-max-slope={float(getattr(cfg, 'river_skeleton_wse_profile_max_slope', 0.005) or 0.0)}")
            cmd.append(f"--wse-profile-min-samples={int(getattr(cfg, 'river_skeleton_wse_profile_min_samples', 10) or 10)}")
            cmd.append(f"--wse-profile-max-query-dist-m={float(getattr(cfg, 'river_skeleton_wse_profile_max_query_dist_m', 250.0) or 0.0)}")
            # Optional: SWOT RiverSP anchoring (vector WSE observations)
            if getattr(cfg, 'river_swot_riversp', None):
                for fp in getattr(cfg, 'river_swot_riversp'):
                    cmd.append(f"--swot-riversp={fp}")
                if getattr(cfg, 'river_swot_wse_field', None):
                    cmd.append(f"--swot-wse-field={getattr(cfg, 'river_swot_wse_field')}")
                if getattr(cfg, 'river_swot_qual_field', None):
                    cmd.append(f"--swot-qual-field={getattr(cfg, 'river_swot_qual_field')}")
                cmd.append(f"--swot-max-dist-m={float(getattr(cfg, 'river_swot_max_dist_m', 300.0) or 0.0)}")
                cmd.append(f"--swot-min-samples={int(getattr(cfg, 'river_swot_min_samples', 5) or 0)}")
                cmd.append(f"--swot-correct-sigma-m={float(getattr(cfg, 'river_swot_correct_sigma_m', 2000.0) or 0.0)}")
                cmd.append(f"--swot-weight={float(getattr(cfg, 'river_swot_weight', 1.0) or 0.0)}")
                cmd.append(f"--swot-max-correction-m={float(getattr(cfg, 'river_swot_max_correction_m', 5.0) or 0.0)}")
                cmd.append(f"--swot-wse-offset-m={float(getattr(cfg, 'river_swot_wse_offset_m', 0.0) or 0.0)}")

                cmd.append(f"--swot-offset-mode={str(getattr(cfg, 'river_swot_offset_mode', 'median_mad') or 'median_mad')}")
                cmd.append(f"--swot-offset-min-samples={int(getattr(cfg, 'river_swot_offset_min_samples', 25) or 0)}")
                cmd.append(f"--swot-offset-mad-z={float(getattr(cfg, 'river_swot_offset_mad_z', 3.5) or 3.5)}")
                cmd.append(f"--swot-offset-max-abs-m={float(getattr(cfg, 'river_swot_offset_max_abs_m', 10.0) or 0.0)}")

        # Junction / confluence handling
        jmode = str(getattr(cfg, "river_skeleton_junction_mode", "smooth")).strip().lower()
        if jmode and jmode != "none":
            cmd.append(f"--junction-mode={jmode}")
            cmd.append(f"--junction-buffer-m={float(getattr(cfg, 'river_skeleton_junction_buffer_m', 120.0))}")
            cmd.append(f"--junction-degree-min={int(getattr(cfg, 'river_skeleton_junction_degree_min', 3))}")
            jsig = float(getattr(cfg, "river_skeleton_junction_smooth_sigma_m", 80.0) or 0.0)
            if jsig > 0.0:
                cmd.append(f"--junction-smooth-sigma-m={jsig}")
            cmd.append(f"--junction-max-width-m={float(getattr(cfg, 'river_skeleton_junction_max_width_m', 300.0))}")

        # Curvature-driven asymmetry (outer-bank deeper in bends)
        amode = str(getattr(cfg, "river_skeleton_asymmetry_mode", "none")).strip().lower()
        if amode and amode != "none":
            cmd.append(f"--asymmetry-mode={amode}")
            cmd.append(f"--asymmetry-strength={float(getattr(cfg, 'river_skeleton_asymmetry_strength', 0.25))}")
            cmd.append(f"--asymmetry-curv-ref={float(getattr(cfg, 'river_skeleton_asymmetry_curv_ref', 0.002))}")
            cmd.append(f"--asymmetry-max-shift={float(getattr(cfg, 'river_skeleton_asymmetry_max_shift', 0.20))}")
            cmd.append(f"--asymmetry-min-width-m={float(getattr(cfg, 'river_skeleton_asymmetry_min_width_m', 10.0))}")
            cmd.append(f"--asymmetry-min-curv={float(getattr(cfg, 'river_skeleton_asymmetry_min_curv', 0.0005))}")
            cmd.append(f"--asymmetry-densify-step-m={float(getattr(cfg, 'river_skeleton_asymmetry_densify_step_m', 20.0))}")
        if getattr(cfg, "river_save_skeleton_debug", False):
            cmd.append(f"--debug-dir={work_dir / 'skeleton_debug'}")

        # Optional: use external soundings (extra XYZ) to refine the skeleton prior and enforce depth anchors
        if getattr(cfg, 'river_soundings', None):
            snd_list = [s.strip() for s in str(cfg.river_soundings).split(',') if s.strip()]
            if snd_list:
                if getattr(cfg, 'extra_xyz_crs', None):
                    cmd.append(f"--soundings-crs={cfg.extra_xyz_crs}")
                cmd.append(f"--soundings-mode={getattr(cfg, 'river_soundings_mode', 'auto')}")
                cmd.append(f"--soundings-max-dist-m={float(getattr(cfg, 'river_soundings_max_dist_m', 1500.0))}")
                cmd.append(f"--soundings-min-r={float(getattr(cfg, 'river_soundings_min_r', 0.25))}")
                if not bool(getattr(cfg, 'river_soundings_enforce', True)):
                    cmd.append('--no-soundings-enforce')
                cmd.extend(['--soundings'] + snd_list)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["skeleton"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not Path(bed_tif).exists():
            log.error("[RIVER] Skeleton bathymetry failed.")
            report["river"]["status"] = "failed"
            return None

        log.info(f"[RIVER] Skeleton bathymetry raster: {bed_tif}")

        # Optional: constrain river outputs to NHDArea polygons (best-effort).
        # NOTE: This must happen AFTER the bed raster exists.
        if bool(getattr(cfg, "river_use_nhdarea", True)):
            try:
                applied = _mask_raster_to_nhdarea(
                    bed_tif,
                    network_gpkg,
                    nhd_layer=getattr(cfg, "river_nhdarea_layer", "nhdarea_clip"),
                    nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                )
                report.setdefault("river", {}).setdefault("masking", {})["nhdarea_bed_masked"] = bool(applied)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        # Final safety: ensure river bed raster is nodata outside the river channel mask.
        try:
            if Path(channel_mask_tif).exists():
                ok = _clip_raster_to_mask(
                    Path(bed_tif),
                    Path(channel_mask_tif),
                    inside_value=1,
                    invert=False,
                    nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                )
                report.setdefault("river", {}).setdefault("masking", {})["channel_mask_clip"] = bool(ok)
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    else:
        # Step 2: Generating cross-sections...
        log.info("[RIVER] Step 2: Generating cross-sections...")
        xs_gpkg = work_dir / "cross_sections.gpkg"

        # SECURITY FIX: Use list-based command construction
        cmd = [
            sys.executable, "xs_builder.py",
            f"--river-gpkg={network_gpkg}",
            f"--dem={cfg.river_dem}",
            f"--out-gpkg={xs_gpkg}",
            f"--spacing-m={cfg.xs_spacing_m}",
            f"--half-width-m={cfg.xs_length_m / 2.0}",
            f"--smoothing-window-m={getattr(cfg, 'xs_smoothing_window_m', 0.0)}",
            f"--deconflict-tol-m={getattr(cfg, 'xs_deconflict_tol_m', 2.0)}",
            f"--junction-snap-m={getattr(cfg, 'xs_junction_snap_m', 30.0)}",
            f"--junction-buffer-m={getattr(cfg, 'xs_junction_buffer_m', 120.0)}",
            f"--densify-step-m={getattr(cfg, 'xs_densify_step_m', 20.0)}",
        ]
        if not bool(getattr(cfg, "xs_trim_overlaps", True)):
            cmd.append("--no-trim-overlaps")
        if not bool(getattr(cfg, "xs_global_deconflict", True)):
            cmd.append("--no-global-deconflict")
        if not bool(getattr(cfg, "xs_skip_junctions", True)):
            cmd.append("--no-skip-junctions")


        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["xs_builder"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not xs_gpkg.exists():
            log.error("[RIVER] Failed to build cross-sections.")
            report["river"]["status"] = "failed"
            return None

        log.info(f"[RIVER] Cross-sections generated: {xs_gpkg}")
        # Step 2b: Build river channel domain mask (so final river raster is river-only)
        channel_mask_tif = work_dir / "river_channel_mask.tif"
        open_water_mask_tif = work_dir / "open_water_mask.tif"

        # Reuse the same domain-mask logic used by the skeleton method for consistency.
        # This ensures river outputs cannot bleed into ocean/lakes when using XS method.
        cache_masks = Path(cfg.cache_root) / "masks"
        aoi_buf = _buffer_aoi(str(cfg.aoi), float(getattr(cfg, 'waffles_aoi_buffer_deg', 0.0145)))

        ocean_mask = None
        with_nhd_mask = None
        try:
            ocean_mask = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks,
                aoi=aoi_buf,
                inc_arcsec=1.0,
                want_nhd=False,
                want_lakes=False,
                prefix="waffles_coastline_ocean_only",
                log=log,
            )
        except Exception as e:
            log.warning(f"[WAFFLES] Ocean-only mask unavailable; ocean bleed protection degraded: {e}")
            ocean_mask = None

        try:
            with_nhd_mask = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks,
                aoi=aoi_buf,
                inc_arcsec=1.0,
                want_nhd=True,
                want_lakes=False,
                prefix="waffles_coastline_with_nhd",
                log=log,
            )
        except Exception as e:
            log.warning(f"[WAFFLES] With-NHD mask unavailable (TNM flaky?): {e}. Proceeding with corridor+ArcGIS NHD flowlines + ocean-only mask.")
            with_nhd_mask = None

        cmd = [
            sys.executable, "river_domain_mask.py",
            f"--river-gpkg={network_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-channel-mask={channel_mask_tif}",
            f"--out-open-water-mask={open_water_mask_tif}",
            f"--channel-buffer-m={getattr(cfg, 'river_channel_buffer_m', 400.0)}",
            f"--max-channel-width-m={getattr(cfg, 'river_max_channel_width_m', 600.0)}",
            f"--mainstem-min-order={getattr(cfg, 'river_mainstem_min_order', 5)}",
            f"--max-mainstem-width-m={getattr(cfg, 'river_max_mainstem_width_m', 2500.0)}",
        ]

        chan_src = str(getattr(cfg, 'river_channel_source', 'auto') or 'auto').strip().lower()
        if chan_src not in ('auto', 'nhdarea', 'corridor'):
            log.warning(f"[RIVER] Unknown river_channel_source='{chan_src}', defaulting to 'auto'.")
            chan_src = 'auto'
        cmd.append(f"--channel-source={chan_src}")

        nhd_allow = getattr(cfg, 'river_nhdarea_allow_ftype', None)
        if nhd_allow is None:
            nhd_allow = "460"
        cmd.append(f"--nhdarea-allow-ftype={nhd_allow}")
        nhd_allow_fcode = getattr(cfg, 'river_nhdarea_allow_fcode', None)
        if nhd_allow_fcode:
            cmd.append(f"--nhdarea-allow-fcode={nhd_allow_fcode}")

        if (chan_src in ('auto', 'nhdarea')) and bool(getattr(cfg, "river_use_nhdarea", True)):
            cmd.append(f"--nhdarea-gpkg={network_gpkg}")
            cmd.append(f"--nhdarea-layer={getattr(cfg, 'river_nhdarea_layer', 'nhdarea_clip')}")

        if ocean_mask and Path(ocean_mask).exists():
            cmd.append(f"--ocean-mask={ocean_mask}")
        oke = float(getattr(cfg, 'river_ocean_keep_dist_m', 0.0) or 0.0)
        if (oke > 0.0):
            cmd.append(f"--ocean-keep-dist-m={oke}")

        if with_nhd_mask and Path(with_nhd_mask).exists():
            cmd.append(f"--water-mask={with_nhd_mask}")

        # Persist the waffles mask used (for downstream final clipping/QA).
        try:
            wm = None
            if with_nhd_mask and Path(with_nhd_mask).exists():
                wm = with_nhd_mask
            elif ocean_mask and Path(ocean_mask).exists():
                wm = ocean_mask
            report.setdefault("river", {}).setdefault("outputs", {})["waffles_water_mask"] = str(wm) if wm else None
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["domain_mask"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or (not channel_mask_tif.exists()):
            log.warning("[RIVER] Failed to build channel mask for XS method; continuing without hard domain constraint.")
        else:
            try:
                cfg.river_domain_mask_for_fusion = Path(channel_mask_tif)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


        # Step 3: infer bathymetry (raster + optional GPKG)
        log.info("[RIVER] Step 3: Inferring bathymetry...")
        bed_tif = cached_bed_tif
        bathy_gpkg = work_dir / "river_bathy.gpkg"

        # SECURITY FIX: Use list-based command construction
        cmd = [
            sys.executable, "xs_infer_bathy_raster.py",
            f"--xs-gpkg={xs_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-gpkg={bathy_gpkg}",
            f"--out-bathy-raster={bed_tif}",
            f"--continuous={getattr(cfg, 'river_continuous', 'walid_aniso')}",
            f"--continuous-k={getattr(cfg, 'river_continuous_k', 12)}",
            f"--idw-power={getattr(cfg, 'river_idw_power', 2.0)}",
            f"--aniso-along-scale-m={getattr(cfg, 'river_aniso_along_scale_m', 500.0)}",
            f"--aniso-cross-scale-m={getattr(cfg, 'river_aniso_cross_scale_m', 30.0)}",
            f"--thalweg-weight={getattr(cfg, 'river_thalweg_weight', 6.0)}",
            f"--nodata={getattr(cfg, 'river_nodata', -9999.0)}",
            f"--overlap-reducer={getattr(cfg, 'river_overlap_reducer', 'min')}",
        ]
        # Hard constrain interpolation/output to the river domain mask when available.
        if 'channel_mask_tif' in locals() and channel_mask_tif is not None and Path(channel_mask_tif).exists():
            cmd.append(f"--channel-mask-raster={channel_mask_tif}")
            cmd.append("--channel-mask-inside-value=1")
        cmd.append(f"--river-gpkg={network_gpkg}")
        cmd.append(f"--prior-mode={cfg.river_prior_mode}")
        cmd.append(f"--mv-a0={cfg.river_mv_a0}")
        cmd.append(f"--mv-bw={cfg.river_mv_bw}")
        cmd.append(f"--mv-ba={cfg.river_mv_ba}")
        cmd.append(f"--mv-bs={cfg.river_mv_bs}")
        cmd.append(f"--mv-eps-a={cfg.river_mv_eps_a}")
        cmd.append(f"--mv-eps-s={cfg.river_mv_eps_s}")
        cmd.append(f"--slope-proxy-window={cfg.river_slope_proxy_window}")
        cmd.append(f"--slope-min={cfg.river_slope_min}")
        cmd.append(f"--slope-max={cfg.river_slope_max}")
        cmd.append(f"--slope-proxy-min-n={cfg.river_slope_proxy_min_n}")

        # Longitudinal WSE profile fit (preferred slope proxy)
        if bool(getattr(cfg, "river_wse_profile_enabled", True)):
            cmd.append("--wse-profile-enabled")
        else:
            cmd.append("--no-wse-profile")
        cmd.append(f"--wse-profile-window={getattr(cfg, 'river_wse_profile_window', cfg.river_slope_proxy_window)}")
        cmd.append(f"--wse-profile-min-n={getattr(cfg, 'river_wse_profile_min_n', cfg.river_slope_proxy_min_n)}")
        if bool(getattr(cfg, "river_wse_profile_monotonic", True)):
            cmd.append("--wse-profile-monotonic")
        else:
            cmd.append("--no-wse-profile-monotonic")

        # Option A: USGS discharge *measurement* anchors
        if cfg.river_usgs_sites:
            cmd.append(f"--usgs-sites={cfg.river_usgs_sites}")
            if cfg.river_usgs_start:
                cmd.append(f"--usgs-start={cfg.river_usgs_start}")
            if cfg.river_usgs_end:
                cmd.append(f"--usgs-end={cfg.river_usgs_end}")
            if cfg.river_usgs_cache_dir:
                cmd.append(f"--usgs-cache-dir={cfg.river_usgs_cache_dir}")
            cmd.append(f"--usgs-max-dist-m={cfg.river_usgs_max_dist_m}")
            cmd.append(f"--usgs-mean-to-dmax={cfg.river_usgs_mean_to_dmax}")
            cmd.append(f"--usgs-a-stat={cfg.river_usgs_a_stat}")
            cmd.append(f"--usgs-q-quantile-lo={cfg.river_usgs_q_quantile_lo}")
            cmd.append(f"--usgs-q-quantile-hi={cfg.river_usgs_q_quantile_hi}")
            cmd.append(f"--usgs-a-cv-warn={cfg.river_usgs_a_cv_warn}")
            cmd.append(f"--usgs-width-ratio-max={cfg.river_usgs_width_ratio_max}")
            cmd.append(f"--gage-snap-max-dist-m={cfg.river_gage_snap_max_dist_m}")
            if cfg.river_usgs_width_ratio_blend:
                cmd.append("--usgs-width-ratio-blend")
            else:
                cmd.append("--no-usgs-width-ratio-blend")

        # Optional: width-stage inversion anchors
        if cfg.river_width_stage_csv:
            cmd.append(f"--width-stage-csv={cfg.river_width_stage_csv}")
            cmd.append(f"--width-stage-max-dist-m={cfg.river_width_stage_max_dist_m}")
            cmd.append(f"--width-stage-min-n={cfg.river_width_stage_min_n}")
            cmd.append(f"--width-stage-min-r2={cfg.river_width_stage_min_r2}")
            cmd.append(f"--width-stage-max-weight={cfg.river_width_stage_max_weight}")


        # Optional: Manning inversion prior (blended)
        if getattr(cfg, "river_manning_mode", "off") != "off":
            cmd.append(f"--manning-mode={cfg.river_manning_mode}")
            if cfg.river_manning_q_cms is not None:
                cmd.append(f"--manning-q-cms={cfg.river_manning_q_cms}")
            if cfg.river_manning_q_field:
                cmd.append(f"--manning-q-field={cfg.river_manning_q_field}")
            cmd.append(f"--manning-n={cfg.river_manning_n}")
            cmd.append(f"--manning-region={getattr(cfg, 'river_manning_region', 'default')}")
            cmd.append(f"--manning-min-confidence={getattr(cfg, 'river_manning_min_confidence', 0.30)}")
            cmd.append(f"--manning-max-weight={cfg.river_manning_max_weight}")
            cmd.append(f"--manning-backwater-slope-thresh={cfg.river_manning_backwater_slope_thresh}")
            if cfg.river_manning_dist_to_mouth_field:
                cmd.append(f"--manning-dist-to-mouth-field={cfg.river_manning_dist_to_mouth_field}")
            cmd.append(f"--manning-dist-to-mouth-km-max={cfg.river_manning_dist_to_mouth_km_max}")



        # Optional: Regional hydraulic geometry curve prior
        if getattr(cfg, "river_regional_curve_enabled", False):
            cmd.append("--regional-curve-enabled")
            cmd.append(f"--regional-curve-region={cfg.river_regional_curve_region}")
            if cfg.river_regional_curve_c is not None:
                cmd.append(f"--regional-curve-c={cfg.river_regional_curve_c}")
            if cfg.river_regional_curve_f is not None:
                cmd.append(f"--regional-curve-f={cfg.river_regional_curve_f}")
            cmd.append(f"--regional-curve-da-units={cfg.river_regional_curve_da_units}")
            cmd.append(f"--regional-curve-depth-units={cfg.river_regional_curve_depth_units}")
            cmd.append(f"--regional-curve-depth-type={cfg.river_regional_curve_depth_type}")
            cmd.append(f"--regional-curve-to-dmax={cfg.river_regional_curve_to_dmax}")
            cmd.append(f"--regional-curve-to-dmax-factor={cfg.river_regional_curve_to_dmax_factor}")
            cmd.append(f"--regional-curve-unc-pct={cfg.river_regional_curve_unc_pct}")
            cmd.append(f"--regional-curve-max-weight={cfg.river_regional_curve_max_weight}")
            cmd.append(f"--regional-curve-min-da-km2={cfg.river_regional_curve_min_da_km2}")
        if getattr(cfg, "river_continuous_buffer_m", None) is not None:
            cmd.append(f"--continuous-buffer-m={cfg.river_continuous_buffer_m}")
        _append_river_soundings_args(cmd, cfg, include_calib_args=False, include_mode_args=True)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["infer_raster"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not bed_tif.exists():
            # Include useful diagnostics inline (command + stderr/stdout tail), so users don't have to open the JSON report.
            log.error("[RIVER] Failed to infer river bed patch raster (rc=%s, exists=%s).", rc, bed_tif.exists())
            log.error("[RIVER] Command: %s", cmd_str)
            if err:
                log.error("[RIVER] stderr_tail:\n%s", err[-4000:])
            if out:
                log.error("[RIVER] stdout_tail:\n%s", out[-4000:])
            report["river"]["status"] = "failed"
            return None

        
    report["river"]["status"] = "success"
    log.info(f"[RIVER] Success: {bed_tif}")
    # Materialize into run output folder for convenience
    # Compute depth relative to DEM terrain surface (negative down): depth = bed_elev - dem
    try:
        compute_depth_from_bed_and_dem(bed_tif, Path(cfg.river_dem), cached_depth_tif, depth_sign="negative_down")
    except Exception as e:
        log.error("[RIVER] Failed to compute depth from bed elevation and DEM: %s", e)
        report["river"]["status"] = "failed"
        return None

    # Final hard guarantee: river outputs (bed + depth) must be nodata outside the river channel domain.
    try:
        _maskp = None
        if getattr(cfg, "river_domain_mask_for_fusion", None):
            mp = Path(getattr(cfg, "river_domain_mask_for_fusion"))
            if mp.exists():
                _maskp = mp
        if _maskp is None and 'channel_mask_tif' in locals():
            try:
                mp2 = Path(channel_mask_tif)
                if mp2.exists():
                    _maskp = mp2
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        if _maskp is not None:
            nval = float(getattr(cfg, "river_nodata", -9999.0))
            ok_bed = _clip_raster_to_mask(Path(bed_tif), _maskp, inside_value=1, invert=False, nodata=nval)
            ok_dep = _clip_raster_to_mask(Path(cached_depth_tif), _maskp, inside_value=1, invert=False, nodata=nval)
            report.setdefault("river", {}).setdefault("masking", {})["channel_mask_clip_bed"] = bool(ok_bed)
            report.setdefault("river", {}).setdefault("masking", {})["channel_mask_clip_depth"] = bool(ok_dep)
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


    # Optional: mask cached river outputs to waffles coastline (if available)
    try:
        if cfg.mask_river_to_waffles:
            wm = _find_latest_waffles_mask(cfg.cache_root)
            if wm and wm.exists():
                md_ok = _mask_raster_to_waffles(cached_depth_tif, wm, nodata=-9999.0)
                mb_ok = _mask_raster_to_waffles(cached_bed_tif, wm, nodata=-9999.0)
                report.setdefault("river", {}).setdefault("masking", {})["waffles_mask"] = str(wm)
                report.setdefault("river", {}).setdefault("masking", {})["masked_depth"] = bool(md_ok)
                report.setdefault("river", {}).setdefault("masking", {})["masked_bed"] = bool(mb_ok)
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    out_depth = river_dir / "river_depth_terrain_patch.tif"
    out_bed = river_dir / "river_bottom_navd88_patch.tif"
    for src, dst in [(cached_depth_tif, out_depth), (bed_tif, out_bed)]:
        try:
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)
        except Exception:
            shutil.copy(src, dst)

    try:
        apply_depth_metadata(out_depth, depth_reference="terrain_surface")
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    try:
        apply_elevation_metadata(out_bed, vertical_datum="NAVD88")
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    report["river"]["outputs"] = {
        "depth_terrain": str(out_depth),
        "bottom_elevation": str(out_bed),
    }

    # Persist cache manifest after successful products
    try:
        manifest_path.write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    return out_depth
# -----------------------------------------------------------------------------
# Fusion
# -----------------------------------------------------------------------------

def fuse(cfg: BathyConfig, sdb_raster: Optional[Path], river_raster: Optional[Path], report: Dict[str, Any]) -> Optional[Path]:
    """
    Build the final combined bathymetry surface.

    Behavior:
    - If only one source exists, that raster becomes the final surface.
    - If both exist, use bathy_fusion.fuse_bathymetry() with strategy="weighted_overlap"
      and C1 weights (0.70 priority / 0.30 secondary) in overlap pixels, plus gap-filling.

    Notes:
    - This function writes outputs under out_dir/combined/.
    - It records fusion decisions and errors into report["fusion"].
    """
    combined_dir = ensure_dir(cfg.out_dir / "combined")
    out_depth = combined_dir / "bathy_combined_depth.tif"
    out_prov = combined_dir / "bathy_combined_provenance.tif"

    # No sources
    if sdb_raster is None and river_raster is None:
        report["fusion"] = {"status": "failed", "reason": "no sources"}
        return None

    # Only one source (fast path)
    if sdb_raster is None and river_raster is not None:
        shutil.copy2(str(river_raster), str(out_depth))
        report["fusion"] = {"status": "success", "mode": "river_only", "outputs": {"depth": str(out_depth)}}
        return out_depth

    if river_raster is None and sdb_raster is not None:
        shutil.copy2(str(sdb_raster), str(out_depth))
        report["fusion"] = {"status": "success", "mode": "sdb_only", "outputs": {"depth": str(out_depth)}}
        return out_depth

    # Both sources available → weighted overlap fusion (C1)
    pri = (cfg.priority or "sdb").strip().lower()
    if pri not in ("sdb", "river"):
        pri = "sdb"
    other = "river" if pri == "sdb" else "sdb"

    # Template grid: use the priority raster if possible
    template = Path(sdb_raster) if pri == "sdb" else Path(river_raster)


    # ------------------------------------------------------------------
    # Sanitize inputs for fusion:
    # SDB can contain large regions of literal 0.0 on land/no-prediction.
    # That blocks "gap filling" with river because fusion treats 0 as valid.
    # Use the SDB land mask (land=1, water=0) to force those cells to nodata.
    # ------------------------------------------------------------------
    sdb_fuse_path = Path(sdb_raster)
    river_fuse_path = Path(river_raster)

    try:
        lm = find_sdb_land_mask(Path(cfg.out_dir) / "sdb")
        if lm and Path(lm).exists():
            import rasterio
            import numpy as np
            from rasterio.warp import reproject, Resampling

            sanitized = combined_dir / "sdb_for_fusion_sanitized.tif"
            with rasterio.open(str(sdb_fuse_path)) as ds:
                prof = ds.profile.copy()
                prof.update(dtype="float32", nodata=-9999.0, compress="deflate", count=1)
                sdb_arr = ds.read(1).astype("float32")
                sdb_nodata = ds.nodata

                # warp land mask onto SDB grid
                land = np.zeros((ds.height, ds.width), dtype="uint8")
                with rasterio.open(str(lm)) as lm_ds:
                    reproject(
                        source=rasterio.band(lm_ds, 1),
                        destination=land,
                        src_transform=lm_ds.transform,
                        src_crs=lm_ds.crs,
                        dst_transform=ds.transform,
                        dst_crs=ds.crs,
                        resampling=Resampling.nearest,
                        src_nodata=lm_ds.nodata,
                        dst_nodata=0,
                    )

                invalid = (land == 1)
                # existing nodata also invalid
                if sdb_nodata is not None:
                    invalid |= (sdb_arr == sdb_nodata)
                invalid |= ~np.isfinite(sdb_arr)

                # Critical: treat 0.0 on land as invalid (typical mask artifact)
                invalid |= ((sdb_arr == 0.0) & (land == 1))

                sdb_arr2 = sdb_arr.copy()
                sdb_arr2[invalid] = prof["nodata"]

                with rasterio.open(str(sanitized), "w", **prof) as dst:
                    dst.write(sdb_arr2.astype("float32"), 1)

            sdb_fuse_path = sanitized
            report.setdefault("fusion", {}).setdefault("inputs_sanitized", {})["sdb_landmask_applied"] = str(lm)
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    # Heuristic: if the SDB raster is overwhelmingly literal 0.0 (common failure mode),
    # treat 0.0 as nodata for fusion so river can gap-fill.
    try:
        from osgeo import gdal
        import numpy as np
        ds0 = gdal.Open(str(sdb_fuse_path))
        if ds0 is not None:
            b0 = ds0.GetRasterBand(1)
            a0 = b0.ReadAsArray()
            nd0 = b0.GetNoDataValue()
            m = np.isfinite(a0)
            if nd0 is not None:
                m &= (a0 != nd0)
            if np.any(m):
                frac0 = float(np.mean(a0[m] == 0.0))
                # Only trigger when nearly all valid pixels are exactly 0
                if frac0 >= 0.95:
                    sanitized0 = combined_dir / "sdb_for_fusion_zeros_sanitized.tif"
                    a1 = a0.astype(np.float32)
                    nodata = -9999.0
                    a1[~np.isfinite(a1)] = nodata
                    if nd0 is not None:
                        a1[a1 == nd0] = nodata
                    a1[a1 == 0.0] = nodata
                    drv = gdal.GetDriverByName("GTiff")
                    out_ds = drv.Create(
                        str(sanitized0),
                        ds0.RasterXSize,
                        ds0.RasterYSize,
                        1,
                        gdal.GDT_Float32,
                        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
                    )
                    out_ds.SetGeoTransform(ds0.GetGeoTransform())
                    out_ds.SetProjection(ds0.GetProjection())
                    ob = out_ds.GetRasterBand(1)
                    ob.SetNoDataValue(nodata)
                    ob.WriteArray(a1)
                    ob.FlushCache()
                    out_ds.FlushCache()
                    out_ds = None
                    sdb_fuse_path = sanitized0
                    report.setdefault("fusion", {}).setdefault("inputs_sanitized", {})["sdb_zero_sanitized"] = {
                        "path": str(sanitized0),
                        "frac0": frac0,
                        "threshold": 0.95,
                    }
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


    def _simple_union_overlay(sdb_path: Path, river_path: Path, out_path: Path, template_path: Path, pri: str) -> None:
        """Guaranteed combine: fill gaps from the secondary raster into the primary on the template grid.
        This is used as a robustness fallback when weighted fusion fails or yields no usable overlap.
        """
        import rasterio
        import numpy as np
        from rasterio.warp import reproject, Resampling

        nodata = -9999.0

        def _read_align(src_path: Path):
            with rasterio.open(template_path) as tmpl:
                prof = tmpl.profile.copy()
                prof.update(dtype="float32", nodata=nodata, count=1, compress="deflate")
                arr = np.full((tmpl.height, tmpl.width), nodata, dtype=np.float32)
                with rasterio.open(src_path) as src:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=arr,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=tmpl.transform,
                        dst_crs=tmpl.crs,
                        resampling=Resampling.nearest,
                        src_nodata=src.nodata,
                        dst_nodata=nodata,
                    )
            return arr, prof

        sdb_a, prof = _read_align(Path(sdb_path))
        riv_a, _ = _read_align(Path(river_path))

        # Enforce strict domain separation:
        #   - Inside river_domain_mask (mask==1): DO NOT allow SDB to contribute.
        #   - Outside river_domain_mask: DO NOT allow river to contribute (safety).
        # This prevents a common failure where the "combined" output looks like S2 imagery inside rivers.
        try:
            if river_domain_mask_for_fusion is not None and Path(river_domain_mask_for_fusion).exists():
                dm_a, _ = _read_align(Path(river_domain_mask_for_fusion))
                dm = (dm_a > 0.5)
                sdb_a = sdb_a.copy()
                riv_a = riv_a.copy()
                sdb_a[dm] = nodata
                riv_a[~dm] = nodata
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        if pri == "river":
            primary, secondary = riv_a, sdb_a
        else:
            primary, secondary = sdb_a, riv_a

        out = primary.copy()
        take = (out == nodata) & (secondary != nodata)
        out[take] = secondary[take]

        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(out.astype(np.float32), 1)


    def _gdal_union_overlay(sdb_path: Path, river_path: Path, out_path: Path, template_path: Path, pri: str) -> None:
        """GDAL-based union overlay fallback (no rasterio dependency).
        Aligns both rasters onto the template grid, then fills primary nodata with secondary values.
        """
        try:
            from osgeo import gdal
            import numpy as np
        except Exception as ee:
            raise RuntimeError(f"GDAL/NumPy not available for fallback fusion: {ee}")

        nodata = -9999.0

        tmpl = gdal.Open(str(template_path))
        if tmpl is None:
            raise RuntimeError(f"Failed to open template raster: {template_path}")

        gt = tmpl.GetGeoTransform()
        proj = tmpl.GetProjection()
        w = tmpl.RasterXSize
        h = tmpl.RasterYSize
        xres = gt[1]
        yres = abs(gt[5])
        xmin = gt[0]
        ymax = gt[3]
        xmax = xmin + xres * w
        ymin = ymax - yres * h

        warp_opts = gdal.WarpOptions(
            format="MEM",
            outputBounds=(xmin, ymin, xmax, ymax),
            xRes=xres,
            yRes=yres,
            dstSRS=proj if proj else None,
            resampleAlg="near",
            dstNodata=nodata,
            targetAlignedPixels=True,
            multithread=True,
        )

        def _warp_to_mem(p: Path):
            ds = gdal.Warp("", str(p), options=warp_opts)
            if ds is None:
                raise RuntimeError(f"gdal.Warp failed for {p}")
            return ds

        sdb_ds = _warp_to_mem(Path(sdb_path))
        riv_ds = _warp_to_mem(Path(river_path))

        sdb_a = sdb_ds.ReadAsArray().astype(np.float32)
        riv_a = riv_ds.ReadAsArray().astype(np.float32)

        # Optional domain enforcement (same policy as main fusion):
        # suppress SDB in river domain; suppress river outside river domain.
        try:
            _dm = river_domain_mask_for_fusion
            if _dm is not None and Path(_dm).exists():
                dm_ds = _warp_to_mem(Path(_dm))
                dm_a = dm_ds.ReadAsArray()
                dm = np.asarray(dm_a) > 0.5
                sdb_a = sdb_a.copy()
                riv_a = riv_a.copy()
                sdb_a[dm] = nodata
                riv_a[~dm] = nodata
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        # ensure finite
        sdb_a[~np.isfinite(sdb_a)] = nodata
        riv_a[~np.isfinite(riv_a)] = nodata

        if pri == "river":
            primary, secondary = riv_a, sdb_a
        else:
            primary, secondary = sdb_a, riv_a

        out = primary.copy()
        take = (out == nodata) & (secondary != nodata)
        out[take] = secondary[take]

        drv = gdal.GetDriverByName("GTiff")
        ds_out = drv.Create(
            str(out_path),
            w,
            h,
            1,
            gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
        )
        if ds_out is None:
            raise RuntimeError(f"Failed to create output raster: {out_path}")

        ds_out.SetGeoTransform(gt)
        if proj:
            ds_out.SetProjection(proj)
        band = ds_out.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.WriteArray(out)
        band.FlushCache()
        ds_out.FlushCache()
        ds_out = None
    try:
        from bathy_fusion import fuse_bathymetry, FusionConfig


        # ------------------------------------------------------------------
        # Domain enforcement (critical):
        #   - River bathymetry must be the ONLY contributor inside the river/channel domain.
        #   - SDB must not overwrite river inside that domain (common "looks like satellite" failure mode).
        # We satisfy this by passing a 1/0 river_domain_mask into bathy_fusion and enabling
        # river_overrides_sdb_in_domain, which suppresses SDB where mask==1.
        # ------------------------------------------------------------------
        river_domain_mask_for_fusion: Optional[Path] = None
        try:
            ro = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
            cand = ro.get("river_channel_mask") or ro.get("channel_mask") or ro.get("river_domain_mask")
            if cand:
                river_domain_mask_for_fusion = Path(str(cand))
        except Exception:
            river_domain_mask_for_fusion = None

        if river_domain_mask_for_fusion is None or (not river_domain_mask_for_fusion.exists()):
            try:
                river_domain_mask_for_fusion = _discover_river_channel_mask(cfg.out_dir, cfg.cache_root)
            except Exception:
                river_domain_mask_for_fusion = None

        if river_domain_mask_for_fusion is not None and river_domain_mask_for_fusion.exists():
            report.setdefault("fusion", {}).setdefault("domain", {})["river_domain_mask"] = str(river_domain_mask_for_fusion)
            report["fusion"]["domain"]["river_overrides_sdb_in_domain"] = True
        else:
            report.setdefault("fusion", {}).setdefault("domain", {})["river_domain_mask"] = None
            report["fusion"]["domain"]["river_overrides_sdb_in_domain"] = False
        cfg_fuse = FusionConfig(
            sdb_raster=Path(sdb_fuse_path) if sdb_fuse_path else None,
            river_raster=Path(river_fuse_path) if river_fuse_path else None,
            measured_raster=None,
            dem_raster=None,
            river_domain_mask=river_domain_mask_for_fusion,
            river_overrides_sdb_in_domain=bool(river_domain_mask_for_fusion is not None and river_domain_mask_for_fusion.exists()),
            out_dir=combined_dir,
            strategy=cfg.fusion_strategy,
            priority_order=[pri, other],
            primary_weight=float(cfg.fusion_primary_weight),
            secondary_weight=float(cfg.fusion_secondary_weight),
            nodata=-9999.0,
            template_raster=template,
            taper_m=float(cfg.fusion_taper_m),
        )

        res = fuse_bathymetry(cfg_fuse)

        if getattr(res, "status", None) != "success" or not getattr(res, "combined_raster", None):
            raise RuntimeError(getattr(res, "error", None) or "fusion did not return a valid combined raster")

        # bathy_fusion writes fixed filenames; copy to our pipeline-stable names
        try:
            shutil.copy2(str(res.combined_raster), str(out_depth))
        except Exception:
            # If copy fails, fall back to the produced path
            out_depth = Path(res.combined_raster)

        if getattr(res, "provenance_raster", None):
            try:
                shutil.copy2(str(res.provenance_raster), str(out_prov))
            except Exception:
                out_prov = Path(res.provenance_raster)

        report["fusion"] = {
            "status": "success",
            "mode": "weighted_overlap",
            "priority": pri,
            "weights": {"primary": float(cfg.fusion_primary_weight), "secondary": float(cfg.fusion_secondary_weight)},
            "outputs": {"depth": str(out_depth), "provenance": str(out_prov) if out_prov else None},
            "bathy_fusion_outputs": {
                "combined": str(getattr(res, "combined_raster", "")),
                "provenance": str(getattr(res, "provenance_raster", "")),
                "uncertainty": str(getattr(res, "uncertainty_raster", "")) if getattr(res, "uncertainty_raster", None) else None,
            },
        }


        # Sanity: ensure the combined raster actually incorporated some river pixels (common failure mode
        # when river was misaligned or fully nodata after reprojection). If not, do a guaranteed union overlay.
        try:
            import rasterio, numpy as np
            has_river = False
            with rasterio.open(str(out_prov if out_prov else out_depth)) as ds:
                arr = ds.read(1)
                # If provenance exists and contains river/blended, we consider river present.
                # Otherwise, fall back to checking if river contributes to combined where sdb is nodata.
            if out_prov and Path(out_prov).exists():
                with rasterio.open(str(out_prov)) as dp:
                    p = dp.read(1)
                    has_river = bool(np.any((p == 2) | (p == 6)))  # PROV_RIVER=2, PROV_BLENDED=6
            if not has_river:
                try:
                    _simple_union_overlay(Path(sdb_fuse_path), Path(river_fuse_path), Path(out_depth), Path(template), pri)
                except Exception:
                    _gdal_union_overlay(Path(sdb_fuse_path), Path(river_fuse_path), Path(out_depth), Path(template), pri)
                report["fusion"]["note"] = "River contribution missing after fusion; applied union overlay fallback."
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)



        # ------------------------------------------------------------------
        # Enforce authoritative XYZ constraints in the final fused raster.
        # This is a hard "burn-in": at pixels containing authoritative soundings,
        # the output bed elevation must equal the sounding bed elevation exactly.
        # ------------------------------------------------------------------
        try:
            import rasterio
            import numpy as np
            from rasterio.transform import rowcol

            xyz_str = str(getattr(cfg, 'river_soundings', '') or '').strip()
            xyz_paths = [p for p in [s.strip() for s in xyz_str.split(',')] if p]
            if xyz_paths and Path(out_depth).exists():
                with rasterio.open(str(out_depth), 'r+') as ds:
                    arr = ds.read(1).astype('float32')
                    nodata = ds.nodata
                    if nodata is None:
                        nodata = -9999.0
                    # Collect per-pixel samples (median if multiple points hit same pixel)
                    h, w = ds.height, ds.width
                    pix_idx = []
                    zvals = []
                    for xp in xyz_paths:
                        fp = Path(xp)
                        if not fp.exists():
                            continue
                        try:
                            dat = np.genfromtxt(str(fp), dtype='float64', delimiter=None)
                            if dat.ndim == 1:
                                dat = dat.reshape(1, -1)
                        except Exception:
                            # try comma-delimited
                            try:
                                dat = np.genfromtxt(str(fp), dtype='float64', delimiter=',')
                                if dat.ndim == 1:
                                    dat = dat.reshape(1, -1)
                            except Exception:
                                continue
                        if dat.size == 0:
                            continue
                        # take first 3 columns
                        if dat.shape[1] < 3:
                            continue
                        x = dat[:, 0]
                        y = dat[:, 1]
                        z = dat[:, 2]
                        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                        if not np.any(m):
                            continue
                        x = x[m]; y = y[m]; z = z[m]
                        rr, cc = rowcol(ds.transform, x, y)
                        rr = np.asarray(rr); cc = np.asarray(cc)
                        mm = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                        if not np.any(mm):
                            continue
                        rr = rr[mm]; cc = cc[mm]; z = z[mm]
                        idx = rr * w + cc
                        pix_idx.append(idx)
                        zvals.append(z)

                    if pix_idx:
                        idx = np.concatenate(pix_idx)
                        zv = np.concatenate(zvals).astype('float32')
                        # median per pixel
                        order = np.argsort(idx)
                        idx = idx[order]
                        zv = zv[order]
                        uniq, start = np.unique(idx, return_index=True)
                        # compute median by splitting (fast enough for typical sizes)
                        med = np.empty_like(uniq, dtype='float32')
                        for i, s0 in enumerate(start):
                            s1 = start[i+1] if i+1 < len(start) else len(idx)
                            med[i] = np.median(zv[s0:s1])
                        rr = (uniq // w).astype('int64')
                        cc = (uniq % w).astype('int64')
                        arr[rr, cc] = med
                        ds.write(arr.astype('float32'), 1)
                report.setdefault('fusion', {}).setdefault('constraints', {})['xyz_burned_into_final'] = True
                report['fusion']['constraints']['xyz_files'] = [str(Path(x).name) for x in xyz_paths]
        except Exception as _burn_e:
            report.setdefault('fusion', {}).setdefault('constraints', {})['xyz_burned_into_final'] = False
            report['fusion']['constraints']['xyz_burn_error'] = str(_burn_e)

        return Path(out_depth)

    except Exception as e:
        # Fall back to priority-copy, but keep the error visible in the report
        err = f"weighted_overlap fusion failed; falling back to priority copy: {e}"
        report["fusion"] = {
            "status": "degraded",
            "mode": "priority_fallback",
            "weighted_overlap_failed": True,
            "fallback_used": True,
            "priority": pri,
            "error": err,
            "outputs": {"depth": str(out_depth)},
        }
        log.warning(err)

        # Robust fallback: union-overlay combine (fill priority nodata with secondary). If that fails, do priority copy.
        try:
            _gdal_union_overlay(Path(sdb_fuse_path), Path(river_fuse_path), Path(out_depth), Path(template), pri)
            report["fusion"]["mode"] = "union_overlay_fallback"
            report["fusion"]["note"] = "Fusion failed; used GDAL union overlay fallback."
            return out_depth
        except Exception as ee2:
            report["fusion"]["note"] = f"Union overlay fallback also failed; priority copy used. union_error={ee2}"

        # Priority copy as last resort
        if pri == "river":
            shutil.copy2(str(river_raster), str(out_depth))
        else:
            shutil.copy2(str(sdb_raster), str(out_depth))
        return out_depth


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Coastal + River Bathymetry Pipeline", conflict_handler="resolve")
    p.add_argument("--aoi", required=True, help="AOI W/E/S/N")
    p.add_argument("--start", dest="start_date", required=True)
    p.add_argument("--end", dest="end_date", required=True)
    # Tile seam stability policy (operational mode)
    # We run on an expanded AOI to reduce boundary effects, then clip to the tile AOI.
    p.add_argument("--tile-buffer-km", type=float, default=15.0,
                   help="Buffer (km) added around --aoi for processing; outputs are clipped back to the tile AOI (default 15 km).")
    p.add_argument("--tile-edge-taper-km", type=float, default=2.0,
                   help="Within this distance (km) of the tile edges, blend toward a low-frequency smooth field to improve seam consistency (default 2 km). Set to 0 to disable.")
    p.add_argument("--tile-edge-smooth-sigma-km", type=float, default=10.0,
                   help="Gaussian smooth sigma (km) for the low-frequency edge taper reference (default 10 km).")
    p.add_argument("--tile-edge-metrics-band-km", type=float, default=2.0,
                   help="Edge band width (km) used for seam diagnostics in run summaries (default 2 km).")
    p.add_argument("--no-tile-edge-taper", dest="tile_edge_taper_enabled", action="store_false",
                   help="Disable tile edge tapering (use raw clipped outputs).")
    p.set_defaults(tile_edge_taper_enabled=True)

    # SDB cross-tile consistency (DEFAULT): bounded model bank reservoir + periodic retrain
    p.add_argument("--sdb-model-bank", default="auto",
                   help="Model bank directory for incremental SDB training. 'auto' => <cache-root>/model_bank/sdb_global_v1")
    p.add_argument("--no-sdb-model-bank", dest="sdb_model_bank_enabled", action="store_false",
                   help="Disable model bank; train only on this AOI (not recommended for seamless tiling).")
    p.set_defaults(sdb_model_bank_enabled=True)
    p.add_argument("--sdb-bank-max-samples", type=int, default=100000,
                   help="Max samples to keep in the model bank reservoir (bounded disk).")
    p.add_argument("--sdb-bank-seed", type=int, default=1337,
                   help="Seed for deterministic reservoir sampling in the model bank.")
    p.add_argument("--sdb-bank-retrain-min-new", type=int, default=2000,
                   help="Only retrain the RF when at least this many new samples have been added to the bank since last training.")

    # Deprecated: regional model cache (kept for backwards compatibility; OFF by default)
    p.add_argument("--sdb-model-cache-key", default="auto",
                   help="[DEPRECATED] Key for regional SDB model reuse (prefer model bank).")
    p.add_argument("--sdb-model-cache", dest="sdb_model_cache_enabled", action="store_true",
                   help="[DEPRECATED] Enable regional SDB model reuse (prefer model bank).")
    p.add_argument("--no-sdb-model-cache", dest="sdb_model_cache_enabled", action="store_false",
                   help="[DEPRECATED] Disable regional SDB model reuse.")
    p.set_defaults(sdb_model_cache_enabled=False)


    p.add_argument("--out-dir", default="output/unified")
    p.add_argument("--methods", default="sdb,river")
    p.add_argument("--priority", default="sdb", choices=["sdb", "river"])

    # SDB passthrough
    p.add_argument("--cloud", type=int, default=70)
    p.add_argument("--icesat", default="all_atl")
    p.add_argument("--sdb-mode", default="all_sdb")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--align-mode", default="median")

    # S2 sun-glint correction (Hedley-style). Only used when methods include 'sdb'.
    p.add_argument("--glint-correct", action="store_true", default=False,
                   help="Enable Hedley-style sun-glint correction on Sentinel-2 composite reflectance (SDB only).")
    p.add_argument("--glint-nir-band", default="B08",
                   help="NIR band used as glint proxy (default B08).")
    p.add_argument("--glint-vis-bands", default="B02,B03,B04",
                   help="Comma-separated visible bands to correct (default B02,B03,B04).")
    p.add_argument("--glint-nir-min-percentile", type=float, default=1.0,
                   help="Percentile for baseline NIR (R_nir,min) over stable-water mask (default 1.0).")
    p.add_argument("--glint-deepwater-b02-max", type=float, default=0.20,
                   help="Max B02 reflectance allowed in deep/stable water mask (default 0.20).")
    p.add_argument("--glint-min-samples", type=int, default=5000,
                   help="Minimum number of stable-water pixels required to fit betas (default 5000).")
    p.add_argument("--glint-max-samples", type=int, default=2000000,
                   help="Max number of samples used to fit betas (default 2000000).")
    p.add_argument("--glint-clip-min", type=float, default=1e-6,
                   help="Clip minimum for corrected reflectance to avoid downstream log/ratio issues (default 1e-6).")


    # River
    p.add_argument("--river-dem", default=None)

    p.add_argument("--working-srs", default="auto",
                   help="Working horizontal CRS for meter-based operations (thinning, river DEM warps). "
                        "Use 'auto' to detect from cached Sentinel-2 RGB_10m.tif or fall back to AOI-center UTM (WGS84).")
    p.add_argument("--working-vcrs-epsg", type=int, default=5703,
                   help="Working vertical CRS EPSG code (default 5703 = NAVD88 height).")
    p.add_argument("--final-out-srs", default="EPSG:4269",
                   help="Final output raster horizontal CRS (default EPSG:4269 = NAD83 geographic). Depth values are relative to water surface; vertical datum is stored as metadata tags, not CRS.")

    # Vertical datum transformation (MSL → NAVD88)
    p.add_argument("--convert-sdb-to-navd88", action="store_true", default=False,
                   help="Convert SDB from MSL to NAVD88 vertical datum using dlim. "
                        "Output: SDB_Prediction_10m_NAVD88.tif. Requires dlim (CUDEM) on PATH.")
    p.add_argument("--sdb-source-vdatum", default="epsg:4269+5714",
                   help="Source compound EPSG (default: epsg:4269+5714 = NAD83+MSL).")
    p.add_argument("--sdb-target-vdatum", default="epsg:4269+5703",
                   help="Target compound EPSG (default: epsg:4269+5703 = NAD83+NAVD88).")

    p.add_argument("--river-dem-auto", action="store_true", default=True,
                   help="If --river-dem is not provided, auto-download TNM 1/3 arc-sec DEM via `fetches` and build a clipped DEM.")
    p.add_argument("--no-river-dem-auto", dest="river_dem_auto", action="store_false",
                   help="Disable auto river DEM download/build; river method requires --river-dem.")
    p.add_argument("--river-dem-source", default="tnm:datasets=3",
                   help="CUDEM fetches source string for river DEM auto-download (default tnm:datasets=3).")
    p.add_argument("--river-dem-res-m", type=float, default=10.0,
                   help="Target resolution (meters) for the auto-built river DEM in working CRS (default 10).")
    # Unified extra bathymetry soundings input (preferred user-facing name).
    # You can repeat this flag, pass a comma-separated list, or pass a directory of files.
    p.add_argument(
        "--extra-xyz",
        dest="extra_xyz",
        action="append",
        default=[],
        help="Extra bathymetry soundings (file, comma-list, or directory). Supports .xyz/.csv/.txt/.dat/.gpkg/.shp/.geojson. Can be repeated."
    )


    p.add_argument("--extra-xyz-crs", default="EPSG:4326", help="CRS for raw extra XYZ files (default EPSG:4326).")
    p.add_argument(
        "--extra-xyz-cudem",
        dest="extra_xyz_cudem",
        action="append",
        default=[],
        help=("Auto-download external soundings with CUDEM dlim into cache (e.g., hydronos, ehydro). "
              "Provide a comma-separated list and/or repeat the flag. Example: --extra-xyz-cudem hydronos,ehydro"),
    )
    p.add_argument(
        "--extra-xyz-cudem-crs",
        default="epsg:4269+5714",
        help=("Compound CRS string passed to dlim -P for --extra-xyz-cudem outputs. "
              "Default: epsg:4269+5714 = NAD83 + MSL height. This ensures downloaded soundings "
              "(which are typically in MLLW) are converted to MSL to match ICESat-2 training data. "
              "Use --convert-sdb-to-navd88 to convert final SDB output from MSL to NAVD88."),
    )

    p.add_argument(
        "--extra-xyz-cudem-thin-res-m",
        type=float,
        default=10.0,
        help=("If set, apply CUDEM dlim filtering/thinning to auto-downloaded XYZ (default: 10 meters). "
              "Implemented as -F block_thin:res=<value>. Set to 0 to disable."),
    )
    p.add_argument(
        "--extra-xyz-cudem-filter",
        default=None,
        help=("Explicit CUDEM dlim -F filter string for auto-downloaded XYZ (overrides --extra-xyz-cudem-thin-res-m). "
              "Example: block_thin:res=10"),
    )
    p.add_argument(
        "--extra-xyz-cudem-force",
        action="store_true",
        help="Force re-download of --extra-xyz-cudem soundings even if cached file exists.",
    )

    # Backwards-compatibility alias (hidden from --help)
    p.add_argument(
        "--river-soundings",
        dest="extra_xyz",
        action="append",
        help=argparse.SUPPRESS
    )
    
    # Adaptive Spatial Sampling (v0.7.0+, ENABLED BY DEFAULT in v0.7.1)
    p.add_argument("--disable-adaptive-sampling", dest="enable_adaptive_sampling",
                   action="store_false", default=True,
                   help="Disable adaptive spatial sampling (sampling is ON by default)")
    p.add_argument("--sampling-target-points", type=int, default=2000,
                   help="Target number of training points after adaptive sampling (default: 2000)")
    p.add_argument("--sampling-min-threshold", type=int, default=3000,
                   help="Minimum input points before adaptive sampling is applied (default: 3000)")
    p.add_argument("--sampling-max-gap-m", type=float, default=100.0,
                   help="Maximum spatial gap allowed in meters (default: 100m)")
    
    # River network parameters
    p.add_argument("--river-hydrography-source", default="arcgis", choices=["arcgis","arcgis_tnm","tnm"],
                   help="Hydrography acquisition for river network: arcgis (default) queries ArcGIS REST only; arcgis_tnm uses TNM as a fallback; tnm prefers TNM (and may still fall back to ArcGIS).")
    p.add_argument("--no-tnm", action="store_true", help="Disable TNM download (if using local flowlines)")
    p.add_argument("--tnm-dataset", default="NHDPlusHR")
    p.add_argument("--snap-m", type=float, default=30.0)


    # River bathymetry method selection
    p.add_argument("--river-method", choices=["hybrid","skeleton", "xs"], default="hybrid",
               help=("River bathy method. 'hybrid' (default) runs XS only on the mainstem and uses the skeleton method elsewhere, then combines them so mainstem depths are continuous. 'skeleton' uses a raster distance-transform channel skeleton (recommended for dense tributaries/meanders/tidal channels). 'xs' uses cross-sections everywhere (more artifact-prone at junctions)."))
    # Skeleton (distance-transform) parameters: domain mask (river vs ocean) + channel profile
    p.add_argument("--river-channel-buffer-m", type=float, default=400.0,
               help="Buffer around NHD flowlines used to define candidate river corridor (meters).")
    p.add_argument("--river-max-channel-width-m", type=float, default=600.0,
               help="Maximum channel width allowed inside the corridor (meters). Prevents filling open bays/ocean.")
    p.add_argument("--river-mainstem-min-order", type=int, default=5,
               help="Stream order threshold for allowing larger widths (if stream order attribute exists).")
    p.add_argument("--river-max-mainstem-width-m", type=float, default=2500.0,
               help="Maximum channel width allowed for mainstem corridor pixels (meters).")

    # Optional: constrain river predictions to NHDArea polygons when available

    # River channel domain policy
    # - auto: prefer NHDArea river/stream polygons when available; otherwise fall back to buffered flowline corridor
    # - nhdarea: require NHDArea river/stream polygons (excludes lakes)
    # - corridor: buffered flowline corridor only
    p.add_argument("--river-channel-source", dest="river_channel_source",
               choices=["auto", "nhdarea", "corridor"], default="auto",
               help="Channel mask source policy for skeleton method: auto|nhdarea|corridor (default: auto).")

    # Connectivity filter for channel domain (recommended on): keeps only water components connected to the river network seed.
    p.add_argument("--river-connectivity-filter", dest="river_connectivity_filter",
               action="store_true", default=True,
               help="Enable connectivity filtering when building the river channel mask (default: on).")
    p.add_argument("--no-river-connectivity-filter", dest="river_connectivity_filter",
               action="store_false",
               help="Disable connectivity filtering when building the river channel mask.")

    # NHDArea filtering: by default keep Stream/River polygons only (FType=460). This excludes lakes/ponds.
    p.add_argument("--river-nhdarea-allow-ftype", dest="river_nhdarea_allow_ftype", default="460",
               help="Comma-separated list of allowed NHDArea FType codes to treat as river/stream area (default: 460 Stream/River).")
    p.add_argument("--river-nhdarea-allow-fcode", dest="river_nhdarea_allow_fcode", default=None,
               help="Optional comma-separated list of allowed NHDArea FCode values (used only if FType is not present).")

    p.add_argument("--no-river-nhdarea", dest="river_use_nhdarea", action="store_false", default=True,
               help="Disable NHDArea polygon constraint for river domain (if available).")
    p.add_argument("--river-nhdarea-layer", default="nhdarea_clip",
               help="Layer name in river_network.gpkg containing NHDArea polygons (default nhdarea_clip).")
    p.add_argument("--river-ocean-keep-dist-m", dest="river_ocean_keep_dist_m", type=float, default=0.0,
               help="Allow ocean-connected water within this distance (m) of flowlines when building the river channel mask. Useful for tidal river mouths/estuaries where the mainstem is classified as ocean water. 0 disables.")
    p.add_argument("--river-shape-exp", type=float, default=0.5,
               help="Depth profile exponent for skeleton method (0.5 ~ U-shape; 1.0 ~ V-shape).")
    p.add_argument("--river-dmax-min-m", type=float, default=0.5, help="Clamp minimum Dmax prior (meters).")
    p.add_argument("--river-dmax-max-m", type=float, default=30.0, help="Clamp maximum Dmax prior (meters).")
    # Optional: longitudinal bed profile constraints (skeleton method)
    p.add_argument("--river-bed-profile-max-slope", dest="river_bed_profile_max_slope", type=float, default=0.0,
               help="Max absolute bed slope |dz/ds| along the river skeleton (m/m). 0 disables.")
    p.add_argument("--river-bed-profile-max-curv", dest="river_bed_profile_max_curv", type=float, default=0.0,
               help="Max absolute bed curvature |d2z/ds2| along the river skeleton (1/m). 0 disables.")
    p.add_argument("--river-bed-profile-step-m", dest="river_bed_profile_step_m", type=float, default=25.0,
               help="Sampling step (m) along flowlines when building the bed profile constraint (default: 25).")
    p.add_argument("--river-bed-profile-strength", dest="river_bed_profile_strength", type=float, default=0.6,
               help="Blend strength (0..1) for applying bed profile constraints (default: 0.6).")
    p.add_argument("--river-bed-profile-power", dest="river_bed_profile_power", type=float, default=2.0,
               help="Distance-decay power for spreading bed profile corrections from the skeleton (default: 2.0).")
    p.add_argument("--river-save-skeleton-debug", action="store_true", default=False,
               help="Write skeleton debug rasters (r, d_bank, d_center, dmax, wse).")
    p.add_argument("--river-skeleton-wse-mode", dest="river_skeleton_wse_mode", default="bank",
               choices=["bank", "skeleton", "bank_profile"],
               help="Skeleton WSE proxy mode: 'bank' uses bank-adjacent DEM samples (recommended); 'skeleton' uses in-channel centerline sampling (legacy).")
    p.add_argument("--river-skeleton-wse-smooth-sigma-m", dest="river_skeleton_wse_smooth_sigma_m", type=float, default=0.0,
               help="Optional Gaussian smoothing sigma (m) for the skeleton WSE proxy field inside the channel.")
# Longitudinal WSE profile controls (river-skeleton-wse-mode=bank_profile)
    p.add_argument("--river-skeleton-wse-profile-step-m", dest="river_skeleton_wse_profile_step_m", type=float, default=20.0,
               help="Densification step (m) for sampling along flowlines when building a longitudinal WSE profile.")
    p.add_argument("--river-skeleton-wse-profile-resample-m", dest="river_skeleton_wse_profile_resample_m", type=float, default=20.0,
               help="Resample step (m) for 1D WSE profile smoothing along flow distance.")
    p.add_argument("--river-skeleton-wse-profile-smooth-sigma-m", dest="river_skeleton_wse_profile_smooth_sigma_m", type=float, default=200.0,
               help="Gaussian smoothing sigma (m) applied to the 1D WSE profile along flow distance.")
    p.add_argument("--river-skeleton-wse-profile-max-slope", dest="river_skeleton_wse_profile_max_slope", type=float, default=0.005,
               help="Optional maximum absolute slope (m/m) enforced along the 1D WSE profile (0 to disable).")
    p.add_argument("--river-skeleton-wse-profile-min-samples", dest="river_skeleton_wse_profile_min_samples", type=int, default=10,
               help="Minimum number of valid samples along flowlines to build a profile; otherwise falls back to wse-mode=bank.")
    p.add_argument("--river-skeleton-wse-profile-max-query-dist-m", dest="river_skeleton_wse_profile_max_query_dist_m", type=float, default=250.0,
               help="Maximum XY distance (m) for assigning profile samples to centerline pixels (0 to disable). Helps avoid cross-reach snapping.")


    # Optional: SWOT RiverSP anchoring for river-skeleton-wse-mode=bank_profile
    p.add_argument("--river-swot-riversp", dest="river_swot_riversp", nargs="+", default=None,
               help="One or more SWOT RiverSP vector files (reach or node product; e.g., .shp/.gpkg/.geojson) with WSE. Used only when river-skeleton-wse-mode=bank_profile.")

    # Auto-fetch (optional): if --river-swot-riversp is omitted, attempt to fetch RiverSP via Earthdata/PO.DAAC.
    p.add_argument("--river-swot-auto", dest="river_swot_auto", action="store_true", default=True,
               help="Enable auto-fetch of SWOT RiverSP when --river-swot-riversp is not provided (default: enabled). Requires Earthdata login (~/.netrc or EARTHDATA_USERNAME/EARTHDATA_PASSWORD).")
    p.add_argument("--no-river-swot-auto", dest="river_swot_auto", action="store_false",
               help="Disable auto-fetch of SWOT RiverSP (only use SWOT if --river-swot-riversp is provided).")
    p.add_argument("--river-swot-cache-root", dest="river_swot_cache_root", default=None,
               help="Cache root for auto-fetched SWOT RiverSP (default: <cache-root>/swot).")
    p.add_argument("--river-swot-product", dest="river_swot_product", choices=["reach","node"], default="reach",
               help="RiverSP product to search when auto-fetching: reach (default) or node.")
    p.add_argument("--river-swot-shortname", dest="river_swot_shortname", default=None,
               help="Optional PO.DAAC short_name to use for auto-fetch (advanced). If omitted, a best-effort search is performed.")

    p.add_argument("--river-swot-wse-field", dest="river_swot_wse_field", default=None,
               help="Column name for SWOT WSE in the RiverSP file(s). If omitted, common candidates will be searched.")
    p.add_argument("--river-swot-qual-field", dest="river_swot_qual_field", default=None,
               help="Optional column name for a SWOT quality flag; if provided, values >0 are treated as bad and filtered out.")
    p.add_argument("--river-swot-max-dist-m", dest="river_swot_max_dist_m", type=float, default=300.0,
               help="Max distance (m) from a flowline sample point to accept a SWOT WSE observation.")
    p.add_argument("--river-swot-min-samples", dest="river_swot_min_samples", type=int, default=5,
               help="Minimum number of SWOT samples on a flowline required to apply anchoring corrections.")
    p.add_argument("--river-swot-correct-sigma-m", dest="river_swot_correct_sigma_m", type=float, default=2000.0,
               help="Smoothing scale (m) for along-channel SWOT correction (Gaussian sigma along distance).")
    p.add_argument("--river-swot-weight", dest="river_swot_weight", type=float, default=1.0,
               help="Weight (0..1) to apply SWOT correction to the bank-derived WSE profile.")
    p.add_argument("--river-swot-max-correction-m", dest="river_swot_max_correction_m", type=float, default=5.0,
               help="Clamp the along-channel correction magnitude (m) applied from SWOT (helps avoid datum/offset mistakes).")
    p.add_argument("--river-swot-wse-offset-m", dest="river_swot_wse_offset_m", type=float, default=0.0,
               help="Constant offset (m) added to SWOT WSE before use. Use to reconcile vertical datums until full datum transform is implemented.")
    p.add_argument("--river-skeleton-junction-mode", dest="river_skeleton_junction_mode", default="smooth",
               choices=["smooth","mask","none"],
               help="How to handle confluence/junction zones (degree>=3 graph nodes): smooth (default), mask, or none.")
    p.add_argument("--river-skeleton-junction-buffer-m", dest="river_skeleton_junction_buffer_m", type=float, default=120.0,
               help="Buffer radius (m) around junction nodes used for smoothing/masking.")
    p.add_argument("--river-skeleton-junction-degree-min", dest="river_skeleton_junction_degree_min", type=int, default=3,
               help="Minimum graph node degree to be treated as a junction.")
    p.add_argument("--river-skeleton-junction-smooth-sigma-m", dest="river_skeleton_junction_smooth_sigma_m", type=float, default=80.0,
               help="Gaussian smoothing sigma (m) used in junction zones when mode=smooth.")
    p.add_argument("--river-skeleton-junction-max-width-m", dest="river_skeleton_junction_max_width_m", type=float, default=300.0,
               help="Limit junction smoothing/masking to pixels with estimated channel width <= this (m). Helps avoid over-smoothing wide confluences.")
    # Curvature-driven asymmetry (outer-bank deeper in bends)
    p.add_argument("--river-skeleton-asymmetry-mode", dest="river_skeleton_asymmetry_mode",
               choices=["none","curvature"], default="none",
               help="Optional curvature-driven asymmetry: biases depth toward the outer bank using signed centerline curvature.")
    p.add_argument("--river-skeleton-asymmetry-strength", dest="river_skeleton_asymmetry_strength", type=float, default=0.25,
               help="Strength of curvature-driven r-shift (dimensionless; typical 0.1–0.4).")
    p.add_argument("--river-skeleton-asymmetry-curv-ref", dest="river_skeleton_asymmetry_curv_ref", type=float, default=0.002,
               help="Reference curvature (1/m) for scaling (e.g., 0.002 ~ 500 m radius).")
    p.add_argument("--river-skeleton-asymmetry-max-shift", dest="river_skeleton_asymmetry_max_shift", type=float, default=0.20,
               help="Max absolute shift applied to r (clamped).")
    p.add_argument("--river-skeleton-asymmetry-min-width-m", dest="river_skeleton_asymmetry_min_width_m", type=float, default=10.0,
               help="Only apply asymmetry where estimated channel width >= this (m).")
    p.add_argument("--river-skeleton-asymmetry-min-curv", dest="river_skeleton_asymmetry_min_curv", type=float, default=0.0005,
               help="Only apply asymmetry where |curvature| >= this (1/m).")
    p.add_argument("--river-skeleton-asymmetry-densify-step-m", dest="river_skeleton_asymmetry_densify_step_m", type=float, default=20.0,
               help="Vertex spacing (m) used when estimating curvature/tangent from flowlines.")

    p.add_argument("--river-swot-offset-mode", dest="river_swot_offset_mode",
                   choices=["none", "median_mad"], default="median_mad",
                   help=("Vertical datum reconciliation mode for SWOT WSE vs bank_profile WSE. "
                         "'median_mad' estimates a robust constant offset from overlapping samples and subtracts it from SWOT WSE. "
                         "'none' disables auto offset estimation (use --river-swot-wse-offset-m instead)."))
    p.add_argument("--river-swot-offset-min-samples", dest="river_swot_offset_min_samples", type=int, default=25,
                   help="Minimum number of SWOT samples required to estimate an auto vertical offset.")
    p.add_argument("--river-swot-offset-mad-z", dest="river_swot_offset_mad_z", type=float, default=3.5,
                   help="MAD-based outlier rejection threshold (in robust-sigma units) for auto vertical offset estimation.")
    p.add_argument("--river-swot-offset-max-abs-m", dest="river_swot_offset_max_abs_m", type=float, default=10.0,
                   help="Clamp absolute value of the auto-estimated vertical offset (m). If exceeded, it is clamped and a warning is logged.")


    p.add_argument("--river-soundings-mode", choices=["auto","depth_pos","depth_neg","bed_elev"], default="auto",
                   help="How to interpret extra XYZ Z values for river skeleton: auto/depth_pos/depth_neg (depths) or bed_elev (bed elevations, same vertical datum as DEM).")
    p.add_argument("--river-soundings-max-dist-m", type=float, default=10000.0,
                   help="Max distance (m) for extra XYZ to influence the skeleton Dmax prior.")
    p.add_argument("--river-soundings-min-r", type=float, default=0.25,
                   help="Minimum r used when converting sounding depth -> implied Dmax (stabilizes near banks).")
    p.add_argument("--no-river-soundings-enforce", dest="river_soundings_enforce", action="store_false", default=True,
               help="Disable enforcing observed soundings at their grid cells (default enforces).")
    p.add_argument("--river-authoritative-bed", default=None,
                   help="Optional authoritative river bed elevation raster to blend/enforce inside the river mask (same vertical datum as DEM).")
    p.add_argument("--river-authoritative-bed-max-dist-m", type=float, default=2000.0,
                   help="Max distance (m) from authoritative bed pixels to influence residual blending.")
    p.add_argument("--river-residual-blend-sigma-m", type=float, default=600.0,
                   help="Gaussian sigma (m) for residual blending smoothing. 0 disables smoothing (nearest-only).")
    p.add_argument("--xs-spacing-m", type=float, default=200.0)
    p.add_argument("--xs-length-m", type=float, default=300.0)

    p.add_argument("--xs-smoothing-window-m", type=float, default=0.0,
                   help="Smoothing window (m) for XS orientation tangents. 0=auto (uses xs-spacing-m).")
    p.add_argument("--xs-deconflict-tol-m", type=float, default=2.0,
                   help="Endpoint tolerance (m) when identifying intersecting cross-sections.")
    p.add_argument("--xs-junction-snap-m", type=float, default=30.0,
                   help="Snapping scale (m) for junction detection from reach endpoints.")
    p.add_argument("--xs-junction-buffer-m", type=float, default=75.0,
                   help="Skip XS within this distance (m) of junction nodes.")
    p.add_argument("--xs-densify-step-m", type=float, default=20.0,
                   help="Densify centerlines to this vertex spacing (m) before computing tangents.")

    p.add_argument("--no-xs-trim-overlaps", dest="xs_trim_overlaps", action="store_false",
                   help="Disable local overlap trimming between adjacent XS.")
    p.set_defaults(xs_trim_overlaps=True)

    p.add_argument("--no-xs-global-deconflict", dest="xs_global_deconflict", action="store_false",
                   help="Disable dropping XS that intersect non-adjacent XS within a reach.")
    p.set_defaults(xs_global_deconflict=True)

    p.add_argument("--no-xs-skip-junctions", dest="xs_skip_junctions", action="store_false",
                   help="Do not skip XS near confluences/junctions.")
    p.set_defaults(xs_skip_junctions=True)

    p.add_argument("--river-thalweg-only", action="store_true",
                   help="Use a thalweg-only control spine (1 control point per cross-section) for river interpolation; also builds corridor from buffered thalweg spine to avoid cross-channel ribbing.")
    p.add_argument("--river-thalweg-densify-factor", type=float, default=0.5,
                   help="Densify thalweg spine vertices to this fraction of the river template raster pixel size (default 0.5 => >=2 vertices per pixel).")
    p.add_argument("--river-thalweg-densify-step-m", type=float, default=None,
                   help="Explicit thalweg spine densify step in meters. Overrides --river-thalweg-densify-factor if provided.")

    # River patch rasterization / interpolation options (forwarded to xs_infer_bathy_raster.py)
    p.add_argument("--river-continuous", choices=["median", "walid", "aidw", "aniso", "walid_aniso"], default="walid_aniso",
                   help="How to produce the river bed patch raster surface from inferred bathy points (see xs_infer_bathy_raster.py --continuous).")
    p.add_argument("--river-continuous-buffer-m", type=float, default=None,
                   help="Buffer (meters) around points used to define interpolation corridor. Default scales with pixel size.")
    p.add_argument("--river-continuous-k", type=int, default=12, help="K nearest points used for IDW/AIDW/anisotropic modes.")
    p.add_argument("--river-idw-power", type=float, default=2.0, help="IDW power for --river-continuous modes.")
    p.add_argument("--river-aniso-along-scale-m", type=float, default=500.0, help="Anisotropic IDW along-channel scale (meters).")
    p.add_argument("--river-aniso-cross-scale-m", type=float, default=30.0, help="Anisotropic IDW cross-channel scale (meters).")
    p.add_argument("--river-thalweg-weight", type=float, default=6.0, help="Extra influence for thalweg control points in walid modes.")
    p.add_argument("--river-overlap-reducer", choices=["min", "median"], default="min",
                   help="When multiple points fall in the same output pixel, how to collapse them. 'min' keeps the deeper bed.")
    p.add_argument("--river-nodata", type=float, default=-9999.0, help="Nodata value for river float rasters.")
    # River depth inference priors / anchors (passed through to xs_infer_bathy_raster.py)
    p.add_argument("--river-prior-mode", choices=["powerlaw", "multivariate"], default="powerlaw",
                   help=("River Dmax prior mode. 'powerlaw' uses Dmax=a*W^b. "
                         "'multivariate' uses W plus reach attributes (drainage area, slope). "
                         "For regional curves DA→Dbkf (Dbkf=c*DA^f), use multivariate with mv_bw=0, mv_ba=f, mv_a0=c."))
    p.add_argument("--river-mv-a0", type=float, default=0.18)
    p.add_argument("--river-mv-bw", type=float, default=0.50)
    p.add_argument("--river-mv-ba", type=float, default=0.0)
    p.add_argument("--river-mv-bs", type=float, default=-0.10)
    p.add_argument("--river-mv-eps-a", type=float, default=1.0)
    p.add_argument("--river-mv-eps-s", type=float, default=1e-4)
    # Slope proxy stabilization (used when slope attribute is missing)
    p.add_argument("--river-slope-proxy-window", type=int, default=9, help="Rolling median window (XS count) for WSE smoothing before slope differencing.")
    p.add_argument("--river-slope-min", type=float, default=1e-5, help="Minimum plausible slope (m/m) for slope proxy.")
    p.add_argument("--river-slope-max", type=float, default=0.05, help="Maximum plausible slope (m/m) for slope proxy.")
    p.add_argument("--river-slope-proxy-min-n", type=int, default=7, help="Minimum XS per river_id required to compute slope proxy.")

    # Longitudinal WSE profile fitting (preferred slope proxy when reach slope attribute is missing)
    p.add_argument("--river-wse-profile-enabled", dest="river_wse_profile_enabled", action="store_true", help="Enable longitudinal WSE profile fitting (default).")
    p.add_argument("--no-river-wse-profile", dest="river_wse_profile_enabled", action="store_false", help="Disable WSE profile fitting; use legacy slope proxy only.")
    p.set_defaults(river_wse_profile_enabled=True)
    p.add_argument("--river-wse-profile-window", type=int, default=9, help="Rolling window (XS count) for WSE profile smoothing.")
    p.add_argument("--river-wse-profile-min-n", type=int, default=7, help="Minimum XS per river_id required to fit a WSE profile.")
    p.add_argument("--river-wse-profile-monotonic", dest="river_wse_profile_monotonic", action="store_true", help="Enforce monotonic WSE along stationing (default).")
    p.add_argument("--no-river-wse-profile-monotonic", dest="river_wse_profile_monotonic", action="store_false", help="Disable monotonic constraint.")
    p.set_defaults(river_wse_profile_monotonic=True)

    # Slope proxy stabilization (used when slope attribute is missing)

    # Option A: USGS discharge *measurement* anchors
    p.add_argument("--river-usgs-sites", default=None,
                   help="Comma-separated USGS site numbers to use as anchors (e.g., 01646500,01651000).")
    p.add_argument("--river-usgs-start", default=None, help="Start date YYYY-MM-DD for USGS discharge measurements.")
    p.add_argument("--river-usgs-end", default=None, help="End date YYYY-MM-DD for USGS discharge measurements.")
    p.add_argument("--river-usgs-cache-dir", default=None, help="Optional cache directory for NWIS responses.")
    p.add_argument("--river-usgs-max-dist-m", type=float, default=5000.0,
                   help="Max distance (m) from gage to XS center for applying USGS anchor.")
    p.add_argument("--river-usgs-mean-to-dmax", default="auto",
                   help="Convert mean depth (area/width) to Dmax for anchor fitting. Use a number (e.g., 1.3) or 'auto' to use 2/(1+bottom_width_frac).")
    p.add_argument("--river-usgs-a-stat", choices=["median", "p90", "mean"], default="median",
                   help="How to aggregate a_site from multiple discharge measurements.")
    p.add_argument("--river-usgs-q-quantile-lo", type=float, default=0.20, help="Lower discharge quantile for filtering USGS measurements (0-1).")
    p.add_argument("--river-usgs-q-quantile-hi", type=float, default=0.80, help="Upper discharge quantile for filtering USGS measurements (0-1).")
    p.add_argument("--river-usgs-a-cv-warn", type=float, default=0.50, help="Warn/soften USGS anchor when CV of fitted a-values exceeds this.")
    p.add_argument("--river-usgs-width-ratio-max", type=float, default=3.0, help="Threshold for (bank width / median measured wet width) to down-weight/skip USGS anchor.")
    p.add_argument("--no-river-usgs-width-ratio-blend", dest="river_usgs_width_ratio_blend", action="store_false", help="Disable blending; skip USGS anchor when width ratio is large.")
    p.set_defaults(river_usgs_width_ratio_blend=True)
    p.add_argument("--river-gage-snap-max-dist-m", type=float, default=1000.0, help="Max distance (m) to snap gages to the river network.")

    # Optional: width-stage inversion anchors
    p.add_argument("--river-width-stage-csv", default=None,
                   help="Comma-separated width-stage CSV paths (station_id,date,width_m,stage_m optional).")
    p.add_argument("--river-width-stage-max-dist-m", type=float, default=5000.0)
    p.add_argument("--river-width-stage-min-n", type=int, default=6)
    p.add_argument("--river-width-stage-min-r2", type=float, default=0.25, help="Minimum R^2 for width–stage fit.")
    p.add_argument("--river-width-stage-max-weight", type=float, default=0.8, help="Maximum blend weight for width–stage anchor.")

    # Optional: Manning inversion prior (secondary, blended; requires Q + W + S)
    p.add_argument("--river-manning-enabled", action="store_true", help="Alias: enable Manning prior (sets --river-manning-mode=q2_regional if mode is off).")
    p.add_argument("--river-manning-mode", choices=["off","constant","from_field","q2_regional"], default="off",
                   help=("Add a Manning-based depth prior and blend it into the river Dmax estimate. "
                         "Mode off=disabled; constant=use a single Q for all reaches; from_field=read Q from a reach attribute field in river gpkg."))
    p.add_argument("--river-manning-q-cms", type=float, default=None,
                   help="Discharge Q (m^3/s) to use when --river-manning-mode=constant.")
    p.add_argument("--river-manning-q-field", default=None,
                   help="Reach attribute field name containing discharge Q (m^3/s) when --river-manning-mode=from_field.")
    p.add_argument("--river-manning-n", type=float, default=0.035, help="Manning n roughness (typical 0.03-0.08).")
    p.add_argument("--river-manning-region", default="auto",
                   help="Region key for --river-manning-mode=q2_regional (uses drainage area). Prefer state/region published regressions.")
    p.add_argument("--river-manning-region-auto-map", default=None,
                   help="Optional JSON mapping US state abbreviations -> manning region keys (used when --river-manning-region=auto).")
    p.add_argument("--river-manning-min-confidence", type=float, default=0.30,
                   help="Minimum confidence required to apply Manning prior (0-1).")
    p.add_argument("--river-manning-max-weight", type=float, default=0.60, help="Maximum blend weight for Manning prior (0-1).")
    p.add_argument("--river-manning-backwater-slope-thresh", type=float, default=1e-4,
                   help="Disable/zero Manning prior when slope is below this (backwater/tidal risk).")
    p.add_argument("--river-manning-dist-to-mouth-field", default=None,
                   help="Optional reach attribute field containing distance-to-mouth (km). If provided, can disable Manning prior near mouth.")
    p.add_argument("--river-manning-dist-to-mouth-km-max", type=float, default=10.0,
                   help="If distance-to-mouth field is provided, disable Manning prior when distance <= this (km).")

    # Optional: Regional hydraulic geometry curves (DA -> bankfull depth) soft prior
    p.add_argument("--river-regional-curve-enabled", action="store_true",
                   help="Enable regional curve prior (DA -> bankfull depth) blended into the Dmax prior.")
    p.add_argument("--river-regional-curve-region", default="default",
                   help="Region key for built-in (illustrative) coefficients. Prefer providing --river-regional-curve-c and --river-regional-curve-f from published regional curves.")
    p.add_argument("--river-regional-curve-c", type=float, default=None,
                   help="Coefficient c in D_bkf = c * DA^f. Units depend on --river-regional-curve-da-units and --river-regional-curve-depth-units.")
    p.add_argument("--river-regional-curve-f", type=float, default=None,
                   help="Exponent f in D_bkf = c * DA^f.")
    p.add_argument("--river-regional-curve-da-units", choices=["km2","mi2"], default="km2",
                   help="Drainage area units expected by the curve coefficients.")
    p.add_argument("--river-regional-curve-depth-units", choices=["m","ft"], default="m",
                   help="Depth units of the curve coefficients.")
    p.add_argument("--river-regional-curve-depth-type", choices=["mean","max"], default="mean",
                   help="Whether curve depth represents mean or max depth at bankfull.")
    p.add_argument("--river-regional-curve-to-dmax", choices=["auto","factor"], default="auto",
                   help="Conversion from bankfull depth to Dmax. 'auto' uses trapezoid mean->Dmax conversion; 'factor' uses --river-regional-curve-to-dmax-factor.")
    p.add_argument("--river-regional-curve-to-dmax-factor", type=float, default=1.25,
                   help="Factor to convert bankfull mean depth to Dmax when --river-regional-curve-to-dmax=factor.")
    p.add_argument("--river-regional-curve-unc-pct", type=float, default=40.0,
                   help="Uncertainty of the regional curve (percent, used to down-weight the prior).")
    p.add_argument("--river-regional-curve-max-weight", type=float, default=0.60,
                   help="Maximum blend weight for regional curve prior (0-1).")
    p.add_argument("--river-regional-curve-min-da-km2", type=float, default=1.0,
                   help="Minimum drainage area (km^2) to apply the regional curve prior.")

    
    # Fusion controls
    p.add_argument("--fusion-strategy", default="spatial_taper",
                   choices=["weighted_overlap", "spatial_taper", "priority", "blend"],
                   help="How to fuse SDB + river surfaces when both exist.")
    p.add_argument("--fusion-primary-weight", type=float, default=0.70,
                   help="Primary weight in overlap pixels (only used by weighted_overlap/spatial_taper).")
    p.add_argument("--fusion-secondary-weight", type=float, default=0.30,
                   help="Secondary weight in overlap pixels (only used by weighted_overlap/spatial_taper).")
    p.add_argument("--fusion-taper-m", type=float, default=75.0,
                   help="Taper distance (meters) for spatial_taper inside the river corridor.")

    # Output masking
    p.add_argument("--no-mask-river-to-waffles", action="store_true", default=False,
                   help="Disable masking river depth+bed outputs to the waffles coastline mask (if available).")

    # Intelligent gap filling (Tier 1–2)
    p.add_argument("--gapfill-enabled", action="store_true", default=False,
                   help="Apply intelligent gap-fill correction to the fused bathymetry raster using residual interpolation against high-quality points.")
    p.add_argument("--gapfill-hq", nargs='*', default=None,
                   help="One or more high-quality point files (CSV/TXT/GPKG/Shp) with x/y/z (or lon/lat/depth) columns. If omitted, uses --extra-xyz (if provided).")
    p.add_argument("--gapfill-water-mask", default=None,
                   help="Optional explicit water mask raster (1=water) to constrain gapfill. If omitted, uses waffles mask when available, otherwise finite prior pixels.")
    p.add_argument("--gapfill-method", default="rbf", choices=["rbf", "gp", "idw"],
                   help="Interpolation method for residuals: rbf (thin-plate spline), gp (Gaussian Process), idw (inverse distance).")
    p.add_argument("--gapfill-river-smooth-sigma", type=float, default=500.0,
                   help="Along-channel smoothing sigma (meters) for river corridors.")
    p.add_argument("--gapfill-prior-sigma", default=None,
                   help="Optional prior uncertainty raster for better uncertainty propagation.")
    p.add_argument("--gapfill-bank-elev", default=None,
                   help="Optional bank elevation raster for enforcing bed < bank constraint.")
    p.add_argument("--gapfill-cudem-xyz", action="store_true", default=False,
                   help="Also output CUDEM-compatible XYZ file with uncertainty column.")

    # Run health gates
    p.add_argument("--strict", action="store_true", default=False,
                   help="Fail the run if contract tests or output sanity checks fail.")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    # ---------------------------------------------------------------------
    # Operational tiling: run on buffered AOI, but keep original tile AOI for final clipping/metrics
    # ---------------------------------------------------------------------
    tile_bbox = _parse_aoi_bbox(getattr(args, "aoi", None))
    args.aoi_tile = getattr(args, "aoi", None)
    args.tile_bbox = tile_bbox
    try:
        buf_km = float(getattr(args, "tile_buffer_km", 0.0) or 0.0)
    except Exception:
        buf_km = 0.0
    if tile_bbox and buf_km > 0:
        expanded = _expand_bbox_km(tile_bbox, buf_km)
        args.aoi = _bbox_to_aoi_str(expanded)
    


    # ---------------------------------------------------------------------
    # Run-scoped logging + flight recorder
    # ---------------------------------------------------------------------
    run_id = None
    try:
        from datetime import datetime, timezone
        from logging_config import add_file_handler, start_flight_recorder

        # Deterministic-enough run id for log grouping
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"bathy_{ts}_{os.getpid()}"

        out_dir = Path(getattr(args, 'out_dir', 'output'))
        (out_dir / "run_logs").mkdir(parents=True, exist_ok=True)

        # Per-run text log
        add_file_handler(out_dir / "run_logs" / f"run_{run_id}.log", level=logging.INFO)

        # Structured flight recorder JSONL
        fr_path = start_flight_recorder(out_dir, run_id=run_id)
        if fr_path is not None:
            log.info(f"[RUN] Flight recorder: {fr_path}")
        log.info(f"[RUN] run_id={run_id}")
    except Exception as e:
        log.debug(f"[RUN] Unable to initialize run logs/flight recorder: {e}")
    # Auto-select published regression/curve coefficients from sdb_config.json based on AOI centroid.
    # This sets:
    #   - --river-manning-region (for Q2 DA->Q2 regressions used by Manning inversion)
    #   - --river-regional-curve-c/f (for DA->bankfull depth curves)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(here, 'sdb_config.json')
        cfg_json = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg_json = json.load(f)
        # Parse AOI as bbox: w/e/s/n
        lon_c = lat_c = None
        try:
            if isinstance(getattr(args, 'aoi', None), str) and '/' in args.aoi:
                w,e,s,n = [float(x) for x in args.aoi.strip().strip('"').split('/')[:4]]
                lon_c = (w + e) / 2.0
                lat_c = (s + n) / 2.0
        except Exception:
            lon_c = lat_c = None

        if lon_c is not None and lat_c is not None and cfg_json:
            from region_resolver import resolve_q2_regression, resolve_bankfull_curve

            # Manning Q2 regression region key
            if str(getattr(args, 'river_manning_region', 'default')).strip().lower() == 'auto':
                q2_curve, msg = resolve_q2_regression(cfg_json, lon_c, lat_c)
                if q2_curve is not None and q2_curve.name:
                    args.river_manning_region = q2_curve.name
                    log.info(f"[RIVER][MANNING] Auto region: {msg}")
                else:
                    # fallback to older coarse mapping
                    from manning_inversion import infer_manning_region_from_aoi
                    region_key, st = infer_manning_region_from_aoi(getattr(args, 'aoi', None), state_region_map_json=getattr(args, 'river_manning_region_auto_map', None), default_region='default')
                    args.river_manning_region = region_key
                    log.warning(f"[RIVER][MANNING] Published regression not found; using coarse region={region_key} (state={st})")

            # Bankfull depth regional curve coefficients
            if getattr(args, 'river_regional_curve_enabled', False) and (getattr(args, 'river_regional_curve_c', None) is None or getattr(args, 'river_regional_curve_f', None) is None):
                bcurve, msg = resolve_bankfull_curve(cfg_json, lon_c, lat_c)
                if bcurve is not None:
                    args.river_regional_curve_region = bcurve.name
                    args.river_regional_curve_c = float(bcurve.c)
                    args.river_regional_curve_f = float(bcurve.f)
                    # units in registry are recorded; keep user-selected units args as-is, but warn if mismatch
                    log.info(f"[RIVER][REGIONAL] Auto coefficients: {msg}")
                else:
                    log.warning(f"[RIVER][REGIONAL] Auto coefficients unavailable: {msg}")
    except Exception as e:
        log.debug(f"[RIVER] Auto coefficient resolution failed: {e}")


    # Optional: auto-fetch SWOT RiverSP if user didn't provide a local path.
    try:
        if (str(getattr(args, "river_skeleton_wse_mode", "bank")).strip().lower() == "bank_profile"
                and getattr(args, "river_swot_riversp", None) is None
                and bool(getattr(args, "river_swot_auto", True))):
            # Parse AOI bbox: w/e/s/n
            w = e = s = n = None
            try:
                if isinstance(getattr(args, "aoi", None), str) and "/" in args.aoi:
                    w, e, s, n = [float(x) for x in args.aoi.strip().strip('"').split("/")[:4]]
            except Exception:
                w = e = s = n = None
    
            if None not in (w, e, s, n):
                swot_cache_root = getattr(args, "river_swot_cache_root", None)
                if not swot_cache_root:
                    # keep RiverSP under cache_root/swot by default
                    swot_cache_root = os.path.join(str(getattr(args, "cache_root", "cache")), "swot")
                from swot_riversp_fetch import fetch_riversp
                res = fetch_riversp(
                    bbox_wesn=(float(w), float(e), float(s), float(n)),
                    start_date=str(getattr(args, "start_date", "")),
                    end_date=str(getattr(args, "end_date", "")),
                    cache_root=str(swot_cache_root),
                    product=str(getattr(args, "river_swot_product", "reach") or "reach"),
                    short_name=getattr(args, "river_swot_shortname", None),
                    logger=log,
                )
                if res.files:
                    args.river_swot_riversp = res.files
                    log.info(f"[SWOT] Auto-fetched RiverSP ({len(res.files)} file(s)) -> {res.cache_dir}")
                else:
                    log.info(f"[SWOT] RiverSP auto-fetch not used: {res.message}")
            else:
                log.debug("[SWOT] AOI bbox parse failed; skipping RiverSP auto-fetch")
    except Exception as _e:
        log.debug(f"[SWOT] RiverSP auto-fetch failed: {_e}")
    # Backwards-compat alias: --river-manning-enabled
    if getattr(args, "river_manning_enabled", False) and str(getattr(args, "river_manning_mode", "off")) == "off":
        args.river_manning_mode = "q2_regional"

    
    # ------------------------------------------------------------------
    # CUDEM / waffles cache
    # ------------------------------------------------------------------
    # Many CUDEM tools (e.g., waffles) use CUDEM_CACHE independent of this
    # pipeline's --cache-root. To keep runs reproducible and avoid failures
    # when the default cache directory doesn't exist, we scope CUDEM_CACHE
    # under --cache-root unless the user explicitly sets CUDEM_CACHE.
    try:
        _cc = os.environ.get("CUDEM_CACHE", "").strip()
        if not _cc:
            cudem_cache_dir = Path(args.cache_root) / "cudem_cache"
            ensure_dir(cudem_cache_dir)
            os.environ["CUDEM_CACHE"] = str(cudem_cache_dir)
            log.info("[CUDEM_CACHE] Using pipeline-scoped CUDEM cache: %s", cudem_cache_dir)
        else:
            ensure_dir(Path(_cc))
            log.info("[CUDEM_CACHE] Using existing CUDEM_CACHE: %s", _cc)
    except Exception as e:
        log.warning("[CUDEM_CACHE] Unable to prepare CUDEM cache directory: %s", e)
    cfg = BathyConfig(
            aoi=args.aoi,
            start_date=args.start_date,
            end_date=args.end_date,
            aoi_tile=getattr(args, "aoi_tile", None),
            tile_bbox=getattr(args, "tile_bbox", None),
            tile_buffer_km=float(getattr(args, "tile_buffer_km", 0.0) or 0.0),
            tile_edge_taper_enabled=bool(getattr(args, "tile_edge_taper_enabled", True)),
            tile_edge_taper_km=float(getattr(args, "tile_edge_taper_km", 0.0) or 0.0),
            tile_edge_smooth_sigma_km=float(getattr(args, "tile_edge_smooth_sigma_km", 0.0) or 0.0),
            tile_edge_metrics_band_km=float(getattr(args, "tile_edge_metrics_band_km", 0.0) or 0.0),
            sdb_model_bank_enabled=bool(getattr(args, "sdb_model_bank_enabled", True)),
            sdb_model_bank=str(getattr(args, "sdb_model_bank", "auto")),
            sdb_bank_max_samples=int(getattr(args, "sdb_bank_max_samples", 100000)),
            sdb_bank_seed=int(getattr(args, "sdb_bank_seed", 1337)),
            sdb_bank_retrain_min_new=int(getattr(args, "sdb_bank_retrain_min_new", 2000)),
            sdb_model_cache_enabled=bool(getattr(args, "sdb_model_cache_enabled", False)),
            sdb_model_cache_key=str(getattr(args, "sdb_model_cache_key", "auto")),
            out_dir=Path(args.out_dir).resolve(),
            methods=[m.strip().lower() for m in args.methods.split(",") if m.strip()],
            priority=args.priority,
    
            cloud=args.cloud,
            icesat=args.icesat,
            sdb_mode=args.sdb_mode,
            cache_root=Path(args.cache_root).resolve(),
            align_mode=args.align_mode,
            glint_correct=args.glint_correct,
            glint_nir_band=args.glint_nir_band,
            glint_vis_bands=args.glint_vis_bands,
            glint_nir_min_percentile=args.glint_nir_min_percentile,
            glint_deepwater_b02_max=args.glint_deepwater_b02_max,
            glint_min_samples=args.glint_min_samples,
            glint_max_samples=args.glint_max_samples,
            glint_clip_min=args.glint_clip_min,
            working_srs=args.working_srs,
            working_vcrs_epsg=args.working_vcrs_epsg,
            final_out_srs=args.final_out_srs,
            river_dem_auto=args.river_dem_auto,
            river_dem_source=args.river_dem_source,
            river_dem_res_m=args.river_dem_res_m,
            extra_xyz_crs=args.extra_xyz_crs,
    
            river_dem=Path(args.river_dem).resolve() if args.river_dem else None,
            river_soundings=None,
            river_prior_mode=args.river_prior_mode,
            river_mv_a0=args.river_mv_a0,
            river_mv_bw=args.river_mv_bw,
            river_mv_ba=args.river_mv_ba,
            river_mv_bs=args.river_mv_bs,
            river_mv_eps_a=args.river_mv_eps_a,
            river_mv_eps_s=args.river_mv_eps_s,
    
            river_slope_proxy_window=args.river_slope_proxy_window,
            river_slope_min=args.river_slope_min,
            river_slope_max=args.river_slope_max,
            river_slope_proxy_min_n=args.river_slope_proxy_min_n,
    
            river_wse_profile_enabled=bool(getattr(args, "river_wse_profile_enabled", True)),
            river_wse_profile_window=int(getattr(args, "river_wse_profile_window", args.river_slope_proxy_window)),
            river_wse_profile_min_n=int(getattr(args, "river_wse_profile_min_n", args.river_slope_proxy_min_n)),
            river_wse_profile_monotonic=bool(getattr(args, "river_wse_profile_monotonic", True)),
    
            river_usgs_sites=args.river_usgs_sites,
            river_usgs_start=args.river_usgs_start,
            river_usgs_end=args.river_usgs_end,
            river_usgs_cache_dir=Path(args.river_usgs_cache_dir) if args.river_usgs_cache_dir else None,
            river_usgs_max_dist_m=args.river_usgs_max_dist_m,
            river_usgs_mean_to_dmax=args.river_usgs_mean_to_dmax,
            river_usgs_a_stat=args.river_usgs_a_stat,
            river_usgs_q_quantile_lo=args.river_usgs_q_quantile_lo,
            river_usgs_q_quantile_hi=args.river_usgs_q_quantile_hi,
            river_usgs_a_cv_warn=args.river_usgs_a_cv_warn,
            river_usgs_width_ratio_max=args.river_usgs_width_ratio_max,
            river_usgs_width_ratio_blend=args.river_usgs_width_ratio_blend,
            river_gage_snap_max_dist_m=args.river_gage_snap_max_dist_m,
    
            river_width_stage_csv=args.river_width_stage_csv,
            river_width_stage_max_dist_m=args.river_width_stage_max_dist_m,
            river_width_stage_min_n=args.river_width_stage_min_n,
            river_width_stage_min_r2=args.river_width_stage_min_r2,
            river_width_stage_max_weight=args.river_width_stage_max_weight,
    
            river_hydrography_source=args.river_hydrography_source,
            tnm_enable=((args.river_hydrography_source in ('arcgis_tnm','tnm')) and (not args.no_tnm)),
            tnm_dataset=args.tnm_dataset,
            snap_m=args.snap_m,
    
    
            river_method=args.river_method,
            river_channel_buffer_m=args.river_channel_buffer_m,
            river_max_channel_width_m=args.river_max_channel_width_m,
            river_mainstem_min_order=args.river_mainstem_min_order,
            river_max_mainstem_width_m=args.river_max_mainstem_width_m,
            river_use_nhdarea=bool(getattr(args, "river_use_nhdarea", True)),
            river_nhdarea_layer=getattr(args, "river_nhdarea_layer", "nhdarea_clip"),
            river_shape_exp=args.river_shape_exp,
            river_dmax_min_m=args.river_dmax_min_m,
            river_dmax_max_m=args.river_dmax_max_m,
            river_bed_profile_max_slope=float(getattr(args,'river_bed_profile_max_slope',0.0) or 0.0),
            river_bed_profile_max_curv=float(getattr(args,'river_bed_profile_max_curv',0.0) or 0.0),
            river_bed_profile_step_m=float(getattr(args,'river_bed_profile_step_m',25.0) or 25.0),
            river_bed_profile_strength=float(getattr(args,'river_bed_profile_strength',0.6) or 0.6),
            river_bed_profile_power=float(getattr(args,'river_bed_profile_power',2.0) or 2.0),
            river_save_skeleton_debug=bool(args.river_save_skeleton_debug),
            river_skeleton_wse_mode=getattr(args, "river_skeleton_wse_mode", "bank"),
            river_skeleton_wse_smooth_sigma_m=float(getattr(args, "river_skeleton_wse_smooth_sigma_m", 0.0)),
            river_skeleton_junction_mode=str(getattr(args, "river_skeleton_junction_mode", "smooth")),
            river_skeleton_junction_buffer_m=float(getattr(args, "river_skeleton_junction_buffer_m", 120.0)),
            river_skeleton_junction_degree_min=int(getattr(args, "river_skeleton_junction_degree_min", 3)),
            river_skeleton_junction_smooth_sigma_m=float(getattr(args, "river_skeleton_junction_smooth_sigma_m", 80.0)),
            river_soundings_mode=args.river_soundings_mode,
            river_soundings_max_dist_m=args.river_soundings_max_dist_m,
            river_soundings_min_r=args.river_soundings_min_r,
            river_soundings_enforce=bool(getattr(args, 'river_soundings_enforce', True)),
    
            xs_spacing_m=args.xs_spacing_m,
            xs_length_m=args.xs_length_m,
            xs_smoothing_window_m=args.xs_smoothing_window_m,
            xs_trim_overlaps=bool(getattr(args, 'xs_trim_overlaps', True)),
            xs_global_deconflict=bool(getattr(args, 'xs_global_deconflict', True)),
            xs_deconflict_tol_m=args.xs_deconflict_tol_m,
            xs_skip_junctions=bool(getattr(args, 'xs_skip_junctions', True)),
            xs_junction_snap_m=args.xs_junction_snap_m,
            xs_junction_buffer_m=args.xs_junction_buffer_m,
            xs_densify_step_m=args.xs_densify_step_m,
            river_continuous=args.river_continuous,
            river_continuous_buffer_m=args.river_continuous_buffer_m,
            river_continuous_k=args.river_continuous_k,
            river_idw_power=args.river_idw_power,
            river_aniso_along_scale_m=args.river_aniso_along_scale_m,
            river_aniso_cross_scale_m=args.river_aniso_cross_scale_m,
            river_thalweg_weight=args.river_thalweg_weight,
            river_thalweg_only=bool(getattr(args,'river_thalweg_only', False)),
            river_thalweg_densify_factor=float(getattr(args,'river_thalweg_densify_factor', 0.5)),
            river_thalweg_densify_step_m=(None if getattr(args,'river_thalweg_densify_step_m', None) is None else float(getattr(args,'river_thalweg_densify_step_m'))),
            river_overlap_reducer=args.river_overlap_reducer,
            river_nodata=args.river_nodata,
            fusion_strategy=args.fusion_strategy,
            fusion_primary_weight=args.fusion_primary_weight,
            fusion_secondary_weight=args.fusion_secondary_weight,
            fusion_taper_m=args.fusion_taper_m,
            mask_river_to_waffles=(not args.no_mask_river_to_waffles),
    
            gapfill_enabled=bool(getattr(args, "gapfill_enabled", False)),
            gapfill_hq=list(getattr(args, "gapfill_hq", None)) if getattr(args, "gapfill_hq", None) else None,
            gapfill_water_mask=Path(getattr(args, "gapfill_water_mask", "")) if getattr(args, "gapfill_water_mask", None) else None,
            gapfill_method=getattr(args, "gapfill_method", "rbf"),
            gapfill_river_smooth_sigma_m=float(getattr(args, "gapfill_river_smooth_sigma", 500.0)),
            gapfill_prior_sigma_raster=Path(getattr(args, "gapfill_prior_sigma", "")) if getattr(args, "gapfill_prior_sigma", None) else None,
            gapfill_bank_elev_raster=Path(getattr(args, "gapfill_bank_elev", "")) if getattr(args, "gapfill_bank_elev", None) else None,
            gapfill_output_cudem_xyz=bool(getattr(args, "gapfill_cudem_xyz", False)),
            strict=args.strict,
    
        )
    # Resolve working CRS (used for river DEM and for meter-based thinning).
    cfg.working_srs = detect_working_srs(cfg)

    ensure_dir(cfg.out_dir)

    log.info("=" * 70)
    log.info("UNIFIED BATHYMETRY PIPELINE")
    log.info("=" * 70)
    log.info(f"AOI: {cfg.aoi}")
    log.info(f"Date range: {cfg.start_date} to {cfg.end_date}")
    log.info(f"Methods: {cfg.methods}")
    log.info(f"Priority: {cfg.priority}")
    log.info(f"Output: {cfg.out_dir}")
    log.info("=" * 70)

    # Optional: auto-download external soundings via CUDEM `dlim` providers.
    # This is a convenience wrapper around commands like:
    #   dlim -R="W/E/S/N" hydronos -P epsg:4269 > nos.xyz
    #   dlim -R="W/E/S/N" ehydro   -P epsg:4269 > usace.xyz
    # (Depth semantics are tracked via GeoTIFF metadata rather than via a compound CRS.)
    if getattr(args, "extra_xyz_cudem", None):
        requested = []
        for item in args.extra_xyz_cudem:
            if item is None:
                continue
            requested.extend([p.strip() for p in str(item).split(",") if p.strip()])

        if requested:
            # Determine output CRS for dlim.
            # For *depth* soundings we prefer a projected working CRS so thinning happens in meters.
            # We intentionally keep this **horizontal-only** (no +NAVD88) because depth is relative
            # to the water surface, not an orthometric height.
            #
            # Preference:
            #   1) user-specified --extra-xyz-cudem-crs
            #   2) working projected CRS (horizontal-only)
            #   3) fallback EPSG:4269 (horizontal-only)
            user_crs = str(getattr(args, "extra_xyz_cudem_crs", "")).strip().lower()
            if user_crs and user_crs not in ("", "auto"):
                dlim_crs = user_crs
            else:
                try:
                    dlim_crs = str(cfg.working_srs)
                except Exception:
                    dlim_crs = "epsg:4269"
            
            auto_xyz, auto_rep = fetch_cudem_soundings_via_dlim(
                aoi=cfg.aoi,
                sources=requested,
                cache_root=Path(cfg.cache_root),
                out_crs=dlim_crs,
                thin_res_m=float(getattr(args, "extra_xyz_cudem_thin_res_m", 10.0))
                    if float(getattr(args, "extra_xyz_cudem_thin_res_m", 10.0)) > 0 else None,
                filter_spec=getattr(args, "extra_xyz_cudem_filter", None),
                force=bool(getattr(args, "extra_xyz_cudem_force", False)),
            )
            # Attach to run report
            # (report is initialized a bit later; stash on args for now)
            setattr(args, "_xyz_auto_report", auto_rep)

            # Make these participate in the normal --extra-xyz normalization flow
            if auto_xyz:
                # Reproject XYZ files from dlim CRS (often epsg:4269) to working CRS
                # This ensures the soundings match the S2/working coordinate system
                # Use the already-resolved working CRS for consistency across modules.
                working_crs = str(cfg.working_srs)
                reprojected_xyz = []
                for xyz_path in auto_xyz:
                    try:
                        reproj_path = _reproject_xyz_file(
                            xyz_path, 
                            # Source CRS is the dlim output CRS (horizontal-only). Z is preserved.
                            src_crs=dlim_crs,  # e.g., EPSG:32616 or EPSG:4269
                            dst_crs=working_crs,  # e.g., EPSG:32616
                            cache_dir=Path(cfg.cache_root) / "xyz"
                        )
                        reprojected_xyz.append(reproj_path)
                        log.info("[XYZ][DLIM] Reprojected %s -> %s (%s)", 
                                xyz_path.name, reproj_path.name, working_crs)
                    except Exception as e:
                        log.warning("[XYZ][DLIM] Failed to reproject %s: %s. Using original.", xyz_path.name, e)
                        reprojected_xyz.append(xyz_path)
                
                if not getattr(args, "extra_xyz", None):
                    args.extra_xyz = []
                args.extra_xyz.extend([str(p) for p in reprojected_xyz])

                # Set the CRS to working CRS since we reprojected
                cfg.extra_xyz_crs = working_crs

    # Normalize extra XYZ inputs (files or directories). Prefer --extra-xyz; --river-soundings is a hidden alias.
    # Supports repeats, comma-separated lists, and directories.
    if getattr(args, "extra_xyz", None):
        xyz_files = []
        for item in args.extra_xyz:
            if item is None:
                continue
            parts = [p.strip() for p in str(item).split(",") if p.strip()]
            for part in parts:
                pth = Path(part)
                if pth.is_dir():
                    for ext in ("*.xyz", "*.csv", "*.txt", "*.dat", "*.gpkg", "*.shp", "*.geojson", "*.json"):
                        xyz_files.extend(sorted(pth.glob(ext)))
                else:
                    xyz_files.append(pth)

        # de-dup while preserving order
        seen = set()
        xyz_files2 = []
        for pth in xyz_files:
            sp = str(pth)
            if sp not in seen:
                seen.add(sp)
                xyz_files2.append(pth)

        if xyz_files2:
            cfg.river_soundings = ",".join(str(p) for p in xyz_files2)
            log.info("[XYZ] Using %d external bathymetry file(s):", len(xyz_files2))
            for pth in xyz_files2:
                log.info("   - %s", str(pth))
    # If glint correction was requested but SDB isn't being run, accept the flag but ignore it.
    if getattr(cfg, "glint_correct", False) and ("sdb" not in cfg.methods):
        log.info("[GLINT] --glint-correct set, but methods does not include 'sdb'; ignoring glint options for this run.")
        cfg.glint_correct = False

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pipeline_version": getattr(constants, "PIPELINE_VERSION", "unknown"),
        "config": {
            "aoi": cfg.aoi,
            "aoi_tile": getattr(cfg, "aoi_tile", None),
            "tile_bbox": list(getattr(cfg, "tile_bbox", None)) if getattr(cfg, "tile_bbox", None) else None,
            "tile_buffer_km": float(getattr(cfg, "tile_buffer_km", 0.0) or 0.0),
            "tile_edge_taper_enabled": bool(getattr(cfg, "tile_edge_taper_enabled", True)),
            "tile_edge_taper_km": float(getattr(cfg, "tile_edge_taper_km", 0.0) or 0.0),
            "tile_edge_smooth_sigma_km": float(getattr(cfg, "tile_edge_smooth_sigma_km", 0.0) or 0.0),
            "sdb_model_bank_enabled": bool(getattr(cfg, "sdb_model_bank_enabled", True)),
            "sdb_model_bank": str(getattr(cfg, "sdb_model_bank", "auto")),
            "sdb_bank_max_samples": int(getattr(cfg, "sdb_bank_max_samples", 100000)),
            "sdb_bank_seed": int(getattr(cfg, "sdb_bank_seed", 1337)),
            "sdb_bank_retrain_min_new": int(getattr(cfg, "sdb_bank_retrain_min_new", 2000)),
            "sdb_model_cache_enabled": bool(getattr(cfg, "sdb_model_cache_enabled", False)),
            "sdb_model_cache_key": str(getattr(cfg, "sdb_model_cache_key", "auto")),
            "start": cfg.start_date,
            "end": cfg.end_date,
            "methods": cfg.methods,
            "priority": cfg.priority,
            "out_dir": str(cfg.out_dir),
            "river_dem": str(cfg.river_dem) if cfg.river_dem else None,
            "working_srs": str(cfg.working_srs),
            "final_out_srs": str(cfg.final_out_srs),
            "depth_value_type": "depth",
            "depth_units": "m",
            "depth_sign": "negative_down",
            "depth_reference": "water_surface",
        }
    }
    # Include any auto-fetched CUDEM soundings in the run report
    if hasattr(args, "_xyz_auto_report"):
        report["xyz_auto"] = getattr(args, "_xyz_auto_report")

    sdb_raster = None
    river_raster = None

    if "sdb" in cfg.methods:
        sdb_raster = run_sdb(cfg, report)

    if "river" in cfg.methods:
        river_raster = run_river(cfg, report)

    final = fuse(cfg, sdb_raster, river_raster, report)

    # Optional intelligent gap filling (Tier 1–2): prior + residual interpolation
    if cfg.gapfill_enabled and final:
        try:
            from gapfill_intelligent import gapfill_depth_raster, GapfillConfig

            # HQ points: prefer explicit --gapfill-hq; otherwise fall back to --extra-xyz (if present).
            hq_files: List[str] = []
            if cfg.gapfill_hq:
                hq_files = [str(p) for p in cfg.gapfill_hq if p]
            else:
                # args.extra_xyz may not exist in older CLI variants
                ex = getattr(args, "extra_xyz", None)
                if ex:
                    if isinstance(ex, (list, tuple)):
                        hq_files = [str(p) for p in ex if p]
                    else:
                        hq_files = [str(ex)]

            if not hq_files:
                log.warning("[GAPFILL] Enabled but no HQ points provided (--gapfill-hq or --extra-xyz). Skipping.")
            else:
                combined_dir = ensure_dir(cfg.out_dir / "combined")
                out_gap = combined_dir / "bathy_combined_depth_gapfill.tif"
                out_sig = combined_dir / "bathy_combined_depth_gapfill_sigma.tif"
                out_prov = combined_dir / "bathy_combined_depth_gapfill_provenance.tif"

                # Water mask preference order:
                #   1) explicit --gapfill-water-mask
                #   2) waffles coastline mask (most recent in cache)
                #   3) implicit: finite pixels in prior
                wm: Optional[Path] = None
                if cfg.gapfill_water_mask and Path(cfg.gapfill_water_mask).exists():
                    wm = Path(cfg.gapfill_water_mask)
                else:
                    wm = _find_latest_waffles_mask(cfg.cache_root)

                # River aids (optional): xs gpkg and river corridor mask from river output.
                xs_gpkg = cfg.out_dir / "river" / "river_xs_params.gpkg"
                if not xs_gpkg.exists():
                    xs_gpkg = None
                river_mask = None
                if river_raster and Path(river_raster).exists():
                    river_mask = Path(river_raster)

                # Build GapfillConfig from pipeline config
                gcfg = GapfillConfig(
                    interpolation_method=cfg.gapfill_method,
                    rbf_function="thin_plate",
                    prior_sigma_default=1.0,
                    rbf_min_points=25,
                    component_min_points=10,
                    river_smooth_sigma_m=cfg.gapfill_river_smooth_sigma_m,
                    residual_sigma_floor=0.25,
                    distance_sigma_scale_m=1500.0,
                    output_cudem_xyz=cfg.gapfill_output_cudem_xyz,
                )
                
                # Call the gap-fill function
                gapfill_stats = gapfill_depth_raster(
                    prior_raster=Path(final),
                    hq_point_files=hq_files,
                    out_raster=out_gap,
                    out_sigma=out_sig,
                    out_provenance=out_prov,
                    cfg=gcfg,
                    prior_sigma_raster=cfg.gapfill_prior_sigma_raster,
                    water_mask_raster=wm,
                    river_mask_raster=river_mask,
                    bank_elev_raster=cfg.gapfill_bank_elev_raster,
                    xs_params_gpkg=xs_gpkg,
                    logger=log,
                )

                report.setdefault("gapfill", {})["status"] = "success"
                report["gapfill"]["stats"] = gapfill_stats
                report["gapfill"]["outputs"] = {
                    "depth": str(out_gap),
                    "sigma": str(out_sig),
                    "provenance": str(out_prov),
                }
                if cfg.gapfill_output_cudem_xyz and "cudem_xyz" in gapfill_stats:
                    report["gapfill"]["outputs"]["cudem_xyz"] = gapfill_stats["cudem_xyz"]
                final = out_gap
        except Exception as e:
            report.setdefault("gapfill", {})["status"] = "failed"
            report["gapfill"]["error"] = str(e)
            log.warning("[GAPFILL] Failed: %s", e)

    # Prefer reporting the horizontal-only final outputs when available
    final_for_user = final

    # Default: also emit reprojected rasters in final_out_srs (horizontal-only; depth metadata tags included).
    try:
        dst_srs = cfg.final_out_srs
        if final:
            final_p = Path(final)
            combined_dir = ensure_dir(cfg.out_dir / "combined")
            warped = warp_raster_to_srs(final_p, combined_dir / "bathy_depth_final_epsg4269.tif", dst_srs)
            if warped:
                report.setdefault("outputs", {})["combined_warped"] = str(warped)
                final_for_user = str(warped)

                # Optional "root copy" for convenience
                try:
                    root_copy = cfg.out_dir / "bathy_depth_final_epsg4269.tif"
                    if Path(warped) != root_copy:
                        shutil.copy2(str(warped), str(root_copy))
                        report.setdefault("outputs", {})["combined_root_copy"] = str(root_copy)
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

            # Final domain clipping is handled by _apply_final_domain_policy() after pipeline completion.
            # Optional extra crop to AOI bbox (acts as a hard safety-net when upstream masks don't cover full grid).
            try:
                bbox = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                if bbox and str(dst_srs).upper().endswith("4269"):
                    _crop_raster_extent_to_bbox(Path(warped), bbox, nodata=float(getattr(cfg, 'final_nodata', -9999.0)))
                    try:
                        root_copy = cfg.out_dir / "bathy_depth_final_epsg4269.tif"
                        if root_copy.exists():
                            _crop_raster_extent_to_bbox(root_copy, bbox, nodata=float(getattr(cfg, 'final_nodata', -9999.0)))
                    except Exception:
                        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                    # Seam-stability edge taper (operational tiling)
                    try:
                        if bool(getattr(cfg, "tile_edge_taper_enabled", True)) and float(getattr(cfg, "tile_edge_taper_km", 0.0) or 0.0) > 0:
                            m = _compute_edge_band_metrics_epsg4269(
                                Path(warped),
                                bbox,
                                band_km=float(getattr(cfg, "tile_edge_metrics_band_km", 2.0) or 2.0),
                                smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                nodata=float(getattr(cfg, "final_nodata", -9999.0)),
                            )
                            report.setdefault("seams", {})["combined_edge_metrics"] = m
                            tapered = _apply_tile_edge_taper_epsg4269(
                                Path(warped),
                                bbox,
                                taper_km=float(getattr(cfg, "tile_edge_taper_km", 2.0) or 2.0),
                                smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                nodata=float(getattr(cfg, "final_nodata", -9999.0)),
                            )
                            if tapered and Path(tapered).exists():
                                # Replace deliverable with tapered version
                                shutil.move(str(tapered), str(Path(warped)))
                                report.setdefault("outputs", {})["combined_edge_tapered"] = str(Path(warped))
                                try:
                                    root_copy = cfg.out_dir / "bathy_depth_final_epsg4269.tif"
                                    if root_copy.exists():
                                        shutil.copy2(str(Path(warped)), str(root_copy))
                                except Exception:
                                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                    except Exception:
                        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


        if sdb_raster:
            sdb_p = Path(sdb_raster)
            sdb_dir = ensure_dir(cfg.out_dir / "sdb")
            warped = warp_raster_to_srs(sdb_p, sdb_dir / "sdb_depth_final_epsg4269.tif", dst_srs)
            if warped:
                report.setdefault("outputs", {})["sdb_warped"] = str(warped)

        if river_raster:
            r_p = Path(river_raster)
            river_dir = ensure_dir(cfg.out_dir / "river")
            warped = warp_raster_to_srs(r_p, river_dir / "river_depth_final_epsg4269.tif", dst_srs)
            if warped:
                report.setdefault("outputs", {})["river_warped"] = str(warped)

                # Final safety clip: keep river outputs inside waffles water mask (water=0, land=1).
                try:
                    wm = None
                    ro = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
                    wm = ro.get("waffles_water_mask")
                    if wm and Path(wm).exists():
                        _clip_raster_to_mask_reproject(Path(warped), Path(wm), inside_value=0, invert=False, nodata=float(getattr(cfg, 'river_nodata', -9999.0)))
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                # Optional crop to tile AOI bbox (EPSG:4269 only) + edge seam taper
                try:
                    bbox = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                    if bbox and str(dst_srs).upper().endswith("4269"):
                        _crop_raster_extent_to_bbox(Path(warped), bbox, nodata=float(getattr(cfg, 'river_nodata', -9999.0)))
                        if bool(getattr(cfg, "tile_edge_taper_enabled", True)) and float(getattr(cfg, "tile_edge_taper_km", 0.0) or 0.0) > 0:
                            m = _compute_edge_band_metrics_epsg4269(
                                Path(warped),
                                bbox,
                                band_km=float(getattr(cfg, "tile_edge_metrics_band_km", 2.0) or 2.0),
                                smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                            )
                            report.setdefault("seams", {})["river_edge_metrics"] = m
                            tapered = _apply_tile_edge_taper_epsg4269(
                                Path(warped),
                                bbox,
                                taper_km=float(getattr(cfg, "tile_edge_taper_km", 2.0) or 2.0),
                                smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                            )
                            if tapered and Path(tapered).exists():
                                shutil.move(str(tapered), str(Path(warped)))
                                report.setdefault("outputs", {})["river_edge_tapered"] = str(Path(warped))
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

            # Also warp river bottom elevation (orthometric heights, typically NAVD88) if available.
            try:
                river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
                bed_src = river_outputs.get("bottom_elevation")
                if bed_src and Path(bed_src).exists():
                    bed_p = Path(bed_src)
                    bed_warp = warp_raster_to_srs(
                        bed_p,
                        river_dir / "river_bottom_navd88_final_epsg4269.tif",
                        dst_srs,
                        write_depth_metadata=False,
                    )
                    if bed_warp:
                        report.setdefault("outputs", {})["river_bottom_warped"] = str(bed_warp)
                        try:
                            apply_elevation_metadata(Path(bed_warp), vertical_datum="NAVD88")
                        except Exception:
                            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

                        # Final safety clip: keep river bottom inside waffles water mask (water=0, land=1).
                        try:
                            wm = None
                            ro = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
                            wm = ro.get("waffles_water_mask")
                            if wm and Path(wm).exists():
                                _clip_raster_to_mask_reproject(Path(bed_warp), Path(wm), inside_value=0, invert=False, nodata=float(getattr(cfg, 'river_nodata', -9999.0)))
                        except Exception:
                            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

                        # Optional extra crop to AOI bbox (EPSG:4269 only).
                        try:
                            bbox = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                            if bbox and str(dst_srs).upper().endswith("4269"):
                                _crop_raster_extent_to_bbox(Path(bed_warp), bbox, nodata=float(getattr(cfg, 'river_nodata', -9999.0)))
                                if bool(getattr(cfg, "tile_edge_taper_enabled", True)) and float(getattr(cfg, "tile_edge_taper_km", 0.0) or 0.0) > 0:
                                    m = _compute_edge_band_metrics_epsg4269(
                                        Path(bed_warp),
                                        bbox,
                                        band_km=float(getattr(cfg, "tile_edge_metrics_band_km", 2.0) or 2.0),
                                        smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                        nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                                    )
                                    report.setdefault("seams", {})["river_bottom_edge_metrics"] = m
                                    tapered = _apply_tile_edge_taper_epsg4269(
                                        Path(bed_warp),
                                        bbox,
                                        taper_km=float(getattr(cfg, "tile_edge_taper_km", 2.0) or 2.0),
                                        smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                        nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                                    )
                                    if tapered and Path(tapered).exists():
                                        shutil.move(str(tapered), str(Path(bed_warp)))
                                        report.setdefault("outputs", {})["river_bottom_edge_tapered"] = str(Path(bed_warp))
                        except Exception:
                            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        # Optional: combined bottom elevation (NAVD88) for river + SDB where possible.
        # River provides bottom directly; SDB requires a WSE(NAVD88) raster to convert depth->bottom.
        try:
            import rasterio
            import numpy as np

            out_combined_bottom = ensure_dir(cfg.out_dir / "combined") / "bathy_bottom_navd88_final_epsg4269.tif"

            river_bottom = report.get("outputs", {}).get("river_bottom_warped")
            sdb_depth = report.get("outputs", {}).get("sdb_warped")

            # Prefer a WSE raster if the SDB output folder contains one.
            sdb_wse = None
            try:
                sdb_out_dir = Path(cfg.out_dir) / "sdb"
                sdb_wse = find_sdb_wse_navd88_raster(sdb_out_dir)
            except Exception:
                sdb_wse = None

            if river_bottom and Path(river_bottom).exists():
                with rasterio.open(river_bottom) as rb:
                    prof = rb.profile.copy()
                    prof.update(dtype="float32", count=1, nodata=float(getattr(cfg, 'river_nodata', -9999.0)), compress="deflate")
                    out_arr = rb.read(1).astype("float32")
                    nod = float(prof.get("nodata", -9999.0))

                # If we can convert SDB depth to bottom, fill remaining nodata outside river.
                if sdb_depth and Path(sdb_depth).exists() and sdb_wse and Path(sdb_wse).exists():
                    with rasterio.open(sdb_depth) as sd, rasterio.open(sdb_wse) as sw:
                        sd_transform = sd.transform
                        sd_crs = sd.crs
                        # Reproject WSE to match depth grid if needed
                        wse = np.zeros((sd.height, sd.width), dtype=np.float32)
                        from rasterio.warp import reproject, Resampling
                        reproject(
                            source=sw.read(1),
                            destination=wse,
                            src_transform=sw.transform,
                            src_crs=sw.crs,
                            dst_transform=sd.transform,
                            dst_crs=sd.crs,
                            resampling=Resampling.bilinear,
                        )
                        depth = sd.read(1).astype("float32")
                        dnod = float(sd.nodata) if sd.nodata is not None else -9999.0
                        # Depth is negative-down in this pipeline; convert to positive-down magnitude
                        depth_mag = np.where(np.isfinite(depth) & (depth != dnod), np.abs(depth), np.nan)
                        bottom_sdb = wse - depth_mag

                    # Warp SDB bottom to match river_bottom grid
                    with rasterio.open(river_bottom) as rb:
                        bottom_sdb_on_rb = np.full((rb.height, rb.width), np.nan, dtype=np.float32)
                        reproject(
                            source=bottom_sdb,
                            destination=bottom_sdb_on_rb,
                            src_transform=sd_transform,
                            src_crs=sd_crs,
                            dst_transform=rb.transform,
                            dst_crs=rb.crs,
                            resampling=Resampling.bilinear,
                        )

                    fill = (out_arr == nod) & np.isfinite(bottom_sdb_on_rb)
                    if np.any(fill):
                        out_arr[fill] = bottom_sdb_on_rb[fill]

                with rasterio.open(out_combined_bottom, "w", **prof) as dst:
                    dst.write(out_arr.astype("float32"), 1)

                try:
                    apply_elevation_metadata(out_combined_bottom, vertical_datum="NAVD88")
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

                # Optional crop/taper to tile AOI for combined bottom output
                try:
                  bbox = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                  if bbox and str(dst_srs).upper().endswith("4269"):
                      _crop_raster_extent_to_bbox(out_combined_bottom, bbox, nodata=float(getattr(cfg, 'river_nodata', -9999.0)))
                      if bool(getattr(cfg, "tile_edge_taper_enabled", True)) and float(getattr(cfg, "tile_edge_taper_km", 0.0) or 0.0) > 0:
                          m = _compute_edge_band_metrics_epsg4269(
                              out_combined_bottom,
                              bbox,
                              band_km=float(getattr(cfg, "tile_edge_metrics_band_km", 2.0) or 2.0),
                              smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                              nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                          )
                          report.setdefault("seams", {})["combined_bottom_edge_metrics"] = m
                          tapered = _apply_tile_edge_taper_epsg4269(
                              out_combined_bottom,
                              bbox,
                              taper_km=float(getattr(cfg, "tile_edge_taper_km", 2.0) or 2.0),
                              smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                              nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                          )
                          if tapered and Path(tapered).exists():
                              shutil.move(str(tapered), str(out_combined_bottom))
                              report.setdefault("outputs", {})["combined_bottom_edge_tapered"] = str(out_combined_bottom)
                except Exception:
                  logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                report.setdefault("outputs", {})["combined_bottom_navd88"] = str(out_combined_bottom)

        except Exception:
            # Optional output only; do not fail the run if conversion inputs are absent.
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    except Exception as e:
        log.warning("[WARP] Could not create final reprojected rasters: %s", e)


    report_path = cfg.out_dir / "bathy_report.json"
    write_json(report_path, report)
    try:
        from flight_recorder import emit_artifact_written
        emit_artifact_written(report_path, kind="json", role="bathy_report")
        # Also emit all artifacts we know about from the report (so summaries can be derived from the flight recorder)
        _emit_artifacts_from_report(report)
    except Exception as e:
        logging.getLogger(__name__).debug("Optional flight-recorder emit failed: %s", e)
    log.info(f"Report written: {report_path}")
    
    # NEW v0.7.1: Create unified bathymetry report
    try:
        from river_diagnostics import create_unified_bathy_report
        
        sdb_out = cfg.out_dir / "sdb" if "sdb" in cfg.methods else None
        river_out = cfg.out_dir / "river" if "river" in cfg.methods else None
        
        unified_path = create_unified_bathy_report(
            output_dir=cfg.out_dir,
            sdb_output=sdb_out,
            river_output=river_out,
            methods=cfg.methods,
            priority=cfg.priority
        )
        log.info(f"Unified report written: {unified_path}")
    except ImportError:
        log.warning("river_diagnostics module not available - unified report not created")
    except Exception as e:
        log.warning(f"Could not create unified report: {e}")

    # Human-friendly, scan-friendly console summary
    try:
        from run_summary import print_human_run_summary

        summary_stats = {
            "command": " ".join(sys.argv),
            "aoi": cfg.aoi,
            "time_window": {"start": cfg.start, "end": cfg.end},
            "methods": list(cfg.methods),
            "priority": cfg.priority,
            "outputs": report.get("outputs", {}),
        }
        print_human_run_summary(summary_stats, log_fn=log.info)
        report["human_summary"] = summary_stats
    except Exception as e:
        log.debug(f"Human summary skipped: {e}")


    # Write detailed run summaries (technical/scientific/human) using the in-memory report
    try:
        from run_summary import write_run_summary_files
        from flight_recorder import current_run_id, current_flight_path

        rid = current_run_id()
        frp = current_flight_path()
        write_run_summary_files(cfg.out_dir, run_id=rid, stats=report, fr_path=(frp or None))
        log.info(f"Run summaries written under: {Path(cfg.out_dir) / 'run_logs'}")
    except Exception as e:
        log.debug(f"Run summary file write skipped: {e}")



    # Strict mode: run contract tests + basic output sanity checks
    if cfg.strict:
        try:
            from contract_tests import run_contract_tests_cli

            rc = run_contract_tests_cli(cfg.out_dir)
            if rc != 0:
                raise RuntimeError(f"Contract tests failed (exit={rc})")
        except Exception as e:
            log.error("[STRICT] %s", e)
            raise

        # Sanity check: final raster should exist and contain at least some finite pixels.
        try:
            import rasterio
            import numpy as np

            final_path = None
            if final_for_user and Path(final_for_user).exists():
                final_path = Path(final_for_user)
            elif final and Path(final).exists():
                final_path = Path(final)

            if final_path is None or not final_path.exists():
                raise RuntimeError("Final depth raster missing")

            with rasterio.open(final_path) as ds:
                arr = ds.read(1, masked=True)
                finite = np.isfinite(arr.filled(np.nan))
                n_finite = int(np.sum(finite))
                if n_finite < 100:
                    raise RuntimeError(f"Final raster has too few valid pixels (n_valid={n_finite})")
                report.setdefault("sanity", {})["final_valid_pixels"] = n_finite
        except Exception as e:
            log.error("[STRICT][SANITY] %s", e)
            raise



    # Output contract: ensure standard filenames exist (metrics/debug rely on these)
    try:
        _ensure_output_contract(
            Path(getattr(args, "out_dir")),
            Path(getattr(args, "cache_root", Path(getattr(args, "out_dir")) / "cache")),
        )
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    # Final domain policy: clip final products based on enabled methods
    try:
        _apply_final_domain_policy(cfg, cfg.out_dir, cfg.cache_root)
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 70)
    log.info(f"SDB: {report.get('sdb', {}).get('status', 'skipped')}")
    log.info(f"River: {report.get('river', {}).get('status', 'skipped')}")
    log.info(f"Fusion: {report.get('fusion', {}).get('status', 'skipped')}")
    log.info(f"Final output: {final_for_user if final_for_user else (final if final else 'None')}")
    log.info("=" * 70)

    return 0 if final else 2


if __name__ == "__main__":
    raise SystemExit(main())
