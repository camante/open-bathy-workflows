"""Raster operations used by the unified bathymetry workflow.

Phase 2B anti-soup refactor: lift-and-shift raster utilities out of
`bathy_main.py` into a focused module.

No behavior changes intended (only relocation).
"""

from __future__ import annotations

import shutil
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.paths import ensure_dir
from process_utils import run_cmd

log = logging.getLogger(__name__)


def _raster_has_valid_pixels(p: Path, max_sample_pixels: int = 250000) -> bool:
    """Return True if raster contains any non-nodata pixels.

    We use this to avoid caching 'successful' warp outputs that are actually
    all nodata (e.g., due to earlier bad nodata handling or sparse masks).
    """
    try:
        import rasterio
        import numpy as np
        with rasterio.open(p) as ds:
            nodata = ds.nodata
            # If nodata is undefined, we still need to guard against NaN-only rasters.
            # Treat non-finite values as invalid.
            # Read a subsampled window if raster is large.
            w = ds.width
            h = ds.height
            n = w * h
            if n <= max_sample_pixels:
                arr = ds.read(1, masked=False)
                finite = np.isfinite(arr)
                if nodata is None:
                    return bool(np.any(finite))
                return bool(np.any(finite & (arr != nodata)))
            # Subsample by stride
            stride = int(math.ceil(math.sqrt(n / max_sample_pixels)))
            arr = ds.read(1, window=((0, h, stride), (0, w, stride)), masked=False)
            finite = np.isfinite(arr)
            if nodata is None:
                return bool(np.any(finite))
            return bool(np.any(finite & (arr != nodata)))
    except Exception:
        # Best-effort: if we cannot read, don't block the pipeline.
        return True


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
        # Treat mask nodata / areas outside coverage as OUTSIDE the mask.
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
        log.debug("ignored", exc_info=True)

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
            # Initialize destination to LAND(1) so pixels outside the reprojected
            # mask footprint remain LAND instead of uninitialized (np.empty()).
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
                log.debug("ignored", exc_info=True)

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
            log.debug("ignored", exc_info=True)
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
        log.debug("ignored", exc_info=True)

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


def warp_raster_to_srs(
    in_raster: Path,
    out_raster: Path,
    dst_srs: str,
    write_depth_metadata: bool = True,
    resample: str = "near",
    reuse_existing: bool = False,
) -> Optional[Path]:
    """Warp a raster to dst_srs (best-effort).

    Notes on resampling:
      * Many of our intermediate/final products are sparse (valid pixels only
        inside masks such as NHDArea / channel corridors).
      * GDAL's bilinear/cubic resampling will output nodata if any contributing
        source pixel is nodata; for very sparse rasters this can produce outputs
        with 0 valid pixels after reprojection.
      * Default to nearest-neighbour ("near") to preserve sparse valid pixels.
        Downstream smoothing/resampling can be applied where appropriate.
    """
    gdalwarp = shutil.which("gdalwarp")
    if gdalwarp is None:
        log.error("gdalwarp not found; cannot reproject %s", in_raster)
        return None
    if out_raster.exists() and out_raster.stat().st_size > 0:
        if reuse_existing and _raster_has_valid_pixels(out_raster):
            if write_depth_metadata:
                apply_depth_metadata(out_raster)
            return out_raster
        # Always overwrite stale/invalid derived outputs unless reuse_existing=True
        for fp in [out_raster, out_raster.with_suffix(out_raster.suffix + ".aux.xml")]:
            try:
                if fp.exists():
                    fp.unlink()
            except Exception:
                log.debug("ignored", exc_info=True)


    # Explicitly propagate nodata through warps.
    # If src/dst nodata are not set, GDAL can yield rasters filled with
    # float32 max (3.402823466e+38) which QGIS shows as huge +/- values.
    src_nodata = None
    try:
        import rasterio
        with rasterio.open(in_raster) as src:
            src_nodata = src.nodata
    except Exception:
        src_nodata = None

    # Fallback nodata for float outputs (depth/bed rasters). Keep conservative.
    nodata = float(src_nodata) if src_nodata is not None else -9999.0

    cmd = [
        gdalwarp, "-overwrite",
        "-t_srs", str(dst_srs),
        "-r", str(resample),
        "-dstnodata", str(nodata),
        "-wo", "INIT_DEST=NO_DATA",
        "-multi",
        "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
    ]
    # Only pass -srcnodata if the source explicitly declares nodata.
    # Otherwise, forcing a guessed -srcnodata can incorrectly mask all data.
    if src_nodata is not None:
        cmd.extend(["-srcnodata", str(nodata)])
    cmd.extend([str(in_raster), str(out_raster)])
    try:
        log.info("%s -> %s (%s)", in_raster.name, out_raster.name, dst_srs)
        run_cmd(cmd, check=True)
        if out_raster.exists() and write_depth_metadata:
            apply_depth_metadata(out_raster)
        return out_raster if out_raster.exists() else None
    except Exception as e:
        log.warning("gdalwarp failed: %s", e)
        return None