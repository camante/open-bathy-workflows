#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
river_domain_mask.py — Build a *river-only* raster mask (river vs. open water) using:
  1) NHD flowlines (from river_network.gpkg) as the river "seed", and
  2) a water/land mask (ideally waffles coastline mask: water=0, land=1) as the boundary.

This replaces the fragile "cross-sections" notion of river geometry with a robust raster domain:
we only generate river bathymetry where pixels are:
  - water, AND
  - within a buffered corridor around NHD flowlines, AND
  - not wider than a configurable threshold (prevents filling bays/ocean).

Outputs:
  - river_channel_mask.tif : 1=river channel domain, 0=everything else
  - open_water_mask.tif    : 1=water but NOT river channel domain (ocean/bay), 0=else

Notes:
- If no --water-mask is supplied, we fall back to treating the buffered corridor as "water".
  This is less reliable (no ocean separation), but keeps the pipeline runnable.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt

LOG = logging.getLogger("river_domain_mask")


def _pixel_size_m(transform: rasterio.Affine) -> float:
    # assume square-ish pixels
    return float((abs(transform.a) + abs(transform.e)) / 2.0)


def _read_template(template_raster: Path) -> Tuple[dict, rasterio.Affine, rasterio.crs.CRS, Tuple[int, int]]:
    with rasterio.open(template_raster) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
    return profile, transform, crs, shape


def _warp_mask_to_template(mask_path: Path, template_profile: dict) -> np.ndarray:
    """Warp an input mask raster to the template grid using nearest neighbor."""
    dst = np.ones((template_profile["height"], template_profile["width"]), dtype=np.uint8)  # default to land
    with rasterio.open(mask_path) as src:
        src_arr = src.read(1)

        # IMPORTANT:
        # Some mask rasters use semantic values (0/1) for water/land and may also
        # set nodata to 0 or 1. If we pass that through as src_nodata, rasterio
        # will treat real water/land as nodata and destroy the mask.
        src_nodata = src.nodata
        if src_nodata in (0, 1):
            src_nodata = None

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=Resampling.nearest,
            src_nodata=src_nodata,
            dst_nodata=1,  # land by default
        )
    return dst


def _save_u8(path: Path, arr_u8: np.ndarray, template_profile: dict, nodata: int = 0) -> None:
    profile = template_profile.copy()
    profile.update(dtype="uint8", count=1, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr_u8.astype("uint8"), 1)


def _save_f32(path: Path, arr_f32: np.ndarray, template_profile: dict, nodata: float = -9999.0) -> None:
    profile = template_profile.copy()
    profile.update(dtype="float32", count=1, nodata=nodata, compress="deflate")
    out = np.where(np.isfinite(arr_f32), arr_f32, nodata).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def _get_stream_order_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    candidates = ["streamorde", "StreamOrde", "STREAMORDE", "streamorde_", "stream_order", "streamorder", "Strahler"]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Build river-only raster mask from NHD flowlines + water mask.")
    p.add_argument("--river-gpkg", required=True, help="river_network.gpkg from river_network.py")
    p.add_argument("--rivers-layer", default="rivers_clip", help="Layer name inside gpkg (default rivers_clip)")
    p.add_argument("--template-raster", required=True, help="Template raster defining output grid (e.g., river_dem.tif)")
    p.add_argument("--water-mask", default=None,
                   help="Optional water/land mask raster. For waffles mask: water=0 land=1.")
    p.add_argument("--ocean-mask", default=None,
                   help="Optional ocean-only waffles coastline mask (land=1, water=0). Used to prevent ocean bleed and to build fallback river domain when NHD water mask is unavailable.")
    p.add_argument("--ocean-keep-dist-m", type=float, default=0.0,
                   help="Allow ocean-connected water within this distance (meters) of a flowline centerline. Useful for tidal river mouths/estuaries where the mainstem is classified as ocean water. 0 disables.")
    p.add_argument("--nhdarea-gpkg", default=None,
                   help="Optional GeoPackage with NHDArea polygons (e.g., river_network.gpkg). If present, used to constrain the river channel mask.")
    p.add_argument("--nhdarea-layer", default="nhdarea_clip",
                   help="Layer name for NHDArea polygons inside --nhdarea-gpkg (default nhdarea_clip).")
    p.add_argument("--channel-buffer-m", type=float, default=400.0,
                   help="Buffer around flowlines to define candidate corridor (meters).")
    p.add_argument("--max-channel-width-m", type=float, default=600.0,
                   help="Max allowed width inside corridor (meters).")
    p.add_argument("--mainstem-min-order", type=int, default=5,
                   help="Stream order threshold for mainstem corridor (if stream order attribute exists).")
    p.add_argument("--max-mainstem-width-m", type=float, default=2500.0,
                   help="Max allowed width for mainstem corridor pixels (meters).")
    p.add_argument("--write-debug", action="store_true", default=False,
                   help="Write debug rasters (width proxy, corridor masks).")
    p.add_argument("--out-channel-mask", required=True, help="Output TIFF for river channel mask (1=river,0=else).")
    p.add_argument("--out-open-water-mask", required=True, help="Output TIFF for open water mask (1=open water,0=else).")

    args = p.parse_args()

    template_profile, transform, crs, shape = _read_template(Path(args.template_raster))
    template_profile["transform"] = transform
    template_profile["crs"] = crs

    pix = _pixel_size_m(transform)
    LOG.info("Template grid: %s x %s | pixel_size≈%.3fm", shape[1], shape[0], pix)

    # This script assumes the template CRS is projected in meters. If it is geographic
    # (degrees), buffers and distances will be wrong. We still run (best-effort) but
    # warn loudly because the resulting mask will be unreliable.
    try:
        if crs is not None and (not crs.is_projected):
            LOG.warning(
                "Template CRS is not projected (crs=%s). Buffer widths are interpreted as degrees, not meters; "
                "river_domain_mask outputs may be incorrect. Use a projected river_dem/template raster.",
                str(crs),
            )
    except Exception:
        pass

    # Load rivers layer
    rivers = gpd.read_file(args.river_gpkg, layer=args.rivers_layer)
    if rivers.empty:
        raise SystemExit(f"No features found in {args.river_gpkg}:{args.rivers_layer}")
    rivers = rivers.to_crs(crs)

    # Optional: NHDArea polygons to constrain channel predictions
    nhdarea_mask = None
    if args.nhdarea_gpkg:
        try:
            areas = gpd.read_file(args.nhdarea_gpkg, layer=args.nhdarea_layer)
            if areas is not None and not areas.empty:
                areas = areas.to_crs(crs)
                areas = areas[areas.geometry.notnull() & (~areas.geometry.is_empty)]
                areas = areas[areas.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                if not areas.empty:
                    area_geom = areas.geometry.unary_union
                    nhdarea_mask = rasterize(
                        [(area_geom, 1)],
                        out_shape=shape,
                        transform=transform,
                        fill=0,
                        all_touched=True,
                        dtype="uint8",
                    ).astype(bool)
                    LOG.info("NHDArea mask enabled: %s:%s (pixels=%d)", args.nhdarea_gpkg, args.nhdarea_layer, int(nhdarea_mask.sum()))
                else:
                    LOG.info("NHDArea layer has no polygon geometries: %s:%s", args.nhdarea_gpkg, args.nhdarea_layer)
            else:
                LOG.info("NHDArea layer empty: %s:%s", args.nhdarea_gpkg, args.nhdarea_layer)
        except Exception as e:
            LOG.warning("Failed to load NHDArea polygons (%s:%s): %s", args.nhdarea_gpkg, args.nhdarea_layer, str(e))

    # Build corridor masks
    corridor_geom = rivers.geometry.buffer(float(args.channel_buffer_m)).unary_union
    corridor = rasterize(
        [(corridor_geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)

    stream_col = _get_stream_order_col(rivers)
    if stream_col:
        main = rivers[rivers[stream_col].fillna(0).astype(float) >= float(args.mainstem_min_order)]
    else:
        main = rivers.iloc[0:0]
    if not main.empty:
        main_geom = main.geometry.buffer(float(args.channel_buffer_m)).unary_union
        corridor_main = rasterize(
            [(main_geom, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)
        LOG.info("Mainstem corridor enabled using '%s' >= %s (n=%d)", stream_col, args.mainstem_min_order, len(main))
    else:
        corridor_main = np.zeros(shape, dtype=bool)
        if stream_col:
            LOG.info("No mainstem features found at '%s' >= %s; using only default corridor.", stream_col, args.mainstem_min_order)
        else:
            LOG.info("No stream order column found; using only default corridor.")



    # Distance to nearest flowline (centerline proximity)
    # This is used both to suppress stray water far from flowlines and (optionally)
    # to keep ocean-connected pixels near flowlines for tidal mouths.
    skel = rasterize(
        [(geom, 1) for geom in rivers.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    inv = np.ones(shape, dtype=np.uint8)
    inv[skel] = 0
    d_center = distance_transform_edt(inv, sampling=pix).astype("float32")

    # Water mask
    # Water mask(s)
    ocean = None
    if args.ocean_mask:
        om = _warp_mask_to_template(Path(args.ocean_mask), template_profile)
        ocean = (om == 0)  # waffles convention: water=0 (ocean-only mask)
        LOG.info("Ocean-only mask loaded: %s (waffles convention water=0 land=1)", args.ocean_mask)

        # Optionally keep ocean-connected pixels that lie close to river flowlines.
        # This is critical for tidal river mouths/estuaries where the mainstem
        # is classified as ocean water by the coastline mask.
        keep_dist = float(args.ocean_keep_dist_m)
        if keep_dist > 0:
            ocean_exclude = ocean & (d_center > keep_dist)
            LOG.info("Ocean keep enabled: keeping ocean-connected pixels within %.1fm of flowlines", keep_dist)
        else:
            ocean_exclude = ocean

    if args.water_mask:
        wm = _warp_mask_to_template(Path(args.water_mask), template_profile)
        water_all = (wm == 0)  # waffles convention: water=0
        if ocean is not None:
            # Inland-water = (rivers+lakes+ocean) minus ocean-only water
            water = water_all & (~ocean_exclude)
            LOG.info("Derived inland-water mask: water_mask & ~ocean_mask")
        else:
            water = water_all
            LOG.info("Water mask loaded: %s (waffles convention water=0 land=1)", args.water_mask)
    else:
        if ocean is not None:
            # Fallback when NHD water mask is unavailable: restrict corridor to non-ocean areas
            water = corridor & (~ocean_exclude)
            LOG.warning("No --water-mask supplied; using corridor constrained to non-ocean areas from --ocean-mask.")
        else:
            # Last-resort fallback: treat corridor as water
            water = corridor.copy()
            LOG.warning("No --water-mask or --ocean-mask supplied; using buffered corridor as 'water' (ocean separation degraded).")


    # If no raster water-mask is available, NHDArea polygons provide a much better water-domain than corridor-only fallback.
    if (nhdarea_mask is not None) and bool(nhdarea_mask.any()) and (not args.water_mask):
        if ocean is not None:
            water = nhdarea_mask & (~ocean_exclude)
            LOG.info("Using NHDArea polygons as water mask (NHDArea & ~ocean).")
        else:
            water = nhdarea_mask
            LOG.info("Using NHDArea polygons as water mask.")
    # Width proxy from distance to boundary (land)
    d_bank = distance_transform_edt(water, sampling=pix).astype("float32")
    width = (2.0 * d_bank).astype("float32")



    maxw = float(args.max_channel_width_m)
    maxw_main = float(args.max_mainstem_width_m)
    width_limit = np.where(corridor_main, maxw_main, maxw).astype("float32")

    # Candidate river domain
    channel = water & corridor & (width <= width_limit)

    # Also require "not too far from a flowline" (helps when water mask has large bays inside corridor buffer)
    channel &= (d_center <= (float(args.channel_buffer_m) + 0.5 * width_limit + 2.0 * pix))

    # Constrain the river channel to NHDArea polygons when available (prevents random spill into adjacent water bodies).
    if (nhdarea_mask is not None) and bool(nhdarea_mask.any()):
        channel &= nhdarea_mask

    # Open water = water but not channel
    open_water = water & (~channel)

    out_channel = Path(args.out_channel_mask)
    out_open = Path(args.out_open_water_mask)
    out_channel.parent.mkdir(parents=True, exist_ok=True)

    _save_u8(out_channel, channel.astype("uint8"), template_profile, nodata=0)
    _save_u8(out_open, open_water.astype("uint8"), template_profile, nodata=0)

    if args.write_debug:
        dbg = out_channel.parent
        _save_f32(dbg / "debug_width_proxy_m.tif", np.where(water, width, np.nan), template_profile)
        _save_f32(dbg / "debug_d_center_m.tif", np.where(water, d_center, np.nan), template_profile)
        _save_u8(dbg / "debug_corridor_mask.tif", corridor.astype("uint8"), template_profile)
        _save_u8(dbg / "debug_mainstem_corridor_mask.tif", corridor_main.astype("uint8"), template_profile)
        _save_u8(dbg / "debug_skeleton_mask.tif", skel.astype("uint8"), template_profile)

    LOG.info("Wrote: %s", out_channel)
    LOG.info("Wrote: %s", out_open)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    raise SystemExit(main())