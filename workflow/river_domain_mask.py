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
import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt, label, binary_closing, binary_fill_holes
from shapely.ops import unary_union

from dominant_trunk import select_dominant_trunks

log = logging.getLogger("river_domain_mask")

LOG = logging.getLogger("river_domain_mask")


def _union_all_geoms(geos):
    """Compatibility wrapper for GeoPandas/Shapely union.

    GeoPandas 0.14+/Shapely 2 exposes GeoSeries.union_all(); older stacks used unary_union.
    """
    try:
        if hasattr(geos, "union_all"):
            return geos.union_all()
    except Exception:
        log.debug("ignored", exc_info=True)
    # fallback
    try:
        return geos.unary_union
    except Exception:
        log.debug("_union_all_geoms: suppressed exception", exc_info=True)
        # list-like
        return unary_union(list(geos))


def _buffer_geoms_safe(geos, dist_m: float):
    """Buffer geometries robustly across GeoPandas/Shapely version combinations.

    Some GeoPandas versions attempt vectorized buffering that is incompatible with
    Shapely>=2.0, raising NotImplementedError about the array interface.
    To keep this script stable, buffer geometries one-by-one.
    """
    out = []
    for g in list(geos):
        if g is None or getattr(g, "is_empty", True):
            continue
        out.append(g.buffer(float(dist_m)))
    return out



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


def _warp_mask_to_template(mask_path: Path, template_profile: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Warp an input mask raster to the template grid using nearest neighbor.

    Returns
    -------
    warped_mask : np.ndarray
        Warped semantic mask values on the template grid.
    footprint : np.ndarray[bool]
        True where the source raster actually contributed coverage after reprojection.
        This prevents projected edge nodata from being misinterpreted as real water/land.
    """
    shape = (template_profile["height"], template_profile["width"])
    dst = np.ones(shape, dtype=np.uint8)  # semantic default to land outside coverage
    footprint = np.zeros(shape, dtype=np.uint8)
    with rasterio.open(mask_path) as src:

        # Some mask rasters use semantic values (0/1) for water/land and may also
        # set nodata to 0 or 1. If we pass that through as src_nodata, rasterio
        # will treat real water/land as nodata and destroy the mask.
        src_nodata = src.nodata
        if src_nodata in (0, 1):
            src_nodata = None

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=Resampling.nearest,
            src_nodata=src_nodata,
            dst_nodata=1,  # semantic fill only; real validity comes from footprint
        )
        reproject(
            source=np.ones((src.height, src.width), dtype=np.uint8),
            destination=footprint,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )
    return dst, footprint.astype(bool)


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


def _derive_output_mainstem_mask(corridor_main: np.ndarray, channel: np.ndarray) -> np.ndarray:
    """Return the hybrid-use mainstem mask constrained to the retained channel.

    river_domain_mask internally uses a broader connected mainstem corridor for width-policy
    and continuity. The exported mainstem mask, however, is consumed downstream as the active
    region where XS should dominate over the skeleton solution. That output must therefore be
    a subset of the retained river channel rather than the full buffered corridor.
    """
    return corridor_main & channel


def _get_stream_order_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    candidates = ["streamorde", "StreamOrde", "STREAMORDE", "streamorde_", "stream_order", "streamorder", "Strahler"]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def _load_mainstem_solve_rivers(args: argparse.Namespace, target_crs) -> Tuple[gpd.GeoDataFrame, str]:
    """Load the network used for dominant-trunk solving.

    Default behavior prefers the explicit mainstem_solve_network layer when available,
    then falls back to rivers_aoi, so the canonical trunk is solved on a larger,
    unclipped network and only then
    rasterized/clipped to the output grid. That reduces AOI-edge drift while
    keeping standalone runs deterministic.
    """
    requested = str(getattr(args, "mainstem_solve_layer", "auto") or "auto").strip()
    if requested == "same":
        requested = str(getattr(args, "rivers_layer", "rivers_clip") or "rivers_clip").strip()

    candidates = []
    if requested == "auto":
        candidates = ["major_system_network", "mainstem_solve_network", "rivers_aoi", str(getattr(args, "rivers_layer", "rivers_clip") or "rivers_clip")]
    else:
        candidates = [requested]

    last_err = None
    for layer in candidates:
        try:
            gdf = gpd.read_file(args.river_gpkg, layer=layer)
        except Exception as exc:
            log.debug("_load_mainstem_solve_rivers: suppressed exception", exc_info=True)
            last_err = exc
            continue
        if gdf is None or gdf.empty:
            continue
        gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].copy()
        if gdf.empty:
            continue
        gdf = gdf.to_crs(target_crs)
        return gdf, layer

    if requested == "auto":
        raise SystemExit(
            "Dominant trunk solve could not load a usable river layer from the GeoPackage. "
            "Expected major_system_network, mainstem_solve_network, rivers_aoi, or the active rivers layer to exist and contain non-empty geometries."
        ) from last_err
    raise SystemExit(
        f"Dominant trunk solve layer '{requested}' could not be loaded or was empty. "
        "Fix the river network artifact rather than degrading to an implicit fallback."
    ) from last_err



def _resolve_channel_source_policy(requested: str, *, has_usable_nhdarea: bool, nhdarea_overlap_frac: Optional[float] = None, min_nhdarea_overlap_frac: float = 0.01) -> Tuple[str, Optional[str]]:
    """Resolve the effective channel-source policy.

    The documented contract for ``channel_source=auto`` is to prefer NHDArea when
    usable and otherwise use the buffered flowline corridor. That is not a
    fallback/glue path; it is the explicit automatic policy. ``channel_source=nhdarea``
    remains strict and must fail closed when NHDArea is unavailable or unusable.

    A non-empty NHDArea raster is still not *usable* for ``auto`` if it has almost
    no overlap with the flowline corridor on the template grid. That condition
    means the NHDArea extraction/filtering is not representative enough to drive
    automatic channel construction for this AOI, so ``auto`` must stay on its
    documented corridor path rather than aborting early.
    """
    req = str(requested or 'auto').strip().lower()
    if req == 'corridor':
        return 'corridor', None
    if req == 'nhdarea':
        return 'nhdarea', None

    overlap_ok = True
    if nhdarea_overlap_frac is not None:
        overlap_ok = float(nhdarea_overlap_frac) >= float(min_nhdarea_overlap_frac)
    if has_usable_nhdarea and overlap_ok:
        return 'nhdarea', None
    if has_usable_nhdarea and not overlap_ok:
        return 'corridor', 'nhdarea_low_corridor_overlap'
    return 'corridor', 'nhdarea_unusable_or_empty'

def main() -> int:
    p = argparse.ArgumentParser(description="Build river-only raster mask from NHD flowlines + water mask.")
    p.add_argument("--river-gpkg", required=True, help="river_network.gpkg from river_network.py")
    p.add_argument("--rivers-layer", default="rivers_clip", help="Layer name inside gpkg (default rivers_clip)")
    p.add_argument("--template-raster", required=True, help="Template raster defining output grid (e.g., river_dem.tif)")
    p.add_argument("--water-mask", default=None,
                   help="Optional water/land mask raster. For WAFFLES coastline masks: water=0 land=1.")
    p.add_argument("--water-mask-role", default="generic", choices=["generic", "waffles_with_nhd", "river_support"],
                   help="Interpretation of --water-mask. 'waffles_with_nhd' means the mask already contains the full water domain (including inland NHD water), so it must not be ocean-subtracted when building the river domain. 'river_support' means a template-aligned inland river support mask built from river structure; it is not a coastal full-water mask and must not be validated against coastal/ocean overlap semantics.")
    p.add_argument("--ocean-mask", default=None,
                   help="Optional ocean-only waffles coastline mask (land=1, water=0). Used to prevent ocean bleed and to build fallback river domain when NHD water mask is unavailable.")
    p.add_argument("--ocean-keep-dist-m", type=float, default=0.0,
                   help="Allow ocean-connected water within this distance (meters) of a flowline centerline. Useful for tidal river mouths/estuaries where the mainstem is classified as ocean water. 0 disables.")
    p.add_argument("--nhdarea-gpkg", default=None,
                   help="Optional GeoPackage with NHDArea polygons (e.g., river_network.gpkg). If present, used to constrain the river channel mask.")
    p.add_argument("--nhdarea-layer", default="nhdarea_clip",
                   help="Layer name for NHDArea polygons inside --nhdarea-gpkg (default nhdarea_clip).")
    p.add_argument(
        "--channel-source",
        default="auto",
        choices=["auto", "nhdarea", "corridor"],
        help=(
            "How to build the river channel domain. 'nhdarea' uses NHD water polygons filtered to Stream/River types; "
            "'corridor' uses a buffered flowline corridor; 'auto' prefers nhdarea when usable, otherwise falls back to corridor."
        ),
    )
    p.add_argument(
        "--nhdarea-allow-ftype",
        default="460",
        help=(
            "Comma-separated list of allowed NHDArea FType codes to treat as river/stream area (default: 460 Stream/River)."
        ),
    )
    p.add_argument(
        "--nhdarea-allow-fcode",
        default=None,
        help="Optional comma-separated list of allowed NHDArea FCode values. Used only if FType is not present.",
    )
    p.add_argument("--channel-buffer-m", type=float, default=400.0,
                   help="Buffer around flowlines to define candidate corridor (meters).")
    p.add_argument("--max-channel-width-m", type=float, default=600.0,
                   help="Max allowed width inside corridor (meters).")
    p.add_argument("--mainstem-method", default="dominant_trunk", choices=["dominant_trunk", "stream_order"],
                   help="How to identify the mainstem corridor. dominant_trunk solves an outlet-connected weighted trunk path on the river network; stream_order uses the legacy stream-order threshold seed.")
    p.add_argument("--mainstem-solve-layer", default="auto",
                   help="River layer used for dominant-trunk solving. 'auto' prefers major_system_network, then mainstem_solve_network, then rivers_aoi, then --rivers-layer for AOI-stable trunk selection. 'same' forces the active rivers layer.")
    p.add_argument("--mainstem-min-order", type=int, default=5,
                   help="Legacy stream order threshold for mainstem corridor when --mainstem-method=stream_order.")
    p.add_argument("--max-mainstem-width-m", type=float, default=2500.0,
                   help="Max allowed width for mainstem corridor pixels (meters).")
    p.add_argument("--write-debug", action="store_true", default=False,
                   help="Write debug rasters (width proxy, corridor masks).")
    p.add_argument("--out-channel-mask", required=True, help="Output TIFF for river channel mask (1=river,0=else).")
    p.add_argument("--out-open-water-mask", required=True, help="Output TIFF for open water mask (1=open water,0=else).")
    p.add_argument("--out-mainstem-mask", default=None, help="Optional output TIFF for mainstem corridor mask (1=mainstem,0=else). Useful for hybrid XS+Skeleton.")
    p.add_argument("--out-policy-json", default=None, help="Optional JSON summary describing river-domain mask policy, harmonization, and overlap diagnostics.")
    p.add_argument("--out-effective-water-mask", default=None, help="Optional output TIFF for the effective water mask used after ocean exclusion and harmonization (1=water,0=else).")
    p.add_argument("--out-corridor-mask", default=None, help="Optional output TIFF for the buffered river corridor mask used by river-domain selection (1=corridor,0=else).")
    p.add_argument("--out-nhdarea-mask", default=None, help="Optional output TIFF for the filtered NHDArea river mask when available (1=river polygon,0=else).")


    args = p.parse_args()

    template_profile, transform, crs, shape = _read_template(Path(args.template_raster))
    template_profile["transform"] = transform
    template_profile["crs"] = crs

    pix = _pixel_size_m(transform)
    LOG.info("Template grid: %s x %s | pixel_size≈%.3fm", shape[1], shape[0], pix)

    ocean = None
    if args.ocean_mask:
        om, ocean_footprint = _warp_mask_to_template(Path(args.ocean_mask), template_profile)
        ocean = (om == 0) & ocean_footprint  # waffles convention: water=0 (ocean-only mask)
        LOG.info("Ocean-only mask loaded: %s (waffles convention water=0 land=1)", args.ocean_mask)

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
        log.debug("ignored", exc_info=True)

    # Load rivers layer
    rivers = gpd.read_file(args.river_gpkg, layer=args.rivers_layer)
    if rivers.empty:
        raise SystemExit(f"No features found in {args.river_gpkg}:{args.rivers_layer}")
    rivers = rivers.to_crs(crs)

    # Optional: NHDArea polygons to constrain channel predictions.
    # Extract river/stream polygons only (exclude lakes/reservoirs).
    nhdarea_mask = None
    fallback_reason = None
    nhdarea_pixels = 0
    if args.nhdarea_gpkg and (args.channel_source in ("auto", "nhdarea")):
        try:
            areas = gpd.read_file(args.nhdarea_gpkg, layer=args.nhdarea_layer)
            if areas is not None and not areas.empty:
                areas = areas.to_crs(crs)
                areas = areas[areas.geometry.notnull() & (~areas.geometry.is_empty)]
                areas = areas[areas.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                if not areas.empty:
                    allow_ftype = set(int(x.strip()) for x in str(args.nhdarea_allow_ftype).split(",") if str(x).strip())
                    allow_fcode = None
                    if args.nhdarea_allow_fcode:
                        allow_fcode = set(int(x.strip()) for x in str(args.nhdarea_allow_fcode).split(",") if str(x).strip())

                    ftype_col = next((c for c in ("FType", "FTYPE", "ftype") if c in areas.columns), None)
                    fcode_col = next((c for c in ("FCode", "FCODE", "fcode") if c in areas.columns), None)

                    areas_f = areas
                    if ftype_col is not None:
                        # Some services export coded value domains as strings (e.g. 'SwampMarsh') which cannot be cast to int.
                        # We treat non-numeric entries as NA (excluded) so we never accidentally include lakes.

                            # Some services export coded value domains as strings (e.g. 'SwampMarsh', 'StreamRiver')
                            # rather than integer codes. We normalize and map common labels back to numeric FType.
                            raw = areas_f[ftype_col]
                            ser_num = pd.to_numeric(raw, errors="coerce")

                            if ser_num.isna().any():
                                def _norm(v: str) -> str:
                                    v = str(v).strip()
                                    v = v.replace("/", " ").replace("-", " ")
                                    v = re.sub(r"\s+", " ", v)
                                    return v.lower().replace(" ", "")

                                # Minimal mapping needed for safe filtering.
                                # Only include allow_ftype values; unrecognized labels are excluded.
                                label_to_ftype = {
                                    _norm("StreamRiver"): 460,
                                    _norm("Stream/River"): 460,
                                    _norm("Submerged Stream"): 461,
                                    _norm("CanalDitch"): 336,
                                    _norm("Canal/Ditch"): 336,
                                    _norm("BayInlet"): 312,
                                    _norm("SeaOcean"): 445,
                                    _norm("SwampMarsh"): 466,
                                    _norm("Swamp/Marsh"): 466,
                                    _norm("LakePond"): 390,
                                    _norm("Lake/Pond"): 390,
                                    _norm("Reservoir"): 436,
                                    _norm("Estuary"): 493,
                                    _norm("IceMass"): 378,
                                    _norm("Playa"): 361,
                                }

                                mapped = raw.astype(str).map(lambda x: label_to_ftype.get(_norm(x), None))
                                ser_num = ser_num.fillna(mapped)

                            n_bad = int(ser_num.isna().sum())
                            if n_bad:
                                LOG.info(
                                    "NHDArea %s contains %d/%d unparseable FType values; excluding them from river-polygon filter.",
                                    ftype_col,
                                    n_bad,
                                    len(areas_f),
                                )

                            areas_f = areas_f[ser_num.fillna(-1).astype(int).isin(allow_ftype)]
                            LOG.info("NHDArea filter: %s in %s (kept %d/%d polygons)", allow_ftype, ftype_col, len(areas_f), len(areas))
                    elif (allow_fcode is not None) and (fcode_col is not None):
                        areas_f = areas_f[areas_f[fcode_col].fillna(-1).astype(int).isin(allow_fcode)]
                        LOG.info("NHDArea filter: %s in %s (kept %d/%d polygons)", allow_fcode, fcode_col, len(areas_f), len(areas))
                    else:
                        msg = (
                            f"NHDArea layer {args.nhdarea_gpkg}:{args.nhdarea_layer} has no FType/FCode field; "
                            "cannot safely extract river polygons (would include lakes)."
                        )
                        if args.channel_source == "nhdarea":
                            raise RuntimeError(msg)
                        LOG.warning("%s Falling back to corridor.", msg)
                        areas_f = areas.iloc[0:0]

                    if not areas_f.empty:
                        area_geom = _union_all_geoms(areas_f.geometry)
                        nhdarea_mask = rasterize(
                            [(area_geom, 1)],
                            out_shape=shape,
                            transform=transform,
                            fill=0,
                            all_touched=True,
                            dtype="uint8",
                        ).astype(bool)
                        LOG.info(
                            "NHDArea river-only mask enabled: %s:%s (pixels=%d)",
                            args.nhdarea_gpkg,
                            args.nhdarea_layer,
                            int(nhdarea_mask.sum()),
                        )
                else:
                    LOG.info("NHDArea layer has no polygon geometries: %s:%s", args.nhdarea_gpkg, args.nhdarea_layer)
            else:
                LOG.info("NHDArea layer empty: %s:%s", args.nhdarea_gpkg, args.nhdarea_layer)
        except Exception as e:
            LOG.warning("Failed to load NHDArea polygons (%s:%s): %s", args.nhdarea_gpkg, args.nhdarea_layer, str(e))

    # Build corridor masks
    corridor_geom = _union_all_geoms(_buffer_geoms_safe(rivers.geometry, float(args.channel_buffer_m)))
    corridor = rasterize(
        [(corridor_geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)

    

    stream_col = _get_stream_order_col(rivers)

    mainstem_method = str(getattr(args, "mainstem_method", "dominant_trunk") or "dominant_trunk").strip().lower()
    mainstem_diag = {"method": mainstem_method}
    if mainstem_method == "dominant_trunk":
        solve_rivers, solve_layer = _load_mainstem_solve_rivers(args, crs)
        trunk_reaches, trunk_diag = select_dominant_trunks(
            solve_rivers,
            ocean_mask=ocean,
            ocean_transform=transform,
            endpoint_tolerance=max(float(pix) * 1.5, 5.0),
            logger=LOG,
        )
        mainstem_diag.update(trunk_diag)
        mainstem_diag["solve_layer"] = solve_layer
        mainstem_diag["solve_reach_count"] = int(len(solve_rivers))
        main = trunk_reaches.copy()
        if main.empty:
            raise SystemExit("Dominant trunk solve returned no reaches. Fix the network topology or geometry instead of degrading to a heuristic mainstem.")
        main_geom = _union_all_geoms(_buffer_geoms_safe(main.geometry, float(args.channel_buffer_m)))
        corridor_main = rasterize(
            [(main_geom, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='uint8',
        ).astype(bool) & corridor
        LOG.info(
            "Mainstem derived from dominant trunk solve using layer %s (selected_reaches=%d, solve_reaches=%d, corridor_pixels=%d)",
            mainstem_diag.get("solve_layer", "unknown"),
            len(main),
            int(mainstem_diag.get("solve_reach_count", len(main))),
            int(corridor_main.sum()),
        )
    else:
        # Legacy stream-order mainstem proxy. Kept only as an explicit opt-in path.
        if stream_col:
            main = rivers[rivers[stream_col].fillna(0).astype(float) >= float(args.mainstem_min_order)]
        else:
            main = rivers.iloc[0:0]

        if main.empty:
            raise SystemExit(
                "Legacy stream-order mainstem method found no eligible reaches. Use --mainstem-method=dominant_trunk or fix the stream-order attributes."
            )
        main_geom = _union_all_geoms(_buffer_geoms_safe(main.geometry, float(args.channel_buffer_m)))
        corridor_main = rasterize(
            [(main_geom, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='uint8',
        ).astype(bool) & corridor
        LOG.info(
            "Mainstem seed enabled using '%s' >= %s (n=%d)",
            stream_col,
            args.mainstem_min_order,
            len(main),
        )

    if np.any(corridor_main):
        corridor_main = binary_closing(corridor_main, structure=np.ones((3, 3), dtype=bool))
        corridor_main = binary_fill_holes(corridor_main)
        corridor_main &= corridor

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

    # Water mask(s)
    keep_ocean = np.zeros(shape, dtype=bool)  # ocean water pixels we explicitly keep near flowlines
    if ocean is not None:
        # Optionally keep ocean-connected pixels that lie close to river flowlines.
        # This is critical for tidal river mouths/estuaries where the mainstem
        # is classified as ocean water by the coastline mask.
        keep_dist = float(args.ocean_keep_dist_m)
        if keep_dist > 0:
            keep_ocean = ocean & (d_center <= keep_dist)
            ocean_exclude = ocean & (~keep_ocean)
            LOG.info("Ocean keep enabled: keeping ocean-connected pixels within %.1fm of flowlines", keep_dist)
        else:
            keep_ocean = np.zeros(shape, dtype=bool)
            ocean_exclude = ocean



    water_source_effective = "unknown"
    water_mask_harmonization_applied = False
    water_mask_harmonization_reason = None
    water_corridor_overlap_frac = None

    if args.water_mask:
        wm, water_footprint = _warp_mask_to_template(Path(args.water_mask), template_profile)
        if args.water_mask_role == "waffles_with_nhd":
            # This mask already represents the intended full water domain for river delivery.
            # Do not subtract the ocean-only WAFFLES mask here; that was the source of the
            # false zero-overlap failure in estuarine/tidal mainstems. Ocean separation is
            # handled later by channel-width / NHDArea / estuary logic, not by erasing the
            # river-capable water mask up front.
            water = (wm == 0) & water_footprint
            LOG.info("Water mask loaded as WAFFLES with-NHD full water domain: %s", args.water_mask)
        else:
            water_all = (wm == 0) & water_footprint  # explicit convention: water=0 only inside footprint

            def _corridor_overlap(mask_bool: np.ndarray) -> float:
                if corridor.sum() <= 0:
                    return 0.0
                return float((mask_bool & corridor).sum()) / float(corridor.sum())

            if ocean is not None:
                water = water_all & (~ocean_exclude)
                LOG.info("Derived inland-water mask: water_mask & ~ocean_mask")
            else:
                water = water_all
                LOG.info("Water mask loaded: %s (explicit convention water=0)", args.water_mask)

        # Provided water masks must agree with the river corridor. If they do not, fail closed
        # instead of silently replacing them with a synthetic corridor-based mask.
        if corridor.sum() > 0:
            overlap_frac = float((water & corridor).sum()) / float(corridor.sum())
            water_corridor_overlap_frac = overlap_frac
            min_overlap = 0.01 if args.water_mask_role == "river_support" else 0.02
            if overlap_frac < min_overlap:
                raise SystemExit(
                    "Provided water mask is inconsistent with the river corridor "
                    f"(water∩corridor={100.0 * overlap_frac:.2f}%). Fix the water-mask path/semantics instead of falling back. "
                    f"water_mask={args.water_mask} role={args.water_mask_role} ocean_mask={args.ocean_mask}"
                )
    else:
        raise SystemExit("--water-mask is required for river domain construction; refusing to synthesize water from corridor/ocean fallbacks.")

    if args.water_mask and args.water_mask_role == "waffles_with_nhd":
        water_source_effective = "with_nhd"
    elif args.water_mask and args.water_mask_role == "river_support":
        water_source_effective = "river_support"
    elif args.water_mask and ocean is not None:
        water_source_effective = "generic_minus_ocean" if not water_mask_harmonization_applied else "generic_harmonized"
    elif args.water_mask:
        water_source_effective = "generic" if not water_mask_harmonization_applied else "generic_harmonized"
    elif ocean is not None:
        water_source_effective = "corridor_minus_ocean"
    else:
        water_source_effective = "corridor_only"
    if corridor.sum() > 0:
        water_corridor_overlap_frac = float((water & corridor).sum()) / float(corridor.sum())

    # Do not replace the water mask with NHDArea here because NHDArea is filtered
    # to river/stream polygons only (excluding lakes). Using that as a general water mask
    # would incorrectly remove valid non-river water needed for width estimation.
    # Width proxy from distance to boundary (land)
    d_bank = distance_transform_edt(water, sampling=pix).astype("float32")
    width = (2.0 * d_bank).astype("float32")



    # Candidate river domain.
    # Do not apply per-pixel width thresholds here. They are too arbitrary for domain construction
    # and can punch holes through the interior of broad river reaches because the center of the river
    # has the largest bank-distance values. Estuary/wide-water separation is handled later by the
    # estuary clip logic using along-channel widening diagnostics.
    channel_candidate = water & corridor
    channel = channel_candidate.copy()

    # Apply channel-source policy.
    # The critical rule here is: once a connected water-in-corridor component is accepted as
    # structurally supported by NHDArea/mainstem/ocean-keep, retain the full connected interior
    # of that component. Do NOT intersect pixelwise with the structure seed, which erodes broad
    # rivers into bank-parallel ribbons and creates interior holes.
    structure_seed = np.zeros(shape, dtype=bool)
    candidate_component_count = 0
    accepted_component_count = 0
    nhd_overlap_frac = None
    if (nhdarea_mask is not None) and bool(nhdarea_mask.any()) and corridor.sum() > 0:
        nhd_overlap_frac = float((nhdarea_mask & corridor).sum()) / float(corridor.sum())
    has_usable_nhdarea = (nhdarea_mask is not None) and bool(nhdarea_mask.any())
    effective_channel_source, channel_source_reason = _resolve_channel_source_policy(
        args.channel_source,
        has_usable_nhdarea=has_usable_nhdarea,
        nhdarea_overlap_frac=nhd_overlap_frac,
    )
    if effective_channel_source == "corridor":
        if str(args.channel_source).strip().lower() == 'auto' and not has_usable_nhdarea:
            LOG.info(
                "Channel-source auto resolved to corridor because NHDArea river polygons were empty or unusable on the template grid."
            )
        structure_seed = channel_candidate.copy()
        candidate_component_count = int(label(channel_candidate)[1]) if np.any(channel_candidate) else 0
        accepted_component_count = candidate_component_count
    else:
        if (nhdarea_mask is not None) and bool(nhdarea_mask.any()):
            if corridor.sum() > 0:
                ov = float((nhdarea_mask & corridor).sum()) / float(corridor.sum())
                if ov < 0.01:
                    raise SystemExit(
                        "NHDArea river polygons have very low overlap with the flowline corridor "
                        f"({100.0 * ov:.2f}%). Fix NHDArea extraction/filtering instead of degrading to corridor-only."
                    )
            structure_seed = (nhdarea_mask | corridor_main | keep_ocean) & channel_candidate
        else:
            raise SystemExit(
                "No usable NHDArea river polygons were extracted for river domain construction. "
                "Fix NHDArea extraction/filtering or explicitly request --channel-source=corridor if that is scientifically intended."
            )

        from scipy.ndimage import label as _label
        labeled_cc, n_cc = _label(channel_candidate)
        candidate_component_count = int(n_cc)
        seed_labels = np.unique(labeled_cc[structure_seed & (labeled_cc > 0)])
        accepted_component_count = int(seed_labels.size)
        if seed_labels.size > 0:
            channel = np.isin(labeled_cc, seed_labels) & channel_candidate
        else:
            channel = np.zeros_like(channel_candidate, dtype=bool)

    channel_after_structure = channel.copy()
    if np.any(channel):
        # Remove rasterization pinholes and fill interior water holes while preserving the
        # accepted wetted component interior. Clip back to water so islands/land stay excluded.
        channel = binary_closing(channel, structure=np.ones((3, 3), dtype=bool))
        channel = binary_fill_holes(channel)
        channel &= water

    # Open water is the ocean/bay subset of the effective water mask, not arbitrary inland
    # disconnected residual water. Restrict strictly to water components that directly
    # touch the ocean mask; do not use dilation here because that can spuriously connect
    # disconnected inland fragments into the ocean/open-water domain.
    if ocean is not None:
        from scipy.ndimage import label as _label
        labeled_water, _nw = _label(water)
        ocean_touch_labels = set(np.unique(labeled_water[ocean & (labeled_water > 0)]))
        if ocean_touch_labels:
            ocean_connected_water = np.isin(labeled_water, list(ocean_touch_labels)) & water
        else:
            ocean_connected_water = np.zeros_like(water, dtype=bool)
        open_water = ocean_connected_water & (~channel)
    else:
        open_water = water & (~channel)

    out_channel = Path(args.out_channel_mask)
    out_open = Path(args.out_open_water_mask)
    out_channel.parent.mkdir(parents=True, exist_ok=True)

    channel_source_effective = effective_channel_source
    mainstem_mask = _derive_output_mainstem_mask(corridor_main, channel)

    policy_summary = {
        "channel_source_requested": str(args.channel_source),
        "channel_source_resolution_reason": channel_source_reason,
        "channel_source_effective": channel_source_effective,
        "effective_water_source": water_source_effective,
        "mask_value_convention": "1=inside,0=outside",
        "domain_width_filter_applied": False,
        "estuary_widening_is_handled_downstream": True,
        "water_mask_harmonization_applied": bool(water_mask_harmonization_applied),
        "water_mask_harmonization_reason": water_mask_harmonization_reason,
        "ocean_mask_available": bool(args.ocean_mask),
        "with_nhd_water_mask_available": bool(args.water_mask),
        "nhdarea_available": bool(args.nhdarea_gpkg),
        "nhdarea_effective": bool(channel_source_effective == "nhdarea"),
        "corridor_pixels": int(corridor.sum()),
        "mainstem_corridor_pixels": int(corridor_main.sum()),
        "mainstem_output_pixels": int(mainstem_mask.sum()),
        "effective_water_pixels": int(water.sum()),
        "water_mask_footprint_pixels": int(water_footprint.sum()) if "water_footprint" in locals() else None,
        "channel_pixels": int(channel.sum()),
        "open_water_pixels": int(open_water.sum()),
        "ocean_pixels": int(ocean.sum()) if ocean is not None else 0,
        "kept_ocean_pixels": int(keep_ocean.sum()) if "keep_ocean" in locals() else 0,
        "nhdarea_pixels": int(nhdarea_mask.sum()) if (nhdarea_mask is not None) else 0,
        "structure_seed_pixels": int(structure_seed.sum()),
        "candidate_component_count": int(candidate_component_count),
        "accepted_component_count": int(accepted_component_count),
        "effective_water_corridor_overlap_frac": water_corridor_overlap_frac,
        "channel_corridor_overlap_frac": float((channel & corridor).sum()) / float(corridor.sum()) if corridor.sum() > 0 else None,
        "mainstem_channel_overlap_frac": float((mainstem_mask & channel).sum()) / float(mainstem_mask.sum()) if mainstem_mask.sum() > 0 else None,
        "open_water_corridor_overlap_frac": float((open_water & corridor).sum()) / float(corridor.sum()) if corridor.sum() > 0 else None,
        "nhdarea_corridor_overlap_frac": nhd_overlap_frac,
        "mainstem_method": mainstem_method,
        "mainstem_diagnostics": mainstem_diag,
    }

    _save_u8(out_channel, channel.astype("uint8"), template_profile, nodata=0)
    _save_u8(out_open, open_water.astype("uint8"), template_profile, nodata=0)
    if args.out_mainstem_mask:
        out_main = Path(args.out_mainstem_mask)
        out_main.parent.mkdir(parents=True, exist_ok=True)
        _save_u8(out_main, mainstem_mask.astype("uint8"), template_profile, nodata=0)
        LOG.info("Wrote: %s", out_main)
    if args.out_effective_water_mask:
        out_eff = Path(args.out_effective_water_mask)
        out_eff.parent.mkdir(parents=True, exist_ok=True)
        _save_u8(out_eff, water.astype("uint8"), template_profile, nodata=0)
        LOG.info("Wrote: %s", out_eff)
    if args.out_corridor_mask:
        out_corr = Path(args.out_corridor_mask)
        out_corr.parent.mkdir(parents=True, exist_ok=True)
        _save_u8(out_corr, corridor.astype("uint8"), template_profile, nodata=0)
        LOG.info("Wrote: %s", out_corr)
    if args.out_nhdarea_mask and (nhdarea_mask is not None):
        out_nhd = Path(args.out_nhdarea_mask)
        out_nhd.parent.mkdir(parents=True, exist_ok=True)
        _save_u8(out_nhd, nhdarea_mask.astype("uint8"), template_profile, nodata=0)
        LOG.info("Wrote: %s", out_nhd)
    if args.out_policy_json:
        out_json = Path(args.out_policy_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(policy_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOG.info("Wrote: %s", out_json)

    if args.write_debug:
        dbg = out_channel.parent
        _save_f32(dbg / "debug_width_proxy_m.tif", np.where(water, width, np.nan), template_profile)
        _save_f32(dbg / "debug_d_center_m.tif", np.where(water, d_center, np.nan), template_profile)
        _save_u8(dbg / "debug_effective_water_mask.tif", water.astype("uint8"), template_profile)
        if "water_footprint" in locals():
            _save_u8(dbg / "debug_water_mask_footprint.tif", water_footprint.astype("uint8"), template_profile)
        _save_u8(dbg / "debug_corridor_mask.tif", corridor.astype("uint8"), template_profile)
        _save_u8(dbg / "debug_mainstem_corridor_mask.tif", corridor_main.astype("uint8"), template_profile)
        _save_u8(dbg / "debug_channel_candidate_from_water_corridor.tif", channel_candidate.astype("uint8"), template_profile)
        _save_u8(dbg / "debug_channel_after_structure_filter.tif", channel_after_structure.astype("uint8"), template_profile)
        _save_u8(dbg / "debug_channel_final.tif", channel.astype("uint8"), template_profile)
        _save_u8(dbg / "debug_skeleton_mask.tif", skel.astype("uint8"), template_profile)

    LOG.info("Wrote: %s", out_channel)
    LOG.info("Wrote: %s", out_open)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    raise SystemExit(main())