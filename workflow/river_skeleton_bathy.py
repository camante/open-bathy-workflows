#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
river_skeleton_bathy.py — Channel-skeleton, distance-transform river bathymetry (NO cross-sections).

Given:
  - a river channel domain mask (1=river channel pixels),
  - NHD flowlines (centerline seed),
  - a DEM on the same grid,

we generate a smooth within-channel depth field that cannot "zipper" across meanders because it is
defined purely in raster space:

  d_bank   = distance to the channel boundary
  d_center = distance to the nearest centerline pixel
  r        = normalized coordinate = d_bank / (d_bank + d_center)   [0 at bank, 1 at centerline]
  Depth    = Dmax * r^shape_exp

Dmax prior:
  - powerlaw: Dmax = a0 * width^bw, where width ≈ 2*d_bank at the centerline.
  - multivariate: optionally uses NHD attributes if present (drainage area, slope); falls back to powerlaw.

Outputs:
  - out-bed.tif : river bed elevation raster (same vertical datum as input DEM)
Optionally (--debug-dir), writes:
  - depth_m.tif, r.tif, d_bank_m.tif, d_center_m.tif, dmax_m.tif, wse_m.tif
"""

import argparse
import logging
import math
import json
from shapely.geometry import LineString, MultiLineString

log = logging.getLogger("river_skeleton_bathy")
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import array_bounds
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation

LOG = logging.getLogger("river_skeleton_bathy")


def _pixel_size_m(transform: rasterio.Affine) -> float:
    return float((abs(transform.a) + abs(transform.e)) / 2.0)


def _read_template(template_raster: Path) -> Tuple[dict, rasterio.Affine, rasterio.crs.CRS, Tuple[int, int]]:
    with rasterio.open(template_raster) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
    return profile, transform, crs, shape


def _warp_to_template(src_path: Path, template_profile: dict, dtype: str = "float32") -> Tuple[np.ndarray, Optional[float]]:
    """Warp a raster to the template grid.

    IMPORTANT correctness note:
      Rasterio's reproject does not necessarily overwrite destination pixels that are not touched
      by the source. If we initialize the destination with zeros, "outside-coverage" pixels can
      become *valid zeros* (a real elevation), which is catastrophic for DEM-driven inference.

    We therefore initialize floating outputs with a deterministic nodata fill.
    """
    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        src_nodata = src.nodata

        is_float = np.issubdtype(np.dtype(dtype), np.floating)
        if is_float:
            # Choose a deterministic dst nodata fill.
            fill = float(src_nodata) if (src_nodata is not None and np.isfinite(src_nodata)) else -9999.0
            dst = np.full((template_profile["height"], template_profile["width"]), fill, dtype=dtype)
            dst_nodata = fill
            resamp = Resampling.bilinear
        else:
            # For masks, default to 0 outside coverage (non-channel/non-water).
            dst = np.zeros((template_profile["height"], template_profile["width"]), dtype=dtype)
            dst_nodata = src_nodata
            resamp = Resampling.nearest

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=resamp,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
        )
    return dst, src_nodata



def _junction_zone_mask(
    river_gpkg: Path,
    nodes_layer: str,
    river_layer: str,
    template_profile: dict,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    shape: Tuple[int, int],
    degree_min: int = 3,
    buffer_m: float = 120.0,
) -> Optional[np.ndarray]:
    """Rasterize a buffered 'junction zone' mask from graph nodes (degree>=degree_min).

    This is used to reduce common artifacts near confluences where multiple centerlines and
    bank geometries meet and raster-space priors can conflict.
    """
    if buffer_m is None or float(buffer_m) <= 0.0:
        return None
    try:
        gdf = gpd.read_file(river_gpkg, layer=nodes_layer)
        if gdf.empty:
            return None
        if "degree" not in gdf.columns:
            LOG.warning("junction mask: nodes layer '%s' lacks 'degree' column; skipping.", nodes_layer)
            return None
        gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].copy()
        gdf = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
        gdf = gdf[pd.to_numeric(gdf["degree"], errors="coerce").fillna(0).astype(int) >= int(degree_min)].copy()
        if gdf.empty:
            return None
        try:
            if gdf.crs is not None and crs is not None:
                c1 = CRS.from_user_input(gdf.crs)
                c2 = CRS.from_user_input(crs)
                if not c1.equals(c2):
                    gdf = gdf.to_crs(c2)
        except Exception:
            # Best effort; if CRS is missing or parsing fails, rasterize as-is.
            log.debug("ignored", exc_info=True)

        # Buffer and dissolve into a single geometry for efficiency
        geom = gdf.geometry.buffer(float(buffer_m))
        try:
            union = geom.union_all()
        except Exception:
            # GeoPandas < 0.14
            union = geom.unary_union
        if union is None:
            return None

        mask_u8 = rasterize(
            [(union, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )
        return (mask_u8 == 1)

    except Exception as e:
        # Common failure: nodes layer missing (e.g., "Null layer"). Fall back to a simple
        # endpoint-degree approximation from the flowlines layer.
        LOG.warning("junction mask: failed building junction zone mask: %s", str(e))
        try:
            fb = _junction_zone_mask_from_flowlines(
                river_gpkg=river_gpkg,
                river_layer=river_layer,
                transform=transform,
                crs=crs,
                shape=shape,
                degree_min=degree_min,
                buffer_m=buffer_m,
            )
            if fb is not None:
                LOG.info("junction mask: using flowline-endpoint fallback (layer=%s).", river_layer)
                return fb
        except Exception as e2:
            LOG.warning("junction mask fallback failed: %s", str(e2))
        return None




def _junction_zone_mask_from_flowlines(
    river_gpkg: Path,
    river_layer: str,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    shape: Tuple[int, int],
    degree_min: int = 3,
    buffer_m: float = 120.0,
    snap_m: float = 5.0,
) -> Optional[np.ndarray]:
    """Fallback junction-zone mask derived from flowline endpoints.

    If a graph node layer is unavailable, approximate junctions by counting how many
    flowline endpoints fall on the same snapped coordinate (in projected meters).

    This is intentionally conservative and meant only to identify confluence neighborhoods
    for smoothing/masking, not to replace a true graph.
    """
    gdf = None
    tried = []
    for lyr in [river_layer, 'rivers_clip', 'rivers', 'flowlines', 'river_network']:
        if lyr in tried or lyr is None:
            continue
        tried.append(lyr)
        try:
            gdf = gpd.read_file(river_gpkg, layer=lyr)
            if gdf is not None and not gdf.empty:
                river_layer = lyr  # record the layer that worked
                break
        except Exception:
            continue

    if gdf is None or gdf.empty:
        LOG.warning("junction mask fallback: could not read any flowlines layer (tried=%s)", tried)
        return None
    gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].copy()
    if gdf.empty:
        return None
    # Reproject to template CRS if needed
    try:
        if gdf.crs is not None and crs is not None:
            c1 = CRS.from_user_input(gdf.crs)
            c2 = CRS.from_user_input(crs)
            if not c1.equals(c2):
                gdf = gdf.to_crs(c2)
    except Exception:
        log.debug("ignored", exc_info=True)

    # Collect coordinates (not just endpoints), tracking which feature each coordinate came from.
    # Confluences are often represented as shared vertices between a mainstem and tributary.
    pts = []  # (x, y, fid)
    for fid, geom in enumerate(gdf.geometry.values):
        if geom is None:
            continue
        gt = geom.geom_type
        if gt == "LineString":
            coords = list(geom.coords)
            if len(coords) >= 2:
                pts.extend([(c[0], c[1], fid) for c in coords])
        elif gt == "MultiLineString":
            for ls in geom.geoms:
                coords = list(ls.coords)
                if len(coords) >= 2:
                    pts.extend([(c[0], c[1], fid) for c in coords])

    if not pts:
        return None

    snap = float(snap_m) if snap_m is not None else 5.0
    if snap <= 0:
        snap = 1.0

    keys = {}  # (x,y) -> set(feature_ids)
    for (x, y, fid) in pts:
        kx = round(float(x) / snap) * snap
        ky = round(float(y) / snap) * snap
        s = keys.get((kx, ky))
        if s is None:
            keys[(kx, ky)] = {fid}
        else:
            s.add(fid)

    # With unsplit flowlines, a confluence is commonly a shared vertex between exactly two features
    # (mainstem + tributary). Treat >=2 as a junction by default, while still respecting a higher
    # requested degree_min if the user supplied one.
    deg_min_eff = max(2, int(degree_min) - 1)
    jpts = [(x, y) for (x, y), fids in keys.items() if len(fids) >= deg_min_eff]
    if not jpts:
        return None

    from shapely.geometry import Point
    geoms = [Point(x, y).buffer(float(buffer_m)) for (x, y) in jpts]
    union = gpd.GeoSeries(geoms).unary_union
    if union is None:
        return None

    mask_u8 = rasterize(
        [(union, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return (mask_u8 == 1)


def _build_wse_from_bank_dem(
    dem: np.ndarray,
    channel: np.ndarray,
    valid_dem: np.ndarray,
    pix_m: float,
    smooth_sigma_m: float = 0.0,
) -> Optional[np.ndarray]:
    """Approximate a water-surface elevation (WSE) field from *bank* DEM samples.

    Rationale:
      Many DEMs have unreliable or missing elevations in the wetted channel (water surface, voids),
      while banks are typically better constrained. Using bank samples reduces the risk of driving
      depths from spurious in-channel DEM values.

    Method:
      1) Identify bank-adjacent channel pixels (channel pixels with at least one non-channel neighbor).
      2) Seed WSE values from DEM at those bank-adjacent pixels.
      3) Fill the channel interior by nearest-neighbor propagation (distance transform indices).
      4) Optionally smooth the filled WSE field.
    """
    if channel.sum() == 0:
        return None

    # Bank-adjacent pixels: channel pixels that touch non-channel
    non_channel = ~channel
    touches_non = binary_dilation(non_channel, structure=np.ones((3, 3), dtype=bool))
    bank = channel & touches_non & valid_dem
    if int(bank.sum()) < 10:
        return None

    wse_seed = np.full(channel.shape, np.nan, dtype="float32")
    wse_seed[bank] = dem[bank].astype("float32")

    valid = np.isfinite(wse_seed)
    if not valid.any():
        return None

    # Nearest-neighbor fill within the channel
    inv = ~valid
    _, (iy, ix) = distance_transform_edt(inv, return_indices=True)
    wse = wse_seed[iy, ix].astype("float32")
    wse = np.where(channel, wse, np.nan).astype("float32")

    if smooth_sigma_m and smooth_sigma_m > 0.0:
        sigma_px = float(smooth_sigma_m) / max(pix_m, 1e-9)
        # Smooth only within channel: smooth full field, then reapply mask.
        wse_sm = gaussian_filter(np.where(np.isfinite(wse), wse, np.nanmean(wse[valid])), sigma=sigma_px)
        wse = np.where(channel, wse_sm, np.nan).astype("float32")
    return wse



def _build_wse_longitudinal_profile(
    bank_wse: np.ndarray,
    channel: np.ndarray,
    template_transform: rasterio.Affine,
    template_crs,
    river_gpkg: Path,
    river_layer: str = "rivers_clip",
    step_m: float = 20.0,
    resample_m: float = 20.0,
    smooth_sigma_m: float = 200.0,
    max_slope: float = 0.0,
    min_samples: int = 10,
    max_query_dist_m: float = 250.0,
    swot_pts: Optional[np.ndarray] = None,
    swot_wse: Optional[np.ndarray] = None,
    swot_max_dist_m: float = 300.0,
    swot_min_samples: int = 5,
    swot_correct_sigma_m: float = 2000.0,
    swot_weight: float = 1.0,
    swot_max_correction_m: float = 5.0,
    fit_mode: str = "isotonic",
) -> Optional[np.ndarray]:
    """Build a longitudinally-consistent WSE surface from bank-derived WSE samples.

    Scientific intent:
      - WSE should vary primarily along-channel and be approximately constant across a cross-section.
      - Bank-adjacent DEM samples are typically more reliable than in-channel DEM in hydro-flattened/noisy products.
      - Build a 1D WSE profile along flowlines, smooth it along distance, optionally enforce a max slope,
        then rasterize to centerline pixels and expand across the channel via nearest-centerline assignment.

    Returns: WSE map on the template grid (float32, NaN outside channel), or None on failure.
    """

    def _pava_non_decreasing(y: np.ndarray) -> np.ndarray:
        """Pool-Adjacent-Violators (PAVA) isotonic regression for non-decreasing sequences.

        Deterministic, O(n), no sklearn dependency.
        """
        y = np.asarray(y, dtype=float)
        n = int(y.size)
        if n <= 1:
            return y.astype(float)

        starts: list[int] = []
        ends: list[int] = []
        means: list[float] = []

        for i in range(n):
            starts.append(i)
            ends.append(i)
            means.append(float(y[i]))

            while len(means) >= 2 and means[-2] > means[-1]:
                s0, e0, m0 = starts[-2], ends[-2], means[-2]
                s1, e1, m1 = starts[-1], ends[-1], means[-1]
                w0 = (e0 - s0 + 1)
                w1 = (e1 - s1 + 1)
                m = (m0 * w0 + m1 * w1) / float(w0 + w1)
                starts[-2] = s0
                ends[-2] = e1
                means[-2] = float(m)
                starts.pop(); ends.pop(); means.pop()

        out = np.empty(n, dtype=float)
        for s, e, m in zip(starts, ends, means):
            out[s:e + 1] = float(m)
        return out

    def _isotonic_non_increasing(y: np.ndarray) -> np.ndarray:
        """Isotonic regression enforcing a non-increasing sequence."""
        y = np.asarray(y, dtype=float)
        return -_pava_non_decreasing(-y)
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, MultiLineString
        from pyproj import CRS
        from rasterio import features
        from scipy.ndimage import distance_transform_edt, gaussian_filter1d
        from scipy.spatial import cKDTree
        swot_tree = None
        if swot_pts is not None and swot_wse is not None and len(swot_pts) > 0 and len(swot_pts) == len(swot_wse):
            try:
                swot_tree = cKDTree(np.asarray(swot_pts, dtype=float))
            except Exception as e:
                LOG.warning("WSE longitudinal profile: could not build SWOT KDTree: %s", e)
                swot_tree = None
        # Robust aggregation helper for multiple SWOT observations near a sample.
        def _robust_median_mad(v: np.ndarray, zmax: float = 4.0) -> float:
            v = np.asarray(v, dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return float("nan")
            if v.size < 3:
                return float(np.nanmedian(v))
            med = float(np.nanmedian(v))
            mad = float(np.nanmedian(np.abs(v - med)))
            if mad <= 0.0 or (not np.isfinite(mad)):
                return float(med)
            z = np.abs(v - med) / (1.4826 * mad)
            v2 = v[z <= float(zmax)]
            if v2.size == 0:
                return float(med)
            return float(np.nanmedian(v2))

    except Exception as e:
        LOG.warning("WSE longitudinal profile: missing deps: %s", e)
        return None

    if bank_wse is None:
        return None

    h, w = bank_wse.shape
    # Read flowlines
    try:
        gdf = gpd.read_file(str(river_gpkg), layer=river_layer)
    except Exception as e:
        LOG.warning("WSE longitudinal profile: could not read %s layer=%s: %s", river_gpkg, river_layer, e)
        return None
    if gdf is None or len(gdf) == 0:
        LOG.warning("WSE longitudinal profile: empty flowlines layer %s", river_layer)
        return None

    # Reproject to template CRS if needed (robust CRS equality)
    try:
        if gdf.crs is not None and template_crs is not None:
            c1 = CRS.from_user_input(gdf.crs)
            c2 = CRS.from_user_input(template_crs)
            if not c1.equals(c2):
                gdf = gdf.to_crs(c2)
    except Exception as e:
        LOG.warning("WSE longitudinal profile: CRS reprojection check failed: %s", e)

    # Densify lines and sample bank_wse at points
    valid = np.isfinite(bank_wse) & channel
    if not valid.any():
        LOG.warning("WSE longitudinal profile: bank_wse has no valid in-channel samples")
        return None

    # helper to sample raster at xy
    def _sample_at_xy(x, y):
        try:
            r, c = rasterio.transform.rowcol(template_transform, x, y)
        except Exception:
            return np.nan
        if r < 0 or c < 0 or r >= h or c >= w:
            return np.nan
        v = float(bank_wse[r, c])
        return v if np.isfinite(v) else np.nan

    all_pts = []
    all_vals = []

    step_m = float(step_m or 20.0)
    step_m = max(step_m, 1.0)
    resample_m = float(resample_m or step_m)
    resample_m = max(resample_m, 1.0)

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = []
        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            L = float(line.length)
            if not np.isfinite(L) or L <= step_m:
                continue
            # points along line
            n = max(int(math.floor(L / step_m)) + 1, 2)
            dists = np.linspace(0.0, L, n)
            xs: list[float] = []
            ys: list[float] = []
            ws: list[float] = []
            swot_ws: list[float] = []
            swot_d: list[float] = []
            swot_used_idx: set[int] = set()
            d_kept: list[float] = []
            for d in dists:
                pt = line.interpolate(d)
                x, y = float(pt.x), float(pt.y)
                v = _sample_at_xy(x, y)
                if np.isfinite(v):
                    xs.append(x)
                    ys.append(y)
                    ws.append(v)
                    d_kept.append(float(d))
                    if swot_tree is not None:
                        try:
                            # Collect all SWOT points within radius and robustly aggregate.
                            # This is more stable than a single nearest neighbor and reduces
                            # sensitivity to mixed-quality samples near confluences.
                            idxs = swot_tree.query_ball_point([float(x), float(y)], r=float(swot_max_dist_m))
                            if idxs:
                                # Prefer unused points to avoid over-weighting a single observation.
                                idxs2 = [ii for ii in idxs if int(ii) not in swot_used_idx]
                                if idxs2:
                                    idxs = idxs2
                                wsvs = np.asarray([float(swot_wse[int(ii)]) for ii in idxs], dtype=float)
                                wsvs = wsvs[np.isfinite(wsvs)]
                                if wsvs.size > 0:
                                    wsv = _robust_median_mad(wsvs, zmax=4.0)
                                    if np.isfinite(wsv):
                                        # Mark the closest index as used (not all indices), so we don't
                                        # repeatedly use the same local observation but also don't
                                        # discard nearby distinct observations.
                                        try:
                                            # approximate closest as min Euclidean in tree space
                                            pts_loc = np.asarray([swot_pts[int(ii)] for ii in idxs], dtype=float)
                                            di = np.sqrt(np.sum((pts_loc - np.asarray([float(x), float(y)]))**2, axis=1))
                                            ii0 = int(idxs[int(np.argmin(di))])
                                            swot_used_idx.add(ii0)
                                        except Exception:
                                            swot_used_idx.add(int(idxs[0]))
                                        swot_ws.append(float(wsv))
                                        swot_d.append(float(d))
                        except Exception:
                            log.debug("ignored", exc_info=True)

            if len(ws) < int(min_samples):
                continue

            # Build 1D profile on regular distance grid for smoothing.
            # d_kept was tracked in the sampling loop above, so it is
            # aligned with ws/xs/ys by construction.
            ws = np.asarray(ws, dtype=float)
            d_kept = np.asarray(d_kept, dtype=float)
            if len(d_kept) != len(ws):
                # Extremely defensive fallback (should not happen).
                d_kept = np.linspace(0.0, L, len(ws), dtype=float)

            # regular grid
            d0, d1 = float(d_kept.min()), float(d_kept.max())
            if d1 <= d0:
                continue
            dgrid = np.arange(d0, d1 + resample_m * 0.5, resample_m, dtype=float)
            wgrid = np.interp(dgrid, d_kept, ws)

            # smooth along distance
            if smooth_sigma_m and float(smooth_sigma_m) > 0.0:
                sigma_px = float(smooth_sigma_m) / resample_m
                if sigma_px > 0.25:
                    wgrid = gaussian_filter1d(wgrid, sigma_px, mode="nearest")

            # enforce max slope if requested
            if max_slope and float(max_slope) > 0.0:
                smax = float(max_slope)
                # forward pass
                for i in range(1, len(wgrid)):
                    dx = dgrid[i] - dgrid[i - 1]
                    if dx <= 0:
                        continue
                    dv = wgrid[i] - wgrid[i - 1]
                    lim = smax * dx
                    if dv > lim:
                        wgrid[i] = wgrid[i - 1] + lim
                    elif dv < -lim:
                        wgrid[i] = wgrid[i - 1] - lim
                # backward pass
                for i in range(len(wgrid) - 2, -1, -1):
                    dx = dgrid[i + 1] - dgrid[i]
                    if dx <= 0:
                        continue
                    dv = wgrid[i] - wgrid[i + 1]
                    lim = smax * dx
                    if dv > lim:
                        wgrid[i] = wgrid[i + 1] + lim
                    elif dv < -lim:
                        wgrid[i] = wgrid[i + 1] - lim

            # SWOT RiverSP anchoring (optional):
            # If SWOT WSE samples exist near this flowline, compute a residual
            # (SWOT - bank_profile) along distance, smooth it longitudinally,
            # clamp it for safety, and apply it as a correction to the 1D WSE.
            if swot_tree is not None and len(swot_ws) >= int(swot_min_samples or 0):
                try:
                    swd = np.asarray(swot_d, dtype=float)
                    sww = np.asarray(swot_ws, dtype=float)
                    ok = np.isfinite(swd) & np.isfinite(sww)
                    swd = swd[ok]
                    sww = sww[ok]
                    if swd.size >= int(swot_min_samples or 0):
                        # sort by distance along line
                        order = np.argsort(swd)
                        swd = swd[order]
                        sww = sww[order]

                        # expected WSE from current profile at SWOT distances
                        w_at = np.interp(swd, dgrid, wgrid)
                        resid = (sww - w_at).astype(float)

                        # interpolate residual onto profile grid
                        resid_grid = np.interp(dgrid, swd, resid)

                        # smooth residual along distance
                        if swot_correct_sigma_m and float(swot_correct_sigma_m) > 0.0:
                            sigma_px_sw = float(swot_correct_sigma_m) / resample_m
                            if sigma_px_sw > 0.25:
                                resid_grid = gaussian_filter1d(resid_grid, sigma_px_sw, mode="nearest")

                        # clamp correction (protect against datum mistakes and outliers)
                        if swot_max_correction_m and float(swot_max_correction_m) > 0.0:
                            resid_grid = np.clip(resid_grid, -float(swot_max_correction_m), float(swot_max_correction_m))

                        # apply weighted correction
                        wgrid = (wgrid + float(swot_weight or 1.0) * resid_grid).astype(float)
                except Exception as e:
                    LOG.warning("SWOT anchoring failed for a flowline segment; continuing without SWOT. Error: %s", e)

            # Enforce a physically consistent monotonic (non-increasing) WSE trend along the flowline.
            # We infer the most likely downstream direction from the endpoints (higher -> lower),
            # then apply an isotonic regression constraint in that direction.
            fm = str(fit_mode or "").strip().lower()
            if fm not in ("", "none", "off", "false", "0"):
                try:
                    if wgrid.size >= 3 and np.isfinite(wgrid).all():
                        rev = bool(wgrid[-1] > wgrid[0])
                        if rev:
                            wfit = _isotonic_non_increasing(wgrid[::-1])[::-1]
                        else:
                            wfit = _isotonic_non_increasing(wgrid)
                        # Preserve original mean level as a safety against gross shifts.
                        # (Isotonic changes shape but should not introduce large offsets.)
                        mu0 = float(np.mean(wgrid))
                        mu1 = float(np.mean(wfit))
                        if np.isfinite(mu0) and np.isfinite(mu1):
                            wgrid = (wfit + (mu0 - mu1)).astype(float)
                        else:
                            wgrid = wfit.astype(float)
                except Exception:
                    log.debug("ignored", exc_info=True)

            # map smoothed profile back to sample points and store for KDTree
            w_s = np.interp(d_kept, dgrid, wgrid)
            for x, y, wv in zip(xs, ys, w_s):
                all_pts.append((float(x), float(y)))
                all_vals.append(float(wv))

    if len(all_vals) < int(min_samples):
        LOG.warning("WSE longitudinal profile: insufficient valid samples overall (%d)", len(all_vals))
        return None

    pts = np.asarray(all_pts, dtype=float)
    vals = np.asarray(all_vals, dtype=float)

    # Rasterize a centerline mask from flowlines to pick pixels to seed WSE
    centerline = features.rasterize(
        ((geom, 1) for geom in gdf.geometry if geom is not None and (not geom.is_empty)),
        out_shape=(h, w),
        transform=template_transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

    # If centerline is empty, can't seed; fall back to bank_wse
    if not centerline.any():
        LOG.warning("WSE longitudinal profile: centerline rasterization produced empty mask")
        return None

    # Assign a WSE value to each centerline pixel using nearest sampled profile point
    tree = cKDTree(pts)
    rr, cc = np.where(centerline)
    xs, ys = rasterio.transform.xy(template_transform, rr, cc, offset="center")
    q = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    dist, idx = tree.query(q, k=1)
    wse_center = np.full((h, w), np.nan, dtype="float32")
    # Reject assignments that are implausibly far away in XY space (often indicates
    # cross-reach snapping near confluences/parallel channels).
    max_q = float(max_query_dist_m or 0.0)
    if max_q > 0.0:
        ok = dist <= max_q
        if np.any(ok):
            wse_center[rr[ok], cc[ok]] = vals[idx[ok]].astype("float32")
    else:
        wse_center[rr, cc] = vals[idx].astype("float32")

    # Expand to channel pixels by nearest-centerline assignment (cross-sectional constancy)
    # Use EDT indices on inverted centerline (False at centerline => distance 0)
    inv = ~centerline
    _, inds = distance_transform_edt(inv, return_indices=True)
    nr = inds[0].astype(int)
    nc = inds[1].astype(int)

    wse_long = wse_center[nr, nc]
    # Mask to channel and validity
    wse_long = np.where(channel, wse_long, np.nan).astype("float32")

    if not np.isfinite(wse_long[channel]).any():
        LOG.warning("WSE longitudinal profile: output has no finite values in channel")
        return None

    return wse_long


def _save_f32(path: Path, arr: np.ndarray, template_profile: dict, nodata: float = -9999.0) -> None:
    profile = template_profile.copy()
    profile.update(dtype="float32", count=1, nodata=nodata, compress="deflate")
    out = np.where(np.isfinite(arr), arr, nodata).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def _get_attr_col(gdf: gpd.GeoDataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def _dmax_from_powerlaw(width_m: np.ndarray, a0: float, bw: float, dmin: float, dmax: float) -> np.ndarray:
    d = (a0 * np.power(np.maximum(width_m, 1e-3), bw)).astype("float32")
    d = np.clip(d, dmin, dmax)
    return d



def _read_swot_riversp_points(paths, template_crs=None, wse_field=None, qual_field=None, wse_offset_m: float = 0.0):
    """Read SWOT RiverSP reach/node vector files and return point samples (x,y,wse).

    Notes:
      - We support Point geometries directly.
      - If geometries are LineString/MultiLineString (reach products), we sample the midpoint as a representative location.
      - Vertical datum handling can be done in two steps:
          (1) Optional user offset via --swot-wse-offset-m (applied additively when reading the data).
          (2) Optional robust auto-reconciliation in bank_profile mode (see --swot-offset-mode), which estimates
              a constant offset between SWOT WSE and your bank-derived WSE and subtracts it from SWOT before anchoring.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point, LineString, MultiLineString
        from pyproj import CRS
    except Exception as e:
        LOG.warning("SWOT RiverSP: missing deps: %s", e)
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    if not paths:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    cand_wse = [wse_field] if wse_field else []
    cand_wse += [
        "wse", "wse_m", "wse_mean", "wse_mean_m", "wse_water", "wse_water_m",
        "wse_reach", "wse_node", "wse_elev", "wse_elevation",
        "WSE", "WSE_M", "WSE_MEAN",
    ]

    pts = []
    vals = []

    for p in paths:
        try:
            gdf = gpd.read_file(p)
        except Exception as e:
            LOG.warning("SWOT RiverSP: could not read %s: %s", p, e)
            continue
        if gdf is None or len(gdf) == 0:
            continue

        # Reproject to template CRS if provided
        try:
            if template_crs is not None and gdf.crs is not None:
                c1 = CRS.from_user_input(gdf.crs)
                c2 = CRS.from_user_input(template_crs)
                if not c1.equals(c2):
                    gdf = gdf.to_crs(c2)
        except Exception as e:
            LOG.warning("SWOT RiverSP: CRS reprojection failed for %s: %s", p, e)

        # Pick WSE column
        wcol = None
        for c in cand_wse:
            if c and c in gdf.columns:
                wcol = c
                break
        if wcol is None:
            # try any numeric column with 'wse' substring
            for c in gdf.columns:
                if c == "geometry":
                    continue
                if "wse" in str(c).lower():
                    try:
                        if str(gdf[c].dtype).startswith(("int", "float")):
                            wcol = c
                            break
                    except Exception:
                        log.debug("ignored", exc_info=True)
        if wcol is None:
            LOG.warning("SWOT RiverSP: could not find a WSE column in %s (provide --swot-wse-field)", p)
            continue

        # Optional quality filter (defensive):
        # If qual_field is provided and present, use it. Otherwise attempt to find a reasonable
        # quality/flag column and apply a conservative filter (keep zeros/False/NaN; drop >0/True).
        qcol = qual_field if (qual_field and qual_field in gdf.columns) else None
        if qcol is None:
            # Common candidate names in RiverSP-like exports
            cand_q = []
            for c in gdf.columns:
                if c == "geometry":
                    continue
                cl = str(c).lower()
                if any(k in cl for k in ("qual", "quality", "flag", "bad", "reject", "valid")):
                    cand_q.append(c)
            # Prefer numeric/boolean columns
            for c in cand_q:
                try:
                    dt = str(gdf[c].dtype).lower()
                    if dt.startswith(("int", "float", "bool")):
                        qcol = c
                        break
                except Exception:
                    continue
        if qcol is not None:
            try:
                qq = gdf[qcol]
                if str(qq.dtype).lower().startswith("bool"):
                    good = (qq.isna()) | (~qq.astype(bool))
                else:
                    qv = pd.to_numeric(qq, errors="coerce")
                    # Keep NaN and 0; drop positive values (treat as bad flags)
                    good = qv.isna() | (qv <= 0.0)
                gdf = gdf[good].copy()
            except Exception:
                log.debug("ignored", exc_info=True)

# Iterate rows
        for geom, wv in zip(gdf.geometry, gdf[wcol]):
            if geom is None or geom.is_empty:
                continue
            try:
                wv = float(wv) + float(wse_offset_m or 0.0)
            except Exception:
                continue
            if not np.isfinite(wv):
                continue

            # Point: use directly; Line: use midpoint
            try:
                if geom.geom_type == "Point":
                    x, y = float(geom.x), float(geom.y)
                elif geom.geom_type in ("LineString", "LinearRing"):
                    line = geom
                    pt = line.interpolate(0.5, normalized=True) if hasattr(line, "interpolate") else None
                    if pt is None or pt.is_empty:
                        continue
                    x, y = float(pt.x), float(pt.y)
                elif geom.geom_type == "MultiLineString":
                    # pick longest part midpoint
                    parts = list(geom.geoms)
                    if not parts:
                        continue
                    parts = sorted(parts, key=lambda g: float(getattr(g, "length", 0.0)), reverse=True)
                    line = parts[0]
                    pt = line.interpolate(0.5, normalized=True)
                    x, y = float(pt.x), float(pt.y)
                else:
                    # fallback: representative point
                    rp = geom.representative_point()
                    x, y = float(rp.x), float(rp.y)
            except Exception:
                continue

            pts.append((x, y))
            vals.append(wv)

    if not pts:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    # Deduplicate exact coincident points (common when exporting mixed node/reach layers).
    # Use a robust median for duplicated locations.
    dd = {}
    for (x, y), wv in zip(pts, vals):
        key = (float(x), float(y))
        dd.setdefault(key, []).append(float(wv))
    pts2 = []
    vals2 = []
    for (x, y), vv in dd.items():
        vv = np.asarray(vv, dtype=float)
        vv = vv[np.isfinite(vv)]
        if vv.size == 0:
            continue
        pts2.append((float(x), float(y)))
        vals2.append(float(np.nanmedian(vv)))
    if not pts2:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    return np.asarray(pts2, dtype=float), np.asarray(vals2, dtype=float)



def _estimate_swot_vertical_offset(
    swot_pts: np.ndarray,
    swot_wse: np.ndarray,
    bank_wse: np.ndarray,
    template_transform,
    channel: Optional[np.ndarray] = None,
    mode: str = "median_mad",
    min_samples: int = 25,
    mad_z: float = 3.5,
    max_abs_m: float = 10.0,
):
    """Estimate a constant vertical offset between SWOT WSE and bank-derived WSE.

    This is a pragmatic vertical-datum reconciliation step that avoids depending on external
    VDatum tooling. It estimates a robust constant offset using matched samples:
        offset ≈ median(SWOT_WSE - BankProfile_WSE)

    If mode is 'median_mad', we apply a MAD-based outlier rejection before taking the median.
    The returned offset is what should be SUBTRACTED from SWOT WSE to align it to bank_profile WSE.
    """
    if swot_pts is None or swot_wse is None:
        return 0.0, 0, {}
    if len(swot_pts) == 0:
        return 0.0, 0, {}
    try:
        from rasterio.transform import rowcol
    except Exception:
        rowcol = None

    # Sample bank_wse at SWOT point locations (nearest pixel)
    rows = []
    cols = []
    try:
        if rowcol is not None:
            rr, cc = rowcol(template_transform, swot_pts[:, 0], swot_pts[:, 1], op=round)
            rows = np.asarray(rr, dtype=int)
            cols = np.asarray(cc, dtype=int)
        else:
            # manual for affine: x = c + a*col + b*row ; y = f + d*col + e*row (north-up: b=d=0)
            a = float(template_transform.a)
            e = float(template_transform.e)
            c0 = float(template_transform.c)
            f0 = float(template_transform.f)
            cols = np.asarray(np.round((swot_pts[:, 0] - c0) / a), dtype=int)
            rows = np.asarray(np.round((swot_pts[:, 1] - f0) / e), dtype=int)
    except Exception:
        return 0.0, 0, {}

    h, w = bank_wse.shape
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if channel is not None:
        try:
            inside = inside & channel[rows.clip(0, h-1), cols.clip(0, w-1)].astype(bool)
        except Exception:
            log.debug("ignored", exc_info=True)
    if not np.any(inside):
        return 0.0, 0, {}

    b = bank_wse[rows[inside], cols[inside]].astype(float)
    s = swot_wse[inside].astype(float)

    good = np.isfinite(b) & np.isfinite(s)
    if not np.any(good):
        return 0.0, 0, {}

    diffs = (s[good] - b[good]).astype(float)
    n0 = int(diffs.size)
    if n0 < int(min_samples or 0):
        return 0.0, n0, {"n": n0, "reason": "min_samples"}

    diffs_used = diffs
    if (mode or "").lower() in ("median_mad", "mad", "robust"):
        med = float(np.nanmedian(diffs))
        mad = float(np.nanmedian(np.abs(diffs - med)))
        if mad > 0:
            # 1.4826*MAD ≈ sigma for normal
            sigma = 1.4826 * mad
            keep = np.abs(diffs - med) <= float(mad_z or 3.5) * sigma
            diffs_used = diffs[keep]
        else:
            diffs_used = diffs

    if diffs_used.size == 0:
        return 0.0, n0, {"n": n0, "reason": "all_rejected"}

    off = float(np.nanmedian(diffs_used))
    if max_abs_m is not None and float(max_abs_m) > 0:
        if abs(off) > float(max_abs_m):
            off_clamped = float(np.clip(off, -float(max_abs_m), float(max_abs_m)))
            return off_clamped, int(diffs_used.size), {"n": int(diffs_used.size), "offset_raw": off, "offset_clamped": off_clamped}
    return off, int(diffs_used.size), {"n": int(diffs_used.size), "offset": off, "offset_p10": float(np.nanpercentile(diffs_used, 10)), "offset_p90": float(np.nanpercentile(diffs_used, 90))}


def _read_soundings_file(path: Path):
    """Read an XYZ-ish file into x/y/z arrays.

    Supported:
      - .xyz/.txt/.csv/.dat : first 3 numeric columns (header optional)
      - .gpkg/.shp/.geojson/.json : point geometries + a numeric Z/depth/elev/value column
      - .parquet : expects x/y/z columns (as produced by xs_infer_bathy_raster.py --write-soundings-subset)
    """
    suf = path.suffix.lower()
    if suf == '.parquet':
        df = pd.read_parquet(path)
        # Common columns from our subset writer: x,y,z
        cols = {c.lower(): c for c in df.columns}
        xc = cols.get('x') or cols.get('lon') or cols.get('longitude')
        yc = cols.get('y') or cols.get('lat') or cols.get('latitude')
        zc = cols.get('z') or cols.get('depth') or cols.get('elev') or cols.get('elevation')
        if not (xc and yc and zc):
            raise ValueError(f"Parquet soundings must contain x/y/z columns: {path}")
        x = pd.to_numeric(df[xc], errors='coerce').to_numpy(dtype='float64')
        y = pd.to_numeric(df[yc], errors='coerce').to_numpy(dtype='float64')
        z = pd.to_numeric(df[zc], errors='coerce').to_numpy(dtype='float64')
        return x, y, z
    if suf in ('.gpkg', '.shp', '.geojson', '.json'):
        gdf = gpd.read_file(path)
        if gdf.empty:
            return [], [], []
        gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].copy()
        gdf = gdf[gdf.geometry.geom_type == 'Point'].copy()
        if gdf.empty:
            return [], [], []
        cand_cols = ['z','Z','depth','Depth','DEPTH','elev','elevation','Elevation','value','Value']
        zcol = None
        for c in cand_cols:
            if c in gdf.columns:
                zcol = c
                break
        if zcol is None:
            for c in gdf.columns:
                if c == 'geometry':
                    continue
                try:
                    if str(gdf[c].dtype).startswith(('int','float')):
                        zcol = c
                        break
                except Exception:
                    log.debug("ignored", exc_info=True)
        if zcol is None:
            raise ValueError(f"Could not find a numeric Z/depth/elev column in {path}")
        x = gdf.geometry.x.to_numpy(dtype='float64')
        y = gdf.geometry.y.to_numpy(dtype='float64')
        z = pd.to_numeric(gdf[zcol], errors='coerce').to_numpy(dtype='float64')
        return x, y, z

    # Text-ish (header optional, delimiter inferred)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline().strip()
    has_alpha = any(ch.isalpha() for ch in first)
    if has_alpha:
        df = pd.read_csv(path, sep=None, engine='python', comment='#')
        cols = {c.lower(): c for c in df.columns}
        xc = cols.get('x') or cols.get('lon') or cols.get('longitude')
        yc = cols.get('y') or cols.get('lat') or cols.get('latitude')
        zc = cols.get('z') or cols.get('depth') or cols.get('elev') or cols.get('elevation')
        if xc and yc and zc:
            x = pd.to_numeric(df[xc], errors='coerce').to_numpy(dtype='float64')
            y = pd.to_numeric(df[yc], errors='coerce').to_numpy(dtype='float64')
            z = pd.to_numeric(df[zc], errors='coerce').to_numpy(dtype='float64')
            return x, y, z
        df2 = df.apply(pd.to_numeric, errors='coerce')
        df2 = df2.dropna(axis=1, how='all')
        if df2.shape[1] < 3:
            return [], [], []
        return df2.iloc[:,0].to_numpy(dtype='float64'), df2.iloc[:,1].to_numpy(dtype='float64'), df2.iloc[:,2].to_numpy(dtype='float64')

    df = pd.read_csv(path, sep=None, engine='python', header=None, comment='#')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')
    if df.shape[1] < 3:
        return [], [], []
    return df.iloc[:,0].to_numpy(dtype='float64'), df.iloc[:,1].to_numpy(dtype='float64'), df.iloc[:,2].to_numpy(dtype='float64')


def _coerce_depth(z, mode: str):
    """Convert raw Z into positive-down depths.

    mode:
      - auto      : if most values are negative, assume negative-down depth and negate.
      - depth_pos : assume z is already positive-down depth
      - depth_neg : assume z is negative-down depth
    """
    z = pd.to_numeric(pd.Series(z), errors='coerce').to_numpy(dtype='float64')
    z = z[pd.notna(z)]
    if z.size == 0:
        return z
    mode = (mode or 'auto').strip().lower()
    if mode == 'depth_pos':
        d = z.copy()
    elif mode == 'depth_neg':
        d = -z
    else:
        frac_neg = float((z < 0.0).mean())
        d = (-z) if frac_neg >= 0.7 else z.copy()
    d = pd.to_numeric(pd.Series(d), errors='coerce').to_numpy(dtype='float64')
    d = pd.Series(d).where(pd.Series(d).notna(), other=pd.NA).to_numpy(dtype='float64')
    d = pd.Series(d).abs().to_numpy(dtype='float64')
    return d



def _soundings_to_grids(
    sounding_files,
    soundings_crs: str | None,
    template_crs,
    transform: rasterio.Affine,
    channel: np.ndarray,
    r: np.ndarray,
    shape_exp: float,
    dmax_min_m: float,
    dmax_max_m: float,
    mode: str,
    min_r: float,
    max_points: int = 2_000_000,
    sample_seed: int = 0,
    cell_percentile: float | None = None,
    wse_map: np.ndarray | None = None,
    diag_json_path: "Path | str | None" = None,
):
    """Rasterize external soundings onto the template grid.

    Returns three grids (float32, NaN where empty):
      - depth_obs_grid (positive-down, meters) when derivable
      - dmax_implied_grid (meters) inferred from depth and r-position
      - bed_obs_grid (meters, same vertical datum as DEM) when mode == 'bed_elev'

    Notes:
      * Only soundings that fall inside the channel mask are used.
      * For bed elevations: we keep the **minimum** elevation per cell (deeper bed).
      * For depths: we keep the **maximum** depth per cell (deeper).
      * dmax inference uses: Dmax = depth / max(r, min_r)^shape_exp, then clamps.
    """
    H, W = channel.shape
    try:
        u = np.unique(channel)
        LOG.info("Channel mask unique values: %s", u.tolist() if hasattr(u,'tolist') else str(u))
    except Exception:
        LOG.debug("ignored", exc_info=True)

    depth_grid = np.full(channel.shape, np.nan, dtype="float32")
    dmax_grid = np.full(channel.shape, np.nan, dtype="float32")
    bed_grid = np.full(channel.shape, np.nan, dtype="float32")

    if not sounding_files:
        return depth_grid, dmax_grid, bed_grid

    # If soundings_crs is not provided, attempt to infer it from parquet metadata
    # produced by xs_infer_bathy_raster.py (it stores a 'crs' column).
    if (not soundings_crs) and sounding_files:
        for fp in sounding_files:
            try:
                pth = Path(fp)
                if pth.suffix.lower() == '.parquet' and pth.exists():
                    import pandas as _pd
                    cols = _pd.read_parquet(pth, nrows=0).columns
                    if 'crs' in cols:
                        dfc = _pd.read_parquet(pth, columns=['crs'])
                        vals = dfc['crs'].dropna().astype(str).str.strip()
                        vals = vals[vals != '']
                        if len(vals) > 0:
                            soundings_crs = vals.iloc[0]
                            LOG.info('Inferred soundings_crs from parquet: %s', soundings_crs)
                            break
            except Exception:
                continue

    # Optional CRS transform into template CRS
    xform = None
    if soundings_crs:
        src = CRS.from_user_input(soundings_crs)
        dst = CRS.from_user_input(template_crs)
        if src != dst:
            xform = Transformer.from_crs(src, dst, always_xy=True)

    def _template_bbox_xy() -> Optional[Tuple[float, float, float, float]]:
        """Return template bbox as (xmin, xmax, ymin, ymax) in template CRS."""
        try:
            # rasterio.transform.array_bounds returns (xmin, ymin, xmax, ymax) in the raster CRS
            xmin, ymin, xmax, ymax = array_bounds(H, W, transform)
            return float(xmin), float(xmax), float(ymin), float(ymax)
        except Exception:
            return None

    def _in_bbox_ratio(xv: np.ndarray, yv: np.ndarray) -> float:
        """Fraction of finite points that fall inside template bbox."""
        bb = _template_bbox_xy()
        if bb is None:
            return float("nan")
        xmin, xmax, ymin, ymax = bb
        m = np.isfinite(xv) & np.isfinite(yv)
        if int(np.count_nonzero(m)) == 0:
            return float("nan")
        xv2, yv2 = xv[m], yv[m]
        inside = (xv2 >= xmin) & (xv2 <= xmax) & (yv2 >= ymin) & (yv2 <= ymax)
        return float(np.count_nonzero(inside)) / float(len(xv2))

    try:
        bb = _template_bbox_xy()
        if bb is not None:
            xmin, xmax, ymin, ymax = bb
            LOG.info('Template bounds (crs=%s): x=[%.3f, %.3f] y=[%.3f, %.3f]', str(template_crs), xmin, xmax, ymin, ymax)
    except Exception:
        log.debug("ignored", exc_info=True)

    xs_all, ys_all, zs_all = [], [], []
    sizes = []
    for fp in sounding_files:
        pth = Path(fp)
        if not pth.exists():
            LOG.warning("Soundings file not found: %s", pth)
            continue
        try:
            x, y, z = _read_soundings_file(pth)
            if len(x) == 0:
                continue
            xs_all.append(np.asarray(x, dtype="float64"))
            ys_all.append(np.asarray(y, dtype="float64"))
            zs_all.append(np.asarray(z, dtype="float64"))
            sizes.append(int(len(x)))
        except Exception as e:
            LOG.warning("Failed reading soundings %s: %s", pth, e)

    if not xs_all:
        return depth_grid, dmax_grid, bed_grid

    # Downsample guard (prevents OOM-kills for very large authoritative point clouds).
    # Preserve per-file proportions as a proxy for per-source composition.
    total_n = int(sum(sizes))
    if max_points and max_points > 0 and total_n > int(max_points):
        rng = np.random.default_rng(int(sample_seed) if sample_seed is not None else 0)
        new_xs, new_ys, new_zs = [], [], []
        for x_i, y_i, z_i, n_i in zip(xs_all, ys_all, zs_all, sizes):
            frac = float(n_i) / float(max(total_n, 1))
            take = max(1, int(round(frac * int(max_points))))
            if n_i <= take:
                new_xs.append(x_i)
                new_ys.append(y_i)
                new_zs.append(z_i)
            else:
                idx = rng.choice(np.arange(n_i, dtype="int64"), size=take, replace=False)
                new_xs.append(x_i[idx])
                new_ys.append(y_i[idx])
                new_zs.append(z_i[idx])
        xs_all, ys_all, zs_all = new_xs, new_ys, new_zs
        LOG.warning("Soundings downsampled to ~%d points (from %d) to avoid OOM (soundings_max_points=%d).",
                    int(sum(len(a) for a in xs_all)), total_n, int(max_points))

    x = np.concatenate(xs_all)
    y = np.concatenate(ys_all)
    z = np.concatenate(zs_all)

    try:
        sx0, sx1 = float(np.nanmin(x)), float(np.nanmax(x))
        sy0, sy1 = float(np.nanmin(y)), float(np.nanmax(y))
        LOG.info('Soundings bounds (raw): x=[%.3f, %.3f] y=[%.3f, %.3f] n=%d', sx0, sx1, sy0, sy1, int(len(x)))
        r0 = _in_bbox_ratio(x, y)
        if np.isfinite(r0):
            LOG.info('Soundings inside template bbox (raw): %.3f', float(r0))
    except Exception:
        log.debug("ignored", exc_info=True)

    n_loaded = int(x.size)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    LOG.info("Soundings: loaded=%d finite=%d", n_loaded, int(x.size))
    if x.size == 0:
        return depth_grid, dmax_grid, bed_grid

    # Apply declared CRS transform (if any)
    if xform is not None:
        x2, y2 = xform.transform(x.tolist(), y.tolist())
        x = np.asarray(x2, dtype="float64")
        y = np.asarray(y2, dtype="float64")

    # Deterministic CRS/axis-order self-check:
    # If the declared CRS yields almost no points inside the template bbox, try alternative
    # geographic interpretations (NAD83/WGS84 + optional axis swap) and pick the one that
    # maximizes in-bbox ratio. Only apply when improvement is overwhelming.
    try:
        r_after = _in_bbox_ratio(x, y)
        if np.isfinite(r_after):
            LOG.info('Soundings inside template bbox (post-declared-crs): %.3f', float(r_after))
        best_ratio = r_after
        best_tr = None
        best_swap = False
        if np.isfinite(r_after) and float(r_after) < 0.01:
            for s in ('EPSG:4269', 'EPSG:4326'):
                try:
                    tr = Transformer.from_crs(s, template_crs, always_xy=True)
                    # as-is
                    xx, yy = tr.transform(x.tolist(), y.tolist())
                    rr0 = _in_bbox_ratio(np.asarray(xx, dtype='float64'), np.asarray(yy, dtype='float64'))
                    if np.isfinite(rr0) and (best_ratio is None or float(rr0) > float(best_ratio)):
                        best_ratio, best_tr, best_swap = rr0, tr, False
                    # swapped
                    xx, yy = tr.transform(y.tolist(), x.tolist())
                    rr1 = _in_bbox_ratio(np.asarray(xx, dtype='float64'), np.asarray(yy, dtype='float64'))
                    if np.isfinite(rr1) and (best_ratio is None or float(rr1) > float(best_ratio)):
                        best_ratio, best_tr, best_swap = rr1, tr, True
                except Exception:
                    continue

            if best_tr is not None and np.isfinite(best_ratio) and float(best_ratio) >= 0.50:
                LOG.warning(
                    'Soundings CRS/axis mismatch detected: in_bbox=%.3f after declared CRS; using %s with swap_xy=%s (in_bbox=%.3f).',
                    float(r_after), str(getattr(best_tr, 'source_crs', 'geo')), str(best_swap), float(best_ratio)
                )
                if best_swap:
                    x_in, y_in = y.tolist(), x.tolist()
                else:
                    x_in, y_in = x.tolist(), y.tolist()
                xx, yy = best_tr.transform(x_in, y_in)
                x = np.asarray(xx, dtype='float64')
                y = np.asarray(yy, dtype='float64')
    except Exception:
        LOG.debug('Soundings CRS self-check failed; continuing.', exc_info=True)

    rr, cc = rowcol(transform, x, y)
    rr = np.asarray(rr, dtype="int64")
    cc = np.asarray(cc, dtype="int64")

    inb = (rr >= 0) & (rr < channel.shape[0]) & (cc >= 0) & (cc < channel.shape[1])
    n_inb = int(np.count_nonzero(inb))
    LOG.info(
        "Soundings: in_template_bbox=%d (grid=%dx%d channel_true_pixels=%d)",
        n_inb,
        int(channel.shape[1]),
        int(channel.shape[0]),
        int(np.count_nonzero(channel)),
    )
    rr, cc, z = rr[inb], cc[inb], z[inb]
    if rr.size == 0:
        return depth_grid, dmax_grid, bed_grid

    # Row/col range diagnostics for in-bounds points (helps catch axis swaps / transform mismatch)
    try:
        LOG.info(
            "Soundings row/col range (in-bounds): row=[%d..%d] col=[%d..%d]",
            int(rr.min()),
            int(rr.max()),
            int(cc.min()),
            int(cc.max()),
        )
    except Exception:
        log.debug("ignored", exc_info=True)

    in_ch = channel[rr, cc]
    n_in_ch = int(np.count_nonzero(in_ch))
    LOG.info("Soundings: in_channel_mask=%d", n_in_ch)
    if n_in_ch == 0:
        # Provide a small spread sample so we can see whether points are systematically off.
        try:
            samp_n = min(12, int(rr.size))
            if samp_n > 0:
                idx = np.linspace(0, int(rr.size) - 1, num=samp_n, dtype=int)
                LOG.info(
                    "Soundings sample (row,col,mask): %s",
                    ", ".join([f"({int(rr[i])},{int(cc[i])},{bool(in_ch[i])})" for i in idx]),
                )
        except Exception:
            log.debug("ignored", exc_info=True)

        # Distance-to-channel diagnostic (pixels). Helps distinguish "mask too narrow" vs "CRS/transform mismatch".
        # Prefer scipy's distance transform when available, but fall back to a deterministic bounded brute-force
        # so the diagnostic still works in minimal environments.
        d = None
        try:
            from scipy.ndimage import distance_transform_edt

            dist_px = distance_transform_edt(~channel)
            d = dist_px[rr, cc]
        except Exception:
            d = None

        if d is None:
            try:
                ch_rc = np.column_stack(np.nonzero(channel))
                if ch_rc.size and rr.size:
                    rs = np.random.RandomState(0)
                    max_ch = 5000
                    if ch_rc.shape[0] > max_ch:
                        ch_rc = ch_rc[rs.choice(ch_rc.shape[0], size=max_ch, replace=False)]
                    max_snd = 2000
                    if rr.size > max_snd:
                        idx2 = rs.choice(rr.size, size=max_snd, replace=False)
                        rc_s = np.column_stack([rr[idx2], cc[idx2]]).astype(np.float64)
                    else:
                        rc_s = np.column_stack([rr, cc]).astype(np.float64)
                    ch_rc_f = ch_rc.astype(np.float64)
                    diff = rc_s[:, None, :] - ch_rc_f[None, :, :]
                    d = np.sqrt((diff * diff).sum(axis=2)).min(axis=1)
            except Exception as e:
                LOG.debug("Soundings distance_to_channel_px fallback failed: %s", str(e))
                d = None

        if d is not None and np.size(d):
            try:
                LOG.info(
                    "Soundings distance_to_channel_px: min=%.2f p50=%.2f p95=%.2f max=%.2f",
                    float(np.min(d)),
                    float(np.percentile(d, 50)),
                    float(np.percentile(d, 95)),
                    float(np.max(d)),
                )
            except Exception:
                log.debug("ignored", exc_info=True)

    # Write a diagnostic receipt so CRS/rowcol/mask overlap is provable from artifacts.
    # This is intentionally written even when 0 points fall in the channel mask.
    if diag_json_path is not None:
        try:
            diag_path = Path(diag_json_path)

            # Swapped-axis diagnostic: if any upstream bug swapped (x,y)->(y,x),
            # this may show non-zero overlaps.
            rr_sw, cc_sw = rasterio.transform.rowcol(transform, y, x)
            rr_sw = np.asarray(rr_sw)
            cc_sw = np.asarray(cc_sw)
            inb_sw = (rr_sw >= 0) & (rr_sw < H) & (cc_sw >= 0) & (cc_sw < W)
            n_inb_sw = int(np.count_nonzero(inb_sw))
            n_inch_sw = (
                int(np.count_nonzero(channel[rr_sw[inb_sw], cc_sw[inb_sw]]))
                if n_inb_sw
                else 0
            )

            receipt = {
                "soundings": {
                    "n_total": int(len(x)),
                    "x_min": float(np.min(x)) if len(x) else None,
                    "x_max": float(np.max(x)) if len(x) else None,
                    "y_min": float(np.min(y)) if len(y) else None,
                    "y_max": float(np.max(y)) if len(y) else None,
                    "z_min": float(np.min(z)) if len(z) else None,
                    "z_max": float(np.max(z)) if len(z) else None,
                },
                "grid": {
                    "crs": str(template_crs) if template_crs is not None else None,
                    "shape": [int(H), int(W)],
                    "transform_gdal": [float(v) for v in transform.to_gdal()],
                    "bounds": {
                        "left": float(bounds.left),
                        "bottom": float(bounds.bottom),
                        "right": float(bounds.right),
                        "top": float(bounds.top),
                    },
                },
                "channel": {
                    "true_count": int(np.count_nonzero(channel)),
                    "false_count": int(channel.size - np.count_nonzero(channel)),
                },
                "rowcol": {
                    "n_in_bounds": int(np.count_nonzero(inb)),
                    "n_in_channel": int(np.count_nonzero(in_ch)),
                    "row_min": int(np.min(rr)) if rr.size else None,
                    "row_max": int(np.max(rr)) if rr.size else None,
                    "col_min": int(np.min(cc)) if cc.size else None,
                    "col_max": int(np.max(cc)) if cc.size else None,
                    "n_in_bounds_swapped": n_inb_sw,
                    "n_in_channel_swapped": n_inch_sw,
                },
            }
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            diag_path.write_text(json.dumps(receipt, indent=2) + "\n")
            LOG.info("Soundings/channel receipt written: %s", str(diag_path))
        except Exception as e:
            LOG.warning("Failed to write soundings/channel receipt: %s", str(e))
    rr, cc, z = rr[in_ch], cc[in_ch], z[in_ch]
    if rr.size == 0:
        return depth_grid, dmax_grid, bed_grid

    mode_l = (mode or "auto").strip().lower()

    # --- Bed elevation mode (NAVD88 bed elevations, etc.) ---
    if mode_l == "bed_elev":
        bed = pd.to_numeric(pd.Series(z), errors="coerce").to_numpy(dtype="float64")
        ok = np.isfinite(bed)
        rr2, cc2, bed = rr[ok], cc[ok], bed[ok]
        if rr2.size == 0:
            return depth_grid, dmax_grid, bed_grid

        
        # Aggregate bed elevation per cell with a percentile to reduce outliers.
        # For NAVD88 bed elevations: lower percentile -> deeper (more conservative) but robust.
        pct = 5.0 if cell_percentile is None else float(cell_percentile)
        idx = (rr2.astype(np.int64) * W + cc2.astype(np.int64))
        df = pd.DataFrame({"idx": idx, "v": bed.astype(np.float64)})
        q = df.groupby("idx")["v"].quantile(pct / 100.0)
        bed_grid.ravel()[q.index.to_numpy(dtype=np.int64)] = q.to_numpy(dtype=np.float64)
        # If we have a WSE proxy, infer depth + implied Dmax from bed elevations
        if wse_map is not None:
            wse = wse_map[rr2, cc2].astype("float64")
            d = (wse - bed).astype("float64")
            ok2 = np.isfinite(d) & (d >= 0.0)
            rr3, cc3, d = rr2[ok2], cc2[ok2], d[ok2]
            if rr3.size > 0:
                
                # Aggregate depths per cell using a high percentile (deeper) to reduce shallow outliers.
                pct_d = 95.0 if cell_percentile is None else (100.0 - float(cell_percentile))
                idx3 = (rr3.astype(np.int64) * W + cc3.astype(np.int64))
                df3 = pd.DataFrame({"idx": idx3, "v": d.astype(np.float64)})
                q3 = df3.groupby("idx")["v"].quantile(pct_d / 100.0)
                depth_grid.ravel()[q3.index.to_numpy(dtype=np.int64)] = q3.to_numpy(dtype=np.float64)
                rvals = r[rr3, cc3].astype("float64")
                ruse = np.maximum(rvals, float(min_r))
                denom = np.power(ruse, float(shape_exp)) + 1e-6
                dmax_imp = (d / denom).astype("float64")
                dmax_imp = np.clip(dmax_imp, float(dmax_min_m), float(dmax_max_m))

                
                # Aggregate implied dmax per cell using the same high-percentile rule as depths.
                pct_dm = 95.0 if cell_percentile is None else (100.0 - float(cell_percentile))
                idx4 = (rr3.astype(np.int64) * W + cc3.astype(np.int64))
                df4 = pd.DataFrame({"idx": idx4, "v": dmax_imp.astype(np.float64)})
                q4 = df4.groupby("idx")["v"].quantile(pct_dm / 100.0)
                dmax_grid.ravel()[q4.index.to_numpy(dtype=np.int64)] = q4.to_numpy(dtype=np.float64)
        else:
            LOG.warning("soundings-mode=bed_elev but no wse_map provided; skipping depth/Dmax inference from soundings.")

        return depth_grid, dmax_grid, bed_grid

    # --- Depth mode (auto/depth_pos/depth_neg) ---
    z_num = pd.to_numeric(pd.Series(z), errors="coerce").to_numpy(dtype="float64")
    ok = np.isfinite(z_num)
    rr2, cc2, z_num = rr[ok], cc[ok], z_num[ok]
    if rr2.size == 0:
        return depth_grid, dmax_grid, bed_grid

    if mode_l == "depth_pos":
        d = z_num
    elif mode_l == "depth_neg":
        d = -z_num
    else:
        frac_neg = float((z_num < 0.0).mean())
        d = (-z_num) if frac_neg >= 0.7 else z_num
    d = np.abs(d).astype("float64")
    ok2 = np.isfinite(d)
    rr3, cc3, d = rr2[ok2], cc2[ok2], d[ok2]
    if rr3.size == 0:
        return depth_grid, dmax_grid, bed_grid

    
    # Aggregate depth per cell with percentile to reduce outliers.
    pct_d = 95.0 if cell_percentile is None else float(cell_percentile)
    idx3 = (rr3.astype(np.int64) * W + cc3.astype(np.int64))
    df3 = pd.DataFrame({"idx": idx3, "v": d.astype(np.float64)})
    q3 = df3.groupby("idx")["v"].quantile(pct_d / 100.0)
    depth_grid.ravel()[q3.index.to_numpy(dtype=np.int64)] = q3.to_numpy(dtype=np.float64)
    rvals = r[rr3, cc3].astype("float64")
    ruse = np.maximum(rvals, float(min_r))
    denom = np.power(ruse, float(shape_exp)) + 1e-6
    dmax_imp = (d / denom).astype("float64")
    dmax_imp = np.clip(dmax_imp, float(dmax_min_m), float(dmax_max_m))

    
    # Aggregate implied dmax per cell using a high percentile (deeper) to reduce shallow outliers.
    pct_dm = 95.0 if cell_percentile is None else float(cell_percentile)
    idx4 = (rr3.astype(np.int64) * W + cc3.astype(np.int64))
    df4 = pd.DataFrame({"idx": idx4, "v": dmax_imp.astype(np.float64)})
    q4 = df4.groupby("idx")["v"].quantile(pct_dm / 100.0)
    dmax_grid.ravel()[q4.index.to_numpy(dtype=np.int64)] = q4.to_numpy(dtype=np.float64)
    return depth_grid, dmax_grid, bed_grid


def _build_residual_adjustment(
    resid_src: np.ndarray,
    src_valid_mask: np.ndarray,
    channel_mask: np.ndarray,
    max_dist_m: float,
    blend_sigma_m: float,
    pixel_size: float,
) -> np.ndarray:
    """Build a smooth residual adjustment field from sparse anchors.

    Steps:
      1) Nearest-neighbor propagate residuals within the channel using EDT indices.
      2) Apply a distance-decay weight that tapers residuals to 0 at max_dist_m.
      3) Optionally apply Gaussian normalized-convolution smoothing (num/den) to reduce seams.
    """
    if (max_dist_m is None) or (max_dist_m <= 0) or (src_valid_mask is None) or (not np.any(src_valid_mask)):
        return np.zeros_like(resid_src, dtype="float32")

    inv = np.ones(resid_src.shape, dtype=np.uint8)
    inv[src_valid_mask] = 0
    dist, inds = distance_transform_edt(inv, return_indices=True)
    dist_m = dist.astype("float32") * float(pixel_size)

    ny, nx = inds[0], inds[1]
    resid_nn = resid_src[ny, nx].astype("float32")

    valid = np.isfinite(resid_nn) & channel_mask & (dist_m <= float(max_dist_m))
    w = np.clip(1.0 - (dist_m / (float(max_dist_m) + 1e-6)), 0.0, 1.0).astype("float32")
    w = np.where(valid, w, 0.0).astype("float32")
    resid_nn = np.where(np.isfinite(resid_nn), resid_nn, 0.0).astype("float32")

    adj = (resid_nn * w).astype("float32")

    if (blend_sigma_m is not None) and (blend_sigma_m > 0):
        sigma_px = float(blend_sigma_m) / max(float(pixel_size), 1e-9)
        if sigma_px > 0.01:
            num = gaussian_filter(adj, sigma=sigma_px, mode="nearest")
            den = gaussian_filter(w, sigma=sigma_px, mode="nearest")
            adj = np.where(den > 1e-6, (num / den), 0.0).astype("float32")

    return np.where(channel_mask & (dist_m <= float(max_dist_m)), adj, 0.0).astype("float32")


def _build_value_field_from_anchors(
    val_src: np.ndarray,
    src_valid_mask: np.ndarray,
    channel_mask: np.ndarray,
    max_dist_m: float,
    blend_sigma_m: float,
    pixel_size: float,
) -> np.ndarray:
    """Build a smooth value field from sparse anchors inside the channel.

    This is like _build_residual_adjustment, but returns a *value* surface rather than an additive adjustment.
    It uses:
      1) nearest-neighbor propagation (EDT indices) within the channel,
      2) distance-decay taper to 0 weight at max_dist_m,
      3) optional normalized-convolution Gaussian smoothing (num/den) to reduce seams.

    Returns:
      float32 array with NaN outside the influenced channel region.
    """
    if (max_dist_m is None) or (max_dist_m <= 0) or (src_valid_mask is None) or (not np.any(src_valid_mask)):
        return np.full_like(val_src, np.nan, dtype="float32")

    inv = np.ones(val_src.shape, dtype=np.uint8)
    inv[src_valid_mask] = 0
    dist, inds = distance_transform_edt(inv, return_indices=True)
    dist_m = dist.astype("float32") * float(pixel_size)

    ny, nx = inds[0], inds[1]
    v_nn = val_src[ny, nx].astype("float32")

    valid = np.isfinite(v_nn) & channel_mask & (dist_m <= float(max_dist_m))
    w = np.clip(1.0 - (dist_m / (float(max_dist_m) + 1e-6)), 0.0, 1.0).astype("float32")
    w = np.where(valid, w, 0.0).astype("float32")

    v0 = np.where(np.isfinite(v_nn), v_nn, 0.0).astype("float32")

    if (blend_sigma_m is not None) and (blend_sigma_m > 0):
        sigma_px = float(blend_sigma_m) / max(float(pixel_size), 1e-9)
        if sigma_px > 0.01:
            num = gaussian_filter(v0 * w, sigma=sigma_px, mode="nearest")
            den = gaussian_filter(w, sigma=sigma_px, mode="nearest")
            out = np.full_like(num, np.nan, dtype="float32")
            np.divide(num, den, out=out, where=(den > 1e-6))
            out = out.astype("float32")
        else:
            out = np.where(w > 0, v_nn.astype("float32"), np.nan).astype("float32")
    else:
        out = np.where(w > 0, v_nn.astype("float32"), np.nan).astype("float32")

    out = np.where(channel_mask & (dist_m <= float(max_dist_m)), out, np.nan).astype("float32")
    return out



def _densify_linestring(ls: LineString, step_m: float) -> LineString:
    """Return a densified LineString with vertices roughly every step_m (projected CRS)."""
    if ls is None or ls.is_empty:
        return ls
    if step_m is None or step_m <= 0:
        return ls
    try:
        length = float(ls.length)
    except Exception:
        return ls
    if not np.isfinite(length) or length <= step_m:
        return ls
    n = max(2, int(math.ceil(length / float(step_m))) + 1)
    # interpolate includes endpoints
    pts = [ls.interpolate(i / (n - 1), normalized=True) for i in range(n)]
    return LineString([(p.x, p.y) for p in pts])


def _iter_lines(geom):
    """Yield LineString geometries from LineString/MultiLineString."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            if isinstance(g, LineString) and (not g.is_empty):
                yield g


def _rasterize_tangent_and_curvature(
    rivers_gdf: "gpd.GeoDataFrame",
    transform,
    out_shape,
    densify_step_m: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize centerline tangent (tx, ty) and signed curvature (1/m) onto centerline pixels.

    Strategy:
      1) Densify each flowline to a regular step.
      2) For each interior vertex i, estimate:
         - tangent as normalized (p_{i+1} - p_{i-1})
         - signed curvature k using the circumcircle formula:
             k = 2*Area / (a*b*c)   (signed by cross product)
      3) Accumulate (tx,ty,k) into raster pixels; average where multiple samples land in the same cell.

    Returns:
      tx_r, ty_r, k_r: float32 rasters with NaN where undefined (non-centerline).
    """
    ny, nx = out_shape
    tx_sum = np.zeros((ny, nx), dtype="float64")
    ty_sum = np.zeros((ny, nx), dtype="float64")
    k_sum = np.zeros((ny, nx), dtype="float64")
    cnt = np.zeros((ny, nx), dtype="int32")

    # Inverse transform helper: map x,y -> row,col (fractional)
    inv = ~transform

    def _accum_point(x, y, tx, ty, k):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(tx) and np.isfinite(ty) and np.isfinite(k)):
            return
        col_f, row_f = inv * (x, y)
        r = int(round(row_f))
        c = int(round(col_f))
        if 0 <= r < ny and 0 <= c < nx:
            tx_sum[r, c] += float(tx)
            ty_sum[r, c] += float(ty)
            k_sum[r, c] += float(k)
            cnt[r, c] += 1

    for geom in rivers_gdf.geometry:
        for ls in _iter_lines(geom):
            ls2 = _densify_linestring(ls, densify_step_m)
            coords = np.asarray(ls2.coords, dtype="float64")
            if coords.shape[0] < 3:
                continue
            # Compute curvature/tangent at interior points
            for i in range(1, coords.shape[0] - 1):
                x0, y0 = coords[i - 1]
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]

                v1x, v1y = (x1 - x0), (y1 - y0)
                v2x, v2y = (x2 - x1), (y2 - y1)

                # Tangent from i-1 to i+1
                tx, ty = (x2 - x0), (y2 - y0)
                tnorm = math.hypot(tx, ty)
                if tnorm <= 1e-9:
                    continue
                tx /= tnorm
                ty /= tnorm

                # Side / signed area (z component of cross product)
                cross = (v1x * v2y) - (v1y * v2x)

                a = math.hypot(x1 - x0, y1 - y0)
                b = math.hypot(x2 - x1, y2 - y1)
                c = math.hypot(x2 - x0, y2 - y0)
                if a <= 1e-6 or b <= 1e-6 or c <= 1e-6:
                    continue

                area2 = cross  # 2*area with sign
                # curvature magnitude = 2*area / (a*b*c)  ; signed by area sign
                k = (area2 / (a * b * c))
                _accum_point(x1, y1, tx, ty, k)

    tx_r = np.full((ny, nx), np.nan, dtype="float32")
    ty_r = np.full((ny, nx), np.nan, dtype="float32")
    k_r = np.full((ny, nx), np.nan, dtype="float32")
    m = cnt > 0
    if np.any(m):
        tx_r[m] = (tx_sum[m] / np.maximum(cnt[m], 1)).astype("float32")
        ty_r[m] = (ty_sum[m] / np.maximum(cnt[m], 1)).astype("float32")
        k_r[m] = (k_sum[m] / np.maximum(cnt[m], 1)).astype("float32")
        # Normalize tangent vectors after averaging
        nrm = np.hypot(tx_r, ty_r).astype("float32")
        good = m & (nrm > 1e-6)
        tx_r[good] = (tx_r[good] / nrm[good]).astype("float32")
        ty_r[good] = (ty_r[good] / nrm[good]).astype("float32")
        tx_r[~good] = np.nan
        ty_r[~good] = np.nan
    return tx_r, ty_r, k_r



def _densify_linestring_to_points(ls, step_m):
    """Return a list of Points along a LineString at approximately step_m spacing (including endpoints)."""
    try:
        import numpy as np
        from shapely.geometry import Point
    except Exception:
        return []
    if ls is None or ls.length == 0:
        return []
    step = max(float(step_m), 1.0)
    n = int(ls.length // step)
    dists = [0.0] + [i * step for i in range(1, n + 1)]
    if dists[-1] < ls.length:
        dists.append(ls.length)
    pts = []
    for d in dists:
        try:
            pts.append(ls.interpolate(d))
        except Exception:
            log.debug("ignored", exc_info=True)
    return pts


def _clamp_profile_slope_curv(z, ds, max_slope=0.0, max_curv=0.0):
    """Clamp a 1D elevation profile by slope and curvature limits.

    max_slope is |dz/ds| in m/m.
    max_curv is |d2z/ds2| in 1/m (approx via second difference / ds^2).
    """
    import numpy as np
    z = np.asarray(z, dtype=float)
    if z.size < 3:
        return z
    ds = float(ds)
    out = z.copy()

    # Slope clamp (forward pass)
    if max_slope and max_slope > 0:
        dz_max = abs(float(max_slope)) * ds
        for i in range(1, out.size):
            dz = out[i] - out[i-1]
            if dz > dz_max:
                out[i] = out[i-1] + dz_max
            elif dz < -dz_max:
                out[i] = out[i-1] - dz_max
        # backward pass to reduce drift
        for i in range(out.size-2, -1, -1):
            dz = out[i] - out[i+1]
            if dz > dz_max:
                out[i] = out[i+1] + dz_max
            elif dz < -dz_max:
                out[i] = out[i+1] - dz_max

    # Curvature clamp (limit slope changes)
    if max_curv and max_curv > 0 and out.size >= 4:
        dslope_max = abs(float(max_curv)) * ds  # since dslope ~ d2z/ds2 * ds
        # work in slopes
        s = np.diff(out) / ds
        # forward clamp on slope changes
        for i in range(1, s.size):
            dslope = s[i] - s[i-1]
            if dslope > dslope_max:
                s[i] = s[i-1] + dslope_max
            elif dslope < -dslope_max:
                s[i] = s[i-1] - dslope_max
        # backward clamp
        for i in range(s.size-2, -1, -1):
            dslope = s[i] - s[i+1]
            if dslope > dslope_max:
                s[i] = s[i+1] + dslope_max
            elif dslope < -dslope_max:
                s[i] = s[i+1] - dslope_max
        # reconstruct
        out2 = np.empty_like(out)
        out2[0] = out[0]
        for i in range(1, out.size):
            out2[i] = out2[i-1] + s[i-1] * ds
        out = out2

    return out


def _apply_bed_profile_constraints(
    bed, template_profile, river_gpkg, channel_mask,
    step_m=25.0, max_slope=0.0, max_curv=0.0, strength=0.6, power=2.0,
    logger=None, debug_dir=None,
):
    """Apply longitudinal bed-profile constraints using flowline geometry as the 1D support.

    Strategy:
      1) sample bed along densified flowlines
      2) clamp the 1D profile by slope/curvature limits
      3) write a correction field anchored on skeleton pixels
      4) spread correction into channel with distance-decay weighting

    This is intentionally conservative: it nudges the bed toward a physically plausible
    longitudinal profile without overriding local anchors everywhere.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.crs import CRS
    from scipy.ndimage import distance_transform_edt

    try:
        import geopandas as gpd
        import fiona
        from shapely.geometry import LineString, MultiLineString
    except Exception as e:
        if logger:
            logger.warning("Bed profile constraints requested but geopandas/fiona not available: %s", e)
        return bed

    if strength <= 0 or (max_slope <= 0 and max_curv <= 0):
        return bed

    gpkg = str(river_gpkg)
    layers = []
    try:
        layers = list(fiona.listlayers(gpkg))
    except Exception as e:
        if logger:
            logger.warning("Failed listing layers in %s: %s", gpkg, e)
        return bed

    # pick first line layer (prefer rivers_aoi / flowlines-ish names)
    line_layer = None
    prefer = ['rivers_aoi', 'flowlines', 'nhdflowline', 'rivers']
    for name in prefer:
        if name in layers:
            try:
                g = gpd.read_file(gpkg, layer=name)
                if len(g) and g.geometry.iloc[0].geom_type.lower().endswith('linestring'):
                    line_layer = name
                    break
            except Exception:
                log.debug("ignored", exc_info=True)
    if line_layer is None:
        for name in layers:
            try:
                g = gpd.read_file(gpkg, layer=name)
                if len(g) and g.geometry.iloc[0].geom_type.lower().endswith('linestring'):
                    line_layer = name
                    break
            except Exception:
                continue

    if line_layer is None:
        if logger:
            logger.warning("No LineString layer found in river gpkg; skipping bed profile constraints")
        return bed

    gdf = gpd.read_file(gpkg, layer=line_layer)
    if gdf.empty:
        return bed

    # Reproject to template CRS
    try:
        crs = CRS.from_wkt(template_profile['crs'].to_wkt()) if hasattr(template_profile.get('crs'), 'to_wkt') else CRS.from_user_input(template_profile['crs'])
    except Exception:
        crs = CRS.from_user_input(template_profile['crs'])
    try:
        gdf = gdf.to_crs(crs)
    except Exception:
        log.debug("ignored", exc_info=True)

    transform = template_profile['transform']
    nodata = template_profile.get('nodata', -9999.0)

    seed = np.full(bed.shape, np.nan, dtype='float32')

    # helper to add sample to seed pixel (average if multiple)
    def _accum(r, c, val):
        if r < 0 or c < 0 or r >= seed.shape[0] or c >= seed.shape[1]:
            return
        if not channel_mask[r, c]:
            return
        if np.isnan(seed[r, c]):
            seed[r, c] = float(val)
        else:
            seed[r, c] = 0.5 * (seed[r, c] + float(val))

    # Build constrained profile per geometry and rasterize to seed pixels
    step = max(float(step_m), 1.0)
    n_profiles = 0
    n_pts = 0
    for geom in gdf.geometry:
        if geom is None:
            continue
        geoms = []
        if isinstance(geom, LineString):
            geoms = [geom]
        elif isinstance(geom, MultiLineString):
            geoms = list(geom.geoms)
        else:
            continue
        for ls in geoms:
            pts = _densify_linestring_to_points(ls, step)
            if len(pts) < 3:
                continue
            # sample bed at points
            rc = [rowcol(transform, p.x, p.y) for p in pts]
            z = []
            ok_rc = []
            for (r, c) in rc:
                if 0 <= r < bed.shape[0] and 0 <= c < bed.shape[1] and channel_mask[r, c]:
                    val = float(bed[r, c])
                    if np.isfinite(val) and val != nodata:
                        z.append(val)
                        ok_rc.append((r, c))
                    else:
                        z.append(np.nan)
                        ok_rc.append((r, c))
                else:
                    z.append(np.nan)
                    ok_rc.append((r, c))
            z = np.array(z, dtype=float)
            # fill small gaps by interpolation to avoid breaking constraints
            if np.all(~np.isfinite(z)):
                continue
            x = np.arange(z.size)
            m = np.isfinite(z)
            z_f = z.copy()
            if m.sum() >= 2:
                z_f[~m] = np.interp(x[~m], x[m], z[m])
            else:
                continue
            z_c = _clamp_profile_slope_curv(z_f, ds=step, max_slope=max_slope, max_curv=max_curv)
            for (r, c), val in zip(ok_rc, z_c):
                _accum(r, c, val)
            n_profiles += 1
            n_pts += len(ok_rc)

    if n_profiles == 0:
        if logger:
            logger.info("Bed profile constraints: no usable profiles in %s", line_layer)
        return bed

    # Spread correction from seed to full channel via nearest seed pixel and distance decay
    seed_mask = np.isfinite(seed)
    if seed_mask.sum() < 10:
        return bed

    # distance_transform_edt expects False=features; invert
    dist, inds = distance_transform_edt(~seed_mask, return_indices=True)
    # distance in pixels; convert to meters using pixel size
    px = abs(float(transform.a))
    d_m = dist * px

    r_idx, c_idx = inds
    seed_nn = seed[r_idx, c_idx]
    bed_nn = bed[r_idx, c_idx]
    delta = (seed_nn - bed_nn).astype('float32')

    # decay weighting
    sigma = max(step * 2.0, px)
    w = np.exp(- (d_m / sigma) ** float(power)).astype('float32')
    w *= float(strength)

    out = bed.copy().astype('float32')
    apply = channel_mask & seed_mask[r_idx, c_idx] & np.isfinite(delta)
    out[apply] = out[apply] + w[apply] * delta[apply]

    if logger:
        logger.info("Applied bed profile constraints: profiles=%d pts=%d max_slope=%.4g max_curv=%.4g strength=%.2f", n_profiles, n_pts, float(max_slope), float(max_curv), float(strength))

    if debug_dir is not None:
        try:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            from pathlib import Path
            from rasterio import open as rio_open
            def _save(name, arr, nod=-9999.0):
                prof = template_profile.copy()
                prof.update(dtype='float32', count=1, nodata=nod)
                with rio_open(str(Path(debug_dir)/name), 'w', **prof) as dst:
                    dst.write(arr.astype('float32'), 1)
            _save('bed_profile_seed.tif', np.where(seed_mask, seed, -9999.0), nod=-9999.0)
            _save('bed_profile_weight.tif', np.where(channel_mask, w, -9999.0), nod=-9999.0)
        except Exception as e:
            if logger:
                logger.warning("Failed writing bed profile debug rasters: %s", e)

    return out


def _detect_mainstem_corridor(
    args,
    channel: "np.ndarray",
    jm: "np.ndarray",
    template_profile: dict,
    transform,
    shape: tuple,
    pix: float,
    debug_corr_path=None,
    debug_jm_path=None,
    debug_ms_path=None,
) -> "Tuple[Optional[np.ndarray], Optional[np.ndarray]]":
    """
    Detect the mainstem corridor and preserve mask from the river graph.

    Returns (mainstem_corridor, preserve_mainstem) or (None, None) on failure.
    """
    preserve_mainstem: "Optional[np.ndarray]" = None
    mainstem_corridor: "Optional[np.ndarray]" = None
    try:
        gpkg = getattr(args, "river_gpkg", None)
        mainstem_attr = str(getattr(args, "mainstem_order_field", "streamorde") or "streamorde")
        if gpkg:
            import geopandas as _gpd
            from rasterio.features import rasterize as _rasterize
            from scipy.ndimage import distance_transform_edt as _edt
    
            g_edges = _gpd.read_file(gpkg, layer="graph_edges")
            template_crs = template_profile.get("crs", None)
            try:
                if template_crs is not None and getattr(g_edges, 'crs', None) is not None and str(g_edges.crs) != str(template_crs):
                    g_edges = g_edges.to_crs(template_crs)
            except Exception:
                log.debug("ignored", exc_info=True)
            if (g_edges is not None) and (not g_edges.empty):
                geoms = list(g_edges.geometry)
    
                # Optional stream order
                orders = None
                if mainstem_attr in g_edges.columns:
                    try:
                        orders = g_edges[mainstem_attr].astype("float64").to_numpy()
                    except Exception:
                        orders = None
    
                # Build adjacency graph from edge endpoints
                def _key_xy(x: float, y: float, nd: int = 6):
                    return (round(float(x), nd), round(float(y), nd))
    
                adj = {}
                edge_ends = []
                edge_len = np.zeros(len(geoms), dtype="float64")
    
                for i, geom in enumerate(geoms):
                    if geom is None or geom.is_empty:
                        edge_ends.append((None, None))
                        edge_len[i] = 0.0
                        continue
                    try:
                        c0 = geom.coords[0]
                        c1 = geom.coords[-1]
                    except Exception:
                        try:
                            lg = max(list(geom.geoms), key=lambda g: g.length)
                            c0 = lg.coords[0]
                            c1 = lg.coords[-1]
                        except Exception:
                            edge_ends.append((None, None))
                            edge_len[i] = 0.0
                            continue
                    u = _key_xy(c0[0], c0[1])
                    v = _key_xy(c1[0], c1[1])
                    edge_ends.append((u, v))
                    L = float(getattr(geom, "length", 0.0) or 0.0)
                    edge_len[i] = L
                    adj.setdefault(u, []).append((v, i, L))
                    adj.setdefault(v, []).append((u, i, L))
    
                # Pick a seed edge: highest finite order, tie by length
                seed_idx = None
                if orders is not None:
                    finite = np.isfinite(orders)
                    if np.any(finite):
                        maxo = float(np.nanmax(orders[finite]))
                        cand = np.where(finite & (orders >= maxo - 1e-6))[0]
                        if cand.size > 0:
                            seed_idx = int(cand[np.argmax(edge_len[cand])])
                if seed_idx is None:
                    # Fallback: longest edge
                    seed_idx = int(np.argmax(edge_len))
    
                u0, v0 = edge_ends[seed_idx]
                if (u0 is not None) and (v0 is not None):
                    # Determine the connected component containing the seed
                    allowed_nodes = set()
                    stack = [u0]
                    allowed_nodes.add(u0)
                    while stack:
                        n0 = stack.pop()
                        for (nb, ei, L) in adj.get(n0, []):
                            if nb not in allowed_nodes:
                                allowed_nodes.add(nb)
                                stack.append(nb)
    
                    # Trace mainstem path outward from the seed, choosing best continuation.
                    mainstem_edge_mask = np.zeros(len(geoms), dtype=bool)
                    mainstem_edge_mask[seed_idx] = True
    
                    def _edge_score(ei: int):
                        # Prefer higher order, then length
                        o = float(orders[ei]) if (orders is not None and np.isfinite(orders[ei])) else -1.0
                        return (o, float(edge_len[ei]))
    
                    def _extend_from(node, prev_node):
                        cur = node
                        prev = prev_node
                        while True:
                            # Candidate incident edges that stay inside component and not already used
                            cands = []
                            for (nb, ei, L) in adj.get(cur, []):
                                if nb not in allowed_nodes:
                                    continue
                                if mainstem_edge_mask[ei]:
                                    continue
                                # avoid immediate backtrack if possible
                                if prev is not None and nb == prev:
                                    continue
                                cands.append((nb, ei))
                            if not cands:
                                # allow backtrack edge if it's the only option
                                for (nb, ei, L) in adj.get(cur, []):
                                    if nb not in allowed_nodes:
                                        continue
                                    if mainstem_edge_mask[ei]:
                                        continue
                                    cands.append((nb, ei))
                            if not cands:
                                break
                            # Choose best by score (order, length)
                            best_nb, best_ei = max(cands, key=lambda t: _edge_score(t[1]))
                            mainstem_edge_mask[best_ei] = True
                            prev, cur = cur, best_nb
    
                    # Extend from both ends of the seed edge
                    _extend_from(u0, v0)
                    _extend_from(v0, u0)
    
                    ms_geoms = [
                        geom for geom, keep in zip(geoms, mainstem_edge_mask)
                        if keep and geom is not None and (not geom.is_empty)
                    ]
                    if ms_geoms:
                        ms_line = _rasterize(
                            [(geom, 1) for geom in ms_geoms],
                            out_shape=channel.shape,
                            transform=transform,
                            fill=0,
                            dtype="uint8",
                            all_touched=True,
                        )
                        ms_sum = int(ms_line.sum())
            if ms_sum == 0:
                try:
                    ms_line = _rasterize(
                        [(geom.buffer(float(pix)*2.0), 1) for geom in ms_geoms],
                        out_shape=channel.shape,
                        transform=transform,
                        fill=0,
                        dtype="uint8",
                        all_touched=True,
                    )
                    ms_sum = int(ms_line.sum())
                except Exception:
                    log.debug("ignored", exc_info=True)
            if ms_sum > 0:
                            # Distance to mainstem line (meters)
                            dist_line = _edt(ms_line == 0) * float(pix)
                            # Local half-width proxy from channel mask (meters)
                            chan_mask = (channel > 0) if channel.dtype != bool else channel
                            halfw = _edt(chan_mask) * float(pix)
                            if getattr(args, "mainstem_mode", "width_proxy") == "width_proxy":
                                # Width-proxy mainstem selection (robust alternative to stream order / graph tracing).
                                                                        # Compute an approximate channel width proxy (meters) from the distance-to-bank field.
                                                                        w_proxy_m = (2.0 * halfw).astype("float32")
                                                                        
                                                                        # Define "mainstem candidates" as the wider portion of the channel.
                                                                        # Threshold is max(absolute_min_width_m, percentile within channel).
                                                                        abs_min_width_m = float(getattr(args, "mainstem_width_min_m", 60.0))
                                                                        pctl = float(getattr(args, "mainstem_width_pctl", 85.0))
                                                                        try:
                                                                            vals = w_proxy_m[channel]
                                                                            vals = vals[np.isfinite(vals)]
                                                                            thr = float(np.nanpercentile(vals, pctl)) if vals.size else abs_min_width_m
                                                                            thr = max(thr, abs_min_width_m)
                                                                        except Exception:
                                                                            thr = abs_min_width_m
                                                                        
                                                                        mainstem_wide = channel & (w_proxy_m >= thr)
                                                                        
                                                                        # Keep only the largest connected component of the wide mask to avoid isolated blobs.
                                                                        try:
                                                                            from scipy.ndimage import label as _label
                                                                            lab, nlab = _label(mainstem_wide.astype("uint8"))
                                                                            if nlab > 1:
                                                                                # choose largest component by pixel count
                                                                                counts = np.bincount(lab.ravel())
                                                                                counts[0] = 0
                                                                                keep = int(np.argmax(counts))
                                                                                mainstem_wide = (lab == keep)
                                                                        except Exception:
                                                                            log.debug("ignored", exc_info=True)
                                                                        
                                                                        # In junction zones, expand protection slightly so tributary smoothing cannot imprint into the mainstem.
                                                                        mainstem_corridor = mainstem_wide | (jm & channel & (w_proxy_m >= 0.8 * thr))
                                                                        preserve_mainstem = mainstem_corridor
                                                                        
                                                                        # Overwrite ms_line-based corridor logic below by setting dist_line very small within mainstem_corridor.
                                                                        # (This prevents later code paths that rely on dist_line from restricting the corridor.)
                                                                        dist_line = np.full(channel.shape, 1e9, dtype="float32")
                                                                        dist_line[mainstem_corridor] = 0.0
                            
    
    
                            # Minimum corridor width to ensure continuity through confluences
                            min_corr = max(75.0, 7.0 * float(pix))
    
                            # Base corridor inside channel; widen slightly in junction zone
                            corr_base = channel & (dist_line <= np.maximum(min_corr, 1.00 * halfw))
                            corr_junc = channel & jm & (dist_line <= np.maximum(min_corr, 1.25 * halfw))
                            corridor = corr_base | corr_junc
    
                            LOG.info(
                                "Mainstem corridor: channel_n=%d jm_n=%d corridor_n=%d",
                                int(np.count_nonzero(channel)),
                                int(np.count_nonzero(jm)),
                                int(np.count_nonzero(corridor)),
                            )
                            try:
                                if debug_corr_path is not None:
                                    _save_f32(debug_corr_path, corridor.astype("float32"), template_profile, nodata=255.0)
                            except Exception:
                                log.debug("ignored", exc_info=True)
                            mainstem_corridor = corridor
                            if int(np.count_nonzero(preserve_mainstem)) == 0:
                                LOG.warning("Mainstem preserve mask is empty; mainstem selection/rasterization likely failed.")
    
                            # Debug rasters
                            try:
                                if debug_jm_path is not None:
                                    _save_f32(debug_jm_path, jm.astype("uint8"), template_profile, nodata=255.0)
                                if (debug_ms_path is not None) and (preserve_mainstem is not None):
                                    _save_f32(debug_ms_path, preserve_mainstem.astype("uint8"), template_profile, nodata=255.0)
                            except Exception:
                                log.debug("ignored", exc_info=True)
    except Exception:
        LOG.warning("Mainstem corridor detection failed", exc_info=True)
        return None, None
    return mainstem_corridor, preserve_mainstem


def main(
) -> int:
    p = argparse.ArgumentParser(description="Generate river bed elevations using a channel-skeleton distance-transform method.")
    p.add_argument("--river-gpkg", required=True, help="river_network.gpkg from river_network.py")
    p.add_argument("--rivers-layer", default="rivers_clip", help="Layer name inside gpkg (default rivers_clip)")
    p.add_argument("--template-raster", required=True, help="Template raster defining output grid (usually river_dem.tif)")
    p.add_argument("--dem", required=True, help="DEM raster (same vertical datum as desired bed elevations)")
    p.add_argument("--channel-mask", required=True, help="River channel mask raster (1=river channel pixels)")
    p.add_argument("--authoritative-bed-raster", default=None,
                   help="Optional authoritative bed elevation raster to blend/enforce inside the river mask (same grid/vertical datum as DEM).")
    p.add_argument("--authoritative-bed-max-dist-m", type=float, default=2000.0,
                   help="Max distance (m) from authoritative bed pixels to influence residual blending.")
    p.add_argument("--residual-blend-sigma-m", type=float, default=600.0,
                   help="Gaussian sigma (m) for residual blending smoothing. 0 disables smoothing (nearest-only).")
    p.add_argument("--out-bed", required=True, help="Output bed elevation raster (GeoTIFF)")
    p.add_argument("--shape-exp", type=float, default=0.5, help="Depth profile exponent (0.5 ~ U-shaped; 1.0 ~ V-shaped)")
    p.add_argument("--dmax-min-m", type=float, default=0.5, help="Clamp minimum Dmax prior (m)")
    p.add_argument("--dmax-max-m", type=float, default=30.0, help="Clamp maximum Dmax prior (m)")
    # Optional: longitudinal bed profile constraints (applied after bed inference)
    p.add_argument("--bed-profile-max-slope", type=float, default=0.0,
                   help="Max absolute bed slope |dz/ds| along flowlines (m/m). 0 disables.")
    p.add_argument("--bed-profile-max-curv", type=float, default=0.0,
                   help="Max absolute bed curvature |d2z/ds2| along flowlines (1/m). 0 disables.")
    p.add_argument("--bed-profile-step-m", type=float, default=25.0,
                   help="Sampling step (m) along flowlines for building the constrained bed profile (default: 25).")
    p.add_argument("--bed-profile-strength", type=float, default=0.6,
                   help="Blend strength (0..1) for applying constrained bed profile back into the raster (default: 0.6).")
    p.add_argument("--bed-profile-power", type=float, default=2.0,
                   help="Distance-decay power for spreading constrained bed correction from the skeleton (default: 2.0).")

    # Priors (match bathy_main / xs_infer args)
    p.add_argument("--prior-mode", default="powerlaw", choices=["powerlaw", "multivariate"], help="Dmax prior mode")
    p.add_argument("--mv-a0", type=float, default=0.18)
    p.add_argument("--mv-bw", type=float, default=0.50)
    p.add_argument("--mv-ba", type=float, default=0.0)
    p.add_argument("--mv-bs", type=float, default=-0.10)
    p.add_argument("--mv-eps-a", type=float, default=1.0)
    p.add_argument("--mv-eps-s", type=float, default=1e-4)

    # WSE (water surface elevation) proxy
    p.add_argument(
        "--wse-mode",
        default="bank",
        choices=["bank", "skeleton", "bank_profile"],
        help=(
            "How to approximate WSE from the DEM. 'bank' (default) uses bank-adjacent DEM samples and fills inward; 'bank_profile' builds a longitudinally-consistent WSE profile from bank samples along flowlines and rasterizes it across the channel; 'skeleton' uses DEM samples on centerlines and fills outward (legacy; can be unstable if DEM is bad in-channel)."
        ),
    )
    p.add_argument(
        "--wse-smooth-sigma-m",
        type=float,
        default=0.0,
        help="Optional Gaussian smoothing sigma (m) applied to the WSE proxy field inside the channel.",
    )

    # Longitudinal WSE profile controls (wse-mode=bank_profile)
    p.add_argument("--wse-profile-river-layer", default="rivers_clip",
               help="Layer name in --river-gpkg containing flowlines to sample along for longitudinal WSE profiling.")
    p.add_argument("--wse-profile-step-m", type=float, default=20.0,
               help="Densification step (m) for sampling along flowlines when building a longitudinal WSE profile.")
    p.add_argument("--wse-profile-resample-m", type=float, default=20.0,
               help="Resample step (m) for 1D profile smoothing along flow distance.")
    p.add_argument("--wse-profile-smooth-sigma-m", type=float, default=200.0,
               help="Gaussian smoothing sigma (m) applied to the 1D WSE profile along flow distance.")
    p.add_argument("--wse-profile-max-slope", type=float, default=0.005,
               help="Optional maximum absolute slope (m/m) enforced along the 1D WSE profile (0 to disable).")
    p.add_argument("--wse-profile-min-samples", type=int, default=10,
               help="Minimum number of valid samples along a flowline to build a profile; otherwise falls back to wse-mode=bank.")
    p.add_argument(
        "--wse-profile-max-query-dist-m",
        type=float,
        default=250.0,
        help=(
            "Maximum XY distance (m) for assigning longitudinal-profile samples to centerline pixels. "
            "Helps avoid cross-reach snapping near confluences/parallel channels (0 to disable)."
        ),
    )

    p.add_argument(
        "--wse-profile-fit",
        default="isotonic",
        choices=["isotonic", "none"],
        help=(
            "Longitudinal constraint applied to the 1D WSE profile along each flowline segment. "
            "'isotonic' enforces a non-increasing WSE trend in the inferred downstream direction (higher→lower); "
            "'none' disables the monotonic constraint."
        ),
    )

    
    # Optional: SWOT RiverSP anchoring for wse-mode=bank_profile
    p.add_argument(
        "--swot-riversp",
        nargs="+",
        default=None,
        help=(
            "One or more SWOT RiverSP vector files (reach or node product; e.g., .shp/.gpkg/.geojson) "
            "containing water surface elevation (WSE). Used only when --wse-mode=bank_profile."
        ),
    )
    p.add_argument(
        "--swot-wse-field",
        default=None,
        help="Column name for SWOT WSE in the RiverSP file(s). If omitted, common candidates will be searched.",
    )
    p.add_argument(
        "--swot-qual-field",
        default=None,
        help="Optional column name for a SWOT quality flag; if provided, values >0 are treated as bad and filtered out.",
    )
    p.add_argument(
        "--swot-max-dist-m",
        type=float,
        default=300.0,
        help="Maximum distance (m) from a flowline sample point to accept a SWOT WSE observation.",
    )
    p.add_argument(
        "--swot-min-samples",
        type=int,
        default=5,
        help="Minimum number of SWOT samples on a flowline required to apply anchoring corrections.",
    )
    p.add_argument(
        "--swot-correct-sigma-m",
        type=float,
        default=2000.0,
        help="Smoothing scale (m) for along-channel SWOT correction (Gaussian sigma along distance).",
    )
    p.add_argument(
        "--swot-weight",
        type=float,
        default=1.0,
        help="Weight (0..1) to apply SWOT correction to the bank-derived WSE profile.",
    )
    p.add_argument(
        "--swot-max-correction-m",
        type=float,
        default=5.0,
        help=(
            "Clamp the along-channel correction magnitude (m) applied from SWOT (helps avoid datum/offset mistakes)."
        ),
    )
    p.add_argument(
        "--swot-wse-offset-m",
        type=float,
        default=0.0,
        help=(
            "Constant offset (m) added to SWOT WSE before use. Use this to reconcile vertical datums "
            "until a full datum transform is implemented."
        ),
    )

    p.add_argument(
        "--swot-offset-mode",
        default="median_mad",
        choices=["none", "median_mad"],
        help=(
            "Vertical datum reconciliation mode for SWOT WSE vs bank_profile WSE. "
            "'median_mad' estimates a robust constant offset from overlapping samples and subtracts it from SWOT WSE. "
            "'none' disables auto offset estimation (use --swot-wse-offset-m instead)."
        ),
    )
    p.add_argument(
        "--swot-offset-min-samples",
        type=int,
        default=25,
        help="Minimum number of SWOT samples required to estimate an auto vertical offset.",
    )
    p.add_argument(
        "--swot-offset-mad-z",
        type=float,
        default=3.5,
        help="MAD-based outlier rejection threshold (in robust-sigma units) for auto vertical offset estimation.",
    )
    p.add_argument(
        "--swot-offset-max-abs-m",
        type=float,
        default=10.0,
        help="Clamp absolute value of the auto-estimated vertical offset (m). If exceeded, it is clamped and a warning is logged.",
    )


    # Junction / confluence handling (skeleton mode can be unstable near confluences)

    p.add_argument(
        "--junction-mode",
        default="smooth",
        choices=["smooth", "mask", "none"],
        help=(
            "How to handle confluence/junction zones (graph_nodes degree>=N). "
            "'smooth' (default) locally smooths WSE/depth in a buffered junction zone to suppress artifacts; "
            "'mask' sets junction-zone outputs to nodata (conservative); 'none' disables this behavior."
        ),
    )
    p.add_argument("--junction-buffer-m", type=float, default=120.0, help="Buffer radius (m) around junction nodes.")
    p.add_argument("--junction-degree-min", type=int, default=3, help="Minimum graph node degree to be treated as a junction.")
    p.add_argument("--junction-nodes-layer", default="graph_nodes", help="Layer in river-gpkg containing graph nodes with 'degree'.")
    p.add_argument(
        "--junction-smooth-sigma-m",
        type=float,
        default=80.0,
        help="Gaussian smoothing sigma (m) used in junction zones when junction-mode=smooth.",
    )

    # Mainstem preserve mask selection
    p.add_argument(
        "--mainstem-mode",
        default="width_proxy",
        choices=["width_proxy", "graph_trace"],
        help=(
            "How to define the mainstem for confluence protection. "
            "'width_proxy' (default) uses a width proxy derived from the channel mask to identify the widest connected channel. "
            "'graph_trace' uses stream-order/graph-based tracing."
        ),
    )
    p.add_argument(
        "--mainstem-width-min-m",
        type=float,
        default=60.0,
        help="Absolute minimum width proxy (m) for pixels to be considered mainstem (width_proxy mode).",
    )
    p.add_argument(
        "--mainstem-width-pctl",
        type=float,
        default=85.0,
        help="Percentile (0-100) of width proxy within channel used as threshold for mainstem (width_proxy mode).",
    )



    p.add_argument(
        "--junction-max-width-m",
        type=float,
        default=300.0,
        help="If >0, limit junction smoothing/masking to pixels with estimated channel width <= this (m). Helps avoid over-smoothing wide confluences.",
    )

    # Curvature-driven asymmetry (outer-bank deeper in bends)
    p.add_argument(
        "--asymmetry-mode",
        default="none",
        choices=["none", "curvature"],
        help=(
            "Optional asymmetry model applied to the cross-channel coordinate r. "
            "'curvature' biases depth toward the outer bank using signed centerline curvature; "
            "'none' disables asymmetry (default)."
        ),
    )
    p.add_argument("--asymmetry-strength", type=float, default=0.25,
                   help="Strength of curvature-driven r-shift (dimensionless; typical 0.1–0.4).")
    p.add_argument("--asymmetry-curv-ref", type=float, default=0.002,
                   help="Reference curvature (1/m) for scaling (e.g., 0.002 ~ 500 m radius).")
    p.add_argument("--asymmetry-max-shift", type=float, default=0.20,
                   help="Max absolute shift applied to r (clamped), dimensionless.")
    p.add_argument("--asymmetry-min-width-m", type=float, default=10.0,
                   help="Only apply asymmetry where estimated channel width >= this (m).")
    p.add_argument("--asymmetry-min-curv", type=float, default=0.0005,
                   help="Only apply asymmetry where |curvature| >= this (1/m).")
    p.add_argument("--asymmetry-densify-step-m", type=float, default=20.0,
                   help="Vertex spacing (m) used when estimating curvature/tangent from flowlines.")

    # Optional constraints from external soundings (extra XYZ)
    p.add_argument("--soundings", nargs="*", default=None,
                   help="Optional XYZ/CSV/GPKG soundings to inform the skeleton depth prior (depths expected; mode controls sign).")
    p.add_argument("--soundings-crs", default=None,
                   help="CRS of soundings (e.g., EPSG:4326). If omitted, assumes soundings already match the template CRS.")
    p.add_argument("--soundings-mode", choices=["auto","depth_pos","depth_neg","bed_elev"], default="auto",
                   help="Interpret soundings Z as depth (auto/depth_pos/depth_neg) or as bed elevation (bed_elev, same vertical datum as DEM).")
    p.add_argument("--soundings-cell-percentile", type=float, default=None,
                   help="Per-cell percentile used when binning multiple XYZ points into the same grid cell. If not set: bed_elev uses 5th percentile (deeper, outlier-robust); depth uses 95th percentile (deeper, outlier-robust).")
    p.add_argument("--soundings-max-dist-m", type=float, default=10000.0,
                   help="Max distance (m) from a sounding to influence skeleton Dmax (nearest-sounding within this distance).")
    p.add_argument("--soundings-max-points", type=int, default=2_000_000,
                   help="Global cap on number of sounding points used (downsampled with per-file proportions) to prevent OOM-kills.")
    p.add_argument("--soundings-sample-seed", type=int, default=0,
                   help="RNG seed for soundings downsampling (for reproducibility).")
    p.add_argument("--soundings-min-r", type=float, default=0.25,
                   help="Minimum r used when converting sounding depth -> implied Dmax (prevents blow-ups near banks).")
    p.add_argument("--no-soundings-enforce", dest="soundings_enforce", action="store_false", default=True,
                   help="Disable enforcing observed sounding depths at their grid cells (default enforces).")
    p.add_argument("--debug-dir", default=None, help="If set, write debug rasters to this directory.")
    args = p.parse_args()

    template_profile, transform, crs, shape = _read_template(Path(args.template_raster))

    # Scientific correctness: river skeleton inference assumes *metric* units (meters).
    # If the template CRS is geographic (degrees), distances/buffers/sigmas become meaningless.
    try:
        _crs_obj = CRS.from_user_input(crs) if crs is not None else None
        if _crs_obj is None or (hasattr(_crs_obj, 'is_projected') and not _crs_obj.is_projected):
            raise ValueError(f"Template CRS must be projected (meters). Got: {crs}")
    except Exception as e:
        raise ValueError(f"Template CRS must be projected (meters). Got: {crs}") from e

    template_profile["transform"] = transform
    template_profile["crs"] = crs
    pix = _pixel_size_m(transform)
    LOG.info("Template grid: %s x %s | pixel_size≈%.3fm", shape[1], shape[0], pix)

    # Load channel mask and DEM on template grid
    ch_raw, _ = _warp_to_template(Path(args.channel_mask), template_profile, dtype="uint8")
    channel = (ch_raw.astype("uint8") == 1)

    dem, dem_nodata = _warp_to_template(Path(args.dem), template_profile, dtype="float32")
    dem = dem.astype("float32")
    valid_dem = np.isfinite(dem)
    if dem_nodata is not None:
        valid_dem &= (dem != float(dem_nodata))

    authoritative_bed = None
    authoritative_bed_nodata = None
    if args.authoritative_bed_raster:
        try:
            authoritative_bed, authoritative_bed_nodata = _warp_to_template(Path(args.authoritative_bed_raster), template_profile, dtype="float32")
            authoritative_bed = authoritative_bed.astype("float32")
            if authoritative_bed_nodata is not None:
                authoritative_bed = np.where(authoritative_bed == float(authoritative_bed_nodata), np.nan, authoritative_bed).astype("float32")
            LOG.info("Loaded authoritative bed raster: %s", args.authoritative_bed_raster)
        except Exception as e:
            LOG.warning("Failed loading authoritative bed raster (%s): %s", args.authoritative_bed_raster, e)

    # Load rivers
    rivers = gpd.read_file(args.river_gpkg, layer=args.rivers_layer)
    if rivers.empty:
        raise SystemExit(f"No features found in {args.river_gpkg}:{args.rivers_layer}")
    rivers = rivers.to_crs(crs)

    # Rasterize skeleton
    skeleton = rasterize(
        [(geom, 1) for geom in rivers.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)

    if not skeleton.any():
        raise SystemExit("No centerline pixels after rasterization (check CRS/grid alignment).")

    # Distance to channel boundary (bank) and to skeleton (centerline)
    d_bank = distance_transform_edt(channel, sampling=pix).astype("float32")
    inv = np.ones(shape, dtype=np.uint8)
    inv[skeleton] = 0
    d_center = distance_transform_edt(inv, sampling=pix).astype("float32")

    # Adjust bank distance so boundary pixels map to r≈0
    d_bank_eff = np.clip(d_bank - pix, 0.0, None).astype("float32")
    denom = (d_bank_eff + d_center + 1e-6).astype("float32")
    r = np.clip(d_bank_eff / denom, 0.0, 1.0).astype("float32")

    # Width proxy (m)
    width = (2.0 * np.maximum(d_bank, 0.0)).astype("float32")

    # Estimate Dmax on skeleton pixels:
    # 1) default powerlaw from width at skeleton
    dmax_skel = _dmax_from_powerlaw(width, float(args.mv_a0), float(args.mv_bw),
                                    float(args.dmax_min_m), float(args.dmax_max_m))
    dmax_skel = np.where(skeleton, dmax_skel, np.nan).astype("float32")

    # 2) optional multivariate factor if attributes exist
    if args.prior_mode == "multivariate":
        area_col = _get_attr_col(rivers, ["TotDASqKM", "TotDASQKM", "totdasqkm", "DASqKM", "DrainArea", "TotDrain"])
        slope_col = _get_attr_col(rivers, ["Slope", "SLOPE", "slope", "SlopePerc", "slp"])
        if area_col or slope_col:
            # Build per-feature multiplier, rasterize onto skeleton, then blend with width-based Dmax.
            # Convert drainage area to m^2 if in km^2.
            area = None
            if area_col:
                a_km2 = rivers[area_col].astype(float).to_numpy()
                area = (np.maximum(a_km2, 0.0) * 1e6).astype("float32")
            slope = None
            if slope_col:
                slope = np.maximum(rivers[slope_col].astype(float).to_numpy(), 0.0).astype("float32")

            # Per-feature depth prior (evaluated at "unit" width, then scaled by width^bw already in dmax_skel).
            # Dmax = a0 * width^bw * (area+eps_a)^ba * (slope+eps_s)^bs
            mult = np.ones(len(rivers), dtype="float32")
            if area is not None:
                mult *= np.power(area + float(args.mv_eps_a), float(args.mv_ba)).astype("float32")
            if slope is not None:
                mult *= np.power(slope + float(args.mv_eps_s), float(args.mv_bs)).astype("float32")

            # Rasterize multiplier onto skeleton pixels (default 1.0)
            mult_r = rasterize(
                [(geom, float(mv)) for geom, mv in zip(rivers.geometry, mult)],
                out_shape=shape,
                transform=transform,
                fill=1.0,
                all_touched=True,
                dtype="float32",
            ).astype("float32")

            dmax_skel = np.where(skeleton, np.clip(dmax_skel * mult_r, float(args.dmax_min_m), float(args.dmax_max_m)), np.nan).astype("float32")
            LOG.info("Applied multivariate Dmax factor using attrs: area=%s slope=%s", area_col, slope_col)
        else:
            LOG.info("prior-mode=multivariate but no suitable area/slope attrs found; using powerlaw only.")
    # Nearest-skeleton propagation (Voronoi via EDT indices)
    inv2 = np.ones(shape, dtype=np.uint8)
    inv2[skeleton] = 0
    _, inds = distance_transform_edt(inv2, return_indices=True)
    ny, nx = inds[0], inds[1]

    # Optional: curvature-driven asymmetry (outer-bank deeper in bends)
    if (str(args.asymmetry_mode).strip().lower() == "curvature"):
        try:
            # Rasterize tangent and signed curvature onto centerline pixels
            tx_r, ty_r, k_r = _rasterize_tangent_and_curvature(
                rivers_gdf=rivers,
                transform=transform,
                out_shape=shape,
                densify_step_m=float(args.asymmetry_densify_step_m or 20.0),
            )

            tx_nn = tx_r[ny, nx].astype("float32")
            ty_nn = ty_r[ny, nx].astype("float32")
            k_nn = k_r[ny, nx].astype("float32")

            # Normal vector (left-hand) from tangent: n = [-ty, tx]
            nx_n = (-ty_nn).astype("float32")
            ny_n = (tx_nn).astype("float32")

            iy, ix = np.indices(shape, dtype="int32")
            dx = (ix - nx).astype("float32")
            dy = (iy - ny).astype("float32")

            # Side of the centerline (+/-) relative to local normal
            side = np.sign((dx * nx_n) + (dy * ny_n)).astype("float32")
            side = np.where(np.isfinite(tx_nn) & np.isfinite(ty_nn), side, 0.0).astype("float32")

            curv_ref = max(float(args.asymmetry_curv_ref or 0.0), 1e-9)
            curv_scaled = np.clip(k_nn / curv_ref, -1.0, 1.0).astype("float32")

            # Only apply where channel is wide enough and curvature is meaningful
            apply = channel & (width >= float(args.asymmetry_min_width_m or 0.0)) & (np.abs(k_nn) >= float(args.asymmetry_min_curv or 0.0))
            # Shape: strongest mid-channel, zero at banks/centerline
            shape_w = (4.0 * r * (1.0 - r)).astype("float32")

            shift = (float(args.asymmetry_strength) * curv_scaled * side * shape_w).astype("float32")
            max_shift = float(args.asymmetry_max_shift or 0.0)
            if max_shift > 0:
                shift = np.clip(shift, -max_shift, max_shift).astype("float32")

            r_asym = np.clip(r + np.where(apply, shift, 0.0), 0.0, 1.0).astype("float32")
            r = r_asym  # override for downstream (soundings conversion, depth computation)

            if args.debug_dir:
                dbg = Path(args.debug_dir)
                dbg.mkdir(parents=True, exist_ok=True)
                _save_f32(dbg / "debug_curvature_k.tif", k_r.astype("float32"), template_profile, nodata=-9999.0)
                _save_f32(dbg / "debug_r_asym.tif", r.astype("float32"), template_profile, nodata=-9999.0)
            LOG.info(
                "Applied curvature-driven asymmetry: strength=%.3f curv_ref=%.5f max_shift=%.3f min_width=%.1f min_curv=%.5f",
                float(args.asymmetry_strength), curv_ref, float(args.asymmetry_max_shift),
                float(args.asymmetry_min_width_m), float(args.asymmetry_min_curv),
            )
        except Exception as e:
            LOG.warning("Failed applying curvature-driven asymmetry; continuing with symmetric r. Error: %s", e)

    # Water surface elevation (WSE) proxy:
    # For correctness, default to a bank-derived WSE (banks are usually better represented than in-channel DEM).
    wse_map = None
    mode = (args.wse_mode or "").strip().lower()
    if mode == "bank_profile":
        # Build a bank-derived WSE field first, then enforce longitudinal consistency along flowlines.
        wse_bank = _build_wse_from_bank_dem(
            dem=dem,
            channel=channel,
            valid_dem=valid_dem,
            pix_m=pix,
            smooth_sigma_m=float(args.wse_smooth_sigma_m or 0.0),
        )
        if wse_bank is not None:
            swot_pts = None
            swot_wse = None
            if getattr(args, "swot_riversp", None):
                swot_pts, swot_wse = _read_swot_riversp_points(
                    paths=getattr(args, "swot_riversp", None),
                    template_crs=template_profile.get("crs"),
                    wse_field=getattr(args, "swot_wse_field", None),
                    qual_field=getattr(args, "swot_qual_field", None),
                    wse_offset_m=float(getattr(args, "swot_wse_offset_m", 0.0) or 0.0),
                )
                if swot_pts is not None and len(swot_pts) > 0:
                    LOG.info(
                        "Loaded SWOT RiverSP WSE samples: n=%d (user_offset=%.3f m).",
                        len(swot_pts),
                        float(getattr(args, "swot_wse_offset_m", 0.0) or 0.0),
                    )

                    # Auto vertical datum reconciliation: estimate a robust constant offset between SWOT WSE and bank_profile WSE
                    # using overlapping samples, then subtract it from SWOT WSE.
                    off_mode = str(getattr(args, "swot_offset_mode", "median_mad") or "median_mad").lower()
                    if off_mode != "none":
                        est_off, n_used, est_stats = _estimate_swot_vertical_offset(
                            swot_pts=swot_pts,
                            swot_wse=swot_wse,
                            bank_wse=wse_bank,
                            template_transform=template_profile.get("transform"),
                            channel=channel,
                            mode=off_mode,
                            min_samples=int(getattr(args, "swot_offset_min_samples", 25) or 0),
                            mad_z=float(getattr(args, "swot_offset_mad_z", 3.5) or 3.5),
                            max_abs_m=float(getattr(args, "swot_offset_max_abs_m", 10.0) or 0.0),
                        )
                        if n_used > 0 and abs(float(est_off)) > 1e-9:
                            swot_wse = (swot_wse - float(est_off)).astype(float)
                            LOG.info(
                                "SWOT vertical offset estimated: %.3f m (n=%d). Applied as swot_wse -= offset. Stats=%s",
                                float(est_off), int(n_used), str(est_stats),
                            )
                        elif n_used > 0:
                            LOG.info("SWOT vertical offset estimated ~0.0 m (n=%d).", int(n_used))
                        else:
                            LOG.warning(
                                "SWOT vertical offset could not be estimated (mode=%s). "
                                "Ensure vertical datums are compatible or provide --swot-wse-offset-m.",
                                off_mode,
                            )
                    else:
                        # Safety warning: datums are not automatically reconciled.
                        if abs(float(getattr(args, "swot_wse_offset_m", 0.0) or 0.0)) < 1e-6:
                            LOG.warning(
                                "SWOT RiverSP anchoring enabled but auto offset is disabled and swot_wse_offset_m is 0.0. "
                                "Ensure SWOT WSE is in the same vertical datum as your DEM/WSE proxy (e.g., NAVD88), "
                                "or set an offset."
                            )

            wse_prof = _build_wse_longitudinal_profile(
                bank_wse=wse_bank,
                channel=channel,
                template_transform=template_profile.get("transform"),
                template_crs=template_profile.get("crs"),
                river_gpkg=Path(args.river_gpkg),
                river_layer=str(getattr(args, "wse_profile_river_layer", "rivers_clip")),
                step_m=float(getattr(args, "wse_profile_step_m", 20.0) or 20.0),
                resample_m=float(getattr(args, "wse_profile_resample_m", 20.0) or 20.0),
                smooth_sigma_m=float(getattr(args, "wse_profile_smooth_sigma_m", 200.0) or 0.0),
                max_slope=float(getattr(args, "wse_profile_max_slope", 0.0) or 0.0),
                min_samples=int(getattr(args, "wse_profile_min_samples", 10) or 10),
                max_query_dist_m=float(getattr(args, "wse_profile_max_query_dist_m", 250.0) or 0.0),
                swot_pts=swot_pts,
                swot_wse=swot_wse,
                swot_max_dist_m=float(getattr(args, "swot_max_dist_m", 300.0) or 0.0),
                swot_min_samples=int(getattr(args, "swot_min_samples", 5) or 0),
                swot_correct_sigma_m=float(getattr(args, "swot_correct_sigma_m", 2000.0) or 0.0),
                swot_weight=float(getattr(args, "swot_weight", 1.0) or 0.0),
                swot_max_correction_m=float(getattr(args, "swot_max_correction_m", 5.0) or 0.0),
                fit_mode=str(getattr(args, "wse_profile_fit", "isotonic") or "isotonic"),
            )
            if wse_prof is not None:
                wse_map = wse_prof
            else:
                LOG.warning("wse-mode=bank_profile could not build a longitudinal WSE profile; falling back to wse-mode=bank")
        else:
            LOG.warning("wse-mode=bank_profile could not build bank WSE; falling back to wse-mode=bank")

    if wse_map is None and mode == "bank":
        wse_map = _build_wse_from_bank_dem(
            dem=dem,
            channel=channel,
            valid_dem=valid_dem,
            pix_m=pix,
            smooth_sigma_m=float(args.wse_smooth_sigma_m or 0.0),
        )
        if wse_map is None:
            LOG.warning("wse-mode=bank could not build WSE from bank samples; falling back to skeleton-derived WSE.")

    if wse_map is None:
        # Legacy behavior: use DEM values along the skeleton and fill outward.
        wse_skel = np.full(shape, np.nan, dtype="float32")
        sk_valid = skeleton & valid_dem
        if sk_valid.any():
            wse_med = float(np.nanmedian(dem[sk_valid]))
            wse_skel[sk_valid] = dem[sk_valid]
            wse_skel[skeleton & (~valid_dem)] = wse_med
        else:
            # fallback: median of DEM where available in channel, else 0.0
            if (channel & valid_dem).any():
                wse_med = float(np.nanmedian(dem[channel & valid_dem]))
            else:
                wse_med = 0.0
            wse_skel[skeleton] = wse_med
            LOG.warning("No valid DEM samples on skeleton; using fallback WSE=%.3f", wse_med)

        wse_map = wse_skel[ny, nx].astype("float32")

    # Optional: Incorporate external soundings (extra XYZ) to refine the Dmax prior and/or enforce bed/depth anchors.
    snd_depth_grid = None
    snd_dmax_grid = None
    snd_bed_grid = None
    snd_dist = None
    snd_dmax_field = None
    if args.soundings:
        try:
            snd_depth_grid, snd_dmax_grid, snd_bed_grid = _soundings_to_grids(
                sounding_files=args.soundings,
                soundings_crs=args.soundings_crs,
                template_crs=crs,
                transform=transform,
                channel=channel,
                r=r,
                shape_exp=float(args.shape_exp),
                dmax_min_m=float(args.dmax_min_m),
                dmax_max_m=float(args.dmax_max_m),
                mode=str(args.soundings_mode),
                min_r=float(args.soundings_min_r),
                max_points=int(getattr(args, 'soundings_max_points', 2_000_000) or 0),
                sample_seed=int(getattr(args, 'soundings_sample_seed', 0) or 0),
                cell_percentile=args.soundings_cell_percentile,
                wse_map=wse_map,
                diag_json_path=Path(args.out_bed).with_name("soundings_channel_receipt.json"),
            )
            snd_mask = np.isfinite(snd_dmax_grid)
            snd_dmax_field = None
            if np.any(snd_mask):
                # Build a *continuous* Dmax field inside the channel from sparse sounding-derived anchors.
                # This helps remove "seams" where gaps between authoritative clusters were previously reverting to the width prior.
                snd_dmax_field = _build_value_field_from_anchors(
                    val_src=snd_dmax_grid.astype("float32"),
                    src_valid_mask=snd_mask,
                    channel_mask=channel,
                    max_dist_m=float(args.soundings_max_dist_m),
                    blend_sigma_m=float(args.residual_blend_sigma_m),
                    pixel_size=float(pix),
                )

                use = skeleton & np.isfinite(snd_dmax_field)
                n_use = int(np.count_nonzero(use))
                if n_use > 0:
                    dmax_skel = np.where(use, np.maximum(dmax_skel, snd_dmax_field), dmax_skel).astype("float32")
                    LOG.info(
                        "Soundings: updated Dmax prior on %d skeleton pixels (max_dist=%.1fm, sigma=%.1fm, mode=%s).",
                        n_use, float(args.soundings_max_dist_m), float(args.residual_blend_sigma_m), str(args.soundings_mode)
                    )
                else:
                    LOG.info("Soundings: Dmax anchors did not influence any skeleton pixels; Dmax prior unchanged.")
            else:
                LOG.info("Soundings provided but no valid points fell inside the channel mask.")
        except Exception as e:
            LOG.warning("Failed to incorporate soundings into Dmax prior: %s", e)

    # Propagate Dmax from skeleton to the full channel domain
    dmax_map = dmax_skel[ny, nx].astype("float32")

    # If we built a continuous sounding-informed Dmax field, merge it into the channel Dmax.
    # This prevents reverting to the (often too-shallow) width prior in data-sparse gaps between authoritative clusters.
    if snd_dmax_field is not None:
        n_infl = int(np.count_nonzero(channel & np.isfinite(snd_dmax_field)))
        if n_infl > 0:
            dmax_map = np.where(np.isfinite(snd_dmax_field), np.maximum(dmax_map, snd_dmax_field), dmax_map).astype("float32")
            LOG.info("Soundings: merged continuous Dmax field into channel (influenced_pixels=%d).", n_infl)


    # Depth field (m, positive downward)
    depth = (dmax_map * np.power(r, float(args.shape_exp))).astype("float32")

    # Optional: enforce observed depths at sounding grid cells (depth modes only).
    if (snd_depth_grid is not None) and (str(getattr(args, "soundings_mode", "auto")).strip().lower() != "bed_elev") and bool(getattr(args, "soundings_enforce", True)):
        try:
            m_snd = np.isfinite(snd_depth_grid) & channel
            n_snd = int(np.count_nonzero(m_snd))
            if n_snd > 0:
                depth[m_snd] = snd_depth_grid[m_snd].astype("float32")
                LOG.info("Soundings: enforced observed depths at %d grid cells.", n_snd)
        except Exception as e:
            LOG.warning("Soundings: failed enforcing observed depths: %s", e)

    
    # Junction/confluence handling: optionally smooth or mask outputs near high-degree graph nodes.
    junction_zone = None
    if str(getattr(args, "junction_mode", "smooth")).strip().lower() != "none":
        junction_zone = _junction_zone_mask(
            river_gpkg=Path(args.river_gpkg),
            nodes_layer=str(getattr(args, "junction_nodes_layer", "graph_nodes")),
            river_layer=str(getattr(args, "wse_profile_river_layer", "rivers_clip")),
            template_profile=template_profile,
            transform=transform,
            crs=crs,
            shape=shape,
            degree_min=int(getattr(args, "junction_degree_min", 3)),
            buffer_m=float(getattr(args, "junction_buffer_m", 120.0) or 0.0),
        )
        if junction_zone is not None:
            # Restrict to channel pixels
            jm = junction_zone & channel
            n_jm = int(np.count_nonzero(jm))
            maxw = float(getattr(args, "junction_max_width_m", 300.0) or 0.0)
            if maxw > 0.0:
                jm = jm & (width <= maxw)
                n_jm = int(np.count_nonzero(jm))
            if n_jm > 0:

                # ------------------------------------------------------------
                # Mainstem preserve mask (defends the main channel from
                # confluence smoothing / constraints imprinting tributary artifacts)
                # Strategy (robust, attribute-tolerant):
                #   - Build a per-component mainstem polyline by tracing from the
                #     highest-order seed edge and walking outward, always choosing
                #     the highest-order continuation (ties by length).
                #   - Rasterize that polyline, build a corridor using channel
                #     half-width (distance to bank) and distance to mainstem line.
                #   - Preserve the corridor everywhere (not only inside jm).
                preserve_mainstem = None
                mainstem_corridor = None
                try:
                    from pathlib import Path as _Path
                    debug_jm_path = None
                    debug_ms_path = None
                    out_bed_path = _Path(getattr(args, "out_bed", "") or "")
                    if out_bed_path:
                        debug_jm_path = out_bed_path.parent / "junction_zone_mask.tif"
                        debug_ms_path = out_bed_path.parent / "mainstem_preserve_mask.tif"
                except Exception:
                    debug_jm_path = None
                    debug_ms_path = None

                try:
                    mainstem_corridor, preserve_mainstem = _detect_mainstem_corridor(
                        args=args,
                        channel=channel,
                        jm=junction_zone,
                        template_profile=template_profile,
                        transform=transform,
                        shape=shape,
                        pix=pix,
                        debug_corr_path=debug_corr_path,
                        debug_jm_path=debug_jm_path,
                        debug_ms_path=debug_ms_path,
                    )
                except Exception:
                    preserve_mainstem = None
                    mainstem_corridor = None
                depth_pre_smooth = depth.copy()
                wse_pre_smooth = wse_map.copy()
                jmode = str(getattr(args, "junction_mode", "smooth")).strip().lower()
                if jmode == "mask":
                    depth[jm] = np.nan
                    wse_map[jm] = np.nan
                    LOG.info("Junction mode=mask: set %d channel cells to nodata within junction zones.", n_jm)
                elif jmode == "smooth":
                    sig_m = float(getattr(args, "junction_smooth_sigma_m", 80.0) or 0.0)
                    if sig_m > 0.0:
                        def _smooth_in_zone(arr: np.ndarray, zone: np.ndarray, sigma_px: float) -> np.ndarray:
                            out = arr.astype("float32", copy=True)
                            # Normalized (mask-aware) Gaussian smoothing to avoid leaking values
                            # from outside-channel/nodata areas into the junction zone.
                            valid = (channel & np.isfinite(out))
                            if np.any(valid):
                                fill = float(np.nanmedian(out[valid]))
                            else:
                                fill = 0.0
                            filled = np.where(np.isfinite(out), out, fill).astype("float32")
                            w = valid.astype("float32")

                            num = gaussian_filter(filled * w, sigma=sigma_px, mode="nearest")
                            den = gaussian_filter(w, sigma=sigma_px, mode="nearest")
                            sm = filled.astype("float32").copy()
                            np.divide(num, den, out=sm, where=(den > 1e-6))
                            sm = sm.astype("float32")

                            out[zone] = sm[zone]
                            return out

                        # Width-scaled junction smoothing
                        # Goal: reduce over-smoothing at small tributary confluences while still
                        # damping junction artifacts where geometry supports it.
                        #
                        # We approximate a spatially-varying sigma by applying two passes:
                        # (1) a "large-width" pass (sigma=sig_m)
                        # (2) a "small-width" pass with sigma scaled by width ratio.
                        #
                        # This remains deterministic and avoids introducing new external deps.
                        jm_w = None
                        try:
                            jm_w = width[jm] if width is not None else None
                        except Exception:
                            jm_w = None

                        sigma_px_large = sig_m / max(pix, 1e-9)
                        sigma_px_small = sigma_px_large
                        jm_small = jm
                        jm_large = jm
                        if (jm_w is not None) and np.any(np.isfinite(jm_w)):
                            # Percentile-derived scaling: avoids hard-coded width thresholds and
                            # remains stable across AOIs with different channel sizes.
                            w_all = jm_w[np.isfinite(jm_w)].astype("float32")
                            w_med = float(np.nanmedian(w_all)) if w_all.size else 0.0

                            if w_all.size:
                                p25, p50, p75 = np.nanpercentile(w_all, [25.0, 50.0, 75.0])
                                p25 = float(max(p25, 0.0))
                                p50 = float(max(p50, p25))
                                p75 = float(max(p75, p50))
                            else:
                                p25, p50, p75 = 0.0, w_med, 0.0

                            # Split covers the full junction zone.
                            jm_small = jm & (width <= p50)
                            jm_large = jm & (width > p50)

                            # Scale small-sigma using (narrow/typical) width ratio.
                            if p75 > 0.0:
                                scale = float(p25 / p75)
                                scale = float(np.clip(scale, 0.25, 1.0))
                                sigma_px_small = sigma_px_large * scale

                            LOG.info(
                                "Junction smoothing width-scaled: p25=%.2fm p50=%.2fm p75=%.2fm sigma_small=%.1fm sigma_large=%.1fm",
                                p25,
                                p50,
                                p75,
                                float(sigma_px_small * pix),
                                float(sigma_px_large * pix),
                            )

                        # Smooth WSE only (avoid circular 'bullseye' depth artifacts at tributary mouths)
                        # Apply large-width pass first, then refine small-width pixels with a smaller sigma.
                        if np.any(jm_large):
                            wse_map = _smooth_in_zone(wse_map, jm_large, sigma_px_large)
                        if np.any(jm_small):
                            wse_map = _smooth_in_zone(wse_map, jm_small, sigma_px_small)
                        # Recompute depth from (smoothed) WSE and current bed to keep mainstem continuity
                        try:
                            depth = (wse_map - bed).astype('float32')
                            # Depth cannot be negative
                            depth = np.where(depth >= 0.0, depth, 0.0).astype('float32')
                        except Exception:
                            log.warning("Depth recompute from WSE-bed failed during junction smoothing", exc_info=True)
                        try:
                            if preserve_mainstem is not None and np.any(preserve_mainstem):
                                depth[preserve_mainstem] = depth_pre_smooth[preserve_mainstem]
                                wse_map[preserve_mainstem] = wse_pre_smooth[preserve_mainstem]
                                LOG.info("Preserved mainstem corridor during junction smoothing (cells=%d).", int(np.count_nonzero(preserve_mainstem)))
                        except Exception:
                            log.warning("Mainstem preservation step failed during junction smoothing", exc_info=True)
                        LOG.info("Junction mode=smooth: locally smoothed WSE (depth recomputed from WSE-bed) in %d junction-zone cells (sigma_base=%.1fm).", n_jm, sig_m)
                    else:
                        LOG.info("Junction mode=smooth requested but sigma<=0; no smoothing applied.")
                else:
                    LOG.info("Junction mode=%s: no action.", jmode)

    # Mask outputs to channel domain
    depth = np.where(channel, depth, np.nan).astype("float32")
    wse_map = np.where(channel, wse_map, np.nan).astype("float32")
    bed = (wse_map - depth).astype("float32")
    bed = np.where(channel, bed, np.nan).astype("float32")

    # Optional: anchor/blend using authoritative bed raster
    if authoritative_bed is not None:
        m_auth = np.isfinite(authoritative_bed) & channel
        n_auth = int(np.count_nonzero(m_auth))
        if n_auth > 0:
            resid_auth = (authoritative_bed - bed).astype("float32")
            adj_auth = _build_residual_adjustment(resid_auth, m_auth, channel, float(args.authoritative_bed_max_dist_m), float(args.residual_blend_sigma_m), float(pix))
            bed = (bed + adj_auth).astype("float32")
            # enforce exact authoritative values where present
            bed[m_auth] = authoritative_bed[m_auth].astype("float32")
            depth = (wse_map - bed).astype("float32")
            LOG.info("Authoritative bed: anchored %d cells; blended to %.1fm (sigma=%.1fm).", n_auth, float(args.authoritative_bed_max_dist_m), float(args.residual_blend_sigma_m))
        else:
            LOG.info("Authoritative bed raster provided but has no finite data within channel mask.")

    # Optional: incorporate bed-elevation soundings (mode=bed_elev).
    # We correct the *bed elevation* surface directly (in DEM's vertical datum), then recompute depth.
    # Optional: incorporate bed-elevation soundings (mode=bed_elev).
    # We correct the *bed elevation* surface directly, then recompute depth.
    if (snd_bed_grid is not None) and (str(getattr(args, "soundings_mode", "auto")).strip().lower() == "bed_elev"):
        m_bed = np.isfinite(snd_bed_grid) & channel
        n_bed = int(np.count_nonzero(m_bed))
        if n_bed > 0:
            try:
                resid = (snd_bed_grid - bed).astype("float32")
                maxd = float(getattr(args, "soundings_max_dist_m", 1500.0))
                adj = _build_residual_adjustment(resid, m_bed, channel, maxd, float(args.residual_blend_sigma_m), float(pix))
                bed = (bed + adj).astype("float32")
                if bool(getattr(args, "soundings_enforce", True)):
                    bed[m_bed] = snd_bed_grid[m_bed].astype("float32")
                depth = (wse_map - bed).astype("float32")
                LOG.info("Soundings: bed-elevation blending using %d cells (max_dist=%.1fm, sigma=%.1fm).", n_bed, maxd, float(args.residual_blend_sigma_m))
            except Exception as e:
                LOG.warning("Soundings: failed bed-elevation blending: %s", e)

    out_bed = Path(args.out_bed)
    out_bed.parent.mkdir(parents=True, exist_ok=True)
    

    # Optional: longitudinal bed profile constraints (slope/curvature)
    try:
        bmaxs = float(getattr(args, 'bed_profile_max_slope', 0.0) or 0.0)
        bmaxc = float(getattr(args, 'bed_profile_max_curv', 0.0) or 0.0)
        bstep = float(getattr(args, 'bed_profile_step_m', 25.0) or 25.0)
        bstr = float(getattr(args, 'bed_profile_strength', 0.6) or 0.0)
        bpow = float(getattr(args, 'bed_profile_power', 2.0) or 2.0)
        if (bstr > 0.0) and ((bmaxs > 0.0) or (bmaxc > 0.0)):
            dbg = None
            if getattr(args, 'debug_dir', None):
                dbg = str(Path(args.debug_dir) / 'bed_profile')
            bed_pre_constraints = bed.copy()
            bed = _apply_bed_profile_constraints(
                bed=bed,
                template_profile=template_profile,
                river_gpkg=Path(args.river_gpkg),
                channel_mask=channel,
                step_m=bstep,
                max_slope=bmaxs,
                max_curv=bmaxc,
                strength=bstr,
                power=bpow,
                logger=LOG,
                debug_dir=dbg,
            )
            # Re-impose mainstem corridor after bed profile constraints as well.
            if (mainstem_corridor is not None) and np.any(mainstem_corridor):
                bed[mainstem_corridor] = bed_pre_constraints[mainstem_corridor]
                LOG.info("Mainstem corridor preserved after profile constraints (cells=%d).", int(np.count_nonzero(mainstem_corridor)))
    except Exception as e:
        LOG.warning('Bed profile constraints failed; continuing without them. Error: %s', e)

    # Use spaces for indentation in this block to avoid TabError.
    _save_f32(out_bed, bed, template_profile)

    if args.debug_dir:
        ddir = Path(args.debug_dir)
        ddir.mkdir(parents=True, exist_ok=True)
        _save_f32(ddir / "depth_m.tif", depth, template_profile)
        _save_f32(ddir / "wse_m.tif", wse_map, template_profile)
        _save_f32(ddir / "debug_w_proxy_m.tif", w_proxy_m, template_profile)
        _save_f32(ddir / "debug_mainstem_corridor_mask.tif", preserve_mainstem.astype("uint8"), template_profile, nodata=255.0)
        _save_f32(ddir / "r.tif", np.where(channel, r, np.nan), template_profile)
        _save_f32(ddir / "d_bank_m.tif", np.where(channel, d_bank, np.nan), template_profile)
        _save_f32(ddir / "d_center_m.tif", np.where(channel, d_center, np.nan), template_profile)
        _save_f32(ddir / "dmax_m.tif", np.where(channel, dmax_map, np.nan), template_profile)
        if snd_depth_grid is not None:
            _save_f32(ddir / 'snd_depth_obs_m.tif', np.where(channel, snd_depth_grid, np.nan), template_profile)
        if snd_bed_grid is not None:
            _save_f32(ddir / 'snd_bed_obs_m.tif', np.where(channel, snd_bed_grid, np.nan), template_profile)
        if snd_dmax_grid is not None:
            _save_f32(ddir / 'snd_dmax_implied_m.tif', np.where(channel, snd_dmax_grid, np.nan), template_profile)
        if snd_dist is not None:
            _save_f32(ddir / 'snd_dist_m.tif', np.where(channel, snd_dist, np.nan), template_profile)

    LOG.info("Wrote bed raster: %s", out_bed)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    raise SystemExit(main())