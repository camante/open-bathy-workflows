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
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt, gaussian_filter

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
    """Warp a raster to the template grid using nearest (for masks) or bilinear (for dem)."""
    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        src_nodata = src.nodata
        dst = np.zeros((template_profile["height"], template_profile["width"]), dtype=dtype)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=Resampling.bilinear if np.issubdtype(dst.dtype, np.floating) else Resampling.nearest,
            src_nodata=src_nodata,
            dst_nodata=src_nodata,
        )
    return dst, src_nodata


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


def _read_soundings_file(path: Path):
    """Read an XYZ-ish file into x/y/z arrays.

    Supported:
      - .xyz/.txt/.csv/.dat : first 3 numeric columns (header optional)
      - .gpkg/.shp/.geojson/.json : point geometries + a numeric Z/depth/elev/value column
    """
    suf = path.suffix.lower()
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
                    pass
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
    cell_percentile: float | None = None,
    wse_map: np.ndarray | None = None,
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
    depth_grid = np.full(channel.shape, np.nan, dtype="float32")
    dmax_grid = np.full(channel.shape, np.nan, dtype="float32")
    bed_grid = np.full(channel.shape, np.nan, dtype="float32")

    if not sounding_files:
        return depth_grid, dmax_grid, bed_grid

    # Optional CRS transform into template CRS
    xform = None
    if soundings_crs:
        src = CRS.from_user_input(soundings_crs)
        dst = CRS.from_user_input(template_crs)
        if src != dst:
            xform = Transformer.from_crs(src, dst, always_xy=True)

    xs_all, ys_all, zs_all = [], [], []
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
        except Exception as e:
            LOG.warning("Failed reading soundings %s: %s", pth, e)

    if not xs_all:
        return depth_grid, dmax_grid, bed_grid

    x = np.concatenate(xs_all)
    y = np.concatenate(ys_all)
    z = np.concatenate(zs_all)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if x.size == 0:
        return depth_grid, dmax_grid, bed_grid

    if xform is not None:
        x2, y2 = xform.transform(x.tolist(), y.tolist())
        x = np.asarray(x2, dtype="float64")
        y = np.asarray(y2, dtype="float64")

    rr, cc = rowcol(transform, x, y)
    rr = np.asarray(rr, dtype="int64")
    cc = np.asarray(cc, dtype="int64")

    inb = (rr >= 0) & (rr < channel.shape[0]) & (cc >= 0) & (cc < channel.shape[1])
    rr, cc, z = rr[inb], cc[inb], z[inb]
    if rr.size == 0:
        return depth_grid, dmax_grid, bed_grid

    in_ch = channel[rr, cc]
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
            out = np.where(den > 1e-6, (num / den), np.nan).astype("float32")
        else:
            out = np.where(w > 0, v_nn.astype("float32"), np.nan).astype("float32")
    else:
        out = np.where(w > 0, v_nn.astype("float32"), np.nan).astype("float32")

    out = np.where(channel_mask & (dist_m <= float(max_dist_m)), out, np.nan).astype("float32")
    return out



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

    # Priors (match bathy_main / xs_infer args)
    p.add_argument("--prior-mode", default="powerlaw", choices=["powerlaw", "multivariate"], help="Dmax prior mode")
    p.add_argument("--mv-a0", type=float, default=0.18)
    p.add_argument("--mv-bw", type=float, default=0.50)
    p.add_argument("--mv-ba", type=float, default=0.0)
    p.add_argument("--mv-bs", type=float, default=-0.10)
    p.add_argument("--mv-eps-a", type=float, default=1.0)
    p.add_argument("--mv-eps-s", type=float, default=1e-4)


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
    p.add_argument("--soundings-min-r", type=float, default=0.25,
                   help="Minimum r used when converting sounding depth -> implied Dmax (prevents blow-ups near banks).")
    p.add_argument("--no-soundings-enforce", dest="soundings_enforce", action="store_false", default=True,
                   help="Disable enforcing observed sounding depths at their grid cells (default enforces).")
    p.add_argument("--debug-dir", default=None, help="If set, write debug rasters to this directory.")
    args = p.parse_args()

    template_profile, transform, crs, shape = _read_template(Path(args.template_raster))
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

    # Water surface elevation (WSE) proxy:
    # Use DEM values along the skeleton (assumes river DEM encodes water surface reasonably in-channel).
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
                cell_percentile=args.soundings_cell_percentile,
            wse_map=wse_map,
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
    _save_f32(out_bed, bed, template_profile)

    if args.debug_dir:
        ddir = Path(args.debug_dir)
        ddir.mkdir(parents=True, exist_ok=True)
        _save_f32(ddir / "depth_m.tif", depth, template_profile)
        _save_f32(ddir / "wse_m.tif", wse_map, template_profile)
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
