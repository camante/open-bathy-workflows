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
from scipy.ndimage import distance_transform_edt

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
    wse_map: np.ndarray | None = None,
):
    """Return (depth_obs_grid, dmax_implied_grid, bed_obs_grid) on the template grid (nan where none).

    - depth_obs_grid: positive-down depths (m), when derivable.
    - dmax_implied_grid: implied Dmax at sounding cells based on r-position.
    - bed_obs_grid: bed elevations (m), only populated when mode == 'bed_elev'.
    """
    depth_grid = np.full(channel.shape, np.nan, dtype="float32")
    dmax_grid = np.full(channel.shape, np.nan, dtype="float32")
    bed_grid = np.full(channel.shape, np.nan, dtype="float32")

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

    if mode_l == "bed_elev":
        # Treat Z as bed elevation in the same vertical datum as DEM/WSE (e.g., NAVD88).
        bed = pd.to_numeric(pd.Series(z), errors="coerce").to_numpy(dtype="float64")
        m3 = np.isfinite(bed)
        rr2, cc2, bed = rr[m3], cc[m3], bed[m3]
        if rr2.size == 0:
            return depth_grid, dmax_grid, bed_grid

        # For elevations: "deeper" = lower elevation. Keep the minimum bed elevation per cell.
        for r0, c0, bv in zip(rr2, cc2, bed):
            cur = bed_grid[r0, c0]
            if (not np.isfinite(cur)) or (bv < cur):
                bed_grid[r0, c0] = float(bv)

        # If we have a WSE map, derive observed depths and implied Dmax.
        if wse_map is not None:
            wse = wse_map[rr2, cc2].astype("float64")
            d = (wse - bed).astype("float64")
            # Keep only physically plausible depths
            ok = np.isfinite(d) & (d >= 0.0)
            rr3, cc3, d = rr2[ok], cc2[ok], d[ok]
            if rr3.size > 0:
                # Depth obs grid (keep deepest per cell)
                for r0, c0, dv in zip(rr3, cc3, d):
                    cur = depth_grid[r0, c0]
                    if (not np.isfinite(cur)) or (dv > cur):
                        depth_grid[r0, c0] = float(dv)

                rvals = r[rr3, cc3].astype("float64")
                ruse = np.maximum(rvals, float(min_r))
                denom = np.power(ruse, float(shape_exp)) + 1e-6
                dmax_imp = (d / denom).astype("float64")
                dmax_imp = np.clip(dmax_imp, float(dmax_min_m), float(dmax_max_m))

                for r0, c0, dv in zip(rr3, cc3, dmax_imp):
                    cur = dmax_grid[r0, c0]
                    if (not np.isfinite(cur)) or (dv > cur):
                        dmax_grid[r0, c0] = float(dv)
        else:
            LOG.warning("soundings-mode=bed_elev but no wse_map provided; skipping depth/Dmax inference from soundings.")

        return depth_grid, dmax_grid, bed_grid

    # Otherwise, treat as depths (with sign handling).
    d = _coerce_depth(z, mode=mode_l)
    if d.size == 0:
        return depth_grid, dmax_grid, bed_grid

    # align lengths with finite depths
    # NOTE: _coerce_depth already drops NaNs but we keep the original rr/cc alignment by re-coercing
    d2 = pd.to_numeric(pd.Series(z), errors="coerce").to_numpy(dtype="float64")
    # We'll recompute depth again with the same mode to preserve elementwise alignment:
    d_aligned = _coerce_depth(d2, mode=mode_l)
    # The alignment step above isn't safe; instead, re-run with elementwise coercion:
    # We'll fall back to simple behavior: treat z numeric and apply mode transform elementwise.
    z_num = pd.to_numeric(pd.Series(z), errors="coerce").to_numpy(dtype="float64")
    frac_neg = float((z_num < 0.0).mean()) if np.isfinite(z_num).any() else 0.0
    if mode_l == "depth_pos":
        d_elem = z_num
    elif mode_l == "depth_neg":
        d_elem = -z_num
    else:
        d_elem = (-z_num) if frac_neg >= 0.7 else z_num
    d_elem = np.abs(d_elem)
    ok = np.isfinite(d_elem)
    rr2, cc2, d_elem = rr[ok], cc[ok], d_elem[ok]
    if rr2.size == 0:
        return depth_grid, dmax_grid, bed_grid

    # Depth obs grid (keep deepest per cell)
    for r0, c0, dv in zip(rr2, cc2, d_elem):
        cur = depth_grid[r0, c0]
        if (not np.isfinite(cur)) or (dv > cur):
            depth_grid[r0, c0] = float(dv)

    # Implied Dmax via r-position
    rvals = r[rr2, cc2].astype("float64")
    ruse = np.maximum(rvals, float(min_r))
    denom = np.power(ruse, float(shape_exp)) + 1e-6
    dmax_imp = (d_elem / denom).astype("float64")
    dmax_imp = np.clip(dmax_imp, float(dmax_min_m), float(dmax_max_m))

    for r0, c0, dv in zip(rr2, cc2, dmax_imp):
        cur = dmax_grid[r0, c0]
        if (not np.isfinite(cur)) or (dv > cur):
            dmax_grid[r0, c0] = float(dv)

    return depth_grid, dmax_grid, bed_grid


def main() -> int:
    p = argparse.ArgumentParser(description="Generate river bed elevations using a channel-skeleton distance-transform method.")
    p.add_argument("--river-gpkg", required=True, help="river_network.gpkg from river_network.py")
    p.add_argument("--rivers-layer", default="rivers_clip", help="Layer name inside gpkg (default rivers_clip)")
    p.add_argument("--template-raster", required=True, help="Template raster defining output grid (usually river_dem.tif)")
    p.add_argument("--dem", required=True, help="DEM raster (same vertical datum as desired bed elevations)")
    p.add_argument("--channel-mask", required=True, help="River channel mask raster (1=river channel pixels)")
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
    p.add_argument("--soundings-max-dist-m", type=float, default=1500.0,
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
                wse_map=wse_map,
            )
            snd_mask = np.isfinite(snd_dmax_grid)
            if np.any(snd_mask):
                inv_snd = np.ones(shape, dtype=np.uint8)
                inv_snd[snd_mask] = 0
                dist_snd, inds_snd = distance_transform_edt(inv_snd, sampling=pix, return_indices=True)
                snd_dist = dist_snd.astype("float32")
                ny_s, nx_s = inds_snd[0], inds_snd[1]
                snd_dmax_map = snd_dmax_grid[ny_s, nx_s].astype("float32")

                use = skeleton & (snd_dist <= float(args.soundings_max_dist_m)) & np.isfinite(snd_dmax_map)
                n_use = int(np.count_nonzero(use))
                if n_use > 0:
                    dmax_skel = np.where(use, np.maximum(dmax_skel, snd_dmax_map), dmax_skel).astype("float32")
                    LOG.info(
                        "Soundings: updated Dmax prior on %d skeleton pixels (max_dist=%.1fm, mode=%s).",
                        n_use, float(args.soundings_max_dist_m), str(args.soundings_mode)
                    )
                else:
                    LOG.info("Soundings: none within max_dist of skeleton; Dmax prior unchanged.")
            else:
                LOG.info("Soundings provided but no valid points fell inside the channel mask.")
        except Exception as e:
            LOG.warning("Failed to incorporate soundings into Dmax prior: %s", e)

    # Propagate Dmax from skeleton to the full channel domain
    dmax_map = dmax_skel[ny, nx].astype("float32")

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

    # Optional: incorporate bed-elevation soundings (mode=bed_elev).
    # We correct the *bed elevation* surface directly (in DEM's vertical datum), then recompute depth.
    if (snd_bed_grid is not None) and (str(getattr(args, "soundings_mode", "auto")).strip().lower() == "bed_elev"):
        m_bed = np.isfinite(snd_bed_grid) & channel
        n_bed = int(np.count_nonzero(m_bed))
        if n_bed > 0:
            try:
                # Residual at sounding cells: observed_bed - predicted_bed
                resid = (snd_bed_grid - bed).astype("float32")

                # Nearest-neighbor residual propagation (fast, stable). Weight decays linearly to 0 at max_dist.
                inv_b = np.ones(shape, dtype=np.uint8)
                inv_b[m_bed] = 0
                dist_b, inds_b = distance_transform_edt(inv_b, sampling=pix, return_indices=True)
                nyb, nxb = inds_b[0], inds_b[1]
                resid_nn = resid[nyb, nxb].astype("float32")

                maxd = float(getattr(args, "soundings_max_dist_m", 1500.0))
                w = np.clip(1.0 - (dist_b.astype("float32") / (maxd + 1e-6)), 0.0, 1.0).astype("float32")

                apply = channel & np.isfinite(resid_nn) & (dist_b <= maxd)
                bed = np.where(apply, bed + (w * resid_nn), bed).astype("float32")

                # Optionally enforce exact observed bed at sounding cells
                if bool(getattr(args, "soundings_enforce", True)):
                    bed[m_bed] = snd_bed_grid[m_bed].astype("float32")

                # Recompute depth from corrected bed
                depth = (wse_map - bed).astype("float32")

                LOG.info("Soundings: applied bed-elevation correction using %d sounding cells (max_dist=%.1fm).", n_bed, maxd)
            except Exception as e:
                LOG.warning("Soundings: failed bed-elevation correction: %s", e)


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
