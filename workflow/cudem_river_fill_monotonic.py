#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cudem_river_fill_monotonic.py

End-to-end river bed patch workflow:
  1) river_network.py
  2) xs_builder.py
  3) xs_adjust_monotonic.py (optional)
  4) xs_infer_bathy_raster.py  (produces patch raster + mask raster)
  5) bank_mask_from_xs.py
  6) measured_mask_from_points.py (optional)
  7) cudem_river_burn_taper.py

Enhancements:
  - Optional waffles-derived channel mask (coastline module; want_nhd=True)
  - OPTIONAL refinement: (waffles water mask) ∧ (buffer(flowlines)) to remove stray water polygons
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("cudem_river_fill")

# ---------------------------
# Utilities
# ---------------------------

def _run(cmd_list: list[str]) -> None:
    pretty = " ".join(cmd_list)
    log.info("[RUN] %s", pretty)
    p = subprocess.run(cmd_list, capture_output=True, text=True)
    if p.stdout and p.stdout.strip():
        log.info(p.stdout.strip())
    if p.returncode != 0:
        if p.stderr and p.stderr.strip():
            log.error(p.stderr.strip())
        raise RuntimeError(f"Stage failed: {pretty}")


def _run_shell(cmd: str) -> None:
    """Disabled for security; use _run(list_args)."""
    raise RuntimeError("shell=True execution is disabled; provide argv list")


def _apply_channel_mask_to_patch(patch_raster: Path, patch_mask: Path, channel_mask: Path, inside_value: int = 0, invert: bool = False) -> None:
    """Hard-clip the patch raster + its mask raster to a channel mask.

    Channel mask semantics (your case):
      - water pixels: value == inside_value (usually 0)
      - everything else: NoData (mask band is 0)

    We enforce:
      - patch_raster: outside channel -> nodata (-9999)
      - patch_mask: outside channel -> 0
    """
    import numpy as np
    import rasterio

    patch_raster = Path(patch_raster)
    patch_mask = Path(patch_mask)
    channel_mask = Path(channel_mask)

    if (not patch_raster.exists()) or (not patch_mask.exists()) or (not channel_mask.exists()):
        return

    with rasterio.open(channel_mask) as cm:
        cm_arr = cm.read(1)
        cm_valid = cm.read_masks(1) > 0

    with rasterio.open(patch_raster) as pr:
        pr_arr = pr.read(1)
        pr_profile = pr.profile.copy()
        pr_nodata = pr.nodata if pr.nodata is not None else -9999.0

    with rasterio.open(patch_mask) as pm:
        pm_arr = pm.read(1)
        pm_profile = pm.profile.copy()

    # Inside channel: valid AND equals inside_value
    inside = cm_valid & (cm_arr == inside_value)
    if invert:
        inside = cm_valid & (cm_arr != inside_value)

    # Clip patch raster
    pr_out = pr_arr.copy()
    pr_out[~inside] = pr_nodata

    # Clip patch mask
    pm_out = pm_arr.copy()
    pm_out[~inside] = 0

    # Write back (keep tiling sane)
    pr_profile.update(
        driver="GTiff",
        compress="DEFLATE",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=pr_nodata,
    )
    pm_profile.update(
        driver="GTiff",
        compress="DEFLATE",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=0,
        dtype="uint8",
    )

    with rasterio.open(patch_raster, "w", **pr_profile) as dst:
        dst.write(pr_out, 1)

    with rasterio.open(patch_mask, "w", **pm_profile) as dst:
        dst.write(pm_out.astype("uint8"), 1)


def _script(path: Path, name: str) -> Path:
    p = (path / name).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Missing required script: {p}")
    return p


def _buffer_aoi(aoi: str, pct: float = 0.05) -> str:
    parts = aoi.split("/")
    if len(parts) != 4:
        raise ValueError(f"AOI must be 'W/E/S/N', got: {aoi}")
    w, e, s, n = map(float, parts)
    dx = (e - w) * pct
    dy = (n - s) * pct
    return f"{w - dx:.8f}/{e + dx:.8f}/{s - dy:.8f}/{n + dy:.8f}"


def _aoi_center(aoi: str) -> tuple[float, float]:
    parts = aoi.split("/")
    w, e, s, n = map(float, parts)
    return ((w + e) / 2.0, (s + n) / 2.0)


# ---------------------------
# Waffles mask generation + alignment
# ---------------------------


def _align_mask_to_target(src_mask: Path, target_raster: Path, dst_mask: Path) -> None:
    """
    Align src_mask to target_raster's grid.

    Semantics expected for channel/water masks in this workflow:
      * water/channel pixels = 0
      * everything else = NoData

    We therefore write an aligned mask with dtype=uint8 and nodata=255, where:
      * 0 = water/channel
      * 255 = NoData (outside)

    NOTE: We DO NOT copy tiling/blocksize metadata from the target raster, because
    some DEMs use unusual block sizes that can be invalid for GeoTIFF creation
    in GDAL/rasterio (e.g., TileWidth not a multiple of 16). We write with a
    sane, portable tiling (256x256) instead.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import reproject
    from rasterio.enums import Resampling

    dst_mask = Path(dst_mask)

    with rasterio.open(target_raster) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        h, w = ref.height, ref.width
        ref_profile = ref.profile

    # Build a clean output profile (do NOT inherit DEM block sizes)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "uint8",
        "crs": ref_crs,
        "transform": ref_transform,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 255,
        # if DEM is BigTIFF-ish, be safe
        "BIGTIFF": "IF_SAFER",
    }

    # Read source; force invalid pixels to 255 so NoData propagates through reprojection
    with rasterio.open(src_mask) as src:
        src_arr = src.read(1)
        src_valid = src.read_masks(1) > 0
        src_arr = np.where(src_valid, src_arr, 255).astype("uint8")
        src_transform = src.transform
        src_crs = src.crs

    dst = np.full((h, w), 255, dtype=np.uint8)

    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
        src_nodata=255,
        dst_nodata=255,
    )

    # Write
    with rasterio.open(dst_mask, "w", **profile) as out:
        out.write(dst, 1)

def generate_waffles_channel_mask(
    *,
    target_raster: Path,
    aoi: str,
    cache_dir: Path,
    inc_arcsec: float,
    want_lakes: bool,    refine_with_flowlines: bool,
    flowlines_gpkg: Optional[Path],
    flowlines_buffer_m: float,
) -> Path:
    """
    Generate a waffles coastline mask (want_nhd=True) and align it to target_raster.
    Uses a dummy datalist file as the required positional argument; if empty datalist yields no output,
    retries with DEM referenced inside the datalist (as '<path> <weight>').
    Optionally refines the mask by intersecting with a buffered flowline corridor.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    aoi_buf = _buffer_aoi(aoi, pct=0.05)
    waffles_inc = f"{inc_arcsec:.9f}s"
    waffles_params = f"want_nhd=True:want_lakes={str(bool(want_lakes))}"

    chash = hashlib.sha1(f"{aoi_buf}|{inc_arcsec:.9f}|{waffles_params}".encode()).hexdigest()[:12]
    base_prefix = cache_dir / f"waffles_channel_{chash}"
    raw_tif = base_prefix.with_suffix(".tif")
    aligned_tif = cache_dir / f"waffles_channel_aligned_{chash}.tif"
    refined_tif = cache_dir / f"waffles_channel_refined_{chash}.tif"

    def _run_waffles() -> None:
        # IMPORTANT: -R=<AOI> to avoid argparse treating negative coords as flags
        cmd_list = [
            "waffles",
            "-M", f"coastline:{waffles_params}",
            f"-R={aoi_buf}",
            "-E", waffles_inc,
            "-O", str(base_prefix),
        ]
        log.info("[WAFFLES] Running: %s", " ".join(cmd_list))
        p = subprocess.run(cmd_list, capture_output=True, text=True)
        if p.stdout and p.stdout.strip():
            log.info(p.stdout.strip())
        if p.returncode != 0:
            if p.stderr and p.stderr.strip():
                log.error(p.stderr.strip())
            raise RuntimeError("Waffles failed. Ensure 'waffles' is on PATH.")


    # 1) Run waffles if needed
    if (not raw_tif.exists()) or raw_tif.stat().st_size == 0:
        _run_waffles()

        # Find produced tif (some builds may vary suffix)
        if not raw_tif.exists():
            cands = list(cache_dir.glob(f"waffles_channel_{chash}*.tif")) + list(cache_dir.glob(f"waffles_channel_{chash}*.tiff"))
            if cands:
                raw_tif = cands[0]

        if (not raw_tif.exists()) or raw_tif.stat().st_size == 0:
            raise RuntimeError("Waffles did not produce a coastline raster (no .tif found).")

    # 2) Align to DEM grid
    if (not aligned_tif.exists()) or aligned_tif.stat().st_size == 0:
        log.info("[MASK] Aligning waffles mask to DEM grid: %s", aligned_tif)
        _align_mask_to_target(raw_tif, target_raster, aligned_tif)

    # 3) Optional refinement: waffles_mask ∧ buffer(flowlines)
    if refine_with_flowlines:
        if flowlines_gpkg is None or not Path(flowlines_gpkg).exists():
            log.warning("[MASK] refine_with_flowlines requested but flowlines gpkg missing; using aligned waffles mask.")
            return aligned_tif

        if (not refined_tif.exists()) or refined_tif.stat().st_size == 0:
            log.info("[MASK] Refining channel mask with buffered flowlines (%.1f m): %s", flowlines_buffer_m, refined_tif)
            _refine_mask_with_flowlines(
                aligned_mask=aligned_tif,
                river_gpkg=Path(flowlines_gpkg),
                target_raster=target_raster,
                out_mask=refined_tif,
                waffles_inside_value=0,  # waffles convention assumed; caller can invert/inside-value downstream if needed
                buffer_m=float(flowlines_buffer_m),
                aoi=aoi,
            )
        return refined_tif

    return aligned_tif


def _refine_mask_with_flowlines(
    *,
    aligned_mask: Path,
    river_gpkg: Path,
    target_raster: Path,
    out_mask: Path,
    waffles_inside_value: int,
    buffer_m: float,
    aoi: str,
) -> None:
    """
    Build a corridor mask by buffering flowlines, rasterize to target grid, then AND with waffles mask.
    Output preserves waffles raster semantics: inside_value stays inside_value, outside is 1-inside_value.
    """
    import numpy as np
    import rasterio
    import geopandas as gpd
    from rasterio.features import rasterize
    from pyproj import CRS

    # Load flowlines layer
    gdf = None
    for layer_name in ("rivers_clip", "rivers_aoi", "edges"):
        try:
            gdf = gpd.read_file(river_gpkg, layer=layer_name)
            if gdf is not None and not gdf.empty:
                break
        except Exception:
            gdf = None
    if gdf is None or gdf.empty:
        # fallback: first layer
        gdf = gpd.read_file(river_gpkg)
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        raise RuntimeError("No flowline geometries available to refine channel mask.")

    # Choose a projected CRS for buffering if needed
    wgs84 = CRS.from_epsg(4326)
    lon, lat = _aoi_center(aoi)
    # build UTM EPSG code
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    utm = CRS.from_epsg(epsg)

    # Project to UTM for buffer if not already projected
    if gdf.crs is None:
        # assume WGS84 if missing (common with some geojson services)
        gdf = gdf.set_crs(wgs84, allow_override=True)

    # buffer in meters
    gdf_utm = gdf.to_crs(utm)
    corridor = gdf_utm.geometry.buffer(buffer_m)
    # dissolve to reduce rasterize cost
    corridor_union = corridor.union_all() if hasattr(corridor, 'union_all') else corridor.unary_union

    # Reproject corridor back to target raster CRS for rasterization
    with rasterio.open(target_raster) as ref:
        ref_crs = ref.crs
        transform = ref.transform
        out_shape = (ref.height, ref.width)

    corridor_gdf = gpd.GeoSeries([corridor_union], crs=utm).to_crs(ref_crs)
    corridor_geom = corridor_gdf.iloc[0]

    bufmask = rasterize(
        [(corridor_geom, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )

    with rasterio.open(aligned_mask) as ms:
        m = ms.read(1)
        valid = ms.read_masks(1) > 0  # True where not NoData

        # waffles mask semantics for this workflow:
        #   water/channel == waffles_inside_value (typically 0)
        #   everything else is NoData (mask==invalid)
        inside = (valid & (m == waffles_inside_value) & (bufmask == 1))

        # Output refined mask with 0=water and NoData=255 outside
        out_arr = np.where(inside, 0, 255).astype("uint8")

        profile = ms.profile.copy()
        profile.update(dtype="uint8", count=1, compress="DEFLATE", tiled=True, nodata=255)

    with rasterio.open(out_mask, "w", **profile) as dst:
        dst.write(out_arr, 1)


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        "cudem_river_fill_monotonic.py – CUDEM river bathy gap-fill (XS-based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--aoi", required=True, help="W/E/S/N")
    ap.add_argument("--base-dem", required=True, help="Base DEM to patch/burn into (grid template)")
    ap.add_argument("--topo-dem", required=True, help="Topo DEM used to build XS (often same as base-dem)")
    ap.add_argument("--out-dem", required=True, help="Final output DEM")
    ap.add_argument("--work-dir", default="output/riverfill_work")
    ap.add_argument("--scripts-dir", default=None, help="Directory containing the companion scripts")

    ap.add_argument("--soundings", default=None, help="Optional point dataset to mark measured bathy locations (protect)")
    ap.add_argument("--soundings-depth-col", default=None)

    ap.add_argument("--enforce-monotonic", action="store_true", help="Enforce downstream monotonic bed profile per centerline")
    ap.add_argument("--monotonic-epsilon-m", type=float, default=0.02)

    ap.add_argument("--taper-m", type=float, default=20.0)
    ap.add_argument("--max-abs-change-m", type=float, default=5.0)
    ap.add_argument("--bank-buffer-m", type=float, default=5.0)

    # Channel mask (preferred): provide a DEM-aligned raster where INSIDE channel pixels have a known value
    # (commonly 0 for waffles-style masks) and outside-channel is either different value or NoData.
    ap.add_argument("--channel-mask-raster", default=None,
                    help="Path to an existing DEM-aligned channel mask raster. If provided, this is used instead of generating a waffles mask.")

    # Waffles channel mask
    ap.add_argument("--use-waffles-channel-mask", action="store_true",
                    help="Generate and use a waffles-derived channel mask (coastline module; want_nhd=True)")
    ap.add_argument("--waffles-inc-arcsec", type=float, default=None,
                    help="Waffles resolution in arc-seconds. If not set, uses WAFFLES_INC_ARCSEC env var or 1.0")
    ap.add_argument("--waffles-want-lakes", action="store_true",
                    help="Include lakes/reservoirs in waffles channel mask (want_lakes=True)")
    ap.add_argument("--channel-mask-inside-value", type=int, default=0,
                    help="Mask value that represents INSIDE the channel (often 0 for waffles masks)")
    ap.add_argument("--channel-mask-invert", action="store_true",
                    help="Invert channel mask semantics when passing to infer stage")

    # NEW: refine waffles with flowlines corridor
    ap.add_argument("--refine-channel-with-flowlines", action="store_true", default=True,
                    help="Refine waffles channel mask by intersecting with buffered flowlines (recommended).")
    ap.add_argument("--flowlines-buffer-m", type=float, default=100.0,
                    help="Buffer distance (meters) around flowlines to keep waffles water polygons (corridor).")

    args = ap.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(args.scripts_dir).resolve() if args.scripts_dir else Path(__file__).resolve().parent

    river_py = _script(scripts_dir, "river_network.py")
    xs_py = _script(scripts_dir, "xs_builder.py")
    adj_py = _script(scripts_dir, "xs_adjust_monotonic.py")
    infer_py = _script(scripts_dir, "xs_infer_bathy_raster.py")
    bank_py = _script(scripts_dir, "bank_mask_from_xs.py")
    meas_py = _script(scripts_dir, "measured_mask_from_points.py")
    burn_py = _script(scripts_dir, "cudem_river_burn_taper.py")

    base_dem = Path(args.base_dem).resolve()
    topo_dem = Path(args.topo_dem).resolve()
    out_dem = Path(args.out_dem).resolve()
    out_provenance = work / (out_dem.stem + "_provenance.tif")

    river_net = work / "river_network.gpkg"
    xs_gpkg = work / "river_xs.gpkg"
    xs_bathy_gpkg = work / "river_xs_bathy.gpkg"
    bank_mask = work / "river_bank_mask.tif"
    meas_mask = work / "river_measured_mask.tif"

    # 1) River network (needed for XS + for refining channel mask)
    _run(["python", str(river_py), f"--aoi={args.aoi}", f"--out-gpkg={river_net}"])

    # 1b) Optional: waffles channel mask (after river_net exists so we can refine it)
    # 1b) Channel mask for interpolation footprint
    # Prefer an explicit mask path if provided. Otherwise optionally generate one from waffles.
    channel_mask = Path(args.channel_mask_raster).resolve() if args.channel_mask_raster else None
    if channel_mask is not None:
        if not channel_mask.exists():
            raise FileNotFoundError(f"--channel-mask-raster not found: {channel_mask}")
        log.info("[MASK] Using provided channel mask: %s", str(channel_mask))
    elif args.use_waffles_channel_mask:
        inc = args.waffles_inc_arcsec
        if inc is None:
            inc = float(os.environ.get("WAFFLES_INC_ARCSEC", "1.0"))

        channel_mask = Path(generate_waffles_channel_mask(
            target_raster=base_dem,
            aoi=args.aoi,
            cache_dir=work,
            inc_arcsec=float(inc),
            want_lakes=bool(args.waffles_want_lakes),
            refine_with_flowlines=bool(args.refine_channel_with_flowlines),
            flowlines_gpkg=river_net,
            flowlines_buffer_m=float(args.flowlines_buffer_m),
        )).resolve()
        log.info("[MASK] Using generated channel mask: %s", str(channel_mask))
    else:
        log.warning("[MASK] No channel mask provided/generated. Interpolation footprint will default to a corridor around cross-section points (may leave gaps between XS).")

    # 2) XS builder
    _run(["python", str(xs_py), f"--river-gpkg={river_net}", f"--dem={topo_dem}", f"--out-gpkg={xs_gpkg}"])

    # 3) Infer bed points (XS -> bathy points gpkg) and (optionally) enforce monotonic downstream bed profile
    #    We always run the infer step to create the bathy points GPKG. Rasterization is performed after monotonic (if enabled).
    xs_bathy_gpkg = work / "river_xs_bathy.gpkg"
    patch_raster_unadjusted = work / "river_bed_patch_unadjusted.tif"
    patch_mask_unadjusted = work / "river_bed_mask_unadjusted.tif"

    infer_cmd = [
        "python", str(infer_py),
        f"--xs-gpkg={xs_gpkg}",
        f"--out-gpkg={xs_bathy_gpkg}",
        f"--template-raster={base_dem}",
        f"--out-bathy-raster={patch_raster_unadjusted}",
        f"--out-mask-raster={patch_mask_unadjusted}",
    ]
    if channel_mask:
        infer_cmd.extend([
            f"--channel-mask-raster={channel_mask}",
            f"--channel-mask-inside-value={int(args.channel_mask_inside_value)}",
        ])
        if args.channel_mask_invert:
            infer_cmd.append("--channel-mask-invert")
    if args.soundings:
        infer_cmd.append(f"--soundings={args.soundings}")
        if args.soundings_depth_col:
            infer_cmd.append(f"--soundings-depth-col={args.soundings_depth_col}")
    _run(infer_cmd)

    bathy_points_layer = "xs_bathy_points"
    bathy_points_layer_to_raster = bathy_points_layer

    if args.enforce_monotonic:
        # xs_adjust_monotonic operates on a bathy-points layer, not on the raw cross-section output.
        bathy_points_layer_to_raster = "xs_bathy_points_monotonic"
        _run([
            "python", str(adj_py),
            f"--in-gpkg={xs_bathy_gpkg}",
            f"--in-layer={bathy_points_layer}",
            f"--out-gpkg={xs_bathy_gpkg}",
            f"--out-layer={bathy_points_layer_to_raster}",
            f"--epsilon-m={args.monotonic_epsilon_m}",
        ])

    # 4) Rasterize (after monotonic, if enabled)
    # IMPORTANT: avoid relying on a separate "only_cross_sections" wrapper, because
    # the CLI may drift. Instead, we call xs_infer_bathy_raster.py in raster-only mode.
    patch_raster = work / "river_bed_patch.tif"
    patch_mask = work / "river_bed_mask.tif"

    raster_cmd = [
        "python", str(infer_py),
        f"--raster-from-gpkg={xs_bathy_gpkg}",
        f"--raster-layer={bathy_points_layer_to_raster}",
        "--raster-value-col=z_bed_pred_m",
        "--raster-uncert-col=uncert_m",
        f"--template-raster={base_dem}",
        f"--out-bathy-raster={patch_raster}",
        f"--out-mask-raster={patch_mask}",
    ]
    _run(raster_cmd)

    # If we have a channel mask (0=water, nodata elsewhere), hard-clip the patch to it.
    if channel_mask:
        _apply_channel_mask_to_patch(patch_raster, patch_mask, Path(channel_mask), inside_value=int(args.channel_mask_inside_value), invert=bool(args.channel_mask_invert))
    # 7) Bank protection mask
    _run([
        "python", str(bank_py),
        f"--xs-gpkg={xs_gpkg}",
        f"--template-raster={base_dem}",
        f"--out-mask={bank_mask}",
        f"--bank-buffer-m={args.bank_buffer_m}",
    ])

    # 6) Optional measured mask
    meas_mask_arg = ""
    if args.soundings:
        cmd = [
            "python", str(meas_py),
            f"--points={args.soundings}",
            f"--template-raster={base_dem}",
            f"--out-mask={meas_mask}",
        ]
        if args.soundings_depth_col:
            cmd.append(f"--depth-col={args.soundings_depth_col}")
        _run(cmd)
        meas_mask_arg = f"--measured-mask={meas_mask}"

    # 7) Burn + taper into DEM
    burn_cmd = [
        "python", str(burn_py),
        f"--base-dem={base_dem}",
        f"--patch-dem={patch_raster}",
        f"--patch-mask={patch_mask}",
        f"--bank-mask={bank_mask}",
        f"--out-dem={out_dem}",
        f"--out-provenance={out_provenance}",
        f"--taper-m={args.taper_m}",
        f"--max-abs-change-m={args.max_abs_change_m}",
    ]
    if meas_mask_arg:
        burn_cmd.append(meas_mask_arg)
    _run(burn_cmd)


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()