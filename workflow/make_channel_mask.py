#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_channel_mask.py – Align a waffles coastline/channel mask to a target raster grid

This small utility script is useful when you want to **use a waffles-derived mask**
(e.g., coastline/channel output) to constrain river products on the **exact grid** of a
target raster (typically a DEM used by the river interpolation workflow).

What it does
------------
- Reads an input mask raster (`--src-mask`) and resamples/reprojects it to match the
  grid, CRS, and extent of a target raster (`--target-raster`).
- Writes an aligned `uint8` GeoTIFF (`--dst-mask`) using nearest-neighbor resampling
  to preserve binary values.

Optional: run waffles
---------------------
If you pass `--run-waffles`, the script will generate `--src-mask` by running `waffles`
*only if the mask does not already exist*. This is handy for one-off debugging runs.

Examples
--------
    # Align an existing waffles mask to a DEM grid
    python make_channel_mask.py \
      --src-mask output/riverfill_work/waffles_channel_raw.tif \
      --target-raster output/USACE_2012_NCMP_Lidar_DEM.tif \
      --dst-mask output/riverfill_work/waffles_channel_mask_aligned.tif

    # Generate the raw mask with waffles first, then align
    python make_channel_mask.py \
      --run-waffles \
      --aoi "-74.41/-74.31/40.45/40.50" \
      --waffles-out output/riverfill_work/waffles_channel_raw \
      --src-mask output/riverfill_work/waffles_channel_raw.tif \
      --target-raster output/USACE_2012_NCMP_Lidar_DEM.tif \
      --dst-mask output/riverfill_work/waffles_channel_mask_aligned.tif

Notes
-----
- AOI format is **W/E/S/N** (lon/lat degrees), consistent with the rest of the workflow.
- If you use waffles output directly, remember waffles typically writes GeoTIFFs with
  a *prefix* path supplied by `-O` and appends `.tif`.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

log = logging.getLogger("make_channel_mask")


def _run_cmd(cmd: str) -> str:
    """Run a shell command (no shell=True) and return stdout; raise on error."""
    res = subprocess.run(shlex.split(cmd), shell=False, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (exit {res.returncode}):\n{cmd}\n{res.stderr}")
    return res.stdout


def _waffles_out_to_tif(path_or_prefix: str) -> Path:
    """Return the expected GeoTIFF path for a waffles -O output prefix."""
    p = Path(path_or_prefix)
    if p.suffix.lower() == ".tif":
        return p
    return Path(str(p) + ".tif")


def run_waffles_coastline(
    *,
    aoi: str,
    waffles_out: str,
    want_nhd: bool = True,
    want_lakes: bool = False,
    resolution: str = "1.0s",
) -> Path:
    """Run waffles coastline module to generate a raw mask GeoTIFF."""
    out_tif = _waffles_out_to_tif(waffles_out)
    cmd = (
        f"waffles -M coastline:want_nhd={str(want_nhd).lower()}:want_lakes={str(want_lakes).lower()} "
        f"-R {aoi} -E {resolution} -O {waffles_out}"
    )
    log.info(f"[RUN] {cmd}")
    _run_cmd(cmd)
    return out_tif


def align_mask_to_target(
    src_mask: str,
    target_raster: str,
    dst_mask: str,
    *,
    resampling: Resampling = Resampling.nearest,
) -> None:
    """Reproject/resample `src_mask` to match the grid of `target_raster`."""
    src_mask_p = Path(src_mask)
    if not src_mask_p.exists():
        raise FileNotFoundError(f"Source mask not found: {src_mask}")
    target_p = Path(target_raster)
    if not target_p.exists():
        raise FileNotFoundError(f"Target raster not found: {target_raster}")

    with rasterio.open(target_raster) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_h, dst_w = ref.height, ref.width
        profile = ref.profile.copy()
        profile.update(driver="GTiff", dtype="uint8", count=1, compress="DEFLATE", tiled=True, nodata=None)

    with rasterio.open(src_mask) as src:
        src_data = src.read(1)

        dst = np.zeros((dst_h, dst_w), dtype=np.uint8)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=None,
            dst_nodata=None,
        )

    out_p = Path(dst_mask)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_p, "w", **profile) as out:
        out.write(dst, 1)

    log.info(f"Wrote aligned mask: {out_p}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Align a waffles-derived mask raster to a target raster grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src-mask", required=True, help="Input mask GeoTIFF (typically from waffles)." )
    parser.add_argument("--target-raster", required=True, help="Target raster whose grid will be matched." )
    parser.add_argument("--dst-mask", required=True, help="Output aligned mask GeoTIFF path.")

    parser.add_argument("--run-waffles", action="store_true", default=False,
                        help="If set, run waffles to generate --src-mask if it does not already exist.")

    parser.add_argument("--aoi", default=None, help="AOI W/E/S/N (required if --run-waffles)." )
    parser.add_argument("--waffles-out", default=None,
                        help="Output prefix for waffles (-O). Required if --run-waffles." )
    parser.add_argument("--resolution", default="1.0s", help="waffles -E resolution (e.g., 1.0s, 10m)." )
    parser.add_argument("--want-nhd", action="store_true", default=False, help="If set, include NHD rivers in the waffles mask." )
    parser.add_argument("--want-lakes", action="store_true", default=False, help="If set, include lakes in the waffles mask." )
    args = parser.parse_args(argv)

    if args.run_waffles:
        if not args.aoi or not args.waffles_out:
            parser.error("--run-waffles requires --aoi and --waffles-out")
        src_p = Path(args.src_mask)
        if not src_p.exists():
            out_tif = run_waffles_coastline(
                aoi=str(args.aoi),
                waffles_out=str(args.waffles_out),
                want_nhd=bool(args.want_nhd),
                want_lakes=bool(args.want_lakes),
                resolution=str(args.resolution),
            )
            # If the user pointed --src-mask to a different path, copy/rename for clarity
            if out_tif.resolve() != src_p.resolve():
                src_p.parent.mkdir(parents=True, exist_ok=True)
                out_tif.replace(src_p)

    align_mask_to_target(args.src_mask, args.target_raster, args.dst_mask, resampling=Resampling.nearest)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    raise SystemExit(main())
