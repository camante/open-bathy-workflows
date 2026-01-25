# save as make_channel_mask.py and run it once
from pathlib import Path
import os
import shlex
import logging
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import subprocess


# Logging is configured by entrypoints (e.g., bathy_main.py / sdb_main.py).
# Standalone scripts configure logging in __main__.

log = logging.getLogger(__name__)

def _run(cmd: str) -> str:
    res = subprocess.run(shlex.split(cmd), shell=False, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd}\n{res.stderr}")
    return res.stdout

def align_mask_to_target(src_mask: str, target_raster: str, dst_mask: str):
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
            resampling=Resampling.nearest,
            src_nodata=None,
            dst_nodata=None,
        )

    with rasterio.open(dst_mask, "w", **profile) as out:
        out.write(dst, 1)

AOI="-74.41/-74.31/40.45/40.50"
DEM="output/USACE_2012_NCMP_Lidar_DEM.tif"
RAW="output/riverfill_work/waffles_channel_raw.tif"
ALIGNED="output/riverfill_work/waffles_channel_mask_aligned.tif"

# run waffles if needed
if not Path(RAW).exists():
    _run(f"waffles -M coastline:want_nhd=True:want_lakes=False -R {AOI} -E 1.0s -O output/riverfill_work/waffles_channel_raw")

align_mask_to_target(RAW, DEM, ALIGNED)
