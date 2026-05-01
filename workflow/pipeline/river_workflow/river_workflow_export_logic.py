from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds




def _rounded_subset_window(*, src_ds, template_ds) -> Window | None:
    if src_ds.crs != template_ds.crs:
        return None
    sx, sy = src_ds.res
    tx, ty = template_ds.res
    if not (np.isclose(float(sx), float(tx), atol=1e-6) and np.isclose(abs(float(sy)), abs(float(ty)), atol=1e-6)):
        return None
    window = from_bounds(*template_ds.bounds, transform=src_ds.transform)
    rounded = Window(
        col_off=int(round(float(window.col_off))),
        row_off=int(round(float(window.row_off))),
        width=int(round(float(window.width))),
        height=int(round(float(window.height))),
    )
    if rounded.col_off < 0 or rounded.row_off < 0:
        return None
    if rounded.col_off + rounded.width > src_ds.width or rounded.row_off + rounded.height > src_ds.height:
        return None
    expected_transform = rasterio.windows.transform(rounded, src_ds.transform)
    if expected_transform != template_ds.transform:
        return None
    if rounded.width != template_ds.width or rounded.height != template_ds.height:
        return None
    return rounded


def subset_float_raster_to_template_exact(*, src_path: Path, template_path: Path, dst_path: Path, dst_nodata: float = -9999.0) -> int:
    with rasterio.open(template_path) as template_ds, rasterio.open(src_path) as src_ds:
        window = _rounded_subset_window(src_ds=src_ds, template_ds=template_ds)
        if window is None:
            raise ValueError('linear_exact_subset_unavailable_float')
        profile = template_ds.profile.copy()
        arr = src_ds.read(1, window=window).astype(np.float32)
        src_nodata = src_ds.nodata
        if src_nodata is not None:
            arr[arr == np.float32(src_nodata)] = np.float32(dst_nodata)
        arr[~np.isfinite(arr)] = np.float32(dst_nodata)
        for k in ("blockxsize", "blockysize", "BLOCKXSIZE", "BLOCKYSIZE"):
            profile.pop(k, None)
        profile.update(dtype='float32', count=1, nodata=np.float32(dst_nodata), compress='deflate')
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink()
        with rasterio.open(dst_path, 'w', **profile) as out_ds:
            out_ds.write(arr, 1)
    return int(np.count_nonzero(arr != np.float32(dst_nodata)))


def subset_binary_mask_to_template_exact(*, src_path: Path, template_path: Path, dst_path: Path) -> int:
    with rasterio.open(template_path) as template_ds, rasterio.open(src_path) as src_ds:
        window = _rounded_subset_window(src_ds=src_ds, template_ds=template_ds)
        if window is None:
            raise ValueError('linear_exact_subset_unavailable_binary')
        profile = template_ds.profile.copy()
        arr = src_ds.read(1, window=window)
        nodata = src_ds.nodata
        mask = np.isfinite(arr)
        if nodata is not None:
            mask &= arr != nodata
        dst = np.asarray(mask, dtype=np.uint8) if arr.dtype.kind == 'f' else np.asarray(arr > 0, dtype=np.uint8)
        for k in ("blockxsize", "blockysize", "BLOCKXSIZE", "BLOCKYSIZE"):
            profile.pop(k, None)
        profile.update(dtype='uint8', count=1, nodata=0, compress='deflate')
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink()
        with rasterio.open(dst_path, 'w', **profile) as out_ds:
            out_ds.write(dst, 1)
    return int(np.count_nonzero(dst))

def warp_float_raster_to_template(*, src_path: Path, template_path: Path, dst_path: Path, dst_nodata: float = -9999.0) -> int:
    with rasterio.open(template_path) as template_ds, rasterio.open(src_path) as src_ds:
        profile = template_ds.profile.copy()
        src = src_ds.read(1).astype(np.float32)
        src_nodata = src_ds.nodata
        if src_nodata is not None:
            src[src == np.float32(src_nodata)] = np.nan
        dst = np.full((template_ds.height, template_ds.width), np.float32(dst_nodata), dtype=np.float32)
        reproject(
            source=src,
            destination=dst,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=template_ds.transform,
            dst_crs=template_ds.crs,
            src_nodata=np.nan,
            dst_nodata=np.float32(dst_nodata),
            resampling=Resampling.nearest,
        )
        dst[~np.isfinite(dst)] = np.float32(dst_nodata)
        for k in ("blockxsize", "blockysize", "BLOCKXSIZE", "BLOCKYSIZE"):
            profile.pop(k, None)
        profile.update(dtype="float32", count=1, nodata=np.float32(dst_nodata), compress="deflate")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink()
        with rasterio.open(dst_path, "w", **profile) as out_ds:
            out_ds.write(dst, 1)
    return int(np.count_nonzero(dst != np.float32(dst_nodata)))


def warp_binary_mask_to_template(*, src_path: Path, template_path: Path, dst_path: Path) -> int:
    with rasterio.open(template_path) as template_ds, rasterio.open(src_path) as src_ds:
        profile = template_ds.profile.copy()
        arr = src_ds.read(1)
        nodata = src_ds.nodata
        mask = np.isfinite(arr)
        if nodata is not None:
            mask &= arr != nodata
        src_mask = np.asarray(mask, dtype=np.uint8)
        dst = np.zeros((template_ds.height, template_ds.width), dtype=np.uint8)
        reproject(
            source=src_mask,
            destination=dst,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=template_ds.transform,
            dst_crs=template_ds.crs,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        dst = np.asarray(dst > 0, dtype=np.uint8)
        for k in ("blockxsize", "blockysize", "BLOCKXSIZE", "BLOCKYSIZE"):
            profile.pop(k, None)
        profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink()
        with rasterio.open(dst_path, "w", **profile) as out_ds:
            out_ds.write(dst, 1)
    return int(np.count_nonzero(dst))


def stage_mask_handoff(*, src_path: Path, dst_path: Path) -> int:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    shutil.copyfile(src_path, dst_path)
    with rasterio.open(dst_path) as ds:
        arr = ds.read(1)
    return int(np.count_nonzero(arr > 0))


def build_take_mask(*, corridor_mask_path: Path, support_mask_path: Path, dst_path: Path) -> int:
    with rasterio.open(corridor_mask_path) as corridor_ds, rasterio.open(support_mask_path) as support_ds:
        corridor = corridor_ds.read(1) > 0
        support = support_ds.read(1) > 0
        take = np.asarray(corridor & ~support, dtype=np.uint8)
        profile = corridor_ds.profile.copy()
        for k in ("blockxsize", "blockysize", "BLOCKXSIZE", "BLOCKYSIZE"):
            profile.pop(k, None)
        profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink()
        with rasterio.open(dst_path, "w", **profile) as out_ds:
            out_ds.write(take, 1)
    return int(np.count_nonzero(take))


__all__ = [
    "subset_float_raster_to_template_exact",
    "subset_binary_mask_to_template_exact",
    "warp_float_raster_to_template",
    "warp_binary_mask_to_template",
    "stage_mask_handoff",
    "build_take_mask",
]
