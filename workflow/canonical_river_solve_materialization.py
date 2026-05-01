from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


def _parse_aoi(aoi: str) -> tuple[float, float, float, float]:
    west, east, south, north = [float(part) for part in str(aoi).split("/")]
    return west, east, south, north


def _project_bounds(aoi: str, dst_crs: str) -> tuple[float, float, float, float]:
    west, east, south, north = _parse_aoi(aoi)
    tx = Transformer.from_crs("EPSG:4269", dst_crs, always_xy=True)
    xs = [west, east, west, east]
    ys = [south, south, north, north]
    px, py = tx.transform(xs, ys)
    return float(min(px)), float(min(py)), float(max(px)), float(max(py))


def _aligned_bounds(bounds: tuple[float, float, float, float], resolution_m: float) -> tuple[float, float, float, float]:
    west, south, east, north = bounds
    res = float(resolution_m)
    return (
        math.floor(west / res) * res,
        math.floor(south / res) * res,
        math.ceil(east / res) * res,
        math.ceil(north / res) * res,
    )


def write_canonical_solve_grid_template(path: Path, *, canonical_solve_aoi: str, projected_crs: str, target_resolution_m: float) -> Path:
    bounds = _aligned_bounds(_project_bounds(canonical_solve_aoi, projected_crs), target_resolution_m)
    west, south, east, north = bounds
    width = max(1, int(round((east - west) / float(target_resolution_m))))
    height = max(1, int(round((north - south) / float(target_resolution_m))))
    transform = from_origin(west, north, float(target_resolution_m), float(target_resolution_m))
    arr = np.full((height, width), -9999.0, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, 'w', driver='GTiff', width=width, height=height, count=1, dtype='float32',
        crs=projected_crs, transform=transform, nodata=-9999.0, compress='deflate'
    ) as ds:
        ds.write(arr, 1)
    return path


def warp_float_to_template(src_path: Path, template_path: Path, dst_path: Path) -> Path:
    with rasterio.open(template_path) as template_ds, rasterio.open(src_path) as src_ds:
        profile = template_ds.profile.copy()
        src = src_ds.read(1).astype(np.float32)
        dst = np.full((template_ds.height, template_ds.width), np.float32(-9999.0), dtype=np.float32)
        src_nodata = src_ds.nodata
        if src_nodata is not None:
            src[src == src_nodata] = np.nan
        reproject(
            source=src, destination=dst, src_transform=src_ds.transform, src_crs=src_ds.crs,
            dst_transform=template_ds.transform, dst_crs=template_ds.crs, src_nodata=np.nan,
            dst_nodata=np.float32(-9999.0), resampling=Resampling.nearest,
        )
        dst[~np.isfinite(dst)] = np.float32(-9999.0)
        for k in ('blockxsize','blockysize','BLOCKXSIZE','BLOCKYSIZE'):
            profile.pop(k, None)
        profile.update(dtype='float32', count=1, nodata=np.float32(-9999.0), compress='deflate')
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, 'w', **profile) as out_ds:
            out_ds.write(dst.astype(np.float32), 1)
    return dst_path


def write_support_mask_from_measured(measured_path: Path, support_mask_path: Path) -> Path:
    with rasterio.open(measured_path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        mask = np.isfinite(arr)
        if nodata is not None:
            mask &= arr != nodata
        mask_u8 = np.asarray(mask, dtype=np.uint8)
        profile = ds.profile.copy()
        for k in ('blockxsize','blockysize','BLOCKXSIZE','BLOCKYSIZE'):
            profile.pop(k, None)
        profile.update(dtype='uint8', nodata=0, count=1, compress='deflate')
        support_mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(support_mask_path, 'w', **profile) as out_ds:
            out_ds.write(mask_u8, 1)
    return support_mask_path




def read_template_resolution_m(path: Path) -> float:
    with rasterio.open(path) as ds:
        return float(abs(ds.transform.a))


def ensure_canonical_solve_grid_template(path: Path, *, canonical_solve_aoi: str, projected_crs: str, target_resolution_m: float) -> Path:
    if Path(path).exists():
        return Path(path)
    return write_canonical_solve_grid_template(path, canonical_solve_aoi=canonical_solve_aoi, projected_crs=projected_crs, target_resolution_m=target_resolution_m)


def ensure_warp_float_to_template(src_path: Path, template_path: Path, dst_path: Path) -> Path:
    if Path(dst_path).exists():
        return Path(dst_path)
    return warp_float_to_template(src_path, template_path, dst_path)


def ensure_support_mask_from_measured(measured_path: Path, support_mask_path: Path) -> Path:
    if Path(support_mask_path).exists():
        return Path(support_mask_path)
    return write_support_mask_from_measured(measured_path, support_mask_path)

__all__ = [
    'write_canonical_solve_grid_template',
    'warp_float_to_template',
    'write_support_mask_from_measured',
    'read_template_resolution_m',
    'ensure_canonical_solve_grid_template',
    'ensure_warp_float_to_template',
    'ensure_support_mask_from_measured',
]
