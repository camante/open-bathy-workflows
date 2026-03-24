
from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from pyproj import Transformer

from sign_semantics import raster_value_semantics
log = logging.getLogger(__name__)



def _parse_aoi(aoi: str) -> Tuple[float, float, float, float]:
    west, east, south, north = (float(v) for v in str(aoi).split('/'))
    if not (west < east and south < north):
        raise ValueError(f"Invalid AOI: {aoi}")
    return west, east, south, north


def _grid_from_aoi(aoi: str, dst_crs: str, res_m: float) -> Tuple["rasterio.Affine", int, int]:
    west, east, south, north = _parse_aoi(aoi)
    transformer = Transformer.from_crs('EPSG:4326', dst_crs, always_xy=True)
    xs, ys = transformer.transform([west, east, west, east], [south, south, north, north])
    minx = min(xs)
    maxx = max(xs)
    miny = min(ys)
    maxy = max(ys)
    res = float(res_m)
    if not np.isfinite(res) or res <= 0:
        raise ValueError(f"Invalid projected resolution: {res_m}")
    minx_s = math.floor(minx / res) * res
    miny_s = math.floor(miny / res) * res
    maxx_s = math.ceil(maxx / res) * res
    maxy_s = math.ceil(maxy / res) * res
    width = max(1, int(round((maxx_s - minx_s) / res)))
    height = max(1, int(round((maxy_s - miny_s) / res)))
    transform = rasterio.Affine(res, 0.0, minx_s, 0.0, -res, maxy_s)
    return transform, width, height


def build_projected_authoritative_raster(
    source_raster: Path | str,
    out_raster: Path | str,
    *,
    aoi: str,
    dst_crs: str,
    res_m: float,
    fallback_raster: Optional[Path | str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log = logger or logging.getLogger(__name__)
    src_path = Path(source_raster)
    out_path = Path(out_raster)
    transform, width, height = _grid_from_aoi(str(aoi), str(dst_crs), float(res_m))

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata if src.nodata is not None else -9999.0
        dst = np.full((height, width), np.float32(src_nodata), dtype='float32')
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=src_nodata,
            dst_nodata=np.float32(src_nodata),
            resampling=rasterio.enums.Resampling.bilinear,
        )
        profile = src.profile.copy()

    if fallback_raster is not None and Path(fallback_raster).exists():
        with rasterio.open(str(fallback_raster)) as fb:
            fb_nodata = fb.nodata if fb.nodata is not None else -9999.0
            fill = np.full((height, width), np.float32(fb_nodata), dtype='float32')
            rasterio.warp.reproject(
                source=rasterio.band(fb, 1),
                destination=fill,
                src_transform=fb.transform,
                src_crs=fb.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                src_nodata=fb_nodata,
                dst_nodata=np.float32(fb_nodata),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            missing = ~np.isfinite(dst) | np.isclose(dst, np.float32(src_nodata))
            valid_fill = np.isfinite(fill) & (~np.isclose(fill, np.float32(fb_nodata)))
            take = missing & valid_fill
            if np.any(take):
                dst[take] = fill[take].astype('float32')
                log.info('[AUTHORITATIVE] Filled %d projected river DEM cells from fallback DEM gaps.', int(np.count_nonzero(take)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(
        driver='GTiff',
        dtype='float32',
        count=1,
        crs=dst_crs,
        transform=transform,
        width=width,
        height=height,
        nodata=float(src_nodata),
        compress='DEFLATE',
        tiled=True,
    )
    with rasterio.open(out_path, 'w', **profile) as dst_ds:
        dst_ds.write(dst.astype('float32'), 1)

    valid = np.isfinite(dst) & (~np.isclose(dst, np.float32(src_nodata)))
    return {
        'path': str(out_path),
        'width': int(width),
        'height': int(height),
        'finite_cells': int(np.count_nonzero(valid)),
        'nodata_cells': int(np.count_nonzero(~valid)),
    }


def _vertical_target_from_compound(text: Optional[str], fallback_epsg: int = 5714) -> int:
    if not text:
        return int(fallback_epsg)
    pieces = str(text).lower().replace('epsg:', '').split('+')
    try:
        return int(pieces[-1]) if pieces else int(fallback_epsg)
    except (TypeError, ValueError):
        return int(fallback_epsg)


def _horizontal_epsg(path: Path | str) -> Optional[int]:
    with rasterio.open(str(path)) as ds:
        try:
            return ds.crs.to_epsg() if ds.crs else None
        except Exception:
            log.debug("_horizontal_epsg: suppressed exception", exc_info=True)
            return None


def export_raster_points_to_csv(
    source_raster: Path | str,
    out_csv: Path | str,
    *,
    out_crs: str,
    max_points: int = 250000,
    negative_only: Optional[bool] = True,
    source_tag: str = 'authoritative_base',
    source_vdatum: Optional[str] = None,
    target_vdatum: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log = logger or logging.getLogger(__name__)
    src_path = Path(source_raster)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as ds:
        arr = ds.read(1).astype('float32')
        nodata = ds.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= ~np.isclose(arr, np.float32(nodata))
        resolved_negative_only = negative_only
        if resolved_negative_only is None:
            resolved_negative_only = raster_value_semantics(ds.tags()) not in {"absolute_elevation"}
        if resolved_negative_only:
            valid &= arr < 0.0
        rows, cols = np.nonzero(valid)
        n_total = int(rows.size)
        if n_total == 0:
            out_path.write_text('x,y,depth_m,source\n', encoding='utf-8')
            return {'path': str(out_path), 'count': 0, 'stride': 1, 'source_raster': str(src_path)}

        stride = max(1, int(math.ceil(math.sqrt(n_total / max(1, int(max_points))))))
        if stride > 1:
            take = ((rows % stride) == 0) & ((cols % stride) == 0)
            rows = rows[take]
            cols = cols[take]
        vals = arr[rows, cols].astype(float)
        tf = ds.transform
        try:
            a = float(getattr(tf, 'a'))
            c = float(getattr(tf, 'c'))
            e = float(getattr(tf, 'e'))
            f = float(getattr(tf, 'f'))
            b = float(getattr(tf, 'b', 0.0) or 0.0)
            d = float(getattr(tf, 'd', 0.0) or 0.0)
        except Exception:
            log.debug("export_raster_points_to_csv: suppressed exception", exc_info=True)
            tf_vals = tuple(tf)[:6]
            if len(tf_vals) < 6:
                raise
            a, b, c, d, e, f = [float(v) for v in tf_vals]
        xs = c + a * (cols.astype(float) + 0.5) + b * (rows.astype(float) + 0.5)
        ys = f + d * (cols.astype(float) + 0.5) + e * (rows.astype(float) + 0.5)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if source_vdatum and target_vdatum and str(source_vdatum) != str(target_vdatum):
            vtx = Transformer.from_crs(str(source_vdatum), str(target_vdatum), always_xy=True)
            txs, tys, tzs = vtx.transform(xs, ys, vals)
            txs = np.asarray(txs, dtype=float)
            tys = np.asarray(tys, dtype=float)
            tzs = np.asarray(tzs, dtype=float)
            if (not np.all(np.isfinite(tzs))) or tzs.shape != np.asarray(vals, dtype=float).shape:
                raise RuntimeError(
                    f"Vertical transform produced invalid coordinates for {src_path}: {source_vdatum} -> {target_vdatum}"
                )
            xs, ys, vals = txs, tys, tzs
            if resolved_negative_only:
                keep = np.asarray(vals, dtype=float) < 0.0
                xs = np.asarray(xs, dtype=float)[keep]
                ys = np.asarray(ys, dtype=float)[keep]
                vals = np.asarray(vals, dtype=float)[keep]
        if out_crs and ds.crs and str(ds.crs) != str(out_crs):
            transformer = Transformer.from_crs(ds.crs, out_crs, always_xy=True)
            xs, ys = transformer.transform(xs, ys)
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)

    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'depth_m', 'source'])
        for x, y, z in zip(xs, ys, vals):
            writer.writerow([f'{float(x):.10f}', f'{float(y):.10f}', f'{float(z):.6f}', source_tag])

    info = {
        'path': str(out_path),
        'count': int(len(vals)),
        'stride': int(stride),
        'source_raster': str(src_path),
        'out_crs': str(out_crs),
        'negative_only': bool(resolved_negative_only),
    }
    log.info('[AUTHORITATIVE] Exported %d raster-derived support points to %s (stride=%d, negative_only=%s)', int(len(vals)), out_path, int(stride), bool(resolved_negative_only))
    return info


def prepare_authoritative_sdb_training_points(
    auth_raster: Path | str,
    out_csv: Path | str,
    *,
    out_crs: str,
    source_vdatum: str,
    target_vdatum: str,
    max_points: int = 250000,
    negative_only: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log = logger or logging.getLogger(__name__)
    auth_path = Path(auth_raster)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    horiz_epsg = _horizontal_epsg(auth_path)
    if horiz_epsg is None:
        raise RuntimeError(f'Could not determine horizontal EPSG for authoritative raster: {auth_path}')

    src_vertical_epsg = _vertical_target_from_compound(source_vdatum, fallback_epsg=5703)
    dst_vertical_epsg = _vertical_target_from_compound(target_vdatum, fallback_epsg=5714)
    src_compound = f'epsg:{horiz_epsg}+{src_vertical_epsg}'
    dst_compound = f'epsg:{horiz_epsg}+{dst_vertical_epsg}'

    info = export_raster_points_to_csv(
        auth_path,
        out_csv,
        out_crs=out_crs,
        max_points=int(max_points),
        negative_only=bool(negative_only),
        source_tag='authoritative_base',
        source_vdatum=src_compound,
        target_vdatum=dst_compound,
        logger=log,
    )
    info['converted_raster'] = None
    info['source_vdatum'] = src_compound
    info['target_vdatum'] = dst_compound
    return info


def prepare_authoritative_river_soundings_points(
    auth_raster: Path | str,
    out_csv: Path | str,
    *,
    out_crs: str,
    max_points: int = 250000,
    negative_only: Optional[bool] = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Export authoritative-base support as river sounding-style XYZ support.

    River-guidance support is a bed-elevation support product, not an optical
    depth-only training set, so inland positive NAVD88 values are valid by
    default. Callers may still force negative_only=True for explicit depth-mode
    exports, but the default for river support is role-aware bed-elevation use.
    """
    info = export_raster_points_to_csv(
        auth_raster,
        out_csv,
        out_crs=out_crs,
        max_points=int(max_points),
        negative_only=negative_only,
        source_tag='authoritative_base',
        logger=logger,
    )
    info['target_role'] = 'river_soundings'
    info['negative_only_resolved'] = bool(info.get('negative_only', False)) if 'negative_only' in info else bool(negative_only)
    return info
