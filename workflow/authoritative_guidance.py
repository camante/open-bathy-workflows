
from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import geopandas as gpd
import fiona

import numpy as np
import rasterio
from rasterio import features
from pyproj import Transformer

from nodata_utils import nodata_mask, sanitize_array, valid_mask, resolve_reproject_nodata_value, resolve_nodata_value
from sign_semantics import raster_value_semantics
from authoritative_river_roles import build_authoritative_river_role_arrays, role_to_code
from raster_contract import validate_gdal_output
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
    resampling_method: str = "bilinear",
) -> Dict[str, Any]:
    log = logger or logging.getLogger(__name__)
    src_path = Path(source_raster)
    out_path = Path(out_raster)
    transform, width, height = _grid_from_aoi(str(aoi), str(dst_crs), float(res_m))

    with rasterio.open(src_path) as src:
        src_nodata = resolve_reproject_nodata_value(src.nodata, default=-9999.0)
        out_nodata = resolve_nodata_value(src.nodata, default=-9999.0)
        dst = np.full((height, width), np.nan, dtype='float32')
        _resampling = rasterio.enums.Resampling.nearest if str(resampling_method).lower() in {'near', 'nearest'} else rasterio.enums.Resampling.bilinear
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=src_nodata,
            dst_nodata=np.float32(np.nan),
            resampling=_resampling,
        )
        # Sanitize bilinear contamination near nodata boundaries
        _contam = ~np.isfinite(dst)
        if np.isfinite(src_nodata):
            nd = float(src_nodata)
            if nd < -100:
                _contam |= (dst < nd * 0.5)
            elif nd > 100:
                _contam |= (dst > nd * 0.5)
        _contam |= (dst < -1000.0) | (dst > 10000.0)
        dst[_contam] = np.nan
        dst = sanitize_array(dst, src_nodata, dtype='float32')
        profile = src.profile.copy()

    fallback_filled_cells = 0
    if fallback_raster is not None and Path(fallback_raster).exists():
        with rasterio.open(str(fallback_raster)) as fb:
            fb_nodata = resolve_reproject_nodata_value(fb.nodata, default=-9999.0)
            fill = np.full((height, width), np.nan, dtype='float32')
            rasterio.warp.reproject(
                source=rasterio.band(fb, 1),
                destination=fill,
                src_transform=fb.transform,
                src_crs=fb.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                src_nodata=fb_nodata,
                dst_nodata=np.float32(np.nan),
                resampling=_resampling,
            )
            _contam_fb = ~np.isfinite(fill)
            if np.isfinite(fb_nodata):
                nd = float(fb_nodata)
                if nd < -100:
                    _contam_fb |= (fill < nd * 0.5)
                elif nd > 100:
                    _contam_fb |= (fill > nd * 0.5)
            _contam_fb |= (fill < -1000.0) | (fill > 10000.0)
            fill[_contam_fb] = np.nan
            fill = sanitize_array(fill, fb_nodata, dtype='float32')
            missing = ~np.isfinite(dst)
            valid_fill = np.isfinite(fill)
            take = missing & valid_fill
            if np.any(take):
                fallback_filled_cells = int(np.count_nonzero(take))
                dst[take] = fill[take].astype('float32')
                log.info('[AUTHORITATIVE] Filled %d projected river DEM cells from fallback DEM gaps.', fallback_filled_cells)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.pop('blockxsize', None)
    profile.pop('blockysize', None)
    profile.update(
        driver='GTiff',
        dtype='float32',
        count=1,
        crs=dst_crs,
        transform=transform,
        width=width,
        height=height,
        nodata=float(out_nodata),
        compress='DEFLATE',
        tiled=bool(width >= 16 and height >= 16),
    )
    write_arr = np.asarray(dst, dtype='float32').copy()
    write_arr[~np.isfinite(write_arr)] = float(out_nodata)
    with rasterio.open(out_path, 'w', **profile) as dst_ds:
        dst_ds.write(write_arr, 1)
    validate_gdal_output(out_path, operation='authoritative_guidance.build_projected_authoritative_raster', expected_crs=dst_crs, expected_nodata=float(out_nodata), expected_dtype='float32', min_allowed=-1000.0, max_allowed=10000.0, source_path=src_path)

    valid = np.isfinite(dst)
    return {
        'path': str(out_path),
        'width': int(width),
        'height': int(height),
        'finite_cells': int(np.count_nonzero(valid)),
        'nodata_cells': int(np.count_nonzero(~valid)),
        'fallback_filled_cells': int(fallback_filled_cells),
    }




def build_projected_authoritative_sampling_raster(
    source_raster: Path | str,
    out_raster: Path | str,
    *,
    aoi: str,
    dst_crs: str,
    res_m: float,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    info = build_projected_authoritative_raster(
        source_raster,
        out_raster,
        aoi=aoi,
        dst_crs=dst_crs,
        res_m=res_m,
        fallback_raster=None,
        logger=logger,
        resampling_method='near',
    )
    info['sampling_contract'] = 'nearest_no_fallback'
    return info


def _load_support_coverage_geometries(support_coverage_path: Path | str, target_crs: Any | None = None) -> tuple[list[Any], Any | None]:
    path = Path(support_coverage_path)
    if not path.exists():
        raise RuntimeError(f"authoritative_support_coverage_missing:{path}")
    try:
        layers = list(fiona.listlayers(path))
    except Exception:
        layers = []
    gdfs: list[gpd.GeoDataFrame] = []
    if layers:
        for layer in layers:
            try:
                gdf = gpd.read_file(path, layer=layer)
            except Exception:
                continue
            if gdf is None or gdf.empty or 'geometry' not in gdf.columns:
                continue
            keep = gdf.geometry.notna() & gdf.geometry.geom_type.astype(str).isin(['Polygon', 'MultiPolygon'])
            if keep.any():
                gdfs.append(gdf.loc[keep, ['geometry']].copy())
    else:
        gdf = gpd.read_file(path)
        if gdf is not None and not gdf.empty and 'geometry' in gdf.columns:
            keep = gdf.geometry.notna() & gdf.geometry.geom_type.astype(str).isin(['Polygon', 'MultiPolygon'])
            if keep.any():
                gdfs.append(gdf.loc[keep, ['geometry']].copy())
    if not gdfs:
        raise RuntimeError(f"authoritative_support_coverage_no_polygon_geometry:{path}")

    coverage = gdfs[0]
    for extra in gdfs[1:]:
        if target_crs is not None and extra.crs is not None and str(extra.crs) != str(target_crs):
            extra = extra.to_crs(target_crs)
        elif coverage.crs is not None and extra.crs is not None and str(extra.crs) != str(coverage.crs):
            extra = extra.to_crs(coverage.crs)
        coverage = pd.concat([coverage, extra], ignore_index=True)
    coverage = gpd.GeoDataFrame(coverage, geometry='geometry', crs=getattr(coverage, 'crs', None))
    if target_crs is not None:
        if coverage.crs is None:
            raise RuntimeError('authoritative_support_coverage_missing_crs')
        if str(coverage.crs) != str(target_crs):
            coverage = coverage.to_crs(target_crs)
    geoms = [geom for geom in coverage.geometry if geom is not None and not geom.is_empty]
    return geoms, coverage.crs


def build_projected_measured_only_authoritative_sampling_raster(
    source_raster: Path | str,
    support_coverage_path: Path | str,
    out_raster: Path | str,
    *,
    aoi: str,
    dst_crs: str,
    res_m: float,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log = logger or logging.getLogger(__name__)
    info = build_projected_authoritative_sampling_raster(
        source_raster,
        out_raster,
        aoi=aoi,
        dst_crs=dst_crs,
        res_m=res_m,
        logger=log,
    )
    out_path = Path(out_raster)
    with rasterio.open(out_path, 'r+') as ds:
        geoms, _ = _load_support_coverage_geometries(support_coverage_path, target_crs=ds.crs)
        mask = features.rasterize(((geom, 1) for geom in geoms), out_shape=(ds.height, ds.width), transform=ds.transform, fill=0, dtype='uint8')
        arr = ds.read(1).astype('float32')
        nodata = ds.nodata if ds.nodata is not None else -9999.0
        arr[mask == 0] = float(nodata)
        ds.write(arr, 1)
    info['sampling_contract'] = 'measured_only_projected_masked_by_support_coverage'
    info['support_coverage_path'] = str(support_coverage_path)
    info['supported_cells'] = int((mask == 1).sum())
    info['masked_out_cells'] = int(mask.size - (mask == 1).sum())
    return info

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
    negative_only: Optional[bool] = False,
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
        arr = sanitize_array(ds.read(1), ds.nodata, dtype='float32')
        valid = np.isfinite(arr)
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




def export_raster_points_to_roleaware_csv(
    source_raster: Path | str,
    out_csv: Path | str,
    *,
    out_crs: str,
    max_points: int = 250000,
    negative_only: Optional[bool] = False,
    source_tag: str = 'authoritative_base',
    role_arrays: Optional[Dict[str, np.ndarray | float]] = None,
    role_contract_path: Optional[Path | str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log = logger or logging.getLogger(__name__)
    src_path = Path(source_raster)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as ds:
        arr = sanitize_array(ds.read(1), ds.nodata, dtype='float32')
        valid = np.isfinite(arr)
        resolved_negative_only = negative_only
        if resolved_negative_only is None:
            resolved_negative_only = raster_value_semantics(ds.tags()) not in {"absolute_elevation"}
        if resolved_negative_only:
            valid &= arr < 0.0
        rows, cols = np.nonzero(valid)
        n_total = int(rows.size)
        if n_total == 0:
            header = [
                'x', 'y', 'depth_m', 'source', 'authoritative_role', 'role_confidence',
                'distance_to_bank_m', 'component_half_width_est_m', 'normalized_channel_position',
                'inside_channel_mask', 'inside_river_guidance_domain', 'inside_mainstem_mask', 'inside_estuary_clip',
            ]
            out_path.write_text(','.join(header) + '\n', encoding='utf-8')
            info = {'path': str(out_path), 'count': 0, 'stride': 1, 'source_raster': str(src_path)}
            if role_contract_path is not None:
                Path(role_contract_path).write_text(json.dumps({'path': str(out_path), 'count': 0, 'source_raster': str(src_path)}, indent=2), encoding='utf-8')
                info['role_contract'] = str(role_contract_path)
            return info

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
            tf_vals = tuple(tf)[:6]
            if len(tf_vals) < 6:
                raise
            a, b, c, d, e, f = [float(v) for v in tf_vals]
        xs = c + a * (cols.astype(float) + 0.5) + b * (rows.astype(float) + 0.5)
        ys = f + d * (cols.astype(float) + 0.5) + e * (rows.astype(float) + 0.5)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if out_crs and ds.crs and str(ds.crs) != str(out_crs):
            transformer = Transformer.from_crs(ds.crs, out_crs, always_xy=True)
            xs, ys = transformer.transform(xs, ys)
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)

        meta = {}
        if role_arrays:
            for key, arr_like in role_arrays.items():
                if isinstance(arr_like, np.ndarray) and arr_like.ndim == 2 and arr_like.shape == arr.shape:
                    meta[key] = arr_like[rows, cols]

    df = pd.DataFrame({
        'x': xs,
        'y': ys,
        'depth_m': np.asarray(vals, dtype=float),
        'source': source_tag,
        'authoritative_role': np.asarray(meta.get('role', np.full(len(vals), 'authoritative_overbank_or_ambiguous', dtype=object)), dtype=object),
        'role_confidence': np.asarray(meta.get('role_confidence', np.zeros(len(vals), dtype=float)), dtype=float),
        'distance_to_bank_m': np.asarray(meta.get('distance_to_bank_m', np.full(len(vals), np.nan, dtype=float)), dtype=float),
        'component_half_width_est_m': np.asarray(meta.get('component_half_width_est_m', np.full(len(vals), np.nan, dtype=float)), dtype=float),
        'normalized_channel_position': np.asarray(meta.get('normalized_channel_position', np.full(len(vals), np.nan, dtype=float)), dtype=float),
        'inside_channel_mask': np.asarray(meta.get('inside_channel_mask', np.zeros(len(vals), dtype=np.uint8)), dtype=np.uint8),
        'inside_river_guidance_domain': np.asarray(meta.get('inside_river_guidance_domain', np.zeros(len(vals), dtype=np.uint8)), dtype=np.uint8),
        'inside_mainstem_mask': np.asarray(meta.get('inside_mainstem_mask', np.zeros(len(vals), dtype=np.uint8)), dtype=np.uint8),
        'inside_estuary_clip': np.asarray(meta.get('inside_estuary_clip', np.zeros(len(vals), dtype=np.uint8)), dtype=np.uint8),
    })

    df['x'] = df['x'].map(lambda v: f'{float(v):.10f}')
    df['y'] = df['y'].map(lambda v: f'{float(v):.10f}')
    df['depth_m'] = df['depth_m'].map(lambda v: f'{float(v):.6f}')
    df['role_confidence'] = df['role_confidence'].map(lambda v: f'{float(v):.4f}')
    df['distance_to_bank_m'] = df['distance_to_bank_m'].map(lambda v: '' if not np.isfinite(v) else f'{float(v):.3f}')
    df['component_half_width_est_m'] = df['component_half_width_est_m'].map(lambda v: '' if not np.isfinite(v) else f'{float(v):.3f}')
    df['normalized_channel_position'] = df['normalized_channel_position'].map(lambda v: '' if not np.isfinite(v) else f'{float(v):.4f}')
    for col in ('inside_channel_mask', 'inside_river_guidance_domain', 'inside_mainstem_mask', 'inside_estuary_clip'):
        df[col] = df[col].astype(int)
    df.to_csv(out_path, index=False)

    role_counts = {}
    try:
        role_counts = pd.Series(np.asarray(meta.get('role', []), dtype=object)).value_counts(dropna=False).to_dict()
    except Exception:
        role_counts = {}

    info = {
        'path': str(out_path),
        'count': int(len(df)),
        'stride': int(stride),
        'source_raster': str(src_path),
        'out_crs': str(out_crs),
        'negative_only': bool(resolved_negative_only),
        'role_counts': {str(k): int(v) for k, v in role_counts.items()},
    }
    if role_arrays:
        info['role_classifier'] = {
            'bank_margin_m': float(role_arrays.get('bank_margin_m', np.nan)) if isinstance(role_arrays.get('bank_margin_m', np.nan), (int, float)) else np.nan,
            'effective_bank_margin_m': float(role_arrays.get('effective_bank_margin_m', np.nan)) if isinstance(role_arrays.get('effective_bank_margin_m', np.nan), (int, float)) else np.nan,
            'core_min_m': float(role_arrays.get('core_min_m', np.nan)) if isinstance(role_arrays.get('core_min_m', np.nan), (int, float)) else np.nan,
            'pixel_size_m': float(role_arrays.get('pixel_size_m', np.nan)) if isinstance(role_arrays.get('pixel_size_m', np.nan), (int, float)) else np.nan,
        }
    if role_contract_path is not None:
        contract = {
            'source_raster': str(src_path),
            'out_csv': str(out_path),
            'out_crs': str(out_crs),
            'count': int(len(df)),
            'stride': int(stride),
            'negative_only': bool(resolved_negative_only),
            'role_counts': info['role_counts'],
            'classifier': info.get('role_classifier', {}),
            'columns': list(df.columns),
        }
        Path(role_contract_path).write_text(json.dumps(contract, indent=2), encoding='utf-8')
        info['role_contract'] = str(role_contract_path)
    log.info('[AUTHORITATIVE] Exported %d role-aware river support points to %s (stride=%d)', int(len(df)), out_path, int(stride))
    return info



def build_sdb_authoritative_support_products(
    auth_raster: Path | str,
    domain_mask: Path | str,
    out_mask: Path | str,
    out_values: Path | str,
    out_points: Path | str,
    out_contract: Path | str,
    *,
    out_crs: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Build explicit SDB authoritative-support products on the SDB candidate-domain grid.

    Semantics:
    - out_mask: uint8 support mask where supported=1, unsupported=0
    - out_values: authoritative value at supported cells, nodata elsewhere
    - out_points: CSV export of supported authoritative values for SDB training/validation
    """
    log = logger or logging.getLogger(__name__)
    auth_path = Path(auth_raster)
    domain_path = Path(domain_mask)
    out_mask = Path(out_mask)
    out_values = Path(out_values)
    out_points = Path(out_points)
    out_contract = Path(out_contract)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    out_values.parent.mkdir(parents=True, exist_ok=True)
    out_points.parent.mkdir(parents=True, exist_ok=True)
    out_contract.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(domain_path) as dom_ds:
        dom = sanitize_array(dom_ds.read(1), dom_ds.nodata, dtype='float32')
        candidate = np.isfinite(dom) & (dom <= 0.5)
        dst_shape = (dom_ds.height, dom_ds.width)
        dst_transform = dom_ds.transform
        dst_crs = dom_ds.crs
        dom_profile = dom_ds.profile.copy()

    with rasterio.open(auth_path) as auth_ds:
        auth = sanitize_array(auth_ds.read(1), auth_ds.nodata, dtype='float32')
        same_grid = (
            auth_ds.crs == dst_crs
            and auth_ds.transform == dst_transform
            and auth_ds.width == dst_shape[1]
            and auth_ds.height == dst_shape[0]
        )
        if same_grid:
            auth_aligned = auth
        else:
            auth_aligned = np.full(dst_shape, np.nan, dtype='float32')
            rasterio.warp.reproject(
                source=auth,
                destination=auth_aligned,
                src_transform=auth_ds.transform,
                src_crs=auth_ds.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                src_nodata=np.float32(np.nan),
                dst_nodata=np.float32(np.nan),
                resampling=rasterio.enums.Resampling.nearest,
            )
            auth_aligned = sanitize_array(auth_aligned, np.nan, dtype='float32')

    support = candidate & np.isfinite(auth_aligned)
    values = np.full(dst_shape, np.nan, dtype='float32')
    values[support] = auth_aligned[support].astype('float32')

    mask_profile = dom_profile.copy()
    mask_profile.pop('blockxsize', None)
    mask_profile.pop('blockysize', None)
    mask_profile.update(driver='GTiff', dtype='uint8', count=1, nodata=0, compress='DEFLATE', tiled=bool(dom_profile.get('width',0) >= 16 and dom_profile.get('height',0) >= 16))
    with rasterio.open(out_mask, 'w', **mask_profile) as ds:
        ds.write(np.where(support, 1, 0).astype('uint8'), 1)

    val_profile = dom_profile.copy()
    val_profile.pop('blockxsize', None)
    val_profile.pop('blockysize', None)
    val_profile.update(driver='GTiff', dtype='float32', count=1, nodata=-9999.0, compress='DEFLATE', tiled=bool(dom_profile.get('width',0) >= 16 and dom_profile.get('height',0) >= 16))
    write_vals = values.copy()
    write_vals[~np.isfinite(write_vals)] = -9999.0
    with rasterio.open(out_values, 'w', **val_profile) as ds:
        ds.write(write_vals, 1)

    point_info = export_raster_points_to_csv(
        out_values,
        out_points,
        out_crs=out_crs,
        max_points=250000,
        negative_only=True,
        source_tag='authoritative_sdb_support',
        logger=log,
    )
    contract = {
        'authoritative_raster': str(auth_path),
        'sdb_candidate_domain_mask': str(domain_path),
        'support_mask': str(out_mask),
        'support_values': str(out_values),
        'support_points': str(out_points),
        'candidate_pixels': int(np.count_nonzero(candidate)),
        'support_pixels': int(np.count_nonzero(support)),
        'unsupported_candidate_pixels': int(np.count_nonzero(candidate & ~support)),
        'grid': {
            'width': int(dst_shape[1]),
            'height': int(dst_shape[0]),
            'crs': str(dst_crs),
        },
        'aligned_from_authoritative_grid': bool(not same_grid),
        'points': point_info,
    }
    out_contract.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    return contract


def _write_roleaware_sidecar_rasters(
    auth_raster: Path | str,
    out_csv: Path | str,
    *,
    role_arrays: Dict[str, Any],
) -> Dict[str, str]:
    auth_path = Path(auth_raster)
    out_path = Path(out_csv)
    base = out_path.with_suffix("")
    out_map = {
        "role_code_raster": base.parent / f"{base.name}_role_code.tif",
        "role_confidence_raster": base.parent / f"{base.name}_role_confidence.tif",
        "distance_to_bank_raster": base.parent / f"{base.name}_distance_to_bank_m.tif",
        "normalized_channel_position_raster": base.parent / f"{base.name}_normalized_channel_position.tif",
    }
    with rasterio.open(auth_path) as src:
        profile = src.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(compress="deflate")
        role_code = np.asarray([role_to_code(v) for v in np.ravel(role_arrays.get("role", np.full((src.height, src.width), "authoritative_overbank_or_ambiguous", dtype=object)))], dtype=np.uint8).reshape((src.height, src.width))
        conf = np.asarray(role_arrays.get("role_confidence", np.zeros((src.height, src.width), dtype=np.float32)), dtype=np.float32)
        dist = np.asarray(role_arrays.get("distance_to_bank_m", np.full((src.height, src.width), np.nan, dtype=np.float32)), dtype=np.float32)
        norm = np.asarray(role_arrays.get("normalized_channel_position", np.full((src.height, src.width), np.nan, dtype=np.float32)), dtype=np.float32)

        role_profile = profile.copy()
        role_profile.update(count=1, dtype="uint8", nodata=0)
        with rasterio.open(out_map["role_code_raster"], "w", **role_profile) as ds:
            ds.write(role_code, 1)
            ds.update_tags(ROLE_CODES=json.dumps({"0": "authoritative_overbank_or_ambiguous", "1": "authoritative_bank_margin", "2": "authoritative_bed_inner", "3": "authoritative_bed_core"}))

        float_profile = profile.copy()
        float_profile.update(count=1, dtype="float32", nodata=-9999.0)
        for key, arr in (("role_confidence_raster", conf), ("distance_to_bank_raster", dist), ("normalized_channel_position_raster", norm)):
            out_arr = np.asarray(arr, dtype=np.float32)
            out_arr = np.where(np.isfinite(out_arr), out_arr, np.float32(-9999.0)).astype(np.float32)
            with rasterio.open(out_map[key], "w", **float_profile) as ds:
                ds.write(out_arr, 1)
    return {k: str(v) for k, v in out_map.items()}


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
    river_channel_mask: Optional[Path | str] = None,
    river_guidance_domain_mask: Optional[Path | str] = None,
    estuary_clip_mask: Optional[Path | str] = None,
    mainstem_mask: Optional[Path | str] = None,
    bank_margin_m: float = 3.0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Export authoritative-base support as role-aware river support points.

    This broad authoritative-raster export preserves absolute elevations by
    default and now carries explicit bed/bank/ambiguous role columns so the
    first river-support handoff no longer flattens all authoritative values
    into a single generic source.
    """
    auth_path = Path(auth_raster)
    out_path = Path(out_csv)
    role_contract = out_path.with_suffix('.role_contract.json')
    with rasterio.open(auth_path) as ds:
        template_profile = {
            'height': ds.height,
            'width': ds.width,
            'transform': ds.transform,
            'crs': ds.crs,
        }
    role_arrays = build_authoritative_river_role_arrays(
        template_profile=template_profile,
        river_channel_mask_path=river_channel_mask,
        river_guidance_domain_mask_path=river_guidance_domain_mask,
        estuary_clip_mask_path=estuary_clip_mask,
        mainstem_mask_path=mainstem_mask,
        bank_margin_m=float(bank_margin_m or 0.0),
    )
    sidecar_rasters = _write_roleaware_sidecar_rasters(
        auth_path,
        out_path,
        role_arrays=role_arrays,
    )
    info = export_raster_points_to_roleaware_csv(
        auth_path,
        out_path,
        out_crs=out_crs,
        max_points=int(max_points),
        negative_only=negative_only,
        source_tag='authoritative_base',
        role_arrays=role_arrays,
        role_contract_path=role_contract,
        logger=logger,
    )
    if role_contract.exists():
        try:
            contract = json.loads(role_contract.read_text(encoding='utf-8'))
        except Exception:
            contract = {}
        contract.setdefault('artifacts', {}).update(sidecar_rasters)
        role_contract.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    info.update(sidecar_rasters)
    info['target_role'] = 'river_soundings'
    info['negative_only_resolved'] = bool(info.get('negative_only', False)) if 'negative_only' in info else bool(negative_only)
    return info
