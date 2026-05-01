from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError
from scipy.spatial import cKDTree

from pipeline.river_workflow.river_workflow_context import RiverWorkflowContext
from pipeline.river_workflow.river_workflow_stage_authoritative import AuthoritativeStageResult
from pipeline.river_workflow.river_workflow_stage_centerline import CenterlineStageResult
from pipeline.river_workflow.river_workflow_validation import validate_wse_proxy_points_gpkg
from pipeline.river_workflow.river_workflow_wse_steps import (
    WSE_PROFILE_CONTRACT_VERSION,
    build_wse_pre_smooth_artifact,
    build_wse_proxy_final_artifact,
    build_wse_support_artifact,
    build_wse_trend_artifact,
    write_wse_stage_contract,
)


@dataclass(frozen=True)
class WSEProxyStageResult:
    centerline_wse_proxy_points_path: Path
    record_count: int
    finite_wse_count: int
    science_summary: dict[str, Any] | None = None
    centerline_wse_support_points_path: Path | None = None
    centerline_wse_trend_points_path: Path | None = None
    centerline_wse_pre_smooth_points_path: Path | None = None
    wse_current_path_audit_path: Path | None = None
    wse_direction_audit_path: Path | None = None
    wse_proxy_audit_path: Path | None = None
    wse_artifact_manifest_path: Path | None = None
    wse_stage_contract_path: Path | None = None


_GROUP_FIELDS = ('component_id', 'levelpath_id', 'reach_id', 'source_reach_key')
# WSE is a canonical longitudinal profile, not a per-reach local interpolation.
# Fit WSE on broad canonical component/levelpath groups so sparse bank-edge
# support defines one parent longitudinal profile instead of failing each short reach.
_WSE_PROFILE_GROUP_FIELDS = ('component_id', 'levelpath_id')

# Active WSE contract: one canonical bank-edge support profile, not a local
# interpolation/repair chain. Keep the stage one-path so noisy bank-edge
# observations refine a simple downstream-decreasing WSE reference surface, while
# bed structure remains the responsibility of modeled offsets and the backbone.
_WSE_PROFILE_CONTRACT_VERSION = WSE_PROFILE_CONTRACT_VERSION
_WSE_REQUIRED_OUTPUT_COLUMNS = (
    'component_id', 'levelpath_id', 'reach_id', 'source_reach_key',
    'station_m', 'station_downstream_m', 'wse_proxy_z_m',
    'wse_profile_z_m', 'wse_support_count', 'wse_support_distance_m',
    'wse_profile_method', 'wse_direction', 'wse_confidence',
)
_WSE_ALLOWED_INPUTS = (
    'canonical_centerline_points',
    'canonical_linear_polygons_or_channel_boundary',
    'canonical_measured_authoritative_raster',
    'canonical_solve_grid_metadata',
)
_WSE_FORBIDDEN_SUPPORT_SOURCES = (
    'local_bed_floor_as_wse',
    'broad_centerline_annulus_upland_topography',
    'aoi_local_products',
    'final_dem_or_conditioned_surface',
    'interpolated_baseline_as_wse_evidence',
)

_WSE_OUTER_RADIUS_M = 80.0
_WSE_QUANTILE = 0.25
_WSE_BANK_EDGE_DISTANCE_M = 18.0
_WSE_BANK_EDGE_SAMPLE_SPACING_M = 10.0
_WSE_BANK_SUPPORT_MIN_RADIUS_M = 120.0
_WSE_BANK_SUPPORT_WIDTH_RADIUS_FACTOR = 0.75
_WSE_FLATNESS_DOMINANT_FRACTION_THRESHOLD = 0.65
_WSE_FLATNESS_MIN_FINITE_COUNT = 20
_WSE_MEANINGFUL_SUPPORT_RANGE_M = 0.05
_WSE_ROUND_DECIMALS = 2
_WSE_DIRECTION_MIN_SUPPORT_DROP_M = 0.05
_WSE_MONOTONE_TOLERANCE_M = 1.0e-6
_WSE_TREND_MAX_FIT_POINTS = 96
_WSE_TREND_MIN_DROP_M = 0.05
_WSE_PROFILE_MIN_BIN_WIDTH_M = 50.0
_WSE_PROFILE_TARGET_BIN_COUNT = 80
_WSE_PROFILE_RESIDUAL_BOUND_M = 2.0
_WSE_PROFILE_EDGE_FRACTION = 0.15
_WSE_PROFILE_MIN_SUPPORT_COUNT = 2



def _finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {'count': 0, 'min': None, 'max': None, 'median': None}
    return {
        'count': int(vals.size),
        'min': float(np.nanmin(vals)),
        'max': float(np.nanmax(vals)),
        'median': float(np.nanmedian(vals)),
    }



def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _dominant_flatness_summary(values: Any) -> dict[str, Any]:
    vals = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float)
    vals = vals[np.isfinite(vals)]
    summary: dict[str, Any] = {
        'finite_count': int(vals.size),
        'min_m': None,
        'max_m': None,
        'range_m': None,
        'dominant_rounded_value_m': None,
        'dominant_rounded_count': 0,
        'dominant_rounded_fraction': None,
        'flatness_fail': False,
        'flatness_threshold': float(_WSE_FLATNESS_DOMINANT_FRACTION_THRESHOLD),
        'flatness_min_finite_count': int(_WSE_FLATNESS_MIN_FINITE_COUNT),
    }
    if vals.size == 0:
        return summary
    min_v = float(np.nanmin(vals))
    max_v = float(np.nanmax(vals))
    summary['min_m'] = min_v
    summary['max_m'] = max_v
    summary['range_m'] = float(max_v - min_v)
    rounded = np.round(vals, _WSE_ROUND_DECIMALS)
    unique, counts = np.unique(rounded, return_counts=True)
    if counts.size:
        idx = int(np.argmax(counts))
        dominant_count = int(counts[idx])
        fraction = float(dominant_count / vals.size)
        summary['dominant_rounded_value_m'] = float(unique[idx])
        summary['dominant_rounded_count'] = dominant_count
        summary['dominant_rounded_fraction'] = fraction
        summary['flatness_fail'] = bool(
            vals.size >= _WSE_FLATNESS_MIN_FINITE_COUNT
            and fraction >= _WSE_FLATNESS_DOMINANT_FRACTION_THRESHOLD
        )
    return summary


def _support_downstream_trend_summary(
    station_values: np.ndarray | None,
    support_values: np.ndarray,
) -> dict[str, Any]:
    """Summarize whether WSE support has a defensible downstream drop.

    The WSE flatness guard should fail only when monotone/profile construction
    destroys a coherent downstream water-surface signal. Low-gradient or noisy
    bank-edge support can have centimeter-scale variation but no trustworthy
    long-profile drop; in that case a nearly flat WSE proxy is not the first
    wrong artifact. The bed/backbone stages remain responsible for channel-bed
    structure.
    """
    sta = np.asarray(station_values if station_values is not None else [], dtype=float)
    vals = np.asarray(support_values, dtype=float)
    if sta.shape != vals.shape:
        sta = np.full(vals.shape, np.nan, dtype=float)
    finite = np.isfinite(sta) & np.isfinite(vals)
    summary: dict[str, Any] = {
        'finite_count': int(np.count_nonzero(finite)),
        'station_range_m': None,
        'left_edge_median_m': None,
        'right_edge_median_m': None,
        'edge_drop_m': None,
        'median_pairwise_slope_m_per_m': None,
        'has_defensible_downstream_drop': False,
        'min_drop_m': float(_WSE_DIRECTION_MIN_SUPPORT_DROP_M),
    }
    if int(summary['finite_count']) < 2:
        return summary

    x = sta[finite]
    y = vals[finite]
    order = np.argsort(x, kind='mergesort')
    x = x[order]
    y = y[order]
    station_range = float(np.nanmax(x) - np.nanmin(x))
    summary['station_range_m'] = station_range if np.isfinite(station_range) else None
    if not np.isfinite(station_range) or station_range <= 1.0e-9:
        return summary

    edge_n = max(1, min(5, int(y.size // 4) if y.size >= 4 else 1))
    left = float(np.nanmedian(y[:edge_n]))
    right = float(np.nanmedian(y[-edge_n:]))
    edge_drop = float(left - right)
    summary['left_edge_median_m'] = left
    summary['right_edge_median_m'] = right
    summary['edge_drop_m'] = edge_drop

    slopes: list[float] = []
    fit_n = min(int(y.size), 64)
    if y.size > fit_n:
        keep = np.linspace(0, y.size - 1, fit_n).round().astype(int)
        fit_x = x[keep]
        fit_y = y[keep]
    else:
        fit_x = x
        fit_y = y
    for i in range(int(fit_x.size) - 1):
        dx = fit_x[i + 1:] - fit_x[i]
        dy = fit_y[i + 1:] - fit_y[i]
        good = np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) > 1.0e-9)
        if np.any(good):
            slopes.extend((dy[good] / dx[good]).astype(float).tolist())
    if slopes:
        slope = float(np.nanmedian(np.asarray(slopes, dtype=float)))
        summary['median_pairwise_slope_m_per_m'] = slope
    else:
        slope = float('nan')

    # In downstream-oriented station coordinates, WSE should be non-increasing.
    # Require both edge drop and a non-positive robust pairwise slope so local
    # noisy bank samples do not make the flatness guard reject a valid low-
    # gradient WSE proxy.
    summary['has_defensible_downstream_drop'] = bool(
        edge_drop >= _WSE_DIRECTION_MIN_SUPPORT_DROP_M
        and (not np.isfinite(slope) or slope <= 1.0e-12)
    )
    return summary


def _classify_wse_group_flatness(
    *,
    support_values: np.ndarray,
    raw_interpolated: np.ndarray,
    pre_monotone: np.ndarray,
    final_proxy: np.ndarray,
    tail_policy: str,
    station_values: np.ndarray | None = None,
) -> dict[str, Any]:
    support_summary = _dominant_flatness_summary(support_values)
    raw_summary = _dominant_flatness_summary(raw_interpolated)
    pre_summary = _dominant_flatness_summary(pre_monotone)
    final_summary = _dominant_flatness_summary(final_proxy)
    support_trend = _support_downstream_trend_summary(station_values, support_values)
    final_is_flat = bool(final_summary.get('flatness_fail'))
    support_count = int(support_summary.get('finite_count') or 0)
    support_range = support_summary.get('range_m')
    support_has_variation = bool(
        support_count >= 2
        and support_range is not None
        and float(support_range) >= _WSE_MEANINGFUL_SUPPORT_RANGE_M
    )
    support_has_defensible_drop = bool(support_trend.get('has_defensible_downstream_drop'))
    if not final_is_flat:
        status = 'ok_not_flat'
        fail_reason = None
    elif support_has_variation and not support_has_defensible_drop:
        status = 'ok_flat_low_gradient_or_noisy_wse_support'
        fail_reason = None
    elif support_has_variation and not bool(pre_summary.get('flatness_fail')):
        status = 'fail_monotone_flattened_supported_signal'
        fail_reason = 'wse_proxy_monotone_flattened_supported_signal'
    elif support_has_variation and bool(raw_summary.get('flatness_fail')):
        status = 'fail_interpolation_flattened_supported_signal'
        fail_reason = 'wse_proxy_interpolation_flattened_supported_signal'
    elif support_count < 2:
        status = 'fail_flat_proxy_insufficient_wse_support'
        fail_reason = 'wse_proxy_flat_with_insufficient_wse_support'
    elif support_range is not None and float(support_range) < _WSE_MEANINGFUL_SUPPORT_RANGE_M:
        # A flat final proxy is not a construction failure when the measured
        # bank-edge WSE support is itself flat. Carry the supported flat WSE
        # forward and let offset/backbone stages determine bed structure.
        status = 'ok_flat_because_wse_support_is_flat'
        fail_reason = None
    elif tail_policy != 'trend_extrapolated_tails' and bool(raw_summary.get('flatness_fail')):
        status = 'fail_flat_proxy_endpoint_tail_fill'
        fail_reason = 'wse_proxy_flat_after_endpoint_tail_fill'
    else:
        status = 'fail_flat_proxy_unclassified'
        fail_reason = 'wse_proxy_flat_unclassified'
    return {
        'status': status,
        'fail_reason': fail_reason,
        'tail_policy': str(tail_policy),
        'support': support_summary,
        'support_downstream_trend': support_trend,
        'raw_interpolated': raw_summary,
        'pre_monotone': pre_summary,
        'final_proxy': final_summary,
    }


def _metric_xy(gdf: gpd.GeoDataFrame, raster_crs) -> np.ndarray:
    work = gdf
    xs = work.geometry.x.to_numpy(dtype=float)
    ys = work.geometry.y.to_numpy(dtype=float)
    try:
        src = CRS.from_user_input(work.crs) if work.crs is not None else None
        dst = CRS.from_user_input(raster_crs) if raster_crs is not None else None
        if src is not None and dst is not None and src != dst:
            transformer = Transformer.from_crs(src, dst, always_xy=True)
            xs, ys = transformer.transform(xs, ys)
    except (ProjError, ValueError, TypeError):
        pass
    arr = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    return _xy_to_metric(arr, raster_crs)


def _xy_to_metric(xy: np.ndarray, crs_like) -> np.ndarray:
    arr = np.asarray(xy, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        arr = arr.reshape((-1, 2))
    try:
        crs = CRS.from_user_input(crs_like) if crs_like is not None else None
        if crs is not None and crs.is_geographic:
            transformer = Transformer.from_crs(crs, CRS.from_epsg(3857), always_xy=True)
            mx, my = transformer.transform(arr[:, 0], arr[:, 1])
            arr = np.column_stack([np.asarray(mx, dtype=float), np.asarray(my, dtype=float)])
    except (ProjError, ValueError, TypeError):
        pass
    return arr


def _iter_line_parts(geom):
    if geom is None or geom.is_empty:
        return
    geom_type = getattr(geom, 'geom_type', '')
    if geom_type == 'LineString':
        yield geom
    elif geom_type in ('MultiLineString', 'GeometryCollection'):
        for part in geom.geoms:
            yield from _iter_line_parts(part)


def _boundary_sample_xy(polygons: gpd.GeoDataFrame, *, spacing_m: float) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    spacing = max(float(spacing_m), 1.0)
    for geom in polygons.geometry:
        if geom is None or geom.is_empty:
            continue
        for line in _iter_line_parts(geom.boundary):
            length = float(getattr(line, 'length', 0.0) or 0.0)
            if not np.isfinite(length) or length <= 0.0:
                continue
            n = max(2, int(np.ceil(length / spacing)) + 1)
            for distance in np.linspace(0.0, length, n):
                point = line.interpolate(float(distance))
                if point is not None and not point.is_empty:
                    pts.append((float(point.x), float(point.y)))
    if not pts:
        return np.empty((0, 2), dtype=float)
    return np.asarray(pts, dtype=float)


def _read_linear_polygons_for_raster(network_gpkg_path: Path, raster_crs) -> gpd.GeoDataFrame | None:
    path = Path(network_gpkg_path)
    if not path.exists():
        return None
    try:
        layers = gpd.list_layers(path)
        layer_names = set(layers['name'].tolist())
    except (OSError, ValueError, RuntimeError):
        layer_names = set()
    if 'linear_polygons' not in layer_names:
        return None
    polygons = gpd.read_file(path, layer='linear_polygons')
    if polygons is None or polygons.empty or 'geometry' not in polygons.columns:
        return None
    polygons = polygons[polygons.geometry.notnull() & ~polygons.geometry.is_empty].copy()
    if polygons.empty:
        return None
    try:
        dst = CRS.from_user_input(raster_crs) if raster_crs is not None else None
        src = CRS.from_user_input(polygons.crs) if polygons.crs is not None else None
        if src is not None and dst is not None and src != dst:
            polygons = polygons.to_crs(dst)
        elif polygons.crs is None and dst is not None:
            polygons = polygons.set_crs(dst, allow_override=True)
    except (ProjError, ValueError, TypeError):
        pass
    return polygons


def _bank_edge_candidate_cells(
    *,
    network_gpkg_path: Path,
    raster_crs,
    sample_xy: np.ndarray,
    vals: np.ndarray,
    raster_resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    meta: dict[str, Any] = {
        'support_source': 'nhdarea_bank_edge_measured_authoritative',
        'network_gpkg_path': str(network_gpkg_path),
        'polygon_count': 0,
        'boundary_sample_count': 0,
        'candidate_cell_count': 0,
        'bank_edge_distance_m': None,
        'raster_resolution_m': float(raster_resolution_m) if np.isfinite(raster_resolution_m) else None,
    }
    polygons = _read_linear_polygons_for_raster(network_gpkg_path, raster_crs)
    if polygons is None or polygons.empty:
        meta['support_source'] = 'missing_linear_polygons'
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float), meta
    meta['polygon_count'] = int(len(polygons))
    boundary_xy = _boundary_sample_xy(polygons, spacing_m=_WSE_BANK_EDGE_SAMPLE_SPACING_M)
    meta['boundary_sample_count'] = int(len(boundary_xy))
    if boundary_xy.size == 0:
        meta['support_source'] = 'empty_polygon_boundaries'
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float), meta
    sample_xy_metric = _xy_to_metric(sample_xy, raster_crs)
    boundary_xy_metric = _xy_to_metric(boundary_xy, raster_crs)
    if sample_xy_metric.size == 0 or boundary_xy_metric.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float), meta
    edge_distance = max(float(_WSE_BANK_EDGE_DISTANCE_M), float(raster_resolution_m) * 1.5 if np.isfinite(raster_resolution_m) else 0.0)
    meta['bank_edge_distance_m'] = float(edge_distance)
    tree = cKDTree(boundary_xy_metric)
    dists, _ = tree.query(sample_xy_metric, k=1, distance_upper_bound=edge_distance)
    near_bank = np.isfinite(dists) & np.isfinite(vals)
    meta['candidate_cell_count'] = int(np.count_nonzero(near_bank))
    return sample_xy_metric[near_bank], np.asarray(vals[near_bank], dtype=float), meta


def _support_radius_for_point(row: pd.Series, default_radius_m: float) -> float:
    radius = max(float(default_radius_m), float(_WSE_BANK_SUPPORT_MIN_RADIUS_M))
    for col in ('mean_width_m', 'width_proxy_m', 'width_m'):
        if col in row:
            try:
                width = float(row.get(col))
            except (TypeError, ValueError):
                width = float('nan')
            if np.isfinite(width) and width > 0.0:
                radius = max(radius, float(width) * float(_WSE_BANK_SUPPORT_WIDTH_RADIUS_FACTOR) + float(_WSE_BANK_EDGE_DISTANCE_M))
                break
    return float(radius)


def _sample_wse_support(
    centerline: gpd.GeoDataFrame,
    measured_path: Path,
    *,
    network_gpkg_path: Path,
    outer_radius_m: float = _WSE_OUTER_RADIUS_M,
    quantile: float = _WSE_QUANTILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample WSE support from measured cells adjacent to NHD/river polygon banks.

    The previous annulus sampler used any measured cell within a broad radius of
    the centerline. In sparse reaches that can select upland or valley-side topo
    and turn it into a water-surface shelf. WSE support is now restricted to
    measured cells close to the canonical linear polygon boundary using a
    lower bank-edge quantile, so the support
    represents bank-edge/near-water evidence instead of arbitrary surrounding
    terrain. Nearest measured values are still returned as audit-only local-bed
    context; they are not used as WSE construction support.
    """
    with rasterio.open(measured_path) as ds:
        arr = ds.read(1, masked=True).filled(np.nan)
        finite = np.isfinite(arr)
        rows, cols = np.where(finite)
        empty = np.full(len(centerline), np.nan)
        if rows.size == 0:
            return empty.copy(), empty.copy(), empty.copy(), {'support_source': 'no_finite_measured_cells'}
        xs, ys = rasterio.transform.xy(ds.transform, rows, cols, offset='center')
        sample_xy = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
        sample_xy_metric_all = _xy_to_metric(sample_xy, ds.crs)
        point_xy = _metric_xy(centerline, ds.crs)
        vals = arr[rows, cols].astype(float)
        try:
            raster_resolution_m = float(max(abs(float(ds.transform.a)), abs(float(ds.transform.e))))
        except (TypeError, ValueError):
            raster_resolution_m = float('nan')

        all_tree = cKDTree(sample_xy_metric_all)
        support_xy, support_vals, support_meta = _bank_edge_candidate_cells(
            network_gpkg_path=Path(network_gpkg_path),
            raster_crs=ds.crs,
            sample_xy=sample_xy,
            vals=vals,
            raster_resolution_m=raster_resolution_m,
        )
        out = np.full((len(centerline),), np.nan)
        dist_out = np.full((len(centerline),), np.nan)
        local_bed = np.full((len(centerline),), np.nan)
        if support_xy.size == 0 or support_vals.size == 0:
            for i, xy in enumerate(point_xy):
                nearest_dist, nearest_idx = all_tree.query(xy, k=1)
                if np.isfinite(nearest_dist):
                    local_bed[i] = float(vals[int(nearest_idx)])
            return out, dist_out, local_bed, support_meta

        support_tree = cKDTree(support_xy)
        for pos, (_, row) in enumerate(centerline.iterrows()):
            xy = point_xy[pos]
            nearest_dist, nearest_idx = all_tree.query(xy, k=1)
            if np.isfinite(nearest_dist):
                local_bed[pos] = float(vals[int(nearest_idx)])
            radius = _support_radius_for_point(row, outer_radius_m)
            idx = support_tree.query_ball_point(xy, radius)
            if not idx:
                continue
            idx_arr = np.asarray(idx, dtype=int)
            cand = support_vals[idx_arr]
            finite_vals = cand[np.isfinite(cand)]
            if finite_vals.size == 0:
                continue
            out[pos] = float(np.nanquantile(finite_vals, quantile))
            dist_out[pos] = float(np.nanmin(np.linalg.norm(support_xy[idx_arr] - xy, axis=1)))
        support_meta['centerline_support_count'] = int(np.count_nonzero(np.isfinite(out)))
        support_meta['centerline_record_count'] = int(len(centerline))
        return out, dist_out, local_bed, support_meta


def _estimate_nonincreasing_tail_slope(station: np.ndarray, values: np.ndarray, *, max_points: int = 8) -> float:
    """Estimate a non-positive WSE slope for unsupported tail extrapolation.

    Station is treated as increasing downstream. The WSE proxy should therefore be
    broadly non-increasing downstream. This helper uses only finite support values;
    it never fabricates a slope when the available support is flat or insufficient.
    """
    sta = np.asarray(station, dtype=float)
    vals = np.asarray(values, dtype=float)
    usable = np.isfinite(sta) & np.isfinite(vals)
    if np.count_nonzero(usable) < 2:
        return 0.0
    sta = sta[usable]
    vals = vals[usable]
    order = np.argsort(sta, kind='mergesort')
    sta = sta[order]
    vals = vals[order]
    if sta.size > max_points:
        left_n = max(2, max_points // 2)
        right_n = max(2, max_points - left_n)
        fit_sta = np.concatenate([sta[:left_n], sta[-right_n:]])
        fit_vals = np.concatenate([vals[:left_n], vals[-right_n:]])
    else:
        fit_sta = sta
        fit_vals = vals
    if np.nanmax(fit_sta) - np.nanmin(fit_sta) <= 1.0e-9:
        return 0.0
    try:
        slope = float(np.polyfit(fit_sta, fit_vals, 1)[0])
    except (np.linalg.LinAlgError, ValueError, TypeError, FloatingPointError):
        slope = float((fit_vals[-1] - fit_vals[0]) / (fit_sta[-1] - fit_sta[0]))
    if not np.isfinite(slope):
        return 0.0
    return float(min(0.0, slope))


def _interpolate_wse_without_flat_tails(station: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Interpolate WSE support while extrapolating tails with measured trend.

    ``pandas.interpolate(limit_direction='both')`` repeats endpoint values outside
    the finite support span. For export AOIs outside the local support, that repeat
    can make the WSE proxy flat and then make ``wse_proxy_z_m - offset_modeled_m``
    flat. This routine keeps interior interpolation but extends unsupported tails
    using the non-increasing slope estimated from finite WSE support. If a trend
    cannot be proven, it falls back to endpoint fill and reports that status.
    """
    sta = np.asarray(station, dtype=float)
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    usable = np.isfinite(sta) & np.isfinite(vals)
    finite_count = int(np.count_nonzero(usable))
    meta: dict[str, Any] = {
        'finite_support_count': finite_count,
        'tail_slope_m_per_m': 0.0,
        'tail_extrapolated_count': 0,
        'tail_policy': 'no_finite_support',
    }
    if finite_count == 0:
        return out, meta
    order = np.argsort(sta[usable], kind='mergesort')
    support_sta = sta[usable][order]
    support_vals = vals[usable][order]
    finite_station = np.isfinite(sta)
    if finite_count == 1 or (float(np.nanmax(support_sta) - np.nanmin(support_sta)) <= 1.0e-9):
        out[finite_station] = float(support_vals[0])
        meta['tail_extrapolated_count'] = int(np.count_nonzero(finite_station & ~usable))
        meta['tail_policy'] = 'single_support_endpoint_fill'
        return out, meta
    out[finite_station] = np.interp(sta[finite_station], support_sta, support_vals)
    slope = _estimate_nonincreasing_tail_slope(support_sta, support_vals)
    meta['tail_slope_m_per_m'] = float(slope)
    left_tail = finite_station & (sta < float(support_sta[0]))
    right_tail = finite_station & (sta > float(support_sta[-1]))
    tail_count = int(np.count_nonzero(left_tail) + np.count_nonzero(right_tail))
    if tail_count and abs(slope) > 1.0e-12:
        out[left_tail] = float(support_vals[0]) + slope * (sta[left_tail] - float(support_sta[0]))
        out[right_tail] = float(support_vals[-1]) + slope * (sta[right_tail] - float(support_sta[-1]))
        meta['tail_policy'] = 'trend_extrapolated_tails'
        meta['tail_extrapolated_count'] = tail_count
    elif tail_count:
        meta['tail_policy'] = 'endpoint_fill_flat_support'
        meta['tail_extrapolated_count'] = tail_count
    else:
        meta['tail_policy'] = 'interior_only_no_tails'
    return out, meta


def _group_key(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in _GROUP_FIELDS if c in frame.columns]
    if not cols:
        return pd.Series(['all'] * len(frame), index=frame.index, dtype=object)
    return frame[cols].astype(str).agg('|'.join, axis=1)



def _profile_group_key(frame: pd.DataFrame) -> pd.Series:
    """Return the broad canonical WSE profile grouping key.

    WSE is fit across the canonical river solution, not per short reach.
    Per-reach fitting makes sparse bank-edge support look insufficient even
    when the parent component/levelpath has enough support to define the
    intended simple downstream WSE trend.
    """
    cols = [c for c in _WSE_PROFILE_GROUP_FIELDS if c in frame.columns]
    if not cols:
        return pd.Series(['canonical_parent'], index=frame.index, dtype=object)
    key_frame = frame[cols].copy()
    key_frame = key_frame.where(pd.notna(key_frame), 'none')
    return key_frame.astype(str).agg('|'.join, axis=1)

def _count_wse_reversals(values: np.ndarray, *, tolerance_m: float = _WSE_MONOTONE_TOLERANCE_M) -> dict[str, Any]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return {'count': 0, 'max_reversal_m': 0.0, 'flat_segment_count': 0}
    diffs = np.diff(vals)
    reversals = diffs[diffs > tolerance_m]
    return {
        'count': int(reversals.size),
        'max_reversal_m': float(np.nanmax(reversals)) if reversals.size else 0.0,
        'flat_segment_count': int(np.count_nonzero(np.abs(diffs) <= tolerance_m)),
    }


def _edge_median_delta(station: np.ndarray, values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    sta = np.asarray(station, dtype=float)
    vals = np.asarray(values, dtype=float)
    usable = np.isfinite(sta) & np.isfinite(vals)
    if np.count_nonzero(usable) < 2:
        return None, None, None
    order = np.argsort(sta[usable], kind='mergesort')
    ordered_vals = vals[usable][order]
    n = int(ordered_vals.size)
    edge_n = max(1, min(5, n // 4 if n >= 4 else 1))
    left = float(np.nanmedian(ordered_vals[:edge_n]))
    right = float(np.nanmedian(ordered_vals[-edge_n:]))
    return left, right, float(right - left)


def _orient_downstream_station_from_wse_support(
    station: np.ndarray,
    support_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose downstream station orientation from WSE support evidence.

    Station order from the centerline scaffold is not treated as automatically
    downstream. When finite WSE support clearly rises with station, the component
    is oriented in the opposite direction so monotone WSE enforcement is applied
    downstream rather than along arbitrary feature order. If support is flat or
    insufficient, the existing station order is retained but the low-confidence
    decision is recorded in the WSE direction audit.
    """
    sta = np.asarray(station, dtype=float)
    support = np.asarray(support_values, dtype=float)
    out = np.full(sta.shape, np.nan, dtype=float)
    finite_station = np.isfinite(sta)
    if not np.any(finite_station):
        return out, {
            'station_direction': 'unknown_no_finite_station',
            'direction_sign': 1,
            'direction_method': 'no_finite_station',
            'direction_confidence': 0.0,
            'finite_support_count': int(np.count_nonzero(np.isfinite(support))),
            'support_edge_delta_m': None,
            'support_left_edge_median_m': None,
            'support_right_edge_median_m': None,
            'station_range_m': None,
        }
    station_min = float(np.nanmin(sta[finite_station]))
    station_max = float(np.nanmax(sta[finite_station]))
    station_range = float(station_max - station_min)
    support_count = int(np.count_nonzero(np.isfinite(sta) & np.isfinite(support)))
    left, right, delta = _edge_median_delta(sta, support)
    if support_count < 2 or delta is None:
        sign = 1
        method = 'insufficient_wse_support_assume_station_m_increases_downstream'
        confidence = 0.0
    elif delta <= -_WSE_DIRECTION_MIN_SUPPORT_DROP_M:
        sign = 1
        method = 'wse_support_decreases_with_station_m'
        confidence = float(min(1.0, abs(delta) / max(_WSE_DIRECTION_MIN_SUPPORT_DROP_M, 1.0)))
    elif delta >= _WSE_DIRECTION_MIN_SUPPORT_DROP_M:
        sign = -1
        method = 'wse_support_increases_with_station_m_reverse_direction'
        confidence = float(min(1.0, abs(delta) / max(_WSE_DIRECTION_MIN_SUPPORT_DROP_M, 1.0)))
    else:
        sign = 1
        method = 'flat_or_weak_wse_support_assume_station_m_increases_downstream'
        confidence = float(abs(delta) / _WSE_DIRECTION_MIN_SUPPORT_DROP_M) if _WSE_DIRECTION_MIN_SUPPORT_DROP_M > 0 else 0.0
    if sign >= 0:
        out[finite_station] = sta[finite_station] - station_min
        station_direction = 'station_m_increases_downstream'
    else:
        out[finite_station] = station_max - sta[finite_station]
        station_direction = 'station_m_decreases_downstream'
    return out, {
        'station_direction': station_direction,
        'direction_sign': int(sign),
        'direction_method': method,
        'direction_confidence': float(confidence),
        'finite_support_count': support_count,
        'support_edge_delta_m': float(delta) if delta is not None and np.isfinite(delta) else None,
        'support_left_edge_median_m': float(left) if left is not None and np.isfinite(left) else None,
        'support_right_edge_median_m': float(right) if right is not None and np.isfinite(right) else None,
        'station_range_m': station_range if np.isfinite(station_range) else None,
        'min_support_drop_for_direction_m': float(_WSE_DIRECTION_MIN_SUPPORT_DROP_M),
    }


def _pava_non_decreasing(values: np.ndarray) -> np.ndarray:
    """Return the least-squares non-decreasing fit using PAVA.

    The routine is intentionally small and dependency-free so WSE monotonicity is
    enforced by an isotonic fit rather than by a running min/max clamp. A clamp
    treats the first low outlier as truth and can flatten an entire downstream
    reach; PAVA distributes the adjustment across adjacent violators while
    preserving as much supported longitudinal signal as possible.
    """
    y = np.asarray(values, dtype=float)
    if y.size <= 1:
        return y.copy()

    block_starts: list[int] = []
    block_ends: list[int] = []
    block_weights: list[float] = []
    block_means: list[float] = []

    for i, value in enumerate(y):
        block_starts.append(i)
        block_ends.append(i + 1)
        block_weights.append(1.0)
        block_means.append(float(value))

        while len(block_means) >= 2 and block_means[-2] > block_means[-1]:
            w0 = block_weights[-2]
            w1 = block_weights[-1]
            merged_weight = w0 + w1
            merged_mean = (block_means[-2] * w0 + block_means[-1] * w1) / merged_weight
            block_ends[-2] = block_ends[-1]
            block_weights[-2] = merged_weight
            block_means[-2] = float(merged_mean)
            block_starts.pop()
            block_ends.pop()
            block_weights.pop()
            block_means.pop()

    fitted = np.empty_like(y, dtype=float)
    for start, end, mean in zip(block_starts, block_ends, block_means, strict=True):
        fitted[start:end] = mean
    return fitted


def _robust_nonincreasing_trend_fit(
    station: np.ndarray,
    values: np.ndarray,
    *,
    min_drop_m: float = _WSE_TREND_MIN_DROP_M,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fit a smooth non-increasing long-profile trend from supported WSE signal.

    Isotonic regression preserves local support but can collapse a sparse/noisy
    component into a single terrace. The WSE proxy is a longitudinal water-
    surface trend, so when isotonic fitting destroys meaningful pre-monotone
    signal this helper constructs one deterministic non-increasing trend from
    the same component values. It refuses to invent a trend when the component
    lacks a defensible downstream drop.
    """
    sta = np.asarray(station, dtype=float)
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(sta) & np.isfinite(vals)
    meta: dict[str, Any] = {
        'method': 'not_attempted',
        'finite_count': int(np.count_nonzero(finite)),
        'station_range_m': None,
        'slope_m_per_m': None,
        'support_drop_m': None,
        'used': False,
        'reason': None,
    }
    if int(meta['finite_count']) < 2:
        meta['reason'] = 'insufficient_finite_values'
        return None, meta

    x = sta[finite]
    y = vals[finite]
    order = np.argsort(x, kind='mergesort')
    x = x[order]
    y = y[order]
    station_range = float(np.nanmax(x) - np.nanmin(x))
    meta['station_range_m'] = station_range
    if not np.isfinite(station_range) or station_range <= 1.0e-9:
        meta['reason'] = 'zero_station_range'
        return None, meta

    edge_n = max(1, min(8, int(y.size // 4) if y.size >= 4 else 1))
    left = float(np.nanmedian(y[:edge_n]))
    right = float(np.nanmedian(y[-edge_n:]))
    support_drop = float(left - right)
    meta['support_drop_m'] = support_drop

    if y.size > _WSE_TREND_MAX_FIT_POINTS:
        idx = np.linspace(0, y.size - 1, _WSE_TREND_MAX_FIT_POINTS).round().astype(int)
        fit_x = x[idx]
        fit_y = y[idx]
    else:
        fit_x = x
        fit_y = y

    slopes: list[float] = []
    n = int(fit_x.size)
    for i in range(n - 1):
        dx = fit_x[i + 1:] - fit_x[i]
        dy = fit_y[i + 1:] - fit_y[i]
        good = np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) > 1.0e-9)
        if np.any(good):
            slopes.extend((dy[good] / dx[good]).astype(float).tolist())
    if slopes:
        slope = float(np.nanmedian(np.asarray(slopes, dtype=float)))
    else:
        try:
            slope = float(np.polyfit(fit_x, fit_y, 1)[0])
        except (np.linalg.LinAlgError, ValueError, TypeError, FloatingPointError):
            slope = float((y[-1] - y[0]) / station_range)

    if not np.isfinite(slope):
        meta['reason'] = 'nonfinite_slope'
        return None, meta

    if slope >= -1.0e-12:
        if support_drop >= min_drop_m:
            slope = -support_drop / station_range
            meta['method'] = 'edge_drop_linear_trend'
        else:
            meta['reason'] = 'no_defensible_downstream_drop'
            meta['slope_m_per_m'] = float(slope)
            return None, meta
    else:
        meta['method'] = 'robust_pairwise_slope_linear_trend'

    if support_drop >= min_drop_m:
        slope = min(float(slope), -support_drop / station_range)
    meta['slope_m_per_m'] = float(slope)

    intercept = float(np.nanmedian(y - slope * x))
    fitted_all = intercept + slope * sta
    fitted_all[~np.isfinite(sta)] = np.nan
    meta['used'] = True
    return fitted_all.astype(float), meta


def _enforce_monotone_profile(
    vals: np.ndarray,
    *,
    station: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a non-increasing WSE profile without creating a flat terrace.

    Station order has already been oriented so index order increases downstream.
    The normal path uses isotonic PAVA. If that step fit collapses meaningful
    pre-monotone WSE signal into a dominant rounded shelf, a robust linear
    long-profile trend is used instead. This keeps the stage linear and fixes
    the construction artifact rather than weakening the flatness guard.
    """
    out = np.asarray(vals, dtype=float).copy()
    meta: dict[str, Any] = {
        'method': 'isotonic_pava_nonincreasing',
        'trend_used': False,
        'trend_meta': None,
        'pre_flatness': _dominant_flatness_summary(out),
        'isotonic_flatness': None,
        'final_flatness': None,
    }
    finite_idx = np.where(np.isfinite(out))[0]
    if finite_idx.size <= 1:
        meta['method'] = 'unchanged_insufficient_finite_values'
        meta['final_flatness'] = _dominant_flatness_summary(out)
        return out, meta

    finite_values = out[finite_idx]
    isotonic = out.copy()
    isotonic[finite_idx] = -_pava_non_decreasing(-finite_values)
    iso_flat = _dominant_flatness_summary(isotonic)
    meta['isotonic_flatness'] = iso_flat

    pre_summary = meta['pre_flatness'] or {}
    pre_range = pre_summary.get('range_m')
    pre_has_signal = bool(
        int(pre_summary.get('finite_count') or 0) >= _WSE_FLATNESS_MIN_FINITE_COUNT
        and pre_range is not None
        and float(pre_range) >= _WSE_MEANINGFUL_SUPPORT_RANGE_M
        and not bool(pre_summary.get('flatness_fail'))
    )
    if pre_has_signal and bool(iso_flat.get('flatness_fail')) and station is not None:
        trend, trend_meta = _robust_nonincreasing_trend_fit(np.asarray(station, dtype=float), out)
        meta['trend_meta'] = trend_meta
        if trend is not None:
            trend_flat = _dominant_flatness_summary(trend)
            iso_fraction = float(iso_flat.get('dominant_rounded_fraction') or 0.0)
            trend_fraction = float(trend_flat.get('dominant_rounded_fraction') or 0.0)
            if (not bool(trend_flat.get('flatness_fail'))) or trend_fraction < iso_fraction:
                meta['method'] = 'robust_linear_nonincreasing_trend_after_isotonic_flattening'
                meta['trend_used'] = True
                meta['final_flatness'] = trend_flat
                return trend, meta
    meta['final_flatness'] = iso_flat
    return isotonic, meta


def _enforce_monotone(vals: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper for tests/imports."""
    fitted, _ = _enforce_monotone_profile(vals, station=None)
    return fitted



def _floor_conflict_summary(proxy: np.ndarray, local_floor: np.ndarray) -> dict[str, Any]:
    px = np.asarray(proxy, dtype=float)
    fl = np.asarray(local_floor, dtype=float)
    finite = np.isfinite(px) & np.isfinite(fl)
    deficit = np.where(finite, fl - px, np.nan)
    conflict = finite & (deficit > _WSE_MONOTONE_TOLERANCE_M)
    vals = deficit[conflict]
    return {
        'count': int(np.count_nonzero(conflict)),
        'max_deficit_m': float(np.nanmax(vals)) if vals.size else 0.0,
        'median_deficit_m': float(np.nanmedian(vals)) if vals.size else 0.0,
    }


def _wse_science_summary(out: gpd.GeoDataFrame) -> dict[str, Any]:
    raw = pd.to_numeric(out.get('wse_support_z_m'), errors='coerce').to_numpy(dtype=float)
    proxy = pd.to_numeric(out.get('wse_proxy_z_m'), errors='coerce').to_numpy(dtype=float)
    local_floor = pd.to_numeric(out.get('local_bed_floor_z_m'), errors='coerce').to_numpy(dtype=float)
    finite_pair = np.isfinite(raw) & np.isfinite(proxy)
    adjustments = proxy[finite_pair] - raw[finite_pair]
    monotone_violations = 0
    max_reversal_m = 0.0
    flat_segment_count = 0
    order_field = 'station_downstream_m' if 'station_downstream_m' in out.columns else 'station_m'
    for _, grp_idx in _group_key(out).groupby(_group_key(out)).groups.items():
        grp = out.loc[list(grp_idx)].copy().sort_values(order_field, kind='mergesort')
        vals = pd.to_numeric(grp['wse_proxy_z_m'], errors='coerce').to_numpy(dtype=float)
        reversal_summary = _count_wse_reversals(vals)
        monotone_violations += int(reversal_summary['count'])
        max_reversal_m = max(max_reversal_m, float(reversal_summary['max_reversal_m']))
        flat_segment_count += int(reversal_summary['flat_segment_count'])
    tail_slopes = pd.to_numeric(out.get('wse_tail_slope_m_per_m', pd.Series(dtype=float)), errors='coerce').to_numpy(dtype=float)
    tail_counts = pd.to_numeric(out.get('wse_tail_extrapolated_count_group', pd.Series(dtype=float)), errors='coerce').to_numpy(dtype=float)
    floor_conflicts = _floor_conflict_summary(proxy, local_floor)
    return {
        'candidate_downstream_direction': 'component_specific_wse_support_direction',
        'expected_wse_trend': 'non_increasing_downstream',
        'direction_evidence_score': None,
        'direction_methods': {str(k): int(v) for k, v in out.get('wse_direction_method', pd.Series(dtype=object)).astype(str).value_counts().to_dict().items()},
        'station_directions': {str(k): int(v) for k, v in out.get('station_direction', pd.Series(dtype=object)).astype(str).value_counts().to_dict().items()},
        'raw_wse': _finite_stats(raw),
        'proxy_wse': _finite_stats(proxy),
        'local_bed_floor': _finite_stats(local_floor),
        'final_proxy_floor_conflict': floor_conflicts,
        'monotone_violation_count': int(monotone_violations),
        'max_monotone_reversal_m': float(max_reversal_m),
        'monotone_adjustment_count': int(np.count_nonzero(np.isfinite(adjustments) & (np.abs(adjustments) > 1.0e-9))),
        'largest_monotone_adjustment_m': float(np.nanmax(np.abs(adjustments))) if adjustments.size else None,
        'flat_segment_count': int(flat_segment_count),
        'wse_quality_flags': {str(k): int(v) for k, v in out.get('wse_quality_flag', pd.Series(dtype=object)).astype(str).value_counts().to_dict().items()},
        'tail_policy_counts': {str(k): int(v) for k, v in out.get('wse_tail_policy', pd.Series(dtype=object)).astype(str).value_counts().to_dict().items()},
        'tail_extrapolated_point_count': int(np.nanmax(tail_counts)) if np.count_nonzero(np.isfinite(tail_counts)) else 0,
        'tail_slope': _finite_stats(tail_slopes),
    }





def _write_gpkg(path: Path, frame: gpd.GeoDataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    frame.to_file(path, driver='GPKG')
    return path


def _copy_gpkg(src: Path, dst: Path) -> Path:
    # GeoPackage is a single-file SQLite container, so a byte copy is safe after to_file closes it.
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    dst.write_bytes(src.read_bytes())
    return dst


def _wse_reports_dir(ctx: RiverWorkflowContext) -> Path:
    return ctx.paths.root.parent / 'reports'


def _wse_artifact_reports_dir(ctx: RiverWorkflowContext) -> Path:
    return _wse_reports_dir(ctx) / 'wse_artifacts'


def _write_wse_current_path_audit(
    ctx: RiverWorkflowContext,
    *,
    canonical_measured: Path,
    centerline_result: CenterlineStageResult,
    authoritative_result: AuthoritativeStageResult,
    out: gpd.GeoDataFrame,
    science_summary: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> Path:
    reports = _wse_reports_dir(ctx)
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / 'WSE_CURRENT_PATH_AUDIT.txt'
    lines = [
        'WSE current path audit',
        '======================',
        '',
        'Purpose:',
        '  Document the active one-path WSE proxy construction used downstream.',
        '',
        'Active construction:',
        f'  centerline input: {centerline_result.centerline_points_path}',
        f'  authoritative measured-only source required by WSE stage: {canonical_measured}',
        f'  authoritative stage measured-only path: {authoritative_result.solve_authoritative_base_measured_only_path}',
        '  support sampling: measured cells close to canonical NHD/river polygon bank edges',
        f'  bank-edge cell distance: {_WSE_BANK_EDGE_DISTANCE_M} m; centerline support search radius uses channel width proxy when available',
        f'  support statistic: bank-edge lower quantile={_WSE_QUANTILE}',
        '  nearest measured value: audit-only; not used as a WSE floor',
        '  profile fit: robust anchors -> decreasing base profile -> bounded residual refinement',
        '  residual smoothing: station-binned median residuals with rolling median window=5',
        '  monotone rule: one final PAVA non-increasing fit in component-specific downstream station order',
        '',
        'Important scientific interpretation:',
        '  WSE support values are restricted to bank-edge/near-water proxy evidence.',
        '  Bank/near-bank support is not treated as bed measurement.',
        '  This pass writes support/profile/proxy artifacts from one active WSE construction path and avoids broad upland annulus support.',
        '',
        'Output artifacts:',
    ]
    for key, value in artifact_paths.items():
        lines.append(f'  {key}: {value}')
    lines.extend([
        '',
        'Counts:',
        f'  records: {len(out)}',
        f"  finite support count: {science_summary.get('raw_wse', {}).get('count')}",
        f"  finite proxy count: {science_summary.get('proxy_wse', {}).get('count')}",
        f"  monotone violation count after proxy: {science_summary.get('monotone_violation_count')}",
        f"  flat segment count after proxy: {science_summary.get('flat_segment_count')}",
        f"  final proxy floor conflict count: {(science_summary.get('final_proxy_floor_conflict') or {}).get('count')}",
        f"  max final proxy floor deficit m: {(science_summary.get('final_proxy_floor_conflict') or {}).get('max_deficit_m')}",
        '',
        'Quality flag counts:',
    ])
    for key, value in sorted((science_summary.get('wse_quality_flags') or {}).items()):
        lines.append(f'  {key}: {value}')
    lines.extend(['', 'Tail policy counts:'])
    for key, value in sorted((science_summary.get('tail_policy_counts') or {}).items()):
        lines.append(f'  {key}: {value}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _write_wse_direction_audit(
    ctx: RiverWorkflowContext,
    *,
    out: gpd.GeoDataFrame,
    group_direction_checks: list[dict[str, Any]],
) -> Path:
    reports = _wse_reports_dir(ctx)
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / 'WSE_DIRECTION_AUDIT.txt'
    lines = [
        'WSE direction audit',
        '===================',
        '',
        'Purpose:',
        '  Record the downstream station orientation used for WSE interpolation and monotone enforcement.',
        '',
        'Rule:',
        '  If WSE support clearly decreases with station_m, station_m is treated as downstream-increasing.',
        '  If WSE support clearly increases with station_m, station order is reversed for this component.',
        '  If support is insufficient or flat, station_m order is retained and low confidence is reported.',
        '',
        f'Component/group count: {len(group_direction_checks)}',
        '',
    ]
    if group_direction_checks:
        methods = pd.Series([str(g.get('direction_method')) for g in group_direction_checks]).value_counts().to_dict()
        lines.append('Direction method counts:')
        for key, value in sorted(methods.items()):
            lines.append(f'  {key}: {int(value)}')
        lines.append('')
        reversed_count = sum(1 for g in group_direction_checks if int(g.get('direction_sign', 1) or 1) < 0)
        ambiguous_count = sum(1 for g in group_direction_checks if float(g.get('direction_confidence', 0.0) or 0.0) < 0.25)
        lines.extend([
            f'Components reversed: {reversed_count}',
            f'Low-confidence direction components: {ambiguous_count}',
            '',
            'Per-component direction checks:',
        ])
        for item in group_direction_checks:
            lines.append(
                '  '
                f"group={item.get('group_key')} "
                f"direction={item.get('station_direction')} "
                f"method={item.get('direction_method')} "
                f"confidence={item.get('direction_confidence')} "
                f"support_count={item.get('finite_support_count')} "
                f"support_delta_m={item.get('support_edge_delta_m')}"
            )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _write_wse_proxy_audit(
    ctx: RiverWorkflowContext,
    *,
    out: gpd.GeoDataFrame,
    science_summary: dict[str, Any],
    group_direction_checks: list[dict[str, Any]],
) -> Path:
    reports = _wse_reports_dir(ctx)
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / 'WSE_PROXY_AUDIT.txt'
    raw = science_summary.get('raw_wse') or {}
    proxy = science_summary.get('proxy_wse') or {}
    lines = [
        'WSE proxy audit',
        '===============',
        '',
        'Purpose:',
        '  Summarize final WSE proxy behavior after direction selection, smoothing, and isotonic monotone enforcement.',
        '',
        'Summary:',
        f"  finite raw support count: {raw.get('count')}",
        f"  finite proxy count: {proxy.get('count')}",
        f"  proxy min/max/median m: {proxy.get('min')} / {proxy.get('max')} / {proxy.get('median')}",
        f"  monotone violation count: {science_summary.get('monotone_violation_count')}",
        f"  max monotone reversal m: {science_summary.get('max_monotone_reversal_m')}",
        f"  flat segment count: {science_summary.get('flat_segment_count')}",
        f"  final proxy floor conflict count: {(science_summary.get('final_proxy_floor_conflict') or {}).get('count')}",
        f"  max final proxy floor deficit m: {(science_summary.get('final_proxy_floor_conflict') or {}).get('max_deficit_m')}",
        '',
        'Station direction counts:',
    ]
    for key, value in sorted((science_summary.get('station_directions') or {}).items()):
        lines.append(f'  {key}: {value}')
    lines.append('')
    lines.append('Direction method counts:')
    for key, value in sorted((science_summary.get('direction_methods') or {}).items()):
        lines.append(f'  {key}: {value}')
    lines.append('')
    lines.append('Quality flag counts:')
    for key, value in sorted((science_summary.get('wse_quality_flags') or {}).items()):
        lines.append(f'  {key}: {value}')
    lines.append('')
    lines.append('Tail policy counts:')
    for key, value in sorted((science_summary.get('tail_policy_counts') or {}).items()):
        lines.append(f'  {key}: {value}')
    lines.append('')
    lines.append('Per-component monotone checks:')
    order_field = 'station_downstream_m' if 'station_downstream_m' in out.columns else 'station_m'
    for _, grp_idx in _group_key(out).groupby(_group_key(out)).groups.items():
        grp = out.loc[list(grp_idx)].copy().sort_values(order_field, kind='mergesort')
        vals = pd.to_numeric(grp.get('wse_proxy_z_m'), errors='coerce').to_numpy(dtype=float)
        rev = _count_wse_reversals(vals)
        group_key = str(_group_key(grp).iloc[0]) if len(grp) else 'unknown'
        finite_vals = vals[np.isfinite(vals)]
        total_drop = float(finite_vals[0] - finite_vals[-1]) if finite_vals.size > 1 else None
        lines.append(
            '  '
            f'group={group_key} '
            f"records={len(grp)} finite={int(finite_vals.size)} total_drop_m={total_drop} "
            f"reversals={rev.get('count')} max_reversal_m={rev.get('max_reversal_m')} "
            f"flat_segments={rev.get('flat_segment_count')}"
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _write_wse_audit_artifacts(
    ctx: RiverWorkflowContext,
    out: gpd.GeoDataFrame,
    *,
    canonical_measured: Path,
    centerline_result: CenterlineStageResult,
    authoritative_result: AuthoritativeStageResult,
    science_summary: dict[str, Any],
    group_direction_checks: list[dict[str, Any]],
) -> dict[str, Path]:
    support = build_wse_support_artifact(out)
    trend = build_wse_trend_artifact(out)
    pre_smooth = build_wse_pre_smooth_artifact(out)
    proxy = build_wse_proxy_final_artifact(out)

    paths = {
        'centerline_wse_support_points': ctx.paths.centerline_wse_support_points,
        'centerline_wse_trend_points': ctx.paths.centerline_wse_trend_points,
        'centerline_wse_pre_smooth_points': ctx.paths.centerline_wse_pre_smooth_points,
        'centerline_wse_proxy_points': ctx.paths.centerline_wse_proxy_points,
    }
    _write_gpkg(paths['centerline_wse_support_points'], support)
    _write_gpkg(paths['centerline_wse_trend_points'], trend)
    _write_gpkg(paths['centerline_wse_pre_smooth_points'], pre_smooth)
    # The final proxy is written by the caller to preserve the existing validation flow.

    retained_dir = _wse_artifact_reports_dir(ctx)
    retained_paths = {
        'retained_centerline_wse_support_points': retained_dir / paths['centerline_wse_support_points'].name,
        'retained_centerline_wse_trend_points': retained_dir / paths['centerline_wse_trend_points'].name,
        'retained_centerline_wse_pre_smooth_points': retained_dir / paths['centerline_wse_pre_smooth_points'].name,
        'retained_centerline_wse_proxy_points': retained_dir / paths['centerline_wse_proxy_points'].name,
    }
    for key, retained in retained_paths.items():
        source_key = key.replace('retained_', '')
        source = paths.get(source_key)
        if source is not None and source.exists():
            _copy_gpkg(source, retained)

    artifact_paths = {**paths, **retained_paths}
    audit_path = _write_wse_current_path_audit(
        ctx,
        canonical_measured=canonical_measured,
        centerline_result=centerline_result,
        authoritative_result=authoritative_result,
        out=out,
        science_summary=science_summary,
        artifact_paths=artifact_paths,
    )
    direction_audit_path = _write_wse_direction_audit(ctx, out=out, group_direction_checks=group_direction_checks)
    proxy_audit_path = _write_wse_proxy_audit(ctx, out=out, science_summary=science_summary, group_direction_checks=group_direction_checks)
    artifact_paths['wse_current_path_audit'] = audit_path
    artifact_paths['wse_direction_audit'] = direction_audit_path
    artifact_paths['wse_proxy_audit'] = proxy_audit_path
    wse_stage_contract_path = write_wse_stage_contract(
        reports_dir=_wse_artifact_reports_dir(ctx),
        paths=paths,
        science_summary=science_summary,
        group_direction_checks=group_direction_checks,
    )
    artifact_paths['wse_stage_contract'] = wse_stage_contract_path
    manifest_path = retained_dir / 'wse_artifact_manifest.json'
    _write_json(
        manifest_path,
        {
            'schema_version': 2,
            'stage': 'centerline_wse_proxy',
            'purpose': 'Expose the current WSE support/trend/pre-smooth/proxy chain and component-wise downstream direction decision.',
            'canonical_measured_source': str(canonical_measured),
            'artifacts': {k: str(v) for k, v in artifact_paths.items()},
            'audit': str(audit_path),
            'direction_audit': str(direction_audit_path),
            'proxy_audit': str(proxy_audit_path),
            'direction_checks': group_direction_checks,
            'science_summary': science_summary,
        },
    )
    artifact_paths['wse_artifact_manifest'] = manifest_path
    return artifact_paths


def _rolling_median_1d(values: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    if window <= 1:
        return arr.copy()
    half = int(window // 2)
    out = np.empty_like(arr, dtype=float)
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        out[i] = float(np.nanmedian(arr[lo:hi]))
    return out


def _edge_anchor_values(station: np.ndarray, support: np.ndarray) -> tuple[float, float, dict[str, Any]]:
    sta = np.asarray(station, dtype=float)
    vals = np.asarray(support, dtype=float)
    finite = np.isfinite(sta) & np.isfinite(vals)
    meta: dict[str, Any] = {
        'finite_support_count': int(np.count_nonzero(finite)),
        'edge_count': 0,
        'upstream_anchor_z_m': None,
        'downstream_anchor_z_m': None,
        'anchor_drop_m': None,
        'anchor_method': 'not_available',
    }
    if int(meta['finite_support_count']) == 0:
        return float('nan'), float('nan'), meta
    x = sta[finite]
    y = vals[finite]
    order = np.argsort(x, kind='mergesort')
    y = y[order]
    edge_n = max(1, int(np.ceil(float(y.size) * _WSE_PROFILE_EDGE_FRACTION)))
    edge_n = min(edge_n, max(1, y.size // 2)) if y.size > 1 else 1
    upstream = float(np.nanmedian(y[:edge_n]))
    downstream = float(np.nanmedian(y[-edge_n:]))
    meta.update({
        'edge_count': int(edge_n),
        'upstream_anchor_z_m': upstream,
        'downstream_anchor_z_m': downstream,
        'anchor_drop_m': float(upstream - downstream),
        'anchor_method': 'median_of_ordered_bank_edge_support_edges',
    })
    return upstream, downstream, meta


def _fit_monotone_longitudinal_wse_profile(
    station_downstream: np.ndarray,
    support_values: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Fit one robust downstream-non-increasing WSE profile from noisy support.

    This is the replacement for the previous interpolation/flatness/repair chain.
    The first-order profile is a simple line between robust upstream/downstream
    anchors. Local bank-edge support is used only as a bounded residual correction,
    and the final output is projected once onto the non-increasing profile
    constraint.
    """
    sta = np.asarray(station_downstream, dtype=float)
    support = np.asarray(support_values, dtype=float)
    finite_station = np.isfinite(sta)
    finite_support = finite_station & np.isfinite(support)
    meta: dict[str, Any] = {
        'method': 'one_path_bank_edge_anchor_residual_monotone_profile',
        'finite_station_count': int(np.count_nonzero(finite_station)),
        'finite_support_count': int(np.count_nonzero(finite_support)),
        'station_min_m': None,
        'station_max_m': None,
        'station_range_m': None,
        'bin_width_m': None,
        'bin_count': 0,
        'used_bin_count': 0,
        'base_slope_m_per_m': None,
        'residual_bound_m': float(_WSE_PROFILE_RESIDUAL_BOUND_M),
        'monotone_violation_count_before_enforcement': 0,
        'monotone_violation_count_after_enforcement': 0,
        'status': 'not_started',
        'reason': None,
    }
    if int(meta['finite_station_count']) == 0:
        meta['status'] = 'failed'
        meta['reason'] = 'no_finite_station'
        return None, None, None, meta
    station_min = float(np.nanmin(sta[finite_station]))
    station_max = float(np.nanmax(sta[finite_station]))
    station_range = float(station_max - station_min)
    meta.update({'station_min_m': station_min, 'station_max_m': station_max, 'station_range_m': station_range})
    if not np.isfinite(station_range) or station_range <= 1.0e-9:
        meta['status'] = 'failed'
        meta['reason'] = 'zero_station_range'
        return None, None, None, meta
    if int(meta['finite_support_count']) < _WSE_PROFILE_MIN_SUPPORT_COUNT:
        meta['status'] = 'failed'
        meta['reason'] = 'insufficient_bank_edge_wse_support'
        return None, None, None, meta

    upstream_anchor, downstream_anchor, anchor_meta = _edge_anchor_values(sta, support)
    meta['anchors'] = anchor_meta
    if not (np.isfinite(upstream_anchor) and np.isfinite(downstream_anchor)):
        meta['status'] = 'failed'
        meta['reason'] = 'invalid_anchor_values'
        return None, None, None, meta

    anchor_drop = float(upstream_anchor - downstream_anchor)
    if anchor_drop < 0.0:
        # Direction should usually have been resolved before this fit. A small
        # negative drop is treated as low-gradient/noisy support; a strong
        # negative drop is still projected to a flat base instead of introducing
        # downstream-rising WSE.
        base_drop = 0.0
        meta['anchor_warning'] = 'oriented_support_not_decreasing; using flat_base_profile'
    else:
        base_drop = anchor_drop
    slope = base_drop / station_range if station_range > 0.0 else 0.0
    meta['base_slope_m_per_m'] = float(slope)
    base = upstream_anchor - slope * (sta - station_min)

    # Bin residuals around the base trend. Binning keeps noisy bank-edge samples
    # from becoming the profile while still allowing broad local refinement.
    finite = finite_support & np.isfinite(base)
    support_x = sta[finite]
    support_residual = support[finite] - base[finite]
    bin_width = max(float(_WSE_PROFILE_MIN_BIN_WIDTH_M), station_range / float(max(1, _WSE_PROFILE_TARGET_BIN_COUNT)))
    meta['bin_width_m'] = float(bin_width)
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        meta['status'] = 'failed'
        meta['reason'] = 'invalid_bin_width'
        return None, None, None, meta
    bin_id = np.floor((support_x - station_min) / bin_width).astype(int)
    binned_x: list[float] = []
    binned_resid: list[float] = []
    for bid in np.unique(bin_id):
        m = bin_id == bid
        if not np.any(m):
            continue
        binned_x.append(float(np.nanmedian(support_x[m])))
        binned_resid.append(float(np.nanmedian(support_residual[m])))
    meta['bin_count'] = int(len(np.unique(bin_id)))
    meta['used_bin_count'] = int(len(binned_x))

    if len(binned_x) >= 2:
        bx = np.asarray(binned_x, dtype=float)
        br = np.asarray(binned_resid, dtype=float)
        order = np.argsort(bx, kind='mergesort')
        bx = bx[order]
        br = br[order]
        br = _rolling_median_1d(br, window=5)
        # Residuals refine the broad trend but cannot dominate it.
        residual_bound = float(_WSE_PROFILE_RESIDUAL_BOUND_M)
        if np.isfinite(residual_bound) and residual_bound > 0.0:
            br = np.clip(br, -residual_bound, residual_bound)
        residual = np.interp(sta, bx, br, left=float(br[0]), right=float(br[-1]))
        meta['residual_method'] = 'station_binned_median_rolling5_bounded'
    else:
        residual = np.full(sta.shape, float(np.nanmedian(support_residual)), dtype=float)
        if not np.isfinite(residual).all():
            residual = np.zeros_like(sta, dtype=float)
        meta['residual_method'] = 'single_residual_median'

    provisional = base + residual
    before = _count_wse_reversals(provisional)
    meta['monotone_violation_count_before_enforcement'] = int(before.get('count', 0))
    final = -_pava_non_decreasing(-provisional)
    after = _count_wse_reversals(final)
    meta['monotone_violation_count_after_enforcement'] = int(after.get('count', 0))
    meta['status'] = 'ok'
    return base, provisional, final, meta


def _profile_group_receipt(
    *,
    group_key: str,
    record_count: int,
    direction_meta: dict[str, Any],
    fit_meta: dict[str, Any],
    support_values: np.ndarray,
    final_profile: np.ndarray,
) -> dict[str, Any]:
    support = np.asarray(support_values, dtype=float)
    final = np.asarray(final_profile, dtype=float)
    return {
        'group_key': group_key,
        'record_count': int(record_count),
        'direction': _jsonable(direction_meta),
        'profile_fit': _jsonable(fit_meta),
        'support': _finite_stats(support),
        'final_profile': _finite_stats(final),
        'final_reversal_summary': _count_wse_reversals(final),
    }


def run_wse_proxy_stage(
    ctx: RiverWorkflowContext,
    centerline_result: CenterlineStageResult,
    authoritative_result: AuthoritativeStageResult,
) -> WSEProxyStageResult:
    centerline = gpd.read_file(centerline_result.centerline_points_path)
    canonical_measured = (
        Path(ctx.linear_inputs.canonical_solve_authoritative_measured_only_path)
        if ctx.linear_inputs is not None and ctx.linear_inputs.canonical_solve_authoritative_measured_only_path is not None
        else None
    )
    if canonical_measured is None or not canonical_measured.exists():
        raise RuntimeError('river_workflow_wse_missing_canonical_measured_only')
    if Path(authoritative_result.solve_authoritative_base_measured_only_path).resolve() != canonical_measured.resolve():
        raise RuntimeError(
            f'river_workflow_wse_noncanonical_measured_source:{authoritative_result.solve_authoritative_base_measured_only_path}:{canonical_measured}'
        )

    raw, dist, nearest_measured_context, support_source_meta = _sample_wse_support(
        centerline,
        canonical_measured,
        network_gpkg_path=ctx.paths.canonical_solve_network,
    )

    out = centerline.copy()
    out['wse_support_z_m'] = raw
    out['support_distance_m'] = dist
    out['wse_support_distance_m'] = dist
    out['wse_support_count'] = 0
    out['wse_profile_z_m'] = np.nan
    out['wse_direction'] = 'unresolved'
    out['nearest_measured_context_z_m'] = nearest_measured_context
    # Retain this legacy-named field for old reports, but do not use it as a WSE
    # floor. The WSE stage now fits one bank-edge longitudinal profile only.
    out['local_bed_floor_z_m'] = np.nan
    out['station_direction'] = 'unresolved'
    out['station_downstream_m'] = np.nan
    out['wse_direction_method'] = 'unresolved'
    out['wse_direction_confidence'] = np.nan
    out['wse_direction_support_delta_m'] = np.nan
    out['wse_proxy_raw_interpolated_z_m'] = np.nan
    out['wse_profile_base_z_m'] = np.nan
    out['wse_profile_residual_z_m'] = np.nan
    out['wse_proxy_pre_monotone_z_m'] = np.nan
    out['wse_proxy_z_m'] = np.nan
    out['wse_final_floor_conflict'] = False
    out['wse_final_floor_conflict_m'] = 0.0
    out['wse_tail_policy'] = 'not_used_one_path_profile'
    out['wse_tail_slope_m_per_m'] = 0.0
    out['wse_tail_extrapolated_count_group'] = 0
    out['wse_monotone_method'] = 'one_path_pava_nonincreasing'
    out['wse_monotone_trend_used'] = True
    out['wse_monotone_trend_slope_m_per_m'] = np.nan
    out['wse_profile_method'] = 'bank_edge_anchor_residual_monotone_profile'
    out['wse_confidence'] = 'unknown'
    out['wse_quality_flag'] = np.where(np.isfinite(raw), 'bank_edge_support_used_for_profile', 'profile_filled_from_group_trend')
    out['wse_valid'] = False
    out['wse_group_support_count'] = 0
    out['wse_group_support_range_m'] = np.nan
    out['wse_group_final_dominant_fraction'] = np.nan
    out['wse_group_flatness_guard_status'] = 'replaced_by_one_path_profile_validation'
    out['wse_group_flatness_fail_reason'] = None

    profile_group_keys = _profile_group_key(out)
    profile_groups: list[dict[str, Any]] = []
    direction_checks: list[dict[str, Any]] = []
    fit_failures: list[dict[str, Any]] = []

    for _, grp_idx in profile_group_keys.groupby(profile_group_keys, sort=False).groups.items():
        grp_indices = list(grp_idx)
        grp_unsorted = out.loc[grp_indices].copy()
        vals_unsorted = pd.to_numeric(grp_unsorted['wse_support_z_m'], errors='coerce').to_numpy(dtype=float)
        station_unsorted = pd.to_numeric(grp_unsorted['station_m'], errors='coerce').to_numpy(dtype=float)
        downstream_station_unsorted, direction_meta = _orient_downstream_station_from_wse_support(station_unsorted, vals_unsorted)
        group_key_value = str(_profile_group_key(grp_unsorted).iloc[0]) if len(grp_unsorted) else 'unknown'
        direction_meta['group_key'] = group_key_value
        direction_meta['record_count'] = int(len(grp_unsorted))
        direction_checks.append(direction_meta)

        out.loc[grp_unsorted.index, 'station_downstream_m'] = downstream_station_unsorted
        out.loc[grp_unsorted.index, 'station_direction'] = str(direction_meta.get('station_direction'))
        out.loc[grp_unsorted.index, 'wse_direction'] = str(direction_meta.get('station_direction'))
        out.loc[grp_unsorted.index, 'wse_direction_method'] = str(direction_meta.get('direction_method'))
        out.loc[grp_unsorted.index, 'wse_direction_confidence'] = float(direction_meta.get('direction_confidence', 0.0) or 0.0)
        delta = direction_meta.get('support_edge_delta_m')
        out.loc[grp_unsorted.index, 'wse_direction_support_delta_m'] = float(delta) if delta is not None else np.nan

        grp = out.loc[grp_indices].copy().sort_values('station_downstream_m', kind='mergesort')
        station = pd.to_numeric(grp['station_downstream_m'], errors='coerce').to_numpy(dtype=float)
        support = pd.to_numeric(grp['wse_support_z_m'], errors='coerce').to_numpy(dtype=float)
        base, provisional, final, fit_meta = _fit_monotone_longitudinal_wse_profile(station, support)
        if final is None or not np.isfinite(final).any():
            failure = {
                'group_key': group_key_value,
                'record_count': int(len(grp)),
                'direction': _jsonable(direction_meta),
                'profile_fit': _jsonable(fit_meta),
            }
            fit_failures.append(failure)
            continue

        residual = final - base if base is not None else np.full(final.shape, np.nan, dtype=float)
        support_count = int(np.count_nonzero(np.isfinite(support)))
        support_range = float(np.nanmax(support[np.isfinite(support)]) - np.nanmin(support[np.isfinite(support)])) if support_count else np.nan
        dominant = _dominant_flatness_summary(final)

        out.loc[grp.index, 'wse_proxy_raw_interpolated_z_m'] = base
        out.loc[grp.index, 'wse_profile_base_z_m'] = base
        out.loc[grp.index, 'wse_profile_residual_z_m'] = residual
        out.loc[grp.index, 'wse_proxy_pre_monotone_z_m'] = provisional if provisional is not None else final
        out.loc[grp.index, 'wse_proxy_z_m'] = final
        out.loc[grp.index, 'wse_profile_z_m'] = final
        out.loc[grp.index, 'wse_valid'] = np.isfinite(final)
        out.loc[grp.index, 'wse_quality_flag'] = np.where(
            np.isfinite(support),
            'bank_edge_support_profile_anchor',
            'profile_interpolated_between_bank_edge_support',
        )
        out.loc[grp.index, 'wse_support_count'] = support_count
        out.loc[grp.index, 'wse_group_support_count'] = support_count
        out.loc[grp.index, 'wse_group_support_range_m'] = support_range
        out.loc[grp.index, 'wse_group_final_dominant_fraction'] = dominant.get('dominant_rounded_fraction')
        out.loc[grp.index, 'wse_group_flatness_guard_status'] = 'ok_one_path_monotone_profile'
        out.loc[grp.index, 'wse_monotone_trend_slope_m_per_m'] = float(fit_meta.get('base_slope_m_per_m')) if fit_meta.get('base_slope_m_per_m') is not None else np.nan
        out.loc[grp.index, 'wse_confidence'] = np.where(
            np.isfinite(support),
            'direct_bank_edge_support',
            'group_profile_interpolated',
        )

        profile_groups.append(_profile_group_receipt(
            group_key=group_key_value,
            record_count=len(grp),
            direction_meta=direction_meta,
            fit_meta=fit_meta,
            support_values=support,
            final_profile=final,
        ))

    if fit_failures:
        receipt_path = ctx.paths.wse_proxy_receipt.parent / 'wse_profile_fit_failure.json'
        _write_json(
            receipt_path,
            {
                'stage': 'centerline_wse_proxy',
                'profile_contract_version': _WSE_PROFILE_CONTRACT_VERSION,
                'wse_profile_grouping_fields': [c for c in _WSE_PROFILE_GROUP_FIELDS if c in out.columns],
                'canonical_measured_source': str(canonical_measured),
                'wse_support_source': _jsonable(support_source_meta),
                'failure_count': int(len(fit_failures)),
                'failures': fit_failures,
            },
        )
        first = fit_failures[0]
        raise RuntimeError(
            'river_workflow_wse_profile_fit_failed:'
            f"reason={first.get('profile_fit', {}).get('reason')}:"
            f"group={first.get('group_key')}:"
            f"summary={receipt_path}"
        )

    finite = int(np.count_nonzero(np.isfinite(pd.to_numeric(out['wse_proxy_z_m'], errors='coerce').to_numpy(dtype=float))))
    if finite == 0:
        raise RuntimeError('river_workflow_wse_profile_all_nan')

    science_summary = _wse_science_summary(out)
    science_summary.update({
        'profile_contract_version': _WSE_PROFILE_CONTRACT_VERSION,
        'active_wse_construction': 'one_path_bank_edge_anchor_residual_monotone_profile',
        'wse_profile_grouping_fields': [c for c in _WSE_PROFILE_GROUP_FIELDS if c in out.columns],
        'wse_support_source': _jsonable(support_source_meta),
        'profile_group_count': int(len(profile_groups)),
        'profile_groups': profile_groups,
        'direction_checks': direction_checks,
        'construction_guard': {
            'status': 'passed_one_path_profile_validation',
            'legacy_flatness_guard': 'disabled_replaced_by_monotone_profile_contract',
            'canonical_measured_source': str(canonical_measured),
        },
        'downstream_handoff_contract': {
            'status': 'ready_for_offset_backbone_stages',
            'wse_role': 'reference_surface_only',
            'expected_observed_offset_formula': 'observed_offset_m = wse_proxy_z_m - authoritative_bed_z_m',
            'expected_backbone_formula': 'bed_backbone_z_m = wse_proxy_z_m - offset_modeled_m',
            'bed_structure_owner': 'modeled_offset_and_backbone_stages',
            'required_columns': list(_WSE_REQUIRED_OUTPUT_COLUMNS),
        },
    })

    artifact_paths = _write_wse_audit_artifacts(
        ctx,
        out,
        canonical_measured=canonical_measured,
        centerline_result=centerline_result,
        authoritative_result=authoritative_result,
        science_summary=science_summary,
        group_direction_checks=direction_checks,
    )

    if ctx.paths.centerline_wse_proxy_points.exists():
        ctx.paths.centerline_wse_proxy_points.unlink()
    out.to_file(ctx.paths.centerline_wse_proxy_points, driver='GPKG')
    validate_wse_proxy_points_gpkg(ctx.paths.centerline_wse_proxy_points)

    retained_proxy = _wse_artifact_reports_dir(ctx) / ctx.paths.centerline_wse_proxy_points.name
    _copy_gpkg(ctx.paths.centerline_wse_proxy_points, retained_proxy)
    artifact_paths['retained_centerline_wse_proxy_points'] = retained_proxy

    receipt_payload = {
        'stage': 'centerline_wse_proxy',
        'status': 'ok',
        'profile_contract_version': _WSE_PROFILE_CONTRACT_VERSION,
        'input_centerline': str(centerline_result.centerline_points_path),
        'canonical_measured_source': str(canonical_measured),
        'output_centerline_wse_proxy_points': str(ctx.paths.centerline_wse_proxy_points),
        'record_count': int(len(out)),
        'finite_wse_count': int(finite),
        'wse_support_source': _jsonable(support_source_meta),
        'downstream_handoff_contract': _jsonable(science_summary.get('downstream_handoff_contract')),
        'wse_one_path_stage_contract': str(artifact_paths.get('wse_stage_contract')) if artifact_paths.get('wse_stage_contract') is not None else None,
        'wse_explicit_steps': [
            'build_wse_support',
            'build_wse_trend',
            'build_wse_pre_smooth',
            'build_wse_proxy_final',
        ],
        'science': _jsonable(science_summary),
    }
    _write_json(ctx.paths.wse_proxy_receipt, receipt_payload)

    if artifact_paths.get('wse_artifact_manifest') is not None:
        manifest_path = Path(artifact_paths['wse_artifact_manifest'])
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest_payload['artifacts'] = {str(k): str(v) for k, v in artifact_paths.items() if str(k) != 'wse_artifact_manifest'}
            manifest_path.write_text(json.dumps(_jsonable(manifest_payload), indent=2, sort_keys=True) + "\n", encoding='utf-8')
        except (OSError, ValueError, TypeError):
            pass

    science_summary['explicit_wse_artifacts'] = {str(k): str(v) for k, v in artifact_paths.items()}
    return WSEProxyStageResult(
        centerline_wse_proxy_points_path=ctx.paths.centerline_wse_proxy_points,
        record_count=int(len(out)),
        finite_wse_count=finite,
        science_summary=science_summary,
        centerline_wse_support_points_path=ctx.paths.centerline_wse_support_points,
        centerline_wse_trend_points_path=ctx.paths.centerline_wse_trend_points,
        centerline_wse_pre_smooth_points_path=ctx.paths.centerline_wse_pre_smooth_points,
        wse_current_path_audit_path=artifact_paths.get('wse_current_path_audit'),
        wse_direction_audit_path=artifact_paths.get('wse_direction_audit'),
        wse_proxy_audit_path=artifact_paths.get('wse_proxy_audit'),
        wse_artifact_manifest_path=artifact_paths.get('wse_artifact_manifest'),
        wse_stage_contract_path=artifact_paths.get('wse_stage_contract'),
    )


__all__ = ['WSEProxyStageResult', 'run_wse_proxy_stage']
