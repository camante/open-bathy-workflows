from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from pipeline.river_workflow.river_workflow_context import RiverWorkflowContext
from pipeline.river_workflow.river_workflow_stage_backbone import BackboneStageResult
from pipeline.river_workflow.river_workflow_stage_corridor import CorridorStageResult
from pipeline.river_workflow.river_workflow_stage_grids import GridStageResult
from pipeline.river_workflow.river_workflow_validation import validate_surface_raster


@dataclass(frozen=True)
class SurfaceStageResult:
    river_primary_surface_solve_path: Path
    seeded_pixel_count: int
    finite_surface_count: int
    corridor_pixel_count: int = 0
    dominant_value_fraction: float | None = None
    abrupt_step_count: int = 0
    channel_shape_pixel_count: int = 0
    lateral_current_path_audit_path: Path | None = None
    lateral_artifact_manifest_path: Path | None = None
    surface_contract_path: Path | None = None


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


def _write_text(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _reports_dir(ctx: RiverWorkflowContext) -> Path:
    return ctx.paths.root.parent / 'reports'


def _lateral_artifacts_dir(ctx: RiverWorkflowContext) -> Path:
    return _reports_dir(ctx) / 'lateral_artifacts'


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return str(dst)


def _write_float_raster(path: Path, array: np.ndarray, profile: dict[str, Any], *, nodata: float = -9999.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    for k in ('blockxsize', 'blockysize', 'BLOCKXSIZE', 'BLOCKYSIZE'):
        out_profile.pop(k, None)
    out_profile.update(dtype='float32', nodata=np.float32(nodata), count=1, compress='deflate')
    data = np.asarray(array, dtype=np.float32)
    data = np.where(np.isfinite(data), data, np.float32(nodata)).astype(np.float32)
    if path.exists():
        path.unlink()
    with rasterio.open(path, 'w', **out_profile) as ds:
        ds.write(data, 1)
    return path


def _numeric_summary(values: Any) -> dict[str, Any]:
    vals = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {'finite_count': 0, 'min': None, 'median': None, 'max': None}
    return {
        'finite_count': int(vals.size),
        'min': float(np.nanmin(vals)),
        'median': float(np.nanmedian(vals)),
        'max': float(np.nanmax(vals)),
    }


def _pixel_centers(rows: np.ndarray, cols: np.ndarray, transform: rasterio.Affine) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _build_seed_rasters(backbone: gpd.GeoDataFrame, grid_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize centerline bed and WSE support points onto the solve grid.

    Multiple points in the same pixel are averaged deterministically.  The surface
    stage uses these as centerline samples, then builds a continuous corridor
    surface from weighted nearby samples instead of nearest-neighbor pixel blocks.
    """
    with rasterio.open(grid_path) as ds:
        bed_values = np.full((ds.height, ds.width), np.nan, dtype=np.float32)
        wse_values = np.full((ds.height, ds.width), np.nan, dtype=np.float32)
        counts = np.zeros((ds.height, ds.width), dtype=np.int32)
        bed = pd.to_numeric(backbone['bed_backbone_z_m'], errors='coerce').to_numpy(dtype=float)
        if 'wse_proxy_z_m' in backbone.columns:
            wse = pd.to_numeric(backbone['wse_proxy_z_m'], errors='coerce').to_numpy(dtype=float)
        else:
            wse = np.full((len(backbone),), np.nan, dtype=float)
        for geom, bed_z, wse_z in zip(backbone.geometry, bed, wse, strict=False):
            if geom is None or geom.is_empty or not np.isfinite(bed_z):
                continue
            try:
                row, col = ds.index(float(geom.x), float(geom.y))
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= row < ds.height and 0 <= col < ds.width:
                n = counts[row, col]
                if n == 0:
                    bed_values[row, col] = np.float32(bed_z)
                    wse_values[row, col] = np.float32(wse_z) if np.isfinite(wse_z) else np.nan
                else:
                    bed_values[row, col] = np.float32((float(bed_values[row, col]) * n + float(bed_z)) / (n + 1))
                    if np.isfinite(wse_z):
                        if np.isfinite(float(wse_values[row, col])):
                            wse_values[row, col] = np.float32((float(wse_values[row, col]) * n + float(wse_z)) / (n + 1))
                        else:
                            wse_values[row, col] = np.float32(wse_z)
                counts[row, col] += 1
    return bed_values, wse_values, counts


def _idw_from_seed_pixels(
    *,
    seed_values: np.ndarray,
    seed_mask: np.ndarray,
    target_mask: np.ndarray,
    transform: rasterio.Affine,
    k_neighbors: int = 8,
) -> np.ndarray:
    """Interpolate seed values across target pixels using local inverse-distance weights."""
    seed_rows, seed_cols = np.nonzero(seed_mask)
    target_rows, target_cols = np.nonzero(target_mask)
    out = np.full(seed_values.shape, np.nan, dtype=np.float32)
    if len(seed_rows) == 0 or len(target_rows) == 0:
        return out
    seed_x, seed_y = _pixel_centers(seed_rows, seed_cols, transform)
    target_x, target_y = _pixel_centers(target_rows, target_cols, transform)
    seed_xy = np.column_stack((seed_x, seed_y))
    target_xy = np.column_stack((target_x, target_y))
    values = seed_values[seed_rows, seed_cols].astype(np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    seed_xy = seed_xy[finite]
    values = values[finite]
    k = int(max(1, min(k_neighbors, len(values))))
    tree = cKDTree(seed_xy)
    distances, indices = tree.query(target_xy, k=k)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    exact = distances <= 1.0e-9
    weights = np.zeros_like(distances, dtype=np.float64)
    exact_any = exact.any(axis=1)
    if np.any(exact_any):
        weights[exact_any] = exact[exact_any].astype(np.float64)
    if np.any(~exact_any):
        d = np.maximum(distances[~exact_any], 1.0e-6)
        weights[~exact_any] = 1.0 / (d * d)
    weight_sum = np.sum(weights, axis=1)
    valid = weight_sum > 0.0
    interpolated = np.full((len(target_rows),), np.nan, dtype=np.float64)
    interpolated[valid] = np.sum(weights[valid] * values[indices[valid]], axis=1) / weight_sum[valid]
    out[target_rows, target_cols] = interpolated.astype(np.float32)
    return out


def _channel_shape_surface(
    *,
    corridor: np.ndarray,
    bed_centerline: np.ndarray,
    wse_centerline: np.ndarray,
    seed_mask: np.ndarray,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> tuple[np.ndarray, dict[str, float | int | None], dict[str, np.ndarray]]:
    """Create a simple channel-shaped surface from centerline bed and WSE profiles.

    The previous implementation assigned every corridor pixel the value of the
    nearest centerline seed.  That produced visible blocky bands.  This function
    keeps the same linear stage contract, but uses a continuous longitudinal bed
    profile plus a lateral centerline-to-bank fraction inside the corridor.
    """
    # Distance to nearest centerline seed inside the corridor.  Zero at seeded
    # centerline pixels and increasing laterally/longitudinally away from seeds.
    distance_to_seed = distance_transform_edt(~seed_mask, sampling=(pixel_size_y_m, pixel_size_x_m)).astype(np.float32)
    # Distance to nearest non-corridor pixel.  This is largest near the channel
    # interior and approaches one pixel near the bank/corridor boundary.
    distance_to_bank = distance_transform_edt(corridor, sampling=(pixel_size_y_m, pixel_size_x_m)).astype(np.float32)
    denom = distance_to_seed + distance_to_bank
    lateral_fraction = np.zeros(corridor.shape, dtype=np.float32)
    valid_fraction = corridor & (denom > 0.0)
    lateral_fraction[valid_fraction] = np.clip(distance_to_seed[valid_fraction] / denom[valid_fraction], 0.0, 1.0).astype(np.float32)

    bed = bed_centerline.astype(np.float32)
    wse = wse_centerline.astype(np.float32)
    relief = wse - bed
    relief = np.where(np.isfinite(relief) & (relief > 0.0), relief, 0.0).astype(np.float32)
    # Controlled smoothstep taper: preserve the centerline/backbone value,
    # then transition monotonically and smoothly toward the bank/WSE proxy.
    # This remains deterministic and parameter-light while avoiding abrupt
    # near-bank ramps from a single power curve.
    lf = np.clip(lateral_fraction.astype(np.float32), 0.0, 1.0)
    taper = (lf * lf * (np.float32(3.0) - np.float32(2.0) * lf)).astype(np.float32)
    shaped = bed + relief * taper
    out = np.full(corridor.shape, np.float32(-9999.0), dtype=np.float32)
    take = corridor & np.isfinite(shaped)
    out[take] = shaped[take].astype(np.float32)
    summary = _surface_quality_summary(out=out, nodata=np.float32(-9999.0), corridor=corridor, seed_mask=seed_mask, lateral_fraction=lateral_fraction)
    lateral_debug = {
        'distance_to_centerline_m': distance_to_seed.astype(np.float32),
        'distance_to_bank_m': distance_to_bank.astype(np.float32),
        'lateral_normalized_position': lateral_fraction.astype(np.float32),
        'lateral_taper_weight': taper.astype(np.float32),
    }
    return out, summary, lateral_debug


def _surface_quality_summary(*, out: np.ndarray, nodata: np.float32, corridor: np.ndarray, seed_mask: np.ndarray, lateral_fraction: np.ndarray) -> dict[str, float | int | None]:
    finite = np.isfinite(out) & (out != nodata)
    values = out[finite]
    if values.size:
        rounded = np.round(values.astype(np.float64), 2)
        _, counts = np.unique(rounded, return_counts=True)
        dominant_fraction = float(counts.max() / counts.sum()) if counts.sum() else None
        value_min = float(np.nanmin(values))
        value_max = float(np.nanmax(values))
    else:
        dominant_fraction = None
        value_min = None
        value_max = None
    abrupt_step_count = 0
    if np.any(finite):
        horizontal = finite[:, 1:] & finite[:, :-1] & (np.abs(out[:, 1:] - out[:, :-1]) > 1.0)
        vertical = finite[1:, :] & finite[:-1, :] & (np.abs(out[1:, :] - out[:-1, :]) > 1.0)
        abrupt_step_count = int(np.count_nonzero(horizontal) + np.count_nonzero(vertical))
    return {
        'corridor_pixel_count': int(np.count_nonzero(corridor)),
        'seeded_pixel_count': int(np.count_nonzero(seed_mask)),
        'finite_surface_count': int(np.count_nonzero(finite)),
        'channel_shape_pixel_count': int(np.count_nonzero(finite & (lateral_fraction > 0.0))),
        'dominant_value_fraction': dominant_fraction,
        'surface_min_m': value_min,
        'surface_max_m': value_max,
        'abrupt_step_count_gt_1m': abrupt_step_count,
        'surface_method': 'idw_longitudinal_profile_plus_smoothstep_lateral_taper',
    }


def _write_surface_summary(ctx: RiverWorkflowContext, summary: dict[str, float | int | None]) -> None:
    ctx.paths.receipts_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.paths.receipts_dir / 'river_primary_surface_summary.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')


def _build_channel_width_points(
    *,
    backbone: gpd.GeoDataFrame,
    grid_path: Path,
    distance_to_bank_m: np.ndarray,
) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []
    with rasterio.open(grid_path) as ds:
        for idx, row in backbone.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                r, c = ds.index(float(geom.x), float(geom.y))
            except (TypeError, ValueError, OverflowError):
                continue
            if not (0 <= r < ds.height and 0 <= c < ds.width):
                continue
            half_width = float(distance_to_bank_m[r, c]) if np.isfinite(distance_to_bank_m[r, c]) else np.nan
            rec: dict[str, Any] = {
                'source_index': int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                'half_width_m': half_width if np.isfinite(half_width) else None,
                'channel_width_m': float(2.0 * half_width) if np.isfinite(half_width) else None,
                'width_source': 'corridor_distance_transform_at_backbone_point',
                'width_confidence': 'audit_only',
                'geometry': geom,
            }
            for field in ('component_id', 'reach_id', 'levelpath_id', 'station_m', 'station_downstream_m', 'bed_backbone_z_m', 'wse_proxy_z_m', 'offset_modeled_m', 'backbone_support_class'):
                if field in backbone.columns:
                    value = row.get(field)
                    if isinstance(value, (np.integer,)):
                        value = int(value)
                    elif isinstance(value, (np.floating,)):
                        value = float(value) if np.isfinite(value) else None
                    rec[field] = value
            rows.append(rec)
    if rows:
        return gpd.GeoDataFrame(rows, geometry='geometry', crs=backbone.crs)
    return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=backbone.crs)


def _nearest_neighbor_xy(xs: np.ndarray, ys: np.ndarray, idx: int) -> tuple[float, float] | None:
    if xs.size < 2:
        return None
    if idx == 0:
        j = 1
        return float(xs[j] - xs[idx]), float(ys[j] - ys[idx])
    if idx == xs.size - 1:
        j = idx - 1
        return float(xs[idx] - xs[j]), float(ys[idx] - ys[j])
    return float(xs[idx + 1] - xs[idx - 1]), float(ys[idx + 1] - ys[idx - 1])


def _build_lateral_cross_section_audit(
    *,
    backbone: gpd.GeoDataFrame,
    grid_path: Path,
    surface: np.ndarray,
    corridor: np.ndarray,
    lateral_debug: dict[str, np.ndarray],
    max_sections: int = 80,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Build diagnostic cross-channel sections from backbone points.

    These sections are audit-only products. They estimate a local normal from
    neighboring backbone points, sample the constructed surface across the
    corridor, and report whether the centerline/backbone value is preserved.
    """
    if backbone.empty:
        empty = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=backbone.crs)
        return empty, {'section_count': 0, 'centerline_preserved_count': 0, 'surface_leak_count': 0}
    station_col = 'station_downstream_m' if 'station_downstream_m' in backbone.columns else 'station_m'
    work = backbone.copy()
    work['_audit_station'] = pd.to_numeric(work[station_col], errors='coerce') if station_col in work.columns else np.arange(len(work), dtype=float)
    if 'component_id' not in work.columns:
        work['component_id'] = 'component_0'
    rows: list[dict[str, Any]] = []
    with rasterio.open(grid_path) as ds:
        transform = ds.transform
        pixel_size = float(max(abs(transform.a), abs(transform.e)))
        for component_id, comp in work.groupby('component_id', dropna=False):
            comp = comp.sort_values('_audit_station', kind='mergesort')
            n = len(comp)
            if n == 0:
                continue
            take_positions = np.arange(n) if n <= max_sections else np.unique(np.linspace(0, n - 1, max_sections, dtype=int))
            xs = comp.geometry.x.to_numpy(dtype=float)
            ys = comp.geometry.y.to_numpy(dtype=float)
            for pos_i in take_positions:
                pos = int(pos_i)
                geom = comp.geometry.iloc[pos]
                if geom is None or geom.is_empty:
                    continue
                tangent = _nearest_neighbor_xy(xs, ys, pos)
                if tangent is None:
                    continue
                tx, ty = tangent
                mag = float(np.hypot(tx, ty))
                if not np.isfinite(mag) or mag <= 0.0:
                    continue
                nx, ny = -ty / mag, tx / mag
                try:
                    r, c = ds.index(float(geom.x), float(geom.y))
                except (TypeError, ValueError, OverflowError):
                    continue
                if not (0 <= r < surface.shape[0] and 0 <= c < surface.shape[1]):
                    continue
                half_width = float(lateral_debug['distance_to_bank_m'][r, c]) if np.isfinite(lateral_debug['distance_to_bank_m'][r, c]) else np.nan
                if not np.isfinite(half_width) or half_width <= 0.0:
                    continue
                half_width = float(min(max(half_width, pixel_size), 250.0))
                sample_values: list[float] = []
                corridor_hits = 0
                outside_finite = 0
                for t in np.linspace(-1.0, 1.0, 25):
                    sx = float(geom.x) + nx * half_width * float(t)
                    sy = float(geom.y) + ny * half_width * float(t)
                    try:
                        sr, sc = ds.index(sx, sy)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if not (0 <= sr < surface.shape[0] and 0 <= sc < surface.shape[1]):
                        continue
                    val = float(surface[sr, sc])
                    in_corridor = bool(corridor[sr, sc])
                    if in_corridor:
                        corridor_hits += 1
                    if np.isfinite(val) and val != -9999.0:
                        sample_values.append(val)
                        if not in_corridor:
                            outside_finite += 1
                center_surface = float(surface[r, c]) if np.isfinite(surface[r, c]) and float(surface[r, c]) != -9999.0 else np.nan
                raw_bed = comp.iloc[pos].get('bed_backbone_z_m', np.nan)
                try:
                    backbone_bed = float(raw_bed)
                except (TypeError, ValueError, OverflowError):
                    backbone_bed = np.nan
                centerline_preserved = bool(np.isfinite(center_surface) and np.isfinite(backbone_bed) and abs(center_surface - backbone_bed) <= 1.0e-4)
                x0 = float(geom.x) - nx * half_width
                y0 = float(geom.y) - ny * half_width
                x1 = float(geom.x) + nx * half_width
                y1 = float(geom.y) + ny * half_width
                rows.append({
                    'component_id': str(component_id),
                    'station_m': float(comp.iloc[pos].get('station_m', np.nan)) if 'station_m' in comp.columns else None,
                    'station_downstream_m': float(comp.iloc[pos].get('station_downstream_m', np.nan)) if 'station_downstream_m' in comp.columns else None,
                    'half_width_m': half_width,
                    'channel_width_m': half_width * 2.0,
                    'centerline_bed_z_m': backbone_bed if np.isfinite(backbone_bed) else None,
                    'centerline_surface_z_m': center_surface if np.isfinite(center_surface) else None,
                    'sample_min_z_m': float(np.nanmin(sample_values)) if sample_values else None,
                    'sample_max_z_m': float(np.nanmax(sample_values)) if sample_values else None,
                    'sample_count': int(len(sample_values)),
                    'corridor_sample_count': int(corridor_hits),
                    'outside_finite_count': int(outside_finite),
                    'centerline_preserved': centerline_preserved,
                    'surface_leaks_outside_domain': bool(outside_finite > 0),
                    'geometry': LineString([(x0, y0), (x1, y1)]),
                })
    if rows:
        gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs=backbone.crs)
    else:
        gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=backbone.crs)
    summary = {
        'section_count': int(len(gdf)),
        'centerline_preserved_count': int(gdf['centerline_preserved'].sum()) if not gdf.empty and 'centerline_preserved' in gdf.columns else 0,
        'surface_leak_count': int(gdf['surface_leaks_outside_domain'].sum()) if not gdf.empty and 'surface_leaks_outside_domain' in gdf.columns else 0,
        'channel_width_summary_m': _numeric_summary(gdf['channel_width_m']) if not gdf.empty and 'channel_width_m' in gdf.columns else _numeric_summary([]),
    }
    return gdf, summary

def _write_lateral_audit_products(
    ctx: RiverWorkflowContext,
    *,
    profile: dict[str, Any],
    backbone: gpd.GeoDataFrame,
    grid_path: Path,
    corridor_result: CorridorStageResult,
    surface_summary: dict[str, Any],
    lateral_debug: dict[str, np.ndarray],
    surface: np.ndarray,
    corridor: np.ndarray,
) -> tuple[Path, Path]:
    reports = _reports_dir(ctx)
    artifacts = _lateral_artifacts_dir(ctx)
    artifacts.mkdir(parents=True, exist_ok=True)

    dist_centerline_path = _write_float_raster(artifacts / 'river_lateral_distance_to_centerline.tif', lateral_debug['distance_to_centerline_m'], profile)
    dist_bank_path = _write_float_raster(artifacts / 'river_lateral_distance_to_bank.tif', lateral_debug['distance_to_bank_m'], profile)
    norm_path = _write_float_raster(artifacts / 'river_lateral_normalized_position.tif', lateral_debug['lateral_normalized_position'], profile)
    taper_path = _write_float_raster(artifacts / 'river_lateral_taper_weight.tif', lateral_debug['lateral_taper_weight'], profile)

    width_gdf = _build_channel_width_points(
        backbone=backbone,
        grid_path=grid_path,
        distance_to_bank_m=lateral_debug['distance_to_bank_m'],
    )
    width_path = artifacts / 'river_channel_width_points.gpkg'
    if width_path.exists():
        width_path.unlink()
    width_gdf.to_file(width_path, driver='GPKG')

    xs_gdf, xs_summary = _build_lateral_cross_section_audit(
        backbone=backbone,
        grid_path=grid_path,
        surface=surface,
        corridor=corridor,
        lateral_debug=lateral_debug,
    )
    xs_path = artifacts / 'lateral_audit_cross_sections.gpkg'
    if xs_path.exists():
        xs_path.unlink()
    xs_gdf.to_file(xs_path, driver='GPKG')

    retained_surface = artifacts / ctx.paths.river_primary_surface_solve.name
    _copy_if_exists(ctx.paths.river_primary_surface_solve, retained_surface)

    finite_lateral = lateral_debug['lateral_normalized_position'][np.isfinite(lateral_debug['lateral_normalized_position'])]
    width_vals = pd.to_numeric(width_gdf.get('channel_width_m'), errors='coerce').to_numpy(dtype=float) if not width_gdf.empty else np.asarray([], dtype=float)
    manifest = {
        'artifact_type': 'lateral_channel_coordinate_audit',
        'stage': 'river_primary_surface_solve',
        'behavior_changed': True,
        'science_phase': 'v818_controlled_lateral_taper_and_cross_section_audit',
        'working_surface_path': ctx.paths.river_primary_surface_solve,
        'working_corridor_mask_path': corridor_result.river_corridor_solve_path,
        'retained_artifacts': {
            'river_lateral_distance_to_centerline': dist_centerline_path,
            'river_lateral_distance_to_bank': dist_bank_path,
            'river_lateral_normalized_position': norm_path,
            'river_lateral_taper_weight': taper_path,
            'river_channel_width_points': width_path,
            'lateral_audit_cross_sections': xs_path,
            'river_primary_surface_solve': retained_surface if retained_surface.exists() else None,
        },
        'surface_summary': surface_summary,
        'lateral_position_summary': _numeric_summary(finite_lateral),
        'channel_width_summary_m': _numeric_summary(width_vals),
        'cross_section_summary': xs_summary,
    }
    manifest_path = _write_json(artifacts / 'lateral_artifact_manifest.json', manifest)

    audit_path = reports / 'LATERAL_CURRENT_PATH_AUDIT.txt'
    lines = [
        'Lateral channel-shape taper and cross-section audit',
        '================================================',
        '',
        'Stage: river_primary_surface_solve',
        '',
        'Controlled construction path:',
        '  river_centerline_bed_backbone_points.gpkg',
        '      -> rasterize centerline bed/WSE seed pixels on solve grid',
        '      -> IDW interpolate longitudinal bed and WSE profiles across the corridor',
        '      -> compute lateral position from distance to seed pixels and distance to corridor edge',
        '      -> apply controlled smoothstep lateral taper toward the WSE edge',
        '      -> river_primary_surface_solve.tif',
        '',
        'Science invariant for this phase:',
        '  Preserve the centerline/backbone value, taper smoothly toward bank/WSE edge,',
        '  keep guidance inside the river corridor, and leave authoritative lock behavior unchanged.',
        '',
        f'Working corridor mask: {corridor_result.river_corridor_solve_path}',
        f'Working primary surface: {ctx.paths.river_primary_surface_solve}',
        f'Retained lateral artifact manifest: {manifest_path}',
        '',
        'Retained lateral artifacts:',
        f'  distance_to_centerline: {dist_centerline_path}',
        f'  distance_to_bank: {dist_bank_path}',
        f'  normalized_position: {norm_path}',
        f'  taper_weight: {taper_path}',
        f'  channel_width_points: {width_path}',
        f'  lateral_cross_sections: {xs_path}',
        f'  primary_surface_copy: {retained_surface}',
        '',
        'Surface summary:',
        json.dumps(_jsonable(surface_summary), indent=2, sort_keys=True),
        '',
        'Lateral normalized position summary:',
        json.dumps(_jsonable(_numeric_summary(finite_lateral)), indent=2, sort_keys=True),
        '',
        'Channel width summary (m):',
        json.dumps(_jsonable(_numeric_summary(width_vals)), indent=2, sort_keys=True),
        '',
        'Cross-section audit summary:',
        json.dumps(_jsonable(xs_summary), indent=2, sort_keys=True),
    ]
    _write_text(audit_path, lines)

    cross_section_audit_path = reports / 'LATERAL_CROSS_SECTION_AUDIT.txt'
    cross_lines = [
        'Lateral cross-section audit',
        '===========================',
        '',
        'Purpose:',
        '  Diagnostic cross sections summarize how the controlled lateral taper behaves across the river corridor.',
        '  They do not participate in terrain generation.',
        '',
        f'Cross-section artifact: {xs_path}',
        '',
        'Cross-section summary:',
        json.dumps(_jsonable(xs_summary), indent=2, sort_keys=True),
        '',
        'Expected checks:',
        '  centerline_preserved_count should generally match section_count unless the nearest raster cell is not a seed cell.',
        '  surface_leak_count should be zero; nonzero values indicate finite guidance outside the corridor sample.',
    ]
    _write_text(cross_section_audit_path, cross_lines)
    return audit_path, manifest_path



def _write_surface_contract_receipt(
    path: Path,
    *,
    surface_path: Path,
    corridor_path: Path,
    backbone_path: Path,
    surface: np.ndarray,
    corridor: np.ndarray,
    bed_seed_values: np.ndarray,
    seed_mask: np.ndarray,
    nodata: float,
    surface_summary: dict[str, Any],
) -> Path:
    """Write and enforce the primary-surface handoff contract.

    This is deliberately a small contract for the one-path surface stage: finite
    guidance must stay inside the canonical corridor, and the centerline/backbone
    seed pixels that fall inside that corridor must be preserved exactly.  This
    keeps lateral spreading subordinate to the canonical bed backbone instead of
    letting raster interpolation mutate the controlling centerline profile.
    """
    finite_surface = np.isfinite(surface) & (surface != np.float32(nodata))
    finite_outside_corridor = finite_surface & ~corridor
    seed_in_corridor = seed_mask & corridor
    seed_outside_corridor = seed_mask & ~corridor
    seed_mismatch = np.zeros(surface.shape, dtype=bool)
    seed_abs_diff = np.zeros(surface.shape, dtype=np.float32)
    if np.any(seed_in_corridor):
        seed_abs_diff[seed_in_corridor] = np.abs(
            surface[seed_in_corridor].astype(np.float32) - bed_seed_values[seed_in_corridor].astype(np.float32)
        )
        seed_mismatch[seed_in_corridor] = seed_abs_diff[seed_in_corridor] > np.float32(1.0e-4)
    payload = {
        'stage': 'river_primary_surface_contract',
        'status': 'PASS',
        'surface_path': str(surface_path),
        'corridor_path': str(corridor_path),
        'backbone_path': str(backbone_path),
        'surface_method': surface_summary.get('surface_method'),
        'corridor_pixel_count': int(np.count_nonzero(corridor)),
        'finite_surface_count': int(np.count_nonzero(finite_surface)),
        'finite_outside_corridor_count': int(np.count_nonzero(finite_outside_corridor)),
        'seed_pixel_count': int(np.count_nonzero(seed_mask)),
        'seed_in_corridor_count': int(np.count_nonzero(seed_in_corridor)),
        'seed_outside_corridor_count': int(np.count_nonzero(seed_outside_corridor)),
        'seed_preserved_count': int(np.count_nonzero(seed_in_corridor & ~seed_mismatch)),
        'seed_mismatch_count': int(np.count_nonzero(seed_mismatch)),
        'seed_max_abs_diff_m': float(np.nanmax(seed_abs_diff[seed_in_corridor])) if np.any(seed_in_corridor) else None,
        'invariants': {
            'finite_guidance_stays_inside_canonical_corridor': bool(not np.any(finite_outside_corridor)),
            'centerline_backbone_seed_pixels_preserved': bool(np.any(seed_in_corridor) and not np.any(seed_mismatch)),
            'surface_is_written_once_from_backbone_to_corridor_taper': True,
        },
    }
    failures: list[str] = []
    if not np.any(corridor):
        failures.append('empty_corridor')
    if not np.any(finite_surface):
        failures.append('empty_surface')
    if np.any(finite_outside_corridor):
        failures.append('finite_surface_outside_corridor')
    if not np.any(seed_in_corridor):
        failures.append('no_backbone_seed_pixels_inside_corridor')
    if np.any(seed_mismatch):
        failures.append('centerline_backbone_seed_pixels_changed')
    if failures:
        payload['status'] = 'FAIL'
        payload['failures'] = failures
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if failures:
        raise RuntimeError('river_workflow_surface_contract_failed:' + ','.join(failures) + f':summary={path}')
    return path

def run_surface_stage(
    ctx: RiverWorkflowContext,
    grid_result: GridStageResult,
    corridor_result: CorridorStageResult,
    backbone_result: BackboneStageResult,
) -> SurfaceStageResult:
    backbone = gpd.read_file(backbone_result.centerline_bed_backbone_points_path)
    if backbone.empty:
        raise RuntimeError('river_workflow_surface_missing_backbone')
    with rasterio.open(corridor_result.river_corridor_solve_path) as corridor_ds:
        corridor = corridor_ds.read(1) > 0
    if not np.any(corridor):
        raise RuntimeError('river_workflow_surface_empty_corridor')
    bed_seed_values, wse_seed_values, seed_counts = _build_seed_rasters(backbone, grid_result.solve_grid_template_path)
    seed_mask = (seed_counts > 0) & np.isfinite(bed_seed_values)
    if not np.any(seed_mask):
        raise RuntimeError('river_workflow_surface_no_seed_pixels')
    with rasterio.open(grid_result.solve_grid_template_path) as template_ds:
        transform = template_ds.transform
        pixel_size_x_m = float(abs(transform.a))
        pixel_size_y_m = float(abs(transform.e))
        profile = template_ds.profile.copy()
    bed_centerline = _idw_from_seed_pixels(
        seed_values=bed_seed_values,
        seed_mask=seed_mask,
        target_mask=corridor,
        transform=transform,
    )
    wse_seed_mask = seed_mask & np.isfinite(wse_seed_values)
    if np.any(wse_seed_mask):
        wse_centerline = _idw_from_seed_pixels(
            seed_values=wse_seed_values,
            seed_mask=wse_seed_mask,
            target_mask=corridor,
            transform=transform,
        )
    else:
        # No WSE field should be rare because the backbone carries WSE inputs.
        # Keep the stage honest: the surface can still be a centerline-bed-only
        # guide, but the summary will show zero channel-shape pixels.
        wse_centerline = bed_centerline.copy()
    out, summary, lateral_debug = _channel_shape_surface(
        corridor=corridor,
        bed_centerline=bed_centerline,
        wse_centerline=wse_centerline,
        seed_mask=seed_mask,
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
    )
    for k in ('blockxsize', 'blockysize', 'BLOCKXSIZE', 'BLOCKYSIZE'):
        profile.pop(k, None)
    profile.update(dtype='float32', nodata=np.float32(-9999.0), count=1, compress='deflate')
    ctx.paths.river_primary_surface_solve.parent.mkdir(parents=True, exist_ok=True)
    if ctx.paths.river_primary_surface_solve.exists():
        ctx.paths.river_primary_surface_solve.unlink()
    with rasterio.open(ctx.paths.river_primary_surface_solve, 'w', **profile) as out_ds:
        out_ds.write(out, 1)
    validate_surface_raster(raster_path=ctx.paths.river_primary_surface_solve, template_path=grid_result.solve_grid_template_path)
    _write_surface_summary(ctx, summary)
    surface_contract_path = _write_surface_contract_receipt(
        ctx.paths.receipts_dir / 'river_primary_surface_contract.json',
        surface_path=ctx.paths.river_primary_surface_solve,
        corridor_path=corridor_result.river_corridor_solve_path,
        backbone_path=backbone_result.centerline_bed_backbone_points_path,
        surface=out,
        corridor=corridor,
        bed_seed_values=bed_seed_values,
        seed_mask=seed_mask,
        nodata=-9999.0,
        surface_summary=summary,
    )
    lateral_audit_path, lateral_manifest_path = _write_lateral_audit_products(
        ctx,
        profile=profile,
        backbone=backbone,
        grid_path=grid_result.solve_grid_template_path,
        corridor_result=corridor_result,
        surface_summary=summary,
        lateral_debug=lateral_debug,
        surface=out,
        corridor=corridor,
    )
    return SurfaceStageResult(
        river_primary_surface_solve_path=ctx.paths.river_primary_surface_solve,
        seeded_pixel_count=int(summary['seeded_pixel_count'] or 0),
        finite_surface_count=int(summary['finite_surface_count'] or 0),
        corridor_pixel_count=int(summary['corridor_pixel_count'] or 0),
        dominant_value_fraction=summary['dominant_value_fraction'],
        abrupt_step_count=int(summary['abrupt_step_count_gt_1m'] or 0),
        channel_shape_pixel_count=int(summary['channel_shape_pixel_count'] or 0),
        lateral_current_path_audit_path=lateral_audit_path,
        lateral_artifact_manifest_path=lateral_manifest_path,
        surface_contract_path=surface_contract_path,
    )


__all__ = ['SurfaceStageResult', 'run_surface_stage']
