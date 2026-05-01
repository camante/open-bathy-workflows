from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Any

import logging
import math
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RiverScaffoldSelection:
    flows: object
    polygons: object
    metadata: dict[str, Any]


def _find_col(gdf, names: Iterable[str]) -> Optional[str]:
    lookup = {str(c).lower(): c for c in getattr(gdf, 'columns', [])}
    for name in names:
        hit = lookup.get(str(name).lower())
        if hit is not None:
            return hit
    return None


def _safe_to_crs(gdf, target_crs):
    if gdf is None or getattr(gdf, 'empty', True) or target_crs is None:
        return gdf
    if getattr(gdf, 'crs', None) is None:
        raise RuntimeError('river_structured_scaffold_missing_input_crs')
    if str(gdf.crs) == str(target_crs):
        return gdf
    return gdf.to_crs(target_crs)


def _geometry_union(geoms):
    geoms = [g for g in geoms if g is not None and not getattr(g, 'is_empty', True)]
    if not geoms:
        return None
    try:
        import geopandas as gpd  # noqa: F401
    except Exception:
        pass
    try:
        from shapely.ops import unary_union
        return unary_union(geoms)
    except Exception:
        merged = geoms[0]
        for geom in geoms[1:]:
            merged = merged.union(geom)
        return merged


def _as_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _component_field(flows_gdf) -> str | None:
    return _find_col(flows_gdf, ('component_id', 'levelpath_id', 'levelpathi', 'levelpathid', 'reach_id', 'nhdplusid', 'comid'))


def _reach_field(flows_gdf) -> str | None:
    return _find_col(flows_gdf, ('reach_id', 'nhdplusid', 'comid', 'permanent_identifier', 'gnis_id'))


def _stream_order_field(flows_gdf) -> str | None:
    return _find_col(flows_gdf, ('stream_order', 'streamorde', 'streamorder', 'streamord', 'strahler'))


def _length_field(flows_gdf) -> str | None:
    return _find_col(flows_gdf, ('lengthkm', 'length_km', 'len_km', 'length_mi', 'length_m'))


def _sample_distances(length_m: float, spacing_m: float) -> list[float]:
    length_m = float(length_m)
    step = max(float(spacing_m), 1.0)
    if not np.isfinite(length_m) or length_m <= 0.0:
        return []
    dists = list(np.arange(0.0, length_m, step, dtype=float))
    if not dists:
        dists = [0.0]
    if (length_m - dists[-1]) > 0.35 * step:
        dists.append(length_m)
    elif not math.isclose(float(dists[-1]), length_m, rel_tol=0.0, abs_tol=1.0e-6):
        dists.append(length_m)
    return [float(d) for d in dists]


def _iter_line_parts(geom):
    if geom is None or getattr(geom, 'is_empty', True):
        return []
    gtype = getattr(geom, 'geom_type', '')
    if gtype == 'LineString':
        return [geom]
    if gtype == 'MultiLineString':
        return [part for part in geom.geoms if part is not None and not part.is_empty and float(part.length or 0.0) > 0.0]
    return []


def _geometry_sort_key(row) -> tuple[float, float, str]:
    geom = row.geometry
    start_x = start_y = 0.0
    try:
        if getattr(geom, 'geom_type', '') == 'LineString':
            c0 = list(geom.coords)[0]
            start_x = float(c0[0])
            start_y = float(c0[1])
        elif getattr(geom, 'geom_type', '') == 'MultiLineString' and len(geom.geoms):
            c0 = list(geom.geoms[0].coords)[0]
            start_x = float(c0[0])
            start_y = float(c0[1])
    except Exception:
        pass
    return (start_x, start_y, str(getattr(row, 'name', '')))


def _station_bounds_for_row(row, line_len_m: float) -> tuple[float | None, float | None]:
    s0 = None
    s1 = None
    for a, b in (('s_m_from', 's_m_to'), ('s_m_min', 's_m_max'), ('from_measure', 'to_measure'), ('measure_from', 'measure_to')):
        if a in row.index and b in row.index:
            va = pd.to_numeric(row.get(a), errors='coerce')
            vb = pd.to_numeric(row.get(b), errors='coerce')
            if np.isfinite(va) and np.isfinite(vb):
                s0, s1 = float(va), float(vb)
                break
    if s0 is None or s1 is None:
        return None, None
    if not np.isfinite(line_len_m) or line_len_m <= 0.0:
        return None, None
    return s0, s1


def _copy_relevant_fields(row, preferred_component_col: str | None, reach_col: str | None, order_col: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    comp_val = _as_str_or_none(row.get(preferred_component_col)) if preferred_component_col else None
    if comp_val is None:
        comp_val = f'flow_idx:{row.name}'
    out['component_id'] = str(comp_val)
    if 'levelpath_id' in row.index:
        out['levelpath_id'] = row.get('levelpath_id')
    elif 'levelpathi' in row.index:
        out['levelpath_id'] = row.get('levelpathi')
    elif 'levelpathid' in row.index:
        out['levelpath_id'] = row.get('levelpathid')
    else:
        out['levelpath_id'] = str(comp_val)
    if reach_col is not None:
        out['reach_id'] = row.get(reach_col)
    else:
        out['reach_id'] = str(row.name)
    out['source_reach_key'] = str(out.get('reach_id', row.name))
    if order_col is not None:
        out['stream_order'] = row.get(order_col)
    for col in ('mean_width_m', 'width_m', 'from_node', 'to_node', 'network_component_id', 'component_id_source'):
        if col in row.index and col not in out:
            out[col] = row.get(col)
    return out


def _sample_raster_values(raster_path: str | Path, points_gdf, field: str = 'z_m'):
    import rasterio

    out_vals = np.full((len(points_gdf),), np.nan, dtype=np.float32)
    if points_gdf is None or getattr(points_gdf, 'empty', True):
        points_gdf[field] = out_vals
        return points_gdf
    with rasterio.open(raster_path) as ds:
        pts_src = points_gdf
        if getattr(points_gdf, 'crs', None) is not None and ds.crs is not None and str(points_gdf.crs) != str(ds.crs):
            pts_src = points_gdf.to_crs(ds.crs)
        xy = list(zip(pts_src.geometry.x.to_numpy(dtype=float), pts_src.geometry.y.to_numpy(dtype=float)))
        vals = []
        for sample in ds.sample(xy):
            val = float(sample[0]) if np.size(sample) else np.nan
            if ds.nodata is not None and np.isfinite(val) and np.isclose(val, float(ds.nodata)):
                val = np.nan
            vals.append(val)
        out_vals = np.asarray(vals, dtype=np.float32)
    points_gdf[field] = out_vals
    return points_gdf


def _mask_points_to_allowed(points_gdf, allowed_mask: Optional[np.ndarray], transform):
    if points_gdf is None or getattr(points_gdf, 'empty', True) or allowed_mask is None or transform is None:
        return points_gdf
    import rasterio.transform

    mask_bool = np.asarray(allowed_mask, dtype=bool)
    xs = points_gdf.geometry.x.to_numpy(dtype=float)
    ys = points_gdf.geometry.y.to_numpy(dtype=float)
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    keep = (rows >= 0) & (rows < mask_bool.shape[0]) & (cols >= 0) & (cols < mask_bool.shape[1])
    keep[keep] &= mask_bool[rows[keep], cols[keep]]
    return points_gdf.loc[keep].copy()


def _collapse_duplicate_centerline_component_station_rows(points_gdf, logger=None):
    if points_gdf is None or getattr(points_gdf, 'empty', True):
        return points_gdf, 0
    pts = points_gdf.copy()
    pts['component_id'] = pts.get('component_id', 'main')
    pts['component_id'] = pts['component_id'].fillna('main').astype(str)
    pts['station_m'] = pd.to_numeric(pts.get('station_m'), errors='coerce')
    pts = pts.loc[np.isfinite(pts['station_m'])].copy()
    if pts.empty:
        return pts, 0
    pts['_station_key'] = np.round(pts['station_m'].to_numpy(dtype=float), 6)
    duplicate_rows = int(pts.duplicated(subset=['component_id', '_station_key'], keep=False).sum())
    if duplicate_rows == 0:
        return pts.drop(columns=['_station_key'], errors='ignore'), 0
    geom_name = pts.geometry.name
    rows = []
    for _, grp in pts.groupby(['component_id', '_station_key'], sort=False, dropna=False):
        chosen = grp.sort_values(['_line_length_m', '_sample_proj_m'], ascending=[False, True], kind='mergesort').iloc[0] if {'_line_length_m', '_sample_proj_m'}.issubset(grp.columns) else grp.iloc[0]
        rows.append(chosen.drop(labels=['_station_key'], errors='ignore'))
    import geopandas as gpd
    out = gpd.GeoDataFrame(rows, geometry=geom_name, crs=points_gdf.crs).reset_index(drop=True)
    removed = int(len(pts) - len(out))
    if removed > 0 and logger is not None:
        logger.warning('[RIVER][CENTERLINE] Collapsed duplicate component/station rows: removed=%d involved=%d', removed, duplicate_rows)
    return out, removed


def normalize_structured_flow_component_ids(flows_gdf, *, logger=None):
    if flows_gdf is None or getattr(flows_gdf, 'empty', True):
        return flows_gdf, {'component_id_source': 'none', 'component_count_before': 0, 'component_count_after': 0}
    flows = flows_gdf.copy()
    original = flows['component_id'].copy() if 'component_id' in flows.columns else None
    preferred = _component_field(flows)
    vals = []
    for idx, row in flows.iterrows():
        raw = _as_str_or_none(row.get(preferred)) if preferred else None
        vals.append(f'{preferred}:{raw}' if raw is not None else f'flow_idx:{idx}')
    if original is not None and 'network_component_id' not in flows.columns:
        flows['network_component_id'] = original
    flows['component_id'] = pd.Series(vals, index=flows.index, dtype='object')
    meta = {
        'component_id_source': preferred or 'flow_idx',
        'component_count_before': int(pd.Series(original).dropna().astype(str).nunique()) if original is not None else 0,
        'component_count_after': int(pd.Series(vals).dropna().astype(str).nunique()),
    }
    if logger is not None:
        logger.info('[RIVER][CENTERLINE] Component namespace: source=%s components=%d', meta['component_id_source'], meta['component_count_after'])
    return flows, meta


def validate_centerline_station_component_contract(
    centerline_points_gdf,
    *,
    expected_component_count: int = 0,
    expected_component_source: str | None = None,
    logger=None,
    context: str = 'centerline_points',
):
    if centerline_points_gdf is None or getattr(centerline_points_gdf, 'empty', True):
        raise RuntimeError(f'{context}_missing_or_empty')
    pts = centerline_points_gdf.copy()
    if 'station_m' not in pts.columns:
        raise RuntimeError(f'{context}_missing_station_m')
    if 'component_id' not in pts.columns:
        raise RuntimeError(f'{context}_missing_component_id')
    pts['component_id'] = pts['component_id'].fillna('main').astype(str)
    pts['station_m'] = pd.to_numeric(pts['station_m'], errors='coerce')
    pts = pts.loc[np.isfinite(pts['station_m'])].copy()
    duplicate_rows = int(pts.duplicated(subset=['component_id', 'station_m'], keep=False).sum())
    if duplicate_rows:
        raise RuntimeError(f'{context}_duplicate_component_station_rows:{duplicate_rows}')
    component_count = int(pts['component_id'].nunique(dropna=True))
    if int(expected_component_count or 0) > 1 and component_count <= 1:
        raise RuntimeError(f'{context}_coarse_component_namespace: expected_components={expected_component_count} actual_components={component_count}')
    summary = {
        'context': str(context),
        'component_count': component_count,
        'expected_component_count': int(expected_component_count or 0),
        'component_id_source': str(expected_component_source or 'unknown'),
        'duplicate_component_station_rows': duplicate_rows,
    }
    if logger is not None:
        logger.info('[RIVER][CENTERLINE] Station/component contract validated: %s', summary)
    return summary


def build_centerline_points(
    flows_gdf,
    polygons_gdf=None,
    *,
    spacing_m: float = 30.0,
    raster_path: str | Path | None = None,
    role: str = 'centerline_control',
    allowed_mask: Optional[np.ndarray] = None,
    transform=None,
    logger=None,
):
    """Build deterministic stationized centerline points from canonical flowlines.

    This is an active producer for the built-in river workflow.  It does not discover
    alternate sources or synthesize a different river network; it only stationizes the
    canonical flowlines handed to the stage, optionally samples the canonical measured
    raster for diagnostics, and returns one GeoDataFrame consumed by the linear stages.
    """
    import geopandas as gpd

    if flows_gdf is None or getattr(flows_gdf, 'empty', True):
        raise RuntimeError('river_structured_centerline_empty_flows')
    flows = flows_gdf.copy()
    flows = flows[flows.geometry.notnull() & ~flows.geometry.is_empty].copy()
    if flows.empty:
        raise RuntimeError('river_structured_centerline_no_valid_flow_geometry')
    if getattr(flows, 'crs', None) is None:
        raise RuntimeError('river_structured_centerline_flows_missing_crs')

    component_col = _component_field(flows)
    reach_col = _reach_field(flows)
    order_col = _stream_order_field(flows)
    rows: list[dict[str, Any]] = []
    centerline_order = 0

    # Stable grouping prevents per-segment station resets inside one component.
    if component_col is not None:
        group_keys = flows[component_col].map(lambda v: _as_str_or_none(v) or 'unknown').astype(str)
    else:
        group_keys = pd.Series([f'flow_idx:{idx}' for idx in flows.index], index=flows.index, dtype='object')

    for comp, idxs in group_keys.groupby(group_keys, sort=True).groups.items():
        comp_flows = flows.loc[list(idxs)].copy()
        if 's_m_from' in comp_flows.columns:
            comp_flows['_sort_station'] = pd.to_numeric(comp_flows['s_m_from'], errors='coerce')
        elif 's_m_min' in comp_flows.columns:
            comp_flows['_sort_station'] = pd.to_numeric(comp_flows['s_m_min'], errors='coerce')
        else:
            comp_flows['_sort_station'] = np.nan
        comp_flows['_sort_len'] = [float(getattr(g, 'length', 0.0) or 0.0) for g in comp_flows.geometry]
        comp_flows['_sort_key'] = [str(i) for i in comp_flows.index]
        comp_flows = comp_flows.sort_values(['_sort_station', '_sort_key'], na_position='last', kind='mergesort')

        cumulative_m = 0.0
        seen_station_keys: set[float] = set()
        for idx, rec in comp_flows.iterrows():
            parts = _iter_line_parts(rec.geometry)
            if not parts:
                continue
            base_fields = _copy_relevant_fields(rec, component_col, reach_col, order_col)
            # Force the chosen group key as component_id so all segments of a path share one station namespace.
            base_fields['component_id'] = str(comp)
            for part_idx, line in enumerate(parts):
                line_len = float(getattr(line, 'length', 0.0) or 0.0)
                if line_len <= 0.0:
                    continue
                s0, s1 = _station_bounds_for_row(rec, line_len)
                for local_d in _sample_distances(line_len, spacing_m):
                    if s0 is not None and s1 is not None:
                        frac = float(local_d / line_len) if line_len > 0 else 0.0
                        station_m = float(s0 + (s1 - s0) * frac)
                    else:
                        station_m = float(cumulative_m + local_d)
                    station_key = round(float(station_m), 6)
                    if station_key in seen_station_keys:
                        continue
                    seen_station_keys.add(station_key)
                    pt = line.interpolate(float(local_d))
                    row = dict(base_fields)
                    row.update({
                        'point_id': f'cl_{centerline_order:07d}',
                        'station_m': float(station_m),
                        'centerline_order': int(centerline_order),
                        'artifact_role': str(role),
                        'flow_idx': int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                        'part_idx': int(part_idx),
                        '_sample_proj_m': float(local_d),
                        '_line_length_m': float(line_len),
                        'geometry': pt,
                    })
                    rows.append(row)
                    centerline_order += 1
                if s0 is None or s1 is None:
                    cumulative_m += line_len

    if not rows:
        raise RuntimeError('river_structured_centerline_no_points_built')
    out = gpd.GeoDataFrame(rows, geometry='geometry', crs=flows.crs)
    out = _mask_points_to_allowed(out, allowed_mask, transform)
    out, _removed = _collapse_duplicate_centerline_component_station_rows(out, logger=logger)
    if out is None or getattr(out, 'empty', True):
        raise RuntimeError('river_structured_centerline_no_points_after_masking')
    out['point_id'] = [f'cl_{i:07d}' for i in range(len(out))]
    out['centerline_order'] = np.arange(len(out), dtype=int)
    # Preserve monotonicity per component for downstream validation.
    out = out.sort_values(['component_id', 'station_m', 'centerline_order'], kind='mergesort').reset_index(drop=True)
    out['point_id'] = [f'cl_{i:07d}' for i in range(len(out))]
    out['centerline_order'] = np.arange(len(out), dtype=int)
    if raster_path is not None and Path(raster_path).exists():
        out = _sample_raster_values(raster_path, out, field='z_m')
    if logger is not None:
        logger.info('[RIVER][CENTERLINE] Built centerline points: flows=%d points=%d components=%d spacing_m=%.3f', len(flows), len(out), out['component_id'].nunique(), float(spacing_m))
    return out


def select_retained_river_features(
    river_gpkg: str | Path,
    *,
    target_crs=None,
    min_stream_order: int = 3,
    min_length_km: float = 0.25,
    keep_top_components: int = 6,
    nhdarea_layer: str = 'nhdarea_clip',
    flows_layer: str = 'rivers_clip',
):
    import geopandas as gpd

    gpkg = Path(river_gpkg)
    flows = gpd.read_file(gpkg, layer=flows_layer)
    flows = flows[flows.geometry.notnull() & ~flows.geometry.is_empty].copy()
    flows = _safe_to_crs(flows, target_crs)
    layers = set(gpd.list_layers(gpkg)['name'].tolist())
    if nhdarea_layer in layers:
        polygons = gpd.read_file(gpkg, layer=nhdarea_layer)
        polygons = polygons[polygons.geometry.notnull() & ~polygons.geometry.is_empty].copy()
        polygons = _safe_to_crs(polygons, target_crs)
    else:
        polygons = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=flows.crs)

    order_col = _stream_order_field(flows)
    length_col = _length_field(flows)
    keep = np.ones((len(flows),), dtype=bool)
    if order_col is not None:
        keep &= pd.to_numeric(flows[order_col], errors='coerce').fillna(-1).to_numpy(dtype=float) >= float(min_stream_order)
    if length_col is not None:
        length = pd.to_numeric(flows[length_col], errors='coerce').fillna(0).to_numpy(dtype=float)
        # NHD length fields are commonly km; if values look like meters, convert threshold accordingly.
        threshold = float(min_length_km if np.nanmedian(length) < 1000.0 else min_length_km * 1000.0)
        keep &= length >= threshold
    retained = flows.loc[keep].copy()
    if retained.empty:
        retained = flows.copy()
    if polygons is not None and not polygons.empty and not retained.empty:
        try:
            merged = _geometry_union([g.buffer(5.0, cap_style=2) for g in retained.geometry])
            if merged is not None:
                polygons = polygons[polygons.geometry.intersects(merged)].copy()
        except Exception:
            pass
    meta = {
        'input_flow_count': int(len(flows)),
        'retained_flow_count': int(len(retained)),
        'retained_polygon_count': int(len(polygons)) if polygons is not None else 0,
        'min_stream_order': int(min_stream_order),
        'min_length_km': float(min_length_km),
        'keep_top_components': int(keep_top_components),
        'selection_mode': 'deterministic_mainstem_filter',
    }
    return RiverScaffoldSelection(flows=retained, polygons=polygons, metadata=meta)


def _not_active(name: str):
    def _fn(*args, **kwargs):
        raise RuntimeError(f'river_structured_scaffold_function_not_active_in_builtin_linear_workflow:{name}')
    return _fn


# These names are retained for legacy import compatibility only.  The active built-in
# linear workflow currently consumes build_centerline_points from this module; older
# pre-linear routines should fail clearly instead of silently generating alternate products.
build_dense_bank_points = _not_active('build_dense_bank_points')
build_xs_support_points = _not_active('build_xs_support_points')
build_centerline_core_influence = _not_active('build_centerline_core_influence')
rasterize_point_seed_surface = _not_active('rasterize_point_seed_surface')
rasterize_point_chainage_surface = _not_active('rasterize_point_chainage_surface')
_nearest_surface_from_points = _not_active('_nearest_surface_from_points')
_clip_gdf_to_allowed_mask = _not_active('_clip_gdf_to_allowed_mask')


__all__ = [
    'RiverScaffoldSelection',
    'select_retained_river_features',
    'build_dense_bank_points',
    'build_centerline_points',
    'build_xs_support_points',
    'build_centerline_core_influence',
    'rasterize_point_seed_surface',
    'rasterize_point_chainage_surface',
    '_nearest_surface_from_points',
    '_clip_gdf_to_allowed_mask',
    'normalize_structured_flow_component_ids',
    'validate_centerline_station_component_contract',
]
