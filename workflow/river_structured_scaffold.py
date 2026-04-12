from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import logging
import math
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RiverScaffoldSelection:
    flows: object
    polygons: object
    metadata: dict


def _find_col(gdf, names: Iterable[str]) -> Optional[str]:
    lookup = {str(c).lower(): c for c in gdf.columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    return None


def _safe_to_crs(gdf, target_crs):
    if gdf is None or getattr(gdf, "empty", True):
        return gdf
    if target_crs is None:
        return gdf
    if gdf.crs is None:
        raise RuntimeError("Cannot reproject GeoDataFrame with undefined CRS")
    if str(gdf.crs) == str(target_crs):
        return gdf
    return gdf.to_crs(target_crs)


def _geometry_union(geoms):
    if geoms is None:
        return None
    if hasattr(geoms, "union_all"):
        return geoms.union_all()
    if hasattr(geoms, "unary_union"):
        return geoms.unary_union
    from shapely.ops import unary_union
    return unary_union(list(geoms))


def select_retained_river_features(
    river_gpkg: str | Path,
    *,
    target_crs=None,
    min_stream_order: int = 3,
    min_length_km: float = 0.25,
    keep_top_components: int = 6,
    nhdarea_layer: str = "nhdarea_clip",
    flows_layer: str = "rivers_clip",
):
    import geopandas as gpd

    gpkg = Path(river_gpkg)
    flows = gpd.read_file(gpkg, layer=flows_layer)
    flows = flows[flows.geometry.notnull() & ~flows.geometry.is_empty].copy()
    flows = _safe_to_crs(flows, target_crs)

    polygons = None
    layer_names = set(gpd.list_layers(gpkg)["name"].tolist())
    if nhdarea_layer in layer_names:
        polygons = gpd.read_file(gpkg, layer=nhdarea_layer)
        polygons = polygons[polygons.geometry.notnull() & ~polygons.geometry.is_empty].copy()
        polygons = _safe_to_crs(polygons, target_crs)
    else:
        polygons = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(flows, "crs", None))

    flow_count0 = int(len(flows))
    col_order = _find_col(flows, ["streamorde", "streamorder", "streamord", "streamleve"])
    col_len = _find_col(flows, ["lengthkm", "length_km", "len_km"])
    col_comp = _find_col(flows, ["component_id"])

    keep = np.ones(len(flows), dtype=bool)
    short_length_filtered = np.zeros(len(flows), dtype=bool)
    if col_len:
        lengths_km = np.nan_to_num(np.asarray(flows[col_len], dtype=float), nan=0.0)
        short_length_filtered = lengths_km < float(min_length_km)
        keep &= ~short_length_filtered
    if col_order:
        keep &= np.nan_to_num(np.asarray(flows[col_order], dtype=float), nan=-1.0) >= float(min_stream_order)
    flows_attr = flows.loc[keep].copy()

    component_lengths = {}
    if col_comp and not flows.empty:
        if col_len:
            tmp = flows[[col_comp, col_len]].copy()
            tmp[col_len] = np.nan_to_num(np.asarray(tmp[col_len], dtype=float), nan=0.0)
            component_lengths = tmp.groupby(col_comp)[col_len].sum().to_dict()
        else:
            component_lengths = flows.groupby(col_comp).size().to_dict()

    top_components = set()
    if component_lengths:
        top_components = set([k for k, _ in sorted(component_lengths.items(), key=lambda kv: kv[1], reverse=True)[: max(int(keep_top_components), 1)]])

    if flows_attr.empty and top_components:
        flows_attr = flows[flows[col_comp].isin(list(top_components))].copy() if col_comp else flows.copy()

    rescued_bridge_count = 0
    bridge_candidates_checked = 0
    col_from = _find_col(flows, ["from_node", "fromnode"])
    col_to = _find_col(flows, ["to_node", "tonode"])
    if col_from and col_to and len(flows) and not flows_attr.empty:
        retained_idx = set(flows_attr.index.tolist())
        if retained_idx:
            retained_node_degree = {}
            for _, row in flows.loc[list(retained_idx), [col_from, col_to]].iterrows():
                for node in (row.get(col_from), row.get(col_to)):
                    if pd.notna(node):
                        retained_node_degree[node] = retained_node_degree.get(node, 0) + 1
            rescue_mask = short_length_filtered.copy()
            if col_order:
                rescue_mask &= np.nan_to_num(np.asarray(flows[col_order], dtype=float), nan=-1.0) >= float(min_stream_order)
            if col_comp and top_components:
                rescue_mask &= flows[col_comp].isin(list(top_components)).to_numpy(dtype=bool)
            if np.any(rescue_mask):
                rescue_idx = []
                for idx, row in flows.loc[rescue_mask, [col_from, col_to] + ([col_comp] if col_comp else [])].iterrows():
                    bridge_candidates_checked += 1
                    from_node = row.get(col_from)
                    to_node = row.get(col_to)
                    if pd.isna(from_node) or pd.isna(to_node):
                        continue
                    from_deg = int(retained_node_degree.get(from_node, 0))
                    to_deg = int(retained_node_degree.get(to_node, 0))
                    if from_deg <= 0 or to_deg <= 0:
                        continue
                    if col_comp:
                        comp_val = row.get(col_comp)
                        neigh = flows.loc[list(retained_idx)]
                        node_touch = ((neigh[col_from] == from_node) | (neigh[col_to] == from_node) | (neigh[col_from] == to_node) | (neigh[col_to] == to_node))
                        same_comp = neigh[col_comp] == comp_val
                        if not bool(np.any(node_touch & same_comp)):
                            continue
                    rescue_idx.append(idx)
                if rescue_idx:
                    rescued = flows.loc[rescue_idx].copy()
                    flows_attr = flows.loc[list(retained_idx) + rescue_idx].copy()
                    flows_attr = flows_attr.loc[~flows_attr.index.duplicated(keep='first')].copy()
                    rescued_bridge_count = int(len(rescue_idx))

    flows_attr = flows_attr[flows_attr.geometry.notnull() & ~flows_attr.geometry.is_empty].copy()

    # Restrict polygons to those intersecting retained flowlines when available.
    retained_poly_count = 0
    if polygons is not None and not polygons.empty and not flows_attr.empty:
        merged = _geometry_union([geom.buffer(max(5.0, 1.5), cap_style=2) for geom in flows_attr.geometry if geom is not None and not geom.is_empty])
        polygons = polygons[polygons.geometry.intersects(merged)].copy()
        retained_poly_count = int(len(polygons))

    meta = {
        "input_flow_count": flow_count0,
        "retained_flow_count": int(len(flows_attr)),
        "retained_polygon_count": retained_poly_count,
        "min_stream_order": int(min_stream_order),
        "min_length_km": float(min_length_km),
        "keep_top_components": int(keep_top_components),
        "top_components": [str(x) for x in list(top_components)[: max(int(keep_top_components), 1)]],
        "selection_mode": "mainstem_and_key_tributaries",
        "dropped_feature_families": ["lakes", "ponds", "minor_streams", "small_disconnected_waterbodies"],
        "rescued_short_bridge_segments": int(rescued_bridge_count),
        "bridge_candidates_checked": int(bridge_candidates_checked),
    }
    return RiverScaffoldSelection(flows=flows_attr, polygons=polygons, metadata=meta)


def _sample_points_along_line(line, spacing_m: float):
    if line is None or line.is_empty:
        return []
    try:
        if line.geom_type == "MultiLineString":
            pts = []
            for geom in line.geoms:
                pts.extend(_sample_points_along_line(geom, spacing_m))
            return pts
        L = float(line.length)
        if L <= 0:
            return []
        step = max(float(spacing_m), 1.0)
        dists = list(np.arange(0.0, L, step, dtype=float))
        if not dists or (L - dists[-1]) > (0.35 * step):
            dists.append(L)
        return [line.interpolate(float(d)) for d in dists]
    except Exception:
        log.debug("_sample_points_along_line: suppressed exception", exc_info=True)
        return []




def _station_component_key(raw) -> Optional[str]:
    if raw is None or pd.isna(raw):
        return None
    if isinstance(raw, str):
        val = raw.strip()
        return val or None
    if isinstance(raw, (np.integer, int)):
        return str(int(raw))
    if isinstance(raw, (np.floating, float)):
        if not np.isfinite(float(raw)):
            return None
        if float(raw).is_integer():
            return str(int(float(raw)))
        return format(float(raw), '.12g')
    val = str(raw).strip()
    return val or None


def normalize_structured_flow_component_ids(flows_gdf, *, logger=None):
    """Promote a branch/path-aware component identifier before stationized products are built.

    Many retained NHD subsets carry a coarse component_id that groups the entire retained network,
    while levelpath fields distinguish the actual mainstem and tributary branches. Stationized river
    products must not collapse those branches into a single component namespace.
    """
    if flows_gdf is None or getattr(flows_gdf, 'empty', True):
        return flows_gdf, {"component_id_source": "none", "component_count_before": 0, "component_count_after": 0}
    flows = flows_gdf.copy()
    original_component_series = None
    if 'component_id' in flows.columns:
        original_component_series = flows['component_id'].copy()
    preferred_col = None
    for cand in ('levelpathi', 'levelpathid'):
        if cand in flows.columns:
            vals = [_station_component_key(v) for v in flows[cand].tolist()]
            unique = {v for v in vals if v is not None}
            if unique:
                preferred_col = cand
                break
    resolved = []
    sources = []
    for idx, row in flows.iterrows():
        preferred_val = _station_component_key(row.get(preferred_col)) if preferred_col else None
        orig_val = _station_component_key(row.get('component_id')) if 'component_id' in flows.columns else None
        if preferred_val is not None:
            resolved.append(f'{preferred_col}:{preferred_val}')
            sources.append(preferred_col)
        elif orig_val is not None:
            resolved.append(f'component_id:{orig_val}')
            sources.append('component_id')
        else:
            resolved.append(f'flow_idx:{idx}')
            sources.append('flow_idx')
    if original_component_series is not None and 'network_component_id' not in flows.columns:
        flows['network_component_id'] = original_component_series
    flows['component_id'] = pd.Series(resolved, index=flows.index, dtype='object')
    flows['component_id_source'] = pd.Series(sources, index=flows.index, dtype='object')
    meta = {
        'component_id_source': preferred_col or ('component_id' if original_component_series is not None else 'flow_idx'),
        'component_count_before': int(pd.Series(original_component_series).dropna().astype(str).nunique()) if original_component_series is not None else 0,
        'component_count_after': int(pd.Series(resolved, index=flows.index, dtype='object').dropna().astype(str).nunique()),
    }
    if logger is not None and meta['component_count_after'] > max(meta['component_count_before'], 0):
        logger.info(
            '[RIVER][GUIDANCE] Structured flow component ids promoted from %s: components_before=%d components_after=%d',
            meta['component_id_source'],
            int(meta['component_count_before']),
            int(meta['component_count_after']),
        )
    return flows, meta




def validate_centerline_station_component_contract(
    centerline_points_gdf,
    *,
    expected_component_count: int = 0,
    expected_component_source: str | None = None,
    logger=None,
    context: str = "centerline_points",
):
    """Validate that stationized centerline points preserve the structured component namespace.

    This is intended to fail at the first producer-stage artifact if branch-aware
    centerline stationization collapses back to a coarse single-component namespace.
    """
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
    if pts.empty:
        raise RuntimeError(f'{context}_has_no_finite_station_rows')
    duplicate_rows = int(pts.duplicated(subset=['component_id', 'station_m'], keep=False).sum())
    if duplicate_rows > 0:
        raise RuntimeError(f'{context}_duplicate_component_station_rows:{duplicate_rows}')
    component_count = int(pts['component_id'].nunique(dropna=True))
    station_contract = getattr(centerline_points_gdf, 'attrs', {}).get('station_contract', {}) if hasattr(centerline_points_gdf, 'attrs') else {}
    coarse_expected = int(expected_component_count or station_contract.get('expected_component_count', 0) or station_contract.get('component_count_after', 0) or 0)
    component_source = str(
        expected_component_source
        or station_contract.get('expected_component_source')
        or station_contract.get('component_id_source', 'unknown')
    )
    if coarse_expected > 1 and component_count <= 1:
        raise RuntimeError(
            f'{context}_coarse_component_namespace: expected_components>1 actual_components={component_count} source={component_source}'
        )
    if logger is not None:
        logger.info(
            '[RIVER][GUIDANCE][DETAIL] Centerline component contract validated: context=%s components=%d expected_components=%d source=%s',
            str(context),
            int(component_count),
            int(coarse_expected),
            component_source,
        )
    return {
        'context': str(context),
        'component_count': int(component_count),
        'expected_component_count': int(coarse_expected),
        'component_id_source': component_source,
        'duplicate_component_station_rows': int(duplicate_rows),
    }


def _infer_component_segment_station_offsets(comp_meta: pd.DataFrame) -> tuple[pd.Series, str]:
    """Infer deterministic cumulative segment offsets for components lacking s_m_from/s_m_to.

    This prevents per-segment local station resets (0..L on every segment) from creating
    repeated (component_id, station_m) keys when a structured branch is represented by
    multiple retained flowlines without explicit along-component station fields.
    """
    if comp_meta is None or comp_meta.empty:
        return pd.Series(dtype=float), 'none'
    work = comp_meta.copy()
    lengths = np.asarray([float(getattr(g, 'length', 0.0) or 0.0) for g in work.geometry], dtype=float)
    work['_line_length_m'] = lengths
    work['_sort_streamorder'] = pd.to_numeric(work.get('streamorde', work.get('streamorder', np.nan)), errors='coerce').fillna(-np.inf)
    work['_sort_flow_idx'] = work.index.astype(str)
    col_from = _find_col(work, ['from_node', 'fromnode'])
    col_to = _find_col(work, ['to_node', 'tonode'])

    ordered_idx: list = []
    visited: set = set()
    method = 'deterministic_length_sort'

    def _row_priority_key(idx):
        row = work.loc[idx]
        return (
            -float(row.get('_sort_streamorder', -np.inf)),
            -float(row.get('_line_length_m', 0.0)),
            str(row.get('_sort_flow_idx', idx)),
        )

    if col_from and col_to:
        valid_topology = work[col_from].notna() & work[col_to].notna()
        topo = work.loc[valid_topology].copy()
        if not topo.empty:
            method = 'topology_chain'
            outgoing: dict = {}
            indegree: dict = {}
            for idx, row in topo.iterrows():
                fn = row[col_from]
                tn = row[col_to]
                outgoing.setdefault(fn, []).append(idx)
                indegree[tn] = indegree.get(tn, 0) + 1
                indegree.setdefault(fn, indegree.get(fn, 0))
            heads = [node for node, outs in outgoing.items() if outs and indegree.get(node, 0) == 0]
            heads = sorted(heads, key=lambda n: min(_row_priority_key(i) for i in outgoing.get(n, [])))
            if not heads:
                heads = sorted(outgoing.keys(), key=lambda n: min(_row_priority_key(i) for i in outgoing.get(n, [])))
            for head in heads:
                node = head
                local_seen = set()
                while node in outgoing:
                    candidates = [i for i in outgoing.get(node, []) if i not in visited]
                    if not candidates:
                        break
                    candidates = sorted(candidates, key=_row_priority_key)
                    idx = candidates[0]
                    ordered_idx.append(idx)
                    visited.add(idx)
                    next_node = topo.loc[idx, col_to]
                    edge_key = (node, next_node, idx)
                    if edge_key in local_seen:
                        break
                    local_seen.add(edge_key)
                    node = next_node
            remaining_topo = [idx for idx in topo.index.tolist() if idx not in visited]
            for idx in sorted(remaining_topo, key=_row_priority_key):
                ordered_idx.append(idx)
                visited.add(idx)
            remaining_non_topo = [idx for idx in work.index.tolist() if idx not in visited]
            for idx in sorted(remaining_non_topo, key=_row_priority_key):
                ordered_idx.append(idx)
                visited.add(idx)
    if not ordered_idx:
        ordered_idx = sorted(work.index.tolist(), key=_row_priority_key)

    offsets = {}
    cumulative = 0.0
    for idx in ordered_idx:
        offsets[idx] = float(cumulative)
        cumulative += float(max(work.loc[idx, '_line_length_m'], 0.0))
    return pd.Series(offsets, dtype=float), method


def _centerline_station_contract_summary(points_before, points_after, *, component_axis_stationization_used: int = 0, component_id_source: str | None = None, component_count_before: int = 0, component_count_after: int = 0, fallback_method_counts: Optional[dict] = None) -> dict:
    summary = {
        'component_id_source': component_id_source or 'unknown',
        'expected_component_source': component_id_source or 'unknown',
        'component_count_before': int(component_count_before or 0),
        'component_count_after': int(component_count_after or 0),
        'expected_component_count': int(component_count_after or 0),
        'component_axis_stationization_used': int(component_axis_stationization_used or 0),
        'fallback_stationization_method_counts': dict(fallback_method_counts or {}),
    }
    if points_before is None or getattr(points_before, 'empty', True):
        summary.update({
            'raw_component_station_count': 0,
            'unique_component_station_count': 0,
            'duplicate_component_station_rows_involved': 0,
            'duplicate_component_station_rows_collapsed': 0,
            'duplicate_component_station_fraction': 0.0,
            'components_with_duplicate_station_rows': 0,
            'max_component_duplicate_fraction': 0.0,
            'worst_duplicate_component_id': None,
        })
        return summary
    pts = points_before.copy()
    if 'component_id' not in pts.columns:
        pts['component_id'] = 'main'
    pts['component_id'] = pts['component_id'].fillna('main').astype(str)
    pts['station_m'] = pd.to_numeric(pts.get('station_m'), errors='coerce')
    pts = pts.loc[np.isfinite(pts['station_m'])].copy()
    if pts.empty:
        summary.update({
            'raw_component_station_count': 0,
            'unique_component_station_count': 0,
            'duplicate_component_station_rows_involved': 0,
            'duplicate_component_station_rows_collapsed': 0,
            'duplicate_component_station_fraction': 0.0,
            'components_with_duplicate_station_rows': 0,
            'max_component_duplicate_fraction': 0.0,
            'worst_duplicate_component_id': None,
        })
        return summary
    pts['_station_key'] = np.round(pts['station_m'].to_numpy(dtype=float), 6)
    raw_count = int(len(pts))
    dup_involved = int(pts.duplicated(subset=['component_id', '_station_key'], keep=False).sum())
    unique_count = int(pts[['component_id', '_station_key']].drop_duplicates().shape[0])
    collapsed = int(max(raw_count - unique_count, 0))
    comp_rows = []
    for comp, grp in pts.groupby('component_id', sort=False, dropna=False):
        grp_rows = int(len(grp))
        grp_unique = int(grp[['_station_key']].drop_duplicates().shape[0])
        grp_dup = int(grp.duplicated(subset=['_station_key'], keep=False).sum())
        grp_collapsed = int(max(grp_rows - grp_unique, 0))
        grp_frac = float(grp_collapsed / grp_rows) if grp_rows > 0 else 0.0
        comp_rows.append({
            'component_id': str(comp),
            'raw_point_count': grp_rows,
            'unique_component_station_count': grp_unique,
            'duplicate_component_station_rows_involved': grp_dup,
            'duplicate_component_station_rows_collapsed': grp_collapsed,
            'duplicate_component_station_fraction': grp_frac,
        })
    worst = max(comp_rows, key=lambda r: (r['duplicate_component_station_fraction'], r['duplicate_component_station_rows_collapsed'])) if comp_rows else None
    summary.update({
        'raw_component_station_count': raw_count,
        'unique_component_station_count': unique_count,
        'duplicate_component_station_rows_involved': dup_involved,
        'duplicate_component_station_rows_collapsed': collapsed,
        'duplicate_component_station_fraction': float(collapsed / raw_count) if raw_count > 0 else 0.0,
        'components_with_duplicate_station_rows': int(sum(1 for r in comp_rows if r['duplicate_component_station_rows_collapsed'] > 0)),
        'max_component_duplicate_fraction': float(worst['duplicate_component_station_fraction']) if worst else 0.0,
        'worst_duplicate_component_id': worst['component_id'] if worst else None,
        'per_component_duplicate_summary': comp_rows,
    })
    return summary


def _collapse_duplicate_centerline_component_station_rows(points_gdf, logger=None):
    try:
        import geopandas as gpd
    except Exception:
        gpd = None
    if points_gdf is None or getattr(points_gdf, 'empty', True):
        return points_gdf, 0
    pts = points_gdf.copy()
    if 'station_m' not in pts.columns:
        return pts, 0
    if 'component_id' not in pts.columns:
        pts['component_id'] = 'main'
    pts['component_id'] = pts['component_id'].fillna('main').astype(str)
    pts['station_m'] = pd.to_numeric(pts['station_m'], errors='coerce')
    pts = pts.loc[np.isfinite(pts['station_m'])].copy()
    if pts.empty:
        return pts, 0
    pts['_station_key'] = np.round(pts['station_m'].to_numpy(dtype=float), 6)
    key_cols = ['component_id', '_station_key']
    duplicate_rows = int(pts.duplicated(subset=key_cols, keep=False).sum())
    if duplicate_rows == 0:
        return pts.drop(columns=['_station_key'], errors='ignore'), 0

    geom_name = getattr(pts, 'geometry', None).name if getattr(pts, 'geometry', None) is not None else None
    numeric_cols = {
        col for col in pts.columns
        if col not in {'_station_key', 'component_id', geom_name}
        and (pd.api.types.is_numeric_dtype(pts[col]) or col.endswith('_m') or col.endswith('_order') or col.endswith('_rank'))
    }

    def _first_nonnull(series: pd.Series):
        for val in series.tolist():
            if pd.isna(val):
                continue
            if isinstance(val, str) and not val.strip():
                continue
            return val
        return pd.NA

    rows = []
    order_cols = [c for c in ('streamorde', 'streamorder', '_line_length_m', '_sample_proj_m', 'flow_idx') if c in pts.columns]
    for _, grp in pts.groupby(key_cols, sort=False, dropna=False):
        grp = grp.copy()
        sort_cols = []
        ascending = []
        if 'streamorde' in grp.columns:
            grp['_sort_streamorde'] = pd.to_numeric(grp['streamorde'], errors='coerce').fillna(-np.inf)
            sort_cols.append('_sort_streamorde'); ascending.append(False)
        elif 'streamorder' in grp.columns:
            grp['_sort_streamorder'] = pd.to_numeric(grp['streamorder'], errors='coerce').fillna(-np.inf)
            sort_cols.append('_sort_streamorder'); ascending.append(False)
        if '_line_length_m' in grp.columns:
            grp['_sort_line_length_m'] = pd.to_numeric(grp['_line_length_m'], errors='coerce').fillna(-np.inf)
            sort_cols.append('_sort_line_length_m'); ascending.append(False)
        if '_sample_proj_m' in grp.columns:
            grp['_sort_sample_proj_m'] = pd.to_numeric(grp['_sample_proj_m'], errors='coerce').fillna(np.inf)
            sort_cols.append('_sort_sample_proj_m'); ascending.append(True)
        if 'flow_idx' in grp.columns:
            grp['_sort_flow_idx'] = grp['flow_idx'].astype(str)
            sort_cols.append('_sort_flow_idx'); ascending.append(True)
        chosen = grp.sort_values(sort_cols, ascending=ascending, kind='mergesort').iloc[0] if sort_cols else grp.iloc[0]
        row = chosen.copy()
        row['station_m'] = float(pd.to_numeric(grp['station_m'], errors='coerce').median())
        row['component_id'] = str(_first_nonnull(grp['component_id']))
        for col in numeric_cols:
            vals = pd.to_numeric(grp[col], errors='coerce')
            row[col] = float(vals.median()) if vals.notna().any() else np.nan
        for col in grp.columns:
            if col in key_cols or col in numeric_cols or col == geom_name or col.startswith('_sort_'):
                continue
            row[col] = _first_nonnull(grp[col])
        rows.append(row)
    collapsed = pd.DataFrame(rows).drop(columns=['_station_key'], errors='ignore')
    drop_cols = [c for c in collapsed.columns if c.startswith('_sort_')]
    collapsed = collapsed.drop(columns=drop_cols, errors='ignore')
    if gpd is not None and geom_name is not None and geom_name in collapsed.columns:
        collapsed = gpd.GeoDataFrame(collapsed, geometry=geom_name, crs=getattr(points_gdf, 'crs', None))
    removed = int(len(pts) - len(collapsed))
    if removed > 0 and logger is not None:
        logger.warning(
            '[RIVER][GUIDANCE] Collapsed duplicate centerline component/station rows before frame build: removed=%d involved=%d',
            removed,
            duplicate_rows,
        )
    return collapsed, removed


def _sample_raster_values(raster_path: str | Path, points_gdf, field: str = "z_m"):
    import rasterio

    out = np.full(len(points_gdf), np.nan, dtype=np.float32)
    if points_gdf is None or getattr(points_gdf, "empty", True):
        points_gdf[field] = out
        return points_gdf
    with rasterio.open(raster_path) as ds:
        pts_src = points_gdf
        try:
            pts_crs = getattr(points_gdf, "crs", None)
            ds_crs = getattr(ds, "crs", None)
            if pts_crs is not None and ds_crs is not None and str(pts_crs) != str(ds_crs):
                pts_src = points_gdf.to_crs(ds_crs)
        except Exception:
            log.debug("_sample_raster_values: failed CRS harmonization; sampling in source CRS", exc_info=True)
            pts_src = points_gdf
        pts = list(zip(pts_src.geometry.x.to_numpy(dtype=float), pts_src.geometry.y.to_numpy(dtype=float)))
        vals = []
        for v in ds.sample(pts):
            val = float(v[0]) if np.size(v) else np.nan
            if ds.nodata is not None and np.isfinite(val) and np.isclose(val, float(ds.nodata)):
                val = np.nan
            vals.append(val)
        out = np.asarray(vals, dtype=np.float32)
    points_gdf[field] = out
    return points_gdf


def _mask_points_to_allowed(points_gdf, allowed_mask: Optional[np.ndarray], transform):
    if points_gdf is None or getattr(points_gdf, "empty", True) or allowed_mask is None or transform is None:
        return points_gdf
    import rasterio.transform

    keep = np.zeros(len(points_gdf), dtype=bool)
    try:
        xs = points_gdf.geometry.x.to_numpy(dtype=float)
        ys = points_gdf.geometry.y.to_numpy(dtype=float)
        rows, cols = rasterio.transform.rowcol(transform, xs, ys)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        ok = (rows >= 0) & (rows < allowed_mask.shape[0]) & (cols >= 0) & (cols < allowed_mask.shape[1])
        keep[ok] = np.asarray(allowed_mask[rows[ok], cols[ok]], dtype=bool)
    except Exception:
        log.debug("_mask_points_to_allowed: suppressed exception", exc_info=True)
        return points_gdf
    return points_gdf.loc[keep].copy()




def _clip_gdf_to_allowed_mask(gdf, allowed_mask: Optional[np.ndarray], transform):
    if gdf is None or getattr(gdf, "empty", True) or allowed_mask is None or transform is None:
        return gdf
    from rasterio.features import shapes
    from shapely.geometry import shape

    mask_bool = np.asarray(allowed_mask, dtype=bool)
    if not np.any(mask_bool):
        return gdf.iloc[0:0].copy()
    try:
        geoms = [shape(geom) for geom, value in shapes(mask_bool.astype("uint8"), transform=transform) if int(value) == 1]
    except Exception:
        log.debug("_clip_gdf_to_allowed_mask: suppressed exception", exc_info=True)
        geoms = []
    if not geoms:
        return gdf.iloc[0:0].copy()
    try:
        allowed_geom = _geometry_union(geoms)
    except Exception:
        log.debug("_clip_gdf_to_allowed_mask: suppressed exception", exc_info=True)
        allowed_geom = geoms[0]
        for geom in geoms[1:]:
            allowed_geom = allowed_geom.union(geom)
    clipped = gdf.copy()
    clipped["geometry"] = [geom.intersection(allowed_geom) if geom is not None else None for geom in clipped.geometry]
    clipped = clipped[clipped.geometry.notnull() & ~clipped.geometry.is_empty].copy()
    return clipped


def _sample_raster_point(ds, x: float, y: float) -> float:
    try:
        v = next(ds.sample([(float(x), float(y))]))
    except Exception:
        log.debug("_sample_raster_point: suppressed exception", exc_info=True)
        return float("nan")
    val = float(v[0]) if np.size(v) else np.nan
    if ds.nodata is not None and np.isfinite(val) and np.isclose(val, float(ds.nodata)):
        return float("nan")
    return val


def _estimate_line_tangent(line, distance: float, eps: float = 1.0):
    try:
        L = float(line.length)
        d0 = max(0.0, min(L, float(distance) - float(eps)))
        d1 = max(0.0, min(L, float(distance) + float(eps)))
        p0 = line.interpolate(d0)
        p1 = line.interpolate(d1)
        dx = float(p1.x - p0.x)
        dy = float(p1.y - p0.y)
        n = math.hypot(dx, dy)
        if not np.isfinite(n) or n <= 0.0:
            return None
        return (dx / n, dy / n)
    except Exception:
        log.debug("_estimate_line_tangent: suppressed exception", exc_info=True)
        return None


def _normal_search_value(ds, x: float, y: float, tangent, max_offset_m: float = 8.0, step_m: float = 1.0):
    if tangent is None:
        return float("nan"), float("inf"), "missing"
    tx, ty = tangent
    nx, ny = -ty, tx
    best_val = float("nan")
    best_dist = float("inf")
    best_status = "missing"
    for sign, status in ((1.0, "normal_search_outward"), (-1.0, "normal_search_inward")):
        offset = max(float(step_m), 0.5)
        while offset <= float(max_offset_m) + 1e-6:
            xx = float(x + sign * nx * offset)
            yy = float(y + sign * ny * offset)
            val = _sample_raster_point(ds, xx, yy)
            if np.isfinite(val):
                if offset < best_dist:
                    best_val = val
                    best_dist = float(offset)
                    best_status = status
                break
            offset += max(float(step_m), 0.5)
    return best_val, best_dist, best_status


def _fill_along_bank(points_gdf, value_field: str, status_field: str, distance_field: str, confidence_field: str):
    if points_gdf is None or getattr(points_gdf, "empty", True) or value_field not in points_gdf.columns:
        return points_gdf
    points_gdf = points_gdf.copy()
    if "bank_group" not in points_gdf.columns:
        points_gdf["bank_group"] = 0
    if "bank_chain_m" not in points_gdf.columns:
        points_gdf["bank_chain_m"] = np.arange(len(points_gdf), dtype=float)
    if confidence_field in points_gdf.columns:
        points_gdf[confidence_field] = pd.to_numeric(points_gdf[confidence_field], errors="coerce").astype(np.float32)
    for _, idx in points_gdf.groupby("bank_group", sort=False).groups.items():
        ids = list(idx)
        vals = np.asarray(points_gdf.loc[ids, value_field], dtype=float)
        chain = np.asarray(points_gdf.loc[ids, "bank_chain_m"], dtype=float)
        valid = np.isfinite(vals)
        if np.count_nonzero(valid) < 2:
            continue
        interp = np.interp(chain, chain[valid], vals[valid])
        missing = ~valid
        if np.any(missing):
            sel = np.asarray(ids)[missing]
            points_gdf.loc[sel, value_field] = interp[missing].astype(np.float32)
            points_gdf.loc[sel, status_field] = "along_bank_interp"
            points_gdf.loc[sel, distance_field] = np.nan
            points_gdf.loc[sel, confidence_field] = np.float32(0.45)
    return points_gdf


def build_dense_bank_points(
    polygons_gdf,
    *,
    spacing_m: float = 40.0,
    raster_path: str | Path | None = None,
    role: str = "bank_boundary_control",
    normal_search_max_m: float = 8.0,
    normal_search_step_m: float = 1.0,
    xs_bank_points=None,
    allowed_mask: Optional[np.ndarray] = None,
    transform=None,
):
    import geopandas as gpd
    from shapely.geometry import Point
    import rasterio

    if polygons_gdf is None or getattr(polygons_gdf, "empty", True):
        return gpd.GeoDataFrame(columns=["artifact_role", "geometry"], geometry="geometry", crs=getattr(polygons_gdf, "crs", None))

    rows = []
    for idx, geom in enumerate(polygons_gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        boundaries = []
        try:
            if geom.geom_type == "Polygon":
                boundaries = [geom.exterior] + list(geom.interiors)
            elif geom.geom_type == "MultiPolygon":
                for g in geom.geoms:
                    if g is None or g.is_empty:
                        continue
                    boundaries.append(g.exterior)
                    boundaries.extend(list(g.interiors))
        except Exception:
            log.debug("build_dense_bank_points: suppressed exception", exc_info=True)
            boundaries = []
        for b in boundaries:
            pts = _sample_points_along_line(b, spacing_m)
            for j, pt in enumerate(pts):
                chain = float(j) * float(max(spacing_m, 1.0))
                rows.append(
                    {
                        "artifact_role": role,
                        "bank_group": int(idx),
                        "bank_chain_m": chain,
                        "bank_sample_status": "missing",
                        "bank_sample_distance_m": np.nan,
                        "bank_confidence": 0.0,
                        "geometry": Point(pt.x, pt.y),
                        "_line_ref": b,
                        "_line_dist": chain,
                    }
                )
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=polygons_gdf.crs)
    out = _mask_points_to_allowed(out, allowed_mask, transform)
    if raster_path is None or out.empty:
        return out.drop(columns=[c for c in ["_line_ref", "_line_dist"] if c in out.columns])


    vals = []
    statuses = []
    dists = []
    confs = []
    with rasterio.open(raster_path) as ds:
        xs = out.geometry.x.to_numpy(dtype=float)
        ys = out.geometry.y.to_numpy(dtype=float)
        pts = list(zip(xs, ys))
        exact_vals = []
        for v in ds.sample(pts):
            val = float(v[0]) if np.size(v) else np.nan
            if ds.nodata is not None and np.isfinite(val) and np.isclose(val, float(ds.nodata)):
                val = np.nan
            exact_vals.append(val)
        exact_vals = np.asarray(exact_vals, dtype=float)
        for i in range(len(out)):
            rec = out.iloc[i]
            x = float(xs[i])
            y = float(ys[i])
            val = float(exact_vals[i]) if i < exact_vals.size else np.nan
            status = "exact"
            dist = 0.0
            conf = 1.0
            if not np.isfinite(val):
                tangent = _estimate_line_tangent(rec["_line_ref"], rec["_line_dist"], eps=max(float(spacing_m) * 0.25, 1.0))
                val, dist, status = _normal_search_value(ds, x, y, tangent, max_offset_m=normal_search_max_m, step_m=normal_search_step_m)
                conf = 0.8 if np.isfinite(val) else 0.0
            vals.append(val)
            statuses.append(status)
            dists.append(dist if np.isfinite(val) else np.nan)
            confs.append(conf if np.isfinite(val) else 0.0)

    out["bank_z_m"] = np.asarray(vals, dtype=np.float32)
    out["bank_sample_status"] = statuses
    out["bank_sample_distance_m"] = np.asarray(dists, dtype=np.float32)
    out["bank_confidence"] = np.asarray(confs, dtype=np.float32)
    out = _fill_along_bank(out, "bank_z_m", "bank_sample_status", "bank_sample_distance_m", "bank_confidence")
    return out.drop(columns=[c for c in ["_line_ref", "_line_dist"] if c in out.columns])


def build_centerline_points(flows_gdf, polygons_gdf=None, *, spacing_m: float = 60.0, raster_path: str | Path | None = None, role: str = "longitudinal_control", allowed_mask: Optional[np.ndarray] = None, transform=None, logger=None):
    import geopandas as gpd
    import time

    if flows_gdf is None or getattr(flows_gdf, "empty", True):
        return gpd.GeoDataFrame(columns=["artifact_role", "geometry"], geometry="geometry", crs=getattr(flows_gdf, "crs", None))
    t0 = time.perf_counter()
    flows = flows_gdf.copy()
    flows, component_meta = normalize_structured_flow_component_ids(flows, logger=logger)
    poly_union = None
    clip_buffer = None
    if polygons_gdf is not None and not getattr(polygons_gdf, "empty", True):
        try:
            polygons_gdf = _safe_to_crs(polygons_gdf, flows.crs)
            poly_union = _geometry_union(polygons_gdf.geometry)
            if poly_union is not None and not poly_union.is_empty:
                clip_buffer = poly_union.buffer(1.0)
                intersects = flows.geometry.intersects(clip_buffer)
                flows = flows.loc[np.asarray(intersects, dtype=bool)].copy()
                flows["geometry"] = [geom.intersection(clip_buffer) if geom is not None else None for geom in flows.geometry]
                flows = flows[flows.geometry.notnull() & ~flows.geometry.is_empty].copy()
        except Exception:
            log.debug("build_centerline_points: suppressed exception", exc_info=True)
            poly_union = None
            clip_buffer = None
    t_union = time.perf_counter()
    rows = []
    n_sampled = 0
    component_axis_stationization_used = 0
    fallback_stationization_method_counts: dict[str, int] = {}
    keep_cols = [c for c in ("component_id", "streamorde", "streamorder", "levelpathi", "levelpathid", "s_m_from", "s_m_to", "s_m_min", "s_m_max") if c in flows.columns]

    def _flow_priority_frame(group):
        meta = group.copy()
        if 'streamorde' in meta.columns:
            meta['_sort_streamorder'] = pd.to_numeric(meta['streamorde'], errors='coerce').fillna(-np.inf)
        elif 'streamorder' in meta.columns:
            meta['_sort_streamorder'] = pd.to_numeric(meta['streamorder'], errors='coerce').fillna(-np.inf)
        else:
            meta['_sort_streamorder'] = -np.inf
        meta['_line_length_m'] = np.asarray([float(getattr(g, 'length', 0.0) or 0.0) for g in meta.geometry], dtype=float)
        meta['_sort_flow_idx'] = meta.index.astype(str)
        meta['_s0'] = pd.to_numeric(meta.get('s_m_from', meta.get('s_m_min', np.nan)), errors='coerce')
        meta['_s1'] = pd.to_numeric(meta.get('s_m_to', meta.get('s_m_max', np.nan)), errors='coerce')
        meta['_s_lo'] = np.minimum(meta['_s0'], meta['_s1'])
        meta['_s_hi'] = np.maximum(meta['_s0'], meta['_s1'])
        return meta

    for comp, comp_flows in flows.groupby('component_id', sort=False, dropna=False):
        comp_meta = _flow_priority_frame(comp_flows)
        valid_station = comp_meta.loc[np.isfinite(comp_meta['_s0']) & np.isfinite(comp_meta['_s1']) & (comp_meta['_line_length_m'] > 0.0)].copy()
        component_rows = []
        if not valid_station.empty:
            comp_min = float(valid_station['_s_lo'].min())
            comp_max = float(valid_station['_s_hi'].max())
            step = max(float(spacing_m), 1.0)
            station_grid = list(np.arange(comp_min, comp_max, step, dtype=float))
            if not station_grid or (comp_max - station_grid[-1]) > (0.35 * step):
                station_grid.append(comp_max)
            tol = max(0.5, 0.51 * step)
            candidate_count = 0
            for station in station_grid:
                covering = valid_station.loc[(valid_station['_s_lo'] - tol <= float(station)) & (valid_station['_s_hi'] + tol >= float(station))].copy()
                if covering.empty:
                    continue
                candidate_count += int(len(covering))
                covering['_station_mid_distance'] = np.abs(((covering['_s_lo'] + covering['_s_hi']) * 0.5) - float(station))
                chosen = covering.sort_values(
                    by=['_sort_streamorder', '_line_length_m', '_station_mid_distance', '_sort_flow_idx'],
                    ascending=[False, False, True, True],
                    kind='mergesort',
                ).iloc[0]
                line = chosen.geometry
                line_len = float(chosen['_line_length_m'])
                s0 = float(chosen['_s0'])
                s1 = float(chosen['_s1'])
                frac = 0.0 if np.isclose(s1, s0) else float((float(station) - s0) / (s1 - s0))
                frac = min(max(frac, 0.0), 1.0)
                pt = line.interpolate(frac * line_len) if line_len > 0.0 else None
                if pt is None or pt.is_empty:
                    continue
                row = {"artifact_role": role, "flow_idx": int(chosen.name) if isinstance(chosen.name, (int, np.integer)) else str(chosen.name)}
                for col in keep_cols:
                    row[col] = chosen[col]
                row['geometry'] = pt
                row['_sample_proj_m'] = float(frac * line_len)
                row['_line_length_m'] = line_len
                row['station_m'] = float(station)
                component_rows.append(row)
                n_sampled += 1
            if component_rows:
                component_axis_stationization_used += 1
                rows.extend(component_rows)
                continue
        fallback_offsets, fallback_method = _infer_component_segment_station_offsets(comp_meta)
        fallback_stationization_method_counts[fallback_method] = fallback_stationization_method_counts.get(fallback_method, 0) + 1
        component_station_keys_seen: set[tuple[str, float]] = set()
        for idx, rec in comp_meta.iterrows():
            line = rec.geometry
            pts = _sample_points_along_line(line, spacing_m)
            if not pts:
                continue
            n_sampled += int(len(pts))
            base = {"artifact_role": role, "flow_idx": int(idx) if isinstance(idx, (int, np.integer)) else str(idx)}
            for col in keep_cols:
                base[col] = rec[col]
            line_len = float(getattr(line, "length", 0.0) or 0.0)
            s0 = pd.to_numeric(rec.get("s_m_from", rec.get("s_m_min", np.nan)), errors="coerce")
            s1 = pd.to_numeric(rec.get("s_m_to", rec.get("s_m_max", np.nan)), errors="coerce")
            use_station = np.isfinite(s0) and np.isfinite(s1) and line_len > 0.0
            segment_offset = float(pd.to_numeric(fallback_offsets.get(idx, 0.0), errors='coerce')) if idx in fallback_offsets.index else 0.0
            for pt in pts:
                row = dict(base)
                row["geometry"] = pt
                proj = float(line.project(pt)) if line_len > 0.0 else 0.0
                row["_sample_proj_m"] = proj
                row["_line_length_m"] = line_len
                if use_station:
                    frac = float(proj / line_len) if line_len > 0.0 else 0.0
                    station_value = float(s0 + ((s1 - s0) * frac))
                else:
                    station_value = float(segment_offset + proj)
                station_key = (str(base.get("component_id", comp)), round(float(station_value), 6))
                if station_key in component_station_keys_seen:
                    continue
                component_station_keys_seen.add(station_key)
                row["station_m"] = station_value
                rows.append(row)
    t_sample = time.perf_counter()
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=flows_gdf.crs)
    n_before_allowed = int(len(out))
    out = _mask_points_to_allowed(out, allowed_mask, transform)
    n_after_allowed = int(len(out))
    out_precollapse = out.copy()
    out, duplicate_rows_collapsed = _collapse_duplicate_centerline_component_station_rows(out, logger=logger)
    if not getattr(out, 'empty', True):
        out = out.reset_index(drop=True)
    t_mask = time.perf_counter()
    if raster_path is not None and not out.empty:
        out = _sample_raster_values(raster_path, out, field="centerline_z_m")
    drop_internal = [c for c in ("_sample_proj_m", "_line_length_m") if c in out.columns]
    if drop_internal:
        out = out.drop(columns=drop_internal, errors='ignore')
    out.attrs['station_contract'] = _centerline_station_contract_summary(
        out_precollapse,
        out,
        component_axis_stationization_used=int(component_axis_stationization_used),
        component_id_source=component_meta.get('component_id_source'),
        component_count_before=int(component_meta.get('component_count_before', 0)),
        component_count_after=int(component_meta.get('component_count_after', 0)),
        fallback_method_counts=fallback_stationization_method_counts,
    )
    out.attrs['station_contract']['points_after_allowed_mask'] = int(n_after_allowed)
    t_raster = time.perf_counter()
    if logger is not None:
        logger.info(
            "[RIVER][GUIDANCE][DETAIL] Centerline point build: flows=%d flows_clipped=%d sampled=%d kept_pre_allowed=%d kept_final=%d union=%.2fs sample=%.2fs mask=%.2fs raster=%.2fs total=%.2fs",
            0 if flows_gdf is None else int(len(flows_gdf)),
            0 if flows is None else int(len(flows)),
            int(n_sampled),
            int(n_before_allowed),
            int(len(out)),
            t_union - t0,
            t_sample - t_union,
            t_mask - t_sample,
            t_raster - t_mask,
            t_raster - t0,
        )
        station_contract = getattr(out, 'attrs', {}).get('station_contract', {}) if hasattr(out, 'attrs') else {}
        logger.info(
            '[RIVER][GUIDANCE][DETAIL] Centerline station contract: component_source=%s components_before=%d components_after=%d duplicate_rows_collapsed=%d duplicate_fraction=%.3f unique_component_station=%d component_axis_stationization_used=%d fallback_methods=%s',
            station_contract.get('component_id_source', 'unknown'),
            int(station_contract.get('component_count_before', 0)),
            int(station_contract.get('component_count_after', 0)),
            int(station_contract.get('duplicate_component_station_rows_collapsed', 0)),
            float(station_contract.get('duplicate_component_station_fraction', 0.0)),
            int(station_contract.get('unique_component_station_count', len(out))),
            int(station_contract.get('component_axis_stationization_used', 0)),
            dict(station_contract.get('fallback_stationization_method_counts', {})),
        )
        if float(station_contract.get('duplicate_component_station_fraction', 0.0)) > 0.0 and station_contract.get('worst_duplicate_component_id') is not None:
            logger.warning(
                '[RIVER][GUIDANCE][DETAIL] Centerline duplicate hotspot: component=%s duplicate_fraction=%.3f components_with_duplicates=%d',
                station_contract.get('worst_duplicate_component_id'),
                float(station_contract.get('max_component_duplicate_fraction', 0.0)),
                int(station_contract.get('components_with_duplicate_station_rows', 0)),
            )
    return out


def build_xs_support_points(
    xs_gpkg: str | Path,
    polygons_gdf=None,
    *,
    spacing_fraction: float = 0.25,
    spacing_m: Optional[float] = None,
    raster_path: str | Path | None = None,
    preferred_raster_path: str | Path | None = None,
    allow_inferred_fallback: bool = False,
    role: str = "cross_stream_control",
    allowed_mask: Optional[np.ndarray] = None,
    transform=None,
    logger=None,
):
    import geopandas as gpd
    import time

    gpkg = Path(xs_gpkg)
    try:
        layers = gpd.list_layers(gpkg)
    except (AttributeError, ImportError, OSError, ValueError) as exc:
        raise RuntimeError(f"Could not inspect XS GeoPackage layers for {gpkg}: {exc}") from exc
    layer_names = [str(v) for v in getattr(layers, "name", layers)]
    if "xs_lines" not in layer_names:
        raise RuntimeError(f"XS GeoPackage must contain layer 'xs_lines': {gpkg} | available_layers={layer_names}")
    try:
        xs = gpd.read_file(gpkg, layer="xs_lines")
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Could not read xs_lines layer from {gpkg}: {exc}") from exc
    if xs.empty:
        raise RuntimeError(f"XS GeoPackage layer 'xs_lines' is empty: {gpkg}")
    xs_input_count = int(len(xs))
    poly_union = None
    clip_buffer = None
    if polygons_gdf is not None and not getattr(polygons_gdf, "empty", True):
        try:
            polygons_gdf = _safe_to_crs(polygons_gdf, xs.crs)
            poly_union = _geometry_union(polygons_gdf.geometry)
            if poly_union is not None and not poly_union.is_empty:
                clip_buffer = poly_union.buffer(1.0)
                intersects = xs.geometry.intersects(clip_buffer)
                xs = xs.loc[np.asarray(intersects, dtype=bool)].copy()
                xs["geometry"] = [geom.intersection(clip_buffer) if geom is not None else None for geom in xs.geometry]
                xs = xs[xs.geometry.notnull() & ~xs.geometry.is_empty].copy()
        except Exception:
            log.debug("build_xs_support_points: suppressed exception", exc_info=True)
            poly_union = None
            clip_buffer = None
    t0 = time.perf_counter()
    rows = []
    frac = float(np.clip(spacing_fraction, 0.1, 0.45))
    n_sampled = 0
    for idx, rec in xs.iterrows():
        line = rec.geometry
        if line is None or line.is_empty:
            continue
        try:
            L = float(line.length)
            if spacing_m is not None and float(spacing_m) > 0.0:
                distances = list(np.arange(max(float(spacing_m), 1.0), max(L - float(spacing_m), 0.0) + 1e-6, max(float(spacing_m), 1.0), dtype=float))
                if len(distances) < 3:
                    distances = [L * frac, L * 0.5, L * (1.0 - frac)]
            else:
                distances = [L * frac, L * 0.5, L * (1.0 - frac)]
            for i, d in enumerate(distances):
                pt = line.interpolate(float(d))
                n_sampled += 1
                rows.append({"artifact_role": role, "xs_idx": int(idx) if isinstance(idx, (int, np.integer)) else str(idx), "xs_node": int(i), "geometry": pt})
        except Exception:
            log.debug("build_xs_support_points: suppressed exception", exc_info=True)
            continue
    t_sample = time.perf_counter()
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=xs.crs)
    n_before_allowed = int(len(out))
    out = _mask_points_to_allowed(out, allowed_mask, transform)
    t_mask = time.perf_counter()
    preferred_sample_count = 0
    fallback_sample_count = 0
    missing_sample_count = int(len(out))
    if not out.empty:
        sample_source = np.full(len(out), "missing", dtype=object)
        if preferred_raster_path is not None:
            out = _sample_raster_values(preferred_raster_path, out, field="xs_z_m")
            xs_vals = pd.to_numeric(out["xs_z_m"], errors="coerce").to_numpy(dtype=float)
            preferred_mask = np.isfinite(xs_vals)
            sample_source[preferred_mask] = "authoritative"
            preferred_sample_count = int(np.count_nonzero(preferred_mask))
        if raster_path is not None and bool(allow_inferred_fallback):
            xs_vals = pd.to_numeric(out["xs_z_m"], errors="coerce").to_numpy(dtype=float) if "xs_z_m" in out.columns else np.full(len(out), np.nan, dtype=float)
            need_fallback = ~np.isfinite(xs_vals)
            if np.any(need_fallback):
                fallback = _sample_raster_values(raster_path, out.loc[need_fallback].copy(), field="xs_z_m")
                out.loc[need_fallback, "xs_z_m"] = fallback["xs_z_m"].to_numpy(dtype=np.float32, copy=False)
                fallback_mask = np.isfinite(pd.to_numeric(out.loc[need_fallback, "xs_z_m"], errors="coerce").to_numpy(dtype=float))
                fallback_idx = np.flatnonzero(need_fallback)
                if fallback_idx.size:
                    sample_source[fallback_idx[fallback_mask]] = "fallback_cached_bed"
                fallback_sample_count = int(np.count_nonzero(fallback_mask))
        out["xs_sample_source"] = pd.Series(sample_source, index=out.index, dtype="object")
        xs_vals = pd.to_numeric(out["xs_z_m"], errors="coerce").to_numpy(dtype=float) if "xs_z_m" in out.columns else np.full(len(out), np.nan, dtype=float)
        missing_sample_count = int(np.count_nonzero(~np.isfinite(xs_vals)))
    t_raster = time.perf_counter()
    xs_support_contract = {
        "xs_input_count": int(xs_input_count),
        "xs_clipped_count": int(len(xs)),
        "sampled_point_count": int(n_sampled),
        "kept_pre_allowed_count": int(n_before_allowed),
        "kept_final_count": int(len(out)),
        "authoritative_sample_count": int(preferred_sample_count),
        "fallback_sample_count": int(fallback_sample_count),
        "missing_sample_count": int(missing_sample_count),
        "allow_inferred_fallback": bool(allow_inferred_fallback),
        "status": "present" if int(len(out)) > 0 else "empty_after_sampling",
    }
    if hasattr(out, "attrs"):
        out.attrs["xs_support_contract"] = xs_support_contract
    if logger is not None:
        logger.info(
            "[RIVER][GUIDANCE][DETAIL] XS support point build: xs=%d xs_clipped=%d sampled=%d kept_pre_allowed=%d kept_final=%d auth_samples=%d fallback_samples=%d missing_samples=%d allow_inferred_fallback=%s sample=%.2fs mask=%.2fs raster=%.2fs total=%.2fs",
            int(xs_input_count),
            int(len(xs)),
            int(n_sampled),
            int(n_before_allowed),
            int(len(out)),
            int(preferred_sample_count),
            int(fallback_sample_count),
            int(missing_sample_count),
            str(bool(allow_inferred_fallback)).lower(),
            t_sample - t0,
            t_mask - t_sample,
            t_raster - t_mask,
            t_raster - t0,
        )
    return out


def rasterize_point_seed_surface(*, shape, transform, domain_mask: np.ndarray, points_gdf, value_field: str):
    """Rasterize point values onto their occupied cells only.

    This is used for true seed fields such as centerline elevation/stationing.
    Unlike _nearest_surface_from_points, it does not laterally spread values
    across the corridor. Duplicate point hits within a cell are averaged.
    """
    import rasterio.transform

    out = np.full(shape, np.nan, dtype=np.float32)
    if points_gdf is None or getattr(points_gdf, "empty", True) or not np.any(domain_mask):
        return out
    vals = np.asarray(points_gdf.get(value_field, np.full(len(points_gdf), np.nan)), dtype=np.float32)
    if vals.size == 0:
        return out
    xs = np.asarray(points_gdf.geometry.x.to_numpy(dtype=float), dtype=float)
    ys = np.asarray(points_gdf.geometry.y.to_numpy(dtype=float), dtype=float)
    valid = np.isfinite(vals) & np.isfinite(xs) & np.isfinite(ys)
    if not np.any(valid):
        return out
    rows, cols = rasterio.transform.rowcol(transform, xs[valid], ys[valid])
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    vals = vals[valid].astype(np.float32)
    domain = np.asarray(domain_mask, dtype=bool)
    accum = {}
    counts = {}
    nrows, ncols = shape
    for rr, cc, vv in zip(rows.tolist(), cols.tolist(), vals.tolist()):
        if rr < 0 or cc < 0 or rr >= int(nrows) or cc >= int(ncols):
            continue
        if not domain[rr, cc]:
            continue
        key = (int(rr), int(cc))
        accum[key] = float(accum.get(key, 0.0)) + float(vv)
        counts[key] = int(counts.get(key, 0)) + 1
    for (rr, cc), total in accum.items():
        out[rr, cc] = np.float32(total / max(counts[(rr, cc)], 1))
    return out


def rasterize_point_chainage_surface(*, shape, transform, domain_mask: np.ndarray, points_gdf, station_field: str = "station_m"):
    """Build a channel-fitted chainage surface across the corridor.

    Each domain cell is assigned the station of its nearest centerline sample point.
    This keeps transport aligned to river chainage instead of leaving stationing only
    on sparse seed cells or relying on broad Euclidean raster spreading later.
    """
    import rasterio.transform
    try:
        from scipy.spatial import cKDTree as _cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy.spatial.cKDTree is required for chainage rasterization") from exc

    out = np.full(shape, np.nan, dtype=np.float32)
    domain = np.asarray(domain_mask, dtype=bool)
    if points_gdf is None or getattr(points_gdf, "empty", True) or not np.any(domain):
        return out
    vals = np.asarray(points_gdf.get(station_field, np.full(len(points_gdf), np.nan)), dtype=np.float32)
    xs_src = np.asarray(points_gdf.geometry.x.to_numpy(dtype=float), dtype=float)
    ys_src = np.asarray(points_gdf.geometry.y.to_numpy(dtype=float), dtype=float)
    valid = np.isfinite(vals) & np.isfinite(xs_src) & np.isfinite(ys_src)
    if not np.any(valid):
        return out
    src_xy = np.column_stack([xs_src[valid], ys_src[valid]])
    src_vals = vals[valid].astype(np.float32)
    tree = _cKDTree(src_xy)
    rows, cols = np.where(domain)
    if rows.size == 0:
        return out
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    q_xy = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    _, idx = tree.query(q_xy, k=1)
    out[rows, cols] = src_vals[np.asarray(idx, dtype=int)]
    return out


def build_centerline_core_influence(*, shape, transform, domain_mask: np.ndarray, points_gdf, bank_distance_m: Optional[np.ndarray] = None, max_distance_m: Optional[float] = None, influence_scale: float = 1.0):
    """Build a centerline-core influence field that stays strongest in the channel core.

    The key requirement is that the scale parameter must change the *usable centerline
    footprint*, not merely tweak the strength of an already corridor-wide field.
    When bank distance is available, interpret the scale as a multiplier on the
    local center-core half-width. This keeps the influence tied to the actual river
    width and prevents large global transition distances from making the raster
    effectively uniform across the corridor.
    """
    import rasterio.transform
    try:
        from scipy.spatial import cKDTree as _cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy.spatial.cKDTree is required for river structured scaffold centerline influence generation") from exc

    influence = np.zeros(shape, dtype=np.float32)
    if points_gdf is None or getattr(points_gdf, "empty", True) or not np.any(domain_mask):
        return influence
    xs_src = np.asarray(points_gdf.geometry.x.to_numpy(dtype=float), dtype=float)
    ys_src = np.asarray(points_gdf.geometry.y.to_numpy(dtype=float), dtype=float)
    valid = np.isfinite(xs_src) & np.isfinite(ys_src)
    if not np.any(valid):
        return influence
    pts = np.column_stack([xs_src[valid], ys_src[valid]])
    tree = _cKDTree(pts)
    rows, cols = np.where(np.asarray(domain_mask, dtype=bool))
    if rows.size == 0:
        return influence
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    q = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    dist, _ = tree.query(q, k=1)
    dist = np.asarray(dist, dtype=np.float32)

    safe_scale = max(float(influence_scale), 1.0e-3)
    infl = np.zeros_like(dist, dtype=np.float32)

    if bank_distance_m is not None:
        bank = np.asarray(bank_distance_m, dtype=np.float32)
        bank_vals = np.clip(bank[rows, cols], 0.0, np.finfo(np.float32).max).astype(np.float32)
        half_width = (bank_vals + dist).astype(np.float32)
        positive = half_width > np.float32(1.0e-6)

        # Normalized lateral distance from the centerline to the bank:
        #   0 -> centerline, 1 -> bank edge.
        norm_dist = np.ones_like(dist, dtype=np.float32)
        if np.any(positive):
            norm_dist[positive] = np.clip(dist[positive] / half_width[positive], 0.0, 1.0).astype(np.float32)

        # Default footprint should be a true core, not most of the corridor.
        # Scale broadens or narrows that local-width-normalized footprint and
        # saturates at the bank edge rather than silently becoming >100%.
        base_core_halfwidth_fraction = np.float32(0.55)
        effective_core_fraction = np.float32(np.clip(base_core_halfwidth_fraction * safe_scale, 0.05, 1.0))

        core = np.clip(1.0 - (norm_dist / effective_core_fraction), 0.0, 1.0).astype(np.float32)

        # Keep exact bank-edge cells at zero influence even when scale is huge.
        near_bank = bank_vals <= np.float32(1.0e-6)
        if np.any(near_bank):
            core[near_bank] = 0.0

        infl = core.astype(np.float32)

        # Optional outer extent cap if the caller wants to suppress centerline
        # participation beyond a fixed absolute distance in very wide domains.
        if max_distance_m is not None:
            cutoff = np.float32(max(float(max_distance_m), 1.0))
            outer = np.clip(1.0 - (dist / cutoff), 0.0, 1.0).astype(np.float32)
            infl *= outer
    else:
        if max_distance_m is None:
            max_distance_m = np.nanpercentile(np.asarray(dist, dtype=float), 80) if len(dist) else 0.0
        base_cutoff = max(float(max_distance_m), 1.0)
        cutoff = np.float32(max(base_cutoff * safe_scale, 1.0))
        infl = np.clip(1.0 - (dist / cutoff), 0.0, 1.0).astype(np.float32)

    influence[rows, cols] = np.clip(infl, 0.0, 1.0).astype(np.float32)
    return influence


def _nearest_surface_from_points(*, shape, transform, domain_mask: np.ndarray, points_gdf, value_field: str, max_distance_m: Optional[float] = None):
    import rasterio.transform
    try:
        from scipy.spatial import cKDTree as _cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy.spatial.cKDTree is required for river structured scaffold nearest-surface generation") from exc

    out = np.full(shape, np.nan, dtype=np.float32)
    influence = np.zeros(shape, dtype=np.float32)
    if points_gdf is None or getattr(points_gdf, "empty", True) or not np.any(domain_mask):
        return out, influence
    vals = np.asarray(points_gdf.get(value_field, np.full(len(points_gdf), np.nan)), dtype=np.float32)
    valid = np.isfinite(vals)
    if not np.any(valid):
        return out, influence
    pts = np.column_stack([points_gdf.geometry.x.to_numpy(dtype=float)[valid], points_gdf.geometry.y.to_numpy(dtype=float)[valid]])
    vals = vals[valid]
    tree = _cKDTree(pts)
    rows, cols = np.where(np.asarray(domain_mask, dtype=bool))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    q = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    dist, idx = tree.query(q, k=1)
    sampled = vals[np.asarray(idx, dtype=int)]
    out[rows, cols] = sampled.astype(np.float32)
    if max_distance_m is None:
        max_distance_m = np.nanpercentile(np.asarray(dist, dtype=float), 80) if len(dist) else 0.0
    cutoff = max(float(max_distance_m), 1.0)
    infl = np.clip(1.0 - (np.asarray(dist, dtype=np.float32) / cutoff), 0.0, 1.0)
    out_mask = np.asarray(dist, dtype=float) > cutoff
    if np.any(out_mask):
        sampled = sampled.copy()
        sampled[out_mask] = np.nan
        infl[out_mask] = 0.0
        out[rows, cols] = sampled.astype(np.float32)
    influence[rows, cols] = infl.astype(np.float32)
    return out, influence
