from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

from river_section_tendency import (
    build_section_from_thalweg,
    classify_section_tendency_family,
    compute_tendency_depth_fraction,
)
from river_source_semantics import station_target_node_source


def _sample_node_authoritative(points_gdf, mask_path: str | Path | None, depth_path: str | Path | None):
    import rasterio

    auth_mask = np.full(len(points_gdf), np.nan, dtype=np.float32)
    auth_depth = np.full(len(points_gdf), np.nan, dtype=np.float32)
    if points_gdf is None or getattr(points_gdf, "empty", True):
        return auth_mask, auth_depth
    if not mask_path or not depth_path:
        return auth_mask, auth_depth
    mp = Path(mask_path)
    dp = Path(depth_path)
    if not mp.exists() or not dp.exists():
        return auth_mask, auth_depth
    pts = points_gdf
    with rasterio.open(mp) as mds, rasterio.open(dp) as dds:
        try:
            if getattr(pts, "crs", None) is not None and mds.crs is not None and str(pts.crs) != str(mds.crs):
                pts = pts.to_crs(mds.crs)
        except Exception:
            pts = points_gdf
        xy = list(zip(pts.geometry.x.to_numpy(dtype=float), pts.geometry.y.to_numpy(dtype=float)))
        for i, (mraw, draw) in enumerate(zip(mds.sample(xy), dds.sample(xy))):
            mv = float(mraw[0]) if np.size(mraw) else np.nan
            dv = float(draw[0]) if np.size(draw) else np.nan
            if mds.nodata is not None and np.isfinite(mv) and np.isclose(mv, float(mds.nodata)):
                mv = np.nan
            if dds.nodata is not None and np.isfinite(dv) and np.isclose(dv, float(dds.nodata)):
                dv = np.nan
            auth_mask[i] = np.float32(mv) if np.isfinite(mv) else np.float32(np.nan)
            auth_depth[i] = np.float32(dv) if np.isfinite(dv) else np.float32(np.nan)
    return auth_mask, auth_depth


def _safe_float(v) -> float:
    try:
        f = float(v)
    except Exception:
        return float('nan')
    return f if np.isfinite(f) else float('nan')


def _interp_series(stations: np.ndarray, src_s: np.ndarray, src_v: np.ndarray) -> np.ndarray:
    out = np.full(stations.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(src_s) & np.isfinite(src_v)
    if np.count_nonzero(valid) == 0:
        return out
    s = np.asarray(src_s[valid], dtype=float)
    v = np.asarray(src_v[valid], dtype=float)
    order = np.argsort(s)
    s = s[order]
    v = v[order]
    if s.size == 1:
        out[:] = np.float32(v[0])
        return out
    uniq_s, inv = np.unique(np.round(s, 6), return_inverse=True)
    agg = np.full(uniq_s.shape, np.nan, dtype=float)
    for i in range(uniq_s.size):
        sel = inv == i
        agg[i] = float(np.nanmedian(v[sel])) if np.any(sel) else np.nan
    valid2 = np.isfinite(agg)
    if np.count_nonzero(valid2) == 0:
        return out
    if np.count_nonzero(valid2) == 1:
        out[:] = np.float32(agg[valid2][0])
        return out
    out[:] = np.interp(stations, uniq_s[valid2], agg[valid2]).astype(np.float32)
    return out


def _component_normals(centerline):
    geom = centerline.geometry.to_list()
    x = np.asarray([g.x for g in geom], dtype=float)
    y = np.asarray([g.y for g in geom], dtype=float)
    n = len(centerline)
    nx = np.full(n, np.nan, dtype=float)
    ny = np.full(n, np.nan, dtype=float)
    for i in range(n):
        i0 = max(i - 1, 0)
        i1 = min(i + 1, n - 1)
        dx = x[i1] - x[i0]
        dy = y[i1] - y[i0]
        norm = float(np.hypot(dx, dy))
        if norm <= 0.0:
            dx, dy, norm = 1.0, 0.0, 1.0
        tx = dx / norm
        ty = dy / norm
        nx[i] = -ty
        ny[i] = tx
    return nx, ny



def _normalized_component_key(component_id: object) -> str:
    comp = str(component_id).strip()
    if not comp:
        return "main"
    try:
        f = float(comp)
        if np.isfinite(f) and abs(f - round(f)) <= 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return comp


def _default_tendency_inner_z(
    *,
    thalweg_z: float,
    left_bank_z: float,
    right_bank_z: float,
    support_class: str,
    node_role: str,
    effective_width_m: float | None = None,
    component_class: str | None = None,
    reconciliation_confidence: float | None = None,
    support_distance_m: float | None = None,
) -> float:
    if not np.isfinite(thalweg_z) or node_role not in {"left_inner", "right_inner"}:
        return float("nan")
    width = float(effective_width_m) if effective_width_m is not None and np.isfinite(effective_width_m) else 20.0
    family = classify_section_tendency_family(width, support_class)
    frac = compute_tendency_depth_fraction(
        width,
        family,
        support_class,
        component_class=component_class,
        reconciliation_confidence=reconciliation_confidence,
        support_distance_m=support_distance_m,
    )
    left_inner, right_inner, _ = build_section_from_thalweg(
        thalweg_z,
        width,
        family,
        frac,
        bank_caps=(left_bank_z, right_bank_z),
    )
    return float(left_inner if node_role == "left_inner" else right_inner)


def _load_station_targets(station_targets_path: str | Path | None) -> pd.DataFrame:
    path = Path(station_targets_path) if station_targets_path else None
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()
    if 'component_id' not in df.columns or 'station_m' not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out['component_id'] = out['component_id'].map(_normalized_component_key)
    out['station_m'] = pd.to_numeric(out['station_m'], errors='coerce')
    out = out.loc[np.isfinite(out['station_m'])].copy()
    if out.empty:
        return pd.DataFrame()
    out['normalized_station_key'] = [f"{c}|{round(float(s), 6):.6f}" for c, s in zip(out['component_id'], out['station_m'])]
    dup = out['normalized_station_key'].duplicated(keep='last')
    if bool(dup.any()):
        out = out.loc[~dup].copy()
    return out

def _collapse_graph_diagnostics_rows(graph_diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if graph_diagnostics is None or getattr(graph_diagnostics, 'empty', True):
        return pd.DataFrame(), 0
    diag = graph_diagnostics.copy()
    if 'component_id' not in diag.columns or 'station_m' not in diag.columns:
        return diag, 0
    diag['component_id'] = diag['component_id'].fillna('main').astype(str)
    diag['station_m'] = pd.to_numeric(diag['station_m'], errors='coerce')
    diag = diag.loc[np.isfinite(diag['station_m'])].copy()
    if diag.empty:
        return diag, 0
    key_cols = ['component_id', 'station_m']
    dup_count = int(diag.duplicated(subset=key_cols, keep=False).sum())
    if dup_count == 0:
        return diag, 0

    agg: dict[str, object] = {}
    for col in diag.columns:
        if col in key_cols:
            continue
        series = diag[col]
        if pd.api.types.is_bool_dtype(series):
            agg[col] = 'max'
        elif pd.api.types.is_numeric_dtype(series):
            def _agg_numeric(s):
                arr = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
                return float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan
            agg[col] = _agg_numeric
        else:
            def _agg_object(s):
                for v in s:
                    if pd.notna(v) and str(v) != '':
                        return v
                return np.nan
            agg[col] = _agg_object
    diag = diag.groupby(key_cols, as_index=False, sort=False).agg(agg)
    return diag, dup_count




def _smooth_component_backbone(stations: np.ndarray, values: np.ndarray, locked: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    n = out.size
    if n == 0:
        return out.astype(np.float32)
    valid = np.isfinite(out)
    if np.count_nonzero(valid) < 2:
        return out.astype(np.float32)
    window = 3 if n < 7 else 5
    half = window // 2
    smoothed = out.copy()
    for i in range(n):
        if bool(locked[i]) or not np.isfinite(out[i]):
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = out[lo:hi]
        local = local[np.isfinite(local)]
        if local.size >= 2:
            smoothed[i] = float(np.nanmedian(local))
    return smoothed.astype(np.float32)




from river_support_uncertainty import graph_solution_confidence, uncertainty_class_from_confidence

from legacy.river.archive_root_scripts.river_graph_backbone_solver import (
    _build_component_station_candidates as _graph_build_component_station_candidates,
    _build_junction_groups as _graph_build_junction_groups,
    _solve_component_backbone as _graph_solve_component_backbone,
    _solve_network_backbone as _graph_solve_network_backbone,
    summarize_graph_physical_plausibility as _graph_summarize_physical_plausibility,
)


def _build_component_station_candidates(sub: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for the graph backbone solver station-candidate assembly."""
    return _graph_build_component_station_candidates(sub)


def _solve_component_backbone(sub: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """Compatibility wrapper for the graph backbone solver component solve."""
    return _graph_solve_component_backbone(sub)


def _solve_network_backbone(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, int], pd.DataFrame]:
    """Compatibility wrapper for the graph backbone solver network solve."""
    return _graph_solve_network_backbone(frame)


def _build_junction_groups(frame: pd.DataFrame, component_fill: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], float]:
    """Compatibility wrapper for graph-backbone junction grouping."""
    return _graph_build_junction_groups(frame, component_fill)




def _normalized_station_key(component_id: Any, station_m: Any) -> tuple[str, float | None]:
    comp = str(component_id) if component_id is not None else 'main'
    try:
        station = float(station_m)
    except Exception:
        return comp, None
    if not np.isfinite(station):
        return comp, None
    return comp, round(station, 6)

def build_channel_scaffold_products(
    *,
    river_dir: str | Path,
    channel_frame_points_path: str | Path | None,
    xs_bathy_gpkg_path: str | Path | None,
    station_targets_path: str | Path | None = None,
    authoritative_support_mask_path: str | Path | None = None,
    authoritative_support_depth_path: str | Path | None = None,
    allow_absolute_bed_fallback: bool = False,
    disable_xs_influence: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    import geopandas as gpd
    from shapely.geometry import Point

    river_dir = Path(river_dir)
    river_dir.mkdir(parents=True, exist_ok=True)
    frame_path = Path(channel_frame_points_path) if channel_frame_points_path else None
    if frame_path is None or not frame_path.exists():
        return {}
    if bool(allow_absolute_bed_fallback):
        raise RuntimeError("absolute_bed_fallback_removed_from_structured_scaffold")
    frame = gpd.read_file(frame_path)
    if frame is None or frame.empty:
        return {}
    if 'station_m' not in frame.columns:
        return {}
    frame['station_m'] = pd.to_numeric(frame['station_m'], errors='coerce')
    frame = frame.loc[np.isfinite(frame['station_m'])].copy()
    if frame.empty:
        return {}
    if 'component_id' not in frame.columns:
        frame['component_id'] = 'main'
    frame['component_id'] = frame['component_id'].fillna('main').map(_normalized_component_key).astype(str)
    frame['normalized_station_key'] = [_normalized_station_key(c, s) for c, s in zip(frame['component_id'], frame['station_m'])]
    station_targets = _load_station_targets(station_targets_path)
    station_target_loaded = False
    station_target_rows = 0
    if station_targets is not None and not station_targets.empty:
        station_target_loaded = True
        station_target_rows = int(len(station_targets))
        station_targets = station_targets.add_prefix('st_')
        frame = frame.merge(
            station_targets,
            left_on='normalized_station_key',
            right_on='st_normalized_station_key',
            how='left',
            validate='one_to_one',
        )
    frame_duplicate_rows = int(pd.Series(frame['normalized_station_key']).duplicated(keep=False).sum())
    if frame_duplicate_rows > 0:
        raise RuntimeError(
            f"channel_frame_duplicate_station_rows: component_station_keys_not_unique duplicate_rows={frame_duplicate_rows}"
        )

    xs_profiles = None
    xs_path = Path(xs_bathy_gpkg_path) if xs_bathy_gpkg_path else None
    if (not disable_xs_influence) and xs_path is not None and xs_path.exists():
        try:
            xs_profiles = gpd.read_file(xs_path, layer='xs_bathy_points')
        except Exception:
            xs_profiles = None
    if xs_profiles is not None and not xs_profiles.empty:
        z_field = 'xs_z_m' if 'xs_z_m' in xs_profiles.columns else ('z_bed_pred_m' if 'z_bed_pred_m' in xs_profiles.columns else None)
        if z_field is not None and 'dist_m' in xs_profiles.columns:
            xs_profiles['dist_m'] = pd.to_numeric(xs_profiles['dist_m'], errors='coerce')
            xs_profiles['width_m'] = pd.to_numeric(xs_profiles.get('width_m', np.nan), errors='coerce')
            xs_profiles['bed_z_m'] = pd.to_numeric(xs_profiles[z_field], errors='coerce')
            xs_profiles = xs_profiles.loc[np.isfinite(xs_profiles['dist_m']) & np.isfinite(xs_profiles['bed_z_m'])].copy()
        else:
            xs_profiles = None
    else:
        xs_profiles = None

    if xs_profiles is not None and not xs_profiles.empty:
        # Build per-XS station and normalized profile summaries.
        recs = []
        for xs_id, grp in xs_profiles.groupby(xs_profiles.get('xs_id', pd.Series(np.arange(len(xs_profiles))).astype(str)), sort=False):
            g = grp.copy()
            if 'station_m' in g.columns:
                station = float(pd.to_numeric(g['station_m'], errors='coerce').dropna().median()) if pd.to_numeric(g['station_m'], errors='coerce').notna().any() else float('nan')
            else:
                # snap XS profile to nearest frame station
                cx = float(g.geometry.x.mean())
                cy = float(g.geometry.y.mean())
                dx = np.asarray(frame.geometry.x.to_numpy(dtype=float) - cx, dtype=float)
                dy = np.asarray(frame.geometry.y.to_numpy(dtype=float) - cy, dtype=float)
                j = int(np.argmin(dx * dx + dy * dy))
                station = float(frame.iloc[j]['station_m'])
            width = float(pd.to_numeric(g.get('width_m', np.nan), errors='coerce').dropna().median()) if pd.to_numeric(g.get('width_m', np.nan), errors='coerce').notna().any() else float('nan')
            dist = pd.to_numeric(g['dist_m'], errors='coerce').to_numpy(dtype=float)
            z = pd.to_numeric(g['bed_z_m'], errors='coerce').to_numpy(dtype=float)
            if np.isfinite(width) and width > 0.0:
                eta = np.clip(dist / width, 0.0, 1.0)
            else:
                dmin = float(np.nanmin(dist))
                dmax = float(np.nanmax(dist))
                if np.isfinite(dmin) and np.isfinite(dmax) and dmax > dmin:
                    eta = np.clip((dist - dmin) / (dmax - dmin), 0.0, 1.0)
                    width = dmax - dmin
                else:
                    continue
            bins = [0.0, 0.25, 0.5, 0.75, 1.0]
            prof = {'station_m': station, 'width_m': width}
            for b in bins:
                sel = np.abs(eta - b) <= 0.15
                prof[f'eta_{b:.2f}'] = float(np.nanmedian(z[sel])) if np.any(sel) else float('nan')
            recs.append(prof)
        xs_summary = pd.DataFrame.from_records(recs)
    else:
        xs_summary = pd.DataFrame(columns=['station_m','width_m','eta_0.00','eta_0.25','eta_0.50','eta_0.75','eta_1.00'])

    nodes = []
    bins = [('left_bank', 0.00), ('left_inner', 0.25), ('thalweg', 0.50), ('right_inner', 0.75), ('right_bank', 1.00)]
    total_auth_nodes = 0
    total_interp_nodes = 0
    width_src = xs_summary['width_m'].to_numpy(dtype=float) if not xs_summary.empty else np.asarray([], dtype=float)
    s_src = xs_summary['station_m'].to_numpy(dtype=float) if not xs_summary.empty else np.asarray([], dtype=float)
    component_width_interp: dict[str, np.ndarray] = {}
    for comp, sub in frame.groupby('component_id', sort=False):
        sub = sub.sort_values('station_m').copy()
        stations = sub['station_m'].to_numpy(dtype=float)
        width_interp = _interp_series(stations, s_src, width_src)
        if not np.any(np.isfinite(width_interp)):
            diffs = np.diff(np.unique(np.round(stations, 6))) if stations.size > 1 else np.asarray([], dtype=float)
            fallback_width = float(np.nanmedian(diffs)) if diffs.size else 20.0
            width_interp[:] = np.float32(max(fallback_width, 5.0))
        component_width_interp[str(comp)] = np.asarray(width_interp, dtype=np.float32)
    component_backbone_fill_map, junction_metrics, graph_diagnostics = _solve_network_backbone(frame)
    graph_diagnostics_available = graph_diagnostics is not None and not getattr(graph_diagnostics, 'empty', True)
    if graph_diagnostics is None:
        graph_diagnostics = pd.DataFrame()
    graph_diag_duplicate_count = 0
    if not graph_diagnostics.empty:
        graph_diag_merge, graph_diag_duplicate_count = _collapse_graph_diagnostics_rows(graph_diagnostics)
        if graph_diag_duplicate_count > 0 and logger is not None:
            logger.warning(
                "[RIVER][SCAFFOLD] Collapsed duplicate graph diagnostics rows on (component_id, station_m): %d",
                graph_diag_duplicate_count,
            )
        graph_diag_merge["component_id"] = graph_diag_merge["component_id"].astype(str)
        graph_diag_merge["station_m"] = pd.to_numeric(graph_diag_merge["station_m"], errors="coerce")
        frame = frame.merge(graph_diag_merge, on=["component_id", "station_m"], how="left", validate="one_to_one")
    else:
        frame["graph_backbone_z_m"] = np.nan
        frame["graph_hard_lock"] = False
        frame["graph_prior_weight_sum"] = 0.0
        frame["graph_edge_weight_sum"] = 0.0
        frame["graph_curvature_weight_sum"] = 0.0
        frame["graph_centering_weight_sum"] = 0.0
        frame["graph_regularization_weight_sum"] = 0.0
        frame["graph_junction_weight_sum"] = 0.0
        frame["graph_junction_constrained"] = False
        frame["graph_residual_to_candidate_z_m"] = np.nan
        frame["graph_solver_support_class"] = "unsupported"
        frame["graph_candidate_source"] = "missing"
        frame["graph_solution_mode"] = "missing"
        frame["graph_unsupported_span_m"] = 0.0
        frame["graph_slope_guard_weight_sum"] = 0.0
        frame["graph_adverse_step_weight_sum"] = 0.0
        frame["graph_physical_guard_weight_sum"] = 0.0
        frame["graph_local_slope"] = np.nan
        frame["graph_local_curvature"] = np.nan
        frame["graph_slope_guard_active"] = False
        frame["graph_adverse_step_guard_active"] = False
    # Second pass: build scaffold nodes using the solved component/junction backbone.
    for comp, sub in frame.groupby('component_id', sort=False):
        sub = sub.sort_values('station_m').copy()
        stations = sub['station_m'].to_numpy(dtype=float)
        nx, ny = _component_normals(sub)
        width_interp = np.asarray(component_width_interp.get(str(comp), np.full(stations.shape, np.nan, dtype=np.float32)), dtype=float)
        if width_interp.size != stations.size:
            raise RuntimeError(
                f"channel_scaffold_component_width_length_mismatch: component={comp} stations={stations.size} widths={width_interp.size}"
            )
        def _numeric_col(name: str) -> np.ndarray:
            if name not in sub.columns:
                return np.full(stations.shape, np.nan, dtype=float)
            return pd.to_numeric(sub[name], errors='coerce').to_numpy(dtype=float)

        def _structural_profile_scale(*, support_class: str, node_role: str, has_actual_xs_support: bool) -> float:
            if has_actual_xs_support:
                return 1.0
            support_class = str(support_class or 'unsupported')
            node_role = str(node_role or 'thalweg')
            if support_class == 'anchored_interpolated':
                base = 0.65
            elif support_class == 'stage_controlled':
                base = 0.50
            elif support_class in {'resolved_backbone', 'graph_backbone'}:
                base = 0.40
            elif support_class == 'unsupported':
                base = 0.30
            else:
                base = 0.35
            if node_role in {'left_inner', 'right_inner'}:
                return float(base)
            if node_role == 'thalweg':
                return float(min(base, 0.45))
            return float(min(base, 0.55))

        frame_auth = _numeric_col('authoritative_hard_bed_z_m')
        frame_auth_bank = _numeric_col('authoritative_bank_margin_z_m')
        frame_backbone = _numeric_col('authoritative_backbone_z_m')
        frame_xs_residual = _numeric_col('xs_residual_to_backbone_z_m')
        frame_stage = _numeric_col('resolved_stage_control_z_m')
        frame_left_bank_fit = _numeric_col('left_bank_fit_z_m')
        frame_right_bank_fit = _numeric_col('right_bank_fit_z_m')
        frame_bank_pair_fit = _numeric_col('bank_pair_fit_z_m')
        frame_active_core = _numeric_col('active_core_support_z_m')
        graph_backbone = _numeric_col('graph_backbone_z_m')
        graph_support_class = sub.get('graph_solver_support_class', pd.Series('unsupported', index=sub.index)).fillna('unsupported').astype(str).to_numpy()
        graph_candidate_source = sub.get('graph_candidate_source', pd.Series('missing', index=sub.index)).fillna('missing').astype(str).to_numpy()
        graph_solution_mode = sub.get('graph_solution_mode', pd.Series('missing', index=sub.index)).fillna('missing').astype(str).to_numpy()
        component_backbone_fill = np.asarray(component_backbone_fill_map.get(str(comp), np.full(stations.shape, np.nan, dtype=np.float32)), dtype=float)
        prof_cols = {b: _interp_series(stations, s_src, xs_summary[f'eta_{b:.2f}'].to_numpy(dtype=float) if f'eta_{b:.2f}' in xs_summary else np.asarray([], dtype=float)) for _, b in bins}
        prof_thalweg = prof_cols.get(0.5, np.full(stations.shape, np.nan, dtype=np.float32))
        for i, (_, row) in enumerate(sub.iterrows()):
            x = float(row.geometry.x)
            y = float(row.geometry.y)
            width = float(width_interp[i]) if np.isfinite(width_interp[i]) else 20.0
            auth = float(frame_auth[i]) if np.isfinite(frame_auth[i]) else float('nan')
            auth_bank = float(frame_auth_bank[i]) if np.isfinite(frame_auth_bank[i]) else float('nan')
            stage = float(frame_stage[i]) if np.isfinite(frame_stage[i]) else float('nan')
            left_bank_fit = float(frame_left_bank_fit[i]) if np.isfinite(frame_left_bank_fit[i]) else (stage if np.isfinite(stage) else float('nan'))
            right_bank_fit = float(frame_right_bank_fit[i]) if np.isfinite(frame_right_bank_fit[i]) else (stage if np.isfinite(stage) else float('nan'))
            bank_pair_fit = float(frame_bank_pair_fit[i]) if np.isfinite(frame_bank_pair_fit[i]) else (stage if np.isfinite(stage) else float('nan'))
            active_core_fit = float(frame_active_core[i]) if np.isfinite(frame_active_core[i]) else float('nan')
            for node_role, eta in bins:
                z_prof = float(prof_cols[eta][i]) if np.isfinite(prof_cols[eta][i]) else float('nan')
                z_prof_thalweg = float(prof_thalweg[i]) if np.isfinite(prof_thalweg[i]) else float('nan')
                z_prof_resid = (z_prof - z_prof_thalweg) if (np.isfinite(z_prof) and np.isfinite(z_prof_thalweg)) else float('nan')
                frame_resid = float(frame_xs_residual[i]) if np.isfinite(frame_xs_residual[i]) else float('nan')
                if disable_xs_influence:
                    frame_resid = float('nan')
                if not np.isfinite(z_prof_resid) and np.isfinite(frame_resid):
                    z_prof_resid = frame_resid
                row_channel_support_class = str(row.get('channel_support_class', 'unsupported') or 'unsupported')
                station_bed_support = bool(row.get('authoritative_bed_support_present', False))
                station_bank_margin_support = bool(row.get('authoritative_bank_margin_present', False))
                target_available = bool(pd.notna(row.get('st_target_source_class')))
                target_left_bank_z = _safe_float(row.get('st_target_left_bank_z_m'))
                target_right_bank_z = _safe_float(row.get('st_target_right_bank_z_m'))
                target_left_inner_z = _safe_float(row.get('st_target_left_inner_z_m'))
                target_right_inner_z = _safe_float(row.get('st_target_right_inner_z_m'))
                target_thalweg_z = _safe_float(row.get('st_target_thalweg_z_m'))
                target_effective_width_m = _safe_float(row.get('st_target_effective_channel_width_m'))
                target_section_tendency_family = str(row.get('st_section_tendency_family', 'missing') or 'missing') if target_available else 'missing'
                target_section_tendency_source = str(row.get('st_section_tendency_source', 'missing') or 'missing') if target_available else 'missing'
                if node_role == 'left_bank':
                    target_role_z = target_left_bank_z
                elif node_role == 'right_bank':
                    target_role_z = target_right_bank_z
                elif node_role == 'left_inner':
                    target_role_z = target_left_inner_z
                elif node_role == 'right_inner':
                    target_role_z = target_right_inner_z
                else:
                    target_role_z = target_thalweg_z
                target_residual_allowed = bool(row.get('st_xs_residual_allowed', False)) if target_available else True
                target_hard_anchor = bool(row.get('st_hard_anchor_present', False)) if target_available else False
                target_anchor_class = str(row.get('st_anchor_class', 'non_anchor') or 'non_anchor') if target_available else 'non_anchor'
                target_anchor_exact = bool(row.get('st_anchor_exact', False)) if target_available else False
                target_anchor_locks_core = bool(row.get('st_anchor_locks_core', target_hard_anchor)) if target_available else target_hard_anchor
                target_anchor_blocks_rebuild = bool(row.get('st_anchor_blocks_rebuild', target_hard_anchor)) if target_available else target_hard_anchor
                target_source_class = str(row.get('st_target_source_class', 'missing') or 'missing') if target_available else 'missing'
                target_policy_reason = str(row.get('st_target_policy_reason', 'missing') or 'missing') if target_available else 'missing'
                target_local_auth_taper = _safe_float(row.get('st_authoritative_reconciliation_delta_m', row.get('st_generalized_longitudinal_bed_local_auth_taper_m')) )
                target_local_auth_reconciliation_weight = _safe_float(row.get('st_authoritative_reconciliation_weight', row.get('st_generalized_longitudinal_bed_local_auth_reconciliation_weight')) )
                target_local_auth_reconciliation_confidence = _safe_float(row.get('st_authoritative_reconciliation_confidence'))
                target_local_auth_support_distance = _safe_float(row.get('st_authoritative_reconciliation_support_distance_m'))
                target_local_auth_reconciled = bool(np.isfinite(target_local_auth_taper) and abs(target_local_auth_taper) > 1.0e-6)
                force_station_target = bool(target_available and (not station_bed_support) and (not bool(row.get('true_measured_xs_qualified', False))))
                graph_backbone_here = float(graph_backbone[i]) if np.isfinite(graph_backbone[i]) else float('nan')
                fallback_backbone_here = float(component_backbone_fill[i]) if np.isfinite(component_backbone_fill[i]) else float('nan')
                solved_backbone = graph_backbone_here if np.isfinite(graph_backbone_here) else (fallback_backbone_here if (not graph_diagnostics_available and np.isfinite(fallback_backbone_here)) else float('nan'))
                solved_support_class = str(graph_support_class[i]) if i < len(graph_support_class) else 'unsupported'
                solved_candidate_source = str(graph_candidate_source[i]) if i < len(graph_candidate_source) else 'missing'
                has_actual_xs_residual_support = False if disable_xs_influence else bool((((np.isfinite(frame_resid) and abs(float(frame_resid)) > 1e-6) or solved_candidate_source == 'xs_profile_resampled' or solved_support_class == 'xs_residual_only')) and target_residual_allowed)
                structural_profile_scale = _structural_profile_scale(
                    support_class=solved_support_class,
                    node_role=node_role,
                    has_actual_xs_support=has_actual_xs_residual_support,
                )
                graph_backbone_missing = graph_diagnostics_available and (not np.isfinite(graph_backbone_here)) and np.isfinite(fallback_backbone_here)
                bank_node = bool(node_role in {'left_bank', 'right_bank'})
                bank_margin_only_station = bool((not station_bed_support) and station_bank_margin_support)
                if np.isfinite(auth):
                    base_bed = auth
                    base_source = 'authoritative_in_channel'
                elif np.isfinite(active_core_fit):
                    base_bed = active_core_fit
                    base_source = 'active_core_support'
                elif np.isfinite(solved_backbone):
                    base_bed = solved_backbone
                    if solved_candidate_source and solved_candidate_source != 'missing':
                        base_source = solved_candidate_source
                    elif solved_support_class == 'stage_controlled':
                        base_source = 'bank_stage_prior'
                    else:
                        base_source = 'resolved_backbone'
                elif np.isfinite(stage):
                    base_bed = stage
                    base_source = 'bank_stage_prior'
                else:
                    base_bed = float('nan')
                    base_source = 'missing'

                station_target_z_source = station_target_node_source(local_reconciled=target_local_auth_reconciled)
                if force_station_target and np.isfinite(target_thalweg_z):
                    base_bed = float(target_thalweg_z)
                    base_source = station_target_z_source
                auth_station = bool(
                    station_bed_support
                    or np.isfinite(frame_auth[i])
                    or solved_support_class == 'authoritative_locked'
                    or base_source == 'authoritative_in_channel'
                    or bool(row.get('graph_hard_lock', False))
                    or target_anchor_locks_core
                )
                backbone_bed = float(base_bed) if np.isfinite(base_bed) else float('nan')
                residual_shape = float('nan')
                if np.isfinite(base_bed):
                    fitted_bank_role_z = left_bank_fit if node_role == 'left_bank' else (right_bank_fit if node_role == 'right_bank' else float('nan'))
                    default_tendency_inner_z = _default_tendency_inner_z(
                        thalweg_z=base_bed,
                        left_bank_z=target_left_bank_z if np.isfinite(target_left_bank_z) else left_bank_fit,
                        right_bank_z=target_right_bank_z if np.isfinite(target_right_bank_z) else right_bank_fit,
                        support_class=str(row_channel_support_class or 'unsupported'),
                        node_role=node_role,
                        effective_width_m=target_effective_width_m,
                        component_class=str(row.get('component_support_class', 'unknown') or 'unknown'),
                        reconciliation_confidence=_safe_float(row.get('authoritative_reconciliation_confidence')),
                        support_distance_m=_safe_float(row.get('component_support_median_distance_m')),
                    )
                    no_local_xs_shape = (not has_actual_xs_residual_support)
                    if auth_station:
                        station_mode = 'authoritative_in_channel'
                        if bank_node and np.isfinite(auth_bank):
                            z = auth_bank
                            residual_shape = float(0.0)
                            z_source = 'authoritative_bank_margin'
                        elif bank_node and np.isfinite(fitted_bank_role_z):
                            z = fitted_bank_role_z
                            residual_shape = float(0.0)
                            z_source = 'bank_fit_profile'
                        elif no_local_xs_shape:
                            if np.isfinite(target_role_z):
                                z = target_role_z
                                residual_shape = float(0.0)
                                z_source = station_target_z_source
                            elif np.isfinite(default_tendency_inner_z):
                                z = default_tendency_inner_z
                                residual_shape = float(0.0)
                                z_source = 'generalized_thalweg_default_tendency'
                            else:
                                z = base_bed
                                residual_shape = float(0.0)
                                z_source = 'authoritative_in_channel'
                        elif np.isfinite(z_prof_resid):
                            residual_shape = float(np.clip(z_prof_resid, -1.5, 1.5))
                            z = base_bed + residual_shape
                            z_source = 'authoritative_in_channel'
                        else:
                            z = base_bed
                            residual_shape = float(0.0)
                            z_source = 'authoritative_in_channel'
                        total_auth_nodes += 1
                    elif bank_margin_only_station and bank_node and np.isfinite(auth_bank):
                        z = auth_bank
                        z_source = 'authoritative_bank_margin'
                        backbone_bed = float(base_bed) if np.isfinite(base_bed) else float(auth_bank)
                        residual_shape = float(0.0)
                        station_mode = 'bank_margin_authoritative'
                        total_interp_nodes += 1
                    else:
                        if force_station_target:
                            station_mode = 'station_target'
                        elif solved_support_class == 'stage_controlled':
                            station_mode = 'xs_supported' if has_actual_xs_residual_support else 'stage_controlled'
                        else:
                            station_mode = 'xs_supported' if has_actual_xs_residual_support else 'resolved_backbone'
                        if force_station_target and np.isfinite(target_role_z):
                            z = target_role_z
                            z_source = station_target_z_source
                            residual_shape = float(0.0)
                        elif no_local_xs_shape:
                            if np.isfinite(target_role_z):
                                z = target_role_z
                                z_source = station_target_z_source
                            elif bank_node and np.isfinite(fitted_bank_role_z):
                                z = fitted_bank_role_z
                                z_source = 'bank_fit_profile'
                            elif np.isfinite(default_tendency_inner_z):
                                z = default_tendency_inner_z
                                z_source = 'generalized_thalweg_default_tendency'
                            else:
                                z = base_bed
                                z_source = 'active_core_support' if base_source in {'active_core_support', 'station_target'} else 'graph_backbone'
                            residual_shape = float(0.0)
                        else:
                            z_source = 'graph_backbone'
                            if np.isfinite(z_prof_resid):
                                scaled_residual = float(z_prof_resid) * float(structural_profile_scale)
                                residual_shape = float(scaled_residual)
                                z = base_bed + scaled_residual
                            else:
                                z = base_bed
                                residual_shape = float(0.0)
                        total_interp_nodes += 1
                elif np.isfinite(z_prof):
                    z = z_prof
                    z_source = 'xs_profile_resampled'
                    backbone_bed = float('nan')
                    residual_shape = float('nan')
                    station_mode = 'xs_only'
                    total_interp_nodes += 1
                elif np.isfinite(stage):
                    z = stage
                    z_source = 'bank_stage_prior'
                    backbone_bed = float(stage)
                    residual_shape = float(0.0)
                    station_mode = 'bank_stage_only'
                else:
                    z = float('nan')
                    z_source = 'missing'
                    backbone_bed = float('nan')
                    residual_shape = float('nan')
                    station_mode = 'missing'
                offset = (eta - 0.5) * width
                gx = x + float(nx[i]) * offset
                gy = y + float(ny[i]) * offset
                nodes.append({
                    'component_id': str(comp),
                    'station_m': float(stations[i]),
                    'node_role': node_role,
                    'cross_stream_eta': float(eta),
                    'half_width_m': float(width * 0.5),
                    'bed_z_m': np.float32(z) if np.isfinite(z) else np.float32(np.nan),
                    'z_source': z_source,
                    'backbone_bed_z_m': np.float32(backbone_bed) if np.isfinite(backbone_bed) else np.float32(np.nan),
                    'bank_pair_fit_z_m': np.float32(bank_pair_fit) if np.isfinite(bank_pair_fit) else np.float32(np.nan),
                    'left_bank_fit_z_m': np.float32(left_bank_fit) if np.isfinite(left_bank_fit) else np.float32(np.nan),
                    'right_bank_fit_z_m': np.float32(right_bank_fit) if np.isfinite(right_bank_fit) else np.float32(np.nan),
                    'active_core_fit_z_m': np.float32(active_core_fit) if np.isfinite(active_core_fit) else np.float32(np.nan),
                    'target_left_bank_z_m': np.float32(target_left_bank_z) if np.isfinite(target_left_bank_z) else np.float32(np.nan),
                    'target_right_bank_z_m': np.float32(target_right_bank_z) if np.isfinite(target_right_bank_z) else np.float32(np.nan),
                    'target_left_inner_z_m': np.float32(target_left_inner_z) if np.isfinite(target_left_inner_z) else np.float32(np.nan),
                    'target_right_inner_z_m': np.float32(target_right_inner_z) if np.isfinite(target_right_inner_z) else np.float32(np.nan),
                    'target_effective_channel_width_m': np.float32(target_effective_width_m) if np.isfinite(target_effective_width_m) else np.float32(np.nan),
                    'section_tendency_family': target_section_tendency_family,
                    'section_tendency_source': target_section_tendency_source,
                    'target_thalweg_z_m': np.float32(target_thalweg_z) if np.isfinite(target_thalweg_z) else np.float32(np.nan),
                    'generalized_longitudinal_bed_local_auth_taper_m': np.float32(target_local_auth_taper) if np.isfinite(target_local_auth_taper) else np.float32(np.nan),
                    'generalized_longitudinal_bed_local_auth_reconciliation_weight': np.float32(target_local_auth_reconciliation_weight) if np.isfinite(target_local_auth_reconciliation_weight) else np.float32(np.nan),
                    'authoritative_reconciliation_delta_m': np.float32(target_local_auth_taper) if np.isfinite(target_local_auth_taper) else np.float32(np.nan),
                    'authoritative_reconciliation_weight': np.float32(target_local_auth_reconciliation_weight) if np.isfinite(target_local_auth_reconciliation_weight) else np.float32(np.nan),
                    'authoritative_reconciliation_confidence': np.float32(target_local_auth_reconciliation_confidence) if np.isfinite(target_local_auth_reconciliation_confidence) else np.float32(np.nan),
                    'authoritative_reconciliation_support_distance_m': np.float32(target_local_auth_support_distance) if np.isfinite(target_local_auth_support_distance) else np.float32(np.nan),
                    'station_target_local_authoritative_reconciled': bool(target_local_auth_reconciled),
                    'target_anchor_class': target_anchor_class,
                    'target_anchor_exact': bool(target_anchor_exact),
                    'target_anchor_locks_core': bool(target_anchor_locks_core),
                    'target_anchor_blocks_rebuild': bool(target_anchor_blocks_rebuild),
                    'target_xs_residual_allowed': bool(False if disable_xs_influence else target_residual_allowed),
                    'target_xs_realism_allowed': bool(False if disable_xs_influence else (bool(row.get('st_xs_realism_allowed', False)) if target_available else True)),
                    'target_rebuild_allowed': bool((bool(row.get('st_rebuild_allowed', False)) if target_available else True)),
                    'target_post_rebuild_monotone_required': bool(row.get('st_post_rebuild_monotone_required', False)) if target_available else False,
                    'residual_shape_z_m': np.float32(residual_shape) if np.isfinite(residual_shape) else np.float32(np.nan),
                    'station_support_mode': station_mode,
                    'station_target_present': bool(target_available),
                    'station_target_source_class': target_source_class,
                    'station_target_policy_reason': target_policy_reason,
                    'channel_support_class': row_channel_support_class,
                    'station_authoritative_backed': bool(auth_station),
                    'station_authoritative_role': str(row.get('authoritative_role', 'no_authoritative_support') or 'no_authoritative_support'),
                    'station_authoritative_bed_support_present': bool(row.get('authoritative_bed_support_present', False)),
                    'station_authoritative_bank_margin_present': bool(row.get('authoritative_bank_margin_present', False)),
                    'graph_backbone_missing': bool(graph_backbone_missing),
                    'graph_candidate_source': str(row.get('graph_candidate_source', 'missing')),
                    'graph_solution_mode': str(row.get('graph_solution_mode', 'missing')),
                    'graph_solver_support_class': str(row.get('graph_solver_support_class', 'unsupported')),
                    'graph_hard_lock': bool(row.get('graph_hard_lock', False)),
                    'graph_junction_constrained': bool(row.get('graph_junction_constrained', False)),
                    'graph_junction_id': str(row.get('graph_junction_id', 'not_in_junction') or 'not_in_junction'),
                    'graph_junction_role': str(row.get('graph_junction_role', 'not_in_junction') or 'not_in_junction'),
                    'graph_junction_target_z_m': np.float32(_safe_float(row.get('graph_junction_target_z_m'))),
                    'graph_junction_adjustment_z_m': np.float32(_safe_float(row.get('graph_junction_adjustment_z_m'))),
                    'graph_junction_distance_m': np.float32(_safe_float(row.get('graph_junction_distance_m'))),
                    'graph_junction_influence_weight': np.float32(_safe_float(row.get('graph_junction_influence_weight'))),
                    'graph_junction_topology_source': str(row.get('graph_junction_topology_source', 'none') or 'none'),
                    'graph_prior_weight_sum': np.float32(_safe_float(row.get('graph_prior_weight_sum'))),
                    'graph_regularization_weight_sum': np.float32(_safe_float(row.get('graph_regularization_weight_sum'))),
                    'graph_junction_weight_sum': np.float32(_safe_float(row.get('graph_junction_weight_sum'))),
                    'graph_residual_to_candidate_z_m': np.float32(_safe_float(row.get('graph_residual_to_candidate_z_m'))),
                    'graph_unsupported_span_m': np.float32(_safe_float(row.get('graph_unsupported_span_m'))),
                    'graph_unsupported_regime': str(row.get('graph_unsupported_regime', 'missing') or 'missing'),
                    'graph_slope_guard_weight_sum': np.float32(_safe_float(row.get('graph_slope_guard_weight_sum'))),
                    'graph_adverse_step_weight_sum': np.float32(_safe_float(row.get('graph_adverse_step_weight_sum'))),
                    'graph_physical_guard_weight_sum': np.float32(_safe_float(row.get('graph_physical_guard_weight_sum'))),
                    'graph_local_slope': np.float32(_safe_float(row.get('graph_local_slope'))),
                    'graph_local_curvature': np.float32(_safe_float(row.get('graph_local_curvature'))),
                    'graph_slope_guard_active': bool(row.get('graph_slope_guard_active', False)),
                    'graph_adverse_step_guard_active': bool(row.get('graph_adverse_step_guard_active', False)),
                    'base_support_class': str(row.get('channel_support_class', 'unsupported')),
                    'component_support_class': str(row.get('component_support_class', 'unknown') or 'unknown'),
                    'component_support_is_mainstem': bool(row.get('component_support_is_mainstem', True)),
                    'component_support_length_m': np.float32(_safe_float(row.get('component_support_length_m'))),
                    'component_support_bed_fraction': np.float32(_safe_float(row.get('component_support_bed_fraction'))),
                    'component_support_anchor_fraction': np.float32(_safe_float(row.get('component_support_anchor_fraction'))),
                    'component_support_anchor_count': int(_safe_float(row.get('component_support_anchor_count')) if np.isfinite(_safe_float(row.get('component_support_anchor_count'))) else 0),
                    'component_support_median_distance_m': np.float32(_safe_float(row.get('component_support_median_distance_m'))),
                    'component_support_smoothing_regime': str(row.get('component_support_smoothing_regime', row.get('component_support_class', 'unknown')) or 'unknown'),
                    'center_x': float(x),
                    'center_y': float(y),
                    'normal_x': float(nx[i]),
                    'normal_y': float(ny[i]),
                    'geometry': Point(gx, gy),
                })
    if not nodes:
        return {}
    import geopandas as gpd
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry='geometry', crs=frame.crs)
    auth_mask_nodes, auth_depth_nodes = _sample_node_authoritative(nodes_gdf, authoritative_support_mask_path, authoritative_support_depth_path)
    auth_local = np.isfinite(auth_mask_nodes) & (auth_mask_nodes > 0.0) & np.isfinite(auth_depth_nodes)
    node_role_str = nodes_gdf['node_role'].astype(str)
    station_bed_support = nodes_gdf.get('station_authoritative_bed_support_present', pd.Series(False, index=nodes_gdf.index)).fillna(False).astype(bool).to_numpy(dtype=bool)
    station_bank_margin_support = nodes_gdf.get('station_authoritative_bank_margin_present', pd.Series(False, index=nodes_gdf.index)).fillna(False).astype(bool).to_numpy(dtype=bool)
    bank_role_nodes = node_role_str.isin(['left_bank', 'right_bank']).to_numpy(dtype=bool)
    auth_local_bed = auth_local & station_bed_support
    auth_local_bank = auth_local & (~station_bed_support) & station_bank_margin_support & bank_role_nodes
    if np.any(auth_local_bed):
        nodes_gdf.loc[auth_local_bed, 'bed_z_m'] = auth_depth_nodes[auth_local_bed].astype(np.float32)
        nodes_gdf.loc[auth_local_bed, 'z_source'] = 'authoritative_in_channel'
    if np.any(auth_local_bank):
        nodes_gdf.loc[auth_local_bank, 'bed_z_m'] = auth_depth_nodes[auth_local_bank].astype(np.float32)
        nodes_gdf.loc[auth_local_bank, 'z_source'] = 'authoritative_bank_margin'
    # If any node at a station is authoritative-supported, promote the whole station to an authoritative-backed scaffold
    # so XS shape only acts as a bounded residual around the local authoritative bed/backbone.
    station_cols = ['component_id', 'station_m']
    station_auth = (
        nodes_gdf.assign(_auth=(
            nodes_gdf['z_source'].astype(str).eq('authoritative_in_channel')
            | nodes_gdf.get('graph_hard_lock', False).astype(bool)
        ))
        .groupby(station_cols, dropna=False)['_auth']
        .transform('any')
        .to_numpy(dtype=bool)
    )
    if np.any(station_auth):
        thalweg_station = station_auth & nodes_gdf['node_role'].astype(str).eq('thalweg').to_numpy()
        nodes_gdf.loc[thalweg_station, 'z_source'] = 'authoritative_in_channel'
        nonauth = station_auth & (~nodes_gdf['z_source'].astype(str).eq('authoritative_in_channel').to_numpy())
        nodes_gdf.loc[nonauth, 'z_source'] = 'authoritative_backbone'

    frame_station_candidate = frame.get('true_measured_xs_candidate', pd.Series(False, index=frame.index)).fillna(False).astype(bool).to_numpy(dtype=bool)
    frame_station_qualified = frame.get('true_measured_xs_qualified', pd.Series(False, index=frame.index)).fillna(False).astype(bool).to_numpy(dtype=bool)
    frame_station_df = frame[['component_id', 'station_m']].copy()
    frame_station_df['true_measured_xs_candidate_station'] = frame_station_candidate.astype(bool)
    frame_station_df['true_measured_xs_qualified_station'] = frame_station_qualified.astype(bool)
    frame_station_df['bank_only_authoritative_station'] = frame.get('bank_only_authoritative', pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    frame_station_df['candidate_support_class'] = frame.get('channel_support_class', pd.Series('missing', index=frame.index)).fillna('missing').astype(str)
    frame_station_df['auth_xs_support_class'] = frame.get('auth_xs_support_class', pd.Series('authoritative_unknown', index=frame.index)).fillna('authoritative_unknown').astype(str)
    frame_station_df['auth_xs_support_inner_count'] = pd.to_numeric(frame.get('auth_xs_support_inner_count', pd.Series(0.0, index=frame.index)), errors='coerce').fillna(0.0)
    frame_station_df['auth_xs_support_bank_count'] = pd.to_numeric(frame.get('auth_xs_support_bank_count', pd.Series(0.0, index=frame.index)), errors='coerce').fillna(0.0)
    frame_station_df['station_authoritative_role'] = frame.get('authoritative_role', pd.Series('no_authoritative_support', index=frame.index)).fillna('no_authoritative_support').astype(str)
    frame_station_df['station_authoritative_role_confidence'] = pd.to_numeric(frame.get('authoritative_role_confidence', pd.Series(np.nan, index=frame.index)), errors='coerce')
    frame_station_df['station_authoritative_distance_to_bank_m'] = pd.to_numeric(frame.get('authoritative_distance_to_bank_m', pd.Series(np.nan, index=frame.index)), errors='coerce')
    frame_station_df['station_authoritative_normalized_channel_position'] = pd.to_numeric(frame.get('authoritative_normalized_channel_position', pd.Series(np.nan, index=frame.index)), errors='coerce')
    frame_station_df['station_authoritative_bed_support_present'] = frame.get('authoritative_bed_support_present', pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    frame_station_df['station_authoritative_bank_margin_present'] = frame.get('authoritative_bank_margin_present', pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    node_role = nodes_gdf['node_role'].astype(str)
    z_source = nodes_gdf['z_source'].astype(str)
    station_mode = nodes_gdf['station_support_mode'].astype(str)
    eligible_measured_roles = node_role.isin(['thalweg', 'left_inner', 'right_inner']).to_numpy(dtype=bool)
    frame_station_df['normalized_station_key'] = [_normalized_station_key(c, s) for c, s in zip(frame_station_df['component_id'], frame_station_df['station_m'])]
    candidate_lookup = frame_station_df.set_index('normalized_station_key')['true_measured_xs_candidate_station'].to_dict()
    qualified_lookup = frame_station_df.set_index('normalized_station_key')['true_measured_xs_qualified_station'].to_dict()
    bank_only_lookup = frame_station_df.set_index('normalized_station_key')['bank_only_authoritative_station'].to_dict()
    auth_class_lookup = frame_station_df.set_index('normalized_station_key')['auth_xs_support_class'].to_dict()
    auth_inner_lookup = frame_station_df.set_index('normalized_station_key')['auth_xs_support_inner_count'].to_dict()
    auth_bank_lookup = frame_station_df.set_index('normalized_station_key')['auth_xs_support_bank_count'].to_dict()
    auth_role_lookup = frame_station_df.set_index('normalized_station_key')['station_authoritative_role'].to_dict()
    auth_role_conf_lookup = frame_station_df.set_index('normalized_station_key')['station_authoritative_role_confidence'].to_dict()
    auth_dist_lookup = frame_station_df.set_index('normalized_station_key')['station_authoritative_distance_to_bank_m'].to_dict()
    auth_norm_lookup = frame_station_df.set_index('normalized_station_key')['station_authoritative_normalized_channel_position'].to_dict()
    auth_bed_lookup = frame_station_df.set_index('normalized_station_key')['station_authoritative_bed_support_present'].to_dict()
    auth_bankmargin_lookup = frame_station_df.set_index('normalized_station_key')['station_authoritative_bank_margin_present'].to_dict()
    nodes_gdf['normalized_station_key'] = [_normalized_station_key(c, s) for c, s in zip(nodes_gdf['component_id'], nodes_gdf['station_m'])]
    nodes_gdf['station_true_measured_xs_candidate'] = [bool(candidate_lookup.get(k, False)) for k in nodes_gdf['normalized_station_key']]
    nodes_gdf['station_true_measured_xs_qualified'] = [bool(qualified_lookup.get(k, False)) for k in nodes_gdf['normalized_station_key']]
    nodes_gdf['station_bank_only_authoritative'] = [bool(bank_only_lookup.get(k, False)) for k in nodes_gdf['normalized_station_key']]
    nodes_gdf['auth_xs_support_class'] = [str(auth_class_lookup.get(k, 'authoritative_unknown') or 'authoritative_unknown') for k in nodes_gdf['normalized_station_key']]
    nodes_gdf['auth_xs_support_inner_count'] = [float(auth_inner_lookup.get(k, 0.0) or 0.0) for k in nodes_gdf['normalized_station_key']]
    nodes_gdf['auth_xs_support_bank_count'] = [float(auth_bank_lookup.get(k, 0.0) or 0.0) for k in nodes_gdf['normalized_station_key']]
    nodes_gdf['station_authoritative_role'] = [str(auth_role_lookup.get(k, 'no_authoritative_support') or 'no_authoritative_support') for k in nodes_gdf['normalized_station_key']]
    nodes_gdf['station_authoritative_role_confidence'] = [float(auth_role_conf_lookup.get(k, np.nan)) if pd.notna(auth_role_conf_lookup.get(k, np.nan)) else np.nan for k in nodes_gdf['normalized_station_key']]
    nodes_gdf['station_authoritative_distance_to_bank_m'] = [float(auth_dist_lookup.get(k, np.nan)) if pd.notna(auth_dist_lookup.get(k, np.nan)) else np.nan for k in nodes_gdf['normalized_station_key']]
    nodes_gdf['station_authoritative_normalized_channel_position'] = [float(auth_norm_lookup.get(k, np.nan)) if pd.notna(auth_norm_lookup.get(k, np.nan)) else np.nan for k in nodes_gdf['normalized_station_key']]
    nodes_gdf['station_authoritative_bed_support_present'] = [bool(auth_bed_lookup.get(k, False)) for k in nodes_gdf['normalized_station_key']]
    nodes_gdf['station_authoritative_bank_margin_present'] = [bool(auth_bankmargin_lookup.get(k, False)) for k in nodes_gdf['normalized_station_key']]
    qualified_mask = np.asarray(nodes_gdf['station_true_measured_xs_qualified'], dtype=bool) & (~np.asarray(nodes_gdf['station_bank_only_authoritative'], dtype=bool))
    authoritative_like = z_source.isin(['authoritative_in_channel', 'authoritative_backbone']).to_numpy(dtype=bool) | nodes_gdf.get('graph_hard_lock', False).astype(bool).to_numpy(dtype=bool)
    propagated_mask = qualified_mask & eligible_measured_roles & (~authoritative_like)
    propagation_mode = np.full(len(nodes_gdf), 'none', dtype=object)
    propagation_mode[qualified_mask & node_role.eq('thalweg').to_numpy(dtype=bool) & (~authoritative_like)] = 'measured_thalweg_only'
    propagation_mode[qualified_mask & node_role.isin(['left_inner', 'right_inner']).to_numpy(dtype=bool) & (~authoritative_like)] = 'measured_partial_inner'
    propagation_mode[qualified_mask & authoritative_like] = 'blocked_authoritative_node'
    propagation_mode[qualified_mask & (~eligible_measured_roles)] = 'non_channel_role'
    pre_mask = propagated_mask.copy()
    post_mask = propagated_mask.copy()
    nodes_gdf['node_true_measured_xs_pre_filter'] = pre_mask.astype(bool)
    nodes_gdf['node_true_measured_xs_post_filter'] = post_mask.astype(bool)
    nodes_gdf['node_true_measured_xs_propagated'] = propagated_mask.astype(bool)
    nodes_gdf['node_true_measured_xs_propagation_mode'] = pd.Series(propagation_mode, index=nodes_gdf.index, dtype=object)
    station_pre = nodes_gdf.assign(_pre=pre_mask).groupby(['component_id', 'station_m'], dropna=False)['_pre'].transform('any').to_numpy(dtype=bool)
    station_post = nodes_gdf.assign(_post=post_mask).groupby(['component_id', 'station_m'], dropna=False)['_post'].transform('any').to_numpy(dtype=bool)
    loss_reason = np.full(len(nodes_gdf), 'no_candidate_station', dtype=object)
    loss_reason[np.asarray(nodes_gdf['station_bank_only_authoritative'], dtype=bool)] = 'bank_only_authoritative_station'
    loss_reason[np.asarray(nodes_gdf['station_true_measured_xs_candidate'], dtype=bool) & (~qualified_mask) & (~np.asarray(nodes_gdf['station_bank_only_authoritative'], dtype=bool))] = 'candidate_station_not_qualified'
    loss_reason[qualified_mask & np.asarray(nodes_gdf['node_true_measured_xs_propagated'], dtype=bool)] = 'qualified_station_propagated_to_node'
    loss_reason[qualified_mask & station_pre] = 'pre_filter_measured_node_present'
    loss_reason[qualified_mask & (~station_pre)] = 'qualified_station_without_measured_node'
    loss_reason[qualified_mask & station_pre & (~station_post)] = 'measured_node_lost_post_filter'
    nodes_gdf['measured_xs_activation_reason'] = pd.Series(loss_reason, index=nodes_gdf.index, dtype=object)
    station_node_audit = nodes_gdf.groupby(['component_id','station_m'], dropna=False).agg(
        true_measured_xs_candidate_station=('station_true_measured_xs_candidate', 'max'),
        true_measured_xs_qualified_station=('station_true_measured_xs_qualified', 'max'),
        measured_xs_nodes_pre_filter=('node_true_measured_xs_pre_filter', 'sum'),
        measured_xs_nodes_post_filter=('node_true_measured_xs_post_filter', 'sum'),
        measured_xs_activation_reason=('measured_xs_activation_reason', lambda s: pd.Series(s).astype(str).mode().iloc[0] if len(s) else 'missing'),
        propagated_true_measured_xs_nodes=('node_true_measured_xs_propagated', 'sum'),
        propagation_mode=('node_true_measured_xs_propagation_mode', lambda s: pd.Series(s).astype(str).mode().iloc[0] if len(s) else 'none'),
    ).reset_index()
    station_node_audit['nodes_lost_true_measured_xs'] = np.maximum(0, station_node_audit['measured_xs_nodes_pre_filter'] - station_node_audit['measured_xs_nodes_post_filter'])
    node_audit_path = river_dir / 'river_measured_xs_node_audit.gpkg'
    station_audit_path = river_dir / 'river_measured_xs_node_propagation.csv'
    handoff_receipt_path = river_dir / 'river_measured_xs_scaffold_handoff_receipt.json'
    if node_audit_path.exists():
        node_audit_path.unlink()
    nodes_gdf.to_file(node_audit_path, driver='GPKG')
    station_node_audit.to_csv(station_audit_path, index=False)
    bank_only_station_count = int(np.count_nonzero(station_node_audit.get('true_measured_xs_candidate_station', pd.Series(False, index=station_node_audit.index)).to_numpy(dtype=bool) & (~station_node_audit.get('true_measured_xs_qualified_station', pd.Series(False, index=station_node_audit.index)).to_numpy(dtype=bool)) & (station_node_audit['measured_xs_activation_reason'].astype(str).eq('bank_only_authoritative_station').to_numpy(dtype=bool))))
    bank_only_measured_node_count = int(np.count_nonzero(np.asarray(nodes_gdf['station_bank_only_authoritative'], dtype=bool) & post_mask))
    qualified_station_without_nodes_count = int(np.count_nonzero(station_node_audit['true_measured_xs_qualified_station'].to_numpy(dtype=bool) & (station_node_audit['measured_xs_nodes_post_filter'].to_numpy(dtype=int) <= 0)))
    handoff_receipt = {
        'stations_with_true_measured_xs_candidates': int(np.count_nonzero(station_node_audit['true_measured_xs_candidate_station'].to_numpy(dtype=bool))),
        'stations_with_true_measured_xs_qualified': int(np.count_nonzero(station_node_audit['true_measured_xs_qualified_station'].to_numpy(dtype=bool))),
        'nodes_marked_true_measured_xs_pre_filter': int(np.count_nonzero(pre_mask)),
        'nodes_marked_true_measured_xs_post_filter': int(np.count_nonzero(post_mask)),
        'nodes_marked_true_measured_xs_propagated': int(np.count_nonzero(propagated_mask)),
        'stations_with_measured_xs_nodes_pre_filter': int(np.count_nonzero(station_node_audit['measured_xs_nodes_pre_filter'].to_numpy(dtype=int) > 0)),
        'stations_with_measured_xs_nodes_post_filter': int(np.count_nonzero(station_node_audit['measured_xs_nodes_post_filter'].to_numpy(dtype=int) > 0)),
        'nodes_lost_true_measured_xs': int(np.sum(station_node_audit['nodes_lost_true_measured_xs'].to_numpy(dtype=int))),
        'loss_reason_counts': {str(k): int(v) for k, v in station_node_audit.loc[station_node_audit['true_measured_xs_candidate_station'].astype(bool), 'measured_xs_activation_reason'].astype(str).value_counts().to_dict().items()},
        'qualified_loss_reason_counts': {str(k): int(v) for k, v in station_node_audit.loc[station_node_audit['true_measured_xs_qualified_station'].astype(bool), 'measured_xs_activation_reason'].astype(str).value_counts().to_dict().items()},
        'bank_only_authoritative_station_count': bank_only_station_count,
        'bank_only_authoritative_measured_node_count': bank_only_measured_node_count,
        'qualified_station_without_measured_nodes_count': qualified_station_without_nodes_count,
        'measured_xs_activation_contract_ok': bool((int(np.count_nonzero(station_node_audit['true_measured_xs_qualified_station'].to_numpy(dtype=bool))) > 0) or (int(np.count_nonzero(post_mask)) == 0)),
        'measured_xs_activation_from_bank_only_forbidden': bool(bank_only_measured_node_count == 0),
        'artifacts': {
            'measured_xs_node_audit': str(node_audit_path),
            'measured_xs_node_propagation': str(station_audit_path),
        },
    }
    if handoff_receipt['stations_with_true_measured_xs_qualified'] == 0 and handoff_receipt['nodes_marked_true_measured_xs_post_filter'] > 0:
        handoff_receipt['contract_violation'] = 'qualified_stations_zero_but_measured_nodes_present'
        (logger or log).warning('[RIVER][FRAME] measured XS activation failed contract: qualified_stations=0 but measured_nodes=%d', int(handoff_receipt['nodes_marked_true_measured_xs_post_filter']))
    if bank_only_measured_node_count > 0:
        handoff_receipt.setdefault('contract_violations', []).append('bank_only_authoritative_station_produced_measured_nodes')
        (logger or log).warning('[RIVER][FRAME] measured XS activation violated bank-only contract: bank_only_measured_nodes=%d', bank_only_measured_node_count)
    if qualified_station_without_nodes_count > 0:
        handoff_receipt.setdefault('warnings', []).append('qualified_stations_without_measured_nodes')
        (logger or log).warning('[RIVER][FRAME] measured XS propagation incomplete: qualified_stations_without_measured_nodes=%d', qualified_station_without_nodes_count)
    handoff_receipt_path.write_text(json.dumps(handoff_receipt, indent=2), encoding='utf-8')
    out_path = river_dir / 'river_channel_scaffold_nodes.gpkg'
    if out_path.exists():
        out_path.unlink()
    nodes_gdf.to_file(out_path, driver='GPKG')

    diag_cols = [
        'component_id', 'station_m', 'graph_backbone_z_m', 'graph_hard_lock', 'graph_prior_weight_sum',
        'graph_edge_weight_sum', 'graph_curvature_weight_sum', 'graph_centering_weight_sum',
        'graph_regularization_weight_sum', 'graph_junction_weight_sum', 'graph_junction_constrained',
        'graph_residual_to_candidate_z_m', 'graph_solver_support_class', 'graph_candidate_source',
        'graph_solution_mode', 'graph_unsupported_span_m', 'graph_unsupported_regime', 'graph_slope_guard_weight_sum',
        'graph_adverse_step_weight_sum', 'graph_physical_guard_weight_sum', 'graph_local_slope',
        'graph_local_curvature', 'graph_slope_guard_active', 'graph_adverse_step_guard_active',
        'authoritative_station_support_strength', 'mainstem_rank', 'network_order', 'distance_to_mouth_m'
    ]
    diag_frame = frame[[c for c in ['component_id', 'station_m', 'geometry'] + diag_cols[2:] if c in frame.columns]].copy()
    if not diag_frame.empty:
        support_strength = diag_frame.get('authoritative_station_support_strength', pd.Series(np.nan, index=diag_frame.index))
        diag_frame['graph_support_strength'] = pd.to_numeric(support_strength, errors='coerce').fillna(0.0).astype(np.float32)
        anchor_spacing = diag_frame.get('graph_anchor_spacing_m', pd.Series(np.nan, index=diag_frame.index))
        diag_frame['graph_anchor_spacing_m'] = pd.to_numeric(anchor_spacing, errors='coerce').fillna(0.0).astype(np.float32)
        mainstem_series = pd.to_numeric(diag_frame.get('mainstem_rank', pd.Series(np.nan, index=diag_frame.index)), errors='coerce')
        order_series = pd.to_numeric(diag_frame.get('network_order', pd.Series(np.nan, index=diag_frame.index)), errors='coerce')
        mouth_series = pd.to_numeric(diag_frame.get('distance_to_mouth_m', pd.Series(np.nan, index=diag_frame.index)), errors='coerce')
        topo_conf = diag_frame.get('graph_topology_confidence', pd.Series(np.nan, index=diag_frame.index))
        topo_conf = pd.to_numeric(topo_conf, errors='coerce')
        topo_conf = topo_conf.where(topo_conf.notna(), np.where(mainstem_series.notna() | order_series.notna() | mouth_series.notna(), 1.0, 0.5))
        diag_frame['graph_topology_confidence'] = topo_conf.astype(np.float32)
        unsupported_regime = diag_frame.get('graph_unsupported_regime', pd.Series('missing', index=diag_frame.index))
        diag_frame['graph_unsupported_regime'] = pd.Series(unsupported_regime, index=diag_frame.index).fillna('missing').astype(str)
        conf = diag_frame.apply(graph_solution_confidence, axis=1).astype(float)
        diag_frame['graph_solution_confidence'] = conf.astype(np.float32)
        diag_frame['graph_uncertainty_class'] = [uncertainty_class_from_confidence(float(v)) for v in conf]
    if 'geometry' in diag_frame.columns:
        diag_frame = gpd.GeoDataFrame(diag_frame, geometry='geometry', crs=frame.crs)
    junction_diag_records = list(getattr(graph_diagnostics, 'attrs', {}).get('junction_diagnostics', [])) if graph_diagnostics is not None else []
    junction_diag_path = river_dir / 'river_junction_diagnostics.gpkg'
    if junction_diag_path.exists():
        junction_diag_path.unlink()
    if junction_diag_records:
        jdf = pd.DataFrame.from_records(junction_diag_records)
        jdf['component_ids'] = jdf.get('component_ids', pd.Series([[]]*len(jdf))).apply(lambda vals: ','.join(map(str, vals)) if isinstance(vals, (list, tuple, set)) else str(vals))
        from shapely.geometry import Point as _Point
        jdf['geometry'] = [
            _Point(float(x), float(y)) if np.isfinite(float(x)) and np.isfinite(float(y)) else None
            for x, y in zip(pd.to_numeric(jdf.get('junction_x', np.nan), errors='coerce').to_numpy(dtype=float), pd.to_numeric(jdf.get('junction_y', np.nan), errors='coerce').to_numpy(dtype=float))
        ]
        jgdf = gpd.GeoDataFrame(jdf, geometry='geometry', crs=frame.crs)
        if not jgdf.empty:
            jgdf.to_file(junction_diag_path, driver='GPKG')
    physical_summary = _graph_summarize_physical_plausibility(diag_frame)
    graph_diag_path = river_dir / 'river_graph_backbone_diagnostics.gpkg'
    if graph_diag_path.exists():
        graph_diag_path.unlink()
    if not getattr(diag_frame, 'empty', True):
        diag_frame.to_file(graph_diag_path, driver='GPKG')
    physical_contract_path = river_dir / 'river_graph_physical_plausibility_contract.json'
    physical_contract = {
        'schema_version': 1,
        'artifact_family': 'river_graph_physical_plausibility',
        'notes': {
            'objective': 'Run-level physical plausibility diagnostics for the graph backbone solve.',
            'interpretation': 'These are deterministic solver diagnostics, not formal probabilistic uncertainty. They summarize where physical guards, slope limits, curvature control, and unsupported spans influenced the backbone solution.',
        },
        'metrics': {
            **physical_summary,
            'graph_junction_constraint_count': int(junction_metrics.get('graph_junction_constraint_count', 0)),
            'graph_topology_guided_constraint_count': int(junction_metrics.get('graph_topology_guided_constraint_count', 0)),
            'graph_distance_weighted_constraint_count': int(junction_metrics.get('graph_distance_weighted_constraint_count', 0)),
        },
        'artifacts': {
            'graph_backbone_diagnostics': str(graph_diag_path),
            'junction_diagnostics': str(junction_diag_path),
        },
    }
    physical_contract_path.write_text(json.dumps(physical_contract, indent=2), encoding='utf-8')

    contract = {
        'schema_version': 1,
        'artifact_family': 'river_channel_scaffold',
        'notes': {
            'objective': 'Regularized channel-fitted scaffold nodes derived from chainaged centerline stations and resampled XS profiles.',
            'literature_alignment': 'Supports channel-fitted interpolation in (s,n) coordinates and longitudinal/transverse XS regularization.',
        },
        'metrics': {
            'node_count': int(len(nodes_gdf)),
            'station_count': int(nodes_gdf[['component_id','station_m']].drop_duplicates().shape[0]),
            'authoritative_node_count': int(np.count_nonzero(nodes_gdf['z_source'].astype(str).eq('authoritative_in_channel'))),
            'xs_profile_resampled_node_count': int(np.count_nonzero(nodes_gdf['z_source'].astype(str).eq('xs_profile_resampled'))),
            'graph_backbone_node_count': int(np.count_nonzero(nodes_gdf['z_source'].astype(str).eq('graph_backbone'))),
            'graph_backbone_missing_node_count': int(np.count_nonzero(nodes_gdf.get('graph_backbone_missing', False).astype(bool))),
            'resolved_channel_bed_node_count': 0,
            'bank_stage_prior_node_count': int(np.count_nonzero(nodes_gdf['z_source'].astype(str).eq('bank_stage_prior'))),
            'xs_only_station_count': int(np.count_nonzero(nodes_gdf['station_support_mode'].astype(str).eq('xs_only'))),
            'absolute_bed_fallback_station_count': 0,
            'absolute_bed_fallback_removed': True,
            'absolute_bed_fallback_node_count': 0,
            'absolute_bed_fallback_suppressed_node_count': 0,
            'resolved_backbone_station_count': int(np.count_nonzero(nodes_gdf['station_support_mode'].astype(str).eq('resolved_backbone'))),
            'component_count': int(nodes_gdf['component_id'].nunique(dropna=True)),
            'node_role_counts': {str(k): int(v) for k, v in nodes_gdf['node_role'].astype(str).value_counts().to_dict().items()},
            'junction_group_count': int(junction_metrics.get('junction_group_count', 0)),
            'junction_adjusted_component_count': int(junction_metrics.get('junction_adjusted_component_count', 0)),
            'junction_adjusted_station_count': int(junction_metrics.get('junction_adjusted_station_count', 0)),
            'dominant_junction_group_count': int(junction_metrics.get('dominant_junction_group_count', 0)),
            'dominant_preserved_component_count': int(junction_metrics.get('dominant_preserved_component_count', 0)),
            'network_junction_count': int(junction_metrics.get('network_junction_count', 0)),
            'topology_guided_junction_count': int(junction_metrics.get('topology_guided_junction_count', 0)),
            'topology_guided_component_count': int(junction_metrics.get('topology_guided_component_count', 0)),
            'geometric_fallback_junction_count': int(junction_metrics.get('geometric_fallback_junction_count', 0)),
            'graph_junction_constraint_count': int(junction_metrics.get('graph_junction_constraint_count', 0)),
            'graph_solution_mode_counts': {str(k): int(v) for k, v in frame.get('graph_solution_mode', pd.Series([], dtype=object)).fillna('missing').astype(str).value_counts().to_dict().items()},
            'graph_unsupported_regime_counts': {str(k): int(v) for k, v in frame.get('graph_unsupported_regime', pd.Series([], dtype=object)).fillna('missing').astype(str).value_counts().to_dict().items()},
            'graph_diagnostics_available': bool(graph_diagnostics_available),
            'graph_diagnostics_duplicate_rows_collapsed': int(graph_diag_duplicate_count),
            'graph_hard_lock_station_count': int(np.count_nonzero(frame.get('graph_hard_lock', pd.Series(False, index=frame.index)).astype(bool).to_numpy(dtype=bool))),
            'graph_junction_constrained_station_count': int(np.count_nonzero(frame.get('graph_junction_constrained', pd.Series(False, index=frame.index)).astype(bool).to_numpy(dtype=bool))),
            'graph_junction_role_counts': {str(k): int(v) for k, v in frame.get('graph_junction_role', pd.Series([], dtype=object)).fillna('not_in_junction').astype(str).value_counts().to_dict().items()},
            'junction_diagnostic_count': int(len(junction_diag_records)),
            'component_hard_lock_count': int(junction_metrics.get('component_hard_lock_count', 0)),
            'component_stage_controlled_count': int(junction_metrics.get('component_stage_controlled_count', 0)),
            'component_anchored_count': int(junction_metrics.get('component_anchored_count', 0)),
            'component_unsupported_count': int(junction_metrics.get('component_unsupported_count', 0)),
            'legacy_mixed_bed_used': False,
            'legacy_mixed_bed_field_ignored': True,
            'xs_influence_disabled': bool(disable_xs_influence),
            'anchor_class_counts': {str(k): int(v) for k, v in nodes_gdf.get('target_anchor_class', pd.Series('missing', index=nodes_gdf.index)).astype(str).value_counts().to_dict().items()},
            'anchor_locks_core_count': int(np.count_nonzero(nodes_gdf.get('target_anchor_locks_core', pd.Series(False, index=nodes_gdf.index)).fillna(False).astype(bool).to_numpy(dtype=bool))),
            'anchor_blocks_rebuild_count': int(np.count_nonzero(nodes_gdf.get('target_anchor_blocks_rebuild', pd.Series(False, index=nodes_gdf.index)).fillna(False).astype(bool).to_numpy(dtype=bool))),
            **physical_summary,
        },
        'artifacts': {
            'channel_scaffold_nodes': str(out_path),
            'graph_backbone_diagnostics': str(graph_diag_path),
            'junction_diagnostics': str(junction_diag_path),
            'graph_physical_plausibility_contract': str(physical_contract_path),
            'measured_xs_node_audit': str(node_audit_path),
            'measured_xs_node_propagation': str(station_audit_path),
            'measured_xs_scaffold_handoff_receipt': str(handoff_receipt_path),
        },
    }
    contract_path = river_dir / 'river_channel_scaffold_contract.json'
    contract_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    xs_like_nodes = int(np.count_nonzero(
        nodes_gdf['z_source'].astype(str).eq('xs_profile_resampled')
        | nodes_gdf['graph_candidate_source'].astype(str).eq('xs_profile_resampled')
        | nodes_gdf['station_support_mode'].astype(str).isin(['xs_only', 'xs_supported', 'xs_residual_only'])
        | nodes_gdf['graph_solver_support_class'].astype(str).eq('xs_residual_only')
    ))
    measured_xs_nodes = int(np.count_nonzero(nodes_gdf.get('node_true_measured_xs_post_filter', pd.Series(False, index=nodes_gdf.index)).fillna(False).astype(bool).to_numpy(dtype=bool)))
    (logger or log).info('[RIVER][FRAME] Channel scaffold built: nodes=%d stations=%d auth_nodes=%d non_auth_nodes=%d xs_like_nodes=%d measured_xs_nodes=%d xs_influence_disabled=%s', int(len(nodes_gdf)), int(nodes_gdf[["component_id","station_m"]].drop_duplicates().shape[0]), int(total_auth_nodes), int(total_interp_nodes), xs_like_nodes, measured_xs_nodes, bool(disable_xs_influence))
    return {'channel_scaffold_nodes': str(out_path), 'channel_scaffold_contract': str(contract_path), 'graph_backbone_diagnostics': str(graph_diag_path), 'graph_physical_plausibility_contract': str(physical_contract_path), 'junction_diagnostics': str(junction_diag_path), 'measured_xs_node_audit': str(node_audit_path), 'measured_xs_node_propagation': str(station_audit_path), 'measured_xs_scaffold_handoff_receipt': str(handoff_receipt_path)}
