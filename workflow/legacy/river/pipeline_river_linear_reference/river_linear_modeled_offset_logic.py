from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

_PROFILE_GROUP_FIELDS = ('component_id', 'levelpath_id')
_IDENTITY_FIELDS = ('component_id', 'levelpath_id', 'reach_id', 'source_reach_key')
_MIN_OFFSET_M = 0.10
_MAX_OFFSET_M = 50.0
_MIN_EXPORT_MODELED_POINTS = 20
_MAX_EXPORT_FLOOR_FRACTION = 0.65
_MIN_TAIL_BLEND_SCALE_M = 500.0
_MAX_TAIL_BLEND_SCALE_M = 3000.0
_NONFLOOR_EPS = 1.0e-6


@dataclass(frozen=True)
class ModeledOffsetBuildResult:
    modeled_points: gpd.GeoDataFrame
    global_observed_median_offset_m: float
    global_prior_offset_m: float
    finite_modeled_offset_count: int
    component_count: int
    component_summaries: list[dict[str, object]] | None = None


@dataclass(frozen=True)
class ExportFloorValidationResult:
    export_overlap_count: int
    finite_export_count: int
    floor_fraction: float | None
    exceeds_limit: bool


@dataclass(frozen=True)
class ExportFloorPolicyResult:
    modeled_points: gpd.GeoDataFrame
    validation_before: ExportFloorValidationResult
    validation_after: ExportFloorValidationResult
    policy_applied: bool
    policy_status: str


def _group_key(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in _PROFILE_GROUP_FIELDS if c in frame.columns]
    if not cols:
        return pd.Series(['all'] * len(frame), index=frame.index, dtype=object)
    return frame[cols].astype(str).agg('|'.join, axis=1)



def _unique_anchor_series(station: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sta = np.asarray(station, dtype=float)
    vals = np.asarray(values, dtype=float)
    usable = np.isfinite(sta) & np.isfinite(vals)
    if np.count_nonzero(usable) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    sta = sta[usable]
    vals = vals[usable]
    order = np.argsort(sta, kind='mergesort')
    sta = sta[order]
    vals = vals[order]
    unique_sta: list[float] = []
    unique_vals: list[float] = []
    i = 0
    while i < len(sta):
        j = i + 1
        while j < len(sta) and abs(sta[j] - sta[i]) <= 1.0e-9:
            j += 1
        unique_sta.append(float(sta[i]))
        unique_vals.append(float(np.nanmedian(vals[i:j])))
        i = j
    return np.asarray(unique_sta, dtype=float), np.asarray(unique_vals, dtype=float)



def _clamp_offset(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = np.where(np.isfinite(vals), np.clip(vals, _MIN_OFFSET_M, _MAX_OFFSET_M), np.nan)
    return vals



def _smooth_anchor_values(values: np.ndarray, *, window: int = 5) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return vals
    smoothed = pd.Series(vals).rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)
    return _clamp_offset(smoothed)



def _component_prior(anchor_values: np.ndarray, *, global_median: float) -> float:
    vals = _clamp_offset(anchor_values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(global_median)
    p75 = float(np.nanpercentile(vals, 75.0))
    med = float(np.nanmedian(vals))
    return float(np.clip(max(global_median, p75, med), _MIN_OFFSET_M, _MAX_OFFSET_M))



def _global_prior_offset(global_observed_offsets: np.ndarray) -> float:
    vals = _clamp_offset(global_observed_offsets)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise RuntimeError('river_linear_modeled_offset_no_observed_support_anywhere')
    nonfloor = vals[vals > (_MIN_OFFSET_M + _NONFLOOR_EPS)]
    prior_source = nonfloor if nonfloor.size else vals
    med = float(np.nanmedian(prior_source))
    p75 = float(np.nanpercentile(prior_source, 75.0))
    return float(np.clip(max(med, p75), _MIN_OFFSET_M, _MAX_OFFSET_M))



def _tail_blend(station: np.ndarray, *, edge_station: float, edge_value: float, prior_value: float, scale_m: float) -> np.ndarray:
    sta = np.asarray(station, dtype=float)
    if scale_m <= 0.0:
        return np.full(sta.shape, fill_value=float(prior_value), dtype=float)
    distance = np.maximum(0.0, np.abs(sta - float(edge_station)))
    weight = np.clip(distance / float(scale_m), 0.0, 1.0)
    blended = (1.0 - weight) * float(edge_value) + weight * float(prior_value)
    return _clamp_offset(blended)



def _prepare_wse_observed_merge(wse: gpd.GeoDataFrame, observed: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    prepared_wse = wse.copy()
    if 'point_id' not in prepared_wse.columns:
        raise RuntimeError('river_linear_modeled_offset_missing_point_id')
    prepared_wse['point_id'] = prepared_wse['point_id'].astype(str)
    if 'station_m' not in prepared_wse.columns:
        raise RuntimeError('river_linear_modeled_offset_missing_station_m')
    prepared_wse['station_m'] = pd.to_numeric(prepared_wse['station_m'], errors='coerce')
    if 'station_downstream_m' in prepared_wse.columns:
        prepared_wse['station_downstream_m'] = pd.to_numeric(prepared_wse['station_downstream_m'], errors='coerce')
    else:
        prepared_wse['station_downstream_m'] = prepared_wse['station_m']
    prepared_wse['offset_model_station_m'] = prepared_wse['station_downstream_m']

    obs_cols = [
        c for c in (
            'point_id', 'observed_offset_m', 'offset_observed_m', 'offset_used_m',
            'offset_qc_pass', 'offset_support_class', 'offset_confidence',
            'offset_reject_reason', 'offset_anchor_weight', 'offset_qc_class',
            'offset_anchor_role', 'offset_low_depth_warning'
        ) if c in observed.columns
    ]
    prepared_observed = observed.copy()
    if 'point_id' not in obs_cols:
        prepared_observed = prepared_observed.iloc[0:0].copy()
        prepared_observed['point_id'] = pd.Series(dtype='object')
        prepared_observed['observed_offset_m'] = pd.Series(dtype='float64')
        prepared_observed['offset_used_m'] = pd.Series(dtype='float64')
        obs_cols = ['point_id', 'observed_offset_m', 'offset_used_m']
    prepared_observed = prepared_observed[obs_cols].copy()
    prepared_observed['point_id'] = prepared_observed['point_id'].astype(str)
    if 'offset_used_m' in prepared_observed.columns:
        prepared_observed['observed_offset_m'] = pd.to_numeric(prepared_observed['offset_used_m'], errors='coerce')
    else:
        prepared_observed['observed_offset_m'] = pd.to_numeric(prepared_observed['observed_offset_m'], errors='coerce')
    if 'offset_anchor_weight' in prepared_observed.columns:
        prepared_observed['offset_anchor_weight'] = pd.to_numeric(prepared_observed['offset_anchor_weight'], errors='coerce').clip(lower=0.0, upper=1.0)
    else:
        prepared_observed['offset_anchor_weight'] = np.where(np.isfinite(prepared_observed['observed_offset_m'].to_numpy(dtype=float)), 1.0, 0.0)
    return prepared_wse.merge(prepared_observed, on='point_id', how='left')


def _anchor_distances(station: np.ndarray, anchor_station: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sta = np.asarray(station, dtype=float)
    anchors = np.asarray(anchor_station, dtype=float)
    nearest = np.full(sta.shape, np.nan, dtype=float)
    upstream = np.full(sta.shape, np.nan, dtype=float)
    downstream = np.full(sta.shape, np.nan, dtype=float)
    anchors = np.sort(anchors[np.isfinite(anchors)])
    if anchors.size == 0:
        return nearest, upstream, downstream
    for i, s in enumerate(sta):
        if not np.isfinite(s):
            continue
        diffs = anchors - s
        nearest[i] = float(np.nanmin(np.abs(diffs)))
        up = anchors[anchors <= s]
        dn = anchors[anchors >= s]
        if up.size:
            upstream[i] = float(s - up.max())
        if dn.size:
            downstream[i] = float(dn.min() - s)
    return nearest, upstream, downstream


def _component_span(station: np.ndarray) -> float:
    sta = np.asarray(station, dtype=float)
    finite = sta[np.isfinite(sta)]
    if finite.size == 0:
        return 0.0
    return float(np.nanmax(finite) - np.nanmin(finite))


def _component_key_value(grp: pd.DataFrame) -> str:
    if len(grp) == 0:
        return 'unknown'
    return str(_group_key(grp).iloc[0])


def build_modeled_offset_points(wse: gpd.GeoDataFrame, observed: gpd.GeoDataFrame) -> ModeledOffsetBuildResult:
    merged = _prepare_wse_observed_merge(wse, observed)
    group_keys = _group_key(merged)
    global_obs = pd.to_numeric(merged['observed_offset_m'], errors='coerce').to_numpy(dtype=float)
    finite_global = global_obs[np.isfinite(global_obs)]
    if finite_global.size == 0:
        raise RuntimeError('river_linear_modeled_offset_no_observed_support_anywhere')
    global_median = float(np.clip(np.nanmedian(finite_global), _MIN_OFFSET_M, _MAX_OFFSET_M))
    global_prior = _global_prior_offset(global_obs)
    merged['offset_modeled_m'] = np.nan
    merged['offset_source'] = ''
    merged['offset_support_class'] = ''
    merged['offset_confidence'] = ''
    merged['distance_to_nearest_observed_offset_m'] = np.nan
    merged['upstream_observed_offset_distance_m'] = np.nan
    merged['downstream_observed_offset_distance_m'] = np.nan
    merged['offset_model_station_m'] = pd.to_numeric(merged.get('offset_model_station_m', merged['station_m']), errors='coerce')
    component_summaries: list[dict[str, object]] = []
    for _, grp_idx in group_keys.groupby(group_keys).groups.items():
        grp = merged.loc[list(grp_idx)].copy()
        grp = grp.sort_values('offset_model_station_m', kind='mergesort')
        sta = pd.to_numeric(grp['offset_model_station_m'], errors='coerce').to_numpy(dtype=float)
        obs = pd.to_numeric(grp['observed_offset_m'], errors='coerce').to_numpy(dtype=float)
        anchor_weight = pd.to_numeric(grp.get('offset_anchor_weight', pd.Series(1.0, index=grp.index)), errors='coerce').fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
        modeled = np.full((len(grp),), np.nan, dtype=float)
        source = np.full((len(grp),), 'unsupported', dtype=object)
        support = np.full((len(grp),), 'unsupported_no_confident_offset', dtype=object)
        confidence = np.full((len(grp),), 'low', dtype=object)
        # Weak shallow anchors remain visible, but are pulled toward the solve-domain
        # prior before they influence interpolation/tails.  Strong anchors retain their
        # measured observed offset at the exact point.
        effective_anchor_obs = np.where(
            np.isfinite(obs) & (anchor_weight > 0.0),
            (anchor_weight * obs) + ((1.0 - anchor_weight) * global_prior),
            np.nan,
        )
        anchor_sta, anchor_val = _unique_anchor_series(sta, _clamp_offset(effective_anchor_obs))
        finite_station = np.isfinite(sta)
        observed_mask = np.isfinite(obs) & (anchor_weight >= 0.75)
        weak_observed_mask = np.isfinite(obs) & (anchor_weight > 0.0) & (anchor_weight < 0.75)
        nearest, upstream, downstream = _anchor_distances(sta, anchor_sta)
        component_station_span = _component_span(sta)
        if anchor_sta.size >= 2:
            smoothed_anchor_val = _smooth_anchor_values(anchor_val)
            prior_value = _component_prior(anchor_val, global_median=global_prior)
            in_range = finite_station & (sta >= float(anchor_sta.min())) & (sta <= float(anchor_sta.max()))
            modeled[in_range] = np.interp(sta[in_range], anchor_sta, smoothed_anchor_val)
            source[in_range] = 'longitudinal_interpolation_between_observed_offsets'
            support[in_range] = 'interpolated_between_observed_offsets'
            confidence[in_range] = 'medium'
            component_anchor_span = float(anchor_sta.max() - anchor_sta.min())
            tail_scale_m = max(_MIN_TAIL_BLEND_SCALE_M, min(_MAX_TAIL_BLEND_SCALE_M, component_anchor_span * 0.33))
            near_threshold_m = max(_MIN_TAIL_BLEND_SCALE_M, min(_MAX_TAIL_BLEND_SCALE_M, tail_scale_m))
            left_tail = finite_station & (sta < float(anchor_sta.min()))
            if np.any(left_tail):
                modeled[left_tail] = _tail_blend(
                    sta[left_tail],
                    edge_station=float(anchor_sta.min()),
                    edge_value=float(smoothed_anchor_val[0]),
                    prior_value=prior_value,
                    scale_m=tail_scale_m,
                )
                source[left_tail] = 'tail_blend_from_nearest_observed_offset_to_component_prior'
                near = left_tail & np.isfinite(nearest) & (nearest <= near_threshold_m)
                far = left_tail & ~near
                support[near] = 'near_observed_offset_supported'
                support[far] = 'low_support_scaffolded'
                confidence[near] = 'low_medium'
                confidence[far] = 'low'
            right_tail = finite_station & (sta > float(anchor_sta.max()))
            if np.any(right_tail):
                modeled[right_tail] = _tail_blend(
                    sta[right_tail],
                    edge_station=float(anchor_sta.max()),
                    edge_value=float(smoothed_anchor_val[-1]),
                    prior_value=prior_value,
                    scale_m=tail_scale_m,
                )
                source[right_tail] = 'tail_blend_from_nearest_observed_offset_to_component_prior'
                near = right_tail & np.isfinite(nearest) & (nearest <= near_threshold_m)
                far = right_tail & ~near
                support[near] = 'near_observed_offset_supported'
                support[far] = 'low_support_scaffolded'
                confidence[near] = 'low_medium'
                confidence[far] = 'low'
        elif anchor_sta.size == 1:
            prior_value = _component_prior(anchor_val, global_median=global_prior)
            tail_scale_m = max(_MIN_TAIL_BLEND_SCALE_M, min(_MAX_TAIL_BLEND_SCALE_M, component_station_span * 0.33))
            modeled[finite_station] = _tail_blend(
                sta[finite_station],
                edge_station=float(anchor_sta[0]),
                edge_value=float(anchor_val[0]),
                prior_value=prior_value,
                scale_m=tail_scale_m,
            )
            source[finite_station] = 'single_observed_offset_blend_to_component_prior'
            near_threshold_m = max(_MIN_TAIL_BLEND_SCALE_M, min(_MAX_TAIL_BLEND_SCALE_M, tail_scale_m))
            near = finite_station & np.isfinite(nearest) & (nearest <= near_threshold_m)
            far = finite_station & ~near
            support[near] = 'near_observed_offset_supported'
            support[far] = 'low_support_scaffolded'
            confidence[near] = 'low_medium'
            confidence[far] = 'low'
        else:
            modeled[finite_station] = global_prior
            source[finite_station] = 'solve_domain_global_prior_scaffold'
            support[finite_station] = 'low_support_scaffolded'
            confidence[finite_station] = 'low'
        if np.any(observed_mask):
            modeled[observed_mask] = _clamp_offset(obs[observed_mask])
            source[observed_mask] = 'observed_offset_anchor'
            support[observed_mask] = 'observed_offset_supported'
            confidence[observed_mask] = 'high'
        if np.any(weak_observed_mask):
            source[weak_observed_mask] = 'weak_observed_offset_blended_anchor'
            support[weak_observed_mask] = 'shallow_observed_low_confidence'
            confidence[weak_observed_mask] = 'low_medium'
        modeled = _clamp_offset(modeled)
        merged.loc[grp.index, 'offset_modeled_m'] = modeled
        merged.loc[grp.index, 'offset_source'] = source
        merged.loc[grp.index, 'offset_support_class'] = support
        merged.loc[grp.index, 'offset_confidence'] = confidence
        merged.loc[grp.index, 'distance_to_nearest_observed_offset_m'] = nearest
        merged.loc[grp.index, 'upstream_observed_offset_distance_m'] = upstream
        merged.loc[grp.index, 'downstream_observed_offset_distance_m'] = downstream
        finite_modeled = modeled[np.isfinite(modeled)]
        component_summaries.append({
            'group_key': _component_key_value(grp),
            'record_count': int(len(grp)),
            'finite_station_count': int(np.count_nonzero(finite_station)),
            'observed_anchor_count': int(anchor_sta.size),
            'strong_observed_anchor_count': int(np.count_nonzero(observed_mask)),
            'weak_observed_anchor_count': int(np.count_nonzero(weak_observed_mask)),
            'station_span_m': float(component_station_span),
            'anchor_span_m': float(anchor_sta.max() - anchor_sta.min()) if anchor_sta.size >= 2 else 0.0,
            'modeled_offset_min_m': float(np.nanmin(finite_modeled)) if finite_modeled.size else None,
            'modeled_offset_median_m': float(np.nanmedian(finite_modeled)) if finite_modeled.size else None,
            'modeled_offset_max_m': float(np.nanmax(finite_modeled)) if finite_modeled.size else None,
            'support_class_counts': {str(k): int(v) for k, v in pd.Series(support, dtype=object).astype(str).value_counts().to_dict().items()},
            'offset_source_counts': {str(k): int(v) for k, v in pd.Series(source, dtype=object).astype(str).value_counts().to_dict().items()},
        })
    if 'wse_proxy_z_m' in merged.columns:
        wse_proxy = pd.to_numeric(merged['wse_proxy_z_m'], errors='coerce').to_numpy(dtype=float)
        modeled_offset = pd.to_numeric(merged['offset_modeled_m'], errors='coerce').to_numpy(dtype=float)
        merged['offset_raw_bed_z_m'] = wse_proxy - modeled_offset
    keep = [c for c in (
        'point_id', 'station_m', 'station_downstream_m', 'offset_model_station_m', *_IDENTITY_FIELDS,
        'wse_proxy_z_m', 'observed_offset_m', 'offset_anchor_weight', 'offset_qc_class',
        'offset_anchor_role', 'offset_low_depth_warning',
        'offset_modeled_m', 'offset_source', 'offset_support_class', 'offset_confidence',
        'distance_to_nearest_observed_offset_m', 'upstream_observed_offset_distance_m',
        'downstream_observed_offset_distance_m',
        'offset_raw_bed_z_m', 'geometry'
    ) if c in merged.columns]
    out = gpd.GeoDataFrame(merged[keep].copy(), geometry='geometry', crs=wse.crs)
    sort_cols = [c for c in (*[c for c in _PROFILE_GROUP_FIELDS if c in out.columns], 'offset_model_station_m', 'point_id') if c in out.columns]
    out = out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    finite = int(np.count_nonzero(np.isfinite(pd.to_numeric(out['offset_modeled_m'], errors='coerce').to_numpy(dtype=float))))
    component_count = int(group_keys.nunique())
    return ModeledOffsetBuildResult(
        modeled_points=out,
        global_observed_median_offset_m=global_median,
        global_prior_offset_m=global_prior,
        finite_modeled_offset_count=finite,
        component_count=component_count,
        component_summaries=component_summaries,
    )


def export_overlap_indices(gdf: gpd.GeoDataFrame, *, template_path) -> np.ndarray:
    with rasterio.open(template_path) as ds:
        template_bounds = box(*ds.bounds)
    geom = gdf.geometry
    return geom.intersects(template_bounds).to_numpy(dtype=bool)



def evaluate_export_floor_collapse(modeled_points: gpd.GeoDataFrame, *, export_overlap: np.ndarray) -> ExportFloorValidationResult:
    overlap_count = int(np.count_nonzero(export_overlap))
    if overlap_count < _MIN_EXPORT_MODELED_POINTS:
        return ExportFloorValidationResult(
            export_overlap_count=overlap_count,
            finite_export_count=0,
            floor_fraction=None,
            exceeds_limit=False,
        )
    export_modeled = pd.to_numeric(modeled_points.loc[export_overlap, 'offset_modeled_m'], errors='coerce').to_numpy(dtype=float)
    finite_export = export_modeled[np.isfinite(export_modeled)]
    finite_count = int(finite_export.size)
    if finite_count < _MIN_EXPORT_MODELED_POINTS:
        return ExportFloorValidationResult(
            export_overlap_count=overlap_count,
            finite_export_count=finite_count,
            floor_fraction=None,
            exceeds_limit=False,
        )
    floor_fraction = float(np.mean(np.isclose(finite_export, _MIN_OFFSET_M)))
    return ExportFloorValidationResult(
        export_overlap_count=overlap_count,
        finite_export_count=finite_count,
        floor_fraction=floor_fraction,
        exceeds_limit=bool(floor_fraction > _MAX_EXPORT_FLOOR_FRACTION),
    )



def _repair_export_floor_collapse(out: gpd.GeoDataFrame, *, export_overlap: np.ndarray, global_prior: float) -> gpd.GeoDataFrame:
    repaired = out.copy()
    export_df = repaired.loc[export_overlap].copy()
    group_keys = _group_key(export_df)
    for _, grp_idx in group_keys.groupby(group_keys).groups.items():
        grp = export_df.loc[list(grp_idx)].copy()
        modeled = pd.to_numeric(grp['offset_modeled_m'], errors='coerce').to_numpy(dtype=float)
        finite = modeled[np.isfinite(modeled)]
        if finite.size < _MIN_EXPORT_MODELED_POINTS:
            continue
        floor_fraction = float(np.mean(np.isclose(finite, _MIN_OFFSET_M)))
        if floor_fraction <= _MAX_EXPORT_FLOOR_FRACTION:
            continue
        has_local_observed_anchor = bool(np.any(grp['offset_source'].astype(str).to_numpy(dtype=object) == 'observed_offset_anchor'))
        if has_local_observed_anchor:
            continue
        full_group = repaired.loc[grp.index].copy()
        full_modeled = pd.to_numeric(full_group['offset_modeled_m'], errors='coerce').to_numpy(dtype=float)
        full_finite = full_modeled[np.isfinite(full_modeled)]
        if full_finite.size == 0:
            component_prior = float(global_prior)
        else:
            nonfloor = full_finite[full_finite > (_MIN_OFFSET_M + _NONFLOOR_EPS)]
            prior_source = nonfloor if nonfloor.size else full_finite
            component_prior = float(np.clip(max(global_prior, float(np.nanmedian(prior_source)), float(np.nanpercentile(prior_source, 75.0))), _MIN_OFFSET_M, _MAX_OFFSET_M))
        repair_mask = grp['offset_source'].astype(str).to_numpy(dtype=object) != 'observed_offset_anchor'
        if not np.any(repair_mask):
            continue
        grp_indices = grp.index.to_numpy()
        repaired.loc[grp_indices[repair_mask], 'offset_modeled_m'] = component_prior
        repaired.loc[grp_indices[repair_mask], 'offset_source'] = 'solve_domain_global_prior_scaffold_repair'
        repaired.loc[grp_indices[repair_mask], 'offset_support_class'] = 'low_support_scaffolded'
        repaired.loc[grp_indices[repair_mask], 'offset_confidence'] = 'low'
    return repaired



def apply_export_floor_policy(modeled_points: gpd.GeoDataFrame, *, export_overlap: np.ndarray, global_prior: float) -> ExportFloorPolicyResult:
    validation_before = evaluate_export_floor_collapse(modeled_points, export_overlap=export_overlap)
    if not validation_before.exceeds_limit:
        return ExportFloorPolicyResult(
            modeled_points=modeled_points,
            validation_before=validation_before,
            validation_after=validation_before,
            policy_applied=False,
            policy_status='not_needed',
        )
    repaired = _repair_export_floor_collapse(modeled_points, export_overlap=export_overlap, global_prior=global_prior)
    validation_after = evaluate_export_floor_collapse(repaired, export_overlap=export_overlap)
    if validation_after.exceeds_limit:
        raise RuntimeError('river_linear_modeled_offset_export_floor_collapse')
    return ExportFloorPolicyResult(
        modeled_points=repaired,
        validation_before=validation_before,
        validation_after=validation_after,
        policy_applied=True,
        policy_status='repaired',
    )


__all__ = [
    'ExportFloorPolicyResult',
    'ExportFloorValidationResult',
    'ModeledOffsetBuildResult',
    'apply_export_floor_policy',
    'build_modeled_offset_points',
    'evaluate_export_floor_collapse',
    'export_overlap_indices',
]
