from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from pipeline.river_linear.river_linear_context import RiverLinearContext
from pipeline.river_linear.river_linear_stage_modeled_offset import ModeledOffsetStageResult
from pipeline.river_linear.river_linear_stage_wse_proxy import WSEProxyStageResult
from pipeline.river_linear.river_linear_validation import validate_backbone_points_gpkg

_GROUP_FIELDS = ('component_id', 'levelpath_id', 'reach_id', 'source_reach_key')
_BACKBONE_PROFILE_GROUP_FIELDS = ('component_id', 'levelpath_id')
_BACKBONE_FLATNESS_DOMINANT_FRACTION_THRESHOLD = 0.65
_BACKBONE_FLATNESS_MIN_FINITE_COUNT = 20
_BACKBONE_MEANINGFUL_RANGE_M = 0.05
_BACKBONE_ROUND_DECIMALS = 2
_BACKBONE_FORMULA_TOLERANCE_M = 1.0e-6
_BACKBONE_MAX_DOWNSTREAM_RISE_PER_STEP_M = 0.25
_BACKBONE_RISE_LIMIT_MAX_ADJUSTMENT_M = 0.75
_BACKBONE_OBSERVED_ANCHOR_CONFLICT_LABELS = {"observed_anchor", "observed", "observed_offset_supported"}


@dataclass(frozen=True)
class BackboneStageResult:
    centerline_bed_backbone_points_path: Path
    record_count: int
    finite_backbone_count: int
    science_summary: dict[str, Any] | None = None



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


def _reports_dir(ctx: RiverLinearContext) -> Path:
    return ctx.paths.root.parent / 'reports'


def _offset_artifacts_dir(ctx: RiverLinearContext) -> Path:
    return _reports_dir(ctx) / 'offset_artifacts'


def _safe_counts(series: Any) -> dict[str, int]:
    if series is None:
        return {}
    counts = pd.Series(series, dtype=object).astype(str).fillna('nan').value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}




def _points_in_export_grid(frame: gpd.GeoDataFrame, export_grid_template: Path) -> gpd.GeoDataFrame:
    """Return points that fall inside the requested AOI export grid.

    The backbone artifact is built on the canonical solve domain, but the v793
    south failure proved that a bad flat segment can be limited to the export
    window. This helper lets the backbone stage evaluate that export subset
    before surface propagation or final DEM construction.
    """
    if frame.empty:
        return frame.copy()
    export_grid_template = Path(export_grid_template)
    if not export_grid_template.exists():
        raise RuntimeError(f'river_linear_backbone_export_grid_missing:{export_grid_template}')
    with rasterio.open(export_grid_template) as ds:
        bounds = ds.bounds
        grid_crs = ds.crs
    work = frame
    if work.crs is not None and grid_crs is not None and str(work.crs) != str(grid_crs):
        work = work.to_crs(grid_crs)
    if work.crs is None and grid_crs is not None:
        raise RuntimeError('river_linear_backbone_export_subset_crs_missing')
    x = work.geometry.x.to_numpy(dtype=float)
    y = work.geometry.y.to_numpy(dtype=float)
    mask = (x >= bounds.left) & (x <= bounds.right) & (y >= bounds.bottom) & (y <= bounds.top)
    return frame.loc[mask].copy()


def _classify_export_subset_backbone_flatness(export_subset: pd.DataFrame) -> dict[str, Any]:
    """Classify flatness in the AOI export subset of the backbone artifact."""
    wse_summary = _value_summary(export_subset.get('wse_proxy_z_m', []))
    offset_summary = _value_summary(export_subset.get('offset_modeled_m', []))
    raw_summary = _value_summary(export_subset.get('bed_backbone_raw_z_m', []))
    final_summary = _value_summary(export_subset.get('bed_backbone_z_m', []))
    wse_flat = bool(wse_summary.get('flatness_fail'))
    offset_flat = bool(offset_summary.get('flatness_fail'))
    raw_flat = bool(raw_summary.get('flatness_fail'))
    final_flat = bool(final_summary.get('flatness_fail'))
    raw_flatness_source = 'raw_not_flat'
    if raw_flat and wse_flat and offset_flat:
        raw_flatness_source = 'wse_and_offset_flat'
    elif raw_flat and wse_flat:
        raw_flatness_source = 'wse_flat'
    elif raw_flat and offset_flat:
        raw_flatness_source = 'flat_after_wse_minus_constant_offset'
    elif raw_flat:
        raw_flatness_source = 'raw_formula_output_flat'

    status = 'passed'
    fail_reason = None
    if final_flat:
        if raw_flat:
            status = 'fail_export_subset_raw_and_final_backbone_flat'
            fail_reason = f'export_subset_backbone_flat:{raw_flatness_source}'
        else:
            status = 'fail_export_subset_smoother_flattened_raw_backbone'
            fail_reason = 'export_subset_backbone_smoother_flattened_raw_signal'
    return {
        'status': status,
        'fail_reason': fail_reason,
        'record_count': int(len(export_subset)),
        'wse_proxy': wse_summary,
        'offset_modeled': offset_summary,
        'raw_bed': raw_summary,
        'final_bed': final_summary,
        'raw_flatness_source': raw_flatness_source,
        'flattening_introduced_by_backbone_smoother': bool(final_flat and not raw_flat),
    }



def validate_backbone_export_subset_flatness(
    ctx: RiverLinearContext,
    backbone_points_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the AOI-export subset of a canonical backbone artifact.

    The canonical solve cache can reuse a previously-built backbone without
    re-running ``run_backbone_stage``.  The active workflow must still reject a
    bad export subset before surface propagation/final DEM export, even when the
    canonical parent artifacts come from cache.
    """
    path = Path(backbone_points_path or ctx.paths.centerline_bed_backbone_points)
    if not path.exists():
        raise RuntimeError(f'river_linear_backbone_points_missing:{path}')
    frame = gpd.read_file(path).copy()
    validate_backbone_points_gpkg(path)
    export_subset = _points_in_export_grid(frame, ctx.paths.export_grid_template)
    export_subset_check = _classify_export_subset_backbone_flatness(export_subset)
    if export_subset_check.get('fail_reason'):
        receipt_path = ctx.paths.backbone_receipt.parent / 'backbone_export_subset_flatness_guard_failure.json'
        _write_json(
            receipt_path,
            {
                'stage': 'centerline_bed_backbone',
                'scope': 'aoi_export_subset',
                'source': 'cached_or_fresh_backbone_artifact',
                'backbone_points': path,
                'export_grid_template': ctx.paths.export_grid_template,
                'check': export_subset_check,
            },
        )
        final_bed = export_subset_check.get('final_bed', {})
        raise RuntimeError(
            'river_linear_backbone_export_subset_flatness_fail:'
            f"reason={export_subset_check.get('fail_reason')}:"
            f"raw_flatness_source={export_subset_check.get('raw_flatness_source')}:"
            f"dominant_value={final_bed.get('dominant_rounded_value_m')}:"
            f"fraction={final_bed.get('dominant_rounded_fraction')}:"
            f"summary={receipt_path}"
        )
    return export_subset_check


def _value_summary(values: Any) -> dict[str, Any]:
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
        'flatness_threshold': float(_BACKBONE_FLATNESS_DOMINANT_FRACTION_THRESHOLD),
        'flatness_min_finite_count': int(_BACKBONE_FLATNESS_MIN_FINITE_COUNT),
    }
    if vals.size == 0:
        return summary
    min_v = float(np.nanmin(vals))
    max_v = float(np.nanmax(vals))
    summary['min_m'] = min_v
    summary['max_m'] = max_v
    summary['range_m'] = float(max_v - min_v)
    rounded = np.round(vals, _BACKBONE_ROUND_DECIMALS)
    unique, counts = np.unique(rounded, return_counts=True)
    if counts.size:
        idx = int(np.argmax(counts))
        dominant_count = int(counts[idx])
        fraction = float(dominant_count / vals.size)
        summary['dominant_rounded_value_m'] = float(unique[idx])
        summary['dominant_rounded_count'] = dominant_count
        summary['dominant_rounded_fraction'] = fraction
        summary['flatness_fail'] = bool(
            vals.size >= _BACKBONE_FLATNESS_MIN_FINITE_COUNT
            and fraction >= _BACKBONE_FLATNESS_DOMINANT_FRACTION_THRESHOLD
        )
    return summary


def _max_abs_diff(a: Any, b: Any) -> float | None:
    av = np.asarray(pd.to_numeric(a, errors='coerce'), dtype=float)
    bv = np.asarray(pd.to_numeric(b, errors='coerce'), dtype=float)
    if av.shape != bv.shape:
        return None
    mask = np.isfinite(av) & np.isfinite(bv)
    if not np.any(mask):
        return None
    return float(np.nanmax(np.abs(av[mask] - bv[mask])))


def _classify_backbone_formula_handoff_group(grp: pd.DataFrame) -> dict[str, Any]:
    wse_summary = _value_summary(grp.get('wse_proxy_z_m', []))
    offset_summary = _value_summary(grp.get('offset_modeled_m', []))
    raw_summary = _value_summary(grp.get('bed_backbone_raw_z_m', []))
    wse_range = wse_summary.get('range_m')
    offset_range = offset_summary.get('range_m')
    wse_varies = bool(wse_range is not None and float(wse_range) >= _BACKBONE_MEANINGFUL_RANGE_M)
    offset_varies = bool(offset_range is not None and float(offset_range) >= _BACKBONE_MEANINGFUL_RANGE_M)
    raw_is_flat = bool(raw_summary.get('flatness_fail'))
    status = 'ok_raw_bed_not_flat'
    fail_reason = None
    if raw_is_flat and wse_varies and offset_varies:
        status = 'fail_raw_bed_flat_after_variable_wse_and_offset'
        fail_reason = 'backbone_raw_bed_flat_after_formula_inputs_varied'
    elif raw_is_flat and wse_varies and not offset_varies:
        status = 'fail_raw_bed_flat_despite_variable_wse_and_constant_offset'
        fail_reason = 'backbone_formula_or_join_inconsistent'
    elif raw_is_flat and not wse_varies:
        status = 'context_raw_bed_flat_because_wse_flat_or_weak'
    return {
        'status': status,
        'fail_reason': fail_reason,
        'wse_proxy': wse_summary,
        'offset_modeled': offset_summary,
        'raw_bed': raw_summary,
    }


def _backbone_formula_handoff_guard(merged: pd.DataFrame) -> list[dict[str, Any]]:
    required = {'wse_proxy_z_m', 'offset_modeled_m', 'bed_backbone_raw_z_m'}
    if not required.issubset(set(merged.columns)):
        return [{
            'group_key': 'all',
            'record_count': int(len(merged)),
            'status': 'fail_missing_formula_inputs',
            'fail_reason': 'backbone_missing_wse_offset_raw_bed_inputs',
            'missing_columns': sorted(required.difference(set(merged.columns))),
        }]
    checks: list[dict[str, Any]] = []
    keys = _group_key(merged)
    for _, grp_idx in keys.groupby(keys).groups.items():
        grp = merged.loc[list(grp_idx)].copy()
        check = _classify_backbone_formula_handoff_group(grp)
        check['group_key'] = str(_group_key(grp).iloc[0]) if len(grp) else 'unknown'
        check['record_count'] = int(len(grp))
        if 'offset_raw_bed_z_m' in grp.columns:
            formula_diff = _max_abs_diff(grp['bed_backbone_raw_z_m'], grp['offset_raw_bed_z_m'])
            check['modeled_offset_raw_bed_max_abs_diff_m'] = formula_diff
            if formula_diff is not None and formula_diff > _BACKBONE_FORMULA_TOLERANCE_M:
                check['status'] = 'fail_offset_raw_bed_disagrees_with_backbone_raw_bed'
                check['fail_reason'] = 'backbone_raw_bed_formula_handoff_mismatch'
        else:
            check['modeled_offset_raw_bed_max_abs_diff_m'] = None
            check['status'] = 'fail_missing_offset_raw_bed_handoff'
            check['fail_reason'] = 'backbone_missing_modeled_offset_raw_bed_handoff'
        if 'wse_proxy_z_m_modeled' in grp.columns:
            wse_diff = _max_abs_diff(grp['wse_proxy_z_m'], grp['wse_proxy_z_m_modeled'])
            check['modeled_offset_wse_max_abs_diff_m'] = wse_diff
            if wse_diff is not None and wse_diff > _BACKBONE_FORMULA_TOLERANCE_M:
                check['status'] = 'fail_wse_disagrees_between_wse_and_modeled_offset_artifacts'
                check['fail_reason'] = 'backbone_wse_handoff_mismatch'
        if 'station_m_modeled' in grp.columns:
            station_diff = _max_abs_diff(grp['station_m'], grp['station_m_modeled'])
            check['modeled_offset_station_max_abs_diff_m'] = station_diff
            if station_diff is not None and station_diff > _BACKBONE_FORMULA_TOLERANCE_M:
                check['status'] = 'fail_station_disagrees_between_wse_and_modeled_offset_artifacts'
                check['fail_reason'] = 'backbone_station_handoff_mismatch'
        checks.append(check)
    return checks


def _apply_backbone_formula_guard_columns(out: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    check_by_group = {str(check.get('group_key')): check for check in checks}
    keys = _group_key(out)
    for group_key, grp_idx in keys.groupby(keys).groups.items():
        check = check_by_group.get(str(group_key), {})
        raw_bed = check.get('raw_bed', {}) if isinstance(check, dict) else {}
        wse_proxy = check.get('wse_proxy', {}) if isinstance(check, dict) else {}
        offset_modeled = check.get('offset_modeled', {}) if isinstance(check, dict) else {}
        idx = list(grp_idx)
        out.loc[idx, 'backbone_formula_guard_status'] = str(check.get('status', 'not_evaluated'))
        out.loc[idx, 'backbone_formula_guard_fail_reason'] = check.get('fail_reason')
        out.loc[idx, 'backbone_group_wse_range_m'] = wse_proxy.get('range_m')
        out.loc[idx, 'backbone_group_offset_range_m'] = offset_modeled.get('range_m')
        out.loc[idx, 'backbone_group_raw_bed_range_m'] = raw_bed.get('range_m')
        out.loc[idx, 'backbone_group_raw_bed_dominant_fraction'] = raw_bed.get('dominant_rounded_fraction')
        out.loc[idx, 'backbone_group_raw_bed_handoff_max_abs_diff_m'] = check.get('modeled_offset_raw_bed_max_abs_diff_m')

def _group_key(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in _BACKBONE_PROFILE_GROUP_FIELDS if c in frame.columns]
    if not cols:
        return pd.Series(['all'] * len(frame), index=frame.index, dtype=object)
    return frame[cols].astype(str).agg('|'.join, axis=1)


def _rolling_nanmedian(values: np.ndarray, *, window: int = 9) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    out = series.rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)
    return out


def _rolling_nanmean(values: np.ndarray, *, window: int = 7) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    out = series.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    return out


def _dominant_rounded_fraction(values: np.ndarray, *, decimals: int = 2) -> tuple[float | None, float | None]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    rounded = np.round(vals, decimals)
    unique, counts = np.unique(rounded, return_counts=True)
    idx = int(np.argmax(counts))
    return float(unique[idx]), float(counts[idx] / vals.size)



def _weighted_isotonic_nonincreasing(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted least-squares isotonic fit constrained to be non-increasing."""
    y = np.asarray(values, dtype=float)
    out = y.copy()
    valid = np.isfinite(y)
    if np.count_nonzero(valid) <= 1:
        return out
    x = -y[valid]
    if weights is None:
        w = np.ones_like(x, dtype=float)
    else:
        warr = np.asarray(weights, dtype=float)
        w = warr[valid]
        w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)
    levels: list[float] = []
    wsum: list[float] = []
    counts: list[int] = []
    for xi, wi in zip(x, w):
        levels.append(float(xi))
        wsum.append(float(wi))
        counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            new_w = wsum[-2] + wsum[-1]
            new_level = (levels[-2] * wsum[-2] + levels[-1] * wsum[-1]) / new_w
            new_count = counts[-2] + counts[-1]
            levels[-2:] = [new_level]
            wsum[-2:] = [new_w]
            counts[-2:] = [new_count]
    fitted_neg = np.concatenate([np.full(c, level, dtype=float) for level, c in zip(levels, counts)])
    out[np.flatnonzero(valid)] = -fitted_neg
    return out


def _support_weights_for_monotone(support_class: np.ndarray | None, n: int) -> np.ndarray:
    weights = np.ones(int(n), dtype=float)
    if support_class is None:
        return weights
    support = np.asarray(support_class, dtype=object).astype(str)
    weights[np.isin(support, (
        'global_prior', 'unsupported_component', 'unsupported', 'unsupported_no_confident_offset',
        'single_anchor_blend_to_prior', 'solve_domain_global_prior', 'low_support_scaffolded'
    ))] = 0.75
    weights[np.isin(support, (
        'inferred', 'linear_interpolation', 'interpolated_between_observed_offsets',
        'near_observed_offset_supported'
    ))] = 1.5
    weights[np.isin(support, ('observed_anchor', 'observed', 'observed_offset_supported'))] = 8.0
    return weights


def _oriented_downstream_station(station_m: Any, wse_proxy_z_m: Any | None = None) -> tuple[np.ndarray, int, float | None]:
    """Return downstream-increasing station and inferred native station orientation.

    If WSE increases with native station, the stationing is probably oriented
    upstream, so the downstream coordinate is flipped for monotone checks.
    """
    sta = np.asarray(pd.to_numeric(station_m, errors='coerce'), dtype=float)
    if sta.size == 0:
        return sta.copy(), 1, None
    sign = 1
    slope = None
    if wse_proxy_z_m is not None:
        wse = np.asarray(pd.to_numeric(wse_proxy_z_m, errors='coerce'), dtype=float)
        mask = np.isfinite(sta) & np.isfinite(wse)
        if np.count_nonzero(mask) >= 3 and float(np.nanmax(sta[mask]) - np.nanmin(sta[mask])) > 0.0:
            try:
                slope = float(np.polyfit(sta[mask], wse[mask], 1)[0])
            except Exception:
                slope = None
            if slope is not None and np.isfinite(slope) and slope > 1.0e-6:
                sign = -1
    finite = sta[np.isfinite(sta)]
    if finite.size == 0:
        return sta.copy(), sign, slope
    downstream = sta - float(np.nanmin(finite)) if sign >= 0 else float(np.nanmax(finite)) - sta
    return downstream, sign, slope

def _is_observed_anchor_support(value: Any) -> bool:
    """Return True when a backbone point came from observed offset support."""
    label = str(value) if value is not None else ""
    return label in _BACKBONE_OBSERVED_ANCHOR_CONFLICT_LABELS


def _limit_downstream_bed_rises(
    values: np.ndarray,
    *,
    support_class: np.ndarray | None = None,
    max_rise_m: float = _BACKBONE_MAX_DOWNSTREAM_RISE_PER_STEP_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Limit local downstream bed rises for the final guidance backbone.

    The backbone is guidance used to shape interpolation; hard authoritative
    cells are reimposed later by the authoritative-lock stage. Therefore even
    observed-offset-supported backbone points should not create large local
    downstream bed rises. When an observed-support point must be adjusted, the
    adjustment is retained in explicit conflict fields rather than hidden.
    """
    vals = np.asarray(values, dtype=float)
    limited = vals.copy()
    adjustment = np.zeros(vals.shape, dtype=float)
    observed_anchor_conflict = np.zeros(vals.shape, dtype=bool)
    support = None
    if support_class is not None:
        support = np.asarray(support_class, dtype=object).astype(str)
        if support.shape != vals.shape:
            support = None
    if vals.size <= 1:
        return limited, adjustment, observed_anchor_conflict
    for i in range(1, vals.size):
        if not (np.isfinite(limited[i]) and np.isfinite(limited[i - 1])):
            continue
        allowed = float(limited[i - 1]) + float(max_rise_m)
        if limited[i] <= allowed:
            continue
        label = support[i] if support is not None else None
        target = allowed
        if target < limited[i]:
            adjustment[i] = float(target - vals[i])
            observed_anchor_conflict[i] = bool(_is_observed_anchor_support(label))
            limited[i] = target
    return limited, adjustment, observed_anchor_conflict


def _bounded_residual_preserving_smooth(
    *,
    station_m: np.ndarray,
    raw_backbone_z_m: np.ndarray,
    support_class: np.ndarray | None,
) -> np.ndarray:
    """Return a lightly smoothed backbone without whole-reach monotone collapse.

    The WSE proxy stage is responsible for the broad downstream water-surface trend.
    This stage should preserve the bed implied by ``wse_proxy_z_m - offset_modeled_m``
    and only remove small point-scale jitter. A final isotonic projection here can turn
    a low-support reach into a single flat value, so this smoother is intentionally
    bounded against the raw backbone and preserves observed anchors exactly.
    """
    sta = np.asarray(station_m, dtype=float)
    raw = np.asarray(raw_backbone_z_m, dtype=float)
    valid = np.isfinite(sta) & np.isfinite(raw)
    out = raw.copy()
    if np.count_nonzero(valid) <= 2:
        return out

    valid_positions = np.flatnonzero(valid)
    order = np.argsort(sta[valid], kind='mergesort')
    ordered_positions = valid_positions[order]
    raw_v = raw[ordered_positions]

    support_v = None
    if support_class is not None:
        support_arr = np.asarray(support_class, dtype=object)
        support_v = support_arr[ordered_positions].astype(str)

    smoothed = _rolling_nanmedian(raw_v, window=7)
    smoothed = _rolling_nanmean(smoothed, window=5)
    raw_range = float(np.nanmax(raw_v) - np.nanmin(raw_v)) if np.count_nonzero(np.isfinite(raw_v)) else 0.0
    max_adjust = min(0.75, max(0.10, raw_range * 0.10))
    candidate = np.clip(smoothed, raw_v - max_adjust, raw_v + max_adjust)

    # Low-support/global-prior reaches should remain close to the raw WSE-minus-offset
    # backbone. Otherwise a smooth prior can erase the only available longitudinal signal.
    blend = np.full(raw_v.shape, 0.35, dtype=float)
    if support_v is not None:
        observed = np.isin(support_v, ('observed_anchor', 'observed', 'observed_offset_supported'))
        medium = np.isin(support_v, ('inferred', 'linear_interpolation', 'interpolated_between_observed_offsets', 'near_observed_offset_supported'))
        low = np.isin(support_v, ('global_prior', 'unsupported_component', 'unsupported', 'unsupported_no_confident_offset', 'single_anchor_blend_to_prior', 'solve_domain_global_prior', 'low_support_scaffolded'))
        blend[medium] = 0.30
        blend[low] = 0.15
        blend[observed] = 0.0
    blended = raw_v + blend * (candidate - raw_v)

    # Do not run the old whole-profile isotonic projection here. Instead, use a
    # support-aware local rise limiter: small bed undulations are retained, but
    # large downstream rises are damped to the allowed step before the final
    # authoritative lock. Observed-offset-supported conflicts are reported
    # through adjustment fields rather than preserved as guidance jumps.
    blended, _rise_limiter_adjustment, _observed_anchor_conflict = _limit_downstream_bed_rises(
        blended,
        support_class=support_v,
    )

    raw_value, raw_fraction = _dominant_rounded_fraction(raw_v, decimals=2)
    final_value, final_fraction = _dominant_rounded_fraction(blended, decimals=2)
    if (
        raw_fraction is not None
        and final_fraction is not None
        and raw_fraction < 0.65
        and final_fraction >= 0.65
        and np.nanmax(raw_v) - np.nanmin(raw_v) > 0.25
    ):
        raise RuntimeError(
            'river_linear_backbone_transform_flattened_raw_signal:'
            f'raw_dominant_value={raw_value}:raw_fraction={raw_fraction}:'
            f'final_dominant_value={final_value}:final_fraction={final_fraction}'
        )

    out[ordered_positions] = blended
    return out




def _write_bed_backbone_audit(
    ctx: RiverLinearContext,
    *,
    out: gpd.GeoDataFrame,
    formula_checks: list[dict[str, Any]],
    science_summary: dict[str, Any],
) -> tuple[Path, Path, Path]:
    reports = _reports_dir(ctx)
    artifacts = _offset_artifacts_dir(ctx)
    artifacts.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts / ctx.paths.centerline_bed_backbone_points.name
    if artifact_path.exists():
        artifact_path.unlink()
    out.to_file(artifact_path, driver='GPKG')
    audit_path = reports / 'BED_BACKBONE_AUDIT.txt'
    manifest_path = artifacts / 'bed_backbone_artifact_manifest.json'
    support_counts = _safe_counts(out.get('backbone_support_class'))
    source_counts = _safe_counts(out.get('bed_backbone_source'))
    lines = [
        'Bed backbone audit',
        '==================',
        '',
        'Stage: centerline_bed_backbone',
        'Construction:',
        '  centerline_wse_proxy_points.gpkg + centerline_offset_modeled_points.gpkg',
        '      -> bed_backbone_raw_z_m = wse_proxy_z_m - offset_modeled_m',
        '      -> bed_backbone_z_m = bounded local smooth(raw bed) + support-aware downstream-rise limiter',
        '',
        'Science invariant:',
        '  Bed backbone is derived from WSE minus modeled offset.',
        '  This stage preserves the raw formula signal, performs bounded local smoothing, and limits large local downstream bed rises before authoritative locking.',
        '  Observed-offset-supported conflicts can be adjusted in the guidance backbone and are reported explicitly; hard authoritative cells are reimposed later.',
        '  Large adjustments are reported; no whole-profile monotone projection is applied.',
        '  Support class and confidence are carried from the modeled-offset stage.',
        '  Low-support reaches remain labeled through the backbone artifact.',
        '',
        f'Backbone artifact: {ctx.paths.centerline_bed_backbone_points}',
        f'Retained backbone artifact: {artifact_path}',
        '',
        f'Record count: {len(out)}',
        f'Finite backbone count: {science_summary.get("finite_backbone_count")}',
        f'Downstream trend violation count: {science_summary.get("downstream_trend_violation_count")}',
        f'Large downstream rise count > allowed: {science_summary.get("large_downstream_rise_count_gt_allowed")}',
        f'Max downstream rise (m): {science_summary.get("max_downstream_rise_m")}',
        f'Largest step (m): {science_summary.get("largest_step_m")}',
        f'Largest adjustment (m): {science_summary.get("largest_adjustment_m")}',
        f'Adjustment > 0.25 m count: {science_summary.get("adjustment_gt_0_25m_count")}',
        f'Adjustment > 0.50 m count: {science_summary.get("adjustment_gt_0_50m_count")}',
        f'Adjustment > 1.00 m count: {science_summary.get("adjustment_gt_1m_count")}',
        f'Adjustment > 2.00 m count: {science_summary.get("adjustment_gt_2m_count")}',
        f'Adjustment > 5.00 m count: {science_summary.get("adjustment_gt_5m_count")}',
        '',
        'Support class counts:',
        json.dumps(support_counts, indent=2, sort_keys=True),
        '',
        'Backbone source counts:',
        json.dumps(source_counts, indent=2, sort_keys=True),
        '',
        'Science summary:',
        json.dumps(_jsonable(science_summary), indent=2, sort_keys=True),
        '',
        'Formula handoff guard:',
        json.dumps(_jsonable({'group_count': len(formula_checks), 'groups': formula_checks}), indent=2, sort_keys=True),
    ]
    _write_text(audit_path, lines)
    _write_json(
        manifest_path,
        {
            'stage': 'centerline_bed_backbone',
            'retained_artifacts': {
                'centerline_bed_backbone_points': str(artifact_path),
                'bed_backbone_audit': str(audit_path),
            },
            'support_class_counts': support_counts,
            'bed_backbone_source_counts': source_counts,
            'science_summary': science_summary,
        },
    )
    return audit_path, artifact_path, manifest_path


def run_backbone_stage(
    ctx: RiverLinearContext,
    wse_result: WSEProxyStageResult,
    modeled_offset_result: ModeledOffsetStageResult,
) -> BackboneStageResult:
    wse = gpd.read_file(wse_result.centerline_wse_proxy_points_path).copy()
    modeled = gpd.read_file(modeled_offset_result.centerline_modeled_offset_points_path).copy()
    if 'point_id' not in wse.columns or 'point_id' not in modeled.columns:
        raise RuntimeError('river_linear_backbone_missing_point_id')
    wse['point_id'] = wse['point_id'].astype(str)
    modeled['point_id'] = modeled['point_id'].astype(str)
    merged = wse.merge(
        modeled.drop(columns=['geometry'], errors='ignore'),
        on='point_id',
        how='inner',
        suffixes=('', '_modeled'),
    )
    if merged.empty:
        raise RuntimeError('river_linear_backbone_empty_join')
    merged['wse_proxy_z_m'] = pd.to_numeric(merged['wse_proxy_z_m'], errors='coerce')
    merged['offset_modeled_m'] = pd.to_numeric(merged['offset_modeled_m'], errors='coerce')
    merged['backbone_profile_group_id'] = _group_key(merged).astype(str)
    merged['bed_backbone_raw_z_m'] = merged['wse_proxy_z_m'] - merged['offset_modeled_m']
    formula_checks = _backbone_formula_handoff_guard(merged)
    formula_failures = [check for check in formula_checks if check.get('fail_reason')]
    if formula_failures:
        receipt_path = ctx.paths.backbone_receipt.parent / 'backbone_formula_handoff_guard_failure.json'
        _write_json(
            receipt_path,
            {
                'stage': 'centerline_bed_backbone',
                'failure_count': int(len(formula_failures)),
                'failures': formula_failures,
            },
        )
        first = formula_failures[0]
        raw_bed = first.get('raw_bed', {})
        raise RuntimeError(
            'river_linear_backbone_formula_handoff_fail:'
            f"reason={first.get('fail_reason')}:"
            f"group={first.get('group_key')}:"
            f"dominant_value={raw_bed.get('dominant_rounded_value_m')}:"
            f"fraction={raw_bed.get('dominant_rounded_fraction')}:"
            f"summary={receipt_path}"
        )
    _apply_backbone_formula_guard_columns(merged, formula_checks)
    merged['station_downstream_m'] = np.nan
    merged['downstream_order_sign'] = np.nan
    merged['wse_station_slope_m_per_m'] = np.nan
    merged['bed_backbone_z_m'] = np.nan
    group_keys = _group_key(merged)
    for _, grp_idx in group_keys.groupby(group_keys).groups.items():
        grp = merged.loc[list(grp_idx)].copy().sort_values('station_m', kind='mergesort')
        raw = pd.to_numeric(grp['bed_backbone_raw_z_m'], errors='coerce').to_numpy(dtype=float)
        if np.count_nonzero(np.isfinite(raw)) == 0:
            continue
        downstream_station, direction_sign, wse_slope = _oriented_downstream_station(
            grp['station_m'], grp['wse_proxy_z_m'] if 'wse_proxy_z_m' in grp.columns else None
        )
        support = grp['offset_support_class'].astype(str).to_numpy(dtype=object) if 'offset_support_class' in grp.columns else None
        merged.loc[grp.index, 'station_downstream_m'] = downstream_station
        merged.loc[grp.index, 'downstream_order_sign'] = int(direction_sign)
        merged.loc[grp.index, 'wse_station_slope_m_per_m'] = wse_slope
        merged.loc[grp.index, 'bed_backbone_z_m'] = _bounded_residual_preserving_smooth(
            station_m=downstream_station,
            raw_backbone_z_m=raw,
            support_class=support,
        )
    merged['bed_backbone_adjustment_m'] = pd.to_numeric(merged['bed_backbone_z_m'], errors='coerce') - pd.to_numeric(merged['bed_backbone_raw_z_m'], errors='coerce')
    adj_abs = np.abs(pd.to_numeric(merged["bed_backbone_adjustment_m"], errors="coerce").to_numpy(dtype=float))
    merged["bed_backbone_adjustment_class"] = "not_evaluated"
    finite_adj_mask = np.isfinite(adj_abs)
    merged.loc[finite_adj_mask & (adj_abs <= 1.0e-9), "bed_backbone_adjustment_class"] = "none"
    merged.loc[finite_adj_mask & (adj_abs > 1.0e-9) & (adj_abs <= 0.25), "bed_backbone_adjustment_class"] = "minor_local_smoothing"
    merged.loc[finite_adj_mask & (adj_abs > 0.25) & (adj_abs <= 0.50), "bed_backbone_adjustment_class"] = "moderate_local_smoothing"
    merged.loc[finite_adj_mask & (adj_abs > 0.50) & (adj_abs <= 1.00), "bed_backbone_adjustment_class"] = "large_local_smoothing_warning"
    merged.loc[finite_adj_mask & (adj_abs > 1.00) & (adj_abs <= 2.00), "bed_backbone_adjustment_class"] = "very_large_adjustment_warning"
    merged.loc[finite_adj_mask & (adj_abs > 2.00) & (adj_abs <= 5.00), "bed_backbone_adjustment_class"] = "severe_adjustment_warning"
    merged.loc[finite_adj_mask & (adj_abs > 5.00), "bed_backbone_adjustment_class"] = "extreme_adjustment_warning"
    merged["bed_backbone_adjustment_reason"] = merged["bed_backbone_adjustment_class"]
    merged['backbone_support_class'] = merged['offset_support_class'].astype(str) if 'offset_support_class' in merged.columns else 'inferred'
    merged['bed_backbone_source'] = merged['offset_source'].astype(str) if 'offset_source' in merged.columns else 'wse_minus_modeled_offset'
    observed_anchor_adjustment_mask = finite_adj_mask & (adj_abs > 1.0e-9) & merged["backbone_support_class"].astype(str).isin(_BACKBONE_OBSERVED_ANCHOR_CONFLICT_LABELS)
    merged.loc[observed_anchor_adjustment_mask, "bed_backbone_adjustment_reason"] = "observed_offset_anchor_adjusted_by_backbone_rise_limiter"
    merged['bed_backbone_confidence'] = merged['offset_confidence'].astype(str) if 'offset_confidence' in merged.columns else 'unknown'
    keep = [c for c in (
        'point_id', 'station_m', 'station_downstream_m', 'offset_model_station_m', 'downstream_order_sign', 'wse_station_slope_m_per_m', *_GROUP_FIELDS, 'backbone_profile_group_id',
        # Keep the direct formula inputs in the backbone artifact so the science
        # audit can identify whether flatness entered through WSE, modeled
        # offset, raw bed construction, or the final bounded smoother.
        'wse_proxy_z_m', 'offset_modeled_m', 'offset_raw_bed_z_m',
        'offset_source', 'offset_support_class', 'offset_confidence',
        'distance_to_nearest_observed_offset_m', 'upstream_observed_offset_distance_m',
        'downstream_observed_offset_distance_m',
        'bed_backbone_raw_z_m', 'bed_backbone_z_m',
        'bed_backbone_adjustment_m', 'bed_backbone_adjustment_class', 'bed_backbone_adjustment_reason', 'backbone_support_class', 'bed_backbone_source', 'bed_backbone_confidence',
        'backbone_formula_guard_status', 'backbone_formula_guard_fail_reason',
        'backbone_group_wse_range_m', 'backbone_group_offset_range_m',
        'backbone_group_raw_bed_range_m', 'backbone_group_raw_bed_dominant_fraction',
        'backbone_group_raw_bed_handoff_max_abs_diff_m',
        'geometry'
    ) if c in merged.columns]
    out = gpd.GeoDataFrame(merged[keep].copy(), geometry='geometry', crs=wse.crs)
    sort_cols = [c for c in (*[c for c in _GROUP_FIELDS if c in out.columns], 'station_m', 'point_id') if c in out.columns]
    out = out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    if ctx.paths.centerline_bed_backbone_points.exists():
        ctx.paths.centerline_bed_backbone_points.unlink()
    out.to_file(ctx.paths.centerline_bed_backbone_points, driver='GPKG')
    export_subset_check = validate_backbone_export_subset_flatness(ctx, ctx.paths.centerline_bed_backbone_points)
    finite = int(np.count_nonzero(np.isfinite(pd.to_numeric(out['bed_backbone_z_m'], errors='coerce').to_numpy(dtype=float))))
    bed = pd.to_numeric(out.get('bed_backbone_z_m'), errors='coerce').to_numpy(dtype=float) if 'bed_backbone_z_m' in out.columns else np.asarray([], dtype=float)
    raw_bed = pd.to_numeric(out.get('bed_backbone_raw_z_m'), errors='coerce').to_numpy(dtype=float) if 'bed_backbone_raw_z_m' in out.columns else np.asarray([], dtype=float)
    finite_bed = bed[np.isfinite(bed)]
    abrupt_step_count = 0
    largest_step = None
    trend_violation_count = 0
    large_downstream_rise_count = 0
    max_downstream_rise_m = None
    for _, grp_idx in _group_key(out).groupby(_group_key(out)).groups.items():
        sort_col = 'station_downstream_m' if 'station_downstream_m' in out.columns else 'station_m'
        grp = out.loc[list(grp_idx)].copy().sort_values(sort_col, kind='mergesort')
        vals_all = pd.to_numeric(grp['bed_backbone_z_m'], errors='coerce').to_numpy(dtype=float)
        sta_all = pd.to_numeric(grp[sort_col], errors='coerce').to_numpy(dtype=float)
        mask = np.isfinite(vals_all) & np.isfinite(sta_all)
        vals = vals_all[mask]
        if vals.size > 1:
            diffs = np.diff(vals)
            absdiff = np.abs(diffs)
            local_largest = float(np.nanmax(absdiff)) if absdiff.size else None
            largest_step = local_largest if largest_step is None else max(largest_step, local_largest)
            if absdiff.size:
                threshold = max(5.0, float(np.nanpercentile(absdiff, 95.0)) * 3.0)
                abrupt_step_count += int(np.count_nonzero(absdiff > threshold))
            trend_violation_count += int(np.count_nonzero(diffs > 1.0e-9))
            positive = diffs[diffs > 1.0e-9]
            if positive.size:
                local_max_rise = float(np.nanmax(positive))
                max_downstream_rise_m = local_max_rise if max_downstream_rise_m is None else max(max_downstream_rise_m, local_max_rise)
                large_downstream_rise_count += int(np.count_nonzero(positive > _BACKBONE_MAX_DOWNSTREAM_RISE_PER_STEP_M))
    direction_counts = out.get('downstream_order_sign', pd.Series(dtype=float)).dropna().astype(int).value_counts().to_dict() if 'downstream_order_sign' in out.columns else {}
    support_counts = out.get('backbone_support_class', pd.Series(dtype=object)).astype(str).value_counts().to_dict() if 'backbone_support_class' in out.columns else {}
    adjustments = bed - raw_bed if bed.size == raw_bed.size else np.asarray([], dtype=float)
    raw_dom_value, raw_dom_fraction = _dominant_rounded_fraction(raw_bed, decimals=2)
    final_dom_value, final_dom_fraction = _dominant_rounded_fraction(bed, decimals=2)
    finite_raw = raw_bed[np.isfinite(raw_bed)]
    science_summary = {
        'backbone_formula': 'bed_backbone_raw_z_m = wse_proxy_z_m - offset_modeled_m; bed_backbone_z_m = bounded_local_smooth_plus_support_aware_rise_limiter(raw)',
        'max_allowed_downstream_rise_per_step_m': float(_BACKBONE_MAX_DOWNSTREAM_RISE_PER_STEP_M),
        'observed_anchor_rise_limiter_policy': 'observed_offset_supported_backbone_points_may_be_adjusted_but_conflicts_are_reported',
        'guidance_rise_limiter_policy': 'limit_all_final_backbone_points_to_allowed_downstream_rise_before_authoritative_lock',
        'raw_bed_min_m': float(np.nanmin(finite_raw)) if finite_raw.size else None,
        'raw_bed_max_m': float(np.nanmax(finite_raw)) if finite_raw.size else None,
        'raw_bed_range_m': float(np.nanmax(finite_raw) - np.nanmin(finite_raw)) if finite_raw.size else None,
        'raw_dominant_rounded_value_m': raw_dom_value,
        'raw_dominant_rounded_fraction': raw_dom_fraction,
        'final_dominant_rounded_value_m': final_dom_value,
        'final_dominant_rounded_fraction': final_dom_fraction,
        'bed_min_m': float(np.nanmin(finite_bed)) if finite_bed.size else None,
        'bed_max_m': float(np.nanmax(finite_bed)) if finite_bed.size else None,
        'bed_median_m': float(np.nanmedian(finite_bed)) if finite_bed.size else None,
        'finite_backbone_count': int(finite),
        'downstream_trend_violation_count': int(trend_violation_count),
        'large_downstream_rise_count_gt_allowed': int(large_downstream_rise_count),
        'max_downstream_rise_m': max_downstream_rise_m,
        'positive_downstream_step_count': int(trend_violation_count),
        'downstream_order_sign_counts': {str(k): int(v) for k, v in direction_counts.items()},
        'abrupt_step_count': int(abrupt_step_count),
        'largest_step_m': largest_step,
        'adjustment_count': int(np.count_nonzero(np.isfinite(adjustments) & (np.abs(adjustments) > 1.0e-9))),
        'largest_adjustment_m': float(np.nanmax(np.abs(adjustments[np.isfinite(adjustments)]))) if np.count_nonzero(np.isfinite(adjustments)) else None,
        'adjustment_gt_0_25m_count': int(np.count_nonzero(np.isfinite(adjustments) & (np.abs(adjustments) > 0.25))),
        'adjustment_gt_0_50m_count': int(np.count_nonzero(np.isfinite(adjustments) & (np.abs(adjustments) > 0.50))),
        'adjustment_gt_1m_count': int(np.count_nonzero(np.isfinite(adjustments) & (np.abs(adjustments) > 1.00))),
        'adjustment_gt_2m_count': int(np.count_nonzero(np.isfinite(adjustments) & (np.abs(adjustments) > 2.00))),
        'adjustment_gt_5m_count': int(np.count_nonzero(np.isfinite(adjustments) & (np.abs(adjustments) > 5.00))),
        'adjustment_class_counts': _safe_counts(out.get('bed_backbone_adjustment_class')),
        'support_class_counts': {str(k): int(v) for k, v in support_counts.items()},
        'formula_handoff_guard': {
            'status': 'passed',
            'group_count': int(len(formula_checks)),
            'failure_count': int(sum(1 for check in formula_checks if check.get('fail_reason'))),
            'groups': formula_checks,
        },
    }
    backbone_audit_path, retained_backbone_path, backbone_manifest_path = _write_bed_backbone_audit(
        ctx,
        out=out,
        formula_checks=formula_checks,
        science_summary=science_summary,
    )
    science_summary['bed_backbone_audit_path'] = str(backbone_audit_path)
    science_summary['retained_bed_backbone_points_path'] = str(retained_backbone_path)
    science_summary['bed_backbone_artifact_manifest_path'] = str(backbone_manifest_path)
    return BackboneStageResult(
        centerline_bed_backbone_points_path=ctx.paths.centerline_bed_backbone_points,
        record_count=int(len(out)),
        finite_backbone_count=finite,
        science_summary=science_summary,
    )


__all__ = ['BackboneStageResult', 'run_backbone_stage', '_classify_backbone_formula_handoff_group', '_weighted_isotonic_nonincreasing', '_oriented_downstream_station']
