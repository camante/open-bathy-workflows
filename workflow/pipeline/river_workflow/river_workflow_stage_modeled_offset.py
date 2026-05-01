from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd

from pipeline.river_workflow.river_workflow_context import RiverWorkflowContext
from pipeline.river_workflow.river_workflow_modeled_offset_logic import build_modeled_offset_points
from pipeline.river_workflow.river_workflow_stage_observed_offset import ObservedOffsetStageResult
from pipeline.river_workflow.river_workflow_stage_wse_proxy import WSEProxyStageResult
from pipeline.river_workflow.river_workflow_validation import validate_modeled_offset_points_gpkg


@dataclass(frozen=True)
class ModeledOffsetStageResult:
    centerline_modeled_offset_points_path: Path
    record_count: int
    finite_modeled_offset_count: int
    global_observed_median_offset_m: float | None
    global_prior_offset_m: float | None = None
    export_floor_fraction_before_policy: float | None = None
    export_floor_fraction_after_policy: float | None = None
    export_floor_policy_applied: bool = False
    export_floor_policy_status: str = 'not_evaluated'
    science_summary: dict[str, Any] | None = None


_PROFILE_GROUP_FIELDS = ('component_id', 'levelpath_id')
_IDENTITY_FIELDS = ('component_id', 'levelpath_id', 'reach_id', 'source_reach_key')
_OFFSET_FLATNESS_DOMINANT_FRACTION_THRESHOLD = 0.65
_OFFSET_FLATNESS_MIN_FINITE_COUNT = 20
_OFFSET_MEANINGFUL_RANGE_M = 0.05
_OFFSET_ROUND_DECIMALS = 2


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


def _offset_artifacts_dir(ctx: RiverWorkflowContext) -> Path:
    return _reports_dir(ctx) / 'offset_artifacts'


def _numeric_summary(values: Any) -> dict[str, Any]:
    vals = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {
            'finite_count': 0,
            'min_m': None,
            'median_m': None,
            'max_m': None,
            'range_m': None,
        }
    return {
        'finite_count': int(finite.size),
        'min_m': float(np.nanmin(finite)),
        'median_m': float(np.nanmedian(finite)),
        'max_m': float(np.nanmax(finite)),
        'range_m': float(np.nanmax(finite) - np.nanmin(finite)),
    }


def _safe_counts(series: Any) -> dict[str, int]:
    if series is None:
        return {}
    counts = pd.Series(series, dtype=object).astype(str).fillna('nan').value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


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
        'flatness_threshold': float(_OFFSET_FLATNESS_DOMINANT_FRACTION_THRESHOLD),
        'flatness_min_finite_count': int(_OFFSET_FLATNESS_MIN_FINITE_COUNT),
    }
    if vals.size == 0:
        return summary
    min_v = float(np.nanmin(vals))
    max_v = float(np.nanmax(vals))
    summary['min_m'] = min_v
    summary['max_m'] = max_v
    summary['range_m'] = float(max_v - min_v)
    rounded = np.round(vals, _OFFSET_ROUND_DECIMALS)
    unique, counts = np.unique(rounded, return_counts=True)
    if counts.size:
        idx = int(np.argmax(counts))
        dominant_count = int(counts[idx])
        fraction = float(dominant_count / vals.size)
        summary['dominant_rounded_value_m'] = float(unique[idx])
        summary['dominant_rounded_count'] = dominant_count
        summary['dominant_rounded_fraction'] = fraction
        summary['flatness_fail'] = bool(
            vals.size >= _OFFSET_FLATNESS_MIN_FINITE_COUNT
            and fraction >= _OFFSET_FLATNESS_DOMINANT_FRACTION_THRESHOLD
        )
    return summary


def _group_key(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in _PROFILE_GROUP_FIELDS if c in frame.columns]
    if not cols:
        return pd.Series(['all'] * len(frame), index=frame.index, dtype=object)
    return frame[cols].astype(str).agg('|'.join, axis=1)


def _classify_modeled_offset_group_flatness(
    *,
    wse_proxy: Any,
    modeled_offset: Any,
    raw_bed: Any,
    offset_sources: Any,
) -> dict[str, Any]:
    wse_summary = _dominant_flatness_summary(wse_proxy)
    offset_summary = _dominant_flatness_summary(modeled_offset)
    raw_bed_summary = _dominant_flatness_summary(raw_bed)
    raw_is_flat = bool(raw_bed_summary.get('flatness_fail'))
    wse_range = wse_summary.get('range_m')
    offset_range = offset_summary.get('range_m')
    wse_varies = bool(wse_range is not None and float(wse_range) >= _OFFSET_MEANINGFUL_RANGE_M)
    offset_varies = bool(offset_range is not None and float(offset_range) >= _OFFSET_MEANINGFUL_RANGE_M)
    source_counts = pd.Series(offset_sources, dtype=object).astype(str).value_counts().to_dict()
    source_counts = {str(k): int(v) for k, v in source_counts.items()}
    if not raw_is_flat:
        status = 'ok_raw_bed_not_flat'
        fail_reason = None
    elif wse_varies and offset_varies:
        status = 'fail_offset_cancels_wse_signal'
        fail_reason = 'modeled_offset_cancels_wse_signal'
    elif wse_varies and not offset_varies:
        status = 'fail_raw_bed_flat_despite_variable_wse_and_flat_offset'
        fail_reason = 'modeled_offset_raw_bed_formula_inconsistent'
    elif not wse_varies:
        status = 'context_raw_bed_flat_because_wse_flat_or_weak'
        fail_reason = None
    else:
        status = 'context_raw_bed_flat_unclassified'
        fail_reason = None
    return {
        'status': status,
        'fail_reason': fail_reason,
        'wse_proxy': wse_summary,
        'offset_modeled': offset_summary,
        'offset_raw_bed': raw_bed_summary,
        'offset_source_counts': source_counts,
    }


def _modeled_offset_construction_guard(out: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    if not {'wse_proxy_z_m', 'offset_modeled_m', 'offset_raw_bed_z_m'}.issubset(set(out.columns)):
        return [{
            'group_key': 'all',
            'record_count': int(len(out)),
            'status': 'fail_missing_formula_inputs',
            'fail_reason': 'modeled_offset_missing_wse_offset_raw_bed_inputs',
        }]
    checks: list[dict[str, Any]] = []
    keys = _group_key(out)
    for _, grp_idx in keys.groupby(keys).groups.items():
        grp = out.loc[list(grp_idx)].copy()
        check = _classify_modeled_offset_group_flatness(
            wse_proxy=grp['wse_proxy_z_m'],
            modeled_offset=grp['offset_modeled_m'],
            raw_bed=grp['offset_raw_bed_z_m'],
            offset_sources=grp.get('offset_source', pd.Series(dtype=object)),
        )
        check['group_key'] = str(_group_key(grp).iloc[0]) if len(grp) else 'unknown'
        check['record_count'] = int(len(grp))
        checks.append(check)
    return checks

def _write_modeled_offset_audit(
    ctx: RiverWorkflowContext,
    *,
    out: gpd.GeoDataFrame,
    build_result: Any,
    construction_checks: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    reports = _reports_dir(ctx)
    artifacts = _offset_artifacts_dir(ctx)
    artifacts.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts / ctx.paths.centerline_modeled_offset_points.name
    if artifact_path.exists():
        artifact_path.unlink()
    out.to_file(artifact_path, driver='GPKG')
    manifest_path = artifacts / 'modeled_offset_artifact_manifest.json'
    support_counts = _safe_counts(out.get('offset_support_class'))
    source_counts = _safe_counts(out.get('offset_source'))
    offset_summary = _numeric_summary(out.get('offset_modeled_m', []))
    distance_summary = _numeric_summary(out.get('distance_to_nearest_observed_offset_m', []))
    raw_bed_summary = _numeric_summary(out.get('offset_raw_bed_z_m', []))
    component_summaries = build_result.component_summaries or []
    audit_path = reports / 'OFFSET_MODEL_AUDIT.txt'
    lines = [
        'Modeled offset audit',
        '====================',
        '',
        'Stage: centerline_modeled_offset',
        'Construction:',
        '  centerline_wse_proxy_points.gpkg + accepted observed offsets',
        '      -> component-wise longitudinal offset model',
        '      -> offset_raw_bed_z_m = wse_proxy_z_m - offset_modeled_m',
        '',
        'Science invariant:',
        '  Observed offsets come only from accepted measured-bed/WSE anchors.',
        '  Modeled offsets are propagated along connected component stationing.',
        '  Low-support estimates are explicitly labeled, not treated as measured anchors.',
        '',
        f'Modeled offset artifact: {ctx.paths.centerline_modeled_offset_points}',
        f'Retained modeled offset artifact: {artifact_path}',
        '',
        f'Record count: {len(out)}',
        f'Finite modeled offset count: {offset_summary["finite_count"]}',
        f'Global observed median offset (m): {_jsonable(build_result.global_observed_median_offset_m)}',
        f'Global prior offset (m): {_jsonable(build_result.global_prior_offset_m)}',
        '',
        'Modeled offset summary:',
        json.dumps(_jsonable(offset_summary), indent=2, sort_keys=True),
        '',
        'Distance to observed-offset anchor summary:',
        json.dumps(_jsonable(distance_summary), indent=2, sort_keys=True),
        '',
        'Raw bed summary:',
        json.dumps(_jsonable(raw_bed_summary), indent=2, sort_keys=True),
        '',
        'Support class counts:',
        json.dumps(support_counts, indent=2, sort_keys=True),
        '',
        'Offset source counts:',
        json.dumps(source_counts, indent=2, sort_keys=True),
        '',
        'Component summary count:',
        str(len(component_summaries)),
        '',
        'Construction guard:',
        json.dumps(_jsonable({'group_count': len(construction_checks), 'groups': construction_checks}), indent=2, sort_keys=True),
    ]
    _write_text(audit_path, lines)
    _write_json(
        manifest_path,
        {
            'stage': 'centerline_modeled_offset',
            'retained_artifacts': {
                'centerline_modeled_offset_points': str(artifact_path),
                'offset_model_audit': str(audit_path),
            },
            'support_class_counts': support_counts,
            'offset_source_counts': source_counts,
            'modeled_offset_summary': offset_summary,
            'distance_to_nearest_observed_offset_summary': distance_summary,
            'component_summaries': component_summaries,
        },
    )
    return audit_path, artifact_path, manifest_path


def run_modeled_offset_stage(
    ctx: RiverWorkflowContext,
    wse_result: WSEProxyStageResult,
    observed_offset_result: ObservedOffsetStageResult,
) -> ModeledOffsetStageResult:
    wse = gpd.read_file(wse_result.centerline_wse_proxy_points_path).copy()
    observed = gpd.read_file(observed_offset_result.centerline_observed_offset_points_path).copy()
    build_result = build_modeled_offset_points(wse, observed)
    out = build_result.modeled_points
    construction_checks = _modeled_offset_construction_guard(out)
    flat_failures = [check for check in construction_checks if check.get('fail_reason')]
    if flat_failures:
        receipt_path = ctx.paths.modeled_offset_receipt.parent / 'modeled_offset_construction_guard_failure.json'
        _write_json(
            receipt_path,
            {
                'stage': 'centerline_modeled_offset',
                'failure_count': int(len(flat_failures)),
                'failures': flat_failures,
            },
        )
        first = flat_failures[0]
        raw_bed = first.get('offset_raw_bed', {})
        raise RuntimeError(
            'river_workflow_modeled_offset_construction_flatness_fail:'
            f"reason={first.get('fail_reason')}:"
            f"group={first.get('group_key')}:"
            f"dominant_value={raw_bed.get('dominant_rounded_value_m')}:"
            f"fraction={raw_bed.get('dominant_rounded_fraction')}:"
            f"summary={receipt_path}"
        )
    policy_applied = False
    policy_status = 'canonical_stage_no_export_policy'
    floor_fraction_before = None
    floor_fraction_after = None
    check_by_group = {str(check.get('group_key')): check for check in construction_checks}
    for group_key, grp_idx in _group_key(out).groupby(_group_key(out)).groups.items():
        check = check_by_group.get(str(group_key), {})
        raw_bed = check.get('offset_raw_bed', {}) if isinstance(check, dict) else {}
        wse_proxy = check.get('wse_proxy', {}) if isinstance(check, dict) else {}
        offset_modeled = check.get('offset_modeled', {}) if isinstance(check, dict) else {}
        out.loc[list(grp_idx), 'offset_group_guard_status'] = str(check.get('status', 'not_evaluated'))
        out.loc[list(grp_idx), 'offset_group_flatness_fail_reason'] = check.get('fail_reason')
        out.loc[list(grp_idx), 'offset_group_wse_range_m'] = wse_proxy.get('range_m')
        out.loc[list(grp_idx), 'offset_group_offset_range_m'] = offset_modeled.get('range_m')
        out.loc[list(grp_idx), 'offset_group_raw_bed_range_m'] = raw_bed.get('range_m')
        out.loc[list(grp_idx), 'offset_group_raw_bed_dominant_fraction'] = raw_bed.get('dominant_rounded_fraction')
    if ctx.paths.centerline_modeled_offset_points.exists():
        ctx.paths.centerline_modeled_offset_points.unlink()
    out.to_file(ctx.paths.centerline_modeled_offset_points, driver='GPKG')
    validate_modeled_offset_points_gpkg(ctx.paths.centerline_modeled_offset_points)
    offset_audit_path, retained_modeled_offset_path, offset_manifest_path = _write_modeled_offset_audit(
        ctx,
        out=out,
        build_result=build_result,
        construction_checks=construction_checks,
    )
    modeled = pd.to_numeric(out.get('offset_modeled_m'), errors='coerce').to_numpy(dtype=float) if 'offset_modeled_m' in out.columns else np.asarray([], dtype=float)
    finite_modeled = modeled[np.isfinite(modeled)]
    support_counts = out.get('offset_support_class', pd.Series(dtype=object)).astype(str).value_counts().to_dict() if 'offset_support_class' in out.columns else {}
    source_counts = out.get('offset_source', pd.Series(dtype=object)).astype(str).value_counts().to_dict() if 'offset_source' in out.columns else {}
    observed_support_count = int(source_counts.get('observed_offset_anchor', 0))
    if observed_support_count > 1:
        support_mode = 'observed_supported'
    elif observed_support_count == 1:
        support_mode = 'sparse_supported'
    else:
        support_mode = 'low_support_inferred'
    science_summary = {
        'modeled_offset_min_m': float(np.nanmin(finite_modeled)) if finite_modeled.size else None,
        'modeled_offset_max_m': float(np.nanmax(finite_modeled)) if finite_modeled.size else None,
        'modeled_offset_median_m': float(np.nanmedian(finite_modeled)) if finite_modeled.size else None,
        'finite_modeled_offset_count': int(build_result.finite_modeled_offset_count),
        'global_observed_median_offset_m': build_result.global_observed_median_offset_m,
        'global_prior_offset_m': build_result.global_prior_offset_m,
        'support_mode': support_mode,
        'support_class_counts': {str(k): int(v) for k, v in support_counts.items()},
        'offset_source_counts': {str(k): int(v) for k, v in source_counts.items()},
        'unsupported_reach_point_count': int(sum(v for k, v in support_counts.items() if 'unsupported' in str(k) or 'global_prior' in str(k))),
        'export_floor_policy_applied': bool(policy_applied),
        'export_floor_policy_status': str(policy_status),
        'offset_model_audit_path': str(offset_audit_path),
        'retained_modeled_offset_points_path': str(retained_modeled_offset_path),
        'offset_artifact_manifest_path': str(offset_manifest_path),
        'component_summaries': build_result.component_summaries or [],
        'construction_guard': {
            'status': 'passed',
            'group_count': int(len(construction_checks)),
            'flat_failure_count': int(sum(1 for check in construction_checks if check.get('fail_reason'))),
            'groups': construction_checks,
        },
    }
    return ModeledOffsetStageResult(
        centerline_modeled_offset_points_path=ctx.paths.centerline_modeled_offset_points,
        record_count=int(len(out)),
        finite_modeled_offset_count=build_result.finite_modeled_offset_count,
        global_observed_median_offset_m=build_result.global_observed_median_offset_m,
        global_prior_offset_m=build_result.global_prior_offset_m,
        export_floor_fraction_before_policy=floor_fraction_before,
        export_floor_fraction_after_policy=floor_fraction_after,
        export_floor_policy_applied=policy_applied,
        export_floor_policy_status=policy_status,
        science_summary=science_summary,
    )


__all__ = ['ModeledOffsetStageResult', 'run_modeled_offset_stage', '_classify_modeled_offset_group_flatness']
