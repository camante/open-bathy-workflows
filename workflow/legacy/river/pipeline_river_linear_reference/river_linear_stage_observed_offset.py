from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd

from pipeline.river_linear.river_linear_context import RiverLinearContext
from pipeline.river_linear.river_linear_stage_authoritative_bed import AuthoritativeBedStageResult
from pipeline.river_linear.river_linear_stage_wse_proxy import WSEProxyStageResult
from pipeline.river_linear.river_linear_validation import validate_observed_offset_points_gpkg
from pipeline.river_linear.river_linear_stage_helpers import validate_observed_offsets

_MODEL_OFFSET_FLOOR_M = 0.10
_SHALLOW_ANCHOR_MAX_WEIGHT = 0.35
_STRONG_ANCHOR_WEIGHT = 1.0


@dataclass(frozen=True)
class ObservedOffsetStageResult:
    centerline_observed_offset_points_path: Path
    record_count: int
    finite_offset_count: int
    science_summary: dict[str, Any] | None = None


def _reports_dir(ctx: RiverLinearContext) -> Path:
    return ctx.paths.root.parent / 'reports'


def _offset_artifacts_dir(ctx: RiverLinearContext) -> Path:
    return _reports_dir(ctx) / 'offset_artifacts'


def _safe_counts(series: pd.Series | None) -> dict[str, int]:
    if series is None:
        return {}
    counts = series.astype(str).fillna('nan').value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _numeric_summary(values: Any) -> dict[str, Any]:
    arr = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            'finite_count': 0,
            'min_m': None,
            'median_m': None,
            'max_m': None,
            'dynamic_range_m': None,
        }
    return {
        'finite_count': int(finite.size),
        'min_m': float(np.nanmin(finite)),
        'median_m': float(np.nanmedian(finite)),
        'max_m': float(np.nanmax(finite)),
        'dynamic_range_m': float(np.nanmax(finite) - np.nanmin(finite)),
    }


def _normalize_frame(gdf: gpd.GeoDataFrame, value_col: str, *, role: str) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out['point_id'] = out['point_id'].astype(str)
    out['station_m'] = pd.to_numeric(out['station_m'], errors='coerce')
    out[value_col] = pd.to_numeric(out[value_col], errors='coerce')
    optional_cols = (
        'component_id', 'levelpath_id', 'reach_id', 'source_reach_key',
        'station_downstream_m', 'wse_direction_method', 'wse_direction_confidence',
        'authoritative_support_present', 'authoritative_source_role',
        'authoritative_source_path', 'authoritative_support_mask_path',
        'canonical_system_id',
    )
    keep = [c for c in ('point_id', 'station_m', value_col, *optional_cols, 'geometry') if c in out.columns]
    out = gpd.GeoDataFrame(out[keep].copy(), geometry='geometry', crs=gdf.crs)
    out = out.dropna(subset=['point_id', 'station_m'])
    out = out.sort_values(['point_id', 'station_m'], kind='mergesort').drop_duplicates(subset=['point_id'], keep='first').reset_index(drop=True)
    if role == 'wse':
        rename = {
            'station_downstream_m': 'wse_station_downstream_m',
            'wse_direction_method': 'wse_direction_method',
            'wse_direction_confidence': 'wse_direction_confidence',
        }
    elif role == 'bed':
        rename = {
            'authoritative_support_present': 'bed_authoritative_support_present',
            'authoritative_source_role': 'bed_authoritative_source_role',
            'authoritative_source_path': 'bed_authoritative_source_path',
            'authoritative_support_mask_path': 'bed_authoritative_support_mask_path',
            'canonical_system_id': 'bed_canonical_system_id',
        }
    else:
        rename = {}
    return out.rename(columns={k: v for k, v in rename.items() if k in out.columns})


def _build_observed_offset_candidates(
    wse_points_gdf: gpd.GeoDataFrame,
    authoritative_bed_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, list[str]]:
    warnings: list[str] = []
    wse = _normalize_frame(wse_points_gdf, 'wse_proxy_z_m', role='wse')
    bed = _normalize_frame(authoritative_bed_gdf, 'authoritative_bed_z_m', role='bed')
    bed_cols = [
        c for c in (
            'point_id', 'authoritative_bed_z_m', 'bed_authoritative_support_present',
            'bed_authoritative_source_role', 'bed_authoritative_source_path',
            'bed_authoritative_support_mask_path', 'bed_canonical_system_id'
        ) if c in bed.columns
    ]
    merged = wse.merge(bed[bed_cols], on='point_id', how='inner', validate='one_to_one')
    merged['observed_offset_m'] = pd.to_numeric(merged['wse_proxy_z_m'], errors='coerce') - pd.to_numeric(merged['authoritative_bed_z_m'], errors='coerce')
    merged['offset_source'] = 'canonical_wse_minus_canonical_authoritative_bed'
    merged['offset_observed_m'] = merged['observed_offset_m']
    wse_finite = np.isfinite(pd.to_numeric(merged['wse_proxy_z_m'], errors='coerce').to_numpy(dtype=float))
    bed_finite = np.isfinite(pd.to_numeric(merged['authoritative_bed_z_m'], errors='coerce').to_numpy(dtype=float))
    off = pd.to_numeric(merged['observed_offset_m'], errors='coerce').to_numpy(dtype=float)
    off_finite = np.isfinite(off)
    positive = off > 0.0
    support_present = np.ones(len(merged), dtype=bool)
    if 'bed_authoritative_support_present' in merged.columns:
        support_present = pd.to_numeric(merged['bed_authoritative_support_present'], errors='coerce').fillna(0).to_numpy(dtype=float) > 0.0
    qc_pass = wse_finite & bed_finite & off_finite & positive & support_present
    # A positive but floor-level observed offset is not impossible, but it is too weak
    # to act as a full-strength longitudinal anchor.  Keep it visible as measured
    # evidence, but give the modeled-offset stage a reduced anchor weight so one
    # very shallow sample cannot pull long unsupported reaches to the model floor.
    floor_level = off_finite & np.isclose(off, _MODEL_OFFSET_FLOOR_M, rtol=0.0, atol=1.0e-6)
    shallow_low_confidence = qc_pass & floor_level
    strong_anchor = qc_pass & ~shallow_low_confidence
    reject_reason = np.full(len(merged), '', dtype=object)
    reject_reason[~wse_finite] = 'nonfinite_wse_proxy'
    reject_reason[wse_finite & ~bed_finite] = 'nonfinite_authoritative_bed'
    reject_reason[wse_finite & bed_finite & ~off_finite] = 'nonfinite_observed_offset'
    reject_reason[wse_finite & bed_finite & off_finite & ~positive] = 'nonpositive_observed_offset_bed_at_or_above_wse'
    reject_reason[wse_finite & bed_finite & off_finite & positive & ~support_present] = 'missing_authoritative_support_mask'
    merged['offset_qc_pass'] = qc_pass.astype(bool)
    merged['offset_reject_reason'] = reject_reason
    merged.loc[merged['offset_qc_pass'], 'offset_reject_reason'] = ''
    merged['offset_low_depth_warning'] = shallow_low_confidence.astype(bool)
    merged['offset_qc_class'] = np.where(
        ~qc_pass,
        'rejected_observed_offset',
        np.where(shallow_low_confidence, 'shallow_observed_low_confidence', 'measured_observed_anchor'),
    )
    merged['offset_anchor_role'] = np.where(
        ~qc_pass,
        'rejected',
        np.where(shallow_low_confidence, 'weak_anchor', 'strong_anchor'),
    )
    merged['offset_anchor_weight'] = np.where(
        ~qc_pass,
        0.0,
        np.where(shallow_low_confidence, _SHALLOW_ANCHOR_MAX_WEIGHT, _STRONG_ANCHOR_WEIGHT),
    )
    merged['offset_support_class'] = np.where(
        ~qc_pass,
        'rejected_observed_offset',
        np.where(shallow_low_confidence, 'shallow_observed_low_confidence', 'observed_offset_supported'),
    )
    merged['offset_confidence'] = np.where(
        ~qc_pass,
        'rejected',
        np.where(shallow_low_confidence, 'low_medium', 'high'),
    )
    merged['distance_to_nearest_observed_offset_m'] = np.where(merged['offset_qc_pass'], 0.0, np.nan)
    merged['upstream_observed_offset_distance_m'] = np.where(merged['offset_qc_pass'], 0.0, np.nan)
    merged['downstream_observed_offset_distance_m'] = np.where(merged['offset_qc_pass'], 0.0, np.nan)
    merged['offset_raw_m'] = merged['observed_offset_m']
    merged['offset_used_m'] = np.where(merged['offset_qc_pass'], merged['observed_offset_m'], np.nan)
    if len(merged) == 0:
        warnings.append('observed_offset_no_shared_canonical_wse_and_bed_support')
    accepted = int(np.count_nonzero(qc_pass))
    if accepted == 0 and len(merged):
        warnings.append('observed_offset_all_candidates_rejected_by_qc')
    keep_cols = [
        c for c in (
            'point_id', 'station_m', 'wse_station_downstream_m', 'component_id', 'levelpath_id',
            'reach_id', 'source_reach_key', 'wse_proxy_z_m', 'authoritative_bed_z_m',
            'observed_offset_m', 'offset_observed_m', 'offset_raw_m', 'offset_used_m',
            'offset_source', 'offset_support_class', 'offset_confidence', 'offset_qc_pass',
            'offset_qc_class', 'offset_anchor_role', 'offset_anchor_weight',
            'offset_low_depth_warning', 'offset_reject_reason',
            'distance_to_nearest_observed_offset_m',
            'upstream_observed_offset_distance_m', 'downstream_observed_offset_distance_m',
            'bed_authoritative_support_present', 'bed_authoritative_source_role',
            'bed_authoritative_source_path', 'bed_authoritative_support_mask_path',
            'bed_canonical_system_id', 'wse_direction_method', 'wse_direction_confidence', 'geometry'
        ) if c in merged.columns
    ]
    out = gpd.GeoDataFrame(merged[keep_cols].copy(), geometry='geometry', crs=wse.crs)
    sort_cols = [c for c in ('component_id', 'station_m', 'point_id') if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    return out, warnings


def _accepted_observed_offsets(candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if 'offset_qc_pass' not in candidates.columns:
        return candidates.copy()
    accepted = candidates.loc[candidates['offset_qc_pass'].astype(bool)].copy()
    if 'offset_support_class' not in accepted.columns:
        accepted['offset_support_class'] = 'observed_offset_supported'
    if 'offset_confidence' not in accepted.columns:
        accepted['offset_confidence'] = 'high'
    if 'offset_anchor_weight' not in accepted.columns:
        accepted['offset_anchor_weight'] = _STRONG_ANCHOR_WEIGHT
    if 'offset_qc_class' not in accepted.columns:
        accepted['offset_qc_class'] = 'measured_observed_anchor'
    if 'offset_anchor_role' not in accepted.columns:
        accepted['offset_anchor_role'] = 'strong_anchor'
    if 'offset_low_depth_warning' not in accepted.columns:
        accepted['offset_low_depth_warning'] = False
    return gpd.GeoDataFrame(accepted, geometry='geometry', crs=candidates.crs)


def _write_text(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path


def _write_offset_current_path_audit(
    ctx: RiverLinearContext,
    *,
    wse_path: Path,
    bed_path: Path,
    output_path: Path,
    candidates: gpd.GeoDataFrame,
    accepted: gpd.GeoDataFrame,
) -> Path:
    reports = _reports_dir(ctx)
    return _write_text(
        reports / 'OFFSET_CURRENT_PATH_AUDIT.txt',
        [
            'Observed offset current path audit',
            '==================================',
            '',
            'Stage: centerline_observed_offset',
            'Current construction:',
            '  centerline_wse_proxy_points.gpkg',
            '      + exact point_id match to',
            '  centerline_authoritative_bed_points.gpkg',
            '      -> observed_offset_m = wse_proxy_z_m - authoritative_bed_z_m',
            '',
            'Science invariant:',
            '  Observed offsets are only anchors where WSE and measured canonical authoritative bed both exist at the same centerline point.',
            '  Bank/WSE evidence is not treated as bed evidence.',
            '  Non-positive offsets are rejected as observed anchors because they place bed at or above WSE.',
            '',
            f'WSE input: {wse_path}',
            f'Authoritative bed input: {bed_path}',
            f'Observed offset output: {output_path}',
            '',
            f'Candidate matched points: {len(candidates)}',
            f'Accepted observed offsets: {len(accepted)}',
            f'Rejected observed offsets: {max(0, len(candidates) - len(accepted))}',
            '',
            'Support class counts:',
            json.dumps(_safe_counts(candidates.get('offset_support_class')), indent=2, sort_keys=True),
            '',
            'Reject reason counts:',
            json.dumps(_safe_counts(candidates.get('offset_reject_reason')), indent=2, sort_keys=True),
        ],
    )


def _write_offset_qc_audit(
    ctx: RiverLinearContext,
    *,
    candidates: gpd.GeoDataFrame,
    accepted: gpd.GeoDataFrame,
    warnings: list[str],
) -> Path:
    reports = _reports_dir(ctx)
    accepted_offsets = accepted['observed_offset_m'] if 'observed_offset_m' in accepted.columns else pd.Series(dtype=float)
    candidate_offsets = candidates['observed_offset_m'] if 'observed_offset_m' in candidates.columns else pd.Series(dtype=float)
    return _write_text(
        reports / 'OFFSET_OBSERVED_QC_AUDIT.txt',
        [
            'Observed offset QC audit',
            '========================',
            '',
            f'Candidate count: {len(candidates)}',
            f'Accepted count: {len(accepted)}',
            f'Rejected count: {max(0, len(candidates) - len(accepted))}',
            '',
            'Candidate offset summary:',
            json.dumps(_numeric_summary(candidate_offsets), indent=2, sort_keys=True),
            '',
            'Accepted offset summary:',
            json.dumps(_numeric_summary(accepted_offsets), indent=2, sort_keys=True),
            '',
            'Offset support class counts:',
            json.dumps(_safe_counts(candidates.get('offset_support_class')), indent=2, sort_keys=True),
            '',
            'Offset QC class counts:',
            json.dumps(_safe_counts(candidates.get('offset_qc_class')), indent=2, sort_keys=True),
            '',
            'Offset anchor role counts:',
            json.dumps(_safe_counts(candidates.get('offset_anchor_role')), indent=2, sort_keys=True),
            '',
            'Offset anchor weight summary:',
            json.dumps(_numeric_summary(candidates.get('offset_anchor_weight')), indent=2, sort_keys=True),
            '',
            'Offset confidence counts:',
            json.dumps(_safe_counts(candidates.get('offset_confidence')), indent=2, sort_keys=True),
            '',
            'Reject reason counts:',
            json.dumps(_safe_counts(candidates.get('offset_reject_reason')), indent=2, sort_keys=True),
            '',
            'Warnings:',
            json.dumps([str(w) for w in warnings], indent=2),
        ],
    )


def _retain_offset_artifacts(
    ctx: RiverLinearContext,
    *,
    candidates: gpd.GeoDataFrame,
    accepted_path: Path,
    current_path_audit: Path,
    qc_audit: Path,
) -> dict[str, str]:
    retained = _offset_artifacts_dir(ctx)
    retained.mkdir(parents=True, exist_ok=True)
    retained_accepted = retained / accepted_path.name
    shutil.copy2(accepted_path, retained_accepted)
    retained_candidates = retained / '08a_observed_offset_candidates_qc.gpkg'
    if retained_candidates.exists():
        retained_candidates.unlink()
    candidates.to_file(retained_candidates, driver='GPKG')
    manifest = {
        'stage': 'centerline_observed_offset',
        'artifacts': {
            'accepted_observed_offsets': str(retained_accepted),
            'qc_candidates': str(retained_candidates),
            'current_path_audit': str(current_path_audit),
            'observed_qc_audit': str(qc_audit),
        },
        'candidate_count': int(len(candidates)),
        'accepted_count': int(candidates.get('offset_qc_pass', pd.Series(dtype=bool)).astype(bool).sum()) if 'offset_qc_pass' in candidates.columns else int(len(candidates)),
        'support_class_counts': _safe_counts(candidates.get('offset_support_class')),
        'qc_class_counts': _safe_counts(candidates.get('offset_qc_class')),
        'anchor_role_counts': _safe_counts(candidates.get('offset_anchor_role')),
        'low_depth_warning_count': int(candidates.get('offset_low_depth_warning', pd.Series(dtype=bool)).astype(bool).sum()) if 'offset_low_depth_warning' in candidates.columns else 0,
        'reject_reason_counts': _safe_counts(candidates.get('offset_reject_reason')),
    }
    manifest_path = retained / 'offset_artifact_manifest.json'
    _write_json(manifest_path, manifest)
    return {str(k): str(v) for k, v in manifest['artifacts'].items()} | {'manifest': str(manifest_path)}


def run_observed_offset_stage(
    ctx: RiverLinearContext,
    wse_result: WSEProxyStageResult,
    authoritative_bed_result: AuthoritativeBedStageResult,
) -> ObservedOffsetStageResult:
    wse = gpd.read_file(wse_result.centerline_wse_proxy_points_path)
    bed = gpd.read_file(authoritative_bed_result.centerline_authoritative_bed_points_path)
    candidates, warnings = _build_observed_offset_candidates(wse, bed)
    gdf = _accepted_observed_offsets(candidates)
    validation = validate_observed_offsets(gdf)
    if not validation.get('valid'):
        current_path_audit = _write_offset_current_path_audit(
            ctx,
            wse_path=wse_result.centerline_wse_proxy_points_path,
            bed_path=authoritative_bed_result.centerline_authoritative_bed_points_path,
            output_path=ctx.paths.centerline_observed_offset_points,
            candidates=candidates,
            accepted=gdf,
        )
        qc_audit = _write_offset_qc_audit(ctx, candidates=candidates, accepted=gdf, warnings=warnings)
        raise RuntimeError(f'river_linear_observed_offset_invalid:{validation}:warnings={warnings}:audits={current_path_audit},{qc_audit}')
    if ctx.paths.centerline_observed_offset_points.exists():
        ctx.paths.centerline_observed_offset_points.unlink()
    gdf.to_file(ctx.paths.centerline_observed_offset_points, driver='GPKG')
    validate_observed_offset_points_gpkg(ctx.paths.centerline_observed_offset_points)
    current_path_audit = _write_offset_current_path_audit(
        ctx,
        wse_path=wse_result.centerline_wse_proxy_points_path,
        bed_path=authoritative_bed_result.centerline_authoritative_bed_points_path,
        output_path=ctx.paths.centerline_observed_offset_points,
        candidates=candidates,
        accepted=gdf,
    )
    qc_audit = _write_offset_qc_audit(ctx, candidates=candidates, accepted=gdf, warnings=warnings)
    retained_artifacts = _retain_offset_artifacts(
        ctx,
        candidates=candidates,
        accepted_path=ctx.paths.centerline_observed_offset_points,
        current_path_audit=current_path_audit,
        qc_audit=qc_audit,
    )
    offsets = pd.to_numeric(gdf.get('observed_offset_m'), errors='coerce').to_numpy(dtype=float) if 'observed_offset_m' in gdf.columns else np.asarray([], dtype=float)
    finite_offsets = offsets[np.isfinite(offsets)]
    science_summary = {
        'authoritative_bed_sample_count': int(len(bed)),
        'observed_offset_candidate_count': int(len(candidates)),
        'valid_observed_offset_count': int(validation.get('finite_observed_offset_count', 0)),
        'observed_offset_min_m': float(np.nanmin(finite_offsets)) if finite_offsets.size else None,
        'observed_offset_max_m': float(np.nanmax(finite_offsets)) if finite_offsets.size else None,
        'observed_offset_median_m': float(np.nanmedian(finite_offsets)) if finite_offsets.size else None,
        'rejected_sample_count': int(max(0, len(candidates) - len(gdf))),
        'warnings': [str(w) for w in warnings],
        'offset_source': 'wse_proxy_z_m_minus_authoritative_bed_z_m',
        'bank_samples_treated_as_authoritative_bed': False,
        'offset_qc_policy': 'finite_wse_and_bed_positive_wse_minus_bed_and_authoritative_support_with_floor_level_offsets_downweighted',
        'support_class_counts': _safe_counts(candidates.get('offset_support_class')),
        'qc_class_counts': _safe_counts(candidates.get('offset_qc_class')),
        'anchor_role_counts': _safe_counts(candidates.get('offset_anchor_role')),
        'low_depth_warning_count': int(candidates.get('offset_low_depth_warning', pd.Series(dtype=bool)).astype(bool).sum()) if 'offset_low_depth_warning' in candidates.columns else 0,
        'qc_class_counts': _safe_counts(candidates.get('offset_qc_class')),
        'anchor_role_counts': _safe_counts(candidates.get('offset_anchor_role')),
        'low_depth_warning_count': int(candidates.get('offset_low_depth_warning', pd.Series(dtype=bool)).astype(bool).sum()) if 'offset_low_depth_warning' in candidates.columns else 0,
        'reject_reason_counts': _safe_counts(candidates.get('offset_reject_reason')),
        'retained_offset_artifacts': retained_artifacts,
    }
    return ObservedOffsetStageResult(
        centerline_observed_offset_points_path=ctx.paths.centerline_observed_offset_points,
        record_count=int(len(gdf)),
        finite_offset_count=int(validation.get('finite_observed_offset_count', 0)),
        science_summary=science_summary,
    )


__all__ = ['ObservedOffsetStageResult', 'run_observed_offset_stage']
