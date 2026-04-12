from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from river_longitudinal_tendency import _distribution
from river_support_semantics import WEAK_SUPPORT_CLASSES, station_semantics

log = logging.getLogger(__name__)

CHANNEL_ROLES = {'thalweg', 'left_inner', 'right_inner'}
BANK_ROLES = {'left_bank', 'right_bank'}


def _bool_series(values, index) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.reindex(index)
    else:
        series = pd.Series(values, index=index)
    return series.map(lambda v: bool(v) if pd.notna(v) else False).astype(bool)


def _station_rebuild_eligibility_reason(row: pd.Series) -> str:
    regime = str(station_semantics(row).get('station_rebuild_regime', 'blocked_not_weak_support'))
    mapping = {
        'eligible': 'eligible',
        'blocked_true_measured_xs': 'true_measured_xs',
        'blocked_channel_authoritative': 'channel_authoritative',
        'blocked_channel_protected': 'channel_protected',
        'blocked_measured_or_missing_template': 'measured_or_missing_template',
        'blocked_not_weak_support': 'not_weak_support',
        'blocked_no_inner_rebuildable_nodes': 'no_inner_rebuildable_nodes',
    }
    return mapping.get(regime, regime)


def write_effectiveness_receipts(
    nodes: pd.DataFrame,
    *,
    river_dir: str | Path,
    xs_realism_summary: Optional[Dict[str, Any]] = None,
    primary_surface_rebuild_summary: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[Dict[str, str], Dict[str, Any]]:
    active_logger = logger or log
    river_dir = Path(river_dir)
    work = nodes.copy()
    outputs: Dict[str, str] = {}
    if work.empty:
        return outputs, {'available': False, 'station_count': 0, 'changed_node_count': 0}

    for col in [
        'station_m', 'primary_surface_rebuild_delta_m', 'primary_surface_rebuild_weight',
        'station_true_measured_xs_fraction', 'station_indirect_xs_fraction', 'station_residual_xs_fraction',
        'station_authoritative_fraction', 'station_authoritative_channel_fraction', 'station_authoritative_bank_fraction',
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors='coerce')
    if 'xs_support_template_class' in work.columns:
        work['xs_support_template_class'] = work['xs_support_template_class'].fillna('missing').astype(str)
    else:
        work['xs_support_template_class'] = 'missing'
    if 'xs_template_type' in work.columns:
        work['xs_template_type'] = work['xs_template_type'].fillna('missing').astype(str)
    else:
        work['xs_template_type'] = 'missing'
    if 'node_role' in work.columns:
        work['node_role'] = work['node_role'].fillna('unknown').astype(str)
    else:
        work['node_role'] = 'unknown'

    rebuild_applied = _bool_series(work.get('primary_surface_rebuild_applied', False), work.index)
    work['primary_surface_rebuild_applied'] = rebuild_applied
    work['primary_surface_rebuild_delta_m'] = pd.to_numeric(work.get('primary_surface_rebuild_delta_m', 0.0), errors='coerce').fillna(0.0)
    work['station_channel_protected'] = _bool_series(work.get('station_channel_protected', False), work.index)
    work['station_bank_protected'] = _bool_series(work.get('station_bank_protected', False), work.index)
    work['station_inner_rebuildable'] = _bool_series(work.get('station_inner_rebuildable', False), work.index)

    station_records = []
    for (component_id, station_m), group in work.groupby(['component_id', 'station_m'], sort=False):
        row0 = group.iloc[0]
        reason = _station_rebuild_eligibility_reason(row0)
        weak_support = str(row0.get('xs_support_template_class', 'missing') or 'missing') in WEAK_SUPPORT_CLASSES
        changed_nodes = int(np.count_nonzero(group['primary_surface_rebuild_applied'].to_numpy(dtype=bool)))
        channel_changed = int(np.count_nonzero(group['primary_surface_rebuild_applied'].to_numpy(dtype=bool) & group['node_role'].isin(CHANNEL_ROLES).to_numpy(dtype=bool)))
        bank_changed = int(np.count_nonzero(group['primary_surface_rebuild_applied'].to_numpy(dtype=bool) & group['node_role'].isin(BANK_ROLES).to_numpy(dtype=bool)))
        delta_abs = np.abs(pd.to_numeric(group['primary_surface_rebuild_delta_m'], errors='coerce').to_numpy(dtype=float))
        weights = pd.to_numeric(group.get('primary_surface_rebuild_weight', 0.0), errors='coerce').to_numpy(dtype=float)
        station_records.append({
            'component_id': str(component_id),
            'station_m': float(station_m),
            'station_provenance_class': str(row0.get('station_provenance_class', 'structural') or 'structural'),
            'xs_support_template_class': str(row0.get('xs_support_template_class', 'missing') or 'missing'),
            'xs_template_type': str(row0.get('xs_template_type', 'missing') or 'missing'),
            'station_true_measured_xs_fraction': float(row0.get('station_true_measured_xs_fraction', 0.0) or 0.0),
            'station_indirect_xs_fraction': float(row0.get('station_indirect_xs_fraction', 0.0) or 0.0),
            'station_residual_xs_fraction': float(row0.get('station_residual_xs_fraction', 0.0) or 0.0),
            'station_authoritative_fraction': float(row0.get('station_authoritative_fraction', 0.0) or 0.0),
            'station_authoritative_channel_fraction': float(row0.get('station_authoritative_channel_fraction', 0.0) or 0.0),
            'station_authoritative_bank_fraction': float(row0.get('station_authoritative_bank_fraction', 0.0) or 0.0),
            'station_bank_protected': bool(row0.get('station_bank_protected', False)),
            'station_channel_protected': bool(row0.get('station_channel_protected', False)),
            'station_inner_rebuildable': bool(row0.get('station_inner_rebuildable', False)),
            'rebuild_eligibility_reason': reason,
            'rebuild_eligible': bool(reason == 'eligible'),
            'weak_support_station': bool(weak_support),
            'station_changed': bool(changed_nodes > 0),
            'changed_node_count': changed_nodes,
            'changed_channel_node_count': channel_changed,
            'changed_bank_node_count': bank_changed,
            'station_delta_abs_max_m': float(np.nanmax(delta_abs)) if delta_abs.size else 0.0,
            'station_delta_abs_mean_m': float(np.nanmean(delta_abs)) if delta_abs.size else 0.0,
            'station_rebuild_weight_max': float(np.nanmax(weights)) if weights.size else 0.0,
            'station_rebuild_weight_mean': float(np.nanmean(weights)) if weights.size else 0.0,
        })

    station_df = pd.DataFrame(station_records)
    station_csv = river_dir / 'river_effectiveness_station_summary.csv'
    station_df.to_csv(station_csv, index=False)
    outputs['river_effectiveness_station_summary'] = str(station_csv)

    changed = work.loc[work['primary_surface_rebuild_applied']].copy()
    role_change_rows = []
    if not changed.empty:
        changed['delta_abs_m'] = np.abs(pd.to_numeric(changed['primary_surface_rebuild_delta_m'], errors='coerce'))
        for (role, support_class), sub in changed.groupby(['node_role', 'xs_support_template_class'], sort=False):
            role_change_rows.append({
                'node_role': str(role),
                'xs_support_template_class': str(support_class),
                'changed_node_count': int(len(sub)),
                'delta_abs_median_m': float(np.nanmedian(sub['delta_abs_m'].to_numpy(dtype=float))),
                'delta_abs_p95_m': float(np.nanpercentile(sub['delta_abs_m'].to_numpy(dtype=float), 95)),
                'rebuild_weight_median': float(np.nanmedian(pd.to_numeric(sub['primary_surface_rebuild_weight'], errors='coerce').to_numpy(dtype=float))),
            })
    role_change_df = pd.DataFrame(role_change_rows)
    role_change_csv = river_dir / 'river_effectiveness_node_changes.csv'
    role_change_df.to_csv(role_change_csv, index=False)
    outputs['river_effectiveness_node_changes'] = str(role_change_csv)

    weak_support_df = station_df.loc[station_df['weak_support_station']].copy()
    weak_support_eligible = weak_support_df.loc[weak_support_df['rebuild_eligible']]
    weak_support_changed = weak_support_df.loc[weak_support_df['station_changed']]
    suppression_counts = {str(k): int(v) for k, v in station_df.loc[~station_df['rebuild_eligible'], 'rebuild_eligibility_reason'].astype(str).value_counts(dropna=False).items()}

    summary: Dict[str, Any] = {
        'available': True,
        'station_count': int(len(station_df)),
        'weak_support_station_count': int(len(weak_support_df)),
        'weak_support_rebuild_eligible_station_count': int(len(weak_support_eligible)),
        'weak_support_changed_station_count': int(len(weak_support_changed)),
        'changed_station_count': int(np.count_nonzero(station_df['station_changed'].to_numpy(dtype=bool))),
        'changed_node_count': int(np.count_nonzero(work['primary_surface_rebuild_applied'].to_numpy(dtype=bool))),
        'changed_channel_node_count': int(np.count_nonzero(work['primary_surface_rebuild_applied'].to_numpy(dtype=bool) & work['node_role'].isin(CHANNEL_ROLES).to_numpy(dtype=bool))),
        'changed_bank_node_count': int(np.count_nonzero(work['primary_surface_rebuild_applied'].to_numpy(dtype=bool) & work['node_role'].isin(BANK_ROLES).to_numpy(dtype=bool))),
        'station_provenance_class_counts': {str(k): int(v) for k, v in station_df['station_provenance_class'].astype(str).value_counts(dropna=False).items()},
        'support_template_class_counts': {str(k): int(v) for k, v in station_df['xs_support_template_class'].astype(str).value_counts(dropna=False).items()},
        'weak_support_class_counts': {str(k): int(v) for k, v in weak_support_df['xs_support_template_class'].astype(str).value_counts(dropna=False).items()},
        'rebuild_eligibility_reason_counts': {str(k): int(v) for k, v in station_df['rebuild_eligibility_reason'].astype(str).value_counts(dropna=False).items()},
        'rebuild_suppression_reason_counts': suppression_counts,
        'station_delta_abs_m': _distribution(np.abs(pd.to_numeric(station_df['station_delta_abs_max_m'], errors='coerce').to_numpy(dtype=float))),
        'node_delta_abs_m': _distribution(np.abs(pd.to_numeric(work['primary_surface_rebuild_delta_m'], errors='coerce').to_numpy(dtype=float))),
        'rebuild_weight_summary': _distribution(pd.to_numeric(work['primary_surface_rebuild_weight'], errors='coerce').to_numpy(dtype=float)),
        'xs_realism_summary': xs_realism_summary or {},
        'primary_surface_rebuild_summary': primary_surface_rebuild_summary or {},
    }
    summary_path = river_dir / 'river_effectiveness_receipt.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    outputs['river_effectiveness_receipt'] = str(summary_path)

    active_logger.info(
        '[RIVER][EFFECT] Effectiveness receipts: weak_support=%d eligible=%d changed=%d changed_nodes=%d changed_channel=%d',
        int(summary['weak_support_station_count']),
        int(summary['weak_support_rebuild_eligible_station_count']),
        int(summary['weak_support_changed_station_count']),
        int(summary['changed_node_count']),
        int(summary['changed_channel_node_count']),
    )
    return outputs, summary


def write_science_effect_summary(
    *,
    river_dir: str | Path,
    effectiveness_summary: Optional[Dict[str, Any]] = None,
    primary_surface_rebuild_summary: Optional[Dict[str, Any]] = None,
    xs_realism_summary: Optional[Dict[str, Any]] = None,
    tendency_summary: Optional[Dict[str, Any]] = None,
    render_mode_counts: Optional[Dict[str, int]] = None,
    render_reason_counts: Optional[Dict[str, int]] = None,
    fast_render_component_ids: Optional[list[str]] = None,
    measured_xs_frame_receipt: Optional[Dict[str, Any]] = None,
    measured_xs_scaffold_receipt: Optional[Dict[str, Any]] = None,
    channel_surface_effect_summary: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[Dict[str, str], Dict[str, Any]]:
    active_logger = logger or log
    river_dir = Path(river_dir)
    eff = effectiveness_summary if isinstance(effectiveness_summary, dict) else {}
    rebuild = primary_surface_rebuild_summary if isinstance(primary_surface_rebuild_summary, dict) else {}
    realism = xs_realism_summary if isinstance(xs_realism_summary, dict) else {}
    tendency = tendency_summary if isinstance(tendency_summary, dict) else {}
    frame_receipt = measured_xs_frame_receipt if isinstance(measured_xs_frame_receipt, dict) else {}
    scaffold_receipt = measured_xs_scaffold_receipt if isinstance(measured_xs_scaffold_receipt, dict) else {}
    channel_effect = channel_surface_effect_summary if isinstance(channel_surface_effect_summary, dict) else {}

    summary: Dict[str, Any] = {
        'available': True,
        'weak_support_station_count': int(eff.get('weak_support_station_count', 0) or 0),
        'weak_support_rebuild_eligible_station_count': int(eff.get('weak_support_rebuild_eligible_station_count', 0) or 0),
        'weak_support_changed_station_count': int(eff.get('weak_support_changed_station_count', 0) or 0),
        'changed_node_count': int(eff.get('changed_node_count', 0) or 0),
        'changed_channel_node_count': int(eff.get('changed_channel_node_count', 0) or 0),
        'changed_bank_node_count': int(eff.get('changed_bank_node_count', 0) or 0),
        'rebuild_suppression_reason_counts': dict(eff.get('rebuild_suppression_reason_counts', {}) or {}),
        'support_template_class_counts': dict(eff.get('support_template_class_counts', {}) or {}),
        'weak_support_class_counts': dict(eff.get('weak_support_class_counts', {}) or {}),
        'longitudinal_adjusted_node_count': int(tendency.get('adjusted_node_count', 0) or 0),
        'longitudinal_support_regime_counts': dict(tendency.get('longitudinal_support_regime_counts', tendency.get('support_regime_counts', {})) or {}),
        'xs_realism_adjusted_node_count': int(realism.get('adjusted_node_count', 0) or 0),
        'xs_realism_template_type_counts': dict(realism.get('template_type_counts', {}) or {}),
        'primary_surface_rebuild_adjusted_node_count': int(rebuild.get('adjusted_node_count', 0) or 0),
        'primary_surface_rebuild_changed_channel_core_node_count': int(rebuild.get('changed_channel_core_node_count', 0) or 0),
        'primary_surface_rebuild_mode_counts': dict(rebuild.get('rebuild_mode_counts', {}) or {}),
        'measured_xs_qualified_station_count': int(frame_receipt.get('stations_with_true_measured_xs_qualified', 0) or 0),
        'measured_xs_inner_support_station_count': int(frame_receipt.get('stations_with_authoritative_xs_inner_support', 0) or 0),
        'measured_xs_bank_only_station_count': int(frame_receipt.get('stations_with_authoritative_bank_only_support', 0) or 0),
        'measured_xs_post_filter_node_count': int(scaffold_receipt.get('nodes_marked_true_measured_xs_post_filter', 0) or 0),
        'measured_xs_activation_contract_ok': bool(scaffold_receipt.get('measured_xs_activation_contract_ok', True)),
        'measured_xs_activation_from_bank_only_forbidden': bool(scaffold_receipt.get('measured_xs_activation_from_bank_only_forbidden', frame_receipt.get('measured_xs_activation_from_bank_only_forbidden', True))),
        'qualified_station_without_measured_nodes_count': int(scaffold_receipt.get('qualified_station_without_measured_nodes_count', 0) or 0),
        'render_mode_counts': {str(k): int(v) for k, v in (render_mode_counts or {}).items()},
        'render_reason_counts': {str(k): int(v) for k, v in (render_reason_counts or {}).items()},
        'fast_render_component_count': int(len(fast_render_component_ids or [])),
        'fast_render_component_ids': [str(v) for v in (fast_render_component_ids or [])],
        'thalweg_render_component_count': int(channel_effect.get('selected_for_thalweg_render_component_count', 0) or 0),
        'thalweg_render_candidate_pixel_count': int(channel_effect.get('candidate_pixel_count', 0) or 0),
        'thalweg_render_modified_pixel_count': int(channel_effect.get('final_changed_pixel_count', 0) or 0),
        'section_target_applied_component_count': int(channel_effect.get('section_target_applied_component_count', 0) or 0),
        'section_target_applied_pixel_count': int(channel_effect.get('section_target_applied_pixel_count', 0) or 0),
        'authoritative_transition_applied_component_count': int(channel_effect.get('authoritative_transition_applied_component_count', 0) or 0),
        'authoritative_transition_applied_pixel_count': int(channel_effect.get('authoritative_transition_applied_pixel_count', 0) or 0),
    }
    summary['science_engaged'] = bool(
        summary['weak_support_rebuild_eligible_station_count'] > 0
        or summary['changed_channel_node_count'] > 0
        or summary['primary_surface_rebuild_adjusted_node_count'] > 0
        or summary['xs_realism_adjusted_node_count'] > 0
        or summary['longitudinal_adjusted_node_count'] > 0
        or summary['measured_xs_post_filter_node_count'] > 0
        or summary['thalweg_render_modified_pixel_count'] > 0
        or summary['section_target_applied_pixel_count'] > 0
        or summary['authoritative_transition_applied_pixel_count'] > 0
    )

    path = river_dir / 'river_science_effect_summary.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    outputs = {'river_science_effect_summary': str(path)}

    active_logger.info(
        '[RIVER][SCIENCE] weak_support=%d eligible=%d changed=%d channel_nodes=%d longitudinal=%d xs_realism=%d rebuild=%d channel_core=%d measured_nodes=%d fast_render=%d',
        summary['weak_support_station_count'],
        summary['weak_support_rebuild_eligible_station_count'],
        summary['weak_support_changed_station_count'],
        summary['changed_channel_node_count'],
        summary['longitudinal_adjusted_node_count'],
        summary['xs_realism_adjusted_node_count'],
        summary['primary_surface_rebuild_adjusted_node_count'],
        summary['primary_surface_rebuild_changed_channel_core_node_count'],
        summary['measured_xs_post_filter_node_count'],
        summary['fast_render_component_count'],
    )
    active_logger.info(
        '[RIVER][SCIENCE] suppression=%s render_modes=%s',
        summary['rebuild_suppression_reason_counts'],
        summary['render_mode_counts'],
    )
    return outputs, summary


__all__ = ['write_effectiveness_receipts', 'write_science_effect_summary', 'WEAK_SUPPORT_CLASSES']
