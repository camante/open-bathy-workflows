from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

REPORTS_DIRNAME = 'reports'
README_NAME = 'README_FIRST.txt'
MANIFEST_NAME = 'reports_manifest.json'

_TOP_OUTPUT_KEYS = [
    ('bathy_report', 'bathy_report.json', 'bathy_report.json'),
    ('io_manifest', 'io_manifest.json', 'io_manifest.json'),
    ('workflow_input_output_trace', 'WORKFLOW_INPUT_OUTPUT_TRACE.txt', 'WORKFLOW_INPUT_OUTPUT_TRACE.txt'),
    ('workflow_stage_trace_json', 'workflow_stage_trace.json', 'workflow_stage_trace.json'),
    ('workflow_stage_trace_jsonl', 'workflow_stage_trace_lines.jsonl', 'workflow_stage_trace_lines.jsonl'),
    ('workflow_file_graph_json', 'workflow_file_graph.json', 'workflow_file_graph.json'),
    ('workflow_explanation_report', 'WORKFLOW_EXPLANATION_REPORT.txt', 'WORKFLOW_EXPLANATION_REPORT.txt'),
    ('workflow_accuracy_anomalies_json', 'workflow_accuracy_anomalies.json', 'workflow_accuracy_anomalies.json'),
    ('workflow_first_bad_artifact_summary_json', 'workflow_first_bad_artifact_summary.json', 'workflow_first_bad_artifact_summary.json'),
    ('workflow_run_diagnosis_summary_json', 'workflow_run_diagnosis_summary.json', 'workflow_run_diagnosis_summary.json'),
    ('workflow_run_diagnosis_summary_txt', 'WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt', 'WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt'),
    ('benchmark_river_active_evaluation_summary_json', 'benchmark_river_active_evaluation_summary.json', 'benchmark_river_active_evaluation_summary.json'),
    ('benchmark_river_mode_summary_json', 'benchmark_river_mode_summary.json', 'benchmark_river_mode_summary.json'),
    ('benchmark_river_withheld_support_receipt_json', 'benchmark_river_withheld_support_receipt.json', 'benchmark_river_withheld_support_receipt.json'),
    ('benchmark_river_receipt_triage_summary_json', 'benchmark_river_receipt_triage_summary.json', 'benchmark_river_receipt_triage_summary.json'),
    ('benchmark_river_primary_focus_summary_json', 'benchmark_river_primary_focus_summary.json', 'benchmark_river_primary_focus_summary.json'),
    ('river_primary_guidance_summary', 'river_primary_guidance_summary.json', 'river_primary_guidance_summary.json'),
    ('river_primary_surface_contract', 'river_primary_surface_contract.json', 'river_primary_surface_contract.json'),
]

_RIVER_OUTPUT_KEYS = [
    ('channel_surface_effect_summary', 'river_effect_summary.json'),
    ('channel_surface_authoritative_transition_summary', 'river_transition_summary.json'),
    ('backbone_smoothing_summary', 'river_backbone_smoothing_summary.json'),
    ('centerline_width_propagation_summary', 'river_centerline_width_propagation_summary.json'),
    ('channel_surface_role_agreement_summary', 'river_role_agreement_summary.json'),
    ('channel_surface_section_target_agreement_summary', 'river_section_target_agreement_summary.json'),
]

_PRIMARY_REPORT_LABELS = {
    "workflow_run_diagnosis_summary_txt",
    "workflow_run_diagnosis_summary_json",
    "river_active_summary",
}

_DEPRECATED_REPORT_LABELS = {
    "workflow_file_graph_json",
}

_PRIORITY_FILES = [
    ('workflow_run_diagnosis_summary_txt', 'WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt', 'Start here for the shortest run-level diagnosis and first-bad-artifact summary.'),
    ('workflow_run_diagnosis_summary_json', 'workflow_run_diagnosis_summary.json', 'Structured diagnosis payload with active branch, anomalies, and recommended fix target.'),
    ('river_active_summary', 'river_active_summary.json', 'One active river story for the run: runtime contract state, river-shape state, evaluation mode, dominant issue, and next action.'),
    ('benchmark_river_active_evaluation_summary_json', 'benchmark_river_active_evaluation_summary.json', 'River-evaluation drilldown: benchmark mode, decision basis, dominant issue, and next action.'),
    ('benchmark_river_withheld_support_receipt_json', 'benchmark_river_withheld_support_receipt.json', 'River-evaluation drilldown for the deterministic withheld-support benchmark plan and receipt.'),
    ('benchmark_river_mode_summary_json', 'benchmark_river_mode_summary.json', 'Benchmark-mode drilldown receipt with holdout coverage and support-aware routing context.'),
    ('benchmark_river_receipt_triage_summary_json', 'benchmark_river_receipt_triage_summary.json', 'Diagnostic drilldown for whether the dominant remaining river problem is thalweg, inner-shape width propagation, or bank behavior.'),
    ('benchmark_river_primary_focus_summary_json', 'benchmark_river_primary_focus_summary.json', 'Diagnostic drilldown for the condensed river benchmark focus and suggested next action.'),
    ('river_active_runtime_summary', 'river_active_runtime_summary.json', 'River runtime/product drilldown: primary guidance product, contract state, degraded safeguards, and final-route role labels.'),
    ('river_primary_guidance_summary', 'river_primary_guidance_summary.json', 'Diagnostic runtime drilldown for the primary river guidance product and degraded continuity/backstop safeguards.'),
    ('river_primary_surface_contract', 'river_primary_surface_contract.json', 'Diagnostic runtime drilldown validating the primary river guidance surface contract.'),
    ('river_active_shape_summary', 'river_active_shape_summary.json', 'River-shape drilldown: backbone movement, width propagation, dominant lateral issue, and next action.'),
    ('river_backbone_smoothing_summary.json', 'river_backbone_smoothing_summary.json', 'River-shape drilldown showing whether backbone smoothing actually moved weak-support stations and why not when it stayed inert.'),
    ('river_centerline_width_propagation_summary.json', 'river_centerline_width_propagation_summary.json', 'Shows whether the backbone influence spread across channel width in weak-support reaches.'),
    ('river_role_agreement_summary.json', 'river_role_agreement_summary.json', 'Role-by-role agreement summary for thalweg, inner-shape, and bank-edge behavior.'),
    ('river_section_target_agreement_summary.json', 'river_section_target_agreement_summary.json', 'Section-target agreement summary for lateral role classes when available.'),
    ('river_effect_summary.json', 'river_effect_summary.json', 'Channel-surface effect receipt including render-mode engagement and applied river influence counts.'),
    ('river_transition_summary.json', 'river_transition_summary.json', 'Authoritative-transition receipt showing whether taper near authoritative support actually engaged.'),
    ('workflow_first_bad_artifact_summary_json', 'workflow_first_bad_artifact_summary.json', 'Structured first-bad-artifact payload for exact stage/module targeting.'),
    ('workflow_accuracy_anomalies_json', 'workflow_accuracy_anomalies.json', 'All connected anomalies, including inert or missing stage effects.'),
    ('WORKFLOW_EXPLANATION_REPORT.txt', 'WORKFLOW_EXPLANATION_REPORT.txt', 'Narrative stage explanation when the structured summaries need more context.'),
    ('WORKFLOW_INPUT_OUTPUT_TRACE.txt', 'WORKFLOW_INPUT_OUTPUT_TRACE.txt', 'Full input/output lineage trace after the higher-priority diagnosis receipts.'),
]


def _resolve_path(value: Any, *, base_dir: Path) -> Path | None:
    if not value:
        return None
    try:
        p = Path(str(value))
    except Exception:
        return None
    if not p.is_absolute():
        p = base_dir / p
    return p


def _artifact_role(label: str) -> str:
    if label in _PRIMARY_REPORT_LABELS:
        return 'primary'
    if label in _DEPRECATED_REPORT_LABELS:
        return 'deprecated'
    return 'diagnostic_only'


def _copy_or_record(src: Path | None, dst: Path, *, copied: list[dict[str, Any]], missing: list[dict[str, Any]], label: str) -> None:
    if src is None:
        missing.append({'label': label, 'reason': 'not_reported', 'artifact_role': _artifact_role(label)})
        return
    if not src.exists():
        missing.append({'label': label, 'reason': 'missing', 'source': str(src), 'artifact_role': _artifact_role(label)})
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
        action = 'copied'
    else:
        action = 'in_place'
    copied.append({'label': label, 'source': str(src), 'dest': str(dst), 'action': action, 'artifact_role': _artifact_role(label)})


def _status_for_dest(*, label: str, dest_name: str, copied_map: dict[str, dict[str, Any]], missing_map: dict[str, dict[str, Any]], reports_dir: Path) -> dict[str, Any]:
    dest_path = str(reports_dir / dest_name)
    copied = copied_map.get(dest_path)
    if copied is not None:
        return {
            'present': True,
            'status': copied.get('action') or 'copied',
            'artifact_role': copied.get('artifact_role') or _artifact_role(label),
            'dest': dest_path,
            'source': copied.get('source'),
        }
    missing = missing_map.get(label)
    if missing is None:
        return {
            'present': False,
            'status': 'not_reported',
            'dest': dest_path,
            'reason': 'not_reported',
        }
    payload = {
        'present': False,
        'status': str(missing.get('reason') or 'missing'),
        'dest': dest_path,
        'reason': str(missing.get('reason') or 'missing'),
        'artifact_role': missing.get('artifact_role') or _artifact_role(label),
    }
    if missing.get('source'):
        payload['source'] = missing.get('source')
    return payload




def _reconcile_manifest_with_reports_dir(*, manifest: dict[str, Any]) -> dict[str, Any]:
    copied = manifest.get('copied', []) if isinstance(manifest.get('copied', []), list) else []
    missing = manifest.get('missing', []) if isinstance(manifest.get('missing', []), list) else []
    reconciled_copied: list[dict[str, Any]] = []
    for entry in copied:
        if not isinstance(entry, dict):
            continue
        dest = entry.get('dest')
        if not dest or not Path(str(dest)).exists():
            missing.append({
                'label': str(entry.get('label') or ''),
                'reason': 'manifest_dest_missing',
                'artifact_role': entry.get('artifact_role') or _artifact_role(str(entry.get('label') or '')),
                'source': entry.get('source'),
                'dest': dest,
            })
            continue
        reconciled_copied.append(entry)
    manifest['copied'] = reconciled_copied
    manifest['missing'] = missing
    return manifest

def _load_json_dict(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _suggest_river_shape_action(*, dominant_issue: str | None) -> str | None:
    mapping = {
        'backbone_smoothing_inert': 'strengthen_backbone_smoothing',
        'backbone_width_not_propagating': 'tighten_width_propagation_from_backbone',
        'thalweg_fit': 'tighten_backbone_or_thalweg_alignment',
        'inner_shape_width_propagation': 'tighten_width_propagation_from_backbone',
        'bank_vs_inner_shape': 'reduce_bank_driven_interior_shaping',
        'role_agreement_unavailable': 'fix_canonical_target_comparison_prep',
        'section_target_agreement_unavailable': 'fix_canonical_target_comparison_prep',
        'river_shape_contract_ok': 'continue_with_targeted_science_validation',
    }
    return mapping.get(str(dominant_issue)) if dominant_issue is not None else None


def _build_active_river_shape_summary(*, out_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    river_outputs = (((report or {}).get('river') or {}).get('outputs') or {})
    backbone_path = _resolve_path(river_outputs.get('backbone_smoothing_summary'), base_dir=out_dir)
    width_path = _resolve_path(river_outputs.get('centerline_width_propagation_summary'), base_dir=out_dir)
    role_path = _resolve_path(river_outputs.get('channel_surface_role_agreement_summary'), base_dir=out_dir)
    section_path = _resolve_path(river_outputs.get('channel_surface_section_target_agreement_summary'), base_dir=out_dir)

    backbone = _load_json_dict(backbone_path) or {}
    width = _load_json_dict(width_path) or {}
    role = _load_json_dict(role_path) or {}
    section = _load_json_dict(section_path) or {}

    adjusted_station_count = int(backbone.get('adjusted_station_count') or 0)
    backbone_moved = adjusted_station_count > 0
    width_station_count = int(width.get('backbone_led_station_count') or 0)
    bank_margin_damping_station_count = int(width.get('bank_margin_damping_station_count') or 0)
    role_available = bool(role.get('available'))
    section_available = bool(section.get('available'))
    lateral_failure_mode = role.get('lateral_failure_mode') if isinstance(role.get('lateral_failure_mode'), str) else None
    weakest_role = role.get('weakest_role') if isinstance(role.get('weakest_role'), str) else None
    weakest_role_class = section.get('weakest_role_class') if isinstance(section.get('weakest_role_class'), str) else None

    dominant_issue: str
    if backbone_path is not None and backbone and not backbone_moved:
        dominant_issue = 'backbone_smoothing_inert'
    elif width_path is not None and width.get('available') and width_station_count <= 0:
        dominant_issue = 'backbone_width_not_propagating'
    elif role_path is not None and role and role_available and lateral_failure_mode:
        dominant_issue = lateral_failure_mode
    elif role_path is not None and role and not role_available:
        dominant_issue = 'role_agreement_unavailable'
    elif section_path is not None and section and not section_available:
        dominant_issue = 'section_target_agreement_unavailable'
    else:
        dominant_issue = 'river_shape_contract_ok'

    primary_shape_signal = (
        'backbone_movement' if backbone_moved else
        'role_agreement_lateral_issue' if role_available and lateral_failure_mode else
        'width_propagation' if bool(width.get('available')) else
        'receipt_availability_only'
    )

    summary = {
        'available': any(path is not None for path in (backbone_path, width_path, role_path, section_path)),
        'artifact_role': 'diagnostic_only',
        'active_river_shape_receipt': 'river_active_shape_summary.json',
        'primary_shape_signal': primary_shape_signal,
        'dominant_remaining_issue': dominant_issue,
        'weakest_lateral_role': weakest_role,
        'weakest_section_role_class': weakest_role_class,
        'lateral_failure_mode': lateral_failure_mode,
        'backbone_smoothing_available': bool(backbone),
        'backbone_adjusted_station_count': adjusted_station_count,
        'backbone_moved': backbone_moved,
        'backbone_reason': backbone.get('reason'),
        'width_propagation_available': bool(width),
        'backbone_led_station_count': width_station_count,
        'bank_margin_damping_station_count': bank_margin_damping_station_count,
        'role_agreement_available': role_available if role else False,
        'role_agreement_reason': role.get('reason') if role else None,
        'section_target_agreement_available': section_available if section else False,
        'section_target_agreement_reason': section.get('reason') if section else None,
        'suggested_next_action': _suggest_river_shape_action(dominant_issue=dominant_issue),
        'diagnostic_receipt_roles': {
            'river_backbone_smoothing_summary.json': 'diagnostic_only',
            'river_centerline_width_propagation_summary.json': 'diagnostic_only',
            'river_role_agreement_summary.json': 'diagnostic_only',
            'river_section_target_agreement_summary.json': 'diagnostic_only',
        },
        'diagnostic_receipt_paths': {
            'river_backbone_smoothing_summary.json': str(backbone_path) if backbone_path else None,
            'river_centerline_width_propagation_summary.json': str(width_path) if width_path else None,
            'river_role_agreement_summary.json': str(role_path) if role_path else None,
            'river_section_target_agreement_summary.json': str(section_path) if section_path else None,
        },
    }
    return summary


def _load_json_path(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _build_active_river_runtime_summary(*, out_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    top = report.get('outputs', {}) if isinstance(report, dict) else {}
    auth_outputs = (((report or {}).get('authoritative_base') or {}).get('outputs') or {}) if isinstance((((report or {}).get('authoritative_base') or {}).get('outputs') or {}), dict) else {}
    final_route = report.get('final_dem_route', {}) if isinstance(report.get('final_dem_route', {}), dict) else {}
    stage_receipts = final_route.get('stage_receipts', {}) if isinstance(final_route.get('stage_receipts', {}), dict) else {}

    guidance_path = _resolve_path(top.get('river_primary_guidance_summary') or auth_outputs.get('river_primary_guidance_summary'), base_dir=out_dir)
    contract_path = _resolve_path(top.get('river_primary_surface_contract') or auth_outputs.get('river_primary_surface_contract'), base_dir=out_dir)
    outputs_receipt_path = _resolve_path(stage_receipts.get('outputs'), base_dir=out_dir)
    final_route_receipt_path = _resolve_path(top.get('final_route_receipt') or auth_outputs.get('final_route_receipt'), base_dir=out_dir)
    if final_route_receipt_path is None:
        final_route_receipt_path = _resolve_path(final_route.get('final_route_receipt'), base_dir=out_dir)

    guidance = _load_json_path(guidance_path)
    contract = _load_json_path(contract_path)
    outputs_receipt = _load_json_path(outputs_receipt_path)
    final_route_receipt = _load_json_path(final_route_receipt_path)

    outputs_roles = outputs_receipt.get('artifact_roles', {}) if isinstance(outputs_receipt.get('artifact_roles', {}), dict) else {}
    final_route_roles = final_route_receipt.get('artifact_roles', {}) if isinstance(final_route_receipt.get('artifact_roles', {}), dict) else {}
    primary_artifact_role_labels = sorted(
        name for name, role in {**outputs_roles, **final_route_roles}.items() if role == 'primary'
    )
    diagnostic_artifact_role_labels = sorted(
        name for name, role in {**outputs_roles, **final_route_roles}.items() if role == 'diagnostic_only'
    )
    contract_ok = bool(contract.get('ok', False)) if contract else False
    degraded_mode_active = bool(guidance.get('degraded_mode_active', False)) if guidance else False
    continuity_safeguard_used = bool(guidance.get('continuity_safeguard_used', False)) if guidance else False
    primary_surface_contract_ok = bool(guidance.get('primary_surface_contract_ok', contract_ok)) if guidance or contract else False

    if not guidance:
        dominant_runtime_issue = 'missing_primary_guidance_summary'
    elif not contract:
        dominant_runtime_issue = 'missing_primary_surface_contract'
    elif not primary_surface_contract_ok:
        dominant_runtime_issue = 'primary_surface_contract_failed'
    elif degraded_mode_active:
        dominant_runtime_issue = 'degraded_runtime_guidance_active'
    elif continuity_safeguard_used:
        dominant_runtime_issue = 'continuity_safeguard_active'
    else:
        dominant_runtime_issue = 'river_runtime_contract_ok'

    if dominant_runtime_issue == 'missing_primary_guidance_summary':
        suggested_next_action = 'check_primary_guidance_summary_generation'
    elif dominant_runtime_issue == 'missing_primary_surface_contract':
        suggested_next_action = 'check_primary_surface_contract_write'
    elif dominant_runtime_issue == 'primary_surface_contract_failed':
        suggested_next_action = 'fix_primary_surface_contract_before_conditioning'
    elif dominant_runtime_issue == 'degraded_runtime_guidance_active':
        suggested_next_action = 'inspect_degraded_guidance_reasons_and_reduce_backstop_use'
    elif dominant_runtime_issue == 'continuity_safeguard_active':
        suggested_next_action = 'inspect_continuity_safeguard_and_restore_primary_surface_coverage'
    else:
        suggested_next_action = 'use_runtime_drilldowns_only_if_shape_or_benchmark_receipts_disagree'

    summary = {
        'available': any(path is not None for path in (guidance_path, contract_path, outputs_receipt_path, final_route_receipt_path)),
        'artifact_role': 'diagnostic_only',
        'active_river_runtime_receipt': 'river_active_runtime_summary.json',
        'active_product_name': str(guidance.get('active_product_name', 'river_primary_surface')) if guidance else 'river_primary_surface',
        'active_product_role': 'primary',
        'primary_builder_mode': guidance.get('primary_builder_mode') if guidance else None,
        'dominant_primary_source_class': guidance.get('dominant_primary_source_class') if guidance else None,
        'primary_surface_contract_ok': primary_surface_contract_ok,
        'contract_failure_count': int(len(contract.get('failures', []))) if isinstance(contract.get('failures', []), list) else 0,
        'degraded_mode_active': degraded_mode_active,
        'degraded_mode_reasons': guidance.get('degraded_mode_reasons', []) if guidance else [],
        'continuity_safeguard_used': continuity_safeguard_used,
        'continuity_safeguard_labels': guidance.get('continuity_safeguard_labels', []) if guidance else [],
        'dominant_runtime_issue': dominant_runtime_issue,
        'suggested_next_action': suggested_next_action,
        'primary_artifact_role_labels': primary_artifact_role_labels,
        'diagnostic_artifact_role_labels': diagnostic_artifact_role_labels,
        'runtime_receipt_paths': {
            'river_primary_guidance_summary.json': str(guidance_path) if guidance_path else None,
            'river_primary_surface_contract.json': str(contract_path) if contract_path else None,
            'final_route_outputs_receipt.json': str(outputs_receipt_path) if outputs_receipt_path else None,
            'final_route_receipt.json': str(final_route_receipt_path) if final_route_receipt_path else None,
        },
        'diagnostic_receipt_roles': {
            'river_primary_guidance_summary.json': 'diagnostic_only',
            'river_primary_surface_contract.json': 'diagnostic_only',
            'final_route_outputs_receipt.json': 'diagnostic_only',
            'final_route_receipt.json': 'diagnostic_only',
        },
    }
    return summary


def _build_active_river_summary(*, reports_dir: Path) -> dict[str, Any]:
    runtime_path = reports_dir / 'river_active_runtime_summary.json'
    shape_path = reports_dir / 'river_active_shape_summary.json'
    evaluation_path = reports_dir / 'benchmark_river_active_evaluation_summary.json'
    withheld_path = reports_dir / 'benchmark_river_withheld_support_receipt.json'

    runtime = _load_json_dict(runtime_path) or {}
    shape = _load_json_dict(shape_path) or {}
    evaluation = _load_json_dict(evaluation_path) or {}
    withheld = _load_json_dict(withheld_path) or {}

    runtime_issue = runtime.get('dominant_runtime_issue') if isinstance(runtime.get('dominant_runtime_issue'), str) else None
    shape_issue = shape.get('dominant_remaining_issue') if isinstance(shape.get('dominant_remaining_issue'), str) else None
    evaluation_issue = evaluation.get('dominant_remaining_issue') if isinstance(evaluation.get('dominant_remaining_issue'), str) else None
    runtime_ok = runtime_issue in {None, 'river_runtime_contract_ok'}
    shape_ok = shape_issue in {None, 'river_shape_contract_ok'}
    evaluation_constraint = bool(evaluation.get('hard_problem_holdout_blind', False)) or bool(evaluation.get('river_specific_withheld_support_benchmark_recommended', False))
    evaluation_ok = evaluation_issue in {None, '', 'river_evaluation_contract_ok'} and not evaluation_constraint

    if not runtime_ok:
        primary_signal = 'runtime_contract'
        dominant_issue = runtime_issue
        suggested_next_action = runtime.get('suggested_next_action')
    elif not shape_ok:
        primary_signal = 'river_shape'
        dominant_issue = shape_issue
        suggested_next_action = shape.get('suggested_next_action')
    elif not evaluation_ok:
        primary_signal = 'evaluation'
        dominant_issue = evaluation_issue or ('hard_problem_holdout_blind' if evaluation_constraint else 'evaluation_drilldown_required')
        suggested_next_action = evaluation.get('suggested_next_action')
    else:
        primary_signal = 'river_contract_ok'
        dominant_issue = 'river_active_contract_ok'
        suggested_next_action = 'continue_with_targeted_validation_or_science_tuning'

    summary = {
        'available': any(path.exists() for path in (runtime_path, shape_path, evaluation_path)),
        'artifact_role': 'primary',
        'active_river_receipt': 'river_active_summary.json',
        'primary_signal': primary_signal,
        'dominant_remaining_issue': dominant_issue,
        'suggested_next_action': suggested_next_action,
        'active_product_name': runtime.get('active_product_name'),
        'active_river_benchmark_mode': evaluation.get('active_river_benchmark_mode'),
        'active_river_benchmark_decision_basis': evaluation.get('active_river_benchmark_decision_basis'),
        'primary_shape_signal': shape.get('primary_shape_signal'),
        'runtime_issue': runtime_issue,
        'shape_issue': shape_issue,
        'evaluation_issue': evaluation_issue,
        'runtime_contract_ok': runtime_ok,
        'river_shape_ok': shape_ok,
        'evaluation_constraint_active': evaluation_constraint,
        'degraded_runtime_mode_active': bool(runtime.get('degraded_mode_active', False)),
        'continuity_safeguard_used': bool(runtime.get('continuity_safeguard_used', False)),
        'backbone_moved': bool(shape.get('backbone_moved', False)),
        'backbone_adjusted_station_count': int(shape.get('backbone_adjusted_station_count') or 0),
        'weakest_lateral_role': shape.get('weakest_lateral_role') or evaluation.get('weakest_lateral_role'),
        'withheld_support_plan_available': bool(evaluation.get('withheld_support_plan_available', False)),
        'withheld_support_plan_path': evaluation.get('withheld_support_plan_path') or withheld.get('river_withheld_support_points'),
        'withheld_support_cli_flag': evaluation.get('withheld_support_cli_flag'),
        'drilldown_receipt_paths': {
            'river_active_runtime_summary.json': str(runtime_path) if runtime_path.exists() else None,
            'river_active_shape_summary.json': str(shape_path) if shape_path.exists() else None,
            'benchmark_river_active_evaluation_summary.json': str(evaluation_path) if evaluation_path.exists() else None,
            'benchmark_river_withheld_support_receipt.json': str(withheld_path) if withheld_path.exists() else None,
        },
        'diagnostic_receipt_roles': {
            'river_active_runtime_summary.json': 'diagnostic_only',
            'river_active_shape_summary.json': 'diagnostic_only',
            'benchmark_river_active_evaluation_summary.json': 'diagnostic_only',
            'benchmark_river_withheld_support_receipt.json': 'diagnostic_only',
        },
    }
    return summary


def _build_priority_status(*, manifest: dict[str, Any], reports_dir: Path) -> list[dict[str, Any]]:
    copied_map = {
        str(Path(entry['dest'])): entry
        for entry in manifest.get('copied', [])
        if isinstance(entry, dict) and entry.get('dest')
    }
    missing_map = {
        str(entry.get('label')): entry
        for entry in manifest.get('missing', [])
        if isinstance(entry, dict) and entry.get('label')
    }
    priority_status: list[dict[str, Any]] = []
    for label, dest_name, description in _PRIORITY_FILES:
        status = _status_for_dest(label=label, dest_name=dest_name, copied_map=copied_map, missing_map=missing_map, reports_dir=reports_dir)
        priority_status.append({
            'label': label,
            'file': dest_name,
            'description': description,
            **status,
        })
    return priority_status


def _build_readme(manifest: dict[str, Any]) -> str:
    priority_status = manifest.get('priority_status', []) if isinstance(manifest.get('priority_status'), list) else []
    present_priority = sum(1 for item in priority_status if isinstance(item, dict) and item.get('present'))
    missing_priority = len(priority_status) - present_priority
    lines = [
        'Open this folder first when diagnosing a run.',
        '',
        'Priority diagnosis order:',
    ]
    for idx, item in enumerate(priority_status, start=1):
        if not isinstance(item, dict):
            continue
        status = str(item.get('status') or 'unknown')
        suffix = ''
        if not item.get('present'):
            reason = str(item.get('reason') or status)
            suffix = f' [missing: {reason}]'
        lines.append(f"{idx}. {item.get('file')} — {item.get('description')}{suffix}")

    lines.extend([
        '',
        'Interpretation guide:',
        '- Use river_active_summary.json as the single primary river receipt for runtime, shape, and evaluation state together.',
        '- Use river_active_runtime_summary.json, river_active_shape_summary.json, and benchmark_river_active_evaluation_summary.json only as river drilldowns.',
        '- Use benchmark_river_withheld_support_receipt.json when the active river summary says a withheld-support benchmark plan is available or recommended.',
        '- Use benchmark_river_mode_summary.json, benchmark_river_receipt_triage_summary.json, and benchmark_river_primary_focus_summary.json only as deeper evaluation drilldowns; hard-problem holdout coverage is diagnostic-only unless a withheld-support benchmark is actually applied.',
        '- Use river_primary_guidance_summary.json and river_primary_surface_contract.json only as runtime drilldowns.',
        '- Use river_backbone_smoothing_summary.json, river_centerline_width_propagation_summary.json, river_role_agreement_summary.json, and river_section_target_agreement_summary.json only as river-shape drilldowns.',
        '',
        f"Priority files present: {present_priority}/{len(priority_status)}",
        f"Copied files: {len(manifest.get('copied', []))}",
        f"Missing/not reported: {len(manifest.get('missing', []))}",
    ])
    if missing_priority > 0:
        missing_files = [str(item.get('file')) for item in priority_status if isinstance(item, dict) and not item.get('present')]
        lines.append(f"Missing priority files: {', '.join(missing_files)}")
    return '\n'.join(lines) + '\n'


def write_reports_hub(*, out_dir: str | Path, report: dict[str, Any]) -> dict[str, str]:
    out_dir = Path(out_dir)
    reports_dir = out_dir / REPORTS_DIRNAME
    reports_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    top = report.get('outputs', {}) if isinstance(report, dict) else {}
    for key, source_name, dest_name in _TOP_OUTPUT_KEYS:
        src = _resolve_path(top.get(key), base_dir=out_dir)
        if src is None:
            src = out_dir / source_name
        _copy_or_record(src, reports_dir / dest_name, copied=copied, missing=missing, label=key)

    river_outputs = (((report or {}).get('river') or {}).get('outputs') or {})
    for key, dest_name in _RIVER_OUTPUT_KEYS:
        src = _resolve_path(river_outputs.get(key), base_dir=out_dir)
        _copy_or_record(src, reports_dir / dest_name, copied=copied, missing=missing, label=key)

    active_river_runtime_summary = _build_active_river_runtime_summary(out_dir=out_dir, report=report)
    active_river_runtime_path = reports_dir / 'river_active_runtime_summary.json'
    active_river_runtime_path.write_text(json.dumps(active_river_runtime_summary, indent=2), encoding='utf-8')
    copied.append({
        'label': 'river_active_runtime_summary',
        'source': None,
        'dest': str(active_river_runtime_path),
        'action': 'generated',
        'artifact_role': _artifact_role('river_active_runtime_summary'),
    })

    active_river_shape_summary = _build_active_river_shape_summary(out_dir=out_dir, report=report)
    active_river_shape_path = reports_dir / 'river_active_shape_summary.json'
    active_river_shape_path.write_text(json.dumps(active_river_shape_summary, indent=2), encoding='utf-8')
    copied.append({
        'label': 'river_active_shape_summary',
        'source': None,
        'dest': str(active_river_shape_path),
        'action': 'generated',
        'artifact_role': _artifact_role('river_active_shape_summary'),
    })

    active_river_summary = _build_active_river_summary(reports_dir=reports_dir)
    active_river_summary_path = reports_dir / 'river_active_summary.json'
    active_river_summary_path.write_text(json.dumps(active_river_summary, indent=2), encoding='utf-8')
    copied.append({
        'label': 'river_active_summary',
        'source': None,
        'dest': str(active_river_summary_path),
        'action': 'generated',
        'artifact_role': _artifact_role('river_active_summary'),
    })

    manifest = {
        'reports_dir': str(reports_dir),
        'copied': copied,
        'missing': missing,
        'artifact_roles': {entry.get('label'): entry.get('artifact_role') for entry in copied if isinstance(entry, dict) and entry.get('label')},
    }
    manifest = _reconcile_manifest_with_reports_dir(manifest=manifest)
    manifest['priority_status'] = _build_priority_status(manifest=manifest, reports_dir=reports_dir)
    manifest_path = reports_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    readme_path = reports_dir / README_NAME
    readme_path.write_text(_build_readme(manifest), encoding='utf-8')

    return {
        'reports_readme': str(readme_path),
        'reports_manifest': str(manifest_path),
    }
