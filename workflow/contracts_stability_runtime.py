from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from contract_enforcement import ContractResult, ContractSuiteResult


def _iter_seam_ios(cfg: Any) -> List[str]:
    out = [str(x) for x in (getattr(cfg, 'seam_compare_with_io', []) or []) if str(x).strip()]
    list_path = getattr(cfg, 'seam_compare_with_io_list', None)
    if list_path:
        p = Path(str(list_path))
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                out.append(s)
    return out


def _existing_json(path: Any) -> Dict[str, Any] | None:
    if not path:
        return None
    try:
        p = Path(path)
    except (TypeError, ValueError, OSError):
        return None
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def run_stability_runtime_contracts(*, cfg: Any, report: Dict[str, Any]) -> ContractSuiteResult:
    suite = ContractSuiteResult(stage='stability')
    seams = report.get('seams', {}) if isinstance(report.get('seams'), dict) else {}
    seam_results = list(seams.get('adjacent_tile_comparisons') or [])
    overlap_eval = seams.get('overlap_identity_evaluation') if isinstance(seams.get('overlap_identity_evaluation'), dict) else {}
    overlap_checks = list(seams.get('overlap_identity_checks') or [])
    requested_neighbors = _iter_seam_ios(cfg)

    seam_stage_ok = True
    seam_failures: List[str] = []
    if requested_neighbors:
        if not seam_results:
            seam_stage_ok = False
            seam_failures.append('seam comparisons were requested but no seam results were recorded')
        else:
            for result in seam_results:
                status = str((result or {}).get('status') or '').strip().lower()
                if status != 'ok':
                    seam_stage_ok = False
                    neighbor = str((result or {}).get('neighbor_io_manifest') or '<unknown>')
                    reason = str((result or {}).get('error') or f'non-ok seam status: {status}')
                    seam_failures.append(f'{neighbor}: {reason}')
    suite.add(ContractResult(
        name='requested_seam_comparisons_recorded',
        stage=suite.stage,
        severity='error',
        passed=seam_stage_ok,
        message='Requested seam comparisons completed successfully.' if seam_stage_ok else '; '.join(seam_failures),
        metrics={'requested_neighbor_count': len(requested_neighbors), 'completed_comparisons': len(seam_results)},
    ))

    overlap_ok = True
    overlap_msg = 'No overlap identity failures were reported.'
    overlap_failures = list(overlap_eval.get('failures') or []) if isinstance(overlap_eval, dict) else []
    if overlap_checks:
        all_ok = overlap_eval.get('all_ok') if isinstance(overlap_eval, dict) else None
        if all_ok is False:
            overlap_ok = False
            overlap_msg = '; '.join(
                f"{str(f.get('artifact') or '<artifact>')} vs {str(f.get('neighbor_io_manifest') or '<neighbor>')}: {str(f.get('reason') or 'overlap identity failure')}"
                for f in overlap_failures
            ) or 'Overlap identity evaluation reported failures.'
    suite.add(ContractResult(
        name='overlap_identity_consistent',
        stage=suite.stage,
        severity='error',
        passed=overlap_ok,
        message=overlap_msg,
        metrics={'checked': len(overlap_checks), 'failure_count': len(overlap_failures)},
    ))

    river_stability = _existing_json(report.get('outputs', {}).get('river_stability_summary') if isinstance(report.get('outputs'), dict) else None)
    nested = river_stability.get('nested_aoi_relationship_to_solve_domain') if isinstance(river_stability, dict) and isinstance(river_stability.get('nested_aoi_relationship_to_solve_domain'), dict) else None
    nested_ok = True
    nested_msg = 'Nested AOI relationship was not available for this run.'
    nested_metrics: Dict[str, Any] = {'available': nested is not None}
    severity = 'warning'
    if nested is not None:
        contains = nested.get('contains_export_aoi')
        overlap_fraction = nested.get('overlap_fraction_of_export')
        nested_metrics['overlap_fraction_of_export'] = float(overlap_fraction) if overlap_fraction is not None else None
        if contains is False:
            nested_ok = False
            severity = 'error'
            nested_msg = 'Solve-domain receipt does not contain the export AOI.'
        else:
            nested_msg = 'Solve-domain receipt contains the export AOI.'
    suite.add(ContractResult(
        name='solve_domain_contains_export_aoi',
        stage=suite.stage,
        severity=severity,
        passed=nested_ok,
        message=nested_msg,
        metrics=nested_metrics,
        artifact_paths={'river_stability_summary': str(report.get('outputs', {}).get('river_stability_summary'))} if isinstance(report.get('outputs'), dict) and report.get('outputs', {}).get('river_stability_summary') else {},
        recommended_action='Review the canonical river scaffold solve-domain receipt when this fails.' if not nested_ok else None,
    ))
    return suite
