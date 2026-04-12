from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from core.json_io import write_json
from seam_metrics import (
    compute_raster_trusted_interior_identity_metrics,
    compute_vector_trusted_interior_identity_metrics,
)
from final_reporting import evaluate_overlap_identity_checks


_RASTER_KEYS: Tuple[str, ...] = (
    'river_channel_surface',
    'river_channel_surface_graph_mode',
    'river_channel_surface_support_class',
    'river_channel_surface_uncertainty',
    'river_channel_surface_hard_lock',
    'river_channel_surface_junction_constrained',
    'river_channel_surface_unsupported_span',
    'river_channel_surface_unsupported_regime',
    'river_channel_surface_residual_to_candidate',
    'river_generalized_longitudinal_bed_base',
    'river_generalized_longitudinal_bed_reconciled',
    'river_longitudinal_profile_local_authoritative_reconciliation',
    'river_longitudinal_profile_local_authoritative_reconciliation_influence',
)

_VECTOR_SPECS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        'river_graph_backbone_diagnostics',
        (
            'graph_backbone_z_m',
            'graph_solution_mode',
            'graph_solver_support_class',
            'graph_unsupported_regime',
        ),
    ),
    (
        'river_channel_scaffold_nodes',
        (
            'bed_z_m',
            'z_source',
            'graph_solution_mode',
            'graph_solver_support_class',
            'graph_unsupported_regime',
        ),
    ),
)


def _load_manifest(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _manifest_lookup(manifest: Dict[str, Any], key: str) -> Optional[str]:
    value = manifest.get(key)
    if value:
        return str(value)
    support = manifest.get('support_artifacts')
    if isinstance(support, dict) and support.get(key):
        return str(support.get(key))
    guidance = manifest.get('guidance_artifacts')
    if isinstance(guidance, dict) and guidance.get(key):
        return str(guidance.get(key))
    return None


def _existing(path: Optional[str | Path]) -> Optional[str]:
    if not path:
        return None
    try:
        p = Path(str(path))
    except (TypeError, ValueError, OSError):
        return None
    return str(p) if p.exists() else None


def run_nested_aoi_river_invariance_test(
    *,
    small_final_outputs_manifest: str | Path,
    large_final_outputs_manifest: str | Path,
    out_path: Optional[str | Path] = None,
    tolerance: float = 1.0e-6,
) -> Dict[str, Any]:
    small_manifest_path = Path(small_final_outputs_manifest)
    large_manifest_path = Path(large_final_outputs_manifest)
    small = _load_manifest(small_manifest_path)
    large = _load_manifest(large_manifest_path)

    small_trusted = _existing(_manifest_lookup(small, 'river_trusted_interior'))
    large_trusted = _existing(_manifest_lookup(large, 'river_trusted_interior'))

    checks = []

    for key in _RASTER_KEYS:
        sp = _existing(_manifest_lookup(small, key))
        lp = _existing(_manifest_lookup(large, key))
        if not (sp and lp and small_trusted and large_trusted):
            checks.append({
                'artifact': f'trusted_interior::{key}',
                'status': 'no_valid',
                'reason': 'missing artifact or trusted interior raster',
            })
            continue
        stats = compute_raster_trusted_interior_identity_metrics(sp, lp, small_trusted, large_trusted)
        stats.update({
            'artifact': f'trusted_interior::{key}',
            'small_final_outputs_manifest': str(small_manifest_path),
            'large_final_outputs_manifest': str(large_manifest_path),
        })
        checks.append(stats)

    for key, fields in _VECTOR_SPECS:
        sp = _existing(_manifest_lookup(small, key))
        lp = _existing(_manifest_lookup(large, key))
        if not (sp and lp and small_trusted and large_trusted):
            checks.append({
                'artifact': f'trusted_interior::{key}',
                'status': 'no_valid',
                'reason': 'missing vector artifact or trusted interior raster',
            })
            continue
        stats = compute_vector_trusted_interior_identity_metrics(
            sp,
            lp,
            small_trusted,
            large_trusted,
            compare_fields=fields,
        )
        stats.update({
            'artifact': f'trusted_interior::{key}',
            'small_final_outputs_manifest': str(small_manifest_path),
            'large_final_outputs_manifest': str(large_manifest_path),
        })
        checks.append(stats)

    evaluation = evaluate_overlap_identity_checks(checks, tolerance=tolerance)
    payload = {
        'small_final_outputs_manifest': str(small_manifest_path),
        'large_final_outputs_manifest': str(large_manifest_path),
        'small_aoi': small.get('aoi'),
        'large_aoi': large.get('aoi'),
        'trusted_interior_artifacts_checked': len(checks),
        'trusted_interior_identity_checks': checks,
        'trusted_interior_identity_evaluation': evaluation,
        'all_trusted_interior_identity_ok': evaluation.get('all_ok'),
        'checked_artifacts': [c.get('artifact') for c in checks],
    }
    if out_path is not None:
        write_json(Path(out_path), payload)
    return payload


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run nested-AOI trusted-interior invariance checks for river artifacts.')
    ap.add_argument('--small-final-outputs-manifest', required=True)
    ap.add_argument('--large-final-outputs-manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tolerance', type=float, default=1.0e-6)
    return ap.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ns = _parse_args(argv)
    run_nested_aoi_river_invariance_test(
        small_final_outputs_manifest=ns.small_final_outputs_manifest,
        large_final_outputs_manifest=ns.large_final_outputs_manifest,
        out_path=ns.out,
        tolerance=float(ns.tolerance),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
