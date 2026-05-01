from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from canonical_river_seam_identity import compare_touching_horizontal_border, load_canonical_identity


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _latest_bathy_report(run_dir: Path) -> Path | None:
    direct = run_dir / 'bathy_report.json'
    if direct.exists():
        return direct
    candidates = sorted(run_dir.glob('derived_cache/*/final_reporting/bathy_report.json'))
    return candidates[-1] if candidates else None


def _resolve_retained_final_output(run_dir: Path, outputs: dict[str, Any] | None) -> Path | None:
    candidates: list[Path] = [run_dir / 'combined' / 'DEM_enhanced.tif']
    if outputs:
        for key in ('combined_warped', 'final', 'combined'):
            value = outputs.get(key)
            if value not in (None, ''):
                candidates.append(Path(value))
    writer_receipt = run_dir / 'river_workflow' / 'manifests' / 'final_dem_writer_receipt.json'
    if writer_receipt.exists():
        try:
            payload = _read_json(writer_receipt)
        except Exception:
            payload = {}
        for key in ('final_dem_path', 'canonical_final_dem_path'):
            value = payload.get(key)
            if value not in (None, ''):
                candidates.append(Path(value))
    final_output_receipt = run_dir / 'final' / 'final_output_receipt.json'
    if final_output_receipt.exists():
        try:
            payload = _read_json(final_output_receipt)
        except Exception:
            payload = {}
        for key in ('final_output_path', 'final_dem_path', 'combined_warped'):
            value = payload.get(key)
            if value not in (None, ''):
                candidates.append(Path(value))
    stage_summary = run_dir / 'reports' / 'river_stage_chain_summary.json'
    if stage_summary.exists():
        try:
            summary = _read_json(stage_summary)
        except Exception:
            summary = {}
        for stage in summary.get('stages', []):
            role = stage.get('primary_artifact_role')
            out = stage.get('primary_output')
            if role in {'final_dem', 'final_output'} and out not in (None, ''):
                candidates.append(Path(out))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _candidate_output_paths(outputs: dict[str, Any] | None) -> list[Path]:
    candidates: list[Path] = []
    if not outputs:
        return candidates
    for key in (
        'linear_canonical_solve_contract',
        'river_workflow_shared_solve_contract',
        'linear_canonical_solve_identity',
        'river_workflow_shared_solve_identity',
        'linear_canonical_solve_identity_receipt',
        'river_workflow_canonical_solve_identity_receipt',
    ):
        value = outputs.get(key)
        if value not in (None, ''):
            candidates.append(Path(value))
    river_outputs = outputs.get('river') if isinstance(outputs.get('river'), dict) else None
    if river_outputs:
        nested_outputs = river_outputs.get('outputs') if isinstance(river_outputs.get('outputs'), dict) else None
        if nested_outputs:
            candidates.extend(_candidate_output_paths(nested_outputs))
        candidates.extend(_candidate_output_paths(river_outputs))
    return candidates


def _report_output_candidates(report_payload: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(_candidate_output_paths(report_payload.get('outputs') if isinstance(report_payload.get('outputs'), dict) else None))
    river_payload = report_payload.get('river') if isinstance(report_payload.get('river'), dict) else None
    if river_payload is not None:
        candidates.extend(_candidate_output_paths(river_payload))
        river_outputs = river_payload.get('outputs') if isinstance(river_payload.get('outputs'), dict) else None
        candidates.extend(_candidate_output_paths(river_outputs))
    return candidates


def _resolve_canonical_identity_source(run_dir: Path, stage_summary: dict[str, Any], report_payload: dict[str, Any] | None) -> Path | None:
    candidates: list[Path] = []
    for stage in stage_summary.get('stages', []):
        role = stage.get('primary_artifact_role')
        out = stage.get('primary_output')
        if role == 'canonical_solve_contract' and out:
            candidates.append(Path(out))
    for stage in stage_summary.get('stages', []):
        if stage.get('stage_name') == 'prepare_linear_canonical_source_bundle' and stage.get('primary_output'):
            candidates.append(Path(stage['primary_output']))
    if report_payload:
        candidates.extend(_report_output_candidates(report_payload))
    candidates.extend([
        run_dir / 'river_workflow' / 'manifests' / 'canonical_solve_identity.json',
        run_dir / 'river_workflow' / 'manifests' / 'canonical_solve_contract.json',
        run_dir / 'reports' / 'canonical_solve_identity.json',
    ])
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def load_export_run_identity(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    stage_summary_path = run_dir / 'reports' / 'river_stage_chain_summary.json'
    if not stage_summary_path.exists():
        raise FileNotFoundError(stage_summary_path)
    stage_summary = _read_json(stage_summary_path)
    final_output_path = None
    for stage in stage_summary.get('stages', []):
        role = stage.get('primary_artifact_role')
        out = stage.get('primary_output')
        if role in {'final_dem', 'final_output'} and out:
            final_output_path = Path(out)
    bathy_report_path = _latest_bathy_report(run_dir)
    report_payload = _read_json(bathy_report_path) if bathy_report_path is not None else {}
    outputs = report_payload.get('outputs', {}) if isinstance(report_payload.get('outputs'), dict) else {}
    retained_final_output = _resolve_retained_final_output(run_dir, outputs)
    if retained_final_output is not None:
        final_output_path = retained_final_output
    canonical_contract_path = _resolve_canonical_identity_source(run_dir, stage_summary, report_payload)
    if canonical_contract_path is None:
        raise ValueError('export_run_missing_canonical_contract')
    if final_output_path is None:
        raise ValueError('export_run_missing_final_output')
    return {
        'run_dir': str(run_dir),
        'canonical_contract_path': str(canonical_contract_path),
        'canonical_identity_source_path': str(canonical_contract_path),
        'final_output_path': str(final_output_path),
        'canonical_identity': load_canonical_identity(canonical_contract_path),
        'stage_summary_path': str(stage_summary_path),
        'bathy_report_path': str(bathy_report_path) if bathy_report_path is not None else None,
    }



def _load_run_contract_for_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    run_contract_path = manifest.get('run_contract')
    if run_contract_path in (None, ''):
        raise ValueError('stage_manifest_missing_run_contract')
    return _read_json(Path(run_contract_path))


def load_canonical_solve_cache_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(Path(path))
    payload['manifest_path'] = str(path)
    return payload


def _cache_manifest_path_for_run(run_dir: Path) -> Path:
    manifest = load_stage_identity_manifest(run_dir)
    run_contract = _load_run_contract_for_manifest(manifest)
    shared = dict(run_contract.get('shared_solve') or {})
    cache_manifest_path = shared.get('cache_manifest_path')
    if cache_manifest_path in (None, ''):
        raise FileNotFoundError('canonical_solve_cache_manifest.json')
    return Path(cache_manifest_path)


def _cache_stage_identity_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    stage_payload = payload.get('stage_identity_artifacts') or {}
    if stage_payload:
        return dict(stage_payload)
    artifacts = {}
    legacy = payload.get('artifacts') or {}
    legacy_map = {
        'centerline_points': 'centerline_points',
        'centerline_wse_proxy_points': 'centerline_wse_proxy',
        'centerline_authoritative_bed_points': 'centerline_authoritative_bed',
        'centerline_observed_offset_points': 'centerline_observed_offset',
        'centerline_modeled_offset_points': 'centerline_modeled_offset',
        'centerline_bed_backbone_points': 'centerline_bed_backbone',
        'river_corridor_solve': 'river_corridor_solve',
        'river_primary_surface_solve': 'river_primary_surface_solve',
        'river_primary_surface_solve_locked': 'river_primary_surface_solve_locked',
    }
    for stage_key, cache_key in legacy_map.items():
        value = legacy.get(cache_key)
        if isinstance(value, dict):
            artifacts[stage_key] = dict(value)
        elif value not in (None, ''):
            pth = Path(value)
            artifacts[stage_key] = {
                'path': str(pth),
                'sha256': (sha256_file(pth) if pth.exists() else None),
            }
    return artifacts


def compare_stage_identity_to_cache(stage_identity_manifest: Path | dict[str, Any], cache_manifest: Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(stage_identity_manifest, (str, Path)):
        stage_path = Path(stage_identity_manifest)
        if stage_path.is_file():
            stage_payload = _read_json(stage_path)
            stage_payload['manifest_path'] = str(stage_path)
        else:
            stage_payload = load_stage_identity_manifest(stage_path)
    else:
        stage_payload = dict(stage_identity_manifest)
    cache_payload = load_canonical_solve_cache_manifest(Path(cache_manifest)) if isinstance(cache_manifest, (str, Path)) else dict(cache_manifest)
    stage_artifacts = dict(stage_payload.get('artifacts') or {})
    cache_artifacts = _cache_stage_identity_artifacts(cache_payload)
    compared_stage_keys = [
        key for key in stage_payload.get('stage_keys', [])
        if key in cache_artifacts
    ]
    comparisons: dict[str, Any] = {}
    first_diverging = None
    for key in compared_stage_keys:
        stage_entry = dict(stage_artifacts.get(key) or {})
        cache_entry = dict(cache_artifacts.get(key) or {})
        same = stage_entry.get('sha256') == cache_entry.get('sha256') and stage_entry.get('exists', True)
        comparisons[key] = {
            'stage_identity_path': stage_entry.get('path'),
            'stage_identity_sha256': stage_entry.get('sha256'),
            'stage_identity_exists': stage_entry.get('exists'),
            'cache_path': cache_entry.get('path'),
            'cache_sha256': cache_entry.get('sha256'),
            'same': bool(same),
        }
        if first_diverging is None and not same:
            first_diverging = str(key)
    return {
        'stage_identity_manifest_path': stage_payload.get('manifest_path'),
        'cache_manifest_path': cache_payload.get('manifest_path'),
        'compared_stage_keys': compared_stage_keys,
        'all_compared_stage_hashes_match': first_diverging is None,
        'first_diverging_stage_key': first_diverging,
        'stage_comparisons': comparisons,
    }


def compare_adjacent_aoi_cached_canonical_products(north_run_dir: Path, south_run_dir: Path) -> dict[str, Any]:
    north_stage = load_stage_identity_manifest(Path(north_run_dir))
    south_stage = load_stage_identity_manifest(Path(south_run_dir))
    north_contract = _load_run_contract_for_manifest(north_stage)
    south_contract = _load_run_contract_for_manifest(south_stage)
    north_shared = dict(north_contract.get('shared_solve') or {})
    south_shared = dict(south_contract.get('shared_solve') or {})
    north_cache_key = north_shared.get('cache_key')
    south_cache_key = south_shared.get('cache_key')
    north_cache_path = _cache_manifest_path_for_run(Path(north_run_dir))
    south_cache_path = _cache_manifest_path_for_run(Path(south_run_dir))
    north_cache = load_canonical_solve_cache_manifest(north_cache_path)
    south_cache = load_canonical_solve_cache_manifest(south_cache_path)
    north_artifacts = _cache_stage_identity_artifacts(north_cache)
    south_artifacts = _cache_stage_identity_artifacts(south_cache)
    compared_stage_keys = sorted(set(north_artifacts.keys()) | set(south_artifacts.keys()))
    comparisons: dict[str, Any] = {}
    first_diverging = None
    for key in compared_stage_keys:
        north_entry = dict(north_artifacts.get(key) or {})
        south_entry = dict(south_artifacts.get(key) or {})
        same = north_entry.get('sha256') == south_entry.get('sha256')
        comparisons[key] = {
            'north_cache_path': north_entry.get('path'),
            'south_cache_path': south_entry.get('path'),
            'north_cache_sha256': north_entry.get('sha256'),
            'south_cache_sha256': south_entry.get('sha256'),
            'same': bool(same),
        }
        if first_diverging is None and not same:
            first_diverging = str(key)
    same_cache_key = north_cache_key == south_cache_key and north_cache_key not in (None, '')
    return {
        'north_cache_manifest_path': str(north_cache_path),
        'south_cache_manifest_path': str(south_cache_path),
        'north_cache_key': north_cache_key,
        'south_cache_key': south_cache_key,
        'same_canonical_solve_cache_key': same_cache_key,
        'compared_stage_keys': compared_stage_keys,
        'all_cached_stage_hashes_match': same_cache_key and first_diverging is None,
        'first_diverging_stage_key': first_diverging,
        'cached_stage_comparisons': comparisons,
    }


def load_stage_identity_manifest(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    candidates = [
        run_dir / 'river_workflow' / 'manifests' / 'canonical_stage_identity.json',
        run_dir / 'manifests' / 'canonical_stage_identity.json',
    ]
    for candidate in candidates:
        if candidate.exists():
            payload = _read_json(candidate)
            payload['manifest_path'] = str(candidate)
            return payload
    raise FileNotFoundError('canonical_stage_identity.json')


def compare_adjacent_aoi_stage_identity(north_run_dir: Path, south_run_dir: Path) -> dict[str, Any]:
    north = load_stage_identity_manifest(Path(north_run_dir))
    south = load_stage_identity_manifest(Path(south_run_dir))
    keys = sorted(set(list(north.get('stage_keys', [])) + list(south.get('stage_keys', []))))
    comparisons: dict[str, Any] = {}
    mismatched = []
    for key in keys:
        n = dict(north.get('artifacts', {}).get(key) or {})
        s = dict(south.get('artifacts', {}).get(key) or {})
        same = n.get('sha256') == s.get('sha256') and n.get('exists') == s.get('exists')
        comparisons[key] = {
            'north_path': n.get('path'),
            'south_path': s.get('path'),
            'north_exists': n.get('exists'),
            'south_exists': s.get('exists'),
            'north_sha256': n.get('sha256'),
            'south_sha256': s.get('sha256'),
            'same': bool(same),
        }
        if not same:
            mismatched.append(str(key))
    return {
        'north_manifest_path': north.get('manifest_path'),
        'south_manifest_path': south.get('manifest_path'),
        'north_canonical_solve_identity_path': north.get('canonical_solve_identity_path'),
        'south_canonical_solve_identity_path': south.get('canonical_solve_identity_path'),
        'all_stage_hashes_match': len(mismatched) == 0,
        'mismatched_stage_keys': mismatched,
        'stage_artifact_comparisons': comparisons,
    }



def compare_adjacent_aoi_verification(north_run_dir: Path, south_run_dir: Path) -> dict[str, Any]:
    export_identity = compare_north_south_export_identity(Path(north_run_dir), Path(south_run_dir))
    stage_identity = compare_adjacent_aoi_stage_identity(Path(north_run_dir), Path(south_run_dir))
    cached_identity = compare_adjacent_aoi_cached_canonical_products(Path(north_run_dir), Path(south_run_dir))
    solve_identity_ok = all([
        bool(export_identity.get('same_identity_version', True)),
        bool(export_identity.get('same_canonical_network_identity_tag', True)),
        bool(export_identity.get('same_authoritative_source_tag', True)),
        bool(export_identity.get('same_baseline_source_tag', True)),
        bool(export_identity.get('same_canonical_solve_cache_key')),
        bool(export_identity.get('same_canonical_solve_grid_hash')),
        bool(export_identity.get('same_canonical_solve_authoritative_measured_only_hash')),
        bool(export_identity.get('same_canonical_solve_authoritative_support_mask_hash')),
        bool(export_identity.get('same_canonical_solve_baseline_background_hash')),
    ])
    stage_ok = bool(stage_identity.get('all_stage_hashes_match'))
    cache_ok = bool(cached_identity.get('all_cached_stage_hashes_match'))
    border_ok = bool((export_identity.get('border_comparison') or {}).get('allclose'))
    first_diverging_stage_key = None
    if not stage_ok:
        mismatched = list(stage_identity.get('mismatched_stage_keys') or [])
        first_diverging_stage_key = (str(mismatched[0]) if mismatched else None)
    elif not cache_ok:
        first_diverging_stage_key = cached_identity.get('first_diverging_stage_key')
    if solve_identity_ok and stage_ok and cache_ok and border_ok:
        status = 'passed'
        reason = 'adjacent_aoi_identity_verified'
    elif not solve_identity_ok:
        status = 'failed'
        reason = 'canonical_solve_identity_mismatch'
    elif not stage_ok:
        status = 'failed'
        reason = 'first_canonical_stage_diverged_between_adjacent_aois'
    elif not cache_ok:
        status = 'failed'
        reason = 'canonical_cached_products_diverged_between_adjacent_aois'
    else:
        status = 'failed'
        reason = 'final_export_border_mismatch'
    return {
        'north_run_dir': str(Path(north_run_dir)),
        'south_run_dir': str(Path(south_run_dir)),
        'status': status,
        'reason': reason,
        'solve_identity_ok': solve_identity_ok,
        'stage_identity_ok': stage_ok,
        'cached_canonical_products_ok': cache_ok,
        'final_border_ok': border_ok,
        'first_diverging_stage_key': first_diverging_stage_key,
        'export_identity': export_identity,
        'stage_identity': stage_identity,
        'cached_canonical_products': cached_identity,
    }

def compare_north_south_export_identity(north_run_dir: Path, south_run_dir: Path) -> dict[str, Any]:
    north = load_export_run_identity(Path(north_run_dir))
    south = load_export_run_identity(Path(south_run_dir))
    north_identity = north['canonical_identity']
    south_identity = south['canonical_identity']
    same_cache_key = north_identity.get('canonical_solve_cache_key') == south_identity.get('canonical_solve_cache_key')
    same_identity_version = north_identity.get('identity_version') == south_identity.get('identity_version')
    same_network_identity_tag = north_identity.get('canonical_network_identity_tag') == south_identity.get('canonical_network_identity_tag')
    same_authoritative_source_tag = north_identity.get('authoritative_source_tag') == south_identity.get('authoritative_source_tag')
    same_baseline_source_tag = north_identity.get('baseline_source_tag') == south_identity.get('baseline_source_tag')
    same_grid_hash = north_identity.get('canonical_solve_grid_path_sha256') == south_identity.get('canonical_solve_grid_path_sha256')
    same_measured_hash = north_identity.get('canonical_solve_authoritative_measured_only_path_sha256') == south_identity.get('canonical_solve_authoritative_measured_only_path_sha256')
    same_support_mask_hash = north_identity.get('canonical_solve_authoritative_support_mask_path_sha256') == south_identity.get('canonical_solve_authoritative_support_mask_path_sha256')
    same_baseline_hash = north_identity.get('canonical_solve_baseline_background_path_sha256') == south_identity.get('canonical_solve_baseline_background_path_sha256')
    border = compare_touching_horizontal_border(Path(north['final_output_path']), Path(south['final_output_path']))
    return {
        'north_run_dir': north['run_dir'],
        'south_run_dir': south['run_dir'],
        'same_canonical_solve_cache_key': same_cache_key,
        'same_identity_version': same_identity_version,
        'same_canonical_network_identity_tag': same_network_identity_tag,
        'same_authoritative_source_tag': same_authoritative_source_tag,
        'same_baseline_source_tag': same_baseline_source_tag,
        'same_canonical_solve_grid_hash': same_grid_hash,
        'same_canonical_solve_authoritative_measured_only_hash': same_measured_hash,
        'same_canonical_solve_authoritative_support_mask_hash': same_support_mask_hash,
        'same_canonical_solve_baseline_background_hash': same_baseline_hash,
        'canonical_contract_paths': {
            'north': north['canonical_contract_path'],
            'south': south['canonical_contract_path'],
        },
        'final_output_paths': {
            'north': north['final_output_path'],
            'south': south['final_output_path'],
        },
        'border_comparison': border,
    }


__all__ = ['load_export_run_identity', 'load_stage_identity_manifest', 'load_canonical_solve_cache_manifest', 'compare_stage_identity_to_cache', 'compare_adjacent_aoi_verification', 'compare_adjacent_aoi_stage_identity', 'compare_adjacent_aoi_cached_canonical_products', 'compare_north_south_export_identity']
