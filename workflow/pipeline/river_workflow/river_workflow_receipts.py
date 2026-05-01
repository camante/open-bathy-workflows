from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.json_io import write_json
from canonical_river_export_identity import compare_adjacent_aoi_verification, compare_stage_identity_to_cache
from canonical_river_seam_identity import sha256_file
from pipeline.river_workflow.river_workflow_contract import WORKFLOW_STAGE_ORDER, diagnostic_contract_for_stage, validation_checks_for_stage


def _stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify(v) for v in value]
    return value



def _first_mapping_item(mapping: Mapping[str, Any]) -> dict[str, Any] | None:
    if not mapping:
        return None
    key = next(iter(mapping.keys()))
    return {'key': str(key), 'path': _stringify(mapping[key])}


def build_first_wrong_artifact_guide(
    *,
    stage_name: str,
    receipt_path: Path,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    validator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build read-only, stage-local guidance for finding the first wrong artifact.

    This is diagnostic metadata only. It must not be used to select a different
    route, patch a raster, or repair a failed output.
    """
    contract = diagnostic_contract_for_stage(stage_name)
    checks = list((validator or {}).get('checks') or validation_checks_for_stage(stage_name))
    return {
        'schema_version': 1,
        'diagnostic_only': True,
        'may_reroute_or_repair_outputs': False,
        'stage_name': str(stage_name),
        'primary_input': _first_mapping_item(inputs),
        'primary_output': _first_mapping_item(outputs),
        'receipt_path': str(receipt_path),
        'critical_invariant': contract.get('critical_invariant'),
        'first_check': checks[0] if checks else None,
        'validation_checks': [str(x) for x in checks],
        'first_wrong_artifact_hint': contract.get('first_wrong_artifact_hint'),
        'declared_primary_input': contract.get('primary_input'),
        'declared_primary_output': contract.get('primary_output'),
    }


def build_stage_validator_summary(stage_name: str, *, status: str = 'passed') -> dict[str, Any]:
    return {
        'status': str(status),
        'checks': list(validation_checks_for_stage(stage_name)),
    }


def build_stage_trace_entry(
    *,
    stage_name: str,
    receipt_path: Path,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    summary: Mapping[str, Any],
    validator: Mapping[str, Any],
) -> dict[str, Any]:
    stage_order_index = WORKFLOW_STAGE_ORDER.index(stage_name)
    primary_output = _first_mapping_item(outputs)
    first_wrong_artifact_guide = build_first_wrong_artifact_guide(
        stage_name=stage_name,
        receipt_path=receipt_path,
        inputs=inputs,
        outputs=outputs,
        validator=validator,
    )
    return {
        'stage_name': stage_name,
        'stage_order_index': int(stage_order_index),
        'status': str(validator.get('status', 'passed')),
        'receipt_path': str(receipt_path),
        'upstream_stage': WORKFLOW_STAGE_ORDER[stage_order_index - 1] if stage_order_index > 0 else None,
        'downstream_stage': WORKFLOW_STAGE_ORDER[stage_order_index + 1] if stage_order_index + 1 < len(WORKFLOW_STAGE_ORDER) else None,
        'validator': _stringify(dict(validator)),
        'inputs': _stringify(dict(inputs)),
        'outputs': _stringify(dict(outputs)),
        'primary_output': primary_output,
        'first_wrong_artifact_guide': _stringify(first_wrong_artifact_guide),
        'summary': _stringify(dict(summary)),
    }


def write_stage_receipt(
    path: Path,
    *,
    stage_name: str,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    summary: Mapping[str, Any],
    validator: Mapping[str, Any] | None = None,
) -> Path:
    resolved_validator = dict(validator or build_stage_validator_summary(stage_name))
    payload = {
        'stage_name': stage_name,
        'inputs': _stringify(dict(inputs)),
        'outputs': _stringify(dict(outputs)),
        'summary': _stringify(dict(summary)),
        'validator': _stringify(resolved_validator),
        'first_wrong_artifact_guide': _stringify(build_first_wrong_artifact_guide(
            stage_name=stage_name,
            receipt_path=path,
            inputs=inputs,
            outputs=outputs,
            validator=resolved_validator,
        )),
    }
    write_json(path, payload)
    return path


def write_trace_summary(path: Path, *, run_contract_path: Path, stage_trace: list[Mapping[str, Any]]) -> Path:
    payload = {
        'run_contract': str(run_contract_path),
        'stage_order': list(WORKFLOW_STAGE_ORDER),
        'stage_trace': [_stringify(dict(entry)) for entry in stage_trace],
        'diagnosis_guide': {
            'first_wrong_artifact_rule': 'Inspect stage_trace in stage_order. Each active stage includes first_wrong_artifact_guide with its primary input, primary output, critical invariant, and first check. The first stage whose guide/receipt/validator/output looks wrong is the first wrong construction stage.',
            'expected_flow': 'Each stage should consume only the explicit upstream artifact paths listed in its inputs and produce only the explicit outputs listed in its outputs.',
        },
    }
    write_json(path, payload)
    return path






def write_canonical_identity_receipt(
    path: Path,
    *,
    canonical_system_id: str,
    user_aoi: Any,
    canonical_domain_bounds: Any,
    fingerprints: Mapping[str, Any],
    source_system_id: str | None = None,
) -> Path:
    """Write the canonical system identity receipt.

    This receipt intentionally records that user AOI, out-dir, timestamps, temp
    directories, and run IDs are excluded from the canonical identity. The user
    AOI is retained only as export metadata for traceability.
    """
    payload = {
        'stage': 'canonical_system_identity',
        'canonical_system_id': str(canonical_system_id),
        'source_system_id': str(source_system_id) if source_system_id is not None else None,
        'user_aoi': str(user_aoi) if user_aoi is not None else None,
        'canonical_domain_bounds': str(canonical_domain_bounds) if canonical_domain_bounds is not None else None,
        'fingerprints': _stringify(dict(fingerprints)),
        'included_user_aoi_in_id': False,
        'included_out_dir_in_id': False,
        'included_timestamp_in_id': False,
        'included_temporary_directory_in_id': False,
        'included_run_id_in_id': False,
    }
    write_json(path, payload)
    return path



def write_canonical_parent_dem_receipt(
    path: Path,
    *,
    canonical_system_id: str | None,
    parent_dem: Path,
    source_dem: Path | None,
    user_aoi_cropped: bool,
    authoritative_lock_applied: bool,
    source_stage_names: list[str] | tuple[str, ...] | None = None,
    summary: Mapping[str, Any] | None = None,
    canonical_cache: Mapping[str, Any] | None = None,
    cache_validation: Mapping[str, Any] | None = None,
) -> Path:
    """Write the receipt for the canonical parent DEM scientific product."""
    payload = {
        'schema_version': 2,
        'stage': 'canonical_parent_dem',
        'stage_class': 'canonical_parent_finalization',
        'canonical_system_id': str(canonical_system_id) if canonical_system_id not in (None, '') else None,
        'path': str(parent_dem),
        'parent_dem_sha256': sha256_file(parent_dem),
        'source_dem': str(source_dem) if source_dem is not None else None,
        'source_dem_sha256': sha256_file(source_dem),
        'source_stage_names': [str(x) for x in (source_stage_names or [])],
        'user_aoi_cropped': bool(user_aoi_cropped),
        'authoritative_lock_applied': bool(authoritative_lock_applied),
        'role': 'canonical_parent_dem',
        'summary': _stringify(dict(summary or {})),
        'canonical_cache': _stringify(dict(canonical_cache or {})),
        'cache_validation': _stringify(dict(cache_validation or {})),
    }
    write_json(path, payload)
    return path


def write_aoi_export_dem_receipt(
    path: Path,
    *,
    canonical_system_id: str | None,
    parent_dem: Path,
    export_dem: Path,
    export_template: Path | None,
    pixel_values_modified: bool,
    resampled: bool,
    reprojected: bool,
    summary: Mapping[str, Any] | None = None,
    canonical_cache: Mapping[str, Any] | None = None,
) -> Path:
    """Write the receipt for the exact AOI DEM export from the parent DEM."""
    payload = {
        'schema_version': 2,
        'stage': 'aoi_export_dem',
        'stage_class': 'aoi_export_only',
        'canonical_system_id': str(canonical_system_id) if canonical_system_id not in (None, '') else None,
        'parent_dem': str(parent_dem),
        'parent_dem_sha256': sha256_file(parent_dem),
        'export_dem': str(export_dem),
        'export_dem_sha256': sha256_file(export_dem),
        'export_template': str(export_template) if export_template is not None else None,
        'export_template_sha256': sha256_file(export_template),
        'role': 'aoi_export_dem',
        'source_role': 'canonical_parent_dem',
        'post_subset_modifications': bool(pixel_values_modified or resampled or reprojected),
        'construction_attempted': False,
        'construction_stages_run_in_aoi_export': False,
        'aoi_outputs_must_be_exact_parent_subsets': True,
        'pixel_values_modified': bool(pixel_values_modified),
        'resampled': bool(resampled),
        'reprojected': bool(reprojected),
        'summary': _stringify(dict(summary or {})),
        'canonical_cache': _stringify(dict(canonical_cache or {})),
    }
    write_json(path, payload)
    return path


def write_aoi_identity_receipt(
    path: Path,
    *,
    canonical_system_id: str | None,
    identity_result: Mapping[str, Any],
) -> Path:
    """Write the AOI export vs canonical parent identity-check receipt."""
    payload = _stringify(dict(identity_result))
    payload.setdefault('stage', 'aoi_identity')
    payload['canonical_system_id'] = str(canonical_system_id) if canonical_system_id not in (None, '') else None
    payload['role'] = 'aoi_identity_receipt'
    write_json(path, payload)
    return path


def write_canonical_stage_identity_manifest(
    path: Path,
    *,
    run_contract_path: Path,
    canonical_solve_identity_path: Path | None,
    outputs: Mapping[str, Any],
    stage_keys: list[str] | None = None,
) -> Path:
    keys = list(stage_keys or [
        'centerline_points',
        'centerline_wse_proxy_points',
        'centerline_authoritative_bed_points',
        'centerline_observed_offset_points',
        'centerline_modeled_offset_points',
        'centerline_bed_backbone_points',
        'river_corridor_solve',
        'river_primary_surface_solve',
        'river_primary_surface_solve_locked',
        'dem_enhanced_final',
    ])
    artifacts: dict[str, Any] = {}
    for key in keys:
        value = outputs.get(key)
        pth = Path(value) if value not in (None, '') else None
        artifacts[str(key)] = {
            'path': (str(pth) if pth is not None else None),
            'exists': bool(pth is not None and pth.exists()),
            'sha256': (sha256_file(pth) if pth is not None and pth.exists() else None),
        }
    payload = {
        'run_contract': str(run_contract_path),
        'canonical_solve_identity_path': (str(canonical_solve_identity_path) if canonical_solve_identity_path is not None else None),
        'stage_keys': keys,
        'artifacts': artifacts,
    }
    write_json(path, payload)
    return path


def write_canonical_cache_consistency_receipt(
    path: Path,
    *,
    run_contract_path: Path,
    canonical_solve_identity_path: Path | None,
    canonical_solve_cache_manifest_path: Path | None,
    stage_identity_manifest_path: Path | None,
) -> Path:
    if canonical_solve_cache_manifest_path is None or stage_identity_manifest_path is None:
        payload = {
            'run_contract': str(run_contract_path),
            'canonical_solve_identity_path': (str(canonical_solve_identity_path) if canonical_solve_identity_path is not None else None),
            'canonical_solve_cache_manifest_path': (str(canonical_solve_cache_manifest_path) if canonical_solve_cache_manifest_path is not None else None),
            'stage_identity_manifest_path': (str(stage_identity_manifest_path) if stage_identity_manifest_path is not None else None),
            'status': 'skipped',
            'reason': 'missing_stage_identity_or_cache_manifest',
            'first_diverging_stage_key': None,
            'all_compared_stage_hashes_match': None,
            'compared_stage_keys': [],
            'stage_comparisons': {},
        }
    else:
        comparison = compare_stage_identity_to_cache(stage_identity_manifest_path, canonical_solve_cache_manifest_path)
        payload = {
            'run_contract': str(run_contract_path),
            'canonical_solve_identity_path': (str(canonical_solve_identity_path) if canonical_solve_identity_path is not None else None),
            'canonical_solve_cache_manifest_path': str(canonical_solve_cache_manifest_path),
            'stage_identity_manifest_path': str(stage_identity_manifest_path),
            'status': ('passed' if comparison.get('all_compared_stage_hashes_match') else 'failed'),
            'reason': ('all_compared_stage_hashes_match' if comparison.get('all_compared_stage_hashes_match') else 'first_stage_hash_diverged_from_canonical_cache'),
            'first_diverging_stage_key': comparison.get('first_diverging_stage_key'),
            'all_compared_stage_hashes_match': comparison.get('all_compared_stage_hashes_match'),
            'compared_stage_keys': comparison.get('compared_stage_keys', []),
            'stage_comparisons': comparison.get('stage_comparisons', {}),
        }
    write_json(path, payload)
    return path



def write_adjacent_aoi_verification_receipt(
    path: Path,
    *,
    north_run_dir: Path,
    south_run_dir: Path,
) -> Path:
    comparison = compare_adjacent_aoi_verification(north_run_dir, south_run_dir)
    payload = {
        'north_run_dir': str(north_run_dir),
        'south_run_dir': str(south_run_dir),
        'status': comparison.get('status'),
        'reason': comparison.get('reason'),
        'solve_identity_ok': comparison.get('solve_identity_ok'),
        'stage_identity_ok': comparison.get('stage_identity_ok'),
        'cached_canonical_products_ok': comparison.get('cached_canonical_products_ok'),
        'final_border_ok': comparison.get('final_border_ok'),
        'first_diverging_stage_key': comparison.get('first_diverging_stage_key'),
        'export_identity': comparison.get('export_identity', {}),
        'stage_identity': comparison.get('stage_identity', {}),
        'cached_canonical_products': comparison.get('cached_canonical_products', {}),
    }
    write_json(path, payload)
    return path



def write_standard_adjacent_aoi_verification_receipt(
    path: Path,
    *,
    run_contract_path: Path,
    current_run_dir: Path,
    peer_run_dir: Path | None,
) -> Path:
    current_run_dir = Path(current_run_dir)
    peer = (Path(peer_run_dir) if peer_run_dir is not None else None)
    if peer is None:
        payload = {
            'run_contract': str(run_contract_path),
            'current_run_dir': str(current_run_dir),
            'peer_run_dir': None,
            'status': 'skipped',
            'reason': 'no_adjacent_aoi_peer_run_dir_provided',
            'solve_identity_ok': None,
            'stage_identity_ok': None,
            'cached_canonical_products_ok': None,
            'final_border_ok': None,
            'first_diverging_stage_key': None,
            'comparison': {},
        }
    elif not peer.exists():
        payload = {
            'run_contract': str(run_contract_path),
            'current_run_dir': str(current_run_dir),
            'peer_run_dir': str(peer),
            'status': 'skipped',
            'reason': 'adjacent_aoi_peer_run_dir_missing',
            'solve_identity_ok': None,
            'stage_identity_ok': None,
            'cached_canonical_products_ok': None,
            'final_border_ok': None,
            'first_diverging_stage_key': None,
            'comparison': {},
        }
    else:
        comparison = compare_adjacent_aoi_verification(peer, current_run_dir)
        payload = {
            'run_contract': str(run_contract_path),
            'current_run_dir': str(current_run_dir),
            'peer_run_dir': str(peer),
            'status': comparison.get('status'),
            'reason': comparison.get('reason'),
            'solve_identity_ok': comparison.get('solve_identity_ok'),
            'stage_identity_ok': comparison.get('stage_identity_ok'),
            'cached_canonical_products_ok': comparison.get('cached_canonical_products_ok'),
            'final_border_ok': comparison.get('final_border_ok'),
            'first_diverging_stage_key': comparison.get('first_diverging_stage_key'),
            'comparison': comparison,
        }
    write_json(path, payload)
    return path



def _read_json_or_none(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return dict(__import__("json").loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def collect_river_receipts(ctx: Any) -> dict[str, Path | None]:
    """Collect the core seamless-DEM receipts from a river context or paths object.

    This keeps the run-summary writer from rediscovering receipt locations.
    Missing receipts are represented as None so the summary can report partial
    status without inventing paths.
    """
    paths = getattr(ctx, "paths", ctx)
    names = {
        "canonical_system_identity": "canonical_system_identity_receipt",
        "canonical_parent_dem": "canonical_parent_receipt",
        "aoi_export_dem": "aoi_export_receipt",
        "final_dem_materialization": "final_materialization_receipt",
        "aoi_identity": "aoi_identity_receipt",
        "stage_chain_summary": "trace_summary",
        "canonical_cache_consistency": "canonical_cache_consistency_receipt",
    }
    out: dict[str, Path | None] = {}
    for role, attr in names.items():
        value = getattr(paths, attr, None)
        out[role] = Path(value) if value not in (None, "") else None
    return out


def write_canonical_solve_identity_manifest(
    path: Path,
    *,
    run_contract_path: Path,
    source_identity_path: Path | None,
) -> Path | None:
    if source_identity_path is None or not Path(source_identity_path).exists():
        return None
    payload = _read_json_or_none(Path(source_identity_path))
    if payload is None:
        return None
    manifest_payload = dict(payload)
    manifest_payload['run_contract'] = str(run_contract_path)
    manifest_payload['source_identity_path'] = str(source_identity_path)
    manifest_payload['manifest_type'] = 'canonical_solve_identity_manifest'
    write_json(path, manifest_payload)
    return path



def write_final_dem_writer_receipt(
    path: Path,
    *,
    run_contract_path: Path,
    final_dem_path: Path,
    final_writer_mode: str,
    canonical_final_dem_path: Path | None,
    canonical_take_mask_path: Path | None,
) -> Path:
    payload = {
        'run_contract': str(run_contract_path),
        'final_dem_path': str(final_dem_path),
        'final_dem_present': bool(final_dem_path.exists()),
        'final_writer_mode': str(final_writer_mode),
        'canonical_final_dem_path': (str(canonical_final_dem_path) if canonical_final_dem_path is not None else None),
        'canonical_final_dem_present': bool(canonical_final_dem_path is not None and canonical_final_dem_path.exists()),
        'canonical_take_mask_path': (str(canonical_take_mask_path) if canonical_take_mask_path is not None else None),
        'canonical_take_mask_present': bool(canonical_take_mask_path is not None and canonical_take_mask_path.exists()),
    }
    write_json(path, payload)
    return path


def write_river_verification_summary(
    path: Path,
    *,
    run_contract_path: Path,
    canonical_solve_identity_path: Path | None,
    stage_identity_manifest_path: Path | None,
    canonical_cache_consistency_receipt_path: Path | None,
    adjacent_aoi_verification_receipt_path: Path | None,
    final_dem_path: Path | None,
    final_dem_receipt_path: Path | None,
    final_dem_writer_receipt_path: Path | None,
) -> Path:
    stage_identity_present = bool(stage_identity_manifest_path is not None and Path(stage_identity_manifest_path).exists())
    canonical_identity_present = bool(canonical_solve_identity_path is not None and Path(canonical_solve_identity_path).exists())
    final_dem_present = bool(final_dem_path is not None and Path(final_dem_path).exists())
    final_dem_receipt_present = bool(final_dem_receipt_path is not None and Path(final_dem_receipt_path).exists())
    final_dem_writer_receipt_present = bool(final_dem_writer_receipt_path is not None and Path(final_dem_writer_receipt_path).exists())

    cache_payload = _read_json_or_none(canonical_cache_consistency_receipt_path)
    adjacent_payload = _read_json_or_none(adjacent_aoi_verification_receipt_path)
    final_writer_payload = _read_json_or_none(final_dem_writer_receipt_path)

    cache_status = (cache_payload or {}).get('status', 'missing')
    adjacent_status = (adjacent_payload or {}).get('status', 'missing')

    first_diverging_stage_key = (adjacent_payload or {}).get('first_diverging_stage_key') or (cache_payload or {}).get('first_diverging_stage_key')

    status = 'passed'
    reason = 'all_available_verification_checks_passed'
    if not canonical_identity_present:
        status = 'failed'
        reason = 'canonical_solve_identity_missing'
    elif not stage_identity_present:
        status = 'failed'
        reason = 'canonical_stage_identity_missing'
    elif cache_status == 'failed':
        status = 'failed'
        reason = 'canonical_cache_consistency_failed'
    elif adjacent_status == 'failed':
        status = 'failed'
        reason = 'adjacent_aoi_verification_failed'
    elif not final_dem_present:
        status = 'failed'
        reason = 'final_dem_missing'
    elif not final_dem_receipt_present:
        status = 'failed'
        reason = 'final_dem_receipt_missing'
    elif not final_dem_writer_receipt_present:
        status = 'failed'
        reason = 'final_dem_writer_receipt_missing'
    elif cache_status == 'skipped' and adjacent_status in {'skipped', 'missing'}:
        status = 'partial'
        reason = 'adjacent_aoi_verification_not_requested'
    elif adjacent_status == 'skipped':
        status = 'partial'
        reason = 'adjacent_aoi_verification_not_requested'

    payload = {
        'run_contract': str(run_contract_path),
        'status': status,
        'reason': reason,
        'canonical_solve_identity_path': (str(canonical_solve_identity_path) if canonical_solve_identity_path is not None else None),
        'canonical_solve_identity_present': canonical_identity_present,
        'canonical_stage_identity_manifest_path': (str(stage_identity_manifest_path) if stage_identity_manifest_path is not None else None),
        'canonical_stage_identity_present': stage_identity_present,
        'canonical_cache_consistency_receipt_path': (str(canonical_cache_consistency_receipt_path) if canonical_cache_consistency_receipt_path is not None else None),
        'canonical_cache_consistency_status': cache_status,
        'adjacent_aoi_verification_receipt_path': (str(adjacent_aoi_verification_receipt_path) if adjacent_aoi_verification_receipt_path is not None else None),
        'adjacent_aoi_verification_status': adjacent_status,
        'first_diverging_stage_key': first_diverging_stage_key,
        'final_dem_path': (str(final_dem_path) if final_dem_path is not None else None),
        'final_dem_present': final_dem_present,
        'final_dem_receipt_path': (str(final_dem_receipt_path) if final_dem_receipt_path is not None else None),
        'final_dem_receipt_present': final_dem_receipt_present,
        'final_dem_writer_receipt_path': (str(final_dem_writer_receipt_path) if final_dem_writer_receipt_path is not None else None),
        'final_dem_writer_receipt_present': final_dem_writer_receipt_present,
        'summary_checks': {
            'canonical_solve_identity_ok': canonical_identity_present,
            'canonical_stage_identity_ok': stage_identity_present,
            'canonical_cache_consistency_ok': (cache_status == 'passed'),
            'adjacent_aoi_verification_ok': (adjacent_status == 'passed'),
            'final_dem_ok': final_dem_present,
            'final_dem_receipt_ok': final_dem_receipt_present,
            'final_dem_writer_receipt_ok': final_dem_writer_receipt_present,
        },
        'cache_consistency': {
            'status': cache_status,
            'reason': (cache_payload or {}).get('reason'),
            'first_diverging_stage_key': (cache_payload or {}).get('first_diverging_stage_key'),
        },
        'adjacent_aoi': {
            'status': adjacent_status,
            'reason': (adjacent_payload or {}).get('reason'),
            'first_diverging_stage_key': (adjacent_payload or {}).get('first_diverging_stage_key'),
            'solve_identity_ok': (adjacent_payload or {}).get('solve_identity_ok'),
            'stage_identity_ok': (adjacent_payload or {}).get('stage_identity_ok'),
            'cached_canonical_products_ok': (adjacent_payload or {}).get('cached_canonical_products_ok'),
            'final_border_ok': (adjacent_payload or {}).get('final_border_ok'),
        },
        'final_dem_writer': {
            'status': ('passed' if final_dem_writer_receipt_present else 'missing'),
            'final_writer_mode': (final_writer_payload or {}).get('final_writer_mode'),
            'canonical_final_dem_present': (final_writer_payload or {}).get('canonical_final_dem_present'),
            'canonical_take_mask_present': (final_writer_payload or {}).get('canonical_take_mask_present'),
        },
    }
    write_json(path, payload)
    return path

def write_bundle_manifest(
    path: Path,
    *,
    run_contract_path: Path,
    stage_receipts: Mapping[str, Path],
    outputs: Mapping[str, Any],
    stage_trace: list[Mapping[str, Any]] | None = None,
    trace_summary_path: Path | None = None,
) -> Path:
    payload = {
        'run_contract': str(run_contract_path),
        'stage_order': list(WORKFLOW_STAGE_ORDER),
        'stage_receipts': {str(k): str(v) for k, v in stage_receipts.items()},
        'trace_summary_path': str(trace_summary_path) if trace_summary_path is not None else None,
        'stage_trace': [_stringify(dict(entry)) for entry in (stage_trace or [])],
        'outputs': _stringify(dict(outputs)),
    }
    write_json(path, payload)
    return path



def write_wse_proxy_receipt(path: Path, *, inputs: Mapping[str, Any], outputs: Mapping[str, Any], science: Mapping[str, Any]) -> Path:
    """Write compact WSE proxy science evidence without creating extra diagnostics."""
    return write_stage_receipt(
        path,
        stage_name='centerline_wse_proxy',
        inputs=inputs,
        outputs=outputs,
        summary={'river_science': _stringify(dict(science))},
        validator=build_stage_validator_summary('centerline_wse_proxy'),
    )


def write_observed_offset_receipt(path: Path, *, inputs: Mapping[str, Any], outputs: Mapping[str, Any], science: Mapping[str, Any]) -> Path:
    """Write compact observed-offset science evidence."""
    return write_stage_receipt(
        path,
        stage_name='centerline_observed_offset',
        inputs=inputs,
        outputs=outputs,
        summary={'river_science': _stringify(dict(science))},
        validator=build_stage_validator_summary('centerline_observed_offset'),
    )


def write_modeled_offset_receipt(path: Path, *, inputs: Mapping[str, Any], outputs: Mapping[str, Any], science: Mapping[str, Any]) -> Path:
    """Write compact modeled-offset science evidence."""
    return write_stage_receipt(
        path,
        stage_name='centerline_modeled_offset',
        inputs=inputs,
        outputs=outputs,
        summary={'river_science': _stringify(dict(science))},
        validator=build_stage_validator_summary('centerline_modeled_offset'),
    )


def write_backbone_receipt(path: Path, *, inputs: Mapping[str, Any], outputs: Mapping[str, Any], science: Mapping[str, Any]) -> Path:
    """Write compact backbone science evidence."""
    return write_stage_receipt(
        path,
        stage_name='centerline_bed_backbone',
        inputs=inputs,
        outputs=outputs,
        summary={'river_science': _stringify(dict(science))},
        validator=build_stage_validator_summary('centerline_bed_backbone'),
    )



def write_receipt_purpose_manifest(
    path: Path,
    *,
    ctx: Any,
    stage_receipts: Mapping[str, Path | None] | None = None,
    canonical_manifest_path: Path | None = None,
    aoi_export_identity_path: Path | None = None,
    final_output_receipt_path: Path | None = None,
    comparison_report_path: Path | None = None,
    human_run_summary_path: Path | None = None,
) -> Path:
    """Write the single receipt index for the active river route.

    This manifest is intentionally read-only. It does not replace the receipts
    it indexes and it must never drive routing, repair, or DEM selection. Its
    purpose is to make the receipt set understandable: one primary receipt for
    each active purpose, with older/diagnostic receipts clearly identified as
    supporting diagnostics rather than peer workflow products.
    """
    paths = getattr(ctx, "paths", ctx)
    stage_receipts = dict(stage_receipts or {})

    def _record(value: Path | str | None, *, role: str, purpose: str) -> dict[str, Any]:
        p = Path(value) if value not in (None, "") else None
        return {
            "role": role,
            "purpose": purpose,
            "path": str(p) if p is not None else None,
            "exists": bool(p is not None and p.exists()),
        }

    final_output_receipt_path = final_output_receipt_path or (getattr(paths, "root", Path(".")) .parent / "final" / "final_output_receipt.json")
    comparison_report_path = comparison_report_path or (getattr(paths, "root", Path(".")) .parent / "comparison.json")
    human_run_summary_path = human_run_summary_path or (getattr(paths, "root", Path(".")) .parent / "reports" / "run_summary.txt")

    primary_receipts = {
        "canonical_parent_manifest": _record(
            canonical_manifest_path,
            role="canonical_parent_manifest",
            purpose="One canonical parent solution manifest; AOI export should consume this as the parent handoff record.",
        ),
        "aoi_export_identity": _record(
            aoi_export_identity_path or getattr(paths, "aoi_export_receipt", None),
            role="aoi_export_identity",
            purpose="One AOI export identity receipt proving the AOI DEM is an exact subset of the canonical parent.",
        ),
        "final_output_receipt": _record(
            final_output_receipt_path,
            role="final_output_receipt",
            purpose="One final output receipt proving the user-facing DEM was materialized from the named AOI export.",
        ),
        "comparison_report": _record(
            comparison_report_path,
            role="comparison_report",
            purpose="One verify-only north/south or AOI comparison report; this must not repair or reroute outputs.",
        ),
        "human_run_summary": _record(
            human_run_summary_path,
            role="human_run_summary",
            purpose="One human-readable summary of the active route and retained products.",
        ),
    }

    ordered_stage_receipts: dict[str, Any] = {}
    for stage_name in WORKFLOW_STAGE_ORDER:
        rec = stage_receipts.get(stage_name)
        ordered_stage_receipts[stage_name] = _record(
            rec,
            role=f"stage_receipt:{stage_name}",
            purpose="One receipt for this active construction/export stage, when the stage actually runs.",
        )

    diagnostic_receipts = {
        "run_contract": _record(getattr(paths, "run_contract", None), role="run_contract", purpose="Diagnostic route contract; not a second DEM route."),
        "trace_summary": _record(getattr(paths, "trace_summary", None), role="trace_summary", purpose="Diagnostic first-wrong-artifact stage trace."),
        "bundle_manifest": _record(getattr(paths, "bundle_manifest", None), role="bundle_manifest", purpose="Diagnostic bundle index; subordinate to primary receipts."),
        "canonical_stage_identity": _record(getattr(paths, "canonical_stage_identity_manifest", None), role="canonical_stage_identity", purpose="Diagnostic cache/stage hash comparison input."),
        "canonical_cache_consistency": _record(getattr(paths, "canonical_cache_consistency_receipt", None), role="canonical_cache_consistency", purpose="Diagnostic cache consistency check; cache is implementation detail."),
        "adjacent_aoi_verification": _record(getattr(paths, "adjacent_aoi_verification_receipt", None), role="adjacent_aoi_verification", purpose="Optional per-run adjacent-AOI diagnostic; compare.sh remains the primary cross-AOI comparison report."),
        "river_verification_summary": _record(getattr(paths, "river_verification_summary", None), role="river_verification_summary", purpose="Diagnostic rollup of internal checks; not a construction product."),
        "final_dem_writer_receipt": _record(getattr(paths, "final_dem_writer_receipt", None), role="final_dem_writer_receipt", purpose="Internal single-writer diagnostic; final_output_receipt is the primary final product receipt."),
    }

    payload = {
        "schema_version": 1,
        "stage": "receipt_purpose_manifest",
        "role": "receipt_purpose_manifest",
        "active_route": "canonical_river_parent_export",
        "policy": {
            "verify_only": True,
            "does_not_drive_routing": True,
            "does_not_repair_outputs": True,
            "one_primary_receipt_per_purpose": True,
            "cache_is_implementation_detail": True,
        },
        "primary_receipts": _stringify(primary_receipts),
        "stage_receipts": _stringify(ordered_stage_receipts),
        "diagnostic_receipts": _stringify(diagnostic_receipts),
        "first_wrong_artifact_rule": "Use primary_receipts first. If a primary receipt fails, inspect the ordered stage_receipts and stop at the first wrong stage artifact.",
    }
    write_json(path, payload)
    return path

__all__ = [
    'build_stage_trace_entry',
    'build_first_wrong_artifact_guide',
    'build_stage_validator_summary',
    'write_adjacent_aoi_verification_receipt',
    'write_standard_adjacent_aoi_verification_receipt',
    'write_bundle_manifest',
    'write_aoi_export_dem_receipt',
    'write_canonical_identity_receipt',
    'write_canonical_parent_dem_receipt',
    'collect_river_receipts',
    'write_aoi_identity_receipt',
    'write_canonical_solve_identity_manifest',
    'write_canonical_stage_identity_manifest',
    'write_canonical_cache_consistency_receipt',
    'write_final_dem_writer_receipt',
    'write_river_verification_summary',
    'write_receipt_purpose_manifest',
    'write_wse_proxy_receipt',
    'write_observed_offset_receipt',
    'write_modeled_offset_receipt',
    'write_backbone_receipt',
    'write_stage_receipt',
    'write_trace_summary',
]
