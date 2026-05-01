from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio

from authoritative_guidance import (
    build_projected_authoritative_sampling_raster,
    build_projected_measured_only_authoritative_sampling_raster,
)
from legacy.river.river_v2_context import RiverV2Context
from legacy.river.river_v2_context_compat import apply_direct_context_compatibility_defaults
from legacy.river.river_v2_execution_contract import RiverV2ExecutionContract, build_river_v2_execution_contract
from legacy.river.river_v2_resolved_inputs import ResolvedRiverV2Inputs, write_resolved_inputs_manifest, resolved_inputs_to_dict
from legacy.river.river_v2_validation import validate_resolved_river_v2_inputs
from core.json_io import write_json


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_river_v2_preflight_receipt(receipt: Dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True), encoding='utf-8')
    return out_path


def resolve_sampling_aoi(ctx: RiverV2Context) -> str:
    solve_aoi = getattr(ctx, 'canonical_solve_aoi', None)
    if solve_aoi:
        return str(solve_aoi)
    cfg_aoi = getattr(ctx.cfg, 'aoi', None)
    if cfg_aoi:
        return str(cfg_aoi)
    if ctx.channel_mask_path is not None and Path(ctx.channel_mask_path).exists():
        with rasterio.open(ctx.channel_mask_path) as ds:
            b = ds.bounds
            return f"{b.left}/{b.right}/{b.bottom}/{b.top}"
    if ctx.authoritative_base_path is not None and Path(ctx.authoritative_base_path).exists():
        with rasterio.open(ctx.authoritative_base_path) as ds:
            b = ds.bounds
            return f"{b.left}/{b.right}/{b.bottom}/{b.top}"
    raise RuntimeError('river_v2_authoritative_sampling_missing_aoi')


def raster_is_projected(path: Path) -> bool:
    try:
        with rasterio.open(path) as ds:
            return bool(ds.crs is not None and not getattr(ds.crs, 'is_geographic', False))
    except Exception:
        return False


def _rewrite_local_artifact_from_source(*, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        rel = os.path.relpath(source, destination.parent)
        destination.symlink_to(rel)
    except Exception:
        shutil.copy2(source, destination)


def ensure_authoritative_sampling_support(ctx: RiverV2Context) -> Path | None:
    if ctx.authoritative_sampling_raster_path is not None and Path(ctx.authoritative_sampling_raster_path).exists():
        return Path(ctx.authoritative_sampling_raster_path)
    source_raster = None
    if getattr(ctx, 'canonical_authoritative_sampling_source_raster_path', None) is not None and Path(ctx.canonical_authoritative_sampling_source_raster_path).exists():
        source_raster = Path(ctx.canonical_authoritative_sampling_source_raster_path)
    elif ctx.authoritative_sampling_source_raster_path is not None and Path(ctx.authoritative_sampling_source_raster_path).exists():
        source_raster = Path(ctx.authoritative_sampling_source_raster_path)
    elif ctx.authoritative_base_path is not None and Path(ctx.authoritative_base_path).exists():
        source_raster = Path(ctx.authoritative_base_path)
    elif ctx.aligned_authoritative_base_path is not None and Path(ctx.aligned_authoritative_base_path).exists():
        source_raster = Path(ctx.aligned_authoritative_base_path)
    if source_raster is None:
        return None

    trusted_mode = str(getattr(ctx, 'trusted_support_mode', 'low_support_no_trusted_support') or 'low_support_no_trusted_support')
    trusted_artifact = Path(ctx.trusted_support_artifact_path) if getattr(ctx, 'trusted_support_artifact_path', None) is not None else None
    if trusted_mode == 'low_support_no_trusted_support' and trusted_artifact is not None and trusted_artifact.exists():
        trusted_mode = 'metadata_proven_only'
        ctx.trusted_support_mode = trusted_mode
        if not getattr(ctx, 'support_policy_source', None):
            ctx.support_policy_source = 'preflight_promoted_existing_support_artifact'
    support_dir = ctx.paths.support_dir
    support_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.paths.authoritative_measured_only_projected
    if out_path.exists() and out_path.stat().st_size > 0:
        ctx.authoritative_sampling_raster_path = out_path
        return out_path

    logger = logging.getLogger(__name__)
    with rasterio.open(ctx.river_dem_path) as river_ds:
        dst_crs = river_ds.crs.to_string() if river_ds.crs else None
        res_m = abs(float(river_ds.transform.a)) if river_ds.transform else None
    if not dst_crs or not np.isfinite(float(res_m)) or float(res_m) <= 0:
        raise RuntimeError('river_v2_authoritative_sampling_missing_projected_grid')

    support_path = None
    probe_only = False
    if trusted_mode == 'low_support_no_trusted_support':
        info = build_projected_authoritative_sampling_raster(
            source_raster,
            out_path,
            aoi=resolve_sampling_aoi(ctx),
            dst_crs=str(dst_crs),
            res_m=float(res_m),
            logger=logger,
        )
        sampling_contract = 'solve_domain_sampling_source_projected_low_support_probe'
        probe_only = True
    elif trusted_mode == 'all_finite_cells_trusted':
        info = build_projected_authoritative_sampling_raster(
            source_raster,
            out_path,
            aoi=resolve_sampling_aoi(ctx),
            dst_crs=str(dst_crs),
            res_m=float(res_m),
            logger=logger,
        )
        sampling_contract = 'all_finite_cells_trusted_projected'
    elif trusted_mode == 'metadata_proven_only':
        support_coverage = trusted_artifact if trusted_artifact is not None and trusted_artifact.exists() else None
        if support_coverage is None:
            write_json(ctx.paths.authoritative_sampling_summary, {
                'status': 'skipped',
                'sampling_raster_path': None,
                'sampling_contract': 'metadata_proven_only_missing_support_artifact',
                'source_raster': str(source_raster),
                'support_coverage_path': str(trusted_artifact) if trusted_artifact else None,
                'trusted_support_mode': trusted_mode,
                'warning': 'trusted_support_artifact_missing',
            })
            return None
        info = build_projected_measured_only_authoritative_sampling_raster(
            source_raster,
            support_coverage,
            out_path,
            aoi=resolve_sampling_aoi(ctx),
            dst_crs=str(dst_crs),
            res_m=float(res_m),
            logger=logger,
        )
        sampling_contract = 'metadata_proven_only_projected_masked_by_support_coverage'
        support_path = support_coverage
    else:
        raise RuntimeError(f'river_v2_unknown_trusted_support_mode:{trusted_mode}')

    ctx.authoritative_sampling_raster_path = out_path if out_path.exists() else None
    write_json(ctx.paths.authoritative_sampling_summary, {
        'status': 'success' if ctx.authoritative_sampling_raster_path else 'failed',
        'sampling_raster_path': str(ctx.authoritative_sampling_raster_path) if ctx.authoritative_sampling_raster_path else None,
        'sampling_contract': sampling_contract,
        'source_raster': str(source_raster),
        'support_coverage_path': str(support_path) if support_path else None,
        'trusted_support_mode': trusted_mode,
        'source_kind': 'authoritative_raster',
        'fallback_used': False,
        'probe_only': bool(probe_only),
        'warning': getattr(ctx, 'authoritative_support_policy_warning', None),
        'projection_info': info,
    })
    return ctx.authoritative_sampling_raster_path


def resolve_authoritative_bed_support_source(ctx: RiverV2Context) -> Path | None:
    candidates = []
    if ctx.authoritative_bed_path is not None:
        candidates.append(Path(ctx.authoritative_bed_path))
    outputs = ctx.report.get('outputs', {}) if isinstance(ctx.report, dict) else {}
    if isinstance(outputs, dict):
        for key in ('river_authoritative_soundings', 'river_authoritative_bed'):
            value = outputs.get(key)
            if value:
                candidates.append(Path(value))
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def ensure_authoritative_bed_support_points(ctx: RiverV2Context) -> Path | None:
    existing = ctx.paths.authoritative_bed_support_points
    if existing.exists() and existing.stat().st_size > 0:
        ctx.authoritative_bed_path = existing
        return existing
    source = resolve_authoritative_bed_support_source(ctx)
    if source is None:
        return None
    existing.parent.mkdir(parents=True, exist_ok=True)
    if existing.exists() or existing.is_symlink():
        existing.unlink()
    try:
        rel = os.path.relpath(source, existing.parent)
        existing.symlink_to(rel)
    except Exception:
        shutil.copy2(source, existing)
    ctx.authoritative_bed_path = existing
    return existing


def ensure_wse_support_source_raster(ctx: RiverV2Context) -> tuple[Path | None, str | None]:
    out_path = ctx.paths.wse_support_source_raster
    candidates: list[tuple[Path | None, str]] = [
        (Path(ctx.canonical_authoritative_sampling_source_raster_path), 'solve_domain_authoritative_sampling_source') if ctx.canonical_authoritative_sampling_source_raster_path is not None else (None, 'solve_domain_authoritative_sampling_source'),
        (Path(ctx.authoritative_sampling_source_raster_path), 'legacy_authoritative_sampling_source') if ctx.authoritative_sampling_source_raster_path is not None else (None, 'legacy_authoritative_sampling_source'),
        (Path(ctx.river_dem_path), 'river_dem_template') if ctx.river_dem_path is not None else (None, 'river_dem_template'),
        (Path(ctx.authoritative_base_path), 'authoritative_base') if ctx.authoritative_base_path is not None else (None, 'authoritative_base'),
        (Path(ctx.aligned_authoritative_base_path), 'aligned_authoritative_base') if ctx.aligned_authoritative_base_path is not None else (None, 'aligned_authoritative_base'),
        (Path(ctx.authoritative_sampling_raster_path), 'measured_only_authoritative_sampling_raster') if ctx.authoritative_sampling_raster_path is not None else (None, 'measured_only_authoritative_sampling_raster'),
    ]
    for source, contract in candidates:
        if source is None or not source.exists():
            continue
        _rewrite_local_artifact_from_source(source=source, destination=out_path)
        return out_path, contract
    return None, None


def resolve_authoritative_lock_base(ctx: RiverV2Context) -> Path | None:
    if ctx.authoritative_base_path is not None and Path(ctx.authoritative_base_path).exists():
        return Path(ctx.authoritative_base_path)
    if ctx.aligned_authoritative_base_path is not None and Path(ctx.aligned_authoritative_base_path).exists():
        return Path(ctx.aligned_authoritative_base_path)
    return None


def resolve_centerline_parameters(ctx: RiverV2Context) -> tuple[float, int]:
    spacing_m = float(ctx.centerline_spacing_m or 25.0)
    min_stream_order = max(int(getattr(ctx.cfg, 'river_mainstem_min_order', 5) or 5), 1)
    return spacing_m, min_stream_order


def build_resolved_river_v2_inputs(
    ctx: RiverV2Context,
    *,
    authoritative_sampling_raster_path: Path | None,
    authoritative_bed_support_points_path: Path | None,
    wse_support_source_raster_path: Path | None,
    wse_support_source_contract: str | None,
) -> ResolvedRiverV2Inputs:
    centerline_spacing_m, min_stream_order = resolve_centerline_parameters(ctx)
    return ResolvedRiverV2Inputs(
        network_gpkg=Path(ctx.canonical_network_gpkg) if ctx.canonical_network_gpkg is not None else None,
        river_dem_path=Path(ctx.river_dem_path) if ctx.river_dem_path is not None else None,
        solve_network_gpkg=Path(ctx.canonical_network_gpkg) if ctx.canonical_network_gpkg is not None else None,
        authoritative_sampling_source_raster_path=Path(ctx.canonical_authoritative_sampling_source_raster_path) if ctx.canonical_authoritative_sampling_source_raster_path is not None else None,
        authoritative_sampling_raster_path=Path(authoritative_sampling_raster_path) if authoritative_sampling_raster_path is not None else None,
        centerline_spacing_m=centerline_spacing_m,
        min_stream_order=min_stream_order,
        authoritative_bed_support_points_path=Path(authoritative_bed_support_points_path) if authoritative_bed_support_points_path is not None else None,
        channel_mask_path=Path(ctx.canonical_channel_mask_path) if ctx.canonical_channel_mask_path is not None else None,
        export_channel_mask_path=Path(ctx.export_channel_mask_path) if ctx.export_channel_mask_path is not None else None,
        authoritative_lock_base_path=Path(ctx.canonical_authoritative_base_path) if ctx.canonical_authoritative_base_path is not None else None,
        solve_channel_mask_path=Path(ctx.canonical_channel_mask_path) if ctx.canonical_channel_mask_path is not None else None,
        solve_authoritative_sampling_source_raster_path=Path(ctx.canonical_authoritative_sampling_source_raster_path) if ctx.canonical_authoritative_sampling_source_raster_path is not None else None,
        solve_authoritative_lock_base_path=Path(ctx.canonical_authoritative_base_path) if ctx.canonical_authoritative_base_path is not None else None,
        wse_support_source_raster_path=Path(wse_support_source_raster_path) if wse_support_source_raster_path is not None else None,
        wse_support_source_contract=wse_support_source_contract,
        solve_aoi=resolve_sampling_aoi(ctx),
        trusted_support_mode=str(getattr(ctx, 'trusted_support_mode', 'unknown') or 'unknown'),
        support_policy_source=str(getattr(ctx, 'support_policy_source', None)) if getattr(ctx, 'support_policy_source', None) is not None else None,
        vertical_reference=str(getattr(ctx, 'vertical_reference', 'unknown') or 'unknown'),
    )


def build_river_v2_preflight_receipt(
    ctx: RiverV2Context,
    *,
    inputs: ResolvedRiverV2Inputs,
    validation: Dict[str, Any],
    warnings: list[str],
) -> Dict[str, Any]:
    source_raster = None
    if getattr(ctx, 'canonical_authoritative_sampling_source_raster_path', None) is not None and Path(ctx.canonical_authoritative_sampling_source_raster_path).exists():
        source_raster = Path(ctx.canonical_authoritative_sampling_source_raster_path)
    elif ctx.authoritative_sampling_source_raster_path is not None and Path(ctx.authoritative_sampling_source_raster_path).exists():
        source_raster = Path(ctx.authoritative_sampling_source_raster_path)
    elif ctx.authoritative_base_path is not None and Path(ctx.authoritative_base_path).exists():
        source_raster = Path(ctx.authoritative_base_path)
    elif ctx.aligned_authoritative_base_path is not None and Path(ctx.aligned_authoritative_base_path).exists():
        source_raster = Path(ctx.aligned_authoritative_base_path)
    return {
        'status': 'success' if bool(validation.get('valid', False)) else 'failed',
        'resolved_inputs_manifest': str(ctx.paths.resolved_inputs_manifest),
        'validation': validation,
        'warnings': list(warnings),
        'source_resolution': {
            'solve_aoi': inputs.solve_aoi,
            'authoritative_sampling_source_raster': str(source_raster) if source_raster else None,
            'authoritative_sampling_source_summary_path': str(ctx.paths.authoritative_sampling_source_summary) if ctx.paths.authoritative_sampling_source_summary.exists() else None,
            'authoritative_sampling_raster_path': str(inputs.authoritative_sampling_raster_path) if inputs.authoritative_sampling_raster_path else None,
            'authoritative_sampling_summary_path': str(ctx.paths.authoritative_sampling_summary) if ctx.paths.authoritative_sampling_summary.exists() else None,
            'authoritative_bed_support_points_path': str(inputs.authoritative_bed_support_points_path) if inputs.authoritative_bed_support_points_path else None,
            'wse_support_source_raster_path': str(inputs.wse_support_source_raster_path) if inputs.wse_support_source_raster_path else None,
            'wse_support_source_contract': inputs.wse_support_source_contract,
            'authoritative_lock_base_path': str(inputs.authoritative_lock_base_path) if inputs.authoritative_lock_base_path else None,
            'solve_network_gpkg': str(inputs.solve_network_gpkg) if inputs.solve_network_gpkg else None,
            'solve_channel_mask_path': str(inputs.solve_channel_mask_path) if inputs.solve_channel_mask_path else None,
            'solve_authoritative_sampling_source_raster_path': str(inputs.solve_authoritative_sampling_source_raster_path) if inputs.solve_authoritative_sampling_source_raster_path else None,
            'solve_authoritative_lock_base_path': str(inputs.solve_authoritative_lock_base_path) if inputs.solve_authoritative_lock_base_path else None,
            'canonical_inputs_receipt_path': str(ctx.canonical_inputs_receipt_path) if getattr(ctx, 'canonical_inputs_receipt_path', None) else None,
            'trusted_support_mode': inputs.trusted_support_mode,
            'support_policy_source': inputs.support_policy_source,
        },
        'resolved_inputs': resolved_inputs_to_dict(inputs),
        'first_wrong_artifact': validation.get('first_wrong_artifact'),
    }




def _first_wrong_artifact(*, inputs: ResolvedRiverV2Inputs, validation: Dict[str, Any], ctx: RiverV2Context) -> Dict[str, Any] | None:
    missing = list(validation.get('missing_required_fields') or [])
    if not validation.get('solve_aoi_valid', False):
        return {
            'artifact': 'solve_aoi',
            'path': getattr(inputs, 'solve_aoi', None),
            'expected_stage': 'canonical_solve_domain',
            'blocked_stage': 'river_centerline',
            'reason': 'missing_or_empty_canonical_solve_aoi',
        }
    if 'solve_network_gpkg' in missing or not validation.get('required_path_exists', {}).get('solve_network_gpkg', False):
        return {
            'artifact': 'solve_network_gpkg',
            'path': str(getattr(inputs, 'solve_network_gpkg', None)) if getattr(inputs, 'solve_network_gpkg', None) else None,
            'expected_stage': 'canonical_solve_domain',
            'blocked_stage': 'river_centerline',
            'reason': 'missing_canonical_solve_network',
        }
    if 'solve_channel_mask_path' in missing or not validation.get('required_path_exists', {}).get('solve_channel_mask_path', False):
        return {
            'artifact': 'solve_channel_mask_path',
            'path': str(getattr(inputs, 'solve_channel_mask_path', None)) if getattr(inputs, 'solve_channel_mask_path', None) else None,
            'expected_stage': 'canonical_inputs_materialization',
            'blocked_stage': 'river_primary_surface',
            'reason': 'missing_canonical_solve_channel_mask',
        }
    if str(getattr(ctx, 'system_support_status', 'unknown') or 'unknown') == 'authoritative_control_found' and getattr(inputs, 'solve_authoritative_lock_base_path', None) is None:
        return {
            'artifact': 'solve_authoritative_lock_base_path',
            'path': None,
            'expected_stage': 'canonical_inputs_materialization',
            'blocked_stage': 'river_primary_surface_authoritative_applied_solve_domain',
            'reason': 'authoritative_mode_requires_canonical_authoritative_base',
        }
    return None


def _raise_preflight_validation_error(*, validation: Dict[str, Any]):
    first_wrong = validation.get('first_wrong_artifact') or {}
    artifact = first_wrong.get('artifact')
    path = first_wrong.get('path')
    expected_stage = first_wrong.get('expected_stage')
    blocked_stage = first_wrong.get('blocked_stage')
    reason = first_wrong.get('reason')
    if artifact:
        raise RuntimeError(
            f"river_v2_preflight_invalid:{artifact}:expected_stage={expected_stage}:blocked_stage={blocked_stage}:reason={reason}:path={path}"
        )
    missing = validation.get('missing_required_fields') or []
    if missing:
        raise RuntimeError('river_v2_preflight_missing_required_inputs:' + ','.join(str(item) for item in missing))
    raise RuntimeError('river_v2_preflight_invalid_resolved_inputs')

def run_river_v2_preflight(ctx: RiverV2Context) -> ResolvedRiverV2Inputs:
    apply_direct_context_compatibility_defaults(ctx)
    warnings: list[str] = []
    authoritative_sampling_raster_path = ensure_authoritative_sampling_support(ctx)
    authoritative_bed_support_points_path = ensure_authoritative_bed_support_points(ctx)
    wse_support_source_raster_path, wse_support_source_contract = ensure_wse_support_source_raster(ctx)
    inputs = build_resolved_river_v2_inputs(
        ctx,
        authoritative_sampling_raster_path=authoritative_sampling_raster_path,
        authoritative_bed_support_points_path=authoritative_bed_support_points_path,
        wse_support_source_raster_path=wse_support_source_raster_path,
        wse_support_source_contract=wse_support_source_contract,
    )
    manifest_path = write_resolved_inputs_manifest(inputs, ctx.paths.resolved_inputs_manifest)
    validation = validate_resolved_river_v2_inputs(inputs)
    if str(getattr(ctx, "system_support_status", "unknown") or "unknown") == "authoritative_control_found" and inputs.solve_authoritative_lock_base_path is None:
        validation["valid"] = False
        missing = validation.setdefault("missing_required_fields", [])
        if "solve_authoritative_lock_base_path" not in missing:
            missing.append("solve_authoritative_lock_base_path")
    validation["first_wrong_artifact"] = _first_wrong_artifact(inputs=inputs, validation=validation, ctx=ctx)
    if not validation.get('valid', False):
        missing = validation.get('missing_required_fields') or []
        if missing:
            warnings.append('missing_required_fields:' + ','.join(str(item) for item in missing))
    receipt = build_river_v2_preflight_receipt(ctx, inputs=inputs, validation=validation, warnings=warnings)
    receipt_path = write_river_v2_preflight_receipt(receipt, ctx.paths.preflight_receipt)
    ctx.remember_preflight_artifacts(resolved_inputs_manifest_path=manifest_path, preflight_receipt_path=receipt_path)
    if not validation.get('valid', False):
        _raise_preflight_validation_error(validation=validation)
    return inputs


def run_river_v2_execution_contract_preflight(ctx: RiverV2Context) -> RiverV2ExecutionContract:
    inputs = run_river_v2_preflight(ctx)
    return build_river_v2_execution_contract(ctx, inputs)


__all__ = [
    'ResolvedRiverV2Inputs',
    'run_river_v2_preflight',
    'run_river_v2_execution_contract_preflight',
    'resolve_sampling_aoi',
    'raster_is_projected',
    'ensure_authoritative_sampling_support',
    'resolve_authoritative_bed_support_source',
    'ensure_authoritative_bed_support_points',
    'ensure_wse_support_source_raster',
    'resolve_authoritative_lock_base',
    'resolve_centerline_parameters',
    'build_resolved_river_v2_inputs',
    'build_river_v2_preflight_receipt',
    'write_river_v2_preflight_receipt',
]
