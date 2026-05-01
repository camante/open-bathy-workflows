from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import shutil
from typing import Any

import numpy as np
import rasterio

from pipeline.river_linear.river_linear_context import RiverLinearContext
from pipeline.river_linear.river_linear_final_dem_logic import FinalDemCompositionSummary, compose_final_dem
from pipeline.river_linear.river_linear_receipts import write_aoi_export_dem_receipt, write_canonical_parent_dem_receipt
from pipeline.river_linear.river_linear_stage_authoritative import AuthoritativeStageResult
from pipeline.river_linear.river_linear_stage_corridor import CorridorStageResult
from pipeline.river_linear.river_linear_stage_export import ExportStageResult
from pipeline.river_linear.river_linear_stage_lock import LockStageResult
from pipeline.river_linear.river_linear_stage_grids import GridStageResult
from pipeline.river_linear.river_linear_export_logic import build_take_mask, subset_float_raster_to_template_exact
from pipeline.river_linear.river_linear_identity import assert_parent_window_matches_export
from pipeline.river_linear.river_linear_validation import validate_final_dem_raster, validate_raster_matches_template
from pipeline.river_linear.river_linear_cache import build_canonical_cache_metadata, validate_cached_parent_dem
from pipeline.river_linear.river_linear_canonical_manifest import (
    build_canonical_solution_manifest_payload,
    write_canonical_solution_manifest,
)


@dataclass(frozen=True)
class FinalDemStageResult:
    dem_enhanced_final_path: Path
    canonical_parent_dem_path: Path
    aoi_export_dem_path: Path
    canonical_parent_receipt_path: Path
    aoi_export_receipt_path: Path
    support_applied_pixel_count: int
    guidance_applied_pixel_count: int
    background_applied_pixel_count: int
    final_finite_count: int
    final_writer_mode: str
    canonical_final_dem_path: Path | None
    canonical_take_mask_path: Path | None


def _sha256_file(path: Path | str | None) -> str | None:
    if path in (None, ''):
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _raster_identity(path: Path | str | None) -> dict[str, Any]:
    if path in (None, ''):
        return {'path': None, 'exists': False, 'sha256': None}
    p = Path(path)
    out: dict[str, Any] = {'path': str(p), 'exists': p.exists(), 'sha256': _sha256_file(p)}
    if p.exists():
        with rasterio.open(p) as ds:
            out.update({
                'width': int(ds.width),
                'height': int(ds.height),
                'crs': str(ds.crs) if ds.crs is not None else None,
                'transform': [float(x) for x in tuple(ds.transform)[:6]],
                'bounds': [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
                'nodata': None if ds.nodata is None else float(ds.nodata),
            })
    return out


def _template_window_identity(parent_dem: Path, export_template: Path) -> dict[str, Any]:
    """Record how the export grid relates to the canonical parent grid.

    This is diagnostic metadata only; the actual export is still performed by
    subset_float_raster_to_template_exact and then template-validated.
    """
    with rasterio.open(parent_dem) as parent, rasterio.open(export_template) as tmpl:
        try:
            window = parent.window(*tmpl.bounds)
            rounded = window.round_offsets().round_lengths()
            return {
                'parent_path': str(parent_dem),
                'export_template_path': str(export_template),
                'export_bounds': [float(tmpl.bounds.left), float(tmpl.bounds.bottom), float(tmpl.bounds.right), float(tmpl.bounds.top)],
                'window_col_off': int(rounded.col_off),
                'window_row_off': int(rounded.row_off),
                'window_width': int(rounded.width),
                'window_height': int(rounded.height),
            }
        except Exception as exc:  # diagnostic only; do not hide export failures elsewhere
            return {
                'parent_path': str(parent_dem),
                'export_template_path': str(export_template),
                'export_bounds': [float(tmpl.bounds.left), float(tmpl.bounds.bottom), float(tmpl.bounds.right), float(tmpl.bounds.top)],
                'window_error': str(exc),
            }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path



def _valid_mask(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != np.asarray(nodata, dtype=arr.dtype)
    return valid


def _count_mismatch(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[int, float | None]:
    if not np.any(mask):
        return 0, 0.0
    mismatched = mask & ~np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True)
    if not np.any(mismatched):
        return 0, 0.0
    diffs = np.abs(a[mismatched].astype(np.float64) - b[mismatched].astype(np.float64))
    finite_diffs = diffs[np.isfinite(diffs)]
    return int(np.count_nonzero(mismatched)), (float(np.max(finite_diffs)) if finite_diffs.size else None)


def _write_final_dem_contract_receipt(
    path: Path,
    *,
    role: str,
    final_dem_path: Path,
    measured_path: Path,
    support_mask_path: Path,
    background_path: Path,
    guidance_path: Path,
    take_mask_path: Path,
) -> Path:
    """Write and enforce the final DEM authoritative/guidance contract."""
    with rasterio.open(final_dem_path) as final_ds, \
         rasterio.open(measured_path) as measured_ds, \
         rasterio.open(support_mask_path) as support_ds, \
         rasterio.open(background_path) as background_ds, \
         rasterio.open(guidance_path) as guidance_ds, \
         rasterio.open(take_mask_path) as take_ds:
        final_arr = final_ds.read(1)
        measured = measured_ds.read(1)
        support = support_ds.read(1) > 0
        background = background_ds.read(1)
        guidance = guidance_ds.read(1)
        take = take_ds.read(1) > 0
        final_valid = _valid_mask(final_arr, final_ds.nodata)
        measured_valid = _valid_mask(measured, measured_ds.nodata)
        background_valid = _valid_mask(background, background_ds.nodata)
        guidance_valid = _valid_mask(guidance, guidance_ds.nodata)

    support_apply = support & measured_valid
    guidance_apply = take & guidance_valid & ~support_apply
    background_apply = ~(support_apply | guidance_apply)
    expected_valid = support_apply | guidance_apply | background_valid

    measured_mismatch_count, measured_mismatch_max = _count_mismatch(final_arr, measured, support_apply)
    guidance_mismatch_count, guidance_mismatch_max = _count_mismatch(final_arr, guidance, guidance_apply)
    background_mismatch_count, background_mismatch_max = _count_mismatch(final_arr, background, background_apply)
    validity_mismatch = final_valid != expected_valid

    guidance_outside_take = (~take) & guidance_valid & ~support_apply
    guidance_outside_take_changed = guidance_outside_take & ~np.isclose(final_arr, background, rtol=0.0, atol=0.0, equal_nan=True)

    failures: list[str] = []
    if measured_mismatch_count:
        failures.append('measured_authoritative_cells_changed')
    if guidance_mismatch_count:
        failures.append('river_guidance_take_cells_not_written')
    if background_mismatch_count:
        failures.append('background_cells_changed_outside_support_and_take')
    if int(np.count_nonzero(validity_mismatch)):
        failures.append('final_validity_mismatch')
    if int(np.count_nonzero(guidance_outside_take_changed)):
        failures.append('guidance_applied_outside_take_mask')

    payload = {
        'stage': 'final_dem_authoritative_guidance_contract',
        'role': role,
        'status': 'FAIL' if failures else 'PASS',
        'failures': failures,
        'final_dem_path': str(final_dem_path),
        'measured_authoritative_path': str(measured_path),
        'support_mask_path': str(support_mask_path),
        'background_path': str(background_path),
        'river_guidance_path': str(guidance_path),
        'take_mask_path': str(take_mask_path),
        'support_applied_pixel_count': int(np.count_nonzero(support_apply)),
        'guidance_applied_pixel_count': int(np.count_nonzero(guidance_apply)),
        'background_applied_pixel_count': int(np.count_nonzero(background_apply)),
        'final_valid_pixel_count': int(np.count_nonzero(final_valid)),
        'measured_cells_changed_count': int(measured_mismatch_count),
        'measured_cells_changed_max_abs_diff_m': measured_mismatch_max,
        'guidance_take_cells_mismatch_count': int(guidance_mismatch_count),
        'guidance_take_cells_mismatch_max_abs_diff_m': guidance_mismatch_max,
        'background_cells_mismatch_count': int(background_mismatch_count),
        'background_cells_mismatch_max_abs_diff_m': background_mismatch_max,
        'validity_mismatch_pixel_count': int(np.count_nonzero(validity_mismatch)),
        'guidance_valid_outside_take_pixel_count': int(np.count_nonzero(guidance_outside_take)),
        'guidance_applied_outside_take_mask_count': int(np.count_nonzero(guidance_outside_take_changed)),
        'invariants': {
            'authoritative_measured_cells_are_hard_control': measured_mismatch_count == 0,
            'river_guidance_applies_only_inside_take_mask_without_measured_support': int(np.count_nonzero(guidance_outside_take_changed)) == 0,
            'background_preserved_elsewhere': background_mismatch_count == 0,
        },
    }
    _write_json(path, payload)
    if failures:
        raise ValueError(f"final_dem_authoritative_guidance_contract_failed:{role}:{','.join(failures)}")
    return path


def _write_canonical_final_parent_identity(
    ctx: RiverLinearContext,
    *,
    canonical_cache: dict[str, Any],
    canonical_final_dem_path: Path,
    parent_dem: Path,
    authoritative_result: AuthoritativeStageResult,
    lock_result: LockStageResult,
    canonical_take_mask_path: Path,
    summary: FinalDemCompositionSummary,
) -> Path:
    payload = {
        'stage': 'canonical_final_parent_identity',
        'role': 'canonical_final_parent_identity',
        'canonical_system_id': getattr(ctx, 'canonical_system_id', None),
        'canonical_cache': canonical_cache,
        'canonical_inputs_are_export_aoi_independent': True,
        'forbidden_parent_inputs': [
            'export_aoi',
            'aoi_local_baseline_background',
            'aoi_local_authoritative_base',
            'out_dir',
            'run_id',
        ],
        'canonical_solve_grid': _raster_identity(getattr(ctx.linear_inputs, 'canonical_solve_grid_template_path', None) if ctx.linear_inputs is not None else None),
        'canonical_authoritative_measured_only': _raster_identity(authoritative_result.solve_authoritative_base_measured_only_path),
        'canonical_authoritative_support_mask': _raster_identity(authoritative_result.solve_authoritative_support_mask_path),
        'canonical_baseline_background': _raster_identity(authoritative_result.solve_baseline_background_path),
        'canonical_locked_river_surface': _raster_identity(lock_result.river_primary_surface_solve_locked_path),
        'canonical_take_mask': _raster_identity(canonical_take_mask_path),
        'canonical_final_dem': _raster_identity(canonical_final_dem_path),
        'canonical_parent_dem': _raster_identity(parent_dem),
        'canonical_final_parent_hash': _sha256_file(parent_dem),
        'composition_summary': {
            'support_applied_pixel_count': int(summary.support_applied_pixel_count),
            'guidance_applied_pixel_count': int(summary.guidance_applied_pixel_count),
            'background_applied_pixel_count': int(summary.background_applied_pixel_count),
            'final_finite_count': int(summary.final_finite_count),
        },
    }
    path = ctx.paths.receipts_dir / 'canonical_final_parent_identity.json'
    written = _write_json(path, payload)
    public_path = ctx.paths.root.parent / 'reports' / 'canonical_parent_identity.json'
    _write_json(public_path, payload)
    return written


def _write_aoi_export_identity(
    ctx: RiverLinearContext,
    *,
    canonical_cache: dict[str, Any],
    parent_dem: Path,
    aoi_export_dem: Path,
    export_template: Path,
    export_result: ExportStageResult,
) -> Path:
    identity_check = assert_parent_window_matches_export(
        parent_path=parent_dem,
        export_template_path=export_template,
        export_path=aoi_export_dem,
    )
    payload = {
        'schema_version': 2,
        'stage': 'aoi_export_identity',
        'stage_class': 'aoi_export_only',
        'role': 'aoi_export_identity',
        'canonical_system_id': getattr(ctx, 'canonical_system_id', None),
        'canonical_cache': canonical_cache,
        'parent_dem': _raster_identity(parent_dem),
        'aoi_export_dem': _raster_identity(aoi_export_dem),
        'export_template': _raster_identity(export_template),
        'parent_to_export_window': identity_check.get('export_window'),
        'export_window': identity_check.get('export_window'),
        'parent_hash': _sha256_file(parent_dem),
        'export_hash': _sha256_file(aoi_export_dem),
        'export_authoritative_measured_only': _raster_identity(export_result.export_authoritative_base_measured_only_path),
        'export_authoritative_support_mask': _raster_identity(export_result.export_authoritative_support_mask_path),
        'export_baseline_background': _raster_identity(export_result.export_baseline_background_path),
        'export_river_guidance': _raster_identity(export_result.river_guidance_export_path),
        'export_river_take_mask': _raster_identity(export_result.river_take_mask_export_path),
        'export_vs_parent': 'PASS',
        'exact_subset_identity': True,
        'max_abs_diff_m': identity_check.get('max_abs_diff_m'),
        'mismatch_count': identity_check.get('mismatch_count'),
        'post_subset_modifications': False,
        'pixel_values_modified_after_parent_subset': False,
        'resampled_after_parent_subset': False,
        'reprojected_after_parent_subset': False,
        'construction_attempted': False,
        'construction_stages_run_in_aoi_export': False,
        'aoi_outputs_must_be_exact_parent_subsets': True,
        'identity_contract': {
            'same_parent_window_required': True,
            'max_abs_diff_required_m': 0.0,
            'post_subset_modification_allowed': False,
            'aoi_construction_allowed': False,
        },
    }
    path = ctx.paths.receipts_dir / 'aoi_export_identity.json'
    written = _write_json(path, payload)
    public_path = ctx.paths.root.parent / 'reports' / 'aoi_export_identity.json'
    _write_json(public_path, payload)
    return written


def build_canonical_parent_dem(
    ctx: RiverLinearContext,
    authoritative_result: AuthoritativeStageResult,
    corridor_result: CorridorStageResult,
    lock_result: LockStageResult,
) -> tuple[Path, FinalDemCompositionSummary, Path, Path]:
    """Build the canonical parent DEM on the solve grid from current canonical inputs.

    The parent DEM must be independent of export AOI.  It is always rebuilt from
    current canonical measured/support/background/guidance/take-mask artifacts
    so stale final DEM products cannot participate in a new AOI export.
    """
    bundle = ctx.linear_inputs
    canonical_final_dem_path = Path(bundle.canonical_solve_final_dem_path) if bundle is not None and bundle.canonical_solve_final_dem_path is not None else None
    canonical_take_mask_path = Path(bundle.canonical_solve_take_mask_path) if bundle is not None and bundle.canonical_solve_take_mask_path is not None else None
    if canonical_final_dem_path is None or canonical_take_mask_path is None:
        raise ValueError('linear_missing_canonical_final_dem_contract')

    if not canonical_take_mask_path.exists():
        build_take_mask(
            corridor_mask_path=corridor_result.river_corridor_solve_path,
            support_mask_path=authoritative_result.solve_authoritative_support_mask_path,
            dst_path=canonical_take_mask_path,
        )

    summary = compose_final_dem(
        measured_path=authoritative_result.solve_authoritative_base_measured_only_path,
        support_mask_path=authoritative_result.solve_authoritative_support_mask_path,
        background_path=authoritative_result.solve_baseline_background_path,
        guidance_path=lock_result.river_primary_surface_solve_locked_path,
        take_mask_path=canonical_take_mask_path,
        dst_path=canonical_final_dem_path,
    )
    validate_final_dem_raster(
        final_dem_path=canonical_final_dem_path,
        measured_path=authoritative_result.solve_authoritative_base_measured_only_path,
        support_mask_path=authoritative_result.solve_authoritative_support_mask_path,
        background_path=authoritative_result.solve_baseline_background_path,
        guidance_path=lock_result.river_primary_surface_solve_locked_path,
        take_mask_path=canonical_take_mask_path,
        template_path=authoritative_result.solve_authoritative_base_measured_only_path,
    )
    _write_final_dem_contract_receipt(
        ctx.paths.receipts_dir / 'canonical_final_dem_authoritative_guidance_contract.json',
        role='canonical_parent',
        final_dem_path=canonical_final_dem_path,
        measured_path=authoritative_result.solve_authoritative_base_measured_only_path,
        support_mask_path=authoritative_result.solve_authoritative_support_mask_path,
        background_path=authoritative_result.solve_baseline_background_path,
        guidance_path=lock_result.river_primary_surface_solve_locked_path,
        take_mask_path=canonical_take_mask_path,
    )

    parent_dem = ctx.paths.canonical_parent_dem
    parent_dem.parent.mkdir(parents=True, exist_ok=True)
    if canonical_final_dem_path.resolve() != parent_dem.resolve():
        if parent_dem.exists():
            parent_dem.unlink()
        shutil.copy2(canonical_final_dem_path, parent_dem)

    # Existing parent validation is kept after the fresh copy to verify that the
    # per-run parent receipt, if present, still belongs to this canonical cache.
    parent_cache_validation = {
        'status': 'fresh_parent_written',
        'passed': True,
        'reason': 'parent_dem_rebuilt_from_current_canonical_final_dem',
    }
    if ctx.paths.canonical_parent_receipt.exists():
        validate_cached_parent_dem(ctx=ctx, parent_dem=parent_dem, parent_receipt=ctx.paths.canonical_parent_receipt)

    canonical_cache = build_canonical_cache_metadata(ctx)
    _write_canonical_final_parent_identity(
        ctx,
        canonical_cache=canonical_cache,
        canonical_final_dem_path=canonical_final_dem_path,
        parent_dem=parent_dem,
        authoritative_result=authoritative_result,
        lock_result=lock_result,
        canonical_take_mask_path=canonical_take_mask_path,
        summary=summary,
    )
    composition_payload = {
        'support_applied_pixel_count': summary.support_applied_pixel_count,
        'guidance_applied_pixel_count': summary.guidance_applied_pixel_count,
        'background_applied_pixel_count': summary.background_applied_pixel_count,
        'final_finite_count': summary.final_finite_count,
    }
    write_canonical_parent_dem_receipt(
        ctx.paths.canonical_parent_receipt,
        canonical_system_id=getattr(ctx, 'canonical_system_id', None),
        parent_dem=parent_dem,
        source_dem=canonical_final_dem_path,
        user_aoi_cropped=False,
        authoritative_lock_applied=True,
        source_stage_names=[
            'authoritative_inputs',
            'river_corridor_solve',
            'river_primary_surface_solve_locked',
            'canonical_take_mask',
        ],
        canonical_cache=canonical_cache,
        cache_validation=parent_cache_validation,
        summary=composition_payload,
    )
    manifest_payload = build_canonical_solution_manifest_payload(
        ctx=ctx,
        canonical_parent_dem=parent_dem,
        canonical_take_mask_path=canonical_take_mask_path,
        canonical_final_dem_path=canonical_final_dem_path,
        solve_grid_template_path=authoritative_result.solve_authoritative_base_measured_only_path,
        authoritative_measured_path=authoritative_result.solve_authoritative_base_measured_only_path,
        authoritative_support_mask_path=authoritative_result.solve_authoritative_support_mask_path,
        baseline_background_path=authoritative_result.solve_baseline_background_path,
        locked_surface_path=lock_result.river_primary_surface_solve_locked_path,
        run_contract_path=ctx.paths.run_contract,
        composition_summary=composition_payload,
    )
    write_canonical_solution_manifest(ctx=ctx, payload=manifest_payload)
    return parent_dem, summary, canonical_final_dem_path, canonical_take_mask_path


def export_aoi_dem_from_parent(
    ctx: RiverLinearContext,
    grid_result: GridStageResult,
    export_result: ExportStageResult,
    parent_dem: Path,
) -> Path:
    """Export the user AOI DEM as an exact subset of the canonical parent DEM."""
    subset_float_raster_to_template_exact(
        src_path=parent_dem,
        template_path=grid_result.export_grid_template_path,
        dst_path=ctx.paths.aoi_export_dem,
    )
    validate_raster_matches_template(raster_path=ctx.paths.aoi_export_dem, template_path=export_result.export_grid_template_path)
    return ctx.paths.aoi_export_dem


def run_final_dem_stage(
    ctx: RiverLinearContext,
    grid_result: GridStageResult,
    authoritative_result: AuthoritativeStageResult,
    corridor_result: CorridorStageResult,
    lock_result: LockStageResult,
    export_result: ExportStageResult,
) -> FinalDemStageResult:
    parent_dem, summary, canonical_final_dem_path, canonical_take_mask_path = build_canonical_parent_dem(
        ctx, authoritative_result, corridor_result, lock_result
    )
    aoi_export_dem = export_aoi_dem_from_parent(ctx, grid_result, export_result, parent_dem)
    aoi_parent_identity = assert_parent_window_matches_export(
        parent_path=parent_dem,
        export_template_path=grid_result.export_grid_template_path,
        export_path=aoi_export_dem,
    )
    validate_final_dem_raster(
        final_dem_path=aoi_export_dem,
        measured_path=export_result.export_authoritative_base_measured_only_path,
        support_mask_path=export_result.export_authoritative_support_mask_path,
        background_path=export_result.export_baseline_background_path,
        guidance_path=export_result.river_guidance_export_path,
        take_mask_path=export_result.river_take_mask_export_path,
        template_path=export_result.export_grid_template_path,
    )
    _write_final_dem_contract_receipt(
        ctx.paths.receipts_dir / 'aoi_final_dem_authoritative_guidance_contract.json',
        role='aoi_export',
        final_dem_path=aoi_export_dem,
        measured_path=export_result.export_authoritative_base_measured_only_path,
        support_mask_path=export_result.export_authoritative_support_mask_path,
        background_path=export_result.export_baseline_background_path,
        guidance_path=export_result.river_guidance_export_path,
        take_mask_path=export_result.river_take_mask_export_path,
    )
    canonical_cache = build_canonical_cache_metadata(ctx)
    _write_aoi_export_identity(
        ctx,
        canonical_cache=canonical_cache,
        parent_dem=parent_dem,
        aoi_export_dem=aoi_export_dem,
        export_template=grid_result.export_grid_template_path,
        export_result=export_result,
    )
    write_aoi_export_dem_receipt(
        ctx.paths.aoi_export_receipt,
        canonical_system_id=getattr(ctx, 'canonical_system_id', None),
        parent_dem=parent_dem,
        export_dem=aoi_export_dem,
        export_template=grid_result.export_grid_template_path,
        pixel_values_modified=False,
        resampled=False,
        reprojected=False,
        canonical_cache=canonical_cache,
        summary={
            'support_applied_pixel_count': summary.support_applied_pixel_count,
            'guidance_applied_pixel_count': summary.guidance_applied_pixel_count,
            'background_applied_pixel_count': summary.background_applied_pixel_count,
            'final_finite_count': summary.final_finite_count,
            'aoi_export_identity': aoi_parent_identity,
            'post_subset_modifications': False,
            'construction_attempted_in_aoi_export': False,
        },
    )
    return FinalDemStageResult(
        dem_enhanced_final_path=aoi_export_dem,
        canonical_parent_dem_path=parent_dem,
        aoi_export_dem_path=aoi_export_dem,
        canonical_parent_receipt_path=ctx.paths.canonical_parent_receipt,
        aoi_export_receipt_path=ctx.paths.aoi_export_receipt,
        support_applied_pixel_count=summary.support_applied_pixel_count,
        guidance_applied_pixel_count=summary.guidance_applied_pixel_count,
        background_applied_pixel_count=summary.background_applied_pixel_count,
        final_finite_count=summary.final_finite_count,
        final_writer_mode='canonical_parent_exact_aoi_export',
        canonical_final_dem_path=canonical_final_dem_path,
        canonical_take_mask_path=canonical_take_mask_path,
    )


__all__ = ["FinalDemStageResult", "build_canonical_parent_dem", "export_aoi_dem_from_parent", "run_final_dem_stage"]
