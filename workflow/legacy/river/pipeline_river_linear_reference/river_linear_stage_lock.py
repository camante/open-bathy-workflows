from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import rasterio

from pipeline.river_linear.river_linear_context import RiverLinearContext
from pipeline.river_linear.river_linear_stage_authoritative import AuthoritativeStageResult
from pipeline.river_linear.river_linear_stage_grids import GridStageResult
from pipeline.river_linear.river_linear_stage_surface import SurfaceStageResult
from pipeline.river_linear.river_linear_validation import validate_locked_surface_raster


@dataclass(frozen=True)
class LockStageResult:
    river_primary_surface_solve_locked_path: Path
    locked_pixel_count: int
    finite_locked_surface_count: int
    authoritative_lock_contract_path: Path | None = None
    measured_cells_changed_count: int = 0
    changed_outside_authoritative_lock_count: int = 0


def _as_float_or_none(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    try:
        if not np.isfinite(value):
            return None
        return float(value)
    except TypeError:
        return None


def _write_lock_contract_receipt(
    path: Path,
    *,
    locked_path: Path,
    unlocked_path: Path,
    measured_path: Path,
    support_mask_path: Path,
    support_mask: np.ndarray,
    measured_valid: np.ndarray,
    lock_mask: np.ndarray,
    locked: np.ndarray,
    unlocked: np.ndarray,
    measured: np.ndarray,
    locked_nodata: float | None,
) -> Path:
    measured_diff = np.zeros_like(locked, dtype=np.float32)
    if np.any(lock_mask):
        measured_diff[lock_mask] = np.abs(locked[lock_mask].astype(np.float32) - measured[lock_mask].astype(np.float32))
        measured_changed = lock_mask & ~np.isclose(locked, measured, rtol=0.0, atol=0.0, equal_nan=False)
    else:
        measured_changed = np.zeros_like(lock_mask, dtype=bool)

    outside_lock_mask = ~lock_mask
    if np.any(outside_lock_mask):
        outside_changed = outside_lock_mask & ~np.isclose(locked, unlocked, rtol=0.0, atol=0.0, equal_nan=True)
    else:
        outside_changed = np.zeros_like(lock_mask, dtype=bool)

    locked_valid = np.isfinite(locked)
    if locked_nodata is not None:
        locked_valid &= locked != np.float32(locked_nodata)

    payload = {
        'stage': 'river_authoritative_lock_contract',
        'status': 'PASS' if not np.any(measured_changed) and not np.any(outside_changed) else 'FAIL',
        'locked_surface_path': str(locked_path),
        'unlocked_surface_path': str(unlocked_path),
        'measured_authoritative_path': str(measured_path),
        'support_mask_path': str(support_mask_path),
        'support_pixel_count': int(np.count_nonzero(support_mask)),
        'measured_valid_pixel_count': int(np.count_nonzero(measured_valid)),
        'authoritative_lock_pixel_count': int(np.count_nonzero(lock_mask)),
        'locked_surface_valid_pixel_count': int(np.count_nonzero(locked_valid)),
        'measured_cells_changed_count': int(np.count_nonzero(measured_changed)),
        'measured_cells_changed_max_abs_diff_m': _as_float_or_none(np.nanmax(measured_diff[lock_mask]) if np.any(lock_mask) else 0.0),
        'changed_outside_authoritative_lock_count': int(np.count_nonzero(outside_changed)),
        'invariants': {
            'measured_authoritative_cells_are_hard_control': bool(not np.any(measured_changed)),
            'non_authoritative_pixels_preserve_unlocked_surface': bool(not np.any(outside_changed)),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if payload['status'] != 'PASS':
        raise ValueError('river_authoritative_lock_contract_failed')
    return path



def write_authoritative_lock_contract_for_paths(
    *,
    receipt_path: Path,
    locked_raster_path: Path,
    unlocked_raster_path: Path,
    measured_path: Path,
    support_mask_path: Path,
) -> Path:
    """Validate and receipt authoritative-lock semantics for fresh or cached surfaces."""
    with rasterio.open(locked_raster_path) as locked_ds, \
         rasterio.open(unlocked_raster_path) as unlocked_ds, \
         rasterio.open(measured_path) as measured_ds, \
         rasterio.open(support_mask_path) as support_ds:
        locked = locked_ds.read(1).astype(np.float32)
        unlocked = unlocked_ds.read(1).astype(np.float32)
        measured = measured_ds.read(1).astype(np.float32)
        support = support_ds.read(1)
        measured_valid = np.isfinite(measured)
        if measured_ds.nodata is not None:
            measured_valid &= measured != np.float32(measured_ds.nodata)
        support_mask = support > 0
        lock_mask = support_mask & measured_valid
        return _write_lock_contract_receipt(
            receipt_path,
            locked_path=locked_raster_path,
            unlocked_path=unlocked_raster_path,
            measured_path=measured_path,
            support_mask_path=support_mask_path,
            support_mask=support_mask,
            measured_valid=measured_valid,
            lock_mask=lock_mask,
            locked=locked,
            unlocked=unlocked,
            measured=measured,
            locked_nodata=locked_ds.nodata,
        )
def run_lock_stage(
    ctx: RiverLinearContext,
    grid_result: GridStageResult,
    authoritative_result: AuthoritativeStageResult,
    surface_result: SurfaceStageResult,
) -> LockStageResult:
    with rasterio.open(surface_result.river_primary_surface_solve_path) as surface_ds, \
         rasterio.open(authoritative_result.solve_authoritative_base_measured_only_path) as measured_ds, \
         rasterio.open(authoritative_result.solve_authoritative_support_mask_path) as support_ds:
        surface = surface_ds.read(1).astype(np.float32)
        measured = measured_ds.read(1).astype(np.float32)
        support = support_ds.read(1)
        out = surface.copy()
        surface_nodata = np.float32(surface_ds.nodata if surface_ds.nodata is not None else -9999.0)
        measured_nodata = measured_ds.nodata
        support_mask = support > 0
        measured_valid = np.isfinite(measured)
        if measured_nodata is not None:
            measured_valid &= measured != np.float32(measured_nodata)
        lock_mask = support_mask & measured_valid
        out[lock_mask] = measured[lock_mask]
        profile = surface_ds.profile.copy()
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
        profile.pop('BLOCKXSIZE', None)
        profile.pop('BLOCKYSIZE', None)
        profile.update(dtype='float32', nodata=surface_nodata, count=1, compress='deflate')
        ctx.paths.river_primary_surface_solve_locked.parent.mkdir(parents=True, exist_ok=True)
        if ctx.paths.river_primary_surface_solve_locked.exists():
            ctx.paths.river_primary_surface_solve_locked.unlink()
        with rasterio.open(ctx.paths.river_primary_surface_solve_locked, 'w', **profile) as out_ds:
            out_ds.write(out, 1)
    validate_locked_surface_raster(
        locked_raster_path=ctx.paths.river_primary_surface_solve_locked,
        unlocked_raster_path=surface_result.river_primary_surface_solve_path,
        measured_path=authoritative_result.solve_authoritative_base_measured_only_path,
        support_mask_path=authoritative_result.solve_authoritative_support_mask_path,
        template_path=grid_result.solve_grid_template_path,
    )
    contract_path = _write_lock_contract_receipt(
        ctx.paths.receipts_dir / 'authoritative_lock_contract.json',
        locked_path=ctx.paths.river_primary_surface_solve_locked,
        unlocked_path=surface_result.river_primary_surface_solve_path,
        measured_path=authoritative_result.solve_authoritative_base_measured_only_path,
        support_mask_path=authoritative_result.solve_authoritative_support_mask_path,
        support_mask=support_mask,
        measured_valid=measured_valid,
        lock_mask=lock_mask,
        locked=out,
        unlocked=surface,
        measured=measured,
        locked_nodata=float(surface_nodata),
    )
    return LockStageResult(
        river_primary_surface_solve_locked_path=ctx.paths.river_primary_surface_solve_locked,
        locked_pixel_count=int(np.count_nonzero(lock_mask)),
        finite_locked_surface_count=int(np.count_nonzero(out != surface_nodata)),
        authoritative_lock_contract_path=contract_path,
        measured_cells_changed_count=0,
        changed_outside_authoritative_lock_count=0,
    )


__all__ = ['LockStageResult', 'run_lock_stage', 'write_authoritative_lock_contract_for_paths']
