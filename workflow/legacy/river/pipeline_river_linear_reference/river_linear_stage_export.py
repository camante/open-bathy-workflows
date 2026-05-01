from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline.river_linear.river_linear_context import RiverLinearContext
from pipeline.river_linear.river_linear_export_logic import build_take_mask, stage_mask_handoff, subset_binary_mask_to_template_exact, subset_float_raster_to_template_exact
from pipeline.river_linear.river_linear_stage_authoritative import AuthoritativeStageResult
from pipeline.river_linear.river_linear_stage_corridor import CorridorStageResult
from pipeline.river_linear.river_linear_stage_grids import GridStageResult
from pipeline.river_linear.river_linear_stage_lock import LockStageResult
from pipeline.river_linear.river_linear_validation import validate_background_covers_measured_only, validate_mask_raster, validate_support_mask_matches_measured_only, validate_surface_raster, validate_take_mask_raster


def _write_support_mask_from_measured(measured_path: Path, support_mask_path: Path) -> int:
    import numpy as np
    import rasterio

    with rasterio.open(measured_path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        mask = np.isfinite(arr)
        if nodata is not None:
            mask &= arr != nodata
        mask_u8 = np.asarray(mask, dtype=np.uint8)
        profile = ds.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("BLOCKXSIZE", None)
        profile.pop("BLOCKYSIZE", None)
        profile.update(dtype="uint8", nodata=0, count=1, compress="deflate")
        support_mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(support_mask_path, "w", **profile) as out_ds:
            out_ds.write(mask_u8, 1)
    return int(mask_u8.sum())




def _exact_or_warp_float(*, src_path: Path, template_path: Path, dst_path: Path) -> int:
    try:
        return subset_float_raster_to_template_exact(src_path=src_path, template_path=template_path, dst_path=dst_path)
    except ValueError as exc:
        raise ValueError(
            f'linear_export_template_not_exact_parent_subset_float:src={src_path};template={template_path};dst={dst_path}'
        ) from exc


def _exact_or_warp_binary(*, src_path: Path, template_path: Path, dst_path: Path) -> int:
    try:
        return subset_binary_mask_to_template_exact(src_path=src_path, template_path=template_path, dst_path=dst_path)
    except ValueError as exc:
        raise ValueError(
            f'linear_export_template_not_exact_parent_subset_binary:src={src_path};template={template_path};dst={dst_path}'
        ) from exc


@dataclass(frozen=True)
class ExportStageResult:
    export_grid_template_path: Path
    export_authoritative_base_measured_only_path: Path
    export_authoritative_support_mask_path: Path
    export_baseline_background_path: Path
    river_guidance_export_path: Path
    river_corridor_mask_export_path: Path
    river_support_mask_export_path: Path
    river_take_mask_export_path: Path
    finite_guidance_export_count: int
    corridor_export_pixel_count: int
    support_export_pixel_count: int
    take_export_pixel_count: int


def _canonical_export_sources(ctx: RiverLinearContext) -> tuple[Path | None, Path | None, Path | None]:
    bundle = ctx.linear_inputs
    if bundle is None:
        return None, None, None
    measured = Path(bundle.canonical_solve_authoritative_measured_only_path) if bundle.canonical_solve_authoritative_measured_only_path is not None else None
    support = Path(bundle.canonical_solve_authoritative_support_mask_path) if bundle.canonical_solve_authoritative_support_mask_path is not None else None
    baseline = Path(bundle.canonical_solve_baseline_background_path) if bundle.canonical_solve_baseline_background_path is not None else None
    if measured is not None and not measured.exists():
        measured = None
    if support is not None and not support.exists():
        support = None
    if baseline is not None and not baseline.exists():
        baseline = None
    return measured, support, baseline


def run_export_stage(
    ctx: RiverLinearContext,
    grid_result: GridStageResult,
    authoritative_result: AuthoritativeStageResult,
    corridor_result: CorridorStageResult,
    lock_result: LockStageResult,
) -> ExportStageResult:
    canonical_export_measured, canonical_export_support, canonical_export_baseline = _canonical_export_sources(ctx)
    measured_export_source = canonical_export_measured or authoritative_result.export_authoritative_source_path
    baseline_export_source = canonical_export_baseline or authoritative_result.export_baseline_source_path
    _exact_or_warp_float(
        src_path=measured_export_source,
        template_path=grid_result.export_grid_template_path,
        dst_path=ctx.paths.export_authoritative_base_measured_only,
    )
    _exact_or_warp_float(
        src_path=baseline_export_source,
        template_path=grid_result.export_grid_template_path,
        dst_path=ctx.paths.export_baseline_background,
    )
    if canonical_export_support is not None:
        support_export_pixel_count = _exact_or_warp_binary(
            src_path=canonical_export_support,
            template_path=grid_result.export_grid_template_path,
            dst_path=ctx.paths.export_authoritative_support_mask,
        )
    else:
        support_export_pixel_count = _write_support_mask_from_measured(
            ctx.paths.export_authoritative_base_measured_only,
            ctx.paths.export_authoritative_support_mask,
        )
    finite_guidance_export_count = _exact_or_warp_float(
        src_path=lock_result.river_primary_surface_solve_locked_path,
        template_path=grid_result.export_grid_template_path,
        dst_path=ctx.paths.river_guidance_export,
    )
    corridor_export_pixel_count = _exact_or_warp_binary(
        src_path=corridor_result.river_corridor_solve_path,
        template_path=grid_result.export_grid_template_path,
        dst_path=ctx.paths.river_corridor_mask_export,
    )
    stage_mask_handoff(
        src_path=ctx.paths.export_authoritative_support_mask,
        dst_path=ctx.paths.river_support_mask_export,
    )
    take_export_pixel_count = build_take_mask(
        corridor_mask_path=ctx.paths.river_corridor_mask_export,
        support_mask_path=ctx.paths.river_support_mask_export,
        dst_path=ctx.paths.river_take_mask_export,
    )
    validate_surface_raster(raster_path=ctx.paths.river_guidance_export, template_path=grid_result.export_grid_template_path)
    validate_mask_raster(raster_path=ctx.paths.river_corridor_mask_export, template_path=grid_result.export_grid_template_path)
    validate_mask_raster(raster_path=ctx.paths.export_authoritative_support_mask, template_path=grid_result.export_grid_template_path)
    validate_mask_raster(raster_path=ctx.paths.river_support_mask_export, template_path=grid_result.export_grid_template_path)
    validate_support_mask_matches_measured_only(
        measured_path=ctx.paths.export_authoritative_base_measured_only,
        support_mask_path=ctx.paths.export_authoritative_support_mask,
    )
    validate_background_covers_measured_only(
        measured_path=ctx.paths.export_authoritative_base_measured_only,
        background_path=ctx.paths.export_baseline_background,
    )
    validate_take_mask_raster(
        take_mask_path=ctx.paths.river_take_mask_export,
        corridor_mask_path=ctx.paths.river_corridor_mask_export,
        support_mask_path=ctx.paths.river_support_mask_export,
        template_path=grid_result.export_grid_template_path,
    )
    return ExportStageResult(
        export_grid_template_path=grid_result.export_grid_template_path,
        export_authoritative_base_measured_only_path=ctx.paths.export_authoritative_base_measured_only,
        export_authoritative_support_mask_path=ctx.paths.export_authoritative_support_mask,
        export_baseline_background_path=ctx.paths.export_baseline_background,
        river_guidance_export_path=ctx.paths.river_guidance_export,
        river_corridor_mask_export_path=ctx.paths.river_corridor_mask_export,
        river_support_mask_export_path=ctx.paths.river_support_mask_export,
        river_take_mask_export_path=ctx.paths.river_take_mask_export,
        finite_guidance_export_count=finite_guidance_export_count,
        corridor_export_pixel_count=corridor_export_pixel_count,
        support_export_pixel_count=support_export_pixel_count,
        take_export_pixel_count=take_export_pixel_count,
    )


__all__ = ["ExportStageResult", "run_export_stage"]
