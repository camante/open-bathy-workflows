from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from pipeline.river_workflow.river_workflow_context import CanonicalRiverSourceContext, RiverWorkflowContext
from pipeline.river_workflow.river_workflow_stage_grids import GridStageResult
from pipeline.river_workflow.river_workflow_stage_solve_domain import SolveDomainStageResult
from pipeline.river_workflow.river_workflow_validation import (
    validate_background_covers_measured_only,
    validate_raster_matches_template,
    validate_support_mask_matches_measured_only,
)
from linear_source_contracts import (
    load_linear_authoritative_source_contract,
    load_linear_baseline_source_contract,
)


@dataclass(frozen=True)
class AuthoritativeStageResult:
    solve_authoritative_base_measured_only_path: Path
    solve_authoritative_support_mask_path: Path
    solve_baseline_background_path: Path
    solve_authoritative_source_path: Path
    export_authoritative_source_path: Path
    export_baseline_source_path: Path
    solve_support_pixel_count: int


def _require_explicit_source(path_value: Path | None, *, field_name: str) -> Path:
    if path_value is None:
        raise RuntimeError(f"river_workflow_missing_explicit_source:{field_name}")
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError(f"river_workflow_missing_explicit_source_file:{field_name}:{path}")
    return path


def _resolve_authoritative_sources_from_bundle(ctx: RiverWorkflowContext) -> tuple[Path, Path, Path]:
    linear_inputs = ctx.linear_inputs
    if linear_inputs is None:
        raise RuntimeError("river_workflow_missing_source_bundle")

    authoritative_contract_path = Path(linear_inputs.authoritative_source_contract_path) if linear_inputs.authoritative_source_contract_path is not None else None
    baseline_contract_path = Path(linear_inputs.baseline_source_contract_path) if linear_inputs.baseline_source_contract_path is not None else None
    if authoritative_contract_path is not None and baseline_contract_path is not None:
        authoritative_contract = load_linear_authoritative_source_contract(authoritative_contract_path)
        baseline_contract = load_linear_baseline_source_contract(baseline_contract_path)
        if str(authoritative_contract.solve_source_role) == 'shared_pending_canonical_prepare':
            raise RuntimeError('river_workflow_authoritative_source_not_canonicalized')
        solve_src = _require_explicit_source(
            authoritative_contract.solve_authoritative_source_path,
            field_name="authoritative_source_contract.solve_authoritative_source_path",
        )
        export_src = _require_explicit_source(
            authoritative_contract.export_authoritative_source_path,
            field_name="authoritative_source_contract.export_authoritative_source_path",
        )
        export_baseline_src = _require_explicit_source(
            baseline_contract.export_baseline_source_path,
            field_name="baseline_source_contract.export_baseline_source_path",
        )
        return solve_src, export_src, export_baseline_src

    solve_src = _require_explicit_source(
        Path(linear_inputs.solve_authoritative_source_path) if linear_inputs.solve_authoritative_source_path is not None else None,
        field_name="solve_authoritative_source_path",
    )
    export_src = _require_explicit_source(
        Path(linear_inputs.export_authoritative_source_path) if linear_inputs.export_authoritative_source_path is not None else None,
        field_name="export_authoritative_source_path",
    )
    export_baseline_src = _require_explicit_source(
        Path(linear_inputs.export_baseline_source_path) if linear_inputs.export_baseline_source_path is not None else None,
        field_name="export_baseline_source_path",
    )
    return solve_src, export_src, export_baseline_src




def _load_canonical_source_context(ctx: RiverWorkflowContext) -> CanonicalRiverSourceContext:
    linear_inputs = ctx.linear_inputs
    if linear_inputs is None or linear_inputs.canonical_source_context_path is None:
        raise RuntimeError("river_workflow_missing_canonical_source_context")
    path = Path(linear_inputs.canonical_source_context_path)
    if not path.exists():
        raise RuntimeError(f"river_workflow_missing_canonical_source_context_file:{path}")
    return CanonicalRiverSourceContext.from_json(path)


def _require_canonical_materialized_authoritative_inputs(
    ctx: RiverWorkflowContext,
    grid_result: GridStageResult,
) -> tuple[CanonicalRiverSourceContext, Path, Path, Path]:
    linear_inputs = ctx.linear_inputs
    if linear_inputs is None:
        raise RuntimeError("river_workflow_missing_source_bundle")
    if str(linear_inputs.authoritative_routing_policy) != 'solve_stages_sample_only_from_canonical_authoritative':
        raise RuntimeError('river_workflow_authoritative_routing_policy_not_canonical_only')
    canonical_ctx = _load_canonical_source_context(ctx)
    canonical_grid = Path(canonical_ctx.canonical_solve_grid_template_path or '')
    measured = Path(canonical_ctx.canonical_solve_authoritative_measured_only_path or '')
    support = Path(canonical_ctx.canonical_solve_authoritative_support_mask_path or '')
    baseline = Path(canonical_ctx.canonical_solve_baseline_background_path or '')
    for path_value, label in ((canonical_grid, 'canonical_grid_template'), (measured, 'canonical_measured_only'), (support, 'canonical_support_mask'), (baseline, 'canonical_baseline_background')):
        if not path_value.exists():
            raise RuntimeError(f"river_workflow_missing_{label}:{path_value}")
    if Path(grid_result.solve_grid_template_path).resolve() != canonical_grid.resolve():
        raise RuntimeError(
            f"river_workflow_solve_grid_not_canonical:{grid_result.solve_grid_template_path}:{canonical_grid}"
        )
    return canonical_ctx, measured, support, baseline

def _warp_float_to_template(src_path: Path, template_path: Path, dst_path: Path) -> None:
    with rasterio.open(template_path) as template_ds, rasterio.open(src_path) as src_ds:
        template_profile = template_ds.profile.copy()
        src = src_ds.read(1).astype(np.float32)
        dst = np.full((template_ds.height, template_ds.width), np.float32(-9999.0), dtype=np.float32)
        src_nodata = src_ds.nodata
        if src_nodata is not None:
            src[src == src_nodata] = np.nan
        reproject(
            source=src,
            destination=dst,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=template_ds.transform,
            dst_crs=template_ds.crs,
            src_nodata=np.nan,
            dst_nodata=np.float32(-9999.0),
            resampling=Resampling.nearest,
        )
        dst[~np.isfinite(dst)] = np.float32(-9999.0)
        template_profile.pop("blockxsize", None)
        template_profile.pop("blockysize", None)
        template_profile.pop("BLOCKXSIZE", None)
        template_profile.pop("BLOCKYSIZE", None)
        template_profile.update(dtype="float32", count=1, nodata=np.float32(-9999.0), compress="deflate")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **template_profile) as out_ds:
            out_ds.write(dst.astype(np.float32), 1)


def _write_support_mask_from_measured(measured_path: Path, support_mask_path: Path) -> int:
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
    return int(np.count_nonzero(mask_u8))


def run_authoritative_stage(
    ctx: RiverWorkflowContext,
    solve_result: SolveDomainStageResult,
    grid_result: GridStageResult,
) -> AuthoritativeStageResult:
    del solve_result  # solve AOI selection is already reflected in the grid + bundle inputs.
    solve_src, export_src, export_baseline_src = _resolve_authoritative_sources_from_bundle(ctx)

    _, solve_measured_path, solve_support_path, solve_baseline_path = _require_canonical_materialized_authoritative_inputs(
        ctx,
        grid_result,
    )
    with rasterio.open(solve_support_path) as ds:
        solve_support_pixel_count = int(np.count_nonzero(ds.read(1)))

    validate_raster_matches_template(
        raster_path=solve_measured_path,
        template_path=grid_result.solve_grid_template_path,
    )
    validate_raster_matches_template(
        raster_path=solve_support_path,
        template_path=grid_result.solve_grid_template_path,
    )
    validate_raster_matches_template(
        raster_path=solve_baseline_path,
        template_path=grid_result.solve_grid_template_path,
    )
    validate_support_mask_matches_measured_only(
        measured_path=solve_measured_path,
        support_mask_path=solve_support_path,
    )
    validate_background_covers_measured_only(
        measured_path=solve_measured_path,
        background_path=solve_baseline_path,
    )

    return AuthoritativeStageResult(
        solve_authoritative_base_measured_only_path=solve_measured_path,
        solve_authoritative_support_mask_path=solve_support_path,
        solve_baseline_background_path=solve_baseline_path,
        solve_authoritative_source_path=solve_src,
        export_authoritative_source_path=export_src,
        export_baseline_source_path=export_baseline_src,
        solve_support_pixel_count=solve_support_pixel_count,
    )


__all__ = ["AuthoritativeStageResult", "run_authoritative_stage"]
