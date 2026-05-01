"""Bathy-main entrypoint for the single active river workflow."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable



@dataclass(frozen=True)
class BathyMainRiverWorkflowFactoryInputs:
    run_river_fn: Callable[..., Any]
    run_river_workflow_direct_fn: Callable[..., Any]
    ensure_dir_fn: Callable[..., Any]
    detect_working_srs_fn: Callable[..., Any]
    estimate_raster_pixel_size_m_for_dst_crs_fn: Callable[..., Any]
    resolve_river_shared_source_artifacts_fn: Callable[..., Any]
    prepare_river_canonical_source_bundle_fn: Callable[..., Any]
    raster_crs_matches_fn: Callable[..., Any]
    warp_raster_to_srs_fn: Callable[..., Any]
    write_final_output_receipt_fn: Callable[..., Any]
    finalize_existing_output_run_stage_fn: Callable[..., Any]
    write_bundle_fn: Callable[..., Any]
    finalize_run_fn: Callable[..., Any]
    write_io_manifest_fn: Callable[..., Any]
    emit_artifacts_fn: Callable[..., Any]
    run_seam_comparisons_fn: Callable[..., Any]
    log: Any
    script_dir: Any
    run_id: str


def build_bathy_main_river_workflow_dependencies(inputs: BathyMainRiverWorkflowFactoryInputs) -> dict[str, Any]:
    """Return active-workflow callbacks using explicit, stable names."""
    callbacks = {field.name: getattr(inputs, field.name) for field in fields(inputs)}
    callbacks.update(
        {
            "run_river_fn": inputs.run_river_fn,
            "run_river_workflow_direct_fn": inputs.run_river_workflow_direct_fn,
            "write_final_output_receipt_fn": inputs.write_final_output_receipt_fn,
            "finalize_existing_output_run_stage_fn": inputs.finalize_existing_output_run_stage_fn,
            "write_bundle_fn": inputs.write_bundle_fn,
            "finalize_run_fn": inputs.finalize_run_fn,
            "write_io_manifest_fn": inputs.write_io_manifest_fn,
            "emit_artifacts_fn": inputs.emit_artifacts_fn,
            "run_seam_comparisons_fn": inputs.run_seam_comparisons_fn,
        }
    )
    return callbacks


def execute_river_workflow_entry(
    *,
    cfg: Any,
    args: Any,
    report: dict[str, Any],
    log: Any,
    fatal_errors: list[str],
    run_id: str,
    dependencies: dict[str, Any],
) -> int:
    """Execute the active river workflow and the shared finalization stage."""
    from active_context import ActiveWorkflowContext
    from active_pipeline import finalize_active_river_workflow, run_active_river_workflow

    ctx = ActiveWorkflowContext(
        cfg=cfg,
        args=args,
        report=report,
        log=log,
        fatal_errors=fatal_errors,
        run_id=str(run_id),
        callbacks=dependencies,
    )
    workflow_result = run_active_river_workflow(ctx)
    finalize_result = finalize_active_river_workflow(ctx, workflow_result)
    return int(finalize_result.exit_code)


__all__ = [
    "BathyMainRiverWorkflowFactoryInputs",
    "build_bathy_main_river_workflow_dependencies",
    "execute_river_workflow_entry",
]
