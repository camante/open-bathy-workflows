"""Single active river runner entrypoint.

The runner is the only bridge from ``bathy_main.py`` into the active
canonical-parent/AOI-export river pipeline.  It resolves the canonical source
bundle once, builds a ``RiverWorkflowContext``, runs the shared-solve pipeline,
and localizes report registration in one helper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from repo_runtime_modes import ACTIVE_RIVER_WORKFLOW
from river_runner_contract import RiverRunnerPrimaryArtifacts, RiverRunnerRegistrationPayload, RiverRunnerResult


def _as_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def _resolve_execution_role(cfg: Any, linear_inputs: Any) -> str:
    """Resolve the explicit execution role without inventing a second method path."""
    role = getattr(cfg, "river_execution_role", None)
    if role not in (None, ""):
        return str(role)
    # Default to canonical_build. The pipeline itself will switch to the
    # exact AOI-export route when a valid canonical manifest + parent already
    # exist. This keeps cache reuse an implementation detail rather than a
    # caller-side fallback branch.
    return "canonical_build"


def register_river_runner_result(report: dict[str, Any], result: Any) -> RiverRunnerRegistrationPayload:
    """Register the active runner result in the shared run report."""
    final_dem = getattr(result, "final_dem_result", None)
    solve = getattr(result, "solve_result", None)
    stage_receipts = getattr(result, "stage_receipts", {}) or {}

    outputs = report.setdefault("outputs", {})
    river_workflow = report.setdefault("river_workflow", {})
    active_river = report.setdefault("active_river", {})

    if final_dem is not None:
        canonical_parent = _as_path(getattr(final_dem, "canonical_parent_dem_path", None))
        aoi_export = _as_path(getattr(final_dem, "aoi_export_dem_path", None))
        final_user = _as_path(getattr(final_dem, "dem_enhanced_final_path", None))
        if canonical_parent is not None:
            outputs["canonical_parent_dem"] = str(canonical_parent)
            river_workflow["canonical_parent_dem"] = str(canonical_parent)
            active_river["canonical_parent_dem"] = str(canonical_parent)
        if aoi_export is not None:
            outputs["aoi_export_dem"] = str(aoi_export)
            river_workflow["aoi_export_dem"] = str(aoi_export)
            active_river["aoi_export_dem"] = str(aoi_export)
        if final_user is not None:
            outputs["river_workflow_final_dem"] = str(final_user)
            river_workflow["river_workflow_final_dem"] = str(final_user)
            active_river["river_workflow_final_dem"] = str(final_user)
        for attr, key in (
            ("canonical_parent_receipt_path", "canonical_parent_dem_receipt"),
            ("aoi_export_receipt_path", "aoi_export_dem_receipt"),
        ):
            value = _as_path(getattr(final_dem, attr, None))
            if value is not None:
                outputs[key] = str(value)
                river_workflow[key] = str(value)
                active_river[key] = str(value)
        writer_mode = getattr(final_dem, "final_writer_mode", None)
        if writer_mode not in (None, ""):
            river_workflow["final_writer_mode"] = str(writer_mode)
            active_river["final_writer_mode"] = str(writer_mode)
            if str(writer_mode) == "existing_canonical_parent_exact_aoi_export":
                river_workflow["aoi_execution_mode"] = "export_only"
                river_workflow["canonical_solution_source"] = "existing_parent"
            else:
                river_workflow.setdefault("aoi_execution_mode", "canonical_build_then_export")
                river_workflow.setdefault("canonical_solution_source", "built_this_run")

    canonical_system_id = getattr(solve, "canonical_system_id", None)
    if canonical_system_id not in (None, ""):
        river_workflow["canonical_system_id"] = str(canonical_system_id)
        active_river["canonical_system_id"] = str(canonical_system_id)

    if getattr(result, "run_contract_path", None) not in (None, ""):
        outputs["river_run_contract"] = str(result.run_contract_path)
        river_workflow["run_contract"] = str(result.run_contract_path)
        active_river["run_contract"] = str(result.run_contract_path)
    if getattr(result, "river_science_chain_summary_path", None) not in (None, ""):
        outputs["river_science_chain_summary"] = str(result.river_science_chain_summary_path)
        river_workflow["science_chain_summary"] = str(result.river_science_chain_summary_path)
        active_river["science_chain_summary"] = str(result.river_science_chain_summary_path)
    if getattr(result, "bundle_manifest_path", None) not in (None, ""):
        outputs["river_bundle_manifest"] = str(result.bundle_manifest_path)
    if stage_receipts:
        river_workflow["stage_receipts"] = {str(k): str(v) for k, v in stage_receipts.items()}
        active_river["stage_receipts"] = {str(k): str(v) for k, v in stage_receipts.items()}

    return RiverRunnerRegistrationPayload(
        outputs=dict(outputs),
        river_workflow=dict(river_workflow),
        active_river=dict(active_river),
    )



def _build_primary_artifacts(result: Any) -> RiverRunnerPrimaryArtifacts:
    final_dem = getattr(result, "final_dem_result", None)
    return RiverRunnerPrimaryArtifacts(
        canonical_parent_dem=_as_path(getattr(final_dem, "canonical_parent_dem_path", None)) if final_dem is not None else None,
        aoi_export_dem=_as_path(getattr(final_dem, "aoi_export_dem_path", None)) if final_dem is not None else None,
        final_user_dem=_as_path(getattr(final_dem, "dem_enhanced_final_path", None)) if final_dem is not None else None,
        run_contract=_as_path(getattr(result, "run_contract_path", None)),
        stage_chain_summary=_as_path(getattr(result, "river_science_chain_summary_path", None)),
    )


def run_river_workflow_direct(
    cfg: Any,
    report: dict[str, Any],
    *,
    logger: logging.Logger | None = None,
    script_dir: Path | None = None,
    ensure_dir_fn: Callable[[Any], Path] | None = None,
    detect_working_srs_fn: Callable[[Any], str] | None = None,
    estimate_raster_pixel_size_m_for_dst_crs_fn: Callable[..., float] | None = None,
    resolve_river_shared_source_artifacts_fn: Callable[..., Any] | None = None,
    prepare_river_canonical_source_bundle_fn: Callable[..., Any] | None = None,
    resolve_river_workflow_shared_source_artifacts_fn: Callable[..., Any] | None = None,
    prepare_river_workflow_canonical_source_bundle_fn: Callable[..., Any] | None = None,
    river_inputs_override: Any | None = None,
    return_river_workflow_details: bool = False,
) -> Any:
    """Run the built-in river workflow and return the underlying pipeline result."""
    log = logger or logging.getLogger("river_runner")
    if detect_working_srs_fn is None:
        raise RuntimeError("river_runner_missing_detect_working_srs_callback")
    resolve_source_bundle_fn = resolve_river_shared_source_artifacts_fn or resolve_river_workflow_shared_source_artifacts_fn
    prepare_source_bundle_fn = prepare_river_canonical_source_bundle_fn or prepare_river_workflow_canonical_source_bundle_fn
    if resolve_source_bundle_fn is None:
        raise RuntimeError("river_runner_missing_source_bundle_callback")
    if prepare_source_bundle_fn is None:
        raise RuntimeError("river_runner_missing_canonical_source_bundle_callback")

    from pipeline.river_workflow.river_workflow_context import RiverWorkflowContext
    from pipeline.river_shared_solve.river_pipeline import run_river_pipeline

    out_dir = Path(getattr(cfg, "out_dir")) / "river_workflow"
    if ensure_dir_fn is not None:
        ensure_dir_fn(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    projected_crs = str(detect_working_srs_fn(cfg))
    log.info("[RIVER][RUNNER][STEP 1/5] Resolve working CRS and active river source bundle: crs=%s", projected_crs)

    if river_inputs_override is not None:
        linear_inputs = river_inputs_override
        log.info("[RIVER][RUNNER][STEP 2/5] Using caller-provided canonical source bundle.")
    else:
        log.info("[RIVER][RUNNER][STEP 2/5] Resolve shared authoritative/network source artifacts for the canonical solve.")
        linear_inputs = resolve_source_bundle_fn(
            cfg,
            report,
            script_dir=Path(script_dir or Path(__file__).parent),
            projected_crs=projected_crs,
        )

    log.info("[RIVER][RUNNER][STEP 3/5] Prepare canonical source bundle and manifest handoff.")
    canonical_inputs = prepare_source_bundle_fn(
        cfg,
        linear_inputs=linear_inputs,
        logger=log,
    )

    target_resolution_m = float(getattr(canonical_inputs, "target_resolution_m", 0.0) or 0.0)
    if not (target_resolution_m > 0):
        target_resolution_m = float(getattr(cfg, "river_dem_res_m", 0.0) or 0.0)
    if not (target_resolution_m > 0):
        raise RuntimeError("river_runner_missing_positive_target_resolution_m")

    execution_role = _resolve_execution_role(cfg, canonical_inputs)
    ctx = RiverWorkflowContext(
        cfg=cfg,
        out_dir=out_dir,
        export_aoi=str(getattr(cfg, "aoi", "")),
        projected_crs=projected_crs,
        target_resolution_m=target_resolution_m,
        run_id=str(getattr(cfg, "run_id", "") or "river_run"),
        workflow_name=ACTIVE_RIVER_WORKFLOW,
        canonical_max_trace_km=float(getattr(cfg, "river_canonical_max_trace_km", 200.0) or 200.0),
        export_network_gpkg=Path(canonical_inputs.export_network_gpkg),
        network_gpkg=Path(canonical_inputs.solve_network_gpkg),
        linear_inputs=canonical_inputs,
        requested_solve_domain=(str(canonical_inputs.requested_solve_domain) if canonical_inputs.requested_solve_domain is not None else None),
        resolved_solve_domain=(str(canonical_inputs.resolved_solve_domain) if canonical_inputs.resolved_solve_domain is not None else None),
        solve_domain_source=str(getattr(canonical_inputs, "solve_domain_source", "derived_from_aoi") or "derived_from_aoi"),
        execution_role=execution_role,
        logger_name="river_workflow",
        adjacent_aoi_peer_run_dir=_as_path(getattr(cfg, "adjacent_aoi_peer_run_dir", None)),
        write_diagnostics=bool(getattr(cfg, "write_river_diagnostics", True)),
        write_core_receipts=bool(getattr(cfg, "write_core_receipts", True)),
        canonical_system_id=getattr(canonical_inputs, "canonical_system_id", None),
        canonical_identity_receipt_path=_as_path(getattr(canonical_inputs, "canonical_identity_receipt_path", None)),
        canonical_domain_bounds=getattr(canonical_inputs, "canonical_domain_bounds", None),
        canonical_trace_distance_km=getattr(canonical_inputs, "canonical_trace_distance_km", None),
    )

    log.info(
        "[RIVER][RUNNER][STEP 4/5] Run canonical-parent/AOI-export river pipeline: "
        "requested_role=%s (effective route is resolved by canonical parent handoff).",
        ctx.execution_role,
    )
    result = run_river_pipeline(ctx)
    final_mode = getattr(getattr(result, "final_dem_result", None), "final_writer_mode", None)
    effective_route = (
        "aoi_export_only_existing_parent"
        if str(final_mode) == "existing_canonical_parent_exact_aoi_export"
        else "canonical_build_then_export"
    )
    log.info("[RIVER][RUNNER][STEP 4/5] Effective river pipeline route: %s", effective_route)

    log.info("[RIVER][RUNNER][STEP 5/5] Register active river products and receipts in the run report.")
    payload = register_river_runner_result(report, result)
    report.setdefault("river_workflow", {})["runner_contract"] = "active_canonical_parent_aoi_export_v1"
    report.setdefault("active_river", {})["runner_contract"] = "active_canonical_parent_aoi_export_v1"

    # The active pipeline expects the underlying pipeline result directly.
    # Keep the wrapper construction local so older direct callers can still
    # inspect the public contract shape without changing the return type.
    _compat_wrapper = RiverRunnerResult(
        primary_artifacts=_build_primary_artifacts(result),
        registration_payload=payload,
        pipeline_result=result,
    )
    report.setdefault("river_workflow", {})["primary_artifacts"] = {
        k: (str(v) if v is not None else None)
        for k, v in _compat_wrapper.primary_artifacts.__dict__.items()
    }
    return result



__all__ = [
    "register_river_runner_result",
    "run_river_workflow_direct",
]
