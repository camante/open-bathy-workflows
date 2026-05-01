from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable
import logging

import numpy as np
import geopandas as gpd
import rasterio
from trusted_interior import build_river_trusted_interior, build_soft_guidance_domain, build_river_admissibility, summarize_trusted_export_region

from core.json_io import write_json

from core.constants import PIPELINE_VERSION

from legacy.river.river_v2_context import RiverV2Context
from legacy.river.river_v2_execution_contract import RiverV2ExecutionContract
from legacy.river.river_v2_preflight import (
    ensure_wse_support_source_raster as _ensure_wse_support_source_raster,
    run_river_v2_execution_contract_preflight,
)
from legacy.river.river_v2_contract import (
    PASS1_STAGE_IDS,
    PASS2_STAGE_IDS,
    PASS3_STAGE_IDS,
    PASS4_STAGE_IDS,
    RiverV2Pass1Result,
    RiverV2Pass2Result,
    RiverV2Pass3Result,
    RiverV2Pass4Result,
    RiverV2RunResult,
    RIVER_V2_ROUTE_MODE_PASS1,
    RIVER_V2_ROUTE_MODE_PASS2,
    RIVER_V2_ROUTE_MODE_PASS3,
    RIVER_V2_ROUTE_MODE_PASS4,
    RIVER_V2_TARGET_MODE,
    STAGE_CENTERLINE_AUTHORITATIVE_BED,
    STAGE_CENTERLINE_BED_BACKBONE,
    STAGE_CENTERLINE_BED_BACKBONE_DENSE,
    STAGE_CENTERLINE_OBSERVED_OFFSET,
    STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS,
    STAGE_CENTERLINE_OFFSET_MODELED,
    STAGE_CENTERLINE_WSE_SUPPORT,
    STAGE_CENTERLINE_WSE_PROXY,
    STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT,
    STAGE_RIVER_CENTERLINE,
    STAGE_RIVER_PRIMARY_SURFACE,
    STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
    STAGE_RIVER_EXPORT_SUBSET,
    mark_v2_stage_failed,
    mark_v2_stage_implemented,
    river_v2_stage_status_placeholder,
    subset_stage_results,
    subset_stage_status,
)
from legacy.river.river_v2_stage_authoritative_bed import run_authoritative_bed_stage
from legacy.river.river_v2_stage_authoritative_support import run_authoritative_support_stage
from legacy.river.river_v2_stage_authoritative_lock import build_authoritative_locked_primary_surface
from legacy.river.river_v2_stage_backbone import run_backbone_stage
from legacy.river.river_v2_stage_backbone_dense import run_backbone_dense_stage
from legacy.river.river_v2_stage_centerline import run_centerline_stage
from legacy.river.river_v2_stage_modeled_offset import run_modeled_offset_stage
from legacy.river.river_v2_stage_offset_transfer_prior import run_offset_transfer_prior_stage
from legacy.river.river_v2_stage_observed_offset import run_observed_offset_stage
from legacy.river.river_v2_stage_primary_surface import build_river_primary_surface
from legacy.river.river_v2_stage_wse_proxy import run_wse_proxy_stage
from legacy.river.river_v2_stage_wse_support import run_wse_support_stage
from legacy.river.river_v2_stage_export_subset import run_export_subset_stage
from legacy.river.river_v2_wse_support import build_river_v2_wse_support_products
from legacy.river.river_v2_reporting import (
    DEFAULT_STAGE_NUMBERS,
    write_compatibility_pass_summary,
    write_pipeline_summary,
    write_stage_products_overview,
    write_stage_trace,
)


_STAGE_NUMBERS = DEFAULT_STAGE_NUMBERS


def _write_stage_trace_snapshot(ctx: RiverV2Context, *, plan, stage_status, stage_results, failed_stage=None, error=None):
    write_stage_trace(
        ctx.paths.stage_trace,
        root=ctx.paths.root,
        plan=plan,
        stage_status=stage_status,
        stage_results=stage_results,
        stage_numbers=_STAGE_NUMBERS,
        failed_stage=failed_stage,
        error=error,
        support_status=str(getattr(ctx, "system_support_status", "unknown") or "unknown"),
        canonical_system_id=str(getattr(ctx, "canonical_system_id", None)) if getattr(ctx, "canonical_system_id", None) is not None else None,
        canonical_solve_aoi=str(getattr(ctx, "canonical_solve_aoi", None)) if getattr(ctx, "canonical_solve_aoi", None) is not None else None,
        export_aoi=str(getattr(ctx, "export_aoi", None)) if getattr(ctx, "export_aoi", None) is not None else None,
    )


@dataclass(frozen=True)
class StageStep:
    stage_id: str
    run: Callable[[Dict[str, Any]], Any]



def _infer_failed_stage(stage_results: Dict[str, Any], *, target_stage_ids: Iterable[str]) -> str:
    target_list = list(target_stage_ids)
    for stage_id in target_list:
        if stage_id not in stage_results:
            return stage_id
    return target_list[-1] if target_list else STAGE_RIVER_CENTERLINE


def _build_run_result(*, success: bool, execution_mode: str, result_cls, stage_status, stage_results, aux_outputs, stage_ids, failed_stage=None, error=None):
    return result_cls(
        success=bool(success),
        execution_mode=str(execution_mode),
        stage_status=subset_stage_status(stage_status, stage_ids),
        stage_results=subset_stage_results(stage_results, stage_ids),
        failed_stage=failed_stage,
        error=error,
        aux_outputs=aux_outputs,
    )




def _success_result(*, stage_status, stage_results, aux_outputs):
    return RiverV2RunResult(
        success=True,
        execution_mode=RIVER_V2_TARGET_MODE,
        stage_status=stage_status,
        stage_results=stage_results,
        aux_outputs=aux_outputs,
    )


def _failure_result(*, stage_status, stage_results, aux_outputs, failed_stage, error):
    return RiverV2RunResult(
        success=False,
        execution_mode=RIVER_V2_TARGET_MODE,
        stage_status=stage_status,
        stage_results=stage_results,
        failed_stage=failed_stage,
        error=error,
        aux_outputs=aux_outputs,
    )


PRODUCTION_STAGE_ORDER = (
    STAGE_RIVER_CENTERLINE,
    STAGE_CENTERLINE_WSE_SUPPORT,
    STAGE_CENTERLINE_WSE_PROXY,
    STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT,
    STAGE_CENTERLINE_AUTHORITATIVE_BED,
    STAGE_CENTERLINE_OBSERVED_OFFSET,
    STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS,
    STAGE_CENTERLINE_OFFSET_MODELED,
    STAGE_CENTERLINE_BED_BACKBONE,
    STAGE_CENTERLINE_BED_BACKBONE_DENSE,
    STAGE_RIVER_PRIMARY_SURFACE,
    STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
    STAGE_RIVER_EXPORT_SUBSET,
)


def _write_trusted_export_artifacts(ctx: RiverV2Context, *, locked_surface_path: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    export_mask_path = ctx.export_channel_mask_path
    if export_mask_path is None or not Path(export_mask_path).exists():
        return outputs
    locked_surface_path = Path(locked_surface_path)
    if not locked_surface_path.exists():
        return outputs
    with rasterio.open(export_mask_path) as mask_ds, rasterio.open(locked_surface_path) as locked_ds:
        if locked_ds.shape != mask_ds.shape or locked_ds.transform != mask_ds.transform or locked_ds.crs != mask_ds.crs:
            raise RuntimeError('river_v2_trusted_export_grid_mismatch')
        channel = (mask_ds.read(1) > 0)
        locked = locked_ds.read(1).astype('float32')
        nodata = locked_ds.nodata
        if nodata is not None:
            locked[np.isclose(locked, np.float32(nodata))] = np.nan
        locked[~np.isfinite(locked)] = np.nan
        if mask_ds.crs is not None and getattr(mask_ds.crs, 'is_geographic', False):
            edge_buffer_px = 0
        else:
            px_x = abs(float(mask_ds.transform.a))
            px_y = abs(float(mask_ds.transform.e))
            px_m = max(min(px_x, px_y), 1.0e-6)
            edge_buffer_px = max(int(round(float(getattr(ctx.cfg, 'river_trusted_halo_m', 60.0) or 0.0) / px_m)), 0)
        trusted_interior = build_river_trusted_interior(channel=channel.astype('uint8'), edge_buffer_px=int(edge_buffer_px))
        soft_guidance = build_soft_guidance_domain(trusted_export_region=trusted_interior, valid_depth=np.isfinite(locked))
        admissibility = build_river_admissibility(soft_guidance_domain=soft_guidance, authoritative_anchor_support=None)
        guidance_weight = admissibility.astype('float32')
        trusted_surface = np.full(locked.shape, np.nan, dtype='float32')
        trusted_mask = admissibility > 0
        trusted_surface[trusted_mask] = locked[trusted_mask]
        profile_u8 = mask_ds.profile.copy(); profile_u8.update(dtype='uint8', nodata=0, compress='deflate', count=1, tiled=False)
        profile_f32 = locked_ds.profile.copy(); profile_f32.update(dtype='float32', nodata=nodata if nodata is not None else np.nan, compress='deflate', count=1, tiled=False)
        for profile in (profile_u8, profile_f32):
            profile.pop('blockxsize', None)
            profile.pop('blockysize', None)
        for path, arr, profile in (
            (ctx.paths.river_trusted_interior, trusted_interior.astype('uint8'), profile_u8),
            (ctx.paths.river_admissibility, admissibility.astype('uint8'), profile_u8),
            (ctx.paths.river_guidance_weight, guidance_weight.astype('float32'), {**profile_f32, 'nodata': 0.0}),
            (ctx.paths.river_primary_surface_trusted_export, trusted_surface.astype('float32'), profile_f32),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, 'w', **profile) as ds:
                ds.write(arr, 1)
        summary = summarize_trusted_export_region(
            channel=channel.astype('uint8'),
            trusted_export_region=trusted_interior.astype('uint8'),
            estuary_transition=None,
            edge_buffer_px=int(edge_buffer_px),
        )
        summary['valid_trusted_export_surface_pixels'] = int(np.count_nonzero(np.isfinite(trusted_surface)))
        write_json(ctx.paths.river_trusted_interior_summary, summary)
    outputs.update({
        'river_v2_primary_surface_trusted_export': str(ctx.paths.river_primary_surface_trusted_export),
        'river_v2_trusted_interior': str(ctx.paths.river_trusted_interior),
        'river_v2_admissibility': str(ctx.paths.river_admissibility),
        'river_v2_guidance_weight': str(ctx.paths.river_guidance_weight),
        'river_v2_trusted_interior_summary': str(ctx.paths.river_trusted_interior_summary),
    })
    return outputs


def _register_pipeline_outputs(ctx: RiverV2Context, stage_results: Dict[str, Any], aux_outputs: Dict[str, str]) -> Dict[str, str]:
    outputs = dict(aux_outputs or {})
    outputs.update({
        "pipeline_version": str(PIPELINE_VERSION),
        "river_method_selected": "v2",
        "river_path_used": "river_v2_only",
        "legacy_river_path_participated": "false",
        "river_v2_summary": str(ctx.paths.pipeline_summary),
        "river_v2_stage_products_overview": str(ctx.paths.stage_products_overview),
        "river_v2_stage_trace": str(ctx.paths.stage_trace),
        "river_v2_system_support_decision": str(ctx.support_decision_json_path or ctx.paths.system_support_decision),
        "river_v2_system_support_status": str(ctx.system_support_status or "unknown"),
        "river_v2_resolved_inputs_manifest": str(ctx.paths.resolved_inputs_manifest),
        "river_v2_preflight_receipt": str(ctx.paths.preflight_receipt),
        "river_v2_corridor_mask": str(ctx.export_channel_mask_path or ctx.channel_mask_path) if (ctx.export_channel_mask_path or ctx.channel_mask_path) is not None else "",
    })
    canonical_keys = {
        STAGE_RIVER_CENTERLINE: "river_v2_centerline_points",
        STAGE_CENTERLINE_WSE_SUPPORT: "river_v2_wse_support_points",
        STAGE_CENTERLINE_WSE_PROXY: "river_v2_wse_points",
        STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT: "river_v2_authoritative_support_raster",
        STAGE_CENTERLINE_AUTHORITATIVE_BED: "river_v2_authoritative_bed_points",
        STAGE_CENTERLINE_OBSERVED_OFFSET: "river_v2_observed_offset_points",
        STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS: "river_v2_component_offset_transfer_priors",
        STAGE_CENTERLINE_OFFSET_MODELED: "river_v2_modeled_offset_points",
        STAGE_CENTERLINE_BED_BACKBONE: "river_v2_backbone_points",
        STAGE_CENTERLINE_BED_BACKBONE_DENSE: "river_v2_backbone_dense_points",
        STAGE_RIVER_PRIMARY_SURFACE: "river_v2_primary_surface_solve_domain",
        STAGE_RIVER_PRIMARY_SURFACE_LOCKED: "river_v2_primary_surface_authoritative_applied_solve_domain",
        STAGE_RIVER_EXPORT_SUBSET: "river_v2_primary_surface_authoritative_applied",
    }
    for stage_id, output_key in canonical_keys.items():
        result = stage_results.get(stage_id)
        if result is not None:
            outputs[output_key] = str(result.output_artifact)
    if "river_v2_component_stream_summary" in outputs:
        outputs["river_v2_component_stream_summary"] = str(outputs["river_v2_component_stream_summary"])
    if "river_v2_component_stream_summary_receipt" in outputs:
        outputs["river_v2_component_stream_summary_receipt"] = str(outputs["river_v2_component_stream_summary_receipt"])
    return outputs


def _remember_stage_outputs(ctx: RiverV2Context, stage_id: str, result: Any, aux_outputs: Dict[str, str]) -> None:
    if result.aux_outputs:
        aux_outputs.update({str(k): str(v) for k, v in result.aux_outputs.items()})
    if stage_id == STAGE_RIVER_PRIMARY_SURFACE:
        aux_outputs["river_v2_primary_surface_solve_domain"] = str(result.output_artifact)
    elif stage_id == STAGE_RIVER_PRIMARY_SURFACE_LOCKED:
        aux_outputs["river_v2_primary_surface_authoritative_applied_solve_domain"] = str(result.output_artifact)
    elif stage_id == STAGE_RIVER_EXPORT_SUBSET:
        aux_outputs.update({str(k): str(v) for k, v in (result.aux_outputs or {}).items()})
        aux_outputs.update(_write_trusted_export_artifacts(ctx, locked_surface_path=Path(result.output_artifact)))



def _ensure_explicit_wse_support(
    ctx: RiverV2Context,
    *,
    centerline_points_path: Path,
    channel_mask_path: Path,
    support_source_raster_path: Path | None,
    support_source_contract: str | None,
) -> dict[str, str]:
    existing: dict[str, str] = {}
    if ctx.bank_wse_edge_guidance_path is not None and Path(ctx.bank_wse_edge_guidance_path).exists():
        existing["bank_wse_edge_guidance_path"] = str(Path(ctx.bank_wse_edge_guidance_path))
    if existing.get("bank_wse_edge_guidance_path"):
        return existing

    support_dir = ctx.paths.support_dir
    support_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = support_dir / "river_bank_wse_materialization_diagnostics.json"

    if channel_mask_path is None or not Path(channel_mask_path).exists():
        write_json(diagnostics_path, {
            "status": "failed",
            "source_contract": None,
            "error": "missing_channel_mask",
            "selected_source_raster": None,
        })
        raise RuntimeError("river_v2_wse_missing_channel_mask")
    if not Path(centerline_points_path).exists():
        write_json(diagnostics_path, {
            "status": "failed",
            "source_contract": None,
            "error": "missing_centerline_points",
            "selected_source_raster": None,
        })
        raise RuntimeError("river_v2_wse_missing_centerline_points")

    support_source = Path(support_source_raster_path) if support_source_raster_path is not None else None
    source_contract = str(support_source_contract) if support_source_contract is not None else None
    if support_source is None or source_contract is None or (not support_source.exists()):
        write_json(diagnostics_path, {
            "status": "failed",
            "source_contract": None,
            "error": "missing_wse_support_source_raster",
            "selected_source_raster": None,
        })
        raise RuntimeError("river_v2_wse_missing_support_source_raster")

    logger = logging.getLogger(__name__)
    logger.info(
        "[RIVER][V2][WSE] Materializing explicit bank guidance from explicit source raster (%s): %s",
        source_contract,
        support_source,
    )
    try:
        summary = build_river_v2_wse_support_products(
            support_raster=support_source,
            cfg=ctx.cfg,
            channel_mask_tif=Path(channel_mask_path),
            centerline_points_path=Path(centerline_points_path),
            support_dir=support_dir,
            logger=logger,
        )
    except Exception as exc:
        write_json(diagnostics_path, {
            "status": "failed",
            "source_contract": source_contract,
            "selected_source_raster": str(support_source),
            "error": str(exc),
            "error_type": type(exc).__name__,
        })
        raise

    edge = summary.get("bank_wse_edge_guidance_path") or summary.get("bank_elevation_path")
    if edge:
        ctx.bank_wse_edge_guidance_path = Path(edge)
        existing["bank_wse_edge_guidance_path"] = str(edge)

    write_json(diagnostics_path, {
        "status": "success",
        "source_contract": source_contract,
        "selected_source_raster": str(support_source),
        "artifacts": existing,
    })
    return existing


def _run_wse_support_stage(ctx: RiverV2Context, contract: RiverV2ExecutionContract, results: Dict[str, Any]):
    centerline_path = results[STAGE_RIVER_CENTERLINE].output_artifact
    support_paths = _ensure_explicit_wse_support(
        ctx,
        centerline_points_path=centerline_path,
        channel_mask_path=contract.solve_channel_mask_path,
        support_source_raster_path=contract.wse_support_source_raster_path,
        support_source_contract=contract.wse_support_source_contract,
    )
    edge_path = support_paths.get("bank_wse_edge_guidance_path")
    if not edge_path:
        raise RuntimeError("river_v2_wse_missing_explicit_bank_wse_support_raster")
    return run_wse_support_stage(
        ctx,
        centerline_points_path=centerline_path,
        bank_wse_edge_guidance_path=Path(edge_path),
    )



def _stage_plan(ctx: RiverV2Context, contract: RiverV2ExecutionContract) -> list[StageStep]:
    return [
        StageStep(
            stage_id=STAGE_RIVER_CENTERLINE,
            run=lambda _results: run_centerline_stage(
                ctx,
                network_gpkg=contract.solve_network_gpkg,
                river_dem_path=contract.river_dem_path,
                sampling_raster_path=contract.authoritative_sampling_raster_path,
                channel_mask_path=contract.solve_channel_mask_path,
                spacing_m=contract.centerline_spacing_m,
                min_stream_order=contract.min_stream_order,
            ),
        ),
        StageStep(
            stage_id=STAGE_CENTERLINE_WSE_SUPPORT,
            run=lambda results: _run_wse_support_stage(ctx, contract, results),
        ),
        StageStep(
            stage_id=STAGE_CENTERLINE_WSE_PROXY,
            run=lambda results: run_wse_proxy_stage(
                ctx,
                support_points_path=results[STAGE_CENTERLINE_WSE_SUPPORT].output_artifact,
            ),
        ),
        StageStep(
            stage_id=STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT,
            run=lambda results: run_authoritative_support_stage(
                ctx,
                authoritative_sampling_raster_path=contract.authoritative_sampling_raster_path,
                authoritative_bed_support_points_path=contract.authoritative_bed_support_points_path,
            ),
        ),
        StageStep(
            stage_id=STAGE_CENTERLINE_AUTHORITATIVE_BED,
            run=lambda results: run_authoritative_bed_stage(
                ctx,
                centerline_points_path=results[STAGE_RIVER_CENTERLINE].output_artifact,
                authoritative_support_raster_path=results[STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT].output_artifact,
            ),
        ),
        StageStep(
            stage_id=STAGE_CENTERLINE_OBSERVED_OFFSET,
            run=lambda results: run_observed_offset_stage(
                ctx,
                wse_points_path=results[STAGE_CENTERLINE_WSE_PROXY].output_artifact,
                authoritative_bed_points_path=results[STAGE_CENTERLINE_AUTHORITATIVE_BED].output_artifact,
            ),
        ),
        StageStep(
            stage_id=STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS,
            run=lambda results: run_offset_transfer_prior_stage(
                ctx,
                wse_points_path=results[STAGE_CENTERLINE_WSE_PROXY].output_artifact,
                observed_offset_points_path=results[STAGE_CENTERLINE_OBSERVED_OFFSET].output_artifact,
            ),
        ),
        StageStep(
            stage_id=STAGE_CENTERLINE_OFFSET_MODELED,
            run=lambda results: run_modeled_offset_stage(
                ctx,
                wse_points_path=results[STAGE_CENTERLINE_WSE_PROXY].output_artifact,
                observed_offset_points_path=results[STAGE_CENTERLINE_OBSERVED_OFFSET].output_artifact,
                transfer_priors_path=results[STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS].output_artifact,
            ),
        ),
        StageStep(
            stage_id=STAGE_CENTERLINE_BED_BACKBONE,
            run=lambda results: run_backbone_stage(
                ctx,
                wse_points_path=results[STAGE_CENTERLINE_WSE_PROXY].output_artifact,
                modeled_offset_points_path=results[STAGE_CENTERLINE_OFFSET_MODELED].output_artifact,
            ),
        ),
        StageStep(
            stage_id=STAGE_CENTERLINE_BED_BACKBONE_DENSE,
            run=lambda results: run_backbone_dense_stage(
                ctx,
                backbone_points_path=results[STAGE_CENTERLINE_BED_BACKBONE].output_artifact,
                centerline_points_path=results[STAGE_RIVER_CENTERLINE].output_artifact,
            ),
        ),
        StageStep(
            stage_id=STAGE_RIVER_PRIMARY_SURFACE,
            run=lambda results: build_river_primary_surface(
                ctx,
                backbone_points_path=results[STAGE_CENTERLINE_BED_BACKBONE_DENSE].output_artifact,
                channel_mask_path=contract.solve_channel_mask_path,
                centerline_points_path=results[STAGE_RIVER_CENTERLINE].output_artifact,
                output_path=ctx.paths.river_primary_surface_solve_domain,
                receipt_path=ctx.paths.river_primary_surface_solve_domain_receipt,
                surface_domain_role="solve_domain",
            ),
        ),
        StageStep(
            stage_id=STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
            run=lambda results: build_authoritative_locked_primary_surface(
                ctx,
                primary_surface_path=results[STAGE_RIVER_PRIMARY_SURFACE].output_artifact,
                authoritative_base_path=contract.solve_authoritative_lock_base_path,
                authoritative_support_raster_path=results[STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT].output_artifact,
                output_path=ctx.paths.river_primary_surface_authoritative_applied_solve_domain,
                receipt_path=ctx.paths.river_primary_surface_authoritative_applied_solve_domain_receipt,
                lock_domain_role="solve_domain",
            ),
        ),
        StageStep(
            stage_id=STAGE_RIVER_EXPORT_SUBSET,
            run=lambda results: run_export_subset_stage(
                ctx,
                solve_primary_surface_path=results[STAGE_RIVER_PRIMARY_SURFACE].output_artifact,
                solve_locked_surface_path=results[STAGE_RIVER_PRIMARY_SURFACE_LOCKED].output_artifact,
                export_channel_mask_path=contract.export_channel_mask_path,
            ),
        ),
    ]


def _build_production_plan(ctx: RiverV2Context, contract: RiverV2ExecutionContract) -> list[StageStep]:
    plan = _stage_plan(ctx, contract)
    plan_by_id = {step.stage_id: step for step in plan}
    return [plan_by_id[stage_id] for stage_id in PRODUCTION_STAGE_ORDER]


def _finalize_success(ctx: RiverV2Context, *, plan, stage_status, stage_results, aux_outputs):
    final_aux_outputs = _register_pipeline_outputs(ctx, stage_results, aux_outputs)
    final_result = _success_result(stage_status=stage_status, stage_results=stage_results, aux_outputs=final_aux_outputs)
    write_stage_products_overview(ctx.paths.stage_products_overview, root=ctx.paths.root, plan=plan, stage_status=stage_status, stage_results=stage_results, stage_numbers=_STAGE_NUMBERS)
    _write_stage_trace_snapshot(ctx, plan=plan, stage_status=stage_status, stage_results=stage_results)
    write_pipeline_summary(path=ctx.paths.pipeline_summary, result=final_result)
    return final_result


def _finalize_failure(ctx: RiverV2Context, *, plan, stage_status, stage_results, aux_outputs, failed_stage, error):
    failed_status = mark_v2_stage_failed(stage_status, stage_id=failed_stage, error=error)
    write_stage_products_overview(ctx.paths.stage_products_overview, root=ctx.paths.root, plan=plan, stage_status=failed_status, stage_results=stage_results, stage_numbers=_STAGE_NUMBERS, failed_stage=failed_stage, error=error)
    _write_stage_trace_snapshot(ctx, plan=plan, stage_status=failed_status, stage_results=stage_results, failed_stage=failed_stage, error=error)
    final_aux_outputs = _register_pipeline_outputs(ctx, stage_results, aux_outputs)
    return _failure_result(stage_status=failed_status, stage_results=stage_results, aux_outputs=final_aux_outputs, failed_stage=failed_stage, error=error)


def _run_stage(ctx: RiverV2Context, stage_status, stage_results, *, plan: list[StageStep], step: StageStep, aux_outputs: Dict[str, str]):
    result = step.run(stage_results)
    stage_results[step.stage_id] = result
    _remember_stage_outputs(ctx, step.stage_id, result, aux_outputs)
    stage_status = mark_v2_stage_implemented(
        stage_status,
        stage_id=step.stage_id,
        output_artifact=result.output_artifact,
        record_count=result.record_count,
        receipt_path=result.receipt_path,
        warnings=result.warnings,
    )
    write_stage_products_overview(ctx.paths.stage_products_overview, root=ctx.paths.root, plan=plan, stage_status=stage_status, stage_results=stage_results, stage_numbers=_STAGE_NUMBERS)
    _write_stage_trace_snapshot(ctx, plan=plan, stage_status=stage_status, stage_results=stage_results)
    return stage_status, result


def _run_production_stage_subset(ctx: RiverV2Context, *, target_stage_ids: Iterable[str]) -> RiverV2RunResult:
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    target_stage_ids = tuple(target_stage_ids)
    execution_contract = run_river_v2_execution_contract_preflight(ctx)
    plan_by_id = {step.stage_id: step for step in _build_production_plan(ctx, execution_contract)}
    try:
        plan = [plan_by_id[stage_id] for stage_id in target_stage_ids]
    except KeyError as exc:
        raise ValueError(f"Unsupported River v2 stage in subset: {exc.args[0]}") from exc
    stage_status = river_v2_stage_status_placeholder(PRODUCTION_STAGE_ORDER)
    stage_results: Dict[str, Any] = {}
    aux_outputs: Dict[str, str] = {}

    try:
        for step in plan:
            stage_status, _ = _run_stage(
                ctx,
                stage_status,
                stage_results,
                plan=plan,
                step=step,
                aux_outputs=aux_outputs,
            )
        return _finalize_success(ctx, plan=plan, stage_status=stage_status, stage_results=stage_results, aux_outputs=aux_outputs)
    except Exception as exc:
        failed_stage = _infer_failed_stage(stage_results, target_stage_ids=target_stage_ids)
        return _finalize_failure(
            ctx,
            plan=plan,
            stage_status=stage_status,
            stage_results=stage_results,
            aux_outputs=aux_outputs,
            failed_stage=failed_stage,
            error=str(exc),
        )


def run_river_v2_pipeline(ctx: RiverV2Context) -> RiverV2RunResult:
    return _run_production_stage_subset(ctx, target_stage_ids=PRODUCTION_STAGE_ORDER)


def _subset_result(full_result, *, stage_ids, execution_mode, result_cls):
    failed_stage = full_result.failed_stage if full_result.failed_stage in stage_ids else None
    error = full_result.error if failed_stage is not None else None
    success = all(stage_id in full_result.stage_results for stage_id in stage_ids) and failed_stage is None
    return _build_run_result(
        success=success,
        execution_mode=execution_mode,
        result_cls=result_cls,
        stage_status=full_result.stage_status,
        stage_results=full_result.stage_results,
        aux_outputs=full_result.aux_outputs,
        stage_ids=stage_ids,
        failed_stage=failed_stage,
        error=error,
    )


# Compatibility/test wrappers only. The active production path should use
# run_river_v2_pipeline(...), which owns the canonical stage order.
def run_river_v2_pass1(ctx: RiverV2Context, legacy_centerline_source=None) -> RiverV2Pass1Result:
    if legacy_centerline_source is not None:
        raise ValueError("legacy_centerline_source is no longer supported in the production River v2 context")
    full_result = _run_production_stage_subset(ctx, target_stage_ids=PASS1_STAGE_IDS)
    result = _subset_result(full_result, stage_ids=PASS1_STAGE_IDS, execution_mode=RIVER_V2_ROUTE_MODE_PASS1, result_cls=RiverV2Pass1Result)
    write_compatibility_pass_summary(pass1_path=ctx.paths.pass1_summary, pass2_path=ctx.paths.pass2_summary, pass3_path=ctx.paths.pass3_summary, pass4_path=ctx.paths.pass4_summary, result=result)
    return result


def run_river_v2_pass2(ctx: RiverV2Context, legacy_centerline_source=None) -> RiverV2Pass2Result:
    if legacy_centerline_source is not None:
        raise ValueError("legacy_centerline_source is no longer supported in the production River v2 context")
    full_result = _run_production_stage_subset(ctx, target_stage_ids=PASS2_STAGE_IDS)
    result = _subset_result(full_result, stage_ids=PASS2_STAGE_IDS, execution_mode=RIVER_V2_ROUTE_MODE_PASS2, result_cls=RiverV2Pass2Result)
    write_compatibility_pass_summary(pass1_path=ctx.paths.pass1_summary, pass2_path=ctx.paths.pass2_summary, pass3_path=ctx.paths.pass3_summary, pass4_path=ctx.paths.pass4_summary, result=result)
    return result


def run_river_v2_pass3(ctx: RiverV2Context, legacy_centerline_source=None) -> RiverV2Pass3Result:
    if legacy_centerline_source is not None:
        raise ValueError("legacy_centerline_source is no longer supported in the production River v2 context")
    full_result = _run_production_stage_subset(ctx, target_stage_ids=PASS3_STAGE_IDS)
    result = _subset_result(full_result, stage_ids=PASS3_STAGE_IDS, execution_mode=RIVER_V2_ROUTE_MODE_PASS3, result_cls=RiverV2Pass3Result)
    write_compatibility_pass_summary(pass1_path=ctx.paths.pass1_summary, pass2_path=ctx.paths.pass2_summary, pass3_path=ctx.paths.pass3_summary, pass4_path=ctx.paths.pass4_summary, result=result)
    return result


def run_river_v2_pass4(ctx: RiverV2Context, legacy_centerline_source=None) -> RiverV2Pass4Result:
    if legacy_centerline_source is not None:
        raise ValueError("legacy_centerline_source is no longer supported in the production River v2 context")
    result = _run_production_stage_subset(ctx, target_stage_ids=PASS4_STAGE_IDS)
    write_compatibility_pass_summary(pass1_path=ctx.paths.pass1_summary, pass2_path=ctx.paths.pass2_summary, pass3_path=ctx.paths.pass3_summary, pass4_path=ctx.paths.pass4_summary, result=result)
    return result


run_river_v2_full_pipeline = run_river_v2_pipeline


__all__ = [
    "run_river_v2_pipeline",
    "run_river_v2_full_pipeline",
    "run_river_v2_pass1",
    "run_river_v2_pass2",
    "run_river_v2_pass3",
    "run_river_v2_pass4",
    "_ensure_wse_support_source_raster",
]
