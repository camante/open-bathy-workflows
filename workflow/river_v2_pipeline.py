from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable
import logging

import numpy as np
import geopandas as gpd
import rasterio

from core.json_io import write_json

from constants import PIPELINE_VERSION

from river_v2_context import RiverV2Context
from river_v2_contract import (
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
    STAGE_CENTERLINE_OFFSET_MODELED,
    STAGE_CENTERLINE_WSE_PROXY,
    STAGE_RIVER_CENTERLINE,
    STAGE_RIVER_PRIMARY_SURFACE,
    STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
    mark_v2_stage_failed,
    mark_v2_stage_implemented,
    river_v2_stage_status_placeholder,
    subset_stage_results,
    subset_stage_status,
)
from river_v2_stage_authoritative_bed import run_authoritative_bed_stage
from river_v2_stage_authoritative_lock import build_authoritative_locked_primary_surface
from river_v2_stage_backbone import run_backbone_stage
from river_v2_stage_backbone_dense import run_backbone_dense_stage
from river_v2_stage_centerline import run_centerline_stage
from river_v2_stage_modeled_offset import run_modeled_offset_stage
from river_v2_stage_observed_offset import run_observed_offset_stage
from river_v2_stage_primary_surface import build_river_primary_surface
from river_v2_stage_wse_proxy import run_wse_proxy_stage
from river_v2_wse_support import build_river_v2_wse_support_products
from authoritative_guidance import build_projected_authoritative_sampling_raster, build_projected_measured_only_authoritative_sampling_raster


_STAGE_NUMBERS = {
    STAGE_RIVER_CENTERLINE: 1,
    STAGE_CENTERLINE_WSE_PROXY: 2,
    STAGE_CENTERLINE_AUTHORITATIVE_BED: 3,
    STAGE_CENTERLINE_OBSERVED_OFFSET: 4,
    STAGE_CENTERLINE_OFFSET_MODELED: 5,
    STAGE_CENTERLINE_BED_BACKBONE: 6,
    STAGE_CENTERLINE_BED_BACKBONE_DENSE: 7,
    STAGE_RIVER_PRIMARY_SURFACE: 8,
    STAGE_RIVER_PRIMARY_SURFACE_LOCKED: 9,
}





def _write_stage_products_overview(ctx: RiverV2Context, *, plan: list[StageStep], stage_status: dict, stage_results: dict, failed_stage: str | None = None, error: str | None = None) -> None:
    stages = []
    for step in plan:
        item = stage_status.get(step.stage_id, {}) if isinstance(stage_status, dict) else {}
        stage_no = _STAGE_NUMBERS.get(step.stage_id, 0)
        entry = {
            'stage_number': stage_no,
            'stage_id': step.stage_id,
            'status': item.get('status', 'not_run'),
            'implemented': bool(item.get('implemented', False)),
            'description': item.get('description'),
            'output_artifact': item.get('output_artifact'),
            'receipt_path': item.get('receipt_path'),
        }
        if step.stage_id in stage_results:
            entry['record_count'] = getattr(stage_results[step.stage_id], 'record_count', None)
        stages.append(entry)
    overview = {
        'root': str(ctx.paths.root),
        'pipeline_version': str(PIPELINE_VERSION),
        'river_method_selected': 'v2',
        'river_path_used': 'river_v2_only',
        'legacy_river_path_participated': False,
        'success': failed_stage is None,
        'failed_stage': failed_stage,
        'error': error,
        'stages': stages,
        'lineage': [
            {
                'stage_number': entry['stage_number'],
                'stage_id': entry['stage_id'],
                'output_artifact': entry.get('output_artifact'),
                'receipt_path': entry.get('receipt_path'),
            }
            for entry in stages
        ],
    }
    ctx.paths.stage_products_dir.mkdir(parents=True, exist_ok=True)
    ctx.paths.stage_products_overview.write_text(json.dumps(_jsonable(overview), indent=2, sort_keys=True), encoding='utf-8')




@dataclass(frozen=True)
class ResolvedRiverV2Inputs:
    network_gpkg: Path | None
    river_dem_path: Path | None
    authoritative_sampling_raster_path: Path | None
    centerline_spacing_m: float
    min_stream_order: int
    authoritative_bed_support_points_path: Path | None
    channel_mask_path: Path | None
    authoritative_lock_base_path: Path | None


def _resolve_river_v2_inputs(ctx: RiverV2Context) -> ResolvedRiverV2Inputs:
    authoritative_sampling_raster_path = _ensure_authoritative_sampling_support(ctx)
    authoritative_bed_support_points_path = _ensure_authoritative_bed_support_points(ctx)
    return ResolvedRiverV2Inputs(
        network_gpkg=Path(ctx.network_gpkg) if ctx.network_gpkg is not None else None,
        river_dem_path=Path(ctx.river_dem_path) if ctx.river_dem_path is not None else None,
        authoritative_sampling_raster_path=Path(authoritative_sampling_raster_path) if authoritative_sampling_raster_path is not None else None,
        centerline_spacing_m=float(ctx.centerline_spacing_m or 25.0),
        min_stream_order=max(int(getattr(ctx.cfg, "river_mainstem_min_order", 5) or 5), 1),
        authoritative_bed_support_points_path=Path(authoritative_bed_support_points_path) if authoritative_bed_support_points_path is not None else None,
        channel_mask_path=Path(ctx.channel_mask_path) if ctx.channel_mask_path is not None else None,
        authoritative_lock_base_path=(Path(ctx.authoritative_base_path) if ctx.authoritative_base_path is not None and Path(ctx.authoritative_base_path).exists() else (Path(ctx.aligned_authoritative_base_path) if ctx.aligned_authoritative_base_path is not None and Path(ctx.aligned_authoritative_base_path).exists() else None)),
    )

@dataclass(frozen=True)
class StageStep:
    stage_id: str
    run: Callable[[Dict[str, Any]], Any]


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


def _write_summary(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


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



def _write_pipeline_summary(*, ctx: RiverV2Context, result: RiverV2Pass4Result) -> Path:
    return _write_summary(ctx.paths.pipeline_summary, result.to_dict())


def _write_compatibility_pass_summary(*, ctx: RiverV2Context, result) -> None:
    stage_ids = tuple(result.stage_results.keys())
    summary_path = None
    if stage_ids == PASS1_STAGE_IDS:
        summary_path = ctx.paths.pass1_summary
    elif stage_ids == PASS2_STAGE_IDS:
        summary_path = ctx.paths.pass2_summary
    elif stage_ids == PASS3_STAGE_IDS:
        summary_path = ctx.paths.pass3_summary
    elif stage_ids == PASS4_STAGE_IDS:
        summary_path = ctx.paths.pass4_summary
    if summary_path is not None:
        _write_summary(summary_path, result.to_dict())


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
    STAGE_CENTERLINE_WSE_PROXY,
    STAGE_CENTERLINE_AUTHORITATIVE_BED,
    STAGE_CENTERLINE_OBSERVED_OFFSET,
    STAGE_CENTERLINE_OFFSET_MODELED,
    STAGE_CENTERLINE_BED_BACKBONE,
    STAGE_CENTERLINE_BED_BACKBONE_DENSE,
    STAGE_RIVER_PRIMARY_SURFACE,
    STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
)


def _register_pipeline_outputs(ctx: RiverV2Context, stage_results: Dict[str, Any], aux_outputs: Dict[str, str]) -> Dict[str, str]:
    outputs = dict(aux_outputs or {})
    outputs.update({
        "pipeline_version": str(PIPELINE_VERSION),
        "river_method_selected": "v2",
        "river_path_used": "river_v2_only",
        "legacy_river_path_participated": "false",
        "river_v2_summary": str(ctx.paths.pipeline_summary),
        "river_v2_stage_products_overview": str(ctx.paths.stage_products_overview),
    })
    canonical_keys = {
        STAGE_RIVER_CENTERLINE: "river_v2_centerline_points",
        STAGE_CENTERLINE_WSE_PROXY: "river_v2_wse_points",
        STAGE_CENTERLINE_AUTHORITATIVE_BED: "river_v2_authoritative_bed_points",
        STAGE_CENTERLINE_OBSERVED_OFFSET: "river_v2_observed_offset_points",
        STAGE_CENTERLINE_OFFSET_MODELED: "river_v2_modeled_offset_points",
        STAGE_CENTERLINE_BED_BACKBONE: "river_v2_backbone_points",
        STAGE_CENTERLINE_BED_BACKBONE_DENSE: "river_v2_backbone_dense_points",
        STAGE_RIVER_PRIMARY_SURFACE: "river_v2_primary_surface",
        STAGE_RIVER_PRIMARY_SURFACE_LOCKED: "river_v2_primary_surface_authoritative_applied",
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


def _remember_stage_outputs(stage_id: str, result: Any, aux_outputs: Dict[str, str]) -> None:
    if result.aux_outputs:
        aux_outputs.update({str(k): str(v) for k, v in result.aux_outputs.items()})
    if stage_id == STAGE_RIVER_PRIMARY_SURFACE:
        aux_outputs["river_primary_surface"] = str(result.output_artifact)
    elif stage_id == STAGE_RIVER_PRIMARY_SURFACE_LOCKED:
        aux_outputs["river_primary_surface_authoritative_applied"] = str(result.output_artifact)


def _rewrite_local_artifact_from_source(*, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        rel = os.path.relpath(source, destination.parent)
        destination.symlink_to(rel)
    except Exception:
        shutil.copy2(source, destination)



def _resolve_sampling_aoi(ctx: RiverV2Context) -> str:
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


def _raster_is_projected(path: Path) -> bool:
    try:
        with rasterio.open(path) as ds:
            return bool(ds.crs is not None and not getattr(ds.crs, 'is_geographic', False))
    except Exception:
        return False


def _ensure_authoritative_sampling_support(ctx: RiverV2Context) -> Path | None:
    if ctx.authoritative_sampling_raster_path is not None and Path(ctx.authoritative_sampling_raster_path).exists():
        return Path(ctx.authoritative_sampling_raster_path)
    source_raster = None
    if ctx.authoritative_base_path is not None and Path(ctx.authoritative_base_path).exists():
        source_raster = Path(ctx.authoritative_base_path)
    elif ctx.aligned_authoritative_base_path is not None and Path(ctx.aligned_authoritative_base_path).exists():
        source_raster = Path(ctx.aligned_authoritative_base_path)
    if source_raster is None:
        return None

    trusted_mode = str(getattr(ctx, 'trusted_support_mode', 'low_support_no_trusted_support') or 'low_support_no_trusted_support')
    trusted_artifact = Path(ctx.trusted_support_artifact_path) if getattr(ctx, 'trusted_support_artifact_path', None) is not None else None
    support_dir = ctx.paths.support_dir
    support_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.paths.authoritative_measured_only_projected
    if out_path.exists() and out_path.stat().st_size > 0:
        ctx.authoritative_sampling_raster_path = out_path
        return out_path

    if trusted_mode == 'low_support_no_trusted_support':
        write_json(ctx.paths.authoritative_sampling_summary, {
            'status': 'skipped',
            'sampling_raster_path': None,
            'sampling_contract': 'no_trusted_support_low_support_mode',
            'source_raster': str(source_raster),
            'support_coverage_path': str(trusted_artifact) if trusted_artifact else None,
            'trusted_support_mode': trusted_mode,
            'warning': getattr(ctx, 'authoritative_support_policy_warning', None),
        })
        return None

    logger = logging.getLogger(__name__)
    with rasterio.open(ctx.river_dem_path) as river_ds:
        dst_crs = river_ds.crs.to_string() if river_ds.crs else None
        res_m = abs(float(river_ds.transform.a)) if river_ds.transform else None
    if not dst_crs or not np.isfinite(float(res_m)) or float(res_m) <= 0:
        raise RuntimeError('river_v2_authoritative_sampling_missing_projected_grid')

    if trusted_mode == 'all_finite_cells_trusted':
        info = build_projected_authoritative_sampling_raster(
            source_raster,
            out_path,
            aoi=_resolve_sampling_aoi(ctx),
            dst_crs=str(dst_crs),
            res_m=float(res_m),
            logger=logger,
        )
        sampling_contract = 'all_finite_cells_trusted_projected'
        support_path = None
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
            aoi=_resolve_sampling_aoi(ctx),
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
        'projection_info': info,
    })
    return ctx.authoritative_sampling_raster_path



def _resolve_authoritative_bed_support_source(ctx: RiverV2Context) -> Path | None:
    candidates = []
    if ctx.authoritative_bed_path is not None:
        candidates.append(Path(ctx.authoritative_bed_path))
    outputs = ctx.report.get("outputs", {}) if isinstance(ctx.report, dict) else {}
    if isinstance(outputs, dict):
        for key in ("river_authoritative_soundings", "river_authoritative_bed"):
            value = outputs.get(key)
            if value:
                candidates.append(Path(value))
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _ensure_authoritative_bed_support_points(ctx: RiverV2Context) -> Path | None:
    existing = ctx.paths.authoritative_bed_support_points
    if existing.exists() and existing.stat().st_size > 0:
        ctx.authoritative_bed_path = existing
        return existing
    source = _resolve_authoritative_bed_support_source(ctx)
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

def _centerline_has_inline_wse_support(centerline_points_path: Path) -> bool:
    centerline = gpd.read_file(centerline_points_path, rows=1)
    columns = set(centerline.columns)
    return any(field in columns for field in (
        "bank_wse_proxy_monotone_m",
        "bank_wse_proxy_interp_m",
        "bank_wse_proxy_raw_m",
        "wse_proxy_z_m",
    ))


def _ensure_wse_support_source_raster(ctx: RiverV2Context) -> tuple[Path | None, str | None]:
    out_path = ctx.paths.wse_support_source_raster
    candidates: list[tuple[Path | None, str]] = [
        (Path(ctx.river_dem_path), "river_dem_template") if ctx.river_dem_path is not None else (None, "river_dem_template"),
        (Path(ctx.authoritative_base_path), "authoritative_base") if ctx.authoritative_base_path is not None else (None, "authoritative_base"),
        (Path(ctx.aligned_authoritative_base_path), "aligned_authoritative_base") if ctx.aligned_authoritative_base_path is not None else (None, "aligned_authoritative_base"),
        (Path(ctx.authoritative_sampling_raster_path), "measured_only_authoritative_sampling_raster") if ctx.authoritative_sampling_raster_path is not None else (None, "measured_only_authoritative_sampling_raster"),
    ]
    for source, contract in candidates:
        if source is None or not source.exists():
            continue
        _rewrite_local_artifact_from_source(source=source, destination=out_path)
        return out_path, contract
    return None, None


def _ensure_explicit_wse_support(ctx: RiverV2Context, *, centerline_points_path: Path) -> dict[str, str]:
    existing: dict[str, str] = {}
    if ctx.bank_wse_edge_guidance_path is not None and Path(ctx.bank_wse_edge_guidance_path).exists():
        existing["bank_wse_edge_guidance_path"] = str(Path(ctx.bank_wse_edge_guidance_path))
    if existing.get("bank_wse_edge_guidance_path"):
        return existing

    support_dir = ctx.paths.support_dir
    support_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = support_dir / "river_bank_wse_materialization_diagnostics.json"

    if ctx.channel_mask_path is None or not Path(ctx.channel_mask_path).exists():
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

    support_source, source_contract = _ensure_wse_support_source_raster(ctx)
    if support_source is None or source_contract is None:
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
            channel_mask_tif=Path(ctx.channel_mask_path),
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


def _run_wse_stage(ctx: RiverV2Context, inputs: ResolvedRiverV2Inputs, results: Dict[str, Any]):
    centerline_path = results[STAGE_RIVER_CENTERLINE].output_artifact
    inline_support = _centerline_has_inline_wse_support(centerline_path)
    support_paths: dict[str, str] = {}
    edge_path: str | None = None
    if not inline_support:
        support_paths = _ensure_explicit_wse_support(ctx, centerline_points_path=centerline_path)
        edge_path = support_paths.get("bank_wse_edge_guidance_path")
        if not edge_path:
            raise RuntimeError("river_v2_wse_missing_explicit_bank_wse_support_raster")
    return run_wse_proxy_stage(
        ctx,
        centerline_points_path=centerline_path,
        bank_wse_edge_guidance_path=Path(edge_path) if edge_path else None,
        bank_wse_profile_summary_path=None,
    )




def _stage_plan(ctx: RiverV2Context, inputs: ResolvedRiverV2Inputs) -> list[StageStep]:
    return [
        StageStep(
            stage_id=STAGE_RIVER_CENTERLINE,
            run=lambda _results: run_centerline_stage(
                ctx,
                network_gpkg=inputs.network_gpkg,
                river_dem_path=inputs.river_dem_path,
                sampling_raster_path=inputs.authoritative_sampling_raster_path,
                spacing_m=inputs.centerline_spacing_m,
                min_stream_order=inputs.min_stream_order,
            ),
        ),
        StageStep(
            stage_id=STAGE_CENTERLINE_WSE_PROXY,
            run=lambda results: _run_wse_stage(ctx, inputs, results),
        ),
        StageStep(
            stage_id=STAGE_CENTERLINE_AUTHORITATIVE_BED,
            run=lambda results: run_authoritative_bed_stage(
                ctx,
                centerline_points_path=results[STAGE_RIVER_CENTERLINE].output_artifact,
                authoritative_support_points_path=inputs.authoritative_bed_support_points_path,
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
            stage_id=STAGE_CENTERLINE_OFFSET_MODELED,
            run=lambda results: run_modeled_offset_stage(
                ctx,
                wse_points_path=results[STAGE_CENTERLINE_WSE_PROXY].output_artifact,
                observed_offset_points_path=results[STAGE_CENTERLINE_OBSERVED_OFFSET].output_artifact,
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
                channel_mask_path=inputs.channel_mask_path,
                centerline_points_path=results[STAGE_RIVER_CENTERLINE].output_artifact,
            ),
        ),
        StageStep(
            stage_id=STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
            run=lambda results: build_authoritative_locked_primary_surface(
                ctx,
                primary_surface_path=results[STAGE_RIVER_PRIMARY_SURFACE].output_artifact,
                authoritative_base_path=inputs.authoritative_lock_base_path,
            ),
        ),
    ]


def _build_production_plan(ctx: RiverV2Context, inputs: ResolvedRiverV2Inputs) -> list[StageStep]:
    plan = _stage_plan(ctx, inputs)
    plan_by_id = {step.stage_id: step for step in plan}
    return [plan_by_id[stage_id] for stage_id in PRODUCTION_STAGE_ORDER]


def _finalize_success(ctx: RiverV2Context, *, plan, stage_status, stage_results, aux_outputs):
    final_aux_outputs = _register_pipeline_outputs(ctx, stage_results, aux_outputs)
    final_result = _success_result(stage_status=stage_status, stage_results=stage_results, aux_outputs=final_aux_outputs)
    _write_stage_products_overview(ctx, plan=plan, stage_status=stage_status, stage_results=stage_results)
    _write_pipeline_summary(ctx=ctx, result=final_result)
    return final_result


def _finalize_failure(ctx: RiverV2Context, *, plan, stage_status, stage_results, aux_outputs, failed_stage, error):
    failed_status = mark_v2_stage_failed(stage_status, stage_id=failed_stage, error=error)
    _write_stage_products_overview(ctx, plan=plan, stage_status=failed_status, stage_results=stage_results, failed_stage=failed_stage, error=error)
    final_aux_outputs = _register_pipeline_outputs(ctx, stage_results, aux_outputs)
    return _failure_result(stage_status=failed_status, stage_results=stage_results, aux_outputs=final_aux_outputs, failed_stage=failed_stage, error=error)


def _run_stage(ctx: RiverV2Context, stage_status, stage_results, *, plan: list[StageStep], step: StageStep, aux_outputs: Dict[str, str]):
    result = step.run(stage_results)
    stage_results[step.stage_id] = result
    _remember_stage_outputs(step.stage_id, result, aux_outputs)
    stage_status = mark_v2_stage_implemented(
        stage_status,
        stage_id=step.stage_id,
        output_artifact=result.output_artifact,
        record_count=result.record_count,
        receipt_path=result.receipt_path,
        warnings=result.warnings,
    )
    _write_stage_products_overview(ctx, plan=plan, stage_status=stage_status, stage_results=stage_results)
    return stage_status, result


def _run_production_stage_subset(ctx: RiverV2Context, *, target_stage_ids: Iterable[str]) -> RiverV2RunResult:
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    target_stage_ids = tuple(target_stage_ids)
    plan_by_id = {step.stage_id: step for step in _build_production_plan(ctx, _resolve_river_v2_inputs(ctx))}
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
    _write_compatibility_pass_summary(ctx=ctx, result=result)
    return result


def run_river_v2_pass2(ctx: RiverV2Context, legacy_centerline_source=None) -> RiverV2Pass2Result:
    if legacy_centerline_source is not None:
        raise ValueError("legacy_centerline_source is no longer supported in the production River v2 context")
    full_result = _run_production_stage_subset(ctx, target_stage_ids=PASS2_STAGE_IDS)
    result = _subset_result(full_result, stage_ids=PASS2_STAGE_IDS, execution_mode=RIVER_V2_ROUTE_MODE_PASS2, result_cls=RiverV2Pass2Result)
    _write_compatibility_pass_summary(ctx=ctx, result=result)
    return result


def run_river_v2_pass3(ctx: RiverV2Context, legacy_centerline_source=None) -> RiverV2Pass3Result:
    if legacy_centerline_source is not None:
        raise ValueError("legacy_centerline_source is no longer supported in the production River v2 context")
    full_result = _run_production_stage_subset(ctx, target_stage_ids=PASS3_STAGE_IDS)
    result = _subset_result(full_result, stage_ids=PASS3_STAGE_IDS, execution_mode=RIVER_V2_ROUTE_MODE_PASS3, result_cls=RiverV2Pass3Result)
    _write_compatibility_pass_summary(ctx=ctx, result=result)
    return result


def run_river_v2_pass4(ctx: RiverV2Context, legacy_centerline_source=None) -> RiverV2Pass4Result:
    if legacy_centerline_source is not None:
        raise ValueError("legacy_centerline_source is no longer supported in the production River v2 context")
    result = _run_production_stage_subset(ctx, target_stage_ids=PASS4_STAGE_IDS)
    _write_compatibility_pass_summary(ctx=ctx, result=result)
    return result


run_river_v2_full_pipeline = run_river_v2_pipeline


__all__ = [
    "run_river_v2_pipeline",
    "run_river_v2_full_pipeline",
    "run_river_v2_pass1",
    "run_river_v2_pass2",
    "run_river_v2_pass3",
    "run_river_v2_pass4",
]
