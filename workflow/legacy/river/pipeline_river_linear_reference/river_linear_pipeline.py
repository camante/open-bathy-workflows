from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Any

from core.json_io import write_json
from pipeline.river_linear.river_linear_context import RiverLinearContext
from pipeline.river_linear.river_linear_execution_contract import AOI_EXPORT_ONLY_ROLE, build_execution_contract, ensure_stage_allowed
from canonical_river_solve_cache import load_canonical_solve_cache, write_canonical_solve_cache
from pipeline.river_linear.river_linear_contract import LINEAR_STAGE_ORDER, validate_run_contract
from pipeline.river_linear.river_linear_receipts import (
    build_stage_trace_entry,
    build_stage_validator_summary,
    write_bundle_manifest,
    write_canonical_cache_consistency_receipt,
    write_canonical_solve_identity_manifest,
    write_canonical_stage_identity_manifest,
    write_final_dem_writer_receipt,
    write_river_verification_summary,
    write_receipt_purpose_manifest,
    write_stage_receipt,
    write_standard_adjacent_aoi_verification_receipt,
    write_trace_summary,
)
from pipeline.river_linear.river_linear_stage_authoritative import AuthoritativeStageResult, run_authoritative_stage
from pipeline.river_linear.river_linear_stage_authoritative_bed import AuthoritativeBedStageResult, run_authoritative_bed_stage
from pipeline.river_linear.river_linear_stage_backbone import (
    BackboneStageResult,
    run_backbone_stage,
    validate_backbone_export_subset_flatness,
)
from pipeline.river_linear.river_linear_stage_centerline import CenterlineStageResult, run_centerline_stage
from pipeline.river_linear.river_linear_stage_corridor import CorridorStageResult, run_corridor_stage
from pipeline.river_linear.river_linear_stage_export import ExportStageResult, run_export_stage
from pipeline.river_linear.river_linear_stage_final_dem import FinalDemStageResult, run_final_dem_stage
from pipeline.river_linear.river_linear_stage_grids import GridStageResult, run_grid_stage, _project_bounds, _write_subset_template_from_parent
from pipeline.river_linear.river_linear_stage_lock import LockStageResult, run_lock_stage, write_authoritative_lock_contract_for_paths
from pipeline.river_linear.river_linear_stage_modeled_offset import ModeledOffsetStageResult, run_modeled_offset_stage
from pipeline.river_linear.river_linear_stage_observed_offset import ObservedOffsetStageResult, run_observed_offset_stage
from pipeline.river_linear.river_linear_stage_solve_domain import SolveDomainStageResult, run_solve_domain_stage
from pipeline.river_linear.river_linear_stage_surface import SurfaceStageResult, run_surface_stage
from pipeline.river_linear.river_linear_stage_wse_proxy import WSEProxyStageResult, run_wse_proxy_stage
from pipeline.river_linear.river_linear_science_audit import write_river_science_chain_summary
from pipeline.river_linear.river_linear_export_logic import subset_float_raster_to_template_exact
from pipeline.river_linear.river_linear_identity import assert_parent_window_matches_export
from pipeline.river_linear.river_linear_canonical_manifest import (
    canonical_solution_cache_manifest_path,
    load_canonical_solution_manifest,
    public_canonical_solution_manifest_path,
    raster_identity,
    sha256_file,
    write_canonical_solution_manifest,
)


@dataclass(frozen=True)
class RiverLinearPipelineResult:
    run_contract_path: Optional[Path]
    shared_solve_reused: bool
    solve_result: SolveDomainStageResult
    grid_result: GridStageResult
    authoritative_result: AuthoritativeStageResult
    centerline_result: CenterlineStageResult
    wse_result: WSEProxyStageResult
    authoritative_bed_result: AuthoritativeBedStageResult
    observed_offset_result: ObservedOffsetStageResult
    modeled_offset_result: ModeledOffsetStageResult
    backbone_result: BackboneStageResult
    corridor_result: CorridorStageResult
    surface_result: SurfaceStageResult
    lock_result: LockStageResult
    export_result: ExportStageResult
    final_dem_result: FinalDemStageResult
    stage_receipts: dict[str, Path]
    bundle_manifest_path: Optional[Path]
    canonical_solve_identity_manifest_path: Optional[Path]
    canonical_system_identity_receipt_path: Optional[Path]
    canonical_stage_identity_manifest_path: Optional[Path]
    canonical_cache_consistency_receipt_path: Optional[Path]
    adjacent_aoi_verification_receipt_path: Optional[Path]
    river_verification_summary_path: Optional[Path]
    final_dem_writer_receipt_path: Optional[Path]
    river_science_chain_summary_path: Optional[Path]


@dataclass(frozen=True)
class CanonicalParentHandoff:
    """Single handoff object from canonical parent resolution to AOI export.

    Cache reuse, existing canonical manifests, and identity-only adoption all
    collapse into this object before AOI export. AOI export should not care how
    the parent was obtained; it may only read the manifest/parent described here
    and write an exact AOI subset.
    """

    manifest_path: Path
    parent_dem_path: Path
    cache_manifest_path: Path | None
    source_kind: str
    reused_from_cache: bool
    adopted_from_existing_parent: bool
    canonical_system_id: str | None
    canonical_cache_key: str | None
    canonical_parent_hash: str | None
    source_context_path: Path | None = None
    manifest_payload: dict[str, Any] | None = None

    def source_summary(self) -> dict[str, Any]:
        """Return a compact, JSON-safe summary for receipts/logging."""
        return {
            "source_kind": self.source_kind,
            "manifest_path": str(self.manifest_path),
            "parent_dem_path": str(self.parent_dem_path),
            "cache_manifest_path": str(self.cache_manifest_path) if self.cache_manifest_path is not None else None,
            "source_context_path": str(self.source_context_path) if self.source_context_path is not None else None,
            "reused_from_cache": bool(self.reused_from_cache),
            "adopted_from_existing_parent": bool(self.adopted_from_existing_parent),
            "canonical_system_id": self.canonical_system_id,
            "canonical_cache_key": self.canonical_cache_key,
            "canonical_parent_hash": self.canonical_parent_hash,
        }



@dataclass(frozen=True)
class CanonicalRiverBuildResult:
    """Typed wrapper for a freshly constructed canonical parent workflow result."""

    result: RiverLinearPipelineResult


@dataclass(frozen=True)
class AOIExportResult:
    """Typed wrapper for an AOI export-only workflow result."""

    result: RiverLinearPipelineResult


@dataclass(frozen=True)
class RiverParentExportWorkflowResult:
    """Typed wrapper for the active parent/export river workflow result."""

    result: RiverLinearPipelineResult



def _read_shared_solve_cache_key(ctx: RiverLinearContext) -> str | None:
    contract_path = getattr(ctx.linear_inputs, 'canonical_solve_contract_path', None) if ctx.linear_inputs is not None else None
    if contract_path is None:
        return None
    try:
        payload = json.loads(Path(contract_path).read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return None
    key = payload.get('canonical_solve_cache_key')
    return str(key) if key not in (None, '') else None


def _build_run_contract(
    ctx: RiverLinearContext,
    *,
    shared_solve_reused: bool,
    shared_solve_cache_manifest_path: Path | None,
) -> dict[str, Any]:
    linear_input_bundle = ctx.linear_inputs.to_dict() if ctx.linear_inputs is not None else None
    if isinstance(linear_input_bundle, dict):
        linear_input_bundle.setdefault('requested_solve_domain', (str(ctx.requested_solve_domain) if ctx.requested_solve_domain is not None else None))
        linear_input_bundle['resolved_solve_domain'] = str(
            linear_input_bundle.get('resolved_solve_domain')
            or ctx.resolved_solve_domain
            or (ctx.linear_inputs.solve_source_aoi if ctx.linear_inputs is not None else '')
        )
        linear_input_bundle['solve_domain_source'] = str(linear_input_bundle.get('solve_domain_source') or ctx.solve_domain_source or 'derived_from_aoi')
    payload = {
        'run_id': str(ctx.run_id),
        'workflow_name': str(ctx.workflow_name),
        'export_aoi': str(ctx.export_aoi),
        'requested_solve_domain': (str(ctx.requested_solve_domain) if ctx.requested_solve_domain is not None else None),
        'resolved_solve_domain': (str(ctx.resolved_solve_domain) if ctx.resolved_solve_domain is not None else None),
        'solve_domain_source': str(ctx.solve_domain_source),
        'projected_crs': str(ctx.projected_crs),
        'target_resolution_m': float(ctx.target_resolution_m),
        'canonical_max_trace_km': float(ctx.canonical_max_trace_km),
        'linear_input_bundle': linear_input_bundle,
        'shared_solve': {
            'reused': bool(shared_solve_reused),
            'cache_manifest_path': (str(shared_solve_cache_manifest_path) if shared_solve_cache_manifest_path is not None else None),
            'contract_path': (str(ctx.linear_inputs.canonical_solve_contract_path) if ctx.linear_inputs is not None and ctx.linear_inputs.canonical_solve_contract_path is not None else None),
            'cache_key': _read_shared_solve_cache_key(ctx),
            'canonical_source_context_path': (str(ctx.linear_inputs.canonical_source_context_path) if ctx.linear_inputs is not None and ctx.linear_inputs.canonical_source_context_path is not None else None),
            'canonical_solve_identity_path': (str(ctx.linear_inputs.canonical_solve_identity_path) if ctx.linear_inputs is not None and getattr(ctx.linear_inputs, 'canonical_solve_identity_path', None) is not None else None),
        },
        'execution_contract': build_execution_contract(ctx.execution_role).to_dict(),
        'stage_order': list(LINEAR_STAGE_ORDER),
        'artifacts': {
            'run_contract': str(ctx.paths.run_contract),
            'canonical_solve_network': str(ctx.paths.canonical_solve_network),
            'canonical_solve_aoi': str(ctx.paths.canonical_solve_aoi),
            'solve_grid_template': str(ctx.paths.solve_grid_template),
            'export_grid_template': str(ctx.paths.export_grid_template),
            'solve_authoritative_base_measured_only': str(ctx.paths.solve_authoritative_base_measured_only),
            'solve_authoritative_support_mask': str(ctx.paths.solve_authoritative_support_mask),
            'export_authoritative_base_measured_only': str(ctx.paths.export_authoritative_base_measured_only),
            'export_authoritative_support_mask': str(ctx.paths.export_authoritative_support_mask),
            'export_baseline_background': str(ctx.paths.export_baseline_background),
            'centerline_points': str(ctx.paths.centerline_points),
            'centerline_wse_support_points': str(ctx.paths.centerline_wse_support_points),
            'centerline_wse_trend_points': str(ctx.paths.centerline_wse_trend_points),
            'centerline_wse_pre_smooth_points': str(ctx.paths.centerline_wse_pre_smooth_points),
            'centerline_wse_proxy_points': str(ctx.paths.centerline_wse_proxy_points),
            'centerline_authoritative_bed_points': str(ctx.paths.centerline_authoritative_bed_points),
            'centerline_observed_offset_points': str(ctx.paths.centerline_observed_offset_points),
            'centerline_modeled_offset_points': str(ctx.paths.centerline_modeled_offset_points),
            'centerline_bed_backbone_points': str(ctx.paths.centerline_bed_backbone_points),
            'river_corridor_solve': str(ctx.paths.river_corridor_solve),
            'river_primary_surface_solve': str(ctx.paths.river_primary_surface_solve),
            'river_primary_surface_solve_locked': str(ctx.paths.river_primary_surface_solve_locked),
            'river_guidance_export': str(ctx.paths.river_guidance_export),
            'river_corridor_mask_export': str(ctx.paths.river_corridor_mask_export),
            'river_support_mask_export': str(ctx.paths.river_support_mask_export),
            'river_take_mask_export': str(ctx.paths.river_take_mask_export),
            'dem_enhanced_final': str(ctx.paths.dem_enhanced_final),
        },
    }
    validate_run_contract(payload)
    if ctx.write_diagnostics or getattr(ctx, "write_core_receipts", True):
        write_json(ctx.paths.run_contract, payload)
    return payload


def _record_stage(
    *,
    ctx: RiverLinearContext,
    stage_trace: list[dict[str, Any]],
    receipt_path: Path,
    stage_name: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    summary: dict[str, Any],
) -> Optional[Path]:
    validator = build_stage_validator_summary(stage_name, status='passed')
    should_write = bool(ctx.write_diagnostics or (getattr(ctx, "write_core_receipts", True) and stage_name == 'final_dem'))
    if not should_write:
        return None
    receipt = write_stage_receipt(
        receipt_path,
        stage_name=stage_name,
        inputs=inputs,
        outputs=outputs,
        summary=summary,
        validator=validator,
    )
    if ctx.write_diagnostics:
        stage_trace.append(
            build_stage_trace_entry(
                stage_name=stage_name,
                receipt_path=receipt,
                inputs=inputs,
                outputs=outputs,
                summary=summary,
                validator=validator,
            )
        )
    return receipt




def _existing_canonical_manifest_for_aoi_export(ctx: RiverLinearContext) -> Path | None:
    manifest_path = canonical_solution_cache_manifest_path(ctx)
    if manifest_path is not None and manifest_path.is_file():
        return manifest_path
    return None




def _bootstrap_cache_manifest_for_existing_parent(ctx: RiverLinearContext, parent_dem: Path | None) -> Path | None:
    """Create required cache manifest for an older existing canonical parent.

    Identity-only migration: this writes metadata for an already-existing parent
    DEM and does not run canonical construction stages.
    """
    if parent_dem is None or not Path(parent_dem).is_file():
        return None
    manifest_path = canonical_solution_cache_manifest_path(ctx)
    if manifest_path is None:
        return None
    if manifest_path.is_file():
        return manifest_path
    bundle = ctx.linear_inputs
    solve_grid_value = getattr(bundle, "canonical_solve_grid_template_path", None) if bundle is not None else None
    solve_grid_path = Path(solve_grid_value) if solve_grid_value not in (None, "") and Path(solve_grid_value).is_file() else Path(parent_dem)
    cache_key = getattr(bundle, "canonical_solve_cache_key", None) if bundle is not None else None
    canonical_system_id = getattr(bundle, "canonical_system_id", None) if bundle is not None else None
    payload = {
        "schema_version": 2,
        "role": "canonical_river_solution",
        "execution_model": "canonical_parent_plus_aoi_export",
        "manifest_creation_mode": "adopt_existing_canonical_parent_identity_only",
        "canonical_stage_class": "canonical_parent_finalization",
        "canonical_system_id": str(canonical_system_id) if canonical_system_id not in (None, "") else None,
        "canonical_cache_key": str(cache_key) if cache_key not in (None, "") else None,
        "canonical_solve_cache_key": str(cache_key) if cache_key not in (None, "") else None,
        "requested_solve_domain": str(ctx.requested_solve_domain) if ctx.requested_solve_domain is not None else None,
        "resolved_solve_domain": str(ctx.resolved_solve_domain) if ctx.resolved_solve_domain is not None else None,
        "canonical_solve_aoi": str(getattr(bundle, "canonical_solve_aoi", None) or ctx.canonical_domain_bounds or ctx.resolved_solve_domain or ""),
        "solve_domain_source": str(ctx.solve_domain_source),
        "projected_crs": str(ctx.projected_crs),
        "target_resolution_m": float(ctx.target_resolution_m),
        "source_context_path": str(getattr(bundle, "canonical_source_context_path", None)) if bundle is not None and getattr(bundle, "canonical_source_context_path", None) is not None else None,
        "canonical_parent_dem_path": str(parent_dem),
        "canonical_parent_dem_sha256": sha256_file(parent_dem),
        "canonical_parent_content_key": sha256_file(parent_dem),
        "canonical_comparison_key": sha256_file(parent_dem),
        "canonical_final_dem_path": str(parent_dem),
        "canonical_final_dem_sha256": sha256_file(parent_dem),
        "canonical_take_mask_path": str(getattr(bundle, "canonical_solve_take_mask_path", None)) if bundle is not None and getattr(bundle, "canonical_solve_take_mask_path", None) is not None else None,
        "canonical_take_mask_sha256": sha256_file(getattr(bundle, "canonical_solve_take_mask_path", None)) if bundle is not None else None,
        "solve_grid": raster_identity(solve_grid_path),
        "canonical_parent_dem": raster_identity(parent_dem),
        "authoritative_measured_only": raster_identity(getattr(bundle, "canonical_solve_authoritative_measured_only_path", None) if bundle is not None else None),
        "authoritative_support_mask": raster_identity(getattr(bundle, "canonical_solve_authoritative_support_mask_path", None) if bundle is not None else None),
        "baseline_background": raster_identity(getattr(bundle, "canonical_solve_baseline_background_path", None) if bundle is not None else None),
        "locked_river_surface": raster_identity(parent_dem),
        "composition_summary": {"adopted_existing_parent": True},
        "canonical_parent_policy": {
            "parent_built_once_on_solve_grid": True,
            "aoi_outputs_must_be_exact_parent_subsets": True,
            "post_subset_modification_allowed": False,
        },
        "aoi_export_policy": {
            "aoi_runs_may_recompute_canonical_construction": False,
            "aoi_runs_must_subset_from_parent": True,
            "cache_is_implementation_detail": True,
            "manifest_handoff_required": True,
            "post_subset_modification_allowed": False,
            "exact_parent_window_identity_required": True,
        },
    }
    write_json(manifest_path, payload)
    return manifest_path


def _raw_existing_canonical_parent_path(ctx: RiverLinearContext) -> Path | None:
    bundle = ctx.linear_inputs
    if bundle is None:
        return None
    for attr in ("canonical_solve_final_dem_path", "canonical_solve_parent_dem_path"):
        value = getattr(bundle, attr, None)
        if value not in (None, ""):
            path = Path(value)
            if path.is_file():
                return path
    return None

def _existing_canonical_parent_for_aoi_export(ctx: RiverLinearContext) -> Path | None:
    manifest_path = _existing_canonical_manifest_for_aoi_export(ctx)
    if manifest_path is None:
        return None
    try:
        manifest = load_canonical_solution_manifest(manifest_path)
    except Exception:
        return None
    parent_value = manifest.get("canonical_parent_dem_path") or manifest.get("canonical_final_dem_path")
    if parent_value in (None, ""):
        return None
    path = Path(str(parent_value))
    return path if path.is_file() else None


def _load_export_canonical_manifest(ctx: RiverLinearContext) -> tuple[Path, dict[str, Any]]:
    ensure_stage_allowed(AOI_EXPORT_ONLY_ROLE, "read_canonical_manifest")
    manifest_path = _existing_canonical_manifest_for_aoi_export(ctx)
    if manifest_path is None:
        raise FileNotFoundError("missing_canonical_manifest_for_aoi_export_only")
    try:
        manifest = load_canonical_solution_manifest(manifest_path)
    except Exception as exc:
        raise ValueError(f"invalid_canonical_manifest_for_aoi_export_only:{manifest_path}") from exc
    parent_value = manifest.get("canonical_parent_dem_path") or manifest.get("canonical_final_dem_path")
    if parent_value in (None, ""):
        raise ValueError(f"canonical_manifest_missing_parent_dem_path:{manifest_path}")
    parent_dem = Path(str(parent_value))
    if not parent_dem.is_file():
        raise FileNotFoundError(f"canonical_parent_dem_missing_for_aoi_export:{parent_dem}")
    expected_hash = manifest.get("canonical_parent_dem_sha256") or manifest.get("canonical_final_dem_sha256")
    actual_hash = sha256_file(parent_dem)
    if expected_hash not in (None, "") and str(expected_hash) != str(actual_hash):
        raise ValueError(
            "canonical_manifest_parent_hash_mismatch:"
            f"manifest={manifest_path}:parent={parent_dem}:expected={expected_hash}:actual={actual_hash}"
        )
    return manifest_path, manifest


def _canonical_parent_handoff_from_manifest(
    ctx: RiverLinearContext,
    *,
    manifest_path: Path,
    cache_manifest_path: Path | None,
    source_kind: str,
    reused_from_cache: bool,
    adopted_from_existing_parent: bool,
) -> CanonicalParentHandoff:
    """Validate a canonical parent manifest and return the single handoff object."""
    ensure_stage_allowed(AOI_EXPORT_ONLY_ROLE, "read_canonical_manifest")
    try:
        manifest = load_canonical_solution_manifest(manifest_path)
    except Exception as exc:
        raise ValueError(f"invalid_canonical_manifest_for_aoi_export_only:{manifest_path}") from exc
    parent_value = manifest.get("canonical_parent_dem_path") or manifest.get("canonical_final_dem_path")
    if parent_value in (None, ""):
        raise ValueError(f"canonical_manifest_missing_parent_dem_path:{manifest_path}")
    parent_dem = Path(str(parent_value))
    if not parent_dem.is_file():
        raise FileNotFoundError(f"canonical_parent_dem_missing_for_aoi_export:{parent_dem}")
    expected_hash = manifest.get("canonical_parent_dem_sha256") or manifest.get("canonical_final_dem_sha256")
    actual_hash = sha256_file(parent_dem)
    if expected_hash not in (None, "") and str(expected_hash) != str(actual_hash):
        raise ValueError(
            "canonical_manifest_parent_hash_mismatch:"
            f"manifest={manifest_path}:parent={parent_dem}:expected={expected_hash}:actual={actual_hash}"
        )
    source_context_value = manifest.get("source_context_path")
    return CanonicalParentHandoff(
        manifest_path=Path(manifest_path),
        parent_dem_path=parent_dem,
        cache_manifest_path=(Path(cache_manifest_path) if cache_manifest_path is not None else None),
        source_kind=str(source_kind),
        reused_from_cache=bool(reused_from_cache),
        adopted_from_existing_parent=bool(adopted_from_existing_parent),
        canonical_system_id=(str(manifest.get("canonical_system_id")) if manifest.get("canonical_system_id") not in (None, "") else None),
        canonical_cache_key=(
            str(manifest.get("canonical_cache_key") or manifest.get("canonical_solve_cache_key"))
            if (manifest.get("canonical_cache_key") or manifest.get("canonical_solve_cache_key")) not in (None, "")
            else None
        ),
        canonical_parent_hash=str(actual_hash),
        source_context_path=(Path(str(source_context_value)) if source_context_value not in (None, "") else None),
        manifest_payload=dict(manifest),
    )


def _load_export_handoff_manifest(handoff: CanonicalParentHandoff) -> tuple[Path, dict[str, Any]]:
    """Return the manifest already validated by CanonicalParentHandoff."""
    if handoff.manifest_payload is not None:
        return handoff.manifest_path, dict(handoff.manifest_payload)
    return handoff.manifest_path, load_canonical_solution_manifest(handoff.manifest_path)


def _window_dict_from_parent_and_template(parent_path: Path, template_path: Path) -> dict[str, int] | None:
    """Return the AOI export template window in parent pixel coordinates."""
    try:
        import rasterio as _rio
        from rasterio.windows import from_bounds as _window_from_bounds
        with _rio.open(parent_path) as parent_ds, _rio.open(template_path) as template_ds:
            if str(parent_ds.crs) != str(template_ds.crs):
                return None
            window = _window_from_bounds(
                template_ds.bounds.left, template_ds.bounds.bottom,
                template_ds.bounds.right, template_ds.bounds.top,
                transform=parent_ds.transform,
            )
            return {
                "row_off": int(round(window.row_off)),
                "col_off": int(round(window.col_off)),
                "height": int(round(window.height)),
                "width": int(round(window.width)),
                "parent_width": int(parent_ds.width),
                "parent_height": int(parent_ds.height),
                "export_width": int(template_ds.width),
                "export_height": int(template_ds.height),
            }
    except Exception:
        return None


def _assert_export_only_result_has_no_construction_stage_receipts(result: RiverLinearPipelineResult) -> None:
    """Guard the architecture: export-only results may not retain construction-stage receipts."""
    forbidden = {
        "solve_domain", "grids", "authoritative_inputs", "centerline",
        "centerline_wse_proxy", "centerline_authoritative_bed",
        "centerline_observed_offset", "centerline_modeled_offset",
        "centerline_bed_backbone", "river_corridor_solve",
        "river_primary_surface_solve", "river_primary_surface_solve_locked",
        "canonical_parent_dem", "final_dem",
    }
    present = sorted(forbidden.intersection(set(result.stage_receipts or {})))
    if present:
        raise ValueError(f"aoi_export_only_retained_construction_stage_receipts:{present}")


def run_aoi_export_from_canonical_parent(ctx: RiverLinearContext, handoff: CanonicalParentHandoff | Path | None = None) -> RiverLinearPipelineResult:
    logger = logging.getLogger(getattr(ctx, "logger_name", "river_linear"))
    if isinstance(handoff, CanonicalParentHandoff):
        parent_handoff = handoff
    else:
        # Backward-compatible path for older tests/imports. Active routing should
        # pass CanonicalParentHandoff so cache/existing/adopted sources collapse
        # before AOI export.
        resolved = resolve_canonical_parent_handoff(ctx)
        if resolved is None:
            raise FileNotFoundError("missing_canonical_manifest_for_aoi_export_only")
        parent_handoff = resolved
    manifest_path, source_manifest = _load_export_handoff_manifest(parent_handoff)
    parent_dem = parent_handoff.parent_dem_path
    _build_run_contract(ctx, shared_solve_reused=True, shared_solve_cache_manifest_path=parent_handoff.cache_manifest_path)
    logger.info("[RIVER][AOI_EXPORT][STEP 1/5] Read canonical parent handoff: source=%s manifest=%s parent=%s", parent_handoff.source_kind, parent_handoff.manifest_path, parent_dem)
    import numpy as _np
    import rasterio as _rio
    with _rio.open(parent_dem) as ds:
        crs = str(ds.crs) if ds.crs is not None else str(ctx.projected_crs)
        solve_shape = (int(ds.height), int(ds.width))
        res_m = abs(float(ds.transform.a))
    logger.info("[RIVER][AOI_EXPORT][STEP 2/5] Build AOI export grid aligned to canonical parent.")
    export_shape = _write_subset_template_from_parent(ctx.paths.export_grid_template, parent_grid_path=parent_dem, bounds=_project_bounds(ctx.export_aoi, crs))
    logger.info("[RIVER][AOI_EXPORT][STEP 3/5] Write exact AOI DEM subset from canonical parent.")
    ctx.paths.canonical_parent_dem.parent.mkdir(parents=True, exist_ok=True)
    if parent_dem.resolve() != ctx.paths.canonical_parent_dem.resolve():
        shutil.copy2(parent_dem, ctx.paths.canonical_parent_dem)
        parent_for_receipts = ctx.paths.canonical_parent_dem
    else:
        parent_for_receipts = parent_dem
    subset_float_raster_to_template_exact(src_path=parent_for_receipts, template_path=ctx.paths.export_grid_template, dst_path=ctx.paths.aoi_export_dem)
    with _rio.open(ctx.paths.aoi_export_dem) as ds:
        arr = ds.read(1)
        valid = _np.isfinite(arr)
        if ds.nodata is not None:
            valid &= arr != _np.asarray(ds.nodata, dtype=arr.dtype)
        finite_count = int(_np.count_nonzero(valid))
    manifest_payload = dict(source_manifest)
    manifest_payload.update({
        "execution_role": AOI_EXPORT_ONLY_ROLE,
        "requested_solve_domain": str(ctx.requested_solve_domain) if ctx.requested_solve_domain is not None else manifest_payload.get("requested_solve_domain"),
        "resolved_solve_domain": str(ctx.resolved_solve_domain) if ctx.resolved_solve_domain is not None else manifest_payload.get("resolved_solve_domain"),
        "projected_crs": crs,
        "target_resolution_m": float(res_m),
        "run_contract_path": str(ctx.paths.run_contract),
        "canonical_parent_dem_path": str(parent_for_receipts),
        "canonical_parent_dem_sha256": sha256_file(parent_for_receipts),
        "canonical_parent_content_key": sha256_file(parent_for_receipts),
        "canonical_comparison_key": sha256_file(parent_for_receipts),
        "canonical_parent_dem": raster_identity(parent_for_receipts),
        "canonical_final_dem_path": str(parent_dem),
        "canonical_final_dem_sha256": sha256_file(parent_dem),
        "composition_summary": {**dict(manifest_payload.get("composition_summary") or {}), "final_finite_count": finite_count},
        "canonical_parent_policy": {
            "parent_built_once_on_solve_grid": True,
            "aoi_outputs_must_be_exact_parent_subsets": True,
            "post_subset_modification_allowed": False,
        },
        "aoi_export_policy": {
            "aoi_runs_may_recompute_canonical_construction": False,
            "aoi_runs_must_subset_from_parent": True,
            "cache_is_implementation_detail": True,
            "manifest_handoff_required": True,
        },
        "source_canonical_manifest_path": str(manifest_path),
        "canonical_parent_handoff": parent_handoff.source_summary(),
    })
    write_canonical_solution_manifest(ctx=ctx, payload=manifest_payload)
    logger.info("[RIVER][AOI_EXPORT][STEP 4/5] Retained canonical manifest: %s", public_canonical_solution_manifest_path(ctx))
    ctx.paths.aoi_export_receipt.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = assert_parent_window_matches_export(
        parent_path=parent_for_receipts,
        export_template_path=ctx.paths.export_grid_template,
        export_path=ctx.paths.aoi_export_dem,
    )
    parent_hash = sha256_file(parent_for_receipts)
    export_hash = sha256_file(ctx.paths.aoi_export_dem)
    parent_to_export_window = parent_identity.get("export_window") or _window_dict_from_parent_and_template(parent_for_receipts, ctx.paths.export_grid_template)
    receipt_payload = {
        "schema_version": 2,
        "stage": "aoi_export_identity",
        "stage_class": "aoi_export_only",
        "role": "aoi_export_identity",
        "execution_role": AOI_EXPORT_ONLY_ROLE,
        "canonical_cache_key": manifest_payload.get("canonical_cache_key"),
        "canonical_solve_cache_key": manifest_payload.get("canonical_solve_cache_key"),
        "canonical_parent_content_key": manifest_payload.get("canonical_parent_content_key") or parent_hash,
        "canonical_comparison_key": manifest_payload.get("canonical_comparison_key") or parent_hash,
        "canonical_system_id": manifest_payload["canonical_system_id"],
        "canonical_manifest_path": str(public_canonical_solution_manifest_path(ctx)),
        "canonical_parent_handoff": parent_handoff.source_summary(),
        "parent_dem": {"path": str(parent_for_receipts), "sha256": parent_hash},
        "aoi_export_dem": {"path": str(ctx.paths.aoi_export_dem), "sha256": export_hash},
        "parent_hash": parent_hash,
        "export_hash": export_hash,
        "parent_to_export_window": parent_to_export_window,
        "export_window": parent_to_export_window,
        "export_vs_parent": "PASS",
        "identity_contract": {
            "same_parent_window_required": True,
            "max_abs_diff_required_m": 0.0,
            "post_subset_modification_allowed": False,
            "aoi_construction_allowed": False,
        },
        "exact_subset_identity": True,
        "max_abs_diff_m": parent_identity.get("max_abs_diff_m"),
        "mismatch_count": parent_identity.get("mismatch_count"),
        "combined_vs_export": "not_evaluated_until_final_materialization",
        "post_subset_modifications": False,
        "pixel_values_modified_after_parent_subset": False,
        "resampled_after_parent_subset": False,
        "reprojected_after_parent_subset": False,
        "construction_attempted": False,
        "construction_stages_run_in_aoi_export": False,
    }
    ctx.paths.aoi_export_receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reports_receipt = ctx.paths.root.parent / "reports" / "aoi_export_identity.json"
    reports_receipt.parent.mkdir(parents=True, exist_ok=True)
    reports_receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_receipt_purpose_manifest(
        ctx.paths.receipt_purpose_manifest,
        ctx=ctx,
        stage_receipts={"aoi_export_only": ctx.paths.aoi_export_receipt},
        canonical_manifest_path=public_canonical_solution_manifest_path(ctx),
        aoi_export_identity_path=reports_receipt,
    )
    logger.info("[RIVER][AOI_EXPORT][STEP 5/5] AOI export-only route complete; no canonical construction stages were run. Retained export identity: %s", reports_receipt)
    grid_result = GridStageResult(parent_for_receipts, ctx.paths.export_grid_template, solve_shape, export_shape, float(res_m), crs, True)
    solve_result = SolveDomainStageResult(Path(getattr(ctx.linear_inputs, "canonical_solve_network_source_gpkg", parent_for_receipts)) if ctx.linear_inputs is not None else parent_for_receipts, ctx.paths.canonical_solve_aoi, str(getattr(ctx.linear_inputs, "canonical_solve_aoi", None) or ctx.resolved_solve_domain or ctx.export_aoi), int(getattr(ctx.linear_inputs, "canonical_selected_reach_count", 0) or 0) if ctx.linear_inputs is not None else 0, manifest_payload["canonical_system_id"], str(getattr(ctx.linear_inputs, "canonical_solve_stop_reason", "existing_canonical_parent") or "existing_canonical_parent") if ctx.linear_inputs is not None else "existing_canonical_parent", ctx.paths.solve_domain_receipt)
    authoritative_result = AuthoritativeStageResult(parent_for_receipts, parent_for_receipts, parent_for_receipts, parent_for_receipts, parent_for_receipts, parent_for_receipts, 0)
    centerline_result = CenterlineStageResult(parent_for_receipts, 0, 0)
    wse_result = WSEProxyStageResult(parent_for_receipts, 0, 0, {"export_only": True})
    authoritative_bed_result = AuthoritativeBedStageResult(parent_for_receipts, 0, 0)
    observed_offset_result = ObservedOffsetStageResult(parent_for_receipts, 0, 0, {"export_only": True})
    modeled_offset_result = ModeledOffsetStageResult(parent_for_receipts, 0, 0, None, None, None, None, False, "not_evaluated", {"export_only": True})
    backbone_result = BackboneStageResult(parent_for_receipts, 0, 0, {"export_only": True})
    corridor_result = CorridorStageResult(parent_for_receipts, 0)
    surface_result = SurfaceStageResult(parent_for_receipts, 0, finite_count, 0, None, 0, 0)
    lock_result = LockStageResult(parent_for_receipts, 0, finite_count, None, 0, 0)
    export_result = ExportStageResult(ctx.paths.export_grid_template, parent_for_receipts, parent_for_receipts, parent_for_receipts, ctx.paths.aoi_export_dem, parent_for_receipts, parent_for_receipts, parent_for_receipts, finite_count, 0, 0, 0)
    final_dem_result = FinalDemStageResult(ctx.paths.aoi_export_dem, parent_for_receipts, ctx.paths.aoi_export_dem, ctx.paths.canonical_parent_receipt, ctx.paths.aoi_export_receipt, 0, 0, finite_count, finite_count, "existing_canonical_parent_exact_aoi_export", parent_dem, None)
    result = RiverLinearPipelineResult(ctx.paths.run_contract, True, solve_result, grid_result, authoritative_result, centerline_result, wse_result, authoritative_bed_result, observed_offset_result, modeled_offset_result, backbone_result, corridor_result, surface_result, lock_result, export_result, final_dem_result, {"aoi_export_only": ctx.paths.aoi_export_receipt}, None, None, None, None, None, None, None, None, None)
    _assert_export_only_result_has_no_construction_stage_receipts(result)
    return result

def run_canonical_river_build(ctx: RiverLinearContext) -> RiverLinearPipelineResult:
    """Build the canonical river parent and then export the requested AOI.

    This function is allowed to run construction stages. AOI export-only routing
    is handled by run_aoi_export_from_canonical_parent().
    """
    logger = logging.getLogger(getattr(ctx, "logger_name", "river_linear"))
    def _log_step(n: int, total: int, title: str, detail: str) -> None:
        if logger is not None:
            logger.info("[RIVER][PIPELINE][STEP %d/%d] %s %s", n, total, title, detail)
    cached_solve = load_canonical_solve_cache(ctx.linear_inputs) if ctx.linear_inputs is not None else None
    shared_solve_reused = cached_solve is not None
    shared_solve_cache_manifest_path = None
    if cached_solve is not None:
        shared_solve_cache_manifest_path = cached_solve.get('manifest_path')
    elif ctx.linear_inputs is not None:
        shared_solve_cache_manifest_path = ctx.linear_inputs.canonical_solve_cache_manifest_path

    _build_run_contract(
        ctx,
        shared_solve_reused=shared_solve_reused,
        shared_solve_cache_manifest_path=(Path(shared_solve_cache_manifest_path) if shared_solve_cache_manifest_path is not None else None),
    )
    stage_trace: list[dict[str, Any]] = []
    if ctx.write_diagnostics:
        stage_trace = [
            build_stage_trace_entry(
                stage_name='run_contract',
                receipt_path=ctx.paths.run_contract,
                inputs={},
                outputs={'run_contract': ctx.paths.run_contract},
                summary={
                    'workflow_name': ctx.workflow_name,
                    'export_aoi': ctx.export_aoi,
                    'requested_solve_domain': ctx.requested_solve_domain,
                    'resolved_solve_domain': ctx.resolved_solve_domain,
                    'solve_domain_source': ctx.solve_domain_source,
                    'shared_solve_reused': shared_solve_reused,
                    'execution_role': ctx.execution_role,
                    'projected_crs': ctx.projected_crs,
                    'target_resolution_m': ctx.target_resolution_m,
                },
                validator=build_stage_validator_summary('run_contract', status='passed'),
            )
        ]

    ensure_stage_allowed(ctx.execution_role, "solve_domain")
    _log_step(1, 14, "Resolve solve domain.", "Build the canonical/shared river solve region and canonical network artifacts.")
    solve_result = run_solve_domain_stage(ctx)
    logger.info("[RIVER][CANONICAL] system_id=%s", solve_result.canonical_system_id)
    logger.info("[RIVER][CANONICAL] domain=%s", solve_result.canonical_solve_aoi)
    logger.info("[RIVER][CANONICAL_BUILD] Requested AOI will be exported from the canonical parent after construction.")
    solve_receipt = _record_stage(
        ctx=ctx,
        stage_trace=stage_trace,
        receipt_path=ctx.paths.solve_domain_receipt,
        stage_name='solve_domain',
        inputs={
            'export_aoi': ctx.export_aoi,
            'solve_source_aoi': ctx.linear_inputs.solve_source_aoi if ctx.linear_inputs is not None else None,
            'network_gpkg': ctx.linear_inputs.solve_network_gpkg if ctx.linear_inputs is not None else ctx.network_gpkg,
            'canonical_solve_network_source_gpkg': ctx.linear_inputs.canonical_solve_network_source_gpkg if ctx.linear_inputs is not None else None,
            'canonical_solve_receipt_source_json': ctx.linear_inputs.canonical_solve_receipt_source_json if ctx.linear_inputs is not None else None,
        },
        outputs={
            'canonical_solve_network': solve_result.canonical_network_path,
            'canonical_solve_aoi': solve_result.canonical_solve_aoi_path,
            'canonical_system_identity_receipt': getattr(ctx, 'canonical_identity_receipt_path', None),
        },
        summary={
            'canonical_system_id': solve_result.canonical_system_id,
            'stop_reason': solve_result.stop_reason,
            'selected_reach_count': solve_result.selected_reach_count,
            'canonical_solve_aoi': solve_result.canonical_solve_aoi,
            'canonical_identity_receipt_path': str(getattr(ctx, 'canonical_identity_receipt_path', '')),
        },
    )

    ensure_stage_allowed(ctx.execution_role, "grids")
    _log_step(2, 14, "Build solve/export grids.", "Create the shared solve grid and the AOI export grid used for final subsetting.")
    grid_result = run_grid_stage(ctx, solve_result)
    grids_receipt = _record_stage(
        ctx=ctx,
        stage_trace=stage_trace,
        receipt_path=ctx.paths.grids_receipt,
        stage_name='grids',
        inputs={
            'canonical_solve_aoi': solve_result.canonical_solve_aoi,
            'export_aoi': ctx.export_aoi,
        },
        outputs={
            'solve_grid_template': grid_result.solve_grid_template_path,
            'export_grid_template': grid_result.export_grid_template_path,
        },
        summary={
            'solve_shape': list(grid_result.solve_shape),
            'export_shape': list(grid_result.export_shape),
            'resolution_m': grid_result.resolution_m,
            'crs': grid_result.crs,
        },
    )

    ensure_stage_allowed(ctx.execution_role, "authoritative_inputs")
    _log_step(3, 14, "Materialize authoritative inputs.", "Prepare authoritative measured/support rasters on the shared solve grid and AOI export grid.")
    authoritative_result = run_authoritative_stage(ctx, solve_result, grid_result)
    authoritative_receipt = _record_stage(
        ctx=ctx,
        stage_trace=stage_trace,
        receipt_path=ctx.paths.authoritative_receipt,
        stage_name='authoritative_inputs',
        inputs={
            'solve_authoritative_source': authoritative_result.solve_authoritative_source_path,
            'export_authoritative_source': authoritative_result.export_authoritative_source_path,
            'export_baseline_source': authoritative_result.export_baseline_source_path,
            'source_bundle': ctx.linear_inputs.to_dict() if ctx.linear_inputs is not None else None,
            'canonical_source_context_path': (ctx.linear_inputs.canonical_source_context_path if ctx.linear_inputs is not None else None),
            'solve_grid_template': grid_result.solve_grid_template_path,
            'export_grid_template': grid_result.export_grid_template_path,
        },
        outputs={
            'solve_authoritative_base_measured_only': authoritative_result.solve_authoritative_base_measured_only_path,
            'solve_authoritative_support_mask': authoritative_result.solve_authoritative_support_mask_path,
            'solve_baseline_background': authoritative_result.solve_baseline_background_path,
        },
        summary={
            'solve_support_pixel_count': authoritative_result.solve_support_pixel_count,
            'solve_authoritative_source_role': (ctx.linear_inputs.solve_authoritative_source_role if ctx.linear_inputs is not None else None),
            'export_authoritative_source_role': (ctx.linear_inputs.export_authoritative_source_role if ctx.linear_inputs is not None else None),
            'authoritative_routing_policy': (ctx.linear_inputs.authoritative_routing_policy if ctx.linear_inputs is not None else None),
            'canonical_solve_identity_path': (ctx.linear_inputs.canonical_solve_identity_path if ctx.linear_inputs is not None else None),
            'canonical_source_context_path': (ctx.linear_inputs.canonical_source_context_path if ctx.linear_inputs is not None else None),
        },
    )

    if cached_solve is not None:
        ensure_stage_allowed(ctx.execution_role, "centerline_points")
        ensure_stage_allowed(ctx.execution_role, "centerline_wse_proxy")
        ensure_stage_allowed(ctx.execution_role, "centerline_authoritative_bed")
        ensure_stage_allowed(ctx.execution_role, "centerline_observed_offset")
        ensure_stage_allowed(ctx.execution_role, "centerline_modeled_offset")
        ensure_stage_allowed(ctx.execution_role, "centerline_bed_backbone")
        ensure_stage_allowed(ctx.execution_role, "river_corridor_solve")
        ensure_stage_allowed(ctx.execution_role, "river_primary_surface_solve")
        ensure_stage_allowed(ctx.execution_role, "river_primary_surface_solve_locked")
        centerline_result = cached_solve['centerline_result']
        wse_result = cached_solve['wse_result']
        authoritative_bed_result = cached_solve['authoritative_bed_result']
        observed_offset_result = cached_solve['observed_offset_result']
        modeled_offset_result = cached_solve['modeled_offset_result']
        backbone_result = cached_solve['backbone_result']
        cached_backbone_export_subset_check = validate_backbone_export_subset_flatness(
            ctx,
            backbone_result.centerline_bed_backbone_points_path,
        )
        corridor_result = cached_solve['corridor_result']
        surface_result = cached_solve['surface_result']
        lock_result = cached_solve['lock_result']
        cached_lock_contract_path = write_authoritative_lock_contract_for_paths(
            receipt_path=ctx.paths.receipts_dir / 'authoritative_lock_contract.json',
            locked_raster_path=lock_result.river_primary_surface_solve_locked_path,
            unlocked_raster_path=surface_result.river_primary_surface_solve_path,
            measured_path=authoritative_result.solve_authoritative_base_measured_only_path,
            support_mask_path=authoritative_result.solve_authoritative_support_mask_path,
        )
        centerline_receipt = _record_stage(
            ctx=ctx, stage_trace=stage_trace, receipt_path=ctx.paths.centerline_receipt, stage_name='centerline_points',
            inputs={'canonical_solve_cache_manifest': cached_solve['manifest_path'], 'canonical_solve_network': solve_result.canonical_network_path},
            outputs={'centerline_points': centerline_result.centerline_points_path},
            summary={'record_count': centerline_result.record_count, 'component_count': centerline_result.component_count, 'cache_reused': True},
        )
        wse_receipt = _record_stage(
            ctx=ctx, stage_trace=stage_trace, receipt_path=ctx.paths.wse_proxy_receipt, stage_name='centerline_wse_proxy',
            inputs={'canonical_solve_cache_manifest': cached_solve['manifest_path'], 'centerline_points': centerline_result.centerline_points_path},
            outputs={'centerline_wse_proxy_points': wse_result.centerline_wse_proxy_points_path},
            summary={'record_count': wse_result.record_count, 'finite_wse_count': wse_result.finite_wse_count, 'cache_reused': True, 'river_science': getattr(wse_result, 'science_summary', None)},
        )
        authoritative_bed_receipt = _record_stage(
            ctx=ctx, stage_trace=stage_trace, receipt_path=ctx.paths.authoritative_bed_receipt, stage_name='centerline_authoritative_bed',
            inputs={'canonical_solve_cache_manifest': cached_solve['manifest_path'], 'centerline_points': centerline_result.centerline_points_path},
            outputs={'centerline_authoritative_bed_points': authoritative_bed_result.centerline_authoritative_bed_points_path},
            summary={'record_count': authoritative_bed_result.record_count, 'finite_bed_count': authoritative_bed_result.finite_bed_count, 'cache_reused': True},
        )
        observed_offset_receipt = _record_stage(
            ctx=ctx, stage_trace=stage_trace, receipt_path=ctx.paths.observed_offset_receipt, stage_name='centerline_observed_offset',
            inputs={'canonical_solve_cache_manifest': cached_solve['manifest_path']},
            outputs={'centerline_observed_offset_points': observed_offset_result.centerline_observed_offset_points_path},
            summary={'record_count': observed_offset_result.record_count, 'finite_offset_count': observed_offset_result.finite_offset_count, 'cache_reused': True, 'river_science': getattr(observed_offset_result, 'science_summary', None)},
        )
        modeled_offset_receipt = _record_stage(
            ctx=ctx, stage_trace=stage_trace, receipt_path=ctx.paths.modeled_offset_receipt, stage_name='centerline_modeled_offset',
            inputs={'canonical_solve_cache_manifest': cached_solve['manifest_path']},
            outputs={'centerline_modeled_offset_points': modeled_offset_result.centerline_modeled_offset_points_path},
            summary={'record_count': modeled_offset_result.record_count, 'finite_modeled_offset_count': modeled_offset_result.finite_modeled_offset_count, 'cache_reused': True, 'global_observed_median_offset_m': modeled_offset_result.global_observed_median_offset_m, 'global_prior_offset_m': modeled_offset_result.global_prior_offset_m, 'export_floor_fraction_before_policy': modeled_offset_result.export_floor_fraction_before_policy, 'export_floor_fraction_after_policy': modeled_offset_result.export_floor_fraction_after_policy, 'export_floor_policy_applied': modeled_offset_result.export_floor_policy_applied, 'export_floor_policy_status': modeled_offset_result.export_floor_policy_status, 'river_science': getattr(modeled_offset_result, 'science_summary', None)},
        )
        backbone_receipt = _record_stage(
            ctx=ctx, stage_trace=stage_trace, receipt_path=ctx.paths.backbone_receipt, stage_name='centerline_bed_backbone',
            inputs={'canonical_solve_cache_manifest': cached_solve['manifest_path']},
            outputs={'centerline_bed_backbone_points': backbone_result.centerline_bed_backbone_points_path},
            summary={
                'record_count': backbone_result.record_count,
                'finite_backbone_count': backbone_result.finite_backbone_count,
                'cache_reused': True,
                'river_science': getattr(backbone_result, 'science_summary', None),
                'export_subset_flatness_guard': cached_backbone_export_subset_check,
            },
        )
        corridor_receipt = _record_stage(
            ctx=ctx, stage_trace=stage_trace, receipt_path=ctx.paths.corridor_receipt, stage_name='river_corridor_solve',
            inputs={'canonical_solve_cache_manifest': cached_solve['manifest_path'], 'solve_grid_template': grid_result.solve_grid_template_path},
            outputs={'river_corridor_solve': corridor_result.river_corridor_solve_path},
            summary={'corridor_pixel_count': corridor_result.corridor_pixel_count, 'corridor_source': getattr(corridor_result, 'corridor_source', None), 'cache_reused': True},
        )
        surface_receipt = _record_stage(
            ctx=ctx, stage_trace=stage_trace, receipt_path=ctx.paths.surface_receipt, stage_name='river_primary_surface_solve',
            inputs={'canonical_solve_cache_manifest': cached_solve['manifest_path'], 'solve_grid_template': grid_result.solve_grid_template_path},
            outputs={'river_primary_surface_solve': surface_result.river_primary_surface_solve_path, 'surface_contract': getattr(surface_result, 'surface_contract_path', None)},
            summary={'seeded_pixel_count': surface_result.seeded_pixel_count, 'finite_surface_count': surface_result.finite_surface_count, 'surface_contract': getattr(surface_result, 'surface_contract_path', None), 'cache_reused': True},
        )
        lock_receipt = _record_stage(
            ctx=ctx, stage_trace=stage_trace, receipt_path=ctx.paths.lock_receipt, stage_name='river_primary_surface_solve_locked',
            inputs={'canonical_solve_cache_manifest': cached_solve['manifest_path'], 'solve_authoritative_support_mask': authoritative_result.solve_authoritative_support_mask_path},
            outputs={'river_primary_surface_solve_locked': lock_result.river_primary_surface_solve_locked_path},
            summary={'locked_pixel_count': lock_result.locked_pixel_count, 'finite_locked_surface_count': lock_result.finite_locked_surface_count, 'cache_reused': True, 'authoritative_lock_contract': cached_lock_contract_path, 'measured_cells_changed_count': 0},
        )
    else:
        ensure_stage_allowed(ctx.execution_role, "centerline_points")
        _log_step(4, 14, "Build centerline.", "Generate the canonical river centerline points for the shared solve domain.")
        centerline_result = run_centerline_stage(ctx, solve_result, authoritative_result, grid_result)
        centerline_receipt = _record_stage(
            ctx=ctx,
            stage_trace=stage_trace,
            receipt_path=ctx.paths.centerline_receipt,
            stage_name='centerline_points',
            inputs={
                'canonical_solve_network': solve_result.canonical_network_path,
                'solve_authoritative_base_measured_only': authoritative_result.solve_authoritative_base_measured_only_path,
            },
            outputs={
                'centerline_points': centerline_result.centerline_points_path,
            },
            summary={
                'record_count': centerline_result.record_count,
                'component_count': centerline_result.component_count,
            },
        )

        ensure_stage_allowed(ctx.execution_role, "centerline_wse_proxy")
        _log_step(5, 14, "Estimate WSE proxy.", "Sample a water-surface proxy along the centerline to guide bed inference.")
        wse_result = run_wse_proxy_stage(ctx, centerline_result, authoritative_result)
        wse_receipt = _record_stage(
            ctx=ctx,
            stage_trace=stage_trace,
            receipt_path=ctx.paths.wse_proxy_receipt,
            stage_name='centerline_wse_proxy',
            inputs={
                'centerline_points': centerline_result.centerline_points_path,
                'solve_authoritative_base_measured_only': authoritative_result.solve_authoritative_base_measured_only_path,
            },
            outputs={
                'centerline_wse_support_points': getattr(wse_result, 'centerline_wse_support_points_path', None),
                'centerline_wse_trend_points': getattr(wse_result, 'centerline_wse_trend_points_path', None),
                'centerline_wse_pre_smooth_points': getattr(wse_result, 'centerline_wse_pre_smooth_points_path', None),
                'centerline_wse_proxy_points': wse_result.centerline_wse_proxy_points_path,
                'wse_current_path_audit': getattr(wse_result, 'wse_current_path_audit_path', None),
                'wse_direction_audit': getattr(wse_result, 'wse_direction_audit_path', None),
                'wse_proxy_audit': getattr(wse_result, 'wse_proxy_audit_path', None),
                'wse_artifact_manifest': getattr(wse_result, 'wse_artifact_manifest_path', None),
                'wse_one_path_stage_contract': getattr(wse_result, 'wse_stage_contract_path', None),
            },
            summary={
                'record_count': wse_result.record_count,
                'finite_wse_count': wse_result.finite_wse_count,
                'river_science': getattr(wse_result, 'science_summary', None),
            },
        )

        ensure_stage_allowed(ctx.execution_role, "centerline_authoritative_bed")
        _log_step(6, 14, "Sample authoritative bed.", "Collect centerline-aligned measured bed support where authoritative coverage exists.")
        authoritative_bed_result = run_authoritative_bed_stage(ctx, centerline_result, authoritative_result)
        authoritative_bed_receipt = _record_stage(
            ctx=ctx,
            stage_trace=stage_trace,
            receipt_path=ctx.paths.authoritative_bed_receipt,
            stage_name='centerline_authoritative_bed',
            inputs={
                'centerline_points': centerline_result.centerline_points_path,
                'solve_authoritative_base_measured_only': authoritative_result.solve_authoritative_base_measured_only_path,
            },
            outputs={
                'centerline_authoritative_bed_points': authoritative_bed_result.centerline_authoritative_bed_points_path,
            },
            summary={
                'record_count': authoritative_bed_result.record_count,
                'finite_bed_count': authoritative_bed_result.finite_bed_count,
            },
        )

        ensure_stage_allowed(ctx.execution_role, "centerline_observed_offset")
        _log_step(7, 14, "Compute observed offsets.", "Convert WSE-minus-bed observations into offset samples along the river.")
        observed_offset_result = run_observed_offset_stage(ctx, wse_result, authoritative_bed_result)
        observed_offset_receipt = _record_stage(
            ctx=ctx,
            stage_trace=stage_trace,
            receipt_path=ctx.paths.observed_offset_receipt,
            stage_name='centerline_observed_offset',
            inputs={
                'centerline_wse_proxy_points': wse_result.centerline_wse_proxy_points_path,
                'centerline_authoritative_bed_points': authoritative_bed_result.centerline_authoritative_bed_points_path,
            },
            outputs={
                'centerline_observed_offset_points': observed_offset_result.centerline_observed_offset_points_path,
            },
            summary={
                'record_count': observed_offset_result.record_count,
                'finite_offset_count': observed_offset_result.finite_offset_count,
                'river_science': getattr(observed_offset_result, 'science_summary', None),
            },
        )

        ensure_stage_allowed(ctx.execution_role, "centerline_modeled_offset")
        _log_step(8, 14, "Model offsets.", "Interpolate or model offsets where direct observed support is sparse or missing.")
        modeled_offset_result = run_modeled_offset_stage(ctx, wse_result, observed_offset_result)
        modeled_offset_receipt = _record_stage(
            ctx=ctx,
            stage_trace=stage_trace,
            receipt_path=ctx.paths.modeled_offset_receipt,
            stage_name='centerline_modeled_offset',
            inputs={
                'centerline_wse_proxy_points': wse_result.centerline_wse_proxy_points_path,
                'centerline_observed_offset_points': observed_offset_result.centerline_observed_offset_points_path,
            },
            outputs={
                'centerline_modeled_offset_points': modeled_offset_result.centerline_modeled_offset_points_path,
            },
            summary={
                'record_count': modeled_offset_result.record_count,
                'finite_modeled_offset_count': modeled_offset_result.finite_modeled_offset_count,
                'global_observed_median_offset_m': modeled_offset_result.global_observed_median_offset_m,
                'global_prior_offset_m': modeled_offset_result.global_prior_offset_m,
                'export_floor_fraction_before_policy': modeled_offset_result.export_floor_fraction_before_policy,
                'export_floor_fraction_after_policy': modeled_offset_result.export_floor_fraction_after_policy,
                'export_floor_policy_applied': modeled_offset_result.export_floor_policy_applied,
                'export_floor_policy_status': modeled_offset_result.export_floor_policy_status,
                'river_science': getattr(modeled_offset_result, 'science_summary', None),
            },
        )

        ensure_stage_allowed(ctx.execution_role, "centerline_bed_backbone")
        _log_step(9, 14, "Build bed backbone.", "Combine WSE proxy and modeled offsets into a centerline bed backbone.")
        backbone_result = run_backbone_stage(ctx, wse_result, modeled_offset_result)
        backbone_receipt = _record_stage(
            ctx=ctx,
            stage_trace=stage_trace,
            receipt_path=ctx.paths.backbone_receipt,
            stage_name='centerline_bed_backbone',
            inputs={
                'centerline_modeled_offset_points': modeled_offset_result.centerline_modeled_offset_points_path,
            },
            outputs={
                'centerline_bed_backbone_points': backbone_result.centerline_bed_backbone_points_path,
            },
            summary={
                'record_count': backbone_result.record_count,
                'finite_backbone_count': backbone_result.finite_backbone_count,
                'river_science': getattr(backbone_result, 'science_summary', None),
            },
        )

        ensure_stage_allowed(ctx.execution_role, "river_corridor_solve")
        _log_step(10, 14, "Build river corridor.", "Spread centerline guidance laterally across the river corridor on the shared solve grid.")
        corridor_result = run_corridor_stage(ctx, solve_result, grid_result, centerline_result)
        corridor_receipt = _record_stage(
            ctx=ctx,
            stage_trace=stage_trace,
            receipt_path=ctx.paths.corridor_receipt,
            stage_name='river_corridor_solve',
            inputs={
                'canonical_solve_network': solve_result.canonical_network_path,
                'centerline_points': centerline_result.centerline_points_path,
                'solve_grid_template': grid_result.solve_grid_template_path,
            },
            outputs={
                'river_corridor_solve': corridor_result.river_corridor_solve_path,
            },
            summary={
                'corridor_pixel_count': corridor_result.corridor_pixel_count,
                'corridor_source': getattr(corridor_result, 'corridor_source', None),
            },
        )

        ensure_stage_allowed(ctx.execution_role, "river_primary_surface_solve")
        _log_step(11, 14, "Create primary surface.", "Generate the shared river primary surface from corridor guidance and bed backbone structure.")
        surface_result = run_surface_stage(ctx, grid_result, corridor_result, backbone_result)
        surface_receipt = _record_stage(
            ctx=ctx,
            stage_trace=stage_trace,
            receipt_path=ctx.paths.surface_receipt,
            stage_name='river_primary_surface_solve',
            inputs={
                'solve_grid_template': grid_result.solve_grid_template_path,
                'river_corridor_solve': corridor_result.river_corridor_solve_path,
                'centerline_bed_backbone_points': backbone_result.centerline_bed_backbone_points_path,
            },
            outputs={
                'river_primary_surface_solve': surface_result.river_primary_surface_solve_path,
                'surface_contract': getattr(surface_result, 'surface_contract_path', None),
            },
            summary={
                'seeded_pixel_count': surface_result.seeded_pixel_count,
                'finite_surface_count': surface_result.finite_surface_count,
                'corridor_pixel_count': getattr(surface_result, 'corridor_pixel_count', 0),
                'channel_shape_pixel_count': getattr(surface_result, 'channel_shape_pixel_count', 0),
                'dominant_value_fraction': getattr(surface_result, 'dominant_value_fraction', None),
                'abrupt_step_count': getattr(surface_result, 'abrupt_step_count', 0),
                'surface_contract': getattr(surface_result, 'surface_contract_path', None),
            },
        )

        ensure_stage_allowed(ctx.execution_role, "river_primary_surface_solve_locked")
        _log_step(12, 14, "Apply authoritative lock.", "Reimpose authoritative measured cells so guidance cannot overwrite hard control.")
        lock_result = run_lock_stage(ctx, grid_result, authoritative_result, surface_result)
        lock_receipt = _record_stage(
            ctx=ctx,
            stage_trace=stage_trace,
            receipt_path=ctx.paths.lock_receipt,
            stage_name='river_primary_surface_solve_locked',
            inputs={
                'river_primary_surface_solve': surface_result.river_primary_surface_solve_path,
                'solve_authoritative_base_measured_only': authoritative_result.solve_authoritative_base_measured_only_path,
                'solve_authoritative_support_mask': authoritative_result.solve_authoritative_support_mask_path,
            },
            outputs={
                'river_primary_surface_solve_locked': lock_result.river_primary_surface_solve_locked_path,
            },
            summary={
                'locked_pixel_count': lock_result.locked_pixel_count,
                'finite_locked_surface_count': lock_result.finite_locked_surface_count,
                'authoritative_lock_contract': lock_result.authoritative_lock_contract_path,
                'measured_cells_changed_count': getattr(lock_result, 'measured_cells_changed_count', 0),
                'changed_outside_authoritative_lock_count': getattr(lock_result, 'changed_outside_authoritative_lock_count', 0),
            },
        )
        write_canonical_solve_cache(
            bundle=ctx.linear_inputs,
            centerline_result=centerline_result,
            wse_result=wse_result,
            authoritative_bed_result=authoritative_bed_result,
            observed_offset_result=observed_offset_result,
            modeled_offset_result=modeled_offset_result,
            backbone_result=backbone_result,
            corridor_result=corridor_result,
            surface_result=surface_result,
            lock_result=lock_result,
        )

    ensure_stage_allowed(ctx.execution_role, "river_export_handoff")
    _log_step(13, 14, "Export AOI subset.", "Subset shared-solve products down to the requested AOI export region.")
    export_result = run_export_stage(ctx, grid_result, authoritative_result, corridor_result, lock_result)
    export_receipt = _record_stage(
        ctx=ctx,
        stage_trace=stage_trace,
        receipt_path=ctx.paths.export_receipt,
        stage_name='river_export_handoff',
        inputs={
            'river_corridor_solve': corridor_result.river_corridor_solve_path,
            'river_primary_surface_solve': surface_result.river_primary_surface_solve_path,
            'river_primary_surface_solve_locked': lock_result.river_primary_surface_solve_locked_path,
            'export_grid_template': grid_result.export_grid_template_path,
            'export_authoritative_source': authoritative_result.export_authoritative_source_path,
            'export_baseline_source': authoritative_result.export_baseline_source_path,
        },
        outputs={
            'river_guidance_export': export_result.river_guidance_export_path,
            'river_corridor_mask_export': export_result.river_corridor_mask_export_path,
            'river_support_mask_export': export_result.river_support_mask_export_path,
            'river_take_mask_export': export_result.river_take_mask_export_path,
            'export_authoritative_base_measured_only': export_result.export_authoritative_base_measured_only_path,
            'export_authoritative_support_mask': export_result.export_authoritative_support_mask_path,
            'export_baseline_background': export_result.export_baseline_background_path,
            'export_grid_template': export_result.export_grid_template_path,
        },
        summary={
            'finite_guidance_export_count': export_result.finite_guidance_export_count,
            'corridor_export_pixel_count': export_result.corridor_export_pixel_count,
            'support_export_pixel_count': export_result.support_export_pixel_count,
            'take_export_pixel_count': export_result.take_export_pixel_count,
        },
    )

    ensure_stage_allowed(ctx.execution_role, "canonical_parent_dem_build")
    _log_step(14, 14, "Build canonical parent DEM and export AOI DEM.", "Build the scientific canonical parent DEM on the solve grid, then export the user AOI as an exact subset.")
    final_dem_result = run_final_dem_stage(ctx, grid_result, authoritative_result, corridor_result, lock_result, export_result)
    final_dem_receipt = _record_stage(
        ctx=ctx,
        stage_trace=stage_trace,
        receipt_path=ctx.paths.final_dem_receipt,
        stage_name='final_dem',
        inputs={
            'export_authoritative_base_measured_only': export_result.export_authoritative_base_measured_only_path,
            'export_authoritative_support_mask': export_result.export_authoritative_support_mask_path,
            'export_baseline_background': export_result.export_baseline_background_path,
            'river_guidance_export': export_result.river_guidance_export_path,
            'river_take_mask_export': export_result.river_take_mask_export_path,
            'export_grid_template': export_result.export_grid_template_path,
        },
        outputs={
            'canonical_parent_dem': final_dem_result.canonical_parent_dem_path,
            'aoi_export_dem': final_dem_result.aoi_export_dem_path,
        },
        summary={
            'final_finite_count': final_dem_result.final_finite_count,
            'support_applied_pixel_count': final_dem_result.support_applied_pixel_count,
            'guidance_applied_pixel_count': final_dem_result.guidance_applied_pixel_count,
            'background_applied_pixel_count': final_dem_result.background_applied_pixel_count,
        },
    )

    receipts = {
        stage_name: receipt for stage_name, receipt in {
            'solve_domain': solve_receipt,
            'grids': grids_receipt,
            'authoritative_inputs': authoritative_receipt,
            'centerline_points': centerline_receipt,
            'centerline_wse_proxy': wse_receipt,
            'centerline_authoritative_bed': authoritative_bed_receipt,
            'centerline_observed_offset': observed_offset_receipt,
            'centerline_modeled_offset': modeled_offset_receipt,
            'centerline_bed_backbone': backbone_receipt,
            'river_corridor_solve': corridor_receipt,
            'river_primary_surface_solve': surface_receipt,
            'river_primary_surface_solve_locked': lock_receipt,
            'river_export_handoff': export_receipt,
            'final_dem': final_dem_receipt,
        }.items() if receipt is not None
    }

    trace_summary = None
    manifest = None
    canonical_solve_identity_manifest = None
    canonical_stage_identity_manifest = None
    canonical_cache_consistency_receipt = None
    adjacent_aoi_verification_receipt = None
    river_verification_summary = None
    final_dem_writer_receipt = None
    river_science_chain_summary = None
    pipeline_outputs = {
        'canonical_solve_network': solve_result.canonical_network_path,
        'solve_grid_template': grid_result.solve_grid_template_path,
        'export_grid_template': grid_result.export_grid_template_path,
        'solve_authoritative_base_measured_only': authoritative_result.solve_authoritative_base_measured_only_path,
        'export_authoritative_base_measured_only': export_result.export_authoritative_base_measured_only_path,
        'centerline_points': centerline_result.centerline_points_path,
        'centerline_wse_proxy_points': wse_result.centerline_wse_proxy_points_path,
        'centerline_authoritative_bed_points': authoritative_bed_result.centerline_authoritative_bed_points_path,
        'centerline_observed_offset_points': observed_offset_result.centerline_observed_offset_points_path,
        'centerline_modeled_offset_points': modeled_offset_result.centerline_modeled_offset_points_path,
        'centerline_bed_backbone_points': backbone_result.centerline_bed_backbone_points_path,
        'river_corridor_solve': corridor_result.river_corridor_solve_path,
        'river_primary_surface_solve': surface_result.river_primary_surface_solve_path,
        'river_primary_surface_solve_locked': lock_result.river_primary_surface_solve_locked_path,
        'river_guidance_export': export_result.river_guidance_export_path,
        'river_corridor_mask_export': export_result.river_corridor_mask_export_path,
        'river_support_mask_export': export_result.river_support_mask_export_path,
        'river_take_mask_export': export_result.river_take_mask_export_path,
        'canonical_parent_dem': final_dem_result.canonical_parent_dem_path,
        'aoi_export_dem': final_dem_result.aoi_export_dem_path,
        'dem_enhanced_final': final_dem_result.dem_enhanced_final_path,
        'canonical_parent_dem_receipt': final_dem_result.canonical_parent_receipt_path,
        'aoi_export_dem_receipt': final_dem_result.aoi_export_receipt_path,
        'canonical_system_identity_receipt': getattr(ctx, 'canonical_identity_receipt_path', None),
    }
    river_science_chain_summary = write_river_science_chain_summary(
        ctx=ctx,
        centerline_result=centerline_result,
        wse_result=wse_result,
        authoritative_bed_result=authoritative_bed_result,
        observed_offset_result=observed_offset_result,
        modeled_offset_result=modeled_offset_result,
        backbone_result=backbone_result,
        corridor_result=corridor_result,
        surface_result=surface_result,
        lock_result=lock_result,
        export_result=export_result,
        final_dem_result=final_dem_result,
    )
    pipeline_outputs['river_science_chain_summary'] = river_science_chain_summary
    if ctx.write_diagnostics:
        trace_summary = write_trace_summary(
            ctx.paths.trace_summary,
            run_contract_path=ctx.paths.run_contract,
            stage_trace=stage_trace,
        )
        manifest = write_bundle_manifest(
            ctx.paths.bundle_manifest,
            stage_receipts=receipts,
            run_contract_path=ctx.paths.run_contract,
            trace_summary_path=trace_summary,
            stage_trace=stage_trace,
            outputs=pipeline_outputs,
        )
    if ctx.write_diagnostics or getattr(ctx, "write_core_receipts", True):
        canonical_solve_identity_manifest = write_canonical_solve_identity_manifest(
            ctx.paths.canonical_solve_identity_manifest,
            run_contract_path=ctx.paths.run_contract,
            source_identity_path=(ctx.linear_inputs.canonical_solve_identity_path if ctx.linear_inputs is not None else None),
        )
        canonical_stage_identity_manifest = write_canonical_stage_identity_manifest(
            ctx.paths.canonical_stage_identity_manifest,
            run_contract_path=ctx.paths.run_contract,
            canonical_solve_identity_path=canonical_solve_identity_manifest,
            outputs=pipeline_outputs,
        )
        canonical_cache_consistency_receipt = write_canonical_cache_consistency_receipt(
            ctx.paths.canonical_cache_consistency_receipt,
            run_contract_path=ctx.paths.run_contract,
            canonical_solve_identity_path=canonical_solve_identity_manifest,
            canonical_solve_cache_manifest_path=(Path(shared_solve_cache_manifest_path) if shared_solve_cache_manifest_path is not None else None),
            stage_identity_manifest_path=canonical_stage_identity_manifest,
        )
        adjacent_aoi_verification_receipt = write_standard_adjacent_aoi_verification_receipt(
            ctx.paths.adjacent_aoi_verification_receipt,
            run_contract_path=ctx.paths.run_contract,
            current_run_dir=ctx.out_dir,
            peer_run_dir=getattr(ctx, 'adjacent_aoi_peer_run_dir', None),
        )
        final_dem_writer_receipt = write_final_dem_writer_receipt(
            ctx.paths.final_dem_writer_receipt,
            run_contract_path=ctx.paths.run_contract,
            final_dem_path=final_dem_result.dem_enhanced_final_path,
            final_writer_mode=final_dem_result.final_writer_mode,
            canonical_final_dem_path=final_dem_result.canonical_final_dem_path,
            canonical_take_mask_path=final_dem_result.canonical_take_mask_path,
        )
        river_verification_summary = write_river_verification_summary(
            ctx.paths.river_verification_summary,
            run_contract_path=ctx.paths.run_contract,
            canonical_solve_identity_path=canonical_solve_identity_manifest,
            stage_identity_manifest_path=canonical_stage_identity_manifest,
            canonical_cache_consistency_receipt_path=canonical_cache_consistency_receipt,
            adjacent_aoi_verification_receipt_path=adjacent_aoi_verification_receipt,
            final_dem_path=final_dem_result.dem_enhanced_final_path,
            final_dem_receipt_path=final_dem_receipt,
            final_dem_writer_receipt_path=final_dem_writer_receipt,
        )
        write_receipt_purpose_manifest(
            ctx.paths.receipt_purpose_manifest,
            ctx=ctx,
            stage_receipts=receipts,
            canonical_manifest_path=public_canonical_solution_manifest_path(ctx),
            aoi_export_identity_path=ctx.paths.aoi_export_receipt,
        )
        cache_consistency_payload = json.loads(canonical_cache_consistency_receipt.read_text(encoding='utf-8'))
        if cache_consistency_payload.get('status') == 'failed':
            raise ValueError(
                f"canonical_solve_cache_consistency_failed:first_diverging_stage={cache_consistency_payload.get('first_diverging_stage_key')}"
            )
    return RiverLinearPipelineResult(
        run_contract_path=(ctx.paths.run_contract if (ctx.write_diagnostics or getattr(ctx, "write_core_receipts", True)) else None),
        shared_solve_reused=bool(shared_solve_reused),
        solve_result=solve_result,
        grid_result=grid_result,
        authoritative_result=authoritative_result,
        centerline_result=centerline_result,
        wse_result=wse_result,
        authoritative_bed_result=authoritative_bed_result,
        observed_offset_result=observed_offset_result,
        modeled_offset_result=modeled_offset_result,
        backbone_result=backbone_result,
        corridor_result=corridor_result,
        surface_result=surface_result,
        lock_result=lock_result,
        export_result=export_result,
        final_dem_result=final_dem_result,
        stage_receipts=receipts,
        bundle_manifest_path=manifest,
        canonical_solve_identity_manifest_path=canonical_solve_identity_manifest,
        canonical_system_identity_receipt_path=getattr(ctx, "canonical_identity_receipt_path", None),
        canonical_stage_identity_manifest_path=canonical_stage_identity_manifest,
        canonical_cache_consistency_receipt_path=canonical_cache_consistency_receipt,
        adjacent_aoi_verification_receipt_path=adjacent_aoi_verification_receipt,
        river_verification_summary_path=river_verification_summary,
        final_dem_writer_receipt_path=final_dem_writer_receipt,
        river_science_chain_summary_path=river_science_chain_summary,
    )

def resolve_canonical_parent_handoff(ctx: RiverLinearContext) -> CanonicalParentHandoff | None:
    """Resolve an existing canonical parent solution for export-only routing.

    This is a pure handoff resolver. It can adopt a raw existing parent into the
    required manifest format, but it does not run construction stages.
    """
    logger = logging.getLogger(getattr(ctx, "logger_name", "river_linear"))
    cached_solve = load_canonical_solve_cache(ctx.linear_inputs) if ctx.linear_inputs is not None else None
    shared_solve_cache_manifest_path = None
    if cached_solve is not None:
        shared_solve_cache_manifest_path = cached_solve.get('manifest_path')
    elif ctx.linear_inputs is not None:
        shared_solve_cache_manifest_path = ctx.linear_inputs.canonical_solve_cache_manifest_path

    existing_manifest = _existing_canonical_manifest_for_aoi_export(ctx)
    raw_existing_parent = _raw_existing_canonical_parent_path(ctx)
    adopted = False
    source_kind = "cached_canonical_parent" if cached_solve is not None else "existing_canonical_manifest"
    if existing_manifest is None and raw_existing_parent is not None:
        adopted_manifest = _bootstrap_cache_manifest_for_existing_parent(ctx, raw_existing_parent)
        if adopted_manifest is not None:
            logger.info(
                "[RIVER][AOI_EXPORT] Adopted existing canonical parent into required manifest handoff without rerunning construction: manifest=%s parent=%s",
                adopted_manifest,
                raw_existing_parent,
            )
            existing_manifest = adopted_manifest
            adopted = True
            source_kind = "adopted_existing_parent"

    if existing_manifest is None:
        if raw_existing_parent is not None:
            raise FileNotFoundError("existing_canonical_parent_without_required_manifest")
        return None

    return _canonical_parent_handoff_from_manifest(
        ctx,
        manifest_path=Path(existing_manifest),
        cache_manifest_path=(Path(shared_solve_cache_manifest_path) if shared_solve_cache_manifest_path is not None else None),
        source_kind=source_kind,
        reused_from_cache=bool(cached_solve is not None),
        adopted_from_existing_parent=adopted,
    )


def run_river_parent_export_workflow(ctx: RiverLinearContext) -> RiverLinearPipelineResult:
    """Run the active canonical-parent/AOI-export river workflow.

    The routing is explicit:
    - AOI-export-only mode requires an existing canonical parent handoff.
    - A reusable parent handoff exports directly without construction stages.
    - Otherwise, the canonical build path constructs the parent and exports AOI.
    """
    logger = logging.getLogger(getattr(ctx, "logger_name", "river_linear"))
    handoff = resolve_canonical_parent_handoff(ctx)

    if ctx.execution_role == AOI_EXPORT_ONLY_ROLE:
        if handoff is None:
            raise FileNotFoundError("missing_canonical_manifest_for_aoi_export_only")
        return run_aoi_export_from_canonical_parent(ctx, handoff)

    if handoff is not None and handoff.reused_from_cache:
        logger.info(
            "[RIVER][AOI_EXPORT] Existing canonical manifest and parent found; exporting AOI without rerunning canonical construction: manifest=%s parent=%s",
            handoff.manifest_path,
            handoff.parent_dem_path,
        )
        return run_aoi_export_from_canonical_parent(ctx, handoff)

    return run_canonical_river_build(ctx)


def run_river_linear_pipeline(ctx: RiverLinearContext) -> RiverLinearPipelineResult:
    """Backward-compatible entry point for the active river parent/export workflow."""
    return run_river_parent_export_workflow(ctx)


# Backward-compatible internal alias for older tests/imports.
_run_existing_parent_aoi_export = run_aoi_export_from_canonical_parent


__all__ = [
    'AOIExportResult',
    'CanonicalParentHandoff',
    'CanonicalRiverBuildResult',
    'RiverLinearPipelineResult',
    'RiverParentExportWorkflowResult',
    'resolve_canonical_parent_handoff',
    'run_aoi_export_from_canonical_parent',
    'run_canonical_river_build',
    'run_river_linear_pipeline',
    'run_river_parent_export_workflow',
]
