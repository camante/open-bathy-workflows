from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from pipeline.river_linear.river_linear_context import CanonicalRiverSourceContext, RiverLinearContext
from pipeline.river_linear.river_linear_stage_authoritative import AuthoritativeStageResult
from pipeline.river_linear.river_linear_stage_centerline import CenterlineStageResult
from pipeline.river_linear.river_linear_validation import validate_authoritative_bed_points_gpkg
from pipeline.river_linear.river_linear_stage_helpers import sample_raster_to_points, validate_centerline_authoritative_bed


@dataclass(frozen=True)
class AuthoritativeBedStageResult:
    centerline_authoritative_bed_points_path: Path
    record_count: int
    finite_bed_count: int


def _load_canonical_source_context(ctx: RiverLinearContext) -> CanonicalRiverSourceContext:
    linear_inputs = ctx.linear_inputs
    if linear_inputs is None or linear_inputs.canonical_source_context_path is None:
        raise RuntimeError('river_linear_missing_canonical_source_context')
    path = Path(linear_inputs.canonical_source_context_path)
    if not path.exists():
        raise RuntimeError(f'river_linear_missing_canonical_source_context_file:{path}')
    return CanonicalRiverSourceContext.from_json(path)


def _require_canonical_authoritative_inputs(
    ctx: RiverLinearContext,
    authoritative_result: AuthoritativeStageResult,
) -> CanonicalRiverSourceContext:
    canonical_ctx = _load_canonical_source_context(ctx)
    linear_inputs = ctx.linear_inputs
    if linear_inputs is None:
        raise RuntimeError('river_linear_missing_source_bundle')
    if str(linear_inputs.authoritative_routing_policy) != 'solve_stages_sample_only_from_canonical_authoritative':
        raise RuntimeError('river_linear_authoritative_routing_policy_not_canonical_only')
    expected_measured = Path(canonical_ctx.canonical_solve_authoritative_measured_only_path or '')
    expected_support = Path(canonical_ctx.canonical_solve_authoritative_support_mask_path or '')
    if not expected_measured.exists():
        raise RuntimeError(f'river_linear_missing_canonical_measured_only:{expected_measured}')
    if not expected_support.exists():
        raise RuntimeError(f'river_linear_missing_canonical_support_mask:{expected_support}')
    actual_measured = Path(authoritative_result.solve_authoritative_base_measured_only_path)
    actual_support = Path(authoritative_result.solve_authoritative_support_mask_path)
    if actual_measured.resolve() != expected_measured.resolve():
        raise RuntimeError(f'river_linear_authoritative_bed_noncanonical_measured_source:{actual_measured}:{expected_measured}')
    if actual_support.resolve() != expected_support.resolve():
        raise RuntimeError(f'river_linear_authoritative_bed_noncanonical_support_source:{actual_support}:{expected_support}')
    return canonical_ctx


def _sample_authoritative_bed_from_canonical_rasters(
    centerline_points_gdf: gpd.GeoDataFrame,
    *,
    measured_raster_path: Path,
    support_mask_path: Path,
    canonical_ctx: CanonicalRiverSourceContext,
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    measured_vals, measured_diag = sample_raster_to_points(centerline_points_gdf, measured_raster_path)
    support_vals, support_diag = sample_raster_to_points(centerline_points_gdf, support_mask_path, search_radius_cells=0)
    support_mask = np.isfinite(support_vals) & (support_vals > 0.0)
    bed_mask = np.isfinite(measured_vals)
    keep_mask = support_mask & bed_mask

    out = centerline_points_gdf.copy()
    out['authoritative_support_present'] = support_mask.astype(np.uint8)
    out['authoritative_bed_z_m'] = pd.to_numeric(measured_vals, errors='coerce')
    out['authoritative_source_role'] = 'canonical_authoritative_bed'
    out['authoritative_source_path'] = str(measured_raster_path)
    out['authoritative_support_mask_path'] = str(support_mask_path)
    out['canonical_system_id'] = str(canonical_ctx.canonical_system_id) if canonical_ctx.canonical_system_id is not None else None
    out = gpd.GeoDataFrame(out.loc[keep_mask].copy(), geometry='geometry', crs=centerline_points_gdf.crs)
    keep_cols = [
        c for c in (
            'point_id', 'station_m', 'component_id', 'levelpath_id', 'reach_id', 'source_reach_key',
            'authoritative_bed_z_m', 'authoritative_support_present', 'authoritative_source_role',
            'authoritative_source_path', 'authoritative_support_mask_path', 'canonical_system_id', 'geometry'
        ) if c in out.columns
    ]
    out = out[keep_cols]
    sort_cols = [c for c in ('station_m', 'point_id') if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    diagnostics = {
        'source_policy': str(canonical_ctx.source_policy),
        'canonical_solve_aoi': str(canonical_ctx.canonical_solve_aoi),
        'canonical_system_id': str(canonical_ctx.canonical_system_id) if canonical_ctx.canonical_system_id is not None else None,
        'canonical_network_gpkg': str(canonical_ctx.canonical_network_gpkg),
        'measured_raster_path': str(measured_raster_path),
        'support_mask_path': str(support_mask_path),
        'stations_attempted': int(len(centerline_points_gdf)),
        'support_positive_count': int(np.count_nonzero(support_mask)),
        'finite_measured_count': int(np.count_nonzero(bed_mask)),
        'written_record_count': int(len(out)),
        'measured_sampling': measured_diag,
        'support_sampling': support_diag,
        'support_search_radius_cells': 0,
    }
    return out, diagnostics


def run_authoritative_bed_stage(
    ctx: RiverLinearContext,
    centerline_result: CenterlineStageResult,
    authoritative_result: AuthoritativeStageResult,
) -> AuthoritativeBedStageResult:
    canonical_ctx = _require_canonical_authoritative_inputs(ctx, authoritative_result)
    centerline = gpd.read_file(centerline_result.centerline_points_path)
    gdf, diagnostics = _sample_authoritative_bed_from_canonical_rasters(
        centerline,
        measured_raster_path=Path(authoritative_result.solve_authoritative_base_measured_only_path),
        support_mask_path=Path(authoritative_result.solve_authoritative_support_mask_path),
        canonical_ctx=canonical_ctx,
    )
    validation = validate_centerline_authoritative_bed(gdf)
    if not validation.get('valid'):
        raise RuntimeError(f"river_linear_authoritative_bed_invalid:{validation}:diag={diagnostics}")
    if ctx.paths.centerline_authoritative_bed_points.exists():
        ctx.paths.centerline_authoritative_bed_points.unlink()
    gdf.to_file(ctx.paths.centerline_authoritative_bed_points, driver='GPKG')
    validate_authoritative_bed_points_gpkg(ctx.paths.centerline_authoritative_bed_points)
    if ctx.write_diagnostics:
        diagnostics_path = ctx.paths.authoritative_bed_receipt.parent / 'authoritative_bed_sampling_diagnostics.json'
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return AuthoritativeBedStageResult(
        centerline_authoritative_bed_points_path=ctx.paths.centerline_authoritative_bed_points,
        record_count=int(len(gdf)),
        finite_bed_count=int(validation.get('finite_authoritative_bed_count', 0)),
    )


__all__ = ['AuthoritativeBedStageResult', 'run_authoritative_bed_stage']
