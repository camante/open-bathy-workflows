from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely.ops import transform as shp_transform

from pipeline.river_workflow.river_workflow_context import RiverWorkflowContext
from pipeline.river_workflow.river_workflow_stage_authoritative import AuthoritativeStageResult
from pipeline.river_workflow.river_workflow_stage_grids import GridStageResult
from pipeline.river_workflow.river_workflow_stage_solve_domain import SolveDomainStageResult
from pipeline.river_workflow.river_workflow_validation import validate_centerline_points_gpkg, validate_centerline_points_overlap_template
from river_structured_scaffold import build_centerline_points
from pipeline.river_workflow.river_workflow_stage_helpers import attach_component_width_proxy, canonicalize_centerline_points, validate_centerline_points


@dataclass(frozen=True)
class CenterlineStageResult:
    centerline_points_path: Path
    record_count: int
    component_count: int


def _read_required_layer(gpkg_path: Path, layer_name: str) -> gpd.GeoDataFrame:
    layers = gpd.list_layers(gpkg_path)
    names = set(layers['name'].tolist())
    if layer_name not in names:
        raise RuntimeError(f'river_workflow_missing_required_layer:{layer_name}')
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    if gdf is None or gdf.empty:
        raise RuntimeError(f'river_workflow_empty_required_layer:{layer_name}')
    return gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()


def _read_optional_layer(gpkg_path: Path, layer_name: str) -> gpd.GeoDataFrame | None:
    try:
        layers = gpd.list_layers(gpkg_path)
        names = set(layers['name'].tolist())
    except (OSError, RuntimeError, ValueError):
        names = set()
    if layer_name not in names:
        return None
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    if gdf is None or gdf.empty:
        return None
    return gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()


def _normalize_to_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if out.crs is None or not target_crs or str(out.crs) == str(target_crs):
        return out.set_crs(str(out.crs), allow_override=True) if out.crs is not None else out
    src_crs = str(out.crs)
    tx = Transformer.from_crs(src_crs, str(target_crs), always_xy=True)
    out = out.set_geometry(out.geometry.apply(lambda geom: shp_transform(tx.transform, geom) if geom is not None and not geom.is_empty else geom))
    out = out.set_crs(str(target_crs), allow_override=True)
    return out


def run_centerline_stage(
    ctx: RiverWorkflowContext,
    solve_result: SolveDomainStageResult,
    authoritative_result: AuthoritativeStageResult,
    grid_result: GridStageResult,
    *,
    spacing_m: float = 30.0,
) -> CenterlineStageResult:
    flows = _normalize_to_crs(_read_required_layer(solve_result.canonical_network_path, 'linear_flows'), ctx.projected_crs)
    polygons = _read_optional_layer(solve_result.canonical_network_path, 'linear_polygons')
    if polygons is not None:
        polygons = _normalize_to_crs(polygons, ctx.projected_crs)
    canonical_measured = (
        Path(ctx.linear_inputs.canonical_solve_authoritative_measured_only_path)
        if ctx.linear_inputs is not None and ctx.linear_inputs.canonical_solve_authoritative_measured_only_path is not None
        else None
    )
    if canonical_measured is None or not canonical_measured.exists():
        raise RuntimeError('river_workflow_centerline_missing_canonical_measured_only')
    if Path(authoritative_result.solve_authoritative_base_measured_only_path).resolve() != canonical_measured.resolve():
        raise RuntimeError(
            f'river_workflow_centerline_noncanonical_measured_source:{authoritative_result.solve_authoritative_base_measured_only_path}:{canonical_measured}'
        )
    centerline = build_centerline_points(
        flows,
        polygons,
        spacing_m=float(spacing_m),
        raster_path=canonical_measured,
        logger=None,
    )
    if centerline is None or getattr(centerline, 'empty', True):
        raise RuntimeError('river_workflow_centerline_no_points')
    centerline = canonicalize_centerline_points(centerline)
    centerline = attach_component_width_proxy(centerline, polygons)
    validation = validate_centerline_points(centerline)
    if not validation.get('valid'):
        raise RuntimeError(f'river_workflow_centerline_invalid:{validation}')
    ctx.paths.centerline_points.parent.mkdir(parents=True, exist_ok=True)
    if ctx.paths.centerline_points.exists():
        ctx.paths.centerline_points.unlink()
    centerline.to_file(ctx.paths.centerline_points, driver='GPKG')
    validate_centerline_points_gpkg(ctx.paths.centerline_points)
    min_export_points = int(getattr(getattr(ctx, 'cfg', None), 'river_workflow_min_centerline_export_points', max(10, int(round(300.0 / max(float(spacing_m), 1.0))))))
    validate_centerline_points_overlap_template(
        path=ctx.paths.centerline_points,
        template_path=grid_result.export_grid_template_path,
        min_points=min_export_points,
    )
    component_fields = [c for c in ('component_id', 'levelpath_id', 'reach_id') if c in centerline.columns]
    if component_fields:
        comp = int(centerline[component_fields].astype(str).agg('|'.join, axis=1).nunique())
    else:
        comp = 1
    return CenterlineStageResult(
        centerline_points_path=ctx.paths.centerline_points,
        record_count=int(len(centerline)),
        component_count=int(comp),
    )


__all__ = ['CenterlineStageResult', 'run_centerline_stage']
