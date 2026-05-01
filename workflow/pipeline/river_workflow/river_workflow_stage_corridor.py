from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.ops import transform as shp_transform

from pipeline.river_workflow.river_workflow_context import RiverWorkflowContext
from pipeline.river_workflow.river_workflow_stage_centerline import CenterlineStageResult
from pipeline.river_workflow.river_workflow_stage_grids import GridStageResult
from pipeline.river_workflow.river_workflow_stage_solve_domain import SolveDomainStageResult
from pipeline.river_workflow.river_workflow_validation import validate_mask_raster


@dataclass(frozen=True)
class CorridorStageResult:
    river_corridor_solve_path: Path
    corridor_pixel_count: int
    corridor_source: str = 'canonical_linear_polygons'


def _read_optional_layer(gpkg_path: Path, layer_name: str):
    try:
        layers = gpd.list_layers(gpkg_path)
        names = set(layers['name'].tolist())
    except (OSError, RuntimeError, ValueError):
        names = set()
    if layer_name not in names:
        return None
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    if gdf is None:
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


def _build_polygon_corridor_mask(network_gpkg_path: Path, grid_path: Path) -> np.ndarray | None:
    polygons = _read_optional_layer(network_gpkg_path, 'linear_polygons')
    if polygons is None or polygons.empty:
        return None
    with rasterio.open(grid_path) as ds:
        work = _normalize_to_crs(polygons, str(ds.crs))
        shapes = [geom for geom in work.geometry if geom is not None and not geom.is_empty]
        if not shapes:
            return None
        mask = rasterize(
            [(geom, 1) for geom in shapes],
            out_shape=(ds.height, ds.width),
            transform=ds.transform,
            fill=0,
            dtype='uint8',
            all_touched=True,
        )
    return mask > 0


def _write_corridor_mask(mask: np.ndarray, template_path: Path, dst_path: Path) -> None:
    with rasterio.open(template_path) as template_ds:
        profile = template_ds.profile.copy()
        for k in ('blockxsize', 'blockysize', 'BLOCKXSIZE', 'BLOCKYSIZE'):
            profile.pop(k, None)
        profile.update(dtype='uint8', nodata=0, count=1, compress='deflate')
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink()
        with rasterio.open(dst_path, 'w', **profile) as out_ds:
            out_ds.write(np.asarray(mask, dtype=np.uint8), 1)


def run_corridor_stage(
    ctx: RiverWorkflowContext,
    solve_result: SolveDomainStageResult,
    grid_result: GridStageResult,
    centerline_result: CenterlineStageResult,
) -> CorridorStageResult:
    centerline = gpd.read_file(centerline_result.centerline_points_path)
    if centerline.empty:
        raise RuntimeError('river_workflow_corridor_missing_centerline')
    corridor = _build_polygon_corridor_mask(solve_result.canonical_network_path, grid_result.solve_grid_template_path)
    if corridor is None:
        raise RuntimeError(
            'river_workflow_corridor_missing_canonical_polygons:'
            'expected linear_polygons layer in canonical_solve_network; '
            'refusing centerline-buffer fallback so corridor construction stays one-path'
        )
    _write_corridor_mask(corridor, grid_result.solve_grid_template_path, ctx.paths.river_corridor_solve)
    validate_mask_raster(raster_path=ctx.paths.river_corridor_solve, template_path=grid_result.solve_grid_template_path)
    return CorridorStageResult(
        river_corridor_solve_path=ctx.paths.river_corridor_solve,
        corridor_pixel_count=int(np.count_nonzero(corridor)),
        corridor_source='canonical_linear_polygons',
    )


__all__ = ['CorridorStageResult', 'run_corridor_stage']
