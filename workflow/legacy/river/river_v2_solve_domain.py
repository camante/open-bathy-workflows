from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from core.json_io import write_json


def _first_layer(path: Path, preferred: tuple[str, ...]) -> str | None:
    layers_df = gpd.list_layers(path)
    names = [str(layer.name) for layer in layers_df.itertuples(index=False)] if layers_df is not None else []
    for name in preferred:
        if name in names:
            return name
    return names[0] if names else None


def _geometry_union(geoms):
    if hasattr(geoms, 'union_all'):
        return geoms.union_all()
    if hasattr(geoms, 'unary_union'):
        return geoms.unary_union
    from shapely.ops import unary_union
    return unary_union(list(geoms))


def materialize_solve_channel_mask_from_network(
    *,
    network_gpkg: str | Path,
    template_raster_path: str | Path,
    out_path: str | Path,
    summary_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Rasterize the solve-domain water mask used by the active River v2 stages.

    When the solve domain expands downstream beyond the export AOI, the early River v2
    stages need a mask on the expanded solve grid rather than the original export mask.
    This helper keeps that handoff explicit and debuggable.
    """
    network_gpkg = Path(network_gpkg)
    template_raster_path = Path(template_raster_path)
    out_path = Path(out_path)
    summary: dict[str, Any] = {
        'status': 'failed',
        'network_gpkg': str(network_gpkg),
        'template_raster_path': str(template_raster_path),
        'polygon_layer_used': None,
        'line_layer_used': None,
        'polygon_count': 0,
        'mask_cell_count': 0,
    }
    line_layer = _first_layer(network_gpkg, ('major_system_network', 'rivers_clip', 'major_system_network_clip'))
    if line_layer is None:
        raise RuntimeError('river_v2_solve_channel_mask_missing_network_layer')
    summary['line_layer_used'] = line_layer
    line_gdf = gpd.read_file(network_gpkg, layer=line_layer)
    line_gdf = line_gdf[line_gdf.geometry.notnull() & ~line_gdf.geometry.is_empty].copy()
    if line_gdf.empty:
        raise RuntimeError('river_v2_solve_channel_mask_empty_network_layer')

    poly_layer = _first_layer(network_gpkg, ('nhdarea_clip', 'nhdarea'))
    if poly_layer is None:
        raise RuntimeError('river_v2_solve_channel_mask_missing_nhdarea_layer')
    summary['polygon_layer_used'] = poly_layer
    poly_gdf = gpd.read_file(network_gpkg, layer=poly_layer)
    poly_gdf = poly_gdf[poly_gdf.geometry.notnull() & ~poly_gdf.geometry.is_empty].copy()
    if poly_gdf.empty:
        raise RuntimeError('river_v2_solve_channel_mask_empty_nhdarea_layer')

    with rasterio.open(template_raster_path) as template_ds:
        target_crs = template_ds.crs
        if target_crs is not None:
            if line_gdf.crs is None:
                raise RuntimeError('river_v2_solve_channel_mask_network_missing_crs')
            if str(line_gdf.crs) != str(target_crs):
                line_gdf = line_gdf.to_crs(target_crs)
            if poly_gdf.crs is None:
                raise RuntimeError('river_v2_solve_channel_mask_polygon_missing_crs')
            if str(poly_gdf.crs) != str(target_crs):
                poly_gdf = poly_gdf.to_crs(target_crs)
        reach_union = _geometry_union(line_gdf.geometry)
        poly_gdf = poly_gdf.loc[poly_gdf.geometry.intersects(reach_union)].copy()
        if poly_gdf.empty:
            raise RuntimeError('river_v2_solve_channel_mask_no_polygons_intersect_selected_reaches')
        summary['polygon_count'] = int(len(poly_gdf))
        shapes = [(geom, 1) for geom in poly_gdf.geometry if geom is not None and not geom.is_empty]
        if not shapes:
            raise RuntimeError('river_v2_solve_channel_mask_no_rasterizable_polygons')
        mask = rasterize(
            shapes,
            out_shape=(template_ds.height, template_ds.width),
            transform=template_ds.transform,
            fill=0,
            all_touched=True,
            dtype='uint8',
        )
        mask_cell_count = int(np.count_nonzero(mask > 0))
        if mask_cell_count <= 0:
            raise RuntimeError('river_v2_solve_channel_mask_empty_after_rasterize')
        summary.update({
            'status': 'success',
            'template_grid_shape': [int(template_ds.height), int(template_ds.width)],
            'template_grid_crs': str(template_ds.crs) if template_ds.crs is not None else None,
            'template_grid_transform': tuple(template_ds.transform),
            'mask_cell_count': mask_cell_count,
        })
        profile = template_ds.profile.copy()
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
        profile.update(driver='GTiff', dtype='uint8', count=1, nodata=0, compress='deflate', tiled=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(mask.astype('uint8'), 1)
    if summary_path is not None:
        write_json(summary_path, summary)
    return out_path, summary


__all__ = ['materialize_solve_channel_mask_from_network']
