import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from river_channel_surface import build_channel_surface_products


def _write_raster(path: Path, arr: np.ndarray, *, nodata: float = -9999.0):
    profile = {
        'driver': 'GTiff',
        'height': arr.shape[0],
        'width': arr.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': from_origin(0, 10, 1, 1),
        'nodata': nodata,
    }
    with rasterio.open(path, 'w', **profile) as ds:
        ds.write(arr.astype(np.float32), 1)


def _make_scaffold(path: Path, *, xs: bool = True):
    recs = []
    for station, x in [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]:
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            z_source = 'xs_profile_resampled' if xs and role != 'thalweg' else 'authoritative_in_channel'
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': 1.0 - 0.05 * station + 0.1 * eta, 'z_source': z_source,
                'graph_solver_support_class': 'xs_residual_only' if xs else 'authoritative_locked',
                'graph_candidate_source': 'xs_profile_resampled' if xs else 'authoritative_backbone',
                'graph_solution_mode': 'missing' if xs else 'hard_locked',
                'graph_hard_lock': not xs,
                'graph_junction_constrained': False,
                'graph_prior_weight_sum': 2.0,
                'graph_regularization_weight_sum': 0.5,
                'graph_junction_weight_sum': 0.0,
                'graph_residual_to_candidate_z_m': 0.0,
                'graph_unsupported_span_m': 0.0,
                'graph_unsupported_regime': 'supported',
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    gdf = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    gdf.to_file(path, driver='GPKG')


def test_build_channel_surface_products_writes_xs_propagation_audit(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    scaffold_path = river_dir / 'scaffold.gpkg'
    _make_scaffold(scaffold_path, xs=True)
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)

    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    assert Path(out['river_xs_propagation_audit']).exists()
    assert Path(out['channel_surface_xs_participation']).exists()
    assert Path(out['channel_surface_influence_class']).exists()

    audit = json.loads(Path(out['river_xs_propagation_audit']).read_text())
    metrics = audit['metrics']
    assert metrics['scaffold_input_xs_node_count'] > 0
    assert metrics['admitted_surface_control_xs_node_count'] > 0
    assert metrics['final_xs_participation_cell_count'] > 0
    assert metrics['xs_propagation_warning'] is False

    with rasterio.open(out['channel_surface_xs_participation']) as ds:
        xs_part = ds.read(1)
    with rasterio.open(out['channel_surface_influence_class']) as ds:
        infl = ds.read(1)
    assert int(np.max(xs_part)) == 1
    assert int(np.max(infl)) >= 2


def test_build_channel_surface_products_flags_when_xs_built_but_never_propagates(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    scaffold_path = river_dir / 'scaffold.gpkg'
    _make_scaffold(scaffold_path, xs=True)
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    authmask = np.zeros((12, 12), dtype=np.float32)
    authdepth = np.full((12, 12), -9999.0, dtype=np.float32)
    authmask[3:8, 1:10] = 1.0
    authdepth[3:8, 1:10] = -3.0
    _write_raster(river_dir / 'authmask.tif', authmask)
    _write_raster(river_dir / 'authdepth.tif', authdepth)

    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
        authoritative_support_mask_path=river_dir / 'authmask.tif',
        authoritative_support_depth_path=river_dir / 'authdepth.tif',
    )
    audit = json.loads(Path(out['river_xs_propagation_audit']).read_text())
    metrics = audit['metrics']
    assert metrics['scaffold_input_xs_node_count'] > 0
    assert metrics['final_xs_participation_in_non_authoritative_support_cells'] == 0
    assert metrics['xs_propagation_warning'] is False  # fully authoritative support leaves no non-authoritative cells to warn on



def test_build_channel_surface_products_writes_support_aware_control_exports(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    scaffold_path = river_dir / 'scaffold.gpkg'
    _make_scaffold(scaffold_path, xs=True)
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)

    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    assert Path(out['river_channel_surface_control_nodes.gpkg']).exists()
    assert Path(out['river_channel_surface_xs_admissibility_mask.tif']).exists()

    nodes = gpd.read_file(out['river_channel_surface_control_nodes.gpkg'])
    assert 'surface_control_admitted' in nodes.columns
    assert 'surface_control_xs_expected' in nodes.columns
    assert int(nodes['surface_control_admitted'].sum()) > 0
    assert int(nodes['surface_control_xs_expected'].sum()) > 0

    audit = json.loads(Path(out['river_xs_propagation_audit']).read_text())
    metrics = audit['metrics']
    assert metrics['xs_expected_surface_control_node_count'] > 0
    assert metrics['final_xs_admissibility_cell_count'] > 0
