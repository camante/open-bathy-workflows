import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from river_channel_frame import build_channel_frame_products
from river_channel_scaffold import build_channel_scaffold_products
from river_channel_surface import build_channel_surface_products, _select_component_render_mode, _apply_component_longitudinal_smoothing, _resolve_station_support_distance_series, _backfill_node_section_target_fields, _write_role_agreement_receipts, _write_section_target_agreement_receipts
from authoritative_river_roles import ROLE_BANK_MARGIN, ROLE_BED_INNER, role_to_code


def _write_raster(path: Path, arr: np.ndarray, *, nodata: float = -9999.0, crs: str = 'EPSG:4326', transform=None):
    profile = {
        'driver': 'GTiff',
        'height': arr.shape[0],
        'width': arr.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': crs,
        'transform': from_origin(0, 10, 1, 1) if transform is None else transform,
        'nodata': nodata,
    }
    with rasterio.open(path, 'w', **profile) as ds:
        ds.write(arr.astype(np.float32), 1)

def _basic_surface_nodes():
    recs = []
    for station, x in [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]:
        bed_by_role = {
            'left_bank': 10.0,
            'left_inner': 7.2,
            'thalweg': 4.0,
            'right_inner': 7.0,
            'right_bank': 10.0,
        }
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            y = 5.5 + (eta - 0.5) * 4.0
            recs.append({
                'component_id': 'main',
                'station_m': station,
                'node_role': role,
                'cross_stream_eta': eta,
                'half_width_m': 2.0,
                'bed_z_m': bed_by_role[role],
                'z_source': 'graph_backbone',
                'graph_solver_support_class': 'unsupported',
                'component_support_class': 'unsupported_mainstem',
                'station_support_regime': 'bank_only_low_confidence',
                'primary_surface_rebuild_applied': role in {'thalweg', 'left_inner', 'right_inner'},
                'primary_surface_backbone_led_inner_targets': True,
                'primary_surface_backbone_led_inner_relief_scale': 0.55,
                'primary_surface_backbone_led_inner_reason': 'bank_margin_only',
                'primary_surface_bank_margin_damped': True,
                'center_x': x,
                'center_y': 5.5,
                'normal_x': 0.0,
                'normal_y': 1.0,
                'geometry': Point(x, y),
            })
    return recs


def test_build_channel_surface_products(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    center = gpd.GeoDataFrame(
        {'station_m': [0.0, 10.0, 20.0], 'component_id': ['main', 'main', 'main']},
        geometry=[Point(2, 8), Point(5, 5), Point(8, 2)],
        crs='EPSG:4326',
    )
    center_path = river_dir / 'centerline_points.gpkg'
    center.to_file(center_path, driver='GPKG')
    xs_support = gpd.GeoDataFrame(
        {'xs_sample_source': ['authoritative', 'authoritative', 'authoritative'], 'xs_z_m': [1.0, 0.5, 0.2]},
        geometry=[Point(2, 8), Point(5, 5), Point(8, 2)],
        crs='EPSG:4326',
    )
    xs_support_path = river_dir / 'xs_support_points.gpkg'
    xs_support.to_file(xs_support_path, driver='GPKG')
    arr = np.full((10, 10), -9999.0, dtype=np.float32)
    arr[2, 2] = 1.0
    arr[5, 5] = 0.5
    arr[8, 8] = 0.2
    for name in ['bank.tif', 'bankinf.tif', 'xselev.tif', 'xsw.tif', 'centelev.tif', 'centinf.tif', 'long.tif', 'authmask.tif', 'authdepth.tif', 'authbed.tif']:
        a = arr.copy()
        if name == 'xsw.tif':
            a[:] = -9999.0
            a[2, 2] = 1.0; a[5, 5] = 1.0; a[8, 8] = 1.0
        if name == 'authmask.tif':
            a[:] = -9999.0
            a[2, 2] = 1.0; a[5, 5] = 1.0; a[8, 8] = 1.0
        _write_raster(river_dir / name, a)
    corridor = np.zeros((10, 10), dtype=np.float32)
    corridor[1:9, 1:9] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    frame = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=center_path,
        xs_support_points_path=xs_support_path,
        bank_elevation_path=river_dir / 'bank.tif',
        bank_influence_path=river_dir / 'bankinf.tif',
        xs_support_elevation_path=river_dir / 'xselev.tif',
        xs_support_weight_path=river_dir / 'xsw.tif',
        centerline_elevation_path=river_dir / 'centelev.tif',
        centerline_influence_path=river_dir / 'centinf.tif',
        longitudinal_profile_elevation_path=river_dir / 'long.tif',
        authoritative_support_mask_path=river_dir / 'authmask.tif',
        authoritative_support_depth_path=river_dir / 'authdepth.tif',
        authoritative_bed_elevation_path=river_dir / 'authbed.tif',
    )
    xs_bathy = gpd.GeoDataFrame(
        {
            'xs_id': ['A', 'A', 'A', 'B', 'B', 'B'],
            'width_m': [10.0] * 6,
            'dist_m': [0.0, 5.0, 10.0, 0.0, 5.0, 10.0],
            'z_bed_pred_m': [2.0, 1.0, 2.0, 1.8, 0.8, 1.8],
        },
        geometry=[Point(2, 8), Point(2, 8), Point(2, 8), Point(8, 2), Point(8, 2), Point(8, 2)],
        crs='EPSG:4326',
    )
    xs_bathy_path = tmp_path / 'river_bathy_xs_mainstem.gpkg'
    xs_bathy.to_file(xs_bathy_path, layer='xs_bathy_points', driver='GPKG')
    scaffold = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame['channel_frame_points'],
        xs_bathy_gpkg_path=xs_bathy_path,
    )
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold['channel_scaffold_nodes'],
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    assert Path(out['channel_surface']).exists()
    assert Path(out['channel_surface_confidence']).exists()
    assert Path(out['channel_surface_contract']).exists()
    with rasterio.open(out['channel_surface']) as ds:
        arr = ds.read(1)
    finite = arr[arr > -9999.0]
    assert finite.size > 0
    with rasterio.open(out['channel_surface_source_class']) as ds:
        src = ds.read(1)
    assert int(np.max(src)) >= 3


def test_build_channel_surface_products_projects_to_thalweg_segments(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    # Direct scaffold nodes so we can verify segment-projected station interpolation.
    roles = ['left_bank', 'left_inner', 'thalweg', 'right_inner', 'right_bank']
    recs = []
    for station, x, z in [(0.0, 2.5, 2.0), (10.0, 5.5, 1.0), (20.0, 8.5, 0.0)]:
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            recs.append({
                'component_id': 'main',
                'station_m': station,
                'node_role': role,
                'cross_stream_eta': eta,
                'half_width_m': 2.0,
                'bed_z_m': z,
                'z_source': 'authoritative_in_channel' if role == 'thalweg' else 'xs_profile_resampled',
                'center_x': x,
                'center_y': 5.5,
                'normal_x': 0.0,
                'normal_y': 1.0,
                'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    with rasterio.open(out['channel_surface']) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
    # Middle corridor cell should be interpolated along the segment between stations,
    # not snapped to the nearest station value.
    mid = float(arr[4, 4])
    assert nodata is not None
    assert mid > nodata
    assert 1.1 < mid < 1.9


def test_build_channel_surface_products_authoritative_override(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x, z in [(0.0, 2.5, 2.0), (10.0, 5.5, 1.0), (20.0, 8.5, 0.0)]:
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': z, 'z_source': 'xs_profile_resampled',
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    authmask = np.zeros((12, 12), dtype=np.float32)
    authdepth = np.full((12, 12), -9999.0, dtype=np.float32)
    authmask[5, 5] = 1.0
    authdepth[5, 5] = -3.5
    _write_raster(river_dir / 'authmask.tif', authmask)
    _write_raster(river_dir / 'authdepth.tif', authdepth)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
        authoritative_support_mask_path=river_dir / 'authmask.tif',
        authoritative_support_depth_path=river_dir / 'authdepth.tif',
    )
    with rasterio.open(out['channel_surface']) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
    assert nodata is not None
    assert arr[5, 5] == np.float32(-3.5)
    with rasterio.open(out['channel_surface_source_class']) as ds:
        src = ds.read(1)
    assert int(src[5, 5]) == 4


def test_build_channel_surface_products_renders_scaffold_bed_directly(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x in [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]:
        bed_by_role = {
            'left_bank': 10.0,
            'left_inner': 7.0,
            'thalweg': 4.0,
            'right_inner': 3.0,
            'right_bank': 6.0,
        }
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': bed_by_role[role], 'z_source': 'graph_backbone',
                'backbone_bed_z_m': 100.0, 'residual_shape_z_m': -50.0,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    with rasterio.open(out['channel_surface']) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
    center_val = float(arr[4, 5])
    top_val = float(arr[6, 5])
    bottom_val = float(arr[3, 5])
    assert nodata is not None
    assert center_val > nodata and top_val > nodata and bottom_val > nodata
    assert 3.5 < center_val < 4.5
    assert 8.0 < top_val < 10.5
    assert 2.5 < bottom_val < 3.7


def test_build_channel_surface_products_prefers_graph_provenance_for_source_and_confidence(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    stations = [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]
    for station, x in stations:
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': 5.0 - eta, 'z_source': 'bank_stage_prior',
                'graph_solver_support_class': 'resolved_backbone',
                'graph_candidate_source': 'longitudinal_profile',
                'graph_solution_mode': 'junction_constrained',
                'graph_hard_lock': False,
                'graph_prior_weight_sum': 1.0,
                'graph_regularization_weight_sum': 2.0,
                'graph_junction_weight_sum': 6.0,
                'graph_residual_to_candidate_z_m': 0.8,
                'graph_unsupported_span_m': 120.0,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    with rasterio.open(out['channel_surface_source_class']) as ds:
        src = ds.read(1)
    with rasterio.open(out['channel_surface_graph_mode']) as ds:
        mode = ds.read(1)
    with rasterio.open(out['channel_surface_confidence']) as ds:
        conf = ds.read(1)
    # Despite scaffold z_source=bank_stage_prior, graph provenance should advertise resolved backbone / junction constrained.
    assert int(src[4, 5]) == 6
    assert int(mode[4, 5]) == 4
    assert float(conf[4, 5]) < 0.7


def test_build_channel_surface_products_uses_graph_hard_lock_for_confidence(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    stations = [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]
    for station, x in stations:
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': 2.0, 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'authoritative_locked',
                'graph_candidate_source': 'authoritative_in_channel',
                'graph_solution_mode': 'hard_locked',
                'graph_hard_lock': True,
                'graph_prior_weight_sum': 0.0,
                'graph_regularization_weight_sum': 0.0,
                'graph_junction_weight_sum': 0.0,
                'graph_residual_to_candidate_z_m': 0.0,
                'graph_unsupported_span_m': 0.0,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    with rasterio.open(out['channel_surface_confidence']) as ds:
        conf = ds.read(1)
    with rasterio.open(out['channel_surface_source_class']) as ds:
        src = ds.read(1)
    with rasterio.open(out['channel_surface_graph_mode']) as ds:
        mode = ds.read(1)
    assert int(src[4, 5]) == 4
    assert int(mode[4, 5]) == 1
    assert float(conf[4, 5]) > 0.9


def test_build_channel_surface_products_writes_support_uncertainty_outputs(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    for station, x, support, mode, hard, span in [(0.0, 2.5, 'authoritative_backbone', 'prior_driven', False, 0.0), (10.0, 5.5, 'unsupported', 'junction_constrained', False, 180.0), (20.0, 8.5, 'authoritative_locked', 'hard_locked', True, 0.0)]:
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': 3.0 - 0.5 * eta, 'z_source': 'graph_backbone' if not hard else 'authoritative_in_channel',
                'graph_solver_support_class': support,
                'graph_candidate_source': 'longitudinal_profile' if not hard else 'authoritative_backbone',
                'graph_solution_mode': mode,
                'graph_hard_lock': hard,
                'graph_prior_weight_sum': 4.0 if support != 'unsupported' else 0.5,
                'graph_regularization_weight_sum': 0.5 if support != 'unsupported' else 3.0,
                'graph_junction_weight_sum': 0.0 if mode != 'junction_constrained' else 4.0,
                'graph_residual_to_candidate_z_m': 0.1 if support != 'unsupported' else 1.0,
                'graph_unsupported_span_m': span,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    assert Path(out['channel_surface_support_class']).exists()
    assert Path(out['channel_surface_uncertainty']).exists()
    assert Path(out['channel_surface_hard_lock']).exists()
    assert Path(out['channel_surface_junction_constrained']).exists()
    assert Path(out['channel_surface_unsupported_span']).exists()
    assert Path(out['channel_surface_residual_to_candidate']).exists()
    assert Path(out['support_uncertainty_contract']).exists()
    with rasterio.open(out['channel_surface_support_class']) as ds:
        support = ds.read(1)
    with rasterio.open(out['channel_surface_uncertainty']) as ds:
        uncertainty = ds.read(1)
    assert int(np.nanmax(support)) >= 1
    assert int(np.nanmax(uncertainty)) >= 1
    contract = Path(out['support_uncertainty_contract']).read_text()
    assert 'surface_support_class_counts' in contract
    assert 'surface_uncertainty_counts' in contract


def test_build_channel_surface_products_writes_junction_audit_outputs(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    for station, x, role_name, adj, dist in [(0.0, 2.5, 'dominant', 0.0, 0.0), (10.0, 5.5, 'constrained_branch', 0.6, 5.0), (20.0, 8.5, 'not_in_junction', 0.0, 20.0)]:
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': 3.0 - 0.5 * eta, 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'graph_backbone',
                'graph_candidate_source': 'longitudinal_profile',
                'graph_solution_mode': 'junction_constrained' if role_name != 'not_in_junction' else 'prior_driven',
                'graph_hard_lock': False,
                'graph_junction_constrained': role_name != 'not_in_junction',
                'graph_junction_role': role_name,
                'graph_junction_adjustment_z_m': adj,
                'graph_junction_distance_m': dist,
                'graph_prior_weight_sum': 2.0,
                'graph_regularization_weight_sum': 1.0,
                'graph_junction_weight_sum': 3.0 if role_name != 'not_in_junction' else 0.0,
                'graph_residual_to_candidate_z_m': adj,
                'graph_unsupported_span_m': 40.0,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    assert Path(out['channel_surface_junction_role']).exists()
    assert Path(out['channel_surface_junction_adjustment']).exists()
    assert Path(out['channel_surface_junction_distance']).exists()
    with rasterio.open(out['channel_surface_junction_role']) as ds:
        junction_role = ds.read(1)
    with rasterio.open(out['channel_surface_junction_adjustment']) as ds:
        junction_adj = ds.read(1)
    with rasterio.open(out['channel_surface_junction_distance']) as ds:
        junction_dist = ds.read(1)
    assert int(np.nanmax(junction_role)) >= 2
    assert float(np.nanmax(junction_adj)) > 0.0
    assert float(np.nanmax(junction_dist)) >= 5.0
    contract = Path(out['channel_surface_contract']).read_text()
    assert 'surface_junction_role_counts' in contract
    assert 'channel_surface_junction_adjustment' in contract


def test_build_channel_surface_products_exports_unsupported_regime_raster(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x, regime in [(0.0, 2.5, 'short_gap_bridge'), (10.0, 5.5, 'long_gap_stiffened'), (20.0, 8.5, 'long_gap_stiffened')]:
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': 1.0 - 0.02 * station, 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'unsupported', 'graph_solution_mode': 'regularization_driven',
                'graph_hard_lock': False, 'graph_junction_constrained': False,
                'graph_prior_weight_sum': 0.5, 'graph_regularization_weight_sum': 2.0, 'graph_junction_weight_sum': 0.0,
                'graph_residual_to_candidate_z_m': 0.1, 'graph_unsupported_span_m': 200.0 if regime == 'long_gap_stiffened' else 20.0,
                'graph_unsupported_regime': regime, 'graph_junction_role': 'not_in_junction',
                'graph_junction_adjustment_z_m': 0.0, 'graph_junction_distance_m': np.nan,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    assert Path(out['channel_surface_unsupported_regime']).exists()
    with rasterio.open(out['channel_surface_unsupported_regime']) as ds:
        regime = ds.read(1)
    assert int(np.max(regime)) >= 4
    contract = json.loads(Path(out['support_uncertainty_contract']).read_text())
    assert 'unsupported_regime_codes' in contract
    assert contract['artifacts']['channel_surface_unsupported_regime'].endswith('river_channel_surface_unsupported_regime.tif')
    assert contract['metrics']['surface_unsupported_regime_counts']['long_gap_stiffened'] > 0


def test_build_channel_surface_products_limits_authoritative_lock_to_authoritative_support_classes(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    stations = [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]
    for station, x in stations:
        for role, eta in [("left_bank", 0.0), ("left_inner", 0.25), ("thalweg", 0.5), ("right_inner", 0.75), ("right_bank", 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': 2.0 - 0.2 * eta, 'z_source': 'xs_profile_resampled',
                'graph_solver_support_class': 'unsupported',
                'graph_candidate_source': 'xs_profile_resampled',
                'graph_solution_mode': 'regularization_driven',
                'graph_hard_lock': False,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    authmask = np.zeros((12, 12), dtype=np.float32)
    authdepth = np.full((12, 12), -9999.0, dtype=np.float32)
    authmask[5, 5] = 1.0
    authdepth[5, 5] = -9.0
    _write_raster(river_dir / 'authmask.tif', authmask)
    _write_raster(river_dir / 'authdepth.tif', authdepth)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
        authoritative_support_mask_path=river_dir / 'authmask.tif',
        authoritative_support_depth_path=river_dir / 'authdepth.tif',
    )
    with rasterio.open(out['channel_surface']) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
    assert nodata is not None
    assert float(arr[5, 5]) == np.float32(-9.0)
    with rasterio.open(out['channel_surface_authoritative_lock_scope']) as ds:
        scope = ds.read(1)
    with rasterio.open(out['channel_surface_authoritative_lock_applied']) as ds:
        applied = ds.read(1)
    assert int(scope[5, 5]) == 0
    assert int(applied[5, 5]) == 1
    audit = json.loads(Path(out['river_xs_propagation_audit']).read_text())
    assert int(audit['metrics']['authoritative_support_cells_outside_lock_scope']) >= 1


def test_build_channel_surface_products_treats_xs_derived_graph_nodes_as_xs_participation(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    stations = [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]
    for station, x in stations:
        for role, eta in [("left_bank", 0.0), ("left_inner", 0.25), ("thalweg", 0.5), ("right_inner", 0.75), ("right_bank", 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': 2.0 - 0.2 * eta, 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'xs_residual_only',
                'graph_candidate_source': 'xs_profile_resampled',
                'station_support_mode': 'xs_residual_only',
                'graph_solution_mode': 'regularization_driven',
                'graph_hard_lock': False,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    audit = json.loads(Path(out['river_xs_propagation_audit']).read_text())
    assert int(audit['metrics']['scaffold_input_xs_node_count']) == len(recs)
    assert int(audit['metrics']['admitted_surface_control_xs_node_count']) == len(recs)
    assert int(audit['metrics']['final_xs_participation_cell_count']) > 0


def test_build_channel_surface_products_writes_longitudinal_tendency_receipts(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    for station, x, thalweg in [(0.0, 2.5, 0.0), (10.0, 4.5, 4.0), (20.0, 6.5, 0.0), (30.0, 8.5, 4.0), (40.0, 10.5, 0.0)]:
        for role, eta, offset in [("left_bank", 0.0, 2.0), ("thalweg", 0.5, 0.0), ("right_bank", 1.0, 2.0)]:
            recs.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "cross_stream_eta": eta,
                "half_width_m": 2.0,
                "bed_z_m": thalweg + offset,
                "z_source": "graph_backbone",
                "graph_solver_support_class": "graph_backbone",
                "graph_hard_lock": False,
                "graph_candidate_source": "missing",
                "station_support_mode": "missing",
                "graph_backbone_z_m": thalweg + offset,
                "backbone_bed_z_m": thalweg + offset,
                "center_x": x,
                "center_y": 6.5,
                "normal_x": 0.0,
                "normal_y": 1.0,
                "geometry": Point(x, 6.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs="EPSG:4326")
    scaffold_path = river_dir / "scaffold.gpkg"
    scaffold.to_file(scaffold_path, driver="GPKG")
    corridor = np.zeros((14, 14), dtype=np.float32)
    corridor[4:10, 1:12] = 1.0
    _write_raster(river_dir / "corridor.tif", corridor)
    pd.DataFrame([{
        "profile_id": "main",
        "reach_id": "main:0",
        "reach_role": "junction_adjusted",
        "station_min_m": 0.0,
        "station_max_m": 40.0,
        "unsupported_fraction": 1.0,
        "supported_fraction": 0.0,
        "authoritative_anchor_fraction": 0.0,
        "xs_support_fraction": 0.0,
        "junction_adjustment_station_fraction": 1.0,
        "component_junction_count": 1,
    }]).to_csv(river_dir / "reach.csv", index=False)
    pd.DataFrame([
        {"profile_id": "main", "station_m": 0.0, "network_backbone_elevation_m": 0.0, "network_junction_adjustment_m": 0.0, "network_backbone_source": "network_backbone_solve", "junction_hierarchy_weight": 1.0, "junction_wse_weight": 1.0},
        {"profile_id": "main", "station_m": 10.0, "network_backbone_elevation_m": 1.0, "network_junction_adjustment_m": 0.6, "network_backbone_source": "network_junction_flow_aware_solve", "junction_hierarchy_weight": 1.2, "junction_wse_weight": 1.1},
        {"profile_id": "main", "station_m": 20.0, "network_backbone_elevation_m": 2.5, "network_junction_adjustment_m": 1.0, "network_backbone_source": "network_junction_flow_aware_solve", "junction_hierarchy_weight": 1.3, "junction_wse_weight": 1.2},
        {"profile_id": "main", "station_m": 30.0, "network_backbone_elevation_m": 1.0, "network_junction_adjustment_m": 0.6, "network_backbone_source": "network_junction_flow_aware_solve", "junction_hierarchy_weight": 1.2, "junction_wse_weight": 1.1},
        {"profile_id": "main", "station_m": 40.0, "network_backbone_elevation_m": 0.0, "network_junction_adjustment_m": 0.0, "network_backbone_source": "network_backbone_solve", "junction_hierarchy_weight": 1.0, "junction_wse_weight": 1.0},
    ]).to_csv(river_dir / "longitudinal_profile.csv", index=False)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / "corridor.tif",
        reach_attributes_path=river_dir / "reach.csv",
        longitudinal_profile_path=river_dir / "longitudinal_profile.csv",
    )
    assert Path(out["longitudinal_tendency_profile"]).exists()
    assert Path(out["longitudinal_tendency_summary"]).exists()
    nodes = gpd.read_file(out["river_channel_surface_control_nodes.gpkg"])
    thalweg = nodes.loc[nodes["node_role"].astype(str).eq("thalweg")].sort_values("station_m")
    assert np.max(np.abs(pd.to_numeric(thalweg["longitudinal_tendency_delta_m"], errors="coerce").to_numpy(dtype=float))) > 0.0
    assert np.max(pd.to_numeric(thalweg["longitudinal_tendency_junction_weight"], errors="coerce").to_numpy(dtype=float)) > 0.0
    summary = json.loads(Path(out["longitudinal_tendency_summary"]).read_text(encoding="utf-8"))
    assert int(summary["junction_targeted_node_count"]) > 0


def test_build_channel_surface_products_writes_xs_realism_receipts(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    for station, x, left_inner, right_inner in [(0.0, 2.5, 7.0, 3.0), (10.0, 5.5, 9.5, 0.0), (20.0, 8.5, 7.0, 3.0)]:
        role_map = {
            "left_bank": 10.0,
            "left_inner": left_inner,
            "thalweg": 4.0,
            "right_inner": right_inner,
            "right_bank": 6.0,
        }
        for role, eta in [("left_bank", 0.0), ("left_inner", 0.25), ("thalweg", 0.5), ("right_inner", 0.75), ("right_bank", 1.0)]:
            recs.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "cross_stream_eta": eta,
                "half_width_m": 2.0,
                "bed_z_m": role_map[role],
                "z_source": "graph_backbone",
                "graph_solver_support_class": "graph_backbone",
                "graph_hard_lock": False,
                "graph_candidate_source": "missing",
                "station_support_mode": "missing",
                "graph_backbone_z_m": role_map[role],
                "backbone_bed_z_m": role_map[role],
                "center_x": x,
                "center_y": 6.5,
                "normal_x": 0.0,
                "normal_y": 1.0,
                "geometry": Point(x, 6.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs="EPSG:4326")
    scaffold_path = river_dir / "scaffold.gpkg"
    scaffold.to_file(scaffold_path, driver="GPKG")
    corridor = np.zeros((14, 14), dtype=np.float32)
    corridor[4:10, 1:12] = 1.0
    _write_raster(river_dir / "corridor.tif", corridor)
    pd.DataFrame([{
        "profile_id": "main",
        "reach_id": "main:0",
        "reach_role": "interior",
        "station_min_m": 0.0,
        "station_max_m": 20.0,
        "unsupported_fraction": 1.0,
        "supported_fraction": 0.0,
        "authoritative_anchor_fraction": 0.0,
        "xs_support_fraction": 0.0,
        "junction_adjustment_station_fraction": 0.0,
        "component_junction_count": 0,
    }]).to_csv(river_dir / "reach.csv", index=False)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / "corridor.tif",
        reach_attributes_path=river_dir / "reach.csv",
    )
    assert Path(out["xs_realism_profile"]).exists()
    assert Path(out["xs_realism_summary"]).exists()
    nodes = gpd.read_file(out["river_channel_surface_control_nodes.gpkg"])
    left_inner = nodes.loc[nodes["node_role"].astype(str).eq("left_inner")].sort_values("station_m")
    right_inner = nodes.loc[nodes["node_role"].astype(str).eq("right_inner")].sort_values("station_m")
    center_left = left_inner.loc[np.isclose(left_inner["station_m"].to_numpy(dtype=float), 10.0)].iloc[0]
    center_right = right_inner.loc[np.isclose(right_inner["station_m"].to_numpy(dtype=float), 10.0)].iloc[0]
    assert abs(float(center_left["xs_realism_delta_m"])) > 0.0
    assert abs(float(center_right["xs_realism_delta_m"])) > 0.0
    summary = json.loads(Path(out["xs_realism_summary"]).read_text(encoding="utf-8"))
    assert int(summary["adjusted_node_count"]) > 0


def test_build_channel_surface_products_does_not_hard_lock_authoritative_backbone_only_nodes(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    for station, x, support, hard in [(0.0, 2.5, "authoritative_backbone", False), (10.0, 5.5, "authoritative_locked", True)]:
        for role, eta in [("left_bank", 0.0), ("left_inner", 0.25), ("thalweg", 0.5), ("right_inner", 0.75), ("right_bank", 1.0)]:
            recs.append({
                "component_id": "main", "station_m": station, "node_role": role, "cross_stream_eta": eta,
                "half_width_m": 2.0, "bed_z_m": 2.0 - 0.25 * eta,
                "z_source": "authoritative_backbone" if station == 0.0 else "authoritative_in_channel",
                "graph_solver_support_class": support,
                "graph_candidate_source": "authoritative_backbone" if station == 0.0 else "authoritative_in_channel",
                "graph_solution_mode": "frame_backbone_resolved" if station == 0.0 else "hard_locked",
                "graph_hard_lock": hard,
                "graph_prior_weight_sum": 3.0,
                "graph_regularization_weight_sum": 1.0,
                "graph_junction_weight_sum": 0.0,
                "graph_residual_to_candidate_z_m": 0.0,
                "graph_unsupported_span_m": 0.0,
                "center_x": x, "center_y": 5.5, "normal_x": 0.0, "normal_y": 1.0, "geometry": Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs="EPSG:4326")
    scaffold_path = river_dir / "scaffold.gpkg"
    scaffold.to_file(scaffold_path, driver="GPKG")
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / "corridor.tif", corridor)
    out = build_channel_surface_products(river_dir=river_dir, channel_scaffold_nodes_path=scaffold_path, corridor_mask_path=river_dir / "corridor.tif")
    with rasterio.open(out["channel_surface_hard_lock"]) as ds:
        hard = ds.read(1)
    assert int(hard[4, 2]) == 0
    assert int(hard[4, 5]) == 1



def test_build_channel_surface_products_writes_primary_surface_rebuild_receipts(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    for station, x, left_inner, thalweg, right_inner in [(0.0, 2.5, 7.0, 3.5, 2.5), (10.0, 5.5, 9.5, 5.0, -0.5), (20.0, 8.5, 7.0, 3.0, 2.0)]:
        role_map = {
            "left_bank": 10.0,
            "left_inner": left_inner,
            "thalweg": thalweg,
            "right_inner": right_inner,
            "right_bank": 6.0,
        }
        for role, eta in [("left_bank", 0.0), ("left_inner", 0.25), ("thalweg", 0.5), ("right_inner", 0.75), ("right_bank", 1.0)]:
            recs.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "cross_stream_eta": eta,
                "half_width_m": 2.0,
                "bed_z_m": role_map[role],
                "z_source": "graph_backbone",
                "graph_solver_support_class": "graph_backbone",
                "graph_hard_lock": False,
                "graph_candidate_source": "missing",
                "station_support_mode": "missing",
                "graph_backbone_z_m": 2.0 if role == "thalweg" else role_map[role],
                "backbone_bed_z_m": 2.0 if role == "thalweg" else role_map[role],
                "center_x": x,
                "center_y": 6.5,
                "normal_x": 0.0,
                "normal_y": 1.0,
                "geometry": Point(x, 6.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs="EPSG:4326")
    scaffold_path = river_dir / "scaffold.gpkg"
    scaffold.to_file(scaffold_path, driver="GPKG")
    corridor = np.zeros((14, 14), dtype=np.float32)
    corridor[4:10, 1:12] = 1.0
    _write_raster(river_dir / "corridor.tif", corridor)
    pd.DataFrame([{
        "profile_id": "main",
        "reach_id": "main:0",
        "reach_role": "interior",
        "station_min_m": 0.0,
        "station_max_m": 20.0,
        "unsupported_fraction": 1.0,
        "supported_fraction": 0.0,
        "authoritative_anchor_fraction": 0.0,
        "xs_support_fraction": 0.0,
        "junction_adjustment_station_fraction": 0.0,
        "component_junction_count": 0,
    }]).to_csv(river_dir / "reach.csv", index=False)
    pd.DataFrame([
        {"profile_id": "main", "station_m": 0.0, "network_backbone_elevation_m": 3.5, "network_junction_adjustment_m": 0.0, "network_backbone_source": "network_backbone_solve", "junction_hierarchy_weight": 1.0, "junction_wse_weight": 1.0},
        {"profile_id": "main", "station_m": 10.0, "network_backbone_elevation_m": 2.0, "network_junction_adjustment_m": 0.0, "network_backbone_source": "network_backbone_solve", "junction_hierarchy_weight": 1.0, "junction_wse_weight": 1.0},
        {"profile_id": "main", "station_m": 20.0, "network_backbone_elevation_m": 1.5, "network_junction_adjustment_m": 0.0, "network_backbone_source": "network_backbone_solve", "junction_hierarchy_weight": 1.0, "junction_wse_weight": 1.0},
    ]).to_csv(river_dir / "longitudinal_profile.csv", index=False)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / "corridor.tif",
        reach_attributes_path=river_dir / "reach.csv",
        longitudinal_profile_path=river_dir / "longitudinal_profile.csv",
    )
    assert Path(out["primary_surface_rebuild_profile"]).exists()
    assert Path(out["primary_surface_rebuild_summary"]).exists()
    nodes = gpd.read_file(out["river_channel_surface_control_nodes.gpkg"])
    center = nodes.loc[nodes["node_role"].astype(str).eq("thalweg") & np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce").to_numpy(dtype=float), 10.0)].iloc[0]
    assert abs(float(center["primary_surface_rebuild_delta_m"])) >= 0.0
    summary = json.loads(Path(out["primary_surface_rebuild_summary"]).read_text(encoding="utf-8"))
    assert int(summary["adjusted_node_count"]) > 0



def test_select_component_render_mode_preserves_full_scaffold_when_not_weak_support():
    rows = []
    for i in range(300):
        role = ["left_bank", "left_inner", "thalweg", "right_inner", "right_bank"][i % 5]
        rows.append({
            "graph_solver_support_class": "authoritative_locked",
            "z_source": "authoritative_in_channel" if role == "thalweg" else "graph_backbone",
            "node_role": role,
            "station_support_mode": "bank_supported",
            "station_support_regime": "anchored_interpolated",
            "primary_surface_rebuild_applied": role in {"thalweg", "left_inner", "right_inner"},
        })
    sub = pd.DataFrame(rows)
    mode, roles, reason, selector = _select_component_render_mode(sub)
    assert mode == "full_scaffold"
    assert [r for r, _ in roles] == ["left_bank", "left_inner", "thalweg", "right_inner", "right_bank"]
    assert reason == "preserve_inner_roles_and_weak_support"
    assert selector['render_mode'] == 'full_scaffold'



def test_select_component_render_mode_promotes_thalweg_dominant_for_weak_support():
    rows = []
    for i in range(300):
        role = ["left_bank", "left_inner", "thalweg", "right_inner", "right_bank"][i % 5]
        rows.append({
            "graph_solver_support_class": "unsupported",
            "z_source": "graph_backbone",
            "node_role": role,
            "station_support_mode": "graph_backbone",
            "station_support_regime": "bank_only_low_confidence",
            "primary_surface_rebuild_applied": role in {"thalweg", "left_inner", "right_inner"},
        })
    sub = pd.DataFrame(rows)
    mode, roles, reason, selector = _select_component_render_mode(sub)
    assert mode == "thalweg_dominant_scaffold"
    assert [r for r, _ in roles] == ["left_bank", "left_inner", "thalweg", "right_inner", "right_bank"]
    assert reason == "weak_support_thalweg_dominant"
    assert selector['render_mode'] == 'thalweg_dominant_scaffold'





def test_select_component_render_mode_promotes_thalweg_dominant_for_longitudinal_only_no_inner_case():
    rows = []
    for i in range(120):
        role = ["left_bank", "thalweg", "right_bank"][i % 3]
        rows.append({
            "component_id": "main",
            "graph_solver_support_class": "unsupported",
            "z_source": "graph_backbone",
            "node_role": role,
            "station_support_mode": "graph_backbone",
            "station_support_regime": "anchored_interpolated",
            "longitudinal_support_regime": "longitudinal_weak_support",
            "station_rebuild_regime": "blocked_no_inner_rebuildable_nodes",
            "target_xs_realism_allowed": False,
            "target_xs_residual_allowed": False,
            "primary_surface_rebuild_applied": role == "thalweg",
        })
    sub = pd.DataFrame(rows)
    mode, roles, reason, selector = _select_component_render_mode(sub)
    assert mode == "thalweg_dominant_scaffold"
    assert [r for r, _ in roles] == ["left_bank", "thalweg", "right_bank"]
    assert reason == "weak_support_thalweg_dominant"
    assert selector['longitudinal_weak'] is True
    assert selector['blocked_no_inner_rebuildable'] is True
    assert selector['xs_disabled_component'] is True



def test_select_component_render_mode_promotes_thalweg_dominant_for_unsupported_backbone_led_component():
    rows = []
    for i in range(120):
        role = ["left_bank", "left_inner", "thalweg", "right_inner", "right_bank"][i % 5]
        rows.append({
            "component_id": "main",
            "component_support_class": "unsupported_mainstem",
            "graph_solver_support_class": "unsupported",
            "z_source": "graph_backbone",
            "node_role": role,
            "station_support_mode": "bank_supported",
            "station_support_regime": "anchored_interpolated",
            "primary_surface_rebuild_applied": role in {"thalweg", "left_inner", "right_inner"},
            "primary_surface_backbone_led_inner_targets": True,
            "primary_surface_bank_margin_damped": True,
        })
    sub = pd.DataFrame(rows)
    mode, roles, reason, selector = _select_component_render_mode(sub)
    assert mode == "thalweg_dominant_scaffold"
    assert reason == "weak_support_thalweg_dominant"
    assert selector['unsupported_component'] is True
    assert selector['backbone_led_inner_target_share'] > 0.9
    assert selector['bank_margin_damped_share'] > 0.9


def test_build_channel_surface_products_uses_thalweg_dominant_render_for_weak_support_components(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x in [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]:
        bed_by_role = {
            'left_bank': 10.0,
            'left_inner': 8.0,
            'thalweg': 4.0,
            'right_inner': 7.0,
            'right_bank': 10.0,
        }
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': bed_by_role[role], 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'unsupported',
                'station_support_regime': 'unsupported',
                'primary_surface_rebuild_applied': role in {'thalweg', 'left_inner', 'right_inner'},
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    with rasterio.open(out['channel_surface']) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
    assert nodata is not None
    center_val = float(arr[4, 5])
    inner_like_val = float(arr[5, 5])
    bankward_val = float(arr[6, 5])
    assert 3.5 < center_val < 4.5
    assert inner_like_val > center_val
    assert inner_like_val < 7.05
    assert bankward_val > inner_like_val
    summary = pd.read_csv(Path(river_dir / 'river_render_mode_summary.csv'))
    assert set(summary['render_mode'].astype(str)) == {'thalweg_dominant_scaffold'}
    selector = pd.read_csv(Path(river_dir / 'river_render_mode_selector_summary.csv'))
    assert set(selector['render_mode'].astype(str)) == {'thalweg_dominant_scaffold'}





def test_build_channel_surface_products_uses_local_authoritative_section_targets_in_weak_support(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x in [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]:
        bed_by_role = {
            'left_bank': 10.0,
            'left_inner': 8.0,
            'thalweg': 4.0,
            'right_inner': 7.0,
            'right_bank': 10.0,
        }
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            y = 5.5 + (eta - 0.5) * 4.0
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': bed_by_role[role], 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'unsupported',
                'station_support_regime': 'unsupported',
                'primary_surface_rebuild_applied': role in {'thalweg', 'left_inner', 'right_inner'},
                'station_target_present': True,
                'station_target_local_authoritative_reconciled': True,
                'authoritative_reconciliation_weight': 0.8,
                'station_authoritative_bed_support_distance_m': 2.0,
                'target_left_bank_z_m': 8.0,
                'target_right_bank_z_m': 8.0,
                'target_thalweg_z_m': 4.0,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, y),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    with rasterio.open(out['channel_surface']) as ds:
        arr = ds.read(1)
    inner_like_val = float(arr[5, 5])
    assert inner_like_val < 6.05
    summary_path = Path(out['channel_surface_section_target_agreement_summary'])
    profile_path = Path(out['channel_surface_section_target_agreement_profile'])
    assert summary_path.exists()
    assert profile_path.exists()
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['available'] is True
    assert summary['comparison_node_count'] > 0
    assert 'True' in summary['by_local_authoritative_reconciled']


def test_build_channel_surface_role_aware_lock_skips_bank_margin_pixels(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x, z in [(0.0, 2.5, 2.0), (10.0, 5.5, 1.0), (20.0, 8.5, 0.0)]:
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': z, 'z_source': 'xs_profile_resampled',
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    authmask = np.zeros((12, 12), dtype=np.float32)
    authdepth = np.full((12, 12), -9999.0, dtype=np.float32)
    authrole = np.zeros((12, 12), dtype=np.float32)
    authmask[5, 5] = 1.0
    authdepth[5, 5] = -3.5
    authrole[5, 5] = float(role_to_code(ROLE_BANK_MARGIN))
    _write_raster(river_dir / 'authmask.tif', authmask)
    _write_raster(river_dir / 'authdepth.tif', authdepth)
    _write_raster(river_dir / 'authrole.tif', authrole)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
        authoritative_support_mask_path=river_dir / 'authmask.tif',
        authoritative_support_depth_path=river_dir / 'authdepth.tif',
        authoritative_role_code_path=river_dir / 'authrole.tif',
    )
    with rasterio.open(out['channel_surface']) as ds:
        arr = ds.read(1)
    with rasterio.open(out['channel_surface_source_class']) as ds:
        src = ds.read(1)
    assert arr[5, 5] != np.float32(-3.5)
    assert int(src[5, 5]) != 4


def test_build_channel_surface_role_aware_lock_keeps_bed_pixels_hard_locked(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x, z in [(0.0, 2.5, 2.0), (10.0, 5.5, 1.0), (20.0, 8.5, 0.0)]:
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': z, 'z_source': 'xs_profile_resampled',
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, 5.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    authmask = np.zeros((12, 12), dtype=np.float32)
    authdepth = np.full((12, 12), -9999.0, dtype=np.float32)
    authrole = np.zeros((12, 12), dtype=np.float32)
    authmask[5, 5] = 1.0
    authdepth[5, 5] = -3.5
    authrole[5, 5] = float(role_to_code(ROLE_BED_INNER))
    _write_raster(river_dir / 'authmask.tif', authmask)
    _write_raster(river_dir / 'authdepth.tif', authdepth)
    _write_raster(river_dir / 'authrole.tif', authrole)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
        authoritative_support_mask_path=river_dir / 'authmask.tif',
        authoritative_support_depth_path=river_dir / 'authdepth.tif',
        authoritative_role_code_path=river_dir / 'authrole.tif',
    )
    with rasterio.open(out['channel_surface']) as ds:
        arr = ds.read(1)
    with rasterio.open(out['channel_surface_source_class']) as ds:
        src = ds.read(1)
    assert arr[5, 5] == np.float32(-3.5)
    assert int(src[5, 5]) == 4


def test_build_channel_surface_products_can_disable_xs_influence(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    for station, x, left_inner, thalweg, right_inner in [(0.0, 2.5, 7.0, 3.5, 2.5), (10.0, 5.5, 9.5, 5.0, -0.5), (20.0, 8.5, 7.0, 3.0, 2.0)]:
        role_map = {
            "left_bank": 10.0,
            "left_inner": left_inner,
            "thalweg": thalweg,
            "right_inner": right_inner,
            "right_bank": 6.0,
        }
        for role, eta in [("left_bank", 0.0), ("left_inner", 0.25), ("thalweg", 0.5), ("right_inner", 0.75), ("right_bank", 1.0)]:
            recs.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "cross_stream_eta": eta,
                "half_width_m": 2.0,
                "bed_z_m": role_map[role],
                "z_source": "xs_profile_resampled" if role in {"left_inner", "right_inner"} else "graph_backbone",
                "graph_solver_support_class": "xs_residual_only",
                "graph_hard_lock": False,
                "graph_candidate_source": "xs_profile_resampled",
                "station_support_mode": "xs_supported",
                "graph_backbone_z_m": 2.0 if role == "thalweg" else role_map[role],
                "backbone_bed_z_m": 2.0 if role == "thalweg" else role_map[role],
                "center_x": x,
                "center_y": 6.5,
                "normal_x": 0.0,
                "normal_y": 1.0,
                "geometry": Point(x, 6.5),
            })
    scaffold = gpd.GeoDataFrame(recs, crs="EPSG:4326")
    scaffold_path = river_dir / "scaffold.gpkg"
    scaffold.to_file(scaffold_path, driver="GPKG")
    corridor = np.zeros((14, 14), dtype=np.float32)
    corridor[4:10, 1:12] = 1.0
    _write_raster(river_dir / "corridor.tif", corridor)
    pd.DataFrame([{
        "profile_id": "main",
        "reach_id": "main:0",
        "reach_role": "interior",
        "station_min_m": 0.0,
        "station_max_m": 20.0,
        "unsupported_fraction": 1.0,
        "supported_fraction": 0.0,
        "authoritative_anchor_fraction": 0.0,
        "xs_support_fraction": 1.0,
        "junction_adjustment_station_fraction": 0.0,
        "component_junction_count": 0,
    }]).to_csv(river_dir / "reach.csv", index=False)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / "corridor.tif",
        reach_attributes_path=river_dir / "reach.csv",
        disable_xs_influence=True,
    )
    xs_summary = json.loads(Path(out["xs_realism_summary"]).read_text(encoding="utf-8"))
    rebuild_summary = json.loads(Path(out["primary_surface_rebuild_summary"]).read_text(encoding="utf-8"))
    contract = json.loads(Path(out["channel_surface_contract"]).read_text(encoding="utf-8"))
    assert xs_summary["disabled_by_option"] is True
    assert rebuild_summary["available"] is True
    assert rebuild_summary["adjusted_node_count"] > 0
    assert contract["metrics"]["xs_influence_disabled"] is True


def test_build_channel_surface_products_writes_authoritative_transition_receipts(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x in [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]:
        bed_by_role = {
            'left_bank': 10.0,
            'left_inner': 8.0,
            'thalweg': 4.0,
            'right_inner': 7.0,
            'right_bank': 10.0,
        }
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            y = 5.5 + (eta - 0.5) * 4.0
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': bed_by_role[role], 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'unsupported',
                'station_support_regime': 'supported_transition' if station == 10.0 else 'unsupported',
                'primary_surface_rebuild_applied': role in {'thalweg', 'left_inner', 'right_inner'},
                'station_target_present': True,
                'station_target_local_authoritative_reconciled': True,
                'authoritative_reconciliation_weight': 0.8,
                'station_authoritative_bed_support_distance_m': 25.0,
                'target_left_bank_z_m': 8.0,
                'target_right_bank_z_m': 8.0,
                'target_thalweg_z_m': 4.0,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, y),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    summary = json.loads(Path(out['channel_surface_authoritative_transition_summary']).read_text(encoding='utf-8'))
    assert summary['available'] is True
    assert summary['nonzero_cell_count'] > 0
    with rasterio.open(out['channel_surface_authoritative_transition_weight']) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
    vals = arr[np.isfinite(arr) & (arr != nodata)]
    assert vals.size > 0
    assert float(np.nanmax(vals)) > 0.0



def test_build_channel_surface_products_writes_centerline_width_propagation_receipts(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    scaffold = gpd.GeoDataFrame(_basic_surface_nodes(), crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    payload = json.loads(Path(out['centerline_width_propagation_summary']).read_text(encoding='utf-8'))
    assert payload['available'] is True
    assert payload['backbone_led_station_count'] > 0
    assert payload['bank_margin_damping_station_count'] > 0
    assert Path(out['centerline_width_propagation_profile']).exists()


def test_build_channel_surface_products_writes_role_agreement_receipts(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x in [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]:
        bed_by_role = {
            'left_bank': 10.0,
            'left_inner': 8.0,
            'thalweg': 4.0,
            'right_inner': 7.0,
            'right_bank': 10.0,
        }
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            y = 5.5 + (eta - 0.5) * 4.0
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': bed_by_role[role], 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'unsupported',
                'station_support_regime': 'unsupported',
                'primary_surface_rebuild_applied': role in {'thalweg', 'left_inner', 'right_inner'},
                'station_target_present': True,
                'station_target_local_authoritative_reconciled': role in {'thalweg', 'left_inner', 'right_inner'},
                'authoritative_reconciliation_weight': 0.8,
                'station_authoritative_bed_support_distance_m': 3.0,
                'target_left_bank_z_m': 8.0,
                'target_right_bank_z_m': 8.0,
                'target_thalweg_z_m': 4.0,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, y),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    summary = json.loads(Path(out['channel_surface_role_agreement_summary']).read_text(encoding='utf-8'))
    assert summary['available'] is True
    assert summary['comparison_node_count'] > 0
    assert summary['thalweg_agreement']['count'] > 0
    assert summary['inner_shape_agreement']['count'] > 0
    assert summary['bank_edge_agreement']['count'] > 0
    assert summary['weakest_role'] in {'thalweg', 'inner_shape', 'bank_edge'}
    assert summary['lateral_failure_mode'] in {'thalweg_fit', 'inner_shape_width_propagation', 'bank_vs_inner_shape'}


def test_build_channel_surface_products_marks_bank_edge_geometry_constraint_source(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x in [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]:
        bed_by_role = {
            'left_bank': 10.0,
            'left_inner': 8.0,
            'thalweg': 4.0,
            'right_inner': 7.0,
            'right_bank': 10.0,
        }
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            y = 5.5 + (eta - 0.5) * 4.0
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': bed_by_role[role], 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'unsupported',
                'station_support_regime': 'unsupported',
                'primary_surface_rebuild_applied': role in {'thalweg', 'left_inner', 'right_inner'},
                'station_target_present': True,
                'station_target_local_authoritative_reconciled': True,
                'authoritative_reconciliation_weight': 0.8,
                'station_authoritative_bed_support_distance_m': 1.0,
                'target_left_bank_z_m': 12.0,
                'target_right_bank_z_m': 12.0,
                'target_thalweg_z_m': 4.0,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, y),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    with rasterio.open(out['channel_surface_source_class']) as ds:
        arr = ds.read(1)
    # bank-edge rows should be marked as bank-edge geometry constraint rather than local authoritative interior control
    assert int(arr[3, 5]) == 11 or int(arr[7, 5]) == 11




def test_build_channel_surface_products_numeric_targets_drive_effective_section_presence_and_effect_receipts(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x in [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]:
        bed_by_role = {
            'left_bank': 10.0,
            'left_inner': 8.2,
            'thalweg': 4.7 if station == 10.0 else 4.4,
            'right_inner': 8.0,
            'right_bank': 10.0,
        }
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            y = 5.5 + (eta - 0.5) * 4.0
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': bed_by_role[role], 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'unsupported',
                'station_support_regime': 'unsupported',
                'primary_surface_rebuild_applied': role in {'thalweg', 'left_inner', 'right_inner'},
                'station_target_present': False,
                'station_target_local_authoritative_reconciled': False,
                'authoritative_reconciliation_weight': np.nan,
                'station_authoritative_bed_support_distance_m': 60.0,
                'target_left_bank_z_m': 8.8,
                'target_right_bank_z_m': 8.8,
                'target_thalweg_z_m': 4.0,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, y),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    effect = json.loads(Path(out['channel_surface_effect_summary']).read_text(encoding='utf-8'))
    assert effect['section_target_applied_component_count'] > 0
    profile = pd.read_csv(Path(out['channel_surface_effect_profile']))
    row = profile.iloc[0]
    assert int(row['section_target_geometry_count']) > 0
    assert int(row['section_target_candidate_count']) > 0
    assert int(row['authoritative_transition_candidate_count']) > 0
    # numeric section geometry should count as a real target candidate even when the boolean flag is false
    assert int(row['section_target_applied_count']) > 0


def test_resolve_station_support_distance_series_prefers_nearest_finite_distance():
    th = pd.DataFrame({
        'station_authoritative_bed_support_distance_m': [500.0, np.nan, 900.0],
        'authoritative_reconciliation_support_distance_m': [120.0, 35.0, np.nan],
        'profile_authoritative_bed_support_distance_m': [250.0, 40.0, 80.0],
        'component_support_median_distance_m': [1000.0, 1000.0, 1000.0],
    })
    resolved = _resolve_station_support_distance_series(th)
    assert np.allclose(resolved, np.asarray([120.0, 35.0, 80.0], dtype=float), equal_nan=False)

def test_build_channel_surface_products_transition_uses_fallback_support_distance_and_numeric_targets(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x in [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]:
        bed_by_role = {
            'left_bank': 10.0,
            'left_inner': 8.0,
            'thalweg': 4.2 if station == 10.0 else 4.0,
            'right_inner': 7.0,
            'right_bank': 10.0,
        }
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            y = 5.5 + (eta - 0.5) * 4.0
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': bed_by_role[role], 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'unsupported',
                'station_support_regime': 'unsupported',
                'primary_surface_rebuild_applied': role in {'thalweg', 'left_inner', 'right_inner'},
                'station_target_present': False,
                'station_target_local_authoritative_reconciled': False,
                'authoritative_reconciliation_weight': np.nan,
                'station_authoritative_bed_support_distance_m': np.nan,
                'authoritative_reconciliation_support_distance_m': 20.0,
                'target_left_bank_z_m': 8.0,
                'target_right_bank_z_m': 8.0,
                'target_thalweg_z_m': 4.0,
                'center_x': x, 'center_y': 5.5, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, y),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:8, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    summary = json.loads(Path(out['channel_surface_authoritative_transition_summary']).read_text(encoding='utf-8'))
    assert summary['available'] is True
    assert summary['distance_valid_cell_count'] > 0
    assert summary['inrange_cell_count'] > 0
    assert summary['candidate_cell_count'] > 0
    assert summary['presemantic_nonzero_cell_count'] > 0
    assert summary['nonzero_cell_count'] > 0
    effect = json.loads(Path(out['channel_surface_effect_summary']).read_text(encoding='utf-8'))
    assert effect['authoritative_transition_applied_component_count'] > 0


def test_apply_component_longitudinal_smoothing_moves_unsupported_interior_values():
    stations = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
    values = np.asarray([4.0, 4.1, 5.2, 4.0, 4.1, 4.0], dtype=float)
    support_distance = np.asarray([5.0, 25.0, 900.0, 900.0, 25.0, 5.0], dtype=float)
    interior_weight = np.ones_like(values, dtype=float)
    smoothed, summary = _apply_component_longitudinal_smoothing(
        stations_m=stations,
        values_z=values,
        support_distance_m=support_distance,
        interior_weight=interior_weight,
        hard_lock_mask=np.asarray([True, False, False, False, False, True], dtype=bool),
    )
    assert summary['available'] is True
    assert summary['eligible_count'] > 0
    assert summary['changed_count'] > 0
    assert smoothed[2] < values[2]
    assert smoothed[0] == values[0]
    assert smoothed[-1] == values[-1]


def test_build_channel_surface_products_writes_longitudinal_smoothing_receipts(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x, thalweg_z in [(0.0, 2.5, 4.0), (10.0, 5.5, 4.2), (20.0, 8.5, 5.4), (30.0, 11.5, 4.1), (40.0, 14.5, 4.0)]:
        bed_by_role = {
            'left_bank': 10.0,
            'left_inner': 8.0,
            'thalweg': thalweg_z,
            'right_inner': 7.5,
            'right_bank': 10.0,
        }
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            y = 8.0 + (eta - 0.5) * 4.0
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': bed_by_role[role], 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'unsupported',
                'station_support_regime': 'unsupported',
                'primary_surface_rebuild_applied': role in {'thalweg', 'left_inner', 'right_inner'},
                'station_target_present': True,
                'station_target_local_authoritative_reconciled': False,
                'authoritative_reconciliation_weight': 0.0,
                'station_authoritative_bed_support_distance_m': 800.0 if role in {'thalweg', 'left_inner', 'right_inner'} else 20.0,
                'target_left_bank_z_m': 8.5,
                'target_right_bank_z_m': 8.5,
                'target_thalweg_z_m': 4.1 if station != 20.0 else 4.3,
                'center_x': x, 'center_y': 8.0, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, y),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    corridor = np.zeros((16, 18), dtype=np.float32)
    corridor[4:13, 1:17] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
    )
    summary_path = Path(out['channel_surface_longitudinal_smoothing_summary'])
    profile_path = Path(out['channel_surface_longitudinal_smoothing_profile'])
    assert summary_path.exists()
    assert profile_path.exists()
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['available'] is True
    assert summary['component_count'] >= 1
    assert 'reason_counts' in summary


def test_backfill_node_section_target_fields_uses_thalweg_station_surfaces_for_missing_role_targets():
    recs = []
    for station, x in [(0.0, 2.5), (10.0, 5.5), (20.0, 8.5)]:
        for role, eta, bed in [('left_bank', 0.0, 10.0), ('left_inner', 0.25, 8.0), ('thalweg', 0.5, 4.0), ('right_inner', 0.75, 8.0), ('right_bank', 1.0, 10.0)]:
            y = 5.5 + (eta - 0.5) * 4.0
            row = {
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 2.0, 'bed_z_m': bed, 'station_target_present': False,
                'station_target_local_authoritative_reconciled': False,
                'authoritative_reconciliation_weight': np.nan,
                'station_authoritative_bed_support_distance_m': np.nan,
                'geometry': Point(x, y),
            }
            if role == 'thalweg':
                row.update({
                    'target_left_bank_z_m': 8.5,
                    'target_right_bank_z_m': 8.5,
                    'target_thalweg_z_m': 4.0,
                    'station_target_present': True,
                    'station_authoritative_bed_support_distance_m': 40.0,
                })
            recs.append(row)
    nodes = gpd.GeoDataFrame(recs, crs='EPSG:4326')
    th = nodes.loc[nodes['node_role'].astype(str) == 'thalweg'].sort_values('station_m').reset_index(drop=True)
    out = _backfill_node_section_target_fields(
        nodes,
        thalweg_by_comp={'main': th},
        station_target_left_bank_surfaces={'main': th['target_left_bank_z_m'].to_numpy(dtype=float)},
        station_target_right_bank_surfaces={'main': th['target_right_bank_z_m'].to_numpy(dtype=float)},
        station_target_thalweg_surfaces={'main': th['target_thalweg_z_m'].to_numpy(dtype=float)},
        station_target_present_surfaces={'main': th['station_target_present'].astype(float).to_numpy(dtype=float)},
        station_target_local_authoritative_reconciled_surfaces={'main': np.zeros(len(th), dtype=float)},
        station_authoritative_reconciliation_weight_surfaces={'main': np.full(len(th), np.nan, dtype=float)},
        station_authoritative_bed_support_distance_surfaces={'main': th['station_authoritative_bed_support_distance_m'].to_numpy(dtype=float)},
    )
    non_thalweg = out.loc[out['node_role'].astype(str) != 'thalweg'].copy()
    assert np.isfinite(pd.to_numeric(non_thalweg['target_left_bank_z_m'], errors='coerce')).all()
    assert np.isfinite(pd.to_numeric(non_thalweg['target_right_bank_z_m'], errors='coerce')).all()
    assert np.isfinite(pd.to_numeric(non_thalweg['target_thalweg_z_m'], errors='coerce')).all()
    assert out['station_target_effective_present'].astype(int).sum() == len(out)


def test_build_channel_surface_products_transition_uses_projected_pixel_distance_from_authoritative_mask(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    recs = []
    for station, x in [(0.0, 250.0), (100.0, 550.0), (200.0, 850.0)]:
        bed_by_role = {
            'left_bank': 10.0,
            'left_inner': 8.0,
            'thalweg': 4.2 if station == 100.0 else 4.0,
            'right_inner': 7.0,
            'right_bank': 10.0,
        }
        for role, eta in [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]:
            y = 550.0 + (eta - 0.5) * 200.0
            recs.append({
                'component_id': 'main', 'station_m': station, 'node_role': role, 'cross_stream_eta': eta,
                'half_width_m': 100.0, 'bed_z_m': bed_by_role[role], 'z_source': 'graph_backbone',
                'graph_solver_support_class': 'unsupported',
                'station_support_regime': 'unsupported',
                'primary_surface_rebuild_applied': role in {'thalweg', 'left_inner', 'right_inner'},
                'station_target_present': False,
                'station_target_local_authoritative_reconciled': False,
                'authoritative_reconciliation_weight': np.nan,
                'station_authoritative_bed_support_distance_m': np.nan,
                'authoritative_reconciliation_support_distance_m': np.nan,
                'target_left_bank_z_m': 8.0,
                'target_right_bank_z_m': 8.0,
                'target_thalweg_z_m': 4.0,
                'center_x': x, 'center_y': 550.0, 'normal_x': 0.0, 'normal_y': 1.0, 'geometry': Point(x, y),
            })
    scaffold = gpd.GeoDataFrame(recs, crs='EPSG:32619')
    scaffold_path = river_dir / 'scaffold.gpkg'
    scaffold.to_file(scaffold_path, driver='GPKG')
    transform = from_origin(0, 1200, 100, 100)
    corridor = np.zeros((12, 12), dtype=np.float32)
    corridor[3:9, 1:10] = 1.0
    _write_raster(river_dir / 'corridor.tif', corridor, crs='EPSG:32619', transform=transform)
    authmask = np.zeros((12, 12), dtype=np.float32)
    authmask[3:9, 1:3] = 1.0
    _write_raster(river_dir / 'authmask.tif', authmask, crs='EPSG:32619', transform=transform)
    out = build_channel_surface_products(
        river_dir=river_dir,
        channel_scaffold_nodes_path=scaffold_path,
        corridor_mask_path=river_dir / 'corridor.tif',
        authoritative_support_mask_path=river_dir / 'authmask.tif',
    )
    summary = json.loads(Path(out['channel_surface_authoritative_transition_summary']).read_text(encoding='utf-8'))
    assert summary['available'] is True
    assert summary['distance_valid_cell_count'] > 0
    assert summary['pixel_distance_assisted_cell_count'] > 0
    assert summary['candidate_cell_count'] > 0
    assert summary['presemantic_nonzero_cell_count'] > 0
    assert summary['nonzero_cell_count'] > 0





def test_role_and_section_receipts_require_canonical_active_target_when_direct_targets_missing(tmp_path: Path):
    import logging
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    nodes = pd.DataFrame({
        'component_id': ['main', 'main', 'main'],
        'station_m': [0.0, 0.0, 0.0],
        'node_role': ['left_bank', 'thalweg', 'right_bank'],
        'center_x': [0.5, 1.5, 2.5],
        'center_y': [1.5, 1.5, 1.5],
        'active_core_fit_z_m': [-2.0, -2.0, -2.0],
        'left_bank_fit_z_m': [-1.0, -1.0, -1.0],
        'right_bank_fit_z_m': [-0.8, -0.8, -0.8],
    })
    z_out = np.array([[0.0, 0.0, 0.0], [-1.0, -2.0, -0.8], [0.0, 0.0, 0.0]], dtype=float)
    transform = from_origin(0, 3, 1, 1)
    role_outputs, role_summary = _write_role_agreement_receipts(river_dir=river_dir, nodes=nodes, z_out=z_out, transform=transform, logger=logging.getLogger('test'))
    section_outputs, section_summary = _write_section_target_agreement_receipts(river_dir=river_dir, nodes=nodes, z_out=z_out, transform=transform, logger=logging.getLogger('test'))
    assert role_summary['available'] is False
    assert section_summary['available'] is False
    assert role_summary['reason'] == 'no_effective_targets'
    assert section_summary['reason'] == 'no_effective_targets'
    assert role_summary['failure_stage'] == 'effective_target_selection'
    assert section_summary['failure_stage'] == 'effective_target_selection'
    assert role_summary['target_stage_counts']['missing_canonical_active_target'] >= 1
    assert section_summary['target_stage_counts']['missing_canonical_active_target'] >= 1
    assert Path(role_outputs['channel_surface_role_agreement_summary']).exists()
    assert Path(section_outputs['channel_surface_section_target_agreement_summary']).exists()

def test_unavailable_role_and_section_receipts_still_write_files(tmp_path: Path):
    import logging
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    nodes = pd.DataFrame({
        'component_id': ['main'],
        'station_m': [0.0],
        'node_role': ['thalweg'],
        'center_x': [0.5],
        'center_y': [0.5],
        'station_target_effective_present': [False],
    })
    z_out = np.zeros((2, 2), dtype=float)
    transform = from_origin(0, 2, 1, 1)
    role_outputs, role_summary = _write_role_agreement_receipts(river_dir=river_dir, nodes=nodes, z_out=z_out, transform=transform, logger=logging.getLogger('test'))
    section_outputs, section_summary = _write_section_target_agreement_receipts(river_dir=river_dir, nodes=nodes, z_out=z_out, transform=transform, logger=logging.getLogger('test'))
    assert role_summary['available'] is False
    assert section_summary['available'] is False
    assert role_summary['reason'] == 'no_effective_targets'
    assert section_summary['reason'] == 'no_effective_targets'
    assert role_summary['failure_stage'] == 'effective_target_selection'
    assert section_summary['failure_stage'] == 'effective_target_selection'
    assert role_summary['weakest_role'] is None
    assert section_summary['weakest_role_class'] is None
    assert Path(role_outputs['channel_surface_role_agreement_summary']).exists()
    assert Path(role_outputs['channel_surface_role_agreement_profile']).exists()
    assert Path(section_outputs['channel_surface_section_target_agreement_summary']).exists()
    assert Path(section_outputs['channel_surface_section_target_agreement_profile']).exists()
