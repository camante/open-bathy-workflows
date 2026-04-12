import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from river_channel_frame import build_channel_frame_products
import river_channel_scaffold
from river_channel_scaffold import build_channel_scaffold_products, _solve_component_backbone
from authoritative_river_roles import ROLE_BANK_MARGIN


def _write_raster(path: Path, arr: np.ndarray, *, nodata: float = -9999.0):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 10, 1, 1),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr.astype(np.float32), 1)


def test_build_channel_scaffold_products(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    center = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0],
            'component_id': ['main', 'main', 'main'],
        },
        geometry=[Point(2,8), Point(5,5), Point(8,2)],
        crs='EPSG:4326',
    )
    center_path = river_dir / 'centerline_points.gpkg'
    center.to_file(center_path, driver='GPKG')
    xs_support = gpd.GeoDataFrame(
        {
            'xs_sample_source': ['authoritative', 'authoritative', 'authoritative'],
            'xs_z_m': [1.0, 0.5, 0.2],
        },
        geometry=[Point(2,8), Point(5,5), Point(8,2)],
        crs='EPSG:4326',
    )
    xs_support_path = river_dir / 'xs_support_points.gpkg'
    xs_support.to_file(xs_support_path, driver='GPKG')
    arr = np.full((10,10), -9999.0, dtype=np.float32)
    arr[2,2] = 1.0
    arr[5,5] = 0.5
    arr[8,8] = 0.2
    for name in ['bank.tif','bankinf.tif','xselev.tif','xsw.tif','centelev.tif','centinf.tif','long.tif','authmask.tif','authdepth.tif','authbed.tif']:
        a = arr.copy()
        if name=='xsw.tif':
            a[:] = -9999.0
            a[2,2] = 1.0; a[5,5] = 1.0; a[8,8] = 1.0
        if name=='authmask.tif':
            a[:] = -9999.0
            a[2,2] = 1.0; a[5,5] = 1.0; a[8,8] = 1.0
        _write_raster(river_dir/name, a)
    frame = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=center_path,
        xs_support_points_path=xs_support_path,
        bank_elevation_path=river_dir/'bank.tif',
        bank_influence_path=river_dir/'bankinf.tif',
        xs_support_elevation_path=river_dir/'xselev.tif',
        xs_support_weight_path=river_dir/'xsw.tif',
        centerline_elevation_path=river_dir/'centelev.tif',
        centerline_influence_path=river_dir/'centinf.tif',
        longitudinal_profile_elevation_path=river_dir/'long.tif',
        authoritative_support_mask_path=river_dir/'authmask.tif',
        authoritative_support_depth_path=river_dir/'authdepth.tif',
        authoritative_bed_elevation_path=river_dir/'authbed.tif',
    )
    xs_bathy = gpd.GeoDataFrame(
        {
            'xs_id': ['A','A','A','B','B','B'],
            'width_m': [10.0]*6,
            'dist_m': [0.0,5.0,10.0,0.0,5.0,10.0],
            'z_bed_pred_m': [2.0,1.0,2.0,1.8,0.8,1.8],
        },
        geometry=[Point(2,8),Point(2,8),Point(2,8),Point(8,2),Point(8,2),Point(8,2)],
        crs='EPSG:4326',
    )
    xs_bathy_path = tmp_path / 'river_bathy_xs_mainstem.gpkg'
    xs_bathy.to_file(xs_bathy_path, layer='xs_bathy_points', driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame['channel_frame_points'],
        xs_bathy_gpkg_path=xs_bathy_path,
    )
    assert Path(out['channel_scaffold_nodes']).exists()
    assert Path(frame['measured_xs_handoff_receipt']).exists()
    assert Path(out['measured_xs_scaffold_handoff_receipt']).exists()
    nodes = gpd.read_file(out['channel_scaffold_nodes'])
    assert not nodes.empty
    assert {'left_bank','left_inner','thalweg','right_inner','right_bank'} <= set(nodes['node_role'].astype(str))
    assert np.count_nonzero(nodes['z_source'].astype(str).eq('authoritative_in_channel')) >= 1
    frame_receipt = json.loads(Path(frame['measured_xs_handoff_receipt']).read_text())
    scaffold_receipt = json.loads(Path(out['measured_xs_scaffold_handoff_receipt']).read_text())
    assert frame_receipt['authoritative_xs_support_samples_found'] == 3
    assert frame_receipt['stations_with_true_measured_xs_candidates'] >= 1
    assert frame_receipt['stations_with_true_measured_xs_qualified'] >= 1
    assert scaffold_receipt['stations_with_true_measured_xs_candidates'] >= 1
    assert scaffold_receipt['stations_with_true_measured_xs_qualified'] >= 1
    assert 'loss_reason_counts' in scaffold_receipt
    assert 'qualified_loss_reason_counts' in scaffold_receipt
    assert scaffold_receipt['measured_xs_activation_from_bank_only_forbidden'] is True


def test_scaffold_reconciles_backbone_near_component_junction(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0, 30.0, 0.0, 10.0, 20.0, 30.0],
            'component_id': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'backbone_z_m': [0.0, np.nan, np.nan, 0.0, 10.0, np.nan, np.nan, 10.0],
            'backbone_mode': ['resolved_backbone'] * 8,
            'resolved_stage_control_z_m': [0.0, np.nan, np.nan, 0.0, 10.0, np.nan, np.nan, 10.0],
            'resolved_channel_bed_z_m': [np.nan] * 8,
            'authoritative_bed_z_m': [np.nan] * 8,
            'authoritative_backbone_z_m': [np.nan] * 8,
            'authoritative_station_support_strength': [0.0] * 8,
            'xs_residual_to_backbone_z_m': [0.0] * 8,
            'channel_support_class': ['unsupported'] * 8,
        },
        geometry=[
            Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0),
            Point(3.05, 0.05), Point(4, 0.05), Point(5, 0.05), Point(6, 0.05),
        ],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    nodes = gpd.read_file(out['channel_scaffold_nodes'])
    thalweg = nodes.loc[nodes['node_role'].astype(str).eq('thalweg')].copy()
    a_near = float(thalweg.loc[(thalweg['component_id'] == 'A') & np.isclose(thalweg['station_m'], 20.0), 'backbone_bed_z_m'].iloc[0])
    b_near = float(thalweg.loc[(thalweg['component_id'] == 'B') & np.isclose(thalweg['station_m'], 10.0), 'backbone_bed_z_m'].iloc[0])
    assert a_near > 0.0 and a_near < 10.0
    assert b_near > 0.0 and b_near < 10.0
    assert not np.isclose(a_near, 0.0)
    assert not np.isclose(b_near, 10.0)
    contract = json.loads((river_dir / 'river_channel_scaffold_contract.json').read_text())
    assert int(contract['metrics']['junction_group_count']) >= 1
    assert int(contract['metrics']['junction_adjusted_component_count']) >= 1
    assert int(contract['metrics']['dominant_junction_group_count']) == 0


def test_scaffold_confluence_prefers_dominant_component_backbone(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0, 30.0, 0.0, 10.0, 20.0],
            'component_id': ['A', 'A', 'A', 'A', 'B', 'B', 'B'],
            'backbone_z_m': [0.0, 0.0, 0.0, 0.0, 10.0, np.nan, 10.0],
            'backbone_mode': ['authoritative_in_channel', 'authoritative_backbone', 'authoritative_backbone', 'authoritative_backbone', 'centerline_bed', 'centerline_bed', 'centerline_bed'],
            'resolved_stage_control_z_m': [0.0, 0.0, 0.0, 0.0, 10.0, np.nan, 10.0],
            'resolved_channel_bed_z_m': [np.nan] * 7,
            'authoritative_bed_z_m': [0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'authoritative_backbone_z_m': [0.0, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan],
            'authoritative_station_support_strength': [5.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            'xs_residual_to_backbone_z_m': [0.0] * 7,
            'channel_support_class': ['authoritative_in_channel'] * 4 + ['unsupported'] * 3,
        },
        geometry=[
            Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0),
            Point(3.02, 0.02), Point(4, 0.02), Point(5, 0.02),
        ],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    nodes = gpd.read_file(out['channel_scaffold_nodes'])
    thalweg = nodes.loc[nodes['node_role'].astype(str).eq('thalweg')].copy()
    b_near = float(thalweg.loc[(thalweg['component_id'] == 'B') & np.isclose(thalweg['station_m'], 0.0), 'backbone_bed_z_m'].iloc[0])
    a_end = float(thalweg.loc[(thalweg['component_id'] == 'A') & np.isclose(thalweg['station_m'], 30.0), 'backbone_bed_z_m'].iloc[0])
    assert b_near < 5.0
    assert np.isclose(a_end, 0.0)
    contract = json.loads((river_dir / 'river_channel_scaffold_contract.json').read_text())
    assert int(contract['metrics']['dominant_junction_group_count']) >= 1
    assert int(contract['metrics']['dominant_preserved_component_count']) >= 1


def test_scaffold_ignores_resolved_channel_bed_in_structured_contract(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    base = {
        'station_m': [0.0, 10.0, 20.0],
        'component_id': ['main', 'main', 'main'],
        'backbone_z_m': [5.0, 4.0, 3.0],
        'backbone_mode': ['resolved_backbone'] * 3,
        'resolved_stage_control_z_m': [np.nan, np.nan, np.nan],
        'authoritative_bed_z_m': [np.nan, np.nan, np.nan],
        'authoritative_backbone_z_m': [np.nan, np.nan, np.nan],
        'authoritative_station_support_strength': [0.0, 0.0, 0.0],
        'xs_residual_to_backbone_z_m': [0.0, -0.5, 0.25],
        'channel_support_class': ['unsupported'] * 3,
    }
    geom = [Point(0, 0), Point(1, 0), Point(2, 0)]

    frame_a = gpd.GeoDataFrame({**base, 'resolved_channel_bed_z_m': [-100.0, -200.0, -300.0]}, geometry=geom, crs='EPSG:4326')
    frame_b = gpd.GeoDataFrame({**base, 'resolved_channel_bed_z_m': [1000.0, 2000.0, 3000.0]}, geometry=geom, crs='EPSG:4326')

    frame_a_path = river_dir / 'frame_a.gpkg'
    frame_b_path = river_dir / 'frame_b.gpkg'
    frame_a.to_file(frame_a_path, driver='GPKG')
    frame_b.to_file(frame_b_path, driver='GPKG')

    (river_dir / 'a').mkdir()
    (river_dir / 'b').mkdir()

    out_a = build_channel_scaffold_products(
        river_dir=river_dir / 'a',
        channel_frame_points_path=frame_a_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    out_b = build_channel_scaffold_products(
        river_dir=river_dir / 'b',
        channel_frame_points_path=frame_b_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )

    nodes_a = gpd.read_file(out_a['channel_scaffold_nodes']).sort_values(['component_id', 'station_m', 'node_role']).reset_index(drop=True)
    nodes_b = gpd.read_file(out_b['channel_scaffold_nodes']).sort_values(['component_id', 'station_m', 'node_role']).reset_index(drop=True)

    assert np.allclose(nodes_a['bed_z_m'].to_numpy(dtype=float), nodes_b['bed_z_m'].to_numpy(dtype=float), equal_nan=True)
    assert list(nodes_a['z_source'].astype(str)) == list(nodes_b['z_source'].astype(str))
    contract = json.loads(((river_dir / 'a') / 'river_channel_scaffold_contract.json').read_text())
    assert contract['metrics']['absolute_bed_fallback_node_count'] == 0
    assert contract['metrics']['absolute_bed_fallback_suppressed_node_count'] == 0
    assert contract['metrics']['legacy_mixed_bed_used'] is False
    assert contract['metrics']['legacy_mixed_bed_field_ignored'] is True


def test_scaffold_prefers_explicit_junction_groups_over_geometric_pairing(tmp_path: Path):
    geom = [Point(0, 0), Point(10, 0), Point(10.1, 0.0), Point(20, 0), Point(10.0, 0.1), Point(20, 0.1)]
    frame = gpd.GeoDataFrame(
        {
            'component_id': ['main', 'main', 'trib_a', 'trib_a', 'trib_b', 'trib_b'],
            'station_m': [0.0, 10.0, 0.0, 10.0, 0.0, 10.0],
            'authoritative_bed_z_m': [np.nan, 0.0, np.nan, 5.0, np.nan, 6.0],
            'authoritative_backbone_z_m': [np.nan, 0.0, np.nan, np.nan, np.nan, np.nan],
            'backbone_z_m': [0.0, 0.0, 5.0, 5.0, 6.0, 6.0],
            'backbone_mode': ['authoritative_backbone', 'authoritative_in_channel', 'longitudinal_profile', 'longitudinal_profile', 'longitudinal_profile', 'longitudinal_profile'],
            'resolved_stage_control_z_m': [np.nan] * 6,
            'xs_residual_to_backbone_z_m': [np.nan] * 6,
            'authoritative_station_support_strength': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            'channel_support_class': ['authoritative_in_channel', 'authoritative_in_channel', 'unsupported', 'unsupported', 'unsupported', 'unsupported'],
            'junction_id': ['j_main', 'j_shared', 'j_shared', 'j_far', 'j_shared', 'j_far2'],
        },
        geometry=geom, crs='EPSG:4326'
    )
    frame_path = tmp_path / 'frame_junction.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    outputs = build_channel_scaffold_products(
        river_dir=tmp_path / 'river',
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        authoritative_support_mask_path=None,
        authoritative_support_depth_path=None,
        allow_absolute_bed_fallback=False,
    )
    with open(outputs['channel_scaffold_contract'], 'r', encoding='utf-8') as f:
        contract = json.load(f)
    assert contract['metrics']['topology_guided_junction_count'] >= 1


def test_component_station_candidates_preserve_non_authoritative_backbone_classes():
    from river_channel_scaffold import _build_component_station_candidates

    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0, 30.0],
            'component_id': ['main'] * 4,
            'authoritative_bed_z_m': [np.nan] * 4,
            'authoritative_backbone_z_m': [np.nan, np.nan, 3.0, np.nan],
            'backbone_z_m': [10.0, 9.0, 3.0, 8.0],
            'backbone_mode': ['longitudinal_profile', 'centerline_bed', 'authoritative_backbone', 'longitudinal_profile'],
            'resolved_stage_control_z_m': [np.nan, np.nan, np.nan, 7.5],
            'authoritative_station_support_strength': [0.0, 2.0, 0.0, 0.0],
            'channel_support_class': ['unsupported', 'unsupported', 'unsupported', 'unsupported'],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)],
        crs='EPSG:4326',
    )

    cand = _build_component_station_candidates(frame)
    assert cand['solver_support_class'].tolist() == [
        'resolved_backbone',
        'anchored_interpolated',
        'authoritative_locked',
        'resolved_backbone',
    ]
    assert cand['solver_candidate_source'].tolist() == [
        'longitudinal_profile',
        'centerline_bed',
        'authoritative_backbone',
        'longitudinal_profile',
    ]


def test_scaffold_combines_explicit_and_geometric_junction_grouping(tmp_path: Path):
    geom = [
        Point(0, 0), Point(10, 0),               # main explicit group
        Point(10.0, 0.05), Point(20, 0.05),      # trib explicit group
        Point(50, 0), Point(60, 0),              # unlabeled pair A
        Point(60.05, 0.05), Point(70, 0.05),     # unlabeled pair B
    ]
    frame = gpd.GeoDataFrame(
        {
            'component_id': ['main', 'main', 'trib_exp', 'trib_exp', 'u1', 'u1', 'u2', 'u2'],
            'station_m': [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0],
            'authoritative_bed_z_m': [np.nan, 0.0, np.nan, 5.0, np.nan, 20.0, np.nan, 25.0],
            'authoritative_backbone_z_m': [np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'backbone_z_m': [0.0, 0.0, 5.0, 5.0, 20.0, 20.0, 25.0, 25.0],
            'backbone_mode': ['authoritative_backbone', 'authoritative_in_channel', 'longitudinal_profile', 'longitudinal_profile', 'centerline_bed', 'centerline_bed', 'centerline_bed', 'centerline_bed'],
            'resolved_stage_control_z_m': [np.nan] * 8,
            'xs_residual_to_backbone_z_m': [np.nan] * 8,
            'authoritative_station_support_strength': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'channel_support_class': ['authoritative_in_channel', 'authoritative_in_channel', 'unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported'],
            'junction_id': ['j_ignore', 'j_shared', 'j_shared', 'j_far', np.nan, np.nan, np.nan, np.nan],
        },
        geometry=geom, crs='EPSG:4326'
    )
    frame_path = tmp_path / 'frame_combined_junction.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    outputs = build_channel_scaffold_products(
        river_dir=tmp_path / 'river',
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        authoritative_support_mask_path=None,
        authoritative_support_depth_path=None,
        allow_absolute_bed_fallback=False,
    )
    contract = json.loads(Path(outputs['channel_scaffold_contract']).read_text())
    assert contract['metrics']['topology_guided_junction_count'] >= 1
    assert contract['metrics']['geometric_fallback_junction_count'] >= 1
    assert contract['metrics']['junction_group_count'] >= 2


def test_build_channel_scaffold_does_not_mark_authoritative_without_authoritative_base(tmp_path: Path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    frame = gpd.GeoDataFrame(
        {
            'component_id': ['A'],
            'station_m': [0.0],
            'authoritative_bed_z_m': [float('nan')],
            'authoritative_backbone_z_m': [float('nan')],
            'backbone_z_m': [-5.0],
            'backbone_mode': ['resolved_backbone'],
            'xs_residual_to_backbone_z_m': [0.0],
            'resolved_stage_control_z_m': [float('nan')],
            'authoritative_station_support_strength': [1.0],
            'channel_support_class': ['anchored_interpolated'],
        },
        geometry=[Point(0, 0)],
        crs='EPSG:4326',
    )
    frame_path = tmp_path / 'frame.gpkg'
    frame.to_file(frame_path, driver='GPKG')

    outputs = build_channel_scaffold_products(
        river_dir=tmp_path,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        authoritative_support_mask_path=None,
        authoritative_support_depth_path=None,
        allow_absolute_bed_fallback=False,
        logger=None,
    )
    nodes = gpd.read_file(outputs['channel_scaffold_nodes'])
    assert set(nodes['station_support_mode'].astype(str)) == {'resolved_backbone'}
    assert not nodes['graph_backbone_missing'].any()
    assert set(nodes['z_source'].astype(str)) == {'graph_backbone', 'generalized_thalweg_default_tendency'}



def test_build_junction_groups_keeps_uncovered_endpoints_for_geometric_fallback():
    import geopandas as gpd
    from shapely.geometry import Point
    from river_channel_scaffold import _build_junction_groups

    frame = gpd.GeoDataFrame(
        {
            "component_id": ["A", "A", "B", "B", "C", "C"],
            "station_m": [0.0, 10.0, 0.0, 10.0, 0.0, 10.0],
            "junction_id": [None, "J1", "J1", None, None, None],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(10.0, 1.0),
            Point(20.0, 1.0),
            Point(20.0, 2.0),
            Point(30.0, 2.0),
        ],
        crs="EPSG:32619",
    )
    component_fill = {
        "A": np.asarray([0.0, 0.0], dtype=np.float32),
        "B": np.asarray([1.0, 1.0], dtype=np.float32),
        "C": np.asarray([2.0, 2.0], dtype=np.float32),
    }

    groups, _ = _build_junction_groups(frame, component_fill)
    topo = [g for g in groups if g.get("topology_source") == "explicit_junction_id"]
    geom = [g for g in groups if g.get("topology_source") == "geometric_fallback"]

    assert len(topo) == 1
    assert {r["component_id"] for r in topo[0]["records"]} == {"A", "B"}
    assert any({r["component_id"] for r in g["records"]} == {"B", "C"} for g in geom)


def test_component_backbone_solver_preserves_hard_locks_and_bridges_gap():
    import pandas as pd
    sub = pd.DataFrame({
        'station_m': [0.0, 10.0, 20.0, 30.0, 40.0],
        'authoritative_bed_z_m': [10.0, np.nan, np.nan, np.nan, 0.0],
        'authoritative_backbone_z_m': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'backbone_z_m': [10.0, np.nan, np.nan, np.nan, 0.0],
        'resolved_stage_control_z_m': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'authoritative_station_support_strength': [5.0, 0.0, 0.0, 0.0, 5.0],
        'backbone_mode': ['authoritative_in_channel', 'missing', 'missing', 'missing', 'authoritative_in_channel'],
        'channel_support_class': ['authoritative_in_channel', 'unsupported', 'unsupported', 'unsupported', 'authoritative_in_channel'],
    })
    solved, metrics = _solve_component_backbone(sub)
    assert np.isclose(float(solved[0]), 10.0)
    assert np.isclose(float(solved[-1]), 0.0)
    assert np.all(np.isfinite(solved[1:-1]))
    assert float(solved[1]) < 10.0 and float(solved[1]) > 0.0
    assert float(solved[2]) < float(solved[1])
    assert float(solved[3]) < float(solved[2])
    assert int(metrics['component_hard_lock_count']) == 2


def test_component_backbone_solver_respects_stage_control_without_promoting_to_lock():
    import pandas as pd
    sub = pd.DataFrame({
        'station_m': [0.0, 10.0, 20.0],
        'authoritative_bed_z_m': [np.nan, np.nan, np.nan],
        'authoritative_backbone_z_m': [np.nan, np.nan, np.nan],
        'backbone_z_m': [2.0, np.nan, 0.0],
        'resolved_stage_control_z_m': [np.nan, 1.0, np.nan],
        'authoritative_station_support_strength': [0.0, 0.0, 0.0],
        'backbone_mode': ['resolved_backbone', 'missing', 'resolved_backbone'],
        'channel_support_class': ['unsupported', 'unsupported', 'unsupported'],
    })
    solved, metrics = _solve_component_backbone(sub)
    assert np.all(np.isfinite(solved))
    assert float(solved[1]) > 0.0 and float(solved[1]) < 2.0
    assert int(metrics['component_stage_controlled_count']) == 1


def test_network_backbone_solver_reports_graph_junction_constraints(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0, 30.0, 0.0, 10.0, 20.0],
            'component_id': ['A', 'A', 'A', 'A', 'B', 'B', 'B'],
            'backbone_z_m': [0.0, 0.0, 0.0, 0.0, 10.0, np.nan, 10.0],
            'backbone_mode': ['authoritative_in_channel', 'authoritative_backbone', 'authoritative_backbone', 'authoritative_backbone', 'centerline_bed', 'centerline_bed', 'centerline_bed'],
            'resolved_stage_control_z_m': [0.0, 0.0, 0.0, 0.0, 10.0, np.nan, 10.0],
            'authoritative_bed_z_m': [0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'authoritative_backbone_z_m': [0.0, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan],
            'authoritative_station_support_strength': [5.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            'xs_residual_to_backbone_z_m': [0.0] * 7,
            'channel_support_class': ['authoritative_in_channel'] * 4 + ['unsupported'] * 3,
        },
        geometry=[
            Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0),
            Point(3.02, 0.02), Point(4, 0.02), Point(5, 0.02),
        ],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    contract = json.loads((river_dir / 'river_channel_scaffold_contract.json').read_text())
    assert int(contract['metrics']['graph_junction_constraint_count']) > 0


def test_network_backbone_solver_support_aware_junction_pull(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0, 0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
            'component_id': ['A','A','A','B','B','B','C','C','C'],
            'backbone_z_m': [0.0, 0.0, 0.0, 8.0, 8.0, 8.0, 12.0, 12.0, 12.0],
            'backbone_mode': ['authoritative_backbone']*3 + ['resolved_backbone']*3 + ['resolved_backbone']*3,
            'resolved_stage_control_z_m': [np.nan]*9,
            'authoritative_bed_z_m': [0.0, np.nan, np.nan] + [np.nan]*6,
            'authoritative_backbone_z_m': [0.0, 0.0, 0.0] + [np.nan]*6,
            'authoritative_station_support_strength': [3.0,2.0,2.0, 0.0,0.0,0.0, 1.0,1.0,1.0],
            'xs_residual_to_backbone_z_m': [0.0]*9,
            'channel_support_class': ['authoritative_in_channel']*3 + ['unsupported']*3 + ['anchored_interpolated']*3,
            'junction_id': ['j1','j1','farA','j1','farB','farB2','j1','farC','farC2'],
        },
        geometry=[
            Point(0,0), Point(1,0), Point(2,0),
            Point(0.02,0.02), Point(1,0.02), Point(2,0.02),
            Point(-0.02,0.02), Point(-1,0.02), Point(-2,0.02),
        ],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    nodes = gpd.read_file(out['channel_scaffold_nodes'])
    thalweg = nodes.loc[nodes['node_role'].astype(str).eq('thalweg')].copy()
    b0 = float(thalweg.loc[(thalweg['component_id'] == 'B') & np.isclose(thalweg['station_m'], 0.0), 'backbone_bed_z_m'].iloc[0])
    c0 = float(thalweg.loc[(thalweg['component_id'] == 'C') & np.isclose(thalweg['station_m'], 0.0), 'backbone_bed_z_m'].iloc[0])
    # Unsupported branch B should be pulled harder toward dominant A than anchored branch C.
    assert abs(b0 - 0.0) < abs(c0 - 0.0)
    contract = json.loads((river_dir / 'river_channel_scaffold_contract.json').read_text())
    assert int(contract['metrics']['graph_junction_constraint_count']) > 0


def test_scaffold_writes_graph_backbone_diagnostics(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0, 0.0, 10.0],
            'component_id': ['A', 'A', 'A', 'B', 'B'],
            'backbone_z_m': [0.0, np.nan, 0.0, 8.0, 8.0],
            'backbone_mode': ['authoritative_backbone', 'resolved_backbone', 'authoritative_backbone', 'resolved_backbone', 'resolved_backbone'],
            'resolved_stage_control_z_m': [0.0, 0.5, 0.0, np.nan, np.nan],
            'authoritative_bed_z_m': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'authoritative_backbone_z_m': [0.0, np.nan, 0.0, np.nan, np.nan],
            'authoritative_station_support_strength': [2.0, 0.0, 2.0, 0.0, 0.0],
            'xs_residual_to_backbone_z_m': [0.0] * 5,
            'channel_support_class': ['authoritative_in_channel', 'unsupported', 'authoritative_in_channel', 'unsupported', 'unsupported'],
            'junction_id': ['j1', 'j1', 'j1', 'j1', 'j1'],
        },
        geometry=[Point(0,0), Point(1,0), Point(2,0), Point(2.02,0.02), Point(3,0.02)],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    contract = json.loads(Path(out['channel_scaffold_contract']).read_text())
    diag_path = Path(contract['artifacts']['graph_backbone_diagnostics'])
    assert diag_path.exists()
    diag = gpd.read_file(diag_path)
    assert not diag.empty
    required = {
        'graph_backbone_z_m', 'graph_hard_lock', 'graph_prior_weight_sum',
        'graph_regularization_weight_sum', 'graph_junction_weight_sum',
        'graph_junction_constrained', 'graph_solution_mode', 'graph_unsupported_span_m'
    }
    assert required <= set(diag.columns)
    assert int(contract['metrics']['graph_hard_lock_station_count']) >= 1
    assert 'graph_solution_mode_counts' in contract['metrics']


def test_scaffold_realizes_graph_backbone_instead_of_frame_backbone_modes(tmp_path: Path, monkeypatch):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 20.0],
            "component_id": ["main", "main", "main"],
            "backbone_z_m": [100.0, 100.0, 100.0],
            "backbone_mode": ["bank_stage_prior", "bank_stage_prior", "bank_stage_prior"],
            "resolved_stage_control_z_m": [9.0, 9.0, 9.0],
            "authoritative_bed_z_m": [np.nan, np.nan, np.nan],
            "authoritative_backbone_z_m": [np.nan, np.nan, np.nan],
            "authoritative_station_support_strength": [0.0, 0.0, 0.0],
            "xs_residual_to_backbone_z_m": [0.0, 0.0, 0.0],
            "channel_support_class": ["unsupported", "unsupported", "unsupported"],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs="EPSG:4326",
    )
    frame_path = river_dir / "channel_frame_points.gpkg"
    frame.to_file(frame_path, driver="GPKG")

    def _fake_solve_network_backbone(frame_df):
        arr = np.asarray([7.0, 6.0, 5.0], dtype=np.float32)
        diag = pd.DataFrame(
            {
                "component_id": ["main", "main", "main"],
                "station_m": [0.0, 10.0, 20.0],
                "graph_backbone_z_m": arr,
                "graph_hard_lock": [False, False, False],
                "graph_prior_weight_sum": [1.0, 1.0, 1.0],
                "graph_edge_weight_sum": [0.5, 0.5, 0.5],
                "graph_curvature_weight_sum": [0.25, 0.25, 0.25],
                "graph_centering_weight_sum": [0.01, 0.01, 0.01],
                "graph_regularization_weight_sum": [0.76, 0.76, 0.76],
                "graph_junction_weight_sum": [0.0, 0.0, 0.0],
                "graph_junction_constrained": [False, False, False],
                "graph_residual_to_candidate_z_m": [-93.0, -94.0, -95.0],
                "graph_solver_support_class": ["resolved_backbone", "resolved_backbone", "resolved_backbone"],
                "graph_candidate_source": ["centerline_bed", "centerline_bed", "centerline_bed"],
                "graph_solution_mode": ["prior_driven", "prior_driven", "prior_driven"],
                "graph_unsupported_span_m": [20.0, 20.0, 20.0],
            }
        )
        return {"main": arr}, {"graph_junction_constraint_count": 0}, diag

    monkeypatch.setattr("river_channel_scaffold._solve_network_backbone", _fake_solve_network_backbone)

    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    nodes = gpd.read_file(out["channel_scaffold_nodes"])
    thalweg = nodes.loc[nodes["node_role"].astype(str).eq("thalweg")].sort_values("station_m").reset_index(drop=True)
    assert np.allclose(thalweg["backbone_bed_z_m"].to_numpy(dtype=float), np.asarray([7.0, 6.0, 5.0], dtype=float))
    assert set(thalweg["z_source"].astype(str)) == {"graph_backbone"}
    assert set(thalweg["station_support_mode"].astype(str)) == {"resolved_backbone"}
    assert not thalweg["graph_backbone_missing"].any()
    assert set(thalweg["graph_candidate_source"].astype(str)) == {"centerline_bed"}


def test_graph_solver_adds_physical_guard_diagnostics_for_unsupported_span(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0, 30.0, 40.0],
            'component_id': ['main'] * 5,
            'backbone_z_m': [0.0, np.nan, np.nan, np.nan, 0.0],
            'backbone_mode': ['authoritative_in_channel', 'missing', 'missing', 'missing', 'authoritative_backbone'],
            'resolved_stage_control_z_m': [0.0, np.nan, np.nan, np.nan, 0.0],
            'authoritative_bed_z_m': [0.0, np.nan, np.nan, np.nan, np.nan],
            'authoritative_backbone_z_m': [0.0, np.nan, np.nan, np.nan, 0.0],
            'authoritative_station_support_strength': [2.0, 0.0, 0.0, 0.0, 2.0],
            'xs_residual_to_backbone_z_m': [0.0] * 5,
            'channel_support_class': ['authoritative_in_channel', 'unsupported', 'unsupported', 'unsupported', 'authoritative_backbone'],
        },
        geometry=[Point(x, 0) for x in range(5)],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    contract = json.loads((river_dir / 'river_channel_scaffold_contract.json').read_text())
    diag = gpd.read_file(contract['artifacts']['graph_backbone_diagnostics'])
    mid = diag.loc[np.isclose(diag['station_m'], 20.0)].iloc[0]
    assert float(mid['graph_physical_guard_weight_sum']) > 0.0
    assert float(mid['graph_slope_guard_weight_sum']) > 0.0
    assert float(mid['graph_unsupported_span_m']) > 0.0
    assert bool(mid['graph_slope_guard_active'])


def test_graph_solver_suppresses_unsupported_spike_without_moving_locks():
    sub = pd.DataFrame(
        {
            'station_m': [0.0, 10.0, 20.0, 30.0, 40.0],
            'component_id': ['main'] * 5,
            'backbone_z_m': [0.0, np.nan, 8.0, np.nan, 0.0],
            'backbone_mode': ['authoritative_in_channel', 'missing', 'centerline_bed', 'missing', 'authoritative_backbone'],
            'resolved_stage_control_z_m': [0.0, np.nan, np.nan, np.nan, 0.0],
            'authoritative_bed_z_m': [0.0, np.nan, np.nan, np.nan, np.nan],
            'authoritative_backbone_z_m': [0.0, np.nan, np.nan, np.nan, 0.0],
            'authoritative_station_support_strength': [2.0, 0.0, 0.0, 0.0, 2.0],
            'xs_residual_to_backbone_z_m': [0.0] * 5,
            'channel_support_class': ['authoritative_in_channel', 'unsupported', 'unsupported', 'unsupported', 'authoritative_backbone'],
        }
    )
    solved, metrics = _solve_component_backbone(sub)
    assert np.isclose(float(solved[0]), 0.0)
    assert np.isclose(float(solved[-1]), 0.0)
    assert abs(float(solved[2])) < 8.0
    assert metrics['component_hard_lock_count'] == 2


def test_scaffold_writes_graph_physical_plausibility_contract(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0, 30.0, 40.0],
            'component_id': ['main'] * 5,
            'backbone_z_m': [0.0, np.nan, np.nan, np.nan, 0.0],
            'backbone_mode': ['authoritative_in_channel', 'missing', 'missing', 'missing', 'authoritative_backbone'],
            'resolved_stage_control_z_m': [0.0, np.nan, np.nan, np.nan, 0.0],
            'authoritative_bed_z_m': [0.0, np.nan, np.nan, np.nan, np.nan],
            'authoritative_backbone_z_m': [0.0, np.nan, np.nan, np.nan, 0.0],
            'authoritative_station_support_strength': [2.0, 0.0, 0.0, 0.0, 2.0],
            'xs_residual_to_backbone_z_m': [0.0] * 5,
            'channel_support_class': ['authoritative_in_channel', 'unsupported', 'unsupported', 'unsupported', 'authoritative_backbone'],
        },
        geometry=[Point(float(x), 0.0) for x in range(5)],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    physical = Path(out['graph_physical_plausibility_contract'])
    assert physical.exists()
    summary = json.loads(physical.read_text())
    assert summary['metrics']['graph_slope_guard_station_count'] > 0
    assert summary['metrics']['graph_physical_guard_station_count'] > 0
    assert summary['metrics']['graph_p95_unsupported_span_m'] > 0.0
    scaffold_contract = json.loads((river_dir / 'river_channel_scaffold_contract.json').read_text())
    assert scaffold_contract['artifacts']['graph_physical_plausibility_contract'].endswith('river_graph_physical_plausibility_contract.json')



def test_graph_solver_junction_physical_regularization_avoids_tributary_cliff(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
            'component_id': ['main', 'main', 'main', 'trib', 'trib', 'trib'],
            'backbone_z_m': [0.0, 0.0, 0.0, 0.0, 8.0, np.nan],
            'backbone_mode': [
                'authoritative_in_channel', 'authoritative_backbone', 'authoritative_backbone',
                'stage_controlled', 'centerline_bed', 'missing'
            ],
            'resolved_stage_control_z_m': [0.0, 0.0, 0.0, 0.0, np.nan, np.nan],
            'authoritative_bed_z_m': [0.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            'authoritative_backbone_z_m': [0.0, 0.0, 0.0, np.nan, np.nan, np.nan],
            'authoritative_station_support_strength': [2.0, 2.0, 2.0, 0.3, 0.0, 0.0],
            'xs_residual_to_backbone_z_m': [0.0] * 6,
            'channel_support_class': [
                'authoritative_in_channel', 'authoritative_backbone', 'authoritative_backbone',
                'stage_controlled', 'unsupported', 'unsupported'
            ],
            'junction_id': ['j1', 'j1', 'j1', 'j1', 'j1', 'j1'],
            'mainstem_rank': [1, 1, 1, 3, 3, 3],
            'network_order': [4, 4, 4, 2, 2, 2],
            'distance_to_mouth_m': [100.0, 90.0, 80.0, 100.0, 110.0, 120.0],
        },
        geometry=[Point(0, 0), Point(10, 0), Point(20, 0), Point(20, 0), Point(20, 10), Point(20, 20)],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    diag = gpd.read_file(river_dir / 'river_graph_backbone_diagnostics.gpkg')
    main_j = diag.loc[(diag['component_id'] == 'main') & np.isclose(diag['station_m'], 20.0)].iloc[0]
    trib_j = diag.loc[(diag['component_id'] == 'trib') & np.isclose(diag['station_m'], 0.0)].iloc[0]
    trib_mid = diag.loc[(diag['component_id'] == 'trib') & np.isclose(diag['station_m'], 10.0)].iloc[0]
    assert bool(trib_mid['graph_slope_guard_active']) or bool(trib_mid['graph_adverse_step_guard_active'])
    assert abs(float(trib_j['graph_backbone_z_m']) - float(main_j['graph_backbone_z_m'])) < 2.5
    assert abs(float(trib_mid['graph_backbone_z_m'])) < 8.0


def test_scaffold_writes_junction_diagnostics_contract_artifact(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'component_id': ['A', 'A', 'B', 'B'],
            'station_m': [0.0, 10.0, 0.0, 10.0],
            'authoritative_bed_z_m': [np.nan, 0.0, np.nan, np.nan],
            'authoritative_backbone_z_m': [np.nan, 0.0, np.nan, np.nan],
            'backbone_z_m': [0.0, 0.0, 4.0, 4.0],
            'backbone_mode': ['authoritative_backbone', 'authoritative_in_channel', 'longitudinal_profile', 'longitudinal_profile'],
            'resolved_stage_control_z_m': [np.nan, np.nan, np.nan, np.nan],
            'xs_residual_to_backbone_z_m': [0.0, 0.0, 0.0, 0.0],
            'authoritative_station_support_strength': [0.0, 1.0, 0.0, 0.0],
            'channel_support_class': ['authoritative_in_channel', 'authoritative_in_channel', 'unsupported', 'unsupported'],
            'junction_id': ['j1', 'j1', 'j1', 'j1'],
        },
        geometry=[Point(0, 0), Point(10, 0), Point(10, 0.1), Point(20, 0.1)],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'frame.gpkg'
    frame.to_file(frame_path, driver='GPKG')

    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    contract = json.loads(Path(out['channel_scaffold_contract']).read_text())
    assert contract['artifacts']['junction_diagnostics'].endswith('river_junction_diagnostics.gpkg')
    assert int(contract['metrics']['junction_diagnostic_count']) >= 1
    assert 'graph_junction_role_counts' in contract['metrics']

    jdiag = gpd.read_file(contract['artifacts']['junction_diagnostics'])
    assert not jdiag.empty
    assert 'dominant_component_id' in jdiag.columns
    assert 'junction_target_z_m' in jdiag.columns
    assert 'topology_source' in jdiag.columns
    assert 'constrained_station_count' in jdiag.columns


def test_junction_diagnostics_balanced_junction_does_not_fake_dominant_branch(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 0.0, 10.0],
            "component_id": ["A", "A", "B", "B"],
            "backbone_z_m": [5.0, 5.0, 5.0, 5.0],
            "backbone_mode": ["graph_backbone"] * 4,
            "resolved_stage_control_z_m": [np.nan] * 4,
            "authoritative_bed_z_m": [np.nan] * 4,
            "authoritative_backbone_z_m": [np.nan] * 4,
            "authoritative_station_support_strength": [0.0] * 4,
            "xs_residual_to_backbone_z_m": [0.0] * 4,
            "channel_support_class": ["unsupported"] * 4,
            "junction_id": ["j_bal", "j_bal", "j_bal", "j_bal"],
            "mainstem_rank": [1, 1, 1, 1],
            "network_order": [1, 1, 1, 1],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(0, 0.02), Point(1, 0.02)],
        crs="EPSG:4326",
    )
    frame_path = river_dir / "channel_frame_points.gpkg"
    frame.to_file(frame_path, driver="GPKG")
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    jdiag = gpd.read_file(out["junction_diagnostics"])
    assert not jdiag.empty
    assert bool(jdiag.loc[0, "has_clear_dominance"]) is False
    assert str(jdiag.loc[0, "dominant_component_id"] or "") == ""
    nodes = gpd.read_file(out["channel_scaffold_nodes"])
    thalweg = nodes.loc[nodes["node_role"].astype(str).eq("thalweg")].copy()
    roles = set(thalweg["graph_junction_role"].astype(str))
    assert "dominant" not in roles
    assert "balanced" in roles


def test_junction_diagnostics_records_topology_source_for_explicit_and_geometric_cases(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0],
            "component_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "backbone_z_m": [0.0, 0.0, 2.0, 2.0, 10.0, 10.0, 12.0, 12.0],
            "backbone_mode": ["graph_backbone"] * 8,
            "resolved_stage_control_z_m": [np.nan] * 8,
            "authoritative_bed_z_m": [np.nan] * 8,
            "authoritative_backbone_z_m": [np.nan] * 8,
            "authoritative_station_support_strength": [0.0] * 8,
            "xs_residual_to_backbone_z_m": [0.0] * 8,
            "channel_support_class": ["unsupported"] * 8,
            "junction_id": ["j_exp", "j_exp", "j_exp", "j_exp", np.nan, np.nan, np.nan, np.nan],
        },
        geometry=[
            Point(0, 0), Point(1, 0), Point(0, 0.02), Point(1, 0.02),
            Point(5, 0), Point(6, 0), Point(5.02, 0.02), Point(6.02, 0.02),
        ],
        crs="EPSG:4326",
    )
    frame_path = river_dir / "channel_frame_points.gpkg"
    frame.to_file(frame_path, driver="GPKG")
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    jdiag = gpd.read_file(out["junction_diagnostics"])
    sources = set(jdiag["topology_source"].astype(str))
    assert "explicit_junction_id" in sources
    assert "geometric_fallback" in sources




def test_regime_specific_regularization_strengthens_long_gaps_more_than_short(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 50.0, 100.0, 0.0, 300.0, 600.0],
            'component_id': ['short', 'short', 'short', 'long', 'long', 'long'],
            'backbone_z_m': [0.0, np.nan, 0.0, 0.0, np.nan, 0.0],
            'backbone_mode': ['authoritative_in_channel', 'missing', 'authoritative_backbone', 'authoritative_in_channel', 'missing', 'authoritative_backbone'],
            'resolved_stage_control_z_m': [0.0, np.nan, 0.0, 0.0, np.nan, 0.0],
            'authoritative_bed_z_m': [0.0, np.nan, np.nan, 0.0, np.nan, np.nan],
            'authoritative_backbone_z_m': [0.0, np.nan, 0.0, 0.0, np.nan, 0.0],
            'authoritative_station_support_strength': [2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
            'xs_residual_to_backbone_z_m': [0.0] * 6,
            'channel_support_class': ['authoritative_in_channel', 'unsupported', 'authoritative_backbone', 'authoritative_in_channel', 'unsupported', 'authoritative_backbone'],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(0, 1), Point(3, 1), Point(6, 1)],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    diag = gpd.read_file(out['graph_backbone_diagnostics'])
    short_mid = diag.loc[(diag['component_id'] == 'short') & np.isclose(diag['station_m'], 50.0)].iloc[0]
    long_mid = diag.loc[(diag['component_id'] == 'long') & np.isclose(diag['station_m'], 300.0)].iloc[0]
    assert str(short_mid['graph_unsupported_regime']) == 'short_gap_bridge'
    assert str(long_mid['graph_unsupported_regime']) == 'long_gap_stiffened'
    assert float(long_mid['graph_regularization_weight_sum']) > float(short_mid['graph_regularization_weight_sum'])
def test_graph_unsupported_regime_classification(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 50.0, 100.0, 0.0, 300.0, 600.0],
            'component_id': ['short', 'short', 'short', 'long', 'long', 'long'],
            'backbone_z_m': [0.0, np.nan, 0.0, 0.0, np.nan, 0.0],
            'backbone_mode': ['resolved_backbone'] * 6,
            'resolved_stage_control_z_m': [np.nan] * 6,
            'authoritative_bed_z_m': [np.nan] * 6,
            'authoritative_backbone_z_m': [np.nan] * 6,
            'authoritative_station_support_strength': [0.0] * 6,
            'xs_residual_to_backbone_z_m': [0.0] * 6,
            'channel_support_class': ['unsupported'] * 6,
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(0, 1), Point(3, 1), Point(6, 1)],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    diag = gpd.read_file(out['graph_backbone_diagnostics'])
    short_mid = diag.loc[(diag['component_id'] == 'short') & np.isclose(diag['station_m'], 50.0), 'graph_unsupported_regime'].iloc[0]
    long_mid = diag.loc[(diag['component_id'] == 'long') & np.isclose(diag['station_m'], 300.0), 'graph_unsupported_regime'].iloc[0]
    assert str(short_mid) == 'short_gap_bridge'
    assert str(long_mid) == 'long_gap_stiffened'
    contract = json.loads((river_dir / 'river_channel_scaffold_contract.json').read_text())
    counts = contract['metrics']['graph_unsupported_regime_counts']
    assert int(counts['short_gap_bridge']) >= 1
    assert int(counts['long_gap_stiffened']) >= 1


def test_scaffold_collapses_duplicate_graph_diagnostics_rows_before_merge(tmp_path: Path, monkeypatch):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 20.0],
            "component_id": ["main", "main", "main"],
            "backbone_z_m": [0.0, np.nan, 0.0],
            "backbone_mode": ["authoritative_in_channel", "missing", "authoritative_backbone"],
            "resolved_stage_control_z_m": [0.0, np.nan, 0.0],
            "authoritative_bed_z_m": [0.0, np.nan, np.nan],
            "authoritative_backbone_z_m": [0.0, np.nan, 0.0],
            "authoritative_station_support_strength": [2.0, 0.0, 2.0],
            "xs_residual_to_backbone_z_m": [0.0, 0.0, 0.0],
            "channel_support_class": ["authoritative_in_channel", "unsupported", "authoritative_backbone"],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs="EPSG:4326",
    )
    frame_path = river_dir / "channel_frame_points.gpkg"
    frame.to_file(frame_path, driver="GPKG")

    def fake_solve_network_backbone(frame_df):
        comp_fill = {"main": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
        junction_metrics = {}
        diagnostics = pd.DataFrame(
            {
                "component_id": ["main", "main", "main", "main"],
                "station_m": [0.0, 10.0, 10.0, 20.0],
                "graph_backbone_z_m": [0.0, 1.0, 1.0, 0.0],
                "graph_hard_lock": [True, False, False, True],
                "graph_prior_weight_sum": [0.0, 1.0, 2.0, 0.0],
                "graph_edge_weight_sum": [0.0, 1.0, 2.0, 0.0],
                "graph_curvature_weight_sum": [0.0, 1.0, 2.0, 0.0],
                "graph_centering_weight_sum": [0.0, 1.0, 2.0, 0.0],
                "graph_regularization_weight_sum": [0.0, 1.0, 2.0, 0.0],
                "graph_junction_weight_sum": [0.0, 0.0, 0.0, 0.0],
                "graph_junction_constrained": [False, False, False, False],
                "graph_residual_to_candidate_z_m": [0.0, 0.0, 0.0, 0.0],
                "graph_solver_support_class": ["authoritative_locked", "resolved_backbone", "resolved_backbone", "authoritative_backbone"],
                "graph_candidate_source": ["authoritative_in_channel", "resolved_backbone", "resolved_backbone", "authoritative_backbone"],
                "graph_solution_mode": ["hard_lock", "solved", "solved", "hard_lock"],
                "graph_unsupported_span_m": [0.0, 10.0, 10.0, 0.0],
                "graph_slope_guard_weight_sum": [0.0, 1.0, 2.0, 0.0],
                "graph_adverse_step_weight_sum": [0.0, 0.0, 0.0, 0.0],
                "graph_physical_guard_weight_sum": [0.0, 1.0, 2.0, 0.0],
                "graph_local_slope": [0.0, 0.1, 0.1, 0.0],
                "graph_local_curvature": [0.0, 0.0, 0.0, 0.0],
                "graph_slope_guard_active": [False, True, True, False],
                "graph_adverse_step_guard_active": [False, False, False, False],
            }
        )
        return comp_fill, junction_metrics, diagnostics

    monkeypatch.setattr(river_channel_scaffold, "_solve_network_backbone", fake_solve_network_backbone)

    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    assert Path(out["channel_scaffold_nodes"]).exists()
    contract = json.loads(Path(out["channel_scaffold_contract"]).read_text())
    assert int(contract["metrics"]["graph_diagnostics_duplicate_rows_collapsed"]) == 2
    nodes = gpd.read_file(out["channel_scaffold_nodes"])
    assert int(nodes[["component_id", "station_m"]].drop_duplicates().shape[0]) == 3


def test_build_channel_scaffold_products_rejects_duplicate_frame_station_keys(tmp_path: Path):
    river_dir = tmp_path / 'river_dup_frame'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 10.0],
            'component_id': ['main', 'main', 'main'],
            'authoritative_bed_z_m': [1.0, 0.8, 0.8],
            'authoritative_backbone_z_m': [1.0, 0.8, 0.8],
            'xs_residual_to_backbone_z_m': [0.0, 0.0, 0.0],
            'resolved_stage_control_z_m': [2.0, 1.8, 1.8],
        },
        geometry=[Point(2, 8), Point(5, 5), Point(5, 5)],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'bad_frame.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    with pytest.raises(RuntimeError, match='channel_frame_duplicate_station_rows'):
        build_channel_scaffold_products(
            river_dir=river_dir,
            channel_frame_points_path=frame_path,
            xs_bathy_gpkg_path=None,
            logger=None,
        )


def test_scaffold_marks_xs_supported_non_auth_nodes(tmp_path: Path):
    import geopandas as gpd
    from shapely.geometry import Point

    frame = gpd.GeoDataFrame(
        {
            'component_id': ['A'],
            'station_m': [0.0],
            'authoritative_bed_z_m': [float('nan')],
            'authoritative_backbone_z_m': [float('nan')],
            'backbone_z_m': [-2.0],
            'backbone_mode': ['resolved_backbone'],
            'xs_residual_to_backbone_z_m': [0.5],
            'resolved_stage_control_z_m': [float('nan')],
            'authoritative_station_support_strength': [0.0],
            'channel_support_class': ['anchored_interpolated'],
        },
        geometry=[Point(0, 0)],
        crs='EPSG:4326',
    )
    frame_path = tmp_path / 'frame_xs_supported.gpkg'
    frame.to_file(frame_path, driver='GPKG')

    outputs = build_channel_scaffold_products(
        river_dir=tmp_path / 'river',
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        authoritative_support_mask_path=None,
        authoritative_support_depth_path=None,
        allow_absolute_bed_fallback=False,
        logger=None,
    )
    nodes = gpd.read_file(outputs['channel_scaffold_nodes'])
    thalweg = nodes[nodes['node_role'].astype(str) == 'thalweg'].copy()
    assert not thalweg.empty
    assert set(thalweg['station_support_mode'].astype(str)) == {'xs_supported'}
    assert set(thalweg['z_source'].astype(str)) == {'graph_backbone'}



def test_build_channel_scaffold_products_does_not_promote_authoritative_backbone_only_station(tmp_path, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import Point
    import river_channel_scaffold

    river_dir = tmp_path / 'river_no_promote'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'component_id': ['main', 'main'],
            'station_m': [0.0, 10.0],
            'center_x': [0.0, 10.0],
            'center_y': [0.0, 0.0],
            'normal_x': [0.0, 0.0],
            'normal_y': [1.0, 1.0],
            'half_width_m': [2.0, 2.0],
            'channel_support_class': ['unsupported', 'authoritative_backbone'],
            'backbone_mode': ['longitudinal_profile', 'authoritative_backbone'],
            'authoritative_bed_elevation_m': [np.nan, np.nan],
            'authoritative_bed_z_m': [np.nan, np.nan],
            'authoritative_backbone_z_m': [np.nan, 0.0],
            'resolved_backbone_z_m': [1.0, 0.5],
            'bank_stage_z_m': [1.2, 0.6],
            'xs_residual_to_backbone_z_m': [0.0, 0.0],
            'graph_candidate_source': ['resolved_backbone', 'authoritative_backbone'],
            'graph_solver_support_class': ['resolved_backbone', 'authoritative_backbone'],
            'graph_solution_mode': ['solved', 'frame_backbone_resolved'],
            'graph_hard_lock': [False, False],
            'geometry': [Point(0,0), Point(10,0)],
        },
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')

    def fake_solve_network_backbone(frame_df):
        comp_fill = {'main': np.array([1.0, 0.5], dtype=np.float32)}
        junction_metrics = {}
        diagnostics = frame_df[['component_id', 'station_m']].copy()
        diagnostics['graph_backbone_z_m'] = [1.0, 0.5]
        diagnostics['graph_hard_lock'] = [False, False]
        diagnostics['graph_prior_weight_sum'] = [1.0, 1.0]
        diagnostics['graph_edge_weight_sum'] = [1.0, 1.0]
        diagnostics['graph_curvature_weight_sum'] = [0.0, 0.0]
        diagnostics['graph_centering_weight_sum'] = [0.0, 0.0]
        diagnostics['graph_regularization_weight_sum'] = [1.0, 1.0]
        diagnostics['graph_junction_weight_sum'] = [0.0, 0.0]
        diagnostics['graph_junction_constrained'] = [False, False]
        diagnostics['graph_residual_to_candidate_z_m'] = [0.0, 0.0]
        diagnostics['graph_solver_support_class'] = ['resolved_backbone', 'authoritative_backbone']
        diagnostics['graph_candidate_source'] = ['resolved_backbone', 'authoritative_backbone']
        diagnostics['graph_solution_mode'] = ['solved', 'frame_backbone_resolved']
        diagnostics['graph_unsupported_span_m'] = [10.0, 10.0]
        diagnostics['graph_slope_guard_weight_sum'] = [1.0, 1.0]
        diagnostics['graph_adverse_step_weight_sum'] = [0.0, 0.0]
        diagnostics['graph_physical_guard_weight_sum'] = [1.0, 1.0]
        diagnostics['graph_local_slope'] = [0.1, 0.1]
        diagnostics['graph_local_curvature'] = [0.0, 0.0]
        diagnostics['graph_slope_guard_active'] = [True, True]
        diagnostics['graph_adverse_step_guard_active'] = [False, False]
        return comp_fill, junction_metrics, diagnostics

    monkeypatch.setattr(river_channel_scaffold, '_solve_network_backbone', fake_solve_network_backbone)

    out = river_channel_scaffold.build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    nodes = gpd.read_file(out['channel_scaffold_nodes'])
    station10 = nodes.loc[np.isclose(pd.to_numeric(nodes['station_m'], errors='coerce').to_numpy(dtype=float), 10.0)]
    assert not station10['z_source'].astype(str).eq('authoritative_in_channel').all()
    assert not station10['station_authoritative_backed'].astype(bool).all()


def test_scaffold_only_marks_xs_supported_when_actual_xs_residual_support_exists(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0],
            'component_id': ['main', 'main', 'main'],
            'backbone_z_m': [5.0, 4.5, 4.0],
            'backbone_mode': ['resolved_backbone'] * 3,
            'resolved_stage_control_z_m': [np.nan, np.nan, np.nan],
            'resolved_channel_bed_z_m': [np.nan] * 3,
            'authoritative_bed_z_m': [np.nan] * 3,
            'authoritative_backbone_z_m': [np.nan] * 3,
            'authoritative_station_support_strength': [0.0] * 3,
            'xs_residual_to_backbone_z_m': [np.nan, np.nan, np.nan],
            'channel_support_class': ['unsupported'] * 3,
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')

    xs_bathy = gpd.GeoDataFrame(
        {
            'xs_id': ['A', 'A', 'A', 'B', 'B', 'B'],
            'width_m': [10.0] * 6,
            'dist_m': [0.0, 5.0, 10.0, 0.0, 5.0, 10.0],
            'z_bed_pred_m': [5.5, 4.0, 5.5, 5.0, 3.5, 5.0],
        },
        geometry=[Point(0, 0), Point(0, 0), Point(0, 0), Point(2, 0), Point(2, 0), Point(2, 0)],
        crs='EPSG:4326',
    )
    xs_bathy_path = river_dir / 'river_bathy_xs_mainstem.gpkg'
    xs_bathy.to_file(xs_bathy_path, layer='xs_bathy_points', driver='GPKG')

    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=xs_bathy_path,
        allow_absolute_bed_fallback=False,
    )
    nodes = gpd.read_file(out['channel_scaffold_nodes'])
    center = nodes.loc[nodes['station_m'].astype(float).round(6).eq(10.0)].copy()
    assert not center.empty
    assert set(center['station_support_mode'].astype(str)) == {'resolved_backbone'}
    assert not np.any(center['z_source'].astype(str).eq('xs_profile_resampled'))



def test_scaffold_keeps_bank_margin_authority_out_of_channel_core(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0],
            'component_id': ['main', 'main', 'main'],
            'backbone_z_m': [5.0, 4.5, 4.0],
            'backbone_mode': ['authoritative_bank_margin'] * 3,
            'resolved_stage_control_z_m': [6.0, 5.5, 5.0],
            'resolved_channel_bed_z_m': [np.nan] * 3,
            'authoritative_bed_z_m': [6.0, 5.5, 5.0],
            'authoritative_hard_bed_z_m': [np.nan] * 3,
            'authoritative_bank_margin_z_m': [6.0, 5.5, 5.0],
            'authoritative_backbone_z_m': [np.nan] * 3,
            'authoritative_station_support_strength': [1.0, 1.0, 1.0],
            'authoritative_role': [ROLE_BANK_MARGIN] * 3,
            'authoritative_bed_support_present': [False, False, False],
            'authoritative_bank_margin_present': [True, True, True],
            'xs_residual_to_backbone_z_m': [0.0, 0.0, 0.0],
            'channel_support_class': ['authoritative_bank_margin'] * 3,
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    nodes = gpd.read_file(out['channel_scaffold_nodes'])
    assert not nodes['z_source'].astype(str).eq('authoritative_in_channel').any()
    bank_nodes = nodes.loc[nodes['node_role'].astype(str).isin(['left_bank', 'right_bank'])]
    inner_nodes = nodes.loc[nodes['node_role'].astype(str).isin(['left_inner', 'right_inner', 'thalweg'])]
    assert bank_nodes['z_source'].astype(str).eq('authoritative_bank_margin').any()
    assert not inner_nodes['z_source'].astype(str).eq('authoritative_bank_margin').any()


def test_scaffold_uses_fitted_banks_and_active_core_when_no_xs_shape(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    frame = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0],
            'component_id': ['main', 'main', 'main'],
            'backbone_z_m': [5.5, 5.0, 4.5],
            'backbone_mode': ['active_core_support'] * 3,
            'resolved_stage_control_z_m': [7.0, 6.8, 6.6],
            'resolved_stage_control_source': ['bank_pair_fit'] * 3,
            'active_core_support_z_m': [5.5, 5.0, 4.5],
            'left_bank_fit_z_m': [7.5, 7.0, 6.5],
            'right_bank_fit_z_m': [7.7, 7.2, 6.7],
            'bank_pair_fit_z_m': [7.6, 7.1, 6.6],
            'authoritative_bed_z_m': [np.nan, np.nan, np.nan],
            'authoritative_hard_bed_z_m': [np.nan, np.nan, np.nan],
            'authoritative_bank_margin_z_m': [np.nan, np.nan, np.nan],
            'authoritative_backbone_z_m': [np.nan, np.nan, np.nan],
            'authoritative_station_support_strength': [0.0, 0.0, 0.0],
            'xs_residual_to_backbone_z_m': [np.nan, np.nan, np.nan],
            'channel_support_class': ['unsupported'] * 3,
            'graph_solver_support_class': ['unsupported'] * 3,
            'graph_candidate_source': ['active_core_support'] * 3,
            'graph_solution_mode': ['component_fill'] * 3,
            'graph_backbone_z_m': [5.5, 5.0, 4.5],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs='EPSG:4326',
    )
    frame_path = river_dir / 'channel_frame_points.gpkg'
    frame.to_file(frame_path, driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame_path,
        xs_bathy_gpkg_path=None,
        allow_absolute_bed_fallback=False,
    )
    nodes = gpd.read_file(out['channel_scaffold_nodes'])
    center = nodes.loc[np.isclose(pd.to_numeric(nodes['station_m'], errors='coerce'), 10.0)].copy()
    zmap = dict(zip(center['node_role'].astype(str), pd.to_numeric(center['bed_z_m'], errors='coerce')))
    srcmap = dict(zip(center['node_role'].astype(str), center['z_source'].astype(str)))
    assert np.isclose(float(zmap['left_bank']), 7.0)
    assert np.isclose(float(zmap['left_inner']), 5.3157730, atol=1e-5)
    assert np.isclose(float(zmap['thalweg']), 5.0)
    assert np.isclose(float(zmap['right_inner']), 5.3157730, atol=1e-5)
    assert np.isclose(float(zmap['right_bank']), 7.2)
    assert srcmap['left_bank'] == 'bank_fit_profile'
    assert srcmap['left_inner'] == 'generalized_thalweg_default_tendency'
    assert srcmap['thalweg'] == 'active_core_support'


def test_build_channel_scaffold_products_can_disable_xs_influence(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    center = gpd.GeoDataFrame(
        {'station_m': [0.0, 10.0, 20.0], 'component_id': ['main', 'main', 'main']},
        geometry=[Point(2,8), Point(5,5), Point(8,2)],
        crs='EPSG:4326',
    )
    center_path = river_dir / 'centerline_points.gpkg'
    center.to_file(center_path, driver='GPKG')
    xs_support = gpd.GeoDataFrame(
        {'xs_sample_source': ['authoritative', 'authoritative', 'authoritative'], 'xs_z_m': [1.0, 0.5, 0.2]},
        geometry=[Point(2,8), Point(5,5), Point(8,2)],
        crs='EPSG:4326',
    )
    xs_support_path = river_dir / 'xs_support_points.gpkg'
    xs_support.to_file(xs_support_path, driver='GPKG')
    arr = np.full((10,10), -9999.0, dtype=np.float32)
    arr[2,2] = 1.0
    arr[5,5] = 0.5
    arr[8,8] = 0.2
    for name in ['bank.tif','bankinf.tif','xselev.tif','xsw.tif','centelev.tif','centinf.tif','long.tif','authmask.tif','authdepth.tif','authbed.tif']:
        a = arr.copy()
        if name=='xsw.tif':
            a[:] = -9999.0
            a[2,2] = 1.0; a[5,5] = 1.0; a[8,8] = 1.0
        if name=='authmask.tif':
            a[:] = -9999.0
            a[2,2] = 1.0; a[5,5] = 1.0; a[8,8] = 1.0
        _write_raster(river_dir/name, a)
    frame = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=center_path,
        xs_support_points_path=xs_support_path,
        bank_elevation_path=river_dir/'bank.tif',
        bank_influence_path=river_dir/'bankinf.tif',
        xs_support_elevation_path=river_dir/'xselev.tif',
        xs_support_weight_path=river_dir/'xsw.tif',
        centerline_elevation_path=river_dir/'centelev.tif',
        centerline_influence_path=river_dir/'centinf.tif',
        longitudinal_profile_elevation_path=river_dir/'long.tif',
        authoritative_support_mask_path=river_dir/'authmask.tif',
        authoritative_support_depth_path=river_dir/'authdepth.tif',
        authoritative_bed_elevation_path=river_dir/'authbed.tif',
    )
    xs_bathy = gpd.GeoDataFrame(
        {
            'xs_id': ['A','A','A','B','B','B'],
            'width_m': [10.0]*6,
            'dist_m': [0.0,5.0,10.0,0.0,5.0,10.0],
            'z_bed_pred_m': [2.0,1.0,2.0,1.8,0.8,1.8],
        },
        geometry=[Point(2,8),Point(2,8),Point(2,8),Point(8,2),Point(8,2),Point(8,2)],
        crs='EPSG:4326',
    )
    xs_bathy_path = tmp_path / 'river_bathy_xs_mainstem.gpkg'
    xs_bathy.to_file(xs_bathy_path, layer='xs_bathy_points', driver='GPKG')
    out = build_channel_scaffold_products(
        river_dir=river_dir,
        channel_frame_points_path=frame['channel_frame_points'],
        xs_bathy_gpkg_path=xs_bathy_path,
        disable_xs_influence=True,
    )
    nodes = gpd.read_file(out['channel_scaffold_nodes'])
    assert not bool(nodes['z_source'].astype(str).eq('xs_profile_resampled').any())
    assert not bool(nodes['target_xs_realism_allowed'].fillna(False).astype(bool).any())
    assert bool(nodes['target_rebuild_allowed'].fillna(False).astype(bool).all())
    contract = json.loads(Path(out['channel_scaffold_contract']).read_text(encoding='utf-8'))
    assert contract['metrics']['xs_influence_disabled'] is True
