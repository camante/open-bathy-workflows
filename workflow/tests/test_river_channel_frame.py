import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from river_channel_frame import build_channel_frame_products


def _write_raster(path: Path, arr: np.ndarray, nodata: float = -9999.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=from_origin(0, 3, 1, 1),
        nodata=nodata,
    ) as ds:
        out = arr.astype('float32').copy()
        out[~np.isfinite(out)] = nodata
        ds.write(out, 1)
    return path


def test_build_channel_frame_products_exports_authoritative_anchors(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            'station_m': [0.0, 10.0, 20.0],
            'component_id': ['a', 'a', 'a'],
            'centerline_z_m': [-2.0, -2.5, -3.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs='EPSG:4326',
    )
    cpts = river_dir / 'river_centerline_points.gpkg'
    centerline.to_file(cpts, driver='GPKG')

    xs = gpd.GeoDataFrame(
        {'xs_sample_source': ['authoritative', 'missing'], 'xs_z_m': [-2.1, np.nan]},
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5)],
        crs='EPSG:4326',
    )
    xsp = river_dir / 'river_xs_support_points.gpkg'
    xs.to_file(xsp, driver='GPKG')

    arr = np.array([
        [np.nan, np.nan, np.nan],
        [-2.0, -2.4, -2.8],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    support = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    depth = np.array([
        [-2.0, np.nan, -2.8],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    bank = np.array([
        [1.0, 1.1, 1.2],
        [1.0, 1.1, 1.2],
        [1.0, 1.1, 1.2],
    ], dtype=np.float32)
    xs_elev = np.array([
        [np.nan, np.nan, np.nan],
        [-2.1, -2.3, -2.7],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    xs_w = np.array([
        [0.0, 0.0, 0.0],
        [0.8, 0.8, 0.8],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    long = np.array([
        [np.nan, np.nan, np.nan],
        [-2.0, -2.5, -3.0],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=xsp,
        bank_elevation_path=_write_raster(river_dir / 'bank.tif', bank),
        bank_influence_path=_write_raster(river_dir / 'bank_inf.tif', np.ones((3,3), dtype=np.float32)),
        xs_support_elevation_path=_write_raster(river_dir / 'xs.tif', xs_elev),
        xs_support_weight_path=_write_raster(river_dir / 'xsw.tif', xs_w),
        centerline_elevation_path=_write_raster(river_dir / 'center.tif', arr),
        centerline_influence_path=_write_raster(river_dir / 'center_inf.tif', np.ones((3,3), dtype=np.float32)),
        longitudinal_profile_elevation_path=_write_raster(river_dir / 'long.tif', long),
        authoritative_support_mask_path=_write_raster(river_dir / 'support.tif', support, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / 'support_depth.tif', depth),
        authoritative_bed_elevation_path=_write_raster(river_dir / 'auth_bed.tif', arr),
    )
    assert Path(outputs['channel_frame_points']).exists()
    assert Path(outputs['channel_frame_contract']).exists()

    frame = gpd.read_file(outputs['channel_frame_points'])
    assert 'channel_support_class' in frame.columns
    assert 'authoritative_anchor_present' in frame.columns
    assert 'authoritative_bed_z_m' in frame.columns
    assert 'authoritative_xs_anchors' in outputs


def test_build_channel_frame_products_densifies_authoritative_backbone_from_supported_centerline(tmp_path: Path):
    river_dir = tmp_path / "river2"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 20.0],
            "component_id": ["a", "a", "a"],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:4326",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    centerline.to_file(cpts, driver="GPKG")
    xs = gpd.GeoDataFrame({"xs_sample_source": [], "xs_z_m": []}, geometry=[], crs="EPSG:4326")
    xsp = river_dir / "river_xs_support_points.gpkg"
    xs.to_file(xsp, driver="GPKG")

    center = np.array([[-2.0, -2.4, -2.8], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    support = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    authbed = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    ones = np.ones((3, 3), dtype=np.float32)
    zeros = np.zeros((3, 3), dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=xsp,
        bank_elevation_path=_write_raster(river_dir / 'bank.tif', np.full((3, 3), np.nan, dtype=np.float32)),
        bank_influence_path=_write_raster(river_dir / 'bank_inf.tif', ones),
        xs_support_elevation_path=_write_raster(river_dir / 'xs.tif', authbed),
        xs_support_weight_path=_write_raster(river_dir / 'xsw.tif', authbed),
        centerline_elevation_path=_write_raster(river_dir / 'center.tif', center),
        centerline_influence_path=_write_raster(river_dir / 'center_inf.tif', ones),
        longitudinal_profile_elevation_path=_write_raster(river_dir / 'long.tif', center),
        authoritative_support_mask_path=_write_raster(river_dir / 'support.tif', support, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / 'support_depth.tif', authbed),
        authoritative_bed_elevation_path=_write_raster(river_dir / 'auth_bed.tif', authbed),
    )
    frame = gpd.read_file(outputs['channel_frame_points'])
    assert np.count_nonzero(np.isfinite(frame['authoritative_backbone_candidate_z_m'])) == 3
    assert np.count_nonzero(np.isfinite(frame['authoritative_backbone_z_m'])) == 3
    assert np.count_nonzero(frame['authoritative_anchor_present'].astype(bool)) == 0


def test_build_channel_frame_products_exports_backbone_contract_fields(tmp_path: Path):
    river_dir = tmp_path / "river3"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 20.0],
            "component_id": ["a", "a", "a"],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:4326",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    centerline.to_file(cpts, driver="GPKG")
    xs = gpd.GeoDataFrame({"xs_sample_source": [], "xs_z_m": []}, geometry=[], crs="EPSG:4326")
    xsp = river_dir / "river_xs_support_points.gpkg"
    xs.to_file(xsp, driver="GPKG")

    center = np.array([[-2.0, -2.4, -2.8], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    xs_elev = np.array([[-2.1, -2.5, -3.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    xs_w = np.array([[0.8, 0.8, 0.8], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    zeros = np.zeros((3, 3), dtype=np.float32)
    ones = np.ones((3, 3), dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=xsp,
        bank_elevation_path=_write_raster(river_dir / 'bank.tif', np.full((3, 3), np.nan, dtype=np.float32)),
        bank_influence_path=_write_raster(river_dir / 'bank_inf.tif', ones),
        xs_support_elevation_path=_write_raster(river_dir / 'xs.tif', xs_elev),
        xs_support_weight_path=_write_raster(river_dir / 'xsw.tif', xs_w),
        centerline_elevation_path=_write_raster(river_dir / 'center.tif', center),
        centerline_influence_path=_write_raster(river_dir / 'center_inf.tif', ones),
        longitudinal_profile_elevation_path=_write_raster(river_dir / 'long.tif', center),
        authoritative_support_mask_path=_write_raster(river_dir / 'support.tif', zeros, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / 'support_depth.tif', zeros),
        authoritative_bed_elevation_path=_write_raster(river_dir / 'auth_bed.tif', zeros),
    )
    frame = gpd.read_file(outputs['channel_frame_points'])
    assert 'backbone_z_m' in frame.columns
    assert 'backbone_mode' in frame.columns
    assert 'xs_residual_to_backbone_z_m' in frame.columns
    assert 'residual_shape_mode' in frame.columns
    assert np.count_nonzero(np.isfinite(frame['backbone_z_m'])) == 3
    assert set(frame['backbone_mode'].astype(str)) == {'longitudinal_profile'}
    assert np.count_nonzero(np.isfinite(frame['xs_residual_to_backbone_z_m'])) == 3
    assert set(frame['residual_shape_mode'].astype(str)) == {'xs_residual_to_backbone'}


def test_build_channel_frame_products_omits_legacy_mixed_bed_by_default(tmp_path: Path):
    river_dir = tmp_path / "river4"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 20.0],
            "component_id": ["a", "a", "a"],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:4326",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    centerline.to_file(cpts, driver="GPKG")
    xs = gpd.GeoDataFrame({"xs_sample_source": [], "xs_z_m": []}, geometry=[], crs="EPSG:4326")
    xsp = river_dir / "river_xs_support_points.gpkg"
    xs.to_file(xsp, driver="GPKG")

    center = np.array([[-2.0, -2.4, -2.8], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    xs_elev = np.array([[-2.1, -2.5, -3.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    xs_w = np.array([[0.8, 0.8, 0.8], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    zeros = np.zeros((3, 3), dtype=np.float32)
    ones = np.ones((3, 3), dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=xsp,
        bank_elevation_path=_write_raster(river_dir / 'bank.tif', np.full((3, 3), np.nan, dtype=np.float32)),
        bank_influence_path=_write_raster(river_dir / 'bank_inf.tif', ones),
        xs_support_elevation_path=_write_raster(river_dir / 'xs.tif', xs_elev),
        xs_support_weight_path=_write_raster(river_dir / 'xsw.tif', xs_w),
        centerline_elevation_path=_write_raster(river_dir / 'center.tif', center),
        centerline_influence_path=_write_raster(river_dir / 'center_inf.tif', ones),
        longitudinal_profile_elevation_path=_write_raster(river_dir / 'long.tif', center),
        authoritative_support_mask_path=_write_raster(river_dir / 'support.tif', zeros, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / 'support_depth.tif', zeros),
        authoritative_bed_elevation_path=_write_raster(river_dir / 'auth_bed.tif', zeros),
    )
    frame = gpd.read_file(outputs['channel_frame_points'])
    assert 'resolved_channel_bed_z_m' not in frame.columns


def test_build_channel_frame_products_exports_legacy_mixed_bed_only_when_requested(tmp_path: Path):
    river_dir = tmp_path / "river5"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 20.0],
            "component_id": ["a", "a", "a"],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:4326",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    centerline.to_file(cpts, driver="GPKG")
    xs = gpd.GeoDataFrame({"xs_sample_source": [], "xs_z_m": []}, geometry=[], crs="EPSG:4326")
    xsp = river_dir / "river_xs_support_points.gpkg"
    xs.to_file(xsp, driver="GPKG")

    center = np.array([[-2.0, -2.4, -2.8], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    xs_elev = np.array([[-2.1, -2.5, -3.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    xs_w = np.array([[0.8, 0.8, 0.8], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    zeros = np.zeros((3, 3), dtype=np.float32)
    ones = np.ones((3, 3), dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=xsp,
        bank_elevation_path=_write_raster(river_dir / 'bank.tif', np.full((3, 3), np.nan, dtype=np.float32)),
        bank_influence_path=_write_raster(river_dir / 'bank_inf.tif', ones),
        xs_support_elevation_path=_write_raster(river_dir / 'xs.tif', xs_elev),
        xs_support_weight_path=_write_raster(river_dir / 'xsw.tif', xs_w),
        centerline_elevation_path=_write_raster(river_dir / 'center.tif', center),
        centerline_influence_path=_write_raster(river_dir / 'center_inf.tif', ones),
        longitudinal_profile_elevation_path=_write_raster(river_dir / 'long.tif', center),
        authoritative_support_mask_path=_write_raster(river_dir / 'support.tif', zeros, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / 'support_depth.tif', zeros),
        authoritative_bed_elevation_path=_write_raster(river_dir / 'auth_bed.tif', zeros),
        export_legacy_mixed_bed=True,
    )
    frame = gpd.read_file(outputs['channel_frame_points'])
    assert 'resolved_channel_bed_z_m' in frame.columns


def test_build_channel_frame_products_preserves_topology_fields(tmp_path: Path):
    import geopandas as gpd
    from shapely.geometry import Point

    river_dir = tmp_path / "river"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0],
            "component_id": ["main", "main"],
            "junction_id": ["j1", "j1"],
            "downstream_component_id": ["mouth", "mouth"],
            "upstream_component_ids": ["trib_a,trib_b", "trib_a,trib_b"],
            "mainstem_rank": [1, 1],
            "network_order": [4, 4],
            "distance_to_mouth_m": [100.0, 90.0],
            "geometry": [Point(0, 0), Point(10, 0)],
        },
        crs="EPSG:32619",
    )
    centerline_path = river_dir / "centerline.gpkg"
    centerline.to_file(centerline_path, driver="GPKG")
    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=centerline_path,
        xs_support_points_path=None,
        bank_elevation_path=None,
        bank_influence_path=None,
        xs_support_elevation_path=None,
        xs_support_weight_path=None,
        centerline_elevation_path=None,
        centerline_influence_path=None,
        longitudinal_profile_elevation_path=None,
        authoritative_support_mask_path=None,
        authoritative_support_depth_path=None,
        authoritative_bed_elevation_path=None,
    )
    frame = gpd.read_file(outputs["channel_frame_points"])
    assert set(["junction_id", "downstream_component_id", "upstream_component_ids", "mainstem_rank", "network_order", "distance_to_mouth_m"]).issubset(frame.columns)


def test_build_channel_frame_products_collapses_duplicate_component_station_rows(tmp_path: Path):
    river_dir = tmp_path / "river_dup"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 10.0, 20.0],
            "component_id": ["a", "a", "a", "a"],
            "flow_idx": ["f0", "f1", "f2", "f3"],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:4326",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    centerline.to_file(cpts, driver="GPKG")
    xs = gpd.GeoDataFrame({"xs_sample_source": [], "xs_z_m": []}, geometry=[], crs="EPSG:4326")
    xsp = river_dir / "river_xs_support_points.gpkg"
    xs.to_file(xsp, driver="GPKG")

    center = np.array([[-2.0, -2.4, -2.8], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    zeros = np.zeros((3, 3), dtype=np.float32)
    ones = np.ones((3, 3), dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=xsp,
        bank_elevation_path=_write_raster(river_dir / 'bank.tif', np.full((3, 3), np.nan, dtype=np.float32)),
        bank_influence_path=_write_raster(river_dir / 'bank_inf.tif', ones),
        xs_support_elevation_path=_write_raster(river_dir / 'xs.tif', zeros),
        xs_support_weight_path=_write_raster(river_dir / 'xsw.tif', zeros),
        centerline_elevation_path=_write_raster(river_dir / 'center.tif', center),
        centerline_influence_path=_write_raster(river_dir / 'center_inf.tif', ones),
        longitudinal_profile_elevation_path=_write_raster(river_dir / 'long.tif', center),
        authoritative_support_mask_path=_write_raster(river_dir / 'support.tif', zeros, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / 'support_depth.tif', zeros),
        authoritative_bed_elevation_path=_write_raster(river_dir / 'auth_bed.tif', zeros),
    )

    frame = gpd.read_file(outputs['channel_frame_points']).sort_values(['component_id', 'station_m']).reset_index(drop=True)
    assert len(frame) == 3
    assert frame[['component_id', 'station_m']].duplicated().sum() == 0

    import json
    contract = json.loads(Path(outputs['channel_frame_contract']).read_text(encoding='utf-8'))
    assert contract['metrics']['duplicate_station_rows_collapsed'] == 1


def test_build_channel_frame_products_adds_xs_fallback_graph_provenance(tmp_path: Path):
    river_dir = tmp_path / "river_xs_fallback"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 20.0],
            "component_id": ["a", "a", "a"],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs='EPSG:4326',
    )
    cpts = river_dir / 'river_centerline_points.gpkg'
    centerline.to_file(cpts, driver='GPKG')

    zeros = np.zeros((3, 3), dtype=np.float32)
    ones = np.ones((3, 3), dtype=np.float32)
    center = np.array([
        [-2.0, -2.4, -2.8],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    xs_elev = np.array([
        [-2.3, -2.7, -3.1],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    xs_w = np.array([
        [0.8, 0.8, 0.8],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=None,
        bank_elevation_path=_write_raster(river_dir / 'bank.tif', np.full((3, 3), np.nan, dtype=np.float32)),
        bank_influence_path=_write_raster(river_dir / 'bank_inf.tif', ones),
        xs_support_elevation_path=_write_raster(river_dir / 'xs.tif', xs_elev),
        xs_support_weight_path=_write_raster(river_dir / 'xsw.tif', xs_w),
        centerline_elevation_path=_write_raster(river_dir / 'center.tif', center),
        centerline_influence_path=_write_raster(river_dir / 'center_inf.tif', ones),
        longitudinal_profile_elevation_path=_write_raster(river_dir / 'long.tif', center),
        authoritative_support_mask_path=_write_raster(river_dir / 'support.tif', zeros, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / 'support_depth.tif', zeros),
        authoritative_bed_elevation_path=_write_raster(river_dir / 'auth_bed.tif', zeros),
    )
    frame = gpd.read_file(outputs['channel_frame_points']).sort_values('station_m').reset_index(drop=True)
    assert set(frame['channel_support_class'].astype(str)) == {'xs_supported'}
    assert set(frame['graph_candidate_source'].astype(str)) == {'xs_profile_resampled'}
    assert set(frame['graph_solver_support_class'].astype(str)) == {'xs_residual_only'}
    assert set(frame['graph_solution_mode'].astype(str)) == {'frame_xs_supported'}


def test_build_channel_frame_products_keeps_mixed_inner_and_bank_authoritative_xs_as_measured(tmp_path: Path):
    river_dir = tmp_path / "river_mixed_auth_xs"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            "station_m": [0.0],
            "component_id": ["a"],
        },
        geometry=[Point(0.5, 2.5)],
        crs="EPSG:4326",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    centerline.to_file(cpts, driver="GPKG")

    xs = gpd.GeoDataFrame(
        {
            "xs_sample_source": ["authoritative", "authoritative", "authoritative"],
            "xs_z_m": [-2.1, -2.2, -2.3],
            "normalized_position": [0.10, 0.50, 0.90],
        },
        geometry=[Point(0.5, 2.5), Point(0.5, 2.5), Point(0.5, 2.5)],
        crs="EPSG:4326",
    )
    xsp = river_dir / "river_xs_support_points.gpkg"
    xs.to_file(xsp, driver="GPKG")

    center = np.array([
        [-2.0, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    bank = np.array([
        [1.0, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    xs_elev = np.array([
        [-2.1, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    xs_w = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    support = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    depth = np.array([
        [-2.0, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=xsp,
        bank_elevation_path=_write_raster(river_dir / 'bank.tif', bank),
        bank_influence_path=_write_raster(river_dir / 'bank_inf.tif', np.ones((3, 3), dtype=np.float32)),
        xs_support_elevation_path=_write_raster(river_dir / 'xs.tif', xs_elev),
        xs_support_weight_path=_write_raster(river_dir / 'xsw.tif', xs_w),
        centerline_elevation_path=_write_raster(river_dir / 'center.tif', center),
        centerline_influence_path=_write_raster(river_dir / 'center_inf.tif', np.ones((3, 3), dtype=np.float32)),
        longitudinal_profile_elevation_path=_write_raster(river_dir / 'long.tif', center),
        authoritative_support_mask_path=_write_raster(river_dir / 'support.tif', support, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / 'support_depth.tif', depth),
        authoritative_bed_elevation_path=_write_raster(river_dir / 'auth_bed.tif', depth),
    )

    frame = gpd.read_file(outputs['channel_frame_points'])
    assert bool(frame['true_measured_xs_candidate'].iloc[0]) is True
    assert bool(frame['true_measured_xs_qualified'].iloc[0]) is True
    assert str(frame['auth_xs_support_class'].iloc[0]) == 'authoritative_bed_mixed'


def test_build_channel_frame_prefers_active_core_support_over_legacy_profile(tmp_path: Path):
    river_dir = tmp_path / "river_active_frame"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 20.0],
            "component_id": ["a", "a", "a"],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:4326",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    centerline.to_file(cpts, driver="GPKG")
    xsp = river_dir / "river_xs_support_points.gpkg"
    gpd.GeoDataFrame({"xs_sample_source": [], "xs_z_m": []}, geometry=[], crs="EPSG:4326").to_file(xsp, driver="GPKG")

    base = np.array([[-2.0, -2.4, -2.8], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    legacy = np.array([[-2.0, -1.0, -2.8], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    active = np.array([[-2.0, -2.4, -2.8], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    zeros = np.zeros((3, 3), dtype=np.float32)
    ones = np.ones((3, 3), dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=xsp,
        bank_elevation_path=_write_raster(river_dir / "bank.tif", zeros),
        bank_influence_path=_write_raster(river_dir / "bank_inf.tif", zeros),
        xs_support_elevation_path=_write_raster(river_dir / "xs.tif", np.full((3, 3), np.nan, dtype=np.float32)),
        xs_support_weight_path=_write_raster(river_dir / "xsw.tif", zeros),
        centerline_elevation_path=_write_raster(river_dir / "center.tif", base),
        centerline_influence_path=_write_raster(river_dir / "center_inf.tif", ones),
        longitudinal_profile_elevation_path=_write_raster(river_dir / "long.tif", legacy),
        active_core_support_elevation_path=_write_raster(river_dir / "active.tif", active),
        authoritative_support_mask_path=_write_raster(river_dir / "support.tif", zeros, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / "support_depth.tif", np.full((3, 3), np.nan, dtype=np.float32)),
        authoritative_bed_elevation_path=_write_raster(river_dir / "auth_bed.tif", np.full((3, 3), np.nan, dtype=np.float32)),
    )

    frame = gpd.read_file(outputs["channel_frame_points"]).sort_values("station_m").reset_index(drop=True)
    assert np.isclose(float(frame.loc[1, "backbone_z_m"]), -2.4)
    assert str(frame.loc[1, "backbone_mode"]) == "active_core_support"


def test_build_channel_frame_products_suppresses_xs_residual_in_long_unsupported_fluvial_reaches(tmp_path: Path):
    river_dir = tmp_path / "river_long_unsupported"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 20.0],
            "component_id": ["1.0", "1.0", "1.0"],
            "profile_inside_fluvial_monotone_domain": [True, True, True],
            "profile_authoritative_bed_support_distance_m": [800.0, 900.0, 1000.0],
            "profile_far_from_authoritative_bed_support": [True, True, True],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:4326",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    centerline.to_file(cpts, driver="GPKG")
    xs = gpd.GeoDataFrame({"xs_sample_source": [], "xs_z_m": []}, geometry=[], crs="EPSG:4326")
    xsp = river_dir / "river_xs_support_points.gpkg"
    xs.to_file(xsp, driver="GPKG")

    center = np.array([[-2.0, -2.4, -2.8], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    xs_elev = np.array([[-2.1, -2.5, -3.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    xs_w = np.array([[0.8, 0.8, 0.8], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    zeros = np.zeros((3, 3), dtype=np.float32)
    ones = np.ones((3, 3), dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=xsp,
        bank_elevation_path=_write_raster(river_dir / 'bank.tif', np.full((3, 3), np.nan, dtype=np.float32)),
        bank_influence_path=_write_raster(river_dir / 'bank_inf.tif', ones),
        xs_support_elevation_path=_write_raster(river_dir / 'xs.tif', xs_elev),
        xs_support_weight_path=_write_raster(river_dir / 'xsw.tif', xs_w),
        centerline_elevation_path=_write_raster(river_dir / 'center.tif', center),
        centerline_influence_path=_write_raster(river_dir / 'center_inf.tif', ones),
        longitudinal_profile_elevation_path=_write_raster(river_dir / 'long.tif', center),
        authoritative_support_mask_path=_write_raster(river_dir / 'support.tif', zeros, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / 'support_depth.tif', zeros),
        authoritative_bed_elevation_path=_write_raster(river_dir / 'auth_bed.tif', zeros),
    )
    frame = gpd.read_file(outputs['channel_frame_points'])
    assert np.count_nonzero(np.isfinite(frame['xs_support_z_m'])) == 3
    assert np.count_nonzero(np.isfinite(frame['xs_residual_to_backbone_z_m'])) == 0
    assert set(frame['residual_shape_mode'].astype(str)) == {'suppressed_longitudinal_core'}


def test_build_channel_frame_products_writes_canonical_station_targets(tmp_path: Path):
    river_dir = tmp_path / "river_targets"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 10.0, 20.0],
            "component_id": ["1.0", "1.0", "1.0"],
            "profile_inside_fluvial_monotone_domain": [True, True, True],
            "station_authoritative_bed_support_distance_m": [25.0, 600.0, 900.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:4326",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    centerline.to_file(cpts, driver="GPKG")
    xs = gpd.GeoDataFrame({"xs_sample_source": [], "xs_z_m": []}, geometry=[], crs="EPSG:4326")
    xsp = river_dir / "river_xs_support_points.gpkg"
    xs.to_file(xsp, driver="GPKG")

    bank = np.array([[1.2, 1.1, 1.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    active = np.array([[-2.0, -2.2, -2.4], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    long = np.array([[-1.9, -2.1, -2.3], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    zeros = np.zeros((3, 3), dtype=np.float32)
    ones = np.ones((3, 3), dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=xsp,
        bank_elevation_path=_write_raster(river_dir / 'bank.tif', bank),
        bank_influence_path=_write_raster(river_dir / 'bank_inf.tif', ones),
        left_bank_fit_elevation_path=_write_raster(river_dir / 'left_bank_fit.tif', bank + 0.1),
        right_bank_fit_elevation_path=_write_raster(river_dir / 'right_bank_fit.tif', bank + 0.2),
        bank_pair_fit_elevation_path=_write_raster(river_dir / 'bank_pair_fit.tif', bank + 0.15),
        xs_support_elevation_path=_write_raster(river_dir / 'xs.tif', zeros),
        xs_support_weight_path=_write_raster(river_dir / 'xsw.tif', zeros),
        centerline_elevation_path=_write_raster(river_dir / 'center.tif', active),
        centerline_influence_path=_write_raster(river_dir / 'center_inf.tif', ones),
        longitudinal_profile_elevation_path=_write_raster(river_dir / 'long.tif', long),
        authoritative_support_mask_path=_write_raster(river_dir / 'support.tif', zeros, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / 'support_depth.tif', zeros),
        authoritative_bed_elevation_path=_write_raster(river_dir / 'auth_bed.tif', zeros),
        active_core_support_elevation_path=_write_raster(river_dir / 'active_core.tif', active),
    )

    station_targets = Path(outputs['station_targets'])
    station_targets_summary = Path(outputs['station_targets_summary'])
    assert station_targets.exists()
    assert station_targets_summary.exists()

    df = __import__('pandas').read_csv(station_targets)
    assert 'target_thalweg_z_m' in df.columns
    assert 'target_source_class' in df.columns
    assert 'xs_realism_allowed' in df.columns
    assert (df['target_source_class'].astype(str) == 'generalized_longitudinal_section').any()


def test_build_channel_frame_products_disable_xs_influence_suppresses_xs_classification(tmp_path: Path):
    river_dir = tmp_path / "river_disable_xs"
    river_dir.mkdir()
    centerline = gpd.GeoDataFrame(
        {"station_m": [0.0, 10.0], "component_id": ["a", "a"], "centerline_z_m": [-2.0, -2.4]},
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5)],
        crs="EPSG:4326",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    centerline.to_file(cpts, driver="GPKG")
    xs = gpd.GeoDataFrame({"xs_sample_source": ["authoritative"], "xs_z_m": [-2.1]}, geometry=[Point(0.5, 2.5)], crs="EPSG:4326")
    xsp = river_dir / "river_xs_support_points.gpkg"
    xs.to_file(xsp, driver="GPKG")

    center = np.array([[-2.0, -2.4], [np.nan, np.nan]], dtype=np.float32)
    xs_elev = np.array([[-2.1, -2.5], [np.nan, np.nan]], dtype=np.float32)
    xs_w = np.array([[0.8, 0.8], [0.0, 0.0]], dtype=np.float32)
    bank = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    zeros = np.zeros((2, 2), dtype=np.float32)
    ones = np.ones((2, 2), dtype=np.float32)

    outputs = build_channel_frame_products(
        river_dir=river_dir,
        centerline_points_path=cpts,
        xs_support_points_path=xsp,
        bank_elevation_path=_write_raster(river_dir / 'bank.tif', bank),
        bank_influence_path=_write_raster(river_dir / 'bank_inf.tif', ones),
        xs_support_elevation_path=_write_raster(river_dir / 'xs.tif', xs_elev),
        xs_support_weight_path=_write_raster(river_dir / 'xsw.tif', xs_w),
        centerline_elevation_path=_write_raster(river_dir / 'center.tif', center),
        centerline_influence_path=_write_raster(river_dir / 'center_inf.tif', ones),
        longitudinal_profile_elevation_path=_write_raster(river_dir / 'long.tif', center),
        authoritative_support_mask_path=_write_raster(river_dir / 'support.tif', zeros, nodata=0.0),
        authoritative_support_depth_path=_write_raster(river_dir / 'support_depth.tif', zeros),
        authoritative_bed_elevation_path=_write_raster(river_dir / 'auth_bed.tif', zeros),
        disable_xs_influence=True,
    )
    frame = gpd.read_file(outputs['channel_frame_points'])
    assert not frame['channel_support_class'].astype(str).eq('xs_supported').any()
    contract = json.loads(Path(outputs['channel_frame_contract']).read_text(encoding='utf-8'))
    assert contract['metrics']['xs_influence_disabled'] is True
    assert Path(outputs['anchor_table']).exists()
    station_targets = pd.read_csv(outputs['station_targets'])
    assert 'anchor_class' in station_targets.columns
