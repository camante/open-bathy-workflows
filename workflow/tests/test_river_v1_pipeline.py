from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon

from river_v1_support import build_v1_support_products
from river_v1_backbone import build_v1_backbone_products
from river_v1_surface import build_v1_surface_products
from river_v1_pipeline import run_river_v1_stage


def _lonlat_from_utm(x, y):
    tx = Transformer.from_crs("EPSG:32619", "EPSG:4326", always_xy=True)
    return tx.transform(x, y)


def _write_dem(path: Path):
    arr = np.array([
        [10.0, 9.0, 8.0, 7.0, 6.0],
        [10.0, 9.0, 8.0, 7.0, 6.0],
        [10.0, 9.0, 8.0, 7.0, 6.0],
        [10.0, 9.0, 8.0, 7.0, 6.0],
        [10.0, 9.0, 8.0, 7.0, 6.0],
    ], dtype=np.float32)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype='float32',
        crs='EPSG:32619',
        transform=from_origin(0.0, 50.0, 10.0, 10.0),
        nodata=-9999.0,
    ) as ds:
        ds.write(arr, 1)


def _write_mask(path: Path):
    arr = np.zeros((5, 5), dtype=np.uint8)
    arr[:, 1:4] = 1
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype='uint8',
        crs='EPSG:32619',
        transform=from_origin(0.0, 50.0, 10.0, 10.0),
        nodata=0,
    ) as ds:
        ds.write(arr, 1)


def _write_network(path: Path):
    flows = gpd.GeoDataFrame(
        {
            'streamorde': [5],
            'lengthkm': [0.5],
            'component_id': ['main'],
            'from_node': [1],
            'to_node': [2],
        },
        geometry=[LineString([(15.0, 45.0), (35.0, 5.0)])],
        crs='EPSG:32619',
    )
    polys = gpd.GeoDataFrame({'id': [1]}, geometry=[Polygon([(5, 48), (45, 48), (45, 2), (5, 2)])], crs='EPSG:32619')
    flows.to_file(path, layer='rivers_clip', driver='GPKG')
    polys.to_file(path, layer='nhdarea_clip', driver='GPKG')


def _write_long_network(path: Path):
    flows = gpd.GeoDataFrame(
        {
            'streamorde': [5],
            'lengthkm': [1.1],
            'component_id': ['main'],
            'from_node': [1],
            'to_node': [2],
        },
        geometry=[LineString([(15.0, 45.0), (105.0, 45.0)])],
        crs='EPSG:32619',
    )
    polys = gpd.GeoDataFrame({'id': [1]}, geometry=[Polygon([(5, 48), (115, 48), (115, 2), (5, 2)])], crs='EPSG:32619')
    flows.to_file(path, layer='rivers_clip', driver='GPKG')
    polys.to_file(path, layer='nhdarea_clip', driver='GPKG')


class _Cfg:
    river_mainstem_min_order = 5
    river_centerline_sample_spacing_m = 10.0
    authoritative_support_density_radius_m = 25.0


def test_build_v1_support_products_classifies_and_writes(tmp_path: Path):
    dem = tmp_path / 'dem.tif'
    mask = tmp_path / 'mask.tif'
    gpkg = tmp_path / 'network.gpkg'
    river_dir = tmp_path / 'river'
    _write_dem(dem)
    _write_mask(mask)
    _write_network(gpkg)
    lon1, lat1 = _lonlat_from_utm(15.0, 45.0)
    lon2, lat2 = _lonlat_from_utm(25.0, 25.0)
    pts = pd.DataFrame({
        'lon': [lon1, lon2],
        'lat': [lat1, lat2],
        'depth_m': [4.0, 5.0],
        'authoritative_role': ['bed_support', 'bank_margin'],
        'source': ['a', 'b'],
    })
    def _loader(_cfg):
        return pts

    out = build_v1_support_products(
        cfg=_Cfg(),
        network_gpkg=gpkg,
        river_dem=dem,
        channel_mask_tif=mask,
        river_dir=river_dir,
        load_support_points_fn=_loader,
    )
    assert Path(out['support_points_path']).exists()
    gdf = gpd.read_file(out['support_points_path'])
    assert set(gdf['support_class']) == {'authoritative_interior', 'bank_margin_only'}


def test_build_v1_backbone_products_writes_adjusted_unsupported(tmp_path: Path):
    dem = tmp_path / 'dem.tif'
    mask = tmp_path / 'mask.tif'
    gpkg = tmp_path / 'network.gpkg'
    river_dir = tmp_path / 'river'
    _write_dem(dem)
    _write_mask(mask)
    _write_network(gpkg)
    river_dir.mkdir(parents=True, exist_ok=True)
    # Put one authoritative support near upstream only so unsupported points downstream must adjust.
    pts = gpd.GeoDataFrame(
        {
            'support_class': ['authoritative_interior'],
            'support_z_m': [2.0],
            'source': ['auth'],
            'authoritative_role': ['bed_support'],
        },
        geometry=gpd.points_from_xy([15.0], [45.0]),
        crs='EPSG:32619',
    )
    support_path = river_dir / 'river_support_points.gpkg'
    pts.to_file(support_path, driver='GPKG')
    out = build_v1_backbone_products(
        cfg=_Cfg(),
        network_gpkg=gpkg,
        river_dem=dem,
        channel_mask_tif=mask,
        river_dir=river_dir,
        support_points_path=support_path,
    )
    assert Path(out['centerline_points_path']).exists()
    assert out['backbone_adjusted_station_count'] == 0
    assert out['unsupported_noop_consistent_base_count'] > 0


def test_build_v1_backbone_products_allows_unsupported_noop_when_base_matches_interp(tmp_path: Path, monkeypatch):
    import river_v1_backbone as mod

    dem = tmp_path / "dem.tif"
    mask = tmp_path / "mask.tif"
    gpkg = tmp_path / "network.gpkg"
    river_dir = tmp_path / "river"
    _write_dem(dem)
    _write_mask(mask)
    _write_network(gpkg)
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "support_class": ["authoritative_interior"],
            "support_z_m": [9.0],
            "source": ["auth"],
            "authoritative_role": ["bed_support"],
        },
        geometry=gpd.points_from_xy([15.0], [45.0]),
        crs="EPSG:32619",
    )
    support_path = river_dir / "river_support_points.gpkg"
    pts.to_file(support_path, driver="GPKG")

    class _Sel:
        flows = gpd.GeoDataFrame({"component_id": ["main"]}, geometry=[LineString([(0, 0), (40, 0)])], crs="EPSG:32619")
        polygons = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(-1, -1), (41, -1), (41, 1), (-1, 1)])], crs="EPSG:32619")
        metadata = {"retained_flow_count": 1, "retained_polygon_count": 1}

    centerline = gpd.GeoDataFrame(
        {"component_id": ["main"] * 5, "station_m": [0.0, 10.0, 20.0, 30.0, 40.0]},
        geometry=gpd.points_from_xy([0, 10, 20, 30, 40], [0, 0, 0, 0, 0]),
        crs="EPSG:32619",
    )

    monkeypatch.setattr(mod, "select_retained_river_features", lambda *a, **k: _Sel())
    monkeypatch.setattr(mod, "build_centerline_points", lambda *a, **k: centerline.copy())
    monkeypatch.setattr(mod, "_sample_raster", lambda *a, **k: np.asarray([9.0, 8.0, 7.0, 6.0, 5.0], dtype=np.float32))
    monkeypatch.setattr(
        mod,
        "_nearest_support",
        lambda *a, **k: pd.DataFrame(
            {
                "support_class": ["authoritative_interior"] * 5,
                "support_z_m": [9.0, 8.0, 7.0, 6.0, 5.0],
                "nearest_support_distance_m": [0.0, 0.0, 30.0, 0.0, 0.0],
            },
            index=centerline.index,
        ),
    )

    out = build_v1_backbone_products(
        cfg=_Cfg(),
        network_gpkg=gpkg,
        river_dem=dem,
        channel_mask_tif=mask,
        river_dir=river_dir,
        support_points_path=support_path,
    )
    assert out["unsupported_centerline_count"] == 1
    assert out["unsupported_requested_adjustment_count"] == 0
    assert out["unsupported_noop_consistent_base_count"] == 1


def test_build_v1_surface_products_writes_primary_surface(tmp_path: Path):
    dem = tmp_path / "dem.tif"
    mask = tmp_path / "mask.tif"
    gpkg = tmp_path / "network.gpkg"
    river_dir = tmp_path / "river"
    _write_dem(dem)
    _write_mask(mask)
    _write_network(gpkg)
    river_dir.mkdir(parents=True, exist_ok=True)
    support = gpd.GeoDataFrame(
        {
            "support_class": ["authoritative_interior"],
            "support_z_m": [2.5],
            "source": ["auth"],
            "authoritative_role": ["bed_support"],
        },
        geometry=gpd.points_from_xy([15.0], [45.0]),
        crs="EPSG:32619",
    )
    support_path = river_dir / "river_support_points.gpkg"
    support.to_file(support_path, driver="GPKG")
    backbone = gpd.GeoDataFrame(
        {
            "component_id": ["main", "main", "main"],
            "station_m": [0.0, 20.0, 40.0],
            "backbone_z_m": [2.5, 5.0, 7.5],
            "support_class": ["authoritative_interior", "unsupported_interior", "unsupported_interior"],
        },
        geometry=gpd.points_from_xy([15.0, 25.0, 35.0], [45.0, 25.0, 5.0]),
        crs="EPSG:32619",
    )
    centerline_path = river_dir / "river_centerline_points.gpkg"
    backbone.to_file(centerline_path, driver="GPKG")
    out = build_v1_surface_products(
        cfg=_Cfg(),
        river_dem=dem,
        channel_mask_tif=mask,
        river_dir=river_dir,
        support_points_path=support_path,
        centerline_points_path=centerline_path,
    )
    assert Path(out["river_primary_surface_path"]).exists()
    with rasterio.open(out["river_primary_surface_path"]) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        finite = np.isfinite(arr) & (arr != nodata)
        assert int(np.count_nonzero(finite)) > 0
        assert np.isclose(arr[0, 1], 2.5, atol=1e-5)


def test_run_river_v1_stage_returns_surface_path(tmp_path: Path):
    dem = tmp_path / "dem.tif"
    mask = tmp_path / "mask.tif"
    gpkg = tmp_path / "network.gpkg"
    out_dir = tmp_path / "out"
    river_dir = out_dir / "river"
    work_dir = out_dir / "work"
    _write_dem(dem)
    _write_mask(mask)
    _write_network(gpkg)

    class _Cfg2(_Cfg):
        river_dem = dem
        strict = False

    lon1, lat1 = _lonlat_from_utm(15.0, 45.0)
    pts = pd.DataFrame({
        "lon": [lon1],
        "lat": [lat1],
        "depth_m": [2.0],
        "authoritative_role": ["bed_support"],
        "source": ["auth"],
    })

    def _loader(_cfg):
        return pts

    def _build_masks(_work_dir, strict=False):
        return mask, None, None

    report = {}
    surface_path = run_river_v1_stage(
        cfg=_Cfg2(),
        report=report,
        river_dir=river_dir,
        work_dir=work_dir,
        network_gpkg=gpkg,
        build_domain_masks_fn=_build_masks,
        load_support_points_fn=_loader,
        logger=type("L", (), {"info": lambda *a, **k: None})(),
    )
    assert Path(surface_path).exists()
    assert report["river"]["status"] == "success"
    assert report["river"]["execution_mode"] == "v1_minimal_surface"
    assert Path(report["river"]["outputs"]["river_primary_surface"]).exists()


def test_v1_pipeline_reports_primary_guidance_surface_output(tmp_path: Path):
    dem = tmp_path / "dem.tif"
    mask = tmp_path / "mask.tif"
    gpkg = tmp_path / "network.gpkg"
    out_dir = tmp_path / "out"
    river_dir = out_dir / "river"
    work_dir = out_dir / "work"
    _write_dem(dem)
    _write_mask(mask)
    _write_network(gpkg)

    class _Cfg2(_Cfg):
        river_dem = dem
        strict = False

    lon1, lat1 = _lonlat_from_utm(15.0, 45.0)
    pts = pd.DataFrame({
        "lon": [lon1],
        "lat": [lat1],
        "depth_m": [2.0],
        "authoritative_role": ["bed_support"],
        "source": ["auth"],
    })

    def _loader(_cfg):
        return pts

    def _build_masks(_work_dir, strict=False):
        return mask, None, None

    class Log:
        def info(self,*a,**k): pass
        def debug(self,*a,**k): pass

    report = {}
    out = run_river_v1_stage(
        cfg=_Cfg2(),
        report=report,
        river_dir=river_dir,
        work_dir=work_dir,
        network_gpkg=gpkg,
        build_domain_masks_fn=_build_masks,
        load_support_points_fn=_loader,
        logger=Log(),
    )
    assert Path(out).exists()
    assert report["river"]["outputs"]["primary_river_guidance_surface"] == str(out)



def test_build_v1_surface_products_respects_component_boundaries(tmp_path: Path):
    dem = tmp_path / "dem.tif"
    mask = tmp_path / "mask.tif"
    river_dir = tmp_path / "river"
    _write_dem(dem)
    _write_mask(mask)
    river_dir.mkdir(parents=True, exist_ok=True)

    support = gpd.GeoDataFrame(
        {
            "support_class": ["authoritative_interior", "authoritative_interior"],
            "support_z_m": [2.0, 8.0],
            "source": ["auth", "auth"],
            "authoritative_role": ["bed_support", "bed_support"],
        },
        geometry=gpd.points_from_xy([15.0, 35.0], [45.0, 45.0]),
        crs="EPSG:32619",
    )
    support_path = river_dir / "river_support_points.gpkg"
    support.to_file(support_path, driver="GPKG")

    backbone = gpd.GeoDataFrame(
        {
            "component_id": ["left", "left", "right", "right"],
            "station_m": [0.0, 40.0, 0.0, 40.0],
            "backbone_z_m": [2.0, 2.0, 8.0, 8.0],
            "support_class": [
                "authoritative_interior",
                "unsupported_interior",
                "authoritative_interior",
                "unsupported_interior",
            ],
        },
        geometry=gpd.points_from_xy([15.0, 15.0, 35.0, 35.0], [45.0, 5.0, 45.0, 5.0]),
        crs="EPSG:32619",
    )
    centerline_path = river_dir / "river_centerline_points.gpkg"
    backbone.to_file(centerline_path, driver="GPKG")

    out = build_v1_surface_products(
        cfg=_Cfg(),
        river_dem=dem,
        channel_mask_tif=mask,
        river_dir=river_dir,
        support_points_path=support_path,
        centerline_points_path=centerline_path,
    )
    assert out["component_constrained_surface"] is True
    with rasterio.open(out["river_primary_surface_path"]) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        finite = np.isfinite(arr) & (arr != nodata)
        assert int(np.count_nonzero(finite)) > 0
        # Left river column should stay near the left component value, right near the right component value.
        assert np.isclose(arr[0, 1], 2.0, atol=0.25)
        assert np.isclose(arr[0, 3], 8.0, atol=0.25)
        # Middle column should resolve to one side rather than blending across both components.
        assert abs(float(arr[0, 2]) - 5.0) > 1.0


def test_build_v1_backbone_products_bridges_missing_base_from_same_component_anchors(tmp_path: Path, monkeypatch):
    import river_v1_backbone as mod

    dem = tmp_path / "dem.tif"
    mask = tmp_path / "mask.tif"
    gpkg = tmp_path / "network.gpkg"
    river_dir = tmp_path / "river"
    _write_dem(dem)
    _write_mask(mask)
    _write_network(gpkg)
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "support_class": ["authoritative_interior", "authoritative_interior"],
            "support_z_m": [9.0, 5.0],
            "source": ["auth", "auth"],
            "authoritative_role": ["bed_support", "bed_support"],
        },
        geometry=gpd.points_from_xy([0.0, 40.0], [0.0, 0.0]),
        crs="EPSG:32619",
    )
    support_path = river_dir / "river_support_points.gpkg"
    pts.to_file(support_path, driver="GPKG")

    class _Sel:
        flows = gpd.GeoDataFrame({"component_id": ["main"]}, geometry=[LineString([(0, 0), (40, 0)])], crs="EPSG:32619")
        polygons = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(-1, -1), (41, -1), (41, 1), (-1, 1)])], crs="EPSG:32619")
        metadata = {"retained_flow_count": 1, "retained_polygon_count": 1}

    centerline = gpd.GeoDataFrame(
        {"component_id": ["main"] * 5, "station_m": [0.0, 10.0, 20.0, 30.0, 40.0]},
        geometry=gpd.points_from_xy([0, 10, 20, 30, 40], [0, 0, 0, 0, 0]),
        crs="EPSG:32619",
    )

    monkeypatch.setattr(mod, "select_retained_river_features", lambda *a, **k: _Sel())
    monkeypatch.setattr(mod, "build_centerline_points", lambda *a, **k: centerline.copy())
    monkeypatch.setattr(mod, "_sample_raster", lambda *a, **k: np.asarray([9.0, np.nan, np.nan, np.nan, 5.0], dtype=np.float32))
    monkeypatch.setattr(
        mod,
        "_nearest_support",
        lambda *a, **k: pd.DataFrame(
            {
                "support_class": ["authoritative_interior", None, None, None, "authoritative_interior"],
                "support_z_m": [9.0, np.nan, np.nan, np.nan, 5.0],
                "nearest_support_distance_m": [0.0, 999.0, 999.0, 999.0, 0.0],
            },
            index=centerline.index,
        ),
    )

    out = build_v1_backbone_products(
        cfg=_Cfg(),
        network_gpkg=gpkg,
        river_dem=dem,
        channel_mask_tif=mask,
        river_dir=river_dir,
        support_points_path=support_path,
    )
    assert out["unsupported_centerline_count"] == 3
    assert out["unsupported_missing_base_count"] == 3
    assert out["unsupported_bridged_from_anchors_count"] == 3
    assert out["unsupported_unbridgeable_missing_base_count"] == 0
    assert out["backbone_adjusted_station_count"] == 3


def test_build_v1_backbone_products_bridges_from_centerline_base_when_support_anchors_sparse(tmp_path: Path, monkeypatch):
    import river_v1_backbone as mod

    dem = tmp_path / "dem.tif"
    mask = tmp_path / "mask.tif"
    gpkg = tmp_path / "network.gpkg"
    river_dir = tmp_path / "river"
    _write_dem(dem)
    _write_mask(mask)
    _write_network(gpkg)
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "support_class": ["authoritative_interior"],
            "support_z_m": [9.0],
            "source": ["auth"],
            "authoritative_role": ["bed_support"],
        },
        geometry=gpd.points_from_xy([0.0], [0.0]),
        crs="EPSG:32619",
    )
    support_path = river_dir / "river_support_points.gpkg"
    pts.to_file(support_path, driver="GPKG")

    class _Sel:
        flows = gpd.GeoDataFrame({"component_id": ["main"]}, geometry=[LineString([(0, 0), (40, 0)])], crs="EPSG:32619")
        polygons = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(-1, -1), (41, -1), (41, 1), (-1, 1)])], crs="EPSG:32619")
        metadata = {"retained_flow_count": 1, "retained_polygon_count": 1}

    centerline = gpd.GeoDataFrame(
        {"component_id": ["main"] * 5, "station_m": [0.0, 10.0, 20.0, 30.0, 40.0]},
        geometry=gpd.points_from_xy([0, 10, 20, 30, 40], [0, 0, 0, 0, 0]),
        crs="EPSG:32619",
    )

    monkeypatch.setattr(mod, "select_retained_river_features", lambda *a, **k: _Sel())
    monkeypatch.setattr(mod, "build_centerline_points", lambda *a, **k: centerline.copy())
    monkeypatch.setattr(mod, "_sample_raster", lambda *a, **k: np.asarray([9.0, 8.0, np.nan, 6.0, 5.0], dtype=np.float32))
    monkeypatch.setattr(
        mod,
        "_nearest_support",
        lambda *a, **k: pd.DataFrame(
            {
                "support_class": ["authoritative_interior", None, None, None, None],
                "support_z_m": [9.0, np.nan, np.nan, np.nan, np.nan],
                "nearest_support_distance_m": [0.0, 50.0, 50.0, 50.0, 50.0],
            },
            index=centerline.index,
        ),
    )

    out = build_v1_backbone_products(
        cfg=_Cfg(),
        network_gpkg=gpkg,
        river_dem=dem,
        channel_mask_tif=mask,
        river_dir=river_dir,
        support_points_path=support_path,
    )
    assert out["unsupported_centerline_count"] == 4
    assert out["unsupported_missing_base_count"] == 1
    assert out["unsupported_bridged_from_anchors_count"] == 1
    assert out["unsupported_unbridgeable_missing_base_count"] == 0
    assert out["centerline_base_anchor_count"] == 4


def test_build_v1_backbone_products_fails_when_missing_base_has_no_same_component_anchor_pair(tmp_path: Path, monkeypatch):
    import river_v1_backbone as mod

    dem = tmp_path / "dem.tif"
    mask = tmp_path / "mask.tif"
    gpkg = tmp_path / "network.gpkg"
    river_dir = tmp_path / "river"
    _write_dem(dem)
    _write_mask(mask)
    _write_network(gpkg)
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "support_class": ["authoritative_interior"],
            "support_z_m": [9.0],
            "source": ["auth"],
            "authoritative_role": ["bed_support"],
        },
        geometry=gpd.points_from_xy([0.0], [0.0]),
        crs="EPSG:32619",
    )
    support_path = river_dir / "river_support_points.gpkg"
    pts.to_file(support_path, driver="GPKG")

    class _Sel:
        flows = gpd.GeoDataFrame({"component_id": ["main"]}, geometry=[LineString([(0, 0), (40, 0)])], crs="EPSG:32619")
        polygons = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(-1, -1), (41, -1), (41, 1), (-1, 1)])], crs="EPSG:32619")
        metadata = {"retained_flow_count": 1, "retained_polygon_count": 1}

    centerline = gpd.GeoDataFrame(
        {"component_id": ["main"] * 5, "station_m": [0.0, 10.0, 20.0, 30.0, 40.0]},
        geometry=gpd.points_from_xy([0, 10, 20, 30, 40], [0, 0, 0, 0, 0]),
        crs="EPSG:32619",
    )

    monkeypatch.setattr(mod, "select_retained_river_features", lambda *a, **k: _Sel())
    monkeypatch.setattr(mod, "build_centerline_points", lambda *a, **k: centerline.copy())
    monkeypatch.setattr(mod, "_sample_raster", lambda *a, **k: np.asarray([9.0, np.nan, np.nan, np.nan, np.nan], dtype=np.float32))
    monkeypatch.setattr(
        mod,
        "_nearest_support",
        lambda *a, **k: pd.DataFrame(
            {
                "support_class": ["authoritative_interior", None, None, None, None],
                "support_z_m": [9.0, np.nan, np.nan, np.nan, np.nan],
                "nearest_support_distance_m": [0.0, 999.0, 999.0, 999.0, 999.0],
            },
            index=centerline.index,
        ),
    )

    try:
        build_v1_backbone_products(
            cfg=_Cfg(),
            network_gpkg=gpkg,
            river_dem=dem,
            channel_mask_tif=mask,
            river_dir=river_dir,
            support_points_path=support_path,
        )
    except RuntimeError as exc:
        assert str(exc) == "river_v1_backbone_unsupported_interior_inert"
    else:
        raise AssertionError("expected RuntimeError for unbridgeable unsupported missing-base backbone")
