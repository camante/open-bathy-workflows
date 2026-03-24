from __future__ import annotations

from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from river_longitudinal_profile import build_and_write_longitudinal_profile


def _write_raster(path: Path, arr: np.ndarray, *, nodata: float = -9999.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=str(arr.dtype),
        crs="EPSG:32619",
        transform=from_origin(0, arr.shape[0], 1, 1),
        nodata=nodata,
    ) as ds:
        ds.write(arr, 1)
    return path


def test_build_and_write_longitudinal_profile_writes_profile_and_rasters(tmp_path: Path):
    river_dir = tmp_path / "river"
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "station_m": [0.0, 10.0, 20.0],
            "centerline_z_m": [5.0, 5.5, 6.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts.to_file(cpts, driver="GPKG")

    base = np.array([[np.nan, np.nan, np.nan], [5.0, 5.5, 6.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    ones = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", base),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[np.nan, np.nan, np.nan], [0.0, 10.0, 20.0], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", np.array([[np.nan, np.nan, np.nan], [4.8, 5.3, 5.8], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", np.array([[np.nan, np.nan, np.nan], [5.2, 5.7, 6.2], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        authoritative_support_depth_path=_write_raster(river_dir / "river_authoritative_support_depth.tif", np.array([[np.nan, np.nan, np.nan], [-1.0, -1.0, -1.0], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )
    assert Path(outputs["longitudinal_profile"]).exists()
    assert Path(outputs["longitudinal_profile_elevation"]).exists()
    assert Path(outputs["longitudinal_profile_uncertainty"]).exists()
    assert Path(outputs["hydraulic_backbone"]).exists()


def test_longitudinal_profile_uses_direct_authoritative_dem_anchors(tmp_path: Path):
    river_dir = tmp_path / "river"
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "station_m": [0.0, 10.0, 20.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts.to_file(cpts, driver="GPKG")

    ones = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    center = np.array([[5.0, 5.5, 6.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    auth_dem = np.array([[4.2, np.nan, 4.8], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    auth_support = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", center),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[np.nan, np.nan, np.nan], [0.0, 10.0, 20.0], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", center),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", center),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        authoritative_support_depth_path=_write_raster(river_dir / "river_authoritative_support_depth.tif", np.array([[np.nan, np.nan, np.nan], [-1.0, -1.0, -1.0], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        authoritative_bed_elevation_path=_write_raster(river_dir / "river_authoritative_bed.tif", auth_dem),
        authoritative_support_mask_path=_write_raster(river_dir / "river_authoritative_support.tif", auth_support),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )
    profile = __import__("pandas").read_csv(outputs["hydraulic_backbone"])
    assert np.isclose(profile.loc[profile["station_m"] == 0.0, "network_backbone_elevation_m"].iloc[0], 4.2)
    assert np.isclose(profile.loc[profile["station_m"] == 20.0, "network_backbone_elevation_m"].iloc[0], 4.8)


def test_hydraulic_backbone_nodes_use_component_specific_endpoints(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 2, 2],
            "station_m": [0.0, 10.0, 0.0, 10.0],
        },
        geometry=[Point(0.5, 3.5), Point(1.5, 3.5), Point(10.5, 1.5), Point(11.5, 1.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    base = np.full((4, 12), np.nan, dtype=np.float32)
    base[0, 0:2] = [5.0, 5.5]
    base[2, 10:12] = [2.0, 2.5]
    ones = np.where(np.isfinite(base), 1.0, 0.0).astype(np.float32)
    corr = np.where(np.isfinite(base), 1, 0).astype(np.uint8)

    edges = gpd.GeoDataFrame(
        {
            "component_id": [1, 2],
            "from_node": [100, 200],
            "to_node": [101, 201],
            "s_m_from": [0.0, 0.0],
            "s_m_to": [10.0, 10.0],
        },
        geometry=[Point(0, 0), Point(0, 0)],
        crs="EPSG:32619",
    )
    edges_path = river_dir / "network_edges.gpkg"
    edges.to_file(edges_path, driver="GPKG")

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", base),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", base),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", base),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", base),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
        network_edges_path=edges_path,
    )

    nodes = gpd.read_file(outputs["hydraulic_backbone_nodes"])
    comp1_start = nodes[(nodes["component_id"] == "1") & (nodes["position"] == "start")].iloc[0]
    comp2_start = nodes[(nodes["component_id"] == "2") & (nodes["position"] == "start")].iloc[0]
    assert np.isclose(comp1_start.geometry.x, 0.5)
    assert np.isclose(comp2_start.geometry.x, 10.5)
