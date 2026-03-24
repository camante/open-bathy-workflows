from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

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


def test_network_backbone_reduces_junction_disagreement(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 2, 2],
            "station_m": [0.0, 10.0, 0.0, 10.0],
            "centerline_z_m": [5.0, 4.0, 6.0, 4.8],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(0.5, 1.5), Point(1.5, 1.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    center = np.array([[5.0, 4.0, np.nan], [6.0, 4.8, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    ones = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    station = np.array([[0.0, 10.0, np.nan], [0.0, 10.0, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)

    edges = gpd.GeoDataFrame(
        {
            "component_id": [1, 2],
            "from_node": [100, 200],
            "to_node": [300, 300],
            "s_m_from": [0.0, 0.0],
            "s_m_to": [10.0, 10.0],
        },
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(0, 1), (1, 0)])],
        crs="EPSG:32619",
    )
    edges_path = river_dir / "scaffold_graph_edges.gpkg"
    edges.to_file(edges_path, driver="GPKG")

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", center),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", station),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", center),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", center),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
        network_edges_path=edges_path,
    )

    prof = pd.read_csv(outputs["longitudinal_profile"])
    ends = prof.sort_values(["profile_id", "station_m"]).groupby("profile_id").tail(1)
    raw_gap = abs(float(ends["bed_elevation_m"].iloc[0]) - float(ends["bed_elevation_m"].iloc[1]))
    solved_gap = abs(float(ends["network_backbone_elevation_m"].iloc[0]) - float(ends["network_backbone_elevation_m"].iloc[1]))
    assert solved_gap < raw_gap
    summary = pd.read_json(river_dir / "river_longitudinal_profile_summary.json", typ="series")
    assert bool(summary["network_aware"]) is True


def test_network_backbone_flow_aware_junction_prefers_mainstem(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 2, 2],
            "station_m": [0.0, 10.0, 0.0, 10.0],
            "centerline_z_m": [5.0, 4.0, 6.0, 4.8],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(0.5, 1.5), Point(1.5, 1.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    center = np.array([[5.0, 4.0, np.nan], [6.0, 4.8, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    ones = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    station = np.array([[0.0, 10.0, np.nan], [0.0, 10.0, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)

    edges = gpd.GeoDataFrame(
        {
            "component_id": [1, 2],
            "from_node": [100, 200],
            "to_node": [300, 300],
            "s_m_from": [0.0, 0.0],
            "s_m_to": [10.0, 10.0],
            "drainage_area_sqkm": [1200.0, 25.0],
            "stream_order": [5, 2],
        },
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(0, 1), (1, 0)])],
        crs="EPSG:32619",
    )
    edges_path = river_dir / "scaffold_graph_edges.gpkg"
    edges.to_file(edges_path, driver="GPKG")

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", center),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", station),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", center),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", center),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
        network_edges_path=edges_path,
    )

    prof = pd.read_csv(outputs["hydraulic_backbone"])
    ends = prof.sort_values(["profile_id", "station_m"]).groupby("profile_id").tail(1).set_index("profile_id")
    target = float(np.average(ends["network_backbone_elevation_m"], weights=ends["junction_hierarchy_weight"]))
    idx1 = 1 if 1 in ends.index else "1"
    idx2 = 2 if 2 in ends.index else "2"
    main_end = float(ends.loc[idx1, "network_backbone_elevation_m"])
    trib_end = float(ends.loc[idx2, "network_backbone_elevation_m"])
    assert abs(main_end - target) < abs(trib_end - target)
    assert float(ends.loc[idx1, "junction_hierarchy_weight"]) > float(ends.loc[idx2, "junction_hierarchy_weight"])


def test_network_backbone_wse_aware_junction_downweights_stage_outlier(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 2, 2, 3, 3],
            "station_m": [0.0, 10.0, 0.0, 10.0, 0.0, 10.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(0.5, 1.5), Point(1.5, 1.5), Point(0.5, 0.5), Point(1.5, 0.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    center = np.array([[5.0, 4.2, np.nan], [5.1, 4.3, np.nan], [6.5, 5.9, np.nan]], dtype=np.float32)
    ones = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    corr = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=np.uint8)
    station = np.array([[0.0, 10.0, np.nan], [0.0, 10.0, np.nan], [0.0, 10.0, np.nan]], dtype=np.float32)
    wse = np.array([[10.0, 9.9, np.nan], [10.1, 10.0, np.nan], [14.0, 13.9, np.nan]], dtype=np.float32)

    edges = gpd.GeoDataFrame(
        {
            "component_id": [1, 2, 3],
            "from_node": [100, 200, 400],
            "to_node": [300, 300, 300],
            "s_m_from": [0.0, 0.0, 0.0],
            "s_m_to": [10.0, 10.0, 10.0],
            "drainage_area_sqkm": [500.0, 450.0, 200.0],
            "stream_order": [4, 4, 3],
        },
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(0, 1), (1, 0)]), LineString([(0, 2), (1, 0)])],
        crs="EPSG:32619",
    )
    edges_path = river_dir / "scaffold_graph_edges.gpkg"
    edges.to_file(edges_path, driver="GPKG")

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", center),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", station),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", center),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", center),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        wse_elevation_path=_write_raster(river_dir / "wse_m.tif", wse),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
        network_edges_path=edges_path,
    )

    prof = pd.read_csv(outputs["hydraulic_backbone"])
    ends = prof.sort_values(["profile_id", "station_m"]).groupby("profile_id").tail(1).set_index("profile_id")
    idx1 = 1 if 1 in ends.index else "1"
    idx2 = 2 if 2 in ends.index else "2"
    idx3 = 3 if 3 in ends.index else "3"
    assert float(ends.loc[idx3, "junction_wse_weight"]) < float(ends.loc[idx1, "junction_wse_weight"])
    assert float(ends.loc[idx3, "junction_wse_weight"]) < float(ends.loc[idx2, "junction_wse_weight"])
    target = float(np.average(ends["network_backbone_elevation_m"], weights=ends["junction_hierarchy_weight"] * ends["junction_wse_weight"]))
    consistent_mean = float(np.mean([ends.loc[idx1, "network_backbone_elevation_m"], ends.loc[idx2, "network_backbone_elevation_m"]]))
    assert abs(target - consistent_mean) < abs(target - float(ends.loc[idx3, "network_backbone_elevation_m"]))
