from pathlib import Path

import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, Polygon

from river_structured_scaffold import (
    build_centerline_points,
    validate_centerline_station_component_contract,
    build_dense_bank_points,
    build_xs_support_points,
    select_retained_river_features,
)


def test_select_retained_river_features_prunes_to_mainstem_and_key_tributaries(tmp_path: Path):
    gpkg = tmp_path / "river_network.gpkg"
    flows = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 2, 3],
            "streamorde": [5, 4, 2, 1],
            "lengthkm": [2.0, 1.0, 0.2, 0.05],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (180, 40)]),
            LineString([(50, -20), (70, -10)]),
            LineString([(10, 50), (15, 55)]),
        ],
        crs="EPSG:32619",
    )
    polys = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Polygon([(-5, -10), (190, -10), (190, 60), (-5, 60)])],
        crs="EPSG:32619",
    )
    flows.to_file(gpkg, layer="rivers_clip", driver="GPKG")
    polys.to_file(gpkg, layer="nhdarea_clip", driver="GPKG")

    sel = select_retained_river_features(gpkg, target_crs="EPSG:32619", min_stream_order=3, min_length_km=0.25, keep_top_components=2)
    assert len(sel.flows) == 2
    assert set(sel.flows["component_id"]) == {1}
    assert len(sel.polygons) == 1
    assert sel.metadata["selection_mode"] == "mainstem_and_key_tributaries"


def test_structured_scaffold_points_are_polygon_bounded(tmp_path: Path):
    poly = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(20, 20), (180, 20), (180, 120), (20, 120)])], crs="EPSG:32619")
    flows = gpd.GeoDataFrame({"component_id": [1], "streamorde": [5]}, geometry=[LineString([(30, 70), (170, 70)])], crs="EPSG:32619")
    xs = gpd.GeoDataFrame({"id": [1]}, geometry=[LineString([(60, 30), (60, 110)])], crs="EPSG:32619")
    xs_gpkg = tmp_path / "xs.gpkg"
    xs.to_file(xs_gpkg, layer="xs_lines", driver="GPKG")

    bank = build_dense_bank_points(poly, spacing_m=25.0)
    center = build_centerline_points(flows, poly, spacing_m=30.0)
    xs_pts = build_xs_support_points(xs_gpkg, poly)

    assert not bank.empty
    assert not center.empty
    assert not xs_pts.empty
    assert set(bank["artifact_role"]) == {"bank_boundary_control"}
    assert set(center["artifact_role"]) == {"longitudinal_control"}
    assert set(xs_pts["artifact_role"]) == {"cross_stream_control"}


def _write_test_raster(path: Path, data: np.ndarray, transform, crs="EPSG:32619"):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as ds:
        ds.write(data.astype("float32"), 1)


def test_dense_bank_points_recover_from_nodata_with_normal_search(tmp_path: Path):
    poly = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(2, 2), (18, 2), (18, 18), (2, 18)])], crs="EPSG:32619")
    arr = np.ones((20, 20), dtype=np.float32) * 5.0
    arr[2, 2:19] = -9999.0
    tif = tmp_path / "bed.tif"
    _write_test_raster(tif, arr, from_origin(0, 20, 1, 1))

    bank = build_dense_bank_points(poly, spacing_m=4.0, raster_path=tif, normal_search_max_m=3.0, normal_search_step_m=1.0)
    assert not bank.empty
    assert np.isfinite(bank["bank_z_m"]).all()
    assert {"normal_search_inward", "normal_search_outward"}.intersection(set(bank["bank_sample_status"]))
    assert "bank_confidence" in bank.columns


def test_dense_bank_points_without_valid_samples_leave_missing_status(tmp_path: Path):
    poly = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(2, 2), (18, 2), (18, 18), (2, 18)])], crs="EPSG:32619")
    arr = np.ones((20, 20), dtype=np.float32) * -9999.0
    tif = tmp_path / "bed2.tif"
    _write_test_raster(tif, arr, from_origin(0, 20, 1, 1))

    bank = build_dense_bank_points(poly, spacing_m=6.0, raster_path=tif, normal_search_max_m=1.5, normal_search_step_m=1.0)
    assert not bank.empty
    assert not np.isfinite(bank["bank_z_m"]).any()
    assert set(bank["bank_sample_status"]) == {"missing"}


def test_build_xs_support_points_requires_xs_lines_layer(tmp_path: Path):
    poly = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(20, 20), (180, 20), (180, 120), (20, 120)])], crs="EPSG:32619")
    xs = gpd.GeoDataFrame({"id": [1]}, geometry=[LineString([(60, 30), (60, 110)])], crs="EPSG:32619")
    xs_gpkg = tmp_path / "xs_missing_lines.gpkg"
    xs.to_file(xs_gpkg, layer="wrong_layer", driver="GPKG")

    import pytest
    with pytest.raises(RuntimeError, match="xs_lines"):
        build_xs_support_points(xs_gpkg, poly)


def test_build_xs_support_points_prefers_authoritative_raster_and_leaves_unsupported_missing_by_default(tmp_path: Path):
    poly = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])], crs="EPSG:32619")
    xs = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[
            LineString([(5, 3), (5, 27)]),
            LineString([(25, 3), (25, 27)]),
        ],
        crs="EPSG:32619",
    )
    xs_gpkg = tmp_path / "xs_pref_auth.gpkg"
    xs.to_file(xs_gpkg, layer="xs_lines", driver="GPKG")

    transform = from_origin(0, 30, 1, 1)
    auth = np.full((30, 30), -9999.0, dtype=np.float32)
    auth[:, 4:7] = 11.0
    fallback = np.full((30, 30), 22.0, dtype=np.float32)
    auth_tif = tmp_path / "auth.tif"
    fallback_tif = tmp_path / "fallback.tif"
    _write_test_raster(auth_tif, auth, transform)
    _write_test_raster(fallback_tif, fallback, transform)

    xs_pts = build_xs_support_points(
        xs_gpkg,
        poly,
        preferred_raster_path=auth_tif,
        raster_path=fallback_tif,
        spacing_m=6.0,
    )

    assert not xs_pts.empty
    left = xs_pts[np.isclose(xs_pts.geometry.x.to_numpy(dtype=float), 5.0)]
    right = xs_pts[np.isclose(xs_pts.geometry.x.to_numpy(dtype=float), 25.0)]
    assert not left.empty and not right.empty
    assert set(left["xs_sample_source"]) == {"authoritative"}
    assert set(right["xs_sample_source"]) == {"missing"}
    assert np.allclose(left["xs_z_m"].to_numpy(dtype=float), 11.0)
    assert not np.isfinite(right["xs_z_m"].to_numpy(dtype=float)).any()


def test_build_xs_support_points_can_explicitly_enable_inferred_fallback(tmp_path: Path):
    poly = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])], crs="EPSG:32619")
    xs = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[
            LineString([(5, 3), (5, 27)]),
            LineString([(25, 3), (25, 27)]),
        ],
        crs="EPSG:32619",
    )
    xs_gpkg = tmp_path / "xs_pref_auth_fallback_optin.gpkg"
    xs.to_file(xs_gpkg, layer="xs_lines", driver="GPKG")

    transform = from_origin(0, 30, 1, 1)
    auth = np.full((30, 30), -9999.0, dtype=np.float32)
    auth[:, 4:7] = 11.0
    fallback = np.full((30, 30), 22.0, dtype=np.float32)
    auth_tif = tmp_path / "auth_optin.tif"
    fallback_tif = tmp_path / "fallback_optin.tif"
    _write_test_raster(auth_tif, auth, transform)
    _write_test_raster(fallback_tif, fallback, transform)

    xs_pts = build_xs_support_points(
        xs_gpkg,
        poly,
        preferred_raster_path=auth_tif,
        raster_path=fallback_tif,
        allow_inferred_fallback=True,
        spacing_m=6.0,
    )

    left = xs_pts[np.isclose(xs_pts.geometry.x.to_numpy(dtype=float), 5.0)]
    right = xs_pts[np.isclose(xs_pts.geometry.x.to_numpy(dtype=float), 25.0)]
    assert set(left["xs_sample_source"]) == {"authoritative"}
    assert set(right["xs_sample_source"]) == {"fallback_cached_bed"}
    assert np.allclose(right["xs_z_m"].to_numpy(dtype=float), 22.0)


def test_sample_raster_values_reprojects_points_to_raster_crs(tmp_path: Path):
    from pyproj import Transformer

    arr = np.full((20, 20), -9999.0, dtype=np.float32)
    arr[10, 10] = 7.5
    transform = from_origin(331000, 4650000, 100, 100)
    tif = tmp_path / "utm.tif"
    _write_test_raster(tif, arr, transform, crs="EPSG:32619")
    lon, lat = Transformer.from_crs("EPSG:32619", "EPSG:4326", always_xy=True).transform(332050, 4648950)
    pts = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(lon, lat)], crs="EPSG:4326")

    from river_structured_scaffold import _sample_raster_values

    sampled = _sample_raster_values(tif, pts.copy(), field="z_m")
    assert np.isfinite(sampled["z_m"]).any()


def test_nearest_surface_from_points_builds_surface(tmp_path: Path):
    import geopandas as gpd
    from river_structured_scaffold import _nearest_surface_from_points

    transform = from_origin(0, 100, 10, 10)
    shape = (10, 10)
    domain_mask = np.zeros(shape, dtype=bool)
    domain_mask[2:8, 2:8] = True
    pts = gpd.GeoDataFrame(
        {"z_m": [1.0, 2.0]},
        geometry=[Point(25, 75), Point(65, 35)],
        crs="EPSG:32619",
    )
    surf, infl = _nearest_surface_from_points(
        shape=shape,
        transform=transform,
        domain_mask=domain_mask,
        points_gdf=pts,
        value_field="z_m",
        max_distance_m=100.0,
    )

    assert np.isfinite(surf[domain_mask]).any()
    assert np.nanmax(infl) > 0.0
    assert np.isnan(surf[~domain_mask]).all()


def test_select_retained_river_features_rescues_short_mainstem_bridge_segments(tmp_path: Path):
    gpkg = tmp_path / "river_network_bridge.gpkg"
    flows = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "streamorde": [8, 8, 8],
            "lengthkm": [0.60, 0.245, 0.55],
            "from_node": [10, 20, 30],
            "to_node": [20, 30, 40],
            "gnis_name": ["Merrimack River", "Merrimack River", "Merrimack River"],
        },
        geometry=[
            LineString([(0, 0), (60, 0)]),
            LineString([(60, 0), (84.5, 0)]),
            LineString([(84.5, 0), (139.5, 0)]),
        ],
        crs="EPSG:32619",
    )
    polys = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Polygon([(-5, -10), (145, -10), (145, 10), (-5, 10)])],
        crs="EPSG:32619",
    )
    flows.to_file(gpkg, layer="rivers_clip", driver="GPKG")
    polys.to_file(gpkg, layer="nhdarea_clip", driver="GPKG")

    sel = select_retained_river_features(gpkg, target_crs="EPSG:32619", min_stream_order=3, min_length_km=0.25, keep_top_components=1)
    assert len(sel.flows) == 3
    assert sel.metadata["rescued_short_bridge_segments"] == 1



def test_build_centerline_points_promotes_component_granularity_from_levelpath():
    poly = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(-5, -10), (205, -10), (205, 110), (-5, 110)])], crs="EPSG:32619")
    flows = gpd.GeoDataFrame(
        {
            "component_id": [1, 1],
            "levelpathi": [101, 202],
            "streamorde": [5, 4],
            "s_m_from": [0.0, 0.0],
            "s_m_to": [100.0, 100.0],
        },
        geometry=[
            LineString([(0, 20), (100, 20)]),
            LineString([(0, 80), (100, 80)]),
        ],
        crs="EPSG:32619",
    )

    center = build_centerline_points(flows, poly, spacing_m=50.0)

    assert not center.empty
    assert center["component_id"].nunique() == 2
    assert set(center["component_id"].astype(str)) == {"levelpathi:101", "levelpathi:202"}
    assert center[["component_id", "station_m"]].drop_duplicates().shape[0] == len(center)
    assert center.attrs["station_contract"]["component_count_before"] == 1
    assert center.attrs["station_contract"]["component_count_after"] == 2



def test_build_centerline_points_collapses_same_levelpath_boundary_duplicates():
    poly = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(-5, -10), (205, -10), (205, 40), (-5, 40)])], crs="EPSG:32619")
    flows = gpd.GeoDataFrame(
        {
            "component_id": [1, 1],
            "levelpathi": [101, 101],
            "streamorde": [5, 5],
            "s_m_from": [0.0, 100.0],
            "s_m_to": [100.0, 200.0],
        },
        geometry=[
            LineString([(0, 20), (100, 20)]),
            LineString([(100, 20), (200, 20)]),
        ],
        crs="EPSG:32619",
    )

    center = build_centerline_points(flows, poly, spacing_m=50.0)

    stations = np.sort(pd.to_numeric(center["station_m"], errors="coerce").to_numpy(dtype=float))
    assert stations.tolist() == [0.0, 50.0, 100.0, 150.0, 200.0]
    assert center[["component_id", "station_m"]].drop_duplicates().shape[0] == len(center)
    assert center.attrs["station_contract"]["duplicate_component_station_rows_collapsed"] == 0
    assert center.attrs["station_contract"]["component_axis_stationization_used"] == 1



def test_build_centerline_points_fallback_stationization_uses_component_axis_offsets_when_station_fields_missing():
    poly = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(-5, -10), (305, -10), (305, 40), (-5, 40)])], crs="EPSG:32619")
    flows = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "levelpathi": [101, 101, 101],
            "streamorde": [5, 5, 5],
            "fromnode": [10, 11, 12],
            "tonode": [11, 12, 13],
        },
        geometry=[
            LineString([(0, 20), (100, 20)]),
            LineString([(100, 20), (200, 20)]),
            LineString([(200, 20), (300, 20)]),
        ],
        crs="EPSG:32619",
    )

    center = build_centerline_points(flows, poly, spacing_m=50.0)

    stations = np.sort(pd.to_numeric(center["station_m"], errors="coerce").to_numpy(dtype=float))
    assert stations.tolist() == [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
    assert center[["component_id", "station_m"]].duplicated().sum() == 0
    contract = center.attrs["station_contract"]
    assert contract["duplicate_component_station_rows_collapsed"] == 0
    assert contract["fallback_stationization_method_counts"].get("topology_chain", 0) == 1
    assert contract["duplicate_component_station_fraction"] == 0.0


def test_validate_centerline_station_component_contract_raises_on_coarse_single_component_namespace():
    import geopandas as gpd
    from shapely.geometry import Point

    center = gpd.GeoDataFrame(
        {
            "component_id": ["main", "main"],
            "station_m": [0.0, 50.0],
        },
        geometry=[Point(0, 0), Point(50, 0)],
        crs="EPSG:32619",
    )

    with pytest.raises(RuntimeError, match="coarse_component_namespace"):
        validate_centerline_station_component_contract(
            center,
            expected_component_count=2,
            expected_component_source="levelpathi",
            context="retained_centerline_points",
        )


def test_build_centerline_points_component_axis_stationization_avoids_overlapping_duplicate_rows():
    poly = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(-5, -10), (205, -10), (205, 40), (-5, 40)])], crs="EPSG:32619")
    flows = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "levelpathi": [101, 101, 101],
            "streamorde": [5, 5, 5],
            "s_m_from": [0.0, 50.0, 100.0],
            "s_m_to": [100.0, 150.0, 200.0],
        },
        geometry=[
            LineString([(0, 20), (100, 20)]),
            LineString([(50, 20), (150, 20)]),
            LineString([(100, 20), (200, 20)]),
        ],
        crs="EPSG:32619",
    )

    center = build_centerline_points(flows, poly, spacing_m=50.0)

    stations = np.sort(pd.to_numeric(center["station_m"], errors="coerce").to_numpy(dtype=float))
    assert stations.tolist() == [0.0, 50.0, 100.0, 150.0, 200.0]
    assert center[["component_id", "station_m"]].duplicated().sum() == 0
    assert center.attrs["station_contract"]["duplicate_component_station_rows_collapsed"] == 0
    assert center.attrs["station_contract"]["component_axis_stationization_used"] == 1
