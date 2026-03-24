from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, Polygon

from river_structured_scaffold import (
    build_centerline_points,
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
