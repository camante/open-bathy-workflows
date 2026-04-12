from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from simple_river_observed_offset_stage import (
    build_centerline_observed_offset_points,
    centerline_observed_offset_field_schema,
    validate_observed_offsets,
)


def test_observed_offset_field_schema_has_required_fields():
    schema = centerline_observed_offset_field_schema()
    assert schema["point_id"] == "string"
    assert schema["observed_offset_m"] == "float64"


def test_validate_observed_offsets_accepts_minimal_valid_gdf():
    gdf = gpd.GeoDataFrame({
        "point_id": ["a", "b"],
        "station_m": [0.0, 1.0],
        "wse_proxy_z_m": [3.0, 2.0],
        "authoritative_bed_z_m": [1.0, 0.5],
        "observed_offset_m": [2.0, 1.5],
        "offset_source": ["x", "x"],
        "offset_valid": [True, True],
    }, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    out = validate_observed_offsets(gdf)
    assert out["valid"] is True


def test_build_centerline_observed_offset_points_from_stage_inputs(tmp_path):
    wse = gpd.GeoDataFrame({
        "point_id": ["a", "b"],
        "station_m": [0.0, 1.0],
        "wse_proxy_z_m": [3.0, 2.0],
        "wse_proxy_source": ["x", "x"],
        "wse_proxy_method": ["x", "x"],
    }, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    bed = gpd.GeoDataFrame({
        "point_id": ["a", "b"],
        "station_m": [0.0, 1.0],
        "authoritative_bed_z_m": [1.0, 0.5],
        "authoritative_bed_source": ["x", "x"],
        "authoritative_bed_method": ["x", "x"],
    }, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    out = build_centerline_observed_offset_points(
        river_context={"wse_points_gdf": wse, "authoritative_bed_gdf": bed},
        out_path=str(tmp_path / "centerline_observed_offset_points.gpkg"),
        receipt_path=str(tmp_path / "centerline_observed_offset_points_receipt.json"),
    )
    assert Path(out["output_artifact"]).exists()
    assert Path(out["receipt_path"]).exists()
    written = gpd.read_file(out["output_artifact"])
    assert "observed_offset_m" in written.columns
    assert list(written["observed_offset_m"]) == [2.0, 1.5]
