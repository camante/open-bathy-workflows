from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from simple_river_authoritative_bed_stage import (
    build_centerline_authoritative_bed_points,
    centerline_authoritative_bed_field_schema,
    validate_centerline_authoritative_bed,
)


def test_authoritative_bed_field_schema_has_required_fields():
    schema = centerline_authoritative_bed_field_schema()
    assert schema["point_id"] == "string"
    assert schema["authoritative_bed_z_m"] == "float64"


def test_validate_centerline_authoritative_bed_accepts_minimal_valid_gdf():
    gdf = gpd.GeoDataFrame({
        "point_id": ["a", "b"],
        "station_m": [0.0, 1.0],
        "authoritative_bed_z_m": [1.0, 0.5],
    }, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    out = validate_centerline_authoritative_bed(gdf)
    assert out["valid"] is True


def test_build_centerline_authoritative_bed_points_from_centerline_field(tmp_path):
    gdf = gpd.GeoDataFrame({
        "point_id": ["a", "b"],
        "station_m": [0.0, 1.0],
        "centerline_z_m": [1.0, 0.5],
    }, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    out = build_centerline_authoritative_bed_points(
        river_context={"centerline_points_gdf": gdf},
        out_path=str(tmp_path / "centerline_authoritative_bed_points.gpkg"),
        receipt_path=str(tmp_path / "centerline_authoritative_bed_points_receipt.json"),
    )
    assert Path(out["output_artifact"]).exists()
    assert Path(out["receipt_path"]).exists()
    written = gpd.read_file(out["output_artifact"])
    assert "authoritative_bed_z_m" in written.columns
