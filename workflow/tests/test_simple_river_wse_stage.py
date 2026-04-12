from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from simple_river_wse_stage import centerline_wse_field_schema, validate_centerline_wse_proxy, build_centerline_wse_proxy_points


def _gdf():
    return gpd.GeoDataFrame({
        "point_id": ["a", "b"],
        "station_m": [0.0, 10.0],
        "bank_wse_proxy_monotone_m": [5.0, 4.0],
    }, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")


def test_centerline_wse_field_schema_has_required_fields():
    schema = centerline_wse_field_schema()
    assert set(["point_id", "station_m", "wse_proxy_z_m", "geometry"]).issubset(schema.keys())


def test_validate_centerline_wse_proxy_accepts_minimal_valid_gdf():
    gdf = _gdf().rename(columns={"bank_wse_proxy_monotone_m": "wse_proxy_z_m"})
    gdf["wse_proxy_source"] = "x"
    gdf["wse_proxy_method"] = "x"
    out = validate_centerline_wse_proxy(gdf)
    assert out["valid"] is True


def test_build_centerline_wse_proxy_points_from_direct_column(tmp_path: Path):
    out = build_centerline_wse_proxy_points(river_context={"centerline_points_gdf": _gdf()}, out_path=str(tmp_path / "wse.gpkg"), receipt_path=str(tmp_path / "wse.json"))
    assert Path(out["output_artifact"]).exists()
    assert Path(out["receipt_path"]).exists()
    assert out["record_count"] == 2


def test_build_centerline_wse_proxy_points_from_raster(tmp_path: Path):
    arr = np.array([[5.0, 4.0]], dtype=np.float32)
    raster_path = tmp_path / "bank.tif"
    with rasterio.open(raster_path, "w", driver="GTiff", height=1, width=2, count=1, dtype="float32", transform=from_origin(-0.5, 0.5, 1, 1), crs="EPSG:4326", nodata=-9999.0) as ds:
        ds.write(arr, 1)
    gdf = gpd.GeoDataFrame({"point_id": ["a", "b"], "station_m": [0.0, 10.0]}, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    out = build_centerline_wse_proxy_points(river_context={"centerline_points_gdf": gdf, "bank_wse_edge_guidance_path": str(raster_path)}, out_path=str(tmp_path / "wse.gpkg"))
    assert out["record_count"] == 2
