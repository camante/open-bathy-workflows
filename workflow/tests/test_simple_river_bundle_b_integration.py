from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from simple_river_bundle_b import run_simple_river_bundle_b
from simple_river_stage_contract import (
    STAGE_CENTERLINE_OBSERVED_OFFSET,
    STAGE_CENTERLINE_WSE_PROXY,
    simple_river_stage_status_placeholder,
)


def test_downstream_stages_prefer_canonical_centerline_path(tmp_path):
    legacy = gpd.GeoDataFrame(
        {
            "station_m": [0.0, 1.0],
            "bank_wse_proxy_monotone_m": [3.0, 2.0],
            "centerline_z_m": [1.0, 0.5],
        },
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:4326",
    )
    status = simple_river_stage_status_placeholder()
    result = run_simple_river_bundle_b(
        river_context={"centerline_points_gdf": legacy},
        out_dir=str(tmp_path),
        stage_status=status,
        upto_stage=STAGE_CENTERLINE_OBSERVED_OFFSET,
    )
    assert result["simple_river_stage_status"][STAGE_CENTERLINE_WSE_PROXY]["implemented"] is True
    assert Path(result["wse_proxy"]["output_artifact"]).exists()
    assert Path(result["authoritative_bed"]["output_artifact"]).exists()
    assert Path(result["observed_offset"]["output_artifact"]).exists()


def test_bundle_b_failure_marks_failed_stage(tmp_path):
    legacy = gpd.GeoDataFrame({"station_m": [0.0, 1.0]}, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    status = simple_river_stage_status_placeholder()
    result = run_simple_river_bundle_b(
        river_context={"centerline_points_gdf": legacy},
        out_dir=str(tmp_path),
        stage_status=status,
        upto_stage=STAGE_CENTERLINE_WSE_PROXY,
    )
    assert result["failed_stage"] == STAGE_CENTERLINE_WSE_PROXY
    assert result["simple_river_stage_status"][STAGE_CENTERLINE_WSE_PROXY]["status"] == "failed"
