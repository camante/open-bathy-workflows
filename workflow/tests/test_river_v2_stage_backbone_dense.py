from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from tests.test_river_v2_pipeline import _build_context
from river_v2_stage_backbone_dense import build_dense_backbone, run_backbone_dense_stage


def _write_sparse_backbone(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "point_id": ["p0", "p1"],
            "component_id": ["c", "c"],
            "station_m": [0.0, 100.0],
            "bed_backbone_z_m": [1.0, 3.0],
        },
        geometry=[Point(0, 0), Point(10, 0)],
        crs="EPSG:4326",
    )
    path = tmp_path / "backbone.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def test_build_dense_backbone_interpolates_between_sparse_anchors(tmp_path: Path):
    path = _write_sparse_backbone(tmp_path)
    centerline_path = _write_centerline(tmp_path)
    dense, diagnostics, warnings = build_dense_backbone(path, centerline_path)
    vals = dense.sort_values("station_m")["bed_backbone_z_m"].to_numpy(dtype=float)
    stations = dense.sort_values("station_m")["station_m"].to_numpy(dtype=float)
    assert stations.tolist() == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert np.allclose(vals, [1.0, 1.5, 2.0, 2.5, 3.0])
    assert diagnostics["interp_count"] == 3
    assert warnings == []


def test_run_backbone_dense_stage_requires_centerline_points(tmp_path: Path):
    ctx = _build_context(tmp_path)
    backbone_path = _write_sparse_backbone(tmp_path)
    with pytest.raises(RuntimeError, match="river_v2_backbone_dense_missing_centerline"):
        run_backbone_dense_stage(ctx, backbone_points_path=backbone_path, centerline_points_path=tmp_path / "missing.gpkg")


def _write_centerline(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "point_id": ["c0", "c1", "c2", "c3", "c4"],
            "component_id": ["c"] * 5,
            "station_m": [0.0, 25.0, 50.0, 75.0, 100.0],
        },
        geometry=[Point(x, 0) for x in (0, 2.5, 5, 7.5, 10)],
        crs="EPSG:4326",
    )
    path = tmp_path / "centerline.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path
