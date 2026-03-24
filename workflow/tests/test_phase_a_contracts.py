from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

import xs_contracts
from xs_contracts import validate_soundings_subset_parquet, validate_xs_artifacts


def test_validate_soundings_subset_parquet(tmp_path: Path, monkeypatch):
    p = tmp_path / "subset.parquet"
    p.write_text("stub")

    def _fake_read_parquet(path):
        assert Path(path) == p
        return pd.DataFrame({
            "x": [0.0, 1.0],
            "y": [2.0, 3.0],
            "z": [-1.0, -2.0],
            "depth_m": [-1.0, -2.0],
            "z_m": [-1.0, -2.0],
            "_src_file": ["a", "a"],
            "crs": ["EPSG:32619", "EPSG:32619"],
        })

    monkeypatch.setattr(xs_contracts.pd, "read_parquet", _fake_read_parquet)
    info = validate_soundings_subset_parquet(p)
    assert info["rows"] == 2
    assert info["depth_col"] == "depth_m"


def test_validate_xs_artifacts(tmp_path: Path):
    gpkg = tmp_path / "xs.gpkg"
    xs_lines = gpd.GeoDataFrame({
        "xs_id": [1],
        "bank_left_dist_m": [0.0],
        "bank_right_dist_m": [1.0],
        "bank_left_z_m": [2.0],
        "bank_right_z_m": [3.0],
        "s_center_m": [0.5],
    }, geometry=[LineString([(0, 0), (1, 0)])], crs="EPSG:4326")
    xs_points = gpd.GeoDataFrame({
        "xs_id": [1, 1],
        "s_m": [0.0, 1.0],
        "z_m": [2.0, 3.0],
        "is_bank_left": [True, False],
        "is_bank_right": [False, True],
    }, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    xs_lines.to_file(gpkg, layer="xs_lines", driver="GPKG")
    xs_points.to_file(gpkg, layer="xs_points", driver="GPKG")
    info = validate_xs_artifacts(gpkg)
    assert info["xs_lines_n"] == 1
    assert info["xs_points_n"] == 2



def test_validate_xs_artifacts_accepts_builder_alias_columns(tmp_path: Path):
    gpkg = tmp_path / "xs_alias.gpkg"
    xs_lines = gpd.GeoDataFrame({
        "xs_id": [1],
        "bank_left_dist_m": [0.0],
        "bank_right_dist_m": [1.0],
        "bank_left_z_m": [2.0],
        "bank_right_z_m": [3.0],
        "s_center_m": [0.5],
    }, geometry=[LineString([(0, 0), (1, 0)])], crs="EPSG:4326")
    xs_points = gpd.GeoDataFrame({
        "xs_id": [1, 1],
        "dist_m": [0.0, 1.0],
        "z_dem": [2.0, 3.0],
        "is_bank_left": [True, False],
        "is_bank_right": [False, True],
    }, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    xs_lines.to_file(gpkg, layer="xs_lines", driver="GPKG")
    xs_points.to_file(gpkg, layer="xs_points", driver="GPKG")
    xs_lines_loaded, xs_points_loaded, info = xs_contracts.load_validated_xs_artifacts(gpkg)
    assert info["xs_points_alias_map"] == {"s_m": "dist_m", "z_m": "z_dem"}
    assert "s_m" in xs_points_loaded.columns
    assert "z_m" in xs_points_loaded.columns
    assert xs_points_loaded["s_m"].tolist() == [0.0, 1.0]
    assert xs_points_loaded["z_m"].tolist() == [2.0, 3.0]
