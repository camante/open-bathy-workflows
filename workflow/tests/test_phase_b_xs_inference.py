from pathlib import Path

import geopandas as gpd
import json
from shapely.geometry import LineString, Point

from xs_infer_bathy_raster import infer_bathy, InferConfig
import xs_infer_bathy_raster


def _write_minimal_xs_fixture(gpkg: Path):
    xs_lines = gpd.GeoDataFrame({
        "xs_id": ["xs1"],
        "bank_left_dist_m": [0.0],
        "bank_right_dist_m": [10.0],
        "bank_left_z_m": [5.0],
        "bank_right_z_m": [5.2],
        "s_center_m": [5.0],
        "component_id": [1],
        "river_id": [1],
    }, geometry=[LineString([(0, 0), (10, 0)])], crs=None)
    pts = []
    for d in [0.0, 2.5, 5.0, 7.5, 10.0]:
        pts.append({
            "xs_id": "xs1",
            "s_m": d,
            "dist_m": d,
            "z_m": 5.0,
            "z_dem": 5.0,
            "z_topo": 5.0,
            "is_bank_left": d == 0.0,
            "is_bank_right": d == 10.0,
            "component_id": 1,
            "river_id": 1,
            "geometry": Point(d, 0),
        })
    xs_points = gpd.GeoDataFrame(pts, geometry="geometry", crs="EPSG:32619")
    xs_lines.to_file(gpkg, layer="xs_lines", driver="GPKG")
    xs_points.to_file(gpkg, layer="xs_points", driver="GPKG")


def test_infer_bathy_minimal_fixture_writes_bank_receipt(tmp_path: Path, monkeypatch):
    xs_gpkg = tmp_path / "xs.gpkg"
    _write_minimal_xs_fixture(xs_gpkg)
    out_gpkg = tmp_path / "out.gpkg"
    out_meta = tmp_path / "meta.json"
    cfg = InferConfig()
    monkeypatch.setattr(xs_infer_bathy_raster, "_attach_curvature_asymmetry", lambda xs_lines, xs_param, cfg: xs_param)
    infer_bathy(
        xs_gpkg=xs_gpkg,
        out_gpkg=out_gpkg,
        dem_path=None,
        soundings_path=None,
        soundings_depth_col=None,
        soundings_elev_col=None,
        soundings_x_col=None,
        soundings_y_col=None,
        soundings_crs=None,
        cfg=cfg,
        out_meta_json=out_meta,
    )
    assert out_gpkg.exists()
    receipt = tmp_path / "xs_bank_contract_receipt.json"
    assert receipt.exists()
    info = json.loads(receipt.read_text())
    assert info["xs_total"] == 1
    assert info["xs_with_both_bank_contract"] == 1
    assert info["xs_with_valid_param_rows"] == 1


def test_infer_bathy_recovers_bank_dists_from_point_flags(tmp_path: Path, monkeypatch):
    xs_gpkg = tmp_path / "xs_missing_bank_dists.gpkg"
    xs_lines = gpd.GeoDataFrame({
        "xs_id": ["xs1"],
        "bank_left_dist_m": [float("nan")],
        "bank_right_dist_m": [float("nan")],
        "bank_left_z_m": [5.0],
        "bank_right_z_m": [5.2],
        "s_center_m": [5.0],
        "component_id": [1],
        "river_id": [1],
    }, geometry=[LineString([(0, 0), (10, 0)])], crs=None)
    pts = []
    for d in [0.0, 2.5, 5.0, 7.5, 10.0]:
        pts.append({
            "xs_id": "xs1",
            "s_m": d,
            "dist_m": d,
            "z_m": 5.0,
            "z_dem": 5.0,
            "z_topo": 5.0,
            "is_bank_left": d == 0.0,
            "is_bank_right": d == 10.0,
            "component_id": 1,
            "river_id": 1,
            "geometry": Point(d, 0),
        })
    xs_points = gpd.GeoDataFrame(pts, geometry="geometry", crs="EPSG:32619")
    xs_lines.to_file(xs_gpkg, layer="xs_lines", driver="GPKG")
    xs_points.to_file(xs_gpkg, layer="xs_points", driver="GPKG")

    out_gpkg = tmp_path / "out.gpkg"
    out_meta = tmp_path / "meta.json"
    monkeypatch.setattr(xs_infer_bathy_raster, "_attach_curvature_asymmetry", lambda xs_lines, xs_param, cfg: xs_param)
    infer_bathy(
        xs_gpkg=xs_gpkg,
        out_gpkg=out_gpkg,
        dem_path=None,
        soundings_path=None,
        soundings_depth_col=None,
        soundings_elev_col=None,
        soundings_x_col=None,
        soundings_y_col=None,
        soundings_crs=None,
        cfg=InferConfig(),
        out_meta_json=out_meta,
    )

    info = json.loads((tmp_path / "xs_bank_contract_receipt.json").read_text())
    assert info["xs_with_valid_param_rows"] == 1
    assert info["n_xs_bank_flag_recovered"] == 1


def test_bank_pair_sanitization_prefers_lower_plausible_bank():
    left, right, stage = xs_infer_bathy_raster._sanitize_bank_pair(
        bank_left_z=2.0,
        bank_right_z=8.0,
        channel_floor_z=0.0,
    )
    assert left == 2.0
    assert right is not None and right < 8.0
    assert stage == 2.0
