from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from river_prediction_confidence import apply_prediction_confidence_to_nodes


def test_apply_prediction_confidence_to_nodes_distinguishes_authoritative_and_structure_only(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    stations = [0.0, 10.0, 20.0]
    xs = [2.5, 5.5, 8.5]
    for station, x in zip(stations, xs):
        for role, eta in [("left_bank", 0.0), ("left_inner", 0.25), ("thalweg", 0.5), ("right_inner", 0.75), ("right_bank", 1.0)]:
            if station == 0.0:
                z_source = "authoritative_in_channel" if role == "thalweg" else "xs_profile_resampled"
                support = "authoritative_locked" if role == "thalweg" else "xs_residual_only"
            else:
                z_source = "graph_backbone"
                support = "unsupported"
            recs.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "cross_stream_eta": eta,
                "bed_z_m": 2.0 - 0.1 * eta,
                "z_source": z_source,
                "graph_solver_support_class": support,
                "graph_hard_lock": role == "thalweg" and station == 0.0,
                "longitudinal_tendency_delta_m": 0.0 if station == 0.0 else 0.35,
                "xs_realism_delta_m": 0.0 if station == 0.0 else 0.20,
                "geometry": Point(x, 5.0),
            })
    nodes = gpd.GeoDataFrame(recs, crs="EPSG:4326")
    nodes_out, outputs, summary = apply_prediction_confidence_to_nodes(nodes, river_dir=river_dir)
    assert Path(outputs["prediction_confidence_profile"]).exists()
    assert Path(outputs["prediction_confidence_summary"]).exists()
    assert summary["available"] is True
    auth_station = nodes_out.loc[nodes_out["station_m"].eq(0.0)]
    weak_station = nodes_out.loc[nodes_out["station_m"].eq(20.0)]
    assert float(np.nanmedian(auth_station["prediction_support_confidence"].to_numpy(dtype=float))) == 1.0
    assert bool(auth_station["prediction_admissible"].all()) is True
    assert float(np.nanmedian(weak_station["prediction_support_confidence"].to_numpy(dtype=float))) < 0.5
    assert bool(weak_station["prediction_low_support_caution"].all()) is True
    assert bool(weak_station["prediction_admissible"].any()) is False
    assert float(np.nanmedian(weak_station["prediction_structure_only_fraction"].to_numpy(dtype=float))) > 0.75


def test_prediction_confidence_limits_unsupported_side_component(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    for station, x in zip([0.0, 10.0, 20.0], [2.5, 5.5, 8.5]):
        for role, eta in [("left_bank", 0.0), ("left_inner", 0.25), ("thalweg", 0.5), ("right_inner", 0.75), ("right_bank", 1.0)]:
            recs.append({
                "component_id": "trib",
                "station_m": station,
                "node_role": role,
                "cross_stream_eta": eta,
                "bed_z_m": 2.0 - 0.05 * eta,
                "z_source": "graph_backbone",
                "graph_solver_support_class": "unsupported",
                "graph_hard_lock": False,
                "longitudinal_tendency_delta_m": 0.40,
                "xs_realism_delta_m": 0.0,
                "component_support_class": "unsupported_side_component",
                "component_support_bed_fraction": 0.0,
                "authoritative_reconciliation_confidence": 0.0,
                "geometry": Point(x, 5.0),
            })
    nodes = gpd.GeoDataFrame(recs, crs="EPSG:4326")
    nodes_out, outputs, summary = apply_prediction_confidence_to_nodes(nodes, river_dir=river_dir)
    assert summary["component_class_counts"]["unsupported_side_component"] == 3
    station = nodes_out.loc[nodes_out["station_m"].eq(20.0)]
    assert float(np.nanmedian(station["prediction_support_confidence"].to_numpy(dtype=float))) < 0.45
    assert bool(station["prediction_low_support_caution"].all()) is True
    assert bool(station["prediction_admissible"].any()) is False
    assert set(nodes_out["prediction_confidence_regime"].astype(str)) == {"component_class_limited"}


def test_prediction_confidence_rejects_weak_component_without_reconciliation(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    for station, x in zip([0.0, 10.0, 20.0], [2.5, 5.5, 8.5]):
        for role, eta in [("left_bank", 0.0), ("left_inner", 0.25), ("thalweg", 0.5), ("right_inner", 0.75), ("right_bank", 1.0)]:
            recs.append({
                "component_id": "tiny",
                "station_m": station,
                "node_role": role,
                "cross_stream_eta": eta,
                "bed_z_m": 2.0 - 0.05 * eta,
                "z_source": "graph_backbone",
                "graph_solver_support_class": "unsupported",
                "graph_hard_lock": False,
                "longitudinal_tendency_delta_m": 0.20,
                "xs_realism_delta_m": 0.0,
                "component_support_class": "tiny_detached_component",
                "component_support_bed_fraction": 0.0,
                "component_support_median_distance_m": 220.0,
                "authoritative_reconciliation_confidence": 0.0,
                "geometry": Point(x, 5.0),
            })
    nodes = gpd.GeoDataFrame(recs, crs="EPSG:4326")
    nodes_out, _, _ = apply_prediction_confidence_to_nodes(nodes, river_dir=river_dir)
    station = nodes_out.loc[nodes_out["station_m"].eq(20.0)]
    assert bool(station["prediction_admissible"].any()) is False
    assert float(np.nanmedian(station["prediction_support_confidence"].to_numpy(dtype=float))) < 0.40


def test_prediction_confidence_rejects_unsupported_side_component_more_aggressively_when_structure_only(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    recs = []
    for station, x in zip([0.0, 10.0, 20.0], [2.5, 5.5, 8.5]):
        for role, eta in [("left_bank", 0.0), ("left_inner", 0.25), ("thalweg", 0.5), ("right_inner", 0.75), ("right_bank", 1.0)]:
            recs.append({
                "component_id": "trib2",
                "station_m": station,
                "node_role": role,
                "cross_stream_eta": eta,
                "bed_z_m": 2.0 - 0.05 * eta,
                "z_source": "graph_backbone",
                "graph_solver_support_class": "unsupported",
                "graph_hard_lock": False,
                "longitudinal_tendency_delta_m": 0.35,
                "xs_realism_delta_m": 0.0,
                "component_support_class": "unsupported_side_component",
                "component_support_bed_fraction": 0.0,
                "component_support_median_distance_m": 180.0,
                "authoritative_reconciliation_confidence": 0.0,
                "geometry": Point(x, 5.0),
            })
    nodes = gpd.GeoDataFrame(recs, crs="EPSG:4326")
    nodes_out, _, _ = apply_prediction_confidence_to_nodes(nodes, river_dir=river_dir)
    station = nodes_out.loc[nodes_out["station_m"].eq(20.0)]
    assert bool(station["prediction_admissible"].any()) is False
    assert float(np.nanmedian(station["prediction_support_confidence"].to_numpy(dtype=float))) < 0.35
