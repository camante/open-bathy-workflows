import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import river_bank_guidance as rbg


def test_build_persistent_bank_network_points_clamps_high_bank_contamination(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {
            "xs_id": ["xs1", "xs1", "xs2", "xs2", "xs3", "xs3"],
            "river_id": [1, 1, 1, 1, 1, 1],
            "component_id": [7, 7, 7, 7, 7, 7],
            "side": ["left", "right", "left", "right", "left", "right"],
            "side_sign": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            "bank_z_m": [2.0, 2.1, 6.5, 2.2, 2.2, 2.3],
            "bank_z_raw_m": [2.0, 2.1, 6.5, 2.2, 2.2, 2.3],
            "bank_z_inner_min_m": [1.9, 2.0, 2.1, 2.1, 2.1, 2.2],
            "bank_z_local_q25_m": [2.0, 2.05, 2.2, 2.15, 2.1, 2.25],
            "bank_z_local_median_m": [2.05, 2.1, 2.3, 2.2, 2.15, 2.3],
            "bank_selected_source": ["picked"] * 6,
            "s_center_m": [0.0, 0.0, 100.0, 100.0, 200.0, 200.0],
            "geometry": [Point(0,0), Point(1,0), Point(0,1), Point(1,1), Point(0,2), Point(1,2)],
        },
        geometry="geometry",
        crs="EPSG:32619",
    )
    monkeypatch.setattr(rbg, "load_xs_bank_points", lambda *args, **kwargs: gdf.copy())
    out = rbg.build_persistent_bank_network_points("dummy.gpkg", target_crs="EPSG:32619")
    left_bad = out[(out["xs_id"] == "xs2") & (out["side"] == "left")].iloc[0]
    assert bool(left_bad["bank_high_contamination_suspect"])
    assert float(left_bad["bank_z_m"]) < 4.0
    assert str(left_bad["bank_adjustment_reason"]) in {"opposite_bank_clamp", "local_candidate_clamp", "longitudinal_spike_clamp", "component_floor_clamp", "neighbor_spike_clamp", "local_envelope_replace", "multi_signal_reject"}


def test_summarize_bank_qc_reports_replacements():
    gdf = gpd.GeoDataFrame(
        {
            "bank_adjustment_m": np.array([0.0, -0.5, -1.0], dtype=np.float32),
            "bank_high_contamination_suspect": [False, True, True],
            "bank_strong_contamination": [False, False, True],
            "bank_selected_source": ["picked", "inner_min", "local_q25"],
        },
        geometry=[Point(0,0), Point(1,0), Point(2,0)],
        crs="EPSG:32619",
    )
    summary = rbg.summarize_bank_qc(gdf)
    row = summary.iloc[0]
    assert int(row["bank_replaced_count"]) == 2
    assert int(row["high_bank_suspect_count"]) == 2
    assert int(row["strong_contamination_count"]) == 1
    assert int(row["selected_from_inner_min_count"]) == 1
    assert int(row["selected_from_local_q25_count"]) == 1


def test_build_persistent_bank_network_points_flags_component_floor_and_neighbor_spikes(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {
            "xs_id": ["xs1", "xs1", "xs2", "xs2", "xs3", "xs3", "xs4", "xs4", "xs5", "xs5", "xs6", "xs6"],
            "river_id": [1] * 12,
            "component_id": [7] * 12,
            "side": ["left", "right"] * 6,
            "side_sign": [-1.0, 1.0] * 6,
            "bank_z_m": [2.0, 2.1, 2.1, 2.2, 4.6, 4.7, 4.7, 4.8, 2.2, 2.3, 2.3, 2.4],
            "bank_z_raw_m": [2.0, 2.1, 2.1, 2.2, 4.6, 4.7, 4.7, 4.8, 2.2, 2.3, 2.3, 2.4],
            "bank_z_inner_min_m": [1.9, 2.0, 2.0, 2.1, 4.3, 4.4, 4.4, 4.5, 2.1, 2.2, 2.2, 2.3],
            "bank_z_local_q25_m": [2.0, 2.05, 2.05, 2.15, 4.4, 4.5, 4.5, 4.6, 2.15, 2.25, 2.25, 2.35],
            "bank_z_local_median_m": [2.05, 2.1, 2.1, 2.2, 4.5, 4.6, 4.6, 4.7, 2.2, 2.3, 2.3, 2.4],
            "bank_selected_source": ["picked"] * 12,
            "s_center_m": [0.0, 0.0, 100.0, 100.0, 200.0, 200.0, 300.0, 300.0, 400.0, 400.0, 500.0, 500.0],
            "geometry": [Point(float(i), 0.0) for i in range(12)],
        },
        geometry="geometry",
        crs="EPSG:32619",
    )
    monkeypatch.setattr(rbg, "load_xs_bank_points", lambda *args, **kwargs: gdf.copy())
    out = rbg.build_persistent_bank_network_points("dummy.gpkg", target_crs="EPSG:32619")
    flagged = out[(out["bank_component_floor_suspect"]) | (out["bank_neighbor_spike_suspect"])].copy()
    assert not flagged.empty
    assert bool(flagged["bank_component_floor_suspect"].any())
    assert bool(flagged["bank_neighbor_spike_suspect"].any())
    summary = rbg.summarize_bank_qc(out).iloc[0]
    assert int(summary["component_floor_suspect_count"]) >= 1
    assert int(summary["neighbor_spike_suspect_count"]) >= 1


def test_build_persistent_bank_network_points_adds_multisignal_qc_actions(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {
            "xs_id": ["xs1", "xs1", "xs2", "xs2", "xs3", "xs3", "xs4", "xs4"],
            "river_id": [1] * 8,
            "component_id": [7] * 8,
            "side": ["left", "right"] * 4,
            "side_sign": [-1.0, 1.0] * 4,
            "bank_z_m": [2.0, 2.1, 7.2, 2.3, 2.1, 2.2, 2.2, 2.3],
            "bank_z_raw_m": [2.0, 2.1, 7.2, 2.3, 2.1, 2.2, 2.2, 2.3],
            "bank_z_inner_min_m": [1.9, 2.0, 2.0, 2.1, 2.0, 2.1, 2.1, 2.2],
            "bank_z_local_q25_m": [2.0, 2.05, 2.05, 2.15, 2.05, 2.15, 2.15, 2.25],
            "bank_z_local_median_m": [2.05, 2.1, 2.1, 2.2, 2.1, 2.2, 2.2, 2.3],
            "bank_selected_source": ["picked"] * 8,
            "s_center_m": [0.0, 0.0, 100.0, 100.0, 200.0, 200.0, 300.0, 300.0],
            "geometry": [Point(float(i), 0.0) for i in range(8)],
        },
        geometry="geometry",
        crs="EPSG:32619",
    )
    monkeypatch.setattr(rbg, "load_xs_bank_points", lambda *args, **kwargs: gdf.copy())
    out = rbg.build_persistent_bank_network_points("dummy.gpkg", target_crs="EPSG:32619")
    severe = out[(out["xs_id"] == "xs2") & (out["side"] == "left")].iloc[0]
    assert bool(severe["bank_local_relief_suspect"])
    assert bool(severe["bank_cross_bank_asymmetry_suspect"])
    assert bool(severe["bank_percentile_suspect"])
    assert str(severe["bank_qc_action"]) in {"reject", "replace_with_local_envelope"}
    summary = rbg.summarize_bank_qc(out).iloc[0]
    assert int(summary["local_relief_suspect_count"]) >= 1
    assert int(summary["cross_bank_asymmetry_suspect_count"]) >= 1
    assert int(summary["percentile_suspect_count"]) >= 1
    assert int(summary["bank_reject_count"]) + int(summary["bank_replace_with_local_envelope_count"]) >= 1
