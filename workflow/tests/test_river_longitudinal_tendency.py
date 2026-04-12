import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_longitudinal_tendency import apply_longitudinal_tendency_to_nodes


def _build_nodes(locked_station=None):
    rows = []
    thalweg_vals = {0.0: 0.0, 10.0: 4.0, 20.0: 0.0, 30.0: 4.0, 40.0: 0.0}
    for station, thalweg in thalweg_vals.items():
        for role, offset in [("left_bank", 2.0), ("thalweg", 0.0), ("right_bank", 2.0)]:
            locked = locked_station is not None and np.isclose(station, locked_station)
            rows.append(
                {
                    "component_id": "main",
                    "station_m": station,
                    "node_role": role,
                    "bed_z_m": thalweg + offset,
                    "z_source": "authoritative_in_channel" if locked and role == "thalweg" else "graph_backbone",
                    "graph_solver_support_class": "authoritative_locked" if locked and role == "thalweg" else "graph_backbone",
                    "graph_hard_lock": bool(locked and role == "thalweg"),
                    "station_support_mode": "missing",
                    "graph_candidate_source": "missing",
                    "graph_backbone_z_m": thalweg + offset,
                    "backbone_bed_z_m": thalweg + offset,
                }
            )
    return pd.DataFrame(rows)


def _write_reach_csv(path: Path):
    pd.DataFrame(
        [
            {
                "profile_id": "main",
                "reach_id": "main:0",
                "reach_role": "interior",
                "station_min_m": 0.0,
                "station_max_m": 40.0,
                "unsupported_fraction": 1.0,
                "supported_fraction": 0.0,
                "authoritative_anchor_fraction": 0.0,
                "xs_support_fraction": 0.0,
                "junction_adjustment_station_fraction": 0.0,
                "component_junction_count": 0,
            }
        ]
    ).to_csv(path, index=False)


def _write_junction_reach_csv(path: Path):
    pd.DataFrame(
        [
            {
                "profile_id": "main",
                "reach_id": "main:0",
                "reach_role": "junction_adjusted",
                "station_min_m": 0.0,
                "station_max_m": 40.0,
                "unsupported_fraction": 1.0,
                "supported_fraction": 0.0,
                "authoritative_anchor_fraction": 0.0,
                "xs_support_fraction": 0.0,
                "junction_adjustment_station_fraction": 1.0,
                "component_junction_count": 1,
            }
        ]
    ).to_csv(path, index=False)


def _write_longitudinal_profile_csv(path: Path):
    pd.DataFrame(
        [
            {"profile_id": "main", "station_m": 0.0, "network_backbone_elevation_m": 0.0, "network_junction_adjustment_m": 0.0, "network_backbone_source": "network_backbone_solve", "junction_hierarchy_weight": 1.0, "junction_wse_weight": 1.0},
            {"profile_id": "main", "station_m": 10.0, "network_backbone_elevation_m": 1.0, "network_junction_adjustment_m": 0.6, "network_backbone_source": "network_junction_flow_aware_solve", "junction_hierarchy_weight": 1.2, "junction_wse_weight": 1.1},
            {"profile_id": "main", "station_m": 20.0, "network_backbone_elevation_m": 2.5, "network_junction_adjustment_m": 1.0, "network_backbone_source": "network_junction_flow_aware_solve", "junction_hierarchy_weight": 1.3, "junction_wse_weight": 1.2},
            {"profile_id": "main", "station_m": 30.0, "network_backbone_elevation_m": 1.0, "network_junction_adjustment_m": 0.6, "network_backbone_source": "network_junction_flow_aware_solve", "junction_hierarchy_weight": 1.2, "junction_wse_weight": 1.1},
            {"profile_id": "main", "station_m": 40.0, "network_backbone_elevation_m": 0.0, "network_junction_adjustment_m": 0.0, "network_backbone_source": "network_backbone_solve", "junction_hierarchy_weight": 1.0, "junction_wse_weight": 1.0},
        ]
    ).to_csv(path, index=False)


def test_apply_longitudinal_tendency_adjusts_structural_nodes_and_writes_receipts(tmp_path: Path):
    nodes = _build_nodes()
    reach_csv = tmp_path / "reach.csv"
    _write_reach_csv(reach_csv)

    updated, outputs, summary = apply_longitudinal_tendency_to_nodes(
        nodes,
        river_dir=tmp_path,
        reach_attributes_path=reach_csv,
    )

    assert summary["available"] is True
    assert Path(outputs["longitudinal_tendency_profile"]).exists()
    assert Path(outputs["longitudinal_tendency_summary"]).exists()

    thalweg = updated.loc[updated["node_role"].eq("thalweg")].sort_values("station_m")
    before = thalweg["bed_z_m_before_longitudinal_tendency"].to_numpy(dtype=float)
    after = thalweg["bed_z_m"].to_numpy(dtype=float)
    assert np.max(np.abs(after - before)) > 0.0
    assert after[1] < before[1]
    assert after[3] < before[3]

    left_bank = updated.loc[updated["node_role"].eq("left_bank")].sort_values("station_m")
    assert np.allclose(
        left_bank["longitudinal_tendency_delta_m"].to_numpy(dtype=float),
        thalweg["longitudinal_tendency_delta_m"].to_numpy(dtype=float),
    )

    payload = json.loads(Path(outputs["longitudinal_tendency_summary"]).read_text(encoding="utf-8"))
    assert payload["adjusted_node_count"] > 0
    assert payload["delta_abs_m"]["p95"] > 0.0


def test_apply_longitudinal_tendency_preserves_authoritative_hard_lock_nodes(tmp_path: Path):
    nodes = _build_nodes(locked_station=20.0)
    reach_csv = tmp_path / "reach.csv"
    _write_reach_csv(reach_csv)

    updated, _, summary = apply_longitudinal_tendency_to_nodes(
        nodes,
        river_dir=tmp_path,
        reach_attributes_path=reach_csv,
    )

    locked = updated.loc[
        updated["node_role"].eq("thalweg")
        & np.isclose(updated["station_m"].to_numpy(dtype=float), 20.0)
    ].iloc[0]
    assert float(locked["longitudinal_tendency_delta_m"]) == 0.0
    assert float(locked["bed_z_m"]) == float(locked["bed_z_m_before_longitudinal_tendency"])
    assert summary["hard_lock_node_count_preserved"] >= 1


def test_apply_longitudinal_tendency_uses_junction_target_context_when_available(tmp_path: Path):
    nodes = _build_nodes()
    reach_csv = tmp_path / "junction_reach.csv"
    profile_csv = tmp_path / "longitudinal_profile.csv"
    _write_junction_reach_csv(reach_csv)
    _write_longitudinal_profile_csv(profile_csv)

    baseline_updated, _, baseline_summary = apply_longitudinal_tendency_to_nodes(
        nodes,
        river_dir=tmp_path / "baseline",
        reach_attributes_path=reach_csv,
    )
    junction_updated, outputs, summary = apply_longitudinal_tendency_to_nodes(
        nodes,
        river_dir=tmp_path / "junction",
        reach_attributes_path=reach_csv,
        longitudinal_profile_path=profile_csv,
    )

    baseline_thalweg = baseline_updated.loc[baseline_updated["node_role"].eq("thalweg")].sort_values("station_m")
    junction_thalweg = junction_updated.loc[junction_updated["node_role"].eq("thalweg")].sort_values("station_m")

    baseline_center = float(baseline_thalweg.loc[np.isclose(baseline_thalweg["station_m"].to_numpy(dtype=float), 20.0), "bed_z_m"].iloc[0])
    junction_center = float(junction_thalweg.loc[np.isclose(junction_thalweg["station_m"].to_numpy(dtype=float), 20.0), "bed_z_m"].iloc[0])
    assert junction_center > baseline_center
    assert summary["junction_targeted_node_count"] > 0
    assert summary["junction_weight_summary"]["max"] > 0.0
    payload = json.loads(Path(outputs["longitudinal_tendency_summary"]).read_text(encoding="utf-8"))
    assert payload["components"][0]["junction_targeted_station_count"] > 0
    assert payload["components"][0]["junction_delta_abs_m"]["count"] > 0

    profile_df = pd.read_csv(outputs["longitudinal_tendency_profile"])
    center_row = profile_df.loc[np.isclose(profile_df["station_m"].to_numpy(dtype=float), 20.0)].iloc[0]
    assert float(center_row["longitudinal_tendency_junction_weight"]) > 0.0
    assert np.isfinite(float(center_row["longitudinal_tendency_junction_target_z_m"]))
    assert baseline_summary["junction_targeted_node_count"] == 0


def test_apply_longitudinal_tendency_reports_backbone_and_guard_receipts(tmp_path: Path):
    nodes = _build_nodes()
    reach_csv = tmp_path / "reach.csv"
    profile_csv = tmp_path / "longitudinal_profile.csv"
    _write_reach_csv(reach_csv)
    _write_longitudinal_profile_csv(profile_csv)

    updated, outputs, summary = apply_longitudinal_tendency_to_nodes(
        nodes,
        river_dir=tmp_path,
        reach_attributes_path=reach_csv,
        longitudinal_profile_path=profile_csv,
    )

    assert summary["backbone_targeted_node_count"] > 0
    assert summary["low_support_guarded_node_count"] >= 0
    payload = json.loads(Path(outputs["longitudinal_tendency_summary"]).read_text(encoding="utf-8"))
    comp = payload["components"][0]
    assert comp["backbone_targeted_station_count"] > 0
    assert "adverse_step_before_m" in comp
    assert "adverse_step_after_m" in comp

    thalweg = updated.loc[updated["node_role"].eq("thalweg")].sort_values("station_m")
    assert np.max(pd.to_numeric(thalweg["longitudinal_tendency_backbone_weight"], errors="coerce").to_numpy(dtype=float)) > 0.0
    assert np.any(np.isfinite(pd.to_numeric(thalweg["longitudinal_tendency_backbone_target_z_m"], errors="coerce").to_numpy(dtype=float)))


def test_apply_longitudinal_tendency_writes_split_xs_provenance_receipts(tmp_path: Path):
    rows = []
    for station, z_source, support_mode, candidate_source, support_class in [
        (0.0, "xs_profile_resampled", "xs_supported", "xs_profile_resampled", "graph_backbone"),
        (10.0, "graph_backbone", "xs_supported", "xs_profile_resampled", "graph_backbone"),
        (20.0, "graph_backbone", "xs_residual_only", "missing", "xs_residual_only"),
        (30.0, "graph_backbone", "missing", "missing", "graph_backbone"),
    ]:
        for role, offset in [("left_bank", 2.0), ("thalweg", 0.0), ("right_bank", 2.0)]:
            rows.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "bed_z_m": float(station / 10.0) + offset,
                "z_source": z_source,
                "graph_solver_support_class": support_class,
                "graph_hard_lock": False,
                "station_support_mode": support_mode,
                "graph_candidate_source": candidate_source,
                "graph_backbone_z_m": float(station / 10.0) + offset,
                "backbone_bed_z_m": float(station / 10.0) + offset,
            })
    nodes = pd.DataFrame(rows)
    reach_csv = tmp_path / "reach.csv"
    _write_reach_csv(reach_csv)
    _, outputs, summary = apply_longitudinal_tendency_to_nodes(
        nodes,
        river_dir=tmp_path,
        reach_attributes_path=reach_csv,
    )
    prov_csv = Path(outputs["river_xs_provenance_station_summary"])
    prov_json = Path(outputs["river_xs_provenance_receipt"])
    assert prov_csv.exists()
    assert prov_json.exists()
    prov_df = pd.read_csv(prov_csv)
    assert set(prov_df["station_provenance_class"].astype(str)) >= {"true_measured_xs", "indirect_xs", "residual_xs", "structural"}
    assert int(summary["true_measured_xs_station_count"]) == 1
    assert int(summary["indirect_xs_station_count"]) == 1
    assert int(summary["residual_xs_station_count"]) == 1
    payload = json.loads(prov_json.read_text(encoding="utf-8"))
    assert payload["provenance_class_counts"]["true_measured_xs"] == 1
    assert payload["provenance_class_counts"]["indirect_xs"] == 1
    assert payload["provenance_class_counts"]["residual_xs"] == 1


def test_longitudinal_gating_treats_indirect_xs_as_weak_support_not_measured(tmp_path: Path):
    rows = []
    specs = [
        (0.0, "xs_profile_resampled", "xs_supported", "xs_profile_resampled", "graph_backbone"),
        (10.0, "graph_backbone", "xs_supported", "xs_profile_resampled", "graph_backbone"),
        (20.0, "graph_backbone", "missing", "missing", "graph_backbone"),
        (30.0, "graph_backbone", "missing", "missing", "graph_backbone"),
        (40.0, "graph_backbone", "missing", "missing", "graph_backbone"),
    ]
    zvals = {0.0: 0.0, 10.0: 4.0, 20.0: 0.0, 30.0: 4.0, 40.0: 0.0}
    for station, z_source, support_mode, candidate_source, support_class in specs:
        for role, offset in [("left_bank", 2.0), ("thalweg", 0.0), ("right_bank", 2.0)]:
            rows.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "bed_z_m": zvals[station] + offset,
                "z_source": z_source,
                "graph_solver_support_class": support_class,
                "graph_hard_lock": False,
                "station_support_mode": support_mode,
                "graph_candidate_source": candidate_source,
                "graph_backbone_z_m": zvals[station] + offset,
                "backbone_bed_z_m": zvals[station] + offset,
            })
    nodes = pd.DataFrame(rows)
    reach_csv = tmp_path / "reach.csv"
    profile_csv = tmp_path / "longitudinal_profile.csv"
    _write_reach_csv(reach_csv)
    _write_longitudinal_profile_csv(profile_csv)
    updated, outputs, summary = apply_longitudinal_tendency_to_nodes(
        nodes,
        river_dir=tmp_path,
        reach_attributes_path=reach_csv,
        longitudinal_profile_path=profile_csv,
    )
    profile_df = pd.read_csv(outputs["longitudinal_tendency_profile"])
    measured = profile_df.loc[np.isclose(profile_df["station_m"].to_numpy(dtype=float), 0.0)].iloc[0]
    indirect = profile_df.loc[np.isclose(profile_df["station_m"].to_numpy(dtype=float), 10.0)].iloc[0]
    assert measured["longitudinal_support_regime"] == "longitudinal_measured_protected"
    assert indirect["longitudinal_support_regime"] == "longitudinal_weak_support"
    assert float(measured["longitudinal_tendency_blend_weight"]) == 0.0
    assert float(indirect["longitudinal_tendency_blend_weight"]) > 0.20
    assert float(indirect["longitudinal_tendency_backbone_weight"]) > 0.30
    regime_df = pd.read_csv(outputs["river_longitudinal_support_regime_summary"])
    assert "longitudinal_support_regime" in regime_df.columns
    assert summary["longitudinal_support_regime_counts"]["longitudinal_weak_support"] >= 1


def test_apply_longitudinal_tendency_station_provenance_uses_full_station_roles(tmp_path: Path):
    rows = []
    for station in [0.0, 10.0, 20.0]:
        for role in ["left_bank", "thalweg", "right_bank"]:
            rows.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "bed_z_m": float(station / 10.0) + (2.0 if role != "thalweg" else 0.0),
                "z_source": "graph_backbone",
                "graph_solver_support_class": "graph_backbone",
                "graph_hard_lock": False,
                "station_support_mode": "missing",
                "graph_candidate_source": "missing",
                "graph_backbone_z_m": float(station / 10.0),
                "backbone_bed_z_m": float(station / 10.0),
                "station_authoritative_role": "authoritative_bank_margin",
                "station_authoritative_role_confidence": 0.7,
                "station_authoritative_distance_to_bank_m": 1.5,
                "station_authoritative_bed_support_present": False,
                "station_authoritative_bank_margin_present": True,
            })
    nodes = pd.DataFrame(rows)
    reach_csv = tmp_path / "reach.csv"
    _write_reach_csv(reach_csv)
    _, outputs, _ = apply_longitudinal_tendency_to_nodes(
        nodes,
        river_dir=tmp_path,
        reach_attributes_path=reach_csv,
    )
    prov_df = pd.read_csv(outputs["river_xs_provenance_station_summary"])
    assert "station_authoritative_role" in prov_df.columns
    assert (prov_df["station_authoritative_role"].astype(str) == "authoritative_bank_margin").all()
    assert (pd.to_numeric(prov_df["station_authoritative_bank_margin_fraction"], errors="coerce") > 0).all()
