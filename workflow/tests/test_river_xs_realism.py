import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_xs_realism import apply_xs_realism_to_nodes


def _build_nodes(xs_station=None, measured_xs_station=None):
    rows = []
    station_specs = {
        0.0: {"left_bank": 10.0, "left_inner": 7.0, "thalweg": 4.0, "right_inner": 3.0, "right_bank": 6.0},
        10.0: {"left_bank": 10.0, "left_inner": 9.5, "thalweg": 4.0, "right_inner": 0.0, "right_bank": 6.0},
        20.0: {"left_bank": 10.0, "left_inner": 7.0, "thalweg": 4.0, "right_inner": 3.0, "right_bank": 6.0},
    }
    for station, role_map in station_specs.items():
        for role, z in role_map.items():
            is_xs = xs_station is not None and np.isclose(station, xs_station)
            is_measured_xs = measured_xs_station is not None and np.isclose(station, measured_xs_station)
            rows.append(
                {
                    "component_id": "main",
                    "station_m": station,
                    "node_role": role,
                    "bed_z_m": z,
                    "z_source": "authoritative_in_channel" if is_measured_xs else ("xs_profile_resampled" if is_xs else "graph_backbone"),
                    "graph_solver_support_class": "authoritative_locked" if is_measured_xs else ("xs_residual_only" if is_xs else "graph_backbone"),
                    "graph_hard_lock": bool(is_measured_xs),
                    "station_support_mode": "xs_supported" if (is_xs or is_measured_xs) else "missing",
                    "graph_candidate_source": "authoritative_in_channel" if is_measured_xs else ("xs_profile_resampled" if is_xs else "missing"),
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
                "station_max_m": 20.0,
                "unsupported_fraction": 1.0,
                "supported_fraction": 0.0,
                "authoritative_anchor_fraction": 0.0,
                "xs_support_fraction": 0.0,
                "junction_adjustment_station_fraction": 0.0,
                "component_junction_count": 0,
            }
        ]
    ).to_csv(path, index=False)


def test_apply_xs_realism_smooths_structural_inner_nodes_and_preserves_banks_thalweg(tmp_path: Path):
    nodes = _build_nodes()
    reach_csv = tmp_path / "reach.csv"
    _write_reach_csv(reach_csv)

    updated, outputs, summary = apply_xs_realism_to_nodes(
        nodes,
        river_dir=tmp_path,
        reach_attributes_path=reach_csv,
    )

    assert summary["available"] is True
    assert Path(outputs["xs_realism_profile"]).exists()
    assert Path(outputs["xs_realism_summary"]).exists()

    left_inner = updated.loc[updated["node_role"].eq("left_inner")].sort_values("station_m")
    right_inner = updated.loc[updated["node_role"].eq("right_inner")].sort_values("station_m")
    banks = updated.loc[updated["node_role"].isin(["left_bank", "right_bank"])].sort_values(["node_role", "station_m"])
    thalweg = updated.loc[updated["node_role"].eq("thalweg")].sort_values("station_m")

    assert float(left_inner.loc[np.isclose(left_inner["station_m"].to_numpy(dtype=float), 10.0), "bed_z_m"].iloc[0]) < 9.5
    assert float(right_inner.loc[np.isclose(right_inner["station_m"].to_numpy(dtype=float), 10.0), "bed_z_m"].iloc[0]) > 0.0
    assert np.allclose(banks["bed_z_m"].to_numpy(dtype=float), banks["bed_z_m_before_xs_realism"].to_numpy(dtype=float))
    assert np.allclose(thalweg["bed_z_m"].to_numpy(dtype=float), thalweg["bed_z_m_before_xs_realism"].to_numpy(dtype=float))

    payload = json.loads(Path(outputs["xs_realism_summary"]).read_text(encoding="utf-8"))
    assert payload["adjusted_node_count"] > 0
    assert payload["delta_abs_m"]["p95"] > 0.0


def test_apply_xs_realism_preserves_measured_xs_station_shape(tmp_path: Path):
    nodes = _build_nodes(xs_station=10.0, measured_xs_station=10.0)
    reach_csv = tmp_path / "reach.csv"
    _write_reach_csv(reach_csv)

    updated, _, summary = apply_xs_realism_to_nodes(
        nodes,
        river_dir=tmp_path,
        reach_attributes_path=reach_csv,
    )

    left_inner = updated.loc[updated["node_role"].eq("left_inner")].sort_values("station_m")
    right_inner = updated.loc[updated["node_role"].eq("right_inner")].sort_values("station_m")

    left_center = left_inner.loc[np.isclose(left_inner["station_m"].to_numpy(dtype=float), 10.0)].iloc[0]
    right_center = right_inner.loc[np.isclose(right_inner["station_m"].to_numpy(dtype=float), 10.0)].iloc[0]
    assert float(left_center["xs_realism_delta_m"]) == 0.0
    assert float(right_center["xs_realism_delta_m"]) == 0.0
    assert float(left_center["bed_z_m"]) == float(left_center["bed_z_m_before_xs_realism"])
    assert float(right_center["bed_z_m"]) == float(right_center["bed_z_m_before_xs_realism"])
    assert summary["adjusted_node_count"] >= 0


def test_apply_xs_realism_assigns_generic_template_in_low_support_bank_only_reach(tmp_path: Path):
    nodes = _build_nodes()
    reach_csv = tmp_path / "reach.csv"
    pd.DataFrame([
        {
            "profile_id": "main",
            "reach_id": "main:0",
            "reach_role": "interior",
            "station_min_m": 0.0,
            "station_max_m": 20.0,
            "unsupported_fraction": 0.95,
            "supported_fraction": 0.0,
            "authoritative_anchor_fraction": 0.0,
            "xs_support_fraction": 0.0,
            "junction_adjustment_station_fraction": 0.0,
            "component_junction_count": 0,
        }
    ]).to_csv(reach_csv, index=False)

    updated, outputs, summary = apply_xs_realism_to_nodes(nodes, river_dir=tmp_path, reach_attributes_path=reach_csv)
    profile = pd.read_csv(outputs["xs_realism_profile"])
    center = profile.loc[np.isclose(profile["station_m"].to_numpy(dtype=float), 10.0)].iloc[0]
    assert center["xs_support_template_class"] == "bank_only_low_confidence"
    assert center["xs_template_type"] == "generic_symmetric"
    assert float(center["left_inner_ratio_after"]) > 0.0
    assert float(center["right_inner_ratio_after"]) < 0.0
    assert abs(abs(float(center["left_inner_ratio_after"])) - abs(float(center["right_inner_ratio_after"]))) < 1.0e-6
    assert summary["template_type_counts"]["generic_symmetric"] >= 1



def test_apply_xs_realism_propagates_template_metadata_to_nodes(tmp_path: Path):
    nodes = _build_nodes()
    reach_csv = tmp_path / "reach.csv"
    pd.DataFrame([{
        "profile_id": "main",
        "reach_id": "main:0",
        "reach_role": "interior",
        "station_min_m": 0.0,
        "station_max_m": 20.0,
        "unsupported_fraction": 1.0,
        "supported_fraction": 0.0,
        "authoritative_anchor_fraction": 0.0,
        "xs_support_fraction": 0.0,
        "junction_adjustment_station_fraction": 0.0,
        "component_junction_count": 0,
    }]).to_csv(reach_csv, index=False)
    out, _, _ = apply_xs_realism_to_nodes(nodes, river_dir=tmp_path, reach_attributes_path=reach_csv)
    center = out.loc[np.isclose(pd.to_numeric(out["station_m"], errors="coerce").to_numpy(dtype=float), 10.0)]
    assert set(center["xs_support_template_class"].astype(str)) == {"bank_only_low_confidence"}
    assert set(center["xs_template_type"].astype(str)) == {"generic_symmetric"}


def test_apply_xs_realism_reports_split_station_provenance(tmp_path: Path):
    rows = []
    for station, z_source, support_mode, candidate_source, support_class in [
        (0.0, "xs_profile_resampled", "xs_supported", "xs_profile_resampled", "graph_backbone"),
        (10.0, "graph_backbone", "xs_supported", "xs_profile_resampled", "graph_backbone"),
        (20.0, "graph_backbone", "xs_residual_only", "missing", "xs_residual_only"),
    ]:
        vals = {"left_bank": 10.0, "left_inner": 7.0, "thalweg": 4.0, "right_inner": 3.0, "right_bank": 6.0}
        for role, z in vals.items():
            rows.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "bed_z_m": z,
                "z_source": z_source,
                "graph_solver_support_class": support_class,
                "graph_hard_lock": False,
                "station_support_mode": support_mode,
                "graph_candidate_source": candidate_source,
            })
    nodes = pd.DataFrame(rows)
    reach_csv = tmp_path / "reach.csv"
    _write_reach_csv(reach_csv)
    _, outputs, summary = apply_xs_realism_to_nodes(nodes, river_dir=tmp_path, reach_attributes_path=reach_csv)
    prov_df = pd.read_csv(outputs["river_xs_realism_provenance_station_summary"])
    assert set(prov_df["station_provenance_class"].astype(str)) == {"true_measured_xs", "indirect_xs", "residual_xs"}
    assert summary["true_measured_xs_station_count"] == 1
    assert summary["indirect_xs_station_count"] == 1
    assert summary["residual_xs_station_count"] == 1


def test_apply_xs_realism_keeps_channel_nodes_rebuildable_when_only_banks_are_authoritative(tmp_path: Path):
    nodes = _build_nodes()
    bank_mask = nodes["node_role"].isin(["left_bank", "right_bank"])
    nodes.loc[bank_mask, "z_source"] = "authoritative_bank"
    nodes.loc[bank_mask, "graph_hard_lock"] = True
    nodes.loc[bank_mask, "graph_solver_support_class"] = "authoritative_locked"
    reach_csv = tmp_path / "reach.csv"
    _write_reach_csv(reach_csv)
    updated, outputs, summary = apply_xs_realism_to_nodes(nodes, river_dir=tmp_path, reach_attributes_path=reach_csv)
    prov_df = pd.read_csv(outputs["river_xs_realism_provenance_station_summary"])
    center = prov_df.loc[np.isclose(prov_df["station_m"].to_numpy(dtype=float), 10.0)].iloc[0]
    assert bool(center["station_bank_protected"]) is True
    assert bool(center["station_channel_protected"]) is False
    assert bool(center["station_inner_rebuildable"]) is True
    assert bool(center["station_core_authoritative"]) is False
    profile_df = pd.read_csv(outputs["xs_realism_profile"])
    profile_center = profile_df.loc[np.isclose(profile_df["station_m"].to_numpy(dtype=float), 10.0)].iloc[0]
    assert 0.0 < float(profile_center["left_inner_blend_weight"]) < 0.35
    assert 0.0 < float(profile_center["right_inner_blend_weight"]) < 0.35
    assert float(profile_center["xs_realism_core_influence_scale"]) < 0.60
    assert summary["channel_protected_station_count"] == 0
    assert summary["inner_rebuildable_station_count"] >= 1


def test_apply_xs_realism_does_not_promote_indirect_xs_to_measured_partial(tmp_path: Path):
    rows = []
    vals = {"left_bank": 10.0, "left_inner": 7.5, "thalweg": 4.0, "right_inner": 2.5, "right_bank": 6.0}
    for station in [0.0, 10.0, 20.0]:
        for role, z in vals.items():
            rows.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "bed_z_m": z,
                "z_source": "graph_backbone",
                "graph_solver_support_class": "graph_backbone",
                "graph_hard_lock": False,
                "station_support_mode": "xs_supported",
                "graph_candidate_source": "xs_profile_resampled",
            })
    nodes = pd.DataFrame(rows)
    reach_csv = tmp_path / "reach.csv"
    pd.DataFrame([{
        "profile_id": "main",
        "reach_id": "main:0",
        "reach_role": "interior",
        "station_min_m": 0.0,
        "station_max_m": 20.0,
        "unsupported_fraction": 0.85,
        "supported_fraction": 0.0,
        "authoritative_anchor_fraction": 0.0,
        "xs_support_fraction": 0.15,
        "junction_adjustment_station_fraction": 0.0,
        "component_junction_count": 0,
    }]).to_csv(reach_csv, index=False)
    _, outputs, summary = apply_xs_realism_to_nodes(nodes, river_dir=tmp_path, reach_attributes_path=reach_csv)
    profile = pd.read_csv(outputs["xs_realism_profile"])
    assert set(profile["station_provenance_class"].astype(str)) == {"indirect_xs"}
    assert "measured_xs_partial" not in set(profile["xs_support_template_class"].astype(str))
    assert set(profile["xs_template_type"].astype(str)) <= {"generic_symmetric", "generic_u", "hybrid_longitudinal"}
    assert summary["indirect_xs_station_count"] == 3


def test_apply_xs_realism_prefers_fitted_section_targets_in_bank_only_reach(tmp_path: Path):
    nodes = _build_nodes()
    for col, mapping in {
        "active_core_fit_z_m": {0.0: 4.0, 10.0: 3.5, 20.0: 3.0},
        "left_bank_fit_z_m": {0.0: 8.0, 10.0: 7.5, 20.0: 7.0},
        "right_bank_fit_z_m": {0.0: 8.2, 10.0: 7.7, 20.0: 7.2},
    }.items():
        nodes[col] = nodes["station_m"].map(mapping).astype(float)
    bank_mask = nodes["node_role"].isin(["left_bank", "right_bank"])
    nodes.loc[bank_mask, "z_source"] = "authoritative_bank"
    nodes.loc[bank_mask, "graph_hard_lock"] = True
    nodes.loc[bank_mask, "graph_solver_support_class"] = "authoritative_locked"
    reach_csv = tmp_path / "reach.csv"
    _write_reach_csv(reach_csv)

    _, outputs, _ = apply_xs_realism_to_nodes(nodes, river_dir=tmp_path, reach_attributes_path=reach_csv)
    profile = pd.read_csv(outputs["xs_realism_profile"])
    center = profile.loc[np.isclose(profile["station_m"].to_numpy(dtype=float), 10.0)].iloc[0]
    assert str(center["xs_support_template_class"]).startswith("bank_only")
    assert float(center["left_inner_ratio_target"]) > 0.30
    assert float(center["right_inner_ratio_target"]) < -0.30
    assert float(center["left_inner_blend_weight"]) >= 0.25
    assert float(center["right_inner_blend_weight"]) >= 0.25


def test_apply_xs_realism_can_be_disabled_globally(tmp_path: Path):
    nodes = _build_nodes()
    reach_csv = tmp_path / "reach.csv"
    _write_reach_csv(reach_csv)

    updated, outputs, summary = apply_xs_realism_to_nodes(
        nodes,
        river_dir=tmp_path,
        reach_attributes_path=reach_csv,
        disabled=True,
    )

    assert summary["available"] is False
    assert summary["disabled_by_option"] is True
    assert Path(outputs["xs_realism_summary"]).exists()
    assert np.allclose(updated["xs_realism_delta_m"].to_numpy(dtype=float), 0.0)
