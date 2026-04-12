import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_primary_surface_rebuild import (
    apply_primary_surface_rebuild_to_nodes,
    _station_inner_targets_from_tendency,
    _station_rebuild_weight,
    _support_distance_geometry_damping,
    _apply_support_aware_backbone_smoothing,
)
import river_primary_surface_rebuild as rpsr


def _nodes(bank_authoritative: bool = False, true_measured: bool = False, bed_authoritative: bool = False) -> pd.DataFrame:
    rows = []
    for station in [0.0, 10.0, 20.0]:
        for role, z in [("left_bank", 10.0), ("left_inner", 8.5), ("thalweg", 6.0), ("right_inner", 5.5), ("right_bank", 9.0)]:
            is_bank = role in {"left_bank", "right_bank"}
            rows.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "bed_z_m": z,
                "z_source": "authoritative_bank" if (bank_authoritative and is_bank) else ("authoritative_in_channel" if (true_measured or (bed_authoritative and not is_bank)) else "graph_backbone"),
                "graph_hard_lock": bool((bank_authoritative and is_bank) or true_measured or (bed_authoritative and not is_bank)),
                "xs_template_type": "generic_symmetric",
                "xs_support_template_class": "bank_only_low_confidence",
                "prediction_support_confidence": 0.25,
                "unsupported_fraction": 0.95,
                "xs_support_fraction": 0.05,
                "authoritative_anchor_fraction": 0.40 if bank_authoritative else 0.0,
                "station_true_measured_xs_fraction": 1.0 if true_measured else 0.0,
                "station_indirect_xs_fraction": 0.0,
                "station_residual_xs_fraction": 0.0,
                "station_authoritative_channel_fraction": 1.0 if (true_measured or bed_authoritative) else 0.0,
                "station_authoritative_bank_fraction": 1.0 if bank_authoritative else 0.0,
                "station_authoritative_bed_support_fraction": 1.0 if (true_measured or bed_authoritative) else 0.0,
                "station_authoritative_bank_margin_fraction": 1.0 if bank_authoritative else 0.0,
                "station_authoritative_bed_core_fraction": 0.8 if (true_measured or bed_authoritative) else 0.0,
                "station_authoritative_ambiguous_fraction": 0.0,
                "station_authoritative_bed_support_present": bool(true_measured or bed_authoritative),
                "station_authoritative_bank_margin_present": bool(bank_authoritative),
                "station_authoritative_role": "authoritative_bed_core" if (true_measured or bed_authoritative) else ("authoritative_bank_margin" if bank_authoritative else "no_authoritative_support"),
                "station_authoritative_bed_support_distance_m": 0.0 if (true_measured or bed_authoritative) else np.nan,
                "station_channel_protected": bool(true_measured),
                "station_inner_rebuildable": not true_measured,
                "component_support_class": "unsupported_mainstem",
                "backbone_bed_z_m": 3.0,
                "profile_network_backbone_z_m": 3.0,
                "graph_backbone_z_m": 3.0,
                "active_core_fit_z_m": 4.0,
                "left_bank_fit_z_m": 8.0,
                "right_bank_fit_z_m": 8.2,
                "bank_pair_fit_z_m": 8.1,
                "xs_realism_left_ratio_after": 0.45,
                "xs_realism_right_ratio_after": -0.45,
            })
    return pd.DataFrame(rows)


def test_rebuild_can_adjust_inner_nodes_when_only_banks_are_authoritative(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False)
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    center = updated.loc[np.isclose(pd.to_numeric(updated["station_m"], errors="coerce").to_numpy(dtype=float), 10.0)]
    inner = center.loc[center["node_role"].isin(["left_inner", "right_inner", "thalweg"])]
    assert bool(inner["primary_surface_rebuild_applied"].astype(bool).any())
    banks = center.loc[center["node_role"].isin(["left_bank", "right_bank"])]
    assert not bool(banks["primary_surface_rebuild_applied"].astype(bool).any())
    assert summary["adjusted_node_count"] > 0
    assert Path(outputs["primary_surface_rebuild_profile"]).exists()


def test_rebuild_stays_off_for_true_measured_sections(tmp_path: Path):
    nodes = _nodes(bank_authoritative=False, true_measured=True)
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    assert not bool(updated["primary_surface_rebuild_applied"].astype(bool).any())
    assert summary["adjusted_node_count"] == 0


def _nodes_roles(roles, bank_authoritative: bool = False, true_measured: bool = False, bed_authoritative: bool = False) -> pd.DataFrame:
    rows = []
    zmap = {"left_bank": 10.0, "left_inner": 8.5, "thalweg": 6.0, "right_inner": 5.5, "right_bank": 9.0}
    for station in [0.0, 10.0, 20.0]:
        for role in roles:
            z = zmap[role]
            is_bank = role in {"left_bank", "right_bank"}
            rows.append({
                "component_id": "main",
                "station_m": station,
                "node_role": role,
                "bed_z_m": z,
                "z_source": "authoritative_bank" if (bank_authoritative and is_bank) else ("authoritative_in_channel" if (true_measured or (bed_authoritative and not is_bank)) else "graph_backbone"),
                "graph_hard_lock": bool((bank_authoritative and is_bank) or true_measured or (bed_authoritative and not is_bank)),
                "xs_template_type": "generic_symmetric",
                "xs_support_template_class": "bank_only_low_confidence",
                "prediction_support_confidence": 0.25,
                "unsupported_fraction": 0.95,
                "xs_support_fraction": 0.05,
                "authoritative_anchor_fraction": 0.40 if bank_authoritative else 0.0,
                "station_true_measured_xs_fraction": 1.0 if true_measured else 0.0,
                "station_indirect_xs_fraction": 0.0,
                "station_residual_xs_fraction": 0.0,
                "station_authoritative_channel_fraction": 1.0 if (true_measured or bed_authoritative) else 0.0,
                "station_authoritative_bank_fraction": 1.0 if bank_authoritative else 0.0,
                "station_authoritative_bed_support_fraction": 1.0 if (true_measured or bed_authoritative) else 0.0,
                "station_authoritative_bank_margin_fraction": 1.0 if bank_authoritative else 0.0,
                "station_authoritative_bed_core_fraction": 0.8 if (true_measured or bed_authoritative) else 0.0,
                "station_authoritative_ambiguous_fraction": 0.0,
                "station_authoritative_bed_support_present": bool(true_measured or bed_authoritative),
                "station_authoritative_bank_margin_present": bool(bank_authoritative),
                "station_authoritative_role": "authoritative_bed_core" if (true_measured or bed_authoritative) else ("authoritative_bank_margin" if bank_authoritative else "no_authoritative_support"),
                "station_authoritative_bed_support_distance_m": 0.0 if (true_measured or bed_authoritative) else np.nan,
                "station_channel_protected": bool(true_measured),
                "station_inner_rebuildable": not true_measured,
                "component_support_class": "unsupported_mainstem",
                "backbone_bed_z_m": 3.0,
                "profile_network_backbone_z_m": 3.0,
                "graph_backbone_z_m": 3.0,
                "active_core_fit_z_m": 4.0,
                "left_bank_fit_z_m": 8.0,
                "right_bank_fit_z_m": 8.2,
                "bank_pair_fit_z_m": 8.1,
                "xs_realism_left_ratio_after": 0.45,
                "xs_realism_right_ratio_after": -0.45,
            })
    return pd.DataFrame(rows)


def test_rebuild_uses_thalweg_only_mode_when_inner_roles_missing(tmp_path: Path):
    nodes = _nodes_roles(["left_bank", "thalweg", "right_bank"], bank_authoritative=True, true_measured=False)
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    center = updated.loc[np.isclose(pd.to_numeric(updated["station_m"], errors="coerce").to_numpy(dtype=float), 10.0)]
    th = center.loc[center["node_role"] == "thalweg"].iloc[0]
    assert bool(th["primary_surface_rebuild_applied"])
    assert th["primary_surface_rebuild_mode"] == "thalweg_only_generic_rebuild"
    prof = pd.read_csv(outputs["primary_surface_rebuild_profile"])
    assert "thalweg_only_generic_rebuild" in set(prof["rebuild_mode"].astype(str))
    assert summary["adjusted_node_count"] > 0


def test_rebuild_uses_inner_only_mode_when_one_inner_role_exists(tmp_path: Path):
    nodes = _nodes_roles(["left_bank", "left_inner", "thalweg", "right_bank"], bank_authoritative=False, true_measured=False)
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    center = updated.loc[np.isclose(pd.to_numeric(updated["station_m"], errors="coerce").to_numpy(dtype=float), 10.0)]
    assert bool(center["primary_surface_rebuild_applied"].astype(bool).any())
    assert "channel_core_generic_rebuild" in set(center["primary_surface_rebuild_mode"].astype(str))
    prof = pd.read_csv(outputs["primary_surface_rebuild_profile"])
    assert "channel_core_generic_rebuild" in set(prof["rebuild_mode"].astype(str))
    assert summary["adjusted_node_count"] > 0


def test_rebuild_prioritizes_channel_core_and_writes_node_regimes(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False)
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    center = updated.loc[np.isclose(pd.to_numeric(updated["station_m"], errors="coerce").to_numpy(dtype=float), 10.0)]
    th = center.loc[center["node_role"] == "thalweg"].iloc[0]
    li = center.loc[center["node_role"] == "left_inner"].iloc[0]
    rb = center.loc[center["node_role"] == "right_bank"].iloc[0]
    assert th["primary_surface_rebuild_node_regime"] == "rebuild_channel_core"
    assert li["primary_surface_rebuild_node_regime"] == "rebuild_inner_generic"
    assert rb["primary_surface_rebuild_node_regime"] in {"protected_authoritative", "preserve_bank"}
    assert float(th["primary_surface_rebuild_weight"]) > float(li["primary_surface_rebuild_weight"]) > 0.0
    assert float(rb["primary_surface_rebuild_weight"]) == 0.0
    assert summary["changed_channel_core_node_count"] > 0


def test_bank_only_support_damps_inner_core_template_influence(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False, bed_authoritative=False)
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    center = updated.loc[np.isclose(pd.to_numeric(updated["station_m"], errors="coerce").to_numpy(dtype=float), 10.0)]
    th = center.loc[center["node_role"] == "thalweg"].iloc[0]
    li = center.loc[center["node_role"] == "left_inner"].iloc[0]
    assert float(li["primary_surface_rebuild_weight"]) < 0.60 * float(th["primary_surface_rebuild_weight"])
    profile_df = pd.read_csv(outputs["primary_surface_rebuild_profile"])
    profile_center = profile_df.loc[np.isclose(profile_df["station_m"].to_numpy(dtype=float), 10.0)].iloc[0]
    assert float(profile_center["core_geometry_scale"]) < 0.60
    assert float(profile_center["inner_weight_scale"]) < 0.60
    assert summary["core_geometry_scale_summary"]["median"] < 0.60
    assert summary["inner_weight_scale_summary"]["median"] < 0.60


def test_rebuild_stays_off_for_authoritative_bed_supported_sections(tmp_path: Path):
    nodes = _nodes(bank_authoritative=False, true_measured=False, bed_authoritative=True)
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    assert not bool(updated["primary_surface_rebuild_applied"].astype(bool).any())
    assert summary["adjusted_node_count"] == 0


def test_bank_only_support_uses_generic_channel_core_mode_not_full_section(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False, bed_authoritative=False)
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    center = updated.loc[np.isclose(pd.to_numeric(updated["station_m"], errors="coerce").to_numpy(dtype=float), 10.0)]
    assert "channel_core_generic_rebuild" in set(center["primary_surface_rebuild_mode"].astype(str))
    assert "full_section_rebuild" not in set(center["primary_surface_rebuild_mode"].astype(str))


def test_post_rebuild_monotone_projection_reduces_channel_core_violations(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False, bed_authoritative=False)
    custom = {
        0.0: {"thalweg": 2.0, "left_inner": 4.0, "right_inner": 3.8},
        10.0: {"thalweg": 4.5, "left_inner": 6.0, "right_inner": 5.8},
        20.0: {"thalweg": 3.0, "left_inner": 4.8, "right_inner": 4.6},
    }
    for idx, row in nodes.iterrows():
        z = custom.get(float(row["station_m"]), {}).get(str(row["node_role"]))
        if z is not None:
            nodes.at[idx, "bed_z_m"] = z
        nodes.at[idx, "profile_inside_fluvial_monotone_domain"] = True
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    profile = pd.read_csv(outputs["post_rebuild_monotone_projection_profile"])
    th = profile.loc[profile["node_role"].astype(str) == "thalweg"].sort_values("station_m")
    assert int(summary["post_rebuild_monotone_pre_violation_count"]) > 0
    assert int(summary["post_rebuild_monotone_post_violation_count"]) == 0
    assert bool(th["projection_applied"].astype(bool).any())
    post = th["post_projection_z_m"].to_numpy(dtype=float)
    assert np.all(np.diff(post[np.isfinite(post)]) <= 1.0e-6)


def test_post_rebuild_monotone_projection_preserves_hard_channel_anchors(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False, bed_authoritative=False)
    for idx, row in nodes.iterrows():
        nodes.at[idx, "profile_inside_fluvial_monotone_domain"] = True
        if float(row["station_m"]) == 10.0 and str(row["node_role"]) == "thalweg":
            nodes.at[idx, "z_source"] = "authoritative_in_channel"
            nodes.at[idx, "graph_hard_lock"] = True
            nodes.at[idx, "bed_z_m"] = 4.25
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    center = updated.loc[(pd.to_numeric(updated["station_m"], errors="coerce") == 10.0) & (updated["node_role"].astype(str) == "thalweg")].iloc[0]
    assert float(center["bed_z_m"]) == 4.25
    assert int(summary["post_rebuild_monotone_post_violation_count"]) == 0
    import json
    monotone_summary = json.loads(Path(outputs["post_rebuild_monotone_projection_summary"]).read_text(encoding="utf-8"))
    assert monotone_summary["post_violation_count"] == 0


def test_rebuild_prefers_generalized_longitudinal_section_in_bank_only_reaches(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False, bed_authoritative=False)
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    center = updated.loc[np.isclose(pd.to_numeric(updated["station_m"], errors="coerce").to_numpy(dtype=float), 10.0)]
    th = center.loc[center["node_role"].astype(str) == "thalweg"].iloc[0]
    li = center.loc[center["node_role"].astype(str) == "left_inner"].iloc[0]
    assert th["primary_surface_rebuild_target_source"] == "generalized_longitudinal_section"
    assert li["primary_surface_rebuild_target_source"] == "generalized_longitudinal_section"
    assert float(th["primary_surface_rebuild_target_z_m"]) > 3.0
    assert float(th["primary_surface_rebuild_target_z_m"]) < 6.0
    profile_df = pd.read_csv(outputs["primary_surface_rebuild_profile"])
    assert "generalized_longitudinal_section" in set(profile_df["primary_surface_rebuild_target_source"].astype(str))
    assert summary["target_source_counts"].get("generalized_longitudinal_section", 0) > 0


def test_rebuild_can_be_disabled_globally(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False)
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path, disabled=True)
    assert summary["available"] is False
    assert summary["disabled_by_option"] is True
    assert not bool(updated["primary_surface_rebuild_applied"].astype(bool).any())
    assert Path(outputs["primary_surface_rebuild_summary"]).exists()


def test_rebuild_preserves_local_authoritative_reconciliation_target_source(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False, bed_authoritative=False)
    nodes["station_target_present"] = True
    nodes["station_target_local_authoritative_reconciled"] = True
    nodes["station_target_source_class"] = "generalized_longitudinal_section_local_authoritative_reconciliation"
    nodes["target_thalweg_z_m"] = 4.25
    nodes["target_left_bank_z_m"] = 8.0
    nodes["target_right_bank_z_m"] = 8.2
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    center = updated.loc[np.isclose(pd.to_numeric(updated["station_m"], errors="coerce").to_numpy(dtype=float), 10.0)]
    th = center.loc[center["node_role"].astype(str) == "thalweg"].iloc[0]
    assert th["primary_surface_rebuild_target_source"] == "generalized_longitudinal_section_local_authoritative_reconciliation"
    profile_df = pd.read_csv(outputs["primary_surface_rebuild_profile"])
    assert "generalized_longitudinal_section_local_authoritative_reconciliation" in set(profile_df["primary_surface_rebuild_target_source"].astype(str))
    assert summary["target_source_counts"].get("generalized_longitudinal_section_local_authoritative_reconciliation", 0) > 0


def test_rebuild_prefers_backbone_bed_reference_over_profile_backbone(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False, bed_authoritative=False)
    nodes["backbone_bed_z_m"] = 4.25
    nodes["profile_network_backbone_z_m"] = 2.5
    nodes["graph_backbone_z_m"] = 2.0
    _, outputs, _ = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    profile_df = pd.read_csv(outputs["primary_surface_rebuild_profile"])
    center = profile_df.loc[np.isclose(profile_df["station_m"].to_numpy(dtype=float), 10.0)].iloc[0]
    assert np.isclose(float(center["backbone_target_z_m"]), 4.25)


def test_unsupported_thalweg_backbone_guardrail_clamps_to_backbone_band(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False, bed_authoritative=False)
    nodes["component_support_class"] = "unsupported_mainstem"
    nodes["backbone_bed_z_m"] = 5.6
    nodes["profile_network_backbone_z_m"] = 2.5
    nodes["graph_backbone_z_m"] = 2.0
    nodes["active_core_fit_z_m"] = 2.0
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    center = updated.loc[np.isclose(pd.to_numeric(updated["station_m"], errors="coerce").to_numpy(dtype=float), 10.0)]
    th = center.loc[center["node_role"].astype(str) == "thalweg"].iloc[0]
    tol = float(th["primary_surface_backbone_guardrail_tolerance_m"])
    ref = float(th["primary_surface_backbone_guardrail_reference_z_m"])
    assert bool(th["primary_surface_backbone_guardrail_applied"])
    assert abs(float(th["bed_z_m"]) - ref) <= tol + 1.0e-6
    assert summary["backbone_guardrail_adjusted_node_count"] > 0
    assert Path(outputs["primary_surface_backbone_guardrail_profile"]).exists()
    assert Path(outputs["primary_surface_backbone_guardrail_summary"]).exists()



def test_station_inner_targets_from_tendency_prefers_explicit_offsets_over_bank_relief():
    row = pd.Series({
        "target_thalweg_z_m": 4.0,
        "target_left_inner_z_m": 4.9,
        "target_right_inner_z_m": 4.7,
        "target_left_bank_z_m": 10.0,
        "target_right_bank_z_m": 9.0,
        "target_effective_channel_width_m": 20.0,
        "section_tendency_family": "flat_u",
        "section_tendency_depth_m": 0.18,
        "section_tendency_inner_relief_m": 0.9,
        "station_authoritative_bed_support_class": "bank_margin_only",
    })
    left_inner, right_inner = _station_inner_targets_from_tendency(
        row,
        target_th=3.5,
        left_bank_target=10.0,
        right_bank_target=9.0,
    )
    assert np.isclose(left_inner, 4.4)
    assert np.isclose(right_inner, 4.2)



def test_station_inner_targets_from_tendency_damps_explicit_offsets_in_backbone_led_bank_only_reaches():
    row = pd.Series({
        "target_thalweg_z_m": 4.0,
        "target_left_inner_z_m": 4.9,
        "target_right_inner_z_m": 4.7,
        "target_left_bank_z_m": 10.0,
        "target_right_bank_z_m": 9.0,
        "target_effective_channel_width_m": 20.0,
        "section_tendency_family": "flat_u",
        "section_tendency_depth_m": 0.18,
        "section_tendency_inner_relief_m": 0.9,
        "station_authoritative_bed_support_class": "no_authoritative_bed_support",
        "station_support_regime": "bank_only_low_confidence",
        "component_support_class": "unsupported_mainstem",
    })
    left_inner, right_inner = _station_inner_targets_from_tendency(
        row,
        target_th=3.5,
        left_bank_target=10.0,
        right_bank_target=9.0,
    )
    assert left_inner > 3.5
    assert right_inner > 3.5
    assert left_inner < 4.4
    assert right_inner < 4.2


def test_support_distance_geometry_damping_treats_missing_unsupported_as_far():
    row = pd.Series({
        "station_authoritative_bed_support_present": False,
        "station_support_regime": "bank_only_low_confidence",
    })
    ratio_scale, inner_scale = _support_distance_geometry_damping(row)
    assert ratio_scale < 0.5
    assert inner_scale < 0.3


def test_station_rebuild_weight_caps_far_unsupported_generic_sections():
    row = pd.Series({
        "unsupported_fraction": 0.95,
        "xs_support_fraction": 0.05,
        "authoritative_anchor_fraction": 0.0,
        "prediction_support_confidence": 0.25,
        "xs_template_type": "generic_symmetric",
        "xs_support_template_class": "bank_only_low_confidence",
        "station_support_regime": "bank_only_low_confidence",
        "station_authoritative_bank_fraction": 0.0,
        "station_authoritative_bed_support_present": False,
        "station_authoritative_bed_support_distance_m": 1500.0,
        "station_inner_rebuildable": True,
        "station_channel_protected": False,
        "station_true_measured_xs_fraction": 0.0,
        "station_indirect_xs_fraction": 0.0,
        "station_residual_xs_fraction": 0.0,
        "station_authoritative_channel_fraction": 0.0,
        "station_authoritative_bank_margin_fraction": 0.0,
        "station_authoritative_bed_support_fraction": 0.0,
        "station_authoritative_bed_core_fraction": 0.0,
        "station_authoritative_ambiguous_fraction": 0.0,
        "station_authoritative_role": "no_authoritative_support",
    })
    weight = _station_rebuild_weight(
        row,
        station_authoritative=False,
        station_measured_xs=False,
        station_channel_protected=False,
        station_inner_rebuildable=True,
    )
    assert weight <= 0.24


def test_station_rebuild_weight_uses_reconciliation_distance_fallback():
    row = pd.Series({
        "unsupported_fraction": 0.95,
        "xs_support_fraction": 0.05,
        "authoritative_anchor_fraction": 0.0,
        "prediction_support_confidence": 0.25,
        "xs_template_type": "generic_symmetric",
        "xs_support_template_class": "bank_only_low_confidence",
        "station_support_regime": "bank_only_low_confidence",
        "station_authoritative_bank_fraction": 0.0,
        "station_authoritative_bed_support_present": False,
        "authoritative_reconciliation_support_distance_m": 1200.0,
        "station_inner_rebuildable": True,
        "station_channel_protected": False,
        "station_true_measured_xs_fraction": 0.0,
        "station_indirect_xs_fraction": 0.0,
        "station_residual_xs_fraction": 0.0,
        "station_authoritative_channel_fraction": 0.0,
        "station_authoritative_bank_margin_fraction": 0.0,
        "station_authoritative_bed_support_fraction": 0.0,
        "station_authoritative_bed_core_fraction": 0.0,
        "station_authoritative_ambiguous_fraction": 0.0,
        "station_authoritative_role": "no_authoritative_support",
    })
    weight = _station_rebuild_weight(
        row,
        station_authoritative=False,
        station_measured_xs=False,
        station_channel_protected=False,
        station_inner_rebuildable=True,
    )
    assert weight <= 0.24


def test_backbone_smoothing_adjusts_only_weak_support_stations(tmp_path: Path):
    nodes = _nodes(bank_authoritative=False, true_measured=False, bed_authoritative=False)
    # Introduce oscillatory backbone targets while keeping middle station transition-like.
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 0.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 3.0
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 10.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 5.0
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 20.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 3.2
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 10.0), "station_support_regime"] = "supported_transition"
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    prof = pd.read_csv(outputs["backbone_smoothing_profile"])
    assert Path(outputs["backbone_smoothing_summary"]).exists()
    assert int(summary["backbone_smoothing_adjusted_station_count"]) > 0
    center = prof.loc[np.isclose(pd.to_numeric(prof["station_m"], errors="coerce"), 10.0)].iloc[0]
    assert bool(center["backbone_smoothing_applied"])
    assert abs(float(center["backbone_smoothing_delta_m"])) <= 0.20 + 1.0e-6
    left = prof.loc[np.isclose(pd.to_numeric(prof["station_m"], errors="coerce"), 0.0)].iloc[0]
    assert bool(left["backbone_smoothing_applied"])



def test_backbone_smoothing_handles_non_range_index_without_label_position_mismatch():
    station_df = pd.DataFrame(
        {
            "component_id": ["main", "main", "main"],
            "station_m": [0.0, 50.0, 100.0],
            "backbone_target_z_m": [-1.0, -2.2, -3.0],
            "component_support_class": ["unsupported_mainstem"] * 3,
            "station_support_regime": ["unsupported"] * 3,
            "station_authoritative": [False, False, False],
            "station_measured_xs": [False, False, False],
            "station_authoritative_bed_support_class": ["no_authoritative_bed_support"] * 3,
        },
        index=[10, 20, 30],
    )

    smoothed_df, profile_df, summary = _apply_support_aware_backbone_smoothing(station_df)

    assert list(smoothed_df.index) == [10, 20, 30]
    assert summary["candidate_station_count"] == 3
    assert len(profile_df) == 3
    assert np.isfinite(profile_df["original_backbone_target_z_m"]).all()

def test_backbone_smoothing_skips_authoritative_station(tmp_path: Path):
    nodes = _nodes(bank_authoritative=False, true_measured=False, bed_authoritative=False)
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 10.0), "z_source"] = "authoritative_in_channel"
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 10.0), "graph_hard_lock"] = True
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 10.0), "station_authoritative_channel_fraction"] = 1.0
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 10.0), "station_authoritative_bed_core_fraction"] = 1.0
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 10.0), "station_authoritative_bed_support_present"] = True
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 10.0), "station_authoritative_bed_support_distance_m"] = 0.0
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    prof = pd.read_csv(outputs["backbone_smoothing_profile"])
    center = prof.loc[np.isclose(pd.to_numeric(prof["station_m"], errors="coerce"), 10.0)].iloc[0]
    assert not bool(center["backbone_smoothing_eligible"])
    assert not bool(center["backbone_smoothing_applied"])


def test_backbone_smoothing_reports_linear_assist(tmp_path: Path):
    nodes = _nodes(bank_authoritative=False, true_measured=False, bed_authoritative=False)
    extra = nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 20.0)].copy()
    extra["station_m"] = 30.0
    nodes = pd.concat([nodes, extra], ignore_index=True)
    # Create a weak-support trend where a leave-one-out weighted line is more informative than a simple local mean.
    station = pd.to_numeric(nodes["station_m"], errors="coerce")
    nodes.loc[np.isclose(station, 0.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 0.0
    nodes.loc[np.isclose(station, 10.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 4.0
    nodes.loc[np.isclose(station, 20.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 1.0
    nodes.loc[np.isclose(station, 30.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 12.0
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    prof = pd.read_csv(outputs["backbone_smoothing_profile"])
    assert int(summary.get("backbone_smoothing_linear_assisted_count", 0)) >= 1
    target = prof.loc[np.isclose(pd.to_numeric(prof["station_m"], errors="coerce"), 20.0)].iloc[0]
    assert bool(target["backbone_smoothing_applied"])
    assert float(target["backbone_smoothing_delta_m"]) > 0.0
    assert float(target["smoothed_backbone_target_z_m"]) > float(target["backbone_target_z_m"])



def test_backbone_smoothing_summary_includes_grouped_receipts():
    station_df = pd.DataFrame({
        'component_id': ['main', 'main', 'main'],
        'station_m': [0.0, 50.0, 100.0],
        'backbone_target_z_m': [-1.0, -2.2, -3.0],
        'component_support_class': ['unsupported_mainstem'] * 3,
        'station_support_regime': ['unsupported'] * 3,
        'station_authoritative': [False, False, False],
        'station_measured_xs': [False, False, False],
        'station_authoritative_bed_support_class': ['no_authoritative_bed_support'] * 3,
    })
    _, profile_df, summary = _apply_support_aware_backbone_smoothing(station_df)
    assert summary['eligible_station_count'] == 3
    assert summary['candidate_station_count'] == 3
    assert 'unsupported_mainstem' in summary['by_component_support_class']
    assert 'unsupported' in summary['by_station_support_regime']
    bucket = summary['by_component_support_class']['unsupported_mainstem']
    assert bucket['station_count'] == 3
    assert bucket['candidate_station_count'] == 3
    assert summary['unsupported_mainstem_candidate_station_count'] == 3
    assert 'candidate_evaluated' in profile_df.columns
    assert 'backbone_smoothing_reference_source' in profile_df.columns



def test_primary_surface_rebuild_writes_centerline_width_propagation_receipts(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False)
    _, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    profile_path = Path(outputs['centerline_width_propagation_profile'])
    summary_path = Path(outputs['centerline_width_propagation_summary'])
    assert profile_path.exists()
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding='utf-8'))
    assert payload['available'] is True
    assert payload['backbone_led_station_count'] > 0
    assert payload['bank_margin_damping_station_count'] > 0
    assert 'unsupported_mainstem' in payload['by_component_support_class']
    assert summary['backbone_led_inner_target_station_count'] > 0



def test_rebuild_emits_canonical_support_and_active_target_fields(tmp_path: Path):
    nodes = _nodes(bank_authoritative=True, true_measured=False)
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    assert "support_class_canonical" in updated.columns
    assert "active_interior_target_source" in updated.columns
    assert "active_interior_target_z_m" in updated.columns
    assert set(updated["support_class_canonical"].astype(str)) == {"bank_margin_only"}
    assert "backbone_led_interior" in set(updated["active_interior_target_source"].astype(str))
    assert "canonical_support_class_counts" in summary
    assert summary["active_interior_target_source_counts"]["backbone_led_interior"] > 0


def test_backbone_smoothing_action_gate_receipt_written(tmp_path: Path):
    nodes = _nodes(bank_authoritative=False, true_measured=False, bed_authoritative=False)
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 0.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 3.0
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 10.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 5.0
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 20.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 3.2
    updated, outputs, summary = apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
    receipt_path = Path(outputs["backbone_action_gate_receipt"])
    assert receipt_path.exists()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["available"] is True
    assert payload["candidate_station_count_unsupported_mainstem"] >= 1
    assert payload["first_blocking_gate"] in {"passed", "requested_adjustment_gate", "application_gate", "clamp_gate", "eligibility_gate"}
    prof = pd.read_csv(outputs["backbone_smoothing_profile"])
    assert "backbone_candidate_group" in prof.columns
    assert "backbone_candidate_exclusion_reason" in prof.columns



def test_backbone_smoothing_fails_honestly_when_requested_adjustments_never_apply(tmp_path: Path, monkeypatch):
    nodes = _nodes(bank_authoritative=False, true_measured=False, bed_authoritative=False)
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 0.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 0.0
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 10.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 4.0
    nodes.loc[np.isclose(pd.to_numeric(nodes["station_m"], errors="coerce"), 20.0), ["profile_network_backbone_z_m", "backbone_bed_z_m"]] = 1.0

    monkeypatch.setattr(rpsr, "_station_backbone_smoothing_strength", lambda row: 0.0 if rpsr._station_backbone_smoothing_eligible(row) else 0.0)
    monkeypatch.setattr(rpsr, "_station_backbone_smoothing_max_shift", lambda row: 1.15)

    try:
        apply_primary_surface_rebuild_to_nodes(nodes, river_dir=tmp_path)
        assert False, "Expected RuntimeError for inert unsupported-mainstem backbone action"
    except RuntimeError as exc:
        assert "requested adjustments but no applied backbone action" in str(exc)
    receipt_path = tmp_path / "river_backbone_action_gate_receipt.json"
    assert receipt_path.exists()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["should_fail"] is True
    assert payload["unsupported_mainstem_gate_counts"]["requested_nonzero_count"] > 0
    assert payload["unsupported_mainstem_gate_counts"]["applied_nonzero_count"] == 0
