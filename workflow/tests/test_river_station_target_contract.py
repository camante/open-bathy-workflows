import pandas as pd

from river_station_target_contract import build_station_target_table


def test_build_station_target_table_uses_generalized_longitudinal_bed_elevation_alias():
    frame = pd.DataFrame(
        {
            "component_id": ["main"],
            "station_m": [100.0],
            "graph_solver_support_class": ["unsupported"],
            "channel_support_class": ["unsupported"],
            "profile_inside_fluvial_monotone_domain": [True],
            "generalized_longitudinal_bed_elevation_m": [1.25],
            "generalized_longitudinal_bed_source": ["generalized_longitudinal_bed"],
            "active_core_support_z_m": [9.5],
            "authoritative_anchor_present": [False],
            "authoritative_anchor_curve_present": [False],
            "true_measured_xs_qualified": [False],
            "authoritative_bank_margin_present": [False],
            "authoritative_bed_support_present": [False],
        }
    )
    target, summary = build_station_target_table(frame)
    row = target.iloc[0]
    assert float(row["active_core_support_z_m"]) == 1.25
    assert float(row["target_thalweg_z_m"]) == 1.25
    assert str(row["target_source_class"]) == "generalized_longitudinal_section"
    assert int(summary["active_core_target_count"]) == 1


def test_build_station_target_table_uses_thalweg_centric_section_tendency_not_bank_relief():
    frame = pd.DataFrame(
        {
            "component_id": ["main"],
            "station_m": [100.0],
            "graph_solver_support_class": ["unsupported"],
            "channel_support_class": ["unsupported"],
            "profile_inside_fluvial_monotone_domain": [True],
            "generalized_longitudinal_bed_elevation_m": [1.0],
            "generalized_longitudinal_bed_source": ["generalized_longitudinal_bed"],
            "left_bank_fit_z_m": [10.0],
            "right_bank_fit_z_m": [12.0],
            "authoritative_distance_to_bank_m": [10.0],
            "authoritative_anchor_present": [False],
            "authoritative_anchor_curve_present": [False],
            "true_measured_xs_qualified": [False],
            "authoritative_bank_margin_present": [False],
            "authoritative_bed_support_present": [False],
        }
    )
    target, summary = build_station_target_table(frame)
    row = target.iloc[0]
    assert float(row["target_thalweg_z_m"]) == 1.0
    assert str(row["section_tendency_family"]) == "flat_u"
    assert str(row["section_tendency_source"]) == "support_class_width_depth_tendency"
    assert float(row["target_left_inner_z_m"]) < 3.0
    assert float(row["target_right_inner_z_m"]) < 3.0
    assert int(summary["section_tendency_family_counts"]["flat_u"]) == 1


def test_build_station_target_table_prefers_reconciled_generalized_bed_fields():
    frame = pd.DataFrame(
        {
            "component_id": ["main"],
            "station_m": [100.0],
            "graph_solver_support_class": ["unsupported"],
            "channel_support_class": ["unsupported"],
            "profile_inside_fluvial_monotone_domain": [True],
            "generalized_longitudinal_bed_base_elevation_m": [1.25],
            "generalized_longitudinal_bed_reconciled_elevation_m": [0.95],
            "authoritative_reconciliation_delta_m": [-0.30],
            "authoritative_reconciliation_weight": [0.75],
            "generalized_longitudinal_bed_source": ["generalized_longitudinal_bed_local_authoritative_reconciliation"],
            "authoritative_anchor_present": [False],
            "authoritative_anchor_curve_present": [False],
            "true_measured_xs_qualified": [False],
            "authoritative_bank_margin_present": [False],
            "authoritative_bed_support_present": [False],
        }
    )
    target, _ = build_station_target_table(frame)
    row = target.iloc[0]
    assert abs(float(row["target_thalweg_z_m"]) - 0.95) < 1e-6
    assert abs(float(row["authoritative_reconciliation_delta_m"]) + 0.30) < 1e-6
    assert str(row["target_source_class"]) == "generalized_longitudinal_section_local_authoritative_reconciliation"


def test_build_station_target_table_weak_component_simplifies_section_tendency():
    frame = pd.DataFrame(
        {
            "component_id": ["trib"],
            "component_support_class": ["tiny_detached_component"],
            "station_m": [100.0],
            "graph_solver_support_class": ["unsupported"],
            "channel_support_class": ["unsupported"],
            "profile_inside_fluvial_monotone_domain": [True],
            "generalized_longitudinal_bed_elevation_m": [1.0],
            "generalized_longitudinal_bed_source": ["generalized_longitudinal_bed"],
            "left_bank_fit_z_m": [10.0],
            "right_bank_fit_z_m": [12.0],
            "authoritative_distance_to_bank_m": [10.0],
            "authoritative_reconciliation_confidence": [0.1],
            "authoritative_reconciliation_support_distance_m": [400.0],
            "authoritative_anchor_present": [False],
            "authoritative_anchor_curve_present": [False],
            "true_measured_xs_qualified": [False],
            "authoritative_bank_margin_present": [False],
            "authoritative_bed_support_present": [False],
        }
    )
    target, _ = build_station_target_table(frame)
    row = target.iloc[0]
    assert str(row["section_tendency_source"]) == "weak_component_simplified_tendency"
    assert float(row["section_tendency_depth_m"]) < 0.35
    assert float(row["section_tendency_confidence"]) < 0.5
    assert float(row["target_left_inner_z_m"]) < 2.5


def test_build_station_target_table_marks_unsupported_mainstem_as_mainstem_calibrated():
    frame = pd.DataFrame(
        {
            "component_id": ["main"],
            "component_support_class": ["unsupported_mainstem"],
            "station_m": [100.0],
            "graph_solver_support_class": ["unsupported"],
            "channel_support_class": ["unsupported"],
            "profile_inside_fluvial_monotone_domain": [True],
            "generalized_longitudinal_bed_elevation_m": [1.0],
            "generalized_longitudinal_bed_source": ["generalized_longitudinal_bed_local_authoritative_reconciliation"],
            "left_bank_fit_z_m": [4.0],
            "right_bank_fit_z_m": [4.5],
            "authoritative_distance_to_bank_m": [18.0],
            "authoritative_reconciliation_delta_m": [0.15],
            "authoritative_reconciliation_confidence": [0.65],
            "authoritative_reconciliation_support_distance_m": [120.0],
            "authoritative_anchor_present": [False],
            "authoritative_anchor_curve_present": [False],
            "true_measured_xs_qualified": [False],
            "authoritative_bank_margin_present": [False],
            "authoritative_bed_support_present": [False],
        }
    )
    target, _ = build_station_target_table(frame)
    row = target.iloc[0]
    assert str(row["section_tendency_source"]) == "mainstem_calibrated_tendency_local_authoritative_reconciliation"
    assert float(row["section_tendency_confidence"]) > 0.55
    assert float(row["section_tendency_depth_m"]) > 0.35
