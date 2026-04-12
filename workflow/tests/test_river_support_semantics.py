import math

from river_support_semantics import (
    LONGITUDINAL_WEAK_SUPPORT,
    MEASURED_CLASS_STRONG,
    station_semantics,
)


def test_station_semantics_promotes_far_measured_gap_to_generic_rebuild():
    row = {
        "station_true_measured_xs_fraction": 0.0,
        "station_indirect_xs_fraction": 0.0,
        "station_residual_xs_fraction": 0.0,
        "station_authoritative_fraction": 0.0,
        "station_authoritative_channel_fraction": 0.0,
        "station_authoritative_bank_fraction": 0.0,
        "unsupported_fraction": 0.0,
        "xs_support_fraction": 1.0,
        "authoritative_anchor_fraction": 0.0,
        "prediction_support_confidence": 0.3,
        "measured_support_distance_median_m": 1200.0,
        "far_from_measured_fraction": 1.0,
        "station_inner_rebuildable": True,
    }
    sem = station_semantics(row)
    assert sem["station_far_from_measured"] is True
    assert math.isclose(float(sem["station_measured_support_distance_m"]), 1200.0)
    assert sem["station_template_type"] == "generic_symmetric"
    assert sem["station_rebuild_regime"] == "eligible"
    assert sem["station_support_regime"] == "bank_only_low_confidence"
    assert sem["station_measured_support_envelope_class"] == "unsupported_upstream_generic"
    assert sem["longitudinal_support_regime"] == LONGITUDINAL_WEAK_SUPPORT


def test_station_semantics_keeps_true_measured_sections_protected():
    row = {
        "station_true_measured_xs_fraction": 0.8,
        "station_true_measured_xs_qualified": True,
        "station_authoritative_channel_fraction": 0.7,
        "station_authoritative_fraction": 0.7,
        "unsupported_fraction": 0.9,
        "xs_support_fraction": 0.1,
        "measured_support_distance_median_m": 0.0,
        "station_inner_rebuildable": False,
    }
    sem = station_semantics(row)
    assert sem["station_measured_support_class"] == MEASURED_CLASS_STRONG
    assert sem["station_template_type"] == "measured"
    assert sem["station_rebuild_regime"] == "blocked_true_measured_xs"
    assert sem["station_protection_regime"] == "true_measured_protected"


def test_station_semantics_treats_authoritative_bed_as_hard_control_and_bank_margin_as_boundary_only():
    bed_row = {
        "station_true_measured_xs_fraction": 0.0,
        "station_authoritative_fraction": 0.8,
        "station_authoritative_channel_fraction": 0.8,
        "station_authoritative_bank_fraction": 0.0,
        "station_authoritative_bed_support_fraction": 0.8,
        "station_authoritative_bank_margin_fraction": 0.0,
        "station_authoritative_bed_core_fraction": 0.6,
        "station_authoritative_ambiguous_fraction": 0.0,
        "station_authoritative_bed_support_present": True,
        "station_authoritative_bank_margin_present": False,
        "station_authoritative_role": "authoritative_bed_core",
        "station_authoritative_bed_support_distance_m": 0.0,
        "unsupported_fraction": 0.9,
        "xs_support_fraction": 0.05,
        "station_inner_rebuildable": True,
    }
    bed_sem = station_semantics(bed_row)
    assert bed_sem["station_authoritative_bed_support_class"] == "authoritative_bed_strong"
    assert bed_sem["station_protection_regime"] == "authoritative_bed_protected"
    assert bed_sem["station_rebuild_regime"] == "blocked_authoritative_bed_support"
    assert bed_sem["longitudinal_support_regime"] != LONGITUDINAL_WEAK_SUPPORT

    bank_row = {
        "station_true_measured_xs_fraction": 0.0,
        "station_authoritative_fraction": 0.4,
        "station_authoritative_channel_fraction": 0.0,
        "station_authoritative_bank_fraction": 0.7,
        "station_authoritative_bed_support_fraction": 0.0,
        "station_authoritative_bank_margin_fraction": 0.7,
        "station_authoritative_bed_core_fraction": 0.0,
        "station_authoritative_ambiguous_fraction": 0.0,
        "station_authoritative_bed_support_present": False,
        "station_authoritative_bank_margin_present": True,
        "station_authoritative_role": "authoritative_bank_margin",
        "station_authoritative_bed_support_distance_m": math.nan,
        "unsupported_fraction": 0.9,
        "xs_support_fraction": 0.05,
        "station_bank_protected": True,
        "station_inner_rebuildable": True,
    }
    bank_sem = station_semantics(bank_row)
    assert bank_sem["station_authoritative_bed_support_class"] == "authoritative_bank_margin_only"
    assert bank_sem["station_protection_regime"] == "bank_only_protected"
    assert bank_sem["station_rebuild_regime"] == "eligible"
    assert bank_sem["longitudinal_support_regime"] == LONGITUDINAL_WEAK_SUPPORT
