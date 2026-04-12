from river_section_tendency import (
    build_section_from_thalweg,
    classify_section_tendency_family,
    compute_tendency_confidence,
    compute_tendency_depth_fraction,
)


def test_section_tendency_classification_and_depth_fraction_are_deterministic():
    assert classify_section_tendency_family(8.0, "unsupported") == "narrow_v"
    assert classify_section_tendency_family(60.0, "unsupported") == "compound_lowflow"
    assert classify_section_tendency_family(18.0, "bed_supported") == "authoritative_section_derived"
    assert compute_tendency_depth_fraction(8.0, "narrow_v", "unsupported") > compute_tendency_depth_fraction(60.0, "compound_lowflow", "unsupported")


def test_build_section_from_thalweg_respects_bank_caps():
    left_inner, right_inner, relief = build_section_from_thalweg(1.0, 20.0, "flat_u", 0.2, bank_caps=(1.6, 10.0))
    assert relief > 0.0
    assert left_inner <= 1.55
    assert right_inner >= left_inner


def test_bank_margin_only_sections_are_shallower_and_lower_confidence_than_bed_supported():
    frac_bed = compute_tendency_depth_fraction(24.0, "authoritative_section_derived", "bed_supported")
    frac_bank = compute_tendency_depth_fraction(24.0, "flat_u", "bank_margin_only")
    conf_bed = compute_tendency_confidence(24.0, "authoritative_section_derived", "bed_supported")
    conf_bank = compute_tendency_confidence(24.0, "flat_u", "bank_margin_only")
    assert frac_bed > frac_bank
    assert conf_bed > conf_bank


def test_weak_component_tendency_is_shallower_than_generic_unsupported():
    generic = compute_tendency_depth_fraction(24.0, "flat_u", "unsupported")
    weak = compute_tendency_depth_fraction(
        24.0,
        "flat_u",
        "unsupported",
        component_class="unsupported_side_component",
        reconciliation_confidence=0.1,
        support_distance_m=400.0,
    )
    tiny = compute_tendency_depth_fraction(
        24.0,
        "flat_u",
        "unsupported",
        component_class="tiny_detached_component",
        reconciliation_confidence=0.1,
        support_distance_m=400.0,
    )
    assert weak < generic
    assert tiny < weak


def test_weak_component_tendency_confidence_is_lower_than_generic_unsupported():
    generic = compute_tendency_confidence(24.0, "flat_u", "unsupported")
    weak = compute_tendency_confidence(
        24.0,
        "flat_u",
        "unsupported",
        component_class="unsupported_side_component",
        reconciliation_confidence=0.1,
        support_distance_m=400.0,
    )
    tiny = compute_tendency_confidence(
        24.0,
        "flat_u",
        "unsupported",
        component_class="tiny_detached_component",
        reconciliation_confidence=0.1,
        support_distance_m=400.0,
    )
    assert weak < generic
    assert tiny < weak


def test_unsupported_mainstem_tendency_stays_more_expressive_than_side_component_when_reconciled():
    mainstem = compute_tendency_depth_fraction(
        36.0,
        "compound_lowflow",
        "unsupported",
        component_class="unsupported_mainstem",
        reconciliation_confidence=0.65,
        support_distance_m=120.0,
    )
    side = compute_tendency_depth_fraction(
        36.0,
        "compound_lowflow",
        "unsupported",
        component_class="unsupported_side_component",
        reconciliation_confidence=0.65,
        support_distance_m=120.0,
    )
    mainstem_conf = compute_tendency_confidence(
        36.0,
        "compound_lowflow",
        "unsupported",
        component_class="unsupported_mainstem",
        reconciliation_confidence=0.65,
        support_distance_m=120.0,
    )
    side_conf = compute_tendency_confidence(
        36.0,
        "compound_lowflow",
        "unsupported",
        component_class="unsupported_side_component",
        reconciliation_confidence=0.65,
        support_distance_m=120.0,
    )
    assert mainstem > side
    assert mainstem_conf > side_conf
