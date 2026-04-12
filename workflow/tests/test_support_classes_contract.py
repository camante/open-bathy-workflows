from support_classes import (
    SupportClass,
    class_allows_river_guidance,
    class_allows_sdb_guidance,
    class_is_authoritative_locked,
    class_requires_low_confidence,
)
from river_support_roles import (
    canonical_river_support_class,
    is_authoritative_interior,
    is_bank_margin_only,
    is_unsupported_interior,
)


def test_support_class_helpers():
    assert class_is_authoritative_locked(int(SupportClass.AUTHORITATIVE_LOCKED))
    assert class_allows_sdb_guidance(int(SupportClass.GUIDANCE_CONDITIONED_SDB))
    assert class_allows_river_guidance(int(SupportClass.GUIDANCE_CONDITIONED_RIVER))
    assert class_requires_low_confidence(int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL))


def test_canonical_river_support_helpers():
    assert canonical_river_support_class(station_authoritative_bed_support_present=True) == "authoritative_interior"
    assert is_authoritative_interior(station_authoritative_bed_support_present=True)
    assert canonical_river_support_class(station_support_regime="bank_only_low_confidence") == "bank_margin_only"
    assert is_bank_margin_only(station_support_regime="bank_only_authoritative")
    assert is_unsupported_interior(component_support_class="unsupported_mainstem")
