from support_classes import (
    SupportClass,
    class_allows_river_guidance,
    class_allows_sdb_guidance,
    class_is_authoritative_locked,
    class_requires_low_confidence,
)


def test_support_class_helpers():
    assert class_is_authoritative_locked(int(SupportClass.AUTHORITATIVE_LOCKED))
    assert class_allows_sdb_guidance(int(SupportClass.GUIDANCE_CONDITIONED_SDB))
    assert class_allows_river_guidance(int(SupportClass.GUIDANCE_CONDITIONED_RIVER))
    assert class_requires_low_confidence(int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL))
