from river_target_contract import (
    ACTIVE_INTERIOR_TARGET_AUTHORITATIVE,
    ACTIVE_INTERIOR_TARGET_BACKBONE,
    normalize_active_target_source,
    is_active_target_authoritative,
    is_active_target_backbone_led,
)


def test_target_contract_normalizes_only_canonical_and_known_aliases():
    assert normalize_active_target_source(ACTIVE_INTERIOR_TARGET_AUTHORITATIVE) == ACTIVE_INTERIOR_TARGET_AUTHORITATIVE
    assert normalize_active_target_source('generalized_longitudinal_section') == ACTIVE_INTERIOR_TARGET_BACKBONE
    assert normalize_active_target_source('unknown_target_path') == 'missing'
    assert is_active_target_authoritative('authoritative_interior')
    assert is_active_target_backbone_led('longitudinal_backbone_template')
    assert not is_active_target_backbone_led('bank_fit_fallback')
