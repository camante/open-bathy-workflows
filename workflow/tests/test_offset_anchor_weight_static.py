from pathlib import Path


def test_observed_offset_stage_emits_anchor_weight_fields() -> None:
    text = Path('pipeline/river_workflow/river_workflow_stage_observed_offset.py').read_text(encoding='utf-8')
    assert 'offset_anchor_weight' in text
    assert 'offset_qc_class' in text
    assert 'offset_anchor_role' in text
    assert 'offset_low_depth_warning' in text
    assert 'shallow_observed_low_confidence' in text
    assert 'finite_wse_and_bed_positive_wse_minus_bed_and_authoritative_support_with_floor_level_offsets_downweighted' in text


def test_modeled_offset_uses_weighted_anchors() -> None:
    text = Path('pipeline/river_workflow/river_workflow_modeled_offset_logic.py').read_text(encoding='utf-8')
    assert 'effective_anchor_obs' in text
    assert 'anchor_weight * obs' in text
    assert 'weak_observed_offset_blended_anchor' in text
    assert 'strong_observed_anchor_count' in text
    assert 'weak_observed_anchor_count' in text
