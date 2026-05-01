from pathlib import Path


def test_active_centerline_builder_is_not_unavailable_stub():
    text = Path('river_structured_scaffold.py').read_text(encoding='utf-8')
    assert 'build_centerline_points = _unavailable' not in text
    assert 'legacy_structured_scaffold_function_unavailable:build_centerline_points' not in text
    assert 'def build_centerline_points(' in text
    assert 'river_structured_centerline_empty_flows' in text


def test_legacy_scaffold_helpers_are_explicitly_not_active():
    text = Path('river_structured_scaffold.py').read_text(encoding='utf-8')
    assert 'river_structured_scaffold_function_not_active_in_builtin_linear_workflow' in text
