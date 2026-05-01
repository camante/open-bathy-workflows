from pathlib import Path


def test_identity_guard_allows_versioned_cache_rotation_only_when_materialized_inputs_match() -> None:
    src = Path('canonical_river_identity_guard.py').read_text(encoding='utf-8')
    assert 'cache_key_rotated_for_workflow_contract' in src
    assert 'allowed_same_canonical_identity_same_materialized_inputs_new_workflow_contract' in src
    assert 'same canonical identity changed cache key and materialized canonical inputs' in src
    assert 'previous_canonical_solve_cache_key' in src


def test_identity_guard_still_fails_when_materialized_inputs_change() -> None:
    src = Path('canonical_river_identity_guard.py').read_text(encoding='utf-8')
    assert 'mismatched_materialized_artifacts' in src
    assert 'same canonical identity/cache key produced different canonical materialized inputs' in src
