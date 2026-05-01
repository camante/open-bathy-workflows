from pathlib import Path


def test_wse_steps_module_owns_four_explicit_artifact_builders() -> None:
    text = Path('pipeline/river_workflow/river_workflow_wse_steps.py').read_text(encoding='utf-8')
    for name in [
        'build_wse_support_artifact',
        'build_wse_trend_artifact',
        'build_wse_pre_smooth_artifact',
        'build_wse_proxy_final_artifact',
        'write_wse_stage_contract',
    ]:
        assert f'def {name}' in text


def test_wse_proxy_stage_imports_steps_instead_of_owning_contract_builders() -> None:
    text = Path('pipeline/river_workflow/river_workflow_stage_wse_proxy.py').read_text(encoding='utf-8')
    assert 'from pipeline.river_workflow.river_workflow_wse_steps import' in text
    assert 'def build_wse_support_artifact' not in text
    assert 'def build_wse_trend_artifact' not in text
    assert 'def build_wse_pre_smooth_artifact' not in text
    assert 'def build_wse_proxy_final_artifact' not in text
    assert 'def _write_wse_stage_contract' not in text
