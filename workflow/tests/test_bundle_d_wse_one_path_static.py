from pathlib import Path


def test_wse_stage_exposes_four_explicit_steps():
    text = Path('pipeline/river_workflow/river_workflow_stage_wse_proxy.py').read_text(encoding='utf-8')
    for name in (
        'build_wse_support_artifact',
        'build_wse_trend_artifact',
        'build_wse_pre_smooth_artifact',
        'build_wse_proxy_final_artifact',
    ):
        assert f'def {name}' in text
    assert 'wse_one_path_stage_contract.json' in text
    assert 'one_path_no_fallbacks_wse_reference_surface_only' in text


def test_active_wse_stage_has_no_broad_exception_handler():
    text = Path('pipeline/river_workflow/river_workflow_stage_wse_proxy.py').read_text(encoding='utf-8')
    assert 'except Exception' not in text
    assert 'except:' not in text


def test_active_contract_doc_uses_river_names():
    text = Path('docs/ACTIVE_RIVER_WORKFLOW_CONTRACT.md').read_text(encoding='utf-8')
    assert 'run_active_river_workflow(ctx)' in text
    assert 'finalize_active_river_workflow(ctx, workflow_result)' in text
    assert 'run_active_linear_workflow(ctx)' not in text
