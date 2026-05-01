from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding='utf-8')


def test_active_runner_uses_river_workflow_package_and_output_root():
    text = read('river_runner.py')
    assert 'pipeline.river_workflow.river_workflow_context' in text
    assert 'Path(getattr(cfg, "out_dir")) / "river_workflow"' in text
    assert 'pipeline.river_linear' not in text
    assert '/ "river_linear"' not in text


def test_active_final_route_imports_new_contract_package():
    for path in ('active_pipeline.py', 'river_workflow_final_route.py', 'pipeline/final_dem_materialization.py'):
        text = read(path)
        assert 'pipeline.river_workflow' in text
        assert 'pipeline.river_linear' not in text


def test_active_implementation_package_exists_and_legacy_package_is_quarantined():
    assert (ROOT / 'pipeline/river_workflow/river_workflow_pipeline.py').is_file()
    assert not (ROOT / 'pipeline/river_linear').exists()
    assert (ROOT / 'legacy/river/pipeline_river_linear_reference/README_DEPRECATED.txt').is_file()


def test_shared_solve_wrapper_points_to_river_workflow_not_river_linear():
    text = read('pipeline/river_shared_solve/river_pipeline.py')
    assert 'pipeline.river_workflow.river_workflow_pipeline' in text
    assert 'pipeline.river_linear' not in text
