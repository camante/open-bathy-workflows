from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_FILES = [
    "bathy_main.py",
    "repo_runtime_modes.py",
    "river_runner.py",
    "active_pipeline.py",
    "pipeline/river_workflow/river_workflow_context.py",
    "guidance_assembly_stage.py",
    "reporting/run_summary.py",
]
ACTIVE_IMPORT_FILES = [
    "bathy_main.py",
    "river_runner.py",
    "active_pipeline.py",
    "pipeline/river_workflow/river_workflow_context.py",
    "guidance_assembly_stage.py",
    "reporting/run_summary.py",
]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_active_runtime_has_no_method_style_aliases() -> None:
    runtime = read("repo_runtime_modes.py")
    assert "linear_v1" not in runtime
    assert "simple_v2" not in runtime
    assert "RIVER_WORKFLOW_ALIASES: FrozenSet[str] = frozenset()" in runtime


def test_active_config_no_longer_exposes_river_method() -> None:
    bathy = read("bathy_main.py")
    config = read("config/default.yaml")
    assert 'add_argument("--river-method"' not in bathy
    assert "river_method:" not in config
    assert "river_method: str" not in bathy


def test_active_river_context_no_deprecated_river_method_alias() -> None:
    ctx = read("pipeline/river_workflow/river_workflow_context.py")
    assert "InitVar" not in ctx
    assert "river_method" not in ctx


def test_active_code_does_not_import_quarantined_river_linear_package() -> None:
    for rel in ACTIVE_IMPORT_FILES:
        text = read(rel)
        assert "pipeline.river_linear" not in text
        assert "legacy/river" not in text
