from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="ignore")


def test_removed_root_level_stage_contract_shims() -> None:
    for rel in [
        "river_workflow_context.py",
        "river_workflow_contract.py",
        "river_workflow_paths.py",
        "river_workflow_receipts.py",
        "river_workflow_validation.py",
    ]:
        assert not (ROOT / rel).exists(), rel


def test_active_pipeline_uses_packaged_stage_artifact_contract() -> None:
    text = _text("active_pipeline.py")
    assert "pipeline.river_workflow.river_workflow_stage_artifact_contract" in text
    assert "from river_workflow_receipts" not in text
    assert "from river_workflow_validation" not in text
    assert "from river_workflow_contract" not in text


def test_active_construction_surface_has_no_broad_exception_masking() -> None:
    active_files = [
        ROOT / "active_pipeline.py",
        ROOT / "river_runner.py",
        ROOT / "river_workflow.py",
        ROOT / "river_workflow_entry.py",
        ROOT / "river_workflow_final_route.py",
        *sorted((ROOT / "pipeline/river_workflow").glob("*.py")),
    ]
    for path in active_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "except Exception" not in text, str(path.relative_to(ROOT))
        assert "except:" not in text, str(path.relative_to(ROOT))


def test_repo_contract_check_knows_current_namespace_layout() -> None:
    text = _text("repo_contract_checks.py")
    assert "tools/debug/workflow_actual_trace.py" not in text
    assert "validation/seam_metrics.py" not in text
    assert "river_workflow_context.py': 'pipeline/river_workflow/river_workflow_context.py" in text
