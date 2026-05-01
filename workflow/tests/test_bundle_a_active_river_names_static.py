from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_active_entry_uses_river_workflow_names() -> None:
    text = (ROOT / "river_workflow_entry.py").read_text(encoding="utf-8")
    assert "run_active_river_workflow" in text
    assert "finalize_active_river_workflow" in text
    assert "run_active_linear_workflow(ctx)" not in text
    assert "finalize_active_linear_workflow(ctx" not in text


def test_active_pipeline_uses_river_entrypoint_names_only() -> None:
    text = (ROOT / "active_pipeline.py").read_text(encoding="utf-8")
    assert "def run_active_river_workflow" in text
    assert "def finalize_active_river_workflow" in text
    assert "run_active_linear_workflow = run_active_river_workflow" not in text
    assert "finalize_active_linear_workflow = finalize_active_river_workflow" not in text


def test_public_river_workflow_facade_exists() -> None:
    text = (ROOT / "river_workflow.py").read_text(encoding="utf-8")
    assert "def run_river_workflow" in text
    assert "run_river_workflow_direct" in text
    assert "canonical_parent_aoi_export" in text
