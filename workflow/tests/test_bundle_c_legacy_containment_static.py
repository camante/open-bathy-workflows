from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_active_runner_no_longer_exports_linear_v1_names() -> None:
    text = (ROOT / "river_runner.py").read_text(encoding="utf-8")
    active_tail = text.split("__all__", 1)[-1]
    assert "run_linear_v1_direct" not in active_tail
    assert "register_linear_v1_runner_result" not in active_tail


def test_linear_v1_names_are_confined_to_archive_wrapper() -> None:
    active_files = [
        ROOT / "active_pipeline.py",
        ROOT / "river_runner.py",
        ROOT / "river_workflow_entry.py",
    ]
    combined = "\n".join(p.read_text(encoding="utf-8") for p in active_files)
    assert "run_active_linear_workflow =" not in combined
    assert "finalize_active_linear_workflow =" not in combined
    assert "run_river_linear_details_fn" not in combined


def test_active_callbacks_use_river_source_bundle_names() -> None:
    entry = (ROOT / "river_workflow_entry.py").read_text(encoding="utf-8")
    pipeline = (ROOT / "active_pipeline.py").read_text(encoding="utf-8")
    assert "resolve_river_shared_source_artifacts_fn" in entry
    assert "prepare_river_canonical_source_bundle_fn" in entry
    assert "resolve_river_shared_source_artifacts_fn" in pipeline
    assert "prepare_river_canonical_source_bundle_fn" in pipeline
