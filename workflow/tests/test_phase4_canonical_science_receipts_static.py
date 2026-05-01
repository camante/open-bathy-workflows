from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_manifest_carries_parent_science_summary() -> None:
    text = (ROOT / "pipeline" / "river_workflow" / "river_workflow_canonical_manifest.py").read_text(encoding="utf-8")
    assert "canonical_science_summary_from_stage_receipts" in text
    assert '"canonical_science_summary"' in text
    assert '"canonical_science_summary_policy"' in text
    assert '"aoi_exports_may_recompute_science": False' in text


def test_run_summary_reads_manifest_science_before_stage_receipt_fallback() -> None:
    text = (ROOT / "pipeline" / "run_summary.py").read_text(encoding="utf-8")
    assert 'manifest.get("canonical_science_summary")' in text
    assert "_canonical_stage_receipt_paths_from_manifest(receipts)" in text
    assert "_path_from_receipt_record" in text
    assert "Path(path_value).is_file()" in text


def test_phase4_documentation_exists() -> None:
    doc = ROOT / "docs" / "RIVER_WORKFLOW_PHASE4_CANONICAL_SCIENCE_RECEIPTS.md"
    text = doc.read_text(encoding="utf-8")
    assert "canonical_parent_dem -> aoi_export_dem -> final_user_dem" in text
    assert "AOI export runs must not recompute" in text
    assert "canonical_science_summary" in text
