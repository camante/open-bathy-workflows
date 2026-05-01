from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PIPELINE = ROOT / "active_pipeline.py"


def test_consolidated_final_receipt_is_written_to_final_folder() -> None:
    text = ACTIVE_PIPELINE.read_text(encoding="utf-8")
    assert "def _write_consolidated_river_workflow_receipt" in text
    assert 'final_dir / "river_workflow_receipt.json"' in text
    assert 'final_dir / "river_workflow_summary.txt"' in text
    assert 'stage_name="write_river_workflow_final_receipt"' in text
    assert 'stage_name="write_river_workflow_summary"' in text


def test_consolidated_receipt_contains_parent_export_identity_fields() -> None:
    text = ACTIVE_PIPELINE.read_text(encoding="utf-8")
    required = [
        '"canonical_parent_dem"',
        '"aoi_export_dem"',
        '"final_dem"',
        '"export_window"',
        '"export_vs_parent_exact"',
        '"max_abs_diff_m"',
        '"mismatch_pixels"',
        '"combined_vs_export_exact"',
        '"single_writer_pass"',
        '"post_subset_modifications"',
        '"support_class_counts"',
        '"stage_statuses"',
    ]
    for needle in required:
        assert needle in text


def test_human_summary_has_one_file_pass_fail_lines() -> None:
    text = ACTIVE_PIPELINE.read_text(encoding="utf-8")
    for line in [
        "CANONICAL BUILD:",
        "AOI EXPORT:",
        "FINAL DEM SOURCE:",
        "EXPORT VS PARENT:",
        "SINGLE WRITER:",
        "COMBINED VS EXPORT:",
        "POST-SUBSET MODIFICATIONS:",
    ]:
        assert line in text
