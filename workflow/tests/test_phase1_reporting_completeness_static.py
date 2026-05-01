from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase1_run_summary_has_explicit_reporting_completeness_contract():
    src = (ROOT / "pipeline" / "run_summary.py").read_text(encoding="utf-8")
    required = [
        "river_phase1_reporting_complete_v1",
        "phase1_reporting_completeness",
        "required_at_summary_write",
        "expected_after_summary_write",
        "workflow_actual_trace.json",
        "canonical_manifest",
        "aoi_export_identity",
        "river_workflow_receipt",
        "final_output_receipt",
    ]
    missing = [item for item in required if item not in src]
    assert not missing, missing


def test_phase1_skipped_science_checks_have_reasons_not_empty_details():
    src = (ROOT / "pipeline" / "run_summary.py").read_text(encoding="utf-8")
    assert "stage_science_receipts_not_retained_or_not_available_in_this_aoi_export" in src
    assert "support_or_composition_counts_not_found_in_retained_receipts" in src
    assert 'checks[name]["reason"] = stage_missing_reason' in src
    assert 'checks["support_class_counts_reported"]["reason"] = support_missing_reason' in src


def test_phase1_text_summary_renders_completeness_section():
    src = (ROOT / "pipeline" / "run_summary.py").read_text(encoding="utf-8")
    assert "PHASE 1 REPORTING COMPLETENESS" in src
    assert "expected after summary write" in src
