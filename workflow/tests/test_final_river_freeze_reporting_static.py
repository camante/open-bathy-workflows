from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_SUMMARY = ROOT / "pipeline" / "run_summary.py"
COMPARE = ROOT / "tools" / "compare_aoi_exports.py"
DOC = ROOT / "docs" / "RIVER_WORKFLOW_FINAL_FREEZE_FOR_SDB_TEMPLATE.md"


def test_backbone_report_uses_large_rise_as_pass_fail_metric():
    text = RUN_SUMMARY.read_text(encoding="utf-8")
    assert "def _backbone_rise_validation" in text
    assert "large_downstream_rise_count_gt_allowed == 0" in text
    assert "positive_downstream_step_count" in text
    assert "largest_step_m may include downstream-deepening drops" in text
    assert "pass means no downstream rise exceeds the allowed local step" in text


def test_compare_output_explains_allowed_small_backbone_steps():
    text = COMPARE.read_text(encoding="utf-8")
    assert "positive downstream step count is diagnostic" in text
    assert "large downstream rises are the failure criterion" in text


def test_final_freeze_doc_declares_sdb_template_contract():
    text = DOC.read_text(encoding="utf-8")
    assert "canonical parent river solution" in text
    assert "exact AOI export" in text
    assert "single-writer final contract" in text
    assert "canonical/source-domain SDB guidance product" in text
