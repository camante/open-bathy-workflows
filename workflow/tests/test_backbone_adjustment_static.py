from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "pipeline" / "river_workflow" / "river_workflow_stage_backbone.py"


def test_backbone_smoother_does_not_reapply_whole_profile_isotonic_projection():
    text = BACKBONE.read_text()
    fn = text.split("def _bounded_residual_preserving_smooth", 1)[1].split("def _write_bed_backbone_audit", 1)[0]
    assert "_weighted_isotonic_nonincreasing" not in fn
    assert "bounded local smoothing" in fn


def test_backbone_artifact_carries_adjustment_classes_and_threshold_counts():
    text = BACKBONE.read_text()
    assert "bed_backbone_raw_z_m" in text
    assert "bed_backbone_adjustment_class" in text
    assert "bed_backbone_adjustment_reason" in text
    assert "adjustment_gt_1m_count" in text
    assert "adjustment_gt_5m_count" in text
