from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "pipeline" / "river_workflow" / "river_workflow_stage_backbone.py"
CONTRACT = ROOT / "canonical_river_solve_contract.py"


def test_backbone_limiter_reports_observed_anchor_conflicts_without_preserving_jumps():
    text = BACKBONE.read_text(encoding="utf-8")
    assert "observed_offset_supported_backbone_points_may_be_adjusted_but_conflicts_are_reported" in text
    assert "limit_all_final_backbone_points_to_allowed_downstream_rise_before_authoritative_lock" in text
    assert "observed_offset_anchor_adjusted_by_backbone_rise_limiter" in text
    assert "target = max(allowed, vals[i] - float(observed_anchor_max_adjustment_m))" not in text


def test_backbone_science_change_rotates_canonical_cache_key():
    text = CONTRACT.read_text(encoding="utf-8")
    assert "seamless_dem_parent_export_v20_backbone_anchor_conflict_reporting" in text
