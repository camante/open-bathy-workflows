from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "pipeline" / "river_workflow" / "river_workflow_stage_backbone.py"
CACHE = ROOT / "pipeline" / "river_workflow" / "river_workflow_cache.py"


def test_phase6_backbone_limiter_is_support_aware():
    text = BACKBONE.read_text(encoding="utf-8")
    assert "def _support_allows_strong_rise_limit" in text
    assert "support_class: np.ndarray | None = None" in text
    assert "limit_guidance_points_to_allowed_downstream_rise" in text
    assert "observed_anchor_max_adjustment_m" in text
    assert "support_class=support_v" in text


def test_phase6_backbone_does_not_reintroduce_whole_profile_projection():
    text = BACKBONE.read_text(encoding="utf-8")
    fn = text.split("def _bounded_residual_preserving_smooth", 1)[1].split("def _write_bed_backbone_audit", 1)[0]
    assert "_weighted_isotonic_nonincreasing" not in fn
    assert "support-aware local rise limiter" in fn


def test_phase6_cache_key_forces_rebuild_after_backbone_algorithm_change():
    text = CACHE.read_text(encoding="utf-8")
    assert "seamless_dem_parent_export_v18_backbone_rise_cleanup" in text
