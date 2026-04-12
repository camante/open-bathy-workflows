from pathlib import Path

from river_domain_policy import evaluate_river_domain_summary, load_river_domain_summary
from river_domain_mask import _resolve_channel_source_policy


def test_evaluate_river_domain_summary_ok():
    summary = {
        "channel_source_requested": "auto",
        "channel_source_effective": "corridor",
        "effective_water_source": "with_nhd_corridor_fallback",
        "water_mask_harmonization_applied": True,
        "corridor_pixels": 1000,
        "channel_pixels": 200,
        "effective_water_corridor_overlap_frac": 0.25,
        "channel_corridor_overlap_frac": 0.10,
        "nhdarea_corridor_overlap_frac": 0.0,
        "nhdarea_effective": False,
    }
    out = evaluate_river_domain_summary(summary)
    assert out["ok"] is True
    assert out["warnings"]


def test_evaluate_river_domain_summary_fails_on_low_overlap():
    summary = {
        "channel_source_requested": "nhdarea",
        "channel_source_effective": "corridor",
        "effective_water_source": "with_nhd",
        "water_mask_harmonization_applied": False,
        "corridor_pixels": 1000,
        "channel_pixels": 0,
        "effective_water_corridor_overlap_frac": 0.001,
        "channel_corridor_overlap_frac": 0.0,
    }
    out = evaluate_river_domain_summary(summary, min_effective_water_corridor_overlap_frac=0.02, min_channel_corridor_overlap_frac=0.01, min_channel_pixels=10)
    assert out["ok"] is False
    checks = {f["check"] for f in out["failures"]}
    assert "effective_water_corridor_overlap" in checks
    assert "channel_pixels" in checks
    assert "channel_source_policy" in checks


def test_load_river_domain_summary(tmp_path: Path):
    p = tmp_path / "summary.json"
    p.write_text('{"corridor_pixels": 5}', encoding='utf-8')
    out = load_river_domain_summary(p)
    assert out["corridor_pixels"] == 5


def test_resolve_channel_source_policy_auto_prefers_corridor_when_nhdarea_empty():
    effective, reason = _resolve_channel_source_policy('auto', has_usable_nhdarea=False)
    assert effective == 'corridor'
    assert reason == 'nhdarea_unusable_or_empty'


def test_resolve_channel_source_policy_explicit_nhdarea_stays_strict():
    effective, reason = _resolve_channel_source_policy('nhdarea', has_usable_nhdarea=False)
    assert effective == 'nhdarea'
    assert reason is None


def test_resolve_channel_source_policy_auto_prefers_corridor_when_nhdarea_overlap_is_too_low():
    effective, reason = _resolve_channel_source_policy('auto', has_usable_nhdarea=True, nhdarea_overlap_frac=0.002)
    assert effective == 'corridor'
    assert reason == 'nhdarea_low_corridor_overlap'
