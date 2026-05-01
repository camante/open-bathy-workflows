from pathlib import Path


def test_wse_stage_is_one_path_profile_contract():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pipeline" / "river_workflow" / "river_workflow_stage_wse_proxy.py").read_text(encoding="utf-8")
    assert "monotone_longitudinal_profile_v4_clean_one_path" in text
    assert "_WSE_PROFILE_GROUP_FIELDS = ('component_id', 'levelpath_id')" in text
    assert "_WSE_LEGACY_REPAIR_PATH_ENABLED" not in text
    assert "blocked_before_legacy_wse_repair_path" not in text
    assert "_write_wse_phase0_contract_receipt" not in text
    assert "def _fit_monotone_longitudinal_wse_profile" in text
    assert "wse_proxy_z_m" in text
    assert "station_downstream_m" in text
