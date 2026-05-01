from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_river_route_metadata_is_recorded():
    text = (ROOT / "bathy_main.py").read_text(encoding="utf-8")
    assert '"active_route": "canonical_river_parent_export"' in text
    assert '"active_methods": ["river"]' in text
    assert '"final_dem_owner": "river"' in text
    assert 'report.setdefault("route", {}).update' in text


def test_route_logs_are_explicit_and_user_facing():
    text = (ROOT / "bathy_main.py").read_text(encoding="utf-8")
    for phrase in [
        "[ROUTE] Requested methods",
        "[ROUTE] Active route",
        "[ROUTE] Active construction methods",
        "[ROUTE] Inactive methods for this run",
        "[ROUTE] Final DEM owner",
        "[ROUTE] Reason",
    ]:
        assert phrase in text


def test_human_summary_uses_route_not_generic_methods_only():
    text = (ROOT / "reporting" / "run_summary.py").read_text(encoding="utf-8")
    assert 'route = get(stats, "route"' in text
    assert "Active route:" in text
    assert "Final DEM owner:" in text
    assert "Active construction methods:" in text
    assert "Inactive methods this run:" in text
    assert "canonical_river_parent_export" in text


def test_finalize_passes_route_to_human_summary():
    text = (ROOT / "finalize_run_helpers.py").read_text(encoding="utf-8")
    assert '"route": report.get("route", {})' in text
    assert '"guidance_domains": report.get("guidance_domains", {})' in text
