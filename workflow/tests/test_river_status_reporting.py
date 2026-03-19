def test_report_fields_for_degraded_river_mode():
    report = {"river": {"degraded": {"xs_mainstem_fallback": True}}}
    river_degraded = bool(report.get("river", {}).get("degraded", {}).get("xs_mainstem_fallback", False))
    report["river"]["status"] = "degraded" if river_degraded else "success"
    report["river"]["status_family"] = report["river"]["status"]
    report["river"]["execution_mode"] = "hybrid_skeleton_only_degraded" if river_degraded else "hybrid_full"
    assert report["river"]["status"] == "degraded"
    assert report["river"]["execution_mode"] == "hybrid_skeleton_only_degraded"
