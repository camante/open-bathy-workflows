from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase7_human_summary_has_canonical_debugging_map():
    src = (ROOT / "reporting" / "run_summary.py").read_text(encoding="utf-8")
    required = [
        "Canonical river route debugging map",
        "Construction chain: canonical_parent_dem → aoi_export_dem → final_user_dem",
        "Canonical parent manifest:",
        "AOI export identity:",
        "River workflow receipt:",
        "Final output receipt:",
        "Final user DEM:",
        "Primary outputs written",
    ]
    missing = [item for item in required if item not in src]
    assert not missing, missing


def test_phase7_canonical_route_does_not_render_fusion_as_active_path():
    src = (ROOT / "reporting" / "run_summary.py").read_text(encoding="utf-8")
    assert "if fusion and not is_shared_solve_river" in src
    assert "inactive and should not read like a participating construction path" in src


def test_phase7_canonical_route_output_order_omits_inactive_method_products():
    src = (ROOT / "reporting" / "run_summary.py").read_text(encoding="utf-8")
    start = src.index("output_key_order = [")
    end = src.index("for k in output_key_order:")
    block = src[start:end]
    canonical_block = block.split("] if is_shared_solve_river else [", 1)[0]
    assert "combined_warped" in canonical_block
    assert "aoi_export_identity_report" in canonical_block
    assert "canonical_river_solution_manifest" in canonical_block
    assert "sdb_warped" not in canonical_block
    assert "river_bottom_warped" not in canonical_block


def test_phase7_canonical_route_detection_is_not_generic_river_workflow_label():
    src = (ROOT / "reporting" / "run_summary.py").read_text(encoding="utf-8")
    route_block = src.split("active_route = str", 1)[1].split("lines = []", 1)[0]
    assert 'active_route == "canonical_river_parent_export"' in route_block
    assert '"canonical_parent_plus_aoi_export" in workflow_label' in route_block
    assert '"river_workflow" in workflow_label' not in route_block
    assert '"linear_v1" in workflow_label' not in route_block
