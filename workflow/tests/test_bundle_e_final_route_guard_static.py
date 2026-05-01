from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_active_pipeline_writes_final_route_guard():
    src = read("active_pipeline.py")
    assert "write_final_route_guard_receipt" in src
    assert "stage_name=\"final_route_guard\"" in src
    assert "river_final_route_guard_receipt" in src
    assert "final_route_guard_passed" in src


def test_final_route_guard_enforces_parent_export_final_chain():
    src = read("river_workflow_final_route.py")
    assert "canonical_parent_dem -> aoi_export_dem -> final_user_dem" in src
    assert "assert_materialization_receipt_valid" in src
    assert "assert_aoi_export_identity_passed" in src
    assert "final_hash_matches_aoi_export_hash" in src
    assert "identity_zero_max_abs_diff" in src
    assert "touch_log_actions_are_materializer_only" in src
    assert "final_route_guard_failed" in src


def test_final_dem_materialization_has_named_aoi_export_writer():
    src = read("pipeline/final_dem_materialization.py")
    assert "def write_final_dem_from_aoi_export" in src
    assert "source_destination_hash_match" in src
    assert "if not src.is_file()" in src
    assert "terrain_interpolation" in src
    assert "blending" in src
    assert "authoritative_locking" in src
