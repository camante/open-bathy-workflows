"""Phase 0 guards for the stable active river contract.

These tests are intentionally static and lightweight. They freeze the current
canonical-parent/AOI-export architecture before cleanup removes or quarantines
legacy river code. They must not import geospatial packages or execute the
workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _manifest() -> dict:
    return json.loads((ROOT / "active_river_modules.json").read_text(encoding="utf-8"))


def test_phase0_active_route_manifest_freezes_parent_export_invariant() -> None:
    manifest = _manifest()
    assert manifest["active_route"] == "canonical_river_parent_export"
    invariant = manifest["active_invariant"]
    assert "shared solve domain" in invariant
    assert "canonical parent river solution" in invariant
    assert "exact AOI export" in invariant
    assert "combined/DEM_enhanced.tif written once" in invariant
    assert manifest["cleanup_policy"]["no_behavior_change"] is True
    assert manifest["cleanup_policy"]["legacy_removal_allowed_in_this_phase"] is False


def test_phase0_active_pipeline_requires_parent_export_and_single_writer() -> None:
    text = _read("active_pipeline.py")
    assert "canonical_parent_plus_aoi_export" in text
    assert "active_river_result_missing_final_dem_result" in text
    assert "active_river_result_missing_parent_export_paths" in text
    assert "Materialize combined/DEM_enhanced.tif from the named AOI export artifact" in text
    assert "assert_single_final_dem_writer" in text
    assert "export_vs_parent=%s" in text
    assert "PASS" in text


def test_phase0_linear_contract_declares_one_stage_order_and_final_roles() -> None:
    text = _read("pipeline/river_workflow/river_workflow_contract.py")
    expected_order = [
        "run_contract",
        "solve_domain",
        "grids",
        "authoritative_inputs",
        "centerline_points",
        "centerline_wse_proxy",
        "centerline_authoritative_bed",
        "centerline_observed_offset",
        "centerline_modeled_offset",
        "centerline_bed_backbone",
        "river_corridor_solve",
        "river_primary_surface_solve",
        "river_primary_surface_solve_locked",
        "river_export_handoff",
        "final_dem",
    ]
    for stage in expected_order:
        assert f'"{stage}"' in text
    assert "CANONICAL_PARENT_DEM_ROLE" in text
    assert "AOI_EXPORT_DEM_ROLE" in text
    assert "FINAL_USER_DEM_ROLE" in text
    assert "FINAL_USER_DEM_RELATIVE_PATH" in text
    assert "combined/DEM_enhanced.tif" in text


def test_phase0_aoi_export_receipt_is_verify_only_and_exact_parent_subset() -> None:
    text = _read("pipeline/river_workflow/river_workflow_stage_final_dem.py")
    assert "aoi_export_identity.json" in text
    assert "canonical_parent_identity.json" in text
    assert "assert_parent_window_matches_export" in text
    assert "identity_contract" in text
    assert "post_subset_modification_allowed" in text
    assert "construction_attempted" in text
    assert "post_subset_modifications" in text


def test_phase0_compare_tool_is_verify_only() -> None:
    text = _read("tools/compare_aoi_exports.py")
    assert "post-run validator" in text
    assert "does not rebuild, rediscover, or" in text
    assert "verify-only comparison; no output repair or rerouting was attempted" in text
    assert "first_wrong_artifact" in text
