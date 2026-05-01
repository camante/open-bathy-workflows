from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_dem_identity_logic_is_not_embedded_in_bathy_main() -> None:
    bathy_main = (ROOT / "bathy_main.py").read_text(encoding="utf-8")
    assert "from pipeline.final_dem_identity import" in bathy_main
    assert "def _verify_dem_enhanced_single_source_of_truth" not in bathy_main
    assert "def _record_dem_enhanced_touch" not in bathy_main
    assert "def _replace_with_symlink_or_copy" not in bathy_main


def test_final_dem_identity_module_owns_verify_only_helpers() -> None:
    module = (ROOT / "pipeline" / "final_dem_identity.py").read_text(encoding="utf-8")
    assert "def verify_dem_enhanced_single_source_of_truth" in module
    assert "def record_dem_enhanced_touch" in module
    assert "def replace_with_symlink_or_copy" in module
    assert "verification_mode" in module
    assert "raster_content_no_rewrite" in module


def test_phase3_documentation_exists() -> None:
    doc = ROOT / "docs" / "RIVER_WORKFLOW_PHASE3_BATHY_MAIN_BOUNDARY.md"
    text = doc.read_text(encoding="utf-8")
    assert "canonical_parent_dem -> aoi_export_dem -> final_user_dem" in text
    assert "pipeline/final_dem_identity.py" in text
