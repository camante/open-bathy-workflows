from pathlib import Path


def test_canonical_parent_content_key_fields_are_written():
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "pipeline" / "river_workflow" / "river_workflow_canonical_manifest.py").read_text()
    pipeline = (root / "pipeline" / "river_workflow" / "river_workflow_pipeline.py").read_text()
    active = (root / "active_pipeline.py").read_text()
    compare = (root / "tools" / "compare_aoi_exports.py").read_text()

    assert "canonical_parent_content_key" in manifest
    assert "canonical_comparison_key" in manifest
    assert "canonical_parent_content_key" in pipeline
    assert "canonical_comparison_key" in pipeline
    assert "canonical_parent_content_key" in active
    assert "canonical_comparison_key" in active
    assert "canonical_comparison_key" in compare
    assert "parent_hash" in compare


def test_compare_script_requires_final_canonical_parent_products():
    root = Path(__file__).resolve().parents[1]
    script = (root / "compare.sh").read_text()
    assert "final/canonical_parent_dem.tif" in script
    assert "final/canonical_parent_dem_hillshade.tif" in script
