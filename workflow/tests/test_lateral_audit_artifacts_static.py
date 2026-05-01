from pathlib import Path


def test_lateral_audit_products_are_retained():
    import repo_runtime_modes as modes

    retained = set(modes.NORMAL_RUN_REPORTS_FILES)
    assert "reports/LATERAL_CURRENT_PATH_AUDIT.txt" in retained
    assert "reports/LATERAL_CROSS_SECTION_AUDIT.txt" in retained
    assert "reports/lateral_artifacts/lateral_artifact_manifest.json" in retained
    assert "reports/lateral_artifacts/river_lateral_distance_to_centerline.tif" in retained
    assert "reports/lateral_artifacts/river_lateral_distance_to_bank.tif" in retained
    assert "reports/lateral_artifacts/river_lateral_normalized_position.tif" in retained
    assert "reports/lateral_artifacts/river_lateral_taper_weight.tif" in retained
    assert "reports/lateral_artifacts/river_channel_width_points.gpkg" in retained
    assert "reports/lateral_artifacts/lateral_audit_cross_sections.gpkg" in retained


def test_surface_stage_contains_lateral_audit_functions():
    path = Path("pipeline/river_workflow/river_workflow_stage_surface.py")
    text = path.read_text(encoding="utf-8")
    assert "def _write_lateral_audit_products" in text
    assert "river_lateral_distance_to_centerline.tif" in text
    assert "river_lateral_distance_to_bank.tif" in text
    assert "river_lateral_normalized_position.tif" in text
    assert "river_channel_width_points.gpkg" in text
    assert "lateral_audit_cross_sections.gpkg" in text
    assert "river_lateral_taper_weight.tif" in text
    assert "behavior_changed': True" in text
