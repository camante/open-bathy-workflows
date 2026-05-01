from pathlib import Path


def test_phase3_parent_export_functions_are_explicit():
    src = Path("pipeline/river_workflow/river_workflow_pipeline.py").read_text(encoding="utf-8")
    required = [
        "def resolve_canonical_parent_handoff(",
        "def run_aoi_export_from_canonical_parent(",
        "def run_canonical_river_build(",
        "def run_river_parent_export_workflow(",
        "def run_river_workflow_pipeline(",
    ]
    missing = [item for item in required if item not in src]
    assert not missing, missing


def test_aoi_export_only_path_keeps_construction_guard():
    src = Path("pipeline/river_workflow/river_workflow_pipeline.py").read_text(encoding="utf-8")
    assert 'ensure_stage_allowed(AOI_EXPORT_ONLY_ROLE, "read_canonical_manifest")' in src
    start = src.index("def run_aoi_export_from_canonical_parent(")
    end = src.index("def run_canonical_river_build(")
    export_only = src[start:end]
    assert "AOI export-only route complete; no canonical construction stages were run" in export_only
    assert "_assert_export_only_result_has_no_construction_stage_receipts(result)" in export_only


def test_backward_compatible_entrypoint_delegates_to_parent_export_workflow():
    src = Path("pipeline/river_workflow/river_workflow_pipeline.py").read_text(encoding="utf-8")
    start = src.index("def run_river_workflow_pipeline(")
    entry = src[start:]
    assert "return run_river_parent_export_workflow(ctx)" in entry


def test_phase3_boundary_documented():
    doc = Path("docs/ACTIVE_RIVER_MODULES.md").read_text(encoding="utf-8")
    assert "Phase 3 parent/export function boundary" in doc
    manifest = Path("active_river_modules.json").read_text(encoding="utf-8")
    assert "active_parent_export_functions" in manifest
