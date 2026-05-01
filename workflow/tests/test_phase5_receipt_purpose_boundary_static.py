from pathlib import Path


def test_phase5_paths_include_receipt_purpose_manifest():
    src = Path("pipeline/river_workflow/river_workflow_paths.py").read_text(encoding="utf-8")
    assert "receipt_purpose_manifest: Path" in src
    assert 'receipt_purpose_manifest=manifests_dir / "receipt_purpose_manifest.json"' in src


def test_phase5_receipt_writer_is_verify_only_index():
    src = Path("pipeline/river_workflow/river_workflow_receipts.py").read_text(encoding="utf-8")
    start = src.index("def write_receipt_purpose_manifest(")
    end = src.index("__all__ = [")
    block = src[start:end]
    required = [
        "primary_receipts",
        "stage_receipts",
        "diagnostic_receipts",
        "does_not_drive_routing",
        "does_not_repair_outputs",
        "one_primary_receipt_per_purpose",
        "first_wrong_artifact_rule",
        "canonical_parent_manifest",
        "aoi_export_identity",
        "final_output_receipt",
        "comparison_report",
        "human_run_summary",
    ]
    missing = [item for item in required if item not in block]
    assert not missing, missing


def test_phase5_pipeline_writes_receipt_purpose_manifest_without_route_change():
    src = Path("pipeline/river_workflow/river_workflow_pipeline.py").read_text(encoding="utf-8")
    assert "write_receipt_purpose_manifest" in src
    assert "ctx.paths.receipt_purpose_manifest" in src
    export_start = src.index("def run_aoi_export_from_canonical_parent(")
    export_end = src.index("def run_canonical_river_build(")
    export_block = src[export_start:export_end]
    assert "write_receipt_purpose_manifest" in export_block
    assert "run_" not in export_block.split("write_receipt_purpose_manifest", 1)[1].split("logger.info", 1)[0]


def test_phase5_boundary_documented():
    doc = Path("docs/ACTIVE_RIVER_MODULES.md").read_text(encoding="utf-8")
    manifest = Path("active_river_modules.json").read_text(encoding="utf-8")
    assert "Phase 5 receipt-purpose boundary" in doc
    assert "active_receipt_purpose_manifest" in manifest
    assert "one_primary_receipt_per_purpose" in manifest
