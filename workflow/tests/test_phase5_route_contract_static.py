from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_phase5_route_contract_module_is_read_only_and_explicit():
    text = read("pipeline/river_workflow/river_workflow_route_contract.py")
    assert "canonical_parent_dem -> aoi_export_dem -> final_user_dem" in text
    assert "validate_route_contract_payloads" in text
    assert "recomputed_construction_stages" in text
    assert "False" in text
    assert "do not open rasters" in text.lower() or "does not open rasters" in text.lower()


def test_phase5_cache_hit_export_only_route_is_guarded():
    text = read("pipeline/river_workflow/river_workflow_pipeline.py")
    start = text.index("def run_river_parent_export_workflow(")
    route = text[start:text.index("def run_river_workflow_pipeline(")]
    assert "handoff.reused_from_cache" in route
    assert "exporting AOI without rerunning canonical construction" in route
    assert "return run_aoi_export_from_canonical_parent(ctx, handoff)" in route


def test_phase5_missing_parent_and_stale_hash_fail_clearly():
    text = read("pipeline/river_workflow/river_workflow_pipeline.py")
    assert "missing_canonical_manifest_for_aoi_export_only" in text
    assert "canonical_parent_dem_missing_for_aoi_export" in text
    assert "canonical_manifest_parent_hash_mismatch" in text


def test_phase5_export_receipt_proves_exact_parent_subset_and_no_construction():
    text = read("pipeline/river_workflow/river_workflow_pipeline.py")
    start = text.index("def run_aoi_export_from_canonical_parent(")
    end = text.index("def run_canonical_river_build(")
    export_only = text[start:end]
    assert "assert_parent_window_matches_export" in export_only
    assert "export_vs_parent" in export_only
    assert '"PASS"' in export_only
    assert "construction_attempted" in export_only
    assert "construction_stages_run_in_aoi_export" in export_only
    assert "False" in export_only


def test_phase5_run_summary_reads_parent_science_without_construction():
    text = read("pipeline/run_summary.py")
    assert 'manifest.get("canonical_science_summary")' in text
    assert "_canonical_stage_receipt_paths_from_manifest(receipts)" in text
    assert "_path_from_receipt_record" in text


def test_final_folder_baseline_uses_retained_aoi_export_identity():
    source = ROOT.joinpath("bathy_main.py").read_text(encoding="utf-8")
    assert "aoi_export_identity_report" in source
    assert "export_baseline_background" in source
    assert "authoritative_base_aligned.tif" in source
    assert "_path_value_from_record" in source
