from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding='utf-8')


def test_canonical_manifest_requires_exact_parent_export_contract():
    text = read('pipeline/river_workflow/river_workflow_canonical_manifest.py')
    assert 'manifest_handoff_required' in text
    assert 'post_subset_modification_allowed' in text
    assert 'exact_parent_window_identity_required' in text
    assert 'canonical_construction_stage_receipts' in text
    assert 'canonical_parent_dem_sha256' in text


def test_aoi_export_identity_has_public_receipt_and_hard_flags():
    text = read('pipeline/river_workflow/river_workflow_stage_final_dem.py')
    assert "reports' / 'aoi_export_identity.json'" in text
    assert "reports' / 'canonical_parent_identity.json'" in text
    assert 'assert_parent_window_matches_export' in text
    assert "'identity_contract'" in text
    assert "'max_abs_diff_required_m': 0.0" in text
    assert "'post_subset_modification_allowed': False" in text
    assert "'aoi_construction_allowed': False" in text
    assert "'construction_stages_run_in_aoi_export': False" in text


def test_export_only_route_receipt_exposes_contract_flags():
    text = read('pipeline/river_workflow/river_workflow_pipeline.py')
    assert 'aoi_export_only_retained_construction_stage_receipts' in text
    assert 'assert_parent_window_matches_export' in text
    assert 'identity_contract' in text
    assert 'exact_parent_window_identity_required' in text


def test_parent_and_export_receipts_record_hashes_and_no_post_subset_modification():
    text = read('pipeline/river_workflow/river_workflow_receipts.py')
    assert "'parent_dem_sha256': sha256_file(parent_dem)" in text
    assert "'export_dem_sha256': sha256_file(export_dem)" in text
    assert "'export_template_sha256': sha256_file(export_template)" in text
    assert "'construction_attempted': False" in text
    assert "'construction_stages_run_in_aoi_export': False" in text
    assert "'aoi_outputs_must_be_exact_parent_subsets': True" in text
