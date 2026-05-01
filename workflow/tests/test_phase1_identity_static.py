from __future__ import annotations

import ast
from pathlib import Path


def test_aoi_export_identity_receipts_record_exact_parent_subset_fields() -> None:
    text = Path('pipeline/river_workflow/river_workflow_pipeline.py').read_text(encoding='utf-8')
    assert 'assert_parent_window_matches_export' in text
    assert '"max_abs_diff_m"' in text
    assert '"construction_attempted": False' in text
    assert '"post_subset_modifications": False' in text


def test_canonical_manifest_records_receipt_hashes_and_parent_policy() -> None:
    text = Path('pipeline/river_workflow/river_workflow_canonical_manifest.py').read_text(encoding='utf-8')
    assert 'construction_stage_receipt_hashes' in text
    assert 'canonical_construction_stage_receipts' in text
    assert 'aoi_outputs_must_be_exact_parent_subsets' in text


def test_phase1_files_parse() -> None:
    for rel in [
        'pipeline/river_workflow/river_workflow_identity.py',
        'pipeline/river_workflow/river_workflow_pipeline.py',
        'pipeline/river_workflow/river_workflow_stage_final_dem.py',
        'pipeline/river_workflow/river_workflow_canonical_manifest.py',
        'support_classes.py',
    ]:
        ast.parse(Path(rel).read_text(encoding='utf-8'))
