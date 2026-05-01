from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATHY_MAIN = (ROOT / "bathy_main.py").read_text(encoding="utf-8")


def test_pre_canonical_source_resolution_does_not_require_export_rasters():
    resolver_start = BATHY_MAIN.index("def _resolve_river_shared_source_artifacts")
    resolver_end = BATHY_MAIN.index("def _canonical_seed_aoi_for_shared_solve", resolver_start)
    resolver = BATHY_MAIN[resolver_start:resolver_end]
    assert "river_workflow_authoritative_source_contract_unavailable" not in resolver
    assert "river_workflow_baseline_source_contract_unavailable" not in resolver
    assert "canonical_materialization_pending" in resolver
    assert "river_workflow_missing_positive_resolution_for_canonical_source_bundle" in resolver


def test_canonical_materialization_writes_authoritative_and_baseline_contracts():
    prepare_start = BATHY_MAIN.index("def _prepare_linear_canonical_source_bundle")
    prepare_end = BATHY_MAIN.index("def _build_simple_river_status_from_v2", prepare_start)
    prepare = BATHY_MAIN[prepare_start:prepare_end]
    assert "canonical_authoritative_contract_path = write_linear_authoritative_source_contract" in prepare
    assert "canonical_baseline_contract_path = write_linear_baseline_source_contract" in prepare
    assert "authoritative_source_contract_path=Path(canonical_authoritative_contract_path)" in prepare
    assert "baseline_source_contract_path=Path(canonical_baseline_contract_path)" in prepare
    assert "solve_stages_sample_only_from_canonical_authoritative" in prepare
