from pathlib import Path


def test_phase4_handoff_dataclass_carries_single_parent_contract():
    src = Path("pipeline/river_workflow/river_workflow_pipeline.py").read_text(encoding="utf-8")
    start = src.index("class CanonicalParentHandoff:")
    end = src.index("class CanonicalRiverBuildResult:")
    block = src[start:end]
    required = [
        "manifest_path: Path",
        "parent_dem_path: Path",
        "cache_manifest_path: Path | None",
        "source_kind: str",
        "canonical_system_id: str | None",
        "canonical_cache_key: str | None",
        "canonical_parent_hash: str | None",
        "def source_summary(",
    ]
    missing = [item for item in required if item not in block]
    assert not missing, missing


def test_phase4_export_consumes_handoff_not_cache_branches():
    src = Path("pipeline/river_workflow/river_workflow_pipeline.py").read_text(encoding="utf-8")
    start = src.index("def run_aoi_export_from_canonical_parent(")
    end = src.index("def run_canonical_river_build(")
    export_block = src[start:end]
    assert "CanonicalParentHandoff" in export_block
    assert "parent_handoff = handoff" in export_block
    assert "parent_handoff.source_summary()" in export_block
    assert "load_canonical_solve_cache" not in export_block
    assert "_bootstrap_cache_manifest_for_existing_parent" not in export_block


def test_phase4_resolver_collapses_existing_sources_into_handoff():
    src = Path("pipeline/river_workflow/river_workflow_pipeline.py").read_text(encoding="utf-8")
    start = src.index("def resolve_canonical_parent_handoff(")
    end = src.index("def run_river_parent_export_workflow(")
    resolver = src[start:end]
    assert "_canonical_parent_handoff_from_manifest(" in resolver
    assert "source_kind =" in resolver
    assert "adopted_existing_parent" in resolver
    assert "return CanonicalParentHandoff(" not in resolver


def test_phase4_boundary_documented():
    doc = Path("docs/ACTIVE_RIVER_MODULES.md").read_text(encoding="utf-8")
    manifest = Path("active_river_modules.json").read_text(encoding="utf-8")
    assert "Phase 4 canonical parent handoff boundary" in doc
    assert "CanonicalParentHandoff" in manifest
    assert "active_parent_handoff" in manifest
