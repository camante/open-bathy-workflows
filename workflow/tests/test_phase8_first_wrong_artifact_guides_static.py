from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase8_contract_declares_stage_diagnostic_guides():
    src = (ROOT / "pipeline" / "river_workflow" / "river_workflow_contract.py").read_text(encoding="utf-8")
    required = [
        "LINEAR_STAGE_DIAGNOSTIC_CONTRACTS",
        "diagnostic_contract_for_stage",
        "critical_invariant",
        "first_wrong_artifact_hint",
        "centerline_wse_proxy",
        "centerline_modeled_offset",
        "river_primary_surface_solve_locked",
        "final_dem",
    ]
    missing = [item for item in required if item not in src]
    assert not missing, missing


def test_phase8_stage_receipts_include_first_wrong_artifact_guide():
    src = (ROOT / "pipeline" / "river_workflow" / "river_workflow_receipts.py").read_text(encoding="utf-8")
    required = [
        "build_first_wrong_artifact_guide",
        "first_wrong_artifact_guide",
        "diagnostic_only",
        "may_reroute_or_repair_outputs",
        "primary_input",
        "primary_output",
        "critical_invariant",
        "first_check",
        "first_wrong_artifact_hint",
    ]
    missing = [item for item in required if item not in src]
    assert not missing, missing


def test_phase8_guides_are_verify_only_not_routing_logic():
    src = (ROOT / "pipeline" / "river_workflow" / "river_workflow_receipts.py").read_text(encoding="utf-8")
    block_start = src.index("def build_first_wrong_artifact_guide")
    block_end = src.index("def build_stage_validator_summary")
    block = src[block_start:block_end]
    assert "may_reroute_or_repair_outputs" in block
    assert "False" in block
    assert "write_json" not in block
    assert "shutil.copy" not in block
    assert "rasterio.open" not in block


def test_phase8_active_module_manifest_documents_policy():
    src = (ROOT / "active_river_modules.json").read_text(encoding="utf-8")
    required = [
        "active_stage_first_wrong_artifact_guides",
        "does_not_drive_routing",
        "does_not_repair_outputs",
        "does_not_change_construction",
        "first_wrong_artifact_hint",
    ]
    missing = [item for item in required if item not in src]
    assert not missing, missing
