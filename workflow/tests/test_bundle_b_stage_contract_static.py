from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stage_artifact_contract_writer_is_active() -> None:
    text = (ROOT / "active_pipeline.py").read_text(encoding="utf-8")
    assert "pipeline.river_workflow.river_workflow_stage_artifact_contract" in text
    assert "write_stage_artifact_contract" in text
    assert "river_stage_artifact_contract" in text
    assert "_write_active_river_stage_artifact_contract(ctx, stage_results)" in text
    assert "_write_active_river_stage_artifact_contract(ctx, workflow_result.stage_results)" in text


def test_active_wrapper_has_no_broad_exception_masking() -> None:
    for rel in [
        "active_pipeline.py",
        "river_workflow.py",
        "river_workflow_entry.py",
        "river_workflow_final_route.py",
        "river_runner.py",
        "pipeline/river_workflow/river_workflow_stage_artifact_contract.py",
    ]:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "except Exception" not in text, rel
        ast.parse(text)


def test_stage_artifact_contract_module_exposes_schema_and_validation() -> None:
    text = (ROOT / "pipeline/river_workflow/river_workflow_stage_artifact_contract.py").read_text(encoding="utf-8")
    assert "ACTIVE_RIVER_STAGE_CONTRACT_SCHEMA" in text
    assert "REQUIRED_STAGE_CONTRACT_FIELDS" in text
    assert "build_stage_artifact_contract" in text
    assert "write_stage_artifact_contract" in text
    assert "validate_stage_artifact_contract" in text
    assert "RiverWorkflowContractError" in text
