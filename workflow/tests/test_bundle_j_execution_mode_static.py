from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bundle_j_execution_receipt_is_wired_into_active_pipeline() -> None:
    text = (ROOT / "pipeline" / "river_workflow" / "river_workflow_pipeline.py").read_text(encoding="utf-8")
    assert "write_execution_mode_receipt" in text
    assert "effective_mode=\"aoi_export_only\"" in text
    assert "effective_mode=\"canonical_build_then_export\"" in text
    assert "receipts[\"river_execution_mode\"]" in text


def test_bundle_j_execution_contract_has_explicit_roles() -> None:
    text = (ROOT / "pipeline" / "river_workflow" / "river_workflow_execution_contract.py").read_text(encoding="utf-8")
    assert "CANONICAL_BUILD_ROLE" in text
    assert "AOI_EXPORT_ONLY_ROLE" in text
    assert "CANONICAL_CONSTRUCTION_STAGES" in text
    assert "ensure_stage_allowed" in text


def test_bundle_j_execution_receipt_rejects_forbidden_export_only_construction() -> None:
    text = (ROOT / "pipeline" / "river_workflow" / "river_workflow_execution_receipt.py").read_text(encoding="utf-8")
    assert "forbidden_construction_stage_receipts_present" in text
    assert "aoi_run_may_recompute_canonical_construction" in text
    assert "cache_is_implementation_detail" in text
