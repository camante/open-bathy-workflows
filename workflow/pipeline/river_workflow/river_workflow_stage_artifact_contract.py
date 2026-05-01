"""Active river stage-artifact contract helpers.

This module is intentionally small and independent of the science stages.  It
records the active workflow's one-artifact/one-receipt stage contract without
creating an alternate construction route.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

ACTIVE_RIVER_WORKFLOW = "canonical_parent_aoi_export"
ACTIVE_RIVER_CONTRACT = "one_canonical_parent_one_aoi_export_one_final_writer"
ACTIVE_RIVER_STAGE_CONTRACT_SCHEMA = "active_river_stage_artifact_contract_v1"
ACTIVE_RIVER_RUNNER_CONTRACT = "active_canonical_parent_aoi_export_v1"
REQUIRED_FINAL_DEM_SOURCE = "aoi_export_dem"

REQUIRED_STAGE_CONTRACT_FIELDS = (
    "stage_name",
    "stage_class",
    "status",
    "primary_input",
    "primary_output",
    "primary_output_exists",
    "primary_artifact_role",
    "receipt_path",
    "receipt_exists",
)


class RiverWorkflowContractError(ValueError):
    """Raised when the active river workflow violates its artifact contract."""


def path_exists(value: Any) -> bool:
    """Return True only when *value* names an existing filesystem object."""
    if value in (None, ""):
        return False
    return Path(value).exists()


def validate_stage_artifact_contract(stage_results: Iterable[Any]) -> None:
    """Validate the one primary output/receipt contract around active stages."""
    missing_outputs: list[str] = []
    invalid_classes: list[str] = []
    for stage in stage_results:
        stage_name = str(getattr(stage, "stage_name", ""))
        status = str(getattr(stage, "status", ""))
        stage_class = str(getattr(stage, "stage_class", ""))
        primary_output = getattr(stage, "primary_output", None)
        if stage_class not in {"solve", "export", "finalize"}:
            invalid_classes.append(stage_name)
        if status == "success" and primary_output in (None, ""):
            missing_outputs.append(stage_name)
    if invalid_classes:
        raise RiverWorkflowContractError(f"active_river_invalid_stage_class:{invalid_classes}")
    if missing_outputs:
        raise RiverWorkflowContractError(f"active_river_stage_missing_primary_output:{missing_outputs}")


def _stage_to_contract_record(stage: Any) -> dict[str, Any]:
    primary_input = getattr(stage, "primary_input", None)
    primary_output = getattr(stage, "primary_output", None)
    receipt_path = getattr(stage, "receipt_path", None)
    return {
        "stage_name": str(getattr(stage, "stage_name", "")),
        "stage_class": str(getattr(stage, "stage_class", "")),
        "status": str(getattr(stage, "status", "")),
        "primary_input": str(primary_input) if primary_input not in (None, "") else None,
        "primary_input_exists": path_exists(primary_input),
        "primary_output": str(primary_output) if primary_output not in (None, "") else None,
        "primary_output_exists": path_exists(primary_output),
        "primary_artifact_role": getattr(stage, "primary_artifact_role", None),
        "receipt_path": str(receipt_path) if receipt_path not in (None, "") else None,
        "receipt_exists": path_exists(receipt_path),
        "detail": getattr(stage, "detail", None),
    }


def build_stage_artifact_contract(
    *,
    run_id: str,
    stage_results: Iterable[Any],
    canonical_system_id: Any = None,
) -> dict[str, Any]:
    """Build a serializable stage-artifact contract from active stage results."""
    records = [_stage_to_contract_record(stage) for stage in stage_results]
    successful = [item for item in records if item.get("status") == "success"]
    missing_outputs = [
        item["stage_name"]
        for item in successful
        if item.get("primary_output") and not item.get("primary_output_exists")
    ]
    missing_receipts = [
        item["stage_name"]
        for item in successful
        if item.get("receipt_path") and not item.get("receipt_exists")
    ]
    return {
        "schema": ACTIVE_RIVER_STAGE_CONTRACT_SCHEMA,
        "workflow": ACTIVE_RIVER_WORKFLOW,
        "contract": ACTIVE_RIVER_CONTRACT,
        "run_id": str(run_id),
        "canonical_system_id": str(canonical_system_id) if canonical_system_id not in (None, "") else None,
        "stage_count": len(records),
        "successful_stage_count": len(successful),
        "failed_stage_count": sum(1 for item in records if item.get("status") == "failed"),
        "skipped_stage_count": sum(1 for item in records if item.get("status") == "skipped"),
        "missing_success_outputs": missing_outputs,
        "missing_success_receipts": missing_receipts,
        "all_success_outputs_exist": not missing_outputs,
        "all_success_receipts_exist": not missing_receipts,
        "stages": records,
    }


def write_stage_artifact_contract(
    *,
    out_dir: Path,
    run_id: str,
    stage_results: Iterable[Any],
    canonical_system_id: Any = None,
) -> tuple[Path, Path]:
    """Write JSON and text summaries for the active river stage contract."""
    reports_dir = Path(out_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = build_stage_artifact_contract(
        run_id=run_id,
        stage_results=stage_results,
        canonical_system_id=canonical_system_id,
    )
    json_path = reports_dir / "river_stage_artifact_contract.json"
    txt_path = reports_dir / "river_stage_artifact_contract.txt"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"workflow: {payload['workflow']}",
        f"contract: {payload['contract']}",
        f"schema: {payload['schema']}",
        f"run_id: {payload['run_id']}",
        f"canonical_system_id: {payload.get('canonical_system_id')}",
        f"stage_count: {payload['stage_count']}",
        f"all_success_outputs_exist: {payload['all_success_outputs_exist']}",
        f"all_success_receipts_exist: {payload['all_success_receipts_exist']}",
        "stages:",
    ]
    for item in payload["stages"]:
        lines.append(
            "  - {stage_name} [{stage_class}] {status}: {role} -> {output}".format(
                stage_name=item.get("stage_name"),
                stage_class=item.get("stage_class"),
                status=item.get("status"),
                role=item.get("primary_artifact_role"),
                output=item.get("primary_output"),
            )
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, txt_path


__all__ = [
    "ACTIVE_RIVER_CONTRACT",
    "ACTIVE_RIVER_RUNNER_CONTRACT",
    "ACTIVE_RIVER_STAGE_CONTRACT_SCHEMA",
    "ACTIVE_RIVER_WORKFLOW",
    "REQUIRED_FINAL_DEM_SOURCE",
    "REQUIRED_STAGE_CONTRACT_FIELDS",
    "RiverWorkflowContractError",
    "build_stage_artifact_contract",
    "path_exists",
    "validate_stage_artifact_contract",
    "write_stage_artifact_contract",
]
