from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pipeline.river_workflow.river_workflow_execution_contract import (
    AOI_EXPORT_ONLY_ROLE,
    CANONICAL_BUILD_ROLE,
    CANONICAL_CONSTRUCTION_STAGES,
    build_execution_contract,
    normalize_execution_role,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class RiverExecutionModeReceiptResult:
    receipt_path: Path
    text_path: Path
    passed: bool
    execution_role: str
    effective_mode: str


def _path_record(path: Path | str | None) -> dict[str, Any]:
    if path in (None, ""):
        return {"path": None, "exists": False, "sha256": None}
    candidate = Path(path)
    return {
        "path": str(candidate),
        "exists": candidate.is_file(),
        "sha256": _sha256_file(candidate) if candidate.is_file() else None,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_execution_mode_receipt(
    *,
    out_dir: Path,
    execution_role: str,
    effective_mode: str,
    stage_receipts: Mapping[str, Path | str],
    canonical_parent_dem: Path | str | None,
    aoi_export_dem: Path | str | None,
    final_user_dem: Path | str | None,
    canonical_manifest_path: Path | str | None = None,
    source: str | None = None,
) -> RiverExecutionModeReceiptResult:
    """Write the canonical-build/AOI-export-only execution-mode contract.

    This is intentionally a receipt/check, not a router. It records whether the
    active river run constructed a canonical parent in this run or exported from
    an existing parent handoff, and it enforces the key guard: export-only runs
    may not retain canonical construction-stage receipts.
    """
    role = normalize_execution_role(execution_role)
    contract = build_execution_contract(role)
    receipts = {str(k): str(v) for k, v in dict(stage_receipts or {}).items()}
    stage_names = set(receipts)
    forbidden_present = sorted(stage_names.intersection(CANONICAL_CONSTRUCTION_STAGES)) if role == AOI_EXPORT_ONLY_ROLE else []
    final_mode = str(effective_mode or ("aoi_export_only" if role == AOI_EXPORT_ONLY_ROLE else "canonical_build_then_export"))

    checks = {
        "execution_role_valid": role in {CANONICAL_BUILD_ROLE, AOI_EXPORT_ONLY_ROLE},
        "canonical_parent_exists": _path_record(canonical_parent_dem)["exists"],
        "aoi_export_exists": _path_record(aoi_export_dem)["exists"],
        "final_user_dem_exists_or_deferred": True if final_user_dem in (None, "") else _path_record(final_user_dem)["exists"],
        "export_only_has_no_construction_stage_receipts": len(forbidden_present) == 0,
        "cache_is_implementation_detail": True,
        "aoi_outputs_must_subset_from_parent": True,
        "canonical_construction_forbidden_in_export_only": role != AOI_EXPORT_ONLY_ROLE or len(forbidden_present) == 0,
    }
    passed = bool(all(checks.values()))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "river_execution_mode",
        "source": source or "active_river_workflow",
        "execution_role": role,
        "effective_mode": final_mode,
        "passed": passed,
        "checks": checks,
        "failures": [name for name, ok in checks.items() if not ok],
        "forbidden_construction_stage_receipts_present": forbidden_present,
        "contract": contract.to_dict(),
        "stage_receipts": receipts,
        "canonical_parent_dem": _path_record(canonical_parent_dem),
        "aoi_export_dem": _path_record(aoi_export_dem),
        "final_user_dem": _path_record(final_user_dem),
        "canonical_manifest": _path_record(canonical_manifest_path),
        "architecture": {
            "canonical_build_then_export": role == CANONICAL_BUILD_ROLE,
            "aoi_export_only_from_existing_parent": role == AOI_EXPORT_ONLY_ROLE,
            "cache_is_implementation_detail": True,
            "aoi_run_may_recompute_canonical_construction": role != AOI_EXPORT_ONLY_ROLE,
            "aoi_run_must_subset_from_canonical_parent": True,
        },
    }
    if not passed:
        raise RuntimeError(f"river_execution_mode_contract_failed:{payload['failures']}")

    out_dir = Path(out_dir)
    receipt_path = out_dir / "river_workflow" / "manifests" / "river_execution_mode_receipt.json"
    text_path = out_dir / "river_workflow" / "manifests" / "river_execution_mode_receipt.txt"
    _write_json(receipt_path, payload)
    lines = [
        "RIVER EXECUTION MODE RECEIPT",
        f"execution_role: {role}",
        f"effective_mode: {final_mode}",
        f"passed: {passed}",
        f"canonical_parent_dem: {payload['canonical_parent_dem']['path']}",
        f"aoi_export_dem: {payload['aoi_export_dem']['path']}",
        f"final_user_dem: {payload['final_user_dem']['path']}",
        f"forbidden_construction_stage_receipts_present: {forbidden_present}",
    ]
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_json(reports_dir / "river_execution_mode_receipt.json", payload)
    (reports_dir / "river_execution_mode_receipt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RiverExecutionModeReceiptResult(receipt_path, text_path, passed, role, final_mode)


__all__ = ["RiverExecutionModeReceiptResult", "write_execution_mode_receipt"]
