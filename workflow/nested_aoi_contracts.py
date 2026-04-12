from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from support_classes import SUPPORT_CLASS_CODE_TO_NAME

REQUIRED_OVERLAP_ARTIFACTS: tuple[str, ...] = (
    "selected_final_depth",
    "support_class",
    "final_provenance_native",
)

REQUIRED_TRUSTED_ARTIFACTS: tuple[str, ...] = (
    "river_trusted_interior",
)


def _existing(path: Optional[str | Path]) -> Optional[str]:
    if not path:
        return None
    p = Path(str(path))
    return str(p) if p.exists() else None


def _artifact_presence(manifest: Dict[str, Any], keys: Iterable[str]) -> Dict[str, bool]:
    return {str(key): bool(_existing(manifest.get(str(key)))) for key in keys}


def _evaluation_status(*, all_ok: bool, missing_required: bool, missing_trusted: bool) -> str:
    if missing_required:
        return "missing_required_artifacts"
    if missing_trusted:
        return "missing_trusted_artifacts"
    return "ok" if all_ok else "failed"


def evaluate_nested_aoi_contracts(
    *,
    current_final_outputs_manifest: str | Path,
    nested_payload: Dict[str, Any],
    overlap_tolerance: float,
    trusted_tolerance: float,
) -> Dict[str, Any]:
    current_manifest_path = Path(str(current_final_outputs_manifest))
    current_manifest = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    current_required_presence = _artifact_presence(current_manifest, REQUIRED_OVERLAP_ARTIFACTS)
    current_trusted_presence = _artifact_presence(current_manifest, REQUIRED_TRUSTED_ARTIFACTS)

    comparisons_out: List[Dict[str, Any]] = []
    any_missing_required = not all(current_required_presence.values())
    any_missing_trusted = not all(current_trusted_presence.values())
    all_overlap_ok = True
    all_trusted_ok = True
    class_scope_rows = []

    for comparison in nested_payload.get("comparisons", []):
        neighbor_path = Path(str(comparison.get("neighbor_final_outputs_manifest")))
        neighbor_manifest = {}
        if neighbor_path.exists():
            neighbor_manifest = json.loads(neighbor_path.read_text(encoding="utf-8"))
        neighbor_required_presence = _artifact_presence(neighbor_manifest, REQUIRED_OVERLAP_ARTIFACTS)
        neighbor_trusted_presence = _artifact_presence(neighbor_manifest, REQUIRED_TRUSTED_ARTIFACTS)

        overlap_eval = comparison.get("overlap_identity_evaluation") or {}
        trusted_eval = comparison.get("trusted_interior_identity_evaluation") or {}
        class_checks = comparison.get("support_class_depth_identity_checks") or []
        trusted_class_checks = comparison.get("trusted_support_class_depth_identity_checks") or []
        for row in class_checks:
            row["contract_tolerance"] = float(overlap_tolerance)
            row["contract_status"] = "ok" if (row.get("status") == "ok" and float(row.get("max_abs") or 0.0) <= float(overlap_tolerance)) else ("no_valid" if row.get("status") != "ok" else "failed")
        for row in trusted_class_checks:
            row["contract_tolerance"] = float(trusted_tolerance)
            row["contract_status"] = "ok" if (row.get("status") == "ok" and float(row.get("max_abs") or 0.0) <= float(trusted_tolerance)) else ("no_valid" if row.get("status") != "ok" else "failed")

        class_scope_rows.extend(class_checks)
        class_scope_rows.extend(trusted_class_checks)

        comparison_missing_required = (not all(current_required_presence.values())) or (not all(neighbor_required_presence.values()))
        comparison_missing_trusted = (not all(current_trusted_presence.values())) or (not all(neighbor_trusted_presence.values()))
        overlap_ok = bool(overlap_eval.get("all_ok")) and not comparison_missing_required
        trusted_ok = bool(trusted_eval.get("all_ok")) and not comparison_missing_trusted

        any_missing_required = any_missing_required or comparison_missing_required
        any_missing_trusted = any_missing_trusted or comparison_missing_trusted
        all_overlap_ok = all_overlap_ok and overlap_ok
        all_trusted_ok = all_trusted_ok and trusted_ok

        comparisons_out.append({
            "neighbor_final_outputs_manifest": str(neighbor_path),
            "neighbor_aoi": comparison.get("neighbor_aoi"),
            "required_overlap_artifacts": {
                "current": current_required_presence,
                "neighbor": neighbor_required_presence,
            },
            "required_trusted_artifacts": {
                "current": current_trusted_presence,
                "neighbor": neighbor_trusted_presence,
            },
            "overlap_tolerance": float(overlap_tolerance),
            "trusted_tolerance": float(trusted_tolerance),
            "overlap_status": _evaluation_status(
                all_ok=overlap_ok,
                missing_required=comparison_missing_required,
                missing_trusted=False,
            ),
            "trusted_status": _evaluation_status(
                all_ok=trusted_ok,
                missing_required=False,
                missing_trusted=comparison_missing_trusted,
            ),
            "overlap_identity_evaluation": overlap_eval,
            "trusted_interior_identity_evaluation": trusted_eval,
            "support_class_depth_identity_checks": class_checks,
            "trusted_support_class_depth_identity_checks": trusted_class_checks,
        })

    class_failures = []
    trusted_class_failures = []
    for row in class_scope_rows:
        scope = str(row.get("scope") or "")
        status = str(row.get("contract_status") or "")
        if status == "failed":
            if scope == "trusted_interior":
                trusted_class_failures.append(row)
            else:
                class_failures.append(row)

    return {
        "current_final_outputs_manifest": str(current_manifest_path),
        "required_overlap_artifacts": list(REQUIRED_OVERLAP_ARTIFACTS),
        "required_trusted_artifacts": list(REQUIRED_TRUSTED_ARTIFACTS),
        "comparisons": comparisons_out,
        "support_class_code_to_name": {str(k): v for k, v in SUPPORT_CLASS_CODE_TO_NAME.items()},
        "all_required_overlap_artifacts_present": not any_missing_required,
        "all_required_trusted_artifacts_present": not any_missing_trusted,
        "all_overlap_contracts_ok": all_overlap_ok,
        "all_trusted_contracts_ok": all_trusted_ok,
        "first_overlap_class_failures": class_failures[:10],
        "first_trusted_class_failures": trusted_class_failures[:10],
    }
