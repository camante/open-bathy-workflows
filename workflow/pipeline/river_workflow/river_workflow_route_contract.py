"""Read-only route-contract checks for the active canonical river workflow.

These helpers intentionally validate only retained receipts/manifests. They do not open rasters, recompute WSE/offset/backbone stages, or discover hidden
inputs. The contract being checked is:

    canonical_parent_dem -> aoi_export_dem -> final_user_dem
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json


def read_json_record(path: Path | str | None) -> dict[str, Any] | None:
    if path in (None, ""):
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _get(payload: Mapping[str, Any] | None, *keys: str) -> Any:
    if not isinstance(payload, Mapping):
        return None
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _nested(payload: Mapping[str, Any] | None, *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _truthy_false(value: Any) -> bool:
    return value is False or str(value).lower() == "false"


def validate_route_contract_payloads(
    *,
    canonical_manifest: Mapping[str, Any] | None,
    aoi_export_identity: Mapping[str, Any] | None,
    river_workflow_receipt: Mapping[str, Any] | None = None,
    final_output_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the active parent/export/final evidence chain from receipts only."""
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    parent_hash = _get(canonical_manifest, "canonical_parent_dem_sha256", "canonical_final_dem_sha256")
    export_vs_parent = _get(aoi_export_identity, "export_vs_parent")
    construction_attempted = _get(aoi_export_identity, "construction_attempted", "construction_stages_run_in_aoi_export")
    identity_contract = _nested(aoi_export_identity, "identity_contract")
    aoi_policy = _nested(canonical_manifest, "aoi_export_policy")
    science_policy = _nested(canonical_manifest, "canonical_science_summary_policy")
    workflow_identity = _nested(river_workflow_receipt, "identity")

    checks["canonical_manifest_present"] = isinstance(canonical_manifest, Mapping)
    checks["canonical_manifest_role"] = _get(canonical_manifest, "role") == "canonical_river_solution"
    checks["canonical_parent_hash_present"] = parent_hash not in (None, "")
    checks["aoi_export_identity_present"] = isinstance(aoi_export_identity, Mapping)
    checks["aoi_export_vs_parent_pass"] = export_vs_parent == "PASS"
    checks["aoi_export_no_construction"] = construction_attempted is False or _truthy_false(construction_attempted)
    checks["identity_requires_exact_parent_window"] = bool(_nested(identity_contract, "same_parent_window_required"))
    checks["identity_forbids_post_subset_modification"] = _nested(identity_contract, "post_subset_modification_allowed") is False
    checks["manifest_requires_parent_subset"] = bool(_nested(aoi_policy, "aoi_runs_must_subset_from_parent"))
    checks["manifest_forbids_aoi_recompute"] = _nested(aoi_policy, "aoi_runs_may_recompute_canonical_construction") is False
    checks["science_summary_read_only"] = _nested(science_policy, "aoi_exports_read_only") is True and _nested(science_policy, "aoi_exports_may_recompute_science") is False

    if isinstance(workflow_identity, Mapping):
        checks["river_workflow_combined_equals_export"] = workflow_identity.get("combined_vs_export_exact") is not False
        checks["river_workflow_single_writer"] = workflow_identity.get("single_writer_pass") is not False
    else:
        checks["river_workflow_combined_equals_export"] = True
        checks["river_workflow_single_writer"] = True

    if isinstance(final_output_receipt, Mapping):
        missing = final_output_receipt.get("missing_outputs")
        checks["final_output_receipt_complete"] = not bool(missing)
    else:
        checks["final_output_receipt_complete"] = True

    details["parent_hash"] = parent_hash
    details["export_hash"] = _get(aoi_export_identity, "export_hash")
    details["route"] = "canonical_parent_dem -> aoi_export_dem -> final_user_dem"
    failed = [name for name, ok in checks.items() if ok is not True]
    return {
        "contract": "river_phase5_parent_export_route_contract_v1",
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "details": details,
        "read_only": True,
        "recomputed_construction_stages": False,
    }


__all__ = ["read_json_record", "validate_route_contract_payloads"]
