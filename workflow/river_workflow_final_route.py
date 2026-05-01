from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pipeline.river_workflow.river_workflow_contract import (
    AOI_EXPORT_DEM_ROLE,
    FINAL_DEM_MATERIALIZER_ROLE,
    FINAL_USER_DEM_ROLE,
    FINAL_USER_DEM_RELATIVE_PATH,
    assert_aoi_export_identity_passed,
    assert_materialization_receipt_valid,
)


@dataclass(frozen=True)
class FinalRouteGuardResult:
    receipt_path: Path
    text_path: Path
    passed: bool
    checks: dict[str, bool]


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path | str | None) -> dict[str, Any]:
    if path in (None, ""):
        return {}
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _path_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    candidate = Path(path)
    return {"path": str(candidate), "exists": candidate.is_file(), "sha256": _sha256_file(candidate)}


def _same_resolved(a: Path | str | None, b: Path | str | None) -> bool:
    if a in (None, "") or b in (None, ""):
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def _final_path_is_contract_path(final_user_dem: Path) -> bool:
    return str(final_user_dem).replace("\\", "/").endswith(FINAL_USER_DEM_RELATIVE_PATH)


def _touch_log_actions_ok(materialization_payload: Mapping[str, Any]) -> bool:
    touch_path = materialization_payload.get("touch_log_path")
    if touch_path in (None, ""):
        return False
    path = Path(str(touch_path))
    if not path.is_file():
        return False
    allowed_actions = {"final_dem_materializer_write"}
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("action") not in allowed_actions:
                return False
            if payload.get("writer_role") != FINAL_DEM_MATERIALIZER_ROLE:
                return False
            seen += 1
    return seen >= 1


def validate_final_route_guard(
    *,
    canonical_parent_dem: Path,
    aoi_export_dem: Path,
    final_user_dem: Path,
    materialization_receipt: Path,
    identity_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the non-negotiable final route contract.

    The active river final route is intentionally narrow:
    canonical_parent_dem -> exact AOI export -> copy-only final user DEM.
    This validator does not repair anything and does not choose a fallback path;
    it only records the checks and raises when a hard invariant is violated.
    """
    parent = Path(canonical_parent_dem)
    export = Path(aoi_export_dem)
    final = Path(final_user_dem)
    materialization = Path(materialization_receipt)
    materialization_payload = _read_json(materialization)

    checks: dict[str, bool] = {
        "canonical_parent_exists": parent.is_file(),
        "aoi_export_exists": export.is_file(),
        "final_user_dem_exists": final.is_file(),
        "materialization_receipt_exists": materialization.is_file(),
        "final_path_is_combined_dem_enhanced": _final_path_is_contract_path(final),
        "materialization_source_is_named_aoi_export": _same_resolved(materialization_payload.get("source_path"), export),
        "materialization_destination_is_final_user_dem": _same_resolved(materialization_payload.get("destination_path"), final),
        "materialization_source_role_is_aoi_export": materialization_payload.get("source_role") == AOI_EXPORT_DEM_ROLE,
        "materialization_destination_role_is_final_user_dem": materialization_payload.get("destination_role") == FINAL_USER_DEM_ROLE,
        "materialization_writer_role_is_final_materializer": materialization_payload.get("writer_role") == FINAL_DEM_MATERIALIZER_ROLE,
        "materialization_copy_only": bool(materialization_payload.get("copy_only")) is True,
        "materialization_no_resample": bool(materialization_payload.get("resampled")) is False,
        "materialization_no_reproject": bool(materialization_payload.get("reprojected")) is False,
        "materialization_no_pixel_modification": bool(materialization_payload.get("pixel_values_modified")) is False,
        "materialization_no_interpolation": bool(materialization_payload.get("terrain_interpolation")) is False,
        "materialization_no_blending": bool(materialization_payload.get("blending")) is False,
        "materialization_no_authoritative_locking": bool(materialization_payload.get("authoritative_locking")) is False,
        "materialization_no_aoi_local_solve": bool(materialization_payload.get("aoi_local_solve")) is False,
        "touch_log_actions_are_materializer_only": _touch_log_actions_ok(materialization_payload),
        "identity_checked_against_parent": bool(identity_result.get("checked_against_parent")),
        "identity_passed": bool(identity_result.get("passed")),
        "identity_zero_tolerance": float(identity_result.get("tolerance", 0.0) or 0.0) == 0.0,
        "identity_zero_mismatch_pixels": int(identity_result.get("mismatch_pixels") or 0) == 0,
        "identity_zero_max_abs_diff": float(identity_result.get("max_abs_diff") or 0.0) == 0.0,
    }
    export_hash = _sha256_file(export)
    final_hash = _sha256_file(final)
    parent_hash = _sha256_file(parent)
    checks["final_hash_matches_aoi_export_hash"] = export_hash is not None and export_hash == final_hash
    checks["parent_hash_recorded"] = parent_hash is not None
    checks["parent_export_final_are_distinct_roles"] = parent.resolve() != final.resolve() and export.resolve() != parent.resolve()

    # Reuse the stricter existing validators so this guard cannot drift from the
    # centralized role contract.
    assert_materialization_receipt_valid(materialization_payload)
    assert_aoi_export_identity_passed(identity_result)

    passed = all(checks.values())
    failures = [name for name, ok in checks.items() if not ok]
    payload = {
        "schema_version": 1,
        "stage": "final_route_guard",
        "route": "canonical_parent_dem -> aoi_export_dem -> final_user_dem",
        "passed": passed,
        "failures": failures,
        "checks": checks,
        "canonical_parent_dem": _path_record(parent),
        "aoi_export_dem": _path_record(export),
        "final_user_dem": _path_record(final),
        "materialization_receipt": _path_record(materialization),
        "identity": {
            "checked_against_parent": bool(identity_result.get("checked_against_parent")),
            "passed": bool(identity_result.get("passed")),
            "tolerance": identity_result.get("tolerance"),
            "mismatch_pixels": identity_result.get("mismatch_pixels"),
            "max_abs_diff": identity_result.get("max_abs_diff"),
            "failure_reason": identity_result.get("failure_reason"),
        },
        "invariant": "Final DEM must be a copy-only materialization of the named AOI export, and that AOI export must be an exact zero-tolerance parent-grid subset of the canonical parent DEM.",
    }
    if not passed:
        raise RuntimeError(f"final_route_guard_failed:{failures}")
    return payload


def write_final_route_guard_receipt(
    *,
    out_dir: Path,
    canonical_parent_dem: Path,
    aoi_export_dem: Path,
    final_user_dem: Path,
    materialization_receipt: Path,
    identity_result: Mapping[str, Any],
) -> FinalRouteGuardResult:
    final_dir = Path(out_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = final_dir / "river_final_route_guard_receipt.json"
    text_path = final_dir / "river_final_route_guard_receipt.txt"
    payload = validate_final_route_guard(
        canonical_parent_dem=Path(canonical_parent_dem),
        aoi_export_dem=Path(aoi_export_dem),
        final_user_dem=Path(final_user_dem),
        materialization_receipt=Path(materialization_receipt),
        identity_result=identity_result,
    )
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "FINAL ROUTE GUARD",
        f"route: {payload['route']}",
        f"passed: {payload['passed']}",
        f"canonical_parent_dem: {payload['canonical_parent_dem']['path']}",
        f"aoi_export_dem: {payload['aoi_export_dem']['path']}",
        f"final_user_dem: {payload['final_user_dem']['path']}",
        f"final_hash_matches_aoi_export_hash: {payload['checks']['final_hash_matches_aoi_export_hash']}",
        f"identity_passed: {payload['checks']['identity_passed']}",
        f"identity_zero_max_abs_diff: {payload['checks']['identity_zero_max_abs_diff']}",
        f"touch_log_actions_are_materializer_only: {payload['checks']['touch_log_actions_are_materializer_only']}",
    ]
    if payload["failures"]:
        lines.append("failures:")
        lines.extend(f"- {item}" for item in payload["failures"])
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return FinalRouteGuardResult(
        receipt_path=receipt_path,
        text_path=text_path,
        passed=bool(payload["passed"]),
        checks=dict(payload["checks"]),
    )


__all__ = [
    "FinalRouteGuardResult",
    "validate_final_route_guard",
    "write_final_route_guard_receipt",
]
