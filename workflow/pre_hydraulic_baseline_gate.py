from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_bathy_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    stats = payload.get("stats")
    if isinstance(stats, dict):
        bathy = stats.get("bathy_report")
        if isinstance(bathy, dict):
            return bathy
    return payload


def _truthy_path(value: Any, *, base_dir: str | Path | None = None) -> Optional[Path]:
    if not value:
        return None
    try:
        path = Path(str(value))
    except (TypeError, ValueError, OSError):
        return None
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def _existing_json(value: Any, *, base_dir: str | Path | None = None) -> Optional[Dict[str, Any]]:
    path = _truthy_path(value, base_dir=base_dir)
    if path is None or not path.exists():
        return None
    return _read_json(path)


def _nested_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def evaluate_pre_hydraulic_baseline(
    *,
    report_json: str | Path,
    river_stability_summary_json: str | Path | None = None,
    expected_sdb_mode: str | None = None,
    require_profile_coverage: bool = True,
    require_unique_frame_station_contract: bool = True,
    require_xs_support_status_receipt: bool = True,
    require_manifest_valid: bool = True,
    require_authoritative_lock_ok: bool = True,
    require_trusted_interior_identity: bool = False,
    max_profile_unsupported_fraction: float | None = None,
) -> Dict[str, Any]:
    report_path = Path(report_json)
    report_base_dir = report_path.resolve().parent if report_path.exists() else report_path.parent
    payload = _read_json(report_json)
    report = _extract_bathy_report(payload)
    river = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    river_outputs = river.get("outputs", {}) if isinstance(river.get("outputs", {}), dict) else {}
    execution_receipts = river.get("execution_receipts", {}) if isinstance(river.get("execution_receipts", {}), dict) else {}
    final_route = report.get("final_dem_route", {}) if isinstance(report.get("final_dem_route", {}), dict) else {}
    manifest_contract = final_route.get("manifest_contract", {}) if isinstance(final_route.get("manifest_contract", {}), dict) else {}
    auth_contract = final_route.get("authoritative_first_contract", {}) if isinstance(final_route.get("authoritative_first_contract", {}), dict) else {}

    checks: list[Dict[str, Any]] = []

    def add_check(name: str, ok: bool | None, *, value: Any = None, expected: Any = None, reason: str | None = None) -> None:
        checks.append(
            {
                "name": name,
                "ok": ok,
                "value": value,
                "expected": expected,
                "reason": reason,
            }
        )

    sdb_status = None
    if isinstance(payload.get("stats"), dict):
        sdb_status = _nested_get(payload, "stats", "status", "sdb")
    if sdb_status is None:
        sdb_status = report.get("sdb", {}).get("status") if isinstance(report.get("sdb", {}), dict) else None
    if expected_sdb_mode:
        normalized = str(expected_sdb_mode).strip().lower()
        if normalized in {"inactive", "skipped", "off"}:
            ok = str(sdb_status).lower() in {"inactive", "skipped", "not_requested", "none"}
        elif normalized in {"active", "run", "requested"}:
            ok = str(sdb_status).lower() not in {"inactive", "skipped", "not_requested", "none", ""}
        else:
            ok = str(sdb_status).lower() == normalized
        add_check("sdb_mode", ok, value=sdb_status, expected=expected_sdb_mode, reason="Unexpected SDB execution mode")

    river_status = river.get("status")
    add_check("river_status_success", str(river_status).lower() == "success", value=river_status, expected="success", reason="River workflow did not finish successfully")

    if require_authoritative_lock_ok:
        ok = bool(auth_contract.get("ok", False)) and int(auth_contract.get("locked_changed_count", 0)) == 0
        add_check(
            "authoritative_lock_ok",
            ok,
            value={
                "ok": auth_contract.get("ok"),
                "locked_changed_count": auth_contract.get("locked_changed_count"),
            },
            expected={"ok": True, "locked_changed_count": 0},
            reason="Authoritative lock contract not preserved",
        )

    if require_manifest_valid:
        add_check(
            "manifest_contract_valid",
            bool(manifest_contract.get("all_manifest_contracts_valid", False)),
            value=manifest_contract.get("all_manifest_contracts_valid"),
            expected=True,
            reason="Final route manifest contract invalid",
        )

    xs_receipt = execution_receipts.get("xs_support_status")
    if require_xs_support_status_receipt:
        has_status = isinstance(xs_receipt, dict) and bool(xs_receipt.get("status"))
        add_check(
            "xs_support_status_receipt_present",
            has_status,
            value=xs_receipt,
            expected="explicit xs_support_status receipt",
            reason="XS support status is not explicitly recorded",
        )

    profile_coverage_path = _truthy_path(river_outputs.get("longitudinal_profile_coverage"), base_dir=report_base_dir)
    profile_summary = _existing_json(river_outputs.get("longitudinal_profile_summary"), base_dir=report_base_dir) or {}
    coverage_summary = profile_summary.get("coverage_summary", {}) if isinstance(profile_summary.get("coverage_summary", {}), dict) else {}
    if require_profile_coverage:
        exists = profile_coverage_path is not None and profile_coverage_path.exists()
        add_check(
            "longitudinal_profile_coverage_present",
            exists,
            value=str(profile_coverage_path) if profile_coverage_path else None,
            expected="existing longitudinal profile coverage artifact",
            reason="Longitudinal profile coverage artifact missing",
        )
    if max_profile_unsupported_fraction is not None:
        unsupported_fraction = coverage_summary.get("unsupported_fraction")
        ok = unsupported_fraction is not None and float(unsupported_fraction) <= float(max_profile_unsupported_fraction)
        add_check(
            "longitudinal_profile_unsupported_fraction",
            ok,
            value=unsupported_fraction,
            expected=f"<= {float(max_profile_unsupported_fraction)}",
            reason="Unsupported longitudinal profile fraction exceeds threshold",
        )

    frame_contract = _existing_json(river_outputs.get("channel_frame_contract"), base_dir=report_base_dir) or {}
    frame_metrics = frame_contract.get("metrics", {}) if isinstance(frame_contract.get("metrics", {}), dict) else {}
    if require_unique_frame_station_contract:
        dup_collapsed = frame_metrics.get("duplicate_station_rows_collapsed")
        ok = dup_collapsed is not None and int(dup_collapsed) == 0
        add_check(
            "channel_frame_duplicate_station_rows_collapsed",
            ok,
            value=dup_collapsed,
            expected=0,
            reason="Channel frame still required duplicate station collapse",
        )

    stability_summary = _read_json(river_stability_summary_json) if river_stability_summary_json else None
    if require_trusted_interior_identity:
        all_ok = None
        if isinstance(stability_summary, dict):
            all_ok = stability_summary.get("all_trusted_interior_identity_ok")
        add_check(
            "trusted_interior_identity_ok",
            bool(all_ok) if all_ok is not None else False,
            value=all_ok,
            expected=True,
            reason="Trusted-interior invariance summary missing or not OK",
        )

    all_required_ok = all(c.get("ok") is not False for c in checks) and all(c.get("ok") is True or c.get("ok") is None for c in checks if c["name"] != "sdb_mode")
    failures = [c for c in checks if c.get("ok") is False]

    return {
        "report_json": str(Path(report_json)),
        "river_stability_summary_json": str(Path(river_stability_summary_json)) if river_stability_summary_json else None,
        "checks": checks,
        "all_required_checks_ok": len(failures) == 0,
        "failure_count": len(failures),
        "failures": failures,
        "summary": {
            "river_status": river_status,
            "sdb_status": sdb_status,
            "manifest_contract_valid": manifest_contract.get("all_manifest_contracts_valid"),
            "authoritative_lock_ok": auth_contract.get("ok"),
            "authoritative_locked_changed_count": auth_contract.get("locked_changed_count"),
            "xs_support_status": xs_receipt.get("status") if isinstance(xs_receipt, dict) else None,
            "profile_coverage_path": str(profile_coverage_path) if profile_coverage_path else None,
            "profile_coverage_summary": coverage_summary,
            "channel_frame_station_count": frame_metrics.get("station_count"),
            "channel_frame_duplicate_station_rows_collapsed": frame_metrics.get("duplicate_station_rows_collapsed"),
            "channel_frame_component_count": frame_metrics.get("component_count"),
            "trusted_interior_identity_ok": stability_summary.get("all_trusted_interior_identity_ok") if isinstance(stability_summary, dict) else None,
        },
    }


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate a pre-hydraulic river baseline run against required contracts.")
    ap.add_argument("--report-json", required=True, help="Path to bathy_report JSON or run_summary_bathy JSON.")
    ap.add_argument("--river-stability-summary-json", default=None)
    ap.add_argument("--expected-sdb-mode", default=None, help="Expected SDB mode for this AOI, e.g. inactive or active.")
    ap.add_argument("--max-profile-unsupported-fraction", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fail-on-regression", action="store_true")
    ap.add_argument("--allow-missing-profile-coverage", action="store_true")
    ap.add_argument("--allow-frame-duplicate-collapse", action="store_true")
    ap.add_argument("--allow-missing-xs-support-status", action="store_true")
    ap.add_argument("--allow-manifest-invalid", action="store_true")
    ap.add_argument("--allow-authoritative-lock-failure", action="store_true")
    ap.add_argument("--require-trusted-interior-identity", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ns = _parse_args(argv)
    payload = evaluate_pre_hydraulic_baseline(
        report_json=ns.report_json,
        river_stability_summary_json=ns.river_stability_summary_json,
        expected_sdb_mode=ns.expected_sdb_mode,
        require_profile_coverage=not ns.allow_missing_profile_coverage,
        require_unique_frame_station_contract=not ns.allow_frame_duplicate_collapse,
        require_xs_support_status_receipt=not ns.allow_missing_xs_support_status,
        require_manifest_valid=not ns.allow_manifest_invalid,
        require_authoritative_lock_ok=not ns.allow_authoritative_lock_failure,
        require_trusted_interior_identity=bool(ns.require_trusted_interior_identity),
        max_profile_unsupported_fraction=ns.max_profile_unsupported_fraction,
    )
    out = Path(ns.out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if ns.fail_on_regression and not payload.get("all_required_checks_ok", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
