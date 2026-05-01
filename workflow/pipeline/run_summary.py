from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify(v) for v in value]
    return value


def _read_json_or_none(path: Path | str | None) -> dict[str, Any] | None:
    if path in (None, ""):
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None



def _sha256_file(path: Path | str | None) -> str | None:
    if path in (None, "") or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_record(path: Path | str | None) -> dict[str, Any]:
    if path in (None, ""):
        return {"path": None, "exists": False, "sha256": None}
    candidate = Path(path)
    return {"path": str(candidate), "exists": candidate.is_file(), "sha256": _sha256_file(candidate)}


def _final_folder_products_from_receipt(receipt: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        return {}
    products: dict[str, Any] = {}
    for section in ("required_outputs", "optional_outputs"):
        values = receipt.get(section)
        if not isinstance(values, Mapping):
            continue
        for name, path_value in values.items():
            if path_value not in (None, ""):
                products[str(name)] = _path_record(path_value)
    return products


def _path_from_receipt_record(value: Any) -> str | None:
    """Return a file path from a manifest receipt reference.

    Canonical manifests may store stage receipts either as raw path strings or
    as audit records such as {\"path\": \"...\", \"exists\": true, \"sha256\": \"...\"}.
    Reporting must read those records without treating dicts as Path objects.
    """
    if value in (None, ""):
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, Mapping):
        for key in ("path", "receipt_path", "file", "filename"):
            candidate = value.get(key)
            if candidate not in (None, ""):
                return str(candidate)
    return None


def _canonical_stage_receipt_paths_from_manifest(
    receipts: Mapping[str, Path | str | None],
) -> dict[str, Path | str]:
    """Return retained canonical-parent stage receipts referenced by the manifest.

    AOI export-only runs should not rerun canonical construction, but their
    summaries should still be able to read the immutable parent science
    receipts when the retained canonical manifest references them.
    """
    manifest = _read_json_or_none(receipts.get("canonical_manifest"))
    if not isinstance(manifest, Mapping):
        return {}
    stage_receipts = manifest.get("canonical_construction_stage_receipts")
    if not isinstance(stage_receipts, Mapping):
        return {}
    out: dict[str, Path | str] = {}
    for key, record in stage_receipts.items():
        path_value = _path_from_receipt_record(record)
        if path_value in (None, ""):
            continue
        if Path(path_value).is_file():
            out[str(key)] = path_value
    return out

def _composition_counts_from_manifest(
    receipts: Mapping[str, Path | str | None],
) -> dict[str, Any]:
    """Return finite/support/guidance/background counts from the parent manifest."""
    manifest = _read_json_or_none(receipts.get("canonical_manifest"))
    if not isinstance(manifest, Mapping):
        return {}
    composition = manifest.get("composition_summary")
    if not isinstance(composition, Mapping):
        return {}
    keep = {
        "support_applied_pixel_count",
        "guidance_applied_pixel_count",
        "background_applied_pixel_count",
        "final_finite_count",
    }
    out: dict[str, Any] = {}
    for key, value in composition.items():
        if str(key) not in keep or value in (None, ""):
            continue
        out[str(key)] = int(value) if isinstance(value, (int, float)) and float(value).is_integer() else value
    return out


def _path_exists(path: Path | str | None) -> bool:
    return path not in (None, "") and Path(path).exists()



def _receipt_completeness(
    *,
    out_dir: Path,
    receipts: Mapping[str, Path | str | None],
    products: Mapping[str, Path | str | None],
) -> dict[str, Any]:
    """Report the Phase 1 deliverables without rediscovering workflow routes.

    The active river route should always leave enough evidence to debug the
    parent/export/final chain.  This helper is intentionally read-only: it
    checks the explicit paths already registered by the workflow and records a
    clear status for each required receipt/product.
    """
    expected_trace = out_dir / "run_logs" / "workflow_actual_trace.json"
    required: dict[str, Path | str | None] = {
        "canonical_parent_dem": products.get("canonical_parent_dem"),
        "aoi_export_dem": products.get("aoi_export_dem"),
        "final_user_dem": products.get("final_user_dem"),
        "canonical_manifest": receipts.get("canonical_manifest"),
        "aoi_export_identity": receipts.get("aoi_identity"),
        "river_workflow_receipt": receipts.get("river_workflow_receipt"),
        "final_output_receipt": receipts.get("final_output_receipt"),
    }
    expected_current_write: dict[str, Path | str | None] = {
        "run_summary_json": out_dir / "reports" / "run_summary.json",
        "run_summary_text": out_dir / "reports" / "run_summary.txt",
    }
    expected_late: dict[str, Path | str | None] = {
        "workflow_actual_trace": expected_trace,
    }

    def record(
        path_value: Path | str | None,
        *,
        expected_current_summary_write: bool = False,
        expected_after_summary: bool = False,
    ) -> dict[str, Any]:
        if path_value in (None, ""):
            return {
                "path": None,
                "exists": False,
                "status": "missing_path",
                "expected_after_summary_write": expected_after_summary,
            }
        candidate = Path(path_value)
        exists = candidate.is_file()
        if exists:
            status = "present"
        elif expected_current_summary_write:
            status = "expected_current_summary_write"
        elif expected_after_summary:
            status = "expected_after_summary_write"
        else:
            status = "missing_file"
        return {
            "path": str(candidate),
            "exists": exists,
            "status": status,
            "expected_current_summary_write": expected_current_summary_write,
            "expected_after_summary_write": expected_after_summary,
        }

    required_records = {name: record(path) for name, path in required.items()}
    current_records = {name: record(path, expected_current_summary_write=True) for name, path in expected_current_write.items()}
    late_records = {name: record(path, expected_after_summary=True) for name, path in expected_late.items()}
    missing_required = [name for name, item in required_records.items() if item.get("status") != "present"]
    return {
        "contract": "river_phase1_reporting_complete_v1",
        "required_at_summary_write": required_records,
        "expected_current_summary_write": current_records,
        "expected_after_summary_write": late_records,
        "missing_required_at_summary_write": missing_required,
        "status": "pass" if not missing_required else "warn",
    }

def _status_from_checks(checks: Mapping[str, Any]) -> tuple[str, list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    required = {
        "final_source_is_aoi_export": checks.get("final_source_is_aoi_export"),
        "export_vs_parent_identity_passed": checks.get("export_vs_parent_identity_passed"),
        "single_writer_passed": checks.get("single_writer_passed"),
    }
    for name, ok in required.items():
        if ok is False:
            errors.append(name)
        elif ok is None:
            warnings.append(f"{name}_not_checked")
    if errors:
        return "failed", warnings, errors
    if warnings:
        return "partial", warnings, errors
    return "passed", warnings, errors


def _science_from_receipts(receipts: Mapping[str, Path | str | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    mapping = {
        "wse_proxy": "centerline_wse_proxy",
        "observed_offset": "centerline_observed_offset",
        "modeled_offset": "centerline_modeled_offset",
        "backbone": "centerline_bed_backbone",
    }
    manifest = _read_json_or_none(receipts.get("canonical_manifest"))
    if isinstance(manifest, Mapping):
        canonical_science = manifest.get("canonical_science_summary")
        if isinstance(canonical_science, Mapping):
            for key, value in canonical_science.items():
                if isinstance(value, Mapping):
                    out[str(key)] = dict(value)
    canonical_stage_receipts = _canonical_stage_receipt_paths_from_manifest(receipts)
    for label, receipt_key in mapping.items():
        receipt_path = receipts.get(receipt_key) or canonical_stage_receipts.get(receipt_key)
        payload = _read_json_or_none(receipt_path)
        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        science = summary.get("river_science") if isinstance(summary.get("river_science"), dict) else None
        # The WSE stage writes its compact science evidence under ``science``
        # directly. Read it here, but do not invent missing metrics.
        if science is None and isinstance(payload.get("science"), dict):
            science = payload.get("science")
        if science is not None:
            out[label] = science
    return out


def _support_counts_from_receipts(
    receipts: Mapping[str, Path | str | None],
    workflow_receipt_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Collect already-reported support/composition counts without scanning data."""
    collected: dict[str, Any] = {}
    if isinstance(workflow_receipt_payload, Mapping):
        workflow_counts = workflow_receipt_payload.get("support_class_counts")
        if isinstance(workflow_counts, Mapping) and workflow_counts:
            collected["river_workflow_receipt"] = dict(workflow_counts)

    receipt_sources: dict[str, Path | str | None] = dict(receipts)
    for key, path_value in _canonical_stage_receipt_paths_from_manifest(receipts).items():
        receipt_sources.setdefault(f"canonical_parent:{key}", path_value)

    manifest_counts = _composition_counts_from_manifest(receipts)
    if manifest_counts:
        collected["canonical_manifest:composition_summary"] = manifest_counts

    for key, receipt_path in receipt_sources.items():
        payload = _read_json_or_none(receipt_path)
        if not isinstance(payload, dict):
            continue
        candidates: list[tuple[str, Any]] = [
            ("top_level", payload.get("support_class_counts")),
        ]
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        candidates.append(("summary", summary.get("support_class_counts")))
        science = summary.get("river_science") if isinstance(summary.get("river_science"), dict) else None
        if science is None and isinstance(payload.get("science"), dict):
            science = payload.get("science")
        if isinstance(science, dict):
            candidates.append(("science", science.get("support_class_counts")))
        for source, counts in candidates:
            if isinstance(counts, Mapping) and counts:
                collected[f"{key}:{source}"] = {
                    str(k): int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v
                    for k, v in counts.items()
                }
                break
    return collected


def _metric_status(value: Any, *, pass_when_zero: bool = True) -> str:
    if value in (None, ""):
        return "skipped"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "skipped"
    if pass_when_zero:
        return "pass" if numeric == 0 else "warn"
    return "pass"


def _backbone_rise_validation(backbone: Mapping[str, Any]) -> dict[str, Any]:
    """Validate backbone downstream rises using the large-rise metric only.

    Small positive local steps are retained as diagnostic counts because the
    bed backbone is a guidance profile, not a hard monotone reconstruction.
    The pass/fail criterion is whether any downstream rise exceeds the active
    per-step allowance recorded by the backbone stage receipt.
    """
    allowed = backbone.get("max_allowed_downstream_rise_per_step_m")
    large_count = backbone.get("large_downstream_rise_count_gt_allowed")
    small_count = backbone.get("downstream_trend_violation_count")
    status = "skipped" if large_count in (None, "") else _metric_status(large_count)
    return {
        "status": status,
        "status_basis": "large_downstream_rise_count_gt_allowed == 0",
        "large_downstream_rise_count_gt_allowed": large_count,
        "positive_downstream_step_count": backbone.get("positive_downstream_step_count", small_count),
        "downstream_trend_violation_count": small_count,
        "max_allowed_downstream_rise_per_step_m": allowed,
        "max_downstream_rise_m": backbone.get("max_downstream_rise_m"),
        "largest_step_m": backbone.get("largest_step_m"),
        "largest_step_note": "largest_step_m may include downstream-deepening drops; max_downstream_rise_m is the rise metric",
        "interpretation": "pass means no downstream rise exceeds the allowed local step; small positive local steps may remain as diagnostics",
    }


def _science_validation_from_existing_reports(
    river_science: Mapping[str, Any],
    support_class_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a read-only science/support summary from retained receipts.

    This is deliberately not a routing layer: it does not fix, select, or
    recompute products. Missing metrics are marked skipped with an explicit
    reason so export-only AOI runs remain easy to diagnose.
    """
    wse = river_science.get("wse_proxy") if isinstance(river_science.get("wse_proxy"), Mapping) else {}
    modeled = river_science.get("modeled_offset") if isinstance(river_science.get("modeled_offset"), Mapping) else {}
    observed = river_science.get("observed_offset") if isinstance(river_science.get("observed_offset"), Mapping) else {}
    backbone = river_science.get("backbone") if isinstance(river_science.get("backbone"), Mapping) else {}
    wse_violations = wse.get("monotone_violation_count")
    backbone_check = _backbone_rise_validation(backbone)
    support_present = bool(support_class_counts)

    wse_status = _metric_status(wse_violations)
    observed_status = "pass" if observed.get("valid_observed_offset_count") not in (None, "") else "skipped"
    modeled_status = "pass" if modeled.get("finite_modeled_offset_count") not in (None, "") else "skipped"
    backbone_status = str(backbone_check.get("status", "skipped"))
    support_status = "pass" if support_present else "skipped"
    stage_missing_reason = "stage_science_receipts_not_retained_or_not_available_in_this_aoi_export"
    support_missing_reason = "support_or_composition_counts_not_found_in_retained_receipts"

    checks: dict[str, Any] = {
        "wse_monotone_downstream": {
            "status": wse_status,
            "monotone_violation_count": wse_violations,
            "max_monotone_reversal_m": wse.get("max_monotone_reversal_m"),
        },
        "observed_offset_support": {
            "status": observed_status,
            "valid_observed_offset_count": observed.get("valid_observed_offset_count"),
            "support_class_counts": observed.get("support_class_counts"),
        },
        "modeled_offset_support": {
            "status": modeled_status,
            "finite_modeled_offset_count": modeled.get("finite_modeled_offset_count"),
            "support_mode": modeled.get("support_mode"),
            "support_class_counts": modeled.get("support_class_counts"),
            "offset_source_counts": modeled.get("offset_source_counts"),
        },
        "backbone_downstream_rise": backbone_check,
        "support_class_counts_reported": {
            "status": support_status,
            "source_count": len(support_class_counts),
            "sources": sorted(str(k) for k in support_class_counts.keys()),
        },
    }

    for name in (
        "wse_monotone_downstream",
        "observed_offset_support",
        "modeled_offset_support",
        "backbone_downstream_rise",
    ):
        if checks[name]["status"] == "skipped":
            checks[name]["reason"] = stage_missing_reason
    if checks["support_class_counts_reported"]["status"] == "skipped":
        checks["support_class_counts_reported"]["reason"] = support_missing_reason

    return {
        "mode": "read_only_existing_receipts",
        "alters_products": False,
        "support_class_counts_present": support_present,
        "checks": checks,
    }


def build_run_summary_payload(
    *,
    out_dir: Path,
    run_id: str | None,
    canonical_system_id: str | None,
    canonical_parent_dem: Path | str | None,
    aoi_export_dem: Path | str | None,
    final_user_dem: Path | str | None,
    receipts: Mapping[str, Path | str | None],
    checks: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the single seamless-DEM run summary payload.

    This summary is intentionally role-based. It does not rediscover products or
    infer hidden routes; it reports the explicit canonical parent -> AOI export
    -> final user DEM chain that the active workflow registered.
    """
    materialization_payload = _read_json_or_none(receipts.get("final_dem_materialization"))
    identity_payload = _read_json_or_none(receipts.get("aoi_identity"))
    parent_payload = _read_json_or_none(receipts.get("canonical_parent_dem"))
    final_output_payload = _read_json_or_none(receipts.get("final_output_receipt"))
    workflow_receipt_payload = _read_json_or_none(receipts.get("river_workflow_receipt"))
    canonical_manifest_payload = _read_json_or_none(receipts.get("canonical_manifest"))
    cache_payload = parent_payload.get("canonical_cache") if isinstance(parent_payload, dict) and isinstance(parent_payload.get("canonical_cache"), dict) else {}
    cache_validation_payload = parent_payload.get("cache_validation") if isinstance(parent_payload, dict) and isinstance(parent_payload.get("cache_validation"), dict) else {}
    final_source_role = checks.get("final_dem_source_role")
    if final_source_role in (None, "") and materialization_payload is not None:
        final_source_role = materialization_payload.get("source_role")
    final_source_is_aoi_export = checks.get("final_source_is_aoi_export")
    if final_source_is_aoi_export is None and final_source_role not in (None, ""):
        final_source_is_aoi_export = (final_source_role == "aoi_export_dem")
    export_vs_parent_identity_passed = checks.get("export_vs_parent_identity_passed")
    if export_vs_parent_identity_passed is None and identity_payload is not None:
        export_vs_parent_identity_passed = identity_payload.get("passed")
    single_writer_passed = checks.get("single_writer_passed")
    if single_writer_passed is None and materialization_payload is not None:
        single_writer_passed = (materialization_payload.get("writer_role") == "final_dem_materializer")
    cache_hit = checks.get("cache_hit")
    cache_validation_passed = checks.get("cache_validation_passed")
    if cache_validation_passed is None and cache_validation_payload:
        cache_validation_passed = cache_validation_payload.get("passed")
    cache_validation_status = checks.get("cache_validation_status") or cache_validation_payload.get("status")
    effective_checks = dict(checks)
    effective_checks["final_dem_source_role"] = final_source_role
    effective_checks["final_source_is_aoi_export"] = final_source_is_aoi_export
    effective_checks["export_vs_parent_identity_passed"] = export_vs_parent_identity_passed
    effective_checks["single_writer_passed"] = single_writer_passed
    status, warnings, errors = _status_from_checks(effective_checks)
    river_science = _science_from_receipts(receipts)
    support_class_counts = _support_counts_from_receipts(receipts, workflow_receipt_payload)
    science_validation = _science_validation_from_existing_reports(river_science, support_class_counts)
    products = {
        "canonical_parent_dem": canonical_parent_dem,
        "aoi_export_dem": aoi_export_dem,
        "final_user_dem": final_user_dem,
    }
    reporting_completeness = _receipt_completeness(out_dir=out_dir, receipts=receipts, products=products)
    payload = {
        "summary_type": "seamless_dem_run_summary",
        "workflow": "river_workflow",
        "route": ["canonical_parent_dem", "aoi_export_dem", "final_user_dem"],
        "phase1_reporting_completeness": reporting_completeness,
        "status": status,
        "run_id": run_id,
        "out_dir": str(out_dir),
        "canonical_system_id": canonical_system_id,
        "products": {
            "canonical_parent_dem": _path_record(canonical_parent_dem),
            "aoi_export_dem": _path_record(aoi_export_dem),
            "final_user_dem": _path_record(final_user_dem),
        },
        "final_dem_source_role": final_source_role,
        "cache": {
            "cache_hit": bool(cache_hit) if cache_hit is not None else None,
            "cache_validation_passed": bool(cache_validation_passed) if cache_validation_passed is not None else None,
            "cache_validation_status": cache_validation_status,
            "canonical_cache_key": checks.get("canonical_cache_key") or cache_payload.get("canonical_cache_key") or (canonical_manifest_payload or {}).get("canonical_cache_key") or (canonical_manifest_payload or {}).get("canonical_solve_cache_key"),
            "workflow_contract_version": cache_payload.get("workflow_contract_version"),
        },
        "checks": {
            "final_source_is_aoi_export": final_source_is_aoi_export,
            "export_vs_parent_identity_passed": export_vs_parent_identity_passed,
            "max_abs_diff": checks.get("max_abs_diff"),
            "mismatch_pixels": checks.get("mismatch_pixels"),
            "single_writer_passed": single_writer_passed,
            "authoritative_lock_passed": checks.get("authoritative_lock_passed"),
        },
        "receipts": {str(k): (str(v) if v not in (None, "") else None) for k, v in receipts.items()},
        "canonical_manifest": {
            "path": str(receipts.get("canonical_manifest")) if receipts.get("canonical_manifest") not in (None, "") else None,
            "exists": _path_exists(receipts.get("canonical_manifest")),
            "canonical_system_id": (canonical_manifest_payload or {}).get("canonical_system_id"),
            "canonical_cache_key": (canonical_manifest_payload or {}).get("canonical_cache_key") or (canonical_manifest_payload or {}).get("canonical_solve_cache_key"),
        },
        "final_folder_products": _final_folder_products_from_receipt(final_output_payload),
        "river_workflow_receipt_status": {
            "path": str(receipts.get("river_workflow_receipt")) if receipts.get("river_workflow_receipt") not in (None, "") else None,
            "exists": _path_exists(receipts.get("river_workflow_receipt")),
            "combined_vs_export_exact": ((workflow_receipt_payload or {}).get("identity") or {}).get("combined_vs_export_exact") if isinstance((workflow_receipt_payload or {}).get("identity"), Mapping) else None,
            "single_writer_pass": ((workflow_receipt_payload or {}).get("identity") or {}).get("single_writer_pass") if isinstance((workflow_receipt_payload or {}).get("identity"), Mapping) else None,
        },
        "river_science": river_science,
        "support_class_counts": support_class_counts,
        "science_validation": science_validation,
        "warnings": warnings + [str(w) for w in checks.get("warnings", [])],
        "errors": errors + [str(e) for e in checks.get("errors", [])],
    }
    if identity_payload is not None:
        payload["checks"]["identity_receipt_passed"] = identity_payload.get("passed")
        payload["checks"]["overlap_pixels"] = identity_payload.get("overlap_pixels")
    if materialization_payload is not None:
        payload["materialization"] = {
            "writer_role": materialization_payload.get("writer_role"),
            "source_role": materialization_payload.get("source_role"),
            "destination_role": materialization_payload.get("destination_role"),
            "mode": materialization_payload.get("mode"),
            "pixel_values_modified": materialization_payload.get("pixel_values_modified"),
            "resampled": materialization_payload.get("resampled"),
            "reprojected": materialization_payload.get("reprojected"),
        }
    return _stringify(payload)


def _render_text(payload: Mapping[str, Any]) -> str:
    products = payload.get("products", {}) if isinstance(payload.get("products"), dict) else {}
    checks = payload.get("checks", {}) if isinstance(payload.get("checks"), dict) else {}
    cache = payload.get("cache", {}) if isinstance(payload.get("cache"), dict) else {}
    lines = [
        "SEAMLESS DEM RUN SUMMARY",
        f"status: {payload.get('status')}",
        f"canonical_system_id: {payload.get('canonical_system_id')}",
        "route: canonical_parent_dem -> aoi_export_dem -> final_user_dem",
        "",
        "PRODUCTS",
    ]
    for role in ("canonical_parent_dem", "aoi_export_dem", "final_user_dem"):
        entry = products.get(role, {}) if isinstance(products, dict) else {}
        lines.append(f"- {role}: {entry.get('path')} exists={entry.get('exists')} sha256={entry.get('sha256')}")
    lines.extend([
        "",
        "CHECKS",
        f"- final_source_is_aoi_export: {checks.get('final_source_is_aoi_export')}",
        f"- export_vs_parent_identity_passed: {checks.get('export_vs_parent_identity_passed')} max_abs_diff={checks.get('max_abs_diff')} mismatch_pixels={checks.get('mismatch_pixels')}",
        f"- single_writer_passed: {checks.get('single_writer_passed')}",
        f"- authoritative_lock_passed: {checks.get('authoritative_lock_passed')}",
        "",
        "CACHE",
        f"- cache_hit: {cache.get('cache_hit')}",
        f"- cache_validation_passed: {cache.get('cache_validation_passed')}",
        f"- cache_validation_status: {cache.get('cache_validation_status')}",
        f"- canonical_cache_key: {cache.get('canonical_cache_key')}",
    ])
    manifest = payload.get("canonical_manifest") if isinstance(payload.get("canonical_manifest"), dict) else {}
    final_products = payload.get("final_folder_products") if isinstance(payload.get("final_folder_products"), dict) else {}
    if manifest:
        lines.extend([
            "",
            "CANONICAL MANIFEST",
            f"- path: {manifest.get('path')} exists={manifest.get('exists')}",
            f"- canonical_system_id: {manifest.get('canonical_system_id')}",
            f"- canonical_cache_key: {manifest.get('canonical_cache_key')}",
        ])
    if final_products:
        lines.extend(["", "FINAL FOLDER PRODUCTS"])
        for name, entry in sorted(final_products.items()):
            if isinstance(entry, dict):
                lines.append(f"- {name}: {entry.get('path')} exists={entry.get('exists')} sha256={entry.get('sha256')}")
    completeness = payload.get("phase1_reporting_completeness") if isinstance(payload.get("phase1_reporting_completeness"), dict) else {}
    if completeness:
        lines.extend(["", "PHASE 1 REPORTING COMPLETENESS", f"- status: {completeness.get('status')}"])
        required = completeness.get("required_at_summary_write") if isinstance(completeness.get("required_at_summary_write"), dict) else {}
        for name, entry in sorted(required.items()):
            if isinstance(entry, dict):
                lines.append(f"- {name}: {entry.get('status')} path={entry.get('path')}")
        current = completeness.get("expected_current_summary_write") if isinstance(completeness.get("expected_current_summary_write"), dict) else {}
        if current:
            lines.append("- expected during current summary write:")
            for name, entry in sorted(current.items()):
                if isinstance(entry, dict):
                    lines.append(f"  - {name}: {entry.get('status')} path={entry.get('path')}")
        late = completeness.get("expected_after_summary_write") if isinstance(completeness.get("expected_after_summary_write"), dict) else {}
        if late:
            lines.append("- expected after summary write:")
            for name, entry in sorted(late.items()):
                if isinstance(entry, dict):
                    lines.append(f"  - {name}: {entry.get('status')} path={entry.get('path')}")
    river_science = payload.get("river_science") if isinstance(payload.get("river_science"), dict) else {}
    support_counts = payload.get("support_class_counts") if isinstance(payload.get("support_class_counts"), dict) else {}
    science_validation = payload.get("science_validation") if isinstance(payload.get("science_validation"), dict) else {}
    if river_science or support_counts or science_validation:
        lines.extend(["", "READ-ONLY SCIENCE/SUPPORT CHECKS"])
        checks_payload = science_validation.get("checks") if isinstance(science_validation.get("checks"), dict) else {}
        for name in ("wse_monotone_downstream", "observed_offset_support", "modeled_offset_support", "backbone_downstream_rise", "support_class_counts_reported"):
            item = checks_payload.get(name) if isinstance(checks_payload.get(name), dict) else {}
            if item:
                detail = {k: v for k, v in item.items() if k != "status" and v not in (None, "", {})}
                lines.append(f"- {name}: {item.get('status')} {detail}")
        if support_counts:
            lines.append("- support_class_count_sources:")
            for label, counts in sorted(support_counts.items()):
                lines.append(f"  - {label}: {counts}")
    warnings = payload.get("warnings") or []
    errors = payload.get("errors") or []
    if warnings:
        lines.append("")
        lines.append("WARNINGS")
        lines.extend(f"- {w}" for w in warnings)
    if errors:
        lines.append("")
        lines.append("ERRORS")
        lines.extend(f"- {e}" for e in errors)
    return "\n".join(lines) + "\n"


def write_run_summary(
    *,
    out_dir: Path | str,
    run_id: str | None,
    canonical_system_id: str | None,
    canonical_parent_dem: Path | str | None,
    aoi_export_dem: Path | str | None,
    final_user_dem: Path | str | None,
    receipts: Mapping[str, Path | str | None],
    checks: Mapping[str, Any],
) -> Path:
    out_path = Path(out_dir)
    reports_dir = out_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = build_run_summary_payload(
        out_dir=out_path,
        run_id=run_id,
        canonical_system_id=canonical_system_id,
        canonical_parent_dem=canonical_parent_dem,
        aoi_export_dem=aoi_export_dem,
        final_user_dem=final_user_dem,
        receipts=receipts,
        checks=checks,
    )
    json_path = reports_dir / "run_summary.json"
    txt_path = reports_dir / "run_summary.txt"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    txt_path.write_text(_render_text(payload), encoding="utf-8")
    return json_path


__all__ = ["build_run_summary_payload", "write_run_summary"]
