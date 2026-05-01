from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from active_context import (
    ActiveRiverFinalizeTargets,
    ActiveRiverWorkflowResult,
    ActiveStageResult,
    ActiveWorkflowContext,
    ActiveWorkflowFinalizeResult,
)


from pipeline.final_dem_materialization import write_final_dem_from_aoi_export
from pipeline.run_summary import write_run_summary
from pipeline.aoi_identity import compare_export_to_parent
from pipeline.river_workflow.river_workflow_contract import (
    AOI_EXPORT_DEM_ROLE,
    assert_aoi_export_identity_passed,
    assert_single_final_dem_writer,
)
from pipeline.river_workflow.river_workflow_receipts import write_aoi_identity_receipt
from pipeline.river_workflow.river_workflow_outputs import final_user_dem_path
from pipeline.river_workflow.river_workflow_stage_artifact_contract import (
    validate_stage_artifact_contract,
    write_stage_artifact_contract,
)
from river_workflow_final_route import write_final_route_guard_receipt


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def _read_json_file(path: Path | str | None) -> dict[str, Any]:
    if path in (None, ""):
        return {}
    try:
        candidate = Path(path)
        if not candidate.is_file():
            return {}
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _path_record(path: Path | str | None) -> dict[str, Any]:
    if path in (None, ""):
        return {"path": None, "exists": False, "sha256": None}
    candidate = Path(path)
    return {"path": str(candidate), "exists": candidate.is_file(), "sha256": _sha256_file(candidate)}


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _collect_support_class_counts(report: dict[str, Any], river_result: object | None) -> dict[str, Any]:
    """Collect already-produced support/composition summaries without inventing new science."""
    collected: dict[str, Any] = {}

    def maybe_collect(label: str, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        for key in ("support_class_counts", "support_counts", "class_counts", "support_summary"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                collected.setdefault(label, value)

    def maybe_collect_manifest_composition() -> None:
        """Read retained canonical manifest composition counts for export-only AOIs."""
        manifest_path: Any = None
        for section_name in ("outputs", "river_workflow", "active_river"):
            section = report.get(section_name) if isinstance(report, dict) else None
            if isinstance(section, dict) and section.get("canonical_river_solution_manifest") not in (None, ""):
                manifest_path = section.get("canonical_river_solution_manifest")
                break
        manifest = _read_json_file(manifest_path)
        if not isinstance(manifest, dict):
            return
        composition = manifest.get("composition_summary")
        if not isinstance(composition, dict):
            return
        keep = {
            "support_applied_pixel_count",
            "guidance_applied_pixel_count",
            "background_applied_pixel_count",
            "final_finite_count",
        }
        counts: dict[str, Any] = {}
        for key, value in composition.items():
            if str(key) not in keep or value in (None, ""):
                continue
            counts[str(key)] = int(value) if isinstance(value, (int, float)) and float(value).is_integer() else value
        if counts:
            collected.setdefault("canonical_manifest:composition_summary", counts)

    maybe_collect("report", report)
    for section_name in ("river_workflow", "active_river", "river", "outputs"):
        section = report.get(section_name) if isinstance(report, dict) else None
        maybe_collect(section_name, section)

    stage_receipts = getattr(river_result, "stage_receipts", None) if river_result is not None else None
    if isinstance(stage_receipts, dict):
        for stage_name, receipt_path in stage_receipts.items():
            payload = _read_json_file(receipt_path)
            maybe_collect(str(stage_name), payload)

    maybe_collect_manifest_composition()
    return collected


def _update_retained_aoi_export_identity(
    ctx: ActiveWorkflowContext,
    *,
    river_result: object,
    aoi_export_dem: Path,
    final_user_dem: Path,
    identity_result: dict[str, Any],
) -> Path:
    reports_dir = Path(ctx.cfg.out_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    retained_path = reports_dir / "aoi_export_identity.json"
    final_dem_result = getattr(river_result, "final_dem_result", None)
    parent_dem = Path(getattr(final_dem_result, "canonical_parent_dem_path")) if final_dem_result is not None and getattr(final_dem_result, "canonical_parent_dem_path", None) not in (None, "") else None
    source_receipt = getattr(final_dem_result, "aoi_export_receipt_path", None) if final_dem_result is not None else None
    payload: dict[str, Any] = {}
    if source_receipt not in (None, "") and Path(source_receipt).is_file():
        try:
            payload = json.loads(Path(source_receipt).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            payload = {}
    payload.setdefault("schema_version", 1)
    payload.setdefault("stage", "aoi_export_identity")
    payload.setdefault("role", "aoi_export_identity")
    payload["canonical_cache_key"] = (
        payload.get("canonical_cache_key")
        or ctx.report.get("river_workflow", {}).get("canonical_cache_key")
        or ctx.report.get("active_river", {}).get("canonical_cache_key")
    )
    payload["canonical_system_id"] = (
        payload.get("canonical_system_id")
        or ctx.report.get("river_workflow", {}).get("canonical_system_id")
        or ctx.report.get("active_river", {}).get("canonical_system_id")
    )
    if parent_dem is not None:
        payload["parent_dem"] = {"path": str(parent_dem), "sha256": _sha256_file(parent_dem)}
        payload["parent_hash"] = payload["parent_dem"]["sha256"]
        payload["canonical_parent_content_key"] = payload.get("canonical_parent_content_key") or payload["parent_hash"]
        payload["canonical_comparison_key"] = payload.get("canonical_comparison_key") or payload["parent_hash"]
    payload["aoi_export_dem"] = {"path": str(aoi_export_dem), "sha256": _sha256_file(aoi_export_dem)}
    payload["export_hash"] = payload["aoi_export_dem"]["sha256"]
    payload["combined_dem"] = {"path": str(final_user_dem), "sha256": _sha256_file(final_user_dem)}
    payload["combined_hash"] = payload["combined_dem"]["sha256"]
    payload["export_vs_parent"] = "PASS" if identity_result.get("passed") else "FAIL"
    payload["combined_vs_export"] = "PASS" if payload["combined_hash"] == payload["export_hash"] else "FAIL"
    payload["max_abs_diff"] = identity_result.get("max_abs_diff")
    payload["mismatch_pixels"] = identity_result.get("mismatch_pixels")
    payload["construction_stages_run_in_aoi_export"] = False if payload.get("execution_role") == "aoi_export_only" else payload.get("construction_stages_run_in_aoi_export")
    retained_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ctx.report.setdefault("outputs", {})["aoi_export_identity_report"] = str(retained_path)
    ctx.report.setdefault("river_workflow", {})["aoi_export_identity_report"] = str(retained_path)
    ctx.report.setdefault("active_river", {})["aoi_export_identity_report"] = str(retained_path)
    return retained_path


def _write_river_architecture_summary(
    ctx: ActiveWorkflowContext,
    *,
    river_result: object | None,
    final_user_dem: Path | None,
    identity_result: dict[str, Any] | None = None,
) -> Path:
    reports_dir = Path(ctx.cfg.out_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "RIVER_ARCHITECTURE_SUMMARY.txt"
    final_dem_result = getattr(river_result, "final_dem_result", None) if river_result is not None else None
    execution_role = getattr(final_dem_result, "final_writer_mode", None) if final_dem_result is not None else None
    report_workflow = ctx.report.get("river_workflow", {}) if isinstance(ctx.report.get("river_workflow"), dict) else {}
    if execution_role == "existing_canonical_parent_exact_aoi_export":
        solution_source = str(report_workflow.get("canonical_solution_source") or "existing_parent")
        aoi_mode = str(report_workflow.get("aoi_execution_mode") or "export_only")
        construction_run = "NO"
        architecture_lock = "PASS"
    else:
        solution_source = str(report_workflow.get("canonical_solution_source") or "built_this_run")
        aoi_mode = str(report_workflow.get("aoi_execution_mode") or "canonical_build_then_export")
        construction_run = "YES"
        architecture_lock = "PASS" if aoi_mode == "canonical_build_then_export" else "REVIEW"
    parent = getattr(final_dem_result, "canonical_parent_dem_path", None) if final_dem_result is not None else None
    export = getattr(final_dem_result, "aoi_export_dem_path", None) if final_dem_result is not None else None
    cache_key = ctx.report.get("river_workflow", {}).get("canonical_cache_key") or ctx.report.get("active_river", {}).get("canonical_cache_key")
    system_id = ctx.report.get("river_workflow", {}).get("canonical_system_id") or ctx.report.get("active_river", {}).get("canonical_system_id")
    manifest_path = Path(ctx.cfg.out_dir) / "reports" / "canonical_river_solution_manifest.json"
    export_identity_path = Path(ctx.cfg.out_dir) / "reports" / "aoi_export_identity.json"
    # Keep retained canonical manifest visible to later human summaries and
    # comparison tooling. This is reporting metadata only; it does not alter
    # construction or export behavior.
    if manifest_path.exists():
        ctx.report.setdefault("outputs", {})["canonical_river_solution_manifest"] = str(manifest_path)
        ctx.report.setdefault("river_workflow", {})["canonical_river_solution_manifest"] = str(manifest_path)
        ctx.report.setdefault("active_river", {})["canonical_river_solution_manifest"] = str(manifest_path)
    lines = [
        "River architecture summary",
        "==========================",
        "",
        "Architecture: canonical_parent_plus_aoi_export",
        f"Architecture lock: {architecture_lock}",
        f"Canonical system id: {system_id or 'unknown'}",
        f"Canonical cache key: {cache_key or 'unknown'}",
        f"Canonical solution source: {solution_source}",
        f"AOI execution mode: {aoi_mode}",
        f"Construction stages run in AOI export: {construction_run}",
        f"Canonical manifest: {manifest_path}",
        f"Canonical manifest retained: {'YES' if manifest_path.exists() else 'NO'}",
        f"AOI export identity: {export_identity_path}",
        f"AOI export identity retained: {'YES' if export_identity_path.exists() else 'NO'}",
        "",
        "Invariant checks:",
        f"- AOI exports may recompute canonical construction: {'NO' if construction_run == 'NO' else 'ONLY DURING EXPLICIT CANONICAL BUILD'}",
        "- Cache is implementation detail: YES",
        "- Final DEM must derive from AOI export: YES",
        "",
        f"Canonical parent DEM: {parent or 'unknown'}",
        f"Canonical parent hash: {_sha256_file(Path(parent)) if parent not in (None, '') else 'unknown'}",
        f"AOI export DEM: {export or 'unknown'}",
        f"AOI export hash: {_sha256_file(Path(export)) if export not in (None, '') else 'unknown'}",
        f"Final DEM: {final_user_dem or 'unknown'}",
        f"Final DEM hash: {_sha256_file(final_user_dem) if final_user_dem is not None else 'unknown'}",
        "",
        f"Export vs parent: {'PASS' if (identity_result or {}).get('passed') else 'UNKNOWN'}",
        f"Max abs diff: {(identity_result or {}).get('max_abs_diff')}",
        f"Mismatch pixels: {(identity_result or {}).get('mismatch_pixels')}",
        "Single final writer: PASS",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ctx.report.setdefault("outputs", {})["river_architecture_summary"] = str(out_path)
    ctx.report.setdefault("river_workflow", {})["architecture_summary"] = str(out_path)
    ctx.report.setdefault("active_river", {})["architecture_summary"] = str(out_path)
    return out_path



def _write_consolidated_river_workflow_receipt(
    ctx: ActiveWorkflowContext,
    *,
    river_result: object | None,
    final_user_dem: Path,
    identity_result: dict[str, Any],
    final_output_receipt: Path | None,
    final_folder_products: dict[str, Path],
    stage_results: list[ActiveStageResult],
) -> tuple[Path, Path]:
    """Write the single user-facing river workflow receipt and text summary."""
    out_dir = Path(ctx.cfg.out_dir)
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = final_dir / "river_workflow_receipt.json"
    summary_path = final_dir / "river_workflow_summary.txt"

    final_dem_result = getattr(river_result, "final_dem_result", None) if river_result is not None else None
    outputs = ctx.report.get("outputs", {}) if isinstance(ctx.report.get("outputs"), dict) else {}
    river_workflow = ctx.report.get("river_workflow", {}) if isinstance(ctx.report.get("river_workflow"), dict) else {}
    active_river = ctx.report.get("active_river", {}) if isinstance(ctx.report.get("active_river"), dict) else {}

    canonical_parent_dem = _first_nonempty(
        getattr(final_dem_result, "canonical_parent_dem_path", None) if final_dem_result is not None else None,
        outputs.get("canonical_parent_dem"),
        outputs.get("final_folder_canonical_parent_dem_source"),
    )
    aoi_export_dem = _first_nonempty(
        getattr(final_dem_result, "aoi_export_dem_path", None) if final_dem_result is not None else None,
        outputs.get("aoi_export_dem"),
        outputs.get("river_workflow_dem_enhanced"),
    )
    materialization_receipt = outputs.get("final_dem_materialization_receipt")
    materialization_payload = _read_json_file(materialization_receipt)
    retained_export_identity = _first_nonempty(outputs.get("aoi_export_identity_report"), out_dir / "reports" / "aoi_export_identity.json")
    retained_parent_identity = _first_nonempty(outputs.get("canonical_parent_identity_report"), out_dir / "reports" / "canonical_parent_identity.json")
    final_output_payload = _read_json_file(final_output_receipt)
    final_route_guard_receipt = outputs.get("river_final_route_guard_receipt") or final_dir / "river_final_route_guard_receipt.json"
    final_route_guard_payload = _read_json_file(final_route_guard_receipt)

    final_source_role = materialization_payload.get("source_role")
    single_writer_pass = materialization_payload.get("writer_role") == "final_dem_materializer"
    export_vs_parent_exact = bool(identity_result.get("passed"))
    post_subset_modifications = False
    combined_vs_export = _sha256_file(final_user_dem) == _sha256_file(Path(aoi_export_dem)) if aoi_export_dem not in (None, "") else None

    final_folder_records = {name: _path_record(path) for name, path in sorted(final_folder_products.items())}

    stage_statuses = [
        {
            "stage_name": stage.stage_name,
            "stage_class": stage.stage_class,
            "status": stage.status,
            "primary_artifact_role": stage.primary_artifact_role,
            "primary_output": str(stage.primary_output) if stage.primary_output is not None else None,
            "receipt_path": str(stage.receipt_path) if stage.receipt_path is not None else None,
        }
        for stage in stage_results
    ]

    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "river_workflow_final_receipt",
        "role": "consolidated_final_receipt",
        "run_id": ctx.run_id,
        "out_dir": str(out_dir),
        "workflow": {
            "architecture": "canonical_parent_plus_aoi_export",
            "canonical_system_id": _first_nonempty(river_workflow.get("canonical_system_id"), active_river.get("canonical_system_id")),
            "canonical_cache_key": _first_nonempty(river_workflow.get("canonical_cache_key"), active_river.get("canonical_cache_key")),
            "canonical_build_role": "canonical_build" if getattr(final_dem_result, "final_writer_mode", None) != "existing_canonical_parent_exact_aoi_export" else "existing_parent_reused",
            "aoi_export_role": "exact_parent_window_export",
            "final_dem_source": "aoi_export_dem",
        },
        "identity": {
            "canonical_parent_dem": _path_record(canonical_parent_dem),
            "aoi_export_dem": _path_record(aoi_export_dem),
            "final_dem": _path_record(final_user_dem),
            "canonical_parent_identity_receipt": _path_record(retained_parent_identity),
            "aoi_export_identity_receipt": _path_record(retained_export_identity),
            "final_output_receipt": _path_record(final_output_receipt),
            "final_route_guard_receipt": _path_record(final_route_guard_receipt),
            "export_window": identity_result.get("window") or identity_result.get("export_window"),
            "export_vs_parent_exact": export_vs_parent_exact,
            "max_abs_diff_m": identity_result.get("max_abs_diff"),
            "mismatch_pixels": identity_result.get("mismatch_pixels"),
            "combined_vs_export_exact": combined_vs_export,
            "single_writer_pass": single_writer_pass,
            "final_source_role": final_source_role,
            "post_subset_modifications": post_subset_modifications,
            "final_route_guard_pass": final_route_guard_payload.get("passed"),
            "final_route_guard_failures": final_route_guard_payload.get("failures", []),
        },
        "final_folder": {
            "path": str(final_dir),
            "products": final_folder_records,
            "final_output_contract_valid": final_output_payload.get("valid"),
            "missing_outputs": final_output_payload.get("missing_outputs", {}),
        },
        "support_class_counts": _collect_support_class_counts(ctx.report, river_result),
        "stage_statuses": stage_statuses,
        "warnings": [],
        "errors": [],
    }

    if not export_vs_parent_exact:
        payload["errors"].append("aoi_export_does_not_match_canonical_parent_window")
    if not single_writer_pass:
        payload["errors"].append("final_dem_single_writer_contract_failed")
    if combined_vs_export is False:
        payload["errors"].append("final_dem_hash_differs_from_aoi_export")
    if final_route_guard_payload.get("passed") is not True:
        payload["errors"].append("final_route_guard_failed_or_missing")
    if not payload["support_class_counts"]:
        payload["warnings"].append("support_class_counts_not_reported_by_current_stage_receipts")

    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_lines = [
        "River workflow summary",
        "======================",
        "",
        f"CANONICAL BUILD: {payload['workflow']['canonical_build_role']}",
        f"AOI EXPORT: {payload['workflow']['aoi_export_role']}",
        f"FINAL DEM SOURCE: {payload['workflow']['final_dem_source']}",
        f"EXPORT VS PARENT: {'PASS' if export_vs_parent_exact else 'FAIL'}",
        f"MAX ABS DIFF: {identity_result.get('max_abs_diff')}",
        f"MISMATCH PIXELS: {identity_result.get('mismatch_pixels')}",
        f"SINGLE WRITER: {'PASS' if single_writer_pass else 'FAIL'}",
        f"COMBINED VS EXPORT: {'PASS' if combined_vs_export else 'FAIL' if combined_vs_export is False else 'UNKNOWN'}",
        f"POST-SUBSET MODIFICATIONS: {post_subset_modifications}",
        f"FINAL ROUTE GUARD: {'PASS' if final_route_guard_payload.get('passed') is True else 'FAIL'}",
        "",
        f"Canonical parent DEM: {_path_record(canonical_parent_dem)['path']}",
        f"Canonical parent hash: {_path_record(canonical_parent_dem)['sha256']}",
        f"AOI export DEM: {_path_record(aoi_export_dem)['path']}",
        f"AOI export hash: {_path_record(aoi_export_dem)['sha256']}",
        f"Final DEM: {_path_record(final_user_dem)['path']}",
        f"Final DEM hash: {_path_record(final_user_dem)['sha256']}",
        "",
        f"Final output receipt: {_path_record(final_output_receipt)['path']}",
        f"Final route guard receipt: {_path_record(final_route_guard_receipt)['path']}",
        f"Consolidated receipt: {receipt_path}",
    ]
    if payload["support_class_counts"]:
        summary_lines.extend(["", "Support class counts:"])
        for label, counts in sorted(payload["support_class_counts"].items()):
            summary_lines.append(f"- {label}: {counts}")
    if payload["warnings"]:
        summary_lines.extend(["", "Warnings:"] + [f"- {w}" for w in payload["warnings"]])
    if payload["errors"]:
        summary_lines.extend(["", "Errors:"] + [f"- {e}" for e in payload["errors"]])
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    ctx.report.setdefault("outputs", {})["river_workflow_receipt"] = str(receipt_path)
    ctx.report.setdefault("outputs", {})["river_workflow_summary"] = str(summary_path)
    ctx.report.setdefault("river_workflow", {})["consolidated_final_receipt"] = str(receipt_path)
    ctx.report.setdefault("river_workflow", {})["consolidated_final_summary"] = str(summary_path)
    ctx.report.setdefault("active_river", {})["consolidated_final_receipt"] = str(receipt_path)
    ctx.report.setdefault("active_river", {})["consolidated_final_summary"] = str(summary_path)
    return receipt_path, summary_path


def _materialize_river_final_dem(ctx: ActiveWorkflowContext, aoi_export_dem: Path) -> tuple[Path, Path]:
    """Materialize the stable final user DEM from the active AOI export."""
    cfg = ctx.cfg
    final_path = final_user_dem_path(Path(cfg.out_dir))
    receipt_path = Path(cfg.out_dir) / "river_workflow" / "receipts" / "final_dem_materialization_receipt.json"
    materialized = write_final_dem_from_aoi_export(
        aoi_export_dem=Path(aoi_export_dem),
        final_dem_path=final_path,
        receipt_path=receipt_path,
    )
    ctx.report.setdefault("outputs", {})["final_dem_materialization_receipt"] = str(receipt_path)
    ctx.report.setdefault("river_workflow", {})["final_dem_materialization_receipt"] = str(receipt_path)
    ctx.report.setdefault("active_river", {})["final_dem_materialization_receipt"] = str(receipt_path)
    return materialized, receipt_path


def _validate_final_output_folder(out_dir: Path, receipt_path: Path | None) -> dict[str, Path]:
    """Validate final/ against the receipt-defined deliverables.

    Authoritative/base comparison files are optional when the final-output
    receipt records that no baseline source was available. This keeps the active
    validator aligned with the final-output receipt instead of hard-requiring
    optional comparison files.
    """
    final_dir = Path(out_dir) / "final"
    receipt_candidate = Path(receipt_path) if receipt_path is not None else final_dir / "final_output_receipt.json"
    receipt = _read_json_file(receipt_candidate)

    required: dict[str, Path] = {}
    receipt_required = receipt.get("required_outputs")
    if isinstance(receipt_required, dict) and receipt_required:
        for name, path_value in receipt_required.items():
            if path_value not in (None, ""):
                required[str(name)] = Path(path_value)
    else:
        required.update({
            "DEM_enhanced": final_dir / "DEM_enhanced.tif",
            "DEM_enhanced_hillshade": final_dir / "DEM_enhanced_hillshade.tif",
        })
        if (final_dir / "authoritative_base_aligned.tif").is_file() or not receipt.get("baseline_skipped_reason"):
            required["authoritative_base_aligned"] = final_dir / "authoritative_base_aligned.tif"
            required["authoritative_base_aligned_hillshade"] = final_dir / "authoritative_base_aligned_hillshade.tif"
        if (final_dir / "canonical_parent_dem.tif").is_file() or receipt.get("canonical_parent_dem_source"):
            required["canonical_parent_dem"] = final_dir / "canonical_parent_dem.tif"
            required["canonical_parent_dem_hillshade"] = final_dir / "canonical_parent_dem_hillshade.tif"

    required["final_output_receipt"] = receipt_candidate

    missing = {name: str(path) for name, path in required.items() if not Path(path).is_file()}
    if missing:
        raise RuntimeError(f"final_output_folder_incomplete:{missing}")
    return required

def _run_aoi_identity_and_writer_checks(
    ctx: ActiveWorkflowContext,
    *,
    river_result: object,
    aoi_export_dem: Path,
    materialization_receipt: Path,
) -> tuple[Path | None, dict[str, Any]]:
    """Verify AOI export identity and final DEM single-writer contract."""
    final_dem_result = getattr(river_result, "final_dem_result", None)
    parent_dem = getattr(final_dem_result, "canonical_parent_dem_path", None) if final_dem_result is not None else None
    if parent_dem in (None, ""):
        raise ValueError("aoi_identity_missing_canonical_parent_dem_path")
    export_receipt = getattr(final_dem_result, "aoi_export_receipt_path", None)
    identity_result = compare_export_to_parent(
        parent_dem=Path(parent_dem),
        export_dem=Path(aoi_export_dem),
        export_receipt=Path(export_receipt) if export_receipt not in (None, "") else None,
        tolerance=0.0,
    )
    canonical_system_id = ctx.report.get("river_workflow", {}).get("canonical_system_id") or ctx.report.get("active_river", {}).get("canonical_system_id")
    identity_receipt = Path(ctx.cfg.out_dir) / "river_workflow" / "receipts" / "aoi_identity_receipt.json"
    write_aoi_identity_receipt(
        identity_receipt,
        canonical_system_id=str(canonical_system_id) if canonical_system_id not in (None, "") else None,
        identity_result=identity_result,
    )
    assert_aoi_export_identity_passed(identity_result)
    materialization_payload = json.loads(Path(materialization_receipt).read_text(encoding="utf-8"))
    assert_single_final_dem_writer([materialization_payload])
    ctx.report.setdefault("outputs", {})["aoi_identity_receipt"] = str(identity_receipt)
    ctx.report.setdefault("river_workflow", {})["aoi_identity_receipt"] = str(identity_receipt)
    ctx.report.setdefault("river_workflow", {})["aoi_identity"] = identity_result
    ctx.report.setdefault("active_river", {})["aoi_identity_receipt"] = str(identity_receipt)
    ctx.report.setdefault("active_river", {})["aoi_identity"] = identity_result
    return identity_receipt, identity_result


def _write_seamless_run_summary(
    ctx: ActiveWorkflowContext,
    *,
    river_result: object | None,
    final_user_dem: Path | None,
) -> Path | None:
    """Write one consolidated summary for the parent/export/final route."""
    final_dem_result = getattr(river_result, "final_dem_result", None) if river_result is not None else None
    canonical_parent_dem = getattr(final_dem_result, "canonical_parent_dem_path", None) if final_dem_result is not None else ctx.report.get("outputs", {}).get("canonical_parent_dem")
    aoi_export_dem = getattr(final_dem_result, "aoi_export_dem_path", None) if final_dem_result is not None else ctx.report.get("outputs", {}).get("aoi_export_dem")
    canonical_system_id = (
        ctx.report.get("river_workflow", {}).get("canonical_system_id")
        or ctx.report.get("active_river", {}).get("canonical_system_id")
    )
    outputs = ctx.report.get("outputs", {}) if isinstance(ctx.report.get("outputs"), dict) else {}
    receipts = {
        "canonical_parent_dem": getattr(final_dem_result, "canonical_parent_receipt_path", None) if final_dem_result is not None else outputs.get("canonical_parent_dem_receipt"),
        "aoi_export_dem": getattr(final_dem_result, "aoi_export_receipt_path", None) if final_dem_result is not None else outputs.get("aoi_export_dem_receipt"),
        "final_dem_materialization": outputs.get("final_dem_materialization_receipt"),
        "aoi_identity": outputs.get("aoi_export_identity_report") or outputs.get("aoi_identity_receipt"),
        "stage_chain_summary": outputs.get("river_stage_chain_summary"),
        "canonical_manifest": outputs.get("canonical_river_solution_manifest") or Path(ctx.cfg.out_dir) / "reports" / "canonical_river_solution_manifest.json",
        "final_output_receipt": outputs.get("final_output_receipt") or Path(ctx.cfg.out_dir) / "final" / "final_output_receipt.json",
        "river_workflow_receipt": outputs.get("river_workflow_receipt") or Path(ctx.cfg.out_dir) / "final" / "river_workflow_receipt.json",
        "final_route_guard": outputs.get("river_final_route_guard_receipt") or Path(ctx.cfg.out_dir) / "final" / "river_final_route_guard_receipt.json",
    }
    stage_receipts = getattr(river_result, "stage_receipts", None) if river_result is not None else None
    if isinstance(stage_receipts, dict):
        for role in ("centerline_wse_proxy", "centerline_observed_offset", "centerline_modeled_offset", "centerline_bed_backbone"):
            if stage_receipts.get(role) is not None:
                receipts[role] = stage_receipts.get(role)
    identity = (
        ctx.report.get("river_workflow", {}).get("aoi_identity")
        if isinstance(ctx.report.get("river_workflow"), dict)
        else None
    )
    if not isinstance(identity, dict):
        identity = {}
    materialization_receipt = outputs.get("final_dem_materialization_receipt")
    final_source_role = None
    single_writer_passed = None
    if materialization_receipt not in (None, ""):
        try:
            payload = json.loads(Path(materialization_receipt).read_text(encoding="utf-8"))
            final_source_role = payload.get("source_role")
            single_writer_passed = payload.get("writer_role") == "final_dem_materializer"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            final_source_role = None
            single_writer_passed = False
    parent_receipt_path = getattr(final_dem_result, "canonical_parent_receipt_path", None) if final_dem_result is not None else outputs.get("canonical_parent_dem_receipt")
    parent_cache_validation = None
    parent_cache_key = None
    if parent_receipt_path not in (None, ""):
        try:
            parent_payload = json.loads(Path(parent_receipt_path).read_text(encoding="utf-8"))
            if isinstance(parent_payload.get("cache_validation"), dict):
                parent_cache_validation = parent_payload.get("cache_validation")
            if isinstance(parent_payload.get("canonical_cache"), dict):
                parent_cache_key = parent_payload["canonical_cache"].get("canonical_cache_key")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            parent_cache_validation = {"passed": False, "status": "parent_receipt_unreadable"}
    checks = {
        "final_dem_source_role": final_source_role,
        "final_source_is_aoi_export": final_source_role == "aoi_export_dem",
        "export_vs_parent_identity_passed": identity.get("passed"),
        "max_abs_diff": identity.get("max_abs_diff"),
        "mismatch_pixels": identity.get("mismatch_pixels"),
        "single_writer_passed": single_writer_passed,
        "final_route_guard_passed": ctx.report.get("river_workflow", {}).get("final_route_guard_passed") if isinstance(ctx.report.get("river_workflow"), dict) else None,
        "authoritative_lock_passed": True if canonical_parent_dem not in (None, "") else None,
        "cache_hit": ctx.report.get("river_workflow", {}).get("canonical_cache_hit") if isinstance(ctx.report.get("river_workflow"), dict) else None,
        "cache_validation_passed": (parent_cache_validation.get("passed") if isinstance(parent_cache_validation, dict) else None),
        "cache_validation_status": (parent_cache_validation.get("status") if isinstance(parent_cache_validation, dict) else None),
        "canonical_cache_key": ((ctx.report.get("river_workflow", {}).get("canonical_cache_key") if isinstance(ctx.report.get("river_workflow"), dict) else None) or parent_cache_key),
    }
    summary_path = write_run_summary(
        out_dir=Path(ctx.cfg.out_dir),
        run_id=ctx.run_id,
        canonical_system_id=str(canonical_system_id) if canonical_system_id not in (None, "") else None,
        canonical_parent_dem=canonical_parent_dem,
        aoi_export_dem=aoi_export_dem,
        final_user_dem=final_user_dem,
        receipts=receipts,
        checks=checks,
    )
    txt_path = summary_path.with_suffix(".txt")
    ctx.report.setdefault("outputs", {})["run_summary"] = str(summary_path)
    ctx.report.setdefault("outputs", {})["run_summary_text"] = str(txt_path)
    ctx.report.setdefault("river_workflow", {})["run_summary"] = str(summary_path)
    ctx.report.setdefault("active_river", {})["run_summary"] = str(summary_path)
    ctx.report["seamless_dem"] = {
        "canonical_system_id": str(canonical_system_id) if canonical_system_id not in (None, "") else None,
        "canonical_parent_dem": str(canonical_parent_dem) if canonical_parent_dem not in (None, "") else None,
        "aoi_export_dem": str(aoi_export_dem) if aoi_export_dem not in (None, "") else None,
        "final_user_dem": str(final_user_dem) if final_user_dem not in (None, "") else None,
        "run_summary": str(summary_path),
        "run_summary_text": str(txt_path),
        "final_source": "aoi_export_dem",
        "identity_passed": identity.get("passed"),
        "single_writer_passed": single_writer_passed,
    }
    ctx.log.info("[RIVER][SUMMARY] %s", txt_path)
    ctx.log.info("[RIVER][SEAMLESS] canonical_system_id=%s", canonical_system_id)
    ctx.log.info("[RIVER][SEAMLESS] final_source=aoi_export_dem")
    ctx.log.info(
        "[RIVER][SEAMLESS] export_vs_parent=%s max_abs_diff=%s",
        "PASS" if identity.get("passed") else "UNKNOWN",
        identity.get("max_abs_diff"),
    )
    ctx.log.info("[RIVER][SEAMLESS] single_writer=%s", "PASS" if single_writer_passed else "FAIL")
    return summary_path


def _stage_to_dict(stage: ActiveStageResult) -> dict[str, object]:
    primary_output = str(stage.primary_output) if stage.primary_output is not None else None
    return {
        "stage_name": stage.stage_name,
        "stage_class": stage.stage_class,
        "status": stage.status,
        "primary_input": str(stage.primary_input) if stage.primary_input is not None else None,
        "primary_output": primary_output,
        "primary_artifact_name": Path(primary_output).name if primary_output is not None else None,
        "primary_artifact_role": stage.primary_artifact_role,
        "receipt_path": str(stage.receipt_path) if stage.receipt_path is not None else None,
        "detail": stage.detail,
    }


def _validate_active_stage_results(stage_results: list[ActiveStageResult]) -> None:
    missing_primary = [stage.stage_name for stage in stage_results if stage.status == "success" and stage.primary_output is None]
    if missing_primary:
        raise ValueError(f"active_river_stage_missing_primary_output:{missing_primary}")




def _validate_stage_classes(stage_results: list[ActiveStageResult]) -> None:
    invalid_export_roles = []
    for stage in stage_results:
        if stage.stage_class == "export" and stage.primary_artifact_role in {
            "shared_source_bundle_network",
            "canonical_solve_network_source",
            "solve_network",
            "solve_grid_template",
            "solve_authoritative_base_measured_only",
            "centerline_points",
            "centerline_wse_proxy_points",
            "centerline_authoritative_bed_points",
            "centerline_observed_offset_points",
            "centerline_modeled_offset_points",
            "centerline_bed_backbone_points",
            "river_corridor_solve",
            "river_primary_surface_solve",
            "river_primary_surface_solve_locked",
        }:
            invalid_export_roles.append((stage.stage_name, stage.primary_artifact_role))
    if invalid_export_roles:
        raise ValueError(f"active_river_export_stage_owns_solve_artifact:{invalid_export_roles}")

def _write_active_river_stage_chain_summary(
    ctx: ActiveWorkflowContext,
    stage_results: list[ActiveStageResult],
) -> Path:
    reports_dir = Path(ctx.cfg.out_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "river_stage_chain_summary.json"
    payload = {
        "workflow": "river_workflow",
        "run_id": ctx.run_id,
        "canonical_system_id": ctx.report.get("river_workflow", {}).get("canonical_system_id") or ctx.report.get("active_river", {}).get("canonical_system_id"),
        "stage_count": len(stage_results),
        "architecture_execution_model": (ctx.report.get("river_workflow", {}) or {}).get("aoi_execution_mode"),
        "canonical_solution_source": (ctx.report.get("river_workflow", {}) or {}).get("canonical_solution_source"),
        "stages": [_stage_to_dict(stage) for stage in stage_results],
        "first_non_success_stage": next((stage.stage_name for stage in stage_results if stage.status not in {"success", "skipped"}), None),
        "solve_stage_count": sum(1 for stage in stage_results if stage.stage_class == "solve"),
        "export_stage_count": sum(1 for stage in stage_results if stage.stage_class == "export"),
        "finalize_stage_count": sum(1 for stage in stage_results if stage.stage_class == "finalize"),
        "construction_stage_names_in_active_chain": [stage.stage_name for stage in stage_results if stage.stage_class == "solve"],
        "no_canonical_construction_in_export_only": (ctx.report.get("river_workflow", {}) or {}).get("aoi_execution_mode") != "export_only" or not any(stage.stage_class == "solve" for stage in stage_results),
        "successful_stages": sum(1 for stage in stage_results if stage.status == "success"),
        "skipped_stages": sum(1 for stage in stage_results if stage.status == "skipped"),
        "stages_missing_primary_output": [stage.stage_name for stage in stage_results if stage.status == "success" and stage.primary_output is None],
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ctx.report.setdefault("outputs", {})["river_stage_chain_summary"] = str(summary_path)
    ctx.report.setdefault("river_workflow", {})["stage_chain_summary"] = str(summary_path)
    ctx.report.setdefault("active_river", {})["stage_chain_summary"] = str(summary_path)
    return summary_path




def _write_active_river_stage_artifact_contract(
    ctx: ActiveWorkflowContext,
    stage_results: list[ActiveStageResult],
) -> tuple[Path, Path]:
    """Write the Bundle B one-stage/one-artifact contract for the active river path."""
    validate_stage_artifact_contract(stage_results)
    canonical_system_id = (
        ctx.report.get("river_workflow", {}).get("canonical_system_id")
        if isinstance(ctx.report.get("river_workflow"), dict)
        else None
    ) or (
        ctx.report.get("active_river", {}).get("canonical_system_id")
        if isinstance(ctx.report.get("active_river"), dict)
        else None
    )
    json_path, txt_path = write_stage_artifact_contract(
        out_dir=Path(ctx.cfg.out_dir),
        run_id=ctx.run_id,
        stage_results=stage_results,
        canonical_system_id=canonical_system_id,
    )
    ctx.report.setdefault("outputs", {})["river_stage_artifact_contract"] = str(json_path)
    ctx.report.setdefault("outputs", {})["river_stage_artifact_contract_text"] = str(txt_path)
    ctx.report.setdefault("river_workflow", {})["stage_artifact_contract"] = str(json_path)
    ctx.report.setdefault("active_river", {})["stage_artifact_contract"] = str(json_path)
    return json_path, txt_path


def _register_active_river_outputs(ctx: ActiveWorkflowContext, river_result: object, river_deliverable: Path) -> Path:
    outputs = ctx.report.setdefault("outputs", {})
    river_report = ctx.report.setdefault("river", {})
    river_report["status"] = "success"
    river_report["execution_mode"] = "canonical_parent_plus_aoi_export"
    river_report["final_source"] = "aoi_export_dem"
    final_dem_result = getattr(river_result, "final_dem_result", None)
    explicit_aoi_export = getattr(final_dem_result, "aoi_export_dem_path", None) if final_dem_result is not None else None
    outputs["river_workflow_dem_enhanced"] = str(explicit_aoi_export or river_deliverable)
    outputs["combined_warped"] = str(river_deliverable)
    outputs["final"] = str(river_deliverable)
    outputs["final_user_dem"] = str(river_deliverable)
    if final_dem_result is not None:
        if getattr(final_dem_result, "canonical_parent_dem_path", None) is not None:
            outputs["canonical_parent_dem"] = str(final_dem_result.canonical_parent_dem_path)
        if getattr(final_dem_result, "aoi_export_dem_path", None) is not None:
            outputs["aoi_export_dem"] = str(final_dem_result.aoi_export_dem_path)
    river_outputs = (((ctx.report.get("river") or {}) if isinstance(ctx.report.get("river"), dict) else {}).get("outputs") or {})
    if isinstance(river_outputs, dict):
        for key, value in river_outputs.items():
            outputs.setdefault(key, value)
    run_contract = getattr(getattr(river_result, "primary_artifacts", None), "run_contract_path", None)
    pipeline_manifest = getattr(getattr(river_result, "primary_artifacts", None), "pipeline_manifest_path", None)
    if run_contract is not None:
        outputs["river_workflow_run_contract"] = str(run_contract)
    if pipeline_manifest is not None:
        outputs["river_workflow_pipeline_manifest"] = str(pipeline_manifest)
    return Path(outputs["final"])


def _build_river_workflow_result(
    river_raster: Optional[Path],
    river_deliverable: Optional[Path],
    stage_results: Optional[list[ActiveStageResult]] = None,
    stage_chain_summary_path: Optional[Path] = None,
) -> ActiveRiverWorkflowResult:
    final_native = Path(river_raster) if river_raster is not None else None
    final_for_user = Path(river_deliverable) if river_deliverable is not None else None
    finalize_targets = ActiveRiverFinalizeTargets(
        final_native=final_native,
        final_for_user=final_for_user,
    )
    return ActiveRiverWorkflowResult(
        river_raster=final_native,
        river_deliverable=final_for_user,
        finalize_targets=finalize_targets,
        stage_results=list(stage_results or []),
        stage_chain_summary_path=stage_chain_summary_path,
    )


def _build_inner_river_stage_results(linear_result: Any, cfg: Any) -> list[ActiveStageResult]:
    final_dem_result = getattr(linear_result, "final_dem_result", None)
    if final_dem_result is None:
        return []
    if str(getattr(final_dem_result, "final_writer_mode", "")) == "existing_canonical_parent_exact_aoi_export":
        parent_value = getattr(final_dem_result, "canonical_parent_dem_path", None)
        export_value = getattr(final_dem_result, "aoi_export_dem_path", None)
        parent = Path(parent_value) if parent_value not in (None, "") else None
        export = Path(export_value) if export_value not in (None, "") else None
        return [
            ActiveStageResult(
                stage_name="read_existing_canonical_parent",
                stage_class="export",
                status="success" if parent is not None else "failed",
                primary_input=parent,
                primary_output=parent,
                receipt_path=None,
                detail="read already-built canonical parent DEM; no river construction stages executed",
                primary_artifact_role="canonical_parent_dem",
            ),
            ActiveStageResult(
                stage_name="aoi_export_dem",
                stage_class="export",
                status="success" if export is not None else "failed",
                primary_input=parent,
                primary_output=export,
                receipt_path=Path(getattr(final_dem_result, "aoi_export_receipt_path")) if getattr(final_dem_result, "aoi_export_receipt_path", None) not in (None, "") else None,
                detail="exact AOI export from existing canonical parent DEM",
                primary_artifact_role="aoi_export_dem",
            ),
        ]
    receipts = dict(getattr(linear_result, "stage_receipts", {}) or {})
    solve_result = getattr(linear_result, "solve_result", None)
    grid_result = getattr(linear_result, "grid_result", None)
    authoritative_result = getattr(linear_result, "authoritative_result", None)
    centerline_result = getattr(linear_result, "centerline_result", None)
    wse_result = getattr(linear_result, "wse_result", None)
    authoritative_bed_result = getattr(linear_result, "authoritative_bed_result", None)
    observed_offset_result = getattr(linear_result, "observed_offset_result", None)
    modeled_offset_result = getattr(linear_result, "modeled_offset_result", None)
    backbone_result = getattr(linear_result, "backbone_result", None)
    corridor_result = getattr(linear_result, "corridor_result", None)
    surface_result = getattr(linear_result, "surface_result", None)
    lock_result = getattr(linear_result, "lock_result", None)
    export_result = getattr(linear_result, "export_result", None)
    stage_specs = [
        ("solve_domain", Path(getattr(cfg, "out_dir", ".")), getattr(solve_result, "canonical_network_path", None), "canonical solve domain and network", "solve_network"),
        ("grids", getattr(solve_result, "canonical_network_path", None), getattr(grid_result, "solve_grid_template_path", None), "solve/export grid templates", "solve_grid_template"),
        ("authoritative_inputs", getattr(grid_result, "solve_grid_template_path", None), getattr(authoritative_result, "solve_authoritative_base_measured_only_path", None), "authoritative measured-only inputs", "solve_authoritative_base_measured_only"),
        ("centerline_points", getattr(authoritative_result, "solve_authoritative_base_measured_only_path", None), getattr(centerline_result, "centerline_points_path", None), "centerline scaffold points", "centerline_points"),
        ("centerline_wse_proxy", getattr(centerline_result, "centerline_points_path", None), getattr(wse_result, "centerline_wse_proxy_points_path", None), "centerline WSE proxy", "centerline_wse_proxy_points"),
        ("centerline_authoritative_bed", getattr(centerline_result, "centerline_points_path", None), getattr(authoritative_bed_result, "centerline_authoritative_bed_points_path", None), "centerline authoritative bed", "centerline_authoritative_bed_points"),
        ("centerline_observed_offset", getattr(wse_result, "centerline_wse_proxy_points_path", None), getattr(observed_offset_result, "centerline_observed_offset_points_path", None), "observed offset points", "centerline_observed_offset_points"),
        ("centerline_modeled_offset", getattr(observed_offset_result, "centerline_observed_offset_points_path", None), getattr(modeled_offset_result, "centerline_modeled_offset_points_path", None), "modeled offset points", "centerline_modeled_offset_points"),
        ("centerline_bed_backbone", getattr(modeled_offset_result, "centerline_modeled_offset_points_path", None), getattr(backbone_result, "centerline_bed_backbone_points_path", None), "bed backbone points", "centerline_bed_backbone_points"),
        ("river_corridor_solve", getattr(grid_result, "solve_grid_template_path", None), getattr(corridor_result, "river_corridor_solve_path", None), "solve corridor raster", "river_corridor_solve"),
        ("river_primary_surface_solve", getattr(backbone_result, "centerline_bed_backbone_points_path", None), getattr(surface_result, "river_primary_surface_solve_path", None), "primary surface solve raster", "river_primary_surface_solve"),
        ("river_primary_surface_solve_locked", getattr(surface_result, "river_primary_surface_solve_path", None), getattr(lock_result, "river_primary_surface_solve_locked_path", None), "authoritative-locked primary surface", "river_primary_surface_solve_locked"),
        ("river_export_handoff", getattr(lock_result, "river_primary_surface_solve_locked_path", None), getattr(export_result, "river_guidance_export_path", None), "export handoff rasters", "river_guidance_export"),
        ("canonical_parent_dem", getattr(lock_result, "river_primary_surface_solve_locked_path", None), getattr(final_dem_result, "canonical_parent_dem_path", None), "canonical parent DEM assembly", "canonical_parent_dem"),
        ("aoi_export_dem", getattr(final_dem_result, "canonical_parent_dem_path", None), getattr(final_dem_result, "aoi_export_dem_path", getattr(final_dem_result, "dem_enhanced_final_path", None)), "exact AOI export from canonical parent DEM", "aoi_export_dem"),
    ]
    stage_results: list[ActiveStageResult] = []
    for stage_name, primary_input, primary_output, detail, artifact_role in stage_specs:
        out_path = Path(primary_output) if primary_output is not None else None
        in_path = Path(primary_input) if primary_input is not None else None
        receipt = receipts.get(stage_name)
        receipt_path = Path(receipt) if receipt is not None else None
        status = "success" if out_path is not None else "failed"
        stage_results.append(ActiveStageResult(
            stage_name=stage_name,
            stage_class="solve",
            status=status,
            primary_input=in_path,
            primary_output=out_path,
            receipt_path=receipt_path,
            detail=detail,
            primary_artifact_role=artifact_role,
        ))
    return stage_results


def run_active_river_workflow(ctx: ActiveWorkflowContext) -> ActiveRiverWorkflowResult:
    cfg = ctx.cfg
    callbacks = ctx.callbacks
    run_river = callbacks["run_river_fn"]
    run_river_workflow_details = callbacks["run_river_workflow_direct_fn"]
    write_final_output_receipt = callbacks["write_final_output_receipt_fn"]

    ctx.log.info("[RIVER][WORKFLOW][STEP 1/5] Resolve export AOI and activate the built-in river workflow.")
    ctx.log.info("[RIVER][WORKFLOW] Built-in AOI-independent river workflow selected. Running canonical parent -> AOI export -> final materialization route and stopping before non-river final-route routing.")
    river_deliverable: Optional[Path] = None
    river_raster: Optional[Path] = None
    stage_results: list[ActiveStageResult] = []
    river_result: object | None = None

    if "river" in cfg.methods:
        ctx.log.info("[RIVER][WORKFLOW][STEP 2/5] Execute the built-in river workflow. This resolves shared inputs, prepares the canonical solve bundle, and runs the detailed river stage pipeline.")
        if run_river_workflow_details is not None:
            river_result = run_river_workflow_details(
                cfg,
                ctx.report,
                logger=ctx.log,
                script_dir=callbacks.get("script_dir"),
                ensure_dir_fn=callbacks.get("ensure_dir_fn"),
                detect_working_srs_fn=callbacks.get("detect_working_srs_fn"),
                estimate_raster_pixel_size_m_for_dst_crs_fn=callbacks.get("estimate_raster_pixel_size_m_for_dst_crs_fn"),
                resolve_river_shared_source_artifacts_fn=callbacks.get("resolve_river_shared_source_artifacts_fn"),
                prepare_river_canonical_source_bundle_fn=callbacks.get("prepare_river_canonical_source_bundle_fn"),
                river_inputs_override=None,
                return_river_workflow_details=True,
            )
        else:
            river_result = run_river(cfg, ctx.report, river_inputs_override=None)
        canonical_system_id = getattr(getattr(river_result, "solve_result", None), "canonical_system_id", None)
        if canonical_system_id not in (None, ""):
            ctx.report.setdefault("river_workflow", {})["canonical_system_id"] = str(canonical_system_id)
            ctx.report.setdefault("active_river", {})["canonical_system_id"] = str(canonical_system_id)
            ctx.log.info("[RIVER][CANONICAL] system_id=%s", canonical_system_id)
        canonical_cache_hit = bool(getattr(river_result, "shared_solve_reused", False))
        ctx.report.setdefault("river_workflow", {})["canonical_cache_hit"] = canonical_cache_hit
        ctx.report.setdefault("active_river", {})["canonical_cache_hit"] = canonical_cache_hit
        run_contract_path = getattr(river_result, "run_contract_path", None)
        if run_contract_path not in (None, "") and Path(run_contract_path).is_file():
            try:
                run_contract = json.loads(Path(run_contract_path).read_text(encoding="utf-8"))
                shared_solve = run_contract.get("shared_solve") if isinstance(run_contract.get("shared_solve"), dict) else {}
                cache_key = shared_solve.get("cache_key")
                if cache_key not in (None, ""):
                    ctx.report.setdefault("river_workflow", {})["canonical_cache_key"] = str(cache_key)
                    ctx.report.setdefault("active_river", {})["canonical_cache_key"] = str(cache_key)
                if shared_solve.get("cache_manifest_path") not in (None, ""):
                    ctx.report.setdefault("river_workflow", {})["canonical_cache_manifest_path"] = str(shared_solve.get("cache_manifest_path"))
                    ctx.report.setdefault("active_river", {})["canonical_cache_manifest_path"] = str(shared_solve.get("cache_manifest_path"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                ctx.log.warning("[RIVER][CANONICAL] Unable to read run contract cache identity from %s: %s", run_contract_path, exc)
        if getattr(river_result, "final_dem_result", None) is None:
            raise RuntimeError("active_river_result_missing_final_dem_result: built-in workflow must return canonical parent and AOI export paths")
        aoi_export_path = getattr(river_result.final_dem_result, "aoi_export_dem_path", None)
        canonical_parent_path = getattr(river_result.final_dem_result, "canonical_parent_dem_path", None)
        if aoi_export_path in (None, "") or canonical_parent_path in (None, ""):
            raise RuntimeError("active_river_result_missing_parent_export_paths: built-in workflow must return explicit canonical parent and AOI export paths")
        river_raster = Path(aoi_export_path)
        stage_results.extend(_build_inner_river_stage_results(river_result, cfg))
    else:
        stage_results.append(ActiveStageResult(
            stage_name="run_river_workflow",
            stage_class="solve",
            status="skipped",
            primary_input=Path(cfg.out_dir),
            primary_output=None,
            detail="river method not requested",
            primary_artifact_role=None,
        ))

    if river_raster is not None:
        ctx.log.info("[RIVER][WORKFLOW][STEP 3/5] Materialize combined/DEM_enhanced.tif from the named AOI export artifact.")
        river_deliverable, materialization_receipt = _materialize_river_final_dem(ctx, river_raster)
        stage_results.append(ActiveStageResult(
            stage_name="materialize_final_dem",
            stage_class="export",
            status="success",
            primary_input=river_raster,
            primary_output=river_deliverable,
            receipt_path=materialization_receipt,
            detail="materialize final user DEM from AOI export",
            primary_artifact_role="final_user_dem",
        ))
        identity_receipt, identity_result = _run_aoi_identity_and_writer_checks(
            ctx,
            river_result=river_result,
            aoi_export_dem=river_raster,
            materialization_receipt=materialization_receipt,
        )
        retained_export_identity = _update_retained_aoi_export_identity(
            ctx,
            river_result=river_result,
            aoi_export_dem=river_raster,
            final_user_dem=river_deliverable,
            identity_result=identity_result,
        )
        canonical_parent_for_guard = getattr(getattr(river_result, "final_dem_result", None), "canonical_parent_dem_path", None)
        if canonical_parent_for_guard in (None, ""):
            raise RuntimeError("final_route_guard_missing_canonical_parent_dem")
        final_route_guard = write_final_route_guard_receipt(
            out_dir=Path(cfg.out_dir),
            canonical_parent_dem=Path(canonical_parent_for_guard),
            aoi_export_dem=river_raster,
            final_user_dem=river_deliverable,
            materialization_receipt=materialization_receipt,
            identity_result=identity_result,
        )
        ctx.report.setdefault("outputs", {})["river_final_route_guard_receipt"] = str(final_route_guard.receipt_path)
        ctx.report.setdefault("outputs", {})["river_final_route_guard_summary"] = str(final_route_guard.text_path)
        ctx.report.setdefault("river_workflow", {})["final_route_guard_receipt"] = str(final_route_guard.receipt_path)
        ctx.report.setdefault("river_workflow", {})["final_route_guard_summary"] = str(final_route_guard.text_path)
        ctx.report.setdefault("river_workflow", {})["final_route_guard_passed"] = bool(final_route_guard.passed)
        ctx.report.setdefault("active_river", {})["final_route_guard_receipt"] = str(final_route_guard.receipt_path)
        ctx.report.setdefault("active_river", {})["final_route_guard_summary"] = str(final_route_guard.text_path)
        ctx.report.setdefault("active_river", {})["final_route_guard_passed"] = bool(final_route_guard.passed)
        architecture_summary = _write_river_architecture_summary(
            ctx,
            river_result=river_result,
            final_user_dem=river_deliverable,
            identity_result=identity_result,
        )
        stage_results.append(ActiveStageResult(
            stage_name="aoi_identity_check",
            stage_class="export",
            status="success" if identity_result.get("passed") else "failed",
            primary_input=river_raster,
            primary_output=identity_receipt,
            receipt_path=identity_receipt,
            detail="verify AOI export is an exact parent-grid subset and final DEM has one writer",
            primary_artifact_role="identity_receipt",
        ))
        stage_results.append(ActiveStageResult(
            stage_name="retain_aoi_export_identity",
            stage_class="export",
            status="success",
            primary_input=river_raster,
            primary_output=retained_export_identity,
            receipt_path=retained_export_identity,
            detail="retain AOI export identity receipt after cleanup",
            primary_artifact_role="aoi_export_identity_report",
        ))
        stage_results.append(ActiveStageResult(
            stage_name="final_route_guard",
            stage_class="export",
            status="success" if final_route_guard.passed else "failed",
            primary_input=river_raster,
            primary_output=final_route_guard.receipt_path,
            receipt_path=final_route_guard.receipt_path,
            detail="hard guard for canonical parent -> AOI export -> final DEM route",
            primary_artifact_role="final_route_guard_receipt",
        ))
        stage_results.append(ActiveStageResult(
            stage_name="write_river_architecture_summary",
            stage_class="finalize",
            status="success",
            primary_input=river_deliverable,
            primary_output=architecture_summary,
            receipt_path=architecture_summary,
            detail="write concise canonical-parent plus AOI-export architecture summary",
            primary_artifact_role="river_architecture_summary",
        ))
        ctx.log.info("[RIVER][ARCHITECTURE] %s", architecture_summary)
        ctx.log.info(
            "[RIVER][IDENTITY] export_vs_parent=%s max_abs_diff=%s mismatch_pixels=%s",
            "PASS" if identity_result.get("passed") else "FAIL",
            identity_result.get("max_abs_diff"),
            identity_result.get("mismatch_pixels"),
        )
        ctx.log.info("[RIVER][FINAL] single_writer=PASS")
        ctx.log.info("[RIVER][WORKFLOW][STEP 4/5] Register final outputs and canonical river-workflow artifacts for downstream reporting/comparison.")
        final_registered = _register_active_river_outputs(ctx, river_result, river_deliverable)
        stage_results.append(ActiveStageResult(
            stage_name="register_river_outputs",
            stage_class="export",
            status="success",
            primary_input=river_deliverable,
            primary_output=final_registered,
            detail="register final and combined_warped outputs",
            primary_artifact_role="registered_final_output",
        ))
        ctx.log.info("[RIVER][WORKFLOW][STEP 5/5] Write final DEM receipt and comparison products. This packages the final raster, baseline reference, and comparison-folder artifacts.")
        receipt_result = write_final_output_receipt(cfg, ctx.report)
        receipt_value = receipt_result or ctx.report.get("outputs", {}).get("final_output_receipt")
        receipt_path = Path(receipt_value) if receipt_value is not None else None
        final_folder_products = _validate_final_output_folder(Path(cfg.out_dir), receipt_path)
        ctx.report.setdefault("outputs", {})["final_folder"] = str(Path(cfg.out_dir) / "final")
        ctx.report.setdefault("outputs", {})["final_folder_required_outputs"] = {
            name: str(path) for name, path in final_folder_products.items()
        }
        stage_results.append(ActiveStageResult(
            stage_name="write_final_output_receipt",
            stage_class="export",
            status="success",
            primary_input=river_deliverable,
            primary_output=receipt_path,
            receipt_path=receipt_path,
            detail="strict final output contract receipt and complete final/ comparison folder",
            primary_artifact_role="final_output_receipt",
        ))
        consolidated_receipt, consolidated_summary = _write_consolidated_river_workflow_receipt(
            ctx,
            river_result=river_result,
            final_user_dem=river_deliverable,
            identity_result=identity_result,
            final_output_receipt=receipt_path,
            final_folder_products=final_folder_products,
            stage_results=stage_results,
        )
        stage_results.append(ActiveStageResult(
            stage_name="write_river_workflow_final_receipt",
            stage_class="finalize",
            status="success",
            primary_input=receipt_path,
            primary_output=consolidated_receipt,
            receipt_path=consolidated_receipt,
            detail="write consolidated river workflow receipt and human-readable summary",
            primary_artifact_role="river_workflow_receipt",
        ))
        stage_results.append(ActiveStageResult(
            stage_name="write_river_workflow_summary",
            stage_class="finalize",
            status="success",
            primary_input=consolidated_receipt,
            primary_output=consolidated_summary,
            receipt_path=consolidated_summary,
            detail="write human-readable river workflow final summary",
            primary_artifact_role="river_workflow_summary",
        ))
        ctx.log.info("[RIVER][FINAL_RECEIPT] %s", consolidated_receipt)
        ctx.log.info("[RIVER][FINAL_SUMMARY] %s", consolidated_summary)
    else:
        stage_results.append(ActiveStageResult(
            stage_name="materialize_final_dem",
            stage_class="export",
            status="skipped",
            primary_input=None,
            primary_output=None,
            detail="no AOI export artifact produced",
            primary_artifact_role=None,
        ))
        stage_results.append(ActiveStageResult(
            stage_name="register_river_outputs",
            stage_class="export",
            status="skipped",
            primary_input=None,
            primary_output=None,
            detail="no deliverable to register",
            primary_artifact_role=None,
        ))
        stage_results.append(ActiveStageResult(
            stage_name="write_final_output_receipt",
            stage_class="export",
            status="skipped",
            primary_input=None,
            primary_output=None,
            detail="no deliverable contract to write",
            primary_artifact_role=None,
        ))

    _validate_active_stage_results(stage_results)
    _validate_stage_classes(stage_results)
    stage_chain_summary_path = _write_active_river_stage_chain_summary(ctx, stage_results)
    _write_active_river_stage_artifact_contract(ctx, stage_results)
    if river_deliverable is not None:
        summary_path = _write_seamless_run_summary(ctx, river_result=river_result, final_user_dem=river_deliverable)
        stage_results.append(ActiveStageResult(
            stage_name="write_run_summary",
            stage_class="finalize",
            status="success" if summary_path is not None else "failed",
            primary_input=river_deliverable,
            primary_output=summary_path,
            receipt_path=summary_path,
            detail="write consolidated seamless DEM run summary",
            primary_artifact_role="run_summary",
        ))
        stage_chain_summary_path = _write_active_river_stage_chain_summary(ctx, stage_results)
        _write_active_river_stage_artifact_contract(ctx, stage_results)

    return _build_river_workflow_result(
        river_raster=river_raster,
        river_deliverable=river_deliverable,
        stage_results=stage_results,
        stage_chain_summary_path=stage_chain_summary_path,
    )


def finalize_active_river_workflow(
    ctx: ActiveWorkflowContext,
    workflow_result: ActiveRiverWorkflowResult,
) -> ActiveWorkflowFinalizeResult:
    callbacks = ctx.callbacks
    finalize_existing_output_run_stage = callbacks["finalize_existing_output_run_stage_fn"]
    exit_code = finalize_existing_output_run_stage(
        cfg=ctx.cfg,
        args=ctx.args,
        report=ctx.report,
        log=ctx.log,
        fatal_errors=ctx.fatal_errors,
        final_native=workflow_result.finalize_targets.final_native,
        final_for_user=workflow_result.finalize_targets.final_for_user,
        final_provenance=None,
        write_bundle_fn=callbacks["write_bundle_fn"],
        finalize_run_fn=callbacks["finalize_run_fn"],
        write_io_manifest_fn=callbacks["write_io_manifest_fn"],
        emit_artifacts_fn=callbacks["emit_artifacts_fn"],
        run_seam_comparisons_fn=callbacks["run_seam_comparisons_fn"],
    )
    finalize_stage = ActiveStageResult(
        stage_name="finalize_existing_output_run_stage",
        stage_class="finalize",
        status="success" if exit_code == 0 else "failed",
        primary_input=workflow_result.finalize_targets.final_for_user or workflow_result.finalize_targets.final_native,
        primary_output=workflow_result.finalize_targets.final_for_user or workflow_result.finalize_targets.final_native,
        receipt_path=Path(ctx.report.get("outputs", {}).get("bathy_report")) if ctx.report.get("outputs", {}).get("bathy_report") else None,
        detail="shared finalize/report path",
        primary_artifact_role="bathy_report",
    )
    workflow_result.stage_results.append(finalize_stage)
    _validate_active_stage_results(workflow_result.stage_results)
    _validate_stage_classes(workflow_result.stage_results)
    workflow_result.stage_chain_summary_path = _write_active_river_stage_chain_summary(ctx, workflow_result.stage_results)
    _write_active_river_stage_artifact_contract(ctx, workflow_result.stage_results)
    return ActiveWorkflowFinalizeResult(workflow_result=workflow_result, exit_code=exit_code)


# Bundle C removes active use of old linear workflow entrypoint names.
# Older imports should use river_workflow_entry.py / run_active_river_workflow.
