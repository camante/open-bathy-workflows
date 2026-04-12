from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from final_reporting import evaluate_overlap_identity_checks, write_river_stability_summary, write_validation_invariance_summary
from nested_aoi_regression import run_nested_aoi_regression
from nested_aoi_contracts import evaluate_nested_aoi_contracts
from postrun_regression_contract import build_postrun_regression_context
from seam_metrics import (
    compute_mask_boundary_seam_metrics,
    compute_seam_metrics,
    load_primary_raster_from_io_manifest,
)


def _load_path_list(values: Iterable[str] | None, list_path: Optional[str]) -> list[str]:
    out = [str(v) for v in (values or []) if str(v).strip()]
    if list_path:
        p = Path(str(list_path))
        if not p.exists():
            raise FileNotFoundError(f"path list not found: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scope", "neighbor", "artifact", "status", "contract_status", "contract_tolerance", "support_class_code", "support_class_name",
        "n_valid", "common_support_pixels", "trusted_common_support_pixels", "mean_abs", "rmse", "max_abs", "p95_abs", "exact_identity_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def run_postrun_regression_stage(*, args, cfg, report, run_id, final, final_for_user, report_path, logger) -> None:
    seam_ios = _load_path_list(getattr(args, "seam_compare_with_io", []) or [], getattr(args, "seam_compare_with_io_list", None))
    nested_outputs = _load_path_list(
        getattr(args, "nested_aoi_compare_with_final_outputs", []) or [],
        getattr(args, "nested_aoi_compare_with_final_outputs_list", None),
    )

    current_final_outputs_manifest = Path(cfg.out_dir) / "final_outputs.json"
    current_io_manifest = Path(cfg.out_dir) / "io_manifest.json"
    context = build_postrun_regression_context(
        current_final_outputs_manifest=current_final_outputs_manifest if current_final_outputs_manifest.exists() else None,
        current_io_manifest=current_io_manifest if current_io_manifest.exists() else None,
        seam_neighbor_io_manifests=seam_ios,
        nested_aoi_neighbor_final_outputs=nested_outputs,
        seam_strip_px=int(getattr(args, "seam_strip_px", 3)),
        overlap_tolerance=float(getattr(args, "nested_aoi_overlap_tolerance", 1.0e-6)),
        trusted_tolerance=float(getattr(args, "nested_aoi_trusted_tolerance", 1.0e-6)),
    )

    this_raster = Path(str(final_for_user)) if final_for_user else (Path(str(final)) if final else None)
    if this_raster is None or not this_raster.exists():
        this_raster = None

    seam_results: list[dict[str, Any]] = []
    seam_json_path: Optional[Path] = None
    if context.seam_neighbor_io_manifests:
        if this_raster is None:
            raise RuntimeError("Seam compare requested but this run has no final raster output.")
        for neighbor_io in context.seam_neighbor_io_manifests:
            if not neighbor_io.exists():
                seam_results.append({
                    "status": "error",
                    "neighbor_io_manifest": str(neighbor_io),
                    "error": "Neighbor io_manifest.json not found",
                })
                continue
            try:
                neighbor_raster = load_primary_raster_from_io_manifest(neighbor_io)
                metrics = compute_seam_metrics(this_raster, neighbor_raster, strip_px=context.seam_strip_px)
                metrics["neighbor_io_manifest"] = str(neighbor_io)
                seam_results.append(metrics)
            except Exception as exc:
                logger.debug("postrun seam comparison failed", exc_info=True)
                seam_results.append({
                    "status": "error",
                    "neighbor_io_manifest": str(neighbor_io),
                    "error": str(exc),
                })
        seam_json_path = Path(cfg.out_dir) / "seam_comparisons.json"
        seam_payload = {
            "this_raster": str(this_raster),
            "strip_px": context.seam_strip_px,
            "comparisons": seam_results,
        }
        seam_json_path.write_text(json.dumps(seam_payload, indent=2), encoding="utf-8")
        report.setdefault("seams", {})["adjacent_tile_comparisons"] = seam_results
        report.setdefault("outputs", {})["seam_comparisons_json"] = str(seam_json_path)

    nested_payload: Optional[dict[str, Any]] = None
    nested_contract_payload: Optional[dict[str, Any]] = None
    nested_json_path: Optional[Path] = None
    nested_contract_json_path: Optional[Path] = None
    nested_csv_path: Optional[Path] = None
    if context.nested_aoi_neighbor_final_outputs:
        if context.current_final_outputs_manifest is None or not context.current_final_outputs_manifest.exists():
            raise RuntimeError("Nested-AOI regression requested but current final_outputs.json is missing.")
        nested_payload = run_nested_aoi_regression(
            current_final_outputs_manifest=context.current_final_outputs_manifest,
            neighbor_final_outputs_manifests=context.nested_aoi_neighbor_final_outputs,
            overlap_tolerance=context.overlap_tolerance,
            trusted_tolerance=context.trusted_tolerance,
        )
        nested_json_path = Path(cfg.out_dir) / "nested_aoi_regression.json"
        nested_json_path.write_text(json.dumps(nested_payload, indent=2), encoding="utf-8")
        nested_contract_payload = evaluate_nested_aoi_contracts(
            current_final_outputs_manifest=context.current_final_outputs_manifest,
            nested_payload=nested_payload,
            overlap_tolerance=context.overlap_tolerance,
            trusted_tolerance=context.trusted_tolerance,
        )
        nested_contract_json_path = Path(cfg.out_dir) / "nested_aoi_contract_evaluation.json"
        nested_contract_json_path.write_text(json.dumps(nested_contract_payload, indent=2), encoding="utf-8")
        rows = []
        for check in nested_payload.get("overlap_identity_checks", []):
            rows.append({
                "scope": "overlap",
                "neighbor": check.get("neighbor_final_outputs_manifest"),
                "artifact": check.get("artifact"),
                "status": check.get("status"),
                "contract_status": "ok" if (check.get("status") == "ok" and float(check.get("max_abs") or 0.0) <= context.overlap_tolerance) else ("no_valid" if check.get("status") != "ok" else "failed"),
                "contract_tolerance": context.overlap_tolerance,
                "n_valid": check.get("n_valid"),
                "mean_abs": check.get("mean_abs"),
                "rmse": check.get("rmse"),
                "max_abs": check.get("max_abs"),
                "p95_abs": check.get("p95_abs"),
                "exact_identity_fraction": check.get("exact_identity_fraction"),
            })
        for check in nested_payload.get("trusted_interior_identity_checks", []):
            rows.append({
                "scope": "trusted_interior",
                "neighbor": check.get("neighbor_final_outputs_manifest"),
                "artifact": check.get("artifact"),
                "status": check.get("status"),
                "contract_status": "ok" if (check.get("status") == "ok" and float(check.get("max_abs") or 0.0) <= context.trusted_tolerance) else ("no_valid" if check.get("status") != "ok" else "failed"),
                "contract_tolerance": context.trusted_tolerance,
                "n_valid": check.get("n_valid"),
                "mean_abs": check.get("mean_abs"),
                "rmse": check.get("rmse"),
                "max_abs": check.get("max_abs"),
                "p95_abs": check.get("p95_abs"),
                "exact_identity_fraction": check.get("exact_identity_fraction"),
            })
        for check in nested_payload.get("support_class_depth_identity_checks", []):
            rows.append({
                "scope": "overlap",
                "neighbor": check.get("neighbor_final_outputs_manifest"),
                "artifact": check.get("artifact"),
                "status": check.get("status"),
                "contract_status": check.get("contract_status"),
                "contract_tolerance": check.get("contract_tolerance"),
                "support_class_code": check.get("support_class_code"),
                "support_class_name": check.get("support_class_name"),
                "n_valid": check.get("n_valid"),
                "common_support_pixels": check.get("common_support_pixels"),
                "trusted_common_support_pixels": check.get("trusted_common_support_pixels"),
                "mean_abs": check.get("mean_abs"),
                "rmse": check.get("rmse"),
                "max_abs": check.get("max_abs"),
                "p95_abs": check.get("p95_abs"),
                "exact_identity_fraction": check.get("exact_identity_fraction"),
            })
        for check in nested_payload.get("trusted_support_class_depth_identity_checks", []):
            rows.append({
                "scope": "trusted_interior",
                "neighbor": check.get("neighbor_final_outputs_manifest"),
                "artifact": check.get("artifact"),
                "status": check.get("status"),
                "contract_status": check.get("contract_status"),
                "contract_tolerance": check.get("contract_tolerance"),
                "support_class_code": check.get("support_class_code"),
                "support_class_name": check.get("support_class_name"),
                "n_valid": check.get("n_valid"),
                "common_support_pixels": check.get("common_support_pixels"),
                "trusted_common_support_pixels": check.get("trusted_common_support_pixels"),
                "mean_abs": check.get("mean_abs"),
                "rmse": check.get("rmse"),
                "max_abs": check.get("max_abs"),
                "p95_abs": check.get("p95_abs"),
                "exact_identity_fraction": check.get("exact_identity_fraction"),
            })
        nested_csv_path = Path(cfg.out_dir) / "nested_aoi_regression_metrics.csv"
        _write_csv(nested_csv_path, rows)
        report.setdefault("seams", {})["nested_aoi_regression"] = nested_payload
        report.setdefault("seams", {})["overlap_identity_checks"] = nested_payload.get("overlap_identity_checks", [])
        report.setdefault("seams", {})["trusted_interior_identity_checks"] = nested_payload.get("trusted_interior_identity_checks", [])
        report.setdefault("seams", {})["overlap_identity_evaluation"] = nested_payload.get("overlap_identity_evaluation", {})
        report.setdefault("seams", {})["trusted_interior_identity_evaluation"] = nested_payload.get("trusted_interior_identity_evaluation", {})
        report.setdefault("seams", {})["nested_aoi_contract_evaluation"] = nested_contract_payload or {}
        report.setdefault("outputs", {})["nested_aoi_regression_json"] = str(nested_json_path)
        report.setdefault("outputs", {})["nested_aoi_contract_evaluation_json"] = str(nested_contract_json_path) if nested_contract_json_path else None
        report.setdefault("outputs", {})["nested_aoi_regression_metrics_csv"] = str(nested_csv_path)

    write_validation_invariance_summary(
        cfg,
        report,
        final_native=Path(final) if final else None,
        final_for_user=Path(str(final_for_user)) if final_for_user else None,
        final_provenance=report.get("outputs", {}).get("selected_final_provenance"),
        logger=logger,
        enforce_hard_fail=True,
    )
    write_river_stability_summary(cfg, report, logger=logger)

    try:
        from river_diagnostics import create_unified_bathy_report

        sdb_out = cfg.out_dir / "sdb" if "sdb" in cfg.methods else None
        river_out = cfg.out_dir / "river" if "river" in cfg.methods else None
        unified_path = create_unified_bathy_report(
            output_dir=cfg.out_dir,
            sdb_output=sdb_out,
            river_output=river_out,
            methods=cfg.methods,
            priority=cfg.priority,
            write_json=bool(getattr(cfg, "write_legacy_unified_report_json", False)),
            write_markdown=bool(getattr(cfg, "write_legacy_unified_report_markdown", False)),
        )
        if bool(getattr(cfg, "write_legacy_unified_report_json", False)):
            logger.info("Unified report written: %s", unified_path)
        else:
            logger.info("Unified legacy report skipped by default; set write_legacy_unified_report_json/write_legacy_unified_report_markdown to enable.")
    except Exception:
        logger.debug("postrun unified report generation failed", exc_info=True)

    try:
        input_receipt = {
            "run_id": run_id,
            "created_utc": report.get("run", {}).get("created_utc"),
            "command": report.get("run", {}).get("command"),
            "aoi": report.get("run", {}).get("aoi"),
            "start": report.get("run", {}).get("start"),
            "end": report.get("run", {}).get("end"),
            "methods_requested": report.get("run", {}).get("methods_requested"),
            "methods_effective": report.get("run", {}).get("methods_effective"),
            "priority": report.get("run", {}).get("priority"),
            "crs_out": report.get("run", {}).get("crs_out"),
            "inputs": report.get("inputs", {}),
            "outputs": report.get("outputs", {}),
        }
        input_receipt_path = Path(cfg.derived_cache_root) / "input_receipt.json"
        input_receipt_path.write_text(json.dumps(input_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("Input receipt written: %s", input_receipt_path)

        river_r = report.get("outputs", {}).get("river_bottom_warped")
        fused_r = report.get("outputs", {}).get("combined_warped")
        river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
        mask_r = (
            river_outputs.get("river_channel_mask")
            or report.get("outputs", {}).get("river_channel_mask")
            or (str(cfg.river_channel_mask) if getattr(cfg, "river_channel_mask", None) else None)
        )
        seam_metrics = {
            "ok": False,
            "reason": "missing_inputs",
            "river_raster": river_r,
            "fused_raster": fused_r,
            "mask_raster": mask_r,
        }
        if river_r and fused_r and mask_r and Path(mask_r).exists():
            seam_metrics = compute_mask_boundary_seam_metrics(
                river_raster=str(river_r),
                fused_raster=str(fused_r),
                mask_raster=str(mask_r),
                boundary_mode="both",
            )
            seam_metrics.update({"river_raster": str(river_r), "fused_raster": str(fused_r), "mask_raster": str(mask_r)})
        seam_receipt_path = Path(cfg.derived_cache_root) / "river_transition_seam_metrics.json"
        seam_receipt_path.write_text(json.dumps(seam_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report.setdefault("outputs", {})["river_transition_seam_metrics"] = str(seam_receipt_path)
        logger.info("River transition seam metrics written: %s", seam_receipt_path)
    except Exception:
        logger.debug("postrun verifiability receipt generation failed", exc_info=True)

    summary = {
        "run_id": run_id,
        "seam_comparisons_json": str(seam_json_path) if seam_json_path else None,
        "nested_aoi_regression_json": str(nested_json_path) if nested_json_path else None,
        "nested_aoi_contract_evaluation_json": str(nested_contract_json_path) if nested_contract_json_path else None,
        "nested_aoi_regression_metrics_csv": str(nested_csv_path) if nested_csv_path else None,
        "seam_neighbor_count": len(context.seam_neighbor_io_manifests),
        "nested_neighbor_count": len(context.nested_aoi_neighbor_final_outputs),
        "overlap_identity_evaluation": (nested_payload or {}).get("overlap_identity_evaluation"),
        "trusted_interior_identity_evaluation": (nested_payload or {}).get("trusted_interior_identity_evaluation"),
        "nested_aoi_contract_evaluation": nested_contract_payload,
    }
    summary_path = Path(cfg.out_dir) / "postrun_regression_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report.setdefault("outputs", {})["postrun_regression_summary"] = str(summary_path)
