from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_TEXTLIKE_EXTS = {".json", ".jsonl", ".log", ".md", ".txt", ".csv"}


def _resolve_path(value: Any, *, base_dir: Path) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        p = Path(value)
        return p if p.is_absolute() else (base_dir / p).resolve()
    except Exception:
        return None


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))




def _run_log_suffix(run_id: Optional[str]) -> Optional[str]:
    if run_id is None:
        return None
    suffix = str(run_id).strip()
    if suffix.startswith("bathy_"):
        suffix = suffix[len("bathy_"):]
    return suffix or None

def _write_index_readme(bundle_dir: Path) -> None:
    text = (
        "Debug bundle\n"
        "============\n\n"
        "This folder collects the most useful logs, summaries, reports, and contracts for a run in one place.\n\n"
        "Suggested order for diagnosis:\n"
        "1. run_logs/screen_*.log\n"
        "2. run_logs/run_summary_*.json\n"
        "3. run_logs/flight_recorder_*.jsonl\n"
        "4. run_logs/run_*.log\n"
        "5. reports/ and contracts/\n"
        "6. reports/WORKFLOW_INPUT_OUTPUT_TRACE.txt and reports/WORKFLOW_EXPLANATION_REPORT.txt\n7. manifest.json for copied files and spatial-output references\n"
    )
    (bundle_dir / "README.txt").write_text(text, encoding="utf-8")


def write_debug_bundle(
    out_dir: Path | str,
    *,
    report: Dict[str, Any],
    run_id: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Collect the most useful debug/analysis outputs into one location.

    Copies small diagnostic files into <out_dir>/debug_bundle and writes a manifest
    with references to larger spatial products that are better inspected in place.
    """
    log = logger or logging.getLogger(__name__)
    out_dir = Path(out_dir)
    bundle_dir = out_dir / "debug_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied: List[Dict[str, str]] = []
    missing: List[Dict[str, str]] = []
    seen_src: set[Path] = set()

    def add_copy(src: Optional[Path], rel_dst: str, *, label: str) -> None:
        if src is None:
            missing.append({"label": label, "reason": "path_unset"})
            return
        try:
            src = src.resolve()
        except Exception:
            missing.append({"label": label, "reason": "unresolvable_path", "path": str(src)})
            return
        if not src.exists() or not src.is_file():
            missing.append({"label": label, "reason": "missing_file", "path": str(src)})
            return
        if src.suffix.lower() not in _TEXTLIKE_EXTS:
            missing.append({"label": label, "reason": "non_textlike_not_copied", "path": str(src)})
            return
        if src in seen_src:
            return
        dst = bundle_dir / rel_dst
        _copy_file(src, dst)
        copied.append({"label": label, "src": str(src), "dst": str(dst)})
        seen_src.add(src)

    # Run logs for this run. Use only the current naming contract so the bundle
    # does not create noisy "missing legacy file" entries on every modern run.
    run_logs_dir = out_dir / "run_logs"
    run_log_suffix = _run_log_suffix(run_id)
    if run_log_suffix:
        add_copy(run_logs_dir / f"screen_bathy_{run_log_suffix}.log", f"run_logs/screen_bathy_{run_log_suffix}.log", label="screen_log")
        add_copy(run_logs_dir / f"run_bathy_{run_log_suffix}.log", f"run_logs/run_bathy_{run_log_suffix}.log", label="run_log")
        add_copy(run_logs_dir / f"flight_recorder_bathy_{run_log_suffix}.jsonl", f"run_logs/flight_recorder_bathy_{run_log_suffix}.jsonl", label="flight_recorder")
        add_copy(run_logs_dir / f"run_summary_bathy_{run_log_suffix}.json", f"run_logs/run_summary_bathy_{run_log_suffix}.json", label="run_summary_json")

    # Core whole-run reports.
    add_copy(out_dir / "bathy_report.json", "reports/bathy_report.json", label="bathy_report")
    add_copy(out_dir / "unified_bathy_report.json", "reports/unified_bathy_report.json", label="unified_bathy_report")
    add_copy(out_dir / "io_manifest.json", "reports/io_manifest.json", label="io_manifest")
    add_copy(out_dir / "io_manifest.md", "reports/io_manifest.md", label="io_manifest_md")
    add_copy(out_dir / "guidance_manifest.json", "reports/guidance_manifest.json", label="guidance_manifest")

    # Useful receipts and contracts from the report.
    river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river"), dict) else {}
    top_outputs = report.get("outputs", {}) if isinstance(report.get("outputs"), dict) else {}
    final_reporting = report.get("final_reporting", {}).get("receipts", {}) if isinstance(report.get("final_reporting"), dict) else {}
    auth_section = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base"), dict) else {}
    auth_river = auth_section.get("river_guidance", {}) if isinstance(auth_section.get("river_guidance"), dict) else {}

    candidate_paths: List[tuple[str, Optional[Path], str]] = [
        ("river_guidance_manifest", _resolve_path(river_outputs.get("guidance_manifest"), base_dir=out_dir), "contracts/river_guidance_manifest.json"),
        ("river_render_mode_summary", _resolve_path(river_outputs.get("river_render_mode_summary"), base_dir=out_dir), "contracts/river_render_mode_summary.csv"),
        ("river_render_mode_selector_summary", _resolve_path(river_outputs.get("river_render_mode_selector_summary"), base_dir=out_dir), "contracts/river_render_mode_selector_summary.csv"),
        ("river_render_mode_selector_summary_json", _resolve_path(river_outputs.get("river_render_mode_selector_summary_json"), base_dir=out_dir), "contracts/river_render_mode_selector_summary.json"),
        ("channel_surface_effect_summary", _resolve_path(river_outputs.get("channel_surface_effect_summary"), base_dir=out_dir), "contracts/river_channel_surface_effect_summary.json"),
        ("channel_surface_effect_profile", _resolve_path(river_outputs.get("channel_surface_effect_profile"), base_dir=out_dir), "contracts/river_channel_surface_effect_summary.csv"),
        ("channel_surface_role_agreement_summary", _resolve_path(river_outputs.get("channel_surface_role_agreement_summary"), base_dir=out_dir), "contracts/river_channel_surface_role_agreement_summary.json"),
        ("channel_surface_role_agreement_profile", _resolve_path(river_outputs.get("channel_surface_role_agreement_profile"), base_dir=out_dir), "contracts/river_channel_surface_role_agreement_profile.csv"),
        ("channel_surface_section_target_agreement_summary", _resolve_path(river_outputs.get("channel_surface_section_target_agreement_summary"), base_dir=out_dir), "contracts/river_channel_surface_section_target_agreement_summary.json"),
        ("channel_surface_section_target_agreement_profile", _resolve_path(river_outputs.get("channel_surface_section_target_agreement_profile"), base_dir=out_dir), "contracts/river_channel_surface_section_target_agreement_profile.csv"),
        ("channel_surface_authoritative_transition_summary", _resolve_path(river_outputs.get("channel_surface_authoritative_transition_summary"), base_dir=out_dir), "contracts/river_channel_surface_authoritative_transition_summary.json"),
        ("channel_surface_longitudinal_smoothing_summary", _resolve_path(river_outputs.get("channel_surface_longitudinal_smoothing_summary"), base_dir=out_dir), "contracts/river_channel_surface_longitudinal_smoothing_summary.json"),
        ("channel_surface_longitudinal_smoothing_profile", _resolve_path(river_outputs.get("channel_surface_longitudinal_smoothing_profile"), base_dir=out_dir), "contracts/river_channel_surface_longitudinal_smoothing_summary.csv"),
        ("backbone_smoothing_summary", _resolve_path(river_outputs.get("backbone_smoothing_summary"), base_dir=out_dir), "contracts/river_backbone_smoothing_summary.json"),
        ("backbone_smoothing_profile", _resolve_path(river_outputs.get("backbone_smoothing_profile"), base_dir=out_dir), "contracts/river_backbone_smoothing_profile.csv"),
        ("channel_frame_contract", _resolve_path(river_outputs.get("channel_frame_contract"), base_dir=out_dir), "contracts/river_channel_frame_contract.json"),
        ("longitudinal_profile_summary", _resolve_path(river_outputs.get("longitudinal_profile_summary"), base_dir=out_dir), "contracts/river_longitudinal_profile_summary.json"),
        ("longitudinal_profile_coverage", _resolve_path(river_outputs.get("longitudinal_profile_coverage"), base_dir=out_dir), "contracts/river_longitudinal_profile_coverage.csv"),
        ("presentation_figures_summary", _resolve_path(top_outputs.get("presentation_figures_summary"), base_dir=out_dir), "reports/presentation_figures_summary.json"),
        ("workflow_input_output_trace", _resolve_path(top_outputs.get("workflow_input_output_trace"), base_dir=out_dir), "reports/WORKFLOW_INPUT_OUTPUT_TRACE.txt"),
        ("workflow_stage_trace_json", _resolve_path(top_outputs.get("workflow_stage_trace_json"), base_dir=out_dir), "reports/workflow_stage_trace.json"),
        ("workflow_stage_trace_jsonl", _resolve_path(top_outputs.get("workflow_stage_trace_jsonl"), base_dir=out_dir), "reports/workflow_stage_trace_lines.jsonl"),
        ("workflow_file_graph_json", _resolve_path(top_outputs.get("workflow_file_graph_json"), base_dir=out_dir), "reports/workflow_file_graph.json"),
        ("workflow_explanation_report", _resolve_path(top_outputs.get("workflow_explanation_report"), base_dir=out_dir), "reports/WORKFLOW_EXPLANATION_REPORT.txt"),
        ("workflow_accuracy_anomalies_json", _resolve_path(top_outputs.get("workflow_accuracy_anomalies_json"), base_dir=out_dir), "reports/workflow_accuracy_anomalies.json"),
        ("workflow_first_bad_artifact_summary_json", _resolve_path(top_outputs.get("workflow_first_bad_artifact_summary_json"), base_dir=out_dir), "reports/workflow_first_bad_artifact_summary.json"),
        ("workflow_run_diagnosis_summary_json", _resolve_path(top_outputs.get("workflow_run_diagnosis_summary_json"), base_dir=out_dir), "reports/workflow_run_diagnosis_summary.json"),
        ("workflow_run_diagnosis_summary_txt", _resolve_path(top_outputs.get("workflow_run_diagnosis_summary_txt"), base_dir=out_dir), "reports/WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt"),
        ("reports_readme", _resolve_path(top_outputs.get("reports_readme"), base_dir=out_dir), "reports/README_FIRST.txt"),
        ("reports_manifest", _resolve_path(top_outputs.get("reports_manifest"), base_dir=out_dir), "reports/reports_manifest.json"),
        ("support_provenance_summary", _resolve_path(final_reporting.get("support_provenance_summary"), base_dir=out_dir), "reports/support_provenance_summary.json"),
        ("final_support_regime_audit", _resolve_path(final_reporting.get("final_support_regime_audit"), base_dir=out_dir), "reports/final_support_regime_audit.json"),
        ("final_dem_selection_receipt", _resolve_path(final_reporting.get("final_dem_selection_receipt"), base_dir=out_dir), "reports/final_dem_selection_receipt.json"),
        ("comparison_package", _resolve_path(final_reporting.get("comparison_package"), base_dir=out_dir), "reports/comparison_package.json"),
        ("authoritative_river_support_csv", _resolve_path(auth_river.get("path"), base_dir=out_dir), "contracts/authoritative_river_support.csv"),
        ("authoritative_river_role_contract", _resolve_path(auth_river.get("role_contract"), base_dir=out_dir), "contracts/authoritative_river_role_contract.json"),
        ("benchmark_summary_json", _resolve_path(top_outputs.get("benchmark_summary_json"), base_dir=out_dir), "reports/benchmark_summary.json"),
        ("benchmark_table_csv", _resolve_path(top_outputs.get("benchmark_table_csv"), base_dir=out_dir), "reports/benchmark_table.csv"),
        ("benchmark_scored_points_csv", _resolve_path(top_outputs.get("benchmark_scored_points_csv"), base_dir=out_dir), "reports/benchmark_holdout_scored.csv"),
        ("benchmark_support_aware_validation_summary_json", _resolve_path(top_outputs.get("benchmark_support_aware_validation_summary_json"), base_dir=out_dir), "reports/benchmark_support_aware_validation_summary.json"),
        ("benchmark_hard_river_summary_json", _resolve_path(top_outputs.get("benchmark_hard_river_summary_json"), base_dir=out_dir), "reports/benchmark_hard_river_summary.json"),
        ("benchmark_authoritative_role_validation_summary_json", _resolve_path(top_outputs.get("benchmark_authoritative_role_validation_summary_json"), base_dir=out_dir), "reports/benchmark_authoritative_role_validation_summary.json"),
        ("benchmark_river_science_summary_json", _resolve_path(top_outputs.get("benchmark_river_science_summary_json"), base_dir=out_dir), "reports/benchmark_river_science_summary.json"),
        ("benchmark_river_receipt_triage_summary_json", _resolve_path(top_outputs.get("benchmark_river_receipt_triage_summary_json"), base_dir=out_dir), "reports/benchmark_river_receipt_triage_summary.json"),
        ("benchmark_river_primary_focus_summary_json", _resolve_path(top_outputs.get("benchmark_river_primary_focus_summary_json"), base_dir=out_dir), "reports/benchmark_river_primary_focus_summary.json"),
    ]
    for label, src, rel_dst in candidate_paths:
        add_copy(src, rel_dst, label=label)

    # Create a reference list for larger spatial outputs that are useful but should not be duplicated here.
    spatial_reference_keys = [
        "selected_final",
        "selected_final_provenance",
        "support_class",
        "conditioned_difference",
        "comparison_hillshade",
        "selected_final_hillshade",
        "presentation_support_class_map",
        "presentation_baseline_vs_enhanced_hillshade",
        "presentation_difference_map_locked_preserved",
        "presentation_river_guidance_construction",
        "authoritative_role_code_raster",
        "authoritative_role_confidence_raster",
        "authoritative_distance_to_bank_raster",
        "authoritative_normalized_channel_position_raster",
    ]
    spatial_references: List[Dict[str, str]] = []
    for key in spatial_reference_keys:
        val = top_outputs.get(key)
        rp = _resolve_path(val, base_dir=out_dir)
        if rp is not None:
            spatial_references.append({"label": key, "path": str(rp)})
    for key in ("role_code_raster", "role_confidence_raster", "distance_to_bank_raster", "normalized_channel_position_raster"):
        rp = _resolve_path(auth_river.get(key), base_dir=out_dir)
        if rp is not None:
            spatial_references.append({"label": f"authoritative_river_{key}", "path": str(rp)})

    referenced_receipts = [{"label": label, "path": str(src) if src is not None else None, "bundle_rel_dst": rel_dst} for label, src, rel_dst in candidate_paths]
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "bundle_dir": str(bundle_dir),
        "copied_files": copied,
        "missing_or_skipped": missing,
        "referenced_receipts": referenced_receipts,
        "spatial_output_references": spatial_references,
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_index_readme(bundle_dir)
    log.info("[DEBUG_BUNDLE] Wrote debug bundle: %s", bundle_dir)
    return bundle_dir
