from __future__ import annotations

import ast
import json
import re
import csv
from pathlib import Path
from typing import Any, Iterable

DEBUG_FILENAME = "RIVER_WORKFLOW_DEBUG.txt"

_CONSOLIDATED_RIVER_DEBUG_ARTIFACTS = [
    "WORKFLOW_INPUT_OUTPUT_TRACE.txt",
    "workflow_stage_trace.json",
    "workflow_stage_trace_lines.jsonl",
    "workflow_file_graph.json",
    "WORKFLOW_EXPLANATION_REPORT.txt",
    "workflow_accuracy_anomalies.json",
    "workflow_first_bad_artifact_summary.json",
    "workflow_run_diagnosis_summary.json",
    "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt",
    "reports/README_FIRST.txt",
    "reports/reports_manifest.json",
    "reports/river_active_runtime_summary.json",
    "reports/river_active_shape_summary.json",
    "reports/river_active_summary.json",
    "reports/benchmark_river_active_evaluation_summary.json",
    "reports/benchmark_river_withheld_support_receipt.json",
    "reports/benchmark_river_mode_summary.json",
    "reports/benchmark_river_receipt_triage_summary.json",
    "reports/benchmark_river_primary_focus_summary.json",
    "reports/river_primary_guidance_summary.json",
    "reports/river_primary_surface_contract.json",
    "combined/river_primary_guidance_summary.json",
    "combined/river_primary_surface_contract.json",
    "combined/final_route_terrain_receipt.json",
    "combined/river_primary_surface_uptake_receipt.json",
    "combined/river_primary_surface_conditioning_effect_receipt.json",
    "reports/river_backbone_smoothing_summary.json",
    "reports/river_centerline_width_propagation_summary.json",
    "reports/river_role_agreement_summary.json",
    "reports/river_section_target_agreement_summary.json",
    "reports/river_effect_summary.json",
    "reports/river_transition_summary.json",
    "debug_bundle",
    "bathy_report.json",
    "io_manifest.json",
    "io_manifest.md",
    "final_outputs.json",
    "guidance_manifest.json",
    "postrun_regression_summary.json",
]



_CONSOLIDATED_DEBUG_OUTPUT_KEYS = (
    "river_primary_guidance_summary",
    "river_primary_surface_contract",
    "terrain_conditioning_receipt",
    "river_primary_surface_uptake_receipt",
    "river_primary_surface_conditioning_effect_receipt",
)


def prune_consolidated_river_debug_outputs_from_report(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        return
    outputs = report.get("outputs")
    if isinstance(outputs, dict):
        for key in _CONSOLIDATED_DEBUG_OUTPUT_KEYS:
            outputs.pop(key, None)
    river = report.get("river")
    if isinstance(river, dict):
        river_outputs = river.get("outputs")
        if isinstance(river_outputs, dict):
            for key in _CONSOLIDATED_DEBUG_OUTPUT_KEYS:
                river_outputs.pop(key, None)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}




def _load_first_csv_row(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return dict(row)
    except (OSError, csv.Error, ValueError, TypeError):
        return {}
    return {}


def _resolve_path(value: Any, *, base_dir: Path) -> Path | None:
    if not value:
        return None
    try:
        p = Path(str(value))
    except Exception:
        return None
    if not p.is_absolute():
        p = base_dir / p
    return p


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_fmt(v) for v in value) if value else "None"
    return str(value)


def _iter_step_lines(steps: dict[str, Any], *, prefix: str = "") -> Iterable[str]:
    for key in sorted(steps):
        value = steps.get(key)
        label = f"{prefix}{key}"
        if isinstance(value, dict):
            status = value.get("status")
            detail_parts: list[str] = []
            for field in (
                "status_family",
                "reason",
                "message",
                "error",
                "exception_type",
                "mode",
                "stage",
            ):
                if value.get(field) not in (None, "", [], {}):
                    detail_parts.append(f"{field}={_fmt(value.get(field))}")
            yield f"- {label}: {status or 'n/a'}" + (f" ({'; '.join(detail_parts)})" if detail_parts else "")
            nested = {k: v for k, v in value.items() if isinstance(v, dict)}
            if nested:
                yield from _iter_step_lines(nested, prefix=f"{label}.")
        else:
            yield f"- {label}: {_fmt(value)}"


def _append_key_values(lines: list[str], title: str, mapping: dict[str, Any], *, keys: Iterable[str] | None = None) -> None:
    lines.append(title)
    if not isinstance(mapping, dict) or not mapping:
        lines.append("- None")
        lines.append("")
        return
    selected = list(keys) if keys is not None else sorted(mapping)
    wrote = False
    for key in selected:
        if key not in mapping:
            continue
        value = mapping.get(key)
        if value in (None, "", [], {}):
            continue
        lines.append(f"- {key}: {_fmt(value)}")
        wrote = True
    if not wrote:
        lines.append("- None")
    lines.append("")


def _find_latest_screen_log(out_dir: Path) -> Path | None:
    run_logs = out_dir / "run_logs"
    if not run_logs.exists():
        return None
    logs = sorted(run_logs.glob("screen_*.log"))
    return logs[-1] if logs else None


def _find_first_mapping(*candidates: Any) -> dict[str, Any]:
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _parse_mapping_from_log_line(line: str) -> dict[str, Any]:
    match = re.search(r"(\{.*\})", line)
    if not match:
        return {}
    try:
        payload = ast.literal_eval(match.group(1))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_screen_log_receipts(screen_log: Path | None) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    if screen_log is None or not screen_log.exists():
        return receipts
    try:
        lines = screen_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return receipts
    for line in lines:
        if "terrain_interpolator_inputs_sanitized" in line:
            payload = _parse_mapping_from_log_line(line)
            if payload:
                receipts["inputs_sanitized"] = payload
        elif "terrain_interpolator_end" in line:
            payload = _parse_mapping_from_log_line(line)
            if payload:
                receipts["terrain_end"] = payload
        elif "[RIVER][V1] Minimal surface artifacts complete:" in line:
            receipts["v1_artifacts"] = {"line": line}
        elif "[DOMAIN] Shared-domain activation:" in line:
            receipts["shared_domain_activation"] = {"line": line}
        elif "[AUTHORITATIVE] Filled" in line and "river DEM cells" in line:
            receipts["river_dem_gapfill"] = {"line": line}
    return receipts


def _line_tail(mapping: dict[str, Any]) -> str | None:
    line = mapping.get("line")
    if not isinstance(line, str):
        return None
    parts = line.split(": ", 1)
    return parts[1] if len(parts) == 2 else line


def cleanup_consolidated_river_debug_outputs(*, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    for rel in _CONSOLIDATED_RIVER_DEBUG_ARTIFACTS:
        path = out_dir / rel
        try:
            if path.is_dir():
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_file() or child.is_symlink():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
                path.rmdir()
            elif path.exists():
                path.unlink()
        except OSError:
            continue


def write_river_workflow_debug_report(*, out_dir: str | Path, report: dict[str, Any]) -> str:
    out_dir = Path(out_dir)
    output_path = out_dir / DEBUG_FILENAME
    outputs = report.get("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}
    river = report.get("river", {}) if isinstance(report.get("river", {}), dict) else {}
    river_outputs = river.get("outputs", {}) if isinstance(river.get("outputs", {}), dict) else {}
    river_steps = river.get("steps", {}) if isinstance(river.get("steps", {}), dict) else {}
    execution_receipts = river.get("execution_receipts", {}) if isinstance(river.get("execution_receipts", {}), dict) else {}
    benchmark = report.get("benchmark", {}) if isinstance(report.get("benchmark", {}), dict) else {}
    final_route = report.get("final_dem_route", {}) if isinstance(report.get("final_dem_route", {}), dict) else {}
    human_summary = report.get("human_summary", {}) if isinstance(report.get("human_summary", {}), dict) else {}

    screen_log = _find_latest_screen_log(out_dir)
    screen_receipts = _extract_screen_log_receipts(screen_log)

    guidance_path = _resolve_path(outputs.get("river_primary_guidance_summary") or river_outputs.get("river_primary_guidance_summary"), base_dir=out_dir)
    contract_path = _resolve_path(outputs.get("river_primary_surface_contract") or river_outputs.get("river_primary_surface_contract"), base_dir=out_dir)
    route_receipt_path = _resolve_path(outputs.get("final_route_receipt") or final_route.get("final_route_receipt"), base_dir=out_dir)
    terrain_receipt_path = _resolve_path(river_outputs.get("terrain_conditioning_receipt") or outputs.get("terrain_conditioning_receipt"), base_dir=out_dir)
    uptake_receipt_path = _resolve_path(river_outputs.get("river_primary_surface_uptake_receipt") or outputs.get("river_primary_surface_uptake_receipt"), base_dir=out_dir)
    effect_receipt_path = _resolve_path(river_outputs.get("river_primary_surface_conditioning_effect_receipt") or outputs.get("river_primary_surface_conditioning_effect_receipt"), base_dir=out_dir)
    xs_bank_qc_summary_path = _resolve_path(river_outputs.get("xs_bank_qc_summary") or outputs.get("xs_bank_qc_summary"), base_dir=out_dir)
    xs_bank_qc_summary = _load_first_csv_row(xs_bank_qc_summary_path)

    guidance = _find_first_mapping(
        report.get("river_primary_guidance_summary"),
        _load_json(guidance_path),
    )
    contract = _find_first_mapping(
        report.get("river_primary_surface_contract"),
        _load_json(contract_path),
    )
    route_receipt = _find_first_mapping(
        report.get("final_route_receipt"),
        _load_json(route_receipt_path),
    )
    terrain_receipt = _find_first_mapping(
        report.get("terrain_conditioning_receipt"),
        _load_json(terrain_receipt_path),
    )
    uptake_receipt = _find_first_mapping(
        report.get("river_primary_surface_uptake_receipt"),
        _load_json(uptake_receipt_path),
    )
    effect_receipt = _find_first_mapping(
        report.get("river_primary_surface_conditioning_effect_receipt"),
        _load_json(effect_receipt_path),
    )

    terrain_end = screen_receipts.get("terrain_end", {})
    terrain_inputs = screen_receipts.get("inputs_sanitized", {})
    if not isinstance(terrain_receipt.get("result_stats"), dict):
        terrain_receipt = {
            **terrain_receipt,
            "result_stats": {
                "locked_pixels": terrain_end.get("authoritative_locked_pixels"),
                "gap_pixels": terrain_end.get("gap_pixels"),
                "eligible_pixels": terrain_end.get("eligible_pixels"),
                "conditioned_finite_pixels": terrain_end.get("conditioned_finite"),
            },
            "structural_inputs": {
                "river_guidance_finite_after_handoff": terrain_inputs.get("river_guidance_finite"),
                "sdb_guidance_finite_after_handoff": terrain_inputs.get("sdb_guidance_finite"),
                "authoritative_finite": terrain_inputs.get("auth_finite"),
            },
        }
    if not uptake_receipt:
        uptake_receipt = {
            "status": None,
            "eligible_pixels": terrain_end.get("eligible_pixels"),
            "gap_pixels": terrain_end.get("gap_pixels"),
            "conditioned_finite_pixels": terrain_end.get("conditioned_finite"),
            "river_guidance_take_domain_pixels": terrain_end.get("river_guidance_take_domain_pixels"),
            "river_guidance_taken_pixels": terrain_end.get("river_guidance_taken_pixels"),
            "background_taken_pixels": terrain_end.get("background_taken_pixels"),
            "authoritative_locked_pixels": terrain_end.get("authoritative_locked_pixels"),
            "river_guidance_finite_after_handoff": terrain_inputs.get("river_guidance_finite"),
        }
    if not guidance and terrain_inputs:
        guidance = {
            "primary_surface_finite_pixels": terrain_inputs.get("river_guidance_finite"),
            "support_note": "derived_from_screen_log",
        }

    dominant_issue = None
    next_action = None
    if int(terrain_inputs.get("river_guidance_finite") or 0) > 0 and int(terrain_end.get("eligible_pixels") or 0) == 0:
        dominant_issue = "river guidance reached conditioning but every pixel was rejected from eligibility"
        next_action = "debug river take domain / canonical direct-primary eligibility path"
    else:
        for candidate in (guidance, terrain_receipt, uptake_receipt, effect_receipt, benchmark):
            if not dominant_issue and isinstance(candidate, dict):
                dominant_issue = candidate.get("dominant_remaining_issue") or candidate.get("dominant_runtime_issue") or candidate.get("status")
                next_action = candidate.get("suggested_next_action") or candidate.get("next_action")

    lines: list[str] = []
    lines.append("RIVER WORKFLOW DEBUG REPORT")
    lines.append("===========================")
    lines.append("")
    lines.append("Use this together with the screen log. This file is the single consolidated river debug receipt for the run.")
    lines.append("It is designed to stand on its own after the extra JSON/manifest outputs are removed.")
    lines.append("")
    lines.append("Run overview")
    lines.append(f"- output_dir: {out_dir}")
    lines.append(f"- screen_log: {screen_log if screen_log else 'None'}")
    lines.append(f"- command: {_fmt(human_summary.get('command'))}")
    lines.append(f"- aoi: {_fmt(human_summary.get('aoi'))}")
    time_window = human_summary.get("time_window") if isinstance(human_summary.get("time_window"), dict) else {}
    lines.append(f"- time_window: {_fmt(time_window.get('start'))} -> {_fmt(time_window.get('end'))}")
    lines.append(f"- river_status: {_fmt(river.get('status'))}")
    lines.append(f"- river_execution_mode: {_fmt(river.get('execution_mode'))}")
    lines.append(f"- final_status: {_fmt(report.get('status'))}")
    lines.append(f"- final_output: {_fmt(outputs.get('final_output') or outputs.get('combined_warped') or outputs.get('combined'))}")
    lines.append("")

    lines.append("What happened")
    lines.append(f"- dominant_issue: {_fmt(dominant_issue)}")
    lines.append(f"- suggested_next_action: {_fmt(next_action)}")
    shared_domain = _line_tail(screen_receipts.get("shared_domain_activation", {}))
    if shared_domain:
        lines.append(f"- shared_domain_activation: {shared_domain}")
    river_gapfill = _line_tail(screen_receipts.get("river_dem_gapfill", {}))
    if river_gapfill:
        lines.append(f"- river_dem_gapfill: {river_gapfill}")
    lines.append("")

    lines.append("Outcome summary")
    lines.append(f"- river_guidance_finite: {_fmt(terrain_inputs.get('river_guidance_finite'))}")
    lines.append(f"- eligible_pixels: {_fmt(terrain_end.get('eligible_pixels'))}")
    lines.append(f"- river_guidance_take_domain_pixels: {_fmt(terrain_end.get('river_guidance_take_domain_pixels'))}")
    lines.append(f"- river_guidance_taken_pixels: {_fmt(terrain_end.get('river_guidance_taken_pixels'))}")
    lines.append(f"- background_taken_pixels: {_fmt(terrain_end.get('background_taken_pixels'))}")
    lines.append(f"- authoritative_locked_pixels: {_fmt(terrain_end.get('authoritative_locked_pixels'))}")
    lines.append(f"- canonical_bank_shaped_pixels: {_fmt((terrain_end.get('canonical_bank_shaped_pixels') if isinstance(terrain_end, dict) else None) or (guidance.get('canonical_bank_shaped_pixels') if isinstance(guidance, dict) else None))}")
    lines.append("")

    _append_key_values(lines, "Primary river runtime summary", guidance, keys=(
        "active_product_name",
        "primary_builder_mode",
        "primary_input_surface_source",
        "dominant_primary_source_class",
        "primary_surface_finite_pixels",
        "primary_domain_pixels",
        "primary_surface_contract_ok",
        "degraded_mode_active",
        "degraded_mode_reasons",
        "continuity_safeguard_used",
        "continuity_safeguard_labels",
        "canonical_direct_primary_mode",
        "canonical_direct_primary_single_path",
        "canonical_direct_primary_weighting_single_path",
        "canonical_direct_primary_downstream_sidepaths_bypassed",
        "canonical_direct_primary_single_write_path",
        "legacy_structured_take_bypassed",
        "primary_builder_inputs_simplified",
        "canonical_bank_input_finite_pixels",
        "canonical_bank_influence_active_pixels",
        "canonical_bank_shaped_pixels",
        "canonical_bank_shaping_active",
        "support_note",
    ))

    _append_key_values(lines, "Primary river surface contract", contract, keys=(
        "ok",
        "source_name",
        "domain_pixels",
        "finite_pixels",
        "domain_finite_pixels",
        "source_pixels",
        "failures",
    ))

    terrain_stats = terrain_receipt.get("result_stats") if isinstance(terrain_receipt.get("result_stats"), dict) else {}
    terrain_structural = terrain_receipt.get("structural_inputs") if isinstance(terrain_receipt.get("structural_inputs"), dict) else {}
    _append_key_values(lines, "Terrain conditioning summary", terrain_stats, keys=(
        "locked_pixels",
        "gap_pixels",
        "eligible_pixels",
        "conditioned_finite_pixels",
        "channel_core_preserve_pixels",
        "channel_core_zone_pixels",
        "channel_core_abs_p95_m",
        "channel_core_bank_pull_risk_p95",
    ))
    _append_key_values(lines, "Terrain handoff summary", terrain_structural, keys=(
        "primary_river_guidance_surface_finite",
        "river_guidance_finite_after_handoff",
        "primary_surface_finite_after_handoff",
        "sdb_guidance_finite_after_handoff",
        "authoritative_finite",
        "baseline_cudem_interpolation",
        "river_guide_points",
    ))
    _append_key_values(lines, "Terrain uptake receipt", uptake_receipt, keys=(
        "status",
        "primary_river_guidance_surface_finite",
        "river_guidance_finite_after_handoff",
        "primary_surface_finite_after_handoff",
        "eligible_pixels",
        "gap_pixels",
        "conditioned_finite_pixels",
        "river_guidance_take_domain_pixels",
        "river_guidance_taken_pixels",
        "background_taken_pixels",
        "authoritative_locked_pixels",
    ))
    _append_key_values(lines, "Terrain conditioning effect receipt", effect_receipt)

    lines.append("River steps executed")
    if river_steps:
        lines.extend(_iter_step_lines(river_steps))
    else:
        lines.append("- None")
    lines.append("")

    _append_key_values(lines, "River execution receipts", execution_receipts)

    _append_key_values(lines, "Bank and WSE guidance usage", {
        "canonical_bank_input_finite_pixels": guidance.get("canonical_bank_input_finite_pixels") if isinstance(guidance, dict) else None,
        "canonical_bank_influence_active_pixels": guidance.get("canonical_bank_influence_active_pixels") if isinstance(guidance, dict) else None,
        "canonical_bank_shaped_pixels": guidance.get("canonical_bank_shaped_pixels") if isinstance(guidance, dict) else None,
        "canonical_bank_shaping_active": guidance.get("canonical_bank_shaping_active") if isinstance(guidance, dict) else None,
    })

    _append_key_values(lines, "Smoothed bank QC summary", xs_bank_qc_summary, keys=(
        "bank_replaced_count",
        "high_bank_suspect_count",
        "strong_contamination_count",
        "bank_clamp_count",
        "bank_replace_with_local_envelope_count",
        "bank_reject_count",
        "bank_monotone_applied_count",
        "bank_monotone_adjustment_max_m",
    ))

    key_inputs = {
        "authoritative_base": (((report.get("authoritative_base") or {}).get("outputs") or {}).get("authoritative_base")),
        "river_dem": river_outputs.get("river_dem") or outputs.get("river_dem"),
        "river_network": river_outputs.get("river_network") or outputs.get("river_network"),
        "river_guidance_domain_mask": river_outputs.get("guidance_domain_mask") or outputs.get("river_guidance_domain_mask"),
        "river_primary_surface": river_outputs.get("river_primary_surface") or outputs.get("river_primary_surface") or outputs.get("river_warped"),
        "river_support_points": river_outputs.get("river_support_points"),
        "river_centerline_points": river_outputs.get("river_centerline_points"),
        "river_centerline_elevation": river_outputs.get("centerline_elevation") or outputs.get("centerline_elevation"),
        "river_bank_points": river_outputs.get("bank_points") or outputs.get("bank_points"),
        "river_bank_qc_points": river_outputs.get("xs_bank_qc_points") or outputs.get("xs_bank_qc_points"),
        "river_bank_qc_summary": river_outputs.get("xs_bank_qc_summary") or outputs.get("xs_bank_qc_summary"),
        "river_bank_elevation_xs": river_outputs.get("bank_elevation_xs") or outputs.get("bank_elevation_xs"),
        "river_bank_influence": river_outputs.get("bank_influence") or outputs.get("bank_influence"),
        "combined_conditioned_raster": outputs.get("combined_warped") or outputs.get("final_output"),
        "combined_hillshade": outputs.get("combined_hillshade"),
    }
    _append_key_values(lines, "Key river artifacts", key_inputs)

    route_artifacts = route_receipt.get("artifact_roles") if isinstance(route_receipt.get("artifact_roles"), dict) else {}
    _append_key_values(lines, "Final route artifact roles", route_artifacts)

    _append_key_values(lines, "Benchmark summary", benchmark, keys=(
        "status",
        "requested",
        "active_river_benchmark_mode",
        "river_specific_withheld_support_benchmark_recommended",
        "hard_problem_holdout_blind",
        "error",
    ))

    lines.append("Failure and warning context")
    fatal_errors = report.get("fatal_errors") if isinstance(report.get("fatal_errors"), list) else []
    postrun_failures = report.get("postrun_failures") if isinstance(report.get("postrun_failures"), list) else []
    if fatal_errors:
        for entry in fatal_errors:
            lines.append(f"- fatal_error: {_fmt(entry)}")
    if postrun_failures:
        for entry in postrun_failures:
            lines.append(f"- postrun_failure: {_fmt(entry)}")
    if not fatal_errors and not postrun_failures:
        lines.append("- None recorded in final report")
    lines.append("")

    lines.append("Raw river report excerpts")
    excerpt = {
        "status": river.get("status"),
        "execution_mode": river.get("execution_mode"),
        "execution_plan": river.get("execution_plan"),
        "notes": river.get("notes"),
    }
    lines.append(json.dumps(excerpt, indent=2, sort_keys=True))
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cleanup_consolidated_river_debug_outputs(out_dir=out_dir)
    return str(output_path)

