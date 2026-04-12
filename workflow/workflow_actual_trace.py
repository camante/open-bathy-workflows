from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from io_artifacts import is_probably_path
from constants import PIPELINE_VERSION


TRACE_DIRNAME = "reports"
TRACE_FILENAME = "WORKFLOW_INPUT_OUTPUT_TRACE.txt"


SECTION_ORDER = [
    ("workflow_execution_state", "Workflow execution state"),
    ("authoritative_base", "Authoritative base"),
    ("authoritative_base_auto", "Authoritative base auto"),
    ("shared_domain_stage", "Shared guidance-domain stage"),
    ("guidance_domains", "Guidance domains"),
    ("river", "River workflow"),
    ("sdb", "SDB workflow"),
    ("fusion", "Fusion"),
    ("final_dem_route", "Final DEM route"),
    ("final_dem_runtime", "Final DEM runtime"),
    ("seams", "Seam diagnostics"),
    ("benchmark", "Benchmark"),
    ("postrun_regression", "Postrun regression"),
    ("final_reporting", "Final reporting receipts"),
    ("validation", "Validation"),
    ("outputs", "Top-level outputs"),
]


SKIP_KEYS = {
    "command",
    "stdout",
    "stderr",
    "message",
    "note",
    "notes",
    "reason",
    "description",
    "summary",
    "details",
    "selection_reason",
}


def _looks_like_path(value: Any) -> bool:
    return isinstance(value, str) and is_probably_path(value)



def _iter_paths(obj: Any, prefix: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            keypath = f"{prefix}.{k}" if prefix else str(k)
            if _looks_like_path(v):
                yield keypath, v
            elif isinstance(v, (dict, list, tuple)) and k not in SKIP_KEYS:
                yield from _iter_paths(v, keypath)
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            keypath = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            if _looks_like_path(v):
                yield keypath, v
            elif isinstance(v, (dict, list, tuple)):
                yield from _iter_paths(v, keypath)



def _dedupe_pairs(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out



def _load_io_manifest(out_dir: Path) -> Dict[str, Any] | None:
    p = out_dir / "io_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None



def build_workflow_actual_trace_text(*, out_dir: str | Path, report: Dict[str, Any]) -> str:
    out_dir = Path(out_dir)
    io_manifest = _load_io_manifest(out_dir)
    lines: List[str] = []
    lines.append("WORKFLOW INPUT/OUTPUT TRACE")
    lines.append(f"created_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append(f"out_dir: {out_dir}")
    lines.append("")

    run_id = report.get("run_id") or (report.get("run") or {}).get("run_id")
    cfg = report.get("config") or {}
    lines.append("RUN CONTEXT")
    lines.append(f"- run_id: {run_id}")
    lines.append(f"- aoi: {cfg.get('aoi')}")
    lines.append(f"- start: {cfg.get('start') or cfg.get('start_date')}")
    lines.append(f"- end: {cfg.get('end') or cfg.get('end_date')}")
    lines.append(f"- pipeline_version: {PIPELINE_VERSION}")
    methods = cfg.get("methods")
    if methods is not None:
        lines.append(f"- requested_methods: {methods}")
    exec_state = report.get("workflow_execution_state") or {}
    final_outputs = exec_state.get("final_outputs") or {}
    if final_outputs:
        lines.append(f"- final_native: {final_outputs.get('final_native')}")
        lines.append(f"- final_for_user: {final_outputs.get('final_for_user')}")
        lines.append(f"- final_provenance: {final_outputs.get('final_provenance')}")
    final_dem_contract = report.get("final_dem_contract") if isinstance(report.get("final_dem_contract"), dict) else {}
    current_route_mode = final_dem_contract.get("current_route_mode") or exec_state.get("current_route_mode")
    target_route_mode = final_dem_contract.get("target_route_mode") or exec_state.get("target_route_mode")
    if current_route_mode:
        lines.append(f"- current_route_mode: {current_route_mode}")
    if target_route_mode:
        lines.append(f"- target_route_mode: {target_route_mode}")
    stage_status = final_dem_contract.get("simple_river_stage_status") if isinstance(final_dem_contract.get("simple_river_stage_status"), dict) else {}
    if stage_status:
        implemented = sum(1 for spec in stage_status.values() if isinstance(spec, dict) and spec.get("implemented"))
        total = len(stage_status)
        lines.append(f"- simple_river_stage_progress: {implemented}/{total} implemented")
    v2_pass4 = ((report.get("river") or {}).get("v2_pass4") if isinstance((report.get("river") or {}).get("v2_pass4"), dict) else {})
    v2_pass3 = ((report.get("river") or {}).get("v2_pass3") if isinstance((report.get("river") or {}).get("v2_pass3"), dict) else {})
    v2_pass2 = ((report.get("river") or {}).get("v2_pass2") if isinstance((report.get("river") or {}).get("v2_pass2"), dict) else {})
    v2_pass1 = ((report.get("river") or {}).get("v2_pass1") if isinstance((report.get("river") or {}).get("v2_pass1"), dict) else {})
    v2_stage_status = v2_pass4.get("stage_status") if isinstance(v2_pass4.get("stage_status"), dict) else (v2_pass3.get("stage_status") if isinstance(v2_pass3.get("stage_status"), dict) else (v2_pass2.get("stage_status") if isinstance(v2_pass2.get("stage_status"), dict) else (v2_pass1.get("stage_status") if isinstance(v2_pass1.get("stage_status"), dict) else {})))
    if v2_stage_status:
        implemented_v2 = sum(1 for spec in v2_stage_status.values() if isinstance(spec, dict) and spec.get("implemented"))
        total_v2 = len(v2_stage_status)
        progress_label = "river_v2_pass4_progress" if v2_pass4 else ("river_v2_pass3_progress" if v2_pass3 else ("river_v2_pass2_progress" if v2_pass2 else "river_v2_pass1_progress"))
        lines.append(f"- {progress_label}: {implemented_v2}/{total_v2} implemented")
    v2_final_route_contract = final_dem_contract.get("final_route_contract", {}).get("river_v2_final_route_contract") if isinstance(final_dem_contract.get("final_route_contract"), dict) else {}
    if isinstance(v2_final_route_contract, dict) and v2_final_route_contract:
        lines.append(f"- river_v2_final_route_active: {bool(v2_final_route_contract.get('active'))}")
        if v2_final_route_contract.get("active_stage"):
            lines.append(f"- river_v2_final_route_stage: {v2_final_route_contract.get('active_stage')}")
        if v2_final_route_contract.get("active_river_guidance_surface"):
            lines.append(f"- river_v2_final_route_artifact: {v2_final_route_contract.get('active_river_guidance_surface')}")
        lines.append(f"- river_v2_legacy_final_route_participation_blocked: {bool(v2_final_route_contract.get('legacy_river_final_route_participation_blocked'))}")
        lines.append(f"- river_v2_final_route_runtime_enforced: {bool(v2_final_route_contract.get('runtime_enforced'))}")
        if v2_final_route_contract.get("river_method_selected"):
            lines.append(f"- river_method_selected: {v2_final_route_contract.get('river_method_selected')}")
        if v2_final_route_contract.get("river_path_used"):
            lines.append(f"- river_path_used: {v2_final_route_contract.get('river_path_used')}")
        lines.append(f"- legacy_river_path_participated: {bool(v2_final_route_contract.get('legacy_river_path_participated'))}")
        if v2_final_route_contract.get("pipeline_version"):
            lines.append(f"- river_v2_pipeline_version: {v2_final_route_contract.get('pipeline_version')}")
        if v2_final_route_contract.get("stage_products_overview"):
            lines.append(f"- river_v2_stage_products_overview: {v2_final_route_contract.get('stage_products_overview')}")
    legacy_artifacts = final_dem_contract.get("legacy_transitional_artifacts_present") if isinstance(final_dem_contract.get("legacy_transitional_artifacts_present"), list) else []
    if legacy_artifacts:
        lines.append(f"- legacy_transitional_artifacts: {', '.join(str(x) for x in legacy_artifacts)}")
    lines.append("")

    simple_failure = report.get("river", {}).get("simple_river_stage_failure") if isinstance(report.get("river", {}), dict) else None
    if isinstance(simple_failure, dict) and simple_failure:
        lines.append(f"- simple_river_stage_failure: {simple_failure.get('failed_stage')}")
        lines.append(f"- simple_river_stage_error: {simple_failure.get('error')}")
        lines.append("")

    v2_failure = (v2_pass4.get("failed_stage") if v2_pass4 else None) or (v2_pass3.get("failed_stage") if v2_pass3 else None) or (v2_pass2.get("failed_stage") if v2_pass2 else None) or (v2_pass1.get("failed_stage") if v2_pass1 else None)
    if v2_failure:
        if v2_pass4:
            lines.append(f"- river_v2_pass4_failure: {v2_failure}")
            lines.append(f"- river_v2_pass4_error: {v2_pass4.get('error')}")
        elif v2_pass3:
            lines.append(f"- river_v2_pass3_failure: {v2_failure}")
            lines.append(f"- river_v2_pass3_error: {v2_pass3.get('error')}")
        elif v2_pass2:
            lines.append(f"- river_v2_pass2_failure: {v2_failure}")
            lines.append(f"- river_v2_pass2_error: {v2_pass2.get('error')}")
        else:
            lines.append(f"- river_v2_pass1_failure: {v2_failure}")
            lines.append(f"- river_v2_pass1_error: {v2_pass1.get('error')}")
        lines.append("")

    if stage_status:
        lines.append("SIMPLE RIVER TARGET STAGE STATUS")
        for stage_id, spec in stage_status.items():
            if not isinstance(spec, dict):
                continue
            line = (
                f"- {stage_id}: status={spec.get('status')} implemented={bool(spec.get('implemented'))} canonical_filename={spec.get('canonical_filename')} legacy_equivalent={spec.get('legacy_equivalent')}"
            )
            if spec.get('output_artifact'):
                line += f" output_artifact={spec.get('output_artifact')}"
            if spec.get('receipt_path'):
                line += f" receipt_path={spec.get('receipt_path')}"
            lines.append(line)
        lines.append("")

    if v2_stage_status:
        lines.append("RIVER V2 PASS4 STAGE STATUS" if v2_pass4 else ("RIVER V2 PASS3 STAGE STATUS" if v2_pass3 else ("RIVER V2 PASS2 STAGE STATUS" if v2_pass2 else "RIVER V2 PASS1 STAGE STATUS")))
        for stage_id, spec in v2_stage_status.items():
            if not isinstance(spec, dict):
                continue
            line = f"- {stage_id}: status={spec.get('status')} implemented={bool(spec.get('implemented'))} canonical_filename={spec.get('canonical_filename')}"
            if spec.get('output_artifact'):
                line += f" output_artifact={spec.get('output_artifact')}"
            if spec.get('receipt_path'):
                line += f" receipt_path={spec.get('receipt_path')}"
            lines.append(line)
        lines.append("")

    if io_manifest:
        lines.append("INPUTS OBSERVED DURING THIS RUN")
        for p in io_manifest.get("inputs", []):
            lines.append(f"- {p}")
        lines.append("")

        lines.append("OUTPUTS OBSERVED DURING THIS RUN (io_manifest)")
        for p in io_manifest.get("outputs", []):
            lines.append(f"- {p}")
        lines.append("")

    lines.append("STAGE-BY-STAGE FILE TRACE")
    lines.append("Each section below lists actual path-like fields found in the in-memory report for this run.")
    lines.append("")

    seen_section_keys: set[str] = set()
    for section_key, section_title in SECTION_ORDER:
        section_obj = report.get(section_key)
        if not isinstance(section_obj, (dict, list, tuple)):
            continue
        pairs = _dedupe_pairs(_iter_paths(section_obj, prefix=section_key))
        if not pairs:
            continue
        seen_section_keys.add(section_key)
        lines.append(section_title.upper())
        for keypath, p in pairs:
            lines.append(f"- {keypath}: {p}")
        lines.append("")

    lines.append("ADDITIONAL TOP-LEVEL REPORT PATHS")
    extra_pairs: List[Tuple[str, str]] = []
    for k, v in report.items():
        if k in seen_section_keys:
            continue
        if _looks_like_path(v):
            extra_pairs.append((k, v))
        elif isinstance(v, (dict, list, tuple)):
            extra_pairs.extend(_iter_paths(v, prefix=k))
    extra_pairs = _dedupe_pairs(extra_pairs)
    for keypath, p in extra_pairs:
        lines.append(f"- {keypath}: {p}")
    lines.append("")

    lines.append("WHERE TO LOOK FIRST WHEN DEBUGGING")
    lines.append("- io_manifest.json / io_manifest.md: exact input/output paths observed during the run")
    lines.append("- bathy_report.json: stage-by-stage structured report with actual file paths")
    lines.append("- reports/workflow_stage_trace.json: connected stage-to-file trace")
    lines.append("- reports/workflow_file_graph.json: producer/consumer artifact graph")
    lines.append("- reports/WORKFLOW_EXPLANATION_REPORT.txt: plain-language stage explanation")
    lines.append("- reports/WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt: first bad artifact and next fix targets")
    lines.append("- river outputs under report['river']['outputs']: most detailed river-stage artifact map")
    lines.append("- benchmark outputs under report['benchmark'] and report['outputs']: postrun evaluation artifacts")
    lines.append("")

    return "\n".join(lines) + "\n"



def write_workflow_actual_trace(*, out_dir: str | Path, report: Dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    reports_dir = out_dir / TRACE_DIRNAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / TRACE_FILENAME
    text = build_workflow_actual_trace_text(out_dir=out_dir, report=report)
    path.write_text(text, encoding="utf-8")
    report.setdefault("outputs", {})["workflow_input_output_trace"] = str(path)
    report.setdefault("final_reporting", {}).setdefault("receipts", {})["workflow_input_output_trace"] = str(path)
    return path
