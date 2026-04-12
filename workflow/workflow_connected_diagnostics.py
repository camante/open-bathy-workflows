from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from io_artifacts import is_probably_path


REPORTS_DIRNAME = "reports"
STAGE_TRACE_JSON = "workflow_stage_trace.json"
STAGE_TRACE_JSONL = "workflow_stage_trace_lines.jsonl"
FILE_GRAPH_JSON = "workflow_file_graph.json"
EXPLANATION_TXT = "WORKFLOW_EXPLANATION_REPORT.txt"


SECTION_ORDER: list[tuple[str, str]] = [
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
    ("final_reporting", "Final reporting"),
    ("validation", "Validation"),
    ("outputs", "Top-level outputs"),
]

REASON_KEYS = (
    "reason",
    "skip_reason",
    "failure_reason",
    "inert_reason",
    "accuracy_warning_reason",
    "selection_reason",
    "hard_problem_blind_reason",
    "notes",
    "note",
)

COUNT_KEYS = (
    "candidate_count",
    "candidate_cell_count",
    "candidate_pixel_count",
    "candidate_station_count",
    "eligible_count",
    "eligible_cell_count",
    "eligible_pixel_count",
    "eligible_station_count",
    "applied_count",
    "applied_cell_count",
    "applied_pixel_count",
    "adjusted_count",
    "adjusted_station_count",
    "changed_count",
    "changed_cell_count",
    "changed_pixel_count",
    "final_changed_pixel_count",
    "nonzero_cell_count",
    "nonzero_weight_count",
    "presemantic_nonzero_cell_count",
)

INPUT_HINTS = (
    "input",
    "source",
    "baseline",
    "reference",
    "upstream",
    "manifest",
    "neighbor",
    "template",
    "scaffold",
    "support",
)

OUTPUT_HINTS = (
    "output",
    "final",
    "summary",
    "profile",
    "contract",
    "receipt",
    "report",
    "trace",
    "bundle",
    "manifest",
    "audit",
)

SKIP_DESCEND_KEYS = {"stdout", "stderr", "command", "message", "description"}
SKIP_PATH_VALUE_KEYS = {"module", "function"}


def _looks_like_path(value: Any) -> bool:
    return isinstance(value, str) and is_probably_path(value)


def _iter_paths(obj: Any, prefix: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            keypath = f"{prefix}.{k}" if prefix else str(k)
            if k in SKIP_PATH_VALUE_KEYS:
                if isinstance(v, (dict, list, tuple)) and k not in SKIP_DESCEND_KEYS:
                    yield from _iter_paths(v, keypath)
                continue
            if _looks_like_path(v):
                yield keypath, v
            elif isinstance(v, (dict, list, tuple)) and k not in SKIP_DESCEND_KEYS:
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


def _collect_scalar_metrics(section_obj: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if not isinstance(section_obj, dict):
        return metrics
    for key in COUNT_KEYS:
        if key in section_obj and isinstance(section_obj.get(key), (int, float, bool)):
            metrics[key] = section_obj.get(key)
    for k, v in section_obj.items():
        if k in metrics:
            continue
        if isinstance(v, (int, float, bool)) and any(token in k for token in ("count", "ratio", "mean", "p95", "max", "min", "frac")):
            metrics[k] = v
        elif isinstance(v, dict) and k.endswith("reason_counts"):
            metrics[k] = v
    return metrics


def _extract_reason(section_obj: Any) -> str | None:
    if not isinstance(section_obj, dict):
        return None
    for key in REASON_KEYS:
        val = section_obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _classify_path(*, keypath: str, path: str, io_inputs: set[str], io_outputs: set[str]) -> str:
    lp = keypath.lower()
    if path in io_inputs:
        return "input"
    if any(tok in lp for tok in INPUT_HINTS):
        return "input"
    if any(tok in lp for tok in OUTPUT_HINTS):
        return "output"
    if path in io_outputs:
        return "output"
    if ".outputs." in lp or ".receipts." in lp:
        return "output"
    return "output"


def _infer_status(section_key: str, section_obj: Any, outputs: list[dict[str, str]], inputs: list[dict[str, str]]) -> str:
    if isinstance(section_obj, dict):
        status = section_obj.get("status")
        if isinstance(status, str) and status.strip():
            return status.strip()
        requested = section_obj.get("requested")
        if requested is False:
            return "skipped"
    if outputs:
        return "ran"
    if inputs and section_key != "outputs":
        return "partial"
    return "skipped"


def _path_exists(path: str) -> bool:
    try:
        return Path(path).exists()
    except (OSError, TypeError, ValueError):
        return False


def _build_stage_entries(*, out_dir: Path, report: Dict[str, Any]) -> list[dict[str, Any]]:
    io_manifest = _load_io_manifest(out_dir)
    io_inputs = set((io_manifest or {}).get("inputs", []) or [])
    io_outputs = set((io_manifest or {}).get("outputs", []) or [])
    stage_entries: list[dict[str, Any]] = []
    stage_index: dict[str, dict[str, Any]] = {}

    for order, (section_key, title) in enumerate(SECTION_ORDER):
        section_obj = report.get(section_key)
        pairs = _dedupe_pairs(_iter_paths(section_obj, prefix=section_key)) if isinstance(section_obj, (dict, list, tuple)) else []
        inputs: list[dict[str, str]] = []
        outputs: list[dict[str, str]] = []
        for keypath, path in pairs:
            item = {"keypath": keypath, "path": path, "exists": _path_exists(path)}
            if _classify_path(keypath=keypath, path=path, io_inputs=io_inputs, io_outputs=io_outputs) == "input":
                inputs.append(item)
            else:
                outputs.append(item)
        status = _infer_status(section_key, section_obj, outputs, inputs)
        reason = _extract_reason(section_obj)
        missing_outputs = [item for item in outputs if not item.get("exists", False)]
        stage = {
            "stage_name": section_key,
            "stage_title": title,
            "module": section_key,
            "function": None,
            "status": status,
            "reason": reason,
            "input_files": inputs,
            "output_files": outputs,
            "summary_metrics": _collect_scalar_metrics(section_obj),
            "upstream_dependencies": [],
            "downstream_consumers": [],
            "artifact_subjects": [{"path": item["path"], "artifact_subject": item["keypath"], "exists": item.get("exists", False)} for item in outputs],
            "order": order,
            "missing_output_files": missing_outputs,
            "missing_output_count": len(missing_outputs),
            "orphan_output_files": [],
            "orphan_output_count": 0,
        }
        if isinstance(section_obj, dict):
            if isinstance(section_obj.get("module"), str):
                stage["module"] = section_obj.get("module")
            if isinstance(section_obj.get("function"), str):
                stage["function"] = section_obj.get("function")
        stage_entries.append(stage)
        stage_index[section_key] = stage

    # Build downstream/upstream relationships by exact path matching.
    path_producers: dict[str, list[str]] = {}
    for stage in stage_entries:
        for item in stage["output_files"]:
            path_producers.setdefault(item["path"], []).append(stage["stage_name"])

    for stage in stage_entries:
        ups: set[str] = set()
        for item in stage["input_files"]:
            for producer in path_producers.get(item["path"], []):
                if producer != stage["stage_name"]:
                    ups.add(producer)
        stage["upstream_dependencies"] = sorted(ups)

    for stage in stage_entries:
        downs: set[str] = set()
        orphan_outputs: list[dict[str, Any]] = []
        for out_item in stage["output_files"]:
            out_path = out_item["path"]
            consumed = False
            for other in stage_entries:
                if other["stage_name"] == stage["stage_name"]:
                    continue
                if any(inp["path"] == out_path for inp in other["input_files"]):
                    downs.add(other["stage_name"])
                    consumed = True
            if not consumed:
                orphan_outputs.append(out_item)
        stage["downstream_consumers"] = sorted(downs)
        stage["orphan_output_files"] = orphan_outputs
        stage["orphan_output_count"] = len(orphan_outputs)

    return stage_entries


def _build_file_graph(stage_entries: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for stage in stage_entries:
        for item in stage["output_files"]:
            art = artifacts.setdefault(item["path"], {"path": item["path"], "producers": [], "consumers": [], "artifact_subjects": [], "exists": item.get("exists", False)})
            art["exists"] = art.get("exists", False) or item.get("exists", False)
            if stage["stage_name"] not in art["producers"]:
                art["producers"].append(stage["stage_name"])
            art["artifact_subjects"].append(item["keypath"])
        for item in stage["input_files"]:
            art = artifacts.setdefault(item["path"], {"path": item["path"], "producers": [], "consumers": [], "artifact_subjects": [], "exists": item.get("exists", False)})
            art["exists"] = art.get("exists", False) or item.get("exists", False)
            if stage["stage_name"] not in art["consumers"]:
                art["consumers"].append(stage["stage_name"])
            art["artifact_subjects"].append(item["keypath"])
    artifact_list = []
    for art in artifacts.values():
        art["artifact_subjects"] = sorted(set(art["artifact_subjects"]))
        art["producers"] = sorted(art["producers"])
        art["consumers"] = sorted(art["consumers"])
        artifact_list.append(art)
    artifact_list.sort(key=lambda x: x["path"])
    return {"artifact_count": len(artifact_list), "artifacts": artifact_list}


def build_connected_diagnostics_payload(*, out_dir: str | Path, report: Dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    stage_entries = _build_stage_entries(out_dir=out_dir, report=report)
    file_graph = _build_file_graph(stage_entries)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "created_utc": created,
        "out_dir": str(out_dir),
        "stage_trace": stage_entries,
        "file_graph": file_graph,
    }


def build_explanation_report_text(*, payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("WORKFLOW EXPLANATION REPORT")
    lines.append(f"created_utc: {payload.get('created_utc')}")
    lines.append(f"out_dir: {payload.get('out_dir')}")
    lines.append("")
    lines.append("This report connects major workflow stages to the files they read/write and explains whether each stage ran, skipped, or only partially engaged.")
    lines.append("")
    for stage in payload.get("stage_trace", []):
        lines.append(stage["stage_title"].upper())
        lines.append(f"- status: {stage.get('status')}")
        if stage.get("reason"):
            lines.append(f"- reason: {stage['reason']}")
        if stage.get("upstream_dependencies"):
            lines.append(f"- upstream_dependencies: {', '.join(stage['upstream_dependencies'])}")
        if stage.get("downstream_consumers"):
            lines.append(f"- downstream_consumers: {', '.join(stage['downstream_consumers'])}")
        metrics = stage.get("summary_metrics") or {}
        if metrics:
            lines.append("- summary_metrics:")
            for k, v in metrics.items():
                lines.append(f"  - {k}: {v}")
        inputs = stage.get("input_files") or []
        if inputs:
            lines.append("- input_files:")
            for item in inputs:
                suffix = "" if item.get("exists", False) else " [missing]"
                lines.append(f"  - {item['keypath']}: {item['path']}{suffix}")
        outputs = stage.get("output_files") or []
        if outputs:
            lines.append("- output_files:")
            for item in outputs:
                suffix = "" if item.get("exists", False) else " [missing]"
                lines.append(f"  - {item['keypath']}: {item['path']}{suffix}")
        if stage.get("status") == "skipped" and not stage.get("reason"):
            lines.append("- explanation: stage did not expose outputs in the final report; inspect upstream configuration and status fields.")
        elif stage.get("status") in {"failed", "partial"}:
            lines.append("- explanation: stage did not complete cleanly or emitted incomplete outputs; inspect the listed files and reason first.")
        elif stage.get("missing_output_count", 0) > 0:
            lines.append("- explanation: stage referenced one or more output files that do not currently exist; inspect the producing code path and write step first.")
        elif stage.get("orphan_output_count", 0) > 0 and stage.get("stage_name") not in {"outputs", "final_reporting", "benchmark"}:
            lines.append("- explanation: stage emitted outputs that are not referenced by any later stage; verify whether they are intended terminal products or disconnected diagnostics.")
        elif not outputs:
            lines.append("- explanation: stage recorded no path-like outputs; inspect its report section for in-memory-only state or missing receipts.")
        else:
            lines.append("- explanation: stage recorded outputs and can be traced through the listed downstream consumers.")
        lines.append("")
    lines.append("FIRST FILES TO OPEN")
    lines.append(f"- {STAGE_TRACE_JSON}")
    lines.append(f"- {FILE_GRAPH_JSON}")
    lines.append("- io_manifest.json")
    lines.append("- bathy_report.json")
    lines.append("- unified_bathy_report.json (legacy compact wrapper; retained for compatibility)")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_connected_diagnostics(*, out_dir: str | Path, report: Dict[str, Any]) -> dict[str, str]:
    out_dir = Path(out_dir)
    reports_dir = out_dir / REPORTS_DIRNAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = build_connected_diagnostics_payload(out_dir=out_dir, report=report)
    stage_trace = payload["stage_trace"]
    file_graph = payload["file_graph"]

    stage_trace_json = reports_dir / STAGE_TRACE_JSON
    stage_trace_json.write_text(json.dumps(stage_trace, indent=2, sort_keys=True), encoding="utf-8")

    stage_trace_jsonl = reports_dir / STAGE_TRACE_JSONL
    with stage_trace_jsonl.open("w", encoding="utf-8") as f:
        for row in stage_trace:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    file_graph_json = reports_dir / FILE_GRAPH_JSON
    file_graph_json.write_text(json.dumps(file_graph, indent=2, sort_keys=True), encoding="utf-8")

    explanation_txt = reports_dir / EXPLANATION_TXT
    explanation_txt.write_text(build_explanation_report_text(payload=payload), encoding="utf-8")

    outputs = {
        "workflow_stage_trace_json": str(stage_trace_json),
        "workflow_stage_trace_jsonl": str(stage_trace_jsonl),
        "workflow_file_graph_json": str(file_graph_json),
        "workflow_explanation_report": str(explanation_txt),
    }
    report.setdefault("outputs", {}).update(outputs)
    report.setdefault("final_reporting", {}).setdefault("receipts", {}).update(outputs)
    return outputs
