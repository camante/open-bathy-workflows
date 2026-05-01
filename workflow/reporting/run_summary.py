"""
Human-readable run summary.

This module provides a drop-in function that prints (and returns) a plain-English,
easy-to-scan recap of what a run did and what it produced.

Design goals:
- Safe: never raise on missing keys; just omit what isn't available.
- Plain English: explain parameters and outcomes without jargon.
- Logging-friendly: you can pass log_fn=logger.info to write into logs instead of stdout.
"""

from __future__ import annotations


import logging
from typing import Any, Dict, Optional, Callable, Union

log = logging.getLogger(__name__)


def print_human_run_summary(stats: Dict[str, Any], log_fn: Optional[Callable[[str], None]] = None) -> str:
    """
    Print (and return) a human-friendly run summary.

    Parameters
    ----------
    stats:
        A dict of run facts/metrics. Missing keys are fine.
        Recommended keys (all optional):
          - command (str)
          - aoi (str or dict)
          - time_window: {"start": "...", "end": "..."}
          - methods (list[str])
          - priority (str)
          - out_dir (str) or output_dir (str)
          - outputs (dict): paths for combined/sdb/river products
          - sdb (dict): nested metrics
          - river (dict): nested metrics
          - fusion (dict): status/mode/weights/errors

    log_fn:
        If provided, each line is sent to log_fn(line). Otherwise, prints to stdout.

    Returns
    -------
    str
        The full summary text.
    """
    def emit(line: str) -> None:
        if log_fn is not None:
            try:
                log_fn(line)
                return
            except (TypeError, ValueError, RuntimeError):
                # Fall back to stdout if logger fails for any reason.
                log.debug("ignored", exc_info=True)
        log.info(line)

    def get(d: Dict[str, Any], *keys: str, default=None):
        for k in keys:
            if isinstance(d, dict) and k in d and d[k] not in (None, ""):
                return d[k]
        return default

    def fmt_int(x):
        try:
            return f"{int(x):,}"
        except (TypeError, ValueError):
            return "n/a"

    def fmt_float(x, nd=2):
        try:
            return f"{float(x):.{nd}f}"
        except (TypeError, ValueError):
            return "n/a"

    def fmt_bool(x):
        if x is True:
            return "yes"
        if x is False:
            return "no"
        return "n/a"

    def existing_path_label(mapping, *keys):
        from pathlib import Path as _PathLocal
        for key in keys:
            if not isinstance(mapping, dict) or key not in mapping:
                continue
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value if _PathLocal(value).exists() else f"(missing) {value}"
            if value not in (None, ""):
                return f"(unverified) {value}"
        return "n/a"

    command = get(stats, "command", "cmd")
    aoi = get(stats, "aoi")
    tw = get(stats, "time_window", default={}) or {}
    start = get(tw, "start")
    end = get(tw, "end")
    methods = get(stats, "methods", default=[])
    priority = get(stats, "priority")
    out_dir = get(stats, "out_dir", "output_dir")
    outputs = get(stats, "outputs", default={}) or {}

    sdb = get(stats, "sdb", default={}) or {}
    river = get(stats, "river", default={}) or {}
    fusion = get(stats, "fusion", default={}) or {}
    river_workflow_report = get(stats, "river_workflow", default={}) or {}
    active_river_report = get(stats, "active_river", default={}) or {}
    seamless_dem = get(stats, "seamless_dem", default={}) or {}
    route = get(stats, "route", default={}) or {}
    guidance_domains = get(stats, "guidance_domains", default={}) or {}

    river_workflow = get(river, "workflow_name", "river_workflow", "method", default=None)
    river_notes = get(river, "notes", default={}) or {}
    river_workflow_notes = river_notes.get("river_workflow") if isinstance(river_notes, dict) else {}
    shared_solve_notes = river_notes.get("shared_solve") if isinstance(river_notes, dict) else {}
    execution_mode = (
        get(river, "execution_mode", default=None)
        or get(river_workflow_report, "execution_mode", default=None)
        or get(active_river_report, "execution_mode", default=None)
        or get(seamless_dem, "final_source", default=None)
    )
    pipeline_name = (
        get(river_workflow_notes, "pipeline", default=None)
        or get(shared_solve_notes, "pipeline", default=None)
        or execution_mode
    )
    workflow_label = str(river_workflow or pipeline_name or get(route, "active_route", default="") or "").lower()
    # Treat only the explicit canonical-parent/AOI-export route as the
    # canonical river route. Reporting must not blur route ownership.
    active_route = str(get(route, "active_route", default="") or "").lower()
    is_shared_solve_river = (
        active_route == "canonical_river_parent_export"
        or "canonical_parent_plus_aoi_export" in workflow_label
        or get(seamless_dem, "final_source", default=None) == "aoi_export_dem"
    )

    lines = []
    lines.append("=" * 72)
    lines.append("Run summary")
    lines.append("=" * 72)

    if command:
        lines.append(f"Command: {command}")
    if aoi or start or end:
        lines.append(f"Area + dates: AOI {aoi or 'n/a'} | {start or 'n/a'} → {end or 'n/a'}")
    requested_methods = get(route, "requested_methods", default=None) or methods
    if requested_methods:
        lines.append(f"What you asked for: methods={','.join(requested_methods)} (priority={get(route, 'requested_priority', default=None) or priority or 'n/a'})")
    if get(route, "active_route", default=None):
        lines.append(f"Active route: {get(route, 'active_route')}")
        lines.append(f"Final DEM owner: {get(route, 'final_dem_owner', default='n/a')}")
    if out_dir:
        lines.append(f"Output folder: {out_dir}")

    # Inputs + key counts (only if available)
    lines.append("")
    lines.append("Key inputs used")
    lines.append("-" * 72)

    sdb_used = get(sdb, "enabled", default=("sdb" in methods if isinstance(methods, list) else None))
    river_used = get(river, "enabled", default=("river" in methods if isinstance(methods, list) else None))
    lines.append(f"• SDB enabled: {fmt_bool(sdb_used)}")
    lines.append(f"• River enabled: {fmt_bool(river_used)}")
    if is_shared_solve_river:
        lines.append("• Active river route: canonical parent solve/export")
        active_methods = get(route, "active_methods", default=None)
        inactive_methods = get(route, "inactive_methods", default=None)
        inactive_reason = get(route, "inactive_reason", default=None)
        if active_methods:
            lines.append(f"• Active construction methods: {','.join(active_methods)}")
        if inactive_methods:
            lines.append(f"• Inactive methods this run: {','.join(inactive_methods)}")
        if inactive_reason:
            lines.append(f"• Inactive-method reason: {inactive_reason}")
        if execution_mode:
            lines.append(f"• River execution mode: {execution_mode}")
        skipped_reason = get(sdb, "skipped_reason", "reason", default=None) or get(river_workflow_report, "sdb_skipped_reason", default=None)
        if skipped_reason:
            lines.append(f"• SDB skipped reason: {skipped_reason}")

    # Optional training summary if provided
    train = get(sdb, "training", default={}) or {}
    n_train = get(train, "n_final", "n_train_final", default=None)
    n_fused = get(train, "n_fused", default=None)
    n_sampled = get(train, "n_sampled", default=None)
    if any(v is not None for v in (n_fused, n_sampled, n_train)):
        lines.append("")
        lines.append("SDB training (high level)")
        lines.append("-" * 72)
        if n_fused is not None:
            lines.append(f"• Fused training points: {fmt_int(n_fused)}")
        if n_sampled is not None:
            lines.append(f"• After spatial sampling: {fmt_int(n_sampled)}")
        if n_train is not None:
            lines.append(f"• Final rows used to train: {fmt_int(n_train)}")

    # Fusion summary. For the canonical river parent/export route, fusion is
    # inactive and should not read like a participating construction path.
    if fusion and not is_shared_solve_river:
        lines.append("")
        lines.append("Fusion (how the final bathy surface was built)")
        lines.append("-" * 72)
        status = get(fusion, "status")
        mode = get(fusion, "mode")
        lines.append(f"• Status: {status or 'n/a'}")
        if mode:
            lines.append(f"• Mode: {mode}")
        if mode == "weighted_overlap":
            pri = get(fusion, "priority") or priority
            weights = get(fusion, "weights", default={}) or {}
            wpri = get(weights, "primary", default=None)
            wsec = get(weights, "secondary", default=None)
            if wpri is not None and wsec is not None:
                lines.append(f"• Overlap rule: weighted (C1) → {pri or 'priority'} {fmt_float(wpri,2)} + other {fmt_float(wsec,2)}")
        err = get(fusion, "error", "reason")
        if err:
            lines.append(f"• Note: {err}")

    if is_shared_solve_river:
        lines.append("")
        lines.append("Canonical river route debugging map")
        lines.append("-" * 72)
        lines.append("• Construction chain: canonical_parent_dem → aoi_export_dem → final_user_dem")
        lines.append(f"• Canonical parent manifest: {existing_path_label(outputs, 'canonical_river_solution_manifest')}")
        lines.append(f"• AOI export identity: {existing_path_label(outputs, 'aoi_export_identity_report', 'aoi_identity_receipt')}")
        lines.append(f"• River workflow receipt: {existing_path_label(outputs, 'river_workflow_receipt')}")
        lines.append(f"• Final output receipt: {existing_path_label(outputs, 'final_output_receipt')}")
        lines.append(f"• Final user DEM: {existing_path_label(outputs, 'combined_warped', 'final_dem')}")

    # Outputs
    lines.append("")
    if is_shared_solve_river:
        lines.append("Primary outputs written")
    else:
        lines.append("Outputs written")
    lines.append("-" * 72)

    # Show combined first, then method-specific.
    # Only claim a path exists after confirming it on disk.
    from pathlib import Path as _Path

    shown_any = False
    output_key_order = [
        "combined_warped",
        "final_dem",
        "run_summary",
        "river_workflow_receipt",
        "river_workflow_summary",
        "final_output_receipt",
        "aoi_export_identity_report",
        "canonical_river_solution_manifest",
    ] if is_shared_solve_river else [
        "combined_warped",
        "combined_root_copy",
        "final_dem",
        "run_summary",
        "river_workflow_receipt",
        "river_workflow_summary",
        "final_output_receipt",
        "aoi_export_identity_report",
        "canonical_river_solution_manifest",
        "sdb_warped",
        "river_warped",
        "river_bottom_warped",
        "combined",
        "sdb",
        "river",
    ]
    for k in output_key_order:
        if k not in outputs:
            continue
        v = outputs.get(k)
        if isinstance(v, str) and v.strip():
            if _Path(v).exists():
                lines.append(f"• {k}: {v}")
                shown_any = True
            else:
                lines.append(f"• {k}: (missing) {v}")
                shown_any = True
        else:
            # Unknown / non-string value; don't pretend we know.
            lines.append(f"• {k}: (unverified) {v}")
            shown_any = True

    if not outputs or not shown_any:
        if is_shared_solve_river:
            lines.append("• (No verified final combined raster recorded yet; for the built-in shared-solve river workflow, verify the river_workflow_* outputs and the conditioned output receipts.)")
        else:
            lines.append("• (No verified output paths recorded in stats.)")

    lines.append("")
    lines.append("What to sanity-check in GIS")
    lines.append("-" * 72)
    if is_shared_solve_river:
        lines.append("• Verify final/DEM_enhanced.tif against final/authoritative_base_aligned.tif and final/canonical_parent_dem.tif.")
        lines.append("• Confirm reports/run_summary.txt says route: canonical_parent_dem -> aoi_export_dem -> final_user_dem.")
        lines.append("• For north/south tests, inspect output/<name>_<version>/comparison.txt; it should report the same canonical parent and a PASS grid relation.")
    else:
        lines.append("• The combined bathy raster should include both nearshore (SDB) and channel (river) where available.")
        lines.append("• In overlap areas, the pipeline should lean toward the priority method (C1 weighted overlap).")
        lines.append("• If the combined raster looks empty, check the land/water mask semantics and training counts in the report JSON.")
    lines.append("=" * 72)

    summary = "\n".join(lines)
    for ln in lines:
        emit(ln)
    return summary


# -----------------------------------------------------------------------------
# File writers + flight recorder summarizer (v11)
# -----------------------------------------------------------------------------

import json
import re
from pathlib import Path
from typing import List, Tuple


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(errors='ignore'))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def summarize_flight_recorder(fr_path: Path) -> Dict[str, Any]:
    """Summarize what ran from a flight_recorder_*.jsonl file."""
    summary: Dict[str, Any] = {
        "flight_path": str(fr_path),
        "steps": [],
        "subprocesses": [],
        "exceptions": [],
        "artifacts": [],
        "log_counts": {},
        "run": {},
    }
    if not fr_path or not fr_path.exists():
        return summary

    step_starts: Dict[str, Dict[str, Any]] = {}
    # subprocess start keyed by an incrementing index if no pid
    sp_starts: List[Dict[str, Any]] = []
    last_ts = None

    def _inc(d: Dict[str, int], k: str) -> None:
        d[k] = int(d.get(k, 0)) + 1

    try:
        for line in fr_path.read_text(errors='ignore').splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            last_ts = obj.get("ts") or last_ts
            kind = obj.get("kind")
            if kind == "log":
                lvl = str(obj.get("level", "UNKNOWN"))
                _inc(summary["log_counts"], lvl)
                continue
            if kind != "event":
                continue
            ev = obj.get("event")
            if ev == "run_start":
                summary["run"]["start_ts"] = obj.get("ts")
                summary["run"]["argv"] = obj.get("argv")
                summary["run"]["pid"] = obj.get("pid")
            elif ev == "run_end":
                summary["run"]["end_ts"] = obj.get("ts")
                summary["run"]["events_written"] = obj.get("events_written")
                summary["run"]["dropped"] = obj.get("dropped")
            elif ev == "artifact_written":
                ap = obj.get("artifact_path")
                if ap:
                    summary["artifacts"].append({
                        "path": str(ap),
                        "kind": str(obj.get("artifact_kind", "")),
                        "role": str(obj.get("artifact_role", "")),
                        "ts": obj.get("ts"),
                    })
            elif ev == "step_start":
                name = obj.get("step") or obj.get("name")
                if name:
                    step_starts[str(name)] = obj
            elif ev == "step_end":
                name = obj.get("step") or obj.get("name")
                if name:
                    start = step_starts.get(str(name), {})
                    summary["steps"].append({
                        "name": str(name),
                        "start_ts": start.get("ts"),
                        "end_ts": obj.get("ts"),
                        "ok": obj.get("ok"),
                        "elapsed_s": obj.get("elapsed_s"),
                    })
            elif ev in ("exception", "uncaught_exception"):
                summary["exceptions"].append({
                    "ts": obj.get("ts"),
                    "event": ev,
                    "where": obj.get("where"),
                    "exc_type": obj.get("exc_type"),
                    "message": obj.get("message"),
                })
            elif ev == "subprocess_start":
                sp_starts.append(obj)
            elif ev == "subprocess_end":
                # match to last unmatched
                start = sp_starts.pop() if sp_starts else {}
                summary["subprocesses"].append({
                    "cmd": start.get("cmd") or obj.get("cmd"),
                    "cwd": start.get("cwd") or obj.get("cwd"),
                    "start_ts": start.get("ts"),
                    "end_ts": obj.get("ts"),
                    "rc": obj.get("rc"),
                })
    except (OSError, TypeError, ValueError):
        log.debug("ignored", exc_info=True)

    if summary.get("run", {}).get("start_ts") and not summary.get("run", {}).get("end_ts") and last_ts is not None:
        summary["run"]["end_ts"] = last_ts

    # Sort steps by elapsed desc if available
    try:
        summary["steps"].sort(key=lambda x: (x.get("elapsed_s") is None, -(float(x.get("elapsed_s") or 0.0))))
    except (TypeError, ValueError):
        log.debug('step sort failed', exc_info=True)
    return summary


def _argv_flag(argv: List[Any], name: str) -> Optional[str]:
    try:
        parts = [str(x) for x in (argv or [])]
        for i, part in enumerate(parts):
            if part == name and i + 1 < len(parts):
                return parts[i + 1]
            if part.startswith(name + "="):
                return part.split("=", 1)[1]
    except Exception:
        log.debug("ignored", exc_info=True)
    return None


def _discover_stats(out_dir: Path, run_id: str) -> Dict[str, Any]:
    """Load a run report without guessing filenames.

    Policy:
      - Prefer explicit top-level reports written by bathy_main.py:
          * bathy_report.json
          * run_report.json
          * unified_bathy_report.json (legacy optional wrapper)
      - Do not recurse/search; stale or retained folders can mislead.
    """
    _ = run_id  # reserved for future use
    for name in ('bathy_report.json', 'run_report.json', 'unified_bathy_report.json'):
        pth = out_dir / name
        d = _load_json(pth)
        if isinstance(d, dict) and d:
            d.setdefault('_stats_source', str(pth))
            return d
    return {}



def build_technical_summary(stats: Dict[str, Any], fr_summary: Dict[str, Any]) -> str:
    """Technical, machine-adjacent summary."""
    lines = []
    lines.append("# Run technical summary")
    src = stats.get("_stats_source")
    if src:
        lines.append(f"- stats source: {src}")
    run = fr_summary.get("run", {}) if isinstance(fr_summary, dict) else {}
    if run:
        lines.append(f"- run_id: {run.get('argv') and ' '.join([str(x) for x in run.get('argv',[])]) or 'n/a'}")
        lines.append(f"- start: {run.get('start_ts','n/a')}  end: {run.get('end_ts','n/a')}")
        lines.append(f"- flight events_written: {run.get('events_written','n/a')}  dropped: {run.get('dropped','n/a')}")
    # Steps
    steps = fr_summary.get("steps", []) if isinstance(fr_summary, dict) else []
    if steps:
        lines.append("\n## Steps (from flight recorder)")
        for s in steps:
            lines.append(f"- {s.get('name')}: elapsed_s={s.get('elapsed_s','n/a')} ok={s.get('ok','n/a')}")
    # Errors
    exc = fr_summary.get("exceptions", []) if isinstance(fr_summary, dict) else []
    if exc:
        lines.append("\n## Exceptions")
        for e in exc[:10]:
            lines.append(f"- {e.get('event')} {e.get('exc_type')}: {e.get('message')}")
        if len(exc) > 10:
            lines.append(f"- … ({len(exc)-10} more)")
    return "\n".join(lines) + "\n"




def _summarize_outputs_and_policy(stats: dict) -> str:
    # Return markdown listing key artifacts + domain/policy flags (best effort).
    lines = []
    if not isinstance(stats, dict):
        return ''

    outputs = stats.get('outputs')
    if isinstance(outputs, dict) and outputs:
        lines.append('## Key output artifacts')
        prefer = [
            'combined_warped', 'combined_root_copy', 'combined_bottom_navd88',
            'sdb_warped', 'river_warped', 'river_bottom_warped',
        ]
        for k in prefer:
            if k in outputs:
                lines.append(f"- {k}: {outputs.get(k)}")
        other = [k for k in outputs.keys() if k not in set(prefer)]
        for k in sorted(other):
            lines.append(f"- {k}: {outputs.get(k)}")

    fusion = stats.get('fusion')
    if isinstance(fusion, dict) and fusion:
        lines.append('')
        lines.append('## Fusion/domain policy')
        for k in [
            'priority', 'mode', 'status',
            'river_overrides_sdb_in_domain',
            'river_domain_mask',
            'sdb_domain_mask',
        ]:
            if k in fusion:
                lines.append(f"- {k}: {fusion.get(k)}")
        w = fusion.get('weights') if isinstance(fusion.get('weights'), dict) else None
        if w:
            lines.append(f"- weights: {w}")
        err = fusion.get('error') or fusion.get('reason')
        if err:
            lines.append(f"- note: {err}")

    return "\n".join(lines)


def build_scientific_summary(stats: Dict[str, Any], fr_summary: Optional[Dict[str, Any]] = None) -> str:
    """Scientific summary: what was inferred, where, and key validity signals."""
    lines = []
    lines.append("# Run scientific summary")
    # Try to read known fields defensively
    aoi = stats.get("aoi") or stats.get("args", {}).get("aoi")
    tw = stats.get("time_window") or stats.get("args", {}).get("time_window") or {}
    run_argv = ((fr_summary or {}).get("run") or {}).get("argv") or []
    if aoi is None:
        aoi = _argv_flag(run_argv, "--aoi")
    if not isinstance(tw, dict) or not tw:
        tw = {
            "start": _argv_flag(run_argv, "--start"),
            "end": _argv_flag(run_argv, "--end"),
        }
    lines.append(f"- AOI: {aoi if aoi is not None else 'n/a'}")
    if isinstance(tw, dict):
        lines.append(f"- time window: {tw.get('start','n/a')} → {tw.get('end','n/a')}")
    # SDB
    sdb = stats.get("sdb", {}) if isinstance(stats, dict) else {}
    if isinstance(sdb, dict) and sdb:
        tr = sdb.get("training", {}) if isinstance(sdb.get("training"), dict) else {}
        if tr:
            lines.append("\n## SDB")
            if "n_fused" in tr:
                lines.append(f"- fused training points: {tr.get('n_fused')}")
            if "n_final" in tr or "n_train_final" in tr:
                lines.append(f"- final training rows: {tr.get('n_final', tr.get('n_train_final'))}")
        pred = sdb.get("prediction", {}) if isinstance(sdb.get("prediction"), dict) else {}
        if pred and "valid_cells" in pred:
            lines.append(f"- predicted valid cells: {pred.get('valid_cells')}")
    # River
    river = stats.get("river", {}) if isinstance(stats, dict) else {}
    if isinstance(river, dict) and river:
        lines.append("\n## River")
        for k in ["mask_fraction", "n_reaches", "xs_built", "xs_dropped"]:
            if k in river:
                lines.append(f"- {k}: {river.get(k)}")
    # Fusion
    fusion = stats.get("fusion", {}) if isinstance(stats, dict) else {}
    if isinstance(fusion, dict) and fusion:
        lines.append("\n## Fusion")
        for k in ["mode", "status", "priority", "error"]:
            if k in fusion:
                lines.append(f"- {k}: {fusion.get(k)}")
    lines.append("\n## Suggested sanity checks")
    lines.append("- Confirm final raster is single-band depth (negative-down if configured).")
    lines.append("- Confirm river domain override: inside river mask, river bathy should dominate.")
    lines.append("- Check for seams at AOI edges if running adjacent tiles; use overlap diagnostics if enabled.")
    
    # Artifacts recorded by flight recorder (useful if report write failed)
    arts = fr_summary.get('artifacts', []) if isinstance(fr_summary, dict) else []
    if arts:
        lines.append("\n## Artifacts recorded")
        # prioritize depth/final-ish artifacts by role
        def _score(a):
            role = str(a.get('role',''))
            p = str(a.get('path',''))
            s = 0
            if 'final' in role or 'final' in p:
                s -= 10
            if 'depth' in role or 'depth' in p:
                s -= 5
            if p.endswith('.tif') or p.endswith('.tiff'):
                s -= 1
            return (s, len(p))
        for a in sorted(arts, key=_score)[:30]:
            p = str(a.get('path',''))
            role = str(a.get('role',''))
            kind = str(a.get('kind',''))
            tag = role or kind or 'artifact'
            lines.append(f"- {tag}: {p}")
    return "\n".join(lines) + "\n"


def build_human_like_summary(stats: Dict[str, Any], fr_summary: Dict[str, Any]) -> str:
    """Short, human-style recap."""
    methods = stats.get("methods") or stats.get("args", {}).get("methods")
    out_dir = stats.get("out_dir") or stats.get("output_dir") or stats.get("args", {}).get("out_dir")
    steps = fr_summary.get("steps", []) if isinstance(fr_summary, dict) else []
    took = None
    try:
        # total time as sum of step_end elapsed
        took = sum(float(s.get("elapsed_s") or 0.0) for s in steps) if steps else None
    except (TypeError, ValueError):
        took = None
    lines = []
    lines.append("What happened in this run")
    lines.append("-" * 40)
    if methods:
        lines.append(f"You asked for: {methods}")
    if took is not None and took > 0:
        lines.append(f"It ran for about {took/60.0:.1f} minutes (sum of recorded steps).")
    if out_dir:
        lines.append(f"Outputs are under: {out_dir}")
    exc = fr_summary.get("exceptions", []) if isinstance(fr_summary, dict) else []
    if exc:
        lines.append(f"There were {len(exc)} exceptions logged; check the flight recorder for details.")
    else:
        lines.append("No uncaught exceptions were recorded.")
    lines.append("If something looks off in GIS, start with the provenance raster (if produced) and the run_logs folder.")
    return "\n".join(lines) + "\n"


def write_run_summary_files(out_dir: Union[str, Path], run_id: str, stats: Optional[Dict[str, Any]] = None, fr_path: Optional[Union[str, Path]] = None, *, write_machine_json: bool = True, write_legacy_text_summaries: bool = False) -> Dict[str, Path]:
    """Write optional legacy run summaries into <out_dir>/run_logs/.

    The connected workflow diagnostics are now the primary default reporting
    system. The legacy machine JSON and text summaries are retained only for
    compatibility and are therefore individually optional.
    """
    out_dir = Path(out_dir)
    log_dir = out_dir / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if stats is None:
        stats = _discover_stats(out_dir, run_id)

    fr_summary: Dict[str, Any] = {}
    if fr_path is not None:
        frp = Path(fr_path)
        fr_summary = summarize_flight_recorder(frp)

    machine = {
        "run_id": run_id,
        "stats": stats,
        "flight": fr_summary,
    }

    paths: Dict[str, Path] = {}
    if write_machine_json:
        p_machine = log_dir / f"run_summary_{run_id}.json"
        p_machine.write_text(json.dumps(machine, indent=2, sort_keys=True), encoding="utf-8")
        paths["machine_json"] = p_machine

    if write_legacy_text_summaries:
        p_tech = log_dir / f"run_summary_technical_{run_id}.md"
        p_tech.write_text(build_technical_summary(stats, fr_summary), encoding="utf-8")
        paths["technical_md"] = p_tech

        p_sci = log_dir / f"run_summary_scientific_{run_id}.md"
        p_sci.write_text(build_scientific_summary(stats, fr_summary), encoding="utf-8")
        paths["scientific_md"] = p_sci

        p_human = log_dir / f"run_summary_human_{run_id}.txt"
        p_human.write_text(build_human_like_summary(stats, fr_summary), encoding="utf-8")
        paths["human_txt"] = p_human

    return paths