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

from typing import Any, Dict, Optional, Callable


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
            except Exception:
                # Fall back to stdout if logger fails for any reason.
                pass
        print(line)

    def get(d: Dict[str, Any], *keys: str, default=None):
        for k in keys:
            if isinstance(d, dict) and k in d and d[k] not in (None, ""):
                return d[k]
        return default

    def fmt_int(x):
        try:
            return f"{int(x):,}"
        except Exception:
            return "n/a"

    def fmt_float(x, nd=2):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "n/a"

    def fmt_bool(x):
        if x is True:
            return "yes"
        if x is False:
            return "no"
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

    lines = []
    lines.append("=" * 72)
    lines.append("Run summary")
    lines.append("=" * 72)

    if command:
        lines.append(f"Command: {command}")
    if aoi or start or end:
        lines.append(f"Area + dates: AOI {aoi or 'n/a'} | {start or 'n/a'} → {end or 'n/a'}")
    if methods:
        lines.append(f"What you asked for: methods={','.join(methods)} (priority={priority or 'n/a'})")
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

    # Fusion summary
    if fusion:
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

    # Outputs
    lines.append("")
    lines.append("Outputs written")
    lines.append("-" * 72)

    # Show combined first, then method-specific
    for k in ["combined_warped", "combined_root_copy", "sdb_warped", "river_warped", "river_bottom_warped", "combined", "sdb", "river"]:
        if k in outputs:
            lines.append(f"• {k}: {outputs[k]}")
    if not outputs:
        lines.append("• (No output paths recorded in stats.)")

    lines.append("")
    lines.append("What to sanity-check in GIS")
    lines.append("-" * 72)
    lines.append("• The combined bathy raster should include both nearshore (SDB) and channel (river) where available.")
    lines.append("• In overlap areas, the pipeline should lean toward the priority method (C1 weighted overlap).")
    lines.append("• If the combined raster looks empty, check the land/water mask semantics and training counts in the report JSON.")
    lines.append("=" * 72)

    summary = "\n".join(lines)
    for ln in lines:
        emit(ln)
    return summary
