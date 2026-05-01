from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def write_run_overview(path: str | Path | None = None, *args: Any, out_dir: str | Path | None = None, report: Mapping[str, Any] | None = None, **kwargs: Any) -> Path:
    """Write a small human-readable run overview.

    Older callers passed an explicit output path.  The active workflow calls this
    with out_dir=... and report=..., so support both forms rather than failing at
    the very end of a successful run.
    """
    if path is None:
        if out_dir is None:
            raise TypeError("write_run_overview requires either path or out_dir")
        p = Path(out_dir) / "run_logs" / "run_overview.txt"
    else:
        p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    rep: Mapping[str, Any] = report or {}
    run = rep.get("run", {}) if isinstance(rep.get("run"), Mapping) else {}
    outputs = rep.get("outputs", {}) if isinstance(rep.get("outputs"), Mapping) else {}
    river = rep.get("river_workflow", {}) if isinstance(rep.get("river_workflow"), Mapping) else {}

    lines = [
        "Workflow run overview",
        "=====================",
        f"run_id: {run.get('run_id', 'unknown')}",
        f"status: {rep.get('status', 'unknown')}",
        f"final_output: {outputs.get('combined_warped') or outputs.get('final_dem') or 'unknown'}",
    ]
    if river:
        lines.extend([
            "",
            "River workflow",
            "--------------",
            f"canonical_system_id: {river.get('canonical_system_id', 'unknown')}",
            f"export_vs_parent: {river.get('export_vs_parent', 'unknown')}",
            f"single_writer: {river.get('single_writer', 'unknown')}",
        ])
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


__all__ = ["write_run_overview"]
