"""SDB execution planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SDBExecutionPlan:
    should_run: bool
    sdb_dir: Path
    script_dir: Path
    logs_dir: Path
    reason: str = ""


def determine_sdb_execution_plan(*, cfg: Any, report: dict[str, Any], ensure_dir_fn, parse_aoi_bbox_fn=None, count_mask_water_pixels_fn=None, logger=None) -> SDBExecutionPlan:
    methods = set(getattr(cfg, "methods", []) or [])
    out_dir = Path(getattr(cfg, "out_dir", "."))
    sdb_dir = ensure_dir_fn(out_dir / "sdb")
    logs_dir = ensure_dir_fn(sdb_dir / "logs")
    should_run = "sdb" in methods
    reason = "method_enabled" if should_run else "sdb_method_not_requested"
    report.setdefault("sdb", {})["execution_plan"] = {"should_run": bool(should_run), "reason": reason, "sdb_dir": str(sdb_dir)}
    if logger is not None:
        logger.info("[SDB] execution_plan should_run=%s reason=%s", should_run, reason)
    return SDBExecutionPlan(should_run=bool(should_run), sdb_dir=Path(sdb_dir), script_dir=Path(__file__).parent, logs_dir=Path(logs_dir), reason=reason)


__all__ = ["SDBExecutionPlan", "determine_sdb_execution_plan"]
