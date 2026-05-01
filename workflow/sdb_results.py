from __future__ import annotations
from pathlib import Path
from typing import Any


def finalize_sdb_run(*, sdb_dir: Path, report: dict[str, Any], apply_depth_metadata=None, logger=None) -> Path | None:
    report.setdefault("sdb", {}).setdefault("finalize", {})["sdb_dir"] = str(sdb_dir)
    if logger is not None:
        logger.info("[SDB] Finalize SDB artifacts from %s", sdb_dir)
    return None

__all__ = ["finalize_sdb_run"]
