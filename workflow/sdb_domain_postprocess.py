from __future__ import annotations
from pathlib import Path
from typing import Any


def load_sdb_artifacts_into_report(*, sdb_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    report.setdefault("sdb", {}).setdefault("artifacts", {})["sdb_dir"] = str(sdb_dir)
    return report

__all__ = ["load_sdb_artifacts_into_report"]
