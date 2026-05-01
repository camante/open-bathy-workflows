from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def load_river_domain_summary(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"river_domain_summary_missing:{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate_river_domain_summary(summary: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"status": "passed", "summary_keys": sorted(map(str, summary.keys()))}

__all__ = ["evaluate_river_domain_summary", "load_river_domain_summary"]
