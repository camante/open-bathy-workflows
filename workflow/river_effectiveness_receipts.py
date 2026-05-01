from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_effectiveness_receipts(*, river_dir: str | Path, **kwargs) -> tuple[dict[str, str], dict[str, Any]]:
    summary = {"available": False, "status": "not_evaluated_in_builtin_linear_workflow_package"}
    path = _write(Path(river_dir) / "contracts" / "river_effectiveness_summary.json", summary)
    return {"river_effectiveness_summary": str(path)}, summary


def write_science_effect_summary(*, river_dir: str | Path, **kwargs) -> tuple[dict[str, str], dict[str, Any]]:
    summary = {"available": False, "status": "not_evaluated_in_builtin_linear_workflow_package"}
    path = _write(Path(river_dir) / "contracts" / "river_science_effect_summary.json", summary)
    return {"river_science_effect_summary": str(path)}, summary

__all__ = ["write_effectiveness_receipts", "write_science_effect_summary"]
