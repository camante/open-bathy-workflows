from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def run_workflow_benchmark(*, context: Any | None = None, **kwargs: Any) -> dict[str, Any]:
    requested = bool(kwargs.get("benchmark_holdout") or kwargs.get("benchmark_auto_holdout"))
    payload = {
        "status": "skipped",
        "reason": "no_benchmark_holdout_configured" if not requested else "benchmark_inputs_not_available_in_packaged_workflow",
        "requested": requested,
        "metrics": {},
    }
    out_path = kwargs.get("out_path")
    if out_path not in (None, ""):
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
