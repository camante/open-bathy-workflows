from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_stage_receipt(
    *,
    stage_id: str,
    output_artifact: str,
    input_artifacts: list[str],
    record_count: int,
    field_schema: dict[str, str],
    vertical_reference: str | None,
    warnings: list[str],
    source_logic: str,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage_id": str(stage_id),
        "output_artifact": str(output_artifact),
        "input_artifacts": [str(v) for v in input_artifacts],
        "record_count": int(record_count),
        "field_schema": {str(k): str(v) for k, v in field_schema.items()},
        "vertical_reference": None if vertical_reference is None else str(vertical_reference),
        "warnings": [str(v) for v in warnings],
        "source_logic": str(source_logic),
        "validation": dict(validation or {}),
    }


def write_stage_receipt(receipt: dict[str, Any], out_path: str | Path) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding='utf-8')
    return str(path)
