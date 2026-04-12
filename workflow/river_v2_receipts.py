from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from constants import PIPELINE_VERSION
except Exception:
    PIPELINE_VERSION = "unknown"


SCHEMA_VERSION = 1


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_river_v2_stage_receipt(
    *,
    stage_id: str,
    output_artifact: str,
    input_artifacts: list[str],
    record_count: int,
    field_schema: dict[str, str],
    vertical_reference: str | None,
    warnings: list[str],
    source_logic: str,
    validation: Dict[str, Any] | None = None,
    extra_artifacts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": str(PIPELINE_VERSION),
        "stage_id": str(stage_id),
        "input_artifacts": [str(v) for v in (input_artifacts or [])],
        "output_artifact": str(output_artifact),
        "record_count": int(record_count),
        "field_schema": {str(k): str(v) for k, v in (field_schema or {}).items()},
        "warnings": [str(v) for v in (warnings or [])],
        "source_logic": str(source_logic),
        "validation": dict(validation or {}),
    }
    if vertical_reference is not None:
        receipt["vertical_reference"] = str(vertical_reference)
    if extra_artifacts:
        receipt["extra_artifacts"] = _jsonable(dict(extra_artifacts))
    return _jsonable(receipt)


def write_river_v2_receipt(receipt: Dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_river_v2_raster_receipt(
    *,
    stage_id: str,
    output_artifacts: list[str],
    input_artifacts: list[str],
    vertical_reference: str | None,
    warnings: list[str],
    source_logic: str,
    validation: Dict[str, Any] | None = None,
    grid_shape: tuple[int, int] | None = None,
    stats: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": str(PIPELINE_VERSION),
        "stage_id": str(stage_id),
        "input_artifacts": [str(v) for v in (input_artifacts or [])],
        "output_artifacts": [str(v) for v in (output_artifacts or [])],
        "warnings": [str(v) for v in (warnings or [])],
        "source_logic": str(source_logic),
        "validation": dict(validation or {}),
    }
    if vertical_reference is not None:
        receipt["vertical_reference"] = str(vertical_reference)
    if grid_shape is not None:
        receipt["grid_shape"] = [int(grid_shape[0]), int(grid_shape[1])]
    if stats:
        essential_stats = {}
        for key in ("finite_pixel_count", "nodata_pixel_count", "min_z_m", "max_z_m"):
            if key in stats:
                essential_stats[key] = stats[key]
        if essential_stats:
            receipt["stats"] = essential_stats
    return _jsonable(receipt)
