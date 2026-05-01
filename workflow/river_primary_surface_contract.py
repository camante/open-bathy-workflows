from __future__ import annotations

"""Contract helpers for final-route river primary surface products."""

import json
from pathlib import Path
from typing import Any


RIVER_PRIMARY_SURFACE_SOURCE_NAMES: dict[int, str] = {
    0: "missing",
    1: "bank_stage_prior",
    2: "resolved_channel_bed",
    3: "xs_profile_resampled",
    4: "authoritative_in_channel",
    5: "authoritative_backbone",
    6: "graph_backbone",
    7: "authoritative_bank_margin",
    8: "station_target_section_tendency",
    9: "station_target_local_authoritative_reconciliation",
    10: "generalized_thalweg_default_tendency",
    11: "bank_edge_geometry_constraint",
}


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


def write_river_primary_surface_contract(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write the river-primary-surface contract without altering stage semantics."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    contract = dict(payload or {})
    contract.setdefault("schema_version", 1)
    contract.setdefault("source_code_names", {str(k): v for k, v in RIVER_PRIMARY_SURFACE_SOURCE_NAMES.items()})
    out.write_text(json.dumps(_jsonable(contract), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


__all__ = ["RIVER_PRIMARY_SURFACE_SOURCE_NAMES", "write_river_primary_surface_contract"]
