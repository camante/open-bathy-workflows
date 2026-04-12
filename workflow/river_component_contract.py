from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _clean_source(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_centerline_station_contract(path_value: Any) -> dict:
    try:
        if not path_value:
            return {}
        path = Path(str(path_value))
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def resolve_centerline_component_expectation(
    *candidates: Mapping[str, Any] | None,
    station_contract_path: Any = None,
) -> dict:
    """Resolve canonical centerline component metadata from runtime and serialized sources.

    This prefers explicit upstream component metadata when present, but can also recover the
    expectation from the persisted centerline station-contract JSON to avoid downstream
    "expected_components=0 source=unknown" degradation after GPKG roundtrips.
    """
    resolved_candidates = [c for c in candidates if isinstance(c, Mapping)]
    station_contract = load_centerline_station_contract(station_contract_path)

    def _first_source(*keys: str) -> Optional[str]:
        for mapping in (station_contract, *resolved_candidates):
            if not isinstance(mapping, Mapping):
                continue
            for key in keys:
                src = _clean_source(mapping.get(key))
                if src:
                    return src
        return None

    def _max_int(*keys: str) -> int:
        return max(
            [_safe_int(mapping.get(key)) for mapping in (station_contract, *resolved_candidates) if isinstance(mapping, Mapping) for key in keys],
            default=0,
        )

    component_id_source = _first_source(
        "expected_component_source",
        "centerline_component_id_source",
        "structured_component_id_source",
        "component_id_source",
    ) or "unknown"

    component_count_before = _max_int(
        "centerline_component_count_before",
        "structured_component_count_before",
        "component_count_before",
    )
    component_count_after = _max_int(
        "centerline_component_count_after",
        "structured_component_count_after",
        "component_count_after",
        "component_count",
    )
    expected_component_count = max(
        component_count_after,
        _max_int(
            "centerline_expected_component_count",
            "expected_component_count",
        ),
    )

    station_path = str(station_contract_path) if station_contract_path else None
    return {
        "component_id_source": str(component_id_source),
        "component_count_before": int(component_count_before),
        "component_count_after": int(component_count_after),
        "expected_component_count": int(expected_component_count),
        "station_contract_path": station_path,
        "station_contract_loaded": bool(station_contract),
    }
