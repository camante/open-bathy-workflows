"""River component expectation helpers."""

from __future__ import annotations

from typing import Any


def resolve_centerline_component_expectation(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return dict(metadata)
    return {"component_expectation": "unspecified", "source": str(metadata) if metadata is not None else None}


__all__ = ["resolve_centerline_component_expectation"]
