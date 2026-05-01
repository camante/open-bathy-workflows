from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class XSConfig:
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None


def build_xs_for_river(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("legacy_xs_builder_unavailable_in_active_river_workflow_package")


def _read_layer(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("legacy_xs_builder_read_layer_unavailable_in_active_river_workflow_package")

__all__ = ["XSConfig", "build_xs_for_river", "_read_layer"]
