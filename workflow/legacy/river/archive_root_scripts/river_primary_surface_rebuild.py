from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def apply_primary_surface_rebuild_to_nodes(nodes: pd.DataFrame, *, river_dir: str | Path | None = None, disabled: bool = False, logger=None) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    out = nodes.copy() if nodes is not None else pd.DataFrame()
    summary = {"available": False, "disabled": bool(disabled), "status": "not_applied_in_builtin_linear_workflow_package", "adjusted_node_count": 0}
    return out, {}, summary

__all__ = ["apply_primary_surface_rebuild_to_nodes"]
