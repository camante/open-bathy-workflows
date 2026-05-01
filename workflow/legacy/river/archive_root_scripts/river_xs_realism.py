from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _distribution(values: Any) -> dict[str, Any]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {"count": int(finite.size), "min": float(np.nanmin(finite)), "median": float(np.nanmedian(finite)), "mean": float(np.nanmean(finite)), "max": float(np.nanmax(finite))}


def apply_xs_realism_to_nodes(nodes: pd.DataFrame, *, river_dir: str | Path | None = None, reach_attributes_path: str | Path | None = None, disabled: bool = False, logger=None) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    out = nodes.copy() if nodes is not None else pd.DataFrame()
    if "target_xs_realism_allowed" not in out.columns:
        out["target_xs_realism_allowed"] = False
    summary = {"available": False, "disabled": bool(disabled), "adjusted_node_count": 0, "residual_distribution": _distribution([]), "status": "not_applied_in_builtin_linear_workflow_package"}
    return out, {}, summary

__all__ = ["apply_xs_realism_to_nodes"]
