from __future__ import annotations

"""Longitudinal tendency helpers for legacy channel-surface modules."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _distribution(values: Any) -> dict[str, Any]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None, "p05": None, "p95": None}
    return {
        "count": int(finite.size),
        "min": float(np.nanmin(finite)),
        "median": float(np.nanmedian(finite)),
        "mean": float(np.nanmean(finite)),
        "max": float(np.nanmax(finite)),
        "p05": float(np.nanpercentile(finite, 5.0)),
        "p95": float(np.nanpercentile(finite, 95.0)),
    }


def apply_longitudinal_tendency_to_nodes(
    nodes: pd.DataFrame,
    *,
    river_dir: str | Path | None = None,
    reach_attributes_path: str | Path | None = None,
    longitudinal_profile_path: str | Path | None = None,
    logger=None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    """Return nodes unchanged with an explicit not-applied summary.

    The active built-in canonical river workflow does not route through this legacy
    channel-surface tendency stage.  This implementation keeps import-time package
    integrity while making any legacy use explicit in receipts instead of hiding a
    silent adjustment.
    """
    out = nodes.copy() if nodes is not None else pd.DataFrame()
    summary = {
        "available": False,
        "status": "not_applied_in_builtin_linear_workflow_package",
        "adjusted_node_count": 0,
        "junction_targeted_node_count": 0,
        "delta_abs_m": _distribution(np.array([], dtype=float)),
        "junction_weight_summary": _distribution(np.array([], dtype=float)),
        "reach_attributes_path": str(reach_attributes_path) if reach_attributes_path else None,
        "longitudinal_profile_path": str(longitudinal_profile_path) if longitudinal_profile_path else None,
    }
    if logger is not None:
        logger.info("[RIVER][LONGITUDINAL] Longitudinal tendency not applied in built-in linear package.")
    return out, {}, summary


__all__ = ["apply_longitudinal_tendency_to_nodes", "_distribution"]
