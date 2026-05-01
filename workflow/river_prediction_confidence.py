from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def apply_prediction_confidence_to_nodes(nodes: pd.DataFrame, *, river_dir: str | Path | None = None, reach_attributes_path: str | Path | None = None, logger=None) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    out = nodes.copy() if nodes is not None else pd.DataFrame()
    if "prediction_support_confidence" not in out.columns:
        out["prediction_support_confidence"] = np.float32(0.0)
    summary = {"available": False, "status": "not_applied_in_builtin_linear_workflow_package", "confidence_summary": {"count": 0}}
    return out, {}, summary

__all__ = ["apply_prediction_confidence_to_nodes"]
