from __future__ import annotations

from pathlib import Path

from pipeline.river_workflow.river_workflow_contract import FINAL_USER_DEM_RELATIVE_PATH
from pipeline.river_workflow.river_workflow_paths import RiverWorkflowPaths, build_river_workflow_paths


def final_user_dem_path(out_dir: str | Path) -> Path:
    """Return the stable user-facing final DEM path for a run directory."""
    return Path(out_dir) / FINAL_USER_DEM_RELATIVE_PATH


__all__ = [
    "RiverWorkflowPaths",
    "build_river_workflow_paths",
    "final_user_dem_path",
]
