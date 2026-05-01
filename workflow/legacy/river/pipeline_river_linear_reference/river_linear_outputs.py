from __future__ import annotations

from pathlib import Path

from pipeline.river_linear.river_linear_contract import FINAL_USER_DEM_RELATIVE_PATH
from pipeline.river_linear.river_linear_paths import RiverLinearPaths, build_river_linear_paths


def final_user_dem_path(out_dir: str | Path) -> Path:
    """Return the stable user-facing final DEM path for a run directory."""
    return Path(out_dir) / FINAL_USER_DEM_RELATIVE_PATH


__all__ = [
    "RiverLinearPaths",
    "build_river_linear_paths",
    "final_user_dem_path",
]
