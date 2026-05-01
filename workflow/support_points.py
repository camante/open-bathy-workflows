from __future__ import annotations
from pathlib import Path
from typing import Any


def load_extra_xyz_points(path: str | Path, *args: Any, **kwargs: Any) -> Any:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"extra_xyz_points_missing:{p}")
    try:
        import pandas as pd
        return pd.read_csv(p, delim_whitespace=True, header=None)
    except Exception as exc:
        raise RuntimeError(f"extra_xyz_points_load_failed:{p}") from exc

__all__ = ["load_extra_xyz_points"]
