"""Lightweight validation helpers for legacy cross-section artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_soundings_subset_parquet(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"soundings_subset_missing:{p}")
    try:
        import pandas as pd
        df = pd.read_parquet(p)
        depth_col = next((c for c in ("depth_m", "z", "elevation_m") if c in df.columns), None)
        return {"path": str(p), "rows": int(len(df)), "depth_col": depth_col or "unknown"}
    except Exception:
        # Do not invent validity; only confirm the file exists when parquet
        # support is unavailable in the runtime environment.
        return {"path": str(p), "rows": -1, "depth_col": "unknown", "validation_mode": "file_exists_only"}


def validate_xs_artifacts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {"status": "not_evaluated", "reason": "legacy_xs_artifact_validation_not_active_in_builtin_river_workflow"}


__all__ = ["validate_soundings_subset_parquet", "validate_xs_artifacts"]
