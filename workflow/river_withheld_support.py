from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_DEFAULT_KEY_COLUMN = "support_point_key"


def _format_coord(series: pd.Series, decimals: int) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = pd.Series([""] * len(vals), index=series.index, dtype=object)
    finite = np.isfinite(vals.to_numpy(dtype=float, copy=False))
    if np.any(finite):
        out.loc[finite] = vals.loc[finite].map(lambda v: f"{float(v):.{decimals}f}")
    return out



def _format_role(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()



def build_support_point_keys(
    df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "depth_m",
    role_col: str = "authoritative_role",
) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=object)
    x = _format_coord(df.get(x_col, pd.Series(np.nan, index=df.index)), 10)
    y = _format_coord(df.get(y_col, pd.Series(np.nan, index=df.index)), 10)
    z = _format_coord(df.get(z_col, pd.Series(np.nan, index=df.index)), 6)
    role = _format_role(df.get(role_col, pd.Series("", index=df.index)))
    return (x + "|" + y + "|" + z + "|" + role).astype(object)



def attach_support_point_keys(
    df: pd.DataFrame,
    *,
    key_col: str = _DEFAULT_KEY_COLUMN,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "depth_m",
    role_col: str = "authoritative_role",
) -> pd.DataFrame:
    out = df.copy()
    if key_col in out.columns and out[key_col].notna().any():
        out[key_col] = out[key_col].fillna("").astype(str)
        return out
    out[key_col] = build_support_point_keys(out, x_col=x_col, y_col=y_col, z_col=z_col, role_col=role_col)
    return out



def load_withheld_support_keys(
    path: Path | str,
    *,
    key_col: str = _DEFAULT_KEY_COLUMN,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "depth_m",
    role_col: str = "authoritative_role",
) -> set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Withheld-support CSV not found: {p}")
    df = pd.read_csv(p)
    df = attach_support_point_keys(df, key_col=key_col, x_col=x_col, y_col=y_col, z_col=z_col, role_col=role_col)
    keys = {str(v) for v in df[key_col].dropna().astype(str).tolist() if str(v)}
    return keys



def summarize_role_counts(df: pd.DataFrame, *, role_col: str = "authoritative_role") -> dict[str, int]:
    if df is None or df.empty or role_col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[role_col].fillna("").astype(str).value_counts(dropna=False).to_dict().items()}


__all__ = [
    "attach_support_point_keys",
    "build_support_point_keys",
    "load_withheld_support_keys",
    "summarize_role_counts",
]
