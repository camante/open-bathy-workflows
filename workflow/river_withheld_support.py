from __future__ import annotations

"""Deterministic support-point key helpers for withheld-support filtering.

This module is intentionally small and side-effect free.  It does not discover
inputs or change support semantics; it only creates stable row keys so a user
provided withheld-support table can exclude matching authoritative support rows.
"""

from pathlib import Path
from typing import Iterable

import hashlib
import pandas as pd
import numpy as np


_KEY_COLUMNS_PRIORITY: tuple[tuple[str, ...], ...] = (
    ("support_point_key",),
    ("point_id",),
    ("source_id",),
    ("id",),
    ("lon", "lat", "depth_m"),
    ("longitude", "latitude", "depth_m"),
    ("x", "y", "depth_m"),
    ("easting", "northing", "depth_m"),
    ("lon", "lat", "z"),
    ("x", "y", "z"),
)

_ROLE_COLUMNS = (
    "authoritative_role",
    "role",
    "source_role",
    "support_role",
    "source_class",
    "data_source",
    "source",
)


def _has_columns(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(str(c) in df.columns for c in cols)


def _clean_scalar(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    try:
        f = float(value)
        if np.isfinite(f):
            return f"{f:.8g}"
    except Exception:
        pass
    return str(value).strip()


def _row_key(row: pd.Series, cols: tuple[str, ...], index_value: object) -> str:
    if cols == ("support_point_key",):
        text = _clean_scalar(row.get("support_point_key"))
        if text:
            return text
    parts = [_clean_scalar(row.get(c)) for c in cols]
    raw = "|".join([str(c) for c in cols] + parts)
    if not any(parts):
        raw = f"row_index|{_clean_scalar(index_value)}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:24]


def attach_support_point_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a deterministic ``support_point_key`` column.

    Existing non-empty ``support_point_key`` values are preserved.  Rows missing a
    key are keyed from the most specific available identifier/coordinate fields;
    if no useful columns are present, the stable row index is used as a last-resort
    package key so filtering remains reproducible within the same CSV.
    """
    out = df.copy()
    chosen = None
    for cols in _KEY_COLUMNS_PRIORITY:
        if _has_columns(out, cols):
            chosen = cols
            break
    if chosen is None:
        chosen = tuple()
    keys = []
    existing = out["support_point_key"] if "support_point_key" in out.columns else pd.Series([""] * len(out), index=out.index)
    for idx, row in out.iterrows():
        existing_key = _clean_scalar(existing.loc[idx]) if idx in existing.index else ""
        if existing_key:
            keys.append(existing_key)
            continue
        key_cols = chosen if chosen else tuple()
        keys.append(_row_key(row, key_cols, idx))
    out["support_point_key"] = pd.Series(keys, index=out.index, dtype="object")
    return out


def load_withheld_support_keys(path: str | Path) -> set[str]:
    """Load withheld support keys from a CSV-like file."""
    df = pd.read_csv(path)
    if df is None or df.empty:
        return set()
    keyed = attach_support_point_keys(df)
    return {str(v).strip() for v in keyed["support_point_key"].tolist() if str(v).strip()}


def summarize_role_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return counts by the first available support-role/source column."""
    if df is None or getattr(df, "empty", True):
        return {}
    for col in _ROLE_COLUMNS:
        if col in df.columns:
            vals = df[col].fillna("missing").astype(str).str.strip().replace({"": "missing"})
            return {str(k): int(v) for k, v in vals.value_counts(dropna=False).to_dict().items()}
    return {"all": int(len(df))}


__all__ = ["attach_support_point_keys", "load_withheld_support_keys", "summarize_role_counts"]
