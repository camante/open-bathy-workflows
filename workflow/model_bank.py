#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_bank.py – Bounded, incremental training cache for SDB models.

Goal:
- Accumulate training samples across many AOIs without unbounded disk usage.
- Maintain a bounded "reservoir" of feature-labeled samples (DataFrame).
- Refit models deterministically from the bounded reservoir after each AOI (or periodically).

This is NOT "true online RF" (sklearn RF does not support partial_fit). Instead we:
- Update a bounded reservoir sample (unbiased) with each new AOI's samples.
- Refit the model from that bounded reservoir, so neighboring tiles converge.

All state is stored under bank_dir:
  - bank_meta.json
  - reservoir.pkl.gz     (pandas DataFrame with features + target)
  - rf_model.pkl / stumpf_lr.pkl (written by train.py)
"""

from __future__ import annotations

import base64
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

META_NAME = "bank_meta.json"
RES_NAME = "reservoir.pkl.gz"

def _json_safe(obj: Any) -> Any:
    """Best-effort conversion to JSON safe types."""
    try:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [_json_safe(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        return str(obj)
    except Exception:
        return str(obj)

def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

def _encode_rng_state(state: object) -> str:
    return base64.b64encode(pickle.dumps(state)).decode("ascii")

def _decode_rng_state(s: str) -> object:
    return pickle.loads(base64.b64decode(s.encode("ascii")))

def _default_meta(max_samples: int, seed: int, target_col: str) -> Dict[str, Any]:
    import datetime as _dt
    return {
        "version": 1,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "updated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "max_samples": int(max_samples),
        "seed": int(seed),
        "target_col": str(target_col),
        "n_seen": 0,
        "n_kept": 0,
        "rng_state": _encode_rng_state(np.random.RandomState(seed).get_state()),
        "feature_columns": None,
        "notes": "Bounded reservoir sample for SDB training.",
    }

def load_bank_meta(bank_dir: Path, max_samples: int, seed: int, target_col: str) -> Dict[str, Any]:
    bank_dir.mkdir(parents=True, exist_ok=True)
    meta_p = bank_dir / META_NAME
    if meta_p.exists():
        meta = _load_json(meta_p)
        # reconcile key fields (do not silently shrink)
        meta["max_samples"] = int(meta.get("max_samples", max_samples))
        meta["seed"] = int(meta.get("seed", seed))
        meta["target_col"] = str(meta.get("target_col", target_col))
        if "rng_state" not in meta:
            meta["rng_state"] = _encode_rng_state(np.random.RandomState(meta["seed"]).get_state())
        return meta
    meta = _default_meta(max_samples=max_samples, seed=seed, target_col=target_col)
    _write_json(meta_p, meta)
    return meta

def load_reservoir(bank_dir: Path) -> pd.DataFrame:
    p = bank_dir / RES_NAME
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_pickle(p, compression="gzip")
    except Exception as ex:
        log.warning("Failed to read reservoir %s: %s", p, ex)
        return pd.DataFrame()

def save_reservoir(bank_dir: Path, df: pd.DataFrame) -> None:
    p = bank_dir / RES_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p, compression="gzip")

def _deterministic_row_order(df: pd.DataFrame) -> pd.DataFrame:
    # Stable ordering reduces run-to-run nondeterminism.
    cols = [c for c in ["lon", "longitude", "x", "lat", "latitude", "y", "z", "depth", "depth_m", "bed_elev"] if c in df.columns]
    if cols:
        try:
            return df.sort_values(by=cols).reset_index(drop=True)
        except Exception:
            return df.reset_index(drop=True)
    return df.reset_index(drop=True)

def reservoir_update(existing: pd.DataFrame, new_df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Unbiased reservoir sampling update."""
    if new_df is None or len(new_df) == 0:
        return existing, meta

    max_samples = int(meta["max_samples"])
    target_col = str(meta["target_col"])

    new_df = _deterministic_row_order(new_df)

    # Ensure target exists
    if target_col not in new_df.columns:
        raise ValueError(f"[MODEL_BANK] target column '{target_col}' missing in new training df")

    # Align columns: union schema
    if existing is None or len(existing) == 0:
        existing = pd.DataFrame(columns=list(new_df.columns))

    all_cols = list(dict.fromkeys(list(existing.columns) + list(new_df.columns)))
    existing = existing.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    # RNG state
    rs = np.random.RandomState()
    rs.set_state(_decode_rng_state(meta["rng_state"]))

    n_seen = int(meta.get("n_seen", 0))
    n_kept = int(meta.get("n_kept", len(existing)))

    # Convert existing to list-of-rows for efficient replacement if needed
    if len(existing) > 0:
        reservoir = existing.copy().reset_index(drop=True)
    else:
        reservoir = pd.DataFrame(columns=all_cols)

    replaced = 0
    added = 0

    n_new = len(new_df)
    new_arr = new_df.to_numpy(copy=False)
    for i in range(n_new):
        # Fast path: iterate over numpy rows to avoid per-row Series overhead
        # Note: reservoir_update is deterministic given meta['rng_state'] and row order.
        row = new_arr[i]
        n_seen += 1
        if len(reservoir) < max_samples:
            reservoir.loc[len(reservoir)] = row
            added += 1
            n_kept = len(reservoir)
        else:
            j = int(rs.randint(0, n_seen))
            if j < max_samples:
                reservoir.iloc[j] = row
                replaced += 1
    meta["n_seen"] = int(n_seen)
    meta["n_kept"] = int(len(reservoir))
    meta["last_added"] = int(added)
    meta["last_replaced"] = int(replaced)
    meta["rng_state"] = _encode_rng_state(rs.get_state())
    import datetime as _dt
    meta["updated_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    # Persist feature_columns for sanity
    meta["feature_columns"] = [c for c in reservoir.columns if c != target_col]

    return reservoir, meta

def update_bank(
    bank_dir: Path,
    new_training_df: pd.DataFrame,
    *,
    target_col: str,
    max_samples: int,
    seed: int,
    extra_meta: Optional[Dict[str, Any]] = None,
    schema_cols: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Update model bank reservoir with new labeled samples.
    Returns the updated reservoir and updated meta (NOT automatically written).
    """
    meta = load_bank_meta(bank_dir, max_samples=max_samples, seed=seed, target_col=target_col)
    if extra_meta:
        meta.setdefault("extra", {})
        meta["extra"].update(_json_safe(extra_meta))

    res = load_reservoir(bank_dir)

    # Enforce a stable schema to avoid the reservoir "growing" columns over time
    # (e.g., when upstream feature engineering changes). This keeps disk bounded
    # and makes cross-tile behavior more consistent.
    if schema_cols:
        cols = list(dict.fromkeys([c for c in schema_cols if c]))
        if target_col not in cols:
            cols.append(target_col)
        try:
            if not res.empty:
                res = res.reindex(columns=cols)
        except Exception:
            log.debug("ignored", exc_info=True)
        try:
            new_training_df = new_training_df.reindex(columns=cols)
        except Exception:
            log.debug("ignored", exc_info=True)
        meta["feature_columns"] = [c for c in cols if c != target_col]

    res2, meta2 = reservoir_update(res, new_training_df, meta)
    # write
    save_reservoir(bank_dir, res2)
    _write_json(bank_dir / META_NAME, meta2)
    return res2, meta2
