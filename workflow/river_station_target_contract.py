from __future__ import annotations

"""Station target table/receipt helpers for channel-frame artifacts."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_counts(series: pd.Series | None) -> dict[str, int]:
    if series is None:
        return {}
    counts = series.fillna("missing").astype(str).replace({"": "missing"}).value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def build_station_target_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = frame.copy() if frame is not None else pd.DataFrame()
    base_cols = [
        "component_id",
        "station_m",
        "active_interior_target_z_m",
        "active_interior_target_source",
        "support_class_canonical",
        "channel_support_class",
        "authoritative_anchor_present",
        "authoritative_bed_support_present",
        "authoritative_bank_margin_present",
        "resolved_backbone_z_m",
        "centerline_bed_z_m",
        "longitudinal_profile_z_m",
    ]
    keep = [c for c in base_cols if c in df.columns]
    out = df[keep].copy() if keep else pd.DataFrame(index=df.index)
    if "component_id" not in out.columns:
        out["component_id"] = df.get("component_id", pd.Series("main", index=df.index)) if not df.empty else pd.Series(dtype="object")
    if "station_m" not in out.columns:
        out["station_m"] = df.get("station_m", pd.Series(dtype="float64"))
    if "active_interior_target_z_m" not in out.columns:
        out["active_interior_target_z_m"] = np.nan
    if "active_interior_target_source" not in out.columns:
        out["active_interior_target_source"] = "missing"
    out["station_m"] = pd.to_numeric(out["station_m"], errors="coerce")
    out["active_interior_target_z_m"] = pd.to_numeric(out["active_interior_target_z_m"], errors="coerce")
    hard_anchor = pd.Series(False, index=out.index)
    for col in ("authoritative_anchor_present", "authoritative_bed_support_present"):
        if col in out.columns:
            hard_anchor |= out[col].fillna(False).astype(bool)
    out["hard_anchor_present"] = hard_anchor.astype(bool)
    sort_cols = [c for c in ("component_id", "station_m") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    summary = {
        "record_count": int(len(out)),
        "finite_target_count": int(np.count_nonzero(np.isfinite(pd.to_numeric(out.get("active_interior_target_z_m", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)))) if len(out) else 0,
        "hard_anchor_present_count": int(out.get("hard_anchor_present", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "hard_anchor_present" in out.columns else 0,
        "target_source_class_counts": _safe_counts(out.get("active_interior_target_source")),
        "support_class_counts": _safe_counts(out.get("support_class_canonical") if "support_class_canonical" in out.columns else out.get("channel_support_class")),
    }
    return out, summary


def write_station_target_artifacts(*, river_dir: str | Path, target_df: pd.DataFrame, summary: dict[str, Any]) -> dict[str, str]:
    base = Path(river_dir)
    contracts = base / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    table_path = contracts / "river_station_targets.csv"
    summary_path = contracts / "river_station_targets_summary.json"
    (target_df if target_df is not None else pd.DataFrame()).to_csv(table_path, index=False)
    summary_path.write_text(json.dumps(_jsonable(summary or {}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"station_targets": str(table_path), "station_targets_summary": str(summary_path)}


__all__ = ["build_station_target_table", "write_station_target_artifacts"]
