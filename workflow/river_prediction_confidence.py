from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from river_longitudinal_tendency import (
    _attach_reach_context,
    _distribution,
    _is_authoritative_like,
    _is_measured_xs,
    _is_xs_like,
    _prepare_reach_attributes,
    _read_optional_csv,
)

log = logging.getLogger(__name__)


def _component_support_class(row: pd.Series) -> str:
    return str(row.get("component_support_class", "unknown") or "unknown")


def _component_class_confidence_penalty(component_class: str) -> float:
    cc = str(component_class or "unknown")
    if cc == "tiny_detached_component":
        return 0.30
    if cc == "unsupported_side_component":
        return 0.20
    if cc == "unsupported_mainstem":
        return 0.09
    if cc == "weakly_supported_mainstem":
        return 0.03
    return 0.0


def _component_class_admissibility_threshold(component_class: str, station_measured_xs: bool, station_xs: bool) -> float:
    base = 0.40 if station_measured_xs else (0.45 if station_xs else 0.50)
    cc = str(component_class or "unknown")
    if cc == "tiny_detached_component":
        return max(base, 0.80)
    if cc == "unsupported_side_component":
        return max(base, 0.70)
    if cc == "unsupported_mainstem":
        return max(base, 0.60)
    if cc == "weakly_supported_mainstem":
        return max(base, 0.52)
    return base


def _component_class_is_weak(component_class: str) -> bool:
    return str(component_class or "unknown") in {"unsupported_side_component", "tiny_detached_component"}




def _series_from_group(df: pd.DataFrame, col: str, default: Any = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.Series(df[col], index=df.index)
    return pd.Series([default] * len(df), index=df.index)



def _nanmedian_series(df: pd.DataFrame, col: str) -> float:
    vals = pd.to_numeric(_series_from_group(df, col, np.nan), errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")

def _distance_to_nearest_anchor(stations: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
    out = np.full(stations.shape, np.nan, dtype=np.float32)
    if stations.size == 0 or not np.any(anchor_mask):
        return out
    anchors = np.asarray(stations[anchor_mask], dtype=float)
    for i, station in enumerate(np.asarray(stations, dtype=float)):
        if not np.isfinite(station):
            continue
        out[i] = np.float32(np.min(np.abs(anchors - station)))
    return out


def _support_confidence_from_row(row: pd.Series) -> float:
    if bool(row.get("station_authoritative", False)):
        return 1.0
    unsupported_fraction = float(pd.to_numeric(row.get("unsupported_fraction"), errors="coerce") if row.get("unsupported_fraction") is not None else 0.0)
    authoritative_anchor_fraction = float(pd.to_numeric(row.get("authoritative_anchor_fraction"), errors="coerce") if row.get("authoritative_anchor_fraction") is not None else 0.0)
    xs_support_fraction = float(pd.to_numeric(row.get("xs_support_fraction"), errors="coerce") if row.get("xs_support_fraction") is not None else 0.0)
    junction_fraction = float(pd.to_numeric(row.get("junction_adjustment_station_fraction"), errors="coerce") if row.get("junction_adjustment_station_fraction") is not None else 0.0)
    measured_anchor_fraction = float(pd.to_numeric(row.get("measured_anchor_fraction"), errors="coerce") if row.get("measured_anchor_fraction") is not None else 0.0)
    structure_only_fraction = float(pd.to_numeric(row.get("structure_only_fraction"), errors="coerce") if row.get("structure_only_fraction") is not None else 0.0)
    anchor_distance_m = float(pd.to_numeric(row.get("measured_anchor_distance_m"), errors="coerce") if row.get("measured_anchor_distance_m") is not None else np.nan)
    tendency_delta_m = float(pd.to_numeric(row.get("station_longitudinal_tendency_abs_delta_m"), errors="coerce") if row.get("station_longitudinal_tendency_abs_delta_m") is not None else 0.0)
    realism_delta_m = float(pd.to_numeric(row.get("station_xs_realism_abs_delta_m"), errors="coerce") if row.get("station_xs_realism_abs_delta_m") is not None else 0.0)
    station_xs = bool(row.get("station_xs", False))
    station_measured_xs = bool(row.get("station_measured_xs", False))
    reach_role = str(row.get("reach_role", "interior") or "interior")
    component_class = _component_support_class(row)
    reconciliation_conf = float(pd.to_numeric(row.get("authoritative_reconciliation_confidence"), errors="coerce") if row.get("authoritative_reconciliation_confidence") is not None else np.nan)
    component_bed_fraction = float(pd.to_numeric(row.get("component_support_bed_fraction"), errors="coerce") if row.get("component_support_bed_fraction") is not None else np.nan)
    component_median_distance_m = float(pd.to_numeric(row.get("component_support_median_distance_m"), errors="coerce") if row.get("component_support_median_distance_m") is not None else np.nan)

    conf = 0.08
    conf += 0.48 * max(authoritative_anchor_fraction, 0.0)
    # Measured XS data provides real constraint; broad XS-informed is weaker
    if station_measured_xs:
        conf += 0.18
    elif station_xs:
        conf += 0.06
    conf += 0.18 * max(measured_anchor_fraction, 0.0)
    conf += 0.04 * max(1.0 - structure_only_fraction, 0.0)
    conf -= 0.35 * max(unsupported_fraction, 0.0)
    conf -= 0.22 * max(structure_only_fraction, 0.0)
    conf -= 0.12 * max(junction_fraction, 0.0)
    if np.isfinite(anchor_distance_m):
        conf -= min(max(anchor_distance_m, 0.0) / 200.0 * 0.22, 0.22)
    else:
        conf -= 0.08
    if np.isfinite(tendency_delta_m):
        conf -= min(abs(tendency_delta_m) / 1.0 * 0.18, 0.18)
    if np.isfinite(realism_delta_m):
        conf -= min(abs(realism_delta_m) / 0.75 * 0.14, 0.14)
    if np.isfinite(reconciliation_conf):
        conf += min(max(reconciliation_conf, 0.0), 1.0) * 0.10
    if np.isfinite(component_bed_fraction):
        conf += min(max(component_bed_fraction, 0.0), 1.0) * 0.08
    if _component_class_is_weak(component_class):
        weak_structure_only = structure_only_fraction >= 0.60 and measured_anchor_fraction <= 0.05
        weak_reconciliation = (not np.isfinite(reconciliation_conf)) or reconciliation_conf < 0.20
        weak_bed_support = (not np.isfinite(component_bed_fraction)) or component_bed_fraction < 0.05
        far_component = (not np.isfinite(component_median_distance_m)) or component_median_distance_m >= 120.0
        if weak_structure_only and weak_reconciliation:
            conf -= 0.12
        if weak_bed_support and far_component:
            conf -= 0.10
        if weak_structure_only and weak_bed_support and far_component:
            conf -= 0.06
    conf -= _component_class_confidence_penalty(component_class)
    if "junction" in reach_role:
        conf -= 0.04
    return float(np.clip(conf, 0.02, 0.95))


def _low_support_caution_from_row(row: pd.Series) -> bool:
    if bool(row.get("station_authoritative", False)):
        return False
    conf = float(pd.to_numeric(row.get("prediction_support_confidence"), errors="coerce") if row.get("prediction_support_confidence") is not None else 0.0)
    unsupported_fraction = float(pd.to_numeric(row.get("unsupported_fraction"), errors="coerce") if row.get("unsupported_fraction") is not None else 0.0)
    measured_anchor_fraction = float(pd.to_numeric(row.get("measured_anchor_fraction"), errors="coerce") if row.get("measured_anchor_fraction") is not None else 0.0)
    structure_only_fraction = float(pd.to_numeric(row.get("structure_only_fraction"), errors="coerce") if row.get("structure_only_fraction") is not None else 1.0)
    anchor_distance_m = float(pd.to_numeric(row.get("measured_anchor_distance_m"), errors="coerce") if row.get("measured_anchor_distance_m") is not None else np.nan)
    component_class = _component_support_class(row)
    aggressive_class = _component_class_is_weak(component_class)
    return bool(
        conf < (0.64 if aggressive_class else 0.50)
        or (measured_anchor_fraction <= 0.05 and structure_only_fraction >= 0.58 and (not np.isfinite(anchor_distance_m) or anchor_distance_m >= (50.0 if aggressive_class else 75.0)))
        or (unsupported_fraction >= (0.25 if aggressive_class else 0.40) and structure_only_fraction >= (0.60 if aggressive_class else 0.75))
    )


def _admissible_from_row(row: pd.Series) -> bool:
    if bool(row.get("station_authoritative", False)):
        return True
    conf = float(pd.to_numeric(row.get("prediction_support_confidence"), errors="coerce") if row.get("prediction_support_confidence") is not None else 0.0)
    station_xs = bool(row.get("station_xs", False))
    station_measured_xs = bool(row.get("station_measured_xs", False))
    caution = bool(row.get("prediction_low_support_caution", False))
    measured_anchor_fraction = float(pd.to_numeric(row.get("measured_anchor_fraction"), errors="coerce") if row.get("measured_anchor_fraction") is not None else 0.0)
    structure_only_fraction = float(pd.to_numeric(row.get("structure_only_fraction"), errors="coerce") if row.get("structure_only_fraction") is not None else 1.0)
    anchor_distance_m = float(pd.to_numeric(row.get("measured_anchor_distance_m"), errors="coerce") if row.get("measured_anchor_distance_m") is not None else np.nan)
    component_class = _component_support_class(row)
    reconciliation_conf = float(pd.to_numeric(row.get("authoritative_reconciliation_confidence"), errors="coerce") if row.get("authoritative_reconciliation_confidence") is not None else np.nan)
    component_bed_fraction = float(pd.to_numeric(row.get("component_support_bed_fraction"), errors="coerce") if row.get("component_support_bed_fraction") is not None else np.nan)
    weak_class = _component_class_is_weak(component_class)
    if caution and structure_only_fraction >= 0.60:
        return False
    if measured_anchor_fraction <= 0.05 and structure_only_fraction >= 0.50 and (not np.isfinite(anchor_distance_m) or anchor_distance_m >= (80.0 if weak_class else 100.0)):
        return False
    if weak_class:
        weak_reconciliation = (not np.isfinite(reconciliation_conf)) or reconciliation_conf < 0.30
        weak_bed_support = (not np.isfinite(component_bed_fraction)) or component_bed_fraction < 0.10
        if structure_only_fraction >= 0.40 and measured_anchor_fraction <= 0.03 and weak_reconciliation and weak_bed_support:
            return False
        if caution and measured_anchor_fraction < 0.10 and weak_reconciliation:
            return False
        if caution and structure_only_fraction >= 0.50 and weak_bed_support:
            return False
    threshold = _component_class_admissibility_threshold(component_class, station_measured_xs, station_xs)
    return bool(conf >= threshold)


def _confidence_regime_from_row(row: pd.Series) -> str:
    if bool(row.get("station_authoritative", False)):
        return "authoritative_anchor"
    caution = bool(row.get("prediction_low_support_caution", False))
    conf = float(pd.to_numeric(row.get("prediction_support_confidence"), errors="coerce") if row.get("prediction_support_confidence") is not None else 0.0)
    station_xs = bool(row.get("station_xs", False))
    station_measured_xs = bool(row.get("station_measured_xs", False))
    measured_anchor_fraction = float(pd.to_numeric(row.get("measured_anchor_fraction"), errors="coerce") if row.get("measured_anchor_fraction") is not None else 0.0)
    component_class = _component_support_class(row)
    if measured_anchor_fraction > 0.0 and conf >= 0.55:
        return "measured_anchor_supported"
    if station_measured_xs and conf >= 0.40:
        return "measured_xs_supported"
    if station_xs and conf >= 0.40:
        return "xs_informed"
    if caution and _component_class_is_weak(component_class):
        return "component_class_limited"
    if caution:
        return "low_support_caution"
    return "structure_only"


def apply_prediction_confidence_to_nodes(
    nodes,
    *,
    river_dir: str | Path,
    reach_attributes_path: str | Path | None = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, Dict[str, str], Dict[str, Any]]:
    active_logger = logger or log
    river_dir = Path(river_dir)
    river_dir.mkdir(parents=True, exist_ok=True)
    work = nodes.copy()
    if "station_m" not in work.columns or "bed_z_m" not in work.columns:
        return work, {}, {"available": False}

    work["station_m"] = pd.to_numeric(work["station_m"], errors="coerce")
    work["bed_z_m"] = pd.to_numeric(work["bed_z_m"], errors="coerce")
    work["component_id"] = work.get("component_id", "main").fillna("main").astype(str)
    work["prediction_support_confidence"] = np.float32(np.nan)
    work["prediction_measured_anchor_fraction"] = np.float32(np.nan)
    work["prediction_structure_only_fraction"] = np.float32(np.nan)
    work["prediction_measured_anchor_distance_m"] = np.float32(np.nan)
    work["prediction_low_support_caution"] = False
    work["prediction_admissible"] = False
    work["prediction_confidence_regime"] = None

    reach_df = _prepare_reach_attributes(_read_optional_csv(reach_attributes_path))
    station_rows: list[pd.DataFrame] = []
    component_receipts: list[dict[str, Any]] = []

    for component_id, comp_nodes in work.groupby("component_id", sort=False):
        comp_nodes = comp_nodes.loc[np.isfinite(comp_nodes["station_m"])].copy()
        if comp_nodes.empty:
            continue
        station_records = []
        for station_m, station_group in comp_nodes.groupby("station_m", sort=True):
            finite_rows = station_group.loc[np.isfinite(pd.to_numeric(station_group["bed_z_m"], errors="coerce"))].copy()
            if finite_rows.empty:
                continue
            authoritative_mask = _is_authoritative_like(finite_rows)
            xs_mask = _is_xs_like(finite_rows)
            finite_count = int(len(finite_rows))
            measured_anchor_fraction = float(np.count_nonzero(authoritative_mask) / finite_count) if finite_count > 0 else 0.0
            structure_only_fraction = float(np.count_nonzero((~authoritative_mask) & np.isfinite(pd.to_numeric(finite_rows["bed_z_m"], errors="coerce"))) / finite_count) if finite_count > 0 else 0.0
            tendency_delta = pd.to_numeric(finite_rows.get("longitudinal_tendency_delta_m", np.nan), errors="coerce")
            realism_delta = pd.to_numeric(finite_rows.get("xs_realism_delta_m", np.nan), errors="coerce")
            station_records.append({
                "profile_id": str(component_id),
                "station_m": float(station_m),
                "station_authoritative": bool(np.any(authoritative_mask)),
                "station_xs": bool(np.any(xs_mask)),
                "station_measured_xs": bool(np.any(_is_measured_xs(finite_rows))),
                "measured_anchor_fraction": measured_anchor_fraction,
                "structure_only_fraction": structure_only_fraction,
                "station_longitudinal_tendency_abs_delta_m": float(np.nanmedian(np.abs(tendency_delta.to_numpy(dtype=float)))) if np.any(np.isfinite(tendency_delta.to_numpy(dtype=float))) else 0.0,
                "station_xs_realism_abs_delta_m": float(np.nanmedian(np.abs(realism_delta.to_numpy(dtype=float)))) if np.any(np.isfinite(realism_delta.to_numpy(dtype=float))) else 0.0,
                "component_support_class": str(_series_from_group(finite_rows, "component_support_class", "unknown").dropna().astype(str).mode(dropna=True).iloc[0]) if len(finite_rows) else "unknown",
                "component_support_bed_fraction": _nanmedian_series(finite_rows, "component_support_bed_fraction") if len(finite_rows) else np.nan,
                "authoritative_reconciliation_confidence": _nanmedian_series(finite_rows, "authoritative_reconciliation_confidence") if len(finite_rows) else np.nan,
            })
        if not station_records:
            continue
        station_df = pd.DataFrame(station_records).sort_values("station_m").reset_index(drop=True)
        station_df = _attach_reach_context(station_df, reach_df, str(component_id))
        stations = station_df["station_m"].to_numpy(dtype=float)
        station_df["measured_anchor_distance_m"] = _distance_to_nearest_anchor(
            stations,
            station_df["station_authoritative"].to_numpy(dtype=bool),
        )
        station_df["prediction_support_confidence"] = station_df.apply(_support_confidence_from_row, axis=1).astype(np.float32)
        station_df["prediction_low_support_caution"] = station_df.apply(_low_support_caution_from_row, axis=1).astype(bool)
        station_df["prediction_admissible"] = station_df.apply(_admissible_from_row, axis=1).astype(bool)
        station_df["prediction_confidence_regime"] = station_df.apply(_confidence_regime_from_row, axis=1).astype(str)
        station_rows.append(station_df)

        component_receipts.append({
            "profile_id": str(component_id),
            "station_count": int(len(station_df)),
            "authoritative_station_count": int(station_df["station_authoritative"].astype(bool).sum()),
            "xs_station_count": int(station_df["station_xs"].astype(bool).sum()),
            "admissible_station_count": int(station_df["prediction_admissible"].astype(bool).sum()),
            "low_support_caution_station_count": int(station_df["prediction_low_support_caution"].astype(bool).sum()),
            "confidence_distribution": _distribution(pd.to_numeric(station_df["prediction_support_confidence"], errors="coerce").to_numpy(dtype=float)),
            "anchor_distance_distribution_m": _distribution(pd.to_numeric(station_df["measured_anchor_distance_m"], errors="coerce").to_numpy(dtype=float)),
            "regime_counts": {str(k): int(v) for k, v in station_df["prediction_confidence_regime"].astype(str).value_counts().to_dict().items()},
            "component_class_counts": {str(k): int(v) for k, v in station_df["component_support_class"].astype(str).value_counts().to_dict().items()},
        })

        comp_idx = work["component_id"].astype(str).eq(str(component_id))
        station_lookup = station_df.set_index("station_m")
        station_values = pd.to_numeric(work.loc[comp_idx, "station_m"], errors="coerce")
        work.loc[comp_idx, "prediction_support_confidence"] = station_values.map(station_lookup["prediction_support_confidence"]).astype("float32").to_numpy()
        work.loc[comp_idx, "prediction_measured_anchor_fraction"] = station_values.map(station_lookup["measured_anchor_fraction"]).astype("float32").to_numpy()
        work.loc[comp_idx, "prediction_structure_only_fraction"] = station_values.map(station_lookup["structure_only_fraction"]).astype("float32").to_numpy()
        work.loc[comp_idx, "prediction_measured_anchor_distance_m"] = station_values.map(station_lookup["measured_anchor_distance_m"]).astype("float32").to_numpy()
        work.loc[comp_idx, "prediction_low_support_caution"] = station_values.map(station_lookup["prediction_low_support_caution"]).fillna(False).to_numpy(dtype=bool)
        work.loc[comp_idx, "prediction_admissible"] = station_values.map(station_lookup["prediction_admissible"]).fillna(False).to_numpy(dtype=bool)
        work.loc[comp_idx, "prediction_confidence_regime"] = station_values.map(station_lookup["prediction_confidence_regime"]).to_numpy()

    if not station_rows:
        return work, {}, {"available": False}

    profile_df = pd.concat(station_rows, ignore_index=True)
    profile_path = river_dir / "river_prediction_confidence_profile.csv"
    summary_path = river_dir / "river_prediction_confidence_summary.json"
    profile_df.to_csv(profile_path, index=False)
    summary = {
        "available": True,
        "station_count": int(len(profile_df)),
        "authoritative_station_count": int(profile_df["station_authoritative"].astype(bool).sum()),
        "xs_station_count": int(profile_df["station_xs"].astype(bool).sum()),
        "admissible_station_count": int(profile_df["prediction_admissible"].astype(bool).sum()),
        "low_support_caution_station_count": int(profile_df["prediction_low_support_caution"].astype(bool).sum()),
        "confidence_distribution": _distribution(pd.to_numeric(profile_df["prediction_support_confidence"], errors="coerce").to_numpy(dtype=float)),
        "measured_anchor_fraction_distribution": _distribution(pd.to_numeric(profile_df["measured_anchor_fraction"], errors="coerce").to_numpy(dtype=float)),
        "structure_only_fraction_distribution": _distribution(pd.to_numeric(profile_df["structure_only_fraction"], errors="coerce").to_numpy(dtype=float)),
        "measured_anchor_distance_distribution_m": _distribution(pd.to_numeric(profile_df["measured_anchor_distance_m"], errors="coerce").to_numpy(dtype=float)),
        "confidence_regime_counts": {str(k): int(v) for k, v in profile_df["prediction_confidence_regime"].astype(str).value_counts().to_dict().items()},
        "component_class_counts": {str(k): int(v) for k, v in profile_df["component_support_class"].astype(str).value_counts().to_dict().items()},
        "components": component_receipts,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    active_logger.info(
        "[RIVER][SCIENCE] Prediction confidence built: stations=%d admissible=%d low_support=%d",
        int(summary["station_count"]),
        int(summary["admissible_station_count"]),
        int(summary["low_support_caution_station_count"]),
    )
    return work, {
        "prediction_confidence_profile": str(profile_path),
        "prediction_confidence_summary": str(summary_path),
    }, summary
