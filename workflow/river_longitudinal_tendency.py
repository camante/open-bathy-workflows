from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from river_support_semantics import (
    LONGITUDINAL_CHANNEL_ANCHORED,
    LONGITUDINAL_MEASURED_PROTECTED,
    LONGITUDINAL_WEAK_SUPPORT,
    station_semantics,
)

log = logging.getLogger(__name__)


def _read_optional_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        log.debug("Failed to read optional CSV %s", p, exc_info=True)
        return pd.DataFrame()


def _is_authoritative_like(df: pd.DataFrame) -> np.ndarray:
    z_source = df.get("z_source", pd.Series(["missing"] * len(df), index=df.index)).fillna("missing").astype(str)
    support_class = df.get("graph_solver_support_class", pd.Series(["unsupported"] * len(df), index=df.index)).fillna("unsupported").astype(str)
    hard_lock = df.get("graph_hard_lock", pd.Series([False] * len(df), index=df.index)).fillna(False).astype(bool)
    return (
        hard_lock
        | z_source.eq("authoritative_in_channel")
        | support_class.eq("authoritative_locked")
    ).to_numpy(dtype=bool)


def _is_measured_xs(df: pd.DataFrame) -> np.ndarray:
    """Strict measured XS support, including explicit propagated measured-node flags."""
    z_source = df.get("z_source", pd.Series(["missing"] * len(df), index=df.index)).fillna("missing").astype(str)
    propagated = df.get("node_true_measured_xs_post_filter", pd.Series([False] * len(df), index=df.index)).fillna(False).astype(bool)
    return (z_source.eq("xs_profile_resampled") | propagated).to_numpy(dtype=bool)


def _is_residual_xs(df: pd.DataFrame) -> np.ndarray:
    """Strict residual-only XS provenance, excluding true measured XS."""
    measured = _is_measured_xs(df)
    station_support_mode = df.get("station_support_mode", pd.Series(["missing"] * len(df), index=df.index)).fillna("missing").astype(str)
    support_class = df.get("graph_solver_support_class", pd.Series(["unsupported"] * len(df), index=df.index)).fillna("unsupported").astype(str)
    residual = station_support_mode.eq("xs_residual_only") | support_class.eq("xs_residual_only")
    return residual.to_numpy(dtype=bool) & (~measured)


def _is_indirect_xs(df: pd.DataFrame) -> np.ndarray:
    """XS-influenced but not true measured XS and not strict residual-only."""
    measured = _is_measured_xs(df)
    residual = _is_residual_xs(df)
    candidate_source = df.get("graph_candidate_source", pd.Series(["missing"] * len(df), index=df.index)).fillna("missing").astype(str)
    station_support_mode = df.get("station_support_mode", pd.Series(["missing"] * len(df), index=df.index)).fillna("missing").astype(str)
    indirect = candidate_source.eq("xs_profile_resampled") | station_support_mode.isin(["xs_only", "xs_supported"])
    return indirect.to_numpy(dtype=bool) & (~measured) & (~residual)


def _is_xs_like(df: pd.DataFrame) -> np.ndarray:
    """Broad: nodes with any XS-derived provenance (measured or residual).

    NOTE: This is intentionally broader than _is_measured_xs.  It is used in
    blend-weight calculations where knowing that XS data *influenced* the
    solution (even indirectly) matters for deciding how much to smooth.
    For XS *participation* tracking in the channel surface, use
    _is_measured_xs instead.
    """
    return _is_measured_xs(df) | _is_indirect_xs(df) | _is_residual_xs(df)


def _distribution(values: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95.0)),
        "max": float(np.max(vals)),
    }


def _profile_metrics(values: np.ndarray) -> Dict[str, Dict[str, float]]:
    vals = np.asarray(values, dtype=float)
    out = {
        "adjacent_step_abs_m": _distribution(np.abs(np.diff(vals)) if vals.size >= 2 else np.array([], dtype=float)),
        "second_difference_abs_m": _distribution(np.abs(np.diff(vals, n=2)) if vals.size >= 3 else np.array([], dtype=float)),
    }
    return out


def _station_provenance_row(station_group: pd.DataFrame) -> Dict[str, float | bool | str]:
    measured = _is_measured_xs(station_group)
    indirect = _is_indirect_xs(station_group)
    residual = _is_residual_xs(station_group)
    authoritative = _is_authoritative_like(station_group)
    roles = station_group.get("node_role", pd.Series(["thalweg"] * len(station_group), index=station_group.index)).fillna("thalweg").astype(str)
    n = max(len(station_group), 1)
    channel_mask = roles.isin(["thalweg", "left_inner", "right_inner"]).to_numpy(dtype=bool)
    bank_mask = roles.isin(["left_bank", "right_bank"]).to_numpy(dtype=bool)

    def _frac(mask: np.ndarray, submask: np.ndarray | None = None) -> float:
        if submask is None:
            denom = n
            num = int(np.count_nonzero(mask))
        else:
            denom = int(np.count_nonzero(submask))
            if denom <= 0:
                return 0.0
            num = int(np.count_nonzero(mask & submask))
        return float(num / max(denom, 1))

    station_authoritative_role = station_group.get("station_authoritative_role", pd.Series(["no_authoritative_support"] * len(station_group), index=station_group.index)).fillna("no_authoritative_support").astype(str)
    station_bed_support = station_group.get("station_authoritative_bed_support_present", pd.Series([False] * len(station_group), index=station_group.index)).fillna(False).astype(bool).to_numpy(dtype=bool)
    station_bank_margin = station_group.get("station_authoritative_bank_margin_present", pd.Series([False] * len(station_group), index=station_group.index)).fillna(False).astype(bool).to_numpy(dtype=bool)
    station_role_conf = pd.to_numeric(station_group.get("station_authoritative_role_confidence", pd.Series(np.nan, index=station_group.index)), errors="coerce")
    station_dist_bank = pd.to_numeric(station_group.get("station_authoritative_distance_to_bank_m", pd.Series(np.nan, index=station_group.index)), errors="coerce")
    return {
        "station_authoritative": bool(np.any(authoritative)),
        "station_measured_xs": bool(np.any(measured)),
        "station_indirect_xs": bool(np.any(indirect)),
        "station_residual_xs": bool(np.any(residual)),
        "station_xs": bool(np.any(measured | indirect | residual)),
        "station_true_measured_xs_fraction": _frac(measured),
        "station_indirect_xs_fraction": _frac(indirect),
        "station_residual_xs_fraction": _frac(residual),
        "station_authoritative_fraction": _frac(authoritative),
        "station_authoritative_channel_fraction": _frac(authoritative, channel_mask),
        "station_authoritative_bank_fraction": _frac(authoritative, bank_mask),
        "station_authoritative_role": str(station_authoritative_role.mode(dropna=True).iloc[0]) if len(station_authoritative_role.mode(dropna=True)) else "no_authoritative_support",
        "station_authoritative_role_confidence": float(np.nanmedian(station_role_conf.to_numpy(dtype=float))) if np.any(np.isfinite(station_role_conf.to_numpy(dtype=float))) else np.nan,
        "station_authoritative_distance_to_bank_m": float(np.nanmedian(station_dist_bank.to_numpy(dtype=float))) if np.any(np.isfinite(station_dist_bank.to_numpy(dtype=float))) else np.nan,
        "station_authoritative_bed_support_fraction": float(np.mean(station_bed_support)) if station_bed_support.size else 0.0,
        "station_authoritative_bank_margin_fraction": float(np.mean(station_bank_margin)) if station_bank_margin.size else 0.0,
        "station_authoritative_bed_core_fraction": float(np.mean(station_authoritative_role.eq("authoritative_bed_core").to_numpy(dtype=bool))) if len(station_authoritative_role) else 0.0,
        "station_authoritative_ambiguous_fraction": float(np.mean(station_authoritative_role.eq("authoritative_overbank_or_ambiguous").to_numpy(dtype=bool))) if len(station_authoritative_role) else 0.0,
        "station_provenance_class": (
            "authoritative"
            if bool(np.any(authoritative)) else
            "true_measured_xs"
            if bool(np.any(measured)) else
            "residual_xs"
            if bool(np.any(residual)) else
            "indirect_xs"
            if bool(np.any(indirect)) else
            "structural"
        ),
    }


def _prepare_reach_attributes(reach_df: pd.DataFrame) -> pd.DataFrame:
    if reach_df.empty:
        return pd.DataFrame()
    work = reach_df.copy()
    if "profile_id" not in work.columns:
        return pd.DataFrame()
    work["profile_id"] = work["profile_id"].astype(str)
    for col in [
        "station_min_m", "station_max_m", "unsupported_fraction", "authoritative_anchor_fraction",
        "xs_support_fraction", "supported_fraction", "junction_adjustment_station_fraction", "component_junction_count",
        "measured_support_distance_median_m", "measured_support_distance_min_m", "far_from_measured_fraction",
        "authoritative_bed_support_fraction", "authoritative_bank_margin_fraction", "authoritative_bed_core_fraction",
        "authoritative_ambiguous_fraction", "authoritative_bed_support_distance_median_m",
        "authoritative_bed_support_distance_min_m", "far_from_authoritative_bed_fraction",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        else:
            work[col] = np.nan
    work["reach_role"] = work.get("reach_role", pd.Series(["interior"] * len(work), index=work.index)).fillna("interior").astype(str)
    return work.sort_values(["profile_id", "station_min_m", "station_max_m"]).reset_index(drop=True)


def _prepare_longitudinal_profile_context(profile_df: pd.DataFrame) -> pd.DataFrame:
    if profile_df.empty:
        return pd.DataFrame()
    work = profile_df.copy()
    if "profile_id" not in work.columns:
        return pd.DataFrame()
    work["profile_id"] = work["profile_id"].astype(str)
    for col in [
        "station_m",
        "network_backbone_elevation_m",
        "network_junction_adjustment_m",
        "junction_hierarchy_weight",
        "junction_wse_weight",
        "profile_authoritative_role_confidence",
        "profile_authoritative_distance_to_bank_m",
        "profile_authoritative_normalized_channel_position",
        "profile_authoritative_bed_support_distance_m",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        else:
            work[col] = np.nan
    work["network_backbone_source"] = work.get(
        "network_backbone_source",
        pd.Series(["missing"] * len(work), index=work.index),
    ).fillna("missing").astype(str)
    work["profile_authoritative_role"] = work.get("profile_authoritative_role", pd.Series(["authoritative_overbank_or_ambiguous"] * len(work), index=work.index)).fillna("authoritative_overbank_or_ambiguous").astype(str)
    work["profile_authoritative_bed_support_present"] = work.get("profile_authoritative_bed_support_present", pd.Series([False] * len(work), index=work.index)).fillna(False).astype(bool)
    work["profile_authoritative_bank_margin_present"] = work.get("profile_authoritative_bank_margin_present", pd.Series([False] * len(work), index=work.index)).fillna(False).astype(bool)
    work["profile_far_from_authoritative_bed_support"] = work.get("profile_far_from_authoritative_bed_support", pd.Series([False] * len(work), index=work.index)).fillna(False).astype(bool)
    work["profile_junction_active"] = (
        np.abs(work["network_junction_adjustment_m"].to_numpy(dtype=float)) > 1.0e-6
    ) | work["network_backbone_source"].str.contains("junction", case=False, na=False).to_numpy(dtype=bool)
    return work.loc[np.isfinite(work["station_m"])].sort_values(["profile_id", "station_m"]).reset_index(drop=True)


def _attach_reach_context(stations_df: pd.DataFrame, reach_df: pd.DataFrame, profile_id: str) -> pd.DataFrame:
    out = stations_df.copy()
    defaults = {
        "reach_id": None,
        "reach_role": "interior",
        "unsupported_fraction": 0.0,
        "supported_fraction": np.nan,
        "authoritative_anchor_fraction": 0.0,
        "xs_support_fraction": 0.0,
        "junction_adjustment_station_fraction": 0.0,
        "component_junction_count": 0.0,
        "measured_support_distance_median_m": np.nan,
        "measured_support_distance_min_m": np.nan,
        "far_from_measured_fraction": 0.0,
        "authoritative_bed_support_fraction": 0.0,
        "authoritative_bank_margin_fraction": 0.0,
        "authoritative_bed_core_fraction": 0.0,
        "authoritative_ambiguous_fraction": 0.0,
        "authoritative_bed_support_distance_median_m": np.nan,
        "authoritative_bed_support_distance_min_m": np.nan,
        "far_from_authoritative_bed_fraction": 0.0,
        "primary_authoritative_role": "authoritative_overbank_or_ambiguous",
    }
    for col, default in defaults.items():
        out[col] = default
    if reach_df.empty:
        return out
    sub = reach_df.loc[reach_df["profile_id"].astype(str) == str(profile_id)].copy()
    if sub.empty:
        return out
    for idx, row in out.iterrows():
        station = float(row["station_m"])
        sel = sub.loc[
            np.isfinite(sub["station_min_m"]) & np.isfinite(sub["station_max_m"]) &
            (station >= sub["station_min_m"] - 1.0e-6) &
            (station <= sub["station_max_m"] + 1.0e-6)
        ]
        if sel.empty:
            # nearest-reach fallback is acceptable here because reach bins are contiguous summaries
            centers = 0.5 * (pd.to_numeric(sub["station_min_m"], errors="coerce") + pd.to_numeric(sub["station_max_m"], errors="coerce"))
            if np.any(np.isfinite(centers.to_numpy(dtype=float))):
                nearest = sub.iloc[int(np.nanargmin(np.abs(centers.to_numpy(dtype=float) - station)))]
            else:
                continue
        else:
            nearest = sel.iloc[0]
        for col in defaults.keys():
            if col in nearest.index:
                out.at[idx, col] = nearest[col]
    return out


def _attach_longitudinal_profile_context(stations_df: pd.DataFrame, profile_df: pd.DataFrame, profile_id: str) -> pd.DataFrame:
    out = stations_df.copy()
    defaults = {
        "profile_network_backbone_z_m": np.nan,
        "profile_junction_adjustment_m": 0.0,
        "profile_junction_hierarchy_weight": 1.0,
        "profile_junction_wse_weight": 1.0,
        "profile_junction_distance_m": np.nan,
        "profile_junction_target_z_m": np.nan,
        "profile_junction_active": False,
        "profile_authoritative_role": "authoritative_overbank_or_ambiguous",
        "profile_authoritative_role_confidence": np.nan,
        "profile_authoritative_distance_to_bank_m": np.nan,
        "profile_authoritative_normalized_channel_position": np.nan,
        "profile_authoritative_bed_support_present": False,
        "profile_authoritative_bank_margin_present": False,
        "profile_authoritative_bed_support_distance_m": np.nan,
        "profile_far_from_authoritative_bed_support": False,
    }
    for col, default in defaults.items():
        out[col] = default
    if profile_df.empty:
        return out
    sub = profile_df.loc[profile_df["profile_id"].astype(str) == str(profile_id)].copy()
    if sub.empty:
        return out
    profile_stations = pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float)
    station_vals = pd.to_numeric(out["station_m"], errors="coerce").to_numpy(dtype=float)
    valid_station = np.isfinite(profile_stations)
    if np.count_nonzero(valid_station) < 1:
        return out

    def _interp_numeric(col: str, fill_value: float = np.nan) -> np.ndarray:
        vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        good = valid_station & np.isfinite(vals)
        if np.count_nonzero(good) == 0:
            return np.full(len(out), fill_value, dtype=np.float32)
        if np.count_nonzero(good) == 1:
            return np.full(len(out), float(vals[good][0]), dtype=np.float32)
        order = np.argsort(profile_stations[good])
        xp = profile_stations[good][order]
        fp = vals[good][order]
        return np.interp(station_vals, xp, fp).astype(np.float32)

    out["profile_network_backbone_z_m"] = _interp_numeric("network_backbone_elevation_m")
    out["profile_junction_adjustment_m"] = _interp_numeric("network_junction_adjustment_m", fill_value=0.0)
    out["profile_junction_hierarchy_weight"] = _interp_numeric("junction_hierarchy_weight", fill_value=1.0)
    out["profile_junction_wse_weight"] = _interp_numeric("junction_wse_weight", fill_value=1.0)
    out["profile_authoritative_role_confidence"] = _interp_numeric("profile_authoritative_role_confidence")
    out["profile_authoritative_distance_to_bank_m"] = _interp_numeric("profile_authoritative_distance_to_bank_m")
    out["profile_authoritative_normalized_channel_position"] = _interp_numeric("profile_authoritative_normalized_channel_position")
    out["profile_authoritative_bed_support_distance_m"] = _interp_numeric("profile_authoritative_bed_support_distance_m")
    out["profile_junction_target_z_m"] = out["profile_network_backbone_z_m"]

    junction_mask = sub["profile_junction_active"].fillna(False).astype(bool).to_numpy(dtype=bool)
    if np.any(junction_mask & valid_station):
        junction_stations = profile_stations[junction_mask & valid_station]
        out["profile_junction_distance_m"] = [float(np.min(np.abs(junction_stations - station))) if np.isfinite(station) else np.nan for station in station_vals]
        out["profile_junction_active"] = pd.Series(out["profile_junction_distance_m"], index=out.index).fillna(np.inf).astype(float) <= 1.0e-6
    else:
        out["profile_junction_distance_m"] = np.nan
        out["profile_junction_active"] = False
    nearest_idx = []
    valid_idx = np.flatnonzero(valid_station)
    valid_station_vals = profile_stations[valid_station]
    for station in station_vals:
        if not np.isfinite(station) or valid_station_vals.size == 0:
            nearest_idx.append(None)
        else:
            nearest_idx.append(int(valid_idx[int(np.nanargmin(np.abs(valid_station_vals - station)))]))
    if nearest_idx:
        roles = sub["profile_authoritative_role"].astype(str).tolist() if "profile_authoritative_role" in sub.columns else ["authoritative_overbank_or_ambiguous"] * len(sub)
        bed_flags = sub["profile_authoritative_bed_support_present"].astype(bool).tolist() if "profile_authoritative_bed_support_present" in sub.columns else [False] * len(sub)
        bank_flags = sub["profile_authoritative_bank_margin_present"].astype(bool).tolist() if "profile_authoritative_bank_margin_present" in sub.columns else [False] * len(sub)
        far_bed_flags = sub["profile_far_from_authoritative_bed_support"].astype(bool).tolist() if "profile_far_from_authoritative_bed_support" in sub.columns else [False] * len(sub)
        out["profile_authoritative_role"] = [roles[i] if i is not None else "authoritative_overbank_or_ambiguous" for i in nearest_idx]
        out["profile_authoritative_bed_support_present"] = [bool(bed_flags[i]) if i is not None else False for i in nearest_idx]
        out["profile_authoritative_bank_margin_present"] = [bool(bank_flags[i]) if i is not None else False for i in nearest_idx]
        out["profile_far_from_authoritative_bed_support"] = [bool(far_bed_flags[i]) if i is not None else False for i in nearest_idx]
    return out


def _window_radius_from_row(row: pd.Series) -> int:
    unsupported_fraction = float(pd.to_numeric(row.get("unsupported_fraction"), errors="coerce") if row.get("unsupported_fraction") is not None else 0.0)
    authoritative_fraction = float(pd.to_numeric(row.get("authoritative_anchor_fraction"), errors="coerce") if row.get("authoritative_anchor_fraction") is not None else 0.0)
    xs_fraction = float(pd.to_numeric(row.get("xs_support_fraction"), errors="coerce") if row.get("xs_support_fraction") is not None else 0.0)
    junction_fraction = float(pd.to_numeric(row.get("junction_adjustment_station_fraction"), errors="coerce") if row.get("junction_adjustment_station_fraction") is not None else 0.0)
    role = str(row.get("reach_role", "interior") or "interior")
    radius = 1.5 + 3.0 * max(unsupported_fraction, 0.0) + 1.0 * max(1.0 - xs_fraction, 0.0)
    radius -= 1.5 * max(authoritative_fraction, 0.0)
    if "junction" in role:
        radius -= 0.75
    if junction_fraction > 0.0:
        radius -= 0.75 * min(junction_fraction, 1.0)
    return int(np.clip(np.round(radius), 1, 5))


def _longitudinal_support_regime_from_row(
    row: pd.Series,
    *,
    station_authoritative: bool,
    station_measured_xs: bool,
) -> str:
    return str(
        station_semantics(
            row,
            station_authoritative=station_authoritative,
            station_measured_xs=station_measured_xs,
        ).get("longitudinal_support_regime", LONGITUDINAL_CHANNEL_ANCHORED)
    )


def _blend_weight_from_row(row: pd.Series, *, station_authoritative: bool, station_measured_xs: bool = False) -> float:
    regime = _longitudinal_support_regime_from_row(
        row,
        station_authoritative=station_authoritative,
        station_measured_xs=station_measured_xs,
    )
    if regime == LONGITUDINAL_MEASURED_PROTECTED:
        return 0.0
    unsupported_fraction = float(pd.to_numeric(row.get("unsupported_fraction"), errors="coerce") if row.get("unsupported_fraction") is not None else 0.0)
    authoritative_fraction = float(pd.to_numeric(row.get("authoritative_anchor_fraction"), errors="coerce") if row.get("authoritative_anchor_fraction") is not None else 0.0)
    xs_fraction = float(pd.to_numeric(row.get("xs_support_fraction"), errors="coerce") if row.get("xs_support_fraction") is not None else 0.0)
    junction_fraction = float(pd.to_numeric(row.get("junction_adjustment_station_fraction"), errors="coerce") if row.get("junction_adjustment_station_fraction") is not None else 0.0)
    role = str(row.get("reach_role", "interior") or "interior")
    blend = 0.08 + 0.28 * max(unsupported_fraction, 0.0) + 0.18 * max(1.0 - xs_fraction, 0.0)
    blend += 0.12 * max(authoritative_fraction, 0.0)
    blend *= max(0.25, 1.0 - 0.65 * max(xs_fraction, 0.0))
    if regime == LONGITUDINAL_CHANNEL_ANCHORED:
        blend = min(blend, 0.18)
    elif regime == LONGITUDINAL_WEAK_SUPPORT:
        blend = max(blend, 0.24 + 0.18 * max(unsupported_fraction, 0.0))
    if "junction" in role:
        blend *= 0.55
    if junction_fraction > 0.0:
        blend *= max(0.35, 1.0 - 0.75 * min(junction_fraction, 1.0))
    return float(np.clip(blend, 0.0, 0.72))


def _backbone_target_weight_from_row(
    row: pd.Series,
    *,
    station_authoritative: bool,
    station_measured_xs: bool,
) -> float:
    regime = _longitudinal_support_regime_from_row(
        row,
        station_authoritative=station_authoritative,
        station_measured_xs=station_measured_xs,
    )
    if regime == LONGITUDINAL_MEASURED_PROTECTED:
        return 0.0
    backbone_target = pd.to_numeric(row.get("profile_network_backbone_z_m"), errors="coerce")
    if not np.isfinite(backbone_target):
        return 0.0
    unsupported_fraction = float(pd.to_numeric(row.get("unsupported_fraction"), errors="coerce") if row.get("unsupported_fraction") is not None else 0.0)
    authoritative_fraction = float(pd.to_numeric(row.get("authoritative_anchor_fraction"), errors="coerce") if row.get("authoritative_anchor_fraction") is not None else 0.0)
    xs_fraction = float(pd.to_numeric(row.get("xs_support_fraction"), errors="coerce") if row.get("xs_support_fraction") is not None else 0.0)
    role = str(row.get("reach_role", "interior") or "interior")
    base = 0.10 + 0.55 * max(unsupported_fraction, 0.0) + 0.15 * max(1.0 - xs_fraction, 0.0)
    base *= 1.0 - 0.40 * min(max(authoritative_fraction, 0.0), 1.0)
    if regime == LONGITUDINAL_CHANNEL_ANCHORED:
        base = min(base, 0.30)
    elif regime == LONGITUDINAL_WEAK_SUPPORT:
        base = max(base, 0.32 + 0.22 * max(unsupported_fraction, 0.0))
    if "junction" in role:
        base *= 0.70
    return float(np.clip(base, 0.0, 0.82))


def _profile_direction(values: np.ndarray, backbone_target: np.ndarray | None = None) -> float:
    if backbone_target is not None:
        vals = np.asarray(backbone_target, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            return -1.0 if vals[-1] <= vals[0] else 1.0
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return -1.0
    return -1.0 if arr[-1] <= arr[0] else 1.0


def _apply_low_support_longitudinal_guard(
    values: np.ndarray,
    *,
    anchor_mask: np.ndarray,
    measured_xs_mask: np.ndarray,
    context_df: pd.DataFrame,
    backbone_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(values, dtype=float).copy()
    n = len(out)
    guard_active = np.zeros(n, dtype=bool)
    if n < 2:
        return out.astype(np.float32), guard_active
    direction = _profile_direction(out, backbone_target)
    unsupported = pd.to_numeric(context_df.get("unsupported_fraction"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    xs_fraction = pd.to_numeric(context_df.get("xs_support_fraction"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    low_support = (~anchor_mask) & (~measured_xs_mask) & np.isfinite(out) & ((unsupported >= 0.5) | (xs_fraction <= 0.2))
    if not np.any(low_support):
        return out.astype(np.float32), guard_active
    if backbone_target is None:
        backbone_target = np.full(n, np.nan, dtype=float)
    else:
        backbone_target = np.asarray(backbone_target, dtype=float)
    for i in range(1, n):
        if not low_support[i] or not np.isfinite(out[i - 1]) or not np.isfinite(out[i]):
            continue
        local_allow = 0.12
        if np.isfinite(backbone_target[i]) and np.isfinite(backbone_target[i - 1]):
            local_allow = max(0.08, min(0.30, abs(float(backbone_target[i] - backbone_target[i - 1])) + 0.08))
        if direction < 0.0:
            limit = out[i - 1] + local_allow
            if out[i] > limit:
                out[i] = limit
                guard_active[i] = True
        else:
            limit = out[i - 1] - local_allow
            if out[i] < limit:
                out[i] = limit
                guard_active[i] = True
    return out.astype(np.float32), guard_active


def _junction_target_weight_from_row(
    row: pd.Series,
    *,
    station_authoritative: bool,
    station_measured_xs: bool,
    station_step_m: float,
) -> float:
    regime = _longitudinal_support_regime_from_row(
        row,
        station_authoritative=station_authoritative,
        station_measured_xs=station_measured_xs,
    )
    if regime == LONGITUDINAL_MEASURED_PROTECTED:
        return 0.0
    junction_adjustment = float(pd.to_numeric(row.get("profile_junction_adjustment_m"), errors="coerce") if row.get("profile_junction_adjustment_m") is not None else 0.0)
    junction_distance = float(pd.to_numeric(row.get("profile_junction_distance_m"), errors="coerce") if row.get("profile_junction_distance_m") is not None else np.nan)
    hierarchy_weight = float(pd.to_numeric(row.get("profile_junction_hierarchy_weight"), errors="coerce") if row.get("profile_junction_hierarchy_weight") is not None else 1.0)
    wse_weight = float(pd.to_numeric(row.get("profile_junction_wse_weight"), errors="coerce") if row.get("profile_junction_wse_weight") is not None else 1.0)
    component_junction_count = float(pd.to_numeric(row.get("component_junction_count"), errors="coerce") if row.get("component_junction_count") is not None else 0.0)
    role = str(row.get("reach_role", "interior") or "interior")
    junction_fraction = float(pd.to_numeric(row.get("junction_adjustment_station_fraction"), errors="coerce") if row.get("junction_adjustment_station_fraction") is not None else 0.0)
    has_junction_signal = bool(abs(junction_adjustment) > 1.0e-6 or "junction" in role or component_junction_count > 0 or junction_fraction > 0.0)
    if not has_junction_signal:
        return 0.0
    magnitude_term = min(abs(junction_adjustment) / 1.0, 1.0)
    base = 0.10 + 0.40 * magnitude_term + 0.15 * min(max(junction_fraction, 0.0), 1.0)
    if "junction" in role:
        base = max(base, 0.20)
    if component_junction_count > 0:
        base *= 1.0 + 0.05 * min(component_junction_count, 3.0)
    if np.isfinite(junction_distance):
        dist_scale = max(float(station_step_m), 1.0) * 3.0
        decay = max(0.20, 1.0 - min(junction_distance / max(dist_scale, 1.0), 1.0))
        base *= decay
    hierarchy_norm = np.clip(hierarchy_weight, 0.0, 1.5) / 1.5
    wse_norm = np.clip(wse_weight, 0.0, 1.5) / 1.5
    base *= 0.85 + 0.10 * hierarchy_norm + 0.05 * wse_norm
    if regime == LONGITUDINAL_CHANNEL_ANCHORED:
        base = min(base, 0.25)
    return float(np.clip(base, 0.0, 0.65))


def _smooth_station_profile(stations: np.ndarray, values: np.ndarray, *, anchor_mask: np.ndarray, xs_mask: np.ndarray, measured_xs_mask: np.ndarray | None = None, context_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if values.size < 3:
        zeros = np.zeros(values.shape, dtype=np.float32)
        nans = np.full(values.shape, np.nan, dtype=np.float32)
        flags = np.zeros(values.shape, dtype=bool)
        return values.astype(np.float32), zeros, nans, zeros, nans, zeros, nans, flags
    if measured_xs_mask is None:
        measured_xs_mask = np.zeros(values.shape, dtype=bool)
    out = np.asarray(values, dtype=float).copy()
    station_steps = np.diff(stations)
    step_scale = float(np.nanmedian(station_steps[np.isfinite(station_steps) & (station_steps > 0.0)])) if np.any(np.isfinite(station_steps)) else 1.0
    step_scale = max(step_scale, 1.0)
    blend = np.zeros(values.shape, dtype=np.float32)
    junction_weight = np.zeros(values.shape, dtype=np.float32)
    junction_target = np.full(values.shape, np.nan, dtype=np.float32)
    backbone_weight = np.zeros(values.shape, dtype=np.float32)
    backbone_target = pd.to_numeric(context_df.get("profile_network_backbone_z_m"), errors="coerce").to_numpy(dtype=float) if "profile_network_backbone_z_m" in context_df.columns else np.full(values.shape, np.nan, dtype=float)

    # Pre-compute anchor distances for blend decay
    anchor_stations = stations[anchor_mask | measured_xs_mask] if np.any(anchor_mask | measured_xs_mask) else np.array([], dtype=float)
    anchor_dist_arr = np.full(len(stations), np.inf, dtype=float)
    if anchor_stations.size > 0:
        for i, s in enumerate(stations):
            anchor_dist_arr[i] = float(np.min(np.abs(anchor_stations - s)))

    for _ in range(1):  # Single pass — two passes amplified noise
        prev = out.copy()
        for i in range(len(out)):
            if not np.isfinite(prev[i]):
                continue
            row = context_df.iloc[i]
            local_blend = _blend_weight_from_row(
                row,
                station_authoritative=bool(anchor_mask[i]),
                station_measured_xs=bool(measured_xs_mask[i]),
            )
            # Decay blend with distance to nearest measured anchor
            if np.isfinite(anchor_dist_arr[i]) and anchor_dist_arr[i] > 0:
                dist_decay = max(0.10, 1.0 - min(anchor_dist_arr[i] / max(step_scale * 8.0, 200.0), 1.0))
                local_blend *= dist_decay
            blend[i] = max(blend[i], np.float32(local_blend))
            if local_blend <= 0.0:
                continue
            radius = _window_radius_from_row(row)
            lo = max(0, i - radius)
            hi = min(len(out), i + radius + 1)
            local_vals = prev[lo:hi]
            local_stations = stations[lo:hi]
            valid = np.isfinite(local_vals)
            if np.count_nonzero(valid) < 2:
                continue
            dist = np.abs(local_stations[valid] - stations[i]) / step_scale
            weights = 1.0 / (1.0 + dist)
            local_anchor = anchor_mask[lo:hi][valid]
            if np.any(local_anchor):
                weights = weights + 1.5 * local_anchor.astype(float)
            target = float(np.average(local_vals[valid], weights=weights))
            local_backbone_target = float(backbone_target[i]) if np.isfinite(backbone_target[i]) else np.nan
            local_backbone_weight = _backbone_target_weight_from_row(
                row,
                station_authoritative=bool(anchor_mask[i]),
                station_measured_xs=bool(measured_xs_mask[i]),
            )
            if np.isfinite(local_backbone_target) and local_backbone_weight > 0.0:
                target = (1.0 - local_backbone_weight) * target + local_backbone_weight * local_backbone_target
                backbone_weight[i] = max(backbone_weight[i], np.float32(local_backbone_weight))
            local_junction_target = float(pd.to_numeric(row.get("profile_junction_target_z_m"), errors="coerce") if row.get("profile_junction_target_z_m") is not None else np.nan)
            local_junction_weight = _junction_target_weight_from_row(
                row,
                station_authoritative=bool(anchor_mask[i]),
                station_measured_xs=bool(measured_xs_mask[i]),
                station_step_m=step_scale,
            )
            if np.isfinite(local_junction_target) and local_junction_weight > 0.0:
                target = (1.0 - local_junction_weight) * target + local_junction_weight * local_junction_target
                junction_weight[i] = max(junction_weight[i], np.float32(local_junction_weight))
                junction_target[i] = np.float32(local_junction_target)
            effective_blend = float(np.clip(local_blend + 0.30 * local_backbone_weight * (1.0 - local_blend) + 0.35 * local_junction_weight * (1.0 - local_blend), 0.0, 0.72))
            blend[i] = max(blend[i], np.float32(effective_blend))
            out[i] = (1.0 - effective_blend) * prev[i] + effective_blend * target
        out[anchor_mask] = values[anchor_mask]
        out[measured_xs_mask] = values[measured_xs_mask]

    # ---- Roughness guard: reject deltas that increase local roughness ----
    delta = out - values
    if values.size >= 3:
        orig_steps = np.abs(np.diff(values))
        new_steps = np.abs(np.diff(out))
        # For each interior station, check if the smoothed version is locally
        # rougher than the original.  If so, attenuate the delta.
        for i in range(1, len(values) - 1):
            if anchor_mask[i] or measured_xs_mask[i]:
                continue
            orig_local = max(orig_steps[i - 1], orig_steps[min(i, len(orig_steps) - 1)])
            new_local = max(new_steps[i - 1], new_steps[min(i, len(new_steps) - 1)])
            if new_local > orig_local * 1.10 and abs(delta[i]) > 1.0e-4:
                # Attenuate delta proportionally
                ratio = orig_local / max(new_local, 1.0e-9)
                delta[i] *= ratio
                out[i] = values[i] + delta[i]
    out, low_support_guard_active = _apply_low_support_longitudinal_guard(
        out,
        anchor_mask=anchor_mask,
        measured_xs_mask=measured_xs_mask,
        context_df=context_df,
        backbone_target=backbone_target,
    )
    delta = out - values
    anchor_dist = np.full(len(stations), np.inf, dtype=float)
    anchor_idx = np.flatnonzero(anchor_mask | measured_xs_mask)
    if anchor_idx.size:
        anchor_stations_final = stations[anchor_idx]
        for i, station in enumerate(stations):
            anchor_dist[i] = float(np.min(np.abs(anchor_stations_final - station)))
    else:
        anchor_dist[:] = np.nan
    return (
        out.astype(np.float32),
        delta.astype(np.float32),
        anchor_dist.astype(np.float32),
        junction_weight.astype(np.float32),
        junction_target.astype(np.float32),
        backbone_weight.astype(np.float32),
        np.asarray(backbone_target, dtype=np.float32),
        low_support_guard_active.astype(bool),
    )


def apply_longitudinal_tendency_to_nodes(
    nodes: pd.DataFrame,
    *,
    river_dir: str | Path,
    reach_attributes_path: str | Path | None = None,
    longitudinal_profile_path: str | Path | None = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Any]]:
    active_logger = logger or log
    river_dir = Path(river_dir)
    river_dir.mkdir(parents=True, exist_ok=True)
    if nodes is None or len(nodes) == 0 or "station_m" not in nodes.columns or "component_id" not in nodes.columns:
        return nodes, {}, {"available": False, "reason": "missing_nodes_or_station_columns"}
    work = nodes.copy()
    work["station_m"] = pd.to_numeric(work.get("station_m"), errors="coerce")
    work["bed_z_m"] = pd.to_numeric(work.get("bed_z_m"), errors="coerce")
    work["component_id"] = work.get("component_id", "main").fillna("main").astype(str)
    if "node_role" not in work.columns:
        work["node_role"] = "thalweg"
    work["node_role"] = work["node_role"].fillna("thalweg").astype(str)
    if work.empty or not np.any(work["node_role"].eq("thalweg")):
        return work, {}, {"available": False, "reason": "no_thalweg_nodes"}

    reach_df = _prepare_reach_attributes(_read_optional_csv(reach_attributes_path))
    longitudinal_profile = _prepare_longitudinal_profile_context(_read_optional_csv(longitudinal_profile_path))
    outputs: Dict[str, str] = {}
    component_receipts: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []

    work["bed_z_m_before_longitudinal_tendency"] = work["bed_z_m"].astype(np.float32)
    work["longitudinal_tendency_delta_m"] = np.float32(0.0)
    work["longitudinal_tendency_blend_weight"] = np.float32(0.0)
    work["longitudinal_tendency_anchor_distance_m"] = np.float32(np.nan)
    work["longitudinal_tendency_junction_weight"] = np.float32(0.0)
    work["longitudinal_tendency_junction_target_z_m"] = np.float32(np.nan)
    work["longitudinal_tendency_junction_distance_m"] = np.float32(np.nan)
    work["longitudinal_tendency_backbone_weight"] = np.float32(0.0)
    work["longitudinal_tendency_backbone_target_z_m"] = np.float32(np.nan)
    work["longitudinal_tendency_low_support_guard_active"] = False
    work["longitudinal_support_regime"] = "unknown"
    work["longitudinal_tendency_suppression_reason"] = "unknown"
    work["longitudinal_tendency_smoothing_strength"] = np.float32(0.0)
    work["longitudinal_tendency_reach_id"] = None
    work["longitudinal_tendency_applied"] = False

    for component_id, comp_nodes in work.groupby("component_id", sort=False):
        thalweg = comp_nodes.loc[comp_nodes["node_role"].astype(str).eq("thalweg")].copy()
        thalweg = thalweg.loc[np.isfinite(thalweg["station_m"]) & np.isfinite(thalweg["bed_z_m"])].copy()
        if len(thalweg) < 3:
            continue
        station_df = thalweg.groupby("station_m", as_index=False).agg(
            bed_z_m=("bed_z_m", "median"),
            station_authoritative=("graph_hard_lock", lambda s: bool(np.any(pd.Series(s).fillna(False).astype(bool)))),
            station_xs=("node_role", lambda s: False),
        )
        # recompute station-level masks from original thalweg rows so support provenance is preserved.
        station_provenance_rows = []
        for station_m, station_group in comp_nodes.groupby("station_m", sort=True):
            row = {"station_m": float(station_m)}
            row.update(_station_provenance_row(station_group))
            station_provenance_rows.append(row)
        station_prov_df = pd.DataFrame(station_provenance_rows)
        if not station_prov_df.empty:
            station_df = station_df.drop(columns=[c for c in station_prov_df.columns if c != "station_m" and c in station_df.columns])
            station_df = station_df.merge(station_prov_df, on="station_m", how="left")
        for col in [
            "station_authoritative", "station_xs", "station_measured_xs", "station_indirect_xs", "station_residual_xs",
        ]:
            if col in station_df.columns:
                station_df[col] = station_df[col].fillna(False).astype(bool)
        for col in [
            "station_true_measured_xs_fraction", "station_indirect_xs_fraction", "station_residual_xs_fraction",
            "station_authoritative_fraction", "station_authoritative_channel_fraction", "station_authoritative_bank_fraction",
            "station_authoritative_role_confidence", "station_authoritative_distance_to_bank_m",
            "station_authoritative_bed_support_fraction", "station_authoritative_bank_margin_fraction",
            "station_authoritative_bed_core_fraction", "station_authoritative_ambiguous_fraction",
        ]:
            if col in station_df.columns:
                station_df[col] = pd.to_numeric(station_df[col], errors="coerce").fillna(0.0)
        if "station_provenance_class" in station_df.columns:
            station_df["station_provenance_class"] = station_df["station_provenance_class"].fillna("structural").astype(str)
        if "station_authoritative_role" in station_df.columns:
            station_df["station_authoritative_role"] = station_df["station_authoritative_role"].fillna("no_authoritative_support").astype(str)
        station_df["profile_id"] = str(component_id)
        station_df = _attach_reach_context(station_df, reach_df, str(component_id))
        station_df = _attach_longitudinal_profile_context(station_df, longitudinal_profile, str(component_id))
        station_df["station_source"] = np.where(
            station_df["station_authoritative"],
            "authoritative",
            np.where(
                station_df["station_measured_xs"],
                "measured_xs",
                np.where(
                    station_df.get("station_residual_xs", False),
                    "residual_xs",
                    np.where(station_df.get("station_indirect_xs", False), "indirect_xs", "structural"),
                ),
            ),
        )
        station_df["longitudinal_support_regime"] = [
            _longitudinal_support_regime_from_row(
                row,
                station_authoritative=bool(row["station_authoritative"]),
                station_measured_xs=bool(row.get("station_measured_xs", False)),
            )
            for _, row in station_df.iterrows()
        ]
        station_df["longitudinal_tendency_suppression_reason"] = np.where(
            station_df["longitudinal_support_regime"].eq("longitudinal_measured_protected"),
            "true_measured_or_channel_authoritative",
            np.where(
                station_df["longitudinal_support_regime"].eq("longitudinal_weak_support"),
                "weak_support_backbone_priority",
                "channel_anchored_local_structure",
            ),
        )
        stations = station_df["station_m"].to_numpy(dtype=float)
        before = station_df["bed_z_m"].to_numpy(dtype=float)
        smoothed, delta, anchor_dist, junction_weight, junction_target, backbone_weight, backbone_target_vals, low_support_guard_active = _smooth_station_profile(
            stations,
            before,
            anchor_mask=station_df["station_authoritative"].to_numpy(dtype=bool),
            xs_mask=station_df["station_xs"].to_numpy(dtype=bool),
            measured_xs_mask=station_df["station_measured_xs"].to_numpy(dtype=bool),
            context_df=station_df,
        )
        station_df["bed_z_m_before_longitudinal_tendency"] = before.astype(np.float32)
        station_df["bed_z_m_after_longitudinal_tendency"] = smoothed.astype(np.float32)
        station_df["longitudinal_tendency_delta_m"] = delta.astype(np.float32)
        station_df["longitudinal_tendency_blend_weight"] = [
            _blend_weight_from_row(row, station_authoritative=bool(row["station_authoritative"]), station_measured_xs=bool(row.get("station_measured_xs", False)))
            for _, row in station_df.iterrows()
        ]
        station_df["longitudinal_tendency_smoothing_strength"] = station_df["longitudinal_tendency_blend_weight"].astype(np.float32)
        station_df["longitudinal_tendency_anchor_distance_m"] = anchor_dist.astype(np.float32)
        station_df["station_measured_support_distance_m"] = pd.to_numeric(
            station_df.get("measured_support_distance_median_m", station_df.get("profile_measured_support_distance_m", np.nan)),
            errors="coerce",
        ).astype(np.float32)
        station_df["station_authoritative_bed_support_distance_m"] = pd.to_numeric(
            station_df.get("authoritative_bed_support_distance_median_m", station_df.get("profile_authoritative_bed_support_distance_m", np.nan)),
            errors="coerce",
        ).astype(np.float32)
        station_df["longitudinal_tendency_junction_weight"] = junction_weight.astype(np.float32)
        station_df["longitudinal_tendency_junction_target_z_m"] = junction_target.astype(np.float32)
        station_df["longitudinal_tendency_junction_distance_m"] = pd.to_numeric(station_df.get("profile_junction_distance_m"), errors="coerce").astype(np.float32)
        station_df["longitudinal_tendency_backbone_weight"] = backbone_weight.astype(np.float32)
        station_df["longitudinal_tendency_backbone_target_z_m"] = backbone_target_vals.astype(np.float32)
        station_df["longitudinal_tendency_low_support_guard_active"] = low_support_guard_active.astype(bool)
        station_semantics_rows = [
            station_semantics(
                row,
                station_authoritative=bool(row.get("station_authoritative", False)),
                station_measured_xs=bool(row.get("station_measured_xs", False)),
            )
            for _, row in station_df.iterrows()
        ]
        station_df["station_measured_support_class"] = [str(s.get("station_measured_support_class", "no_authoritative_xs_support")) for s in station_semantics_rows]
        station_df["station_authoritative_bed_support_class"] = [str(s.get("station_authoritative_bed_support_class", "no_authoritative_bed_support")) for s in station_semantics_rows]
        station_df["station_far_from_authoritative_bed"] = [bool(s.get("station_far_from_authoritative_bed", False)) for s in station_semantics_rows]
        station_rows.extend(station_df.to_dict(orient="records"))

        delta_by_station = station_df.set_index("station_m")["longitudinal_tendency_delta_m"].astype(float)
        blend_by_station = station_df.set_index("station_m")["longitudinal_tendency_blend_weight"].astype(float)
        reach_by_station = station_df.set_index("station_m")["reach_id"]
        anchor_dist_by_station = station_df.set_index("station_m")["longitudinal_tendency_anchor_distance_m"].astype(float)
        junction_weight_by_station = station_df.set_index("station_m")["longitudinal_tendency_junction_weight"].astype(float)
        junction_target_by_station = station_df.set_index("station_m")["longitudinal_tendency_junction_target_z_m"].astype(float)
        junction_distance_by_station = station_df.set_index("station_m")["longitudinal_tendency_junction_distance_m"].astype(float)
        backbone_weight_by_station = station_df.set_index("station_m")["longitudinal_tendency_backbone_weight"].astype(float)
        backbone_target_by_station = station_df.set_index("station_m")["longitudinal_tendency_backbone_target_z_m"].astype(float)
        low_support_guard_by_station = station_df.set_index("station_m")["longitudinal_tendency_low_support_guard_active"].astype(bool)
        regime_by_station = station_df.set_index("station_m")["longitudinal_support_regime"].astype(str)
        suppression_reason_by_station = station_df.set_index("station_m")["longitudinal_tendency_suppression_reason"].astype(str)
        measured_support_distance_by_station = station_df.set_index("station_m")["station_measured_support_distance_m"].astype(float)
        reach_distance_median_by_station = station_df.set_index("station_m")["measured_support_distance_median_m"].astype(float)
        reach_distance_min_by_station = station_df.set_index("station_m")["measured_support_distance_min_m"].astype(float)
        far_from_measured_fraction_by_station = station_df.set_index("station_m")["far_from_measured_fraction"].astype(float)
        bed_support_distance_by_station = station_df.set_index("station_m")["station_authoritative_bed_support_distance_m"].astype(float)
        reach_bed_distance_median_by_station = station_df.set_index("station_m")["authoritative_bed_support_distance_median_m"].astype(float)
        reach_bed_distance_min_by_station = station_df.set_index("station_m")["authoritative_bed_support_distance_min_m"].astype(float)
        far_from_bed_fraction_by_station = station_df.set_index("station_m")["far_from_authoritative_bed_fraction"].astype(float)
        component_auth_like = _is_authoritative_like(comp_nodes)
        component_measured_xs = _is_measured_xs(comp_nodes)
        station_vals = comp_nodes["station_m"].to_numpy(dtype=float)
        interp_delta = np.interp(station_vals, delta_by_station.index.to_numpy(dtype=float), delta_by_station.to_numpy(dtype=float)).astype(np.float32)
        interp_blend = np.interp(station_vals, blend_by_station.index.to_numpy(dtype=float), blend_by_station.to_numpy(dtype=float)).astype(np.float32)
        interp_anchor_dist = np.interp(station_vals, anchor_dist_by_station.index.to_numpy(dtype=float), anchor_dist_by_station.to_numpy(dtype=float)).astype(np.float32)
        interp_junction_weight = np.interp(station_vals, junction_weight_by_station.index.to_numpy(dtype=float), junction_weight_by_station.to_numpy(dtype=float)).astype(np.float32)
        interp_junction_target = np.interp(station_vals, junction_target_by_station.index.to_numpy(dtype=float), junction_target_by_station.to_numpy(dtype=float)).astype(np.float32)
        interp_junction_distance = np.interp(station_vals, junction_distance_by_station.index.to_numpy(dtype=float), junction_distance_by_station.to_numpy(dtype=float)).astype(np.float32)
        interp_backbone_weight = np.interp(station_vals, backbone_weight_by_station.index.to_numpy(dtype=float), backbone_weight_by_station.to_numpy(dtype=float)).astype(np.float32)
        interp_backbone_target = np.interp(station_vals, backbone_target_by_station.index.to_numpy(dtype=float), backbone_target_by_station.to_numpy(dtype=float)).astype(np.float32)
        guard_station_vals = low_support_guard_by_station.reindex(delta_by_station.index, fill_value=False)
        interp_low_support_guard = np.interp(station_vals, guard_station_vals.index.to_numpy(dtype=float), guard_station_vals.to_numpy(dtype=float).astype(float)).astype(np.float32) > 0.5
        comp_index = comp_nodes.index.to_numpy()
        work.loc[comp_index, "longitudinal_tendency_delta_m"] = interp_delta
        work.loc[comp_index, "longitudinal_tendency_blend_weight"] = interp_blend
        work.loc[comp_index, "longitudinal_tendency_anchor_distance_m"] = interp_anchor_dist
        work.loc[comp_index, "longitudinal_tendency_junction_weight"] = interp_junction_weight
        work.loc[comp_index, "longitudinal_tendency_junction_target_z_m"] = interp_junction_target
        work.loc[comp_index, "longitudinal_tendency_junction_distance_m"] = interp_junction_distance
        work.loc[comp_index, "longitudinal_tendency_backbone_weight"] = interp_backbone_weight
        work.loc[comp_index, "longitudinal_tendency_backbone_target_z_m"] = interp_backbone_target
        work.loc[comp_index, "longitudinal_tendency_low_support_guard_active"] = interp_low_support_guard
        work.loc[comp_index, "longitudinal_tendency_reach_id"] = [reach_by_station.get(float(s), None) for s in station_vals]
        work.loc[comp_index, "longitudinal_support_regime"] = [regime_by_station.get(float(s), "unknown") for s in station_vals]
        work.loc[comp_index, "longitudinal_tendency_suppression_reason"] = [suppression_reason_by_station.get(float(s), "unknown") for s in station_vals]
        work.loc[comp_index, "longitudinal_tendency_smoothing_strength"] = interp_blend
        work.loc[comp_index, "station_measured_support_distance_m"] = [measured_support_distance_by_station.get(float(s), np.nan) for s in station_vals]
        work.loc[comp_index, "measured_support_distance_median_m"] = [reach_distance_median_by_station.get(float(s), np.nan) for s in station_vals]
        work.loc[comp_index, "measured_support_distance_min_m"] = [reach_distance_min_by_station.get(float(s), np.nan) for s in station_vals]
        work.loc[comp_index, "far_from_measured_fraction"] = [far_from_measured_fraction_by_station.get(float(s), np.nan) for s in station_vals]
        work.loc[comp_index, "station_authoritative_bed_support_distance_m"] = [bed_support_distance_by_station.get(float(s), np.nan) for s in station_vals]
        work.loc[comp_index, "authoritative_bed_support_distance_median_m"] = [reach_bed_distance_median_by_station.get(float(s), np.nan) for s in station_vals]
        work.loc[comp_index, "authoritative_bed_support_distance_min_m"] = [reach_bed_distance_min_by_station.get(float(s), np.nan) for s in station_vals]
        work.loc[comp_index, "far_from_authoritative_bed_fraction"] = [far_from_bed_fraction_by_station.get(float(s), np.nan) for s in station_vals]
        apply_mask = (~component_auth_like) & (~component_measured_xs) & np.isfinite(comp_nodes["bed_z_m"].to_numpy(dtype=float)) & np.isfinite(interp_delta)
        work.loc[comp_index[apply_mask], "bed_z_m"] = (
            work.loc[comp_index[apply_mask], "bed_z_m"].to_numpy(dtype=float) + interp_delta[apply_mask]
        ).astype(np.float32)
        for maybe_col in ["graph_backbone_z_m", "backbone_bed_z_m"]:
            if maybe_col in work.columns:
                vals = pd.to_numeric(work.loc[comp_index, maybe_col], errors="coerce").to_numpy(dtype=float)
                finite = apply_mask & np.isfinite(vals)
                if np.any(finite):
                    work.loc[comp_index[finite], maybe_col] = (vals[finite] + interp_delta[finite]).astype(np.float32)
        work.loc[comp_index[apply_mask], "longitudinal_tendency_applied"] = np.abs(interp_delta[apply_mask]) > 1.0e-6

        before_metrics = _profile_metrics(before)
        after_metrics = _profile_metrics(smoothed)
        adverse_before = np.diff(before)
        adverse_after = np.diff(smoothed)
        direction = _profile_direction(smoothed, backbone_target_vals)
        if direction < 0.0:
            adverse_before = adverse_before[adverse_before > 0.0]
            adverse_after = adverse_after[adverse_after > 0.0]
        else:
            adverse_before = (-adverse_before)[(-adverse_before) > 0.0]
            adverse_after = (-adverse_after)[(-adverse_after) > 0.0]
        component_receipts.append({
            "component_id": str(component_id),
            "station_count": int(len(station_df)),
            "authoritative_station_count": int(np.count_nonzero(station_df["station_authoritative"].to_numpy(dtype=bool))),
            "xs_station_count": int(np.count_nonzero(station_df["station_xs"].to_numpy(dtype=bool))),
            "measured_xs_station_count": int(np.count_nonzero(station_df["station_measured_xs"].to_numpy(dtype=bool))),
            "indirect_xs_station_count": int(np.count_nonzero(station_df.get("station_indirect_xs", False).astype(bool).to_numpy(dtype=bool))),
            "residual_xs_station_count": int(np.count_nonzero(station_df.get("station_residual_xs", False).astype(bool).to_numpy(dtype=bool))),
            "provenance_class_counts": {str(k): int(v) for k, v in station_df["station_provenance_class"].astype(str).value_counts(dropna=False).items()},
            "measured_support_class_counts": {str(k): int(v) for k, v in station_df["station_measured_support_class"].astype(str).value_counts(dropna=False).items()},
            "authoritative_bed_support_class_counts": {str(k): int(v) for k, v in station_df["station_authoritative_bed_support_class"].astype(str).value_counts(dropna=False).items()},
            "longitudinal_support_regime_counts": {str(k): int(v) for k, v in station_df["longitudinal_support_regime"].astype(str).value_counts(dropna=False).items()},
            "adjusted_station_count": int(np.count_nonzero(np.abs(delta) > 1.0e-6)),
            "delta_abs_m": _distribution(np.abs(delta)),
            "junction_targeted_station_count": int(np.count_nonzero(np.isfinite(junction_target) & (junction_weight > 0.0))),
            "junction_weight_summary": _distribution(junction_weight),
            "junction_delta_abs_m": _distribution(np.abs(delta[np.isfinite(junction_target) & (junction_weight > 0.0)])),
            "backbone_targeted_station_count": int(np.count_nonzero(np.isfinite(backbone_target_vals) & (backbone_weight > 0.0))),
            "backbone_weight_summary": _distribution(backbone_weight),
            "low_support_guard_station_count": int(np.count_nonzero(low_support_guard_active)),
            "adverse_step_before_m": _distribution(adverse_before),
            "adverse_step_after_m": _distribution(adverse_after),
            "before": before_metrics,
            "after": after_metrics,
        })

    summary: Dict[str, Any] = {
        "available": bool(component_receipts),
        "component_count": int(len(component_receipts)),
        "adjusted_node_count": int(np.count_nonzero(work.get("longitudinal_tendency_applied", False))),
        "hard_lock_node_count_preserved": int(np.count_nonzero(_is_authoritative_like(work))),
        "junction_targeted_node_count": int(np.count_nonzero(pd.to_numeric(work.get("longitudinal_tendency_junction_weight"), errors="coerce").to_numpy(dtype=float) > 0.0)),
        "backbone_targeted_node_count": int(np.count_nonzero(pd.to_numeric(work.get("longitudinal_tendency_backbone_weight"), errors="coerce").to_numpy(dtype=float) > 0.0)),
        "low_support_guarded_node_count": int(np.count_nonzero(pd.Series(work.get("longitudinal_tendency_low_support_guard_active", False)).fillna(False).astype(bool).to_numpy(dtype=bool))),
        "delta_abs_m": _distribution(np.abs(pd.to_numeric(work.get("longitudinal_tendency_delta_m"), errors="coerce").to_numpy(dtype=float))),
        "junction_weight_summary": _distribution(pd.to_numeric(work.get("longitudinal_tendency_junction_weight"), errors="coerce").to_numpy(dtype=float)),
        "backbone_weight_summary": _distribution(pd.to_numeric(work.get("longitudinal_tendency_backbone_weight"), errors="coerce").to_numpy(dtype=float)),
        "components": component_receipts,
        "station_provenance_class_counts": {str(k): int(v) for k, v in pd.Series([r.get("station_provenance_class", "structural") for r in station_rows], dtype="object").astype(str).value_counts(dropna=False).items()} if station_rows else {},
        "measured_support_class_counts": {str(k): int(v) for k, v in pd.Series([r.get("station_measured_support_class", "no_authoritative_xs_support") for r in station_rows], dtype="object").astype(str).value_counts(dropna=False).items()} if station_rows else {},
        "authoritative_bed_support_class_counts": {str(k): int(v) for k, v in pd.Series([r.get("station_authoritative_bed_support_class", "no_authoritative_bed_support") for r in station_rows], dtype="object").astype(str).value_counts(dropna=False).items()} if station_rows else {},
        "longitudinal_support_regime_counts": {str(k): int(v) for k, v in pd.Series([r.get("longitudinal_support_regime", "unknown") for r in station_rows], dtype="object").astype(str).value_counts(dropna=False).items()} if station_rows else {},
        "true_measured_xs_station_count": int(np.count_nonzero(pd.Series([bool(r.get("station_measured_xs", False)) for r in station_rows], dtype=bool).to_numpy(dtype=bool))),
        "indirect_xs_station_count": int(np.count_nonzero(pd.Series([bool(r.get("station_indirect_xs", False)) for r in station_rows], dtype=bool).to_numpy(dtype=bool))),
        "residual_xs_station_count": int(np.count_nonzero(pd.Series([bool(r.get("station_residual_xs", False)) for r in station_rows], dtype=bool).to_numpy(dtype=bool))),
        "inputs": {
            "reach_attributes": str(reach_attributes_path) if reach_attributes_path else None,
            "longitudinal_profile": str(longitudinal_profile_path) if longitudinal_profile_path else None,
        },
    }
    if not longitudinal_profile.empty and "profile_id" in longitudinal_profile.columns and "station_m" in longitudinal_profile.columns and "bed_elevation_m" in longitudinal_profile.columns:
        try:
            summary["longitudinal_profile_component_count"] = int(pd.Series(longitudinal_profile["profile_id"].astype(str)).nunique())
        except Exception:
            summary["longitudinal_profile_component_count"] = 0

    if component_receipts:
        profile_csv = river_dir / "river_longitudinal_tendency_profile.csv"
        profile_df = pd.DataFrame(station_rows)
        profile_df.to_csv(profile_csv, index=False)
        provenance_cols = [
            "profile_id", "station_m", "station_source", "station_provenance_class",
            "station_authoritative", "station_measured_xs", "station_indirect_xs", "station_residual_xs", "station_xs",
            "station_true_measured_xs_fraction", "station_indirect_xs_fraction", "station_residual_xs_fraction",
            "station_authoritative_fraction", "station_authoritative_channel_fraction", "station_authoritative_bank_fraction",
            "station_authoritative_role", "station_authoritative_role_confidence", "station_authoritative_distance_to_bank_m",
            "station_authoritative_bed_support_fraction", "station_authoritative_bank_margin_fraction",
            "station_authoritative_bed_core_fraction", "station_authoritative_ambiguous_fraction",
            "station_authoritative_bed_support_class", "station_far_from_authoritative_bed",
            "station_authoritative_bed_support_distance_m",
        ]
        provenance_df = profile_df[[c for c in provenance_cols if c in profile_df.columns]].copy()
        provenance_csv = river_dir / "river_xs_provenance_station_summary.csv"
        provenance_df.to_csv(provenance_csv, index=False)
        regime_summary_csv = river_dir / "river_longitudinal_support_regime_summary.csv"
        profile_df[[c for c in ["profile_id", "station_m", "longitudinal_support_regime", "longitudinal_tendency_suppression_reason", "longitudinal_tendency_smoothing_strength", "longitudinal_tendency_backbone_weight", "longitudinal_tendency_low_support_guard_active"] if c in profile_df.columns]].to_csv(regime_summary_csv, index=False)
        provenance_payload = {
            "schema_version": 1,
            "component_count": int(len(component_receipts)),
            "station_count": int(len(provenance_df)),
            "provenance_class_counts": summary.get("station_provenance_class_counts", {}),
            "true_measured_xs_station_count": summary.get("true_measured_xs_station_count", 0),
            "indirect_xs_station_count": summary.get("indirect_xs_station_count", 0),
            "residual_xs_station_count": summary.get("residual_xs_station_count", 0),
        }
        provenance_path = river_dir / "river_xs_provenance_receipt.json"
        provenance_path.write_text(json.dumps(provenance_payload, indent=2), encoding="utf-8")
        summary_path = river_dir / "river_longitudinal_tendency_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        outputs["longitudinal_tendency_profile"] = str(profile_csv)
        outputs["river_xs_provenance_station_summary"] = str(provenance_csv)
        outputs["river_longitudinal_support_regime_summary"] = str(regime_summary_csv)
        outputs["river_xs_provenance_receipt"] = str(provenance_path)
        outputs["longitudinal_tendency_summary"] = str(summary_path)
    else:
        active_logger.info("[RIVER][LONGITUDINAL] Longitudinal tendency pass skipped: insufficient thalweg stations")
    return work, outputs, summary


__all__ = ["apply_longitudinal_tendency_to_nodes", "_is_indirect_xs", "_is_residual_xs", "_station_provenance_row", "_longitudinal_support_regime_from_row"]
