from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

log = logging.getLogger(__name__)


def _interp_fill_nan(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(vals)
    if not np.any(finite):
        return vals
    if np.count_nonzero(finite) == 1:
        vals[~finite] = float(vals[finite][0])
        return vals
    x = np.arange(vals.size, dtype=float)
    vals[~finite] = np.interp(x[~finite], x[finite], vals[finite])
    return vals


def _rolling_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    vals = pd.Series(np.asarray(values, dtype=float))
    w = int(max(window, 1))
    if w <= 1:
        return vals.to_numpy(dtype=float)
    med = vals.rolling(window=w, center=True, min_periods=max(1, w // 2)).median()
    out = med.to_numpy(dtype=float)
    missing = ~np.isfinite(out)
    if np.any(missing):
        out[missing] = vals.to_numpy(dtype=float)[missing]
    return out


def _robust_station_step(stations: np.ndarray) -> float:
    stations = np.asarray(stations, dtype=float)
    stations = stations[np.isfinite(stations)]
    if stations.size < 2:
        return 1.0
    diffs = np.diff(np.unique(np.sort(stations)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _pava_isotonic(y: np.ndarray, *, increasing: bool, weights: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n <= 1:
        return y.copy()
    work = y.copy()
    if not increasing:
        work = -work
    means = work.copy()
    if weights is None:
        wts = np.ones(n, dtype=float)
    else:
        wts = np.asarray(weights, dtype=float)
        if wts.shape != work.shape:
            raise ValueError("pava_weights_shape_mismatch")
        wts = np.where(np.isfinite(wts) & (wts > 0.0), wts, 1.0)
    starts = np.arange(n, dtype=int)
    ends = np.arange(n, dtype=int)
    m = 0
    for i in range(n):
        starts[m] = i
        ends[m] = i
        means[m] = work[i]
        wts[m] = float(wts[i])
        while m > 0 and means[m - 1] > means[m]:
            tot_w = wts[m - 1] + wts[m]
            means[m - 1] = ((means[m - 1] * wts[m - 1]) + (means[m] * wts[m])) / tot_w
            wts[m - 1] = tot_w
            ends[m - 1] = ends[m]
            m -= 1
        m += 1
    out = np.empty(n, dtype=float)
    for b in range(m):
        out[starts[b]: ends[b] + 1] = means[b]
    if not increasing:
        out = -out
    return out


def _component_downstream_increasing(endpoint_meta: Dict[str, Dict[str, object]], component_id: str) -> bool:
    meta = endpoint_meta.get(str(component_id), {}) if endpoint_meta else {}
    return str(meta.get("station_direction", "increasing_station_downstream")) != "decreasing_station_downstream"


def _normalize_component_key(value) -> str:
    if value is None:
        return "main"
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return "main"
        try:
            f = float(s)
            if np.isfinite(f) and abs(f - round(f)) < 1.0e-9:
                return str(int(round(f)))
        except Exception:
            return s
        return s
    try:
        f = float(value)
        if np.isfinite(f) and abs(f - round(f)) < 1.0e-9:
            return str(int(round(f)))
        if np.isfinite(f):
            return f"{f}"
    except Exception:
        pass
    return str(value)


def _series_from_frame(df: pd.DataFrame, column: str, default, *, numeric: bool = False) -> pd.Series:
    if column in df.columns:
        series = df[column]
    else:
        series = pd.Series(default, index=df.index)
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=df.index)
    series = pd.Series(series, index=df.index)
    if numeric:
        return pd.to_numeric(series, errors="coerce")
    return series


def load_bank_points(bank_points_path: str | Path | None) -> gpd.GeoDataFrame:
    cols = [
        "component_id", "river_id", "side", "s_center_m", "bank_z_raw_m", "bank_z_m",
        "bank_z_final_m", "bank_longitudinal_ref_m", "continuity_weight",
        "bank_high_contamination_suspect", "bank_strong_contamination",
        "bank_qc_action", "bank_qc_weight",
    ]
    empty = gpd.GeoDataFrame({c: [] for c in cols}, geometry=[], crs=None)
    if bank_points_path is None:
        return empty
    path = Path(bank_points_path)
    if not path.exists():
        return empty
    try:
        gdf = gpd.read_file(path)
    except Exception:
        log.debug("load_bank_points: failed reading %s", path, exc_info=True)
        return empty
    if gdf.empty:
        return empty
    out = gdf.copy()
    if "component_id" not in out.columns:
        out["component_id"] = "main"
    out["component_id"] = out["component_id"].map(_normalize_component_key)
    if "river_id" not in out.columns:
        out["river_id"] = "unknown"
    out["river_id"] = out["river_id"].fillna("unknown").astype(str)
    if "side" not in out.columns:
        out["side"] = "unknown"
    out["side"] = out["side"].fillna("unknown").astype(str)
    out["s_center_m"] = _series_from_frame(out, "s_center_m", np.nan, numeric=True)
    raw_fallback = _series_from_frame(out, "bank_z_m", np.nan, numeric=True)
    out["bank_z_raw_m"] = _series_from_frame(out, "bank_z_raw_m", raw_fallback, numeric=True)
    out["bank_z_m"] = _series_from_frame(out, "bank_z_m", out["bank_z_raw_m"], numeric=True)
    out["bank_z_final_m"] = _series_from_frame(out, "bank_z_final_m", out["bank_z_m"], numeric=True)
    out["bank_longitudinal_ref_m"] = _series_from_frame(out, "bank_longitudinal_ref_m", np.nan, numeric=True)
    out["continuity_weight"] = _series_from_frame(out, "continuity_weight", 1.0, numeric=True).fillna(1.0)
    out["bank_high_contamination_suspect"] = _series_from_frame(out, "bank_high_contamination_suspect", False).fillna(False).astype(bool)
    out["bank_strong_contamination"] = _series_from_frame(out, "bank_strong_contamination", False).fillna(False).astype(bool)
    out["bank_qc_action"] = _series_from_frame(out, "bank_qc_action", "keep").fillna("keep").astype(str)
    out["bank_qc_weight"] = _series_from_frame(out, "bank_qc_weight", 1.0, numeric=True).fillna(1.0).clip(lower=0.0, upper=1.0)
    out = out[np.isfinite(out["s_center_m"].to_numpy(dtype=float))].copy()
    return out


def fit_bank_longitudinal_points(
    bank_points: gpd.GeoDataFrame,
    *,
    endpoint_meta: Dict[str, Dict[str, object]],
    smoothing_span_m: float = 120.0,
) -> tuple[gpd.GeoDataFrame, Dict[str, Dict[str, float]]]:
    if bank_points is None or getattr(bank_points, "empty", True):
        return load_bank_points(None), {}
    out = bank_points.copy()
    out["bank_fit_raw_m"] = pd.to_numeric(out.get("bank_z_final_m", out.get("bank_z_m")), errors="coerce")
    out["bank_fit_smoothed_m"] = np.nan
    out["bank_fit_monotone_m"] = np.nan
    out["bank_fit_adjustment_m"] = 0.0
    out["bank_fit_residual_m"] = np.nan
    out["bank_fit_supported"] = False
    diagnostics: Dict[str, Dict[str, float]] = {}

    for (component_id, side), idx in out.groupby(["component_id", "side"], sort=False).groups.items():
        sub = out.loc[idx].sort_values("s_center_m").copy()
        stations = pd.to_numeric(sub["s_center_m"], errors="coerce").to_numpy(dtype=float)
        raw = pd.to_numeric(sub.get("bank_z_final_m", sub.get("bank_z_m")), errors="coerce").to_numpy(dtype=float)
        ref = pd.to_numeric(sub.get("bank_longitudinal_ref_m"), errors="coerce").to_numpy(dtype=float)
        continuity = _series_from_frame(sub, "continuity_weight", 1.0, numeric=True).fillna(1.0).to_numpy(dtype=float)
        suspect = _series_from_frame(sub, "bank_high_contamination_suspect", False).fillna(False).to_numpy(dtype=bool)
        strong = _series_from_frame(sub, "bank_strong_contamination", False).fillna(False).to_numpy(dtype=bool)
        qc_action = _series_from_frame(sub, "bank_qc_action", "keep").fillna("keep").astype(str).to_numpy(dtype=object)
        qc_weight = np.clip(_series_from_frame(sub, "bank_qc_weight", 1.0, numeric=True).fillna(1.0).to_numpy(dtype=float), 0.0, 1.0)
        reject = np.asarray([str(v).lower() == "reject" for v in qc_action], dtype=bool)
        raw = np.where(reject, np.nan, raw)
        ref = np.where(reject, np.nan, ref)
        valid = np.isfinite(raw)
        if np.count_nonzero(valid) < 1:
            continue
        step = _robust_station_step(stations)
        window = int(np.clip(round(max(5.0, smoothing_span_m / max(step, 1.0))), 3, 15))
        smooth_seed = np.where(np.isfinite(ref), ref, raw)
        smooth_seed = _rolling_nanmedian(smooth_seed, window)
        smooth_seed = _interp_fill_nan(smooth_seed) if np.any(np.isfinite(smooth_seed)) else smooth_seed
        weights = np.clip(np.nan_to_num(continuity, nan=1.0), 0.15, 1.0)
        weights = weights * np.clip(np.nan_to_num(qc_weight, nan=1.0), 0.0, 1.0)
        weights = np.where(suspect, weights * 0.65, weights)
        weights = np.where(strong, weights * 0.35, weights)
        weights = np.where(reject, 0.0, weights)
        downstream_increasing = _component_downstream_increasing(endpoint_meta, str(component_id))
        finite = np.isfinite(smooth_seed)
        monotone = smooth_seed.copy()
        if np.count_nonzero(finite) >= 2:
            ordered_vals = smooth_seed[finite] if downstream_increasing else smooth_seed[finite][::-1]
            ordered_wts = weights[finite] if downstream_increasing else weights[finite][::-1]
            ordered_fit = _pava_isotonic(ordered_vals, increasing=False, weights=ordered_wts)
            ordered_fit = ordered_fit if downstream_increasing else ordered_fit[::-1]
            monotone[finite] = ordered_fit
        elif np.count_nonzero(finite) == 1:
            monotone[finite] = smooth_seed[finite]
        smooth_seed[reject] = np.nan
        monotone[reject] = np.nan
        out.loc[sub.index, "bank_fit_smoothed_m"] = smooth_seed.astype(np.float32)
        out.loc[sub.index, "bank_fit_monotone_m"] = monotone.astype(np.float32)
        out.loc[sub.index, "bank_fit_adjustment_m"] = (monotone - raw).astype(np.float32)
        out.loc[sub.index, "bank_fit_residual_m"] = (raw - monotone).astype(np.float32)
        out.loc[sub.index, "bank_fit_supported"] = np.isfinite(monotone)

        pre = smooth_seed[finite] if downstream_increasing else smooth_seed[finite][::-1]
        post = monotone[finite] if downstream_increasing else monotone[finite][::-1]
        comp_key = _normalize_component_key(component_id)
        diagnostics[f"{comp_key}:{side}"] = {
            "n_rows": int(len(sub)),
            "raw_rows": int(np.count_nonzero(np.isfinite(raw))),
            "supported_rows": int(np.count_nonzero(np.isfinite(monotone))),
            "raw_range_m": float(np.nanmax(raw) - np.nanmin(raw)) if np.count_nonzero(np.isfinite(raw)) >= 2 else 0.0,
            "fit_range_m": float(np.nanmax(monotone) - np.nanmin(monotone)) if np.count_nonzero(np.isfinite(monotone)) >= 2 else 0.0,
            "pre_monotone_violation_count": int(np.count_nonzero(np.diff(pre) > 0.0)) if pre.size >= 2 else 0,
            "post_monotone_violation_count": int(np.count_nonzero(np.diff(post) > 1.0e-6)) if post.size >= 2 else 0,
            "max_adjustment_m": float(np.nanmax(np.abs(monotone - raw))) if np.count_nonzero(np.isfinite(monotone) & np.isfinite(raw)) >= 1 else 0.0,
            "median_adjustment_m": float(np.nanmedian(np.abs(monotone - raw))) if np.count_nonzero(np.isfinite(monotone) & np.isfinite(raw)) >= 1 else 0.0,
            "strong_contamination_rows": int(np.count_nonzero(strong)),
            "rejected_rows": int(np.count_nonzero(reject)),
            "downgraded_rows": int(np.count_nonzero([str(v).lower() == "downgrade" for v in qc_action])),
            "clamped_rows": int(np.count_nonzero([str(v).lower() == "clamp" for v in qc_action])),
            "replaced_rows": int(np.count_nonzero([str(v).lower() == "replace_with_local_envelope" for v in qc_action])),
        }
    return out, diagnostics


def _interp_profile(series_x: np.ndarray, series_y: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    series_x = np.asarray(series_x, dtype=float)
    series_y = np.asarray(series_y, dtype=float)
    query_x = np.asarray(query_x, dtype=float)
    valid = np.isfinite(series_x) & np.isfinite(series_y)
    if np.count_nonzero(valid) == 0:
        return np.full(query_x.shape, np.nan, dtype=float)
    sx = series_x[valid]
    sy = series_y[valid]
    order = np.argsort(sx)
    sx = sx[order]
    sy = sy[order]
    if sx.size == 1:
        return np.full(query_x.shape, float(sy[0]), dtype=float)
    return np.interp(query_x, sx, sy, left=float(sy[0]), right=float(sy[-1]))


def _low_biased_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    out = np.full(left.shape, np.nan, dtype=float)
    has_left = np.isfinite(left)
    has_right = np.isfinite(right)
    both = has_left & has_right
    if np.any(both):
        low = np.minimum(left[both], right[both])
        high = np.maximum(left[both], right[both])
        avg = 0.5 * (left[both] + right[both])
        asym = high - low
        low_bias = np.clip((asym - 0.25) / 0.75, 0.0, 1.0)
        out[both] = np.where(asym >= 1.0, low, (low_bias * low) + ((1.0 - low_bias) * avg))
    out[has_left & ~has_right] = left[has_left & ~has_right]
    out[has_right & ~has_left] = right[has_right & ~has_left]
    return out


def sample_bank_fit_to_profile(profile: pd.DataFrame, bank_points_fit: gpd.GeoDataFrame) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    if profile.empty:
        return profile, {}
    out = profile.copy()
    for col in [
        "left_bank_raw_m", "right_bank_raw_m", "left_bank_fit_z_m", "right_bank_fit_z_m",
        "bank_pair_raw_m", "bank_pair_fit_z_m", "bank_pair_fit_residual_m", "bank_pair_fit_supported",
    ]:
        out[col] = np.nan if col != "bank_pair_fit_supported" else False
    diagnostics: Dict[str, Dict[str, float]] = {}
    if bank_points_fit is None or getattr(bank_points_fit, "empty", True):
        return out, diagnostics

    work = bank_points_fit.copy()
    work["component_id"] = work["component_id"].map(_normalize_component_key)
    work["side"] = work["side"].fillna("unknown").astype(str)

    for comp, idx in out.groupby("profile_id", sort=False).groups.items():
        sub = out.loc[idx].sort_values("station_m").copy()
        stations = pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float)
        comp_key = _normalize_component_key(comp)
        comp_points = work.loc[work["component_id"] == comp_key].copy()
        if comp_points.empty:
            continue
        left_points = comp_points.loc[comp_points["side"].astype(str) == "left"].sort_values("s_center_m")
        right_points = comp_points.loc[comp_points["side"].astype(str) == "right"].sort_values("s_center_m")
        left_raw = _interp_profile(left_points.get("s_center_m", np.array([])), pd.to_numeric(left_points.get("bank_fit_raw_m"), errors="coerce"), stations)
        right_raw = _interp_profile(right_points.get("s_center_m", np.array([])), pd.to_numeric(right_points.get("bank_fit_raw_m"), errors="coerce"), stations)
        left_fit = _interp_profile(left_points.get("s_center_m", np.array([])), pd.to_numeric(left_points.get("bank_fit_monotone_m"), errors="coerce"), stations)
        right_fit = _interp_profile(right_points.get("s_center_m", np.array([])), pd.to_numeric(right_points.get("bank_fit_monotone_m"), errors="coerce"), stations)
        pair_raw = _low_biased_pair(left_raw, right_raw)
        pair_fit = _low_biased_pair(left_fit, right_fit)
        out.loc[sub.index, "left_bank_raw_m"] = left_raw.astype(np.float32)
        out.loc[sub.index, "right_bank_raw_m"] = right_raw.astype(np.float32)
        out.loc[sub.index, "left_bank_fit_z_m"] = left_fit.astype(np.float32)
        out.loc[sub.index, "right_bank_fit_z_m"] = right_fit.astype(np.float32)
        out.loc[sub.index, "bank_pair_raw_m"] = pair_raw.astype(np.float32)
        out.loc[sub.index, "bank_pair_fit_z_m"] = pair_fit.astype(np.float32)
        out.loc[sub.index, "bank_pair_fit_residual_m"] = (pair_raw - pair_fit).astype(np.float32)
        out.loc[sub.index, "bank_pair_fit_supported"] = np.isfinite(pair_fit)
        diagnostics[comp_key] = {
            "n_rows": int(len(sub)),
            "left_fit_rows": int(np.count_nonzero(np.isfinite(left_fit))),
            "right_fit_rows": int(np.count_nonzero(np.isfinite(right_fit))),
            "pair_fit_rows": int(np.count_nonzero(np.isfinite(pair_fit))),
            "pair_raw_range_m": float(np.nanmax(pair_raw) - np.nanmin(pair_raw)) if np.count_nonzero(np.isfinite(pair_raw)) >= 2 else 0.0,
            "pair_fit_range_m": float(np.nanmax(pair_fit) - np.nanmin(pair_fit)) if np.count_nonzero(np.isfinite(pair_fit)) >= 2 else 0.0,
            "pair_fit_adjustment_max_m": float(np.nanmax(np.abs(pair_raw - pair_fit))) if np.count_nonzero(np.isfinite(pair_raw) & np.isfinite(pair_fit)) >= 1 else 0.0,
            "pair_fit_adjustment_median_m": float(np.nanmedian(np.abs(pair_raw - pair_fit))) if np.count_nonzero(np.isfinite(pair_raw) & np.isfinite(pair_fit)) >= 1 else 0.0,
        }
    return out, diagnostics


__all__ = [
    "load_bank_points",
    "fit_bank_longitudinal_points",
    "sample_bank_fit_to_profile",
]
