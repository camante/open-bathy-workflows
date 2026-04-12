from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or window <= 1:
        return arr.copy()
    out = np.full(arr.shape, np.nan, dtype=float)
    half = max(int(window) // 2, 0)
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        sub = arr[lo:hi]
        finite = np.isfinite(sub)
        if np.any(finite):
            out[i] = float(np.mean(sub[finite]))
    return out


def _station_step(stations: np.ndarray) -> float:
    st = np.asarray(stations, dtype=float)
    st = st[np.isfinite(st)]
    if st.size < 2:
        return 1.0
    d = np.diff(np.unique(st))
    d = d[np.isfinite(d) & (d > 0.0)]
    if d.size == 0:
        return 1.0
    return float(np.nanmedian(d))


def _observation_gap_stats(obs_station: np.ndarray, fallback_step: float) -> tuple[np.ndarray, float]:
    if obs_station.size <= 1:
        gap = np.array([max(float(fallback_step), 1.0)], dtype=float)
        return gap, float(gap[0])
    gaps = np.diff(obs_station.astype(float))
    gaps = gaps[np.isfinite(gaps) & (gaps > 0.0)]
    if gaps.size == 0:
        gaps = np.array([max(float(fallback_step), 1.0)], dtype=float)
    return gaps, float(np.nanmedian(gaps))


def _component_class_reconciliation_calibration(component_class: str) -> tuple[float, float, float]:
    cc = str(component_class or "").strip().lower()
    if cc == "unsupported_mainstem":
        return 1.2, 0.9, 1.1
    if cc == "unsupported_side_component":
        return 0.8, 1.15, 0.85
    if cc == "tiny_detached_component":
        return 0.65, 1.3, 0.75
    return 1.0, 1.0, 1.0


def _gap_aware_backbone(
    stations: np.ndarray,
    obs_station: np.ndarray,
    obs_resid: np.ndarray,
    obs_conf: np.ndarray,
    *,
    taper_radius_m: float,
    component_class: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stations = np.asarray(stations, dtype=float)
    obs_station = np.asarray(obs_station, dtype=float)
    obs_resid = np.asarray(obs_resid, dtype=float)
    obs_conf = np.asarray(obs_conf, dtype=float)
    n = stations.size
    if n == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    if obs_station.size == 1:
        return (
            np.full(n, obs_resid[0], dtype=float),
            np.full(n, np.clip(obs_conf[0], 0.0, 1.0), dtype=float),
            np.zeros(n, dtype=float),
        )

    step = _station_step(stations)
    gaps, median_gap = _observation_gap_stats(obs_station, step)
    propagation_scale, gap_penalty_scale, confidence_scale = _component_class_reconciliation_calibration(component_class)
    smooth_scale = max(float(taper_radius_m) * 1.75 * propagation_scale, median_gap * 1.25 * propagation_scale, step * 3.0)
    # Kernel backbone carries residual guidance through sparse support gaps more continuously than
    # straight interpolation, while still respecting observation confidence and spacing.
    dist = np.abs(stations[:, None] - obs_station[None, :])
    kernel = np.exp(-0.5 * np.square(dist / max(smooth_scale, 1.0e-6)))
    kernel *= np.clip(obs_conf[None, :], 0.0, 1.0)
    denom = np.sum(kernel, axis=1)
    weighted = np.sum(kernel * obs_resid[None, :], axis=1)
    residual_backbone = np.divide(weighted, denom, out=np.zeros(n, dtype=float), where=denom > 0.0)
    confidence_backbone = np.divide(np.sum(kernel * obs_conf[None, :], axis=1), denom, out=np.zeros(n, dtype=float), where=denom > 0.0)

    insert = np.searchsorted(obs_station, stations)
    left_idx = np.clip(insert - 1, 0, obs_station.size - 1)
    right_idx = np.clip(insert, 0, obs_station.size - 1)
    gap_span = np.abs(obs_station[right_idx] - obs_station[left_idx])
    gap_span = np.where(insert == 0, np.abs(obs_station[1] - obs_station[0]), gap_span)
    gap_span = np.where(insert >= obs_station.size, np.abs(obs_station[-1] - obs_station[-2]), gap_span)
    gap_span = np.where(np.isfinite(gap_span) & (gap_span > 0.0), gap_span, median_gap)

    # Confidence decays in wide unsupported gaps, but not as abruptly as the prior local taper.
    gap_penalty = 1.0 - 0.35 * gap_penalty_scale * np.clip((gap_span - median_gap) / max(2.0 * median_gap, 1.0), 0.0, 1.0)
    gap_penalty = np.clip(gap_penalty, 0.35 if str(component_class or "").strip().lower() == "unsupported_mainstem" else 0.45, 1.0)

    window = int(np.clip(np.round(smooth_scale / max(step, 1.0)), 3, 21))
    if window % 2 == 0:
        window += 1
    residual_backbone = _rolling_nanmean(residual_backbone, window)
    confidence_backbone = np.clip(_rolling_nanmean(confidence_backbone, max(3, window // 2)), 0.0, 1.0)
    confidence_backbone = np.clip(confidence_backbone * gap_penalty * confidence_scale, 0.0, 1.0)
    return residual_backbone, confidence_backbone, gap_span


def build_authoritative_reconciliation_observations(
    *,
    stations: np.ndarray,
    generalized_bed: np.ndarray,
    authoritative_bed: np.ndarray,
    support_distance_m: np.ndarray,
    fluvial_mask: Optional[np.ndarray] = None,
    exact_anchor_mask: Optional[np.ndarray] = None,
    taper_radius_m: float = 250.0,
) -> pd.DataFrame:
    stations = np.asarray(stations, dtype=float)
    generalized_bed = np.asarray(generalized_bed, dtype=float)
    authoritative_bed = np.asarray(authoritative_bed, dtype=float)
    support_distance_m = np.asarray(support_distance_m, dtype=float)
    n = stations.size
    fluvial = np.ones(n, dtype=bool) if fluvial_mask is None else np.asarray(fluvial_mask, dtype=bool)
    exact_anchor = np.zeros(n, dtype=bool) if exact_anchor_mask is None else np.asarray(exact_anchor_mask, dtype=bool)
    valid = (
        np.isfinite(stations)
        & np.isfinite(generalized_bed)
        & np.isfinite(authoritative_bed)
        & np.isfinite(support_distance_m)
        & fluvial
        & (~exact_anchor)
    )
    cols = [
        "station_m",
        "generalized_bed_elevation_m",
        "authoritative_bed_elevation_m",
        "residual_delta_m",
        "support_distance_m",
        "observation_confidence",
        "exact_anchor",
    ]
    if not np.any(valid):
        return pd.DataFrame(columns=cols)
    obs_station = stations[valid]
    obs_generalized = generalized_bed[valid]
    obs_authoritative = authoritative_bed[valid]
    obs_distance = np.clip(support_distance_m[valid], 0.0, None)
    obs_weight = 0.5 * (1.0 + np.cos(np.pi * np.clip(obs_distance / max(float(taper_radius_m), 1.0e-6), 0.0, 1.0)))
    obs_weight = np.clip(obs_weight, 0.0, 1.0)
    obs = pd.DataFrame(
        {
            "station_m": obs_station,
            "generalized_bed_elevation_m": obs_generalized,
            "authoritative_bed_elevation_m": obs_authoritative,
            "residual_delta_m": obs_authoritative - obs_generalized,
            "support_distance_m": obs_distance,
            "observation_confidence": obs_weight,
            "exact_anchor": np.zeros(np.count_nonzero(valid), dtype=bool),
        }
    )
    return (
        obs.sort_values("station_m")
        .groupby("station_m", as_index=False)
        .agg(
            {
                "generalized_bed_elevation_m": "mean",
                "authoritative_bed_elevation_m": "mean",
                "residual_delta_m": "mean",
                "support_distance_m": "min",
                "observation_confidence": "max",
                "exact_anchor": "max",
            }
        )
    )


def solve_authoritative_reconciliation_field(
    *,
    stations: np.ndarray,
    observations: pd.DataFrame,
    support_distance_m: np.ndarray,
    fluvial_mask: Optional[np.ndarray] = None,
    exact_anchor_mask: Optional[np.ndarray] = None,
    taper_radius_m: float = 250.0,
    max_abs_delta_m: float = 1.5,
    component_class: str = "",
) -> pd.DataFrame:
    stations = np.asarray(stations, dtype=float)
    support_distance_m = np.asarray(support_distance_m, dtype=float)
    n = stations.size
    fluvial = np.ones(n, dtype=bool) if fluvial_mask is None else np.asarray(fluvial_mask, dtype=bool)
    exact_anchor = np.zeros(n, dtype=bool) if exact_anchor_mask is None else np.asarray(exact_anchor_mask, dtype=bool)
    field = pd.DataFrame(
        {
            "station_m": stations,
            "authoritative_reconciliation_delta_m": np.zeros(n, dtype=float),
            "authoritative_reconciliation_weight": np.zeros(n, dtype=float),
            "authoritative_reconciliation_confidence": np.zeros(n, dtype=float),
            "authoritative_reconciliation_support_distance_m": support_distance_m.astype(float, copy=True),
            "authoritative_reconciliation_exact_anchor": exact_anchor.astype(bool, copy=True),
        }
    )
    if n == 0 or observations is None or observations.empty:
        return field
    obs = observations.sort_values("station_m").reset_index(drop=True)
    obs_station = pd.to_numeric(obs["station_m"], errors="coerce").to_numpy(dtype=float)
    obs_resid = pd.to_numeric(obs["residual_delta_m"], errors="coerce").to_numpy(dtype=float)
    obs_conf = np.clip(np.nan_to_num(pd.to_numeric(obs.get("observation_confidence", 1.0), errors="coerce").to_numpy(dtype=float), nan=0.0), 0.0, 1.0)
    valid_obs = np.isfinite(obs_station) & np.isfinite(obs_resid)
    if not np.any(valid_obs):
        return field
    obs_station = obs_station[valid_obs]
    obs_resid = obs_resid[valid_obs]
    obs_conf = obs_conf[valid_obs]

    propagation_scale, gap_penalty_scale, confidence_scale = _component_class_reconciliation_calibration(component_class)

    residual_backbone, confidence_backbone, gap_span = _gap_aware_backbone(
        stations, obs_station, obs_resid, obs_conf, taper_radius_m=taper_radius_m, component_class=component_class
    )

    insert = np.searchsorted(obs_station, stations)
    left_idx = np.clip(insert - 1, 0, obs_station.size - 1)
    right_idx = np.clip(insert, 0, obs_station.size - 1)
    nearest_station_dist = np.minimum(np.abs(stations - obs_station[left_idx]), np.abs(stations - obs_station[right_idx]))
    support_dist = np.where(np.isfinite(support_distance_m), support_distance_m, nearest_station_dist)
    support_dist = np.where(np.isfinite(support_dist), np.clip(support_dist, 0.0, None), np.inf)

    local_weight = 0.5 * (1.0 + np.cos(np.pi * np.clip(support_dist / max(float(taper_radius_m), 1.0e-6), 0.0, 1.0)))
    local_weight = np.where(np.isfinite(support_dist) & (support_dist < taper_radius_m), local_weight, 0.0)
    along_gap_scale = np.clip(1.0 - 0.45 * gap_penalty_scale * np.clip(nearest_station_dist / np.maximum(gap_span, 1.0), 0.0, 1.0), 0.4 if str(component_class or "").strip().lower() == "unsupported_mainstem" else 0.55, 1.0)
    # Carry some residual guidance farther along-channel across sparse support gaps instead of forcing it to die locally.
    propagation_weight = np.exp(-0.5 * np.square(nearest_station_dist / max(float(taper_radius_m) * 2.5 * propagation_scale, 1.0)))
    propagation_weight = np.clip(propagation_weight * along_gap_scale * confidence_scale, 0.0, 1.0)
    weight = np.maximum(local_weight, propagation_weight * np.clip(confidence_backbone, 0.0, 1.0))
    weight = np.where(fluvial, weight, 0.0)
    confidence = np.clip(weight * np.nan_to_num(confidence_backbone, nan=0.0), 0.0, 1.0)
    delta = np.clip(np.nan_to_num(residual_backbone, nan=0.0) * weight, -float(max_abs_delta_m), float(max_abs_delta_m))
    delta = np.where(exact_anchor, 0.0, delta)
    weight = np.where(exact_anchor, 0.0, weight)
    confidence = np.where(exact_anchor, 1.0, confidence)
    support_dist = np.where(exact_anchor, 0.0, support_dist)
    field["authoritative_reconciliation_delta_m"] = delta.astype(float)
    field["authoritative_reconciliation_weight"] = weight.astype(float)
    field["authoritative_reconciliation_confidence"] = confidence.astype(float)
    field["authoritative_reconciliation_support_distance_m"] = support_dist.astype(float)
    return field


__all__ = ["build_authoritative_reconciliation_observations", "solve_authoritative_reconciliation_field"]
