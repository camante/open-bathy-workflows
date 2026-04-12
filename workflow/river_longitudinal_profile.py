from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from nodata_utils import nodata_mask
from authoritative_river_roles import ROLE_BED_CORE, ROLE_BED_INNER, ROLE_BANK_MARGIN, ROLE_AMBIGUOUS, role_to_code
from river_authoritative_reconciliation import build_authoritative_reconciliation_observations, solve_authoritative_reconciliation_field
from river_bank_longitudinal_fit import load_bank_points, fit_bank_longitudinal_points, sample_bank_fit_to_profile, _normalize_component_key
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


def _sample_raster_at_points(gdf, raster_path: Optional[str | Path], field: str) -> pd.Series:
    out = pd.Series(np.nan, index=gdf.index, dtype="float32")
    if raster_path is None:
        return out
    path = Path(raster_path)
    if not path.exists() or gdf is None or getattr(gdf, "empty", True):
        return out
    xs = gdf.geometry.x.to_numpy(dtype=float)
    ys = gdf.geometry.y.to_numpy(dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(valid):
        return out
    with rasterio.open(path) as ds:
        pts = list(zip(xs[valid], ys[valid]))
        vals = []
        for v in ds.sample(pts):
            val = float(v[0]) if np.size(v) else np.nan
            if nodata_mask(np.asarray([val], dtype=np.float32), ds.nodata)[0]:
                val = np.nan
            vals.append(val)
    out.loc[gdf.index[valid]] = np.asarray(vals, dtype=np.float32)
    out.name = field
    return out


def _load_authoritative_support_points_table(csv_path: Optional[str | Path]) -> pd.DataFrame:
    cols = [
        "x", "y", "support_z_m", "authoritative_role", "role_confidence", "distance_to_bank_m",
        "component_half_width_est_m", "normalized_channel_position",
        "inside_channel_mask", "inside_river_guidance_domain", "inside_mainstem_mask", "inside_estuary_clip",
    ]
    if csv_path is None:
        return pd.DataFrame(columns=cols)
    path = Path(csv_path)
    if not path.exists() or path.suffix.lower() not in {".csv", ".txt"}:
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception:
        log.debug("failed reading authoritative support table %s", path, exc_info=True)
        return pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    if not {"x", "y"}.issubset(df.columns):
        return pd.DataFrame(columns=cols)
    support_z = pd.to_numeric(df.get("support_z_m", df.get("depth_m")), errors="coerce")
    half_width = pd.to_numeric(df.get("component_half_width_est_m", df.get("half_width_est_m")), errors="coerce")
    out = pd.DataFrame({
        "x": pd.to_numeric(df.get("x"), errors="coerce"),
        "y": pd.to_numeric(df.get("y"), errors="coerce"),
        "support_z_m": support_z,
        "authoritative_role": df.get("authoritative_role", pd.Series([ROLE_AMBIGUOUS] * len(df), index=df.index)).fillna(ROLE_AMBIGUOUS).astype(str),
        "role_confidence": pd.to_numeric(df.get("role_confidence"), errors="coerce"),
        "distance_to_bank_m": pd.to_numeric(df.get("distance_to_bank_m"), errors="coerce"),
        "component_half_width_est_m": half_width,
        "normalized_channel_position": pd.to_numeric(df.get("normalized_channel_position"), errors="coerce"),
        "inside_channel_mask": pd.to_numeric(df.get("inside_channel_mask", pd.Series([0.0] * len(df), index=df.index)), errors="coerce").fillna(0.0),
        "inside_river_guidance_domain": pd.to_numeric(df.get("inside_river_guidance_domain", pd.Series([0.0] * len(df), index=df.index)), errors="coerce").fillna(0.0),
        "inside_mainstem_mask": pd.to_numeric(df.get("inside_mainstem_mask", pd.Series([0.0] * len(df), index=df.index)), errors="coerce").fillna(0.0),
        "inside_estuary_clip": pd.to_numeric(df.get("inside_estuary_clip", pd.Series([0.0] * len(df), index=df.index)), errors="coerce").fillna(0.0),
    })
    out = out[np.isfinite(out["x"].to_numpy(dtype=float)) & np.isfinite(out["y"].to_numpy(dtype=float))].reset_index(drop=True)
    return out


def _station_local_support_radius_m(half_width_m: np.ndarray) -> np.ndarray:
    hw = np.asarray(half_width_m, dtype=float)
    radius = np.full(hw.shape, 5.0, dtype=float)
    finite = np.isfinite(hw) & (hw > 0.0)
    if np.any(finite):
        radius[finite] = np.clip(np.maximum(5.0, 0.75 * hw[finite]), 5.0, 60.0)
    return radius.astype(np.float32)


def _nearest_support_metadata(centerline_points, support_points_path: Optional[str | Path]) -> pd.DataFrame:
    out = pd.DataFrame(index=centerline_points.index)
    default_role = np.full(len(centerline_points), role_to_code(ROLE_AMBIGUOUS), dtype=np.uint8)
    out["authoritative_support_point_distance_m"] = np.nan
    out["authoritative_support_point_role_code"] = default_role
    out["authoritative_support_point_role_confidence"] = np.nan
    out["authoritative_support_point_distance_to_bank_m"] = np.nan
    out["authoritative_support_point_normalized_channel_position"] = np.nan
    out["authoritative_support_point_half_width_est_m"] = np.nan
    out["authoritative_support_point_z_m"] = np.nan
    out["authoritative_support_point_inside_channel_mask"] = 0
    out["authoritative_support_point_inside_river_guidance_domain"] = 0
    out["authoritative_support_point_inside_mainstem_mask"] = 0
    out["authoritative_support_point_inside_estuary_clip"] = 0
    out["authoritative_bed_support_point_distance_m"] = np.nan
    out["authoritative_bank_margin_point_distance_m"] = np.nan
    out["authoritative_bed_core_point_distance_m"] = np.nan
    out["authoritative_bed_support_point_z_m"] = np.nan
    out["authoritative_bed_core_point_z_m"] = np.nan
    out["authoritative_support_local_radius_m"] = np.float32(5.0)

    support = _load_authoritative_support_points_table(support_points_path)
    if support.empty or centerline_points is None or getattr(centerline_points, "empty", True):
        return out

    query_xy = np.column_stack([
        centerline_points.geometry.x.to_numpy(dtype=float),
        centerline_points.geometry.y.to_numpy(dtype=float),
    ])

    def _query_subset(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not np.any(mask):
            return np.full(len(query_xy), np.nan, dtype=float), np.full(len(query_xy), -1, dtype=int)
        pts = support.loc[mask, ["x", "y"]].to_numpy(dtype=float)
        tree = cKDTree(pts)
        dist, idx = tree.query(query_xy, k=1)
        support_idx = support.index.to_numpy(dtype=int)[mask][np.asarray(idx, dtype=int)]
        return np.asarray(dist, dtype=float), np.asarray(support_idx, dtype=int)

    all_mask = np.ones(len(support), dtype=bool)
    all_dist, all_idx = _query_subset(all_mask)
    out["authoritative_support_point_distance_m"] = all_dist.astype(np.float32)
    if np.any(all_idx >= 0):
        valid = all_idx >= 0
        nearest = support.loc[all_idx[valid]].reset_index(drop=True)
        out.loc[valid, "authoritative_support_point_role_code"] = np.asarray([role_to_code(v) for v in nearest["authoritative_role"].astype(str)], dtype=np.uint8)
        out.loc[valid, "authoritative_support_point_role_confidence"] = pd.to_numeric(nearest["role_confidence"], errors="coerce").to_numpy(dtype=float)
        out.loc[valid, "authoritative_support_point_distance_to_bank_m"] = pd.to_numeric(nearest["distance_to_bank_m"], errors="coerce").to_numpy(dtype=float)
        out.loc[valid, "authoritative_support_point_normalized_channel_position"] = pd.to_numeric(nearest["normalized_channel_position"], errors="coerce").to_numpy(dtype=float)
        out.loc[valid, "authoritative_support_point_z_m"] = pd.to_numeric(nearest["support_z_m"], errors="coerce").to_numpy(dtype=float)
        out.loc[valid, "authoritative_support_point_inside_channel_mask"] = pd.to_numeric(nearest["inside_channel_mask"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out.loc[valid, "authoritative_support_point_inside_river_guidance_domain"] = pd.to_numeric(nearest["inside_river_guidance_domain"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out.loc[valid, "authoritative_support_point_inside_mainstem_mask"] = pd.to_numeric(nearest["inside_mainstem_mask"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out.loc[valid, "authoritative_support_point_inside_estuary_clip"] = pd.to_numeric(nearest["inside_estuary_clip"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        half_width = pd.to_numeric(nearest["component_half_width_est_m"], errors="coerce").to_numpy(dtype=float)
        out.loc[valid, "authoritative_support_point_half_width_est_m"] = half_width
        out.loc[valid, "authoritative_support_local_radius_m"] = _station_local_support_radius_m(half_width)

    bed_mask = support["authoritative_role"].astype(str).isin({ROLE_BED_INNER, ROLE_BED_CORE}).to_numpy(dtype=bool)
    bank_mask = support["authoritative_role"].astype(str).eq(ROLE_BANK_MARGIN).to_numpy(dtype=bool)
    core_mask = support["authoritative_role"].astype(str).eq(ROLE_BED_CORE).to_numpy(dtype=bool)
    bed_dist, bed_idx = _query_subset(bed_mask)
    core_dist, core_idx = _query_subset(core_mask)
    out["authoritative_bed_support_point_distance_m"] = bed_dist.astype(np.float32)
    out["authoritative_bank_margin_point_distance_m"] = _query_subset(bank_mask)[0].astype(np.float32)
    out["authoritative_bed_core_point_distance_m"] = core_dist.astype(np.float32)
    valid_bed = bed_idx >= 0
    if np.any(valid_bed):
        nearest_bed = support.loc[bed_idx[valid_bed]].reset_index(drop=True)
        out.loc[valid_bed, "authoritative_bed_support_point_z_m"] = pd.to_numeric(nearest_bed["support_z_m"], errors="coerce").to_numpy(dtype=float)
    valid_core = core_idx >= 0
    if np.any(valid_core):
        nearest_core = support.loc[core_idx[valid_core]].reset_index(drop=True)
        out.loc[valid_core, "authoritative_bed_core_point_z_m"] = pd.to_numeric(nearest_core["support_z_m"], errors="coerce").to_numpy(dtype=float)
    return out


def _infer_profile_id(df: pd.DataFrame) -> pd.Series:
    for col in ("component_id", "levelpathi", "levelpathid", "flow_idx"):
        if col in df.columns:
            return df[col].astype(str)
    return pd.Series(["main"] * len(df), index=df.index, dtype=object)


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



def _support_distance_array(sub: pd.DataFrame) -> np.ndarray:
    dist = _numeric_array_from_frame(sub, "profile_authoritative_bed_support_distance_m")
    if not np.any(np.isfinite(dist)):
        dist = _numeric_array_from_frame(sub, "authoritative_bed_support_point_distance_m")
    return dist


def _gaussian_smooth_1d(values: np.ndarray, sigma_points: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0 or not np.isfinite(sigma_points) or sigma_points <= 0.25:
        return vals.copy()
    finite = np.isfinite(vals)
    if np.count_nonzero(finite) < 2:
        return vals.copy()
    filled = vals.copy()
    missing = ~finite
    if np.any(missing):
        filled[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(finite), vals[finite])
    half = int(np.ceil(3.0 * sigma_points))
    if half < 1:
        return filled
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1, dtype=float) / float(sigma_points)) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(filled, half, mode="reflect")
    out = np.convolve(padded, kernel, mode="valid")
    return out[: vals.size]


def _distance_aware_profile_smoothing(
    values: np.ndarray,
    stations: np.ndarray,
    support_distance_m: np.ndarray,
    *,
    anchor_mask: np.ndarray | None = None,
    downstream_increasing: bool = True,
    near_distance_m: float = 120.0,
    far_distance_m: float = 900.0,
    moderate_window_m: float = 180.0,
    strong_window_m: float = 420.0,
) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    sta = np.asarray(stations, dtype=float)
    dist = np.asarray(support_distance_m, dtype=float)
    if vals.size < 5 or np.count_nonzero(np.isfinite(vals)) < 3:
        return vals.copy()
    anchors = np.asarray(anchor_mask, dtype=bool) if anchor_mask is not None else np.zeros(vals.size, dtype=bool)
    step = max(_robust_station_step(sta), 1.0)
    moderate_sigma = float(np.clip(moderate_window_m / max(step, 1.0), 1.0, 8.0))
    strong_sigma = float(np.clip(strong_window_m / max(step, 1.0), moderate_sigma + 0.5, 16.0))
    moderate = _gaussian_smooth_1d(vals, moderate_sigma)
    strong = _gaussian_smooth_1d(vals, strong_sigma)
    # Missing support distances in unsupported reaches should behave as far-from-support.
    frac = np.clip((dist - near_distance_m) / max(far_distance_m - near_distance_m, 1.0), 0.0, 1.0)
    frac = np.where(np.isfinite(frac), frac, 1.0)
    frac = np.where(anchors, 0.0, frac)
    out = vals.copy()
    finite = np.isfinite(vals)
    if np.any(finite):
        out[finite] = (1.0 - frac[finite]) * vals[finite] + frac[finite] * ((1.0 - frac[finite]) * moderate[finite] + frac[finite] * strong[finite])
    if np.any(anchors):
        out[anchors] = vals[anchors]
    ordered_vals = out if downstream_increasing else out[::-1]
    ordered_wts = np.where(anchors, 1.0e9, 1.0 + 4.0 * frac)
    ordered_wts = ordered_wts if downstream_increasing else ordered_wts[::-1]
    mono = _pava_isotonic(ordered_vals, increasing=False, weights=ordered_wts)
    mono = mono if downstream_increasing else mono[::-1]
    if np.any(anchors):
        mono[anchors] = vals[anchors]
    return mono.astype(float)

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
    filled = med.to_numpy(dtype=float)
    missing = ~np.isfinite(filled)
    if np.any(missing):
        filled[missing] = vals.to_numpy(dtype=float)[missing]
    return filled


def _pava_isotonic(y: np.ndarray, *, increasing: bool, weights: Optional[np.ndarray] = None) -> np.ndarray:
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


def _component_downstream_increasing(endpoint_meta: Dict[str, Dict[str, object]], profile_id: str) -> bool:
    meta = endpoint_meta.get(str(profile_id), {}) if endpoint_meta else {}
    return str(meta.get("station_direction", "increasing_station_downstream")) != "decreasing_station_downstream"


def _resolve_profile_fluvial_monotone_domain(profile: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    index = profile.index
    decay = pd.to_numeric(profile.get("bank_estuary_side_decay_median", pd.Series(np.nan, index=index)), errors="coerce").to_numpy(dtype=float)
    decay = np.nan_to_num(decay, nan=1.0)
    estuary_clip = pd.to_numeric(profile.get("profile_nearest_support_inside_estuary_clip", pd.Series(0.0, index=index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    score = decay.copy()
    explicit_estuary = estuary_clip > 0.5
    if np.any(explicit_estuary):
        score[explicit_estuary] = np.minimum(score[explicit_estuary], 0.0)
    source = np.full(len(profile), "bank_estuary_side_decay", dtype=object)
    source[explicit_estuary] = "explicit_estuary_clip"
    mask = score >= 0.75
    return (
        pd.Series(mask, index=index, dtype=bool),
        pd.Series(score.astype(np.float32), index=index, dtype="float32"),
        pd.Series(source, index=index, dtype=object),
    )


def _rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    window = int(max(1, window))
    if window == 1:
        return arr.copy()
    return pd.Series(arr, dtype=float).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def _numeric_array_from_frame(frame: pd.DataFrame, name: str, *, default: float = np.nan) -> np.ndarray:
    if name not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    vals = pd.to_numeric(frame[name], errors="coerce")
    if isinstance(vals, pd.Series):
        return vals.to_numpy(dtype=float)
    arr = np.asarray(vals, dtype=float)
    if arr.shape == ():
        return np.full(len(frame), float(arr), dtype=float)
    return arr.astype(float, copy=False)


def _smooth_longitudinal_offset_fit(
    *,
    stations: np.ndarray,
    offset_obs: np.ndarray,
    fallback_offset: np.ndarray,
    fluvial: np.ndarray,
    bed_support_dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Fit a smooth bank-to-bed offset along the channel.

    The offset represents the vertical distance from the bank reference
    elevation to the channel bed.  Where authoritative bed anchors exist
    the offset is observed directly; elsewhere it is interpolated and
    smoothed so the resulting channel-bed model decreases downstream
    without sharp jumps at support boundaries.
    """
    n = len(stations)
    offset_fit = np.full(n, np.nan, dtype=float)
    offset_source = np.full(n, "none", dtype=object)
    diagnostics = {
        "anchor_offset_rows": 0,
        "fallback_offset_rows": 0,
        "window_points": 0,
        "window_m": 0.0,
        "far_smoothing_fraction_mean": 0.0,
    }
    if n == 0:
        return offset_fit, offset_source, diagnostics
    stations = np.asarray(stations, dtype=float)
    offset_obs = np.asarray(offset_obs, dtype=float)
    fallback_offset = np.asarray(fallback_offset, dtype=float)
    fluvial = np.asarray(fluvial, dtype=bool)
    bed_support_dist = np.asarray(bed_support_dist, dtype=float)
    if not np.any(fluvial):
        return offset_fit, offset_source, diagnostics
    finite_anchor = np.isfinite(offset_obs) & fluvial
    finite_fallback = np.isfinite(fallback_offset) & fluvial
    diagnostics["anchor_offset_rows"] = int(np.count_nonzero(finite_anchor))
    diagnostics["fallback_offset_rows"] = int(np.count_nonzero(finite_fallback))
    step_m = _robust_station_step(stations)
    support_med = float(np.nanmedian(bed_support_dist[np.isfinite(bed_support_dist) & fluvial])) if np.any(np.isfinite(bed_support_dist) & fluvial) else 0.0
    window_m = float(np.clip(max(300.0, 350.0 + (2.0 * support_med)), 300.0, 1400.0))
    window = int(np.clip(round(window_m / max(step_m, 1.0)), 5, 81))
    if window % 2 == 0:
        window += 1
    diagnostics["window_points"] = int(window)
    diagnostics["window_m"] = float(window * step_m)

    # Build a merged offset seed that combines authoritative anchor
    # observations (high trust) with backbone-derived fallback offsets
    # (lower trust) so interpolation has support across the full reach
    # rather than extrapolating long stretches from sparse endpoints.
    merged_obs = offset_obs.copy()
    if np.any(finite_fallback & (~finite_anchor)):
        fill_mask = finite_fallback & (~finite_anchor) & fluvial
        merged_obs[fill_mask] = fallback_offset[fill_mask]

    finite_merged = np.isfinite(merged_obs) & fluvial
    seed = float(np.nanmedian(offset_obs[finite_anchor])) if np.any(finite_anchor) else np.nan
    if not np.isfinite(seed):
        seed = float(np.nanmedian(fallback_offset[finite_fallback])) if np.any(finite_fallback) else 1.50
    if not np.isfinite(seed):
        seed = 1.50

    if np.count_nonzero(finite_anchor) >= 2:
        # Interpolate from anchor observations for the primary structure.
        interp = np.interp(
            stations,
            stations[finite_anchor],
            offset_obs[finite_anchor],
            left=float(offset_obs[finite_anchor][0]),
            right=float(offset_obs[finite_anchor][-1]),
        )
        interp = np.clip(interp, 0.10, None)
        smooth = _rolling_nanmedian(interp, window)
        smooth = _rolling_nanmean(smooth, window)
        smooth = _rolling_nanmean(smooth, max(3, window // 2))
        smooth = _interp_fill_nan(smooth) if np.any(np.isfinite(smooth)) else smooth
        # Far from support: relax toward the seed gradually but preserve
        # the smoothed longitudinal trend instead of flattening to a
        # constant.  This keeps the generalized shape continuous.
        far_fraction = np.where(
            np.isfinite(bed_support_dist),
            np.clip((bed_support_dist - 200.0) / 1200.0, 0.0, 0.75),
            0.75,
        )
        diagnostics["far_smoothing_fraction_mean"] = float(np.nanmean(far_fraction[fluvial])) if np.any(fluvial) else 0.0
        smooth = np.where(fluvial, ((1.0 - far_fraction) * smooth) + (far_fraction * seed), np.nan)
        offset_fit = np.where(fluvial, np.clip(smooth, 0.10, None), offset_fit)
        offset_source[fluvial & np.isfinite(offset_fit)] = "authoritative_bed_anchor_curve_offset_fit"
        return offset_fit, offset_source, diagnostics

    if np.count_nonzero(finite_anchor) == 1:
        only = float(offset_obs[finite_anchor][0])
        # With a single anchor, use fallback offsets to add longitudinal
        # variation rather than applying a flat constant.
        if np.count_nonzero(finite_fallback) >= 3:
            fb_smooth = _rolling_nanmedian(
                _interp_fill_nan(np.where(finite_fallback, fallback_offset, np.nan)),
                max(3, window // 2),
            )
            fb_smooth = _interp_fill_nan(fb_smooth) if np.any(np.isfinite(fb_smooth)) else fb_smooth
            fb_med = float(np.nanmedian(fb_smooth[np.isfinite(fb_smooth)])) if np.any(np.isfinite(fb_smooth)) else only
            if np.isfinite(fb_med) and abs(fb_med) > 1e-6:
                # Rescale fallback trend so its median matches the single
                # authoritative observation.
                ratio = only / max(abs(fb_med), 0.10)
                fb_smooth = fb_smooth * ratio
                fb_smooth = np.clip(fb_smooth, 0.10, None)
                offset_fit = np.where(fluvial, fb_smooth, offset_fit)
            else:
                offset_fit = np.where(fluvial, max(0.10, only), offset_fit)
        else:
            offset_fit = np.where(fluvial, max(0.10, only), offset_fit)
        offset_source[fluvial & np.isfinite(offset_fit)] = "single_authoritative_bed_anchor_offset"
        return offset_fit, offset_source, diagnostics

    # No authoritative anchors at all: use fallback offsets with smoothing
    # to preserve whatever longitudinal structure exists.
    if np.count_nonzero(finite_fallback) >= 3:
        fb_smooth = _rolling_nanmedian(
            _interp_fill_nan(np.where(finite_fallback, fallback_offset, np.nan)),
            max(3, window // 2),
        )
        fb_smooth = _interp_fill_nan(fb_smooth) if np.any(np.isfinite(fb_smooth)) else fb_smooth
        fb_smooth = np.clip(fb_smooth, 0.10, None)
        offset_fit = np.where(fluvial, fb_smooth, offset_fit)
    else:
        offset_fit = np.where(fluvial, max(0.10, seed), offset_fit)
    offset_source[fluvial & np.isfinite(offset_fit)] = "bank_backbone_offset_fallback"
    return offset_fit, offset_source, diagnostics


def _apply_active_core_support(profile: pd.DataFrame, endpoint_meta: Dict[str, Dict[str, object]]) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    if profile.empty:
        return profile, {}
    out = profile.copy()
    out["active_core_support_elevation_m"] = pd.to_numeric(out.get("network_backbone_elevation_m", out.get("bed_elevation_m")), errors="coerce")
    out["active_core_support_uncertainty_m"] = pd.to_numeric(out.get("network_backbone_uncertainty_m", out.get("uncertainty_m")), errors="coerce")
    out["active_core_support_source"] = out.get("network_backbone_source", pd.Series(["network_backbone"] * len(out), index=out.index)).astype(object)
    out["active_core_support_monotone_adjustment_m"] = 0.0
    out["active_core_support_bank_reference_m"] = np.nan
    out["active_core_support_bank_offset_m"] = np.nan
    out["active_core_support_model_elevation_m"] = np.nan
    out["active_core_support_model_weight"] = 0.0
    out["active_core_support_offset_source"] = "none"
    out["generalized_longitudinal_bed_elevation_m"] = pd.to_numeric(out.get("network_backbone_elevation_m", out.get("bed_elevation_m")), errors="coerce")
    out["generalized_longitudinal_bed_z_m"] = pd.to_numeric(out["generalized_longitudinal_bed_elevation_m"], errors="coerce")
    out["generalized_longitudinal_bed_uncertainty_m"] = pd.to_numeric(out.get("network_backbone_uncertainty_m", out.get("uncertainty_m")), errors="coerce")
    out["generalized_longitudinal_bed_source"] = out.get("network_backbone_source", pd.Series(["network_backbone"] * len(out), index=out.index)).astype(object)
    out["generalized_longitudinal_bed_base_elevation_m"] = pd.to_numeric(out["generalized_longitudinal_bed_elevation_m"], errors="coerce")
    out["generalized_longitudinal_bed_reconciled_elevation_m"] = pd.to_numeric(out["generalized_longitudinal_bed_elevation_m"], errors="coerce")
    out["generalized_longitudinal_bed_local_auth_taper_m"] = 0.0
    out["generalized_longitudinal_bed_local_auth_reconciliation_weight"] = 0.0
    out["authoritative_reconciliation_delta_m"] = 0.0
    out["authoritative_reconciliation_weight"] = 0.0
    out["authoritative_reconciliation_confidence"] = 0.0
    out["authoritative_reconciliation_support_distance_m"] = np.nan
    out["authoritative_reconciliation_exact_anchor"] = False
    out["authoritative_reconciliation_observation_used"] = False
    out["authoritative_reconciliation_observed_bed_z_m"] = np.nan
    out["authoritative_reconciliation_observed_residual_m"] = np.nan
    out["authoritative_reconciliation_observation_confidence"] = np.nan
    fluvial_mask, fluvial_score, fluvial_source = _resolve_profile_fluvial_monotone_domain(out)
    out["profile_inside_fluvial_monotone_domain"] = fluvial_mask
    out["profile_fluvial_monotone_score"] = fluvial_score
    out["profile_fluvial_monotone_domain_source"] = fluvial_source
    diagnostics: Dict[str, Dict[str, float]] = {}

    def _project_monotone(values: np.ndarray, weights: np.ndarray, anchors: np.ndarray, anchor_values: np.ndarray, *, downstream_increasing: bool) -> np.ndarray:
        solved = values.copy()
        if np.count_nonzero(np.isfinite(solved)) < 2:
            return solved
        missing = ~np.isfinite(solved)
        if np.any(missing):
            finite = np.isfinite(solved)
            if np.count_nonzero(finite) >= 2:
                solved[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(finite), solved[finite])
            elif np.count_nonzero(finite) == 1:
                solved[missing] = solved[finite][0]
        solved = np.where(anchors, anchor_values, solved)
        ordered_vals = solved if downstream_increasing else solved[::-1]
        ordered_wts = weights if downstream_increasing else weights[::-1]
        mono = _pava_isotonic(ordered_vals, increasing=False, weights=ordered_wts)
        mono = mono if downstream_increasing else mono[::-1]
        if np.any(anchors):
            mono = np.where(anchors, anchor_values, mono)
        return mono

    for comp, idx in out.groupby("profile_id", sort=False).groups.items():
        sub = out.loc[idx].sort_values("station_m").copy()
        base = _numeric_array_from_frame(sub, "network_backbone_elevation_m")
        if not np.any(np.isfinite(base)):
            base = _numeric_array_from_frame(sub, "bed_elevation_m")
        unc = _numeric_array_from_frame(sub, "network_backbone_uncertainty_m")
        fallback_unc = _numeric_array_from_frame(sub, "uncertainty_m")
        unc = np.where(np.isfinite(unc), unc, fallback_unc)
        unc = np.where(np.isfinite(unc) & (unc > 0.0), unc, 0.25)
        anchor_vals = _numeric_array_from_frame(sub, "authoritative_anchor_elevation_m")
        anchor_present = sub.get("authoritative_anchor_present", pd.Series(False, index=sub.index)).fillna(False).to_numpy(dtype=bool) & np.isfinite(anchor_vals)
        fluvial = sub.get("profile_inside_fluvial_monotone_domain", pd.Series(True, index=sub.index)).fillna(True).to_numpy(dtype=bool)
        bed_support_dist = _numeric_array_from_frame(sub, "profile_authoritative_bed_support_distance_m")
        downstream_increasing = _component_downstream_increasing(endpoint_meta, str(comp))
        bank_pair_fit = _numeric_array_from_frame(sub, "bank_pair_fit_z_m")
        bank_profile = _numeric_array_from_frame(sub, "bank_profile_monotone_m")
        raw_bank = _numeric_array_from_frame(sub, "bank_elevation_m")
        bank_ref = np.where(np.isfinite(bank_pair_fit), bank_pair_fit, np.where(np.isfinite(bank_profile), bank_profile, raw_bank))
        anchor_curve_present = sub.get("authoritative_anchor_curve_present", pd.Series(False, index=sub.index)).fillna(False).to_numpy(dtype=bool)

        solved = base.copy()
        solved_unc = unc.copy()
        local_taper = np.zeros(len(sub), dtype=float)
        local_taper_weight = np.zeros(len(sub), dtype=float)
        local_taper_confidence = np.zeros(len(sub), dtype=float)

        if np.count_nonzero(np.isfinite(solved) & fluvial) >= 2:
            run_start = None
            for i, active in enumerate(list(fluvial) + [False]):
                if active and run_start is None:
                    run_start = i
                elif (not active) and (run_start is not None):
                    seg = slice(run_start, i)
                    seg_vals = solved[seg].copy()
                    seg_unc = solved_unc[seg].copy()
                    seg_anchor = anchor_present[seg].copy()
                    seg_anchor_vals = anchor_vals[seg].copy()
                    seg_support_dist = bed_support_dist[seg].copy()
                    seg_weights = np.where(np.isfinite(seg_unc) & (seg_unc > 0.0), 1.0 / np.square(np.maximum(seg_unc, 0.05)), 1.0)
                    near_support = np.isfinite(seg_support_dist)
                    if np.any(near_support):
                        support_boost = np.clip(1.0 + 0.35 * (1.0 - np.clip(seg_support_dist / 600.0, 0.0, 1.0)), 1.0, 1.35)
                        seg_weights = np.where(near_support, seg_weights * support_boost, seg_weights)
                    seg_weights = np.where(seg_anchor, 1.0e9, seg_weights)
                    mono = _project_monotone(seg_vals, seg_weights, seg_anchor, seg_anchor_vals, downstream_increasing=downstream_increasing)
                    solved[seg] = mono
                    delta = np.abs(mono - base[seg])
                    solved_unc[seg] = np.sqrt(np.square(np.nan_to_num(seg_unc, nan=0.25)) + np.square(np.nan_to_num(delta, nan=0.0)))
                    solved_unc[seg] = np.where(seg_anchor, 0.02, np.maximum(solved_unc[seg], 0.05))
                    run_start = None

        auth_bed_z = _numeric_array_from_frame(sub, "authoritative_bed_support_point_z_m")
        auth_bed_z_core = _numeric_array_from_frame(sub, "authoritative_bed_core_point_z_m")
        local_auth_z = np.where(np.isfinite(auth_bed_z_core), auth_bed_z_core, auth_bed_z)
        local_auth_dist = _numeric_array_from_frame(sub, "profile_authoritative_bed_support_distance_m")
        if not np.any(np.isfinite(local_auth_dist)):
            local_auth_dist = _numeric_array_from_frame(sub, "authoritative_bed_support_point_distance_m")
        taper_radius_m = 250.0
        comp_class = str(sub.get("component_support_class", pd.Series(["anchored_mainstem"] * len(sub), index=sub.index)).astype(str).mode(dropna=True).iloc[0]) if len(sub) else "anchored_mainstem"
        generalized_base = solved.copy()
        can_taper = (
            np.isfinite(generalized_base) & np.isfinite(local_auth_z) & np.isfinite(local_auth_dist)
            & (local_auth_dist > 0.0) & (local_auth_dist < taper_radius_m)
            & fluvial & (~anchor_present)
        )
        reconciliation_observations = build_authoritative_reconciliation_observations(
            stations=pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float),
            generalized_bed=generalized_base,
            authoritative_bed=local_auth_z,
            support_distance_m=local_auth_dist,
            fluvial_mask=fluvial,
            exact_anchor_mask=anchor_present,
            taper_radius_m=taper_radius_m,
        )
        if not reconciliation_observations.empty:
            reconciliation_field = solve_authoritative_reconciliation_field(
                stations=pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float),
                observations=reconciliation_observations,
                support_distance_m=local_auth_dist,
                fluvial_mask=fluvial,
                exact_anchor_mask=anchor_present,
                taper_radius_m=taper_radius_m,
                max_abs_delta_m=1.5,
                component_class=comp_class,
            )
            proposed_delta = pd.to_numeric(reconciliation_field["authoritative_reconciliation_delta_m"], errors="coerce").to_numpy(dtype=float)
            local_taper_weight = pd.to_numeric(reconciliation_field["authoritative_reconciliation_weight"], errors="coerce").to_numpy(dtype=float)
            local_taper_confidence = pd.to_numeric(reconciliation_field["authoritative_reconciliation_confidence"], errors="coerce").to_numpy(dtype=float)
            solved = generalized_base + np.nan_to_num(proposed_delta, nan=0.0)
            taper_rows = np.isfinite(local_taper_weight) & (local_taper_weight > 0.0)
            solved_unc[taper_rows] = np.maximum(solved_unc[taper_rows] * (1.0 - 0.3 * np.clip(local_taper_weight[taper_rows], 0.0, 1.0)), 0.05)
            if np.count_nonzero(np.isfinite(solved) & fluvial) >= 2:
                run_start = None
                for i, active in enumerate(list(fluvial) + [False]):
                    if active and run_start is None:
                        run_start = i
                    elif (not active) and (run_start is not None):
                        seg = slice(run_start, i)
                        seg_vals = solved[seg].copy()
                        seg_unc = solved_unc[seg].copy()
                        seg_anchor = anchor_present[seg].copy()
                        seg_anchor_vals = anchor_vals[seg].copy()
                        seg_weights = np.where(np.isfinite(seg_unc) & (seg_unc > 0.0), 1.0 / np.square(np.maximum(seg_unc, 0.05)), 1.0)
                        seg_weights = np.where(seg_anchor, 1.0e9, seg_weights)
                        solved[seg] = _project_monotone(seg_vals, seg_weights, seg_anchor, seg_anchor_vals, downstream_increasing=downstream_increasing)
                        run_start = None
        else:
            reconciliation_field = pd.DataFrame(
                {
                    "authoritative_reconciliation_delta_m": np.zeros(len(sub), dtype=float),
                    "authoritative_reconciliation_weight": np.zeros(len(sub), dtype=float),
                    "authoritative_reconciliation_confidence": np.zeros(len(sub), dtype=float),
                    "authoritative_reconciliation_support_distance_m": local_auth_dist,
                    "authoritative_reconciliation_exact_anchor": anchor_present,
                }
            )

        if np.count_nonzero(np.isfinite(solved) & fluvial) >= 2:
            run_start = None
            for i, active in enumerate(list(fluvial) + [False]):
                if active and run_start is None:
                    run_start = i
                elif (not active) and (run_start is not None):
                    seg = slice(run_start, i)
                    seg_vals = solved[seg].copy()
                    seg_unc = solved_unc[seg].copy()
                    seg_anchor = anchor_present[seg].copy()
                    seg_anchor_vals = anchor_vals[seg].copy()
                    seg_support_dist = local_auth_dist[seg].copy()
                    seg_weights = np.where(np.isfinite(seg_unc) & (seg_unc > 0.0), 1.0 / np.square(np.maximum(seg_unc, 0.05)), 1.0)
                    seg_weights = np.where(seg_anchor, 1.0e9, seg_weights)
                    seg_vals = _project_monotone(seg_vals, seg_weights, seg_anchor, seg_anchor_vals, downstream_increasing=downstream_increasing)
                    seg_vals = _distance_aware_profile_smoothing(
                        seg_vals,
                        pd.to_numeric(sub.iloc[seg]["station_m"], errors="coerce").to_numpy(dtype=float),
                        seg_support_dist,
                        anchor_mask=seg_anchor,
                        downstream_increasing=downstream_increasing,
                        near_distance_m=120.0,
                        far_distance_m=1000.0,
                        moderate_window_m=180.0,
                        strong_window_m=520.0,
                    )
                    seg_vals = _project_monotone(seg_vals, seg_weights, seg_anchor, seg_anchor_vals, downstream_increasing=downstream_increasing)
                    solved[seg] = seg_vals
                    run_start = None
            local_taper = solved - generalized_base

        source = np.full(len(sub), "generalized_longitudinal_bed", dtype=object)
        unchanged = np.isfinite(base) & np.isfinite(solved) & (np.abs(solved - base) <= 1.0e-6)
        if "network_backbone_source" in sub.columns:
            source[unchanged] = sub["network_backbone_source"].astype(str).to_numpy(dtype=object)[unchanged]
        source[np.abs(local_taper) > 1.0e-6] = "generalized_longitudinal_bed_local_authoritative_reconciliation"
        anchor_source = sub.get("authoritative_anchor_source", pd.Series(["authoritative_anchor"] * len(sub), index=sub.index)).astype(str).to_numpy(dtype=object)
        source[anchor_present] = anchor_source[anchor_present]

        out.loc[sub.index, "active_core_support_elevation_m"] = solved
        out.loc[sub.index, "active_core_support_uncertainty_m"] = solved_unc
        out.loc[sub.index, "active_core_support_monotone_adjustment_m"] = solved - base
        out.loc[sub.index, "active_core_support_bank_reference_m"] = bank_ref
        out.loc[sub.index, "active_core_support_bank_offset_m"] = np.where(np.isfinite(bank_ref) & np.isfinite(solved), bank_ref - solved, np.nan)
        out.loc[sub.index, "active_core_support_model_elevation_m"] = solved
        out.loc[sub.index, "active_core_support_model_weight"] = np.zeros(len(sub), dtype=float)
        out.loc[sub.index, "active_core_support_offset_source"] = np.where(np.abs(local_taper) > 1.0e-6, "authoritative_reconciliation_field", "none")
        out.loc[sub.index, "active_core_support_source"] = source
        out.loc[sub.index, "generalized_longitudinal_bed_base_elevation_m"] = generalized_base
        out.loc[sub.index, "generalized_longitudinal_bed_reconciled_elevation_m"] = solved
        out.loc[sub.index, "generalized_longitudinal_bed_elevation_m"] = solved
        out.loc[sub.index, "generalized_longitudinal_bed_z_m"] = solved
        out.loc[sub.index, "generalized_longitudinal_bed_uncertainty_m"] = solved_unc
        out.loc[sub.index, "generalized_longitudinal_bed_source"] = source
        out.loc[sub.index, "generalized_longitudinal_bed_local_auth_taper_m"] = local_taper
        out.loc[sub.index, "generalized_longitudinal_bed_local_auth_reconciliation_weight"] = local_taper_weight
        out.loc[sub.index, "authoritative_reconciliation_delta_m"] = local_taper
        out.loc[sub.index, "authoritative_reconciliation_weight"] = local_taper_weight
        out.loc[sub.index, "authoritative_reconciliation_confidence"] = local_taper_confidence
        out.loc[sub.index, "authoritative_reconciliation_support_distance_m"] = pd.to_numeric(reconciliation_field.get("authoritative_reconciliation_support_distance_m", local_auth_dist), errors="coerce").to_numpy(dtype=float)
        out.loc[sub.index, "authoritative_reconciliation_exact_anchor"] = np.asarray(reconciliation_field.get("authoritative_reconciliation_exact_anchor", anchor_present), dtype=bool)
        obs_station_lookup = {float(r["station_m"]): (float(r["authoritative_bed_elevation_m"]), float(r["residual_delta_m"]), float(r["observation_confidence"])) for _, r in reconciliation_observations.iterrows()} if not reconciliation_observations.empty else {}
        obs_used = []
        obs_bed = []
        obs_resid = []
        obs_conf = []
        for st in pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float):
            if np.isfinite(st) and float(st) in obs_station_lookup:
                bed_val, resid_val, conf_val = obs_station_lookup[float(st)]
                obs_used.append(True)
                obs_bed.append(bed_val)
                obs_resid.append(resid_val)
                obs_conf.append(conf_val)
            else:
                obs_used.append(False)
                obs_bed.append(np.nan)
                obs_resid.append(np.nan)
                obs_conf.append(np.nan)
        out.loc[sub.index, "authoritative_reconciliation_observation_used"] = np.asarray(obs_used, dtype=bool)
        out.loc[sub.index, "authoritative_reconciliation_observed_bed_z_m"] = np.asarray(obs_bed, dtype=float)
        out.loc[sub.index, "authoritative_reconciliation_observed_residual_m"] = np.asarray(obs_resid, dtype=float)
        out.loc[sub.index, "authoritative_reconciliation_observation_confidence"] = np.asarray(obs_conf, dtype=float)

        base_ordered = base if downstream_increasing else base[::-1]
        solved_ordered = solved if downstream_increasing else solved[::-1]
        base_ordered = base_ordered[np.isfinite(base_ordered)]
        solved_ordered = solved_ordered[np.isfinite(solved_ordered)]
        diagnostics[str(comp)] = {
            "n_rows": int(len(sub)),
            "fluvial_rows": int(np.count_nonzero(fluvial)),
            "authoritative_anchor_rows": int(np.count_nonzero(anchor_present)),
            "curve_anchor_rows": int(np.count_nonzero(anchor_curve_present)),
            "bank_pair_fit_rows": int(np.count_nonzero(np.isfinite(bank_pair_fit))),
            "bank_reference_rows": int(np.count_nonzero(np.isfinite(bank_ref))),
            "generalized_bed_rows": int(np.count_nonzero(np.isfinite(solved))),
            "pre_monotone_violation_count": int(np.count_nonzero(np.diff(base_ordered) > 0.0)) if base_ordered.size >= 2 else 0,
            "post_monotone_violation_count": int(np.count_nonzero(np.diff(solved_ordered) > 1.0e-6)) if solved_ordered.size >= 2 else 0,
            "max_adjustment_m": float(np.nanmax(np.abs(solved - base))) if np.count_nonzero(np.isfinite(solved) & np.isfinite(base)) >= 1 else 0.0,
            "median_adjustment_m": float(np.nanmedian(np.abs(solved - base))) if np.count_nonzero(np.isfinite(solved) & np.isfinite(base)) >= 1 else 0.0,
            "median_bank_offset_m": float(np.nanmedian((bank_ref - solved)[np.isfinite(bank_ref) & np.isfinite(solved)])) if np.count_nonzero(np.isfinite(bank_ref) & np.isfinite(solved)) >= 1 else 0.0,
            "max_bank_offset_m": float(np.nanmax(np.abs(bank_ref - solved)[np.isfinite(bank_ref) & np.isfinite(solved)])) if np.count_nonzero(np.isfinite(bank_ref) & np.isfinite(solved)) >= 1 else 0.0,
            "offset_source": "authoritative_reconciliation_field" if np.any(np.abs(local_taper) > 1.0e-6) else "none",
            "authoritative_reconciliation_observation_count": int(len(reconciliation_observations)),
            "local_auth_taper_rows": int(np.count_nonzero(can_taper)),
            "local_auth_taper_max_abs_m": float(np.nanmax(np.abs(local_taper))) if np.count_nonzero(np.abs(local_taper) > 0.0) else 0.0,
            "local_auth_reconciliation_weight_max": float(np.nanmax(local_taper_weight)) if np.count_nonzero(local_taper_weight > 0.0) else 0.0,
        }
    return out, diagnostics

def _apply_bank_longitudinal_reference(profile: pd.DataFrame, endpoint_meta: Dict[str, Dict[str, object]]) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    if profile.empty:
        return profile, {}
    out = profile.copy()
    out["bank_profile_raw_m"] = np.nan
    out["bank_profile_smoothed_m"] = np.nan
    out["bank_profile_monotone_m"] = np.nan
    out["bank_to_bed_offset_m"] = np.nan
    out["bank_bed_reference_m"] = np.nan
    out["bank_profile_adjustment_m"] = 0.0
    out["bank_profile_monotone_adjustment_m"] = 0.0
    out["bank_profile_supported"] = False
    out["bank_profile_active_source"] = "missing"
    diagnostics: Dict[str, Dict[str, float]] = {}

    for comp, idx in out.groupby("profile_id", sort=False).groups.items():
        sub = out.loc[idx].sort_values("station_m").copy()
        stations = pd.to_numeric(sub.get("station_m"), errors="coerce").to_numpy(dtype=float)
        raw_bank = pd.to_numeric(sub.get("bank_elevation_m"), errors="coerce").to_numpy(dtype=float)
        bank_fit = pd.to_numeric(sub.get("bank_pair_fit_z_m"), errors="coerce").to_numpy(dtype=float)
        bank = np.where(np.isfinite(bank_fit), bank_fit, raw_bank)
        auth = pd.to_numeric(sub.get("authoritative_anchor_elevation_m"), errors="coerce").to_numpy(dtype=float)
        auth_present = sub.get("authoritative_anchor_present", pd.Series(False, index=sub.index)).fillna(False).to_numpy(dtype=bool)
        xs = pd.to_numeric(sub.get("xs_support_elevation_m"), errors="coerce").to_numpy(dtype=float)
        center = pd.to_numeric(sub.get("centerline_elevation_m"), errors="coerce").to_numpy(dtype=float)
        bank_w = np.clip(pd.to_numeric(sub.get("bank_weight"), errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
        xs_w = np.clip(pd.to_numeric(sub.get("xs_support_weight"), errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
        center_w = np.clip(pd.to_numeric(sub.get("centerline_weight"), errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)

        observed_bed = np.where(auth_present & np.isfinite(auth), auth, np.where(np.isfinite(xs), xs, np.where(np.isfinite(center), center, np.nan)))
        offset = np.where(np.isfinite(bank) & np.isfinite(observed_bed), bank - observed_bed, np.nan)
        step = _robust_station_step(stations)
        window = int(np.clip(round(max(5.0, (80.0 / max(step, 1.0)))), 3, 11))

        bank_sm = _rolling_nanmedian(bank, window)
        bank_sm = _interp_fill_nan(bank_sm) if np.any(np.isfinite(bank_sm)) else bank_sm
        monotone_target = bank_sm.copy()
        downstream_increasing = _component_downstream_increasing(endpoint_meta, str(comp))
        finite_bank = np.isfinite(monotone_target)
        if np.count_nonzero(finite_bank) >= 2:
            ordered = _interp_fill_nan(monotone_target if downstream_increasing else monotone_target[::-1])
            ordered = _pava_isotonic(ordered, increasing=False)
            bank_mono = ordered if downstream_increasing else ordered[::-1]
        else:
            bank_mono = monotone_target.copy()

        offset_sm = _rolling_nanmedian(offset, window)
        if np.any(np.isfinite(offset_sm)):
            offset_sm = _interp_fill_nan(offset_sm)
        elif np.any(np.isfinite(bank) & np.isfinite(center)):
            offset_sm = np.where(np.isfinite(bank) & np.isfinite(center), bank - center, np.nan)
            offset_sm = _rolling_nanmedian(offset_sm, window)
            offset_sm = _interp_fill_nan(offset_sm) if np.any(np.isfinite(offset_sm)) else offset_sm
        bank_ref = np.where(np.isfinite(bank_mono) & np.isfinite(offset_sm), bank_mono - offset_sm, np.nan)

        supported = np.isfinite(bank_ref)

        out.loc[sub.index, "bank_profile_raw_m"] = bank
        out.loc[sub.index, "bank_profile_smoothed_m"] = bank_sm
        out.loc[sub.index, "bank_profile_monotone_m"] = bank_mono
        out.loc[sub.index, "bank_to_bed_offset_m"] = offset_sm
        out.loc[sub.index, "bank_bed_reference_m"] = bank_ref
        out.loc[sub.index, "bank_profile_adjustment_m"] = np.zeros(len(sub), dtype=float)
        out.loc[sub.index, "bank_profile_monotone_adjustment_m"] = bank_mono - bank_sm
        out.loc[sub.index, "bank_profile_supported"] = supported
        out.loc[sub.index, "bank_profile_active_source"] = np.where(np.isfinite(bank_fit), "bank_pair_fit", np.where(np.isfinite(raw_bank), "bank_elevation_xs", "missing"))

        diagnostics[str(comp)] = {
            "n_rows": int(len(sub)),
            "bank_rows": int(np.count_nonzero(np.isfinite(raw_bank))),
            "bank_pair_fit_rows": int(np.count_nonzero(np.isfinite(bank_fit))),
            "offset_rows": int(np.count_nonzero(np.isfinite(offset))),
            "supported_rows": int(np.count_nonzero(supported)),
            "bank_raw_range_m": float(np.nanmax(raw_bank) - np.nanmin(raw_bank)) if np.count_nonzero(np.isfinite(raw_bank)) >= 2 else 0.0,
            "bank_active_range_m": float(np.nanmax(bank) - np.nanmin(bank)) if np.count_nonzero(np.isfinite(bank)) >= 2 else 0.0,
            "bank_monotone_adjustment_max_m": float(np.nanmax(np.abs(bank_mono - bank_sm))) if np.count_nonzero(np.isfinite(bank_mono) & np.isfinite(bank_sm)) >= 1 else 0.0,
            "bed_adjustment_max_m": 0.0,
        }
    return out, diagnostics


def _component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy.ndimage import label

    labels, count = label(np.asarray(mask, dtype=bool))
    return labels.astype(np.int32), int(count)


def _seed_mask_from_points(*, shape, transform, domain_mask: np.ndarray, points_gdf) -> np.ndarray:
    import rasterio.transform

    mask = np.zeros(shape, dtype=bool)
    if points_gdf is None or getattr(points_gdf, "empty", True) or not np.any(domain_mask):
        return mask
    xs = np.asarray(points_gdf.geometry.x.to_numpy(dtype=float), dtype=float)
    ys = np.asarray(points_gdf.geometry.y.to_numpy(dtype=float), dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(valid):
        return mask
    rows, cols = rasterio.transform.rowcol(transform, xs[valid], ys[valid])
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    dom = np.asarray(domain_mask, dtype=bool)
    nrows, ncols = shape
    inside = (rows >= 0) & (cols >= 0) & (rows < int(nrows)) & (cols < int(ncols))
    if not np.any(inside):
        return mask
    rows = rows[inside]
    cols = cols[inside]
    good = dom[rows, cols]
    mask[rows[good], cols[good]] = True
    return mask


def _row_support_reference(profile: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    support_cols = [
        "authoritative_anchor_elevation_m",
        "active_core_support_elevation_m",
        "network_backbone_elevation_m",
        "centerline_elevation_m",
        "xs_support_elevation_m",
    ]
    cols = [c for c in support_cols if c in profile.columns]
    n = len(profile)
    if not cols:
        return np.full(n, np.nan, dtype=float), np.full(n, np.nan, dtype=float), {"component_class": "all_nan"}
    stack = np.column_stack([pd.to_numeric(profile[c], errors="coerce").to_numpy(dtype=float) for c in cols]).astype(float)
    valid = np.isfinite(stack)
    counts = np.sum(valid, axis=1)
    ref = np.full(n, np.nan, dtype=float)
    spread = np.full(n, np.nan, dtype=float)
    supported = counts > 0
    if np.any(supported):
        ref[supported] = np.nanmedian(stack[supported], axis=1)
    multi = counts > 1
    if np.any(multi):
        spread[multi] = np.nanstd(stack[multi], axis=1, ddof=0)
    return ref, spread


def _compute_station_distance_to_mask(
    stations: np.ndarray,
    support_mask: np.ndarray,
    component_ids: np.ndarray,
) -> np.ndarray:
    """Per-station distance (metres) to nearest authoritative anchor within same component.

    Returns np.inf for stations in components with zero authoritative anchors.
    This is the physically meaningful measure of support — geometric XS/centerline
    availability does NOT constitute measured support.
    """
    stations = np.asarray(stations, dtype=float)
    support_mask = np.asarray(support_mask, dtype=bool)
    component_ids = np.asarray(component_ids)
    dist = np.full(stations.shape, np.inf, dtype=float)
    if stations.size == 0:
        return dist
    finite_station = np.isfinite(stations)
    for comp in np.unique(component_ids):
        mask = (component_ids == comp) & finite_station
        if not np.any(mask):
            continue
        comp_idx = np.flatnonzero(mask)
        comp_stations = stations[comp_idx]
        auth_stations = np.sort(comp_stations[support_mask[comp_idx]])
        if auth_stations.size == 0:
            continue
        insert_idx = np.searchsorted(auth_stations, comp_stations)
        left_idx = np.clip(insert_idx - 1, 0, auth_stations.size - 1)
        right_idx = np.clip(insert_idx, 0, auth_stations.size - 1)
        left_dist = np.abs(comp_stations - auth_stations[left_idx])
        right_dist = np.abs(comp_stations - auth_stations[right_idx])
        dist[comp_idx] = np.minimum(left_dist, right_dist)
    return dist


# Distance threshold (metres) beyond which geometric XS/centerline support
# is reclassified as unsupported.  5× a typical 100 m station spacing — any
# station >500 m from the nearest authoritative anchor has no real measured
# control and should be treated as unsupported.
_MEASURED_SUPPORT_DISTANCE_THRESHOLD_M = 500.0


def _coerce_role_codes(series: pd.Series | np.ndarray | None, length: int) -> np.ndarray:
    if series is None:
        return np.zeros(length, dtype=np.uint8)
    vals = pd.to_numeric(pd.Series(series), errors="coerce").fillna(0).to_numpy(dtype=float)
    vals = np.clip(np.rint(vals), 0, 255).astype(np.uint8)
    if vals.size == length:
        return vals
    out = np.zeros(length, dtype=np.uint8)
    out[: min(length, vals.size)] = vals[: min(length, vals.size)]
    return out


def _role_summary_from_codes(codes: np.ndarray) -> tuple[str, bool, bool, bool, bool]:
    flat = np.asarray(codes, dtype=np.uint8).ravel()
    if flat.size == 0:
        return ROLE_AMBIGUOUS, False, False, False, True
    if np.any(flat == 3):
        return ROLE_BED_CORE, True, False, True, False
    if np.any(flat == 2):
        return ROLE_BED_INNER, True, False, False, False
    if np.any(flat == 1):
        return ROLE_BANK_MARGIN, False, True, False, False
    return ROLE_AMBIGUOUS, False, False, False, True


def _classify_longitudinal_profile_coverage(profile: pd.DataFrame) -> pd.DataFrame:
    work = profile.copy()
    authoritative = work.get("authoritative_anchor_present", pd.Series(False, index=work.index)).fillna(False).to_numpy(dtype=bool)
    xs = np.isfinite(pd.to_numeric(work.get("xs_support_elevation_m"), errors="coerce").to_numpy(dtype=float)) if "xs_support_elevation_m" in work.columns else np.zeros(len(work), dtype=bool)
    center = np.isfinite(pd.to_numeric(work.get("centerline_elevation_m"), errors="coerce").to_numpy(dtype=float)) if "centerline_elevation_m" in work.columns else np.zeros(len(work), dtype=bool)
    bank_ref = np.isfinite(pd.to_numeric(work.get("bank_bed_reference_m"), errors="coerce").to_numpy(dtype=float)) if "bank_bed_reference_m" in work.columns else np.zeros(len(work), dtype=bool)
    bank = np.isfinite(pd.to_numeric(work.get("bank_elevation_m"), errors="coerce").to_numpy(dtype=float)) if "bank_elevation_m" in work.columns else np.zeros(len(work), dtype=bool)
    wse = np.isfinite(pd.to_numeric(work.get("wse_elevation_m"), errors="coerce").to_numpy(dtype=float)) if "wse_elevation_m" in work.columns else np.zeros(len(work), dtype=bool)
    source_count = authoritative.astype(int) + xs.astype(int) + center.astype(int) + bank_ref.astype(int) + bank.astype(int) + wse.astype(int)

    # ---- Compute measured-support distance per station ----
    stations = pd.to_numeric(work.get("station_m"), errors="coerce").to_numpy(dtype=float)
    comp_ids = work.get("profile_id", work.get("component_id", pd.Series("default", index=work.index))).to_numpy()
    measured_dist = _compute_station_distance_to_mask(stations, authoritative, comp_ids)
    work["profile_measured_support_distance_m"] = measured_dist.astype(np.float32)

    # ---- Classify: geometric-only XS/centerline support beyond the measured
    #      distance threshold is reclassified as unsupported ----
    far_from_measured = measured_dist > _MEASURED_SUPPORT_DISTANCE_THRESHOLD_M

    measured_support_present = authoritative | (~far_from_measured & (xs | center | bank | bank_ref))

    coverage = np.full(len(work), "unsupported", dtype=object)
    coverage = np.where((bank | bank_ref) & ~far_from_measured, "bank_only", coverage)
    # Only grant xs/centerline support to stations near real authoritative data.
    coverage = np.where(center & ~far_from_measured, "centerline_supported", coverage)
    coverage = np.where(xs & ~far_from_measured, "xs_supported", coverage)
    coverage = np.where(authoritative, "authoritative_anchor", coverage)

    bed_support = work.get("profile_authoritative_bed_support_present", pd.Series(False, index=work.index)).fillna(False).to_numpy(dtype=bool)
    bank_margin = work.get("profile_authoritative_bank_margin_present", pd.Series(False, index=work.index)).fillna(False).to_numpy(dtype=bool)
    point_bed_support_dist = pd.to_numeric(work.get("profile_authoritative_bed_support_distance_m"), errors="coerce").to_numpy(dtype=float) if "profile_authoritative_bed_support_distance_m" in work.columns else np.full(len(work), np.nan, dtype=float)
    if np.any(np.isfinite(point_bed_support_dist)):
        bed_support_dist = point_bed_support_dist
    else:
        bed_support_dist = _compute_station_distance_to_mask(stations, bed_support, comp_ids)
    work["profile_authoritative_bed_support_distance_m"] = bed_support_dist.astype(np.float32)
    work["profile_far_from_authoritative_bed_support"] = (~np.isfinite(bed_support_dist)) | (bed_support_dist > _MEASURED_SUPPORT_DISTANCE_THRESHOLD_M)

    work["profile_support_source_count"] = source_count.astype(np.int16)
    work["profile_support_class"] = coverage
    work["profile_support_present"] = measured_support_present
    work["profile_bank_support_present"] = bank | bank_ref
    work["profile_centerline_support_present"] = center
    work["profile_xs_support_present"] = xs
    work["profile_authoritative_anchor_present"] = authoritative
    work["profile_authoritative_bed_support_present"] = bed_support
    work["profile_authoritative_bank_margin_present"] = bank_margin
    work["profile_wse_support_present"] = wse
    work["profile_far_from_measured_support"] = far_from_measured
    return work


def _summarize_longitudinal_profile_coverage(profile: pd.DataFrame) -> Dict[str, object]:
    classes = profile.get("profile_support_class", pd.Series(dtype=object)).astype(str)
    counts = classes.value_counts(dropna=False).to_dict()
    component_unsupported = {}
    unsupported_component_count = 0
    component_class_counts: Dict[str, int] = {}
    if "profile_id" in profile.columns and not profile.empty:
        for comp, sub in profile.groupby("profile_id", sort=False):
            comp_classes = sub.get("profile_support_class", pd.Series(dtype=object)).astype(str)
            unsupported = int((comp_classes == "unsupported").sum())
            total = int(len(sub))
            comp_class = str(sub.get("component_support_class", pd.Series(["unknown"] * len(sub), index=sub.index)).astype(str).mode(dropna=True).iloc[0]) if total else "unknown"
            component_class_counts[comp_class] = component_class_counts.get(comp_class, 0) + 1
            component_unsupported[str(comp)] = {
                "component_class": comp_class,
                "unsupported_station_count": unsupported,
                "station_count": total,
                "unsupported_fraction": float(unsupported / total) if total else 0.0,
                "support_fraction": float(pd.to_numeric(sub.get("component_support_bed_fraction"), errors="coerce").median()) if "component_support_bed_fraction" in sub.columns else 0.0,
                "median_support_distance_m": float(pd.to_numeric(sub.get("component_support_median_distance_m"), errors="coerce").median()) if "component_support_median_distance_m" in sub.columns else float("nan"),
                "unsupported_step_p95_m": float(pd.to_numeric(sub.get("component_support_step_p95_m"), errors="coerce").median()) if "component_support_step_p95_m" in sub.columns else 0.0,
                "unsupported_curvature_p95_m": float(pd.to_numeric(sub.get("component_support_curvature_p95_m"), errors="coerce").median()) if "component_support_curvature_p95_m" in sub.columns else 0.0,
                "smoothing_regime_used": str(sub.get("component_support_smoothing_regime", pd.Series([comp_class] * len(sub), index=sub.index)).astype(str).mode(dropna=True).iloc[0]) if total else comp_class,
            }
            if unsupported > 0:
                unsupported_component_count += 1
    source_count = pd.to_numeric(profile.get("profile_support_source_count"), errors="coerce").to_numpy(dtype=float) if "profile_support_source_count" in profile.columns else np.array([], dtype=float)
    # Measured-support-distance summary: how far stations are from real authoritative data
    measured_dist = pd.to_numeric(profile.get("profile_measured_support_distance_m"), errors="coerce").to_numpy(dtype=float) if "profile_measured_support_distance_m" in profile.columns else np.array([], dtype=float)
    finite_dist = measured_dist[np.isfinite(measured_dist)] if measured_dist.size else np.array([], dtype=float)
    measured_dist_summary = {}
    if finite_dist.size > 0:
        measured_dist_summary = {
            "median_m": float(np.median(finite_dist)),
            "p75_m": float(np.percentile(finite_dist, 75)),
            "p95_m": float(np.percentile(finite_dist, 95)),
            "max_m": float(np.max(finite_dist)),
            "fraction_gt_500m": float(np.mean(finite_dist > 500.0)),
            "fraction_gt_1000m": float(np.mean(finite_dist > 1000.0)),
        }
    role_counts = profile.get("profile_authoritative_role", pd.Series(dtype=object)).astype(str).value_counts(dropna=False).to_dict()
    bed_support_dist = pd.to_numeric(profile.get("profile_authoritative_bed_support_distance_m"), errors="coerce").to_numpy(dtype=float) if "profile_authoritative_bed_support_distance_m" in profile.columns else np.array([], dtype=float)
    finite_bed_dist = bed_support_dist[np.isfinite(bed_support_dist)] if bed_support_dist.size else np.array([], dtype=float)
    bed_support_distance_summary = {}
    if finite_bed_dist.size > 0:
        bed_support_distance_summary = {
            "median_m": float(np.median(finite_bed_dist)),
            "p75_m": float(np.percentile(finite_bed_dist, 75)),
            "p95_m": float(np.percentile(finite_bed_dist, 95)),
            "max_m": float(np.max(finite_bed_dist)),
            "fraction_gt_500m": float(np.mean(finite_bed_dist > 500.0)),
            "fraction_gt_1000m": float(np.mean(finite_bed_dist > 1000.0)),
        }
    centerline_auth_station = profile.get("centerline_authoritative_support_station_present", pd.Series(False, index=profile.index)).fillna(False).astype(bool)
    centerline_source_counts = profile.get("centerline_sample_source", pd.Series(dtype=object)).astype(str)
    centerline_source_counts = centerline_source_counts[centerline_source_counts.astype(str).str.len() > 0].value_counts(dropna=False).to_dict()
    return {
        "coverage_by_class": {str(k): int(v) for k, v in counts.items()},
        "authoritative_role_counts": {str(k): int(v) for k, v in role_counts.items()},
        "unsupported_station_count": int(counts.get("unsupported", 0)),
        "supported_station_count": int(len(profile) - counts.get("unsupported", 0)),
        "unsupported_component_count": int(unsupported_component_count),
        "component_class_counts": {str(k): int(v) for k, v in component_class_counts.items()},
        "components_with_unsupported_stations": component_unsupported,
        "max_support_source_count": int(np.nanmax(source_count)) if source_count.size and np.any(np.isfinite(source_count)) else 0,
        "min_support_source_count": int(np.nanmin(source_count)) if source_count.size and np.any(np.isfinite(source_count)) else 0,
        "measured_support_distance_summary": measured_dist_summary,
        "authoritative_bed_support_distance_summary": bed_support_distance_summary,
        "centerline_authoritative_support_station_count": int(centerline_auth_station.sum()),
        "centerline_authoritative_support_station_fraction": float(centerline_auth_station.mean()) if len(profile) else 0.0,
        "centerline_sample_source_counts": {str(k): int(v) for k, v in centerline_source_counts.items()},
    }



def _sanitize_profile_for_rasterization(profile: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    if profile.empty:
        return profile, {}
    work = profile.copy()
    diagnostics: Dict[str, Dict[str, float]] = {}
    profile_ref, profile_spread = _row_support_reference(work)
    work["_local_support_ref"] = profile_ref
    work["_local_support_spread"] = np.nan_to_num(profile_spread, nan=0.0)
    keep_parts = []
    for comp, sub in work.groupby("profile_id", sort=False):
        sub = sub.sort_values("station_m").copy()
        stations = pd.to_numeric(sub.get("station_m"), errors="coerce").to_numpy(dtype=float)
        step = _robust_station_step(stations)
        span = float(np.nanmax(stations) - np.nanmin(stations)) if np.any(np.isfinite(stations)) else 0.0
        support_ref = pd.to_numeric(sub.get("_local_support_ref"), errors="coerce").to_numpy(dtype=float)
        support_spread = np.nan_to_num(pd.to_numeric(sub.get("_local_support_spread"), errors="coerce").to_numpy(dtype=float), nan=0.0)
        uncertainty = pd.to_numeric(sub.get("network_backbone_uncertainty_m", sub.get("uncertainty_m")), errors="coerce").to_numpy(dtype=float)
        network = pd.to_numeric(sub.get("network_backbone_elevation_m", sub.get("bed_elevation_m")), errors="coerce").to_numpy(dtype=float)
        bed = pd.to_numeric(sub.get("bed_elevation_m"), errors="coerce").to_numpy(dtype=float)
        anchor_present = sub.get("authoritative_anchor_present", pd.Series(False, index=sub.index)).fillna(False).to_numpy(dtype=bool)
        row_supported = np.isfinite(support_ref) | anchor_present
        min_rows = 2
        min_span = max(1.0 * step, 5.0)
        min_supported = 2
        drop_component = (len(sub) < min_rows) or (span < min_span) or (int(np.count_nonzero(row_supported)) < min_supported)
        component_ref = np.nanmedian(support_ref[row_supported]) if np.any(row_supported) else np.nanmedian(network[np.isfinite(network)])
        component_spread = np.nanstd(support_ref[row_supported], ddof=0) if np.count_nonzero(np.isfinite(support_ref[row_supported])) >= 2 else np.nanstd(network[np.isfinite(network)], ddof=0)
        component_tol = max(4.0, 6.0 * max(float(np.nan_to_num(component_spread, nan=0.25)), 0.25))
        local_tol = np.maximum(2.0, np.maximum(4.0 * np.nan_to_num(support_spread, nan=0.25), 3.0 * np.nan_to_num(uncertainty, nan=0.25)))
        outlier = np.isfinite(network) & np.isfinite(support_ref) & (np.abs(network - support_ref) > local_tol)
        if np.isfinite(component_ref):
            outlier |= np.isfinite(network) & (np.abs(network - component_ref) > component_tol)
        if np.any(outlier):
            network[outlier] = np.nan
            bed[outlier] = np.where(np.isfinite(support_ref[outlier]), support_ref[outlier], np.nan)
            if "network_backbone_uncertainty_m" in sub.columns:
                sub.loc[outlier, "network_backbone_uncertainty_m"] = np.maximum(np.nan_to_num(uncertainty[outlier], nan=0.25), 2.0)
        sub.loc[:, "network_backbone_elevation_m"] = network
        sub.loc[:, "bed_elevation_m"] = bed
        finite_backbone = np.isfinite(network)
        diagnostics[str(comp)] = {
            "n_rows": int(len(sub)),
            "station_span_m": float(span),
            "supported_rows": int(np.count_nonzero(row_supported)),
            "outlier_rows": int(np.count_nonzero(outlier)),
            "kept_rows": int(np.count_nonzero(finite_backbone)),
        }
        if drop_component or int(np.count_nonzero(finite_backbone)) < min_supported:
            diagnostics[str(comp)]["dropped"] = 1.0
            continue
        diagnostics[str(comp)]["dropped"] = 0.0
        keep_parts.append(sub.drop(columns=["_local_support_ref", "_local_support_spread"], errors="ignore"))
    if not keep_parts:
        return profile.iloc[0:0].copy(), diagnostics
    out = pd.concat(keep_parts, ignore_index=True)
    return out, diagnostics



def _build_centerline_profile_grids(*, shape, transform, domain_mask: np.ndarray, stationing: np.ndarray, centerline_points, profile: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    from scipy.spatial import cKDTree
    import rasterio.transform

    centerline_mask = np.asarray(domain_mask, dtype=bool) & np.isfinite(stationing)
    value_grid = np.full(shape, np.nan, dtype=np.float32)
    uncertainty_grid = np.full(shape, np.nan, dtype=np.float32)
    component_index = np.full(shape, -1, dtype=np.int32)
    if not np.any(centerline_mask) or centerline_points is None or getattr(centerline_points, "empty", True) or profile.empty:
        return value_grid, uncertainty_grid, component_index, {}

    cpts = centerline_points.copy()
    cpts["profile_id"] = _infer_profile_id(cpts).astype(str)
    valid_geom = cpts.geometry.notnull()
    cpts = cpts.loc[valid_geom].copy()
    if cpts.empty:
        return value_grid, uncertainty_grid, component_index, {}

    src_xy = np.column_stack([cpts.geometry.x.to_numpy(dtype=float), cpts.geometry.y.to_numpy(dtype=float)])
    tree = cKDTree(src_xy)
    rows, cols = np.where(centerline_mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    q_xy = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    _, idx = tree.query(q_xy, k=1)
    assigned_profile = cpts.iloc[np.asarray(idx, dtype=int)]["profile_id"].astype(str).to_numpy()
    q_station = stationing[rows, cols].astype(float)

    unique_profile_ids = sorted({str(pid) for pid in assigned_profile.tolist()})
    profile_lookup = {pid: i for i, pid in enumerate(unique_profile_ids)}
    inverse_lookup = {int(i): str(pid) for pid, i in profile_lookup.items()}
    for pid in unique_profile_ids:
        comp_mask = assigned_profile == pid
        sub = profile.loc[profile["profile_id"].astype(str) == str(pid)].sort_values("station_m").copy()
        if sub.empty:
            continue
        ref_station = pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float)
        ref_value = pd.to_numeric(sub.get("active_core_support_elevation_m", sub["network_backbone_elevation_m"]), errors="coerce").to_numpy(dtype=float)
        ref_unc = pd.to_numeric(sub.get("active_core_support_uncertainty_m", sub["network_backbone_uncertainty_m"]), errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(ref_station) & np.isfinite(ref_value)
        if np.count_nonzero(finite) < 2:
            continue
        ref_station = ref_station[finite]
        ref_value = ref_value[finite]
        ref_unc = ref_unc[finite] if ref_unc.shape == finite.shape else np.full(ref_station.shape, np.nan, dtype=float)
        order = np.argsort(ref_station)
        ref_station = ref_station[order]
        ref_value = ref_value[order]
        ref_unc = ref_unc[order]
        qs = q_station[comp_mask]
        inside = (qs >= (float(ref_station[0]) - 0.5 * _robust_station_step(ref_station))) & (qs <= (float(ref_station[-1]) + 0.5 * _robust_station_step(ref_station)))
        if not np.any(inside):
            continue
        interp_val = np.full(qs.shape, np.nan, dtype=np.float32)
        interp_unc = np.full(qs.shape, np.nan, dtype=np.float32)
        interp_val[inside] = np.interp(qs[inside], ref_station, ref_value).astype(np.float32)
        finite_unc = np.isfinite(ref_unc)
        if np.count_nonzero(finite_unc) >= 2:
            interp_unc[inside] = np.interp(qs[inside], ref_station[finite_unc], ref_unc[finite_unc]).astype(np.float32)
        elif np.any(finite_unc):
            interp_unc[inside] = np.float32(np.nanmedian(ref_unc[finite_unc]))
        r = rows[comp_mask]
        c = cols[comp_mask]
        value_grid[r, c] = interp_val
        uncertainty_grid[r, c] = interp_unc
        component_index[r, c] = int(profile_lookup[pid])
    return value_grid, uncertainty_grid, component_index, inverse_lookup



def _transport_centerline_profile_to_corridor(*, domain_mask: np.ndarray, centerline_mask: np.ndarray, stationing: np.ndarray, value_grid: np.ndarray, uncertainty_grid: np.ndarray, component_index: np.ndarray, pixel_size_m: float, along_scale_m: float, lateral_decay_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import distance_transform_edt

    domain_mask = np.asarray(domain_mask, dtype=bool)
    centerline_mask = np.asarray(centerline_mask, dtype=bool) & domain_mask
    transported = np.full(value_grid.shape, np.nan, dtype=np.float32)
    transported_unc = np.full(value_grid.shape, np.nan, dtype=np.float32)
    influence = np.zeros(value_grid.shape, dtype=np.float32)
    if not np.any(domain_mask) or not np.any(centerline_mask):
        return transported, transported_unc, influence

    valid_centerline = centerline_mask & np.isfinite(value_grid) & (component_index >= 0)
    if not np.any(valid_centerline):
        return transported, transported_unc, influence
    dist_to_center, nearest_centerline = distance_transform_edt(~centerline_mask, return_indices=True)
    dist_to_center = dist_to_center.astype(np.float32) * np.float32(max(pixel_size_m, 1.0e-6))
    query_component = np.full(domain_mask.shape, -1, dtype=np.int32)
    query_component[domain_mask] = component_index[nearest_centerline[0][domain_mask], nearest_centerline[1][domain_mask]]
    safe_lateral = max(float(lateral_decay_m), max(float(pixel_size_m), 1.0))

    valid_components = sorted({int(v) for v in np.unique(component_index[valid_centerline]).tolist() if int(v) >= 0})
    for comp_idx in valid_components:
        comp_valid = valid_centerline & (component_index == int(comp_idx))
        comp_center = centerline_mask & (component_index == int(comp_idx))
        if not np.any(comp_center) or np.count_nonzero(comp_valid) < 2:
            continue

        ref_station = stationing[comp_valid].astype(np.float32)
        ref_value = value_grid[comp_valid].astype(np.float32)
        ref_unc = uncertainty_grid[comp_valid].astype(np.float32)
        finite = np.isfinite(ref_station) & np.isfinite(ref_value)
        if np.count_nonzero(finite) < 2:
            continue
        ref_station = ref_station[finite]
        ref_value = ref_value[finite]
        ref_unc = ref_unc[finite] if ref_unc.shape == finite.shape else np.full(ref_station.shape, np.nan, dtype=np.float32)
        order = np.argsort(ref_station, kind="mergesort")
        ref_station = ref_station[order]
        ref_value = ref_value[order]
        ref_unc = ref_unc[order]

        uniq_station, inverse = np.unique(ref_station, return_inverse=True)
        if uniq_station.size < 2:
            continue
        agg_value = np.full(uniq_station.shape, np.nan, dtype=np.float32)
        agg_unc = np.full(uniq_station.shape, np.nan, dtype=np.float32)
        for i in range(uniq_station.size):
            sel = inverse == i
            vals = ref_value[sel]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                agg_value[i] = np.float32(np.nanmedian(vals))
            uncs = ref_unc[sel]
            uncs = uncs[np.isfinite(uncs)]
            if uncs.size:
                agg_unc[i] = np.float32(np.nanmedian(uncs))

        qmask = domain_mask & np.isfinite(stationing) & (query_component == int(comp_idx))
        if not np.any(qmask):
            continue
        q_station = stationing[qmask].astype(np.float32)
        step = float(np.nanmedian(np.diff(uniq_station))) if uniq_station.size >= 2 else max(float(pixel_size_m), 1.0)
        tol = max(step * 1.5, max(float(pixel_size_m), 1.0))
        inside = (q_station >= (float(uniq_station[0]) - tol)) & (q_station <= (float(uniq_station[-1]) + tol))
        if not np.any(inside):
            continue
        q_rows, q_cols = np.nonzero(qmask)
        q_rows = q_rows[inside]
        q_cols = q_cols[inside]
        q_station = q_station[inside]
        transported[q_rows, q_cols] = np.interp(q_station, uniq_station, agg_value).astype(np.float32)

        finite_unc = np.isfinite(agg_unc)
        if np.count_nonzero(finite_unc) >= 2:
            transported_unc[q_rows, q_cols] = np.interp(q_station, uniq_station[finite_unc], agg_unc[finite_unc]).astype(np.float32)
        elif np.any(finite_unc):
            transported_unc[q_rows, q_cols] = np.float32(np.nanmedian(agg_unc[finite_unc]))

        ins = np.searchsorted(uniq_station, q_station)
        best_along = np.full(q_station.shape, np.inf, dtype=np.float32)
        right = ins < int(uniq_station.size)
        if np.any(right):
            best_along[right] = np.minimum(best_along[right], np.abs(q_station[right] - uniq_station[ins[right]]).astype(np.float32))
        left = ins > 0
        if np.any(left):
            best_along[left] = np.minimum(best_along[left], np.abs(q_station[left] - uniq_station[ins[left] - 1]).astype(np.float32))
        safe_along = max(float(along_scale_m), max(float(step), 1.0))
        lateral = dist_to_center[q_rows, q_cols]
        along_influence = np.exp(-0.5 * np.square(best_along / safe_along)).astype(np.float32)
        lateral_influence = np.exp(-0.5 * np.square(lateral / safe_lateral)).astype(np.float32)
        influence_vals = (along_influence * lateral_influence).astype(np.float32)
        influence[q_rows, q_cols] = np.clip(influence_vals, 0.0, 1.0).astype(np.float32)
        if np.any(np.isfinite(transported_unc[q_rows, q_cols])):
            transported_unc[q_rows, q_cols] = np.where(
                np.isfinite(transported_unc[q_rows, q_cols]),
                transported_unc[q_rows, q_cols] + (0.15 * (lateral / safe_lateral)).astype(np.float32) + (0.10 * (best_along / safe_along)).astype(np.float32),
                transported_unc[q_rows, q_cols],
            ).astype(np.float32)
    influence[~np.isfinite(transported)] = 0.0
    return transported, transported_unc, influence



def _sanitize_transported_profile_raster(
    *,
    profile: pd.DataFrame,
    transported: np.ndarray,
    transported_unc: np.ndarray,
    influence: np.ndarray,
    component_index: np.ndarray,
    component_lookup: dict[int, str],
    centerline_mask: np.ndarray,
    domain_mask: np.ndarray,
    pixel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    transported = np.asarray(transported, dtype=np.float32).copy()
    transported_unc = np.asarray(transported_unc, dtype=np.float32).copy()
    influence = np.asarray(influence, dtype=np.float32).copy()
    component_index = np.asarray(component_index, dtype=np.int32)
    centerline_mask = np.asarray(centerline_mask, dtype=bool)
    domain_mask = np.asarray(domain_mask, dtype=bool)
    diagnostics: Dict[str, Dict[str, float]] = {}
    if profile.empty:
        influence[~np.isfinite(transported)] = 0.0
        return transported, transported_unc, influence, diagnostics

    min_cells = max(3, int(np.ceil(2.0 * max(float(pixel_size_m), 1.0))))
    for comp_idx, profile_id in component_lookup.items():
        sub = profile.loc[profile["profile_id"].astype(str) == str(profile_id)].copy()
        finite_backbone = pd.to_numeric(sub.get("network_backbone_elevation_m"), errors="coerce").to_numpy(dtype=float)
        finite_backbone = finite_backbone[np.isfinite(finite_backbone)]
        if finite_backbone.size == 0:
            continue
        ref = float(np.nanmedian(finite_backbone))
        uncertainty = pd.to_numeric(sub.get("network_backbone_uncertainty_m"), errors="coerce").to_numpy(dtype=float)
        uncertainty = uncertainty[np.isfinite(uncertainty)]
        spread = float(np.nanstd(finite_backbone, ddof=0)) if finite_backbone.size >= 2 else 0.0
        unc_med = float(np.nanmedian(uncertainty)) if uncertainty.size else 0.25
        tol = max(8.0, 6.0 * max(spread, unc_med, 0.25))
        comp_mask = domain_mask & (component_index == int(comp_idx)) & np.isfinite(transported)
        center_mask = comp_mask & centerline_mask
        component_cells = int(np.count_nonzero(comp_mask))
        center_cells = int(np.count_nonzero(center_mask))
        diagnostics[str(profile_id)] = {
            "component_cells": float(component_cells),
            "centerline_cells": float(center_cells),
            "component_ref_m": ref,
            "component_tol_m": tol,
        }
        if component_cells < min_cells or center_cells < 2:
            transported[comp_mask] = np.nan
            transported_unc[comp_mask] = np.nan
            influence[comp_mask] = 0.0
            diagnostics[str(profile_id)]["dropped_component"] = 1.0
            continue
        lower = ref - tol
        upper = ref + tol
        outlier = comp_mask & ((transported < lower) | (transported > upper) | ~np.isfinite(transported))
        if np.any(outlier):
            transported[outlier] = np.nan
            transported_unc[outlier] = np.nan
            influence[outlier] = 0.0
        remain_center = center_mask & np.isfinite(transported)
        remain_comp = (component_index == int(comp_idx)) & domain_mask & np.isfinite(transported)
        diagnostics[str(profile_id)]["outlier_cells_removed"] = float(np.count_nonzero(outlier))
        diagnostics[str(profile_id)]["remaining_component_cells"] = float(np.count_nonzero(remain_comp))
        diagnostics[str(profile_id)]["remaining_centerline_cells"] = float(np.count_nonzero(remain_center))
        if np.count_nonzero(remain_center) < 2 or np.count_nonzero(remain_comp) < min_cells:
            transported[remain_comp] = np.nan
            transported_unc[remain_comp] = np.nan
            influence[remain_comp] = 0.0
            diagnostics[str(profile_id)]["dropped_after_outlier_gate"] = 1.0
        else:
            diagnostics[str(profile_id)]["dropped_after_outlier_gate"] = 0.0

    influence[~np.isfinite(transported)] = 0.0
    transported_unc[~np.isfinite(transported)] = np.nan
    return transported, transported_unc, influence, diagnostics


def _first_numeric_series(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[pd.Series]:
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if np.isfinite(vals.to_numpy(dtype=float)).any():
                return vals
    return None


def _component_hierarchy_metadata(edges) -> Dict[str, Dict[str, float]]:
    if edges is None or len(edges) == 0 or "component_id" not in edges.columns:
        return {}
    work = edges.copy()
    drainage = _first_numeric_series(work, (
        "drainage_area_sqkm", "drainage_area_km2", "drainage_area", "totdasqkm",
        "divdasqkm", "areasqkm", "area_sqkm", "catchment_km2", "uparea_km2",
        "arbolatesu",
    ))
    stream_order = _first_numeric_series(work, (
        "stream_order", "streamorde", "streamorde_", "streamlevel", "strahler",
        "strahler_order", "order_", "order",
    ))
    if drainage is not None:
        work["_drainage_proxy"] = drainage
    if stream_order is not None:
        work["_stream_order_proxy"] = stream_order
    meta: Dict[str, Dict[str, float]] = {}
    for comp_id, grp in work.groupby("component_id", dropna=False):
        comp = str(comp_id)
        drainage_val = np.nan
        order_val = np.nan
        if "_drainage_proxy" in grp.columns:
            arr = pd.to_numeric(grp["_drainage_proxy"], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0.0)]
            if arr.size:
                drainage_val = float(np.nanmax(arr))
        if "_stream_order_proxy" in grp.columns:
            arr = pd.to_numeric(grp["_stream_order_proxy"], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0.0)]
            if arr.size:
                order_val = float(np.nanmax(arr))
        base_priority = 1.0
        if np.isfinite(drainage_val) and drainage_val > 0.0:
            base_priority *= max(np.sqrt(drainage_val), 1.0)
        if np.isfinite(order_val) and order_val > 0.0:
            base_priority *= max(1.0 + 0.35 * (order_val - 1.0), 1.0)
        meta[comp] = {
            "drainage_area_proxy": drainage_val,
            "stream_order_proxy": order_val,
            "hierarchy_priority": float(base_priority),
        }
    return meta



def _compute_local_gradient(stations: np.ndarray, values: np.ndarray) -> np.ndarray:
    stations = np.asarray(stations, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(stations) & np.isfinite(values)
    idx = np.where(valid)[0]
    if idx.size < 2:
        return out
    for pos, i in enumerate(idx):
        if pos == 0:
            j = idx[pos + 1]
            ds = stations[j] - stations[i]
            if np.isfinite(ds) and abs(ds) > 0.0:
                out[i] = (values[j] - values[i]) / ds
        elif pos == idx.size - 1:
            j = idx[pos - 1]
            ds = stations[i] - stations[j]
            if np.isfinite(ds) and abs(ds) > 0.0:
                out[i] = (values[i] - values[j]) / ds
        else:
            j0 = idx[pos - 1]
            j1 = idx[pos + 1]
            ds = stations[j1] - stations[j0]
            if np.isfinite(ds) and abs(ds) > 0.0:
                out[i] = (values[j1] - values[j0]) / ds
    return out

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        vals = values[np.isfinite(values)]
        return float(np.median(vals)) if vals.size else np.nan
    return float(np.sum(values[finite] * weights[finite]) / np.sum(weights[finite]))


def _weighted_spread(values: np.ndarray, weights: np.ndarray, center: float) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite) or not np.isfinite(center):
        vals = values[np.isfinite(values)]
        if vals.size >= 2:
            return float(np.nanstd(vals, ddof=0))
        return 0.0
    diffs = values[finite] - center
    return float(np.sqrt(np.sum(weights[finite] * diffs * diffs) / np.sum(weights[finite])))




def _smooth_anchor_curve_candidates(values: np.ndarray, station: np.ndarray, support_distance_m: np.ndarray | None = None, base_window_points: int = 3) -> tuple[np.ndarray, Dict[str, float]]:
    vals = np.asarray(values, dtype=float)
    sta = np.asarray(station, dtype=float)
    if vals.size == 0:
        return vals.copy(), {"window_points": 0.0, "window_m": 0.0, "median_window_points": 0.0, "far_support_fraction_mean": 0.0}
    n = int(vals.size)
    span = float(np.nanmax(sta) - np.nanmin(sta)) if np.count_nonzero(np.isfinite(sta)) >= 2 else 0.0
    step = _robust_station_step(sta)
    component_seed = int(round(span / max(80.0, 6.0 * max(step, 1.0))))
    component_scale = max(base_window_points, int(np.clip(component_seed, 3, max(3, min(11, n if n % 2 == 1 else max(1, n - 1))))))
    if component_scale % 2 == 0:
        component_scale += 1
    component_scale = min(component_scale, n if n % 2 == 1 else max(1, n - 1))
    component_scale = max(1, component_scale)
    median_window = min(component_scale, n if n % 2 == 1 else max(1, n - 1))
    smoothed = _rolling_nanmedian(vals, median_window)
    mean_window = min(max(component_scale, base_window_points + 2), n)
    mean_smoothed = _rolling_nanmean(smoothed, mean_window)
    out = np.where(np.isfinite(mean_smoothed), mean_smoothed, smoothed)
    far_support = np.zeros(n, dtype=float)
    if support_distance_m is not None:
        dist = np.asarray(support_distance_m, dtype=float)
        far_support = np.clip((dist - 75.0) / 250.0, 0.0, 1.0)
        far_support = np.where(np.isfinite(far_support), far_support, 0.0)
        seed = float(np.nanmedian(vals[np.isfinite(vals)])) if np.any(np.isfinite(vals)) else np.nan
        if np.isfinite(seed):
            out = np.where(np.isfinite(out), (1.0 - 0.35 * far_support) * out + (0.35 * far_support) * seed, out)
    return out, {
        "window_points": float(mean_window),
        "window_m": float(mean_window * max(step, 1.0)),
        "median_window_points": float(median_window),
        "far_support_fraction_mean": float(np.mean(far_support)) if far_support.size else 0.0,
    }

def _build_authoritative_bed_anchor_curve(profile: pd.DataFrame, endpoint_meta: Dict[str, Dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    if profile.empty:
        return profile, pd.DataFrame(), {}
    out = profile.copy()
    raw_present = out.get("authoritative_anchor_present", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    raw_elev = pd.to_numeric(out.get("authoritative_anchor_elevation_m"), errors="coerce")
    raw_source = out.get("authoritative_anchor_source", pd.Series(["none"] * len(out), index=out.index)).astype(str)
    out["authoritative_anchor_raw_present"] = raw_present
    out["authoritative_anchor_raw_elevation_m"] = raw_elev
    out["authoritative_anchor_raw_source"] = raw_source
    out["authoritative_anchor_curve_present"] = False
    out["authoritative_anchor_curve_elevation_m"] = np.nan
    out["authoritative_anchor_curve_bin_m"] = np.nan
    out["authoritative_anchor_curve_count"] = 0
    out["authoritative_anchor_curve_dispersion_m"] = np.nan
    out["authoritative_anchor_curve_adjustment_m"] = np.nan
    out["authoritative_anchor_curve_source"] = "none"
    rows = []
    diagnostics: Dict[str, Dict[str, float]] = {}

    for comp, idx in out.groupby("profile_id", sort=False).groups.items():
        sub = out.loc[idx].sort_values("station_m").copy()
        station = pd.to_numeric(sub.get("station_m"), errors="coerce").to_numpy(dtype=float)
        anchor_present = sub.get("authoritative_anchor_raw_present", pd.Series(False, index=sub.index)).fillna(False).to_numpy(dtype=bool)
        anchor_z = pd.to_numeric(sub.get("authoritative_anchor_raw_elevation_m"), errors="coerce").to_numpy(dtype=float)
        finite_anchor = anchor_present & np.isfinite(anchor_z) & np.isfinite(station)
        if not np.any(finite_anchor):
            diagnostics[str(comp)] = {
                "raw_anchor_rows": 0,
                "curve_anchor_rows": 0,
                "bin_width_m": 0.0,
                "smoothing_window_points": 0.0,
                "smoothing_window_m": 0.0,
                "median_window_points": 0.0,
                "far_support_fraction_mean": 0.0,
                "pre_monotone_violation_count": 0,
                "post_monotone_violation_count": 0,
                "max_adjustment_m": 0.0,
                "median_adjustment_m": 0.0,
                "exact_fixed_rows": 0,
            }
            continue
        step = _robust_station_step(station)
        support_distance = pd.to_numeric(sub.get("profile_authoritative_bed_support_distance_m"), errors="coerce").to_numpy(dtype=float)
        anchor_station = station[finite_anchor]
        anchor_span = float(np.nanmax(anchor_station) - np.nanmin(anchor_station)) if anchor_station.size >= 2 else 0.0
        anchor_count = int(np.count_nonzero(finite_anchor))
        long_component = anchor_span >= 500.0 or anchor_count >= 12
        coarse_bin_width = float(np.clip(max(60.0 if long_component else 50.0, (8.0 if long_component else 5.0) * max(step, 1.0)), 60.0 if long_component else 50.0, 180.0 if long_component else 100.0))
        endpoint_preserving_width = max(2.0 * max(step, 1.0), anchor_span / (4.0 if long_component else 2.0)) if anchor_station.size >= 2 else coarse_bin_width
        bin_width = float(min(coarse_bin_width, endpoint_preserving_width))
        base_station = float(np.nanmin(anchor_station))
        groups = np.rint((station[finite_anchor] - base_station) / bin_width).astype(int)
        src = sub.get("authoritative_anchor_raw_source", pd.Series(["none"] * len(sub), index=sub.index)).astype(str).to_numpy(dtype=object)
        fluvial = sub.get("profile_inside_fluvial_monotone_domain", pd.Series(True, index=sub.index)).fillna(True).to_numpy(dtype=bool)
        candidate_idx = np.flatnonzero(finite_anchor)
        bin_records = []
        for gid in np.unique(groups):
            sel_global = candidate_idx[groups == gid]
            if sel_global.size == 0:
                continue
            gstation = station[sel_global]
            gz = anchor_z[sel_global]
            center_station = float(np.nanmedian(gstation))
            rep = int(sel_global[int(np.nanargmin(np.abs(gstation - center_station)))])
            gdisp = float(np.nanstd(gz, ddof=0)) if gz.size >= 2 else 0.0
            gcount = int(gz.size)
            gvalue = float(np.nanmedian(gz))
            source_values = np.asarray(src[sel_global].astype(str), dtype=object)
            unique_sources = [str(v) for v in pd.unique(pd.Series(source_values)) if str(v) and str(v) != "nan"]
            if len(unique_sources) == 1:
                gsource = unique_sources[0]
            else:
                gsource = "authoritative_anchor_curve"
                if np.any(np.char.find(source_values.astype(str), "bed_core") >= 0):
                    gsource = "authoritative_bed_core_anchor_curve"
                elif np.any(np.char.find(source_values.astype(str), "support_point") >= 0):
                    gsource = "authoritative_bed_support_anchor_curve"
            exact_fixed = any((v == "authoritative_anchor_raster") or ("bed_core" in v) for v in unique_sources)
            bin_records.append({
                "gid": int(gid),
                "rep": rep,
                "station_m": float(station[rep]),
                "value_m": gvalue,
                "count": gcount,
                "dispersion_m": gdisp,
                "source": gsource,
                "fluvial": bool(np.any(fluvial[sel_global])),
                "exact_fixed": bool(exact_fixed),
            })
        if not bin_records:
            continue
        curve = pd.DataFrame(bin_records).sort_values("station_m").reset_index(drop=True)
        raw_curve = curve["value_m"].to_numpy(dtype=float)
        solved_curve = raw_curve.copy()
        weights = np.clip(curve["count"].to_numpy(dtype=float) / np.square(np.maximum(curve["dispersion_m"].to_numpy(dtype=float), 0.05)), 1.0, 1.0e6)
        curve_support_distance = np.full(len(curve), np.nan, dtype=float)
        if support_distance.size == len(station):
            for i, rep in enumerate(curve["rep"].to_numpy(dtype=int)):
                curve_support_distance[i] = float(support_distance[rep]) if rep < support_distance.size else np.nan
        smoothed_curve, smoothing_diag = _smooth_anchor_curve_candidates(raw_curve, curve["station_m"].to_numpy(dtype=float), curve_support_distance, base_window_points=3 if not long_component else 5)
        prefit_curve = np.where(np.isfinite(smoothed_curve), smoothed_curve, raw_curve)
        downstream_increasing = _component_downstream_increasing(endpoint_meta, str(comp))
        fluvial_curve = curve["fluvial"].to_numpy(dtype=bool)
        run_start = None
        for i, active in enumerate(list(fluvial_curve) + [False]):
            if active and run_start is None:
                run_start = i
            elif (not active) and (run_start is not None):
                seg = slice(run_start, i)
                seg_vals = prefit_curve[seg]
                seg_wts = weights[seg]
                if seg_vals.size >= 2:
                    ordered_vals = seg_vals if downstream_increasing else seg_vals[::-1]
                    ordered_wts = seg_wts if downstream_increasing else seg_wts[::-1]
                    fitted = _pava_isotonic(ordered_vals, increasing=False, weights=ordered_wts)
                    fitted = fitted if downstream_increasing else fitted[::-1]
                    solved_curve[seg] = fitted
                run_start = None
        exact_fixed_curve = curve["exact_fixed"].to_numpy(dtype=bool) if "exact_fixed" in curve.columns else np.zeros(len(curve), dtype=bool)
        if np.any(exact_fixed_curve):
            solved_curve[exact_fixed_curve] = raw_curve[exact_fixed_curve]
        curve["curve_elevation_m"] = solved_curve
        curve["adjustment_m"] = solved_curve - raw_curve
        for _, row in curve.iterrows():
            rep = int(row["rep"])
            out.at[out.index[rep], "authoritative_anchor_curve_present"] = True
            out.at[out.index[rep], "authoritative_anchor_curve_elevation_m"] = float(row["curve_elevation_m"])
            out.at[out.index[rep], "authoritative_anchor_curve_bin_m"] = float(row["gid"] * bin_width + base_station)
            out.at[out.index[rep], "authoritative_anchor_curve_count"] = int(row["count"])
            out.at[out.index[rep], "authoritative_anchor_curve_dispersion_m"] = float(row["dispersion_m"])
            out.at[out.index[rep], "authoritative_anchor_curve_adjustment_m"] = float(row["adjustment_m"])
            out.at[out.index[rep], "authoritative_anchor_curve_source"] = str(row["source"])
            rows.append({
                "profile_id": str(comp),
                "station_m": float(row["station_m"]),
                "authoritative_anchor_curve_elevation_m": float(row["curve_elevation_m"]),
                "authoritative_anchor_curve_count": int(row["count"]),
                "authoritative_anchor_curve_dispersion_m": float(row["dispersion_m"]),
                "authoritative_anchor_curve_adjustment_m": float(row["adjustment_m"]),
                "authoritative_anchor_curve_source": str(row["source"]),
                "profile_inside_fluvial_monotone_domain": bool(row["fluvial"]),
                "authoritative_anchor_curve_bin_m": float(row["gid"] * bin_width + base_station),
            })
        raw_ordered = raw_curve if downstream_increasing else raw_curve[::-1]
        solved_ordered = solved_curve if downstream_increasing else solved_curve[::-1]
        diagnostics[str(comp)] = {
            "raw_anchor_rows": int(np.count_nonzero(finite_anchor)),
            "curve_anchor_rows": int(len(curve)),
            "bin_width_m": bin_width,
            "smoothing_window_points": float(smoothing_diag.get("window_points", 0.0)),
            "smoothing_window_m": float(smoothing_diag.get("window_m", 0.0)),
            "median_window_points": float(smoothing_diag.get("median_window_points", 0.0)),
            "far_support_fraction_mean": float(smoothing_diag.get("far_support_fraction_mean", 0.0)),
            "pre_monotone_violation_count": int(np.count_nonzero(np.diff(raw_ordered) > 0.0)) if raw_ordered.size >= 2 else 0,
            "post_monotone_violation_count": int(np.count_nonzero(np.diff(solved_ordered) > 1.0e-6)) if solved_ordered.size >= 2 else 0,
            "max_adjustment_m": float(np.nanmax(np.abs(solved_curve - raw_curve))) if raw_curve.size else 0.0,
            "median_adjustment_m": float(np.nanmedian(np.abs(solved_curve - raw_curve))) if raw_curve.size else 0.0,
            "exact_fixed_rows": int(np.count_nonzero(exact_fixed_curve)),
        }

    curve_present = out["authoritative_anchor_curve_present"].fillna(False).to_numpy(dtype=bool)
    if np.any(curve_present):
        out["authoritative_anchor_present"] = curve_present
        out["authoritative_anchor_elevation_m"] = np.where(curve_present, pd.to_numeric(out["authoritative_anchor_curve_elevation_m"], errors="coerce"), np.nan)
        out["authoritative_anchor_source"] = np.where(curve_present, out["authoritative_anchor_curve_source"].astype(str), "none")
    anchor_curve = pd.DataFrame(rows, columns=[
        "profile_id", "station_m", "authoritative_anchor_curve_elevation_m",
        "authoritative_anchor_curve_count", "authoritative_anchor_curve_dispersion_m",
        "authoritative_anchor_curve_adjustment_m", "authoritative_anchor_curve_source",
        "profile_inside_fluvial_monotone_domain", "authoritative_anchor_curve_bin_m",
    ])
    return out, anchor_curve, diagnostics


def _build_profile_dataframe(centerline_points, sampled: pd.DataFrame) -> pd.DataFrame:
    df = centerline_points.copy()
    df["profile_id"] = _infer_profile_id(df)
    df["station_m"] = pd.to_numeric(df.get("station_m"), errors="coerce")
    df["centerline_elevation_m"] = pd.to_numeric(sampled["centerline_elevation_m"], errors="coerce")
    df["centerline_influence"] = pd.to_numeric(sampled["centerline_influence"], errors="coerce").fillna(0.0)
    df["centerline_sample_source"] = sampled.get("centerline_sample_source", pd.Series(["missing"] * len(df), index=df.index, dtype=object)).fillna("missing").astype(str)
    df["centerline_authoritative_support_applied"] = sampled.get("centerline_authoritative_support_applied", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["centerline_authoritative_support_distance_m"] = pd.to_numeric(sampled.get("centerline_authoritative_support_distance_m"), errors="coerce")
    df["xs_support_elevation_m"] = pd.to_numeric(sampled["xs_support_elevation_m"], errors="coerce")
    df["xs_support_weight"] = pd.to_numeric(sampled["xs_support_weight"], errors="coerce").fillna(0.0)
    df["bank_elevation_m"] = pd.to_numeric(sampled["bank_elevation_m"], errors="coerce")
    bank_weight = pd.to_numeric(sampled["bank_influence"], errors="coerce").fillna(0.0)
    bank_weight *= pd.to_numeric(sampled["bank_graph_confidence"], errors="coerce").fillna(1.0)
    bank_weight *= pd.to_numeric(sampled["bank_continuity_weight"], errors="coerce").fillna(1.0)
    bank_weight *= pd.to_numeric(sampled["bank_confluence_damping"], errors="coerce").fillna(1.0)
    bank_weight *= pd.to_numeric(sampled["bank_estuary_side_decay"], errors="coerce").fillna(1.0)
    df["bank_weight"] = bank_weight.clip(lower=0.0, upper=1.0)
    df["authoritative_support_depth_m"] = pd.to_numeric(sampled["authoritative_support_depth_m"], errors="coerce")
    df["authoritative_anchor_elevation_m"] = pd.to_numeric(sampled.get("authoritative_anchor_elevation_m"), errors="coerce")
    authoritative_mask = pd.to_numeric(sampled.get("authoritative_support_mask"), errors="coerce").fillna(0.0)
    df["authoritative_support_mask"] = authoritative_mask
    df.loc[df["authoritative_support_mask"] <= 0.0, "authoritative_anchor_elevation_m"] = np.nan
    df["authoritative_role_code"] = _coerce_role_codes(sampled.get("authoritative_role_code"), len(df))
    df["authoritative_role_confidence"] = pd.to_numeric(sampled.get("authoritative_role_confidence"), errors="coerce")
    df["authoritative_distance_to_bank_m"] = pd.to_numeric(sampled.get("authoritative_distance_to_bank_m"), errors="coerce")
    df["authoritative_normalized_channel_position"] = pd.to_numeric(sampled.get("authoritative_normalized_channel_position"), errors="coerce")
    df["authoritative_support_point_distance_m"] = pd.to_numeric(sampled.get("authoritative_support_point_distance_m"), errors="coerce")
    df["authoritative_support_point_role_code"] = _coerce_role_codes(sampled.get("authoritative_support_point_role_code"), len(df))
    df["authoritative_support_point_role_confidence"] = pd.to_numeric(sampled.get("authoritative_support_point_role_confidence"), errors="coerce")
    df["authoritative_support_point_distance_to_bank_m"] = pd.to_numeric(sampled.get("authoritative_support_point_distance_to_bank_m"), errors="coerce")
    df["authoritative_support_point_normalized_channel_position"] = pd.to_numeric(sampled.get("authoritative_support_point_normalized_channel_position"), errors="coerce")
    df["authoritative_support_point_half_width_est_m"] = pd.to_numeric(sampled.get("authoritative_support_point_half_width_est_m"), errors="coerce")
    df["authoritative_support_point_z_m"] = pd.to_numeric(sampled.get("authoritative_support_point_z_m"), errors="coerce")
    df["authoritative_support_point_inside_channel_mask"] = pd.to_numeric(sampled.get("authoritative_support_point_inside_channel_mask"), errors="coerce")
    df["authoritative_support_point_inside_river_guidance_domain"] = pd.to_numeric(sampled.get("authoritative_support_point_inside_river_guidance_domain"), errors="coerce")
    df["authoritative_support_point_inside_mainstem_mask"] = pd.to_numeric(sampled.get("authoritative_support_point_inside_mainstem_mask"), errors="coerce")
    df["authoritative_support_point_inside_estuary_clip"] = pd.to_numeric(sampled.get("authoritative_support_point_inside_estuary_clip"), errors="coerce")
    df["authoritative_support_local_radius_m"] = pd.to_numeric(sampled.get("authoritative_support_local_radius_m"), errors="coerce")
    df["authoritative_bed_support_point_distance_m"] = pd.to_numeric(sampled.get("authoritative_bed_support_point_distance_m"), errors="coerce")
    df["authoritative_bank_margin_point_distance_m"] = pd.to_numeric(sampled.get("authoritative_bank_margin_point_distance_m"), errors="coerce")
    df["authoritative_bed_core_point_distance_m"] = pd.to_numeric(sampled.get("authoritative_bed_core_point_distance_m"), errors="coerce")
    df["authoritative_bed_support_point_z_m"] = pd.to_numeric(sampled.get("authoritative_bed_support_point_z_m"), errors="coerce")
    df["authoritative_bed_core_point_z_m"] = pd.to_numeric(sampled.get("authoritative_bed_core_point_z_m"), errors="coerce")
    df["wse_elevation_m"] = pd.to_numeric(sampled.get("wse_elevation_m"), errors="coerce")
    df["wse_influence"] = pd.to_numeric(sampled.get("wse_influence"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    if np.any(np.isfinite(df["wse_elevation_m"].to_numpy(dtype=float))) and not np.any(df["wse_influence"].to_numpy(dtype=float) > 0.0):
        df.loc[np.isfinite(df["wse_elevation_m"]), "wse_influence"] = 1.0
    df["wse_uncertainty_m"] = pd.to_numeric(sampled.get("wse_uncertainty_m"), errors="coerce")

    rows = []
    for profile_id, grp in df.groupby("profile_id", dropna=False):
        grp = grp[np.isfinite(grp["station_m"].to_numpy(dtype=float))].copy()
        if grp.empty:
            continue
        step = _robust_station_step(grp["station_m"].to_numpy(dtype=float))
        grp["station_bin_m"] = np.round(grp["station_m"].to_numpy(dtype=float) / step) * step
        grp = grp.sort_values("station_bin_m")
        for station_bin, sg in grp.groupby("station_bin_m", sort=True):
            center = sg["centerline_elevation_m"].to_numpy(dtype=float)
            xs = sg["xs_support_elevation_m"].to_numpy(dtype=float)
            bank = sg["bank_elevation_m"].to_numpy(dtype=float)
            center_w = np.clip(sg["centerline_influence"].to_numpy(dtype=float), 0.0, 1.0)
            if not np.any(center_w > 0.0):
                center_w = np.where(np.isfinite(center), 1.0, 0.0)
            center_source_series = sg.get("centerline_sample_source", pd.Series(["missing"] * len(sg), index=sg.index)).fillna("missing").astype(str)
            center_auth_applied = sg.get("centerline_authoritative_support_applied", pd.Series(False, index=sg.index)).fillna(False).to_numpy(dtype=bool)
            center_auth_station = bool(np.any(center_auth_applied & np.isfinite(center)))
            if center_auth_station:
                center_w = np.where(center_auth_applied & np.isfinite(center), np.clip(center_w * 1.5, 0.0, 1.0), center_w)
            center_source_counts = center_source_series[center_source_series.astype(str).str.len() > 0].value_counts(dropna=False).to_dict()
            center_station_source = "authoritative_support_point" if center_auth_station else (str(max(center_source_counts, key=center_source_counts.get)) if center_source_counts else "missing")
            xs_w = np.clip(sg["xs_support_weight"].to_numpy(dtype=float), 0.0, 1.0)
            bank_w = np.clip(sg["bank_weight"].to_numpy(dtype=float), 0.0, 1.0)
            authoritative_anchor = sg["authoritative_anchor_elevation_m"].to_numpy(dtype=float)
            authoritative_support_mask = sg["authoritative_support_mask"].to_numpy(dtype=float)
            exact_authoritative_present = np.any(np.isfinite(authoritative_anchor) & (authoritative_support_mask > 0.0))
            exact_authoritative_anchor_value = float(np.nanmedian(authoritative_anchor[np.isfinite(authoritative_anchor) & (authoritative_support_mask > 0.0)])) if exact_authoritative_present else np.nan
            role_codes = pd.to_numeric(sg.get("authoritative_role_code"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            role_codes = np.clip(np.rint(role_codes), 0, 255).astype(np.uint8)
            point_role_codes = pd.to_numeric(sg.get("authoritative_support_point_role_code"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            point_role_codes = np.clip(np.rint(point_role_codes), 0, 255).astype(np.uint8)
            support_point_distance = pd.to_numeric(sg.get("authoritative_support_point_distance_m"), errors="coerce").to_numpy(dtype=float)
            support_local_radius = pd.to_numeric(sg.get("authoritative_support_local_radius_m"), errors="coerce").to_numpy(dtype=float)
            support_point_z = pd.to_numeric(sg.get("authoritative_support_point_z_m"), errors="coerce").to_numpy(dtype=float)
            support_point_inside_channel = pd.to_numeric(sg.get("authoritative_support_point_inside_channel_mask"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            support_point_inside_guidance = pd.to_numeric(sg.get("authoritative_support_point_inside_river_guidance_domain"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            support_point_inside_estuary = pd.to_numeric(sg.get("authoritative_support_point_inside_estuary_clip"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            local_support = np.isfinite(support_point_distance) & np.isfinite(support_local_radius) & (support_point_distance <= support_local_radius)
            local_role_codes = point_role_codes[local_support]
            active_role_codes = role_codes[(authoritative_support_mask > 0.0) & np.isfinite(authoritative_anchor)] if exact_authoritative_present else np.array([], dtype=np.uint8)
            support_role_codes = local_role_codes if local_role_codes.size else active_role_codes
            authoritative_role, bed_support_present, bank_margin_present, bed_core_present, ambiguous_present = _role_summary_from_codes(support_role_codes)
            role_confidence_vals = pd.to_numeric(sg.get("authoritative_role_confidence"), errors="coerce").to_numpy(dtype=float)
            distance_to_bank_vals = pd.to_numeric(sg.get("authoritative_distance_to_bank_m"), errors="coerce").to_numpy(dtype=float)
            norm_pos_vals = pd.to_numeric(sg.get("authoritative_normalized_channel_position"), errors="coerce").to_numpy(dtype=float)
            point_role_confidence_vals = pd.to_numeric(sg.get("authoritative_support_point_role_confidence"), errors="coerce").to_numpy(dtype=float)
            point_distance_to_bank_vals = pd.to_numeric(sg.get("authoritative_support_point_distance_to_bank_m"), errors="coerce").to_numpy(dtype=float)
            point_norm_pos_vals = pd.to_numeric(sg.get("authoritative_support_point_normalized_channel_position"), errors="coerce").to_numpy(dtype=float)
            bed_support_distance_vals = pd.to_numeric(sg.get("authoritative_bed_support_point_distance_m"), errors="coerce").to_numpy(dtype=float)
            bank_margin_distance_vals = pd.to_numeric(sg.get("authoritative_bank_margin_point_distance_m"), errors="coerce").to_numpy(dtype=float)
            bed_core_distance_vals = pd.to_numeric(sg.get("authoritative_bed_core_point_distance_m"), errors="coerce").to_numpy(dtype=float)
            point_bed_support_z_vals = pd.to_numeric(sg.get("authoritative_bed_support_point_z_m"), errors="coerce").to_numpy(dtype=float)
            point_bed_core_z_vals = pd.to_numeric(sg.get("authoritative_bed_core_point_z_m"), errors="coerce").to_numpy(dtype=float)
            station_local_radius = float(np.nanmedian(support_local_radius[np.isfinite(support_local_radius)])) if np.any(np.isfinite(support_local_radius)) else 5.0
            authoritative_bed_support_distance = float(np.nanmin(bed_support_distance_vals)) if np.any(np.isfinite(bed_support_distance_vals)) else np.nan
            authoritative_bank_margin_distance = float(np.nanmin(bank_margin_distance_vals)) if np.any(np.isfinite(bank_margin_distance_vals)) else np.nan
            authoritative_bed_core_distance = float(np.nanmin(bed_core_distance_vals)) if np.any(np.isfinite(bed_core_distance_vals)) else np.nan
            if np.isfinite(authoritative_bed_support_distance):
                bed_support_present = authoritative_bed_support_distance <= station_local_radius
            if np.isfinite(authoritative_bank_margin_distance):
                bank_margin_present = authoritative_bank_margin_distance <= station_local_radius and not bed_support_present
            if np.isfinite(authoritative_bed_core_distance):
                bed_core_present = authoritative_bed_core_distance <= station_local_radius
            if local_role_codes.size:
                ambiguous_present = False
            point_bed_anchor_mask = (
                np.isfinite(support_point_z)
                & local_support
                & np.isin(point_role_codes, [role_to_code(ROLE_BED_INNER), role_to_code(ROLE_BED_CORE)])
                & (support_point_inside_channel > 0.5)
                & (support_point_inside_guidance > 0.5)
                & (support_point_inside_estuary < 0.5)
            )
            point_bed_core_anchor_mask = point_bed_anchor_mask & (point_role_codes == role_to_code(ROLE_BED_CORE))
            authoritative_present = exact_authoritative_present or bool(np.any(point_bed_anchor_mask))
            if exact_authoritative_present:
                authoritative_anchor_value = exact_authoritative_anchor_value
                authoritative_anchor_source = "authoritative_anchor_raster"
            elif np.any(point_bed_core_anchor_mask):
                authoritative_anchor_value = float(np.nanmedian(support_point_z[point_bed_core_anchor_mask]))
                authoritative_anchor_source = "authoritative_bed_core_support_point"
            elif np.any(point_bed_anchor_mask):
                authoritative_anchor_value = float(np.nanmedian(support_point_z[point_bed_anchor_mask]))
                authoritative_anchor_source = "authoritative_bed_support_point"
            else:
                authoritative_anchor_value = np.nan
                authoritative_anchor_source = "none"
            if local_role_codes.size and np.any(np.isfinite(point_role_confidence_vals[local_support])):
                authoritative_role_confidence = float(np.nanmedian(point_role_confidence_vals[local_support]))
            else:
                authoritative_role_confidence = float(np.nanmedian(role_confidence_vals[(authoritative_support_mask > 0.0) & np.isfinite(role_confidence_vals)])) if exact_authoritative_present and np.any(np.isfinite(role_confidence_vals)) else np.nan
            if local_role_codes.size and np.any(np.isfinite(point_distance_to_bank_vals[local_support])):
                authoritative_distance_to_bank = float(np.nanmedian(point_distance_to_bank_vals[local_support]))
            else:
                authoritative_distance_to_bank = float(np.nanmedian(distance_to_bank_vals[(authoritative_support_mask > 0.0) & np.isfinite(distance_to_bank_vals)])) if exact_authoritative_present and np.any(np.isfinite(distance_to_bank_vals)) else np.nan
            if local_role_codes.size and np.any(np.isfinite(point_norm_pos_vals[local_support])):
                authoritative_normalized_channel_position = float(np.nanmedian(point_norm_pos_vals[local_support]))
            else:
                authoritative_normalized_channel_position = float(np.nanmedian(norm_pos_vals[(authoritative_support_mask > 0.0) & np.isfinite(norm_pos_vals)])) if exact_authoritative_present and np.any(np.isfinite(norm_pos_vals)) else np.nan
            nearest_support_inside_estuary_clip = float(np.nanmax(support_point_inside_estuary[local_support])) if np.any(local_support) else 0.0
            wse_vals = pd.to_numeric(sg["wse_elevation_m"], errors="coerce").to_numpy(dtype=float)
            wse_influence = pd.to_numeric(sg["wse_influence"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            wse_unc = pd.to_numeric(sg["wse_uncertainty_m"], errors="coerce").to_numpy(dtype=float)
            wse_present = np.any(np.isfinite(wse_vals) & (wse_influence > 0.0))
            wse_value = _weighted_mean(wse_vals, np.where(np.isfinite(wse_vals), np.maximum(wse_influence, 1e-6), 0.0)) if wse_present else np.nan
            if wse_present and np.any(np.isfinite(wse_unc)):
                wse_uncertainty = _weighted_mean(wse_unc, np.where(np.isfinite(wse_unc) & (wse_influence > 0.0), np.maximum(wse_influence, 1e-6), 0.0))
            elif wse_present:
                wse_uncertainty = float(max(0.10, 0.30 - 0.20 * float(np.nanmax(wse_influence))))
            else:
                wse_uncertainty = np.nan

            center_value = float(np.nanmedian(center)) if np.any(np.isfinite(center)) else np.nan
            # Use a thalweg-like XS metric instead of a section median so lateral bank
            # elevations do not raise the longitudinal bed profile in unsupported reaches.
            xs_thalweg_value = float(np.nanmin(xs)) if np.any(np.isfinite(xs)) else np.nan
            bank_value = float(np.nanquantile(bank[np.isfinite(bank)], 0.25)) if np.any(np.isfinite(bank)) else np.nan

            comp_vals = np.array([
                authoritative_anchor_value,
                center_value,
                xs_thalweg_value,
                bank_value,
            ], dtype=float)
            comp_w = np.array([
                1.0 if authoritative_present else 0.0,
                float(np.nanmax(center_w)) if np.any(np.isfinite(center)) else 0.0,
                float(np.nanmax(xs_w)) if np.any(np.isfinite(xs)) else 0.0,
                float(np.nanmax(bank_w)) if np.any(np.isfinite(bank)) else 0.0,
            ], dtype=float)
            bed_comp_vals = np.array([
                authoritative_anchor_value,
                center_value,
                xs_thalweg_value,
            ], dtype=float)
            bed_comp_w = np.array([
                comp_w[0],
                comp_w[1],
                comp_w[2],
            ], dtype=float)
            if authoritative_present:
                bed = authoritative_anchor_value
                spread = 0.0
                dominant_source = "authoritative_anchor"
            else:
                bed = _weighted_mean(bed_comp_vals, bed_comp_w)
                spread = _weighted_spread(comp_vals, np.where(np.isfinite(comp_vals), np.maximum(comp_w, 1e-6), 0.0), bed)
                dominant_source = ["authoritative_anchor", "centerline", "xs_support"][int(np.nanargmax(bed_comp_w))] if np.any(bed_comp_w > 0.0) else "none"
            rows.append({
                "profile_id": str(profile_id),
                "station_m": float(station_bin),
                "station_step_m": float(step),
                "authoritative_anchor_elevation_m": authoritative_anchor_value,
                "authoritative_anchor_present": bool(authoritative_present),
                "authoritative_anchor_source": authoritative_anchor_source,
                "centerline_elevation_m": center_value,
                "xs_support_elevation_m": xs_thalweg_value,
                "bank_elevation_m": bank_value,
                "authoritative_support_depth_m": float(np.nanmedian(sg["authoritative_support_depth_m"].to_numpy(dtype=float))) if np.any(np.isfinite(sg["authoritative_support_depth_m"].to_numpy(dtype=float))) else np.nan,
                "profile_authoritative_role": authoritative_role,
                "profile_authoritative_role_code": int(np.max(support_role_codes)) if support_role_codes.size else 0,
                "profile_authoritative_role_confidence": authoritative_role_confidence,
                "profile_authoritative_distance_to_bank_m": authoritative_distance_to_bank,
                "profile_authoritative_normalized_channel_position": authoritative_normalized_channel_position,
                "profile_authoritative_bed_support_present": bool(bed_support_present),
                "profile_authoritative_bank_margin_present": bool(bank_margin_present),
                "profile_authoritative_bed_core_present": bool(bed_core_present),
                "profile_authoritative_ambiguous_present": bool(ambiguous_present),
                "profile_authoritative_bed_support_distance_m": authoritative_bed_support_distance,
                "profile_authoritative_bank_margin_distance_m": authoritative_bank_margin_distance,
                "profile_authoritative_station_local_support_radius_m": station_local_radius,
                "profile_nearest_support_inside_estuary_clip": nearest_support_inside_estuary_clip,
                "wse_elevation_m": wse_value,
                "wse_influence": float(np.nanmax(wse_influence)) if wse_present else 0.0,
                "wse_uncertainty_m": wse_uncertainty,
                "centerline_sample_source": center_station_source,
                "centerline_authoritative_support_station_present": bool(center_auth_station),
                "centerline_weight": comp_w[1],
                "xs_support_weight": comp_w[2],
                "bank_weight": comp_w[3],
                "bank_estuary_side_decay_median": float(np.nanmedian(pd.to_numeric(sg.get("bank_estuary_side_decay"), errors="coerce").to_numpy(dtype=float))) if "bank_estuary_side_decay" in sg.columns and np.any(np.isfinite(pd.to_numeric(sg.get("bank_estuary_side_decay"), errors="coerce").to_numpy(dtype=float))) else 1.0,
                "authoritative_anchor_weight": comp_w[0],
                "bed_elevation_m": bed,
                "component_spread_m": float(spread),
                "n_points": int(len(sg)),
                "dominant_source": dominant_source,
            })
    profile = pd.DataFrame(rows)
    if profile.empty:
        return profile
    profile = profile.sort_values(["profile_id", "station_m"]).reset_index(drop=True)
    fluvial_mask, fluvial_score, fluvial_source = _resolve_profile_fluvial_monotone_domain(profile)
    profile["profile_inside_fluvial_monotone_domain"] = fluvial_mask
    profile["profile_fluvial_monotone_score"] = fluvial_score
    profile["profile_fluvial_monotone_domain_source"] = fluvial_source
    profile["longitudinal_roughness_m"] = 0.0
    for _, idx in profile.groupby("profile_id").groups.items():
        sub = profile.loc[idx].sort_values("station_m")
        vals = sub["bed_elevation_m"].to_numpy(dtype=float)
        rough = np.zeros(len(sub), dtype=float)
        if len(sub) > 1:
            dif = np.abs(np.diff(vals))
            rough[1:] = np.maximum(rough[1:], dif)
            rough[:-1] = np.maximum(rough[:-1], dif)
            rough *= 0.5
        profile.loc[sub.index, "longitudinal_roughness_m"] = rough
    profile["uncertainty_m"] = np.sqrt(
        np.square(np.nan_to_num(profile["component_spread_m"].to_numpy(dtype=float), nan=0.0))
        + np.square(np.nan_to_num(profile["longitudinal_roughness_m"].to_numpy(dtype=float), nan=0.0))
    )
    profile["uncertainty_m"] = np.maximum(profile["uncertainty_m"].to_numpy(dtype=float), 0.05)
    if "authoritative_anchor_present" in profile.columns:
        auth_mask = profile["authoritative_anchor_present"].fillna(False).to_numpy(dtype=bool)
        profile.loc[auth_mask, "uncertainty_m"] = 0.02
    profile["wse_local_slope_mpm"] = np.nan
    if "wse_elevation_m" in profile.columns:
        for _, idx in profile.groupby("profile_id").groups.items():
            sub = profile.loc[idx].sort_values("station_m")
            grads = _compute_local_gradient(
                pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(sub["wse_elevation_m"], errors="coerce").to_numpy(dtype=float),
            )
            profile.loc[sub.index, "wse_local_slope_mpm"] = grads
    return profile


def _read_network_edges(network_edges_path: Optional[str | Path]):
    if network_edges_path is None:
        return None
    path = Path(network_edges_path)
    if not path.exists():
        return None
    try:
        import geopandas as gpd
        return gpd.read_file(path)
    except Exception:
        log.debug("_read_network_edges: suppressed exception", exc_info=True)
        return None


def _component_endpoint_metadata(edges) -> Dict[str, Dict[str, object]]:
    if edges is None or len(edges) == 0 or "component_id" not in edges.columns:
        return {}
    work = edges.copy()
    for col in ("from_node", "to_node", "s_m_from", "s_m_to", "s_m_min", "s_m_max"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    meta: Dict[str, Dict[str, object]] = {}
    for comp_id, grp in work.groupby("component_id", dropna=False):
        grp = grp.copy()
        candidates = []
        for _, row in grp.iterrows():
            s0 = row.get("s_m_from", np.nan)
            s1 = row.get("s_m_to", np.nan)
            fn = row.get("from_node", np.nan)
            tn = row.get("to_node", np.nan)
            if np.isfinite(s0) and np.isfinite(fn):
                candidates.append((float(s0), int(fn)))
            if np.isfinite(s1) and np.isfinite(tn):
                candidates.append((float(s1), int(tn)))
            smin = row.get("s_m_min", np.nan)
            smax = row.get("s_m_max", np.nan)
            if np.isfinite(smin) and np.isfinite(fn):
                candidates.append((float(smin), int(fn)))
            if np.isfinite(smax) and np.isfinite(tn):
                candidates.append((float(smax), int(tn)))
        if not candidates:
            continue
        comp = str(comp_id)
        start_station, start_node = min(candidates, key=lambda x: x[0])
        end_station, end_node = max(candidates, key=lambda x: x[0])
        station_delta = pd.to_numeric(grp.get("s_m_to"), errors="coerce") - pd.to_numeric(grp.get("s_m_from"), errors="coerce")
        direction = "increasing_station_downstream"
        if np.isfinite(station_delta.to_numpy(dtype=float)).any():
            direction = "increasing_station_downstream" if float(np.nanmedian(station_delta.to_numpy(dtype=float))) >= 0.0 else "decreasing_station_downstream"
        meta[comp] = {
            "start_node": int(start_node),
            "end_node": int(end_node),
            "start_station_m": float(start_station),
            "end_station_m": float(end_station),
            "station_direction": direction,
        }
    return meta


def _adjacency_from_endpoint_metadata(endpoint_meta: Dict[str, Dict[str, object]]) -> Dict[int, list[Tuple[str, str]]]:
    adjacency: Dict[int, list[Tuple[str, str]]] = {}
    for comp, meta in endpoint_meta.items():
        for pos in ("start", "end"):
            node = meta.get(f"{pos}_node")
            if node is None:
                continue
            adjacency.setdefault(int(node), []).append((str(comp), pos))
    return adjacency




def _series_to_float_array(df: pd.DataFrame, field: str) -> np.ndarray:
    if field in df.columns:
        return pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), np.nan, dtype=float)

def _classify_component_support_regime(sub: pd.DataFrame) -> Dict[str, float | str | bool]:
    stations = pd.to_numeric(sub.get("station_m"), errors="coerce").to_numpy(dtype=float)
    finite_stations = stations[np.isfinite(stations)]
    component_length_m = float(np.nanmax(finite_stations) - np.nanmin(finite_stations)) if finite_stations.size >= 2 else 0.0
    bed_support_present = sub.get("profile_authoritative_bed_support_present", pd.Series(False, index=sub.index)).fillna(False).to_numpy(dtype=bool)
    anchor_present = sub.get("authoritative_anchor_present", pd.Series(False, index=sub.index)).fillna(False).to_numpy(dtype=bool)
    support_distance = _support_distance_array(sub)
    finite_support_distance = support_distance[np.isfinite(support_distance)]
    bed_support_fraction = float(np.mean(bed_support_present)) if bed_support_present.size else 0.0
    anchor_fraction = float(np.mean(anchor_present)) if anchor_present.size else 0.0
    anchor_count = int(np.count_nonzero(anchor_present))
    median_support_distance_m = float(np.median(finite_support_distance)) if finite_support_distance.size else float("nan")
    hierarchy_weight = _series_to_float_array(sub, "junction_hierarchy_weight")
    stream_order = _series_to_float_array(sub, "stream_order_proxy")
    drainage = _series_to_float_array(sub, "drainage_area_proxy")
    hierarchy_score = float(np.nanmax(hierarchy_weight)) if np.any(np.isfinite(hierarchy_weight)) else 1.0
    stream_order_score = float(np.nanmax(stream_order)) if np.any(np.isfinite(stream_order)) else float("nan")
    drainage_score = float(np.nanmax(drainage)) if np.any(np.isfinite(drainage)) else float("nan")
    is_mainstem = bool(
        (np.isfinite(stream_order_score) and stream_order_score >= 3.0)
        or (np.isfinite(drainage_score) and drainage_score >= 25.0)
        or hierarchy_score >= 1.75
        or component_length_m >= 1200.0
    )
    support_weak = (bed_support_fraction < 0.05 and anchor_count == 0) or (not np.isfinite(median_support_distance_m)) or (median_support_distance_m >= 450.0)
    weakly_supported = (bed_support_fraction < 0.15 and anchor_fraction < 0.10) or (np.isfinite(median_support_distance_m) and median_support_distance_m >= 250.0)
    if component_length_m < 180.0 and anchor_count == 0 and bed_support_fraction <= 0.0:
        component_class = "tiny_detached_component"
    elif is_mainstem and support_weak:
        component_class = "unsupported_mainstem"
    elif (not is_mainstem) and support_weak:
        component_class = "unsupported_side_component"
    elif is_mainstem and weakly_supported:
        component_class = "weakly_supported_mainstem"
    else:
        component_class = "anchored_mainstem" if is_mainstem else "unsupported_side_component"
    return {
        "component_class": component_class,
        "component_length_m": component_length_m,
        "is_mainstem": bool(is_mainstem),
        "bed_support_fraction": bed_support_fraction,
        "anchor_fraction": anchor_fraction,
        "anchor_count": anchor_count,
        "median_support_distance_m": median_support_distance_m,
    }


def _solve_component_profile(sub: pd.DataFrame, *, lam_data: float = 1.0, lam_smooth: float = 0.40) -> Tuple[np.ndarray, np.ndarray, Dict[str, float | str | bool]]:
    vals = pd.to_numeric(sub["bed_elevation_m"], errors="coerce").to_numpy(dtype=float)
    unc = pd.to_numeric(sub["uncertainty_m"], errors="coerce").to_numpy(dtype=float)
    authoritative_anchor = pd.to_numeric(sub.get("authoritative_anchor_elevation_m"), errors="coerce").to_numpy(dtype=float) if "authoritative_anchor_elevation_m" in sub.columns else np.full(len(sub), np.nan, dtype=float)
    authoritative_present = sub.get("authoritative_anchor_present", pd.Series(False, index=sub.index)).fillna(False).to_numpy(dtype=bool) if "authoritative_anchor_present" in sub.columns else np.zeros(len(sub), dtype=bool)
    n = len(vals)
    if n == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float), {"component_class": "empty"}
    obs = vals.copy()
    component_meta = _classify_component_support_regime(sub)
    component_class = str(component_meta.get("component_class", "anchored_mainstem"))
    if not np.any(np.isfinite(obs)):
        return np.full(n, np.nan, dtype=float), np.full(n, np.nan, dtype=float), component_meta
    mask = np.isfinite(obs)
    fill = np.interp(np.arange(n), np.where(mask)[0], obs[mask]) if np.sum(mask) >= 2 else np.where(mask, obs, np.nanmedian(obs[mask]))
    weights = np.where(np.isfinite(unc) & (unc > 0.0), 1.0 / np.square(np.maximum(unc, 0.05)), 0.0)
    weights = np.where(mask, np.maximum(weights, 0.25), 0.0)
    anchor_mask = authoritative_present & np.isfinite(authoritative_anchor)
    fill = np.where(anchor_mask, authoritative_anchor, fill)
    weights = np.where(anchor_mask, np.maximum(weights, 2500.0), weights)

    # ---- Per-station adaptive smoothing: much stronger far from authoritative support ----
    support_distance = _support_distance_array(sub)
    if not np.any(np.isfinite(support_distance)):
        anchor_indices = np.flatnonzero(anchor_mask)
        if anchor_indices.size > 0:
            idx_dist = np.array([float(np.min(np.abs(anchor_indices - i))) for i in range(n)], dtype=float)
            support_distance = idx_dist * max(_robust_station_step(pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float)), 1.0)
        else:
            support_distance = np.full(n, np.nan, dtype=float)
    support_far_frac = np.clip((support_distance - 80.0) / 900.0, 0.0, 1.0)
    support_far_frac = np.where(np.isfinite(support_far_frac), support_far_frac, 1.0)
    support_far_frac = np.where(anchor_mask, 0.0, support_far_frac)
    class_smooth_multiplier = 1.0
    class_data_floor = 0.10
    near_distance_m = 120.0
    far_distance_m = 1000.0
    moderate_window_m = 180.0
    strong_window_m = 480.0
    if component_class == "weakly_supported_mainstem":
        class_smooth_multiplier = 1.35
        class_data_floor = 0.08
        near_distance_m = 100.0
        far_distance_m = 900.0
        moderate_window_m = 240.0
        strong_window_m = 620.0
    elif component_class == "unsupported_mainstem":
        class_smooth_multiplier = 1.75
        class_data_floor = 0.06
        near_distance_m = 80.0
        far_distance_m = 800.0
        moderate_window_m = 300.0
        strong_window_m = 760.0
    elif component_class == "unsupported_side_component":
        class_smooth_multiplier = 2.15
        class_data_floor = 0.05
        near_distance_m = 60.0
        far_distance_m = 650.0
        moderate_window_m = 360.0
        strong_window_m = 920.0
    elif component_class == "tiny_detached_component":
        class_smooth_multiplier = 2.75
        class_data_floor = 0.03
        near_distance_m = 40.0
        far_distance_m = 500.0
        moderate_window_m = 420.0
        strong_window_m = 1100.0
    smooth_scale = (1.0 + 10.0 * support_far_frac) * class_smooth_multiplier
    # Also downweight data fidelity for unsupported stations — the DEM-sampled
    # "observations" in unsupported reaches are not real measurements.
    data_weight_scale = np.where(anchor_mask, 1.0, np.clip(1.0 - 0.85 * support_far_frac, class_data_floor, 1.0))
    effective_weights = weights * data_weight_scale

    A = np.diag(effective_weights * lam_data + 1e-6)
    b = (effective_weights * lam_data) * fill
    if n > 1:
        for i in range(n - 1):
            # Use the max of the two adjacent smooth_scale values so the
            # stronger-smoothing station dominates the pair.
            local_smooth = lam_smooth * max(smooth_scale[i], smooth_scale[i + 1])
            A[i, i] += local_smooth
            A[i + 1, i + 1] += local_smooth
            A[i, i + 1] -= local_smooth
            A[i + 1, i] -= local_smooth
    try:
        solved = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        solved = fill

    # ---- Monotonic tendency enforcement ----
    delta = np.nanmedian(np.diff(fill)) if n > 1 else 0.0
    # Tighter tolerance far from support (0.005 m) vs near support (0.02 m).
    local_tol_arr = np.where(support_far_frac > 0.45, 0.005, 0.02)
    if np.isfinite(delta) and delta <= 0.0:
        for i in range(1, n):
            local_tol = float(local_tol_arr[i])
            solved[i] = min(solved[i], solved[i - 1] + local_tol)
    else:
        for i in range(1, n):
            local_tol = float(local_tol_arr[i])
            solved[i] = max(solved[i], solved[i - 1] - local_tol)
    solved = np.where(anchor_mask, authoritative_anchor, solved)

    # ---- Post-solve Gaussian low-pass for unsupported reaches ----
    # Apply only to stations far from anchors (idx_dist > 3), preserving
    # anchor values exactly.  sigma_stations = 4 gives ~400 m smoothing at
    # 100 m station spacing — enough to suppress DEM noise oscillations
    # while preserving reach-scale trends.
    if n >= 7:
        solved = _distance_aware_profile_smoothing(
            solved,
            pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float),
            support_distance,
            anchor_mask=anchor_mask,
            downstream_increasing=(np.isfinite(delta) and delta <= 0.0),
            near_distance_m=near_distance_m,
            far_distance_m=far_distance_m,
            moderate_window_m=moderate_window_m,
            strong_window_m=strong_window_m,
        )
        solved = np.where(anchor_mask, authoritative_anchor, solved)

    residual = np.abs(np.nan_to_num(solved - fill, nan=0.0))
    solved_unc = np.sqrt(np.square(np.nan_to_num(unc, nan=np.nanmedian(np.nan_to_num(unc, nan=0.25)))) + np.square(residual))
    solved_unc = np.maximum(solved_unc, 0.05)
    solved_unc = np.where(anchor_mask, 0.02, solved_unc)
    finite_step = np.abs(np.diff(solved[np.isfinite(solved)])) if np.count_nonzero(np.isfinite(solved)) >= 2 else np.array([], dtype=float)
    finite_curv = np.abs(np.diff(np.diff(solved[np.isfinite(solved)]))) if np.count_nonzero(np.isfinite(solved)) >= 3 else np.array([], dtype=float)
    component_meta = dict(component_meta)
    component_meta.update({
        "unsupported_step_p95": float(np.percentile(finite_step, 95.0)) if finite_step.size else 0.0,
        "unsupported_curvature_p95": float(np.percentile(finite_curv, 95.0)) if finite_curv.size else 0.0,
        "smoothing_regime_used": component_class,
        "far_support_smoothing_applied": bool(class_smooth_multiplier > 1.0 or np.any(support_far_frac > 0.35)),
    })
    return solved.astype(float), solved_unc.astype(float), component_meta


def _apply_network_continuity(
    profile: pd.DataFrame,
    endpoint_meta: Dict[str, Dict[str, object]],
    hierarchy_meta: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    if profile.empty or not endpoint_meta:
        profile["network_backbone_elevation_m"] = pd.to_numeric(profile.get("bed_elevation_m"), errors="coerce")
        profile["network_backbone_uncertainty_m"] = pd.to_numeric(profile.get("uncertainty_m"), errors="coerce")
        profile["network_junction_adjustment_m"] = 0.0
        profile["network_backbone_source"] = "local_profile"
        profile["junction_hierarchy_weight"] = 1.0
        profile["drainage_area_proxy"] = np.nan
        profile["stream_order_proxy"] = np.nan
        profile["junction_wse_weight"] = 1.0
        profile["wse_endpoint_consensus_residual_m"] = np.nan
        return profile

    hierarchy_meta = hierarchy_meta or {}
    solved_parts = []
    for comp, sub in profile.groupby("profile_id", sort=False):
        sub = sub.sort_values("station_m").copy()
        solved, solved_unc, support_meta = _solve_component_profile(sub)
        comp_meta = hierarchy_meta.get(str(comp), {})
        hierarchy_priority = float(comp_meta.get("hierarchy_priority", 1.0))
        sub["drainage_area_proxy"] = float(comp_meta.get("drainage_area_proxy", np.nan))
        sub["stream_order_proxy"] = float(comp_meta.get("stream_order_proxy", np.nan))
        sub["junction_hierarchy_weight"] = hierarchy_priority
        sub["component_support_class"] = str(support_meta.get("component_class", "anchored_mainstem"))
        sub["component_support_is_mainstem"] = bool(support_meta.get("is_mainstem", True))
        sub["component_support_length_m"] = float(support_meta.get("component_length_m", np.nan))
        sub["component_support_bed_fraction"] = float(support_meta.get("bed_support_fraction", np.nan))
        sub["component_support_anchor_fraction"] = float(support_meta.get("anchor_fraction", np.nan))
        sub["component_support_anchor_count"] = int(support_meta.get("anchor_count", 0) or 0)
        sub["component_support_median_distance_m"] = float(support_meta.get("median_support_distance_m", np.nan))
        sub["component_support_step_p95_m"] = float(support_meta.get("unsupported_step_p95", 0.0))
        sub["component_support_curvature_p95_m"] = float(support_meta.get("unsupported_curvature_p95", 0.0))
        sub["component_support_smoothing_regime"] = str(support_meta.get("smoothing_regime_used", support_meta.get("component_class", "anchored_mainstem")))
        sub["component_support_far_smoothing_applied"] = bool(support_meta.get("far_support_smoothing_applied", False))
        sub["network_backbone_elevation_m"] = solved
        sub["network_backbone_uncertainty_m"] = solved_unc
        sub["network_junction_adjustment_m"] = 0.0
        sub["network_backbone_source"] = "network_component_solve"
        sub["junction_wse_weight"] = 1.0
        sub["wse_endpoint_consensus_residual_m"] = np.nan
        solved_parts.append(sub)
    out = pd.concat(solved_parts, ignore_index=True)

    adjacency = _adjacency_from_endpoint_metadata(endpoint_meta)
    decay_stations = 4.0
    for _ in range(3):
        for node, members in adjacency.items():
            if len(members) < 2:
                continue
            endpoint_rows = []
            for comp, pos in members:
                sub = out.loc[out["profile_id"].astype(str) == str(comp)].sort_values("station_m")
                if sub.empty:
                    continue
                endpoint_rows.append((comp, pos, sub.index[0] if pos == "start" else sub.index[-1]))
            if len(endpoint_rows) < 2:
                continue
            vals = np.array([float(out.at[idx, "network_backbone_elevation_m"]) for _, _, idx in endpoint_rows], dtype=float)
            uncs = np.array([float(out.at[idx, "network_backbone_uncertainty_m"]) for _, _, idx in endpoint_rows], dtype=float)
            priorities = np.array([float(hierarchy_meta.get(str(comp), {}).get("hierarchy_priority", 1.0)) for comp, _, _ in endpoint_rows], dtype=float)
            priorities = np.where(np.isfinite(priorities) & (priorities > 0.0), priorities, 1.0)
            anchor_strength = np.array([
                8.0 if bool(out.at[idx, "authoritative_anchor_present"]) else 1.0
                for _, _, idx in endpoint_rows
            ], dtype=float) if "authoritative_anchor_present" in out.columns else np.ones(len(endpoint_rows), dtype=float)
            wse_vals = np.array([float(out.at[idx, "wse_elevation_m"]) if "wse_elevation_m" in out.columns else np.nan for _, _, idx in endpoint_rows], dtype=float)
            wse_uncs = np.array([float(out.at[idx, "wse_uncertainty_m"]) if "wse_uncertainty_m" in out.columns else np.nan for _, _, idx in endpoint_rows], dtype=float)
            wse_influence = np.array([float(out.at[idx, "wse_influence"]) if "wse_influence" in out.columns else 0.0 for _, _, idx in endpoint_rows], dtype=float)
            valid = np.isfinite(vals)
            if np.sum(valid) < 2:
                continue
            base_weights = np.where(np.isfinite(uncs) & (uncs > 0.0), 1.0 / np.square(np.maximum(uncs, 0.05)), 1.0)
            wse_valid = np.isfinite(wse_vals) & (wse_influence > 0.0)
            wse_weight = np.ones(len(endpoint_rows), dtype=float)
            wse_consensus = np.nan
            if np.sum(wse_valid) >= 2:
                wse_var = np.where(np.isfinite(wse_uncs) & (wse_uncs > 0.0), np.square(np.maximum(wse_uncs, 0.05)), 0.04)
                wse_base = np.where(wse_valid, np.maximum(wse_influence, 1e-3) / wse_var, 0.0)
                if np.sum(wse_base) > 0.0:
                    wse_consensus = float(np.sum(wse_vals[wse_valid] * wse_base[wse_valid]) / np.sum(wse_base[wse_valid]))
                    resid = np.abs(wse_vals - wse_consensus)
                    resid_scale = np.maximum(np.where(np.isfinite(wse_uncs), wse_uncs, 0.20), 0.10)
                    stage_spread = float(np.nanstd(wse_vals[wse_valid], ddof=0)) if np.sum(wse_valid) >= 2 else 0.0
                    resid_scale = np.maximum(resid_scale, max(stage_spread, 0.25))
                    wse_weight = np.where(wse_valid, np.clip(np.exp(-resid / (2.0 * resid_scale)), 0.10, 1.0), 1.0)
                    for (_, _, idx), ww, rr in zip(endpoint_rows, wse_weight, resid):
                        out.at[idx, "junction_wse_weight"] = float(ww)
                        if np.isfinite(rr):
                            out.at[idx, "wse_endpoint_consensus_residual_m"] = float(rr)
            target_weights = base_weights * priorities * anchor_strength * wse_weight
            target = float(np.sum(vals[valid] * target_weights[valid]) / np.sum(target_weights[valid]))
            max_priority = float(np.nanmax(priorities[valid])) if np.any(valid) else 1.0
            max_wse_weight = float(np.nanmax(wse_weight[valid])) if np.any(valid) else 1.0
            for (comp, pos, idx), local_priority, local_anchor_strength, local_wse_weight in zip(endpoint_rows, priorities, anchor_strength, wse_weight):
                sub = out.loc[out["profile_id"].astype(str) == str(comp)].sort_values("station_m")
                local_val = float(out.at[idx, "network_backbone_elevation_m"])
                if not np.isfinite(local_val):
                    continue
                hierarchy_share = float(local_priority / max(max_priority, 1e-6))
                wse_share = float(local_wse_weight / max(max_wse_weight, 1e-6))
                mobility = float(np.clip(1.0 / max(local_anchor_strength, 1.0), 0.125, 1.0))
                delta = (target - local_val) * mobility
                stations = pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float)
                if stations.size == 0:
                    continue
                if pos == "start":
                    dist = stations - stations[0]
                else:
                    dist = stations[-1] - stations
                step = max(float(np.nanmedian(np.diff(stations))) if stations.size > 1 else 1.0, 1.0)
                decay_len = max(decay_stations * step, step)
                taper = np.exp(-np.clip(dist, 0.0, None) / decay_len)
                strength = 0.65 * np.clip((0.25 + 0.55 * hierarchy_share) * (0.40 + 0.60 * wse_share), 0.20, 1.0)
                adj = strength * delta * taper
                anchor_mask = sub.get("authoritative_anchor_present", pd.Series(False, index=sub.index)).fillna(False).to_numpy(dtype=bool)
                adj = np.where(anchor_mask, 0.0, adj)
                out.loc[sub.index, "network_backbone_elevation_m"] = out.loc[sub.index, "network_backbone_elevation_m"].to_numpy(dtype=float) + adj
                out.loc[sub.index, "network_junction_adjustment_m"] = out.loc[sub.index, "network_junction_adjustment_m"].to_numpy(dtype=float) + adj
                out.loc[idx, "network_backbone_source"] = "network_junction_flow_aware_solve"
    adj_mag = np.abs(pd.to_numeric(out["network_junction_adjustment_m"], errors="coerce").to_numpy(dtype=float))
    out["network_backbone_uncertainty_m"] = np.sqrt(
        np.square(pd.to_numeric(out["network_backbone_uncertainty_m"], errors="coerce").to_numpy(dtype=float)) + np.square(adj_mag)
    )
    return out


def _assign_profile_values_to_points(centerline_points, profile: pd.DataFrame) -> pd.DataFrame:
    pts = centerline_points.copy()
    pts["profile_id"] = _infer_profile_id(pts)
    pts["station_m"] = pd.to_numeric(pts.get("station_m"), errors="coerce")
    pts["longitudinal_profile_elevation_m"] = np.nan
    pts["longitudinal_profile_uncertainty_m"] = np.nan
    pts["network_backbone_elevation_m"] = np.nan
    pts["network_backbone_uncertainty_m"] = np.nan
    pts["network_backbone_source"] = None
    pts["active_core_support_elevation_m"] = np.nan
    pts["active_core_support_uncertainty_m"] = np.nan
    pts["active_core_support_source"] = None
    pts["generalized_longitudinal_bed_source"] = None
    pts["left_bank_fit_z_m"] = np.nan
    pts["right_bank_fit_z_m"] = np.nan
    pts["bank_pair_fit_z_m"] = np.nan
    pts["longitudinal_profile_source"] = None
    pts["generalized_longitudinal_bed_base_elevation_m"] = np.nan
    pts["generalized_longitudinal_bed_reconciled_elevation_m"] = np.nan
    pts["generalized_longitudinal_bed_local_auth_taper_m"] = np.nan
    pts["generalized_longitudinal_bed_local_auth_reconciliation_weight"] = np.nan
    if "centerline_sample_source" not in pts.columns:
        pts["centerline_sample_source"] = None
    if "centerline_authoritative_support_applied" not in pts.columns:
        pts["centerline_authoritative_support_applied"] = False
    if "centerline_authoritative_support_distance_m" not in pts.columns:
        pts["centerline_authoritative_support_distance_m"] = np.nan
    pts["authoritative_reconciliation_delta_m"] = np.nan
    pts["authoritative_reconciliation_weight"] = np.nan
    pts["authoritative_reconciliation_confidence"] = np.nan
    pts["authoritative_reconciliation_support_distance_m"] = np.nan
    pts["authoritative_reconciliation_observation_used"] = False
    pts["authoritative_reconciliation_observed_bed_z_m"] = np.nan
    pts["authoritative_reconciliation_observed_residual_m"] = np.nan
    pts["authoritative_reconciliation_observation_confidence"] = np.nan
    # Carry support/coverage semantics directly onto exported profile points so
    # downstream diagnostics do not depend on fragile float-key merges.
    for col, default in [
        ("profile_support_class", None),
        ("profile_support_source_count", np.nan),
        ("profile_support_present", False),
        ("profile_authoritative_anchor_present", False),
        ("profile_xs_support_present", False),
        ("profile_centerline_support_present", False),
        ("profile_bank_support_present", False),
        ("profile_wse_support_present", False),
        ("profile_measured_support_distance_m", np.nan),
        ("profile_far_from_measured_support", False),
        ("profile_authoritative_role", None),
        ("profile_authoritative_bed_support_present", False),
        ("profile_authoritative_bank_margin_present", False),
        ("profile_authoritative_bed_support_distance_m", np.nan),
        ("profile_far_from_authoritative_bed_support", False),
        ("centerline_authoritative_support_station_present", False),
    ]:
        pts[col] = default
    if profile.empty:
        return pts

    def _nearest_lookup(values: np.ndarray, ref_values: np.ndarray) -> np.ndarray:
        ref_values = np.asarray(ref_values)
        if ref_values.size == 0:
            return np.full(len(values), -1, dtype=int)
        idx = np.searchsorted(ref_values, values, side="left")
        idx = np.clip(idx, 0, ref_values.size - 1)
        prev_idx = np.clip(idx - 1, 0, ref_values.size - 1)
        idx_dist = np.abs(ref_values[idx] - values)
        prev_dist = np.abs(ref_values[prev_idx] - values)
        return np.where(prev_dist <= idx_dist, prev_idx, idx).astype(int)

    for pid, sub in profile.groupby("profile_id"):
        mask = pts["profile_id"].astype(str) == str(pid)
        if not mask.any():
            continue
        stations = pd.to_numeric(pts.loc[mask, "station_m"], errors="coerce").to_numpy(dtype=float)
        ref_station = sub["station_m"].to_numpy(dtype=float)
        order = np.argsort(ref_station)
        ref_station = ref_station[order]
        sub_ordered = sub.iloc[order].reset_index(drop=True)
        elev = np.interp(stations, ref_station, sub_ordered["bed_elevation_m"].to_numpy(dtype=float), left=np.nan, right=np.nan)
        unc = np.interp(stations, ref_station, sub_ordered["uncertainty_m"].to_numpy(dtype=float), left=np.nan, right=np.nan)
        net_elev = np.interp(stations, ref_station, sub_ordered["network_backbone_elevation_m"].to_numpy(dtype=float), left=np.nan, right=np.nan)
        net_unc = np.interp(stations, ref_station, sub_ordered["network_backbone_uncertainty_m"].to_numpy(dtype=float), left=np.nan, right=np.nan)
        active_elev = np.interp(stations, ref_station, pd.to_numeric(sub_ordered.get("active_core_support_elevation_m", sub_ordered["network_backbone_elevation_m"]), errors="coerce").to_numpy(dtype=float), left=np.nan, right=np.nan)
        active_unc = np.interp(stations, ref_station, pd.to_numeric(sub_ordered.get("active_core_support_uncertainty_m", sub_ordered["network_backbone_uncertainty_m"]), errors="coerce").to_numpy(dtype=float), left=np.nan, right=np.nan)
        left_bank_fit = np.interp(stations, ref_station, _numeric_array_from_frame(sub_ordered, "left_bank_fit_z_m"), left=np.nan, right=np.nan)
        right_bank_fit = np.interp(stations, ref_station, _numeric_array_from_frame(sub_ordered, "right_bank_fit_z_m"), left=np.nan, right=np.nan)
        bank_pair_fit = np.interp(stations, ref_station, _numeric_array_from_frame(sub_ordered, "bank_pair_fit_z_m"), left=np.nan, right=np.nan)
        generalized_base_ref = _numeric_array_from_frame(sub_ordered, "generalized_longitudinal_bed_base_elevation_m")
        if not np.any(np.isfinite(generalized_base_ref)) and "generalized_longitudinal_bed_elevation_m" in sub_ordered.columns:
            generalized_base_ref = pd.to_numeric(sub_ordered["generalized_longitudinal_bed_elevation_m"], errors="coerce").to_numpy(dtype=float)
        generalized_base = np.interp(stations, ref_station, generalized_base_ref, left=np.nan, right=np.nan)
        generalized_reconciled_ref = _numeric_array_from_frame(sub_ordered, "generalized_longitudinal_bed_reconciled_elevation_m")
        if not np.any(np.isfinite(generalized_reconciled_ref)) and "generalized_longitudinal_bed_elevation_m" in sub_ordered.columns:
            generalized_reconciled_ref = pd.to_numeric(sub_ordered["generalized_longitudinal_bed_elevation_m"], errors="coerce").to_numpy(dtype=float)
        generalized_reconciled = np.interp(stations, ref_station, generalized_reconciled_ref, left=np.nan, right=np.nan)
        local_auth_taper = np.interp(stations, ref_station, _numeric_array_from_frame(sub_ordered, "generalized_longitudinal_bed_local_auth_taper_m", default=0.0), left=np.nan, right=np.nan)
        local_auth_reconciliation_weight = np.interp(stations, ref_station, _numeric_array_from_frame(sub_ordered, "generalized_longitudinal_bed_local_auth_reconciliation_weight", default=0.0), left=np.nan, right=np.nan)
        reconciliation_delta = np.interp(stations, ref_station, _numeric_array_from_frame(sub_ordered, "authoritative_reconciliation_delta_m", default=0.0), left=np.nan, right=np.nan)
        reconciliation_weight = np.interp(stations, ref_station, _numeric_array_from_frame(sub_ordered, "authoritative_reconciliation_weight", default=0.0), left=np.nan, right=np.nan)
        reconciliation_confidence = np.interp(stations, ref_station, _numeric_array_from_frame(sub_ordered, "authoritative_reconciliation_confidence", default=0.0), left=np.nan, right=np.nan)
        reconciliation_support_distance = np.interp(stations, ref_station, _numeric_array_from_frame(sub_ordered, "authoritative_reconciliation_support_distance_m"), left=np.nan, right=np.nan)
        pts.loc[mask, "longitudinal_profile_elevation_m"] = elev
        pts.loc[mask, "longitudinal_profile_uncertainty_m"] = unc
        pts.loc[mask, "network_backbone_elevation_m"] = net_elev
        pts.loc[mask, "network_backbone_uncertainty_m"] = net_unc
        pts.loc[mask, "active_core_support_elevation_m"] = active_elev
        pts.loc[mask, "active_core_support_uncertainty_m"] = active_unc
        pts.loc[mask, "left_bank_fit_z_m"] = left_bank_fit
        pts.loc[mask, "right_bank_fit_z_m"] = right_bank_fit
        pts.loc[mask, "bank_pair_fit_z_m"] = bank_pair_fit
        pts.loc[mask, "generalized_longitudinal_bed_base_elevation_m"] = generalized_base
        pts.loc[mask, "generalized_longitudinal_bed_reconciled_elevation_m"] = generalized_reconciled
        pts.loc[mask, "generalized_longitudinal_bed_local_auth_taper_m"] = local_auth_taper
        pts.loc[mask, "generalized_longitudinal_bed_local_auth_reconciliation_weight"] = local_auth_reconciliation_weight
        pts.loc[mask, "authoritative_reconciliation_delta_m"] = reconciliation_delta
        pts.loc[mask, "authoritative_reconciliation_weight"] = reconciliation_weight
        pts.loc[mask, "authoritative_reconciliation_confidence"] = reconciliation_confidence
        pts.loc[mask, "authoritative_reconciliation_support_distance_m"] = reconciliation_support_distance
        # Use nearest-station mapping for categorical/source fields so tiny float
        # differences do not collapse valid semantics to None.
        nearest_idx = _nearest_lookup(stations, ref_station)
        nearest_rows = sub_ordered.iloc[nearest_idx].reset_index(drop=True)
        dom_series = nearest_rows.get("dominant_source")
        active_src_series = nearest_rows.get("active_core_support_source", nearest_rows.get("network_backbone_source"))
        generalized_src_series = nearest_rows.get("generalized_longitudinal_bed_source", active_src_series)
        pts.loc[mask, "longitudinal_profile_source"] = dom_series.to_numpy(dtype=object) if dom_series is not None else None
        pts.loc[mask, "network_backbone_source"] = nearest_rows.get("network_backbone_source").to_numpy(dtype=object)
        pts.loc[mask, "active_core_support_source"] = active_src_series.to_numpy(dtype=object)
        pts.loc[mask, "generalized_longitudinal_bed_source"] = generalized_src_series.to_numpy(dtype=object)
        for col in [
            "profile_support_class", "profile_support_source_count", "profile_support_present",
            "profile_authoritative_anchor_present", "profile_xs_support_present",
            "profile_centerline_support_present", "profile_bank_support_present",
            "profile_wse_support_present", "profile_measured_support_distance_m",
            "profile_far_from_measured_support", "profile_authoritative_role",
            "profile_authoritative_bed_support_present", "profile_authoritative_bank_margin_present",
            "profile_authoritative_bed_support_distance_m", "profile_far_from_authoritative_bed_support",
            "centerline_authoritative_support_station_present",
            "authoritative_reconciliation_observation_used", "authoritative_reconciliation_observed_bed_z_m",
            "authoritative_reconciliation_observed_residual_m", "authoritative_reconciliation_observation_confidence",
        ]:
            if col in nearest_rows.columns:
                pts.loc[mask, col] = nearest_rows[col].to_numpy()
    return pts


def _nearest_surface_from_points(*, shape, transform, domain_mask, points_gdf, value_field: str, max_distance_m: float):
    from scipy.spatial import cKDTree
    import rasterio.transform

    out = np.full(shape, np.nan, dtype=np.float32)
    influence = np.zeros(shape, dtype=np.float32)
    if points_gdf is None or getattr(points_gdf, "empty", True) or not np.any(domain_mask):
        return out, influence
    vals = np.asarray(points_gdf.get(value_field, np.full(len(points_gdf), np.nan)), dtype=np.float32)
    valid = np.isfinite(vals)
    if not np.any(valid):
        return out, influence
    pts = np.column_stack([points_gdf.geometry.x.to_numpy(dtype=float)[valid], points_gdf.geometry.y.to_numpy(dtype=float)[valid]])
    vals = vals[valid]
    tree = cKDTree(pts)
    rows, cols = np.where(np.asarray(domain_mask, dtype=bool))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    q = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    dist, idx = tree.query(q, k=1)
    idx = np.asarray(idx, dtype=int)
    dist = np.asarray(dist, dtype=float)
    within = np.isfinite(dist) & (dist <= float(max_distance_m))
    if not np.any(within):
        return out, influence
    out_rows = rows[within]
    out_cols = cols[within]
    out[out_rows, out_cols] = vals[idx[within]].astype(np.float32)
    influence[out_rows, out_cols] = np.clip(1.0 - (dist[within] / max(float(max_distance_m), 1e-6)), 0.0, 1.0).astype(np.float32)
    return out, influence


def build_and_write_longitudinal_profile(
    *,
    river_dir: str | Path,
    centerline_points_path: str | Path,
    centerline_elevation_path: str | Path,
    centerline_influence_path: Optional[str | Path] = None,
    centerline_stationing_path: Optional[str | Path] = None,
    xs_support_elevation_path: Optional[str | Path] = None,
    xs_support_weight_path: Optional[str | Path] = None,
    bank_elevation_path: Optional[str | Path] = None,
    bank_influence_path: Optional[str | Path] = None,
    bank_graph_confidence_path: Optional[str | Path] = None,
    bank_continuity_weight_path: Optional[str | Path] = None,
    bank_confluence_damping_path: Optional[str | Path] = None,
    bank_estuary_side_decay_path: Optional[str | Path] = None,
    bank_points_path: Optional[str | Path] = None,
    authoritative_support_depth_path: Optional[str | Path] = None,
    authoritative_bed_elevation_path: Optional[str | Path] = None,
    authoritative_support_mask_path: Optional[str | Path] = None,
    authoritative_support_points_path: Optional[str | Path] = None,
    authoritative_role_code_path: Optional[str | Path] = None,
    authoritative_role_confidence_path: Optional[str | Path] = None,
    authoritative_distance_to_bank_path: Optional[str | Path] = None,
    authoritative_normalized_channel_position_path: Optional[str | Path] = None,
    wse_elevation_path: Optional[str | Path] = None,
    wse_influence_path: Optional[str | Path] = None,
    wse_uncertainty_path: Optional[str | Path] = None,
    corridor_mask_path: Optional[str | Path] = None,
    network_edges_path: Optional[str | Path] = None,
) -> Dict[str, str]:
    import geopandas as gpd

    river_dir = Path(river_dir)
    cpts = Path(centerline_points_path)
    if not cpts.exists():
        raise ValueError(f"missing_centerline_points_for_longitudinal_profile: {cpts}")
    centerline_points = gpd.read_file(cpts)
    if centerline_points.empty or "station_m" not in centerline_points.columns:
        raise ValueError("centerline_points_missing_station_m_for_longitudinal_profile")

    sampled = pd.DataFrame(index=centerline_points.index)
    sampled["centerline_elevation_m"] = _sample_raster_at_points(centerline_points, centerline_elevation_path, "centerline_elevation_m")
    sampled["centerline_influence"] = _sample_raster_at_points(centerline_points, centerline_influence_path, "centerline_influence")
    sampled["xs_support_elevation_m"] = _sample_raster_at_points(centerline_points, xs_support_elevation_path, "xs_support_elevation_m")
    sampled["xs_support_weight"] = _sample_raster_at_points(centerline_points, xs_support_weight_path, "xs_support_weight")
    sampled["bank_elevation_m"] = _sample_raster_at_points(centerline_points, bank_elevation_path, "bank_elevation_m")
    sampled["bank_influence"] = _sample_raster_at_points(centerline_points, bank_influence_path, "bank_influence")
    sampled["bank_graph_confidence"] = _sample_raster_at_points(centerline_points, bank_graph_confidence_path, "bank_graph_confidence")
    sampled["bank_continuity_weight"] = _sample_raster_at_points(centerline_points, bank_continuity_weight_path, "bank_continuity_weight")
    sampled["bank_confluence_damping"] = _sample_raster_at_points(centerline_points, bank_confluence_damping_path, "bank_confluence_damping")
    sampled["bank_estuary_side_decay"] = _sample_raster_at_points(centerline_points, bank_estuary_side_decay_path, "bank_estuary_side_decay")
    sampled["authoritative_support_depth_m"] = _sample_raster_at_points(centerline_points, authoritative_support_depth_path, "authoritative_support_depth_m")
    sampled["authoritative_anchor_elevation_m"] = _sample_raster_at_points(centerline_points, authoritative_bed_elevation_path, "authoritative_anchor_elevation_m")
    sampled["authoritative_support_mask"] = _sample_raster_at_points(centerline_points, authoritative_support_mask_path, "authoritative_support_mask")
    sampled["authoritative_role_code"] = _sample_raster_at_points(centerline_points, authoritative_role_code_path, "authoritative_role_code")
    sampled["authoritative_role_confidence"] = _sample_raster_at_points(centerline_points, authoritative_role_confidence_path, "authoritative_role_confidence")
    sampled["authoritative_distance_to_bank_m"] = _sample_raster_at_points(centerline_points, authoritative_distance_to_bank_path, "authoritative_distance_to_bank_m")
    sampled["authoritative_normalized_channel_position"] = _sample_raster_at_points(centerline_points, authoritative_normalized_channel_position_path, "authoritative_normalized_channel_position")
    support_point_meta = _nearest_support_metadata(centerline_points, authoritative_support_points_path)
    for col in support_point_meta.columns:
        sampled[col] = support_point_meta[col]
    sampled["wse_elevation_m"] = _sample_raster_at_points(centerline_points, wse_elevation_path, "wse_elevation_m")
    sampled["wse_influence"] = _sample_raster_at_points(centerline_points, wse_influence_path, "wse_influence")
    sampled["wse_uncertainty_m"] = _sample_raster_at_points(centerline_points, wse_uncertainty_path, "wse_uncertainty_m")
    sampled["centerline_sample_source"] = centerline_points.get("centerline_sample_source", pd.Series(["missing"] * len(centerline_points), index=centerline_points.index, dtype=object)).fillna("missing").astype(str)
    sampled["centerline_authoritative_support_applied"] = centerline_points.get("centerline_authoritative_support_applied", pd.Series(False, index=centerline_points.index)).fillna(False).astype(bool)
    sampled["centerline_authoritative_support_distance_m"] = pd.to_numeric(centerline_points.get("centerline_authoritative_support_distance_m"), errors="coerce")
    direct_centerline = pd.to_numeric(centerline_points.get("centerline_z_m"), errors="coerce") if "centerline_z_m" in centerline_points.columns else pd.Series(np.nan, index=centerline_points.index, dtype=float)
    direct_centerline_mask = sampled["centerline_authoritative_support_applied"].fillna(False).to_numpy(dtype=bool) & np.isfinite(direct_centerline.to_numpy(dtype=float))
    if np.any(direct_centerline_mask):
        sampled.loc[direct_centerline_mask, "centerline_elevation_m"] = direct_centerline.loc[direct_centerline_mask].to_numpy(dtype=np.float32)

    if not np.any(np.isfinite(sampled["centerline_elevation_m"].to_numpy(dtype=float))):
        if "centerline_z_m" in centerline_points.columns:
            centerline_attr = pd.to_numeric(centerline_points["centerline_z_m"], errors="coerce")
            if np.any(np.isfinite(centerline_attr.to_numpy(dtype=float))):
                sampled["centerline_elevation_m"] = centerline_attr.astype(np.float32)
            else:
                raise ValueError("centerline_elevation_sampling_missing_for_longitudinal_profile")
        else:
            raise ValueError("centerline_elevation_sampling_missing_for_longitudinal_profile")

    profile = _build_profile_dataframe(centerline_points, sampled)
    if profile.empty:
        raise ValueError("empty_longitudinal_profile_after_sampling")

    edges = _read_network_edges(network_edges_path)
    endpoint_meta = _component_endpoint_metadata(edges)
    hierarchy_meta = _component_hierarchy_metadata(edges)
    profile, authoritative_bed_anchor_curve, authoritative_bed_anchor_curve_diagnostics = _build_authoritative_bed_anchor_curve(profile, endpoint_meta)
    bank_points_fit, bank_fit_point_diagnostics = fit_bank_longitudinal_points(
        load_bank_points(bank_points_path),
        endpoint_meta=endpoint_meta,
    )
    profile, bank_fit_profile_diagnostics = sample_bank_fit_to_profile(profile, bank_points_fit)
    bank_fit_handoff_rows = []
    for comp, sub in profile.groupby("profile_id", sort=False):
        comp_key = _normalize_component_key(comp)
        bank_pair_fit = pd.to_numeric(sub.get("bank_pair_fit_z_m"), errors="coerce").to_numpy(dtype=float)
        left_fit = pd.to_numeric(sub.get("left_bank_fit_z_m"), errors="coerce").to_numpy(dtype=float)
        right_fit = pd.to_numeric(sub.get("right_bank_fit_z_m"), errors="coerce").to_numpy(dtype=float)
        raw_bank = pd.to_numeric(sub.get("bank_elevation_m"), errors="coerce").to_numpy(dtype=float)
        bank_fit_handoff_rows.append({
            "profile_id": str(comp),
            "normalized_profile_id": comp_key,
            "n_rows": int(len(sub)),
            "left_bank_fit_rows": int(np.count_nonzero(np.isfinite(left_fit))),
            "right_bank_fit_rows": int(np.count_nonzero(np.isfinite(right_fit))),
            "bank_pair_fit_rows": int(np.count_nonzero(np.isfinite(bank_pair_fit))),
            "raw_bank_rows": int(np.count_nonzero(np.isfinite(raw_bank))),
            "bank_pair_fit_missing_rows": int(np.count_nonzero(~np.isfinite(bank_pair_fit))),
            "bank_pair_fit_overwrite_candidate_rows": int(np.count_nonzero(np.isfinite(bank_pair_fit) & np.isfinite(raw_bank))),
        })
    bank_fit_handoff = pd.DataFrame(bank_fit_handoff_rows)
    profile, bank_profile_diagnostics = _apply_bank_longitudinal_reference(profile, endpoint_meta)
    profile = _apply_network_continuity(profile, endpoint_meta, hierarchy_meta=hierarchy_meta)
    profile, active_core_support_diagnostics = _apply_active_core_support(profile, endpoint_meta)
    profile, profile_component_diagnostics = _sanitize_profile_for_rasterization(profile)
    if profile.empty:
        raise ValueError("empty_longitudinal_profile_after_component_sanitization")
    profile = _classify_longitudinal_profile_coverage(profile)
    coverage_summary = _summarize_longitudinal_profile_coverage(profile)

    out_csv = river_dir / "river_longitudinal_profile.csv"
    profile.to_csv(out_csv, index=False)
    out_coverage_csv = river_dir / "river_longitudinal_profile_coverage.csv"
    coverage_cols = [
        "profile_id", "station_m", "profile_support_class", "profile_support_source_count",
        "profile_support_present", "profile_authoritative_anchor_present", "profile_xs_support_present",
        "profile_centerline_support_present", "profile_bank_support_present", "profile_wse_support_present",
        "profile_measured_support_distance_m", "profile_far_from_measured_support",
        "profile_authoritative_role", "profile_authoritative_role_code", "profile_authoritative_role_confidence",
        "profile_authoritative_distance_to_bank_m", "profile_authoritative_normalized_channel_position",
        "profile_authoritative_bed_support_present", "profile_authoritative_bank_margin_present",
        "profile_authoritative_bed_core_present", "profile_authoritative_ambiguous_present",
        "profile_authoritative_bed_support_distance_m", "profile_authoritative_bank_margin_distance_m",
        "profile_authoritative_station_local_support_radius_m", "profile_far_from_authoritative_bed_support",
        "centerline_authoritative_support_station_present", "centerline_sample_source",
    ]
    coverage_cols = [c for c in coverage_cols if c in profile.columns]
    profile[coverage_cols].to_csv(out_coverage_csv, index=False)

    profile_points = _assign_profile_values_to_points(centerline_points, profile)
    out_gpkg = river_dir / "river_longitudinal_profile_points.gpkg"
    if out_gpkg.exists():
        out_gpkg.unlink()
    profile_points.to_file(out_gpkg, driver="GPKG")

    out_recon_points_csv = river_dir / "river_authoritative_reconciliation_points.csv"
    recon_point_cols = [
        "profile_id", "station_m", "generalized_longitudinal_bed_base_elevation_m",
        "authoritative_reconciliation_observed_bed_z_m", "authoritative_reconciliation_observed_residual_m",
        "authoritative_reconciliation_observation_confidence",
    ]
    recon_points = profile_points.loc[profile_points.get("authoritative_reconciliation_observation_used", pd.Series(False, index=profile_points.index)).fillna(False).astype(bool)].copy()
    if not recon_points.empty:
        recon_points["x"] = recon_points.geometry.x.astype(float)
        recon_points["y"] = recon_points.geometry.y.astype(float)
    recon_points[[c for c in recon_point_cols + ["x", "y"] if c in recon_points.columns]].to_csv(out_recon_points_csv, index=False)

    out_recon_field_csv = river_dir / "river_authoritative_reconciliation_field.csv"
    recon_field_cols = [
        "profile_id", "station_m", "generalized_longitudinal_bed_base_elevation_m",
        "generalized_longitudinal_bed_reconciled_elevation_m", "authoritative_reconciliation_delta_m",
        "authoritative_reconciliation_weight", "authoritative_reconciliation_confidence",
        "authoritative_reconciliation_support_distance_m", "authoritative_reconciliation_exact_anchor",
    ]
    profile[[c for c in recon_field_cols if c in profile.columns]].to_csv(out_recon_field_csv, index=False)

    out_backbone_csv = river_dir / "river_hydraulic_backbone.csv"
    backbone_cols = [
        "profile_id", "station_m", "station_step_m", "bed_elevation_m", "uncertainty_m",
        "wse_elevation_m", "wse_influence", "wse_uncertainty_m", "wse_local_slope_mpm",
        "network_backbone_elevation_m", "network_backbone_uncertainty_m", "network_junction_adjustment_m",
        "network_backbone_source", "active_core_support_elevation_m", "active_core_support_uncertainty_m", "active_core_support_source", "active_core_support_monotone_adjustment_m", "active_core_support_bank_reference_m", "active_core_support_bank_offset_m", "active_core_support_model_elevation_m", "active_core_support_model_weight", "active_core_support_offset_source", "profile_inside_fluvial_monotone_domain", "profile_fluvial_monotone_score", "profile_fluvial_monotone_domain_source", "authoritative_anchor_source", "authoritative_anchor_raw_source", "authoritative_anchor_curve_source", "authoritative_anchor_curve_present", "authoritative_anchor_curve_elevation_m", "authoritative_anchor_curve_count", "authoritative_anchor_curve_dispersion_m", "authoritative_anchor_curve_adjustment_m", "junction_hierarchy_weight", "junction_wse_weight", "wse_endpoint_consensus_residual_m", "drainage_area_proxy", "stream_order_proxy",
        "generalized_longitudinal_bed_base_elevation_m", "generalized_longitudinal_bed_reconciled_elevation_m", "authoritative_reconciliation_delta_m", "authoritative_reconciliation_weight", "authoritative_reconciliation_confidence", "authoritative_reconciliation_support_distance_m",
        "left_bank_fit_z_m", "right_bank_fit_z_m", "bank_pair_fit_z_m", "bank_pair_fit_residual_m", "bank_profile_active_source",
    ]
    backbone_cols = [c for c in backbone_cols if c in profile.columns]
    profile[backbone_cols].to_csv(out_backbone_csv, index=False)

    out_bank_points = river_dir / "river_bank_longitudinal_fit_points.gpkg"
    if out_bank_points.exists():
        out_bank_points.unlink()
    if bank_points_fit is not None and not getattr(bank_points_fit, "empty", True):
        bank_points_fit.to_file(out_bank_points, driver="GPKG")

    out_nodes = river_dir / "river_hydraulic_backbone_nodes.gpkg"
    node_rows = []
    points_by_comp = profile_points.copy()
    points_by_comp["profile_id"] = _infer_profile_id(points_by_comp)
    points_by_comp["station_m"] = pd.to_numeric(points_by_comp.get("station_m"), errors="coerce")
    for comp, meta in endpoint_meta.items():
        sub = profile.loc[profile["profile_id"].astype(str) == str(comp)].sort_values("station_m")
        comp_points = points_by_comp.loc[points_by_comp["profile_id"].astype(str) == str(comp)].sort_values("station_m")
        if sub.empty or comp_points.empty:
            continue
        start_point = comp_points.geometry.iloc[0]
        end_point = comp_points.geometry.iloc[-1]
        node_rows.append({"component_id": str(comp), "node_id": int(meta["start_node"]), "position": "start", "bed_elevation_m": float(sub["network_backbone_elevation_m"].iloc[0]), "uncertainty_m": float(sub["network_backbone_uncertainty_m"].iloc[0]), "station_m": float(sub["station_m"].iloc[0]), "geometry": start_point})
        node_rows.append({"component_id": str(comp), "node_id": int(meta["end_node"]), "position": "end", "bed_elevation_m": float(sub["network_backbone_elevation_m"].iloc[-1]), "uncertainty_m": float(sub["network_backbone_uncertainty_m"].iloc[-1]), "station_m": float(sub["station_m"].iloc[-1]), "geometry": end_point})
    if node_rows:
        nodes_gdf = gpd.GeoDataFrame(node_rows, geometry="geometry", crs=centerline_points.crs)
        nodes_gdf = nodes_gdf[nodes_gdf.geometry.notnull()].copy()
        if not nodes_gdf.empty:
            if out_nodes.exists():
                out_nodes.unlink()
            nodes_gdf.to_file(out_nodes, driver="GPKG")

    out_edges = river_dir / "river_hydraulic_backbone_edges.gpkg"
    if edges is not None and len(edges):
        edges_out = edges.copy()
        edge_profile = profile.groupby("profile_id", dropna=False).agg(
            network_backbone_min_m=("network_backbone_elevation_m", "min"),
            network_backbone_max_m=("network_backbone_elevation_m", "max"),
            network_backbone_uncertainty_m=("network_backbone_uncertainty_m", "median"),
        ).reset_index()
        edges_out["component_id"] = edges_out["component_id"].astype(str)
        edges_out = edges_out.merge(edge_profile, left_on="component_id", right_on="profile_id", how="left")
        if out_edges.exists():
            out_edges.unlink()
        edges_out.to_file(out_edges, driver="GPKG")

    if centerline_stationing_path is not None and Path(centerline_stationing_path).exists():
        with rasterio.open(centerline_stationing_path) as ds:
            profile_meta = ds.profile.copy()
            profile_meta.pop("blockxsize", None)
            profile_meta.pop("blockysize", None)
            profile_meta.pop("tiled", None)
            profile_meta.update(dtype="float32", count=1, nodata=np.float32(np.nan))
            stationing = ds.read(1).astype(np.float32)
            if corridor_mask_path is not None and Path(corridor_mask_path).exists():
                with rasterio.open(corridor_mask_path) as cm:
                    domain_mask = cm.read(1, out_shape=(ds.height, ds.width), resampling=rasterio.enums.Resampling.nearest) > 0
            else:
                domain_mask = np.isfinite(stationing)
            centerline_mask = _seed_mask_from_points(
                shape=(ds.height, ds.width),
                transform=ds.transform,
                domain_mask=domain_mask,
                points_gdf=centerline_points,
            )
            step_candidates = pd.to_numeric(profile["station_step_m"], errors="coerce").to_numpy(dtype=float)
            valid_steps = step_candidates[np.isfinite(step_candidates) & (step_candidates > 0.0)]
            step_m = float(np.median(valid_steps)) if valid_steps.size else 20.0
            pixel_size_m = float(max(abs(ds.transform.a), abs(ds.transform.e), 1.0))
            centerline_value_grid, centerline_unc_grid, component_index, component_lookup = _build_centerline_profile_grids(
                shape=(ds.height, ds.width),
                transform=ds.transform,
                domain_mask=domain_mask,
                stationing=stationing,
                centerline_points=centerline_points,
                profile=profile,
            )
            elev_raster, unc_raster, elev_influence = _transport_centerline_profile_to_corridor(
                domain_mask=domain_mask,
                centerline_mask=centerline_mask,
                stationing=stationing,
                value_grid=centerline_value_grid,
                uncertainty_grid=centerline_unc_grid,
                component_index=component_index,
                pixel_size_m=pixel_size_m,
                along_scale_m=max(step_m * 20.0, 200.0),
                lateral_decay_m=max(step_m * 3.5, 8.0 * pixel_size_m, 25.0),
            )
            elev_raster, unc_raster, elev_influence, raster_component_diagnostics = _sanitize_transported_profile_raster(
                profile=profile,
                transported=elev_raster,
                transported_unc=unc_raster,
                influence=elev_influence,
                component_index=component_index,
                component_lookup=component_lookup,
                centerline_mask=centerline_mask,
                domain_mask=domain_mask,
                pixel_size_m=pixel_size_m,
            )
        out_elev = river_dir / "river_longitudinal_profile_elevation.tif"
        out_base_bed = river_dir / "river_generalized_longitudinal_bed_base_elevation.tif"
        out_reconciled_bed = river_dir / "river_generalized_longitudinal_bed_reconciled_elevation.tif"
        out_unc = river_dir / "river_longitudinal_profile_uncertainty.tif"
        out_inf = river_dir / "river_longitudinal_profile_influence.tif"
        out_recon = river_dir / "river_longitudinal_profile_local_authoritative_reconciliation.tif"
        out_recon_inf = river_dir / "river_longitudinal_profile_local_authoritative_reconciliation_influence.tif"
        with rasterio.open(out_elev, "w", **profile_meta) as dst:
            dst.write(elev_raster.astype(np.float32), 1)
            dst.update_tags(VALUE_TYPE="elevation", UNITS="meters", VERTICAL_SEMANTICS="absolute_elevation", ROLE="river_network_hydraulic_backbone")
        base_points = profile_points.loc[np.isfinite(pd.to_numeric(profile_points.get("generalized_longitudinal_bed_base_elevation_m"), errors="coerce").to_numpy(dtype=float))].copy()
        if not base_points.empty:
            base_bed_raster, _ = _nearest_surface_from_points(
                shape=(ds.height, ds.width),
                transform=ds.transform,
                domain_mask=domain_mask,
                points_gdf=base_points,
                value_field="generalized_longitudinal_bed_base_elevation_m",
                max_distance_m=max(step_m * 3.5, 8.0 * pixel_size_m, 25.0),
            )
        else:
            base_bed_raster = np.full((ds.height, ds.width), np.nan, dtype=np.float32)
        with rasterio.open(out_base_bed, "w", **profile_meta) as dst:
            dst.write(base_bed_raster.astype(np.float32), 1)
            dst.update_tags(VALUE_TYPE="elevation", UNITS="meters", VERTICAL_SEMANTICS="absolute_elevation", ROLE="river_generalized_longitudinal_bed_base")
        reconciled_points = profile_points.loc[np.isfinite(pd.to_numeric(profile_points.get("generalized_longitudinal_bed_reconciled_elevation_m"), errors="coerce").to_numpy(dtype=float))].copy()
        if not reconciled_points.empty:
            reconciled_bed_raster, _ = _nearest_surface_from_points(
                shape=(ds.height, ds.width),
                transform=ds.transform,
                domain_mask=domain_mask,
                points_gdf=reconciled_points,
                value_field="generalized_longitudinal_bed_reconciled_elevation_m",
                max_distance_m=max(step_m * 3.5, 8.0 * pixel_size_m, 25.0),
            )
        else:
            reconciled_bed_raster = np.full((ds.height, ds.width), np.nan, dtype=np.float32)
        with rasterio.open(out_reconciled_bed, "w", **profile_meta) as dst:
            dst.write(reconciled_bed_raster.astype(np.float32), 1)
            dst.update_tags(VALUE_TYPE="elevation", UNITS="meters", VERTICAL_SEMANTICS="absolute_elevation", ROLE="river_generalized_longitudinal_bed_reconciled")
        with rasterio.open(out_unc, "w", **profile_meta) as dst:
            dst.write(unc_raster.astype(np.float32), 1)
            dst.update_tags(VALUE_TYPE="uncertainty", UNITS="meters", ROLE="river_network_hydraulic_backbone_uncertainty")
        with rasterio.open(out_inf, "w", **profile_meta) as dst:
            dst.write(elev_influence.astype(np.float32), 1)
            dst.update_tags(VALUE_TYPE="fraction", ROLE="river_network_hydraulic_backbone_influence")
        reconciliation_points = profile_points.loc[pd.to_numeric(profile_points.get("authoritative_reconciliation_weight", profile_points.get("generalized_longitudinal_bed_local_auth_reconciliation_weight")), errors="coerce").fillna(0.0) > 0.0].copy()
        if not reconciliation_points.empty:
            recon_delta, _ = _nearest_surface_from_points(
                shape=(ds.height, ds.width),
                transform=ds.transform,
                domain_mask=domain_mask,
                points_gdf=reconciliation_points,
                value_field="authoritative_reconciliation_delta_m",
                max_distance_m=250.0,
            )
            recon_weight, recon_decay = _nearest_surface_from_points(
                shape=(ds.height, ds.width),
                transform=ds.transform,
                domain_mask=domain_mask,
                points_gdf=reconciliation_points,
                value_field="authoritative_reconciliation_weight",
                max_distance_m=250.0,
            )
            recon_influence = np.clip(np.nan_to_num(recon_weight, nan=0.0) * np.nan_to_num(recon_decay, nan=0.0), 0.0, 1.0).astype(np.float32)
        else:
            recon_delta = np.full((ds.height, ds.width), np.nan, dtype=np.float32)
            recon_influence = np.zeros((ds.height, ds.width), dtype=np.float32)
        with rasterio.open(out_recon, "w", **profile_meta) as dst:
            dst.write(recon_delta.astype(np.float32), 1)
            dst.update_tags(VALUE_TYPE="elevation_delta", UNITS="meters", ROLE="river_longitudinal_profile_local_authoritative_reconciliation_delta")
        with rasterio.open(out_recon_inf, "w", **profile_meta) as dst:
            dst.write(recon_influence.astype(np.float32), 1)
            dst.update_tags(VALUE_TYPE="fraction", ROLE="river_longitudinal_profile_local_authoritative_reconciliation_influence")
        out_left_bank_fit = river_dir / "river_left_bank_fit_elevation.tif"
        out_right_bank_fit = river_dir / "river_right_bank_fit_elevation.tif"
        out_bank_pair_fit = river_dir / "river_bank_pair_fit_elevation.tif"
        left_bank_fit, _ = _nearest_surface_from_points(
            shape=(ds.height, ds.width),
            transform=ds.transform,
            domain_mask=domain_mask,
            points_gdf=(bank_points_fit.loc[bank_points_fit["side"].astype(str) == "left"].copy() if bank_points_fit is not None and not getattr(bank_points_fit, "empty", True) else None),
            value_field="bank_fit_monotone_m",
            max_distance_m=max(step_m * 6.0, 60.0),
        )
        right_bank_fit, _ = _nearest_surface_from_points(
            shape=(ds.height, ds.width),
            transform=ds.transform,
            domain_mask=domain_mask,
            points_gdf=(bank_points_fit.loc[bank_points_fit["side"].astype(str) == "right"].copy() if bank_points_fit is not None and not getattr(bank_points_fit, "empty", True) else None),
            value_field="bank_fit_monotone_m",
            max_distance_m=max(step_m * 6.0, 60.0),
        )
        bank_pair_fit, _ = _nearest_surface_from_points(
            shape=(ds.height, ds.width),
            transform=ds.transform,
            domain_mask=domain_mask,
            points_gdf=profile_points,
            value_field="bank_pair_fit_z_m",
            max_distance_m=max(step_m * 4.0, 50.0),
        )
        for arr, out_path, role in [
            (left_bank_fit, out_left_bank_fit, "left_bank_longitudinal_fit"),
            (right_bank_fit, out_right_bank_fit, "right_bank_longitudinal_fit"),
            (bank_pair_fit, out_bank_pair_fit, "bank_pair_longitudinal_fit"),
        ]:
            with rasterio.open(out_path, "w", **profile_meta) as dst:
                dst.write(arr.astype(np.float32), 1)
                dst.update_tags(VALUE_TYPE="elevation", UNITS="meters", VERTICAL_SEMANTICS="absolute_elevation", ROLE=role)

        out_active_elev = river_dir / "river_active_core_support_elevation.tif"
        out_active_unc = river_dir / "river_active_core_support_uncertainty.tif"
        out_active_inf = river_dir / "river_active_core_support_influence.tif"
        for src_path, dst_path in [(out_elev, out_active_elev), (out_unc, out_active_unc), (out_inf, out_active_inf)]:
            if dst_path.exists():
                dst_path.unlink()
            dst_path.write_bytes(src_path.read_bytes())
    else:
        out_elev = river_dir / "river_longitudinal_profile_elevation.tif"
        out_base_bed = river_dir / "river_generalized_longitudinal_bed_base_elevation.tif"
        out_reconciled_bed = river_dir / "river_generalized_longitudinal_bed_reconciled_elevation.tif"
        out_unc = river_dir / "river_longitudinal_profile_uncertainty.tif"
        out_inf = river_dir / "river_longitudinal_profile_influence.tif"
        out_recon = river_dir / "river_longitudinal_profile_local_authoritative_reconciliation.tif"
        out_recon_inf = river_dir / "river_longitudinal_profile_local_authoritative_reconciliation_influence.tif"
        out_active_elev = river_dir / "river_active_core_support_elevation.tif"
        out_active_unc = river_dir / "river_active_core_support_uncertainty.tif"
        out_active_inf = river_dir / "river_active_core_support_influence.tif"
        out_left_bank_fit = river_dir / "river_left_bank_fit_elevation.tif"
        out_right_bank_fit = river_dir / "river_right_bank_fit_elevation.tif"
        out_bank_pair_fit = river_dir / "river_bank_pair_fit_elevation.tif"

    if endpoint_meta:
        continuity_residuals = []
        adjacency = _adjacency_from_endpoint_metadata(endpoint_meta)
        for _, members in adjacency.items():
            vals = []
            for comp, pos in members:
                sub = profile.loc[profile["profile_id"].astype(str) == str(comp)].sort_values("station_m")
                if sub.empty:
                    continue
                vals.append(float(sub["network_backbone_elevation_m"].iloc[0 if pos == "start" else -1]))
            if len(vals) >= 2:
                continuity_residuals.append(float(np.nanmax(vals) - np.nanmin(vals)))
    else:
        continuity_residuals = []

    summary = {
        "profile_count": int(profile["profile_id"].nunique()),
        "station_count": int(len(profile)),
        "station_min_m": float(np.nanmin(profile["station_m"].to_numpy(dtype=float))),
        "station_max_m": float(np.nanmax(profile["station_m"].to_numpy(dtype=float))),
        "has_authoritative_support_depth": bool(np.any(np.isfinite(profile["authoritative_support_depth_m"].to_numpy(dtype=float)))),
        "network_aware": bool(endpoint_meta),
        "network_component_count": int(len(endpoint_meta)),
        "flow_aware_junction_weighting": bool(bool(hierarchy_meta)),
        "wse_aware_junction_weighting": bool(np.any(np.isfinite(pd.to_numeric(profile.get("wse_elevation_m"), errors="coerce").to_numpy(dtype=float))) if "wse_elevation_m" in profile.columns else False),
        "junction_count": int(sum(1 for _, m in _adjacency_from_endpoint_metadata(endpoint_meta).items() if len(m) > 1)),
        "max_junction_continuity_residual_m": float(np.nanmax(continuity_residuals)) if continuity_residuals else 0.0,
        "authoritative_anchor_station_count": int(np.sum(profile.get("authoritative_anchor_present", pd.Series(False, index=profile.index)).fillna(False).to_numpy(dtype=bool))),
        "bank_longitudinal_fit": {
            "bank_point_stage": bank_fit_point_diagnostics,
            "profile_stage": bank_fit_profile_diagnostics,
        },
        "bank_longitudinal_reference": bank_profile_diagnostics,
        "active_core_support": active_core_support_diagnostics,
        "authoritative_reconciliation": {
            "observation_count": int(profile.get("authoritative_reconciliation_observation_used", pd.Series(False, index=profile.index)).fillna(False).sum()),
            "max_abs_delta_m": float(np.nanmax(np.abs(pd.to_numeric(profile.get("authoritative_reconciliation_delta_m"), errors="coerce").to_numpy(dtype=float)))) if "authoritative_reconciliation_delta_m" in profile.columns and np.any(np.isfinite(pd.to_numeric(profile.get("authoritative_reconciliation_delta_m"), errors="coerce").to_numpy(dtype=float))) else 0.0,
            "max_weight": float(np.nanmax(pd.to_numeric(profile.get("authoritative_reconciliation_weight"), errors="coerce").to_numpy(dtype=float))) if "authoritative_reconciliation_weight" in profile.columns and np.any(np.isfinite(pd.to_numeric(profile.get("authoritative_reconciliation_weight"), errors="coerce").to_numpy(dtype=float))) else 0.0,
        },
        "authoritative_bed_anchor_curve": authoritative_bed_anchor_curve_diagnostics,
        "component_sanitization": profile_component_diagnostics,
        "raster_component_sanitization": raster_component_diagnostics,
        "coverage_summary": coverage_summary,
        "notes": [
            "This is the workflow-integrated network-aware 1D river backbone object.",
            "bed_elevation_m remains the local station-indexed tendency, but the generalized longitudinal bed exported here is now driven by the network/backbone solve rather than by writing bank-derived references back into bed truth.",
            "network_backbone_elevation_m applies component smoothing plus cross-component junction solving using scaffold graph topology; when drainage-area or stream-order metadata are present, junction targets are flow-aware so dominant branches exert more downstream control while preserving authoritative anchor stations as hard vertical control.",
            "active_core_support_elevation_m is the explicit fluvial active channel-core support exported for downstream scaffold/surface use; in unsupported fluvial reaches it is solved from the generalized longitudinal bed, then adjusted by one explicit authoritative reconciliation field built from nearby authoritative interior-bed residual observations without replacing authoritative bathymetry itself.",
            "When a reliable WSE surface is available, endpoint junction arbitration is additionally WSE-aware: branches whose observed WSE is closer to junction consensus get more influence, so hydraulic stage agreement can reinforce hierarchy and continuity without overriding authoritative anchors.",
            "authoritative_support_depth_m is still auxiliary evidence and is not merged directly into absolute bed elevation without an explicit water-surface reference.",
        ],
    }
    out_bank_fit_handoff_csv = river_dir / "river_bank_fit_handoff_station_table.csv"
    bank_fit_handoff.to_csv(out_bank_fit_handoff_csv, index=False)
    out_bank_fit_summary = river_dir / "river_bank_longitudinal_fit_summary.json"
    out_bank_fit_summary.write_text(
        json.dumps({
            "bank_point_stage": bank_fit_point_diagnostics,
            "profile_stage": bank_fit_profile_diagnostics,
            "handoff_stage": {row["normalized_profile_id"]: {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in row.items() if k not in {"profile_id", "normalized_profile_id"}} for row in bank_fit_handoff.to_dict(orient="records")},
        }, indent=2),
        encoding="utf-8",
    )
    out_authoritative_anchor_curve = river_dir / "river_authoritative_bed_anchor_curve.csv"
    authoritative_bed_anchor_curve.to_csv(out_authoritative_anchor_curve, index=False)
    out_authoritative_anchor_curve_summary = river_dir / "river_authoritative_bed_anchor_curve_summary.json"
    out_authoritative_anchor_curve_summary.write_text(
        json.dumps({
            "components": authoritative_bed_anchor_curve_diagnostics,
            "total_curve_anchor_rows": int(len(authoritative_bed_anchor_curve)),
            "notes": [
                "This receipt describes the stationized authoritative interior-bed anchor curve used to thin raw authoritative bed support.",
                "Dense raw authoritative anchors are binned in longitudinal station space and cleaned for downstream monotonicity in fluvial runs before they are used as hard anchors by the active core-support solve.",
            ],
        }, indent=2),
        encoding="utf-8",
    )
    out_summary = river_dir / "river_longitudinal_profile_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "longitudinal_profile": str(out_csv),
        "longitudinal_profile_points": str(out_gpkg),
        "longitudinal_profile_summary": str(out_summary),
        "longitudinal_profile_coverage": str(out_coverage_csv),
        "authoritative_reconciliation_points": str(out_recon_points_csv) if out_recon_points_csv.exists() else None,
        "authoritative_reconciliation_field": str(out_recon_field_csv) if out_recon_field_csv.exists() else None,
        "longitudinal_profile_elevation": str(out_elev) if out_elev.exists() else None,
        "generalized_longitudinal_bed_base": str(out_base_bed) if out_base_bed.exists() else None,
        "generalized_longitudinal_bed_reconciled": str(out_reconciled_bed) if out_reconciled_bed.exists() else None,
        "longitudinal_profile_uncertainty": str(out_unc) if out_unc.exists() else None,
        "longitudinal_profile_influence": str(out_inf) if out_inf.exists() else None,
        "longitudinal_profile_local_authoritative_reconciliation": str(out_recon) if out_recon.exists() else None,
        "longitudinal_profile_local_authoritative_reconciliation_influence": str(out_recon_inf) if out_recon_inf.exists() else None,
        "active_core_support_elevation": str(out_active_elev) if out_active_elev.exists() else None,
        "active_core_support_uncertainty": str(out_active_unc) if out_active_unc.exists() else None,
        "active_core_support_influence": str(out_active_inf) if out_active_inf.exists() else None,
        "bank_longitudinal_fit_points": str(out_bank_points) if out_bank_points.exists() else None,
        "bank_longitudinal_fit_summary": str(out_bank_fit_summary) if out_bank_fit_summary.exists() else None,
        "bank_fit_handoff_station_table": str(out_bank_fit_handoff_csv) if out_bank_fit_handoff_csv.exists() else None,
        "authoritative_bed_anchor_curve": str(out_authoritative_anchor_curve) if out_authoritative_anchor_curve.exists() else None,
        "authoritative_bed_anchor_curve_summary": str(out_authoritative_anchor_curve_summary) if out_authoritative_anchor_curve_summary.exists() else None,
        "left_bank_fit_elevation": str(out_left_bank_fit) if out_left_bank_fit.exists() else None,
        "right_bank_fit_elevation": str(out_right_bank_fit) if out_right_bank_fit.exists() else None,
        "bank_pair_fit_elevation": str(out_bank_pair_fit) if out_bank_pair_fit.exists() else None,
        "hydraulic_backbone": str(out_backbone_csv),
        "hydraulic_backbone_nodes": str(out_nodes) if out_nodes.exists() else None,
        "hydraulic_backbone_edges": str(out_edges) if out_edges.exists() else None,
    }


__all__ = ["build_and_write_longitudinal_profile"]
