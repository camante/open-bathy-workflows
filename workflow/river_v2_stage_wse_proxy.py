from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

from river_v2_context import RiverV2Context
from river_v2_contract import RiverV2StageResult, STAGE_CENTERLINE_WSE_PROXY
from river_v2_receipts import build_river_v2_stage_receipt, write_river_v2_receipt

_COMBINED_BANK_FIELD_CANDIDATES = (
    "bank_wse_proxy_raw_m",
    "bank_wse_proxy_interp_m",
    "bank_wse_proxy_monotone_m",
    "wse_proxy_z_m",
    "bank_low_stage_z_m",
    "bank_elevation_m",
)
_GROUP_FIELD_CANDIDATES = ("component_id", "levelpath_id", "reach_id", "source_reach_key")
_REQUIRED_OUTPUT_FIELDS = (
    "point_id",
    "station_m",
    "wse_proxy_z_m",
    "wse_valid",
    "wse_quality_flag",
    "geometry",
)


def _identity_columns(frame: pd.DataFrame) -> list[str]:
    return [field for field in ["point_id", "station_m", "station_direction", "stream_order", "mean_width_m", "geometry", *_GROUP_FIELD_CANDIDATES] if field in frame.columns]


def wse_proxy_field_schema() -> dict[str, str]:
    return {
        "point_id": "string",
        "station_m": "float64",
        "station_direction": "string",
        "wse_support_z_m": "float64",
        "support_type": "string",
        "has_support": "bool",
        "wse_anchor_raw_z_m": "float64",
        "wse_interp_z_m": "float64",
        "wse_trend_z_m": "float64",
        "wse_pre_smooth_z_m": "float64",
        "stream_order": "float64",
        "mean_width_m": "float64",
        "wse_proxy_z_m": "float64",
        "final_adjustment_m": "float64",
        "final_adjustment_reason": "string",
        "wse_valid": "bool",
        "wse_quality_flag": "string",
        "geometry": "Point",
    }


def _load_centerline_points(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise RuntimeError(f"river_v2_wse_centerline_missing:{path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise RuntimeError("river_v2_wse_centerline_empty")
    if "station_m" not in gdf.columns:
        raise RuntimeError("river_v2_wse_missing_station_m")
    if "point_id" not in gdf.columns:
        gdf["point_id"] = [f"pt_{idx}" for idx in range(len(gdf))]
    return gdf


def _sample_local_raster_support(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    *,
    radius_m: float = 36.0,
    quantile: float = 0.25,
    adaptive_radius_steps: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0),
) -> pd.DataFrame:
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1, masked=False).astype(float)
        nodata = ds.nodata
        if nodata is not None:
            if isinstance(nodata, float) and np.isnan(nodata):
                arr[~np.isfinite(arr)] = np.nan
            else:
                arr[np.isclose(arr, float(nodata))] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        rows, cols = np.where(np.isfinite(arr))
        if rows.size == 0:
            return pd.DataFrame({
                "candidate_count": np.zeros((len(gdf),), dtype=int),
                "wse_raw_local_z_m": np.full((len(gdf),), np.nan),
            })
        xs, ys = rasterio.transform.xy(ds.transform, rows, cols, offset="center")
        edge_xy = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
        edge_vals = arr[rows, cols].astype(float)

        station_gdf = gdf.copy()
        raster_crs = ds.crs
        metric_edge_xy = edge_xy
        metric_station_xy = np.column_stack([station_gdf.geometry.x.to_numpy(dtype=float), station_gdf.geometry.y.to_numpy(dtype=float)])
        metric_radius = max(float(radius_m or 0.0), 1.0)
        if raster_crs is not None:
            try:
                crs = CRS.from_user_input(raster_crs)
                if station_gdf.crs is not None and str(station_gdf.crs) != str(raster_crs):
                    station_gdf = station_gdf.to_crs(raster_crs)
                    metric_station_xy = np.column_stack([station_gdf.geometry.x.to_numpy(dtype=float), station_gdf.geometry.y.to_numpy(dtype=float)])
                if crs.is_geographic:
                    metric_crs = CRS.from_epsg(3857)
                    tx = Transformer.from_crs(crs, metric_crs, always_xy=True)
                    ex, ey = tx.transform(edge_xy[:, 0], edge_xy[:, 1])
                    sx, sy = tx.transform(metric_station_xy[:, 0], metric_station_xy[:, 1])
                    metric_edge_xy = np.column_stack([np.asarray(ex, dtype=float), np.asarray(ey, dtype=float)])
                    metric_station_xy = np.column_stack([np.asarray(sx, dtype=float), np.asarray(sy, dtype=float)])
            except Exception:
                pass
        tree = cKDTree(metric_edge_xy)
        out = {
            "candidate_count": np.zeros((len(station_gdf),), dtype=int),
            "wse_raw_local_z_m": np.full((len(station_gdf),), np.nan),
        }
        for i, point_xy in enumerate(metric_station_xy):
            idx: list[int] = []
            for scale in adaptive_radius_steps:
                idx = tree.query_ball_point(point_xy, metric_radius * float(scale))
                if idx:
                    break
            if not idx:
                continue
            candidates = edge_vals[np.asarray(idx, dtype=int)]
            finite = candidates[np.isfinite(candidates)]
            if finite.size == 0:
                continue
            out["candidate_count"][i] = int(finite.size)
            out["wse_raw_local_z_m"][i] = float(np.nanquantile(finite, quantile))
        return pd.DataFrame(out)


def _coerce_numeric_series(frame: pd.DataFrame, field_candidates: tuple[str, ...]) -> tuple[np.ndarray | None, str | None]:
    for field in field_candidates:
        if field in frame.columns:
            vals = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
            if np.any(np.isfinite(vals)):
                return vals, field
    return None, None


def _join_profile_values(centerline: gpd.GeoDataFrame, profile_path: Path) -> tuple[np.ndarray | None, str | None]:
    if not profile_path.exists() or profile_path.suffix.lower() not in {".csv", ".txt"}:
        return None, None
    table = pd.read_csv(profile_path)
    if table.empty:
        return None, None
    station_field = "station_m" if "station_m" in table.columns else None
    if station_field is None:
        for cand in ("centerline_station_m", "station", "s_m"):
            if cand in table.columns:
                station_field = cand
                break
    if station_field is None:
        return None, None
    value_field = None
    for cand in ("bank_wse_proxy_monotone_m", "bank_wse_proxy_interp_m", "bank_wse_proxy_raw_m", "wse_proxy_z_m"):
        if cand in table.columns:
            value_field = cand
            break
    if value_field is None:
        return None, None
    sta = pd.to_numeric(table[station_field], errors="coerce").to_numpy(dtype=float)
    vals = pd.to_numeric(table[value_field], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(sta) & np.isfinite(vals)
    if np.count_nonzero(finite) == 0:
        return None, None
    order = np.argsort(sta[finite], kind="mergesort")
    sta_f = sta[finite][order]
    vals_f = vals[finite][order]
    sta_u, idx_u = np.unique(sta_f, return_index=True)
    vals_u = vals_f[idx_u]
    target_sta = pd.to_numeric(centerline["station_m"], errors="coerce").to_numpy(dtype=float)
    out = np.full(target_sta.shape, np.nan, dtype=float)
    finite_target = np.isfinite(target_sta)
    if sta_u.size == 1:
        out[finite_target] = vals_u[0]
    else:
        out[finite_target] = np.interp(target_sta[finite_target], sta_u, vals_u)
    return out, f"profile_csv:{profile_path.name}:{value_field}"


def _group_keys(frame: pd.DataFrame) -> list[str]:
    return [field for field in _GROUP_FIELD_CANDIDATES if field in frame.columns]


def _fill_linear_by_station(station: np.ndarray, values: np.ndarray) -> np.ndarray:
    sta = np.asarray(station, dtype=float)
    vals = np.asarray(values, dtype=float)
    out = vals.copy()
    finite = np.isfinite(sta) & np.isfinite(vals)
    if np.count_nonzero(finite) == 0:
        return out
    if np.count_nonzero(finite) == 1:
        out[np.isfinite(sta)] = vals[finite][0]
        return out
    sta_f = sta[finite]
    vals_f = vals[finite]
    order = np.argsort(sta_f, kind="mergesort")
    sta_u, idx_u = np.unique(sta_f[order], return_index=True)
    vals_u = vals_f[order][idx_u]
    fill_mask = np.isfinite(sta)
    out[fill_mask] = np.interp(sta[fill_mask], sta_u, vals_u)
    return out




def _reduce_support_anchor_runs(station: np.ndarray, support: np.ndarray, *, tol: float = 1.0e-4) -> tuple[np.ndarray, np.ndarray]:
    sta = np.asarray(station, dtype=float)
    vals = np.asarray(support, dtype=float)
    finite = np.isfinite(sta) & np.isfinite(vals)
    if np.count_nonzero(finite) <= 1:
        return sta[finite], vals[finite]
    sta_f = sta[finite]
    vals_f = vals[finite]
    order = np.argsort(sta_f, kind="mergesort")
    sta_o = sta_f[order]
    vals_o = vals_f[order]
    out_sta: list[float] = []
    out_vals: list[float] = []
    start = 0
    n = len(sta_o)
    while start < n:
        end = start + 1
        while end < n and np.isfinite(vals_o[end]) and abs(vals_o[end] - vals_o[start]) <= tol:
            end += 1
        run_sta = sta_o[start:end]
        run_vals = vals_o[start:end]
        out_sta.append(float(np.nanmean(run_sta)))
        out_vals.append(float(np.nanmean(run_vals)))
        start = end
    return np.asarray(out_sta, dtype=float), np.asarray(out_vals, dtype=float)

def _rolling_nanmean(values: np.ndarray, half_window: int = 1) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    for i in range(len(vals)):
        lo = max(0, i - int(half_window))
        hi = min(len(vals), i + int(half_window) + 1)
        window = vals[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[i] = float(np.nanmean(finite))
    return out


def _rolling_nanmedian(values: np.ndarray, half_window: int = 1) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    for i in range(len(vals)):
        lo = max(0, i - int(half_window))
        hi = min(len(vals), i + int(half_window) + 1)
        window = vals[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[i] = float(np.nanmedian(finite))
    return out


def _rolling_nanquantile(values: np.ndarray, q: float, half_window: int = 1) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    q = float(min(1.0, max(0.0, q)))
    for i in range(len(vals)):
        lo = max(0, i - int(half_window))
        hi = min(len(vals), i + int(half_window) + 1)
        window = vals[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[i] = float(np.nanquantile(finite, q))
    return out


def _project_monotone(values: np.ndarray, *, mode: str, min_step: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=float)
    out = vals.copy()
    adjustment = np.zeros(vals.shape, dtype=float)
    last = np.nan
    for i in range(len(out)):
        val = out[i]
        if not np.isfinite(val):
            continue
        if not np.isfinite(last):
            last = val
            continue
        if mode == "nonincreasing":
            target = last - float(min_step)
            if val > target:
                out[i] = target
                adjustment[i] = target - val
                last = target
            else:
                last = val
        else:
            target = last + float(min_step)
            if val < target:
                out[i] = target
                adjustment[i] = target - val
                last = target
            else:
                last = val
    return out, adjustment


def _resolve_component_station_direction(station: np.ndarray, support: np.ndarray, support_type: np.ndarray) -> str:
    sta = np.asarray(station, dtype=float)
    vals = np.asarray(support, dtype=float)
    stype = np.asarray(support_type, dtype=object)
    direct_mask = np.isfinite(sta) & np.isfinite(vals) & (stype == "direct_local_support")
    use_mask = direct_mask if np.count_nonzero(direct_mask) >= 2 else (np.isfinite(sta) & np.isfinite(vals))
    if np.count_nonzero(use_mask) < 2:
        return "downstream_increasing"
    vals_use = vals[use_mask]
    dec_adj = float(np.nansum(np.abs(_project_monotone(vals_use, mode="nonincreasing", min_step=0.0)[1])))
    inc_adj = float(np.nansum(np.abs(_project_monotone(vals_use, mode="nondecreasing", min_step=0.0)[1])))
    return "downstream_increasing" if dec_adj <= inc_adj else "upstream_increasing"


def _monotone_mode_from_station_direction(station_direction: str) -> str:
    return "nonincreasing" if str(station_direction) == "downstream_increasing" else "nondecreasing"


def _count_direction_violations(values: np.ndarray, *, mode: str) -> int:
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size < 2:
        return 0
    diffs = np.diff(finite)
    if mode == "nonincreasing":
        return int(np.count_nonzero(diffs > 1.0e-9))
    return int(np.count_nonzero(diffs < -1.0e-9))


def _contiguous_false_runs(mask: np.ndarray) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    current: list[int] = []
    for i, flag in enumerate(mask.astype(bool)):
        if not flag:
            current.append(i)
        elif current:
            runs.append(np.asarray(current, dtype=int))
            current = []
    if current:
        runs.append(np.asarray(current, dtype=int))
    return runs


def _series_negative_step_count(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if np.count_nonzero(finite) < 2:
        return 0
    diffs = np.diff(vals[finite])
    return int(np.count_nonzero(diffs < -1.0e-9))


def _apply_profile_edge_seeds(
    station: np.ndarray,
    direct_support: np.ndarray,
    profile_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(direct_support, dtype=float).copy()
    support_type = np.full(support.shape, 'no_support', dtype=object)
    direct_mask = np.isfinite(support)
    support_type[direct_mask] = 'direct_local_support'
    if not np.isfinite(profile_vals).any():
        return support, support_type
    finite_station = np.isfinite(station)
    if not finite_station.any():
        return support, support_type
    valid_idx = np.where(finite_station)[0]
    direct_idx = np.where(np.isfinite(support) & finite_station)[0]
    if direct_idx.size == 0:
        start_idx = int(valid_idx[0])
        end_idx = int(valid_idx[-1])
        if np.isfinite(profile_vals[start_idx]):
            support[start_idx] = float(profile_vals[start_idx])
            support_type[start_idx] = 'profile_edge_seed'
        if end_idx != start_idx and np.isfinite(profile_vals[end_idx]):
            support[end_idx] = float(profile_vals[end_idx])
            support_type[end_idx] = 'profile_edge_seed'
        return support, support_type
    first_direct = int(direct_idx[0])
    last_direct = int(direct_idx[-1])
    if first_direct > valid_idx[0]:
        start_idx = int(valid_idx[0])
        if np.isfinite(profile_vals[start_idx]):
            support[start_idx] = float(profile_vals[start_idx])
            support_type[start_idx] = 'profile_edge_seed'
    if last_direct < valid_idx[-1]:
        end_idx = int(valid_idx[-1])
        if np.isfinite(profile_vals[end_idx]):
            support[end_idx] = float(profile_vals[end_idx])
            support_type[end_idx] = 'profile_edge_seed'
    return support, support_type


def _apply_station_gradient_floor(values: np.ndarray, station: np.ndarray, *, mode: str, slope_floor_m_per_m: float = 8.0e-5) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=float).copy()
    sta = np.asarray(station, dtype=float)
    adj = np.zeros(vals.shape, dtype=float)
    if vals.size == 0:
        return vals, adj
    for i in range(1, len(vals)):
        if not (np.isfinite(vals[i-1]) and np.isfinite(vals[i]) and np.isfinite(sta[i-1]) and np.isfinite(sta[i])):
            continue
        delta_sta = max(float(sta[i] - sta[i-1]), 0.0)
        min_step = float(slope_floor_m_per_m) * delta_sta
        if mode == "nonincreasing":
            target = vals[i-1] - min_step
            if vals[i] > target:
                adj[i] += target - vals[i]
                vals[i] = target
        else:
            target = vals[i-1] + min_step
            if vals[i] < target:
                adj[i] += target - vals[i]
                vals[i] = target
    return vals, adj


def _finalize_component(final_vals: np.ndarray, has_support: np.ndarray, station: np.ndarray, *, station_direction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    final = np.asarray(final_vals, dtype=float).copy()
    station_vals = np.asarray(station, dtype=float)
    reason = np.full(final.shape, "none", dtype=object)
    adj_total = np.zeros(final.shape, dtype=float)

    if final.size == 0:
        return final, adj_total, reason

    if not np.all(np.isfinite(station_vals)):
        station_vals = np.arange(final.size, dtype=float)

    filled = _fill_linear_by_station(station_vals, final)
    fill_mask = ~np.isfinite(final) & np.isfinite(filled)
    if np.any(fill_mask):
        final[fill_mask] = filled[fill_mask]
        reason[fill_mask] = "tiny_gap_cleanup"

    support_mask = np.asarray(has_support, dtype=bool) & np.isfinite(final)
    support_idx = np.where(support_mask)[0]
    monotone_mode = _monotone_mode_from_station_direction(station_direction)
    if support_idx.size == 0:
        projected, proj_adj = _project_monotone(final, mode=monotone_mode, min_step=0.0)
        proj_mask = np.abs(proj_adj) > 0.0
        if np.any(proj_mask):
            final[proj_mask] = projected[proj_mask]
            adj_total[proj_mask] += proj_adj[proj_mask]
            reason[proj_mask] = "tiny_reversal_cleanup"
        return final, adj_total, reason

    support_vals = final[support_idx]
    projected_support, support_adj = _project_monotone(support_vals, mode=monotone_mode, min_step=0.0)
    support_adj_mask = np.abs(support_adj) > 0.0
    if np.any(support_adj_mask):
        support_fix_idx = support_idx[support_adj_mask]
        final[support_fix_idx] = projected_support[support_adj_mask]
        adj_total[support_fix_idx] += support_adj[support_adj_mask]
        reason[support_fix_idx] = "tiny_reversal_cleanup"

    unsupported_runs = _contiguous_false_runs(np.asarray(has_support, dtype=bool))
    slope_floor_m_per_m = 8.0e-5
    for run in unsupported_runs:
        run_station = station_vals[run]
        forced = np.full(run.shape, np.nan, dtype=float)
        left_idx = int(run[0] - 1) if run[0] > 0 else None
        right_idx = int(run[-1] + 1) if run[-1] < final.size - 1 else None
        left_ok = left_idx is not None and bool(support_mask[left_idx])
        right_ok = right_idx is not None and bool(support_mask[right_idx])
        if left_ok and right_ok:
            forced = np.interp(
                run_station,
                [float(station_vals[left_idx]), float(station_vals[right_idx])],
                [float(final[left_idx]), float(final[right_idx])],
            )
            reason_label = "unsupported_span_bridge"
        elif left_ok:
            anchor_sta = float(station_vals[left_idx])
            anchor_val = float(final[left_idx])
            if monotone_mode == "nonincreasing":
                forced = anchor_val - slope_floor_m_per_m * (run_station - anchor_sta)
            else:
                forced = anchor_val + slope_floor_m_per_m * (run_station - anchor_sta)
            reason_label = "unsupported_span_gradient_floor"
        elif right_ok:
            anchor_sta = float(station_vals[right_idx])
            anchor_val = float(final[right_idx])
            if monotone_mode == "nonincreasing":
                forced = anchor_val + slope_floor_m_per_m * (anchor_sta - run_station)
            else:
                forced = anchor_val - slope_floor_m_per_m * (anchor_sta - run_station)
            reason_label = "unsupported_span_gradient_floor"
        else:
            continue
        delta = forced - final[run]
        use = np.isfinite(delta) & (np.abs(delta) > 0.0)
        if np.any(use):
            idx = run[use]
            final[idx] = forced[use]
            adj_total[idx] += delta[use]
            reason[idx] = reason_label

    projected, proj_adj = _project_monotone(final, mode=monotone_mode, min_step=0.0)
    unsupported_finite = (~support_mask) & np.isfinite(final)
    proj_mask = unsupported_finite & (np.abs(proj_adj) > 0.0)
    if np.any(proj_mask):
        final[proj_mask] = projected[proj_mask]
        adj_total[proj_mask] += proj_adj[proj_mask]
        preserve_bridge = reason[proj_mask] == "unsupported_span_bridge"
        preserve_floor = reason[proj_mask] == "unsupported_span_gradient_floor"
        reason[proj_mask] = np.where(
            preserve_bridge,
            "unsupported_span_bridge",
            np.where(preserve_floor, "unsupported_span_gradient_floor", "tiny_reversal_cleanup"),
        )

    grad_final, grad_adj = _apply_station_gradient_floor(final, station_vals, mode=monotone_mode, slope_floor_m_per_m=8.0e-5)
    grad_mask = np.isfinite(grad_adj) & (np.abs(grad_adj) > 0.0)
    if np.any(grad_mask):
        final[grad_mask] = grad_final[grad_mask]
        adj_total[grad_mask] += grad_adj[grad_mask]
        preserve_bridge = reason[grad_mask] == "unsupported_span_bridge"
        preserve_floor = reason[grad_mask] == "unsupported_span_gradient_floor"
        reason[grad_mask] = np.where(
            preserve_bridge,
            "unsupported_span_bridge",
            np.where(preserve_floor, "unsupported_span_gradient_floor", "gradient_floor_cleanup"),
        )
    return final, adj_total, reason


def _write_gpkg(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG")
    return path


def _summary_stats(values: np.ndarray) -> dict[str, float | int | None]:
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {"finite_count": 0, "dynamic_range_m": None}
    return {
        "finite_count": int(finite.size),
        "dynamic_range_m": float(np.nanmax(finite) - np.nanmin(finite)),
    }


def _direction_summary(frame: pd.DataFrame, value_field: str) -> dict[str, Any]:
    vals = pd.to_numeric(frame[value_field], errors="coerce").to_numpy(dtype=float) if value_field in frame.columns else np.full((len(frame),), np.nan)
    summary = _summary_stats(vals)
    group_fields = _group_keys(frame)
    group_iter = [(None, frame)] if not group_fields else frame.groupby(group_fields, dropna=False, sort=False)
    violation_count = 0
    station_direction_counts: dict[str, int] = {}
    for _, grp in group_iter:
        grp_vals = pd.to_numeric(grp[value_field], errors="coerce").to_numpy(dtype=float)
        station_direction = str(grp["station_direction"].iloc[0]) if "station_direction" in grp.columns else "downstream_increasing"
        station_direction_counts[station_direction] = station_direction_counts.get(station_direction, 0) + 1
        violation_count += _count_direction_violations(grp_vals, mode=_monotone_mode_from_station_direction(station_direction))
    summary["direction_violation_count"] = int(violation_count)
    if station_direction_counts:
        summary["station_direction_counts"] = station_direction_counts
    return summary


def build_wse_support(
    ctx: RiverV2Context,
    centerline: gpd.GeoDataFrame,
    aux_outputs: dict[str, str],
) -> tuple[gpd.GeoDataFrame, Path, dict[str, Any]]:
    work = centerline.copy()
    n = len(work)
    station = pd.to_numeric(work["station_m"], errors="coerce").to_numpy(dtype=float)
    direct_support = np.full((n,), np.nan, dtype=float)
    support_type = np.full((n,), "no_support", dtype=object)
    support_strength = np.zeros((n,), dtype=float)
    wse_source = np.full((n,), "none", dtype=object)

    edge_path = aux_outputs.get("bank_wse_edge_guidance_path")
    if edge_path and Path(edge_path).exists():
        local_support = _sample_local_raster_support(work, Path(edge_path), radius_m=36.0, quantile=0.25)
        local_vals = pd.to_numeric(local_support.get("wse_raw_local_z_m"), errors="coerce").to_numpy(dtype=float)
        local_hits = np.isfinite(local_vals)
        direct_support[local_hits] = local_vals[local_hits]
        support_type[local_hits] = "direct_local_support"
        support_strength[local_hits] = 1.0
        wse_source[local_hits] = f"edge_raster_local_quantile:{Path(edge_path).name}"

    combined_vals, combined_field = _coerce_numeric_series(work, _COMBINED_BANK_FIELD_CANDIDATES)
    if combined_vals is not None:
        use_combined = ~np.isfinite(direct_support) & np.isfinite(combined_vals)
        direct_support[use_combined] = combined_vals[use_combined]
        support_type[use_combined] = "direct_local_support"
        support_strength[use_combined] = 0.85
        wse_source[use_combined] = f"centerline_field:{combined_field}"

    support = direct_support.copy()

    work["station_direction"] = "downstream_increasing"
    group_fields = _group_keys(work)
    group_iter = [(None, work)] if not group_fields else work.groupby(group_fields, dropna=False, sort=False)
    for _, grp in group_iter:
        idx = grp.index.to_numpy(dtype=int)
        work.loc[idx, "station_direction"] = _resolve_component_station_direction(station[idx], support[idx], support_type[idx])

    work["wse_support_z_m"] = support
    work["support_type"] = pd.Series(support_type, dtype="object")
    work["has_support"] = np.isfinite(support)

    keep = [c for c in [*_identity_columns(work), "wse_support_z_m", "support_type", "has_support"] if c in work.columns]
    support_gdf = gpd.GeoDataFrame(work[keep].copy(), geometry="geometry", crs=centerline.crs)
    support_path = _write_gpkg(support_gdf, ctx.paths.centerline_wse_support_points)
    support_type_vals = support_gdf["support_type"].astype(str).to_numpy(dtype=object)
    summary = {
        "supported_station_count": int(np.count_nonzero(support_gdf["has_support"].to_numpy(dtype=bool))),
        "unsupported_station_count": int(len(support_gdf) - np.count_nonzero(support_gdf["has_support"].to_numpy(dtype=bool))),
        "direct_local_support_count": int(np.count_nonzero(support_type_vals == "direct_local_support")),

    }
    return support_gdf, support_path, summary


def build_wse_trend(
    ctx: RiverV2Context,
    support_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, Path, dict[str, Any]]:
    work = support_gdf.copy()
    work["wse_anchor_raw_z_m"] = np.nan
    work["wse_interp_z_m"] = np.nan
    work["wse_trend_z_m"] = np.nan
    collapsed_runs_total = 0
    anchor_count_total = 0
    group_fields = _group_keys(work)
    group_iter = [(None, work)] if not group_fields else work.groupby(group_fields, dropna=False, sort=False)
    for _, grp in group_iter:
        idx = grp.index.to_numpy(dtype=int)
        sta = pd.to_numeric(work.loc[idx, "station_m"], errors="coerce").to_numpy(dtype=float)
        support = pd.to_numeric(work.loc[idx, "wse_support_z_m"], errors="coerce").to_numpy(dtype=float)
        support_type = work.loc[idx, "support_type"].astype(str).to_numpy(dtype=object)
        anchor_raw = support.copy()
        interpolated = np.full(sta.shape, np.nan, dtype=float)
        trend = np.full(sta.shape, np.nan, dtype=float)
        finite_station = np.isfinite(sta)
        support_mask = finite_station & np.isfinite(support)
        if np.count_nonzero(support_mask) > 0:
            station_direction = _resolve_component_station_direction(sta, support, support_type)
            monotone_mode = _monotone_mode_from_station_direction(station_direction)
            work.loc[idx, "station_direction"] = station_direction
            reduced_sta, reduced_support = _reduce_support_anchor_runs(sta[support_mask], support[support_mask])
            anchor_count_total += int(reduced_support.size)
            collapsed_runs_total += int(np.count_nonzero(support_mask) - reduced_support.size)
            if reduced_support.size == 1:
                interpolated[finite_station] = float(reduced_support[0])
            elif reduced_support.size > 1:
                interpolated[finite_station] = np.interp(sta[finite_station], reduced_sta, reduced_support)
            if reduced_support.size > 0:
                nearest_idx = np.full(reduced_support.shape, -1, dtype=int)
                finite_idx = np.where(finite_station)[0]
                if finite_idx.size:
                    nearest_idx = finite_idx[np.abs(sta[finite_idx][:, None] - reduced_sta[None, :]).argmin(axis=0)]
                    anchor_raw[:] = np.nan
                    anchor_raw[nearest_idx] = reduced_support
            if np.isfinite(interpolated).any():
                q = 0.25 if monotone_mode == "nonincreasing" else 0.75
                envelope = _rolling_nanquantile(interpolated, q=q, half_window=4)
                smoothed = _rolling_nanmean(envelope, half_window=2)
                monotone, _ = _project_monotone(smoothed, mode=monotone_mode, min_step=0.0)
                trend[finite_station] = monotone[finite_station]
        work.loc[idx, "wse_anchor_raw_z_m"] = anchor_raw
        work.loc[idx, "wse_interp_z_m"] = interpolated
        work.loc[idx, "wse_trend_z_m"] = trend
    keep = [c for c in [*_identity_columns(work), "wse_support_z_m", "support_type", "has_support", "wse_anchor_raw_z_m", "wse_interp_z_m", "wse_trend_z_m"] if c in work.columns]
    trend_gdf = gpd.GeoDataFrame(work[keep].copy(), geometry="geometry", crs=support_gdf.crs)
    trend_path = _write_gpkg(trend_gdf, ctx.paths.centerline_wse_trend_points)
    summary = _direction_summary(trend_gdf, "wse_trend_z_m")
    summary["sparse_anchor_count"] = int(anchor_count_total)
    summary["collapsed_support_run_count"] = int(collapsed_runs_total)
    return trend_gdf, trend_path, summary


def build_wse_pre_smooth(
    ctx: RiverV2Context,
    trend_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, Path, dict[str, Any]]:
    work = trend_gdf.copy()
    trend = pd.to_numeric(work["wse_trend_z_m"], errors="coerce").to_numpy(dtype=float)
    support = pd.to_numeric(work["wse_support_z_m"], errors="coerce").to_numpy(dtype=float)
    support_type = work["support_type"].astype(str).to_numpy(dtype=object)
    alpha = np.where(support_type == "direct_local_support", 0.10, 0.0).astype(float)
    tolerance = np.where(support_type == "direct_local_support", 0.20, 0.0).astype(float)
    delta = support - trend
    pre = trend.copy()
    use = np.isfinite(trend) & np.isfinite(support) & (alpha > 0.0) & np.isfinite(delta) & (np.abs(delta) <= tolerance)
    pre[use] = trend[use] + alpha[use] * delta[use]
    work["wse_pre_smooth_z_m"] = pre
    keep = [c for c in [*_identity_columns(work), "wse_support_z_m", "support_type", "has_support", "wse_anchor_raw_z_m", "wse_interp_z_m", "wse_trend_z_m", "wse_pre_smooth_z_m"] if c in work.columns]
    pre_gdf = gpd.GeoDataFrame(work[keep].copy(), geometry="geometry", crs=trend_gdf.crs)
    pre_path = _write_gpkg(pre_gdf, ctx.paths.centerline_wse_pre_smooth_points)
    summary = _direction_summary(pre_gdf, "wse_pre_smooth_z_m")
    return pre_gdf, pre_path, summary


def build_wse_proxy_final(
    ctx: RiverV2Context,
    pre_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, Path, dict[str, Any]]:
    work = pre_gdf.copy()
    work["wse_proxy_z_m"] = np.nan
    work["final_adjustment_m"] = 0.0
    work["final_adjustment_reason"] = "none"
    group_fields = _group_keys(work)
    group_iter = [(None, work)] if not group_fields else work.groupby(group_fields, dropna=False, sort=False)
    for _, grp in group_iter:
        idx = grp.index.to_numpy(dtype=int)
        sta = pd.to_numeric(work.loc[idx, "station_m"], errors="coerce").to_numpy(dtype=float)
        pre = pd.to_numeric(work.loc[idx, "wse_pre_smooth_z_m"], errors="coerce").to_numpy(dtype=float)
        has_support = work.loc[idx, "has_support"].astype(bool).to_numpy(dtype=bool)
        station_direction = str(work.loc[idx, "station_direction"].iloc[0]) if "station_direction" in work.columns else "downstream_increasing"
        final, adj, reason = _finalize_component(pre, has_support, sta, station_direction=station_direction)
        work.loc[idx, "wse_proxy_z_m"] = final
        work.loc[idx, "final_adjustment_m"] = adj
        work.loc[idx, "final_adjustment_reason"] = reason
    work["wse_valid"] = np.isfinite(pd.to_numeric(work["wse_proxy_z_m"], errors="coerce").to_numpy(dtype=float))
    work["wse_quality_flag"] = np.where(
        work["wse_valid"].to_numpy(dtype=bool),
        np.where(work["has_support"].to_numpy(dtype=bool), "supported", "trend_gap_filled"),
        "invalid",
    )
    keep = [c for c in [*_identity_columns(work), "wse_support_z_m", "support_type", "has_support", "wse_anchor_raw_z_m", "wse_interp_z_m", "wse_trend_z_m", "wse_pre_smooth_z_m", "wse_proxy_z_m", "final_adjustment_m", "final_adjustment_reason", "wse_valid", "wse_quality_flag"] if c in work.columns]
    final_gdf = gpd.GeoDataFrame(work[keep].copy(), geometry="geometry", crs=pre_gdf.crs)
    write_cols = [c for c in [*_identity_columns(final_gdf), "wse_proxy_z_m", "has_support", "wse_valid"] if c in final_gdf.columns]
    persisted_gdf = gpd.GeoDataFrame(final_gdf[write_cols].copy(), geometry="geometry", crs=pre_gdf.crs)
    final_path = _write_gpkg(persisted_gdf, ctx.paths.centerline_wse_proxy_points)
    adjustments = np.abs(pd.to_numeric(final_gdf["final_adjustment_m"], errors="coerce").to_numpy(dtype=float))
    support_mask = final_gdf["has_support"].astype(bool).to_numpy(dtype=bool) if "has_support" in final_gdf.columns else np.zeros((len(final_gdf),), dtype=bool)
    adjusted_mask = np.isfinite(adjustments) & (adjustments > 0.0)
    summary = {
        **_direction_summary(final_gdf, "wse_proxy_z_m"),
        "adjusted_station_count": int(np.count_nonzero(adjusted_mask)),
        "adjusted_supported_station_count": int(np.count_nonzero(adjusted_mask & support_mask)),
        "adjusted_unsupported_station_count": int(np.count_nonzero(adjusted_mask & ~support_mask)),
        "max_abs_final_adjustment_m": float(np.nanmax(adjustments)) if np.isfinite(adjustments).any() else 0.0,
        "mean_abs_final_adjustment_m": float(np.nanmean(adjustments[np.isfinite(adjustments)])) if np.isfinite(adjustments).any() else 0.0,
    }
    return final_gdf, final_path, summary


def _write_stage_summary_text(path: Path, diagnostics: dict[str, Any]) -> None:
    lines = ["River v2 WSE stage summary", ""]
    for step_name in ("support", "trend", "pre_smooth", "final"):
        step = diagnostics.get(step_name, {}) or {}
        lines.append(f"[{step_name}]")
        for key, value in step.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _validate_wse_output(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    missing = [field for field in _REQUIRED_OUTPUT_FIELDS if field not in gdf.columns]
    vals = pd.to_numeric(gdf["wse_proxy_z_m"], errors="coerce").to_numpy(dtype=float) if "wse_proxy_z_m" in gdf.columns else np.full((len(gdf),), np.nan)
    finite = int(np.count_nonzero(np.isfinite(vals)))
    monotone_violation_count = 0
    group_fields = _group_keys(gdf)
    group_iter = [(None, gdf)] if not group_fields else gdf.groupby(group_fields, dropna=False, sort=False)
    for _, grp in group_iter:
        group_vals = pd.to_numeric(grp["wse_proxy_z_m"], errors="coerce").to_numpy(dtype=float)
        station_direction = str(grp["station_direction"].iloc[0]) if "station_direction" in grp.columns else "downstream_increasing"
        monotone_violation_count += _count_direction_violations(group_vals, mode=_monotone_mode_from_station_direction(station_direction))
    return {
        "valid": not missing and finite > 0 and monotone_violation_count == 0,
        "missing_required_fields": missing,
        "finite_wse_count": finite,
        "monotone_violation_count": int(monotone_violation_count),
    }


def run_wse_proxy_stage(
    ctx: RiverV2Context,
    *,
    centerline_points_path: Path,
    bank_wse_edge_guidance_path: Path | None,
    bank_wse_profile_summary_path: Path | None = None,
) -> RiverV2StageResult:
    aux_outputs = {}
    if bank_wse_edge_guidance_path is not None:
        aux_outputs["bank_wse_edge_guidance_path"] = str(bank_wse_edge_guidance_path)
    if bank_wse_profile_summary_path is not None:
        aux_outputs["bank_wse_profile_summary_path"] = str(bank_wse_profile_summary_path)
    centerline = _load_centerline_points(centerline_points_path)
    support_gdf, support_path, support_summary = build_wse_support(ctx, centerline, aux_outputs)
    trend_gdf, trend_path, trend_summary = build_wse_trend(ctx, support_gdf)
    pre_gdf, pre_path, pre_summary = build_wse_pre_smooth(ctx, trend_gdf)
    final_gdf, final_path, final_summary = build_wse_proxy_final(ctx, pre_gdf)

    validation = _validate_wse_output(final_gdf)
    diagnostics = {
        "stage_id": STAGE_CENTERLINE_WSE_PROXY,
        "source_logic": "modified_bank_guidance:support_to_trend_to_pre_smooth_to_final",
        "support": support_summary,
        "trend": trend_summary,
        "pre_smooth": pre_summary,
        "final": final_summary,
        "artifacts": {
            "support_path": str(support_path),
            "trend_path": str(trend_path),
            "pre_smooth_path": str(pre_path),
            "final_wse_path": str(final_path),
        },
        "validation": validation,
        "warnings": [],
    }
    ctx.paths.centerline_wse_stage_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    ctx.paths.centerline_wse_stage_diagnostics.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    _write_stage_summary_text(ctx.paths.centerline_wse_stage_summary, diagnostics)

    receipt = build_river_v2_stage_receipt(
        stage_id=STAGE_CENTERLINE_WSE_PROXY,
        output_artifact=str(final_path),
        input_artifacts=ctx.direct_stage_input_artifacts(
            centerline_points_path,
            aux_outputs.get("bank_wse_edge_guidance_path"),
            aux_outputs.get("bank_wse_profile_summary_path"),
        ),
        record_count=int(len(final_gdf)),
        field_schema=wse_proxy_field_schema(),
        vertical_reference=ctx.vertical_reference,
        warnings=[],
        source_logic="modified_bank_guidance:support_to_trend_to_pre_smooth_to_final",
        validation=validation,
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.centerline_wse_proxy_points_receipt)
    aux = dict(aux_outputs)
    aux.update(
        {
            "centerline_wse_support_points": str(support_path),
            "centerline_wse_trend_points": str(trend_path),
            "centerline_wse_pre_smooth_points": str(pre_path),
            "centerline_wse_stage_diagnostics": str(ctx.paths.centerline_wse_stage_diagnostics),
            "centerline_wse_stage_summary": str(ctx.paths.centerline_wse_stage_summary),
        }
    )
    return RiverV2StageResult(
        stage_id=STAGE_CENTERLINE_WSE_PROXY,
        output_artifact=final_path,
        receipt_path=receipt_path,
        record_count=int(len(final_gdf)),
        validation=dict(validation or {}),
        warnings=[],
        aux_outputs=aux,
    )
