from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio

log = logging.getLogger(__name__)


def _sample_raster_at_points(gdf, raster_path: Optional[str | Path], field: str) -> pd.Series:
    out = pd.Series(np.nan, index=gdf.index, dtype="float32")
    if raster_path is None:
        return out
    path = Path(raster_path)
    if not path.exists() or gdf is None or getattr(gdf, "empty", True):
        return out
    with rasterio.open(path) as ds:
        pts = list(zip(gdf.geometry.x.to_numpy(dtype=float), gdf.geometry.y.to_numpy(dtype=float)))
        vals = []
        for v in ds.sample(pts):
            val = float(v[0]) if np.size(v) else np.nan
            if ds.nodata is not None and np.isfinite(val) and np.isclose(val, float(ds.nodata)):
                val = np.nan
            vals.append(val)
    return pd.Series(np.asarray(vals, dtype=np.float32), index=gdf.index, name=field)


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


def _build_profile_dataframe(centerline_points, sampled: pd.DataFrame) -> pd.DataFrame:
    df = centerline_points.copy()
    df["profile_id"] = _infer_profile_id(df)
    df["station_m"] = pd.to_numeric(df.get("station_m"), errors="coerce")
    df["centerline_elevation_m"] = pd.to_numeric(sampled["centerline_elevation_m"], errors="coerce")
    df["centerline_influence"] = pd.to_numeric(sampled["centerline_influence"], errors="coerce").fillna(0.0)
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
            xs_w = np.clip(sg["xs_support_weight"].to_numpy(dtype=float), 0.0, 1.0)
            bank_w = np.clip(sg["bank_weight"].to_numpy(dtype=float), 0.0, 1.0)
            authoritative_anchor = sg["authoritative_anchor_elevation_m"].to_numpy(dtype=float)
            authoritative_support_mask = sg["authoritative_support_mask"].to_numpy(dtype=float)
            authoritative_present = np.any(np.isfinite(authoritative_anchor) & (authoritative_support_mask > 0.0))
            authoritative_anchor_value = float(np.nanmedian(authoritative_anchor[np.isfinite(authoritative_anchor) & (authoritative_support_mask > 0.0)])) if authoritative_present else np.nan
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
            comp_vals = np.array([
                authoritative_anchor_value,
                np.nanmedian(center) if np.any(np.isfinite(center)) else np.nan,
                np.nanmedian(xs) if np.any(np.isfinite(xs)) else np.nan,
                np.nanmedian(bank) if np.any(np.isfinite(bank)) else np.nan,
            ], dtype=float)
            comp_w = np.array([
                1.0 if authoritative_present else 0.0,
                float(np.nanmax(center_w)) if np.any(np.isfinite(center)) else 0.0,
                float(np.nanmax(xs_w)) if np.any(np.isfinite(xs)) else 0.0,
                float(np.nanmax(bank_w)) if np.any(np.isfinite(bank)) else 0.0,
            ], dtype=float)
            if authoritative_present:
                bed = authoritative_anchor_value
                spread = 0.0
                dominant_source = "authoritative_anchor"
            else:
                bed = _weighted_mean(comp_vals, comp_w)
                spread = _weighted_spread(comp_vals, np.where(np.isfinite(comp_vals), np.maximum(comp_w, 1e-6), 0.0), bed)
                dominant_source = ["authoritative_anchor", "centerline", "xs_support", "bank"][int(np.nanargmax(comp_w))] if np.any(comp_w > 0.0) else "none"
            rows.append({
                "profile_id": str(profile_id),
                "station_m": float(station_bin),
                "station_step_m": float(step),
                "authoritative_anchor_elevation_m": authoritative_anchor_value,
                "authoritative_anchor_present": bool(authoritative_present),
                "centerline_elevation_m": comp_vals[1],
                "xs_support_elevation_m": comp_vals[2],
                "bank_elevation_m": comp_vals[3],
                "authoritative_support_depth_m": float(np.nanmedian(sg["authoritative_support_depth_m"].to_numpy(dtype=float))) if np.any(np.isfinite(sg["authoritative_support_depth_m"].to_numpy(dtype=float))) else np.nan,
                "wse_elevation_m": wse_value,
                "wse_influence": float(np.nanmax(wse_influence)) if wse_present else 0.0,
                "wse_uncertainty_m": wse_uncertainty,
                "centerline_weight": comp_w[1],
                "xs_support_weight": comp_w[2],
                "bank_weight": comp_w[3],
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


def _solve_component_profile(sub: pd.DataFrame, *, lam_data: float = 1.0, lam_smooth: float = 0.40) -> Tuple[np.ndarray, np.ndarray]:
    vals = pd.to_numeric(sub["bed_elevation_m"], errors="coerce").to_numpy(dtype=float)
    unc = pd.to_numeric(sub["uncertainty_m"], errors="coerce").to_numpy(dtype=float)
    authoritative_anchor = pd.to_numeric(sub.get("authoritative_anchor_elevation_m"), errors="coerce").to_numpy(dtype=float) if "authoritative_anchor_elevation_m" in sub.columns else np.full(len(sub), np.nan, dtype=float)
    authoritative_present = sub.get("authoritative_anchor_present", pd.Series(False, index=sub.index)).fillna(False).to_numpy(dtype=bool) if "authoritative_anchor_present" in sub.columns else np.zeros(len(sub), dtype=bool)
    n = len(vals)
    if n == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    obs = vals.copy()
    if not np.any(np.isfinite(obs)):
        return np.full(n, np.nan, dtype=float), np.full(n, np.nan, dtype=float)
    mask = np.isfinite(obs)
    fill = np.interp(np.arange(n), np.where(mask)[0], obs[mask]) if np.sum(mask) >= 2 else np.where(mask, obs, np.nanmedian(obs[mask]))
    weights = np.where(np.isfinite(unc) & (unc > 0.0), 1.0 / np.square(np.maximum(unc, 0.05)), 0.0)
    weights = np.where(mask, np.maximum(weights, 0.25), 0.0)
    anchor_mask = authoritative_present & np.isfinite(authoritative_anchor)
    fill = np.where(anchor_mask, authoritative_anchor, fill)
    weights = np.where(anchor_mask, np.maximum(weights, 2500.0), weights)
    A = np.diag(weights * lam_data + 1e-6)
    b = (weights * lam_data) * fill
    if n > 1:
        for i in range(n - 1):
            A[i, i] += lam_smooth
            A[i + 1, i + 1] += lam_smooth
            A[i, i + 1] -= lam_smooth
            A[i + 1, i] -= lam_smooth
    try:
        solved = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        solved = fill
    delta = np.nanmedian(np.diff(fill)) if n > 1 else 0.0
    tol = 0.02
    if np.isfinite(delta) and delta <= 0.0:
        for i in range(1, n):
            solved[i] = min(solved[i], solved[i - 1] + tol)
    else:
        for i in range(1, n):
            solved[i] = max(solved[i], solved[i - 1] - tol)
    solved = np.where(anchor_mask, authoritative_anchor, solved)
    residual = np.abs(np.nan_to_num(solved - fill, nan=0.0))
    solved_unc = np.sqrt(np.square(np.nan_to_num(unc, nan=np.nanmedian(np.nan_to_num(unc, nan=0.25)))) + np.square(residual))
    solved_unc = np.maximum(solved_unc, 0.05)
    solved_unc = np.where(anchor_mask, 0.02, solved_unc)
    return solved.astype(float), solved_unc.astype(float)


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
        solved, solved_unc = _solve_component_profile(sub)
        comp_meta = hierarchy_meta.get(str(comp), {})
        hierarchy_priority = float(comp_meta.get("hierarchy_priority", 1.0))
        sub["drainage_area_proxy"] = float(comp_meta.get("drainage_area_proxy", np.nan))
        sub["stream_order_proxy"] = float(comp_meta.get("stream_order_proxy", np.nan))
        sub["junction_hierarchy_weight"] = hierarchy_priority
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
    pts["longitudinal_profile_source"] = None
    if profile.empty:
        return pts
    for pid, sub in profile.groupby("profile_id"):
        mask = pts["profile_id"].astype(str) == str(pid)
        if not mask.any():
            continue
        stations = pd.to_numeric(pts.loc[mask, "station_m"], errors="coerce").to_numpy(dtype=float)
        ref_station = sub["station_m"].to_numpy(dtype=float)
        order = np.argsort(ref_station)
        ref_station = ref_station[order]
        elev = np.interp(stations, ref_station, sub["bed_elevation_m"].to_numpy(dtype=float)[order], left=np.nan, right=np.nan)
        unc = np.interp(stations, ref_station, sub["uncertainty_m"].to_numpy(dtype=float)[order], left=np.nan, right=np.nan)
        net_elev = np.interp(stations, ref_station, sub["network_backbone_elevation_m"].to_numpy(dtype=float)[order], left=np.nan, right=np.nan)
        net_unc = np.interp(stations, ref_station, sub["network_backbone_uncertainty_m"].to_numpy(dtype=float)[order], left=np.nan, right=np.nan)
        pts.loc[mask, "longitudinal_profile_elevation_m"] = elev
        pts.loc[mask, "longitudinal_profile_uncertainty_m"] = unc
        pts.loc[mask, "network_backbone_elevation_m"] = net_elev
        pts.loc[mask, "network_backbone_uncertainty_m"] = net_unc
        dom = sub.set_index("station_m")["dominant_source"].to_dict()
        net_src = sub.set_index("station_m")["network_backbone_source"].to_dict()
        pts.loc[mask, "longitudinal_profile_source"] = [dom.get(float(s), None) for s in stations]
        pts.loc[mask, "network_backbone_source"] = [net_src.get(float(s), None) for s in stations]
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
    authoritative_support_depth_path: Optional[str | Path] = None,
    authoritative_bed_elevation_path: Optional[str | Path] = None,
    authoritative_support_mask_path: Optional[str | Path] = None,
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
    sampled["wse_elevation_m"] = _sample_raster_at_points(centerline_points, wse_elevation_path, "wse_elevation_m")
    sampled["wse_influence"] = _sample_raster_at_points(centerline_points, wse_influence_path, "wse_influence")
    sampled["wse_uncertainty_m"] = _sample_raster_at_points(centerline_points, wse_uncertainty_path, "wse_uncertainty_m")

    profile = _build_profile_dataframe(centerline_points, sampled)
    if profile.empty:
        raise ValueError("empty_longitudinal_profile_after_sampling")

    edges = _read_network_edges(network_edges_path)
    endpoint_meta = _component_endpoint_metadata(edges)
    hierarchy_meta = _component_hierarchy_metadata(edges)
    profile = _apply_network_continuity(profile, endpoint_meta, hierarchy_meta=hierarchy_meta)

    out_csv = river_dir / "river_longitudinal_profile.csv"
    profile.to_csv(out_csv, index=False)

    profile_points = _assign_profile_values_to_points(centerline_points, profile)
    out_gpkg = river_dir / "river_longitudinal_profile_points.gpkg"
    if out_gpkg.exists():
        out_gpkg.unlink()
    profile_points.to_file(out_gpkg, driver="GPKG")

    out_backbone_csv = river_dir / "river_hydraulic_backbone.csv"
    profile[[
        "profile_id", "station_m", "station_step_m", "bed_elevation_m", "uncertainty_m",
        "wse_elevation_m", "wse_influence", "wse_uncertainty_m", "wse_local_slope_mpm",
        "network_backbone_elevation_m", "network_backbone_uncertainty_m", "network_junction_adjustment_m",
        "network_backbone_source", "junction_hierarchy_weight", "junction_wse_weight", "wse_endpoint_consensus_residual_m", "drainage_area_proxy", "stream_order_proxy",
    ]].to_csv(out_backbone_csv, index=False)

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
            if corridor_mask_path is not None and Path(corridor_mask_path).exists():
                with rasterio.open(corridor_mask_path) as cm:
                    domain_mask = cm.read(1, out_shape=(ds.height, ds.width), resampling=rasterio.enums.Resampling.nearest) > 0
            else:
                domain_mask = np.isfinite(ds.read(1))
            step_candidates = pd.to_numeric(profile["station_step_m"], errors="coerce").to_numpy(dtype=float)
            valid_steps = step_candidates[np.isfinite(step_candidates) & (step_candidates > 0.0)]
            step_m = float(np.median(valid_steps)) if valid_steps.size else 20.0
            max_distance_m = max(step_m * 6.0, 60.0)
            elev_raster, elev_influence = _nearest_surface_from_points(
                shape=(ds.height, ds.width),
                transform=ds.transform,
                domain_mask=domain_mask,
                points_gdf=profile_points,
                value_field="network_backbone_elevation_m",
                max_distance_m=max_distance_m,
            )
            unc_raster, _ = _nearest_surface_from_points(
                shape=(ds.height, ds.width),
                transform=ds.transform,
                domain_mask=domain_mask,
                points_gdf=profile_points,
                value_field="network_backbone_uncertainty_m",
                max_distance_m=max_distance_m,
            )
        out_elev = river_dir / "river_longitudinal_profile_elevation.tif"
        out_unc = river_dir / "river_longitudinal_profile_uncertainty.tif"
        out_inf = river_dir / "river_longitudinal_profile_influence.tif"
        with rasterio.open(out_elev, "w", **profile_meta) as dst:
            dst.write(elev_raster.astype(np.float32), 1)
            dst.update_tags(VALUE_TYPE="elevation", UNITS="meters", VERTICAL_SEMANTICS="absolute_elevation", ROLE="river_network_hydraulic_backbone")
        with rasterio.open(out_unc, "w", **profile_meta) as dst:
            dst.write(unc_raster.astype(np.float32), 1)
            dst.update_tags(VALUE_TYPE="uncertainty", UNITS="meters", ROLE="river_network_hydraulic_backbone_uncertainty")
        with rasterio.open(out_inf, "w", **profile_meta) as dst:
            dst.write(elev_influence.astype(np.float32), 1)
            dst.update_tags(VALUE_TYPE="fraction", ROLE="river_network_hydraulic_backbone_influence")
    else:
        out_elev = river_dir / "river_longitudinal_profile_elevation.tif"
        out_unc = river_dir / "river_longitudinal_profile_uncertainty.tif"
        out_inf = river_dir / "river_longitudinal_profile_influence.tif"

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
        "notes": [
            "This is the workflow-integrated network-aware 1D river backbone object.",
            "bed_elevation_m remains the local station-indexed tendency built from centerline, XS-support, bank-elevation evidence, and direct authoritative DEM anchors where authoritative support exists.",
            "network_backbone_elevation_m applies component smoothing plus cross-component junction solving using scaffold graph topology; when drainage-area or stream-order metadata are present, junction targets are flow-aware so dominant branches exert more downstream control while preserving authoritative anchor stations as hard vertical control.",
            "When a reliable WSE surface is available, endpoint junction arbitration is additionally WSE-aware: branches whose observed WSE is closer to junction consensus get more influence, so hydraulic stage agreement can reinforce hierarchy and continuity without overriding authoritative anchors.",
            "authoritative_support_depth_m is still auxiliary evidence and is not merged directly into absolute bed elevation without an explicit water-surface reference.",
        ],
    }
    out_summary = river_dir / "river_longitudinal_profile_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "longitudinal_profile": str(out_csv),
        "longitudinal_profile_points": str(out_gpkg),
        "longitudinal_profile_summary": str(out_summary),
        "longitudinal_profile_elevation": str(out_elev) if out_elev.exists() else None,
        "longitudinal_profile_uncertainty": str(out_unc) if out_unc.exists() else None,
        "longitudinal_profile_influence": str(out_inf) if out_inf.exists() else None,
        "hydraulic_backbone": str(out_backbone_csv),
        "hydraulic_backbone_nodes": str(out_nodes) if out_nodes.exists() else None,
        "hydraulic_backbone_edges": str(out_edges) if out_edges.exists() else None,
    }


__all__ = ["build_and_write_longitudinal_profile"]
