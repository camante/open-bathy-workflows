from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def compute_bank_edge_mask(corridor_mask: np.ndarray) -> np.ndarray:
    """Return the interior corridor edge mask.

    The edge is defined as corridor cells that touch at least one non-corridor
    neighbor in the 8-neighborhood. This keeps the product tied to the WAFFLES /
    NHD-derived corridor boundary rather than to any inferred depth surface.
    """
    corridor = np.asarray(corridor_mask, dtype=bool)
    if corridor.size == 0 or not np.any(corridor):
        return np.zeros(corridor.shape, dtype=bool)

    padded = np.pad(corridor, 1, mode="constant", constant_values=False)
    edge = np.zeros(corridor.shape, dtype=bool)
    for r_off in range(3):
        for c_off in range(3):
            if r_off == 1 and c_off == 1:
                continue
            nbr = padded[r_off:r_off + corridor.shape[0], c_off:c_off + corridor.shape[1]]
            edge |= corridor & (~nbr)
    return edge



def compute_bank_distance_influence(
    corridor_mask: np.ndarray,
    *,
    pixel_size_m: float,
    full_influence_m: float = 0.0,
    zero_influence_m: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance and influence fields tied to corridor-bank proximity.

    Influence is 1 at the interior bank edge and decays to 0 farther into the
    channel interior. Outside the corridor the influence is 0.
    """
    from scipy.ndimage import distance_transform_edt

    corridor = np.asarray(corridor_mask, dtype=bool)
    edge = compute_bank_edge_mask(corridor)
    bank_distance_m = np.full(corridor.shape, np.inf, dtype=np.float32)
    bank_influence = np.zeros(corridor.shape, dtype=np.float32)
    if not np.any(corridor):
        return edge.astype(np.uint8), bank_distance_m, bank_influence

    safe_px = max(float(pixel_size_m or 0.0), 1.0)
    if np.any(edge):
        dist_px = distance_transform_edt(~edge)
        bank_distance_m = (dist_px.astype(np.float32) * safe_px).astype(np.float32)
    else:
        bank_distance_m[corridor] = 0.0

    inner = np.nan_to_num(bank_distance_m, nan=np.inf, posinf=np.inf).astype(np.float32)
    bank_influence[corridor] = 1.0
    safe_full = max(float(full_influence_m or 0.0), 0.0)
    safe_zero = max(float(zero_influence_m or 0.0), safe_full + safe_px)
    span = max(safe_zero - safe_full, safe_px)
    taper = 1.0 - np.clip((inner - safe_full) / span, 0.0, 1.0)
    bank_influence[corridor] = taper[corridor].astype(np.float32)
    bank_influence[~corridor] = 0.0
    return edge.astype(np.uint8), bank_distance_m, bank_influence



def compute_bank_elevation_surface_from_authoritative(
    auth: np.ndarray,
    corridor_mask: np.ndarray,
    *,
    max_bank_distance_m: Optional[float] = None,
    bank_distance_m: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Nearest authoritative bank-elevation surface for river corridor cells.

    Uses authoritative cells *outside* the corridor as soft bank controls for
    channel-margin conditioning. This intentionally uses the corridor boundary as
    the locator, while leaving in-channel authoritative bathy as the hard
    control for the river interior.
    """
    from scipy.ndimage import distance_transform_edt

    auth = np.asarray(auth, dtype=np.float32)
    corridor = np.asarray(corridor_mask, dtype=bool)
    out = np.full(auth.shape, np.nan, dtype=np.float32)
    if not np.any(corridor):
        return out

    bank_auth = np.isfinite(auth) & (~corridor)
    if not np.any(bank_auth):
        return out

    _, nearest = distance_transform_edt(~bank_auth, return_indices=True)
    out[corridor] = auth[nearest[0][corridor], nearest[1][corridor]].astype(np.float32)
    if max_bank_distance_m is not None and bank_distance_m is not None:
        too_far = corridor & (np.asarray(bank_distance_m, dtype=np.float32) > float(max_bank_distance_m))
        out[too_far] = np.nan
    return out



def _clamp_distance(line, dist_m: float) -> float:
    return float(min(max(float(dist_m), 0.0), max(float(line.length), 0.0)))







def compute_lower_bank_wse_proxy_from_edge_guidance(
    bank_elevation: np.ndarray,
    bank_influence: np.ndarray,
    *,
    centerline_points_path: str | Path,
    transform,
    source_crs=None,
    proxy_radius_m: float,
    quantile: float = 0.25,
    adaptive_radius_steps: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0),
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a simple continuous lower-bank WSE proxy from edge-band bank samples.

    One path:
    1. sample lower-bank values near centerline stations from the edge band,
    2. interpolate missing stations along each component,
    3. smooth and enforce nonincreasing downstream,
    4. project the final station profile back onto the edge band only.
    """
    import geopandas as gpd
    import rasterio.transform
    from scipy.spatial import cKDTree

    bank = np.asarray(bank_elevation, dtype=np.float32)
    infl = np.asarray(bank_influence, dtype=np.float32)
    edge_band = np.isfinite(bank) & np.isfinite(infl) & (infl > 0.0)
    proxy = np.full(bank.shape, np.nan, dtype=np.float32)
    empty_cols = [
        "component_id", "station_m", "bank_wse_proxy_raw_m",
        "bank_wse_proxy_interp_m", "bank_wse_proxy_monotone_m",
        "bank_wse_proxy_adjustment_m",
        "bank_wse_proxy_search_radius_m",
        "bank_wse_proxy_search_attempt_count",
    ]
    if not np.any(edge_band):
        return proxy, pd.DataFrame(columns=empty_cols)

    cl_path = Path(centerline_points_path)
    if not cl_path.exists():
        raise FileNotFoundError(cl_path)
    cl = gpd.read_file(cl_path)
    if cl.empty:
        raise RuntimeError("river_bank_wse_proxy_missing_centerline_points")
    if source_crs is not None and cl.crs is not None and str(cl.crs) != str(source_crs):
        try:
            cl = cl.to_crs(source_crs)
        except Exception as exc:
            raise RuntimeError("river_bank_wse_proxy_centerline_reprojection_failed") from exc
    if "station_m" not in cl.columns:
        raise RuntimeError("river_bank_wse_proxy_missing_station_m")
    if "component_id" not in cl.columns:
        cl["component_id"] = "main"
    cl = cl[cl.geometry.notnull() & ~cl.geometry.is_empty].copy()
    if cl.empty:
        raise RuntimeError("river_bank_wse_proxy_empty_centerline_points")

    station_m = pd.to_numeric(cl["station_m"], errors="coerce").to_numpy(dtype=float)
    component_id = cl["component_id"].astype(str).to_numpy()
    clx = cl.geometry.x.to_numpy(dtype=np.float64)
    cly = cl.geometry.y.to_numpy(dtype=np.float64)
    clxy = np.column_stack([clx, cly])

    rows, cols = np.where(edge_band)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    edge_xy = np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])
    edge_z = bank[rows, cols].astype(np.float32)

    safe_radius = max(float(proxy_radius_m or 0.0), 1.0)
    metric_clxy = clxy
    metric_edge_xy = edge_xy
    if source_crs is not None:
        try:
            from pyproj import CRS, Transformer
            src_crs = CRS.from_user_input(source_crs)
            if src_crs.is_geographic:
                metric_crs = CRS.from_epsg(3857)
                tx = Transformer.from_crs(src_crs, metric_crs, always_xy=True)
                clmx, clmy = tx.transform(clxy[:, 0], clxy[:, 1])
                edmx, edmy = tx.transform(edge_xy[:, 0], edge_xy[:, 1])
                metric_clxy = np.column_stack([np.asarray(clmx, dtype=np.float64), np.asarray(clmy, dtype=np.float64)])
                metric_edge_xy = np.column_stack([np.asarray(edmx, dtype=np.float64), np.asarray(edmy, dtype=np.float64)])
        except Exception:
            pass
    edge_tree = cKDTree(metric_edge_xy)

    raw_proxy = np.full((len(cl),), np.nan, dtype=np.float32)
    interp_proxy = np.full((len(cl),), np.nan, dtype=np.float32)
    mono_proxy = np.full((len(cl),), np.nan, dtype=np.float32)

    search_radii_used = np.full((len(cl),), np.nan, dtype=np.float32)
    search_attempt_count = np.zeros((len(cl),), dtype=np.int32)
    for i, q in enumerate(metric_clxy):
        chosen = None
        chosen_radius = np.nan
        attempts = 0
        for step in tuple(float(s) for s in (adaptive_radius_steps or (1.0,))):
            attempts += 1
            trial_radius = max(safe_radius * max(step, 1.0), safe_radius)
            idx = edge_tree.query_ball_point(q, r=trial_radius)
            if not idx:
                continue
            vals = edge_z[np.asarray(idx, dtype=int)]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                chosen = vals
                chosen_radius = float(trial_radius)
                break
        search_attempt_count[i] = attempts
        if chosen is None:
            continue
        search_radii_used[i] = np.float32(chosen_radius)
        raw_proxy[i] = np.float32(np.nanquantile(chosen, quantile))

    def _interp_component(sta: np.ndarray, vals: np.ndarray) -> np.ndarray:
        out = vals.astype(np.float32).copy()
        finite = np.isfinite(out) & np.isfinite(sta)
        if np.count_nonzero(finite) == 0:
            return out
        if np.count_nonzero(finite) == 1:
            out[:] = out[finite][0]
            return out
        out[:] = np.interp(sta, sta[finite], out[finite]).astype(np.float32)
        return out

    for comp in pd.unique(component_id):
        comp_idx = np.flatnonzero(component_id == comp)
        if comp_idx.size == 0:
            continue
        order = np.argsort(station_m[comp_idx])
        ordered_idx = comp_idx[order]
        ordered_sta = station_m[ordered_idx].astype(np.float32)
        ordered_raw = raw_proxy[ordered_idx].astype(np.float32)
        ordered_interp = _interp_component(ordered_sta, ordered_raw)
        ordered_smooth = _nan_rolling_quantile(ordered_interp, half_window=2, quantile=quantile)
        ordered_mono = _weighted_pava_nondecreasing(
            ordered_smooth.astype(np.float32),
            np.ones(ordered_smooth.size, dtype=np.float32),
        )
        interp_proxy[ordered_idx] = ordered_interp
        mono_proxy[ordered_idx] = ordered_mono

    valid_cl = np.isfinite(mono_proxy)
    if np.any(valid_cl):
        cl_tree = cKDTree(metric_clxy[valid_cl])
        _, nn = cl_tree.query(metric_edge_xy, k=1)
        nn = np.asarray(nn, dtype=int)
        proxy_vals = mono_proxy[valid_cl][nn].astype(np.float32)
        proxy[rows, cols] = proxy_vals

    cl["bank_wse_proxy_raw_m"] = raw_proxy.astype(np.float32)
    cl["bank_wse_proxy_interp_m"] = interp_proxy.astype(np.float32)
    cl["bank_wse_proxy_monotone_m"] = mono_proxy.astype(np.float32)
    cl["bank_wse_proxy_adjustment_m"] = (mono_proxy - interp_proxy).astype(np.float32)
    cl["bank_wse_proxy_search_radius_m"] = search_radii_used.astype(np.float32)
    cl["bank_wse_proxy_search_attempt_count"] = search_attempt_count.astype(np.int32)
    return proxy.astype(np.float32), pd.DataFrame(cl[empty_cols])
def compute_bank_edge_guidance_from_authoritative(
    auth: np.ndarray,
    corridor_mask: np.ndarray,
    *,
    pixel_size_m: float,
    edge_guidance_distance_m: float,
    max_bank_distance_m: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a narrow edge-only bank guidance product.

    This is intentionally *not* a wall-to-wall cross-channel bank surface. It
    samples bank elevations from authoritative cells outside the corridor, then
    keeps them only within a narrow interior edge band so bank guidance acts as
    a boundary tendency rather than as a surrogate channel bed.
    """
    edge, bank_distance_m, bank_influence = compute_bank_distance_influence(
        corridor_mask,
        pixel_size_m=pixel_size_m,
        full_influence_m=0.0,
        zero_influence_m=max(float(max_bank_distance_m or 0.0), float(edge_guidance_distance_m or 0.0), max(float(pixel_size_m or 0.0), 1.0)),
    )
    bank_surface = compute_bank_elevation_surface_from_authoritative(
        auth,
        corridor_mask,
        max_bank_distance_m=max_bank_distance_m,
        bank_distance_m=bank_distance_m,
    ).astype(np.float32)
    safe_edge_distance = max(float(edge_guidance_distance_m or 0.0), max(float(pixel_size_m or 0.0), 1.0))
    edge_band = np.asarray(corridor_mask, dtype=bool) & np.isfinite(bank_surface) & (bank_distance_m <= safe_edge_distance)
    bank_surface = np.where(edge_band, bank_surface, np.nan).astype(np.float32)
    bank_influence = np.where(edge_band, bank_influence, np.float32(0.0)).astype(np.float32)
    return edge.astype(np.uint8), bank_distance_m.astype(np.float32), bank_influence.astype(np.float32), bank_surface.astype(np.float32)

def load_xs_bank_points(xs_gpkg: str | Path, *, target_crs: object | None = None):
    """Load left/right bank points from xs_builder output.

    The returned GeoDataFrame includes side-tagged bank points derived from the
    explicit bank distances recorded on each cross-section. This is a higher-
    fidelity bank signal than the corridor edge alone because it carries side-
    specific bank elevations sampled from topo/DEM during XS construction.
    """
    import geopandas as gpd
    import pandas as pd

    xs_gpkg = Path(xs_gpkg)
    if not xs_gpkg.exists():
        raise FileNotFoundError(xs_gpkg)
    xs = gpd.read_file(xs_gpkg, layer="xs_lines")
    if xs.empty:
        return gpd.GeoDataFrame(
            columns=[
                "xs_id", "river_id", "component_id", "side", "side_sign", "bank_z_m", "bank_z_raw_m",
                "bank_z_inner_min_m", "bank_z_local_q25_m", "bank_z_local_median_m", "bank_selected_source",
                "s_center_m", "geometry",
            ],
            geometry="geometry",
            crs=target_crs,
        )
    if target_crs is not None and xs.crs is not None and str(xs.crs) != str(target_crs):
        xs = xs.to_crs(target_crs)

    rows = []
    for rec in xs.itertuples():
        geom = getattr(rec, "geometry", None)
        if geom is None or geom.is_empty:
            continue
        xs_id = getattr(rec, "xs_id", None)
        river_id = getattr(rec, "river_id", None)
        component_id = getattr(rec, "component_id", None)
        s_center_m = float(getattr(rec, "s_center_m", np.nan)) if hasattr(rec, "s_center_m") else np.nan
        for side, dist_field, z_field, side_sign in (
            ("left", "bank_left_dist_m", "bank_left_z_m", -1.0),
            ("right", "bank_right_dist_m", "bank_right_z_m", 1.0),
        ):
            dist_val = getattr(rec, dist_field, np.nan)
            z_val = getattr(rec, z_field, np.nan)
            if not np.isfinite(dist_val) or not np.isfinite(z_val):
                continue
            pt = geom.interpolate(_clamp_distance(geom, float(dist_val)))
            rows.append({
                "xs_id": xs_id,
                "river_id": river_id,
                "component_id": int(component_id) if component_id is not None and str(component_id) != "" else -1,
                "side": side,
                "side_sign": float(side_sign),
                "bank_z_m": float(z_val),
                "bank_z_raw_m": float(getattr(rec, f"bank_{side}_raw_z_m", z_val)),
                "bank_z_inner_min_m": float(getattr(rec, f"bank_{side}_inner_min_z_m", np.nan)),
                "bank_z_local_q25_m": float(getattr(rec, f"bank_{side}_local_q25_z_m", np.nan)),
                "bank_z_local_median_m": float(getattr(rec, f"bank_{side}_local_median_z_m", np.nan)),
                "bank_selected_source": str(getattr(rec, f"bank_{side}_selected_source", "missing") or "missing"),
                "s_center_m": s_center_m,
                "geometry": pt,
            })
    if not rows:
        return gpd.GeoDataFrame(
            columns=[
                "xs_id", "river_id", "component_id", "side", "side_sign", "bank_z_m", "bank_z_raw_m",
                "bank_z_inner_min_m", "bank_z_local_q25_m", "bank_z_local_median_m", "bank_selected_source",
                "s_center_m", "geometry",
            ],
            geometry="geometry",
            crs=xs.crs,
        )
    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs=xs.crs)
    return out



def _nan_rolling_median(values: np.ndarray, half_window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.copy()
    hw = max(int(half_window), 0)
    if hw == 0:
        return values.copy()
    out = values.copy()
    for i in range(values.size):
        lo = max(0, i - hw)
        hi = min(values.size, i + hw + 1)
        win = values[lo:hi]
        finite = np.isfinite(win)
        out[i] = np.nanmedian(win[finite]).astype(np.float32) if np.any(finite) else values[i]
    return out.astype(np.float32)


def _nan_rolling_quantile(values: np.ndarray, half_window: int, quantile: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.copy()
    hw = max(int(half_window), 0)
    if hw == 0:
        return values.copy()
    out = values.copy()
    for i in range(values.size):
        lo = max(0, i - hw)
        hi = min(values.size, i + hw + 1)
        win = values[lo:hi]
        finite = np.isfinite(win)
        out[i] = np.nanquantile(win[finite], quantile).astype(np.float32) if np.any(finite) else values[i]
    return out.astype(np.float32)


def _other_side(side: str) -> str:
    return "right" if str(side).lower().startswith("l") else "left"


def _pair_lookup(bank_points: pd.DataFrame) -> dict[tuple[object, str], float]:
    lookup: dict[tuple[object, str], float] = {}
    if bank_points.empty or "xs_id" not in bank_points.columns:
        return lookup
    for rec in bank_points.itertuples():
        lookup[(getattr(rec, "xs_id", None), getattr(rec, "side", None))] = float(getattr(rec, "bank_z_raw_m", np.nan))
    return lookup


def _nanmin_finite(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmin(finite))


def _nanmedian_finite(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmedian(finite))


def _weighted_pava_nonincreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32).copy()
    wts = np.asarray(weights, dtype=np.float32).copy()
    if vals.size == 0:
        return vals
    wts = np.where(np.isfinite(wts) & (wts > 0.0), wts, np.float32(1.0)).astype(np.float32)
    blocks = []
    for i, (v, w) in enumerate(zip(vals, wts)):
        blocks.append([i, i, float(v), float(w)])
        while len(blocks) >= 2 and blocks[-2][2] < blocks[-1][2]:
            s0, e0, m0, w0 = blocks[-2]
            s1, e1, m1, w1 = blocks[-1]
            w = w0 + w1
            m = ((m0 * w0) + (m1 * w1)) / max(w, 1.0e-6)
            blocks[-2:] = [[s0, e1, m, w]]
    out = np.empty_like(vals, dtype=np.float32)
    for s, e, m, _ in blocks:
        out[s:e+1] = np.float32(m)
    return out


def _weighted_pava_nondecreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        return vals.astype(np.float32)
    return (-_weighted_pava_nonincreasing(-vals, weights)).astype(np.float32)


def summarize_bank_qc(bank_points: pd.DataFrame) -> pd.DataFrame:
    if bank_points is None or bank_points.empty:
        return pd.DataFrame([{
            "bank_point_count": 0,
            "bank_replaced_count": 0,
            "bank_replaced_fraction": 0.0,
            "high_bank_suspect_count": 0,
            "strong_contamination_count": 0,
            "component_floor_suspect_count": 0,
            "neighbor_spike_suspect_count": 0,
            "local_relief_suspect_count": 0,
            "cross_bank_asymmetry_suspect_count": 0,
            "percentile_suspect_count": 0,
            "bank_keep_count": 0,
            "bank_downgrade_count": 0,
            "bank_clamp_count": 0,
            "bank_replace_with_local_envelope_count": 0,
            "bank_reject_count": 0,
            "selected_from_inner_min_count": 0,
            "selected_from_local_q25_count": 0,
        }])
    df = bank_points.copy()
    total = int(len(df))
    adjustments = pd.to_numeric(df.get("bank_adjustment_m"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    selected_source = df.get("bank_selected_source", pd.Series("missing", index=df.index)).fillna("missing").astype(str)
    qc_action = df.get("bank_qc_action", pd.Series("keep", index=df.index)).fillna("keep").astype(str)
    return pd.DataFrame([{
        "bank_point_count": total,
        "bank_replaced_count": int(np.count_nonzero(adjustments < -0.01)),
        "bank_replaced_fraction": float(np.count_nonzero(adjustments < -0.01) / total) if total else 0.0,
        "high_bank_suspect_count": int(np.count_nonzero(df.get("bank_high_contamination_suspect", pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=bool))),
        "strong_contamination_count": int(np.count_nonzero(df.get("bank_strong_contamination", pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=bool))),
        "component_floor_suspect_count": int(np.count_nonzero(df.get("bank_component_floor_suspect", pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=bool))),
        "neighbor_spike_suspect_count": int(np.count_nonzero(df.get("bank_neighbor_spike_suspect", pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=bool))),
        "local_relief_suspect_count": int(np.count_nonzero(df.get("bank_local_relief_suspect", pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=bool))),
        "cross_bank_asymmetry_suspect_count": int(np.count_nonzero(df.get("bank_cross_bank_asymmetry_suspect", pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=bool))),
        "percentile_suspect_count": int(np.count_nonzero(df.get("bank_percentile_suspect", pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=bool))),
        "bank_keep_count": int(np.count_nonzero(qc_action.eq("keep"))),
        "bank_downgrade_count": int(np.count_nonzero(qc_action.eq("downgrade"))),
        "bank_clamp_count": int(np.count_nonzero(qc_action.eq("clamp"))),
        "bank_replace_with_local_envelope_count": int(np.count_nonzero(qc_action.eq("replace_with_local_envelope"))),
        "bank_reject_count": int(np.count_nonzero(qc_action.eq("reject"))),
        "bank_monotone_applied_count": int(np.count_nonzero(pd.to_numeric(df.get("bank_monotone_adjustment_m", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float) != 0.0)),
        "bank_monotone_adjustment_max_m": float(np.nanmax(np.abs(pd.to_numeric(df.get("bank_monotone_adjustment_m", pd.Series(0.0, index=df.index)), errors="coerce").to_numpy(dtype=float)))) if total else 0.0,
        "selected_from_inner_min_count": int(np.count_nonzero(selected_source.eq("inner_min"))),
        "selected_from_local_q25_count": int(np.count_nonzero(selected_source.eq("local_q25"))),
    }])



def _neighbor_peak_limit(values: np.ndarray, index: int) -> float:
    prev_val = float(values[index - 1]) if index > 0 and np.isfinite(values[index - 1]) else float("nan")
    next_val = float(values[index + 1]) if index + 1 < len(values) and np.isfinite(values[index + 1]) else float("nan")
    finite = [v for v in (prev_val, next_val) if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(max(finite))



def build_persistent_bank_network_points(
    xs_gpkg: str | Path,
    *,
    target_crs: object | None = None,
    smoothing_half_window: int = 2,
    max_gap_factor: float = 3.0,
    local_spike_threshold_m: float = 1.25,
    opposite_bank_threshold_m: float = 2.0,
    candidate_relaxation_m: float = 0.5,
    component_floor_relaxation_m: float = 0.75,
    neighbor_spike_threshold_m: float = 0.9,
    local_relief_threshold_m: float = 1.1,
    cross_bank_asymmetry_threshold_m: float = 1.4,
    percentile_relaxation_m: float = 0.7,
):
    """Build side-specific persistent bank points with longitudinal continuity metadata.

    This upgrades raw XS bank picks into a simple bank network model by:
    - grouping by component_id/river_id and side
    - sorting by along-river ``s_center_m``
    - smoothing bank elevations longitudinally
    - computing a continuity score that drops when spacing gaps are large

    The intent is not to claim perfect hydrologic topology; it is to create a
    deterministic, side-consistent bank guidance product that survives local AOI
    changes better than isolated nearest-bank picks.
    """
    bank_points = load_xs_bank_points(xs_gpkg, target_crs=target_crs)
    if bank_points.empty:
        return bank_points

    bank_points = bank_points.copy()
    bank_points["bank_z_raw_m"] = bank_points["bank_z_raw_m"].astype(np.float32)
    bank_points["bank_z_m"] = bank_points["bank_z_m"].astype(np.float32)
    bank_points["bank_z_final_m"] = bank_points["bank_z_m"].astype(np.float32)
    bank_points["continuity_weight"] = np.float32(0.0)
    bank_points["bank_longitudinal_ref_m"] = np.float32(np.nan)
    bank_points["bank_local_lower_ref_m"] = np.float32(np.nan)
    bank_points["bank_component_floor_ref_m"] = np.float32(np.nan)
    bank_points["bank_neighbor_upper_ref_m"] = np.float32(np.nan)
    bank_points["bank_percentile_ref_m"] = np.float32(np.nan)
    bank_points["bank_opposite_raw_z_m"] = np.float32(np.nan)
    bank_points["bank_high_contamination_suspect"] = False
    bank_points["bank_strong_contamination"] = False
    bank_points["bank_component_floor_suspect"] = False
    bank_points["bank_neighbor_spike_suspect"] = False
    bank_points["bank_local_relief_suspect"] = False
    bank_points["bank_cross_bank_asymmetry_suspect"] = False
    bank_points["bank_percentile_suspect"] = False
    bank_points["bank_contamination_score"] = np.float32(0.0)
    bank_points["bank_qc_action"] = "keep"
    bank_points["bank_qc_weight"] = np.float32(1.0)
    bank_points["bank_adjustment_m"] = np.float32(0.0)
    bank_points["bank_adjustment_reason"] = "none"
    bank_points["bank_monotone_downstream_m"] = np.float32(np.nan)
    bank_points["bank_monotone_adjustment_m"] = np.float32(0.0)

    pair_lookup = _pair_lookup(bank_points)

    group_cols = ["component_id", "river_id", "side"]
    for _, idx in bank_points.groupby(group_cols, dropna=False).groups.items():
        sub = bank_points.loc[idx].sort_values("s_center_m").copy()
        z = sub["bank_z_m"].to_numpy(dtype=np.float32)
        s = sub["s_center_m"].to_numpy(dtype=np.float32)
        if z.size == 0:
            continue
        z_sm = _nan_rolling_median(z, half_window=smoothing_half_window)
        z_low = _nan_rolling_quantile(z, half_window=smoothing_half_window, quantile=0.25)
        if z_sm.size >= 3:
            z_sm[1:-1] = (0.25 * z_sm[:-2] + 0.5 * z_sm[1:-1] + 0.25 * z_sm[2:]).astype(np.float32)
        candidate_floor = np.nanmin(
            np.vstack([
                sub.get("bank_z_inner_min_m", pd.Series(np.nan, index=sub.index)).to_numpy(dtype=np.float32),
                sub.get("bank_z_local_q25_m", pd.Series(np.nan, index=sub.index)).to_numpy(dtype=np.float32),
                sub.get("bank_z_local_median_m", pd.Series(np.nan, index=sub.index)).to_numpy(dtype=np.float32),
                z_low,
            ]),
            axis=0,
        ).astype(np.float32)
        component_half_window = max(int(smoothing_half_window) + 2, 4)
        component_floor = _nan_rolling_quantile(candidate_floor, half_window=component_half_window, quantile=0.20)
        component_floor = _nan_rolling_median(component_floor, half_window=max(1, smoothing_half_window))
        percentile_env = _nan_rolling_quantile(z, half_window=max(component_half_window + 2, 6), quantile=0.35)
        percentile_env = _nan_rolling_median(percentile_env, half_window=max(1, smoothing_half_window))
        neighbor_upper = np.full(len(sub), np.nan, dtype=np.float32)
        for j in range(len(sub)):
            neighbor_upper[j] = _neighbor_peak_limit(candidate_floor, j)
        z_final = z.copy()
        opposite = np.full(len(sub), np.nan, dtype=np.float32)
        contamination = np.zeros(len(sub), dtype=bool)
        strong = np.zeros(len(sub), dtype=bool)
        component_floor_suspect = np.zeros(len(sub), dtype=bool)
        neighbor_spike_suspect = np.zeros(len(sub), dtype=bool)
        local_relief_suspect = np.zeros(len(sub), dtype=bool)
        cross_bank_asymmetry_suspect = np.zeros(len(sub), dtype=bool)
        percentile_suspect = np.zeros(len(sub), dtype=bool)
        contamination_score = np.zeros(len(sub), dtype=np.float32)
        qc_weight = np.ones(len(sub), dtype=np.float32)
        qc_action = np.array(["keep"] * len(sub), dtype=object)
        reasons = np.array(["none"] * len(sub), dtype=object)
        selected_source = sub.get("bank_selected_source", pd.Series("missing", index=sub.index)).fillna("missing").astype(str).to_numpy(dtype=object)
        for j, rec in enumerate(sub.itertuples()):
            opp = pair_lookup.get((getattr(rec, "xs_id", None), _other_side(getattr(rec, "side", ""))), np.nan)
            opposite[j] = float(opp) if np.isfinite(opp) else np.nan
            raw = float(getattr(rec, "bank_z_raw_m", np.nan))
            local_ref = float(z_sm[j]) if np.isfinite(z_sm[j]) else np.nan
            lower_ref = float(candidate_floor[j]) if np.isfinite(candidate_floor[j]) else np.nan
            component_ref = float(component_floor[j]) if np.isfinite(component_floor[j]) else np.nan
            percentile_ref = float(percentile_env[j]) if np.isfinite(percentile_env[j]) else np.nan
            neighbor_ref = float(neighbor_upper[j]) if np.isfinite(neighbor_upper[j]) else np.nan
            source_penalty = 0.35 if str(selected_source[j]).lower() in {"picked", "local_median"} else 0.0
            component_limit = component_ref + max(float(component_floor_relaxation_m) - source_penalty, 0.2) if np.isfinite(component_ref) else np.nan
            neighbor_limit = neighbor_ref + max(float(neighbor_spike_threshold_m) - source_penalty, 0.2) if np.isfinite(neighbor_ref) else np.nan
            percentile_limit = percentile_ref + max(float(percentile_relaxation_m) - source_penalty, 0.2) if np.isfinite(percentile_ref) else np.nan
            local_relief_limit = lower_ref + max(float(local_relief_threshold_m) - source_penalty, 0.25) if np.isfinite(lower_ref) else np.nan
            cross_bank_limit = float(opposite[j]) + max(float(cross_bank_asymmetry_threshold_m) - source_penalty, 0.25) if np.isfinite(opposite[j]) else np.nan
            component_floor_suspect[j] = bool(np.isfinite(component_limit) and raw > component_limit + 1.0e-6)
            neighbor_spike_suspect[j] = bool(np.isfinite(neighbor_limit) and raw > neighbor_limit + 1.0e-6)
            local_relief_suspect[j] = bool(np.isfinite(local_relief_limit) and raw > local_relief_limit + 1.0e-6)
            cross_bank_asymmetry_suspect[j] = bool(np.isfinite(cross_bank_limit) and raw > cross_bank_limit + 1.0e-6)
            percentile_suspect[j] = bool(np.isfinite(percentile_limit) and raw > percentile_limit + 1.0e-6)
            contamination_score[j] = float(
                component_floor_suspect[j]
                + neighbor_spike_suspect[j]
                + local_relief_suspect[j]
                + cross_bank_asymmetry_suspect[j]
                + 0.75 * percentile_suspect[j]
            )
            limits = [raw]
            if np.isfinite(local_ref):
                limits.append(local_ref + float(local_spike_threshold_m))
            if np.isfinite(lower_ref):
                limits.append(lower_ref + float(candidate_relaxation_m))
            if np.isfinite(opposite[j]):
                limits.append(float(opposite[j]) + float(opposite_bank_threshold_m))
            if np.isfinite(component_limit):
                limits.append(component_limit)
            if np.isfinite(neighbor_limit):
                limits.append(neighbor_limit)
            if np.isfinite(local_relief_limit):
                limits.append(local_relief_limit)
            if np.isfinite(cross_bank_limit):
                limits.append(cross_bank_limit)
            if np.isfinite(percentile_limit):
                limits.append(percentile_limit)
            bank_limit = _nanmin_finite(limits)
            envelope_ref = _nanmedian_finite([lower_ref, component_ref, percentile_ref, local_ref])
            if np.isfinite(raw) and np.isfinite(bank_limit) and raw > bank_limit + 1.0e-6:
                contamination[j] = True
                strong[j] = bool(raw > (bank_limit + max(1.25, float(local_spike_threshold_m))) or contamination_score[j] >= 3.0)
                severe_low_quality = str(selected_source[j]).lower() in {"picked", "local_median"} and strong[j]
                if severe_low_quality and contamination_score[j] >= 2.75:
                    qc_action[j] = "reject"
                    qc_weight[j] = np.float32(0.0)
                    z_final[j] = np.float32(np.nan)
                    reasons[j] = "multi_signal_reject"
                elif contamination_score[j] >= 1.75 and np.isfinite(envelope_ref):
                    qc_action[j] = "replace_with_local_envelope"
                    qc_weight[j] = np.float32(0.3)
                    z_final[j] = np.float32(envelope_ref)
                    reasons[j] = "local_envelope_replace"
                else:
                    qc_action[j] = "clamp"
                    qc_weight[j] = np.float32(0.45)
                    z_final[j] = np.float32(bank_limit)
                    if component_floor_suspect[j] and raw > bank_limit + 1.0e-6:
                        reasons[j] = "component_floor_clamp"
                    elif neighbor_spike_suspect[j] and raw > bank_limit + 1.0e-6:
                        reasons[j] = "neighbor_spike_clamp"
                    elif cross_bank_asymmetry_suspect[j] and np.isfinite(opposite[j]):
                        reasons[j] = "cross_bank_asymmetry_clamp"
                    elif local_relief_suspect[j] and np.isfinite(lower_ref):
                        reasons[j] = "local_relief_clamp"
                    elif percentile_suspect[j] and np.isfinite(percentile_ref):
                        reasons[j] = "percentile_clamp"
                    elif np.isfinite(opposite[j]) and raw > opposite[j] + float(opposite_bank_threshold_m):
                        reasons[j] = "opposite_bank_clamp"
                    elif np.isfinite(lower_ref) and raw > lower_ref + float(candidate_relaxation_m):
                        reasons[j] = "local_candidate_clamp"
                    else:
                        reasons[j] = "longitudinal_spike_clamp"
            elif contamination_score[j] >= 1.0:
                qc_action[j] = "downgrade"
                qc_weight[j] = np.float32(0.7 if contamination_score[j] < 2.0 else 0.55)
                reasons[j] = "multi_signal_downgrade"
        z_final = _nan_rolling_median(z_final, half_window=max(1, smoothing_half_window))
        monotone = z_final.copy()
        finite_final = np.isfinite(monotone)
        if np.count_nonzero(finite_final) >= 2:
            mono_weights = np.clip(qc_weight[finite_final], 0.05, 1.0).astype(np.float32)
            monotone_vals = _weighted_pava_nonincreasing(monotone[finite_final].astype(np.float32), mono_weights)
            monotone[finite_final] = monotone_vals.astype(np.float32)
        z_final = monotone.astype(np.float32)
        spacing = np.diff(s) if s.size > 1 else np.array([], dtype=np.float32)
        typical_spacing = float(np.nanmedian(spacing[np.isfinite(spacing)])) if spacing.size and np.any(np.isfinite(spacing)) else np.nan
        if not np.isfinite(typical_spacing) or typical_spacing <= 0.0:
            continuity = np.ones_like(z_sm, dtype=np.float32)
        else:
            prev_gap = np.full_like(z_sm, typical_spacing, dtype=np.float32)
            next_gap = np.full_like(z_sm, typical_spacing, dtype=np.float32)
            if spacing.size:
                prev_gap[1:] = spacing.astype(np.float32)
                next_gap[:-1] = spacing.astype(np.float32)
            worst_gap = np.maximum(prev_gap, next_gap)
            continuity = 1.0 - np.clip((worst_gap - typical_spacing) / max(typical_spacing * max(float(max_gap_factor) - 1.0, 1.0), 1.0), 0.0, 1.0)
            continuity = np.clip(continuity, 0.0, 1.0).astype(np.float32)
            if z_sm.size == 1:
                continuity[:] = 0.5
        bank_points.loc[sub.index, "bank_z_m"] = z_final.astype(np.float32)
        bank_points.loc[sub.index, "bank_z_final_m"] = z_final.astype(np.float32)
        bank_points.loc[sub.index, "continuity_weight"] = continuity.astype(np.float32)
        bank_points.loc[sub.index, "bank_longitudinal_ref_m"] = z_sm.astype(np.float32)
        bank_points.loc[sub.index, "bank_local_lower_ref_m"] = candidate_floor.astype(np.float32)
        bank_points.loc[sub.index, "bank_component_floor_ref_m"] = component_floor.astype(np.float32)
        bank_points.loc[sub.index, "bank_neighbor_upper_ref_m"] = neighbor_upper.astype(np.float32)
        bank_points.loc[sub.index, "bank_percentile_ref_m"] = percentile_env.astype(np.float32)
        bank_points.loc[sub.index, "bank_opposite_raw_z_m"] = opposite.astype(np.float32)
        bank_points.loc[sub.index, "bank_high_contamination_suspect"] = contamination
        bank_points.loc[sub.index, "bank_strong_contamination"] = strong
        bank_points.loc[sub.index, "bank_component_floor_suspect"] = component_floor_suspect
        bank_points.loc[sub.index, "bank_neighbor_spike_suspect"] = neighbor_spike_suspect
        bank_points.loc[sub.index, "bank_local_relief_suspect"] = local_relief_suspect
        bank_points.loc[sub.index, "bank_cross_bank_asymmetry_suspect"] = cross_bank_asymmetry_suspect
        bank_points.loc[sub.index, "bank_percentile_suspect"] = percentile_suspect
        bank_points.loc[sub.index, "bank_contamination_score"] = contamination_score.astype(np.float32)
        bank_points.loc[sub.index, "bank_qc_action"] = qc_action
        bank_points.loc[sub.index, "bank_qc_weight"] = qc_weight.astype(np.float32)
        bank_points.loc[sub.index, "bank_adjustment_m"] = (z_final - z).astype(np.float32)
        bank_points.loc[sub.index, "bank_adjustment_reason"] = reasons
        bank_points.loc[sub.index, "bank_monotone_downstream_m"] = z_final.astype(np.float32)
        bank_points.loc[sub.index, "bank_monotone_adjustment_m"] = (z_final - z_sm).astype(np.float32)

    return bank_points



def compute_xs_bank_guidance_surfaces(
    *,
    corridor_mask: np.ndarray,
    transform,
    auth: np.ndarray,
    xs_gpkg: str | Path,
    raster_crs: object | None = None,
    max_bank_distance_m: Optional[float] = None,
    bank_points_gdf=None,
) -> dict[str, np.ndarray]:
    """Build side-specific and blended bank-elevation surfaces from XS bank picks.

    This uses explicit left/right bank elevations sampled by ``xs_builder.py``.
    For each corridor cell, it queries the nearest left-bank and right-bank XS
    bank points independently, then blends them by inverse distance. That gives
    a longitudinal bank-control surface that tracks the river corridor better
    than a simple nearest outside-corridor authoritative raster lookup.

    When provided with ``bank_points_gdf`` from
    ``build_persistent_bank_network_points()``, the surfaces are driven by a
    side-consistent, longitudinally smoothed bank network rather than isolated
    raw XS bank picks.
    """
    import rasterio.transform
    from scipy.spatial import cKDTree

    corridor = np.asarray(corridor_mask, dtype=bool)
    auth = np.asarray(auth, dtype=np.float32)
    shp = corridor.shape
    nan = np.full(shp, np.nan, dtype=np.float32)
    zero = np.zeros(shp, dtype=np.float32)
    inf = np.full(shp, np.inf, dtype=np.float32)
    if not np.any(corridor):
        return {
            "left_bank_elevation": nan.copy(),
            "right_bank_elevation": nan.copy(),
            "bank_elevation": nan.copy(),
            "bank_pair_weight": zero.copy(),
            "bank_continuity_weight": zero.copy(),
            "left_bank_distance_m": inf.copy(),
            "right_bank_distance_m": inf.copy(),
        }

    bank_points = bank_points_gdf if bank_points_gdf is not None else build_persistent_bank_network_points(xs_gpkg, target_crs=raster_crs)
    if bank_points is None or bank_points.empty:
        return {
            "left_bank_elevation": nan.copy(),
            "right_bank_elevation": nan.copy(),
            "bank_elevation": nan.copy(),
            "bank_pair_weight": zero.copy(),
            "bank_continuity_weight": zero.copy(),
            "left_bank_distance_m": inf.copy(),
            "right_bank_distance_m": inf.copy(),
        }

    rows, cols = np.where(corridor)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    qxy = np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])
    safe_px = max(abs(float(getattr(transform, "a", 1.0))), abs(float(getattr(transform, "e", 1.0))), 1.0)

    left_surface = nan.copy()
    right_surface = nan.copy()
    left_dist = inf.copy()
    right_dist = inf.copy()
    left_cont = zero.copy()
    right_cont = zero.copy()

    def _query(side_name: str):
        side = bank_points[bank_points["side"] == side_name]
        if side.empty:
            return None, None, None
        pts = np.column_stack([side.geometry.x.to_numpy(dtype=np.float64), side.geometry.y.to_numpy(dtype=np.float64)])
        vals = side["bank_z_m"].to_numpy(dtype=np.float32)
        cont = side.get("continuity_weight", np.ones(len(side), dtype=np.float32))
        cont = np.asarray(cont, dtype=np.float32)
        tree = cKDTree(pts)
        dist, idx = tree.query(qxy, k=1)
        idx = np.asarray(idx, dtype=int)
        return np.asarray(dist, dtype=np.float32), vals[idx], cont[idx]

    ldist, lvals, lcont = _query("left")
    rdist, rvals, rcont = _query("right")
    if ldist is not None:
        left_surface[rows, cols] = lvals.astype(np.float32)
        left_dist[rows, cols] = ldist.astype(np.float32)
        left_cont[rows, cols] = np.clip(lcont, 0.0, 1.0).astype(np.float32)
    if rdist is not None:
        right_surface[rows, cols] = rvals.astype(np.float32)
        right_dist[rows, cols] = rdist.astype(np.float32)
        right_cont[rows, cols] = np.clip(rcont, 0.0, 1.0).astype(np.float32)

    blend = nan.copy()
    pair_weight = zero.copy()
    continuity_weight = zero.copy()
    has_left = np.isfinite(left_surface) & corridor
    has_right = np.isfinite(right_surface) & corridor
    both = has_left & has_right
    if np.any(both):
        wl = 1.0 / np.maximum(left_dist[both], safe_px)
        wr = 1.0 / np.maximum(right_dist[both], safe_px)
        z_low = np.minimum(left_surface[both], right_surface[both])
        z_high = np.maximum(left_surface[both], right_surface[both])
        z_avg = ((wl * left_surface[both]) + (wr * right_surface[both])) / (wl + wr)
        asym = z_high - z_low
        # Strongly prefer the lower plausible bank when side asymmetry grows;
        # in unsupported reaches the high side is much more likely to be a
        # terrace/roadfill/high-edge artifact than true stage control.
        guard = np.float32(1.0)
        low_bias = np.clip((asym - 0.25) / max(float(guard - 0.25), 0.25), 0.0, 1.0).astype(np.float32)
        blend[both] = np.where(asym >= guard, z_low, (low_bias * z_low + (1.0 - low_bias) * z_avg)).astype(np.float32)
        ratio = np.minimum(wl, wr) / np.maximum(wl, wr)
        pair_weight[both] = np.clip((ratio * (1.0 - 0.65 * low_bias)).astype(np.float32), 0.0, 1.0)
        continuity_weight[both] = np.minimum(left_cont[both], right_cont[both]).astype(np.float32)
    left_only = has_left & ~has_right
    right_only = has_right & ~has_left
    if np.any(left_only):
        blend[left_only] = left_surface[left_only]
        continuity_weight[left_only] = left_cont[left_only].astype(np.float32)
    if np.any(right_only):
        blend[right_only] = right_surface[right_only]
        continuity_weight[right_only] = right_cont[right_only].astype(np.float32)

    if max_bank_distance_m is not None:
        cutoff = float(max_bank_distance_m)
        too_far = corridor & (np.minimum(left_dist, right_dist) > cutoff)
        blend[too_far] = np.nan
        left_surface[too_far] = np.nan
        right_surface[too_far] = np.nan
        pair_weight[too_far] = 0.0
        continuity_weight[too_far] = 0.0

    valid_auth = np.isfinite(auth)
    if np.any(valid_auth) and np.any(~np.isfinite(blend) & corridor):
        fallback = compute_bank_elevation_surface_from_authoritative(
            auth,
            corridor,
            max_bank_distance_m=max_bank_distance_m,
        )
        use_fallback = corridor & ~np.isfinite(blend) & np.isfinite(fallback)
        blend[use_fallback] = fallback[use_fallback]

    return {
        "left_bank_elevation": left_surface.astype(np.float32),
        "right_bank_elevation": right_surface.astype(np.float32),
        "bank_elevation": blend.astype(np.float32),
        "bank_pair_weight": pair_weight.astype(np.float32),
        "bank_continuity_weight": np.clip(continuity_weight, 0.0, 1.0).astype(np.float32),
        "left_bank_distance_m": left_dist.astype(np.float32),
        "right_bank_distance_m": right_dist.astype(np.float32),
    }


def compute_graph_informed_bank_context_surfaces(
    *,
    corridor_mask: np.ndarray,
    transform,
    bank_points_gdf,
    estuary_transition: Optional[np.ndarray] = None,
    k_neighbors: int = 6,
    estuary_decay_distance_m: float = 250.0,
) -> dict[str, np.ndarray]:
    """Return graph-informed bank context surfaces for intermediate bank hardening.

    This is intentionally lighter-weight than a full river-graph bank solver. It
    uses the persistent XS-derived bank network to estimate:
    - graph confidence: whether nearby bank picks belong to a coherent component
    - confluence damping: reduce confidence where multiple components compete
    - estuary side decay: reduce side-specific confidence near estuary transition

    The result is a deterministic, AOI-stable modifier layer that improves bank
    continuity without yet introducing full branch inheritance logic.
    """
    import rasterio.transform
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree

    corridor = np.asarray(corridor_mask, dtype=bool)
    shp = corridor.shape
    zero = np.zeros(shp, dtype=np.float32)
    if not np.any(corridor) or bank_points_gdf is None or getattr(bank_points_gdf, 'empty', True):
        return {
            'bank_graph_confidence': zero.copy(),
            'bank_confluence_damping': np.ones(shp, dtype=np.float32),
            'bank_estuary_side_decay': np.ones(shp, dtype=np.float32),
        }

    rows, cols = np.where(corridor)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    qxy = np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])

    pts = np.column_stack([bank_points_gdf.geometry.x.to_numpy(dtype=np.float64), bank_points_gdf.geometry.y.to_numpy(dtype=np.float64)])
    tree = cKDTree(pts)
    k = max(1, min(int(k_neighbors), len(bank_points_gdf)))
    dist, idx = tree.query(qxy, k=k)
    if k == 1:
        dist = np.asarray(dist, dtype=np.float32)[:, None]
        idx = np.asarray(idx, dtype=int)[:, None]
    else:
        dist = np.asarray(dist, dtype=np.float32)
        idx = np.asarray(idx, dtype=int)

    component = np.asarray(bank_points_gdf.get('component_id', np.full(len(bank_points_gdf), -1)), dtype=np.int64)
    continuity = np.asarray(bank_points_gdf.get('continuity_weight', np.ones(len(bank_points_gdf))), dtype=np.float32)
    side = np.asarray(bank_points_gdf.get('side_sign', np.zeros(len(bank_points_gdf))), dtype=np.float32)

    near_component = component[idx]
    near_cont = np.clip(continuity[idx], 0.0, 1.0).astype(np.float32)
    near_side = side[idx]

    graph_conf = np.zeros(rows.shape[0], dtype=np.float32)
    confluence_damping = np.ones(rows.shape[0], dtype=np.float32)
    for i in range(rows.shape[0]):
        comp_i = near_component[i]
        cont_i = near_cont[i]
        side_i = near_side[i]
        finite = np.isfinite(cont_i)
        if not np.any(finite):
            continue
        comp_vals = comp_i[finite]
        unique_comps = np.unique(comp_vals)
        dominant_share = float(np.max([(comp_vals == c).mean() for c in unique_comps])) if unique_comps.size else 0.0
        mean_cont = float(np.nanmean(cont_i[finite])) if np.any(np.isfinite(cont_i[finite])) else 0.0
        side_balance = float(min(np.sum(side_i[finite] < 0), np.sum(side_i[finite] > 0)) / max(np.sum(finite), 1))
        graph_conf[i] = np.clip(0.55 * mean_cont + 0.30 * dominant_share + 0.15 * min(side_balance * 2.0, 1.0), 0.0, 1.0)
        # More competing components nearby => stronger damping. Keep conservative floor.
        confluence_damping[i] = np.clip(1.0 - 0.22 * max(int(unique_comps.size) - 1, 0), 0.45, 1.0)

    graph_surface = zero.copy()
    confluence_surface = np.ones(shp, dtype=np.float32)
    graph_surface[rows, cols] = graph_conf.astype(np.float32)
    confluence_surface[rows, cols] = confluence_damping.astype(np.float32)

    estuary_decay = np.ones(shp, dtype=np.float32)
    if estuary_transition is not None and np.any(np.asarray(estuary_transition, dtype=bool) & corridor):
        est = np.asarray(estuary_transition, dtype=bool) & corridor
        safe_px = max(abs(float(getattr(transform, 'a', 1.0))), abs(float(getattr(transform, 'e', 1.0))), 1.0)
        dist_px = distance_transform_edt(~est)
        dist_m = dist_px.astype(np.float32) * safe_px
        decay_scale = max(float(estuary_decay_distance_m or 0.0), safe_px)
        estuary_decay[corridor] = np.clip(0.35 + 0.65 * (dist_m[corridor] / decay_scale), 0.35, 1.0).astype(np.float32)
        estuary_decay[~corridor] = 1.0

    return {
        'bank_graph_confidence': graph_surface.astype(np.float32),
        'bank_confluence_damping': confluence_surface.astype(np.float32),
        'bank_estuary_side_decay': estuary_decay.astype(np.float32),
    }


__all__ = [
    "compute_bank_edge_mask",
    "compute_bank_distance_influence",
    "compute_bank_elevation_surface_from_authoritative",
    "load_xs_bank_points",
    "build_persistent_bank_network_points",
    "summarize_bank_qc",
    "compute_xs_bank_guidance_surfaces",
    "compute_graph_informed_bank_context_surfaces",
]
