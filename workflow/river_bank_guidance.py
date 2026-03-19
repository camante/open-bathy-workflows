from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


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
            columns=["xs_id", "river_id", "component_id", "side", "side_sign", "bank_z_m", "s_center_m", "geometry"],
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
                "bank_z_raw_m": float(z_val),
                "s_center_m": s_center_m,
                "geometry": pt,
            })
    if not rows:
        return gpd.GeoDataFrame(
            columns=["xs_id", "river_id", "component_id", "side", "side_sign", "bank_z_m", "bank_z_raw_m", "s_center_m", "geometry"],
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



def build_persistent_bank_network_points(
    xs_gpkg: str | Path,
    *,
    target_crs: object | None = None,
    smoothing_half_window: int = 2,
    max_gap_factor: float = 3.0,
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
    bank_points["continuity_weight"] = np.float32(0.0)

    group_cols = ["component_id", "river_id", "side"]
    for _, idx in bank_points.groupby(group_cols, dropna=False).groups.items():
        sub = bank_points.loc[idx].sort_values("s_center_m").copy()
        z = sub["bank_z_m"].to_numpy(dtype=np.float32)
        s = sub["s_center_m"].to_numpy(dtype=np.float32)
        if z.size == 0:
            continue
        z_sm = _nan_rolling_median(z, half_window=smoothing_half_window)
        if z_sm.size >= 3:
            z_sm[1:-1] = (0.25 * z_sm[:-2] + 0.5 * z_sm[1:-1] + 0.25 * z_sm[2:]).astype(np.float32)
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
        bank_points.loc[sub.index, "bank_z_m"] = z_sm.astype(np.float32)
        bank_points.loc[sub.index, "continuity_weight"] = continuity.astype(np.float32)

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
        blend[both] = ((wl * left_surface[both]) + (wr * right_surface[both])) / (wl + wr)
        ratio = np.minimum(wl, wr) / np.maximum(wl, wr)
        pair_weight[both] = np.clip(ratio.astype(np.float32), 0.0, 1.0)
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
    "compute_xs_bank_guidance_surfaces",
    "compute_graph_informed_bank_context_surfaces",
]
