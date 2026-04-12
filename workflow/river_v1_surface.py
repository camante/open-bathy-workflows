from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy

from core.json_io import write_json
from core.paths import ensure_dir

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover - fallback handled below
    cKDTree = None


def _pixel_centers_for_mask(mask: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.where(mask > 0)
    if rows.size == 0:
        return rows.astype(np.int32), cols.astype(np.int32), np.zeros((0, 2), dtype=np.float64)
    xs, ys = xy(transform, rows, cols, offset="center")
    coords = np.column_stack([
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
    ])
    return rows.astype(np.int32), cols.astype(np.int32), coords


def _query_backbone_surface(coords: np.ndarray, center_coords: np.ndarray, center_z: np.ndarray, *, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if coords.size == 0 or center_coords.size == 0:
        return np.full((coords.shape[0],), np.nan, dtype=np.float32), np.full((coords.shape[0],), -1, dtype=np.int32)
    valid = np.isfinite(center_coords).all(axis=1) & np.isfinite(center_z)
    center_coords = center_coords[valid]
    center_z = center_z[valid]
    if center_coords.shape[0] == 0:
        return np.full((coords.shape[0],), np.nan, dtype=np.float32), np.full((coords.shape[0],), -1, dtype=np.int32)
    k = max(1, min(int(k), int(center_coords.shape[0])))
    if cKDTree is not None:
        tree = cKDTree(center_coords)
        dist, idx = tree.query(coords, k=k)
    else:  # pragma: no cover - scipy is expected, but keep a safe fallback
        diff = coords[:, None, :] - center_coords[None, :, :]
        dist_full = np.sqrt(np.sum(diff * diff, axis=2))
        idx = np.argsort(dist_full, axis=1)[:, :k]
        dist = np.take_along_axis(dist_full, idx, axis=1)
    if k == 1:
        idx = np.asarray(idx, dtype=np.int32).reshape((-1, 1))
        dist = np.asarray(dist, dtype=np.float64).reshape((-1, 1))
    else:
        idx = np.asarray(idx, dtype=np.int32)
        dist = np.asarray(dist, dtype=np.float64)
    neighbor_z = center_z[idx]
    direct = dist[:, 0] <= 1.0e-9
    out = np.full((coords.shape[0],), np.nan, dtype=np.float32)
    out[direct] = neighbor_z[direct, 0].astype(np.float32)
    if np.any(~direct):
        safe_dist = np.maximum(dist[~direct], 1.0e-6)
        weights = 1.0 / safe_dist
        z = neighbor_z[~direct]
        numer = np.sum(weights * z, axis=1)
        denom = np.sum(weights, axis=1)
        valid_denom = denom > 0
        if np.any(valid_denom):
            vals = np.full((weights.shape[0],), np.nan, dtype=np.float32)
            vals[valid_denom] = (numer[valid_denom] / denom[valid_denom]).astype(np.float32)
            out[~direct] = vals
    return out, idx[:, 0].astype(np.int32)


def _component_constrained_surface(
    coords: np.ndarray,
    center_coords: np.ndarray,
    center_z: np.ndarray,
    center_component: np.ndarray,
    *,
    k: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coords.size == 0 or center_coords.size == 0:
        n = coords.shape[0]
        return (
            np.full((n,), np.nan, dtype=np.float32),
            np.full((n,), -1, dtype=np.int32),
            np.full((n,), '', dtype=object),
        )
    valid = np.isfinite(center_coords).all(axis=1) & np.isfinite(center_z)
    center_coords = center_coords[valid]
    center_z = center_z[valid]
    center_component = np.asarray(center_component, dtype=object)[valid]
    if center_coords.shape[0] == 0:
        n = coords.shape[0]
        return (
            np.full((n,), np.nan, dtype=np.float32),
            np.full((n,), -1, dtype=np.int32),
            np.full((n,), '', dtype=object),
        )
    components = pd.unique(pd.Series(center_component.astype(str)))
    best_dist = np.full((coords.shape[0],), np.inf, dtype=np.float64)
    best_vals = np.full((coords.shape[0],), np.nan, dtype=np.float32)
    best_idx = np.full((coords.shape[0],), -1, dtype=np.int32)
    best_component = np.full((coords.shape[0],), '', dtype=object)
    for comp in components:
        comp_mask = center_component.astype(str) == str(comp)
        if not np.any(comp_mask):
            continue
        vals, local_idx = _query_backbone_surface(coords, center_coords[comp_mask], center_z[comp_mask], k=k)
        if np.all(~np.isfinite(vals)):
            continue
        comp_coords = center_coords[comp_mask]
        nearest_coords = np.full((coords.shape[0], 2), np.nan, dtype=np.float64)
        good_local = local_idx >= 0
        if np.any(good_local):
            nearest_coords[good_local] = comp_coords[local_idx[good_local]]
        comp_dist = np.sqrt(np.sum((coords - nearest_coords) ** 2, axis=1))
        better = np.isfinite(vals) & (comp_dist < best_dist)
        if np.any(better):
            best_dist[better] = comp_dist[better]
            best_vals[better] = vals[better]
            global_idx = np.flatnonzero(comp_mask)[local_idx[better]]
            best_idx[better] = global_idx.astype(np.int32)
            best_component[better] = str(comp)
    return best_vals, best_idx, best_component


def _apply_authoritative_pixel_lock(arr: np.ndarray, *, ds_transform, ds_shape: tuple[int, int], support: gpd.GeoDataFrame) -> int:
    if support is None or getattr(support, "empty", True):
        return 0
    use = support.copy()
    if "support_class" in use.columns:
        use = use.loc[use["support_class"].astype(str) == "authoritative_interior"].copy()
    if use.empty or "support_z_m" not in use.columns:
        return 0
    x = use.geometry.x.to_numpy(dtype=float)
    y = use.geometry.y.to_numpy(dtype=float)
    z = np.asarray(use["support_z_m"], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(finite):
        return 0
    rows, cols = rowcol(ds_transform, x[finite], y[finite], op=np.floor)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    inside = (rows >= 0) & (rows < ds_shape[0]) & (cols >= 0) & (cols < ds_shape[1])
    if not np.any(inside):
        return 0
    rows = rows[inside]
    cols = cols[inside]
    vals = z[finite][inside]
    # If multiple authoritative points land in the same pixel, use their mean.
    accum: dict[tuple[int, int], list[float]] = {}
    for r, c, v in zip(rows, cols, vals):
        accum.setdefault((int(r), int(c)), []).append(float(v))
    for (r, c), samples in accum.items():
        arr[r, c] = float(np.nanmean(np.asarray(samples, dtype=float)))
    return len(accum)



def build_v1_surface_products(
    *,
    cfg,
    river_dem: Path,
    channel_mask_tif: Path,
    river_dir: Path,
    support_points_path: Path,
    centerline_points_path: Path,
    logger=None,
) -> Dict[str, Any]:
    ensure_dir(river_dir)
    surface_path = river_dir / "river_primary_surface.tif"
    surface_summary_path = river_dir / "river_surface_summary.json"
    contract_path = river_dir / "river_primary_surface_contract.json"
    for p in [surface_path, surface_summary_path, contract_path]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    centerline = gpd.read_file(centerline_points_path) if Path(centerline_points_path).exists() else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry")
    if centerline is None or centerline.empty or "backbone_z_m" not in centerline.columns:
        raise RuntimeError("river_v1_surface_missing_backbone_points")
    centerline = centerline.loc[centerline.geometry.notnull() & ~centerline.geometry.is_empty].copy()
    if centerline.empty:
        raise RuntimeError("river_v1_surface_empty_backbone_points")
    center_coords = np.column_stack([
        centerline.geometry.x.to_numpy(dtype=np.float64),
        centerline.geometry.y.to_numpy(dtype=np.float64),
    ])
    center_z = np.asarray(centerline["backbone_z_m"], dtype=float)
    support = gpd.read_file(support_points_path) if Path(support_points_path).exists() else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=centerline.crs)
    center_support_class = centerline.get("support_class", np.repeat("unsupported_interior", len(centerline)))
    center_support_class = np.asarray(center_support_class, dtype=object)
    center_component_id = centerline.get("component_id", np.repeat("main", len(centerline)))
    center_component_id = np.asarray(center_component_id, dtype=object)

    with rasterio.open(river_dem) as dem_ds, rasterio.open(channel_mask_tif) as mask_ds:
        mask = mask_ds.read(1)
        if mask_ds.crs and dem_ds.crs and str(mask_ds.crs) != str(dem_ds.crs):
            raise RuntimeError("river_v1_surface_mask_crs_mismatch")
        profile = dem_ds.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
        out = np.full((dem_ds.height, dem_ds.width), float(profile["nodata"]), dtype=np.float32)
        rows, cols, coords = _pixel_centers_for_mask(mask > 0, dem_ds.transform)
        vals, nearest_idx, nearest_component = _component_constrained_surface(
            coords,
            center_coords,
            center_z,
            center_component_id,
            k=3,
        )
        valid = np.isfinite(vals)
        if np.any(valid):
            out[rows[valid], cols[valid]] = vals[valid]
        primary_source_class_counts: dict[str, int] = {}
        primary_component_counts: dict[str, int] = {}
        if nearest_idx.size:
            valid_idx = nearest_idx >= 0
            if np.any(valid_idx):
                nearest_classes = center_support_class[nearest_idx[valid_idx]].astype(str)
                uniq, counts = np.unique(nearest_classes, return_counts=True)
                primary_source_class_counts = {str(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}
                comp_vals = np.asarray(nearest_component[valid_idx], dtype=str)
                uniq_comp, comp_counts = np.unique(comp_vals, return_counts=True)
                primary_component_counts = {str(k): int(v) for k, v in zip(uniq_comp.tolist(), comp_counts.tolist())}
        authoritative_locked_pixel_count = _apply_authoritative_pixel_lock(
            out,
            ds_transform=dem_ds.transform,
            ds_shape=(dem_ds.height, dem_ds.width),
            support=support,
        )
        with rasterio.open(surface_path, "w", **profile) as out_ds:
            out_ds.write(out, 1)

    finite = np.isfinite(out) & (out != float(profile["nodata"]))
    unsupported_mask = np.zeros_like(finite, dtype=bool)
    if nearest_idx.size:
        valid_px = valid & (nearest_idx >= 0)
        if np.any(valid_px):
            nearest_classes = center_support_class[nearest_idx[valid_px]].astype(str)
            unsupported_here = nearest_classes == "unsupported_interior"
            unsupported_mask[rows[valid_px][unsupported_here], cols[valid_px][unsupported_here]] = True
    summary = {
        "status": "success",
        "workflow_stage": "river_v1_surface",
        "river_primary_surface_path": str(surface_path),
        "river_primary_surface_contract_path": str(contract_path),
        "river_surface_summary_path": str(surface_summary_path),
        "finite_pixel_count": int(np.count_nonzero(finite)),
        "channel_mask_pixel_count": int(np.count_nonzero(mask > 0)),
        "authoritative_locked_pixel_count": int(authoritative_locked_pixel_count),
        "unsupported_backbone_influenced_pixel_count": int(np.count_nonzero(unsupported_mask)),
        "primary_builder_mode": "component_constrained_backbone_idw_width_fill",
        "active_product_name": "river_primary_surface",
        "primary_surface_contract_ok": bool(np.count_nonzero(finite) > 0),
        "primary_source_class_counts": primary_source_class_counts,
        "primary_component_counts": primary_component_counts,
        "component_constrained_surface": True,
        "river_dem": str(river_dem),
        "channel_mask_tif": str(channel_mask_tif),
        "support_points_path": str(support_points_path),
        "centerline_points_path": str(centerline_points_path),
    }
    contract = {
        "status": "success",
        "active_product_name": "river_primary_surface",
        "active_product_role": "primary",
        "primary_surface_contract_ok": bool(np.count_nonzero(finite) > 0),
        "finite_pixel_count": int(np.count_nonzero(finite)),
        "authoritative_locked_pixel_count": int(authoritative_locked_pixel_count),
        "primary_builder_mode": "component_constrained_backbone_idw_width_fill",
        "component_constrained_surface": True,
        "direct_builder_inputs": {
            "support_points_path": str(support_points_path),
            "centerline_points_path": str(centerline_points_path),
            "river_dem": str(river_dem),
            "channel_mask_tif": str(channel_mask_tif),
        },
    }
    write_json(surface_summary_path, summary)
    write_json(contract_path, contract)
    return summary
