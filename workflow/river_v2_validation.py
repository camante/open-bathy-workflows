from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional


def first_existing_path(*candidates: Any) -> Optional[Path]:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def ensure_existing_path(path: str | Path | None, label: str) -> Path:
    if path is None:
        raise RuntimeError(f"{label}_missing")
    resolved = Path(path)
    if not resolved.exists():
        raise RuntimeError(f"{label}_missing:{resolved}")
    return resolved


def dedupe_input_artifacts(paths: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in paths:
        if value is None:
            continue
        item = str(value)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def validate_support_rasters(valid_mask_path: str | Path, distance_path: str | Path, domain_mask_path: str | Path) -> dict[str, Any]:
    valid_mask_path = ensure_existing_path(valid_mask_path, "valid_mask")
    distance_path = ensure_existing_path(distance_path, "distance_to_centerline")
    domain_mask_path = ensure_existing_path(domain_mask_path, "domain_mask")
    with rasterio.open(valid_mask_path) as valid_ds, rasterio.open(distance_path) as dist_ds, rasterio.open(domain_mask_path) as domain_ds:
        valid_mask = valid_ds.read(1).astype(bool)
        dist = dist_ds.read(1)
        domain = domain_ds.read(1)
        shape_match = valid_ds.shape == dist_ds.shape == domain_ds.shape
        transform_match = valid_ds.transform == dist_ds.transform == domain_ds.transform
        crs_match = valid_ds.crs == dist_ds.crs == domain_ds.crs
        domain_valid = np.isfinite(domain) & (domain > 0)
        finite_distance = np.isfinite(dist[valid_mask]) if np.count_nonzero(valid_mask) else np.array([], dtype=bool)
        return {
            "valid": bool(shape_match and transform_match and crs_match and np.count_nonzero(valid_mask) > 0 and np.count_nonzero(valid_mask & domain_valid) == np.count_nonzero(valid_mask) and np.all(finite_distance)),
            "grid_shape": [int(valid_ds.height), int(valid_ds.width)],
            "valid_cell_count": int(np.count_nonzero(valid_mask)),
            "distance_finite_count": int(np.count_nonzero(np.isfinite(dist[valid_mask]))),
            "shape_match": bool(shape_match),
            "transform_match": bool(transform_match),
            "crs_match": bool(crs_match),
        }


def read_valid_data_stats(raster_path: str | Path) -> dict[str, Any]:
    raster_path = ensure_existing_path(raster_path, "raster")
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        valid = np.isfinite(arr) if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)) else (np.isfinite(arr) & (arr != nodata))
        if np.count_nonzero(valid) == 0:
            return {"valid_count": 0, "min": None, "max": None, "mean": None}
        vals = arr[valid]
        return {"valid_count": int(vals.size), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)), "mean": float(np.nanmean(vals))}


def validate_primary_surface_raster(primary_surface_path: str | Path, valid_mask_path: str | Path, domain_mask_path: str | Path | None) -> dict[str, Any]:
    primary_surface_path = ensure_existing_path(primary_surface_path, "primary_surface")
    valid_mask_path = ensure_existing_path(valid_mask_path, "primary_surface_valid_mask")
    domain_mask_path = ensure_existing_path(domain_mask_path, "primary_surface_domain_mask")
    with rasterio.open(primary_surface_path) as surf_ds, rasterio.open(valid_mask_path) as valid_ds, rasterio.open(domain_mask_path) as domain_ds:
        surf = surf_ds.read(1)
        valid_mask = valid_ds.read(1).astype(bool)
        domain = domain_ds.read(1)
        domain_valid = np.isfinite(domain) & (domain > 0)
        shape_match = surf_ds.shape == valid_ds.shape == domain_ds.shape
        transform_match = surf_ds.transform == valid_ds.transform == domain_ds.transform
        crs_match = surf_ds.crs == valid_ds.crs == domain_ds.crs
        surface_valid = np.isfinite(surf) & valid_mask
        outside_domain_valid = int(np.count_nonzero(np.isfinite(surf) & ~domain_valid))
        stats = read_valid_data_stats(primary_surface_path)
        return {
            "valid": bool(shape_match and transform_match and crs_match and stats["valid_count"] > 0 and outside_domain_valid == 0),
            "grid_shape": [int(surf_ds.height), int(surf_ds.width)],
            "shape_match": bool(shape_match),
            "transform_match": bool(transform_match),
            "crs_match": bool(crs_match),
            "valid_cell_count": int(stats["valid_count"]),
            "outside_domain_valid_cell_count": outside_domain_valid,
            "min_z": stats["min"],
            "max_z": stats["max"],
            "mean_z": stats["mean"],
            "domain_valid_cell_count": int(np.count_nonzero(domain_valid)),
            "support_valid_cell_count": int(np.count_nonzero(valid_mask)),
            "surface_valid_mask_overlap_count": int(np.count_nonzero(surface_valid)),
        }


def validate_authoritative_locked_raster(locked_path: str | Path, primary_surface_path: str | Path, authoritative_base_path: str | Path) -> dict[str, Any]:
    locked_path = ensure_existing_path(locked_path, "locked_primary_surface")
    primary_surface_path = ensure_existing_path(primary_surface_path, "primary_surface_for_lock_validation")
    authoritative_base_path = ensure_existing_path(authoritative_base_path, "authoritative_base_for_lock_validation")
    with rasterio.open(locked_path) as locked_ds, rasterio.open(primary_surface_path) as primary_ds, rasterio.open(authoritative_base_path) as auth_ds:
        locked = locked_ds.read(1).astype("float32")
        primary = primary_ds.read(1).astype("float32")
        shape_match = locked_ds.shape == primary_ds.shape
        transform_match = locked_ds.transform == primary_ds.transform
        crs_match = locked_ds.crs == primary_ds.crs
        auth = np.full((locked_ds.height, locked_ds.width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(auth_ds, 1),
            destination=auth,
            src_transform=auth_ds.transform,
            src_crs=auth_ds.crs,
            src_nodata=auth_ds.nodata,
            dst_transform=locked_ds.transform,
            dst_crs=locked_ds.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        auth_valid = np.isfinite(auth)
        locked_match = auth_valid & np.isfinite(locked) & (np.abs(locked - auth) <= 1.0e-6)
        unlocked_valid = (~auth_valid) & np.isfinite(locked)
        authoritative_locked_cell_count = int(np.count_nonzero(auth_valid))
        authoritative_locked_mismatch_count = int(np.count_nonzero(auth_valid & (~locked_match)))
        lock_mode = "authoritative_lock_applied" if authoritative_locked_cell_count > 0 else "no_lockable_authoritative_cells"
        return {
            "valid": bool(shape_match and transform_match and crs_match and authoritative_locked_mismatch_count == 0),
            "shape_match": bool(shape_match),
            "transform_match": bool(transform_match),
            "crs_match": bool(crs_match),
            "authoritative_lock_mode": lock_mode,
            "authoritative_locked_cell_count": authoritative_locked_cell_count,
            "authoritative_locked_match_count": int(np.count_nonzero(locked_match)),
            "authoritative_locked_mismatch_count": authoritative_locked_mismatch_count,
            "unlocked_valid_cell_count": int(np.count_nonzero(unlocked_valid)),
            "primary_surface_valid_cell_count": int(np.count_nonzero(np.isfinite(primary))),
        }
