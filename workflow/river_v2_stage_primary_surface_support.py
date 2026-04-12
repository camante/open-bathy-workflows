from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from river_v2_context import RiverV2Context
from river_v2_contract import RiverV2StageResult, STAGE_RIVER_PRIMARY_SURFACE_SUPPORT
from river_v2_receipts import build_river_v2_raster_receipt, write_river_v2_receipt
from river_v2_validation import ensure_existing_path


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int
    transform: Affine
    crs: Any
    nodata: float
    dtype: str




def _primary_surface_domain_distance_limit(ctx: RiverV2Context, grid_spec: GridSpec) -> float | None:
    spacing = None
    if grid_spec.crs is not None:
        try:
            if bool(grid_spec.crs.is_projected):
                spacing = float(max(abs(grid_spec.transform.a), abs(grid_spec.transform.e)))
        except Exception:
            spacing = None
    if spacing is None or not np.isfinite(spacing) or spacing <= 0.0:
        return None
    centerline_spacing = float(getattr(getattr(ctx, "cfg", None), "river_centerline_sample_spacing_m", 0.0) or 0.0)
    limit = max(centerline_spacing * 3.0, spacing * 4.0, 60.0)
    return float(limit)


def build_primary_surface_domain(*, valid_mask: np.ndarray, distance_to_centerline: np.ndarray, max_distance_m: float | None) -> np.ndarray:
    if max_distance_m is None or (not np.isfinite(float(max_distance_m))) or float(max_distance_m) <= 0.0:
        return valid_mask.copy()
    domain = valid_mask & np.isfinite(distance_to_centerline) & (distance_to_centerline <= float(max_distance_m))
    if np.count_nonzero(domain) == 0:
        raise RuntimeError('river_v2_primary_surface_domain_empty_after_distance_limit')
    return domain

def resolve_primary_surface_target_grid(ctx: RiverV2Context, domain_mask_path: Path) -> GridSpec:
    with rasterio.open(domain_mask_path) as ds:
        nodata = float(ds.nodata) if ds.nodata is not None else 0.0
        return GridSpec(
            width=int(ds.width),
            height=int(ds.height),
            transform=ds.transform,
            crs=ds.crs,
            nodata=nodata,
            dtype=str(ds.dtypes[0]),
        )


def rasterize_centerline_presence(centerline_points_path: Path, grid_spec: GridSpec) -> np.ndarray:
    import geopandas as gpd

    gdf = gpd.read_file(centerline_points_path)
    if len(gdf) == 0:
        raise RuntimeError("river_v2_primary_support_no_centerline_points")
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        raise RuntimeError("river_v2_primary_support_centerline_geometry_empty")
    arr = rasterize(
        shapes,
        out_shape=(grid_spec.height, grid_spec.width),
        transform=grid_spec.transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return arr.astype(bool)


def compute_distance_to_centerline(valid_mask: np.ndarray, centerline_presence: np.ndarray, transform: Affine) -> np.ndarray:
    if valid_mask.shape != centerline_presence.shape:
        raise RuntimeError("river_v2_primary_support_shape_mismatch")
    if np.count_nonzero(centerline_presence & valid_mask) == 0:
        raise RuntimeError("river_v2_primary_support_no_centerline_cells_in_domain")
    sampling = (abs(float(transform.e)), abs(float(transform.a)))
    background = ~(centerline_presence & valid_mask)
    dist = distance_transform_edt(background, sampling=sampling).astype("float32")
    dist[~valid_mask] = np.nan
    dist[centerline_presence & valid_mask] = 0.0
    return dist


def write_primary_surface_domain_raster(*, primary_surface_domain: np.ndarray, grid_spec: GridSpec, domain_path: Path) -> Path:
    domain_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": grid_spec.width,
        "height": grid_spec.height,
        "count": 1,
        "crs": grid_spec.crs,
        "transform": grid_spec.transform,
        "compress": "deflate",
        "tiled": False,
    }
    with rasterio.open(domain_path, "w", dtype="uint8", nodata=0, **profile) as ds:
        ds.write(primary_surface_domain.astype("uint8"), 1)
    if not domain_path.exists():
        raise RuntimeError("river_v2_primary_support_write_failed")
    return domain_path


def validate_primary_surface_domain(domain_path: Path, *, expected_shape: tuple[int, int], expected_transform: Affine, expected_crs: Any) -> dict[str, Any]:
    with rasterio.open(domain_path) as ds:
        domain = ds.read(1)
        valid = np.isfinite(domain) & (domain > 0)
        return {
            "valid": bool(ds.shape == expected_shape and ds.transform == expected_transform and ds.crs == expected_crs and np.count_nonzero(valid) > 0),
            "grid_shape": [int(ds.height), int(ds.width)],
            "shape_match": bool(ds.shape == expected_shape),
            "transform_match": bool(ds.transform == expected_transform),
            "crs_match": bool(ds.crs == expected_crs),
            "domain_cell_count": int(np.count_nonzero(valid)),
        }


def build_primary_surface_support(ctx: RiverV2Context, *, centerline_points_path: Path, channel_mask_path: Path) -> RiverV2StageResult:
    domain_mask_path = ensure_existing_path(channel_mask_path, "river_v2_primary_support_domain_mask")
    centerline_points_path = ensure_existing_path(centerline_points_path, "river_v2_primary_support_centerline_points")
    grid_spec = resolve_primary_surface_target_grid(ctx, domain_mask_path)
    with rasterio.open(domain_mask_path) as ds:
        domain = ds.read(1)
    valid_mask = np.isfinite(domain) & (domain > 0)
    centerline_presence = rasterize_centerline_presence(centerline_points_path, grid_spec)
    distance_to_centerline = compute_distance_to_centerline(valid_mask, centerline_presence, grid_spec.transform)
    max_distance_m = _primary_surface_domain_distance_limit(ctx, grid_spec)
    primary_surface_domain = build_primary_surface_domain(
        valid_mask=valid_mask,
        distance_to_centerline=distance_to_centerline,
        max_distance_m=max_distance_m,
    )
    domain_path = write_primary_surface_domain_raster(
        primary_surface_domain=primary_surface_domain,
        grid_spec=grid_spec,
        domain_path=ctx.paths.river_primary_surface_domain,
    )
    validation = validate_primary_surface_domain(
        domain_path,
        expected_shape=(grid_spec.height, grid_spec.width),
        expected_transform=grid_spec.transform,
        expected_crs=grid_spec.crs,
    )
    if not validation.get("valid"):
        raise RuntimeError(f"river_v2_primary_support_invalid:{validation}")
    receipt = build_river_v2_raster_receipt(
        stage_id=STAGE_RIVER_PRIMARY_SURFACE_SUPPORT,
        output_artifacts=[str(domain_path)],
        input_artifacts=ctx.direct_stage_input_artifacts(centerline_points_path, domain_mask_path),
        vertical_reference=ctx.vertical_reference,
        warnings=[],
        source_logic="primary_surface_domain_from_channel_mask_clipped_by_centerline_distance_limit",
        validation=validation,
        grid_shape=(grid_spec.height, grid_spec.width),
        stats={
            "domain_cell_count": int(np.count_nonzero(primary_surface_domain)),
            "valid_mask_cell_count": int(np.count_nonzero(valid_mask)),
            "centerline_cell_count": int(np.count_nonzero(centerline_presence & valid_mask)),
            "max_distance_to_centerline_in_domain_m": float(np.nanmax(distance_to_centerline[primary_surface_domain])) if np.count_nonzero(primary_surface_domain) else None,
            "domain_distance_limit_m": max_distance_m,
        },
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.primary_surface_support_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_RIVER_PRIMARY_SURFACE_SUPPORT,
        output_artifact=domain_path,
        receipt_path=receipt_path,
        record_count=int(np.count_nonzero(valid_mask)),
        validation=validation,
        warnings=[],
        aux_outputs={},
    )
