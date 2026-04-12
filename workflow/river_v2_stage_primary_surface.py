from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from river_v2_context import RiverV2Context
from river_v2_contract import RiverV2StageResult, STAGE_RIVER_PRIMARY_SURFACE
from river_v2_receipts import build_river_v2_raster_receipt, write_river_v2_receipt
from river_v2_validation import ensure_existing_path, validate_primary_surface_raster


_REQUIRED_FIELDS = ("point_id", "station_m", "bed_backbone_z_m", "geometry")


class GridSpec:
    def __init__(self, width: int, height: int, transform, crs, nodata: float, dtype: str):
        self.width = int(width)
        self.height = int(height)
        self.transform = transform
        self.crs = crs
        self.nodata = float(nodata)
        self.dtype = str(dtype)


def resolve_primary_surface_target_grid(ctx: RiverV2Context, domain_mask_path: Path) -> GridSpec:
    with rasterio.open(domain_mask_path) as ds:
        nodata = float(ds.nodata) if ds.nodata is not None else 0.0
        return GridSpec(width=ds.width, height=ds.height, transform=ds.transform, crs=ds.crs, nodata=nodata, dtype=ds.dtypes[0])


def rasterize_centerline_presence(centerline_points_path: Path, grid_spec: GridSpec) -> np.ndarray:
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


def compute_distance_to_centerline(valid_mask: np.ndarray, centerline_presence: np.ndarray, transform) -> np.ndarray:
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


def validate_primary_surface_domain(domain_path: Path, *, expected_shape: tuple[int, int], expected_transform, expected_crs) -> dict[str, Any]:
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


def _numeric(values: pd.Series | Any) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def load_backbone_points(backbone_points_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(backbone_points_path)
    missing = [f for f in _REQUIRED_FIELDS if f not in gdf.columns]
    if missing:
        raise RuntimeError(f"river_v2_primary_surface_missing_fields:{missing}")
    if len(gdf) == 0:
        raise RuntimeError("river_v2_primary_surface_no_backbone_points")
    gdf = gdf.copy()
    gdf["station_m"] = pd.to_numeric(gdf["station_m"], errors="coerce")
    gdf["bed_backbone_z_m"] = pd.to_numeric(gdf["bed_backbone_z_m"], errors="coerce")
    gdf = gdf.sort_values(["station_m", "point_id"], kind="mergesort").reset_index(drop=True)
    return gdf


def rasterize_backbone_to_grid(backbone_gdf: gpd.GeoDataFrame, out_shape, transform) -> np.ndarray:
    values = []
    shapes = []
    for _, row in backbone_gdf.iterrows():
        geom = row.geometry
        z = row.get("bed_backbone_z_m")
        if geom is None or geom.is_empty or not np.isfinite(z):
            continue
        shapes.append(geom)
        values.append(float(z))
    if not shapes:
        raise RuntimeError("river_v2_primary_surface_no_finite_backbone_values")
    out = rasterize(
        list(zip(shapes, values)),
        out_shape=out_shape,
        transform=transform,
        fill=np.nan,
        all_touched=True,
        dtype="float32",
    )
    return out.astype("float32")


def interpolate_longitudinal_backbone_surface(backbone_raster: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    finite = np.isfinite(backbone_raster) & valid_mask
    if np.count_nonzero(finite) == 0:
        raise RuntimeError("river_v2_primary_surface_no_seed_cells")
    indices = distance_transform_edt(~finite, return_distances=False, return_indices=True)
    filled = backbone_raster[tuple(indices)]
    filled[~valid_mask] = np.nan
    return filled.astype("float32")


def _distance_to_bank(valid_mask: np.ndarray, transform) -> np.ndarray:
    sx = abs(float(getattr(transform, "a", 1.0) or 1.0))
    sy = abs(float(getattr(transform, "e", 1.0) or 1.0))
    dist = distance_transform_edt(valid_mask, sampling=(sy, sx)).astype("float32")
    # Make the first in-channel pixel at the NHD edge behave like distance 0 so
    # the taper can actually reach the bank, rather than stopping one pixel short.
    edge_step = float(min(sx, sy))
    if np.isfinite(edge_step) and edge_step > 0.0:
        dist = np.maximum(dist - edge_step, 0.0)
    dist[~valid_mask] = np.nan
    return dist.astype("float32")


def _load_aligned_reference(reference_path: Path | None, out_shape, transform, crs) -> np.ndarray | None:
    if reference_path is None or not Path(reference_path).exists():
        return None
    import rasterio.warp
    with rasterio.open(reference_path) as src:
        arr = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None:
            arr[np.isclose(arr, np.float32(nod))] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        if src.shape == tuple(out_shape) and tuple(src.transform) == tuple(transform) and str(src.crs) == str(crs):
            return arr
        dst = np.full(out_shape, np.nan, dtype="float32")
        rasterio.warp.reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=rasterio.warp.Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return dst




def _resolve_bank_taper_reference_path(ctx: RiverV2Context) -> Path | None:
    candidates = [
        getattr(ctx, "baseline_interpolated_path", None),
        getattr(ctx, "aligned_authoritative_base_path", None),
        getattr(ctx, "river_dem_path", None),
    ]
    for cand in candidates:
        if cand is None:
            continue
        path = Path(cand)
        if path.exists():
            return path
    fallback = ctx.out_dir.parent / "final" / "authoritative_base_aligned.tif"
    return fallback if fallback.exists() else None

def _apply_bank_taper(centerline_surface: np.ndarray, valid_mask: np.ndarray, distance_to_centerline_m: np.ndarray, distance_to_bank_m: np.ndarray, reference_surface: np.ndarray | None) -> tuple[np.ndarray, int]:
    if reference_surface is None:
        return centerline_surface.astype("float32"), 0
    out = centerline_surface.astype("float32").copy()
    d_center = np.asarray(distance_to_centerline_m, dtype="float32")
    d_bank = np.asarray(distance_to_bank_m, dtype="float32")
    denom = d_center + d_bank
    lateral_fraction = np.zeros_like(out, dtype="float32")
    finite = np.isfinite(d_center) & np.isfinite(d_bank) & (denom > 0.0)
    lateral_fraction[finite] = d_center[finite] / denom[finite]
    # Use a smooth full-width taper so the surface transitions gradually from
    # the centerline toward the bank reference across the entire channel width,
    # and reaches the baseline exactly at the NHD river edge.
    edge_weight = np.clip(lateral_fraction, 0.0, 1.0) ** 1.25
    blend_mask = valid_mask & np.isfinite(out) & np.isfinite(reference_surface) & (edge_weight > 0.0)
    if not np.any(blend_mask):
        return out, 0
    out[blend_mask] = ((1.0 - edge_weight[blend_mask]) * out[blend_mask]) + (edge_weight[blend_mask] * reference_surface[blend_mask])
    return out, int(np.count_nonzero(blend_mask))


def apply_valid_domain(centerline_surface: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = centerline_surface.astype("float32").copy()
    out[~valid_mask] = np.nan
    return out


def build_river_primary_surface(
    ctx: RiverV2Context,
    *,
    backbone_points_path: Path,
    channel_mask_path: Path,
    centerline_points_path: Path,
) -> RiverV2StageResult:
    backbone_points_path = ensure_existing_path(backbone_points_path, "river_v2_primary_surface_backbone")
    channel_mask_path = ensure_existing_path(channel_mask_path, "river_v2_primary_surface_channel_mask")
    centerline_points_path = ensure_existing_path(centerline_points_path, "river_v2_primary_surface_centerline")
    backbone = load_backbone_points(backbone_points_path)
    grid_spec = resolve_primary_surface_target_grid(ctx, channel_mask_path)
    with rasterio.open(channel_mask_path) as domain_ds:
        valid_mask = domain_ds.read(1)
        profile = domain_ds.profile.copy()
        out_shape = (domain_ds.height, domain_ds.width)
        transform = domain_ds.transform
    valid_mask = np.isfinite(valid_mask) & (valid_mask > 0)
    centerline_presence = rasterize_centerline_presence(centerline_points_path, grid_spec)
    distance_to_centerline = compute_distance_to_centerline(valid_mask, centerline_presence, transform)
    distance_to_bank = _distance_to_bank(valid_mask, transform)
    # Use the full NHD/river channel mask as the primary-surface domain so the
    # center-to-bank taper can extend all the way to the mapped river edge.
    max_distance = None
    primary_surface_domain = valid_mask.copy()
    domain_path = write_primary_surface_domain_raster(primary_surface_domain=primary_surface_domain, grid_spec=grid_spec, domain_path=ctx.paths.river_primary_surface_domain)
    domain_validation = validate_primary_surface_domain(domain_path, expected_shape=out_shape, expected_transform=transform, expected_crs=grid_spec.crs)
    if not domain_validation.get("valid"):
        raise RuntimeError(f"river_v2_primary_surface_domain_invalid:{domain_validation}")
    backbone_raster = rasterize_backbone_to_grid(backbone, out_shape, transform)
    centerline_surface = interpolate_longitudinal_backbone_surface(backbone_raster, primary_surface_domain)
    bank_reference_path = _resolve_bank_taper_reference_path(ctx)
    bank_reference = _load_aligned_reference(bank_reference_path, out_shape, transform, grid_spec.crs)
    tapered_surface, tapered_cell_count = _apply_bank_taper(
        centerline_surface,
        primary_surface_domain,
        distance_to_centerline,
        distance_to_bank,
        bank_reference,
    )
    primary_surface = apply_valid_domain(tapered_surface, primary_surface_domain)
    out_path = ctx.paths.river_primary_surface
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("BLOCKXSIZE", None)
    profile.pop("BLOCKYSIZE", None)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan, compress="deflate", tiled=False)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(primary_surface.astype("float32"), 1)
    validation = validate_primary_surface_raster(out_path, domain_path, domain_path)
    if not validation.get("valid"):
        raise RuntimeError(f"river_v2_primary_surface_invalid:{validation}")
    receipt = build_river_v2_raster_receipt(
        stage_id=STAGE_RIVER_PRIMARY_SURFACE,
        output_artifacts=[str(out_path)],
        input_artifacts=ctx.direct_stage_input_artifacts(backbone_points_path, channel_mask_path, centerline_points_path),
        vertical_reference=ctx.vertical_reference,
        warnings=[],
        source_logic="primary_surface_from_dense_backbone_point_rasterization_then_nearest_fill_within_full_nhd_channel_domain_followed_by_full_width_lateral_taper_to_nhd_edge",
        validation=validation,
        grid_shape=out_shape,
        stats={
            "seed_cell_count": int(np.count_nonzero(np.isfinite(backbone_raster) & primary_surface_domain)),
            "domain_cell_count": int(np.count_nonzero(primary_surface_domain)),
            "valid_cell_count": int(np.count_nonzero(np.isfinite(primary_surface) & primary_surface_domain)),
            "max_distance_to_centerline_m": float(np.nanmax(distance_to_centerline[primary_surface_domain])) if np.count_nonzero(primary_surface_domain) else None,
            "max_distance_to_bank_m": float(np.nanmax(distance_to_bank[primary_surface_domain])) if np.count_nonzero(primary_surface_domain) else None,
            "bank_taper_reference_path": str(bank_reference_path) if bank_reference_path is not None else None,
            "bank_taper_blended_cell_count": int(tapered_cell_count),
            "domain_distance_limit_m": max_distance,
        },
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.river_primary_surface_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_RIVER_PRIMARY_SURFACE,
        output_artifact=out_path,
        receipt_path=receipt_path,
        record_count=int(np.count_nonzero(np.isfinite(primary_surface) & primary_surface_domain)),
        validation=validation,
        warnings=[],
        aux_outputs={"primary_surface_domain": str(domain_path)},
    )
