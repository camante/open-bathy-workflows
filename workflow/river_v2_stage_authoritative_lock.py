from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from river_v2_context import RiverV2Context
from river_v2_contract import RiverV2StageResult, STAGE_RIVER_PRIMARY_SURFACE_LOCKED
from river_v2_receipts import build_river_v2_raster_receipt, write_river_v2_receipt
from river_v2_validation import ensure_existing_path, validate_authoritative_locked_raster


def _read_aligned(src_path: Path, *, out_shape, out_transform, out_crs, nodata_value: float) -> np.ndarray:
    with rasterio.open(src_path) as src:
        if src.shape == out_shape and src.transform == out_transform and src.crs == out_crs:
            arr = src.read(1).astype("float32")
        else:
            arr = np.full(out_shape, nodata_value, dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=out_transform,
                dst_crs=out_crs,
                dst_nodata=nodata_value,
                resampling=Resampling.nearest,
            )
    return arr.astype("float32")


def build_authoritative_locked_primary_surface(
    ctx: RiverV2Context,
    *,
    primary_surface_path: Path,
    authoritative_base_path: Path,
) -> RiverV2StageResult:
    primary_surface_path = ensure_existing_path(primary_surface_path, "river_v2_primary_surface_for_lock")
    authoritative_base = ensure_existing_path(authoritative_base_path, "river_v2_authoritative_lock_authoritative_base")

    with rasterio.open(primary_surface_path) as surf_ds:
        surface = surf_ds.read(1).astype("float32")
        profile = surf_ds.profile.copy()
        out_shape = (surf_ds.height, surf_ds.width)
        out_transform = surf_ds.transform
        out_crs = surf_ds.crs
        surf_nodata = surf_ds.nodata
        if surf_nodata is None or (isinstance(surf_nodata, float) and np.isnan(surf_nodata)):
            surf_nodata = -9999.0
        surface_valid = np.isfinite(surface)

    auth = _read_aligned(authoritative_base, out_shape=out_shape, out_transform=out_transform, out_crs=out_crs, nodata_value=float(surf_nodata))
    auth_valid = np.isfinite(auth) & (auth != float(surf_nodata))

    locked = surface.copy()
    diff = np.full(out_shape, np.nan, dtype="float32")
    changed = auth_valid & surface_valid & (np.abs(surface - auth) > 1.0e-6)
    diff[auth_valid] = np.where(surface_valid[auth_valid], surface[auth_valid] - auth[auth_valid], np.nan).astype("float32")
    locked[auth_valid] = auth[auth_valid]
    locked[~np.isfinite(locked)] = np.nan

    out_path = ctx.paths.river_primary_surface_authoritative_applied
    diff_path = ctx.paths.river_primary_surface_authoritative_lock_diff
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("BLOCKXSIZE", None)
    profile.pop("BLOCKYSIZE", None)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan, compress="deflate", tiled=False)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(locked.astype("float32"), 1)
    with rasterio.open(diff_path, "w", **profile) as ds:
        ds.write(diff.astype("float32"), 1)

    validation = validate_authoritative_locked_raster(out_path, primary_surface_path, authoritative_base)
    if not validation.get("valid"):
        raise RuntimeError(f"river_v2_authoritative_lock_invalid:{validation}")

    receipt = build_river_v2_raster_receipt(
        stage_id=STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
        output_artifacts=[str(out_path), str(diff_path)],
        input_artifacts=ctx.direct_stage_input_artifacts(primary_surface_path, authoritative_base),
        vertical_reference=ctx.vertical_reference,
        warnings=[],
        source_logic="authoritative_lock_overwrites_finite_authoritative_base_cells_onto_river_primary_surface_before_final_dem_routing",
        validation=validation,
        grid_shape=out_shape,
        stats={
            "primary_surface_valid_cell_count": int(np.count_nonzero(surface_valid)),
            "authoritative_locked_cell_count": int(np.count_nonzero(auth_valid)),
            "changed_before_lock_count": int(np.count_nonzero(changed)),
            "unlocked_primary_surface_valid_cell_count": int(np.count_nonzero(np.isfinite(locked) & (~auth_valid))),
        },
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.river_primary_surface_authoritative_applied_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_RIVER_PRIMARY_SURFACE_LOCKED,
        output_artifact=out_path,
        receipt_path=receipt_path,
        record_count=int(np.count_nonzero(np.isfinite(locked))),
        validation=validation,
        warnings=[],
        aux_outputs={"lock_diff_before_overwrite": str(diff_path)},
    )
