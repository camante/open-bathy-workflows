from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


@dataclass(frozen=True)
class FinalDemCompositionSummary:
    support_applied_pixel_count: int
    guidance_applied_pixel_count: int
    background_applied_pixel_count: int
    final_finite_count: int


def _summary_from_arrays(
    *,
    measured: np.ndarray,
    support: np.ndarray,
    background: np.ndarray,
    guidance: np.ndarray,
    take: np.ndarray,
    measured_nodata: float | None,
    guidance_nodata: float | None,
    final_arr: np.ndarray,
    final_nodata: float | None,
) -> FinalDemCompositionSummary:
    measured_valid = np.isfinite(measured)
    if measured_nodata is not None:
        measured_valid &= measured != np.float32(measured_nodata)

    guidance_valid = np.isfinite(guidance)
    if guidance_nodata is not None:
        guidance_valid &= guidance != np.float32(guidance_nodata)

    support_apply = support & measured_valid
    guidance_apply = take & guidance_valid & ~support_apply
    background_apply = ~(support_apply | guidance_apply)

    final_finite = np.isfinite(final_arr)
    if final_nodata is not None:
        final_finite &= final_arr != np.float32(final_nodata)

    return FinalDemCompositionSummary(
        support_applied_pixel_count=int(np.count_nonzero(support_apply)),
        guidance_applied_pixel_count=int(np.count_nonzero(guidance_apply)),
        background_applied_pixel_count=int(np.count_nonzero(background_apply)),
        final_finite_count=int(np.count_nonzero(final_finite)),
    )


def compose_final_dem(*, measured_path: Path, support_mask_path: Path, background_path: Path, guidance_path: Path, take_mask_path: Path, dst_path: Path) -> FinalDemCompositionSummary:
    with rasterio.open(measured_path) as measured_ds, \
         rasterio.open(support_mask_path) as support_ds, \
         rasterio.open(background_path) as background_ds, \
         rasterio.open(guidance_path) as guidance_ds, \
         rasterio.open(take_mask_path) as take_ds:
        measured = measured_ds.read(1).astype(np.float32)
        support = support_ds.read(1) > 0
        background = background_ds.read(1).astype(np.float32)
        guidance = guidance_ds.read(1).astype(np.float32)
        take = take_ds.read(1) > 0

        measured_nodata = measured_ds.nodata
        background_nodata = background_ds.nodata
        guidance_nodata = guidance_ds.nodata

        measured_valid = np.isfinite(measured)
        if measured_nodata is not None:
            measured_valid &= measured != np.float32(measured_nodata)

        guidance_valid = np.isfinite(guidance)
        if guidance_nodata is not None:
            guidance_valid &= guidance != np.float32(guidance_nodata)

        support_apply = support & measured_valid
        guidance_apply = take & guidance_valid & ~support_apply

        out = background.copy()
        out[support_apply] = measured[support_apply]
        out[guidance_apply] = guidance[guidance_apply]

        out_nodata = np.float32(background_nodata if background_nodata is not None else -9999.0)
        out[~np.isfinite(out)] = out_nodata

        profile = background_ds.profile.copy()
        for k in ("blockxsize", "blockysize", "BLOCKXSIZE", "BLOCKYSIZE"):
            profile.pop(k, None)
        profile.update(dtype="float32", count=1, nodata=out_nodata, compress="deflate")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink()
        with rasterio.open(dst_path, "w", **profile) as out_ds:
            out_ds.write(out.astype(np.float32), 1)

    with rasterio.open(dst_path) as final_ds:
        final_arr = final_ds.read(1).astype(np.float32)
        final_nodata = final_ds.nodata

    return _summary_from_arrays(
        measured=measured,
        support=support,
        background=background,
        guidance=guidance,
        take=take,
        measured_nodata=measured_nodata,
        guidance_nodata=guidance_nodata,
        final_arr=final_arr,
        final_nodata=final_nodata,
    )


def summarize_existing_final_dem(*, measured_path: Path, support_mask_path: Path, background_path: Path, guidance_path: Path, take_mask_path: Path, final_dem_path: Path) -> FinalDemCompositionSummary:
    with rasterio.open(measured_path) as measured_ds, \
         rasterio.open(support_mask_path) as support_ds, \
         rasterio.open(background_path) as background_ds, \
         rasterio.open(guidance_path) as guidance_ds, \
         rasterio.open(take_mask_path) as take_ds, \
         rasterio.open(final_dem_path) as final_ds:
        measured = measured_ds.read(1).astype(np.float32)
        support = support_ds.read(1) > 0
        background = background_ds.read(1).astype(np.float32)
        guidance = guidance_ds.read(1).astype(np.float32)
        take = take_ds.read(1) > 0
        final_arr = final_ds.read(1).astype(np.float32)
        measured_nodata = measured_ds.nodata
        guidance_nodata = guidance_ds.nodata
        final_nodata = final_ds.nodata

    return _summary_from_arrays(
        measured=measured,
        support=support,
        background=background,
        guidance=guidance,
        take=take,
        measured_nodata=measured_nodata,
        guidance_nodata=guidance_nodata,
        final_arr=final_arr,
        final_nodata=final_nodata,
    )


__all__ = ["FinalDemCompositionSummary", "compose_final_dem", "summarize_existing_final_dem"]
