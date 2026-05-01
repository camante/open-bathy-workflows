from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from legacy.river.river_v2_context import RiverV2Context
from legacy.river.river_v2_contract import RiverV2StageResult, STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT
from legacy.river.river_v2_receipts import build_river_v2_raster_receipt, write_river_v2_receipt


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_local_artifact_from_source(*, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        rel = os.path.relpath(source, destination.parent)
        destination.symlink_to(rel)
    except Exception:
        shutil.copy2(source, destination)






def _resolve_support_template(ctx: RiverV2Context) -> Path | None:
    for cand in (getattr(ctx, "river_dem_path", None), getattr(ctx, "channel_mask_path", None), getattr(ctx, "authoritative_base_path", None)):
        if cand is None:
            continue
        path = Path(cand)
        if path.exists():
            return path
    return None


def _write_support_raster_aligned_to_template(*, source: Path, template_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(template_path) as template_ds:
        profile = template_ds.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("BLOCKXSIZE", None)
        profile.pop("BLOCKYSIZE", None)
        nodata = template_ds.nodata
        if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
            nodata = -9999.0
        profile.update(driver="GTiff", dtype="float32", count=1, compress="deflate", tiled=False, nodata=float(nodata))
        dest_arr = np.full((template_ds.height, template_ds.width), float(nodata), dtype="float32")
        with rasterio.open(source) as src_ds:
            if src_ds.shape == template_ds.shape and src_ds.transform == template_ds.transform and src_ds.crs == template_ds.crs:
                arr = src_ds.read(1).astype("float32")
                src_nodata = src_ds.nodata
                if src_nodata is not None and not (isinstance(src_nodata, float) and np.isnan(src_nodata)):
                    arr[np.isclose(arr, np.float32(src_nodata))] = np.nan
                arr[~np.isfinite(arr)] = float(nodata)
                dest_arr = arr.astype("float32")
            else:
                reproject(
                    source=rasterio.band(src_ds, 1),
                    destination=dest_arr,
                    src_transform=src_ds.transform,
                    src_crs=src_ds.crs,
                    src_nodata=src_ds.nodata,
                    dst_transform=template_ds.transform,
                    dst_crs=template_ds.crs,
                    dst_nodata=float(nodata),
                    resampling=Resampling.nearest,
                )
        with rasterio.open(destination, "w", **profile) as dst:
            dst.write(dest_arr, 1)

def _write_empty_support_raster_from_template(*, template_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(template_path) as src:
        profile = src.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("BLOCKXSIZE", None)
        profile.pop("BLOCKYSIZE", None)
        profile.update(driver="GTiff", dtype="float32", count=1, compress="deflate", tiled=False)
        nodata = src.nodata
        if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
            nodata = -9999.0
        profile["nodata"] = float(nodata)
        arr = np.full((src.height, src.width), float(nodata), dtype="float32")
        with rasterio.open(destination, "w", **profile) as dst:
            dst.write(arr, 1)


def _resolve_empty_support_template(ctx: RiverV2Context) -> Path | None:
    for cand in (getattr(ctx, "channel_mask_path", None), getattr(ctx, "river_dem_path", None), getattr(ctx, "authoritative_base_path", None)):
        if cand is None:
            continue
        path = Path(cand)
        if path.exists():
            return path
    return None

def _validate_authoritative_support_raster(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float32")
        nodata = ds.nodata
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            arr[np.isclose(arr, np.float32(nodata))] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        finite = int(np.count_nonzero(np.isfinite(arr)))
        return {
            "valid": True,
            "has_finite_pixels": finite > 0,
            "finite_pixel_count": finite,
            "grid_shape": [int(ds.height), int(ds.width)],
            "crs": ds.crs.to_string() if ds.crs is not None else None,
            "is_projected": bool(ds.crs is not None and not getattr(ds.crs, "is_geographic", False)),
            "transform": tuple(ds.transform) if ds.transform is not None else None,
            "nodata": None if nodata is None else float(nodata) if np.isfinite(nodata) else None,
            "min_z_m": None if finite == 0 else float(np.nanmin(arr)),
            "max_z_m": None if finite == 0 else float(np.nanmax(arr)),
        }


def run_authoritative_support_stage(
    ctx: RiverV2Context,
    *,
    authoritative_sampling_raster_path: Path | None,
    authoritative_bed_support_points_path: Path | None = None,
) -> RiverV2StageResult:
    selected = Path(authoritative_sampling_raster_path) if authoritative_sampling_raster_path is not None else None
    trusted_mode = str(getattr(ctx, "trusted_support_mode", "unknown") or "unknown")
    out_path = ctx.paths.authoritative_sampling_support
    source_logic = "river_v2_authoritative_support:explicit_projected_authoritative_sampling_raster"
    warnings: list[str] = []
    if selected is None or not selected.exists():
        if trusted_mode == "low_support_no_trusted_support" or str(getattr(ctx, "system_support_status", "")) == "scaffold_inferred":
            template = _resolve_empty_support_template(ctx)
            if template is None:
                raise RuntimeError("river_v2_authoritative_support_missing_sampling_raster")
            _write_empty_support_raster_from_template(template_path=template, destination=out_path)
            selected = out_path
            source_logic = "river_v2_authoritative_support:explicit_empty_raster_for_low_support_mode"
            warnings.append("authoritative_support_missing_sampling_raster_low_support_empty_artifact")
        else:
            raise RuntimeError("river_v2_authoritative_support_missing_sampling_raster")
    else:
        template = _resolve_support_template(ctx)
        if template is None:
            _write_local_artifact_from_source(source=selected, destination=out_path)
        else:
            _write_support_raster_aligned_to_template(source=selected, template_path=template, destination=out_path)
    diagnostics = {
        "selected_authoritative_sampling_raster": str(selected),
        "trusted_support_mode": trusted_mode,
        "support_policy_source": str(getattr(ctx, "support_policy_source", "unknown") or "unknown"),
        "authoritative_bed_support_points_path": str(authoritative_bed_support_points_path) if authoritative_bed_support_points_path is not None else None,
        "trusted_support_artifact_path": str(getattr(ctx, "trusted_support_artifact_path", None)) if getattr(ctx, "trusted_support_artifact_path", None) is not None else None,
    }
    validation = _validate_authoritative_support_raster(out_path)
    diagnostics.update(validation)
    ctx.paths.authoritative_sampling_support_diagnostics.write_text(json.dumps(_jsonable(diagnostics), indent=2, sort_keys=True), encoding="utf-8")
    receipt = build_river_v2_raster_receipt(
        stage_id=STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT,
        output_artifacts=[str(out_path)],
        input_artifacts=ctx.direct_stage_input_artifacts(
            selected,
            authoritative_bed_support_points_path,
            ctx.trusted_support_artifact_path,
            ctx.resolved_inputs_manifest_path,
        ),
        vertical_reference=ctx.vertical_reference,
        warnings=warnings + (["authoritative_support_zero_finite_pixels"] if not validation.get("has_finite_pixels", False) else []),
        source_logic=source_logic,
        validation=validation,
        grid_shape=tuple(validation.get("grid_shape", [])) if validation.get("grid_shape") else None,
        stats=validation,
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.authoritative_sampling_support_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_CENTERLINE_AUTHORITATIVE_SUPPORT,
        output_artifact=out_path,
        receipt_path=receipt_path,
        record_count=int(validation.get("finite_pixel_count", 0)),
        validation=validation,
        warnings=warnings + (["authoritative_support_zero_finite_pixels"] if not validation.get("has_finite_pixels", False) else []),
        aux_outputs={"authoritative_support_raster": str(out_path)},
    )


__all__ = ["run_authoritative_support_stage"]
