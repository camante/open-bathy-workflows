from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds

from legacy.river.river_v2_context import RiverV2Context
from legacy.river.river_v2_contract import RiverV2StageResult
from legacy.river.river_v2_receipts import build_river_v2_raster_receipt, write_river_v2_receipt


@dataclass(frozen=True)
class RiverV2ExportSubsetResult:
    exported_primary_surface_path: Optional[Path]
    exported_locked_surface_path: Path
    export_receipt_path: Path


def _bounds_overlap(source_bounds, export_bounds, *, tolerance: float = 1.0e-6) -> bool:
    sw, ss, se, sn = [float(v) for v in source_bounds]
    ew, es, ee, en = [float(v) for v in export_bounds]
    overlap_w = max(sw, ew)
    overlap_e = min(se, ee)
    overlap_s = max(ss, es)
    overlap_n = min(sn, en)
    return (overlap_e - overlap_w) > -tolerance and (overlap_n - overlap_s) > -tolerance


def _source_crs_export_bounds(*, src_crs, export_mask_path: Path | None, export_aoi: str | None) -> tuple[tuple[float, float, float, float], str | None, bool]:
    transformed = False
    export_crs = None
    if export_mask_path is not None and Path(export_mask_path).exists():
        with rasterio.open(export_mask_path) as mask_ds:
            export_bounds = tuple(float(v) for v in mask_ds.bounds)
            export_crs = mask_ds.crs
    else:
        if not export_aoi:
            raise RuntimeError('river_v2_export_subset_missing_export_aoi')
        west, east, south, north = [float(v) for v in str(export_aoi).split('/')]
        export_bounds = (west, south, east, north)
        export_crs = rasterio.crs.CRS.from_epsg(4326)

    if src_crs is not None and export_crs is not None and str(src_crs) != str(export_crs):
        export_bounds = tuple(float(v) for v in transform_bounds(export_crs, src_crs, *export_bounds, densify_pts=21))
        transformed = True
    return export_bounds, (str(export_crs) if export_crs is not None else None), transformed


def _validate_export_overlap_with_solve_extent(source_path: Path, *, export_mask_path: Path | None, export_aoi: str | None) -> dict:
    with rasterio.open(source_path) as src:
        source_bounds = tuple(float(v) for v in src.bounds)
        export_bounds_source_crs, export_bounds_crs, transformed = _source_crs_export_bounds(
            src_crs=src.crs,
            export_mask_path=export_mask_path,
            export_aoi=export_aoi,
        )
        overlaps = _bounds_overlap(source_bounds, export_bounds_source_crs)
        if not overlaps:
            raise RuntimeError(
                f"river_v2_export_subset_outside_solve_extent:source_bounds={source_bounds}:export_bounds={export_bounds_source_crs}:export_bounds_crs={export_bounds_crs}"
            )
        return {
            'source_bounds': list(source_bounds),
            'export_bounds_in_source_crs': list(export_bounds_source_crs),
            'export_bounds_crs': export_bounds_crs,
            'export_overlaps_source_extent': True,
            'export_bounds_transformed_to_source_crs': bool(transformed),
            'source_width': int(src.width),
            'source_height': int(src.height),
            'source_crs': str(src.crs) if src.crs is not None else None,
        }


def _subset_to_export_grid(source_path: Path, destination_path: Path, *, export_mask_path: Path | None, export_aoi: str | None) -> tuple[Path, dict]:
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict = _validate_export_overlap_with_solve_extent(source_path, export_mask_path=export_mask_path, export_aoi=export_aoi)
    if export_mask_path is not None and Path(export_mask_path).exists():
        with rasterio.open(source_path) as src, rasterio.open(export_mask_path) as mask_ds:
            dst_nodata = float(src.nodata) if src.nodata is not None and np.isfinite(float(src.nodata)) else -9999.0
            dst_arr = np.full((mask_ds.height, mask_ds.width), np.float32(dst_nodata), dtype='float32')
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=mask_ds.transform,
                dst_crs=mask_ds.crs,
                src_nodata=src.nodata,
                dst_nodata=dst_nodata,
                resampling=Resampling.nearest,
            )
            mask = mask_ds.read(1) > 0
            dst_arr[~mask] = np.float32(dst_nodata)
            profile = mask_ds.profile.copy()
            profile.update(driver='GTiff', count=1, dtype='float32', nodata=dst_nodata, compress='deflate')
            profile.pop('blockxsize', None); profile.pop('blockysize', None)
            with rasterio.open(destination_path, 'w', **profile) as dst:
                dst.write(dst_arr.astype('float32'), 1)
            stats.update({
                'subset_only': True,
                'export_grid_source': str(export_mask_path),
                'resampling': 'nearest',
                'finite_pixel_count': int(np.count_nonzero(np.isfinite(dst_arr) & ~np.isclose(dst_arr, np.float32(dst_nodata)))),
                'nodata_pixel_count': int(np.count_nonzero(np.isclose(dst_arr, np.float32(dst_nodata)) | ~np.isfinite(dst_arr))),
            })
            finite = dst_arr[np.isfinite(dst_arr) & ~np.isclose(dst_arr, np.float32(dst_nodata))]
            if finite.size:
                stats['min_z_m'] = float(np.min(finite))
                stats['max_z_m'] = float(np.max(finite))
        return destination_path, stats

    with rasterio.open(source_path) as src:
        if not export_aoi:
            raise RuntimeError('river_v2_export_subset_missing_export_aoi')
        export_bounds_source_crs, _, _ = _source_crs_export_bounds(
            src_crs=src.crs,
            export_mask_path=None,
            export_aoi=export_aoi,
        )
        west, south, east, north = export_bounds_source_crs
        window = from_bounds(west, south, east, north, src.transform)
        window = window.round_offsets().round_lengths()
        dst_nodata = float(src.nodata) if src.nodata is not None and np.isfinite(float(src.nodata)) else -9999.0
        data = src.read(1, window=window, boundless=True, fill_value=dst_nodata).astype('float32')
        transform = rasterio.windows.transform(window, src.transform)
        profile = src.profile.copy()
        profile.update(height=data.shape[0], width=data.shape[1], transform=transform, dtype='float32', nodata=dst_nodata, compress='deflate')
        profile.pop('blockxsize', None); profile.pop('blockysize', None)
        with rasterio.open(destination_path, 'w', **profile) as dst:
            dst.write(data, 1)
        stats.update({
            'subset_only': True,
            'export_grid_source': 'aoi_window',
            'resampling': 'none',
            'finite_pixel_count': int(np.count_nonzero(np.isfinite(data) & ~np.isclose(data, np.float32(dst_nodata)))),
            'nodata_pixel_count': int(np.count_nonzero(np.isclose(data, np.float32(dst_nodata)) | ~np.isfinite(data))),
        })
        finite = data[np.isfinite(data) & ~np.isclose(data, np.float32(dst_nodata))]
        if finite.size:
            stats['min_z_m'] = float(np.min(finite))
            stats['max_z_m'] = float(np.max(finite))
    return destination_path, stats


def run_export_subset_stage(
    ctx: RiverV2Context,
    *,
    solve_primary_surface_path: Path | None,
    solve_locked_surface_path: Path,
    export_channel_mask_path: Path | None,
) -> RiverV2StageResult:
    exported_primary = None
    warnings: list[str] = []
    export_mask = Path(export_channel_mask_path) if export_channel_mask_path is not None else None
    if solve_primary_surface_path is not None and Path(solve_primary_surface_path).exists():
        exported_primary, _ = _subset_to_export_grid(
            Path(solve_primary_surface_path),
            ctx.paths.river_primary_surface,
            export_mask_path=export_mask,
            export_aoi=ctx.export_aoi or ctx.canonical_solve_aoi,
        )
    exported_locked, stats = _subset_to_export_grid(
        Path(solve_locked_surface_path),
        ctx.paths.river_primary_surface_authoritative_applied,
        export_mask_path=export_mask,
        export_aoi=ctx.export_aoi or ctx.canonical_solve_aoi,
    )
    validation = {
        'export_aoi': str(ctx.export_aoi or ctx.canonical_solve_aoi or ''),
        'export_mask_path': str(export_mask) if export_mask is not None else None,
        'source_role': 'solve_domain',
        'artifact_role': 'public_export_subset',
        'subset_only': True,
        'solve_primary_surface_path': str(solve_primary_surface_path) if solve_primary_surface_path is not None else None,
        'solve_locked_surface_path': str(solve_locked_surface_path),
        'exported_primary_surface_path': str(exported_primary) if exported_primary is not None else None,
        'exported_locked_surface_path': str(exported_locked),
        'source_bounds': stats.get('source_bounds'),
        'export_bounds_in_source_crs': stats.get('export_bounds_in_source_crs'),
        'export_bounds_crs': stats.get('export_bounds_crs'),
        'export_overlaps_source_extent': stats.get('export_overlaps_source_extent'),
        'export_bounds_transformed_to_source_crs': stats.get('export_bounds_transformed_to_source_crs'),
        'source_width': stats.get('source_width'),
        'source_height': stats.get('source_height'),
        'source_crs': stats.get('source_crs'),
    }
    receipt = build_river_v2_raster_receipt(
        stage_id='river_export_subset',
        output_artifacts=[str(v) for v in [exported_primary, exported_locked] if v is not None],
        input_artifacts=[str(v) for v in [solve_primary_surface_path, solve_locked_surface_path, export_mask] if v is not None],
        vertical_reference=str(getattr(ctx, 'vertical_reference', 'unknown') or 'unknown'),
        warnings=warnings,
        source_logic='Subset solve-domain River v2 rasters to the export AOI/grid without re-solving or re-locking.',
        validation=validation,
        stats=stats,
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.river_export_subset_receipt)
    if exported_primary is not None:
        write_river_v2_receipt(receipt, ctx.paths.river_primary_surface_receipt)
    write_river_v2_receipt(receipt, ctx.paths.river_primary_surface_authoritative_applied_receipt)
    aux_outputs = {
        'river_primary_surface': str(exported_primary) if exported_primary is not None else '',
        'river_primary_surface_authoritative_applied': str(exported_locked),
        'river_v2_export_subset_receipt': str(receipt_path),
        'river_v2_primary_surface_export': str(exported_primary) if exported_primary is not None else '',
        'river_v2_primary_surface_authoritative_applied_export': str(exported_locked),
        'river_v2_primary_surface_solve_domain': str(solve_primary_surface_path) if solve_primary_surface_path is not None else '',
        'river_v2_primary_surface_authoritative_applied_solve_domain': str(solve_locked_surface_path),
    }
    return RiverV2StageResult(
        stage_id='river_export_subset',
        output_artifact=exported_locked,
        receipt_path=receipt_path,
        record_count=int(stats.get('finite_pixel_count', 0)),
        validation=validation,
        warnings=warnings,
        aux_outputs={k: v for k, v in aux_outputs.items() if v},
    )
