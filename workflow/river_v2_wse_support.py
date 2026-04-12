from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

from core.json_io import write_json
from river_bank_guidance import compute_bank_edge_guidance_from_authoritative, compute_lower_bank_wse_proxy_from_edge_guidance


def build_river_v2_wse_support_products(*, cfg: Any, support_raster: str | Path, channel_mask_tif: str | Path, centerline_points_path: str | Path, support_dir: str | Path, logger=None) -> dict[str, Any]:
    """Materialize explicit bank-edge WSE support products for the v2 pipeline.

    This is intentionally small and self-contained so the active v2 path no longer
    borrows the older river_v1 bank-guidance writer under the hood.
    """
    river_dem = Path(support_raster)
    channel_mask_tif = Path(channel_mask_tif)
    centerline_points_path = Path(centerline_points_path)
    support_dir = Path(support_dir)
    support_dir.mkdir(parents=True, exist_ok=True)

    bank_influence_path = support_dir / 'river_bank_influence.tif'
    bank_elevation_path = support_dir / 'river_bank_wse_edge_guidance.tif'
    bank_valid_mask_path = support_dir / 'river_bank_wse_edge_guidance_valid_mask.tif'
    bank_proxy_raster_path = support_dir / 'river_bank_wse_proxy_raster.tif'
    bank_profile_summary_path = support_dir / 'river_bank_wse_proxy_profile_summary.csv'
    bank_materialization_diagnostics_path = support_dir / 'river_bank_wse_materialization_diagnostics.json'
    bank_summary_path = support_dir / 'river_bank_guidance_summary.json'

    with rasterio.open(river_dem) as dem_ds:
        profile = dem_ds.profile.copy()
        auth = dem_ds.read(1).astype(np.float32)
        nodata = dem_ds.nodata
        if nodata is not None:
            auth[np.isclose(auth, np.float32(nodata))] = np.nan
        auth[~np.isfinite(auth)] = np.nan
        px_m = max(abs(float(getattr(dem_ds.transform, 'a', 1.0) or 1.0)), abs(float(getattr(dem_ds.transform, 'e', 1.0) or 1.0)), 1.0)

        with rasterio.open(channel_mask_tif) as mask_ds:
            if (mask_ds.width == dem_ds.width and mask_ds.height == dem_ds.height and str(mask_ds.crs) == str(dem_ds.crs) and tuple(mask_ds.transform) == tuple(dem_ds.transform)):
                corridor = mask_ds.read(1) > 0
                corridor_reprojected = False
            else:
                corridor_aligned = np.zeros((dem_ds.height, dem_ds.width), dtype=np.uint8)
                reproject(
                    source=rasterio.band(mask_ds, 1),
                    destination=corridor_aligned,
                    src_transform=mask_ds.transform,
                    src_crs=mask_ds.crs,
                    dst_transform=dem_ds.transform,
                    dst_crs=dem_ds.crs,
                    resampling=Resampling.nearest,
                    src_nodata=0,
                    dst_nodata=0,
                )
                corridor = corridor_aligned > 0
                corridor_reprojected = True

        max_bank_distance_m = max(float(getattr(cfg, 'river_scaffold_transition_m', 800.0) or 800.0) * 0.35, 120.0)
        edge_guidance_distance_m = max(px_m * 6.0, 18.0)
        _, bank_distance_m, bank_influence, raw_bank_elevation = compute_bank_edge_guidance_from_authoritative(
            auth,
            corridor,
            pixel_size_m=px_m,
            edge_guidance_distance_m=edge_guidance_distance_m,
            max_bank_distance_m=max_bank_distance_m,
        )
        proxy_radius_m = max(float(edge_guidance_distance_m) * 2.0, 30.0)
        bank_proxy_raster, bank_profile_summary = compute_lower_bank_wse_proxy_from_edge_guidance(
            raw_bank_elevation,
            bank_influence,
            centerline_points_path=centerline_points_path,
            transform=dem_ds.transform,
            source_crs=dem_ds.crs,
            proxy_radius_m=proxy_radius_m,
            quantile=0.25,
        )
        if not bank_profile_summary.empty:
            bank_profile_summary.to_csv(bank_profile_summary_path, index=False)

        finite_bank_pixels = int(np.count_nonzero(np.isfinite(raw_bank_elevation) & corridor))
        active_bank_influence_pixels = int(np.count_nonzero((bank_influence > 0.05) & np.isfinite(raw_bank_elevation) & corridor))
        if finite_bank_pixels <= 0:
            raise RuntimeError('river_v2_explicit_bank_guidance_missing_finite_pixels')
        if active_bank_influence_pixels <= 0:
            raise RuntimeError('river_v2_explicit_bank_guidance_missing_active_influence')

        out_prof = profile.copy()
        out_prof.pop('blockxsize', None)
        out_prof.pop('blockysize', None)
        out_prof.pop('BLOCKXSIZE', None)
        out_prof.pop('BLOCKYSIZE', None)
        out_prof.update(driver='GTiff', dtype='float32', count=1, nodata=np.float32(-9999.0), compress='deflate', tiled=False)
        nodata_out = np.float32(out_prof['nodata'])
        with rasterio.open(bank_influence_path, 'w', **out_prof) as dst:
            dst.write(bank_influence.astype(np.float32), 1)
        with rasterio.open(bank_elevation_path, 'w', **out_prof) as dst:
            dst.write(np.where(np.isfinite(raw_bank_elevation), raw_bank_elevation, nodata_out).astype(np.float32), 1)
        with rasterio.open(bank_valid_mask_path, 'w', **{**out_prof, 'dtype': 'uint8', 'nodata': 0}) as dst:
            dst.write((np.isfinite(raw_bank_elevation)).astype(np.uint8), 1)
        with rasterio.open(bank_proxy_raster_path, 'w', **out_prof) as dst:
            dst.write(np.where(np.isfinite(bank_proxy_raster), bank_proxy_raster, nodata_out).astype(np.float32), 1)

        diagnostics = {
            'status': 'success',
            'workflow_stage': 'river_v2_wse_support',
            'bank_guidance_source_raster': str(river_dem),
            'bank_guidance_source_crs': str(dem_ds.crs),
            'bank_guidance_source_shape': [int(profile['height']), int(profile['width'])],
            'corridor_mask_path': str(channel_mask_tif),
            'corridor_mask_reprojected_to_source_grid': bool(corridor_reprojected),
            'edge_guidance_finite_pixel_count': int(np.count_nonzero(np.isfinite(raw_bank_elevation))),
            'edge_guidance_unique_value_count': int(np.unique(np.round(raw_bank_elevation[np.isfinite(raw_bank_elevation)], 3)).size) if np.any(np.isfinite(raw_bank_elevation)) else 0,
            'edge_guidance_min': float(np.nanmin(raw_bank_elevation)) if np.any(np.isfinite(raw_bank_elevation)) else None,
            'edge_guidance_max': float(np.nanmax(raw_bank_elevation)) if np.any(np.isfinite(raw_bank_elevation)) else None,
            'edge_guidance_pixels_inside_corridor': finite_bank_pixels,
            'edge_guidance_pixels_outside_corridor': int(np.count_nonzero(np.isfinite(raw_bank_elevation) & (~corridor))),
            'profile_summary_rows': int(len(bank_profile_summary)),
            'proxy_radius_m': proxy_radius_m,
            'max_bank_distance_m': max_bank_distance_m,
            'edge_guidance_distance_m': edge_guidance_distance_m,
            'bank_valid_mask_path': str(bank_valid_mask_path),
        }
        write_json(bank_materialization_diagnostics_path, diagnostics)

    summary = {
        'status': 'success',
        'workflow_stage': 'river_v2_wse_support',
        'bank_guidance_source': 'station_lower_bank_wse_proxy_edge_band',
        'bank_guidance_source_raster': str(river_dem),
        'bank_profile_summary_path': str(bank_profile_summary_path) if bank_profile_summary_path.exists() else None,
        'bank_wse_profile_summary_path': str(bank_profile_summary_path) if bank_profile_summary_path.exists() else None,
        'corridor_mask_reprojected_to_source_grid': bool(diagnostics['corridor_mask_reprojected_to_source_grid']),
        'bank_influence_path': str(bank_influence_path),
        'bank_elevation_path': str(bank_elevation_path),
        'bank_wse_edge_guidance_path': str(bank_elevation_path),
        'bank_valid_mask_path': str(bank_valid_mask_path),
        'bank_materialization_diagnostics_path': str(bank_materialization_diagnostics_path),
        'bank_proxy_raster_path': str(bank_proxy_raster_path),
        'finite_bank_pixels': finite_bank_pixels,
        'active_bank_influence_pixels': active_bank_influence_pixels,
        'max_bank_distance_m': max_bank_distance_m,
        'edge_guidance_distance_m': edge_guidance_distance_m,
        'proxy_radius_m': proxy_radius_m,
        'bank_summary_path': str(bank_summary_path),
    }
    write_json(bank_summary_path, summary)
    if logger is not None and hasattr(logger, 'info'):
        logger.info('[RIVER][V2][WSE_SUPPORT] Explicit bank-edge guidance written: finite_bank_pixels=%d active_bank_influence_pixels=%d', finite_bank_pixels, active_bank_influence_pixels)
    return summary
