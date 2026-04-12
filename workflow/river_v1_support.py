from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point

from core.json_io import write_json
from core.paths import ensure_dir
from river_structured_scaffold import select_retained_river_features
from river_bank_guidance import compute_bank_edge_guidance_from_authoritative, compute_lower_bank_wse_proxy_from_edge_guidance


def _build_point_gdf(df: pd.DataFrame, *, target_crs, logger=None) -> gpd.GeoDataFrame:
    if df is None or getattr(df, "empty", True):
        return gpd.GeoDataFrame(df if df is not None else pd.DataFrame(), geometry=[], crs=str(target_crs) if target_crs is not None else None)
    use = df.copy()
    if "lon" not in use.columns or "lat" not in use.columns:
        raise RuntimeError("river_v1_support_missing_lon_lat")
    lon = pd.to_numeric(use["lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(use["lat"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat)
    use = use.loc[valid].reset_index(drop=True)
    lon = lon[valid]
    lat = lat[valid]
    if target_crs is not None and str(target_crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
        try:
            from pyproj import Transformer
            tx = Transformer.from_crs("EPSG:4326", str(target_crs), always_xy=True)
            x, y = tx.transform(lon, lat)
        except Exception:
            if logger is not None:
                logger.debug("[RIVER][V1][SUPPORT] Failed projecting support points to target CRS.", exc_info=True)
            raise
        geom = gpd.points_from_xy(x, y)
        gdf = gpd.GeoDataFrame(use, geometry=geom, crs=str(target_crs))
    else:
        geom = gpd.points_from_xy(lon, lat)
        gdf = gpd.GeoDataFrame(use, geometry=geom, crs="EPSG:4326")
    return gdf


def _classify_support_role(row) -> str:
    role = str(getattr(row, "authoritative_role", "") or "").strip().lower()
    if "bank" in role:
        return "bank_margin_only"
    return "authoritative_interior"


def _mask_membership(coords_xy: np.ndarray, *, mask_path: Path) -> np.ndarray:
    if coords_xy.size == 0:
        return np.zeros((0,), dtype=bool)
    with rasterio.open(mask_path) as ds:
        data = ds.read(1)
        rows, cols = rowcol(ds.transform, coords_xy[:, 0], coords_xy[:, 1], op=np.floor)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        inside = (
            (rows >= 0)
            & (rows < ds.height)
            & (cols >= 0)
            & (cols < ds.width)
        )
        out = np.zeros_like(inside, dtype=bool)
        valid_idx = np.where(inside)[0]
        if valid_idx.size:
            out[valid_idx] = data[rows[valid_idx], cols[valid_idx]] > 0
        return out




def _write_bank_guidance_products(*, cfg, river_dem: Path, channel_mask_tif: Path, centerline_points_path: Path, river_dir: Path, logger=None) -> dict[str, str | int | float | None]:
    bank_influence_path = river_dir / "river_bank_influence.tif"
    bank_elevation_path = river_dir / "river_bank_wse_edge_guidance.tif"
    bank_valid_mask_path = river_dir / "river_bank_wse_edge_guidance_valid_mask.tif"
    bank_summary_path = river_dir / "river_bank_guidance_summary.json"
    bank_materialization_diagnostics_path = river_dir / "river_bank_wse_materialization_diagnostics.json"
    bank_profile_summary_path = river_dir / "river_bank_wse_proxy_profile_summary.csv"
    with rasterio.open(river_dem) as dem_ds:
        prof = dem_ds.profile.copy()
        auth = dem_ds.read(1).astype(np.float32)
        nod = dem_ds.nodata
        if nod is not None:
            auth[np.isclose(auth, np.float32(nod))] = np.nan
        auth[~np.isfinite(auth)] = np.nan
        px_m = max(abs(float(getattr(dem_ds.transform, "a", 1.0) or 1.0)), abs(float(getattr(dem_ds.transform, "e", 1.0) or 1.0)), 1.0)
    corridor_reprojected = False
    with rasterio.open(channel_mask_tif) as mask_ds:
        if (
            mask_ds.width == prof["width"]
            and mask_ds.height == prof["height"]
            and str(mask_ds.crs) == str(dem_ds.crs)
            and tuple(mask_ds.transform) == tuple(dem_ds.transform)
        ):
            corridor = mask_ds.read(1) > 0
        else:
            from rasterio.warp import reproject, Resampling
            corridor_aligned = np.zeros((prof["height"], prof["width"]), dtype=np.uint8)
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
    max_bank_distance_m = max(float(getattr(cfg, "river_scaffold_transition_m", 800.0) or 800.0) * 0.35, 120.0)
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
        transform=prof["transform"],
        source_crs=dem_ds.crs,
        proxy_radius_m=proxy_radius_m,
        quantile=0.25,
    )
    if not bank_profile_summary.empty:
        bank_profile_summary.to_csv(bank_profile_summary_path, index=False)
    finite_bank_pixels = int(np.count_nonzero(np.isfinite(raw_bank_elevation) & corridor))
    active_bank_influence_pixels = int(np.count_nonzero((bank_influence > 0.05) & np.isfinite(raw_bank_elevation) & corridor))
    monotone_adjustment_max_m = 0.0
    if not bank_profile_summary.empty and "bank_wse_proxy_adjustment_m" in bank_profile_summary.columns:
        adj = pd.to_numeric(bank_profile_summary["bank_wse_proxy_adjustment_m"], errors="coerce").to_numpy(dtype=float)
        if np.any(np.isfinite(adj)):
            monotone_adjustment_max_m = float(np.nanmax(np.abs(adj)))
    if logger is not None and hasattr(logger, "info"):
        logger.info(
            "[RIVER][GUIDANCE] Lower-bank WSE proxy derived from authoritative DEM edge samples: finite_bank_pixels=%d active_bank_influence_pixels=%d edge_guidance_distance_m=%.1f proxy_radius_m=%.1f monotone_adjustment_max_m=%.2f",
            finite_bank_pixels,
            active_bank_influence_pixels,
            edge_guidance_distance_m,
            proxy_radius_m,
            monotone_adjustment_max_m,
        )
    if finite_bank_pixels <= 0:
        raise RuntimeError("river_v1_explicit_bank_guidance_missing_finite_pixels")
    if active_bank_influence_pixels <= 0:
        raise RuntimeError("river_v1_explicit_bank_guidance_missing_active_influence")
    out_prof = prof.copy()
    out_prof.update(dtype="float32", count=1, nodata=np.float32(-9999.0), compress="deflate")
    nodata_out = np.float32(out_prof["nodata"])
    with rasterio.open(bank_influence_path, "w", **out_prof) as dst:
        dst.write(bank_influence.astype(np.float32), 1)
    with rasterio.open(bank_elevation_path, "w", **out_prof) as dst:
        dst.write(np.where(np.isfinite(raw_bank_elevation), raw_bank_elevation, nodata_out).astype(np.float32), 1)
    with rasterio.open(bank_valid_mask_path, "w", **{**out_prof, "dtype": "uint8", "nodata": 0}) as dst:
        dst.write((np.isfinite(raw_bank_elevation)).astype(np.uint8), 1)
    bank_proxy_raster_path = river_dir / "river_bank_wse_proxy_raster.tif"
    with rasterio.open(bank_proxy_raster_path, "w", **out_prof) as dst:
        dst.write(np.where(np.isfinite(bank_proxy_raster), bank_proxy_raster, nodata_out).astype(np.float32), 1)
    materialization = {
        "status": "success",
        "bank_guidance_source_raster": str(river_dem),
        "bank_guidance_source_crs": str(dem_ds.crs),
        "bank_guidance_source_shape": [int(prof["height"]), int(prof["width"])],
        "corridor_mask_path": str(channel_mask_tif),
        "corridor_mask_reprojected_to_source_grid": bool(corridor_reprojected),
        "edge_guidance_finite_pixel_count": int(np.count_nonzero(np.isfinite(raw_bank_elevation))),
        "edge_guidance_unique_value_count": int(np.unique(np.round(raw_bank_elevation[np.isfinite(raw_bank_elevation)], 3)).size) if np.any(np.isfinite(raw_bank_elevation)) else 0,
        "edge_guidance_min": float(np.nanmin(raw_bank_elevation)) if np.any(np.isfinite(raw_bank_elevation)) else None,
        "edge_guidance_max": float(np.nanmax(raw_bank_elevation)) if np.any(np.isfinite(raw_bank_elevation)) else None,
        "edge_guidance_pixels_inside_corridor": int(np.count_nonzero(np.isfinite(raw_bank_elevation) & corridor)),
        "edge_guidance_pixels_outside_corridor": int(np.count_nonzero(np.isfinite(raw_bank_elevation) & (~corridor))),
        "profile_summary_rows": int(len(bank_profile_summary)),
        "profile_summary_unique_monotone_values": int(pd.unique(pd.to_numeric(bank_profile_summary["bank_wse_proxy_monotone_m"], errors="coerce").dropna().round(3)).size) if (not bank_profile_summary.empty and "bank_wse_proxy_monotone_m" in bank_profile_summary.columns) else 0,
        "materialization_status": "success",
        "bank_valid_mask_path": str(bank_valid_mask_path),
    }
    write_json(bank_materialization_diagnostics_path, materialization)
    summary = {
        "status": "success",
        "workflow_stage": "river_v1_bank_guidance",
        "bank_guidance_source": "station_lower_bank_wse_proxy_edge_band",
        "bank_guidance_source_raster": str(river_dem),
        "bank_profile_summary_path": str(bank_profile_summary_path) if bank_profile_summary_path.exists() else None,
        "corridor_mask_reprojected_to_source_grid": bool(corridor_reprojected),
        "bank_influence_path": str(bank_influence_path),
        "bank_elevation_path": str(bank_elevation_path),
        "bank_valid_mask_path": str(bank_valid_mask_path),
        "bank_materialization_diagnostics_path": str(bank_materialization_diagnostics_path),
        "bank_proxy_raster_path": str(bank_proxy_raster_path),
        "finite_bank_pixels": finite_bank_pixels,
        "active_bank_influence_pixels": active_bank_influence_pixels,
        "max_bank_distance_m": max_bank_distance_m,
        "edge_guidance_distance_m": edge_guidance_distance_m,
        "proxy_radius_m": proxy_radius_m,
        "monotone_adjustment_max_m": monotone_adjustment_max_m,
    }
    write_json(bank_summary_path, summary)
    summary["bank_summary_path"] = str(bank_summary_path)
    summary["bank_wse_edge_guidance_path"] = str(bank_elevation_path)
    summary["bank_wse_proxy_raster_path"] = str(bank_proxy_raster_path)
    return summary


def build_v1_support_products(
    *,
    cfg,
    network_gpkg: Path,
    river_dem: Path,
    channel_mask_tif: Path,
    river_dir: Path,
    load_support_points_fn: Callable[[Any], Optional[pd.DataFrame]],
    logger=None,
) -> Dict[str, Any]:
    ensure_dir(river_dir)
    support_points_path = river_dir / "river_support_points.gpkg"
    support_summary_path = river_dir / "river_support_summary.json"

    if support_points_path.exists():
        support_points_path.unlink()

    with rasterio.open(river_dem) as dem_ds:
        target_crs = dem_ds.crs

    selection = select_retained_river_features(
        network_gpkg,
        target_crs=target_crs,
        min_stream_order=max(int(getattr(cfg, "river_mainstem_min_order", 5) or 5), 1),
        keep_top_components=6,
    )
    flows = selection.flows
    polygons = selection.polygons
    retained_meta = selection.metadata
    support_df = load_support_points_fn(cfg)
    support_gdf = _build_point_gdf(support_df, target_crs=target_crs, logger=logger) if support_df is not None else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=str(target_crs) if target_crs is not None else None)
    if not support_gdf.empty:
        support_gdf = support_gdf[support_gdf.geometry.notnull() & ~support_gdf.geometry.is_empty].copy()
        coords = np.column_stack([support_gdf.geometry.x.to_numpy(dtype=float), support_gdf.geometry.y.to_numpy(dtype=float)])
        inside_mask = _mask_membership(coords, mask_path=channel_mask_tif)
        support_gdf["inside_channel_mask"] = inside_mask.astype(np.uint8)
        support_gdf = support_gdf.loc[inside_mask].copy()
    else:
        support_gdf["inside_channel_mask"] = np.array([], dtype=np.uint8)

    if not support_gdf.empty:
        support_gdf["support_class"] = [
            _classify_support_role(row) for row in support_gdf.itertuples()
        ]
        support_gdf["support_z_m"] = pd.to_numeric(support_gdf.get("depth_m"), errors="coerce").astype(float)
        keep_cols = [
            c for c in [
                "source",
                "value_semantics",
                "authoritative_role",
                "role_confidence",
                "distance_to_bank_m",
                "component_half_width_est_m",
                "normalized_channel_position",
                "inside_channel_mask",
                "support_class",
                "support_z_m",
            ] if c in support_gdf.columns
        ]
        if polygons is not None and not getattr(polygons, "empty", True):
            try:
                support_gdf = gpd.sjoin(support_gdf, polygons[["geometry"]], predicate="within", how="left")
                inside_poly = support_gdf["index_right"].notna().to_numpy(dtype=bool)
                support_gdf["inside_retained_polygon"] = inside_poly.astype(np.uint8)
                support_gdf = support_gdf.loc[inside_poly].copy()
                keep_cols.append("inside_retained_polygon")
                if "index_right" in support_gdf.columns:
                    support_gdf = support_gdf.drop(columns=["index_right"])
            except Exception:
                if logger is not None and hasattr(logger, "debug"):
                    logger.debug("[RIVER][V1][SUPPORT] Polygon join failed; keeping channel-mask-filtered support only.", exc_info=True)
        support_gdf.to_file(support_points_path, driver="GPKG")
    else:
        empty = gpd.GeoDataFrame(columns=["support_class", "support_z_m", "inside_channel_mask", "geometry"], geometry="geometry", crs=str(target_crs) if target_crs is not None else None)
        empty.to_file(support_points_path, driver="GPKG")

    class_counts = support_gdf["support_class"].astype(str).value_counts().to_dict() if not support_gdf.empty and "support_class" in support_gdf.columns else {}
    source_counts = support_gdf["source"].astype(str).value_counts().to_dict() if not support_gdf.empty and "source" in support_gdf.columns else {}
    summary = {
        "status": "success",
        "workflow_stage": "river_v1_support",
        "support_points_path": str(support_points_path),
        "support_point_count": int(len(support_gdf)),
        "support_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "retained_flow_count": int(retained_meta.get("retained_flow_count", 0)),
        "retained_polygon_count": int(retained_meta.get("retained_polygon_count", 0)),
        "channel_mask_tif": str(channel_mask_tif),
        "river_dem": str(river_dem),
        "support_semantics": "authoritative_interior_or_bank_margin_only",
    }
    write_json(support_summary_path, summary)
    summary["support_summary_path"] = str(support_summary_path)
    return summary


def build_v1_bank_guidance_products(*, cfg, river_dem: Path, channel_mask_tif: Path, centerline_points_path: Path, river_dir: Path, logger=None) -> Dict[str, Any]:
    return _write_bank_guidance_products(
        cfg=cfg,
        river_dem=river_dem,
        channel_mask_tif=channel_mask_tif,
        centerline_points_path=centerline_points_path,
        river_dir=river_dir,
        logger=logger,
    )
