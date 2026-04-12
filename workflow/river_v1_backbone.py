from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from core.json_io import write_json
from core.paths import ensure_dir
from river_structured_scaffold import build_centerline_points, select_retained_river_features


def _sample_raster(points: gpd.GeoDataFrame, raster_path: Path) -> np.ndarray:
    if points is None or getattr(points, "empty", True):
        return np.asarray([], dtype=np.float32)
    coords = [(geom.x, geom.y) if geom is not None and not geom.is_empty else (np.nan, np.nan) for geom in points.geometry]
    out = np.full((len(coords),), np.nan, dtype=np.float32)
    with rasterio.open(raster_path) as ds:
        nodata = ds.nodata
        valid_idx = [i for i, (x, y) in enumerate(coords) if np.isfinite(x) and np.isfinite(y)]
        if valid_idx:
            vals = list(ds.sample([coords[i] for i in valid_idx]))
            arr = np.asarray(vals, dtype=float).reshape((-1,))
            if nodata is not None:
                arr[np.isclose(arr, float(nodata))] = np.nan
            out[np.asarray(valid_idx, dtype=int)] = arr.astype(np.float32)
    return out


def _nearest_support(centerline: gpd.GeoDataFrame, support: gpd.GeoDataFrame) -> pd.DataFrame:
    if centerline is None or getattr(centerline, "empty", True):
        return pd.DataFrame(index=pd.RangeIndex(0))
    cols = [c for c in ["support_class", "support_z_m", "source", "authoritative_role"] if c in support.columns]
    if support is None or getattr(support, "empty", True):
        return pd.DataFrame(index=centerline.index)
    try:
        joined = gpd.sjoin_nearest(centerline, support[cols + ["geometry"]], how="left", distance_col="nearest_support_distance_m")
    except Exception:
        return pd.DataFrame(index=centerline.index)
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])
    return pd.DataFrame(joined.drop(columns=["geometry"]))


def _rolling_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    window = max(int(window), 1)
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.full(values.shape, np.nan, dtype=np.float32)
    for i in range(values.size):
        sl = values[max(0, i - half): min(values.size, i + half + 1)]
        if np.any(np.isfinite(sl)):
            out[i] = float(np.nanmedian(sl))
    return out


def build_v1_backbone_products(
    *,
    cfg,
    network_gpkg: Path,
    river_dem: Path,
    channel_mask_tif: Path,
    river_dir: Path,
    support_points_path: Path,
    logger=None,
) -> Dict[str, Any]:
    ensure_dir(river_dir)
    centerline_points_path = river_dir / "river_centerline_points.gpkg"
    backbone_summary_path = river_dir / "river_backbone_summary.json"
    if centerline_points_path.exists():
        centerline_points_path.unlink()

    with rasterio.open(river_dem) as dem_ds:
        target_crs = dem_ds.crs
        pixel_m = max(abs(float(dem_ds.transform.a)), abs(float(dem_ds.transform.e)))
    spacing_m = float(getattr(cfg, "river_centerline_sample_spacing_m", None) or max(pixel_m * 3.0, 20.0))

    selection = select_retained_river_features(
        network_gpkg,
        target_crs=target_crs,
        min_stream_order=max(int(getattr(cfg, "river_mainstem_min_order", 5) or 5), 1),
        keep_top_components=6,
    )
    flows = selection.flows
    polygons = selection.polygons
    retained_meta = selection.metadata
    centerline = build_centerline_points(flows, polygons, spacing_m=spacing_m, raster_path=river_dem, logger=logger)
    if centerline is None or getattr(centerline, "empty", True):
        summary = {
            "status": "failed",
            "reason": "no_centerline_points",
            "retained_flow_count": int(retained_meta.get("retained_flow_count", 0)),
        }
        write_json(backbone_summary_path, summary)
        raise RuntimeError("river_v1_backbone_no_centerline_points")

    centerline = centerline.sort_values(by=[c for c in ["component_id", "station_m"] if c in centerline.columns], kind="mergesort").reset_index(drop=True)
    centerline["centerline_base_z_m"] = _sample_raster(centerline, river_dem)

    support = gpd.read_file(support_points_path) if Path(support_points_path).exists() else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)
    nearest = _nearest_support(centerline[["geometry"]].copy(), support)
    if not nearest.empty:
        for col in nearest.columns:
            centerline[col] = nearest[col].to_numpy()
    else:
        centerline["support_class"] = None
        centerline["support_z_m"] = np.nan
        centerline["nearest_support_distance_m"] = np.nan

    near_dist = pd.to_numeric(centerline.get("nearest_support_distance_m"), errors="coerce").to_numpy(dtype=float)
    support_cls = centerline.get("support_class", pd.Series([None] * len(centerline))).astype(str)
    support_z = pd.to_numeric(centerline.get("support_z_m"), errors="coerce").to_numpy(dtype=float)

    canonical = np.full((len(centerline),), "unsupported_interior", dtype=object)
    authoritative = np.isfinite(support_z) & (support_cls == "authoritative_interior") & (near_dist <= max(spacing_m, 25.0))
    bank_only = np.isfinite(support_z) & (support_cls == "bank_margin_only") & (near_dist <= max(spacing_m, 25.0))
    weak = (~authoritative) & (~bank_only) & np.isfinite(near_dist) & (near_dist <= max(float(getattr(cfg, "authoritative_support_density_radius_m", 250.0) or 250.0), spacing_m * 2.0))
    canonical[authoritative] = "authoritative_interior"
    canonical[bank_only] = "bank_margin_only"
    canonical[weak] = "weak_supported_interior"
    centerline["support_class"] = canonical
    centerline["is_mainstem"] = True

    backbone = centerline["centerline_base_z_m"].to_numpy(dtype=float)
    base_z = centerline["centerline_base_z_m"].to_numpy(dtype=float)
    station = pd.to_numeric(centerline.get("station_m"), errors="coerce").to_numpy(dtype=float)
    component = centerline.get("component_id", pd.Series(["main"] * len(centerline))).astype(str).to_numpy(dtype=object)
    source_kind = np.where(authoritative, "authoritative_support", np.where(np.isfinite(base_z), "centerline_base_authoritative", "missing")).astype(object)
    bridged_from_anchors = np.zeros((len(centerline),), dtype=bool)
    unbridgeable_missing_base = np.zeros((len(centerline),), dtype=bool)
    update_mask = canonical != "authoritative_interior"
    centerline_base_anchor = np.isfinite(base_z) & np.isfinite(station)
    for comp in pd.unique(component):
        comp_mask = component == comp
        comp_station = station[comp_mask]
        comp_indices = np.flatnonzero(comp_mask)
        comp_update_mask = update_mask[comp_mask]
        comp_support_anchor_mask = authoritative[comp_mask] & np.isfinite(support_z[comp_mask]) & np.isfinite(comp_station)
        comp_base_anchor_mask = centerline_base_anchor[comp_mask]
        # Prefer authoritative support anchors, but fall back to any finite centerline-base anchors
        # because river_dem is already the authoritative-only river DEM product.
        if np.count_nonzero(comp_support_anchor_mask) >= 2:
            anchor_x = comp_station[comp_support_anchor_mask]
            anchor_z = support_z[comp_mask][comp_support_anchor_mask]
            anchor_source = "support"
        elif np.count_nonzero(comp_base_anchor_mask) >= 2:
            anchor_x = comp_station[comp_base_anchor_mask]
            anchor_z = base_z[comp_mask][comp_base_anchor_mask]
            anchor_source = "centerline_base"
        else:
            anchor_x = np.asarray([], dtype=float)
            anchor_z = np.asarray([], dtype=float)
            anchor_source = "none"
        if anchor_x.size >= 2:
            order = np.argsort(anchor_x)
            interp = np.interp(comp_station, anchor_x[order], anchor_z[order])
            target_idx = comp_indices[comp_update_mask]
            backbone[target_idx] = interp[comp_update_mask]
            source_kind[target_idx] = f"interpolated_backbone_{anchor_source}"
            bridged_from_anchors[target_idx] = ~np.isfinite(base_z[target_idx])
        else:
            smooth = _rolling_nanmedian(backbone[comp_mask], window=5)
            smooth_idx = comp_indices[comp_update_mask & np.isfinite(smooth)]
            backbone[smooth_idx] = smooth[comp_update_mask & np.isfinite(smooth)]
            source_kind[smooth_idx] = "smoothed_backbone"
            missing_idx = comp_indices[comp_update_mask & (~np.isfinite(base_z[comp_mask]))]
            if missing_idx.size:
                unbridgeable_missing_base[missing_idx] = True

    delta = backbone - base_z
    unsupported = canonical == "unsupported_interior"
    requested_nonzero = unsupported & (
        (np.isfinite(delta) & (np.abs(delta) > 0.01))
        | (bridged_from_anchors & np.isfinite(backbone))
    )
    changed = requested_nonzero.copy()
    no_op_candidates = unsupported & np.isfinite(delta) & (np.abs(delta) <= 0.01)
    missing_base = unsupported & (~np.isfinite(base_z))
    centerline["backbone_z_m"] = backbone.astype(np.float32)
    centerline["backbone_adjustment_delta_m"] = delta.astype(np.float32)
    centerline["backbone_adjustment_applied"] = np.asarray(requested_nonzero, dtype=np.uint8)
    centerline["source_kind"] = source_kind
    centerline["backbone_bridged_from_anchors"] = np.asarray(bridged_from_anchors, dtype=np.uint8)
    centerline["backbone_unbridgeable_missing_base"] = np.asarray(unbridgeable_missing_base, dtype=np.uint8)
    centerline["active_target_source"] = np.where(canonical == "authoritative_interior", "authoritative_interior", "backbone_led_interior")
    centerline["active_target_z_m"] = centerline["backbone_z_m"].astype(np.float32)
    centerline["backbone_noop_consistent_base"] = np.asarray(no_op_candidates, dtype=np.uint8)
    centerline["backbone_missing_base_z"] = np.asarray(missing_base, dtype=np.uint8)
    centerline["backbone_inert_failure"] = np.asarray(unsupported & (~changed) & (~no_op_candidates), dtype=np.uint8)
    centerline.to_file(centerline_points_path, driver="GPKG")

    counts = pd.Series(canonical).value_counts().to_dict()
    summary = {
        "status": "success",
        "workflow_stage": "river_v1_backbone",
        "centerline_points_path": str(centerline_points_path),
        "centerline_point_count": int(len(centerline)),
        "retained_flow_count": int(retained_meta.get("retained_flow_count", 0)),
        "retained_polygon_count": int(retained_meta.get("retained_polygon_count", 0)),
        "support_class_counts": {str(k): int(v) for k, v in counts.items()},
        "unsupported_centerline_count": int(np.count_nonzero(unsupported)),
        "unsupported_requested_adjustment_count": int(np.count_nonzero(requested_nonzero)),
        "unsupported_noop_consistent_base_count": int(np.count_nonzero(no_op_candidates)),
        "unsupported_missing_base_count": int(np.count_nonzero(missing_base)),
        "unsupported_bridged_from_anchors_count": int(np.count_nonzero(unsupported & bridged_from_anchors)),
        "unsupported_unbridgeable_missing_base_count": int(np.count_nonzero(unsupported & unbridgeable_missing_base)),
        "support_anchor_count": int(np.count_nonzero(authoritative & np.isfinite(support_z))),
        "centerline_base_anchor_count": int(np.count_nonzero(centerline_base_anchor)),
        "backbone_adjusted_station_count": int(np.count_nonzero(changed)),
        "backbone_adjustment_max_abs_m": float(np.nanmax(np.abs(delta[np.isfinite(delta)]))) if np.any(np.isfinite(delta)) else 0.0,
        "backbone_spacing_m": float(spacing_m),
        "river_dem": str(river_dem),
        "channel_mask_tif": str(channel_mask_tif),
    }
    write_json(backbone_summary_path, summary)
    summary["backbone_summary_path"] = str(backbone_summary_path)
    should_fail = int(summary["unsupported_requested_adjustment_count"]) > 0 and int(summary["backbone_adjusted_station_count"]) <= 0
    if int(summary["unsupported_unbridgeable_missing_base_count"]) > 0:
        should_fail = True
    if should_fail:
        fail_summary = dict(summary)
        failure_reason = "unsupported_backbone_requested_but_inert"
        if int(summary["unsupported_unbridgeable_missing_base_count"]) > 0:
            failure_reason = "unsupported_backbone_missing_same_component_anchors"
        fail_summary.update({
            "status": "failed",
            "failure_reason": failure_reason,
        })
        write_json(backbone_summary_path, fail_summary)
        raise RuntimeError("river_v1_backbone_unsupported_interior_inert")
    return summary
