from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from river_structured_scaffold import build_centerline_points, select_retained_river_features
from river_v2_context import RiverV2Context
from river_v2_contract import RiverV2StageResult, STAGE_RIVER_CENTERLINE
from river_v2_receipts import build_river_v2_stage_receipt, write_river_v2_receipt

_REQUIRED_FIELDS = ("point_id", "station_m", "geometry")
_OPTIONAL_ALIASES = {
    "reach_id": ("reach_id", "comid", "nhdplusid"),
    "levelpath_id": ("levelpath_id", "levelpathi", "level_path_id"),
    "centerline_order": ("centerline_order", "order", "rank"),
    "stream_order": ("stream_order", "streamorde", "streamorder", "streamord", "strahler", "Strahler"),
}


def centerline_field_schema() -> dict[str, str]:
    return {
        "point_id": "string",
        "station_m": "float64",
        "geometry": "Point",
        "reach_id": "string",
        "levelpath_id": "string",
        "stream_order": "float64",
        "mean_width_m": "float64",
        "centerline_order": "int64",
        "source_reach_key": "string",
        "is_endpoint": "bool",
    }


def _normalize_point_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if "point_id" not in out.columns:
        out["point_id"] = [f"cl_{i:07d}" for i in range(len(out))]
    else:
        values = out["point_id"].astype(str).str.strip()
        mask = values.eq("") | values.eq("None") | values.eq("nan")
        if mask.any():
            repl = values.copy()
            for pos, idx in enumerate(list(out.index[mask])):
                repl.loc[idx] = f"cl_{pos:07d}"
            values = repl
        out["point_id"] = values
    return out


def _group_fields(frame: gpd.GeoDataFrame) -> list[str]:
    return [c for c in ("component_id", "levelpath_id", "reach_id", "source_reach_key") if c in frame.columns]


def _attach_component_width_proxy(points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    out = points.copy()
    if out.empty or polygons is None or getattr(polygons, "empty", True):
        if "mean_width_m" not in out.columns:
            out["mean_width_m"] = np.nan
        return out
    group_cols = _group_fields(out)
    poly = polygons.copy()
    poly = poly[poly.geometry.notnull() & ~poly.geometry.is_empty].copy()
    if poly.empty:
        if "mean_width_m" not in out.columns:
            out["mean_width_m"] = np.nan
        return out
    poly = poly.reset_index(drop=True)
    poly["_poly_idx"] = np.arange(len(poly), dtype=int)
    if group_cols:
        point_group = out[group_cols].astype(str).agg("|".join, axis=1)
    else:
        point_group = pd.Series(["_all"] * len(out), index=out.index, dtype=object)
    out["_component_group_key"] = point_group.to_numpy(dtype=object)

    poly_geoms = poly.set_index("_poly_idx").geometry.to_dict()
    poly_areas = {int(poly_idx): float(getattr(poly_geom, "area", 0.0) or 0.0) for poly_idx, poly_geom in poly_geoms.items()}
    width_by_group: dict[str, float] = {}
    for group_key, group_points in out.groupby("_component_group_key", dropna=False, sort=False):
        station = pd.to_numeric(group_points.get("station_m"), errors="coerce")
        finite_station = station[np.isfinite(station)]
        component_length = float(finite_station.max() - finite_station.min()) if len(finite_station) >= 2 else float("nan")
        unique_poly_ids: set[int] = set()
        for geom in group_points.geometry:
            if geom is None or geom.is_empty:
                continue
            for poly_idx, poly_geom in poly_geoms.items():
                try:
                    if poly_geom is not None and not poly_geom.is_empty and geom.intersects(poly_geom):
                        unique_poly_ids.add(int(poly_idx))
                except Exception:
                    continue
        component_area = float(sum(float(poly_areas.get(pid, 0.0) or 0.0) for pid in sorted(unique_poly_ids))) if unique_poly_ids else float("nan")
        width = float("nan")
        if np.isfinite(component_length) and component_length > 0.0 and np.isfinite(component_area) and component_area > 0.0:
            width = float(component_area / component_length)
        width_by_group[str(group_key)] = width

    if "mean_width_m" not in out.columns:
        out["mean_width_m"] = np.nan
    existing = pd.to_numeric(out["mean_width_m"], errors="coerce").to_numpy(dtype=float)
    derived = np.asarray([width_by_group.get(str(k), np.nan) for k in out["_component_group_key"].to_numpy(dtype=object)], dtype=float)
    take = (~np.isfinite(existing)) & np.isfinite(derived)
    existing[take] = derived[take]
    out["mean_width_m"] = existing
    return out.drop(columns=["_component_group_key"], errors="ignore")


def _attach_dominant_stream_order(points: gpd.GeoDataFrame, flows: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    out = points.copy()
    if "stream_order" not in out.columns:
        out["stream_order"] = np.nan
    existing = pd.to_numeric(out["stream_order"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(existing).all():
        return out
    if flows is None or getattr(flows, "empty", True):
        return out
    order_col = None
    for cand in ("stream_order", "streamorde", "streamorder", "streamord", "strahler", "Strahler"):
        if cand in flows.columns:
            order_col = cand
            break
    if order_col is None:
        return out
    flow_order = pd.to_numeric(flows[order_col], errors="coerce")
    if not flow_order.notna().any():
        return out
    dominant_order = float(flow_order.median())
    existing[~np.isfinite(existing)] = dominant_order
    out["stream_order"] = existing
    return out


def _canonicalize_centerline_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None:
        return gpd.GeoDataFrame(columns=list(centerline_field_schema().keys()), geometry="geometry")
    out = gdf.copy()
    if getattr(out, "geometry", None) is None:
        out = gpd.GeoDataFrame(out, geometry="geometry", crs=getattr(gdf, "crs", None))
    for target, aliases in _OPTIONAL_ALIASES.items():
        if target not in out.columns:
            for alias in aliases:
                if alias in out.columns:
                    out[target] = out[alias]
                    break
    if "station_m" not in out.columns:
        for cand in ("centerline_m", "chainage_m", "distance_m"):
            if cand in out.columns:
                out["station_m"] = pd.to_numeric(out[cand], errors="coerce")
                break
    out["station_m"] = pd.to_numeric(out.get("station_m"), errors="coerce")
    out = _normalize_point_ids(out)
    if "source_reach_key" not in out.columns:
        if "levelpath_id" in out.columns:
            out["source_reach_key"] = out["levelpath_id"].astype(str)
        elif "reach_id" in out.columns:
            out["source_reach_key"] = out["reach_id"].astype(str)
        else:
            out["source_reach_key"] = "unknown"
    if "centerline_order" not in out.columns:
        out["centerline_order"] = range(len(out))
    if "stream_order" in out.columns:
        out["stream_order"] = pd.to_numeric(out["stream_order"], errors="coerce")
    if "mean_width_m" in out.columns:
        out["mean_width_m"] = pd.to_numeric(out["mean_width_m"], errors="coerce")
    order = pd.to_numeric(out["centerline_order"], errors="coerce")
    if order.isna().any():
        fallback = pd.Series(range(len(out)), index=out.index, dtype="float64")
        order = order.where(~order.isna(), fallback)
    out["centerline_order"] = order.astype(int)
    if "is_endpoint" not in out.columns:
        out["is_endpoint"] = False
    sort_cols = [c for c in ("levelpath_id", "reach_id", "station_m", "centerline_order") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    keep_cols = [c for c in centerline_field_schema().keys() if c in out.columns]
    extra_cols = [c for c in out.columns if c not in keep_cols]
    return gpd.GeoDataFrame(out[keep_cols + extra_cols], geometry="geometry", crs=out.crs)


def _sample_authoritative_support_counts(gdf: gpd.GeoDataFrame, sampling_raster_path: Path | None) -> pd.Series:
    if gdf is None or gdf.empty or sampling_raster_path is None or not Path(sampling_raster_path).exists():
        return pd.Series(0, index=gdf.index if gdf is not None else None, dtype="int64")
    out = pd.Series(0, index=gdf.index, dtype="int64")
    try:
        with rasterio.open(sampling_raster_path) as ds:
            coords = [(geom.x, geom.y) if geom is not None and not geom.is_empty else (np.nan, np.nan) for geom in gdf.geometry]
            sampled = []
            for value in ds.sample(coords):
                try:
                    sampled.append(float(value[0]))
                except Exception:
                    sampled.append(np.nan)
            arr = np.asarray(sampled, dtype=float)
            nodata = ds.nodata
            if nodata is not None:
                arr[np.isclose(arr, nodata, equal_nan=False)] = np.nan
            out.loc[:] = np.isfinite(arr).astype(np.int64)
    except Exception:
        return out
    return out


def _filter_centerline_components_simple(gdf: gpd.GeoDataFrame, sampling_raster_path: Path | None) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    out = gdf.copy()
    if out.empty:
        return out, {"filter_applied": False, "reason": "empty_centerline"}
    if "levelpath_id" not in out.columns:
        out["keep_component"] = True
        out["drop_reason"] = ""
        return out, {"filter_applied": False, "reason": "missing_levelpath_id"}

    out["levelpath_id"] = out["levelpath_id"].astype(str)
    support_hits = _sample_authoritative_support_counts(out, Path(sampling_raster_path) if sampling_raster_path is not None else None)
    out["authoritative_support_hit"] = support_hits.astype(int)

    stats = []
    for levelpath_id, grp in out.groupby("levelpath_id", sort=False, dropna=False):
        stations = pd.to_numeric(grp.get("station_m"), errors="coerce")
        finite = stations[np.isfinite(stations)]
        if len(finite) >= 2:
            length_m = float(finite.max() - finite.min())
        else:
            length_m = float(len(grp))
        stats.append({
            "levelpath_id": str(levelpath_id),
            "point_count": int(len(grp)),
            "length_m": length_m,
            "authoritative_support_count": int(pd.to_numeric(grp.get("authoritative_support_hit", 0), errors="coerce").fillna(0).astype(int).sum()),
        })
    stats_df = pd.DataFrame(stats)
    if stats_df.empty:
        out["keep_component"] = True
        out["drop_reason"] = ""
        return out, {"filter_applied": False, "reason": "no_component_stats"}

    main_idx = stats_df["length_m"].astype(float).idxmax()
    main_levelpath_id = str(stats_df.loc[main_idx, "levelpath_id"])
    keep_map = {}
    drop_reason_map = {}
    for row in stats_df.itertuples(index=False):
        lp = str(row.levelpath_id)
        keep = False
        reason = ""
        if lp == main_levelpath_id:
            keep = True
            reason = "main_component"
        elif int(row.authoritative_support_count) > 0:
            keep = True
            reason = "secondary_with_authoritative_support"
        else:
            keep = False
            reason = "secondary_without_authoritative_support"
        keep_map[lp] = keep
        drop_reason_map[lp] = reason

    out["keep_component"] = out["levelpath_id"].map(keep_map).fillna(False).astype(bool)
    out["drop_reason"] = out["levelpath_id"].map(drop_reason_map).fillna("secondary_without_authoritative_support")
    filtered = out.loc[out["keep_component"]].copy()
    if filtered.empty:
        filtered = out.copy()
        filtered["keep_component"] = True
        filtered["drop_reason"] = "filter_fallback_keep_all"
        summary = {"filter_applied": False, "reason": "filtered_empty_keep_all"}
    else:
        summary = {
            "filter_applied": True,
            "group_field": "levelpath_id",
            "main_levelpath_id": main_levelpath_id,
            "all_component_count": int(len(stats_df)),
            "kept_component_count": int(stats_df[stats_df["levelpath_id"].astype(str).map(keep_map).fillna(False)].shape[0]),
            "dropped_component_count": int(stats_df[~stats_df["levelpath_id"].astype(str).map(keep_map).fillna(False)].shape[0]),
            "kept_record_count": int(len(filtered)),
            "dropped_record_count": int((~out["keep_component"]).sum()),
            "component_decisions": stats_df.assign(keep_component=stats_df["levelpath_id"].astype(str).map(keep_map), drop_reason=stats_df["levelpath_id"].astype(str).map(drop_reason_map)).to_dict(orient="records"),
            "connectivity_note": "main_levelpath_retained_in_full; secondary levelpaths kept only when authoritative support exists",
        }
    keep_cols = [c for c in filtered.columns if c not in {"authoritative_support_hit"}]
    return filtered[keep_cols], summary


def _derive_centerline_points(*, inline_centerline_points_gdf: gpd.GeoDataFrame | None, provided_source: Path | None, network_gpkg: Path | None, river_dem_path: Path | None, sampling_raster_path: Path | None, spacing_m: float, min_stream_order: int) -> gpd.GeoDataFrame:
    if inline_centerline_points_gdf is not None:
        gdf = _canonicalize_centerline_points(inline_centerline_points_gdf)
        if network_gpkg is not None and river_dem_path is not None:
            with rasterio.open(river_dem_path) as dem_ds:
                selection = select_retained_river_features(network_gpkg, target_crs=dem_ds.crs, min_stream_order=max(int(min_stream_order or 5), 1), keep_top_components=6)
            gdf = _attach_dominant_stream_order(gdf, selection.flows)
            gdf = _attach_component_width_proxy(gdf, selection.polygons)
        return gdf
    if provided_source is not None and provided_source.exists():
        gdf = _canonicalize_centerline_points(gpd.read_file(provided_source))
        if network_gpkg is not None and river_dem_path is not None:
            with rasterio.open(river_dem_path) as dem_ds:
                selection = select_retained_river_features(network_gpkg, target_crs=dem_ds.crs, min_stream_order=max(int(min_stream_order or 5), 1), keep_top_components=6)
            gdf = _attach_dominant_stream_order(gdf, selection.flows)
            gdf = _attach_component_width_proxy(gdf, selection.polygons)
        return gdf
    if network_gpkg is None or river_dem_path is None:
        raise RuntimeError("river_v2_centerline_missing_explicit_network_inputs")
    return _build_centerline_from_network(
        network_gpkg=network_gpkg,
        river_dem_path=river_dem_path,
        sampling_raster_path=sampling_raster_path,
        spacing_m=spacing_m,
        min_stream_order=min_stream_order,
    )


def _build_centerline_from_network(*, network_gpkg: Path, river_dem_path: Path, sampling_raster_path: Path | None, spacing_m: float, min_stream_order: int) -> gpd.GeoDataFrame:
    with rasterio.open(river_dem_path) as dem_ds:
        target_crs = dem_ds.crs
    selection = select_retained_river_features(
        network_gpkg,
        target_crs=target_crs,
        min_stream_order=max(int(min_stream_order or 5), 1),
        keep_top_components=6,
    )
    sampling_raster = Path(sampling_raster_path) if sampling_raster_path is not None and Path(sampling_raster_path).exists() else None
    centerline = build_centerline_points(
        selection.flows,
        selection.polygons,
        spacing_m=float(spacing_m),
        raster_path=sampling_raster,
        logger=logging.getLogger(__name__),
    )
    if centerline is None or getattr(centerline, "empty", True):
        raise RuntimeError("river_v2_centerline_no_points")
    centerline = _canonicalize_centerline_points(centerline)
    centerline = _attach_component_width_proxy(centerline, selection.polygons)
    return centerline


def validate_centerline_points(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    missing = [f for f in _REQUIRED_FIELDS if f not in gdf.columns]
    record_count = int(len(gdf)) if gdf is not None else 0
    null_station_count = int(pd.to_numeric(gdf["station_m"], errors="coerce").isna().sum()) if "station_m" in getattr(gdf, "columns", []) else record_count
    null_geometry_count = int(gdf.geometry.isna().sum()) if gdf is not None and getattr(gdf, "geometry", None) is not None else record_count
    geometry_valid = bool(getattr(gdf, "geometry", None) is not None and gdf.geometry.notna().all() and gdf.is_valid.all()) if record_count > 0 else True
    point_id_unique = bool(gdf["point_id"].is_unique) if "point_id" in getattr(gdf, "columns", []) else False
    station_monotonic_by_reach = True
    if record_count > 0 and "station_m" in gdf.columns:
        group_cols = [c for c in ("levelpath_id", "reach_id") if c in gdf.columns]
        if group_cols:
            for _, grp in gdf.groupby(group_cols, dropna=False, sort=False):
                vals = pd.to_numeric(grp["station_m"], errors="coerce")
                if vals.isna().any() or not vals.is_monotonic_increasing:
                    station_monotonic_by_reach = False
                    break
        else:
            vals = pd.to_numeric(gdf["station_m"], errors="coerce")
            station_monotonic_by_reach = bool((not vals.isna().any()) and vals.is_monotonic_increasing)
    return {
        "valid": not missing and geometry_valid and point_id_unique and null_station_count == 0 and null_geometry_count == 0 and station_monotonic_by_reach,
        "record_count": record_count,
        "required_fields_present": not missing,
        "missing_required_fields": missing,
        "geometry_valid": geometry_valid,
        "nonempty_geometry": null_geometry_count == 0,
        "point_id_unique": point_id_unique,
        "station_monotonic_by_reach": bool(station_monotonic_by_reach),
        "duplicate_point_ids": int(0 if "point_id" not in getattr(gdf, "columns", []) else gdf["point_id"].duplicated().sum()),
        "null_station_count": null_station_count,
        "null_geometry_count": null_geometry_count,
    }


def write_centerline_points_gpkg(gdf: gpd.GeoDataFrame, out_path: str | Path, *, slim: bool = False) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    out = gdf.copy()
    if slim:
        keep = [c for c in ("point_id", "station_m", "component_id", "levelpath_id", "reach_id", "stream_order", "mean_width_m", "source_reach_key", "geometry") if c in out.columns]
        out = gpd.GeoDataFrame(out[keep], geometry="geometry", crs=out.crs)
    out.to_file(path, driver="GPKG")
    return path


def run_centerline_stage(ctx: RiverV2Context, *, network_gpkg: str | Path | None, river_dem_path: str | Path | None, sampling_raster_path: str | Path | None, spacing_m: float, min_stream_order: int) -> RiverV2StageResult:
    paths = ctx.paths
    network_path = Path(network_gpkg) if network_gpkg is not None else None
    dem_path = Path(river_dem_path) if river_dem_path is not None else None
    sampling_path = Path(sampling_raster_path) if sampling_raster_path is not None else None
    gdf_all = _derive_centerline_points(
        inline_centerline_points_gdf=getattr(ctx, "centerline_points_gdf", None),
        provided_source=None,
        network_gpkg=network_path,
        river_dem_path=dem_path,
        sampling_raster_path=sampling_path,
        spacing_m=float(spacing_m),
        min_stream_order=int(min_stream_order),
    )
    all_path = paths.root / "river_centerline_points_all.gpkg"
    write_centerline_points_gpkg(gdf_all, all_path, slim=True)
    gdf, filter_summary = _filter_centerline_components_simple(gdf_all, sampling_path)
    validation = validate_centerline_points(gdf)
    if not validation.get("valid"):
        raise RuntimeError(f"river_v2_centerline_invalid:{validation}")
    written_path = write_centerline_points_gpkg(gdf, paths.river_centerline_points, slim=True)
    receipt = build_river_v2_stage_receipt(
        stage_id=STAGE_RIVER_CENTERLINE,
        output_artifact=str(written_path),
        input_artifacts=ctx.direct_stage_input_artifacts(network_path, dem_path, sampling_path),
        record_count=int(len(gdf)),
        field_schema=centerline_field_schema(),
        vertical_reference=ctx.vertical_reference,
        warnings=[],
        source_logic="canonicalized_centerline_points_with_component_width_proxy_then_simple_levelpath_filter",
        validation={**validation, "component_filter": filter_summary},
    )
    receipt_path = write_river_v2_receipt(receipt, paths.river_centerline_points_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_RIVER_CENTERLINE,
        output_artifact=written_path,
        receipt_path=receipt_path,
        record_count=int(len(gdf)),
        validation={**validation, "component_filter": filter_summary},
        warnings=[],
        aux_outputs={"river_centerline_points_all": str(all_path)},
    )
