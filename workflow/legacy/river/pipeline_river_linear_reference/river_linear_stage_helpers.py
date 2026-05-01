from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.warp import transform as rio_transform
from shapely.ops import transform as shp_transform

from core.nodata_utils import sanitize_array

_CENTERLINE_REQUIRED_FIELDS = ("point_id", "station_m", "geometry")
_CENTERLINE_OPTIONAL_ALIASES = {
    "reach_id": ("reach_id", "comid", "nhdplusid"),
    "levelpath_id": ("levelpath_id", "levelpathi", "level_path_id"),
    "centerline_order": ("centerline_order", "order", "rank"),
    "stream_order": ("stream_order", "streamorde", "streamorder", "streamord", "strahler", "Strahler"),
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


def attach_component_width_proxy(points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    out = points.copy()
    if out.empty or polygons is None or getattr(polygons, "empty", True):
        if "mean_width_m" not in out.columns:
            out["mean_width_m"] = np.nan
        return out
    poly = polygons.copy()
    poly = poly[poly.geometry.notnull() & ~poly.geometry.is_empty].copy()
    if poly.empty:
        if "mean_width_m" not in out.columns:
            out["mean_width_m"] = np.nan
        return out
    group_cols = _group_fields(out)
    poly = poly.reset_index(drop=True)
    poly["_poly_idx"] = np.arange(len(poly), dtype=int)
    if group_cols:
        point_group = out[group_cols].astype(str).agg("|".join, axis=1)
    else:
        point_group = pd.Series(["_all"] * len(out), index=out.index, dtype=object)
    out["_component_group_key"] = point_group.to_numpy(dtype=object)
    poly_geoms = poly.set_index("_poly_idx").geometry.to_dict()
    poly_areas = {int(idx): float(getattr(geom, "area", 0.0) or 0.0) for idx, geom in poly_geoms.items()}
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


def canonicalize_centerline_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None:
        return gpd.GeoDataFrame(columns=list(_CENTERLINE_REQUIRED_FIELDS), geometry="geometry")
    out = gdf.copy()
    if getattr(out, "geometry", None) is None:
        out = gpd.GeoDataFrame(out, geometry="geometry", crs=getattr(gdf, "crs", None))
    for target, aliases in _CENTERLINE_OPTIONAL_ALIASES.items():
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
    return gpd.GeoDataFrame(out, geometry="geometry", crs=out.crs)


def validate_centerline_points(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    missing = [f for f in _CENTERLINE_REQUIRED_FIELDS if f not in gdf.columns]
    record_count = int(len(gdf))
    geometry_valid = bool(getattr(gdf, "geometry", None) is not None and gdf.geometry.notna().all() and gdf.is_valid.all()) if record_count > 0 else True
    station = pd.to_numeric(gdf["station_m"], errors="coerce") if "station_m" in gdf.columns else pd.Series(dtype=float)
    point_id_unique = bool(gdf["point_id"].is_unique) if "point_id" in gdf.columns else False
    station_monotonic_by_reach = True
    if record_count > 1 and "station_m" in gdf.columns:
        group_cols = [c for c in ("levelpath_id", "reach_id", "source_reach_key") if c in gdf.columns]
        if group_cols:
            for _, grp in gdf.groupby(group_cols, dropna=False, sort=False):
                sta = pd.to_numeric(grp["station_m"], errors="coerce")
                if sta.isna().any() or not sta.is_monotonic_increasing:
                    station_monotonic_by_reach = False
                    break
        else:
            station_monotonic_by_reach = bool((not station.isna().any()) and station.is_monotonic_increasing)
    width_vals = pd.to_numeric(gdf["mean_width_m"], errors="coerce").to_numpy(dtype=float) if "mean_width_m" in gdf.columns else np.full((record_count,), np.nan)
    return {
        "valid": not missing and geometry_valid and point_id_unique and station_monotonic_by_reach,
        "record_count": record_count,
        "required_fields_present": not missing,
        "missing_required_fields": missing,
        "geometry_valid": geometry_valid,
        "point_id_unique": point_id_unique,
        "station_monotonic_by_reach": station_monotonic_by_reach,
        "finite_station_count": int(np.count_nonzero(np.isfinite(station.to_numpy(dtype=float)))) if "station_m" in gdf.columns else 0,
        "finite_width_count": int(np.count_nonzero(np.isfinite(width_vals))),
        "duplicate_point_ids": int(0 if "point_id" not in gdf.columns else gdf["point_id"].duplicated().sum()),
    }


def sample_raster_to_points(gdf: gpd.GeoDataFrame, raster_path: Path, *, search_radius_cells: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(raster_path) as ds:
        if gdf.crs is None:
            raise RuntimeError('river_linear_points_missing_crs')
        target_crs = ds.crs
        xs = gdf.geometry.x.to_numpy(dtype=float)
        ys = gdf.geometry.y.to_numpy(dtype=float)
        if target_crs is None:
            raise RuntimeError('river_linear_raster_missing_crs')
        if str(gdf.crs) != str(target_crs):
            tx, ty = rio_transform(gdf.crs, target_crs, xs.tolist(), ys.tolist())
            xs = np.asarray(tx, dtype=float)
            ys = np.asarray(ty, dtype=float)
            reprojected = True
        else:
            reprojected = False
        arr = sanitize_array(ds.read(1), ds.nodata, dtype='float32')
        finite_xy = np.isfinite(xs) & np.isfinite(ys)
        rows = np.full((len(gdf),), np.nan, dtype=float)
        cols = np.full((len(gdf),), np.nan, dtype=float)
        if np.any(finite_xy):
            rr, cc = rasterio.transform.rowcol(ds.transform, xs[finite_xy], ys[finite_xy], op=np.floor)
            rows[finite_xy] = np.asarray(rr, dtype=float)
            cols[finite_xy] = np.asarray(cc, dtype=float)
        in_extent = finite_xy & np.isfinite(rows) & np.isfinite(cols) & (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
        vals = np.full((len(gdf),), np.nan, dtype=float)
        exact_finite_count = 0
        window_fallback_count = 0
        if np.any(in_extent):
            row_i = rows[in_extent].astype(int)
            col_i = cols[in_extent].astype(int)
            direct = arr[row_i, col_i].astype(float)
            vals[in_extent] = direct
            exact_good = np.isfinite(direct)
            exact_finite_count = int(np.count_nonzero(exact_good))
            fallback_locs = np.where(in_extent)[0][~exact_good]
            if fallback_locs.size > 0 and int(search_radius_cells) > 0:
                radius = int(search_radius_cells)
                for idx in fallback_locs.tolist():
                    r0 = int(rows[idx]); c0 = int(cols[idx])
                    rmin = max(0, r0 - radius); rmax = min(ds.height, r0 + radius + 1)
                    cmin = max(0, c0 - radius); cmax = min(ds.width, c0 + radius + 1)
                    window = arr[rmin:rmax, cmin:cmax]
                    good = np.isfinite(window)
                    if not np.any(good):
                        continue
                    rrw, ccw = np.where(good)
                    rr_abs = rrw + rmin
                    cc_abs = ccw + cmin
                    dist2 = (rr_abs - r0) ** 2 + (cc_abs - c0) ** 2
                    take = int(np.argmin(dist2))
                    vals[idx] = float(arr[rr_abs[take], cc_abs[take]])
                    window_fallback_count += 1
        vals[~np.isfinite(vals)] = np.nan
        diagnostics = {
            'sample_raster_path': str(raster_path),
            'sample_raster_finite_pixel_count': int(np.count_nonzero(np.isfinite(arr))),
            'sample_raster_crs': str(target_crs) if target_crs else None,
            'centerline_crs': str(gdf.crs) if gdf.crs else None,
            'centerline_reprojected_to_raster_crs': bool(reprojected),
            'stations_attempted': int(len(gdf)),
            'stations_in_raster_extent': int(np.count_nonzero(in_extent)),
            'stations_outside_raster_extent': int(len(gdf) - np.count_nonzero(in_extent)),
            'sample_rows_min': float(np.nanmin(rows)) if rows.size else None,
            'sample_rows_max': float(np.nanmax(rows)) if rows.size else None,
            'sample_cols_min': float(np.nanmin(cols)) if cols.size else None,
            'sample_cols_max': float(np.nanmax(cols)) if cols.size else None,
            'exact_finite_count': int(exact_finite_count),
            'window_fallback_count': int(window_fallback_count),
            'search_radius_cells': int(search_radius_cells),
        }
        return vals, diagnostics


def validate_centerline_authoritative_bed(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    required_fields = ("point_id", "station_m", "authoritative_bed_z_m", "geometry")
    missing = [f for f in required_fields if f not in gdf.columns]
    recs = int(len(gdf))
    bed_vals = pd.to_numeric(gdf["authoritative_bed_z_m"], errors="coerce").to_numpy(dtype=float) if "authoritative_bed_z_m" in gdf.columns else np.full((recs,), np.nan)
    finite = int(np.count_nonzero(np.isfinite(bed_vals)))
    geometry_valid = bool(getattr(gdf, "geometry", None) is not None and gdf.geometry.notna().all() and gdf.is_valid.all()) if recs > 0 else True
    station_monotonic_by_reach = True
    if recs > 1 and "station_m" in gdf.columns:
        group_cols = [c for c in ("levelpath_id", "reach_id", "source_reach_key") if c in gdf.columns]
        if group_cols:
            for _, grp in gdf.groupby(group_cols, dropna=False, sort=False):
                sta = pd.to_numeric(grp["station_m"], errors="coerce")
                if sta.isna().any() or not sta.is_monotonic_increasing:
                    station_monotonic_by_reach = False
                    break
        else:
            sta = pd.to_numeric(gdf["station_m"], errors="coerce")
            station_monotonic_by_reach = bool((not sta.isna().any()) and sta.is_monotonic_increasing)
    point_id_unique = bool(gdf["point_id"].is_unique) if "point_id" in gdf.columns else False
    return {
        "valid": not missing and geometry_valid and point_id_unique and station_monotonic_by_reach,
        "record_count": recs,
        "required_fields_present": not missing,
        "missing_required_fields": missing,
        "finite_authoritative_bed_count": finite,
        "geometry_valid": geometry_valid,
        "station_monotonic_by_reach": station_monotonic_by_reach,
        "point_id_unique": point_id_unique,
        "duplicate_point_ids": int(0 if "point_id" not in gdf.columns else gdf["point_id"].duplicated().sum()),
    }


def validate_observed_offsets(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    required_fields = ("point_id", "station_m", "observed_offset_m", "geometry")
    missing = [f for f in required_fields if f not in gdf.columns]
    recs = int(len(gdf))
    vals = pd.to_numeric(gdf["observed_offset_m"], errors="coerce").to_numpy(dtype=float) if "observed_offset_m" in gdf.columns else np.full((recs,), np.nan)
    finite = int(np.count_nonzero(np.isfinite(vals)))
    negative = int(np.count_nonzero(np.isfinite(vals) & (vals < 0.0)))
    geometry_valid = bool(getattr(gdf, "geometry", None) is not None and gdf.geometry.notna().all() and gdf.is_valid.all()) if recs > 0 else True
    monotonic_by_reach = True
    if recs > 1 and "station_m" in gdf.columns:
        group_cols = [c for c in ("levelpath_id", "reach_id", "source_reach_key") if c in gdf.columns]
        if group_cols:
            for _, grp in gdf.groupby(group_cols, dropna=False, sort=False):
                sta = pd.to_numeric(grp["station_m"], errors="coerce")
                if sta.isna().any() or not sta.is_monotonic_increasing:
                    monotonic_by_reach = False
                    break
        else:
            sta = pd.to_numeric(gdf["station_m"], errors="coerce")
            monotonic_by_reach = bool((not sta.isna().any()) and sta.is_monotonic_increasing)
    point_id_unique = bool(gdf["point_id"].is_unique) if "point_id" in gdf.columns else False
    wse_vals = pd.to_numeric(gdf["wse_proxy_z_m"], errors="coerce").to_numpy(dtype=float) if "wse_proxy_z_m" in gdf.columns else np.full((recs,), np.nan)
    bed_vals = pd.to_numeric(gdf["authoritative_bed_z_m"], errors="coerce").to_numpy(dtype=float) if "authoritative_bed_z_m" in gdf.columns else np.full((recs,), np.nan)
    finite_wse = wse_vals[np.isfinite(wse_vals)]
    finite_bed = bed_vals[np.isfinite(bed_vals)]
    finite_off = vals[np.isfinite(vals)]
    wse_unique = int(np.unique(np.round(finite_wse, 6)).size) if finite_wse.size else 0
    bed_unique = int(np.unique(np.round(finite_bed, 6)).size) if finite_bed.size else 0
    offset_unique = int(np.unique(np.round(finite_off, 6)).size) if finite_off.size else 0
    dominant_wse_fraction = 0.0
    dominant_offset_fraction = 0.0
    if finite_wse.size:
        _, counts = np.unique(np.round(finite_wse, 6), return_counts=True)
        dominant_wse_fraction = float(np.max(counts) / max(finite_wse.size, 1))
    if finite_off.size:
        _, counts = np.unique(np.round(finite_off, 6), return_counts=True)
        dominant_offset_fraction = float(np.max(counts) / max(finite_off.size, 1))
    return {
        "valid": not missing and geometry_valid and monotonic_by_reach and point_id_unique,
        "record_count": recs,
        "required_fields_present": not missing,
        "missing_required_fields": missing,
        "finite_observed_offset_count": finite,
        "negative_observed_offset_count": negative,
        "geometry_valid": geometry_valid,
        "station_monotonic_by_reach": monotonic_by_reach,
        "point_id_unique": point_id_unique,
        "duplicate_point_ids": int(0 if "point_id" not in gdf.columns else gdf["point_id"].duplicated().sum()),
        "wse_unique_value_count": wse_unique,
        "authoritative_bed_unique_value_count": bed_unique,
        "observed_offset_unique_value_count": offset_unique,
        "dominant_wse_value_fraction": dominant_wse_fraction,
        "dominant_observed_offset_value_fraction": dominant_offset_fraction,
        "wse_dynamic_range_m": float(np.nanmax(finite_wse) - np.nanmin(finite_wse)) if finite_wse.size else float("nan"),
        "observed_offset_dynamic_range_m": float(np.nanmax(finite_off) - np.nanmin(finite_off)) if finite_off.size else float("nan"),
        "wse_constant_like": bool(finite_wse.size and (wse_unique <= max(3, int(0.01 * finite_wse.size)) or dominant_wse_fraction >= 0.5)),
        "observed_offset_constant_like": bool(finite_off.size and (offset_unique <= max(3, int(0.01 * finite_off.size)) or dominant_offset_fraction >= 0.5)),
    }


def read_network_lines(network_gpkg: Path) -> tuple[gpd.GeoDataFrame, str]:
    preferred_layers = (
        "mainstem_solve_network",
        "major_system_network",
        "major_system_network_clip",
        "rivers_clip",
    )
    layers = [layer.name for layer in gpd.list_layers(network_gpkg).itertuples(index=False)]
    if not layers:
        raise ValueError("network_has_no_layers")

    def _read_valid_layer(layer_name: str) -> Optional[gpd.GeoDataFrame]:
        gdf = gpd.read_file(network_gpkg, layer=layer_name)
        if gdf.empty:
            return None
        if "geometry" not in gdf.columns:
            return None
        if not {"from_node", "to_node"}.issubset(gdf.columns):
            return None
        return gdf

    for chosen in preferred_layers:
        if chosen not in layers:
            continue
        gdf = _read_valid_layer(chosen)
        if gdf is not None:
            return gdf, chosen
    for chosen in layers:
        gdf = _read_valid_layer(chosen)
        if gdf is not None:
            return gdf, chosen
    raise ValueError("network_missing_required_columns")
