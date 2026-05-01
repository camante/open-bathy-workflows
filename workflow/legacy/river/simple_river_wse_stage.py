from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from legacy.river.simple_river_stage_contract import STAGE_CENTERLINE_WSE_PROXY
from legacy.river.simple_river_stage_receipts import build_stage_receipt, write_stage_receipt

_REQUIRED_FIELDS = ("point_id", "station_m", "wse_proxy_z_m", "geometry")
_WSE_FIELD_CANDIDATES = (
    "wse_proxy_z_m",
    "bank_wse_proxy_monotone_m",
    "bank_wse_proxy_interp_m",
    "wse_elevation_m",
    "bank_low_stage_z_m",
    "bank_elevation_m",
)


def centerline_wse_field_schema() -> dict[str, str]:
    return {
        "point_id": "string",
        "station_m": "float64",
        "wse_proxy_z_m": "float64",
        "wse_proxy_source": "string",
        "wse_proxy_method": "string",
        "geometry": "Point",
    }


def _load_centerline_points(*, river_context: dict[str, Any]) -> gpd.GeoDataFrame:
    path = river_context.get("centerline_points_path") or river_context.get("existing_centerline_path")
    if path and Path(path).exists():
        return gpd.read_file(path)
    gdf = river_context.get("centerline_points_gdf")
    if gdf is not None:
        return gdf.copy()
    raise RuntimeError("simple_river_wse_stage_missing_centerline_points")


def _sample_raster_to_points(gdf: gpd.GeoDataFrame, raster_path: str | Path) -> np.ndarray:
    path = Path(raster_path)
    with rasterio.open(path) as ds:
        pts = list(zip(gdf.geometry.x.to_numpy(dtype=float), gdf.geometry.y.to_numpy(dtype=float)))
        vals = np.asarray([v[0] for v in ds.sample(pts)], dtype=float)
        nod = ds.nodata
        if nod is not None:
            vals[np.isclose(vals, float(nod))] = np.nan
        vals[~np.isfinite(vals)] = np.nan
        return vals


def _wse_from_columns(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray | None, str | None]:
    for field in _WSE_FIELD_CANDIDATES:
        if field in gdf.columns:
            vals = pd.to_numeric(gdf[field], errors="coerce").to_numpy(dtype=float)
            if np.any(np.isfinite(vals)):
                return vals, field
    return None, None


def _wse_from_profile_csv(gdf: gpd.GeoDataFrame, csv_path: str | Path) -> tuple[np.ndarray | None, str | None]:
    path = Path(csv_path)
    if not csv_path or not path.exists() or path.is_dir():
        return None, None
    df = pd.read_csv(path)
    if df.empty or "bank_wse_proxy_monotone_m" not in df.columns:
        return None, None
    vals = pd.to_numeric(df["bank_wse_proxy_monotone_m"], errors="coerce").to_numpy(dtype=float)
    if not np.any(np.isfinite(vals)):
        return None, None
    out = np.full((len(gdf),), np.nan, dtype=float)
    if "point_id" in df.columns and "point_id" in gdf.columns:
        lookup = pd.Series(vals, index=df["point_id"].astype(str)).groupby(level=0).last()
        matched = gdf["point_id"].astype(str).map(lookup)
        out = pd.to_numeric(matched, errors="coerce").to_numpy(dtype=float)
    elif "station_m" in df.columns and "station_m" in gdf.columns:
        sta = pd.to_numeric(df["station_m"], errors="coerce").to_numpy(dtype=float)
        use = np.isfinite(sta) & np.isfinite(vals)
        target = pd.to_numeric(gdf["station_m"], errors="coerce").to_numpy(dtype=float)
        target_valid = np.isfinite(target)
        if np.count_nonzero(use) >= 1 and np.any(target_valid):
            order = np.argsort(sta[use], kind="mergesort")
            sta_use = sta[use][order]
            vals_use = vals[use][order]
            if sta_use.size == 1:
                out[target_valid] = vals_use[0]
            else:
                sta_unique, unique_idx = np.unique(sta_use, return_index=True)
                vals_unique = vals_use[unique_idx]
                out[target_valid] = np.interp(target[target_valid], sta_unique, vals_unique).astype(float)
    if np.any(np.isfinite(out)):
        return out, "bank_wse_proxy_profile_summary"
    return None, None


def estimate_centerline_wse_proxy(centerline_points_gdf: gpd.GeoDataFrame, river_context: dict[str, Any]) -> tuple[gpd.GeoDataFrame, list[str]]:
    gdf = centerline_points_gdf.copy()
    warnings: list[str] = []
    vals, source = _wse_from_columns(gdf)
    method = "direct_centerline_field" if source else None
    if vals is None:
        vals, source = _wse_from_profile_csv(gdf, river_context.get("bank_wse_profile_summary_path", ""))
        method = "profile_csv_join" if source else None
    if vals is None:
        raster_path = river_context.get("bank_wse_edge_guidance_path") or river_context.get("bank_elevation_path")
        if raster_path and Path(raster_path).exists():
            vals = _sample_raster_to_points(gdf, raster_path)
            source = str(raster_path)
            method = "sample_bank_wse_edge_guidance"
    if vals is None:
        raise RuntimeError("simple_river_wse_stage_missing_wse_source")
    gdf["wse_proxy_z_m"] = pd.to_numeric(vals, errors="coerce")
    gdf["wse_proxy_source"] = str(source)
    gdf["wse_proxy_method"] = str(method)
    if np.count_nonzero(np.isfinite(gdf["wse_proxy_z_m"].to_numpy(dtype=float))) == 0:
        raise RuntimeError("simple_river_wse_stage_no_finite_wse_values")
    if np.count_nonzero(~np.isfinite(gdf["wse_proxy_z_m"].to_numpy(dtype=float))) > 0:
        warnings.append("wse_proxy_contains_missing_values")
    keep = [c for c in ("point_id", "station_m", "wse_proxy_z_m", "wse_proxy_source", "wse_proxy_method", "geometry") if c in gdf.columns]
    out = gpd.GeoDataFrame(gdf[keep].copy(), geometry="geometry", crs=gdf.crs)
    sort_cols = [c for c in ("station_m", "point_id") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out, warnings


def validate_centerline_wse_proxy(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    missing = [f for f in _REQUIRED_FIELDS if f not in gdf.columns]
    vals = pd.to_numeric(gdf["wse_proxy_z_m"], errors="coerce").to_numpy(dtype=float) if "wse_proxy_z_m" in gdf.columns else np.full((len(gdf),), np.nan)
    recs = int(len(gdf))
    finite = int(np.count_nonzero(np.isfinite(vals)))
    negative = int(np.count_nonzero(np.isfinite(vals) & (vals < 0.0)))
    geometry_valid = bool(getattr(gdf, 'geometry', None) is not None and gdf.geometry.notna().all() and gdf.is_valid.all()) if recs > 0 else True
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
    return {
        "valid": not missing and finite > 0 and geometry_valid and monotonic_by_reach and point_id_unique,
        "record_count": recs,
        "required_fields_present": not missing,
        "missing_required_fields": missing,
        "finite_wse_count": finite,
        "missing_wse_count": int(recs - finite),
        "negative_wse_count": negative,
        "geometry_valid": geometry_valid,
        "station_monotonic_by_reach": monotonic_by_reach,
        "point_id_unique": point_id_unique,
        "duplicate_point_ids": int(0 if "point_id" not in gdf.columns else gdf["point_id"].duplicated().sum()),
    }


def write_centerline_wse_proxy_gpkg(gdf: gpd.GeoDataFrame, out_path: str) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG")
    return str(path)


def build_centerline_wse_proxy_points(*, river_context: dict[str, Any], out_path: str, receipt_path: str | None = None) -> dict[str, Any]:
    centerline = _load_centerline_points(river_context=river_context)
    gdf, warnings = estimate_centerline_wse_proxy(centerline, river_context)
    validation = validate_centerline_wse_proxy(gdf)
    if not validation.get("valid"):
        raise RuntimeError(f"simple_river_wse_stage_invalid:{validation}")
    written_path = write_centerline_wse_proxy_gpkg(gdf, out_path)
    receipt = build_stage_receipt(
        stage_id=STAGE_CENTERLINE_WSE_PROXY,
        output_artifact=written_path,
        input_artifacts=[str(v) for v in river_context.get("input_artifacts", [])],
        record_count=int(len(gdf)),
        field_schema=centerline_wse_field_schema(),
        vertical_reference=str(river_context.get("vertical_reference") or "unknown"),
        warnings=warnings,
        source_logic=str(gdf["wse_proxy_method"].iloc[0]) if len(gdf) else "unknown",
        validation=validation,
    )
    written_receipt = write_stage_receipt(receipt, receipt_path) if receipt_path else None
    return {
        "stage_id": STAGE_CENTERLINE_WSE_PROXY,
        "output_artifact": written_path,
        "receipt_path": written_receipt,
        "record_count": int(len(gdf)),
        "validation": validation,
        "warnings": warnings,
    }
