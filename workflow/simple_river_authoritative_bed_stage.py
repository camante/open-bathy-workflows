from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from simple_river_stage_contract import STAGE_CENTERLINE_AUTHORITATIVE_BED
from simple_river_stage_receipts import build_stage_receipt, write_stage_receipt

_REQUIRED_FIELDS = ("point_id", "station_m", "authoritative_bed_z_m", "geometry")
_AUTHORITATIVE_FIELD_CANDIDATES = (
    "authoritative_bed_z_m",
    "centerline_z_m",
    "bed_elevation_m",
    "z_m",
)


def centerline_authoritative_bed_field_schema() -> dict[str, str]:
    return {
        "point_id": "string",
        "station_m": "float64",
        "authoritative_bed_z_m": "float64",
        "authoritative_bed_source": "string",
        "authoritative_bed_method": "string",
        "geometry": "Point",
    }


def _load_centerline_points(*, river_context: dict[str, Any]) -> gpd.GeoDataFrame:
    path = river_context.get("centerline_points_path") or river_context.get("existing_centerline_path")
    if path and Path(path).exists():
        return gpd.read_file(path)
    gdf = river_context.get("centerline_points_gdf")
    if gdf is not None:
        return gdf.copy()
    raise RuntimeError("simple_river_authoritative_bed_stage_missing_centerline_points")


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


def _authoritative_bed_from_columns(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray | None, str | None]:
    for field in _AUTHORITATIVE_FIELD_CANDIDATES:
        if field in gdf.columns:
            vals = pd.to_numeric(gdf[field], errors="coerce").to_numpy(dtype=float)
            if np.any(np.isfinite(vals)):
                return vals, field
    return None, None


def sample_authoritative_bed_to_centerline(centerline_points_gdf: gpd.GeoDataFrame, river_context: dict[str, Any]) -> tuple[gpd.GeoDataFrame, list[str]]:
    gdf = centerline_points_gdf.copy()
    warnings: list[str] = []
    vals, source = _authoritative_bed_from_columns(gdf)
    method = "direct_centerline_field" if source else None
    if vals is None:
        raster_path = river_context.get("authoritative_bed_path") or river_context.get("authoritative_base_path") or river_context.get("aligned_authoritative_base_path")
        if raster_path and Path(raster_path).exists():
            vals = _sample_raster_to_points(gdf, raster_path)
            source = str(raster_path)
            method = "sample_authoritative_raster"
    if vals is None:
        raise RuntimeError("simple_river_authoritative_bed_stage_missing_authoritative_source")
    gdf["authoritative_bed_z_m"] = pd.to_numeric(vals, errors="coerce")
    gdf["authoritative_bed_source"] = str(source)
    gdf["authoritative_bed_method"] = str(method)
    finite_mask = np.isfinite(gdf["authoritative_bed_z_m"].to_numpy(dtype=float))
    if np.count_nonzero(finite_mask) == 0:
        raise RuntimeError("simple_river_authoritative_bed_stage_no_finite_authoritative_bed_values")
    if np.count_nonzero(~finite_mask) > 0:
        warnings.append("authoritative_bed_contains_missing_values")
    keep = [c for c in ("point_id", "station_m", "authoritative_bed_z_m", "authoritative_bed_source", "authoritative_bed_method", "geometry") if c in gdf.columns]
    out = gpd.GeoDataFrame(gdf.loc[finite_mask, keep].copy(), geometry="geometry", crs=gdf.crs)
    sort_cols = [c for c in ("station_m", "point_id") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out, warnings


def validate_centerline_authoritative_bed(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    missing = [f for f in _REQUIRED_FIELDS if f not in gdf.columns]
    vals = pd.to_numeric(gdf["authoritative_bed_z_m"], errors="coerce").to_numpy(dtype=float) if "authoritative_bed_z_m" in gdf.columns else np.full((len(gdf),), np.nan)
    recs = int(len(gdf))
    finite = int(np.count_nonzero(np.isfinite(vals)))
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
    return {
        "valid": not missing and recs > 0 and finite > 0 and geometry_valid and monotonic_by_reach and point_id_unique,
        "record_count": recs,
        "required_fields_present": not missing,
        "missing_required_fields": missing,
        "finite_authoritative_bed_count": finite,
        "geometry_valid": geometry_valid,
        "station_monotonic_by_reach": monotonic_by_reach,
        "point_id_unique": point_id_unique,
        "duplicate_point_ids": int(0 if "point_id" not in gdf.columns else gdf["point_id"].duplicated().sum()),
    }


def write_centerline_authoritative_bed_gpkg(gdf: gpd.GeoDataFrame, out_path: str) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG")
    return str(path)


def build_centerline_authoritative_bed_points(*, river_context: dict[str, Any], out_path: str, receipt_path: str | None = None) -> dict[str, Any]:
    centerline = _load_centerline_points(river_context=river_context)
    gdf, warnings = sample_authoritative_bed_to_centerline(centerline, river_context)
    validation = validate_centerline_authoritative_bed(gdf)
    if not validation.get("valid"):
        raise RuntimeError(f"simple_river_authoritative_bed_stage_invalid:{validation}")
    written_path = write_centerline_authoritative_bed_gpkg(gdf, out_path)
    receipt = build_stage_receipt(
        stage_id=STAGE_CENTERLINE_AUTHORITATIVE_BED,
        output_artifact=written_path,
        input_artifacts=[str(v) for v in river_context.get("input_artifacts", [])],
        record_count=int(len(gdf)),
        field_schema=centerline_authoritative_bed_field_schema(),
        vertical_reference=str(river_context.get("vertical_reference") or "unknown"),
        warnings=warnings,
        source_logic="centerline_authoritative_bed_from_centerline_field_or_authoritative_raster",
        validation=validation,
    )
    written_receipt = None
    if receipt_path:
        written_receipt = write_stage_receipt(receipt, receipt_path)
    return {
        "stage_id": STAGE_CENTERLINE_AUTHORITATIVE_BED,
        "output_artifact": written_path,
        "receipt_path": written_receipt,
        "record_count": int(len(gdf)),
        "validation": validation,
        "warnings": warnings,
    }
