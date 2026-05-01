from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from legacy.river.simple_river_stage_contract import STAGE_CENTERLINE_OBSERVED_OFFSET
from legacy.river.simple_river_stage_receipts import build_stage_receipt, write_stage_receipt

_REQUIRED_FIELDS = ("point_id", "station_m", "wse_proxy_z_m", "authoritative_bed_z_m", "observed_offset_m", "geometry")


def centerline_observed_offset_field_schema() -> dict[str, str]:
    return {
        "point_id": "string",
        "station_m": "float64",
        "wse_proxy_z_m": "float64",
        "authoritative_bed_z_m": "float64",
        "observed_offset_m": "float64",
        "offset_source": "string",
        "offset_valid": "bool",
        "geometry": "Point",
    }


def _load_points_from_context(*, river_context: dict[str, Any], key_gdf: str, key_path: str, stage_name: str) -> gpd.GeoDataFrame:
    gdf = river_context.get(key_gdf)
    if gdf is not None:
        return gdf.copy()
    path = river_context.get(key_path)
    if path and Path(path).exists():
        return gpd.read_file(path)
    raise RuntimeError(f"simple_river_observed_offset_stage_missing_{stage_name}")


def _normalize_join_columns(gdf: gpd.GeoDataFrame, value_col: str) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["point_id"] = out["point_id"].astype(str)
    out["station_m"] = pd.to_numeric(out["station_m"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    return out


def compute_observed_offsets(wse_points_gdf: gpd.GeoDataFrame, authoritative_bed_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list[str]]:
    warnings: list[str] = []
    wse = _normalize_join_columns(wse_points_gdf, "wse_proxy_z_m")
    bed = _normalize_join_columns(authoritative_bed_gdf, "authoritative_bed_z_m")
    if "point_id" not in wse.columns or "point_id" not in bed.columns:
        raise RuntimeError("simple_river_observed_offset_stage_missing_point_id_for_join")
    wse = wse.sort_values([c for c in ("point_id", "station_m") if c in wse.columns], kind="mergesort").drop_duplicates(subset=["point_id"], keep="last")
    bed = bed.sort_values([c for c in ("point_id", "station_m") if c in bed.columns], kind="mergesort").drop_duplicates(subset=["point_id"], keep="last")
    bed_cols = [c for c in ("point_id", "station_m", "authoritative_bed_z_m") if c in bed.columns]
    merged = wse.merge(bed[bed_cols], on=["point_id"], how="inner", suffixes=("", "_bed"))
    if len(merged) == 0:
        raise RuntimeError("simple_river_observed_offset_stage_no_joined_points")
    if "station_m_bed" in merged.columns:
        station_delta = np.abs(pd.to_numeric(merged["station_m"], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(merged["station_m_bed"], errors="coerce").to_numpy(dtype=float))
        mismatch = np.isfinite(station_delta) & (station_delta > 1e-6)
        if np.count_nonzero(mismatch) > 0:
            warnings.append("observed_offset_station_mismatch_between_wse_and_bed")
        merged["station_m"] = pd.to_numeric(merged["station_m"], errors="coerce").where(~pd.to_numeric(merged["station_m"], errors="coerce").isna(), pd.to_numeric(merged["station_m_bed"], errors="coerce"))
    merged["observed_offset_m"] = pd.to_numeric(merged["wse_proxy_z_m"], errors="coerce") - pd.to_numeric(merged["authoritative_bed_z_m"], errors="coerce")
    merged["offset_source"] = "wse_minus_authoritative_bed"
    merged["offset_valid"] = np.isfinite(merged["observed_offset_m"].to_numpy(dtype=float))
    negative_count = int(np.count_nonzero(merged["offset_valid"].to_numpy(dtype=bool) & (merged["observed_offset_m"].to_numpy(dtype=float) < 0.0)))
    if negative_count > 0:
        warnings.append("observed_offset_contains_negative_values")
    missing_count = int(np.count_nonzero(~merged["offset_valid"].to_numpy(dtype=bool)))
    if missing_count > 0:
        warnings.append("observed_offset_contains_missing_values")
    keep = [c for c in ("point_id", "station_m", "wse_proxy_z_m", "authoritative_bed_z_m", "observed_offset_m", "offset_source", "offset_valid", "geometry") if c in merged.columns]
    out = gpd.GeoDataFrame(merged[keep].copy(), geometry="geometry", crs=wse.crs)
    sort_cols = [c for c in ("levelpath_id", "reach_id", "source_reach_key", "station_m", "point_id") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out, warnings


def validate_observed_offsets(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    missing = [f for f in _REQUIRED_FIELDS if f not in gdf.columns]
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
    return {
        "valid": not missing and recs > 0 and finite > 0 and geometry_valid and monotonic_by_reach and point_id_unique,
        "record_count": recs,
        "required_fields_present": not missing,
        "missing_required_fields": missing,
        "finite_observed_offset_count": finite,
        "negative_observed_offset_count": negative,
        "geometry_valid": geometry_valid,
        "station_monotonic_by_reach": monotonic_by_reach,
        "point_id_unique": point_id_unique,
        "duplicate_point_ids": int(0 if "point_id" not in gdf.columns else gdf["point_id"].duplicated().sum()),
    }


def write_centerline_observed_offset_gpkg(gdf: gpd.GeoDataFrame, out_path: str) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG")
    return str(path)


def build_centerline_observed_offset_points(*, river_context: dict[str, Any], out_path: str, receipt_path: str | None = None) -> dict[str, Any]:
    wse = _load_points_from_context(river_context=river_context, key_gdf="wse_points_gdf", key_path="wse_points_path", stage_name="wse_points")
    bed = _load_points_from_context(river_context=river_context, key_gdf="authoritative_bed_gdf", key_path="authoritative_bed_points_path", stage_name="authoritative_bed_points")
    gdf, warnings = compute_observed_offsets(wse, bed)
    validation = validate_observed_offsets(gdf)
    if not validation.get("valid"):
        raise RuntimeError(f"simple_river_observed_offset_stage_invalid:{validation}")
    written_path = write_centerline_observed_offset_gpkg(gdf, out_path)
    receipt = build_stage_receipt(
        stage_id=STAGE_CENTERLINE_OBSERVED_OFFSET,
        output_artifact=written_path,
        input_artifacts=[str(v) for v in river_context.get("input_artifacts", [])],
        record_count=int(len(gdf)),
        field_schema=centerline_observed_offset_field_schema(),
        vertical_reference=str(river_context.get("vertical_reference") or "unknown"),
        warnings=warnings,
        source_logic="observed_offset_from_centerline_wse_minus_authoritative_bed",
        validation=validation,
    )
    written_receipt = write_stage_receipt(receipt, receipt_path) if receipt_path else None
    return {
        "stage_id": STAGE_CENTERLINE_OBSERVED_OFFSET,
        "output_artifact": written_path,
        "receipt_path": written_receipt,
        "record_count": int(len(gdf)),
        "validation": validation,
        "warnings": warnings,
    }
