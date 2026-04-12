from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from river_v2_context import RiverV2Context
from river_v2_contract import RiverV2StageResult, STAGE_CENTERLINE_OBSERVED_OFFSET
from river_v2_receipts import build_river_v2_stage_receipt, write_river_v2_receipt

_REQUIRED_FIELDS = ("point_id", "station_m", "wse_proxy_z_m", "authoritative_bed_z_m", "observed_offset_m", "geometry")


def centerline_observed_offset_field_schema() -> dict[str, str]:
    return {
        "point_id": "string",
        "station_m": "float64",
        "wse_proxy_z_m": "float64",
        "authoritative_bed_z_m": "float64",
        "observed_offset_m": "float64",
        "geometry": "Point",
    }


def _load_points(path: Path, label: str) -> gpd.GeoDataFrame:
    if path.exists():
        return gpd.read_file(path)
    raise RuntimeError(f"river_v2_observed_offset_missing_{label}:{path}")


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
        raise RuntimeError("river_v2_observed_offset_missing_point_id_for_join")
    wse = wse.sort_values([c for c in ("point_id", "station_m") if c in wse.columns], kind="mergesort").drop_duplicates(subset=["point_id"], keep="last")
    bed = bed.sort_values([c for c in ("point_id", "station_m") if c in bed.columns], kind="mergesort").drop_duplicates(subset=["point_id"], keep="last")
    bed_cols = [c for c in ("point_id", "station_m", "authoritative_bed_z_m") if c in bed.columns]
    if len(bed) == 0:
        warnings.append("observed_offset_zero_authoritative_bed_points_low_support_mode")
        keep = [c for c in (
            "point_id", "station_m", "component_id", "levelpath_id", "reach_id", "source_reach_key",
            "wse_proxy_z_m", "authoritative_bed_z_m", "observed_offset_m", "geometry"
        ) if c in wse.columns]
        empty = wse[keep].copy()
        if "authoritative_bed_z_m" not in empty.columns:
            empty["authoritative_bed_z_m"] = np.nan
        if "observed_offset_m" not in empty.columns:
            empty["observed_offset_m"] = np.nan
        empty = empty.iloc[0:0].copy()
        out = gpd.GeoDataFrame(empty, geometry="geometry", crs=wse.crs)
        return out, warnings
    merged = wse.merge(bed[bed_cols], on=["point_id"], how="inner", suffixes=("", "_bed"))
    if len(merged) == 0:
        warnings.append("observed_offset_zero_joined_points_low_support_mode")
        keep = [c for c in (
            "point_id", "station_m", "component_id", "levelpath_id", "reach_id", "source_reach_key",
            "wse_proxy_z_m", "authoritative_bed_z_m", "observed_offset_m", "geometry"
        ) if c in wse.columns]
        empty = wse[keep].copy()
        if "authoritative_bed_z_m" not in empty.columns:
            empty["authoritative_bed_z_m"] = np.nan
        if "observed_offset_m" not in empty.columns:
            empty["observed_offset_m"] = np.nan
        empty = empty.iloc[0:0].copy()
        out = gpd.GeoDataFrame(empty, geometry="geometry", crs=wse.crs)
        return out, warnings
    if "station_m_bed" in merged.columns:
        sta = pd.to_numeric(merged["station_m"], errors="coerce").to_numpy(dtype=float)
        bed_sta = pd.to_numeric(merged["station_m_bed"], errors="coerce").to_numpy(dtype=float)
        mismatch = np.isfinite(sta) & np.isfinite(bed_sta) & (np.abs(sta - bed_sta) > 1.0e-6)
        if np.count_nonzero(mismatch) > 0:
            warnings.append("observed_offset_station_mismatch_between_wse_and_bed")
        merged["station_m"] = pd.to_numeric(merged["station_m"], errors="coerce").where(~pd.to_numeric(merged["station_m"], errors="coerce").isna(), pd.to_numeric(merged["station_m_bed"], errors="coerce"))
    merged["observed_offset_m"] = pd.to_numeric(merged["wse_proxy_z_m"], errors="coerce") - pd.to_numeric(merged["authoritative_bed_z_m"], errors="coerce")
    merged["offset_source"] = "wse_minus_authoritative_bed"
    merged["offset_valid"] = np.isfinite(merged["observed_offset_m"].to_numpy(dtype=float))
    offset_vals = merged["observed_offset_m"].to_numpy(dtype=float)
    if np.count_nonzero(merged["offset_valid"].to_numpy(dtype=bool) & (offset_vals < 0.0)) > 0:
        warnings.append("observed_offset_contains_negative_values")
    if np.count_nonzero(~merged["offset_valid"].to_numpy(dtype=bool)) > 0:
        warnings.append("observed_offset_contains_missing_values")
    wse_vals = pd.to_numeric(merged["wse_proxy_z_m"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(wse_vals).any():
        rounded = np.round(wse_vals[np.isfinite(wse_vals)], 6)
        uniq, counts = np.unique(rounded, return_counts=True)
        dominant_fraction = float(np.max(counts) / max(rounded.size, 1))
        if uniq.size <= max(3, int(0.01 * rounded.size)) or dominant_fraction >= 0.5:
            warnings.append("observed_offset_wse_input_constant_like")
    if np.isfinite(offset_vals).any():
        rounded_off = np.round(offset_vals[np.isfinite(offset_vals)], 6)
        uniq_off, counts_off = np.unique(rounded_off, return_counts=True)
        dominant_off_fraction = float(np.max(counts_off) / max(rounded_off.size, 1))
        if uniq_off.size <= max(3, int(0.01 * rounded_off.size)) or dominant_off_fraction >= 0.5:
            warnings.append("observed_offset_constant_like")
    keep = [c for c in (
        "point_id", "station_m", "component_id", "levelpath_id", "reach_id", "source_reach_key",
        "wse_proxy_z_m", "authoritative_bed_z_m", "observed_offset_m", "geometry"
    ) if c in merged.columns]
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


def write_centerline_observed_offset_gpkg(gdf: gpd.GeoDataFrame, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG")
    return path


def run_observed_offset_stage(
    ctx: RiverV2Context,
    *,
    wse_points_path: Path,
    authoritative_bed_points_path: Path,
) -> RiverV2StageResult:
    wse = _load_points(wse_points_path, "wse_points")
    bed = _load_points(authoritative_bed_points_path, "authoritative_bed_points")
    gdf, warnings = compute_observed_offsets(wse, bed)
    validation = validate_observed_offsets(gdf)
    if not validation.get("valid"):
        raise RuntimeError(f"river_v2_observed_offset_invalid:{validation}")
    written_path = write_centerline_observed_offset_gpkg(gdf, ctx.paths.centerline_observed_offset_points)
    receipt = build_river_v2_stage_receipt(
        stage_id=STAGE_CENTERLINE_OBSERVED_OFFSET,
        output_artifact=str(written_path),
        input_artifacts=ctx.direct_stage_input_artifacts(wse_points_path, authoritative_bed_points_path),
        record_count=int(len(gdf)),
        field_schema=centerline_observed_offset_field_schema(),
        vertical_reference=ctx.vertical_reference,
        warnings=warnings,
        source_logic="observed_offset_from_centerline_wse_minus_authoritative_bed_native_v2",
        validation=validation,
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.centerline_observed_offset_points_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_CENTERLINE_OBSERVED_OFFSET,
        output_artifact=written_path,
        receipt_path=receipt_path,
        record_count=int(len(gdf)),
        validation=validation,
        warnings=warnings,
    )
