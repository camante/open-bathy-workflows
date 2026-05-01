from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rio_transform

from core.nodata_utils import sanitize_array

from legacy.river.river_v2_context import RiverV2Context
from legacy.river.river_v2_contract import RiverV2StageResult, STAGE_CENTERLINE_AUTHORITATIVE_BED
from legacy.river.river_v2_receipts import build_river_v2_stage_receipt, write_river_v2_receipt

_REQUIRED_FIELDS = ("point_id", "station_m", "authoritative_bed_z_m", "geometry")
def centerline_authoritative_bed_field_schema() -> dict[str, str]:
    return {
        "point_id": "string",
        "station_m": "float64",
        "authoritative_bed_z_m": "float64",
        "geometry": "Point",
    }


def _load_centerline_points(path: Path) -> gpd.GeoDataFrame:
    if path.exists():
        return gpd.read_file(path)
    raise RuntimeError("river_v2_authoritative_bed_missing_centerline_points")

def _write_authoritative_bed_diagnostics(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_authoritative_support_points_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"x", "y", "depth_m"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"river_v2_authoritative_bed_csv_missing_columns:{missing}")
    for col in ("x", "y", "depth_m"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "authoritative_role" in df.columns:
        role = df["authoritative_role"].astype(str).str.lower()
        keep = role.str.contains("bed") | role.str.contains("channel")
        if keep.any():
            df = df.loc[keep].copy()
    df = df.loc[np.isfinite(df["x"]) & np.isfinite(df["y"]) & np.isfinite(df["depth_m"])].copy()
    return df.reset_index(drop=True)


def _sample_support_points_to_centerline(gdf: gpd.GeoDataFrame, csv_path: Path, *, max_distance_m: float = 25.0) -> tuple[np.ndarray, dict[str, Any]]:
    pts = _load_authoritative_support_points_csv(csv_path)
    diagnostics = {
        "sample_csv_path": str(csv_path),
        "sample_csv_point_count": int(len(pts)),
        "sample_max_distance_m": float(max_distance_m),
        "centerline_reprojected_to_points_crs": False,
    }
    vals = np.full((len(gdf),), np.nan, dtype=float)
    if len(gdf) == 0 or pts.empty:
        diagnostics.update({"matched_point_count": 0, "miss_point_count": int(len(gdf))})
        return vals, diagnostics
    if gdf.crs is None:
        raise RuntimeError('river_v2_authoritative_bed_centerline_missing_crs')
    # assume support points are already in working/projected CRS used for export
    xs = gdf.geometry.x.to_numpy(dtype=float)
    ys = gdf.geometry.y.to_numpy(dtype=float)
    sx = pts["x"].to_numpy(dtype=float)
    sy = pts["y"].to_numpy(dtype=float)
    sz = pts["depth_m"].to_numpy(dtype=float)
    max_d2 = float(max_distance_m) ** 2
    matched = 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        d2 = (sx - x) ** 2 + (sy - y) ** 2
        if d2.size == 0:
            continue
        j = int(np.argmin(d2))
        if np.isfinite(d2[j]) and d2[j] <= max_d2:
            vals[i] = float(sz[j])
            matched += 1
    diagnostics.update({
        "matched_point_count": int(matched),
        "miss_point_count": int(len(gdf) - matched),
    })
    return vals, diagnostics


def _sample_raster_to_points(gdf: gpd.GeoDataFrame, raster_path: Path, *, search_radius_cells: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(raster_path) as ds:
        if gdf.crs is None:
            raise RuntimeError('river_v2_authoritative_bed_centerline_missing_crs')
        target_crs = ds.crs
        xs = gdf.geometry.x.to_numpy(dtype=float)
        ys = gdf.geometry.y.to_numpy(dtype=float)
        if target_crs is None:
            raise RuntimeError('river_v2_authoritative_bed_raster_missing_crs')
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
        in_extent = (
            finite_xy & np.isfinite(rows) & np.isfinite(cols) &
            (rows >= 0) & (rows < ds.height) &
            (cols >= 0) & (cols < ds.width)
        )

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


def _finalize_authoritative_bed_output(gdf: gpd.GeoDataFrame, vals: np.ndarray, *, warnings: list[str], method: str | None, diagnostics: dict[str, Any], zero_value_warning: str) -> tuple[gpd.GeoDataFrame, list[str], str | None, dict[str, Any]]:
    out_gdf = gdf.copy()
    out_gdf["authoritative_bed_z_m"] = pd.to_numeric(vals, errors="coerce")
    finite_mask = np.isfinite(out_gdf["authoritative_bed_z_m"].to_numpy(dtype=float))
    diagnostics.update({
        "finite_authoritative_sample_count": int(np.count_nonzero(finite_mask)),
        "nodata_rejected_count": int(np.count_nonzero(~finite_mask)),
    })
    if np.count_nonzero(finite_mask) == 0 and zero_value_warning not in warnings:
        warnings.append(zero_value_warning)
    keep = [c for c in (
        "point_id", "station_m", "component_id", "levelpath_id", "reach_id", "source_reach_key",
        "authoritative_bed_z_m", "geometry"
    ) if c in out_gdf.columns]
    out = gpd.GeoDataFrame(out_gdf.loc[finite_mask, keep].copy(), geometry="geometry", crs=out_gdf.crs)
    sort_cols = [c for c in ("station_m", "point_id") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    diagnostics["written_record_count"] = int(len(out))
    return out, warnings, method, diagnostics


def _sample_authoritative_bed_from_support_points(centerline_points_gdf: gpd.GeoDataFrame, authoritative_support_points_path: Path | None) -> tuple[gpd.GeoDataFrame, list[str], str | None, dict[str, Any]]:
    gdf = centerline_points_gdf.copy()
    warnings: list[str] = []
    diagnostics: dict[str, Any] = {
        "stations_attempted": int(len(gdf)),
        "authoritative_source_kind": "missing",
        "support_points_path": str(authoritative_support_points_path) if authoritative_support_points_path is not None else None,
    }
    vals = np.full((len(gdf),), np.nan, dtype=float)
    method: str | None = None
    sample_diag: dict[str, Any] = {}

    if authoritative_support_points_path is None:
        warnings.append("authoritative_bed_missing_support_points_artifact")
        method = "missing_authoritative_support_points"
        diagnostics["authoritative_bed_failure_reason"] = "missing_explicit_support_points_artifact"
    else:
        support_path = Path(authoritative_support_points_path)
        if not support_path.exists():
            warnings.append("authoritative_bed_missing_support_points_artifact")
            method = "missing_authoritative_support_points"
            diagnostics["authoritative_bed_failure_reason"] = "missing_explicit_support_points_artifact"
        else:
            vals, sample_diag = _sample_support_points_to_centerline(gdf, support_path)
            method = "nearest_authoritative_support_points"
            diagnostics["authoritative_source_kind"] = "csv_support_points"

    diagnostics.update(sample_diag)
    diagnostics.setdefault("matched_point_count", int(np.count_nonzero(np.isfinite(vals))))
    diagnostics.setdefault("miss_point_count", int(len(gdf) - np.count_nonzero(np.isfinite(vals))))
    diagnostics.setdefault("sample_csv_path", str(authoritative_support_points_path) if authoritative_support_points_path is not None else None)
    diagnostics.setdefault("sample_csv_point_count", 0)
    return _finalize_authoritative_bed_output(
        gdf,
        vals,
        warnings=warnings,
        method=method,
        diagnostics=diagnostics,
        zero_value_warning="authoritative_bed_no_finite_values",
    )


def _sample_authoritative_bed_from_raster_path(
    centerline_points_gdf: gpd.GeoDataFrame,
    authoritative_support_raster_path: Path | None,
    *,
    trusted_mode: str | None = None,
    policy_warning: str | None = None,
) -> tuple[gpd.GeoDataFrame, list[str], str | None, dict[str, Any]]:
    gdf = centerline_points_gdf.copy()
    warnings: list[str] = []
    diagnostics: dict[str, Any] = {
        "stations_attempted": int(len(gdf)),
        "trusted_support_mode": str(trusted_mode or "unknown"),
        "policy_warning": str(policy_warning) if policy_warning else None,
        "authoritative_support_raster_path": str(authoritative_support_raster_path) if authoritative_support_raster_path is not None else None,
    }
    vals = np.full((len(gdf),), np.nan, dtype=float)
    method = None
    sample_diag: dict[str, Any] = {}
    support_raster = Path(authoritative_support_raster_path) if authoritative_support_raster_path is not None else None
    if support_raster is None or not support_raster.exists():
        if policy_warning:
            warnings.append(str(policy_warning))
        warnings.append("authoritative_bed_support_raster_missing")
        method = "missing_authoritative_support_raster"
        diagnostics["authoritative_source_kind"] = "missing"
        diagnostics["authoritative_bed_failure_reason"] = "missing_authoritative_support_raster"
    else:
        vals, sample_diag = _sample_raster_to_points(gdf, support_raster)
        method = "explicit_authoritative_support_raster"
        diagnostics["authoritative_source_kind"] = "sampling_raster"
    diagnostics.update(sample_diag)
    if np.count_nonzero(np.isfinite(vals)) == 0:
        diagnostics.setdefault("authoritative_bed_failure_reason", "no_finite_samples_from_explicit_authoritative_support_raster")
    return _finalize_authoritative_bed_output(
        gdf,
        vals,
        warnings=warnings,
        method=method,
        diagnostics=diagnostics,
        zero_value_warning="authoritative_bed_zero_valid_points_from_explicit_support_raster",
    )


def _sample_authoritative_bed_from_ctx(ctx: RiverV2Context, centerline_points_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list[str], str | None, dict[str, Any]]:
    gdf = centerline_points_gdf.copy()
    warnings: list[str] = []
    trusted_mode = str(getattr(ctx, "trusted_support_mode", "low_support_no_trusted_support") or "low_support_no_trusted_support")
    policy_warning = getattr(ctx, "authoritative_support_policy_warning", None)
    sampling_raster = Path(ctx.authoritative_sampling_raster_path) if ctx.authoritative_sampling_raster_path is not None else None
    diagnostics: dict[str, Any] = {
        "stations_attempted": int(len(gdf)),
        "authoritative_dem_mode": str(getattr(ctx, "authoritative_dem_mode", "mixed_requires_metadata") or "mixed_requires_metadata"),
        "trusted_support_mode": trusted_mode,
        "trusted_support_artifact_path": str(getattr(ctx, "trusted_support_artifact_path", "") or "") or None,
        "sampling_raster_path": str(sampling_raster) if sampling_raster is not None else None,
        "policy_warning": str(policy_warning) if policy_warning else None,
    }
    vals = np.full((len(gdf),), np.nan, dtype=float)
    method = None
    sample_diag: dict[str, Any] = {}

    if trusted_mode == "low_support_no_trusted_support":
        if policy_warning:
            warnings.append(str(policy_warning))
        else:
            warnings.append("authoritative_bed_metadata_unavailable_low_support_mode")
        method = "low_support_no_trusted_support"
        diagnostics["authoritative_bed_failure_reason"] = "trusted_support_mode_low_support"
    else:
        if sampling_raster is None or not sampling_raster.exists():
            warnings.append("authoritative_bed_sampling_raster_missing_low_support_mode")
            method = "missing_sampling_raster"
            diagnostics["authoritative_bed_failure_reason"] = "missing_authoritative_sampling_raster"
        else:
            vals, sample_diag = _sample_raster_to_points(gdf, sampling_raster)
            method = f"sampling_raster:{trusted_mode}"
            diagnostics["authoritative_source_kind"] = "sampling_raster"

    diagnostics.update(sample_diag)
    if np.count_nonzero(np.isfinite(vals)) == 0:
        diagnostics.setdefault("authoritative_bed_failure_reason", "no_finite_samples_from_resolved_support_policy")
    return _finalize_authoritative_bed_output(
        gdf,
        vals,
        warnings=warnings,
        method=method,
        diagnostics=diagnostics,
        zero_value_warning="authoritative_bed_zero_valid_points_low_support_mode",
    )


def sample_authoritative_bed_to_centerline(ctx_or_centerline_points_gdf: RiverV2Context | gpd.GeoDataFrame, centerline_points_gdf_or_support_points_path: gpd.GeoDataFrame | Path | None) -> tuple[gpd.GeoDataFrame, list[str], str | None, dict[str, Any]]:
    if isinstance(ctx_or_centerline_points_gdf, RiverV2Context):
        if not isinstance(centerline_points_gdf_or_support_points_path, gpd.GeoDataFrame):
            raise TypeError("river_v2_authoritative_bed_expected_centerline_gdf_for_context_mode")
        return _sample_authoritative_bed_from_ctx(ctx_or_centerline_points_gdf, centerline_points_gdf_or_support_points_path)
    if not isinstance(ctx_or_centerline_points_gdf, gpd.GeoDataFrame):
        raise TypeError("river_v2_authoritative_bed_expected_centerline_gdf")
    support_path = Path(centerline_points_gdf_or_support_points_path) if centerline_points_gdf_or_support_points_path is not None else None
    return _sample_authoritative_bed_from_support_points(ctx_or_centerline_points_gdf, support_path)


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
    point_id_unique = True if recs == 0 else (bool(gdf["point_id"].is_unique) if "point_id" in gdf.columns else False)
    return {
        "valid": not missing and geometry_valid and monotonic_by_reach and point_id_unique,
        "record_count": recs,
        "required_fields_present": not missing,
        "missing_required_fields": missing,
        "finite_authoritative_bed_count": finite,
        "geometry_valid": geometry_valid,
        "station_monotonic_by_reach": monotonic_by_reach,
        "point_id_unique": point_id_unique,
        "duplicate_point_ids": int(0 if "point_id" not in gdf.columns else gdf["point_id"].duplicated().sum()),
        "zero_authoritative_bed_points": bool(recs == 0),
        "zero_finite_authoritative_bed_points": bool(finite == 0),
    }


def write_centerline_authoritative_bed_gpkg(gdf: gpd.GeoDataFrame, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG")
    return path


def run_authoritative_bed_stage(ctx: RiverV2Context, *, centerline_points_path: Path, authoritative_support_raster_path: Path | None) -> RiverV2StageResult:
    centerline = _load_centerline_points(centerline_points_path)
    authoritative_source = Path(authoritative_support_raster_path) if authoritative_support_raster_path is not None else None
    gdf, warnings, method, diagnostics = _sample_authoritative_bed_from_raster_path(
        centerline,
        authoritative_source,
        trusted_mode=str(getattr(ctx, "trusted_support_mode", "unknown") or "unknown"),
        policy_warning=getattr(ctx, "authoritative_support_policy_warning", None),
    )
    validation = validate_centerline_authoritative_bed(gdf)
    finite_pixels_in_source = int(pd.to_numeric(pd.Series([diagnostics.get("sample_raster_finite_pixel_count")]), errors="coerce").fillna(0).iloc[0])
    no_samples_in_solve_domain = bool(
        (validation.get("zero_authoritative_bed_points") or validation.get("zero_finite_authoritative_bed_points"))
        and finite_pixels_in_source > 0
    )
    support_status_at_entry = str(getattr(ctx, "system_support_status", "unknown") or "unknown")
    if no_samples_in_solve_domain:
        diagnostics["support_status_at_stage_entry"] = support_status_at_entry
        diagnostics["authoritative_bed_failure_reason"] = "no_solve_domain_centerline_intersection_with_authoritative_support_raster"
        if support_status_at_entry == "authoritative_control_found":
            raise RuntimeError(
                "river_v2_authoritative_bed_missing_centerline_samples_for_canonical_authoritative_mode"
            )
        continuation_warning = "authoritative_bed_no_samples_in_solve_domain_continuing_scaffold_inferred"
        if continuation_warning not in warnings:
            warnings.append(continuation_warning)
        ctx.system_support_status = "scaffold_inferred"
        ctx.authoritative_control_found = False
        ctx.scaffold_inference_allowed = True
        diagnostics["continuation_mode"] = "scaffold_inferred_modeled_offsets_from_transfer_priors_or_placeholders"
        diagnostics["support_status_at_stage_exit"] = str(ctx.system_support_status)
        logging.getLogger(__name__).warning(
            "[RIVER][V2][AUTHORITATIVE_BED] No solve-domain authoritative bed samples intersect the centerline; continuing in scaffold-inferred mode."
        )

    diag_payload = {
        "selected_authoritative_source": str(authoritative_source) if authoritative_source is not None else None,
        "candidate_sources": [str(authoritative_source)] if authoritative_source is not None else [],
        "method": method,
        "diagnostics": diagnostics,
        "validation": validation,
        "warnings": warnings,
        "centerline_points_path": str(centerline_points_path),
        "centerline_point_count": int(len(centerline)),
        "debug_interpretation": (
            "authoritative_bed_stage_continues_in_scaffold_inferred_mode because the solve-domain centerline has no authoritative bed intersections"
            if no_samples_in_solve_domain else
            (
                "authoritative_bed_stage_failed_validation"
                if not validation.get("valid") else
                "authoritative_bed_stage_succeeded"
            )
        ),
    }
    _write_authoritative_bed_diagnostics(ctx.paths.centerline_authoritative_bed_diagnostics, diag_payload)
    if not validation.get("valid"):
        raise RuntimeError(f"river_v2_authoritative_bed_invalid:{validation}")
    written_path = write_centerline_authoritative_bed_gpkg(gdf, ctx.paths.centerline_authoritative_bed_points)
    receipt = build_river_v2_stage_receipt(
        stage_id=STAGE_CENTERLINE_AUTHORITATIVE_BED,
        output_artifact=str(written_path),
        input_artifacts=ctx.direct_stage_input_artifacts(
            centerline_points_path,
            authoritative_source,
        ),
        record_count=int(len(gdf)),
        field_schema=centerline_authoritative_bed_field_schema(),
        vertical_reference=ctx.vertical_reference,
        warnings=warnings,
        source_logic=f"centerline_authoritative_bed_native_v2:{method or 'unknown'}:explicit_authoritative_support_raster",
        validation={**validation, **diagnostics},
        extra_artifacts={
            "diagnostics_path": str(ctx.paths.centerline_authoritative_bed_diagnostics),
        },
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.centerline_authoritative_bed_points_receipt)
    aux_outputs = {"authoritative_support_raster": str(authoritative_source)} if authoritative_source is not None else {}
    if no_samples_in_solve_domain:
        aux_outputs["river_v2_authoritative_bed_continuation_mode"] = "scaffold_inferred"
    return RiverV2StageResult(
        stage_id=STAGE_CENTERLINE_AUTHORITATIVE_BED,
        output_artifact=written_path,
        receipt_path=receipt_path,
        record_count=int(len(gdf)),
        validation=validation,
        warnings=warnings,
        aux_outputs=aux_outputs,
    )
