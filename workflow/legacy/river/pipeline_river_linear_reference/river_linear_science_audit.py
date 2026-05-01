from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

from pipeline.river_linear.river_linear_context import RiverLinearContext


FLATNESS_DOMINANT_FRACTION_THRESHOLD = 0.65
FLATNESS_MIN_FINITE_COUNT = 20
ROUND_DECIMALS = 2


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _finite_values(values: Any) -> np.ndarray:
    vals = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    return vals[np.isfinite(vals)]


def _value_summary(values: Any, *, threshold: float = FLATNESS_DOMINANT_FRACTION_THRESHOLD) -> dict[str, Any]:
    vals = _finite_values(values)
    summary: dict[str, Any] = {
        "finite_count": int(vals.size),
        "min_m": None,
        "max_m": None,
        "median_m": None,
        "range_m": None,
        "dominant_rounded_value_m": None,
        "dominant_rounded_count": 0,
        "dominant_rounded_fraction": None,
        "flatness_threshold": float(threshold),
        "flatness_min_finite_count": int(FLATNESS_MIN_FINITE_COUNT),
        "flatness_fail": False,
    }
    if vals.size == 0:
        return summary
    summary.update(
        {
            "min_m": float(np.nanmin(vals)),
            "max_m": float(np.nanmax(vals)),
            "median_m": float(np.nanmedian(vals)),
            "range_m": float(np.nanmax(vals) - np.nanmin(vals)),
        }
    )
    rounded = np.round(vals, ROUND_DECIMALS)
    unique, counts = np.unique(rounded, return_counts=True)
    if counts.size:
        idx = int(np.argmax(counts))
        dominant_count = int(counts[idx])
        fraction = float(dominant_count / vals.size)
        summary.update(
            {
                "dominant_rounded_value_m": float(unique[idx]),
                "dominant_rounded_count": dominant_count,
                "dominant_rounded_fraction": fraction,
                "flatness_fail": bool(vals.size >= FLATNESS_MIN_FINITE_COUNT and fraction >= threshold),
            }
        )
    return summary


def _read_export_bounds(export_grid_template: Path) -> tuple[Any, Any]:
    with rasterio.open(export_grid_template) as ds:
        return ds.bounds, ds.crs


def _points_in_export_bounds(path: Path, export_grid_template: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gdf
    bounds, grid_crs = _read_export_bounds(export_grid_template)
    work = gdf
    try:
        if work.crs is not None and grid_crs is not None and str(work.crs) != str(grid_crs):
            work = work.to_crs(grid_crs)
    except Exception:
        # If CRS transformation fails, keep the unfiltered points so the audit
        # still records stage-level values rather than hiding the problem.
        return gdf
    try:
        x = work.geometry.x.to_numpy(dtype=float)
        y = work.geometry.y.to_numpy(dtype=float)
        mask = (x >= bounds.left) & (x <= bounds.right) & (y >= bounds.bottom) & (y <= bounds.top)
        return gdf.loc[mask].copy()
    except Exception:
        return gdf


def _point_stage_summary(
    *,
    name: str,
    path: Path,
    export_grid_template: Path,
    value_column: str,
    extra_columns: tuple[str, ...] = (),
    threshold: float = FLATNESS_DOMINANT_FRACTION_THRESHOLD,
    flatness_is_failure: bool = True,
) -> dict[str, Any]:
    stage: dict[str, Any] = {
        "stage": name,
        "artifact": str(path),
        "kind": "points",
        "value_column": value_column,
        "exists": bool(path.exists()),
        "record_count_export_aoi": 0,
        "flatness_is_failure": bool(flatness_is_failure),
        "value": _value_summary([], threshold=threshold),
        "extra_values": {},
    }
    if not path.exists():
        stage["error"] = "missing_artifact"
        return stage
    try:
        gdf = _points_in_export_bounds(path, export_grid_template)
        stage["record_count_export_aoi"] = int(len(gdf))
        if value_column in gdf.columns:
            stage["value"] = _value_summary(gdf[value_column], threshold=threshold)
        else:
            stage["error"] = f"missing_value_column:{value_column}"
        for col in extra_columns:
            if col in gdf.columns:
                stage["extra_values"][col] = _value_summary(gdf[col], threshold=threshold)
        if {"bed_backbone_raw_z_m", "bed_backbone_z_m"}.issubset(gdf.columns):
            raw_summary = _value_summary(gdf["bed_backbone_raw_z_m"], threshold=threshold)
            final_summary = _value_summary(gdf["bed_backbone_z_m"], threshold=threshold)
            wse_summary = _value_summary(gdf["wse_proxy_z_m"], threshold=threshold) if "wse_proxy_z_m" in gdf.columns else None
            offset_summary = _value_summary(gdf["offset_modeled_m"], threshold=threshold) if "offset_modeled_m" in gdf.columns else None
            raw_is_flat = bool(raw_summary.get("flatness_fail"))
            final_is_flat = bool(final_summary.get("flatness_fail"))
            wse_is_flat = bool(wse_summary and wse_summary.get("flatness_fail"))
            offset_is_flat = bool(offset_summary and offset_summary.get("flatness_fail"))
            if raw_is_flat and wse_is_flat and offset_is_flat:
                raw_flatness_source = "wse_and_offset_flat"
            elif raw_is_flat and wse_is_flat:
                raw_flatness_source = "wse_flat"
            elif raw_is_flat and offset_is_flat:
                raw_flatness_source = "flat_after_wse_minus_constant_offset"
            elif raw_is_flat:
                raw_flatness_source = "raw_formula_output_flat"
            else:
                raw_flatness_source = "raw_not_flat"
            modeled_raw_summary = _value_summary(gdf["offset_raw_bed_z_m"], threshold=threshold) if "offset_raw_bed_z_m" in gdf.columns else None
            raw_handoff_max_abs_diff_m = None
            if "offset_raw_bed_z_m" in gdf.columns:
                raw_vals = np.asarray(pd.to_numeric(gdf["bed_backbone_raw_z_m"], errors="coerce"), dtype=float)
                modeled_raw_vals = np.asarray(pd.to_numeric(gdf["offset_raw_bed_z_m"], errors="coerce"), dtype=float)
                mask = np.isfinite(raw_vals) & np.isfinite(modeled_raw_vals)
                if np.any(mask):
                    raw_handoff_max_abs_diff_m = float(np.nanmax(np.abs(raw_vals[mask] - modeled_raw_vals[mask])))
            positive_downstream_step_count = None
            max_positive_downstream_step_m = None
            if "station_downstream_m" in gdf.columns and "bed_backbone_z_m" in gdf.columns:
                sta = np.asarray(pd.to_numeric(gdf["station_downstream_m"], errors="coerce"), dtype=float)
                vals = np.asarray(pd.to_numeric(gdf["bed_backbone_z_m"], errors="coerce"), dtype=float)
                mask = np.isfinite(sta) & np.isfinite(vals)
                if np.count_nonzero(mask) > 1:
                    order = np.argsort(sta[mask], kind="mergesort")
                    diffs = np.diff(vals[mask][order])
                    pos = diffs[diffs > 1.0e-6]
                    positive_downstream_step_count = int(pos.size)
                    max_positive_downstream_step_m = float(np.nanmax(pos)) if pos.size else 0.0
            stage["backbone_transform_check"] = {
                "wse_proxy": wse_summary,
                "offset_modeled": offset_summary,
                "modeled_offset_raw_bed": modeled_raw_summary,
                "raw_bed": raw_summary,
                "final_bed": final_summary,
                "raw_flatness_source": raw_flatness_source,
                "raw_dominant_fraction": raw_summary.get("dominant_rounded_fraction"),
                "final_dominant_fraction": final_summary.get("dominant_rounded_fraction"),
                "raw_range_m": raw_summary.get("range_m"),
                "final_range_m": final_summary.get("range_m"),
                "raw_handoff_max_abs_diff_m": raw_handoff_max_abs_diff_m,
                "positive_downstream_step_count": positive_downstream_step_count,
                "max_positive_downstream_step_m": max_positive_downstream_step_m,
                "downstream_order_sign_counts": {str(k): int(v) for k, v in gdf["downstream_order_sign"].astype(str).value_counts(dropna=False).to_dict().items()} if "downstream_order_sign" in gdf.columns else {},
                "formula_guard_status_counts": {str(k): int(v) for k, v in gdf["backbone_formula_guard_status"].astype(str).value_counts(dropna=False).to_dict().items()} if "backbone_formula_guard_status" in gdf.columns else {},
                "flattening_introduced_by_stage": bool(
                    raw_summary.get("dominant_rounded_fraction") is not None
                    and final_summary.get("dominant_rounded_fraction") is not None
                    and raw_summary.get("dominant_rounded_fraction") < threshold
                    and final_summary.get("dominant_rounded_fraction") >= threshold
                ),
            }
        for cat_col in ("offset_source", "offset_support_class", "backbone_support_class", "wse_quality_flag", "wse_tail_policy"):
            if cat_col in gdf.columns:
                stage[f"{cat_col}_counts"] = {str(k): int(v) for k, v in gdf[cat_col].astype(str).value_counts(dropna=False).to_dict().items()}
    except Exception as exc:
        stage["error"] = f"read_failed:{type(exc).__name__}:{exc}"
    return stage


def _read_raster_values(
    raster_path: Path,
    *,
    mask_path: Path | None = None,
    export_grid_template: Path | None = None,
) -> np.ndarray:
    with rasterio.open(raster_path) as ds:
        window = None
        if export_grid_template is not None:
            try:
                bounds, grid_crs = _read_export_bounds(export_grid_template)
                if grid_crs is None or ds.crs is None or str(grid_crs) == str(ds.crs):
                    window = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, transform=ds.transform)
                    window = window.round_offsets().round_lengths()
            except Exception:
                window = None
        arr = ds.read(1, window=window, masked=True).astype("float64").filled(np.nan)
        nodata = ds.nodata
        if nodata is not None:
            arr[np.isclose(arr, float(nodata), equal_nan=False)] = np.nan
        if mask_path is not None and Path(mask_path).exists():
            try:
                with rasterio.open(mask_path) as ms:
                    mask_window = None
                    if window is not None and ms.width == ds.width and ms.height == ds.height and str(ms.transform) == str(ds.transform):
                        mask_window = window
                    mask = ms.read(1, window=mask_window, masked=True).filled(0)
                    if mask.shape == arr.shape:
                        arr = np.where(mask > 0, arr, np.nan)
            except Exception:
                pass
        return arr[np.isfinite(arr)]


def _raster_stage_summary(
    *,
    name: str,
    path: Path,
    mask_path: Path | None = None,
    export_grid_template: Path | None = None,
    threshold: float = FLATNESS_DOMINANT_FRACTION_THRESHOLD,
) -> dict[str, Any]:
    stage: dict[str, Any] = {
        "stage": name,
        "artifact": str(path),
        "kind": "raster",
        "mask": str(mask_path) if mask_path is not None else None,
        "exists": bool(path.exists()),
        "value": _value_summary([], threshold=threshold),
    }
    if not path.exists():
        stage["error"] = "missing_artifact"
        return stage
    try:
        values = _read_raster_values(path, mask_path=mask_path, export_grid_template=export_grid_template)
        stage["value"] = _value_summary(values, threshold=threshold)
    except Exception as exc:
        stage["error"] = f"read_failed:{type(exc).__name__}:{exc}"
    return stage


def _first_bad_stage(stages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for stage in stages:
        value = stage.get("value") if isinstance(stage, dict) else None
        if not bool(stage.get("flatness_is_failure", True)):
            continue
        if isinstance(value, dict) and bool(value.get("flatness_fail")):
            return {
                "stage": stage.get("stage"),
                "artifact": stage.get("artifact"),
                "kind": stage.get("kind"),
                "value_column": stage.get("value_column"),
                "dominant_rounded_value_m": value.get("dominant_rounded_value_m"),
                "dominant_rounded_fraction": value.get("dominant_rounded_fraction"),
                "finite_count": value.get("finite_count"),
            }
    return None


def write_river_science_chain_summary(
    *,
    ctx: RiverLinearContext,
    centerline_result: Any,
    wse_result: Any,
    authoritative_bed_result: Any,
    observed_offset_result: Any,
    modeled_offset_result: Any,
    backbone_result: Any,
    corridor_result: Any,
    surface_result: Any,
    lock_result: Any,
    export_result: Any,
    final_dem_result: Any,
    threshold: float = FLATNESS_DOMINANT_FRACTION_THRESHOLD,
) -> Path:
    """Write a compact audit for the river science chain in the current export AOI.

    This audit is intentionally narrow: it answers where a dominant rounded
    elevation first appears in the centerline/surface/final chain. It does not
    change the river science.
    """
    report_path = Path(ctx.cfg.out_dir) / "reports" / "river_science_chain_summary.json"
    export_grid = Path(export_result.export_grid_template_path)
    solve_export_grid = export_grid if export_grid.exists() else Path(ctx.paths.export_grid_template)
    solve_mask = Path(corridor_result.river_corridor_solve_path)
    solve_take_mask = None
    if ctx.linear_inputs is not None and getattr(ctx.linear_inputs, "canonical_solve_take_mask_path", None) is not None:
        solve_take_mask = Path(ctx.linear_inputs.canonical_solve_take_mask_path)
    export_corridor_mask = Path(export_result.river_corridor_mask_export_path)
    export_take_mask = Path(getattr(export_result, "take_export_mask_path", export_result.river_take_mask_export_path))

    stages = [
        _point_stage_summary(
            name="centerline_wse_proxy_points",
            path=Path(wse_result.centerline_wse_proxy_points_path),
            export_grid_template=solve_export_grid,
            value_column="wse_proxy_z_m",
            extra_columns=("wse_support_z_m", "wse_proxy_raw_interpolated_z_m", "wse_proxy_pre_monotone_z_m", "local_bed_floor_z_m", "wse_group_support_count", "wse_group_support_range_m", "wse_group_final_dominant_fraction"),
            threshold=threshold,
        ),
        _point_stage_summary(
            name="centerline_authoritative_bed_points",
            path=Path(authoritative_bed_result.centerline_authoritative_bed_points_path),
            export_grid_template=solve_export_grid,
            value_column="authoritative_bed_z_m",
            threshold=threshold,
        ),
        _point_stage_summary(
            name="centerline_observed_offset_points",
            path=Path(observed_offset_result.centerline_observed_offset_points_path),
            export_grid_template=solve_export_grid,
            value_column="observed_offset_m",
            extra_columns=("wse_proxy_z_m", "authoritative_bed_z_m"),
            threshold=threshold,
        ),
        _point_stage_summary(
            name="centerline_modeled_offset_points",
            path=Path(modeled_offset_result.centerline_modeled_offset_points_path),
            export_grid_template=solve_export_grid,
            value_column="offset_modeled_m",
            extra_columns=("wse_proxy_z_m", "offset_raw_bed_z_m", "offset_group_guard_status", "offset_group_wse_range_m", "offset_group_offset_range_m", "offset_group_raw_bed_range_m", "offset_group_raw_bed_dominant_fraction"),
            threshold=threshold,
            # A constant modeled depth/offset prior can be scientifically honest in
            # low-support reaches. It is diagnostic context for the backbone formula,
            # not by itself a flat DEM failure.
            flatness_is_failure=False,
        ),
        _point_stage_summary(
            name="centerline_bed_backbone_points",
            path=Path(backbone_result.centerline_bed_backbone_points_path),
            export_grid_template=solve_export_grid,
            value_column="bed_backbone_z_m",
            extra_columns=("offset_raw_bed_z_m", "bed_backbone_raw_z_m", "bed_backbone_adjustment_m", "station_downstream_m", "downstream_order_sign", "wse_station_slope_m_per_m", "backbone_group_wse_range_m", "backbone_group_offset_range_m", "backbone_group_raw_bed_range_m", "backbone_group_raw_bed_dominant_fraction", "backbone_group_raw_bed_handoff_max_abs_diff_m"),
            threshold=threshold,
        ),
        _raster_stage_summary(
            name="river_primary_surface_solve",
            path=Path(surface_result.river_primary_surface_solve_path),
            mask_path=solve_mask,
            export_grid_template=solve_export_grid,
            threshold=threshold,
        ),
        _raster_stage_summary(
            name="river_primary_surface_solve_locked",
            path=Path(lock_result.river_primary_surface_solve_locked_path),
            mask_path=solve_mask,
            export_grid_template=solve_export_grid,
            threshold=threshold,
        ),
        _raster_stage_summary(
            name="river_guidance_export",
            path=Path(export_result.river_guidance_export_path),
            mask_path=export_corridor_mask,
            threshold=threshold,
        ),
        _raster_stage_summary(
            name="canonical_parent_dem",
            path=Path(final_dem_result.canonical_parent_dem_path),
            mask_path=solve_take_mask,
            export_grid_template=solve_export_grid,
            threshold=threshold,
        ),
        _raster_stage_summary(
            name="aoi_export_dem",
            path=Path(final_dem_result.aoi_export_dem_path),
            mask_path=export_take_mask,
            threshold=threshold,
        ),
    ]
    first_bad = _first_bad_stage(stages)
    payload = {
        "version": 1,
        "purpose": "Identify the first river science-chain artifact that becomes dominated by one rounded elevation in the current export AOI.",
        "run_id": str(ctx.run_id),
        "export_aoi": str(ctx.export_aoi),
        "requested_solve_domain": str(ctx.requested_solve_domain) if ctx.requested_solve_domain is not None else None,
        "resolved_solve_domain": str(ctx.resolved_solve_domain) if ctx.resolved_solve_domain is not None else None,
        "canonical_system_id": str(ctx.canonical_system_id) if ctx.canonical_system_id is not None else None,
        "dominant_round_decimals": int(ROUND_DECIMALS),
        "flatness_threshold": float(threshold),
        "flatness_min_finite_count": int(FLATNESS_MIN_FINITE_COUNT),
        "first_bad_stage": first_bad,
        "river_flatness_fail": first_bad is not None,
        "stages": stages,
    }
    path = _write_json(report_path, payload)
    if first_bad is not None:
        raise RuntimeError(
            "river_science_chain_flatness_fail:"
            f"first_bad_stage={first_bad.get('stage')}:"
            f"dominant_value={first_bad.get('dominant_rounded_value_m')}:"
            f"fraction={first_bad.get('dominant_rounded_fraction')}:"
            f"summary={path}"
        )
    return path


__all__ = ["write_river_science_chain_summary"]
