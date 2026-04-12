from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from river_v2_context import RiverV2Context
from river_v2_contract import RiverV2StageResult, STAGE_CENTERLINE_OFFSET_MODELED
from river_v2_receipts import build_river_v2_stage_receipt, write_river_v2_receipt

_REQUIRED_FIELDS = (
    "point_id",
    "station_m",
    "observed_offset_m",
    "offset_modeled_m",
    "offset_source",
    "offset_support_class",
    "geometry",
)
_MIN_OFFSET_M = 0.10
_MAX_PLACEHOLDER_OFFSET_M = 12.0
_GROUP_COLS = ("component_id", "levelpath_id", "reach_id", "source_reach_key")
_STREAM_ORDER_COLS = ("stream_order", "streamorde", "streamorder", "streamord", "strahler", "Strahler")
_WIDTH_COLS = ("mean_width_m", "channel_width_m", "width_m", "bankfull_width_m", "nhd_width_m")
_SLOPE_COLS = ("mean_slope", "slope", "slope_m_per_m", "slopelenkm")


def centerline_modeled_offset_field_schema() -> dict[str, str]:
    return {
        "point_id": "string",
        "station_m": "float64",
        "observed_offset_m": "float64",
        "offset_modeled_m": "float64",
        "offset_source": "string",
        "offset_confidence": "string",
        "offset_support_class": "string",
        "placeholder_basis": "string",
        "stream_order": "float64",
        "mean_width_m": "float64",
        "mean_slope": "float64",
        "geometry": "Point",
    }


def component_stream_summary_field_schema() -> dict[str, str]:
    return {
        "component_id": "string",
        "levelpath_id": "string",
        "reach_id": "string",
        "source_reach_key": "string",
        "component_length_m": "float64",
        "stream_order": "float64",
        "mean_width_m": "float64",
        "mean_slope": "float64",
        "observed_offset_count": "int64",
        "observed_offset_min_m": "float64",
        "observed_offset_median_m": "float64",
        "observed_offset_max_m": "float64",
        "placeholder_offset_m": "float64",
        "placeholder_basis": "string",
        "placeholder_confidence": "string",
        "geometry": "Point",
    }


def _read_gpkg(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise RuntimeError(f"river_v2_modeled_offset_missing_input:{path}")
    return gpd.read_file(path)


def _numeric(values: pd.Series | Any) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _group_columns(gdf: gpd.GeoDataFrame) -> list[str]:
    cols = [c for c in _GROUP_COLS if c in gdf.columns]
    return cols if cols else []


def _rolling_nanmean(values: np.ndarray, half_window: int = 1) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    for i in range(len(vals)):
        lo = max(0, i - int(half_window))
        hi = min(len(vals), i + int(half_window) + 1)
        window = vals[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[i] = float(np.nanmean(finite))
    return out


def _unique_anchor_series(station: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sta = np.asarray(station, dtype=float)
    vals = np.asarray(values, dtype=float)
    usable = np.isfinite(sta) & np.isfinite(vals)
    if np.count_nonzero(usable) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    sta_u = sta[usable]
    val_u = vals[usable]
    order = np.argsort(sta_u, kind="mergesort")
    sta_u = sta_u[order]
    val_u = val_u[order]
    unique_sta: list[float] = []
    unique_val: list[float] = []
    i = 0
    while i < len(sta_u):
        j = i + 1
        while j < len(sta_u) and abs(sta_u[j] - sta_u[i]) <= 1.0e-9:
            j += 1
        unique_sta.append(float(sta_u[i]))
        unique_val.append(float(np.nanmedian(val_u[i:j])))
        i = j
    return np.asarray(unique_sta, dtype=float), np.asarray(unique_val, dtype=float)


def _endpoint_slope(anchor_station: np.ndarray, anchor_values: np.ndarray, *, from_start: bool, slope_cap: float = 3.0e-4) -> float:
    if anchor_station.size < 2:
        return 0.0
    if from_start:
        x0, x1 = float(anchor_station[0]), float(anchor_station[1])
        y0, y1 = float(anchor_values[0]), float(anchor_values[1])
    else:
        x0, x1 = float(anchor_station[-2]), float(anchor_station[-1])
        y0, y1 = float(anchor_values[-2]), float(anchor_values[-1])
    dx = x1 - x0
    if not np.isfinite(dx) or abs(dx) < 1.0e-9:
        return 0.0
    slope = (y1 - y0) / dx
    return float(np.clip(slope, -slope_cap, slope_cap))


def _simple_modeled_series(station: np.ndarray, anchor_station: np.ndarray, anchor_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sta = np.asarray(station, dtype=float)
    offset_interp = np.full((sta.size,), np.nan, dtype=float)
    offset_continued = np.full((sta.size,), np.nan, dtype=float)
    modeled = np.full((sta.size,), np.nan, dtype=float)
    support = np.full((sta.size,), "unsupported_component", dtype=object)
    source = np.full((sta.size,), "unsupported_component_no_modeled_offset", dtype=object)
    finite_station = np.isfinite(sta)
    if anchor_station.size == 0:
        return offset_interp, offset_continued, modeled, support, source
    if anchor_station.size == 1:
        modeled[finite_station] = float(max(anchor_values[0], _MIN_OFFSET_M))
        offset_continued[finite_station] = modeled[finite_station]
        support[finite_station] = "inferred"
        source[finite_station] = "single_anchor_hold"
        return offset_interp, offset_continued, modeled, support, source

    offset_interp[finite_station] = np.interp(sta[finite_station], anchor_station, anchor_values)
    modeled[finite_station] = offset_interp[finite_station]

    first_sta = float(anchor_station[0])
    last_sta = float(anchor_station[-1])
    left_slope = _endpoint_slope(anchor_station, anchor_values, from_start=True)
    right_slope = _endpoint_slope(anchor_station, anchor_values, from_start=False)

    between_mask = finite_station & (sta >= first_sta) & (sta <= last_sta)
    if np.any(between_mask):
        support[between_mask] = "inferred"
        source[between_mask] = "linear_interpolation"

    upstream_mask = finite_station & (sta < first_sta)
    if np.any(upstream_mask):
        cont = float(anchor_values[0]) + left_slope * (sta[upstream_mask] - first_sta)
        cont = np.maximum(cont, _MIN_OFFSET_M)
        modeled[upstream_mask] = cont
        offset_continued[upstream_mask] = cont
        support[upstream_mask] = "inferred"
        source[upstream_mask] = "endpoint_continuation"

    downstream_mask = finite_station & (sta > last_sta)
    if np.any(downstream_mask):
        cont = float(anchor_values[-1]) + right_slope * (sta[downstream_mask] - last_sta)
        cont = np.maximum(cont, _MIN_OFFSET_M)
        modeled[downstream_mask] = cont
        offset_continued[downstream_mask] = cont
        support[downstream_mask] = "inferred"
        source[downstream_mask] = "endpoint_continuation"

    return offset_interp, offset_continued, modeled, support, source


def _join_inputs(wse_points_path: Path, observed_offset_points_path: Path) -> gpd.GeoDataFrame:
    wse = _read_gpkg(wse_points_path)
    observed = _read_gpkg(observed_offset_points_path)
    if "point_id" not in wse.columns or "point_id" not in observed.columns:
        raise RuntimeError("river_v2_modeled_offset_missing_point_id")
    wse = wse.copy()
    observed = observed.copy()
    wse["point_id"] = wse["point_id"].astype(str)
    observed["point_id"] = observed["point_id"].astype(str)
    keep_wse = [c for c in ("point_id", "station_m", *_GROUP_COLS, "wse_proxy_z_m", *_STREAM_ORDER_COLS, *_WIDTH_COLS, *_SLOPE_COLS, "geometry") if c in wse.columns]
    keep_observed = [c for c in ("point_id", "observed_offset_m") if c in observed.columns]
    merged = wse[keep_wse].merge(observed[keep_observed], on=["point_id"], how="left")
    if "station_m" not in merged.columns:
        raise RuntimeError("river_v2_modeled_offset_missing_station_m")
    if "wse_proxy_z_m" not in merged.columns:
        raise RuntimeError("river_v2_modeled_offset_missing_wse_proxy")
    merged["station_m"] = pd.to_numeric(merged["station_m"], errors="coerce")
    merged["wse_proxy_z_m"] = pd.to_numeric(merged["wse_proxy_z_m"], errors="coerce")
    if "observed_offset_m" in merged.columns:
        merged["observed_offset_m"] = pd.to_numeric(merged["observed_offset_m"], errors="coerce")
    sort_cols = [c for c in (*_group_columns(gpd.GeoDataFrame(merged, geometry="geometry", crs=wse.crs)), "station_m", "point_id") if c in merged.columns]
    merged = merged.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=wse.crs)


def _first_finite_from_columns(gdf: gpd.GeoDataFrame, columns: tuple[str, ...]) -> np.ndarray:
    out = np.full((len(gdf),), np.nan, dtype=float)
    for col in columns:
        if col in gdf.columns:
            vals = _numeric(gdf[col])
            take = ~np.isfinite(out) & np.isfinite(vals)
            out[take] = vals[take]
    return out


def _component_key(group: gpd.GeoDataFrame) -> dict[str, Any]:
    return {col: (None if group[col].isna().all() else group.iloc[0][col]) for col in _GROUP_COLS if col in group.columns}


def _component_geometry(group: gpd.GeoDataFrame):
    if len(group) == 0:
        return None
    try:
        union_method = getattr(group.geometry, "union_all", None)
        geom = union_method() if callable(union_method) else group.geometry.unary_union
        return geom.centroid
    except Exception:
        return group.geometry.iloc[0]


def _component_summary_for_group(group: gpd.GeoDataFrame) -> dict[str, Any]:
    station = _numeric(group["station_m"]) if "station_m" in group.columns else np.full((len(group),), np.nan)
    observed = _numeric(group["observed_offset_m"]) if "observed_offset_m" in group.columns else np.full((len(group),), np.nan)
    width_vals = _first_finite_from_columns(group, _WIDTH_COLS)
    slope_vals = _first_finite_from_columns(group, _SLOPE_COLS)
    stream_order_vals = _first_finite_from_columns(group, _STREAM_ORDER_COLS)
    summary = {
        **_component_key(group),
        "component_length_m": float(np.nanmax(station) - np.nanmin(station)) if np.count_nonzero(np.isfinite(station)) >= 2 else None,
        "stream_order": float(np.nanmedian(stream_order_vals)) if np.count_nonzero(np.isfinite(stream_order_vals)) else None,
        "mean_width_m": float(np.nanmedian(width_vals)) if np.count_nonzero(np.isfinite(width_vals)) else None,
        "mean_slope": float(np.nanmedian(slope_vals)) if np.count_nonzero(np.isfinite(slope_vals)) else None,
        "observed_offset_count": int(np.count_nonzero(np.isfinite(observed))),
        "observed_offset_min_m": float(np.nanmin(observed)) if np.count_nonzero(np.isfinite(observed)) else None,
        "observed_offset_median_m": float(np.nanmedian(observed)) if np.count_nonzero(np.isfinite(observed)) else None,
        "observed_offset_max_m": float(np.nanmax(observed)) if np.count_nonzero(np.isfinite(observed)) else None,
        "geometry": _component_geometry(group),
    }
    return summary


def _build_component_stream_summary(joined: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    group_cols = _group_columns(joined)
    if group_cols:
        groups = [grp for _, grp in joined.groupby(group_cols, dropna=False, sort=False)]
    else:
        groups = [joined]
    rows = [_component_summary_for_group(group) for group in groups]
    summary = gpd.GeoDataFrame(rows, geometry="geometry", crs=joined.crs)
    return summary.reset_index(drop=True)


def _placeholder_from_width(width_m: float) -> float:
    if width_m < 5.0:
        return 0.75
    if width_m < 15.0:
        return 1.5
    if width_m < 40.0:
        return 3.0
    return 5.0


def _placeholder_from_stream_order(stream_order: float) -> float:
    if stream_order <= 2.0:
        return 0.75
    if stream_order <= 4.0:
        return 1.75
    if stream_order <= 6.0:
        return 3.0
    return 4.5


def _assign_component_placeholder_offset(summary_row: pd.Series) -> tuple[float, str, str, str]:
    width_m = pd.to_numeric(summary_row.get("mean_width_m"), errors="coerce")
    stream_order = pd.to_numeric(summary_row.get("stream_order"), errors="coerce")
    mean_slope = pd.to_numeric(summary_row.get("mean_slope"), errors="coerce")
    if np.isfinite(width_m):
        depth = _placeholder_from_width(float(width_m))
        basis = "width"
    elif np.isfinite(stream_order):
        depth = _placeholder_from_stream_order(float(stream_order))
        basis = "stream_order"
    else:
        depth = 1.5
        basis = "coarse_default"
    slope_factor = 1.0
    if np.isfinite(mean_slope):
        slope_factor = float(np.clip(1.0 + np.clip(float(mean_slope), -0.02, 0.02) * 5.0, 0.9, 1.1))
        if basis == "width":
            basis = "width+slope"
        elif basis == "stream_order":
            basis = "stream_order+slope"
    depth = float(np.clip(depth * slope_factor, _MIN_OFFSET_M, _MAX_PLACEHOLDER_OFFSET_M))
    return depth, basis, "low", "nhdplus_placeholder"


def _build_modeled_offsets_for_group(group: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, Any], list[str]]:
    out = group.copy().sort_values([c for c in ("station_m", "point_id") if c in group.columns], kind="mergesort").reset_index(drop=True)
    warnings: list[str] = []
    station = _numeric(out["station_m"]) if "station_m" in out.columns else np.full((len(out),), np.nan)
    observed = _numeric(out["observed_offset_m"]) if "observed_offset_m" in out.columns else np.full((len(out),), np.nan)

    anchor_mask = np.isfinite(observed)
    negative_obs = int(np.count_nonzero(anchor_mask & (observed < 0.0)))
    if negative_obs > 0:
        warnings.append("modeled_offset_negative_observed_offsets_clamped")
    observed = np.where(anchor_mask, np.maximum(observed, _MIN_OFFSET_M), np.nan)
    anchor_mask = np.isfinite(observed)
    anchor_count = int(np.count_nonzero(anchor_mask))

    if anchor_count >= 1:
        anchor_station, anchor_values = _unique_anchor_series(station[anchor_mask], observed[anchor_mask])
        _, _, modeled, support_class, source = _simple_modeled_series(station, anchor_station, anchor_values)
        smooth = _rolling_nanmean(modeled, half_window=1)
        smooth_mask = np.isfinite(smooth) & (~anchor_mask)
        modeled[smooth_mask] = smooth[smooth_mask]
        offset_confidence = np.where(anchor_mask, "high", "medium").astype(object)
        placeholder_basis = np.full((len(out),), "", dtype=object)
    else:
        warnings.append("modeled_offset_component_has_no_anchor_support")
        summary_row = pd.Series(_component_summary_for_group(out))
        placeholder_offset_m, placeholder_basis_value, placeholder_confidence_value, placeholder_source = _assign_component_placeholder_offset(summary_row)
        modeled = np.full((len(out),), placeholder_offset_m, dtype=float)
        support_class = np.full((len(out),), "inferred", dtype=object)
        source = np.full((len(out),), placeholder_source, dtype=object)
        offset_confidence = np.full((len(out),), placeholder_confidence_value, dtype=object)
        placeholder_basis = np.full((len(out),), placeholder_basis_value, dtype=object)

    positive_mask = np.isfinite(modeled)
    modeled[positive_mask] = np.maximum(modeled[positive_mask], _MIN_OFFSET_M)
    modeled[anchor_mask] = observed[anchor_mask]
    source[anchor_mask] = "observed_offset_anchor"
    support_class[anchor_mask] = "observed_anchor"
    if anchor_count >= 1:
        source[(~anchor_mask) & (source == "linear_interpolation")] = "observed_interpolated"
        source[(~anchor_mask) & (source == "endpoint_continuation")] = "observed_smoothed"
        source[(~anchor_mask) & (source == "single_anchor_hold")] = "observed_smoothed"

    out["observed_offset_m"] = observed
    out["offset_modeled_m"] = modeled
    out["offset_source"] = source.astype(str)
    out["offset_confidence"] = offset_confidence.astype(str)
    out["offset_support_class"] = support_class.astype(str)
    out["placeholder_basis"] = placeholder_basis.astype(str)
    out["stream_order"] = _first_finite_from_columns(out, _STREAM_ORDER_COLS)
    out["mean_width_m"] = _first_finite_from_columns(out, _WIDTH_COLS)
    out["mean_slope"] = _first_finite_from_columns(out, _SLOPE_COLS)

    diagnostics = {
        "record_count": int(len(out)),
        "anchor_count": anchor_count,
        "finite_modeled_count": int(np.count_nonzero(np.isfinite(modeled))),
        "median_modeled_offset_m": float(np.nanmedian(modeled)) if np.count_nonzero(np.isfinite(modeled)) else None,
        "inferred_count": int(np.count_nonzero(out["offset_support_class"].astype(str) == "inferred")),
        "unsupported_component_count": int(np.count_nonzero(out["offset_support_class"].astype(str) == "unsupported_component")),
        "placeholder_modeled_point_count": int(np.count_nonzero(out["offset_source"].astype(str) == "nhdplus_placeholder")),
    }
    return out, diagnostics, warnings


def build_modeled_offsets(joined: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[str], dict[str, Any]]:
    component_summary = _build_component_stream_summary(joined)
    group_cols = _group_columns(joined)
    frames: list[gpd.GeoDataFrame] = []
    warnings: list[str] = []
    diagnostics_list: list[dict[str, Any]] = []
    group_iter = [(None, joined)] if not group_cols else joined.groupby(group_cols, dropna=False, sort=False)
    for _, group in group_iter:
        modeled_group, group_diag, group_warnings = _build_modeled_offsets_for_group(group)
        frames.append(modeled_group)
        diagnostics_list.append(group_diag)
        warnings.extend(group_warnings)
    plain_frames = []
    for frame in frames:
        plain = pd.DataFrame(frame.drop(columns="geometry")).copy()
        plain["geometry"] = list(frame.geometry)
        plain_frames.append(plain)
    out = pd.concat(plain_frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=str(joined.crs) if joined.crs is not None else None)
    sort_cols = [c for c in (*group_cols, "station_m", "point_id") if c in out.columns]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    keep = [c for c in ("point_id", "station_m", *_GROUP_COLS, "observed_offset_m", "offset_modeled_m", "offset_source", "offset_confidence", "offset_support_class", "placeholder_basis", "stream_order", "mean_width_m", "mean_slope", "geometry") if c in out.columns]
    out = gpd.GeoDataFrame(out[keep].copy(), geometry="geometry", crs=out.crs)

    if len(component_summary):
        placeholder_info = []
        for _, row in component_summary.iterrows():
            offset_m, basis, confidence, _ = _assign_component_placeholder_offset(row)
            placeholder_info.append((offset_m, basis, confidence))
        component_summary = component_summary.copy()
        component_summary["placeholder_offset_m"] = [v[0] for v in placeholder_info]
        component_summary["placeholder_basis"] = [v[1] for v in placeholder_info]
        component_summary["placeholder_confidence"] = [v[2] for v in placeholder_info]

    diagnostics = {
        "record_count": int(len(out)),
        "group_count": int(len(diagnostics_list)),
        "anchor_count": int(sum(d["anchor_count"] for d in diagnostics_list)),
        "finite_modeled_count": int(sum(d["finite_modeled_count"] for d in diagnostics_list)),
        "median_modeled_offset_m": float(np.nanmedian(_numeric(out["offset_modeled_m"]))) if len(out) else None,
        "inferred_count": int(sum(d.get("inferred_count", 0) for d in diagnostics_list)),
        "unsupported_component_count": int(sum(d.get("unsupported_component_count", 0) for d in diagnostics_list)),
        "placeholder_modeled_point_count": int(sum(d.get("placeholder_modeled_point_count", 0) for d in diagnostics_list)),
        "supported_component_count": int(np.count_nonzero(pd.to_numeric(component_summary.get("observed_offset_count"), errors="coerce").to_numpy(dtype=float) > 0.0)) if len(component_summary) else 0,
        "unsupported_component_count_from_summary": int(np.count_nonzero(pd.to_numeric(component_summary.get("observed_offset_count"), errors="coerce").to_numpy(dtype=float) <= 0.0)) if len(component_summary) else 0,
    }
    return out, component_summary, sorted(set(warnings)), diagnostics


def validate_modeled_offsets(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    missing = [f for f in _REQUIRED_FIELDS if f not in gdf.columns]
    recs = int(len(gdf))
    station = _numeric(gdf["station_m"]) if "station_m" in gdf.columns else np.full((recs,), np.nan)
    modeled = _numeric(gdf["offset_modeled_m"]) if "offset_modeled_m" in gdf.columns else np.full((recs,), np.nan)
    source = gdf["offset_source"].astype(str).to_numpy(dtype=object) if "offset_source" in gdf.columns else np.full((recs,), "", dtype=object)
    placeholder_mask = source == "nhdplus_placeholder"
    geometry_valid = bool(getattr(gdf, "geometry", None) is not None and gdf.geometry.notna().all() and gdf.is_valid.all()) if recs > 0 else True
    station_monotonic_by_group = True
    group_cols = _group_columns(gdf)
    for _, grp in ([(None, gdf)] if not group_cols else gdf.groupby(group_cols, dropna=False, sort=False)):
        sta = _numeric(grp.sort_values([c for c in ("station_m", "point_id") if c in grp.columns], kind="mergesort")["station_m"])
        finite_sta = sta[np.isfinite(sta)]
        if finite_sta.size != len(grp) or (finite_sta.size > 1 and np.any(np.diff(finite_sta) < -1.0e-9)):
            station_monotonic_by_group = False
            break
    finite_modeled_ok = bool(np.all(np.isfinite(modeled))) if recs > 0 else True
    positive_modeled_ok = bool(np.all(modeled[np.isfinite(modeled)] > 0.0)) if recs > 0 else True
    placeholder_positive_ok = bool(np.all(modeled[placeholder_mask] > 0.0)) if np.any(placeholder_mask) else True
    source_nonempty_ok = bool(np.all(pd.Series(source).str.len().to_numpy(dtype=int) > 0)) if recs > 0 else True
    return {
        "valid": not missing and recs > 0 and geometry_valid and finite_modeled_ok and positive_modeled_ok and placeholder_positive_ok and source_nonempty_ok and np.count_nonzero(np.isfinite(station)) == recs and station_monotonic_by_group,
        "record_count": recs,
        "missing_required_fields": missing,
        "finite_modeled_offset_count": int(np.count_nonzero(np.isfinite(modeled))),
        "required_finite_modeled_offset_count": int(np.count_nonzero(np.isfinite(modeled))),
        "required_record_count": recs,
        "unsupported_component_count": int(np.count_nonzero(gdf["offset_support_class"].astype(str).to_numpy(dtype=object) == "unsupported_component")) if "offset_support_class" in gdf.columns else 0,
        "placeholder_modeled_point_count": int(np.count_nonzero(placeholder_mask)),
        "min_modeled_offset_m": float(np.nanmin(modeled)) if np.count_nonzero(np.isfinite(modeled)) else None,
        "max_modeled_offset_m": float(np.nanmax(modeled)) if np.count_nonzero(np.isfinite(modeled)) else None,
        "geometry_valid": geometry_valid,
        "station_monotonic_by_group": bool(station_monotonic_by_group),
        "source_nonempty_ok": bool(source_nonempty_ok),
    }


def run_modeled_offset_stage(
    ctx: RiverV2Context,
    *,
    wse_points_path: Path,
    observed_offset_points_path: Path,
) -> RiverV2StageResult:
    joined = _join_inputs(wse_points_path, observed_offset_points_path)
    modeled, component_summary, warnings, diagnostics = build_modeled_offsets(joined)
    validation = validate_modeled_offsets(modeled)
    if int(validation.get("finite_modeled_offset_count", 0)) == 0:
        raise RuntimeError(f"river_v2_modeled_offset_no_finite_values:{diagnostics}")
    if not validation.get("valid"):
        raise RuntimeError(f"river_v2_modeled_offset_invalid:{validation}")

    out_path = ctx.paths.centerline_offset_modeled_points
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    modeled.to_file(out_path, driver="GPKG")

    component_summary_path = ctx.paths.component_stream_summary
    if component_summary_path.exists():
        component_summary_path.unlink()
    component_summary.to_file(component_summary_path, driver="GPKG")

    component_summary_receipt = build_river_v2_stage_receipt(
        stage_id=f"{STAGE_CENTERLINE_OFFSET_MODELED}_component_summary",
        output_artifact=str(component_summary_path),
        input_artifacts=ctx.direct_stage_input_artifacts(wse_points_path, observed_offset_points_path),
        record_count=int(len(component_summary)),
        field_schema=component_stream_summary_field_schema(),
        vertical_reference=ctx.vertical_reference,
        warnings=[],
        source_logic="component_summary_from_centerline_wse_attributes_and_observed_offset_counts",
        validation={
            "record_count": int(len(component_summary)),
            "supported_component_count": int(np.count_nonzero(pd.to_numeric(component_summary.get("observed_offset_count"), errors="coerce").to_numpy(dtype=float) > 0.0)) if len(component_summary) else 0,
            "unsupported_component_count": int(np.count_nonzero(pd.to_numeric(component_summary.get("observed_offset_count"), errors="coerce").to_numpy(dtype=float) <= 0.0)) if len(component_summary) else 0,
        },
    )
    component_summary_receipt_path = write_river_v2_receipt(component_summary_receipt, ctx.paths.component_stream_summary_receipt)

    receipt = build_river_v2_stage_receipt(
        stage_id=STAGE_CENTERLINE_OFFSET_MODELED,
        output_artifact=str(out_path),
        input_artifacts=ctx.direct_stage_input_artifacts(wse_points_path, observed_offset_points_path),
        record_count=int(len(modeled)),
        field_schema=centerline_modeled_offset_field_schema(),
        vertical_reference=ctx.vertical_reference,
        warnings=warnings,
        source_logic="modeled_offset_from_observed_offsets_or_explicit_nhdplus_placeholder_depth_below_wse_never_dem_bed_proxy",
        validation={**validation, **diagnostics},
        extra_artifacts={
            "component_stream_summary": str(component_summary_path),
            "component_stream_summary_receipt": str(component_summary_receipt_path),
        },
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.centerline_offset_modeled_points_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_CENTERLINE_OFFSET_MODELED,
        output_artifact=out_path,
        receipt_path=receipt_path,
        record_count=int(len(modeled)),
        validation={**validation, **diagnostics},
        warnings=warnings,
        aux_outputs={
            "river_v2_component_stream_summary": str(component_summary_path),
            "river_v2_component_stream_summary_receipt": str(component_summary_receipt_path),
        },
    )
