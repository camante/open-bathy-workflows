from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from river_v2_context import RiverV2Context
from river_v2_contract import RiverV2StageResult, STAGE_CENTERLINE_BED_BACKBONE
from river_v2_receipts import build_river_v2_stage_receipt, write_river_v2_receipt

_REQUIRED_FIELDS = (
    "point_id",
    "station_m",
    "wse_proxy_z_m",
    "offset_modeled_m",
    "bed_backbone_z_m",
    "geometry",
)
_GROUP_COLS = ("component_id", "levelpath_id", "reach_id", "source_reach_key")


def _numeric(values: pd.Series | Any) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _read_gpkg(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise RuntimeError(f"river_v2_backbone_missing_input:{path}")
    return gpd.read_file(path)


def _normalize_join_keys(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if "point_id" not in out.columns:
        raise RuntimeError("river_v2_backbone_missing_point_id")
    out["point_id"] = out["point_id"].astype(str)
    if "station_m" in out.columns:
        out["station_m"] = pd.to_numeric(out["station_m"], errors="coerce")
    return out


def _merge_inputs(wse_points_path: Path, modeled_offset_points_path: Path) -> gpd.GeoDataFrame:
    wse = _normalize_join_keys(_read_gpkg(wse_points_path))
    modeled = _normalize_join_keys(_read_gpkg(modeled_offset_points_path))
    wse_cols = [c for c in ("point_id", "station_m", "wse_proxy_z_m", "geometry", *_GROUP_COLS) if c in wse.columns]
    modeled_cols = [c for c in ("point_id", "offset_modeled_m", "observed_offset_m", "offset_support_class", *_GROUP_COLS) if c in modeled.columns]
    merged = wse[wse_cols].merge(modeled[modeled_cols], on=["point_id"], how="inner", suffixes=("", "_modeled"))
    if len(merged) == 0:
        raise RuntimeError("river_v2_backbone_no_joined_points")
    for col in ("wse_proxy_z_m", "offset_modeled_m", "observed_offset_m"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    if "offset_support_class" in merged.columns:
        merged["offset_support_class"] = merged["offset_support_class"].astype(str)
    merged["authoritative_anchor"] = False
    merged["authoritative_bed_z_m"] = np.nan
    merged["bed_backbone_raw_z_m"] = pd.to_numeric(merged["wse_proxy_z_m"], errors="coerce") - pd.to_numeric(merged["offset_modeled_m"], errors="coerce")
    merged["bed_backbone_z_m"] = merged["bed_backbone_raw_z_m"]
    merged["bed_backbone_adjustment_m"] = 0.0
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=wse.crs)
    sort_cols = [c for c in (*_GROUP_COLS, "station_m", "point_id") if c in merged.columns]
    merged = merged.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return merged


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


def _enforce_nonincreasing_forward(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for i in range(1, len(out)):
        if np.isfinite(out[i - 1]) and np.isfinite(out[i]):
            out[i] = min(out[i], out[i - 1])
    return out


def _enforce_nonincreasing_backward_to_anchor(values: np.ndarray, anchor_value: float) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    downstream = float(anchor_value)
    for i in range(len(out) - 1, -1, -1):
        if np.isfinite(out[i]):
            out[i] = max(out[i], downstream)
            downstream = out[i]
    return out


def _interpolate_between_anchor_values(stations: np.ndarray, start_station: float, start_value: float, end_station: float, end_value: float) -> np.ndarray:
    if len(stations) == 0:
        return np.array([], dtype=float)
    if not np.isfinite(start_station) or not np.isfinite(end_station) or abs(end_station - start_station) <= 1.0e-9:
        return np.full((len(stations),), float(min(start_value, end_value)), dtype=float)
    frac = (np.asarray(stations, dtype=float) - float(start_station)) / (float(end_station) - float(start_station))
    frac = np.clip(frac, 0.0, 1.0)
    return (float(start_value) + frac * (float(end_value) - float(start_value))).astype(float)


def _smooth_group_backbone(group: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = group.sort_values([c for c in ("station_m", "point_id") if c in group.columns], kind="mergesort").reset_index(drop=True).copy()
    raw = _numeric(work["bed_backbone_raw_z_m"])
    stations = _numeric(work["station_m"]) if "station_m" in work.columns else np.arange(len(work), dtype=float)
    anchor_mask = np.zeros((len(work),), dtype=bool)
    if "offset_support_class" in work.columns:
        anchor_mask |= work["offset_support_class"].astype(str).eq("observed_anchor").to_numpy(dtype=bool)
    if "authoritative_anchor" in work.columns:
        anchor_mask |= work["authoritative_anchor"].to_numpy(dtype=bool)
    smoothed = raw.copy()
    infer_mask = np.isfinite(raw) & (~anchor_mask)
    if np.count_nonzero(infer_mask) > 0:
        smooth_all = _rolling_nanmean(raw, half_window=1)
        smoothed[infer_mask] = smooth_all[infer_mask]
    projected = smoothed.copy()
    anchor_idx = np.flatnonzero(anchor_mask & np.isfinite(raw))
    if anchor_idx.size == 0:
        projected = _enforce_nonincreasing_forward(projected)
    else:
        projected[anchor_idx] = raw[anchor_idx]
        first_anchor = int(anchor_idx[0])
        if first_anchor > 0:
            projected[:first_anchor] = _enforce_nonincreasing_backward_to_anchor(projected[:first_anchor], raw[first_anchor])
        for start_idx, end_idx in zip(anchor_idx[:-1], anchor_idx[1:]):
            lo = int(start_idx) + 1
            hi = int(end_idx)
            if hi <= lo:
                continue
            projected[lo:hi] = _interpolate_between_anchor_values(
                stations[lo:hi],
                stations[int(start_idx)],
                raw[int(start_idx)],
                stations[int(end_idx)],
                raw[int(end_idx)],
            )
        last_anchor = int(anchor_idx[-1])
        if last_anchor < len(projected) - 1:
            tail = projected[last_anchor + 1 :].copy()
            tail = _enforce_nonincreasing_forward(np.r_[raw[last_anchor], tail])[1:]
            projected[last_anchor + 1 :] = tail
        projected[anchor_idx] = raw[anchor_idx]
    work["authoritative_anchor"] = anchor_mask
    work["authoritative_bed_z_m"] = np.where(anchor_mask, raw, np.nan)
    work["bed_backbone_z_m"] = projected
    work["bed_backbone_adjustment_m"] = work["bed_backbone_z_m"] - work["bed_backbone_raw_z_m"]
    if "offset_support_class" in work.columns:
        support = work["offset_support_class"].astype(str)
        work["backbone_support_class"] = np.where(support.eq("observed_anchor"), "observed_anchor", "inferred")
    else:
        work["backbone_support_class"] = np.where(anchor_mask, "observed_anchor", "inferred")
    return work

def build_backbone(wse_points_path: Path, modeled_offset_points_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, Any], list[str]]:
    merged = _merge_inputs(wse_points_path, modeled_offset_points_path)
    warnings: list[str] = []
    dropped_unsupported_component_count = 0
    if "offset_support_class" in merged.columns:
        unsupported_mask = merged["offset_support_class"].astype(str).eq("unsupported_component")
        if bool(np.any(unsupported_mask)):
            dropped_unsupported_component_count = int(np.count_nonzero(unsupported_mask))
            warnings.append("backbone_dropped_unsupported_component_points")
            merged = merged.loc[~unsupported_mask].copy()
    if len(merged) == 0:
        raise RuntimeError("river_v2_backbone_no_supported_modeled_offset_points")
    group_cols = _group_columns(merged)
    frames: list[gpd.GeoDataFrame] = []
    input_record_count = int(len(merged))
    group_iter = [(None, merged)] if not group_cols else merged.groupby(group_cols, dropna=False, sort=False)
    for _, group in group_iter:
        if len(group) == 0:
            continue
        frames.append(_smooth_group_backbone(group))
    if not frames:
        raise RuntimeError("river_v2_backbone_no_finite_backbone_points")
    plain_frames = []
    for frame in frames:
        plain = pd.DataFrame(frame.drop(columns="geometry")).copy()
        plain["geometry"] = list(frame.geometry)
        plain_frames.append(plain)
    backbone = pd.concat(plain_frames, ignore_index=True)
    keep = [c for c in (
        "point_id", "station_m", *_GROUP_COLS, "wse_proxy_z_m", "offset_modeled_m", "offset_support_class", "authoritative_bed_z_m",
        "bed_backbone_raw_z_m", "bed_backbone_z_m", "bed_backbone_adjustment_m", "authoritative_anchor", "backbone_support_class", "geometry"
    ) if c in backbone.columns]
    backbone = gpd.GeoDataFrame(backbone[keep].copy(), geometry="geometry", crs=merged.crs)
    sort_cols = [c for c in (*group_cols, "station_m", "point_id") if c in backbone.columns]
    backbone = backbone.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    adjustments = _numeric(backbone["bed_backbone_adjustment_m"]) if "bed_backbone_adjustment_m" in backbone.columns else np.full((len(backbone),), np.nan)
    diagnostics = {
        "input_record_count": input_record_count,
        "record_count": int(len(backbone)),
        "sparse_record_count": int(len(backbone)),
        "finite_backbone_count": int(np.count_nonzero(np.isfinite(_numeric(backbone["bed_backbone_z_m"])))),
        "median_backbone_z_m": float(np.nanmedian(_numeric(backbone["bed_backbone_z_m"]))) if len(backbone) else None,
        "max_abs_backbone_adjustment_m": float(np.nanmax(np.abs(adjustments))) if np.isfinite(adjustments).any() else 0.0,
        "authoritative_anchor_count": int(np.count_nonzero(backbone["authoritative_anchor"].to_numpy(dtype=bool))) if "authoritative_anchor" in backbone.columns else 0,
        "inferred_point_count": int(np.count_nonzero(~backbone["authoritative_anchor"].to_numpy(dtype=bool))) if "authoritative_anchor" in backbone.columns else int(len(backbone)),
        "dropped_unsupported_component_point_count": int(dropped_unsupported_component_count),
    }
    return backbone, diagnostics, warnings


def validate_backbone(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    missing = [f for f in _REQUIRED_FIELDS if f not in gdf.columns]
    recs = int(len(gdf))
    backbone = _numeric(gdf["bed_backbone_z_m"]) if "bed_backbone_z_m" in gdf.columns else np.full((recs,), np.nan)
    geometry_valid = bool(getattr(gdf, "geometry", None) is not None and gdf.geometry.notna().all() and gdf.is_valid.all()) if recs > 0 else True
    anchor = gdf["authoritative_anchor"].to_numpy(dtype=bool) if "authoritative_anchor" in gdf.columns else np.zeros((recs,), dtype=bool)
    group_cols = _group_columns(gdf)
    station_monotonic = True
    for _, grp in ([(None, gdf)] if not group_cols else gdf.groupby(group_cols, dropna=False, sort=False)):
        s = _numeric(grp.sort_values([c for c in ("station_m", "point_id") if c in grp.columns], kind="mergesort")["station_m"])
        sf = s[np.isfinite(s)]
        if sf.size > 1 and np.any(np.diff(sf) < -1.0e-9):
            station_monotonic = False
            break
    diffs = np.diff(backbone[np.isfinite(backbone)]) if np.count_nonzero(np.isfinite(backbone)) > 1 else np.array([], dtype=float)
    return {
        "valid": not missing and recs > 0 and geometry_valid and np.count_nonzero(np.isfinite(backbone)) == recs,
        "record_count": recs,
        "missing_required_fields": missing,
        "finite_backbone_count": int(np.count_nonzero(np.isfinite(backbone))),
        "geometry_valid": geometry_valid,
        "authoritative_anchor_count": int(np.count_nonzero(anchor)),
        "inferred_point_count": int(np.count_nonzero(~anchor)),
        "max_downstream_rise_m": float(max(0.0, -np.min(diffs))) if diffs.size else 0.0,
        "station_monotonic": bool(station_monotonic),
    }


def run_backbone_stage(
    ctx: RiverV2Context,
    *,
    wse_points_path: Path,
    modeled_offset_points_path: Path,
) -> RiverV2StageResult:
    backbone_gdf, diagnostics, warnings = build_backbone(
        Path(wse_points_path), Path(modeled_offset_points_path)
    )
    out_path = ctx.paths.river_centerline_bed_backbone_points
    out_path.parent.mkdir(parents=True, exist_ok=True)
    persist_cols = [c for c in ("point_id", "station_m", *_GROUP_COLS, "bed_backbone_z_m", "backbone_support_class", "geometry") if c in backbone_gdf.columns]
    persisted_gdf = gpd.GeoDataFrame(backbone_gdf[persist_cols].copy(), geometry="geometry", crs=backbone_gdf.crs)
    persisted_gdf.to_file(out_path, driver="GPKG")
    validation = validate_backbone(persisted_gdf)
    receipt = build_river_v2_stage_receipt(
        stage_id=STAGE_CENTERLINE_BED_BACKBONE,
        output_artifact=str(out_path),
        input_artifacts=ctx.direct_stage_input_artifacts(wse_points_path, modeled_offset_points_path),
        record_count=int(len(backbone_gdf)),
        field_schema={
            "point_id": "string",
            "station_m": "float64",
            "wse_proxy_z_m": "float64",
            "offset_modeled_m": "float64",
            "bed_backbone_raw_z_m": "float64",
            "bed_backbone_z_m": "float64",
            "bed_backbone_adjustment_m": "float64",
            "backbone_support_class": "string",
            "geometry": "Point",
        },
        vertical_reference=ctx.vertical_reference,
        warnings=warnings,
        source_logic="wse_proxy_minus_modeled_offset_then_preserve_observed_anchors_and_apply_light_monotone_smoothing_to_inferred_spans_to_form_sparse_backbone",
        validation={**validation, **{f"diagnostic_{k}": v for k, v in diagnostics.items()}},
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.river_centerline_bed_backbone_points_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_CENTERLINE_BED_BACKBONE,
        output_artifact=out_path,
        receipt_path=receipt_path,
        record_count=int(len(backbone_gdf)),
        validation=validation,
        warnings=warnings,
        aux_outputs={},
    )
