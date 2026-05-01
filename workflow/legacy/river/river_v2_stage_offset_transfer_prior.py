from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from legacy.river.river_v2_context import RiverV2Context
from legacy.river.river_v2_contract import RiverV2StageResult, STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS
from legacy.river.river_v2_receipts import build_river_v2_stage_receipt, write_river_v2_receipt

_GROUP_COLS = ("component_id", "levelpath_id", "reach_id", "source_reach_key")
_STREAM_ORDER_COLS = ("stream_order", "streamorde", "streamorder", "streamord", "strahler", "Strahler")
_WIDTH_COLS = ("mean_width_m", "channel_width_m", "width_m", "bankfull_width_m", "nhd_width_m")
_SLOPE_COLS = ("mean_slope", "slope", "slope_m_per_m", "slopelenkm")
_REQUIRED_FIELDS = (
    "component_id",
    "prior_basis",
    "prior_confidence",
    "prior_source_component_id",
    "prior_median_offset_m",
    "geometry",
)


def _read_gpkg(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise RuntimeError(f"river_v2_transfer_prior_missing_input:{path}")
    return gpd.read_file(path)


def _numeric(values: pd.Series | Any) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _first_finite_from_columns(gdf: gpd.GeoDataFrame, columns: tuple[str, ...]) -> np.ndarray:
    out = np.full((len(gdf),), np.nan, dtype=float)
    for col in columns:
        if col in gdf.columns:
            vals = _numeric(gdf[col])
            take = ~np.isfinite(out) & np.isfinite(vals)
            out[take] = vals[take]
    return out


def component_offset_transfer_prior_field_schema() -> dict[str, str]:
    return {
        "component_id": "string",
        "levelpath_id": "string",
        "reach_id": "string",
        "source_reach_key": "string",
        "observed_offset_count": "int64",
        "prior_basis": "string",
        "prior_confidence": "string",
        "prior_source_component_id": "string",
        "prior_source_levelpath_id": "string",
        "prior_anchor_count": "int64",
        "prior_distance_component_steps": "int64",
        "prior_upstream_offset_m": "float64",
        "prior_downstream_offset_m": "float64",
        "prior_median_offset_m": "float64",
        "prior_slope_m_per_m": "float64",
        "stream_order": "float64",
        "mean_width_m": "float64",
        "mean_slope": "float64",
        "geometry": "Point",
    }


def _component_geometry(group: gpd.GeoDataFrame):
    if len(group) == 0:
        return None
    try:
        union_method = getattr(group.geometry, "union_all", None)
        geom = union_method() if callable(union_method) else group.geometry.unary_union
        return geom.centroid
    except Exception:
        return group.geometry.iloc[0]


def _component_summary(joined: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [c for c in _GROUP_COLS if c in joined.columns]
    group_iter = [(None, joined)] if not group_cols else joined.groupby(group_cols, dropna=False, sort=False)
    for _, group in group_iter:
        grp = group.sort_values([c for c in ("station_m", "point_id") if c in group.columns], kind="mergesort").reset_index(drop=True)
        observed = _numeric(grp["observed_offset_m"]) if "observed_offset_m" in grp.columns else np.full((len(grp),), np.nan)
        station = _numeric(grp["station_m"]) if "station_m" in grp.columns else np.full((len(grp),), np.nan)
        mask = np.isfinite(observed)
        obs_count = int(np.count_nonzero(mask))
        if obs_count > 0:
            obs_sta = station[mask]
            obs_val = observed[mask]
            order = np.argsort(obs_sta, kind="mergesort") if obs_sta.size else np.asarray([], dtype=int)
            obs_sta = obs_sta[order] if obs_sta.size else obs_sta
            obs_val = obs_val[order] if obs_val.size else obs_val
            upstream_offset = float(obs_val[0])
            downstream_offset = float(obs_val[-1])
            median_offset = float(np.nanmedian(obs_val))
            slope = 0.0
            if obs_sta.size >= 2 and np.isfinite(obs_sta[-1] - obs_sta[0]) and abs(obs_sta[-1] - obs_sta[0]) > 1.0e-9:
                slope = float(np.clip((obs_val[-1] - obs_val[0]) / (obs_sta[-1] - obs_sta[0]), -3.0e-4, 3.0e-4))
        else:
            upstream_offset = np.nan
            downstream_offset = np.nan
            median_offset = np.nan
            slope = np.nan
        row = {
            "component_id": str(grp.iloc[0].get("component_id", "")) if len(grp) else "",
            "levelpath_id": str(grp.iloc[0].get("levelpath_id", "")) if "levelpath_id" in grp.columns and len(grp) else "",
            "reach_id": str(grp.iloc[0].get("reach_id", "")) if "reach_id" in grp.columns and len(grp) else "",
            "source_reach_key": str(grp.iloc[0].get("source_reach_key", "")) if "source_reach_key" in grp.columns and len(grp) else "",
            "observed_offset_count": obs_count,
            "observed_offset_min_m": float(np.nanmin(observed)) if obs_count > 0 else np.nan,
            "observed_offset_max_m": float(np.nanmax(observed)) if obs_count > 0 else np.nan,
            "observed_offset_median_m": median_offset,
            "component_station_min_m": float(np.nanmin(station)) if np.count_nonzero(np.isfinite(station)) else np.nan,
            "component_station_max_m": float(np.nanmax(station)) if np.count_nonzero(np.isfinite(station)) else np.nan,
            "upstream_anchor_offset_m": upstream_offset,
            "downstream_anchor_offset_m": downstream_offset,
            "anchor_trend_slope_m_per_m": slope,
            "stream_order": float(_first_finite_from_columns(grp, _STREAM_ORDER_COLS)[0]) if len(grp) else np.nan,
            "mean_width_m": float(_first_finite_from_columns(grp, _WIDTH_COLS)[0]) if len(grp) else np.nan,
            "mean_slope": float(_first_finite_from_columns(grp, _SLOPE_COLS)[0]) if len(grp) else np.nan,
            "geometry": _component_geometry(grp),
        }
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=joined.crs).reset_index(drop=True)


def _find_best_donor(summary: gpd.GeoDataFrame, idx: int) -> tuple[pd.Series | None, str, int]:
    target = summary.iloc[idx]
    supported = summary[pd.to_numeric(summary["observed_offset_count"], errors="coerce") > 0].copy()
    if supported.empty:
        return None, "no_observed_support_available", -1
    target_comp = str(target.get("component_id", "") or "")
    target_levelpath = str(target.get("levelpath_id", "") or "")
    target_center = float(np.nanmean([pd.to_numeric(target.get("component_station_min_m"), errors="coerce"), pd.to_numeric(target.get("component_station_max_m"), errors="coerce")]))

    same_levelpath = supported[supported["levelpath_id"].astype(str) == target_levelpath].copy()
    same_levelpath = same_levelpath[same_levelpath["component_id"].astype(str) != target_comp]
    if not same_levelpath.empty:
        centers = 0.5 * (pd.to_numeric(same_levelpath["component_station_min_m"], errors="coerce") + pd.to_numeric(same_levelpath["component_station_max_m"], errors="coerce"))
        if np.isfinite(target_center):
            dist = np.abs(centers.to_numpy(dtype=float) - target_center)
            best_idx = int(np.nanargmin(dist))
            return same_levelpath.iloc[best_idx], "same_levelpath_observed_component", int(best_idx)
        return same_levelpath.iloc[0], "same_levelpath_observed_component", 0

    others = supported[supported["component_id"].astype(str) != target_comp].copy()
    if others.empty:
        return None, "no_observed_support_available", -1
    width_target = pd.to_numeric(target.get("mean_width_m"), errors="coerce")
    order_target = pd.to_numeric(target.get("stream_order"), errors="coerce")
    width_other = pd.to_numeric(others["mean_width_m"], errors="coerce").to_numpy(dtype=float)
    order_other = pd.to_numeric(others["stream_order"], errors="coerce").to_numpy(dtype=float)
    score = np.zeros((len(others),), dtype=float)
    if np.isfinite(width_target):
        score += np.abs(width_other - float(width_target))
    if np.isfinite(order_target):
        score += 5.0 * np.abs(order_other - float(order_target))
    score += 10.0 / np.maximum(pd.to_numeric(others["observed_offset_count"], errors="coerce").to_numpy(dtype=float), 1.0)
    best_idx = int(np.nanargmin(score))
    return others.iloc[best_idx], "nearest_supported_component", best_idx


def build_offset_transfer_priors(joined: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list[str], dict[str, Any]]:
    summary = _component_summary(joined)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    for idx in range(len(summary)):
        row = summary.iloc[idx]
        observed_count = int(pd.to_numeric(row.get("observed_offset_count"), errors="coerce") or 0)
        if observed_count > 0:
            prior = {
                **row.to_dict(),
                "prior_basis": "local_observed_component",
                "prior_confidence": "high",
                "prior_source_component_id": str(row.get("component_id", "") or ""),
                "prior_source_levelpath_id": str(row.get("levelpath_id", "") or ""),
                "prior_anchor_count": observed_count,
                "prior_distance_component_steps": 0,
                "prior_upstream_offset_m": pd.to_numeric(row.get("upstream_anchor_offset_m"), errors="coerce"),
                "prior_downstream_offset_m": pd.to_numeric(row.get("downstream_anchor_offset_m"), errors="coerce"),
                "prior_median_offset_m": pd.to_numeric(row.get("observed_offset_median_m"), errors="coerce"),
                "prior_slope_m_per_m": pd.to_numeric(row.get("anchor_trend_slope_m_per_m"), errors="coerce"),
            }
        else:
            donor, basis, distance_steps = _find_best_donor(summary, idx)
            if donor is None:
                warnings.append("transfer_prior_component_has_no_donor")
                prior = {
                    **row.to_dict(),
                    "prior_basis": basis,
                    "prior_confidence": "low",
                    "prior_source_component_id": "",
                    "prior_source_levelpath_id": "",
                    "prior_anchor_count": 0,
                    "prior_distance_component_steps": -1,
                    "prior_upstream_offset_m": np.nan,
                    "prior_downstream_offset_m": np.nan,
                    "prior_median_offset_m": np.nan,
                    "prior_slope_m_per_m": np.nan,
                }
            else:
                warnings.append("transfer_prior_component_uses_transferred_donor")
                prior = {
                    **row.to_dict(),
                    "prior_basis": basis,
                    "prior_confidence": "medium",
                    "prior_source_component_id": str(donor.get("component_id", "") or ""),
                    "prior_source_levelpath_id": str(donor.get("levelpath_id", "") or ""),
                    "prior_anchor_count": int(pd.to_numeric(donor.get("observed_offset_count"), errors="coerce") or 0),
                    "prior_distance_component_steps": int(distance_steps),
                    "prior_upstream_offset_m": pd.to_numeric(donor.get("upstream_anchor_offset_m"), errors="coerce"),
                    "prior_downstream_offset_m": pd.to_numeric(donor.get("downstream_anchor_offset_m"), errors="coerce"),
                    "prior_median_offset_m": pd.to_numeric(donor.get("observed_offset_median_m"), errors="coerce"),
                    "prior_slope_m_per_m": pd.to_numeric(donor.get("anchor_trend_slope_m_per_m"), errors="coerce"),
                }
        rows.append(prior)
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=summary.crs)
    diagnostics = {
        "record_count": int(len(out)),
        "local_observed_component_count": int(np.count_nonzero(out["prior_basis"].astype(str) == "local_observed_component")) if len(out) else 0,
        "transferred_prior_component_count": int(np.count_nonzero(out["prior_basis"].astype(str).isin(["same_levelpath_observed_component", "nearest_supported_component"]))) if len(out) else 0,
        "undonored_component_count": int(np.count_nonzero(out["prior_basis"].astype(str) == "no_observed_support_available")) if len(out) else 0,
    }
    return out.reset_index(drop=True), sorted(set(warnings)), diagnostics


def validate_offset_transfer_priors(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    missing = [f for f in _REQUIRED_FIELDS if f not in gdf.columns]
    recs = int(len(gdf))
    geom_valid = bool(getattr(gdf, "geometry", None) is not None and gdf.geometry.notna().all() and gdf.is_valid.all()) if recs > 0 else True
    basis_nonempty = bool(np.all(gdf["prior_basis"].astype(str).str.len().to_numpy(dtype=int) > 0)) if recs > 0 and "prior_basis" in gdf.columns else False
    return {
        "valid": recs > 0 and not missing and geom_valid and basis_nonempty,
        "record_count": recs,
        "missing_required_fields": missing,
        "geometry_valid": geom_valid,
        "basis_nonempty": basis_nonempty,
    }


def _join_inputs(wse_points_path: Path, observed_offset_points_path: Path) -> gpd.GeoDataFrame:
    wse = _read_gpkg(wse_points_path).copy()
    observed = _read_gpkg(observed_offset_points_path).copy()
    if "point_id" not in wse.columns or "point_id" not in observed.columns:
        raise RuntimeError("river_v2_transfer_prior_missing_point_id")
    wse["point_id"] = wse["point_id"].astype(str)
    observed["point_id"] = observed["point_id"].astype(str)
    keep_wse = [c for c in ("point_id", "station_m", *_GROUP_COLS, *_STREAM_ORDER_COLS, *_WIDTH_COLS, *_SLOPE_COLS, "geometry") if c in wse.columns]
    keep_observed = [c for c in ("point_id", "observed_offset_m") if c in observed.columns]
    merged = wse[keep_wse].merge(observed[keep_observed], on=["point_id"], how="left")
    merged["station_m"] = pd.to_numeric(merged["station_m"], errors="coerce")
    if "observed_offset_m" in merged.columns:
        merged["observed_offset_m"] = pd.to_numeric(merged["observed_offset_m"], errors="coerce")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=wse.crs).sort_values([c for c in (*[c for c in _GROUP_COLS if c in merged.columns], "station_m", "point_id") if c in merged.columns], kind="mergesort").reset_index(drop=True)


def run_offset_transfer_prior_stage(
    ctx: RiverV2Context,
    *,
    wse_points_path: Path,
    observed_offset_points_path: Path,
) -> RiverV2StageResult:
    joined = _join_inputs(wse_points_path, observed_offset_points_path)
    priors, warnings, diagnostics = build_offset_transfer_priors(joined)
    validation = validate_offset_transfer_priors(priors)
    if not validation.get("valid"):
        raise RuntimeError(f"river_v2_transfer_prior_invalid:{validation}")
    out_path = ctx.paths.component_offset_transfer_priors
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    priors.to_file(out_path, driver="GPKG")
    receipt = build_river_v2_stage_receipt(
        stage_id=STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS,
        output_artifact=str(out_path),
        input_artifacts=ctx.direct_stage_input_artifacts(wse_points_path, observed_offset_points_path),
        record_count=int(len(priors)),
        field_schema=component_offset_transfer_prior_field_schema(),
        vertical_reference=ctx.vertical_reference,
        warnings=warnings,
        source_logic="component_level_offset_priors_from_local_observed_offsets_first_same_levelpath_or_nearest_supported_component_second_placeholder_not_allowed_here",
        validation={**validation, **diagnostics},
    )
    receipt_path = write_river_v2_receipt(receipt, ctx.paths.component_offset_transfer_priors_receipt)
    return RiverV2StageResult(
        stage_id=STAGE_COMPONENT_OFFSET_TRANSFER_PRIORS,
        output_artifact=out_path,
        receipt_path=receipt_path,
        record_count=int(len(priors)),
        validation={**validation, **diagnostics},
        warnings=warnings,
    )
