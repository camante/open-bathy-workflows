from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from authoritative_river_roles import ROLE_BED_CORE, ROLE_BED_INNER, ROLE_BANK_MARGIN, ROLE_AMBIGUOUS, code_to_role
from river_anchor_policy import build_anchor_policy_table, write_anchor_policy_artifacts
from river_station_target_contract import build_station_target_table, write_station_target_artifacts

log = logging.getLogger(__name__)


def _sample_raster_at_points(points_gdf, raster_path: str | Path | None, field: str) -> pd.Series:
    import rasterio

    out = pd.Series(np.nan, index=points_gdf.index, dtype="float32", name=field)
    if points_gdf is None or getattr(points_gdf, "empty", True) or raster_path is None:
        return out
    path = Path(raster_path)
    if not path.exists():
        return out
    pts = points_gdf
    with rasterio.open(path) as ds:
        try:
            if getattr(pts, "crs", None) is not None and ds.crs is not None and str(pts.crs) != str(ds.crs):
                pts = pts.to_crs(ds.crs)
        except Exception:
            log.debug("_sample_raster_at_points: failed CRS harmonization for %s", path, exc_info=True)
            pts = points_gdf
        vals = []
        xy = list(zip(pts.geometry.x.to_numpy(dtype=float), pts.geometry.y.to_numpy(dtype=float)))
        for raw in ds.sample(xy):
            val = float(raw[0]) if np.size(raw) else np.nan
            if ds.nodata is not None and np.isfinite(val) and np.isclose(val, float(ds.nodata)):
                val = np.nan
            vals.append(val)
    return pd.Series(np.asarray(vals, dtype=np.float32), index=points_gdf.index, name=field)




def _normalize_topology_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Carry explicit network-topology metadata into the frame contract when available."""
    out = frame.copy()
    if "downstream_component_id" not in out.columns:
        out["downstream_component_id"] = pd.Series(pd.NA, index=out.index, dtype="object")
    else:
        out["downstream_component_id"] = out["downstream_component_id"].where(~pd.isna(out["downstream_component_id"]), pd.NA).astype("object")

    if "upstream_component_ids" not in out.columns:
        out["upstream_component_ids"] = pd.Series(pd.NA, index=out.index, dtype="object")
    else:
        vals = []
        for val in out["upstream_component_ids"].tolist():
            if pd.isna(val):
                vals.append(pd.NA)
            elif isinstance(val, (list, tuple, set)):
                vals.append(",".join(str(v) for v in val if pd.notna(v)) or pd.NA)
            else:
                vals.append(str(val))
        out["upstream_component_ids"] = pd.Series(vals, index=out.index, dtype="object")

    if "junction_id" not in out.columns:
        out["junction_id"] = pd.Series(pd.NA, index=out.index, dtype="object")
    else:
        out["junction_id"] = out["junction_id"].where(~pd.isna(out["junction_id"]), pd.NA).astype("object")

    for fld in ("mainstem_rank", "network_order", "distance_to_mouth_m"):
        if fld not in out.columns:
            out[fld] = pd.Series(np.nan, index=out.index, dtype="float32")
        else:
            out[fld] = pd.to_numeric(out[fld], errors="coerce").astype("float32")
    return out

def _as_float_series(frame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float32", name=name)
    return pd.to_numeric(frame[name], errors="coerce").astype("float32")


def _collapse_duplicate_station_rows(frame, logger: Optional[logging.Logger] = None) -> tuple[pd.DataFrame, int]:
    try:
        import geopandas as gpd
    except Exception:
        gpd = None

    if frame is None or getattr(frame, "empty", True):
        return frame, 0

    work = frame.copy()
    if "station_m" not in work.columns:
        return work, 0
    if "component_id" not in work.columns:
        work["component_id"] = "main"
    work["component_id"] = work["component_id"].fillna("main").astype(str)
    work["station_m"] = pd.to_numeric(work["station_m"], errors="coerce")
    work = work.loc[np.isfinite(work["station_m"])].copy()
    if work.empty:
        return work, 0

    work["_station_key"] = np.round(work["station_m"].to_numpy(dtype=float), 6)
    key_cols = ["component_id", "_station_key"]
    duplicate_rows = int(work.duplicated(subset=key_cols, keep=False).sum())
    if duplicate_rows == 0:
        return work.drop(columns=["_station_key"]), 0

    def _first_nonnull(series: pd.Series):
        for val in series.tolist():
            if pd.isna(val):
                continue
            if isinstance(val, str) and not val.strip():
                continue
            return val
        return pd.NA

    rows = []
    geom_name = getattr(work, "geometry", None).name if getattr(work, "geometry", None) is not None else None
    numeric_cols = {
        col for col in work.columns
        if col not in {"_station_key", "component_id", geom_name}
        and (pd.api.types.is_numeric_dtype(work[col]) or col.endswith("_m") or col.endswith("_count") or col.endswith("_weight") or col.endswith("_strength") or col.endswith("_confidence") or col.endswith("_influence") or col.endswith("_rank") or col.endswith("_order"))
    }
    bool_cols = {col for col in work.columns if col not in {"_station_key", geom_name} and pd.api.types.is_bool_dtype(work[col])}
    for (_, _), grp in work.groupby(key_cols, sort=False, dropna=False):
        grp = grp.copy()
        chosen_idx = grp.index[0]
        if "centerline_influence" in grp.columns:
            infl = pd.to_numeric(grp["centerline_influence"], errors="coerce")
            if infl.notna().any():
                chosen_idx = infl.fillna(-np.inf).idxmax()
        row = grp.loc[chosen_idx].copy()
        row["station_m"] = float(pd.to_numeric(grp["station_m"], errors="coerce").median())
        row["component_id"] = str(_first_nonnull(grp["component_id"]))
        for col in numeric_cols:
            vals = pd.to_numeric(grp[col], errors="coerce")
            if vals.notna().any():
                row[col] = float(vals.median())
            else:
                row[col] = np.nan
        for col in bool_cols:
            vals = grp[col].fillna(False).astype(bool)
            row[col] = bool(vals.any())
        for col in grp.columns:
            if col in key_cols or col in numeric_cols or col in bool_cols or col == geom_name:
                continue
            if col == "station_m":
                continue
            row[col] = _first_nonnull(grp[col])
        rows.append(row)

    collapsed = pd.DataFrame(rows).drop(columns=["_station_key"], errors="ignore")
    if gpd is not None and geom_name is not None and geom_name in collapsed.columns:
        collapsed = gpd.GeoDataFrame(collapsed, geometry=geom_name, crs=getattr(frame, "crs", None))
    removed_count = int(len(work) - len(collapsed))
    if removed_count > 0 and logger is not None:
        logger.warning(
            "[RIVER][FRAME] Collapsed duplicate centerline/frame station rows on (component_id, station_m): removed=%d involved=%d",
            removed_count,
            duplicate_rows,
        )
    return collapsed, removed_count




def _first_nonempty_string(series: pd.Series, default: str) -> pd.Series:
    vals = series.fillna('').astype(str).str.strip()
    return vals.where(vals.ne(''), default)


def _apply_frame_graph_provenance_fallbacks(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure frame rows expose support/backbone provenance even when upstream centerline points
    do not already carry graph_* diagnostics. This keeps XS-supported rows truthfully tagged for
    downstream scaffold construction instead of silently degrading to generic backbone support.
    """
    out = frame.copy()
    support = _first_nonempty_string(out.get("channel_support_class", pd.Series(index=out.index, dtype="object")), "unsupported")
    backbone_mode = _first_nonempty_string(out.get("backbone_mode", pd.Series(index=out.index, dtype="object")), "missing")
    residual_mode = _first_nonempty_string(out.get("residual_shape_mode", pd.Series(index=out.index, dtype="object")), "none")

    cand = _first_nonempty_string(out["graph_candidate_source"], "missing") if "graph_candidate_source" in out.columns else pd.Series(["missing"] * len(out), index=out.index, dtype="object")
    gclass = _first_nonempty_string(out["graph_solver_support_class"], "unsupported") if "graph_solver_support_class" in out.columns else pd.Series(["unsupported"] * len(out), index=out.index, dtype="object")
    gmode = _first_nonempty_string(out["graph_solution_mode"], "missing") if "graph_solution_mode" in out.columns else pd.Series(["missing"] * len(out), index=out.index, dtype="object")

    xs_mask = support.eq("xs_supported") & residual_mode.eq("xs_residual_to_backbone")
    bank_mask = support.eq("bank_stage_only")
    auth_mask = support.eq("authoritative_in_channel")
    backbone_mask = backbone_mode.eq("authoritative_backbone")
    center_mask = backbone_mode.eq("centerline_bed")
    long_mask = backbone_mode.eq("longitudinal_profile")
    active_mask = backbone_mode.eq("active_core_support")

    cand = cand.copy()
    cand.loc[cand.eq("missing") & xs_mask] = "xs_profile_resampled"
    cand.loc[cand.eq("missing") & auth_mask] = "authoritative_in_channel"
    cand.loc[cand.eq("missing") & backbone_mask] = "authoritative_backbone"
    cand.loc[cand.eq("missing") & long_mask] = "longitudinal_profile"
    cand.loc[cand.eq("missing") & active_mask] = "active_core_support"
    cand.loc[cand.eq("missing") & center_mask] = "centerline_bed"

    gclass = gclass.copy()
    gclass.loc[gclass.eq("unsupported") & xs_mask] = "xs_residual_only"
    gclass.loc[gclass.eq("unsupported") & auth_mask] = "authoritative_locked"
    gclass.loc[gclass.eq("unsupported") & backbone_mask] = "authoritative_backbone"
    gclass.loc[gclass.eq("unsupported") & bank_mask] = "stage_controlled"
    gclass.loc[gclass.eq("unsupported") & (long_mask | center_mask | active_mask)] = "resolved_backbone"

    gmode = gmode.copy()
    gmode.loc[gmode.eq("missing") & xs_mask] = "frame_xs_supported"
    gmode.loc[gmode.eq("missing") & auth_mask] = "frame_authoritative_anchor"
    gmode.loc[gmode.eq("missing") & bank_mask] = "frame_bank_stage_only"
    gmode.loc[gmode.eq("missing") & (long_mask | center_mask | backbone_mask | active_mask)] = "frame_backbone_resolved"

    out["graph_candidate_source"] = pd.Series(cand, index=out.index, dtype="object")
    out["graph_solver_support_class"] = pd.Series(gclass, index=out.index, dtype="object")
    out["graph_solution_mode"] = pd.Series(gmode, index=out.index, dtype="object")
    if "graph_hard_lock" not in out.columns:
        out["graph_hard_lock"] = auth_mask.astype(bool)
    else:
        out["graph_hard_lock"] = out["graph_hard_lock"].fillna(False).astype(bool) | auth_mask.astype(bool)
    return out


def _metric_gdf(gdf):
    try:
        if gdf is None or getattr(gdf, "empty", True):
            return gdf
        if getattr(getattr(gdf, "crs", None), "is_projected", False):
            return gdf
        try:
            target = gdf.estimate_utm_crs()
        except Exception:
            target = None
        return gdf.to_crs(target or "EPSG:3857")
    except Exception:
        return gdf


_BANK_ZONE_TOKENS = {"bank", "left_bank", "right_bank", "bank_edge", "edge_bank"}


def _first_present_str(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name].fillna("").astype(str)
    return pd.Series("", index=df.index, dtype=object)


def _first_present_numeric(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _classify_authoritative_xs_support(xs):
    xs = xs.copy()
    zone = _first_present_str(xs, ["support_zone", "xs_support_zone", "xs_zone", "zone", "sample_zone"]).str.lower().str.strip()
    role = _first_present_str(xs, ["node_role", "support_role", "xs_role", "role"]).str.lower().str.strip()
    norm = _first_present_numeric(xs, ["normalized_position", "normalized_dist", "xs_fraction", "norm_pos", "rel_pos"]) 
    support_zone = np.full(len(xs), 'unknown', dtype=object)
    is_bank = zone.isin(_BANK_ZONE_TOKENS).to_numpy(dtype=bool) | role.isin(["left_bank", "right_bank", "bank"]).to_numpy(dtype=bool)
    is_core = zone.isin(["channel_core", "core", "thalweg"]).to_numpy(dtype=bool) | role.isin(["thalweg", "channel_core"]).to_numpy(dtype=bool)
    is_inner = zone.isin(["left_inner", "right_inner", "inner"]).to_numpy(dtype=bool) | role.isin(["left_inner", "right_inner", "inner"]).to_numpy(dtype=bool)
    norm_arr = norm.to_numpy(dtype=float)
    finite_norm = np.isfinite(norm_arr)
    is_bank |= finite_norm & ((norm_arr <= 0.15) | (norm_arr >= 0.85))
    is_core |= finite_norm & ((norm_arr >= 0.40) & (norm_arr <= 0.60)) & (~is_bank)
    is_inner |= finite_norm & ((norm_arr > 0.15) & (norm_arr < 0.85)) & (~is_bank) & (~is_core)
    support_zone[is_bank] = 'bank'
    support_zone[is_inner] = 'inner'
    support_zone[is_core] = 'core'
    support_class = np.full(len(xs), 'authoritative_unknown', dtype=object)
    support_class[is_bank] = 'authoritative_bank_edge'
    support_class[is_inner] = 'authoritative_bed_inner'
    support_class[is_core] = 'authoritative_bed_core'
    counts_as_true = np.isin(support_class, ['authoritative_bed_inner', 'authoritative_bed_core'])
    xs['support_zone_class'] = pd.Series(support_zone, index=xs.index, dtype=object)
    xs['support_class'] = pd.Series(support_class, index=xs.index, dtype=object)
    xs['counts_as_true_measured_xs'] = pd.Series(counts_as_true, index=xs.index, dtype=bool)
    return xs


def _summarize_station_authoritative_xs_class(classes: pd.Series) -> str:
    values = pd.Series(classes).fillna("authoritative_unknown").astype(str)
    value_counts = values.value_counts()
    inner_count = int(values.isin(["authoritative_bed_inner", "authoritative_bed_core"]).sum())
    bank_count = int(values.eq("authoritative_bank_edge").sum())
    if inner_count > 0 and bank_count > 0:
        return "authoritative_bed_mixed"
    if inner_count > 0:
        inner_modes = value_counts.reindex(["authoritative_bed_core", "authoritative_bed_inner"]).fillna(0)
        if int(inner_modes.sum()) > 0:
            return str(inner_modes.idxmax())
        return "authoritative_bed_inner"
    if bank_count > 0:
        return "authoritative_bank_edge"
    if not value_counts.empty:
        return str(value_counts.index[0])
    return "authoritative_unknown"


def build_channel_frame_products(
    *,
    river_dir: str | Path,
    centerline_points_path: str | Path | None,
    xs_support_points_path: str | Path | None,
    bank_elevation_path: str | Path | None,
    bank_influence_path: str | Path | None,
    left_bank_fit_elevation_path: str | Path | None = None,
    right_bank_fit_elevation_path: str | Path | None = None,
    bank_pair_fit_elevation_path: str | Path | None = None,
    xs_support_elevation_path: str | Path | None,
    xs_support_weight_path: str | Path | None,
    disable_xs_influence: bool = False,
    centerline_elevation_path: str | Path | None,
    centerline_influence_path: str | Path | None,
    longitudinal_profile_elevation_path: str | Path | None,
    authoritative_support_mask_path: str | Path | None,
    authoritative_support_depth_path: str | Path | None,
    authoritative_bed_elevation_path: str | Path | None,
    active_core_support_elevation_path: str | Path | None = None,
    authoritative_role_code_path: str | Path | None = None,
    authoritative_role_confidence_path: str | Path | None = None,
    authoritative_distance_to_bank_path: str | Path | None = None,
    authoritative_normalized_channel_position_path: str | Path | None = None,
    export_legacy_mixed_bed: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    import geopandas as gpd

    river_dir = Path(river_dir)
    cpts = Path(centerline_points_path) if centerline_points_path else None
    if cpts is None or not cpts.exists():
        return {}
    centerline = gpd.read_file(cpts)
    if centerline is None or centerline.empty:
        return {}
    frame = _normalize_topology_fields(centerline.copy())
    if "station_m" not in frame.columns:
        raise RuntimeError("channel_frame_requires_stationed_centerline_points")
    frame["station_m"] = pd.to_numeric(frame["station_m"], errors="coerce")
    if "component_id" not in frame.columns:
        frame["component_id"] = "main"
    frame["component_id"] = frame["component_id"].fillna("main").astype(str)
    frame["frame_role"] = "regular_centerline_station"
    frame["centerline_bed_z_m"] = _sample_raster_at_points(frame, centerline_elevation_path, "centerline_bed_z_m")
    frame["centerline_influence"] = _sample_raster_at_points(frame, centerline_influence_path, "centerline_influence")
    frame["xs_support_z_m"] = _sample_raster_at_points(frame, xs_support_elevation_path, "xs_support_z_m")
    frame["xs_support_weight"] = _sample_raster_at_points(frame, xs_support_weight_path, "xs_support_weight")
    if disable_xs_influence:
        frame["xs_support_z_m"] = pd.Series(np.nan, index=frame.index, dtype="float32")
        frame["xs_support_weight"] = pd.Series(0.0, index=frame.index, dtype="float32")
    frame["bank_low_stage_z_m"] = _sample_raster_at_points(frame, bank_elevation_path, "bank_low_stage_z_m")
    frame["left_bank_fit_z_m"] = _sample_raster_at_points(frame, left_bank_fit_elevation_path, "left_bank_fit_z_m")
    frame["right_bank_fit_z_m"] = _sample_raster_at_points(frame, right_bank_fit_elevation_path, "right_bank_fit_z_m")
    frame["bank_pair_fit_z_m"] = _sample_raster_at_points(frame, bank_pair_fit_elevation_path, "bank_pair_fit_z_m")
    frame["bank_influence"] = _sample_raster_at_points(frame, bank_influence_path, "bank_influence")
    frame["longitudinal_profile_z_m"] = _sample_raster_at_points(frame, longitudinal_profile_elevation_path, "longitudinal_profile_z_m")
    frame["active_core_support_z_m"] = _sample_raster_at_points(frame, active_core_support_elevation_path, "active_core_support_z_m")
    frame["authoritative_support_mask"] = _sample_raster_at_points(frame, authoritative_support_mask_path, "authoritative_support_mask")
    frame["authoritative_support_depth_m"] = _sample_raster_at_points(frame, authoritative_support_depth_path, "authoritative_support_depth_m")
    frame["authoritative_bed_z_m"] = _sample_raster_at_points(frame, authoritative_bed_elevation_path, "authoritative_bed_z_m")
    frame["authoritative_role_code"] = _sample_raster_at_points(frame, authoritative_role_code_path, "authoritative_role_code")
    frame["authoritative_role_confidence"] = _sample_raster_at_points(frame, authoritative_role_confidence_path, "authoritative_role_confidence")
    frame["authoritative_distance_to_bank_m"] = _sample_raster_at_points(frame, authoritative_distance_to_bank_path, "authoritative_distance_to_bank_m")
    frame["authoritative_normalized_channel_position"] = _sample_raster_at_points(frame, authoritative_normalized_channel_position_path, "authoritative_normalized_channel_position")
    frame, duplicate_station_rows_collapsed = _collapse_duplicate_station_rows(frame, logger=logger)
    frame = frame.reset_index(drop=True)
    auth_present = np.isfinite(_as_float_series(frame, "authoritative_support_mask")) & (_as_float_series(frame, "authoritative_support_mask") > 0.0) & np.isfinite(_as_float_series(frame, "authoritative_bed_z_m"))
    role_code_values = pd.to_numeric(frame.get("authoritative_role_code", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=float)
    role_codes = np.clip(np.rint(np.nan_to_num(role_code_values, nan=0.0)), 0, 255).astype(np.uint8)
    role_strings = np.array([code_to_role(v) for v in role_codes], dtype=object)
    missing_role_codes = np.asarray(auth_present) & (~np.isfinite(role_code_values))
    role_strings[missing_role_codes] = ROLE_BED_INNER
    role_strings[~np.asarray(auth_present)] = "no_authoritative_support"
    frame["authoritative_role"] = pd.Series(role_strings, index=frame.index, dtype="object")
    bed_support_present = (np.asarray(auth_present) & np.isin(role_strings, [ROLE_BED_CORE, ROLE_BED_INNER]))
    bank_margin_present = (np.asarray(auth_present) & pd.Series(role_strings, index=frame.index).eq(ROLE_BANK_MARGIN).to_numpy(dtype=bool))
    ambiguous_present = (np.asarray(auth_present) & pd.Series(role_strings, index=frame.index).eq(ROLE_AMBIGUOUS).to_numpy(dtype=bool))
    frame["authoritative_bed_support_present"] = bed_support_present
    frame["authoritative_bank_margin_present"] = bank_margin_present
    frame["authoritative_ambiguous_present"] = ambiguous_present
    frame["authoritative_anchor_present"] = auth_present.astype(bool)
    frame.loc[~auth_present, "authoritative_bed_z_m"] = np.nan
    frame["authoritative_hard_bed_z_m"] = pd.Series(np.where(bed_support_present, _as_float_series(frame, "authoritative_bed_z_m"), np.nan), index=frame.index, dtype="float32")
    frame["authoritative_bank_margin_z_m"] = pd.Series(np.where(bank_margin_present, _as_float_series(frame, "authoritative_bed_z_m"), np.nan), index=frame.index, dtype="float32")
    frame["authoritative_anchor_count"] = auth_present.astype(np.int16)
    auth_strength = np.where(np.isfinite(_as_float_series(frame, "authoritative_support_mask")), _as_float_series(frame, "authoritative_support_mask"), 0.0)
    frame["authoritative_station_support_strength"] = pd.Series(np.asarray(auth_strength, dtype=np.float32), index=frame.index)
    supported_station = (auth_strength > 0.0) & np.isfinite(_as_float_series(frame, "centerline_bed_z_m"))
    authoritative_backbone_candidate = np.where(
        np.asarray(bed_support_present),
        _as_float_series(frame, "authoritative_hard_bed_z_m"),
        np.where(supported_station, _as_float_series(frame, "centerline_bed_z_m"), np.nan),
    )
    frame["authoritative_backbone_candidate_z_m"] = pd.Series(np.asarray(authoritative_backbone_candidate, dtype=np.float32), index=frame.index)

    xs_weight = _as_float_series(frame, "xs_support_weight")
    xs_present = np.isfinite(_as_float_series(frame, "xs_support_z_m")) & (xs_weight > 0.05)
    bank_present = np.isfinite(_as_float_series(frame, "bank_low_stage_z_m"))

    support_class = np.full(len(frame), "unsupported", dtype=object)
    support_class[np.asarray(bank_present)] = "bank_stage_only"
    support_class[np.asarray(xs_present)] = "xs_supported"
    support_class[np.asarray(ambiguous_present)] = "authoritative_ambiguous"
    support_class[np.asarray(bank_margin_present)] = "authoritative_bank_margin"
    support_class[np.asarray(bed_support_present)] = "authoritative_in_channel"
    frame["channel_support_class"] = pd.Series(support_class, index=frame.index, dtype="object")

    authoritative_backbone = np.full(len(frame), np.nan, dtype=np.float32)
    for comp, sub in frame.groupby("component_id", sort=False):
        s = pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float)
        auth_z = pd.to_numeric(sub.get("authoritative_backbone_candidate_z_m", np.nan), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(s) & np.isfinite(auth_z)
        if np.count_nonzero(valid) == 0:
            continue
        if np.count_nonzero(valid) == 1:
            authoritative_backbone[sub.index.to_numpy()] = np.float32(auth_z[valid][0])
            continue
        order = np.argsort(s[valid])
        sv = s[valid][order]
        zv = auth_z[valid][order]
        uniq_s, inv = np.unique(np.round(sv, 6), return_inverse=True)
        agg = np.full(uniq_s.shape, np.nan, dtype=float)
        for i in range(uniq_s.size):
            sel = inv == i
            agg[i] = float(np.nanmedian(zv[sel])) if np.any(sel) else np.nan
        good = np.isfinite(agg)
        if np.count_nonzero(good) == 1:
            authoritative_backbone[sub.index.to_numpy()] = np.float32(agg[good][0])
        elif np.count_nonzero(good) >= 2:
            authoritative_backbone[sub.index.to_numpy()] = np.interp(s, uniq_s[good], agg[good]).astype(np.float32)
    frame["authoritative_backbone_z_m"] = pd.Series(authoritative_backbone, index=frame.index, dtype="float32")

    resolved_backbone_candidate = np.where(
        np.asarray(auth_present),
        _as_float_series(frame, "authoritative_bed_z_m"),
        np.where(
            np.isfinite(_as_float_series(frame, "authoritative_backbone_z_m")),
            _as_float_series(frame, "authoritative_backbone_z_m"),
            np.where(
                np.isfinite(_as_float_series(frame, "active_core_support_z_m")),
                _as_float_series(frame, "active_core_support_z_m"),
                np.where(
                    np.isfinite(_as_float_series(frame, "longitudinal_profile_z_m")),
                    _as_float_series(frame, "longitudinal_profile_z_m"),
                    _as_float_series(frame, "centerline_bed_z_m"),
                ),
            ),
        ),
    )
    frame["resolved_backbone_candidate_z_m"] = pd.Series(np.asarray(resolved_backbone_candidate, dtype=np.float32), index=frame.index, dtype="float32")

    resolved_backbone = np.full(len(frame), np.nan, dtype=np.float32)
    resolved_backbone_source = np.full(len(frame), "missing", dtype=object)
    for comp, sub in frame.groupby("component_id", sort=False):
        s = pd.to_numeric(sub["station_m"], errors="coerce").to_numpy(dtype=float)
        rb = pd.to_numeric(sub.get("resolved_backbone_candidate_z_m", np.nan), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(s) & np.isfinite(rb)
        if np.count_nonzero(valid) == 0:
            continue
        if np.count_nonzero(valid) == 1:
            resolved_backbone[sub.index.to_numpy()] = np.float32(rb[valid][0])
        else:
            order = np.argsort(s[valid])
            sv = s[valid][order]
            zv = rb[valid][order]
            uniq_s, inv = np.unique(np.round(sv, 6), return_inverse=True)
            agg = np.full(uniq_s.shape, np.nan, dtype=float)
            for i in range(uniq_s.size):
                sel = inv == i
                agg[i] = float(np.nanmedian(zv[sel])) if np.any(sel) else np.nan
            good = np.isfinite(agg)
            if np.count_nonzero(good) == 1:
                resolved_backbone[sub.index.to_numpy()] = np.float32(agg[good][0])
            elif np.count_nonzero(good) >= 2:
                resolved_backbone[sub.index.to_numpy()] = np.interp(s, uniq_s[good], agg[good]).astype(np.float32)
        sub_idx = sub.index.to_numpy()
        auth_sub = np.isfinite(pd.to_numeric(sub.get("authoritative_hard_bed_z_m", np.nan), errors="coerce").to_numpy(dtype=float))
        bank_margin_sub = np.isfinite(pd.to_numeric(sub.get("authoritative_bank_margin_z_m", np.nan), errors="coerce").to_numpy(dtype=float))
        auth_backbone_sub = np.isfinite(pd.to_numeric(sub.get("authoritative_backbone_z_m", np.nan), errors="coerce").to_numpy(dtype=float))
        active_sub = np.isfinite(pd.to_numeric(sub.get("active_core_support_z_m", np.nan), errors="coerce").to_numpy(dtype=float))
        long_sub = np.isfinite(pd.to_numeric(sub.get("longitudinal_profile_z_m", np.nan), errors="coerce").to_numpy(dtype=float))
        center_sub = np.isfinite(pd.to_numeric(sub.get("centerline_bed_z_m", np.nan), errors="coerce").to_numpy(dtype=float))
        source_sub = np.full(len(sub), "missing", dtype=object)
        source_sub[center_sub] = "centerline_bed"
        source_sub[long_sub] = "longitudinal_profile"
        source_sub[active_sub] = "active_core_support"
        source_sub[bank_margin_sub] = "authoritative_bank_margin"
        source_sub[auth_backbone_sub] = "authoritative_backbone"
        source_sub[auth_sub] = "authoritative_in_channel"
        resolved_backbone_source[sub_idx] = source_sub
    frame["resolved_backbone_z_m"] = pd.Series(resolved_backbone, index=frame.index, dtype="float32")
    frame["resolved_backbone_source"] = pd.Series(resolved_backbone_source, index=frame.index, dtype="object")

    backbone_z = np.where(
        np.asarray(bed_support_present),
        _as_float_series(frame, "authoritative_hard_bed_z_m"),
        _as_float_series(frame, "resolved_backbone_z_m"),
    )
    backbone_mode = np.where(
        np.asarray(bed_support_present),
        "authoritative_in_channel",
        np.where(
            np.isfinite(_as_float_series(frame, "authoritative_backbone_z_m")),
            "authoritative_backbone",
            np.where(
                np.asarray(bank_margin_present),
                "authoritative_bank_margin",
                np.where(
                    np.isfinite(_as_float_series(frame, "active_core_support_z_m")),
                    "active_core_support",
                    np.where(
                        np.isfinite(_as_float_series(frame, "longitudinal_profile_z_m")),
                        "longitudinal_profile",
                        np.where(
                            np.isfinite(_as_float_series(frame, "centerline_bed_z_m")),
                            "centerline_bed",
                            "missing",
                        ),
                    ),
                ),
            ),
        ),
    )
    frame["backbone_z_m"] = pd.Series(np.asarray(backbone_z, dtype=np.float32), index=frame.index, dtype="float32")
    frame["backbone_mode"] = pd.Series(np.asarray(backbone_mode, dtype=object), index=frame.index, dtype="object")
    stage_control = np.where(
        np.isfinite(_as_float_series(frame, "bank_pair_fit_z_m")),
        _as_float_series(frame, "bank_pair_fit_z_m"),
        np.where(
            np.isfinite(_as_float_series(frame, "bank_low_stage_z_m")),
            _as_float_series(frame, "bank_low_stage_z_m"),
            _as_float_series(frame, "longitudinal_profile_z_m"),
        ),
    )
    frame["resolved_stage_control_z_m"] = pd.Series(stage_control, index=frame.index, dtype="float32")
    stage_source = np.where(
        np.isfinite(_as_float_series(frame, "bank_pair_fit_z_m")),
        "bank_pair_fit",
        np.where(
            np.isfinite(_as_float_series(frame, "bank_low_stage_z_m")),
            "bank_elevation_xs",
            np.where(np.isfinite(_as_float_series(frame, "longitudinal_profile_z_m")), "longitudinal_profile", "missing"),
        ),
    )
    frame["resolved_stage_control_source"] = pd.Series(stage_source, index=frame.index, dtype="object")
    xs_residual = np.where(
        np.asarray(xs_present) & np.isfinite(_as_float_series(frame, "backbone_z_m")),
        _as_float_series(frame, "xs_support_z_m") - _as_float_series(frame, "backbone_z_m"),
        np.nan,
    )
    residual_mode = np.full(len(frame), "none", dtype=object)
    residual_keep = np.isfinite(xs_residual)
    bed_support_dist = pd.to_numeric(
        frame.get("profile_authoritative_bed_support_distance_m", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    far_from_bed = pd.Series(
        frame.get("profile_far_from_authoritative_bed_support", pd.Series(False, index=frame.index)),
        index=frame.index,
    ).fillna(False).astype(bool).to_numpy(dtype=bool)
    inside_fluvial = pd.Series(
        frame.get("profile_inside_fluvial_monotone_domain", pd.Series(False, index=frame.index)),
        index=frame.index,
    ).fillna(False).astype(bool).to_numpy(dtype=bool)
    support_class_series = pd.Series(frame["channel_support_class"], index=frame.index).fillna("unsupported").astype(str)
    support_class_values = support_class_series.to_numpy(dtype=object)
    long_unsupported = (
        inside_fluvial
        & (~np.asarray(bed_support_present))
        & (
            far_from_bed
            | ((np.isfinite(bed_support_dist)) & (bed_support_dist >= 500.0))
        )
    )
    weak_support = np.isin(support_class_values, ["bank_stage_only", "authoritative_bank_margin", "unsupported", "xs_supported"])
    suppress_residual = np.asarray(bank_margin_present) | (long_unsupported & weak_support)
    xs_residual = np.where(suppress_residual, np.nan, xs_residual)
    residual_mode[np.isfinite(xs_residual)] = "xs_residual_to_backbone"
    residual_mode[np.asarray(bank_margin_present) & np.isfinite(_as_float_series(frame, "xs_support_z_m"))] = "suppressed_bank_margin"
    residual_mode[long_unsupported & weak_support & np.isfinite(_as_float_series(frame, "xs_support_z_m"))] = "suppressed_longitudinal_core"
    frame["xs_residual_to_backbone_z_m"] = pd.Series(np.asarray(xs_residual, dtype=np.float32), index=frame.index, dtype="float32")
    frame["residual_shape_mode"] = pd.Series(residual_mode, index=frame.index, dtype="object")
    frame = _apply_frame_graph_provenance_fallbacks(frame)

    if export_legacy_mixed_bed:
        resolved = np.where(
            np.asarray(auth_present),
            _as_float_series(frame, "authoritative_bed_z_m"),
            np.where(
                np.isfinite(_as_float_series(frame, "resolved_backbone_z_m")),
                _as_float_series(frame, "resolved_backbone_z_m"),
                np.where(
                    np.asarray(xs_present),
                    _as_float_series(frame, "xs_support_z_m"),
                    np.where(
                        np.isfinite(_as_float_series(frame, "authoritative_backbone_z_m")),
                        _as_float_series(frame, "authoritative_backbone_z_m"),
                        np.where(
                            np.isfinite(_as_float_series(frame, "longitudinal_profile_z_m")),
                            _as_float_series(frame, "longitudinal_profile_z_m"),
                            _as_float_series(frame, "centerline_bed_z_m"),
                        ),
                    ),
                ),
            ),
        )
        frame["resolved_channel_bed_z_m"] = pd.Series(np.asarray(resolved, dtype=np.float32), index=frame.index)

    frame_path = river_dir / "river_channel_frame_points.gpkg"
    if frame_path.exists():
        frame_path.unlink()
    frame.to_file(frame_path, driver="GPKG")

    auth_centerline = frame.loc[np.asarray(auth_present)].copy()
    auth_centerline["artifact_role"] = "authoritative_centerline_anchor"
    auth_centerline_path = river_dir / "river_authoritative_centerline_anchors.gpkg"
    if auth_centerline_path.exists():
        auth_centerline_path.unlink()
    if not auth_centerline.empty:
        auth_centerline.to_file(auth_centerline_path, driver="GPKG")

    auth_xs_path = river_dir / "river_authoritative_xs_anchors.gpkg"
    xs_audit_points_path = river_dir / "river_measured_xs_audit_support_points.gpkg"
    xs_station_audit_path = river_dir / "river_measured_xs_station_audit.csv"
    xs_handoff_receipt_path = river_dir / "river_measured_xs_handoff_receipt.json"
    xs_anchor_count = 0
    xs_auth = None
    if xs_support_points_path and Path(xs_support_points_path).exists():
        try:
            xs = gpd.read_file(xs_support_points_path)
        except Exception:
            xs = None
        if xs is not None and not xs.empty:
            src = xs.get("xs_sample_source", pd.Series("", index=xs.index)).fillna("").astype(str)
            xs_auth = xs.loc[src.eq("authoritative")].copy()
            if not xs_auth.empty:
                xs_auth = _classify_authoritative_xs_support(xs_auth)
                xs_auth["artifact_role"] = "authoritative_xs_anchor"
                if auth_xs_path.exists():
                    auth_xs_path.unlink()
                xs_auth.to_file(auth_xs_path, driver="GPKG")
                xs_anchor_count = int(len(xs_auth))

    station_audit = frame[["component_id", "station_m"]].copy()
    station_audit["xs_support_z_m"] = pd.to_numeric(frame.get("xs_support_z_m", pd.Series(np.nan, index=frame.index)), errors="coerce").astype(np.float32)
    station_audit["xs_support_weight"] = pd.to_numeric(frame.get("xs_support_weight", pd.Series(np.nan, index=frame.index)), errors="coerce").astype(np.float32)
    station_audit["xs_support_present"] = np.isfinite(station_audit["xs_support_z_m"].to_numpy(dtype=float))
    station_audit["channel_support_class"] = frame.get("channel_support_class", pd.Series("missing", index=frame.index)).fillna("missing").astype(str)
    station_audit["authoritative_anchor_present"] = frame.get("authoritative_anchor_present", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    station_audit["xs_residual_to_backbone_z_m"] = pd.to_numeric(frame.get("xs_residual_to_backbone_z_m", pd.Series(np.nan, index=frame.index)), errors="coerce").astype(np.float32)
    station_audit["auth_xs_sample_count_near_station"] = 0
    station_audit["auth_xs_min_distance_m"] = np.float32(np.nan)
    station_audit["auth_xs_support_z_m"] = np.float32(np.nan)
    station_audit["auth_xs_support_inner_count"] = 0
    station_audit["auth_xs_support_bank_count"] = 0
    station_audit["auth_xs_support_class"] = pd.Series("authoritative_unknown", index=station_audit.index, dtype=object)
    if xs_auth is not None and not xs_auth.empty and not frame.empty:
        frame_metric = _metric_gdf(frame[["component_id", "station_m", "geometry", "channel_support_class", "xs_support_z_m", "xs_support_weight"]].copy()).reset_index().rename(columns={"index": "frame_row_index"})
        xs_metric = _metric_gdf(xs_auth.copy()).reset_index(drop=True)
        frame_xy = np.column_stack([frame_metric.geometry.x.to_numpy(dtype=float), frame_metric.geometry.y.to_numpy(dtype=float)])
        xs_xy = np.column_stack([xs_metric.geometry.x.to_numpy(dtype=float), xs_metric.geometry.y.to_numpy(dtype=float)])
        if frame_xy.size and xs_xy.size:
            dx = xs_xy[:, None, 0] - frame_xy[None, :, 0]
            dy = xs_xy[:, None, 1] - frame_xy[None, :, 1]
            d2 = dx * dx + dy * dy
            nearest_idx = np.argmin(d2, axis=1)
            nearest_dist = np.sqrt(np.min(d2, axis=1))
            nearest_station = frame_metric.iloc[nearest_idx].reset_index(drop=True)
            xs_auth = xs_auth.reset_index(drop=True).copy()
            xs_auth["nearest_frame_row_index"] = pd.to_numeric(nearest_station["frame_row_index"], errors="coerce").astype(int)
            xs_auth["nearest_component_id"] = nearest_station["component_id"].astype(str)
            xs_auth["nearest_station_m"] = pd.to_numeric(nearest_station["station_m"], errors="coerce").astype(np.float32)
            xs_auth["nearest_station_distance_m"] = nearest_dist.astype(np.float32)
            xs_auth["nearest_channel_support_class"] = nearest_station.get("channel_support_class", pd.Series("missing", index=nearest_station.index)).fillna("missing").astype(str)
            xs_auth["nearest_xs_support_z_m"] = pd.to_numeric(nearest_station.get("xs_support_z_m", np.nan), errors="coerce").astype(np.float32)
            xs_auth["nearest_xs_support_weight"] = pd.to_numeric(nearest_station.get("xs_support_weight", np.nan), errors="coerce").astype(np.float32)
            fallback_bed = xs_auth["support_class"].astype(str).eq("authoritative_unknown") & xs_auth["nearest_channel_support_class"].astype(str).isin(["xs_supported", "authoritative_in_channel"]) & np.isfinite(pd.to_numeric(xs_auth["nearest_xs_support_weight"], errors="coerce")) & (pd.to_numeric(xs_auth["nearest_xs_support_weight"], errors="coerce") > 0.05) & (pd.to_numeric(xs_auth["nearest_station_distance_m"], errors="coerce") <= 1.0)
            xs_auth.loc[fallback_bed, "support_zone_class"] = "inner"
            xs_auth.loc[fallback_bed, "support_class"] = "authoritative_bed_inner"
            xs_auth.loc[fallback_bed, "counts_as_true_measured_xs"] = True
            auth_z = pd.to_numeric(xs_auth.get("xs_z_m", xs_auth.get("z_m", xs_auth.get("depth_m", np.nan))), errors="coerce")
            xs_auth["authoritative_xs_support_z_m"] = auth_z.astype(np.float32)
            grouped = xs_auth.groupby("nearest_frame_row_index", dropna=False).agg(
                auth_xs_sample_count_near_station=("nearest_station_distance_m", "count"),
                auth_xs_min_distance_m=("nearest_station_distance_m", "min"),
                auth_xs_support_z_m=("authoritative_xs_support_z_m", lambda s: float(np.nanmedian(pd.to_numeric(s, errors="coerce"))) if np.any(np.isfinite(pd.to_numeric(s, errors="coerce"))) else np.nan),
                auth_xs_support_inner_count=("counts_as_true_measured_xs", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
                auth_xs_support_bank_count=("support_class", lambda s: int(pd.Series(s).astype(str).eq("authoritative_bank_edge").sum())),
                auth_xs_support_class=("support_class", _summarize_station_authoritative_xs_class),
            )
            for frame_row_index, row in grouped.iterrows():
                if pd.isna(frame_row_index):
                    continue
                i = int(frame_row_index)
                if 0 <= i < len(station_audit):
                    station_audit.at[i, "auth_xs_sample_count_near_station"] = int(row.get("auth_xs_sample_count_near_station", 0) or 0)
                    station_audit.at[i, "auth_xs_min_distance_m"] = np.float32(row.get("auth_xs_min_distance_m", np.nan))
                    station_audit.at[i, "auth_xs_support_z_m"] = np.float32(row.get("auth_xs_support_z_m", np.nan))
                    station_audit.at[i, "auth_xs_support_inner_count"] = int(row.get("auth_xs_support_inner_count", 0) or 0)
                    station_audit.at[i, "auth_xs_support_bank_count"] = int(row.get("auth_xs_support_bank_count", 0) or 0)
                    station_audit.at[i, "auth_xs_support_class"] = str(row.get("auth_xs_support_class", "authoritative_unknown") or "authoritative_unknown")
            if xs_audit_points_path.exists():
                xs_audit_points_path.unlink()
            xs_auth.to_file(xs_audit_points_path, driver="GPKG")

    support_present = station_audit["xs_support_present"].to_numpy(dtype=bool)
    weight = pd.to_numeric(station_audit["xs_support_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    auth_count = pd.to_numeric(station_audit["auth_xs_sample_count_near_station"], errors="coerce").fillna(0).to_numpy(dtype=int)
    auth_inner_count = pd.to_numeric(station_audit["auth_xs_support_inner_count"], errors="coerce").fillna(0).to_numpy(dtype=int)
    auth_bank_count = pd.to_numeric(station_audit["auth_xs_support_bank_count"], errors="coerce").fillna(0).to_numpy(dtype=int)
    auth_support_present = np.isfinite(pd.to_numeric(station_audit["auth_xs_support_z_m"], errors="coerce").to_numpy(dtype=float))
    auth_support_class = station_audit["auth_xs_support_class"].astype(str).to_numpy()
    support_class = station_audit["channel_support_class"].astype(str).to_numpy()
    bank_only_authoritative = auth_support_present & (auth_bank_count > 0) & (auth_inner_count <= 0)
    candidate = auth_support_present & (auth_count > 0) & (auth_inner_count > 0)
    mixed_or_inner_support = np.isin(auth_support_class, ["authoritative_bed_inner", "authoritative_bed_core", "authoritative_bed_mixed"])
    qualified = (
        candidate
        & ((weight > 0.05) | (auth_inner_count > 0))
        & np.isin(support_class, ["xs_supported", "authoritative_in_channel"])
        & (~bank_only_authoritative)
        & mixed_or_inner_support
    )
    station_audit["true_measured_xs_candidate"] = candidate.astype(bool)
    station_audit["true_measured_xs_qualified"] = qualified.astype(bool)
    station_audit["bank_only_authoritative"] = bank_only_authoritative.astype(bool)
    station_audit["true_measured_xs_candidate_reason"] = np.where(candidate, "candidate_station_with_authoritative_bed_support", np.where(bank_only_authoritative, "bank_only_authoritative_support", np.where(support_present, "xs_support_present_but_without_authoritative_bed_support", "missing_authoritative_xs_support")))
    station_audit["true_measured_xs_rejection_reason"] = np.where(
        qualified,
        "qualified_true_measured_xs",
        np.where(
            bank_only_authoritative,
            "authoritative_bank_only_support",
            np.where(
                ~candidate,
                "missing_authoritative_bed_support",
                np.where(
                    ~mixed_or_inner_support,
                    "authoritative_xs_not_inner_or_mixed",
                    np.where(weight <= 0.05, "xs_support_weight_too_low", "unsupported_channel_support_class"),
                ),
            ),
        ),
    )
    for col in ["auth_xs_sample_count_near_station", "auth_xs_min_distance_m", "auth_xs_support_z_m", "auth_xs_support_inner_count", "auth_xs_support_bank_count", "auth_xs_support_class", "bank_only_authoritative", "true_measured_xs_candidate", "true_measured_xs_qualified", "true_measured_xs_candidate_reason", "true_measured_xs_rejection_reason"]:
        frame[col] = station_audit[col].to_numpy()
    station_audit.to_csv(xs_station_audit_path, index=False)
    if frame_path.exists():
        frame_path.unlink()
    frame.to_file(frame_path, driver="GPKG")
    bank_only_qualified_count = int(np.count_nonzero(station_audit['bank_only_authoritative'].to_numpy(dtype=bool) & station_audit['true_measured_xs_qualified'].to_numpy(dtype=bool)))
    xs_handoff_receipt = {
        "authoritative_xs_support_samples_found": int(xs_anchor_count),
        "true_measured_xs_candidate_samples": int(xs_anchor_count if xs_auth is not None else 0),
        "stations_with_true_measured_xs_candidates": int(np.count_nonzero(station_audit["true_measured_xs_candidate"].to_numpy(dtype=bool))),
        "stations_with_true_measured_xs_qualified": int(np.count_nonzero(station_audit["true_measured_xs_qualified"].to_numpy(dtype=bool))),
        "stations_missing_true_measured_xs_candidates": int(len(station_audit) - np.count_nonzero(station_audit["true_measured_xs_candidate"].to_numpy(dtype=bool))),
        "candidate_station_support_class_counts": {str(k): int(v) for k, v in station_audit.loc[station_audit["true_measured_xs_candidate"], "channel_support_class"].astype(str).value_counts().to_dict().items()},
        "qualified_station_support_class_counts": {str(k): int(v) for k, v in station_audit.loc[station_audit["true_measured_xs_qualified"], "channel_support_class"].astype(str).value_counts().to_dict().items()},
        "stations_with_authoritative_xs_support_z": int(np.count_nonzero(np.isfinite(pd.to_numeric(station_audit["auth_xs_support_z_m"], errors="coerce").to_numpy(dtype=float)))),
        "stations_with_authoritative_xs_inner_support": int(np.count_nonzero(pd.to_numeric(station_audit["auth_xs_support_inner_count"], errors="coerce").fillna(0).to_numpy(dtype=int) > 0)),
        "stations_with_authoritative_bank_only_support": int(np.count_nonzero(station_audit["bank_only_authoritative"].to_numpy(dtype=bool))),
        "authoritative_xs_support_class_counts": {str(k): int(v) for k, v in station_audit["auth_xs_support_class"].astype(str).value_counts().to_dict().items()},
        "qualification_rejection_reason_counts": {str(k): int(v) for k, v in station_audit.loc[~station_audit["true_measured_xs_qualified"], "true_measured_xs_rejection_reason"].astype(str).value_counts().to_dict().items()},
        "bank_only_authoritative_station_qualified_count": bank_only_qualified_count,
        "measured_xs_activation_from_bank_only_forbidden": bool(bank_only_qualified_count == 0),
        "artifacts": {
            "measured_xs_audit_support_points": str(xs_audit_points_path) if xs_audit_points_path.exists() else None,
            "measured_xs_station_audit": str(xs_station_audit_path),
        },
    }
    if bank_only_qualified_count > 0:
        xs_handoff_receipt['contract_violation'] = 'bank_only_authoritative_station_qualified_as_true_measured_xs'
    xs_handoff_receipt_path.write_text(json.dumps(xs_handoff_receipt, indent=2), encoding="utf-8")

    station_target_df, station_target_summary = build_station_target_table(frame)
    station_target_outputs = write_station_target_artifacts(
        river_dir=river_dir,
        target_df=station_target_df,
        summary=station_target_summary,
    )
    anchor_policy_df, anchor_policy_summary = build_anchor_policy_table(station_target_df)
    anchor_policy_outputs = write_anchor_policy_artifacts(
        river_dir=river_dir,
        anchor_df=anchor_policy_df,
        summary=anchor_policy_summary,
    )

    # Write scaffold/contract summary.
    per_component: Dict[str, Dict[str, Any]] = {}
    for comp, sub in frame.groupby("component_id", dropna=False):
        station = pd.to_numeric(sub["station_m"], errors="coerce").dropna().sort_values().to_numpy(dtype=float)
        diffs = np.diff(np.unique(np.round(station, 6))) if station.size > 1 else np.asarray([], dtype=float)
        per_component[str(comp)] = {
            "station_count": int(len(sub)),
            "authoritative_anchor_count": int(sub["authoritative_anchor_present"].fillna(False).sum()),
            "xs_supported_count": int(np.count_nonzero(sub["channel_support_class"].astype(str).eq("xs_supported"))),
            "bank_stage_only_count": int(np.count_nonzero(sub["channel_support_class"].astype(str).eq("bank_stage_only"))),
            "station_spacing_median": float(np.nanmedian(diffs)) if diffs.size else None,
            "station_spacing_p95": float(np.nanpercentile(diffs, 95.0)) if diffs.size else None,
        }
    xs_candidate_station_count = int(np.count_nonzero(station_audit["true_measured_xs_candidate"].to_numpy(dtype=bool)))
    contract = {
        "schema_version": 1,
        "artifact_family": "river_channel_frame",
        "notes": {
            "objective": "Authoritative-first channel-frame scaffold for future channel-fitted river-surface solving in (s,n) coordinates.",
            "authoritative_role": "Authoritative in-channel DEM cells are exported as hard channel anchors, not soft raster guidance.",
            "bank_role": "Bank values act as stage/margin priors only and do not override authoritative bed anchors.",
            "legacy_mixed_bed_role": "resolved_channel_bed_z_m is exported only for legacy compatibility when explicitly requested.",
        },
        "metrics": {
            "station_count": int(len(frame)),
            "authoritative_centerline_anchor_count": int(len(auth_centerline)),
            "authoritative_xs_anchor_count": int(xs_anchor_count),
            "xs_candidate_station_count": int(xs_candidate_station_count),
            "xs_qualified_station_count": int(np.count_nonzero(station_audit["true_measured_xs_qualified"].to_numpy(dtype=bool))),
            "component_count": int(frame["component_id"].nunique(dropna=True)),
            "support_class_counts": {str(k): int(v) for k, v in frame["channel_support_class"].astype(str).value_counts().to_dict().items()},
            "backbone_mode_counts": {str(k): int(v) for k, v in frame["backbone_mode"].astype(str).value_counts().to_dict().items()},
            "residual_shape_mode_counts": {str(k): int(v) for k, v in frame["residual_shape_mode"].astype(str).value_counts().to_dict().items()},
            "topology_fields_present": {
                "downstream_component_id": bool(frame["downstream_component_id"].notna().any()),
                "upstream_component_ids": bool(frame["upstream_component_ids"].notna().any()),
                "junction_id": bool(frame["junction_id"].notna().any()),
                "mainstem_rank": bool(pd.to_numeric(frame["mainstem_rank"], errors="coerce").notna().any()),
                "network_order": bool(pd.to_numeric(frame["network_order"], errors="coerce").notna().any()),
                "distance_to_mouth_m": bool(pd.to_numeric(frame["distance_to_mouth_m"], errors="coerce").notna().any()),
            },
            "legacy_mixed_bed_exported": bool(export_legacy_mixed_bed),
            "duplicate_station_rows_collapsed": int(duplicate_station_rows_collapsed),
            "station_target_source_class_counts": station_target_summary.get("target_source_class_counts", {}),
            "station_target_hard_anchor_count": int(station_target_summary.get("hard_anchor_present_count", 0)),
            "anchor_class_counts": anchor_policy_summary.get("anchor_class_counts", {}),
            "anchor_locks_core_count": int(anchor_policy_summary.get("anchor_locks_core_count", 0)),
            "anchor_blocks_rebuild_count": int(anchor_policy_summary.get("anchor_blocks_rebuild_count", 0)),
            "xs_influence_disabled": bool(disable_xs_influence),
        },
        "per_component": per_component,
        "artifacts": {
            "channel_frame_points": str(frame_path),
            "authoritative_centerline_anchors": str(auth_centerline_path) if auth_centerline_path.exists() else None,
            "authoritative_xs_anchors": str(auth_xs_path) if auth_xs_path.exists() else None,
            "measured_xs_audit_support_points": str(xs_audit_points_path) if xs_audit_points_path.exists() else None,
            "measured_xs_station_audit": str(xs_station_audit_path),
            "measured_xs_handoff_receipt": str(xs_handoff_receipt_path),
            "station_targets": station_target_outputs["station_targets"],
            "station_targets_summary": station_target_outputs["station_targets_summary"],
            "anchor_table": anchor_policy_outputs["anchor_table"],
            "anchor_summary": anchor_policy_outputs["anchor_summary"],
        },
    }
    contract_path = river_dir / "river_channel_frame_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (logger or log).info(
        "[RIVER][FRAME] Channel frame built: stations=%d auth_centerline=%d auth_xs=%d components=%d",
        int(len(frame)),
        int(len(auth_centerline)),
        int(xs_anchor_count),
        int(frame["component_id"].nunique(dropna=True)),
    )
    out: Dict[str, str] = {
        "channel_frame_points": str(frame_path),
        "channel_frame_contract": str(contract_path),
    }
    if auth_centerline_path.exists():
        out["authoritative_centerline_anchors"] = str(auth_centerline_path)
    if auth_xs_path.exists():
        out["authoritative_xs_anchors"] = str(auth_xs_path)
    if xs_audit_points_path.exists():
        out["measured_xs_audit_support_points"] = str(xs_audit_points_path)
    out["measured_xs_station_audit"] = str(xs_station_audit_path)
    out["measured_xs_handoff_receipt"] = str(xs_handoff_receipt_path)
    out.update(station_target_outputs)
    out.update(anchor_policy_outputs)
    return out
