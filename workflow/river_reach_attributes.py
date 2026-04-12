from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    log.debug("river_reach_attributes: suppressed exception", exc_info=True)
    gpd = None


def _infer_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        hit = lower_map.get(str(cand).lower())
        if hit is not None:
            return str(hit)
    raise ValueError(f"Could not find {label} column. Tried {candidates}. Columns={list(df.columns)}")


def _coerce_bool(series: pd.Series | None, length: int) -> pd.Series:
    if series is None:
        return pd.Series(np.zeros(length, dtype=bool))
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(0).astype(bool)


def _safe_mode(series: pd.Series) -> str:
    if series is None or len(series) == 0:
        return "unknown"
    cleaned = series.astype(str)
    if cleaned.empty:
        return "unknown"
    modes = cleaned.mode(dropna=True)
    if len(modes) == 0:
        return "unknown"
    return str(modes.iloc[0])


def _distribution(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def _read_profile(profile_path: Path) -> pd.DataFrame:
    df = pd.read_csv(profile_path)
    if df.empty:
        raise ValueError("river_longitudinal_profile_empty")
    profile_col = _infer_col(df, ["profile_id", "component_id", "levelpathi"], "profile/component id")
    station_col = _infer_col(df, ["station_m", "station"], "station")
    out = df.copy()
    out[profile_col] = out[profile_col].astype(str)
    out[station_col] = pd.to_numeric(out[station_col], errors="coerce")
    out = out.loc[np.isfinite(out[station_col].to_numpy(dtype=float))].copy()
    if out.empty:
        raise ValueError("river_longitudinal_profile_no_finite_station")
    out = out.sort_values([profile_col, station_col]).reset_index(drop=True)
    out["_profile_id"] = out[profile_col].astype(str)
    out["_station_m"] = pd.to_numeric(out[station_col], errors="coerce")
    out["profile_support_class"] = out.get("profile_support_class", pd.Series(["unknown"] * len(out), index=out.index)).astype(str)
    out["profile_support_source_count"] = pd.to_numeric(out.get("profile_support_source_count"), errors="coerce")
    out["profile_support_present"] = _coerce_bool(out.get("profile_support_present"), len(out))
    out["profile_authoritative_anchor_present"] = _coerce_bool(out.get("profile_authoritative_anchor_present"), len(out))
    out["profile_xs_support_present"] = _coerce_bool(out.get("profile_xs_support_present"), len(out))
    out["profile_centerline_support_present"] = _coerce_bool(out.get("profile_centerline_support_present"), len(out))
    out["profile_bank_support_present"] = _coerce_bool(out.get("profile_bank_support_present"), len(out))
    out["profile_wse_support_present"] = _coerce_bool(out.get("profile_wse_support_present"), len(out))
    out["profile_far_from_measured_support"] = _coerce_bool(out.get("profile_far_from_measured_support"), len(out))
    out["profile_measured_support_distance_m"] = pd.to_numeric(out.get("profile_measured_support_distance_m"), errors="coerce")
    out["profile_authoritative_role"] = out.get("profile_authoritative_role", pd.Series(["authoritative_overbank_or_ambiguous"] * len(out), index=out.index)).fillna("authoritative_overbank_or_ambiguous").astype(str)
    out["profile_authoritative_bed_support_present"] = _coerce_bool(out.get("profile_authoritative_bed_support_present"), len(out))
    out["profile_authoritative_bank_margin_present"] = _coerce_bool(out.get("profile_authoritative_bank_margin_present"), len(out))
    out["profile_authoritative_bed_core_present"] = _coerce_bool(out.get("profile_authoritative_bed_core_present"), len(out))
    out["profile_authoritative_ambiguous_present"] = _coerce_bool(out.get("profile_authoritative_ambiguous_present"), len(out))
    out["profile_authoritative_bed_support_distance_m"] = pd.to_numeric(out.get("profile_authoritative_bed_support_distance_m"), errors="coerce")
    out["profile_far_from_authoritative_bed_support"] = _coerce_bool(out.get("profile_far_from_authoritative_bed_support"), len(out))
    out["network_backbone_elevation_m"] = pd.to_numeric(out.get("network_backbone_elevation_m"), errors="coerce")
    out["network_junction_adjustment_m"] = pd.to_numeric(out.get("network_junction_adjustment_m"), errors="coerce")
    out["junction_hierarchy_weight"] = pd.to_numeric(out.get("junction_hierarchy_weight"), errors="coerce")
    out["junction_wse_weight"] = pd.to_numeric(out.get("junction_wse_weight"), errors="coerce")
    out["drainage_area_proxy"] = pd.to_numeric(out.get("drainage_area_proxy"), errors="coerce")
    out["stream_order_proxy"] = pd.to_numeric(out.get("stream_order_proxy"), errors="coerce")
    out["network_backbone_source"] = out.get("network_backbone_source", pd.Series(["unknown"] * len(out), index=out.index)).astype(str)
    return out


def _segment_length_for_profile(stations: np.ndarray, requested_length_m: float) -> float:
    stations = np.asarray(stations, dtype=float)
    stations = stations[np.isfinite(stations)]
    if stations.size < 2:
        return float(max(requested_length_m, 1.0))
    diffs = np.diff(np.sort(stations))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    median_step = float(np.median(diffs)) if diffs.size else 0.0
    return float(max(requested_length_m, median_step * 20.0, 50.0))


def _read_topology_truth(edges_path: Optional[Path], longitudinal_summary: dict[str, Any]) -> dict[str, Any]:
    if edges_path is None or not Path(edges_path).exists():
        return {"available": False, "reason": "hydraulic_backbone_edges_missing"}
    if gpd is None:
        return {"available": False, "reason": "geopandas_unavailable"}
    try:
        edges = gpd.read_file(edges_path)
    except Exception:
        log.debug("Failed to read hydraulic backbone edges %s", edges_path, exc_info=True)
        return {"available": False, "reason": "hydraulic_backbone_edges_read_failed", "path": str(edges_path)}
    if edges.empty:
        return {"available": False, "reason": "hydraulic_backbone_edges_empty", "path": str(edges_path)}

    comp_col = _infer_col(pd.DataFrame(edges.drop(columns=[c for c in ["geometry"] if c in edges.columns])), ["component_id", "profile_id", "levelpathi"], "component id")
    from_col = None
    to_col = None
    for cand in ["from_node", "fromnode", "FromNode", "FROMNODE"]:
        if cand in edges.columns:
            from_col = cand
            break
    for cand in ["to_node", "tonode", "ToNode", "TONODE"]:
        if cand in edges.columns:
            to_col = cand
            break
    component_ids = edges[comp_col].astype(str)
    component_count = int(component_ids.nunique())
    out: dict[str, Any] = {
        "available": True,
        "path": str(edges_path),
        "component_count": component_count,
        "topology_has_explicit_nodes": bool(from_col and to_col),
        "topology_source": "explicit_nodes" if (from_col and to_col) else "geometry_only",
        "junction_node_count": 0,
        "junction_component_count": 0,
        "component_junction_counts": {},
    }
    if from_col and to_col:
        node_df = pd.DataFrame({
            "node_id": pd.concat([edges[from_col], edges[to_col]], axis=0).astype(str),
            "component_id": pd.concat([component_ids, component_ids], axis=0).astype(str),
        })
        node_df = node_df.loc[node_df["node_id"].notna() & (node_df["node_id"].astype(str) != "")].copy()
        if not node_df.empty:
            grouped = node_df.groupby("node_id")
            junction_components: set[str] = set()
            component_junction_counts: dict[str, int] = {}
            junction_nodes = 0
            for node_id, sub in grouped:
                components_here = set(sub["component_id"].astype(str))
                node_degree = int(len(sub))
                is_junction = (len(components_here) > 1) or (node_degree > 2)
                if not is_junction:
                    continue
                junction_nodes += 1
                for comp in components_here:
                    junction_components.add(comp)
                    component_junction_counts[comp] = component_junction_counts.get(comp, 0) + 1
            out["junction_node_count"] = int(junction_nodes)
            out["junction_component_count"] = int(len(junction_components))
            out["component_junction_counts"] = {str(k): int(v) for k, v in sorted(component_junction_counts.items())}

    longitudinal_junction_count = int(longitudinal_summary.get("junction_count", 0) or 0) if isinstance(longitudinal_summary, dict) else 0
    longitudinal_component_count = int(longitudinal_summary.get("network_component_count", 0) or 0) if isinstance(longitudinal_summary, dict) else 0
    out["longitudinal_summary_component_count"] = longitudinal_component_count
    out["longitudinal_summary_junction_count"] = longitudinal_junction_count
    out["component_count_matches_longitudinal_summary"] = bool(longitudinal_component_count == component_count) if longitudinal_component_count > 0 else False
    out["junction_count_matches_longitudinal_summary"] = bool(longitudinal_junction_count == out["junction_node_count"]) if longitudinal_junction_count >= 0 else False
    out["topology_truth_warning"] = None
    if out["junction_node_count"] > 0 and longitudinal_junction_count == 0:
        out["topology_truth_warning"] = "graph_edges_show_junctions_but_longitudinal_summary_reports_zero"
    elif longitudinal_junction_count > 0 and out["junction_node_count"] == 0:
        out["topology_truth_warning"] = "longitudinal_summary_reports_junctions_but_graph_edges_show_zero"
    return out


def build_and_write_reach_attributes(
    *,
    river_dir: str | Path,
    longitudinal_profile_path: str | Path | None,
    longitudinal_profile_summary_path: str | Path | None = None,
    hydraulic_backbone_edges_path: str | Path | None = None,
    segment_length_m: float = 500.0,
) -> dict[str, str]:
    river_dir = Path(river_dir)
    if longitudinal_profile_path is None or not Path(longitudinal_profile_path).exists():
        raise FileNotFoundError("longitudinal_profile_required_for_reach_attributes")
    profile = _read_profile(Path(longitudinal_profile_path))

    longitudinal_summary: dict[str, Any] = {}
    if longitudinal_profile_summary_path is not None and Path(longitudinal_profile_summary_path).exists():
        try:
            longitudinal_summary = json.loads(Path(longitudinal_profile_summary_path).read_text(encoding="utf-8"))
        except Exception:
            log.debug("Failed to read longitudinal profile summary %s", longitudinal_profile_summary_path, exc_info=True)
            longitudinal_summary = {}

    topology_truth = _read_topology_truth(
        Path(hydraulic_backbone_edges_path) if hydraulic_backbone_edges_path is not None else None,
        longitudinal_summary,
    )
    component_junction_counts = topology_truth.get("component_junction_counts", {}) if isinstance(topology_truth, dict) else {}

    rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for profile_id, sub in profile.groupby("_profile_id", dropna=False):
        sub = sub.sort_values("_station_m").copy()
        stations = sub["_station_m"].to_numpy(dtype=float)
        if stations.size == 0:
            continue
        seg_len = _segment_length_for_profile(stations, requested_length_m=float(segment_length_m))
        start_bins = np.floor(stations / seg_len).astype(int)
        sub["reach_bin_index"] = start_bins
        sub["reach_station_min_m"] = start_bins * seg_len
        sub["reach_station_max_m"] = (start_bins + 1) * seg_len
        component_has_junction = int(component_junction_counts.get(str(profile_id), 0)) > 0

        component_rows.append({
            "profile_id": str(profile_id),
            "station_min_m": float(np.nanmin(stations)),
            "station_max_m": float(np.nanmax(stations)),
            "station_span_m": float(np.nanmax(stations) - np.nanmin(stations)) if stations.size else 0.0,
            "point_count": int(len(sub)),
            "segment_length_m": float(seg_len),
            "authoritative_anchor_fraction": float(sub["profile_authoritative_anchor_present"].mean()),
            "xs_support_fraction": float(sub["profile_xs_support_present"].mean()),
            "supported_fraction": float(sub["profile_support_present"].mean()),
            "measured_support_distance_median_m": float(np.nanmedian(sub["profile_measured_support_distance_m"].to_numpy(dtype=float))) if np.any(np.isfinite(sub["profile_measured_support_distance_m"].to_numpy(dtype=float))) else float("nan"),
            "measured_support_distance_min_m": float(np.nanmin(sub["profile_measured_support_distance_m"].to_numpy(dtype=float))) if np.any(np.isfinite(sub["profile_measured_support_distance_m"].to_numpy(dtype=float))) else float("nan"),
            "far_from_measured_fraction": float(sub["profile_far_from_measured_support"].mean()),
            "authoritative_bed_support_fraction": float(sub["profile_authoritative_bed_support_present"].mean()),
            "authoritative_bank_margin_fraction": float(sub["profile_authoritative_bank_margin_present"].mean()),
            "authoritative_bed_core_fraction": float(sub["profile_authoritative_bed_core_present"].mean()),
            "authoritative_ambiguous_fraction": float(sub["profile_authoritative_ambiguous_present"].mean()),
            "authoritative_bed_support_distance_median_m": float(np.nanmedian(sub["profile_authoritative_bed_support_distance_m"].to_numpy(dtype=float))) if np.any(np.isfinite(sub["profile_authoritative_bed_support_distance_m"].to_numpy(dtype=float))) else float("nan"),
            "authoritative_bed_support_distance_min_m": float(np.nanmin(sub["profile_authoritative_bed_support_distance_m"].to_numpy(dtype=float))) if np.any(np.isfinite(sub["profile_authoritative_bed_support_distance_m"].to_numpy(dtype=float))) else float("nan"),
            "far_from_authoritative_bed_fraction": float(sub["profile_far_from_authoritative_bed_support"].mean()),
            "primary_support_class": _safe_mode(sub["profile_support_class"]),
            "primary_authoritative_role": _safe_mode(sub["profile_authoritative_role"]),
            "component_junction_count": int(component_junction_counts.get(str(profile_id), 0)),
            "component_has_topology_junction": bool(component_has_junction),
            "drainage_area_proxy_median": float(np.nanmedian(sub["drainage_area_proxy"].to_numpy(dtype=float))) if np.any(np.isfinite(sub["drainage_area_proxy"].to_numpy(dtype=float))) else float("nan"),
            "stream_order_proxy_median": float(np.nanmedian(sub["stream_order_proxy"].to_numpy(dtype=float))) if np.any(np.isfinite(sub["stream_order_proxy"].to_numpy(dtype=float))) else float("nan"),
        })

        for reach_bin, seg in sub.groupby("reach_bin_index", dropna=False):
            seg = seg.sort_values("_station_m").copy()
            seg_stations = seg["_station_m"].to_numpy(dtype=float)
            seg_bed = seg["network_backbone_elevation_m"].to_numpy(dtype=float)
            station_diffs = np.diff(seg_stations) if seg_stations.size >= 2 else np.array([], dtype=float)
            bed_diffs = np.diff(seg_bed) if seg_bed.size >= 2 else np.array([], dtype=float)
            valid_slope = np.isfinite(station_diffs) & (station_diffs != 0.0) & np.isfinite(bed_diffs)
            slopes = np.divide(bed_diffs[valid_slope], station_diffs[valid_slope], out=np.full(np.count_nonzero(valid_slope), np.nan, dtype=float), where=station_diffs[valid_slope] != 0.0) if np.any(valid_slope) else np.array([], dtype=float)
            junction_adjustment = np.abs(pd.to_numeric(seg["network_junction_adjustment_m"], errors="coerce").to_numpy(dtype=float))
            reach_station_min = float(np.nanmin(seg_stations))
            reach_station_max = float(np.nanmax(seg_stations))
            point_count = int(len(seg))
            reach_role = "interior"
            if np.isclose(reach_station_min, float(np.nanmin(stations))):
                reach_role = "upstream_window"
            if np.isclose(reach_station_max, float(np.nanmax(stations))):
                reach_role = "downstream_window" if reach_role == "interior" else "endpoint_window"
            if np.any(np.isfinite(junction_adjustment) & (junction_adjustment > 1.0e-6)):
                reach_role = "junction_adjusted"
            elif component_has_junction and (reach_role in {"upstream_window", "downstream_window", "endpoint_window"}):
                reach_role = "junction_endpoint_window"

            support_classes = seg["profile_support_class"].astype(str)
            rows.append({
                "profile_id": str(profile_id),
                "reach_bin_index": int(reach_bin),
                "reach_id": f"{profile_id}:{int(reach_bin)}",
                "reach_role": reach_role,
                "station_min_m": reach_station_min,
                "station_max_m": reach_station_max,
                "station_span_m": float(reach_station_max - reach_station_min) if point_count else 0.0,
                "point_count": point_count,
                "segment_length_m": float(seg_len),
                "station_step_median_m": float(np.nanmedian(station_diffs)) if station_diffs.size else float("nan"),
                "supported_fraction": float(seg["profile_support_present"].mean()),
                "unsupported_fraction": float((~seg["profile_support_present"]).mean()),
                "authoritative_anchor_fraction": float(seg["profile_authoritative_anchor_present"].mean()),
                "xs_support_fraction": float(seg["profile_xs_support_present"].mean()),
                "centerline_support_fraction": float(seg["profile_centerline_support_present"].mean()),
                "bank_support_fraction": float(seg["profile_bank_support_present"].mean()),
                "wse_support_fraction": float(seg["profile_wse_support_present"].mean()),
                "measured_support_distance_median_m": float(np.nanmedian(seg["profile_measured_support_distance_m"].to_numpy(dtype=float))) if np.any(np.isfinite(seg["profile_measured_support_distance_m"].to_numpy(dtype=float))) else float("nan"),
                "measured_support_distance_min_m": float(np.nanmin(seg["profile_measured_support_distance_m"].to_numpy(dtype=float))) if np.any(np.isfinite(seg["profile_measured_support_distance_m"].to_numpy(dtype=float))) else float("nan"),
                "far_from_measured_fraction": float(seg["profile_far_from_measured_support"].mean()),
                "authoritative_bed_support_fraction": float(seg["profile_authoritative_bed_support_present"].mean()),
                "authoritative_bank_margin_fraction": float(seg["profile_authoritative_bank_margin_present"].mean()),
                "authoritative_bed_core_fraction": float(seg["profile_authoritative_bed_core_present"].mean()),
                "authoritative_ambiguous_fraction": float(seg["profile_authoritative_ambiguous_present"].mean()),
                "authoritative_bed_support_distance_median_m": float(np.nanmedian(seg["profile_authoritative_bed_support_distance_m"].to_numpy(dtype=float))) if np.any(np.isfinite(seg["profile_authoritative_bed_support_distance_m"].to_numpy(dtype=float))) else float("nan"),
                "authoritative_bed_support_distance_min_m": float(np.nanmin(seg["profile_authoritative_bed_support_distance_m"].to_numpy(dtype=float))) if np.any(np.isfinite(seg["profile_authoritative_bed_support_distance_m"].to_numpy(dtype=float))) else float("nan"),
                "far_from_authoritative_bed_fraction": float(seg["profile_far_from_authoritative_bed_support"].mean()),
                "support_source_count_mean": float(np.nanmean(seg["profile_support_source_count"].to_numpy(dtype=float))) if np.any(np.isfinite(seg["profile_support_source_count"].to_numpy(dtype=float))) else float("nan"),
                "support_source_count_max": float(np.nanmax(seg["profile_support_source_count"].to_numpy(dtype=float))) if np.any(np.isfinite(seg["profile_support_source_count"].to_numpy(dtype=float))) else float("nan"),
                "primary_support_class": _safe_mode(support_classes),
                "primary_authoritative_role": _safe_mode(seg["profile_authoritative_role"]),
                "support_class_count_authoritative": int(np.count_nonzero(support_classes == "authoritative_anchor")),
                "support_class_count_xs": int(np.count_nonzero(support_classes.str.contains("xs", case=False, na=False))),
                "component_has_topology_junction": bool(component_has_junction),
                "component_junction_count": int(component_junction_counts.get(str(profile_id), 0)),
                "junction_adjustment_station_fraction": float(np.mean(np.isfinite(junction_adjustment) & (junction_adjustment > 1.0e-6))),
                "junction_adjustment_abs_p95_m": float(np.percentile(junction_adjustment[np.isfinite(junction_adjustment)], 95.0)) if np.any(np.isfinite(junction_adjustment)) else float("nan"),
                "network_backbone_source_mode": _safe_mode(seg["network_backbone_source"]),
                "network_backbone_slope_m_per_m": float(np.nanmedian(slopes)) if slopes.size else float("nan"),
                "network_backbone_bed_range_m": float(np.nanmax(seg_bed) - np.nanmin(seg_bed)) if np.any(np.isfinite(seg_bed)) else float("nan"),
                "drainage_area_proxy_median": float(np.nanmedian(seg["drainage_area_proxy"].to_numpy(dtype=float))) if np.any(np.isfinite(seg["drainage_area_proxy"].to_numpy(dtype=float))) else float("nan"),
                "stream_order_proxy_median": float(np.nanmedian(seg["stream_order_proxy"].to_numpy(dtype=float))) if np.any(np.isfinite(seg["stream_order_proxy"].to_numpy(dtype=float))) else float("nan"),
            })

    reach_df = pd.DataFrame(rows)
    component_df = pd.DataFrame(component_rows)
    out_csv = river_dir / "river_reach_attributes.csv"
    reach_df.to_csv(out_csv, index=False)
    out_component_csv = river_dir / "river_reach_components.csv"
    component_df.to_csv(out_component_csv, index=False)

    junction_dist = _distribution(reach_df.get("junction_adjustment_abs_p95_m", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float))
    summary = {
        "available": True,
        "segment_length_m_default": float(segment_length_m),
        "component_count": int(component_df["profile_id"].nunique()) if not component_df.empty else 0,
        "reach_count": int(len(reach_df)),
        "reach_role_counts": {str(k): int(v) for k, v in reach_df.get("reach_role", pd.Series(dtype=str)).value_counts(dropna=False).to_dict().items()} if not reach_df.empty else {},
        "primary_support_class_counts": {str(k): int(v) for k, v in reach_df.get("primary_support_class", pd.Series(dtype=str)).value_counts(dropna=False).to_dict().items()} if not reach_df.empty else {},
        "authoritative_anchor_fraction_summary": _distribution(reach_df.get("authoritative_anchor_fraction", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "xs_support_fraction_summary": _distribution(reach_df.get("xs_support_fraction", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "supported_fraction_summary": _distribution(reach_df.get("supported_fraction", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "measured_support_distance_median_summary": _distribution(reach_df.get("measured_support_distance_median_m", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "measured_support_distance_min_summary": _distribution(reach_df.get("measured_support_distance_min_m", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "far_from_measured_fraction_summary": _distribution(reach_df.get("far_from_measured_fraction", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "authoritative_bed_support_fraction_summary": _distribution(reach_df.get("authoritative_bed_support_fraction", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "authoritative_bank_margin_fraction_summary": _distribution(reach_df.get("authoritative_bank_margin_fraction", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "authoritative_bed_core_fraction_summary": _distribution(reach_df.get("authoritative_bed_core_fraction", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "authoritative_bed_support_distance_median_summary": _distribution(reach_df.get("authoritative_bed_support_distance_median_m", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "authoritative_bed_support_distance_min_summary": _distribution(reach_df.get("authoritative_bed_support_distance_min_m", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "far_from_authoritative_bed_fraction_summary": _distribution(reach_df.get("far_from_authoritative_bed_fraction", pd.Series(dtype=float)).to_numpy(dtype=float)) if not reach_df.empty else _distribution(np.array([], dtype=float)),
        "primary_authoritative_role_counts": {str(k): int(v) for k, v in reach_df.get("primary_authoritative_role", pd.Series(dtype=str)).value_counts(dropna=False).to_dict().items()} if not reach_df.empty else {},
        "junction_adjustment_abs_p95_summary": junction_dist,
        "topology_truth": topology_truth,
        "inputs": {
            "longitudinal_profile": str(longitudinal_profile_path),
            "longitudinal_profile_summary": str(longitudinal_profile_summary_path) if longitudinal_profile_summary_path else None,
            "hydraulic_backbone_edges": str(hydraulic_backbone_edges_path) if hydraulic_backbone_edges_path else None,
        },
        "outputs": {
            "reach_attributes_csv": str(out_csv),
            "reach_components_csv": str(out_component_csv),
        },
    }
    out_summary = river_dir / "river_reach_attributes_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "reach_attributes": str(out_csv),
        "reach_components": str(out_component_csv),
        "reach_attributes_summary": str(out_summary),
    }


__all__ = ["build_and_write_reach_attributes"]
