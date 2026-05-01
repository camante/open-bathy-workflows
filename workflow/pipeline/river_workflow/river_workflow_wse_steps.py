from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


WSE_PROFILE_CONTRACT_VERSION = "monotone_longitudinal_profile_v4_clean_one_path"


_PREFERRED_WSE_COLUMNS = [
    "point_id", "component_id", "levelpath_id", "reach_id", "source_reach_key",
    "station_m", "station_downstream_m", "station_direction",
    "wse_direction_method", "wse_direction_confidence", "wse_direction_support_delta_m",
    "wse_support_z_m", "support_distance_m", "wse_support_distance_m", "wse_support_count",
    "wse_profile_z_m", "wse_direction", "local_bed_floor_z_m",
    "wse_proxy_raw_interpolated_z_m", "wse_proxy_pre_monotone_z_m", "wse_proxy_z_m",
    "wse_final_floor_conflict", "wse_final_floor_conflict_m",
    "wse_tail_policy", "wse_tail_slope_m_per_m", "wse_tail_extrapolated_count_group",
    "wse_monotone_method", "wse_monotone_trend_used", "wse_monotone_trend_slope_m_per_m",
    "wse_quality_flag", "wse_valid", "wse_group_support_count", "wse_group_support_range_m",
    "wse_group_final_dominant_fraction", "wse_group_flatness_guard_status",
    "wse_group_flatness_fail_reason",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
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


def _series_or_default(frame: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _base_wse_columns(frame: gpd.GeoDataFrame) -> list[str]:
    return [c for c in _PREFERRED_WSE_COLUMNS if c in frame.columns]


def build_wse_support_artifact(out: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Build the explicit WSE support artifact consumed by the WSE trend step."""
    base_cols = _base_wse_columns(out)
    support = out[base_cols + ["geometry"]].copy()
    support["wse_step"] = "01_support"
    support["support_source"] = "canonical_measured_only_authoritative_bank_edge_profile_support"
    support["support_class"] = np.where(
        np.isfinite(pd.to_numeric(support.get("wse_support_z_m"), errors="coerce")),
        "finite_wse_support",
        "missing_wse_support",
    )
    support["rejected"] = False
    support["reject_reason"] = None
    return support


def build_wse_trend_artifact(out: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Build the explicit WSE trend artifact consumed by pre-smoothing."""
    base_cols = _base_wse_columns(out)
    trend = out[base_cols + ["geometry"]].copy()
    trend["wse_step"] = "02_trend"
    trend["wse_trend_z_m"] = _series_or_default(trend, "wse_proxy_raw_interpolated_z_m")
    trend["trend_source"] = "one_path_anchor_residual_monotone_profile"
    trend["downstream_direction_source"] = _series_or_default(trend, "wse_direction_method", "unknown")
    return trend


def build_wse_pre_smooth_artifact(out: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Build the explicit pre-final-monotone WSE artifact."""
    base_cols = _base_wse_columns(out)
    pre_smooth = out[base_cols + ["geometry"]].copy()
    pre_smooth["wse_step"] = "03_pre_smooth"
    pre_smooth["wse_pre_smooth_z_m"] = _series_or_default(pre_smooth, "wse_proxy_pre_monotone_z_m")
    pre_smooth["pre_smooth_source"] = "bounded_residual_profile_before_final_monotone_enforcement"
    return pre_smooth


def build_wse_proxy_final_artifact(out: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Build the final WSE proxy artifact consumed by offset/backbone stages."""
    proxy = out.copy()
    proxy["wse_step"] = "04_proxy_final"
    proxy["wse_artifact_role"] = "final_proxy_consumed_by_offset_stage"
    return proxy


def build_wse_stage_check_records(
    *,
    paths: dict[str, Path],
    science_summary: dict[str, Any],
    group_direction_checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = science_summary.get("raw_wse") or {}
    proxy = science_summary.get("proxy_wse") or {}
    monotone_violations = int(science_summary.get("monotone_violation_count") or 0)
    support_count = int(raw.get("count") or 0)
    proxy_count = int(proxy.get("count") or 0)
    direction_count = int(len(group_direction_checks))
    return [
        {
            "step": "01_support",
            "name": "build_wse_support",
            "input": "canonical_centerline_points + canonical_measured_authoritative_bank_edge_cells",
            "output": str(paths["centerline_wse_support_points"]),
            "check": "finite bank-edge WSE support is counted and missing support is explicit",
            "status": "ok" if support_count > 0 else "review_low_support",
            "finite_support_count": support_count,
        },
        {
            "step": "02_trend",
            "name": "build_wse_trend",
            "input": str(paths["centerline_wse_support_points"]),
            "output": str(paths["centerline_wse_trend_points"]),
            "check": "downstream direction is resolved per WSE profile group",
            "status": "ok" if direction_count > 0 else "failed_no_direction_groups",
            "direction_group_count": direction_count,
        },
        {
            "step": "03_pre_smooth",
            "name": "build_wse_pre_smooth",
            "input": str(paths["centerline_wse_trend_points"]),
            "output": str(paths["centerline_wse_pre_smooth_points"]),
            "check": "bounded residual profile exists before final monotone projection",
            "status": "ok" if proxy_count > 0 else "failed_no_pre_smooth_values",
            "finite_proxy_count": proxy_count,
        },
        {
            "step": "04_proxy_final",
            "name": "build_wse_proxy_final",
            "input": str(paths["centerline_wse_pre_smooth_points"]),
            "output": str(paths["centerline_wse_proxy_points"]),
            "check": "final WSE proxy is finite and non-increasing downstream by group",
            "status": "ok" if proxy_count > 0 and monotone_violations == 0 else "failed_final_proxy_check",
            "finite_proxy_count": proxy_count,
            "monotone_violation_count": monotone_violations,
        },
    ]


def write_wse_stage_contract(
    *,
    reports_dir: Path,
    paths: dict[str, Path],
    science_summary: dict[str, Any],
    group_direction_checks: list[dict[str, Any]],
) -> Path:
    """Write the four-step WSE one-path contract beside retained WSE artifacts."""
    contract_path = Path(reports_dir) / "wse_one_path_stage_contract.json"
    checks = build_wse_stage_check_records(
        paths=paths,
        science_summary=science_summary,
        group_direction_checks=group_direction_checks,
    )
    payload = {
        "schema": "wse_one_path_stage_contract_v2",
        "stage": "centerline_wse_proxy",
        "profile_contract_version": WSE_PROFILE_CONTRACT_VERSION,
        "purpose": "Expose WSE as four explicit implementation steps: support -> trend -> pre_smooth -> proxy_final.",
        "construction_policy": "one_path_no_fallbacks_wse_reference_surface_only",
        "all_required_outputs_declared": all(bool(item.get("output")) for item in checks),
        "failed_steps": [item["name"] for item in checks if str(item.get("status", "")).startswith("failed")],
        "steps": checks,
    }
    _write_json(contract_path, payload)
    txt_path = contract_path.with_suffix(".txt")
    lines = [
        "WSE one-path stage contract",
        "===========================",
        f"schema: {payload['schema']}",
        f"profile_contract_version: {WSE_PROFILE_CONTRACT_VERSION}",
        f"construction_policy: {payload['construction_policy']}",
        "",
        "steps:",
    ]
    for item in checks:
        lines.append(f"  {item['step']} {item['name']}: {item['status']} -> {item['output']}")
        lines.append(f"    check: {item['check']}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return contract_path


__all__ = [
    "WSE_PROFILE_CONTRACT_VERSION",
    "build_wse_support_artifact",
    "build_wse_trend_artifact",
    "build_wse_pre_smooth_artifact",
    "build_wse_proxy_final_artifact",
    "build_wse_stage_check_records",
    "write_wse_stage_contract",
]
