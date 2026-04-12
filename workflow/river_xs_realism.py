from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from river_longitudinal_tendency import (
    _attach_reach_context,
    _distribution,
    _is_authoritative_like,
    _is_indirect_xs,
    _is_measured_xs,
    _is_residual_xs,
    _is_xs_like,
    _station_provenance_row,
    _prepare_reach_attributes,
    _read_optional_csv,
)
from river_support_semantics import station_semantics

log = logging.getLogger(__name__)


def _role_value(station_rows: pd.DataFrame, role: str, column: str = "bed_z_m") -> float:
    sub = station_rows.loc[station_rows["node_role"].astype(str).eq(role)]
    if sub.empty:
        return float("nan")
    vals = pd.to_numeric(sub[column], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.nanmedian(vals))


def _ratio(inner_z: float, bank_z: float, thalweg_z: float) -> float:
    if not (np.isfinite(inner_z) and np.isfinite(bank_z) and np.isfinite(thalweg_z)):
        return float("nan")
    denom = bank_z - thalweg_z
    if not np.isfinite(denom) or abs(denom) < 1.0e-6:
        return float("nan")
    return float((inner_z - thalweg_z) / denom)


def _window_radius_from_row(row: pd.Series) -> int:
    unsupported_fraction = float(pd.to_numeric(row.get("unsupported_fraction"), errors="coerce") if row.get("unsupported_fraction") is not None else 0.0)
    xs_fraction = float(pd.to_numeric(row.get("xs_support_fraction"), errors="coerce") if row.get("xs_support_fraction") is not None else 0.0)
    authoritative_fraction = float(pd.to_numeric(row.get("authoritative_anchor_fraction"), errors="coerce") if row.get("authoritative_anchor_fraction") is not None else 0.0)
    junction_fraction = float(pd.to_numeric(row.get("junction_adjustment_station_fraction"), errors="coerce") if row.get("junction_adjustment_station_fraction") is not None else 0.0)
    role = str(row.get("reach_role", "interior") or "interior")
    radius = 1.5 + 2.0 * max(unsupported_fraction, 0.0) + 0.75 * max(1.0 - xs_fraction, 0.0)
    radius -= 0.75 * max(authoritative_fraction, 0.0)
    if "junction" in role:
        radius -= 0.5
    if junction_fraction > 0.0:
        radius -= 0.5 * min(junction_fraction, 1.0)
    return int(np.clip(np.round(radius), 1, 5))


def _support_distance_damping_factor(row: pd.Series) -> float:
    support_dist = pd.to_numeric(row.get("station_authoritative_bed_support_distance_m"), errors="coerce")
    support_dist = float(support_dist) if np.isfinite(support_dist) else np.nan
    if not np.isfinite(support_dist):
        return 1.0
    if support_dist >= 1000.0:
        return 0.35
    if support_dist >= 750.0:
        return 0.45
    if support_dist >= 500.0:
        return 0.55
    if support_dist >= 300.0:
        return 0.72
    if support_dist >= 150.0:
        return 0.88
    return 1.0


def _station_core_influence_scale(row: pd.Series, *, station_core_authoritative: bool, station_measured_xs: bool = False) -> float:
    if station_core_authoritative or station_measured_xs:
        return 1.0
    support_class = str(row.get("station_support_regime", row.get("xs_support_template_class", "missing")) or "missing")
    bed_support_class = str(row.get("station_authoritative_bed_support_class", "no_authoritative_bed_support") or "no_authoritative_bed_support")
    true_measured_fraction = float(pd.to_numeric(row.get("station_true_measured_xs_fraction"), errors="coerce") if row.get("station_true_measured_xs_fraction") is not None else 0.0)
    indirect_fraction = float(pd.to_numeric(row.get("station_indirect_xs_fraction"), errors="coerce") if row.get("station_indirect_xs_fraction") is not None else 0.0)
    residual_fraction = float(pd.to_numeric(row.get("station_residual_xs_fraction"), errors="coerce") if row.get("station_residual_xs_fraction") is not None else 0.0)
    scale = 1.0
    if bed_support_class == "authoritative_bank_margin_only" or support_class in {"bank_only_low_confidence", "bank_only_authoritative"}:
        scale = 0.45
    elif support_class in {"supported_transition", "bank_supported_with_good_longitudinal_context", "authoritative_nearby_but_not_measured_xs"}:
        scale = 0.70
    if true_measured_fraction <= 0.10 and max(indirect_fraction, residual_fraction) >= 0.25:
        scale *= 0.85
    scale *= _support_distance_damping_factor(row)
    return float(np.clip(scale, 0.15, 1.0))



def _canonical_station_target_ratios(row: pd.Series) -> tuple[float, float]:
    if not bool(row.get("station_target_present", False)):
        return float("nan"), float("nan")
    left_bank = pd.to_numeric(row.get("target_left_bank_z_m"), errors="coerce")
    right_bank = pd.to_numeric(row.get("target_right_bank_z_m"), errors="coerce")
    left_inner = pd.to_numeric(row.get("target_left_inner_z_m"), errors="coerce")
    right_inner = pd.to_numeric(row.get("target_right_inner_z_m"), errors="coerce")
    thalweg = pd.to_numeric(row.get("target_thalweg_z_m"), errors="coerce")
    left_ratio = _ratio(float(left_inner), float(left_bank), float(thalweg)) if np.isfinite(left_inner) else float("nan")
    right_ratio = _ratio(float(right_inner), float(right_bank), float(thalweg)) if np.isfinite(right_inner) else float("nan")
    if np.isfinite(left_ratio):
        left_ratio = float(np.clip(left_ratio, 0.05, 0.95))
    if np.isfinite(right_ratio):
        right_ratio = float(np.clip(right_ratio, -0.95, -0.05))
    return left_ratio, right_ratio


def _fitted_section_target_ratios(row: pd.Series) -> tuple[float, float]:
    target_left_ratio, target_right_ratio = _canonical_station_target_ratios(row)
    if np.isfinite(target_left_ratio) and np.isfinite(target_right_ratio):
        return target_left_ratio, target_right_ratio
    left_bank_fit = pd.to_numeric(row.get("left_bank_fit_z_m"), errors="coerce")
    right_bank_fit = pd.to_numeric(row.get("right_bank_fit_z_m"), errors="coerce")
    core_fit = pd.to_numeric(row.get("active_core_fit_z_m"), errors="coerce")
    left_inner_fit = pd.to_numeric(row.get("left_inner_fit_z_m"), errors="coerce")
    right_inner_fit = pd.to_numeric(row.get("right_inner_fit_z_m"), errors="coerce")
    if np.isfinite(left_inner_fit):
        left_ratio = _ratio(float(left_inner_fit), float(left_bank_fit), float(core_fit))
    else:
        left_ratio = float("nan")
    if np.isfinite(right_inner_fit):
        right_ratio = _ratio(float(right_inner_fit), float(right_bank_fit), float(core_fit))
    else:
        right_ratio = float("nan")
    if not np.isfinite(left_ratio) and np.isfinite(left_bank_fit) and np.isfinite(core_fit):
        left_ratio = 0.45
    if not np.isfinite(right_ratio) and np.isfinite(right_bank_fit) and np.isfinite(core_fit):
        right_ratio = -0.45
    if np.isfinite(left_ratio):
        left_ratio = float(np.clip(left_ratio, 0.05, 0.95))
    if np.isfinite(right_ratio):
        right_ratio = float(np.clip(right_ratio, -0.95, -0.05))
    return left_ratio, right_ratio

def _blend_weight_from_row(row: pd.Series, *, station_core_authoritative: bool, station_xs: bool, station_measured_xs: bool = False) -> float:
    if station_core_authoritative:
        return 0.0
    if station_measured_xs:
        return 0.0
    unsupported_fraction = float(pd.to_numeric(row.get("unsupported_fraction"), errors="coerce") if row.get("unsupported_fraction") is not None else 0.0)
    xs_fraction = float(pd.to_numeric(row.get("xs_support_fraction"), errors="coerce") if row.get("xs_support_fraction") is not None else 0.0)
    authoritative_fraction = float(pd.to_numeric(row.get("authoritative_anchor_fraction"), errors="coerce") if row.get("authoritative_anchor_fraction") is not None else 0.0)
    junction_fraction = float(pd.to_numeric(row.get("junction_adjustment_station_fraction"), errors="coerce") if row.get("junction_adjustment_station_fraction") is not None else 0.0)
    role = str(row.get("reach_role", "interior") or "interior")
    blend = 0.15 + 0.45 * max(unsupported_fraction, 0.0) + 0.10 * max(1.0 - xs_fraction, 0.0)
    blend -= 0.30 * max(authoritative_fraction, 0.0)
    # Mild reduction for XS-informed (but not measured) stations
    if station_xs:
        blend *= 0.55
    residual_fraction = float(pd.to_numeric(row.get("station_residual_xs_fraction"), errors="coerce") if row.get("station_residual_xs_fraction") is not None else 0.0)
    indirect_fraction = float(pd.to_numeric(row.get("station_indirect_xs_fraction"), errors="coerce") if row.get("station_indirect_xs_fraction") is not None else 0.0)
    if max(indirect_fraction, residual_fraction) >= 0.25 and not station_measured_xs:
        blend *= 0.70
    blend *= _support_distance_damping_factor(row)
    blend *= _station_core_influence_scale(
        row,
        station_core_authoritative=station_core_authoritative,
        station_measured_xs=station_measured_xs,
    )
    if "junction" in role:
        blend *= 0.65
    if junction_fraction > 0.0:
        blend *= max(0.45, 1.0 - 0.50 * min(junction_fraction, 1.0))
    return float(np.clip(blend, 0.0, 0.55))



def _classify_xs_support_template(row: pd.Series, *, station_core_authoritative: bool, station_xs: bool, station_measured_xs: bool) -> tuple[str, str]:
    semantics = station_semantics(
        row,
        station_authoritative=station_core_authoritative,
        station_measured_xs=station_measured_xs,
    )
    return str(semantics.get("station_support_regime", "bank_supported_with_good_longitudinal_context")), str(semantics.get("station_template_type", "generic_u"))


def _station_node_protection_flags(station_group: pd.DataFrame) -> dict[str, bool]:
    roles = station_group.get("node_role", pd.Series([], dtype="object")).astype(str)
    authoritative = np.asarray(_is_authoritative_like(station_group), dtype=bool)
    measured = np.asarray(_is_measured_xs(station_group), dtype=bool)
    bank_mask = roles.isin(["left_bank", "right_bank"]).to_numpy(dtype=bool)
    channel_mask = roles.isin(["thalweg", "left_inner", "right_inner"]).to_numpy(dtype=bool)
    return {
        "station_bank_protected": bool(np.any(authoritative & bank_mask) or np.any(measured & bank_mask)),
        "station_channel_protected": bool(np.any(authoritative & channel_mask) or np.any(measured & channel_mask)),
        "station_inner_rebuildable": bool(np.any((~authoritative) & (~measured) & channel_mask)),
    }


def _enforce_generic_template_targets(
    station_df: pd.DataFrame,
    *,
    left_after: np.ndarray,
    right_after: np.ndarray,
    left_target_ratio: np.ndarray,
    right_target_ratio: np.ndarray,
    left_blend: np.ndarray,
    right_blend: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_after = np.asarray(left_after, dtype=np.float32).copy()
    right_after = np.asarray(right_after, dtype=np.float32).copy()
    left_target_ratio = np.asarray(left_target_ratio, dtype=np.float32).copy()
    right_target_ratio = np.asarray(right_target_ratio, dtype=np.float32).copy()
    left_blend = np.asarray(left_blend, dtype=np.float32).copy()
    right_blend = np.asarray(right_blend, dtype=np.float32).copy()

    left_vals = pd.to_numeric(station_df.get("left_inner_ratio_before"), errors="coerce").to_numpy(dtype=float)
    right_vals = pd.to_numeric(station_df.get("right_inner_ratio_before"), errors="coerce").to_numpy(dtype=float)
    left_mag_global = np.abs(left_vals[np.isfinite(left_vals)])
    right_mag_global = np.abs(right_vals[np.isfinite(right_vals)])
    global_mag = np.concatenate([left_mag_global, right_mag_global])
    global_mag_ref = float(np.nanmedian(global_mag)) if global_mag.size else 0.5
    global_mag_ref = float(np.clip(global_mag_ref, 0.20, 0.85))

    template_type = station_df.get("xs_template_type", pd.Series(["measured"] * len(station_df), index=station_df.index)).astype(str)
    for idx, tpl in enumerate(template_type.tolist()):
        if tpl not in {"generic_symmetric", "generic_u"}:
            continue
        station_core_authoritative = bool(station_df.iloc[idx].get("station_core_authoritative", False))
        station_measured_xs = bool(station_df.iloc[idx].get("station_measured_xs", False))
        if station_core_authoritative or station_measured_xs:
            continue
        lo = max(0, idx - 2)
        hi = min(len(station_df), idx + 3)
        local_left = left_vals[lo:hi]
        local_right = right_vals[lo:hi]
        local_mag = np.concatenate([np.abs(local_left[np.isfinite(local_left)]), np.abs(local_right[np.isfinite(local_right)])])
        magnitude = float(np.nanmedian(local_mag)) if local_mag.size else global_mag_ref
        if not np.isfinite(magnitude):
            magnitude = global_mag_ref
        influence_scale = _station_core_influence_scale(
            station_df.iloc[idx],
            station_core_authoritative=station_core_authoritative,
            station_measured_xs=station_measured_xs,
        )
        if tpl == "generic_symmetric":
            magnitude = float(np.clip(magnitude, 0.25, 0.70))
            magnitude = float(np.clip(magnitude * max(influence_scale, 0.40), 0.12, 0.70))
            target_left = magnitude
            target_right = -magnitude
            blend_floor = 0.45 * max(influence_scale, 0.35)
        else:
            magnitude = float(np.clip(magnitude, 0.20, 0.55))
            magnitude = float(np.clip(magnitude * max(influence_scale, 0.40), 0.10, 0.55))
            target_left = float(np.clip(0.90 * magnitude, 0.10, 0.60))
            target_right = float(-np.clip(0.90 * magnitude, 0.10, 0.60))
            blend_floor = 0.30 * max(influence_scale, 0.35)
        left_target_ratio[idx] = np.float32(target_left)
        right_target_ratio[idx] = np.float32(target_right)
        left_after[idx] = np.float32(target_left)
        right_after[idx] = np.float32(target_right)
        left_blend[idx] = np.float32(max(float(left_blend[idx]), blend_floor))
        right_blend[idx] = np.float32(max(float(right_blend[idx]), blend_floor))
    return left_after, right_after, left_target_ratio, right_target_ratio, left_blend, right_blend


def _enforce_fitted_section_targets(
    station_df: pd.DataFrame,
    *,
    left_after: np.ndarray,
    right_after: np.ndarray,
    left_target_ratio: np.ndarray,
    right_target_ratio: np.ndarray,
    left_blend: np.ndarray,
    right_blend: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_after = np.asarray(left_after, dtype=np.float32).copy()
    right_after = np.asarray(right_after, dtype=np.float32).copy()
    left_target_ratio = np.asarray(left_target_ratio, dtype=np.float32).copy()
    right_target_ratio = np.asarray(right_target_ratio, dtype=np.float32).copy()
    left_blend = np.asarray(left_blend, dtype=np.float32).copy()
    right_blend = np.asarray(right_blend, dtype=np.float32).copy()
    for idx, row in station_df.iterrows():
        station_core_authoritative = bool(row.get("station_core_authoritative", False))
        station_measured_xs = bool(row.get("station_measured_xs", False))
        if station_core_authoritative or station_measured_xs:
            continue
        station_target_present = bool(row.get("station_target_present", False))
        target_xs_realism_allowed = bool(row.get("target_xs_realism_allowed", True))
        support_class = str(row.get("xs_support_template_class", row.get("station_support_regime", "missing")) or "missing")
        if station_target_present and (not target_xs_realism_allowed):
            left_ratio, right_ratio = _canonical_station_target_ratios(row)
            if np.isfinite(left_ratio) and np.isfinite(right_ratio):
                left_target_ratio[idx] = np.float32(left_ratio)
                right_target_ratio[idx] = np.float32(right_ratio)
                left_after[idx] = np.float32(left_ratio)
                right_after[idx] = np.float32(right_ratio)
                left_blend[idx] = np.float32(0.0)
                right_blend[idx] = np.float32(0.0)
            continue
        if support_class not in {
            "bank_only_low_confidence",
            "bank_only_authoritative",
            "supported_transition",
            "authoritative_nearby_but_not_measured_xs",
            "bank_supported_with_good_longitudinal_context",
        } and not station_target_present:
            continue
        left_ratio, right_ratio = _fitted_section_target_ratios(row)
        if not (np.isfinite(left_ratio) and np.isfinite(right_ratio)):
            continue
        influence_scale = _station_core_influence_scale(
            row,
            station_core_authoritative=station_core_authoritative,
            station_measured_xs=station_measured_xs,
        )
        blend_floor = 0.60 if support_class.startswith("bank_only") else 0.45
        blend_floor *= max(influence_scale, 0.35)
        left_target_ratio[idx] = np.float32(left_ratio)
        right_target_ratio[idx] = np.float32(right_ratio)
        left_after[idx] = np.float32(left_ratio)
        right_after[idx] = np.float32(right_ratio)
        left_blend[idx] = np.float32(max(float(left_blend[idx]), blend_floor))
        right_blend[idx] = np.float32(max(float(right_blend[idx]), blend_floor))
    return left_after, right_after, left_target_ratio, right_target_ratio, left_blend, right_blend

def _smooth_ratio_profile(
    stations: np.ndarray,
    ratios: np.ndarray,
    *,
    protected_mask: np.ndarray,
    station_xs_mask: np.ndarray,
    station_core_authoritative_mask: np.ndarray | None = None,
    station_measured_xs_mask: np.ndarray | None = None,
    context_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = np.asarray(ratios, dtype=float)
    out = vals.copy()
    blend = np.zeros(vals.shape, dtype=np.float32)
    target_ratio = np.full(vals.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(vals)
    if np.count_nonzero(finite) < 2:
        return out.astype(np.float32), blend, target_ratio
    if station_core_authoritative_mask is None:
        station_core_authoritative_mask = np.zeros(vals.shape, dtype=bool)
    if station_measured_xs_mask is None:
        station_measured_xs_mask = np.zeros(vals.shape, dtype=bool)
    observed = vals[finite]
    ratio_min = float(np.nanpercentile(observed, 5.0)) if observed.size >= 3 else float(np.nanmin(observed))
    ratio_max = float(np.nanpercentile(observed, 95.0)) if observed.size >= 3 else float(np.nanmax(observed))
    ratio_min = min(ratio_min - 0.10, -0.75)
    ratio_max = max(ratio_max + 0.10, 1.10)
    n = len(vals)
    for i in range(n):
        if not np.isfinite(vals[i]):
            continue
        row = context_df.iloc[i]
        local_blend = _blend_weight_from_row(
            row,
            station_core_authoritative=bool(station_core_authoritative_mask[i]),
            station_xs=bool(station_xs_mask[i]),
            station_measured_xs=bool(station_measured_xs_mask[i]),
        )
        if local_blend <= 0.0:
            continue
        radius = _window_radius_from_row(row)
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        local_vals = vals[lo:hi]
        valid = np.isfinite(local_vals)
        if np.count_nonzero(valid) < 2:
            continue
        local_stations = stations[lo:hi][valid]
        dist = np.abs(local_stations - float(stations[i]))
        weights = 1.0 / np.maximum(1.0, dist)
        local_protected = protected_mask[lo:hi][valid]
        if np.any(local_protected):
            weights = weights + 1.5 * local_protected.astype(float)
        target = float(np.average(local_vals[valid], weights=weights))
        target = float(np.clip(target, ratio_min, ratio_max))
        target_ratio[i] = np.float32(target)
        if protected_mask[i]:
            local_blend = min(local_blend, 0.10)
        blend[i] = np.float32(local_blend)
        out[i] = (1.0 - local_blend) * vals[i] + local_blend * target
    out[protected_mask & finite] = vals[protected_mask & finite]
    return out.astype(np.float32), blend.astype(np.float32), target_ratio.astype(np.float32)


def apply_xs_realism_to_nodes(
    nodes,
    *,
    river_dir: str | Path,
    reach_attributes_path: str | Path | None = None,
    disabled: bool = False,
    logger: Optional[logging.Logger] = None,
):
    active_logger = logger or log
    river_dir = Path(river_dir)
    river_dir.mkdir(parents=True, exist_ok=True)
    work = nodes.copy()
    if "node_role" not in work.columns or "station_m" not in work.columns or "bed_z_m" not in work.columns:
        return work, {}, {"available": False}
    work["node_role"] = work["node_role"].fillna("thalweg").astype(str)
    work["station_m"] = pd.to_numeric(work["station_m"], errors="coerce")
    work["bed_z_m"] = pd.to_numeric(work["bed_z_m"], errors="coerce")
    work["component_id"] = work.get("component_id", "main").fillna("main").astype(str)
    work["bed_z_m_before_xs_realism"] = pd.to_numeric(work["bed_z_m"], errors="coerce").astype(np.float32)
    work["xs_realism_delta_m"] = np.float32(0.0)
    work["xs_realism_blend_weight"] = np.float32(0.0)
    work["xs_realism_target_ratio"] = np.float32(np.nan)
    work["xs_realism_reach_id"] = None
    work["xs_realism_applied"] = False

    if disabled:
        profile_csv = river_dir / "river_xs_realism_profile.csv"
        provenance_csv = river_dir / "river_xs_realism_provenance_station_summary.csv"
        summary_path = river_dir / "river_xs_realism_summary.json"
        empty_profile = pd.DataFrame(columns=["profile_id", "station_m", "station_provenance_class", "xs_support_template_class", "xs_template_type"])
        empty_profile.to_csv(profile_csv, index=False)
        empty_profile.to_csv(provenance_csv, index=False)
        summary = {
            "available": False,
            "disabled_by_option": True,
            "component_count": 0,
            "adjusted_node_count": 0,
            "delta_abs_m": _distribution(np.array([], dtype=float)),
            "blend_weight_summary": _distribution(np.array([], dtype=float)),
            "support_template_class_counts": {},
            "template_type_counts": {},
            "core_influence_scale_summary": _distribution(np.array([], dtype=float)),
            "station_provenance_class_counts": {},
            "true_measured_xs_station_count": 0,
            "channel_protected_station_count": 0,
            "inner_rebuildable_station_count": 0,
            "indirect_xs_station_count": 0,
            "residual_xs_station_count": 0,
            "components": [],
            "inputs": {
                "reach_attributes": str(reach_attributes_path) if reach_attributes_path else None,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        active_logger.info("[RIVER][XS] XS realism pass disabled by option; longitudinal/core guidance retained without XS inner-shape adjustment")
        outputs = {
            "xs_realism_profile": str(profile_csv),
            "river_xs_realism_provenance_station_summary": str(provenance_csv),
            "xs_realism_summary": str(summary_path),
        }
        return work, outputs, summary

    reach_df = _prepare_reach_attributes(_read_optional_csv(reach_attributes_path))
    station_rows = []
    component_receipts = []

    for component_id, comp_nodes in work.groupby("component_id", sort=False):
        comp_nodes = comp_nodes.loc[np.isfinite(comp_nodes["station_m"])].copy()
        if comp_nodes.empty:
            continue
        station_records = []
        for station_m, station_group in comp_nodes.groupby("station_m", sort=True):
            roles_present = set(station_group["node_role"].astype(str).unique())
            if not {"left_inner", "thalweg", "right_inner"}.issubset(roles_present):
                continue
            left_bank = _role_value(station_group, "left_bank")
            left_inner = _role_value(station_group, "left_inner")
            thalweg = _role_value(station_group, "thalweg")
            right_inner = _role_value(station_group, "right_inner")
            right_bank = _role_value(station_group, "right_bank")
            station_records.append({
                "profile_id": str(component_id),
                "station_m": float(station_m),
                "left_bank_z_m": left_bank,
                "left_inner_z_m": left_inner,
                "thalweg_z_m": thalweg,
                "right_inner_z_m": right_inner,
                "right_bank_z_m": right_bank,
                "left_bank_relief_m": left_bank - thalweg if np.isfinite(left_bank) and np.isfinite(thalweg) else np.nan,
                "right_bank_relief_m": right_bank - thalweg if np.isfinite(right_bank) and np.isfinite(thalweg) else np.nan,
                "left_inner_ratio_before": _ratio(left_inner, left_bank, thalweg),
                "right_inner_ratio_before": _ratio(right_inner, right_bank, thalweg),
                "active_core_fit_z_m": pd.to_numeric(station_group.get("active_core_fit_z_m", pd.Series(dtype=float)), errors="coerce").dropna().median() if "active_core_fit_z_m" in station_group.columns else np.nan,
                "left_bank_fit_z_m": pd.to_numeric(station_group.loc[station_group["node_role"].astype(str) == "left_bank", "left_bank_fit_z_m"], errors="coerce").dropna().median() if "left_bank_fit_z_m" in station_group.columns else np.nan,
                "right_bank_fit_z_m": pd.to_numeric(station_group.loc[station_group["node_role"].astype(str) == "right_bank", "right_bank_fit_z_m"], errors="coerce").dropna().median() if "right_bank_fit_z_m" in station_group.columns else np.nan,
                "left_inner_fit_z_m": pd.to_numeric(station_group.loc[station_group["node_role"].astype(str) == "left_inner", "bed_z_m_before_xs_realism"], errors="coerce").dropna().median() if "bed_z_m_before_xs_realism" in station_group.columns else np.nan,
                "right_inner_fit_z_m": pd.to_numeric(station_group.loc[station_group["node_role"].astype(str) == "right_inner", "bed_z_m_before_xs_realism"], errors="coerce").dropna().median() if "bed_z_m_before_xs_realism" in station_group.columns else np.nan,
                "station_target_present": bool(station_group.get("station_target_present", pd.Series(False, index=station_group.index)).fillna(False).astype(bool).any()) if "station_target_present" in station_group.columns else False,
                "station_target_source_class": str(station_group.get("station_target_source_class", pd.Series("missing", index=station_group.index)).dropna().astype(str).iloc[0]) if "station_target_source_class" in station_group.columns and not station_group.get("station_target_source_class").dropna().empty else "missing",
                "station_target_policy_reason": str(station_group.get("station_target_policy_reason", pd.Series("missing", index=station_group.index)).dropna().astype(str).iloc[0]) if "station_target_policy_reason" in station_group.columns and not station_group.get("station_target_policy_reason").dropna().empty else "missing",
                "target_xs_realism_allowed": bool(pd.to_numeric(station_group.get("target_xs_realism_allowed", pd.Series(True, index=station_group.index)), errors="coerce").fillna(1.0).astype(bool).any()) if "target_xs_realism_allowed" in station_group.columns else True,
                "target_left_bank_z_m": pd.to_numeric(station_group.loc[station_group["node_role"].astype(str) == "left_bank", "target_left_bank_z_m"], errors="coerce").dropna().median() if "target_left_bank_z_m" in station_group.columns else np.nan,
                "target_right_bank_z_m": pd.to_numeric(station_group.loc[station_group["node_role"].astype(str) == "right_bank", "target_right_bank_z_m"], errors="coerce").dropna().median() if "target_right_bank_z_m" in station_group.columns else np.nan,
                "target_left_inner_z_m": pd.to_numeric(station_group.loc[station_group["node_role"].astype(str) == "left_inner", "target_left_inner_z_m"], errors="coerce").dropna().median() if "target_left_inner_z_m" in station_group.columns else np.nan,
                "target_right_inner_z_m": pd.to_numeric(station_group.loc[station_group["node_role"].astype(str) == "right_inner", "target_right_inner_z_m"], errors="coerce").dropna().median() if "target_right_inner_z_m" in station_group.columns else np.nan,
                "target_thalweg_z_m": pd.to_numeric(station_group.loc[station_group["node_role"].astype(str) == "thalweg", "target_thalweg_z_m"], errors="coerce").dropna().median() if "target_thalweg_z_m" in station_group.columns else np.nan,
                **_station_provenance_row(station_group),
            })
        if not station_records:
            continue
        station_df = pd.DataFrame(station_records).sort_values("station_m").reset_index(drop=True)
        station_df = _attach_reach_context(station_df, reach_df, str(component_id))
        protection_rows = []
        for station_m, station_group in comp_nodes.groupby("station_m", sort=True):
            flags = _station_node_protection_flags(station_group)
            flags["station_m"] = float(station_m)
            protection_rows.append(flags)
        if protection_rows:
            station_df = station_df.merge(pd.DataFrame(protection_rows), on="station_m", how="left")
        for col in ["station_bank_protected", "station_channel_protected", "station_inner_rebuildable"]:
            if col not in station_df.columns:
                station_df[col] = False
            else:
                station_df[col] = station_df[col].fillna(False).astype(bool)
        station_df["station_protected"] = station_df["station_channel_protected"].astype(bool)
        station_semantics_rows = [
            station_semantics(
                row,
                station_authoritative=bool(row.get("station_authoritative", False)),
                station_measured_xs=bool(row.get("station_measured_xs", False)),
            )
            for _, row in station_df.iterrows()
        ]
        station_df["station_measured_support_class"] = [str(s.get("station_measured_support_class", "no_authoritative_xs_support")) for s in station_semantics_rows]
        station_df["station_authoritative_bed_support_class"] = [str(s.get("station_authoritative_bed_support_class", "no_authoritative_bed_support")) for s in station_semantics_rows]
        station_df["station_authoritative_bed_support_present"] = [bool(s.get("station_authoritative_bed_support_present", False)) for s in station_semantics_rows]
        station_df["station_authoritative_bank_margin_present"] = [bool(s.get("station_authoritative_bank_margin_present", False)) for s in station_semantics_rows]
        station_df["station_far_from_authoritative_bed"] = [bool(s.get("station_far_from_authoritative_bed", False)) for s in station_semantics_rows]
        station_df["station_support_regime"] = [str(s.get("station_support_regime", "missing")) for s in station_semantics_rows]
        station_df["station_protection_regime"] = [str(s.get("station_protection_regime", "unprotected")) for s in station_semantics_rows]
        station_df["station_rebuild_regime"] = [str(s.get("station_rebuild_regime", "blocked_not_weak_support")) for s in station_semantics_rows]
        station_df["station_core_authoritative"] = (
            station_df["station_channel_protected"].astype(bool)
            | station_df["station_authoritative_bed_support_present"].astype(bool)
        )
        support_pairs = station_df.apply(
            lambda row: _classify_xs_support_template(
                row,
                station_core_authoritative=bool(row.get("station_core_authoritative", False)),
                station_xs=bool(row.get("station_xs", False)),
                station_measured_xs=bool(row.get("station_measured_xs", False)),
            ),
            axis=1,
        )
        station_df["xs_support_template_class"] = [pair[0] for pair in support_pairs]
        station_df["xs_template_type"] = [pair[1] for pair in support_pairs]
        station_df["xs_realism_core_influence_scale"] = station_df.apply(
            lambda row: _station_core_influence_scale(
                row,
                station_core_authoritative=bool(row.get("station_core_authoritative", False)),
                station_measured_xs=bool(row.get("station_measured_xs", False)),
            ),
            axis=1,
        )
        for _col, _default in [("auth_xs_support_class", "authoritative_unknown"), ("auth_xs_support_inner_count", 0.0), ("auth_xs_support_bank_count", 0.0)]:
            if _col not in station_df.columns:
                station_df[_col] = _default
            elif _col == "auth_xs_support_class":
                station_df[_col] = station_df[_col].fillna(_default).astype(str)
            else:
                station_df[_col] = pd.to_numeric(station_df[_col], errors="coerce").fillna(float(_default))

        stations = station_df["station_m"].to_numpy(dtype=float)
        protected = station_df["station_protected"].to_numpy(dtype=bool)
        station_core_authoritative_mask = station_df["station_core_authoritative"].astype(bool).to_numpy(dtype=bool)
        station_xs_mask = (
            (pd.to_numeric(station_df.get("station_true_measured_xs_fraction", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) >= 0.20)
            | (pd.to_numeric(station_df.get("station_authoritative_channel_fraction", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) >= 0.25)
        )
        station_measured_xs_mask = station_df["station_measured_xs"].astype(bool).to_numpy(dtype=bool)
        left_after, left_blend, left_target_ratio = _smooth_ratio_profile(
            stations,
            station_df["left_inner_ratio_before"].to_numpy(dtype=float),
            protected_mask=protected,
            station_xs_mask=station_xs_mask,
            station_core_authoritative_mask=station_core_authoritative_mask,
            station_measured_xs_mask=station_measured_xs_mask,
            context_df=station_df,
        )
        right_after, right_blend, right_target_ratio = _smooth_ratio_profile(
            stations,
            station_df["right_inner_ratio_before"].to_numpy(dtype=float),
            protected_mask=protected,
            station_xs_mask=station_xs_mask,
            station_core_authoritative_mask=station_core_authoritative_mask,
            station_measured_xs_mask=station_measured_xs_mask,
            context_df=station_df,
        )
        left_after, right_after, left_target_ratio, right_target_ratio, left_blend, right_blend = _enforce_generic_template_targets(
            station_df,
            left_after=left_after,
            right_after=right_after,
            left_target_ratio=left_target_ratio,
            right_target_ratio=right_target_ratio,
            left_blend=left_blend,
            right_blend=right_blend,
        )
        left_after, right_after, left_target_ratio, right_target_ratio, left_blend, right_blend = _enforce_fitted_section_targets(
            station_df,
            left_after=left_after,
            right_after=right_after,
            left_target_ratio=left_target_ratio,
            right_target_ratio=right_target_ratio,
            left_blend=left_blend,
            right_blend=right_blend,
        )
        station_df["left_inner_ratio_after"] = left_after
        station_df["right_inner_ratio_after"] = right_after
        station_df["left_inner_ratio_target"] = left_target_ratio
        station_df["right_inner_ratio_target"] = right_target_ratio
        station_df["left_inner_blend_weight"] = left_blend
        station_df["right_inner_blend_weight"] = right_blend

        # Propagate station-level support/template classification back to node rows so
        # downstream channel-surface construction can rebuild low-support sections from
        # the cleaned longitudinal + generic XS controls rather than only from the raw
        # scaffold node pattern.
        station_lookup = station_df.set_index("station_m")[[
            "xs_support_template_class",
            "xs_template_type",
            "left_inner_ratio_after",
            "right_inner_ratio_after",
            "left_inner_blend_weight",
            "right_inner_blend_weight",
            "station_bank_protected",
            "station_channel_protected",
            "station_inner_rebuildable",
            "station_provenance_class",
            "station_true_measured_xs_fraction",
            "station_indirect_xs_fraction",
            "station_residual_xs_fraction",
            "station_authoritative_fraction",
            "station_authoritative_channel_fraction",
            "station_authoritative_bank_fraction",
            "station_core_authoritative",
            "station_authoritative_bed_support_present",
            "station_authoritative_bank_margin_present",
            "station_target_present",
            "station_target_source_class",
            "station_target_policy_reason",
            "target_xs_realism_allowed",
            "target_left_bank_z_m",
            "target_right_bank_z_m",
            "target_left_inner_z_m",
            "target_right_inner_z_m",
            "target_thalweg_z_m",
            "xs_realism_core_influence_scale",
            "auth_xs_support_class",
            "auth_xs_support_inner_count",
            "auth_xs_support_bank_count",
            "station_measured_support_class",
            "station_authoritative_bed_support_class",
            "station_far_from_authoritative_bed",
            "station_support_regime",
            "station_protection_regime",
            "station_rebuild_regime",
        ]]
        for station_val, station_meta in station_lookup.iterrows():
            station_mask = comp_nodes["station_m"].to_numpy(dtype=float) == float(station_val)
            if not np.any(station_mask):
                continue
            idxs = comp_nodes.index[station_mask]
            work.loc[idxs, "xs_support_template_class"] = str(station_meta["xs_support_template_class"])
            work.loc[idxs, "xs_template_type"] = str(station_meta["xs_template_type"])
            work.loc[idxs, "xs_realism_left_ratio_after"] = np.float32(station_meta["left_inner_ratio_after"])
            work.loc[idxs, "xs_realism_right_ratio_after"] = np.float32(station_meta["right_inner_ratio_after"])
            work.loc[idxs, "xs_realism_left_blend_weight"] = np.float32(station_meta["left_inner_blend_weight"])
            work.loc[idxs, "xs_realism_right_blend_weight"] = np.float32(station_meta["right_inner_blend_weight"])
            work.loc[idxs, "station_bank_protected"] = bool(station_meta["station_bank_protected"])
            work.loc[idxs, "station_channel_protected"] = bool(station_meta["station_channel_protected"])
            work.loc[idxs, "station_inner_rebuildable"] = bool(station_meta["station_inner_rebuildable"])
            work.loc[idxs, "station_provenance_class"] = str(station_meta["station_provenance_class"])
            work.loc[idxs, "station_true_measured_xs_fraction"] = np.float32(station_meta["station_true_measured_xs_fraction"])
            work.loc[idxs, "station_indirect_xs_fraction"] = np.float32(station_meta["station_indirect_xs_fraction"])
            work.loc[idxs, "station_residual_xs_fraction"] = np.float32(station_meta["station_residual_xs_fraction"])
            work.loc[idxs, "station_authoritative_fraction"] = np.float32(station_meta["station_authoritative_fraction"])
            work.loc[idxs, "station_authoritative_channel_fraction"] = np.float32(station_meta["station_authoritative_channel_fraction"])
            work.loc[idxs, "station_authoritative_bank_fraction"] = np.float32(station_meta["station_authoritative_bank_fraction"])
            work.loc[idxs, "station_core_authoritative"] = bool(station_meta.get("station_core_authoritative", False))
            work.loc[idxs, "station_authoritative_bed_support_present"] = bool(station_meta.get("station_authoritative_bed_support_present", False))
            work.loc[idxs, "station_authoritative_bank_margin_present"] = bool(station_meta.get("station_authoritative_bank_margin_present", False))
            work.loc[idxs, "station_target_present"] = bool(station_meta.get("station_target_present", False))
            work.loc[idxs, "station_target_source_class"] = str(station_meta.get("station_target_source_class", "missing"))
            work.loc[idxs, "station_target_policy_reason"] = str(station_meta.get("station_target_policy_reason", "missing"))
            work.loc[idxs, "target_xs_realism_allowed"] = bool(station_meta.get("target_xs_realism_allowed", True))
            work.loc[idxs, "target_left_bank_z_m"] = np.float32(station_meta.get("target_left_bank_z_m", np.nan))
            work.loc[idxs, "target_right_bank_z_m"] = np.float32(station_meta.get("target_right_bank_z_m", np.nan))
            work.loc[idxs, "target_left_inner_z_m"] = np.float32(station_meta.get("target_left_inner_z_m", np.nan))
            work.loc[idxs, "target_right_inner_z_m"] = np.float32(station_meta.get("target_right_inner_z_m", np.nan))
            work.loc[idxs, "target_thalweg_z_m"] = np.float32(station_meta.get("target_thalweg_z_m", np.nan))
            work.loc[idxs, "xs_realism_core_influence_scale"] = np.float32(station_meta.get("xs_realism_core_influence_scale", 1.0))
            work.loc[idxs, "auth_xs_support_class"] = str(station_meta.get("auth_xs_support_class", "authoritative_unknown"))
            work.loc[idxs, "auth_xs_support_inner_count"] = np.float32(station_meta.get("auth_xs_support_inner_count", 0.0))
            work.loc[idxs, "auth_xs_support_bank_count"] = np.float32(station_meta.get("auth_xs_support_bank_count", 0.0))
            work.loc[idxs, "station_measured_support_class"] = str(station_meta.get("station_measured_support_class", "no_authoritative_xs_support"))
            work.loc[idxs, "station_authoritative_bed_support_class"] = str(station_meta.get("station_authoritative_bed_support_class", "no_authoritative_bed_support"))
            work.loc[idxs, "station_far_from_authoritative_bed"] = bool(station_meta.get("station_far_from_authoritative_bed", False))
            work.loc[idxs, "station_support_regime"] = str(station_meta["station_support_regime"])
            work.loc[idxs, "station_protection_regime"] = str(station_meta["station_protection_regime"])
            work.loc[idxs, "station_rebuild_regime"] = str(station_meta["station_rebuild_regime"])

        station_delta_values = []
        adjusted_station_count = 0
        for _, row in station_df.iterrows():
            station = float(row["station_m"])
            station_mask = comp_nodes["station_m"].to_numpy(dtype=float) == station
            station_group = comp_nodes.loc[station_mask].copy()
            thalweg = float(pd.to_numeric(row.get("thalweg_z_m"), errors="coerce")) if row.get("thalweg_z_m") is not None else np.nan
            updates = {}
            for side, bank_role, inner_role, bank_relief_col, ratio_col, blend_col in [
                ("left", "left_bank", "left_inner", "left_bank_relief_m", "left_inner_ratio_after", "left_inner_blend_weight"),
                ("right", "right_bank", "right_inner", "right_bank_relief_m", "right_inner_ratio_after", "right_inner_blend_weight"),
            ]:
                bank_relief = float(pd.to_numeric(row.get(bank_relief_col), errors="coerce")) if row.get(bank_relief_col) is not None else np.nan
                ratio = float(pd.to_numeric(row.get(ratio_col), errors="coerce")) if row.get(ratio_col) is not None else np.nan
                blend = float(pd.to_numeric(row.get(blend_col), errors="coerce")) if row.get(blend_col) is not None else 0.0
                if not (np.isfinite(thalweg) and np.isfinite(bank_relief) and np.isfinite(ratio)):
                    continue
                target_inner = float(thalweg + ratio * bank_relief)
                inner_idx = station_group.index[station_group["node_role"].astype(str).eq(inner_role)]
                if len(inner_idx) < 1:
                    continue
                idx = int(inner_idx[0])
                current_inner = float(pd.to_numeric(work.at[idx, "bed_z_m"], errors="coerce"))
                if not np.isfinite(current_inner):
                    continue
                if bool(_is_authoritative_like(work.loc[[idx]]).item()):
                    continue
                updates[idx] = {
                    "target_inner": target_inner,
                    "blend": blend,
                    "ratio": ratio,
                    "side": side,
                }
            station_adjusted = False
            station_delta_m = 0.0
            for idx, info in updates.items():
                current_inner = float(pd.to_numeric(work.at[idx, "bed_z_m"], errors="coerce"))
                target_inner = float(info["target_inner"])
                blend = float(info["blend"])
                updated_inner = (1.0 - blend) * current_inner + blend * target_inner
                delta = updated_inner - current_inner
                if abs(delta) <= 1.0e-6:
                    updated_inner = current_inner
                    delta = 0.0
                if abs(delta) > 1.0e-6:
                    station_adjusted = True
                    station_delta_m = max(station_delta_m, abs(delta))
                work.at[idx, "bed_z_m"] = np.float32(updated_inner)
                work.at[idx, "xs_realism_delta_m"] = np.float32(delta)
                work.at[idx, "xs_realism_blend_weight"] = np.float32(blend)
                work.at[idx, "xs_realism_target_ratio"] = np.float32(info["ratio"])
                work.at[idx, "xs_realism_reach_id"] = row.get("reach_id")
                work.at[idx, "xs_realism_applied"] = abs(delta) > 1.0e-6
            station_df.loc[station_df["station_m"].eq(station), "xs_realism_station_adjusted"] = bool(station_adjusted)
            station_df.loc[station_df["station_m"].eq(station), "xs_realism_station_delta_abs_m"] = float(station_delta_m)
            if station_adjusted:
                adjusted_station_count += 1
            station_delta_values.append(float(station_delta_m))

        station_rows.extend(station_df.to_dict(orient="records"))
        component_receipts.append({
            "component_id": str(component_id),
            "station_count": int(len(station_df)),
            "protected_station_count": int(np.count_nonzero(protected)),
            "adjusted_station_count": int(adjusted_station_count),
            "measured_support_class_counts": {str(k): int(v) for k, v in station_df["station_measured_support_class"].astype(str).value_counts(dropna=False).items()},
            "authoritative_bed_support_class_counts": {str(k): int(v) for k, v in station_df["station_authoritative_bed_support_class"].astype(str).value_counts(dropna=False).items()},
            "support_template_class_counts": {str(k): int(v) for k, v in station_df["xs_support_template_class"].astype(str).value_counts(dropna=False).items()},
            "provenance_class_counts": {str(k): int(v) for k, v in station_df["station_provenance_class"].astype(str).value_counts(dropna=False).items()},
            "template_type_counts": {str(k): int(v) for k, v in station_df["xs_template_type"].astype(str).value_counts(dropna=False).items()},
            "core_influence_scale_summary": _distribution(pd.to_numeric(station_df["xs_realism_core_influence_scale"], errors="coerce").to_numpy(dtype=float)),
            "left_ratio_shift": _distribution(np.abs(pd.to_numeric(station_df["left_inner_ratio_after"], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(station_df["left_inner_ratio_before"], errors="coerce").to_numpy(dtype=float))),
            "right_ratio_shift": _distribution(np.abs(pd.to_numeric(station_df["right_inner_ratio_after"], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(station_df["right_inner_ratio_before"], errors="coerce").to_numpy(dtype=float))),
            "station_delta_abs_m": _distribution(np.asarray(station_delta_values, dtype=float)),
        })

    outputs: Dict[str, str] = {}
    summary: Dict[str, Any] = {
        "available": bool(component_receipts),
        "component_count": int(len(component_receipts)),
        "adjusted_node_count": int(np.count_nonzero(pd.to_numeric(work.get("xs_realism_applied"), errors="coerce").fillna(False).astype(bool))),
        "delta_abs_m": _distribution(np.abs(pd.to_numeric(work.get("xs_realism_delta_m"), errors="coerce").to_numpy(dtype=float))),
        "blend_weight_summary": _distribution(pd.to_numeric(work.get("xs_realism_blend_weight"), errors="coerce").to_numpy(dtype=float)),
        "support_template_class_counts": {},
        "template_type_counts": {},
        "core_influence_scale_summary": _distribution(pd.to_numeric(work.get("xs_realism_core_influence_scale", pd.Series([1.0] * len(work), index=work.index)), errors="coerce").to_numpy(dtype=float)),
        "station_provenance_class_counts": {str(k): int(v) for k, v in pd.Series([row.get("station_provenance_class", "structural") for row in station_rows], dtype="object").astype(str).value_counts(dropna=False).items()},
        "true_measured_xs_station_count": int(np.count_nonzero(pd.Series([bool(row.get("station_measured_xs", False)) for row in station_rows], dtype=bool).to_numpy(dtype=bool))),
        "channel_protected_station_count": int(np.count_nonzero(pd.Series([bool(row.get("station_channel_protected", False)) for row in station_rows], dtype=bool).to_numpy(dtype=bool))),
        "inner_rebuildable_station_count": int(np.count_nonzero(pd.Series([bool(row.get("station_inner_rebuildable", False)) for row in station_rows], dtype=bool).to_numpy(dtype=bool))),
        "indirect_xs_station_count": int(np.count_nonzero(pd.Series([bool(row.get("station_indirect_xs", False)) for row in station_rows], dtype=bool).to_numpy(dtype=bool))),
        "residual_xs_station_count": int(np.count_nonzero(pd.Series([bool(row.get("station_residual_xs", False)) for row in station_rows], dtype=bool).to_numpy(dtype=bool))),
        "components": component_receipts,
        "inputs": {
            "reach_attributes": str(reach_attributes_path) if reach_attributes_path else None,
        },
    }
    if component_receipts:
        class_counts = {}
        template_counts = {}
        for rec in component_receipts:
            for k, v in rec.get("support_template_class_counts", {}).items():
                class_counts[k] = class_counts.get(k, 0) + int(v)
            for k, v in rec.get("template_type_counts", {}).items():
                template_counts[k] = template_counts.get(k, 0) + int(v)
        summary["support_template_class_counts"] = class_counts
        summary["template_type_counts"] = template_counts
        profile_csv = river_dir / "river_xs_realism_profile.csv"
        profile_df = pd.DataFrame(station_rows)
        profile_df.to_csv(profile_csv, index=False)
        provenance_cols = [
            "profile_id", "station_m", "station_provenance_class", "station_authoritative", "station_measured_xs",
            "station_bank_protected", "station_channel_protected", "station_inner_rebuildable",
            "station_indirect_xs", "station_residual_xs", "station_xs",
            "station_true_measured_xs_fraction", "station_indirect_xs_fraction", "station_residual_xs_fraction",
            "station_authoritative_fraction", "station_authoritative_channel_fraction", "station_authoritative_bank_fraction",
            "station_core_authoritative", "station_authoritative_bed_support_present", "station_authoritative_bank_margin_present",
            "xs_realism_core_influence_scale", "xs_support_template_class", "xs_template_type",
        ]
        provenance_csv = river_dir / "river_xs_realism_provenance_station_summary.csv"
        profile_df[[c for c in provenance_cols if c in profile_df.columns]].to_csv(provenance_csv, index=False)
        summary_path = river_dir / "river_xs_realism_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        outputs["xs_realism_profile"] = str(profile_csv)
        outputs["river_xs_realism_provenance_station_summary"] = str(provenance_csv)
        outputs["xs_realism_summary"] = str(summary_path)
        active_logger.info(
            "[RIVER][XS] XS realism pass applied: adjusted_nodes=%d components=%d",
            int(summary["adjusted_node_count"]),
            int(summary["component_count"]),
        )
    else:
        active_logger.info("[RIVER][XS] XS realism pass skipped: insufficient inner/bank/thalweg station roles")
    return work, outputs, summary


__all__ = ["apply_xs_realism_to_nodes"]
