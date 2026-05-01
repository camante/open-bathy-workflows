from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from river_longitudinal_tendency import apply_longitudinal_tendency_to_nodes
from legacy.river.archive_root_scripts.river_xs_realism import apply_xs_realism_to_nodes
from river_prediction_confidence import apply_prediction_confidence_to_nodes
from legacy.river.archive_root_scripts.river_primary_surface_rebuild import apply_primary_surface_rebuild_to_nodes
from river_effectiveness_receipts import write_effectiveness_receipts, write_science_effect_summary
from river_support_semantics import WEAK_SUPPORT_CLASSES, LONGITUDINAL_WEAK_SUPPORT
from river_longitudinal_tendency import _distribution
from river_support_uncertainty import (
    SUPPORT_CLASS_CODES,
    SOLUTION_MODE_CODES,
    UNCERTAINTY_CLASS_CODES,
    graph_solution_confidence,
    uncertainty_class_from_confidence,
    support_class_code,
    solution_mode_code,
    uncertainty_class_code,
    unsupported_regime_code,
    UNSUPPORTED_REGIME_CODES,
)
from authoritative_river_roles import ROLE_BED_CORE, ROLE_BED_INNER, code_to_role
from river_source_semantics import is_surface_guidance_source
from river_target_contract import normalize_active_target_source

log = logging.getLogger(__name__)

SOURCE_CODES = {
    'missing': 0,
    'bank_stage_prior': 1,
    'resolved_channel_bed': 2,
    'xs_profile_resampled': 3,
    'authoritative_in_channel': 4,
    'authoritative_backbone': 5,
    'graph_backbone': 6,
    'authoritative_bank_margin': 7,
    'station_target_section_tendency': 8,
    'station_target_local_authoritative_reconciliation': 9,
    'generalized_thalweg_default_tendency': 10,
    'bank_edge_geometry_constraint': 11,
}
SOURCE_CONF = {
    0: 0.0,
    1: 0.35,
    2: 0.55,
    3: 0.8,
    4: 1.0,
    5: 0.95,
    6: 0.7,
    7: 0.78,
    8: 0.72,
    9: 0.82,
    10: 0.62,
    11: 0.66,
}
GRAPH_MODE_CODES = SOLUTION_MODE_CODES
JUNCTION_ROLE_CODES = {'not_in_junction': 0, 'dominant': 1, 'constrained_branch': 2, 'balanced': 3}
ROLE_ORDER = [('left_bank', 0.0), ('left_inner', 0.25), ('thalweg', 0.5), ('right_inner', 0.75), ('right_bank', 1.0)]
FAST_ROLE_ORDER = [('left_bank', 0.0), ('thalweg', 0.5), ('right_bank', 1.0)]
THALWEG_DOMINANT_RENDER_EXPONENT = 2.0
THALWEG_DOMINANT_LINEAR_BLEND = 0.05
THALWEG_DOMINANT_INNER_RECONCILIATION_WEIGHT = 0.20
SECTION_TARGET_BASE_BLEND = 0.12
SECTION_TARGET_LOCAL_RECONCILED_BLEND_FLOOR = 0.45
SECTION_TARGET_LOCAL_RECONCILED_BLEND_CEILING = 0.85
SECTION_TARGET_GENERIC_BLEND_CEILING = 0.35
SECTION_TARGET_PLAUSIBILITY_MIN_M = 5.0
AUTHORITATIVE_TRANSITION_INNER_M = 50.0
AUTHORITATIVE_TRANSITION_OUTER_M = 150.0
AUTHORITATIVE_TRANSITION_CURVE_POWER = 2.0
AUTHORITATIVE_TRANSITION_GENERIC_CEILING = 0.18
CHANNEL_SURFACE_LONGITUDINAL_SMOOTHING_SIGMA_STATIONS = 2.5
CHANNEL_SURFACE_LONGITUDINAL_SMOOTHING_MIN_SAMPLES = 5
CHANNEL_SURFACE_LONGITUDINAL_SMOOTHING_BASE_BLEND = 0.22
CHANNEL_SURFACE_LONGITUDINAL_SMOOTHING_MAX_SHIFT_M = 0.45


BANK_EDGE_ROLES = {'left_bank', 'right_bank'}
THALWEG_ROLES = {'thalweg'}
SUBORDINATE_INNER_SHAPE_ROLES = {'left_inner', 'right_inner'}
INTERIOR_BED_ROLES = THALWEG_ROLES | SUBORDINATE_INNER_SHAPE_ROLES


def _role_semantic_group(role_name: str) -> str:
    role = str(role_name or '')
    if role in BANK_EDGE_ROLES:
        return 'bank_edge'
    if role in THALWEG_ROLES:
        return 'thalweg'
    if role in SUBORDINATE_INNER_SHAPE_ROLES:
        return 'inner_shape'
    return 'other'


def _interior_semantic_weight(eta: float) -> float:
    r = np.clip(abs(float(eta) - 0.5) / 0.5, 0.0, 1.0)
    return float(np.clip(1.0 - (r ** 2.0), 0.0, 1.0))


def _linear_eta_value(eta: float, eta_coords: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if np.count_nonzero(valid) >= 2:
        return float(np.interp(eta, eta_coords[valid], values[valid]))
    if np.count_nonzero(valid) == 1:
        return float(values[valid][0])
    return float('nan')


def _thalweg_dominant_eta_value(eta: float, eta_coords: np.ndarray, values: np.ndarray) -> float:
    eta_axis = np.asarray(eta_coords, dtype=float)
    z_axis = np.asarray(values, dtype=float)
    linear = _linear_eta_value(float(eta), eta_axis, z_axis)
    valid = np.isfinite(z_axis)
    if np.count_nonzero(valid) < 2:
        return linear

    role_lookup = {round(float(e), 6): i for i, e in enumerate(eta_axis)}
    th_idx = role_lookup.get(round(0.5, 6))
    if th_idx is None or not np.isfinite(z_axis[th_idx]):
        return linear
    thalweg_z = float(z_axis[th_idx])

    is_left = float(eta) <= 0.5
    bank_idx = role_lookup.get(round(0.0 if is_left else 1.0, 6))
    inner_idx = role_lookup.get(round(0.25 if is_left else 0.75, 6))
    bank_z = float(z_axis[bank_idx]) if bank_idx is not None and np.isfinite(z_axis[bank_idx]) else np.nan
    inner_z = float(z_axis[inner_idx]) if inner_idx is not None and np.isfinite(z_axis[inner_idx]) else np.nan

    side_controls = np.asarray([v for v in [thalweg_z, inner_z, bank_z] if np.isfinite(v)], dtype=float)
    if side_controls.size < 2:
        return linear
    side_max = float(np.nanmax(side_controls))
    if not np.isfinite(side_max) or side_max <= thalweg_z + 1e-6:
        return linear

    r = np.clip(abs(float(eta) - 0.5) / 0.5, 0.0, 1.0)
    shaped = thalweg_z + (side_max - thalweg_z) * (r ** THALWEG_DOMINANT_RENDER_EXPONENT)

    if np.isfinite(inner_z):
        inner_expected = thalweg_z + (side_max - thalweg_z) * (0.5 ** THALWEG_DOMINANT_RENDER_EXPONENT)
        inner_delta = inner_z - inner_expected
        taper = max(0.0, 1.0 - abs(r - 0.5) / 0.5)
        shaped += inner_delta * taper * THALWEG_DOMINANT_INNER_RECONCILIATION_WEIGHT

    if np.isfinite(bank_z):
        bank_taper = r ** 2
        shaped = (1.0 - bank_taper) * shaped + bank_taper * bank_z

    if np.isfinite(linear):
        shaped = (1.0 - THALWEG_DOMINANT_LINEAR_BLEND) * shaped + THALWEG_DOMINANT_LINEAR_BLEND * linear

    lo = float(np.nanmin(side_controls))
    hi = float(np.nanmax(side_controls))
    return float(np.clip(shaped, lo, hi)) if np.isfinite(shaped) else linear





def _thalweg_dominant_eta_values(etas: np.ndarray, eta_coords: np.ndarray, values_by_column: np.ndarray) -> np.ndarray:
    etas_arr = np.asarray(etas, dtype=float)
    eta_axis = np.asarray(eta_coords, dtype=float)
    vals = np.asarray(values_by_column, dtype=float)
    if vals.ndim != 2:
        raise ValueError('values_by_column must be 2-D with shape (roles, columns)')
    if vals.shape[1] != etas_arr.size:
        raise ValueError('etas and values_by_column column counts must match')
    out = np.full(etas_arr.shape, np.nan, dtype=float)
    for i, eta in enumerate(etas_arr):
        out[i] = _thalweg_dominant_eta_value(float(eta), eta_axis, vals[:, i])
    return out


def _section_target_eta_values(
    etas: np.ndarray,
    *,
    target_left_bank_z: np.ndarray,
    target_thalweg_z: np.ndarray,
    target_right_bank_z: np.ndarray,
) -> np.ndarray:
    etas_arr = np.asarray(etas, dtype=float)
    left = np.asarray(target_left_bank_z, dtype=float)
    thalweg = np.asarray(target_thalweg_z, dtype=float)
    right = np.asarray(target_right_bank_z, dtype=float)
    if not (etas_arr.size == left.size == thalweg.size == right.size):
        raise ValueError('section target arrays must all have the same size')
    eta_axis = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    values = np.vstack([left, np.full_like(left, np.nan), thalweg, np.full_like(left, np.nan), right])
    return _thalweg_dominant_eta_values(etas_arr, eta_axis, values)


def _section_target_blend_weights(
    *,
    target_present: np.ndarray,
    local_authoritative_reconciled: np.ndarray,
    authoritative_reconciliation_weight: np.ndarray,
    authoritative_bed_support_distance_m: np.ndarray,
) -> np.ndarray:
    present = np.asarray(target_present, dtype=bool)
    local = np.asarray(local_authoritative_reconciled, dtype=bool)
    recon = np.asarray(authoritative_reconciliation_weight, dtype=float)
    dist = np.asarray(authoritative_bed_support_distance_m, dtype=float)
    if not (present.size == local.size == recon.size == dist.size):
        raise ValueError('blend-weight arrays must all have the same size')
    weight = np.zeros(present.shape, dtype=float)
    active = present
    weight[active] = SECTION_TARGET_BASE_BLEND

    generic = active & ~local
    weight[generic & np.isfinite(dist) & (dist <= 10.0)] += 0.12
    weight[generic & np.isfinite(dist) & (dist > 10.0) & (dist <= 25.0)] += 0.06
    recon_finite = generic & np.isfinite(recon)
    weight[recon_finite] = np.maximum(weight[recon_finite], 0.25 * recon[recon_finite])
    weight[generic] = np.clip(weight[generic], 0.0, SECTION_TARGET_GENERIC_BLEND_CEILING)

    local_idx = active & local
    if np.any(local_idx):
        local_weight = np.maximum(weight[local_idx], SECTION_TARGET_LOCAL_RECONCILED_BLEND_FLOOR)
        recon_local = recon[local_idx]
        finite_recon_local = np.isfinite(recon_local)
        if np.any(finite_recon_local):
            local_weight[finite_recon_local] = np.maximum(local_weight[finite_recon_local], recon_local[finite_recon_local])
        dist_local = dist[local_idx]
        local_weight[np.isfinite(dist_local) & (dist_local <= 3.0)] = np.maximum(local_weight[np.isfinite(dist_local) & (dist_local <= 3.0)], 0.80)
        local_weight[np.isfinite(dist_local) & (dist_local > 3.0) & (dist_local <= 10.0)] = np.maximum(local_weight[np.isfinite(dist_local) & (dist_local > 3.0) & (dist_local <= 10.0)], 0.65)
        local_weight[np.isfinite(dist_local) & (dist_local > 10.0) & (dist_local <= 25.0)] = np.maximum(local_weight[np.isfinite(dist_local) & (dist_local > 10.0) & (dist_local <= 25.0)], 0.55)
        weight[local_idx] = np.clip(local_weight, 0.0, SECTION_TARGET_LOCAL_RECONCILED_BLEND_CEILING)

    weight[~active] = 0.0
    return weight


def _authoritative_transition_blend_weights(
    *,
    authoritative_bed_support_distance_m: np.ndarray,
    local_authoritative_reconciled: np.ndarray,
    target_present: np.ndarray,
) -> np.ndarray:
    dist = np.asarray(authoritative_bed_support_distance_m, dtype=float)
    local = np.asarray(local_authoritative_reconciled, dtype=bool)
    present = np.asarray(target_present, dtype=bool)
    if not (dist.size == local.size == present.size):
        raise ValueError('transition-weight arrays must all have the same size')
    out = np.zeros(dist.shape, dtype=float)
    active = np.isfinite(dist) & (local | present | (dist <= AUTHORITATIVE_TRANSITION_OUTER_M))
    if not np.any(active):
        return out
    d = dist[active]
    base = np.zeros(d.shape, dtype=float)
    base[d <= AUTHORITATIVE_TRANSITION_INNER_M] = 1.0
    between = (d > AUTHORITATIVE_TRANSITION_INNER_M) & (d < AUTHORITATIVE_TRANSITION_OUTER_M)
    if np.any(between):
        frac = 1.0 - ((d[between] - AUTHORITATIVE_TRANSITION_INNER_M) / max(AUTHORITATIVE_TRANSITION_OUTER_M - AUTHORITATIVE_TRANSITION_INNER_M, 1.0))
        base[between] = np.clip(frac, 0.0, 1.0) ** AUTHORITATIVE_TRANSITION_CURVE_POWER
    generic = np.full(d.shape, AUTHORITATIVE_TRANSITION_GENERIC_CEILING, dtype=float)
    ceiling = np.where(local[active], 0.85, np.where(present[active], 0.45, generic))
    out[active] = np.clip(base * ceiling, 0.0, ceiling)
    return out
def _section_target_eta_value(
    eta: float,
    *,
    target_left_bank_z: float,
    target_thalweg_z: float,
    target_right_bank_z: float,
) -> float:
    eta_axis = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    values = np.asarray([
        target_left_bank_z,
        np.nan,
        target_thalweg_z,
        np.nan,
        target_right_bank_z,
    ], dtype=float)
    return _thalweg_dominant_eta_value(float(eta), eta_axis, values)


def _section_target_blend_weight(
    *,
    target_present: bool,
    local_authoritative_reconciled: bool,
    authoritative_reconciliation_weight: float,
    authoritative_bed_support_distance_m: float,
) -> float:
    if not target_present:
        # No explicit station target means there is nothing valid to blend toward.
        return 0.0
    weight = float(SECTION_TARGET_BASE_BLEND)
    if local_authoritative_reconciled:
        weight = max(weight, SECTION_TARGET_LOCAL_RECONCILED_BLEND_FLOOR)
        if np.isfinite(authoritative_reconciliation_weight):
            weight = max(weight, float(authoritative_reconciliation_weight))
        if np.isfinite(authoritative_bed_support_distance_m):
            if authoritative_bed_support_distance_m <= 3.0:
                weight = max(weight, 0.80)
            elif authoritative_bed_support_distance_m <= 10.0:
                weight = max(weight, 0.65)
            elif authoritative_bed_support_distance_m <= 25.0:
                weight = max(weight, 0.55)
        return float(np.clip(weight, 0.0, SECTION_TARGET_LOCAL_RECONCILED_BLEND_CEILING))
    if np.isfinite(authoritative_bed_support_distance_m):
        if authoritative_bed_support_distance_m <= 10.0:
            weight += 0.12
        elif authoritative_bed_support_distance_m <= 25.0:
            weight += 0.06
    if np.isfinite(authoritative_reconciliation_weight):
        weight = max(weight, 0.25 * float(authoritative_reconciliation_weight))
    return float(np.clip(weight, 0.0, SECTION_TARGET_GENERIC_BLEND_CEILING))


def _authoritative_transition_blend_weight(
    *,
    authoritative_bed_support_distance_m: float,
    local_authoritative_reconciled: bool,
    target_present: bool,
) -> float:
    if (not np.isfinite(authoritative_bed_support_distance_m)) or (
        (not target_present) and (not local_authoritative_reconciled) and (float(authoritative_bed_support_distance_m) > AUTHORITATIVE_TRANSITION_OUTER_M)
    ):
        return 0.0
    d = float(authoritative_bed_support_distance_m)
    if d <= AUTHORITATIVE_TRANSITION_INNER_M:
        base = 1.0
    elif d >= AUTHORITATIVE_TRANSITION_OUTER_M:
        base = 0.0
    else:
        frac = 1.0 - ((d - AUTHORITATIVE_TRANSITION_INNER_M) / max(AUTHORITATIVE_TRANSITION_OUTER_M - AUTHORITATIVE_TRANSITION_INNER_M, 1.0))
        base = float(np.clip(frac, 0.0, 1.0) ** AUTHORITATIVE_TRANSITION_CURVE_POWER)
    if local_authoritative_reconciled:
        ceiling = 0.85
    elif target_present:
        ceiling = 0.45
    else:
        ceiling = AUTHORITATIVE_TRANSITION_GENERIC_CEILING
    return float(np.clip(base * ceiling, 0.0, ceiling))


def _component_plausibility_tolerance_m(component_values: np.ndarray) -> float:
    vals = np.asarray(component_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size >= 2:
        span = float(np.nanpercentile(vals, 95) - np.nanpercentile(vals, 5))
    elif vals.size == 1:
        span = 0.0
    else:
        span = np.nan
    if np.isfinite(span):
        return float(max(SECTION_TARGET_PLAUSIBILITY_MIN_M, 4.0 * span))
    return float(SECTION_TARGET_PLAUSIBILITY_MIN_M)


def _section_target_agreement_target_for_row(row: pd.Series) -> tuple[float, str, str]:
    role = str(row.get('node_role', 'missing') or 'missing')
    role_group = _role_semantic_group(role)
    left_bank = pd.to_numeric(row.get('target_left_bank_z_m'), errors='coerce')
    right_bank = pd.to_numeric(row.get('target_right_bank_z_m'), errors='coerce')
    active_target = pd.to_numeric(row.get('active_interior_target_z_m'), errors='coerce')
    active_target_source = normalize_active_target_source(row.get('active_interior_target_source'))

    if role == 'left_bank' and np.isfinite(left_bank):
        return float(left_bank), role_group, 'direct_bank_target'
    if role == 'right_bank' and np.isfinite(right_bank):
        return float(right_bank), role_group, 'direct_bank_target'

    if not np.isfinite(active_target) or active_target_source == 'missing':
        return float('nan'), 'missing', 'missing_canonical_active_target'

    if role == 'thalweg':
        return float(active_target), active_target_source, 'canonical_active_target'

    eta_lookup = {
        'left_inner': 0.25,
        'right_inner': 0.75,
    }
    eta = eta_lookup.get(role)
    if eta is None:
        return float('nan'), 'missing', 'unknown_role'

    if not np.isfinite(left_bank) and np.isfinite(right_bank):
        left_bank = right_bank
    if not np.isfinite(right_bank) and np.isfinite(left_bank):
        right_bank = left_bank
    if not (np.isfinite(left_bank) and np.isfinite(right_bank)):
        return float('nan'), active_target_source, 'missing_canonical_bank_bounds'

    target = _section_target_eta_value(
        float(eta),
        target_left_bank_z=float(left_bank),
        target_thalweg_z=float(active_target),
        target_right_bank_z=float(right_bank),
    )
    return float(target), active_target_source, 'canonical_active_target'



def _backfill_node_section_target_fields(
    nodes: pd.DataFrame,
    *,
    thalweg_by_comp: dict[str, pd.DataFrame],
    station_target_left_bank_surfaces: dict[str, np.ndarray],
    station_target_right_bank_surfaces: dict[str, np.ndarray],
    station_target_thalweg_surfaces: dict[str, np.ndarray],
    station_target_present_surfaces: dict[str, np.ndarray],
    station_target_local_authoritative_reconciled_surfaces: dict[str, np.ndarray],
    station_authoritative_reconciliation_weight_surfaces: dict[str, np.ndarray],
    station_authoritative_bed_support_distance_surfaces: dict[str, np.ndarray],
) -> pd.DataFrame:
    if nodes.empty or 'component_id' not in nodes.columns or 'station_m' not in nodes.columns:
        return nodes
    work = nodes.copy()
    station_vals = pd.to_numeric(work.get('station_m', np.nan), errors='coerce').to_numpy(dtype=float)
    comp_vals = work.get('component_id', pd.Series([''] * len(work), index=work.index)).astype(str).to_numpy()
    field_specs = [
        ('target_left_bank_z_m', station_target_left_bank_surfaces, 'float'),
        ('target_right_bank_z_m', station_target_right_bank_surfaces, 'float'),
        ('target_thalweg_z_m', station_target_thalweg_surfaces, 'float'),
        ('station_target_present', station_target_present_surfaces, 'boolish'),
        ('station_target_local_authoritative_reconciled', station_target_local_authoritative_reconciled_surfaces, 'boolish'),
        ('authoritative_reconciliation_weight', station_authoritative_reconciliation_weight_surfaces, 'float'),
        ('station_authoritative_bed_support_distance_m', station_authoritative_bed_support_distance_surfaces, 'float'),
    ]
    column_updates: dict[str, pd.Series | np.ndarray] = {}
    for field_name, surfaces, mode in field_specs:
        if field_name in work.columns:
            existing = pd.to_numeric(work[field_name], errors='coerce').to_numpy(dtype=float)
        else:
            existing = np.full(len(work), np.nan, dtype=float)
        fill_mask = ~np.isfinite(existing)
        out = existing.copy()
        if np.any(fill_mask):
            for comp in pd.unique(comp_vals[fill_mask]):
                comp = str(comp)
                th = thalweg_by_comp.get(comp)
                surf = surfaces.get(comp)
                if th is None or th.empty or surf is None:
                    continue
                stations = pd.to_numeric(th.get('station_m', np.nan), errors='coerce').to_numpy(dtype=float)
                surf = np.asarray(surf, dtype=float)
                valid = np.isfinite(stations) & np.isfinite(surf)
                if np.count_nonzero(valid) < 1:
                    continue
                sel = fill_mask & (comp_vals == comp) & np.isfinite(station_vals)
                if not np.any(sel):
                    continue
                if np.count_nonzero(valid) == 1:
                    out[sel] = float(surf[valid][0])
                else:
                    out[sel] = np.interp(station_vals[sel], stations[valid], surf[valid])
        if mode == 'boolish':
            finite = np.isfinite(out)
            field_vals = np.where(finite, out >= 0.5, False)
            column_updates[field_name] = pd.Series(field_vals, index=work.index, dtype=bool)
        else:
            column_updates[field_name] = out

    if column_updates:
        work = work.assign(**column_updates)

    target_present = work.get('station_target_present', pd.Series(False, index=work.index)).astype(bool).to_numpy()
    target_local = work.get('station_target_local_authoritative_reconciled', pd.Series(False, index=work.index)).astype(bool).to_numpy()
    target_thalweg = pd.to_numeric(work.get('target_thalweg_z_m', np.nan), errors='coerce').to_numpy(dtype=float)
    target_left = pd.to_numeric(work.get('target_left_bank_z_m', np.nan), errors='coerce').to_numpy(dtype=float)
    target_right = pd.to_numeric(work.get('target_right_bank_z_m', np.nan), errors='coerce').to_numpy(dtype=float)
    target_geom = np.isfinite(target_thalweg) & (np.isfinite(target_left) | np.isfinite(target_right))
    effective = target_present | target_local | target_geom
    work = work.assign(station_target_effective_present=effective.astype(np.int16))
    return work




def _p95_from_error_bucket(bucket: Any) -> float:
    if not isinstance(bucket, dict):
        return np.nan
    abs_err = bucket.get("abs_error_m", {}) if isinstance(bucket.get("abs_error_m", {}), dict) else {}
    value = abs_err.get("p95", np.nan)
    return float(value) if value is not None and np.isfinite(value) else np.nan


def _classify_role_agreement_focus(*, thalweg_p95: float, inner_p95: float, bank_p95: float) -> dict[str, Any]:
    role_candidates = [(name, value) for name, value in (("thalweg", thalweg_p95), ("inner_shape", inner_p95), ("bank_edge", bank_p95)) if np.isfinite(value)]
    weakest_role = max(role_candidates, key=lambda kv: kv[1])[0] if role_candidates else None
    weakest_role_p95 = max((value for _, value in role_candidates), default=np.nan)
    bank_minus_inner = float(bank_p95 - inner_p95) if np.isfinite(bank_p95) and np.isfinite(inner_p95) else np.nan
    inner_minus_thalweg = float(inner_p95 - thalweg_p95) if np.isfinite(inner_p95) and np.isfinite(thalweg_p95) else np.nan
    if weakest_role == "bank_edge" and (not np.isfinite(bank_minus_inner) or bank_minus_inner > 0.05):
        failure_mode = "bank_vs_inner_shape"
        failure_reason = "Bank-edge error exceeds inner-shape error, so bank-margin behavior is the dominant lateral discrepancy."
    elif weakest_role == "inner_shape" and (not np.isfinite(inner_minus_thalweg) or inner_minus_thalweg > 0.05):
        failure_mode = "inner_shape_width_propagation"
        failure_reason = "Inner-shape error exceeds thalweg error, so the centerline signal is not spreading across the channel width strongly enough."
    elif weakest_role == "thalweg":
        failure_mode = "thalweg_fit"
        failure_reason = "Thalweg has the largest role error, so longitudinal backbone preservation remains the dominant discrepancy."
    elif weakest_role == "inner_shape":
        failure_mode = "inner_shape_width_propagation"
        failure_reason = "Inner-shape has the largest role error, so channel-width reconstruction remains the dominant lateral discrepancy."
    elif weakest_role == "bank_edge":
        failure_mode = "bank_vs_inner_shape"
        failure_reason = "Bank-edge has the largest role error, so bank-margin behavior remains the dominant lateral discrepancy."
    else:
        failure_mode = None
        failure_reason = "No finite role-agreement comparisons were available."
    return {
        "weakest_role": weakest_role,
        "weakest_role_p95_abs_error_m": float(weakest_role_p95) if np.isfinite(weakest_role_p95) else np.nan,
        "bank_minus_inner_p95_abs_error_m": bank_minus_inner,
        "inner_minus_thalweg_p95_abs_error_m": inner_minus_thalweg,
        "lateral_failure_mode": failure_mode,
        "lateral_failure_reason": failure_reason,
    }


def _prepare_section_target_comparison_frame(*, nodes: pd.DataFrame, z_out: np.ndarray, transform) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import rasterio.transform
    except ImportError:
        return pd.DataFrame(), {'available': False, 'reason': 'rasterio_transform_import_failed'}

    work = nodes.copy()
    if work.empty:
        return work, {'available': False, 'reason': 'no_nodes', 'node_count': 0}

    target_pairs = work.apply(_section_target_agreement_target_for_row, axis=1)
    work['section_target_z_m'] = [pair[0] for pair in target_pairs]
    work['section_target_role_class'] = [pair[1] for pair in target_pairs]
    work['section_target_target_stage'] = [pair[2] for pair in target_pairs]
    if 'geometry' in work.columns:
        xs = np.asarray([geom.x if geom is not None and not getattr(geom, 'is_empty', True) else np.nan for geom in work['geometry']], dtype=float)
        ys = np.asarray([geom.y if geom is not None and not getattr(geom, 'is_empty', True) else np.nan for geom in work['geometry']], dtype=float)
    else:
        xs_series = work['center_x'] if 'center_x' in work.columns else pd.Series(np.nan, index=work.index)
        ys_series = work['center_y'] if 'center_y' in work.columns else pd.Series(np.nan, index=work.index)
        xs = pd.to_numeric(xs_series, errors='coerce').to_numpy(dtype=float)
        ys = pd.to_numeric(ys_series, errors='coerce').to_numpy(dtype=float)
    fallback_xs_series = work['center_x'] if 'center_x' in work.columns else pd.Series(np.nan, index=work.index)
    fallback_ys_series = work['center_y'] if 'center_y' in work.columns else pd.Series(np.nan, index=work.index)
    fallback_xs = pd.to_numeric(fallback_xs_series, errors='coerce').to_numpy(dtype=float)
    fallback_ys = pd.to_numeric(fallback_ys_series, errors='coerce').to_numpy(dtype=float)
    xs = np.where(np.isfinite(xs), xs, fallback_xs)
    ys = np.where(np.isfinite(ys), ys, fallback_ys)
    work['comparison_x'] = xs
    work['comparison_y'] = ys

    valid_geom = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(valid_geom):
        return work, {'available': False, 'reason': 'missing_geometry', 'node_count': int(len(work)), 'geometry_valid_count': 0}

    rows, cols = rasterio.transform.rowcol(transform, xs, ys, op=np.floor)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    inside = (rows >= 0) & (rows < z_out.shape[0]) & (cols >= 0) & (cols < z_out.shape[1])
    sampled = np.full(len(work), np.nan, dtype=float)
    good = valid_geom & inside
    sampled[good] = z_out[rows[good], cols[good]]
    work['sampled_surface_z_m'] = sampled

    canonical_support = work.get('support_class_canonical', pd.Series('missing', index=work.index)).astype(str)
    active_target_source = work.get('active_interior_target_source', pd.Series('missing', index=work.index)).astype(str)
    target_vals = pd.to_numeric(work['section_target_z_m'], errors='coerce').to_numpy(dtype=float)
    explicit_effective = work.get('station_target_effective_present', pd.Series(False, index=work.index)).astype(bool).to_numpy()
    derived_effective = np.isfinite(target_vals) & work.get('section_target_target_stage', pd.Series('missing_target', index=work.index)).astype(str).ne('missing_target').to_numpy()
    effective_present = explicit_effective | derived_effective
    reasons = np.full(len(work), 'ok', dtype=object)
    reasons[~effective_present] = 'target_not_effective'
    reasons[effective_present & ~np.isfinite(target_vals)] = 'target_missing_numeric'
    reasons[np.isfinite(target_vals) & ~valid_geom] = 'missing_geometry'
    reasons[np.isfinite(target_vals) & valid_geom & ~inside] = 'outside_raster'
    reasons[np.isfinite(target_vals) & good & ~np.isfinite(sampled)] = 'sampled_nan'
    work['comparison_status'] = reasons
    available_mask = np.isfinite(target_vals) & np.isfinite(sampled)
    counts = {str(k): int(v) for k, v in pd.Series(reasons).value_counts(dropna=False).to_dict().items()}
    target_stage_counts = {str(k): int(v) for k, v in work.get('section_target_target_stage', pd.Series(dtype='object')).astype(str).value_counts(dropna=False).to_dict().items()}
    active_target_source_counts = {str(k): int(v) for k, v in active_target_source.value_counts(dropna=False).to_dict().items()}
    canonical_support_class_counts = {str(k): int(v) for k, v in canonical_support.value_counts(dropna=False).to_dict().items()}
    canonical_target_stage_count = int(work.get('section_target_target_stage', pd.Series(dtype='object')).astype(str).eq('canonical_active_target').sum())
    if np.any(available_mask):
        reason = 'prepared'
        failure_stage = None
    elif int(np.count_nonzero(effective_present)) == 0:
        reason = 'no_effective_targets'
        failure_stage = 'effective_target_selection'
    elif int(np.count_nonzero(active_target_source.ne('missing').to_numpy(dtype=bool))) == 0:
        reason = 'missing_active_interior_target_source'
        failure_stage = 'canonical_active_target_source'
    elif int(np.count_nonzero(np.isfinite(pd.to_numeric(work.get('active_interior_target_z_m', np.nan), errors='coerce').to_numpy(dtype=float)))) == 0:
        reason = 'missing_active_interior_target_z'
        failure_stage = 'canonical_active_target_z'
    elif int(np.count_nonzero(np.isfinite(target_vals))) == 0:
        reason = 'no_finite_target_z'
        failure_stage = 'target_numeric_construction'
    elif int(np.count_nonzero(valid_geom)) == 0:
        reason = 'missing_geometry'
        failure_stage = 'comparison_geometry'
    elif int(np.count_nonzero(valid_geom & inside)) == 0:
        reason = 'no_inside_raster_rows'
        failure_stage = 'raster_bounds'
    elif int(np.count_nonzero(np.isfinite(sampled))) == 0:
        reason = 'no_finite_sample_rows'
        failure_stage = 'raster_sampling'
    else:
        reason = 'no_finite_target_comparisons'
        failure_stage = 'comparison_join'
    prep = {
        'available': bool(np.any(available_mask)),
        'reason': reason,
        'failure_stage': failure_stage,
        'node_count': int(len(work)),
        'geometry_valid_count': int(np.count_nonzero(valid_geom)),
        'inside_raster_count': int(np.count_nonzero(valid_geom & inside)),
        'effective_target_count': int(np.count_nonzero(effective_present)),
        'explicit_effective_target_count': int(np.count_nonzero(explicit_effective)),
        'derived_effective_target_count': int(np.count_nonzero(derived_effective)),
        'finite_target_count': int(np.count_nonzero(np.isfinite(target_vals))),
        'finite_sample_count': int(np.count_nonzero(np.isfinite(sampled))),
        'comparison_count': int(np.count_nonzero(available_mask)),
        'comparison_status_counts': counts,
        'target_stage_counts': target_stage_counts,
        'active_target_source_counts': active_target_source_counts,
        'canonical_support_class_counts': canonical_support_class_counts,
        'uses_canonical_active_target': bool(canonical_target_stage_count > 0),
        'canonical_active_target_count': int(canonical_target_stage_count),
    }
    return work, prep

def _write_unavailable_section_target_or_role_receipts(*, river_dir: Path, work: pd.DataFrame, prep: dict[str, Any], logger: logging.Logger, kind: str) -> tuple[dict[str, str], dict[str, Any]]:
    if kind == 'section_target':
        profile_cols = [
            'component_id', 'station_m', 'node_role', 'component_support_class', 'station_support_regime',
            'station_target_local_authoritative_reconciled', 'section_target_role_class', 'comparison_status',
            'section_target_z_m', 'sampled_surface_z_m',
        ]
        profile_name = 'river_channel_surface_section_target_agreement_profile.csv'
        summary_name = 'river_channel_surface_section_target_agreement_summary.json'
        output_keys = {
            'river_channel_surface_section_target_agreement_profile.csv': str(river_dir / profile_name),
            'river_channel_surface_section_target_agreement_summary.json': str(river_dir / summary_name),
            'channel_surface_section_target_agreement_profile': str(river_dir / profile_name),
            'channel_surface_section_target_agreement_summary': str(river_dir / summary_name),
        }
        summary = {
            'available': False,
            'reason': str(prep.get('reason', 'not_available') or 'not_available'),
            'comparison_node_count': int(prep.get('comparison_count', 0) or 0),
            'node_count': int(prep.get('node_count', len(work))),
            'geometry_valid_count': int(prep.get('geometry_valid_count', 0) or 0),
            'inside_raster_count': int(prep.get('inside_raster_count', 0) or 0),
            'effective_target_count': int(prep.get('effective_target_count', 0) or 0),
            'finite_target_count': int(prep.get('finite_target_count', 0) or 0),
            'finite_sample_count': int(prep.get('finite_sample_count', 0) or 0),
            'comparison_status_counts': prep.get('comparison_status_counts', {}) if isinstance(prep.get('comparison_status_counts', {}), dict) else {},
            'failure_stage': prep.get('failure_stage'),
            'target_stage_counts': prep.get('target_stage_counts', {}) if isinstance(prep.get('target_stage_counts', {}), dict) else {},
            'uses_canonical_active_target': bool(prep.get('uses_canonical_active_target', False)),
            'canonical_active_target_count': int(prep.get('canonical_active_target_count', 0) or 0),
            'active_target_source_counts': prep.get('active_target_source_counts', {}) if isinstance(prep.get('active_target_source_counts', {}), dict) else {},
            'canonical_support_class_counts': prep.get('canonical_support_class_counts', {}) if isinstance(prep.get('canonical_support_class_counts', {}), dict) else {},
            'abs_error_m': _distribution(np.array([], dtype=float)),
            'signed_error_m': _distribution(np.array([], dtype=float)),
            'by_component_support_class': {},
            'by_local_authoritative_reconciled': {},
            'by_section_target_role_class': {},
            'weakest_role_class': None,
            'weakest_role_class_p95_abs_error_m': np.nan,
        }
    else:
        profile_cols = [
            'component_id', 'station_m', 'node_role', 'role_semantic_group', 'component_support_class', 'station_support_regime',
            'station_target_local_authoritative_reconciled', 'role_target_source_class', 'comparison_status',
            'role_target_z_m', 'sampled_surface_z_m',
        ]
        profile_name = 'river_channel_surface_role_agreement_profile.csv'
        summary_name = 'river_channel_surface_role_agreement_summary.json'
        output_keys = {
            'river_channel_surface_role_agreement_profile.csv': str(river_dir / profile_name),
            'river_channel_surface_role_agreement_summary.json': str(river_dir / summary_name),
            'channel_surface_role_agreement_profile': str(river_dir / profile_name),
            'channel_surface_role_agreement_summary': str(river_dir / summary_name),
        }
        summary = {
            'available': False,
            'reason': str(prep.get('reason', 'not_available') or 'not_available'),
            'comparison_node_count': int(prep.get('comparison_count', 0) or 0),
            'node_count': int(prep.get('node_count', len(work))),
            'geometry_valid_count': int(prep.get('geometry_valid_count', 0) or 0),
            'inside_raster_count': int(prep.get('inside_raster_count', 0) or 0),
            'effective_target_count': int(prep.get('effective_target_count', 0) or 0),
            'finite_target_count': int(prep.get('finite_target_count', 0) or 0),
            'finite_sample_count': int(prep.get('finite_sample_count', 0) or 0),
            'comparison_status_counts': prep.get('comparison_status_counts', {}) if isinstance(prep.get('comparison_status_counts', {}), dict) else {},
            'failure_stage': prep.get('failure_stage'),
            'target_stage_counts': prep.get('target_stage_counts', {}) if isinstance(prep.get('target_stage_counts', {}), dict) else {},
            'uses_canonical_active_target': bool(prep.get('uses_canonical_active_target', False)),
            'canonical_active_target_count': int(prep.get('canonical_active_target_count', 0) or 0),
            'active_target_source_counts': prep.get('active_target_source_counts', {}) if isinstance(prep.get('active_target_source_counts', {}), dict) else {},
            'canonical_support_class_counts': prep.get('canonical_support_class_counts', {}) if isinstance(prep.get('canonical_support_class_counts', {}), dict) else {},
            'abs_error_m': _distribution(np.array([], dtype=float)),
            'signed_error_m': _distribution(np.array([], dtype=float)),
            'by_role_semantic_group': {},
            'by_component_support_class': {},
            'by_local_authoritative_reconciled': {},
            'thalweg_agreement': {'count': 0, 'abs_error_m': _distribution(np.array([], dtype=float))},
            'inner_shape_agreement': {'count': 0, 'abs_error_m': _distribution(np.array([], dtype=float))},
            'bank_edge_agreement': {'count': 0, 'abs_error_m': _distribution(np.array([], dtype=float))},
            'weakest_role': None,
            'weakest_role_p95_abs_error_m': np.nan,
            'bank_minus_inner_p95_abs_error_m': np.nan,
            'inner_minus_thalweg_p95_abs_error_m': np.nan,
            'lateral_failure_mode': None,
            'lateral_failure_reason': 'No finite role-agreement comparisons were available.',
        }
    profile = work.loc[:, [c for c in profile_cols if c in work.columns]].copy() if not work.empty else pd.DataFrame(columns=profile_cols)
    missing_cols = [c for c in profile_cols if c not in profile.columns]
    for col in missing_cols:
        profile[col] = pd.Series(dtype='object')
    profile = profile.loc[:, profile_cols]
    profile_path = river_dir / profile_name
    profile.to_csv(profile_path, index=False)
    summary_path = river_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('[RIVER][FRAME] %s agreement unavailable: reason=%s comparisons=%d', 'Section-target' if kind == 'section_target' else 'Role', summary['reason'], int(summary['comparison_node_count']))
    return output_keys, summary


def _write_section_target_agreement_receipts(
    *,
    river_dir: Path,
    nodes: pd.DataFrame,
    z_out: np.ndarray,
    transform,
    logger: logging.Logger,
) -> tuple[dict[str, str], dict[str, Any]]:
    work, prep = _prepare_section_target_comparison_frame(nodes=nodes, z_out=z_out, transform=transform)
    if not prep.get('available', False):
        return _write_unavailable_section_target_or_role_receipts(river_dir=river_dir, work=work, prep=prep, logger=logger, kind='section_target')

    sampled = pd.to_numeric(work['sampled_surface_z_m'], errors='coerce').to_numpy(dtype=float)
    work['section_target_sampled_surface_z_m'] = sampled
    work['section_target_error_m'] = sampled - pd.to_numeric(work['section_target_z_m'], errors='coerce').to_numpy(dtype=float)
    work['section_target_abs_error_m'] = np.abs(work['section_target_error_m'].to_numpy(dtype=float))
    available_mask = np.isfinite(pd.to_numeric(work['section_target_z_m'], errors='coerce').to_numpy(dtype=float)) & np.isfinite(sampled)

    summary = {
        'available': True,
        'comparison_node_count': int(np.count_nonzero(available_mask)),
        'node_count': int(prep.get('node_count', len(work))),
        'geometry_valid_count': int(prep.get('geometry_valid_count', 0)),
        'inside_raster_count': int(prep.get('inside_raster_count', 0)),
        'effective_target_count': int(prep.get('effective_target_count', 0)),
        'finite_target_count': int(prep.get('finite_target_count', 0)),
        'finite_sample_count': int(prep.get('finite_sample_count', 0)),
        'comparison_status_counts': prep.get('comparison_status_counts', {}),
        'failure_stage': prep.get('failure_stage'),
        'target_stage_counts': prep.get('target_stage_counts', {}),
        'uses_canonical_active_target': bool(prep.get('uses_canonical_active_target', False)),
        'canonical_active_target_count': int(prep.get('canonical_active_target_count', 0) or 0),
        'active_target_source_counts': prep.get('active_target_source_counts', {}),
        'canonical_support_class_counts': prep.get('canonical_support_class_counts', {}),
        'abs_error_m': _distribution(work.loc[available_mask, 'section_target_abs_error_m'].to_numpy(dtype=float)),
        'signed_error_m': _distribution(work.loc[available_mask, 'section_target_error_m'].to_numpy(dtype=float)),
        'by_component_support_class': {},
        'by_local_authoritative_reconciled': {},
    }
    for key, col in [
        ('by_component_support_class', 'component_support_class'),
        ('by_local_authoritative_reconciled', 'station_target_local_authoritative_reconciled'),
        ('by_section_target_role_class', 'section_target_role_class'),
    ]:
        if col not in work.columns:
            continue
        bucket = {}
        for value, sub in work.loc[available_mask].groupby(col, dropna=False):
            errs = pd.to_numeric(sub['section_target_abs_error_m'], errors='coerce').to_numpy(dtype=float)
            errs = errs[np.isfinite(errs)]
            bucket[str(value)] = {'count': int(errs.size), 'abs_error_m': _distribution(errs)}
        summary[key] = bucket
    role_class_candidates = []
    for name, bucket in (summary.get('by_section_target_role_class', {}) if isinstance(summary.get('by_section_target_role_class', {}), dict) else {}).items():
        value = _p95_from_error_bucket(bucket)
        if np.isfinite(value):
            role_class_candidates.append((str(name), value))
    if role_class_candidates:
        summary['weakest_role_class'], weakest_role_class_p95 = max(role_class_candidates, key=lambda kv: kv[1])
        summary['weakest_role_class_p95_abs_error_m'] = float(weakest_role_class_p95)
    else:
        summary['weakest_role_class'] = None
        summary['weakest_role_class_p95_abs_error_m'] = np.nan

    profile_cols = [
        'component_id', 'station_m', 'node_role', 'component_support_class', 'station_support_regime',
        'station_target_local_authoritative_reconciled', 'section_target_role_class', 'section_target_target_stage', 'comparison_status',
        'section_target_z_m', 'section_target_sampled_surface_z_m', 'section_target_error_m', 'section_target_abs_error_m',
    ]
    profile = work.loc[:, [c for c in profile_cols if c in work.columns]].copy()
    profile = profile.sort_values(['component_id', 'station_m', 'node_role'], na_position='last')
    profile_path = river_dir / 'river_channel_surface_section_target_agreement_profile.csv'
    profile.to_csv(profile_path, index=False)
    summary_path = river_dir / 'river_channel_surface_section_target_agreement_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('[RIVER][FRAME] Section-target agreement: comparisons=%d p95_abs=%.3f m', int(summary['comparison_node_count']), float(summary['abs_error_m'].get('p95', 0.0) or 0.0))
    return ({
        'river_channel_surface_section_target_agreement_profile.csv': str(profile_path),
        'river_channel_surface_section_target_agreement_summary.json': str(summary_path),
        'channel_surface_section_target_agreement_profile': str(profile_path),
        'channel_surface_section_target_agreement_summary': str(summary_path),
    }, summary)

def _write_role_agreement_receipts(
    *,
    river_dir: Path,
    nodes: pd.DataFrame,
    z_out: np.ndarray,
    transform,
    logger: logging.Logger,
) -> tuple[dict[str, str], dict[str, Any]]:
    work, prep = _prepare_section_target_comparison_frame(nodes=nodes, z_out=z_out, transform=transform)
    if not prep.get('available', False):
        return _write_unavailable_section_target_or_role_receipts(river_dir=river_dir, work=work, prep=prep, logger=logger, kind='role')

    work['role_target_z_m'] = pd.to_numeric(work['section_target_z_m'], errors='coerce').to_numpy(dtype=float)
    work['role_target_source_class'] = work.get('section_target_role_class', pd.Series('missing', index=work.index)).astype(str)
    work['role_semantic_group'] = work.get('node_role', pd.Series(dtype='object')).map(_role_semantic_group).fillna('missing')
    sampled = pd.to_numeric(work['sampled_surface_z_m'], errors='coerce').to_numpy(dtype=float)
    work['role_sampled_surface_z_m'] = sampled
    work['role_error_m'] = sampled - pd.to_numeric(work['role_target_z_m'], errors='coerce').to_numpy(dtype=float)
    work['role_abs_error_m'] = np.abs(work['role_error_m'].to_numpy(dtype=float))
    available_mask = np.isfinite(work['role_target_z_m'].to_numpy(dtype=float)) & np.isfinite(sampled)

    summary = {
        'available': True,
        'comparison_node_count': int(np.count_nonzero(available_mask)),
        'node_count': int(prep.get('node_count', len(work))),
        'geometry_valid_count': int(prep.get('geometry_valid_count', 0)),
        'inside_raster_count': int(prep.get('inside_raster_count', 0)),
        'effective_target_count': int(prep.get('effective_target_count', 0)),
        'finite_target_count': int(prep.get('finite_target_count', 0)),
        'finite_sample_count': int(prep.get('finite_sample_count', 0)),
        'comparison_status_counts': prep.get('comparison_status_counts', {}),
        'failure_stage': prep.get('failure_stage'),
        'target_stage_counts': prep.get('target_stage_counts', {}),
        'uses_canonical_active_target': bool(prep.get('uses_canonical_active_target', False)),
        'canonical_active_target_count': int(prep.get('canonical_active_target_count', 0) or 0),
        'active_target_source_counts': prep.get('active_target_source_counts', {}),
        'canonical_support_class_counts': prep.get('canonical_support_class_counts', {}),
        'abs_error_m': _distribution(work.loc[available_mask, 'role_abs_error_m'].to_numpy(dtype=float)),
        'signed_error_m': _distribution(work.loc[available_mask, 'role_error_m'].to_numpy(dtype=float)),
        'by_role_semantic_group': {},
        'by_component_support_class': {},
        'by_local_authoritative_reconciled': {},
    }
    for key, col in [
        ('by_role_semantic_group', 'role_semantic_group'),
        ('by_component_support_class', 'component_support_class'),
        ('by_local_authoritative_reconciled', 'station_target_local_authoritative_reconciled'),
    ]:
        if col not in work.columns:
            continue
        bucket = {}
        for value, sub in work.loc[available_mask].groupby(col, dropna=False):
            errs = pd.to_numeric(sub['role_abs_error_m'], errors='coerce').to_numpy(dtype=float)
            errs = errs[np.isfinite(errs)]
            bucket[str(value)] = {'count': int(errs.size), 'abs_error_m': _distribution(errs)}
        summary[key] = bucket
    summary['thalweg_agreement'] = summary['by_role_semantic_group'].get('thalweg', {'count': 0, 'abs_error_m': {}})
    summary['inner_shape_agreement'] = summary['by_role_semantic_group'].get('inner_shape', {'count': 0, 'abs_error_m': {}})
    summary['bank_edge_agreement'] = summary['by_role_semantic_group'].get('bank_edge', {'count': 0, 'abs_error_m': {}})
    summary.update(_classify_role_agreement_focus(
        thalweg_p95=_p95_from_error_bucket(summary['thalweg_agreement']),
        inner_p95=_p95_from_error_bucket(summary['inner_shape_agreement']),
        bank_p95=_p95_from_error_bucket(summary['bank_edge_agreement']),
    ))

    profile_cols = [
        'component_id', 'station_m', 'node_role', 'role_semantic_group', 'component_support_class', 'station_support_regime',
        'station_target_local_authoritative_reconciled', 'role_target_source_class', 'section_target_target_stage', 'comparison_status',
        'role_target_z_m', 'role_sampled_surface_z_m', 'role_error_m', 'role_abs_error_m',
    ]
    profile = work.loc[:, [c for c in profile_cols if c in work.columns]].copy()
    profile = profile.sort_values(['component_id', 'station_m', 'node_role'], na_position='last')
    profile_path = river_dir / 'river_channel_surface_role_agreement_profile.csv'
    profile.to_csv(profile_path, index=False)
    summary_path = river_dir / 'river_channel_surface_role_agreement_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('[RIVER][FRAME] Role agreement: comparisons=%d p95_abs=%.3f m', int(summary['comparison_node_count']), float(summary['abs_error_m'].get('p95', 0.0) or 0.0))
    return ({
        'river_channel_surface_role_agreement_profile.csv': str(profile_path),
        'river_channel_surface_role_agreement_summary.json': str(summary_path),
        'channel_surface_role_agreement_profile': str(profile_path),
        'channel_surface_role_agreement_summary': str(summary_path),
    }, summary)

def _support_aware_longitudinal_blend_weights(*, support_distance_m: np.ndarray, interior_weight: np.ndarray) -> np.ndarray:
    distances = np.asarray(support_distance_m, dtype=float)
    weights = np.zeros_like(distances, dtype=float)
    finite = np.isfinite(distances)
    if np.any(finite):
        d = distances[finite]
        base = np.interp(np.clip(d, 0.0, 1000.0), [0.0, 150.0, 500.0, 1000.0], [0.0, 0.08, 0.22, CHANNEL_SURFACE_LONGITUDINAL_SMOOTHING_BASE_BLEND])
        weights[finite] = base
    missing = ~finite
    if np.any(missing):
        weights[missing] = CHANNEL_SURFACE_LONGITUDINAL_SMOOTHING_BASE_BLEND * 0.6
    weights *= np.clip(np.asarray(interior_weight, dtype=float), 0.0, 1.0)
    return np.clip(weights, 0.0, CHANNEL_SURFACE_LONGITUDINAL_SMOOTHING_BASE_BLEND)


def _apply_component_longitudinal_smoothing(*, stations_m: np.ndarray, values_z: np.ndarray, support_distance_m: np.ndarray, interior_weight: np.ndarray, hard_lock_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    stations = np.asarray(stations_m, dtype=float)
    values = np.asarray(values_z, dtype=float)
    support_distance = np.asarray(support_distance_m, dtype=float)
    interior = np.clip(np.asarray(interior_weight, dtype=float), 0.0, 1.0)
    hard_lock = np.zeros(len(values), dtype=bool) if hard_lock_mask is None else np.asarray(hard_lock_mask, dtype=bool)
    result = values.copy()
    summary = {
        'available': False,
        'candidate_count': int(len(values)),
        'eligible_count': 0,
        'applied_count': 0,
        'changed_count': 0,
        'abs_adjustment_m': _distribution(np.array([], dtype=float)),
        'mean_blend_weight': 0.0,
        'reason': 'insufficient_samples',
    }
    finite = np.isfinite(stations) & np.isfinite(values)
    if np.count_nonzero(finite) < CHANNEL_SURFACE_LONGITUDINAL_SMOOTHING_MIN_SAMPLES:
        return result, summary
    unique_station_count = int(np.unique(np.round(stations[finite], 6)).size)
    if unique_station_count < 4:
        summary['reason'] = 'insufficient_unique_stations'
        return result, summary
    sigma = max(float(np.nanmedian(np.diff(np.unique(np.sort(stations[finite]))))) * CHANNEL_SURFACE_LONGITUDINAL_SMOOTHING_SIGMA_STATIONS, 1.0)
    blend_weights = _support_aware_longitudinal_blend_weights(support_distance_m=support_distance, interior_weight=interior)
    eligible = finite & (blend_weights > 0.0) & (~hard_lock)
    summary['eligible_count'] = int(np.count_nonzero(eligible))
    if summary['eligible_count'] == 0:
        summary['reason'] = 'no_eligible_samples'
        summary['available'] = True
        return result, summary
    for i in np.where(eligible)[0]:
        neighbor_mask = finite & (~hard_lock)
        neighbor_mask[i] = False
        if np.count_nonzero(neighbor_mask) < 2:
            continue
        dist = np.abs(stations[neighbor_mask] - stations[i])
        kernel = np.exp(-0.5 * (dist / sigma) ** 2)
        if not np.any(kernel > 0.0):
            continue
        smooth_target = float(np.average(values[neighbor_mask], weights=kernel))
        requested = smooth_target - values[i]
        max_shift = float(np.interp(np.clip(support_distance[i] if np.isfinite(support_distance[i]) else 1000.0, 0.0, 1000.0), [0.0, 150.0, 500.0, 1000.0], [0.05, 0.18, 0.45, CHANNEL_SURFACE_LONGITUDINAL_SMOOTHING_MAX_SHIFT_M]))
        shift = float(np.clip(requested * blend_weights[i], -max_shift, max_shift))
        if abs(shift) <= 1.0e-6:
            continue
        result[i] = values[i] + shift
    adjustments = result - values
    changed = finite & (np.abs(adjustments) > 1.0e-6)
    summary.update({
        'available': True,
        'applied_count': int(np.count_nonzero(eligible)),
        'changed_count': int(np.count_nonzero(changed)),
        'abs_adjustment_m': _distribution(np.abs(adjustments[changed])) if np.any(changed) else _distribution(np.array([], dtype=float)),
        'mean_blend_weight': float(np.nanmean(blend_weights[eligible])) if np.any(eligible) else 0.0,
        'reason': 'applied' if np.any(changed) else 'no_nonzero_adjustments',
    })
    return result, summary


def _write_channel_surface_longitudinal_smoothing_receipts(*, river_dir: Path, rows: list[dict[str, Any]], logger: logging.Logger) -> tuple[dict[str, str], dict[str, Any]]:
    if not rows:
        return {}, {'available': False, 'reason': 'no_component_rows', 'component_count': 0}
    df = pd.DataFrame(rows).sort_values(['component_id'], na_position='last')
    abs_adj = pd.to_numeric(df.get('abs_adjustment_mean_m', pd.Series(dtype=float)), errors='coerce').to_numpy(dtype=float)
    summary = {
        'available': True,
        'component_count': int(len(df)),
        'eligible_component_count': int(np.count_nonzero(pd.to_numeric(df.get('eligible_count', pd.Series(dtype=float)), errors='coerce').fillna(0).to_numpy(dtype=float) > 0.0)),
        'changed_component_count': int(np.count_nonzero(pd.to_numeric(df.get('changed_count', pd.Series(dtype=float)), errors='coerce').fillna(0).to_numpy(dtype=float) > 0.0)),
        'eligible_pixel_count': int(pd.to_numeric(df.get('eligible_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()),
        'changed_pixel_count': int(pd.to_numeric(df.get('changed_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()),
        'abs_adjustment_mean_m': _distribution(abs_adj[np.isfinite(abs_adj)]),
        'reason_counts': {str(k): int(v) for k, v in df.get('reason', pd.Series(dtype='object')).astype(str).value_counts(dropna=False).to_dict().items()},
    }
    profile_path = river_dir / 'river_channel_surface_longitudinal_smoothing_summary.csv'
    df.to_csv(profile_path, index=False)
    summary_path = river_dir / 'river_channel_surface_longitudinal_smoothing_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('[RIVER][FRAME] Channel longitudinal smoothing: eligible_components=%d changed_pixels=%d', int(summary['eligible_component_count']), int(summary['changed_pixel_count']))
    return ({
        'river_channel_surface_longitudinal_smoothing_summary.csv': str(profile_path),
        'river_channel_surface_longitudinal_smoothing_summary.json': str(summary_path),
        'channel_surface_longitudinal_smoothing_profile': str(profile_path),
        'channel_surface_longitudinal_smoothing_summary': str(summary_path),
    }, summary)


def _write_channel_surface_effect_receipts(
    *,
    river_dir: Path,
    component_effect_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not component_effect_rows:
        summary = {
            'available': False,
            'reason': 'no_component_effect_rows',
            'component_count': 0,
        }
        return {}, summary
    effect_df = pd.DataFrame(component_effect_rows)
    if not effect_df.empty and 'component_id' in effect_df.columns:
        effect_df = effect_df.sort_values(['component_id'], na_position='last')
    summary = {
        'available': True,
        'component_count': int(len(effect_df)),
        'selected_for_thalweg_render_component_count': int(np.count_nonzero(effect_df.get('selected_for_thalweg_render', pd.Series(dtype=bool)).fillna(False).astype(bool).to_numpy(dtype=bool))) if not effect_df.empty else 0,
        'weak_support_semantic_component_count': int(np.count_nonzero(effect_df.get('weak_support_semantic_present', pd.Series(dtype=bool)).fillna(False).astype(bool).to_numpy(dtype=bool))) if not effect_df.empty else 0,
        'candidate_pixel_count': int(pd.to_numeric(effect_df.get('candidate_pixel_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not effect_df.empty else 0,
        'thalweg_render_delta_pixel_count': int(pd.to_numeric(effect_df.get('thalweg_render_delta_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not effect_df.empty else 0,
        'section_target_applied_component_count': int(np.count_nonzero(pd.to_numeric(effect_df.get('section_target_applied_count', pd.Series(dtype=float)), errors='coerce').fillna(0).to_numpy(dtype=float) > 0.0)) if not effect_df.empty else 0,
        'section_target_applied_pixel_count': int(pd.to_numeric(effect_df.get('section_target_applied_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not effect_df.empty else 0,
        'authoritative_transition_applied_component_count': int(np.count_nonzero(pd.to_numeric(effect_df.get('authoritative_transition_applied_count', pd.Series(dtype=float)), errors='coerce').fillna(0).to_numpy(dtype=float) > 0.0)) if not effect_df.empty else 0,
        'authoritative_transition_applied_pixel_count': int(pd.to_numeric(effect_df.get('authoritative_transition_applied_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not effect_df.empty else 0,
        'final_changed_pixel_count': int(pd.to_numeric(effect_df.get('final_changed_pixel_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not effect_df.empty else 0,
        'fallback_delta_abs_m': _distribution(pd.to_numeric(effect_df.get('fallback_delta_abs_mean_m', pd.Series(dtype=float)), errors='coerce').to_numpy(dtype=float)) if not effect_df.empty else _distribution(np.array([], dtype=float)),
        'render_mode_counts': {str(k): int(v) for k, v in effect_df.get('render_mode', pd.Series(dtype='object')).astype(str).value_counts(dropna=False).to_dict().items()} if 'render_mode' in effect_df.columns else {},
        'selection_reason_counts': {str(k): int(v) for k, v in effect_df.get('selection_reason', pd.Series(dtype='object')).astype(str).value_counts(dropna=False).to_dict().items()} if 'selection_reason' in effect_df.columns else {},
    }
    profile_path = river_dir / 'river_channel_surface_effect_summary.csv'
    effect_df.to_csv(profile_path, index=False)
    summary_path = river_dir / 'river_channel_surface_effect_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('[RIVER][FRAME] Channel surface effect summary: thalweg_components=%d candidate_pixels=%d final_changed=%d transition_pixels=%d', int(summary['selected_for_thalweg_render_component_count']), int(summary['candidate_pixel_count']), int(summary['final_changed_pixel_count']), int(summary['authoritative_transition_applied_pixel_count']))
    return ({
        'river_channel_surface_effect_summary.csv': str(profile_path),
        'river_channel_surface_effect_summary.json': str(summary_path),
        'channel_surface_effect_summary': str(summary_path),
        'channel_surface_effect_profile': str(profile_path),
    }, summary)


def _select_component_render_mode(sub: pd.DataFrame) -> tuple[str, list[tuple[str, float]], str, dict[str, Any]]:
    authoritative_like = sub['graph_solver_support_class'].astype(str).isin(['authoritative_locked'])
    if 'z_source' in sub.columns:
        authoritative_like = authoritative_like | sub['z_source'].astype(str).isin(['authoritative_in_channel'])
    auth_share = float(authoritative_like.mean()) if len(sub) else 0.0
    has_xs_like = bool(
        sub['node_role'].astype(str).eq('thalweg').any()
        and sub.get('station_support_mode', pd.Series([], dtype='object')).astype(str).isin(['xs_supported', 'xs_residual_only']).any()
    )
    support_regime_weak = False
    if 'station_support_regime' in sub.columns:
        support_regime_weak = bool(sub['station_support_regime'].astype(str).isin(list(WEAK_SUPPORT_CLASSES)).any())
    longitudinal_weak = False
    if 'longitudinal_support_regime' in sub.columns:
        longitudinal_weak = bool(sub['longitudinal_support_regime'].astype(str).eq(LONGITUDINAL_WEAK_SUPPORT).any())
    rebuild_regimes = sub.get('station_rebuild_regime', pd.Series([], dtype='object')).astype(str)
    rebuild_eligible = bool(rebuild_regimes.eq('eligible').any())
    blocked_no_inner_rebuildable = bool(rebuild_regimes.eq('blocked_no_inner_rebuildable_nodes').any())
    xs_disabled_component = False
    if {'target_xs_realism_allowed', 'target_xs_residual_allowed'}.issubset(sub.columns):
        xs_realism_allowed = sub['target_xs_realism_allowed'].fillna(False).astype(bool).to_numpy(dtype=bool)
        xs_residual_allowed = sub['target_xs_residual_allowed'].fillna(False).astype(bool).to_numpy(dtype=bool)
        xs_disabled_component = bool((not np.any(xs_realism_allowed)) and (not np.any(xs_residual_allowed)))
    channel_core_changed = False
    if {'primary_surface_rebuild_applied', 'node_role'}.issubset(sub.columns):
        applied = sub['primary_surface_rebuild_applied'].fillna(False).astype(bool).to_numpy(dtype=bool)
        channel_roles = sub['node_role'].astype(str).isin(['thalweg', 'left_inner', 'right_inner']).to_numpy(dtype=bool)
        channel_core_changed = bool(np.any(applied & channel_roles))
    role_names = sub['node_role'].astype(str)
    has_inner_roles = bool(role_names.isin(sorted(SUBORDINATE_INNER_SHAPE_ROLES)).any())
    has_banks = bool(BANK_EDGE_ROLES.issubset(set(role_names.tolist())))
    has_thalweg = bool(role_names.isin(sorted(THALWEG_ROLES)).any())
    no_inner_longitudinal_candidate = bool(
        xs_disabled_component and blocked_no_inner_rebuildable and (not has_inner_roles) and has_banks and has_thalweg
    )
    component_support_classes = sub.get('component_support_class', pd.Series([], dtype='object')).astype(str)
    component_support_counts = component_support_classes.value_counts(dropna=False) if len(component_support_classes) else pd.Series(dtype='int64')
    dominant_component_support_class = str(component_support_counts.index[0]) if len(component_support_counts) else 'unknown'
    unsupported_component = dominant_component_support_class in {'unsupported_mainstem', 'unsupported_side_component', 'tiny_detached_component'}
    backbone_led_inner_targets = pd.to_numeric(sub.get('primary_surface_backbone_led_inner_targets', pd.Series([], dtype=float)), errors='coerce').fillna(False).astype(bool)
    bank_margin_damped = pd.to_numeric(sub.get('primary_surface_bank_margin_damped', pd.Series([], dtype=float)), errors='coerce').fillna(False).astype(bool)
    backbone_led_share = float(backbone_led_inner_targets.mean()) if len(backbone_led_inner_targets) else 0.0
    bank_margin_damped_share = float(bank_margin_damped.mean()) if len(bank_margin_damped) else 0.0
    weak_support_present = bool(
        support_regime_weak
        or longitudinal_weak
        or rebuild_eligible
        or no_inner_longitudinal_candidate
        or (unsupported_component and has_banks and has_thalweg and (backbone_led_share > 0.0 or auth_share < 0.50))
    )
    backbone_reference_implausible = False
    if has_thalweg and 'backbone_bed_z_m' in sub.columns and 'bed_z_m' in sub.columns:
        th_rows = sub.loc[role_names.eq('thalweg')].copy()
        th_backbone = pd.to_numeric(th_rows.get('backbone_bed_z_m', np.nan), errors='coerce').to_numpy(dtype=float)
        th_bed = pd.to_numeric(th_rows.get('bed_z_m', np.nan), errors='coerce').to_numpy(dtype=float)
        valid_backbone = np.isfinite(th_backbone) & np.isfinite(th_bed)
        if np.any(valid_backbone):
            comp_bed = pd.to_numeric(sub.get('bed_z_m', np.nan), errors='coerce').to_numpy(dtype=float)
            finite_comp_bed = comp_bed[np.isfinite(comp_bed)]
            if finite_comp_bed.size >= 2:
                bed_span = float(np.nanpercentile(finite_comp_bed, 95) - np.nanpercentile(finite_comp_bed, 5))
            elif finite_comp_bed.size == 1:
                bed_span = 0.0
            else:
                bed_span = np.nan
            plausible_tol = max(5.0, 4.0 * bed_span) if np.isfinite(bed_span) else 5.0
            backbone_reference_implausible = bool(np.nanmedian(np.abs(th_backbone[valid_backbone] - th_bed[valid_backbone])) > plausible_tol)
    selector = {
        'component_id': str(sub['component_id'].iloc[0]) if 'component_id' in sub.columns and len(sub) else '',
        'node_count': int(len(sub)),
        'auth_share': float(auth_share),
        'has_xs_like': bool(has_xs_like),
        'support_regime_weak': bool(support_regime_weak),
        'longitudinal_weak': bool(longitudinal_weak),
        'rebuild_eligible': bool(rebuild_eligible),
        'blocked_no_inner_rebuildable': bool(blocked_no_inner_rebuildable),
        'xs_disabled_component': bool(xs_disabled_component),
        'no_inner_longitudinal_candidate': bool(no_inner_longitudinal_candidate),
        'weak_support_present': bool(weak_support_present),
        'channel_core_changed': bool(channel_core_changed),
        'has_inner_roles': bool(has_inner_roles),
        'has_banks': bool(has_banks),
        'has_thalweg': bool(has_thalweg),
        'dominant_component_support_class': str(dominant_component_support_class),
        'unsupported_component': bool(unsupported_component),
        'backbone_led_inner_target_share': float(backbone_led_share),
        'bank_margin_damped_share': float(bank_margin_damped_share),
        'backbone_reference_implausible': bool(backbone_reference_implausible),
    }
    if auth_share >= 0.85 and len(sub) >= 250 and (not has_xs_like) and (not weak_support_present) and (not channel_core_changed) and (not has_inner_roles):
        selector['selection_reason'] = 'strong_authoritative_no_inner_or_weak_support'
        selector['render_mode'] = 'authoritative_fast_path'
        return 'authoritative_fast_path', list(FAST_ROLE_ORDER), 'strong_authoritative_no_inner_or_weak_support', selector
    if weak_support_present and (not backbone_reference_implausible) and has_banks and has_thalweg:
        active_roles = list(ROLE_ORDER) if has_inner_roles else list(FAST_ROLE_ORDER)
        selector['selection_reason'] = 'weak_support_thalweg_dominant'
        selector['render_mode'] = 'thalweg_dominant_scaffold'
        return 'thalweg_dominant_scaffold', active_roles, 'weak_support_thalweg_dominant', selector
    if channel_core_changed or has_inner_roles:
        selector['selection_reason'] = 'preserve_inner_roles_and_weak_support'
        selector['render_mode'] = 'full_scaffold'
        return 'full_scaffold', list(ROLE_ORDER), 'preserve_inner_roles_and_weak_support', selector
    selector['selection_reason'] = 'default_full_scaffold'
    selector['render_mode'] = 'full_scaffold'
    return 'full_scaffold', list(ROLE_ORDER), 'default_full_scaffold', selector

INFLUENCE_CLASS_CODES = {
    "missing": 0,
    "authoritative_only": 1,
    "xs_only": 2,
    "mixed_authoritative_xs": 3,
    "other_scaffold_only": 4,
}


def _bool_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))




def _compute_authoritative_transition_pixel_distance_m(*, auth_mask: np.ndarray | None, corridor: np.ndarray, transform: Any, template_crs: Any) -> np.ndarray | None:
    """Compute local pixel-level distance to nearest authoritative support cell in meters."""
    if auth_mask is None or template_crs is None:
        return None
    try:
        is_projected = bool(getattr(template_crs, 'is_projected', False))
    except Exception:
        is_projected = False
    if not is_projected:
        return None
    support = np.isfinite(auth_mask) & (auth_mask > 0) & np.asarray(corridor, dtype=bool)
    if not np.any(support):
        return None
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return None
    try:
        xres = float(abs(getattr(transform, 'a', np.nan)))
        yres = float(abs(getattr(transform, 'e', np.nan)))
    except Exception:
        return None
    if not (np.isfinite(xres) and np.isfinite(yres) and xres > 0.0 and yres > 0.0):
        return None
    dist = distance_transform_edt(~support, sampling=(yres, xres)).astype(np.float32)
    dist[~np.asarray(corridor, dtype=bool)] = np.float32(np.nan)
    dist[support] = np.float32(0.0)
    return dist


def _combine_support_distance_arrays(primary_distance_m: np.ndarray, pixel_distance_m: np.ndarray | None) -> np.ndarray:
    combined = np.asarray(primary_distance_m, dtype=float).copy()
    if pixel_distance_m is None:
        return combined
    pix = np.asarray(pixel_distance_m, dtype=float)
    if pix.shape != combined.shape:
        return combined
    both = np.isfinite(combined) & np.isfinite(pix)
    combined[both] = np.minimum(combined[both], pix[both])
    fill = (~np.isfinite(combined)) & np.isfinite(pix)
    combined[fill] = pix[fill]
    return combined

def _resolve_station_support_distance_series(th: pd.DataFrame) -> np.ndarray:
    """Resolve per-station authoritative bed support distance with disciplined fallbacks.

    Prefer the nearest finite support-distance diagnostic available at each station rather
    than letting a coarser early field permanently block a closer later field. This keeps
    transition weighting tied to the nearest credible support proximity instead of
    collapsing to zero because only a component-scale distance survived first.
    """
    if th.empty:
        return np.asarray([], dtype=float)
    resolved = np.full(len(th), np.nan, dtype=float)
    for field in (
        'station_authoritative_bed_support_distance_m',
        'authoritative_reconciliation_support_distance_m',
        'profile_authoritative_bed_support_distance_m',
        'authoritative_bed_support_point_distance_m',
        'component_support_median_distance_m',
    ):
        if field not in th.columns:
            continue
        vals = pd.to_numeric(th[field], errors='coerce').to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        if np.any(~np.isfinite(resolved) & finite):
            fill = ~np.isfinite(resolved) & finite
            resolved[fill] = vals[fill]
        both = np.isfinite(resolved) & finite
        if np.any(both):
            resolved[both] = np.minimum(resolved[both], vals[both])
    return resolved


def _write_xs_propagation_audit(
    *,
    river_dir: Path,
    audit: Dict[str, Any],
) -> Path:
    path = river_dir / 'river_xs_propagation_audit.json'
    path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return path


def _write_active_driver_receipts(*, river_dir: Path, nodes: pd.DataFrame) -> tuple[Path, Path]:
    role_series = nodes.get("node_role", pd.Series(["missing"] * len(nodes), index=nodes.index)).astype(str)
    preferred = nodes.loc[role_series.eq("thalweg")]
    if preferred.empty:
        preferred = nodes
    driver_fields = [
        "component_id",
        "station_m",
        "bank_pair_fit_z_m",
        "active_core_support_source",
        "active_core_support_bank_reference_m",
        "active_core_support_bank_offset_m",
        "authoritative_anchor_present",
        "authoritative_anchor_curve_present",
        "residual_shape_mode",
        "xs_realism_target_source",
        "primary_surface_rebuild_target_source",
        "support_class_canonical",
        "active_interior_target_source",
        "active_interior_target_z_m",
        "primary_surface_backbone_led_inner_targets",
        "primary_surface_backbone_led_inner_relief_scale",
        "primary_surface_backbone_led_inner_reason",
        "primary_surface_bank_margin_damped",
        "post_rebuild_monotone_applied",
        "station_authoritative_bed_support_distance_m",
        "generalized_longitudinal_bed_local_auth_taper_m",
        "generalized_longitudinal_bed_local_auth_reconciliation_weight",
        "authoritative_reconciliation_delta_m",
        "authoritative_reconciliation_weight",
        "authoritative_reconciliation_confidence",
        "station_target_local_authoritative_reconciled",
    ]
    present_fields = [field for field in driver_fields if field in preferred.columns]
    preferred = preferred.loc[:, present_fields].copy()
    preferred["component_id"] = preferred.get("component_id", pd.Series(["missing"] * len(preferred), index=preferred.index)).astype(str)
    preferred["_station_key"] = pd.to_numeric(preferred.get("station_m", pd.Series([np.nan] * len(preferred), index=preferred.index)), errors="coerce")
    preferred = preferred.sort_values(["component_id", "_station_key"], na_position="last").copy()
    station_rows = preferred.groupby(["component_id", "_station_key"], dropna=False, as_index=False).first()
    def _series(name: str, default, numeric: bool = False):
        if name in station_rows.columns:
            s = station_rows[name]
        else:
            s = pd.Series([default] * len(station_rows), index=station_rows.index)
        return pd.to_numeric(s, errors="coerce") if numeric else s
    bank_pair_fit = _series("bank_pair_fit_z_m", np.nan, numeric=True)
    out = pd.DataFrame({
        "component_id": station_rows.get("component_id", pd.Series(dtype=str)).astype(str),
        "station_m": pd.to_numeric(station_rows.get("_station_key", pd.Series(dtype=float)), errors="coerce"),
        "bank_pair_fit_used": bank_pair_fit.notna(),
        "active_core_support_source": _series("active_core_support_source", "missing").astype(str),
        "active_core_support_bank_reference_m": _series("active_core_support_bank_reference_m", np.nan, numeric=True),
        "active_core_support_bank_offset_m": _series("active_core_support_bank_offset_m", np.nan, numeric=True),
        "authoritative_anchor_present": _series("authoritative_anchor_present", False).fillna(False).astype(bool),
        "authoritative_anchor_curve_present": _series("authoritative_anchor_curve_present", False).fillna(False).astype(bool),
        "residual_shape_mode": _series("residual_shape_mode", "missing").astype(str),
        "xs_realism_target_source": _series("xs_realism_target_source", "missing").astype(str),
        "primary_surface_rebuild_target_source": _series("primary_surface_rebuild_target_source", "missing").astype(str),
        "support_class_canonical": _series("support_class_canonical", "missing").astype(str),
        "active_interior_target_source": _series("active_interior_target_source", "missing").astype(str),
        "active_interior_target_z_m": _series("active_interior_target_z_m", np.nan, numeric=True),
        "primary_surface_backbone_led_inner_targets": _series("primary_surface_backbone_led_inner_targets", False).fillna(False).astype(bool),
        "primary_surface_backbone_led_inner_relief_scale": _series("primary_surface_backbone_led_inner_relief_scale", np.nan, numeric=True),
        "primary_surface_backbone_led_inner_reason": _series("primary_surface_backbone_led_inner_reason", "missing").astype(str),
        "primary_surface_bank_margin_damped": _series("primary_surface_bank_margin_damped", False).fillna(False).astype(bool),
        "post_rebuild_monotone_applied": _series("post_rebuild_monotone_applied", False).fillna(False).astype(bool),
        "station_authoritative_bed_support_distance_m": _series("station_authoritative_bed_support_distance_m", np.nan, numeric=True),
        "generalized_longitudinal_bed_local_auth_taper_m": _series("generalized_longitudinal_bed_local_auth_taper_m", np.nan, numeric=True),
        "generalized_longitudinal_bed_local_auth_reconciliation_weight": _series("generalized_longitudinal_bed_local_auth_reconciliation_weight", np.nan, numeric=True),
        "authoritative_reconciliation_delta_m": _series("authoritative_reconciliation_delta_m", np.nan, numeric=True),
        "authoritative_reconciliation_weight": _series("authoritative_reconciliation_weight", np.nan, numeric=True),
        "authoritative_reconciliation_confidence": _series("authoritative_reconciliation_confidence", np.nan, numeric=True),
        "station_target_local_authoritative_reconciled": _series("station_target_local_authoritative_reconciled", False).fillna(False).astype(bool),
    })
    csv_path = river_dir / "river_active_driver_station_table.csv"
    out.to_csv(csv_path, index=False)
    summary = {
        "station_count": int(len(out)),
        "bank_pair_fit_used_count": int(out["bank_pair_fit_used"].sum()),
        "active_core_support_source_counts": {str(k): int(v) for k, v in out["active_core_support_source"].value_counts(dropna=False).to_dict().items()},
        "residual_shape_mode_counts": {str(k): int(v) for k, v in out["residual_shape_mode"].value_counts(dropna=False).to_dict().items()},
        "xs_realism_target_source_counts": {str(k): int(v) for k, v in out["xs_realism_target_source"].value_counts(dropna=False).to_dict().items()},
        "primary_surface_rebuild_target_source_counts": {str(k): int(v) for k, v in out["primary_surface_rebuild_target_source"].value_counts(dropna=False).to_dict().items()},
        "primary_driver_kind": "canonical_v314_active_target",
        "primary_active_target_source_counts": {str(k): int(v) for k, v in out["active_interior_target_source"].value_counts(dropna=False).to_dict().items()},
        "canonical_support_class_counts": {str(k): int(v) for k, v in out["support_class_canonical"].value_counts(dropna=False).to_dict().items()},
        "backbone_led_station_count": int(out["primary_surface_backbone_led_inner_targets"].sum()),
        "bank_margin_damped_station_count": int(out["primary_surface_bank_margin_damped"].sum()),
        "backbone_led_reason_counts": {str(k): int(v) for k, v in out["primary_surface_backbone_led_inner_reason"].value_counts(dropna=False).to_dict().items()},
        "backbone_led_inner_relief_scale_summary": _distribution(pd.to_numeric(out["primary_surface_backbone_led_inner_relief_scale"], errors="coerce").to_numpy(dtype=float)),
        "post_rebuild_monotone_applied_count": int(out["post_rebuild_monotone_applied"].sum()),
        "local_authoritative_reconciliation_count": int(out["station_target_local_authoritative_reconciled"].sum()),
        "local_authoritative_reconciliation_max_abs_m": (lambda _vals: float(np.nanmax(np.abs(_vals))) if np.any(np.isfinite(_vals)) else 0.0)(pd.to_numeric(out["authoritative_reconciliation_delta_m"], errors="coerce").to_numpy(dtype=float)) if len(out) else 0.0,
    }
    json_path = river_dir / "river_active_driver_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


def _support_class_name_from_code(code: int) -> str:
    for name, value in SUPPORT_CLASS_CODES.items():
        if int(value) == int(code):
            return str(name)
    return 'missing'


def _safe_interp(stations: np.ndarray, src_s: np.ndarray, src_v: np.ndarray) -> np.ndarray:
    out = np.full(stations.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(src_s) & np.isfinite(src_v)
    if np.count_nonzero(valid) == 0:
        return out
    s = np.asarray(src_s[valid], dtype=float)
    v = np.asarray(src_v[valid], dtype=float)
    order = np.argsort(s)
    s = s[order]
    v = v[order]
    uniq_s, inv = np.unique(np.round(s, 6), return_inverse=True)
    agg = np.full(uniq_s.shape, np.nan, dtype=float)
    for i in range(uniq_s.size):
        sel = inv == i
        agg[i] = float(np.nanmedian(v[sel])) if np.any(sel) else np.nan
    valid2 = np.isfinite(agg)
    if np.count_nonzero(valid2) == 0:
        return out
    if np.count_nonzero(valid2) == 1:
        out[:] = np.float32(agg[valid2][0])
        return out
    out[:] = np.interp(stations, uniq_s[valid2], agg[valid2]).astype(np.float32)
    return out


def _smooth_profile_preserve_support(values: np.ndarray, source_codes: np.ndarray | None = None, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    if arr.size < 3:
        return arr
    src = np.asarray(source_codes, dtype=np.float32) if source_codes is not None else np.zeros(arr.shape, dtype=np.float32)
    preserve = np.isfinite(arr) & (src >= SOURCE_CODES['xs_profile_resampled'])
    half = max(int(window) // 2, 1)
    out = arr.copy()
    for i in range(arr.size):
        if preserve[i] or not np.isfinite(arr[i]):
            continue
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        win = arr[lo:hi]
        good = np.isfinite(win)
        if np.count_nonzero(good) >= 2:
            out[i] = np.float32(np.nanmedian(win[good]))
    return out





def _graph_source_code(row: pd.Series) -> int:
    graph_support = str(row.get('graph_solver_support_class', 'unsupported') or 'unsupported')
    graph_mode = str(row.get('graph_solution_mode', 'missing') or 'missing')
    candidate_source = str(row.get('graph_candidate_source', 'missing') or 'missing')
    z_source = str(row.get('z_source', 'missing') or 'missing')
    if graph_support == 'authoritative_locked' or z_source == 'authoritative_in_channel':
        return int(SOURCE_CODES['authoritative_in_channel'])
    if z_source == 'authoritative_bank_margin':
        return int(SOURCE_CODES['authoritative_bank_margin'])
    if graph_support == 'authoritative_backbone' or candidate_source == 'authoritative_backbone':
        return int(SOURCE_CODES['authoritative_backbone'])
    if graph_support == 'stage_controlled' or candidate_source == 'bank_stage_prior':
        return int(SOURCE_CODES['bank_stage_prior'])
    # Strict: only assign xs_profile_resampled for actually measured XS data
    if _is_measured_xs_row(row):
        return int(SOURCE_CODES['xs_profile_resampled'])
    # Nodes with residual/indirect XS influence go to graph_backbone
    if _is_xs_like_row(row):
        return int(SOURCE_CODES['graph_backbone'])
    if graph_mode in ('prior_driven', 'regularization_driven', 'junction_constrained'):
        return int(SOURCE_CODES['graph_backbone'])
    if z_source in SOURCE_CODES:
        return int(SOURCE_CODES[z_source])
    return int(SOURCE_CODES['missing'])


def _graph_confidence(row: pd.Series, src_code: int, eta: float) -> float:
    conf = graph_solution_confidence(row)
    base = float(SOURCE_CONF.get(int(src_code), 0.0))
    conf = 0.65 * conf + 0.35 * base
    lateral_core = float(max(0.0, 1.0 - abs(eta - 0.5) / 0.5))
    conf *= (0.75 + 0.25 * lateral_core)
    return float(np.clip(conf, 0.0, 1.0))


def _is_xs_like_row(row: pd.Series) -> bool:
    """Broad XS classification — any XS-derived provenance (measured or residual).

    Used for source-code attribution in _graph_source_code.  For
    XS *participation* tracking in the channel surface raster, use
    _is_measured_xs_row instead.
    """
    z_source = str(row.get('z_source', 'missing') or 'missing')
    candidate_source = str(row.get('graph_candidate_source', 'missing') or 'missing')
    station_support_mode = str(row.get('station_support_mode', 'missing') or 'missing')
    support_class = str(row.get('graph_solver_support_class', 'missing') or 'missing')
    return bool(
        z_source == 'xs_profile_resampled'
        or candidate_source == 'xs_profile_resampled'
        or station_support_mode in {'xs_only', 'xs_supported', 'xs_residual_only'}
        or support_class == 'xs_residual_only'
    )


def _is_measured_xs_row(row: pd.Series) -> bool:
    """Strict measured XS support, including explicit propagated measured-node flags."""
    return bool((str(row.get('z_source', 'missing') or 'missing') == 'xs_profile_resampled') or bool(row.get('node_true_measured_xs_post_filter', False)))


def _xs_like_mask(nodes: pd.DataFrame) -> np.ndarray:
    """Broad: any XS-derived provenance (used for node counts/attribution)."""
    z_source = nodes.get('z_source', pd.Series(['missing'] * len(nodes), index=nodes.index)).fillna('missing').astype(str)
    candidate_source = nodes.get('graph_candidate_source', pd.Series(['missing'] * len(nodes), index=nodes.index)).fillna('missing').astype(str)
    station_support_mode = nodes.get('station_support_mode', pd.Series(['missing'] * len(nodes), index=nodes.index)).fillna('missing').astype(str)
    support_class = nodes.get('graph_solver_support_class', pd.Series(['missing'] * len(nodes), index=nodes.index)).fillna('missing').astype(str)
    return (
        z_source.eq('xs_profile_resampled')
        | candidate_source.eq('xs_profile_resampled')
        | station_support_mode.isin(['xs_only', 'xs_supported', 'xs_residual_only'])
        | support_class.eq('xs_residual_only')
    ).to_numpy(dtype=bool)


def _measured_xs_mask(nodes: pd.DataFrame) -> np.ndarray:
    """Strict measured XS support, including explicit propagated measured-node flags."""
    z_source = nodes.get('z_source', pd.Series(['missing'] * len(nodes), index=nodes.index)).fillna('missing').astype(str)
    propagated = nodes.get('node_true_measured_xs_post_filter', pd.Series([False] * len(nodes), index=nodes.index)).fillna(False).astype(bool)
    return (z_source.eq('xs_profile_resampled') | propagated).to_numpy(dtype=bool)


def _build_segment_index(thalweg: pd.DataFrame):
    components = {}
    midpoints = []
    meta = []
    for comp, sub in thalweg.groupby('component_id', sort=False):
        sub = sub.sort_values('station_m').reset_index(drop=True)
        components[str(comp)] = sub
        if len(sub) < 2:
            continue
        xs = sub['center_x'].to_numpy(dtype=float)
        ys = sub['center_y'].to_numpy(dtype=float)
        ss = sub['station_m'].to_numpy(dtype=float)
        for i in range(len(sub) - 1):
            x0, y0, s0 = xs[i], ys[i], ss[i]
            x1, y1, s1 = xs[i + 1], ys[i + 1], ss[i + 1]
            if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(s0) and np.isfinite(x1) and np.isfinite(y1) and np.isfinite(s1)):
                continue
            dx = x1 - x0
            dy = y1 - y0
            seg_len2 = dx * dx + dy * dy
            if seg_len2 <= 0.0:
                continue
            midpoints.append(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
            meta.append((str(comp), x0, y0, s0, x1, y1, s1, seg_len2))
    return components, np.asarray(midpoints, dtype=float) if midpoints else np.empty((0, 2), dtype=float), meta


def _nearest_segment_projection(xs: np.ndarray, ys: np.ndarray, *, midpoint_tree, midpoint_coords: np.ndarray, segment_meta, k: int = 8):
    n = xs.shape[0]
    best_comp = np.full(n, '', dtype=object)
    best_station = np.full(n, np.nan, dtype=np.float32)
    best_cx = np.full(n, np.nan, dtype=np.float32)
    best_cy = np.full(n, np.nan, dtype=np.float32)
    best_nx = np.full(n, np.nan, dtype=np.float32)
    best_ny = np.full(n, np.nan, dtype=np.float32)
    best_dist2 = np.full(n, np.inf, dtype=float)
    if midpoint_coords.size == 0 or len(segment_meta) == 0:
        return best_comp, best_station, best_cx, best_cy, best_nx, best_ny
    q = np.column_stack([xs, ys])
    if midpoint_tree is not None:
        kk = min(max(int(k), 1), len(segment_meta))
        _, idxs = midpoint_tree.query(q, k=kk)
        if kk == 1:
            idxs = idxs[:, None]
    else:
        idxs = np.tile(np.arange(len(segment_meta), dtype=int), (n, 1))
    for i in range(n):
        cand = np.atleast_1d(idxs[i]).astype(int)
        x = float(xs[i]); y = float(ys[i])
        for seg_idx in cand:
            comp, x0, y0, s0, x1, y1, s1, seg_len2 = segment_meta[int(seg_idx)]
            dx = x1 - x0
            dy = y1 - y0
            t = ((x - x0) * dx + (y - y0) * dy) / seg_len2
            t = max(0.0, min(1.0, float(t)))
            px = x0 + t * dx
            py = y0 + t * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_dist2[i]:
                seg_len = float(np.sqrt(seg_len2))
                nx = (-dy / seg_len) if seg_len > 0 else np.nan
                ny = (dx / seg_len) if seg_len > 0 else np.nan
                best_dist2[i] = d2
                best_comp[i] = comp
                best_station[i] = np.float32(s0 + t * (s1 - s0))
                best_cx[i] = np.float32(px)
                best_cy[i] = np.float32(py)
                best_nx[i] = np.float32(nx)
                best_ny[i] = np.float32(ny)
    return best_comp, best_station, best_cx, best_cy, best_nx, best_ny


def _base_channel_surface_artifacts(*, river_dir: Path,
    src_path: Path, mode_path: Path, cnt_path: Path, support_class_path: Path, uncertainty_path: Path,
    hard_lock_path: Path, junction_flag_path: Path, junction_role_path: Path, junction_adjustment_path: Path,
    junction_distance_path: Path, unsupported_path: Path, unsupported_regime_path: Path, residual_path: Path,
    xs_participation_path: Path, authoritative_participation_path: Path, influence_class_path: Path,
) -> Dict[str, str]:
    return {
        'channel_surface': str(river_dir / 'river_channel_surface.tif'),
        'channel_surface_confidence': str(river_dir / 'river_channel_surface_confidence.tif'),
        'channel_surface_source_class': str(src_path),
        'channel_surface_graph_mode': str(mode_path),
        'channel_surface_support_count': str(cnt_path),
        'channel_surface_support_class': str(support_class_path),
        'channel_surface_uncertainty': str(uncertainty_path),
        'channel_surface_hard_lock': str(hard_lock_path),
        'channel_surface_junction_constrained': str(junction_flag_path),
        'channel_surface_junction_role': str(junction_role_path),
        'channel_surface_junction_adjustment': str(junction_adjustment_path),
        'channel_surface_junction_distance': str(junction_distance_path),
        'channel_surface_unsupported_span': str(unsupported_path),
        'channel_surface_unsupported_regime': str(unsupported_regime_path),
        'channel_surface_residual_to_candidate': str(residual_path),
        'channel_surface_xs_participation': str(xs_participation_path),
        'channel_surface_authoritative_participation': str(authoritative_participation_path),
        'channel_surface_influence_class': str(influence_class_path),
    }


def _channel_surface_output_aliases(*,
    xs_admissibility_path: Path,
    auth_lock_scope_path: Path,
    auth_lock_applied_path: Path,
    control_nodes_path: Path | None,
    audit_path: Path,
    contract_path: Path,
    support_contract_path: Path,
) -> Dict[str, str]:
    outputs = {
        'channel_surface_xs_admissibility_mask': str(xs_admissibility_path),
        'river_channel_surface_xs_admissibility_mask.tif': str(xs_admissibility_path),
        'channel_surface_authoritative_lock_scope': str(auth_lock_scope_path),
        'river_channel_surface_authoritative_lock_scope.tif': str(auth_lock_scope_path),
        'channel_surface_authoritative_lock_applied': str(auth_lock_applied_path),
        'river_channel_surface_authoritative_lock_applied.tif': str(auth_lock_applied_path),
        'river_xs_propagation_audit': str(audit_path),
        'channel_surface_contract': str(contract_path),
        'support_uncertainty_contract': str(support_contract_path),
    }
    if control_nodes_path is not None:
        outputs['channel_surface_control_nodes'] = str(control_nodes_path)
        outputs['river_channel_surface_control_nodes.gpkg'] = str(control_nodes_path)
    return outputs


def _build_xs_propagation_audit_payload(*,
    river_dir: Path,
    nodes: pd.DataFrame,
    admitted_nodes: pd.DataFrame,
    input_nodes: int,
    input_xs_nodes: int,
    input_measured_xs_nodes: int,
    admitted_xs_nodes: int,
    admitted_measured_xs_nodes: int,
    admitted_auth_nodes: int,
    admitted_graph_nodes: int,
    admitted_bank_nodes: int,
    node_rejection_counts: Dict[str, int],
    admitted_node_role_counts: Dict[str, int],
    populated_mask: np.ndarray,
    xs_participation_out: np.ndarray,
    authoritative_participation_out: np.ndarray,
    influence_class_out: np.ndarray,
    non_authoritative_support_mask: np.ndarray,
    xs_admissibility_mask: np.ndarray,
    authoritative_lock_scope_out: np.ndarray,
    authoritative_lock_applied_out: np.ndarray,
    auth_mask: np.ndarray | None,
    corridor: np.ndarray,
    xs_warning: bool,
    artifacts: Dict[str, str],
) -> Dict[str, Any]:
    xs_expected_mask = nodes['surface_control_xs_expected'].to_numpy(dtype=np.int16) > 0
    admitted_mask = nodes['surface_control_admitted'].to_numpy(dtype=np.int16) > 0
    xs_expected_admitted_mask = nodes['surface_control_xs_expected_admitted'].to_numpy(dtype=np.int16) > 0
    xs_expected_gap = _bool_count(populated_mask & xs_admissibility_mask & (xs_participation_out <= 0))
    return {
        'schema_version': 1,
        'artifact_family': 'river_xs_propagation_audit',
        'notes': {
            'objective': 'Trace XS contribution from scaffold nodes to admitted surface controls to final populated channel-surface cells.',
            'truth_rule': 'XS propagation is only considered active when xs_profile_resampled scaffold nodes survive into admitted controls and participate in at least one populated channel-surface cell.',
        },
        'metrics': {
            'scaffold_input_node_count': int(input_nodes),
            'scaffold_input_xs_node_count': int(input_xs_nodes),
            'scaffold_input_measured_xs_node_count': int(input_measured_xs_nodes),
            'scaffold_input_xs_nodes_in_support_expected_zone': int(np.count_nonzero(xs_expected_mask)),
            'admitted_surface_control_node_count': int(len(admitted_nodes)),
            'admitted_surface_control_xs_node_count': int(admitted_xs_nodes),
            'admitted_surface_control_measured_xs_node_count': int(admitted_measured_xs_nodes),
            'admitted_surface_control_xs_nodes_in_support_expected_zone': int(np.count_nonzero(admitted_mask & xs_expected_mask)),
            'admitted_surface_control_authoritative_node_count': int(admitted_auth_nodes),
            'admitted_surface_control_graph_node_count': int(admitted_graph_nodes),
            'admitted_surface_control_bank_node_count': int(admitted_bank_nodes),
            'node_rejection_counts': node_rejection_counts,
            'admitted_node_role_counts': admitted_node_role_counts,
            'final_populated_cell_count': _bool_count(populated_mask),
            'final_xs_participation_cell_count': _bool_count(xs_participation_out > 0),
            'final_authoritative_participation_cell_count': _bool_count(authoritative_participation_out > 0),
            'final_mixed_authoritative_xs_cell_count': _bool_count((xs_participation_out > 0) & (authoritative_participation_out > 0)),
            'final_authoritative_only_cell_count': _bool_count(influence_class_out == INFLUENCE_CLASS_CODES['authoritative_only']),
            'final_xs_only_cell_count': _bool_count(influence_class_out == INFLUENCE_CLASS_CODES['xs_only']),
            'final_other_scaffold_only_cell_count': _bool_count(influence_class_out == INFLUENCE_CLASS_CODES['other_scaffold_only']),
            'final_non_authoritative_support_cell_count': _bool_count(populated_mask & non_authoritative_support_mask),
            'final_xs_admissibility_cell_count': _bool_count(populated_mask & xs_admissibility_mask),
            'final_xs_participation_in_non_authoritative_support_cells': _bool_count((xs_participation_out > 0) & populated_mask & non_authoritative_support_mask),
            'final_xs_expected_but_missing_cell_count': int(xs_expected_gap),
            'authoritative_lock_scope_cell_count': _bool_count(authoritative_lock_scope_out > 0),
            'authoritative_lock_applied_cell_count': _bool_count(authoritative_lock_applied_out > 0),
            'authoritative_support_cells_outside_lock_scope': _bool_count((auth_mask > 0) & corridor & (authoritative_lock_scope_out <= 0)) if auth_mask is not None else 0,
            'authoritative_lock_applied_outside_scope_cell_count': _bool_count((authoritative_lock_applied_out > 0) & (authoritative_lock_scope_out <= 0)),
            'xs_expected_surface_control_node_count': int(np.count_nonzero(xs_expected_mask)),
            'xs_expected_admitted_surface_control_node_count': int(np.count_nonzero(xs_expected_admitted_mask)),
            'xs_propagation_warning': bool(xs_warning),
        },
        'influence_class_codes': {k: int(v) for k, v in INFLUENCE_CLASS_CODES.items()},
        'artifacts': dict(artifacts),
    }

def _grid_xy(transform, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = transform.c + (cols.astype(float) + 0.5) * transform.a + (rows.astype(float) + 0.5) * transform.b
    y = transform.f + (cols.astype(float) + 0.5) * transform.d + (rows.astype(float) + 0.5) * transform.e
    return x, y


def _read_optional_aligned_raster(
    *,
    path: str | Path | None,
    expected_shape: tuple[int, int],
    logger: logging.Logger,
    label: str,
    cast_float32: bool = False,
) -> Optional[np.ndarray]:
    if not path:
        return None
    import rasterio
    from rasterio.errors import RasterioIOError

    try:
        with rasterio.open(path) as ds:
            arr = ds.read(1)
    except (RasterioIOError, OSError, ValueError):
        logger.debug('build_channel_surface_products: failed reading %s', label, exc_info=True)
        return None
    if cast_float32:
        arr = arr.astype(np.float32, copy=False)
    if arr.shape != expected_shape:
        logger.debug(
            'build_channel_surface_products: ignoring %s with mismatched shape %s != %s',
            label,
            arr.shape,
            expected_shape,
        )
        return None
    return arr


def _safe_harmonize_nodes_crs(nodes, template_crs, logger: logging.Logger):
    if getattr(nodes, 'crs', None) is None or template_crs is None or str(nodes.crs) == str(template_crs):
        return nodes
    try:
        return nodes.to_crs(template_crs)
    except (ValueError, AttributeError, TypeError):
        logger.debug('build_channel_surface_products: failed CRS harmonization', exc_info=True)
        return nodes


def _try_build_ckdtree(points: np.ndarray):
    if points.size == 0:
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    return cKDTree(points)


def _write_control_nodes_export(nodes, path: Path, logger: logging.Logger) -> Optional[str]:
    try:
        nodes.to_file(path, driver='GPKG')
    except (OSError, ValueError, RuntimeError):
        logger.warning('[RIVER][FRAME] Failed to write channel surface control-node export: %s', path, exc_info=True)
        return None
    return str(path)


def build_channel_surface_products(
    *,
    river_dir: str | Path,
    channel_scaffold_nodes_path: str | Path | None,
    corridor_mask_path: str | Path | None,
    authoritative_support_mask_path: str | Path | None = None,
    authoritative_support_depth_path: str | Path | None = None,
    authoritative_role_code_path: str | Path | None = None,
    longitudinal_profile_path: str | Path | None = None,
    reach_attributes_path: str | Path | None = None,
    disable_xs_influence: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    import geopandas as gpd
    import rasterio

    river_dir = Path(river_dir)
    scaffold_path = Path(channel_scaffold_nodes_path) if channel_scaffold_nodes_path else None
    corridor_path = Path(corridor_mask_path) if corridor_mask_path else None
    if scaffold_path is None or corridor_path is None or not scaffold_path.exists() or not corridor_path.exists():
        return {}

    nodes = gpd.read_file(scaffold_path)
    if nodes is None or nodes.empty:
        return {}
    if 'station_m' not in nodes.columns or 'node_role' not in nodes.columns:
        return {}
    nodes['station_m'] = pd.to_numeric(nodes['station_m'], errors='coerce')
    nodes['bed_z_m'] = pd.to_numeric(nodes.get('bed_z_m', np.nan), errors='coerce')
    nodes['half_width_m'] = pd.to_numeric(nodes.get('half_width_m', np.nan), errors='coerce')
    nodes['center_x'] = pd.to_numeric(nodes.get('center_x', np.nan), errors='coerce')
    nodes['center_y'] = pd.to_numeric(nodes.get('center_y', np.nan), errors='coerce')
    nodes['normal_x'] = pd.to_numeric(nodes.get('normal_x', np.nan), errors='coerce')
    nodes['normal_y'] = pd.to_numeric(nodes.get('normal_y', np.nan), errors='coerce')
    nodes['component_id'] = nodes.get('component_id', 'main').fillna('main').astype(str)
    nodes['z_source'] = nodes.get('z_source', 'missing').fillna('missing').astype(str)
    nodes['backbone_bed_z_m'] = pd.to_numeric(nodes.get('backbone_bed_z_m', np.nan), errors='coerce')
    nodes['residual_shape_z_m'] = pd.to_numeric(nodes.get('residual_shape_z_m', np.nan), errors='coerce')
    if 'station_support_mode' in nodes.columns:
        nodes['station_support_mode'] = nodes['station_support_mode'].fillna('missing').astype(str)
    else:
        nodes['station_support_mode'] = pd.Series(['missing'] * len(nodes), index=nodes.index, dtype='object')
    nodes['graph_backbone_z_m'] = pd.to_numeric(nodes.get('graph_backbone_z_m', np.nan), errors='coerce')
    nodes['graph_prior_weight_sum'] = pd.to_numeric(nodes.get('graph_prior_weight_sum', np.nan), errors='coerce')
    nodes['graph_regularization_weight_sum'] = pd.to_numeric(nodes.get('graph_regularization_weight_sum', np.nan), errors='coerce')
    nodes['graph_junction_weight_sum'] = pd.to_numeric(nodes.get('graph_junction_weight_sum', np.nan), errors='coerce')
    nodes['graph_residual_to_candidate_z_m'] = pd.to_numeric(nodes.get('graph_residual_to_candidate_z_m', np.nan), errors='coerce')
    nodes['graph_unsupported_span_m'] = pd.to_numeric(nodes.get('graph_unsupported_span_m', np.nan), errors='coerce')
    if 'graph_hard_lock' in nodes.columns:
        nodes['graph_hard_lock'] = nodes['graph_hard_lock'].fillna(False).astype(bool)
    else:
        nodes['graph_hard_lock'] = False
    if 'graph_junction_constrained' in nodes.columns:
        nodes['graph_junction_constrained'] = nodes['graph_junction_constrained'].fillna(False).astype(bool)
    else:
        nodes['graph_junction_constrained'] = False
    if 'graph_junction_role' in nodes.columns:
        nodes['graph_junction_role'] = nodes['graph_junction_role'].fillna('not_in_junction').astype(str)
    else:
        nodes['graph_junction_role'] = pd.Series(['not_in_junction'] * len(nodes), index=nodes.index, dtype='object')
    nodes['graph_junction_adjustment_z_m'] = pd.to_numeric(nodes.get('graph_junction_adjustment_z_m', np.nan), errors='coerce')
    nodes['graph_junction_distance_m'] = pd.to_numeric(nodes.get('graph_junction_distance_m', np.nan), errors='coerce')
    nodes['graph_junction_influence_weight'] = pd.to_numeric(nodes.get('graph_junction_influence_weight', np.nan), errors='coerce')
    if 'graph_solver_support_class' in nodes.columns:
        nodes['graph_solver_support_class'] = nodes['graph_solver_support_class'].fillna('unsupported').astype(str)
    else:
        nodes['graph_solver_support_class'] = pd.Series(['unsupported'] * len(nodes), index=nodes.index, dtype='object')
    if 'graph_candidate_source' in nodes.columns:
        nodes['graph_candidate_source'] = nodes['graph_candidate_source'].fillna('missing').astype(str)
    else:
        nodes['graph_candidate_source'] = pd.Series(['missing'] * len(nodes), index=nodes.index, dtype='object')
    if 'graph_solution_mode' in nodes.columns:
        nodes['graph_solution_mode'] = nodes['graph_solution_mode'].fillna('missing').astype(str)
    else:
        nodes['graph_solution_mode'] = pd.Series(['missing'] * len(nodes), index=nodes.index, dtype='object')
    if 'graph_unsupported_regime' in nodes.columns:
        nodes['graph_unsupported_regime'] = nodes['graph_unsupported_regime'].fillna('missing').astype(str)
    else:
        nodes['graph_unsupported_regime'] = pd.Series(['missing'] * len(nodes), index=nodes.index, dtype='object')
    nodes = nodes.loc[np.isfinite(nodes['station_m'])].copy()
    if nodes.empty:
        return {}

    with rasterio.open(corridor_path) as ds:
        corridor = ds.read(1) > 0
        profile = ds.profile.copy()
        transform = ds.transform
        template_crs = ds.crs
    if not np.any(corridor):
        return {}

    active_logger = logger or log
    auth_mask = _read_optional_aligned_raster(
        path=authoritative_support_mask_path,
        expected_shape=corridor.shape,
        logger=active_logger,
        label='authoritative support mask',
        cast_float32=False,
    )
    auth_depth = _read_optional_aligned_raster(
        path=authoritative_support_depth_path,
        expected_shape=corridor.shape,
        logger=active_logger,
        label='authoritative support depth',
        cast_float32=True,
    )
    auth_role_code = _read_optional_aligned_raster(
        path=authoritative_role_code_path,
        expected_shape=corridor.shape,
        logger=active_logger,
        label='authoritative role code',
        cast_float32=False,
    )
    authoritative_transition_pixel_distance_m = _compute_authoritative_transition_pixel_distance_m(
        auth_mask=auth_mask,
        corridor=corridor,
        transform=transform,
        template_crs=template_crs,
    )

    nodes = _safe_harmonize_nodes_crs(nodes, template_crs, active_logger)
    if disable_xs_influence:
        for _col in ('target_xs_realism_allowed', 'target_xs_residual_allowed'):
            nodes[_col] = False
    nodes, tendency_outputs, tendency_summary = apply_longitudinal_tendency_to_nodes(
        nodes,
        river_dir=river_dir,
        reach_attributes_path=reach_attributes_path,
        longitudinal_profile_path=longitudinal_profile_path,
        logger=active_logger,
    )
    nodes, xs_realism_outputs, xs_realism_summary = apply_xs_realism_to_nodes(
        nodes,
        river_dir=river_dir,
        reach_attributes_path=reach_attributes_path,
        disabled=disable_xs_influence,
        logger=active_logger,
    )
    nodes, prediction_confidence_outputs, prediction_confidence_summary = apply_prediction_confidence_to_nodes(
        nodes,
        river_dir=river_dir,
        reach_attributes_path=reach_attributes_path,
        logger=active_logger,
    )
    nodes, primary_surface_rebuild_outputs, primary_surface_rebuild_summary = apply_primary_surface_rebuild_to_nodes(
        nodes,
        river_dir=river_dir,
        disabled=False,
        logger=active_logger,
    )
    required_canonical_cols = ["support_class_canonical", "active_interior_target_source", "active_interior_target_z_m"]
    missing_canonical_cols = [col for col in required_canonical_cols if col not in nodes.columns]
    if missing_canonical_cols:
        raise ValueError(f"channel surface build requires canonical v314 columns: {missing_canonical_cols}")
    effectiveness_outputs, effectiveness_summary = write_effectiveness_receipts(
        nodes,
        river_dir=river_dir,
        xs_realism_summary=xs_realism_summary,
        primary_surface_rebuild_summary=primary_surface_rebuild_summary,
        logger=active_logger,
    )

    thalweg = nodes.loc[nodes['node_role'].astype(str).eq('thalweg')].copy()
    if thalweg.empty:
        return {}
    thalweg = thalweg.loc[np.isfinite(thalweg['center_x']) & np.isfinite(thalweg['center_y'])].copy()
    if thalweg.empty:
        return {}

    component_thalwegs, segment_midpoints, segment_meta = _build_segment_index(thalweg)
    tree = _try_build_ckdtree(segment_midpoints)

    role_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    source_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_mode_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_support_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_uncertainty_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_prior_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_reg_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_junction_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_junction_role_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_junction_adjustment_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_junction_distance_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_hard_lock_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_junction_flag_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_residual_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_unsupported_span_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_regime_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    graph_confidence_surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    station_prediction_confidence_surfaces: Dict[str, np.ndarray] = {}
    station_measured_anchor_fraction_surfaces: Dict[str, np.ndarray] = {}
    station_structure_only_fraction_surfaces: Dict[str, np.ndarray] = {}
    station_low_support_caution_surfaces: Dict[str, np.ndarray] = {}
    station_prediction_admissibility_surfaces: Dict[str, np.ndarray] = {}
    station_target_left_bank_surfaces: Dict[str, np.ndarray] = {}
    station_target_right_bank_surfaces: Dict[str, np.ndarray] = {}
    station_target_thalweg_surfaces: Dict[str, np.ndarray] = {}
    station_target_present_surfaces: Dict[str, np.ndarray] = {}
    station_target_local_authoritative_reconciled_surfaces: Dict[str, np.ndarray] = {}
    station_authoritative_reconciliation_weight_surfaces: Dict[str, np.ndarray] = {}
    station_authoritative_bed_support_distance_surfaces: Dict[str, np.ndarray] = {}
    width_surfaces: Dict[str, np.ndarray] = {}
    component_render_modes: Dict[str, str] = {}
    component_render_mode_reasons: Dict[str, str] = {}
    component_active_roles: Dict[str, list[tuple[str, float]]] = {}
    component_render_selector_rows: list[dict[str, Any]] = []
    component_render_selector_by_id: Dict[str, dict[str, Any]] = {}
    for comp, sub in nodes.groupby('component_id', sort=False):
        comp_key = str(comp)
        render_mode, active_roles, render_reason, selector_row = _select_component_render_mode(sub)
        component_render_selector_rows.append(dict(selector_row))
        component_render_selector_by_id[comp_key] = dict(selector_row)
        component_render_modes[comp_key] = render_mode
        component_render_mode_reasons[comp_key] = render_reason
        component_active_roles[comp_key] = list(active_roles)
        role_surfaces[comp_key] = {}
        source_surfaces[str(comp)] = {}
        graph_mode_surfaces[str(comp)] = {}
        graph_support_surfaces[str(comp)] = {}
        graph_uncertainty_surfaces[str(comp)] = {}
        graph_prior_surfaces[str(comp)] = {}
        graph_reg_surfaces[str(comp)] = {}
        graph_junction_surfaces[str(comp)] = {}
        graph_junction_role_surfaces[str(comp)] = {}
        graph_junction_adjustment_surfaces[str(comp)] = {}
        graph_junction_distance_surfaces[str(comp)] = {}
        graph_hard_lock_surfaces[str(comp)] = {}
        graph_junction_flag_surfaces[str(comp)] = {}
        graph_residual_surfaces[str(comp)] = {}
        graph_unsupported_span_surfaces[str(comp)] = {}
        graph_regime_surfaces[str(comp)] = {}
        graph_confidence_surfaces[str(comp)] = {}
        th = thalweg.loc[thalweg['component_id'].astype(str).eq(str(comp))].sort_values('station_m').copy()
        s_grid = th['station_m'].to_numpy(dtype=float)
        width_surfaces[str(comp)] = _safe_interp(s_grid, sub['station_m'].to_numpy(dtype=float), (2.0 * sub['half_width_m'].to_numpy(dtype=float)))
        def _th_numeric_series(name: str, default: float = np.nan) -> pd.Series:
            if name in th.columns:
                return pd.to_numeric(th[name], errors='coerce')
            return pd.Series([default] * len(th), index=th.index, dtype='float64')
        if 'prediction_support_confidence' in th.columns:
            station_prediction_confidence_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), _th_numeric_series('prediction_support_confidence').to_numpy(dtype=float))
            station_measured_anchor_fraction_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), _th_numeric_series('prediction_measured_anchor_fraction').to_numpy(dtype=float))
            station_structure_only_fraction_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), _th_numeric_series('prediction_structure_only_fraction').to_numpy(dtype=float))
            station_low_support_caution_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), _th_numeric_series('prediction_low_support_caution', 0.0).astype(float).to_numpy(dtype=float))
            station_prediction_admissibility_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), _th_numeric_series('prediction_admissible', 0.0).astype(float).to_numpy(dtype=float))
        else:
            nan_station_prof = np.full(s_grid.shape, np.nan, dtype=np.float32)
            station_prediction_confidence_surfaces[str(comp)] = nan_station_prof.copy()
            station_measured_anchor_fraction_surfaces[str(comp)] = nan_station_prof.copy()
            station_structure_only_fraction_surfaces[str(comp)] = nan_station_prof.copy()
            station_low_support_caution_surfaces[str(comp)] = nan_station_prof.copy()
            station_prediction_admissibility_surfaces[str(comp)] = nan_station_prof.copy()
        station_target_left_bank_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), _th_numeric_series('target_left_bank_z_m').to_numpy(dtype=float))
        station_target_right_bank_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), _th_numeric_series('target_right_bank_z_m').to_numpy(dtype=float))
        station_target_thalweg_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), _th_numeric_series('target_thalweg_z_m').to_numpy(dtype=float))
        station_target_present_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), _th_numeric_series('station_target_present', 0.0).astype(float).to_numpy(dtype=float))
        station_target_local_authoritative_reconciled_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), _th_numeric_series('station_target_local_authoritative_reconciled', 0.0).astype(float).to_numpy(dtype=float))
        station_authoritative_reconciliation_weight_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), _th_numeric_series('authoritative_reconciliation_weight').to_numpy(dtype=float))
        resolved_support_distance = _resolve_station_support_distance_series(th)
        station_authoritative_bed_support_distance_surfaces[str(comp)] = _safe_interp(s_grid, th['station_m'].to_numpy(dtype=float), resolved_support_distance)
        for role, role_eta in ROLE_ORDER:
            rs = sub.loc[sub['node_role'].astype(str).eq(role)].copy()
            if rs.empty:
                nan_prof = np.full(s_grid.shape, np.nan, dtype=np.float32)
                role_surfaces[str(comp)][role] = nan_prof.copy()
                source_surfaces[str(comp)][role] = nan_prof.copy()
                graph_mode_surfaces[str(comp)][role] = nan_prof.copy()
                graph_support_surfaces[str(comp)][role] = nan_prof.copy()
                graph_uncertainty_surfaces[str(comp)][role] = nan_prof.copy()
                graph_prior_surfaces[str(comp)][role] = nan_prof.copy()
                graph_reg_surfaces[str(comp)][role] = nan_prof.copy()
                graph_junction_surfaces[str(comp)][role] = nan_prof.copy()
                graph_junction_role_surfaces[str(comp)][role] = nan_prof.copy()
                graph_junction_adjustment_surfaces[str(comp)][role] = nan_prof.copy()
                graph_junction_distance_surfaces[str(comp)][role] = nan_prof.copy()
                graph_hard_lock_surfaces[str(comp)][role] = nan_prof.copy()
                graph_junction_flag_surfaces[str(comp)][role] = nan_prof.copy()
                graph_residual_surfaces[str(comp)][role] = nan_prof.copy()
                graph_unsupported_span_surfaces[str(comp)][role] = nan_prof.copy()
                graph_regime_surfaces[str(comp)][role] = nan_prof.copy()
                graph_confidence_surfaces[str(comp)][role] = nan_prof.copy()
                continue
            z_prof = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), rs['bed_z_m'].to_numpy(dtype=float))
            rs['graph_surface_source_code'] = rs.apply(_graph_source_code, axis=1)
            src_prof = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), rs['graph_surface_source_code'].to_numpy(dtype=float))
            graph_mode_codes = rs['graph_solution_mode'].map(lambda v: GRAPH_MODE_CODES.get(str(v), 0)).to_numpy(dtype=float)
            support_codes = rs['graph_solver_support_class'].map(lambda v: SUPPORT_CLASS_CODES.get(str(v), 0)).to_numpy(dtype=float)
            if 'graph_uncertainty_class' in rs.columns:
                uncertainty_codes = rs['graph_uncertainty_class'].map(lambda v: UNCERTAINTY_CLASS_CODES.get(str(v), 0)).to_numpy(dtype=float)
            else:
                uncertainty_codes = rs.apply(lambda row: float(UNCERTAINTY_CLASS_CODES.get(uncertainty_class_from_confidence(graph_solution_confidence(row)), 0)), axis=1).to_numpy(dtype=float)
            confidence_codes = rs.apply(lambda row: _graph_confidence(row, int(row.get('graph_surface_source_code', 0) or 0), role_eta), axis=1).to_numpy(dtype=float)
            role_surfaces[str(comp)][role] = _smooth_profile_preserve_support(z_prof, src_prof, window=5)
            source_surfaces[str(comp)][role] = src_prof
            graph_mode_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), graph_mode_codes)
            graph_support_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), support_codes)
            graph_uncertainty_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), uncertainty_codes)
            graph_prior_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), rs['graph_prior_weight_sum'].to_numpy(dtype=float))
            graph_reg_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), rs['graph_regularization_weight_sum'].to_numpy(dtype=float))
            graph_junction_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), rs['graph_junction_weight_sum'].to_numpy(dtype=float))
            graph_junction_role_codes = rs['graph_junction_role'].map(lambda v: JUNCTION_ROLE_CODES.get(str(v), 0)).to_numpy(dtype=float)
            graph_junction_role_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), graph_junction_role_codes)
            graph_junction_adjustment_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), rs['graph_junction_adjustment_z_m'].to_numpy(dtype=float))
            graph_junction_distance_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), rs['graph_junction_distance_m'].to_numpy(dtype=float))
            graph_hard_lock_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), rs['graph_hard_lock'].astype(float).to_numpy(dtype=float))
            graph_junction_flag_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), rs['graph_junction_constrained'].astype(float).to_numpy(dtype=float))
            graph_residual_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), rs['graph_residual_to_candidate_z_m'].to_numpy(dtype=float))
            graph_unsupported_span_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), rs['graph_unsupported_span_m'].to_numpy(dtype=float))
            graph_regime_codes = rs['graph_unsupported_regime'].map(lambda v: UNSUPPORTED_REGIME_CODES.get(str(v), 0)).to_numpy(dtype=float)
            graph_regime_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), graph_regime_codes)
            graph_confidence_surfaces[str(comp)][role] = _safe_interp(s_grid, rs['station_m'].to_numpy(dtype=float), confidence_codes)

    role_set = {role for role, _ in ROLE_ORDER}
    input_nodes = int(len(nodes))
    xs_like_mask = _xs_like_mask(nodes)
    measured_xs_like_mask = _measured_xs_mask(nodes)
    input_xs_nodes = int(np.count_nonzero(xs_like_mask))
    input_measured_xs_nodes = int(np.count_nonzero(measured_xs_like_mask))
    missing_bed_mask = ~np.isfinite(nodes['bed_z_m'].to_numpy(dtype=float))
    admitted_mask = np.isfinite(nodes['bed_z_m'].to_numpy(dtype=float)) & nodes['node_role'].astype(str).isin(role_set).to_numpy(dtype=bool)
    admitted_nodes = nodes.loc[admitted_mask].copy()
    admitted_xs_nodes = int(np.count_nonzero(_xs_like_mask(admitted_nodes))) if not admitted_nodes.empty else 0
    admitted_measured_xs_nodes = int(np.count_nonzero(_measured_xs_mask(admitted_nodes))) if not admitted_nodes.empty else 0
    admitted_auth_nodes = int(np.count_nonzero((admitted_nodes['z_source'].astype(str).isin(['authoritative_in_channel'])) | admitted_nodes['graph_hard_lock'].fillna(False).astype(bool))) if not admitted_nodes.empty else 0
    admitted_graph_nodes = int(np.count_nonzero(admitted_nodes['z_source'].astype(str).map(is_surface_guidance_source).to_numpy(dtype=bool) & (~_xs_like_mask(admitted_nodes)))) if not admitted_nodes.empty else 0
    admitted_bank_nodes = int(np.count_nonzero(admitted_nodes['z_source'].astype(str).isin(['bank_stage_prior', 'authoritative_bank_margin']))) if not admitted_nodes.empty else 0
    admitted_node_role_counts = {str(k): int(v) for k, v in admitted_nodes['node_role'].astype(str).value_counts().to_dict().items()} if not admitted_nodes.empty else {}
    node_rejection_counts = {
        'missing_bed_z': int(np.count_nonzero(missing_bed_mask)),
        'inactive_role': int(np.count_nonzero(~nodes['node_role'].astype(str).isin(role_set).to_numpy(dtype=bool))),
    }

    support_class_col = nodes['graph_solver_support_class'].astype(str) if 'graph_solver_support_class' in nodes.columns else pd.Series(['missing'] * len(nodes), index=nodes.index, dtype='object')
    support_expected_classes = {'anchored_interpolated', 'stage_controlled', 'graph_backbone', 'xs_residual_only', 'unsupported'}
    authoritative_classes = {'authoritative_locked'}
    support_expected_codes = {int(SUPPORT_CLASS_CODES[name]) for name in support_expected_classes if name in SUPPORT_CLASS_CODES}
    authoritative_codes = {int(SUPPORT_CLASS_CODES[name]) for name in authoritative_classes if name in SUPPORT_CLASS_CODES}
    role_mask = nodes['node_role'].astype(str).isin(role_set).to_numpy(dtype=bool)
    support_expected_mask = support_class_col.astype(str).isin(support_expected_classes).to_numpy(dtype=bool)
    xs_candidate_mask = xs_like_mask
    nodes['surface_control_admitted'] = admitted_mask.astype(np.int16)
    nodes['surface_control_admission_reason'] = np.where(
        missing_bed_mask,
        'missing_bed_z',
        np.where(role_mask, 'admitted', 'inactive_role'),
    )
    nodes['surface_control_source_role'] = nodes['z_source'].astype(str)
    nodes['surface_control_support_class'] = support_class_col.astype(str)
    nodes['surface_control_support_expected_zone'] = support_expected_mask.astype(np.int16)
    nodes['surface_control_authoritative_locked'] = ((support_class_col.astype(str).isin(authoritative_classes)) | nodes['z_source'].astype(str).isin(['authoritative_in_channel'])).astype(np.int16)
    nodes['surface_control_xs_candidate'] = xs_candidate_mask.astype(np.int16)
    nodes['surface_control_xs_expected'] = (xs_candidate_mask & support_expected_mask).astype(np.int16)
    nodes['surface_control_xs_expected_admitted'] = (xs_candidate_mask & support_expected_mask & admitted_mask).astype(np.int16)

    rows, cols = np.where(corridor)
    xs, ys = _grid_xy(transform, rows, cols)
    z_out = np.full(corridor.shape, np.nan, dtype=np.float32)
    conf_out = np.full(corridor.shape, np.nan, dtype=np.float32)
    src_out = np.zeros(corridor.shape, dtype=np.int16)
    mode_out = np.zeros(corridor.shape, dtype=np.int16)
    support_out = np.zeros(corridor.shape, dtype=np.int16)
    support_class_out = np.zeros(corridor.shape, dtype=np.int16)
    uncertainty_out = np.zeros(corridor.shape, dtype=np.int16)
    hard_lock_out = np.zeros(corridor.shape, dtype=np.int16)
    junction_flag_out = np.zeros(corridor.shape, dtype=np.int16)
    junction_role_out = np.zeros(corridor.shape, dtype=np.int16)
    junction_adjustment_out = np.full(corridor.shape, np.nan, dtype=np.float32)
    junction_distance_out = np.full(corridor.shape, np.nan, dtype=np.float32)
    unsupported_span_out = np.full(corridor.shape, np.nan, dtype=np.float32)
    unsupported_regime_out = np.zeros(corridor.shape, dtype=np.int16)
    residual_to_candidate_out = np.full(corridor.shape, np.nan, dtype=np.float32)
    xs_participation_out = np.zeros(corridor.shape, dtype=np.int16)
    authoritative_participation_out = np.zeros(corridor.shape, dtype=np.int16)
    influence_class_out = np.zeros(corridor.shape, dtype=np.int16)
    authoritative_lock_scope_out = np.zeros(corridor.shape, dtype=np.int16)
    authoritative_lock_applied_out = np.zeros(corridor.shape, dtype=np.int16)
    prediction_support_confidence_out = np.full(corridor.shape, np.float32(np.nan), dtype=np.float32)
    prediction_measured_anchor_fraction_out = np.full(corridor.shape, np.float32(np.nan), dtype=np.float32)
    prediction_structure_only_fraction_out = np.full(corridor.shape, np.float32(np.nan), dtype=np.float32)
    prediction_low_support_caution_out = np.zeros(corridor.shape, dtype=np.int16)
    prediction_admissibility_out = np.zeros(corridor.shape, dtype=np.int16)
    authoritative_transition_weight_out = np.full(corridor.shape, np.float32(np.nan), dtype=np.float32)
    component_effect_rows: list[dict[str, Any]] = []
    component_longitudinal_smoothing_rows: list[dict[str, Any]] = []
    component_longitudinal_smoothing_by_id: Dict[str, Dict[str, Any]] = {}

    # Precompute per-component thalweg station arrays for interpolation lookup.
    thalweg_by_comp = {
        str(comp): sub.sort_values('station_m').reset_index(drop=True)
        for comp, sub in thalweg.groupby('component_id', sort=False)
    }

    proj_comp, proj_station, proj_cx, proj_cy, proj_nx, proj_ny = _nearest_segment_projection(
        xs, ys, midpoint_tree=tree, midpoint_coords=segment_midpoints, segment_meta=segment_meta
    )

    def _nearest_value(eta_value: float, eta_axis: np.ndarray, valid_mask: np.ndarray, arr: np.ndarray, default: float = 0.0) -> float:
        if np.count_nonzero(valid_mask) < 1:
            return float(default)
        idx = int(np.argmin(np.abs(eta_axis[valid_mask] - eta_value)))
        return float(arr[valid_mask][idx])

    unique_components = [str(c) for c in pd.unique(proj_comp) if str(c)]
    for comp in unique_components:
        comp_sel = np.asarray([str(c) == comp for c in proj_comp], dtype=bool)
        if not np.any(comp_sel):
            continue
        th = thalweg_by_comp.get(comp)
        if th is None or th.empty:
            continue
        valid_geom = comp_sel & np.isfinite(proj_cx) & np.isfinite(proj_cy) & np.isfinite(proj_nx) & np.isfinite(proj_ny) & np.isfinite(proj_station)
        if not np.any(valid_geom):
            continue
        idxs = np.where(valid_geom)[0]
        stations = th['station_m'].to_numpy(dtype=float)
        station_arr = proj_station[idxs].astype(float)
        width_prof = width_surfaces.get(comp)
        if width_prof is not None and np.any(np.isfinite(width_prof)):
            width_arr = np.interp(station_arr, stations, width_prof)
        else:
            width_arr = np.full(station_arr.shape, float(np.nanmedian(2.0 * th['half_width_m'].to_numpy(dtype=float))), dtype=float)
        width_arr = np.maximum(width_arr, 1.0)
        pc_prof = station_prediction_confidence_surfaces.get(comp)
        ma_prof = station_measured_anchor_fraction_surfaces.get(comp)
        so_prof = station_structure_only_fraction_surfaces.get(comp)
        lc_prof = station_low_support_caution_surfaces.get(comp)
        pa_prof = station_prediction_admissibility_surfaces.get(comp)
        target_left_bank_prof = station_target_left_bank_surfaces.get(comp)
        target_right_bank_prof = station_target_right_bank_surfaces.get(comp)
        target_thalweg_prof = station_target_thalweg_surfaces.get(comp)
        target_present_prof = station_target_present_surfaces.get(comp)
        target_local_reconciled_prof = station_target_local_authoritative_reconciled_surfaces.get(comp)
        target_reconciliation_weight_prof = station_authoritative_reconciliation_weight_surfaces.get(comp)
        target_bed_support_distance_prof = station_authoritative_bed_support_distance_surfaces.get(comp)
        signed_arr = ((xs[idxs] - proj_cx[idxs]) * proj_nx[idxs] + (ys[idxs] - proj_cy[idxs]) * proj_ny[idxs])
        eta_arr = np.clip(0.5 + signed_arr / width_arr, 0.0, 1.0)

        active_roles = component_active_roles.get(comp, ROLE_ORDER)
        eta_coords = np.asarray([e for _, e in active_roles], dtype=float)
        n_roles = len(active_roles)
        n_pix = idxs.size

        th_support_classes = th['graph_solver_support_class'].astype(str).to_numpy() if 'graph_solver_support_class' in th.columns else np.full(len(th), 'unsupported', dtype=object)
        nearest_idx = np.searchsorted(stations, station_arr)
        nearest_idx = np.clip(nearest_idx, 0, max(len(stations) - 1, 0))
        prev_idx = np.clip(nearest_idx - 1, 0, max(len(stations) - 1, 0))
        choose_prev = np.abs(station_arr - stations[prev_idx]) <= np.abs(station_arr - stations[nearest_idx])
        nearest_idx = np.where(choose_prev, prev_idx, nearest_idx)
        nearest_support_class = th_support_classes[nearest_idx] if len(th_support_classes) else np.full(n_pix, 'unsupported', dtype=object)

        z_bins_arr = np.full((n_roles, n_pix), np.nan, dtype=float)
        src_bins_arr = np.full((n_roles, n_pix), 0.0, dtype=float)
        mode_bins_arr = np.full((n_roles, n_pix), 0.0, dtype=float)
        support_bins_arr = np.full((n_roles, n_pix), 0.0, dtype=float)
        uncertainty_bins_arr = np.full((n_roles, n_pix), 0.0, dtype=float)
        hard_lock_bins_arr = np.full((n_roles, n_pix), 0.0, dtype=float)
        junction_bins_arr = np.full((n_roles, n_pix), 0.0, dtype=float)
        junction_role_bins_arr = np.full((n_roles, n_pix), 0.0, dtype=float)
        junction_adjustment_bins_arr = np.full((n_roles, n_pix), np.nan, dtype=float)
        junction_distance_bins_arr = np.full((n_roles, n_pix), np.nan, dtype=float)
        residual_bins_arr = np.full((n_roles, n_pix), np.nan, dtype=float)
        unsupported_bins_arr = np.full((n_roles, n_pix), np.nan, dtype=float)
        regime_bins_arr = np.full((n_roles, n_pix), 0.0, dtype=float)
        conf_bins_arr = np.full((n_roles, n_pix), 0.0, dtype=float)
        prediction_confidence_bins_arr = np.full((n_roles, n_pix), np.nan, dtype=float)
        measured_anchor_fraction_bins_arr = np.full((n_roles, n_pix), np.nan, dtype=float)
        structure_only_fraction_bins_arr = np.full((n_roles, n_pix), np.nan, dtype=float)
        low_support_caution_bins_arr = np.full((n_roles, n_pix), np.nan, dtype=float)
        prediction_admissibility_bins_arr = np.full((n_roles, n_pix), np.nan, dtype=float)
        target_left_bank_arr = np.interp(station_arr, stations, target_left_bank_prof) if target_left_bank_prof is not None and np.any(np.isfinite(target_left_bank_prof)) else np.full(n_pix, np.nan, dtype=float)
        target_right_bank_arr = np.interp(station_arr, stations, target_right_bank_prof) if target_right_bank_prof is not None and np.any(np.isfinite(target_right_bank_prof)) else np.full(n_pix, np.nan, dtype=float)
        target_thalweg_arr = np.interp(station_arr, stations, target_thalweg_prof) if target_thalweg_prof is not None and np.any(np.isfinite(target_thalweg_prof)) else np.full(n_pix, np.nan, dtype=float)
        target_present_arr = np.interp(station_arr, stations, target_present_prof) if target_present_prof is not None and np.any(np.isfinite(target_present_prof)) else np.zeros(n_pix, dtype=float)
        target_local_reconciled_arr = np.interp(station_arr, stations, target_local_reconciled_prof) if target_local_reconciled_prof is not None and np.any(np.isfinite(target_local_reconciled_prof)) else np.zeros(n_pix, dtype=float)
        target_reconciliation_weight_arr = np.interp(station_arr, stations, target_reconciliation_weight_prof) if target_reconciliation_weight_prof is not None and np.any(np.isfinite(target_reconciliation_weight_prof)) else np.full(n_pix, np.nan, dtype=float)
        target_bed_support_distance_arr = np.interp(station_arr, stations, target_bed_support_distance_prof) if target_bed_support_distance_prof is not None and np.any(np.isfinite(target_bed_support_distance_prof)) else np.full(n_pix, np.nan, dtype=float)
        local_pixel_support_distance_arr = authoritative_transition_pixel_distance_m[rows[idxs], cols[idxs]].astype(float) if authoritative_transition_pixel_distance_m is not None else np.full(n_pix, np.nan, dtype=float)
        station_support_distance_arr = target_bed_support_distance_arr.copy()
        target_bed_support_distance_arr = _combine_support_distance_arrays(target_bed_support_distance_arr, local_pixel_support_distance_arr)
        component_plausibility_tol_m = _component_plausibility_tolerance_m(sub.get('bed_z_m', np.nan))

        for ridx, (role, role_eta) in enumerate(active_roles):
            z_prof = role_surfaces.get(comp, {}).get(role)
            if z_prof is None or z_prof.size == 0:
                continue
            s_prof = source_surfaces.get(comp, {}).get(role)
            m_prof = graph_mode_surfaces.get(comp, {}).get(role)
            sc_prof = graph_support_surfaces.get(comp, {}).get(role)
            uc_prof = graph_uncertainty_surfaces.get(comp, {}).get(role)
            hl_prof = graph_hard_lock_surfaces.get(comp, {}).get(role)
            jf_prof = graph_junction_flag_surfaces.get(comp, {}).get(role)
            jr_prof = graph_junction_role_surfaces.get(comp, {}).get(role)
            ja_prof = graph_junction_adjustment_surfaces.get(comp, {}).get(role)
            jd_prof = graph_junction_distance_surfaces.get(comp, {}).get(role)
            gr_prof = graph_residual_surfaces.get(comp, {}).get(role)
            us_prof = graph_unsupported_span_surfaces.get(comp, {}).get(role)
            ur_prof = graph_regime_surfaces.get(comp, {}).get(role)
            cf_prof = graph_confidence_surfaces.get(comp, {}).get(role)

            z_bins_arr[ridx, :] = np.interp(station_arr, stations, z_prof) if np.any(np.isfinite(z_prof)) else np.nan
            src_bins_arr[ridx, :] = np.interp(station_arr, stations, s_prof) if s_prof is not None and np.any(np.isfinite(s_prof)) else 0.0
            mode_bins_arr[ridx, :] = np.interp(station_arr, stations, m_prof) if m_prof is not None and np.any(np.isfinite(m_prof)) else 0.0
            support_bins_arr[ridx, :] = np.interp(station_arr, stations, sc_prof) if sc_prof is not None and np.any(np.isfinite(sc_prof)) else np.asarray([SUPPORT_CLASS_CODES.get(str(v), 0) for v in nearest_support_class], dtype=float)
            uncertainty_bins_arr[ridx, :] = np.interp(station_arr, stations, uc_prof) if uc_prof is not None and np.any(np.isfinite(uc_prof)) else float(UNCERTAINTY_CLASS_CODES['moderate'])
            hard_lock_bins_arr[ridx, :] = (np.interp(station_arr, stations, hl_prof) >= 0.5).astype(float) if hl_prof is not None and np.any(np.isfinite(hl_prof)) else 0.0
            junction_bins_arr[ridx, :] = (np.interp(station_arr, stations, jf_prof) >= 0.5).astype(float) if jf_prof is not None and np.any(np.isfinite(jf_prof)) else 0.0
            junction_role_bins_arr[ridx, :] = np.interp(station_arr, stations, jr_prof) if jr_prof is not None and np.any(np.isfinite(jr_prof)) else 0.0
            junction_adjustment_bins_arr[ridx, :] = np.interp(station_arr, stations, ja_prof) if ja_prof is not None and np.any(np.isfinite(ja_prof)) else np.nan
            junction_distance_bins_arr[ridx, :] = np.interp(station_arr, stations, jd_prof) if jd_prof is not None and np.any(np.isfinite(jd_prof)) else np.nan
            residual_bins_arr[ridx, :] = np.interp(station_arr, stations, gr_prof) if gr_prof is not None and np.any(np.isfinite(gr_prof)) else np.nan
            unsupported_bins_arr[ridx, :] = np.interp(station_arr, stations, us_prof) if us_prof is not None and np.any(np.isfinite(us_prof)) else np.nan
            regime_bins_arr[ridx, :] = np.interp(station_arr, stations, ur_prof) if ur_prof is not None and np.any(np.isfinite(ur_prof)) else float(UNSUPPORTED_REGIME_CODES['missing'])
            conf_bins_arr[ridx, :] = np.interp(station_arr, stations, cf_prof) if cf_prof is not None and np.any(np.isfinite(cf_prof)) else 0.0
            prediction_confidence_bins_arr[ridx, :] = np.interp(station_arr, stations, pc_prof) if pc_prof is not None and np.any(np.isfinite(pc_prof)) else np.nan
            measured_anchor_fraction_bins_arr[ridx, :] = np.interp(station_arr, stations, ma_prof) if ma_prof is not None and np.any(np.isfinite(ma_prof)) else np.nan
            structure_only_fraction_bins_arr[ridx, :] = np.interp(station_arr, stations, so_prof) if so_prof is not None and np.any(np.isfinite(so_prof)) else np.nan
            low_support_caution_bins_arr[ridx, :] = np.interp(station_arr, stations, lc_prof) if lc_prof is not None and np.any(np.isfinite(lc_prof)) else np.nan
            prediction_admissibility_bins_arr[ridx, :] = np.interp(station_arr, stations, pa_prof) if pa_prof is not None and np.any(np.isfinite(pa_prof)) else np.nan

        finite_roles_arr = np.sum(np.isfinite(z_bins_arr), axis=0)
        render_mode = component_render_modes.get(str(comp), 'full_scaffold')
        target_left_bank_vals = np.asarray(target_left_bank_arr[:len(idxs)], dtype=float)
        target_right_bank_vals = np.asarray(target_right_bank_arr[:len(idxs)], dtype=float)
        target_thalweg_vals = np.asarray(target_thalweg_arr[:len(idxs)], dtype=float)
        target_present_vals = np.asarray(target_present_arr[:len(idxs)] >= 0.5, dtype=bool)
        target_local_reconciled_vals = np.asarray(target_local_reconciled_arr[:len(idxs)] >= 0.5, dtype=bool)
        target_reconciliation_weight_vals = np.asarray(target_reconciliation_weight_arr[:len(idxs)], dtype=float)
        target_bed_support_distance_vals = np.asarray(target_bed_support_distance_arr[:len(idxs)], dtype=float)
        section_target_z_vals = np.full(len(idxs), np.nan, dtype=float)
        section_target_weight_vals = np.zeros(len(idxs), dtype=float)
        authoritative_transition_weight_vals = np.zeros(len(idxs), dtype=float)
        fallback_linear_z_vals = np.full(len(idxs), np.nan, dtype=float)
        thalweg_render_z_vals = np.full(len(idxs), np.nan, dtype=float)
        interior_semantic_weight_vals = np.clip(1.0 - ((np.abs(eta_arr - 0.5) / 0.5) ** 2.0), 0.0, 1.0)
        interior_semantic_weight_vals[(eta_arr >= 0.15) & (eta_arr <= 0.85)] = 1.0
        if render_mode == 'thalweg_dominant_scaffold' and len(idxs) > 0:
            if z_bins_arr.shape[1] == len(idxs):
                thalweg_render_z_vals = _thalweg_dominant_eta_values(eta_arr, eta_coords, z_bins_arr)
                for local_i in range(len(idxs)):
                    z_col = z_bins_arr[:, local_i]
                    valid = np.isfinite(z_col)
                    if np.count_nonzero(valid) >= 2:
                        fallback_linear_z_vals[local_i] = float(np.interp(float(eta_arr[local_i]), eta_coords[valid], z_col[valid]))
                    elif np.any(valid):
                        fallback_linear_z_vals[local_i] = _nearest_value(float(eta_arr[local_i]), eta_coords, valid, z_col, np.nan)
            else:
                thalweg_render_z_vals = np.full(len(idxs), np.nan, dtype=float)
            missing_lr = np.isfinite(target_thalweg_vals) & (~np.isfinite(target_left_bank_vals)) & np.isfinite(target_right_bank_vals)
            target_left_bank_vals[missing_lr] = target_right_bank_vals[missing_lr]
            missing_rl = np.isfinite(target_thalweg_vals) & (~np.isfinite(target_right_bank_vals)) & np.isfinite(target_left_bank_vals)
            target_right_bank_vals[missing_rl] = target_left_bank_vals[missing_rl]
            section_target_valid = np.isfinite(target_thalweg_vals) & (np.isfinite(target_left_bank_vals) | np.isfinite(target_right_bank_vals))
            section_target_present_mask = target_present_vals | target_local_reconciled_vals | section_target_valid
            if np.any(section_target_valid):
                section_target_z_vals[section_target_valid] = _section_target_eta_values(
                    eta_arr[section_target_valid],
                    target_left_bank_z=target_left_bank_vals[section_target_valid],
                    target_thalweg_z=target_thalweg_vals[section_target_valid],
                    target_right_bank_z=target_right_bank_vals[section_target_valid],
                )
            section_target_weight_vals = _section_target_blend_weights(
                target_present=section_target_present_mask,
                local_authoritative_reconciled=target_local_reconciled_vals,
                authoritative_reconciliation_weight=target_reconciliation_weight_vals,
                authoritative_bed_support_distance_m=target_bed_support_distance_vals,
            )
            authoritative_transition_weight_vals = _authoritative_transition_blend_weights(
                authoritative_bed_support_distance_m=target_bed_support_distance_vals,
                local_authoritative_reconciled=target_local_reconciled_vals,
                target_present=section_target_present_mask,
            )
            thalweg_role_idx = None
            for ridx, (role, _eta_role) in enumerate(component_active_roles.get(str(comp), ROLE_ORDER)):
                if role == 'thalweg':
                    thalweg_role_idx = ridx
                    break
            hard_lock_candidate_vals = np.zeros(len(idxs), dtype=float)
            if thalweg_role_idx is not None and hard_lock_bins_arr.shape[1] == len(idxs):
                hard_lock_candidate_vals = np.asarray(hard_lock_bins_arr[thalweg_role_idx, :len(idxs)], dtype=float)
            smoothed_thalweg_render_z_vals, longitudinal_smoothing_summary = _apply_component_longitudinal_smoothing(
                stations_m=station_arr,
                values_z=thalweg_render_z_vals,
                support_distance_m=target_bed_support_distance_vals,
                interior_weight=interior_semantic_weight_vals,
                hard_lock_mask=np.isfinite(hard_lock_candidate_vals) & (hard_lock_candidate_vals >= 0.5),
            )
            smoothing_row = {
                'component_id': str(comp),
                'render_mode': str(render_mode),
                'eligible_count': int(longitudinal_smoothing_summary.get('eligible_count', 0) or 0),
                'applied_count': int(longitudinal_smoothing_summary.get('applied_count', 0) or 0),
                'changed_count': int(longitudinal_smoothing_summary.get('changed_count', 0) or 0),
                'abs_adjustment_mean_m': float(longitudinal_smoothing_summary.get('abs_adjustment_m', {}).get('mean', 0.0) or 0.0),
                'reason': str(longitudinal_smoothing_summary.get('reason', 'unknown')),
            }
            component_longitudinal_smoothing_rows.append(smoothing_row)
            component_longitudinal_smoothing_by_id[str(comp)] = smoothing_row
            thalweg_render_z_vals = smoothed_thalweg_render_z_vals
        else:
            section_target_present_mask = target_present_vals | target_local_reconciled_vals
            thalweg_render_z_vals = np.full(len(idxs), np.nan, dtype=float)
            smoothing_row = {
                'component_id': str(comp),
                'render_mode': str(render_mode),
                'eligible_count': 0,
                'applied_count': 0,
                'changed_count': 0,
                'abs_adjustment_mean_m': 0.0,
                'reason': 'render_mode_not_thalweg_dominant',
            }
            component_longitudinal_smoothing_rows.append(smoothing_row)
            component_longitudinal_smoothing_by_id[str(comp)] = smoothing_row
        for local_i, k in enumerate(idxs):
            eta = float(eta_arr[local_i])
            z_col = z_bins_arr[:, local_i]
            src_col = src_bins_arr[:, local_i]
            mode_col = mode_bins_arr[:, local_i]
            support_col = support_bins_arr[:, local_i]
            uncertainty_col = uncertainty_bins_arr[:, local_i]
            hard_col = hard_lock_bins_arr[:, local_i]
            junc_col = junction_bins_arr[:, local_i]
            jrole_col = junction_role_bins_arr[:, local_i]
            jadj_col = junction_adjustment_bins_arr[:, local_i]
            jdist_col = junction_distance_bins_arr[:, local_i]
            resid_col = residual_bins_arr[:, local_i]
            unsup_col = unsupported_bins_arr[:, local_i]
            regime_col = regime_bins_arr[:, local_i]
            conf_col = conf_bins_arr[:, local_i]
            prediction_confidence_col = prediction_confidence_bins_arr[:, local_i]
            measured_anchor_fraction_col = measured_anchor_fraction_bins_arr[:, local_i]
            structure_only_fraction_col = structure_only_fraction_bins_arr[:, local_i]
            low_support_caution_col = low_support_caution_bins_arr[:, local_i]
            prediction_admissibility_col = prediction_admissibility_bins_arr[:, local_i]
            valid = np.isfinite(z_col)
            valid_src = valid & np.isfinite(src_col)
            valid_mode = valid & np.isfinite(mode_col)
            valid_support = valid & np.isfinite(support_col)
            valid_uncertainty = valid & np.isfinite(uncertainty_col)
            valid_hard = valid & np.isfinite(hard_col)
            valid_junc = valid & np.isfinite(junc_col)
            valid_junc_role = valid & np.isfinite(jrole_col)
            valid_junc_adj = valid & np.isfinite(jadj_col)
            valid_junc_dist = valid & np.isfinite(jdist_col)
            valid_resid = valid & np.isfinite(resid_col)
            valid_unsup = valid & np.isfinite(unsup_col)
            valid_regime = valid & np.isfinite(regime_col)
            valid_conf = valid & np.isfinite(conf_col)
            valid_prediction_confidence = valid & np.isfinite(prediction_confidence_col)
            valid_measured_anchor_fraction = valid & np.isfinite(measured_anchor_fraction_col)
            valid_structure_only_fraction = valid & np.isfinite(structure_only_fraction_col)
            valid_low_support_caution = valid & np.isfinite(low_support_caution_col)
            valid_prediction_admissibility = valid & np.isfinite(prediction_admissibility_col)
            if np.count_nonzero(valid) >= 2:
                if render_mode == 'thalweg_dominant_scaffold':
                    z_val = float(thalweg_render_z_vals[local_i])
                    if not np.isfinite(z_val):
                        z_val = _linear_eta_value(float(eta), eta_coords, z_col)
                else:
                    z_val = _linear_eta_value(float(eta), eta_coords, z_col)
                src_val = _nearest_value(eta, eta_coords, valid_src, src_col)
                mode_val = _nearest_value(eta, eta_coords, valid_mode, mode_col)
                support_val = _nearest_value(eta, eta_coords, valid_support, support_col)
                uncertainty_val = _nearest_value(eta, eta_coords, valid_uncertainty, uncertainty_col)
                hard_lock_val = _nearest_value(eta, eta_coords, valid_hard, hard_col)
                junction_val = _nearest_value(eta, eta_coords, valid_junc, junc_col)
                junction_role_val = _nearest_value(eta, eta_coords, valid_junc_role, jrole_col)
                junction_adjustment_val = float(np.interp(eta, eta_coords[valid_junc_adj], jadj_col[valid_junc_adj])) if np.count_nonzero(valid_junc_adj) >= 2 else _nearest_value(eta, eta_coords, valid_junc_adj, jadj_col, np.nan)
                junction_distance_val = float(np.interp(eta, eta_coords[valid_junc_dist], jdist_col[valid_junc_dist])) if np.count_nonzero(valid_junc_dist) >= 2 else _nearest_value(eta, eta_coords, valid_junc_dist, jdist_col, np.nan)
                residual_val = float(np.interp(eta, eta_coords[valid_resid], resid_col[valid_resid])) if np.count_nonzero(valid_resid) >= 2 else _nearest_value(eta, eta_coords, valid_resid, resid_col, np.nan)
                unsupported_val = float(np.interp(eta, eta_coords[valid_unsup], unsup_col[valid_unsup])) if np.count_nonzero(valid_unsup) >= 2 else _nearest_value(eta, eta_coords, valid_unsup, unsup_col, np.nan)
                regime_val = float(np.interp(eta, eta_coords[valid_regime], regime_col[valid_regime])) if np.count_nonzero(valid_regime) >= 2 else _nearest_value(eta, eta_coords, valid_regime, regime_col, 0.0)
                conf = float(np.interp(eta, eta_coords[valid_conf], conf_col[valid_conf])) if np.count_nonzero(valid_conf) >= 2 else _nearest_value(eta, eta_coords, valid_conf, conf_col, 0.0)
                prediction_confidence_val = float(np.interp(eta, eta_coords[valid_prediction_confidence], prediction_confidence_col[valid_prediction_confidence])) if np.count_nonzero(valid_prediction_confidence) >= 2 else _nearest_value(eta, eta_coords, valid_prediction_confidence, prediction_confidence_col, np.nan)
                measured_anchor_fraction_val = float(np.interp(eta, eta_coords[valid_measured_anchor_fraction], measured_anchor_fraction_col[valid_measured_anchor_fraction])) if np.count_nonzero(valid_measured_anchor_fraction) >= 2 else _nearest_value(eta, eta_coords, valid_measured_anchor_fraction, measured_anchor_fraction_col, np.nan)
                structure_only_fraction_val = float(np.interp(eta, eta_coords[valid_structure_only_fraction], structure_only_fraction_col[valid_structure_only_fraction])) if np.count_nonzero(valid_structure_only_fraction) >= 2 else _nearest_value(eta, eta_coords, valid_structure_only_fraction, structure_only_fraction_col, np.nan)
                low_support_caution_val = _nearest_value(eta, eta_coords, valid_low_support_caution, low_support_caution_col, np.nan)
                prediction_admissibility_val = _nearest_value(eta, eta_coords, valid_prediction_admissibility, prediction_admissibility_col, np.nan)
            elif np.count_nonzero(valid) == 1:
                z_val = float(z_col[valid][0])
                src_val = _nearest_value(eta, eta_coords, valid_src, src_col)
                mode_val = _nearest_value(eta, eta_coords, valid_mode, mode_col)
                support_val = _nearest_value(eta, eta_coords, valid_support, support_col)
                uncertainty_val = _nearest_value(eta, eta_coords, valid_uncertainty, uncertainty_col)
                hard_lock_val = _nearest_value(eta, eta_coords, valid_hard, hard_col)
                junction_val = _nearest_value(eta, eta_coords, valid_junc, junc_col)
                junction_role_val = _nearest_value(eta, eta_coords, valid_junc_role, jrole_col)
                junction_adjustment_val = _nearest_value(eta, eta_coords, valid_junc_adj, jadj_col, np.nan)
                junction_distance_val = _nearest_value(eta, eta_coords, valid_junc_dist, jdist_col, np.nan)
                residual_val = _nearest_value(eta, eta_coords, valid_resid, resid_col, np.nan)
                unsupported_val = _nearest_value(eta, eta_coords, valid_unsup, unsup_col, np.nan)
                regime_val = _nearest_value(eta, eta_coords, valid_regime, regime_col, 0.0)
                conf = _nearest_value(eta, eta_coords, valid_conf, conf_col, 0.0)
                prediction_confidence_val = _nearest_value(eta, eta_coords, valid_prediction_confidence, prediction_confidence_col, np.nan)
                measured_anchor_fraction_val = _nearest_value(eta, eta_coords, valid_measured_anchor_fraction, measured_anchor_fraction_col, np.nan)
                structure_only_fraction_val = _nearest_value(eta, eta_coords, valid_structure_only_fraction, structure_only_fraction_col, np.nan)
                low_support_caution_val = _nearest_value(eta, eta_coords, valid_low_support_caution, low_support_caution_col, np.nan)
                prediction_admissibility_val = _nearest_value(eta, eta_coords, valid_prediction_admissibility, prediction_admissibility_col, np.nan)
            else:
                z_val = np.nan
                src_val = 0.0
                mode_val = 0.0
                support_val = 0.0
                uncertainty_val = 0.0
                hard_lock_val = 0.0
                junction_val = 0.0
                junction_role_val = 0.0
                junction_adjustment_val = np.nan
                junction_distance_val = np.nan
                residual_val = np.nan
                unsupported_val = np.nan
                regime_val = 0.0
                conf = 0.0
                prediction_confidence_val = np.nan
                measured_anchor_fraction_val = np.nan
                structure_only_fraction_val = np.nan
                low_support_caution_val = np.nan
                prediction_admissibility_val = np.nan
            target_left_bank_val = float(target_left_bank_vals[local_i]) if local_i < target_left_bank_vals.size else float('nan')
            target_right_bank_val = float(target_right_bank_vals[local_i]) if local_i < target_right_bank_vals.size else float('nan')
            target_thalweg_val = float(target_thalweg_vals[local_i]) if local_i < target_thalweg_vals.size else float('nan')
            target_present_val = bool(local_i < target_present_vals.size and target_present_vals[local_i])
            target_local_reconciled_val = bool(local_i < target_local_reconciled_vals.size and target_local_reconciled_vals[local_i])
            effective_target_present_val = bool(local_i < section_target_present_mask.size and section_target_present_mask[local_i])
            target_reconciliation_weight_val = float(target_reconciliation_weight_vals[local_i]) if local_i < target_reconciliation_weight_vals.size else float('nan')
            target_bed_support_distance_val = float(target_bed_support_distance_vals[local_i]) if local_i < target_bed_support_distance_vals.size else float('nan')
            authoritative_transition_weight_val = 0.0
            if render_mode == 'thalweg_dominant_scaffold' and np.isfinite(z_val):
                target_section_z = float(section_target_z_vals[local_i]) if local_i < section_target_z_vals.size else float('nan')
                target_section_weight = float(section_target_weight_vals[local_i]) if local_i < section_target_weight_vals.size else 0.0
                authoritative_transition_weight_val = float(authoritative_transition_weight_vals[local_i]) if local_i < authoritative_transition_weight_vals.size else 0.0
                if np.isfinite(target_section_z) and (target_section_weight > 0.0 or authoritative_transition_weight_val > 0.0):
                    interior_semantic_weight = float(interior_semantic_weight_vals[local_i]) if local_i < interior_semantic_weight_vals.size else _interior_semantic_weight(float(eta))
                    generic_section_weight = min(float(target_section_weight), float(SECTION_TARGET_GENERIC_BLEND_CEILING))
                    authoritative_bonus_weight = max(float(target_section_weight) - generic_section_weight, 0.0)
                    authoritative_transition_weight_val *= interior_semantic_weight
                    target_section_weight = generic_section_weight + (authoritative_bonus_weight * interior_semantic_weight)
                    if target_local_reconciled_val and max(target_section_weight, authoritative_transition_weight_val) > 0.0:
                        authoritative_transition_weight_val = max(authoritative_transition_weight_val, min(max(float(target_section_weight), authoritative_transition_weight_val), 0.85) * interior_semantic_weight)
                    elif effective_target_present_val and max(target_section_weight, authoritative_transition_weight_val) > 0.0 and np.isfinite(target_bed_support_distance_val):
                        authoritative_transition_weight_val = max(authoritative_transition_weight_val, min(max(float(target_section_weight), authoritative_transition_weight_val), 0.45) * interior_semantic_weight)
                    target_section_weight = max(target_section_weight, authoritative_transition_weight_val)
                    if abs(float(target_section_z) - float(z_val)) <= component_plausibility_tol_m or target_local_reconciled_val or authoritative_transition_weight_val > 0.0:
                        z_val = (1.0 - target_section_weight) * float(z_val) + target_section_weight * float(target_section_z)
                        current_src_code = int(np.clip(np.rint(src_val), 0, max(SOURCE_CODES.values())))
                        if target_local_reconciled_val and target_section_weight >= 0.40 and interior_semantic_weight >= 0.45:
                            src_val = float(SOURCE_CODES['station_target_local_authoritative_reconciliation'])
                        elif effective_target_present_val and target_section_weight >= 0.18 and current_src_code not in (SOURCE_CODES['authoritative_in_channel'], SOURCE_CODES['authoritative_bank_margin']):
                            if interior_semantic_weight < 0.45:
                                src_val = float(SOURCE_CODES['bank_edge_geometry_constraint'])
                            else:
                                src_val = float(SOURCE_CODES['station_target_section_tendency'])
            src_code = int(np.clip(np.rint(src_val), 0, max(SOURCE_CODES.values())))
            mode_code = int(np.clip(np.rint(mode_val), 0, max(GRAPH_MODE_CODES.values())))
            finite_roles = int(finite_roles_arr[local_i])
            rounded_src = np.clip(np.rint(src_col), 0, max(SOURCE_CODES.values())).astype(int)
            support_codes_local = np.clip(np.rint(support_col), 0, max(SUPPORT_CLASS_CODES.values())).astype(int)
            xs_support_codes = {
                int(SUPPORT_CLASS_CODES.get('xs_residual_only', 0)),
                int(SUPPORT_CLASS_CODES.get('xs_supported', 0)),
            }
            xs_participates = bool(
                np.any(valid_src & (rounded_src == SOURCE_CODES['xs_profile_resampled']))
                or np.any(valid_support & np.isin(support_codes_local, list(xs_support_codes)))
            )
            authoritative_participates = bool(np.any(valid_src & np.isin(rounded_src, [SOURCE_CODES['authoritative_in_channel']])))
            authoritative_bank_participates = bool(np.any(valid_src & np.isin(rounded_src, [SOURCE_CODES['authoritative_bank_margin']])))
            authoritative_transition_weight_out[rows[k], cols[k]] = np.float32(authoritative_transition_weight_val) if np.isfinite(authoritative_transition_weight_val) else np.float32(np.nan)
            if auth_mask is not None and auth_depth is not None:
                am = auth_mask[rows[k], cols[k]]
                ad = auth_depth[rows[k], cols[k]]
                auth_role_supports_bed_lock = True
                if auth_role_code is not None:
                    raw_role_code = auth_role_code[rows[k], cols[k]]
                    if np.isfinite(raw_role_code):
                        auth_role_supports_bed_lock = code_to_role(int(np.clip(np.rint(raw_role_code), 0, 255))) in {ROLE_BED_CORE, ROLE_BED_INNER}
                support_code_prelock = int(np.clip(np.rint(support_val), 0, max(SUPPORT_CLASS_CODES.values()))) if np.isfinite(support_val) else 0
                src_is_authoritative = src_code in (SOURCE_CODES['authoritative_in_channel'],)
                support_aware_lock_expected = (
                    support_code_prelock in authoritative_codes
                    or src_is_authoritative
                    or (np.isfinite(hard_lock_val) and hard_lock_val >= 0.5)
                )
                if np.isfinite(am) and float(am) > 0.0 and np.isfinite(ad):
                    authoritative_lock_scope_out[rows[k], cols[k]] = np.int16(1 if (support_aware_lock_expected and auth_role_supports_bed_lock) else 0)
                    if auth_role_supports_bed_lock:
                        z_val = float(ad)
                        src_code = int(SOURCE_CODES['authoritative_in_channel'])
                        mode_code = int(GRAPH_MODE_CODES['hard_locked'])
                        support_val = float(SUPPORT_CLASS_CODES['authoritative_locked'])
                        uncertainty_val = float(UNCERTAINTY_CLASS_CODES['very_low'])
                        hard_lock_val = 1.0
                        junction_val = 0.0
                        junction_role_val = 0.0
                        junction_adjustment_val = 0.0
                        junction_distance_val = 0.0
                        residual_val = 0.0
                        unsupported_val = 0.0
                        regime_val = float(UNSUPPORTED_REGIME_CODES['supported'])
                        conf = 1.0
                        finite_roles = max(int(finite_roles), 1)
                        authoritative_participates = True
                        authoritative_lock_applied_out[rows[k], cols[k]] = np.int16(1)
                    else:
                        authoritative_bank_participates = True
            if np.isfinite(z_val):
                if np.isfinite(prediction_confidence_val):
                    if bool(np.isfinite(hard_lock_val) and hard_lock_val >= 0.5) or bool(authoritative_participates and src_code == SOURCE_CODES['authoritative_in_channel']):
                        conf = 1.0
                        prediction_confidence_val = 1.0
                        low_support_caution_val = 0.0
                        prediction_admissibility_val = 1.0
                        measured_anchor_fraction_val = 1.0 if not np.isfinite(measured_anchor_fraction_val) else max(float(measured_anchor_fraction_val), 1.0)
                        structure_only_fraction_val = 0.0
                    else:
                        conf = float(np.clip(0.55 * conf + 0.45 * prediction_confidence_val - (0.10 if np.isfinite(low_support_caution_val) and low_support_caution_val >= 0.5 else 0.0), 0.0, 1.0))
                prediction_support_confidence_out[rows[k], cols[k]] = np.float32(prediction_confidence_val) if np.isfinite(prediction_confidence_val) else np.float32(np.nan)
                prediction_measured_anchor_fraction_out[rows[k], cols[k]] = np.float32(measured_anchor_fraction_val) if np.isfinite(measured_anchor_fraction_val) else np.float32(np.nan)
                prediction_structure_only_fraction_out[rows[k], cols[k]] = np.float32(structure_only_fraction_val) if np.isfinite(structure_only_fraction_val) else np.float32(np.nan)
                prediction_low_support_caution_out[rows[k], cols[k]] = np.int16(1 if np.isfinite(low_support_caution_val) and low_support_caution_val >= 0.5 else 0)
                prediction_admissibility_out[rows[k], cols[k]] = np.int16(1 if np.isfinite(prediction_admissibility_val) and prediction_admissibility_val >= 0.5 else 0)
            z_out[rows[k], cols[k]] = np.float32(z_val) if np.isfinite(z_val) else np.float32(np.nan)
            conf_out[rows[k], cols[k]] = np.float32(conf)
            src_out[rows[k], cols[k]] = np.int16(src_code)
            mode_out[rows[k], cols[k]] = np.int16(mode_code)
            support_out[rows[k], cols[k]] = np.int16(max(1, finite_roles) if np.isfinite(z_val) else 0)
            support_class_out[rows[k], cols[k]] = np.int16(np.clip(np.rint(support_val), 0, max(SUPPORT_CLASS_CODES.values()))) if np.isfinite(z_val) else np.int16(0)
            uncertainty_out[rows[k], cols[k]] = np.int16(np.clip(np.rint(uncertainty_val), 0, max(UNCERTAINTY_CLASS_CODES.values()))) if np.isfinite(z_val) else np.int16(0)
            hard_lock_out[rows[k], cols[k]] = np.int16(1 if hard_lock_val >= 0.5 and np.isfinite(z_val) else 0)
            junction_flag_out[rows[k], cols[k]] = np.int16(1 if junction_val >= 0.5 and np.isfinite(z_val) else 0)
            junction_role_out[rows[k], cols[k]] = np.int16(np.clip(np.rint(junction_role_val), 0, max(JUNCTION_ROLE_CODES.values()))) if np.isfinite(z_val) else np.int16(0)
            junction_adjustment_out[rows[k], cols[k]] = np.float32(junction_adjustment_val) if np.isfinite(z_val) and np.isfinite(junction_adjustment_val) else np.float32(np.nan)
            junction_distance_out[rows[k], cols[k]] = np.float32(junction_distance_val) if np.isfinite(z_val) and np.isfinite(junction_distance_val) else np.float32(np.nan)
            unsupported_span_out[rows[k], cols[k]] = np.float32(unsupported_val) if np.isfinite(z_val) and np.isfinite(unsupported_val) else np.float32(np.nan)
            unsupported_regime_out[rows[k], cols[k]] = np.int16(np.clip(np.rint(regime_val), 0, max(UNSUPPORTED_REGIME_CODES.values()))) if np.isfinite(z_val) else np.int16(0)
            residual_to_candidate_out[rows[k], cols[k]] = np.float32(residual_val) if np.isfinite(z_val) and np.isfinite(residual_val) else np.float32(np.nan)
            xs_participation_out[rows[k], cols[k]] = np.int16(1 if np.isfinite(z_val) and xs_participates else 0)
            authoritative_participation_out[rows[k], cols[k]] = np.int16(1 if np.isfinite(z_val) and authoritative_participates else 0)
            if np.isfinite(z_val):
                if authoritative_participates and xs_participates:
                    influence_class_out[rows[k], cols[k]] = np.int16(INFLUENCE_CLASS_CODES['mixed_authoritative_xs'])
                elif authoritative_participates:
                    influence_class_out[rows[k], cols[k]] = np.int16(INFLUENCE_CLASS_CODES['authoritative_only'])
                elif xs_participates:
                    influence_class_out[rows[k], cols[k]] = np.int16(INFLUENCE_CLASS_CODES['xs_only'])
                else:
                    influence_class_out[rows[k], cols[k]] = np.int16(INFLUENCE_CLASS_CODES['other_scaffold_only'])
            else:
                influence_class_out[rows[k], cols[k]] = np.int16(INFLUENCE_CLASS_CODES['missing'])
        selector_row = component_render_selector_by_id.get(str(comp), {})
        selected_for_thalweg_render = bool(render_mode == 'thalweg_dominant_scaffold')
        candidate_pixel_count = int(np.count_nonzero(np.isfinite(thalweg_render_z_vals))) if selected_for_thalweg_render else 0
        thalweg_render_delta_mask = np.isfinite(fallback_linear_z_vals) & np.isfinite(thalweg_render_z_vals)
        thalweg_render_delta_vals = np.abs(thalweg_render_z_vals[thalweg_render_delta_mask] - fallback_linear_z_vals[thalweg_render_delta_mask]) if candidate_pixel_count > 0 else np.array([], dtype=float)
        section_target_apply_mask = np.isfinite(section_target_z_vals) & np.isfinite(section_target_weight_vals) & (section_target_weight_vals > 0.0)
        transition_distance_valid_mask = np.isfinite(target_bed_support_distance_vals)
        transition_inrange_mask = transition_distance_valid_mask & (target_bed_support_distance_vals <= AUTHORITATIVE_TRANSITION_OUTER_M)
        transition_candidate_mask = np.isfinite(authoritative_transition_weight_vals) & (transition_distance_valid_mask | np.isfinite(section_target_z_vals))
        transition_presemantic_nonzero_mask = np.isfinite(authoritative_transition_weight_vals) & (authoritative_transition_weight_vals > 0.0)
        transition_apply_mask = np.isfinite(authoritative_transition_weight_vals) & (authoritative_transition_weight_vals > 0.0)
        smoothing_row = component_longitudinal_smoothing_by_id.get(str(comp), {})
        final_component_vals = z_out[rows[idxs], cols[idxs]].astype(float) if len(idxs) > 0 else np.array([], dtype=float)
        final_changed_mask = np.isfinite(final_component_vals) & np.isfinite(fallback_linear_z_vals) & (np.abs(final_component_vals - fallback_linear_z_vals) > 1.0e-6)
        component_effect_rows.append({
            'component_id': str(comp),
            'render_mode': str(render_mode),
            'selection_reason': str(component_render_mode_reasons.get(str(comp), 'unknown')),
            'selected_for_thalweg_render': selected_for_thalweg_render,
            'weak_support_semantic_present': bool(selector_row.get('weak_support_present', False)),
            'thalweg_render_candidate_present': bool(candidate_pixel_count > 0),
            'candidate_pixel_count': candidate_pixel_count,
            'thalweg_render_delta_count': int(np.count_nonzero(thalweg_render_delta_mask & (np.abs(thalweg_render_z_vals - fallback_linear_z_vals) > 1.0e-6))) if candidate_pixel_count > 0 else 0,
            'fallback_comparison_pixel_count': int(np.count_nonzero(np.isfinite(fallback_linear_z_vals))),
            'fallback_delta_abs_mean_m': float(np.nanmean(thalweg_render_delta_vals)) if thalweg_render_delta_vals.size else 0.0,
            'fallback_delta_abs_p95_m': float(np.nanpercentile(thalweg_render_delta_vals, 95)) if thalweg_render_delta_vals.size else 0.0,
            'section_target_geometry_count': int(np.count_nonzero(np.isfinite(section_target_z_vals))),
            'section_target_candidate_count': int(np.count_nonzero(np.isfinite(section_target_z_vals) & (section_target_weight_vals > 0.0))),
            'section_target_applied_count': int(np.count_nonzero(section_target_apply_mask)),
            'authoritative_transition_distance_valid_count': int(np.count_nonzero(transition_distance_valid_mask)),
            'authoritative_transition_inrange_count': int(np.count_nonzero(transition_inrange_mask)),
            'authoritative_transition_candidate_count': int(np.count_nonzero(transition_candidate_mask)),
            'authoritative_transition_presemantic_nonzero_count': int(np.count_nonzero(transition_presemantic_nonzero_mask)),
            'authoritative_transition_applied_count': int(np.count_nonzero(transition_apply_mask)),
            'authoritative_transition_pixel_distance_assisted_count': int(np.count_nonzero(np.isfinite(local_pixel_support_distance_arr) & ~np.isfinite(station_support_distance_arr))),
            'longitudinal_smoothing_changed_count': int(smoothing_row.get('changed_count', 0) or 0),
            'final_changed_pixel_count': int(np.count_nonzero(final_changed_mask)),
            'backbone_reference_implausible': bool(selector_row.get('backbone_reference_implausible', False)),
            'auth_share': float(selector_row.get('auth_share', np.nan)) if selector_row else np.nan,
        })
    outputs = {}
    profile.update(dtype='float32', count=1, compress='deflate')
    if not profile.get('tiled'):
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
    for name, arr in [
        ('river_channel_surface.tif', z_out),
        ('river_channel_surface_confidence.tif', conf_out),
    ]:
        out = river_dir / name
        profile_one = profile.copy()
        profile_one['nodata'] = -9999.0
        with rasterio.open(out, 'w', **profile_one) as ds:
            write = np.where(np.isfinite(arr), arr.astype(np.float32), np.float32(-9999.0))
            ds.write(write, 1)
        outputs[name] = str(out)
    src_profile = profile.copy()
    src_profile.update(dtype='int16', nodata=0, compress='deflate')
    src_path = river_dir / 'river_channel_surface_source_class.tif'
    with rasterio.open(src_path, 'w', **src_profile) as ds:
        ds.write(src_out.astype(np.int16), 1)
    outputs['river_channel_surface_source_class.tif'] = str(src_path)
    mode_profile = profile.copy()
    mode_profile.update(dtype='int16', nodata=0, compress='deflate')
    mode_path = river_dir / 'river_channel_surface_graph_mode.tif'
    with rasterio.open(mode_path, 'w', **mode_profile) as ds:
        ds.write(mode_out.astype(np.int16), 1)
    outputs['river_channel_surface_graph_mode.tif'] = str(mode_path)
    cnt_profile = profile.copy()
    cnt_profile.update(dtype='int16', nodata=-32768, compress='deflate')
    cnt_path = river_dir / 'river_channel_surface_support_count.tif'
    with rasterio.open(cnt_path, 'w', **cnt_profile) as ds:
        ds.write(np.where(corridor, support_out, np.int16(-32768)).astype(np.int16), 1)
    outputs['river_channel_surface_support_count.tif'] = str(cnt_path)

    support_class_profile = profile.copy()
    support_class_profile.update(dtype='int16', nodata=0, compress='deflate')
    support_class_path = river_dir / 'river_channel_surface_support_class.tif'
    with rasterio.open(support_class_path, 'w', **support_class_profile) as ds:
        ds.write(support_class_out.astype(np.int16), 1)
    outputs['river_channel_surface_support_class.tif'] = str(support_class_path)

    uncertainty_profile = profile.copy()
    uncertainty_profile.update(dtype='int16', nodata=0, compress='deflate')
    uncertainty_path = river_dir / 'river_channel_surface_uncertainty.tif'
    with rasterio.open(uncertainty_path, 'w', **uncertainty_profile) as ds:
        ds.write(uncertainty_out.astype(np.int16), 1)
    outputs['river_channel_surface_uncertainty.tif'] = str(uncertainty_path)

    flag_profile = profile.copy()
    flag_profile.update(dtype='int16', nodata=0, compress='deflate')
    hard_lock_path = river_dir / 'river_channel_surface_hard_lock.tif'
    with rasterio.open(hard_lock_path, 'w', **flag_profile) as ds:
        ds.write(hard_lock_out.astype(np.int16), 1)
    outputs['river_channel_surface_hard_lock.tif'] = str(hard_lock_path)
    junction_flag_path = river_dir / 'river_channel_surface_junction_constrained.tif'
    with rasterio.open(junction_flag_path, 'w', **flag_profile) as ds:
        ds.write(junction_flag_out.astype(np.int16), 1)
    outputs['river_channel_surface_junction_constrained.tif'] = str(junction_flag_path)
    junction_role_path = river_dir / 'river_channel_surface_junction_role.tif'
    with rasterio.open(junction_role_path, 'w', **flag_profile) as ds:
        ds.write(junction_role_out.astype(np.int16), 1)
    outputs['river_channel_surface_junction_role.tif'] = str(junction_role_path)

    float_profile = profile.copy()
    float_profile.update(dtype='float32', nodata=-9999.0, compress='deflate')
    unsupported_path = river_dir / 'river_channel_surface_unsupported_span.tif'
    with rasterio.open(unsupported_path, 'w', **float_profile) as ds:
        ds.write(np.where(np.isfinite(unsupported_span_out), unsupported_span_out, np.float32(-9999.0)), 1)
    outputs['river_channel_surface_unsupported_span.tif'] = str(unsupported_path)
    unsupported_regime_path = river_dir / 'river_channel_surface_unsupported_regime.tif'
    with rasterio.open(unsupported_regime_path, 'w', **flag_profile) as ds:
        ds.write(unsupported_regime_out.astype(np.int16), 1)
    outputs['river_channel_surface_unsupported_regime.tif'] = str(unsupported_regime_path)
    junction_adjustment_path = river_dir / 'river_channel_surface_junction_adjustment.tif'
    with rasterio.open(junction_adjustment_path, 'w', **float_profile) as ds:
        ds.write(np.where(np.isfinite(junction_adjustment_out), junction_adjustment_out, np.float32(-9999.0)), 1)
    outputs['river_channel_surface_junction_adjustment.tif'] = str(junction_adjustment_path)
    junction_distance_path = river_dir / 'river_channel_surface_junction_distance.tif'
    with rasterio.open(junction_distance_path, 'w', **float_profile) as ds:
        ds.write(np.where(np.isfinite(junction_distance_out), junction_distance_out, np.float32(-9999.0)), 1)
    outputs['river_channel_surface_junction_distance.tif'] = str(junction_distance_path)
    residual_path = river_dir / 'river_channel_surface_residual_to_candidate.tif'
    with rasterio.open(residual_path, 'w', **float_profile) as ds:
        ds.write(np.where(np.isfinite(residual_to_candidate_out), residual_to_candidate_out, np.float32(-9999.0)), 1)
    outputs['river_channel_surface_residual_to_candidate.tif'] = str(residual_path)
    xs_participation_path = river_dir / 'river_channel_surface_xs_participation.tif'
    with rasterio.open(xs_participation_path, 'w', **flag_profile) as ds:
        ds.write(xs_participation_out.astype(np.int16), 1)
    outputs['river_channel_surface_xs_participation.tif'] = str(xs_participation_path)
    authoritative_participation_path = river_dir / 'river_channel_surface_authoritative_participation.tif'
    with rasterio.open(authoritative_participation_path, 'w', **flag_profile) as ds:
        ds.write(authoritative_participation_out.astype(np.int16), 1)
    outputs['river_channel_surface_authoritative_participation.tif'] = str(authoritative_participation_path)
    influence_class_path = river_dir / 'river_channel_surface_influence_class.tif'
    with rasterio.open(influence_class_path, 'w', **flag_profile) as ds:
        ds.write(influence_class_out.astype(np.int16), 1)
    outputs['river_channel_surface_influence_class.tif'] = str(influence_class_path)
    prediction_confidence_path = river_dir / 'river_channel_surface_prediction_support_confidence.tif'
    with rasterio.open(prediction_confidence_path, 'w', **float_profile) as ds:
        ds.write(np.where(np.isfinite(prediction_support_confidence_out), prediction_support_confidence_out, np.float32(-9999.0)), 1)
    outputs['river_channel_surface_prediction_support_confidence.tif'] = str(prediction_confidence_path)
    measured_anchor_fraction_path = river_dir / 'river_channel_surface_measured_anchor_fraction.tif'
    with rasterio.open(measured_anchor_fraction_path, 'w', **float_profile) as ds:
        ds.write(np.where(np.isfinite(prediction_measured_anchor_fraction_out), prediction_measured_anchor_fraction_out, np.float32(-9999.0)), 1)
    outputs['river_channel_surface_measured_anchor_fraction.tif'] = str(measured_anchor_fraction_path)
    structure_only_fraction_path = river_dir / 'river_channel_surface_structure_only_fraction.tif'
    with rasterio.open(structure_only_fraction_path, 'w', **float_profile) as ds:
        ds.write(np.where(np.isfinite(prediction_structure_only_fraction_out), prediction_structure_only_fraction_out, np.float32(-9999.0)), 1)
    outputs['river_channel_surface_structure_only_fraction.tif'] = str(structure_only_fraction_path)
    prediction_caution_path = river_dir / 'river_channel_surface_low_support_caution.tif'
    with rasterio.open(prediction_caution_path, 'w', **flag_profile) as ds:
        ds.write(prediction_low_support_caution_out.astype(np.int16), 1)
    outputs['river_channel_surface_low_support_caution.tif'] = str(prediction_caution_path)
    prediction_admissibility_path = river_dir / 'river_channel_surface_prediction_admissibility.tif'
    with rasterio.open(prediction_admissibility_path, 'w', **flag_profile) as ds:
        ds.write(prediction_admissibility_out.astype(np.int16), 1)
    outputs['river_channel_surface_prediction_admissibility.tif'] = str(prediction_admissibility_path)
    authoritative_transition_weight_path = river_dir / 'river_channel_surface_authoritative_transition_weight.tif'
    with rasterio.open(authoritative_transition_weight_path, 'w', **float_profile) as ds:
        ds.write(np.where(np.isfinite(authoritative_transition_weight_out), authoritative_transition_weight_out, np.float32(-9999.0)), 1)
    outputs['river_channel_surface_authoritative_transition_weight.tif'] = str(authoritative_transition_weight_path)
    outputs['channel_surface_authoritative_transition_weight'] = str(authoritative_transition_weight_path)

    non_authoritative_support_mask = np.isin(support_class_out, [
        SUPPORT_CLASS_CODES['anchored_interpolated'],
        SUPPORT_CLASS_CODES['stage_controlled'],
        SUPPORT_CLASS_CODES['graph_backbone'],
        SUPPORT_CLASS_CODES['xs_residual_only'],
        SUPPORT_CLASS_CODES['unsupported'],
    ])
    xs_admissibility_mask = corridor & non_authoritative_support_mask
    auth_lock_scope_path = river_dir / 'river_channel_surface_authoritative_lock_scope.tif'
    with rasterio.open(auth_lock_scope_path, 'w', **flag_profile) as ds:
        ds.write(authoritative_lock_scope_out.astype(np.int16), 1)
    outputs['river_channel_surface_authoritative_lock_scope.tif'] = str(auth_lock_scope_path)
    outputs['channel_surface_authoritative_lock_scope'] = str(auth_lock_scope_path)
    auth_lock_applied_path = river_dir / 'river_channel_surface_authoritative_lock_applied.tif'
    with rasterio.open(auth_lock_applied_path, 'w', **flag_profile) as ds:
        ds.write(authoritative_lock_applied_out.astype(np.int16), 1)
    outputs['river_channel_surface_authoritative_lock_applied.tif'] = str(auth_lock_applied_path)
    outputs['channel_surface_authoritative_lock_applied'] = str(auth_lock_applied_path)
    xs_admissibility_path = river_dir / 'river_channel_surface_xs_admissibility_mask.tif'
    with rasterio.open(xs_admissibility_path, 'w', **flag_profile) as ds:
        ds.write(xs_admissibility_mask.astype(np.int16), 1)
    outputs['river_channel_surface_xs_admissibility_mask.tif'] = str(xs_admissibility_path)

    control_nodes_path = river_dir / 'river_channel_surface_control_nodes.gpkg'
    control_nodes_export = _write_control_nodes_export(nodes, control_nodes_path, active_logger)
    exported_control_nodes_path = Path(control_nodes_export) if control_nodes_export is not None else None
    if exported_control_nodes_path is not None:
        outputs['river_channel_surface_control_nodes.gpkg'] = str(exported_control_nodes_path)

    populated_mask = np.isfinite(z_out)
    xs_warning = bool(input_xs_nodes > 0 and _bool_count(xs_participation_out > 0) == 0 and _bool_count(populated_mask & non_authoritative_support_mask) > 0)
    nodes = _backfill_node_section_target_fields(
        nodes,
        thalweg_by_comp=thalweg_by_comp,
        station_target_left_bank_surfaces=station_target_left_bank_surfaces,
        station_target_right_bank_surfaces=station_target_right_bank_surfaces,
        station_target_thalweg_surfaces=station_target_thalweg_surfaces,
        station_target_present_surfaces=station_target_present_surfaces,
        station_target_local_authoritative_reconciled_surfaces=station_target_local_authoritative_reconciled_surfaces,
        station_authoritative_reconciliation_weight_surfaces=station_authoritative_reconciliation_weight_surfaces,
        station_authoritative_bed_support_distance_surfaces=station_authoritative_bed_support_distance_surfaces,
    )
    section_target_outputs, section_target_summary = _write_section_target_agreement_receipts(
        river_dir=river_dir,
        nodes=nodes,
        z_out=z_out,
        transform=transform,
        logger=active_logger,
    )
    role_agreement_outputs, role_agreement_summary = _write_role_agreement_receipts(
        river_dir=river_dir,
        nodes=nodes,
        z_out=z_out,
        transform=transform,
        logger=active_logger,
    )
    channel_surface_effect_outputs, channel_surface_effect_summary = _write_channel_surface_effect_receipts(
        river_dir=river_dir,
        component_effect_rows=component_effect_rows,
        logger=active_logger,
    )
    channel_surface_longitudinal_smoothing_outputs, channel_surface_longitudinal_smoothing_summary = _write_channel_surface_longitudinal_smoothing_receipts(
        river_dir=river_dir,
        rows=component_longitudinal_smoothing_rows,
        logger=active_logger,
    )
    authoritative_transition_summary_path = river_dir / 'river_channel_surface_authoritative_transition_summary.json'
    effect_df = pd.DataFrame(component_effect_rows)
    authoritative_transition_summary = {
        'available': True,
        'distance_valid_cell_count': int(pd.to_numeric(effect_df.get('authoritative_transition_distance_valid_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not effect_df.empty else 0,
        'inrange_cell_count': int(pd.to_numeric(effect_df.get('authoritative_transition_inrange_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not effect_df.empty else 0,
        'candidate_cell_count': int(pd.to_numeric(effect_df.get('authoritative_transition_candidate_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not effect_df.empty else 0,
        'presemantic_nonzero_cell_count': int(pd.to_numeric(effect_df.get('authoritative_transition_presemantic_nonzero_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not effect_df.empty else 0,
        'pixel_distance_assisted_cell_count': int(pd.to_numeric(effect_df.get('authoritative_transition_pixel_distance_assisted_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not effect_df.empty else 0,
        'nonzero_cell_count': _bool_count(np.isfinite(authoritative_transition_weight_out) & (authoritative_transition_weight_out > 0)),
        'weight_distribution': _distribution(authoritative_transition_weight_out[np.isfinite(authoritative_transition_weight_out)]),
        'inner_m': AUTHORITATIVE_TRANSITION_INNER_M,
        'outer_m': AUTHORITATIVE_TRANSITION_OUTER_M,
        'curve_power': AUTHORITATIVE_TRANSITION_CURVE_POWER,
        'generic_ceiling': AUTHORITATIVE_TRANSITION_GENERIC_CEILING,
    }
    authoritative_transition_summary_path.write_text(json.dumps(authoritative_transition_summary, indent=2), encoding='utf-8')
    outputs['channel_surface_authoritative_transition_summary'] = str(authoritative_transition_summary_path)

    base_artifacts = _base_channel_surface_artifacts(
        river_dir=river_dir,
        src_path=src_path,
        mode_path=mode_path,
        cnt_path=cnt_path,
        support_class_path=support_class_path,
        uncertainty_path=uncertainty_path,
        hard_lock_path=hard_lock_path,
        junction_flag_path=junction_flag_path,
        junction_role_path=junction_role_path,
        junction_adjustment_path=junction_adjustment_path,
        junction_distance_path=junction_distance_path,
        unsupported_path=unsupported_path,
        unsupported_regime_path=unsupported_regime_path,
        residual_path=residual_path,
        xs_participation_path=xs_participation_path,
        authoritative_participation_path=authoritative_participation_path,
        influence_class_path=influence_class_path,
    )
    audit_artifacts = {
        'channel_surface_xs_participation': str(xs_participation_path),
        'channel_surface_authoritative_participation': str(authoritative_participation_path),
        'channel_surface_influence_class': str(influence_class_path),
        'channel_surface_prediction_support_confidence': str(prediction_confidence_path),
        'channel_surface_measured_anchor_fraction': str(measured_anchor_fraction_path),
        'channel_surface_structure_only_fraction': str(structure_only_fraction_path),
        'channel_surface_low_support_caution': str(prediction_caution_path),
        'channel_surface_prediction_admissibility': str(prediction_admissibility_path),
        'channel_surface_authoritative_transition_weight': str(authoritative_transition_weight_path),
        'channel_surface_xs_admissibility_mask': str(xs_admissibility_path),
        'channel_surface_authoritative_lock_scope': str(auth_lock_scope_path),
        'channel_surface_authoritative_lock_applied': str(auth_lock_applied_path),
        **({'channel_surface_control_nodes': str(exported_control_nodes_path)} if exported_control_nodes_path is not None else {}),
        **({k: v for k, v in section_target_outputs.items() if v}),
        **({k: v for k, v in role_agreement_outputs.items() if v}),
        **({k: v for k, v in channel_surface_effect_outputs.items() if v}),
        **({k: v for k, v in channel_surface_longitudinal_smoothing_outputs.items() if v}),
    }
    audit_payload = _build_xs_propagation_audit_payload(
        river_dir=river_dir,
        nodes=nodes,
        admitted_nodes=admitted_nodes,
        input_nodes=input_nodes,
        input_xs_nodes=input_xs_nodes,
        input_measured_xs_nodes=input_measured_xs_nodes,
        admitted_xs_nodes=admitted_xs_nodes,
        admitted_measured_xs_nodes=admitted_measured_xs_nodes,
        admitted_auth_nodes=admitted_auth_nodes,
        admitted_graph_nodes=admitted_graph_nodes,
        admitted_bank_nodes=admitted_bank_nodes,
        node_rejection_counts=node_rejection_counts,
        admitted_node_role_counts=admitted_node_role_counts,
        populated_mask=populated_mask,
        xs_participation_out=xs_participation_out,
        authoritative_participation_out=authoritative_participation_out,
        influence_class_out=influence_class_out,
        non_authoritative_support_mask=non_authoritative_support_mask,
        xs_admissibility_mask=xs_admissibility_mask,
        authoritative_lock_scope_out=authoritative_lock_scope_out,
        authoritative_lock_applied_out=authoritative_lock_applied_out,
        auth_mask=auth_mask,
        corridor=corridor,
        xs_warning=xs_warning,
        artifacts=audit_artifacts,
    )
    audit_path = _write_xs_propagation_audit(river_dir=river_dir, audit=audit_payload)
    outputs['river_xs_propagation_audit.json'] = str(audit_path)
    outputs.update({k: v for k, v in tendency_outputs.items() if v})
    outputs.update({k: v for k, v in xs_realism_outputs.items() if v})
    outputs.update({k: v for k, v in primary_surface_rebuild_outputs.items() if v})
    outputs.update({k: v for k, v in channel_surface_longitudinal_smoothing_outputs.items() if v})

    active_driver_table_path, active_driver_summary_path = _write_active_driver_receipts(river_dir=river_dir, nodes=nodes)
    outputs['river_active_driver_station_table'] = str(active_driver_table_path)
    outputs['river_active_driver_summary'] = str(active_driver_summary_path)

    render_mode_summary_path = river_dir / 'river_render_mode_summary.csv'
    pd.DataFrame([
        {
            'component_id': str(comp),
            'render_mode': str(component_render_modes.get(str(comp), 'full_scaffold')),
            'render_reason': str(component_render_mode_reasons.get(str(comp), 'unknown')),
            'active_roles': ','.join(role for role, _ in component_active_roles.get(str(comp), ROLE_ORDER)),
        }
        for comp in sorted(component_render_modes.keys(), key=str)
    ]).to_csv(render_mode_summary_path, index=False)
    render_mode_selector_summary_path = river_dir / 'river_render_mode_selector_summary.csv'
    selector_summary = pd.DataFrame(component_render_selector_rows)
    if not selector_summary.empty:
        selector_summary = selector_summary.sort_values(['component_id'], na_position='last')
    selector_summary.to_csv(render_mode_selector_summary_path, index=False)
    render_mode_selector_json_path = river_dir / 'river_render_mode_selector_summary.json'
    render_mode_selector_json_path.write_text(json.dumps({'components': selector_summary.to_dict(orient='records')}, indent=2), encoding='utf-8')

    render_mode_counts = {mode: int(sum(1 for v in component_render_modes.values() if v == mode)) for mode in sorted(set(component_render_modes.values()))}
    render_reason_counts = {reason: int(sum(1 for v in component_render_mode_reasons.values() if v == reason)) for reason in sorted(set(component_render_mode_reasons.values()))}
    fast_render_component_ids = [str(comp) for comp, mode in component_render_modes.items() if mode == 'authoritative_fast_path']
    science_effect_outputs, science_effect_summary = write_science_effect_summary(
        river_dir=river_dir,
        effectiveness_summary=effectiveness_summary,
        primary_surface_rebuild_summary=primary_surface_rebuild_summary,
        xs_realism_summary=xs_realism_summary,
        tendency_summary=tendency_summary,
        render_mode_counts=render_mode_counts,
        render_reason_counts=render_reason_counts,
        fast_render_component_ids=fast_render_component_ids,
        channel_surface_effect_summary=channel_surface_effect_summary,
        logger=active_logger,
    )

    contract_metrics = {
        'populated_cell_count': int(np.count_nonzero(np.isfinite(z_out))),
        'source_class_counts': {name: int(np.count_nonzero(src_out == code)) for name, code in SOURCE_CODES.items() if code > 0},
        'authoritative_cells': int(np.count_nonzero(src_out == SOURCE_CODES['authoritative_in_channel'])),
        'authoritative_backbone_cells': int(np.count_nonzero(src_out == SOURCE_CODES['authoritative_backbone'])),
        'bank_prior_cells': int(np.count_nonzero(src_out == SOURCE_CODES['bank_stage_prior'])),
        'channel_surface_longitudinal_smoothing_summary': channel_surface_longitudinal_smoothing_summary,
        'scaffold_input_node_count': int(input_nodes),
        'scaffold_input_xs_node_count': int(input_xs_nodes),
        'scaffold_input_measured_xs_node_count': int(input_measured_xs_nodes),
        'admitted_surface_control_node_count': int(len(admitted_nodes)),
        'admitted_surface_control_xs_node_count': int(admitted_xs_nodes),
        'admitted_surface_control_measured_xs_node_count': int(admitted_measured_xs_nodes),
        'admitted_surface_control_authoritative_node_count': int(admitted_auth_nodes),
        'final_xs_participation_cell_count': _bool_count(xs_participation_out > 0),
        'final_authoritative_participation_cell_count': _bool_count(authoritative_participation_out > 0),
        'final_mixed_authoritative_xs_cell_count': _bool_count((xs_participation_out > 0) & (authoritative_participation_out > 0)),
        'final_non_authoritative_support_cell_count': _bool_count(populated_mask & non_authoritative_support_mask),
        'final_xs_participation_in_non_authoritative_support_cells': _bool_count((xs_participation_out > 0) & populated_mask & non_authoritative_support_mask),
        'node_rejection_counts': node_rejection_counts,
        'admitted_node_role_counts': admitted_node_role_counts,
        'graph_hard_locked_cells': int(np.count_nonzero(mode_out == GRAPH_MODE_CODES['hard_locked'])),
        'graph_prior_driven_cells': int(np.count_nonzero(mode_out == GRAPH_MODE_CODES['prior_driven'])),
        'graph_regularization_driven_cells': int(np.count_nonzero(mode_out == GRAPH_MODE_CODES['regularization_driven'])),
        'graph_junction_constrained_cells': int(np.count_nonzero(junction_flag_out > 0)),
        'surface_junction_role_counts': {name: int(np.count_nonzero(junction_role_out == code)) for name, code in JUNCTION_ROLE_CODES.items() if code > 0},
        'surface_support_class_counts': {name: int(np.count_nonzero(support_class_out == code)) for name, code in SUPPORT_CLASS_CODES.items() if code > 0},
        'surface_uncertainty_counts': {name: int(np.count_nonzero(uncertainty_out == code)) for name, code in UNCERTAINTY_CLASS_CODES.items() if code > 0},
        'surface_hard_lock_cell_count': int(np.count_nonzero(hard_lock_out > 0)),
        'surface_junction_constrained_cell_count': int(np.count_nonzero(junction_flag_out > 0)),
        'surface_mean_unsupported_span_m': float(np.nanmean(unsupported_span_out)) if np.any(np.isfinite(unsupported_span_out)) else 0.0,
        'surface_unsupported_regime_counts': {name: int(np.count_nonzero(unsupported_regime_out == code)) for name, code in UNSUPPORTED_REGIME_CODES.items() if code > 0},
        'prediction_confidence_available': bool(prediction_confidence_summary.get('available', False)) if isinstance(prediction_confidence_summary, dict) else False,
        'prediction_admissible_cell_count': _bool_count(prediction_admissibility_out > 0),
        'prediction_low_support_caution_cell_count': _bool_count(prediction_low_support_caution_out > 0),
        'prediction_support_confidence_distribution': _distribution(prediction_support_confidence_out[np.isfinite(prediction_support_confidence_out)]),
        'prediction_measured_anchor_fraction_distribution': _distribution(prediction_measured_anchor_fraction_out[np.isfinite(prediction_measured_anchor_fraction_out)]),
        'prediction_structure_only_fraction_distribution': _distribution(prediction_structure_only_fraction_out[np.isfinite(prediction_structure_only_fraction_out)]),
        'longitudinal_tendency_available': bool(tendency_summary.get('available', False)) if isinstance(tendency_summary, dict) else False,
        'longitudinal_tendency_adjusted_node_count': int(tendency_summary.get('adjusted_node_count', 0)) if isinstance(tendency_summary, dict) else 0,
        'longitudinal_tendency_junction_targeted_node_count': int(tendency_summary.get('junction_targeted_node_count', 0)) if isinstance(tendency_summary, dict) else 0,
        'longitudinal_tendency_delta_abs_m': tendency_summary.get('delta_abs_m', {}) if isinstance(tendency_summary, dict) else {},
        'longitudinal_tendency_junction_weight_summary': tendency_summary.get('junction_weight_summary', {}) if isinstance(tendency_summary, dict) else {},
        'xs_realism_available': bool(xs_realism_summary.get('available', False)) if isinstance(xs_realism_summary, dict) else False,
        'xs_realism_adjusted_node_count': int(xs_realism_summary.get('adjusted_node_count', 0)) if isinstance(xs_realism_summary, dict) else 0,
        'xs_realism_delta_abs_m': xs_realism_summary.get('delta_abs_m', {}) if isinstance(xs_realism_summary, dict) else {},
        'xs_realism_blend_weight_summary': xs_realism_summary.get('blend_weight_summary', {}) if isinstance(xs_realism_summary, dict) else {},
        'primary_surface_rebuild_available': bool(primary_surface_rebuild_summary.get('available', False)) if isinstance(primary_surface_rebuild_summary, dict) else False,
        'primary_surface_rebuild_adjusted_node_count': int(primary_surface_rebuild_summary.get('adjusted_node_count', 0)) if isinstance(primary_surface_rebuild_summary, dict) else 0,
        'primary_surface_rebuild_delta_abs_m': primary_surface_rebuild_summary.get('delta_abs_m', {}) if isinstance(primary_surface_rebuild_summary, dict) else {},
        'effectiveness_available': bool(effectiveness_summary.get('available', False)) if isinstance(effectiveness_summary, dict) else False,
        'effectiveness_weak_support_station_count': int(effectiveness_summary.get('weak_support_station_count', 0)) if isinstance(effectiveness_summary, dict) else 0,
        'effectiveness_weak_support_rebuild_eligible_station_count': int(effectiveness_summary.get('weak_support_rebuild_eligible_station_count', 0)) if isinstance(effectiveness_summary, dict) else 0,
        'effectiveness_weak_support_changed_station_count': int(effectiveness_summary.get('weak_support_changed_station_count', 0)) if isinstance(effectiveness_summary, dict) else 0,
        'effectiveness_changed_node_count': int(effectiveness_summary.get('changed_node_count', 0)) if isinstance(effectiveness_summary, dict) else 0,
        'effectiveness_changed_channel_node_count': int(effectiveness_summary.get('changed_channel_node_count', 0)) if isinstance(effectiveness_summary, dict) else 0,
        'effectiveness_rebuild_suppression_reason_counts': effectiveness_summary.get('rebuild_suppression_reason_counts', {}) if isinstance(effectiveness_summary, dict) else {},
        'surface_p95_residual_to_candidate_z_m': float(np.nanpercentile(np.abs(residual_to_candidate_out[np.isfinite(residual_to_candidate_out)]), 95)) if np.any(np.isfinite(residual_to_candidate_out)) else 0.0,
        'component_render_mode_counts': render_mode_counts,
        'component_render_reason_counts': render_reason_counts,
        'fast_render_component_ids': fast_render_component_ids,
        'science_effect_available': bool(science_effect_summary.get('available', False)) if isinstance(science_effect_summary, dict) else False,
        'science_effect_engaged': bool(science_effect_summary.get('science_engaged', False)) if isinstance(science_effect_summary, dict) else False,
        'section_target_agreement_available': bool(section_target_summary.get('available', False)) if isinstance(section_target_summary, dict) else False,
        'section_target_agreement_node_count': int(section_target_summary.get('comparison_node_count', 0)) if isinstance(section_target_summary, dict) else 0,
        'section_target_agreement_abs_error_m': section_target_summary.get('abs_error_m', {}) if isinstance(section_target_summary, dict) else {},
        'section_target_agreement_by_component_support_class': section_target_summary.get('by_component_support_class', {}) if isinstance(section_target_summary, dict) else {},
        'section_target_agreement_reason': section_target_summary.get('reason') if isinstance(section_target_summary, dict) else None,
        'role_agreement_reason': role_agreement_summary.get('reason') if isinstance(role_agreement_summary, dict) else None,
        'authoritative_transition_weight_distribution': _distribution(authoritative_transition_weight_out[np.isfinite(authoritative_transition_weight_out)]),
        'authoritative_transition_nonzero_cell_count': _bool_count(np.isfinite(authoritative_transition_weight_out) & (authoritative_transition_weight_out > 0)),
        'xs_influence_disabled': bool(disable_xs_influence),
        'anchor_class_counts': {str(k): int(v) for k, v in nodes.get('target_anchor_class', pd.Series('missing', index=nodes.index)).astype(str).value_counts().to_dict().items()},
        'anchor_locks_core_count': int(np.count_nonzero(nodes.get('target_anchor_locks_core', pd.Series(False, index=nodes.index)).fillna(False).astype(bool).to_numpy(dtype=bool))),
        'anchor_blocks_rebuild_count': int(np.count_nonzero(nodes.get('target_anchor_blocks_rebuild', pd.Series(False, index=nodes.index)).fillna(False).astype(bool).to_numpy(dtype=bool))),
    }
    contract_artifacts = dict(base_artifacts)
    contract_artifacts.update({k: v for k, v in tendency_outputs.items() if v})
    contract_artifacts.update({k: v for k, v in xs_realism_outputs.items() if v})
    contract_artifacts.update({k: v for k, v in prediction_confidence_outputs.items() if v})
    contract_artifacts.update({k: v for k, v in primary_surface_rebuild_outputs.items() if v})
    contract_artifacts.update({k: v for k, v in effectiveness_outputs.items() if v})
    contract_artifacts.update({k: v for k, v in science_effect_outputs.items() if v})
    contract_artifacts.update({k: v for k, v in section_target_outputs.items() if v})
    contract_artifacts.update({k: v for k, v in role_agreement_outputs.items() if v})
    contract_artifacts['river_xs_propagation_audit'] = str(audit_path)
    contract_artifacts['river_render_mode_summary'] = str(render_mode_summary_path)
    contract_artifacts['river_render_mode_selector_summary'] = str(render_mode_selector_summary_path)
    contract_artifacts['river_render_mode_selector_summary_json'] = str(render_mode_selector_json_path)
    contract_artifacts.update({k: v for k, v in channel_surface_effect_outputs.items() if v})
    contract_artifacts.update({k: v for k, v in channel_surface_longitudinal_smoothing_outputs.items() if v})
    contract = {
        'schema_version': 2,
        'artifact_family': 'river_channel_surface',
        'notes': {
            'objective': 'Channel-fitted river channel-surface raster built from regular scaffold nodes in (s,n)-style coordinates with authoritative in-channel DEM assimilation.',
            'method': 'Project each corridor cell to the nearest thalweg segment to recover local station, center, and normal; render the solved scaffold node bed_z_m field across cross-stream eta bins at that projected station; in weak-support rebuild components, use a thalweg-dominant lateral shape so the reconciled centerline/backbone controls more of the channel width; then override with authoritative in-channel depth where explicit authoritative support exists.',
            'limitation': 'This is a structured rasterized surface product rendered from the scaffold node solution, not yet a full branch-junction mesh solver, but authoritative in-channel anchors are treated as hard interior controls.',
        },
        'source_class_codes': {k: int(v) for k, v in SOURCE_CODES.items()},
        'graph_solution_mode_codes': {k: int(v) for k, v in GRAPH_MODE_CODES.items()},
        'metrics': contract_metrics,
        'artifacts': contract_artifacts,
    }
    contract_path = river_dir / 'river_channel_surface_contract.json'
    contract_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    outputs['river_channel_surface_contract.json'] = str(contract_path)

    support_contract_artifacts = {
        'channel_surface_support_class': str(support_class_path),
        'channel_surface_solution_mode': str(mode_path),
        'channel_surface_uncertainty': str(uncertainty_path),
        'channel_surface_hard_lock': str(hard_lock_path),
        'channel_surface_junction_constrained': str(junction_flag_path),
        'channel_surface_junction_role': str(junction_role_path),
        'channel_surface_junction_adjustment': str(junction_adjustment_path),
        'channel_surface_junction_distance': str(junction_distance_path),
        'channel_surface_unsupported_span': str(unsupported_path),
        'channel_surface_unsupported_regime': str(unsupported_regime_path),
        'channel_surface_residual_to_candidate': str(residual_path),
        'channel_surface_xs_participation': str(xs_participation_path),
        'channel_surface_authoritative_participation': str(authoritative_participation_path),
        'channel_surface_influence_class': str(influence_class_path),
        'channel_surface_xs_admissibility_mask': str(xs_admissibility_path),
        **({'channel_surface_control_nodes': str(exported_control_nodes_path)} if exported_control_nodes_path is not None else {}),
        'river_xs_propagation_audit': str(audit_path),
        'graph_backbone_diagnostics_expected': str(river_dir / 'river_graph_backbone_diagnostics.gpkg'),
        'channel_surface_prediction_support_confidence': str(prediction_confidence_path),
        'channel_surface_measured_anchor_fraction': str(measured_anchor_fraction_path),
        'channel_surface_structure_only_fraction': str(structure_only_fraction_path),
        'channel_surface_low_support_caution': str(prediction_caution_path),
        'channel_surface_prediction_admissibility': str(prediction_admissibility_path),
        'channel_surface_authoritative_transition_weight': str(authoritative_transition_weight_path),
        **({k: v for k, v in prediction_confidence_outputs.items() if v}),
        **({k: v for k, v in effectiveness_outputs.items() if v}),
        **({k: v for k, v in science_effect_outputs.items() if v}),
        **({k: v for k, v in section_target_outputs.items() if v}),
        **({k: v for k, v in role_agreement_outputs.items() if v}),
        **({k: v for k, v in channel_surface_effect_outputs.items() if v}),
        **({k: v for k, v in channel_surface_longitudinal_smoothing_outputs.items() if v}),
    }
    support_contract = {
        'schema_version': 1,
        'artifact_family': 'river_support_uncertainty',
        'support_class_codes': {k: int(v) for k, v in SUPPORT_CLASS_CODES.items()},
        'solution_mode_codes': {k: int(v) for k, v in GRAPH_MODE_CODES.items()},
        'uncertainty_class_codes': {k: int(v) for k, v in UNCERTAINTY_CLASS_CODES.items()},
        'unsupported_regime_codes': {k: int(v) for k, v in UNSUPPORTED_REGIME_CODES.items()},
        'metrics': contract_metrics,
        'artifacts': support_contract_artifacts,
    }
    support_contract_path = river_dir / 'river_support_uncertainty_contract.json'
    support_contract_path.write_text(json.dumps(support_contract, indent=2), encoding='utf-8')
    outputs['river_support_uncertainty_contract.json'] = str(support_contract_path)

    final_outputs = dict(base_artifacts)
    final_outputs.update({k: v for k, v in tendency_outputs.items() if v})
    final_outputs.update({k: v for k, v in xs_realism_outputs.items() if v})
    final_outputs.update({k: v for k, v in prediction_confidence_outputs.items() if v})
    final_outputs.update({k: v for k, v in primary_surface_rebuild_outputs.items() if v})
    final_outputs.update({k: v for k, v in effectiveness_outputs.items() if v})
    final_outputs.update({k: v for k, v in science_effect_outputs.items() if v})
    final_outputs.update({k: v for k, v in section_target_outputs.items() if v})
    final_outputs.update({k: v for k, v in role_agreement_outputs.items() if v})
    final_outputs.update({k: v for k, v in channel_surface_effect_outputs.items() if v})
    final_outputs.update({k: v for k, v in channel_surface_longitudinal_smoothing_outputs.items() if v})
    final_outputs['channel_surface_authoritative_transition_summary'] = str(authoritative_transition_summary_path)
    final_outputs['channel_surface_authoritative_transition_weight'] = str(authoritative_transition_weight_path)
    final_outputs['river_render_mode_selector_summary'] = str(render_mode_selector_summary_path)
    final_outputs['river_render_mode_selector_summary_json'] = str(render_mode_selector_json_path)
    final_outputs['river_render_mode_summary'] = str(render_mode_summary_path)
    final_outputs.update(_channel_surface_output_aliases(
        xs_admissibility_path=xs_admissibility_path,
        auth_lock_scope_path=auth_lock_scope_path,
        auth_lock_applied_path=auth_lock_applied_path,
        control_nodes_path=exported_control_nodes_path,
        audit_path=audit_path,
        contract_path=contract_path,
        support_contract_path=support_contract_path,
    ))
    if xs_warning:
        (logger or log).warning('[RIVER][FRAME] XS propagation audit: scaffold xs nodes were built but no populated non-authoritative-support cells show downstream XS participation')
    xs_expected_gap = int(audit_payload['metrics']['final_xs_expected_but_missing_cell_count'])
    (logger or log).info('[RIVER][FRAME] Channel surface built: populated=%d auth=%d xs=%d xs_participation=%d non_authoritative_xs=%d xs_expected_gap=%d auth_lock_scope=%d auth_lock_applied=%d weak_support=%d eligible=%d changed=%d', int(np.count_nonzero(np.isfinite(z_out))), int(np.count_nonzero(src_out == SOURCE_CODES['authoritative_in_channel'])), int(np.count_nonzero(src_out == SOURCE_CODES['xs_profile_resampled'])), _bool_count(xs_participation_out > 0), _bool_count((xs_participation_out > 0) & populated_mask & non_authoritative_support_mask), xs_expected_gap, _bool_count(authoritative_lock_scope_out > 0), _bool_count(authoritative_lock_applied_out > 0), int(effectiveness_summary.get('weak_support_station_count', 0)) if isinstance(effectiveness_summary, dict) else 0, int(effectiveness_summary.get('weak_support_rebuild_eligible_station_count', 0)) if isinstance(effectiveness_summary, dict) else 0, int(effectiveness_summary.get('weak_support_changed_station_count', 0)) if isinstance(effectiveness_summary, dict) else 0)
    (logger or log).info('[RIVER][FRAME] Science summary: engaged=%s changed_channel_nodes=%d rebuild_channel_core=%d fast_render=%d', bool(science_effect_summary.get('science_engaged', False)) if isinstance(science_effect_summary, dict) else False, int(science_effect_summary.get('changed_channel_node_count', 0)) if isinstance(science_effect_summary, dict) else 0, int(science_effect_summary.get('primary_surface_rebuild_changed_channel_core_node_count', 0)) if isinstance(science_effect_summary, dict) else 0, int(science_effect_summary.get('fast_render_component_count', 0)) if isinstance(science_effect_summary, dict) else 0)
    return final_outputs
