from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _interp_series(stations: np.ndarray, src_s: np.ndarray, src_v: np.ndarray) -> np.ndarray:
    out = np.full(stations.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(src_s) & np.isfinite(src_v)
    if np.count_nonzero(valid) == 0:
        return out
    s = np.asarray(src_s[valid], dtype=float)
    v = np.asarray(src_v[valid], dtype=float)
    order = np.argsort(s)
    s = s[order]
    v = v[order]
    if s.size == 1:
        out[:] = np.float32(v[0])
        return out
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



def _smooth_component_backbone(stations: np.ndarray, values: np.ndarray, locked: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    n = out.size
    if n == 0:
        return out.astype(np.float32)
    valid = np.isfinite(out)
    if np.count_nonzero(valid) < 2:
        return out.astype(np.float32)
    window = 3 if n < 7 else 5
    half = window // 2
    smoothed = out.copy()
    for i in range(n):
        if bool(locked[i]) or not np.isfinite(out[i]):
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = out[lo:hi]
        local = local[np.isfinite(local)]
        if local.size >= 2:
            smoothed[i] = float(np.nanmedian(local))
    return smoothed.astype(np.float32)





def _build_component_station_candidates(sub: pd.DataFrame) -> pd.DataFrame:
    """Assemble explicit absolute-bed candidates for one component before solving."""
    out = sub.sort_values('station_m').copy()
    out['station_m'] = pd.to_numeric(out.get('station_m', np.nan), errors='coerce')
    out['authoritative_bed_z_m'] = pd.to_numeric(out.get('authoritative_bed_z_m', np.nan), errors='coerce')
    out['authoritative_hard_bed_z_m'] = pd.to_numeric(out.get('authoritative_hard_bed_z_m', out.get('authoritative_bed_z_m', np.nan)), errors='coerce')
    out['authoritative_backbone_z_m'] = pd.to_numeric(out.get('authoritative_backbone_z_m', np.nan), errors='coerce')
    out['backbone_z_m'] = pd.to_numeric(out.get('backbone_z_m', np.nan), errors='coerce')
    out['resolved_stage_control_z_m'] = pd.to_numeric(out.get('resolved_stage_control_z_m', np.nan), errors='coerce')
    out['authoritative_station_support_strength'] = pd.to_numeric(out.get('authoritative_station_support_strength', np.nan), errors='coerce')
    out['backbone_mode'] = out.get('backbone_mode', pd.Series('missing', index=out.index)).fillna('missing').astype(str)
    out['channel_support_class'] = out.get('channel_support_class', pd.Series('unsupported', index=out.index)).fillna('unsupported').astype(str)

    auth = out['authoritative_hard_bed_z_m'].to_numpy(dtype=float)
    auth_backbone = out['authoritative_backbone_z_m'].to_numpy(dtype=float)
    contract = out['backbone_z_m'].to_numpy(dtype=float)
    stage = out['resolved_stage_control_z_m'].to_numpy(dtype=float)
    support_strength = out['authoritative_station_support_strength'].to_numpy(dtype=float)
    support_class = out['channel_support_class'].astype(str).to_numpy()
    backbone_mode = out['backbone_mode'].astype(str).to_numpy()

    hard_lock = np.isfinite(auth) | np.isfinite(auth_backbone)
    hard_value = np.where(np.isfinite(auth), auth, auth_backbone)
    preferred = np.where(np.isfinite(hard_value), hard_value, np.where(np.isfinite(contract), contract, stage))

    support_mode = np.full(len(out), 'unsupported', dtype=object)
    support_mode[hard_lock] = 'authoritative_locked'

    finite_contract = (~hard_lock) & np.isfinite(contract)
    authoritative_contract = finite_contract & np.isin(backbone_mode, ['authoritative_in_channel', 'authoritative_backbone'])
    support_mode[authoritative_contract] = 'authoritative_backbone'

    anchored = finite_contract & (~authoritative_contract) & (support_strength > 0.0)
    support_mode[anchored] = 'anchored_interpolated'

    resolved = finite_contract & (~authoritative_contract) & (~anchored)
    support_mode[resolved] = 'resolved_backbone'

    support_mode[(~hard_lock) & (~np.isfinite(contract)) & np.isfinite(stage)] = 'stage_controlled'
    support_mode[(support_mode == 'unsupported') & np.isin(support_class, ['xs_only', 'xs_residual_only'])] = 'xs_residual_only'

    candidate_source = np.where(
        np.isfinite(auth),
        'authoritative_in_channel',
        np.where(
            np.isfinite(auth_backbone),
            'authoritative_backbone',
            np.where(
                np.isfinite(contract),
                backbone_mode,
                np.where(np.isfinite(stage), 'bank_stage_prior', 'missing'),
            ),
        ),
    )

    out['solver_support_class'] = support_mode
    out['solver_backbone_candidate_z_m'] = preferred
    out['solver_hard_lock'] = hard_lock.astype(bool)
    out['solver_candidate_source'] = candidate_source
    return out


def _support_data_weight(mode: str) -> float:
    mode = str(mode)
    if mode == 'authoritative_locked':
        return 0.0
    if mode == 'authoritative_backbone':
        return 200.0
    if mode == 'anchored_interpolated':
        return 40.0
    if mode == 'resolved_backbone':
        return 12.0
    if mode == 'stage_controlled':
        return 6.0
    return 0.0



def _support_edge_weight(mode_left: str, mode_right: str) -> float:
    modes = {str(mode_left), str(mode_right)}
    if 'authoritative_locked' in modes:
        return 0.0
    if modes <= {'unsupported', 'xs_residual_only'}:
        return 0.02
    if 'stage_controlled' in modes:
        return 0.015
    if 'resolved_backbone' in modes:
        return 0.01
    if 'anchored_interpolated' in modes or 'authoritative_backbone' in modes:
        return 0.004
    return 0.005



def _support_curvature_weight(mode_center: str) -> float:
    mode = str(mode_center)
    if mode == 'authoritative_locked':
        return 0.0
    if mode in ('unsupported', 'xs_residual_only'):
        return 1.5
    if mode == 'stage_controlled':
        return 1.0
    if mode == 'resolved_backbone':
        return 0.5
    if mode == 'anchored_interpolated':
        return 0.2
    if mode == 'authoritative_backbone':
        return 0.05
    return 0.1


def _support_physical_guard_factor(mode_left: str, mode_right: str) -> float:
    modes = {str(mode_left), str(mode_right)}
    if 'authoritative_locked' in modes:
        return 0.0
    if modes <= {'unsupported', 'xs_residual_only'}:
        return 1.75
    if 'stage_controlled' in modes:
        return 1.3
    if 'resolved_backbone' in modes:
        return 1.0
    if 'anchored_interpolated' in modes:
        return 0.7
    if 'authoritative_backbone' in modes:
        return 0.25
    return 0.5


def _support_slope_cap(mode_left: str, mode_right: str) -> float:
    modes = {str(mode_left), str(mode_right)}
    if 'authoritative_locked' in modes or 'authoritative_backbone' in modes:
        return 0.25
    if 'anchored_interpolated' in modes:
        return 0.18
    if 'resolved_backbone' in modes:
        return 0.14
    if 'stage_controlled' in modes:
        return 0.12
    if modes <= {'unsupported', 'xs_residual_only'}:
        return 0.08
    return 0.10


def _classify_unsupported_regime(
    support_class: str,
    unsupported_span_m: float,
    anchor_spacing_m: float,
    junction_constrained: bool,
    junction_distance_m: float,
    topology_confidence: float,
) -> str:
    support = str(support_class or 'unsupported')
    span = float(unsupported_span_m) if np.isfinite(unsupported_span_m) else 0.0
    anchor = float(anchor_spacing_m) if np.isfinite(anchor_spacing_m) else float('nan')
    jdist = float(junction_distance_m) if np.isfinite(junction_distance_m) else float('nan')
    topo = float(topology_confidence) if np.isfinite(topology_confidence) else 0.0
    if support in ('authoritative_locked', 'authoritative_backbone', 'anchored_interpolated'):
        return 'supported'
    if bool(junction_constrained) and topo >= 0.5 and np.isfinite(jdist) and jdist <= max(120.0, 0.35 * max(span, 1.0)):
        return 'junction_dominated_unsupported'
    short_bridge = span <= 120.0 or (np.isfinite(anchor) and anchor <= 120.0 and span <= 180.0)
    if short_bridge:
        return 'short_gap_bridge'
    if span <= 200.0:
        return 'medium_gap_regularized'
    return 'long_gap_stiffened'



def _unsupported_span_lengths(stations: np.ndarray, support_mode: np.ndarray) -> np.ndarray:
    n = int(len(stations))
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    weak = np.isin(np.asarray(support_mode, dtype=object).astype(str), ['unsupported', 'xs_residual_only'])
    run_start = None
    for i in range(n + 1):
        is_weak = bool(i < n and weak[i])
        if is_weak and run_start is None:
            run_start = i
        elif (not is_weak) and run_start is not None:
            run_end = i - 1
            st = stations[run_start:run_end + 1]
            if np.count_nonzero(np.isfinite(st)) >= 2:
                span = float(np.nanmax(st) - np.nanmin(st))
            elif run_end + 1 < n and np.isfinite(stations[run_start]) and np.isfinite(stations[run_end + 1]):
                span = float(abs(stations[run_end + 1] - stations[run_start]))
            else:
                span = 0.0
            out[run_start:run_end + 1] = max(span, 0.0)
            run_start = None
    return out


def _unsupported_span_factors(stations: np.ndarray, support_mode: np.ndarray) -> np.ndarray:
    spans = _unsupported_span_lengths(stations, support_mode)
    return 1.0 + np.minimum(np.maximum(spans, 0.0) / 400.0, 2.0)


def _estimate_anchor_spacing_for_solver(stations: np.ndarray, preferred: np.ndarray, locked: np.ndarray) -> float:
    stations = np.asarray(stations, dtype=float)
    preferred = np.asarray(preferred, dtype=float)
    locked = np.asarray(locked, dtype=bool)
    anchor_stations = stations[(locked | np.isfinite(preferred)) & np.isfinite(stations)]
    if anchor_stations.size >= 2:
        diffs = np.diff(np.sort(anchor_stations))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.nanmedian(diffs))
    if np.count_nonzero(np.isfinite(stations)) >= 2:
        diffs = np.diff(np.sort(stations[np.isfinite(stations)]))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.nanmedian(diffs))
    return float('nan')


def _regime_behavior_factors(regime: str) -> dict[str, float]:
    regime = str(regime or 'supported')
    if regime == 'short_gap_bridge':
        return {'edge': 0.85, 'curve': 0.85, 'phys': 0.8, 'centering': 0.75, 'slope_cap_mult': 1.15, 'junction': 0.9}
    if regime == 'medium_gap_regularized':
        return {'edge': 1.15, 'curve': 1.2, 'phys': 1.1, 'centering': 1.0, 'slope_cap_mult': 0.95, 'junction': 1.0}
    if regime == 'long_gap_stiffened':
        return {'edge': 1.5, 'curve': 1.8, 'phys': 1.5, 'centering': 1.2, 'slope_cap_mult': 0.8, 'junction': 1.05}
    if regime == 'junction_dominated_unsupported':
        return {'edge': 1.25, 'curve': 1.1, 'phys': 1.2, 'centering': 1.0, 'slope_cap_mult': 0.9, 'junction': 1.35}
    return {'edge': 1.0, 'curve': 1.0, 'phys': 1.0, 'centering': 1.0, 'slope_cap_mult': 1.0, 'junction': 1.0}


def _classify_solver_regimes(
    stations: np.ndarray,
    support_mode: np.ndarray,
    anchor_spacing_m: float,
    *,
    near_junction: np.ndarray | None = None,
    junction_distance: np.ndarray | None = None,
    topology_confidence: float = 0.0,
) -> np.ndarray:
    stations = np.asarray(stations, dtype=float)
    support_mode = np.asarray(support_mode, dtype=object).astype(str)
    spans = _unsupported_span_lengths(stations, support_mode)
    if near_junction is None:
        near_junction = np.zeros(len(stations), dtype=bool)
    if junction_distance is None:
        junction_distance = np.full(len(stations), np.nan, dtype=float)
    return np.asarray([
        _classify_unsupported_regime(sm, spans[i], anchor_spacing_m, bool(near_junction[i]), float(junction_distance[i]) if np.isfinite(junction_distance[i]) else np.nan, topology_confidence)
        for i, sm in enumerate(support_mode)
    ], dtype=object)


def _regularized_component_solve(
    stations: np.ndarray,
    preferred: np.ndarray,
    locked: np.ndarray,
    support_mode: np.ndarray,
    extra_targets: list[dict[str, float]] | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    n = int(len(stations))
    if n == 0:
        return np.asarray([], dtype=np.float32)
    init = _interp_series(stations, stations, preferred).astype(float)
    explicit = np.isfinite(preferred)
    init[explicit] = preferred[explicit]
    if np.count_nonzero(np.isfinite(init)) == 0:
        return init.astype(np.float32)
    z_locked = np.where(locked & np.isfinite(preferred), preferred, np.nan)
    unknown = (~locked) & np.isfinite(init)
    if not np.any(unknown):
        out = init.copy()
        out[locked & np.isfinite(z_locked)] = z_locked[locked & np.isfinite(z_locked)]
        return out.astype(np.float32)

    idx = np.where(unknown)[0]
    pos = {int(j): k for k, j in enumerate(idx)}
    m = len(idx)
    Q = np.zeros((m, m), dtype=float)
    b = np.zeros(m, dtype=float)
    slope_guard_sum = np.zeros(n, dtype=float)
    adverse_step_sum = np.zeros(n, dtype=float)
    physical_guard_sum = np.zeros(n, dtype=float)
    unsupported_factor = _unsupported_span_factors(stations, support_mode)
    anchor_spacing = _estimate_anchor_spacing_for_solver(stations, preferred, locked)
    regime = _classify_solver_regimes(stations, support_mode, anchor_spacing)

    def add_row(coeffs: dict[int, float], target: float, weight: float, diag_name: str | None = None) -> None:
        if not np.isfinite(weight) or weight <= 0.0:
            return
        filt = {int(k): float(v) for k, v in coeffs.items() if int(k) in pos and np.isfinite(v)}
        if not filt:
            return
        target_eff = float(target)
        for k, v in coeffs.items():
            if int(k) not in pos and np.isfinite(v):
                lock_val = z_locked[int(k)] if int(k) < len(z_locked) else np.nan
                if np.isfinite(lock_val):
                    target_eff -= float(v) * float(lock_val)
        cols = [pos[int(k)] for k in filt]
        vals = np.asarray([filt[int(k)] for k in filt], dtype=float)
        if diag_name is not None:
            for k in filt:
                if diag_name == 'slope':
                    slope_guard_sum[int(k)] += weight
                elif diag_name == 'adverse':
                    adverse_step_sum[int(k)] += weight
                elif diag_name == 'physical':
                    physical_guard_sum[int(k)] += weight
        for a, ca in enumerate(cols):
            b[ca] += weight * vals[a] * target_eff
            for cb, vb in zip(cols, vals):
                Q[ca, cb] += weight * vals[a] * vb

    for i in range(n):
        if not unknown[i] or not np.isfinite(preferred[i]):
            continue
        add_row({i: 1.0}, float(preferred[i]), _support_data_weight(support_mode[i]))

    for i in range(n - 1):
        if not (np.isfinite(init[i]) and np.isfinite(init[i + 1])):
            continue
        ds = max(abs(float(stations[i + 1] - stations[i])), 1e-6)
        rf = _regime_behavior_factors(regime[i if regime.size else 0])
        rf2 = _regime_behavior_factors(regime[i + 1 if regime.size else 0])
        reg_edge = max(rf['edge'], rf2['edge'])
        reg_phys = max(rf['phys'], rf2['phys'])
        reg_slope = min(rf['slope_cap_mult'], rf2['slope_cap_mult'])
        w = (_support_edge_weight(support_mode[i], support_mode[i + 1]) * reg_edge) / ds
        add_row({i: -1.0, i + 1: 1.0}, 0.0, w)
        phys = _support_physical_guard_factor(support_mode[i], support_mode[i + 1]) * max(unsupported_factor[i], unsupported_factor[i + 1]) * reg_phys
        if phys > 0.0:
            slope_cap = _support_slope_cap(support_mode[i], support_mode[i + 1]) * reg_slope
            target_dz = float(np.clip(init[i + 1] - init[i], -slope_cap * ds, slope_cap * ds))
            add_row({i: -1.0, i + 1: 1.0}, target_dz, (0.015 / ds) * phys, diag_name='slope')
            if abs(init[i + 1] - init[i]) > slope_cap * ds:
                add_row({i: -1.0, i + 1: 1.0}, target_dz, (0.03 / ds) * phys, diag_name='adverse')

    for i in range(1, n - 1):
        if not (np.isfinite(init[i - 1]) and np.isfinite(init[i]) and np.isfinite(init[i + 1])):
            continue
        rf = _regime_behavior_factors(regime[i if regime.size else 0])
        w = _support_curvature_weight(support_mode[i]) * unsupported_factor[i] * rf['curve']
        add_row({i - 1: 1.0, i: -2.0, i + 1: 1.0}, 0.0, w)
        phys_c = 0.01 * unsupported_factor[i] * _support_physical_guard_factor(support_mode[i - 1], support_mode[i + 1]) * rf['phys']
        add_row({i - 1: 1.0, i: -2.0, i + 1: 1.0}, 0.0, phys_c, diag_name='physical')

    # weak centering on interpolated initialization to avoid drift in long unsupported tails
    for i in np.where(unknown & np.isfinite(init))[0]:
        rf = _regime_behavior_factors(regime[i if regime.size else 0])
        add_row({int(i): 1.0}, float(init[i]), 0.01 * rf['centering'])

    if extra_targets:
        for item in extra_targets:
            try:
                idx_target = int(item.get("idx", -1))
            except Exception:
                continue
            if idx_target < 0 or idx_target >= n or locked[idx_target]:
                continue
            target = float(item.get("target", np.nan))
            weight = float(item.get("weight", 0.0))
            if not np.isfinite(target) or not np.isfinite(weight) or weight <= 0.0:
                continue
            add_row({idx_target: 1.0}, target, weight)

    if Q.size == 0:
        out = init.copy()
    else:
        Q.flat[:: m + 1] += 1e-8
        rhs = b.copy()
        try:
            sol = np.linalg.solve(Q, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(Q, rhs, rcond=None)
        out = init.copy()
        out[idx] = sol
    if np.any(locked & np.isfinite(z_locked)):
        out[locked & np.isfinite(z_locked)] = z_locked[locked & np.isfinite(z_locked)]
    return out.astype(np.float32)



def _solve_component_backbone(
    sub: pd.DataFrame,
    extra_targets: list[dict[str, float]] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    cand = _build_component_station_candidates(sub)
    stations = cand['station_m'].to_numpy(dtype=float)
    preferred = cand['solver_backbone_candidate_z_m'].to_numpy(dtype=float)
    locked = cand['solver_hard_lock'].to_numpy(dtype=bool)
    support_mode = cand['solver_support_class'].astype(str).to_numpy()

    solved = _regularized_component_solve(stations, preferred, locked, support_mode, extra_targets=extra_targets).astype(float)
    solved = _smooth_component_backbone(stations, solved, locked).astype(float)
    if np.any(locked & np.isfinite(preferred)):
        solved[locked & np.isfinite(preferred)] = preferred[locked & np.isfinite(preferred)]
    if np.count_nonzero(np.isfinite(solved)) >= 2:
        finite_idx = np.where(np.isfinite(solved))[0]
        for k in range(1, len(finite_idx)):
            i0 = finite_idx[k-1]
            i1 = finite_idx[k]
            ds = max(abs(float(stations[i1] - stations[i0])), 1e-6)
            dz = float(solved[i1] - solved[i0])
            slope = dz / ds
            if abs(slope) > 0.25:
                solved[i1] = solved[i0] + np.sign(dz) * 0.25 * ds
        if np.any(locked & np.isfinite(preferred)):
            solved[locked & np.isfinite(preferred)] = preferred[locked & np.isfinite(preferred)]
    metrics = {
        'component_hard_lock_count': int(np.count_nonzero(locked)),
        'component_stage_controlled_count': int(np.count_nonzero(support_mode == 'stage_controlled')),
        'component_anchored_count': int(np.count_nonzero(np.isin(support_mode, ['anchored_interpolated', 'authoritative_backbone']))),
        'component_unsupported_count': int(np.count_nonzero(support_mode == 'unsupported')),
    }
    return solved.astype(np.float32), metrics



def _component_anchor_spacing(sub: pd.DataFrame) -> float:
    stations = pd.to_numeric(sub.get("station_m", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(dtype=float)
    auth = pd.to_numeric(sub.get("authoritative_bed_z_m", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(dtype=float)
    auth_backbone = pd.to_numeric(sub.get("authoritative_backbone_z_m", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(dtype=float)
    contract = pd.to_numeric(sub.get("backbone_z_m", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(dtype=float)
    backbone_mode = sub.get("backbone_mode", pd.Series("missing", index=sub.index)).fillna("missing").astype(str).to_numpy()
    anchor_mask = np.isfinite(auth) | np.isfinite(auth_backbone) | (np.isfinite(contract) & np.isin(backbone_mode, ["authoritative_in_channel", "authoritative_backbone"]))
    anchor_stations = stations[anchor_mask & np.isfinite(stations)]
    if anchor_stations.size >= 2:
        diffs = np.diff(np.sort(anchor_stations))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.nanmedian(diffs))
    if np.count_nonzero(np.isfinite(stations)) >= 2:
        diffs = np.diff(np.sort(stations[np.isfinite(stations)]))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.nanmedian(diffs))
    return float("nan")


def _component_topology_confidence(topo: dict[str, Any]) -> float:
    score = 0.0
    if topo.get("downstream_component_id"):
        score += 1.0
    if topo.get("upstream_component_ids"):
        score += 1.0
    if np.isfinite(float(topo.get("mainstem_rank", np.nan))):
        score += 1.0
    if np.isfinite(float(topo.get("network_order", np.nan))):
        score += 1.0
    if np.isfinite(float(topo.get("distance_to_mouth_m", np.nan))):
        score += 1.0
    return float(score / 5.0)


def _junction_constraint_params(
    *,
    group: dict[str, Any],
    comp: str,
    loc: int,
    records: list[dict[str, Any]],
    dominant: str,
    has_clear_dominance: bool,
    component_stats: dict[str, dict[str, float]],
    component_subs: dict[str, pd.DataFrame],
) -> tuple[float, float, float]:
    stats = component_stats.get(comp, {})
    support_mode = str(stats.get("endpoint_support_mode", {}).get(int(loc), "unsupported"))
    anchor_spacing = float(stats.get("anchor_spacing_m", np.nan))
    topology_conf = float(stats.get("topology_confidence", 0.0))
    group_topology = str(group.get("topology_source", "geometric_fallback"))

    if comp == dominant:
        return 0.0, 0.0, 0.0

    base_weight = 52.0
    if group_topology == "explicit_junction_id":
        base_weight *= 1.25
    else:
        base_weight *= 0.9
    base_weight *= (0.8 + 0.4 * max(0.0, min(1.0, topology_conf)))

    if support_mode == "authoritative_backbone":
        base_weight *= 0.45
    elif support_mode == "anchored_interpolated":
        base_weight *= 0.75
    elif support_mode == "resolved_backbone":
        base_weight *= 1.0
    elif support_mode == "stage_controlled":
        base_weight *= 1.15
    elif support_mode in ("unsupported", "xs_residual_only"):
        base_weight *= 1.3

    pull = 0.8 if has_clear_dominance else 0.55
    if support_mode == "authoritative_backbone":
        pull *= 0.5
    elif support_mode == "anchored_interpolated":
        pull *= 0.75
    elif support_mode == "stage_controlled":
        pull *= 1.05
    elif support_mode in ("unsupported", "xs_residual_only"):
        pull *= 1.15
    pull = float(min(max(pull, 0.2), 0.95))

    sub = component_subs.get(comp)
    if sub is None or sub.empty:
        return base_weight, pull, 1.0
    stations = pd.to_numeric(sub.get("station_m", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(dtype=float)
    if 0 <= loc < stations.size and np.isfinite(stations[loc]):
        local_spacing = float(anchor_spacing) if np.isfinite(anchor_spacing) and anchor_spacing > 0 else float("nan")
        if not np.isfinite(local_spacing):
            diffs = np.diff(np.sort(stations[np.isfinite(stations)]))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            local_spacing = float(np.nanmedian(diffs)) if diffs.size else 1.0
        dist_scale = max(local_spacing * 1.25, 1.0)
    else:
        dist_scale = 1.0
    return float(base_weight), float(pull), float(dist_scale)


def _component_topology_records(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    recs: dict[str, dict[str, Any]] = {}
    for comp, sub in frame.groupby("component_id", sort=False):
        sub = sub.sort_values("station_m").copy()
        downstream = next((str(v) for v in sub.get("downstream_component_id", pd.Series([pd.NA]*len(sub))).tolist() if pd.notna(v) and str(v).strip()), None)
        upstream_vals = set()
        if "upstream_component_ids" in sub.columns:
            for raw in sub["upstream_component_ids"].tolist():
                if pd.isna(raw):
                    continue
                for part in str(raw).split(","):
                    part = part.strip()
                    if part:
                        upstream_vals.add(part)
        mainstem_rank = pd.to_numeric(sub.get("mainstem_rank", pd.Series(np.nan, index=sub.index)), errors="coerce")
        network_order = pd.to_numeric(sub.get("network_order", pd.Series(np.nan, index=sub.index)), errors="coerce")
        distance_to_mouth = pd.to_numeric(sub.get("distance_to_mouth_m", pd.Series(np.nan, index=sub.index)), errors="coerce")
        recs[str(comp)] = {
            "downstream_component_id": downstream,
            "upstream_component_ids": sorted(upstream_vals),
            "mainstem_rank": float(np.nanmedian(mainstem_rank)) if np.any(np.isfinite(mainstem_rank)) else np.nan,
            "network_order": float(np.nanmedian(network_order)) if np.any(np.isfinite(network_order)) else np.nan,
            "distance_to_mouth_m": float(np.nanmin(distance_to_mouth)) if np.any(np.isfinite(distance_to_mouth)) else np.nan,
        }
    return recs


def _endpoint_records_for_component(frame: pd.DataFrame, component_fill: dict[str, np.ndarray], comp: str) -> list[dict[str, Any]]:
    sub = frame.loc[frame["component_id"].astype(str) == str(comp)].sort_values("station_m").copy()
    arr = np.asarray(component_fill.get(str(comp), np.asarray([], dtype=np.float32)), dtype=float)
    if arr.size != len(sub) or arr.size == 0:
        return []
    backbone_mode = sub.get("backbone_mode", pd.Series("missing", index=sub.index)).fillna("missing").astype(str).to_numpy()
    auth_strength = pd.to_numeric(sub.get("authoritative_station_support_strength", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(dtype=float)
    recs = []
    for endpoint_name, idx in (("start", 0), ("end", len(sub) - 1)):
        geom = sub.geometry.iloc[idx]
        if geom is None or idx >= arr.size or not np.isfinite(arr[idx]):
            continue
        recs.append({
            "component_id": str(comp),
            "endpoint": endpoint_name,
            "idx": int(idx),
            "x": float(geom.x),
            "y": float(geom.y),
            "z": float(arr[idx]),
            "backbone_mode": str(backbone_mode[idx]) if idx < len(backbone_mode) else "missing",
            "auth_strength": float(auth_strength[idx]) if idx < len(auth_strength) and np.isfinite(auth_strength[idx]) else 0.0,
        })
    return recs


def _junction_groups_from_explicit_topology(frame: pd.DataFrame, component_fill: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    if "junction_id" not in frame.columns or frame.empty or frame.geometry is None:
        return [], set()
    topo_groups: list[dict[str, Any]] = []
    covered_endpoints: set[tuple[str, int]] = set()
    work = frame.copy()
    work["junction_id"] = work["junction_id"].astype(object)
    valid = work["junction_id"].notna() & work["junction_id"].astype(str).str.strip().ne("")
    if not bool(valid.any()):
        return [], covered_endpoints
    for junction_id, sub in work.loc[valid].groupby("junction_id", sort=False):
        records = []
        local_components = []
        for comp in sub["component_id"].astype(str).dropna().unique().tolist():
            comp_records = _endpoint_records_for_component(work, component_fill, comp)
            if not comp_records:
                continue
            # Keep only endpoints that are actually tagged with this junction when possible.
            comp_sub = sub.loc[sub["component_id"].astype(str) == str(comp)].sort_values("station_m").copy()
            valid_positions = set()
            full_sub = work.loc[work["component_id"].astype(str) == str(comp)].sort_values("station_m").copy()
            for ridx in comp_sub.index:
                try:
                    valid_positions.add(int(full_sub.index.get_loc(ridx)))
                except KeyError:
                    continue
            selected = [r for r in comp_records if r["idx"] in valid_positions] or comp_records
            if selected:
                records.extend(selected[:1])
                local_components.append(str(comp))
        if len(set(local_components)) >= 2 and len(records) >= 2:
            topo_groups.append({"junction_id": str(junction_id), "records": records, "topology_source": "explicit_junction_id"})
            covered_endpoints.update((str(r["component_id"]), int(r["idx"])) for r in records)
    return topo_groups, covered_endpoints


def _build_junction_groups(frame: pd.DataFrame, component_fill: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], float]:
    if frame.empty or not component_fill or frame.geometry is None:
        return [], float('nan')
    explicit_groups, covered_endpoints = _junction_groups_from_explicit_topology(frame, component_fill)

    endpoints = []
    for comp in frame['component_id'].astype(str).dropna().unique().tolist():
        for rec in _endpoint_records_for_component(frame, component_fill, str(comp)):
            endpoint_key = (str(rec["component_id"]), int(rec["idx"]))
            if endpoint_key in covered_endpoints:
                continue
            endpoints.append(rec)
    if len(endpoints) < 2:
        return explicit_groups, float('nan')
    endpoints_df = pd.DataFrame.from_records(endpoints)
    xy = endpoints_df[['x', 'y']].to_numpy(dtype=float)
    dists = []
    for i in range(len(xy)):
        for j in range(i+1, len(xy)):
            if endpoints_df.iloc[i]['component_id'] == endpoints_df.iloc[j]['component_id']:
                continue
            d = float(np.hypot(xy[i,0]-xy[j,0], xy[i,1]-xy[j,1]))
            if np.isfinite(d) and d > 0:
                dists.append(d)
    if not dists:
        return explicit_groups, float('nan')
    threshold = float(np.nanpercentile(np.asarray(dists, dtype=float), 25)) * 1.25
    if not np.isfinite(threshold) or threshold <= 0.0:
        return explicit_groups, float('nan')
    adj = {i:set() for i in range(len(endpoints_df))}
    for i in range(len(endpoints_df)):
        for j in range(i+1, len(endpoints_df)):
            if endpoints_df.iloc[i]['component_id'] == endpoints_df.iloc[j]['component_id']:
                continue
            d = float(np.hypot(xy[i,0]-xy[j,0], xy[i,1]-xy[j,1]))
            if np.isfinite(d) and d <= threshold:
                adj[i].add(j); adj[j].add(i)
    groups=[]; seen=set()
    for i in range(len(endpoints_df)):
        if i in seen:
            continue
        stack=[i]; members=[]
        while stack:
            k=stack.pop()
            if k in seen:
                continue
            seen.add(k); members.append(k); stack.extend(sorted(adj.get(k,())))
        if len(members) >= 2:
            records = [endpoints_df.iloc[k].to_dict() for k in sorted(members)]
            groups.append({'junction_id': f'junction_{len(explicit_groups)+len(groups)+1}', 'records': records, 'topology_source': 'geometric_fallback'})
    return explicit_groups + groups, threshold


def _global_graph_backbone_solve(
    frame: pd.DataFrame,
    component_stats: dict[str, dict[str, float]],
    groups: list[dict[str, Any]],
    component_fill_init: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int, int, pd.DataFrame]:
    component_subs: dict[str, pd.DataFrame] = {}
    cand_parts: list[pd.DataFrame] = []
    for comp, sub in frame.groupby("component_id", sort=False):
        comp_key = str(comp)
        ordered = sub.sort_values("station_m").copy()
        component_subs[comp_key] = ordered
        cand = pd.DataFrame(_build_component_station_candidates(ordered).copy().drop(columns=["geometry"], errors="ignore"))
        cand["component_id"] = comp_key
        cand["component_local_idx"] = np.arange(len(cand), dtype=int)
        cand_parts.append(cand)
    if not cand_parts:
        return {}, 0, 0, pd.DataFrame()
    all_cand = pd.concat(cand_parts, ignore_index=True)
    stations = pd.to_numeric(all_cand.get("station_m", np.nan), errors="coerce").to_numpy(dtype=float)
    preferred = pd.to_numeric(all_cand.get("solver_backbone_candidate_z_m", np.nan), errors="coerce").to_numpy(dtype=float)
    locked = all_cand.get("solver_hard_lock", pd.Series(False, index=all_cand.index)).astype(bool).to_numpy(dtype=bool)
    support_mode = all_cand.get("solver_support_class", pd.Series("unsupported", index=all_cand.index)).fillna("unsupported").astype(str).to_numpy()
    candidate_source = all_cand.get("solver_candidate_source", pd.Series("missing", index=all_cand.index)).fillna("missing").astype(str).to_numpy()

    init = _interp_series(stations, stations, preferred).astype(float)
    explicit = np.isfinite(preferred)
    init[explicit] = preferred[explicit]
    offset = 0
    for comp_key, sub in component_subs.items():
        base = np.asarray(component_fill_init.get(comp_key, np.asarray([], dtype=np.float32)), dtype=float)
        n_local = len(sub)
        if n_local and base.size == n_local:
            init[offset:offset+n_local] = base
        offset += n_local

    if np.count_nonzero(np.isfinite(init)) == 0:
        diag = all_cand[["component_id", "component_local_idx", "station_m"]].copy()
        diag["graph_backbone_z_m"] = np.nan
        return {comp: arr.astype(np.float32) for comp, arr in component_fill_init.items()}, 0, 0, diag

    z_locked = np.where(locked & np.isfinite(preferred), preferred, np.nan)
    unknown = (~locked) & np.isfinite(init)
    idx = np.where(unknown)[0]
    if idx.size == 0:
        diag = all_cand[["component_id", "component_local_idx", "station_m"]].copy()
        diag["graph_backbone_z_m"] = init.astype(np.float32)
        diag["graph_hard_lock"] = locked.astype(bool)
        diag["graph_prior_weight_sum"] = 0.0
        diag["graph_edge_weight_sum"] = 0.0
        diag["graph_curvature_weight_sum"] = 0.0
        diag["graph_centering_weight_sum"] = 0.0
        diag["graph_regularization_weight_sum"] = 0.0
        diag["graph_junction_weight_sum"] = 0.0
        diag["graph_junction_constrained"] = False
        diag["graph_residual_to_candidate_z_m"] = np.where(np.isfinite(preferred), init - preferred, np.nan)
        diag["graph_solver_support_class"] = support_mode
        diag["graph_candidate_source"] = candidate_source
        diag["graph_solution_mode"] = np.where(locked, "hard_locked", "prior_driven")
        diag["graph_unsupported_span_m"] = 0.0
        diag["graph_anchor_spacing_m"] = diag["component_id"].map(lambda c: float(component_stats.get(str(c), {}).get('anchor_spacing_m', np.nan))).astype(np.float32)
        diag["graph_topology_confidence"] = diag["component_id"].map(lambda c: float(component_stats.get(str(c), {}).get('topology_confidence', 0.0))).astype(np.float32)
        diag["graph_unsupported_regime"] = [
            _classify_unsupported_regime(sc, span, anchor, False, np.nan, topo)
            for sc, span, anchor, topo in zip(
                diag["graph_solver_support_class"].astype(str),
                pd.to_numeric(diag["graph_unsupported_span_m"], errors='coerce').to_numpy(dtype=float),
                pd.to_numeric(diag["graph_anchor_spacing_m"], errors='coerce').to_numpy(dtype=float),
                pd.to_numeric(diag["graph_topology_confidence"], errors='coerce').to_numpy(dtype=float),
            )
        ]
        return {comp: arr.astype(np.float32) for comp, arr in component_fill_init.items()}, 0, 0, diag
    pos = {int(j): k for k, j in enumerate(idx)}
    m = len(idx)
    Q = np.zeros((m, m), dtype=float)
    b = np.zeros(m, dtype=float)

    prior_weight_sum = np.zeros(len(all_cand), dtype=float)
    edge_weight_sum = np.zeros(len(all_cand), dtype=float)
    curvature_weight_sum = np.zeros(len(all_cand), dtype=float)
    centering_weight_sum = np.zeros(len(all_cand), dtype=float)
    junction_weight_sum = np.zeros(len(all_cand), dtype=float)
    junction_id_diag = np.full(len(all_cand), '', dtype=object)
    junction_role_diag = np.full(len(all_cand), 'not_in_junction', dtype=object)
    junction_target_diag = np.full(len(all_cand), np.nan, dtype=float)
    junction_distance_diag = np.full(len(all_cand), np.nan, dtype=float)
    junction_influence_diag = np.zeros(len(all_cand), dtype=float)
    junction_topology_diag = np.full(len(all_cand), 'none', dtype=object)
    slope_guard_weight_sum = np.zeros(len(all_cand), dtype=float)
    adverse_step_weight_sum = np.zeros(len(all_cand), dtype=float)
    physical_guard_weight_sum = np.zeros(len(all_cand), dtype=float)

    def add_row(coeffs: dict[int, float], target: float, weight: float, diag_name: str | None = None) -> None:
        if not np.isfinite(weight) or weight <= 0.0:
            return
        filt = {int(k): float(v) for k, v in coeffs.items() if int(k) in pos and np.isfinite(v)}
        if not filt:
            return
        target_eff = float(target)
        for k, v in coeffs.items():
            ik = int(k)
            if ik not in pos and np.isfinite(v):
                lock_val = z_locked[ik] if 0 <= ik < len(z_locked) else np.nan
                if np.isfinite(lock_val):
                    target_eff -= float(v) * float(lock_val)
        cols = [pos[int(k)] for k in filt]
        vals = np.asarray([filt[int(k)] for k in filt], dtype=float)
        if diag_name is not None:
            for k in filt:
                if diag_name == 'slope':
                    slope_guard_weight_sum[int(k)] += weight
                elif diag_name == 'adverse':
                    adverse_step_weight_sum[int(k)] += weight
                elif diag_name == 'physical':
                    physical_guard_weight_sum[int(k)] += weight
        for a, ca in enumerate(cols):
            b[ca] += weight * vals[a] * target_eff
            for cb, vb in zip(cols, vals):
                Q[ca, cb] += weight * vals[a] * vb

    for i in range(len(all_cand)):
        if not unknown[i] or not np.isfinite(preferred[i]):
            continue
        w = _support_data_weight(support_mode[i])
        prior_weight_sum[i] += w
        add_row({i: 1.0}, float(preferred[i]), w)

    global_idx_by_comp: dict[str, np.ndarray] = {}
    offset = 0
    for comp_key, sub in component_subs.items():
        n_local = len(sub)
        global_idx_by_comp[comp_key] = np.arange(offset, offset+n_local, dtype=int)
        offset += n_local

    for comp_key, gidx in global_idx_by_comp.items():
        if gidx.size == 0:
            continue
        st = stations[gidx]
        sm = support_mode[gidx]
        ini = init[gidx]
        unsupported_factor = _unsupported_span_factors(st, sm)
        anchor_spacing = float(component_stats.get(comp_key, {}).get('anchor_spacing_m', np.nan))
        topo_conf = float(component_stats.get(comp_key, {}).get('topology_confidence', 0.0))
        near_junction = np.zeros(len(gidx), dtype=bool)
        junction_distance = np.full(len(gidx), np.nan, dtype=float)
        for group in groups:
            records = [r for r in list(group.get('records', [])) if str(r.get('component_id')) == str(comp_key)]
            if not records:
                continue
            for rec in records:
                loc = int(rec.get('idx', -1))
                if 0 <= loc < len(gidx):
                    near_junction[loc] = True
                    if np.isfinite(st[loc]):
                        dist = np.abs(st - st[loc])
                        junction_distance = np.where(np.isfinite(junction_distance), np.minimum(junction_distance, dist), dist)
        regime = _classify_solver_regimes(st, sm, anchor_spacing, near_junction=near_junction, junction_distance=junction_distance, topology_confidence=topo_conf)
        for a in range(len(gidx)-1):
            i = int(gidx[a]); j = int(gidx[a+1])
            if not (np.isfinite(ini[a]) and np.isfinite(ini[a+1])):
                continue
            ds = max(abs(float(st[a+1]-st[a])), 1e-6)
            rf = _regime_behavior_factors(regime[a if regime.size else 0])
            rf2 = _regime_behavior_factors(regime[a + 1 if regime.size else 0])
            reg_edge = max(rf['edge'], rf2['edge'])
            reg_phys = max(rf['phys'], rf2['phys'])
            reg_slope = min(rf['slope_cap_mult'], rf2['slope_cap_mult'])
            w = (_support_edge_weight(sm[a], sm[a+1]) * max(unsupported_factor[a], unsupported_factor[a+1]) * reg_edge) / ds
            edge_weight_sum[i] += w
            edge_weight_sum[j] += w
            add_row({i: -1.0, j: 1.0}, 0.0, w)
            phys = _support_physical_guard_factor(sm[a], sm[a+1]) * max(unsupported_factor[a], unsupported_factor[a+1]) * reg_phys
            if phys > 0.0:
                slope_cap = _support_slope_cap(sm[a], sm[a+1]) * reg_slope
                target_dz = float(np.clip(ini[a+1] - ini[a], -slope_cap * ds, slope_cap * ds))
                sg_w = (0.015 / ds) * phys
                add_row({i: -1.0, j: 1.0}, target_dz, sg_w, diag_name='slope')
                if abs(ini[a+1] - ini[a]) > slope_cap * ds:
                    add_row({i: -1.0, j: 1.0}, target_dz, (0.03 / ds) * phys, diag_name='adverse')
        for a in range(1, len(gidx)-1):
            i0=int(gidx[a-1]); i1=int(gidx[a]); i2=int(gidx[a+1])
            if not (np.isfinite(ini[a-1]) and np.isfinite(ini[a]) and np.isfinite(ini[a+1])):
                continue
            rf = _regime_behavior_factors(regime[a if regime.size else 0])
            w = _support_curvature_weight(sm[a]) * unsupported_factor[a] * rf['curve']
            curvature_weight_sum[[i0, i1, i2]] += w
            add_row({i0:1.0, i1:-2.0, i2:1.0}, 0.0, w)
            phys_c = 0.01 * unsupported_factor[a] * _support_physical_guard_factor(sm[a-1], sm[a+1]) * rf['phys']
            add_row({i0:1.0, i1:-2.0, i2:1.0}, 0.0, phys_c, diag_name='physical')

    prelim_regime = np.full(len(all_cand), 'supported', dtype=object)
    for comp_key, gidx in global_idx_by_comp.items():
        st = stations[gidx]
        sm = support_mode[gidx]
        anchor_spacing = float(component_stats.get(comp_key, {}).get('anchor_spacing_m', np.nan))
        topo_conf = float(component_stats.get(comp_key, {}).get('topology_confidence', 0.0))
        near_junction = np.zeros(len(gidx), dtype=bool)
        junction_distance = np.full(len(gidx), np.nan, dtype=float)
        for group in groups:
            records = [r for r in list(group.get('records', [])) if str(r.get('component_id')) == str(comp_key)]
            if not records:
                continue
            for rec in records:
                loc = int(rec.get('idx', -1))
                if 0 <= loc < len(gidx):
                    near_junction[loc] = True
                    if np.isfinite(st[loc]):
                        dist = np.abs(st - st[loc])
                        junction_distance = np.where(np.isfinite(junction_distance), np.minimum(junction_distance, dist), dist)
        prelim_regime[gidx] = _classify_solver_regimes(st, sm, anchor_spacing, near_junction=near_junction, junction_distance=junction_distance, topology_confidence=topo_conf)
    for i in np.where(unknown & np.isfinite(init))[0]:
        rf = _regime_behavior_factors(prelim_regime[i])
        centering_weight_sum[i] += 0.01 * rf['centering']
        add_row({int(i): 1.0}, float(init[i]), 0.01 * rf['centering'])

    constraint_count = 0
    constrained_nodes: set[int] = set()
    junction_summaries: list[dict[str, Any]] = []
    for group in groups:
        records = list(group.get("records", []))
        comps = {str(r.get("component_id")) for r in records}
        if len(comps) < 2:
            continue
        score_items = sorted(((str(c), float(component_stats.get(str(c), {}).get("dominance_score", 0.0))) for c in comps), key=lambda kv: kv[1], reverse=True)
        dominant = score_items[0][0]
        dominant_score = score_items[0][1]
        second_score = score_items[1][1] if len(score_items) > 1 else float("-inf")
        has_clear_dominance = bool(np.isfinite(dominant_score) and (not np.isfinite(second_score) or (dominant_score - second_score) > 1.0))
        if has_clear_dominance:
            dominant_records = [r for r in records if str(r.get("component_id")) == dominant]
            target_vals = []
            for r in dominant_records:
                comp = str(r.get("component_id")); loc = int(r.get("idx", -1))
                arr = component_fill_init.get(comp)
                if arr is not None and 0 <= loc < arr.size and np.isfinite(arr[loc]):
                    target_vals.append(float(arr[loc]))
            if not target_vals:
                continue
            target = float(np.nanmedian(target_vals))
        else:
            vals = []
            for r in records:
                comp = str(r.get("component_id")); loc = int(r.get("idx", -1))
                arr = component_fill_init.get(comp)
                if arr is not None and 0 <= loc < arr.size and np.isfinite(arr[loc]):
                    vals.append(float(arr[loc]))
            if not vals:
                continue
            target = float(np.nanmedian(vals))
            dominant = None
        summary = {
            "junction_id": str(group.get("junction_id", f"junction_{len(junction_summaries)+1}")),
            "component_ids": sorted(comps),
            "dominant_component_id": str(dominant) if dominant is not None else "",
            "topology_source": str(group.get("topology_source", "geometric_fallback")),
            "has_clear_dominance": bool(has_clear_dominance),
            "junction_target_z_m": float(target),
            "junction_x": float(np.nanmean([float(r.get("x", np.nan)) for r in records])) if records else float("nan"),
            "junction_y": float(np.nanmean([float(r.get("y", np.nan)) for r in records])) if records else float("nan"),
        }
        local_adjustments = []
        constrained_station_counter = 0
        for r in records:
            comp = str(r.get("component_id"))
            loc = int(r.get("idx", -1))
            gidx = global_idx_by_comp.get(comp)
            if gidx is None or loc < 0 or loc >= len(gidx):
                continue
            current_arr = component_fill_init.get(comp)
            if current_arr is None or loc >= current_arr.size:
                continue
            stations_local = pd.to_numeric(component_subs[comp].get("station_m", pd.Series(np.nan, index=component_subs[comp].index)), errors="coerce").to_numpy(dtype=float) if comp in component_subs else None
            lo = max(0, loc - 2)
            hi = min(len(gidx), loc + 3)
            endpoint_station = float(stations_local[loc]) if stations_local is not None and 0 <= loc < len(stations_local) and np.isfinite(stations_local[loc]) else float(loc)
            if has_clear_dominance and comp == dominant:
                base_weight, pull, dist_scale = 0.0, 0.0, 1.0
                branch_role = "dominant"
            else:
                base_weight, pull, dist_scale = _junction_constraint_params(
                    group=group,
                    comp=comp,
                    loc=loc,
                    records=records,
                    dominant=dominant,
                    has_clear_dominance=has_clear_dominance,
                    component_stats=component_stats,
                    component_subs=component_subs,
                )
                branch_role = "constrained_branch" if has_clear_dominance else "balanced"
            for jloc in range(lo, hi):
                gi = int(gidx[jloc])
                current = float(current_arr[jloc]) if np.isfinite(current_arr[jloc]) else float("nan")
                if stations_local is not None and 0 <= jloc < len(stations_local) and np.isfinite(stations_local[jloc]):
                    along_dist = abs(float(stations_local[jloc]) - endpoint_station)
                else:
                    along_dist = float(abs(jloc - loc))
                if has_clear_dominance and comp == dominant:
                    weight = 0.0
                else:
                    decay = float(np.exp(-along_dist / max(dist_scale, 1e-6)))
                    local_pull = pull * max(0.15, decay)
                    regime_factor = _regime_behavior_factors(prelim_regime[gi])['junction'] if gi < len(prelim_regime) else 1.0
                    if jloc == loc or not np.isfinite(current):
                        local_target = target
                        weight = base_weight * local_pull * regime_factor
                    else:
                        local_target = (1.0 - local_pull) * current + local_pull * target
                        weight = (base_weight * 0.4) * local_pull * regime_factor
                    junction_weight_sum[gi] += weight
                    add_row({gi: 1.0}, float(local_target), float(weight))
                    constraint_count += 1
                    constrained_nodes.add(int(gi))
                    constrained_station_counter += 1
                    if np.isfinite(current):
                        local_adjustments.append(abs(float(local_target) - current))
                if (has_clear_dominance and comp == dominant) or weight >= junction_influence_diag[gi]:
                    junction_id_diag[gi] = summary["junction_id"]
                    junction_role_diag[gi] = branch_role
                    junction_target_diag[gi] = float(target)
                    junction_distance_diag[gi] = float(along_dist)
                    junction_influence_diag[gi] = max(junction_influence_diag[gi], float(weight))
                    junction_topology_diag[gi] = summary["topology_source"]
        summary["constrained_station_count"] = int(constrained_station_counter)
        summary["mean_adjustment_m"] = float(np.nanmean(local_adjustments)) if local_adjustments else 0.0
        summary["max_adjustment_m"] = float(np.nanmax(local_adjustments)) if local_adjustments else 0.0
        summary["topology_confidence"] = float(np.nanmean([float(component_stats.get(c, {}).get("topology_confidence", 0.0)) for c in summary["component_ids"]])) if summary["component_ids"] else 0.0
        summary["support_strength_sum"] = float(np.nansum([float(r.get("auth_strength", 0.0)) for r in records]))
        junction_summaries.append(summary)

    Q.flat[::m+1] += 1e-8
    rhs = b.copy()
    try:
        sol = np.linalg.solve(Q, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(Q, rhs, rcond=None)
    out = init.copy()
    out[idx] = sol
    if np.any(locked & np.isfinite(z_locked)):
        out[locked & np.isfinite(z_locked)] = z_locked[locked & np.isfinite(z_locked)]

    result: dict[str, np.ndarray] = {}
    for comp_key, gidx in global_idx_by_comp.items():
        arr = np.asarray(out[gidx], dtype=float).copy()
        st = stations[gidx]
        pref = preferred[gidx]
        loc_locked = locked[gidx] & np.isfinite(pref)
        if np.count_nonzero(np.isfinite(arr)) >= 2:
            finite_idx = np.where(np.isfinite(arr))[0]
            for k in range(1, len(finite_idx)):
                i0 = finite_idx[k-1]
                i1 = finite_idx[k]
                ds = max(abs(float(st[i1]-st[i0])), 1e-6)
                dz = float(arr[i1]-arr[i0])
                slope = dz / ds
                if abs(slope) > 0.25 and not loc_locked[i1]:
                    arr[i1] = arr[i0] + np.sign(dz) * 0.25 * ds
        if np.any(loc_locked):
            arr[loc_locked] = pref[loc_locked]
        out[gidx] = arr
        result[comp_key] = arr.astype(np.float32)

    diagnostics = all_cand[["component_id", "component_local_idx", "station_m"]].copy()
    diagnostics["graph_backbone_z_m"] = out.astype(np.float32)
    diagnostics["graph_hard_lock"] = locked.astype(bool)
    diagnostics["graph_prior_weight_sum"] = prior_weight_sum.astype(np.float32)
    diagnostics["graph_edge_weight_sum"] = edge_weight_sum.astype(np.float32)
    diagnostics["graph_curvature_weight_sum"] = curvature_weight_sum.astype(np.float32)
    diagnostics["graph_centering_weight_sum"] = centering_weight_sum.astype(np.float32)
    diagnostics["graph_slope_guard_weight_sum"] = slope_guard_weight_sum.astype(np.float32)
    diagnostics["graph_adverse_step_weight_sum"] = adverse_step_weight_sum.astype(np.float32)
    diagnostics["graph_physical_guard_weight_sum"] = physical_guard_weight_sum.astype(np.float32)
    reg_sum = edge_weight_sum + curvature_weight_sum + centering_weight_sum + physical_guard_weight_sum
    diagnostics["graph_regularization_weight_sum"] = reg_sum.astype(np.float32)
    diagnostics["graph_junction_weight_sum"] = junction_weight_sum.astype(np.float32)
    diagnostics["graph_junction_constrained"] = (junction_weight_sum > 0.0)
    diagnostics["graph_junction_id"] = junction_id_diag
    diagnostics["graph_junction_role"] = junction_role_diag
    diagnostics["graph_junction_target_z_m"] = junction_target_diag.astype(np.float32)
    diagnostics["graph_residual_to_candidate_z_m"] = np.where(np.isfinite(preferred), out - preferred, np.nan).astype(np.float32)
    diagnostics["graph_solver_support_class"] = support_mode.astype(object)
    diagnostics["graph_candidate_source"] = candidate_source.astype(object)

    solution_mode = np.full(len(diagnostics), "regularization_driven", dtype=object)
    solution_mode[locked] = "hard_locked"
    unlocked = ~locked
    solution_mode[unlocked & (junction_weight_sum > np.maximum(prior_weight_sum, reg_sum))] = "junction_constrained"
    solution_mode[unlocked & (prior_weight_sum >= np.maximum(reg_sum, junction_weight_sum)) & (prior_weight_sum > 0.0)] = "prior_driven"
    diagnostics["graph_solution_mode"] = solution_mode

    unsupported_span = np.zeros(len(diagnostics), dtype=np.float32)
    for comp_key, gidx in global_idx_by_comp.items():
        st = stations[gidx]
        sm = support_mode[gidx]
        weak = np.isin(sm, ["unsupported", "xs_residual_only"])
        run_start = None
        for ii in range(len(gidx) + 1):
            is_weak = bool(ii < len(gidx) and weak[ii])
            if is_weak and run_start is None:
                run_start = ii
            elif (not is_weak) and run_start is not None:
                run_end = ii - 1
                finite_st = st[run_start:run_end+1]
                if np.count_nonzero(np.isfinite(finite_st)) >= 2:
                    span = float(np.nanmax(finite_st) - np.nanmin(finite_st))
                elif np.count_nonzero(np.isfinite(st)) >= 2 and run_end + 1 < len(st):
                    span = float(abs(st[run_end+1] - st[run_start]))
                else:
                    span = 0.0
                unsupported_span[gidx[run_start:run_end+1]] = np.float32(max(span, 0.0))
                run_start = None
    diagnostics["graph_unsupported_span_m"] = unsupported_span
    local_slope = np.full(len(diagnostics), np.nan, dtype=np.float32)
    local_curv = np.full(len(diagnostics), np.nan, dtype=np.float32)
    for comp_key, gidx in global_idx_by_comp.items():
        st = stations[gidx]
        vals = out[gidx]
        for j in range(1, len(gidx)):
            if np.isfinite(vals[j]) and np.isfinite(vals[j-1]) and np.isfinite(st[j]) and np.isfinite(st[j-1]):
                ds = max(abs(float(st[j] - st[j-1])), 1e-6)
                local_slope[gidx[j]] = np.float32((float(vals[j]) - float(vals[j-1])) / ds)
        for j in range(1, len(gidx)-1):
            if np.isfinite(vals[j-1]) and np.isfinite(vals[j]) and np.isfinite(vals[j+1]):
                local_curv[gidx[j]] = np.float32(float(vals[j-1]) - 2.0 * float(vals[j]) + float(vals[j+1]))
    diagnostics['graph_junction_adjustment_z_m'] = np.where(np.isfinite(junction_target_diag), out - junction_target_diag, np.nan).astype(np.float32)
    diagnostics['graph_junction_distance_m'] = junction_distance_diag.astype(np.float32)
    diagnostics['graph_junction_influence_weight'] = junction_influence_diag.astype(np.float32)
    diagnostics['graph_junction_topology_source'] = junction_topology_diag
    diagnostics['graph_anchor_spacing_m'] = diagnostics['component_id'].map(lambda c: float(component_stats.get(str(c), {}).get('anchor_spacing_m', np.nan))).astype(np.float32)
    diagnostics['graph_topology_confidence'] = diagnostics['component_id'].map(lambda c: float(component_stats.get(str(c), {}).get('topology_confidence', 0.0))).astype(np.float32)
    diagnostics['graph_unsupported_regime'] = [
        _classify_unsupported_regime(sc, span, anchor, jc, jd, topo)
        for sc, span, anchor, jc, jd, topo in zip(
            diagnostics['graph_solver_support_class'].astype(str),
            pd.to_numeric(diagnostics['graph_unsupported_span_m'], errors='coerce').to_numpy(dtype=float),
            pd.to_numeric(diagnostics['graph_anchor_spacing_m'], errors='coerce').to_numpy(dtype=float),
            diagnostics['graph_junction_constrained'].fillna(False).astype(bool).to_numpy(dtype=bool),
            pd.to_numeric(diagnostics['graph_junction_distance_m'], errors='coerce').to_numpy(dtype=float),
            pd.to_numeric(diagnostics['graph_topology_confidence'], errors='coerce').to_numpy(dtype=float),
        )
    ]
    diagnostics['graph_local_slope'] = local_slope
    diagnostics['graph_local_curvature'] = local_curv
    diagnostics['graph_slope_guard_active'] = slope_guard_weight_sum > 0.0
    diagnostics['graph_adverse_step_guard_active'] = adverse_step_weight_sum > 0.0
    diagnostics.attrs['junction_diagnostics'] = junction_summaries
    return result, int(constraint_count), int(len(constrained_nodes)), diagnostics


def summarize_graph_physical_plausibility(diag: pd.DataFrame) -> dict[str, float | int]:
    if diag is None or getattr(diag, "empty", True):
        return {
            'graph_slope_guard_station_count': 0,
            'graph_adverse_step_guard_station_count': 0,
            'graph_physical_guard_station_count': 0,
            'graph_max_abs_local_slope': 0.0,
            'graph_p95_abs_local_slope': 0.0,
            'graph_max_abs_local_curvature': 0.0,
            'graph_p95_abs_local_curvature': 0.0,
            'graph_mean_unsupported_span_m': 0.0,
            'graph_p95_unsupported_span_m': 0.0,
        }
    idx = getattr(diag, 'index', pd.RangeIndex(1))
    slope = pd.to_numeric(diag.get('graph_local_slope', pd.Series(np.nan, index=idx)), errors='coerce')
    curv = pd.to_numeric(diag.get('graph_local_curvature', pd.Series(np.nan, index=idx)), errors='coerce')
    span = pd.to_numeric(diag.get('graph_unsupported_span_m', pd.Series(np.nan, index=idx)), errors='coerce')
    slope = np.asarray(slope.to_numpy(dtype=float) if hasattr(slope, 'to_numpy') else np.asarray(slope, dtype=float), dtype=float)
    curv = np.asarray(curv.to_numpy(dtype=float) if hasattr(curv, 'to_numpy') else np.asarray(curv, dtype=float), dtype=float)
    span = np.asarray(span.to_numpy(dtype=float) if hasattr(span, 'to_numpy') else np.asarray(span, dtype=float), dtype=float)
    slope_abs = np.abs(slope[np.isfinite(slope)])
    curv_abs = np.abs(curv[np.isfinite(curv)])
    span_fin = span[np.isfinite(span)]
    return {
        'graph_slope_guard_station_count': int(np.count_nonzero(pd.Series(diag.get('graph_slope_guard_active', False)).fillna(False).astype(bool).to_numpy(dtype=bool))),
        'graph_adverse_step_guard_station_count': int(np.count_nonzero(pd.Series(diag.get('graph_adverse_step_guard_active', False)).fillna(False).astype(bool).to_numpy(dtype=bool))),
        'graph_physical_guard_station_count': int(np.count_nonzero(np.asarray(pd.to_numeric(diag.get('graph_physical_guard_weight_sum', pd.Series(0.0, index=idx)), errors='coerce').fillna(0.0).to_numpy(dtype=float), dtype=float) > 0.0)),
        'graph_max_abs_local_slope': float(np.nanmax(slope_abs)) if slope_abs.size else 0.0,
        'graph_p95_abs_local_slope': float(np.nanpercentile(slope_abs, 95)) if slope_abs.size else 0.0,
        'graph_max_abs_local_curvature': float(np.nanmax(curv_abs)) if curv_abs.size else 0.0,
        'graph_p95_abs_local_curvature': float(np.nanpercentile(curv_abs, 95)) if curv_abs.size else 0.0,
        'graph_mean_unsupported_span_m': float(np.nanmean(span_fin)) if span_fin.size else 0.0,
        'graph_p95_unsupported_span_m': float(np.nanpercentile(span_fin, 95)) if span_fin.size else 0.0,
        'graph_unsupported_regime_counts': {str(k): int(v) for k, v in pd.Series(diag.get('graph_unsupported_regime', pd.Series([], dtype=object))).fillna('missing').astype(str).value_counts().to_dict().items()},
    }


def _solve_network_backbone(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, int], pd.DataFrame]:
    component_fill: dict[str, np.ndarray] = {}
    metrics = {
        'component_hard_lock_count': 0,
        'component_stage_controlled_count': 0,
        'component_anchored_count': 0,
        'component_unsupported_count': 0,
        'junction_group_count': 0,
        'junction_adjusted_component_count': 0,
        'junction_adjusted_station_count': 0,
        'dominant_junction_group_count': 0,
        'dominant_preserved_component_count': 0,
        'network_junction_count': 0,
        'topology_guided_junction_count': 0,
        'topology_guided_component_count': 0,
        'geometric_fallback_junction_count': 0,
        'graph_junction_constraint_count': 0,
        'graph_topology_guided_constraint_count': 0,
        'graph_distance_weighted_constraint_count': 0,
    }
    component_stats = {}
    topology = _component_topology_records(frame)
    for comp, sub in frame.groupby('component_id', sort=False):
        sub = sub.sort_values('station_m').copy()
        comp_key = str(comp)
        solved, comp_metrics = _solve_component_backbone(sub)
        component_fill[comp_key] = solved
        for k, v in comp_metrics.items():
            metrics[k] += int(v)
        backbone_mode = sub.get('backbone_mode', pd.Series('missing', index=sub.index)).fillna('missing').astype(str).to_numpy()
        auth_strength = pd.to_numeric(sub.get('authoritative_station_support_strength', pd.Series(np.nan, index=sub.index)), errors='coerce').to_numpy(dtype=float)
        stations = pd.to_numeric(sub.get('station_m', pd.Series(np.nan, index=sub.index)), errors='coerce').to_numpy(dtype=float)
        station_span = float(np.nanmax(stations) - np.nanmin(stations)) if np.count_nonzero(np.isfinite(stations)) >= 2 else 0.0
        auth_count = int(np.count_nonzero(np.char.find(backbone_mode.astype(str), 'authoritative') >= 0))
        long_count = int(np.count_nonzero(backbone_mode.astype(str) == 'longitudinal_profile'))
        topo = topology.get(comp_key, {})
        mainstem_rank = float(topo.get('mainstem_rank', np.nan))
        network_order = float(topo.get('network_order', np.nan))
        distance_to_mouth = float(topo.get('distance_to_mouth_m', np.nan))
        topo_bonus = 0.0
        if np.isfinite(mainstem_rank):
            topo_bonus += max(0.0, 5.0 - mainstem_rank) * 4.0
        if np.isfinite(network_order):
            topo_bonus += max(network_order, 0.0) * 1.5
        if np.isfinite(distance_to_mouth):
            topo_bonus += max(0.0, 100000.0 - distance_to_mouth) / 10000.0
        if topo.get('downstream_component_id') or topo.get('upstream_component_ids'):
            topo_bonus += 1.0
        anchor_spacing = _component_anchor_spacing(sub)
        endpoint_support_mode = {}
        if len(sub) > 0:
            cand = _build_component_station_candidates(sub)
            endpoint_support_mode[0] = str(cand["solver_support_class"].iloc[0])
            endpoint_support_mode[len(sub) - 1] = str(cand["solver_support_class"].iloc[-1])
        component_stats[comp_key] = {
            'dominance_score': float(8.0 * auth_count + 2.0 * long_count + 0.25 * max(station_span, 0.0) + float(np.nansum(np.clip(auth_strength, 0.0, None))) + 0.1 * float(len(sub)) + topo_bonus),
            'anchor_spacing_m': float(anchor_spacing) if np.isfinite(anchor_spacing) else np.nan,
            'topology_confidence': _component_topology_confidence(topo),
            'endpoint_support_mode': endpoint_support_mode,
        }

    groups, _ = _build_junction_groups(frame, component_fill)
    metrics['topology_guided_junction_count'] = int(sum(1 for g in groups if g.get('topology_source') == 'explicit_junction_id'))
    metrics['geometric_fallback_junction_count'] = int(sum(1 for g in groups if g.get('topology_source') == 'geometric_fallback'))
    metrics['junction_group_count'] = len(groups)
    metrics['network_junction_count'] = len(groups)
    topology_guided_components: set[str] = set()
    for group in groups:
        if group.get('topology_source') == 'explicit_junction_id':
            topology_guided_components.update(str(r.get('component_id')) for r in group.get('records', []))
        records = list(group.get('records', []))
        comps = {str(r.get('component_id')) for r in records}
        if len(comps) >= 2:
            metrics['junction_adjusted_component_count'] += max(0, len(comps) - 1)
            score_items = sorted(((str(c), float(component_stats.get(str(c), {}).get('dominance_score', 0.0))) for c in comps), key=lambda kv: kv[1], reverse=True)
            if score_items:
                dominant_score = score_items[0][1]
                second_score = score_items[1][1] if len(score_items) > 1 else float('-inf')
                has_clear_dominance = bool(np.isfinite(dominant_score) and (not np.isfinite(second_score) or (dominant_score - second_score) > 1.0))
                if has_clear_dominance:
                    metrics['dominant_junction_group_count'] += 1
                    metrics['dominant_preserved_component_count'] += 1

    component_fill, constraint_count, constrained_node_count, diagnostics = _global_graph_backbone_solve(frame, component_stats, groups, component_fill)
    metrics['graph_junction_constraint_count'] = int(constraint_count)
    metrics['graph_topology_guided_constraint_count'] = int(sum(1 for g in groups if g.get('topology_source') == 'explicit_junction_id'))
    metrics['graph_distance_weighted_constraint_count'] = int(constrained_node_count)
    metrics['junction_adjusted_station_count'] = int(constrained_node_count)
    metrics['topology_guided_component_count'] = int(len(topology_guided_components))
    return component_fill, metrics, diagnostics
