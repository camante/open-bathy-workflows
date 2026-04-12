from __future__ import annotations

import math
from typing import Mapping, Any

SUPPORT_CLASS_CODES = {
    'missing': 0,
    'authoritative_locked': 1,
    'authoritative_backbone': 2,
    'anchored_interpolated': 3,
    'stage_controlled': 4,
    'graph_backbone': 5,
    'resolved_backbone': 5,
    'xs_residual_only': 6,
    'xs_only': 6,
    'unsupported': 7,
}

SOLUTION_MODE_CODES = {
    'missing': 0,
    'hard_locked': 1,
    'prior_driven': 2,
    'regularization_driven': 3,
    'junction_constrained': 4,
    'mixed_graph_solution': 5,
}

UNCERTAINTY_CLASS_CODES = {
    'missing': 0,
    'very_low': 1,
    'low': 2,
    'moderate': 3,
    'high': 4,
    'very_high': 5,
}


UNSUPPORTED_REGIME_CODES = {
    'missing': 0,
    'supported': 1,
    'short_gap_bridge': 2,
    'medium_gap_regularized': 3,
    'long_gap_stiffened': 4,
    'junction_dominated_unsupported': 5,
}


def _f(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float('nan')
    return x if math.isfinite(x) else float('nan')


def graph_solution_confidence(row: Mapping[str, Any]) -> float:
    support = str(row.get('graph_solver_support_class', 'unsupported') or 'unsupported')
    mode = str(row.get('graph_solution_mode', 'missing') or 'missing')
    hard_lock = bool(row.get('graph_hard_lock', False))
    prior_w = max(_f(row.get('graph_prior_weight_sum')), 0.0) if math.isfinite(_f(row.get('graph_prior_weight_sum'))) else 0.0
    reg_w = max(_f(row.get('graph_regularization_weight_sum')), 0.0) if math.isfinite(_f(row.get('graph_regularization_weight_sum'))) else 0.0
    jun_w = max(_f(row.get('graph_junction_weight_sum')), 0.0) if math.isfinite(_f(row.get('graph_junction_weight_sum'))) else 0.0
    resid = abs(_f(row.get('graph_residual_to_candidate_z_m')))
    unsupported_span = max(_f(row.get('graph_unsupported_span_m')), 0.0) if math.isfinite(_f(row.get('graph_unsupported_span_m'))) else 0.0
    if hard_lock and support == 'authoritative_locked':
        return 0.99
    base = {
        'authoritative_backbone': 0.90,
        'anchored_interpolated': 0.75,
        'graph_backbone': 0.62,
        'stage_controlled': 0.48,
        'xs_residual_only': 0.40,
        'unsupported': 0.28,
    }.get(support, 0.2)
    total = prior_w + reg_w + jun_w
    prior_frac = prior_w / total if total > 0 else 0.0
    reg_frac = reg_w / total if total > 0 else 0.0
    jun_frac = jun_w / total if total > 0 else 0.0
    mode_adj = {
        'hard_locked': 0.08,
        'prior_driven': 0.06,
        'regularization_driven': -0.05,
        'junction_constrained': -0.08,
        'mixed_graph_solution': -0.02,
    }.get(mode, -0.05)
    conf = base + 0.10 * prior_frac - 0.06 * reg_frac - 0.08 * jun_frac + mode_adj
    if math.isfinite(resid):
        conf -= min(resid * 0.08, 0.18)
    if math.isfinite(unsupported_span):
        conf -= min(unsupported_span / 750.0, 0.20)
    return max(0.0, min(1.0, conf))


def uncertainty_class_from_confidence(conf: float) -> str:
    if not math.isfinite(conf):
        return 'missing'
    if conf >= 0.90:
        return 'very_low'
    if conf >= 0.75:
        return 'low'
    if conf >= 0.55:
        return 'moderate'
    if conf >= 0.35:
        return 'high'
    return 'very_high'


def support_class_code(name: str) -> int:
    return int(SUPPORT_CLASS_CODES.get(str(name), 0))


def solution_mode_code(name: str) -> int:
    return int(SOLUTION_MODE_CODES.get(str(name), 0))


def uncertainty_class_code(name: str) -> int:
    return int(UNCERTAINTY_CLASS_CODES.get(str(name), 0))



def unsupported_regime_code(name: str) -> int:
    return int(UNSUPPORTED_REGIME_CODES.get(str(name), 0))
