from __future__ import annotations

"""Deterministic graph-backbone helpers used by legacy channel-scaffold code.

The active built-in river workflow does not route through this graph solver, but
fresh-package imports require these helpers to exist because the surrounding
channel-scaffold module still exposes legacy diagnostics.  The implementation is
small, deterministic, and side-effect free: it builds a station-wise candidate
from existing backbone/bed columns and returns diagnostics that make the support
state explicit instead of silently inventing a second science path.
"""

from typing import Any

import numpy as np
import pandas as pd


_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "active_core_z_m",
    "station_target_z_m",
    "bed_backbone_z_m",
    "backbone_bed_z_m",
    "centerline_bed_z_m",
    "resolved_bed_z_m",
    "authoritative_bed_z_m",
    "z_m",
)


def _to_float_array(values: Any, n: int) -> np.ndarray:
    try:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    except Exception:
        arr = np.full(n, np.nan, dtype=float)
    if arr.size != n:
        out = np.full(n, np.nan, dtype=float)
        m = min(n, arr.size)
        if m:
            out[:m] = arr[:m]
        return out
    return arr


def _candidate_source_for_row(row: pd.Series) -> tuple[float, str, str]:
    for col in _CANDIDATE_COLUMNS:
        if col in row.index:
            try:
                val = float(row.get(col))
            except Exception:
                val = float("nan")
            if np.isfinite(val):
                if col in {"authoritative_bed_z_m", "active_core_z_m"}:
                    support = "authoritative_anchor"
                elif col in {"station_target_z_m", "bed_backbone_z_m", "backbone_bed_z_m", "centerline_bed_z_m"}:
                    support = "backbone_guidance"
                else:
                    support = "weak_guidance"
                return val, col, support
    return float("nan"), "missing", "unsupported"


def _smooth_fill(stations: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    out = np.asarray(candidates, dtype=float).copy()
    good = np.isfinite(out) & np.isfinite(stations)
    if np.count_nonzero(good) >= 2:
        order = np.argsort(stations[good])
        xs = stations[good][order]
        ys = out[good][order]
        all_good_station = np.isfinite(stations)
        out[all_good_station & ~np.isfinite(out)] = np.interp(
            stations[all_good_station & ~np.isfinite(out)], xs, ys, left=ys[0], right=ys[-1]
        )
    elif np.count_nonzero(good) == 1:
        out[~np.isfinite(out)] = float(out[good][0])
    return out.astype(np.float32)


def _build_component_station_candidates(sub: pd.DataFrame) -> pd.DataFrame:
    """Return one station-wise candidate table for a component."""
    if sub is None or getattr(sub, "empty", True):
        return pd.DataFrame(columns=["component_id", "station_m", "candidate_z_m", "candidate_source", "solver_support_class"])
    work = sub.copy()
    if "station_m" in work.columns:
        work["station_m"] = pd.to_numeric(work["station_m"], errors="coerce")
    else:
        work["station_m"] = np.arange(len(work), dtype=float)
    if "component_id" not in work.columns:
        work["component_id"] = "main"
    recs: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        z, source, support = _candidate_source_for_row(row)
        recs.append(
            {
                "component_id": str(row.get("component_id", "main")),
                "station_m": float(row.get("station_m")) if np.isfinite(row.get("station_m")) else np.nan,
                "candidate_z_m": z,
                "candidate_source": source,
                "solver_support_class": support,
            }
        )
    return pd.DataFrame.from_records(recs)


def _solve_component_backbone(sub: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """Solve a deterministic component backbone from available candidate columns."""
    if sub is None or getattr(sub, "empty", True):
        return np.asarray([], dtype=np.float32), {"component_empty_count": 1}
    candidates = _build_component_station_candidates(sub)
    stations = pd.to_numeric(candidates.get("station_m", pd.Series(np.arange(len(candidates)))), errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(candidates.get("candidate_z_m", pd.Series(np.nan, index=candidates.index)), errors="coerce").to_numpy(dtype=float)
    solved = _smooth_fill(stations, z)
    metrics = {
        "component_station_count": int(len(candidates)),
        "component_candidate_count": int(np.count_nonzero(np.isfinite(z))),
        "component_unsupported_count": int(np.count_nonzero(~np.isfinite(z))),
        "component_hard_lock_count": 0,
        "component_stage_controlled_count": int(np.count_nonzero(np.isfinite(z))),
        "component_anchored_count": int(np.count_nonzero(np.isfinite(z))),
    }
    return solved, metrics


def _solve_network_backbone(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, int], pd.DataFrame]:
    """Return component solved arrays, summary metrics, and diagnostics."""
    if frame is None or getattr(frame, "empty", True):
        metrics = {
            "junction_group_count": 0,
            "junction_adjusted_component_count": 0,
            "junction_adjusted_station_count": 0,
            "dominant_junction_group_count": 0,
            "dominant_preserved_component_count": 0,
            "network_junction_count": 0,
            "topology_guided_junction_count": 0,
            "topology_guided_component_count": 0,
            "geometric_fallback_junction_count": 0,
            "graph_junction_constraint_count": 0,
            "graph_topology_guided_constraint_count": 0,
            "graph_distance_weighted_constraint_count": 0,
            "component_hard_lock_count": 0,
            "component_stage_controlled_count": 0,
            "component_anchored_count": 0,
            "component_unsupported_count": 0,
        }
        return {}, metrics, pd.DataFrame()

    work = frame.copy()
    if "component_id" not in work.columns:
        work["component_id"] = "main"
    if "station_m" not in work.columns:
        work["station_m"] = np.arange(len(work), dtype=float)
    work["component_id"] = work["component_id"].fillna("main").astype(str)
    work["station_m"] = pd.to_numeric(work["station_m"], errors="coerce")

    component_fill: dict[str, np.ndarray] = {}
    diag_records: list[dict[str, Any]] = []
    metrics = {
        "junction_group_count": 0,
        "junction_adjusted_component_count": 0,
        "junction_adjusted_station_count": 0,
        "dominant_junction_group_count": 0,
        "dominant_preserved_component_count": 0,
        "network_junction_count": 0,
        "topology_guided_junction_count": 0,
        "topology_guided_component_count": 0,
        "geometric_fallback_junction_count": 0,
        "graph_junction_constraint_count": 0,
        "graph_topology_guided_constraint_count": 0,
        "graph_distance_weighted_constraint_count": 0,
        "component_hard_lock_count": 0,
        "component_stage_controlled_count": 0,
        "component_anchored_count": 0,
        "component_unsupported_count": 0,
    }

    for comp, sub in work.groupby("component_id", sort=False):
        solved, comp_metrics = _solve_component_backbone(sub)
        component_fill[str(comp)] = solved
        for k in ("component_hard_lock_count", "component_stage_controlled_count", "component_anchored_count", "component_unsupported_count"):
            metrics[k] += int(comp_metrics.get(k, 0))
        candidates = _build_component_station_candidates(sub).reset_index(drop=True)
        stations = candidates["station_m"].to_numpy(dtype=float) if not candidates.empty else np.asarray([], dtype=float)
        finite = np.isfinite(solved)
        local_slope = np.full(len(solved), np.nan, dtype=float)
        if len(solved) >= 2:
            ds = np.gradient(stations)
            dz = np.gradient(solved.astype(float))
            with np.errstate(divide="ignore", invalid="ignore"):
                local_slope = np.where(np.abs(ds) > 0, dz / ds, np.nan)
        for i, (_, row) in enumerate(candidates.iterrows()):
            z = float(solved[i]) if i < len(solved) and np.isfinite(solved[i]) else float("nan")
            raw = row.get("candidate_z_m", np.nan)
            try:
                raw_f = float(raw)
            except Exception:
                raw_f = float("nan")
            support = str(row.get("solver_support_class", "unsupported"))
            source = str(row.get("candidate_source", "missing"))
            if source == "missing" and np.isfinite(z):
                mode = "interpolated_component_backbone"
            elif np.isfinite(z):
                mode = "candidate_preserved"
            else:
                mode = "missing"
            diag_records.append(
                {
                    "component_id": str(comp),
                    "station_m": float(row.get("station_m")) if np.isfinite(row.get("station_m")) else np.nan,
                    "graph_backbone_z_m": z,
                    "graph_hard_lock": False,
                    "graph_prior_weight_sum": np.float32(1.0 if np.isfinite(z) else 0.0),
                    "graph_edge_weight_sum": np.float32(0.0),
                    "graph_curvature_weight_sum": np.float32(0.0),
                    "graph_centering_weight_sum": np.float32(0.0),
                    "graph_regularization_weight_sum": np.float32(0.0),
                    "graph_junction_weight_sum": np.float32(0.0),
                    "graph_junction_constrained": False,
                    "graph_residual_to_candidate_z_m": np.float32(z - raw_f) if np.isfinite(z) and np.isfinite(raw_f) else np.float32(np.nan),
                    "graph_solver_support_class": support,
                    "graph_candidate_source": source,
                    "graph_solution_mode": mode,
                    "graph_unsupported_span_m": np.float32(0.0 if source != "missing" else 1.0),
                    "graph_unsupported_regime": "supported" if source != "missing" else "unsupported",
                    "graph_slope_guard_weight_sum": np.float32(0.0),
                    "graph_adverse_step_weight_sum": np.float32(0.0),
                    "graph_physical_guard_weight_sum": np.float32(0.0),
                    "graph_local_slope": np.float32(local_slope[i]) if i < len(local_slope) and np.isfinite(local_slope[i]) else np.float32(np.nan),
                    "graph_local_curvature": np.float32(np.nan),
                    "graph_slope_guard_active": False,
                    "graph_adverse_step_guard_active": False,
                    "graph_junction_id": "not_in_junction",
                    "graph_junction_role": "not_in_junction",
                    "graph_junction_target_z_m": np.float32(np.nan),
                    "graph_junction_adjustment_z_m": np.float32(0.0),
                    "graph_junction_distance_m": np.float32(np.nan),
                    "graph_junction_influence_weight": np.float32(0.0),
                    "graph_junction_topology_source": "none",
                    "graph_support_strength": np.float32(1.0 if source != "missing" else 0.0),
                    "graph_anchor_spacing_m": np.float32(0.0),
                    "graph_topology_confidence": np.float32(0.0),
                }
            )
    diagnostics = pd.DataFrame.from_records(diag_records)
    diagnostics.attrs["junction_diagnostics"] = []
    return component_fill, metrics, diagnostics


def _build_junction_groups(frame: pd.DataFrame, component_fill: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], float]:
    """Return no junction groups for the package-integrity implementation."""
    return [], 0.0


def summarize_graph_physical_plausibility(diag_frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize graph diagnostics without asserting scientific validation."""
    if diag_frame is None or getattr(diag_frame, "empty", True):
        return {
            "graph_node_count": 0,
            "finite_graph_backbone_count": 0,
            "adverse_step_guard_active_count": 0,
            "slope_guard_active_count": 0,
            "max_abs_local_slope": None,
            "status": "empty",
        }
    z = pd.to_numeric(diag_frame.get("graph_backbone_z_m", pd.Series(np.nan, index=diag_frame.index)), errors="coerce")
    slope = pd.to_numeric(diag_frame.get("graph_local_slope", pd.Series(np.nan, index=diag_frame.index)), errors="coerce")
    finite_slope = slope[np.isfinite(slope)]
    return {
        "graph_node_count": int(len(diag_frame)),
        "finite_graph_backbone_count": int(np.count_nonzero(np.isfinite(z.to_numpy(dtype=float)))),
        "adverse_step_guard_active_count": int(np.count_nonzero(diag_frame.get("graph_adverse_step_guard_active", pd.Series(False, index=diag_frame.index)).astype(bool).to_numpy(dtype=bool))),
        "slope_guard_active_count": int(np.count_nonzero(diag_frame.get("graph_slope_guard_active", pd.Series(False, index=diag_frame.index)).astype(bool).to_numpy(dtype=bool))),
        "max_abs_local_slope": float(np.nanmax(np.abs(finite_slope.to_numpy(dtype=float)))) if len(finite_slope) else None,
        "status": "package_integrity_helper_not_active_solver",
    }
