#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swot_adjust.py – 1D along-river smoothing + constraint-based depth correction
for SWOT vs SDB samples.

This operates ONLY on the point samples produced by your SWOT diagnostics
(e.g., SWOT_vs_SDB_Samples.csv from vis.plot_swot_vs_sdb).

Core idea:
    - Treat the samples as 1D profiles along-river (s_km or s_m).
    - Compute bed elevation = ws_h_m + depth_raw_m (depth is negative-down).
    - Smooth the bed profile along the river.
    - Enforce min/max water-column thickness constraints:
          min_depth_m <= depth_mag <= max_depth_m
      where depth_mag is the positive water-column thickness.
    - Output adjusted depths at SWOT nodes plus diagnostics.

Expected columns in the input CSV:
    Required:
        - ws_h_m        : water surface elevation [m] at SWOT point
        - depth_raw_m   : SDB depth at that point [m, negative-down]
        - s_km or s_m   : along-river distance (km or m)

    Optional (but handled if present):
        - bed_est_m     : precomputed ws_h_m + depth_raw_m
        - reach_id      : integer/reach ID; if present, smoothing is done per-reach
        - river_id      : alternative grouping ID if reach_id absent

Output CSV columns (extra):
    - bed_raw_m        : raw bed elevation (ws_h_m + depth_raw_m)
    - bed_smooth_m     : smoothed bed elevation
    - depth_adj_m      : adjusted depth [m, negative-down]
    - bed_adj_m        : adjusted bed elevation
    - delta_depth_m    : depth_adj_m - depth_raw_m
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd


log = logging.getLogger("swot_adjust")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize the minimal columns we need.

    We require:
        - ws_h_m
        - depth_raw_m
        - s_km or s_m

    We also add:
        - s_km (if only s_m exists)
        - bed_est_m, if missing (ws_h_m + depth_raw_m)
    """
    required = ["ws_h_m", "depth_raw_m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns {missing} not found in samples CSV. "
            f"Columns available: {list(df.columns)}"
        )

    # Ensure numeric
    df["ws_h_m"] = df["ws_h_m"].astype("float64")
    df["depth_raw_m"] = df["depth_raw_m"].astype("float64")

    # Along-river distance
    if "s_km" in df.columns:
        df["s_km"] = df["s_km"].astype("float64")
    elif "s_m" in df.columns:
        df["s_km"] = df["s_m"].astype("float64") / 1000.0
    else:
        raise ValueError(
            "Input CSV must contain either 's_km' or 's_m' to define along-river distance."
        )

    # Bed estimate
    if "bed_est_m" not in df.columns:
        df["bed_est_m"] = df["ws_h_m"] + df["depth_raw_m"]

    return df


def _moving_average_smoother(
    s_km: np.ndarray,
    bed_raw: np.ndarray,
    smooth_km: float,
) -> np.ndarray:
    """Simple 1D moving-average smoother for bed elevation.

    Parameters
    ----------
    s_km : np.ndarray
        Along-river distance in km; assumed sorted.
    bed_raw : np.ndarray
        Raw bed elevation values.
    smooth_km : float
        Approximate smoothing window in km.

    Returns
    -------
    np.ndarray
        Smoothed bed elevation (same shape as bed_raw).
    """
    if len(bed_raw) < 3 or smooth_km <= 0:
        return bed_raw.copy()

    # Estimate typical spacing
    ds = np.diff(s_km)
    ds_ok = ds[np.isfinite(ds) & (ds > 0)]
    if ds_ok.size == 0:
        return bed_raw.copy()

    median_ds = float(np.median(ds_ok))
    if median_ds <= 0:
        return bed_raw.copy()

    # Number of points in smoothing window
    window_n = max(3, int(round(smooth_km / median_ds)))
    if window_n < 3:
        window_n = 3
    if window_n > len(bed_raw):
        window_n = len(bed_raw)

    kernel = np.ones(window_n, dtype="float64") / float(window_n)

    pad = window_n // 2
    bed_pad = np.pad(bed_raw, pad_width=pad, mode="edge")

    smoothed = np.convolve(bed_pad, kernel, mode="valid")
    if len(smoothed) != len(bed_raw):
        smoothed = smoothed[: len(bed_raw)]

    return smoothed


def _adjust_profile(
    df: pd.DataFrame,
    min_depth_m: float,
    max_depth_m: float,
    smooth_km: float,
) -> Tuple[pd.DataFrame, dict]:
    """Adjust a single 1D profile (one river or reach) in-place.

    Assumes df has columns:
        - ws_h_m, depth_raw_m, s_km

    Returns a new DataFrame and stats dict.
    """
    df = df.copy()

    # Sort by along-river distance
    df.sort_values("s_km", inplace=True)
    df.reset_index(drop=True, inplace=True)

    ws = df["ws_h_m"].to_numpy(dtype="float64")
    d_raw = df["depth_raw_m"].to_numpy(dtype="float64")  # negative-down
    s_km = df["s_km"].to_numpy(dtype="float64")

    # Raw bed elevation
    bed_raw = ws + d_raw

    # Smooth bed
    bed_smooth = _moving_average_smoother(s_km, bed_raw, smooth_km=smooth_km)

    # Water-column thickness implied by smoothed bed (positive if bed below surface)
    d_mag_smooth = ws - bed_smooth

    # Clip to plausible [min_depth, max_depth]
    d_mag_adj = np.clip(d_mag_smooth, min_depth_m, max_depth_m)

    # Negative-down depth
    depth_adj = -d_mag_adj

    # Adjusted bed
    bed_adj = ws + depth_adj

    # Package fields
    df["bed_raw_m"] = bed_raw
    df["bed_smooth_m"] = bed_smooth
    df["depth_adj_m"] = depth_adj
    df["bed_adj_m"] = bed_adj
    df["delta_depth_m"] = depth_adj - d_raw

    # Stats: how much did we change depth magnitude?
    d_mag_raw = np.abs(d_raw)
    d_mag_adj = np.abs(depth_adj)
    # Guard against all-nan
    if np.isfinite(d_mag_raw).any() and np.isfinite(d_mag_adj).any():
        depth_change_rmse = float(
            np.sqrt(np.nanmean((d_mag_adj - d_mag_raw) ** 2))
        )
        depth_change_bias = float(np.nanmean(d_mag_adj - d_mag_raw))
    else:
        depth_change_rmse = float("nan")
        depth_change_bias = float("nan")

    stats = {
        "n_points": int(len(df)),
        "min_depth_m": float(min_depth_m),
        "max_depth_m": float(max_depth_m),
        "smooth_km": float(smooth_km),
        "delta_depth_rmse_mag_m": depth_change_rmse,
        "delta_depth_bias_mag_m": depth_change_bias,
    }

    return df, stats


def adjust_swot_sdb_profiles(
    df_samples: pd.DataFrame,
    min_depth_m: float = 0.5,
    max_depth_m: float = 20.0,
    smooth_km: float = 5.0,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Apply 1D smoothing + depth constraints to one or more river profiles.

    Parameters
    ----------
    df_samples : pd.DataFrame
        Input samples DataFrame.
    min_depth_m : float
        Minimum allowed water-column thickness (m).
    max_depth_m : float
        Maximum allowed water-column thickness (m).
    smooth_km : float
        Smoothing window along river (km).
    group_col : str or None
        Optional grouping column (e.g., 'reach_id' or 'river_id').
        If provided and present, smoothing is applied independently per group.
        If None or not present, all points are treated as a single profile.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of input with additional columns:
            - bed_raw_m, bed_smooth_m, depth_adj_m, bed_adj_m, delta_depth_m
    stats_global : dict
        Aggregated stats (sum of n_points, overall RMSE/bias changes, etc.).
    """
    df = df_samples.copy()
    df = _ensure_core_columns(df)

    # Decide grouping
    if group_col is None:
        if "reach_id" in df.columns:
            group_col = "reach_id"
        elif "river_id" in df.columns:
            group_col = "river_id"

    # If no group column, single-profile treatment
    if group_col is None or group_col not in df.columns:
        log.info("[SWOT_ADJUST] No group_col provided or found; treating as single profile.")
        df_adj, stats = _adjust_profile(df, min_depth_m, max_depth_m, smooth_km)
        stats_global = stats
        return df_adj, stats_global

    # Grouped treatment
    dfs: List[pd.DataFrame] = []
    stats_list: List[dict] = []
    for gid, df_grp in df.groupby(group_col):
        log.info(
            "[SWOT_ADJUST] Adjusting group '%s' with %d points...",
            str(gid),
            len(df_grp),
        )
        df_adj, stats = _adjust_profile(df_grp, min_depth_m, max_depth_m, smooth_km)
        df_adj[group_col] = gid
        dfs.append(df_adj)
        stats["group"] = gid
        stats_list.append(stats)

    df_out = pd.concat(dfs, ignore_index=True)

    # Aggregate stats: simple combined RMSE on depth magnitude changes
    all_raw = []
    all_adj = []
    for gid, df_grp in df_out.groupby(group_col):
        dr = np.abs(df_grp["depth_raw_m"].to_numpy(dtype="float64"))
        da = np.abs(df_grp["depth_adj_m"].to_numpy(dtype="float64"))
        if np.isfinite(dr).any() and np.isfinite(da).any():
            all_raw.append(dr)
            all_adj.append(da)
    if all_raw and all_adj:
        dr_all = np.concatenate(all_raw)
        da_all = np.concatenate(all_adj)
        delta_rmse = float(np.sqrt(np.nanmean((da_all - dr_all) ** 2)))
        delta_bias = float(np.nanmean(da_all - dr_all))
    else:
        delta_rmse = float("nan")
        delta_bias = float("nan")

    stats_global = {
        "n_points": int(len(df_out)),
        "min_depth_m": float(min_depth_m),
        "max_depth_m": float(max_depth_m),
        "smooth_km": float(smooth_km),
        "delta_depth_rmse_mag_m": delta_rmse,
        "delta_depth_bias_mag_m": delta_bias,
        "n_groups": int(len(stats_list)),
    }

    return df_out, stats_global


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "swot_adjust.py – 1D along-river smoothing & depth constraint for SWOT vs SDB samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--samples-csv",
        required=True,
        help="Path to SWOT_vs_SDB_Samples.csv produced by vis.plot_swot_vs_sdb.",
    )
    p.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV for adjusted samples (created/overwritten).",
    )
    p.add_argument(
        "--min-depth",
        type=float,
        default=0.5,
        help="Minimum allowed water depth (m).",
    )
    p.add_argument(
        "--max-depth",
        type=float,
        default=20.0,
        help="Maximum allowed water depth (m).",
    )
    p.add_argument(
        "--smooth-km",
        type=float,
        default=5.0,
        help="Smoothing window along river in km.",
    )
    p.add_argument(
        "--group-col",
        default=None,
        help="Optional grouping column (e.g., 'reach_id' or 'river_id'). "
             "If omitted, will try 'reach_id' then 'river_id', else treat as single profile.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    samples_path = Path(args.samples_csv)
    out_path = Path(args.out_csv)

    if not samples_path.exists():
        raise FileNotFoundError(f"Samples CSV not found: {samples_path}")

    log.info("[SWOT_ADJUST] Reading samples from %s", samples_path)
    df_in = pd.read_csv(samples_path)

    df_out, stats = adjust_swot_sdb_profiles(
        df_in,
        min_depth_m=float(args.min_depth),
        max_depth_m=float(args.max_depth),
        smooth_km=float(args.smooth_km),
        group_col=args.group_col,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    log.info("[SWOT_ADJUST] Wrote adjusted samples to %s", out_path)

    log.info(
        "[SWOT_ADJUST] Global stats: N=%d, groups=%d, min_depth=%.2f, max_depth=%.2f, "
        "smooth_km=%.2f, delta_depth_RMSE_mag=%.3f m, bias_mag=%.3f m",
        stats["n_points"],
        stats.get("n_groups", 1),
        stats["min_depth_m"],
        stats["max_depth_m"],
        stats["smooth_km"],
        stats["delta_depth_rmse_mag_m"],
        stats["delta_depth_bias_mag_m"],
    )


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
