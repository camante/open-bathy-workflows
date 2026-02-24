#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion.py – Hierarchical fusion of ATL03, ATL24, and extra XYZ training data

This module handles the fusion of multiple bathymetric data sources with:
- XYZ reference data priority (high-quality surveys)
- ATL03/ATL24 consistency checking
- Weighted sample generation for RF training
"""


import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

# Use centralized logging - get logger, don't configure root here
log = logging.getLogger("sdb.fusion")

# -----------------------------------------------------------------------------
# Funnel diagnostics (counts + depth distribution)
# -----------------------------------------------------------------------------

def _depth_funnel_stats(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Return simple depth distribution stats for debugging filters."""
    out: Dict[str, Any] = {"n": int(len(df)) if df is not None else 0}
    if df is None or len(df) == 0 or "depth_m" not in df.columns:
        return out

    d = pd.to_numeric(df["depth_m"], errors="coerce").to_numpy()
    d = d[np.isfinite(d)]
    out["finite_n"] = int(d.size)
    if d.size == 0:
        return out

    # Percentiles for quick sanity checks
    for p in (0, 5, 25, 50, 75, 95, 100):
        out[f"p{p:02d}"] = float(np.nanpercentile(d, p))

    # Fixed histogram 0..40 m @ 1 m bins (counts)
    bins = np.arange(0.0, 41.0, 1.0, dtype=float)
    hist, edges = np.histogram(d, bins=bins)
    out["hist_0_40_1m"] = hist.astype(int).tolist()
    out["hist_edges_0_40_1m"] = edges.astype(float).tolist()
    return out


def _log_funnel(stage: str, df: Optional[pd.DataFrame], rr: Optional[Any] = None) -> None:
    """Log funnel stage count/depth stats and (optionally) record into RunReport."""
    stats = _depth_funnel_stats(df)
    if "finite_n" in stats:
        log.info(
            f"[Funnel] {stage}: n={stats['n']} finite={stats['finite_n']} "
            f"p50={stats.get('p50', float('nan')):.2f} p95={stats.get('p95', float('nan')):.2f} "
            f"max={stats.get('p100', float('nan')):.2f}"
        )
    else:
        log.info(f"[Funnel] {stage}: n={stats['n']}")

    if rr is not None:
        try:
            if hasattr(rr, "add"):
                rr.add(f"funnel.{stage}", stats)
            elif hasattr(rr, "data") and isinstance(rr.data, dict):
                rr.data[f"funnel.{stage}"] = stats
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

# -----------------------------------------------------------------------------
# Helper: local metric projection
# -----------------------------------------------------------------------------

def _build_local_transformer(latitudes: np.ndarray, longitudes: np.ndarray) -> Transformer:
    """
    Build a local metric projection (UTM-style) based on mean lat/lon.
    
    Args:
        latitudes: Array of latitude values
        longitudes: Array of longitude values
    
    Returns:
        Transformer that maps lon/lat (EPSG:4326) to X/Y in meters.
    """
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)

    if latitudes.size == 0 or longitudes.size == 0:
        log.warning("[Fusion] Empty lat/lon arrays in _build_local_transformer; using EPSG:3857 fallback.")
        return Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    mean_lat = float(np.nanmean(latitudes))
    mean_lon = float(np.nanmean(longitudes))

    if not np.isfinite(mean_lat) or not np.isfinite(mean_lon):
        log.warning("[Fusion] Non-finite mean lat/lon; using EPSG:3857 fallback.")
        return Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    try:
        utm_zone = int(((mean_lon + 180.0) % 360) // 6) + 1
        utm_zone = max(1, min(60, utm_zone))  # Clamp to valid range
        if mean_lat >= 0:
            epsg_code = f"EPSG:{32600 + utm_zone}"
        else:
            epsg_code = f"EPSG:{32700 + utm_zone}"
        log.info(f"[Fusion] Using local metric CRS {epsg_code} for KDTree distances.")
        return Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
    except Exception as exc:
        log.warning(f"[Fusion] Failed to build UTM CRS ({exc}); using EPSG:3857.")
        return Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# -----------------------------------------------------------------------------
# 1. Primary conflict check: ATL vs. high-quality XYZ
# -----------------------------------------------------------------------------

def filter_by_reference_data(
    df_atl: pd.DataFrame,
    df_xyz: pd.DataFrame,
    max_dist_m: float = 10.0,
    max_abs_diff_m: float = 0.5,
) -> pd.DataFrame:
    """
    Filter out ATL (03/24) points that conflict with nearby high-quality
    user-provided XYZ reference data.
    """
    if df_atl.empty or df_xyz.empty:
        log.info("[XYZ Filter] Skipping: ATL or XYZ is empty.")
        return df_atl

    _log_funnel('fusion.xyz_filter.func.atl_in', df_atl, None)
    _log_funnel('fusion.xyz_filter.func.xyz_in', df_xyz, None)

    log.info(
        f"[XYZ Filter] Comparing {len(df_atl)} ATL points against {len(df_xyz)} XYZ points "
        f"(max_dist={max_dist_m:.1f} m, max_abs_diff={max_abs_diff_m:.2f} m)."
    )

    # Build local metric transform
    all_lats = np.concatenate([df_atl["latitude"].values, df_xyz["latitude"].values])
    all_lons = np.concatenate([df_atl["longitude"].values, df_xyz["longitude"].values])
    tfm = _build_local_transformer(all_lats, all_lons)

    # Transform coordinates to X/Y in meters
    lon_atl = df_atl["longitude"].to_numpy(np.float64)
    lat_atl = df_atl["latitude"].to_numpy(np.float64)
    lon_xyz = df_xyz["longitude"].to_numpy(np.float64)
    lat_xyz = df_xyz["latitude"].to_numpy(np.float64)

    x_atl, y_atl = tfm.transform(lon_atl, lat_atl)
    x_xyz, y_xyz = tfm.transform(lon_xyz, lat_xyz)

    atl_xy = np.column_stack((x_atl, y_atl))
    xyz_xy = np.column_stack((x_xyz, y_xyz))

    tree_xyz = cKDTree(xyz_xy)

    dist, idx = tree_xyz.query(atl_xy, k=1, distance_upper_bound=max_dist_m)
    collocated_mask = (idx != tree_xyz.n)

    if not np.any(collocated_mask):
        log.info("[XYZ Filter] No ATL points collocated with XYZ data.")
        _log_funnel('fusion.xyz_filter.func.atl_out', df_atl, None)
        return df_atl

    df_atl_collocated = df_atl[collocated_mask].copy().reset_index(drop=True)
    df_xyz_matched = df_xyz.iloc[idx[collocated_mask]].copy().reset_index(drop=True)

    d_atl = df_atl_collocated["depth_m"].to_numpy(np.float64)
    d_xyz = df_xyz_matched["depth_m"].to_numpy(np.float64)
    d_diff = np.abs(d_atl - d_xyz)

    conflict_mask = d_diff > max_abs_diff_m
    n_conflict = int(np.sum(conflict_mask))

    if n_conflict > 0:
        original_indices_to_drop = df_atl[collocated_mask].index[conflict_mask]
        df_atl_final = df_atl.drop(index=original_indices_to_drop).reset_index(drop=True)
        log.info(
            f"[XYZ Filter] Dropped {n_conflict} ATL points conflicting with XYZ "
            f"(abs diff > {max_abs_diff_m:.2f} m). Remaining ATL: {len(df_atl_final)}."
        )
        _log_funnel('fusion.xyz_filter.func.atl_out', df_atl_final, None)
        return df_atl_final

    log.info("[XYZ Filter] All collocated ATL points agree with XYZ within thresholds.")
    _log_funnel('fusion.xyz_filter.func.atl_out', df_atl, None)
    return df_atl


# -----------------------------------------------------------------------------
# 2. Secondary conflict check: ATL03 vs. ATL24
# -----------------------------------------------------------------------------

def check_atl_consistency_and_fuse(
    df_atl03: pd.DataFrame,
    df_atl24: pd.DataFrame,
    max_dist_m: float = 20.0,
    max_abs_diff_m: float = 1.0,
    max_rel_diff: float = 0.15,
    rel_gate_m: float = 0.5,
    require_agreement_depth_min_m: float = 2.0,
    keep_unmatched_atl03: bool = True,
    keep_unmatched_atl24: bool = True,
) -> pd.DataFrame:
    """
    Collocation-aware fusion between ATL03 and ATL24.

    Strategy (requested):
      - KEEP all ATL03 and ATL24 points that are NOT collocated.
      - For collocated pairs:
          * If they AGREE (within thresholds) -> keep ONE point (mean depth) labeled "atl_agreed".
          * If they DISAGREE -> drop BOTH points (remove conflicts only when collocated).

    Notes:
      - Agreement thresholds are enforced for collocated pairs with mean depth >= require_agreement_depth_min_m.
        For shallower pairs (< require_agreement_depth_min_m), agreement is not enforced and BOTH points are kept
        as their native sources (no averaging), to avoid creating "fake" stable depths in the surf/very-shallow zone.
      - Distances are computed in a local UTM-style metric CRS.
    """
    df_atl03 = df_atl03 if df_atl03 is not None else pd.DataFrame()
    df_atl24 = df_atl24 if df_atl24 is not None else pd.DataFrame()

    if df_atl03.empty and df_atl24.empty:
        log.info("[ATL Fusion] Both inputs empty; returning empty DataFrame.")
        return pd.DataFrame(columns=["longitude", "latitude", "depth_m", "source"])

    if df_atl03.empty and not df_atl24.empty:
        log.info(f"[ATL Fusion] ATL03 empty; keeping all ATL24 points: {len(df_atl24)}.")
        return df_atl24.copy()

    if df_atl24.empty and not df_atl03.empty:
        log.info(f"[ATL Fusion] ATL24 empty; keeping all ATL03 points: {len(df_atl03)}.")
        return df_atl03.copy()

    log.info(
        f"[ATL Fusion] Collocation-aware start: ATL03={len(df_atl03)}, ATL24={len(df_atl24)}. "
        f"max_dist={max_dist_m:.1f} m, max_abs_diff={max_abs_diff_m:.2f} m, max_rel_diff={max_rel_diff:.2f}."
    )

    all_lats = np.concatenate([df_atl03["latitude"].values, df_atl24["latitude"].values])
    all_lons = np.concatenate([df_atl03["longitude"].values, df_atl24["longitude"].values])
    tfm = _build_local_transformer(all_lats, all_lons)

    lon03 = df_atl03["longitude"].to_numpy(np.float64)
    lat03 = df_atl03["latitude"].to_numpy(np.float64)
    lon24 = df_atl24["longitude"].to_numpy(np.float64)
    lat24 = df_atl24["latitude"].to_numpy(np.float64)

    x03, y03 = tfm.transform(lon03, lat03)
    x24, y24 = tfm.transform(lon24, lat24)

    atl03_xy = np.column_stack((x03, y03))
    atl24_xy = np.column_stack((x24, y24))

    tree24 = cKDTree(atl24_xy)
    dist_m, idx = tree24.query(atl03_xy, k=1, distance_upper_bound=max_dist_m)
    collocated_mask03 = (idx != tree24.n)

    n_collocated = int(np.sum(collocated_mask03))
    log.info(f"[ATL Fusion] {n_collocated} ATL03 points collocated with ATL24 within {max_dist_m:.1f} m.")

    if n_collocated == 0:
        log.info("[ATL Fusion] No collocated pairs; keeping all ATL03 and ATL24 points.")
        # Diagnostic: show spatial extent of both datasets
        log.info(f"[ATL Fusion] ATL03 lat range: [{df_atl03['latitude'].min():.6f}, {df_atl03['latitude'].max():.6f}]")
        log.info(f"[ATL Fusion] ATL24 lat range: [{df_atl24['latitude'].min():.6f}, {df_atl24['latitude'].max():.6f}]")
        log.info(f"[ATL Fusion] ATL03 lon range: [{df_atl03['longitude'].min():.6f}, {df_atl03['longitude'].max():.6f}]")
        log.info(f"[ATL Fusion] ATL24 lon range: [{df_atl24['longitude'].min():.6f}, {df_atl24['longitude'].max():.6f}]")
        return pd.concat([df_atl03, df_atl24], ignore_index=True)

    df03_c = df_atl03[collocated_mask03].copy().reset_index(drop=False)
    idx24_c = idx[collocated_mask03]
    df24_c = df_atl24.iloc[idx24_c].copy().reset_index(drop=False)

    d03 = df03_c["depth_m"].to_numpy(np.float64)
    d24 = df24_c["depth_m"].to_numpy(np.float64)
    d_avg = (d03 + d24) / 2.0
    d_diff = np.abs(d03 - d24)
    
    # Diagnostic: show depth difference statistics for collocated pairs
    log.info(f"[ATL Fusion] Collocated depth stats:")
    log.info(f"  ATL03 depth: min={d03.min():.2f}, median={np.median(d03):.2f}, max={d03.max():.2f} m")
    log.info(f"  ATL24 depth: min={d24.min():.2f}, median={np.median(d24):.2f}, max={d24.max():.2f} m")
    log.info(f"  Difference:  min={d_diff.min():.2f}, median={np.median(d_diff):.2f}, max={d_diff.max():.2f} m")
    log.info(f"  Pairs with |diff| <= {max_abs_diff_m}m: {np.sum(d_diff <= max_abs_diff_m)} / {len(d_diff)}")

    # Depths in this pipeline are negative-down. Apply thresholds on magnitude.
    d_mag = np.abs(d_avg)

    deep_enough = d_mag >= require_agreement_depth_min_m

    abs_ok = d_diff <= max_abs_diff_m
    eps = 1e-6
    deep_for_rel = d_mag > rel_gate_m
    rel_term = max_rel_diff * np.maximum(d_mag, eps)
    rel_ok = (~deep_for_rel) | (d_diff <= rel_term)

    agree = abs_ok & rel_ok

    keep_agreed = deep_enough & agree
    drop_conflict = deep_enough & (~agree)
    keep_shallow_both = ~deep_enough

    n_keep_agreed = int(np.sum(keep_agreed))
    n_drop_conflict = int(np.sum(drop_conflict))
    n_keep_shallow = int(np.sum(keep_shallow_both))

    log.info(
        f"[ATL Fusion] Collocated pairs: agreed(deep)={n_keep_agreed}, conflicts_dropped(deep)={n_drop_conflict}, "
        f"shallow_pairs_kept_as_both={n_keep_shallow} (<{require_agreement_depth_min_m:.2f} m)."
    )

    out_parts = []

    if n_keep_agreed > 0:
        df_ag = df24_c.loc[keep_agreed].copy()
        df_ag["depth_m_atl03"] = d03[keep_agreed]
        df_ag["depth_m_atl24"] = d24[keep_agreed]
        df_ag["atl03_atl24_dz_m"] = d_diff[keep_agreed]
        df_ag["atl24_match_dist_m"] = dist_m[collocated_mask03][keep_agreed]
        df_ag["depth_m"] = d_avg[keep_agreed]
        df_ag["source"] = "atl_agreed"
        out_parts.append(df_ag)

    if n_keep_shallow > 0:
        df03_s = df03_c.loc[keep_shallow_both].copy()
        df03_s["atl24_match_dist_m"] = dist_m[collocated_mask03][keep_shallow_both]
        df03_s["atl03_atl24_dz_m"] = d_diff[keep_shallow_both]
        out_parts.append(df03_s)

        df24_s = df24_c.loc[keep_shallow_both].copy()
        df24_s["atl24_match_dist_m"] = dist_m[collocated_mask03][keep_shallow_both]
        df24_s["atl03_atl24_dz_m"] = d_diff[keep_shallow_both]
        out_parts.append(df24_s)

    if keep_unmatched_atl03:
        df03_u = df_atl03.loc[~collocated_mask03].copy()
        if not df03_u.empty:
            log.info(f"[ATL Fusion] Keeping {len(df03_u)} unmatched ATL03 points.")
            out_parts.append(df03_u)

    if keep_unmatched_atl24:
        all24_idx = np.arange(len(df_atl24))
        matched24 = np.isin(all24_idx, idx24_c)
        df24_u = df_atl24.loc[~matched24].copy()
        if not df24_u.empty:
            log.info(f"[ATL Fusion] Keeping {len(df24_u)} unmatched ATL24 points.")
            out_parts.append(df24_u)

    if not out_parts:
        log.warning("[ATL Fusion] No points produced after fusion.")
        return pd.DataFrame(columns=["longitude", "latitude", "depth_m", "source"])

    df_out = pd.concat(out_parts, ignore_index=True)
    log.info(f"[ATL Fusion] Final ATL points after collocation-aware fusion: {len(df_out)}.")
    return df_out
# -----------------------------------------------------------------------------
# 3. Schema enforcement & final training frame builder
# -----------------------------------------------------------------------------

def _ensure_training_schema(
    df: pd.DataFrame,
    atl03_weight: float,
    atl24_weight: float,
    xyz_weight: float,
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    required_cols = ["longitude", "latitude", "depth_m", "source"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"_ensure_training_schema: required column '{c}' is missing.")

    # Add missing optional columns
    for opt_col in ["sample_weight", "ws_h", "photon_height", "conf"]:
        if opt_col not in df.columns:
            df[opt_col] = np.nan

    # Normalize source to lowercase
    df["source"] = df["source"].astype(str).str.lower()

    # Apply default weights only where sample_weight is NaN
    m_nan = df["sample_weight"].isna()

    m_atl03 = (df["source"] == "atl03") & m_nan
    m_atl24 = (df["source"] == "atl24") & m_nan
    m_xyz = (df["source"] == "extra_xyz") & m_nan

    if m_atl03.any():
        df.loc[m_atl03, "sample_weight"] = float(atl03_weight)
    if m_atl24.any():
        df.loc[m_atl24, "sample_weight"] = float(atl24_weight)
    if m_xyz.any():
        df.loc[m_xyz, "sample_weight"] = float(xyz_weight)

    # Agreement-only ATL fusion label(s)
    # Old logic: (w1 + w2)/2 -> could be LOWER than atl03_weight
    # New logic: w1 + w2 -> boost high-confidence agreed points
    m_agreed = df["source"].isin(["atl_agreed", "atl03+atl24_agree", "atl03_atl24_agree"]) & m_nan
    if m_agreed.any():
        df.loc[m_agreed, "sample_weight"] = float(atl03_weight) + float(atl24_weight)

    # Final safety: any remaining NaN/inf sample_weight -> 1.0
    df["sample_weight"] = pd.to_numeric(df["sample_weight"], errors="coerce")
    bad = ~np.isfinite(df["sample_weight"].to_numpy())
    if np.any(bad):
        df.loc[bad, "sample_weight"] = 1.0

    # Logging weight distribution for verification (mean weight by source)
    try:
        stats = df.groupby("source")["sample_weight"].mean().to_dict()
        log.info(f"[Fusion] Weight distribution (mean by source): {stats}")
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    # Enforce column order
    col_order = [
        "longitude", "latitude", "depth_m", "source",
        "sample_weight", "ws_h", "photon_height", "conf",
    ]
    extra_cols = [c for c in df.columns if c not in col_order]
    df = df[col_order + extra_cols]

    # === DIAGNOSTIC LOGGING: Data composition by source ===
    if not df.empty and 'source' in df.columns:
        log.info("=" * 60)
        log.info("[Fusion] FINAL DATA COMPOSITION:")
        log.info("=" * 60)
        
        source_counts = df['source'].value_counts()
        total_rows = len(df)
        
        for source in source_counts.index:
            count = source_counts[source]
            pct = 100.0 * count / total_rows
            
            # Get sample weights for this source
            source_mask = df['source'] == source
            weights = df.loc[source_mask, 'sample_weight']
            mean_weight = weights.mean()
            total_weight = weights.sum()
            
            log.info(
                f"  {source:20s}: n={count:6d} ({pct:5.1f}%), "
                f"weight_mean={mean_weight:5.1f}, weight_total={total_weight:8.0f}"
            )
        
        # Total weighted samples
        total_weighted = df['sample_weight'].sum()
        log.info(f"\n  Total samples: {total_rows:,}")
        log.info(f"  Total weighted: {total_weighted:,.0f}")
        
        # Effective sample counts (if all had weight=1.0)
        log.info(f"\n  Effective contribution by source:")
        for source in source_counts.index:
            source_mask = df['source'] == source
            weight_contrib = df.loc[source_mask, 'sample_weight'].sum()
            effective_pct = 100.0 * weight_contrib / total_weighted
            log.info(f"    {source:20s}: {effective_pct:5.1f}% of training influence")
        
        log.info("=" * 60)
    
    return df


def build_fused_training_dataframe(
    atl03_df: Optional[pd.DataFrame],
    atl24_df: Optional[pd.DataFrame],
    xyz_df: Optional[pd.DataFrame] = None,
    xyz_max_dist_m: float = 100.0,        # INCREASED from 30.0 for sparse surveys
    xyz_max_abs_diff_m: float = 1.5,      # INCREASED from 0.5 for realistic tolerance
    atl_max_dist_m: float = 20.0,
    atl_max_abs_diff_m: float = 1.0,
    atl_max_rel_diff: float = 0.15,
    atl_rel_gate_m: float = 0.5,
    atl03_weight: float = 5.0,
    atl24_weight: float = 1.0,
    xyz_weight: float = 10.0,
    rr: Optional[object] = None,
    # NEW in v0.7.0: Adaptive Spatial Sampling (ENABLED BY DEFAULT)
    enable_adaptive_sampling: bool = True,  # Changed from False to True
    sampling_target_points: int = 2000,
    sampling_min_threshold: int = 3000,
    sampling_max_gap_m: float = 100.0,
) -> pd.DataFrame:
    """
    Build fused training dataframe from ATL03, ATL24, and optional XYZ data.
    
    FIXES in v0.6.1:
    - Relaxed XYZ filtering (100m radius, 1.5m tolerance) for sparse surveys
    - Force source normalization to "extra_xyz" for all XYZ data
    - Added comprehensive diagnostic logging
    """
    # >>> FIX: Initialize with empty schema to avoid KeyError if one source is missing <<<
    empty_schema = pd.DataFrame(columns=["longitude", "latitude", "depth_m", "source"])
    
    atl03_df = atl03_df.copy() if (atl03_df is not None and not atl03_df.empty) else empty_schema.copy()
    atl24_df = atl24_df.copy() if (atl24_df is not None and not atl24_df.empty) else empty_schema.copy()
    xyz_df = xyz_df.copy() if (xyz_df is not None and not xyz_df.empty) else empty_schema.copy()

    _log_funnel('fusion.input.atl03', atl03_df, rr)
    try:
        if not atl03_df.empty and 'depth_m' in atl03_df.columns:
            _d = pd.to_numeric(atl03_df['depth_m'], errors='coerce')
            _d = _d[np.isfinite(_d)]
            if len(_d) >= 50:
                dmin = float(np.nanmin(_d.to_numpy()))
                p95 = float(np.nanpercentile(_d.to_numpy(), 95))
                spread95 = p95 - dmin
                if spread95 < 0.25:
                    log.warning(
                        "[ATL03_QC] ATL03 depths are tightly clustered near their shallow limit (min=%.2f, p95=%.2f, spread95=%.2f m). This often indicates bottom-picking clipping / poor penetration. Consider checking ATL03 params (min-depth-atl03, conf/min-bottom-photons, water class) and rely more on ATL24/XYZ for this run.",
                        dmin, p95, spread95,
                    )
        
    except Exception:
        logging.getLogger(__name__).debug("Optional ATL03 QC warning failed", exc_info=True)
    _log_funnel('fusion.input.atl24', atl24_df, rr)
    _log_funnel('fusion.input.xyz', xyz_df, rr)

    # Defensive ATL03 QC: if ATL03 collapses near the shallow floor (failed bottom-pick signature),
    # prevent it from poisoning training when stronger anchors exist.
    try:
        if not atl03_df.empty and 'depth_m' in atl03_df.columns and len(atl03_df) >= 50:
            _d = pd.to_numeric(atl03_df['depth_m'], errors='coerce').to_numpy()
            _d = _d[np.isfinite(_d)]
            if _d.size >= 50:
                _dmin = float(np.nanmin(_d))
                _p50 = float(np.nanpercentile(_d, 50))
                _p95 = float(np.nanpercentile(_d, 95))
                _spread95 = _p95 - _dmin
                _shallow_collapsed = (_spread95 < 0.25) and (_p95 > -1.25)
                if _shallow_collapsed and ((not atl24_df.empty) or (not xyz_df.empty)):
                    log.warning(
                        "[ATL03_QC] Dropping ATL03 from fusion: shallow-floor collapse detected (min=%.2f, p50=%.2f, p95=%.2f, spread95=%.2f m) with ATL24/XYZ anchors available. Check ATL03 params (depth/confidence/refraction/water mask).",
                        _dmin, _p50, _p95, _spread95,
                    )
                    atl03_df = atl03_df.iloc[0:0].copy()
                    _log_funnel('fusion.input.atl03.quarantined', atl03_df, rr)
    except Exception:
        logging.getLogger(__name__).debug("Optional ATL03 quarantine QC failed", exc_info=True)

    log.info(
        f"[Fusion] Starting build_fused_training_dataframe with "
        f"ATL03={len(atl03_df)}, ATL24={len(atl24_df)}, XYZ={len(xyz_df)}."
    )

    # Normalize sensor sources. Keep XYZ provenance if provided (e.g., extra_xyz:<subset>).
    # We only fill missing/empty XYZ source values; we do NOT clobber informative tags.
    if not atl03_df.empty:
        atl03_df["source"] = "atl03"
    if not atl24_df.empty:
        atl24_df["source"] = "atl24"
    if not xyz_df.empty:
        if ("source" not in xyz_df.columns) or xyz_df["source"].isna().all():
            xyz_df["source"] = "extra_xyz"
            log.info(f"[Fusion] XYZ source normalized: {len(xyz_df)} points set to 'extra_xyz'")
        else:
            # Fill only missing entries; preserve existing tags
            n_missing = int(xyz_df["source"].isna().sum())
            if n_missing > 0:
                xyz_df.loc[xyz_df["source"].isna(), "source"] = "extra_xyz"
            log.info(f"[Fusion] XYZ source preserved: {len(xyz_df)} points (missing filled={n_missing})")
    # Step 1: Filter ATL against XYZ, if XYZ provided
    if not xyz_df.empty and (not atl03_df.empty or not atl24_df.empty):
        log.info("[Fusion] Stage 1: Filter ATL03/ATL24 against high-quality XYZ data.")
        parts = [df for df in (atl03_df, atl24_df) if df is not None and not df.empty]
        df_atl_combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        _log_funnel('fusion.xyz_filter.atl_combined.pre', df_atl_combined, rr)
        _log_funnel('fusion.xyz_filter.atl03.pre', atl03_df, rr)
        _log_funnel('fusion.xyz_filter.atl24.pre', atl24_df, rr)
        df_atl_filtered = filter_by_reference_data(
            df_atl=df_atl_combined,
            df_xyz=xyz_df,
            max_dist_m=xyz_max_dist_m,
            max_abs_diff_m=xyz_max_abs_diff_m,
        )

        _log_funnel('fusion.xyz_filter.atl_filtered.post', df_atl_filtered, rr)

        # Re-separate ATL03 and ATL24
        atl03_df = df_atl_filtered[df_atl_filtered["source"].str.lower() == "atl03"].copy()
        atl24_df = df_atl_filtered[df_atl_filtered["source"].str.lower() == "atl24"].copy()
        _log_funnel('fusion.xyz_filter.atl03.post', atl03_df, rr)
        _log_funnel('fusion.xyz_filter.atl24.post', atl24_df, rr)
    else:
        log.info("[Fusion] Skipping XYZ filter: XYZ is empty or no ATL data.")

    # Step 2: ATL03 vs ATL24 fusion
    log.info("[Fusion] Stage 2: ATL03 vs ATL24 consistency check and fusion.")
    _log_funnel('fusion.atl_fuse.atl03.pre', atl03_df, rr)
    _log_funnel('fusion.atl_fuse.atl24.pre', atl24_df, rr)
    atl_fused = check_atl_consistency_and_fuse(
        df_atl03=atl03_df,
        df_atl24=atl24_df,
        max_dist_m=atl_max_dist_m,
        max_abs_diff_m=atl_max_abs_diff_m,
        max_rel_diff=atl_max_rel_diff,
        rel_gate_m=atl_rel_gate_m,
        keep_unmatched_atl03=True,
        keep_unmatched_atl24=True,
    )
    _log_funnel('fusion.atl_fuse.post', atl_fused, rr)

    # Step 3: Final combination
    if not xyz_df.empty:
        final_df = pd.concat([atl_fused, xyz_df], ignore_index=True)
    else:
        final_df = atl_fused

    log.info(f"[Fusion] Stage 3: Combined ATL (fused) + XYZ -> total points: {len(final_df)}.")
    _log_funnel('fusion.combined.post', final_df, rr)

    if final_df.empty:
        log.warning("[Fusion] Final fused DataFrame is empty.")
        return final_df

    # Step 4: Enforce schema + weights
    final_df = _ensure_training_schema(
        final_df,
        atl03_weight=atl03_weight,
        atl24_weight=atl24_weight,
        xyz_weight=xyz_weight,
    )
    log.info("[Fusion] Final training DataFrame is schema-consistent and weighted.")
    _log_funnel('fusion.schema_weighted.final', final_df, rr)
    
    # Step 5: Adaptive Spatial Sampling (NEW in v0.7.0)
    # Apply sophisticated sampling if enabled
    if enable_adaptive_sampling:
        try:
            from spatial_sampling import adaptive_spatial_sample, SamplingConfig, DEFAULT_SOURCE_CONFIGS
            
            # Check if sampling should be applied based on point count
            apply_sampling = len(final_df) >= sampling_min_threshold
            
            if apply_sampling:
                log.info(f"[Fusion] Applying adaptive spatial sampling (input: {len(final_df):,} points)")
                
                # Create sampling config
                sampling_config = SamplingConfig(
                    target_total_points=sampling_target_points,
                    max_gap_m=sampling_max_gap_m,
                    depth_bins=10,
                    grid_high_complexity=15.0,
                    grid_medium_complexity=30.0,
                    grid_low_complexity=60.0
                )
                
                # Apply sampling
                sampled_df, sampling_stats = adaptive_spatial_sample(
                    final_df,
                    source_configs=DEFAULT_SOURCE_CONFIGS,
                    sampling_config=sampling_config
                )
                
                # Log results
                log.info(f"[Fusion] Adaptive sampling complete: {len(final_df):,} → {len(sampled_df):,} points "
                        f"({sampling_stats['reduction_pct']:.1f}% reduction)")
                
                # Save sampling statistics to run report
                if rr is not None:
                    try:
                        if hasattr(rr, "add"):
                            rr.add("fusion.adaptive_sampling", sampling_stats)
                        elif hasattr(rr, "data") and isinstance(rr.data, dict):
                            rr.data["fusion.adaptive_sampling"] = sampling_stats
                    except Exception:
                        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                
                _log_funnel('fusion.sampled.final', sampled_df, rr)
                return sampled_df
            else:
                log.info(f"[Fusion] Skipping adaptive sampling (only {len(final_df):,} points, threshold: {sampling_min_threshold:,})")
                return final_df
                
        except ImportError as e:
            log.warning(f"[Fusion] Adaptive sampling module not available: {e}")
            log.warning(f"[Fusion] Proceeding with unsampled data ({len(final_df):,} points)")
            return final_df
        except Exception as e:
            log.error(f"[Fusion] Adaptive sampling failed: {e}")
            log.error(f"[Fusion] Proceeding with unsampled data ({len(final_df):,} points)")
            import traceback
            traceback.print_exc()
            return final_df
    else:
        log.info(f"[Fusion] Adaptive sampling disabled (use --enable-adaptive-sampling to enable)")
        return final_df

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def _read_optional_csv(path: Optional[str]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if (not p.exists()) or p.stat().st_size == 0:
        log.warning(f"[CLI] CSV not found or empty: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p)
    log.info(f"[CLI] Loaded {len(df)} rows from {p}")
    return df


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fuse ATL03, ATL24, and extra XYZ.")
    parser.add_argument("--atl03-csv", type=str)
    parser.add_argument("--atl24-csv", type=str)
    parser.add_argument("--xyz-csv", type=str)
    parser.add_argument("--out", type=str, default="training_fused.csv")

    parser.add_argument("--xyz-max-dist-m", type=float, default=30.0)
    parser.add_argument("--xyz-max-abs-diff-m", type=float, default=0.5)
    parser.add_argument("--atl-max-dist-m", type=float, default=20.0)
    parser.add_argument("--atl-max-abs-diff-m", type=float, default=1.0)
    parser.add_argument("--atl-max-rel-diff", type=float, default=0.15)
    parser.add_argument("--atl-rel-gate-m", type=float, default=0.5)
    parser.add_argument("--atl03-weight", type=float, default=5.0)
    parser.add_argument("--atl24-weight", type=float, default=1.0)
    parser.add_argument("--xyz-weight", type=float, default=10.0)

    args = parser.parse_args(argv)

    atl03_df = _read_optional_csv(args.atl03_csv)
    atl24_df = _read_optional_csv(args.atl24_csv)
    xyz_df = _read_optional_csv(args.xyz_csv)

    if atl03_df.empty and atl24_df.empty and xyz_df.empty:
        log.error("No input CSVs provided.")
        return 1

    fused_df = build_fused_training_dataframe(
        atl03_df=atl03_df,
        atl24_df=atl24_df,
        xyz_df=xyz_df,
        xyz_max_dist_m=args.xyz_max_dist_m,
        xyz_max_abs_diff_m=args.xyz_max_abs_diff_m,
        atl_max_dist_m=args.atl_max_dist_m,
        atl_max_abs_diff_m=args.atl_max_abs_diff_m,
        atl_max_rel_diff=args.atl_max_rel_diff,
        atl_rel_gate_m=args.atl_rel_gate_m,
        atl03_weight=args.atl03_weight,
        atl24_weight=args.atl24_weight,
        xyz_weight=args.xyz_weight,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_csv(out_path, index=False)
    log.info(f"[CLI] Wrote {len(fused_df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
