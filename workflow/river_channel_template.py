#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
river_channel_template.py – Normalized cross-section template learning and synthesis.

Implements the "learn shape → propagate → mold" paradigm for river channel inference:

  1. Extract normalized XS profiles from measured (sounding-supported) sections
  2. Compute a mean template shape (depth/max_depth vs position/width) with spread
  3. Fit a locally-calibrated width→max_depth relation from measured sections
  4. Synthesize bed profiles at unmapped sections using template + depth prior
  5. Evaluate template quality via hold-one-out cross-validation

Design principles:
  - Template is secondary guidance, not authoritative truth
  - Authoritative soundings deform the template; template never overrides soundings
  - Falls back gracefully to Leopold & Maddock when local calibration is too sparse
  - All outputs carry uncertainty / confidence metadata
  - Pure numpy/pandas – no heavy geospatial imports required for core logic

References:
  - Leopold & Maddock (1953): hydraulic geometry scaling (D ∝ W^b)
  - Downstream hydraulic geometry: width, depth, velocity scale with discharge
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("river_channel_template")


def _valid_dmax_value(value: float) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(v) and (0.0 < v < 200.0)

def _valid_bed_elevation(value: float) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(v) and (-1.0e3 <= v <= 1.0e3) and (abs(v + 999999.0) > 1.0) and (abs(v - 999999.0) > 1.0)


# Heavy geo imports are lazy — loaded only inside functions that need spatial
# filtering (estuary boundary, junction points).  Core template logic is pure
# numpy/pandas so it works in lightweight test environments.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TemplateConfig:
    """Configuration for channel template extraction and synthesis."""

    # Normalized profile resolution (number of bins across [0, 1] width)
    n_bins: int = 50

    # Minimum requirements for template extraction
    min_measured_xs: int = 3         # minimum XS with soundings to build a template
    min_sounding_depth_m: float = 0.3  # ignore shallower soundings (noise floor)

    # Width → max-depth power law fallback (Leopold & Maddock 1953)
    fallback_a: float = 0.18
    fallback_b: float = 0.50

    # Local fit constraints
    fit_min_xs: int = 5              # minimum XS for local power-law fit
    fit_min_depth_span_m: float = 1.0  # minimum range of observed max depths
    fit_b_range: Tuple[float, float] = (0.20, 0.80)  # plausible exponent range
    fit_a_range: Tuple[float, float] = (0.01, 2.00)  # plausible coefficient range

    # Depth clamps (same semantics as InferConfig)
    dmin_m: float = 0.50
    dmax_m: float = 15.0

    # Template shape fallback: parabolic / U-shaped profile when no measured
    # sections are available at all
    fallback_bottom_frac: float = 0.30

    # Training filter / propagation controls
    estuary_boundary_buffer_m: float = 500.0
    junction_buffer_m: float = 120.0
    width_depth_ratio_max: Optional[float] = None
    distance_sigma_m: float = 2000.0

    # Leave-one-out quality gate for the local width→depth fit. The learned
    # template shape is preserved, but the depth scaling downgrades to fallback
    # when the local fit is not stable enough.
    loo_max_rmse_norm: float = 0.25
    loo_max_dmax_error_m: float = 1.5

    # Diagnostic outputs
    write_diagnostics: bool = True


# ---------------------------------------------------------------------------
# Normalized profile extraction
# ---------------------------------------------------------------------------

def _normalize_xs_profile(
    dist_m: np.ndarray,
    z_bed: np.ndarray,
    bank_left_dist_m: float,
    bank_right_dist_m: float,
    wse_m: float,
    n_bins: int = 50,
    min_depth_m: float = 0.3,
) -> Optional[Dict]:
    """Extract a single normalized XS profile from raw XS data.

    Returns a dict with:
      - norm_pos: array of normalized lateral positions [0, 1] (left bank = 0)
      - norm_depth: array of normalized depths [0, 1] (surface = 0, thalweg = 1)
      - width_m: bank-to-bank width
      - max_depth_m: maximum depth below WSE
      - wse_m: water surface elevation used
      - n_valid: number of valid depth samples
      - asymmetry: signed offset of thalweg from center (-0.5..+0.5)

    Returns None if the profile is unusable.
    """
    dist_m = np.asarray(dist_m, dtype="float64")
    z_bed = np.asarray(z_bed, dtype="float64")

    if not (np.isfinite(bank_left_dist_m) and np.isfinite(bank_right_dist_m)):
        return None
    if not np.isfinite(wse_m):
        return None

    left = min(bank_left_dist_m, bank_right_dist_m)
    right = max(bank_left_dist_m, bank_right_dist_m)
    W = right - left
    if W <= 0:
        return None

    # Mask to bank-to-bank region
    inside = (dist_m >= left) & (dist_m <= right) & np.isfinite(z_bed)
    if np.any(inside):
        inside &= np.vectorize(_valid_bed_elevation, otypes=[bool])(z_bed)
    if inside.sum() < 5:
        return None

    d = dist_m[inside]
    z = z_bed[inside]

    # Compute depth below WSE (positive downward)
    depth = wse_m - z
    # Only keep physically plausible depths and reject sentinel contamination
    valid = (depth >= min_depth_m) & np.isfinite(depth) & (depth <= 1.0e3)
    if valid.sum() < 3:
        return None

    max_depth = float(np.nanmax(depth[valid]))
    if max_depth < min_depth_m:
        return None

    # Normalize position: 0 = left bank, 1 = right bank
    norm_pos = (d - left) / W

    # Normalize depth: 0 = surface, 1 = thalweg
    norm_depth = np.where(depth > 0, depth / max_depth, 0.0)
    norm_depth = np.clip(norm_depth, 0.0, 1.0)

    # Bin into regular grid for averaging across sections
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned_depth = np.full(n_bins, np.nan, dtype="float64")

    for b in range(n_bins):
        mask = (norm_pos >= bin_edges[b]) & (norm_pos < bin_edges[b + 1])
        if b == n_bins - 1:
            mask = (norm_pos >= bin_edges[b]) & (norm_pos <= bin_edges[b + 1])
        vals = norm_depth[mask]
        if len(vals) > 0:
            binned_depth[b] = float(np.nanmedian(vals))

    n_valid_bins = int(np.isfinite(binned_depth).sum())
    if n_valid_bins < n_bins * 0.3:
        return None

    # Asymmetry: where is the deepest point relative to center?
    thalweg_idx = np.nanargmax(binned_depth)
    thalweg_pos = bin_centers[thalweg_idx]
    asymmetry = thalweg_pos - 0.5  # negative = deeper toward left bank

    return {
        "norm_pos": bin_centers,
        "norm_depth": binned_depth,
        "width_m": float(W),
        "max_depth_m": float(max_depth),
        "wse_m": float(wse_m),
        "n_valid_bins": n_valid_bins,
        "asymmetry": float(asymmetry),
    }


def _estimate_wse_from_xs_profile(
    dist_m: np.ndarray,
    z_bed: np.ndarray,
    bank_left_dist_m: float,
    bank_right_dist_m: float,
    bank_left_z_m: float,
    bank_right_z_m: float,
    center_frac: float = 0.15,
    quantile: float = 0.10,
) -> float:
    """Estimate water surface elevation from a cross-section profile.

    Priority:
      1. Bank pick elevations (bank_left_z_m, bank_right_z_m) — these are the
         water-land boundary elevations from xs_builder's corridor-aware bank
         picking, which is the best proxy for WSE.
      2. Profile-based: highest z at the immediate bank-pick locations in the
         sampled profile (fallback when bank picks are missing).

    IMPORTANT: Do NOT use terrain z values "near the bank edges" because for
    CUDEM / topobathy DEMs, those are terrain elevations well above the actual
    water surface.  The xs_builder bank picks specifically target the
    water-land transition, not the bank top.
    """
    # Method 1 (primary): use the xs_builder bank pick elevations directly.
    # These are picked at the water-land boundary and are the best WSE proxy.
    bank_z = []
    if np.isfinite(bank_left_z_m):
        bank_z.append(float(bank_left_z_m))
    if np.isfinite(bank_right_z_m):
        bank_z.append(float(bank_right_z_m))
    if bank_z:
        return float(min(bank_z))

    # Method 2 (fallback): look at z at the bank-pick distance locations
    # in the sampled profile.  Only used when bank_z is unavailable.
    dist_m = np.asarray(dist_m, dtype="float64")
    z_bed = np.asarray(z_bed, dtype="float64")

    left = min(bank_left_dist_m, bank_right_dist_m) if (
        np.isfinite(bank_left_dist_m) and np.isfinite(bank_right_dist_m)
    ) else np.nan
    right = max(bank_left_dist_m, bank_right_dist_m) if np.isfinite(left) else np.nan
    W = right - left if np.isfinite(left) else 0.0

    if W > 0:
        # Sample z very close to the bank-pick locations (within 2m or 3% of width)
        tol = max(min(W * 0.03, 5.0), 2.0)
        near_left = np.abs(dist_m - left) <= tol
        near_right = np.abs(dist_m - right) <= tol
        edge_z = z_bed[(near_left | near_right) & np.isfinite(z_bed)]
        if len(edge_z) >= 1:
            return float(np.nanmin(edge_z))

    return float("nan")


def extract_normalized_profiles(
    xs_lines: pd.DataFrame,
    xs_points: pd.DataFrame,
    sounding_calibration: Optional[pd.DataFrame] = None,
    cfg: Optional[TemplateConfig] = None,
    wse_override: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Extract normalized profiles from all measured (sounding-supported) XS.

    Parameters
    ----------
    xs_lines : DataFrame
        Must have xs_id, bank_left_dist_m, bank_right_dist_m, bank_left_z_m,
        bank_right_z_m, s_center_m, river_id, component_id.
    xs_points : DataFrame
        Must have xs_id, dist_m (or s_m), z_dem, z_topo.
    sounding_calibration : DataFrame, optional
        If provided, must have xs_id and calib_n >= 1.  Only XS present in this
        table (with calib_n >= 1) are treated as "measured".  If None, all XS
        with finite bank picks are used (useful for testing, but not the
        recommended production path).
    cfg : TemplateConfig

    Returns
    -------
    profiles : list of dicts
        Each dict from _normalize_xs_profile, plus xs_id/river_id/component_id/station.
    summary : DataFrame
        Per-XS summary (xs_id, width, max_depth, asymmetry, n_valid_bins).
    """
    if cfg is None:
        cfg = TemplateConfig()

    # Determine which XS are "measured" (have sounding support)
    if sounding_calibration is not None and not sounding_calibration.empty:
        calib = sounding_calibration.copy()
        calib["calib_n"] = pd.to_numeric(calib.get("calib_n", 0), errors="coerce").fillna(0)
        measured_ids = set(calib.loc[calib["calib_n"] >= 1, "xs_id"].astype(str).tolist())
    else:
        measured_ids = None  # use all

    pts = xs_points.copy()
    # Normalize profile IDs up front so xs_lines / calibration tables / point tables
    # cannot silently mismatch on numeric-vs-string xs_id representations.
    pts["xs_id"] = pts.get("xs_id", pd.Series(index=pts.index, dtype=object)).astype(str)
    dist_col = "dist_m" if "dist_m" in pts.columns else "s_m"
    pts["_dist"] = pd.to_numeric(pts[dist_col], errors="coerce")
    # Prefer z_topo where available, fall back to z_dem
    z_topo = pd.to_numeric(pts.get("z_topo"), errors="coerce") if "z_topo" in pts.columns else pd.Series(np.nan, index=pts.index)
    z_dem = pd.to_numeric(pts.get("z_dem"), errors="coerce") if "z_dem" in pts.columns else pd.Series(np.nan, index=pts.index)
    pts["_z"] = np.where(np.isfinite(z_topo), z_topo, z_dem)

    xs_lines = xs_lines.copy()
    xs_lines["xs_id"] = xs_lines.get("xs_id", pd.Series(index=xs_lines.index, dtype=object)).astype(str)
    grouped = pts.groupby("xs_id", sort=False)

    profiles: List[Dict] = []
    rows: List[Dict] = []

    for _, xsl in xs_lines.iterrows():
        xsid = str(xsl.get("xs_id", ""))
        if measured_ids is not None and xsid not in measured_ids:
            continue
        if xsid not in grouped.groups:
            continue

        grp = grouped.get_group(xsid)
        d = grp["_dist"].to_numpy(dtype="float64")
        z = grp["_z"].to_numpy(dtype="float64")

        bl = float(xsl.get("bank_left_dist_m", np.nan))
        br = float(xsl.get("bank_right_dist_m", np.nan))
        blz = float(xsl.get("bank_left_z_m", np.nan))
        brz = float(xsl.get("bank_right_z_m", np.nan))

        # WSE estimation priority:
        # 1. wse_override dict (from infer_bathy's _estimate_wse_from_profile)
        # 2. Explicit wse_m column (from infer_bathy or SWOT)
        # 3. Bank pick elevations (min of bank_left_z, bank_right_z)
        # 4. Profile-based fallback
        wse = np.nan
        if wse_override is not None and xsid in wse_override:
            wse = float(wse_override[xsid])
        if not np.isfinite(wse):
            wse = float(xsl.get("wse_m", np.nan))
        if not np.isfinite(wse):
            wse = _estimate_wse_from_xs_profile(
                d, z, bl, br, blz, brz
            )
        if not np.isfinite(wse):
            continue

        result = _normalize_xs_profile(
            d, z, bl, br, wse,
            n_bins=cfg.n_bins,
            min_depth_m=cfg.min_sounding_depth_m,
        )
        if result is None:
            continue

        result["xs_id"] = xsid
        result["river_id"] = str(xsl.get("river_id", ""))
        result["component_id"] = int(xsl.get("component_id", -1)) if pd.notna(xsl.get("component_id", np.nan)) else -1
        result["s_center_m"] = float(xsl.get("s_center_m", np.nan))
        profiles.append(result)

        rows.append({
            "xs_id": xsid,
            "river_id": result["river_id"],
            "component_id": result["component_id"],
            "s_center_m": result["s_center_m"],
            "width_m": result["width_m"],
            "max_depth_m": result["max_depth_m"],
            "asymmetry": result["asymmetry"],
            "n_valid_bins": result["n_valid_bins"],
        })

    summary = pd.DataFrame(rows)
    log.info(
        "Extracted %d normalized profiles from %d candidate XS (measured_ids=%s).",
        len(profiles),
        len(xs_lines),
        "all" if measured_ids is None else str(len(measured_ids)),
    )
    return profiles, summary


def _xs_centers(xs_lines: pd.DataFrame):
    """Return XS center points as a GeoDataFrame, or None if geometry unavailable."""
    if xs_lines is None or len(xs_lines) == 0 or "geometry" not in xs_lines.columns:
        return None
    try:
        import geopandas as gpd
    except ImportError:
        log.debug("geopandas not available; skipping XS center point extraction")
        return None
        return None
    g = xs_lines[["xs_id", "geometry"]].copy()
    g["xs_id"] = g["xs_id"].astype(str)
    try:
        geom = [geom.interpolate(0.5, normalized=True) if geom is not None and not geom.is_empty else geom for geom in g.geometry]
    except Exception:
        geom = [geom.centroid if geom is not None and not geom.is_empty else geom for geom in g.geometry]
    return gpd.GeoDataFrame(g.drop(columns=["geometry"]), geometry=geom, crs=getattr(xs_lines, "crs", None))


def _best_effort_project_geom(geom, src_crs, dst_crs):
    if geom is None or getattr(geom, "is_empty", True) or src_crs is None or dst_crs is None:
        return geom
    try:
        from pyproj import CRS, Transformer
        from shapely.ops import transform as shp_transform
    except ImportError:
        return geom
        return geom
    c1 = CRS.from_user_input(src_crs)
    c2 = CRS.from_user_input(dst_crs)
    if c1 == c2:
        return geom
    tr = Transformer.from_crs(c1, c2, always_xy=True)
    return shp_transform(tr.transform, geom)


def _load_estuary_boundary(estuary_clip_mask: Optional[Path], target_crs=None):
    if estuary_clip_mask is None:
        return None
    path = Path(estuary_clip_mask)
    if not path.exists():
        return None
    try:
        import rasterio
        from rasterio.features import shapes as rio_shapes
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError:
        log.debug("geo libraries not available; skipping estuary boundary loading")
        return None
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            mask = np.isfinite(arr) & (arr > 0)
            if not mask.any():
                return None
            polys = [shape(geom) for geom, val in rio_shapes(mask.astype("uint8"), mask=mask, transform=src.transform) if int(val) == 1]
            if not polys:
                return None
            boundary = unary_union(polys).boundary
            return _best_effort_project_geom(boundary, src.crs, target_crs)
    except Exception:
        log.debug("failed to load estuary boundary", exc_info=True)
        return None


def _load_junction_points(river_gpkg: Optional[Path], target_crs=None):
    if river_gpkg is None:
        return None
    path = Path(river_gpkg)
    if not path.exists():
        return None
    try:
        import fiona
        import geopandas as gpd
    except ImportError:
        log.debug("geo libraries not available; skipping junction point loading")
        return None
    try:
        layers = set(fiona.listlayers(path))
    except Exception:
        log.debug("failed to list layers for %s", path, exc_info=True)
        return None
    for layer in ("graph_nodes", "nodes", "river_nodes"):
        if layer not in layers:
            continue
        try:
            gdf = gpd.read_file(path, layer=layer)
            if gdf.empty:
                continue
            deg_col = next((c for c in gdf.columns if str(c).lower() == "degree"), None)
            if deg_col is None:
                continue
            deg = pd.to_numeric(gdf[deg_col], errors="coerce")
            gdf = gdf.loc[deg >= 3, [deg_col, "geometry"]].copy()
            if gdf.empty:
                return None
            if target_crs is not None and getattr(gdf, "crs", None) is not None and str(gdf.crs) != str(target_crs):
                gdf = gdf.to_crs(target_crs)
            return gdf
        except Exception:
            log.debug("failed to load junction points from %s:%s", path, layer, exc_info=True)
    return None


def select_template_training_sections(
    xs_lines: pd.DataFrame,
    profiles: List[Dict],
    summary: pd.DataFrame,
    cfg: Optional[TemplateConfig] = None,
    estuary_clip_mask: Optional[Path] = None,
    river_gpkg: Optional[Path] = None,
) -> Tuple[List[Dict], pd.DataFrame, Dict, pd.DataFrame]:
    if cfg is None:
        cfg = TemplateConfig()
    if summary is None or summary.empty or not profiles:
        empty_cols = list(summary.columns) if summary is not None else []
        empty = pd.DataFrame(columns=empty_cols)
        return [], empty.copy(), {"n_input": int(len(profiles)), "n_selected": 0}, empty

    work = summary.copy()
    work["xs_id"] = work["xs_id"].astype(str)
    work["keep_for_template"] = True
    work["reject_reason"] = ""

    centers = _xs_centers(xs_lines)
    if centers is not None and not centers.empty:
        import geopandas as gpd
        work = work.merge(centers[["xs_id", "geometry"]], on="xs_id", how="left")
        work = gpd.GeoDataFrame(work, geometry="geometry", crs=centers.crs)

    ratios = pd.to_numeric(work.get("width_m"), errors="coerce") / pd.to_numeric(work.get("max_depth_m"), errors="coerce").clip(lower=max(cfg.min_sounding_depth_m, 1e-6))
    work["width_depth_ratio"] = ratios
    ratio_thresh = cfg.width_depth_ratio_max
    finite_ratios = ratios[np.isfinite(ratios)]
    if ratio_thresh is None and finite_ratios.size >= 4:
        q1 = float(np.nanpercentile(finite_ratios, 25))
        q3 = float(np.nanpercentile(finite_ratios, 75))
        iqr = max(0.0, q3 - q1)
        ratio_thresh = q3 + 1.5 * iqr
    if ratio_thresh is not None and np.isfinite(ratio_thresh):
        bad = np.isfinite(ratios) & (ratios > float(ratio_thresh))
        work.loc[bad, "keep_for_template"] = False
        work.loc[bad, "reject_reason"] = np.where(work.loc[bad, "reject_reason"].astype(str) == "", "width_depth", work.loc[bad, "reject_reason"].astype(str) + ";width_depth")

    estuary_boundary = _load_estuary_boundary(estuary_clip_mask, target_crs=getattr(work, "crs", None))
    if estuary_boundary is not None and "geometry" in work.columns:
        dist = work.geometry.apply(lambda g: g.distance(estuary_boundary) if g is not None and not g.is_empty else np.nan)
        work["dist_to_estuary_boundary_m"] = pd.to_numeric(dist, errors="coerce")
        bad = np.isfinite(work["dist_to_estuary_boundary_m"]) & (work["dist_to_estuary_boundary_m"] <= float(cfg.estuary_boundary_buffer_m))
        work.loc[bad, "keep_for_template"] = False
        work.loc[bad, "reject_reason"] = np.where(work.loc[bad, "reject_reason"].astype(str) == "", "estuary_boundary", work.loc[bad, "reject_reason"].astype(str) + ";estuary_boundary")

    junctions = _load_junction_points(river_gpkg, target_crs=getattr(work, "crs", None))
    if junctions is not None and not junctions.empty and "geometry" in work.columns:
        from shapely.ops import unary_union
        j_union = unary_union([g for g in junctions.geometry if g is not None and not g.is_empty])
        dist = work.geometry.apply(lambda g: g.distance(j_union) if g is not None and not g.is_empty else np.nan)
        work["dist_to_junction_m"] = pd.to_numeric(dist, errors="coerce")
        bad = np.isfinite(work["dist_to_junction_m"]) & (work["dist_to_junction_m"] <= float(cfg.junction_buffer_m))
        work.loc[bad, "keep_for_template"] = False
        work.loc[bad, "reject_reason"] = np.where(work.loc[bad, "reject_reason"].astype(str) == "", "junction", work.loc[bad, "reject_reason"].astype(str) + ";junction")

    keep_ids = set(work.loc[work["keep_for_template"], "xs_id"].astype(str).tolist())
    selected_profiles = [p for p in profiles if str(p.get("xs_id", "")) in keep_ids]
    selected_summary = pd.DataFrame(work.drop(columns=["geometry"], errors="ignore"))
    selected_summary = selected_summary.loc[selected_summary["keep_for_template"]].copy()

    info = {
        "n_input": int(len(profiles)),
        "n_selected": int(len(selected_profiles)),
        "n_rejected": int(len(profiles) - len(selected_profiles)),
        "estuary_boundary_buffer_m": float(cfg.estuary_boundary_buffer_m),
        "junction_buffer_m": float(cfg.junction_buffer_m),
        "width_depth_ratio_max": float(ratio_thresh) if ratio_thresh is not None and np.isfinite(ratio_thresh) else None,
        "reject_counts": {str(k): int(v) for k, v in work.loc[~work["keep_for_template"], "reject_reason"].fillna("unknown").astype(str).value_counts().items()},
    }
    profile_diagnostics = pd.DataFrame(work.drop(columns=["geometry"], errors="ignore")).copy()
    return selected_profiles, selected_summary, info, profile_diagnostics


# ---------------------------------------------------------------------------
# Mean template computation
# ---------------------------------------------------------------------------

@dataclass
class ChannelTemplate:
    """A learned channel cross-section template."""

    # Normalized positions (n_bins,) — always [0, 1]
    norm_pos: np.ndarray

    # Mean normalized depth at each position
    mean_depth: np.ndarray

    # Standard deviation of normalized depth at each position
    std_depth: np.ndarray

    # Number of profiles that contributed to each bin
    n_profiles_per_bin: np.ndarray

    # Total number of measured XS used
    n_profiles: int = 0

    # Mean asymmetry across profiles
    mean_asymmetry: float = 0.0
    std_asymmetry: float = 0.0

    # Width → max-depth power law: Dmax = a * W^b
    depth_a: float = 0.18
    depth_b: float = 0.50
    depth_fit_r2: float = 0.0
    depth_fit_n: int = 0
    depth_fit_source: str = "fallback_leopold_maddock"

    # Metadata
    river_id: str = ""
    component_id: int = -1

    # Effective shape exponent for skeleton integration (r^exp depth profile)
    shape_exponent: float = 0.5

    def predict_dmax(self, width_m: float, dmin: float = 0.5, dmax: float = 15.0) -> float:
        """Predict maximum depth from width using the fitted/fallback relation."""
        if not np.isfinite(width_m) or width_m <= 0:
            return float("nan")
        d = self.depth_a * (width_m ** self.depth_b)
        return float(np.clip(d, dmin, dmax))

    def synthesize_bed(
        self,
        dist_from_left: np.ndarray,
        width_m: float,
        wse_m: float,
        dmax_override: Optional[float] = None,
        dmin: float = 0.5,
        dmax_cap: float = 15.0,
    ) -> np.ndarray:
        """Synthesize a bed elevation profile using this template.

        Parameters
        ----------
        dist_from_left : array
            Distance from left bank (meters) for each output point.
        width_m : float
            Bank-to-bank width.
        wse_m : float
            Water surface elevation.
        dmax_override : float, optional
            If provided, use this max depth instead of the width→depth relation.
        dmin, dmax_cap : float
            Depth clamps.

        Returns
        -------
        z_bed : array
            Bed elevation (meters) at each point. NaN outside the channel.
        """
        dist = np.asarray(dist_from_left, dtype="float64")
        out = np.full_like(dist, np.nan, dtype="float64")

        if not (np.isfinite(width_m) and width_m > 0 and np.isfinite(wse_m)):
            return out

        Dmax = dmax_override if (dmax_override is not None and np.isfinite(dmax_override)) else self.predict_dmax(width_m, dmin, dmax_cap)
        if not np.isfinite(Dmax) or Dmax <= 0:
            return out

        # Normalize position
        norm_pos = dist / width_m
        inside = (norm_pos >= 0.0) & (norm_pos <= 1.0)

        if not inside.any():
            return out

        # Interpolate template to get normalized depth at each position
        # Use the mean template; fill NaN bins with linear interpolation
        template_depth = self.mean_depth.copy()
        finite_mask = np.isfinite(template_depth)
        if finite_mask.sum() < 2:
            # Template too sparse, fall back to simple cosine
            x = np.clip(norm_pos[inside], 0.0, 1.0)
            depth_frac = 0.5 * (1.0 - np.cos(np.pi * np.sin(np.pi * x)))
            out[inside] = wse_m - depth_frac * Dmax
            return out

        # Interpolate NaN gaps in template
        if not finite_mask.all():
            template_depth = np.interp(
                self.norm_pos,
                self.norm_pos[finite_mask],
                template_depth[finite_mask],
            )

        # Ensure template touches zero at edges (bank constraint)
        template_depth[0] = 0.0
        template_depth[-1] = 0.0

        # Interpolate onto output positions
        depth_frac = np.interp(norm_pos[inside], self.norm_pos, template_depth)
        depth_frac = np.clip(depth_frac, 0.0, 1.0)

        out[inside] = wse_m - depth_frac * Dmax
        return out

    def to_dict(self) -> Dict:
        """Serialize to JSON-safe dict."""
        return {
            "norm_pos": self.norm_pos.tolist(),
            "mean_depth": self.mean_depth.tolist(),
            "std_depth": self.std_depth.tolist(),
            "n_profiles_per_bin": self.n_profiles_per_bin.tolist(),
            "n_profiles": self.n_profiles,
            "mean_asymmetry": self.mean_asymmetry,
            "std_asymmetry": self.std_asymmetry,
            "depth_a": self.depth_a,
            "depth_b": self.depth_b,
            "depth_fit_r2": self.depth_fit_r2,
            "depth_fit_n": self.depth_fit_n,
            "depth_fit_source": self.depth_fit_source,
            "shape_exponent": self.shape_exponent,
            "river_id": self.river_id,
            "component_id": self.component_id,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ChannelTemplate":
        return cls(
            norm_pos=np.asarray(d["norm_pos"], dtype="float64"),
            mean_depth=np.asarray(d["mean_depth"], dtype="float64"),
            std_depth=np.asarray(d["std_depth"], dtype="float64"),
            n_profiles_per_bin=np.asarray(d["n_profiles_per_bin"], dtype="int64"),
            n_profiles=int(d.get("n_profiles", 0)),
            mean_asymmetry=float(d.get("mean_asymmetry", 0.0)),
            std_asymmetry=float(d.get("std_asymmetry", 0.0)),
            depth_a=float(d.get("depth_a", 0.18)),
            depth_b=float(d.get("depth_b", 0.50)),
            depth_fit_r2=float(d.get("depth_fit_r2", 0.0)),
            depth_fit_n=int(d.get("depth_fit_n", 0)),
            depth_fit_source=str(d.get("depth_fit_source", "")),
            shape_exponent=float(d.get("shape_exponent", 0.5)),
            river_id=str(d.get("river_id", "")),
            component_id=int(d.get("component_id", -1)),
        )


@dataclass
class ChannelTemplateBuildResult:
    template: Optional[ChannelTemplate]
    raw_profiles: List[Dict]
    selected_profiles: List[Dict]
    raw_summary: pd.DataFrame
    selected_summary: pd.DataFrame
    profile_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    loo_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    filter_info: Dict = field(default_factory=dict)


def _write_channel_template_diagnostics(
    out_dir: Optional[Path],
    raw_summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
    profile_diagnostics: pd.DataFrame,
    filter_info: Dict,
    loo_results: Optional[pd.DataFrame] = None,
) -> None:
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if raw_summary is not None and not raw_summary.empty:
        raw_summary.to_csv(out_dir / "channel_template_profiles_raw.csv", index=False)
        raw_summary.to_csv(out_dir / "channel_template_profiles.csv", index=False)
    if selected_summary is not None and not selected_summary.empty:
        selected_summary.to_csv(out_dir / "channel_template_profiles_selected.csv", index=False)
        selected_summary.to_csv(out_dir / "channel_template_profiles.csv", index=False)
    if profile_diagnostics is not None and not profile_diagnostics.empty:
        profile_diagnostics.to_csv(out_dir / "channel_template_profile_diagnostics.csv", index=False)
    if filter_info:
        (out_dir / "channel_template_filter.json").write_text(json.dumps(filter_info, indent=2, default=str) + "\n")
    if loo_results is not None and not loo_results.empty:
        loo_results.to_csv(out_dir / "channel_template_loo.csv", index=False)


def diagnose_template_profile_usability(
    profiles: List[Dict],
    cfg: Optional[TemplateConfig] = None,
) -> pd.DataFrame:
    """Return per-profile usability diagnostics for mean-template aggregation."""
    if cfg is None:
        cfg = TemplateConfig()
    n_bins = int(cfg.n_bins)
    min_profile_bins = max(5, int(np.ceil(n_bins / 4.0)))
    rows: List[Dict] = []
    for p in profiles:
        xs_id = str(p.get("xs_id", ""))
        nd = p.get("norm_depth")
        reason = "usable"
        n_finite = 0
        bin_count = np.nan
        if nd is None:
            reason = "missing_norm_depth"
        else:
            nd = np.asarray(nd, dtype="float64")
            bin_count = int(nd.shape[0]) if nd.ndim == 1 else np.nan
            if nd.ndim != 1 or nd.shape[0] != n_bins:
                reason = "wrong_bin_count"
            else:
                n_finite = int(np.isfinite(nd).sum())
                if n_finite < min_profile_bins:
                    reason = "insufficient_finite_bins"
        rows.append({
            "xs_id": xs_id,
            "template_mean_usable": bool(reason == "usable"),
            "template_mean_reject_reason": reason,
            "template_mean_finite_bins": int(n_finite),
            "template_mean_min_finite_bins": int(min_profile_bins),
            "template_mean_expected_bins": int(n_bins),
            "template_mean_bin_count": None if not np.isfinite(bin_count) else int(bin_count),
        })
    return pd.DataFrame(rows)


def compute_mean_template(
    profiles: List[Dict],
    cfg: Optional[TemplateConfig] = None,
) -> Optional[ChannelTemplate]:
    """Compute a mean normalized template from extracted profiles.

    Returns None if too few usable profiles are available.
    """
    if cfg is None:
        cfg = TemplateConfig()

    if len(profiles) < cfg.min_measured_xs:
        log.warning(
            "Only %d measured profiles available (need %d); cannot build template.",
            len(profiles), cfg.min_measured_xs,
        )
        return None

    n_bins = cfg.n_bins
    min_profile_bins = max(5, int(np.ceil(n_bins / 4.0)))
    usable_profiles: List[Dict] = []
    usable_depths: List[np.ndarray] = []
    for p in profiles:
        nd = p.get("norm_depth")
        if nd is None:
            continue
        nd = np.asarray(nd, dtype="float64")
        if nd.shape[0] != n_bins:
            continue
        n_finite = int(np.isfinite(nd).sum())
        if n_finite < min_profile_bins:
            continue
        usable_profiles.append(p)
        usable_depths.append(nd)

    if len(usable_profiles) < cfg.min_measured_xs:
        log.warning(
            "Only %d usable profiles after NaN filtering (need %d); cannot build template.",
            len(usable_profiles), cfg.min_measured_xs,
        )
        return None

    all_depths = np.stack(usable_depths, axis=0)
    n_per_bin = np.sum(np.isfinite(all_depths), axis=0).astype("int64")
    stats_min_count = max(2, int(np.ceil(len(usable_profiles) / 4.0)))
    mean_depth = np.full(n_bins, np.nan, dtype="float64")
    std_depth = np.full(n_bins, np.nan, dtype="float64")
    for i in range(n_bins):
        vals = all_depths[:, i]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean_depth[i] = float(np.mean(vals))
        if vals.size >= stats_min_count:
            std_depth[i] = float(np.std(vals, ddof=0))

    mean_depth[0] = 0.0
    mean_depth[-1] = 0.0
    std_depth[0] = 0.0
    std_depth[-1] = 0.0

    finite = np.isfinite(mean_depth)
    if finite.sum() < 3:
        log.warning("Mean template has too few valid bins; cannot build template.")
        return None
    norm_pos = np.linspace(0.0, 1.0, n_bins)
    if not finite.all():
        mean_depth = np.interp(norm_pos, norm_pos[finite], mean_depth[finite])
    std_finite = np.isfinite(std_depth)
    if std_finite.any() and not std_finite.all():
        std_depth = np.interp(norm_pos, norm_pos[std_finite], std_depth[std_finite])
    elif not std_finite.any():
        std_depth = np.zeros(n_bins, dtype="float64")

    asymmetries = np.array([p.get("asymmetry", 0.0) for p in usable_profiles], dtype="float64")
    mean_asym = float(np.nanmean(asymmetries))
    std_asym = float(np.nanstd(asymmetries, ddof=0))

    template = ChannelTemplate(
        norm_pos=norm_pos,
        mean_depth=mean_depth,
        std_depth=std_depth,
        n_profiles_per_bin=n_per_bin,
        n_profiles=len(usable_profiles),
        mean_asymmetry=mean_asym,
        std_asymmetry=std_asym,
    )

    log.info(
        "Built channel template from %d usable profiles (%d raw): peak_norm_depth=%.3f, mean_asymmetry=%.3f±%.3f",
        len(usable_profiles),
        len(profiles),
        float(np.nanmax(mean_depth)),
        mean_asym,
        std_asym,
    )
    return template


# ---------------------------------------------------------------------------
# Width → max-depth fitting
# ---------------------------------------------------------------------------

def fit_width_depth_relation(
    profiles: List[Dict],
    cfg: Optional[TemplateConfig] = None,
) -> Tuple[float, float, float, int, str]:
    """Fit a power-law width→max_depth relation from measured profiles.

    Returns (a, b, r2, n_used, source_label).
    Falls back to Leopold & Maddock defaults if local data is insufficient.
    """
    if cfg is None:
        cfg = TemplateConfig()

    widths = []
    depths = []
    for p in profiles:
        w = p.get("width_m", np.nan)
        d = p.get("max_depth_m", np.nan)
        if np.isfinite(w) and w > 0 and np.isfinite(d) and d >= cfg.min_sounding_depth_m:
            widths.append(w)
            depths.append(d)

    n = len(widths)
    if n < cfg.fit_min_xs:
        log.info(
            "Too few measured XS for local fit (%d < %d); using Leopold & Maddock fallback.",
            n, cfg.fit_min_xs,
        )
        return cfg.fallback_a, cfg.fallback_b, 0.0, 0, "fallback_leopold_maddock"

    W = np.array(widths, dtype="float64")
    D = np.array(depths, dtype="float64")

    depth_span = float(np.nanmax(D) - np.nanmin(D))
    if depth_span < cfg.fit_min_depth_span_m:
        log.info(
            "Measured depth span too narrow (%.2f m < %.2f m); using Leopold & Maddock fallback.",
            depth_span, cfg.fit_min_depth_span_m,
        )
        return cfg.fallback_a, cfg.fallback_b, 0.0, 0, "fallback_leopold_maddock"

    # Log-linear regression: log(D) = log(a) + b * log(W)
    logW = np.log(W)
    logD = np.log(D)

    # Check for degenerate cases
    if np.std(logW) < 1e-6:
        log.info("Width variance too low for power-law fit; using Leopold & Maddock fallback.")
        return cfg.fallback_a, cfg.fallback_b, 0.0, 0, "fallback_leopold_maddock"

    # Ordinary least squares
    A_mat = np.column_stack([np.ones(n), logW])
    try:
        coefs, residuals, rank, sv = np.linalg.lstsq(A_mat, logD, rcond=None)
    except np.linalg.LinAlgError:
        log.warning("lstsq failed; using Leopold & Maddock fallback.")
        return cfg.fallback_a, cfg.fallback_b, 0.0, 0, "fallback_leopold_maddock"

    log_a, b = float(coefs[0]), float(coefs[1])
    a = float(np.exp(log_a))

    # R² in log space
    ss_res = float(np.sum((logD - A_mat @ coefs) ** 2))
    ss_tot = float(np.sum((logD - np.mean(logD)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Sanity check: if fit is physically implausible, fall back
    b_lo, b_hi = cfg.fit_b_range
    a_lo, a_hi = cfg.fit_a_range
    if not (b_lo <= b <= b_hi) or not (a_lo <= a <= a_hi) or r2 < 0.1:
        log.warning(
            "Local fit implausible (a=%.4f, b=%.3f, R²=%.3f); using Leopold & Maddock fallback.",
            a, b, r2,
        )
        return cfg.fallback_a, cfg.fallback_b, 0.0, 0, "fallback_leopold_maddock"

    log.info(
        "Local width→depth fit: Dmax = %.4f × W^%.3f  (R²=%.3f, n=%d)",
        a, b, r2, n,
    )
    return a, b, r2, n, "local_power_law"


def attach_depth_fit_to_template(
    template: ChannelTemplate,
    profiles: List[Dict],
    cfg: Optional[TemplateConfig] = None,
) -> ChannelTemplate:
    """Fit width→depth and shape exponent, attach to an existing template."""
    if cfg is None:
        cfg = TemplateConfig()
    a, b, r2, n, source = fit_width_depth_relation(profiles, cfg)
    template.depth_a = a
    template.depth_b = b
    template.depth_fit_r2 = r2
    template.depth_fit_n = n
    template.depth_fit_source = source
    template.shape_exponent = fit_shape_exponent(template)
    return template


def fit_shape_exponent(template: ChannelTemplate) -> float:
    """Fit an effective shape exponent for skeleton integration.

    The skeleton computes depth = Dmax * r^exp where r is the normalized
    distance from bank (0=bank, 1=centerline).  This function finds the
    exponent that best matches the learned template profile.

    The template uses norm_pos (0=left bank, 1=right bank), so the
    equivalent skeleton 'r' at position x is: r = 2*min(x, 1-x) — which
    maps both banks to 0 and the center to 1.

    Returns the best-fit exponent, clipped to [0.2, 2.0].
    """
    pos = template.norm_pos
    depth = template.mean_depth.copy()
    if pos is None or depth is None or len(pos) < 5:
        return 0.5

    finite = np.isfinite(depth)
    if finite.sum() < 5:
        return 0.5

    # Convert template positions to skeleton 'r' coordinate
    r = 2.0 * np.minimum(pos, 1.0 - pos)  # 0 at banks, 1 at center
    r = np.clip(r, 0.01, 1.0)  # avoid log(0)

    # Normalize depth to [0, 1]
    peak = np.nanmax(depth)
    if not np.isfinite(peak) or peak <= 0:
        return 0.5
    d_norm = np.clip(depth / peak, 0.0, 1.0)

    # Fit: d_norm ≈ r^exp  →  log(d_norm) ≈ exp * log(r)
    mask = finite & (d_norm > 0.05) & (r > 0.05)  # exclude near-zero edges
    if mask.sum() < 5:
        return 0.5

    log_r = np.log(r[mask])
    log_d = np.log(d_norm[mask])

    # Weighted least squares: weight by depth (deeper points matter more)
    w = d_norm[mask]
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return 0.5

    exp_fit = float(np.sum(w * log_d * log_r) / np.sum(w * log_r * log_r))
    exp_fit = float(np.clip(exp_fit, 0.2, 2.0))

    log.info("Fitted shape exponent: %.3f (from %d template bins)", exp_fit, int(mask.sum()))
    return exp_fit


# ---------------------------------------------------------------------------
# Per-component templates
# ---------------------------------------------------------------------------

def build_per_component_templates(
    profiles: List[Dict],
    cfg: Optional[TemplateConfig] = None,
) -> Dict[int, ChannelTemplate]:
    """Build separate templates per component_id where data supports it.

    Components with fewer than cfg.min_measured_xs profiles are excluded
    (callers should fall back to the global template for those).

    Returns a dict mapping component_id -> ChannelTemplate.
    """
    if cfg is None:
        cfg = TemplateConfig()

    # Group profiles by component_id
    by_comp: Dict[int, List[Dict]] = {}
    for p in profiles:
        cid = int(p.get("component_id", -1))
        by_comp.setdefault(cid, []).append(p)

    templates: Dict[int, ChannelTemplate] = {}
    for cid, comp_profiles in by_comp.items():
        if len(comp_profiles) < cfg.min_measured_xs:
            log.info(
                "Component %d: only %d profiles (need %d); skipping per-component template.",
                cid, len(comp_profiles), cfg.min_measured_xs,
            )
            continue

        t = compute_mean_template(comp_profiles, cfg)
        if t is None:
            continue
        t = attach_depth_fit_to_template(t, comp_profiles, cfg)
        t.component_id = cid
        templates[cid] = t
        log.info(
            "Component %d template: n=%d, fit=%s (a=%.4f, b=%.3f, R²=%.3f, exp=%.3f)",
            cid, t.n_profiles, t.depth_fit_source,
            t.depth_a, t.depth_b, t.depth_fit_r2, t.shape_exponent,
        )

    return templates


# ---------------------------------------------------------------------------
# Template-based synthesis
# ---------------------------------------------------------------------------

def synthesize_xs_from_template(
    template: ChannelTemplate,
    dist_from_left: np.ndarray,
    width_m: float,
    wse_m: float,
    dmax_override: Optional[float] = None,
    cfg: Optional[TemplateConfig] = None,
) -> Tuple[np.ndarray, Dict]:
    """Synthesize a bed profile at an unmapped cross-section.

    Returns (z_bed, metadata) where z_bed is bed elevation at each dist point.
    """
    if cfg is None:
        cfg = TemplateConfig()

    meta = {
        "width_m": float(width_m) if np.isfinite(width_m) else None,
        "wse_m": float(wse_m) if np.isfinite(wse_m) else None,
        "dmax_override": float(dmax_override) if dmax_override is not None and np.isfinite(dmax_override) else None,
        "source": "channel_template",
        "template_n_profiles": template.n_profiles,
        "depth_fit_source": template.depth_fit_source,
    }

    z_bed = template.synthesize_bed(
        dist_from_left,
        width_m,
        wse_m,
        dmax_override=dmax_override,
        dmin=cfg.dmin_m,
        dmax_cap=cfg.dmax_m,
    )

    dmax_used = dmax_override if (dmax_override is not None and np.isfinite(dmax_override)) else template.predict_dmax(width_m, cfg.dmin_m, cfg.dmax_m)
    meta["dmax_used_m"] = float(dmax_used) if np.isfinite(dmax_used) else None

    return z_bed, meta


# ---------------------------------------------------------------------------
# Cross-validation / evaluation
# ---------------------------------------------------------------------------

def evaluate_template_loo(
    profiles: List[Dict],
    cfg: Optional[TemplateConfig] = None,
) -> pd.DataFrame:
    """Leave-one-out cross-validation of the template approach.

    For each measured profile:
      1. Build template from all OTHER profiles
      2. Predict the held-out profile using template + width→depth
      3. Compute RMSE between predicted and observed normalized depth

    Returns a DataFrame with per-XS error metrics.
    """
    if cfg is None:
        cfg = TemplateConfig()

    if len(profiles) < cfg.min_measured_xs + 1:
        log.info("Too few profiles for LOO evaluation (%d).", len(profiles))
        return pd.DataFrame()

    rows = []
    for i in range(len(profiles)):
        held_out = profiles[i]
        train = [p for j, p in enumerate(profiles) if j != i]

        template = compute_mean_template(train, cfg)
        if template is None:
            continue

        template = attach_depth_fit_to_template(template, train, cfg)

        # Predict held-out XS depth
        obs_depth = np.asarray(held_out["norm_depth"], dtype="float64")
        pred_depth = template.mean_depth.copy()

        # Compare only at bins where both are valid
        valid = np.isfinite(obs_depth) & np.isfinite(pred_depth)
        if valid.sum() < 5:
            continue

        rmse_norm = float(np.sqrt(np.nanmean((obs_depth[valid] - pred_depth[valid]) ** 2)))
        mae_norm = float(np.nanmean(np.abs(obs_depth[valid] - pred_depth[valid])))
        bias_norm = float(np.nanmean(pred_depth[valid] - obs_depth[valid]))

        # Also evaluate absolute depth using width→depth relation
        obs_dmax = held_out["max_depth_m"]
        pred_dmax = template.predict_dmax(held_out["width_m"], cfg.dmin_m, cfg.dmax_m)
        dmax_error_m = float(pred_dmax - obs_dmax) if (np.isfinite(pred_dmax) and _valid_dmax_value(obs_dmax)) else np.nan

        rows.append({
            "xs_id": held_out.get("xs_id", ""),
            "width_m": held_out["width_m"],
            "obs_max_depth_m": obs_dmax,
            "pred_max_depth_m": float(pred_dmax),
            "dmax_error_m": dmax_error_m,
            "rmse_norm_depth": rmse_norm,
            "mae_norm_depth": mae_norm,
            "bias_norm_depth": bias_norm,
            "n_valid_bins": int(valid.sum()),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        log.info(
            "LOO evaluation: n=%d, median_rmse_norm=%.4f, median_dmax_error=%.2f m",
            len(result),
            float(result["rmse_norm_depth"].median()),
            float(result["dmax_error_m"].median()),
        )
    return result


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def build_channel_template(
    xs_lines: pd.DataFrame,
    xs_points: pd.DataFrame,
    sounding_calibration: Optional[pd.DataFrame] = None,
    cfg: Optional[TemplateConfig] = None,
    out_dir: Optional[Path] = None,
    estuary_clip_mask: Optional[Path] = None,
    river_gpkg: Optional[Path] = None,
    return_details: bool = False,
    wse_override: Optional[Dict[str, float]] = None,
) -> Optional[ChannelTemplate]:
    """Full pipeline: extract → filter → average → fit depth → evaluate → write diagnostics."""
    if cfg is None:
        cfg = TemplateConfig()

    raw_profiles, raw_summary = extract_normalized_profiles(
        xs_lines, xs_points,
        sounding_calibration=sounding_calibration,
        cfg=cfg,
        wse_override=wse_override,
    )

    selected_profiles, selected_summary, filter_info, profile_diagnostics = select_template_training_sections(
        xs_lines, raw_profiles, raw_summary, cfg=cfg,
        estuary_clip_mask=estuary_clip_mask,
        river_gpkg=river_gpkg,
    )
    mean_diag = diagnose_template_profile_usability(selected_profiles, cfg)
    if profile_diagnostics is None or profile_diagnostics.empty:
        profile_diagnostics = raw_summary.copy()
    if mean_diag is not None and not mean_diag.empty:
        profile_diagnostics = profile_diagnostics.merge(mean_diag, on="xs_id", how="left")
        _keep = pd.Series(profile_diagnostics.get("keep_for_template", False), index=profile_diagnostics.index)
        _keep = _keep.astype(bool)
        _mean_usable = pd.Series(profile_diagnostics.get("template_mean_usable", False), index=profile_diagnostics.index)
        _mean_usable = _mean_usable.where(pd.notna(_mean_usable), False).astype(bool)
        profile_diagnostics["keep_for_template"] = _keep
        profile_diagnostics["template_selected"] = _keep & _mean_usable
        profile_diagnostics["template_final_reject_reason"] = ""
        _rej = profile_diagnostics.get("reject_reason", "").fillna("").astype(str)
        _mean_rej = profile_diagnostics.get("template_mean_reject_reason", "").fillna("").astype(str)
        profile_diagnostics.loc[~profile_diagnostics["keep_for_template"], "template_final_reject_reason"] = _rej[~profile_diagnostics["keep_for_template"]]
        _sel_keep = _keep & ~_mean_usable
        profile_diagnostics.loc[_sel_keep, "template_final_reject_reason"] = _mean_rej[_sel_keep]
        profile_diagnostics.loc[profile_diagnostics["template_selected"], "template_final_reject_reason"] = "selected"

    log.info(
        "Channel template profile screening: raw=%d selected=%d rejected=%d",
        len(raw_profiles),
        len(selected_profiles),
        max(0, len(raw_profiles) - len(selected_profiles)),
    )
    if len(selected_profiles) < cfg.min_measured_xs:
        log.warning(
            "Insufficient selected profiles for template (%d < %d); no channel template built.",
            len(selected_profiles), cfg.min_measured_xs,
        )
        _write_channel_template_diagnostics(out_dir, raw_summary, selected_summary, profile_diagnostics, filter_info)
        result = ChannelTemplateBuildResult(
            template=None,
            raw_profiles=raw_profiles,
            selected_profiles=selected_profiles,
            raw_summary=raw_summary,
            selected_summary=selected_summary,
            profile_diagnostics=profile_diagnostics,
            loo_results=pd.DataFrame(),
            filter_info=filter_info,
        )
        return result if return_details else None

    template = compute_mean_template(selected_profiles, cfg)
    if template is None:
        _write_channel_template_diagnostics(out_dir, raw_summary, selected_summary, profile_diagnostics, filter_info)
        result = ChannelTemplateBuildResult(
            template=None,
            raw_profiles=raw_profiles,
            selected_profiles=selected_profiles,
            raw_summary=raw_summary,
            selected_summary=selected_summary,
            profile_diagnostics=profile_diagnostics,
            loo_results=pd.DataFrame(),
            filter_info=filter_info,
        )
        return result if return_details else None

    template = attach_depth_fit_to_template(template, selected_profiles, cfg)

    # Build per-component templates where data supports it
    component_templates: Dict[int, ChannelTemplate] = {}
    if len(selected_profiles) >= cfg.min_measured_xs:
        component_templates = build_per_component_templates(selected_profiles, cfg)

    loo_results = pd.DataFrame()
    if len(selected_profiles) >= cfg.min_measured_xs + 1:
        loo_results = evaluate_template_loo(selected_profiles, cfg)

    loo_downgraded = False
    if not loo_results.empty and template.depth_fit_source == "local_power_law":
        med_rmse = float(loo_results["rmse_norm_depth"].median())
        med_dmax_err = float(loo_results["dmax_error_m"].abs().median())
        if med_rmse > cfg.loo_max_rmse_norm or med_dmax_err > cfg.loo_max_dmax_error_m:
            log.warning(
                "LOO quality gate FAILED: rmse_norm=%.4f (max=%.4f), dmax_error=%.2f m (max=%.2f m). Downgrading depth fit to fallback; template shape preserved.",
                med_rmse, cfg.loo_max_rmse_norm, med_dmax_err, cfg.loo_max_dmax_error_m,
            )
            template.depth_a = cfg.fallback_a
            template.depth_b = cfg.fallback_b
            template.depth_fit_source = "loo_downgraded_to_fallback"
            template.depth_fit_r2 = 0.0
            loo_downgraded = True
            for ct in component_templates.values():
                ct.depth_a = cfg.fallback_a
                ct.depth_b = cfg.fallback_b
                ct.depth_fit_source = "loo_downgraded_to_fallback"
                ct.depth_fit_r2 = 0.0
        else:
            log.info("LOO quality gate PASSED: rmse_norm=%.4f, dmax_error=%.2f m.", med_rmse, med_dmax_err)

    if cfg.write_diagnostics and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        template_path = out_dir / "channel_template.json"
        tpl_dict = template.to_dict()
        if component_templates:
            tpl_dict["per_component"] = {str(cid): ct.to_dict() for cid, ct in component_templates.items()}
        tpl_dict["loo_quality_gate"] = "FAILED" if loo_downgraded else ("PASSED" if not loo_results.empty else "skipped")
        tpl_dict["loo_downgraded"] = loo_downgraded
        with open(template_path, "w") as f:
            json.dump(tpl_dict, f, indent=2, default=str)
        if not raw_summary.empty:
            raw_summary.to_csv(out_dir / "channel_template_profiles_raw.csv", index=False)
        if not selected_summary.empty:
            selected_summary.to_csv(out_dir / "channel_template_profiles_selected.csv", index=False)
            selected_summary.to_csv(out_dir / "channel_template_profiles.csv", index=False)
        elif not raw_summary.empty:
            raw_summary.to_csv(out_dir / "channel_template_profiles.csv", index=False)
        if filter_info:
            (out_dir / "channel_template_filter.json").write_text(json.dumps(filter_info, indent=2, default=str) + "\n")
        if not loo_results.empty:
            loo_results.to_csv(out_dir / "channel_template_loo.csv", index=False)
        wd_rows = [{"width_m": p["width_m"], "max_depth_m": p["max_depth_m"]} for p in selected_profiles]
        wd_df = pd.DataFrame(wd_rows)
        if not wd_df.empty:
            wd_df.to_csv(out_dir / "channel_template_width_depth.csv", index=False)
        try:
            payload = {}
            for i, p in enumerate(selected_profiles):
                payload[f"norm_depth_{i}"] = np.asarray(p.get("norm_depth", []), dtype="float64")
                payload[f"meta_{i}"] = np.array([
                    float(p.get("width_m", np.nan)),
                    float(p.get("max_depth_m", np.nan)),
                    float(p.get("s_center_m", np.nan)),
                    float(p.get("component_id", -1)),
                ], dtype="float64")
            payload["xs_ids"] = np.array([str(p.get("xs_id", "")) for p in selected_profiles])
            payload["n_profiles"] = np.array([len(selected_profiles)], dtype="int64")
            np.savez_compressed(out_dir / "channel_template_profiles.npz", **payload)
        except Exception:
            log.debug("Failed to save profile npz", exc_info=True)

    result = ChannelTemplateBuildResult(
        template=template,
        raw_profiles=raw_profiles,
        selected_profiles=selected_profiles,
        raw_summary=raw_summary,
        selected_summary=selected_summary,
        profile_diagnostics=profile_diagnostics,
        loo_results=loo_results,
        filter_info=filter_info,
    )
    return result if return_details else template


def compute_template_support_weights(
    xs_table: pd.DataFrame,
    measured_summary: pd.DataFrame,
    sigma_m: float,
) -> pd.DataFrame:
    sigma_m = float(max(1.0, sigma_m))
    out = xs_table[["xs_id"]].copy()
    out["xs_id"] = out["xs_id"].astype(str)
    out["template_nearest_measured_dist_m"] = np.nan
    out["template_decay_weight"] = 0.0
    if xs_table.empty or measured_summary is None or measured_summary.empty:
        return out
    group_col = "component_id" if "component_id" in xs_table.columns and "component_id" in measured_summary.columns else ("river_id" if "river_id" in xs_table.columns and "river_id" in measured_summary.columns else None)
    if group_col is None or "s_center_m" not in xs_table.columns or "s_center_m" not in measured_summary.columns:
        return out
    targets = xs_table[["xs_id", group_col, "s_center_m"]].copy()
    targets["xs_id"] = targets["xs_id"].astype(str)
    measured = measured_summary[["xs_id", group_col, "s_center_m"]].copy()
    measured["xs_id"] = measured["xs_id"].astype(str)
    targets["s_center_m"] = pd.to_numeric(targets["s_center_m"], errors="coerce")
    measured["s_center_m"] = pd.to_numeric(measured["s_center_m"], errors="coerce")
    for gval, tg in targets.groupby(group_col, dropna=False):
        mg = measured.loc[measured[group_col] == gval]
        s_meas = pd.to_numeric(mg["s_center_m"], errors="coerce").dropna().to_numpy(dtype="float64")
        if s_meas.size == 0:
            continue
        s_t = pd.to_numeric(tg["s_center_m"], errors="coerce").to_numpy(dtype="float64")
        for idx, sval in zip(tg.index, s_t):
            if not np.isfinite(sval):
                continue
            d = float(np.nanmin(np.abs(s_meas - sval)))
            w = float(np.exp(-0.5 * (d / sigma_m) ** 2))
            out.loc[idx, "template_nearest_measured_dist_m"] = d
            out.loc[idx, "template_decay_weight"] = w
    return out


def build_local_deformed_shapes(
    xs_table: pd.DataFrame,
    template: ChannelTemplate,
    measured_profiles: List[Dict],
    sigma_m: float,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    sigma_m = float(max(1.0, sigma_m))
    base = template.mean_depth.astype("float64")
    shape_map: Dict[str, np.ndarray] = {}
    meta = xs_table[["xs_id"]].copy()
    meta["xs_id"] = meta["xs_id"].astype(str)
    meta["template_shape_weight"] = 0.0
    meta["template_shape_nearest_measured_dist_m"] = np.nan
    if xs_table.empty or not measured_profiles:
        return shape_map, meta
    group_col = "component_id" if "component_id" in xs_table.columns else ("river_id" if "river_id" in xs_table.columns else None)
    if group_col is None or "s_center_m" not in xs_table.columns:
        return shape_map, meta

    prof_rows = []
    for p in measured_profiles:
        nd = np.asarray(p.get("norm_depth"), dtype="float64")
        if nd.shape != base.shape:
            continue
        finite = np.isfinite(nd)
        if finite.sum() < 3:
            continue
        interp = nd.copy()
        if not finite.all():
            interp = np.interp(template.norm_pos, template.norm_pos[finite], nd[finite])
        resid = interp - base
        prof_rows.append({
            "xs_id": str(p.get("xs_id", "")),
            group_col: p.get(group_col),
            "s_center_m": float(p.get("s_center_m", np.nan)),
            "residual": resid,
        })
    if not prof_rows:
        return shape_map, meta
    prof_df = pd.DataFrame(prof_rows)

    targets = xs_table[["xs_id", group_col, "s_center_m"]].copy()
    targets["xs_id"] = targets["xs_id"].astype(str)
    targets["s_center_m"] = pd.to_numeric(targets["s_center_m"], errors="coerce")

    for gval, tg in targets.groupby(group_col, dropna=False):
        mg = prof_df.loc[prof_df[group_col] == gval].copy()
        if mg.empty:
            continue
        s_meas = pd.to_numeric(mg["s_center_m"], errors="coerce").to_numpy(dtype="float64")
        residuals = list(mg["residual"])
        for idx, row in tg.iterrows():
            sval = float(row.get("s_center_m", np.nan))
            if not np.isfinite(sval):
                continue
            ds = np.abs(s_meas - sval)
            good = np.isfinite(ds)
            if not np.any(good):
                continue
            ds = ds[good]
            res_use = [residuals[i] for i, ok in enumerate(good) if ok]
            if np.any(ds == 0.0):
                w = (ds == 0.0).astype("float64")
            else:
                w = np.exp(-0.5 * (ds / sigma_m) ** 2)
            if w.size == 0 or float(np.nanmax(w)) <= 0.0:
                continue
            wsum = float(np.sum(w))
            resid = np.sum(np.stack([ww * rr for ww, rr in zip(w, res_use)], axis=0), axis=0) / max(wsum, 1e-12)
            local = np.clip(base + resid, 0.0, 1.0)
            local[0] = 0.0
            local[-1] = 0.0
            xsid = str(row.get("xs_id", ""))
            shape_map[xsid] = local.astype("float64")
            meta.loc[idx, "template_shape_weight"] = float(np.nanmax(w))
            meta.loc[idx, "template_shape_nearest_measured_dist_m"] = float(np.nanmin(ds))
    return shape_map, meta


# ---------------------------------------------------------------------------
# Parabolic fallback template
# ---------------------------------------------------------------------------

def make_fallback_template(cfg: Optional[TemplateConfig] = None) -> ChannelTemplate:
    """Build a synthetic parabolic / U-shaped template when no measured data exists.

    This provides a rounded natural-river fallback rather than a flat-bottom
    trapezoid, packaged as a ChannelTemplate object so the calling code can use
    a uniform interface.
    """
    if cfg is None:
        cfg = TemplateConfig()

    n_bins = cfg.n_bins
    norm_pos = np.linspace(0.0, 1.0, n_bins)
    center = 0.5
    x = (norm_pos - center) / 0.5
    depth = np.clip(1.0 - x**2, 0.0, 1.0).astype("float64")
    depth[0] = 0.0
    depth[-1] = 0.0

    return ChannelTemplate(
        norm_pos=norm_pos,
        mean_depth=depth,
        std_depth=np.full(n_bins, 0.15, dtype="float64"),  # generic uncertainty
        n_profiles_per_bin=np.zeros(n_bins, dtype="int64"),
        n_profiles=0,
        depth_a=cfg.fallback_a,
        depth_b=cfg.fallback_b,
        depth_fit_source="fallback_parabolic",
    )


# ---------------------------------------------------------------------------
# Along-channel residual correction (Improvement: "mold shape to measurements")
# ---------------------------------------------------------------------------

def compute_along_channel_residual_correction(
    xs_table: pd.DataFrame,
    template: ChannelTemplate,
    measured_profiles: List[Dict],
    sigma_m: float = 2000.0,
    min_weight: float = 0.01,
) -> pd.DataFrame:
    """Compute per-XS Dmax residual correction from measured vs template-predicted.

    At each measured XS: residual = observed_max_depth - template_predicted_max_depth.
    For every XS (measured or not): corrected_dmax = template_dmax + weighted interpolation
    of nearby measured residuals, using Gaussian decay along the channel.

    This is the "mold that shape to measurements" step — the template provides the
    general shape, and residual correction locally adjusts it to match authoritative data.

    Returns the xs_table with added columns:
      - residual_correction_m: additive correction to Dmax (meters)
      - residual_confidence: weight of the correction (0=no nearby measurements, 1=exact match)
      - residual_n_anchors: number of measured XS contributing to the correction
    """
    sigma_m = float(max(1.0, sigma_m))

    out = xs_table[["xs_id"]].copy()
    out["xs_id"] = out["xs_id"].astype(str)
    out["residual_correction_m"] = 0.0
    out["residual_confidence"] = 0.0
    out["residual_n_anchors"] = 0

    if not measured_profiles or template is None:
        return out

    group_col = "component_id" if "component_id" in xs_table.columns else (
        "river_id" if "river_id" in xs_table.columns else None)
    if group_col is None or "s_center_m" not in xs_table.columns:
        return out

    # Build measured residuals: observed_dmax - template_predicted_dmax
    meas_rows = []
    for p in measured_profiles:
        dmax_obs = float(p.get("max_depth_m", np.nan))
        width = float(p.get("width_m", np.nan))
        if not (_valid_dmax_value(dmax_obs) and np.isfinite(width) and width > 0):
            continue
        dmax_pred = template.predict_dmax(width)
        resid = dmax_obs - dmax_pred
        meas_rows.append({
            "xs_id": str(p.get("xs_id", "")),
            group_col: p.get(group_col),
            "s_center_m": float(p.get("s_center_m", np.nan)),
            "dmax_obs": dmax_obs,
            "dmax_pred": dmax_pred,
            "residual": resid,
        })

    if not meas_rows:
        return out
    meas_df = pd.DataFrame(meas_rows)

    targets = xs_table[["xs_id", "s_center_m"]].copy()
    if group_col in xs_table.columns:
        targets[group_col] = xs_table[group_col]
    targets["xs_id"] = targets["xs_id"].astype(str)
    targets["s_center_m"] = pd.to_numeric(targets["s_center_m"], errors="coerce")

    groups = targets.groupby(group_col, dropna=False) if group_col in targets.columns else [(None, targets)]
    for gval, tg in groups:
        mg = meas_df[meas_df[group_col] == gval] if group_col in meas_df.columns else meas_df
        if mg.empty:
            continue
        s_meas = mg["s_center_m"].to_numpy(dtype="float64")
        resids = mg["residual"].to_numpy(dtype="float64")
        valid = np.isfinite(s_meas) & np.isfinite(resids)
        s_meas = s_meas[valid]
        resids = resids[valid]
        if len(s_meas) == 0:
            continue

        for idx, row in tg.iterrows():
            sval = float(row.get("s_center_m", np.nan))
            if not np.isfinite(sval):
                continue
            ds = np.abs(s_meas - sval)
            w = np.exp(-0.5 * (ds / sigma_m) ** 2)
            w[w < min_weight] = 0.0
            wsum = float(np.sum(w))
            if wsum <= 0:
                continue
            correction = float(np.sum(w * resids) / wsum)
            confidence = float(np.max(w))
            n_anchors = int(np.sum(w > 0))
            out.loc[idx, "residual_correction_m"] = correction
            out.loc[idx, "residual_confidence"] = confidence
            out.loc[idx, "residual_n_anchors"] = n_anchors

    n_corrected = int((out["residual_confidence"] > 0).sum())
    if n_corrected > 0:
        med_corr = float(out.loc[out["residual_confidence"] > 0, "residual_correction_m"].median())
        log.info(
            "Along-channel residual: corrected %d/%d XS, median correction=%.3f m, sigma=%.0f m",
            n_corrected, len(out), med_corr, sigma_m,
        )
    return out


# ---------------------------------------------------------------------------
# Confidence field for gap-fill and guidance layers
# ---------------------------------------------------------------------------

def compute_gap_fill_confidence(
    bed_valid_before: "np.ndarray",
    channel: "np.ndarray",
    measured_xs_mask: "np.ndarray | None" = None,
    sigma_m: float = 500.0,
    pixel_size_m: float = 10.0,
) -> "np.ndarray":
    """Compute a [0, 1] confidence field for the river bed raster.

    Confidence assignment:
      - XS-measured pixels (bed_valid_before): confidence = 1.0
      - Gap-filled pixels near measured: Gaussian decay from nearest measured pixel
      - Gap-filled pixels far from measured: low confidence approaching 0

    Returns float32 array with NaN outside channel.
    """
    from scipy.ndimage import distance_transform_edt

    conf = np.full(channel.shape, np.nan, dtype="float32")
    if not np.any(channel):
        return conf

    # Distance from each channel pixel to the nearest XS-measured pixel
    anchor = channel & bed_valid_before
    if measured_xs_mask is not None:
        anchor = anchor | (channel & measured_xs_mask)

    if not np.any(anchor):
        conf[channel] = 0.0
        return conf

    inv = np.ones(channel.shape, dtype=np.uint8)
    inv[anchor] = 0
    dist = distance_transform_edt(inv, sampling=pixel_size_m).astype("float32")

    # Gaussian confidence decay
    w = np.exp(-0.5 * (dist / max(sigma_m, 1.0)) ** 2).astype("float32")

    # Measured pixels get full confidence
    w[anchor] = 1.0

    conf[channel] = w[channel]
    return conf


# ---------------------------------------------------------------------------
# Estuary template transition (blend fluvial → flat-bottom near tidal zone)
# ---------------------------------------------------------------------------

def build_estuary_transition_weights(
    xs_table: pd.DataFrame,
    estuary_transition_mask: "np.ndarray | None" = None,
    width_gradient_threshold: float = 2.0,
    transition_distance_m: float = 500.0,
) -> pd.DataFrame:
    """Compute per-XS blending weight between fluvial template and estuary profile.

    Detection uses two signals:
      1. Estuary transition mask (from domain analysis) — if a XS centroid falls
         inside the mask, it gets a blend weight approaching 0 (flat-bottom).
      2. Width gradient — where the channel widens rapidly downstream, the fluvial
         template becomes inappropriate. Width ratio > threshold triggers transition.

    Returns xs_table with added columns:
      - estuary_blend_weight: 1.0 = pure fluvial template, 0.0 = flat-bottom estuary
      - estuary_transition_flag: bool, True if in transition zone
    """
    out = xs_table[["xs_id"]].copy()
    out["estuary_blend_weight"] = 1.0
    out["estuary_transition_flag"] = False

    if "s_center_m" not in xs_table.columns or "width_m" not in xs_table.columns:
        return out

    s = pd.to_numeric(xs_table.get("s_center_m"), errors="coerce").to_numpy(dtype="float64")
    w = pd.to_numeric(xs_table.get("width_m"), errors="coerce").to_numpy(dtype="float64")

    # Width gradient detection: compare each XS width to the median of its upstream neighbors
    order = np.argsort(s)
    widths_sorted = w[order]
    n = len(widths_sorted)
    if n < 5:
        return out

    # Rolling median of width over the upstream 5 XS
    kernel = min(5, n // 2)
    median_w = np.full(n, np.nan, dtype="float64")
    for i in range(kernel, n):
        median_w[i] = float(np.nanmedian(widths_sorted[max(0, i - kernel):i]))

    # Width ratio: current / upstream median
    ratio = np.full(n, 1.0, dtype="float64")
    valid = np.isfinite(median_w) & (median_w > 0) & np.isfinite(widths_sorted)
    ratio[valid] = widths_sorted[valid] / median_w[valid]

    # Where ratio exceeds threshold, start transitioning
    # Compute distance into transition zone
    in_transition = ratio > width_gradient_threshold
    if np.any(in_transition):
        first_transition_s = s[order[in_transition]][0]  # station of first widening
        for i in range(n):
            orig_idx = order[i]
            si = s[orig_idx]
            if not np.isfinite(si):
                continue
            dist_into_transition = si - first_transition_s
            if dist_into_transition >= 0:
                # Smooth blend: 1.0 at transition start → 0.0 at transition_distance_m
                blend = float(np.clip(1.0 - dist_into_transition / max(transition_distance_m, 1.0), 0.0, 1.0))
                out.iloc[orig_idx, out.columns.get_loc("estuary_blend_weight")] = blend
                out.iloc[orig_idx, out.columns.get_loc("estuary_transition_flag")] = (blend < 1.0)

    n_transition = int(out["estuary_transition_flag"].sum())
    if n_transition > 0:
        log.info(
            "Estuary transition: %d/%d XS in transition zone (threshold=%.1f, transition_dist=%.0f m)",
            n_transition, len(out), width_gradient_threshold, transition_distance_m,
        )
    return out


def apply_estuary_blend_to_dmax(
    dmax_prior: float,
    estuary_blend_weight: float,
    width_m: float,
    wse_m: float,
    estuary_depth_fraction: float = 0.3,
) -> float:
    """Blend fluvial Dmax toward a shallower estuary depth.

    In estuarine transition zones, the fluvial template's deep V/U channel
    shape becomes inappropriate — estuaries tend toward broad, shallow, flat-bottomed
    geometry.  This function blends:
      weight=1.0 → pure fluvial Dmax (template prior)
      weight=0.0 → estuary_depth_fraction * width_m^0.3 (shallow flat)

    The estuary depth model is deliberately conservative — it produces a floor
    that the authoritative data will override where available.
    """
    if not np.isfinite(estuary_blend_weight) or estuary_blend_weight >= 1.0:
        return dmax_prior
    # Conservative estuary depth: shallow, width-insensitive
    dmax_estuary = float(estuary_depth_fraction * max(width_m, 1.0) ** 0.3)
    dmax_estuary = float(np.clip(dmax_estuary, 0.5, 5.0))
    return float(estuary_blend_weight * dmax_prior + (1.0 - estuary_blend_weight) * dmax_estuary)
