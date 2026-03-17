"""sdb_tier.py — Tiered SDB model selection and confidence-weighted output.

Implements automatic degradation from RF to Stumpf-linear to physics-only
based on training data quality, producing confidence-weighted pseudo-soundings
that blend seamlessly across CUDEM DEM tile boundaries.

**Tier 1** (n >= TIER1_MIN, good spatial coverage):
    Full RF model. Confidence = f(sample_size, spatial_coverage, RMSE).

**Tier 2** (TIER2_MIN <= n < TIER1_MIN, or poor spatial coverage):
    Stumpf log-ratio linear model only (2 DOF). More spatially stable
    across tile boundaries than RF. Confidence reduced.

**Tier 3** (n < TIER2_MIN, or no reliable ATL):
    Physics-only depth limit from Kd estimation. Emits constraint surface
    (max-depth mask) rather than pseudo-soundings.

Literature basis:
    - Stumpf et al. (2003): Band-ratio robustness to bottom type variation
    - Caballero & Stumpf (2020): Multi-temporal compositing for turbid waters
    - Zhang et al. (2022): ICESat-2 Kd-derived physics SDB without priors
    - Parrish et al. (2025): ATL24 accuracy (RMSE 0.42m high-conf points)
    - PHY-SDB (2025): Stumpf-inspired loss as physics regularizer
    - Copernicus Marine (2025): Per-pixel uncertainty + retrieval method traceability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier thresholds (configurable)
# ---------------------------------------------------------------------------

TIER1_MIN_POINTS = 300      # Minimum reliable points for RF
TIER2_MIN_POINTS = 50       # Minimum points for Stumpf linear
TIER1_MIN_SPATIAL_COV = 0.15  # Convex hull / AOI area ratio
TIER1_MIN_HIGH_CONF_FRAC = 0.3  # Fraction of high-confidence ATL points
TIER1_MIN_DEPTH_RANGE_M = 2.0   # Minimum depth range for RF to generalize
TIER1_MIN_DEPTH_STD_M = 0.5     # Minimum depth std for RF to learn signal
TIER1_MIN_UNIQUE_TRACKS = 3     # Minimum independent ICESat-2 passes

# Edge taper width as fraction of tile dimension (each side)
EDGE_TAPER_FRAC = 0.10


# ---------------------------------------------------------------------------
# Data quality assessment
# ---------------------------------------------------------------------------

@dataclass
class TrainingDataQuality:
    """Assessment of training data suitability for model selection."""
    n_total: int = 0
    n_high_confidence: int = 0
    n_sources: int = 0
    n_unique_tracks: int = 0
    spatial_coverage_frac: float = 0.0
    depth_range_m: float = 0.0
    depth_std_m: float = 0.0
    high_conf_frac: float = 0.0

    # Source breakdown
    n_atl03: int = 0
    n_atl24: int = 0
    n_extra_xyz: int = 0

    # Kd-based optical depth limit (if available)
    kd_max_depth_m: Optional[float] = None

    @property
    def selected_tier(self) -> int:
        """Auto-select model tier based on data quality metrics."""
        if self.n_total < TIER2_MIN_POINTS:
            return 3

        if self.n_total < TIER1_MIN_POINTS:
            return 2

        # Enough points but poor spatial coverage or low confidence
        if self.spatial_coverage_frac < TIER1_MIN_SPATIAL_COV:
            return 2
        if self.high_conf_frac < TIER1_MIN_HIGH_CONF_FRAC:
            return 2

        # Enough points but insufficient depth diversity for RF
        if self.depth_range_m < TIER1_MIN_DEPTH_RANGE_M:
            return 2
        if self.depth_std_m < TIER1_MIN_DEPTH_STD_M:
            return 2

        # Too few independent tracks — RF will overfit to one pass
        if 0 < self.n_unique_tracks < TIER1_MIN_UNIQUE_TRACKS:
            return 2

        return 1

    @property
    def tier_reason(self) -> str:
        """Human-readable reason for tier selection."""
        tier = self.selected_tier
        if tier == 3:
            return "insufficient training points (n=%d < %d)" % (self.n_total, TIER2_MIN_POINTS)
        if tier == 2:
            reasons = []
            if self.n_total < TIER1_MIN_POINTS:
                reasons.append("n=%d < %d" % (self.n_total, TIER1_MIN_POINTS))
            if self.spatial_coverage_frac < TIER1_MIN_SPATIAL_COV:
                reasons.append("spatial_cov=%.2f < %.2f" % (
                    self.spatial_coverage_frac, TIER1_MIN_SPATIAL_COV))
            if self.high_conf_frac < TIER1_MIN_HIGH_CONF_FRAC:
                reasons.append("high_conf_frac=%.2f < %.2f" % (
                    self.high_conf_frac, TIER1_MIN_HIGH_CONF_FRAC))
            if self.depth_range_m < TIER1_MIN_DEPTH_RANGE_M:
                reasons.append("depth_range=%.1fm < %.1fm" % (
                    self.depth_range_m, TIER1_MIN_DEPTH_RANGE_M))
            if self.depth_std_m < TIER1_MIN_DEPTH_STD_M:
                reasons.append("depth_std=%.2fm < %.2fm" % (
                    self.depth_std_m, TIER1_MIN_DEPTH_STD_M))
            if 0 < self.n_unique_tracks < TIER1_MIN_UNIQUE_TRACKS:
                reasons.append("tracks=%d < %d" % (
                    self.n_unique_tracks, TIER1_MIN_UNIQUE_TRACKS))
            if not reasons:
                reasons.append("unspecified quality degradation")
            return "degraded to Stumpf-linear: " + "; ".join(reasons)
        return "full RF model (n=%d, spatial_cov=%.2f, depth_range=%.1fm)" % (
            self.n_total, self.spatial_coverage_frac, self.depth_range_m)


def assess_training_data(
    df,
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None,
    kd_max_depth_m: Optional[float] = None,
) -> TrainingDataQuality:
    """Assess training data quality for tier selection.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with columns: longitude, latitude, depth_m, source,
        and optionally atl03_conf or confidence.
    aoi_bounds : tuple, optional
        (W, E, S, N) bounding box for spatial coverage calculation.
    kd_max_depth_m : float, optional
        Physics-derived optical depth limit from Kd estimation.

    Returns
    -------
    TrainingDataQuality
    """
    import pandas as pd

    q = TrainingDataQuality()
    q.kd_max_depth_m = kd_max_depth_m

    if df is None or len(df) == 0:
        return q

    q.n_total = len(df)

    # Source breakdown
    if "source" in df.columns:
        src_lower = df["source"].astype(str).str.lower()
        q.n_atl03 = int((src_lower.str.contains("atl03")).sum())
        q.n_atl24 = int((src_lower.str.contains("atl24")).sum())
        q.n_extra_xyz = int((src_lower.str.contains("extra_xyz")).sum())
        q.n_sources = int(src_lower.nunique())

    # High-confidence fraction
    if "atl03_conf" in df.columns:
        conf = pd.to_numeric(df["atl03_conf"], errors="coerce")
        q.n_high_confidence = int((conf >= 4).sum())
    elif "confidence" in df.columns:
        conf = pd.to_numeric(df["confidence"], errors="coerce")
        q.n_high_confidence = int((conf >= 0.9).sum())
    else:
        # If no confidence column, assume all are medium confidence
        q.n_high_confidence = q.n_total // 2

    q.high_conf_frac = q.n_high_confidence / max(q.n_total, 1)

    # Unique tracks (independent ICESat-2 passes)
    for col in ("track", "track_id", "rgt", "gt"):
        if col in df.columns:
            q.n_unique_tracks = int(df[col].nunique())
            break
    # No fallback to source.nunique() — source categories (atl03/atl24/atl_agreed)
    # are NOT independent tracks. Leave n_unique_tracks=0 if no track column found.

    # Depth statistics
    if "depth_m" in df.columns:
        depths = pd.to_numeric(df["depth_m"], errors="coerce").dropna().abs()
        if len(depths) > 0:
            q.depth_range_m = float(depths.max() - depths.min())
            q.depth_std_m = float(depths.std())

    # Spatial coverage (convex hull area / AOI area)
    if aoi_bounds is not None and "longitude" in df.columns and "latitude" in df.columns:
        try:
            lons = pd.to_numeric(df["longitude"], errors="coerce").dropna().values
            lats = pd.to_numeric(df["latitude"], errors="coerce").dropna().values
            if len(lons) >= 3:
                from scipy.spatial import ConvexHull
                pts = np.column_stack([lons, lats])
                hull = ConvexHull(pts)
                hull_area = hull.volume  # 2D: volume = area

                w, e, s, n = aoi_bounds
                aoi_area = abs((e - w) * (n - s))
                if aoi_area > 0:
                    q.spatial_coverage_frac = min(1.0, hull_area / aoi_area)
        except Exception:
            # ConvexHull can fail for collinear points
            q.spatial_coverage_frac = 0.0

    return q


# ---------------------------------------------------------------------------
# Confidence weight computation
# ---------------------------------------------------------------------------

def compute_pixel_confidence(
    depth: np.ndarray,
    model_tier: int,
    *,
    rmse_m: float = 1.0,
    max_depth_m: float = 20.0,
    support_distance: Optional[np.ndarray] = None,
    support_decay_m: float = 500.0,
) -> np.ndarray:
    """Compute per-pixel confidence weight [0, 1] for SDB guide points.

    Factors:
        1. Model tier base confidence (tier1=1.0, tier2=0.6, tier3=0.3)
        2. Depth-dependent decay (confidence decreases near optical limit)
        3. RMSE-based scaling (lower RMSE = higher confidence)
        4. Distance from training support (exponential decay)

    Parameters
    ----------
    depth : np.ndarray
        Predicted depth values (positive down, meters).
    model_tier : int
        1, 2, or 3.
    rmse_m : float
        Model RMSE in meters.
    max_depth_m : float
        Estimated optical depth limit.
    support_distance : np.ndarray, optional
        Distance to nearest training point (meters).
    support_decay_m : float
        Half-decay distance for support-distance weighting.

    Returns
    -------
    np.ndarray of float32, same shape as depth, values in [0, 1].
    """
    # Base confidence by tier
    tier_base = {1: 1.0, 2: 0.6, 3: 0.3}.get(model_tier, 0.3)

    # Depth-dependent: linear decay from 1.0 at surface to 0.2 at max_depth
    depth_abs = np.abs(np.where(np.isfinite(depth), depth, 0.0))
    safe_max = max(max_depth_m, 1.0)
    depth_factor = np.clip(1.0 - 0.8 * (depth_abs / safe_max), 0.2, 1.0)

    # RMSE scaling: conf = 1 / (1 + rmse)
    rmse_factor = 1.0 / (1.0 + max(rmse_m, 0.0))

    conf = np.full_like(depth, tier_base, dtype=np.float32)
    conf *= depth_factor.astype(np.float32)
    conf *= float(rmse_factor)

    # Support distance decay
    if support_distance is not None:
        dist_factor = np.exp(-support_distance / max(support_decay_m, 1.0))
        conf *= dist_factor.astype(np.float32)

    # NaN pixels get zero confidence
    conf[~np.isfinite(depth)] = 0.0

    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def apply_edge_taper(
    confidence: np.ndarray,
    *,
    taper_frac: float = EDGE_TAPER_FRAC,
) -> np.ndarray:
    """Apply cosine edge taper to confidence weights for seamless tile blending.

    Reduces confidence to zero at tile edges using a cosine taper over
    the outer ``taper_frac`` of each dimension. Adjacent overlapping tiles
    will blend smoothly because each tile's influence fades at the boundary.

    Parameters
    ----------
    confidence : np.ndarray
        2D confidence array (height, width).
    taper_frac : float
        Fraction of each dimension to taper (default 0.10 = 10% on each side).

    Returns
    -------
    np.ndarray tapered confidence, same shape.
    """
    if confidence.ndim != 2:
        return confidence
    if taper_frac <= 0:
        return confidence.copy()

    h, w = confidence.shape
    taper_h = max(1, int(h * taper_frac))
    taper_w = max(1, int(w * taper_frac))

    # Build 1D taper profiles
    taper_y = np.ones(h, dtype=np.float32)
    taper_x = np.ones(w, dtype=np.float32)

    # Top/bottom cosine ramp: 0 at edge → 1 at taper_h pixels in
    if taper_h > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_h)))
        taper_y[:taper_h] = ramp
        taper_y[-taper_h:] = ramp[::-1]

    # Left/right cosine ramp
    if taper_w > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_w)))
        taper_x[:taper_w] = ramp
        taper_x[-taper_w:] = ramp[::-1]

    # Outer product → 2D taper mask
    taper_2d = np.outer(taper_y, taper_x)

    return (confidence * taper_2d).astype(np.float32)


# ---------------------------------------------------------------------------
# Tier selection entry point (for use in train_sdb_model)
# ---------------------------------------------------------------------------

def select_model_tier(
    df,
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None,
    kd_max_depth_m: Optional[float] = None,
    *,
    override_tier: Optional[int] = None,
) -> Tuple[int, TrainingDataQuality]:
    """Assess training data and select model tier.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    aoi_bounds : tuple, optional
        (W, E, S, N) for spatial coverage calculation.
    kd_max_depth_m : float, optional
        Physics-derived depth limit.
    override_tier : int, optional
        Force a specific tier (1, 2, or 3). For testing/debugging.

    Returns
    -------
    (tier, quality) tuple.
    """
    quality = assess_training_data(df, aoi_bounds=aoi_bounds,
                                    kd_max_depth_m=kd_max_depth_m)

    if override_tier is not None and override_tier in (1, 2, 3):
        tier = override_tier
        log.info("Model tier forced to %d (override)", tier)
    else:
        tier = quality.selected_tier

    log.info("Model tier selected: %d — %s", tier, quality.tier_reason)
    log.info("Training data quality: n=%d, high_conf=%.0f%%, spatial_cov=%.2f, "
             "depth_range=%.1fm, sources=%d",
             quality.n_total, quality.high_conf_frac * 100,
             quality.spatial_coverage_frac, quality.depth_range_m, quality.n_sources)

    return tier, quality


# ---------------------------------------------------------------------------
# Provenance metadata for guide points
# ---------------------------------------------------------------------------

@dataclass
class GuidePointMeta:
    """Metadata for a set of SDB guide points emitted for CUDEM."""
    model_tier: int = 1
    tier_reason: str = ""
    n_points: int = 0
    rmse_m: float = float("nan")
    max_depth_m: float = float("nan")
    confidence_mean: float = float("nan")
    confidence_min: float = float("nan")
    edge_taper_applied: bool = False
    taper_frac: float = EDGE_TAPER_FRAC
    kd_max_depth_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_tier": self.model_tier,
            "tier_reason": self.tier_reason,
            "n_points": self.n_points,
            "rmse_m": self.rmse_m,
            "max_depth_m": self.max_depth_m,
            "confidence_mean": self.confidence_mean,
            "confidence_min": self.confidence_min,
            "edge_taper_applied": self.edge_taper_applied,
            "taper_frac": self.taper_frac,
            "kd_max_depth_m": self.kd_max_depth_m,
        }


# ---------------------------------------------------------------------------
# Improvement 1: Kd-based ATL point reliability filter
# ---------------------------------------------------------------------------

# ICESat-2 bottom detection is reliable when Kd(490) < 0.12 m⁻¹
# (per Hallström et al. 2024, Zhang et al. 2022)
KD_ATL_RELIABLE_MAX = 0.12  # m⁻¹
KD_ATL_MARGINAL_MAX = 0.20  # Above this, ATL depths are unreliable


def filter_atl_by_kd(
    df,
    kd_490: float,
    *,
    reliable_max: float = KD_ATL_RELIABLE_MAX,
    marginal_max: float = KD_ATL_MARGINAL_MAX,
) -> tuple:
    """Filter ATL training points based on scene Kd(490) water clarity.

    ICESat-2 photons can reliably penetrate to the seafloor only in clear
    water.  When Kd(490) exceeds ~0.12 m⁻¹, bottom detections become
    unreliable; above ~0.20 m⁻¹ they should be rejected.

    For marginal conditions (0.12 < Kd < 0.20), we keep high-confidence
    ATL points but reduce their sample weight to prevent them from
    dominating the model.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with columns: source, depth_m, and optionally
        atl03_conf or confidence.
    kd_490 : float
        Scene-level median Kd(490) in m⁻¹.
    reliable_max : float
        Kd threshold below which all ATL points are fully reliable.
    marginal_max : float
        Kd threshold above which all ATL points are rejected.

    Returns
    -------
    (filtered_df, filter_meta) : tuple
        filtered_df has unreliable points removed and marginal points
        down-weighted.  filter_meta is a dict with filtering statistics.
    """
    import pandas as pd

    meta = {
        "kd_490": kd_490,
        "reliable_max": reliable_max,
        "marginal_max": marginal_max,
        "action": "none",
        "n_input": len(df),
        "n_output": len(df),
        "n_removed": 0,
        "n_downweighted": 0,
    }

    if df is None or len(df) == 0:
        return df, meta

    if kd_490 <= reliable_max:
        meta["action"] = "clear_water_all_reliable"
        log.info("[Kd-ATL] Kd(490)=%.3f <= %.2f: all ATL points reliable",
                 kd_490, reliable_max)
        return df, meta

    # Identify ATL-sourced rows
    is_atl = pd.Series(False, index=df.index)
    if "source" in df.columns:
        src = df["source"].astype(str).str.lower()
        is_atl = src.str.contains("atl03") | src.str.contains("atl24")

    if not is_atl.any():
        meta["action"] = "no_atl_points"
        return df, meta

    if kd_490 > marginal_max:
        # Reject all ATL points — water too turbid for reliable lidar bathy
        n_before = len(df)
        df_out = df[~is_atl].copy()
        n_removed = n_before - len(df_out)
        meta.update({
            "action": "turbid_water_atl_rejected",
            "n_output": len(df_out),
            "n_removed": n_removed,
        })
        log.warning("[Kd-ATL] Kd(490)=%.3f > %.2f: REJECTED %d ATL points "
                    "(water too turbid for reliable lidar bathymetry)",
                    kd_490, marginal_max, n_removed)
        return df_out, meta

    # Marginal: keep high-confidence ATL but down-weight
    # Weight decays linearly from 1.0 at reliable_max to 0.2 at marginal_max
    weight_factor = max(0.2, 1.0 - 0.8 * (kd_490 - reliable_max) /
                        max(marginal_max - reliable_max, 0.01))

    df_out = df.copy()
    if "sample_weight" not in df_out.columns:
        df_out["sample_weight"] = 1.0

    # Only down-weight low-confidence ATL points
    is_low_conf = pd.Series(False, index=df_out.index)
    if "atl03_conf" in df_out.columns:
        conf = pd.to_numeric(df_out["atl03_conf"], errors="coerce")
        is_low_conf = is_atl & (conf < 4)
    elif "confidence" in df_out.columns:
        conf = pd.to_numeric(df_out["confidence"], errors="coerce")
        is_low_conf = is_atl & (conf < 0.9)
    else:
        is_low_conf = is_atl  # No confidence info → down-weight all ATL

    n_downweighted = int(is_low_conf.sum())
    df_out.loc[is_low_conf, "sample_weight"] *= weight_factor

    meta.update({
        "action": "marginal_water_atl_downweighted",
        "weight_factor": weight_factor,
        "n_output": len(df_out),
        "n_downweighted": n_downweighted,
    })
    log.info("[Kd-ATL] Kd(490)=%.3f marginal: down-weighted %d low-conf ATL "
             "points by %.2f", kd_490, n_downweighted, weight_factor)
    return df_out, meta


def build_physics_constraint_surface(
    kd_map: np.ndarray,
    *,
    confidence_level: str = "moderate",
    hard_cap_m: float = 50.0,
) -> np.ndarray:
    """Build a per-pixel maximum depth constraint surface from Kd.

    This is the tier-3 output: instead of pseudo-soundings, emit a raster
    that constrains interpolation depth.  Each pixel's value is the
    physics-derived maximum depth based on local water clarity.

    Parameters
    ----------
    kd_map : np.ndarray
        Per-pixel Kd(490) estimates (m⁻¹).  NaN where unknown.
    confidence_level : str
        'conservative' (1.5/Kd), 'moderate' (2.3/Kd), or 'optimistic' (3.5/Kd).
    hard_cap_m : float
        Absolute maximum depth (meters).

    Returns
    -------
    np.ndarray (float32)
        Max-depth constraint surface.  NaN where Kd is unknown.
    """
    factors = {"conservative": 1.5, "moderate": 2.3, "optimistic": 3.5}
    factor = factors.get(confidence_level, 2.3)

    kd_safe = np.where(np.isfinite(kd_map) & (kd_map > 0.01), kd_map, np.nan)
    max_depth = factor / kd_safe
    max_depth = np.clip(max_depth, 0.0, hard_cap_m)

    log.info("[Kd] Built physics constraint surface: factor=%.1f, "
             "median max_depth=%.1fm",
             factor, float(np.nanmedian(max_depth)))
    return max_depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Improvement 2: Stumpf-as-regularizer for tier-1 RF
# ---------------------------------------------------------------------------

# When RF and Stumpf disagree by more than this many sigma,
# mark the pixel as low-confidence
STUMPF_REGULARIZER_SIGMA = 2.0


def compute_stumpf_rf_disagreement(
    rf_depth: np.ndarray,
    stumpf_depth: np.ndarray,
    *,
    sigma_threshold: float = STUMPF_REGULARIZER_SIGMA,
) -> tuple:
    """Flag pixels where RF and Stumpf predictions disagree significantly.

    Even when RF is selected (tier 1), the Stumpf linear model provides
    a physics-grounded baseline.  Where the two diverge beyond a threshold,
    RF is likely extrapolating into unsupported space — those pixels should
    receive reduced confidence.

    Parameters
    ----------
    rf_depth : np.ndarray
        RF-predicted depths (positive down, meters).
    stumpf_depth : np.ndarray
        Stumpf-linear predicted depths (same convention).
    sigma_threshold : float
        Number of standard deviations of the residual beyond which
        a pixel is flagged as disagreeing.

    Returns
    -------
    (disagreement_mask, penalty_factor, meta) : tuple
        disagreement_mask : bool array, True where they disagree
        penalty_factor : float array [0, 1], 1.0 = no penalty
        meta : dict with statistics
    """
    valid = np.isfinite(rf_depth) & np.isfinite(stumpf_depth)
    residual = np.full_like(rf_depth, np.nan, dtype=np.float64)
    residual[valid] = rf_depth[valid] - stumpf_depth[valid]

    # Robust scale estimate (MAD → sigma)
    valid_resid = residual[valid]
    if len(valid_resid) < 10:
        # Not enough data to estimate scale
        return (np.zeros_like(rf_depth, dtype=bool),
                np.ones_like(rf_depth, dtype=np.float32),
                {"n_valid": int(valid.sum()), "n_flagged": 0,
                 "robust_sigma": float("nan"), "action": "insufficient_overlap"})

    median_resid = float(np.median(valid_resid))
    mad = float(np.median(np.abs(valid_resid - median_resid)))
    robust_sigma = max(mad * 1.4826, 0.1)  # MAD → σ, floor at 0.1m

    # Compute z-scores
    z = np.full_like(rf_depth, 0.0, dtype=np.float64)
    z[valid] = np.abs(residual[valid] - median_resid) / robust_sigma

    # Flag where |z| > threshold
    disagree = valid & (z > sigma_threshold)
    n_flagged = int(disagree.sum())

    # Penalty: smooth decay from 1.0 (no penalty) at z=0 to 0.2 at z=2*threshold
    penalty = np.ones_like(rf_depth, dtype=np.float32)
    over = valid & (z > sigma_threshold * 0.5)
    if over.any():
        excess = (z[over] - sigma_threshold * 0.5) / max(sigma_threshold, 0.5)
        penalty[over] = np.clip(1.0 - 0.8 * excess, 0.2, 1.0).astype(np.float32)

    frac_flagged = n_flagged / max(int(valid.sum()), 1)
    log.info("[Stumpf-regularizer] σ=%.2fm, flagged %d/%d pixels (%.1f%%) "
             "where RF-Stumpf |Δ| > %.1fσ",
             robust_sigma, n_flagged, int(valid.sum()),
             frac_flagged * 100, sigma_threshold)

    meta = {
        "n_valid": int(valid.sum()),
        "n_flagged": n_flagged,
        "frac_flagged": frac_flagged,
        "robust_sigma": robust_sigma,
        "median_residual": median_resid,
        "sigma_threshold": sigma_threshold,
    }
    return disagree, penalty, meta


def apply_stumpf_regularization(
    confidence: np.ndarray,
    rf_depth: np.ndarray,
    stumpf_depth: np.ndarray,
    *,
    sigma_threshold: float = STUMPF_REGULARIZER_SIGMA,
) -> tuple:
    """Apply Stumpf-based regularization to confidence weights.

    Reduces confidence where RF diverges from the physics-grounded Stumpf
    baseline, catching RF extrapolation artifacts that would introduce
    seams between tiles.

    Parameters
    ----------
    confidence : np.ndarray
        Existing per-pixel confidence weights [0, 1].
    rf_depth : np.ndarray
        RF-predicted depths.
    stumpf_depth : np.ndarray
        Stumpf-linear predicted depths.
    sigma_threshold : float
        Disagreement threshold in sigma units.

    Returns
    -------
    (regularized_confidence, meta) : tuple
    """
    disagree, penalty, meta = compute_stumpf_rf_disagreement(
        rf_depth, stumpf_depth, sigma_threshold=sigma_threshold)

    regularized = (confidence * penalty).astype(np.float32)

    meta["confidence_reduction_mean"] = float(
        np.nanmean(confidence[disagree] - regularized[disagree])
    ) if disagree.any() else 0.0

    return regularized, meta
