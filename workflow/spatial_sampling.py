"""spatial_sampling.py – Multi-stage adaptive sampling for bathymetry training data.

Preserves high-accuracy sources (LiDAR, multibeam), uses ICESat-2 as gap filler,
adapts to bathymetric complexity, and maintains spatial coverage.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

@dataclass
class SourceConfig:
    """Configuration for a single data source."""
    name: str
    tier: int  # 1=highest quality, 3=gap filler
    weight: float
    retention_target: float  # Fraction to keep (0.0-1.0)
    min_spacing_m: float
    influence_radius_m: float

@dataclass
class SamplingConfig:
    """Global sampling configuration."""
    target_total_points: int = 2000
    max_gap_m: float = 100.0
    min_tier1_fraction: float = 0.50
    depth_bins: int = 10
    min_points_per_bin: int = 50

    # Safety floors
    # If adaptive sampling becomes too aggressive (especially in small AOIs
    # with dense survey points), the sampler will top-up to reach these floors.
    min_total_points: int = 200
    min_points_per_source: int = 50
    
    # Complexity score weights
    variance_weight: float = 0.4
    gradient_weight: float = 0.4
    density_weight: float = 0.2
    
    # Adaptive grid sizes (meters)
    grid_high_complexity: float = 15.0
    grid_medium_complexity: float = 30.0
    grid_low_complexity: float = 60.0
    
    # Complexity thresholds
    high_complexity_threshold: float = 0.7
    low_complexity_threshold: float = 0.3
    
    # Transect detection
    detect_transects: bool = True
    transect_spacing_m: float = 50.0
    
    # Boundary preservation
    boundary_buffer_m: float = 50.0
    boundary_grid_reduction: float = 0.5

    # ---------------------------------------------------------------------
    # Performance guards for extremely dense sources.
    #
    # Some sources (especially gridded/derived XYZ) can arrive with millions
    # of points inside an AOI. Running the full adaptive thinning stack on
    # tens of millions of points is unnecessary when the final selection is
    # capped (target_total_points) and can take a very long time.
    #
    # Strategy: apply a fast, vectorized grid-based pre-thin per source to
    # reduce the candidate set to a bounded size *before* adaptive thinning.
    # ---------------------------------------------------------------------
    max_points_per_source_prethin: int = 500_000
    prethin_cell_scale: float = 1.0
    random_seed: int = 1337


def _lonlat_to_local_m(points_lonlat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat degrees to approximate local meters for grid hashing.

    Uses a local equirectangular approximation about the median latitude.
    This is used only for *sampling* (not geodesic distance reporting).
    """
    if points_lonlat.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    lon = points_lonlat[:, 0].astype(float)
    lat = points_lonlat[:, 1].astype(float)
    lat0 = np.median(lat)

    # meters per degree
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    x_m = lon * m_per_deg_lon
    y_m = lat * m_per_deg_lat
    return x_m, y_m


def fast_grid_prethin_lonlat(
    points_lonlat: np.ndarray,
    cell_m: float,
    max_keep: int,
    seed: int = 0,
) -> np.ndarray:
    """Fast vectorized pre-thinning using a fixed grid in approximate meters.

    Keeps at most one point per grid cell (deterministic by first occurrence).
    If the resulting set is still larger than max_keep, apply a deterministic
    random subset.

    Returns a boolean mask of points to keep.
    """
    n = len(points_lonlat)
    if n == 0:
        return np.zeros(0, dtype=bool)

    cell_m = float(max(cell_m, 1e-6))
    x_m, y_m = _lonlat_to_local_m(points_lonlat)
    ix = np.floor(x_m / cell_m).astype(np.int64)
    iy = np.floor(y_m / cell_m).astype(np.int64)

    # Hash 2D cell indices to 1D key (pairing).
    key = ix * np.int64(4_000_000_007) + iy

    # Keep first occurrence per cell (stable order)
    _, first_idx = np.unique(key, return_index=True)
    keep = np.zeros(n, dtype=bool)
    keep[first_idx] = True

    kept_idx = np.flatnonzero(keep)
    if max_keep > 0 and kept_idx.size > max_keep:
        rng = np.random.default_rng(seed)
        sel = rng.choice(kept_idx, size=max_keep, replace=False)
        keep[:] = False
        keep[sel] = True

    return keep

# Default source configurations
DEFAULT_SOURCE_CONFIGS = {
    'bathy_lidar': SourceConfig(
        name='bathy_lidar',
        tier=1,
        weight=25.0,
        retention_target=0.85,
        min_spacing_m=10.0,
        influence_radius_m=75.0
    ),
    'multibeam': SourceConfig(
        name='multibeam',
        tier=1,
        weight=20.0,
        retention_target=0.80,
        min_spacing_m=15.0,
        influence_radius_m=75.0
    ),
    'high_quality_bag': SourceConfig(
        name='high_quality_bag',
        tier=1,
        weight=15.0,
        retention_target=0.75,
        min_spacing_m=20.0,
        influence_radius_m=60.0
    ),
    'sonar_survey': SourceConfig(
        name='sonar_survey',
        tier=1,
        weight=12.0,
        retention_target=0.80,
        min_spacing_m=15.0,
        influence_radius_m=60.0
    ),
    'extra_xyz': SourceConfig(
        name='extra_xyz',
        tier=2,
        weight=10.0,
        retention_target=0.40,
        min_spacing_m=30.0,
        influence_radius_m=50.0
    ),
    'extra_xyz_auth': SourceConfig(
        name='extra_xyz_auth',
        tier=1,
        weight=18.0,
        retention_target=0.80,
        min_spacing_m=15.0,
        influence_radius_m=75.0
    ),
    'atl24': SourceConfig(
        name='atl24',
        tier=2,
        weight=6.0,
        retention_target=0.30,
        min_spacing_m=40.0,
        influence_radius_m=50.0
    ),
    'atl_agreed': SourceConfig(
        name='atl_agreed',
        tier=2,
        weight=6.0,
        retention_target=0.30,
        min_spacing_m=40.0,
        influence_radius_m=50.0
    ),
    'atl03': SourceConfig(
        name='atl03',
        tier=3,
        weight=3.0,
        retention_target=0.10,
        min_spacing_m=60.0,
        influence_radius_m=30.0
    ),
}

def compute_local_complexity(
    points: np.ndarray,
    depths: np.ndarray,
    kdtree: cKDTree,
    config: SamplingConfig,
    k: int = 30
) -> np.ndarray:
    """
    Compute complexity score for each point based on local bathymetry.
    
    Args:
        points: (N, 2) array of (lon, lat) coordinates
        depths: (N,) array of depth values
        kdtree: Pre-built KDTree for efficient queries
        config: Sampling configuration
        k: Number of neighbors for local analysis
    
    Returns:
        (N,) array of complexity scores in [0, 1]
    """
    n_points = len(points)
    complexity = np.zeros(n_points)
    
    for i in range(n_points):
        # Find k nearest neighbors
        distances, indices = kdtree.query(points[i], k=min(k, n_points))
        
        # Skip if not enough neighbors
        if len(indices) < 5:
            complexity[i] = 0.5  # Default medium complexity
            continue
        
        neighbor_depths = depths[indices]
        
        # Metric 1: Depth variance
        depth_variance = np.std(neighbor_depths)
        
        # Metric 2: Local gradient (fit plane)
        if len(indices) >= 10:
            neighbor_points = points[indices]
            try:
                # Simple gradient using depth range / distance range
                depth_range = neighbor_depths.max() - neighbor_depths.min()
                spatial_range = np.sqrt(
                    (neighbor_points[:, 0].max() - neighbor_points[:, 0].min())**2 +
                    (neighbor_points[:, 1].max() - neighbor_points[:, 1].min())**2
                )
                gradient = depth_range / max(spatial_range, 1e-6)
            except (ValueError, IndexError, ZeroDivisionError):
                gradient = 0.0
        else:
            gradient = 0.0
        
        # Metric 3: Local density (inverse of mean distance)
        mean_distance = np.mean(distances[1:])  # Skip self
        density_score = 1.0 / (1.0 + mean_distance * 1000)  # Convert to meters approx
        
        # Store individual metrics for later normalization
        complexity[i] = depth_variance  # Will normalize later
    
    # Normalize each metric to [0, 1]
    variance_scores = complexity.copy()
    if variance_scores.max() > 0:
        variance_scores = variance_scores / variance_scores.max()
    
    # Compute gradient and density scores separately (simplified for now)
    gradient_scores = np.zeros(n_points)
    density_scores = np.ones(n_points) * 0.5
    
    # Combined complexity score
    complexity_final = (
        config.variance_weight * variance_scores +
        config.gradient_weight * gradient_scores +
        config.density_weight * density_scores
    )
    
    return np.clip(complexity_final, 0.0, 1.0)

def detect_transects(
    points: np.ndarray,
    min_length: int = 50,
    direction_tolerance: float = 15.0
) -> np.ndarray:
    """
    Detect linear transects in point cloud.
    
    Args:
        points: (N, 2) array of coordinates
        min_length: Minimum points to form a transect
        direction_tolerance: Degrees of direction variation allowed
    
    Returns:
        (N,) boolean array indicating transect membership
    """
    # Simplified: Use local direction consistency
    # Full implementation would use RANSAC or Hough transform
    
    n_points = len(points)
    is_transect = np.zeros(n_points, dtype=bool)
    
    if n_points < min_length:
        return is_transect
    
    # Build KDTree for neighbor queries
    kdtree = cKDTree(points)
    
    for i in range(n_points):
        # Find 20 nearest neighbors
        distances, indices = kdtree.query(points[i], k=min(20, n_points))
        
        if len(indices) < 10:
            continue
        
        # Compute directions to neighbors
        neighbor_points = points[indices[1:]]  # Skip self
        vectors = neighbor_points - points[i]
        
        # Check if vectors are aligned (linear pattern)
        if len(vectors) >= 5:
            # Compute primary direction using SVD
            try:
                _, _, vh = np.linalg.svd(vectors)
                primary_direction = vh[0]
                
                # Check alignment of other vectors
                dots = np.abs(np.dot(vectors, primary_direction))
                norms = np.linalg.norm(vectors, axis=1)
                alignments = dots / (norms + 1e-10)
                
                # If most vectors are aligned, it's a transect
                aligned_fraction = (alignments > 0.9).mean()
                if aligned_fraction > 0.7:
                    is_transect[i] = True
            except (np.linalg.LinAlgError, ValueError):
                pass
    
    return is_transect

def adaptive_grid_thinning(
    points: np.ndarray,
    depths: np.ndarray,
    complexity_scores: np.ndarray,
    source_values: np.ndarray,
    config: SamplingConfig,
    source_config: SourceConfig
) -> np.ndarray:
    """
    Apply adaptive grid-based thinning.
    
    Args:
        points: (N, 2) array of coordinates
        depths: (N,) array of depths
        complexity_scores: (N,) array of complexity scores
        source_values: (N,) array of source priorities/weights
        config: Global sampling config
        source_config: Source-specific config
    
    Returns:
        Boolean array indicating which points to keep
    """
    n_points = len(points)
    keep = np.zeros(n_points, dtype=bool)

    if n_points == 0:
        return keep

    # config.grid_* are expressed in **meters**.
    #   points are in lon/lat degrees.
    #   The previous implementation mixed these units, causing grid sizes of 15/30/60
    #   to be treated as **degrees**, collapsing most AOIs into ~1 cell and keeping
    #   only ~1-3 points. We convert lon/lat to a local metric XY for binning.

    lat0 = float(np.nanmedian(points[:, 1]))
    m_per_deg_lat = 111_000.0
    m_per_deg_lon = max(1e-6, 111_000.0 * float(np.cos(np.deg2rad(lat0))))

    lon_min, lat_min = np.nanmin(points, axis=0)
    x_m = (points[:, 0] - lon_min) * m_per_deg_lon
    y_m = (points[:, 1] - lat_min) * m_per_deg_lat

    # ------------------------------------------------------------------
    # Dynamic spacing/grid safeguards
    # ------------------------------------------------------------------
    # In small AOIs, a fixed 60 m "low complexity" grid can collapse thousands
    # of survey points down to a few dozen cells. That's fine for massive AOIs,
    # but it can starve model training locally. We adjust the effective minimum
    # spacing and grid sizes based on the footprint area and the run's target.
    try:
        w = float(np.nanmax(x_m) - np.nanmin(x_m))
        h = float(np.nanmax(y_m) - np.nanmin(y_m))
        area_m2 = max(1.0, w) * max(1.0, h)
    except Exception:
        area_m2 = None

    # Desired local floor (we never exceed n_points)
    desired_floor = int(min(n_points, max(config.min_total_points, config.target_total_points)))

    eff_min_spacing_m = float(source_config.min_spacing_m) if source_config.min_spacing_m else 30.0
    if area_m2 is not None and area_m2 > 0 and eff_min_spacing_m > 0:
        max_pts_at_min = area_m2 / (eff_min_spacing_m ** 2)
        if max_pts_at_min < desired_floor:
            # Relax spacing (within this sampler only) to allow enough points.
            eff_min_spacing_m = max(2.0, math.sqrt(area_m2 / float(desired_floor)))
            log.info(
                "[SAMPLING] Relaxed min_spacing for '%s': %.1fm -> %.1fm (AOI too small for %d pts)",
                source_config.name, float(source_config.min_spacing_m), eff_min_spacing_m, desired_floor
            )

    # Scale grid sizes down when needed (but not below eff_min_spacing)
    grid_hi = float(max(eff_min_spacing_m, min(config.grid_high_complexity, eff_min_spacing_m * 2.0)))
    grid_med = float(max(eff_min_spacing_m, min(config.grid_medium_complexity, eff_min_spacing_m * 2.0)))
    grid_lo = float(max(eff_min_spacing_m, min(config.grid_low_complexity, eff_min_spacing_m * 2.0)))

    # Complexity class masks
    hi = complexity_scores >= config.high_complexity_threshold
    med = (complexity_scores >= config.low_complexity_threshold) & (~hi)
    lo = ~hi & ~med

    def _thin(mask: np.ndarray, grid_m: float, allow_multi: bool) -> None:
        """Thin points in a complexity class by selecting best point(s) per grid cell.

        Performance:
          The old implementation did:
              for cell_id in np.unique(cell_ids):
                  np.where(cell_ids == cell_id)
          which becomes extremely slow for multi-million point sources.
          We instead sort once (O(N log N)) and scan contiguous runs (O(N)).
        """
        if not np.any(mask):
            return
        idx = np.where(mask)[0]

        g = float(max(grid_m, eff_min_spacing_m))
        if g <= 0:
            g = float(source_config.min_spacing_m) if source_config.min_spacing_m > 0 else 30.0

        xb = np.floor(x_m[idx] / g).astype(np.int64)
        yb = np.floor(y_m[idx] / g).astype(np.int64)
        cell_ids = xb * 10_000_000 + yb

        # Priority score for selection within a cell
        cell_priorities = source_values[idx] * (1.0 + complexity_scores[idx])

        order = np.argsort(cell_ids, kind='mergesort')  # stable
        cell_ids_s = cell_ids[order]
        prio_s = cell_priorities[order]
        idx_s = idx[order]

        if cell_ids_s.size == 0:
            return

        boundaries = np.flatnonzero(np.diff(cell_ids_s)) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, cell_ids_s.size]

        for s, e in zip(starts, ends):
            seg = slice(int(s), int(e))
            seg_prio = prio_s[seg]
            if seg_prio.size == 0:
                continue

            best_local = int(np.argmax(seg_prio))
            best_idx = int(idx_s[seg][best_local])
            keep[best_idx] = True

            if allow_multi and (e - s) > 3:
                k = 3
                if (e - s) <= k:
                    top_locals = np.arange(e - s, dtype=np.int64)
                else:
                    part = np.argpartition(seg_prio, -k)[-k:]
                    top_locals = part[np.argsort(seg_prio[part])]
                for tl in top_locals:
                    keep[int(idx_s[seg][int(tl)])] = True

    _thin(hi, grid_hi, allow_multi=True)
    _thin(med, grid_med, allow_multi=False)
    _thin(lo, grid_lo, allow_multi=False)

    # Ensure we don't under-sample relative to the configured retention_target.
    # The grid-based thinning above enforces spacing/representativeness, but can
    # inadvertently keep far fewer points than intended for dense survey sources.
    desired_n_raw = int(np.ceil(float(source_config.retention_target) * n_points))
    # Avoid runaway growth per-source; a global cap later enforces target_total_points.
    desired_cap = int(max(config.target_total_points * 2, 5000))
    desired_n = int(min(n_points, min(desired_n_raw, desired_cap)))
    kept_n = int(keep.sum())
    if desired_n > 0 and kept_n < desired_n:
        # Densify by allowing multiple points per grid cell (still spatially balanced).
        # Use the relaxed min spacing (if any) when densifying.
        #
        # For very large sources (multi-million points), building a Python dict
        # of cells->indices can dominate runtime/memory. In those cases, fall back to
        # a fast vectorized top-k selection from the remaining points.
        if n_points > 2_000_000:
            need = int(desired_n - kept_n)
            if need > 0:
                rng = np.random.default_rng(int(getattr(config, 'random_seed', 1337)))
                remaining = np.where(~keep)[0]
                if remaining.size > 0:
                    # Prefer higher complexity and slightly prefer deeper points.
                    prio = (
                        np.nan_to_num(complexity_scores[remaining], nan=0.0) * 1.0
                        + np.nan_to_num(np.abs(depths[remaining]), nan=0.0) * 0.05
                        + rng.random(remaining.size) * 1e-6
                    )
                    if need >= remaining.size:
                        keep[remaining] = True
                    else:
                        pick = remaining[np.argpartition(prio, -need)[-need:]]
                        keep[pick] = True
            return keep
        g_base = float(max(eff_min_spacing_m, grid_lo / 2.0))
        if g_base <= 0:
            g_base = float(max(eff_min_spacing_m, 30.0))

        xb_all = np.floor(x_m / g_base).astype(np.int64)
        yb_all = np.floor(y_m / g_base).astype(np.int64)

        # Priority within a cell: favor higher complexity and slightly favor deeper points.
        prio = (np.nan_to_num(complexity_scores, nan=0.0) * 1.0) + (np.nan_to_num(np.abs(depths), nan=0.0) * 0.05)
        rng = np.random.default_rng(int(getattr(config, 'random_seed', 1337)))
        prio = prio + (rng.random(n_points) * 1e-6)

        cells: dict[tuple[int, int], list[int]] = {}
        for ii in range(n_points):
            key = (int(xb_all[ii]), int(yb_all[ii]))
            cells.setdefault(key, []).append(int(ii))

        n_cells = max(1, len(cells))
        k = int(np.ceil(desired_n / n_cells))
        max_k = int(min(50, max(5, k * 3)))

        selected = set(np.where(keep)[0].tolist())
        while len(selected) < desired_n and k <= max_k:
            for idx_list in cells.values():
                if len(selected) >= desired_n:
                    break
                if len(idx_list) <= k:
                    top = idx_list
                else:
                    arr = np.array(idx_list, dtype=np.int64)
                    top = arr[np.argsort(prio[arr])[-k:]].tolist()
                for jj in top:
                    selected.add(int(jj))
                    if len(selected) >= desired_n:
                        break
            k += 1

        keep[:] = False
        keep[list(selected)] = True

    return keep

def depth_stratified_sampling(
    df: pd.DataFrame,
    config: SamplingConfig
) -> pd.DataFrame:
    """
    Apply depth stratification to ensure depth range coverage.
    
    Args:
        df: DataFrame with depth_m column
        config: Sampling configuration
    
    Returns:
        DataFrame with balanced depth representation
    """
    # Create quantile-based depth bins
    depths = df['depth_m'].values
    valid_depths = depths[np.isfinite(depths)]
    
    if len(valid_depths) < config.min_points_per_bin * 2:
        log.warning("Too few points for depth stratification")
        return df
    
    # Compute quantile bins
    try:
        quantiles = np.linspace(0, 1, config.depth_bins + 1)
        bin_edges = np.quantile(valid_depths, quantiles)
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        df['depth_bin'] = pd.cut(df['depth_m'], bins=bin_edges, labels=False, include_lowest=True)
        
        # Log depth distribution
        bin_counts = df['depth_bin'].value_counts().sort_index()
        log.info("Depth stratification: %s bins", len(bin_edges)-1)
        for bin_idx, count in bin_counts.items():
            if np.isfinite(bin_idx):
                depth_range = f"[{bin_edges[int(bin_idx)]:.1f}, {bin_edges[int(bin_idx)+1]:.1f}]m"
                log.info("  Bin %s: %s → %s points", int(bin_idx), depth_range, count)
        
        return df
        
    except Exception as e:
        log.warning("Depth stratification failed: %s", e)
        df['depth_bin'] = 0
        return df

def compute_influence_zones(
    points: np.ndarray,
    influence_radius_m: float
) -> cKDTree:
    """
    Compute spatial influence zones for high-priority points.
    
    Args:
        points: (N, 2) array of coordinates
        influence_radius_m: Radius in meters
    
    Returns:
        KDTree for efficient radius queries
    """
    return cKDTree(points)

def identify_gap_regions(
    all_points: np.ndarray,
    covered_points_kdtree: cKDTree,
    influence_radius_m: float,
    grid_resolution: float = 0.001  # ~100m
) -> np.ndarray:
    """
    Identify spatial gaps not covered by high-priority data.

    This must be fast for multi-million point sources. Use a vectorized KDTree query.

    Args:
        all_points: All point locations (lon/lat degrees)
        covered_points_kdtree: KDTree of Tier 1 points (lon/lat degrees)
        influence_radius_m: Coverage radius in meters
        grid_resolution: (unused; retained for backward compatibility)

    Returns:
        Boolean array indicating points in gap regions
    """
    if len(all_points) == 0:
        return np.zeros(0, dtype=bool)

    # If there is no Tier 1 coverage, everything is a gap.
    try:
        n_cov = int(getattr(covered_points_kdtree, 'n', 0))
    except Exception:
        n_cov = 0
    if n_cov == 0:
        return np.ones(len(all_points), dtype=bool)

    # Rough conversion meters -> degrees (OK for small AOIs / local use)
    influence_radius_deg = float(influence_radius_m) / 111_000.0

    # Vectorized nearest-neighbor distances. Chunk large queries to avoid peak memory
    # for multi-million point sources.
    pts = np.asarray(all_points, dtype=np.float32)
    out = np.empty((len(pts),), dtype=np.float32)
    chunk = 1_000_000
    for i0 in range(0, len(pts), chunk):
        sl = slice(i0, min(i0 + chunk, len(pts)))
        try:
            d, _ = covered_points_kdtree.query(pts[sl], k=1, workers=-1)
        except TypeError:
            # Older scipy: no 'workers' arg
            d, _ = covered_points_kdtree.query(pts[sl], k=1)
        out[sl] = np.asarray(d, dtype=np.float32)

    return out > float(influence_radius_deg)

def adaptive_spatial_sample(
    df: pd.DataFrame,
    source_configs: Optional[Dict[str, SourceConfig]] = None,
    sampling_config: Optional[SamplingConfig] = None,
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main adaptive sampling function with priority-based source preservation.
    
    Args:
        df: Input dataframe with columns: longitude, latitude, depth_m, source, sample_weight
        source_configs: Dictionary mapping source names to SourceConfig objects
        sampling_config: Global sampling configuration
        aoi_bounds: (W, E, S, N) bounds for gap detection
    
    Returns:
        Tuple of (sampled_df, statistics_dict)
    """
    log.info("Adaptive spatial sampling (priority-based).")
    
    # Use default configs if not provided
    if source_configs is None:
        source_configs = DEFAULT_SOURCE_CONFIGS.copy()
    
    if sampling_config is None:
        sampling_config = SamplingConfig()
    
    # Input statistics
    n_input = len(df)
    log.info("Input: %s points", format(n_input, ","))

    # Keep a stable identifier so we can top-up from unselected rows later.
    df = df.copy()
    if '_row_id' not in df.columns:
        df['_row_id'] = np.arange(len(df), dtype=np.int64)
    
    # Ensure required columns
    required_cols = ['longitude', 'latitude', 'depth_m', 'source']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Add sample_weight if missing
    if 'sample_weight' not in df.columns:
        df['sample_weight'] = 1.0
    
    # Normalize source names
    df['source_normalized'] = df['source'].str.lower().str.strip()

    # Derive a stable source_key used for tiering.
    # This lets 'extra_xyz' carry per-file provenance (e.g., extra_xyz:hydronos),
    # and promotes vetted lidar/sonar subsets to Tier 1 by heuristic.
    AUTH_TAGS = (
        "hydronos", "ehydro", "lidar", "topobathy", "sonar", "multibeam", "mbes",
        "singlebeam", "sounding", "noaa", "usace", "usgs"
    )
    def _source_key(s: str) -> str:
        if not isinstance(s, str):
            return "atl03"
        s = s.lower().strip()
        if s.startswith("extra_xyz:"):
            tag = s.split(":", 1)[1]
            if any(t in tag for t in AUTH_TAGS):
                return "extra_xyz_auth"
            return "extra_xyz"
        return s

    df['source_key'] = df['source_normalized'].map(_source_key)
    
    # =========================================================================
    # STAGE 1: DEPTH STRATIFICATION
    # =========================================================================
    log.info("[STAGE 1] Depth Stratification (%s bins)", sampling_config.depth_bins)
    df = depth_stratified_sampling(df, sampling_config)
    
    # =========================================================================
    # STAGE 2: COMPUTE COMPLEXITY SCORES
    # =========================================================================
    log.info("[STAGE 2] Computing Bathymetric Complexity")
    
    # OPTIMIZATION:
    #   Complexity computation requires neighbor queries; doing it directly on millions
    #   of points is expensive. We therefore compute complexity on a *coarse* subset,
    #   then transfer those scores back to the full dataset via nearest-neighbor lookup.
    if len(df) > 100000:
        log.info("Large dataset detected (%s points)", format(len(df), ","))
        log.info("Applying coarse pre-sampling for efficiency (complexity only)...")

        points_all = df[['longitude', 'latitude']].to_numpy()
        lon_min, lat_min = np.nanmin(points_all, axis=0)

        # ~100m grid in degrees (rough; used only to thin for complexity computation)
        grid_size = 0.001
        lon_bins = ((points_all[:, 0] - lon_min) / grid_size).astype(np.int64)
        lat_bins = ((points_all[:, 1] - lat_min) / grid_size).astype(np.int64)
        grid_cell = lon_bins * 10_000 + lat_bins

        # Priority: prefer higher-weight sources, with tiny jitter to break ties
        if 'source_weight' in df.columns:
            w = pd.to_numeric(df['source_weight'], errors='coerce').fillna(1.0).to_numpy()
        elif 'sample_weight' in df.columns:
            w = pd.to_numeric(df['sample_weight'], errors='coerce').fillna(1.0).to_numpy()
        elif 'weight' in df.columns:
            w = pd.to_numeric(df['weight'], errors='coerce').fillna(1.0).to_numpy()
        else:
            w = df['source_normalized'].map(lambda s: source_configs.get(s, source_configs['atl03']).weight).to_numpy()
        rng = np.random.default_rng(int(getattr(sampling_config, 'random_seed', 1337)))
        priority = w * (1.0 + rng.random(len(df)) * 0.01)

        tmp = df.copy()
        tmp['_grid_cell'] = grid_cell
        tmp['_priority'] = priority

        tmp_sorted = tmp.sort_values(['_grid_cell', '_priority'], ascending=[True, False])
        coarse_mask = ~tmp_sorted['_grid_cell'].duplicated()
        coarse_df = tmp_sorted.loc[coarse_mask].copy()
        coarse_df = coarse_df.drop(columns=['_grid_cell', '_priority'])

        log.info("Coarse pre-sampling: %s → %s points", format(len(df), ","), format(len(coarse_df), ","))

        # Compute complexity on the coarse set
        c_points = coarse_df[['longitude', 'latitude']].values
        c_depths = coarse_df['depth_m'].values
        c_tree = cKDTree(c_points)
        c_complexity = compute_local_complexity(c_points, c_depths, c_tree, sampling_config)
        coarse_df['complexity_score'] = c_complexity

        # Detect transects on coarse set (optional) and map to full set later
        if sampling_config.detect_transects:
            log.info("Detecting linear transects...")
            coarse_df['is_transect'] = detect_transects(c_points)
            log.info(f"Detected {int(coarse_df['is_transect'].sum())} transect points")
        else:
            coarse_df['is_transect'] = False

        # Transfer complexity (and transect flag) to all points by NN in coarse set
        nn_dist, nn_idx = c_tree.query(points_all, k=1)
        df = df.copy()
        df['complexity_score'] = coarse_df['complexity_score'].to_numpy()[nn_idx]
        df['is_transect'] = coarse_df['is_transect'].to_numpy()[nn_idx]
        working_df = df
        complexity_scores = df['complexity_score'].to_numpy()

    else:
        # Compute complexity directly on the full set
        points = df[['longitude', 'latitude']].values
        depths = df['depth_m'].values
        kdtree = cKDTree(points)
        complexity_scores = compute_local_complexity(points, depths, kdtree, sampling_config)
        df = df.copy()
        df['complexity_score'] = complexity_scores

        if sampling_config.detect_transects:
            log.info("Detecting linear transects...")
            df['is_transect'] = detect_transects(points)
            log.info(f"Detected {int(df['is_transect'].sum())} transect points")
        else:
            df['is_transect'] = False

        working_df = df
    
    log.info("Complexity distribution:")
    log.info(f"  High (>{sampling_config.high_complexity_threshold}): "
            f"{(complexity_scores > sampling_config.high_complexity_threshold).sum()} points")
    
    medium_count = ((complexity_scores >= sampling_config.low_complexity_threshold) & 
                    (complexity_scores <= sampling_config.high_complexity_threshold)).sum()
    log.info(f"  Medium ({sampling_config.low_complexity_threshold}-{sampling_config.high_complexity_threshold}): "
            f"{medium_count} points")
    
    log.info(f"  Low (<{sampling_config.low_complexity_threshold}): "
            f"{(complexity_scores < sampling_config.low_complexity_threshold).sum()} points")
    
    # (Transect flags already set above)
    
    # =========================================================================
    # STAGE 3: PRIORITY-BASED SOURCE PRESERVATION
    # =========================================================================
    log.info("[STAGE 3] Priority-Based Source Preservation")
    
    # Assign tiers and priorities to sources
    working_df['tier'] = working_df['source_key'].map(
        lambda s: source_configs.get(s, source_configs['atl03']).tier
    )
    working_df['source_weight'] = working_df['source_key'].map(
        lambda s: source_configs.get(s, source_configs['atl03']).weight
    )
    working_df['retention_target'] = working_df['source_key'].map(
        lambda s: source_configs.get(s, source_configs['atl03']).retention_target
    )
    
    # Separate by tier
    tier1_mask = working_df['tier'] == 1
    tier2_mask = working_df['tier'] == 2
    tier3_mask = working_df['tier'] == 3
    
    tier1_df = working_df[tier1_mask].copy()
    tier2_df = working_df[tier2_mask].copy()
    tier3_df = working_df[tier3_mask].copy()
    
    log.info("Tier distribution:")
    log.info(f"  Tier 1 (High accuracy): {len(tier1_df):,} points from {tier1_df['source'].nunique()} sources")
    log.info(f"  Tier 2 (Medium accuracy): {len(tier2_df):,} points from {tier2_df['source'].nunique()} sources")
    log.info(f"  Tier 3 (Gap fillers): {len(tier3_df):,} points from {tier3_df['source'].nunique()} sources")
    
    # -------------------------------------------------------------------------
    # 3.1: Process Tier 1 (Preserve aggressively)
    # -------------------------------------------------------------------------
    selected_tier1 = []
    
    if len(tier1_df) > 0:
        log.info("[TIER 1] Processing high-accuracy sources:")
        
        for source in tier1_df['source'].unique():
            source_mask = tier1_df['source'] == source
            source_df = tier1_df[source_mask].copy()
            source_norm = source_df['source_key'].iloc[0]
            
            if source_norm not in source_configs:
                log.warning("[TIER 1] Unknown source %r, using default Tier 1 config", source)
                source_config = SourceConfig(source_norm, tier=1, weight=10, retention_target=0.75,
                                            min_spacing_m=20, influence_radius_m=60)
            else:
                source_config = source_configs[source_norm]
            
            log.info("  %s: %s points (target retention: %.0f%%)", source, len(source_df), source_config.retention_target*100)
            
            # Apply gentle adaptive thinning
            keep_mask = adaptive_grid_thinning(
                source_df[['longitude', 'latitude']].values,
                source_df['depth_m'].values,
                source_df['complexity_score'].values,
                source_df['source_weight'].values,
                sampling_config,
                source_config
            )
            
            selected_source = source_df[keep_mask].copy()
            selected_tier1.append(selected_source)
            
            retention_rate = len(selected_source) / len(source_df)
            log.info("    → Kept %s points (%.1f%% retention)", len(selected_source), retention_rate*100)
    
    tier1_selected = pd.concat(selected_tier1) if selected_tier1 else pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # 3.2: Identify gaps not covered by Tier 1
    # -------------------------------------------------------------------------
    log.info("[GAP ANALYSIS] Identifying spatial gaps:")
    
    if len(tier1_selected) > 0:
        tier1_kdtree = compute_influence_zones(
            tier1_selected[['longitude', 'latitude']].values,
            influence_radius_m=75.0  # Max influence radius
        )
        
        # Mark Tier 2/3 points that are in gaps
        if len(tier2_df) > 0:
            tier2_in_gap = identify_gap_regions(
                tier2_df[['longitude', 'latitude']].values,
                tier1_kdtree,
                influence_radius_m=50.0
            )
            tier2_df['in_gap'] = tier2_in_gap
        else:
            tier2_in_gap = np.array([], dtype=bool)
        
        if len(tier3_df) > 0:
            tier3_in_gap = identify_gap_regions(
                tier3_df[['longitude', 'latitude']].values,
                tier1_kdtree,
                influence_radius_m=30.0
            )
            tier3_df['in_gap'] = tier3_in_gap
        else:
            tier3_in_gap = np.array([], dtype=bool)
        
        log.info("  Tier 2 points in gaps: %s / %s", format(tier2_in_gap.sum(), ","), format(len(tier2_df), ","))
        log.info("  Tier 3 points in gaps: %s / %s", format(tier3_in_gap.sum(), ","), format(len(tier3_df), ","))
    else:
        # No Tier 1 data - all Tier 2/3 needed
        log.info("  No Tier 1 data - all areas are gaps")
        if len(tier2_df) > 0:
            tier2_df['in_gap'] = True
        if len(tier3_df) > 0:
            tier3_df['in_gap'] = True
    
    # -------------------------------------------------------------------------
    # 3.3: Process Tier 2 (Fill moderate gaps)
    # -------------------------------------------------------------------------
    selected_tier2 = []
    
    if len(tier2_df) > 0:
        log.info("[TIER 2] Processing gap-filling sources:")
        
        # Only sample from gaps (or all if no Tier 1)
        tier2_gaps = tier2_df[tier2_df.get('in_gap', True)].copy()
        
        for source in tier2_gaps['source'].unique():
            source_mask = tier2_gaps['source'] == source
            source_df = tier2_gaps[source_mask].copy()
            source_norm = source_df['source_key'].iloc[0]
            
            if source_norm not in source_configs:
                source_config = SourceConfig(source_norm, tier=2, weight=6, retention_target=0.3,
                                            min_spacing_m=40, influence_radius_m=50)
            else:
                source_config = source_configs[source_norm]
            
            # Fast pre-thin for extremely dense sources before adaptive thinning.
            # This prevents pathological runtimes (multi-million point sources)
            # when the final output is globally capped anyway.
            max_keep = int(getattr(sampling_config, 'max_points_per_source_prethin', 500_000))
            if max_keep > 0 and len(source_df) > max_keep:
                cell_m = float(source_config.min_spacing_m) * float(getattr(sampling_config, 'prethin_cell_scale', 1.0))
                pts = source_df[['longitude', 'latitude']].values
                pre_keep = fast_grid_prethin_lonlat(pts, cell_m=cell_m, max_keep=max_keep, seed=0)
                before_n = len(source_df)
                source_df = source_df[pre_keep].copy()
                log.info(
                    f"  {source}: {before_n:,} gap points → pre-thin to {len(source_df):,} "
                    f"(cell={cell_m:.1f} m, cap={max_keep:,}; target retention: {source_config.retention_target*100:.0f}%)"
                )
            else:
                log.info("  %s: %s gap points (target retention: %.0f%%)", source, format(len(source_df), ","), source_config.retention_target*100)
            
            # Apply moderate thinning
            keep_mask = adaptive_grid_thinning(
                source_df[['longitude', 'latitude']].values,
                source_df['depth_m'].values,
                source_df['complexity_score'].values,
                source_df['source_weight'].values,
                sampling_config,
                source_config
            )
            
            selected_source = source_df[keep_mask].copy()
            selected_tier2.append(selected_source)
            
            retention_rate = len(selected_source) / len(source_df) if len(source_df) > 0 else 0
            log.info("    → Kept %s points (%.1f%% retention)", len(selected_source), retention_rate*100)
    
    tier2_selected = pd.concat(selected_tier2) if selected_tier2 else pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # 3.4: Process Tier 3 (Fill remaining gaps)
    # -------------------------------------------------------------------------
    selected_tier3 = []
    
    if len(tier3_df) > 0:
        log.info("[TIER 3] Processing emergency gap fillers:")
        
        # Only sample from gaps
        tier3_gaps = tier3_df[tier3_df.get('in_gap', True)].copy()
        
        for source in tier3_gaps['source'].unique():
            source_mask = tier3_gaps['source'] == source
            source_df = tier3_gaps[source_mask].copy()
            source_norm = source_df['source_key'].iloc[0]
            
            if source_norm not in source_configs:
                source_config = SourceConfig(source_norm, tier=3, weight=3, retention_target=0.1,
                                            min_spacing_m=60, influence_radius_m=30)
            else:
                source_config = source_configs[source_norm]
            
            max_keep = int(getattr(sampling_config, 'max_points_per_source_prethin', 500_000))
            if max_keep > 0 and len(source_df) > max_keep:
                cell_m = float(source_config.min_spacing_m) * float(getattr(sampling_config, 'prethin_cell_scale', 1.0))
                pts = source_df[['longitude', 'latitude']].values
                pre_keep = fast_grid_prethin_lonlat(pts, cell_m=cell_m, max_keep=max_keep, seed=0)
                before_n = len(source_df)
                source_df = source_df[pre_keep].copy()
                log.info(
                    f"  {source}: {before_n:,} gap points → pre-thin to {len(source_df):,} "
                    f"(cell={cell_m:.1f} m, cap={max_keep:,}; target retention: {source_config.retention_target*100:.0f}%)"
                )
            else:
                log.info("  %s: %s gap points (target retention: %.0f%%)", source, format(len(source_df), ","), source_config.retention_target*100)
            
            # Apply aggressive thinning
            keep_mask = adaptive_grid_thinning(
                source_df[['longitude', 'latitude']].values,
                source_df['depth_m'].values,
                source_df['complexity_score'].values,
                source_df['source_weight'].values,
                sampling_config,
                source_config
            )
            
            selected_source = source_df[keep_mask].copy()
            selected_tier3.append(selected_source)
            
            retention_rate = len(selected_source) / len(source_df) if len(source_df) > 0 else 0
            log.info("    → Kept %s points (%.1f%% retention)", len(selected_source), retention_rate*100)
    
    tier3_selected = pd.concat(selected_tier3) if selected_tier3 else pd.DataFrame()
    
    # =========================================================================
    # COMBINE & VALIDATE
    # =========================================================================
    log.info("Combining selected points:")
    
    final_dfs = []
    if len(tier1_selected) > 0:
        final_dfs.append(tier1_selected)
    if len(tier2_selected) > 0:
        final_dfs.append(tier2_selected)
    if len(tier3_selected) > 0:
        final_dfs.append(tier3_selected)
    
    if not final_dfs:
        log.error("No points survived sampling.")
        return df.head(100), {}  # Return something to avoid complete failure
    
    result_df = pd.concat(final_dfs, ignore_index=True)

    # ---------------------------------------------------------------------
    # Global cap: keep the dataset bounded for downstream raster sampling.
    # The earlier stages focus on spatial representativeness; this step
    # enforces an overall size target while preserving tier priority.
    # ---------------------------------------------------------------------
    target_n = int(sampling_config.target_total_points)
    if target_n > 0 and len(result_df) > target_n:
        log.info("Capping sampled set to target_total_points=%s (current=%s)", format(target_n, ","), format(len(result_df), ","))

        # Priority score: (tier priority) * (source_weight) * (sample_weight) * (1 + complexity)
        sw = pd.to_numeric(result_df.get('source_weight', 1.0), errors='coerce').fillna(1.0)
        w = pd.to_numeric(result_df.get('sample_weight', 1.0), errors='coerce').fillna(1.0)
        c = pd.to_numeric(result_df.get('complexity_score', 0.0), errors='coerce').fillna(0.0)
        rng = np.random.default_rng(int(getattr(sampling_config, 'random_seed', 1337)))
        jitter = 1.0 + (rng.random(len(result_df)) * 0.01)
        priority = (sw.to_numpy() * w.to_numpy() * (1.0 + c.to_numpy())) * jitter

        # Always keep all Tier 1 points if present; then fill Tier 2, then Tier 3
        tier_vals = pd.to_numeric(result_df.get('tier', 3), errors='coerce').fillna(3).astype(int).to_numpy()
        keep_idx: list[int] = []

        for tier_id in (1, 2, 3):
            idx = np.where(tier_vals == tier_id)[0]
            if idx.size == 0:
                continue
            if len(keep_idx) >= target_n:
                break
            remaining = target_n - len(keep_idx)
            if idx.size <= remaining:
                keep_idx.extend(idx.tolist())
            else:
                # Take top-N by priority within tier
                ord_idx = idx[np.argsort(priority[idx])[::-1]]
                keep_idx.extend(ord_idx[:remaining].tolist())

        keep_idx = np.array(sorted(set(keep_idx)), dtype=int)
        result_df = result_df.iloc[keep_idx].copy()

    # ---------------------------------------------------------------------
    # Minimum-size floor: if we ended up with too few points, top-up from
    # remaining candidates (preserving tier priority where possible).
    # ---------------------------------------------------------------------
    min_n = int(getattr(sampling_config, 'min_total_points', 200))
    if min_n > 0 and len(result_df) < min_n and n_input > len(result_df):
        log.warning(
            "[SAMPLING] Sampled set is small (%d pts). Topping up to min_total_points=%d.",
            len(result_df), min_n
        )
        selected_ids = set(pd.to_numeric(result_df.get('_row_id', []), errors='coerce').dropna().astype(int).tolist())
        remaining = df[~df['_row_id'].isin(selected_ids)].copy()

        # Prefer Tier 2, then Tier 3, then Tier 1 (Tier 1 should normally already be fully kept).
        tiers = pd.to_numeric(remaining.get('tier', 3), errors='coerce').fillna(3).astype(int)
        remaining = remaining.assign(_tier=tiers)

        need = int(min_n - len(result_df))
        topups = []
        for tier_id in (2, 3, 1):
            if need <= 0:
                break
            cand = remaining[remaining['_tier'] == tier_id]
            if cand.empty:
                continue
            take_n = int(min(need, len(cand)))
            topups.append(cand.sample(n=take_n, replace=False, random_state=sampling_config.random_seed))
            need -= take_n
        if topups:
            result_df = pd.concat([result_df, *topups], ignore_index=True)
    
    # Recompute tier breakdown after any cap/top-up
    tier_vals_final = pd.to_numeric(result_df.get('tier', 3), errors='coerce').fillna(3).astype(int)
    tier1_selected = result_df[tier_vals_final == 1]
    tier2_selected = result_df[tier_vals_final == 2]
    tier3_selected = result_df[tier_vals_final == 3]

    # Compute statistics
    stats = {
        'input_points': n_input,
        'output_points': len(result_df),
        'reduction_pct': 100.0 * (1.0 - len(result_df) / n_input),
        'tier1': {
            'input': len(tier1_df),
            'output': len(tier1_selected),
            'retention': len(tier1_selected) / max(len(tier1_df), 1),
            'sources': tier1_selected['source'].unique().tolist() if len(tier1_selected) > 0 else [],
        },
        'tier2': {
            'input': len(tier2_df),
            'output': len(tier2_selected),
            'retention': len(tier2_selected) / max(len(tier2_df), 1),
            'sources': tier2_selected['source'].unique().tolist() if len(tier2_selected) > 0 else [],
        },
        'tier3': {
            'input': len(tier3_df),
            'output': len(tier3_selected),
            'retention': len(tier3_selected) / max(len(tier3_df), 1),
            'sources': tier3_selected['source'].unique().tolist() if len(tier3_selected) > 0 else [],
        },
        'effective_weights': {}
    }
    
    # Compute effective training influence
    for tier_name, tier_df in [('tier1', tier1_selected), ('tier2', tier2_selected), ('tier3', tier3_selected)]:
        if len(tier_df) > 0:
            tier_weight = (tier_df['sample_weight'] * tier_df['source_weight']).sum()
            stats['effective_weights'][tier_name] = float(tier_weight)
    
    total_effective_weight = sum(stats['effective_weights'].values())
    
    log.info("  Tier 1: %s points (%.1f%%)", format(len(tier1_selected), ","), len(tier1_selected)/len(result_df)*100)
    log.info("  Tier 2: %s points (%.1f%%)", format(len(tier2_selected), ","), len(tier2_selected)/len(result_df)*100)
    log.info("  Tier 3: %s points (%.1f%%)", format(len(tier3_selected), ","), len(tier3_selected)/len(result_df)*100)
    log.info("  Total: %s points (%.1f%% of input)", format(len(result_df), ","), 100.0 * len(result_df)/n_input)
    
    log.info("Effective Training Influence:")
    for tier_name in ['tier1', 'tier2', 'tier3']:
        if tier_name in stats['effective_weights']:
            tier_weight = stats['effective_weights'][tier_name]
            tier_pct = 100.0 * tier_weight / max(total_effective_weight, 1)
            log.info("  %s: %.1f%% influence", tier_name.upper(), tier_pct)
    
    return result_df, stats
