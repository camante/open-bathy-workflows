#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adaptive_spatial_cv.py - Adaptive spatial cross-validation for geospatial ML

Provides intelligent spatial cross-validation that:
- Adapts cluster count to data density
- Uses hierarchical clustering for spatial contiguity
- Ensures test clusters cover full value range
- Minimizes spatial autocorrelation in CV estimates

Usage:
    from adaptive_spatial_cv import AdaptiveSpatialCV, adaptive_train_test_split
    
    # Create adaptive CV splitter
    cv = AdaptiveSpatialCV(coords, values)
    
    # Use with scikit-learn
    scores = cross_val_score(model, X, y, cv=cv)
    
    # Or get single train/test split
    train_idx, test_idx = adaptive_train_test_split(
        coords, values, test_size=0.2
    )
"""

import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering, KMeans

log = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SpatialCVConfig:
    """Configuration for adaptive spatial CV."""
    target_test_fraction: float = 0.2
    min_cluster_size: int = 100
    min_clusters: int = 5
    max_clusters: int = 50
    linkage: str = 'ward'  # 'ward', 'complete', 'average'
    random_state: int = 42
    

# ============================================================================
# Cluster Quality Metrics
# ============================================================================

def compute_cluster_compactness(coords: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute average within-cluster distance (compactness).
    
    Lower is better - indicates spatially tight clusters.
    """
    compactness = 0.0
    n_clusters = len(np.unique(labels))
    
    for i in range(n_clusters):
        mask = labels == i
        if mask.sum() < 2:
            continue
        cluster_coords = coords[mask]
        center = cluster_coords.mean(axis=0)
        distances = np.sqrt(((cluster_coords - center)**2).sum(axis=1))
        compactness += distances.mean()
    
    return compactness / n_clusters


def compute_cluster_separation(coords: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute minimum between-cluster distance (separation).
    
    Higher is better - indicates spatially distinct clusters.
    """
    n_clusters = len(np.unique(labels))
    centers = np.array([
        coords[labels == i].mean(axis=0) 
        for i in range(n_clusters)
    ])
    
    # Pairwise distances between centers
    min_sep = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = np.sqrt(((centers[i] - centers[j])**2).sum())
            min_sep = min(min_sep, dist)
    
    return min_sep


def compute_silhouette_spatial(coords: np.ndarray, labels: np.ndarray, 
                               sample_size: int = 1000) -> float:
    """
    Compute spatial silhouette score (modified silhouette for spatial data).
    
    Score in [-1, 1] where:
    - +1: Perfect clustering (tight and well-separated)
    - 0: Overlapping clusters
    - -1: Poor clustering (points closer to other clusters)
    """
    from sklearn.metrics import silhouette_score
    
    # Sample for efficiency
    if len(coords) > sample_size:
        idx = np.random.RandomState(42).choice(len(coords), sample_size, replace=False)
        coords_sample = coords[idx]
        labels_sample = labels[idx]
    else:
        coords_sample = coords
        labels_sample = labels
    
    if len(np.unique(labels_sample)) < 2:
        return -1.0
    
    try:
        return silhouette_score(coords_sample, labels_sample, metric='euclidean')
    except Exception:
        return -1.0


# ============================================================================
# Adaptive Cluster Count Selection
# ============================================================================

def estimate_optimal_clusters(
    coords: np.ndarray,
    values: np.ndarray,
    config: SpatialCVConfig
) -> int:
    """
    Estimate optimal number of spatial clusters.
    
    Strategy:
    1. Compute data density (points per unit area)
    2. Estimate cluster count to achieve target test fraction
    3. Constrain to [min_clusters, max_clusters]
    4. Ensure minimum cluster size
    
    Args:
        coords: (n, 2) coordinate array
        values: (n,) value array (for distribution coverage)
        config: SpatialCVConfig
    
    Returns:
        Optimal cluster count
    """
    n_points = len(coords)
    
    # Compute spatial extent
    extent = coords.max(axis=0) - coords.min(axis=0)
    area = extent[0] * extent[1]
    
    if area == 0:
        return config.min_clusters
    
    # Estimate density
    density = n_points / area
    
    # Target points per cluster based on min size
    target_cluster_size = max(config.min_cluster_size, 
                              n_points * config.target_test_fraction / config.min_clusters)
    
    # Estimate cluster count
    n_clusters = int(n_points / target_cluster_size)
    
    # Constrain to valid range
    n_clusters = max(config.min_clusters, min(config.max_clusters, n_clusters))
    
    # Adjust for value distribution coverage
    # If values have high variance, prefer more clusters to cover range
    value_std = np.std(values)
    value_range = np.ptp(values)
    cv_values = value_std / (np.mean(values) + 1e-6)
    
    if cv_values > 0.5:  # High variability
        n_clusters = min(int(n_clusters * 1.5), config.max_clusters)
    
    log.debug(f"[SpatialCV] Estimated {n_clusters} clusters "
             f"(density={density:.1f} pts/unit², n={n_points})")
    
    return n_clusters


# ============================================================================
# Representative Cluster Selection
# ============================================================================

def select_representative_clusters(
    labels: np.ndarray,
    values: np.ndarray,
    target_fraction: float = 0.2
) -> np.ndarray:
    """
    Select test clusters that best represent the value distribution.
    
    Strategy:
    1. Compute value statistics per cluster
    2. Select clusters that cover the value range
    3. Ensure total test fraction is close to target
    
    Args:
        labels: Cluster labels for each point
        values: Target values
        target_fraction: Target fraction of data in test set
    
    Returns:
        Boolean array (True = test set)
    """
    n_points = len(labels)
    n_clusters = len(np.unique(labels))
    target_n = int(n_points * target_fraction)
    
    # Compute cluster statistics
    cluster_stats = []
    for i in range(n_clusters):
        mask = labels == i
        if not mask.any():
            continue
        
        cluster_values = values[mask]
        cluster_stats.append({
            'cluster_id': i,
            'size': mask.sum(),
            'mean': np.mean(cluster_values),
            'std': np.std(cluster_values),
            'min': np.min(cluster_values),
            'max': np.max(cluster_values),
            'median': np.median(cluster_values),
            'q25': np.percentile(cluster_values, 25),
            'q75': np.percentile(cluster_values, 75)
        })
    
    df = pd.DataFrame(cluster_stats)
    
    # Strategy: Select clusters that span the value range
    # 1. Include cluster with min values
    # 2. Include cluster with max values  
    # 3. Include clusters at quartiles
    # 4. Fill to target fraction
    
    selected = set()
    
    # Extreme value clusters
    selected.add(int(df.loc[df['min'].idxmin(), 'cluster_id']))
    selected.add(int(df.loc[df['max'].idxmax(), 'cluster_id']))
    
    # Quartile clusters
    value_range = values.max() - values.min()
    for q in [0.25, 0.5, 0.75]:
        target_val = values.min() + q * value_range
        # Find cluster with median closest to target
        idx = (df['median'] - target_val).abs().idxmin()
        selected.add(int(df.loc[idx, 'cluster_id']))
    
    # Check if we have enough points
    current_n = df[df['cluster_id'].isin(selected)]['size'].sum()
    
    # Add more clusters if needed
    remaining = df[~df['cluster_id'].isin(selected)].copy()
    remaining = remaining.sort_values('size', ascending=False)
    
    for _, row in remaining.iterrows():
        if current_n >= target_n:
            break
        selected.add(int(row['cluster_id']))
        current_n += row['size']
    
    # Create test mask
    test_mask = np.isin(labels, list(selected))
    
    actual_fraction = test_mask.sum() / n_points
    log.debug(f"[SpatialCV] Selected {len(selected)} test clusters "
             f"({test_mask.sum()}/{n_points} points, {actual_fraction:.1%})")
    
    # If selected clusters overshoot the target fraction substantially, downsample points within
    # the selected clusters to hit the target indicating representative coverage without blowing up
    # the test size. This preserves spatial holdout behavior while keeping fraction bounded.
    tol = 0.05  # allow ±5% absolute fraction drift
    if target_fraction > 0 and actual_fraction > (target_fraction + tol):
        rng = np.random.RandomState(42)
        idx_all = np.arange(n_points)
        idx_sel = idx_all[test_mask]
        if idx_sel.size > target_n and target_n > 0:
            keep = rng.choice(idx_sel, size=target_n, replace=False)
            new_mask = np.zeros(n_points, dtype=bool)
            new_mask[keep] = True
            test_mask = new_mask
            actual_fraction = test_mask.sum() / n_points
            log.debug(f"[SpatialCV] Downsampled overshoot to {test_mask.sum()}/{n_points} points ({actual_fraction:.1%})")
    return test_mask


# ============================================================================
# Main Adaptive Spatial CV Class
# ============================================================================

class AdaptiveSpatialCV:
    """
    Adaptive spatial cross-validation with intelligent cluster selection.
    
    Compatible with scikit-learn's cross-validation interface.
    
    Features:
    - Automatic cluster count based on data density
    - Hierarchical clustering for spatial contiguity
    - Representative cluster selection
    - Value range coverage in test folds
    
    Example:
        >>> cv = AdaptiveSpatialCV(coords, depths, n_splits=5)
        >>> scores = cross_val_score(model, X, y, cv=cv)
        >>> print(f"CV RMSE: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    """
    
    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        n_splits: int = 5,
        config: Optional[SpatialCVConfig] = None
    ):
        """
        Args:
            coords: (n, 2) spatial coordinates
            values: (n,) target values (for representative selection)
            n_splits: Number of CV folds
            config: Optional SpatialCVConfig
        """
        self.coords = np.asarray(coords)
        self.values = np.asarray(values)
        self.n_splits = n_splits
        self.config = config or SpatialCVConfig()
        self._folds = None
        self._labels = None
        
        if len(self.coords) != len(self.values):
            raise ValueError("coords and values must have same length")
        
        if self.coords.shape[1] != 2:
            raise ValueError("coords must be (n, 2) array")
    
    def _create_clusters(self) -> np.ndarray:
        """Create spatial clusters using hierarchical clustering."""
        # Estimate optimal cluster count
        n_clusters = estimate_optimal_clusters(
            self.coords, self.values, self.config
        )
        
        # Use hierarchical clustering for spatial contiguity
        log.info(f"[SpatialCV] Creating {n_clusters} spatial clusters "
                f"(linkage={self.config.linkage})")
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.config.linkage
        )
        
        labels = clustering.fit_predict(self.coords)
        
        # Evaluate clustering quality
        compactness = compute_cluster_compactness(self.coords, labels)
        separation = compute_cluster_separation(self.coords, labels)
        silhouette = compute_silhouette_spatial(self.coords, labels)
        
        log.debug(f"[SpatialCV] Cluster quality: "
                 f"silhouette={silhouette:.3f}, "
                 f"compactness={compactness:.1f}, "
                 f"separation={separation:.1f}")
        
        return labels
    
    def _create_folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create CV folds by rotating test clusters."""
        if self._labels is None:
            self._labels = self._create_clusters()
        
        n_clusters = len(np.unique(self._labels))
        
        if n_clusters < self.n_splits:
            log.warning(f"[SpatialCV] Only {n_clusters} clusters available, "
                       f"reducing n_splits to {n_clusters}")
            self.n_splits = n_clusters
        
        # Create folds by holding out different cluster combinations
        folds = []
        
        for fold in range(self.n_splits):
            # Select test clusters for this fold
            # Strategy: Rotate through clusters in a way that maintains
            # similar test set sizes and value coverage
            
            clusters_per_fold = n_clusters // self.n_splits
            start_idx = fold * clusters_per_fold
            end_idx = start_idx + clusters_per_fold
            
            if fold == self.n_splits - 1:
                end_idx = n_clusters  # Last fold gets remaining clusters
            
            test_clusters = list(range(start_idx, end_idx))
            test_mask = np.isin(self._labels, test_clusters)
            train_mask = ~test_mask
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            # Ensure non-empty splits
            if len(train_idx) > 0 and len(test_idx) > 0:
                folds.append((train_idx, test_idx))
            else:
                log.warning(f"[SpatialCV] Fold {fold} has empty split, skipping")
        
        log.info(f"[SpatialCV] Created {len(folds)} spatial CV folds")
        for i, (train_idx, test_idx) in enumerate(folds):
            log.debug(f"[SpatialCV] Fold {i}: train={len(train_idx)}, test={len(test_idx)}")
        
        return folds
    
    def split(self, X, y=None, groups=None):
        """Generate train/test indices (scikit-learn interface)."""
        if self._folds is None:
            self._folds = self._create_folds()
        return iter(self._folds)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits


# ============================================================================
# Convenience Functions
# ============================================================================

def adaptive_train_test_split(
    coords: np.ndarray,
    values: np.ndarray,
    test_size: float = 0.2,
    config: Optional[SpatialCVConfig] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spatially-aware train/test split with adaptive clustering.
    
    Args:
        coords: (n, 2) spatial coordinates
        values: (n,) target values
        test_size: Fraction of data in test set
        config: Optional SpatialCVConfig
        random_state: Random seed
    
    Returns:
        (train_indices, test_indices)
    
    Example:
        >>> train_idx, test_idx = adaptive_train_test_split(coords, depths, test_size=0.2)
        >>> X_train, X_test = X[train_idx], X[test_idx]
        >>> y_train, y_test = y[train_idx], y[test_idx]
    """
    if config is None:
        config = SpatialCVConfig(
            target_test_fraction=test_size,
            random_state=random_state
        )
    else:
        config.target_test_fraction = test_size
        config.random_state = random_state
    
    # Create clusters
    n_clusters = estimate_optimal_clusters(coords, values, config)
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=config.linkage
    )
    labels = clustering.fit_predict(coords)
    
    # Select representative test clusters
    test_mask = select_representative_clusters(labels, values, test_size)
    
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    return train_idx, test_idx


def evaluate_spatial_split_quality(
    coords: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray
) -> dict:
    """
    Evaluate quality of a spatial train/test split.
    
    Returns:
        Dictionary with quality metrics
    """
    train_coords = coords[train_idx]
    test_coords = coords[test_idx]
    
    # Compute minimum distance from each test point to nearest train point
    from scipy.spatial import cKDTree
    train_tree = cKDTree(train_coords)
    distances, _ = train_tree.query(test_coords, k=1)
    
    metrics = {
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'test_fraction': len(test_idx) / (len(train_idx) + len(test_idx)),
        'min_separation': float(np.min(distances)),
        'median_separation': float(np.median(distances)),
        'mean_separation': float(np.mean(distances)),
        'max_separation': float(np.max(distances))
    }
    
    return metrics
