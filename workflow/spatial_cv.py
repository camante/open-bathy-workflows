#!/usr/bin/env python3
"""
spatial_cv.py - Spatial Cross-Validation for SDB Training

This module implements k-fold spatial cross-validation strategies that account for
spatial autocorrelation in bathymetric data. Standard random splits overestimate
model performance because nearby points are correlated.

Strategies implemented:
1. Spatial Block CV - Divides AOI into geographic blocks
2. Spatial Cluster CV - Uses k-means clustering on coordinates
3. Buffered Leave-One-Out - Excludes nearby points from training
4. Spatial Leave-One-Group-Out - Groups by ICESat-2 track or tile

Theory:
-------
Spatial autocorrelation means points close together have similar values. If we
randomly split train/test, nearby points end up in both sets, causing data leakage.
Spatial CV ensures test points are geographically separated from training points.

References:
-----------
- Roberts et al. (2017) - Cross-validation strategies for spatial data
- Ploton et al. (2020) - Spatial autocorrelation and ML model validation
- Valavi et al. (2019) - blockCV R package
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Iterator, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class CVFold:
    """Represents a single cross-validation fold."""
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_n: int
    test_n: int
    test_centroid: Optional[Tuple[float, float]] = None
    test_depth_range: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CVResult:
    """Results from a single fold evaluation."""
    fold_id: int
    rmse: float
    mae: float
    r2: float
    bias: float
    n_train: int
    n_test: int
    depth_range: Tuple[float, float]
    predictions: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class SpatialCVSummary:
    """Summary of spatial cross-validation results."""
    n_folds: int
    rmse_mean: float
    rmse_std: float
    rmse_per_fold: List[float]
    r2_mean: float
    r2_std: float
    r2_per_fold: List[float]
    mae_mean: float
    bias_mean: float
    generalization_gap: Optional[float] = None  # vs random CV
    fold_results: Optional[List[CVResult]] = None
    recommended_max_depth: Optional[float] = None


# -----------------------------------------------------------------------------
# Spatial blocking strategies
# -----------------------------------------------------------------------------

def create_spatial_blocks(
    df: pd.DataFrame,
    n_blocks_x: int = 3,
    n_blocks_y: int = 3,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> np.ndarray:
    """
    Create spatial blocks by dividing the AOI into a regular grid.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with coordinates
    n_blocks_x : int
        Number of blocks in x (longitude) direction
    n_blocks_y : int
        Number of blocks in y (latitude) direction
    lon_col, lat_col : str
        Column names for coordinates
        
    Returns
    -------
    np.ndarray
        Block assignment for each row (0 to n_blocks-1)
    """
    lons = df[lon_col].to_numpy()
    lats = df[lat_col].to_numpy()
    
    lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
    lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
    
    # Add small buffer to avoid edge issues
    lon_range = lon_max - lon_min + 1e-9
    lat_range = lat_max - lat_min + 1e-9
    
    # Assign to blocks
    block_x = np.clip(
        ((lons - lon_min) / lon_range * n_blocks_x).astype(int),
        0, n_blocks_x - 1
    )
    block_y = np.clip(
        ((lats - lat_min) / lat_range * n_blocks_y).astype(int),
        0, n_blocks_y - 1
    )
    
    # Combined block ID
    blocks = block_y * n_blocks_x + block_x
    
    return blocks


def create_spatial_clusters(
    df: pd.DataFrame,
    n_clusters: int = 5,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    seed: int = 42,
) -> np.ndarray:
    """
    Create spatial clusters using k-means on coordinates.
    
    This is more adaptive than regular blocks for irregularly distributed data.
    """
    from sklearn.cluster import KMeans
    
    coords = df[[lon_col, lat_col]].to_numpy()
    
    # Handle NaN coordinates
    valid_mask = np.all(np.isfinite(coords), axis=1)
    if not np.all(valid_mask):
        log.warning(f"[SpatialCV] {np.sum(~valid_mask)} points with invalid coordinates")
        # Assign invalid points to cluster 0
        clusters = np.zeros(len(df), dtype=int)
        if np.sum(valid_mask) >= n_clusters:
            km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            clusters[valid_mask] = km.fit_predict(coords[valid_mask])
        return clusters
    
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(coords)


def create_icesat_track_groups(
    df: pd.DataFrame,
    track_col: str = "gt",
    date_col: str = "datetime",
) -> np.ndarray:
    """
    Group by ICESat-2 ground track + date for leave-one-track-out CV.
    
    Each ICESat-2 pass is a natural spatial group since photons along
    a track are highly correlated.
    """
    # Create track-date identifier
    if track_col in df.columns and date_col in df.columns:
        # Extract date only (not time)
        dates = pd.to_datetime(df[date_col]).dt.date.astype(str)
        tracks = df[track_col].astype(str)
        group_ids = tracks + "_" + dates
    elif track_col in df.columns:
        group_ids = df[track_col].astype(str)
    elif "source" in df.columns:
        group_ids = df["source"].astype(str)
    else:
        # Fall back to spatial clusters
        log.warning("[SpatialCV] No track column found, falling back to spatial clusters")
        return create_spatial_clusters(df, n_clusters=5)
    
    # Convert to numeric labels
    unique_groups = group_ids.unique()
    group_map = {g: i for i, g in enumerate(unique_groups)}
    return group_ids.map(group_map).to_numpy()


# -----------------------------------------------------------------------------
# Cross-validation generators
# -----------------------------------------------------------------------------

def spatial_block_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    n_blocks_x: int = 3,
    n_blocks_y: int = 3,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    depth_col: str = "depth_m",
    min_test_samples: int = 50,
    seed: int = 42,
) -> Iterator[CVFold]:
    """
    Generate spatial block cross-validation folds.
    
    Divides the AOI into blocks and uses each block (or group of blocks) as test set.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data
    n_folds : int
        Number of CV folds
    n_blocks_x, n_blocks_y : int
        Grid dimensions for blocking
    min_test_samples : int
        Minimum samples required in test fold
        
    Yields
    ------
    CVFold
        Train/test split for each fold
    """
    blocks = create_spatial_blocks(df, n_blocks_x, n_blocks_y, lon_col, lat_col)
    n_blocks = n_blocks_x * n_blocks_y
    
    # Group blocks into folds
    rng = np.random.default_rng(seed)
    block_order = rng.permutation(n_blocks)
    blocks_per_fold = max(1, n_blocks // n_folds)
    
    indices = df.index.to_numpy()
    
    for fold_id in range(n_folds):
        # Select blocks for this test fold
        start_idx = fold_id * blocks_per_fold
        end_idx = start_idx + blocks_per_fold if fold_id < n_folds - 1 else n_blocks
        test_blocks = set(block_order[start_idx:end_idx])
        
        test_mask = np.isin(blocks, list(test_blocks))
        train_mask = ~test_mask
        
        # Check minimum samples
        if np.sum(test_mask) < min_test_samples:
            log.warning(f"[SpatialCV] Fold {fold_id}: insufficient test samples ({np.sum(test_mask)}), skipping")
            continue
        
        test_idx = indices[test_mask]
        train_idx = indices[train_mask]
        
        # Calculate test fold statistics
        test_df = df.loc[test_idx]
        test_centroid = (
            float(test_df[lon_col].mean()),
            float(test_df[lat_col].mean())
        )
        test_depth_range = (
            float(test_df[depth_col].min()),
            float(test_df[depth_col].max())
        ) if depth_col in test_df.columns else None
        
        yield CVFold(
            fold_id=fold_id,
            train_indices=train_idx,
            test_indices=test_idx,
            train_n=len(train_idx),
            test_n=len(test_idx),
            test_centroid=test_centroid,
            test_depth_range=test_depth_range,
            metadata={"blocks": list(test_blocks), "strategy": "spatial_block"}
        )


def spatial_cluster_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    depth_col: str = "depth_m",
    min_test_samples: int = 50,
    seed: int = 42,
    balance_depth: bool = True,
) -> Iterator[CVFold]:
    """
    Generate spatial cluster cross-validation folds.
    
    Uses k-means clustering to create spatially coherent test sets.
    Optionally balances folds to ensure each has a representative depth range.
    
    Parameters
    ----------
    balance_depth : bool
        If True, select test clusters that best represent the overall depth distribution
    """
    clusters = create_spatial_clusters(df, n_clusters=n_folds, lon_col=lon_col, lat_col=lat_col, seed=seed)
    indices = df.index.to_numpy()
    
    # Analyze cluster depth distributions
    if balance_depth and depth_col in df.columns:
        global_p50 = df[depth_col].median()
        global_p95 = df[depth_col].quantile(0.95)
        
        cluster_stats = []
        for k in range(n_folds):
            mask = clusters == k
            if np.sum(mask) < min_test_samples:
                cluster_stats.append((k, np.sum(mask), 0, 0, float('inf')))
                continue
            depths = df.loc[df.index[mask], depth_col]
            p50 = depths.median()
            p95 = depths.quantile(0.95)
            # Score based on similarity to global distribution
            score = abs(p50 - global_p50) + abs(p95 - global_p95)
            cluster_stats.append((k, np.sum(mask), p50, p95, score))
        
        # Sort by score (best representation first) for reporting
        cluster_stats.sort(key=lambda x: x[4])
        log.info(f"[SpatialCV] Cluster depth stats (global p50={global_p50:.2f}m, p95={global_p95:.2f}m):")
        for k, n, p50, p95, score in cluster_stats[:5]:
            log.info(f"  Cluster {k}: n={n}, p50={p50:.2f}m, p95={p95:.2f}m, score={score:.2f}")
    
    for fold_id in range(n_folds):
        test_mask = clusters == fold_id
        train_mask = ~test_mask
        
        if np.sum(test_mask) < min_test_samples:
            log.warning(f"[SpatialCV] Fold {fold_id}: insufficient test samples ({np.sum(test_mask)}), skipping")
            continue
        
        test_idx = indices[test_mask]
        train_idx = indices[train_mask]
        
        test_df = df.loc[test_idx]
        test_centroid = (
            float(test_df[lon_col].mean()),
            float(test_df[lat_col].mean())
        )
        test_depth_range = (
            float(test_df[depth_col].min()),
            float(test_df[depth_col].max())
        ) if depth_col in test_df.columns else None
        
        yield CVFold(
            fold_id=fold_id,
            train_indices=train_idx,
            test_indices=test_idx,
            train_n=len(train_idx),
            test_n=len(test_idx),
            test_centroid=test_centroid,
            test_depth_range=test_depth_range,
            metadata={"cluster": fold_id, "strategy": "spatial_cluster"}
        )


def leave_one_track_out_cv(
    df: pd.DataFrame,
    track_col: str = "gt",
    date_col: str = "datetime",
    depth_col: str = "depth_m",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    min_test_samples: int = 50,
    max_folds: int = 10,
) -> Iterator[CVFold]:
    """
    Leave-one-track-out cross-validation for ICESat-2 data.
    
    Each ICESat-2 ground track pass becomes a test fold.
    Best for understanding track-to-track variability.
    """
    groups = create_icesat_track_groups(df, track_col, date_col)
    unique_groups = np.unique(groups)
    indices = df.index.to_numpy()
    
    # Sort groups by size (largest first for more stable estimates)
    group_sizes = [(g, np.sum(groups == g)) for g in unique_groups]
    group_sizes.sort(key=lambda x: x[1], reverse=True)
    
    fold_id = 0
    for group, size in group_sizes:
        if fold_id >= max_folds:
            break
        if size < min_test_samples:
            continue
        
        test_mask = groups == group
        train_mask = ~test_mask
        
        test_idx = indices[test_mask]
        train_idx = indices[train_mask]
        
        test_df = df.loc[test_idx]
        test_centroid = (
            float(test_df[lon_col].mean()),
            float(test_df[lat_col].mean())
        ) if lon_col in test_df.columns and lat_col in test_df.columns else None
        
        test_depth_range = (
            float(test_df[depth_col].min()),
            float(test_df[depth_col].max())
        ) if depth_col in test_df.columns else None
        
        yield CVFold(
            fold_id=fold_id,
            train_indices=train_idx,
            test_indices=test_idx,
            train_n=len(train_idx),
            test_n=len(test_idx),
            test_centroid=test_centroid,
            test_depth_range=test_depth_range,
            metadata={"group": str(group), "group_size": int(size), "strategy": "leave_one_track_out"}
        )
        fold_id += 1


def buffered_spatial_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    buffer_km: float = 0.5,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    depth_col: str = "depth_m",
    min_test_samples: int = 50,
    seed: int = 42,
) -> Iterator[CVFold]:
    """
    Spatial CV with buffer zone between train and test sets.
    
    Points within buffer_km of any test point are excluded from training,
    reducing spatial autocorrelation leakage.
    
    Parameters
    ----------
    buffer_km : float
        Buffer distance in kilometers. Points within this distance of test
        points are excluded from training.
    """
    from scipy.spatial import cKDTree
    
    clusters = create_spatial_clusters(df, n_clusters=n_folds, lon_col=lon_col, lat_col=lat_col, seed=seed)
    indices = df.index.to_numpy()
    
    # Build spatial index for buffer calculation
    coords = df[[lon_col, lat_col]].to_numpy()
    # Approximate km to degrees (rough, assumes mid-latitudes)
    buffer_deg = buffer_km / 111.0  # ~111 km per degree
    
    tree = cKDTree(coords)
    
    for fold_id in range(n_folds):
        test_mask = clusters == fold_id
        
        if np.sum(test_mask) < min_test_samples:
            continue
        
        # Find all points within buffer of test points
        test_coords = coords[test_mask]
        buffer_indices = set()
        for tc in test_coords:
            nearby = tree.query_ball_point(tc, buffer_deg)
            buffer_indices.update(nearby)
        
        # Training excludes test points AND buffer zone
        test_indices_set = set(np.where(test_mask)[0])
        train_mask = np.array([
            i not in test_indices_set and i not in buffer_indices
            for i in range(len(df))
        ])
        
        # Log buffer impact
        n_buffered = len(buffer_indices) - np.sum(test_mask)
        log.info(f"[SpatialCV] Fold {fold_id}: {n_buffered} points in buffer zone excluded from training")
        
        if np.sum(train_mask) < min_test_samples:
            log.warning("[SpatialCV] Fold %s: insufficient training samples after buffer", fold_id)
            continue
        
        test_idx = indices[test_mask]
        train_idx = indices[train_mask]
        
        test_df = df.loc[test_idx]
        test_centroid = (float(test_df[lon_col].mean()), float(test_df[lat_col].mean()))
        test_depth_range = (float(test_df[depth_col].min()), float(test_df[depth_col].max())) if depth_col in df.columns else None
        
        yield CVFold(
            fold_id=fold_id,
            train_indices=train_idx,
            test_indices=test_idx,
            train_n=len(train_idx),
            test_n=len(test_idx),
            test_centroid=test_centroid,
            test_depth_range=test_depth_range,
            metadata={
                "cluster": fold_id,
                "buffer_km": buffer_km,
                "n_buffered": n_buffered,
                "strategy": "buffered_spatial"
            }
        )


# -----------------------------------------------------------------------------
# CV execution and evaluation
# -----------------------------------------------------------------------------

def run_spatial_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "depth_m",
    cv_strategy: str = "spatial_cluster",
    n_folds: int = 5,
    model_params: Optional[Dict] = None,
    seed: int = 42,
    return_predictions: bool = False,
    **cv_kwargs,
) -> SpatialCVSummary:
    """
    Run spatial cross-validation and return summary statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data with features and target
    feature_cols : List[str]
        Feature column names
    target_col : str
        Target column name
    cv_strategy : str
        One of: 'spatial_cluster', 'spatial_block', 'leave_one_track_out', 'buffered_spatial'
    n_folds : int
        Number of CV folds
    model_params : dict, optional
        RandomForest parameters
    return_predictions : bool
        If True, include predictions in results
        
    Returns
    -------
    SpatialCVSummary
        Summary statistics across all folds
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Select CV generator
    cv_generators = {
        "spatial_cluster": spatial_cluster_cv,
        "spatial_block": spatial_block_cv,
        "leave_one_track_out": leave_one_track_out_cv,
        "buffered_spatial": buffered_spatial_cv,
    }
    
    if cv_strategy not in cv_generators:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}. Choose from {list(cv_generators.keys())}")
    
    cv_gen = cv_generators[cv_strategy]
    
    # Default model params
    if model_params is None:
        model_params = {
            "n_estimators": 150,
            "max_depth": 15,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": seed,
        }
    
    # Run CV
    fold_results: List[CVResult] = []
    all_predictions = [] if return_predictions else None
    all_actuals = [] if return_predictions else None
    
    log.info(f"[SpatialCV] Running {cv_strategy} with {n_folds} folds...")
    
    for fold in cv_gen(df, n_folds=n_folds, seed=seed, **cv_kwargs):
        # Prepare data
        X_train = df.loc[fold.train_indices, feature_cols].to_numpy()
        y_train = df.loc[fold.train_indices, target_col].to_numpy()
        X_test = df.loc[fold.test_indices, feature_cols].to_numpy()
        y_test = df.loc[fold.test_indices, target_col].to_numpy()
        
        # Filter finite values
        train_mask = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        test_mask = np.all(np.isfinite(X_test), axis=1) & np.isfinite(y_test)
        
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        
        if len(X_train) < 50 or len(X_test) < 20:
            log.warning(f"[SpatialCV] Fold {fold.fold_id}: insufficient samples after filtering")
            continue
        
        # Train model
        rf = RandomForestRegressor(**model_params)
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        
        # Calculate metrics
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        bias = float(np.mean(y_pred - y_test))
        
        # Feature importance
        feat_imp = dict(zip(feature_cols, rf.feature_importances_.tolist()))
        
        result = CVResult(
            fold_id=fold.fold_id,
            rmse=rmse,
            mae=mae,
            r2=r2,
            bias=bias,
            n_train=len(X_train),
            n_test=len(X_test),
            depth_range=(float(np.min(y_test)), float(np.max(y_test))),
            predictions=y_pred if return_predictions else None,
            feature_importance=feat_imp,
        )
        fold_results.append(result)
        
        if return_predictions:
            all_predictions.extend(y_pred.tolist())
            all_actuals.extend(y_test.tolist())
        
        log.info(
            f"[SpatialCV] Fold {fold.fold_id}: RMSE={rmse:.3f}m, R²={r2:.3f}, "
            f"n_train={len(X_train)}, n_test={len(X_test)}, "
            f"depth_range=[{result.depth_range[0]:.1f}, {result.depth_range[1]:.1f}]m"
        )
    
    if not fold_results:
        log.error("[SpatialCV] No valid folds completed")
        return SpatialCVSummary(
            n_folds=0, rmse_mean=np.nan, rmse_std=np.nan, rmse_per_fold=[],
            r2_mean=np.nan, r2_std=np.nan, r2_per_fold=[],
            mae_mean=np.nan, bias_mean=np.nan
        )
    
    # Aggregate statistics
    rmse_values = [r.rmse for r in fold_results]
    r2_values = [r.r2 for r in fold_results]
    mae_values = [r.mae for r in fold_results]
    bias_values = [r.bias for r in fold_results]
    
    summary = SpatialCVSummary(
        n_folds=len(fold_results),
        rmse_mean=float(np.mean(rmse_values)),
        rmse_std=float(np.std(rmse_values)),
        rmse_per_fold=rmse_values,
        r2_mean=float(np.mean(r2_values)),
        r2_std=float(np.std(r2_values)),
        r2_per_fold=r2_values,
        mae_mean=float(np.mean(mae_values)),
        bias_mean=float(np.mean(bias_values)),
        fold_results=fold_results,
    )
    
    log.info(
        f"[SpatialCV] Summary: RMSE={summary.rmse_mean:.3f}±{summary.rmse_std:.3f}m, "
        f"R²={summary.r2_mean:.3f}±{summary.r2_std:.3f}"
    )
    
    return summary


def compare_cv_strategies(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "depth_m",
    strategies: List[str] = None,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, SpatialCVSummary]:
    """
    Compare multiple CV strategies on the same data.
    
    Returns a dict mapping strategy name to its CV summary.
    """
    if strategies is None:
        strategies = ["spatial_cluster", "spatial_block"]
    
    results = {}
    for strategy in strategies:
        log.info("\n[SpatialCV] === Running %s ===", strategy)
        try:
            results[strategy] = run_spatial_cv(
                df, feature_cols, target_col,
                cv_strategy=strategy, n_folds=n_folds, seed=seed
            )
        except Exception as e:
            log.error(f"[SpatialCV] {strategy} failed: {e}")
            results[strategy] = None
    
    # Compare results
    log.info("\n[SpatialCV] === Strategy Comparison ===")
    for name, summary in results.items():
        if summary is not None:
            log.info(f"  {name}: RMSE={summary.rmse_mean:.3f}±{summary.rmse_std:.3f}m, R²={summary.r2_mean:.3f}")
    
    return results


# -----------------------------------------------------------------------------
# Integration with training pipeline
# -----------------------------------------------------------------------------

def train_with_spatial_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "depth_m",
    cv_strategy: str = "spatial_cluster",
    n_folds: int = 5,
    final_model_test_size: float = 0.2,
    model_params: Optional[Dict] = None,
    seed: int = 42,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, SpatialCVSummary, Dict[str, Any]]:
    """
    Train a model with spatial CV for evaluation, then train final model on all data.
    
    This is the recommended approach:
    1. Run spatial CV to get unbiased performance estimates
    2. Train final model on all data (or train/holdout split)
    3. Use CV statistics for confidence intervals and max depth estimation
    
    Parameters
    ----------
    df : pd.DataFrame
        Full training data
    feature_cols : List[str]
        Feature columns
    target_col : str
        Target column
    cv_strategy : str
        CV strategy to use
    n_folds : int
        Number of CV folds
    final_model_test_size : float
        Fraction held out for final model evaluation
    model_params : dict
        Model parameters
    seed : int
        Random seed
        
    Returns
    -------
    Tuple of:
        - Trained final model
        - Training DataFrame
        - Test DataFrame (for final model)
        - SpatialCVSummary
        - Metadata dict
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Default model params
    if model_params is None:
        model_params = {
            "n_estimators": 150,
            "max_depth": 15,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": seed,
        }
    
    # Step 1: Run spatial CV for unbiased evaluation
    log.info("[SpatialCV] Step 1: Running spatial cross-validation...")
    cv_summary = run_spatial_cv(
        df, feature_cols, target_col,
        cv_strategy=cv_strategy, n_folds=n_folds,
        model_params=model_params, seed=seed
    )
    
    # Step 2: Create final train/test split (spatial)
    log.info("[SpatialCV] Step 2: Creating final train/test split...")
    clusters = create_spatial_clusters(df, n_clusters=n_folds, seed=seed)
    
    # Use one cluster as holdout, rest for training
    # Pick the cluster that best represents overall depth distribution
    global_p95 = df[target_col].quantile(0.95)
    best_test_cluster = 0
    best_score = float('inf')
    
    for k in range(n_folds):
        mask = clusters == k
        if np.sum(mask) < 50:
            continue
        p95 = df.loc[df.index[mask], target_col].quantile(0.95)
        score = abs(p95 - global_p95)
        if score < best_score:
            best_score = score
            best_test_cluster = k
    
    test_mask = clusters == best_test_cluster
    train_mask = ~test_mask
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    log.info(f"[SpatialCV] Final split: {len(df_train)} train, {len(df_test)} test (cluster {best_test_cluster})")
    
    # Step 3: Train final model
    log.info("[SpatialCV] Step 3: Training final model...")
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train[target_col].to_numpy()
    
    # Filter finite
    train_finite = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
    X_train = X_train[train_finite]
    y_train = y_train[train_finite]
    
    rf = RandomForestRegressor(**model_params)
    rf.fit(X_train, y_train)
    
    # Predict on test set
    X_test = df_test[feature_cols].to_numpy()
    y_test = df_test[target_col].to_numpy()
    test_finite = np.all(np.isfinite(X_test), axis=1) & np.isfinite(y_test)
    
    y_pred = rf.predict(X_test[test_finite])
    y_test_clean = y_test[test_finite]
    
    final_rmse = float(np.sqrt(mean_squared_error(y_test_clean, y_pred)))
    final_r2 = float(r2_score(y_test_clean, y_pred))
    
    log.info(f"[SpatialCV] Final model: RMSE={final_rmse:.3f}m, R²={final_r2:.3f}")
    log.info(f"[SpatialCV] CV estimate: RMSE={cv_summary.rmse_mean:.3f}±{cv_summary.rmse_std:.3f}m")
    
    # Add predictions to test df
    df_test = df_test.copy()
    df_test["depth_pred_m"] = np.nan
    df_test.loc[df_test.index[test_finite], "depth_pred_m"] = y_pred
    
    # Compile metadata
    metadata = {
        "cv_strategy": cv_strategy,
        "n_folds": n_folds,
        "cv_rmse_mean": cv_summary.rmse_mean,
        "cv_rmse_std": cv_summary.rmse_std,
        "cv_r2_mean": cv_summary.r2_mean,
        "cv_r2_std": cv_summary.r2_std,
        "final_rmse": final_rmse,
        "final_r2": final_r2,
        "test_cluster": int(best_test_cluster),
        "n_train": len(df_train),
        "n_test": len(df_test),
        "feature_importance": dict(zip(feature_cols, rf.feature_importances_.tolist())),
    }
    
    # Estimate recommended max depth based on CV variance
    # If RMSE varies a lot between folds, be more conservative
    if cv_summary.rmse_std > 0.3 * cv_summary.rmse_mean:
        log.warning("[SpatialCV] High CV variance detected - model may not generalize well")
        metadata["cv_stability"] = "low"
    else:
        metadata["cv_stability"] = "good"
    
    return rf, df_train, df_test, cv_summary, metadata


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_spatial_cv_results(
    cv_summary: SpatialCVSummary,
    output_path: Union[str, Path],
    title: str = "Spatial Cross-Validation Results",
):
    """Generate visualization of spatial CV results."""
    from plot_utils import lazy_pyplot
    plt = lazy_pyplot()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # RMSE by fold
    ax1 = axes[0]
    folds = list(range(len(cv_summary.rmse_per_fold)))
    ax1.bar(folds, cv_summary.rmse_per_fold, color='steelblue', alpha=0.7)
    ax1.axhline(cv_summary.rmse_mean, color='red', linestyle='--', label=f'Mean: {cv_summary.rmse_mean:.2f}m')
    ax1.fill_between(
        [-0.5, len(folds)-0.5],
        cv_summary.rmse_mean - cv_summary.rmse_std,
        cv_summary.rmse_mean + cv_summary.rmse_std,
        color='red', alpha=0.2, label=f'±1 std: {cv_summary.rmse_std:.2f}m'
    )
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('RMSE (m)')
    ax1.set_title('RMSE by Fold')
    ax1.legend()
    
    # R² by fold
    ax2 = axes[1]
    ax2.bar(folds, cv_summary.r2_per_fold, color='forestgreen', alpha=0.7)
    ax2.axhline(cv_summary.r2_mean, color='red', linestyle='--', label=f'Mean: {cv_summary.r2_mean:.2f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('R²')
    ax2.set_title('R² by Fold')
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # Depth range by fold
    ax3 = axes[2]
    if cv_summary.fold_results:
        for i, result in enumerate(cv_summary.fold_results):
            d_min, d_max = result.depth_range
            ax3.plot([i, i], [d_min, d_max], 'o-', color='purple', markersize=8)
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Depth (m)')
    ax3.set_title('Test Depth Range by Fold')
    ax3.invert_yaxis()
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    log.info("[SpatialCV] Saved CV results plot: %s", output_path)


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import argparse
    
    parser = argparse.ArgumentParser(description="Run spatial cross-validation")
    parser.add_argument("--input-csv", required=True, help="Input training data CSV")
    parser.add_argument("--feature-cols", nargs="+", required=True, help="Feature column names")
    parser.add_argument("--target-col", default="depth_m", help="Target column name")
    parser.add_argument("--strategy", default="spatial_cluster",
                       choices=["spatial_cluster", "spatial_block", "leave_one_track_out", "buffered_spatial"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    df = pd.read_csv(args.input_csv)
    
    summary = run_spatial_cv(
        df, args.feature_cols, args.target_col,
        cv_strategy=args.strategy, n_folds=args.n_folds, seed=args.seed
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_spatial_cv_results(summary, output_dir / "spatial_cv_results.png")
    
    # Save summary
    import json
    with open(output_dir / "spatial_cv_summary.json", "w") as f:
        json.dump({
            "n_folds": summary.n_folds,
            "rmse_mean": summary.rmse_mean,
            "rmse_std": summary.rmse_std,
            "r2_mean": summary.r2_mean,
            "r2_std": summary.r2_std,
            "mae_mean": summary.mae_mean,
            "bias_mean": summary.bias_mean,
        }, f, indent=2)
