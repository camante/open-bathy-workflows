#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hyperparameter_tuning.py - Automated hyperparameter optimization for SDB models

Provides automated hyperparameter tuning for Random Forest and optional
ensemble models using RandomizedSearchCV with spatial cross-validation support.

Features:
- RF hyperparameter search with sensible ranges
- Gradient Boosting comparison
- Ensemble stacking
- Spatial CV aware
- Caching of results

Usage:
    from hyperparameter_tuning import tune_rf_model, tune_ensemble
    
    best_model, results = tune_rf_model(
        X_train, y_train,
        spatial_cv=True,
        coords=coords_train,
        n_iter=50
    )
"""

import logging
import json
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint

log = logging.getLogger(__name__)


# ============================================================================
# Parameter Distributions
# ============================================================================

def get_rf_param_dist(search_type: str = "full") -> dict:
    """
    Get parameter distributions for Random Forest.
    
    Args:
        search_type: "quick" (20 combos), "full" (50 combos), "extensive" (100 combos)
    
    Returns:
        Parameter distribution dictionary for RandomizedSearchCV
    """
    if search_type == "quick":
        return {
            'n_estimators': randint(50, 150),
            'max_depth': [10, 20, None],
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2'],
        }
    
    elif search_type == "extensive":
        return {
            'n_estimators': randint(50, 300),
            'max_depth': [5, 10, 15, 20, 25, 30, None],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
            'min_impurity_decrease': uniform(0.0, 0.01),
        }
    
    else:  # full (default)
        return {
            'n_estimators': randint(50, 200),
            'max_depth': [10, 20, 30, None],
            'min_samples_split': randint(2, 15),
            'min_samples_leaf': randint(1, 8),
            'max_features': ['sqrt', 'log2', 0.5],
        }


def get_gb_param_dist(search_type: str = "full") -> dict:
    """Get parameter distributions for Gradient Boosting."""
    if search_type == "quick":
        return {
            'n_estimators': randint(50, 150),
            'learning_rate': uniform(0.05, 0.15),
            'max_depth': randint(3, 7),
            'subsample': uniform(0.7, 0.3),
        }
    
    else:  # full
        return {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 10),
            'subsample': uniform(0.6, 0.4),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5),
        }


# ============================================================================
# Spatial Cross-Validation Support
# ============================================================================

def create_spatial_cv(coords: np.ndarray, n_splits: int = 5, random_state: int = 42):
    """
    Create spatial cross-validation folds using k-means clustering.
    
    Args:
        coords: (n, 2) array of coordinates
        n_splits: Number of CV folds
        random_state: Random seed
    
    Returns:
        List of (train_idx, test_idx) tuples
    """
    from sklearn.cluster import KMeans
    
    # Cluster points spatially
    kmeans = KMeans(n_clusters=n_splits, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(coords)
    
    # Create folds by holding out each cluster
    folds = []
    for i in range(n_splits):
        test_idx = np.where(clusters == i)[0]
        train_idx = np.where(clusters != i)[0]
        folds.append((train_idx, test_idx))
    
    return folds


class SpatialCV:
    """
    Spatial cross-validation splitter compatible with scikit-learn CV.
    """
    
    def __init__(self, coords: np.ndarray, n_splits: int = 5, random_state: int = 42):
        self.coords = coords
        self.n_splits = n_splits
        self.random_state = random_state
        self._folds = None
    
    def split(self, X, y=None, groups=None):
        """Generate train/test indices."""
        if self._folds is None:
            self._folds = create_spatial_cv(
                self.coords, 
                self.n_splits, 
                self.random_state
            )
        return iter(self._folds)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ============================================================================
# Tuning Functions
# ============================================================================

@dataclass
class TuningResults:
    """Results from hyperparameter tuning."""
    best_params: dict
    best_score: float
    cv_results: pd.DataFrame
    search_type: str
    n_iter: int
    spatial_cv: bool
    time_seconds: float


def tune_rf_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    spatial_cv: bool = False,
    coords: Optional[np.ndarray] = None,
    n_iter: int = 50,
    n_cv_folds: int = 5,
    search_type: str = "full",
    random_state: int = 42,
    n_jobs: int = -1,
    cache_path: Optional[Path] = None
) -> Tuple[RandomForestRegressor, TuningResults]:
    """
    Tune Random Forest hyperparameters.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        spatial_cv: Use spatial cross-validation
        coords: Coordinates for spatial CV (n_samples, 2)
        n_iter: Number of random parameter combinations
        n_cv_folds: Number of CV folds
        search_type: "quick", "full", or "extensive"
        random_state: Random seed
        n_jobs: Number of parallel jobs (-1 = all cores)
        cache_path: Optional path to cache results
    
    Returns:
        (best_model, tuning_results)
    """
    import time
    start_time = time.time()
    
    log.info(f"[TUNE] Starting RF hyperparameter search ({search_type}, {n_iter} iterations)")
    
    # Check cache
    if cache_path and cache_path.exists():
        log.info(f"[TUNE] Loading cached results from {cache_path}")
        cached = joblib.load(cache_path)
        return cached['model'], cached['results']
    
    # Setup CV strategy
    if spatial_cv:
        if coords is None:
            raise ValueError("coords required for spatial_cv=True")
        cv = SpatialCV(coords, n_splits=n_cv_folds, random_state=random_state)
        log.info(f"[TUNE] Using spatial CV with {n_cv_folds} folds")
    else:
        cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
        log.info(f"[TUNE] Using standard KFold CV with {n_cv_folds} folds")
    
    # Base model
    rf = RandomForestRegressor(
        random_state=random_state,
        n_jobs=1,  # Parallel at search level, not tree level
        warm_start=False
    )
    
    # Parameter distribution
    param_dist = get_rf_param_dist(search_type)
    
    # Randomized search
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        return_train_score=True
    )
    
    log.info(f"[TUNE] Fitting {n_iter} parameter combinations...")
    search.fit(X_train, y_train)
    
    # Extract results
    results = TuningResults(
        best_params=search.best_params_,
        best_score=-search.best_score_,  # Convert back to positive RMSE
        cv_results=pd.DataFrame(search.cv_results_),
        search_type=search_type,
        n_iter=n_iter,
        spatial_cv=spatial_cv,
        time_seconds=time.time() - start_time
    )
    
    # Best model
    best_model = search.best_estimator_
    best_model.n_jobs = -1  # Use all cores for prediction
    
    log.info(f"[TUNE] Best RMSE: {results.best_score:.4f}")
    log.info(f"[TUNE] Best params: {results.best_params}")
    log.info(f"[TUNE] Tuning completed in {results.time_seconds:.1f}s")
    
    # Cache results
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': best_model, 'results': results}, cache_path)
        log.info(f"[TUNE] Cached results to {cache_path}")
    
    return best_model, results


def tune_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    spatial_cv: bool = False,
    coords: Optional[np.ndarray] = None,
    n_iter: int = 30,
    n_cv_folds: int = 5,
    search_type: str = "full",
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[GradientBoostingRegressor, TuningResults]:
    """
    Tune Gradient Boosting hyperparameters.
    
    Args:
        Similar to tune_rf_model
    
    Returns:
        (best_model, tuning_results)
    """
    import time
    start_time = time.time()
    
    log.info(f"[TUNE] Starting GB hyperparameter search ({search_type}, {n_iter} iterations)")
    
    # Setup CV
    if spatial_cv:
        if coords is None:
            raise ValueError("coords required for spatial_cv=True")
        cv = SpatialCV(coords, n_splits=n_cv_folds, random_state=random_state)
    else:
        cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    
    # Base model
    gb = GradientBoostingRegressor(random_state=random_state)
    
    # Parameter distribution
    param_dist = get_gb_param_dist(search_type)
    
    # Search
    search = RandomizedSearchCV(
        gb,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        return_train_score=True
    )
    
    search.fit(X_train, y_train)
    
    results = TuningResults(
        best_params=search.best_params_,
        best_score=-search.best_score_,
        cv_results=pd.DataFrame(search.cv_results_),
        search_type=search_type,
        n_iter=n_iter,
        spatial_cv=spatial_cv,
        time_seconds=time.time() - start_time
    )
    
    log.info(f"[TUNE] GB Best RMSE: {results.best_score:.4f}")
    log.info(f"[TUNE] GB completed in {results.time_seconds:.1f}s")
    
    return search.best_estimator_, results


def tune_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    spatial_cv: bool = False,
    coords: Optional[np.ndarray] = None,
    n_iter_rf: int = 30,
    n_iter_gb: int = 20,
    n_cv_folds: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[StackingRegressor, Dict[str, TuningResults]]:
    """
    Create and tune an ensemble stacking model.
    
    Trains both RF and GB with hyperparameter tuning, then stacks them
    with a Ridge regression meta-learner.
    
    Args:
        X_train: Training features
        y_train: Training targets
        spatial_cv: Use spatial CV
        coords: Coordinates for spatial CV
        n_iter_rf: Iterations for RF tuning
        n_iter_gb: Iterations for GB tuning
        n_cv_folds: CV folds
        random_state: Random seed
        n_jobs: Parallel jobs
    
    Returns:
        (stacked_model, {rf_results, gb_results})
    """
    log.info("[TUNE] Training ensemble stacking model")
    
    # Tune RF
    rf_model, rf_results = tune_rf_model(
        X_train, y_train,
        spatial_cv=spatial_cv,
        coords=coords,
        n_iter=n_iter_rf,
        n_cv_folds=n_cv_folds,
        search_type="full",
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    # Tune GB
    gb_model, gb_results = tune_gradient_boosting(
        X_train, y_train,
        spatial_cv=spatial_cv,
        coords=coords,
        n_iter=n_iter_gb,
        n_cv_folds=n_cv_folds,
        search_type="full",
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    # Create stacking ensemble
    log.info("[TUNE] Creating stacking ensemble")
    
    if spatial_cv and coords is not None:
        cv = SpatialCV(coords, n_splits=n_cv_folds, random_state=random_state)
    else:
        cv = n_cv_folds
    
    stacked = StackingRegressor(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=cv,
        n_jobs=n_jobs
    )
    
    log.info("[TUNE] Fitting stacked ensemble...")
    stacked.fit(X_train, y_train)
    
    log.info("[TUNE] Ensemble training complete")
    
    return stacked, {'rf': rf_results, 'gb': gb_results}


# ============================================================================
# Comparison & Reporting
# ============================================================================

def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models on test set.
    
    Args:
        models: Dict of {name: model}
        X_test: Test features
        y_test: Test targets
    
    Returns:
        DataFrame with metrics for each model
    """
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'model': name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
    
    df = pd.DataFrame(results).sort_values('rmse')
    return df


def save_tuning_report(
    results: TuningResults,
    output_path: Path,
    model_name: str = "RF"
):
    """Save tuning results to JSON."""
    report = {
        'model': model_name,
        'best_params': results.best_params,
        'best_cv_rmse': float(results.best_score),
        'search_type': results.search_type,
        'n_iterations': results.n_iter,
        'spatial_cv': results.spatial_cv,
        'tuning_time_seconds': results.time_seconds,
        'top_10_params': results.cv_results
            .nsmallest(10, 'rank_test_score')[
                ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
            ].to_dict('records')
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    log.info(f"[TUNE] Saved tuning report to {output_path}")
