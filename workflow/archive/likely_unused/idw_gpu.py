#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
idw_gpu.py - GPU-accelerated Inverse Distance Weighting interpolation

Provides GPU-accelerated IDW interpolation using CuPy with automatic
fallback to CPU (scipy/sklearn/numpy) when GPU is unavailable.

Performance improvements:
- CPU (numpy): Baseline
- CPU (scipy KDTree): 5-10x faster
- GPU (CuPy): 10-50x faster for large grids

Requirements:
    GPU mode: cupy-cuda11x or cupy-cuda12x
    CPU mode: scipy (recommended) or sklearn

Usage:
    from idw_gpu import idw_interpolate
    
    result = idw_interpolate(
        pts_xy, pts_val, q_xy,
        k=12, power=2.0,
        use_gpu=True  # Auto-detects GPU
    )
"""

import logging
from typing import Tuple, Optional
import numpy as np

log = logging.getLogger(__name__)

# ============================================================================
# GPU Detection and Imports
# ============================================================================

GPU_AVAILABLE = False
GPU_ERROR = None

try:
    import cupy as cp
    from cupyx.scipy.spatial import cKDTree as cuKDTree
    GPU_AVAILABLE = True
    log.info("[IDW] GPU acceleration available (CuPy)")
except ImportError as e:
    GPU_ERROR = f"CuPy not installed: {e}"
except Exception as e:
    GPU_ERROR = f"GPU initialization failed: {e}"

if GPU_ERROR:
    log.debug(f"[IDW] {GPU_ERROR}")


# CPU KDTree backend selection
CPU_BACKEND = None
try:
    from scipy.spatial import cKDTree
    CPU_BACKEND = "scipy"
except ImportError:
    try:
        from sklearn.neighbors import KDTree
        CPU_BACKEND = "sklearn"
    except ImportError:
        CPU_BACKEND = "numpy"
        log.warning("[IDW] Neither scipy nor sklearn available. Using slow numpy fallback.")


# ============================================================================
# GPU Implementation
# ============================================================================

def _idw_gpu(
    pts_xy: np.ndarray,
    pts_val: np.ndarray,
    q_xy: np.ndarray,
    k: int = 12,
    power: float = 2.0,
    eps: float = 1e-6
) -> np.ndarray:
    """
    GPU-accelerated IDW using CuPy.
    
    Args:
        pts_xy: (n, 2) data point coordinates
        pts_val: (n,) data point values
        q_xy: (m, 2) query point coordinates
        k: Number of nearest neighbors
        power: IDW power parameter
        eps: Small value to prevent division by zero
    
    Returns:
        (m,) interpolated values at query points
    """
    # Transfer to GPU
    pts_xy_gpu = cp.asarray(pts_xy, dtype=cp.float32)
    pts_val_gpu = cp.asarray(pts_val, dtype=cp.float32)
    q_xy_gpu = cp.asarray(q_xy, dtype=cp.float32)
    
    # Build KD-tree on GPU
    tree = cuKDTree(pts_xy_gpu)
    
    # Query k-nearest neighbors
    k_eff = min(k, len(pts_xy))
    distances, indices = tree.query(q_xy_gpu, k=k_eff)
    
    # Handle single neighbor case
    if distances.ndim == 1:
        distances = distances[:, cp.newaxis]
        indices = indices[:, cp.newaxis]
    
    # Compute IDW weights
    weights = 1.0 / cp.power(distances + eps, power)
    
    # Get values at nearest neighbors
    values = pts_val_gpu[indices]
    
    # Weighted average
    numerator = cp.sum(weights * values, axis=1)
    denominator = cp.sum(weights, axis=1)
    result = numerator / cp.maximum(denominator, eps)
    
    # Handle exact hits (distance == 0)
    exact_hits = distances[:, 0] <= eps
    if cp.any(exact_hits):
        result[exact_hits] = pts_val_gpu[indices[exact_hits, 0]]
    
    # Transfer back to CPU
    return cp.asnumpy(result)


# ============================================================================
# CPU Implementations
# ============================================================================

def _idw_scipy(
    pts_xy: np.ndarray,
    pts_val: np.ndarray,
    q_xy: np.ndarray,
    k: int = 12,
    power: float = 2.0,
    eps: float = 1e-6
) -> np.ndarray:
    """CPU IDW using scipy KDTree (fast)."""
    tree = cKDTree(pts_xy)
    k_eff = min(k, len(pts_xy))
    distances, indices = tree.query(q_xy, k=k_eff)
    
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]
    
    weights = 1.0 / np.power(distances + eps, power)
    values = pts_val[indices]
    
    numerator = np.sum(weights * values, axis=1)
    denominator = np.sum(weights, axis=1)
    result = numerator / np.maximum(denominator, eps)
    
    # Exact hits
    exact_hits = distances[:, 0] <= eps
    if np.any(exact_hits):
        result[exact_hits] = pts_val[indices[exact_hits, 0]]
    
    return result


def _idw_sklearn(
    pts_xy: np.ndarray,
    pts_val: np.ndarray,
    q_xy: np.ndarray,
    k: int = 12,
    power: float = 2.0,
    eps: float = 1e-6
) -> np.ndarray:
    """CPU IDW using sklearn KDTree."""
    tree = KDTree(pts_xy)
    k_eff = min(k, len(pts_xy))
    distances, indices = tree.query(q_xy, k=k_eff, return_distance=True)
    
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]
    
    weights = 1.0 / np.power(distances + eps, power)
    values = pts_val[indices]
    
    numerator = np.sum(weights * values, axis=1)
    denominator = np.sum(weights, axis=1)
    result = numerator / np.maximum(denominator, eps)
    
    # Exact hits
    exact_hits = distances[:, 0] <= eps
    if np.any(exact_hits):
        result[exact_hits] = pts_val[indices[exact_hits, 0]]
    
    return result


def _idw_numpy_chunked(
    pts_xy: np.ndarray,
    pts_val: np.ndarray,
    q_xy: np.ndarray,
    k: int = 12,
    power: float = 2.0,
    eps: float = 1e-6,
    chunk_size: int = 10000
) -> np.ndarray:
    """
    Fallback numpy implementation with chunking (slow but no dependencies).
    """
    k_eff = min(k, len(pts_xy))
    result = np.empty(len(q_xy), dtype=np.float64)
    
    for i in range(0, len(q_xy), chunk_size):
        chunk_end = min(i + chunk_size, len(q_xy))
        q_chunk = q_xy[i:chunk_end]
        
        # Compute all distances
        dx = q_chunk[:, np.newaxis, 0] - pts_xy[np.newaxis, :, 0]
        dy = q_chunk[:, np.newaxis, 1] - pts_xy[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        
        # Find k nearest
        idx_k = np.argpartition(dist_sq, kth=k_eff-1, axis=1)[:, :k_eff]
        
        # Sort within k nearest
        rows = np.arange(idx_k.shape[0])[:, np.newaxis]
        dist_sq_k = dist_sq[rows, idx_k]
        sort_idx = np.argsort(dist_sq_k, axis=1)
        indices = idx_k[rows, sort_idx]
        distances = np.sqrt(dist_sq[rows, indices])
        
        # IDW weights
        weights = 1.0 / np.power(distances + eps, power)
        values = pts_val[indices]
        
        numerator = np.sum(weights * values, axis=1)
        denominator = np.sum(weights, axis=1)
        chunk_result = numerator / np.maximum(denominator, eps)
        
        # Exact hits
        exact_hits = distances[:, 0] <= eps
        if np.any(exact_hits):
            chunk_result[exact_hits] = pts_val[indices[exact_hits, 0]]
        
        result[i:chunk_end] = chunk_result
    
    return result


# ============================================================================
# Adaptive IDW (AIDW)
# ============================================================================

def _adaptive_idw(
    pts_xy: np.ndarray,
    pts_val: np.ndarray,
    q_xy: np.ndarray,
    k: int = 12,
    base_power: float = 2.0,
    power_range: Tuple[float, float] = (1.5, 4.0),
    eps: float = 1e-6,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Adaptive IDW with varying power parameter based on point density.
    
    In sparse areas (large distance spread), use higher power (more local).
    In dense areas (small distance spread), use lower power (smoother).
    
    Args:
        power_range: (min_power, max_power) based on density
    """
    # Get distances first
    if use_gpu and GPU_AVAILABLE:
        pts_xy_gpu = cp.asarray(pts_xy, dtype=cp.float32)
        q_xy_gpu = cp.asarray(q_xy, dtype=cp.float32)
        tree = cuKDTree(pts_xy_gpu)
        k_eff = min(k, len(pts_xy))
        distances, indices = tree.query(q_xy_gpu, k=k_eff)
        distances = cp.asnumpy(distances)
        indices = cp.asnumpy(indices)
    elif CPU_BACKEND == "scipy":
        tree = cKDTree(pts_xy)
        k_eff = min(k, len(pts_xy))
        distances, indices = tree.query(q_xy, k=k_eff)
    else:
        raise NotImplementedError("AIDW requires scipy or GPU")
    
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]
    
    # Adaptive power based on distance spread
    d_nearest = distances[:, 0]
    d_farthest = distances[:, -1]
    ratio = np.clip(d_farthest / np.maximum(d_nearest, eps), 1.0, 10.0)
    
    # Map ratio to power range using log scale
    power_min, power_max = power_range
    adaptive_power = power_min + (np.log(ratio) / np.log(10.0)) * (power_max - power_min)
    adaptive_power = adaptive_power[:, np.newaxis]
    
    # Compute weights with adaptive power
    weights = 1.0 / np.power(distances + eps, adaptive_power)
    values = pts_val[indices]
    
    numerator = np.sum(weights * values, axis=1)
    denominator = np.sum(weights, axis=1)
    result = numerator / np.maximum(denominator, eps)
    
    # Exact hits
    exact_hits = distances[:, 0] <= eps
    if np.any(exact_hits):
        result[exact_hits] = pts_val[indices[exact_hits, 0]]
    
    return result


# ============================================================================
# Main Interface
# ============================================================================

def idw_interpolate(
    pts_xy: np.ndarray,
    pts_val: np.ndarray,
    q_xy: np.ndarray,
    k: int = 12,
    power: float = 2.0,
    adaptive: bool = False,
    use_gpu: bool = True,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Inverse Distance Weighting interpolation with automatic backend selection.
    
    Automatically selects the fastest available backend:
    1. GPU (CuPy) if available and use_gpu=True
    2. CPU (scipy KDTree) if available
    3. CPU (sklearn KDTree) if available
    4. CPU (numpy) as fallback
    
    Args:
        pts_xy: (n, 2) data point coordinates
        pts_val: (n,) data point values
        q_xy: (m, 2) query point coordinates
        k: Number of nearest neighbors (default: 12)
        power: IDW power parameter (default: 2.0)
        adaptive: Use adaptive power based on density (default: False)
        use_gpu: Try to use GPU if available (default: True)
        eps: Small value to prevent division by zero
    
    Returns:
        (m,) interpolated values at query points
    
    Example:
        >>> pts_xy = np.random.rand(1000, 2) * 100
        >>> pts_val = np.sin(pts_xy[:, 0]) * np.cos(pts_xy[:, 1])
        >>> q_xy = np.random.rand(10000, 2) * 100
        >>> result = idw_interpolate(pts_xy, pts_val, q_xy, k=12, power=2.0)
    """
    # Validate inputs
    pts_xy = np.asarray(pts_xy, dtype=np.float64)
    pts_val = np.asarray(pts_val, dtype=np.float64)
    q_xy = np.asarray(q_xy, dtype=np.float64)
    
    if pts_xy.shape[0] != pts_val.shape[0]:
        raise ValueError(f"pts_xy ({pts_xy.shape[0]}) and pts_val ({pts_val.shape[0]}) size mismatch")
    
    if pts_xy.shape[1] != 2 or q_xy.shape[1] != 2:
        raise ValueError("Coordinate arrays must be (n, 2) shape")
    
    if pts_xy.shape[0] == 0:
        raise ValueError("No data points provided")
    
    if q_xy.shape[0] == 0:
        return np.array([])
    
    # Handle adaptive mode
    if adaptive:
        return _adaptive_idw(pts_xy, pts_val, q_xy, k=k, base_power=power, 
                            use_gpu=use_gpu, eps=eps)
    
    # Select backend
    if use_gpu and GPU_AVAILABLE:
        log.debug(f"[IDW] Using GPU backend (n={len(pts_xy)}, m={len(q_xy)}, k={k})")
        return _idw_gpu(pts_xy, pts_val, q_xy, k=k, power=power, eps=eps)
    
    elif CPU_BACKEND == "scipy":
        log.debug(f"[IDW] Using scipy backend (n={len(pts_xy)}, m={len(q_xy)}, k={k})")
        return _idw_scipy(pts_xy, pts_val, q_xy, k=k, power=power, eps=eps)
    
    elif CPU_BACKEND == "sklearn":
        log.debug(f"[IDW] Using sklearn backend (n={len(pts_xy)}, m={len(q_xy)}, k={k})")
        return _idw_sklearn(pts_xy, pts_val, q_xy, k=k, power=power, eps=eps)
    
    else:
        log.warning(f"[IDW] Using slow numpy fallback (n={len(pts_xy)}, m={len(q_xy)}, k={k})")
        return _idw_numpy_chunked(pts_xy, pts_val, q_xy, k=k, power=power, eps=eps)


def get_backend_info() -> dict:
    """
    Get information about available IDW backends.
    
    Returns:
        Dictionary with backend availability and performance info
    """
    return {
        "gpu_available": GPU_AVAILABLE,
        "gpu_error": GPU_ERROR,
        "cpu_backend": CPU_BACKEND,
        "backends": {
            "gpu_cupy": GPU_AVAILABLE,
            "cpu_scipy": CPU_BACKEND == "scipy",
            "cpu_sklearn": CPU_BACKEND == "sklearn",
            "cpu_numpy": True  # Always available
        },
        "recommended": "gpu" if GPU_AVAILABLE else CPU_BACKEND
    }
