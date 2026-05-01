#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict_chunked.py

This repo's main prediction implementation (predict.py::predict_scene) already processes
the AOI in *windows/tiles* and streams band reads from disk. In practice, that provides
the memory behavior most users expect from a "chunked" predictor.

Earlier versions of this file attempted to implement a second, independent chunked
pipeline with a different API. That caused real integration bugs (mismatched function
signatures, missing imports, and ambiguous outputs).

Current behavior:
- Provide a *thin*, API-compatible wrapper used by sdb_main.py.
- Delegate to predict.predict_scene (windowed) and return a dict containing:
    {"output_raster": Path(...)} for sdb_main to post-process.

If you truly need a separate chunked mosaicking approach in the future, implement it
behind this same interface without changing sdb_main.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Any

log = logging.getLogger(__name__)

def estimate_memory_requirement_gb(width: int, height: int, n_features: int = 14, dtype_bytes: int = 4, overhead_factor: float = 3.0) -> float:
    """Rough in-memory requirement estimate for standard (non-chunked) prediction.

    Delegates to :func:`chunked_processing.estimate_memory_requirement_gb` when
    available, falling back to a simple heuristic otherwise.  The canonical
    memory-estimation logic lives in ``chunked_processing.py``; this wrapper
    preserves the legacy (width, height, n_features) call signature used by
    ``sdb_main.py``.
    """
    try:
        from core.chunked_processing import estimate_memory_requirement_gb as _canonical
        import numpy as np
        return _canonical(
            height=int(height),
            width=int(width),
            n_bands=int(n_features),
            dtype=np.dtype(f"f{dtype_bytes}"),
            n_arrays=int(overhead_factor),
        )
    except Exception:
        log.debug("estimate_memory_requirement_gb: suppressed exception", exc_info=True)
        # Fallback: simple heuristic if chunked_processing is unavailable
        try:
            n_pix = int(width) * int(height)
            bytes_needed = float(n_pix) * float(n_features) * float(dtype_bytes) * float(overhead_factor)
            return bytes_needed / 1e9
        except Exception:
            log.debug("estimate_memory_requirement_gb: suppressed exception", exc_info=True)
            return 0.0


try:
    import predict
except Exception as e:
    predict = None
    log.warning("predict import failed: %s", e)


def predict_scene_chunked(
    model_dir: Path,
    band_paths: Dict[str, Any],
    out_dir: Path,
    final_out_path: Optional[str] = None,
    land_mask_path: Optional[str] = None,
    tile_size: int = 512,
    overlap: int = 0,
    max_memory_gb: float = 8.0,
    **kwargs,
) -> Dict[str, Path]:
    """API-compatible wrapper for sdb_main.py.

    Parameters are intentionally permissive (Path or str for paths) because the caller
    may pass strings.

    Returns:
        dict with key 'output_raster' pointing to the created GeoTIFF.
    """
    if predict is None:
        raise ImportError("predict.py is not available")

    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write to a distinct path so the caller can copy into its chosen final output
    # (no canonical filename assumptions).
    if final_out_path is not None:
        stem = Path(final_out_path).stem
        out_raster = out_dir / f"{stem}_chunked_tmp.tif"
    else:
        out_raster = out_dir / "prediction_chunked_tmp.tif"

    rf_model_path = model_dir / "rf_model.pkl"
    meta_json_path = model_dir / "model_meta.json"
    stumpf_lr_path = model_dir / "stumpf_lr.pkl"

    if not rf_model_path.exists():
        raise FileNotFoundError(f"RF model not found: {rf_model_path}")
    if not meta_json_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {meta_json_path}")

    if land_mask_path is None:
        raise ValueError("land_mask_path is required")

    log.info(
        "[predict_chunked] Delegating to predict.predict_scene (windowed). tile_size=%s overlap=%s max_memory_gb=%s",
        tile_size, overlap, max_memory_gb
    )

    _known = {
        "sdb_mode", "s2_smooth_kernel", "enable_doa",
        "cw_min", "land_max", "land_mask_type", "land_mask_water_val",
        "land_mask_invert", "land_mask_threshold",
        "linf_estimate_deepwater", "linf_deepwater_nir_max", "linf_deepwater_bright_max", "linf_percentile",
        "align_mode", "align_tie_points_gpkg", "align_min_points", "align_depth_bins", "align_source_priority",
        "align_extra_points", "align_max_abs_residual_m_for_fit",
        "write_confidence", "write_provenance", "min_confidence_threshold",
    }
    predict.predict_scene(
        s2_paths={k: str(v) for k, v in band_paths.items()},
        land_mask_path=str(land_mask_path),
        rf_model_path=str(rf_model_path),
        meta_json_path=str(meta_json_path),
        stumpf_lr_path=str(stumpf_lr_path) if stumpf_lr_path.exists() else None,
        out_path=str(out_raster),
        tile_size=int(tile_size),
        **{k: v for k, v in kwargs.items() if k in _known},
    )

    return {"output_raster": out_raster}
