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

    This is a heuristic used for auto-enabling chunked prediction. It assumes
    n_features float32 arrays plus temporary buffers (overhead_factor).
    """
    try:
        n_pix = int(width) * int(height)
        bytes_needed = float(n_pix) * float(n_features) * float(dtype_bytes) * float(overhead_factor)
        return bytes_needed / 1e9
    except Exception:
        return 0.0


try:
    import predict
except Exception as e:
    predict = None
    log.warning("[predict_chunked] predict import failed: %s", e)


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

    if not rf_model_path.exists():
        raise FileNotFoundError(f"RF model not found: {rf_model_path}")
    if not meta_json_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {meta_json_path}")

    if land_mask_path is None:
        raise ValueError("land_mask_path is required")

    # Delegate to the main predictor (already windowed).
    log.info(
        "[predict_chunked] Delegating to predict.predict_scene (windowed). tile_size=%s overlap=%s max_memory_gb=%s",
        tile_size, overlap, max_memory_gb
    )

    predict.predict_scene(
        s2_paths={k: str(v) for k, v in band_paths.items()},
        land_mask_path=str(land_mask_path),
        rf_model_path=str(rf_model_path),
        meta_json_path=str(meta_json_path),
        out_path=str(out_raster),
        tile_size=int(tile_size),
        # pass through extra kwargs that predict_scene understands (safe to ignore unknown in caller)
        **{k: v for k, v in kwargs.items() if k in {
            "sdb_mode", "s2_smooth_kernel", "enable_doa",
            "cw_min", "land_max", "land_mask_type", "land_mask_water_val",
            "land_mask_invert", "land_mask_threshold",
            "linf_estimate_deepwater", "linf_deepwater_nir_max", "linf_deepwater_bright_max", "linf_percentile",
            "align_mode", "align_tie_points_gpkg", "align_min_points", "align_depth_bins", "align_source_priority",
            "align_extra_points", "align_max_abs_residual_m_for_fit",
        }},
    )

    return {"output_raster": out_raster}
