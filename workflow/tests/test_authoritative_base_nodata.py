from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from cudem_authoritative import _cached_authoritative_outputs_valid, apply_mask_to_mosaic


def _write_tif(path: Path, arr: np.ndarray, *, nodata: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4269",
        "transform": from_origin(-71.0, 42.0, 1.0, 1.0),
        "nodata": float(nodata),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
    return path


def test_apply_mask_to_mosaic_converts_nan_outside_support_to_workflow_nodata():
    mosaic = np.array([[[1.0, -999999.0], [5.0, np.nan]]], dtype=np.float32)
    mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    out = apply_mask_to_mosaic(mosaic, mask, -9999.0)
    assert out.shape == (1, 2, 2)
    assert float(out[0, 0, 0]) == 1.0
    assert float(out[0, 0, 1]) == -9999.0
    assert float(out[0, 1, 1]) == -9999.0


def test_cached_authoritative_outputs_valid_rejects_legacy_negative_999999_nodata(tmp_path: Path):
    auth = _write_tif(tmp_path / "authoritative_base.tif", np.array([[1.0, -999999.0]], dtype=np.float32), nodata=-999999.0)
    baseline = _write_tif(tmp_path / "cudem_baseline_interpolation.tif", np.array([[1.0, -999999.0]], dtype=np.float32), nodata=-999999.0)
    assert _cached_authoritative_outputs_valid(auth_path=auth, baseline_path=baseline, expected_nodata=-9999.0) is False
