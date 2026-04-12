from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from authoritative_guidance import build_projected_authoritative_raster
from nodata_utils import read_band_sanitized
from raster_contract import cached_raster_semantics_valid, validate_gdal_output


def _write_float_tif(path: Path, arr: np.ndarray, *, nodata: float) -> Path:
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4269",
        "transform": from_origin(-71.0, 42.0, 0.001, 0.001),
        "nodata": float(nodata),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
    return path


def test_common_read_path_sanitizes_declared_and_common_sentinels(tmp_path: Path):
    p = _write_float_tif(tmp_path / "in.tif", np.array([[1.0, -999999.0], [-99999.0, 5.0]], dtype=np.float32), nodata=-999999.0)
    arr = read_band_sanitized(p, 1, dtype=np.float32)
    assert np.isfinite(arr[0, 0])
    assert np.isnan(arr[0, 1])
    assert np.isnan(arr[1, 0])
    assert np.isfinite(arr[1, 1])


def test_cached_raster_semantics_rejects_alternate_sentinel_payload(tmp_path: Path):
    p = _write_float_tif(tmp_path / "bad.tif", np.array([[1.0, -999999.0], [2.0, 3.0]], dtype=np.float32), nodata=-9999.0)
    ok, reason, _ = cached_raster_semantics_valid(p, expected_nodata=-9999.0, expected_dtype="float32", min_allowed=-1000.0, max_allowed=10000.0)
    assert ok is False
    assert "unexpected sentinel-valued pixels" in reason


def test_build_projected_authoritative_raster_normalizes_output_nodata(tmp_path: Path):
    src = _write_float_tif(tmp_path / "src.tif", np.array([[1.0, -999999.0], [2.0, 3.0]], dtype=np.float32), nodata=-999999.0)
    out = tmp_path / "out.tif"
    info = build_projected_authoritative_raster(src, out, aoi="-71/-70.998/41.998/42", dst_crs="EPSG:32619", res_m=50.0)
    assert Path(info["path"]).exists()
    with rasterio.open(out) as ds:
        assert ds.nodata == -9999.0
    validate_gdal_output(out, operation="test", expected_nodata=-9999.0, expected_dtype="float32", min_allowed=-1000.0, max_allowed=10000.0)
