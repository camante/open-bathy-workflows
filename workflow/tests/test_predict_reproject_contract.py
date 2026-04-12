from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from predict import _source_expected_nodata


def _write_tif(path: Path, *, nodata: float | None) -> Path:
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32619",
        "transform": from_origin(0.0, 0.0, 1.0, 1.0),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 1)
    return path


def test_source_expected_nodata_preserves_declared_source_nodata(tmp_path: Path):
    p = _write_tif(tmp_path / "in.tif", nodata=-9999.0)
    assert _source_expected_nodata(str(p)) == -9999.0


def test_source_expected_nodata_returns_none_when_source_has_no_nodata(tmp_path: Path):
    p = _write_tif(tmp_path / "in_no_nodata.tif", nodata=None)
    assert _source_expected_nodata(str(p)) is None
