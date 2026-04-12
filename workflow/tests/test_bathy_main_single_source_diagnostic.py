from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

import bathy_main


def _write_raster(path: Path, arr: np.ndarray, *, nodata: float = -9999.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, arr.shape[0], 1, 1),
        nodata=nodata,
    ) as ds:
        ds.write(arr.astype("float32"), 1)
    return path


def test_single_source_fuse_uses_baseline_background_for_diagnostic_candidate(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    baseline = _write_raster(tmp_path / "cudem_baseline_interpolation.tif", np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32))
    auth = _write_raster(tmp_path / "authoritative_base.tif", np.array([[-9999.0, -9999.0], [-9999.0, -9999.0]], dtype=np.float32), nodata=-9999.0)
    river = _write_raster(tmp_path / "river_depth.tif", np.array([[-9999.0, -2.0], [-9999.0, -9999.0]], dtype=np.float32), nodata=-9999.0)
    cfg = SimpleNamespace(out_dir=out_dir, authoritative_base=auth)
    report = {"authoritative_base_auto": {"baseline_cudem_interpolation": str(baseline)}}
    out = bathy_main.fuse(cfg, sdb_raster=None, river_raster=river, report=report)
    assert out is not None
    with rasterio.open(out) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata
    arr = np.where(arr == nodata, np.nan, arr)
    assert arr[0, 0] == 10.0
    assert arr[0, 1] == -2.0
    assert arr[1, 0] == 12.0
    assert report["fusion"]["diagnostic_candidate"]["used_baseline_background"] is True
