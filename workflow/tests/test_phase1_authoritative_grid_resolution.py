from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import bathy_main
import xs_infer_bathy_raster


def _write_projected_raster(path: Path, res: float = 3.0):
    arr = np.arange(16, dtype="float32").reshape(4, 4)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32619",
        transform=from_origin(500000.0, 4700000.0, res, res),
        nodata=-9999.0,
    ) as ds:
        ds.write(arr, 1)


def test_resolve_river_dem_resolution_uses_authoritative_grid(tmp_path):
    auth = tmp_path / "auth.tif"
    _write_projected_raster(auth, res=3.0)
    cfg = type("Cfg", (), {"river_dem_res_m": 0.0, "authoritative_base": auth})()
    report = {}

    resolved = bathy_main._resolve_river_dem_resolution_m(cfg, "EPSG:32619", report)

    assert abs(resolved - 3.0) < 1e-6
    assert abs(cfg.river_dem_res_m - 3.0) < 1e-6
    info = report["river"]["dem_auto"]
    assert info["river_dem_res_m_source"] == "authoritative_base_grid"
    assert abs(info["river_dem_res_m_resolved"] - 3.0) < 1e-6


def test_require_template_pixel_size_m_uses_template_grid(tmp_path):
    tmpl = tmp_path / "template.tif"
    _write_projected_raster(tmpl, res=3.0)

    with rasterio.open(tmpl) as ds:
        px = xs_infer_bathy_raster._require_template_pixel_size_m(ds)

    assert abs(px - 3.0) < 1e-6
