from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from authoritative_guidance import build_projected_measured_only_authoritative_sampling_raster


def test_build_projected_measured_only_authoritative_sampling_raster_masks_outside_support(tmp_path: Path):
    src = tmp_path / "source.tif"
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:32619",
        transform=from_origin(0, 2, 1, 1),
        nodata=-9999.0,
    ) as ds:
        ds.write(arr, 1)

    support = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 1, 1, 2)], crs="EPSG:32619")
    gpkg = tmp_path / "support.gpkg"
    support.to_file(gpkg, driver="GPKG")

    out = tmp_path / "measured_only.tif"
    info = build_projected_measured_only_authoritative_sampling_raster(
        src,
        gpkg,
        out,
        aoi="0/2/0/2",
        dst_crs="EPSG:32619",
        res_m=1.0,
    )

    assert out.exists()
    with rasterio.open(out) as ds:
        data = ds.read(1)
        nodata = ds.nodata
    finite = data[data != nodata]
    assert finite.size == 1
    assert float(finite[0]) == 1.0
    assert info["sampling_contract"] == "measured_only_projected_masked_by_support_coverage"
