from pathlib import Path

import pandas as pd
import rasterio
import numpy as np

from authoritative_guidance import prepare_authoritative_river_soundings_points


def test_prepare_authoritative_river_soundings_points(tmp_path: Path):
    src = tmp_path / "auth.tif"
    arr = np.array([[-1.0, 2.0], [-3.5, -9999.0]], dtype="float32")
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=(0.01, 0.0, -71.0, 0.0, -0.01, 43.0),
        nodata=-9999.0,
    ) as ds:
        ds.write(arr, 1)
    out_csv = tmp_path / "river_pts.csv"
    info = prepare_authoritative_river_soundings_points(src, out_csv, out_crs="EPSG:4326")
    df = pd.read_csv(out_csv)
    assert info["target_role"] == "river_soundings"
    assert len(df) == 2
    assert (df["depth_m"] < 0).all()
