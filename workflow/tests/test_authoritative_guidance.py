from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from authoritative_guidance import export_raster_points_to_csv


def test_export_raster_points_to_csv_negative_only(tmp_path: Path):
    src = tmp_path / 'auth.tif'
    arr = np.array([[1.0, -2.0], [-3.0, np.nan]], dtype='float32')
    with rasterio.open(
        src,
        'w',
        driver='GTiff',
        height=2,
        width=2,
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=(0.01, 0.0, -71.0, 0.0, -0.01, 43.0),
        nodata=-9999.0,
    ) as ds:
        out = np.where(np.isfinite(arr), arr, -9999.0)
        ds.write(out, 1)
    csv_path = tmp_path / 'pts.csv'
    info = export_raster_points_to_csv(src, csv_path, out_crs='EPSG:4326', max_points=100, negative_only=True)
    df = pd.read_csv(csv_path)
    assert info['count'] == 2
    assert len(df) == 2
    assert (df['depth_m'] < 0).all()
    assert set(df['source']) == {'authoritative_base'}
