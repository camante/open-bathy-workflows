from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from authoritative_guidance import export_raster_points_to_csv, build_projected_authoritative_raster


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


def test_build_projected_authoritative_raster_uses_baseline_fallback_for_bank_gaps(tmp_path: Path):
    src = tmp_path / 'auth_only.tif'
    fallback = tmp_path / 'baseline.tif'
    auth = np.array([[np.nan, np.nan, np.nan], [np.nan, 2.0, np.nan], [np.nan, np.nan, np.nan]], dtype='float32')
    base = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], dtype='float32')
    transform = (0.00009, 0.0, 0.0, 0.0, -0.00009, 0.00027)
    for path, arr in ((src, auth), (fallback, base)):
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=3,
            width=3,
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
            nodata=-9999.0,
        ) as ds:
            ds.write(np.where(np.isfinite(arr), arr, -9999.0).astype('float32'), 1)
    out = tmp_path / 'projected.tif'
    info = build_projected_authoritative_raster(
        src,
        out,
        aoi='0/0.00027/0/0.00027',
        dst_crs='EPSG:4326',
        res_m=0.00009,
        fallback_raster=fallback,
    )
    with rasterio.open(out) as ds:
        arr = ds.read(1)
        arr = np.where(arr == ds.nodata, np.nan, arr)
    assert arr.shape == (3, 3)
    assert info['fallback_filled_cells'] > 0
    assert np.isclose(arr[1, 1], 2.0, atol=1e-6)
    assert np.isclose(arr[0, 0], 10.0, atol=1e-6)
