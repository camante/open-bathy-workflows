from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

import bathy_main


def _write_raster(path: Path, arr: np.ndarray, nodata: float = -9999.0):
    profile = {
        'driver': 'GTiff',
        'height': arr.shape[0],
        'width': arr.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': from_origin(0.0, float(arr.shape[0]), 1.0, 1.0),
        'nodata': nodata,
    }
    with rasterio.open(path, 'w', **profile) as ds:
        ds.write(arr.astype('float32'), 1)


def test_load_river_authoritative_support_points_uses_authoritative_fallback():
    cfg = SimpleNamespace(
        river_soundings=None,
        river_authoritative_soundings='/tmp/auth.csv',
        extra_xyz_files=None,
        river_soundings_crs='EPSG:26919',
        working_srs='EPSG:32619',
        extra_xyz_crs='EPSG:4326',
        aoi='-71/-70/42/43',
    )
    fake = pd.DataFrame({
        'longitude': [-70.9],
        'latitude': [42.8],
        'depth_m': [3.0],
        'source': ['authoritative_base'],
    })
    with mock.patch('support_points.load_extra_xyz_points', return_value=fake) as m:
        out = bathy_main._load_river_authoritative_support_points(cfg)
    assert m.call_args.kwargs['crs'] == 'EPSG:26919'
    assert out.iloc[0]['value_semantics'] == 'absolute_elevation'


def test_recover_river_depth_from_bank_proxy_when_bed_minus_dem_collapses(tmp_path: Path):
    depth = np.zeros((5, 5), dtype='float32')
    dem = np.array([
        [5, 5, 5, 5, 5],
        [5, 4, 4, 4, 5],
        [5, 4, 4, 4, 5],
        [5, 4, 4, 4, 5],
        [5, 5, 5, 5, 5],
    ], dtype='float32')
    bed = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 1, 1.5, 1, np.nan],
        [np.nan, 1.5, 2, 1.5, np.nan],
        [np.nan, 1, 1.5, 1, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype='float32')
    channel = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype='uint8')
    # use the same DEM as bed in-channel to force base collapse to ~0
    dem_same = dem.copy()
    dem_same[channel > 0] = bed[channel > 0]

    depth_tif = tmp_path / 'depth.tif'
    bed_tif = tmp_path / 'bed.tif'
    dem_tif = tmp_path / 'dem.tif'
    ch_tif = tmp_path / 'channel.tif'
    _write_raster(depth_tif, depth, nodata=-9999.0)
    _write_raster(bed_tif, np.nan_to_num(bed, nan=-9999.0), nodata=-9999.0)
    _write_raster(dem_tif, dem_same, nodata=-9999.0)
    _write_raster(ch_tif, channel.astype('float32'), nodata=0.0)

    cfg = SimpleNamespace(
        river_dem=dem_tif,
        river_soundings=None,
        river_authoritative_soundings=None,
        extra_xyz_files=None,
        river_soundings_crs='EPSG:4326',
        working_srs='EPSG:4326',
        extra_xyz_crs='EPSG:4326',
        aoi='0/5/0/5',
    )
    got = bathy_main._recover_river_depth_from_support(
        cfg=cfg,
        depth_tif=depth_tif,
        bed_tif=bed_tif,
        channel_mask_tif=ch_tif,
        logger=bathy_main.log,
    )
    assert got['recovered'] is True
    with rasterio.open(depth_tif) as ds:
        arr = ds.read(1)
        vals = arr[channel > 0]
    assert np.nanpercentile(vals, 99) - np.nanpercentile(vals, 1) > 0.05
    assert np.nanmax(vals) <= 0.0
    assert np.nanmin(vals) < -1.0
