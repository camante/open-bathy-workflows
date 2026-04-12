from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

import bathy_main
from geo.raster_ops import compute_depth_from_bed_and_dem


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


def test_load_river_authoritative_support_points_reads_direct_csv_preserving_semantics(tmp_path: Path):
    csv_path = tmp_path / 'auth.csv'
    csv_path.write_text('x,y,depth_m\n500000,4700000,2.5\n', encoding='utf-8')
    cfg = SimpleNamespace(
        river_soundings=None,
        river_authoritative_soundings=str(csv_path),
        extra_xyz_files=None,
        river_soundings_crs='EPSG:32619',
        working_srs='EPSG:32619',
        extra_xyz_crs='EPSG:4326',
        aoi='-71/-70/42/43',
    )
    with mock.patch('support_points.load_extra_xyz_points') as m:
        out = bathy_main._load_river_authoritative_support_points(cfg)
    assert out is not None
    assert not m.called
    assert out.iloc[0]['value_semantics'] == 'absolute_elevation'
    assert float(out.iloc[0]['depth_m']) == 2.5
    assert np.isfinite(float(out.iloc[0]['lon']))
    assert np.isfinite(float(out.iloc[0]['lat']))


def test_compute_depth_from_bed_and_dem_uses_channel_mask_fallback(tmp_path: Path):
    bed = np.array([[ -9999, -9999, -9999, -9999, -9999],
                    [ -9999, 1.0, 1.5, 1.0, -9999],
                    [ -9999, 1.5, 2.0, 1.5, -9999],
                    [ -9999, 1.0, 1.5, 1.0, -9999],
                    [ -9999, -9999, -9999, -9999, -9999]], dtype='float32')
    dem = np.array([[5, 5, 5, 5, 5],
                    [5, -9999, -9999, -9999, 5],
                    [5, -9999, -9999, -9999, 5],
                    [5, -9999, -9999, -9999, 5],
                    [5, 5, 5, 5, 5]], dtype='float32')
    channel = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0]], dtype='float32')
    bed_tif = tmp_path / 'bed_depth_compute.tif'
    dem_tif = tmp_path / 'dem_depth_compute.tif'
    ch_tif = tmp_path / 'channel_depth_compute.tif'
    out_tif = tmp_path / 'out_depth_compute.tif'
    _write_raster(bed_tif, bed, nodata=-9999.0)
    _write_raster(dem_tif, dem, nodata=-9999.0)
    _write_raster(ch_tif, channel, nodata=0.0)
    compute_depth_from_bed_and_dem(bed_tif, dem_tif, out_tif, channel_mask_tif=ch_tif)
    with rasterio.open(out_tif) as ds:
        arr = ds.read(1)
    vals = arr[channel > 0]
    assert np.all(np.isfinite(vals))
    assert np.nanmax(vals) <= 0.0
    assert np.nanmin(vals) < -2.0


def test_compute_depth_from_bed_and_dem_rejects_misaligned_channel_mask(tmp_path: Path):
    arr = np.ones((3, 3), dtype='float32')
    bed_tif = tmp_path / 'bed_misaligned.tif'
    dem_tif = tmp_path / 'dem_misaligned.tif'
    ch_tif = tmp_path / 'channel_misaligned.tif'
    out_tif = tmp_path / 'out_misaligned.tif'
    _write_raster(bed_tif, arr, nodata=-9999.0)
    _write_raster(dem_tif, arr, nodata=-9999.0)
    profile = {
        'driver': 'GTiff', 'height': 2, 'width': 2, 'count': 1, 'dtype': 'float32',
        'crs': 'EPSG:4326', 'transform': from_origin(100.0, 100.0, 2.0, 2.0), 'nodata': 0.0,
    }
    with rasterio.open(ch_tif, 'w', **profile) as ds:
        ds.write(np.ones((2, 2), dtype='float32'), 1)
    import pytest
    with pytest.raises(ValueError, match='channel_mask_tif'):
        compute_depth_from_bed_and_dem(bed_tif, dem_tif, out_tif, channel_mask_tif=ch_tif)


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


def test_compute_depth_from_bed_and_dem_prefers_same_component_surface_before_bank_fallback(tmp_path: Path):
    bed = np.array([
        [-9999, -9999, -9999, -9999, -9999, -9999],
        [-9999, 1.0, 1.2, 1.4, 1.6, -9999],
        [-9999, 1.1, 1.3, 1.5, 1.7, -9999],
        [-9999, 1.2, 1.4, 1.6, 1.8, -9999],
        [-9999, -9999, -9999, -9999, -9999, -9999],
    ], dtype='float32')
    dem = np.array([
        [5, 5, 5, 5, 5, 5],
        [5, 4, -9999, -9999, 4.4, 5],
        [5, 4.1, -9999, -9999, 4.5, 5],
        [5, 4.2, -9999, -9999, 4.6, 5],
        [5, 5, 5, 5, 5, 5],
    ], dtype='float32')
    channel = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype='float32')
    bed_tif = tmp_path / 'bed_component_surface.tif'
    dem_tif = tmp_path / 'dem_component_surface.tif'
    ch_tif = tmp_path / 'channel_component_surface.tif'
    out_tif = tmp_path / 'out_component_surface.tif'
    _write_raster(bed_tif, bed, nodata=-9999.0)
    _write_raster(dem_tif, dem, nodata=-9999.0)
    _write_raster(ch_tif, channel, nodata=0.0)
    receipt = compute_depth_from_bed_and_dem(bed_tif, dem_tif, out_tif, channel_mask_tif=ch_tif)
    assert int(receipt['dem_gap_pixels']) == 6
    assert int(receipt['in_channel_surface_propagation_pixels']) == 6
    assert int(receipt['bank_derived_wse_fallback_pixels']) == 0
    assert int(receipt['unresolved_dem_gap_pixels']) == 0
    with rasterio.open(out_tif) as ds:
        arr = ds.read(1)
    vals = arr[channel > 0]
    assert np.all(np.isfinite(vals))
    assert np.nanmin(vals) < -2.0


def test_compute_depth_from_bed_and_dem_reports_per_component_recovery_modes(tmp_path: Path):
    bed = np.array([
        [-9999, -9999, -9999, -9999, -9999, -9999, -9999],
        [-9999, 1.0, 1.1, 1.2, -9999, 2.0, -9999],
        [-9999, 1.1, 1.2, 1.3, -9999, 2.1, -9999],
        [-9999, 1.2, 1.3, 1.4, -9999, 2.2, -9999],
        [-9999, -9999, -9999, -9999, -9999, -9999, -9999],
    ], dtype='float32')
    dem = np.array([
        [5, 5, 5, 5, 5, 5, 5],
        [5, 4.0, -9999, 4.2, 5, -9999, 4.8],
        [5, 4.1, -9999, 4.3, 5, -9999, 4.9],
        [5, 4.2, -9999, 4.4, 5, -9999, 5.0],
        [5, 5, 5, 5, 5, 5, 5],
    ], dtype='float32')
    channel = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype='float32')
    bed_tif = tmp_path / 'bed_component_modes.tif'
    dem_tif = tmp_path / 'dem_component_modes.tif'
    ch_tif = tmp_path / 'channel_component_modes.tif'
    out_tif = tmp_path / 'out_component_modes.tif'
    _write_raster(bed_tif, bed, nodata=-9999.0)
    _write_raster(dem_tif, dem, nodata=-9999.0)
    _write_raster(ch_tif, channel, nodata=0.0)
    receipt = compute_depth_from_bed_and_dem(bed_tif, dem_tif, out_tif, channel_mask_tif=ch_tif)
    assert int(receipt['components_with_dem_gaps']) == 2
    assert int(receipt['components_using_in_channel_surface_propagation']) == 1
    assert int(receipt['components_using_adjacent_channel_surface_propagation']) == 1
    assert int(receipt['components_using_bank_derived_wse_fallback']) == 0
    summaries = receipt['component_fallback_summaries']
    assert len(summaries) == 2
    modes = {s['recovery_mode'] for s in summaries}
    assert 'same_component_surface' in modes
    assert 'adjacent_channel_surface' in modes


def test_compute_depth_from_bed_and_dem_uses_adjacent_channel_surface_before_bank_fallback(tmp_path: Path):
    bed = np.array([
        [-9999, -9999, -9999, -9999, -9999, -9999, -9999],
        [-9999, 1.0, 1.1, 1.2, -9999, 2.0, -9999],
        [-9999, 1.1, 1.2, 1.3, -9999, 2.1, -9999],
        [-9999, 1.2, 1.3, 1.4, -9999, 2.2, -9999],
        [-9999, -9999, -9999, -9999, -9999, -9999, -9999],
    ], dtype='float32')
    dem = np.array([
        [5, 5, 5, 5, 5, 5, 5],
        [5, 4.0, -9999, 4.2, 5, -9999, 4.8],
        [5, 4.1, -9999, 4.3, 5, -9999, 4.9],
        [5, 4.2, -9999, 4.4, 5, -9999, 5.0],
        [5, 5, 5, 5, 5, 5, 5],
    ], dtype='float32')
    channel = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype='float32')
    bed_tif = tmp_path / 'bed_adjacent_component_surface.tif'
    dem_tif = tmp_path / 'dem_adjacent_component_surface.tif'
    ch_tif = tmp_path / 'channel_adjacent_component_surface.tif'
    out_tif = tmp_path / 'out_adjacent_component_surface.tif'
    _write_raster(bed_tif, bed, nodata=-9999.0)
    _write_raster(dem_tif, dem, nodata=-9999.0)
    _write_raster(ch_tif, channel, nodata=0.0)
    receipt = compute_depth_from_bed_and_dem(bed_tif, dem_tif, out_tif, channel_mask_tif=ch_tif)
    assert int(receipt['components_with_dem_gaps']) == 2
    assert int(receipt['components_using_adjacent_channel_surface_propagation']) == 1
    assert int(receipt['bank_derived_wse_fallback_pixels']) == 0
    summaries = receipt['component_fallback_summaries']
    assert len(summaries) == 2
    adjacent = [s for s in summaries if s['recovery_mode'] == 'adjacent_channel_surface']
    assert len(adjacent) == 1
    assert adjacent[0]['adjacent_component_ids'] == [1]
    with rasterio.open(out_tif) as ds:
        arr = ds.read(1)
    vals = arr[channel > 0]
    assert np.all(np.isfinite(vals))
    assert np.nanmin(vals) < -2.0


def test_compute_depth_from_bed_and_dem_uses_expanded_channel_surface_before_bank_fallback(tmp_path: Path):
    bed = np.array([
        [-9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999],
        [-9999, 1.0, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, 2.0, -9999],
        [-9999, 1.1, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, 2.1, -9999],
        [-9999, 1.2, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, 2.2, -9999],
        [-9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999],
    ], dtype='float32')
    dem = np.array([
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, -9999, 5, 5, 5, 5, 5, 5, 5, 5, 4.8, 5],
        [5, -9999, 5, 5, 5, 5, 5, 5, 5, 5, 4.9, 5],
        [5, -9999, 5, 5, 5, 5, 5, 5, 5, 5, 5.0, 5],
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    ], dtype='float32')
    channel = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='float32')
    bed_tif = tmp_path / 'bed_expanded_component_surface.tif'
    dem_tif = tmp_path / 'dem_expanded_component_surface.tif'
    ch_tif = tmp_path / 'channel_expanded_component_surface.tif'
    out_tif = tmp_path / 'out_expanded_component_surface.tif'
    _write_raster(bed_tif, bed, nodata=-9999.0)
    _write_raster(dem_tif, dem, nodata=-9999.0)
    _write_raster(ch_tif, channel, nodata=0.0)
    receipt = compute_depth_from_bed_and_dem(bed_tif, dem_tif, out_tif, channel_mask_tif=ch_tif)
    assert int(receipt['components_with_dem_gaps']) == 1
    assert int(receipt['components_using_adjacent_channel_surface_propagation']) == 0
    assert int(receipt['components_using_expanded_channel_surface_propagation']) == 1
    assert int(receipt['bank_derived_wse_fallback_pixels']) == 0
    summaries = receipt['component_fallback_summaries']
    expanded = [s for s in summaries if s['recovery_mode'] == 'expanded_channel_surface']
    assert len(expanded) == 1
    assert expanded[0]['expanded_adjacent_component_ids'] == [2]
    with rasterio.open(out_tif) as ds:
        arr = ds.read(1)
    vals = arr[channel > 0]
    assert np.all(np.isfinite(vals))
