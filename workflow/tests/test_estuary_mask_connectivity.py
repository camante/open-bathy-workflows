import sys
import tempfile
from pathlib import Path

# Avoid the repo-local workflow/rasterio.py shadowing the installed rasterio package.
sys.path = [p for p in sys.path if '/workflow/tests' not in p and not p.endswith('/workflow')]

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.append(str(Path(__file__).resolve().parents[1]))
import river_masking


class DummyCfg:
    estuary_width_ratio_thresh = 3.0


def _write_u8(path: Path, arr: np.ndarray):
    transform = from_origin(0.0, float(arr.shape[0]), 1.0, 1.0)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype='uint8',
        crs='EPSG:4326',
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(arr.astype('uint8'), 1)


def test_estuary_ocean_connectivity_uses_channel_domain(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        channel = np.zeros((20, 20), dtype='uint8')
        channel[5:15, 2:18] = 1
        channel_path = td / 'channel.tif'
        _write_u8(channel_path, channel)

        # Ocean mask convention here follows the workflow path: value 0 = ocean water.
        ocean = np.ones((20, 20), dtype='uint8')
        ocean[:, :2] = 0
        ocean_path = td / 'ocean.tif'
        _write_u8(ocean_path, ocean)

        width_estuary = np.zeros((20, 20), dtype=bool)
        width_estuary[7:13, 8:16] = True  # connected to ocean only through channel pixels

        monkeypatch.setattr(river_masking, 'build_hydraulic_estuary_hint_mask', lambda **kwargs: (np.zeros_like(channel), {'backwater_slope_reaches': 0}))
        monkeypatch.setattr(river_masking, 'distance_transform_edt', lambda arr, sampling=None: np.where(channel > 0, 5.0, 0.0))

        # Force width ratio signal to become our synthetic estuary candidate mask.
        # Patch binary-opening/dilation as identity-like so the test focuses on connectivity.
        monkeypatch.setattr(river_masking, 'binary_dilation', lambda arr, iterations=1: arr)
        monkeypatch.setattr(river_masking, 'binary_opening', lambda arr, iterations=1: arr)

        # Custom EDT for width ratio call path: when called on channel bool, return values that
        # make width_estuary the estuary candidate mask through thresholding.
        def fake_edt(arr, sampling=None):
            if arr.dtype == bool and arr.shape == channel.shape and np.array_equal(arr, channel > 0):
                out = np.ones_like(arr, dtype='float32')
                out[width_estuary] = 10.0
                return out
            return np.zeros_like(arr, dtype='float32')

        monkeypatch.setattr(river_masking, 'distance_transform_edt', fake_edt)

        removed, mask_path = river_masking.clip_channel_mask_for_estuary(
            channel_path,
            DummyCfg(),
            ocean_mask_path=ocean_path,
            report={},
        )

        assert removed > 0
        assert mask_path is not None and mask_path.exists()
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1) > 0
        # The upstream estuary candidate should survive because it is ocean-connected via the channel domain.
        assert int(mask.sum()) >= int(width_estuary.sum())


def test_estuary_channel_distance_cap_limits_mask_extent(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        channel = np.zeros((30, 160), dtype='uint8')
        channel[10:20, 2:150] = 1
        channel_path = td / 'channel_long.tif'
        _write_u8(channel_path, channel)

        ocean = np.ones((30, 160), dtype='uint8')
        ocean[:, :2] = 0
        ocean_path = td / 'ocean.tif'
        _write_u8(ocean_path, ocean)

        width_estuary = np.zeros((30, 160), dtype=bool)
        width_estuary[10:20, 8:28] = True

        monkeypatch.setattr(river_masking, 'build_hydraulic_estuary_hint_mask', lambda **kwargs: (np.zeros_like(channel), {'backwater_slope_reaches': 0}))
        monkeypatch.setattr(river_masking, 'binary_dilation', lambda arr, iterations=1: arr)
        monkeypatch.setattr(river_masking, 'binary_opening', lambda arr, iterations=1: arr)

        def fake_edt(arr, sampling=None):
            if arr.dtype == bool and arr.shape == channel.shape and np.array_equal(arr, channel > 0):
                out = np.ones_like(arr, dtype='float32')
                out[width_estuary] = 10.0
                return out
            return np.zeros_like(arr, dtype='float32')

        monkeypatch.setattr(river_masking, 'distance_transform_edt', fake_edt)

        removed, mask_path = river_masking.clip_channel_mask_for_estuary(
            channel_path,
            DummyCfg(),
            ocean_mask_path=ocean_path,
            report={},
        )

        assert removed > 0
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1) > 0
        flagged_cols = np.where(mask.any(axis=0))[0]
        assert flagged_cols.size > 0
        assert flagged_cols.max() <= 32


def test_estuary_transition_mask_is_retained_channel_subset(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        channel = np.zeros((30, 80), dtype='uint8')
        channel[10:20, 2:70] = 1
        channel_path = td / 'channel_transition.tif'
        _write_u8(channel_path, channel)

        ocean = np.ones((30, 80), dtype='uint8')
        ocean[:, :2] = 0
        ocean_path = td / 'ocean.tif'
        _write_u8(ocean_path, ocean)

        width_estuary = np.zeros((30, 80), dtype=bool)
        width_estuary[10:20, 6:20] = True

        monkeypatch.setattr(river_masking, 'build_hydraulic_estuary_hint_mask', lambda **kwargs: (np.zeros_like(channel), {'backwater_slope_reaches': 0}))
        monkeypatch.setattr(river_masking, 'binary_opening', lambda arr, iterations=1: arr)

        def fake_edt(arr, sampling=None):
            if arr.dtype == bool and arr.shape == channel.shape and np.array_equal(arr, channel > 0):
                out = np.ones_like(arr, dtype='float32')
                out[width_estuary] = 10.0
                return out
            return np.zeros_like(arr, dtype='float32')

        monkeypatch.setattr(river_masking, 'distance_transform_edt', fake_edt)

        report = {}
        removed, mask_path = river_masking.clip_channel_mask_for_estuary(
            channel_path,
            DummyCfg(),
            ocean_mask_path=ocean_path,
            report=report,
        )

        assert removed > 0
        transition_path = td / 'estuary_transition_mask.tif'
        debug_receipt = td / 'estuary_debug_receipt.json'
        assert transition_path.exists()
        assert debug_receipt.exists()

        with rasterio.open(channel_path) as ds_ch, rasterio.open(transition_path) as ds_tr:
            channel_final = ds_ch.read(1) > 0
            transition = ds_tr.read(1) > 0
        assert np.any(transition)
        assert np.all(transition <= channel_final)
        assert int(transition.sum()) <= int(channel_final.sum())
        assert report['river']['outputs']['estuary_transition'].endswith('estuary_transition_mask.tif')
