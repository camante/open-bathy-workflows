from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from hybrid_merge import merge_hybrid_river_bed, reconstruct_xs_mainstem_relative


def _write_raster(path: Path, arr: np.ndarray, nodata: float = -9999.0):
    profile = {
        'driver': 'GTiff',
        'height': arr.shape[0],
        'width': arr.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:32619',
        'transform': from_origin(0, arr.shape[0], 1, 1),
        'nodata': nodata,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr.astype('float32'), 1)


def test_hybrid_merge_receipt_and_precedence(tmp_path: Path):
    nod = -9999.0
    template = tmp_path / 'template.tif'
    xs = tmp_path / 'xs.tif'
    sk = tmp_path / 'sk.tif'
    ms = tmp_path / 'mainstem.tif'
    ch = tmp_path / 'channel.tif'
    out = tmp_path / 'merged.tif'
    receipt = tmp_path / 'receipt.json'

    template_arr = np.zeros((4, 4), dtype=np.float32)
    xs_arr = np.full((4, 4), nod, dtype=np.float32)
    xs_arr[1, 1] = 10.0
    xs_arr[1, 2] = 20.0
    sk_arr = np.full((4, 4), 5.0, dtype=np.float32)
    ms_arr = np.zeros((4, 4), dtype=np.float32)
    ms_arr[1, 1] = 1.0
    ms_arr[1, 2] = 1.0
    ms_arr[2, 1] = 1.0
    ch_arr = np.ones((4, 4), dtype=np.float32)

    _write_raster(template, template_arr, nod)
    _write_raster(xs, xs_arr, nod)
    _write_raster(sk, sk_arr, nod)
    _write_raster(ms, ms_arr, nod)
    _write_raster(ch, ch_arr, nod)

    info = merge_hybrid_river_bed(xs, sk, ms, ch, out, template, nod, receipt_json=receipt)
    assert out.exists()
    assert receipt.exists()
    with rasterio.open(out) as ds:
        merged = ds.read(1)
    assert 5.0 < merged[1, 1] < 10.0
    assert 5.0 < merged[1, 2] < 20.0
    # mainstem pixel with no XS should remain skeleton-backed, not unresolved
    assert merged[2, 1] == 5.0
    saved = json.loads(receipt.read_text())
    assert saved['xs_wins_mainstem_pixels'] == 2
    assert saved['skeleton_only_mainstem_pixels'] == 1
    assert saved['unresolved_mainstem_pixels'] == 0
    assert info['merge_rule'] == 'xs_mainstem_with_skeleton_mainstem_helper_only'


def test_reconstruct_xs_mainstem_relative_suppresses_transverse_steps(tmp_path: Path):
    nod = -9999.0
    template = tmp_path / 'template.tif'
    xs = tmp_path / 'xs_raw.tif'
    sk = tmp_path / 'sk.tif'
    ms = tmp_path / 'mainstem.tif'
    out = tmp_path / 'xs_reconstructed.tif'
    receipt = tmp_path / 'xs_reconstructed_receipt.json'

    shape = (5, 12)
    template_arr = np.zeros(shape, dtype=np.float32)
    xs_arr = np.full(shape, nod, dtype=np.float32)
    sk_arr = np.full(shape, 10.0, dtype=np.float32)
    ms_arr = np.zeros(shape, dtype=np.float32)
    ms_arr[1:4, 1:11] = 1.0

    # Two neighboring XS slices with different absolute bed levels. Raw raster has
    # a transverse step that should be reduced when converted to backbone-relative guidance.
    xs_arr[1:4, 2:5] = 8.0
    xs_arr[1:4, 5:8] = 6.0
    xs_arr[1:4, 8:10] = 8.0

    _write_raster(template, template_arr, nod)
    _write_raster(xs, xs_arr, nod)
    _write_raster(sk, sk_arr, nod)
    _write_raster(ms, ms_arr, nod)

    info = reconstruct_xs_mainstem_relative(xs, sk, ms, out, template, nod, receipt_json=receipt)
    assert out.exists()
    assert receipt.exists()
    with rasterio.open(out) as ds:
        recon = ds.read(1)

    # The reconstructed raster should remain valid on the mainstem and should not
    # preserve the full raw section-to-section jump.
    assert np.all(np.isfinite(recon[1:4, 2:10]))
    raw_jump = float(xs_arr[2, 6] - xs_arr[2, 3])
    recon_jump = float(recon[2, 6] - recon[2, 3])
    assert raw_jump < -1.5
    assert recon_jump > raw_jump
    assert info['reconstruction_rule'] == 'skeleton_backbone_plus_xs_negative_core_anomaly'


def test_hybrid_merge_drops_weak_stage_skeleton_helper(tmp_path: Path):
    nod = -9999.0
    template = tmp_path / 'template.tif'
    xs = tmp_path / 'xs.tif'
    sk = tmp_path / 'sk.tif'
    ms = tmp_path / 'mainstem.tif'
    ch = tmp_path / 'channel.tif'
    stage = tmp_path / 'sk_stage.tif'
    out = tmp_path / 'merged.tif'

    template_arr = np.zeros((4, 4), dtype=np.float32)
    xs_arr = np.full((4, 4), nod, dtype=np.float32)
    sk_arr = np.full((4, 4), 5.0, dtype=np.float32)
    ms_arr = np.zeros((4, 4), dtype=np.float32)
    ms_arr[1, 1] = 1.0
    ms_arr[1, 2] = 1.0
    ch_arr = np.ones((4, 4), dtype=np.float32)
    stage_arr = np.zeros((4, 4), dtype=np.float32)
    stage_arr[1, 1] = 5.0  # weak dem proxy should be dropped
    stage_arr[1, 2] = 10.0  # bank-stage supported should remain allowed

    _write_raster(template, template_arr, nod)
    _write_raster(xs, xs_arr, nod)
    _write_raster(sk, sk_arr, nod)
    _write_raster(ms, ms_arr, nod)
    _write_raster(ch, ch_arr, nod)
    _write_raster(stage, stage_arr, 0.0)

    info = merge_hybrid_river_bed(xs, sk, ms, ch, out, template, nod, skeleton_stage_support=stage)
    with rasterio.open(out) as ds:
        merged = ds.read(1)
    assert merged[1, 1] == nod
    assert merged[1, 2] == 5.0
    assert info['skeleton_stage_supported_pixels'] == 1
    assert info['skeleton_stage_weak_or_missing_pixels'] >= 1
