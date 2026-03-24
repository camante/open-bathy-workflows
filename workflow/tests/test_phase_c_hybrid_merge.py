from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from hybrid_merge import merge_hybrid_river_bed


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
    assert merged[1, 1] == 10.0
    assert merged[1, 2] == 20.0
    # mainstem pixel with no XS should remain skeleton-backed, not unresolved
    assert merged[2, 1] == 5.0
    saved = json.loads(receipt.read_text())
    assert saved['xs_wins_mainstem_pixels'] == 2
    assert saved['skeleton_only_mainstem_pixels'] == 1
    assert saved['unresolved_mainstem_pixels'] == 0
    assert info['merge_rule'] == 'xs_dominates_mainstem_when_finite_else_skeleton'
