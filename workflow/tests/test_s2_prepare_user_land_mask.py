import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path = [p for p in sys.path if '/workflow/tests' not in p and not p.endswith('/workflow')]
sys.path.append(str(Path(__file__).resolve().parents[1]))

from s2_optics import prepare_user_land_mask


def _write(path: Path, arr: np.ndarray, *, transform, crs='EPSG:32619', nodata=None):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=str(arr.dtype),
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as ds:
        ds.write(arr, 1)


def test_prepare_user_land_mask_aligns_to_reference_grid(tmp_path: Path):
    src = tmp_path / 'mask.tif'
    ref = tmp_path / 'ref.tif'
    out = tmp_path / 'aligned.tif'

    _write(src, np.array([[0, 1], [1, 0]], dtype=np.uint8), transform=from_origin(0, 4, 2, 2), nodata=255)
    _write(ref, np.zeros((4, 4), dtype=np.uint16), transform=from_origin(0, 4, 1, 1), nodata=0)

    prepare_user_land_mask(str(src), str(ref), str(out))

    with rasterio.open(out) as ds:
        arr = ds.read(1)
        assert arr.shape == (4, 4)
        assert arr[0, 0] == 0
        assert arr[0, 3] == 1
        assert arr[3, 0] == 1
        assert arr[3, 3] == 0
