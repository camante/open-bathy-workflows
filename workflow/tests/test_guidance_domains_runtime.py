import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

from guidance_domains import _write_river_guidance_domain_mask


def _write_mask(path: Path, arr: np.ndarray):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32619",
        "transform": from_origin(0, arr.shape[0], 10, 10),
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint8), 1)


def test_transition_exclusion_applies_even_with_zero_bank_margin(tmp_path):
    channel = np.zeros((8, 8), dtype=np.uint8)
    channel[2:6, 1:7] = 1
    transition = np.zeros_like(channel)
    transition[2:6, 5:7] = 1
    ch = tmp_path / "channel.tif"
    tr = tmp_path / "transition.tif"
    out = tmp_path / "river_guidance.tif"
    _write_mask(ch, channel)
    _write_mask(tr, transition)

    diag = _write_river_guidance_domain_mask(
        channel_mask_path=ch,
        out_path=out,
        bank_margin_m=0.0,
        logger=__import__("logging").getLogger("test"),
        estuary_transition_mask_path=tr,
    )
    with rasterio.open(out) as ds:
        arr = ds.read(1)
    assert diag["transition_pixels_excluded"] > 0
    assert np.count_nonzero(arr[:, 5:7]) == 0


def test_positive_bank_margin_erodes_channel_edges(tmp_path):
    channel = np.zeros((9, 9), dtype=np.uint8)
    channel[1:8, 1:8] = 1
    ch = tmp_path / "channel.tif"
    out = tmp_path / "river_guidance.tif"
    _write_mask(ch, channel)

    diag = _write_river_guidance_domain_mask(
        channel_mask_path=ch,
        out_path=out,
        bank_margin_m=3.0,
        logger=__import__("logging").getLogger("test"),
    )
    with rasterio.open(out) as ds:
        arr = ds.read(1)
    assert diag["output_pixels"] < diag["input_pixels"]
    assert arr[1, 1] == 0
    assert arr[4, 4] == 1
