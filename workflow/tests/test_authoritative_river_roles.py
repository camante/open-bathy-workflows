from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from authoritative_river_roles import build_authoritative_river_role_arrays


def _write_mask(path: Path, arr: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:32619",
        transform=from_origin(0, arr.shape[0], 1, 1),
        nodata=0,
    ) as ds:
        ds.write(arr.astype("uint8"), 1)


def test_build_authoritative_river_role_arrays_separates_bank_inner_and_ambiguous(tmp_path: Path):
    channel = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ], dtype=np.uint8)
    guidance = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    estuary = np.zeros_like(channel, dtype=np.uint8)
    estuary[0, 1] = 1
    ch = tmp_path / "channel.tif"
    gd = tmp_path / "guidance.tif"
    est = tmp_path / "estuary.tif"
    _write_mask(ch, channel)
    _write_mask(gd, guidance)
    _write_mask(est, estuary)
    profile = {"height": 5, "width": 5, "transform": from_origin(0, 5, 1, 1), "crs": rasterio.crs.CRS.from_epsg(32619)}
    arrays = build_authoritative_river_role_arrays(
        template_profile=profile,
        river_channel_mask_path=ch,
        river_guidance_domain_mask_path=gd,
        estuary_clip_mask_path=est,
        bank_margin_m=0.5,
    )
    role = arrays["role"]
    assert role[2, 2] == "authoritative_bed_core"
    assert role[1, 1] == "authoritative_bank_margin"
    assert role[0, 1] == "authoritative_overbank_or_ambiguous"
    assert role[0, 0] == "authoritative_overbank_or_ambiguous"
