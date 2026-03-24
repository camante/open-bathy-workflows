from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from river_mask_stage import _sync_mainstem_to_final_channel, validate_river_mask_stage_outputs


def _write_mask(path: Path, arr: np.ndarray, *, nodata=0, water_land=False):
    profile = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1, "dtype": "uint8", "crs": "EPSG:32619", "transform": from_origin(0, arr.shape[0], 1, 1), "nodata": nodata}
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr.astype("uint8"), 1)


def test_river_mask_stage_valid_contract(tmp_path: Path):
    ch = tmp_path / "channel.tif"
    ow = tmp_path / "open.tif"
    ms = tmp_path / "mainstem.tif"
    est = tmp_path / "estuary.tif"
    channel = np.zeros((6, 6), dtype=np.uint8)
    channel[1:5, 1:5] = 1
    channel[1:3, 1:3] = 0
    openw = np.zeros((6, 6), dtype=np.uint8)
    openw[0, :] = 1
    mainstem = np.zeros((6, 6), dtype=np.uint8)
    mainstem[3:5, 3:5] = 1
    estuary = np.zeros((6, 6), dtype=np.uint8)
    estuary[1:3, 1:3] = 1
    _write_mask(ch, channel)
    _write_mask(ow, openw)
    _write_mask(ms, mainstem)
    _write_mask(est, estuary, nodata=255)
    ocean = tmp_path / "ocean.tif"
    oceanw = np.zeros((6, 6), dtype=np.uint8)
    oceanw[0, :] = 0
    oceanland = np.ones((6, 6), dtype=np.uint8)
    oceanland[0, :] = 0
    _write_mask(ocean, oceanland, nodata=255)
    rec = validate_river_mask_stage_outputs(channel_mask_tif=ch, open_water_mask_tif=ow, mainstem_mask_tif=ms, estuary_clip_mask_tif=est, ocean_mask_tif=ocean)
    assert rec["mainstem_subset_of_channel"] is True
    assert rec["estuary_excluded_from_final_channel"] is True
    assert rec["open_water_ocean_connected_only"] is True


def test_river_mask_stage_rejects_open_water_overlap(tmp_path: Path):
    ch = tmp_path / "channel.tif"
    ow = tmp_path / "open.tif"
    ms = tmp_path / "mainstem.tif"
    est = tmp_path / "estuary.tif"
    channel = np.zeros((4, 4), dtype=np.uint8)
    channel[1:3, 1:3] = 1
    openw = np.zeros((4, 4), dtype=np.uint8)
    openw[1, 1] = 1
    mainstem = np.zeros((4, 4), dtype=np.uint8)
    mainstem[1, 1] = 1
    estuary = np.zeros((4, 4), dtype=np.uint8)
    _write_mask(ch, channel)
    _write_mask(ow, openw)
    _write_mask(ms, mainstem)
    _write_mask(est, estuary, nodata=255)
    try:
        validate_river_mask_stage_outputs(channel_mask_tif=ch, open_water_mask_tif=ow, mainstem_mask_tif=ms, estuary_clip_mask_tif=est)
    except RuntimeError as e:
        assert "open_water_mask overlaps river_channel_mask" in str(e)
    else:
        raise AssertionError("expected overlap failure")


def test_river_mask_stage_records_border_connected_non_ocean_open_water(tmp_path: Path):
    ch = tmp_path / "channel.tif"
    ow = tmp_path / "open.tif"
    ms = tmp_path / "mainstem.tif"
    est = tmp_path / "estuary.tif"
    ocean = tmp_path / "ocean.tif"
    channel = np.zeros((6, 6), dtype=np.uint8)
    channel[2:4, 2:4] = 1
    openw = np.zeros((6, 6), dtype=np.uint8)
    openw[-1, 1:5] = 1  # touches border but not ocean mask
    mainstem = np.zeros((6, 6), dtype=np.uint8)
    mainstem[2:4, 2:4] = 1
    estuary = np.zeros((6, 6), dtype=np.uint8)
    oceanland = np.ones((6, 6), dtype=np.uint8)
    oceanland[0, :] = 0  # ocean only on top border
    _write_mask(ch, channel)
    _write_mask(ow, openw)
    _write_mask(ms, mainstem)
    _write_mask(est, estuary, nodata=255)
    _write_mask(ocean, oceanland, nodata=255)
    rec = validate_river_mask_stage_outputs(channel_mask_tif=ch, open_water_mask_tif=ow, mainstem_mask_tif=ms, estuary_clip_mask_tif=est, ocean_mask_tif=ocean)
    assert rec["open_water_ocean_connected_only"] is False
    assert rec["open_water_outside_ocean_connected_full_pixels"] == 4


def test_river_mask_stage_accepts_ocean_mask_on_different_grid(tmp_path: Path):
    ch = tmp_path / "channel.tif"
    ow = tmp_path / "open.tif"
    ms = tmp_path / "mainstem.tif"
    est = tmp_path / "estuary.tif"
    ocean = tmp_path / "ocean_coarse.tif"

    channel = np.zeros((6, 6), dtype=np.uint8)
    channel[2:4, 2:4] = 1
    openw = np.zeros((6, 6), dtype=np.uint8)
    openw[0, :] = 1
    mainstem = np.zeros((6, 6), dtype=np.uint8)
    mainstem[2:4, 2:4] = 1
    estuary = np.zeros((6, 6), dtype=np.uint8)
    ocean_coarse = np.ones((3, 3), dtype=np.uint8)
    ocean_coarse[0, :] = 0

    _write_mask(ch, channel)
    _write_mask(ow, openw)
    _write_mask(ms, mainstem)
    _write_mask(est, estuary, nodata=255)
    with rasterio.open(
        ocean,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="uint8",
        crs="EPSG:32619",
        transform=from_origin(0, 6, 2, 2),
        nodata=255,
    ) as ds:
        ds.write(ocean_coarse, 1)

    rec = validate_river_mask_stage_outputs(
        channel_mask_tif=ch,
        open_water_mask_tif=ow,
        mainstem_mask_tif=ms,
        estuary_clip_mask_tif=est,
        ocean_mask_tif=ocean,
    )
    assert rec["open_water_ocean_connected_only"] is True


def test_mainstem_sync_uses_final_channel_not_only_estuary(tmp_path: Path):
    ch = tmp_path / "channel.tif"
    ms = tmp_path / "mainstem.tif"
    est = tmp_path / "estuary.tif"

    channel = np.zeros((6, 6), dtype=np.uint8)
    channel[1:5, 1:5] = 1
    channel[1, 1] = 0  # non-estuary channel trim, analogous to border erosion
    channel[4, 4] = 0  # final channel also excludes the estuary-clipped pixel

    mainstem = np.zeros((6, 6), dtype=np.uint8)
    mainstem[1:5, 1:5] = 1

    estuary = np.zeros((6, 6), dtype=np.uint8)
    estuary[4, 4] = 1

    _write_mask(ch, channel)
    _write_mask(ms, mainstem)
    _write_mask(est, estuary, nodata=255)

    rec = _sync_mainstem_to_final_channel(
        mainstem_mask_tif=ms,
        channel_mask_tif=ch,
        estuary_clip_mask_tif=est,
    )

    with rasterio.open(ms) as ds:
        synced = ds.read(1) == 1

    assert np.all(synced <= (channel == 1))
    assert rec["mainstem_pixels_removed_by_estuary"] == 1
    assert rec["mainstem_pixels_removed_by_non_estuary_channel_trim"] == 1
    assert rec["mainstem_pixels_removed_total"] == 2


def test_preclip_mainstem_sync_repairs_broad_corridor_export(tmp_path: Path):
    ch = tmp_path / "channel.tif"
    ms = tmp_path / "mainstem.tif"

    channel = np.zeros((8, 8), dtype=np.uint8)
    channel[2:6, 2:6] = 1

    # Simulate a stale river_domain_mask export that wrote the broad corridor
    # instead of a channel-confined mainstem output.
    mainstem = np.zeros((8, 8), dtype=np.uint8)
    mainstem[1:7, 1:7] = 1

    _write_mask(ch, channel)
    _write_mask(ms, mainstem)

    rec = _sync_mainstem_to_final_channel(
        mainstem_mask_tif=ms,
        channel_mask_tif=ch,
        estuary_clip_mask_tif=None,
    )

    with rasterio.open(ms) as ds:
        synced = ds.read(1) == 1

    assert np.all(synced <= (channel == 1))
    assert rec["mainstem_pixels_removed_total"] == int(np.count_nonzero((mainstem == 1) & (channel == 0)))
    assert rec["mainstem_pixels_removed_by_estuary"] == 0
