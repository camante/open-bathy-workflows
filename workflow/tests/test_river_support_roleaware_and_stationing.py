from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from authoritative_guidance import prepare_authoritative_river_soundings_points
from terrain_interpolator import _nearest_surface_within_domain


def test_prepare_authoritative_river_soundings_keeps_positive_elevations(tmp_path: Path):
    arr = np.array([[1.5, 2.0], [-0.5, 3.0]], dtype=np.float32)
    src = tmp_path / "auth.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:32619",
        transform=from_origin(0, 2, 1, 1),
        nodata=-9999.0,
    ) as ds:
        ds.write(arr, 1)
        ds.update_tags(VALUE_TYPE="elevation")
    out_csv = tmp_path / "river.csv"
    info = prepare_authoritative_river_soundings_points(src, out_csv, out_crs="EPSG:32619")
    text = out_csv.read_text(encoding="utf-8")
    assert info["negative_only"] is False
    assert "1.500000" in text
    assert "3.000000" in text


def test_stationing_aware_nearest_prefers_along_channel_source():
    shape = (3, 5)
    domain = np.ones(shape, dtype=bool)
    centerline = np.zeros(shape, dtype=bool)
    centerline[1, :] = True
    station = np.full(shape, np.nan, dtype=np.float32)
    station[1, :] = np.arange(5, dtype=np.float32) * 100.0
    valid = np.zeros(shape, dtype=bool)
    values = np.full(shape, np.nan, dtype=np.float32)
    # Same-thalweg upstream point
    valid[1, 0] = True
    values[1, 0] = 10.0
    # Opposite-bank nearby point across channel near query column 2
    valid[0, 2] = True
    values[0, 2] = 99.0
    out = _nearest_surface_within_domain(
        valid,
        values,
        domain,
        centerline_mask=centerline,
        stationing_raster=station,
        along_scale_m=500.0,
        cross_scale_m=30.0,
        pixel_size_m=20.0,
    )
    # Query on the centerline near column 2 should prefer along-channel same-thalweg source
    assert np.isclose(out[1, 2], 10.0)
