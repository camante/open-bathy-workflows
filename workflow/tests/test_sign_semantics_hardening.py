from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from alignment import TiePoints, residuals_against_points
from bathy_main import _is_depth_raster_like


def _write_tif(path: Path, arr: np.ndarray, tags: dict | None = None):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "transform": from_origin(0, 2, 1, 1),
        "crs": "EPSG:4326",
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr.astype("float32"), 1)
        if tags:
            ds.update_tags(**tags)


def test_is_depth_raster_like_accepts_absolute_elevation_tag(tmp_path: Path):
    tif = tmp_path / "elev.tif"
    _write_tif(tif, np.array([[22.0, 23.0], [24.0, 25.0]], dtype=np.float32), tags={"VALUE_TYPE": "elevation", "VERTICAL_DATUM": "NAVD88"})
    ok, reason = _is_depth_raster_like(tif, expect_negative=True)
    assert ok is True
    assert "absolute_elevation" in reason


def test_alignment_residuals_honor_explicit_positive_down_points(tmp_path: Path):
    tif = tmp_path / "depth.tif"
    _write_tif(tif, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), tags={"VALUE_TYPE": "depth", "DEPTH_SIGN": "positive_down"})
    pts = TiePoints(
        lon=np.array([0.5, 1.5]),
        lat=np.array([1.5, 0.5]),
        depth_m=np.array([1.0, 4.0]),
        source=None,
        depth_positive_down=True,
    )
    out = residuals_against_points(str(tif), pts)
    assert out["status"] == "ok"
    assert np.allclose(out["residuals_m"], np.array([0.0, 0.0]))


def test_raster_value_semantics_accepts_legacy_bed_elevation_value_type():
    from sign_semantics import raster_value_semantics
    assert raster_value_semantics({"VALUE_TYPE": "bed_elevation"}) == "absolute_elevation"


def test_apply_sdb_metadata_tags_bed_as_absolute_elevation(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from sdb_main import apply_sdb_metadata

    path = tmp_path / "sdb.tif"
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 0, 1, 1),
        "nodata": -9999.0,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(np.array([[-1.0, 0.5], [2.0, -9999.0]], dtype='float32'), 1)

    apply_sdb_metadata(path)
    with rasterio.open(path) as ds:
        tags = ds.tags()
    assert tags.get('VALUE_TYPE') == 'elevation'
    assert tags.get('ELEVATION_ROLE') == 'bed_elevation'
    assert tags.get('SIGN_CONVENTION') == 'relative_to_datum'
