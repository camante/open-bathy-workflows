from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile

import bathy_main
from geo.raster_ops import _clip_raster_to_mask_reproject


def _write_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)


def test_write_river_guidance_artifacts_uses_point_geometries(monkeypatch, tmp_path: Path) -> None:
    depth = np.full((10, 10), -5.0, dtype=np.float32)
    depth_tif = tmp_path / "river" / "depth.tif"
    bed_tif = tmp_path / "river" / "bed.tif"
    _write_tif(depth_tif, depth)
    _write_tif(bed_tif, depth)

    support = pd.DataFrame(
        {
            "lon": [-70.955, -70.925],
            "lat": [42.955, 42.925],
            "depth_m": [-3.0, -4.0],
            "source": ["authoritative_base", "authoritative_base"],
        }
    )
    monkeypatch.setattr(bathy_main, "_load_river_authoritative_support_points", lambda cfg: support)

    seen = {"point_geoms": 0}

    def fake_rasterize(shapes, out_shape, transform, fill, dtype, **kwargs):
        arr = np.full(out_shape, fill, dtype=np.float32 if dtype == "float32" else np.uint8)
        seq = list(shapes)
        for geom, value in seq:
            assert isinstance(geom, dict)
            assert geom.get("type") == "Point"
            seen["point_geoms"] += 1
            arr[0, 0] = value
        return arr.astype(np.float32 if dtype == "float32" else np.uint8)

    features_mod = types.ModuleType("rasterio.features")
    features_mod.rasterize = fake_rasterize
    monkeypatch.setitem(sys.modules, "rasterio.features", features_mod)

    import rasterio
    rasterio.errors = types.SimpleNamespace(RasterioError=RuntimeError)

    cfg = SimpleNamespace(
        river_nodata=-9999.0,
        river_trusted_halo_m=60.0,
        river_guidance_near_bank_m=25.0,
        river_guidance_decay_m=100.0,
        river_guidance_min_weight=0.1,
        river_guidance_max_weight=1.0,
        river_guidance_support_radius_m=100.0,
        river_guidance_exact_support=True,
        derived_cache_root=tmp_path / "derived_cache",
        aoi="-71/-70.75/42.75/43",
        river_network_halo_km=0.0,
    )

    artifacts = bathy_main._write_river_guidance_artifacts(
        cfg,
        bed_tif=bed_tif,
        depth_tif=depth_tif,
        channel_mask_tif=None,
        river_dir=tmp_path / "river",
        report={},
    )

    assert seen["point_geoms"] >= 2
    assert Path(artifacts["authoritative_support"]).exists()


def test_clip_raster_to_mask_reproject_skips_rewrite_when_mask_is_noop(monkeypatch, tmp_path: Path) -> None:
    raster_path = tmp_path / "raster.tif"
    mask_path = tmp_path / "mask.tif"
    data = np.full((5, 5), -2.0, dtype=np.float32)
    mask = np.zeros((5, 5), dtype=np.uint8)  # inside_value=0 keeps all pixels
    _write_tif(raster_path, data)
    _write_tif(mask_path, mask)

    warp_mod = types.ModuleType("rasterio.warp")
    warp_mod.Resampling = type("Resampling", (), {"nearest": 0})()

    def fake_reproject(source, destination, **kwargs):
        destination[...] = np.asarray(source, dtype=destination.dtype)

    warp_mod.reproject = fake_reproject
    monkeypatch.setitem(sys.modules, "rasterio.warp", warp_mod)

    import rasterio

    real_open = rasterio.open

    def guarded_open(*args, **kwargs):
        mode = args[1] if len(args) >= 2 else kwargs.get("mode", "r")
        if mode == "w":
            raise AssertionError("unexpected rewrite for no-op clip")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(rasterio, "open", guarded_open)
    assert _clip_raster_to_mask_reproject(raster_path, mask_path, inside_value=0, nodata=-9999.0) is True
