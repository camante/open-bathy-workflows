import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class _FakeSupportGDF:
    def to_file(self, path, driver="GPKG"):
        Path(path).write_text("fake-gpkg", encoding="utf-8")


class TestAuthoritativeBaseCacheReuse(unittest.TestCase):

    def test_identical_aoi_reuses_cached_authoritative_base(self):
        # Extend the lightweight geo stubs installed by tests/conftest.py so the
        # cudem_authoritative module can import under pytest without pulling real
        # geospatial dependencies.
        rio = sys.modules.get("rasterio")
        self.assertIsNotNone(rio)
        if not hasattr(rio, "band"):
            rio.band = lambda src, idx: src.read(idx)
        if not hasattr(rio, "Affine"):
            rio.Affine = tuple
        if "rasterio.features" not in sys.modules:
            feat = types.ModuleType("rasterio.features")
            feat.rasterize = lambda *a, **kw: np.zeros(kw.get("out_shape", (2, 2)), dtype=np.uint8)
            sys.modules["rasterio.features"] = feat
        if "rasterio.merge" not in sys.modules:
            m = types.ModuleType("rasterio.merge")
            m.merge = lambda *a, **kw: (np.zeros((1, 2, 2), dtype=np.float32), None)
            sys.modules["rasterio.merge"] = m
        if "rasterio.warp" not in sys.modules:
            w = types.ModuleType("rasterio.warp")
            w.transform_bounds = lambda *a, **kw: (-71.0, 42.75, -70.75, 43.0)
            sys.modules["rasterio.warp"] = w
        shapely_mod = sys.modules.get("shapely")
        if shapely_mod is None:
            shapely_mod = types.ModuleType("shapely")
            sys.modules["shapely"] = shapely_mod
        shp = sys.modules.get("shapely.geometry")
        if shp is None:
            shp = types.ModuleType("shapely.geometry")
            sys.modules["shapely.geometry"] = shp
        if not hasattr(shp, "box"):
            shp.box = lambda *a, **kw: None
        shapely_mod.geometry = shp

        if "geopandas" not in sys.modules:
            gpd = types.ModuleType("geopandas")
            gpd.read_file = lambda *a, **kw: object()
            sys.modules["geopandas"] = gpd

        import cudem_authoritative
        importlib.reload(cudem_authoritative)

        calls = {"download_tiles": 0}

        def fake_download_file(url, dest, **kwargs):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            return dest

        def fake_extract_zip(zip_path, extract_dir, **kwargs):
            extract_dir.mkdir(parents=True, exist_ok=True)
            return extract_dir

        def fake_select_tiles(tile_index_gdf, aoi_bounds_4269, url_field):
            return [cudem_authoritative.TileRecord(tile_name="tile_a.tif", tile_url="https://example.com/tile_a.tif", tile_geom=None, meta_path=Path("meta_a.gpkg"))]

        def fake_attach_metadata_paths(records, meta_root, missing_meta_policy, logger):
            return records

        def fake_download_tiles(records, tile_dir, **kwargs):
            calls["download_tiles"] += 1
            tile_dir.mkdir(parents=True, exist_ok=True)
            out = []
            for rec in records:
                p = tile_dir / rec.tile_name
                p.write_bytes(b"tile")
                out.append(p)
            return out

        def fake_build_dem_mosaic(tile_paths, aoi):
            mosaic = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
            transform = None
            profile = {
                "driver": "GTiff",
                "height": 2,
                "width": 2,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:4269",
                "transform": transform,
                "nodata": -9999.0,
            }
            return mosaic, transform, profile

        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td) / "cache"
            with patch.object(cudem_authoritative, "download_file", side_effect=fake_download_file), \
                 patch.object(cudem_authoritative, "extract_zip", side_effect=fake_extract_zip), \
                 patch.object(cudem_authoritative, "discover_vector_file", return_value=Path("dummy.shp")), \
                 patch.object(cudem_authoritative, "read_tile_index", return_value=object()), \
                 patch.object(cudem_authoritative, "detect_url_field", return_value="URL"), \
                 patch.object(cudem_authoritative, "select_tiles", side_effect=fake_select_tiles), \
                 patch.object(cudem_authoritative, "attach_metadata_paths", side_effect=fake_attach_metadata_paths), \
                 patch.object(cudem_authoritative, "write_tile_manifest", return_value=None), \
                 patch.object(cudem_authoritative, "download_tiles", side_effect=fake_download_tiles), \
                 patch.object(cudem_authoritative, "build_dem_mosaic", side_effect=fake_build_dem_mosaic), \
                 patch.object(cudem_authoritative, "collect_support_geometries", return_value=_FakeSupportGDF()), \
                 patch.object(cudem_authoritative, "rasterize_support_mask", return_value=np.array([[1, 0], [0, 1]], dtype=np.uint8)), \
                 patch.object(cudem_authoritative, "write_raster", side_effect=lambda path, array, profile: Path(path).write_bytes(b"raster")):
                first = cudem_authoritative.materialize_authoritative_base_for_aoi(
                    aoi="-71/-70.75/42.75/43",
                    cache_root=cache_root,
                )
                second = cudem_authoritative.materialize_authoritative_base_for_aoi(
                    aoi="-71/-70.75/42.75/43",
                    cache_root=cache_root,
                )

            self.assertFalse(bool(first["cache_hit"]))
            self.assertTrue(bool(second["cache_hit"]))
            self.assertEqual(Path(first["authoritative_base"]), Path(second["authoritative_base"]))
            self.assertEqual(calls["download_tiles"], 1)
            self.assertTrue(Path(first["authoritative_base"]).exists())
            self.assertTrue(Path(first["shared_tile_cache"]).exists())


if __name__ == "__main__":
    unittest.main()
