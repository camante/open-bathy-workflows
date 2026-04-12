import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestReplaceWithSymlinkOrCopy(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._replace_with_symlink_or_copy

    def test_same_path_is_noop_and_does_not_create_symlink_loop(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DEM_enhanced.tif"
            p.write_bytes(b"real-raster")
            self.fn(p, p)
            self.assertTrue(p.exists())
            self.assertFalse(p.is_symlink())
            self.assertEqual(p.read_bytes(), b"real-raster")


    def test_same_crs_warp_short_circuit_does_not_mutate_tags(self):
        import rasterio
        from rasterio.transform import from_origin
        from geo.raster_ops import warp_raster_to_srs

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "same_crs.tif"
            with rasterio.open(
                p,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(0.0, 2.0, 1.0, 1.0),
                nodata=-9999.0,
            ) as ds:
                ds.write(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 1)
                ds.update_tags(VALUE_TYPE="elevation")
            out = warp_raster_to_srs(p, Path(td) / "unused.tif", "EPSG:4326")
            self.assertEqual(out, p)
            with rasterio.open(p) as ds:
                tags = ds.tags()
            self.assertEqual(tags.get("VALUE_TYPE"), "elevation")
            self.assertIsNone(tags.get("DEPTH_REFERENCE"))


if __name__ == "__main__":
    unittest.main()
