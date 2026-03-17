"""Tests for xs_infer_context.py — XSInferContext dataclass."""

import unittest
from types import SimpleNamespace
from pathlib import Path


class TestXSInferContext(unittest.TestCase):

    def setUp(self):
        from xs_infer_context import XSInferContext
        self.cls = XSInferContext

    def test_default_construction(self):
        ctx = self.cls()
        self.assertIsNone(ctx.xs_gpkg)
        self.assertIsNone(ctx.dem_path)
        self.assertEqual(ctx.xs_lines_layer, "xs_lines")
        self.assertEqual(ctx.idw_power, 2.0)
        self.assertEqual(ctx.nodata, -9999.0)

    def test_from_args(self):
        args = SimpleNamespace(
            soundings=None, soundings_depth_col="depth",
            soundings_elev_col=None, soundings_x_col="x",
            soundings_y_col="y", soundings_crs="EPSG:4326",
            rivers_layer="rivers_clip",
            drain_area_field=None, slope_field=None,
            manning_q_field=None, manning_dist_to_mouth_field=None,
            usgs_sites=None, usgs_start=None, usgs_end=None,
            usgs_cache_dir=None, width_stage_csv=None,
            raster_value_col="z_bed_pred_m", raster_uncert_col="uncert_m",
            continuous="walid", continuous_buffer_m=None,
            continuous_k=12, idw_power=2.0,
            aniso_along_scale_m=500.0, aniso_cross_scale_m=30.0,
            thalweg_weight=6.0, nodata=-9999.0,
            overlap_reducer="min",
            channel_mask_raster=None, channel_mask_inside_value=1,
            channel_mask_invert=False,
            max_query_dist_m=None, thalweg_only=False,
            thalweg_densify_step_m=None,
            out_accounting_json=None, out_meta_json=None,
        )
        ctx = self.cls.from_args(
            args,
            xs_gpkg=Path("/tmp/xs.gpkg"),
            out_gpkg=Path("/tmp/out.gpkg"),
            dem_path=Path("/tmp/dem.tif"),
        )
        self.assertEqual(ctx.xs_gpkg, Path("/tmp/xs.gpkg"))
        self.assertEqual(ctx.soundings_depth_col, "depth")
        self.assertEqual(ctx.soundings_crs, "EPSG:4326")

    def test_to_kwargs_has_all_keys(self):
        ctx = self.cls()
        kw = ctx.to_kwargs()
        # Should have all 46 parameters
        self.assertGreaterEqual(len(kw), 40)
        self.assertIn("xs_gpkg", kw)
        self.assertIn("nodata", kw)
        self.assertIn("thalweg_only", kw)
        self.assertIn("out_meta_json", kw)

    def test_to_kwargs_roundtrip(self):
        ctx = self.cls(
            xs_gpkg=Path("/a/b.gpkg"),
            idw_power=3.0,
            thalweg_only=True,
        )
        kw = ctx.to_kwargs()
        self.assertEqual(kw["xs_gpkg"], Path("/a/b.gpkg"))
        self.assertEqual(kw["idw_power"], 3.0)
        self.assertTrue(kw["thalweg_only"])


if __name__ == "__main__":
    unittest.main()
