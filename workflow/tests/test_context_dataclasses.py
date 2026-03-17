"""Tests for xs_infer_context.py and s2_composite_context.py dataclasses."""

import unittest
from pathlib import Path
from types import SimpleNamespace


class TestXSInferContext(unittest.TestCase):

    def setUp(self):
        from xs_infer_context import XSInferContext
        self.cls = XSInferContext

    def test_default_construction(self):
        ctx = self.cls()
        self.assertIsNone(ctx.xs_gpkg)
        self.assertIsNone(ctx.dem_path)
        self.assertEqual(ctx.xs_lines_layer, "xs_lines")
        self.assertEqual(ctx.continuous, "walid")
        self.assertEqual(ctx.nodata, -9999.0)
        self.assertFalse(ctx.thalweg_only)

    def test_to_kwargs(self):
        ctx = self.cls(xs_gpkg=Path("/tmp/xs.gpkg"), idw_power=3.0)
        kw = ctx.to_kwargs()
        self.assertEqual(kw["xs_gpkg"], Path("/tmp/xs.gpkg"))
        self.assertEqual(kw["idw_power"], 3.0)
        # Should have all fields
        self.assertIn("soundings_path", kw)
        self.assertIn("channel_mask_raster", kw)
        self.assertIn("cfg", kw)

    def test_from_args(self):
        args = SimpleNamespace(
            soundings="/tmp/s.csv",
            soundings_depth_col="depth", soundings_elev_col=None,
            soundings_x_col="x", soundings_y_col="y",
            soundings_crs="EPSG:4326",
            rivers_layer="rivers_clip",
            drain_area_field=None, slope_field=None,
            manning_q_field=None, manning_dist_to_mouth_field=None,
            usgs_sites=None, usgs_start=None, usgs_end=None,
            usgs_cache_dir=None, width_stage_csv=None,
            raster_value_col="z_bed_pred_m", raster_uncert_col="uncert_m",
            continuous="walid", continuous_buffer_m=None,
            continuous_k=12, idw_power=2.5, 
            aniso_along_scale_m=500.0, aniso_cross_scale_m=30.0,
            thalweg_weight=8.0, nodata=-9999.0, overlap_reducer="min",
            channel_mask_raster=None, channel_mask_inside_value=1,
            channel_mask_invert=False, max_query_dist_m=None,
            thalweg_only=False, thalweg_densify_step_m=None,
            out_accounting_json=None, out_meta_json=None,
        )
        ctx = self.cls.from_args(
            args, xs_gpkg=Path("/tmp/xs.gpkg"), out_gpkg=Path("/tmp/out.gpkg"),
            dem_path=Path("/tmp/dem.tif"),
        )
        self.assertEqual(ctx.xs_gpkg, Path("/tmp/xs.gpkg"))
        self.assertEqual(ctx.idw_power, 2.5)
        self.assertEqual(ctx.thalweg_weight, 8.0)

    def test_incremental_population(self):
        ctx = self.cls()
        ctx.xs_gpkg = Path("/updated.gpkg")
        self.assertEqual(ctx.xs_gpkg, Path("/updated.gpkg"))


class TestS2CompositeContext(unittest.TestCase):

    def setUp(self):
        from s2_composite_context import S2CompositeContext
        self.cls = S2CompositeContext

    def test_default_construction(self):
        ctx = self.cls()
        self.assertIsNone(ctx.out_dir)
        self.assertEqual(ctx.max_cloud, 20.0)
        self.assertTrue(ctx.harmonize)
        self.assertFalse(ctx.single_best_date)
        self.assertEqual(ctx.edge_weight_power, 2.0)

    def test_from_args(self):
        args = SimpleNamespace(
            start="2024-01-01", end="2024-06-30",
            cloud=30.0, s2_scene_limit=15,
            preferred_months=[1, 2, 3],
            min_scene_valid_frac=0.85,
            stac_max_items=3000, stac_page_limit=None,
            stac_chunk_months=2,
            download_workers=4, scl_dilate=2,
            harmonize=True,
            deepwater_nir_max=0.04, deepwater_bright_max=0.20,
            mask_bright_pixels=0.30,
            allow_bright_shallow_pixels=True,
            bright_shallow_nir_max=0.05,
            apply_gl_turbidity_reject=True,
            gl_nir_max=0.08, gl_nir_green_ratio_max=0.40, gl_red_max=0.06,
            edge_weight_power=3.0, edge_weight_min=0.1,
            temporal_median_k=5, single_best_date=False,
            date_qc_enable=True,
            clean_cache=False, cache_strict=False, cache_code_strict=False,
        )
        ctx = self.cls.from_args(
            args, out_dir=Path("/tmp/s2"), bbox_wesn=(-78, -77, 25, 26),
            cache_dir=Path("/tmp/cache"),
        )
        self.assertEqual(ctx.out_dir, Path("/tmp/s2"))
        self.assertEqual(ctx.bbox_wesn, (-78, -77, 25, 26))
        self.assertEqual(ctx.max_cloud, 30.0)
        self.assertEqual(ctx.scene_limit, 15)
        self.assertEqual(ctx.preferred_months, [1, 2, 3])
        self.assertTrue(ctx.apply_gl_turbidity_reject)
        self.assertEqual(ctx.edge_weight_power, 3.0)
        self.assertEqual(ctx.temporal_median_k, 5)

    def test_to_kwargs(self):
        ctx = self.cls(max_cloud=15.0, harmonize=False)
        kw = ctx.to_kwargs()
        self.assertEqual(kw["max_cloud"], 15.0)
        self.assertFalse(kw["harmonize"])
        self.assertIn("bbox_wesn", kw)
        self.assertIn("coastline_mask_path", kw)

    def test_to_kwargs_has_all_fields(self):
        ctx = self.cls()
        kw = ctx.to_kwargs()
        # Spot-check key fields exist
        for key in ["out_dir", "bbox_wesn", "start_date", "end_date",
                     "max_cloud", "stac_url", "harmonize", "edge_weight_power",
                     "single_best_date", "date_qc_enable"]:
            self.assertIn(key, kw)

    def test_mutable_defaults_isolated(self):
        c1 = self.cls()
        c2 = self.cls()
        self.assertIsNone(c1.preferred_months)
        self.assertIsNone(c2.preferred_months)


if __name__ == "__main__":
    unittest.main()
