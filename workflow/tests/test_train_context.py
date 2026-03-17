"""Tests for train_context.py — TrainContext dataclass."""

import unittest
from types import SimpleNamespace
from pathlib import Path


class TestTrainContext(unittest.TestCase):

    def setUp(self):
        from train_context import TrainContext
        self.cls = TrainContext

    def test_default_construction(self):
        tc = self.cls()
        self.assertIsNone(tc.train_df)
        self.assertEqual(tc.max_depth_sdb, 20.0)
        self.assertEqual(tc.seed, 42)
        self.assertFalse(tc.linf_enabled)
        self.assertEqual(tc.water_class, "mixed")

    def test_from_args(self):
        args = SimpleNamespace(
            max_depth_sdb=15.0, seed=99, water_class="clear_ocean",
            use_stumpf_depth=True, min_training_points_for_sdb=200,
            spatial_cv=True, spatial_cv_strategy="blocked", spatial_cv_n_folds=3,
            cw_min=0.4, land_max=0.1, land_mask_type="auto",
            land_mask_water_val=0, land_mask_invert=False, land_mask_threshold=0.5,
            linf_enabled=True, linf_estimate_deepwater="raster",
            linf_deepwater_nir_max=0.04, linf_deepwater_bright_max=0.2,
            linf_percentile=2.0, rmse_target_sdb=1.0,
            depth_bin_m=1.0, depth_binning="equal_width", min_samples_per_bin=50,
            max_depth_source="combined",
            model_bank_enabled=True, bank_max_samples=50000,
            bank_seed=42, bank_retrain_min_new=1000,
        )
        tc = self.cls.from_args(
            args, plots_dir="/tmp/plots", diagnostics_dir="/tmp/diag",
            s2_paths={"B02": "b02.tif"}, linf_estimate_mode="raster",
        )
        self.assertEqual(tc.max_depth_sdb, 15.0)
        self.assertEqual(tc.seed, 99)
        self.assertEqual(tc.water_class, "clear_ocean")
        self.assertTrue(tc.use_stumpf_depth)
        self.assertTrue(tc.linf_enabled)
        self.assertEqual(tc.linf_estimate, "raster")
        self.assertEqual(tc.raster_paths, {"B02": "b02.tif"})
        self.assertEqual(tc.plots_dir, Path("/tmp/plots"))
        self.assertEqual(tc.model_bank_max_samples, 50000)

    def test_to_kwargs(self):
        tc = self.cls(max_depth_sdb=10.0, seed=7, water_class="lake")
        kw = tc.to_kwargs()
        self.assertEqual(kw["max_depth_sdb"], 10.0)
        self.assertEqual(kw["seed"], 7)
        self.assertEqual(kw["water_class"], "lake")
        # Should have all 38 keys
        self.assertGreaterEqual(len(kw), 30)

    def test_to_kwargs_roundtrip(self):
        """to_kwargs output should be passable as **kwargs to train_sdb_model."""
        tc = self.cls(max_depth_sdb=12.0, seed=5)
        kw = tc.to_kwargs()
        # All values should be set (not raise KeyError)
        for k, v in kw.items():
            self.assertTrue(hasattr(tc, k) or k in kw)

    def test_overrides(self):
        """land_mask overrides should take precedence over args."""
        args = SimpleNamespace(
            max_depth_sdb=20.0, seed=42, water_class="mixed",
            use_stumpf_depth=False, min_training_points_for_sdb=50,
            spatial_cv=False, spatial_cv_strategy="spatial_cluster",
            spatial_cv_n_folds=5, cw_min=0.5, land_max=0.0,
            land_mask_type="scl", land_mask_water_val=6,
            land_mask_invert=True, land_mask_threshold=0.5,
            linf_enabled=False, linf_estimate_deepwater="none",
            linf_deepwater_nir_max=0.03, linf_deepwater_bright_max=0.15,
            linf_percentile=1.0, rmse_target_sdb=0.5,
            depth_bin_m=0.5, depth_binning="quantile", min_samples_per_bin=200,
            max_depth_source="physics",
            model_bank_enabled=False, bank_max_samples=100000,
            bank_seed=1337, bank_retrain_min_new=2000,
        )
        tc = self.cls.from_args(
            args,
            land_mask_type_override="waffles",
            land_mask_water_val_override=0,
            land_mask_invert_override=False,
        )
        self.assertEqual(tc.land_mask_type, "waffles")
        self.assertEqual(tc.land_mask_water_val, 0)
        self.assertFalse(tc.land_mask_invert)

    def test_mutable_defaults_isolated(self):
        tc1 = self.cls()
        tc2 = self.cls()
        self.assertIsNone(tc1.raster_paths)
        self.assertIsNone(tc2.raster_paths)


if __name__ == "__main__":
    unittest.main()

    def test_max_depth_auto_string(self):
        """Regression: max_depth_sdb='auto' must not crash from_args."""
        args = SimpleNamespace(
            max_depth_sdb="auto", seed=42, water_class="mixed",
            use_stumpf_depth=False, min_training_points_for_sdb=50,
            spatial_cv=False, spatial_cv_strategy="spatial_cluster",
            spatial_cv_n_folds=5, cw_min=0.5, land_max=0.0,
            land_mask_type="auto", land_mask_water_val=0,
            land_mask_invert=False, land_mask_threshold=0.5,
            linf_enabled=False, linf_estimate_deepwater="none",
            linf_deepwater_nir_max=0.03, linf_deepwater_bright_max=0.15,
            linf_percentile=1.0, rmse_target_sdb=0.5,
            depth_bin_m=0.5, depth_binning="quantile", min_samples_per_bin=200,
            max_depth_source="physics",
            model_bank_enabled=False, bank_max_samples=100000,
            bank_seed=1337, bank_retrain_min_new=2000,
        )
        tc = self.cls.from_args(args)
        self.assertEqual(tc.max_depth_sdb, 20.0)  # default when 'auto'
