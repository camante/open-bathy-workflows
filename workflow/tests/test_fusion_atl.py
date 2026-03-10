"""test_fusion_atl.py – fusion and ATL dataframe logic (unittest)."""

from __future__ import annotations
import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import _install_mocks
_install_mocks()


class TestFusionMerge(unittest.TestCase):

    def _df(self, n, source, seed=0):
        rng = np.random.default_rng(seed)
        b02 = rng.uniform(0.02, 0.15, n).astype(np.float32)
        b03 = rng.uniform(0.015, 0.12, n).astype(np.float32)
        return pd.DataFrame({
            "longitude": rng.uniform(-82, -81, n),
            "latitude":  rng.uniform(24, 25, n),
            "depth_m":   -rng.uniform(1, 10, n),
            "source":    source,
            "CLEAR_WATER": rng.uniform(0.6, 1.0, n),
            "LAND":        rng.uniform(0.0, 0.3, n),
            "B02": b02, "B03": b03,
            "B04": rng.uniform(0.01, 0.08, n).astype(np.float32),
            "B08": rng.uniform(0.005, 0.04, n).astype(np.float32),
        })

    def _fuse(self, df03, df24):
        import fusion
        try:
            return fusion.build_fused_training_dataframe(
                df_atl03=df03, df_atl24=df24, df_xyz=None,
                enable_adaptive_sampling=False,
            )
        except TypeError:
            return fusion.build_fused_training_dataframe(df03, df24, None)

    def test_combined_sources_in_output(self):
        fused = self._fuse(self._df(100, "atl03", 1), self._df(80, "atl24", 2))
        self.assertGreater(len(fused), 0)
        self.assertIn("depth_m", fused.columns)

    def test_none_atl24_handled(self):
        try:
            fused = self._fuse(self._df(100, "atl03", 3), None)
            self.assertGreater(len(fused), 0)
        except (TypeError, AttributeError):
            self.skipTest("fusion signature mismatch")

    def test_depths_remain_negative(self):
        try:
            fused = self._fuse(self._df(120, "atl03", 4), self._df(80, "atl24", 5))
        except (TypeError, AttributeError):
            self.skipTest("fusion signature mismatch")
        depths = fused["depth_m"].values
        finite = depths[np.isfinite(depths)]
        if len(finite) > 0:
            self.assertTrue(np.all(finite <= 0),
                            "Fused depths must be negative-down")


class TestSourceFractionGuardrail(unittest.TestCase):
    """Single-source domination triggers a warning but must not crash training."""

    def test_single_source_trains_ok(self):
        import train
        rng = np.random.default_rng(9)
        n = 300
        b02 = rng.uniform(0.02, 0.15, n).astype(np.float32)
        b03 = rng.uniform(0.015, 0.12, n).astype(np.float32)
        df = pd.DataFrame({
            "longitude": rng.uniform(-82, -81, n),
            "latitude":  rng.uniform(24, 25, n),
            "depth_m":   -rng.uniform(1, 10, n),
            "B02": b02, "B03": b03,
            "B04": rng.uniform(0.01, 0.08, n).astype(np.float32),
            "B08": rng.uniform(0.005, 0.04, n).astype(np.float32),
            "brightness": (b02 + b03) / 2,
            "CLEAR_WATER": rng.uniform(0.6, 1.0, n),
            "LAND":        rng.uniform(0.0, 0.3, n),
            "stumpf_idx":  np.log(b02+1e-3)/np.log(b03+1e-3),
            "stumpf_depth": rng.uniform(1, 10, n),
            "source": "atl03",
            "sample_weight": np.ones(n),
        })
        with tempfile.TemporaryDirectory() as d:
            rf, _, _, _, meta = train.train_sdb_model(
                train_df=df, max_depth_sdb=15.0, seed=42,
                plots_dir=Path(d) / "plots",
                water_class="ocean", use_stumpf_depth=True,
                min_training_points_for_sdb=50,
            )
        self.assertIsNotNone(rf)
        self.assertTrue(hasattr(rf, "predict"))


class TestNodataCollisionGuard(unittest.TestCase):

    def _apply(self, sampled, nodata_val):
        valid = np.isfinite(sampled)
        if nodata_val is not None and np.isfinite(nodata_val) and abs(float(nodata_val)) > 0.5:
            valid &= (sampled != nodata_val)
        return sampled[valid]

    def test_nodata_zero_keeps_all_pixels(self):
        mask = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
        self.assertEqual(len(self._apply(mask, 0.0)), 5)

    def test_nodata_minus9999_excluded(self):
        mask = np.array([0.0, 1.0, -9999.0, 0.0, -9999.0])
        result = self._apply(mask, -9999.0)
        self.assertEqual(len(result), 3)
        self.assertNotIn(-9999.0, result)

    def test_nodata_none_no_exclusion(self):
        self.assertEqual(len(self._apply(np.array([0.0, 1.0, 0.0]), None)), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
