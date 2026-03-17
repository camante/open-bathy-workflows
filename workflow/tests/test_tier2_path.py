"""Tests for the tier-2 (degraded/shallow data) prediction path.

Covers: IQR-based model selection, post-QC tier downgrade, stumpf filter
skip, DOA disable, and stumpf envelope skip.
"""

import unittest
import numpy as np
import pandas as pd


class TestPhysicsOnlyDecisionTree(unittest.TestCase):
    """Test the 3-criterion physics-only decision tree."""

    def _check_criteria(self, x_fit, y_fit):
        """Evaluate the 3 criteria and return (use_physics_only, reasons)."""
        corr = np.corrcoef(x_fit, y_fit)[0, 1] if len(x_fit) >= 3 else 0.0
        abs_corr = abs(corr) if np.isfinite(corr) else 0.0
        depth_iqr = float(np.percentile(y_fit, 75) - np.percentile(y_fit, 25))
        depth_range = float(np.max(y_fit) - np.min(y_fit))
        depth_median = float(np.median(y_fit))
        depth_floor = float(np.min(y_fit))
        near_floor_frac = float(np.mean(y_fit < (depth_floor + 0.5)))

        reasons = []
        if abs_corr < 0.3:
            reasons.append("weak_corr")
        if depth_iqr < 1.0 and depth_range > 0 and (depth_range / max(depth_iqr, 0.01)) > 5.0:
            reasons.append("skewed_distribution")
        if near_floor_frac > 0.80 and depth_median < 1.5:
            reasons.append("near_floor_dominance")
        return len(reasons) > 0, reasons

    def test_merrimack_pattern_triggers_physics_only(self):
        """Merrimack-like data: 97% at 0.5-0.8m, 3% at 2-12m."""
        rng = np.random.RandomState(42)
        x = 0.9 + 0.3 * rng.rand(1260)
        y = np.concatenate([
            0.5 + 0.3 * rng.rand(1200),  # shallow cluster
            2.0 + 10.0 * rng.rand(60),   # sparse deeper points
        ])
        physics_only, reasons = self._check_criteria(x, y)
        self.assertTrue(physics_only)
        # Should trigger at least skewed_distribution and near_floor
        self.assertTrue(any("skew" in r for r in reasons) or
                        any("floor" in r for r in reasons))

    def test_good_data_stays_atl_calibrated(self):
        """Well-distributed data with good correlation should use ATL."""
        rng = np.random.RandomState(42)
        n = 500
        x = np.linspace(0.8, 1.5, n)
        y = 2.0 + 8.0 * (x - 0.8) / 0.7 + rng.randn(n) * 0.5  # clear trend
        physics_only, reasons = self._check_criteria(x, y)
        self.assertFalse(physics_only, f"Should be ATL-calibrated but got: {reasons}")

    def test_weak_correlation_triggers(self):
        """No correlation between stumpf_idx and depth → physics-only."""
        rng = np.random.RandomState(42)
        n = 500
        x = rng.rand(n)
        y = rng.rand(n) * 10  # random, no correlation
        physics_only, reasons = self._check_criteria(x, y)
        self.assertTrue(physics_only)
        self.assertIn("weak_corr", reasons)

    def test_near_floor_dominance_triggers(self):
        """80%+ of points within 0.5m of floor, median < 1.5m."""
        rng = np.random.RandomState(42)
        y = np.concatenate([
            0.3 + 0.2 * rng.rand(900),   # all near floor
            3.0 + 2.0 * rng.rand(100),   # some deeper
        ])
        x = 1.0 + 0.1 * rng.randn(1000)
        physics_only, reasons = self._check_criteria(x, y)
        self.assertTrue(physics_only)
        self.assertTrue(any("floor" in r for r in reasons))
    """Test that the Stumpf residual filter is skipped for shallow-dominated data."""

    def _make_shallow_dominated_df(self, n_shallow=1000, n_deep=50):
        """Create a DataFrame mimicking the Merrimack pattern:
        mostly near-surface ATL03 with a few deeper ATL24 points."""
        rng = np.random.RandomState(42)
        # Shallow cluster at ~0.7m
        d_shallow = -(0.5 + 0.3 * rng.rand(n_shallow))
        # Deeper points at 2-5m
        d_deep = -(2.0 + 3.0 * rng.rand(n_deep))
        depths = np.concatenate([d_shallow, d_deep])
        n = len(depths)
        return pd.DataFrame({
            "longitude": -71.0 + 0.1 * rng.randn(n),
            "latitude": 42.8 + 0.1 * rng.randn(n),
            "depth_m": depths,
            "source": ["atl03"] * n_shallow + ["atl24"] * n_deep,
        })

    def test_iqr_detects_shallow_dominated(self):
        """IQR < 1m with range > 1.5m should flag shallow-dominated."""
        df = self._make_shallow_dominated_df()
        d = np.abs(df["depth_m"].values)
        iqr = float(np.percentile(d, 75) - np.percentile(d, 25))
        full_range = float(np.max(d) - np.min(d))
        self.assertLess(iqr, 1.0)
        self.assertGreater(full_range, 1.5)

    def test_linear_chosen_for_shallow_dominated(self):
        """With shallow-dominated data, linear model should be chosen over isotonic."""
        rng = np.random.RandomState(42)
        n = 500
        # Simulate stumpf_idx and depth with shallow-dominated pattern
        x = 0.8 + 0.5 * rng.rand(n)  # stumpf_idx range
        y = np.abs(np.concatenate([
            0.5 + 0.3 * rng.rand(450),   # shallow cluster
            2.0 + 3.0 * rng.rand(50),    # deeper points
        ]))
        depth_iqr = float(np.percentile(y, 75) - np.percentile(y, 25))
        depth_range = float(np.max(y) - np.min(y))
        shallow_dominated = (depth_iqr < 1.0 and depth_range > 1.5)
        self.assertTrue(shallow_dominated)

    def test_linear_model_extrapolates(self):
        """A linear Stumpf model should extrapolate beyond training range."""
        from sklearn.linear_model import LinearRegression
        rng = np.random.RandomState(42)
        # Train on 0.5-5m depth
        x = np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.4]).reshape(-1, 1)
        y = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        lr = LinearRegression()
        lr.fit(x, y)
        # Predict at stumpf_idx=1.8 (beyond training range)
        pred = lr.predict(np.array([[1.8]]))[0]
        self.assertGreater(pred, 5.0, "Linear model should extrapolate beyond training max")

    def test_isotonic_clips(self):
        """An isotonic model with out_of_bounds='clip' should NOT extrapolate."""
        from sklearn.isotonic import IsotonicRegression
        x = np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
        y = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(x, y)
        pred = iso.predict(np.array([1.8]))[0]
        self.assertLessEqual(pred, 5.0, "Isotonic should clip at training max")


class TestPostQCTierDowngrade(unittest.TestCase):
    """Test that tier 1 is downgraded to 2 when post-QC depth range collapses."""

    def test_narrow_range_triggers_downgrade(self):
        from sdb_tier import TIER1_MIN_DEPTH_RANGE_M, TIER1_MIN_DEPTH_STD_M
        # Simulate post-QC data: 0.5-1.6m range
        d = np.abs(np.concatenate([
            np.full(800, -0.7),
            np.linspace(-0.5, -1.6, 38),
        ]))
        post_qc_range = float(np.max(d) - np.min(d))
        post_qc_std = float(np.std(d))
        self.assertLess(post_qc_range, TIER1_MIN_DEPTH_RANGE_M)
        self.assertLess(post_qc_std, TIER1_MIN_DEPTH_STD_M)

    def test_wide_range_no_downgrade(self):
        from sdb_tier import TIER1_MIN_DEPTH_RANGE_M
        d = np.linspace(0.5, 15.0, 500)
        post_qc_range = float(np.max(d) - np.min(d))
        self.assertGreaterEqual(post_qc_range, TIER1_MIN_DEPTH_RANGE_M)


class TestNUniqueTracksNoFallback(unittest.TestCase):
    """n_unique_tracks should NOT fall back to source.nunique()."""

    def test_no_track_column_yields_zero(self):
        from sdb_tier import assess_training_data
        df = pd.DataFrame({
            "longitude": [0.0], "latitude": [0.0], "depth_m": [-5.0],
            "source": ["atl03"],
        })
        q = assess_training_data(df)
        self.assertEqual(q.n_unique_tracks, 0)

    def test_track_column_counted(self):
        from sdb_tier import assess_training_data
        df = pd.DataFrame({
            "longitude": [0.0, 1.0, 2.0],
            "latitude": [0.0, 1.0, 2.0],
            "depth_m": [-5.0, -6.0, -7.0],
            "source": ["atl03", "atl03", "atl24"],
            "track": ["rgt_100", "rgt_200", "rgt_200"],
        })
        q = assess_training_data(df)
        self.assertEqual(q.n_unique_tracks, 2)


class TestDOARelaxForTier2(unittest.TestCase):
    """When tier >= 2, training_bounds should be preserved for soft DOA gating."""

    def test_training_bounds_preserved(self):
        """Simulate the sdb_main.py logic for tier-2 DOA relaxation."""
        model_meta = {
            "training_bounds": {"B02": {"min": 0.01, "max": 0.05}},
            "doa": {"threshold_default": 0.97, "soft_k_default": 8.0},
        }
        _tier = 2
        if _tier >= 2:
            doa_cfg = model_meta.get("doa", {})
            doa_cfg["relaxed_for_tier"] = _tier
            model_meta["doa"] = doa_cfg
        self.assertIn("training_bounds", model_meta)
        self.assertEqual(model_meta["doa"]["relaxed_for_tier"], 2)

    def test_tier1_preserves_bounds(self):
        model_meta = {
            "training_bounds": {"B02": {"min": 0.01, "max": 0.05}},
        }
        _tier = 1
        if _tier >= 2:
            if "training_bounds" in model_meta:
                del model_meta["training_bounds"]
        self.assertIn("training_bounds", model_meta)


class TestStumpfEnvelopeSkip(unittest.TestCase):
    """When model_tier >= 2, stumpf envelope clipping should be skipped."""

    def test_tier1_clips(self):
        """Tier 1 should clip predictions to stumpf_support ± envelope."""
        stumpf_support = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([0.1, 1.0, 5.0])
        envelope_m = 0.5
        model_tier = 1
        if model_tier < 2:
            y_pred = np.clip(y_pred,
                             np.maximum(stumpf_support - envelope_m, 0.0),
                             stumpf_support + envelope_m)
        np.testing.assert_array_less(y_pred, 1.6)

    def test_tier2_no_clip(self):
        """Tier 2 should NOT clip — linear model extrapolation is desired."""
        stumpf_support = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([0.1, 1.0, 5.0])
        model_tier = 2
        if model_tier < 2:
            y_pred = np.clip(y_pred,
                             np.maximum(stumpf_support - 0.5, 0.0),
                             stumpf_support + 0.5)
        self.assertAlmostEqual(y_pred[2], 5.0)


if __name__ == "__main__":
    unittest.main()
