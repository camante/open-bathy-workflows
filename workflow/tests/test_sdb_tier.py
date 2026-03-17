"""Tests for sdb_tier.py — tiered model selection and confidence weighting."""

import unittest
import numpy as np
import pandas as pd


class TestTrainingDataQuality(unittest.TestCase):

    def setUp(self):
        from sdb_tier import assess_training_data, TrainingDataQuality
        self.assess = assess_training_data
        self.cls = TrainingDataQuality

    def _make_df(self, n, sources=None, high_conf_frac=0.5):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "longitude": -78.0 + 0.01 * rng.randn(n),
            "latitude": 25.0 + 0.01 * rng.randn(n),
            "depth_m": -(1.0 + 10.0 * rng.rand(n)),
            "source": sources or (["atl03"] * n),
            "atl03_conf": ([4] * int(n * high_conf_frac) +
                           [2] * (n - int(n * high_conf_frac))),
        })
        return df

    def test_empty_df(self):
        q = self.assess(pd.DataFrame())
        self.assertEqual(q.n_total, 0)
        self.assertEqual(q.selected_tier, 3)

    def test_none_df(self):
        q = self.assess(None)
        self.assertEqual(q.n_total, 0)
        self.assertEqual(q.selected_tier, 3)

    def test_tier3_few_points(self):
        df = self._make_df(30)
        q = self.assess(df)
        self.assertEqual(q.selected_tier, 3)
        self.assertIn("insufficient", q.tier_reason)

    def test_tier2_moderate_points(self):
        df = self._make_df(150, high_conf_frac=0.5)
        q = self.assess(df, aoi_bounds=(-78.5, -77.5, 24.5, 25.5))
        self.assertEqual(q.selected_tier, 2)

    def test_tier1_many_points_good_coverage(self):
        # Wide spatial spread for good coverage + sufficient depth range + multiple tracks
        rng = np.random.RandomState(42)
        n = 500
        df = pd.DataFrame({
            "longitude": -78.0 + 0.3 * rng.randn(n),
            "latitude": 25.0 + 0.3 * rng.randn(n),
            "depth_m": -(1.0 + 10.0 * rng.rand(n)),
            "source": ["atl03"] * n,
            "atl03_conf": [4] * n,
            "track": [f"track_{i % 5}" for i in range(n)],  # 5 unique tracks
        })
        q = self.assess(df, aoi_bounds=(-78.5, -77.5, 24.5, 25.5))
        self.assertEqual(q.selected_tier, 1)

    def test_tier2_poor_spatial_coverage(self):
        """Many points but clustered in a tiny area → tier 2."""
        rng = np.random.RandomState(42)
        n = 500
        df = pd.DataFrame({
            "longitude": -78.0 + 0.001 * rng.randn(n),  # very tight cluster
            "latitude": 25.0 + 0.001 * rng.randn(n),
            "depth_m": -(1.0 + 10.0 * rng.rand(n)),
            "source": ["atl03"] * n,
            "atl03_conf": [4] * n,
        })
        q = self.assess(df, aoi_bounds=(-79.0, -77.0, 24.0, 26.0))
        self.assertEqual(q.selected_tier, 2)
        self.assertIn("spatial_cov", q.tier_reason)

    def test_tier2_low_confidence(self):
        """Many points but mostly low confidence → tier 2."""
        df = self._make_df(400, high_conf_frac=0.1)
        rng = np.random.RandomState(42)
        df["longitude"] = -78.0 + 0.3 * rng.randn(400)
        df["latitude"] = 25.0 + 0.3 * rng.randn(400)
        q = self.assess(df, aoi_bounds=(-78.5, -77.5, 24.5, 25.5))
        self.assertEqual(q.selected_tier, 2)
        self.assertIn("high_conf_frac", q.tier_reason)

    def test_source_breakdown(self):
        df = pd.DataFrame({
            "longitude": [-78.0] * 5,
            "latitude": [25.0] * 5,
            "depth_m": [-5.0] * 5,
            "source": ["atl03", "atl03", "atl24", "extra_xyz", "extra_xyz"],
        })
        q = self.assess(df)
        self.assertEqual(q.n_atl03, 2)
        self.assertEqual(q.n_atl24, 1)
        self.assertEqual(q.n_extra_xyz, 2)

    def test_kd_max_depth_passthrough(self):
        q = self.assess(pd.DataFrame(), kd_max_depth_m=15.0)
        self.assertEqual(q.kd_max_depth_m, 15.0)

    def test_tier2_shallow_depth_range(self):
        """Shallow data (range < 2m) should force tier 2."""
        rng = np.random.RandomState(42)
        n = 500
        df = pd.DataFrame({
            "longitude": -78.0 + 0.3 * rng.randn(n),
            "latitude": 25.0 + 0.3 * rng.randn(n),
            "depth_m": -(0.5 + 1.0 * rng.rand(n)),
            "source": ["atl03"] * n,
            "atl03_conf": [4] * n,
            "track": [f"track_{i % 5}" for i in range(n)],
        })
        q = self.assess(df, aoi_bounds=(-78.5, -77.5, 24.5, 25.5))
        self.assertEqual(q.selected_tier, 2)
        self.assertIn("depth_range", q.tier_reason)

    def test_tier2_low_depth_std(self):
        """Low depth std (<0.5m) should force tier 2."""
        rng = np.random.RandomState(42)
        n = 500
        df = pd.DataFrame({
            "longitude": -78.0 + 0.3 * rng.randn(n),
            "latitude": 25.0 + 0.3 * rng.randn(n),
            "depth_m": -(5.0 + 0.1 * rng.randn(n)),
            "source": ["atl03"] * n,
            "atl03_conf": [4] * n,
            "track": [f"track_{i % 5}" for i in range(n)],
        })
        q = self.assess(df, aoi_bounds=(-78.5, -77.5, 24.5, 25.5))
        self.assertEqual(q.selected_tier, 2)
        self.assertIn("depth_std", q.tier_reason)


class TestSelectModelTier(unittest.TestCase):

    def test_override(self):
        from sdb_tier import select_model_tier
        df = pd.DataFrame({"depth_m": [1.0]})
        tier, q = select_model_tier(df, override_tier=1)
        self.assertEqual(tier, 1)

    def test_auto_selection(self):
        from sdb_tier import select_model_tier
        df = pd.DataFrame({"depth_m": [1.0] * 10})
        tier, q = select_model_tier(df)
        self.assertEqual(tier, 3)  # too few points


class TestPixelConfidence(unittest.TestCase):

    def setUp(self):
        from sdb_tier import compute_pixel_confidence
        self.fn = compute_pixel_confidence

    def test_tier1_higher_than_tier2(self):
        depth = np.array([5.0, 10.0, 15.0])
        c1 = self.fn(depth, model_tier=1, max_depth_m=20.0)
        c2 = self.fn(depth, model_tier=2, max_depth_m=20.0)
        self.assertTrue(np.all(c1 >= c2))

    def test_tier2_higher_than_tier3(self):
        depth = np.array([5.0, 10.0])
        c2 = self.fn(depth, model_tier=2, max_depth_m=20.0)
        c3 = self.fn(depth, model_tier=3, max_depth_m=20.0)
        self.assertTrue(np.all(c2 >= c3))

    def test_shallow_higher_than_deep(self):
        depth = np.array([2.0, 18.0])
        conf = self.fn(depth, model_tier=1, max_depth_m=20.0)
        self.assertGreater(conf[0], conf[1])

    def test_nan_gets_zero(self):
        depth = np.array([5.0, np.nan, 10.0])
        conf = self.fn(depth, model_tier=1, max_depth_m=20.0)
        self.assertEqual(conf[1], 0.0)

    def test_support_distance_decay(self):
        depth = np.array([5.0, 5.0])
        dist = np.array([0.0, 1000.0])
        conf = self.fn(depth, model_tier=1, max_depth_m=20.0,
                       support_distance=dist, support_decay_m=500.0)
        self.assertGreater(conf[0], conf[1])

    def test_output_range(self):
        depth = np.random.RandomState(42).uniform(0, 30, size=100)
        conf = self.fn(depth, model_tier=1, max_depth_m=20.0)
        self.assertTrue(np.all(conf >= 0.0))
        self.assertTrue(np.all(conf <= 1.0))

    def test_lower_rmse_higher_confidence(self):
        depth = np.array([5.0])
        c_good = self.fn(depth, model_tier=1, rmse_m=0.5, max_depth_m=20.0)
        c_bad = self.fn(depth, model_tier=1, rmse_m=3.0, max_depth_m=20.0)
        self.assertGreater(c_good[0], c_bad[0])


class TestEdgeTaper(unittest.TestCase):

    def setUp(self):
        from sdb_tier import apply_edge_taper
        self.fn = apply_edge_taper

    def test_center_unchanged(self):
        conf = np.ones((100, 100), dtype=np.float32)
        tapered = self.fn(conf, taper_frac=0.10)
        # Center pixel should be 1.0
        self.assertAlmostEqual(tapered[50, 50], 1.0, places=5)

    def test_edges_reduced(self):
        conf = np.ones((100, 100), dtype=np.float32)
        tapered = self.fn(conf, taper_frac=0.10)
        # Corner should be near zero
        self.assertLess(tapered[0, 0], 0.05)
        # Edge midpoint should be reduced
        self.assertLess(tapered[0, 50], 0.6)

    def test_symmetry(self):
        conf = np.ones((100, 100), dtype=np.float32)
        tapered = self.fn(conf, taper_frac=0.10)
        np.testing.assert_array_almost_equal(tapered[0, :], tapered[-1, :])
        np.testing.assert_array_almost_equal(tapered[:, 0], tapered[:, -1])

    def test_no_taper(self):
        conf = np.ones((50, 50), dtype=np.float32) * 0.8
        tapered = self.fn(conf, taper_frac=0.0)
        np.testing.assert_array_almost_equal(tapered, conf)

    def test_preserves_zeros(self):
        conf = np.zeros((50, 50), dtype=np.float32)
        tapered = self.fn(conf, taper_frac=0.10)
        np.testing.assert_array_equal(tapered, 0.0)

    def test_1d_passthrough(self):
        """1D arrays should pass through unchanged."""
        conf = np.ones(100, dtype=np.float32)
        result = self.fn(conf, taper_frac=0.10)
        np.testing.assert_array_equal(result, conf)


class TestGuidePointMeta(unittest.TestCase):

    def test_to_dict(self):
        from sdb_tier import GuidePointMeta
        m = GuidePointMeta(model_tier=2, n_points=500, rmse_m=1.2)
        d = m.to_dict()
        self.assertEqual(d["model_tier"], 2)
        self.assertEqual(d["n_points"], 500)
        self.assertAlmostEqual(d["rmse_m"], 1.2)


if __name__ == "__main__":
    unittest.main()


class TestFilterAtlByKd(unittest.TestCase):

    def setUp(self):
        from sdb_tier import filter_atl_by_kd
        self.fn = filter_atl_by_kd

    def _make_df(self, n_atl=50, n_xyz=20):
        rng = np.random.RandomState(42)
        sources = ["atl03"] * n_atl + ["extra_xyz"] * n_xyz
        n = n_atl + n_xyz
        return pd.DataFrame({
            "longitude": rng.uniform(-78, -77, n),
            "latitude": rng.uniform(25, 26, n),
            "depth_m": -(1 + 10 * rng.rand(n)),
            "source": sources,
            "atl03_conf": [4] * (n_atl // 2) + [2] * (n_atl - n_atl // 2) + [0] * n_xyz,
        })

    def test_clear_water_keeps_all(self):
        df = self._make_df()
        out, meta = self.fn(df, kd_490=0.05)
        self.assertEqual(len(out), len(df))
        self.assertEqual(meta["action"], "clear_water_all_reliable")

    def test_turbid_water_rejects_atl(self):
        df = self._make_df(n_atl=50, n_xyz=20)
        out, meta = self.fn(df, kd_490=0.25)
        self.assertEqual(meta["action"], "turbid_water_atl_rejected")
        self.assertEqual(len(out), 20)  # only xyz remain
        self.assertEqual(meta["n_removed"], 50)

    def test_marginal_downweights(self):
        df = self._make_df()
        out, meta = self.fn(df, kd_490=0.15)
        self.assertEqual(meta["action"], "marginal_water_atl_downweighted")
        self.assertEqual(len(out), len(df))
        self.assertGreater(meta["n_downweighted"], 0)
        # Low-conf ATL points should have reduced weight
        low_conf_atl = out[(out["source"].str.contains("atl")) &
                           (out["atl03_conf"] < 4)]
        if len(low_conf_atl) > 0:
            self.assertTrue((low_conf_atl["sample_weight"] < 1.0).all())

    def test_empty_df(self):
        out, meta = self.fn(pd.DataFrame(), kd_490=0.3)
        self.assertEqual(len(out), 0)

    def test_no_atl_sources(self):
        df = pd.DataFrame({
            "depth_m": [-5.0], "source": ["extra_xyz"],
        })
        out, meta = self.fn(df, kd_490=0.25)
        self.assertEqual(meta["action"], "no_atl_points")
        self.assertEqual(len(out), 1)


class TestBuildPhysicsConstraintSurface(unittest.TestCase):

    def setUp(self):
        from sdb_tier import build_physics_constraint_surface
        self.fn = build_physics_constraint_surface

    def test_basic(self):
        kd = np.full((10, 10), 0.1, dtype=np.float32)
        surface = self.fn(kd, confidence_level="moderate")
        expected = 2.3 / 0.1  # 23m
        np.testing.assert_array_almost_equal(surface, expected, decimal=0)

    def test_nan_propagation(self):
        kd = np.array([[0.1, np.nan], [0.05, 0.0]])
        surface = self.fn(kd, confidence_level="moderate")
        self.assertTrue(np.isnan(surface[0, 1]))  # nan input
        self.assertTrue(np.isnan(surface[1, 1]))  # zero kd → nan

    def test_hard_cap(self):
        kd = np.full((5, 5), 0.01, dtype=np.float32)  # very clear
        surface = self.fn(kd, confidence_level="optimistic", hard_cap_m=40.0)
        self.assertTrue(np.all(surface[np.isfinite(surface)] <= 40.0))

    def test_turbid_shallow(self):
        kd = np.full((5, 5), 0.5, dtype=np.float32)  # turbid
        surface = self.fn(kd, confidence_level="conservative")
        expected = 1.5 / 0.5  # 3m
        np.testing.assert_array_almost_equal(surface, expected, decimal=0)


class TestStumpfRfDisagreement(unittest.TestCase):

    def setUp(self):
        from sdb_tier import compute_stumpf_rf_disagreement
        self.fn = compute_stumpf_rf_disagreement

    def test_agreement(self):
        """When RF and Stumpf agree, no pixels flagged."""
        rng = np.random.RandomState(42)
        n = 1000
        stumpf = rng.uniform(1, 20, n)
        rf = stumpf + rng.normal(0, 0.5, n)  # small noise
        disagree, penalty, meta = self.fn(rf, stumpf, sigma_threshold=2.0)
        self.assertLess(meta["frac_flagged"], 0.10)

    def test_disagreement(self):
        """Large divergence should flag pixels."""
        n = 1000
        stumpf = np.full(n, 10.0)
        rf = np.full(n, 10.0)
        rf[800:] = 25.0  # last 200 diverge by 15m
        disagree, penalty, meta = self.fn(rf, stumpf, sigma_threshold=2.0)
        self.assertGreater(meta["n_flagged"], 100)
        # Divergent pixels should have reduced penalty
        self.assertTrue(np.all(penalty[800:] < 0.9))

    def test_insufficient_overlap(self):
        rf = np.array([5.0, 6.0])
        stumpf = np.array([5.0, np.nan])
        disagree, penalty, meta = self.fn(rf, stumpf)
        self.assertEqual(meta["action"], "insufficient_overlap")

    def test_penalty_range(self):
        rng = np.random.RandomState(42)
        rf = rng.uniform(1, 20, 500)
        stumpf = rf + rng.normal(0, 3, 500)
        _, penalty, _ = self.fn(rf, stumpf)
        self.assertTrue(np.all(penalty >= 0.2))
        self.assertTrue(np.all(penalty <= 1.0))


class TestApplyStumpfRegularization(unittest.TestCase):

    def setUp(self):
        from sdb_tier import apply_stumpf_regularization
        self.fn = apply_stumpf_regularization

    def test_reduces_confidence_on_disagreement(self):
        n = 500
        conf = np.ones(n, dtype=np.float32) * 0.8
        stumpf = np.full(n, 10.0)
        rf = np.full(n, 10.0)
        rf[400:] = 30.0  # big divergence at end
        reg_conf, meta = self.fn(conf, rf, stumpf)
        # Divergent pixels should have lower confidence
        self.assertLess(reg_conf[450], 0.8)
        # Agreeing pixels should be unchanged
        self.assertAlmostEqual(reg_conf[0], 0.8, places=2)

    def test_no_effect_when_agreeing(self):
        n = 200
        conf = np.ones(n, dtype=np.float32) * 0.9
        depth = np.linspace(1, 15, n)
        reg_conf, meta = self.fn(conf, depth, depth + 0.01)
        np.testing.assert_array_almost_equal(reg_conf, conf, decimal=2)
