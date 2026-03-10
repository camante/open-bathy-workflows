"""Tests for spatial_sampling.py – pure numpy/pandas, no geo dependencies."""

import sys, os, unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial_sampling import (
    fast_grid_prethin_lonlat,
    compute_local_complexity,
    detect_transects,
    adaptive_grid_thinning,
    depth_stratified_sampling,
    identify_gap_regions,
    SamplingConfig,
    SourceConfig,
    _lonlat_to_local_m,
)
from scipy.spatial import cKDTree


class TestLonlatToLocalMeters(unittest.TestCase):

    def test_empty_input(self):
        x, y = _lonlat_to_local_m(np.zeros((0, 2)))
        self.assertEqual(len(x), 0)
        self.assertEqual(len(y), 0)

    def test_equator_scale(self):
        """At the equator, 1 degree lon ≈ 1 degree lat ≈ 111 km."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        x, y = _lonlat_to_local_m(pts)
        # x spans ~111 km, y spans ~111 km
        self.assertAlmostEqual(x[1] - x[0], 111_320.0, delta=500)
        self.assertAlmostEqual(y[2] - y[0], 111_320.0, delta=500)

    def test_high_latitude_lon_shrinks(self):
        """At 60°N, 1 degree lon ≈ 55 km (cos(60°) ≈ 0.5)."""
        pts = np.array([[0.0, 60.0], [1.0, 60.0]])
        x, _ = _lonlat_to_local_m(pts)
        dx = abs(x[1] - x[0])
        self.assertGreater(dx, 50_000)
        self.assertLess(dx, 60_000)


class TestFastGridPrethin(unittest.TestCase):

    def test_empty_input(self):
        mask = fast_grid_prethin_lonlat(np.zeros((0, 2)), 100.0, 100)
        self.assertEqual(len(mask), 0)

    def test_single_point_kept(self):
        pts = np.array([[-81.0, 25.0]])
        mask = fast_grid_prethin_lonlat(pts, 100.0, 10)
        self.assertTrue(mask[0])

    def test_output_respects_max_keep(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform([-82, 24], [-81, 25], size=(10_000, 2))
        mask = fast_grid_prethin_lonlat(pts, 10.0, 500)
        self.assertLessEqual(mask.sum(), 500)

    def test_output_length_matches_input(self):
        rng = np.random.default_rng(1)
        pts = rng.uniform([-82, 24], [-81, 25], size=(200, 2))
        mask = fast_grid_prethin_lonlat(pts, 50.0, 1000)
        self.assertEqual(len(mask), 200)

    def test_deterministic(self):
        rng = np.random.default_rng(2)
        pts = rng.uniform([-82, 24], [-81, 25], size=(500, 2))
        m1 = fast_grid_prethin_lonlat(pts, 50.0, 200, seed=42)
        m2 = fast_grid_prethin_lonlat(pts, 50.0, 200, seed=42)
        np.testing.assert_array_equal(m1, m2)

    def test_coarser_grid_keeps_fewer(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform([-82, 24], [-81, 25], size=(5000, 2))
        m_fine = fast_grid_prethin_lonlat(pts, 10.0, 99999)
        m_coarse = fast_grid_prethin_lonlat(pts, 200.0, 99999)
        self.assertGreater(m_fine.sum(), m_coarse.sum())

    def test_all_same_location_keeps_one(self):
        pts = np.full((100, 2), [-81.5, 24.5])
        mask = fast_grid_prethin_lonlat(pts, 50.0, 99999)
        self.assertEqual(mask.sum(), 1)


class TestComputeLocalComplexity(unittest.TestCase):

    def _make_data(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        pts = rng.uniform([-82, 24], [-81, 25], size=(n, 2))
        depths = rng.uniform(-20, -1, n)
        kdtree = cKDTree(pts)
        return pts, depths, kdtree

    def test_output_shape(self):
        pts, depths, kdt = self._make_data(100)
        cfg = SamplingConfig()
        scores = compute_local_complexity(pts, depths, kdt, cfg)
        self.assertEqual(scores.shape, (100,))

    def test_output_bounded_zero_one(self):
        pts, depths, kdt = self._make_data(300)
        cfg = SamplingConfig()
        scores = compute_local_complexity(pts, depths, kdt, cfg)
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))

    def test_flat_bottom_low_complexity(self):
        """Constant depth → low variance → low complexity."""
        rng = np.random.default_rng(10)
        pts = rng.uniform([-82, 24], [-81, 25], size=(200, 2))
        depths = np.full(200, -5.0)  # perfectly flat
        kdt = cKDTree(pts)
        cfg = SamplingConfig()
        scores = compute_local_complexity(pts, depths, kdt, cfg)
        # Variance component should be 0, so scores should be low
        self.assertLess(np.mean(scores), 0.4)

    def test_variable_bottom_higher_complexity(self):
        """Highly variable depth → higher complexity than flat."""
        rng = np.random.default_rng(11)
        pts = rng.uniform([-82, 24], [-81, 25], size=(200, 2))
        kdt = cKDTree(pts)
        cfg = SamplingConfig()

        flat_depths = np.full(200, -5.0)
        varied_depths = rng.uniform(-30, -1, 200)

        scores_flat = compute_local_complexity(pts, flat_depths, kdt, cfg)
        scores_varied = compute_local_complexity(pts, varied_depths, kdt, cfg)
        self.assertGreater(np.mean(scores_varied), np.mean(scores_flat))


class TestDetectTransects(unittest.TestCase):

    def test_random_cloud_no_transects(self):
        """Randomly scattered points should have few transect detections."""
        rng = np.random.default_rng(20)
        pts = rng.uniform(0, 1000, size=(200, 2))
        is_transect = detect_transects(pts, min_length=10)
        self.assertLess(is_transect.sum(), len(pts) * 0.3)

    def test_linear_points_detected(self):
        """Points along a straight line should be detected as a transect."""
        n = 100
        x = np.linspace(0, 1000, n)
        y = np.full(n, 500.0) + np.random.default_rng(21).normal(0, 0.5, n)
        pts = np.column_stack([x, y])
        is_transect = detect_transects(pts, min_length=10)
        self.assertGreater(is_transect.sum(), n * 0.5)

    def test_too_few_points_returns_all_false(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        is_transect = detect_transects(pts, min_length=50)
        self.assertEqual(is_transect.sum(), 0)

    def test_output_shape(self):
        rng = np.random.default_rng(22)
        pts = rng.uniform(0, 100, size=(80, 2))
        is_transect = detect_transects(pts, min_length=10)
        self.assertEqual(len(is_transect), 80)
        self.assertEqual(is_transect.dtype, bool)


class TestAdaptiveGridThinning(unittest.TestCase):

    def _make_data(self, n=500, seed=0):
        rng = np.random.default_rng(seed)
        pts = rng.uniform([-82, 24], [-81, 25], size=(n, 2))
        depths = rng.uniform(-20, -1, n)
        complexity = rng.uniform(0, 1, n)
        source_vals = np.ones(n)
        return pts, depths, complexity, source_vals

    def test_output_is_boolean_mask(self):
        pts, depths, comp, sv = self._make_data()
        cfg = SamplingConfig()
        sc = SourceConfig("test", tier=1, weight=10.0, retention_target=0.5,
                          min_spacing_m=30.0, influence_radius_m=50.0)
        keep = adaptive_grid_thinning(pts, depths, comp, sv, cfg, sc)
        self.assertEqual(keep.dtype, bool)
        self.assertEqual(len(keep), len(pts))

    def test_empty_input(self):
        pts = np.zeros((0, 2))
        cfg = SamplingConfig()
        sc = SourceConfig("test", tier=1, weight=10.0, retention_target=0.5,
                          min_spacing_m=30.0, influence_radius_m=50.0)
        keep = adaptive_grid_thinning(pts, np.array([]), np.array([]),
                                      np.array([]), cfg, sc)
        self.assertEqual(len(keep), 0)

    def test_at_least_one_point_kept(self):
        pts, depths, comp, sv = self._make_data(100)
        cfg = SamplingConfig()
        sc = SourceConfig("test", tier=1, weight=10.0, retention_target=0.5,
                          min_spacing_m=30.0, influence_radius_m=50.0)
        keep = adaptive_grid_thinning(pts, depths, comp, sv, cfg, sc)
        self.assertGreater(keep.sum(), 0)

    def test_high_retention_keeps_more(self):
        pts, depths, comp, sv = self._make_data(1000, seed=5)
        cfg = SamplingConfig()
        sc_lo = SourceConfig("lo", tier=1, weight=10.0, retention_target=0.1,
                             min_spacing_m=60.0, influence_radius_m=50.0)
        sc_hi = SourceConfig("hi", tier=1, weight=10.0, retention_target=0.9,
                             min_spacing_m=10.0, influence_radius_m=50.0)
        keep_lo = adaptive_grid_thinning(pts, depths, comp, sv, cfg, sc_lo)
        keep_hi = adaptive_grid_thinning(pts, depths, comp, sv, cfg, sc_hi)
        self.assertGreaterEqual(keep_hi.sum(), keep_lo.sum())


class TestDepthStratifiedSampling(unittest.TestCase):

    def test_adds_depth_bin_column(self):
        rng = np.random.default_rng(30)
        df = pd.DataFrame({
            "depth_m": rng.uniform(-30, -1, 500),
            "longitude": rng.uniform(-82, -81, 500),
            "latitude": rng.uniform(24, 25, 500),
        })
        cfg = SamplingConfig(depth_bins=5, min_points_per_bin=10)
        result = depth_stratified_sampling(df, cfg)
        self.assertIn("depth_bin", result.columns)

    def test_too_few_points_returns_unchanged(self):
        df = pd.DataFrame({"depth_m": [-5.0, -10.0]})
        cfg = SamplingConfig(min_points_per_bin=50)
        result = depth_stratified_sampling(df, cfg)
        self.assertEqual(len(result), 2)

    def test_preserves_all_rows(self):
        """Stratification labels bins but does not drop rows."""
        rng = np.random.default_rng(31)
        n = 300
        df = pd.DataFrame({
            "depth_m": rng.uniform(-25, -1, n),
        })
        cfg = SamplingConfig(depth_bins=5, min_points_per_bin=10)
        result = depth_stratified_sampling(df, cfg)
        self.assertEqual(len(result), n)


class TestIdentifyGapRegions(unittest.TestCase):

    def test_no_coverage_everything_is_gap(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        empty_tree = cKDTree(np.zeros((0, 2)).reshape(0, 2) if False else np.array([[999, 999]]))
        # With coverage very far away, all points are gaps
        gaps = identify_gap_regions(pts, cKDTree(np.array([[999.0, 999.0]])),
                                     influence_radius_m=100.0)
        self.assertTrue(np.all(gaps))

    def test_full_coverage_no_gaps(self):
        """All query points sit right on top of coverage points."""
        pts = np.array([[-81.0, 25.0], [-81.01, 25.01]])
        tree = cKDTree(pts.copy())
        gaps = identify_gap_regions(pts, tree, influence_radius_m=1000.0)
        self.assertFalse(np.any(gaps))

    def test_empty_input(self):
        gaps = identify_gap_regions(np.zeros((0, 2)),
                                     cKDTree(np.array([[0.0, 0.0]])),
                                     influence_radius_m=100.0)
        self.assertEqual(len(gaps), 0)

    def test_distant_point_is_gap(self):
        coverage = np.array([[0.0, 0.0]])
        tree = cKDTree(coverage)
        query = np.array([[0.0, 0.0], [10.0, 10.0]])  # second point is ~1500 km away
        gaps = identify_gap_regions(query, tree, influence_radius_m=1000.0)
        self.assertFalse(gaps[0])  # on top of coverage
        self.assertTrue(gaps[1])   # very far


if __name__ == "__main__":
    unittest.main()
