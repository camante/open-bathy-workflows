"""Tests for alignment.py – pure numpy/pandas functions, no rasterio/geopandas."""

import sys, os, unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignment import (
    _normalize_source,
    select_tiepoints,
    _guess_lonlat_cols,
    _guess_depth_col,
    _detect_bimodal,
    _residual_summary,
    fit_median_shift,
    fit_depth_stratified_shift,
    _depth_bins_from_spec,
    _huber_weights,
)


class TestNormalizeSource(unittest.TestCase):

    def test_known_aliases(self):
        self.assertEqual(_normalize_source("atl24"), "atl24")
        self.assertEqual(_normalize_source("ATL24"), "atl24")
        self.assertEqual(_normalize_source("atl03"), "atl03")
        self.assertEqual(_normalize_source("atl03_refraction"), "atl03")
        self.assertEqual(_normalize_source("extra_xyz"), "xyz")
        self.assertEqual(_normalize_source("multibeam"), "xyz")
        self.assertEqual(_normalize_source("soundings"), "xyz")

    def test_unknown_returns_unknown(self):
        self.assertEqual(_normalize_source("random_source"), "unknown")
        self.assertEqual(_normalize_source(""), "unknown")
        self.assertEqual(_normalize_source(None), "unknown")

    def test_whitespace_stripped(self):
        self.assertEqual(_normalize_source("  atl24  "), "atl24")


class TestGuessColumns(unittest.TestCase):

    def test_standard_names(self):
        lon, lat = _guess_lonlat_cols(["longitude", "latitude", "depth_m"])
        self.assertEqual(lon, "longitude")
        self.assertEqual(lat, "latitude")

    def test_short_names(self):
        lon, lat = _guess_lonlat_cols(["lon", "lat", "z"])
        self.assertEqual(lon, "lon")
        self.assertEqual(lat, "lat")

    def test_xy_names(self):
        lon, lat = _guess_lonlat_cols(["x", "y", "depth"])
        self.assertEqual(lon, "x")
        self.assertEqual(lat, "y")

    def test_missing_returns_none(self):
        lon, lat = _guess_lonlat_cols(["foo", "bar"])
        self.assertIsNone(lon)
        self.assertIsNone(lat)

    def test_guess_depth_col(self):
        self.assertEqual(_guess_depth_col(["lon", "lat", "depth_m"]), "depth_m")
        self.assertEqual(_guess_depth_col(["x", "y", "z"]), "z")
        self.assertEqual(_guess_depth_col(["x", "y", "elevation_m"]), "elevation_m")
        self.assertIsNone(_guess_depth_col(["a", "b", "c"]))


class TestSelectTiepoints(unittest.TestCase):

    def _make_df(self):
        return pd.DataFrame({
            "longitude": [-81.0, -81.1, -81.2, -81.3, -81.4, -81.5],
            "latitude": [25.0, 25.1, 25.2, 25.3, 25.4, 25.5],
            "depth_m": [-5, -8, -3, -10, -7, -2],
            "source": ["atl24", "atl24", "atl03", "atl03", "extra_xyz", "multibeam"],
        })

    def test_filter_atl24_only(self):
        df = self._make_df()
        out = select_tiepoints(df, mode="atl24")
        self.assertEqual(len(out), 2)

    def test_filter_xyz_only(self):
        df = self._make_df()
        out = select_tiepoints(df, mode="xyz")
        # "extra_xyz" and "multibeam" both normalize to "xyz"
        self.assertEqual(len(out), 2)

    def test_stacked_includes_all_known(self):
        df = self._make_df()
        out = select_tiepoints(df, mode="stacked")
        # xyz(2) + atl03(2) + atl24(2) = 6
        self.assertEqual(len(out), 6)

    def test_per_source_max(self):
        df = self._make_df()
        out = select_tiepoints(df, mode="stacked", per_source_max=1, seed=42)
        # At most 1 per source category: xyz(1) + atl03(1) + atl24(1) = 3
        self.assertEqual(len(out), 3)

    def test_empty_df(self):
        df = pd.DataFrame(columns=["longitude", "latitude", "depth_m", "source"])
        out = select_tiepoints(df, mode="atl24")
        self.assertEqual(len(out), 0)

    def test_none_input(self):
        out = select_tiepoints(None)
        self.assertIsNone(out)

    def test_invalid_mode_raises(self):
        df = self._make_df()
        with self.assertRaises(ValueError):
            select_tiepoints(df, mode="invalid_mode")




class TestDetectBimodal(unittest.TestCase):

    def test_unimodal_gaussian(self):
        rng = np.random.default_rng(40)
        r = rng.normal(0, 1, 500)
        result = _detect_bimodal(r)
        self.assertFalse(result["is_bimodal"])

    def test_bimodal_mixture(self):
        rng = np.random.default_rng(41)
        r = np.concatenate([rng.normal(-5, 0.5, 300), rng.normal(5, 0.5, 300)])
        result = _detect_bimodal(r)
        self.assertTrue(result["is_bimodal"])
        self.assertGreaterEqual(result["n_modes"], 2)

    def test_too_few_samples(self):
        r = np.array([1.0, 2.0, 3.0])
        result = _detect_bimodal(r)
        self.assertFalse(result["is_bimodal"])
        self.assertEqual(result["reason"], "insufficient_samples")

    def test_all_nan(self):
        r = np.full(100, np.nan)
        result = _detect_bimodal(r)
        self.assertFalse(result["is_bimodal"])


class TestResidualSummary(unittest.TestCase):

    def test_known_distribution(self):
        r = np.array([1.0, -1.0, 2.0, -2.0, 0.0])
        s = _residual_summary(r)
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["median_m"], 0.0)
        self.assertAlmostEqual(s["mean_m"], 0.0)
        self.assertGreater(s["rmse_m"], 0.0)

    def test_empty_input(self):
        s = _residual_summary(np.array([]))
        self.assertEqual(s["n"], 0)

    def test_all_nan(self):
        s = _residual_summary(np.full(10, np.nan))
        self.assertEqual(s["n"], 0)

    def test_constant_residual(self):
        r = np.full(100, 2.5)
        s = _residual_summary(r)
        self.assertAlmostEqual(s["median_m"], 2.5)
        self.assertAlmostEqual(s["mean_m"], 2.5)
        self.assertAlmostEqual(s["mad_m"], 0.0)
        self.assertAlmostEqual(s["rmse_m"], 2.5)


class TestFitMedianShift(unittest.TestCase):

    def test_recovers_known_offset(self):
        r = np.array([2.0, 2.1, 1.9, 2.05, 1.95])
        result = fit_median_shift(r)
        self.assertAlmostEqual(result["dz_m"], 2.0, places=1)

    def test_zero_residuals(self):
        result = fit_median_shift(np.zeros(50))
        self.assertAlmostEqual(result["dz_m"], 0.0)


class TestFitDepthStratifiedShift(unittest.TestCase):

    def test_uniform_offset_all_bins(self):
        """Constant offset across all depths → same dz in every bin."""
        bins = np.array([0, 5, 10, 15, 20], dtype=np.float64)
        depths = np.array([2, 3, 7, 8, 12, 13, 17, 18], dtype=np.float64)
        residuals = np.full(8, 1.5)
        result = fit_depth_stratified_shift(depths, residuals, bins=bins)
        for dz in result["dz_by_bin_m"]:
            self.assertAlmostEqual(dz, 1.5)

    def test_depth_dependent_bias(self):
        """Different bias in shallow vs deep → different dz per bin."""
        bins = np.array([0, 10, 20], dtype=np.float64)
        depths = np.array([3, 4, 5, 13, 14, 15], dtype=np.float64)
        residuals = np.array([1, 1, 1, 5, 5, 5], dtype=np.float64)
        result = fit_depth_stratified_shift(depths, residuals, bins=bins)
        self.assertAlmostEqual(result["dz_by_bin_m"][0], 1.0)
        self.assertAlmostEqual(result["dz_by_bin_m"][1], 5.0)

    def test_empty_input(self):
        bins = np.array([0, 10, 20], dtype=np.float64)
        result = fit_depth_stratified_shift(np.array([]), np.array([]), bins=bins)
        self.assertEqual(result["n_by_bin"], [0, 0])

    def test_n_by_bin_sums_correctly(self):
        bins = np.array([0, 10, 20], dtype=np.float64)
        depths = np.array([2, 5, 8, 12, 15], dtype=np.float64)
        residuals = np.zeros(5)
        result = fit_depth_stratified_shift(depths, residuals, bins=bins)
        self.assertEqual(sum(result["n_by_bin"]), 5)


class TestDepthBinsFromSpec(unittest.TestCase):

    def test_auto(self):
        edges = _depth_bins_from_spec("auto")
        self.assertGreater(len(edges), 3)
        self.assertEqual(edges[0], 0)

    def test_none(self):
        edges = _depth_bins_from_spec(None)
        self.assertGreater(len(edges), 3)

    def test_custom_spec(self):
        edges = _depth_bins_from_spec("0,5,10,20")
        np.testing.assert_array_equal(edges, [0, 5, 10, 20])

    def test_sorted_and_deduped(self):
        edges = _depth_bins_from_spec("10,0,5,5,20")
        np.testing.assert_array_equal(edges, [0, 5, 10, 20])

    def test_too_few_edges_raises(self):
        with self.assertRaises(ValueError):
            _depth_bins_from_spec("0,5")


class TestHuberWeights(unittest.TestCase):

    def test_small_residuals_weight_one(self):
        u = np.array([0.1, -0.2, 0.3])
        w = _huber_weights(u, c=1.0)
        np.testing.assert_allclose(w, 1.0)

    def test_large_residuals_downweighted(self):
        u = np.array([0.5, 5.0, 10.0])
        w = _huber_weights(u, c=1.0)
        self.assertAlmostEqual(w[0], 1.0)
        self.assertLess(w[1], 1.0)
        self.assertLess(w[2], w[1])

    def test_symmetric(self):
        u = np.array([-3.0, 3.0])
        w = _huber_weights(u, c=1.0)
        self.assertAlmostEqual(w[0], w[1])


if __name__ == "__main__":
    unittest.main()
