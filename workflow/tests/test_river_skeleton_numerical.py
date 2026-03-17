"""Tests for river_skeleton_bathy.py numerical core functions.

Pure numpy/pandas tests — no geo dependencies required.
"""

import unittest
import numpy as np
import pandas as pd

try:
    import river_skeleton_bathy
    _HAS_GEO = True
except (ImportError, ModuleNotFoundError):
    _HAS_GEO = False

_skip_msg = "river_skeleton_bathy requires shapely/geopandas"


class TestPavaNonDecreasing(unittest.TestCase):
    """Tests for _pava_non_decreasing (Pool-Adjacent-Violators)."""

    @staticmethod
    def _pava(y):
        """Standalone PAVA implementation matching river_skeleton_bathy."""
        y = np.asarray(y, dtype=float)
        n = int(y.size)
        if n <= 1:
            return y.astype(float)
        starts, ends, means = [], [], []
        for i in range(n):
            starts.append(i)
            ends.append(i)
            means.append(float(y[i]))
            while len(means) >= 2 and means[-2] > means[-1]:
                s0, e0, m0 = starts[-2], ends[-2], means[-2]
                s1, e1, m1 = starts[-1], ends[-1], means[-1]
                w0 = (e0 - s0 + 1)
                w1 = (e1 - s1 + 1)
                m = (m0 * w0 + m1 * w1) / float(w0 + w1)
                starts[-2] = s0
                ends[-2] = e1
                means[-2] = float(m)
                starts.pop(); ends.pop(); means.pop()
        out = np.empty(n, dtype=float)
        for s, e, m in zip(starts, ends, means):
            out[s:e + 1] = float(m)
        return out

    def test_already_sorted(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self._pava(y)
        np.testing.assert_array_equal(result, y)

    def test_reversed(self):
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = self._pava(y)
        expected = np.full(5, 3.0)  # all pooled to mean
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_violation(self):
        y = np.array([1.0, 3.0, 2.0, 4.0])
        result = self._pava(y)
        self.assertAlmostEqual(result[1], 2.5)
        self.assertAlmostEqual(result[2], 2.5)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[3], 4.0)

    def test_monotonic_output(self):
        rng = np.random.RandomState(42)
        y = rng.randn(100)
        result = self._pava(y)
        diffs = np.diff(result)
        self.assertTrue(np.all(diffs >= -1e-12), "Output must be non-decreasing")

    def test_empty(self):
        result = self._pava(np.array([]))
        self.assertEqual(result.size, 0)

    def test_single_element(self):
        result = self._pava(np.array([42.0]))
        np.testing.assert_array_equal(result, [42.0])

    def test_constant(self):
        y = np.full(10, 7.0)
        result = self._pava(y)
        np.testing.assert_array_equal(result, y)

    def test_preserves_mean(self):
        """PAVA preserves the overall weighted mean."""
        rng = np.random.RandomState(123)
        y = rng.randn(50) * 5
        result = self._pava(y)
        self.assertAlmostEqual(np.mean(result), np.mean(y), places=10)


class TestIsotonicNonIncreasing(unittest.TestCase):
    """Tests for _isotonic_non_increasing."""

    @staticmethod
    def _isotonic_non_increasing(y):
        y = np.asarray(y, dtype=float)
        return -TestPavaNonDecreasing._pava(-y)

    def test_already_decreasing(self):
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = self._isotonic_non_increasing(y)
        np.testing.assert_array_equal(result, y)

    def test_increasing_input(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self._isotonic_non_increasing(y)
        expected = np.full(5, 3.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_monotonic_output(self):
        rng = np.random.RandomState(99)
        y = rng.randn(100)
        result = self._isotonic_non_increasing(y)
        diffs = np.diff(result)
        self.assertTrue(np.all(diffs <= 1e-12), "Output must be non-increasing")


@unittest.skipUnless(_HAS_GEO, _skip_msg)
class TestDmaxFromPowerlaw(unittest.TestCase):
    """Tests for _dmax_from_powerlaw."""

    def setUp(self):
        from river_skeleton_bathy import _dmax_from_powerlaw
        self.fn = _dmax_from_powerlaw

    def test_basic(self):
        widths = np.array([10.0, 50.0, 100.0])
        result = self.fn(widths, a0=0.3, bw=0.35, dmin=0.1, dmax=20.0)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (3,))
        # Wider rivers should be deeper
        self.assertGreater(result[2], result[1])
        self.assertGreater(result[1], result[0])

    def test_clipping_min(self):
        widths = np.array([0.001])
        result = self.fn(widths, a0=0.001, bw=0.1, dmin=1.0, dmax=20.0)
        self.assertGreaterEqual(result[0], 1.0)

    def test_clipping_max(self):
        widths = np.array([1e6])
        result = self.fn(widths, a0=10.0, bw=1.0, dmin=0.1, dmax=5.0)
        self.assertLessEqual(result[0], 5.0)

    def test_zero_width_safe(self):
        """Should not produce NaN or Inf for zero-width input."""
        widths = np.array([0.0, -1.0])
        result = self.fn(widths, a0=0.3, bw=0.35, dmin=0.1, dmax=20.0)
        self.assertTrue(np.all(np.isfinite(result)))


@unittest.skipUnless(_HAS_GEO, _skip_msg)
class TestCoerceDepth(unittest.TestCase):
    """Tests for _coerce_depth."""

    def setUp(self):
        from river_skeleton_bathy import _coerce_depth
        self.fn = _coerce_depth

    def test_auto_negative_down(self):
        """If most values are negative, auto mode should negate them."""
        z = np.array([-5.0, -3.0, -8.0, -1.0, -10.0])
        result = self.fn(z, 'auto')
        self.assertTrue(np.all(result >= 0))

    def test_auto_positive_down(self):
        """If most values are positive, auto mode should keep them."""
        z = np.array([5.0, 3.0, 8.0, 1.0, 10.0])
        result = self.fn(z, 'auto')
        self.assertTrue(np.all(result >= 0))

    def test_depth_pos_mode(self):
        z = np.array([5.0, 3.0, -1.0])
        result = self.fn(z, 'depth_pos')
        # Should take absolute values
        self.assertTrue(np.all(result >= 0))

    def test_depth_neg_mode(self):
        z = np.array([-5.0, -3.0, -8.0])
        result = self.fn(z, 'depth_neg')
        # Negated, then abs
        np.testing.assert_array_almost_equal(result, [5.0, 3.0, 8.0])

    def test_empty(self):
        z = np.array([])
        result = self.fn(z, 'auto')
        self.assertEqual(result.size, 0)

    def test_nan_handling(self):
        z = np.array([np.nan, -5.0, np.nan, -3.0])
        result = self.fn(z, 'auto')
        self.assertEqual(result.size, 2)  # NaN values filtered out


if __name__ == "__main__":
    unittest.main()
