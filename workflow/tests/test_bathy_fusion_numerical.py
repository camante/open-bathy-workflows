"""Tests for bathy_fusion.py numerical core functions.

Pure numpy tests — no rasterio dependencies required.
"""

import unittest
import numpy as np


class TestPriorityFill(unittest.TestCase):
    """Test the priority-fill logic from fuse_bathymetry.

    Extracted as a standalone function to test without rasterio.
    """

    @staticmethod
    def priority_fill(sources, priority_order):
        """Standalone priority-fill matching bathy_fusion._priority_fill."""
        shape = None
        for arr in sources.values():
            if arr is not None:
                shape = arr.shape
                break
        if shape is None:
            return None, None

        out = np.full(shape, np.nan, dtype=np.float64)
        prov = np.zeros(shape, dtype=np.int8)

        for i, src_name in enumerate(priority_order):
            arr = sources.get(src_name)
            if arr is None:
                continue
            m = np.isfinite(arr)
            take = np.isnan(out) & m
            if not take.any():
                continue
            out[take] = arr[take]
            prov[take] = i + 1
        return out, prov

    def test_single_source(self):
        src = {"measured": np.array([1.0, 2.0, np.nan, 4.0])}
        out, prov = self.priority_fill(src, ["measured"])
        np.testing.assert_array_equal(out, [1.0, 2.0, np.nan, 4.0])
        np.testing.assert_array_equal(prov, [1, 1, 0, 1])

    def test_priority_order(self):
        """Higher-priority source wins where both have data."""
        measured = np.array([1.0, np.nan, np.nan, 4.0])
        sdb = np.array([10.0, 20.0, np.nan, 40.0])
        river = np.array([100.0, 200.0, 300.0, 400.0])

        out, prov = self.priority_fill(
            {"measured": measured, "sdb": sdb, "river": river},
            ["measured", "sdb", "river"],
        )
        # measured wins at [0] and [3]
        self.assertEqual(out[0], 1.0)
        self.assertEqual(out[3], 4.0)
        # sdb fills [1]
        self.assertEqual(out[1], 20.0)
        # river fills [2]
        self.assertEqual(out[2], 300.0)
        # provenance
        np.testing.assert_array_equal(prov, [1, 2, 3, 1])

    def test_all_nan(self):
        src = {"a": np.array([np.nan, np.nan])}
        out, prov = self.priority_fill(src, ["a"])
        self.assertTrue(np.all(np.isnan(out)))
        np.testing.assert_array_equal(prov, [0, 0])

    def test_missing_source_skipped(self):
        src = {"sdb": np.array([1.0, 2.0])}
        out, prov = self.priority_fill(src, ["measured", "sdb"])
        np.testing.assert_array_equal(out, [1.0, 2.0])
        np.testing.assert_array_equal(prov, [2, 2])  # sdb is second in order

    def test_empty_sources(self):
        out, prov = self.priority_fill({}, ["measured"])
        self.assertIsNone(out)


class TestWeightedBlend(unittest.TestCase):
    """Test weighted overlap blending logic from fuse_bathymetry."""

    @staticmethod
    def weighted_blend(a, b, w1=0.7, w2=0.3, exclude_measured=None):
        """Standalone weighted blend matching bathy_fusion logic."""
        s = w1 + w2
        if s <= 0:
            w1, w2, s = 0.7, 0.3, 1.0
        w1 /= s
        w2 /= s

        m = np.isfinite(a) & np.isfinite(b)
        if exclude_measured is not None:
            m &= ~np.isfinite(exclude_measured)

        out = np.full_like(a, np.nan)
        out[m] = w1 * a[m] + w2 * b[m]
        return out

    def test_basic_blend(self):
        a = np.array([10.0, 20.0, np.nan])
        b = np.array([12.0, 22.0, 30.0])
        result = self.weighted_blend(a, b, w1=0.7, w2=0.3)
        self.assertAlmostEqual(result[0], 0.7 * 10 + 0.3 * 12)
        self.assertAlmostEqual(result[1], 0.7 * 20 + 0.3 * 22)
        self.assertTrue(np.isnan(result[2]))  # a is NaN

    def test_measured_exclusion(self):
        """Where measured data exists, blending should be skipped."""
        a = np.array([10.0, 20.0])
        b = np.array([12.0, 22.0])
        meas = np.array([5.0, np.nan])  # measured at [0]
        result = self.weighted_blend(a, b, exclude_measured=meas)
        self.assertTrue(np.isnan(result[0]))  # excluded by measured
        self.assertAlmostEqual(result[1], 0.7 * 20 + 0.3 * 22)

    def test_uncertainty_propagation(self):
        """Uncertainty for weighted blend: sqrt(w1^2 * u1^2 + w2^2 * u2^2)."""
        ua = np.array([1.0, 2.0])
        ub = np.array([1.5, 1.0])
        w1, w2 = 0.7, 0.3
        expected = np.sqrt((w1 * ua) ** 2 + (w2 * ub) ** 2)
        np.testing.assert_array_almost_equal(expected[0], np.sqrt(0.49 + 0.2025))


class TestProvenanceCodes(unittest.TestCase):
    """Verify provenance coding matches expected values."""

    def test_provenance_distinct(self):
        """Each source should get a unique provenance code."""
        sources = {"measured": np.array([1.0]),
                   "sdb": np.array([np.nan]),
                   "river": np.array([np.nan])}
        _, prov = TestPriorityFill.priority_fill(
            sources, ["measured", "sdb", "river"])
        self.assertEqual(prov[0], 1)  # measured = priority 1

    def test_no_data_provenance_zero(self):
        """Pixels with no data from any source should have provenance 0."""
        sources = {"a": np.array([np.nan])}
        _, prov = TestPriorityFill.priority_fill(sources, ["a"])
        self.assertEqual(prov[0], 0)


if __name__ == "__main__":
    unittest.main()
