"""Tests for s2_optics.py numerical and metadata functions.

Pure numpy tests — no rasterio/STAC/network dependencies required.
"""

import unittest
import warnings
import numpy as np


class TestSafeRintToUint8(unittest.TestCase):
    """Tests for _safe_rint_to_uint8."""

    def setUp(self):
        from s2_optics import _safe_rint_to_uint8
        self.fn = _safe_rint_to_uint8

    def test_basic_rounding(self):
        arr = np.array([0.4, 1.6, 127.5, 254.9])
        result = self.fn(arr)
        np.testing.assert_array_equal(result, [0, 2, 128, 255])
        self.assertEqual(result.dtype, np.uint8)

    def test_nan_filled(self):
        arr = np.array([np.nan, 5.0, np.nan, 10.0])
        result = self.fn(arr, fill=99)
        self.assertEqual(result[0], 99)
        self.assertEqual(result[2], 99)
        self.assertEqual(result[1], 5)
        self.assertEqual(result[3], 10)

    def test_clipping(self):
        arr = np.array([-10.0, 0.0, 300.0, 255.0])
        result = self.fn(arr)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[2], 255)
        self.assertEqual(result[3], 255)

    def test_all_nan(self):
        arr = np.array([np.nan, np.nan])
        result = self.fn(arr, fill=0)
        np.testing.assert_array_equal(result, [0, 0])

    def test_empty(self):
        arr = np.array([], dtype=np.float64)
        result = self.fn(arr)
        self.assertEqual(result.size, 0)
        self.assertEqual(result.dtype, np.uint8)


class TestNanmeanStackNoWarn(unittest.TestCase):
    """Tests for _nanmean_stack_no_warn."""

    def setUp(self):
        from s2_optics import _nanmean_stack_no_warn
        self.fn = _nanmean_stack_no_warn

    def test_basic_mean(self):
        stack = np.array([[[1.0, 2.0], [3.0, 4.0]],
                          [[5.0, 6.0], [7.0, 8.0]]])  # (2, 2, 2)
        result = self.fn(stack)
        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_with_nans(self):
        stack = np.array([[[1.0, np.nan], [3.0, 4.0]],
                          [[np.nan, 6.0], [7.0, 8.0]]])
        result = self.fn(stack)
        self.assertAlmostEqual(result[0, 0], 1.0, places=5)
        self.assertAlmostEqual(result[0, 1], 6.0, places=5)
        self.assertAlmostEqual(result[1, 0], 5.0, places=5)

    def test_all_nan_column(self):
        """All-NaN across axis=0 should produce NaN without warnings."""
        stack = np.array([[[np.nan], [1.0]],
                          [[np.nan], [2.0]]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = self.fn(stack)
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertAlmostEqual(result[1, 0], 1.5, places=5)

    def test_output_dtype(self):
        stack = np.ones((3, 4, 5), dtype=np.float64)
        result = self.fn(stack)
        self.assertEqual(result.dtype, np.float32)


class TestNanmedianStackNoWarn(unittest.TestCase):
    """Tests for _nanmedian_stack_no_warn."""

    def setUp(self):
        from s2_optics import _nanmedian_stack_no_warn
        self.fn = _nanmedian_stack_no_warn

    def test_basic_median(self):
        stack = np.array([[[1.0], [4.0]],
                          [[2.0], [5.0]],
                          [[3.0], [6.0]]])  # (3, 2, 1)
        result = self.fn(stack)
        np.testing.assert_array_almost_equal(result, [[2.0], [5.0]])

    def test_all_nan_no_warning(self):
        stack = np.array([[[np.nan]], [[np.nan]]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = self.fn(stack)
        self.assertTrue(np.isnan(result[0, 0]))


class TestComputeBrightness(unittest.TestCase):
    """Tests for compute_brightness."""

    def setUp(self):
        from s2_optics import compute_brightness
        self.fn = compute_brightness

    def test_mean_of_bands(self):
        b02 = np.array([0.1, 0.2, 0.3])
        b03 = np.array([0.2, 0.3, 0.4])
        b04 = np.array([0.3, 0.4, 0.5])
        result = self.fn(b02, b03, b04)
        expected = np.array([0.2, 0.3, 0.4])
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d(self):
        shape = (10, 10)
        b02 = np.ones(shape) * 0.1
        b03 = np.ones(shape) * 0.2
        b04 = np.ones(shape) * 0.3
        result = self.fn(b02, b03, b04)
        np.testing.assert_array_almost_equal(result, np.full(shape, 0.2))


class TestDeepWaterMask(unittest.TestCase):
    """Tests for deep_water_mask."""

    def setUp(self):
        from s2_optics import deep_water_mask
        self.fn = deep_water_mask

    def test_basic(self):
        b08 = np.array([0.01, 0.05, 0.02, np.nan])
        bright = np.array([0.10, 0.10, 0.20, 0.05])
        mask = self.fn(b08, bright, nir_max=0.03, bright_max=0.15)
        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_with_b02_filter(self):
        b08 = np.array([0.01, 0.01])
        bright = np.array([0.05, 0.05])
        b02 = np.array([0.02, 0.10])
        mask = self.fn(b08, bright, nir_max=0.03, bright_max=0.15,
                       b02=b02, b02_max=0.05)
        np.testing.assert_array_equal(mask, [True, False])


class TestMetadataExtractors(unittest.TestCase):
    """Tests for STAC metadata extraction helpers."""

    def test_month_from_iso(self):
        from s2_optics import month_from_iso
        self.assertEqual(month_from_iso("2024-01-15T12:00:00Z"), 1)
        self.assertEqual(month_from_iso("2024-12-01T00:00:00Z"), 12)
        self.assertEqual(month_from_iso("2024-07-20"), 7)

    def test_extract_datetime(self):
        from s2_optics import extract_datetime
        item = {"properties": {"datetime": "2024-06-15T10:30:00Z"}}
        self.assertEqual(extract_datetime(item), "2024-06-15T10:30:00Z")

        item_fallback = {"properties": {"start_datetime": "2024-06-15T10:30:00Z"}}
        self.assertEqual(extract_datetime(item_fallback), "2024-06-15T10:30:00Z")

        self.assertIsNone(extract_datetime({"properties": {}}))
        self.assertIsNone(extract_datetime({}))

    def test_extract_cloud(self):
        from s2_optics import extract_cloud
        self.assertEqual(extract_cloud({"properties": {"eo:cloud_cover": 15.5}}), 15.5)
        self.assertEqual(extract_cloud({"properties": {}}), 100.0)
        self.assertEqual(extract_cloud({}), 100.0)
        self.assertEqual(extract_cloud({"properties": {"eo:cloud_cover": "bad"}}), 100.0)

    def test_extract_tile(self):
        from s2_optics import extract_tile
        item = {"properties": {"s2:mgrs_tile": "15TXM"}}
        self.assertEqual(extract_tile(item), "15TXM")

        # From item ID fallback — needs tile between underscores
        item2 = {"properties": {}, "id": "S2A_MSIL2A_20240615T103021_N0510_R108_T15TXM_20240615T143024"}
        tile = extract_tile(item2)
        self.assertEqual(tile, "15TXM")


class TestDateHelpers(unittest.TestCase):
    """Tests for date utility functions."""

    def test_days_in_month(self):
        from s2_optics import _days_in_month
        self.assertEqual(_days_in_month(2024, 2), 29)  # leap year
        self.assertEqual(_days_in_month(2023, 2), 28)
        self.assertEqual(_days_in_month(2024, 1), 31)
        self.assertEqual(_days_in_month(2024, 4), 30)

    def test_add_months(self):
        from s2_optics import _add_months
        self.assertEqual(_add_months((2024, 1, 15), 1), (2024, 2, 15))
        self.assertEqual(_add_months((2024, 12, 15), 1), (2025, 1, 15))
        # Day clamping for short months
        result = _add_months((2024, 1, 31), 1)
        self.assertEqual(result, (2024, 2, 29))  # 2024 is leap year


class TestExtractOrbit(unittest.TestCase):
    """Tests for extract_orbit."""

    def test_from_properties(self):
        from s2_optics import extract_orbit
        item = {"properties": {"sat:relative_orbit": 42}}
        self.assertEqual(extract_orbit(item), "042")

    def test_from_id_fallback(self):
        from s2_optics import extract_orbit
        item = {"properties": {}, "id": "S2A_MSIL2A_20240615T103021_N0510_R108_T15TXM"}
        result = extract_orbit(item)
        self.assertEqual(result, "108")


if __name__ == "__main__":
    unittest.main()
