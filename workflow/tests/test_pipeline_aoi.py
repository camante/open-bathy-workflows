"""Tests for pipeline/aoi.py — the canonical AOI parsing module.

Validates parse_aoi_wesn, parse_aoi_bbox, and utility functions.
"""

import unittest
import math


class TestParseAoiWesn(unittest.TestCase):
    """Tests for parse_aoi_wesn (canonical parser)."""

    def setUp(self):
        from pipeline.aoi import parse_aoi_wesn
        self.fn = parse_aoi_wesn

    def test_slash_separated(self):
        w, e, s, n = self.fn("-78/-77/25/26")
        self.assertAlmostEqual(w, -78.0)
        self.assertAlmostEqual(e, -77.0)
        self.assertAlmostEqual(s, 25.0)
        self.assertAlmostEqual(n, 26.0)

    def test_comma_separated(self):
        w, e, s, n = self.fn("-78,-77,25,26")
        self.assertAlmostEqual(w, -78.0)
        self.assertAlmostEqual(e, -77.0)

    def test_space_separated(self):
        w, e, s, n = self.fn("-78 -77 25 26")
        self.assertAlmostEqual(w, -78.0)
        self.assertAlmostEqual(e, -77.0)

    def test_mixed_separators(self):
        w, e, s, n = self.fn(" -78 , -77 / 25  26 ")
        self.assertAlmostEqual(w, -78.0)

    def test_none_returns_none(self):
        self.assertIsNone(self.fn(None))

    def test_empty_returns_none(self):
        self.assertIsNone(self.fn(""))
        self.assertIsNone(self.fn("   "))

    def test_too_few_parts(self):
        self.assertIsNone(self.fn("1/2/3"))

    def test_too_many_parts(self):
        self.assertIsNone(self.fn("1/2/3/4/5"))

    def test_non_numeric(self):
        self.assertIsNone(self.fn("a/b/c/d"))

    def test_auto_fix_reversed_lon(self):
        """If E < W, should auto-swap."""
        w, e, s, n = self.fn("10/-10/25/26")
        self.assertAlmostEqual(w, -10.0)
        self.assertAlmostEqual(e, 10.0)

    def test_auto_fix_reversed_lat(self):
        """If N < S, should auto-swap."""
        w, e, s, n = self.fn("-78/-77/30/20")
        self.assertAlmostEqual(s, 20.0)
        self.assertAlmostEqual(n, 30.0)

    def test_zero_width_returns_none(self):
        self.assertIsNone(self.fn("10/10/25/26"))

    def test_zero_height_returns_none(self):
        self.assertIsNone(self.fn("-78/-77/25/25"))

    def test_strict_mode_raises(self):
        with self.assertRaises(ValueError):
            self.fn("garbage", strict=True)

    def test_strict_mode_none_raises(self):
        with self.assertRaises(ValueError):
            self.fn(None, strict=True)

    def test_strict_mode_success(self):
        result = self.fn("-78/-77/25/26", strict=True)
        self.assertIsNotNone(result)

    def test_decimal_precision(self):
        w, e, s, n = self.fn("-77.75/-77.50/25.25/25.75")
        self.assertAlmostEqual(w, -77.75)
        self.assertAlmostEqual(e, -77.50)
        self.assertAlmostEqual(s, 25.25)
        self.assertAlmostEqual(n, 25.75)


class TestParseAoiBbox(unittest.TestCase):
    """Tests for parse_aoi_bbox (OGC convention)."""

    def setUp(self):
        from pipeline.aoi import parse_aoi_bbox
        self.fn = parse_aoi_bbox

    def test_returns_wsne_order(self):
        """parse_aoi_bbox returns (W, S, E, N) = (minx, miny, maxx, maxy)."""
        result = self.fn("-78/-77/25/26")
        w, s, e, n = result
        self.assertAlmostEqual(w, -78.0)
        self.assertAlmostEqual(s, 25.0)
        self.assertAlmostEqual(e, -77.0)
        self.assertAlmostEqual(n, 26.0)

    def test_none_input(self):
        self.assertIsNone(self.fn(None))


class TestBboxToAoiStr(unittest.TestCase):
    """Tests for bbox_to_aoi_str."""

    def test_roundtrip(self):
        from pipeline.aoi import parse_aoi_bbox, bbox_to_aoi_str
        bbox = parse_aoi_bbox("-78/-77/25/26")
        s = bbox_to_aoi_str(bbox)
        # Should produce "W/E/S/N"
        self.assertEqual(s, "-78.0/-77.0/25.0/26.0")


class TestAoiCentroid(unittest.TestCase):
    """Tests for aoi_centroid."""

    def test_basic(self):
        from pipeline.aoi import aoi_centroid
        # bbox is (W, S, E, N)
        lon, lat = aoi_centroid((-80.0, 24.0, -78.0, 26.0))
        self.assertAlmostEqual(lon, -79.0)
        self.assertAlmostEqual(lat, 25.0)


class TestExpandBboxKm(unittest.TestCase):
    """Tests for expand_bbox_km."""

    def test_expands(self):
        from pipeline.aoi import expand_bbox_km
        bbox = (-80.0, 24.0, -78.0, 26.0)
        expanded = expand_bbox_km(bbox, 10.0)
        self.assertLess(expanded[0], bbox[0])  # W moved west
        self.assertLess(expanded[1], bbox[1])  # S moved south
        self.assertGreater(expanded[2], bbox[2])  # E moved east
        self.assertGreater(expanded[3], bbox[3])  # N moved north

    def test_zero_buffer(self):
        from pipeline.aoi import expand_bbox_km
        bbox = (-80.0, 24.0, -78.0, 26.0)
        self.assertEqual(expand_bbox_km(bbox, 0.0), bbox)


class TestExpandBboxFrac(unittest.TestCase):
    """Tests for expand_bbox_frac."""

    def test_ten_percent(self):
        from pipeline.aoi import expand_bbox_frac
        bbox = (-80.0, 24.0, -78.0, 26.0)
        expanded = expand_bbox_frac(bbox, 0.10)
        # Width is 2 deg, 10% = 0.2, half on each side = 0.1
        self.assertAlmostEqual(expanded[0], -80.1)
        self.assertAlmostEqual(expanded[2], -77.9)

    def test_none_frac(self):
        from pipeline.aoi import expand_bbox_frac
        bbox = (-80.0, 24.0, -78.0, 26.0)
        self.assertEqual(expand_bbox_frac(bbox, None), bbox)

    def test_zero_frac(self):
        from pipeline.aoi import expand_bbox_frac
        bbox = (-80.0, 24.0, -78.0, 26.0)
        self.assertEqual(expand_bbox_frac(bbox, 0.0), bbox)


class TestUtmEpsg(unittest.TestCase):
    """Tests for utm_epsg_from_lonlat."""

    def test_north(self):
        from pipeline.aoi import utm_epsg_from_lonlat
        self.assertEqual(utm_epsg_from_lonlat(-74.0, 40.0), "EPSG:32618")

    def test_south(self):
        from pipeline.aoi import utm_epsg_from_lonlat
        self.assertEqual(utm_epsg_from_lonlat(151.0, -34.0), "EPSG:32756")


class TestBufferAoi(unittest.TestCase):
    """Tests for buffer_aoi."""

    def test_zero_buffer(self):
        from pipeline.aoi import buffer_aoi
        result = buffer_aoi("-78/-77/25/26", buf_deg=0)
        parts = [float(x) for x in result.split("/")]
        self.assertAlmostEqual(parts[0], -78.0)
        self.assertAlmostEqual(parts[1], -77.0)

    def test_positive_buffer(self):
        from pipeline.aoi import buffer_aoi
        result = buffer_aoi("-78/-77/25/26", buf_deg=0.1)
        parts = [float(x) for x in result.split("/")]
        self.assertAlmostEqual(parts[0], -78.1, places=5)
        self.assertAlmostEqual(parts[1], -76.9, places=5)

    def test_invalid_raises(self):
        from pipeline.aoi import buffer_aoi
        with self.assertRaises(ValueError):
            buffer_aoi("garbage", buf_deg=0.1)


if __name__ == "__main__":
    unittest.main()
