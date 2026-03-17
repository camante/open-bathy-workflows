"""Tests for river_network.py numerical and utility functions.

Pure tests — skips if geopandas not available.
"""

import unittest
import hashlib

try:
    import river_network
    _HAS_GEO = True
except (ImportError, ModuleNotFoundError):
    _HAS_GEO = False

_skip_msg = "river_network requires geopandas/shapely"


@unittest.skipUnless(_HAS_GEO, _skip_msg)
class TestAutoUtmEpsg(unittest.TestCase):
    """Tests for _auto_utm_epsg_from_lonlat."""

    def setUp(self):
        from river_network import _auto_utm_epsg_from_lonlat
        self.fn = _auto_utm_epsg_from_lonlat

    def test_new_york(self):
        # NYC is ~lon -74, lat 40 → UTM zone 18N → EPSG 32618
        self.assertEqual(self.fn(-74.0, 40.0), 32618)

    def test_london(self):
        # London is ~lon 0, lat 51 → UTM zone 31N → EPSG 32631
        self.assertEqual(self.fn(0.0, 51.0), 32631)

    def test_southern_hemisphere(self):
        # Sydney is ~lon 151, lat -34 → UTM zone 56S → EPSG 32756
        self.assertEqual(self.fn(151.0, -34.0), 32756)

    def test_dateline(self):
        # lon 179 → zone 60
        result = self.fn(179.0, 10.0)
        self.assertEqual(result, 32660)

    def test_negative_lon(self):
        # lon -180 → zone 1
        result = self.fn(-180.0, 45.0)
        self.assertEqual(result, 32601)


@unittest.skipUnless(_HAS_GEO, _skip_msg)
class TestAoiHash(unittest.TestCase):
    """Tests for _aoi_hash."""

    def setUp(self):
        from river_network import _aoi_hash
        self.fn = _aoi_hash

    def test_deterministic(self):
        aoi = (-78.0, -77.0, 25.0, 26.0)
        h1 = self.fn(aoi)
        h2 = self.fn(aoi)
        self.assertEqual(h1, h2)

    def test_length(self):
        h = self.fn((-78.0, -77.0, 25.0, 26.0))
        self.assertEqual(len(h), 10)

    def test_different_aois(self):
        h1 = self.fn((-78.0, -77.0, 25.0, 26.0))
        h2 = self.fn((-79.0, -78.0, 25.0, 26.0))
        self.assertNotEqual(h1, h2)


@unittest.skipUnless(_HAS_GEO, _skip_msg)
class TestAoiCenter(unittest.TestCase):
    """Tests for _aoi_center."""

    def setUp(self):
        from river_network import _aoi_center
        self.fn = _aoi_center

    def test_basic(self):
        lon, lat = self.fn((-80.0, -78.0, 25.0, 27.0))
        self.assertAlmostEqual(lon, -79.0)
        self.assertAlmostEqual(lat, 26.0)

    def test_crosses_zero(self):
        lon, lat = self.fn((-1.0, 1.0, -1.0, 1.0))
        self.assertAlmostEqual(lon, 0.0)
        self.assertAlmostEqual(lat, 0.0)


@unittest.skipUnless(_HAS_GEO, _skip_msg)
class TestGuessHydroRiversRegion(unittest.TestCase):
    """Tests for _guess_hydrorivers_region."""

    def setUp(self):
        from river_network import _guess_hydrorivers_region
        self.fn = _guess_hydrorivers_region

    def test_north_america(self):
        # US East coast
        self.assertEqual(self.fn((-80.0, -78.0, 30.0, 35.0)), "na")

    def test_europe(self):
        self.assertEqual(self.fn((0.0, 5.0, 48.0, 52.0)), "eu")

    def test_australia(self):
        self.assertEqual(self.fn((140.0, 150.0, -38.0, -30.0)), "au")

    def test_south_america(self):
        self.assertEqual(self.fn((-60.0, -50.0, -30.0, -20.0)), "sa")

    def test_africa(self):
        self.assertEqual(self.fn((20.0, 30.0, -5.0, 5.0)), "af")

    def test_asia(self):
        self.assertEqual(self.fn((100.0, 110.0, 15.0, 25.0)), "as")


@unittest.skipUnless(_HAS_GEO, _skip_msg)
class TestParseAoiDelegation(unittest.TestCase):
    """Tests that _parse_aoi delegates to canonical parser."""

    def setUp(self):
        from river_network import _parse_aoi
        self.fn = _parse_aoi

    def test_standard_aoi(self):
        w, e, s, n = self.fn("-78/-77/25/26")
        self.assertAlmostEqual(w, -78.0)
        self.assertAlmostEqual(e, -77.0)
        self.assertAlmostEqual(s, 25.0)
        self.assertAlmostEqual(n, 26.0)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.fn("garbage")

    def test_too_few_parts(self):
        with self.assertRaises(ValueError):
            self.fn("1/2/3")


@unittest.skipUnless(_HAS_GEO, _skip_msg)
class TestFingerprintPath(unittest.TestCase):
    """Tests for fingerprint_path."""

    def test_nonexistent_returns_none(self):
        from river_network import fingerprint_path
        from pathlib import Path
        result = fingerprint_path(Path("/nonexistent/file.shp"))
        self.assertIsNone(result)


@unittest.skipUnless(_HAS_GEO, _skip_msg)
class TestSnapKey(unittest.TestCase):
    """Tests for _snap_key grid snapping."""

    def setUp(self):
        from river_network import _snap_key
        self.fn = _snap_key

    def test_nearby_points_same_key(self):
        """Points within snap distance should hash to the same key."""
        snap_m = 5.0
        # Two points ~1m apart
        k1 = self.fn(100.0, 200.0, snap_m)
        k2 = self.fn(101.0, 201.0, snap_m)
        self.assertEqual(k1, k2)

    def test_distant_points_different_key(self):
        """Points far apart should hash to different keys."""
        snap_m = 5.0
        k1 = self.fn(100.0, 200.0, snap_m)
        k2 = self.fn(200.0, 300.0, snap_m)
        self.assertNotEqual(k1, k2)

    def test_deterministic(self):
        k1 = self.fn(42.5, 99.3, 10.0)
        k2 = self.fn(42.5, 99.3, 10.0)
        self.assertEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
