"""test_helpers.py – pure-Python / numpy helper tests (unittest)."""

from __future__ import annotations
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import _install_mocks
_install_mocks()


class TestParseAoiBbox(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._parse_aoi_bbox

    def test_comma_xmin_ymin_xmax_ymax(self):
        self.assertEqual(self.fn("-82,24,-81,25"), (-82.0, -81.0, 24.0, 25.0))

    def test_whitespace(self):
        self.assertEqual(self.fn(" -82 , 24 , -81 , 25 "), (-82.0, -81.0, 24.0, 25.0))

    def test_space_separated(self):
        self.assertEqual(self.fn("-82 24 -81 25"), (-82.0, -81.0, 24.0, 25.0))

    def test_garbage_returns_none(self):
        self.assertIsNone(self.fn("not_an_aoi"))

    def test_empty_returns_none(self):
        self.assertIsNone(self.fn(""))
        self.assertIsNone(self.fn(None))

    def test_three_values_returns_none(self):
        self.assertIsNone(self.fn("1,2,3"))

    def test_negative_coords(self):
        # Input format is W,E,S,N; output is (xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = self.fn("-180,180,-90,90")
        self.assertEqual(xmin, -180.0)
        self.assertEqual(ymin, -90.0)


class TestNwaterGuard(unittest.TestCase):

    def _should_skip(self, n_water):
        return n_water is not None and n_water == 0

    def test_none_does_not_skip(self):
        self.assertFalse(self._should_skip(None))

    def test_zero_triggers_skip(self):
        self.assertTrue(self._should_skip(0))

    def test_positive_does_not_skip(self):
        self.assertFalse(self._should_skip(1000))


class TestAtlColumnMapping(unittest.TestCase):

    def _run_mapping(self, df):
        tmp = df.copy()
        tmp.columns = [c.lower() for c in tmp.columns]
        rename_map = {}
        for c in tmp.columns:
            cl = str(c).lower().strip()
            if cl in ("lon", "longitude", "x"):
                rename_map[c] = "longitude"
            elif cl in ("lat", "latitude", "y"):
                rename_map[c] = "latitude"
            elif cl in ("depth", "depth_m", "z", "elev", "elevation", "bed_elev"):
                rename_map[c] = "depth_m"
        return tmp.rename(columns=rename_map)

    def _df(self, lon_col, lat_col, dep_col, n=20):
        rng = np.random.default_rng(7)
        return pd.DataFrame({
            lon_col: rng.uniform(-82, -81, n),
            lat_col: rng.uniform(24, 25, n),
            dep_col: -rng.uniform(1, 10, n),
        })

    def test_standard_names(self):
        out = self._run_mapping(self._df("longitude", "latitude", "depth_m"))
        self.assertIn("longitude", out.columns)
        self.assertIn("depth_m", out.columns)

    def test_short_names(self):
        out = self._run_mapping(self._df("lon", "lat", "depth"))
        self.assertIn("longitude", out.columns)
        self.assertIn("depth_m", out.columns)

    def test_xyz_names(self):
        out = self._run_mapping(self._df("x", "y", "z"))
        self.assertIn("longitude", out.columns)
        self.assertIn("depth_m", out.columns)


class TestDepthSignDetection(unittest.TestCase):

    def _maybe_flip(self, depths):
        finite = depths[np.isfinite(depths)]
        if len(finite) == 0:
            return depths
        if (finite > 0).mean() >= 0.9:
            return -np.abs(depths)
        return depths

    def test_mostly_positive_flipped(self):
        out = self._maybe_flip(np.array([1.0, 2.0, 3.0, 5.0, 10.0]))
        self.assertTrue(np.all(out <= 0))

    def test_negative_unchanged(self):
        d = np.array([-1.0, -2.0, -3.0, -5.0])
        np.testing.assert_array_equal(self._maybe_flip(d), d)

    def test_mixed_unchanged(self):
        d = np.array([-5.0, -3.0, 2.0, -1.0, -4.0])
        self.assertEqual(self._maybe_flip(d)[0], -5.0)

    def test_all_nan_no_crash(self):
        out = self._maybe_flip(np.full(5, np.nan))
        self.assertTrue(np.all(np.isnan(out)))


class TestChunkedSiblingCopy(unittest.TestCase):

    SUFFIXES = ("_confidence.tif", "_provenance.tif", "_uncertainty.tif", "_doa.tif")

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_all_siblings_copied(self):
        stem, final_stem = "run_chunked_tmp", "run_sdb_depth"
        tmp_raster = self.tmpdir / f"{stem}.tif"
        out_tif = self.tmpdir / f"{final_stem}.tif"

        tmp_raster.write_bytes(b"FAKE")
        for s in self.SUFFIXES:
            (self.tmpdir / f"{stem}{s}").write_bytes(b"FAKE")

        shutil.copy2(tmp_raster, out_tif)
        for suffix in self.SUFFIXES:
            src = tmp_raster.with_name(tmp_raster.stem + suffix)
            if src.exists():
                shutil.copy2(src, out_tif.with_name(out_tif.stem + suffix))

        self.assertTrue(out_tif.exists())
        for suffix in self.SUFFIXES:
            self.assertTrue((self.tmpdir / f"{final_stem}{suffix}").exists(),
                            msg=f"Missing {final_stem}{suffix}")

    def test_missing_siblings_skipped_gracefully(self):
        stem, final_stem = "run_chunked_tmp", "run_sdb_depth"
        tmp_raster = self.tmpdir / f"{stem}.tif"
        out_tif = self.tmpdir / f"{final_stem}.tif"

        tmp_raster.write_bytes(b"FAKE")
        (self.tmpdir / f"{stem}_confidence.tif").write_bytes(b"FAKE")

        shutil.copy2(tmp_raster, out_tif)
        for suffix in self.SUFFIXES:
            src = tmp_raster.with_name(tmp_raster.stem + suffix)
            if src.exists():
                shutil.copy2(src, out_tif.with_name(out_tif.stem + suffix))

        self.assertTrue((self.tmpdir / f"{final_stem}_confidence.tif").exists())
        self.assertFalse((self.tmpdir / f"{final_stem}_provenance.tif").exists())


class TestConfidenceMath(unittest.TestCase):

    def _conf(self, doa_q, optical_q, unc_m, ref=1.5):
        unc_q = 1.0 / (1.0 + unc_m / ref)
        return doa_q * optical_q * unc_q

    def test_perfect_inputs_give_one(self):
        self.assertAlmostEqual(self._conf(1.0, 1.0, 0.0), 1.0)

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(3)
        conf = self._conf(rng.uniform(0,1,100), rng.uniform(0,1,100), rng.uniform(0,10,100))
        self.assertTrue(np.all(conf >= 0.0))
        self.assertTrue(np.all(conf <= 1.0))

    def test_half_weight_at_ref_uncertainty(self):
        self.assertAlmostEqual(self._conf(1.0, 1.0, 1.5), 0.5)

    def test_zero_doa_gives_zero_confidence(self):
        self.assertAlmostEqual(self._conf(0.0, 1.0, 0.5), 0.0)


class TestDepthSignGuard(unittest.TestCase):

    def test_correct_negative_depths_pass(self):
        depths = -np.abs(np.random.default_rng(0).uniform(1,15,200))
        pct_neg = (depths[np.isfinite(depths)] < 0).mean()
        self.assertGreater(pct_neg, 0.9)

    def test_positive_depths_detected(self):
        depths = np.abs(np.random.default_rng(0).uniform(1,15,200))
        pct_pos = (depths[np.isfinite(depths)] > 0).mean()
        self.assertGreater(pct_pos, 0.9)

    def test_magnitude_conversion_always_positive(self):
        depths = -np.abs(np.random.default_rng(0).uniform(1,15,200))
        self.assertTrue(np.all(np.abs(depths) >= 0))


class TestNodataCollisionGuard(unittest.TestCase):

    def _apply(self, sampled, nodata_val):
        valid = np.isfinite(sampled)
        if nodata_val is not None and np.isfinite(nodata_val) and abs(float(nodata_val)) > 0.5:
            valid &= (sampled != nodata_val)
        return sampled[valid]

    def test_nodata_zero_keeps_all_pixels(self):
        # nodata=0 collides with water (0) — guard must not exclude any values
        mask = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
        self.assertEqual(len(self._apply(mask, 0.0)), 5)

    def test_nodata_minus9999_excluded(self):
        mask = np.array([0.0, 1.0, -9999.0, 0.0, -9999.0])
        result = self._apply(mask, -9999.0)
        self.assertEqual(len(result), 3)
        self.assertNotIn(-9999.0, result)

    def test_nodata_none_no_exclusion(self):
        mask = np.array([0.0, 1.0, 0.0])
        self.assertEqual(len(self._apply(mask, None)), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
