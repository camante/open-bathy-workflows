"""Tests for bathy_main.py utility functions.

Pure-logic tests for path parsing, scoring, normalization, hashing,
IO manifest building, and SDB artifact discovery. No geo dependencies.
"""

import unittest
import json
import tempfile
from pathlib import Path


class TestStableHashStr(unittest.TestCase):

    def setUp(self):
        # _stable_hash_str is nested inside a class; reimport from module scope
        import bathy_main
        self.fn = bathy_main._stable_hash_str

    def test_deterministic(self):
        self.assertEqual(self.fn("hello", 12), self.fn("hello", 12))

    def test_length(self):
        h = self.fn("test_input", 8)
        self.assertEqual(len(h), 8)

    def test_different_inputs(self):
        self.assertNotEqual(self.fn("a", 12), self.fn("b", 12))


class TestIsProbablyPath(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._is_probably_path

    def test_unix_path(self):
        self.assertTrue(self.fn("/home/user/file.tif"))

    def test_relative_path(self):
        self.assertTrue(self.fn("rasters/depth.tif"))

    def test_filename_with_ext(self):
        self.assertTrue(self.fn("depth.tif"))

    def test_url_rejected(self):
        self.assertFalse(self.fn("https://example.com/file.tif"))

    def test_whitespace_rejected(self):
        self.assertFalse(self.fn("some command with spaces"))

    def test_empty_rejected(self):
        self.assertFalse(self.fn(""))
        self.assertFalse(self.fn(None))

    def test_short_no_ext(self):
        self.assertFalse(self.fn("justtext"))


class TestCollectPathsFromObj(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._collect_paths_from_obj

    def test_nested_dict(self):
        obj = {"a": "/path/to/file.tif", "b": {"c": "other.gpkg"}}
        out = []
        self.fn(obj, out)
        self.assertIn("/path/to/file.tif", out)
        self.assertIn("other.gpkg", out)

    def test_list(self):
        obj = ["/a/b.tif", "not a path", "/c/d.shp"]
        out = []
        self.fn(obj, out)
        self.assertIn("/a/b.tif", out)
        self.assertIn("/c/d.shp", out)

    def test_non_path_strings_excluded(self):
        obj = {"key": "just a word", "url": "https://example.com"}
        out = []
        self.fn(obj, out)
        self.assertEqual(len(out), 0)


class TestParsePathsFromCommand(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._parse_paths_from_command

    def test_output_flag(self):
        result = self.fn("waffles -O /tmp/out.tif --dem /data/dem.tif")
        self.assertIn("/tmp/out.tif", result["outputs"])
        self.assertIn("/data/dem.tif", result["inputs"])

    def test_equals_syntax(self):
        result = self.fn("tool --out=/path/output.tif --input=/data/input.csv")
        self.assertIn("/path/output.tif", result["outputs"])
        self.assertIn("/data/input.csv", result["inputs"])

    def test_empty_string(self):
        result = self.fn("")
        self.assertEqual(result["inputs"], [])
        self.assertEqual(result["outputs"], [])

    def test_no_flags(self):
        result = self.fn("echo hello world")
        self.assertEqual(result["inputs"], [])
        self.assertEqual(result["outputs"], [])


class TestScoreSdbCandidate(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._score_sdb_candidate

    def test_bad_tokens_negative(self):
        with tempfile.NamedTemporaryFile(suffix="_rgb_mask.tif") as f:
            score = self.fn(Path(f.name))
            self.assertLess(score, 0)

    def test_sdb_depth_scores_positive(self):
        with tempfile.NamedTemporaryFile(suffix="_sdb_depth_rf_10m.tif") as f:
            score = self.fn(Path(f.name))
            self.assertGreater(score, 5.0)  # sdb + depth + rf + 10m

    def test_sdb_beats_generic(self):
        with tempfile.NamedTemporaryFile(suffix="_sdb_depth.tif") as f1, \
             tempfile.NamedTemporaryFile(suffix="_output.tif") as f2:
            self.assertGreater(self.fn(Path(f1.name)), self.fn(Path(f2.name)))


class TestNormalizeCudemSourceName(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._normalize_cudem_source_name

    def test_empty(self):
        self.assertEqual(self.fn(""), "")
        self.assertEqual(self.fn(None), "")

    def test_lowercase(self):
        result = self.fn("CHART")
        self.assertEqual(result, result.lower())

    def test_strips(self):
        result = self.fn("  chart  ")
        self.assertFalse(result.startswith(" "))


class TestFindSdbDepthRaster(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main.find_sdb_depth_raster

    def test_nonexistent_dir(self):
        self.assertIsNone(self.fn(Path("/nonexistent/dir")))

    def test_with_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            sdb_dir = Path(td)
            depth_tif = sdb_dir / "sdb_depth.tif"
            depth_tif.write_bytes(b"fake raster")
            manifest = {"depth_raster": "sdb_depth.tif"}
            (sdb_dir / "artifacts_sdb.json").write_text(json.dumps(manifest))
            result = self.fn(sdb_dir)
            self.assertIsNotNone(result)
            self.assertEqual(result.name, "sdb_depth.tif")

    def test_no_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.fn(Path(td))
            self.assertIsNone(result)

    def test_manifest_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = {"depth_raster": "nonexistent.tif"}
            (Path(td) / "artifacts_sdb.json").write_text(json.dumps(manifest))
            result = self.fn(Path(td))
            self.assertIsNone(result)


class TestFindSdbLandMask(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main.find_sdb_land_mask

    def test_nonexistent_dir(self):
        self.assertIsNone(self.fn(Path("/nonexistent/dir")))

    def test_with_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            sdb_dir = Path(td)
            mask_tif = sdb_dir / "LAND_MASK_aligned.tif"
            mask_tif.write_bytes(b"fake")
            manifest = {"land_mask": "LAND_MASK_aligned.tif"}
            (sdb_dir / "artifacts_sdb.json").write_text(json.dumps(manifest))
            result = self.fn(sdb_dir)
            self.assertIsNotNone(result)


class TestBuildIoManifest(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main.build_io_manifest

    def test_empty_report(self):
        result = self.fn({})
        self.assertIn("inputs", result)
        self.assertIn("outputs", result)

    def test_with_recorded_paths(self):
        report = {
            "recorded_outputs": {
                "depth_raster": "/out/depth.tif",
            },
            "commands": [],
        }
        result = self.fn(report)
        self.assertIsInstance(result, dict)


class TestParseAoiBoundsDeg(unittest.TestCase):
    """Tests for _parse_aoi_bounds_deg (now delegates to canonical parser)."""

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._parse_aoi_bounds_deg

    def test_standard(self):
        w, e, s, n = self.fn("-78/-77/25/26")
        self.assertAlmostEqual(w, -78.0)
        self.assertAlmostEqual(e, -77.0)

    def test_garbage(self):
        self.assertIsNone(self.fn("not_an_aoi"))

    def test_auto_fix_reversed(self):
        result = self.fn("10/-10/30/20")
        self.assertIsNotNone(result)
        w, e, s, n = result
        self.assertLess(w, e)
        self.assertLess(s, n)


if __name__ == "__main__":
    unittest.main()
