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


class TestSupportDistanceDensityGuidance(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._compute_support_distance_density_guidance

    def test_guidance_influence_increases_with_distance(self):
        import numpy as np
        locked = np.zeros((7, 7), dtype=bool)
        auth = np.full((7, 7), np.nan, dtype=np.float32)
        locked[3, 3] = True
        auth[3, 3] = -5.0
        dist_m, density, influence, nearest = self.fn(
            locked, auth, pixel_size_m=10.0, support_decay_m=20.0, density_radius_m=20.0
        )
        self.assertEqual(float(dist_m[3, 3]), 0.0)
        self.assertAlmostEqual(float(influence[3, 3]), 0.0, places=6)
        self.assertGreater(float(influence[0, 0]), float(influence[3, 4]))
        self.assertTrue(np.isfinite(nearest[0, 0]))

    def test_density_reduces_guidance_influence_for_same_distance(self):
        import numpy as np
        locked_sparse = np.zeros((9, 9), dtype=bool)
        auth_sparse = np.full((9, 9), np.nan, dtype=np.float32)
        locked_sparse[4, 4] = True
        auth_sparse[4, 4] = -5.0

        locked_dense = locked_sparse.copy()
        auth_dense = auth_sparse.copy()
        locked_dense[4, 3] = True
        locked_dense[4, 5] = True
        auth_dense[4, 3] = -5.0
        auth_dense[4, 5] = -5.0

        _, _, influence_sparse, _ = self.fn(
            locked_sparse, auth_sparse, pixel_size_m=10.0, support_decay_m=30.0, density_radius_m=30.0
        )
        _, _, influence_dense, _ = self.fn(
            locked_dense, auth_dense, pixel_size_m=10.0, support_decay_m=30.0, density_radius_m=30.0
        )
        self.assertLess(float(influence_dense[3, 4]), float(influence_sparse[3, 4]))




class TestRiverAnchorSupportFields(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._compute_river_anchor_support_fields

    def test_scaffold_confidence_increases_away_from_anchor(self):
        import numpy as np
        anchor = np.zeros((9, 9), dtype=bool)
        anchor[4, 4] = True
        domain = np.ones((9, 9), dtype=bool)
        guide = np.ones((9, 9), dtype=np.float32)
        dist_m, density, conf = self.fn(anchor, guide, domain, pixel_size_m=10.0, density_radius_m=20.0, scaffold_transition_m=30.0)
        self.assertEqual(float(dist_m[4, 4]), 0.0)
        self.assertAlmostEqual(float(conf[4, 4]), 0.0, places=6)
        self.assertGreater(float(conf[0, 0]), float(conf[4, 5]))
        self.assertLess(float(density[0, 0]), float(density[4, 4]))

    def test_dense_anchor_support_reduces_scaffold_confidence(self):
        import numpy as np
        domain = np.ones((11, 11), dtype=bool)
        guide = np.ones((11, 11), dtype=np.float32)
        sparse = np.zeros((11, 11), dtype=bool)
        sparse[5, 5] = True
        dense = sparse.copy()
        dense[5, 4] = True
        dense[5, 6] = True
        _, _, conf_sparse = self.fn(sparse, guide, domain, pixel_size_m=10.0, density_radius_m=30.0, scaffold_transition_m=40.0)
        _, _, conf_dense = self.fn(dense, guide, domain, pixel_size_m=10.0, density_radius_m=30.0, scaffold_transition_m=40.0)
        self.assertLess(float(conf_dense[4, 5]), float(conf_sparse[4, 5]))


class TestCoastalSdbSupportConfidence(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._compute_coastal_sdb_support_confidence

    def test_confidence_increases_with_optical_stability_and_distance(self):
        import numpy as np
        domain = np.ones((7, 7), dtype=bool)
        gw = np.zeros((7, 7), dtype=np.float32)
        gw[3, 3] = 1.0
        ti = np.zeros((7, 7), dtype=np.uint8)
        ti[3, 3] = 1
        dist = np.full((7, 7), 500.0, dtype=np.float32)
        dist[3, 3] = 0.0
        density = np.zeros((7, 7), dtype=np.float32)
        conf = self.fn(domain, gw, ti, dist, density, support_transition_m=200.0)
        self.assertGreater(float(conf[3, 3]), 0.0)
        self.assertGreater(float(conf[3, 3]), float(conf[0, 0]))

    def test_dense_authoritative_support_reduces_coastal_confidence(self):
        import numpy as np
        domain = np.ones((5, 5), dtype=bool)
        gw = np.ones((5, 5), dtype=np.float32)
        ti = np.ones((5, 5), dtype=np.uint8)
        dist = np.full((5, 5), 300.0, dtype=np.float32)
        sparse_density = np.zeros((5, 5), dtype=np.float32)
        dense_density = np.full((5, 5), 0.8, dtype=np.float32)
        conf_sparse = self.fn(domain, gw, ti, dist, sparse_density, support_transition_m=200.0)
        conf_dense = self.fn(domain, gw, ti, dist, dense_density, support_transition_m=200.0)
        self.assertLess(float(conf_dense[2, 2]), float(conf_sparse[2, 2]))

class TestAuthoritativePassthroughArgs(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._authoritative_passthrough_args

    def test_sdb_passthrough_uses_existing_authoritative_base(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            auth = Path(td) / "authoritative_base.tif"
            auth.write_bytes(b"x")
            cfg = SimpleNamespace(authoritative_base=auth, river_authoritative_bed=None)
            args = self.fn(cfg, for_river=False)
            self.assertIn(f"--authoritative-base={auth}", args)
            self.assertIn("--no-authoritative-base-auto", args)

    def test_river_passthrough_prefers_explicit_bed(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            auth = Path(td) / "authoritative_base.tif"
            bed = Path(td) / "river_bed.tif"
            auth.write_bytes(b"x")
            bed.write_bytes(b"x")
            cfg = SimpleNamespace(authoritative_base=auth, river_authoritative_bed=bed)
            args = self.fn(cfg, for_river=True)
            self.assertIn(f"--authoritative-bed-raster={bed}", args)
            self.assertNotIn(f"--authoritative-bed-raster={auth}", args)
            self.assertIn("--no-authoritative-bed-auto", args)


class TestExplicitFinalOutputsManifest(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._write_explicit_final_outputs_manifest

    def test_manifest_records_selected_final_outputs(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            native = out_dir / "combined" / "native.tif"
            user = out_dir / "deliver" / "user.tif"
            prov = out_dir / "combined" / "prov.tif"
            native.parent.mkdir(parents=True, exist_ok=True)
            user.parent.mkdir(parents=True, exist_ok=True)
            prov.parent.mkdir(parents=True, exist_ok=True)
            native.write_bytes(b"x")
            user.write_bytes(b"x")
            prov.write_bytes(b"x")
            cfg = SimpleNamespace(out_dir=out_dir, authoritative_base=None)
            report = {"authoritative_base": {"outputs": {"aligned_authoritative_base": None}}}
            out_json = self.fn(cfg, report, final_native=native, final_for_user=user, final_provenance=prov)
            payload = json.loads(Path(out_json).read_text())
            self.assertEqual(payload["final_depth_native"], str(native))
            self.assertEqual(payload["final_depth_user"], str(user))
            self.assertEqual(payload["selected_final_depth"], str(user))
            self.assertEqual(payload["selected_final_provenance"], str(prov))


class TestAuthoritativeCacheReceipt(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._write_authoritative_cache_receipt

    def test_receipt_records_cache_and_passthrough(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            report = {
                "authoritative_base_auto": {
                    "mode": "auto_cudem",
                    "cache_key": "abc123",
                    "cache_hit": True,
                    "cache_dir": str(out_dir / "cache"),
                    "authoritative_base": str(out_dir / "authoritative_base.tif"),
                    "shared_cache_reuse": {"tile_downloads": {"reused_existing": 2}},
                    "downstream_child_passthrough": {"sdb": {"auto_materialization_disabled": True}},
                }
            }
            cfg = SimpleNamespace(out_dir=out_dir, authoritative_base=None)
            out_json = self.fn(cfg, report)
            payload = json.loads(Path(out_json).read_text())
            self.assertEqual(payload["cache_key"], "abc123")
            self.assertTrue(payload["cache_hit"])
            self.assertIn("sdb", payload["downstream_child_passthrough"])
