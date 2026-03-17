"""Tests for sdb_context.py — SDBRunContext dataclass."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace


class TestSDBRunContext(unittest.TestCase):
    """Tests for SDBRunContext construction and helpers."""

    def setUp(self):
        from sdb_context import SDBRunContext
        self.cls = SDBRunContext

    def test_default_construction(self):
        ctx = self.cls()
        self.assertEqual(ctx.run_id, "")
        self.assertIsNone(ctx.args)
        self.assertIsNone(ctx.out_root)
        self.assertIsNone(ctx.s2_paths)
        self.assertFalse(ctx.land_mask_invert)

    def test_from_args(self):
        args = SimpleNamespace(
            land_mask_type="auto",
            land_mask_water_val=0,
            land_mask_invert=False,
            land_mask_threshold=0.5,
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run_001"
            ctx = self.cls.from_args(args, out_root=out, run_id="test_run")
            self.assertEqual(ctx.run_id, "test_run")
            self.assertEqual(ctx.out_root, out)
            self.assertEqual(ctx.dir_data, out / "data")
            self.assertEqual(ctx.dir_rast, out / "rasters")
            self.assertEqual(ctx.dir_model, out / "model")
            self.assertEqual(ctx.dir_logs, out / "logs")
            self.assertEqual(ctx.dir_plot, out / "plots")
            self.assertEqual(ctx.land_mask_type, "auto")

    def test_ensure_dirs(self):
        args = SimpleNamespace()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run_002"
            ctx = self.cls.from_args(args, out_root=out, run_id="test_run")
            ctx.ensure_dirs()
            self.assertTrue(ctx.dir_data.exists())
            self.assertTrue(ctx.dir_rast.exists())
            self.assertTrue(ctx.dir_model.exists())
            self.assertTrue(ctx.dir_logs.exists())
            self.assertTrue(ctx.dir_plot.exists())

    def test_set_bbox_from_aoi(self):
        ctx = self.cls()
        ctx.set_bbox_from_aoi("-78/-77/25/26")
        self.assertEqual(ctx.bbox_wesn, (-78.0, -77.0, 25.0, 26.0))
        self.assertEqual(ctx.bbox_list, [-78.0, 25.0, -77.0, 26.0])

    def test_set_bbox_from_aoi_strict(self):
        ctx = self.cls()
        with self.assertRaises(ValueError):
            ctx.set_bbox_from_aoi("garbage")

    def test_summary(self):
        ctx = self.cls(run_id="r1", chosen_max_depth=15.0)
        s = ctx.summary()
        self.assertEqual(s["run_id"], "r1")
        self.assertEqual(s["chosen_max_depth"], 15.0)
        self.assertFalse(s["has_s2"])
        self.assertFalse(s["has_fused_df"])

    def test_atl_files_map_default(self):
        """Each instance should get its own default dict (no shared mutable)."""
        ctx1 = self.cls()
        ctx2 = self.cls()
        ctx1.atl_files_map["ATL03"].append("file1.h5")
        self.assertEqual(len(ctx2.atl_files_map["ATL03"]), 0)

    def test_incremental_population(self):
        """Fields can be set incrementally as pipeline progresses."""
        ctx = self.cls(run_id="inc")
        self.assertIsNone(ctx.s2_paths)
        ctx.s2_paths = {"B02": "/path/to/b02.tif"}
        self.assertIsNotNone(ctx.s2_paths)
        ctx.chosen_max_depth = 20.0
        self.assertEqual(ctx.chosen_max_depth, 20.0)


if __name__ == "__main__":
    unittest.main()
