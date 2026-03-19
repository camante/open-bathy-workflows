import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile


class TestFinalDemSelectionReceipt(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._write_final_dem_selection_receipt

    def test_receipt_explains_gapfill_preference(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            fusion = out_dir / "combined" / "fusion.tif"
            conditioned = out_dir / "combined" / "conditioned.tif"
            gapfill = out_dir / "combined" / "gapfill.tif"
            user = out_dir / "deliver" / "user.tif"
            prov = out_dir / "combined" / "prov.tif"
            for p in (fusion, conditioned, gapfill, user, prov):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
            cfg = SimpleNamespace(out_dir=out_dir)
            report = {
                "fusion": {"outputs": {"depth": str(fusion)}},
                "authoritative_base": {"outputs": {"conditioned_depth": str(conditioned), "aligned_authoritative_base": str(conditioned)}},
                "gapfill": {"outputs": {"depth": str(gapfill)}},
            }
            out = self.fn(cfg, report, final_native=gapfill, final_for_user=user, final_provenance=prov)
            payload = json.loads(Path(out).read_text())
            self.assertEqual(payload["selected_final_stage"], "gapfill")
            self.assertEqual(payload["selected_final_depth"], str(user))
            joined = " ".join(payload["selection_rationale"])
            self.assertIn("authoritative", joined.lower())
            self.assertIn("gapfill", joined.lower())


class TestComparisonSummary(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._write_comparison_summary

    def test_summary_reports_class_stats(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            final = td / "final.tif"
            diff = td / "diff.tif"
            support = td / "support.tif"
            prov = td / "prov.tif"
            tifffile.imwrite(final, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
            tifffile.imwrite(diff, np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32))
            tifffile.imwrite(support, np.array([[1, 3], [4, 5]], dtype=np.uint8))
            tifffile.imwrite(prov, np.array([[10, 30], [40, 50]], dtype=np.uint8))
            cfg = SimpleNamespace(aoi="-71/-70.75/42.75/43", tile_bbox=None)
            report = {
                "authoritative_base": {
                    "policy": {
                        "support_class_codes": {"1": "authoritative_locked", "3": "sdb", "4": "river", "5": "scaffold"},
                        "provenance_class_codes": {"10": "authoritative_locked", "30": "sdb", "40": "river", "50": "scaffold"},
                    }
                },
                "outputs": {},
            }
            packaged = {
                "final_depth_native": str(final),
                "baseline_cudem_interpolation_aligned_to_final": str(final),
                "conditioned_minus_baseline_cudem": str(diff),
                "support_class": str(support),
                "final_provenance_native": str(prov),
            }
            out = self.fn(cfg, report, packaged, td)
            payload = json.loads(Path(out).read_text())
            self.assertIn("support_class_summary", payload)
            self.assertIn("provenance_class_summary", payload)
            self.assertEqual(payload["support_class_summary"]["by_class"]["3"]["label"], "sdb")
            self.assertAlmostEqual(payload["conditioned_minus_baseline_overall"]["rmse"], np.sqrt((0 + 1 + 4 + 1) / 4))


class TestNestedAoiOverlapIdentity(unittest.TestCase):

    def setUp(self):
        import bathy_main
        self.fn = bathy_main._support_weighted_condition_arrays

    def test_nested_aoi_overlap_is_identical_when_support_context_is_internal(self):
        full_shape = (21, 21)
        yy, xx = np.mgrid[0:full_shape[0], 0:full_shape[1]]
        candidate_full = (0.25 * xx + 0.5 * yy).astype(np.float32)
        auth_full = np.full(full_shape, np.nan, dtype=np.float32)
        # Keep all hard-control anchors inside the nested AOI so outside context is irrelevant.
        auth_full[8:13, 8] = 2.0
        auth_full[8:13, 12] = 3.0
        auth_full[8, 8:13] = 2.5
        auth_full[12, 8:13] = 3.5
        sdb_ok_full = np.zeros(full_shape, dtype=bool)
        sdb_ok_full[7:14, 7:14] = True
        river_ok_full = np.zeros(full_shape, dtype=bool)

        full = self.fn(
            candidate=candidate_full,
            auth=auth_full,
            sdb_ok=sdb_ok_full,
            river_ok=river_ok_full,
            sdb_gw=np.where(sdb_ok_full, 0.8, 0.0).astype(np.float32),
            sdb_ti=sdb_ok_full.astype(np.uint8),
            river_gw=None,
            river_ti=None,
            river_support=None,
            river_support_depth=None,
            pixel_size_m=10.0,
            support_decay_m=300.0,
            support_density_radius_m=60.0,
            coastal_sdb_support_transition_m=200.0,
            river_anchor_density_radius_m=200.0,
            river_scaffold_transition_m=800.0,
        )

        sl = np.s_[5:16, 5:16]
        nested = self.fn(
            candidate=candidate_full[sl],
            auth=auth_full[sl],
            sdb_ok=sdb_ok_full[sl],
            river_ok=river_ok_full[sl],
            sdb_gw=np.where(sdb_ok_full[sl], 0.8, 0.0).astype(np.float32),
            sdb_ti=sdb_ok_full[sl].astype(np.uint8),
            river_gw=None,
            river_ti=None,
            river_support=None,
            river_support_depth=None,
            pixel_size_m=10.0,
            support_decay_m=300.0,
            support_density_radius_m=60.0,
            coastal_sdb_support_transition_m=200.0,
            river_anchor_density_radius_m=200.0,
            river_scaffold_transition_m=800.0,
        )

        np.testing.assert_allclose(full["conditioned"][sl], nested["conditioned"], atol=1e-6)
        np.testing.assert_array_equal(full["support"][sl], nested["support"])
        np.testing.assert_array_equal(full["provenance"][sl], nested["provenance"])


if __name__ == "__main__":
    unittest.main()
