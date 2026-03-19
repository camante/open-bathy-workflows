import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

import authoritative_conditioning as ac


class TestAuthoritativeConditioningEndToEnd(unittest.TestCase):

    def _run_case(self, shape=(25, 25)):
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        candidate = (0.1 * xx + 0.2 * yy).astype(np.float32)
        auth = np.full(shape, np.nan, dtype=np.float32)
        # Internal authoritative support only, so nested overlap should be stable.
        auth[8:17, 8] = 1.0
        auth[8:17, 16] = 2.0
        auth[8, 8:17] = 1.5
        auth[16, 8:17] = 2.5
        sdb_ok = np.zeros(shape, dtype=bool)
        sdb_ok[7:18, 7:18] = True
        river_ok = np.zeros(shape, dtype=bool)
        river_ok[10:15, 10:15] = True
        river_support = np.zeros(shape, dtype=np.uint8)
        river_support[11:14, 11] = 1
        river_support_depth = np.full(shape, np.nan, dtype=np.float32)
        river_support_depth[11:14, 11] = 1.75
        return ac.support_weighted_condition_arrays(
            candidate=candidate,
            auth=auth,
            sdb_ok=sdb_ok,
            river_ok=river_ok,
            sdb_gw=np.where(sdb_ok, 0.75, 0.0).astype(np.float32),
            sdb_ti=sdb_ok.astype(np.uint8),
            river_gw=np.where(river_ok, 0.65, 0.0).astype(np.float32),
            river_ti=river_ok.astype(np.uint8),
            river_support=river_support,
            river_support_depth=river_support_depth,
            pixel_size_m=10.0,
            support_decay_m=300.0,
            support_density_radius_m=60.0,
            coastal_sdb_support_transition_m=200.0,
            river_anchor_density_radius_m=100.0,
            river_scaffold_transition_m=500.0,
        )

    def test_rerun_identity_and_nested_overlap_on_written_outputs(self):
        first = self._run_case()
        second = self._run_case()
        np.testing.assert_allclose(first["conditioned"], second["conditioned"], atol=1e-6)
        np.testing.assert_array_equal(first["support"], second["support"])
        np.testing.assert_array_equal(first["provenance"], second["provenance"])

        sl = np.s_[5:20, 5:20]
        # Rebuild the true nested inputs from the original arrays.
        base = self._run_case()
        yy, xx = np.mgrid[0:25, 0:25]
        candidate = (0.1 * xx + 0.2 * yy).astype(np.float32)[sl]
        auth = np.full((25, 25), np.nan, dtype=np.float32)
        auth[8:17, 8] = 1.0
        auth[8:17, 16] = 2.0
        auth[8, 8:17] = 1.5
        auth[16, 8:17] = 2.5
        sdb_ok = np.zeros((25, 25), dtype=bool)
        sdb_ok[7:18, 7:18] = True
        river_ok = np.zeros((25, 25), dtype=bool)
        river_ok[10:15, 10:15] = True
        river_support = np.zeros((25, 25), dtype=np.uint8)
        river_support[11:14, 11] = 1
        river_support_depth = np.full((25, 25), np.nan, dtype=np.float32)
        river_support_depth[11:14, 11] = 1.75
        nested = ac.support_weighted_condition_arrays(
            candidate=candidate,
            auth=auth[sl],
            sdb_ok=sdb_ok[sl],
            river_ok=river_ok[sl],
            sdb_gw=np.where(sdb_ok[sl], 0.75, 0.0).astype(np.float32),
            sdb_ti=sdb_ok[sl].astype(np.uint8),
            river_gw=np.where(river_ok[sl], 0.65, 0.0).astype(np.float32),
            river_ti=river_ok[sl].astype(np.uint8),
            river_support=river_support[sl],
            river_support_depth=river_support_depth[sl],
            pixel_size_m=10.0,
            support_decay_m=300.0,
            support_density_radius_m=60.0,
            coastal_sdb_support_transition_m=200.0,
            river_anchor_density_radius_m=100.0,
            river_scaffold_transition_m=500.0,
        )
        np.testing.assert_allclose(first["conditioned"][sl], nested["conditioned"], atol=1e-6)
        np.testing.assert_array_equal(first["support"][sl], nested["support"])
        np.testing.assert_array_equal(first["provenance"][sl], nested["provenance"])

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            out1 = td / "run1_conditioned.tif"
            out2 = td / "run2_conditioned.tif"
            tifffile.imwrite(out1, first["conditioned"].astype(np.float32))
            tifffile.imwrite(out2, second["conditioned"].astype(np.float32))
            self.assertEqual(out1.read_bytes(), out2.read_bytes())


if __name__ == "__main__":
    unittest.main()
