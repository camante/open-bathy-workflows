import unittest
import numpy as np
import pandas as pd

from train import _select_stumpf_calibration_subset


class TestStumpfCalibrationSubset(unittest.TestCase):
    def _base_df(self):
        return pd.DataFrame({
            "source": [],
            "source_norm": [],
            "granule": [],
            "beam": [],
            "stumpf_idx": [],
            "depth_m": [],
        })

    def test_rejects_shallow_bad_atl_groups(self):
        n = 60
        df = pd.DataFrame({
            "source": ["atl03"] * n,
            "source_norm": ["atl03"] * n,
            "granule": ["g1"] * n,
            "beam": ["gt1l"] * n,
            "stumpf_idx": np.linspace(0.2, 0.25, n),
            "depth_m": -0.2 - np.concatenate([np.zeros(n - 5), np.linspace(0.0, 2.0, 5)]),
        })
        subset, meta = _select_stumpf_calibration_subset(df)
        self.assertEqual(len(subset), 0)
        self.assertTrue(meta["physics_only_recommended"])
        self.assertEqual(meta["atl_groups_selected"], 0)
        self.assertGreaterEqual(meta["atl_groups_rejected"], 1)

    def test_keeps_good_atl_group(self):
        n = 80
        si = np.linspace(0.2, 1.2, n)
        depth = -(1.0 + 6.0 * si)
        df = pd.DataFrame({
            "source": ["atl24"] * n,
            "source_norm": ["atl24"] * n,
            "granule": ["g2"] * n,
            "beam": ["gt2r"] * n,
            "stumpf_idx": si,
            "depth_m": depth,
        })
        subset, meta = _select_stumpf_calibration_subset(df)
        self.assertEqual(len(subset), n)
        self.assertFalse(meta["physics_only_recommended"])
        self.assertEqual(meta["atl_groups_selected"], 1)
        self.assertEqual(meta["rows_atl_selected"], n)

    def test_keeps_anchor_rows_while_rejecting_bad_atl(self):
        n_atl = 40
        n_anchor = 25
        atl = pd.DataFrame({
            "source": ["atl03"] * n_atl,
            "source_norm": ["atl03"] * n_atl,
            "granule": ["g3"] * n_atl,
            "beam": ["gt1r"] * n_atl,
            "stumpf_idx": np.linspace(0.1, 0.12, n_atl),
            "depth_m": -0.3 - np.concatenate([np.zeros(n_atl - 3), np.array([0.1, 0.3, 1.5])]),
        })
        anchor = pd.DataFrame({
            "source": ["extra_xyz"] * n_anchor,
            "source_norm": ["extra_xyz"] * n_anchor,
            "granule": [None] * n_anchor,
            "beam": [None] * n_anchor,
            "stumpf_idx": np.linspace(0.2, 1.0, n_anchor),
            "depth_m": -(2.0 + 3.0 * np.linspace(0.2, 1.0, n_anchor)),
        })
        df = pd.concat([atl, anchor], ignore_index=True)
        subset, meta = _select_stumpf_calibration_subset(df)
        self.assertEqual(len(subset), n_anchor)
        self.assertTrue((subset["source_norm"] == "extra_xyz").all())
        self.assertTrue(meta["used_anchor_only"])
        self.assertEqual(meta["atl_groups_selected"], 0)
        self.assertEqual(meta["rows_anchor"], n_anchor)


if __name__ == "__main__":
    unittest.main()
