import unittest


class TestValidationInvariants(unittest.TestCase):
    def test_negative_r2_requires_guidance_only(self):
        from sdb_main import _derive_validation_invariants
        out = _derive_validation_invariants(
            {"n": 100, "r2": -0.4, "rmse": 2.0},
            {"n": 80, "r2": 0.3, "rmse": 2.2},
            {},
        )
        self.assertTrue(out["negative_validation_detected"])
        self.assertTrue(out["guidance_only_required"])
        self.assertTrue(out["representative_spatial_holdout"])

    def test_missing_spatial_holdout_requires_guidance_only(self):
        from sdb_main import _derive_validation_invariants
        out = _derive_validation_invariants(
            {"n": 100, "r2": 0.4, "rmse": 1.0},
            {"n": 0, "r2": None, "rmse": None},
            {"reason": "no_representative_cluster_holdout"},
        )
        self.assertFalse(out["representative_spatial_holdout"])
        self.assertTrue(out["guidance_only_required"])

    def test_healthy_validation_not_guidance_only(self):
        from sdb_main import _derive_validation_invariants
        out = _derive_validation_invariants(
            {"n": 100, "r2": 0.5, "rmse": 1.0},
            {"n": 80, "r2": 0.35, "rmse": 1.2},
            {},
        )
        self.assertFalse(out["guidance_only_required"])
        self.assertFalse(out["negative_validation_detected"])
        self.assertTrue(out["representative_spatial_holdout"])


if __name__ == "__main__":
    unittest.main()
