"""Tests for the anchor_support_good / direct-depth prediction path.

Verifies:
1. anchor_support_good detection from dense deep data
2. PhysicsGuidedResidualModel.predict() bypasses Stumpf when anchor_support_good
3. _predict_physics_guided_magnitude() bypasses Stumpf when anchor_support_good
4. Non-anchor runs stay on the residual path

Uses unittest for compatibility with run_regression.sh (unittest discover).
"""
import unittest
import numpy as np
import pandas as pd


class TestAnchorSupportDetection(unittest.TestCase):

    def _make_df(self, n=5000, depth_span=20.0, source_prefix="extra_xyz:hydronos"):
        rng = np.random.RandomState(42)
        depths = -rng.uniform(1.0, 1.0 + depth_span, size=n)
        return pd.DataFrame({
            "depth_m": depths,
            "source_norm": [source_prefix] * n,
            "source": [source_prefix] * n,
            "stumpf_depth": rng.uniform(0.5, 15.0, size=n),
            "stumpf_idx": rng.uniform(0.8, 1.2, size=n),
        })

    def test_anchor_good_with_dense_xyz(self):
        from train import _compute_physics_guidance_settings
        df = self._make_df(n=5000, depth_span=20.0, source_prefix="extra_xyz:hydronos_test")
        gs = _compute_physics_guidance_settings(df)
        self.assertTrue(gs["anchor_support_good"])
        self.assertEqual(gs["correction_alpha"], 1.0)
        self.assertEqual(gs["residual_clip_m"], 15.0)

    def test_anchor_good_depth_fallback(self):
        from train import _compute_physics_guidance_settings
        df = self._make_df(n=6000, depth_span=20.0, source_prefix="unknown_source_type")
        gs = _compute_physics_guidance_settings(df)
        self.assertTrue(gs["anchor_support_good"],
                        f"Expected depth fallback: n={gs['n_points']}, span={gs['depth_span_m']}")

    def test_no_anchor_sparse_atl(self):
        from train import _compute_physics_guidance_settings
        df = self._make_df(n=500, depth_span=3.0, source_prefix="atl03")
        gs = _compute_physics_guidance_settings(df)
        self.assertFalse(gs["anchor_support_good"])


class TestPhysicsGuidedResidualModel(unittest.TestCase):

    def _make_model(self, anchor_good=True):
        from train import PhysicsGuidedResidualModel
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(42)
        feat_cols = ["f0", "f1", "f2", "stumpf_depth", "f4"]
        X = rng.uniform(0, 1, size=(200, 5))
        y = rng.uniform(1, 30, size=200)

        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)

        gs = {
            "anchor_support_good": anchor_good,
            "correction_alpha": 1.0 if anchor_good else 0.18,
            "residual_clip_m": 15.0 if anchor_good else 0.75,
            "stumpf_envelope_m": 20.0 if anchor_good else 1.0,
        }
        model = PhysicsGuidedResidualModel(rf, feat_cols, gs)
        return model, X, feat_cols

    def test_anchor_bypasses_stumpf(self):
        model, X, _ = self._make_model(anchor_good=True)
        pred_anchor = model.predict(X)
        pred_residual = model.predict_residual(X)
        expected = np.maximum(pred_residual, 0.0).astype(np.float32)
        np.testing.assert_array_almost_equal(pred_anchor, expected, decimal=5)

    def test_non_anchor_uses_stumpf(self):
        model, X, _ = self._make_model(anchor_good=False)
        pred = model.predict(X)
        raw = model.predict_residual(X)
        stumpf_base = np.maximum(X[:, 3].astype(np.float32), 0.0)
        mean_offset = np.abs(pred - stumpf_base).mean()
        self.assertLess(mean_offset, np.abs(raw).mean(),
                        "Non-anchor prediction should be constrained near Stumpf base")


class TestPredictPhysicsGuidedMagnitude(unittest.TestCase):

    def test_anchor_mode_bypasses_stumpf(self):
        from train import _predict_physics_guided_magnitude, PhysicsGuidedResidualModel
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(42)
        n = 100
        feat_cols = ["f0", "f1", "f2", "stumpf_depth", "f4"]
        X = rng.uniform(0, 1, size=(n, 5))
        y = rng.uniform(1, 30, size=n)

        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)

        gs = {"anchor_support_good": True, "correction_alpha": 1.0,
              "residual_clip_m": 15.0, "stumpf_envelope_m": 20.0}
        model = PhysicsGuidedResidualModel(rf, feat_cols, gs)

        df = pd.DataFrame(X, columns=feat_cols)
        pred = _predict_physics_guided_magnitude(model, df, feat_cols, gs)
        self.assertEqual(pred.shape, (n,))
        self.assertTrue(np.all(pred >= 0))
        self.assertGreater(pred.mean(), 5.0,
                           f"Direct depth should predict full range, got mean={pred.mean()}")

    def test_non_anchor_uses_stumpf(self):
        from train import _predict_physics_guided_magnitude
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(42)
        n = 100
        feat_cols = ["f0", "f1", "f2", "stumpf_depth", "f4"]
        X = rng.uniform(0, 1, size=(n, 5))
        y = rng.uniform(0, 5, size=n)

        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)

        gs = {"anchor_support_good": False, "correction_alpha": 0.18,
              "residual_clip_m": 0.75, "stumpf_envelope_m": 1.0}
        df = pd.DataFrame(X, columns=feat_cols)
        pred = _predict_physics_guided_magnitude(rf, df, feat_cols, gs)
        stumpf = df["stumpf_depth"].values
        offset = np.abs(pred - stumpf).mean()
        self.assertLess(offset, 2.0,
                        f"Residual mode should stay near Stumpf, offset={offset}")


if __name__ == "__main__":
    unittest.main()
