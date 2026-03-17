"""test_pipeline.py – integration tests for train_sdb_model and predict_scene."""

from __future__ import annotations
import sys
import json
import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from sklearn.isotonic import IsotonicRegression

from conftest import (
    _install_mocks, build_synthetic_scene, build_training_df,
    build_model_artifacts, FEATURE_COLS,
)
_install_mocks()


def _predict_kwargs(scene_paths, model_dir, out_path, **extra):
    return dict(
        s2_paths={k: str(v) for k, v in scene_paths.items() if k != "land_mask"},
        land_mask_path=str(scene_paths["land_mask"]),
        rf_model_path=str(model_dir / "rf_model.pkl"),
        meta_json_path=str(model_dir / "model_meta.json"),
        stumpf_lr_path=str(model_dir / "stumpf_lr.pkl"),
        out_path=str(out_path),
        tile_size=64,
        **extra,
    )


class TestTrainSdbModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._session = Path(tempfile.mkdtemp())
        cls.training_df = build_training_df()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._session, ignore_errors=True)

    def _train(self, df=None, **kw):
        import train
        defaults = dict(
            max_depth_sdb=20.0, seed=42,
            plots_dir=self._session / "plots",
            water_class="ocean", use_stumpf_depth=True,
            min_training_points_for_sdb=50,
        )
        defaults.update(kw)
        return train.train_sdb_model(
            train_df=df if df is not None else self.training_df,
            **defaults,
        )

    def test_returns_rf_with_predict(self):
        rf, _, _, _, _ = self._train()
        self.assertIsNotNone(rf)
        self.assertTrue(hasattr(rf, "predict"))

    def test_feature_columns_in_meta(self):
        _, _, _, _, meta = self._train()
        self.assertIn("feature_columns", meta)
        self.assertGreater(len(meta["feature_columns"]), 0)

    def test_train_test_split_covers_input(self):
        _, _, df_train, df_test, _ = self._train()
        self.assertGreater(len(df_train), 0)
        self.assertGreater(len(df_test), 0)
        total = len(df_train) + len(df_test)
        self.assertAlmostEqual(total, len(self.training_df), delta=10)

    def test_rf_predicts_positive_magnitudes(self):
        """The RF must output positive depth magnitudes (negation happens in predict.py)."""
        rf, _, _, _, meta = self._train()
        b02 = self.training_df["B02"].values.astype(np.float32)
        b03 = self.training_df["B03"].values.astype(np.float32)
        b04 = self.training_df["B04"].values.astype(np.float32)
        b08 = self.training_df["B08"].values.astype(np.float32)
        brt = self.training_df["brightness"].values.astype(np.float32)
        feat_map = {
            "B02": b02, "B03": b03, "B04": b04, "B08": b08,
            "log_B02": np.log(b02+1e-3), "log_B03": np.log(b03+1e-3),
            "log_B04": np.log(b04+1e-3), "log_B08": np.log(b08+1e-3),
            "brightness": brt,
            "B03_B02": b03/(b02+1e-6), "B04_B03": b04/(b03+1e-6),
            "nbri": (b08-b03)/(b08+b03+1e-6),
            "stumpf_idx": self.training_df["stumpf_idx"].values,
            "stumpf_depth": self.training_df["stumpf_depth"].values,
        }
        X = np.column_stack([feat_map[c] for c in meta["feature_columns"]]).astype(np.float32)
        preds = rf.predict(X)
        self.assertTrue(np.all(preds >= 0),
                        f"RF predictions should be positive magnitudes; got min={preds.min():.3f}")

    def test_rf_wrapper_roundtrips_through_joblib(self):
        import joblib
        rf, _, _, _, meta = self._train()
        model_path = self._session / "rf_wrapper_roundtrip.pkl"
        joblib.dump(rf, model_path)
        loaded = joblib.load(model_path)

        b02 = self.training_df["B02"].values.astype(np.float32)
        b03 = self.training_df["B03"].values.astype(np.float32)
        b04 = self.training_df["B04"].values.astype(np.float32)
        b08 = self.training_df["B08"].values.astype(np.float32)
        brt = self.training_df["brightness"].values.astype(np.float32)
        feat_map = {
            "B02": b02, "B03": b03, "B04": b04, "B08": b08,
            "log_B02": np.log(b02+1e-3), "log_B03": np.log(b03+1e-3),
            "log_B04": np.log(b04+1e-3), "log_B08": np.log(b08+1e-3),
            "brightness": brt,
            "B03_B02": b03/(b02+1e-6), "B04_B03": b04/(b03+1e-6),
            "nbri": (b08-b03)/(b08+b03+1e-6),
            "stumpf_idx": self.training_df["stumpf_idx"].values,
            "stumpf_depth": self.training_df["stumpf_depth"].values,
        }
        X = np.column_stack([feat_map[c] for c in meta["feature_columns"]]).astype(np.float32)
        preds = loaded.predict(X)
        residual = loaded.predict_residual(X)

        self.assertEqual(preds.shape[0], X.shape[0])
        self.assertEqual(residual.shape[0], X.shape[0])
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all(np.isfinite(residual)))
        self.assertTrue(np.all(preds >= 0),
                        f"Loaded RF wrapper predictions should be positive magnitudes; got min={preds.min():.3f}")

    def test_input_depths_are_negative(self):
        self.assertTrue(np.all(self.training_df["depth_m"] < 0),
                        "Training fixture must use negative-down depths")

    def test_meta_has_max_depth_key(self):
        _, _, _, _, meta = self._train()
        depth_keys = [k for k in meta if "max_depth" in k]
        self.assertGreater(len(depth_keys), 0)

    def test_train_report_written_to_explicit_diagnostics_dir(self):
        diag_dir = self._session / "diag_report_out"
        self._train(diagnostics_dir=diag_dir)
        report_path = diag_dir / "train_report.json"
        self.assertTrue(report_path.exists(), "train_report.json should be written to diagnostics_dir")
        payload = json.loads(report_path.read_text())
        self.assertIn("train", payload)
        self.assertIn("max_depth_sdb_auto_m", payload["train"])

    def test_too_few_points_does_not_raise(self):
        tiny = pd.DataFrame({
            "longitude": [-81.5], "latitude": [24.5],
            "depth_m": [-5.0],
            "B02": [0.05], "B03": [0.04], "B04": [0.03], "B08": [0.01],
            "brightness": [0.04], "CLEAR_WATER": [0.8], "LAND": [0.1],
            "stumpf_idx": [1.1], "stumpf_depth": [5.0],
            "source": ["atl03"], "sample_weight": [1.0],
        })
        rf, _, _, _, meta = self._train(df=tiny, min_training_points_for_sdb=200)
        self.assertIsNotNone(meta)


class TestPredictScene(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._session = Path(tempfile.mkdtemp())
        cls.scene_paths = build_synthetic_scene(cls._session / "scene")
        cls.model_dir = cls._session / "model"
        build_model_artifacts(cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._session, ignore_errors=True)

    def _run(self, out_name="depth.tif", **extra):
        import predict
        out = self._session / out_name
        kw = _predict_kwargs(self.scene_paths, self.model_dir, out)
        kw.update(extra)
        predict.predict_scene(**kw)
        return out

    def test_depth_raster_written(self):
        out = self._run("d_basic.tif", write_confidence=False, write_provenance=False)
        self.assertTrue(out.exists())

    def test_confidence_raster_written_when_enabled(self):
        out = self._run("d_conf.tif", write_confidence=True, write_provenance=False)
        conf = out.with_name(out.stem + "_confidence.tif")
        self.assertTrue(conf.exists(), "Confidence raster missing")

    def test_confidence_raster_absent_when_disabled(self):
        out = self._run("d_noconf.tif", write_confidence=False, write_provenance=False)
        conf = out.with_name(out.stem + "_confidence.tif")
        self.assertFalse(conf.exists(), "Confidence raster should not be written")

    def test_provenance_raster_written_when_enabled(self):
        out = self._run("d_prov.tif", write_confidence=False, write_provenance=True)
        prov = out.with_name(out.stem + "_provenance.tif")
        self.assertTrue(prov.exists(), "Provenance raster missing")

    def test_uncertainty_raster_always_written(self):
        out = self._run("d_unc.tif", write_confidence=False, write_provenance=False)
        unc = out.with_name(out.stem + "_uncertainty.tif")
        self.assertTrue(unc.exists(), "Uncertainty raster must always be written")

    def test_provenance_invariant(self):
        """Wherever provenance==1 depth must be finite and not nodata."""
        out = self._run("d_inv.tif", write_confidence=False, write_provenance=True)
        prov_path = out.with_name(out.stem + "_provenance.tif")

        depth_arr = tifffile.imread(str(out)).astype(np.float32)
        prov_arr = tifffile.imread(str(prov_path))
        predicted = prov_arr == 1

        if predicted.any():
            d = depth_arr[predicted]
            self.assertTrue(np.all(np.isfinite(d)),
                            "provenance=1 pixels must have finite depth")
            self.assertTrue(np.all(d != -9999.0),
                            "provenance=1 pixels must not be nodata")

    def test_depth_values_negative_down(self):
        out = self._run("d_sign.tif", write_confidence=False, write_provenance=False)
        arr = tifffile.imread(str(out)).astype(np.float32)
        valid = arr[np.isfinite(arr) & (arr != -9999.0)]
        if len(valid) > 0:
            self.assertTrue(np.all(valid <= 0),
                            f"Depths must be ≤0 (negative-down); max={valid.max():.3f}")

    def test_min_confidence_threshold_reduces_coverage(self):
        out_full = self._run("d_full.tif", write_confidence=True,
                             write_provenance=False, min_confidence_threshold=0.0)
        out_filt = self._run("d_filt.tif", write_confidence=True,
                             write_provenance=False, min_confidence_threshold=0.9)

        n_full = np.sum(np.isfinite(tifffile.imread(str(out_full)).astype(np.float32)) &
                        (tifffile.imread(str(out_full)).astype(np.float32) != -9999.0))
        n_filt = np.sum(np.isfinite(tifffile.imread(str(out_filt)).astype(np.float32)) &
                        (tifffile.imread(str(out_filt)).astype(np.float32) != -9999.0))

        self.assertLessEqual(n_filt, n_full,
                             f"Threshold 0.9 should ≤ coverage of threshold 0.0: {n_filt} vs {n_full}")

    def test_threshold_with_no_confidence_emits_warning(self):
        import logging
        import predict
        out = self._session / "d_warn.tif"
        kw = _predict_kwargs(self.scene_paths, self.model_dir, out,
                             write_confidence=False, write_provenance=False,
                             min_confidence_threshold=0.5)
        with self.assertLogs("sdb.predict", level=logging.WARNING) as cm:
            predict.predict_scene(**kw)
        self.assertTrue(any("min_confidence_threshold" in m for m in cm.output),
                        "Expected warning about threshold having no effect")

    def test_support_neighbor_gate_blocks_singleton_support(self):
        import predict
        meta_path = self.model_dir / "model_meta.json"
        original = meta_path.read_text()
        try:
            meta = json.loads(original)
            meta["physics_guidance"] = {
                "max_train_point_dist_m": 1.0e6,
                "min_support_neighbors": 3,
                "support_points_lonlat": [[-81.0, 25.0]],
                "trusted_halo_px": 32,
                "low_support": True,
            }
            meta_path.write_text(json.dumps(meta, indent=2))

            out = self._session / "d_singleton_support.tif"
            kw = _predict_kwargs(self.scene_paths, self.model_dir, out,
                                 write_confidence=False, write_provenance=False)
            predict.predict_scene(**kw)
            arr = tifffile.imread(str(out)).astype(np.float32)
            valid = np.isfinite(arr) & (arr != -9999.0)
            self.assertEqual(int(valid.sum()), 0,
                             "A singleton support point must not open broad prediction coverage when min_support_neighbors=3")
        finally:
            meta_path.write_text(original)

    def test_isotonic_stumpf_model_does_not_emit_linear_coeff_warning(self):
        import logging
        import joblib
        import predict

        iso_model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        stumpf_idx = np.linspace(0.8, 1.6, 64).astype(np.float32)
        depths = np.linspace(1.0, 8.0, 64).astype(np.float32)
        iso_model.fit(stumpf_idx, depths)
        joblib.dump(iso_model, self.model_dir / "stumpf_lr.pkl")

        out = self._session / "d_isotonic.tif"
        kw = _predict_kwargs(self.scene_paths, self.model_dir, out,
                             write_confidence=False, write_provenance=False)
        with self.assertLogs("sdb.predict", level=logging.INFO) as cm:
            predict.predict_scene(**kw)
        logs = "\n".join(cm.output)
        self.assertIn("IsotonicRegression", logs)
        self.assertNotIn("Could not extract Stumpf LR coefficients", logs)


class TestPredictChunked(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._session = Path(tempfile.mkdtemp())
        cls.scene_paths = build_synthetic_scene(cls._session / "scene")
        cls.model_dir = cls._session / "model"
        build_model_artifacts(cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._session, ignore_errors=True)

    def test_chunked_wrapper_produces_output(self):
        from predict_chunked import predict_scene_chunked
        out_dir = self._session / "chunked_out"
        result = predict_scene_chunked(
            model_dir=self.model_dir,
            band_paths={k: v for k, v in self.scene_paths.items() if k != "land_mask"},
            out_dir=out_dir,
            final_out_path=str(out_dir / "final_depth.tif"),
            land_mask_path=str(self.scene_paths["land_mask"]),
            tile_size=64,
            write_confidence=True,
            write_provenance=True,
        )
        self.assertTrue(result["output_raster"].exists())

    def test_chunked_discovers_stumpf_lr(self):
        """Chunked path must pick up stumpf_lr.pkl without explicit argument."""
        self.assertTrue((self.model_dir / "stumpf_lr.pkl").exists(),
                        "Fixture must include stumpf_lr.pkl")
        from predict_chunked import predict_scene_chunked
        out_dir = self._session / "chunked_stumpf"
        result = predict_scene_chunked(
            model_dir=self.model_dir,
            band_paths={k: v for k, v in self.scene_paths.items() if k != "land_mask"},
            out_dir=out_dir,
            land_mask_path=str(self.scene_paths["land_mask"]),
            tile_size=64,
            write_confidence=False,
            write_provenance=False,
        )
        self.assertTrue(result["output_raster"].exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
