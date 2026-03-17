from __future__ import annotations
import unittest
from unittest import mock
import tempfile
import shutil
from pathlib import Path

from conftest import _install_mocks, build_training_df
_install_mocks()


class TestTrainFallbackRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._session = Path(tempfile.mkdtemp())
        cls.training_df = build_training_df()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._session, ignore_errors=True)

    def test_spatial_cv_failure_records_degraded_fallback(self):
        import train
        from errors_scientific import FallbackRegistry

        reg = FallbackRegistry()
        with mock.patch("spatial_cv.run_spatial_cv", side_effect=ValueError("boom")):
            train.train_sdb_model(
                train_df=self.training_df,
                max_depth_sdb=20.0,
                seed=42,
                plots_dir=self._session / "plots",
                water_class="ocean",
                use_stumpf_depth=True,
                min_training_points_for_sdb=50,
                spatial_cv_enabled=True,
                fallback_registry=reg,
            )
        names = [e.name for e in reg.events]
        self.assertIn("spatial_cv_failed", names)
        evt = next(e for e in reg.events if e.name == "spatial_cv_failed")
        self.assertEqual(evt.fallback_class.value, "degraded")


if __name__ == "__main__":
    unittest.main()
