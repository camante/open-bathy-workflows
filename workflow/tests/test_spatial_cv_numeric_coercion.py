import unittest
import numpy as np
import pandas as pd

from spatial_cv import _coerce_numeric_matrix, _coerce_numeric_vector, run_spatial_cv


class TestSpatialCVNumericCoercion(unittest.TestCase):
    def test_coerce_numeric_matrix_from_object_frame(self):
        df = pd.DataFrame({
            'a': ['1.0', '2.5', 'bad'],
            'b': [3, '4.5', None],
        })
        arr = _coerce_numeric_matrix(df)
        self.assertEqual(arr.dtype.kind, 'f')
        self.assertTrue(np.isfinite(arr[0, 0]))
        self.assertTrue(np.isnan(arr[2, 0]))
        self.assertTrue(np.isnan(arr[2, 1]))

    def test_coerce_numeric_vector_from_object_series(self):
        s = pd.Series(['1.0', 'bad', 3])
        arr = _coerce_numeric_vector(s)
        self.assertEqual(arr.dtype.kind, 'f')
        self.assertTrue(np.isnan(arr[1]))
        self.assertEqual(arr[2], 3.0)

    def test_run_spatial_cv_handles_object_dtype_features(self):
        n = 240
        df = pd.DataFrame({
            'longitude': np.linspace(-70.9, -70.8, n),
            'latitude': np.linspace(42.8, 42.9, n),
            'f1': [f"{x:.3f}" for x in np.linspace(0.1, 1.0, n)],
            'f2': np.linspace(1.0, 2.0, n).astype(object),
            'depth_m': np.linspace(-8.0, -1.0, n),
        })
        summary = run_spatial_cv(
            df,
            feature_cols=['f1', 'f2'],
            target_col='depth_m',
            cv_strategy='spatial_cluster',
            n_folds=3,
            seed=42,
        )
        self.assertGreaterEqual(summary.n_folds, 1)
        self.assertTrue(np.isfinite(summary.rmse_mean))


if __name__ == '__main__':
    unittest.main()
