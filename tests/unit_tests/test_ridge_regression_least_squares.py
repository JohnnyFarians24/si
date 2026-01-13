import sys
from pathlib import Path
import unittest
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.data.dataset import Dataset
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares
from si.metrics.mse import mse


class TestRidgeRegressionLeastSquares(TestCase):

    def test_fit_sets_parameters_with_expected_shapes(self):
        # Build a small full-rank regression problem.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        true_theta = np.array([2.0, -1.0, 0.5])
        true_intercept = 3.0
        y = true_intercept + X @ true_theta

        ds = Dataset(X=X, y=y, features=['f1', 'f2', 'f3'], label='y')

        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        model.fit(ds)

        # After fitting, theta and theta_zero should exist.
        self.assertIsNotNone(model.theta)
        self.assertIsNotNone(model.theta_zero)

        # theta should match number of features.
        self.assertEqual(model.theta.shape, (3,))
        # theta_zero should be a scalar-like.
        self.assertIsInstance(model.theta_zero, float)

        # mean/std should be available when scaling.
        self.assertIsNotNone(model.mean)
        self.assertIsNotNone(model.std)
        self.assertEqual(model.mean.shape, (3,))
        self.assertEqual(model.std.shape, (3,))

    def test_predict_shape_and_training_error_small_without_regularization(self):
        # With lambda=0 and noiseless data, the closed-form solution should fit nearly perfectly.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 4))
        true_theta = np.array([1.0, -2.0, 0.25, 4.0])
        true_intercept = -0.5
        y = true_intercept + X @ true_theta

        ds = Dataset(X=X, y=y, features=[f"f{i}" for i in range(4)], label='y')

        model = RidgeRegressionLeastSquares(l2_penalty=0.0, scale=False)
        model.fit(ds)
        preds = model.predict(ds)

        # Predictions must be 1D with n_samples.
        self.assertEqual(preds.shape, (ds.shape()[0],))

        # MSE should be essentially zero (numerical tolerance).
        self.assertLess(mse(y, preds), 1e-10)

    def test_score_matches_mse_metric(self):
        # score() should return the same value as mse(y_true, y_pred).
        rng = np.random.default_rng(2)
        X = rng.normal(size=(40, 2))
        y = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1]
        ds = Dataset(X=X, y=y)

        model = RidgeRegressionLeastSquares(l2_penalty=0.1, scale=True)
        model.fit(ds)
        preds = model.predict(ds)

        self.assertAlmostEqual(float(model.score(ds)), float(mse(ds.y, preds)), places=12)

    def test_scaling_handles_constant_feature(self):
        # If a feature has zero variance, scaling should not divide by zero.
        # This test ensures fit/predict work when one column is constant.
        X = np.array([
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 10.0],
            [4.0, 10.0],
        ])
        # y depends only on the first feature.
        y = 5.0 + 2.0 * X[:, 0]
        ds = Dataset(X=X, y=y, features=['varying', 'constant'], label='y')

        # With a constant feature, X_aug can become singular if l2_penalty=0.
        # Use a tiny positive ridge penalty to make the system solvable.
        model = RidgeRegressionLeastSquares(l2_penalty=1e-8, scale=True)
        model.fit(ds)
        preds = model.predict(ds)

        # No NaNs/infs should appear.
        self.assertTrue(np.isfinite(preds).all())
        # Should fit (near) perfectly for noiseless data (numerical tolerance).
        self.assertLess(mse(y, preds), 1e-8)

    def test_stronger_regularization_shrinks_coefficients(self):
        # Larger l2_penalty should shrink coefficient norm (holding data constant).
        rng = np.random.default_rng(3)
        X = rng.normal(size=(100, 5))
        true_theta = np.array([1.5, -0.5, 2.0, 0.0, -1.0])
        y = 0.25 + X @ true_theta
        ds = Dataset(X=X, y=y)

        weak = RidgeRegressionLeastSquares(l2_penalty=1e-6, scale=False).fit(ds)
        strong = RidgeRegressionLeastSquares(l2_penalty=1e3, scale=False).fit(ds)

        # Coefficient vector norm should be smaller under stronger regularization.
        self.assertLess(np.linalg.norm(strong.theta), np.linalg.norm(weak.theta))


if __name__ == '__main__':
    unittest.main()
