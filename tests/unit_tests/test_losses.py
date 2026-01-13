import sys
from pathlib import Path
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.neural_networks.losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    MeanSquaredError,
)


class TestMeanSquaredError(TestCase):

    def test_loss_known_value(self):
        # Simple example where we can compute MSE by hand.
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.0, 2.0, 1.0])
        # Squared errors: [0, 1, 1] => mean = 2/3
        expected = 2.0 / 3.0

        mse = MeanSquaredError()
        self.assertAlmostEqual(mse.loss(y_true, y_pred), expected, places=12)

    def test_derivative_matches_finite_differences(self):
        # Validate dLoss/dy_pred numerically with central differences.
        rs = np.random.RandomState(0)
        y_true = rs.randn(4)
        y_pred = rs.randn(4)

        mse = MeanSquaredError()
        analytic = mse.derivative(y_true, y_pred)

        eps = 1e-6
        numeric = np.zeros_like(y_pred)
        for i in range(y_pred.shape[0]):
            yp = y_pred.copy(); yp[i] += eps
            ym = y_pred.copy(); ym[i] -= eps
            numeric[i] = (mse.loss(y_true, yp) - mse.loss(y_true, ym)) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


class TestBinaryCrossEntropy(TestCase):

    def test_loss_is_finite_at_extremes(self):
        # BCE must not produce inf/nan when probabilities hit 0 or 1 (clip protects this).
        y_true = np.array([0.0, 1.0, 1.0, 0.0])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])

        bce = BinaryCrossEntropy()
        val = bce.loss(y_true, y_pred)
        self.assertTrue(np.isfinite(val))

    def test_derivative_matches_finite_differences(self):
        rs = np.random.RandomState(1)
        y_true = rs.randint(0, 2, size=6).astype(float)
        y_pred = rs.rand(6)

        bce = BinaryCrossEntropy()
        analytic = bce.derivative(y_true, y_pred)

        eps = 1e-6
        numeric = np.zeros_like(y_pred)
        for i in range(y_pred.shape[0]):
            yp = y_pred.copy(); yp[i] += eps
            ym = y_pred.copy(); ym[i] -= eps
            numeric[i] = (bce.loss(y_true, yp) - bce.loss(y_true, ym)) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


class TestCategoricalCrossEntropy(TestCase):

    def test_loss_known_value_uniform(self):
        # For uniform probabilities and one-hot targets, loss = -log(1/C)
        y_true = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        y_pred = np.full((3, 3), 1.0 / 3.0)

        cce = CategoricalCrossEntropy()
        expected = -np.log(1.0 / 3.0)
        self.assertAlmostEqual(cce.loss(y_true, y_pred), float(expected), places=12)

    def test_loss_is_near_zero_for_perfect_predictions(self):
        # When the model assigns probability ~1 to the true class, loss should be near 0.
        y_true = np.array([[1.0, 0.0, 0.0]])
        y_pred = np.array([[1.0, 0.0, 0.0]])

        cce = CategoricalCrossEntropy()
        val = cce.loss(y_true, y_pred)
        self.assertTrue(np.isfinite(val))
        self.assertLess(val, 1e-10)

    def test_derivative_matches_finite_differences(self):
        rs = np.random.RandomState(2)
        y_true = np.eye(4)[rs.randint(0, 4, size=3)]  # one-hot
        logits = rs.randn(3, 4)

        # Build a valid probability distribution per row (softmax-like).
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        y_pred = exp / np.sum(exp, axis=1, keepdims=True)

        cce = CategoricalCrossEntropy()
        analytic = cce.derivative(y_true, y_pred)

        eps = 1e-6
        numeric = np.zeros_like(y_pred)
        for r in range(y_pred.shape[0]):
            for c in range(y_pred.shape[1]):
                yp = y_pred.copy(); yp[r, c] += eps
                ym = y_pred.copy(); ym[r, c] -= eps
                numeric[r, c] = (cce.loss(y_true, yp) - cce.loss(y_true, ym)) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    import unittest

    unittest.main(verbosity=2)
