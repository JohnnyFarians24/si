import sys
from pathlib import Path
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.neural_networks.activation import SoftmaxActivation, TanhActivation


class TestTanhActivation(TestCase):

    def test_forward_matches_numpy_tanh(self):
        x = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
        layer = TanhActivation()
        layer.set_input_shape((x.shape[1],))

        out = layer.forward_propagation(x, training=True)
        np.testing.assert_allclose(out, np.tanh(x), rtol=1e-12, atol=1e-12)

        # Output must be in [-1, 1].
        self.assertTrue(np.all(out <= 1.0))
        self.assertTrue(np.all(out >= -1.0))

    def test_backward_uses_derivative(self):
        x = np.random.RandomState(0).randn(3, 4)
        layer = TanhActivation()
        layer.set_input_shape((x.shape[1],))
        layer.forward_propagation(x, training=True)

        upstream = np.ones_like(x)
        grad = layer.backward_propagation(upstream)
        expected = (1.0 - np.tanh(x) ** 2) * upstream
        np.testing.assert_allclose(grad, expected, rtol=1e-12, atol=1e-12)


class TestSoftmaxActivation(TestCase):

    def test_forward_is_stable_and_sums_to_one(self):
        # Use large values to stress numerical stability.
        x = np.array([[1000.0, 1001.0, 999.0],
                      [-1000.0, -999.0, -1001.0]])

        layer = SoftmaxActivation()
        layer.set_input_shape((x.shape[1],))
        out = layer.forward_propagation(x, training=True)

        # No NaNs or infs.
        self.assertFalse(np.any(~np.isfinite(out)))

        # Each row is a probability distribution.
        row_sums = np.sum(out, axis=1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-12, atol=1e-12)

        # Invariance to adding a constant shift per row.
        shifted = x + 12345.0
        out_shifted = layer.activation_function(shifted)
        np.testing.assert_allclose(out, out_shifted, rtol=1e-12, atol=1e-12)

    def test_backward_matches_finite_differences(self):
        # Validate the Jacobian-vector product via finite differences.
        rs = np.random.RandomState(1)
        x = rs.randn(1, 5)
        upstream = rs.randn(1, 5)

        layer = SoftmaxActivation()
        layer.set_input_shape((x.shape[1],))
        layer.forward_propagation(x, training=True)
        analytic = layer.backward_propagation(upstream)

        # Define a scalar function f(x) = sum(softmax(x) * upstream).
        def f(z):
            s = layer.activation_function(z)
            return float(np.sum(s * upstream))

        eps = 1e-6
        numeric = np.zeros_like(x)
        for i in range(x.shape[1]):
            x_plus = x.copy(); x_plus[0, i] += eps
            x_minus = x.copy(); x_minus[0, i] -= eps
            numeric[0, i] = (f(x_plus) - f(x_minus)) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)

    def test_backward_with_ones_is_zero(self):
        # If upstream gradient is all ones, f(x)=sum(softmax(x))=1, so gradient must be zero.
        x = np.random.RandomState(2).randn(4, 3)
        upstream = np.ones_like(x)

        layer = SoftmaxActivation()
        layer.set_input_shape((x.shape[1],))
        layer.forward_propagation(x, training=True)
        grad = layer.backward_propagation(upstream)

        np.testing.assert_allclose(grad, np.zeros_like(x), rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    import unittest

    unittest.main(verbosity=2)
