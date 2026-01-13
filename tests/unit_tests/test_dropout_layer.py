import sys
from pathlib import Path
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.neural_networks.layers import Dropout


class TestDropoutLayer(TestCase):

    def test_forward_training_applies_mask_and_scales(self):
        # Fix the RNG so the dropout mask is deterministic.
        np.random.seed(0)

        # Use an input of ones to make the expected output values easy to reason about.
        x = np.ones((20, 10), dtype=float)

        dropout = Dropout(probability=0.5)
        dropout.set_input_shape((10,))

        out = dropout.forward_propagation(x, training=True)

        # Dropout does not change the shape.
        self.assertEqual(out.shape, x.shape)

        # With inverted dropout and p=0.5, kept units are scaled by 1/(1-p) = 2.
        # Therefore, outputs should be either 0 (dropped) or 2 (kept, since input is 1).
        unique_vals = set(np.unique(out).tolist())
        self.assertTrue(unique_vals.issubset({0.0, 2.0}))

        # Ensure we actually dropped at least one unit (given the fixed seed).
        self.assertGreater(np.sum(out == 0.0), 0)

    def test_forward_inference_is_noop(self):
        x = np.random.RandomState(123).randn(5, 3)

        dropout = Dropout(probability=0.3)
        dropout.set_input_shape((3,))

        out = dropout.forward_propagation(x, training=False)

        # In inference, dropout does nothing: output equals input.
        np.testing.assert_array_equal(out, x)

    def test_backward_multiplies_by_mask(self):
        np.random.seed(0)

        x = np.ones((4, 6), dtype=float)
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape((6,))

        # Run a training forward pass to generate and store the mask.
        dropout.forward_propagation(x, training=True)
        self.assertIsNotNone(dropout.mask)

        # Gradient flowing from the next layer.
        output_error = np.ones_like(x)

        # Backward propagation should block gradients where mask == 0.
        input_error = dropout.backward_propagation(output_error)
        # With inverted dropout, gradients are also scaled by the same factor used in forward.
        np.testing.assert_array_equal(input_error, output_error * dropout.mask * dropout.scaling_factor)

    def test_output_shape_and_parameters(self):
        dropout = Dropout(probability=0.2)
        dropout.set_input_shape((7,))

        self.assertEqual(dropout.output_shape(), (7,))
        self.assertEqual(dropout.parameters(), 0)


if __name__ == '__main__':
    import unittest

    unittest.main(verbosity=2)
