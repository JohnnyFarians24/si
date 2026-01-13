import sys
from pathlib import Path
import unittest
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.metrics.rmse import rmse


class TestRMSE(TestCase):

    def test_rmse_zero_when_perfect_prediction(self):
        # When predictions match targets exactly, RMSE must be 0.
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.0, 1.0, 2.0])
        self.assertEqual(rmse(y_true, y_pred), 0.0)

    def test_rmse_known_value(self):
        # Simple hand-checkable example:
        # errors = [1, -1] -> squared = [1, 1] -> mean = 1 -> sqrt = 1
        y_true = [0, 0]
        y_pred = [1, -1]
        self.assertAlmostEqual(rmse(y_true, y_pred), 1.0, places=12)

    def test_rmse_returns_python_float(self):
        # Ensure the function returns a built-in float (not a numpy scalar).
        out = rmse([1, 2, 3], [1, 2, 4])
        self.assertIsInstance(out, float)

    def test_rmse_shape_mismatch_raises(self):
        # Input shapes must match; otherwise the metric is undefined.
        with self.assertRaises(ValueError):
            rmse([1, 2, 3], [1, 2])


if __name__ == '__main__':
    unittest.main()
