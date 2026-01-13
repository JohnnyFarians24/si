import sys
from pathlib import Path
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.neural_networks.optimizers import Adam


class TestAdamOptimizer(TestCase):

    def test_first_update_matches_manual_formula(self):
        # On the first update (t=1), with m=v=0 initially, Adam reduces to:
        #   m = (1-beta1)*g, v = (1-beta2)*g^2
        #   m_hat = g, v_hat = g^2
        #   w_new = w - lr * g / (sqrt(g^2) + eps) = w - lr * g / (|g| + eps)
        w = np.array([1.0, -2.0, 3.0])
        g = np.array([0.1, -0.2, 0.0])

        opt = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        w_new = opt.update(w, g)

        expected = w - 0.01 * (g / (np.abs(g) + 1e-8))
        np.testing.assert_allclose(w_new, expected, rtol=1e-12, atol=1e-12)

        # State should be initialized.
        self.assertEqual(opt.t, 1)
        self.assertIsNotNone(opt.m)
        self.assertIsNotNone(opt.v)
        self.assertEqual(opt.m.shape, w.shape)
        self.assertEqual(opt.v.shape, w.shape)

    def test_second_update_changes_state(self):
        # Ensure that repeated updates modify the internal moments and advance time.
        w = np.array([1.0, 2.0])
        g1 = np.array([0.5, -0.5])
        g2 = np.array([0.25, -0.25])

        opt = Adam(learning_rate=0.001)
        w1 = opt.update(w, g1)
        t1, m1, v1 = opt.t, opt.m.copy(), opt.v.copy()

        w2 = opt.update(w1, g2)
        self.assertEqual(opt.t, t1 + 1)
        self.assertFalse(np.allclose(opt.m, m1))
        self.assertFalse(np.allclose(opt.v, v1))

        # Shape must be preserved.
        self.assertEqual(w2.shape, w.shape)


if __name__ == '__main__':
    import unittest

    unittest.main(verbosity=2)
