import sys
from pathlib import Path
import unittest
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.data.dataset import Dataset
from si.models.knn_regressor import KNNRegressor


class TestKNNRegressor(TestCase):

    def test_init_invalid_k_raises(self):
        # k must be a positive integer.
        with self.assertRaises(ValueError):
            KNNRegressor(k=0)
        with self.assertRaises(ValueError):
            KNNRegressor(k=-3)

    def test_predict_before_fit_raises(self):
        # Predicting before fitting should fail because the training dataset is missing.
        model = KNNRegressor(k=1)
        ds = Dataset(X=np.array([[0.0], [1.0]]), y=np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            model.predict(ds)

    def test_predict_returns_mean_of_k_nearest_targets(self):
        # Build a tiny 1D regression problem where neighbors are obvious.
        # Train points: x=[0,1,2] with y=[0,1,2].
        train = Dataset(X=np.array([[0.0], [1.0], [2.0]]), y=np.array([0.0, 1.0, 2.0]))

        # Query a point near 1 and 2. With k=2, the nearest are x=1 and x=2.
        test = Dataset(X=np.array([[1.1]]), y=np.array([0.0]))  # y not used for predict

        model = KNNRegressor(k=2)
        model.fit(train)
        pred = model.predict(test)

        # Expected mean of targets of neighbors (1 and 2) => 1.5
        self.assertEqual(pred.shape, (1,))
        self.assertAlmostEqual(float(pred[0]), 1.5, places=12)

    def test_score_uses_rmse(self):
        # If we predict the same points as training with k=1,
        # each point's nearest neighbor is itself, so predictions equal y and RMSE=0.
        ds = Dataset(X=np.array([[0.0], [1.0], [2.0]]), y=np.array([0.0, 1.0, 2.0]))
        model = KNNRegressor(k=1)
        model.fit(ds)
        score = model.score(ds)
        self.assertAlmostEqual(float(score), 0.0, places=12)

    def test_k_greater_than_training_size_is_handled(self):
        # If k > n_train, numpy slicing will just take all samples.
        train = Dataset(X=np.array([[0.0], [10.0]]), y=np.array([0.0, 10.0]))
        test = Dataset(X=np.array([[5.0]]), y=np.array([0.0]))

        model = KNNRegressor(k=10)
        model.fit(train)
        pred = model.predict(test)

        # With all samples used, prediction is mean(y) = 5.
        self.assertAlmostEqual(float(pred[0]), 5.0, places=12)


if __name__ == '__main__':
    unittest.main()
