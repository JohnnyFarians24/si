import sys
from pathlib import Path
import unittest
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.data.dataset import Dataset
from si.models.random_forest_classifier import RandomForestClassifier


class TestRandomForestClassifier(TestCase):

    def test_init_invalid_n_estimators_raises(self):
        # n_estimators must be > 0.
        with self.assertRaises(ValueError):
            RandomForestClassifier(n_estimators=0)
        with self.assertRaises(ValueError):
            RandomForestClassifier(n_estimators=-5)

    def test_fit_requires_labels(self):
        # RandomForestClassifier requires both X and y.
        ds = Dataset(X=np.array([[1.0, 2.0], [3.0, 4.0]]), y=None)
        model = RandomForestClassifier(n_estimators=3, seed=0)
        with self.assertRaises(ValueError):
            model.fit(ds)

    def test_predict_before_fit_raises(self):
        # Predicting before fitting should fail.
        ds = Dataset(X=np.array([[1.0, 2.0], [3.0, 4.0]]), y=np.array([0, 1]))
        model = RandomForestClassifier(n_estimators=3, seed=0)
        with self.assertRaises(ValueError):
            model.predict(ds)

    def test_fit_creates_expected_number_of_trees_and_feature_subsets(self):
        # Fit should build exactly n_estimators trees and store feature indices per tree.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 9))
        y = rng.integers(0, 2, size=50)
        ds = Dataset(X=X, y=y)

        n_estimators = 7
        # For 9 features and max_features=None => sqrt(9) = 3 features per tree
        model = RandomForestClassifier(n_estimators=n_estimators, max_features=None, seed=123)
        model.fit(ds)

        self.assertEqual(len(model.trees), n_estimators)
        for feature_idx, tree in model.trees:
            # Each entry stores (feature_indices, DecisionTreeClassifier)
            self.assertEqual(feature_idx.shape, (3,))
            # Feature indices should be unique within a tree (sampled without replacement)
            self.assertEqual(len(np.unique(feature_idx)), feature_idx.shape[0])
            # Indices should be within [0, n_features)
            self.assertTrue(np.all(feature_idx >= 0))
            self.assertTrue(np.all(feature_idx < X.shape[1]))
            # Tree should be fitted (tree.tree exists)
            self.assertIsNotNone(getattr(tree, 'tree', None))

    def test_max_features_out_of_range_raises(self):
        # max_features must be in [1, n_features].
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 2, size=10)
        ds = Dataset(X=X, y=y)

        with self.assertRaises(ValueError):
            RandomForestClassifier(n_estimators=3, max_features=0, seed=0).fit(ds)
        with self.assertRaises(ValueError):
            RandomForestClassifier(n_estimators=3, max_features=10, seed=0).fit(ds)

    def test_predict_shape_and_reproducibility_with_seed(self):
        # With a fixed seed, training should be reproducible (same bootstrap/features).
        rng = np.random.default_rng(42)
        X = rng.normal(size=(120, 6))
        # Create a simple separable-ish target to reduce flakiness.
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        ds = Dataset(X=X, y=y)

        m1 = RandomForestClassifier(n_estimators=15, max_features=2, seed=7)
        m2 = RandomForestClassifier(n_estimators=15, max_features=2, seed=7)

        m1.fit(ds)
        m2.fit(ds)

        p1 = m1.predict(ds)
        p2 = m2.predict(ds)

        # Predictions should be a 1D array of length n_samples.
        self.assertEqual(p1.shape, (ds.shape()[0],))
        # Same seed should give same predictions.
        np.testing.assert_array_equal(p1, p2)

    def test_score_is_between_0_and_1(self):
        # score() uses accuracy, which must be in [0, 1].
        rng = np.random.default_rng(123)
        X = rng.normal(size=(80, 5))
        y = (X[:, 0] > 0).astype(int)
        ds = Dataset(X=X, y=y)

        model = RandomForestClassifier(n_estimators=10, max_features=2, seed=1)
        model.fit(ds)
        s = float(model.score(ds))
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)


if __name__ == '__main__':
    unittest.main()
