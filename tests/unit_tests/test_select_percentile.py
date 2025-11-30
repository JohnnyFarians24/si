from unittest import TestCase
import numpy as np
import os
from datasets import DATASETS_PATH

from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification


class TestSelectPercentile(TestCase):

    def setUp(self):
        # Load iris dataset for feature selection tests
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        # Fit SelectPercentile and verify score arrays match feature count
        sp = SelectPercentile(score_func=f_classification, percentile=40)
        sp.fit(self.dataset)
        self.assertEqual(self.dataset.X.shape[1], sp.F.shape[0])
        self.assertEqual(self.dataset.X.shape[1], sp.p.shape[0])

    def test_transform_shape_matches_percentile(self):
        # After transform, number of features should match requested percentile
        pct = 40
        n_features = self.dataset.X.shape[1]
        expected_k = max(1, int(np.ceil(n_features * pct / 100.0)))
        sp = SelectPercentile(score_func=f_classification, percentile=pct)
        sp.fit(self.dataset)
        new_ds = sp.transform(self.dataset)
        self.assertEqual(expected_k, new_ds.X.shape[1])
        self.assertLess(new_ds.X.shape[1], self.dataset.X.shape[1])

    def test_ties_handling_at_threshold(self):
        # Create a dataset with 10 features and tie scores at the threshold
        X = np.arange(50, dtype=float).reshape(5, 10)
        y = np.array([0, 1, 0, 1, 0])
        ds = read_csv(filename=self.csv_file, features=True, label=True)
        # Override data to ensure 10 features and custom labels
        ds.X = X
        ds.y = y
        ds.features = [f"f{i}" for i in range(10)]

        # Predefined F-scores with ties and zero p-values for simplicity
        F_values = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
        p_values = np.zeros_like(F_values)

        # Dummy scorer returns the fixed F and p arrays
        def dummy_score_func(_):
            return F_values, p_values

        sp = SelectPercentile(score_func=dummy_score_func, percentile=40)
        sp.fit(ds)
        new_ds = sp.transform(ds)
        # Expect k=4: select scores > 5.6 (6,7) then the first two ties at 5.6 (3,5)
        expected_indices = [6, 7, 3, 5]
        expected_features = [f"f{i}" for i in expected_indices]
        self.assertEqual(expected_features, new_ds.features)
        self.assertEqual(4, new_ds.X.shape[1])
