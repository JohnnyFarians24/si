import sys
from pathlib import Path
import unittest
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split


class TestStratifiedTrainTestSplit(TestCase):

    def setUp(self):
        iris_csv = PROJECT_ROOT / 'datasets' / 'iris' / 'iris.csv'
        self.dataset = read_csv(filename=str(iris_csv), features=True, label=True)

    def test_shapes_and_labels_present(self):
        # Split while preserving class stratification.
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=0)
        # Ensure all samples are accounted for and labels exist.
        self.assertEqual(train.X.shape[0] + test.X.shape[0], self.dataset.X.shape[0])
        self.assertIsNotNone(train.y)
        self.assertIsNotNone(test.y)

    def test_class_proportions_preserved(self):
        # Perform stratified split and compare per-class counts.
        train, test = stratified_train_test_split(self.dataset, test_size=0.3, random_state=42)
        # Compute class distributions.
        def counts(y):
            unique, c = np.unique(y, return_counts=True)
            return dict(zip(unique, c))
        total = counts(self.dataset.y)
        tr = counts(train.y)
        te = counts(test.y)
        # For each class, test count should be round(total * test_size).
        for cls, tot in total.items():
            expected_test = int(round(tot * 0.3))
            self.assertEqual(te.get(cls, 0), expected_test)
            self.assertEqual(tr.get(cls, 0) + te.get(cls, 0), tot)

    def test_invalid_params(self):
        # Invalid test_size values should raise ValueError.
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.dataset, test_size=0.0)
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.dataset, test_size=1.0)


if __name__ == '__main__':
    unittest.main()

