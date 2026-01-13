import sys
from pathlib import Path
import unittest
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split

class TestSplits(TestCase):

    def setUp(self):
        iris_csv = PROJECT_ROOT / 'datasets' / 'iris' / 'iris.csv'
        # Read iris into our Dataset abstraction (X features, y label).
        self.dataset = read_csv(filename=str(iris_csv), features=True, label=True)

    def test_train_test_split(self):
        # Basic sanity checks: sizes add up correctly.
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

        # Metadata should be preserved.
        np.testing.assert_array_equal(np.array(train.features), np.array(self.dataset.features))
        np.testing.assert_array_equal(np.array(test.features), np.array(self.dataset.features))
        self.assertEqual(train.label, self.dataset.label)
        self.assertEqual(test.label, self.dataset.label)

    def test_train_test_split_is_reproducible_with_same_seed(self):
        # Same random_state should produce the same split every time.
        train1, test1 = train_test_split(self.dataset, test_size=0.25, random_state=42)
        train2, test2 = train_test_split(self.dataset, test_size=0.25, random_state=42)

        np.testing.assert_array_equal(train1.X, train2.X)
        np.testing.assert_array_equal(test1.X, test2.X)
        np.testing.assert_array_equal(train1.y, train2.y)
        np.testing.assert_array_equal(test1.y, test2.y)

    def test_train_test_split_changes_with_different_seed(self):
        # Different seeds should (very likely) yield different splits.
        train1, test1 = train_test_split(self.dataset, test_size=0.25, random_state=1)
        train2, test2 = train_test_split(self.dataset, test_size=0.25, random_state=2)

        # It's possible (though extremely unlikely) that splits match; we guard against
        # false positives by checking at least one of train/test X differs.
        self.assertTrue(
            not np.array_equal(train1.X, train2.X) or not np.array_equal(test1.X, test2.X)
        )


if __name__ == '__main__':
    unittest.main()