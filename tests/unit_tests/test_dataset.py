import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dropna_removes_rows_and_updates_y(self):
        # Build X with NaNs and y aligned to rows
        X = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0], [np.nan, 6.0, 7.0]])
        y = np.array([10, 20, 30])
        ds = Dataset(X, y)

        # Drop rows containing NaN; ensure y stays in sync
        res = ds.dropna()
        self.assertIs(res, ds)
        self.assertEqual(ds.X.shape, (1, 3))
        self.assertEqual(ds.y.shape, (1,))
        np.testing.assert_array_equal(ds.X, np.array([[3.0, 4.0, 5.0]]))
        np.testing.assert_array_equal(ds.y, np.array([20]))

    def test_fillna_with_median_replaces_nans(self):
        # Prepare X with NaNs for median imputation
        X = np.array([[30.0, 2.0, np.nan], [3.0, 4.0, 5.0], [np.nan, 6.0, 7.0]])
        ds = Dataset(X)

        # Replace NaNs with per-feature medians
        res = ds.fillna(value="median")
        self.assertIs(res, ds)
        expected = np.array([[30.0, 2.0, 6.0], [3.0, 4.0, 5.0], [16.5, 6.0, 7.0]])
        np.testing.assert_allclose(ds.X, expected)

    def test_remove_by_index_updates_X_and_y(self):
        # Remove a single row by index; X and y must remain aligned
        X = np.array([[30.0, 2.0, 0.0], [3.0, 4.0, 5.0], [1.0, 6.0, 7.0]])
        y = np.array([10, 20, 30])
        ds = Dataset(X, y)

        # Delete row at index 1
        res = ds.remove_by_index(index=1)
        self.assertIs(res, ds)
        np.testing.assert_array_equal(ds.X, np.array([[30.0, 2.0, 0.0], [1.0, 6.0, 7.0]]))
        np.testing.assert_array_equal(ds.y, np.array([10, 30]))

    def test_remove_by_index_out_of_bounds_raises(self):
        # Deleting an invalid index should raise an error
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        ds = Dataset(X)
        with self.assertRaises((IndexError, ValueError)):
            ds.remove_by_index(index=5)
