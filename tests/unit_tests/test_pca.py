import os
import sys
from pathlib import Path
import unittest
from unittest import TestCase

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.decomposition.pca import PCA
from si.data.dataset import Dataset
from si.io.csv_file import read_csv


class TestPCA(TestCase):

    def setUp(self):
        # This dataset is used as a realistic, non-trivial input for PCA.
        self.iris_csv = PROJECT_ROOT / 'datasets' / 'iris' / 'iris.csv'
        # Read iris into our Dataset abstraction (X features, y label).
        self.ds = read_csv(filename=str(self.iris_csv), features=True, label=True)

    def test_init_invalid_n_components_raises(self):
        # PCA should reject non-positive number of components.
        with self.assertRaises(ValueError):
            PCA(n_components=0)
        with self.assertRaises(ValueError):
            PCA(n_components=-1)

    def test_transform_before_fit_raises(self):
        # Calling transform() before fit() should fail because mean/components
        # are not defined yet.
        pca = PCA(n_components=2)
        with self.assertRaises(ValueError):
            pca.transform(self.ds)

    def test_fit_sets_attributes_and_shapes(self):
        # After fitting, PCA should expose:
        # - mean: used for centering
        # - components: principal axes (eigenvectors)
        # - explained_variance: variance ratio per component
        pca = PCA(n_components=2)
        pca.fit(self.ds)

        self.assertIsNotNone(pca.mean)
        self.assertIsNotNone(pca.components)
        self.assertIsNotNone(pca.explained_variance)

        self.assertEqual(pca.mean.shape[0], self.ds.X.shape[1])
        self.assertEqual(pca.components.shape, (2, self.ds.X.shape[1]))
        self.assertEqual(pca.explained_variance.shape, (2,))

        # Explained variance ratios should be within [0, 1] and sum <= 1.
        # (Numerical tolerance is included.)
        self.assertTrue(np.all(pca.explained_variance >= 0))
        self.assertTrue(np.all(pca.explained_variance <= 1))
        self.assertLessEqual(float(np.sum(pca.explained_variance)), 1.0 + 1e-8)

        # The selected principal axes should be (approximately) orthonormal.
        # For orthonormal vectors, the Gram matrix equals the identity.
        gram = pca.components @ pca.components.T
        np.testing.assert_allclose(gram, np.eye(2), rtol=1e-5, atol=1e-6)

    def test_transform_reduces_dim_and_preserves_metadata(self):
        # transform() should:
        # - reduce X to (n_samples, n_components)
        # - keep labels y and label name
        # - generate new feature names (PC1, PC2, ...)
        pca = PCA(n_components=2)
        pca.fit(self.ds)
        ds_red = pca.transform(self.ds)

        self.assertEqual(ds_red.X.shape, (self.ds.X.shape[0], 2))
        self.assertEqual(ds_red.features, ['PC1', 'PC2'])
        self.assertEqual(ds_red.label, self.ds.label)
        np.testing.assert_array_equal(ds_red.y, self.ds.y)

        # Because PCA centers data using the fitted mean, the projected components
        # should be close to zero-mean.
        np.testing.assert_allclose(np.mean(ds_red.X, axis=0), np.zeros(2), atol=1e-10)

    def test_n_components_greater_than_n_features_caps(self):
        # If n_components is larger than the number of original features,
        # PCA should cap the number of retained components to n_features.
        # Create a tiny dataset with 2 features to test this behavior.
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])
        ds_small = Dataset(X=X, y=y, features=['f1', 'f2'], label='y')

        pca = PCA(n_components=10)
        pca.fit(ds_small)
        ds_red = pca.transform(ds_small)

        # With only 2 original features, we can only have 2 principal components.
        self.assertEqual(pca.components.shape, (2, 2))
        self.assertEqual(ds_red.X.shape, (3, 2))
        self.assertEqual(ds_red.features, ['PC1', 'PC2'])


if __name__ == '__main__':
    unittest.main()
