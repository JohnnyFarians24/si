import unittest
import numpy as np

from si.statistics.tanimoto_similarity import tanimoto_similarity


class TestTanimotoSimilarity(unittest.TestCase):

    def test_basic_cases(self):
        # Binary query vector
        x = np.array([1, 0, 1, 0], dtype=int)
        # Compare against several candidate vectors
        y = np.array([
            [1, 0, 1, 0],  # identical -> distance 0
            [1, 1, 0, 0],  # partial overlap -> distance 2/3
            [0, 0, 0, 0],  # empty vs non-empty -> distance 1
            [0, 1, 0, 1]   # disjoint -> distance 1
        ], dtype=int)
        # Compute Tanimoto distance for each candidate
        d = tanimoto_similarity(x, y)
        expected = np.array([0.0, 1 - (1/3), 1.0, 1.0])
        np.testing.assert_allclose(d, expected)

    def test_all_zero_vectors(self):
        # Both vectors empty -> defined distance 0
        x = np.array([0, 0, 0])
        y = np.array([[0, 0, 0], [0, 0, 0]])
        d = tanimoto_similarity(x, y)
        np.testing.assert_array_equal(d, np.array([0.0, 0.0]))

    def test_shape_validation(self):
        # x must be 1D
        x = np.array([[1, 0, 1]])  # not 1D
        y = np.array([[1, 0, 1]])
        with self.assertRaises(ValueError):
            tanimoto_similarity(x, y)
        # y must be 2D
        x2 = np.array([1, 0])
        y2 = np.array([1, 0, 1])  # not 2D
        with self.assertRaises(ValueError):
            tanimoto_similarity(x2, y2)
        # x and y must have matching number of features
        x3 = np.array([1, 0, 1])
        y3 = np.array([[1, 0]])  # feature mismatch
        with self.assertRaises(ValueError):
            tanimoto_similarity(x3, y3)

if __name__ == '__main__':
    unittest.main()