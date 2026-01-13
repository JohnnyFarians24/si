import sys
from pathlib import Path
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.io.data_file import read_data_file
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression


class TestRandomizedSearchCV(TestCase):

    def setUp(self):
        # Use the exact dataset requested by the exercise protocol.
        csv_file = PROJECT_ROOT / 'datasets' / 'breast_bin' / 'breast-bin.csv'
        self.dataset = read_data_file(filename=str(csv_file), label=True, sep=",")

    def test_randomized_search_protocol(self):
        # Make results reproducible by fixing NumPy's RNG state.
        # This affects both: sampling hyperparameter combinations and the CV shuffling.
        np.random.seed(42)

        model = LogisticRegression()

        # Discrete distributions as specified in the exercise statement.
        # Note: np.linspace returns floats; LogisticRegression should accept them for l2_penalty/alpha.
        parameter_distributions = {
            'l2_penalty': tuple(np.linspace(1, 10, 10)),
            'alpha': tuple(np.linspace(0.001, 0.0001, 100)),
            'max_iter': tuple(np.linspace(1000, 2000, 200).astype(int)),
        }

        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=parameter_distributions,
            cv=3,
            n_iter=10,
        )

        # The result dictionary must include the keys requested by the spec.
        self.assertIn('hyperparameters', results)
        self.assertIn('scores', results)
        self.assertIn('best_hyperparameters', results)
        self.assertIn('best_score', results)

        # n_iter controls how many configurations are evaluated.
        self.assertEqual(len(results['hyperparameters']), 10)
        self.assertEqual(len(results['scores']), 10)

        # Every trial should have a score in [0, 1].
        for s in results['scores']:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

        # best_score should be equal to the maximum of the sampled scores.
        self.assertEqual(results['best_score'], max(results['scores']))

        # best_hyperparameters should be one of the tried hyperparameter configurations.
        self.assertIn(results['best_hyperparameters'], results['hyperparameters'])

        # Each hyperparameter dict must contain exactly the requested keys.
        for hp in results['hyperparameters']:
            self.assertEqual(set(hp.keys()), {'l2_penalty', 'alpha', 'max_iter'})


if __name__ == '__main__':
    import unittest

    unittest.main(verbosity=2)
