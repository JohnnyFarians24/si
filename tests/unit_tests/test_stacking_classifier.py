import sys
from pathlib import Path
import unittest
from unittest import TestCase

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression


class TestStackingClassifier(TestCase):

    def setUp(self):
        # Exercise protocol: use breast-bin.csv, split train/test.
        breast_csv = PROJECT_ROOT / 'datasets' / 'breast_bin' / 'breast-bin.csv'
        self.dataset = read_data_file(filename=str(breast_csv), label=True, sep=",")
        self.train_ds, self.test_ds = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit_and_predict_shapes(self):
        # Base models (level-0)
        knn = KNNClassifier(k=3)
        lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=400, patience=5, scale=True)
        dt = DecisionTreeClassifier(min_sample_split=2, max_depth=10, mode='gini')

        # Final model (level-1)
        final_knn = KNNClassifier(k=3)

        model = StackingClassifier(models=[knn, lg, dt], final_model=final_knn)
        model.fit(self.train_ds)

        preds = model.predict(self.test_ds)
        self.assertEqual(preds.shape[0], self.test_ds.shape()[0])

    def test_score_reasonable(self):
        # This test checks end-to-end behavior and that accuracy is within [0, 1].
        knn = KNNClassifier(k=3)
        lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=400, patience=5, scale=True)
        dt = DecisionTreeClassifier(min_sample_split=2, max_depth=10, mode='gini')
        final_knn = KNNClassifier(k=3)

        model = StackingClassifier(models=[knn, lg, dt], final_model=final_knn)
        model.fit(self.train_ds)

        score = float(model.score(self.test_ds))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # On this dataset/protocol, we expect a decent score (avoid overly strict thresholds).
        self.assertGreaterEqual(score, 0.85)

    def test_predict_is_deterministic_given_fixed_split(self):
        # With deterministic base models and fixed split, predictions should be stable.
        knn = KNNClassifier(k=3)
        lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=400, patience=5, scale=True)
        dt = DecisionTreeClassifier(min_sample_split=2, max_depth=10, mode='gini')
        final_knn = KNNClassifier(k=3)

        model1 = StackingClassifier(models=[knn, lg, dt], final_model=final_knn)
        model2 = StackingClassifier(
            models=[KNNClassifier(k=3), LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=400, patience=5, scale=True),
                    DecisionTreeClassifier(min_sample_split=2, max_depth=10, mode='gini')],
            final_model=KNNClassifier(k=3),
        )

        model1.fit(self.train_ds)
        model2.fit(self.train_ds)

        p1 = model1.predict(self.test_ds)
        p2 = model2.predict(self.test_ds)
        np.testing.assert_array_equal(p1, p2)


if __name__ == '__main__':
    unittest.main()
