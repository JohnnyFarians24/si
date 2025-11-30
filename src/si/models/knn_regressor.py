from typing import Callable
import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """Simple k-NN regressor.

    Predicts mean target of the k closest training samples.
    k : int > 0 number of neighbors
    distance : callable(sample, X_train) -> distances array

    After fit: dataset stored for neighbor lookups.
    """

    def __init__(self, k: int = 3, distance: Callable = euclidean_distance, **kwargs):
        super().__init__(**kwargs)
        if k <= 0:  # validate k
            raise ValueError("k must be a positive integer")
        self.k = int(k)  # number of neighbors
        self.distance = distance  # distance function
        self.dataset: Dataset | None = None  # training data reference

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        self.dataset = dataset  # store training set
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        if self.dataset is None:
            raise ValueError("Model must be fitted before predicting.")
        # Predict a single sample: distances -> k indices -> mean target
        def predict_one(sample: np.ndarray) -> float:
            distances = self.distance(sample, self.dataset.X)  # distance to all train samples
            k_idx = np.argsort(distances)[:self.k]  # indices of k nearest
            return float(np.mean(self.dataset.y[k_idx]))  # mean target value
        # Apply over all rows in input X
        return np.apply_along_axis(predict_one, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        return rmse(dataset.y, predictions)  # lower RMSE is better
