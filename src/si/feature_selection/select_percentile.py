from typing import Callable
import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """Select top scoring features by percentile.

    Parameters
    score_func : callable (dataset -> (scores, p_values)), default f_classification
    percentile : float in (0,100] percentage of features to keep

    After fit
    F : (n_features,) scores
    p : (n_features,) p-values
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 10, **kwargs):
        super().__init__(**kwargs)
        # Validate percentile range
        if not (0 < percentile <= 100):
            raise ValueError("percentile must be in (0, 100]")
        self.score_func = score_func  # scoring function
        self.percentile = percentile  # retention percentage
        self.F = None  # feature scores
        self.p = None  # feature p-values

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        # Compute feature scores and p-values
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        n_features = dataset.X.shape[1]
        # Compute number of features to retain (ceil ensures at least 1)
        k = max(1, int(np.ceil(n_features * (self.percentile / 100.0))))
        if k >= n_features:
            # Retain all features
            idxs = np.arange(n_features)
        else:
            # Score threshold: value at (100 - percentile) percentile of scores
            threshold = float(np.percentile(self.F, 100.0 - self.percentile))
            greater_idxs = np.where(self.F > threshold)[0]  # strictly above threshold

            if greater_idxs.size >= k:
                # Enough strictly greater scores: select top-k among them
                order = greater_idxs[np.argsort(self.F[greater_idxs])[::-1]]
                idxs = order[:k]
            else:
                # Need to include ties at the threshold to reach k
                equal_idxs = np.where(self.F == threshold)[0]
                need = k - greater_idxs.size
                # Sort strictly greater by descending score, ties by ascending index for determinism
                greater_ordered = greater_idxs[np.argsort(self.F[greater_idxs])[::-1]]
                ties_ordered = np.sort(equal_idxs)[:need]
                idxs = np.concatenate([greater_ordered, ties_ordered])

        # Map selected indices to feature names if available
        features = None
        if dataset.features is not None:
            features = list(np.array(dataset.features)[idxs])
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=features, label=dataset.label)
