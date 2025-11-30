from typing import List, Literal, Tuple
import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from .decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier(Model):
    """Basic random forest classifier (bagging + feature subsampling).

    n_estimators : int number of trees (>0)
    max_features : int | None features per tree (None -> sqrt)
    min_sample_split : int minimum samples to split (tree param)
    max_depth : int max tree depth
    mode : 'gini' | 'entropy' impurity
    seed : int | None RNG seed

    After fit: trees = list of (feature_indices, DecisionTreeClassifier).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: int | None = None,
        min_sample_split: int = 2,
        max_depth: int = 10,
        mode: Literal['gini', 'entropy'] = 'gini',
        seed: int | None = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        self.trees: List[Tuple[np.ndarray, DecisionTreeClassifier]] = []

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        if dataset is None or dataset.X is None or dataset.y is None:
            raise ValueError("Dataset must contain X and y for training")

        n_samples, n_features = dataset.shape()
        rng = np.random.default_rng(self.seed)  # RNG for bootstrap + feature sampling

        # Determine number of features per tree
        max_features = self.max_features
        if max_features is None:
            max_features = max(1, int(np.sqrt(n_features)))  # sqrt heuristic
        else:
            max_features = int(max_features)
            if not (1 <= max_features <= n_features):
                raise ValueError("max_features must be in [1, n_features]")

        self.trees = []  # reset tree list

        for _ in range(self.n_estimators):
            # Bootstrap sample rows (with replacement)
            sample_indices = rng.integers(0, n_samples, size=n_samples)
            # Feature subsampling (without replacement)
            feature_indices = rng.choice(n_features, size=max_features, replace=False)

            # Build bootstrap subset dataset
            X_boot = dataset.X[sample_indices][:, feature_indices]
            y_boot = dataset.y[sample_indices]
            features = None
            if dataset.features is not None:
                features = list(np.array(dataset.features)[feature_indices])
            boot_ds = Dataset(X=X_boot, y=y_boot, features=features, label=dataset.label)

            # Train decision tree on subset
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode,
            ).fit(boot_ds)

            # Store feature indices with trained tree
            self.trees.append((feature_indices, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        if not self.trees:
            raise ValueError("RandomForestClassifier must be fitted before predicting")

        n_samples = dataset.X.shape[0]
        all_preds = []  # per-tree predictions
        for feature_indices, tree in self.trees:
            X_sub = dataset.X[:, feature_indices]  # select tree's features
            ds_sub = Dataset(X=X_sub, y=dataset.y, features=None, label=dataset.label)
            preds = tree.predict(ds_sub)  # tree predictions
            all_preds.append(preds)

        votes = np.vstack(all_preds)  # shape (n_estimators, n_samples)

        # Majority vote over trees
        final_preds = []
        for j in range(n_samples):
            col = votes[:, j]
            labels, counts = np.unique(col, return_counts=True)
            final_preds.append(labels[np.argmax(counts)])
        return np.array(final_preds)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        return accuracy(dataset.y, predictions)  # higher accuracy is better
