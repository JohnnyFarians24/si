from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """Basic random split into train and test.

    Parameters
    dataset : Dataset
    test_size : float in (0,1) fraction for test
    random_state : int seed

    Returns (train, test) as Dataset objects preserving metadata.
    """
    np.random.seed(random_state)  # set seed for reproducible shuffling
    n_samples = dataset.shape()[0]  # total samples
    n_test = int(n_samples * test_size)  # test count (floor/truncate)
    permutations = np.random.permutation(n_samples)  # shuffled indices
    test_idxs = permutations[:n_test]  # first slice -> test
    train_idxs = permutations[n_test:]  # remainder -> train
    # Build Dataset objects with same features/label
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """Stratified split preserving per-class proportions.

    Parameters
    dataset : labeled Dataset
    test_size : float in (0,1) fraction for test
    random_state : int seed

    Returns (train, test) stratified datasets.
    """
    if not dataset.has_label():  # need labels for stratification
        raise ValueError("Stratified split requires labeled dataset (y).")
    if not (0 < test_size < 1):  # validate range
        raise ValueError("test_size must be in (0, 1)")

    np.random.seed(random_state)  # set seed for reproducible per-class shuffle

    y = dataset.y
    classes = dataset.get_classes()  # unique class labels
    train_indices = []
    test_indices = []

    for cls in classes:  # process each class separately
        cls_indices = np.where(y == cls)[0]  # indices of this class
        n_cls = cls_indices.size
        n_test_cls = int(np.round(n_cls * test_size))  # desired per-class test count
        n_test_cls = max(1 if n_cls > 0 else 0, n_test_cls)  # ensure at least 1 if class exists
        n_test_cls = min(n_test_cls, n_cls)  # cannot exceed available samples
        perm = np.random.permutation(cls_indices)  # shuffle class indices
        test_cls = perm[:n_test_cls]  # test slice
        train_cls = perm[n_test_cls:]  # train slice
        test_indices.append(test_cls)
        train_indices.append(train_cls)

    if len(train_indices) == 0:  # sanity check
        raise ValueError("No samples available for training after stratified split.")

    train_idx = np.concatenate(train_indices)  # merge class-wise train indices (order not preserved)
    test_idx = np.concatenate(test_indices)  # merge class-wise test indices (order not preserved)

    # Build output datasets with original metadata
    train = Dataset(dataset.X[train_idx], dataset.y[train_idx], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idx], dataset.y[test_idx], features=dataset.features, label=dataset.label)
    return train, test