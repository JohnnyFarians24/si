from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Tuple

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(model,
                         dataset: Dataset,
                         hyperparameter_grid: Dict[str, Tuple],
                         scoring: Callable = None,
                         cv: int = 5,
                         n_iter: int = 10) -> Dict[str, Any]:
    """Performs randomized search cross validation on a model.

    This function samples ``n_iter`` random hyperparameter combinations from the full
    cartesian product of ``hyperparameter_grid`` and evaluates each combination via
    k-fold cross validation.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid: Dict[str, Tuple]
        Hyperparameter grid (discrete distributions) to sample from.
        Keys are parameter names and values are iterables of candidate values.
    scoring: Callable
        Scoring function with signature ``scoring(y_true, y_pred)``.
        If None, uses ``model.score(dataset_test)``.
    cv: int
        Number of cross validation folds.
    n_iter: int
        Number of random hyperparameter combinations to test.

    Returns
    -------
    results: Dict[str, Any]
        Dictionary with:
        - 'hyperparameters': list of sampled hyperparameter dicts
        - 'scores': list of mean CV scores for each sampled combination
        - 'best_hyperparameters': dict with best combination
        - 'best_score': best score
    """
    # 1) Validate that the provided hyperparameter names exist on the model.
    #    This follows the same behavior as grid_search_cv.
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    # 2) Enumerate all possible hyperparameter combinations.
    #    Each combination is a tuple aligned with hyperparameter_grid.keys().
    all_combinations = list(itertools.product(*hyperparameter_grid.values()))
    total = len(all_combinations)

    if total == 0:
        raise ValueError("hyperparameter_grid must define at least one combination")
    if n_iter <= 0:
        raise ValueError("n_iter must be a positive integer")
    if n_iter > total:
        raise ValueError(
            f"n_iter={n_iter} is larger than the number of possible combinations ({total})."
        )

    # Sample a set of random combinations from the full set.
    # Using replace=False avoids evaluating the same configuration twice.
    sampled_indices = np.random.choice(total, size=n_iter, replace=False)

    results: Dict[str, Any] = {'scores': [], 'hyperparameters': []}

    # 3-6) For each sampled combination: set model parameters, cross-validate, store results.
    for idx in sampled_indices:
        combination = all_combinations[int(idx)]

        # Build a dict of the current hyperparameter configuration.
        parameters: Dict[str, Any] = {}

        # 3) Set the model hyperparameters for this trial.
        for parameter, value in zip(hyperparameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # 4) Cross validate using the existing k_fold_cross_validation function.
        scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # 5) Store the mean CV score and the parameters that produced it.
        results['scores'].append(float(np.mean(scores)))
        results['hyperparameters'].append(parameters)

    # 7) Pick the best trial.
    best_idx = int(np.argmax(results['scores']))
    results['best_hyperparameters'] = results['hyperparameters'][best_idx]
    results['best_score'] = results['scores'][best_idx]

    return results
