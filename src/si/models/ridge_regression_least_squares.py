import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """Closed-form ridge regression (L2 regularized least squares).

    l2_penalty : float lambda value
    scale : bool standardize features before solving

    After fit: theta_zero (intercept), theta (coeffs), mean/std if scaled.
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = float(l2_penalty)  # strength of L2 regularization (lambda)
        self.scale = bool(scale)  # standardize X before fitting/predicting
        self.theta = None  # coefficients (without intercept)
        self.theta_zero = None  # intercept term
        self.mean = None  # feature means (if scaling)
        self.std = None  # feature stds (if scaling)

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        X = dataset.X
        y = dataset.y

        # Optional feature scaling (store mean & std)
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            std_safe = np.where(self.std == 0, 1.0, self.std)  # avoid divide by zero
            X_scaled = (X - self.mean) / std_safe
        else:
            X_scaled = X

        m, n = X_scaled.shape  # samples, features

        # Add intercept column of ones
        X_aug = np.c_[np.ones((m, 1)), X_scaled]

        # Build regularization matrix (no penalty on intercept)
        I = np.eye(n + 1)
        I[0, 0] = 0.0
        reg = self.l2_penalty * I

        # Closed-form solution: (X^T X + λI)^{-1} X^T y
        XtX = X_aug.T @ X_aug
        XtY = X_aug.T @ y
        theta_all = np.linalg.inv(XtX + reg) @ XtY

        # Split intercept and coefficients
        self.theta_zero = float(theta_all[0])
        self.theta = theta_all[1:]
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        X = dataset.X
        # Apply stored scaling if enabled
        if self.scale:
            std_safe = np.where(self.std == 0, 1.0, self.std)
            X = (X - self.mean) / std_safe
        m = X.shape[0]
        # Add intercept column and compute linear combination
        X_aug = np.c_[np.ones((m, 1)), X]
        theta_all = np.r_[self.theta_zero, self.theta]
        return X_aug @ theta_all

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        return mse(dataset.y, predictions)  # lower MSE is better
