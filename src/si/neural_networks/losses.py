from abc import ABCMeta, abstractmethod

import numpy as np


class LossFunction(metaclass=ABCMeta):
    """Base class for neural-network loss functions."""

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the scalar loss value given targets and predictions."""
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute dLoss/d(y_pred) with the same shape as y_pred."""
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """Mean Squared Error (MSE) loss.

    Commonly used for regression tasks.

    Loss:
        MSE = mean((y_true - y_pred)^2)

    Derivative (w.r.t. y_pred):
        dMSE/dy_pred = 2 * (y_pred - y_true) / N
    where N is the number of samples.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n_samples = y_true.shape[0] if y_true.ndim > 0 else 1
        return (2.0 / n_samples) * (y_pred - y_true)


class BinaryCrossEntropy(LossFunction):
    """Binary Cross-Entropy (BCE) loss.

    Commonly used for binary classification when y_pred are probabilities in (0, 1).

    Loss:
        BCE = -mean(y*log(p) + (1-y)*log(1-p))

    Notes
    -----
    Uses np.clip to avoid log(0) and division by 0.
    """

    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Clip probabilities for numerical stability.
        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        n_samples = y_true.shape[0] if y_true.ndim > 0 else 1

        # d/dp [-y log p - (1-y) log(1-p)] = -(y/p) + (1-y)/(1-p)
        return (-(y_true / p) + ((1.0 - y_true) / (1.0 - p))) / n_samples


class CategoricalCrossEntropy(LossFunction):
    """Categorical Cross-Entropy (CCE) loss.

    Used for multi-class classification with one-hot encoded targets.

    Loss:
        CCE = -mean(sum(y_true * log(y_pred)))

    Notes
    -----
    Uses np.clip to avoid log(0) and division by 0.
    """

    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        p = np.clip(y_pred, self.eps, 1.0 - self.eps)

        # Sum across classes (last axis) and average across samples.
        per_sample = -np.sum(y_true * np.log(p), axis=-1)
        return float(np.mean(per_sample))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        n_samples = y_true.shape[0] if y_true.ndim > 0 else 1

        # d/dp [-sum(y_true * log(p))] = -(y_true / p)
        return (-(y_true / p)) / n_samples
