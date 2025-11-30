import numpy as np

def rmse(y_true, y_pred) -> float:
    """Compute Root Mean Squared Error (RMSE).

    y_true : array-like true targets
    y_pred : array-like predicted targets
    Returns float RMSE.
    """
    # Convert inputs to float arrays
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Shape check (must match)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    # Residuals
    err = y_true - y_pred
    # Mean squared error then square root
    return float(np.sqrt(np.mean(err ** 2)))

__all__ = ["rmse"]