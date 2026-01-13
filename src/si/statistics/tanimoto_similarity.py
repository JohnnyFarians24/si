import numpy as np

def tanimoto_similarity(x, y):
    """Compute Tanimoto (Jaccard) distance between 1D binary sample x and each row of y.

    Parameters
    x : array-like (n_features,) binary/0-1
    y : array-like (n_samples, n_features) binary/0-1 rows

    Returns
    distances : ndarray (n_samples,) = 1 - (|A ∩ B| / |A ∪ B|);
        empty union (both all zeros) gives distance 0.
    """
    # Convert inputs to numpy arrays
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    # Check expected dimensions
    if x_arr.ndim != 1:
        raise ValueError("x must be 1D")
    if y_arr.ndim != 2:
        raise ValueError("y must be 2D")
    if y_arr.shape[1] != x_arr.shape[0]:
        raise ValueError("Feature dimension mismatch between x and y")

    # Cast to boolean (non-zero -> True)
    x_bool = x_arr.astype(bool)
    y_bool = y_arr.astype(bool)

    # Intersection count per row: AND then sum
    intersection = np.sum(y_bool & x_bool, axis=1)
    # Individual set sizes
    x_sum = np.sum(x_bool)
    y_sum = np.sum(y_bool, axis=1)
    # Union size: |A| + |B| - |A∩B|
    union = x_sum + y_sum - intersection

    # Allocate similarity result
    similarity = np.empty_like(union, dtype=float)
    # Rows with empty union (both vectors all zeros)
    zero_union_mask = union == 0
    # Define similarity for empty union as 1.0
    similarity[zero_union_mask] = 1.0
    # Regular similarity otherwise
    similarity[~zero_union_mask] = intersection[~zero_union_mask] / union[~zero_union_mask]

    # Distance = 1 - similarity
    distances = 1.0 - similarity
    return distances

__all__ = ["tanimoto_similarity"]