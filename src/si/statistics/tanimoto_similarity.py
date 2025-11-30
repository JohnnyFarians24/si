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
    # Convert inputs to ndarray
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    # Validate dimensionality
    if x_arr.ndim != 1:
        raise ValueError("x must be 1D")
    if y_arr.ndim != 2:
        raise ValueError("y must be 2D")
    if y_arr.shape[1] != x_arr.shape[0]:
        raise ValueError("Feature dimension mismatch between x and y")

    # Cast to boolean (treat non‑zero as True)
    x_bool = x_arr.astype(bool)
    y_bool = y_arr.astype(bool)

    # Intersection counts per row (logical AND then sum)
    intersection = np.sum(y_bool & x_bool, axis=1)
    # Individual set sizes
    x_sum = np.sum(x_bool)
    y_sum = np.sum(y_bool, axis=1)
    # Union count formula: |A| + |B| - |A∩B|
    union = x_sum + y_sum - intersection

    # Allocate similarity array
    similarity = np.empty_like(union, dtype=float)
    # Identify rows with empty union (both all zeros)
    zero_union_mask = union == 0
    # Similarity is 1.0 when union is empty
    similarity[zero_union_mask] = 1.0
    # Regular similarity otherwise
    similarity[~zero_union_mask] = intersection[~zero_union_mask] / union[~zero_union_mask]

    # Convert similarity to distance
    distances = 1.0 - similarity
    return distances

__all__ = ["tanimoto_similarity"]