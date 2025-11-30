import numpy as np
from typing import Optional, List

from si.base.transformer import Transformer
from si.data.dataset import Dataset


class PCA(Transformer):
    """Simple PCA via eigen-decomposition of the covariance matrix.

    Parameters
    n_components : int
        Number of components to retain (>0).

    After fit
    mean : (n_features,) sample mean used for centering
    components : (n_components, n_features) principal axes (eigenvectors)
    explained_variance : (n_components,) variance fraction per component
    """

    def __init__(self, n_components: int, **kwargs):
        super().__init__(**kwargs)
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        self.n_components = int(n_components)
        self.mean: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None
        self.explained_variance: Optional[np.ndarray] = None

    def _fit(self, dataset: Dataset) -> 'PCA':
        X = dataset.X
        # Center data (subtract feature means)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        # Covariance matrix (features as columns)
        cov = np.cov(X_centered, rowvar=False)
        # Eigenvalues / eigenvectors of covariance
        eigvals, eigvecs = np.linalg.eig(cov)
        # Remove any tiny imaginary parts from numerical errors
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        # Sort components by descending eigenvalue
        order = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[order]
        eigvecs_sorted = eigvecs[:, order]
        # Select top n_components
        n = min(self.n_components, eigvecs_sorted.shape[1])
        # Store principal axes as rows
        self.components = eigvecs_sorted[:, :n].T
        # Compute explained variance fraction
        total_var = np.sum(eigvals_sorted)
        if total_var <= 0:  # handle degenerate case
            self.explained_variance = np.zeros(n)
        else:
            self.explained_variance = (eigvals_sorted / total_var)[:n]
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        if self.components is None or self.mean is None:
            raise ValueError("PCA must be fitted before calling transform().")
        X = dataset.X
        # Center with stored mean
        X_centered = X - self.mean
        # Project onto principal axes
        X_reduced = np.dot(X_centered, self.components.T)
        # Name components PC1, PC2, ...
        features: List[str] = [f"PC{i+1}" for i in range(self.components.shape[0])]
        return Dataset(X=X_reduced, y=dataset.y, features=features, label=dataset.label)
