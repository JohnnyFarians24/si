import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
    """Stacking classifier (two-level ensemble).

    This ensemble trains a set of base models (level-0). Their predictions are then
    used as features to train a final model (level-1).

    Parameters
    ----------
    models : list[Model]
        Base models to train and use for generating meta-features.
    final_model : Model
        Model trained on the meta-features (base predictions) to produce final predictions.

    Notes
    -----
    This implementation assumes a classification setting.
    It is designed to work cleanly with the course protocol (breast-bin.csv => labels 0/1).
    """

    def __init__(self, models, final_model, **kwargs):
        super().__init__(**kwargs)
        # Level-0 (base) models whose predictions will become meta-features.
        self.models = models
        # Level-1 (meta) model trained on the base predictions.
        self.final_model = final_model

    @staticmethod
    def _stack_predictions(predictions_list: list[np.ndarray]) -> np.ndarray:
        """Convert a list of per-model prediction vectors into a 2D meta-feature matrix."""
        # Each base model returns a vector of predictions with shape (n_samples,).
        # We coerce every output into a 1D array, then stack them column-wise.
        # The resulting matrix has shape (n_samples, n_models) and is used as X for the meta-model.
        cols = []
        for pred in predictions_list:
            p = np.asarray(pred)
            if p.ndim != 1:
                # Some models may return column vectors (n_samples, 1); flatten to (n_samples,).
                p = p.reshape(-1)
            cols.append(p)
        # Stack into (n_samples, n_models)
        return np.column_stack(cols)

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        # 1) Fit each base model on the original training dataset.
        #    After this, every base model can generate predictions.
        for model in self.models:
            model.fit(dataset)

        # 2) Generate meta-features by running the base models on the same training data.
        #    Each column corresponds to one model's predicted class labels.
        base_preds = [model.predict(dataset) for model in self.models]
        X_meta = self._stack_predictions(base_preds)

        # 3) Train the final (level-1) model on the meta-features.
        #    The targets are the original labels.
        meta_ds = Dataset(X=X_meta, y=dataset.y, features=None, label=dataset.label)
        self.final_model.fit(meta_ds)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        # 1) Run the fitted base models on the input data to build meta-features.
        base_preds = [model.predict(dataset) for model in self.models]
        X_meta = self._stack_predictions(base_preds)

        # 2) Predict using the final model on top of the meta-features.
        meta_ds = Dataset(X=X_meta, y=dataset.y, features=None, label=dataset.label)
        return self.final_model.predict(meta_ds)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        return accuracy(dataset.y, predictions)
