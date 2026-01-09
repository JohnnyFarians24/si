from typing import Tuple, Iterator

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.neural_networks.layers import Layer
from si.neural_networks.losses import LossFunction, MeanSquaredError
from si.neural_networks.optimizers import Optimizer, SGD
from si.metrics.mse import mse


class NeuralNetwork(Model):
    """
    It represents a neural network model that is made by a sequence of layers.
    """

    def __init__(self, epochs: int = 100, batch_size: int = 128, optimizer: Optimizer = SGD,
                 learning_rate: float = 0.01, verbose: bool = False, loss: LossFunction = MeanSquaredError,
                 metric: callable = mse, **kwargs):
        """
        Initialize the neural network.

        Parameters
        ----------
        epochs: int
            The number of epochs to train the neural network.
        batch_size: int
            The batch size to use for training the neural network.
        optimizer: Optimizer
            The optimizer to use for training the neural network.
        learning_rate: float
            The learning rate to use for training the neural network.
        verbose: bool
            Whether to print the loss and the metric at each epoch.
        loss: LossFunction
            The loss function to use for training the neural network.
        metric: callable
            The metric to use for training the neural network.
        **kwargs
            Additional keyword arguments passed to the optimizer.
        """
        # arguments
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer(learning_rate=learning_rate, **kwargs)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        # attributes
        self.layers = []
        self.history = {}

    def add(self, layer: Layer) -> 'NeuralNetwork':
        """
        Add a layer to the neural network.

        Parameters
        ----------
        layer: Layer
            The layer to add.

        Returns
        -------
        NeuralNetwork
            The neural network with the added layer.
        """
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def _get_mini_batches(self, X: np.ndarray, y: np.ndarray = None,
                          shuffle: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate mini-batches for the given data.

        Parameters
        ----------
        X: numpy.ndarray
            The feature matrix.
        y: numpy.ndarray
            The label vector.
        shuffle: bool
            Whether to shuffle the data or not.

        Returns
        -------
        Iterator[Tuple[numpy.ndarray, numpy.ndarray]]
            The mini-batches.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples - self.batch_size + 1, self.batch_size):
            if y is not None:
                yield X[indices[start:start + self.batch_size]], y[indices[start:start + self.batch_size]]
            else:
                yield X[indices[start:start + self.batch_size]], None

    def _forward_propagation(self, X: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        X: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def _backward_propagation(self, output_error: float) -> float:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def _fit(self, dataset: Dataset) -> 'NeuralNetwork':
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        for epoch in range(1, self.epochs + 1):
            # store mini-batch data for epoch loss and quality metrics calculation
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self._get_mini_batches(X, y):
                # Forward propagation
                output = self._forward_propagation(X_batch, training=True)
                # Backward propagation
                error = self.loss.derivative(y_batch, output)
                self._backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)

            # compute loss
            loss = self.loss.loss(y_all, output_x_all)

            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric:.4f}"
            else:
                metric_s = "NA"
                metric = 'NA'

            # save loss and metric for each epoch
            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the labels for the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict.

        Returns
        -------
        numpy.ndarray
            The predicted labels.
        """
        return self._forward_propagation(dataset.X, training=False)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the score of the neural network on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to score.
        predictions: np.ndarray
            Predictions

        Returns
        -------
        float
            The score of the neural network.
        """
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")


if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.neural_networks.layers import Layer, DenseLayer, Dropout
    from si.neural_networks.activation import TanhActivation, SoftmaxActivation
    from si.neural_networks.losses import MeanSquaredError, CategoricalCrossEntropy
    from si.metrics.mse import mse
    from si.metrics.accuracy import accuracy
    from si.io.csv_file import read_csv

    # training data
    dataset = read_csv('../../../datasets/iris/iris.csv', sep=',', features=True, label=True)
    # convert labels to one-hot encoding
    new_y = np.zeros((dataset.y.shape[0], 3))
    for i, label in enumerate(dataset.y):
        if label == 'Iris-setosa':
            new_y[i] = [1, 0, 0]
        elif label == 'Iris-versicolor':
            new_y[i] = [0, 1, 0]
        else:
            new_y[i] = [0, 0, 1]
    dataset.y = new_y

    # network
    net = NeuralNetwork(epochs=1000, batch_size=16, optimizer=SGD, learning_rate=0.01, verbose=True,
                        loss=CategoricalCrossEntropy, metric=accuracy)
    n_features = dataset.X.shape[1]
    net.add(DenseLayer(6, (n_features,)))
    net.add(TanhActivation())
    net.add(Dropout(0.25))
    net.add(DenseLayer(4))
    net.add(TanhActivation())
    net.add(DenseLayer(3))
    net.add(SoftmaxActivation())

    # train
    net.fit(dataset)

    # test
    out = net.predict(dataset)
    print(out[:3])
    out2 = net.predict(dataset)
    print(out2[:3])

    print(net.score(dataset))