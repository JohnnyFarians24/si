from abc import ABCMeta, abstractmethod
import copy

import numpy as np

from si.neural_networks.optimizers import Optimizer


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training: bool):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, output_error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    
class DenseLayer(Layer):
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    
    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        # SHAPES: (batch_size, input_columns) = (batch_size, output_columns) * (output_columns, input_columns)
        input_error = np.dot(output_error, self.weights.T)

        # computes the weight error: dE/dW = X.T * dE/dY
        # SHAPES: (input_columns, output_columns) = (input_columns, batch_size) * (batch_size, output_columns)
        weights_error = np.dot(self.input.T, output_error)
        # computes the bias error: dE/dB = dE/dY
        # SHAPES: (1, output_columns) = SUM over the rows of a matrix of shape (batch_size, output_columns)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # updates parameters
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error
    
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,) 


class Dropout(Layer):
    """Dropout layer.

    Dropout is a regularization technique that randomly sets a fraction of the inputs
    to zero during training. To keep the expected activation magnitude unchanged, this
    implementation uses *inverted dropout*, i.e., scales activations by
    ``1 / (1 - probability)`` during training.

    Notes
    -----
    - During inference (`training=False`), dropout does nothing and returns the input.
    - This layer has no learnable parameters.
    """

    def __init__(self, probability: float):
        super().__init__()
        if probability < 0 or probability >= 1:
            # probability == 1 would drop everything and also makes the scaling factor undefined.
            raise ValueError("probability must be in the range [0, 1).")

        # Dropout rate (fraction of units to drop).
        self.probability = probability

        # Cached values from the forward pass.
        self.mask = None
        self.input = None
        self.output = None
        # Cached scaling factor used for inverted dropout.
        self.scaling_factor = 1.0

    def parameters(self) -> int:
        # Dropout has no trainable parameters.
        return 0

    def output_shape(self) -> tuple:
        # Dropout does not change the tensor shape.
        return self.input_shape()

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """Apply dropout mask during training, no-op during inference.

        Parameters
        ----------
        input: numpy.ndarray
            Input activations to apply dropout to.
        training: bool
            If True, apply dropout; if False, return the input unchanged.
        """
        self.input = input

        if not training:
            # In inference mode, dropout is disabled.
            self.mask = None
            self.scaling_factor = 1.0
            self.output = input
            return self.output

        # In training mode:
        # 1) Compute scaling factor for inverted dropout.
        self.scaling_factor = 1.0 / (1.0 - self.probability) if self.probability != 0 else 1.0

        # 2) Sample a binary mask where 1 means "keep" and 0 means "drop".
        #    Probability of keeping a unit is (1 - probability).
        self.mask = np.random.binomial(1, 1.0 - self.probability, size=input.shape)

        # 3) Apply mask and scaling.
        self.output = input * self.mask * self.scaling_factor
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """Backpropagate through dropout.

        The gradient is blocked for the dropped units by multiplying by the same mask
        used in the forward pass.
        """
        if self.mask is None:
            # If dropout was not applied (inference mode), pass gradients through unchanged.
            return output_error

        # In inverted dropout, the forward pass scales activations by `scaling_factor`,
        # therefore gradients should be scaled in the same way.
        return output_error * self.mask * self.scaling_factor