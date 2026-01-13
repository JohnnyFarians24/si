from abc import abstractmethod
from typing import Union

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

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
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0
    
class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input >= 0, 1, 0)


class TanhActivation(ActivationLayer):
    """Hyperbolic tangent activation.

    The tanh function squashes inputs to the range [-1, 1].
    """

    def activation_function(self, input: np.ndarray):
        # NumPy provides a stable and vectorized tanh implementation.
        return np.tanh(input)

    def derivative(self, input: np.ndarray):
        # d/dx tanh(x) = 1 - tanh(x)^2
        t = np.tanh(input)
        return 1.0 - t ** 2


class SoftmaxActivation(ActivationLayer):
    """Softmax activation.

    Softmax transforms raw scores (logits) into a probability distribution that sums to 1.
    This implementation uses a numerically stable softmax by subtracting the per-row max
    before applying exp.

    Important
    ---------
    The derivative of softmax is not element-wise (it is a Jacobian matrix). Therefore,
    we override `backward_propagation` to compute the Jacobian-vector product efficiently.
    """

    def activation_function(self, input: np.ndarray):
        # Compute softmax along the last axis (commonly class dimension).
        # Stable version: exp(x - max(x)) prevents overflow.
        x = np.asarray(input)

        # Work in 2D for simplicity: (batch, classes).
        squeeze_back = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_back = True

        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        if squeeze_back:
            return probs.reshape(-1)
        return probs

    def derivative(self, input: np.ndarray):
        # The full softmax derivative is a Jacobian; we do not return it here because it is
        # expensive (O(C^2)) and not element-wise. See backward_propagation for the correct
        # Jacobian-vector product used in backprop.
        s = self.activation_function(input)
        return s * (1.0 - s)

    def backward_propagation(self, output_error: float):
        """Backprop through softmax using a Jacobian-vector product.

        Given softmax output `s` and upstream gradient `g` (dE/ds), the gradient w.r.t.
        logits `z` is:
            dE/dz = J^T g
        where J is the softmax Jacobian.

        Efficient form (per sample):
            dE/dz = s * (g - sum(g * s))
        """
        g = np.asarray(output_error)

        # Prefer the cached forward output when available.
        s = getattr(self, 'output', None)
        if s is None:
            s = self.activation_function(self.input)

        # Ensure 2D shapes: (batch, classes)
        squeeze_back = False
        if s.ndim == 1:
            s = s.reshape(1, -1)
            g = g.reshape(1, -1)
            squeeze_back = True

        dot = np.sum(g * s, axis=1, keepdims=True)
        grad = s * (g - dot)

        if squeeze_back:
            return grad.reshape(-1)
        return grad
