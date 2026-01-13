from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient


class Adam(Optimizer):
    """Adam optimizer.

    Adam combines momentum (1st moment estimate) and RMSProp-like adaptive learning rates
    (2nd moment estimate).

    Parameters
    ----------
    learning_rate: float
        Step size.
    beta_1: float
        Exponential decay rate for the 1st moment estimates.
    beta_2: float
        Exponential decay rate for the 2nd moment estimates.
    epsilon: float
        Small constant for numerical stability.

    Estimated parameters
    --------------------
    m: numpy.ndarray
        Moving average of the gradients.
    v: numpy.ndarray
        Moving average of the squared gradients.
    t: int
        Time step (epoch/update counter), starts at 0.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # State variables (initialized lazily to match the shape of w).
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """Update parameters using Adam.

        Algorithm (per update):
        1) Initialize m and v with zeros (same shape as w) if needed
        2) t += 1
        3) m = beta_1 * m + (1 - beta_1) * grad
        4) v = beta_2 * v + (1 - beta_2) * (grad^2)
        5) Bias correction:
           m_hat = m / (1 - beta_1^t)
           v_hat = v / (1 - beta_2^t)
        6) w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        """
        grad = np.asarray(grad_loss_w)

        # 1) Lazy initialization of moment vectors.
        if self.m is None or self.v is None:
            self.m = np.zeros_like(w, dtype=float)
            self.v = np.zeros_like(w, dtype=float)

        # 2) Update time step.
        self.t += 1

        # 3) Update biased first moment estimate.
        self.m = self.beta_1 * self.m + (1.0 - self.beta_1) * grad

        # 4) Update biased second raw moment estimate.
        self.v = self.beta_2 * self.v + (1.0 - self.beta_2) * (grad ** 2)

        # 5) Compute bias-corrected moments.
        m_hat = self.m / (1.0 - (self.beta_1 ** self.t))
        v_hat = self.v / (1.0 - (self.beta_2 ** self.t))

        # 6) Update parameters.
        return w - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))