from initializers.initializer import Initializer
from layers.layer import Layer


import numpy as np


class Affine(Layer):
    """Fully connected (dense) layer performing affine transformation.

    Computes: output = input @ W + b
    """

    def __init__(self, output_size: int, init: Initializer):
        """Initialize Affine layer.

        Args:
            output_size: Number of output neurons.
            init: Initializer object for weight initialization.
        """
        self.params = {}
        self.output_size = output_size
        self.input_size = 0
        self.dw = None
        self.db = None
        self.initializer = init
        super().__init__()

    def init_weights(self, input_size):
        """Initialize weights and biases.

        Args:
            input_size: Number of input features.
        """
        W = self.initializer.init(input_size, self.output_size)
        b = np.zeros(self.output_size)
        self.input_size = input_size
        self.params[f"W{self.id}"] = W
        self.params[f"b{self.id}"] = b
        self.dw = None
        self.db = None

    def forward(self, x):
        """Forward pass through the affine layer.

        Args:
            x: Input of shape (batch_size, input_size).

        Returns:
            Output of shape (batch_size, output_size).
        """
        out = np.dot(x, self.params[f"W{self.id}"]) + self.params[f"b{self.id}"]
        self.x = x
        return out

    def backward(self, dout):
        """Backward pass computing gradients.

        Args:
            dout: Gradient from next layer of shape (batch_size, output_size).

        Returns:
            Gradient with respect to input of shape (batch_size, input_size).
        """
        dx = np.dot(dout, self.params[f"W{self.id}"].T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

    def parameters(self):
        """Get layer parameters.

        Returns:
            Dictionary containing weights and biases.
        """
        return self.params

    def grads(self):
        """Get parameter gradients.

        Returns:
            Dictionary containing weight and bias gradients.
        """
        return {f"W{self.id}": self.dw, f"b{self.id}": self.db}
