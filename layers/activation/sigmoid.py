from layers.layer import Layer
import numpy as np


class Sigmoid(Layer):
    """Sigmoid activation function.

    Computes: output = 1 / (1 + exp(-input))
    Maps input to range (0, 1).
    """

    def __init__(self):
        """Initialize Sigmoid layer."""
        self.out = None
        super().__init__()

    def forward(self, x):
        """Forward pass through sigmoid.

        Args:
            x: Input data.

        Returns:
            Output with sigmoid activation applied, values in (0, 1).
        """
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        """Backward pass for sigmoid gradient computation.

        Args:
            dout: Gradient from next layer.

        Returns:
            Gradient with respect to input.
        """
        dx = dout * (1.0 - self.out) * self.out
        return dx
