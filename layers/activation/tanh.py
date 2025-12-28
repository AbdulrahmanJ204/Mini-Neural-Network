from layers.layer import Layer
import numpy as np


class Tanh(Layer):
    """Hyperbolic tangent (tanh) activation function.

    Computes: output = tanh(input)
    Maps input to range (-1, 1).
    """

    def __init__(self):
        """Initialize Tanh layer."""
        self.out = None
        super().__init__()

    def forward(self, x):
        """Forward pass through tanh.

        Args:
            x: Input data.

        Returns:
            Output with tanh activation applied, values in (-1, 1).
        """
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        """Backward pass for tanh gradient computation.

        Args:
            dout: Gradient from next layer.

        Returns:
            Gradient with respect to input.
        """
        dx = dout * (1 - self.out**2)
        return dx
