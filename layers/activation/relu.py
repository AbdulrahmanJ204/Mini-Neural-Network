from layers.layer import Layer


class Relu(Layer):
    """Rectified Linear Unit (ReLU) activation function.

    Computes: output = max(0, input)
    """

    def __init__(self):
        """Initialize ReLU layer."""
        self.mask = None
        super().__init__()

    def forward(self, x):
        """Forward pass through ReLU.

        Args:
            x: Input data.

        Returns:
            Output with ReLU activation applied.
        """
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """Backward pass for ReLU gradient computation.

        Args:
            dout: Gradient from next layer.

        Returns:
            Gradient with respect to input.
        """
        dout[self.mask] = 0
        dx = dout
        return dx
