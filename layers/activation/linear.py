from layers.layer import Layer


# Identity activation layer - passes input through unchanged
class Linear(Layer):
    """Linear (Identity) activation function.

    This layer is a pass-through that outputs the input unchanged.
    Useful for regression tasks or when no activation is desired.
    """

    def forward(self, x):
        """Forward pass through linear (identity) layer.

        Args:
            x: Input data.

        Returns:
            Same as input (identity function).
        """
        return x

    def backward(self, dout):
        """Backward pass for linear layer.

        Args:
            dout: Gradient from next layer.

        Returns:
            Same gradient (identity gradient).
        """
        return dout
