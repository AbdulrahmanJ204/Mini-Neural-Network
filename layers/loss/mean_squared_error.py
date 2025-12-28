import numpy as np

from layers.loss.loss import Loss


class MeanSquaredError(Loss):
    """Mean Squared Error (MSE) loss function.

    Computes: MSE = mean((y - t)^2)
    Commonly used for regression tasks.
    """

    def __init__(self):
        """Initialize MSE loss layer."""
        self.loss = None
        self.y = None
        self.t = None
        super().__init__()

    def forward(self, y, t):
        """Compute mean squared error loss.

        Args:
            y: Predicted values of shape (batch_size, output_size).
            t: Target values of shape (batch_size, output_size).

        Returns:
            MSE loss value (scalar).
        """
        self.y = y
        self.t = t

        self.loss = np.sum((y - t) ** 2) / len(self.t)
        return self.loss

    def backward(self, dout=1):
        """Compute gradient of MSE loss.

        Args:
            dout: Gradient from previous layer (default: 1).

        Returns:
            Gradient with respect to predictions.
        """
        dx = dout * 2 * (self.y - self.t) / len(self.t)
        return dx
