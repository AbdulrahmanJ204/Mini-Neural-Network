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

        self.loss = np.mean(np.mean((y - t) ** 2, axis=1))
        return self.loss

    def backward(self, dout=1):
        """Compute gradient of MSE loss.

        Args:
            dout: Gradient from previous layer (default: 1).

        Returns:
            Gradient with respect to predictions.
        """
        batch_size = self.y.shape[0]
        n_features = self.y.shape[1] if self.y.ndim > 1 else 1
        dx = dout * 2 * (self.y - self.t) / (n_features * batch_size)
        return dx
