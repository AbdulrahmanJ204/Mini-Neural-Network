import numpy as np
from layers.layer import Layer


class Dropout(Layer):
    """Dropout regularization layer.

    Randomly drops activations during training to prevent co-adaptation.
    Helps reduce overfitting by forcing the network to learn redundant representations.
    Disabled during inference.
    """

    def __init__(self, dropout_ratio=0.5):
        """Initialize Dropout layer.

        Args:
            dropout_ratio: Probability of dropping a unit (default: 0.5).
        """
        self.dropout_ratio = dropout_ratio
        self.mask = None
        super().__init__()

    def forward(self, x, train_flg=True):
        """Forward pass through dropout.

        During training: Randomly sets activations to 0 with probability dropout_ratio.
        During inference: Scales activations by (1 - dropout_ratio).

        Args:
            x: Input activations.
            train_flg: Whether in training mode (default: True).

        Returns:
            Activations after dropout applied.
        """
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        """Backward pass for dropout.

        Args:
            dout: Gradient from next layer.

        Returns:
            Gradient with same mask applied.
        """
        return dout * self.mask
