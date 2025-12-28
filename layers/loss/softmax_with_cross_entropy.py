from layers.loss.loss import Loss
import numpy as np


def softmax(x):
    """Compute softmax function for numerical stability.

    Subtracts max to prevent overflow:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        x: Input array of shape (batch_size, num_classes).

    Returns:
        Softmax probabilities of same shape as input.
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    """Compute cross-entropy loss.

    Args:
        y: Predicted probabilities of shape (batch_size, num_classes).
        t: Target labels (either one-hot or class indices).

    Returns:
        Cross-entropy loss value (scalar).
    """
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    eps = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + eps)) / batch_size


class SoftMaxWithCrossEntropy(Loss):
    """Combined Softmax and Cross-Entropy loss layer.

    Applies softmax to logits then computes cross-entropy loss.
    Commonly used for multi-class classification tasks.
    """

    def __init__(self):
        """Initialize Softmax+CrossEntropy loss layer."""
        self.loss = None
        self.y = None
        self.t = None
        super().__init__()

    def forward(self, x, t):
        """Compute softmax and cross-entropy loss.

        Args:
            x: Logits (pre-softmax predictions) of shape (batch_size, num_classes).
            t: Target labels (one-hot or class indices) of shape (batch_size, num_classes) or (batch_size,).

        Returns:
            Cross-entropy loss value (scalar).
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        """Compute gradient of cross-entropy loss with softmax.

        Args:
            dout: Gradient from previous layer (default: 1).

        Returns:
            Gradient with respect to logits.
        """
        dx = dout * (self.y - self.t) / len(self.t)
        return dx
