from layers.loss.loss import Loss
import numpy as np


class BCE(Loss):
    """Binary Cross-Entropy"""

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        super().__init__()

    def forward(self, x, t):

        self.t = t
        self.y = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

        self.loss = np.mean(np.maximum(0, x) - x * t + np.log(1 + np.exp(-np.abs(x))))

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size
        return dx
