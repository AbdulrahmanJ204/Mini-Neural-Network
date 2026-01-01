import numpy as np

from layers.loss.loss import Loss


class MeanSquaredError(Loss):
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        super().__init__()

    def forward(self, y, t):
        self.y = y
        self.t = t

        self.loss = np.mean(np.mean((y - t) ** 2, axis=1))
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        n_features = self.y.shape[1] if self.y.ndim > 1 else 1
        dx = dout * 2 * (self.y - self.t) / (n_features * batch_size)
        return dx
