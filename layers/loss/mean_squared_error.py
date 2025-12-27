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

        self.loss = np.sum((y - t) ** 2) / len(self.t)
        return self.loss

    def backward(self, dout=1):
        dx = dout * 2 * (self.y - self.t) / len(self.t)
        return dx


