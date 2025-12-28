from layers.loss.loss import Loss
import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    eps = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + eps)) / batch_size


class SoftMaxWithCrossEntropy(Loss):
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        super().__init__()

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        dx = dout * (self.y - self.t) / len(self.t)
        return dx
