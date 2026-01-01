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
        
        if x.ndim == 1:
            x = x.reshape(1, x.size)
            t = t.reshape(1, t.size)

        self.t = t

        x = x - np.max(x, axis=-1, keepdims=True) # overflow prevent

        if t.shape == x.shape:  
            t = t.argmax(axis=1)

        sum = np.sum(np.exp(x), axis=-1, keepdims=True)
        log_sum = np.log(sum)
        self.y = np.exp(x - log_sum)

        batch_size = x.shape[0]
        self.loss = -np.sum(x[np.arange(batch_size), t] - log_sum) / batch_size

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size
        return dx
