from layers.layer import Layer
import numpy as np


class Sigmoid(Layer):
    def __init__(self):
        self.out = None
        super().__init__()

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
