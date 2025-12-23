from layers.layer import Layer


import numpy as np


class Tanh(Layer):
    def __init__(self):
        self.out = None
        super().__init__()

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1 - self.out**2)
        return dx
