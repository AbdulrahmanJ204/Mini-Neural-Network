from helpers.math import cross_entropy_error, softmax


import numpy as np

from layers.layer import Layer


class SoftMaxWithCrossEntropy(Layer):
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
