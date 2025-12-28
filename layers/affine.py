from initializers.initializer import Initializer
from layers.layer import Layer


import numpy as np


class Affine(Layer):

    def __init__(self, output_size: int, init: Initializer):
        self.params = {}
        self.output_size = output_size
        self.input_size = 0
        self.dw = None
        self.db = None
        self.initializer = init
        super().__init__()

    def init_weights(self, input_size):
        W = self.initializer.init(input_size, self.output_size)
        b = np.zeros(self.output_size)
        self.input_size = input_size
        self.params[f"W{self.id}"] = W
        self.params[f"b{self.id}"] = b
        self.dw = None
        self.db = None

    def forward(self, x):
        out = np.dot(x, self.params[f"W{self.id}"]) + self.params[f"b{self.id}"]
        self.x = x
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.params[f"W{self.id}"].T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

    def parameters(self):
        return self.params

    def grads(self):
        return {f"W{self.id}": self.dw, f"b{self.id}": self.db}
