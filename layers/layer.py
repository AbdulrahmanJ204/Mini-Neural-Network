from abc import ABC, abstractmethod

from layers.initializers.XavierNormal import XavierNormal
from layers.initializers.initializer import Initializer


class Layer(ABC):
    counter = 0

    def __init__(self):
        self.cnt = Layer.counter
        Layer.counter += 1

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    def parameters(self):
        return {}

    def grads(self):
        return {}


import numpy as np


class Affine(Layer):

    def __init__(self, output_size: int, init: Initializer):
        self.output_size = output_size
        self.initializer = init
        super().__init__()

    def init_weights(self, input_size):
        W = self.initializer.init(input_size, self.output_size)
        b = np.zeros(self.output_size)

        self.params = {}
        self.params[f"W{self.cnt}"] = W
        self.params[f"b{self.cnt}"] = b
        self.dw = None
        self.db = None

    def forward(self, x):
        out = np.dot(x, self.params[f"W{self.cnt}"]) + self.params[f"b{self.cnt}"]
        self.x = x
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.params[f"W{self.cnt}"].T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

    def parameters(self):
        return self.params

    def grads(self):
        return {f"W{self.cnt}": self.dw, f"b{self.cnt}": self.db}
