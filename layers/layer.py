from abc import ABC, abstractmethod


class Layer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    def hasGrads(self):
        return False


import numpy as np


class Dense(Layer):

    def __init__(self, input_size: int, output_size: int, activation: Layer):

        self.activation = activation
        w = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        b = np.zeros(output_size)
        self.affine = Affine(w, b)

    def forward(self, x):
        x = self.affine.forward(x)
        x = self.activation.forward(x)
        return x

    def hasGrads(self):
        return True

    def backward(self, dout):
        dout = self.activation.backward(dout)
        dout = self.affine.backward(dout)
        return dout

    def grads(self):
        return self.affine.grads()

    def params(self):
        return self.affine.params


class Affine(Layer):
    counter = 0

    def __init__(self, W, b):

        Affine.counter += 1
        self.cnt = Affine.counter
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

    def grads(self):
        return {f"W{self.cnt}": self.dw, f"b{self.cnt}": self.db}
