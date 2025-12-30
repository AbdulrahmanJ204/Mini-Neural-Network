from abc import ABC, abstractmethod


class Layer(ABC):
    counter = 0

    def __init__(self):
        self.id = Layer.counter
        Layer.counter += 1

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @classmethod
    def reset_counter(cls):
        cls.counter = 0

    def parameters(self):
        return {}

    def grads(self):
        return {}
