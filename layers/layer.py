from abc import ABC, abstractmethod

from layers.initializers.xavier_normal import XavierNormal


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
    @classmethod
    def reset_counter(cls):
        cls.counter = 0    
    def parameters(self):
        return {}

    def grads(self):
        return {}
