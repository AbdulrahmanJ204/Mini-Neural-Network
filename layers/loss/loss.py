from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def forward(self, y, t):
        """Compute loss"""
        pass

    @abstractmethod
    def backward(self, dout=1):
        """Compute gradient"""
        pass