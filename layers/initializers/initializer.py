from abc import ABC, abstractmethod


class Initializer(ABC):
    @abstractmethod
    def init(self, input, output):
        pass
