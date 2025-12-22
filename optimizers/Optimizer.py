from abc import ABC, abstractmethod


class Optimizer(ABC):
    def update(self, network, grads):

        for layer, g in zip(network.layers, grads):
            params = layer.parameters()
            self.updateParams(params, g)

    @abstractmethod
    def updateParams(self, params, grads):
        pass
