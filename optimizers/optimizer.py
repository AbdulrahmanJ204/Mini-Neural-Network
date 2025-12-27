from abc import ABC, abstractmethod


class Optimizer(ABC):
    def update(self, network, grads):

        for layer, g in zip(network.layers, grads):
            params = layer.parameters()
            self.update_params(params, g)

    @abstractmethod
    def update_params(self, params, grads):
        pass
