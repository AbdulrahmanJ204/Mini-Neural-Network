from abc import ABC, abstractmethod


class Optimizer(ABC):
    def update(self, network, grads):
        i = 0
        for layer in network.layers:
            if not layer.hasGrads():
                continue
            params = layer.params()
            self.updateParams(params, grads[i])
            i += 1

    @abstractmethod
    def updateParams(self, params, grads):
        pass
