from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base class for optimization algorithms.

    This abstract class defines the interface for all optimizer implementations
    used to update network parameters during training.
    """

    def update(self, network, grads):
        """Update network parameters using computed gradients.

        Args:
            network: The neural network to update.
            grads: List of gradient dictionaries for each layer.
        """
        for layer, g in zip(network.layers, grads):
            params = layer.parameters()
            self.update_params(params, g)

    @abstractmethod
    def update_params(self, params, grads):
        """Update parameters based on gradients.

        Args:
            params: Dictionary of layer parameters to update.
            grads: Dictionary of gradients corresponding to parameters.
        """
        pass
