from abc import ABC, abstractmethod


class Layer(ABC):
    """Base class for all neural network layers.

    This abstract class defines the interface that all layer implementations must follow.
    Layers can be activation functions, dense layers, normalization layers, etc.
    """

    counter = 0

    def __init__(self):
        """Initialize layer with a unique ID."""
        self.id = Layer.counter
        Layer.counter += 1

    @abstractmethod
    def forward(self, x):
        """Forward pass through the layer.

        Args:
            x: Input data.

        Returns:
            Output after forward pass.
        """
        pass

    @abstractmethod
    def backward(self, dout):
        """Backward pass through the layer (gradient computation).

        Args:
            dout: Gradient from the next layer.

        Returns:
            Gradient with respect to the input.
        """
        pass

    @classmethod
    def reset_counter(cls):
        """Reset the layer counter to 0."""
        cls.counter = 0

    def parameters(self):
        """Get layer parameters.

        Returns:
            Dictionary of layer parameters (empty dict if no parameters).
        """
        return {}

    def grads(self):
        """Get gradients of layer parameters.

        Returns:
            Dictionary of parameter gradients (empty dict if no parameters).
        """
        return {}
