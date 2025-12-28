from initializers.initializer import Initializer
import numpy as np


class XavierNormal(Initializer):
    """Xavier (Glorot) Normal weight initialization.

    Good for sigmoid and tanh activations.
    Initializes weights from a normal distribution with std = 1 / sqrt(input_size).
    Helps maintain consistent variance of gradients across layers during backprop.
    """

    def init(self, input_size, output_size):
        """Initialize weights using Xavier initialization.

        Args:
            input_size: Number of input features.
            output_size: Number of output features.

        Returns:
            Weight matrix of shape (input_size, output_size) with Xavier initialization.
        """
        return np.random.randn(input_size, output_size) / np.sqrt(input_size)
