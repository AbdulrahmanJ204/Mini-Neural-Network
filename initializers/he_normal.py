from initializers.initializer import Initializer
import numpy as np


class HeNormal(Initializer):
    """He Normal weight initialization.

    Recommended for layers with ReLU activation.
    Initializes weights from a normal distribution with std = sqrt(2 / input_size).
    This helps maintain consistent variance of activations across layers.
    """

    def init(self, input_size, output_size):
        """Initialize weights using He initialization.

        Args:
            input_size: Number of input features.
            output_size: Number of output features.

        Returns:
            Weight matrix of shape (input_size, output_size) with He initialization.
        """
        return np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
