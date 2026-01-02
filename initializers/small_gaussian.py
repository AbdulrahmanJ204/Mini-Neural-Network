from initializers.initializer import Initializer
import numpy as np


class SmallGaussian(Initializer):
    """Small Gaussian weight initialization.

    Simple initialization that samples from a normal distribution scaled by 0.01.
    Suitable for networks with tanh or sigmoid activations.
    """

    def init(self, input_size, output_size):
        """Initialize weights from a small Gaussian distribution.

        Args:
            input_size: Number of input features.
            output_size: Number of output features.

        Returns:
            Weight matrix of shape (input_size, output_size) from N(0, 0.01^2).
        """
        return np.random.randn(input_size, output_size) * 0.01
