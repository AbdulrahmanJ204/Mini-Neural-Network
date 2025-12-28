from abc import ABC, abstractmethod


class Initializer(ABC):
    """Base class for weight initialization methods.

    Different initialization strategies can significantly impact training convergence
    and the quality of learned representations.
    """

    @abstractmethod
    def init(self, input_size, output_size):
        """Initialize weights for a layer.

        Args:
            input_size: Number of input features.
            output_size: Number of output features.

        Returns:
            Weight matrix of shape (input_size, output_size).
        """
        pass
