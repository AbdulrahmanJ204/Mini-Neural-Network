from abc import ABC, abstractmethod


class Loss(ABC):
    """Base class for loss functions.

    Loss functions measure the difference between predicted and target values.
    They provide both forward pass (loss computation) and backward pass (gradient computation).
    """

    @abstractmethod
    def forward(self, y, t):
        """Compute loss value.

        Args:
            y: Predicted output.
            t: Target output.

        Returns:
            Loss value (scalar).
        """
        pass

    @abstractmethod
    def backward(self, dout=1):
        """Compute loss gradient with respect to predictions.

        Args:
            dout: Gradient from previous layer (default: 1).

        Returns:
            Gradient with respect to predictions.
        """
        pass
