from optimizers.optimizer import Optimizer
import numpy as np


class Momentum(Optimizer):
    """Momentum optimizer.

    Uses velocity (momentum) term to accelerate convergence in consistent directions
    and dampen oscillations in other directions.
    """

    def __init__(self, lr=0.01, momentum=0.9):
        """Initialize Momentum optimizer.

        Args:
            lr: Learning rate (default: 0.01).
            momentum: Momentum coefficient (default: 0.9).
        """
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def update_params(self, params, grads):
        """Update parameters using momentum rule.

        Updates velocity as: v = momentum * v - lr * grad
        Updates parameters as: param += v

        Args:
            params: Dictionary of parameters to update.
            grads: Dictionary of gradients.
        """
        for key, val in params.items():
            if key not in self.v:
                self.v[key] = np.zeros_like(val)
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
