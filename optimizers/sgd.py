from optimizers.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Implements simple SGD with constant learning rate.
    """

    def __init__(self, lr=0.01):
        """Initialize SGD optimizer.

        Args:
            lr: Learning rate (default: 0.01).
        """
        self.lr = lr

    def update_params(self, params, grads):
        """Update parameters using SGD rule: param -= lr * grad.

        Args:
            params: Dictionary of parameters to update.
            grads: Dictionary of gradients.
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]
