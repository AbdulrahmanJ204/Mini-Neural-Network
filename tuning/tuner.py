from abc import ABC, abstractmethod
from typing import Type

from layers import Affine, Loss


class Tuner(ABC):
    """Base class for hyperparameter tuning algorithms.

    Provides interface for exploring different network architectures and
    hyperparameters to find the best configuration.
    """

    def __init__(self):
        """Initialize tuner.

        Attributes:
            best_params: Dictionary of best hyperparameters found.
            best_trainer: Trainer object with best configuration.
        """
        self.best_params = None
        self.best_trainer = None


    @abstractmethod
    def optimize(
        self,
        x_train,
        x_val,
        t_train,
        t_val,
        output_layer: Affine,
        loss_layer_cls: Type[Loss],
        params,
    ):
        """Optimize hyperparameters for the network.

        Args:
            x_train: Training input data.
            x_val: Validation input data.
            t_train: Training target labels.
            t_val: Validation target labels.
            output_layer: Output layer (typically Affine layer for classification).
            loss_layer_cls: Loss function class.
            params: Dictionary of hyperparameter search space.

        Returns:
            Dictionary of best found parameters.
        """
        pass

    def print_best_params(self):
        """Print best found hyperparameters in formatted JSON style."""
        import json

        print(json.dumps(self.best_params, indent=4, default=str))
