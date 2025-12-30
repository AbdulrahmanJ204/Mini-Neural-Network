from abc import ABC, abstractmethod
from typing import Type

from layers import Affine, Loss


class Tuner(ABC):
    def __init__(self):
        self.best_params = None
        self.best_trainer = None
    @abstractmethod
    def optimize(
            self, x_train, x_test, t_train, t_test, output_layer: Affine, loss_layer_cls: Type[Loss], params

    ):
        pass

    def print_best_params(self):
        import json
        print(json.dumps(self.best_params, indent=4, default=str))
