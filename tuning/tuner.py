from abc import ABC, abstractmethod
from typing import Type

from layers import Affine, Loss


class Tuner(ABC):
    @abstractmethod
    def get_best_params(
            self, x_train, x_test, t_train, t_test, output_layer: Affine, loss_layer_cls: Type[Loss], params

    ):
        pass
