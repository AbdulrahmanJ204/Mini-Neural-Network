from abc import ABC, abstractmethod


class Tuner(ABC):
    @abstractmethod
    def get_best_params(
        self, x_train, x_test, t_train, t_test, output_layer, loss_layer, params
    ):
        pass
