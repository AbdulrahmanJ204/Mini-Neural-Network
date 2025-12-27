from layers.initializers.initializer import Initializer
import numpy as np


class HeNormal(Initializer):
    def init(self, input_size, output):
        return np.random.randn(input_size, output) * np.sqrt(2 / input_size)
