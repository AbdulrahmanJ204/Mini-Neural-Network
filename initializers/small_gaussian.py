from initializers.initializer import Initializer
import numpy as np


class SmallGaussian(Initializer):
    def init(self, input_size, output):
        return np.random.randn(input_size, output) * 0.01
