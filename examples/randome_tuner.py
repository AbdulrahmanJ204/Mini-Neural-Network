import sys

import numpy as np

sys.path.append("..")
import random
from tuning import RandomTuner
from utils import fetchData
from layers import Affine, SoftMaxWithCrossEntropy, Relu, Tanh, Sigmoid
from layers.initializers import SmallGaussian, XavierNormal
from optimizers import AdaGrad, Adam, Momentum, SGD

np.random.seed(42)
random.seed(42)


x_train, x_test, t_train, t_test = fetchData()
h = RandomTuner()

best_params = h.get_best_params(
    x_train=x_train,
    x_test=x_test,
    t_train=t_train,
    t_test=t_test,
    output_layer=Affine(10, SmallGaussian()),
    loss_layer_cls=SoftMaxWithCrossEntropy,
    params={
        "hidden_number": [2, 3],
        "layer_props": {
            "layer_neurons_number": [20, 50],
            "init_method": [XavierNormal],
            "dropout_rate": [0.0, 0.1],
            "activation": [Sigmoid, Tanh, Relu],
            "batch_normalization": [False, True],
        },
        "optimizer": [Adam, Momentum],
        "learning_rate": [0.001, 0.01],
        "beta1": [0.9],
        "beta2": [0.999],
        "momentum": [0.9],
        "batch_size": [128],
        "epochs": [2, 3],
    },
)
trainer = best_params["trainer"]
print(trainer.network.structure())
print(trainer.evaluate(x_test, t_test))