import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import random
from tuning import RandomTuner
from utils import fetch_mnist_data
from layers import Affine, SoftMaxWithCrossEntropy, Relu, Tanh, Sigmoid
from initializers import SmallGaussian, XavierNormal, HeNormal
from optimizers import Adam, Momentum, AdaGrad

np.random.seed(42)
random.seed(42)

x_train, x_test, t_train, t_test = fetch_mnist_data()
tuner = RandomTuner()
search_space = {
    "hidden_number": [2, 3, 5],
    "layer_props": {
        "layer_neurons_number": [20, 50, 100],
        "init_method": [XavierNormal, HeNormal],
        "dropout_rate": [0.0, 0.1, 0.3],
        "activation": [Sigmoid, Tanh, Relu],
        "batch_normalization": [False, True],
    },
    "optimizer": [Adam, Momentum, AdaGrad],
    "learning_rate": [0.001, 0.01, 0.1],
    "beta1": [0.9, 0.2],
    "beta2": [0.999, 0.9],
    "momentum": [0.9, 0, 7],
    "batch_size": [128, 64, 256],
    "epochs": [2, 3, 4, 1],
}
best_params = tuner.optimize(
    x_train=x_train,
    x_val=x_test,
    t_train=t_train,
    t_val=t_test,
    output_layer=Affine(10, SmallGaussian()),
    loss_layer_cls=SoftMaxWithCrossEntropy,
    params=search_space,
    n_samples=2,
)
trainer = tuner.best_trainer

tuner.print_best_params()

print(trainer.network.structure())
print(trainer.evaluate(x_test, t_test))
