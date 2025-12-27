import sys
sys.path.append("..")

from utils import fetchData

from tuning import GridTuner

x_train, x_test, t_train, t_test = fetchData()
from layers import Affine, SoftMaxWithCrossEntropy, Relu, Tanh, Sigmoid
from layers.initializers import SmallGaussian, XavierNormal
from optimizers import AdaGrad, Adam, Momentum, SGD

h = GridTuner()

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
            "activation": [Sigmoid],
            "batch_normalization": [True],
        },
        "optimizer": [Adam],
        "learning_rate": [0.001, 0.01],
        "beta1": [0.9],
        "beta2": [0.999],
        "momentum": [0.9],
        "batch_size": [128],
        "epochs": [3],
    },
)
print(best_params["trainer"].network.structure())
