import sys
sys.path.append("..")

from utils import fetch_mnist_data

from tuning import GridTuner

x_train, x_test, t_train, t_test = fetch_mnist_data()
from layers import Affine, SoftMaxWithCrossEntropy, Sigmoid
from initializers import SmallGaussian, XavierNormal
from optimizers import Adam

tuner = GridTuner()
# used small search space to reduce execution time.
search_space = {
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
    }
best_params = tuner.optimize(
    x_train=x_train,
    x_test=x_test,
    t_train=t_train,
    t_test=t_test,
    output_layer=Affine(10, SmallGaussian()),
    loss_layer_cls=SoftMaxWithCrossEntropy,
    params=search_space,
)

trainer = tuner.best_trainer

tuner.print_best_params()

print(trainer.network.structure())
print(trainer.evaluate(x_test, t_test))
