from layers import Affine, Relu, Sigmoid, BatchNormalization, SoftMaxWithCrossEntropy, Tanh
from layers.initializers import SmallGaussian
from utils import fetchData, normailze_mnist_data, plotResults
from models import NeuralNetwork
from training import Trainer
from optimizers import Adam, Momentum
from tuning import RandomTuner


def get_net():
    return NeuralNetwork(
        [
            Affine(100, SmallGaussian()),
            Tanh(),
            BatchNormalization(),
            Affine(10, SmallGaussian()),
        ],
        SoftMaxWithCrossEntropy(),
    )


#
net = get_net()
opt = Adam
trainer = Trainer(net, opt())

x_train, x_test, t_train, t_test = fetchData()
epochs = 2
loss, accuracy = trainer.fit(x_train, x_test, t_train, t_test, epochs=epochs)
#
net = get_net()
trainer = Trainer(net, opt())
#
x_train, x_test = normailze_mnist_data(x_train, x_test)
loss1, accuracy1 = trainer.fit(x_train, x_test, t_train, t_test, epochs=epochs)
plotResults(loss, accuracy, loss1, accuracy1, f"Adam")
h = RandomTuner()

print(
    h.get_best_params(
        x_train=x_train,
        x_test=x_test,
        t_train=t_train,
        t_test=t_test,
        output_layer=Affine(10, SmallGaussian()),
        loss_layer=SoftMaxWithCrossEntropy(),
        params={
            "hidden_number": [2, 3],
            "layer_props": {
                "layer_neurons_number": [20, 50],
                "init_method": [XavierNormal],
                "dropout_rate": [0.0, 0.1],
                "activation": [Sigmoid],
                "batch_normalization": [False],
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
)
