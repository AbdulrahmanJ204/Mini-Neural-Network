from network.neural_network import NeuralNetwork
from network.trainer import Trainer
from network.tuner import HyperParameterTuner
from helpers.plot_helpers import plotResults
from helpers.data_helpers import normailze_mnist_data, fetchData
from layers.activation.linear import Linear
from layers.activation.sigmoid import Sigmoid
from layers.initializers.he_normal import HeNormal
from layers.initializers.small_gaussian import SmallGaussian
from layers.initializers.xavier_normal import XavierNormal
from layers.affine import Affine
from layers.loss.softmax_with_cross_entropy import SoftMaxWithCrossEntropy
from layers.optimization.batch_normalization import BatchNormalization
from layers.optimization.dropout import Dropout
from optimizers.adagrad import AdaGrad
from optimizers.adam import Adam
from optimizers.momentum import Momentum
from optimizers.sgd import SGD


def get_net():
    return NeuralNetwork(
        [
            Affine(100, SmallGaussian()),
            Sigmoid(),
            BatchNormalization(),
            Affine(10, SmallGaussian()),
        ],
        SoftMaxWithCrossEntropy(),
    )


#
# net = getNet()
opt = Adam
# trainer = Trainer(net, opt())
#
x_train, x_test, t_train, t_test = fetchData()
# epochs = 2
# loss, accuracy = trainer.fit(x_train, x_test, t_train, t_test , epochs=epochs)
#
net = get_net()
trainer = Trainer(net, opt())
#
x_train, x_test = normailze_mnist_data(x_train, x_test)
# loss1, accuracy1 = trainer.fit(x_train, x_test, t_train, t_test , epochs= epochs)
# plotResults(loss, accuracy, loss1, accuracy1 , f"Adam")
h = HyperParameterTuner()

print(
    h.tune(
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
