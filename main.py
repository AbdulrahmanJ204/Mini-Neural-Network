from Network.neuralNetwork import NeuralNetwork
from Network.trainer import Trainer
from Network.tuner import HyperParameterTuner
from helpers.plot_helpers import plotResults
from helpers.data_helpers import normailze_mnist_data, fetchData
from layers.activation.Linear import Linear
from layers.activation.Sigmoid import Sigmoid
from layers.initializers.HeNormal import HeNormal
from layers.initializers.SmallGaussian import SmallGaussian
from layers.initializers.XavierNormal import XavierNormal
from layers.Affine import Affine
from layers.loss.SoftMaxWithCrossEntropy import SoftMaxWithCrossEntropy
from layers.optimization.BatchNormalization import BatchNormalization
from layers.optimization.Dropout import Dropout
from optimizers.AdaGrad import AdaGrad
from optimizers.Adam import Adam
from optimizers.Momentum import Momentum
from optimizers.SGD import SGD

# def getNet():
#     return NeuralNetwork(
#         [
#             Affine(100, SmallGaussian()),
#             Sigmoid(),
#             BatchNormalization(),
#             Affine(10, SmallGaussian()),
#         ],
#         SoftMaxWithCrossEntropy(),
#     )
#
#
# net = getNet()
# opt = Adam
# trainer = Trainer(net, opt())
#
x_train, x_test, t_train, t_test = fetchData()
# epochs = 2
# loss, accuracy = trainer.fit(x_train, x_test, t_train, t_test , epochs=epochs)
#
# net = getNet()
# trainer = Trainer(net, opt())
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
