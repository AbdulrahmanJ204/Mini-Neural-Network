from Network.neuralNetwork import NeuralNetwork
from Network.trainer import Trainer
from helpers.dataAndPlotHelpers import fetchData, normalizeData, plotResults
from layers.activation.Linear import Linear
from layers.activation.Sigmoid import Sigmoid
from layers.initializers.SmallGaussian import SmallGaussian
from layers.initializers.XavierNormal import XavierNormal
from layers.layer import Affine
from layers.loss.SoftMaxWithCrossEntropy import SoftMaxWithCrossEntropy
from layers.optimization.BatchNormalization import BatchNormalization
from layers.optimization.Dropout import Dropout
from optimizers.AdaGrad import AdaGrad
from optimizers.Adam import Adam
from optimizers.Momentum import Momentum
from optimizers.SGD import SGD


def getNet():
    return NeuralNetwork(
        [
            Affine(100, SmallGaussian()),
            Sigmoid(),
            BatchNormalization(),
            Affine(10, SmallGaussian()),
        ],
        SoftMaxWithCrossEntropy(),
    )


net = getNet()
opt = Adam
trainer = Trainer(net, opt())

x_train, x_test, t_train, t_test = fetchData()

loss, accuracy = trainer.fit(x_train, x_test, t_train, t_test)

net = getNet()
trainer = Trainer(net, opt())

x_train, x_test = normalizeData(x_train, x_test)
loss1, accuracy1 = trainer.fit(x_train, x_test, t_train, t_test)
plotResults(loss, accuracy, loss1, accuracy1)
