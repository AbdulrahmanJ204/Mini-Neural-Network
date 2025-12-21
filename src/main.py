from Network.neuralNetwork import NeuralNetwork
from Network.trainer import Trainer
from helpers.dataAndPlotHelpers import fetchData, normalizeData, plotResults
from layers.activation.Sigmoid import Sigmoid
from layers.layer import Dense
from layers.loss.SoftMaxWithCrossEntropy import SoftMaxWithCrossEntropy
from layers.optimization import BatchNormalization
from layers.optimization.Dropout import Dropout
from optimizers.AdaGrad import AdaGrad
from optimizers.Adam import Adam
from optimizers.Momentum import Momentum
from optimizers.SGD import SGD


def getNet():
    return NeuralNetwork(
        [
            Dense(28 * 28, 100, Sigmoid()),
            
            # Dropout(0.3),
            Dense(100, 10, Sigmoid()),
        ],
        SoftMaxWithCrossEntropy(),
    )


net = getNet()
opt = Momentum
trainer = Trainer(net, opt())

x_train, x_test, t_train, t_test = fetchData()

loss, accuracy = trainer.train(x_train, x_test, t_train, t_test)

net = getNet()
trainer = Trainer(net, opt())

x_train, x_test = normalizeData(x_train, x_test)
loss1, accuracy1 = trainer.train(x_train, x_test, t_train, t_test)
plotResults(loss, accuracy, loss1, accuracy1)
