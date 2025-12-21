# recieves a network and an optimizer
# in the training funciton , we do every thing as lab5 except we call optimizer.update(network , grads)


from Network.neuralNetwork import NeuralNetwork
from optimizers.Optimizer import Optimizer
import numpy as np


class Trainer:
    def __init__(self, network: NeuralNetwork, optimizer: Optimizer):
        self.network = network
        self.optimizer = optimizer
    # TODO: Complete this as required in pdf
    def train(self, x_train, x_test, t_train, t_test):
        batchSize = 100
        lr = 0.1
        iterations = 500
        trainingSize = len(x_train)
        network = self.network
        lossList = []
        accuracyList = []
        iterPerEpoch = trainingSize // batchSize
        for i in range(iterations):
            batch_mask = np.random.choice(trainingSize, batchSize)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            grads = network.gradient(x_batch, t_batch)

            self.optimizer.update(network, grads)

            loss = network.loss(x_batch, t_batch)
            # print(loss)
            lossList.append(loss)
            if (i + 1) % iterPerEpoch == 0:
                print(i + 1)
                accuracyList.append(network.accuracy(x_test, t_test))
        return lossList, accuracyList
