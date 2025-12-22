from Network.neuralNetwork import NeuralNetwork
from optimizers.Optimizer import Optimizer
import numpy as np


class Trainer:
    def __init__(self, network: NeuralNetwork, optimizer: Optimizer):
        self.network = network
        self.optimizer = optimizer

    
    def fit(self, x_train, x_test, t_train, t_test, batch_size=100, epochs=10):

        if not self.network.initalized:
            inputSize = x_train.shape[1]
            print(inputSize)
            self.network.init_weights(inputSize)

        trainingSize = len(x_train)
        iterPerEpoch = trainingSize // batch_size

        loss_hist = []
        accuracy_hist = []

        for i in range(epochs):
            for _ in range(iterPerEpoch):
                batch_mask = np.random.choice(trainingSize, batch_size)
                x_batch = x_train[batch_mask]
                t_batch = t_train[batch_mask]

                loss = self.train_step(x_batch, t_batch)
                loss_hist.append(loss)

            acc = self.network.accuracy(x_test, t_test)
            accuracy_hist.append(acc)
            print(f"Epoch {i+1}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        return loss_hist, accuracy_hist

    def train_step(self, x_batch, t_batch):
        grads = self.network.gradient(x_batch, t_batch)

        self.optimizer.update(self.network, grads)

        loss = self.network.loss(x_batch, t_batch)
        return loss
