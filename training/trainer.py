import random
from models.neural_network import NeuralNetwork
from optimizers.optimizer import Optimizer
import numpy as np



class Trainer:
    def __init__(self, network: NeuralNetwork, optimizer: Optimizer):
        self.network = network
        self.optimizer = optimizer

    def fit(self, x_train, x_test, t_train, t_test, batch_size=100, epochs=10):

        if not self.network.initialized:
            input_size = x_train.shape[1]
            self.network.init_weights(input_size)

        loss_hist = []
        accuracy_hist = []
        loss = 0

        def batch_generator(x, t, batch_size):
            n = len(x)
            indices = np.arange(n)
            np.random.shuffle(indices)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                yield x[batch_idx], t[batch_idx]

        for i in range(epochs):
            for x_batch, t_batch in batch_generator(x_train, t_train, batch_size):
                loss = self.train_step(x_batch, t_batch)
                loss_hist.append(loss)

            acc = self.network.accuracy(x_test, t_test)
            accuracy_hist.append(acc)

            print(f" Epoch {i + 1}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        return loss_hist, accuracy_hist

    def train_step(self, x_batch, t_batch):
        grads = self.network.gradient(x_batch, t_batch)

        self.optimizer.update(self.network, grads)

        loss = self.network.loss(x_batch, t_batch)
        return loss

    def evaluate(self, x_test, t_test):
        return self.network.accuracy(x_test, t_test)
