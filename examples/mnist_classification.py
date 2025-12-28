"""
MNIST Classification Example
============================
Train a neural network on MNIST dataset.
"""

import random
import sys

import numpy as np

sys.path.append("..")


from layers import Affine, Relu
from initializers import HeNormal
from layers.loss import SoftMaxWithCrossEntropy
from models import NeuralNetwork
from training import Trainer
from optimizers import Adam
from utils import fetch_mnist_data, normalize_mnist_data, plot_single_train


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

set_seed()

# Load data
x_train, x_test, t_train, t_test = fetch_mnist_data()
x_train, x_test = normalize_mnist_data(x_train, x_test)

# Create network
net = NeuralNetwork(
    layers=[
        Affine(128, HeNormal()),
        Relu(),
        Affine(64, HeNormal()),
        Relu(),
        Affine(10, HeNormal()),
    ],
    loss_layer=SoftMaxWithCrossEntropy(),
)

# Train
trainer = Trainer(net, Adam(lr=0.001))
loss_hist, acc_hist = trainer.fit(
    x_train, x_test, t_train, t_test, batch_size=100, epochs=10
)

# Evaluate
test_acc = trainer.evaluate(x_test, t_test)
print(f"Test Accuracy: {test_acc:.4f}\n")
print(net.structure())
plot_single_train(loss_hist, acc_hist, "MNIST Classification with Neural Network")
