import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np

from layers import Affine, Relu, BatchNormalization, Dropout
from initializers import HeNormal
from layers.loss.bce import BCE
from models import NeuralNetwork
from training import Trainer
from optimizers import Adam
from utils.visualization import plot_single_train
from utils.data import fetch_breast_cancer_data


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)



set_seed()

# Load data
x_train, x_test, t_train, t_test = fetch_breast_cancer_data()


net = NeuralNetwork(
    layers=[
        Affine(64, HeNormal()),
        BatchNormalization(),
        Relu(),
        Dropout(0.3),
        Affine(32, HeNormal()),
        Relu(),
        Dropout(0.2),
        Affine(16, HeNormal()),
        Relu(),
        Affine(1, HeNormal()),  
    ],
    loss_layer=BCE(),  
)

# Train
trainer = Trainer(net, Adam(lr=0.001))
loss_hist, acc_hist = trainer.fit(
    x_train, x_test, t_train, t_test, batch_size=32, epochs=50
)


test_acc = trainer.evaluate(x_test, t_test)
print(f"Test Accuracy: {test_acc:.4f}\n")

print(net.structure())


plot_single_train(loss_hist, acc_hist, "Breast_Cancer_Binary_Classification")