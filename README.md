# Neural Network Library

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```python
from layers import Affine, Relu, Sigmoid
from layers.initializers import XavierNormal
from layers.loss import SoftMaxWithCrossEntropy
from models import NeuralNetwork
from training import Trainer
from optimizers import Adam

# Create network
net = NeuralNetwork(
    layers=[
        Affine(128, XavierNormal()), Relu(),
        Affine(64, XavierNormal()), Relu(),
        Affine(10, XavierNormal())
    ],
    last_layer=SoftMaxWithCrossEntropy()
)

# Train
trainer = Trainer(net, Adam(lr=0.001))
loss_hist, acc_hist = trainer.fit(x_train, x_val, t_train, t_val)
```