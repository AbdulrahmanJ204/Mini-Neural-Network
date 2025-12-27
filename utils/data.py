from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def fetchData():
    X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
    x_train, x_test, t_train, t_test = train_test_split(
        X, y, test_size=10000, shuffle=False
    )
    # Reshape
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # One-hot encode
    encoder = OneHotEncoder(sparse_output=False)

    # Convert to integers
    t_train = t_train.astype(int)
    t_test = t_test.astype(int)

    t_train = encoder.fit_transform(t_train.reshape(-1, 1))
    t_test = encoder.transform(t_test.reshape(-1, 1))
    return x_train, x_test, t_train, t_test


def normailze_mnist_data(x_train, x_test):
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train /= 255.0
    x_test /= 255.0
    return x_train, x_test
