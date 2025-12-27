import numpy as np
import sys
from time import time



def plot_two_train(loss1, acc1, loss2, acc2, figName="training_plot"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # --- Loss Plot ---
    plt.subplot(1, 2, 1)
    plt.plot(loss1, label="Loss (Raw Data)")
    plt.plot(loss2, label="Loss (Normalized Data)")
    plt.title("Loss Curve")
    plt.legend()

    # --- Accuracy Plot ---
    plt.subplot(1, 2, 2)
    plt.plot(acc1, label="Accuracy (Raw Data)")
    plt.plot(acc2, label="Accuracy (Normalized Data)")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{figName}.png")


def plot_single_train(loss, acc, figName="training_plot"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # --- Loss Plot ---
    plt.subplot(1, 2, 1)
    plt.plot(loss, label="Loss")
    plt.title("Loss Curve")
    plt.legend()

    # --- Accuracy Plot ---
    plt.subplot(1, 2, 2)
    plt.plot(acc, label="Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{figName}.png")
