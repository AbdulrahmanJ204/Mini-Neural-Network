import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_two_train(loss1, acc1, loss2, acc2, figName="training_plot"):
    """Plot training curves comparing two different configurations.

    Creates side-by-side plots of loss and accuracy curves for two training runs,
    useful for comparing normalized vs. raw data or different hyperparameters.

    Args:
        loss1: List of loss values for first configuration.
        acc1: List of accuracy values for first configuration.
        loss2: List of loss values for second configuration.
        acc2: List of accuracy values for second configuration.
        figName: Name for the output PNG file (default: "training_plot").
    """
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
    """Plot training curves for a single training run.

    Creates side-by-side plots of loss and accuracy curves during training.

    Args:
        loss: List of loss values across training iterations/epochs.
        acc: List of accuracy values across epochs.
        figName: Name for the output PNG file (default: "training_plot").
    """
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
