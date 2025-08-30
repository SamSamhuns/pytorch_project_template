import matplotlib.pyplot as plt
import numpy as np


def plot_scatter_loss(
        y_score: np.ndarray,
        y_true: np.ndarray,
        savepath: str = "ae_scatter_loss.png",
        title: str = "Scatter plot",
        xlabel: str = "Sequence Points",
        ylabel: str = "AE loss") -> None:
    """Plot the loss scores of the autoencoder."""
    fig = plt.figure(figsize=(10, 10))

    idx_dict = [
        {"idx": np.where(y_true == 0)[0], "color": "blue",
         "mkr": 'o', "label": "Normal"},
        {"idx": np.where(y_true == 1)[0], "color": "red", "mkr": 'x', "label": "Anomaly"}]
    for idict in idx_dict:
        plt.scatter(
            idict["idx"], y_score[idict["idx"]],
            color=idict["color"], s=5, alpha=0.6,
            marker=idict["mkr"], label=idict["label"])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')

    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1, dpi=350)
    plt.close(fig)


def plot_scatter_2d(
        X: np.ndarray,
        y: np.ndarray,
        savepath: str,
        title: str = "Scatter plot",
        xlabel: str = "t-SNE Dimension 1",
        ylabel: str = "t-SNE Dimension 2") -> None:
    """Create a scatter plot
    Parameters:
        X (np.ndarray): The input data points, 2D array of shape (n_samples, 2).
        y (np.ndarray): The labels for each data point, 1D array of shape (n_samples,).
        savepath (str): The filename to save the scatter plot image.
        title (str, optional): The title of the plot. Defaults to "Scatter plot".
        xlabel (str, optional): The label for the x-axis. Defaults to "t-SNE Dimension 1".
        ylabel (str, optional): The label for the y-axis. Defaults to "t-SNE Dimension 2".
    """
    fig = plt.figure(figsize=(15, 10))
    for label in np.unique(y):
        indices = (y == label).squeeze()
        plt.scatter(X[indices, 0],
                    X[indices, 1],
                    label=f"Class: {label}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def plot_scatter_3d(
        X: np.ndarray,
        y: np.ndarray,
        savepath: str,
        title: str = "Scatter plot 3D",
        xlabel: str = "Dimension 1",
        ylabel: str = "Dimension 2",
        zlabel: str = "Dimension 3") -> None:
    """Create a 3D scatter plot
    Parameters:
        X (np.ndarray): The input data points, 2D array of shape (n_samples, 3).
        y (np.ndarray): The labels for each data point, 1D array of shape (n_samples,).
        savepath (str): The filename to save the scatter plot image.
        title (str, optional): The title of the plot. Defaults to "Scatter plot 3D".
        xlabel (str, optional): The label for the x-axis. Defaults to "Dimension 1".
        ylabel (str, optional): The label for the y-axis. Defaults to "Dimension 2".
        zlabel (str, optional): The label for the z-axis. Defaults to "Dimension 3".
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(y):
        indices = (y == label).squeeze()
        ax.scatter(X[indices, 0],
                   X[indices, 1],
                   X[indices, 2],
                   label=f"Class: {label}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()

    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
