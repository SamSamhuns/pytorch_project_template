from typing import Union

import numpy as np
from sklearn.manifold import TSNE

from modules.utils.visualization.common import plot_scatter_2d, plot_scatter_3d


def plot_tsne_2d(
        X: np.ndarray, y: np.ndarray,
        savepath: str,
        learning_rate: Union[str, float] = "auto",
        title: str = "t-SNE plot",
        xlabel: str = "t-SNE dimension 1",
        ylabel: str = "t-SNE dimension 2",
        perplexity: int = 5,
        random_state: int = 42) -> None:
    """
    Generate a t-SNE plot and save a scatter plot with matplotlib.
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    # Fit and transform the data with the t-SNE model
    x_embedded = TSNE(n_components=2,
                      learning_rate=learning_rate,
                      perplexity=perplexity,
                      init='random',
                      random_state=random_state).fit_transform(X)

    # Create a scatter plot
    plot_scatter_2d(x_embedded, y, savepath,
                    title=title, xlabel=xlabel, ylabel=ylabel)


def plot_tsne_3d(
        X: np.ndarray, y: np.ndarray,
        savepath: str,
        learning_rate: Union[str, float] = "auto",
        title: str = "t-SNE plot 3D",
        xlabel: str = "Dimension 1",
        ylabel: str = "Dimension 2",
        zlabel: str = "Dimension 3",
        perplexity: int = 5,
        random_state: int = 42) -> None:
    """
    Generate a 3D t-SNE plot and save a scatter plot with matplotlib.
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    # Fit and transform the data with the t-SNE model for 3 dimensions
    x_embedded = TSNE(n_components=3,
                      learning_rate=learning_rate,
                      perplexity=perplexity,
                      init='random',
                      random_state=random_state).fit_transform(X)

    # Create a 3D scatter plot
    plot_scatter_3d(x_embedded, y, savepath,
                    title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)


if __name__ == "__main__":
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1],
                 [1, 1, 1], [1, 1, 0], [0, 0, 1]])
    y = np.expand_dims(np.array([1, 1, 2, 1, 1, 2]), axis=-1)
    plot_tsne_2d(X, y, "test.png")
