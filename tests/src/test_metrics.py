import numpy as np
import pytest

from src.metrics import PLOTS, calc_metric, plot_metric


def test_top_k_accuracy_uses_class_probabilities():
    """top_k_accuracy_score needs y_score, the class probabilities, not the predicted labels"""
    y_true = np.array([0, 1, 2, 2])
    y_score = np.array(
        [
            [0.6, 0.3, 0.1],  # top-2: 0, 1  -> hit
            [0.5, 0.1, 0.4],  # top-2: 0, 2  -> miss
            [0.2, 0.3, 0.5],  # top-2: 2, 1  -> hit
            [0.1, 0.2, 0.7],
        ]
    )  # top-2: 2, 1  -> hit
    y_pred = y_score.argmax(axis=1)
    assert calc_metric("top_k_accuracy_score", y_true=y_true, y_score=y_score, y_pred=y_pred) == pytest.approx(0.75)


def test_average_comes_from_the_caller():
    """A 3-class val split that holds 2 of the classes was scored as binary, which only counts class 1"""
    y_true, y_pred = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    assert calc_metric("f1_score", y_true, y_pred=y_pred, average="macro") == pytest.approx(0.7333, abs=1e-4)
    assert calc_metric("f1_score", y_true, y_pred=y_pred, average="binary") == pytest.approx(0.8)


def test_plot_metric_saves_each_plot_and_closes_it(tmp_path):
    import matplotlib.pyplot as plt

    y_true, y_score = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
    for name in PLOTS:
        plot_metric(name, y_true, y_score, str(tmp_path / f"{name}.png"))
        assert (tmp_path / f"{name}.png").is_file()
    assert plt.get_fignums() == []
