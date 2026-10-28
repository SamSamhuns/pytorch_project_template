import numpy as np
import pytest

from src.metrics import calc_metric


def test_top_k_accuracy_uses_class_probabilities():
    """top_k_accuracy_score needs y_score, the class probabilities, not the predicted labels"""
    y_true = np.array([0, 1, 2, 2])
    y_score = np.array([[0.6, 0.3, 0.1],   # top-2: 0, 1  -> hit
                        [0.5, 0.1, 0.4],   # top-2: 0, 2  -> miss
                        [0.2, 0.3, 0.5],   # top-2: 2, 1  -> hit
                        [0.1, 0.2, 0.7]])  # top-2: 2, 1  -> hit
    y_pred = y_score.argmax(axis=1)
    assert calc_metric("top_k_accuracy_score", y_true=y_true, y_score=y_score, y_pred=y_pred) == pytest.approx(0.75)
