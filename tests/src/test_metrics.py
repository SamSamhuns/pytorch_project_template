import pytest
import torch
from src.metrics import accuracy_topk_torch, tpr, tnr, ppv, npv, fpr, fnr, fdr, comprehensive_clsf_metrics
from src.metrics.base_metrics import calc_clsf_metrics


def test_accuracy_topk():
    y_pred = torch.tensor(
        [[0.1, 0.2, 0.7], [0.4, 0.5, 0.1],
         [0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    y_true = torch.tensor([2, 1, 2, 1])
    topk_results = accuracy_topk_torch(y_pred, y_true, topk=(1, 2))

    # Check top-1 accuracy
    assert topk_results[0] == 50, "Expected 50% accuracy for top-1"

    # Check top-2 accuracy
    assert topk_results[1] == 75, "Expected 75% accuracy for top-2"


def test_binary_classification_metrics():
    y_pred = [0, 1, 0, 1]
    y_true = [0, 1, 1, 0]

    # Test individual basic metrics
    tp, tn, fp, fn = calc_clsf_metrics(y_pred, y_true)
    assert tp == 1 and tn == 1 and fp == 1 and fn == 1, "Expected each metric to be 1"

    # Test advanced metrics
    assert tpr(y_pred, y_true) == 0.5, "Expected TPR of 0.5"
    assert tnr(y_pred, y_true) == 0.5, "Expected TNR of 0.5"
    assert ppv(y_pred, y_true) == 0.5, "Expected PPV of 0.5"
    assert npv(y_pred, y_true) == 0.5, "Expected NPV of 0.5"
    assert fpr(y_pred, y_true) == 0.5, "Expected FPR of 0.5"
    assert fnr(y_pred, y_true) == 0.5, "Expected FNR of 0.5"
    assert fdr(y_pred, y_true) == 0.5, "Expected FDR of 0.5"


def test_comprehensive_metrics():
    y_pred = [0, 1, 0, 1]
    y_true = [0, 1, 1, 0]
    metrics = comprehensive_clsf_metrics(y_pred, y_true)
    expected_values = {
        "tpr": 0.5, "tnr": 0.5, "ppv": 0.5, "npv": 0.5,
        "fpr": 0.5, "fnr": 0.5, "fdr": 0.5, "acc": 0.5
    }
    for key, val in expected_values.items():
        assert pytest.approx(
            metrics[key], abs=1e-6) == val, f"Expected {key} to be {val}"
