# classification metrics
import numpy as np
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix)
from .base_metrics import (
    accuracy_topk_torch, comprehensive_clsf_metrics,
    tpr, tnr, ppv, npv, fpr, fnr, fdr)


IMPLEMENTED_METRICS = {
    "accuracy_topk_torch": accuracy_topk_torch,
    "accuracy_score": accuracy_score,
    "top_k_accuracy_score": top_k_accuracy_score,
    "f1_score": f1_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "roc_auc_score": roc_auc_score,
    "average_precision_score": average_precision_score,
    "confusion_matrix": confusion_matrix,
    "classification_report": classification_report,
    "true_positive_rate": tpr,
    "true_negative_rate": tnr,
    "positive_predictive_value": ppv,
    "negative_predictive_value": npv,
    "false_positive_rate": fpr,
    "false_negative_rate": fnr,
    "false_discovery_rate": fdr,
}


def calc_metric(metric_name: str, **kwargs):
    """Initialize and calculate the metric."""
    try:
        if metric_name in {"roc_auc_score", "average_precision_score"}:
            kwargs = {k: v for k, v in kwargs.items() if k != "y_pred"}
        else:
            kwargs = {k: v for k, v in kwargs.items() if k != "y_score"}
        n_classes = np.unique(kwargs["y_true"]).size
        if n_classes > 2 and metric_name in {"f1_score", "precision_score", "recall_score"}:
            kwargs = {**kwargs, **{"average": "macro"}}
        metric = IMPLEMENTED_METRICS[metric_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{metric_name} is not implemented. " +
            f"Available Metrics: {IMPLEMENTED_METRICS.keys()}") from exc
    return metric
