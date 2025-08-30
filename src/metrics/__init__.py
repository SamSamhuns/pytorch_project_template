# classification metrics
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from .base_metrics import (
    accuracy_topk_torch,
    comprehensive_clsf_metrics,
    fdr,
    fnr,
    fpr,
    npv,
    optimal_thres_from_roc_curve,
    ppv,
    tnr,
    tpr,
)
from .plots import plot_calibration_curve, plot_pr_curve, plot_roc_curve

IMPLEMENTED_METRICS = {
    "accuracy_score": accuracy_score,
    "top_k_accuracy_score": top_k_accuracy_score,
    "f1_score": f1_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "roc_auc_score": roc_auc_score,
    "average_precision_score": average_precision_score,
    "classification_report": classification_report,
    "confusion_matrix": confusion_matrix,
    "accuracy_topk_torch": accuracy_topk_torch,
    "comprehensive_clsf_metrics": comprehensive_clsf_metrics,
    "optimal_thres_from_roc_curve": optimal_thres_from_roc_curve,
    "true_positive_rate": tpr,
    "true_negative_rate": tnr,
    "positive_predictive_value": ppv,
    "negative_predictive_value": npv,
    "false_positive_rate": fpr,
    "false_negative_rate": fnr,
    "false_discovery_rate": fdr,
}


IMPLEMENTED_METRICS_PLOTS = {
    "roc_curve": plot_roc_curve,
    "pr_curve": plot_pr_curve,
    "calibration_curve": plot_calibration_curve,
}


def plot_metric(metric_name: str, **kwargs) -> None:
    """Plot the metric."""
    try:
        if metric_name in {
                "roc_curve",
                "pr_curve"}:
            kwargs = {k: v for k, v in kwargs.items() if k != "y_score"}
        elif metric_name in {
                "calibration_curve"}:
            kwargs = {k: v for k, v in kwargs.items() if k != "y_pred"}
        IMPLEMENTED_METRICS_PLOTS[metric_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{metric_name} is not implemented. " +
            f"Available metrics for plotting are {IMPLEMENTED_METRICS_PLOTS.keys()}") from exc


def calc_metric(metric_name: str, **kwargs):
    """Initialize and calculate the metric."""
    try:
        if metric_name in {
                "roc_auc_score",
                "average_precision_score",
                "optimal_thres_from_roc_curve"}:
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
