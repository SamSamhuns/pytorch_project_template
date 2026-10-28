"""Classification metrics from scikit-learn, and the curves of binary classifiers"""

from numpy.typing import ArrayLike
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

from .plots import PLOTS, plot_metric

# the metrics a config can list under metrics.val and metrics.test, besides the PLOTS
METRICS = {
    "accuracy_score": accuracy_score,
    "top_k_accuracy_score": top_k_accuracy_score,
    "f1_score": f1_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "roc_auc_score": roc_auc_score,
    "average_precision_score": average_precision_score,
    "classification_report": classification_report,
    "confusion_matrix": confusion_matrix,
}
# these score the class probabilities, y_score; the others score the predicted classes, y_pred
SCORE_METRICS = {"roc_auc_score", "average_precision_score", "top_k_accuracy_score"}
# these average the per-class scores as the config's metrics.average says, e.g. macro, or binary for class 1 alone
AVERAGED_METRICS = {"f1_score", "precision_score", "recall_score"}


def calc_metric(
    name: str, y_true: ArrayLike, y_score: ArrayLike | None = None, y_pred: ArrayLike | None = None, average="macro"
):
    """Score the predictions with the metric called name, one of METRICS"""
    kwargs = {"average": average} if name in AVERAGED_METRICS else {}
    return METRICS[name](y_true, y_score if name in SCORE_METRICS else y_pred, **kwargs)
