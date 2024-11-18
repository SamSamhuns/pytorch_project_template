import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay


def plot_roc_curve(y_true: ArrayLike, y_pred: ArrayLike, savepath: str = "roc_curve.png",  **kwargs):
    """Plot the ROC curve for a binary classification problem."""
    fig = plt.figure(figsize=(15, 15))
    RocCurveDisplay.from_predictions(y_true, y_pred, **kwargs)

    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1, dpi=350)
    plt.close(fig)


def plot_pr_curve(y_true: ArrayLike, y_pred: ArrayLike, savepath: str = "precision_recall_curve.png", **kwargs):
    """Plot the Precision-Recall curve for a binary classification problem."""
    fig = plt.figure(figsize=(15, 15))
    PrecisionRecallDisplay.from_predictions(y_true, y_pred, **kwargs)

    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1, dpi=350)
    plt.close(fig)


def plot_calibration_curve(y_true: ArrayLike, y_score: ArrayLike, savepath: str = "calibration_curve.png", **kwargs):
    """Plot the Calibration curve for a binary classification problem."""
    fig = plt.figure(figsize=(15, 15))
    CalibrationDisplay.from_predictions(y_true, y_prob=y_score, n_bins=10, **kwargs)

    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1, dpi=350)
    plt.close(fig)
