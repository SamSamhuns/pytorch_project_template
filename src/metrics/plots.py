"""Curves of binary classifiers. plot_metric imports pyplot itself, as importing it at the top
would add about 0.3 s to the startup of every script, even the ones that never plot
"""

from functools import partial

from numpy.typing import ArrayLike
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

# the plots a config can list under metrics.val and metrics.test
PLOTS = {
    "roc_curve": RocCurveDisplay.from_predictions,
    "pr_curve": PrecisionRecallDisplay.from_predictions,
    "calibration_curve": partial(CalibrationDisplay.from_predictions, n_bins=10),
}


def plot_metric(name: str, y_true: ArrayLike, y_score: ArrayLike, savepath: str) -> None:
    """Plot the curve called name, one of PLOTS, of a binary classifier and save it to savepath.
    y_score is P(class 1)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 15))
    PLOTS[name](y_true, y_score, ax=ax)
    fig.savefig(savepath, bbox_inches="tight", pad_inches=0.1, dpi=350)
    plt.close(fig)
