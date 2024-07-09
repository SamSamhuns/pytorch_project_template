"""
Base Metrics
"""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy_topk_torch(y_pred: torch.tensor, y_true: torch.tensor, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_true.size(0)

    # Get top maxk indices of predictions (shape: [batch_size, maxk])
    _, pred = y_pred.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # Transpose to make it [maxk, batch_size]

    # Compare with ground truth label and count corrects
    correct = pred.eq(y_true.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # Count correct predictions in top-k
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
        res.append(correct_k * 100.0 / batch_size)
    return res


def calc_metrics(y_pred, y_true, pos_class=1, neg_class=0):
    """
    Calculate and return clsf tp, tn, fp, and fn metrics 
    given binary classification labels.
    """
    tp = sum((p == t == pos_class for p, t in zip(y_pred, y_true)))
    tn = sum((p == t == neg_class for p, t in zip(y_pred, y_true)))
    fp = sum((p == pos_class and t == neg_class for p, t in zip(y_pred, y_true)))
    fn = sum((p == neg_class and t == pos_class for p, t in zip(y_pred, y_true)))
    return tp, tn, fp, fn


def tpr(y_pred, y_true, pos_class=1, neg_class=0):
    """tpv, sensitivity, recall or true_positive_rate"""
    tp, _, _, fn = calc_metrics(y_pred, y_true, pos_class, neg_class)
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def tnr(y_pred, y_true, pos_class=1, neg_class=0):
    """tnv, specificity or true_negative_rate"""
    _, tn, fp, _ = calc_metrics(y_pred, y_true, pos_class, neg_class)
    if tn + fp == 0:
        return 0
    return tn / (tn + fp)


def ppv(y_pred, y_true, pos_class=1, neg_class=0):
    """ppv, precision or positive_predictive_value"""
    tp, _, fp, _ = calc_metrics(y_pred, y_true, pos_class, neg_class)
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def npv(y_pred, y_true, pos_class=1, neg_class=0):
    """npv, or negative predictive value"""
    _, tn, _, fn = calc_metrics(y_pred, y_true, pos_class, neg_class)
    if tn + fn == 0:
        return 0
    return tn / (tn + fn)


def fpr(y_pred, y_true, pos_class=1, neg_class=0):
    """fpr, or false positive rate"""
    _, tn, fp, _ = calc_metrics(y_pred, y_true, pos_class, neg_class)
    if fp + tn == 0:
        return 0
    return fp / (fp + tn)


def fnr(y_pred, y_true, pos_class=1, neg_class=0):
    """fnr, or false negative rate"""
    tp, _, _, fn = calc_metrics(y_pred, y_true, pos_class, neg_class)
    if fn + tp == 0:
        return 0
    return fn / (fn + tp)


def fdr(y_pred, y_true, pos_class=1, neg_class=0):
    """fdr, or false discovery rate"""
    tp, _, fp, _ = calc_metrics(y_pred, y_true, pos_class, neg_class)
    if fp + tp == 0:
        return 0
    return fp / (fp + tp)


def comprehensive_clsf_metrics(y_pred, y_true):
    """
    Calculate and return a comprehensive set of classification metrics.

    This function computes a variety of metrics used to evaluate the performance 
    of a binary classification model. These metrics include true positives (tp), 
    true negatives (tn), false positives (fp), false negatives (fn), true positive 
    rate (tpr), true negative rate (tnr), positive predictive value (ppv), 
    negative predictive value (npv), false positive rate (fpr), false negative rate (fnr), 
    false discovery rate (fdr), and overall accuracy (acc).

    Parameters:
    - y_pred (array-like): Predicted binary labels from the model. Should be 1 for positive class and 0 for negative class.
    - y_true (array-like): True binary labels. Should be 1 for positive class and 0 for negative class.

    Returns:
    - dict: A dictionary containing the following key-value pairs:
        - 'tp': Number of true positives
        - 'tn': Number of true negatives
        - 'fp': Number of false positives
        - 'fn': Number of false negatives
        - 'tpr': True positive rate (sensitivity, recall)
        - 'tnr': True negative rate (specificity)
        - 'ppv': Positive predictive value (precision)
        - 'npv': Negative predictive value
        - 'fpr': False positive rate
        - 'fnr': False negative rate
        - 'fdr': False discovery rate
        - 'acc': Overall accuracy

    Note:
    - This function assumes binary classification and the inputs must be binary arrays.
    - The function does not handle missing values or check input types. 
      Ensure that the inputs are clean and of the correct type before using this function.
    """
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    # sensitivity, hit rate, recall, or true positive rate
    tpr = tp/(tp+fn)
    # specificity or true negative rate
    tnr = tn/(tn+fp)
    # precision or positive predictive value
    ppv = tp/(tp+fp)
    # negative predictive value
    npv = tn/(tn+fn)
    # fall out or false positive rate
    fpr = fp/(fp+tn)
    # false negative rate
    fnr = fn/(tp+fn)
    # false discovery rate
    fdr = fp/(tp+fp)

    # overall accuracy
    acc = (tp+tn)/(tp+fp+fn+tn)
    return {
        "tpr": tpr,
        "tnr": tnr,
        "ppv": ppv,
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
        "fdr": fdr,
        "acc": acc
    }
