"""
Stores custom metrics
"""
import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy_topk_torch(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def comprehensive_clsf_metrics(output, target):
    """
    Calculate comprehensive classification metrics.
    """
    cm = confusion_matrix(target, output)
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
