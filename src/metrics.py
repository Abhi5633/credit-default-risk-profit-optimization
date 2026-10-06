import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def gini(y_true, y_score):
    return 2 * roc_auc_score(y_true, y_score) - 1

def ks_statistic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))

def classification_metrics(y_true, y_score):
    return {
        "AUC": roc_auc_score(y_true, y_score),
        "Gini": gini(y_true, y_score),
        "KS": ks_statistic(y_true, y_score),
    }
