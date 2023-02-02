import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, r2_score
from scipy import special


def ks_stats(y_true, y_pred, **kwargs):
    fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
    ks_value = max(tpr - fpr)
    return ks_value


def eval_ks(y_pred, y_true_dmatrix):
    # preds: If custom fobj is specified, predicted values are returned before any transformation,
    # e.g. they are raw margin instead of probability of positive class for binary task in this case.
    def ks_stats(y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value

    y_true = y_true_dmatrix.get_label()
    # init score will not influence ranking
    y_pred = special.expit(y_pred)
    ks = ks_stats(y_true, y_pred)
    return 'ks_score', ks, True


def eval_top(preds, train_data):
    # preds: If custom fobj is specified, predicted values are returned before any transformation,
    # e.g. they are raw margin instead of probability of positive class for binary task in this case.
    preds = special.expit(preds)
    labels = train_data.get_label()
    auc = roc_auc_score(labels, preds)
    dct = pd.DataFrame({'pred': preds, 'percent': preds, 'labels': labels})
    key = dct['percent'].quantile(0.05)
    dct['percent'] = dct['percent'].map(lambda x: 1 if x >= key else 0)
    result = np.mean(dct[dct.percent == 1]['labels'] == 1) * 0.2 + auc * 0.8
    return 'top_positive_ratio', result, True


def eval_auc(y_pred, y_true_dmatrix):
    # preds: If custom fobj is specified, predicted values are returned before any transformation,
    # e.g. they are raw margin instead of probability of positive class for binary task in this case.
    def auc_stats(y_true, y_pred, **kwargs):
        auc = roc_auc_score(y_true, y_pred, **kwargs)
        return auc

    y_true = y_true_dmatrix.get_label()
    # init score will not influence ranking
    y_pred = special.expit(y_pred)
    auc = auc_stats(y_true, y_pred)
    return 'auc_score', auc, True


def eval_r2(y_pred, y_true_dmatrix):
    y_true = y_true_dmatrix.get_label()
    return 'r2', r2_score(y_true, y_pred), True