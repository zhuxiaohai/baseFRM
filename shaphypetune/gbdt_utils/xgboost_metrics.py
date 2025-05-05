import numpy as np
from sklearn.metrics import f1_score, roc_curve, r2_score


def xgb_f1_score(y_true, y_pred):
    y_hat = np.zeros_like(y_pred)
    y_hat[y_pred > 0.5] = 1
    f1 = f1_score(y_true, y_hat)
    return f1


def xgb_ks_score(y_true, y_pred):
    def ks_stats(y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value
    ks = ks_stats(y_true, y_pred)
    return ks


def xgb_f1_score_negative(y_true, y_pred):
    y_hat = np.zeros_like(y_pred)
    y_hat[y_pred > 0.5] = 1
    f1 = f1_score(y_true, y_hat)
    return -f1


def xgb_ks_score_negative(y_true, y_pred):
    def ks_stats(y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value
    ks = ks_stats(y_true, y_pred)
    return -ks


def xgb_r2_score_negative(y_true, y_pred):
    return -r2_score(y_true, y_pred)
