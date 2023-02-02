import numpy as np
from sklearn.metrics import f1_score, roc_curve, r2_score


def xgb_f1_score(y_pred, y_true_dmatrix):
    y_true = y_true_dmatrix.get_label()
    y_hat = np.zeros_like(y_pred)
    y_hat[y_pred > 0.5] = 1
    f1 = f1_score(y_true, y_hat)
    return 'f1', f1


def xgb_ks_score(y_pred, y_true_dmatrix):
    def ks_stats(y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value
    y_true = y_true_dmatrix.get_label()
    ks = ks_stats(y_true, y_pred)
    return 'ks', ks


def xgb_r2_score(y_pred, y_true_dmatrix):
    y_true = y_true_dmatrix.get_label()
    return 'r2', r2_score(y_true, y_pred)


def xgb_f1_score_negative(y_pred, y_true_dmatrix):
    y_true = y_true_dmatrix.get_label()
    y_hat = np.zeros_like(y_pred)
    y_hat[y_pred > 0.5] = 1
    f1 = f1_score(y_true, y_hat)
    return 'f1', -f1


def xgb_ks_score_negative(y_pred, y_true_dmatrix):
    def ks_stats(y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value
    y_true = y_true_dmatrix.get_label()
    ks = ks_stats(y_true, y_pred)
    return 'ks', -ks


def xgb_r2_score_negative(y_pred, y_true_dmatrix):
    y_true = y_true_dmatrix.get_label()
    return 'r2', -r2_score(y_true, y_pred)


def regularize_metric(eval_metric):
    """
    :param eval_metric: list of str, each str should be a built-in metric of sklearn
    """
    eval_metric_list = []
    feval = None
    for metric in eval_metric:
        if metric == 'f1':
            # For custom function, you can specify the optimization direction in lgb api,
            # but for built in gbdt_utils, lgb will automatically decide
            # according to the metric type, e.g., the smaller the better for error
            # but the bigger the better for auc, etc.
            feval = xgb_f1_score
        elif metric == 'accuracy':
            # It is calculated as #(wrong cases)/#(all cases).
            # The evaluation will regard the instances
            # with prediction value larger than 0.5 as positive instances, i.e., 1 instances)
            eval_metric_list.append('error')
        elif metric == 'roc_auc':
            eval_metric_list.append('auc')
        elif metric == 'ks':
            # same logic as f1
            feval = xgb_ks_score
        else:
            raise RuntimeError('not-supported metric: {}'.format(metric))
    return feval, eval_metric_list
