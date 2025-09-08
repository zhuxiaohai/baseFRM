import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from hyperopt import hp
from numpy.random import default_rng
from shaphypetune import BoostRFA
from shaphypetune._classes import _SearchingRFA
from shaphypetune.scorecard.utils import cal_ks


def xgb_ks_score(y_true, y_pred):
    def ks_stats(y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value
    ks = ks_stats(y_true, y_pred)
    return -ks


def test_searchingrfa_clc():
    X_clf, y_clf = make_classification(n_samples=6000, n_features=20, n_classes=2,
                                       n_informative=4, n_redundant=6, random_state=0)

    X_clf_train, X_clf_valid, y_clf_train, y_clf_valid = train_test_split(
        X_clf, y_clf, test_size=0.3, shuffle=False)
    param_dist_hyperopt = {
        'max_depth': 1 + hp.randint('max_depth', 5),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
    }

    clf_xgb = XGBClassifier(n_estimators=200, verbosity=0, n_jobs=2, use_label_encoder=False, early_stopping_rounds=6, eval_metric=xgb_ks_score)
    sampling_seed = 0
    model = _SearchingRFA(clf_xgb, param_grid=param_dist_hyperopt, min_features_to_select=0, step=1, n_iter=8, sampling_seed=sampling_seed)
    model.fit(X_clf_train, y_clf_train,
              eval_set=[(X_clf_valid, y_clf_valid)], verbose=True,
              step_back=True)
    print('n_features', model.n_features_)
    print('features', np.arange(X_clf_train.shape[1])[model.support_])

    np.testing.assert_almost_equal([cal_ks(model.predict_proba(X_clf_valid)[:, 1], y_clf_valid)[0]],
                                   [-model.best_score_], decimal=5)
    np.testing.assert_almost_equal([min(model.score_history_)],
                                   [model.best_score_], decimal=5)

    print(model.estimator)
    print('best_params', model.best_params_)
    print('best_iter', model.best_iter_)
    afsxc = XGBClassifier(n_estimators=200, verbosity=0, n_jobs=2, use_label_encoder=False, early_stopping_rounds=6, eval_metric=xgb_ks_score, **model.best_params_)
    afsxc.fit(X_clf_train[:, model.support_], y_clf_train,
              eval_set=[(X_clf_valid[:, model.support_], y_clf_valid)],
              verbose=True)
    test_pred = afsxc.predict_proba(X_clf_valid[:, model.support_])[:, 1]
    np.testing.assert_almost_equal([cal_ks(test_pred, y_clf_valid)[0]],
                                   [-model.best_score_], decimal=5)
    

def test_rfa_clc():
    X_clf, y_clf = make_classification(n_samples=6000, n_features=20, n_classes=2,
                                       n_informative=4, n_redundant=6, random_state=0)

    X_clf_train, X_clf_valid, y_clf_train, y_clf_valid = train_test_split(
        X_clf, y_clf, test_size=0.3, shuffle=False)

    clf_xgb = XGBClassifier(n_estimators=200, verbosity=0, n_jobs=2, use_label_encoder=False, 
                            early_stopping_rounds=6, eval_metric=xgb_ks_score)

    # min_features_to_select的意识是，就是最后如果只剩min_features_to_select个特征可选了，就停止。因此要把所有特征都考虑一遍的话，应该设置成0
    model = BoostRFA(clf_xgb, min_features_to_select=1, step=1, train_importance=False, importance_type="shap_importances", verbose=1)

    model.fit(X_clf_train, y_clf_train,
                eval_set=[(X_clf_valid, y_clf_valid)], verbose=True,
                step_back=True)

    print("selected features number:", sum(model.support_))
    best_score = min(model.score_history_)
    np.testing.assert_almost_equal([cal_ks(model.predict_proba(X_clf_valid)[:, 1], y_clf_valid)[0]],
                                    [-best_score], decimal=5)
    afsxc = XGBClassifier(n_estimators=200, verbosity=0, n_jobs=2, use_label_encoder=False, early_stopping_rounds=6, eval_metric=xgb_ks_score)
    afsxc.fit(X_clf_train[:, model.support_], y_clf_train,
                eval_set=[(X_clf_valid[:, model.support_], y_clf_valid)],
                verbose=True)
    test_pred = afsxc.predict_proba(X_clf_valid[:, model.support_])[:, 1]
    np.testing.assert_almost_equal([cal_ks(test_pred, y_clf_valid)[0]],
                                    [-best_score], decimal=5)

