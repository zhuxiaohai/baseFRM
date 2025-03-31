import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor
from shaphypetune._classes import _FastRFE
from shaphypetune.gbdt_utils.xgboost_metrics import xgb_ks_score_negative, xgb_r2_score_negative
from shaphypetune.scorecard.utils import cal_ks


def test_fastrfe_classificatiion():
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    param_dist = {
        'max_depth': [2, 3, 4],
        'subsample': [0.7],
        'min_child_weight': [20, 30, 10],
        'base_score': [0.5, 0.7, 0.6, 0.4],
        'reg_lambda': [1, 5, 10, 15],
        'learning_rate': [0.1, 0.08, 0.12],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'reg_alpha': [1, 10],
        'random_state': stats.rv_discrete(values=([i*8 for i in range(1000)], [1/1000]*1000))
    }
    clf_xgb = XGBClassifier(n_estimators=200, verbosity=0, n_jobs=2)
    model = _FastRFE(clf_xgb, min_features_to_select=5, param_grid=param_dist, n_iter=5, n_warmup_iter=1,
                     sampling_seed=1, verbose=2)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=6)

    print('n_features', model.n_features_)
    np.testing.assert_almost_equal([cal_ks(model.predict_proba(x_valid)[:, 1], y_valid)[0]],
                                   [model.best_score_], decimal=5)
    np.testing.assert_almost_equal([cal_ks(model.estimator_.predict_proba(x_valid[:, model.support_])[:, 1], y_valid)[0]],
                                   [model.best_score_], decimal=5)

    print(model.best_params_)
    afsxc = XGBClassifier(n_estimators=200, verbosity=0, n_jobs=2, **model.best_params_)
    afsxc.fit(x_train[:, model.support_], y_train,
              eval_set=[(x_valid[:, model.support_], y_valid)],
              early_stopping_rounds=6,
              eval_metric=xgb_ks_score_negative)
    test_pred = afsxc.predict_proba(x_valid[:, model.support_])[:, 1]
    np.testing.assert_almost_equal([cal_ks(test_pred, y_valid)[0]],
                                   [model.best_score_], decimal=5)


def test_fastrfe_regression():
    X, y = load_boston(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    param_dist = {
        'max_depth': [2, 3, 4],
        'subsample': [0.7],
        'min_child_weight': [20, 30, 10],
        'reg_lambda': [1, 5, 10, 15],
        'learning_rate': [0.1, 0.08, 0.12],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'reg_alpha': [1, 10],
        'random_state': stats.rv_discrete(values=([i*8 for i in range(1000)], [1/1000]*1000))
    }
    clf_xgb = XGBRegressor(n_estimators=200, verbosity=0, n_jobs=2, base_score=np.mean(y_train))
    model = _FastRFE(clf_xgb, min_features_to_select=5, param_grid=param_dist, n_iter=5, n_warmup_iter=3,
                     importance_type='shap_importances', train_importance=False, sampling_seed=1, verbose=2)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=6, verbose=False)

    print('n_features', model.n_features_)
    np.testing.assert_almost_equal([r2_score(y_valid, model.predict(x_valid))],
                                   [model.best_score_], decimal=5)
    np.testing.assert_almost_equal([r2_score(y_valid, model.estimator_.predict(x_valid[:, model.support_]))],
                                   [model.best_score_], decimal=5)

    print(model.best_params_)
    afsxc = XGBRegressor(n_estimators=200, verbosity=0, n_jobs=2, base_score=np.mean(y_train), **model.best_params_)
    afsxc.fit(x_train[:, model.support_], y_train,
              eval_set=[(x_valid[:, model.support_], y_valid)],
              early_stopping_rounds=6,
              eval_metric=xgb_r2_score_negative)
    test_pred = afsxc.predict(x_valid[:, model.support_])
    np.testing.assert_almost_equal([r2_score(y_valid, test_pred)],
                                   [model.best_score_], decimal=5)