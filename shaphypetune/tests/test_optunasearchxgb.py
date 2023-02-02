import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb

from shaphypetune.optunahypetune.xgboost.xgboost_search import OptunaSearchXGB
from shaphypetune.gbdt_utils.xgboost_metrics import regularize_metric
from shaphypetune.optunahypetune.utils import make_train_val
from shaphypetune.scorecard.utils import cal_ks


def test_optunasearchxgb_classification():
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    x_train_val, y_train_val, cv, folds = make_train_val(x_train, y_train, [(x_valid, y_valid)], cv=1, random_state=5)
    feval, eval_metric_list = regularize_metric(['roc_auc', 'ks'])
    dmatrix = xgb.DMatrix(x_train_val, label=y_train_val)
    op = OptunaSearchXGB('ks', coef_train_val_disparity=0.4, optuna_verbosity=1, n_warmup_steps=1,
                         optuna_njobs=1, n_trials=10, n_startup_trials=1)
    tuning_param_dict = {'objective': 'binary:logistic',
                         'verbosity': 1,
                         'seed': 2,
                         'num_parallel_tree': ('int', {'low': 1, 'high': 4}),
                         'max_depth': ('int', {'low': 2, 'high': 6}),
                         'reg_lambda': ('int', {'low': 1, 'high': 20}),
                         'reg_alpha': ('int', {'low': 1, 'high': 20}),
                         'gamma': ('int', {'low': 0, 'high': 3}),
                         'min_child_weight': ('int', {'low': 1, 'high': 30}),
                         'base_score': ('discrete_uniform', {'low': 0.5, 'high': 0.9, 'q': 0.1}),
                         'colsample_bytree': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'colsample_bylevel': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'colsample_bynode': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'subsample': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'eta': ('discrete_uniform', {'low': 0.07, 'high': 1.2, 'q': 0.01}),
                         'rate_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'skip_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'tree_method': ('categorical', {'choices': ['auto', 'exact', 'approx', 'hist']}),
                         'booster': ('categorical', {'choices': ['gbtree', 'dart']}),
                         'sample_type': ('categorical', {'choices': ['uniform', 'weighted']}),
                         'normalize_type': ('categorical', {'choices': ['tree', 'forest']})}
    op.search(tuning_param_dict, dmatrix, nfold=cv, folds=folds, early_stopping_rounds=30,
              feval=feval, metrics=eval_metric_list)
    train_dmatrix = xgb.DMatrix(x_train, y_train)
    test_dmatrix = xgb.DMatrix(x_valid, y_valid)
    train_param = op.get_params()
    print(train_param)
    afsxc = xgb.train(train_param[0], train_dmatrix, num_boost_round=train_param[1]['n_iterations'])
    train_pred = afsxc.predict(train_dmatrix)
    test_pred = afsxc.predict(test_dmatrix)
    np.testing.assert_almost_equal([cal_ks(train_pred, y_train)[0]],
                                   [float(train_param[1]['train_score'])], decimal=5)
    np.testing.assert_almost_equal([cal_ks(test_pred, y_valid)[0]],
                                   [float(train_param[1]['val_score'])], decimal=5)
    # op.plot_optimization().show()
    # op.plot_importance(list(op.dynamic_param.keys())).show()
    # op.plot_score()
