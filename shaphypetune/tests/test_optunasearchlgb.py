import numpy as np
from scipy import special
from sklearn.metrics import roc_auc_score, mean_squared_error
import lightgbm as lgb

from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split

from shaphypetune.optunahypetune.lightgbm.lightgbm_search import OptunaSearchLGB
from shaphypetune.gbdt_utils.lightgbm_callbacks import early_stopping
from shaphypetune.gbdt_utils.lightgbm_lossfn import FocalLoss
from shaphypetune.gbdt_utils.lightgbm_metrics import eval_ks, eval_top, eval_auc, ks_stats
from shaphypetune.optunahypetune.utils import make_train_val


def test_optunasearchlgb_classification_focalloss():
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

    feval = [eval_auc, eval_ks, eval_top]
    x_train_val, y_train_val, cv, folds = make_train_val(x_train, y_train, [(x_valid, y_valid)], cv=1, random_state=5)
    dmatrix = lgb.Dataset(x_train_val, label=y_train_val)
    op = OptunaSearchLGB('top_positive_ratio',
                         optimization_direction='maximize', coef_train_val_disparity=0,
                         n_trials=10, n_startup_trials=1, n_warmup_steps=1)
    tuning_param_dict = {
                         # 'objective': 'binary',
                         'fobj': (FocalLoss, {'gamma': ('float', {'low': 1, 'high': 3, 'step': 1}),
                                              'alpha': ('float', {'low': 0.1, 'high': 0.8, 'step': 0.1})}),
                         'verbosity': -1,
                         'seed': 2,
                         'deterministic': True,
                         'boosting': 'gbdt',
                         'eta': ('discrete_uniform', {'low': 0.07, 'high': 1.2, 'q': 0.01}),
                         'max_depth': ('int', {'low': 2, 'high': 6}),
                         'reg_lambda': ('int', {'low': 1, 'high': 20}),
                         'reg_alpha': ('int', {'low': 1, 'high': 20}),
                         'min_gain_to_split': ('int', {'low': 0, 'high': 3}),
                         'min_child_weight': ('int', {'low': 1, 'high': 30}),
                         'colsample_bytree': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'colsample_bynode': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'subsample': ('discrete_uniform', {'low': 0.7, 'high': 0.95, 'q': 0.05}),
                         'subsample_freq': ('int', {'low': 1, 'high': 10}),
                         'rate_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'skip_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'uniform_drop': ('categorical', {'choices': [True, False]})
    }
    op.search(tuning_param_dict, dmatrix, callbacks=[(early_stopping, {'stopping_rounds': 10, 'w': 0, 'verbose': True})],
              folds=folds, nfold=cv, shuffle=False, stratified=False, feval=feval)
    train_param = op.get_params()
    print(train_param)
    fl = FocalLoss(**train_param[1])
    init_score = fl.init_score(y_train_val)
    train_dmatrix = lgb.Dataset(x_train, label=y_train, reference=dmatrix,
                                init_score=np.full_like(y_train, init_score, dtype=float))
    test_dmatrix = lgb.Dataset(x_valid, label=y_valid,
                               reference=dmatrix, init_score=np.full_like(y_valid, init_score, dtype=float))

    afsxc = lgb.train(train_param[0], train_dmatrix,
                      num_boost_round=train_param[2]['n_iterations'], fobj=fl.get_loss,
                      valid_sets=[train_dmatrix, test_dmatrix], valid_names=['train', 'val'], feval=feval,
                      keep_training_booster=True)
    train_pred = special.expit(init_score + afsxc.predict(x_train))
    assert roc_auc_score(y_train, train_pred) == afsxc.eval_train(feval)[0][2]
    assert ks_stats(y_train, train_pred) == afsxc.eval_train(feval)[1][2]
    test_pred = special.expit(init_score + afsxc.predict(x_valid))
    assert roc_auc_score(y_valid, test_pred) == afsxc.eval_valid(feval)[0][2]
    assert ks_stats(y_valid, test_pred) == afsxc.eval_valid(feval)[1][2]


def test_optunasearchlgb_classification():
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

    feval=[eval_auc, eval_ks, eval_top]
    x_train_val, y_train_val, cv, folds = make_train_val(x_train, y_train, [(x_valid, y_valid)], cv=1, random_state=5)
    dmatrix = lgb.Dataset(x_train_val, label=y_train_val)
    op = OptunaSearchLGB('top_positive_ratio',
                         optimization_direction='maximize', coef_train_val_disparity=0,
                         n_warmup_steps=1, n_startup_trials=1, n_trials=10)
    tuning_param_dict = {
                         'objective': 'binary',
                         'verbosity': -1,
                         'seed': 2,
                         'deterministic': True,
                         'boosting': 'gbdt',
                         'eta': ('discrete_uniform', {'low': 0.07, 'high': 1.2, 'q': 0.01}),
                         'max_depth': ('int', {'low': 2, 'high': 6}),
                         'reg_lambda': ('int', {'low': 1, 'high': 20}),
                         'reg_alpha': ('int', {'low': 1, 'high': 20}),
                         'min_gain_to_split': ('int', {'low': 0, 'high': 3}),
                         'min_child_weight': ('int', {'low': 1, 'high': 30}),
                         'colsample_bytree': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'colsample_bynode': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'subsample': ('discrete_uniform', {'low': 0.7, 'high': 0.95, 'q': 0.05}),
                         'subsample_freq': ('int', {'low': 1, 'high': 10}),
                         'rate_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'skip_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'uniform_drop': ('categorical', {'choices': [True, False]})
    }
    op.search(tuning_param_dict, dmatrix, callbacks=[(early_stopping, {'stopping_rounds': 10, 'w': 0, 'verbose': True})],
              folds=folds, nfold=cv, shuffle=False, stratified=False, feval=feval)
    train_param = op.get_params()
    print(train_param)
    train_dmatrix = lgb.Dataset(x_train, label=y_train, reference=dmatrix)
    test_dmatrix = lgb.Dataset(x_valid, label=y_valid, reference=dmatrix)

    afsxc = lgb.train(train_param[0], train_dmatrix,
                      num_boost_round=train_param[2]['n_iterations'],
                      valid_sets=[train_dmatrix, test_dmatrix], valid_names=['train', 'val'], feval=feval,
                      keep_training_booster=True)
    train_pred = afsxc.predict(x_train)
    assert roc_auc_score(y_train, train_pred) == afsxc.eval_train(feval)[1][2]
    assert ks_stats(y_train, train_pred) == afsxc.eval_train(feval)[2][2]
    test_pred = afsxc.predict(x_valid)
    assert roc_auc_score(y_valid, test_pred) == afsxc.eval_valid(feval)[1][2]
    assert ks_stats(y_valid, test_pred) == afsxc.eval_valid(feval)[2][2]


def test_classification_train():
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

    fl = FocalLoss(alpha=0.9, gamma=0.05)
    fit = lgb.Dataset(x_train, y_train, init_score=np.full_like(y_train, fl.init_score(y_train), dtype=float))
    val = lgb.Dataset(x_valid, y_valid, reference=fit, init_score=np.full_like(y_valid, fl.init_score(y_valid), dtype=float))

    model = lgb.train(
        params={
            'verbosity': -1,
            'learning_rate': 0.01,
            'seed': 2021
        },
        train_set=fit,
        num_boost_round=20,
        valid_sets=(fit, val),
        valid_names=('fit', 'val'),
        early_stopping_rounds=5,
        verbose_eval=1,
        fobj=fl.get_loss,
        feval=[fl.eval_focal_loss, eval_auc, eval_ks],
    )
    # init_score affects loss, but not ranking scores like auc and ks
    logit = fl.init_score(y_valid) + model.predict(x_valid)
    y_pred = special.expit(fl.init_score(y_valid) + model.predict(x_valid))
    # init_score affects loss, but not ranking scores like auc and ks
    # note that we can not use eval_train, eval_valid because the prediction will use the iteration at stopping
    # but it is not the best iteration
    best_score = model.best_score['val']
    assert fl.eval_focal_loss(logit, val)[1] == best_score['focal_loss']
    assert roc_auc_score(y_valid, y_pred) == best_score['auc_score']
    assert ks_stats(y_valid, y_pred) == best_score['ks_score']


def test_classification_cv():
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

    fl = FocalLoss(alpha=0.9, gamma=0.05)
    x_train_val, y_train_val, cv, folds = make_train_val(x_train, y_train, [(x_valid, y_valid)], cv=1, random_state=5)
    fit = lgb.Dataset(x_train_val, y_train_val, init_score=np.full_like(y_train_val, fl.init_score(y_train_val), dtype=float))
    result = lgb.cv(
        params={
            'verbosity': -1,
            'learning_rate': 0.01,
            'seed': 2021
        },
        train_set=fit,
        nfold=cv,
        folds=folds,
        num_boost_round=100,
        early_stopping_rounds=20,
        verbose_eval=True,
        eval_train_metric=True,
        return_cvbooster=True,
        fobj=fl.get_loss,
        feval=[fl.eval_focal_loss, eval_auc, eval_ks]
    )
    n_boosting = len(result['valid ks_score-mean'])

    train = lgb.Dataset(x_train, y_train, reference=fit,
                        init_score=np.full_like(y_train, fl.init_score(y_train_val), dtype=float))
    val = lgb.Dataset(x_valid, y_valid, reference=fit,
                      init_score=np.full_like(y_valid, fl.init_score(y_train_val), dtype=float))
    model = lgb.train(
        params={
            'verbosity': -1,
            'learning_rate': 0.01,
            'seed': 2021
        },
        train_set=train,
        num_boost_round=n_boosting,
        valid_sets=(train, val),
        valid_names=('fit', 'val'),
        verbose_eval=100,
        fobj=fl.get_loss,
        feval=[fl.eval_focal_loss, eval_auc, eval_ks]
    )
    # init_score affects loss, but not ranking scores like auc and ks
    logit = fl.init_score(y_train_val) + model.predict(x_valid)
    y_pred = special.expit(fl.init_score(y_train_val) + model.predict(x_valid))
    # note that we can not use eval_train, eval_valid because the prediction will use the iteration at stopping
    # but it is not the best iteration
    best_score = model.best_score['val']
    np.testing.assert_almost_equal([fl.eval_focal_loss(logit, val)[1]], [best_score['focal_loss']], decimal=5)
    np.testing.assert_almost_equal([roc_auc_score(y_valid, y_pred)], [best_score['auc_score']], decimal=5)
    np.testing.assert_almost_equal([ks_stats(y_valid, y_pred)], [best_score['ks_score']], decimal=5)

    # the booster returned is from the last iteration, not the best iteration
    model = result['cvbooster'].boosters[0]
    logit = fl.init_score(y_train_val) + model.predict(x_valid)
    y_pred = special.expit(fl.init_score(y_train_val) + model.predict(x_valid))
    print("test loss {} ks {} auc {}".format(fl.eval_focal_loss(logit, val)[1],
                                             ks_stats(y_valid, y_pred),
                                             roc_auc_score(y_valid, y_pred)))


def test_regression_cv():
    X, y = load_boston(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

    x_train_val, y_train_val, cv, folds = make_train_val(x_train, y_train, [(x_valid, y_valid)], cv=1, random_state=5)
    fit = lgb.Dataset(x_train_val, y_train_val)
    result = lgb.cv(
        params={
            'verbosity': -1,
            'learning_rate': 0.01,
            'seed': 2021,
            'objective': 'regression'
        },
        train_set=fit,
        nfold=cv,
        folds=folds,
        num_boost_round=100,
        early_stopping_rounds=20,
        verbose_eval=True,
        eval_train_metric=True,
        return_cvbooster=False,
    )
    n_boosting = len(result['valid l2-mean'])
    train = lgb.Dataset(x_train, y_train, reference=fit)
    val = lgb.Dataset(x_valid, y_valid, reference=fit)
    model = lgb.train(
        params={
            'verbosity': -1,
            'learning_rate': 0.01,
            'seed': 2021,
            'objective': 'regression'
        },
        train_set=train,
        num_boost_round=n_boosting,
        valid_sets=(train, val),
        valid_names=('fit', 'val'),
    )

    y_pred = model.predict(x_valid)
    np.testing.assert_almost_equal([mean_squared_error(y_valid, y_pred)],
                                   [model.best_score['val']['l2']], decimal=5)


def test_optunasearchlgb_regression():
    X, y = load_boston(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

    x_train_val, y_train_val, cv, folds = make_train_val(x_train, y_train, [(x_valid, y_valid)], cv=1, random_state=5)
    dmatrix = lgb.Dataset(x_train_val, label=y_train_val)
    op = OptunaSearchLGB('l2', n_warmup_steps=1, n_startup_trials=1,
                         optimization_direction='minimize', coef_train_val_disparity=0, n_trials=10)
    tuning_param_dict = {
                         'objective': 'regression',
                         'verbosity': -1,
                         'seed': 2,
                         'deterministic': True,
                         'boosting': 'gbdt',
                         'eta': ('discrete_uniform', {'low': 0.07, 'high': 1.2, 'q': 0.01}),
                         'max_depth': ('int', {'low': 2, 'high': 6}),
                         'reg_lambda': ('int', {'low': 1, 'high': 20}),
                         'reg_alpha': ('int', {'low': 1, 'high': 20}),
                         'min_gain_to_split': ('int', {'low': 0, 'high': 3}),
                         'min_child_weight': ('int', {'low': 1, 'high': 30}),
                         'colsample_bytree': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'colsample_bynode': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'subsample': ('discrete_uniform', {'low': 0.7, 'high': 0.95, 'q': 0.05}),
                         'subsample_freq': ('int', {'low': 1, 'high': 10}),
                         'rate_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'skip_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'uniform_drop': ('categorical', {'choices': [True, False]})
    }
    op.search(tuning_param_dict, dmatrix, callbacks=[(early_stopping, {'stopping_rounds': 10, 'w': 0, 'verbose': True})],
              folds=folds, nfold=cv, shuffle=False, stratified=False)
    train_param = op.get_params()
    print(train_param)
    train_dmatrix = lgb.Dataset(x_train, label=y_train, reference=dmatrix)
    test_dmatrix = lgb.Dataset(x_valid, label=y_valid, reference=dmatrix)

    afsxc = lgb.train(train_param[0], train_dmatrix,
                      num_boost_round=train_param[2]['n_iterations'],
                      valid_sets=[train_dmatrix, test_dmatrix], valid_names=['train', 'val'])
    train_pred = afsxc.predict(x_train)
    np.testing.assert_almost_equal([mean_squared_error(y_train, train_pred)],
                                   [afsxc.best_score['train']['l2']], decimal=5)
    test_pred = afsxc.predict(x_valid)
    np.testing.assert_almost_equal([mean_squared_error(y_valid, test_pred)],
                                   [afsxc.best_score['val']['l2']], decimal=5)