import optuna
with optuna._imports.try_import() as _imports:
    import xgboost as xgb

from shaphypetune.optunahypetune.utils import train_val_score
from shaphypetune.gbdt_utils.xgboost_callbacks import EvaluationMonitor, EarlyStopping, XGBoostPruningCallback

import warnings
warnings.filterwarnings('ignore')


class ObjectiveXGB(object):
    def __init__(self, tuning_param_dict, train_set, monitor, coef_train_val_disparity, maximize, **kwargs):
        self.tuning_param_dict = tuning_param_dict
        self.train_set = train_set
        self.monitor = monitor
        self.coef_train_val_disparity = coef_train_val_disparity
        self.maximize = maximize
        self.kwargs = kwargs

    def __call__(self, trial):
        trial_param_dict = {}
        param = self.tuning_param_dict.get('booster')
        if isinstance(param, tuple):
            suggest_type = param[0]
            suggest_param = param[1]
            trial_param_dict['booster'] = eval('trial.suggest_' + suggest_type)('booster', **suggest_param)
        elif param is not None:
            trial_param_dict['booster'] = param
        booster = trial_param_dict.get('booster')
        for key, param in self.tuning_param_dict.items():
            if key == 'booster':
                continue
            if (booster is None) or booster == 'gbtree':
                if key in ['sample_type', 'normalize_type', 'one_drop', 'rate_drop', 'skip_drop']:
                    continue
            if isinstance(param, tuple):
                suggest_type = param[0]
                suggest_param = param[1]
                trial_param_dict[key] = eval('trial.suggest_' + suggest_type)(key, **suggest_param)
            else:
                trial_param_dict[key] = param

        monitor = EvaluationMonitor(rank=0,
                                    period=1,
                                    w=self.coef_train_val_disparity,
                                    show_stdv=True)
        earlystopping = EarlyStopping(rounds=self.kwargs.get('early_stopping_rounds', 10),
                                      w=self.coef_train_val_disparity,
                                      maximize=self.maximize,
                                      save_best=False)
        pruning = XGBoostPruningCallback(trial)

        # when folds is defined in xgb.cv, nfold will not be used
        cvresult = xgb.cv(params=trial_param_dict,
                          dtrain=self.train_set,
                          as_pandas=True,
                          maximize=self.maximize,
                          callbacks=[monitor, pruning, earlystopping]
                          if self.tuning_param_dict.get('verbosity', 0) > 0 else [pruning, earlystopping],
                          **self.kwargs)
        train_score = cvresult['train-' + self.monitor + '-mean'].iloc[-1]
        val_score = cvresult['test-' + self.monitor + '-mean'].iloc[-1]
        best_score = train_val_score(train_score, val_score, self.coef_train_val_disparity)
        num_parallel_tree = trial_param_dict.get('num_parallel_tree', 1)
        trial.set_user_attr("n_iterations", cvresult.shape[0])
        trial.set_user_attr("n_estimators", cvresult.shape[0]*num_parallel_tree)
        trial.set_user_attr("train_score", train_score)
        trial.set_user_attr("val_score", val_score)
        return best_score
