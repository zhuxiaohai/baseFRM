import numpy as np
import optuna
with optuna._imports.try_import() as _imports:
    import lightgbm as lgb
from optuna.integration.lightgbm import LightGBMPruningCallback

from shaphypetune.optunahypetune.utils import train_val_score

import warnings
warnings.filterwarnings('ignore')


class ObjectiveLGB(object):
    def __init__(self, tuning_param_dict, train_set, monitor, coef_train_val_disparity, callbacks, **kwargs):
        self.tuning_param_dict = tuning_param_dict
        self.train_set = train_set
        self.monitor = monitor
        self.coef_train_val_disparity = coef_train_val_disparity
        self.callbacks = callbacks
        self.kwargs = kwargs

    def __call__(self, trial):
        trial_param_dict = {}
        param = self.tuning_param_dict.get('boosting')
        if isinstance(param, tuple):
            suggest_type = param[0]
            suggest_param = param[1]
            trial_param_dict['boosting'] = eval('trial.suggest_' + suggest_type)('boosting', **suggest_param)
        elif param is not None:
            trial_param_dict['boosting'] = param
        booster = trial_param_dict.get('boosting')

        if self.tuning_param_dict.get('fobj', None) is not None:
            fobj_class = self.tuning_param_dict.get('fobj')[0]
            fobj_trial_param_dict = {}
            for key, param in self.tuning_param_dict.get('fobj')[1].items():
                if isinstance(param, tuple):
                    suggest_type = param[0]
                    suggest_param = param[1]
                    fobj_trial_param_dict[key] = eval('trial.suggest_' + suggest_type)(key, **suggest_param)
                else:
                    fobj_trial_param_dict[key] = param
            fobj_instance = fobj_class(**fobj_trial_param_dict)
            fobj = fobj_instance.get_loss
            self.train_set.set_init_score(np.full_like(self.train_set.get_label(),
                                          fobj_instance.init_score(self.train_set.get_label()),
                                          dtype=float))

        for key, param in self.tuning_param_dict.items():
            if (key == 'boosting') or (key == 'fobj'):
                continue
            if (booster is None) or (booster != 'dart'):
                if key in ['uniform_drop', 'rate_drop', 'skip_drop']:
                    continue
            if isinstance(param, tuple):
                suggest_type = param[0]
                suggest_param = param[1]
                trial_param_dict[key] = eval('trial.suggest_' + suggest_type)(key, **suggest_param)
            else:
                trial_param_dict[key] = param

        callbacks = [callback_class(**callback_param) for callback_class, callback_param in self.callbacks]
        callbacks.append(LightGBMPruningCallback(trial, 'valid '+self.monitor))
        # when folds is defined in xgb.cv, nfold will not be used
        if self.tuning_param_dict.get('fobj', None) is None:
            cvresult = lgb.cv(params=trial_param_dict,
                              train_set=self.train_set,
                              eval_train_metric=True,
                              callbacks=callbacks,
                              **self.kwargs
                              )
        else:
            cvresult = lgb.cv(params=trial_param_dict,
                              train_set=self.train_set,
                              fobj=fobj,
                              eval_train_metric=True,
                              callbacks=callbacks,
                              **self.kwargs
                              )

        n_iterations = len(cvresult['valid ' + self.monitor + '-mean'])
        trial.set_user_attr("n_iterations", n_iterations)

        val_score = cvresult['valid ' + self.monitor + '-mean'][-1]
        train_score = cvresult['train ' + self.monitor + '-mean'][-1]
        trial.set_user_attr("val_score_{}".format(self.monitor), val_score)
        trial.set_user_attr("train_score_{}".format(self.monitor), train_score)
        if self.coef_train_val_disparity > 0:
            best_score = train_val_score(train_score, val_score, self.coef_train_val_disparity)
        else:
            best_score = val_score
        return best_score