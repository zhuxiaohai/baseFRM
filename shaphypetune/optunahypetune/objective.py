from sklearn.base import clone


class Objective(object):
    def __init__(self, estimator, tuning_param_dict, coef_train_val_disparity,
                 train_kargs, val_kargs):
        self.estimator = estimator
        self.tuning_param_dict = tuning_param_dict
        self.coef_train_val_disparity = coef_train_val_disparity
        self.train_kargs = train_kargs
        self.val_kargs = val_kargs

    def train_val_score(self, train_score, val_score, w):
        output_scores = val_score - abs(train_score - val_score) * w
        return output_scores

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
        learner = clone(self.estimator, safe=False)
        for key, value in trial_param_dict.items():
            setattr(learner, key, value)
        # learner = self.estimator(**trial_param_dict)
        learner.fit(**self.train_kargs)
        val_score = learner.score(**self.val_kargs)
        trial.set_user_attr("val_score", val_score)
        if self.coef_train_val_disparity > 0:
            train_score = learner.score(**self.train_kargs)
            trial.set_user_attr("train_score", train_score)
            best_score = self.train_val_score(train_score, val_score, self.coef_train_val_disparity)
        else:
            best_score = val_score
        return best_score