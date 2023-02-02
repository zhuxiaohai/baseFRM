import optuna
from shaphypetune.optunahypetune.optunasearch import OptunaSearchBase
from .xgboost_objective import ObjectiveXGB


class OptunaSearchXGB(OptunaSearchBase):
    def __init__(self, monitor, *args, **kwargs):
        self.monitor = monitor
        super().__init__(*args, **kwargs)

    def get_params(self):
        """
        how to use the best params returned by this function:
        train_param = instance.get_params()
        model = xgb.train(train_param, train_dmatrix, num_boost_round=train_param['n_iterations'])
        test_probability_1d_array = model.predict(test_dmatrix)
        """
        if self.study:
            best_trial = self.study.best_trial
            best_param = best_trial.params

            output_param = {key: best_param[key] for key in best_param.keys() if key in self.dynamic_params}
            output_param.update(self.static_params)
            return output_param, best_trial.user_attrs
        else:
            return None

    def search(self, params, train_set, **kwargs):
        for key, param in params.items():
            if not isinstance(param, tuple):
                self.static_params[key] = param
            else:
                self.dynamic_params[key] = param

        objective = ObjectiveXGB(params, train_set, self.monitor, self.coef_train_val_disparity,
                                 True if self.optimization_direction == 'maximize' else False, **kwargs)
        if self.optuna_verbosity == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        # prune a step(a boosting round) if it's worse than the bottom (1 - percentile) in history
        pruner = optuna.pruners.PercentilePruner(percentile=self.pruning_percentile,
                                                 n_warmup_steps=self.n_warmup_steps,
                                                 interval_steps=self.interval_steps,
                                                 n_startup_trials=self.n_startup_trials)
        study = optuna.create_study(direction=self.optimization_direction, sampler=sampler, pruner=pruner)
        study.optimize(objective, timeout=self.maximum_time, n_trials=self.n_trials, n_jobs=self.optuna_njobs)
        self.study = study
        print("Number of finished trials: ", len(study.trials))

