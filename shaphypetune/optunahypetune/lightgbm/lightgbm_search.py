import optuna
from shaphypetune.optunahypetune.optunasearch import OptunaSearchBase
from .lightgbm_objective import ObjectiveLGB


class OptunaSearchLGB(OptunaSearchBase):
    def __init__(self, monitor, *args, **kwargs):
        self.monitor = monitor
        super().__init__(*args, **kwargs)

    def search(self, params, train_set, callbacks=[], **kwargs):
        for key, param in params.items():
            if key == 'fobj':
                for fobj_key, fobj_param in param[1].items():
                    if not isinstance(fobj_param, tuple):
                        self.fobj_static_params[fobj_key] = fobj_param
                    else:
                        self.fobj_dynamic_params[fobj_key] = fobj_param
            else:
                if not isinstance(param, tuple):
                    self.static_params[key] = param
                else:
                    self.dynamic_params[key] = param

        kwargs.pop('callbacks', None)
        kwargs.pop('eval_train_metric', None)
        objective = ObjectiveLGB(params, train_set, self.monitor, self.coef_train_val_disparity, callbacks, **kwargs)
        if self.optuna_verbosity == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        # prune a step(a boosting round) if it's worse than the bottom (1 - percentile) in history
        pruner = optuna.pruners.PercentilePruner(percentile=self.pruning_percentile,
                                                 n_warmup_steps=self.n_warmup_steps,
                                                 interval_steps=self.interval_steps,
                                                 n_startup_trials=self.n_startup_trials)
        study = optuna.create_study(direction=self.optimization_direction, sampler=sampler, pruner=pruner)
        study.optimize(objective, timeout=self.maximum_time, n_trials=self.n_trials, n_jobs=1)
        self.study = study
        print("Number of finished trials: ", len(study.trials))