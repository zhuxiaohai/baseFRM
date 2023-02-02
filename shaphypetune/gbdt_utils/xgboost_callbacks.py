import optuna
with optuna._imports.try_import() as _imports:
    import xgboost as xgb
from xgboost.callback import TrainingCallback
from xgboost.core import XGBoostError, Booster
from xgboost import rabit
from shaphypetune.optunahypetune.utils import train_val_score

import warnings
warnings.filterwarnings('ignore')


class EvaluationMonitor(TrainingCallback):
    '''Print the evaluation result at each iteration.

    .. versionadded:: 1.3.0

    Parameters
    ----------
    rank : int
        Which worker should be used for printing the result.
    period : int
        How many epoches between printing.
    w: float
        coefficient to balance between train and eval
    show_stdv : bool
        Used in cv to show standard deviation.  Users should not specify it.
    '''
    def __init__(self, rank=0, period=1, w=0.2, show_stdv=False):
        self.printer_rank = rank
        self.show_stdv = show_stdv
        self.period = period
        self.w = w
        self.metric = None
        assert period > 0
        # last error message, useful when early stopping and period are used together.
        self._latest = None
        super().__init__()

    def _fmt_metric(self, data, metric, score, std):
        if std is not None and self.show_stdv:
            msg = '\t{0}:{1:.5f}+{2:.5f}'.format(data + '-' + metric, score, std)
        else:
            msg = '\t{0}:{1:.5f}'.format(data + '-' + metric, score)
        return msg

    def after_iteration(self, model, epoch, evals_log):
        if not evals_log:
            return False
        warn = 'Must have 2 datasets for early stopping, the first for train and the other for eval.'
        assert len(evals_log.keys()) == 2, warn

        if not self.metric:
            data_name = list(evals_log.keys())[-1]
            data_log = evals_log[data_name]
            self.metric = list(data_log.keys())[-1]

        msg = '[{}]'.format(epoch)
        if rabit.get_rank() == self.printer_rank:
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    if isinstance(log[-1], tuple):
                        score = log[-1][0]
                        stdv = log[-1][1]
                    else:
                        score = log[-1]
                        stdv = None
                    msg += self._fmt_metric(data, metric_name, score, stdv)
            data_names = list(evals_log.keys())
            if isinstance(evals_log[data_names[0]][self.metric][-1], tuple):
                train_score, _ = evals_log[data_names[0]][self.metric][-1]
                val_score, _ = evals_log[data_names[1]][self.metric][-1]
            else:
                train_score = evals_log[data_names[0]][self.metric][-1]
                val_score = evals_log[data_names[1]][self.metric][-1]
            score = train_val_score(train_score, val_score, self.w)
            msg += '\t total-{0}: {1:.5f}'.format(self.metric, score)
            msg += '\n'

            if (epoch % self.period) == 0 or self.period == 1:
                rabit.tracker_print(msg)
                self._latest = None
            else:
                # There is skipped message
                self._latest = msg
        return False

    def after_training(self, model):
        if rabit.get_rank() == self.printer_rank and self._latest is not None:
            rabit.tracker_print(self._latest)
        return model


class EarlyStopping(TrainingCallback):
    ''' Callback function for xgb cv early stopping
    Parameters
    ----------
    rounds : int
        Early stopping rounds.
    w ： float
        Coefficient to balance the performance between train and val
    maximize : bool
        Whether to maximize evaluation metric.  None means auto (discouraged).
    mode: str
        'cv' or 'train'
    save_best : bool
        Whether training should return the best model or the last model.
    '''

    def __init__(self, rounds, w=0.2, maximize=None, mode='cv', save_best=False):
        self.metric = None
        self.rounds = rounds
        self.w = w
        self.save_best = save_best if mode == 'train' else False
        if maximize:
            self.improve_op = lambda x, y: x > y
        else:
            self.improve_op = lambda x, y: x < y
        self.stopping_history = {}
        self.current_rounds = 0
        self.best_scores = {}
        super().__init__()

    def _update_rounds(self, score, model: Booster, epoch):
        if not self.stopping_history:
            # First round
            self.current_rounds = 0
            self.stopping_history = [score]
            self.best_scores = [score]
            model.set_attr(best_score=str(score), best_iteration=str(epoch))
        elif not self.improve_op(score, self.best_scores[-1]):
            # Not improved
            self.stopping_history.append(score)
            self.current_rounds += 1
        else:
            # Improved
            self.stopping_history.append(score)
            self.best_scores.append(score)
            record = self.stopping_history[-1]
            model.set_attr(best_score=str(record), best_iteration=str(epoch))
            self.current_rounds = 0  # reset

        if self.current_rounds >= self.rounds:
            # Should stop
            return True
        return False

    def after_iteration(self, model: Booster, epoch, evals_log):
        msg = 'Must have 2 datasets for early stopping, the first for train and the other for eval.'
        assert len(evals_log.keys()) == 2, msg

        if not self.metric:
            data_name = list(evals_log.keys())[-1]
            data_log = evals_log[data_name]
            self.metric = list(data_log.keys())[-1]

        data_names = list(evals_log.keys())
        if isinstance(evals_log[data_names[0]][self.metric][-1], tuple):
            train_score, _ = evals_log[data_names[0]][self.metric][-1]
            val_score, _ = evals_log[data_names[1]][self.metric][-1]
        else:
            train_score = evals_log[data_names[0]][self.metric][-1]
            val_score = evals_log[data_names[1]][self.metric][-1]
        score = train_val_score(train_score, val_score, self.w)
        return self._update_rounds(score, model, epoch)

    def after_training(self, model: Booster):
        try:
            if self.save_best:
                model = model[: int(model.attr('best_iteration')) + 1]
        except XGBoostError as e:
            raise XGBoostError('`save_best` is not applicable to current booster') from e
        return model


class XGBoostPruningCallback(TrainingCallback):
    def __init__(self, trial: optuna.trial.Trial) -> None:
        _imports.check()
        self._trial = trial
        self.metric = None
        super().__init__()

    def after_iteration(self, model, epoch, evals_log):
        msg = 'Must have 2 datasets for pruning, the first for train and the other for eval.'
        assert len(evals_log.keys()) == 2, msg

        if not self.metric:
            data_name = list(evals_log.keys())[-1]
            data_log = evals_log[data_name]
            self.metric = list(data_log.keys())[-1]

        data_names = list(evals_log.keys())
        if isinstance(evals_log[data_names[0]][self.metric][-1], tuple):
            val_score, _ = evals_log[data_names[1]][self.metric][-1]
        else:
            val_score = evals_log[data_names[1]][self.metric][-1]
        current_score = val_score
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at iteration {}.".format(epoch)
            raise optuna.TrialPruned(message)