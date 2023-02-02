from operator import gt, lt
from lightgbm.basic import _ConfigAliases, _log_info, _log_warning
from lightgbm.callback import EarlyStopException, _format_eval_result
from shaphypetune.optunahypetune.utils import train_val_score


def early_stopping(stopping_rounds, w=0.2, verbose=True, period=1, show_stdv=True):
    """Create a callback that activates early stopping for cv.

    Activates early stopping.
    The model will train until the validation score stops improving.
    Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.

    Parameters
    ----------
    stopping_rounds : int
       The possible number of rounds without the trend occurrence.
    w : float
       Coefficient to balance train and val scores
    verbose : bool, optional (default=True)
        Whether to print message with early stopping information.
    period : int, optional (default=1)
        The period to print the evaluation results.
    show_stdv : bool, optional (default=True)
        Whether to show stdv (if provided).

    Returns
    -------
    callback : function
        The callback that activates early stopping.
    """
    # always use the last metric for early stopping
    last_metric_only = True
    best_score = []
    best_iter = []
    best_score_list = []
    cmp_op = []
    enabled = [True]
    last_metric = ['']
    best_balance_score = [0.0]
    best_balance_iter = [0]
    best_balance_score_list = [None]

    def _init(env):
        enabled[0] = not any(env.params.get(boost_alias, "") == 'dart' for boost_alias
                             in _ConfigAliases.get("boosting"))
        if not enabled[0]:
            _log_warning('Early stopping is not available in dart mode')
            return
        if not env.evaluation_result_list:
            raise ValueError('For early stopping, '
                             'at least one dataset and eval metric is required for evaluation')
        if env.evaluation_result_list[0][0] != "cv_agg":
            raise ValueError('the early stopping is only customized for cv')

        if verbose:
            _log_info("Training until validation scores don't improve for {} rounds".format(stopping_rounds))

        # # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
        last_metric[0] = env.evaluation_result_list[-1][1].split(" ")[-1]
        best_balance_score[0] = float('-inf') if env.evaluation_result_list[-1][3] else float('inf')
        best_balance_iter[0] = 0
        best_balance_score_list[0] = None
        if w > 0:
            if env.evaluation_result_list[0][1].split(" ")[0] != 'train':
                raise ValueError('train data must be available to balance train and val')
        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            if eval_ret[3]:
                best_score.append(float('-inf'))
                cmp_op.append(gt)
            else:
                best_score.append(float('inf'))
                cmp_op.append(lt)

    def _final_iteration_check(env, eval_name_splitted, i):
        if env.iteration == env.end_iteration - 1:
            if verbose:
                _log_info('Did not meet early stopping. Best iteration is:\n[%d]\t%s' % (
                    best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                # if last_metric_only:
                #     _log_info("Evaluated only: {}".format(eval_name_splitted[-1]))
            raise EarlyStopException(best_iter[i], best_score_list[i])

    def _fetch_balance_train_score(env):
        for i in range(len(env.evaluation_result_list)):
            data_type, metric_name = env.evaluation_result_list[i][1].split(" ")
            if (data_type == 'train') and (metric_name == last_metric[0]):
                return env.evaluation_result_list[i][2]

    def _callback(env):
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        log_flag = verbose and env.evaluation_result_list and ((env.iteration + 1) % period == 0)
        if log_flag:
            log_result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
        # get the train score of the last metric before hand
        if w > 0:
            balance_train_score = _fetch_balance_train_score(env)
        for i in range(len(env.evaluation_result_list)):
            eval_name_splitted = env.evaluation_result_list[i][1].split(" ")
            score = env.evaluation_result_list[i][2]
            # record best score as of now for whatever metric and dataset
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
            if (eval_name_splitted[0] == 'train') or (eval_name_splitted[-1] != last_metric[0]):
                _final_iteration_check(env, eval_name_splitted, i)
                continue
            assert (eval_name_splitted[0] != 'train') and (eval_name_splitted[-1] == last_metric[0])
            # the codes below will be executed only when dataset is not train and metric is last_metric
            if w > 0:
                balance_score = train_val_score(balance_train_score, score, w)
                if log_flag:
                    log_result += '\t%s\'s %s: %g' % ('balance', eval_name_splitted[-1], balance_score)
            else:
                balance_score = score
            if cmp_op[i](balance_score, best_balance_score[0]):
                best_balance_score[0] = balance_score
                best_balance_iter[0] = env.iteration
                best_balance_score_list[0] = env.evaluation_result_list
            if env.iteration - best_balance_iter[0] >= stopping_rounds:
                if log_flag:
                    _log_info('[%d]\t%s' % (env.iteration + 1, log_result))
                if verbose:
                    _log_info('Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_balance_iter[0] + 1, '\t'.join([_format_eval_result(x) for x in best_balance_score_list[0]])))
                    if last_metric_only:
                        _log_info("Evaluated only: {}".format(eval_name_splitted[-1]))
                raise EarlyStopException(best_balance_iter[0], best_balance_score[0])
            _final_iteration_check(env, eval_name_splitted, i)
        if log_flag:
            _log_info('[%d]\t%s' % (env.iteration + 1, log_result))
    _callback.order = 30
    return _callback