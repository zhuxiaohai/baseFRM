import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit,  StratifiedKFold


def train_val_score(train_score, val_score, w):
    output_scores = val_score - abs(train_score - val_score) * w
    return output_scores


def make_train_val(x_train, y_train, eval_set, cv, random_state):
    """
    :param x_train: array-like(dim=2), Feature matrix
    :param y_train: array-like(dim=1), Label
    :param eval_set: a list of (X, y) tuples, only one tuple is supported as of now
                     if this is set, cv will always be 1
    :param cv: int, number of folds for stratified cross_validation
    :param random_state: int
    """
    if eval_set is not None:
        print('using self-defined eval-set')
        assert len(eval_set) == 1
        cv = 1
        if type(x_train) is pd.core.frame.DataFrame:
            x_train_val = pd.concat([x_train, eval_set[0][0]], axis=0)
            y_train_val = pd.concat([y_train, eval_set[0][1]], axis=0)
        else:
            x_train_val = np.concatenate((x_train, eval_set[0][0]), axis=0)
            y_train_val = np.concatenate((y_train, eval_set[0][1]), axis=0)
        # initialize all indices to 0 except the section of training
        # to -1, which means this part will not be in validation.
        # So only one fold is made
        test_fold = np.zeros(x_train_val.shape[0])
        test_fold[:x_train.shape[0]] = -1
        ps = PredefinedSplit(test_fold=test_fold)
        folds = []
        for train_indices_array, val_indices_array in ps.split():
            folds.append((train_indices_array.tolist(),
                          val_indices_array.tolist()))
    else:
        print('using cv {}'.format(cv))
        x_train_val = x_train
        y_train_val = y_train
        # folds = None
        skf = StratifiedKFold(n_splits=cv, shuffle=True,
                              random_state=random_state)
        folds = []
        for train_indices_array, val_indices_array in skf.split(
                x_train_val, y_train_val):
            folds.append((train_indices_array.tolist(),
                          val_indices_array.tolist()))
    return x_train_val, y_train_val, cv, folds