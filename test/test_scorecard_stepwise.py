import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from shaphypetune.scorecard.searchstepwise import select_variables_stepwise


def test_scorecard_stepwise():
    x_train, y_train = load_breast_cancer(return_X_y=True)
    x_train = pd.DataFrame(x_train, columns=['col{}'.format(i) for i in range(x_train.shape[1])])
    included = select_variables_stepwise(x_train, y_train, check_positive_coef=False)
    model = Logit(y_train, add_constant(x_train[included])).fit()
    prob = model.predict(add_constant(x_train[included]))
    logit = x_train[included].dot(model.params.iloc[1:]) + model.params.iloc[0]
    prob2 = 1 / (1 + np.exp(-logit))
    np.testing.assert_almost_equal(prob2.values, prob.values)