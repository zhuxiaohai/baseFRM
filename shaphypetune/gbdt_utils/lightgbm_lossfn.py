import numpy as np
from scipy import special, optimize


class FocalLoss:
    def __init__(self, gamma, alpha=None):
        # 使用FocalLoss只需要设定以上两个参数,如果alpha=None,默认取值为1
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        # alpha 参数, 根据FL的定义函数,正样本权重为self.alpha,负样本权重为1 - self.alpha
        if (self.alpha is None) or (self.alpha == 1):
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        # pt和p的关系
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        # 即FL的计算公式
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        # 一阶导数
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        # 二阶导数
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        # 样本初始值寻找过程
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def get_loss(self, preds, train_data):
        # preds: The predicted values. Predicted values are returned before any transformation,
        # e.g. they are raw margin instead of probability of positive class for binary task.
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def eval_focal_loss(self, preds, train_data):
        # preds: If custom fobj is specified, predicted values are returned before any transformation,
        # e.g. they are raw margin instead of probability of positive class for binary task in this case.
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better