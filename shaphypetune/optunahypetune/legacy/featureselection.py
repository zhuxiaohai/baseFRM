# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import random
import warnings
from hyperopt import hp, fmin, tpe
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

C_SET = ['gray', 'green', 'yellow', 'blue', 'black', 'red', 'gold', 'indigo', 'aqua', 'khaki']


def ks_score(preds, trainDmatrix):
    from scipy.stats import ks_2samp
    y_true = trainDmatrix.get_label()
    return 'ks', -ks_2samp(preds[np.array(y_true) == 1], preds[np.array(y_true) != 1]).statistic


class tuning_llb():
    def __init__(self, train_set, valid_set, x_input, output_file='result.txt', use_x_cnt=50, if_plot=True,
                 max_iters=100, mode='ks_list', init_kwargs=None):
        self.train_set = train_set
        self.valid_set = valid_set
        self.x_input = x_input
        self.output_file = output_file
        self.use_x_cnt = use_x_cnt
        self.if_plot = if_plot
        #        self.iter_rate=iter_rate
        self.max_iters = max_iters
        self.mode = mode
        self.opt_trace = []
        self.init_kwargs = init_kwargs

    #    def __xgb_test(self, tmp_params, tmp_input,feval=None):
    def __xgb_test(self, tmp_params, tmp_input, **kwargs):
        clf = xgb.XGBClassifier(n_estimators=200,
                                max_depth=tmp_params[0],
                                subsample=tmp_params[1],
                                min_child_weight=tmp_params[2],
                                base_score=tmp_params[3],
                                reg_lambda=tmp_params[4],
                                learning_rate=tmp_params[5],
                                colsample_bytree=tmp_params[6],
                                reg_alpha=tmp_params[7],
                                nthread=8,
                                random_state=tmp_params[8],
                                missing=self.init_kwargs.get('missing'))

        if "eval_metric" in kwargs.keys():
            if not callable(kwargs['eval_metric']):
                kwargs['eval_metric'] = ks_score
        else:
            kwargs['eval_metric'] = ks_score
        print("__xgb_test kwargs=", kwargs)
        clf.fit(self.train_set[0][tmp_input].values
                , self.train_set[1].values
                #                ,verbose=True
                #                ,early_stopping_rounds=30
                #                ,eval_metric=feval
                , eval_set=[(k[0][tmp_input].values, k[1].values) for k in self.valid_set]
                , **kwargs)
        ks_t = [-clf.evals_result()['validation_' + str(len_eval)]['ks'][clf.best_iteration] for len_eval in
                range(self.valid_len)]
        #        tmpre={'params':tmp_params,'result':[clf.best_iteration]+ks_t,'x':','.join(sorted(tmp_input))}
        tmpre = {'params': tmp_params, 'result': [clf.best_iteration] + ks_t, 'x': [item for item in tmp_input]}

        fea_imp = pd.Series(clf.feature_importances_, index=tmp_input)
        print("clf.best_score", clf.best_score)
        return ks_t, tmpre, fea_imp

    def __imp_iter(self, x):
        if self.mode == 'ks_list':
            f = lambda x: sum(x) / len(x) if len(x) > 0 else 0.5
            return f(x)
        else:
            f = lambda a, b: 1 if len(a) == 0 else 1.1 if a[-1] > np.percentile(b, 30) else 0.9 if a[
                                                                                                       -1] < np.percentile(
                b, 30) else 1
            return f(x, self.ks_list)

    #    def start(self,feval=None,verbose=True,early_stopping_rounds=30):
    def start(self, **kwargs):
        print('init.............................')

        self.valid_len = len(self.valid_set)
        f = open(self.output_file, 'w')
        f.writelines(
            'params|best_ntree|' + str('|'.join(['ks_valid_' + str(i) for i in range(self.valid_len)])) + '|x_list\n')
        f.close()
        ax = []
        if self.if_plot:
            import matplotlib.pyplot as plt
            c_set = C_SET

            plt.scatter(0, 0)
            plt.show()
            plt.ion()
            k = 0
            for i in range(self.valid_len):
                ax.append([])
        else:
            for i in range(self.valid_len):
                ax.append([])

        k = 0
        params = [[2, 3, 4], [0.7], [20, 30, 10], [0.5, 0.7, 0.6, 0.4], [1, 5, 10, 15], [0.1, 0.08, 0.12],
                  [0.5, 0.7, 0.9], [1, 10], [i * 8 for i in range(1000)]]

        self.x_imp = pd.DataFrame(1, index=self.x_input, columns=['imp'])
        self.x_imp['ks_list'] = pd.Series([[] for i in range(len(self.x_input))], index=self.x_input)
        self.ks_list = []
        best_result = 0.0
        best_param = {}
        while k <= self.max_iters:
            k = k + 1
            self.ks_list.sort(reverse=True)

            if k <= 10:
                self.tmp_input = random.sample(self.x_input, self.use_x_cnt)
            else:
                if self.mode == 'ks_list':
                    tmp_x_imp = (np.random.rand(len(self.x_imp)) * self.x_imp.imp).sort_values(ascending=False)
                else:
                    tmp_x_imp = (np.random.rand(len(self.x_imp)) * self.x_imp.imp).sort_values(ascending=False)

                self.tmp_input = list((tmp_x_imp[:self.use_x_cnt].index))

                print(tmp_x_imp.head(5))

            tmp_params = [random.sample(params_i, 1)[0] for params_i in params]

            ks_t, tmpre, fea_imp = self.__xgb_test(tmp_params, self.tmp_input, **kwargs)
            self.opt_trace.append(tmpre)

            print(k, "tmpre['result']=", tmpre['result'])
            #            if tmpre['result'][0]==0:
            #                k=k-1
            #                continue
            self.ks_list.append(min(ks_t))
            print("params:" + str(tmp_params))

            for i in range(self.valid_len):
                ax[i].append(tmpre['result'][i + 1])
            for i in self.tmp_input:
                if self.mode == 'ks_list':
                    self.x_imp.loc[i, 'ks_list'].append((fea_imp[i] + 0.001) * min(ks_t))
                else:
                    self.x_imp.loc[i, 'ks_list'].append(min(ks_t))

            if self.mode == 'ks_list':
                self.x_imp.imp = self.x_imp.ks_list.apply(self.__imp_iter).sort_values(ascending=False)
            else:
                self.x_imp.imp.loc[self.tmp_input] = self.x_imp.imp.loc[self.tmp_input] * self.x_imp.ks_list.loc[
                    self.tmp_input].apply(self.__imp_iter)

            if best_result < sum(tmpre['result'][1:]):
                best_result = sum(tmpre['result'][1:])
                best_param = tmpre
            print('ks:' + str(tmpre['result']))
            f = open(self.output_file, 'a')
            f.writelines(str(tmpre['params']) + '|' + '|'.join([str(round(i, 4)) for i in tmpre['result']]) + '|' + str(
                self.tmp_input) + '\n')
            f.close()
            if self.if_plot:
                f.close()
                plt.cla()
                for i in range(self.valid_len):
                    plt.scatter(range(k), ax[i], c=c_set[i], label='valid_' + str(i))

                plt.legend(loc='upper right')
                plt.show()
                plt.pause(0.2)
        return best_result, best_param


class AmFeatSelXgbClf(xgb.XGBClassifier):
    """含有固定特征选择的调参Estimater

    Parameters
    ----------
    x_input : list
        调参变量池
    output_file : str
        调参日志输出文件，默认'result.txt'
    use_x_cnt:int
        入模自变量数量，默认50
    if_plot:bool
        是否画图，默认True
    max_iters:int
        迭代次数，默认100

    Attributes
    ----------
    best_result :float
        最佳效果，返回所有调参测试集效果之和
    best_param :dict
        最佳参数以及效果字典
    features_select_list:list
        最终入模特征
    opt_trace:list
        保存的是模型每一轮的调参日志，每次调参过程以字典形式记录，'param'为参数，
        result第一个值为模型轮数，后面的值为调参测试集的效果，'x'代表的是入选的模型特征
        eg：{'params': [3, 0.7, 20, 0.7, 5, 0.12, 0.9, 1, 6528],
             'result': [89, 0.275087, 0.310883],
             'x': ['cust_age_x^_19.8809,22.7693)',
              'device_score_ppx^_0.2083,0.2359)',
              'shfw_bj_tt_usetime_4^_-inf,-98.999)',
              'td_3m_financing_cnt',
              'kx_location_score^_-98.999,469.0)'
              ...}

    """

    def __init__(self, x_input, output_file='result.txt',
                 use_x_cnt=50, if_plot=True,
                 max_iters=100, mode='ks_list',
                 **kwargs):
        self.x_input = x_input
        self.output_file = output_file
        self.use_x_cnt = use_x_cnt
        self.if_plot = if_plot
        self.max_iters = max_iters
        self.mode = mode
        self.opt_trace = []
        self.init_kwargs = kwargs

        # super(AmFeatSelXgbClf, self).__init__(max_depth, learning_rate,
        #          n_estimators, silent,
        #          objective,nthread, gamma, min_child_weight,
        #          max_delta_step, subsample,
        #          colsample_bytree, colsample_bylevel,
        #          reg_alpha, reg_lambda, scale_pos_weight,
        #          base_score, random_state, missing)

        super(AmFeatSelXgbClf, self).__init__(**kwargs)

    #        print ("missing",self.missing)
    #        print ("get_xgb_params",self.get_xgb_params)

    def fit(self, data, y=None, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True, **kwargs):
        tl = tuning_llb([data, y], eval_set, self.x_input, self.output_file,
                        self.use_x_cnt, self.if_plot, self.max_iters, self.mode, init_kwargs=self.init_kwargs)
        # print ("sample_weight_eval_set=",kwargs.keys())
        self.best_result, self.best_param = tl.start(eval_metric=eval_metric
                                                     , verbose=verbose
                                                     , early_stopping_rounds=early_stopping_rounds
                                                     , sample_weight=sample_weight
                                                     , sample_weight_eval_set=kwargs.get('sample_weight_eval_set'),
                                                     )

        self.n_estimators = self.best_param['result'][0] + 1
        self.max_depth = self.best_param['params'][0]
        self.subsample = self.best_param['params'][1]
        self.min_child_weight = self.best_param['params'][2]
        self.base_score = self.best_param['params'][3]
        self.reg_lambda = self.best_param['params'][4]
        self.learning_rate = self.best_param['params'][5]
        self.colsample_bytree = self.best_param['params'][6]
        self.reg_alpha = self.best_param['params'][7]
        self.random_state = self.best_param['params'][8]
        self.features_select_list = self.best_param['x']
        self.opt_trace = tl.opt_trace

        #        print ("features_select_list",self.features_select_list)
        #        print ("missing3",self.missing)
        #        print ("objective3",self.objective)
        #        print ("get_xgb_params3",self.get_xgb_params)
        super(AmFeatSelXgbClf, self).fit(data[self.features_select_list], y
                                         , eval_set=[(i[0][self.features_select_list], i[1]) for i in eval_set]
                                         , verbose=verbose
                                         , sample_weight=sample_weight
                                         , eval_metric=eval_metric
                                         , sample_weight_eval_set=kwargs.get('sample_weight_eval_set'))

        from scipy.stats import ks_2samp

        preds = self.predict_proba(eval_set[0][0][self.features_select_list])[:, 1]
        print("score_ks_test", ks_2samp(preds[np.array(eval_set[0][1]) == 1],
                                        preds[np.array(eval_set[0][1]) != 1]).statistic)

        # print(" self.feature_importances_",self.feature_importances_)
        return self


