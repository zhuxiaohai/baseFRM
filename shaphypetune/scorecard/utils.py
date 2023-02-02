import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def cal_ks(predict, target, sample_weight=None, plot=False):
    """
    ks经济学意义:
      将预测为坏账的概率从大到小排序，然后按从大到小依次选取一个概率值作为阈值，
      大于阈值的部分为预测为坏账的部分--记录其中真实为坏账的个数， 真实为好账的个数，
      上述记录值每次累加且除以总的坏账个数即累计坏账率，除以总好账个数为累计好账率, 累加结果存入列表
    sklearn.gbdt_utils.roc_curve（二分类标签，预测为正例的概率或得分）:
      将预测为正例（默认为1）的概率（0-1间）或得分（不限大小）从大到小排序, 然后按从大到小依次选取一个值作为阈值
      大于阈值的部分为预测为正例的部分--其中真实为正例的个数即TP, 真实为负例的个数即为FP
      上述值每次累加且除以总的正例个数为TPR, 除以总的负例个数为FPR，累加结果存入列表
    ks = max(累计坏账率list - 累计好账率list) = max(TPR_list - FPR_list)
    :param predict: list like, 可以为某个数值型特征字段，也可以是预测为坏账的概率的字段
    :param target: list like, 好坏账标签字段，字段中1为坏账
    :param plot: bool, 是否画图
    :return: ks, ks_thresh
    """
    # fpr即FPR_list, tpr即TPR_list, thresholds为上述所谓依次选取的阈值
    # thresholds一定是递减的，第一个值为max(预测为正例的概率或得分)+1
    fpr, tpr, thresholds = roc_curve(target, predict, sample_weight=sample_weight)
    ks = (tpr-fpr).max()
    ks_index = np.argmax(tpr-fpr)
    ks_thresh = thresholds[ks_index]
    if plot:
        # 绘制曲线
        plt.plot(tpr, label='bad_cum', linewidth=2)
        plt.plot(fpr, label='good_cum', linewidth=2)
        plt.plot(tpr-fpr, label='ks_curve', linewidth=2)
        # 标记ks点
        x_point = (ks_index, ks_index)
        y_point = (fpr[ks_index], tpr[ks_index])
        plt.plot(x_point, y_point, label='ks {:.2f}@{:.2f}'.format(ks, ks_thresh),
                 color='r', marker='o', markerfacecolor='r',
                 markersize=5)
        plt.scatter(x_point, y_point, color='r')
        # 绘制x轴（阈值）, thresholds第一个值为max(预测为正例的概率或得分)+1, 因此不画出来
        effective_indices_num = thresholds[1:].shape[0]
        if effective_indices_num > 5:
            # 向下取整
            increment = int(effective_indices_num / 5)
        else:
            increment = 1
        indices = range(1, thresholds.shape[0], increment)
        plt.xticks(indices, [round(i, 2) for i in thresholds[indices]])
        plt.xlabel('thresholds')
        plt.legend()
        plt.show()
    return ks, ks_thresh


def cal_psi_score(actual_array, expected_array,
                  bins=10, quantile=True, detail=False):
    """
    :param actual_array: np.array
    :param expected_array: np.array
    :param bins: int, number_of_bins you want for calculating psi
    :param quantile: bool
    :param detail: bool, if True, print the process of calculation
    """
    # 异常处理，所有取值都相同时, 说明该变量是常量, 返回None
    if np.min(expected_array) == np.max(expected_array):
        return None
    expected_array = pd.Series(expected_array).dropna()
    actual_array = pd.Series(actual_array).dropna()

    """step1: 确定分箱间隔"""
    def scale_range(input_array, scaled_min, scaled_max):
        """
        功能: 对input_array线性放缩至[scaled_min, scaled_max]
        :param input_array: numpy array of original values, 需放缩的原始数列
        :param scaled_min: float, 放缩后的最小值
        :param scaled_max: float, 放缩后的最大值
        :return input_array: numpy array of original values, 放缩后的数列
        """
        input_array += -np.min(input_array) # 此时最小值放缩到0
        if scaled_max == scaled_min:
            raise Exception('放缩后的数列scaled_min = scaled_min, 值为{}, '
                            '请检查expected_array数值！'.format(scaled_max))
        scaled_slope = np.max(input_array) * 1.0 / (scaled_max - scaled_min)
        input_array /= scaled_slope
        input_array += scaled_min
        return input_array

    breakpoints = np.arange(0, bins + 1) / bins * 100  # 等距分箱百分比
    if not quantile:
        # 等距分箱
        breakpoints = scale_range(breakpoints,
                                  np.min(expected_array),
                                  np.max(expected_array))
    else:
        # 等频分箱
        breakpoints = np.stack([np.percentile(expected_array, b)
                                for b in breakpoints])

    """step2: 统计区间内样本占比"""
    def generate_counts(arr, breakpoints):
        """
        功能: Generates counts for each bucket by using the bucket values
        :param arr: ndarray of actual values
        :param breakpoints: list of bucket values
        :return cnt_array: counts for elements in each bucket,
                           length of breakpoints array minus one
        :return score_range_array: 分箱区间
        """
        def count_in_range(input_arr, low, high, start):
            """
            功能: 统计给定区间内的样本数(Counts elements in array between
                 low and high values)
            :param input_arr: ndarray of actual values
            :param low: float, 左边界
            :param high: float, 右边界
            :param start: bool, 取值为Ture时，区间闭合方式[low, high],否则为(low, high]
            :return cnt_in_range: int, 给定区间内的样本数
            """
            if start:
                cnt_in_range = len(np.where(np.logical_and(input_arr >= low,
                                                           input_arr <= high))[0])
            else:
                cnt_in_range = len(np.where(np.logical_and(input_arr > low,
                                                           input_arr <= high))[0])
            return cnt_in_range
        cnt_array = np.zeros(len(breakpoints) - 1)
        range_array = [''] * (len(breakpoints) - 1)
        for i in range(1, len(breakpoints)):
            cnt_array[i - 1] = count_in_range(arr,
                                              breakpoints[i - 1],
                                              breakpoints[i], i == 1)
            if 1 == i:
                range_array[i - 1] = '[' + \
                                     str(round(breakpoints[i - 1], 4)) \
                                     + ',' + str(round(breakpoints[i], 4)) \
                                     + ']'
            else:
                range_array[i - 1] = '(' + \
                                     str(round(breakpoints[i - 1], 4)) \
                                     + ',' + str(round(breakpoints[i], 4)) \
                                     + ']'

        return cnt_array, range_array

    expected_cnt, score_range_array = generate_counts(expected_array,
                                                      breakpoints)
    expected_percents = expected_cnt / len(expected_array)
    actual_cnt = generate_counts(actual_array, breakpoints)[0]
    actual_percents = actual_cnt / len(actual_array)
    delta_percents = actual_percents - expected_percents
    score_range_array = generate_counts(expected_array, breakpoints)[1]

    """step3: 得到最终稳定性指标"""
    def sub_psi(e_perc, a_perc):
        """
        功能: 计算单个分箱内的psi值
        :param e_perc: float, 期望占比
        :param a_perc: float, 实际占比
        :return value: float, 单个分箱内的psi值
        """
        if a_perc == 0: # 实际占比
            a_perc = 0.001
        if e_perc == 0: # 期望占比
            e_perc = 0.001
        value = (e_perc - a_perc) * np.log(e_perc * 1.0 / a_perc)
        return value
    sub_psi_array = [sub_psi(expected_percents[i], actual_percents[i])
                     for i in range(0, len(expected_percents))]
    if detail:
        psi_value = pd.DataFrame()
        psi_value['score_range'] = score_range_array
        psi_value['expecteds'] = expected_cnt
        psi_value['expected(%)'] = expected_percents * 100
        psi_value['actucals'] = actual_cnt
        psi_value['actucal(%)'] = actual_percents * 100
        psi_value['ac - ex(%)'] = delta_percents * 100
        psi_value['actucal(%)'] = psi_value['actucal(%)'].apply(
            lambda x: round(x, 2))
        psi_value['ac - ex(%)'] = psi_value['ac - ex(%)'].apply(
            lambda x: round(x, 2))
        psi_value['ln(ac/ex)'] = psi_value.apply(
            lambda row: np.log((row['actucal(%)'] + 0.001)
                               / (row['expected(%)'] + 0.001)), axis=1)
        psi_value['psi'] = sub_psi_array
        flag = lambda x: '<<<<<<<' if x == psi_value.psi.max() else ''
        psi_value['max'] = psi_value.psi.apply(flag)
        psi_value = psi_value.append([{'score_range': '>>> summary',
                                       'expecteds': sum(expected_cnt),
                                       'expected(%)': 100,
                                       'actucals': sum(actual_cnt),
                                       'actucal(%)': 100,
                                       'ac - ex(%)': np.nan,
                                       'ln(ac/ex)': np.nan,
                                       'psi': np.sum(sub_psi_array),
                                       'max': '<<< result'}], ignore_index=True)
    else:
        psi_value = np.sum(sub_psi_array)
    return psi_value


def bin_bad_rate(df, col, target):
    """
    :param df: dataframe
    :param col: string, 需要计算好坏率的变量名
    :param target:string, 目标变量的字段名
    :return:dataframe[col, bin_total, bin_bad, bin_good, bin_bad_rate]
            按照col取值去重后从小到大排列， index一定是从0开始逐渐编号
    """
    # 按col的值去重后从小到大排列，并作为index
    total = df.groupby([col])[target].count()
    bad = df.groupby([col])[target].sum()
    regroup = pd.merge(pd.DataFrame({'bin_total': total}),
                       pd.DataFrame({'bin_bad': bad}),
                       left_index=True, right_index=True, how='left')
    # 默认drop=false，原来的index，即col的值去重后从小到大排列的值，另外生成一列
    regroup = regroup.reset_index()
    # 计算根据col分组后每组的违约率, 对于无顺序型数据，需要计算好坏比来代替原来离散的数值
    regroup['bin_good'] = regroup['bin_total'] - regroup['bin_bad']
    regroup['bin_bad_rate'] = regroup['bin_bad'] / regroup['bin_total']
    return regroup


def compute_woe_iv(bin_df, col_good, col_bad, col_total):
    """
    param bin_df:DataFrame|分箱表, 按照bin_range,
        或col_values取值去重后从小到大排列
    param col_good:str|feature名称，统计每个分箱的好样本个数
    param col_bad:str|feature名称，统计每个分箱的坏样本个数
    param col_total:str|feature名称，统计每个分箱一共有多少样本
    return: 原dataframe增添多列
    """
    d2 = bin_df.copy()
    total = d2[col_total].sum()
    bad = d2[col_bad].sum()
    good = total - bad

    d2['bin_good_rate'] = d2[col_good] / d2[col_total]
    d2['badattr'] = d2[col_bad] / bad
    d2['goodattr'] = d2[col_good] / good
    sum_bad = 0
    acc_bad_rate_list = []
    for i in d2[col_bad]:
        sum_bad += i
        acc_bad_rate_list.append(sum_bad / bad)
    d2['acc_bad_rate'] = acc_bad_rate_list
    sum_good = 0
    acc_good_rate_list = []
    for i in d2[col_good]:
        sum_good += i
        acc_good_rate_list.append(sum_good / good)
    d2['acc_good_rate'] = acc_good_rate_list
    d2['bin_total_rate'] = d2[col_total] / total
    d2['total_bad_rate'] = bad / total

    def compute_woe(badattr, goodattr):
        return np.log(badattr / (goodattr + 1e-6) + 1e-6)
    # 如果有全bad的分箱，则该分箱woe=inf，表示该分箱的坏账率远大于整体坏账率
    # 如果有全good的分箱，则woe=-inf，表示该分箱坏账率远小于整体坏账率
    # 如果分箱的坏账率和整体一致，则woe=0
    # 因此woe取值范围是(-inf---inf)
    d2['bin_woe'] = d2.apply(
        lambda x: compute_woe(x['badattr'], x['goodattr']),
        axis=1)
    # 如果有全bad或全good的分箱，则iv=inf，表示该分箱的坏账率水平和整体相比差异大
    #   该分箱预测能力极高
    # 如果分箱坏账率和整体一致，则iv=0，表示该分箱没有预测能力
    # 因此iv取值范围是[0----inf)
    d2['bin_iv'] = d2.apply(
        lambda x: (x['badattr'] - x['goodattr']) * x['bin_woe'],
        axis=1)
    # 变量的总iv
    d2['iv'] = d2['bin_iv'].sum()

    d2.drop(['badattr', 'goodattr'], axis=1, inplace=True)

    return d2


def unique_to_toad(all_df, features_unique=[]):
    rules = {}
    for name in features_unique:
        new_name = name + '_unique'
        all_df[new_name] = all_df[name].astype(str)
        if all_df[name].dtype != 'object':
            rule = np.sort(all_df[new_name].unique().astype(float)).astype(str)
        else:
            rule = np.array(all_df[new_name].value_counts().sort_values(ascending=False).index)
        if 'nan' in rule:
            rule = np.concatenate((rule[rule != 'nan'], np.array(['nan'])))
        rules[new_name] = rule[..., np.newaxis]
    return rules, all_df


def scorecardpy_to_toad(bins, columns_dict):
    """
    import scorecardpy as sc
    bins=sc.woebin(train_all_df[['loan_app_cnt', 'istrans']],y='istrans',x=None
                  , special_values=[-999]
                  , count_distr_limit=0.05
                  , stop_limit=0.05
                  , max_num_bin=5
                  , positive='bad|1'
                  , no_cores=None
                  , replace_blank=True
                  , print_step=1
                  , method='chimerge'   #chimerge  tree
                )
    """
    rules = {}
    for name, dtype in columns_dict.items():
        bin_df = bins[name]
        bin_df = pd.concat([bin_df[bin_df['is_special_values'] != True], bin_df[bin_df['is_special_values'] == True]], axis=0)
        if dtype == 'object':
            breaks = [i.split('%,%') for i in bin_df['breaks'].tolist()]
        else:
            breaks = []
            for value in bin_df['breaks']:
                if value in ['missing', 'nan']:
                    breaks.append(np.nan)
                elif value == 'inf':
                    pass
                else:
                    breaks.append(float(value))
            breaks = np.sort(breaks).tolist()
        rules[name] = breaks
    return rules


def optbinning_to_toad(binning_process, columns):
    """
    from optbinning import BinningProcess
    binning_process = BinningProcess(['app_channel'], max_n_bins=5, max_pvalue=0.1, special_codes=['dfdf'])
    binning_process.fit(train_all_df[['app_channel']], train_all_df['istrans'])
    """
    binning_dict = {}
    for col in columns:
        optb = binning_process.get_binned_variable(col)
        binning_table = optb.binning_table.build()
        if optb.dtype == 'numerical':
            splits = optb.binning_table.splits
            if binning_table.set_index('Bin').loc['Missing', 'Count'] > 0:
                splits = np.append(splits, np.nan)
            binning_dict[col] = splits
        else:
            splits = []
            for i in binning_table['Bin'].iloc[:-1]:
                if (str(i) != 'Special') & (str(i) != 'Missing'):
                    splits.append(i.tolist())
            missing = binning_table[binning_table['Bin'].astype(str)=='Missing']['Count'].iloc[0]
            if missing > 0:
                splits.append(['nan'])
            binning_dict[col] = splits
    return binning_dict
