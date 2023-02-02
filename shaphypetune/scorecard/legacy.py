import numpy as np
import pandas as pd
import itertools
import datetime
from .utils import compute_woe_iv


def woe_to_score(woe, weight, pdo=60, rate=2):
    """
    :param woe: array-like
    :param weight: float
    """
    factor = pdo / np.log(rate)
    s = -factor * weight * woe
    return s


def _psi(expected, actual, weight=None):
    """
    :param expected: array-like with values mapped to woe or bin number or bin_range
    :param actual: array-like with values mapped to woe or bin number or bin_range
    :param weight: float
    """
    expected_prop = pd.Series(expected).value_counts(normalize=True, dropna=False).sort_index()
    actual_prop = pd.Series(actual).value_counts(normalize=True, dropna=False).sort_index()
    frame = pd.DataFrame({
        'expected_prop': expected_prop,
        'actual_prop': actual_prop,
        'psi': (expected_prop - actual_prop) * np.log(expected_prop / actual_prop)
    })
    psi = frame['psi'].sum()
    frame['psi_total'] = psi
    frame.index.name = 'value'
    frame = frame.loc[expected_prop.index, :]
    if weight is not None:
        frame['score'] = woe_to_score(frame.index.tolist(), weight)
        frame['delta_prop'] = frame['actual'] - frame['expected']
        frame['csi'] = frame('score') * frame['delta_prop']
        frame['csi_total'] = frame['csi'].sum()
    return psi, frame.reset_index()


def cal_psi_bin(expected, actual, weight=None, return_frame=False):
    """calculate PSI
    Args:
        expected pd.DataFrame or pd.Series
            with values mapped to woe or bin number or bin_range
        actual pd.DataFrame or pd.Series
            with values mapped to woe or bin number or bin_range
        weight pd.Series or float
        return_frame (bool): if need to return frame of proportion
    Returns:
        float|Series
    """
    def unpack_tuple(x):
        if len(x) == 1:
            return x[0]
        else:
            return x
    psi = list()
    frame = list()

    if isinstance(expected, pd.DataFrame):
        for col in expected:
            if weight is not None:
                p, f = _psi(expected[col], actual[col], weight.loc[col])
            else:
                p, f = _psi(expected[col], actual[col], weight)
            psi.append(p)
            frame.append(f)

        psi = pd.Series(psi, index=expected.columns)

        frame = pd.concat(
            frame,
            keys=expected.columns,
            names=['columns', 'id'],
        ).reset_index()
        frame = frame.drop(columns='id')
    else:
        psi, frame = _psi(expected, actual, weight)

    res = (psi,)

    if return_frame:
        res += (frame,)

    return unpack_tuple(res)


def check_trend(var_series, time_series, interval):
    """
    :param var_series: pd.Series变量序列 or a list of pd_series
    :param time_series: pd.Series, 时间序列，当var_series为列表时不使用
    :param interval: int,间隔,以天为单位，当var_series为列表时不使用
    """
    if isinstance(var_series, list):
        series = var_series[0]
    else:
        series = var_series
    var_type = series.dtype
    if var_type != 'object':
        statistics_list = ['0.5q', 'mean', 'null_rate']
    else:
        statistics_list = []
        name_list = series.value_counts().iloc[:2].index.tolist()
        statistics_list.append('top1%: ' + str(name_list[0]))
        statistics_list.append('top2%: ' + str(name_list[1]))
        statistics_list.append('null_rate')

    def agg_func(input_series):
        num = input_series.shape[0]
        if num == 0:
            return [np.nan]*3
        else:
            if var_type != 'object':
                return [input_series.quantile(0.5),
                        input_series.mean(),
                        input_series.isnull().sum() / num]
            else:
                output_value_list = []
                output_value_list.append(
                    input_series[input_series == name_list[0]].shape[0] / num)
                output_value_list.append(
                    input_series[input_series == name_list[1]].shape[0] / num)
                output_value_list.append(
                    input_series.isnull().sum() / num)
                return output_value_list

    df_values = []
    if isinstance(var_series, list):
        for data_set in var_series:
            df_values.append(agg_func(data_set))
        output_df = pd.DataFrame(df_values, columns=statistics_list)
    else:
        time_series = pd.to_datetime(time_series)
        first = time_series.min()
        last = time_series.max()
        date_list = []
        for i in range(0, (last - first).days - interval, interval):
            left = first + datetime.timedelta(days=i)
            right = left + datetime.timedelta(days=interval)
            date_list.append('[' + left.strftime("%Y-%m-%d") + ' - '
                             + right.strftime("%Y-%m-%d") + ')')
            data_set = var_series[(time_series >= left) & (time_series < right)]
            df_values.append(agg_func(data_set))
        left = right
        right = last
        date_list.append('[' + left.strftime("%Y-%m-%d") + ' - '
                         + right.strftime("%Y-%m-%d") + ']')
        data_set = var_series[(time_series >= left) & (time_series <= right)]
        df_values.append(agg_func(data_set))
        output_df = pd.DataFrame(df_values, index=date_list,
                                 columns=statistics_list)
    return output_df


def check_statistics(df, var_list=None):
    """
    :param df: pd.dataframe
    :param var_list: list of string, 变量名列表
    """
    if var_list is None:
        var_list = df.columns.tolist()

    def agg_func(input_series):
        num = input_series.shape[0]
        if input_series.dtype != 'object':
            output_value_list = np.concatenate(
                (input_series.quantile([0.25, 0.5, 0.75]),
                 [input_series.max(), input_series.min(),
                  input_series.mean(), input_series.isnull().sum() / num]))
        else:
            output_series = input_series.value_counts() / num
            output_value_list = \
                [str(index) + ':' + str(value)
                 for index, value
                 in zip(output_series.iloc[:3].index, output_series.iloc[:3].values)]
            output_value_list += \
                [str(index) + ':' + str(value)
                 for index, value
                 in zip(output_series.iloc[-3:].index, output_series.iloc[-3:].values)]
            output_value_list.append(input_series.isnull().sum() / num)
        return pd.Series(output_value_list,
                         index=['0.25q or top1%', '0.5q or top2%', '0.75q or top3%',
                                'max or bottom3%', 'min or bottom 2%', 'mean or bottom1%',
                                'null_rate'])

    output_df = df.loc[:, var_list].apply(agg_func, axis=0)
    return output_df.T


def check_monotone(regroup, col, ascending):
    """
    :param regroup: dataframe, bin_df with bin_woe
    :param col: str, the column to be chekcked
    :param ascending: True for ascending, False for descending, None for any one
    """
    if regroup.shape[0] <= 2 and (ascending is None):
        return True
    my_list = regroup[col].tolist()

    # ascending
    flag1 = all(x < y for x, y in zip(my_list, my_list[1:]))
    # descending
    flag2 = all(x > y for x, y in zip(my_list, my_list[1:]))
    if ascending is not None:
        if ascending:
            return flag1
        else:
            return flag2
    else:
        return flag1 or flag2


def check_bin(regroup, col, continuous_flag, min_binpct=0.05, monoticity=True):
    """
    :param regroup: dataframe [col, bin_total, bin_bad, bin_good, bin_bad_rate]
       indices must be increasing from 0 with increment 1
    :param col: string, col name
    :param min_binpct: float
    :param monoticity: bool. If false, bad_rate could be of no order
    :return: dataframe [col, bin_total, bin_bad, bin_good, bin_bad_rate]
      indices is increasing from 0 with increment 1
    """
    bin_df = regroup.copy()
    # 检查是否有箱只有好样本或者只有坏样本
    min_bad_rate = bin_df['bin_bad_rate'].min()
    max_bad_rate = bin_df['bin_bad_rate'].max()
    while (min_bad_rate == 0.0 or max_bad_rate == 1.0) and bin_df.shape[0] > 2:
        # 违约率为1或0的箱体的index. index总是返回列表，因此用[0]取第一个元素
        bad_index = bin_df[bin_df['bin_bad_rate'].isin([0.0, 1.0])].index[0]
        if bad_index == (bin_df.shape[0] - 1):
            bin_df = combine_bin_df(bin_df, col, bad_index - 1, continuous_flag)
        elif bad_index == 0:
            bin_df = combine_bin_df(bin_df, col, 0, continuous_flag)
        else:
            # 计算bad_index和前面一个的箱体的卡方值
            chi_df = bin_df.loc[bad_index - 1:bad_index, :]
            chi1 = cal_chi2(chi_df, 'bin_bad', 'bin_good', 'bin_total')
            # 计算bad_index和后一个箱体之间的卡方值
            chi_df = bin_df.loc[bad_index:bad_index + 1, :]
            chi2 = cal_chi2(chi_df, 'bin_bad', 'bin_good', 'bin_total')
            if chi1 < chi2:
                # 当chi1<chi2时, 合并bad_index之前与bad_index对应的箱体
                bin_df = combine_bin_df(bin_df, col, bad_index - 1, continuous_flag)
            else:
                # 当chi1>=chi2时,合并bad_index和bad_index之后对应的箱体
                bin_df = combine_bin_df(bin_df, col, bad_index, continuous_flag)

        # 计算每个区间的违约率, 其index一定是从0开始从小到大编号
        min_bad_rate = bin_df['bin_bad_rate'].min()
        max_bad_rate = bin_df['bin_bad_rate'].max()
    if (min_bad_rate == 0.0 or max_bad_rate == 1.0) and bin_df.shape[0] <= 2:
        print('all good or bad, failure')
        return None

    # 检查分箱后的最小占比
    if min_binpct > 0.0:
        # 得出最小的区间占比
        min_pct = (bin_df['bin_total'] / bin_df['bin_total'].sum()).min()
        # 当最小的区间占比小于min_binpct且箱体的个数大于2
        while min_pct < min_binpct and bin_df.shape[0] > 2:
            # 下面的逻辑基本与“检验是否有箱体只有好/坏样本”的一致
            bad_index = \
                bin_df[(bin_df['bin_total']
                        / bin_df['bin_total'].sum()) == min_pct].index[0]
            if bad_index == (bin_df.shape[0] - 1):
                bin_df = combine_bin_df(bin_df, col, bad_index - 1, continuous_flag)
            elif bad_index == 0:
                bin_df = combine_bin_df(bin_df, col, 0, continuous_flag)
            else:
                chi_df = bin_df.loc[bad_index - 1:bad_index, :]
                chi1 = cal_chi2(chi_df, 'bin_bad', 'bin_good', 'bin_total')
                chi_df = bin_df.loc[bad_index:bad_index + 1, :]
                chi2 = cal_chi2(chi_df, 'bin_bad', 'bin_good', 'bin_total')
                if chi1 < chi2:
                    bin_df = combine_bin_df(bin_df, col, bad_index - 1, continuous_flag)
                else:
                    bin_df = combine_bin_df(bin_df, col, bad_index, continuous_flag)
            min_pct = (bin_df['bin_total'] / bin_df['bin_total'].sum()).min()
        if min_pct < min_binpct and bin_df.shape[0] <= 2:
            print('few data in a bin, failure')
            return None

    if monoticity:
        bin_df2 = compute_woe_iv(bin_df, 'bin_good', 'bin_bad', 'bin_total')
        monotone = check_monotone(bin_df2, 'bin_woe', None)
        if not monotone:
            print('woe not monotone, failure')
            return None
    return bin_df


def cal_chi2(df, bad_col, good_col, total_col):
    """
    :param df: 只有两个分箱的样本数据.
    :param bad_col: 列明string, 统计某个取值的坏样本数
    :param good_col: 列明string, 统计某个取值的好样本数
    :param total_col: 列明string, 统计某个取值的全部样本数
    :return: chi2 二联卡卡方值
    """
    all_bad_rate = df[bad_col].sum() * 1.0 / df[total_col].sum()
    all_good_rate = df[good_col].sum() * 1.0 / df[total_col].sum()
    # 当全部样本只有好或者坏样本时，卡方值为0
    if all_bad_rate in [0, 1]:
        return 0.0
    df2 = df.copy()
    # 计算每组的坏用户期望数量
    df2['bad_expected'] = df2[total_col] * all_bad_rate
    df2['good_expected'] = df2[total_col] * all_good_rate
    # 遍历每组的坏用户期望数量和实际数量
    bad_combined = zip(df2['bad_expected'], df2[bad_col])
    good_combined = zip(df2['good_expected'], df2[good_col])
    # 计算每组的卡方值
    bad_chi = [(i[0] - i[1]) ** 2 / (i[0] + 1e-6) for i in bad_combined]
    good_chi = [(i[0] - i[1]) ** 2 / (i[0] + 1e-6) for i in good_combined]
    # 计算总的卡方值
    chi2 = sum(bad_chi) + sum(good_chi)
    return chi2


def combine_bin_df(regroup, col, best_combined_index, continuous_flag=False):
    """
    :param regroup: dataframe [feature_name(optional), col, bin_total] or
      [feature_name(optional), col, bin_total, bin_bad, bin_good, bin_bad_rate]
      each row of col is a list if continuous_flag is False, else a pd.Interval.
      the indices must be increasing from 0 with increment of 1
    :param col: string, name of the feature range
    :param best_combined_index: int, the rows of this index
      and the next index of regroup will be combined and then filled
      into the row of this index, and then the row of the next index will be removed
    :param continuous_flag: bool
    """
    regroup_df = regroup.copy()
    # do this because pd.categorical dtype is not allowed to edit
    if continuous_flag:
        regroup_df[[col]] = regroup_df[[col]].astype('object')
    combine_df = regroup_df.loc[best_combined_index:best_combined_index+1, :]
    if continuous_flag:
        # pd.dataframe.at is like loc,
        # must use the index and column name, rather than positions
        regroup_df.at[best_combined_index, col] = pd.Interval(
            left=combine_df[col][best_combined_index].left,
            right=combine_df[col][best_combined_index+1].right)
    else:
        regroup_df.at[best_combined_index, col] = \
            combine_df[col][best_combined_index] + combine_df[col][best_combined_index+1]
    regroup_df.at[best_combined_index, 'bin_total'] = combine_df['bin_total'].sum()
    if 'bin_bad_rate' in regroup.columns.tolist():
        regroup_df.at[best_combined_index, 'bin_bad'] = combine_df['bin_bad'].sum()
        regroup_df.at[best_combined_index, 'bin_good'] = combine_df['bin_good'].sum()
        regroup_df.at[best_combined_index, 'bin_bad_rate'] = \
            regroup_df['bin_bad'][best_combined_index] / regroup_df['bin_total'][best_combined_index]
    # 删除合并前的右区间, regroup_df的index始终保持从0开始从小到大排序
    regroup_df = regroup_df.loc[regroup_df.index != (best_combined_index+1), :]
    # make sure the indices are increasing from 0 with increment 1 when bin_df's returned
    regroup_df.reset_index(drop=True, inplace=True)
    return regroup_df


def chi2_merge(regroup, col, max_bin_num=5, min_binpct=0.0, monoticity=True):
    """
    :param regroup: dataframe [col, bin_total, bin_bad, bin_good, bin_bad_rate]
       rows of col are all list
       indices must be increasing from 0 with increment 1
    :param col: string, col name
    :param max_bin_num: int
    :param min_binpct: float
    :param monoticity: bool. If false, bad_rate could be of no order
    :return: dataframe [col, bin_total, bin_bad, bin_good, bin_bad_rate]
      indices is increasing from 0 with increment 1
    """
    monotone = False
    while not monotone:
        # 按卡方合并箱体
        # 当group_interval的长度大于max_bin时，执行while循环
        while regroup.shape[0] > max_bin_num:
            chi_list = []
            for i in range(regroup.shape[0] - 1):
                # 计算每一对相邻区间的卡方值
                chi_df = regroup.loc[i:i+1, :]
                chi_value = cal_chi2(chi_df, 'bin_bad', 'bin_good', 'bin_total')
                chi_list.append(chi_value)
            # 最小的卡方值的索引,如最小值不止一个则index函数保证取第一个
            best_combined_index = chi_list.index(min(chi_list))
            # 将卡方值最小的一对区间进行合并
            regroup = combine_bin_df(regroup, col, best_combined_index)

        bin_df = regroup.copy()
        # 检查是否有箱只有好样本或者只有坏样本
        min_bad_rate = bin_df['bin_bad_rate'].min()
        max_bad_rate = bin_df['bin_bad_rate'].max()
        while min_bad_rate == 0.0 or max_bad_rate == 1.0:
            # 违约率为1或0的箱体的index. index总是返回列表，因此用[0]取第一个元素
            bad_index = bin_df[bin_df['bin_bad_rate'].isin([0.0, 1.0])].index[0]
            if bad_index == (bin_df.shape[0] - 1):
                bin_df = combine_bin_df(bin_df, col, bad_index-1)
            elif bad_index == 0:
                bin_df = combine_bin_df(bin_df, col, 0)
            else:
                # 计算bad_index和前面一个的箱体的卡方值
                chi_df = bin_df.loc[bad_index-1:bad_index, :]
                chi1 = cal_chi2(chi_df, 'bin_bad', 'bin_good', 'bin_total')
                # 计算bad_index和后一个箱体之间的卡方值
                chi_df = bin_df.loc[bad_index:bad_index+1, :]
                chi2 = cal_chi2(chi_df, 'bin_bad', 'bin_good', 'bin_total')
                if chi1 < chi2:
                    # 当chi1<chi2时, 合并bad_index之前与bad_index对应的箱体
                    bin_df = combine_bin_df(bin_df, col, bad_index-1)
                else:
                    # 当chi1>=chi2时,合并bad_index和bad_index之后对应的箱体
                    bin_df = combine_bin_df(bin_df, col, bad_index)

            # 计算每个区间的违约率, 其index一定是从0开始从小到大编号
            min_bad_rate = bin_df['bin_bad_rate'].min()
            max_bad_rate = bin_df['bin_bad_rate'].max()

        # 检查分箱后的最小占比
        if min_binpct > 0.0:
            # 得出最小的区间占比
            min_pct = (bin_df['bin_total'] / bin_df['bin_total'].sum()).min()
            # 当最小的区间占比小于min_binpct且箱体的个数大于3
            while min_pct < min_binpct and bin_df.shape[0] > 3:
                # 下面的逻辑基本与“检验是否有箱体只有好/坏样本”的一致
                bad_index = \
                    bin_df[(bin_df['bin_total']
                            / bin_df['bin_total'].sum()) == min_pct].index[0]
                if bad_index == (bin_df.shape[0] - 1):
                    bin_df = combine_bin_df(bin_df, col, bad_index-1)
                elif bad_index == 0:
                    bin_df = combine_bin_df(bin_df, col, 0)
                else:
                    chi_df = bin_df.loc[bad_index-1:bad_index, :]
                    chi1 = cal_chi2(chi_df, 'bin_bad', 'bin_good', 'bin_total')
                    chi_df = bin_df.loc[bad_index:bad_index+1, :]
                    chi2 = cal_chi2(chi_df, 'bin_bad', 'bin_good', 'bin_total')
                    if chi1 < chi2:
                        bin_df = combine_bin_df(bin_df, col, bad_index-1)
                    else:
                        bin_df = combine_bin_df(bin_df, col, bad_index)
                min_pct = (bin_df['bin_total'] / bin_df['bin_total'].sum()).min()
        if monoticity:
            bin_df2 = compute_woe_iv(bin_df, 'bin_good', 'bin_bad', 'bin_total')
            monotone = check_monotone(bin_df2, 'bin_woe', None)
            if not monotone:
                print('another chi2 merge is going to be made as the bad rate is not monotonic')
            max_bin_num -= 1
        else:
            break
    return bin_df


def sub_chisq(ks_points, start, end, bc_gap, bad_rate):
    if start >= end - 1 or sum(ks_points[end][1:]) - sum(ks_points[start][1:]) < 2 * bc_gap:
        return []
    temp = []
    start_good, start_bad = ks_points[start][1], ks_points[start][2]
    good, bad = max(ks_points[end][1] - start_good, 1e-6), max(ks_points[end][2] - start_bad, 1e-6)
    if good == 0 or bad == 0:
        return []
    for j in range(start, end):
        cur_good = ks_points[j][1]
        cur_bad = ks_points[j][2]

        if cur_good + cur_bad > 0:
            left_total = cur_good - start_good + cur_bad - start_bad
            right_total = good - cur_good + bad - cur_bad
            left_bad_expected = int(left_total * bad_rate)
            right_bad_expected = int(right_total * bad_rate)
            left_good_expected = int(left_total * (1 - bad_rate))
            right_good_expected = int(right_total * (1 - bad_rate))
            a11 = 0 if left_good_expected == 0 else np.square(
                (cur_good - start_good) - left_good_expected) / left_good_expected
            a12 = 0 if left_bad_expected == 0 else \
                np.square((cur_bad - start_bad) - left_bad_expected) / left_bad_expected
            a21 = 0 if right_good_expected == 0 else np.square(
                (good - cur_good) - right_good_expected) / right_good_expected
            a22 = 0 if right_bad_expected == 0 else \
                np.square((bad - cur_bad) - right_bad_expected) / right_bad_expected
            chisq = a11 + a12 + a21 + a22

        else:
            chisq = -1
        temp.append((chisq, j - start))
    max_index = max(temp, key=lambda x: x[0])[1]
    while temp[max_index][0] > 0 and \
            (sum(ks_points[max_index + start][1:]) - sum(ks_points[start][1:]) < bc_gap
             or sum(ks_points[end][1:]) - sum(ks_points[max_index + start][1:]) < bc_gap):
        temp[max_index] = (-1, -1)
        ks_points[max_index + start] = (ks_points[max_index + start][0], 0, 0)
        max_index = max(temp, key=lambda x: x[0])[1]

    if temp[max_index][0] <= 0:
        return []
    return [ks_points[max_index + start]] + sub_chisq(ks_points, start, max_index + start, bc_gap, bad_rate) \
           + sub_chisq(ks_points, max_index + start, end, bc_gap, bad_rate)


def best_combine(ks_points, good_bad, piece, monotonicity, lower, upper):
    num_cut_point = len(ks_points)
    num_of_cut = min(piece - 1, num_cut_point)
    cut_points = list(itertools.combinations(range(num_cut_point), num_of_cut))
    sol, max_iv = None, 0
    for cut in cut_points:
        old, iv, iv_list, woe_list = [0, 0], 0, [], []
        for c in cut:
            good_pcnt = (ks_points[c][-2] - old[0]) / good_bad[0]
            bad_pcnt = (ks_points[c][-1] - old[1]) / good_bad[1]
            old = (ks_points[c][-2], ks_points[c][-1])
            woe_list.append(np.log(bad_pcnt / (good_pcnt + 1e-6) + 1e-6))
            iv_list.append(woe_list[-1] * (bad_pcnt - good_pcnt))
            iv += iv_list[-1]
        good_pcnt = (good_bad[0] - old[0]) / good_bad[0]
        bad_pcnt = (good_bad[1] - old[1]) / good_bad[1]
        woe_list.append(np.log(bad_pcnt / (good_pcnt + 1e-6) + 1e-6))
        iv_list.append(woe_list[-1] * (bad_pcnt - good_pcnt))
        iv += iv_list[-1]
        if iv > max_iv:
            flag = True
            flag2 = True
            if monotonicity:
                for i in range(1, len(woe_list)):
                    if woe_list[i] < woe_list[i - 1]:
                        flag = False
                        break
                for i in range(1, len(woe_list)):
                    if woe_list[i] >= woe_list[i - 1]:
                        flag2 = False
                        break
                flag = flag or flag2
            if upper < lower:
                for i in range(1, len(iv_list)):
                    if iv_list[i] < lower or iv_list[i] > upper:
                        flag = False
                        break
            if flag:
                sol = cut
                max_iv = iv
    return sol


def calc_all_information(ks_points, good_bad, sol):
    result, old = [], [0, 0]
    for t in range(len(sol)):
        c = sol[t]
        if t == 0:
            bin_name = pd.Interval(left=float('-inf'), right=ks_points[c][0])
        else:
            bin_name = pd.Interval(left=ks_points[sol[t - 1]][0], right=ks_points[c][0])
        gd_cum, bd_cum = ks_points[c][-2], ks_points[c][-1]
        gd, bd = gd_cum - old[0], bd_cum - old[1]
        total = gd + bd
        bad_rate = bd / total if total > 0 else 0.0
        old = (ks_points[c][-2], ks_points[c][-1])
        result.append([bin_name, total, gd, bd, round(bad_rate, 6)])
    bin_name = pd.Interval(left=ks_points[sol[-1]][0], right=float('inf'))
    gd_cum, bd_cum = float(good_bad[0]), float(good_bad[1])
    gd, bd = gd_cum - old[0], bd_cum - old[1]
    total = gd + bd
    bad_rate = bd / total if total > 0 else 1.0
    result.append([bin_name, total, gd, bd, round(bad_rate, 6)])
    return result


def best_chisq_bin(group, bc_piece, bc_good_bad, bc_gap, bc_strict_monotonicity,
                   bc_iv_lower, bc_iv_upper, merge):
    feat_name = group[0][0]
    good = bc_good_bad[0]
    bad = bc_good_bad[1]
    bad_rate = 1.0 * bad / (bad + good)
    group = sorted(group, key=lambda x: x[1])
    counter = [1.0, 0.0] if group[0][-1] == 0 else [0.0, 1.0]

    ks_points = [(feat_name, counter[0], counter[1])]
    for j in range(1, len(group)):
        if group[j][1] != group[j - 1][1]:
            ks_points.append((group[j - 1][1], counter[0], counter[1]))
        counter[0 if group[j][-1] == 0 else 1] += 1
    if group[-1][1] != ks_points[-1][0]:
        ks_points.append((group[-1][1], counter[0], counter[1]))

    if merge:
        ks_points = sub_chisq(ks_points, 0, len(ks_points) - 1, bc_gap, bad_rate)
        ks_points = sorted(ks_points, key=lambda x: x[0])
        sub_piece = bc_piece
    else:
        sub_piece = len(ks_points) - 1
        ks_points = ks_points[1:-1]

    can_be_bin = False
    for tmp_piece in range(sub_piece - 1):
        sol = best_combine(ks_points, bc_good_bad, sub_piece - tmp_piece,
                           bc_strict_monotonicity, bc_iv_lower, bc_iv_upper)
        if sol:
            can_be_bin = True
            result = calc_all_information(ks_points, bc_good_bad, sol)
            break
    if not can_be_bin:
        print("不能对字段{}进行分箱".format(feat_name))
        return None
    return pd.DataFrame(result, columns=['bin_range', 'bin_total', 'bin_good', 'bin_bad', 'bin_bad_rate'])