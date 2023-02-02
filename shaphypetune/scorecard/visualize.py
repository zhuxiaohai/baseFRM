import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mc
import colorsys
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import toad
from .utils import bin_bad_rate, compute_woe_iv


def plot_trend(df_statistics, title,
               bar_cols=None, curve_cols=None, save_dir=None):
    """
    看某个变量(名为title)随时间变化的趋势
    :param df_statistics: pd.dataframe,
         dataframe的index为时间趋势变量, 横轴为统计量
    :param title: str, 变量名称
    :param bar_cols: list of str, 要绘制柱状图的统计量名字
    :param curve_cols: list of str, 要绘制折线图的统计量名字
    :param save_dir: string, 当前路径下的相对路径
    """
    if curve_cols is None:
        curve_cols = df_statistics.columns.tolist()
    fig, ax1 = plt.subplots()
    cmap = mpl.colors.ListedColormap(['g', 'r', 'b', 'k',
                                      '#A0FFA0', '#FFA0A0', '#A0A0FF'])
    df_statistics[curve_cols].plot(ax=ax1, cmap=cmap, marker='o', markersize=3)
    ax1.set_ylabel("curve")
    lines1, labels1 = ax1.get_legend_handles_labels()
    if bar_cols is not None:
        ax2 = ax1.twinx()
        df_statistics[bar_cols].plot(ax=ax2, kind='bar',
                                     title=title, align='center')
        ax2.set_ylabel("bar")
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines1 += lines2
        labels1 += labels2
        ax2.legend([])
    # 绘制x轴（阈值）
    effective_indices_num = df_statistics.index.shape[0]
    if effective_indices_num > 5:
        # 向下取整
        increment = int(effective_indices_num / 5)
    else:
        increment = 1
    indices = range(0, effective_indices_num, increment)
    ax1.set_xticks(indices)
    ax1.set_xticklabels([i for i in df_statistics.index[indices]],
                        rotation=-20, horizontalalignment='left')
    ax1.legend(lines1, labels1, bbox_to_anchor=(1.05, 1),
               loc='upper left')
    if save_dir is not None:
        current_dir = os.getcwd()
        new_dir = os.path.join(current_dir, save_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        save_path = os.path.join(new_dir, title + '.jpg')
        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()
        plt.close('all')
    else:
        plt.show()


def score_plot(scoredata, score='score', target='target', score_bond=None, line_flag=True):
    if score_bond is None:
        left = scoredata[score].min()
        right = scoredata[score].max()
        score_bond = np.arange(left, right + 50, 50)
    labels = list(range(len(score_bond) - 1))

    baddf = scoredata[scoredata[target] == 1]
    gooddf = scoredata[scoredata[target] == 0]
    badstat = pd.cut(baddf[score], bins=score_bond, labels=labels, include_lowest=True)
    goodstat = pd.cut(gooddf[score], bins=score_bond, labels=labels, include_lowest=True)
    allstat = pd.cut(scoredata[score], bins=score_bond, labels=labels, include_lowest=True)

    # 统计各分数段样本数量
    bad_count = pd.value_counts(badstat, sort=False).values
    good_count = pd.value_counts(goodstat, sort=False).values
    y_count = pd.value_counts(allstat, sort=False).values

    # 计算区间坏账率
    num_ticks = len(score_bond)
    ticks = ['(%d,%d]' % (score_bond[i], score_bond[i + 1])
             for (i, x) in enumerate(score_bond) if i < len(score_bond) - 1]
    num_ticks = len(ticks)
    score_stat_df = pd.DataFrame({'range': ticks,
                                  'bad_count': bad_count,
                                  'good_count': good_count,
                                  'y_count': y_count,
                                  'y_rate': bad_count / y_count})

    # 坐标轴名称
    x_label = "Scores"
    y_label_left = "Counts"
    y_label_right = "Bad Rates"
    graph_title = "Score Distribution"

    x = np.arange(num_ticks)
    y1 = score_stat_df['y_count']
    y2 = score_stat_df['y_rate']

    fig = plt.figure(figsize=(15.0, 8.0))
    # 画柱子
    ax1 = fig.add_subplot(111)
    # alpha透明度， edgecolor边框颜色，color柱子颜色 linewidth width 配合去掉柱子间距
    ax1.bar(x, y1, alpha=0.8, edgecolor='k', color='#836FFF', linewidth=1, width=1)
    # 获取 y 最大值 最高位 + 1 的数值 比如 201取300，320取400，1800取2000
    y1_lim = int(str(int(str(max(y1))[0]) + 1) + '0' * (len(str(max(y1))) - 1))
    # 设置 y轴 边界
    ax1.set_ylim([0, y1_lim])
    # 设置 y轴 标题
    ax1.set_ylabel(y_label_left, fontsize='15')
    ax1.set_xlabel(x_label, fontsize='15')
    # 将分值标注在图形上
    for x_i, y_i in zip(x, y1):
        ax1.text(x_i, y_i + y1_lim / 20, str(y_i), ha='center', va='center', fontsize=13, rotation=0)
    ax1.set_title(graph_title, fontsize='20')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ticks, rotation=-45, horizontalalignment='left')

    # 画折线图
    if line_flag:
        ax2 = ax1.twinx()  # 这个很重要噢
        ax2.plot(x, y2, 'r', marker='*', ms=0)
        try:
            y2_lim = (int(max(y2) * 10) + 1) / 10
        except:
            y2_lim = 1
        ax2.set_ylim([0, y2_lim])
        ax2.set_ylabel(y_label_right, fontsize='15')
        ax2.set_xlabel(x_label, fontsize='15')
        for x_i, y_i in zip(x, y2):
            ax2.text(x_i, y_i + y2_lim / 20, '%.2f%%' % (y_i * 100), ha='center', va='top', fontsize=13, rotation=0)

    plt.grid(True)
    plt.show()
    return score_stat_df


def plot_bin_df(bin_df_list, name_list=[''], save_dir=None):
    """
    :param bin_df_list: list of dataframe
    :param name_list: bin_df_list里面每个元素的名字
    :param save_dir: string, 前路径下的相对路径
    """
    assert len(bin_df_list) == len(name_list)
    if len(name_list) == 1:
        bar_width = 0.9
    else:
        bar_width = 0.8
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    df_bin_total = pd.DataFrame()
    df_bad_rate = pd.DataFrame()
    for bin_df, name in zip(bin_df_list, name_list):
        bin_pct_series = bin_df['bin_total_rate'].rename(name + ' bin_pct')
        df_bin_total = pd.concat([df_bin_total, bin_pct_series], axis=1)
        if 'bin_bad_rate' in bin_df.columns.tolist():
            bad_rate_series = bin_df['bin_bad_rate'].rename(name + ' bad_rate')
            df_bad_rate = pd.concat([df_bad_rate, bad_rate_series], axis=1)
    df_bin_total.index = bin_df['bin_no']
    title = bin_df['feature_name'][0]
    fig, ax = plt.subplots()
    df_bin_total.plot(kind='bar', cmap=cm_light, ax=ax,
                      width=bar_width, title=title)
    ax.set_ylabel("count_ratio")
    ax.set_xlim([-0.5, bin_df.shape[0] - 0.5])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if 'bin_bad_rate' in bin_df.columns.tolist():
        ax_curve = ax.twinx()
        df_bad_rate.plot(ax=ax_curve, marker='o', markersize=5, cmap=cm_dark)
        ax_curve.set_ylabel('bad_rate')
        ax_curve.legend(bbox_to_anchor=(1.05, 0), loc='lower left')
    plt.xlabel('bin no.')
    if save_dir is not None:
        current_dir = os.getcwd()
        new_dir = os.path.join(current_dir, save_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        save_path = os.path.join(new_dir, title + '.jpg')
        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()
        plt.close('all')
    else:
        plt.show()


def plot_roc(label, prediction):
    fpr, tpr, thresholds = roc_curve(label, prediction, pos_label=1)
    metric = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve, auc:%.4f' % metric)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()


def plot_pr(label, prediction,
            recall_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
    precisions, recalls, thresholds = precision_recall_curve(label, prediction)
    average_precision = average_precision_score(label, prediction)
    columns = ['recall', 'precision', 'improve']
    lift_df = []
    for recall in recall_list:
        index = np.argmin(np.abs(recalls - recall))
        lift_df.append([recall, precisions[index], precisions[index] / precisions[0]])
    lift_df = pd.DataFrame(lift_df, columns=columns)
    plt.plot(lift_df['recall'], lift_df['precision'])
    plt.title('PR-curve, ap:%.4f' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()
    return lift_df


def calc_lift(df, pred, target, groupnum=None, range_col=None, title_name='lift'):
    cm_light = '#A0A0FF'
    cm_dark = 'r'
    if groupnum is None:
        groupnum = df[range_col].unique().shape[0]

    def n0(x):
        return sum(x == 0)

    def n1(x):
        return sum(x == 1)

    def total(x):
        return x.shape[0]

    def name(x):
        return '[{:.2f}'.format(x.iloc[0]) + ', ' + '{:.2f}]'.format(x.iloc[-1])

    if range_col is None:
        dfkslift = df.sort_values(pred, ascending=True).reset_index(drop=True) \
            .assign(group=lambda x: np.ceil((x.index + 1) / (len(x.index) / groupnum))) \
            .groupby('group').agg({target: [n0, n1, total], pred: name}) \
            .reset_index().rename(columns={'name': 'range', 'n0': 'good', 'n1': 'bad', 'total': 'count'})
        columns = dfkslift.columns.droplevel(0).tolist()
        columns[0] = 'group'
    else:
        dfkslift = df.sort_values(pred, ascending=True).reset_index(drop=True) \
            .groupby(range_col).agg({target: [n0, n1, total]}) \
            .reset_index().rename(columns={range_col: 'range', 'n0': 'good', 'n1': 'bad', 'total': 'count'})
        columns = dfkslift.columns.droplevel(0).tolist()
        columns[0] = 'range'
    dfkslift.columns = columns
    dfkslift = dfkslift.assign(
        good_distri=lambda x: x.good / sum(x.good),
        bad_distri=lambda x: x.bad / sum(x.bad),
        total_distri=lambda x: x['count'] / sum(x['count']),
        cumgood_distri=lambda x: np.cumsum(x.good) / sum(x.good),
        cumbad_distri=lambda x: np.cumsum(x.bad) / sum(x.bad),
        badrate=lambda x: x.bad / (x.good + x.bad),
        cumbadrate=lambda x: np.cumsum(x.bad) / np.cumsum(x.good + x.bad),
        lift=lambda x: (np.cumsum(x.bad) / np.cumsum(x.good + x.bad)) / (sum(x.bad) / sum(x.good + x.bad))) \
        .assign(ks=lambda x: abs(x.cumbad_distri - x.cumgood_distri))
    dfkslift['lift'] = dfkslift.bad_distri / dfkslift.total_distri

    fig, ax = plt.subplots()
    dfkslift[['total_distri']].plot(kind='bar', width=0.3, color=cm_light, ax=ax, legend=False)
    ax.set_ylabel('total_distri')
    ax_curve = ax.twinx()
    dfkslift[['badrate']].plot(ax=ax_curve, marker='o', markersize=5, color=cm_dark, legend=False)
    ax_curve.set_ylabel('1_distri')
    ax_curve.grid()
    ax_curve.plot([0, groupnum - 1], [dfkslift['cumbadrate'].iloc[-1], dfkslift['cumbadrate'].iloc[-1]], 'r--')
    ax.set_xticks(np.arange(groupnum))
    ax.set_xticklabels(dfkslift['range'].values, rotation=-20, horizontalalignment='left')
    ax.set_xlim([-0.5, groupnum - 0.5])
    ax.set_title(title_name)
    return dfkslift, ax


def plot_stats(plot_spec, ax, bbox_to_anchor=(1, 1), loc='upper left'):
    def lighten_color(color, amount=0.5):
        """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
        """
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    cm_light = mpl.colors.ListedColormap(
        [lighten_color('g'), lighten_color('r'), lighten_color('b'), lighten_color('#F034A3')])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b', '#F034A3'])
    x = plot_spec[0]
    if plot_spec[1] == 'single':
        ax.plot(x.index, x.values, marker='o')
    elif plot_spec[1] == 'multiple':
        for bin_range in x.columns:
            ax.plot(x.index, x.loc[:, bin_range], marker='o')
        ax.legend(labels=x.columns, bbox_to_anchor=bbox_to_anchor, loc=loc)
    elif plot_spec[1] == 'stacked_bins':
        ax.bar(x.index, x.iloc[:, 0], align="center",
               tick_label=x.index, label=x.columns[0])
        for j in range(1, x.shape[1]):
            ax.bar(x.index, x.iloc[:, j], bottom=x.iloc[:, :j].sum(axis=1),
                   align="center", tick_label=x.index, label=x.columns[j])
        ax.legend(bbox_to_anchor=bbox_to_anchor, loc=loc)
    elif plot_spec[1] == 'bins':
        ax.bar(x.index, x.values, align="center", tick_label=x.index)
    elif plot_spec[1] == 'distr':
        ax = sns.distplot(x, bins=20, ax=ax)
    elif plot_spec[1] == 'mixed':
        bin_df, curve_df = x
        bin_df, curve_df = bin_df.T, curve_df.T
        bin_name, curve_name = plot_spec[2][1:]
        bin_df.plot(kind='bar', width=0.8, ax=ax, cmap=cm_light)
        ax.set_ylabel(bin_name)
        ax.legend(bbox_to_anchor=bbox_to_anchor, loc=loc)
        ax_curve = ax.twinx()
        curve_df.plot(ax=ax_curve, marker='o', markersize=5, legend=False, cmap=cm_dark)
        ax.set_xlim([-0.5, bin_df.shape[0] - 0.5])
        ax_curve.set_ylabel(curve_name)
        ax_curve.grid(False)

    if isinstance(x, tuple):
        labels = x[0].T.index
        title = plot_spec[2][0]
    else:
        labels = x.index
        title = plot_spec[2]
    if plot_spec[3]:
        if plot_spec[1] != 'distr':
            #             ax.set_xticks(labels)
            ax.set_xticklabels(labels=labels, rotation=-20, horizontalalignment='left')
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax.grid()
    ax.set_title(title)


def monitor(all_df, combiners, features, target, cur_dir,
            group='set', set_array=['1train', '2test', '3oot'], NA=-9999,
            fig_size=(8, 16), plot_configs=[{'psi_series': 1, 'iv_series': 1,
                                             'countratio_df': 2, 'lift_df': 1.5, 'mixed': 2}]):
    def check_columns(expected_df, actual_bins_df):
        for col in actual_bins_df.index:
            if col not in expected_df.columns:
                expected_df[col] = np.nan
        return expected_df[expected_df.columns.sort_values()]

    assert len(combiners) <= 2
    assert len(plot_configs) <= 2
    assert len(combiners) == len(plot_configs)

    df = all_df.copy()
    expected = df.loc[df[group] == set_array[0], :]

    #     mixed = True
    #     dir_name = 'eda_mob3_k11_test'
    #     features = selected_psi
    #     target = y

    mpl.use('Agg')
    writer = pd.ExcelWriter(os.path.join(cur_dir, 'stats.xlsx'), engine='openpyxl')
    for start_col_index, col in enumerate(features):
        fig = plt.figure(figsize=fig_size)
        gs = gridspec.GridSpec(len(plot_configs[0]), len(plot_configs),
                               height_ratios=[height for height in plot_configs[0].values()])
        for combiner_index, (combiner, plot_config) in enumerate(zip(combiners, plot_configs)):
            plot_dict = {}
            expected_bin = combiner.transform(expected[[col, target]], labels=True)
            bins_df = bin_bad_rate(expected_bin, col, target)
            bins_df = compute_woe_iv(bins_df, 'bin_good', 'bin_bad', 'bin_total')
            assert (expected.shape[0] == bins_df['bin_total'].sum())
            nullrate_series = pd.Series()
            if expected[col].dtype == 'object':
                bins_df[col] = bins_df[col].apply(lambda x: x[:10])
                nullrate_series.loc[set_array[0]] = expected[(expected[col] == str(NA)) |
                                                             (expected[col] == 'nan') |
                                                             (expected[col].isnull())].shape[0] / expected.shape[0]
                expected_bin_notnull = combiner.transform(expected[~((expected[col] == str(NA)) |
                                                                     (expected[col] == 'nan') |
                                                                     (expected[col].isnull()))][[col, target]],
                                                          labels=True)
            else:
                nullrate_series.loc[set_array[0]] = expected[(expected[col] == NA) |
                                                             (expected[col].isnull())].shape[0] / expected.shape[0]
                expected_bin_notnull = combiner.transform(expected[~((expected[col] == NA) |
                                                                     (expected[col].isnull()))][[col, target]],
                                                          labels=True)
            bins_df_notnull = bin_bad_rate(expected_bin_notnull, col, target)
            bins_df_notnull = bins_df_notnull.set_index(col)
            expected_count_ratio = bins_df_notnull['bin_total'] / bins_df_notnull['bin_total'].sum()
            expected_bad_ratio = bins_df_notnull['bin_bad'] / bins_df_notnull['bin_bad'].sum()
            bins_df = bins_df.set_index(col)
            badrate_df = pd.DataFrame(columns=bins_df.index)
            badrate_df.loc[set_array[0], bins_df.index] = bins_df.bin_bad_rate
            woe_df = pd.DataFrame(columns=bins_df.index)
            woe_df.loc[set_array[0], bins_df.index] = bins_df.bin_woe
            countratio_df = pd.DataFrame(columns=bins_df.index)
            countratio_df.loc[set_array[0], bins_df.index] = bins_df.bin_total_rate
            lift_df = pd.DataFrame(columns=bins_df.index)
            lift_df.loc[set_array[0], bins_df.index] = (bins_df.bin_bad / bins_df.bin_bad.sum()) / (
                        bins_df.bin_total / bins_df.bin_total.sum())
            total_series = pd.Series()
            total_series.loc[set_array[0]] = bins_df['bin_total'].sum()
            psi_series = pd.Series()
            psi_series.loc[set_array[0]] = np.nan
            psi_positive_series = pd.Series()
            psi_positive_series.loc[set_array[0]] = np.nan
            iv_series = pd.Series()
            iv_series.loc[set_array[0]] = toad.stats.IV(expected_bin[col], expected_bin[target])
            for i in range(1, len(set_array)):
                actual = df.loc[(df[group] == set_array[i]), [col, target]]
                actual_bin = combiner.transform(actual, labels=True)
                bins_df = bin_bad_rate(actual_bin, col, target)
                bins_df = compute_woe_iv(bins_df, 'bin_good', 'bin_bad', 'bin_total')
                assert (actual.shape[0] == bins_df['bin_total'].sum())
                if actual[col].dtype == 'object':
                    bins_df[col] = bins_df[col].apply(lambda x: x[:10])
                    nullrate_series.loc[set_array[i]] = actual[(actual[col] == str(NA)) |
                                                               (expected[col] == 'nan') |
                                                               (actual[col].isnull())].shape[0] / actual.shape[0]
                    actual_bin_notnull = combiner.transform(actual[~((actual[col] == str(NA)) |
                                                                     (expected[col] == 'nan') |
                                                                     (actual[col].isnull()))][[col, target]],
                                                            labels=True)
                else:
                    nullrate_series.loc[set_array[i]] = actual[(actual[col] == NA) |
                                                               (actual[col].isnull())].shape[0] / actual.shape[0]
                    actual_bin_notnull = combiner.transform(actual[~((actual[col] == NA) |
                                                                     (actual[col].isnull()))][[col, target]],
                                                            labels=True)
                bins_df_notnull = bin_bad_rate(actual_bin_notnull, col, target)
                bins_df_notnull = bins_df_notnull.set_index(col)
                actual_count_ratio = bins_df_notnull['bin_total'] / bins_df_notnull['bin_total'].sum()
                actual_bad_ratio = bins_df_notnull['bin_bad'] / bins_df_notnull['bin_bad'].sum()
                bins_df = bins_df.set_index(col)
                # newly added
                badrate_df = check_columns(badrate_df, bins_df)
                # newly ----
                badrate_df.loc[set_array[i], bins_df.index] = bins_df.bin_bad_rate
                # newly added
                woe_df = check_columns(woe_df, bins_df)
                # newly ----
                woe_df.loc[set_array[i], bins_df.index] = bins_df.bin_woe
                # newly added
                countratio_df = check_columns(countratio_df, bins_df)
                # newly ----
                countratio_df.loc[set_array[i], bins_df.index] = bins_df.bin_total_rate
                # newly added
                lift_df = check_columns(lift_df, bins_df)
                # newly ----
                lift_df.loc[set_array[i], bins_df.index] = (bins_df.bin_bad / bins_df.bin_bad.sum()) / (
                            bins_df.bin_total / bins_df.bin_total.sum())
                total_series.loc[set_array[i]] = bins_df['bin_total'].sum()
                psi_series.loc[set_array[i]] = np.sum((expected_count_ratio - actual_count_ratio) *
                                                      np.log(expected_count_ratio / actual_count_ratio))
                psi_positive_series.loc[set_array[i]] = np.sum((expected_bad_ratio - actual_bad_ratio) *
                                                               np.log(expected_bad_ratio / actual_bad_ratio))
                iv_series.loc[set_array[i]] = toad.stats.IV(actual_bin[col], actual_bin[target])

            plot_dict['total_series'] = (total_series, 'single', col + '_total')
            plot_dict['nullrate_series'] = (nullrate_series, 'single', col + '_nullrate')
            plot_dict['psi_series'] = (psi_series, 'single', col + '_psi')
            plot_dict['psi_positive_series'] = (psi_positive_series, 'single', col + '_psi_p')
            plot_dict['iv_series'] = (iv_series, 'single', col + '_iv')
            plot_dict['countratio_df'] = (countratio_df, 'stacked_bins', col + '_countratio')
            plot_dict['badrate_df'] = (badrate_df, 'multiple', col + '_1prob')
            plot_dict['lift_df'] = (lift_df, 'multiple', col + '_liftratio')
            plot_dict['woe_df'] = (woe_df, 'multiple', col + '_woe')
            plot_dict['mixed'] = ((countratio_df, badrate_df), 'mixed', (col + '_bin', 'count_ratio', 'badrate'))

            for key, item in plot_dict.items():
                if key == 'mixed':
                    continue
                if isinstance(item[0], pd.DataFrame):
                    temp_df = item[0]
                    temp_df.columns = [col + '_' + temp_col_name for temp_col_name in temp_df.columns]
                    startcol = start_col_index * 6
                    selected_columns = [temp_col_name for temp_col_name in temp_df.columns if
                                        temp_col_name.find('.nan') < 0]
                    headtail_df = temp_df[selected_columns]
                    head_df = headtail_df.iloc[:, 0]
                    tail_df = headtail_df.iloc[:, -1]
                    head_df.to_excel(writer, sheet_name=key + '_head_combiner' + str(combiner_index),
                                     startcol=start_col_index, index=False)
                    tail_df.to_excel(writer, sheet_name=key + '_tail_combiner' + str(combiner_index),
                                     startcol=start_col_index, index=False)
                else:
                    temp_df = pd.DataFrame(item[0])
                    temp_df.columns = [col]
                    startcol = start_col_index
                temp_df.to_excel(writer, sheet_name=key + '_combiner' + str(combiner_index), startcol=startcol,
                                 index=False)

            config_length = len(plot_config)
            plot_setting = []
            for i, (config_name, fig_ratio) in enumerate(plot_config.items()):
                if i != (config_length - 1):
                    plot_setting.append(plot_dict[config_name] + (False, True, fig_ratio))
                else:
                    if config_name == 'mixed':
                        plot_setting.append(plot_dict[config_name] + (True, False, fig_ratio))
                    else:
                        plot_setting.append(plot_dict[config_name] + (True, True, fig_ratio))

            #     if mixed:
            #         plot_setting = [
            #                         #(total_series, 'single', col+'_total', False, True, 1),
            #                         #(nullrate_series, 'single', col+'_nullrate', False, True, 1),
            #                         (psi_series, 'single', col+'_psi', False, True, 1),
            #                         (iv_series, 'single', col+'_iv', False, True, 1),
            #                         (countratio_df, 'stacked_bins', col+'_countratio', False, True, 2),
            #                         #(badrate_df, 'multiple', col+'_1prob', False, True, 1.5),
            #                         (lift_df, 'multiple', col+'_liftratio', False, True, 1.5),
            #                         #(woe_df, 'multiple', col+'_woe', True, True, 1.5),
            #                         ((countratio_df, badrate_df), 'mixed', (col+'_bin', 'count_ratio', 'badrate'), True, False, 2)
            #                         ]
            #     else:
            #         plot_setting = [
            #                         (total_series, 'single', col+'_total', False, True, 1),
            #                         (nullrate_series, 'single', col+'_nullrate', False, True, 1),
            #                         (psi_series, 'single', col+'_psi', False, True, 1),
            #                         (iv_series, 'single', col+'_iv', False, True, 1),
            #                         (countratio_df, 'stacked_bins', col+'_countratio', False, True, 2),
            #                         (badrate_df, 'multiple', col+'_1prob', False, True, 1.5),
            #                         (woe_df, 'multiple', col+'_woe', True, True, 1.5)
            #                        ]
            #     cur_dir = os.path.join(os.getcwd(), dir_name)
            # if not os.path.exists(cur_dir):
            #    os.makedirs(cur_dir)
            # fig = plt.figure(figsize=fig_size)
            # gs = gridspec.GridSpec(len(plot_setting), 1, height_ratios=[setting[-1] for setting in plot_setting])
            if (combiner_index == 0) and (len(plot_configs) > 1):
                bbox_to_anchor = (-0.1, 1)
                loc = 'upper right'
            else:
                bbox_to_anchor = (1.1, 1)
                loc = 'upper left'
            ax0 = fig.add_subplot(gs[0, combiner_index])
            plot_stats(plot_setting[0], ax0, bbox_to_anchor, loc)
            for i in range(1, len(plot_setting)):
                if plot_setting[i][4]:
                    ax = fig.add_subplot(gs[i, combiner_index], sharex=ax0)
                else:
                    ax = fig.add_subplot(gs[i, combiner_index])
                plot_stats(plot_setting[i], ax, bbox_to_anchor, loc)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        save_dir = os.path.join(cur_dir, col + '_stats.jpg')
        fig.savefig(save_dir, bbox_inches='tight')
        plt.close('all')
    writer.save()