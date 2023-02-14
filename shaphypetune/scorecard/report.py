import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import toad

from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import ticker
from matplotlib.pyplot import style
import pylab
pylab.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
pylab.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
style.use('seaborn-white')
mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

import re
import os
import warnings
warnings.filterwarnings('ignore')


# 需要用到的函数
def full_describe(df):
    cat_vars = list(df.dtypes[df.dtypes == 'O'].index)
    num_vars = list(df.dtypes[df.dtypes != 'O'].index)
    miss_cat = pd.DataFrame(df[cat_vars].apply(lambda x: '{:.2%}'.format(1 - x.count()/x.size)), columns=['miss%'])
    miss_num = pd.DataFrame(df[num_vars].apply(lambda x: '{:.2%}'.format(1 - x.count()/x.size)), columns=['miss%'])
#    top = pd.DataFrame(df[cat_vars].apply(lambda x : x.value_counts()[:7].to_dict()),columns=['top'])
    top = pd.DataFrame({x: pd.Series([str(k)+': '+str(v) for k, v in df[x].value_counts()[:7].to_dict().items()]) for x in cat_vars}).T
    top.columns = ['value'+str(i+1) for i in range(7)]
    desc = df[cat_vars].describe().T[['count', 'unique']]
    cat_desc = pd.concat([miss_cat, desc, top], axis=1)
    num_desc = pd.concat([miss_num, df[num_vars].describe(percentiles=[0.95]).T], axis=1)
    return cat_desc, num_desc


#%% 判断单调函数
def monotonic_trend(x):
    if len(set(x.values)) <= 1:
        return 'no_Mono'
    else:
        corr = np.corrcoef(x, [i+1 for i in range(len(x))]).min()
        if corr <= -0.65:
            return 'descending'
        elif corr >= 0.65:
            return "ascending"
        else:
            return "no_Mono"
 

def get_out_excel():
    """修改获取路径方式，使用io打开"""
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'out_all_in_one_template.xlsx')
    # f = pd.read_excel(file_path)
    return file_path


#%% 准备数据，跨时切分以及 变量分箱体
def get_df_dt(df, dt='shouxin_date', dt_cut='month'):
    '''
    Parameters
    ----------
    df : dataframe
        包含时间列的df.
    dt : string, optional
        时间字段. The default is 'shouxin_date'.
    dt_cut : string\list\int, optional
        The default is 'month'.
        样本拆分月自定义切点，list自定义 如:[-np.inf,20190101,20190601,20190901,np.inf]
        或者指定某一列 如'shouxin_month','month1' 
        或者是整数，2,3 表示将样本等分切成2份，三份

    Returns
    -------
    df with dt.

    '''
    # 判断跨时字段
    if isinstance(dt_cut, int):
        _index = df.sort_values(by=dt).index
        df['dt_cut'] = 0
        parts = 1
        for index_slice in np.array_split(_index, dt_cut):
            df.loc[index_slice, 'dt_cut'] = parts
            parts += 1
    else:
        df['dt_cut'] = df[dt_cut].astype('str')
    return df


def cal_ks(data,prob,y):
    return ks_2samp(data[prob][(data[y] == 1)],
                    data[prob][(data[y] == 0)]).statistic


def cal_iv(x, y):
    df = y.groupby(x).agg(['count', 'sum'])
    df.columns = ['total', 'count_1']
    df['count_0'] = df.total - df.count_1
    df['pct'] = df.total/df.total.sum()
    df['bad_rate'] = df.count_1/df.total
    df['woe'] = np.log((df.count_0/df.count_0.sum())/(df.count_1/df.count_1.sum()))
    df['woe'] = df['woe'].replace(np.inf,0)
#    print(df)
    rate = ((df.count_0/df.count_0.sum())-(df.count_1/df.count_1.sum()))
    IV = np.sum(rate*df.woe)
    return IV, df


def beautify_str(x, digit=4):
    if x[-4:] == '.nan':
        return x
    split = x.split(' ~ ')
    if len(split) == 1:
        return x[:x.find('.')]
    else:
        a, b = split
    pre, a = a.split('[')
    b, post = b.split(')')
    a = round(float(a), digit)
    b = round(float(b), digit)
    return pre + '[' + str(a) + ' ~ ' + str(b) + ')' + post


def append_nan_to_cat_binning(cut, add_nan=False):
    if not isinstance(cut, list):
        cut = cut.tolist()
    for i in range(len(cut)):
        bins = cut[i]
        break_flag = False
        for j in range(len(bins)):
            value = bins[j]
            if value == 'nan':
                bins.pop(j)
                break_flag = True
                break
        if break_flag:
            break
    if break_flag:
        cut.append(['nan'])
    elif add_nan:
        cut.append(['nan'])
    return cut


def get_x_group(df, x, y, score_cut=5, method='quantile', binning_col='binning', binning_set='1train'):
    combiner = toad.transform.Combiner()
    if not isinstance(score_cut, int):
        combiner.set_rules({x: score_cut})
    else:
        combiner.fit(df[(df[binning_col] == binning_set)][[x, y]],
                     y=y,
                     n_bins=score_cut,
                     method=method,
                     empty_separate=True)
        if df[x].isnull().sum() > 0:
            score_cut = combiner.export()[x]
            if (df[x].dtypes != 'object') and (not np.isnan(score_cut).any()):
                score_cut = np.append(score_cut, np.nan)
                combiner.set_rules({x: score_cut})
            elif df[x].dtypes == 'object':
                score_cut = append_nan_to_cat_binning(score_cut, add_nan=True)
                combiner.set_rules({x: score_cut})
    df['group'] = combiner.transform(df[x], labels=True)
    return df['group'], combiner.export()


#%% out all in one 把上面 psi，null，iv等指标包在一个里面统一输出
def out_all_in_one(df, inputx, y='fpd4', dt='event_date', dt_cut='month', miss_values=[-99],
                   score_cut=None, method='quantile', digit=4,
                   binning_col='binning', binning_set='1train', output_path='out_all_in_one_report.xlsx'):
    '''
    Parameters
    ----------
    df : pd.dataframe
        输入dataframe,必须dt.x
    inputx : list
        模型分\特征、变量 如['x1','x2'],字符型，数值型都可
    y : string
        样本的因变量 如 'fpd4', 因变量为0,1的int格式输入，y=None是，样本不含y标签.The default is 'fpd4'.
    dt : string, optional
        样本拆分的日期，一般我们建模中常用的，建案时间，交易时间等等。格式为 20190101、2019-01-01均可. The default is 'event_date'.
    dt_cut : list,int,string optional
        =list 时 如：[-np.inf,20190101,20190601,20190901,np.inf]
        ='month' 或者指定df中的某一列 如'shouxin_month','month1'
        =1,2,3 样本等分成3份。。 
        The default is 'month', 不管样本中是否包含month列，都可以直接按月拆分.
    miss_values : list
        缺失值列表，缺失值单独一箱 The default is [-99].
    score_cut : list, optional
        默认None,按照method方法分箱，支持字符和数值格式
        = int, 特征等频分箱的组数，如等频分为10箱. 
        = list 可自定义模型分切点 如：[547.0,570.0,584.0,600.0, 614.0, 626.0, 639.0, 654.0]
        注意：自定义切点时，inputx 只能放自定义切点的这个变量。
    method : string
        特征分箱方法，有'optb','auto'两种，前者是optbinning分箱方法，后者是自动等频分箱。均支持字符特征的自动分箱。
        The default is 'optb'.
    output_path : 存储地址
        默认'out_all_in_one_report.xlsx' 导出excel =None时，不导出excel

    Returns
    -------
    out_vardf : dataframe
        inputx 按跨时dt_cut的 miss_rate, psi (y=None是有), iv, ks, auc, bottom_lift, top_lift, monotonic(y=None是无)
    out_bindf : dataframe
        inputx 按跨时dt_cut，分箱每一箱的 miss_rate, len,bad,mean,lift.

    '''
    df = df.copy()
    temp_df = get_df_dt(df, dt=dt, dt_cut=dt_cut)
    temp_df.replace(miss_values, np.nan, inplace=True)
    output_score_cut = {}
    
    TB1 = pd.DataFrame()
    TB2 = pd.DataFrame()
    for x in inputx:
        print(x)
        if y is None:
            cut = 10
            temp_df['fake_y'] = 1
            y = 'fake_y'
        else:
            cut = score_cut[x]

        temp_df['group'], x_score_cut = get_x_group(temp_df, x=x, y=y, score_cut=cut, method=method,
                                                    binning_col=binning_col, binning_set=binning_set)
        output_score_cut.update(x_score_cut)
        
        #算空值率
        null_rate = temp_df.groupby(temp_df['dt_cut']).apply(lambda a: a[a['group'].str[-4:] == '.nan'].size/a.size)
        
        cnt = pd.crosstab(temp_df['group'],temp_df['dt_cut'])       
        pct = pd.crosstab(temp_df['group'],temp_df['dt_cut'], normalize='columns')
        # cnt.groupby(level=[0]).apply(lambda ii: ii/ii.sum()).unstack([0]).droplevel(0,axis=1).fillna(0.000001).replace(0,0.0001)
        
        A = pd.concat([(pct.iloc[:, i]-pct.iloc[:, 0])*np.log(pct.iloc[:, i]/pct.iloc[:, 0])
                       for i in range(pct.shape[1])],
                      axis=1)
        A.columns = pct.columns
        psi = A.sum().replace(np.inf, 1)

        if y == 'fake_y':
            tb1 = pd.DataFrame([null_rate,psi]).T
            tb1.columns = ['miss_rate', 'psi']
            tb1['variable'] = x
            tb1 = tb1.reset_index().set_index('variable') 
            
            tb2 = pd.concat([pct.unstack(), cnt.unstack()], axis=1)
            tb2.columns=['pct', 'len']
            tb2['variable'] = x
            tb2 = tb2.reset_index().set_index('variable')
        else:
            # 分组明细数据
            risk = temp_df.pivot_table(index=['group'], values=[y], columns='dt_cut',
                                       aggfunc=[len, np.sum, np.mean], fill_value=0).droplevel(1, axis=1)
            risk_temp = risk[risk.index.str[-4:] != '.nan'].copy()
            groupks = (np.cumsum(risk_temp['sum'])/risk_temp['sum'].sum() - np.cumsum(risk_temp['len']-risk_temp['sum'])/(risk_temp['len']-risk_temp['sum']).sum()).abs()
            risk = risk.stack().swaplevel(0, 1)
            
            # 算IV,KS
            iv_list = temp_df.groupby(temp_df['dt_cut']).apply(lambda k: cal_iv(k['group'], k[y])[0])
            try:
                ks_list = temp_df[temp_df['group'].str[-4:] != '.nan'].groupby(temp_df['dt_cut']).apply(
                    lambda k1: cal_ks(k1, x, y))
                auc_list = temp_df[temp_df['group'].str[-4:] != '.nan'].groupby(temp_df['dt_cut']).apply(
                    lambda k2: metrics.roc_auc_score(k2[y], k2[x]))
            except:
                ks_list = groupks.max()
                auc_list = pd.Series()
    
            # 算高低风险倍数
            lift = temp_df.groupby(['dt_cut', 'group']).agg({y: 'mean'})/temp_df.groupby(['dt_cut']).agg({y: 'mean'})
            lift = lift.unstack([0]).droplevel(0, axis=1).fillna(0)
            mono = lift[lift.index.str[-4:] != '.nan'].apply(monotonic_trend)
            bottom_lift = lift[lift.index.str[-4:] != '.nan'].iloc[0, :] # 非空组的第一组
            top_lift = lift[lift.index.str[-4:] != '.nan'].iloc[-1, :]# 非空组的最后一组
            
            tb1 = pd.DataFrame([null_rate, psi, iv_list, ks_list, auc_list, mono, bottom_lift, top_lift]).T
            tb1.columns=['miss_rate', 'psi', 'iv', 'ks', 'auc', 'monotonic', 'bottom_lift', 'top_lift']
            tb1['variable'] = x
            tb1 = tb1.reset_index().set_index('variable').rename({'index': 'dt_cut'}, axis=1)
            tb1['psi'] = tb1['psi'].replace(np.inf, 1)

            tb2 = pd.concat([pct.unstack(), risk, lift.unstack(), groupks.unstack()], axis=1)
            tb2.columns = ['pct', 'len', 'sum', 'mean', 'lift', 'ks']
            tb2['variable'] = x
            tb2 = tb2.reset_index().set_index('variable') 
            # tb2 = tb2[['group','dt_cut','len','pct','sum','mean','lift']]
        
        TB1 = pd.concat([TB1,tb1])
        TB2 = pd.concat([TB2,tb2])

    TB2['group'] = TB2['group'].apply(lambda x: beautify_str(x, digit=digit))
    for col in ['psi', 'miss_rate', 'iv', 'ks', 'auc', 'bottom_lift', 'top_lift']:
        TB1[col] = TB1[col].astype(float).round(digit)
    for col in ['pct', 'mean', 'lift', 'ks']:
        TB2[col] = TB2[col].astype(float).round(digit)
    for col in ['len', 'sum']:
        TB2[col] = TB2[col].astype(int)

    if output_path is None:
        return TB1,TB2,output_score_cut
    else:
        # 结果写入excel
        import xlwings as xw
        app = xw.App(visible=False, add_book=False)
        app.display_alerts=False
        app.screen_updating=True
        cat_desc,num_desc = full_describe(df)
        app = xw.App(visible=False, add_book=False)
        app.display_alerts=False
        app.screen_updating=True
        file_path = get_out_excel()
        wb = app.books.open(file_path)
        wb.sheets('字符型变量描述').range('a2').options(index=True,header=False).value = cat_desc
        wb.sheets('数值型变量描述').range('a2').options(index=True,header=False).value = num_desc
        wb.sheets('指标数据').range('a2').options(index=True,header=False).value = TB1
        wb.sheets('分组明细').range('a2').options(index=True,header=False).value = TB2
    
        wb.api.RefreshAll()
        print('报表已生成...')   
        wb.save(output_path)
        wb.close()
        app.kill()
        return TB1, TB2, output_score_cut


#%% 画图# 单马赛克图，
def plt_mosaic(df,inputx, y='fpd4', dt='event_date', miss_values=[-99], score_cut=None,
               method='quantile', digit=4, binning_col='binning', binning_set='1train',
               bs=3, if_plot=True, output_path=None):
    '''
    Parameters
    ----------
    df : pd.dataframe
        输入dataframe,必须dt.x
    inputx : list
        模型分\特征、变量 如['x1','x2'],字符型，数值型都可
    y : string
        样本的因变量 如 'fpd4', 因变量为0,1的int格式输入，y=None是，样本不含y标签.The default is 'fpd4'.
    dt : string, optional
        样本拆分的日期，一般我们建模中常用的，建案时间，交易时间等等。格式为 20190101、2019-01-01均可. The default is 'event_date'.
    miss_values : list
        缺失值列表，缺失值单独一箱 The default is [-99].
    score_cut : list, optional
        默认None,按照method方法分箱，支持字符和数值格式
        = int, 特征等频分箱的组数，如等频分为10箱. 
        = list 可自定义模型分切点 如：[547.0,570.0,584.0,600.0, 614.0, 626.0, 639.0, 654.0]
        注意：自定义切点时，inputx 只能放自定义切点的这个变量。
    method : string
        特征分箱方法，有'optb','auto'两种，前者是optbinning分箱方法，后者是自动等频分箱。均支持字符特征的自动分箱。
        The default is 'optb'.
    ----------------------------------------------------新增参数----------------------------------------------------------------------
    bs : int
        放大倍数，画马赛克图有时因y占比太少，导致画图不清，这里放大橙色部分方便看图 默认=3
    if_plot : bool 
        默认True, 默认直接画图 
    ---------------------------------------------------------------------------------------------------------------------------------
    output_path : 存储地址
        默认None 不存图，仅展示图片 

    Returns
    -------
    out_vardf : dataframe
        inputx 按跨时dt_cut的 miss_rate, psi (y=None是有), iv, ks, auc, bottom_lift, top_lift, monotonic(y=None是无)
    out_bindf : dataframe
        inputx 按跨时dt_cut，分箱每一箱的 miss_rate, len,bad,mean,lift.

    '''
    cross_vardf, cross_bindf, _ = out_all_in_one(df, inputx, y=y, dt=dt, dt_cut=3, miss_values=miss_values,
                                                 score_cut=score_cut, method=method,
                                                 digit=digit, binning_col=binning_col, binning_set=binning_set,
                                                 output_path=None)
    vardf, bindf, out_cuts = out_all_in_one(df, inputx, y=y, dt=dt, dt_cut=1, miss_values=miss_values,
                                            score_cut=score_cut, method=method,
                                            digit=digit, binning_col=binning_col, binning_set=binning_set,
                                            output_path=None)
    
    max_psi = cross_vardf.groupby(cross_vardf.index)[['psi']].max()
    min_iv = cross_vardf.groupby(cross_vardf.index)[['iv']].min()
    
    if if_plot:
        for x in inputx:
            the_bins_df = bindf[bindf.index == x].set_index('group')
            the_var_df = vardf[vardf.index == x]

            max_bs = the_bins_df[the_bins_df['group'].str[-4:] != '.nan']['lift'].iloc[-1]  # 最后一箱的风险
            max_pct = the_bins_df[the_bins_df['group'].str[-4:] != '.nan']['pct'].iloc[-1]  # 最后一箱的占比
            max_badn = the_bins_df[the_bins_df['group'].str[-4:] != '.nan']['sum'].iloc[-1]  # 最后一箱的坏客户数

            min_bs = the_bins_df[the_bins_df['group'].str[-4:] != '.nan']['lift'].iloc[0]  # 第一箱的风险
            min_pct = the_bins_df[the_bins_df['group'].str[-4:] != '.nan']['pct'].iloc[0]  # 第一箱的占比
            min_badn = the_bins_df[the_bins_df['group'].str[-4:] != '.nan']['sum'].iloc[0]  # 第一箱的坏客户数
                
            bad_rate = the_bins_df['mean']*bs
            the_pct = the_bins_df['pct']
            x_pos =[0]+ list(np.cumsum(the_pct))[:-1]
            x_label_pos = []
            for j in range(len(list(the_pct))):
                if j ==0:
                    x_label_pos.append(round(list(the_pct)[j]/2,4))
                else:
                    x_label_pos.append(round((sum(list(the_pct)[:j])+list(the_pct)[j]/2),4))       
            ax = plt.subplot()
            ax.text(0, 0.9, " 首:{:.1f}倍,{}个,占比{:.1%}; \n 尾:{:.1f}倍,{}个,占比{:.1%}".format(
                min_bs,min_badn,min_pct,max_bs,max_badn,max_pct),  fontdict={'size': '11', 'color': 'b'}) # 写平均风险值
            ax.set_title('{} 样本：{}/{}({:.2%}), \n dt:{} \n IV:{:.2f}, KS:{:.2f}, PSI:{:.2f}, minIV:{:.2f}'.format(
                x, df.shape[0], df[y].sum(), df[y].mean(),
                vardf.dt_cut.min(), the_var_df.iv.values[0], the_var_df.ks.values[0],
                max_psi[max_psi.index == x].psi.iloc[0], min_iv[min_iv.index == x].iv.iloc[0]),
                size=12) #表标题

            ax.bar(x_pos, [1]*len(bad_rate), width=the_pct, align='edge', ec='w', lw=1, label='N', color='silver', alpha=0.8)
            ax.bar(x_pos, bad_rate, width=the_pct, align='edge', ec='w', lw=1, label='Y', color='coral', alpha=0.8)
            ax.axhline(y=df[y].mean()*bs, color='r', label='warning line1', linestyle='--', linewidth=1)
            ax.text(0.95, df[y].mean()*bs+0.05, 'avg:'+str(round(df[y].mean()*100, 1)), ha='center', fontsize=9, color='r')
            ax.set_xticks(x_label_pos)
            # ax.set_yticks(y_label_pos)
            ax.set_xticklabels([str(xj) for xj in list(the_bins_df.index)], rotation=90, fontsize=10)
            # ax.set_yticklabels(['Y','N'])    
            for n1, n2, n3 in zip(x_label_pos,bad_rate, [i/bs for i in bad_rate]):
                ax.text(n1, n2/2, str(round(n3*100, 1)), ha='center', fontsize=9)
                
            if output_path==None:
                plt.show()
            else:   
                if os.path.exists(output_path):
                    pass
                else:
                    os.makedirs(output_path)
                plt.savefig('{}/{}_{}_{:.3f}.png'.format(output_path,
                                                         re.sub(r"[\/\\\:\*\?\"\<\>\|]", '', x),
                                                         the_var_df.loc[x,'monotonic'],
                                                         the_var_df.loc[x,'iv']
                                                        )),
                plt.close()
    return vardf, bindf, out_cuts


#%%
# 适用于变量，字段； 字符型、数值型变量均会自动分箱
def plt_multi_mosaic(df, inputx, y='fpd4', dt='event_date', dt_cut='month', miss_values=[-99], score_cut=10,
                     method='quantile', digit=4, binning_col='binning', binning_set='1train',
                     bs=3, if_plot=True, output_path=None):
    '''
    Parameters
    ----------
    df : pd.dataframe
        输入dataframe,必须dt.x
    inputx : list
        模型分\特征、变量 如['x1','x2'],字符型，数值型都可
    y : string
        样本的因变量 如 'fpd4', 因变量为0,1的int格式输入，y=None是，样本不含y标签.The default is 'fpd4'.
    dt : string, optional
        样本拆分的日期，一般我们建模中常用的，建案时间，交易时间等等。格式为 20190101、2019-01-01均可. The default is 'event_date'.
    dt_cut : list,int,string optional
        =list 时 如：[-np.inf,20190101,20190601,20190901,np.inf]
        ='month' 或者指定df中的某一列 如'shouxin_month','month1'
        =1,2,3 样本等分成3份。。 
        The default is 'month', 不管样本中是否包含month列，都可以直接按月拆分.
    miss_values : list
        缺失值列表，缺失值单独一箱 The default is [-99].
    score_cut : list, optional
        默认值10, 特征等频分箱的组数，等频分为10箱. 也可自定义模型分切点 如：[547.0,570.0,584.0,600.0, 614.0, 626.0, 639.0, 654.0]
        注意：自定义切点时，inputx 只能放自定义切点的这个变量。
    method : string
        特征分箱方法，有'optb','auto'两种，前者是optbinning分箱方法，后者是自动等频分箱。均支持字符特征的自动分箱。
        The default is 'optb'.
    bs : int
        放大倍数，画马赛克图有时因y占比太少，导致画图不清，这里放大橙色部分方便看图 默认=3
    if_plot : bool 
        默认True, 默认直接画图 
    output_path : 存储地址
        默认None 不存图，仅展示图片 

    Returns
    -------
    out_vardf : dataframe
        inputx 按跨时dt_cut的 miss_rate, psi (y=None是有), iv, ks, auc, bottom_lift, top_lift, monotonic(y=None是无)
    out_bindf : dataframe
        inputx 按跨时dt_cut，分箱每一箱的 miss_rate, len,bad,mean,lift.

    '''
    cross_vardf, cross_bindf, _ = out_all_in_one(df, inputx, y=y, dt=dt, dt_cut=dt_cut, miss_values=miss_values,
                                                 score_cut=score_cut, method=method,
                                                 digit=digit, binning_col=binning_col, binning_set=binning_set,
                                                 output_path=None)
    vardf, bindf, out_cuts = out_all_in_one(df, inputx, y=y, dt=dt, dt_cut=1, miss_values=miss_values,
                                            score_cut=score_cut, method=method,
                                            digit=digit, binning_col=binning_col, binning_set=binning_set,
                                            output_path=None)
    dt_range = str(df[dt].min()) + '~' + str(df[dt].max())
    vardf['dt_cut'] = dt_range
    bindf['dt_cut'] = dt_range

    tot_size = df.shape[0]

    out_vardf = pd.concat([vardf, cross_vardf])
    out_bindf = pd.concat([bindf, cross_bindf])

    if if_plot:
        for x in inputx:
            tot_missing = vardf.loc[x, 'miss_rate']
            tot_iv = vardf.loc[x, 'iv']
            tot_ks = vardf.loc[x, 'ks']
            tot_mono = vardf.loc[x, 'monotonic']

            c = cross_bindf.loc[x].reset_index().set_index('dt_cut')
            the_cross_vardf = cross_vardf.loc[x].reset_index().set_index('dt_cut')

            cross_len = c.groupby(c.index)['len'].sum()
            cross_sum = c.groupby(c.index)['sum'].sum()
            cross_mean = cross_sum / cross_len

            # 画分时段马赛克图
            l = sorted(set(cross_vardf['dt_cut']))

            fig = plt.figure(figsize=(5 * len(l), 4))
            fig.suptitle('{},{}, 总样本量：{}, 时间跨度:{}\n\n总IV:{:.2f}, 总KS:{:.2f}, 总空值率:{:.2%}, 总趋势:{}'.format(
                x, y, tot_size, dt_range, tot_iv, tot_ks, tot_missing, tot_mono),
                x=0.1, y=1.2, ha='left', size=15, bbox=dict(facecolor='grey', alpha=0.1))

            for k in l:
                i = l.index(k)
                max_bs = c[c['group'].str[-4:] != '.nan']['lift'][k].iloc[-1]  # 最后一箱的风险
                max_pct = c[c['group'].str[-4:] != '.nan']['pct'][k].iloc[-1]  # 最后一箱的占比
                max_badn = c[c['group'].str[-4:] != '.nan']['sum'][k].iloc[-1]  # 最后一箱的坏客户数

                min_bs = c[c['group'].str[-4:] != '.nan']['lift'][k].iloc[0]  # 第一箱的风险
                min_pct = c[c['group'].str[-4:] != '.nan']['pct'][k].iloc[0]  # 第一箱的占比
                min_badn = c[c['group'].str[-4:] != '.nan']['sum'][k].iloc[0]  # 第一箱的坏客户数
                bad_rate = [ic if ic * bs > 1 else ic * bs for ic in c['mean'][k]]
                the_pct = c['pct'][k]
                x_pos = [0] + list(np.cumsum(the_pct))[:-1]
                x_label_pos = []
                for j in range(len(list(the_pct))):
                    if j == 0:
                        x_label_pos.append(round(list(the_pct)[j] / 2, 4))
                    else:
                        x_label_pos.append(round((sum(list(the_pct)[:j]) + list(the_pct)[j] / 2), 4))
                ax = plt.subplot(1, len(l), i + 1)
                ax.text(0, 0.9,
                        " 首:{:.1f}倍,{}个,占比{:.1%}; \n 尾:{:.1f}倍,{}个,占比{:.1%}".format(
                            min_bs, min_badn, min_pct, max_bs, max_badn, max_pct),
                        fontdict={'size': '11', 'color': 'b'})  # 写平均风险值
                ax.set_title(' {}, 样本：{}/{}({:.2%}), \n IV:{:.2f}, KS:{:.2f}, PSI:{:.2f}'.format(
                    str(k), cross_len[k], cross_sum[k], cross_mean[k], the_cross_vardf['iv'][k],
                    the_cross_vardf['ks'][k], the_cross_vardf['psi'][k]), size=12)  # 表标题

                ax.bar(x_pos, [1] * len(bad_rate), width=the_pct, align='edge', ec='w', lw=1, label='N',
                       color='silver', alpha=0.8)
                ax.bar(x_pos, bad_rate, width=the_pct, align='edge', ec='w', lw=1, label='Y', color='coral', alpha=0.8)
                ax.axhline(y=cross_mean[k] * bs, color='r', label='warning line1', linestyle='--', linewidth=1)
                ax.text(0.95, cross_mean[k] * bs + 0.05, 'avg:' + str(round(cross_mean[k] * 100, 1)), ha='center',
                        fontsize=9, color='r')
                ax.set_xticks(x_label_pos)
                # ax.set_yticks(y_label_pos)
                ax.set_xticklabels([str(xj) for xj in list(c['group'].unique())], rotation=90, fontsize=10)
                # ax.set_yticklabels(['Y','N'])
                for n1, n2, n3 in zip(x_label_pos, bad_rate, [i / bs for i in bad_rate]):
                    ax.text(n1, n2 / 2, str(round(n3 * 100, 1)), ha='center', fontsize=9)

            # 存图
            if output_path == None:
                plt.show()
            else:
                if os.path.exists(output_path):
                    pass
                else:
                    os.makedirs(output_path)
                fig.savefig('{}/{}_{}_{:.3f}.png'.format(output_path,
                                                         re.sub(r"[\/\\\:\*\?\"\<\>\|]", '', x),
                                                         tot_mono,
                                                         tot_iv
                                                         ),
                            pad_inches=0.3, dpi=100, papertype='a4', bbox_inches='tight')
                plt.close()

    return out_vardf,out_bindf


#%%
# 该函数使用于模型分时看风险趋势
def plt_multi_rsk_trend(df, inputx, y='fpd4', dt='event_date', dt_cut='month', miss_values=[-99], score_cut=10,
                        method='quantile', digit=4, binning_col='binning', binning_set='1train',
                        if_plot=True, output_path=None):
    '''
    Parameters
    ----------
    df : pd.dataframe
        输入dataframe,必须dt.x
    inputx : list
        模型分\特征、变量 如['x1','x2'],字符型，数值型都可
    y : string
        样本的因变量 如 'fpd4', 因变量为0,1的int格式输入，y=None是，样本不含y标签.The default is 'fpd4'.
    dt : string, optional
        样本拆分的日期，一般我们建模中常用的，建案时间，交易时间等等。格式为 20190101、2019-01-01均可. The default is 'event_date'.
    ----------------------------------------------------新增参数----------------------------------------------------------------------
    dt_cut : list,int,string optional
        =list 时 如：[-np.inf,20190101,20190601,20190901,np.inf]
        ='month' 或者指定df中的某一列 如'shouxin_month','month1'
        =1,2,3 样本等分成3份。。 
        The default is 'month', 不管样本中是否包含month列，都可以直接按月拆分.
    ---------------------------------------------------------------------------------------------------------------------------------
    miss_values : list
        缺失值列表，缺失值单独一箱 The default is [-99].
    score_cut : list, optional
        默认10,特征等频分箱的组数，如等频分为10箱. 
        = None, 按照method方法分箱，支持字符和数值格式
        = list 可自定义模型分切点 如：[547.0,570.0,584.0,600.0, 614.0, 626.0, 639.0, 654.0]
        注意：自定义切点时，inputx 只能放自定义切点的这个变量。
    method : string
        特征分箱方法，有'optb','auto'两种，前者是optbinning分箱方法，后者是自动等频分箱。均支持字符特征的自动分箱。
        The default is 'optb'.
    if_plot : bool 
        默认True, 默认直接画图 
    output_path : 存储地址
        默认None 不存图，仅展示图片 

    Returns
    -------
    out_vardf : dataframe
        inputx 按跨时dt_cut的 miss_rate, psi (y=None是有), iv, ks, auc, bottom_lift, top_lift, monotonic(y=None是无)
    out_bindf : dataframe
        inputx 按跨时dt_cut，分箱每一箱的 miss_rate, len,bad,mean,lift.
    out_cuts : np.array
        inputx 切点
    '''
    cross_vardf,cross_bindf,_ = out_all_in_one(df,inputx,y=y,dt =dt,dt_cut=dt_cut,miss_values=miss_values,
                                               score_cut=score_cut,method=method,
                                               digit=digit,binning_col=binning_col,binning_set=binning_set,output_path=None)
    vardf,bindf,out_cuts = out_all_in_one(df,inputx,y=y,dt =dt,dt_cut=1,miss_values=miss_values,
                                                   score_cut=score_cut,method=method,
                                                   digit=digit,binning_col=binning_col,binning_set=binning_set,output_path=None)
    dt_range = str(df[dt].min()) + '~' + str(df[dt].max())
    vardf['dt_cut'] = dt_range
    bindf['dt_cut'] = dt_range
    
    tot_size = df.shape[0]
    
    out_vardf = pd.concat([vardf, cross_vardf])
    out_bindf = pd.concat([bindf, cross_bindf])

    if if_plot:
        if y is not None:
            for x in inputx:
                tot_missing = vardf.loc[x, 'miss_rate']
                tot_iv = vardf.loc[x, 'iv']
                tot_ks = vardf.loc[x, 'ks']
                tot_auc = vardf.loc[x, 'auc']
                tot_mono = vardf.loc[x, 'monotonic']
                
                c = cross_bindf.loc[x].reset_index().set_index('dt_cut')
                
                if cross_vardf.shape[0]==1:
                    the_cross_vardf = cross_vardf.loc[x].to_frame().T.reset_index().set_index('dt_cut')
                else:
                    the_cross_vardf = cross_vardf.loc[x].reset_index().set_index('dt_cut')
                
                cross_len = c.groupby(c.index)['len'].sum()
                cross_sum = c.groupby(c.index)['sum'].sum()
                cross_mean = cross_sum/cross_len
    
                #画分时段马赛克图        
                l = sorted(set(cross_vardf['dt_cut']))
        
                fig = plt.figure(figsize=(5*len(l),4))
                fig.suptitle('{},{}, 总样本量：{}, 时间跨度:{}\n\n总IV:{:.2f}, 总KS:{:.2f}, 总AUC:{:.2f}, 总空值率:{:.2%}, 总趋势:{}'.format(
                    x, y, tot_size, dt_range, tot_iv, tot_ks, tot_auc, tot_missing, tot_mono),
                    x=0.1, y=1.2, ha='left', size=15, bbox=dict(facecolor='grey', alpha=0.1))
            
                for k in l:  
                    i = l.index(k)
                    max_bs = c[c['group'].str[-4:] != '.nan']['lift'][k].iloc[-1] # 最后一箱的风险
                    max_pct = c[c['group'].str[-4:] != '.nan']['pct'][k].iloc[-1]# 最后一箱的占比
                    max_badn = c[c['group'].str[-4:] != '.nan']['sum'][k].iloc[-1]# 最后一箱的坏客户数

                    min_bs = c[c['group'].str[-4:] != '.nan']['lift'][k].iloc[0] # 第一箱的风险
                    min_pct = c[c['group'].str[-4:] != '.nan']['pct'][k].iloc[0]# 第一箱的占比
                    min_badn = c[c['group'].str[-4:] != '.nan']['sum'][k].iloc[0]# 第一箱的坏客户数

                    # 柱状图
                    the_pct = c['pct'][k]
                    the_ks = c['ks'][k]
                    color = ['b' if ia >= the_ks.max() and ia>0 else 'k' for ia in the_ks]
                    xlabel = [str(x) for x in list(c['group'].unique())]
                    xlabel = [ib[:8]+'..'+ib[-8:] if len(ib) > 50 else ib for ib in xlabel]
                    ax1 = plt.subplot(1, len(l), i+1)
                    ax1.set_xticklabels(xlabel, rotation=90, fontsize=10)
                    ax1.bar(xlabel, the_pct, alpha=0.2, color=color)
                    # sns.barplot(xlabel, the_pct,ax = ax1,alpha=0.2,color = color)
                    ax1.text(0.01, 0.88, "首:{:.1f}倍,{}个,占{:.1%};  \n尾:{:.1f}倍,{} 个,占{:.1%}".format(
                        min_bs, min_badn, min_pct, max_bs, max_badn, max_pct),
                             transform=ax1.transAxes, fontdict={'size': '11', 'color': 'b'}) # 写平均风险值
                    ax1.set_title(' {}, 样本：{}/{}({:.2%}), \n IV:{:.2f}, KS:{:.2f}, AUC:{:.2f}, PSI:{:.2f}, miss:{:.0%}'.format(
                        str(k), cross_len[k], cross_sum[k], cross_mean[k], the_cross_vardf['iv'][k],
                        the_cross_vardf['ks'][k], the_cross_vardf['auc'][k],
                        the_cross_vardf['psi'][k], the_cross_vardf['miss_rate'][k]),
                        size=12) #表标题
                    ax1.set_ylim([0, c['pct'][c['pct']<0.5].max()*1])
                    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
                    ax1.axes.set_ylabel('')
                    ax1.tick_params(axis='y', labelcolor='#696969')
                    
                    # badrate 曲线
                    bad_rate = c['mean'][k]
                    frsk = bad_rate.copy()
                    frsk[c['group'][k].str[-4:] == '.nan'] = np.nan
                    # if the_cross_vardf['miss_rate'].max()>0:
                    #     frsk.iloc[-1] = np.nan
                    ax2 = ax1.twinx()
                    ylim_max = c['mean'].max()
                    sns.pointplot(xlabel, frsk, ax=ax2, alpha=0.2, color='red', scale=0.5)
                    for a,b in zip([i for i in range(len(xlabel))], frsk):
                        ax2.annotate("{:.1%}".format(b), xy=(a,b), xytext=(-12, 5), textcoords='offset points', weight='heavy')
                    ax2.axhline(y=cross_mean[k], color='grey', label='avg_risk', linestyle='--', linewidth=1) # 平均风险线
                    ax2.set_ylim([0, ylim_max*1.2])
                    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))
                    ax2.axes.set_ylabel('')
                    ax2.tick_params(axis='y', labelcolor='#A52A2A')
                    
                     # 空箱badrate 打点
                    if the_cross_vardf['miss_rate'][k] > 0:
                        ax3 = ax1.twinx()
                        frsk = bad_rate.copy()
                        frsk[c['group'][k].str[-4:] != '.nan'] = np.nan
                        frsk[frsk >= ylim_max] = ylim_max * 1.
                        sns.pointplot(xlabel, frsk, ax=ax3, alpha=0.2, color='blue', scale=0.6)  # 画在ax3画布上
                        for a, b in zip(np.arange(len(xlabel)), frsk):
                            if not np.isnan(b):
                                ax3.annotate("{:.1%}".format(b), xy=(a,b), xytext=(-8, 5),
                                             textcoords='offset points', weight='heavy', color='blue')
                        
                        ax3.set_ylim([0, ylim_max*1.2])
                        ax3.yaxis.set_ticklabels([]) 
                        ax3.axes.set_ylabel('')
                if output_path is None:
                    plt.show()
                else:
                    if os.path.exists(output_path):
                        pass
                    else:
                        os.makedirs(output_path)
                    fig.savefig('{}/{}.png'.format(output_path,
                                                   re.sub(r"[\/\\\:\*\?\"\<\>\|]", '', x)),
                                pad_inches=0.3, dpi=100, papertype='a4', bbox_inches='tight')
                    plt.close()
        else:
            for x in inputx:
                tot_missing = vardf.loc[x, 'miss_rate']
                c = cross_bindf.loc[x].reset_index().set_index('dt_cut')
                if cross_vardf.shape[0] == 1:
                    #Dataframe如果只提取一行则会自动转换为Series，因此需要再转换
                    the_cross_vardf = cross_vardf.loc[x].to_frame().T.reset_index().set_index('dt_cut')
                else:
                    the_cross_vardf = cross_vardf.loc[x].reset_index().set_index('dt_cut')
                cross_len = c.groupby(c.index)['len'].sum()
    
                #画分时段风险趋势图       
                l = sorted(set(cross_vardf['dt_cut']))
        
                fig = plt.figure(figsize=(5*len(l), 4))
                fig.suptitle('{}, 总样本量：{}, 时间跨度:{}\n\n总空值率:{:.2%}'.format(
                    x, tot_size, dt_range, tot_missing),
                    x=0.1, y=1.2, ha='left', size=15, bbox=dict(facecolor='grey', alpha=0.1))
            
                for k in l:  
                    i = l.index(k)        
                    # 柱状图
                    the_pct = c['pct'][k]
                    xlabel = [str(xj) for xj in list(c['group'].unique())]
                    xlabel = [ia[:8]+'..'+ia[-8:] if len(ia) > 50 else ia for ia in xlabel]
                    ax1 = plt.subplot(1, len(l), i+1)
                    ax1.set_xticklabels(xlabel, rotation=90, fontsize=10)
                    sns.barplot(xlabel, the_pct, ax=ax1, alpha=0.2, color='k')
                    ax1.set_title(' {}, 样本：{}, \n PSI:{:.2f}, miss:{:.0%}'.format(
                        str(k), cross_len[k], the_cross_vardf['psi'][k], the_cross_vardf['miss_rate'][k]),
                        size=12) #表标题
                    ax1.set_ylim([0, c['pct'][c['pct']<0.5].max()*1])
                    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
                    ax1.axes.set_ylabel('')
                    ax1.tick_params(axis='y', labelcolor='#696969')

                if output_path is None:
                    plt.show()
                else:
                    if os.path.exists(output_path):
                        pass
                    else:
                        os.makedirs(output_path)
                    fig.savefig('{}/{}.png'.format(output_path,
                                                   re.sub(r"[\/\\\:\*\?\"\<\>\|]", '', x)),
                                pad_inches=0.3, dpi=100, papertype='a4',bbox_inches='tight')
                    plt.close()
        
    return out_vardf, out_bindf, out_cuts


def expand_stat(df, digit=4):
    def cum_stat(df2):
        base = df2['p'].sum() / df2['count'].sum()
        df2['count_cum'] = df2['count'].cumsum()
        df2['p_cum'] = df2['p'].cumsum()
        df2['n_cum'] = df2['n'].cumsum()
        df2['p_rate_cum'] = df2['p_cum'] / df2['count_cum']
        df2['n_rate_cum'] = df2['n_cum'] / df2['count_cum']
        df2['odd_cum'] = df2['p_rate_cum'] / df2['n_rate_cum']
        df2['lift_cum'] = df2['p_rate_cum'] / base
        return df2
    df = df.reset_index()
    df = df.rename(columns={'pct': 'count_ratio', 'len': 'count', 'sum': 'p', 'mean': 'p_rate'})
    df['n'] = df['count'] - df['p']
    df['n_rate'] = df['n'] / df['count']
    df['odd'] = df['p_rate'] / df['n_rate']
    df = df.groupby(['variable', 'dt_cut']).apply(cum_stat).set_index('variable')
    df = df[['dt_cut', 'group', 'count', 'count_ratio', 'p', 'p_rate', 'n', 'n_rate', 'odd', 'lift', 'ks',
            'count_cum', 'p_cum', 'n_cum', 'p_rate_cum', 'n_rate_cum', 'odd_cum', 'lift_cum']]
    for col in ['count_ratio', 'p_rate', 'n_rate', 'odd', 'lift', 'ks',
                'p_rate_cum', 'n_rate_cum', 'odd_cum', 'lift_cum']:
        df[col] = df[col].astype(float).round(digit)
    return df


def check_score_cut(score_cut):
    for key in score_cut.keys():
        cut = np.array(score_cut[key])
        cut[np.isnan(cut)] = -99
        cut[cut == -99] = -99 + 0.1
        score_cut[key] = cut
    return score_cut


def set_column_width(df, worksheet, offset=0):
    for idx, col in enumerate(df):
        series = df[col]
        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
        )) + 1  # adding a little extra space
        worksheet.set_column(idx+offset, idx+offset, max_len)  # set column width
    return worksheet


range_color = ['#16934d', '#57b55f', '#93d067', '#c7e77f', '#edf7a7',
               '#fef1a6', '#fdce7b', '#fa9a57', '#ee613d', '#d22b26']
def color_pct(val):
    num_range_color = len(range_color)
    color = range_color[val % num_range_color]
    return 'background-color: %s' % color


def format_null(df):
    df = (df.style.applymap(color_pct)
          .format("{:.2%}").set_properties(**{'font-family': 'Calibri',
                                              'font-size': '12px', 'max-width': '10px',
                                              'border-color': 'grey', 'border-style': 'solid',
                                              'border-width': '0.05px', 'border-collaps': 'collaps'}))

    return df


def write_report(df, models_cut_dict, vars_cut_dict, y, dt_cut, digit, mode,
                 binning_col, binning_set, output_path):
    """
    vars_cut_dict = {'apply_rate_n_90': [25, 77, np.nan],
                     'apply_rate_n_90_success': [25, 77, np.nan]}
    models_cut_dict = {y: {'apply_rate_n_90': 5, 'apply_rate_n_90_success': 3}}
    write_report(df, models_cut_dict, vars_cut_dict, y, dt_cut='set', digit=4, mode='bivar',
                 binning_col='set', binning_set='1train', output_path='myreport')
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    writer = pd.ExcelWriter(os.path.join(output_path, y+'_report.xlsx'), engine='xlsxwriter')
    if not os.path.exists(os.path.join(output_path, 'var')):
        os.makedirs(os.path.join(output_path, 'var'))
    temp_df = df[df[y].isin([0, 1])].copy()
    if mode == 'mosaic':
        var_stat, var_bin_stat = plt_multi_mosaic(temp_df,
                                                  list(vars_cut_dict.keys()),
                                                  y=y,
                                                  dt=dt_cut,
                                                  dt_cut=dt_cut,
                                                  miss_values=[-99, np.nan],
                                                  score_cut=vars_cut_dict,
                                                  digit=digit,
                                                  binning_col=binning_col,
                                                  binning_set=binning_set,
                                                  if_plot=True,
                                                  output_path=os.path.join(output_path, 'var')
                                                  )
    else:
        var_stat, var_bin_stat, _ = plt_multi_rsk_trend(temp_df,
                                                        list(vars_cut_dict.keys()),
                                                        y=y,
                                                        dt=dt_cut,
                                                        dt_cut=dt_cut,
                                                        miss_values=[-99, np.nan],
                                                        score_cut=vars_cut_dict,
                                                        digit=digit,
                                                        binning_col=binning_col,
                                                        binning_set=binning_set,
                                                        if_plot=True,
                                                        output_path=os.path.join(output_path, 'var')
                                                        )
    var_stat = var_stat.reset_index()
    var_bin_stat = var_bin_stat.reset_index()
    format_null(var_stat['variable'].rank(method='dense').astype(int).rename('num').to_frame()).to_excel(
        writer, sheet_name='var_stat', startcol=0, index=False)
    var_stat.to_excel(writer, sheet_name='var_stat', startcol=1, index=False)
    sheet = writer.sheets['var_stat']
    set_column_width(var_stat, sheet, 1)
    format_null(var_bin_stat['variable'].rank(method='dense').astype(int).rename('num').to_frame()).to_excel(
        writer, sheet_name='var_bin_stat', startcol=0, index=False)
    var_bin_stat.to_excel(writer, sheet_name='var_bin_stat', startcol=1, index=False)
    sheet = writer.sheets['var_bin_stat']
    set_column_width(var_bin_stat, sheet, 1)
    for label in models_cut_dict.keys():
        if not os.path.exists(os.path.join(output_path, label)):
            os.makedirs(os.path.join(output_path, label))
        temp_df = df[df[label].isin([0, 1])].copy()
        model_stat, model_bin_stat, _ = plt_multi_rsk_trend(temp_df,
                                                            list(models_cut_dict[label].keys()),
                                                            y=label,
                                                            miss_values=[-99, np.nan],
                                                            dt=dt_cut,
                                                            dt_cut=dt_cut,
                                                            score_cut=models_cut_dict[label],
                                                            digit=digit,
                                                            binning_col=binning_col,
                                                            binning_set=binning_set,
                                                            if_plot=True,
                                                            output_path=os.path.join(output_path, label)
                                                            )
        model_stat = model_stat.reset_index()
        model_bin_stat = expand_stat(model_bin_stat, digit)
        model_bin_stat = model_bin_stat.reset_index()
        format_null(model_stat['variable'].rank(method='dense').astype(int).rename('num').to_frame()).to_excel(
            writer, sheet_name=label+'_model_stat', startcol=0, index=False)
        model_stat.to_excel(writer, sheet_name=label+'_model_stat', startcol=1, index=False)
        sheet = writer.sheets[label+'_model_stat']
        set_column_width(model_stat, sheet, 1)
        format_null(model_bin_stat['variable'].rank(method='dense').astype(int).rename('num').to_frame()).to_excel(
            writer, sheet_name=label+'_model_bin_stat', startcol=0, index=False)
        model_bin_stat.to_excel(writer, sheet_name=label+'_model_bin_stat', startcol=1, index=False)
        sheet = writer.sheets[label+'_model_bin_stat']
        set_column_width(model_bin_stat, sheet, 1)
        sheet = writer.sheets[label+'_model_stat']
        for i, model_name in enumerate(models_cut_dict[label].keys()):
            sheet.insert_image(row=model_stat.shape[0]+1+30*i, col=0,
                               filename=os.path.join(output_path, label, model_name + '.png'))
    writer.save()
