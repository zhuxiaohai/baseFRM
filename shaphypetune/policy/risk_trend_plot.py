import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
from sklearn import metrics
from scipy.stats import ks_2samp
from matplotlib.pyplot import style
style.use('seaborn-white')
import pylab
pylab.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
pylab.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import warnings
warnings.filterwarnings('ignore')

def risk_trend_plot(dt_test, result_test, col_y, n, trend_title='risk_trend', plot_save_path='-', bins='freq',if_plot=True):
    '''模型分趋势图画图插件

    Parameters
    ----------
    dt_test:pd.DataFrame
        分析样本
    result_test:list
        测试集模型分数的列名
    col_y:str
        因变量在数据集中的列名，例如：'t3_k11' 
    n:int
        分成几组 如：10
    plot_save_path:str
        趋势图的保存路径，例如: '测试集分组响应趋势图'
    trend_title:str
        趋势图名称
    bins:str
        分组方法,传入'freq'或'dist'或一个列表\
                'freq' 做等频分组； 'dist' 做等距分组； 列表中为从小到大排序的切点值，以这些切点分组(需包含极小值和极大值)
    if_plot:bool
        是否画图
       
    Returns
    -------
    temp_df:pd.DataFrame
        每个分组客户的具体信息：包括：响应客户、未响应客户的数量，响应率，人数
    bins:list
        本次画图客户分组的节点

    '''

    dt_test['result'] = dt_test[result_test]
    if bins =='freq':
        bins = sorted(list(set(dt_test["result"].quantile([(i+1)/n for i in range(n-1)])))+[-9999, 9999])
    elif bins =='dist':
        bins = sorted(list(set(np.linspace(dt_test["result"].min(),dt_test["result"].max(),num=n+1)[1:-1]))+[-9999,9999])
    cut_label = 'cut_label'
    dt_test[cut_label] = pd.cut(dt_test['result'], bins=bins)


    # 画图开始
    result_plot = pd.crosstab(dt_test[col_y], dt_test[cut_label]).T
    result_plot = result_plot.rename(columns={0: 'No', 1: 'Yes'})
    result_plot = result_plot.reindex(columns=['Yes', 'No'])
    temp_df = result_plot.assign(
        risk_ratio=lambda x: x.Yes / (x.Yes + x.No + 0.001),
        group_size=lambda x: (x.Yes + x.No)
        # ,risk_amount=lambda x: x['Yes'].groupby(x.index).cumsum()
    ).reset_index()
    temp_df["cum_Yes_cnt"]=temp_df["Yes"].cumsum()
    temp_df["cum_group_size"]=temp_df["group_size"].cumsum()
    temp_df["AR"]=temp_df["cum_group_size"]/temp_df["group_size"].sum()
    temp_df["cum_yes_rate"]=temp_df["cum_Yes_cnt"]/temp_df["cum_group_size"].replace(0,1)
    
    if if_plot:
        sns.set_style("dark")
    
        fig = plt.figure(figsize=[10, 5])  # 设定图片大小
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        ax1 = fig.add_subplot(111)  # 添加第一副图
        ax2 = ax1.twinx()  # 共享x轴，这句很关键！
        """
            画总人数随分组变化的柱状图
        """
        sns.barplot(x=cut_label, y='group_size', data=temp_df, ax=ax1, alpha=0.2, color="k")  # 画柱状图,注意ax是选择画布
    
        ax1.axes.set_ylabel(u"总人数", fontsize=15)
        ax1.axes.set_xticklabels(temp_df[cut_label], rotation=90)
    
        """
           画响应比例随分组变化的折线图
        """
        label = temp_df[cut_label]
        y = temp_df['risk_ratio']
        x = [i for i in range(len(label))]
        sns.pointplot(temp_df[cut_label], temp_df['risk_ratio'], ax=ax2, alpha=0.3)  # 画在ax2画布上
        for a, b in zip(x, y):
            plt.text(a, b, '%.2f%%' % (b * 100), ha='center', va='bottom', fontsize=12)  # 将数值显示在图形上
        ax2.axes.set_ylabel(u"响应比例", fontsize=15)  # 设置y坐标label
        right_axis_ceiling = temp_df['risk_ratio'].max() * 1.2
        ax2.set_ylim([0, right_axis_ceiling])  # 设置y轴取值范围
        ax2.axes.set_title(trend_title, fontsize=15)
        # 加平均响应
        avg_risk_ratio = round(dt_test[col_y].sum() / len(dt_test), 4)
        ax2.hlines(y=avg_risk_ratio, xmin=0, xmax=len(bins)-2, color='r', label='avg_risk', linestyle='--', linewidth=1)
    
        """
            xy=(横坐标，纵坐标)  箭头尖端
            xytext=(横坐标，纵坐标) 文字的坐标，指的是最左边的坐标
            arrowprops= {facecolor= '颜色',shrink = '数字' <1  收缩箭头 }
        """
        avg_risk_percentage = round(dt_test[col_y].sum() / len(dt_test), 4) * 100
        ax2.annotate('平均响应：%.2f%%' % avg_risk_percentage, xy=(1, avg_risk_ratio), xytext=(1.2, avg_risk_ratio * 1.1),
                     arrowprops=dict(facecolor='red', shrink=1))

        plt.show()
    if plot_save_path !='-':
        fig.savefig(plot_save_path, pad_inches=0.3, dpi=100, papertype='a4', bbox_inches='tight')  # 保存图片
        plt.close()  # 把图片释放掉，否则循环批量跑的时候保存图片会产生重叠
    return temp_df, bins


def feature_ana_plot(feature_ana_df,path=None,file_prefix=None,excel=None,sheet_name=None,plot_insert_excel=False,if_plot=True):
    '''特征趋势图画图插件

    Parameters
    ----------
    feature_ana_df:pd.DataFrame
        特征分析表，FeatureStablityTransformer、FeaturePsiStablityTransformer等特征稳定性插件生成
    path:None or str
        默认None，图像生成路径
    file_prefix:None or str
        默认None，生成图片文件名称带的前缀
    excel:None or str
        默认None，生成excel报告的excel名称
    sheet_name:None or str
        默认None，生成excel报告中的sheet名称
    plot_insert_excel:bool
        默认False，生成的plot文件插入excel中
    if_plot:bool
        是否画图
        
    Returns
    -------
    fig_list:list
        画图列表
    '''
    sns.set_style("dark")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    # fig = plt.figure(figsize=[10, 5])  # 设定图片大小
    fig_list=[]

    for col in set(feature_ana_df.variable.tolist()):
        print (col)
        fig = plt.figure(figsize=(8*len(feature_ana_df['data_parts'].unique()),4))
        for dp in enumerate(feature_ana_df['data_parts'].unique()):
            # print (dp)
            # feature_ana_df['data_parts'].unique()
            if 'bet' in feature_ana_df.columns.tolist():
                temp_df=feature_ana_df.loc[(feature_ana_df['variable']==col) &\
                                           (feature_ana_df['data_parts']==dp[1]),\
                                               ['count_distr','bin','avg_risk','bet','total_iv']]
            else:
                temp_df=feature_ana_df.loc[(feature_ana_df['variable']==col) &\
                                           (feature_ana_df['data_parts']==dp[1]),\
                                               ['count_distr','bin']]
                
            try:
                temp_df=temp_df.assign(sortkey=lambda x: [float(re.search(r'\[(.+),.*\)',i).group(1)) for i in x['bin']]).sort_values(by='sortkey')
            except:
                pass
            
            ax1 = plt.subplot(1,len(feature_ana_df['data_parts'].unique()),dp[0]+1)
            ax1.axes.set_xticklabels(temp_df['bin'], rotation=90)
            # ax1 = fig.add_subplot(111) 
            sns.barplot(x=temp_df['bin'], y=temp_df['count_distr'], data=temp_df, ax=ax1, alpha=0.2, color="k")
    
            for a, b in zip(np.arange(len(temp_df['bin'])), temp_df['count_distr']):
                plt.text(a, b, '%.1f%%' % (b * 100), ha='center', va='bottom', fontsize=12)  # 将数值显示在图形上
            # print("bet",'bet' in feature_ana_df.columns.tolist())
            if 'bet' in temp_df.columns.tolist() :
                if temp_df.bet.isnull().sum()==0:
                    ax2 = ax1.twinx()
                    sns.pointplot(x=temp_df['bin'], y=temp_df['bet'], ax=ax2, alpha=0.3)
                    right_axis_ceiling = feature_ana_df['bet'].max() * 1.2
                    ax2.set_ylim([0, right_axis_ceiling])
                    for c,d in zip(np.arange(len(temp_df['bin'])),temp_df['bet']):
                        plt.text(c,d,str(round(d,2)),ha='right',fontsize=12,color='mediumvioletred')
                    # print ("col:",col)
                    ax1.axes.set_title("{}|{}|{:.4f}".format(col,dp[1],temp_df.total_iv.values[0]), fontsize=10)
                else:
                    ax1.set_ylim([0, 1.05])
                    ax1.axes.set_title("{}|{}".format(col,dp[1]), fontsize=10)
            else:
                ax1.set_ylim([0, 1.05])
                ax1.axes.set_title("{}|{}".format(col,dp[1]), fontsize=10)
            # plt.close() 
        if if_plot:
            plt.show()
        if path:
            if not os.path.exists(path):#如果路径不存在
               os.makedirs(path)
            if file_prefix:
                file="{}/{}_{}.png".format(path,file_prefix,col)
            else:
                file="{}/{}.png".format(path,col)
            fig.savefig(file.format(path,col), pad_inches=0.3, dpi=100, papertype='a4', bbox_inches='tight')
            fig_list.append(file)
            
    if excel:
        if sheet_name is None:
            sheet_name='feature_ana_df'
        writer= pd.ExcelWriter(excel)
    
        feature_ana_df.reset_index(drop=True).to_excel(writer,float_format="%.2f",freeze_panes=(1,0),sheet_name=sheet_name,index=False)
        #count_distr
        writer.sheets[sheet_name].conditional_format('E2:E' + str(len(feature_ana_df)+1), {'type': 'data_bar'})
        #total_iv
        writer.sheets[sheet_name].conditional_format('K2:K' + str(len(feature_ana_df)+1), {'type': 'data_bar','bar_color': '#CC00FF'})
        #bet
        writer.sheets[sheet_name].conditional_format('O2:O' + str(len(feature_ana_df)+1), {'type': 'data_bar','bar_color': '#63C384'})
        #std
        writer.sheets[sheet_name].conditional_format('S2:S' + str(len(feature_ana_df)+1), {'type': 'data_bar','bar_color': '#FF0000'})
     
        if plot_insert_excel:
            sheet = writer.book.add_worksheet(sheet_name + '_plot')
            if path and plot_insert_excel:
                for i in enumerate(fig_list):
                    print (i)
                    sheet.write(i[0] * 30 + 1,0  ,i[1])
                    sheet.insert_image(i[0] * 30 + 2,0,i[1])
        writer.close()
        
    return fig_list


def plt_multi_rsk_trend(df, x, y='fpd_k4', miss_values=[-999], has_dt=1, dt='event_date', dt_cut='month',
                        score_cut=None, n=10, name=''):
    '''
    Parameters
    ----------
    df : pd.dataframe
        输入dataframe,需包含y，dt.
    x : string
        模型分 如'ali_cs_duotou_score',连续值
    y : string
        样本的因变量 如 'fpd4', 因变量只能为0,1的int格式输入.
    miss_values: list
        缺失值列表，缺失值单独一箱
    has_dt: 0,1
        针对没有dt的dataframe
    dt : string, optional
        样本拆分的日期，一般我们建模中常用的，建案时间，交易时间等等。格式为 20190101、2019-01-01均可. The default is 'event_date'.
    n : int, optional
        特征等频分箱的组数，默认q=10，等频分为10箱. The default is 10.
    dt_cut : list, optional
        样本拆分月自定义切点，list自定义 如:[-np.inf,20190101,20190601,20190901,np.inf]
        有"month","dt" 和自定义切点三种模式
        "month"--- 按照dt拆分每月看
        "dt" --- 按照dt字段看-这里dt就可以自定义是周的字段还是其他
    score_cut : list, optional
        默认值 None的时候是按照n=xx等频切分，可自定义固定模型分切点
    name : string, optional
        图片名称. The default is ''.

    Returns
    -------
    plot图片.
    bins -- 模型分的切点，可用在别的样本分组上
    风险分布明细数据

    '''

    def cal_ks(data, prob, y):
        return ks_2samp(data[prob][data[y] == 1], data[prob][data[y] == 0]).statistic

    def cal_iv(x, y):
        df = y.groupby(x).agg(['count', 'sum'])
        df.columns = ['total', 'count_1']
        df['count_0'] = df.total - df.count_1
        df['pct'] = df.total / df.total.sum()
        df['bad_rate'] = df.count_1 / df.total
        df['woe'] = np.log((df.count_0 / df.count_0.sum()) / (df.count_1 / df.count_1.sum()))
        df['woe'] = df['woe'].replace(np.inf, 0)
        #    print(df)
        rate = ((df.count_0 / df.count_0.sum()) - (df.count_1 / df.count_1.sum()))
        IV = np.sum(rate * df.woe)
        return IV, df

    if has_dt == 1:
        temp_df = df[[x, y, dt]]
        temp_df[x] = temp_df[x].replace(miss_values, -99)
        if dt_cut == 'month':
            temp_df['dt_cut'] = temp_df[dt].astype('str').str.replace('-', '').str[0:6]
        elif dt_cut == 'dt':
            temp_df['dt_cut'] = temp_df[dt]
        else:
            temp_df[dt] = temp_df[dt].astype('str').str.replace('-', '').astype(int)
            temp_df['dt_cut'] = pd.cut(temp_df[dt], dt_cut).astype('object')

    else:
        temp_df = df[[x, y]]
        temp_df[x] = temp_df[x].replace(miss_values, -99)
        temp_df['dt_cut'] = 'all'

    if score_cut != None:
        temp_df['group'] = pd.cut(temp_df[x], score_cut)
    else:
        _, bins = pd.qcut(temp_df[x], n, duplicates='drop', retbins=True)
        bins[0] = -99
        bins[-1] = np.inf
        bins = sorted(set(np.insert(bins, 0, [-np.inf, -99, np.inf])))
        temp_df['group'] = pd.cut(temp_df[x], bins)

    a = temp_df.pivot_table(index=['dt_cut', 'group'], values=y, aggfunc=np.size, fill_value=0)
    a = a.unstack(level=0).droplevel(0, axis=1)
    a.index = a.index.astype('str')
    pct = a / a.sum()
    A = pd.concat(
        [(pct.iloc[:, i] - pct.iloc[:, 0]) * np.log(pct.iloc[:, i] / pct.iloc[:, 0]) for i in range(pct.shape[1])],
        axis=1)
    A.columns = pct.columns
    psi = A.sum()

    y_list = temp_df.groupby(temp_df['dt_cut'])[y].apply(lambda x: sum(x) / len(x))
    num_list = temp_df.groupby(temp_df['dt_cut'])[y].apply(lambda x: len(x))
    iv_list = temp_df.groupby(temp_df['dt_cut']).apply(lambda k: cal_iv(k['group'], k[y])[0])
    ks_list = temp_df.groupby(temp_df['dt_cut']).apply(lambda k1: cal_ks(k1, x, y))
    auc_list = temp_df.groupby(temp_df['dt_cut']).apply(lambda k2: metrics.roc_auc_score(k2[y], k2[x]))

    c = temp_df.pivot_table(index=['group'], values=[y], columns='dt_cut',
                            aggfunc=[len, np.sum, lambda x: sum(x) / len(x)], fill_value=0)
    c.columns.set_levels(['len', 'sum', 'ratio'], level=0, inplace=True)
    c = c.droplevel(1, axis=1)

    #    tot_y = temp_df[y].sum()/temp_df[y].shape[0]
    tot_size = temp_df[y].shape[0]
    tot_iv = cal_iv(temp_df['group'], temp_df[y])[0]
    tot_ks = cal_ks(temp_df, x, y)
    tot_auc = metrics.roc_auc_score(temp_df[y], temp_df[x])

    l = sorted(set(temp_df['dt_cut']))

    fig = plt.figure(figsize=(6 * len(l), 4))
    fig.suptitle(
        '{}{},{}, 总样本量：{}, 总IV:{:.2f}, 总KS:{:.2f}, 总AUC:{:.2f}'.format(x, name, y, tot_size, tot_iv, tot_ks, tot_auc),
        x=0.08, y=1.07, ha='left', size=15, bbox=dict(facecolor='grey', alpha=0.1))

    for k in l:
        i = l.index(k)
        max_bs = c[('ratio', k)].iloc[-1] / y_list[k]

        ax1 = plt.subplot(1, len(l), i + 1)
        ax1.set_xticklabels([str(x) for x in list(c.index)], rotation=90, fontsize=10)
        sns.barplot(list(c.index), c[('len', k)], ax=ax1, alpha=0.2, color='k')
        ax1.text(0.01, 0.95, "平均风险:{:.2%}  {:.1f}倍".format(y_list[k], max_bs), transform=ax1.transAxes,
                 fontdict={'size': '10', 'color': 'b'})  # 写平均风险值
        ax1.set_title(
            '{}, 样本量：{}, \n IV:{:.2f}, KS:{:.2f}, AUC:{:.2f}, PSI:{:.2f}'.format(k, num_list[k], iv_list[k], ks_list[k],
                                                                                 auc_list[k], psi[k]), size=12)  # 表标题
        ax1.axes.set_ylabel('')

        ax2 = ax1.twinx()
        sns.pointplot(list(c.index), c[('ratio', k)], ax=ax2, alpha=0.2, color='red', scale=0.5)  # 画在ax2画布上
        for a, b in zip([i for i in range(len(c.index))], c[('ratio', k)]):
            ax2.annotate("{:.2%}".format(b), xy=(a, b), xytext=(-20, 5), textcoords='offset points', weight='heavy')
        ax2.axhline(y=y_list[k], color='grey', label='avg_risk', linestyle='--', linewidth=1)  # 平均风险线
        ax2.set_ylim([0, c['ratio'].max().max() * 1.2])
        ax2.axes.set_ylabel('')

    plt.show()
    plt.close()
    return bins, c
