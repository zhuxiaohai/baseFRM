import os
import pandas as pd
import toad  
from optbinning import BinningProcess
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from shaphypetune.scorecard.report import plt_multi_rsk_trend, plot_reg_bins_trend
from shaphypetune.scorecard import visualize, utils  


def test_plt_multi_rsk_trend():
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    afsxc = XGBClassifier(n_estimators=200, verbosity=0, n_jobs=2, early_stopping_rounds=6)
    afsxc.fit(x_train, y_train,
                eval_set=[(x_valid, y_valid)])
    cols = [f"col{i}" for i in range(x_train.shape[-1])]
    train = pd.DataFrame(x_train, columns=cols)
    train["y"] = y_train
    train["set"] = "1train"
    valid = pd.DataFrame(x_valid, columns=cols)
    valid["y"] = y_valid
    valid["set"] = "2test"
    df = pd.concat([train, valid], axis=0)
    df["pred"] = afsxc.predict_proba(df[cols].values)[:, 1]

    _ = plt_multi_rsk_trend(df, ["pred"], y="y", dt='set', dt_cut='set', miss_values=[-99], score_cut={"pred": 20},
                        method='quantile', digit=4, binning_col='set', binning_set='1train',
                        if_plot=True, output_path=None)


def test_plot_reg_bins_trend():
    # 加载糖尿病回归数据集
    X, y = load_diabetes(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    
    # 训练XGBoost回归模型
    model = XGBRegressor(n_estimators=100, verbosity=0, n_jobs=2, early_stopping_rounds=6)
    model.fit(x_train, y_train,
             eval_set=[(x_valid, y_valid)],
             )
    
    # 准备数据
    cols = [f"col{i}" for i in range(x_train.shape[-1])]
    train = pd.DataFrame(x_train, columns=cols)
    train["y_true"] = y_train
    train["set"] = "1train"
    valid = pd.DataFrame(x_valid, columns=cols)
    valid["y_true"] = y_valid
    valid["set"] = "2test"
    
    # 合并数据并生成预测值
    df = pd.concat([train, valid], axis=0)
    df["y_pred"] = model.predict(df[cols].values)
    
    # 测试plot_reg_bins_trend函数
    _, _, _ = plot_reg_bins_trend(
        df=df,
        group_col="set",
        base_group="1train", 
        y_pred_col="y_pred",
        y_true_col="y_true",
        n_bins=10
    )

def test_monitor():
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    cols = [f"col{i}" for i in range(x_train.shape[-1])]
    train = pd.DataFrame(x_train, columns=cols)
    train["y"] = y_train
    train["set"] = "1train"
    valid = pd.DataFrame(x_valid, columns=cols)
    valid["y"] = y_valid
    valid["set"] = "2test"
    train_all_df = pd.concat([train, valid], axis=0)
    y = "y"
    selected_features = cols[:3]

    combiner2 = toad.transform.Combiner()
    combiner2.fit(train_all_df[(train_all_df['set']=='1train')][selected_features + [y]], 
                y=y, 
                n_bins=5, 
                method='quantile',
                empty_separate=True)  
    binning_process = BinningProcess(selected_features, max_n_bins=5, max_pvalue=0.1, special_codes=None)
    binning_process.fit(train_all_df[(train_all_df['set']=='1train')][selected_features], 
                        train_all_df[(train_all_df['set']=='1train')][y])   
    toad_dict = utils.optbinning_to_toad(binning_process, selected_features) 
    combiner = toad.transform.Combiner()
    combiner.set_rules(toad_dict)
    output_dir = os.path.join(os.getcwd(), 'v0')
    os.makedirs(output_dir, exist_ok=True) 
    visualize.monitor(train_all_df, 
                    [combiner2, combiner], selected_features, y, output_dir, NA=-99, fig_size=(8, 16),
                    group='set',
                    set_array=['1train', '2test'],
                    plot_configs=[
                        {'psi_series': 1, 'iv_series': 1, 'countratio_df': 2, 'lift_df': 1.5, 'mixed': 2},
                        {'psi_series': 1, 'iv_series': 1, 'countratio_df': 2, 'lift_df': 1.5, 'mixed': 2}
                        ]
                    )  
