import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from shaphypetune._classes import _FastRFE
from shaphypetune.gbdt_utils.xgboost_cv import XGBClassifierCV, XGBRegressorCV
from shaphypetune.gbdt_utils.xgboost_metrics import xgb_ks_score_negative
from shaphypetune.scorecard.utils import cal_ks


def test_classification():
    # 生成分类数据
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=10, 
        n_clusters_per_class=1, 
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"类别分布: {np.bincount(y_train)}")
    
    # 创建CV分类器
    clf_cv = XGBClassifierCV(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20,
        tree_method="exact",
        # CV参数
        cv_nfold=5,
        cv_stratified=True,
        cv_shuffle=True,
        cv_seed=42,
        cv_as_pandas=True,
        cv_verbose_eval=10,
        cv_show_stdv=True
    )
    
    print("\n开始训练...")
    
    # 训练模型
    clf_cv.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 预测
    y_pred = clf_cv.predict(X_test)
    y_pred_proba = clf_cv.predict_proba(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"测试集AUC: {auc:.4f}")
    
    # 获取CV结果
    cv_results = clf_cv.get_cv_results()
    print(f"\nCV结果形状: {cv_results.shape}")
    print("\nCV结果列名:")
    print(cv_results.columns.tolist())
    
    # 显示最后几轮的CV结果
    print("\n最后5轮CV结果:")
    print(cv_results.tail())
    

def test_regression():
    # 生成回归数据
    X, y = make_regression(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        noise=0.1, 
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"目标变量范围: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # 创建CV回归器
    reg_cv = XGBRegressorCV(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='rmse',
        early_stopping_rounds=20,
        tree_method="exact",
        # CV参数
        cv_nfold=5,
        cv_stratified=False,  # 回归通常不用分层
        cv_shuffle=True,
        cv_seed=42,
        cv_as_pandas=True,
        cv_verbose_eval=10,
        cv_show_stdv=True
    )
    
    print("\n开始训练...")
    
    # 训练模型
    reg_cv.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 预测
    y_pred = reg_cv.predict(X_test)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n测试集RMSE: {rmse:.4f}")
    print(f"测试集R²: {r2:.4f}")
    
    # 获取CV结果
    cv_results = reg_cv.get_cv_results()
    print(f"\nCV结果形状: {cv_results.shape}")
    print("\nCV结果列名:")
    print(cv_results.columns.tolist())
    
    # 显示最后几轮的CV结果
    print("\n最后5轮CV结果:")
    print(cv_results.tail())
    

def test_fastrfe():
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    param_dist = {
        'max_depth': [2, 3, 4],
        'subsample': [0.7],
        'min_child_weight': [20, 30, 10],
        'base_score': [0.5, 0.7, 0.6, 0.4],
        'reg_lambda': [1, 5, 10, 15],
        'learning_rate': [0.1, 0.08, 0.12],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'reg_alpha': [1, 10],
        'random_state': stats.rv_discrete(values=([i*8 for i in range(1000)], [1/1000]*1000))
    }
    clf_xgb = XGBClassifierCV(n_estimators=200, verbosity=0, n_jobs=2, early_stopping_rounds=6, 
                              tree_method="exact", eval_metric=xgb_ks_score_negative,
                                cv_nfold=5,
                                cv_stratified=True,
                                cv_shuffle=True,
                                cv_seed=42,
                                cv_as_pandas=True,
                                cv_verbose_eval=10,
                                cv_show_stdv=True)
    model = _FastRFE(clf_xgb, min_features_to_select=5, param_grid=param_dist, n_iter=5, n_warmup_iter=1,
                        sampling_seed=1, verbose=2, importance_type="shap_importances", train_importance=False)
    model.fit(x_train, y_train, eval_set=[(x_train, y_train)])
    np.testing.assert_almost_equal([cal_ks(model.predict_proba(x_train)[:, 1], y_train)[0]],
                                   [model.best_score_], decimal=5)
    

    clf_xgb2 = XGBClassifierCV(n_estimators=200, verbosity=0, n_jobs=2, early_stopping_rounds=6, tree_method="exact", 
                               eval_metric=xgb_ks_score_negative, **model.best_params_,
                               cv_nfold=5,
                                cv_stratified=True,
                                cv_shuffle=True,
                                cv_seed=42,
                                cv_as_pandas=True,
                                cv_verbose_eval=10,
                                cv_show_stdv=True)
    clf_xgb2.fit(x_train[:, model.support_], y_train, eval_set=[(x_train[:, model.support_], y_train)])
    np.testing.assert_almost_equal([cal_ks(clf_xgb2.predict_proba(x_train[:, model.support_])[:, 1], y_train)[0]],
                                   [model.best_score_], decimal=5)
    
    clf_xgb3 = XGBClassifier(n_estimators=clf_xgb2.cv_best_iteration_, verbosity=0, n_jobs=2, tree_method="exact", 
                             eval_metric=xgb_ks_score_negative, **model.best_params_)
    clf_xgb3.fit(x_train[:, model.support_], y_train, eval_set=[(x_train[:, model.support_], y_train)])
    np.testing.assert_almost_equal([cal_ks(clf_xgb3.predict_proba(x_train[:, model.support_])[:, 1], y_train)[0]],
                                   [model.best_score_], decimal=5)
    

def test_custom_stratified_kfold():
    """测试自定义 StratifiedKFold 作为 cv_folds 参数"""
    from sklearn.model_selection import StratifiedKFold

    # 生成分类数据
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=10, 
        n_clusters_per_class=1, 
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[-1])], index=[f"my_index_{i}" for i in range(X.shape[0])])
    df["y"] = y

    X_train, X_test, y_train, y_test = train_test_split(
        df, df["y"], test_size=0.2, random_state=42, stratify=df["y"]
    )

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 创建自定义的 StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 生成 fold 索引列表
    folds = []
    for train_idx, valid_idx in skf.split(X_train, y_train):
        folds.append((train_idx, valid_idx))

    print(f"\n创建了 {len(folds)} 个 folds")
    for i, (train_idx, valid_idx) in enumerate(folds):
        print(f"Fold {i+1}: 训练集 {len(train_idx)} 样本, 验证集 {len(valid_idx)} 样本")
        print(f"  训练集类别分布: {np.bincount(y_train.iloc[train_idx])}")
        print(f"  验证集类别分布: {np.bincount(y_train.iloc[valid_idx])}")

    # 创建 CV 分类器，使用自定义 folds
    clf_cv = XGBClassifierCV(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        tree_method="exact",
        early_stopping_rounds=10,
        # CV参数 - 使用自定义 folds
        cv_folds=folds,  # 传入自定义的 fold 索引
        cv_nfold=None,   # 当使用 cv_folds 时，cv_nfold 会被忽略
        cv_stratified=None,  # 当使用 cv_folds 时，cv_stratified 会被忽略
        cv_shuffle=None,     # 当使用 cv_folds 时，cv_shuffle 会被忽略
        cv_seed=42,
        cv_as_pandas=True,
        cv_verbose_eval=10,
        cv_show_stdv=True,
    )

    print("\n开始训练（使用自定义 folds）...")

    # 训练模型
    clf_cv.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # 预测
    y_pred = clf_cv.predict(X_test)
    y_pred_proba = clf_cv.predict_proba(X_test)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"测试集AUC: {auc:.4f}")
    print(f"最佳迭代轮数: {clf_cv.cv_best_iteration_}")

    # 获取CV结果
    cv_results = clf_cv.get_cv_results()
    print(f"\nCV结果形状: {cv_results.shape}")
    print("\nCV结果列名:")
    print(cv_results.columns.tolist())

    # 显示最后几轮的CV结果
    print("\n最后5轮CV结果:")
    print(cv_results.tail())

    # 验证使用了正确的 fold 数量
    # CV 结果的长度应该等于最终训练的轮数
    print(f"\nCV训练轮数: {len(cv_results)}")
    assert len(folds) == 3, "应该使用3个folds"