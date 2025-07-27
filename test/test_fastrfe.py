import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_digits, load_breast_cancer, load_iris, fetch_california_housing, make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor
from shaphypetune._classes import _FastRFE
from shaphypetune.gbdt_utils.xgboost_metrics import xgb_ks_score_negative, xgb_r2_score_negative
from shaphypetune.scorecard.utils import cal_ks


def test_fastrfe_classification():
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
    clf_xgb = XGBClassifier(n_estimators=200, verbosity=0, n_jobs=2, early_stopping_rounds=6, eval_metric=xgb_ks_score_negative)
    model = _FastRFE(clf_xgb, min_features_to_select=5, param_grid=param_dist, n_iter=5, n_warmup_iter=1,
                     sampling_seed=1, verbose=2, importance_type="shap_importances", train_importance=False)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])

    print('n_features', model.n_features_)
    np.testing.assert_almost_equal([cal_ks(model.predict_proba(x_valid)[:, 1], y_valid)[0]],
                                   [model.best_score_], decimal=5)
    np.testing.assert_almost_equal([cal_ks(model.estimator_.predict_proba(x_valid[:, model.support_])[:, 1], y_valid)[0]],
                                   [model.best_score_], decimal=5)

    print(model.best_params_)
    afsxc = XGBClassifier(n_estimators=200, verbosity=0, n_jobs=2, early_stopping_rounds=6, eval_metric=xgb_ks_score_negative, verbose=True, **model.best_params_)
    afsxc.fit(x_train[:, model.support_], y_train,
              eval_set=[(x_valid[:, model.support_], y_valid)],)
    test_pred = afsxc.predict_proba(x_valid[:, model.support_])[:, 1]
    np.testing.assert_almost_equal([cal_ks(test_pred, y_valid)[0]],
                                   [model.best_score_], decimal=5)


def test_fastrfe_regression():
    X, y = fetch_california_housing(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    param_dist = {
        'max_depth': [2, 3, 4],
        'subsample': [0.7],
        'min_child_weight': [20, 30, 10],
        'reg_lambda': [1, 5, 10, 15],
        'learning_rate': [0.1, 0.08, 0.12],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'reg_alpha': [1, 10],
        'random_state': stats.rv_discrete(values=([i*8 for i in range(1000)], [1/1000]*1000))
    }
    clf_xgb = XGBRegressor(n_estimators=200, verbosity=0, n_jobs=2, base_score=np.mean(y_train), early_stopping_rounds=6, eval_metric=xgb_r2_score_negative)
    model = _FastRFE(clf_xgb, min_features_to_select=5, param_grid=param_dist, n_iter=5, n_warmup_iter=3,
                     importance_type='shap_importances', train_importance=False, sampling_seed=1, verbose=2)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    print('n_features', model.n_features_)
    np.testing.assert_almost_equal([r2_score(y_valid, model.predict(x_valid))],
                                   [model.best_score_], decimal=5)
    np.testing.assert_almost_equal([r2_score(y_valid, model.estimator_.predict(x_valid[:, model.support_]))],
                                   [model.best_score_], decimal=5)

    print(model.best_params_)
    afsxc = XGBRegressor(n_estimators=200, verbosity=0, n_jobs=2, base_score=np.mean(y_train), early_stopping_rounds=6, eval_metric=xgb_r2_score_negative, **model.best_params_)
    afsxc.fit(x_train[:, model.support_], y_train,
              eval_set=[(x_valid[:, model.support_], y_valid)]
              )
    test_pred = afsxc.predict(x_valid[:, model.support_])
    np.testing.assert_almost_equal([r2_score(y_valid, test_pred)],
                                   [model.best_score_], decimal=5)


def test_fastrfe_multiclass():
    def xgb_accuracy_score_negative(y_true, y_pred):
        # 对于二分类，需要将概率值转换为类别
        if len(np.unique(y_true)) == 2:
            y_pred_labels = (y_pred > 0.5).astype(int)
        else:
            # 对于多分类，将概率值转换为类别标签
            y_pred_labels = np.argmax(y_pred, axis=1)
            
        acc = accuracy_score(y_true, y_pred_labels)
        return -acc

    
    # 加载iris数据集
    X, y = load_iris(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    
    # 定义参数网格
    param_dist = {
        'max_depth': [2, 3, 4],
        'subsample': [0.7],
        'min_child_weight': [5, 10, 15],
        'reg_lambda': [1, 5, 10],
        'learning_rate': [0.1, 0.05, 0.15],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'reg_alpha': [1, 5],
        'random_state': stats.rv_discrete(values=([i*8 for i in range(1000)], [1/1000]*1000))
    }
    
    # 初始化多分类XGBoost分类器
    clf_xgb = XGBClassifier(n_estimators=100, verbosity=0, n_jobs=2, objective='multi:softprob', early_stopping_rounds=6, eval_metric=xgb_accuracy_score_negative, num_class=3)
    
    # 初始化FastRFE
    model = _FastRFE(clf_xgb, min_features_to_select=2, param_grid=param_dist, n_iter=5, n_warmup_iter=1,
                     sampling_seed=1, verbose=2)
    
    # 训练模型
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
    
    print('n_features', model.n_features_)
    
    # 验证模型预测结果
    pred = model.predict(x_valid)
    estimator_pred = model.estimator_.predict(x_valid[:, model.support_])
    
    np.testing.assert_almost_equal([accuracy_score(y_valid, pred)],
                                   [model.best_score_], decimal=5)
    np.testing.assert_almost_equal([accuracy_score(y_valid, estimator_pred)],
                                   [model.best_score_], decimal=5)
    
    print(model.best_params_)
    
    # 使用最佳参数重新训练模型
    afsxc = XGBClassifier(n_estimators=100, verbosity=0, n_jobs=2, 
                          early_stopping_rounds=6, eval_metric=xgb_accuracy_score_negative,
                         objective='multi:softprob', num_class=3,
                         **model.best_params_)
    
    afsxc.fit(x_train[:, model.support_], y_train,
              eval_set=[(x_valid[:, model.support_], y_valid)])
              
    test_pred = afsxc.predict(x_valid[:, model.support_])
    np.testing.assert_almost_equal([accuracy_score(y_valid, test_pred)],
                                   [model.best_score_], decimal=5)


def test_xgb_with_categorical():
    # 创建示例数据，包含数值型和类别型特征
    import pandas as pd
    np.random.seed(42)
    n_samples = 1000
    
    # 数值型特征
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    
    # 类别型特征
    education = np.random.choice(['高中', '本科', '硕士', '博士'], n_samples)
    
    # 将特征组合在一起
    X_numeric = np.column_stack([age, income])
    
    # 使用OneHotEncoder处理类别型特征
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse_output=False)
    education_encoded = enc.fit_transform(education.reshape(-1, 1))
    
    # 展示编码与原始类别的对应关系
    print("\n类别编码映射关系:")
    categories = enc.categories_[0]
    for i, category in enumerate(categories):
        print(f"{category}: {enc.transform([[category]])[0]}")
    
    # 将编码后的类别特征与数值特征组合
    X = np.column_stack([X_numeric, education_encoded])
    
    # 创建目标变量（示例：是否获得贷款）
    # 修改计算方式以适应one-hot编码后的特征
    y = (age * 0.1 + income * 0.00003 + 
         np.sum(education_encoded * np.array([1, 2, 3, 4]), axis=1) + 
         np.random.normal(0, 1, n_samples)) > np.median(
        age * 0.1 + income * 0.00003 + np.sum(education_encoded * np.array([1, 2, 3, 4]), axis=1)
    )
    y = y.astype(int)
    
    # 划分训练集和测试集
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义XGBoost模型
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        verbosity=0,
        early_stopping_rounds=10
    )
    
    # 训练模型
    clf.fit(x_train, y_train,
            eval_set=[(x_valid, y_valid)])
    
    # 预测并评估
    y_pred = clf.predict(x_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f'模型准确率: {accuracy:.4f}')
    
    # 特征重要性
    feature_names = ['年龄', '收入'] + [f'教育程度_{cat}' for cat in enc.categories_[0]]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    })
    print('\n特征重要性:')
    print(feature_importance.sort_values('importance', ascending=False))


def test_xgb_with_native_categorical():
    np.random.seed(42)
    n_samples = 1000

    # 数值型特征
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(50000, 20000, n_samples)

    # 类别型特征
    education = np.random.choice(['高中', '本科', '硕士', '博士'], n_samples)

    # 将特征组合在一起
    X_numeric = np.column_stack([age, income])

    # 使用OneHotEncoder处理类别型特征
    enc = OneHotEncoder(sparse_output=False)
    education_encoded = enc.fit_transform(education.reshape(-1, 1))

    # 展示编码与原始类别的对应关系
    print("\n类别编码映射关系:")
    categories = enc.categories_[0]
    for i, category in enumerate(categories):
        print(f"{category}: {enc.transform([[category]])[0]}")

    # 创建目标变量（示例：是否获得贷款）
    # 修改计算方式以适应one-hot编码后的特征
    y = (age * 0.1 + income * 0.00003 + 
            np.sum(education_encoded * np.array([1, 2, 3, 4]), axis=1) + 
            np.random.normal(0, 1, n_samples)) > np.median(
        age * 0.1 + income * 0.00003 + np.sum(education_encoded * np.array([1, 2, 3, 4]), axis=1)
    )
    y = y.astype(int)

    df = pd.DataFrame(X_numeric, columns=["age", "income"])
    df["education"] = education
    df["target"] = y
    df["education"] = df["education"].astype("category")
    features = ["age", "income", "education"]
    cat_feats = ["education"]
    feature_types = ["c" if fn in cat_feats else "q" for fn in features]

    # 划分训练集和测试集
    x_train, x_valid, y_train, y_valid = train_test_split(df[features], df["target"], test_size=0.2, random_state=42)

    # Create an encoder based on training data.
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    enc = enc.fit(x_train[cat_feats])
    x_train[cat_feats] = enc.transform(x_train[cat_feats]).astype(int)
    x_valid[cat_feats] = enc.transform(x_valid[cat_feats]).astype(int)

    # 定义XGBoost模型
    clf_xgb = XGBClassifier(
        n_estimators=100,
        verbosity=0,
        early_stopping_rounds=10,
        tree_method="hist", 
        enable_categorical=True,
        feature_types=feature_types,
        eval_metric=xgb_ks_score_negative
    )

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

    model = _FastRFE(clf_xgb, min_features_to_select=3, param_grid=param_dist, n_iter=5, n_warmup_iter=1,
                    sampling_seed=1, verbose=2, importance_type="shap_importances", train_importance=False)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])

    print('n_features', model.n_features_)
    np.testing.assert_almost_equal([cal_ks(model.predict_proba(x_valid)[:, 1], y_valid)[0]],
                                    [model.best_score_], decimal=5)
    np.testing.assert_almost_equal([cal_ks(model.estimator_.predict_proba(x_valid.loc[:, model.support_])[:, 1], y_valid)[0]],
                                    [model.best_score_], decimal=5)

    print(model.best_params_)
    afsxc = XGBClassifier(
        n_estimators=100,
        verbosity=0,
        early_stopping_rounds=10,
        tree_method="hist", 
        enable_categorical=True,
        feature_types=feature_types,
        eval_metric=xgb_ks_score_negative,
        verbose=True, 
        **model.best_params_
    )
    afsxc.fit(x_train.loc[:, model.support_], y_train,
            eval_set=[(x_valid.loc[:, model.support_], y_valid)],)
    test_pred = afsxc.predict_proba(x_valid.loc[:, model.support_])[:, 1]
    np.testing.assert_almost_equal([cal_ks(test_pred, y_valid)[0]],
                                    [model.best_score_], decimal=5)
   

def test_xgb_with_multilabel_clc(): 
    def xgb_auc_score_negative(y_true, y_pred):
        y_true = y_true.reshape(y_pred.shape)
        auc = roc_auc_score(y_true, y_pred)
        return -auc

    X, y = make_multilabel_classification(n_features=10, n_classes=3, random_state=0)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    param_dist = {
        'max_depth': [2, 6],
        'random_state': stats.rv_discrete(values=([i*8 for i in range(1000)], [1/1000]*1000))
    }
    clf_xgb = XGBClassifier(n_estimators=200, verbosity=0, n_jobs=2, eval_metric=xgb_auc_score_negative, early_stopping_rounds=6, multi_strategy="one_output_per_tree")
    model = _FastRFE(clf_xgb, min_features_to_select=10, param_grid=param_dist, n_iter=5, n_warmup_iter=1,
                    sampling_seed=1, verbose=2, importance_type="shap_importances", train_importance=False)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=True)

    print('n_features', model.n_features_)
    np.testing.assert_almost_equal([-xgb_auc_score_negative(y_valid, model.predict_proba(x_valid))],
                                    [model.best_score_], decimal=5)
    np.testing.assert_almost_equal([-xgb_auc_score_negative(y_valid, model.estimator_.predict_proba(x_valid[:, model.support_]))],
                                    [model.best_score_], decimal=5)

    print(model.best_params_)
    afsxc = XGBClassifier(n_estimators=200, verbosity=0, n_jobs=2, eval_metric=xgb_auc_score_negative, early_stopping_rounds=6, multi_strategy="one_output_per_tree", **model.best_params_)
    afsxc.fit(x_train[:, model.support_], y_train,
            eval_set=[(x_valid[:, model.support_], y_valid)], verbose=True)
    test_pred = afsxc.predict_proba(x_valid[:, model.support_])
    np.testing.assert_almost_equal([-xgb_auc_score_negative(y_valid, test_pred)],
                                [model.best_score_], decimal=5)
    

def test_incremental_learning():
    X, y = load_digits(n_class=2, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, 
                                                        shuffle=True, stratify=y,
                                                        random_state=100)
    
    print("-+-" * 25)
    params1 = {'tree_method': 'hist', "n_estimators": 3}
    model1 = XGBClassifier(**params1)
    model1.fit(X_train, y_train)
    print(len(model1.get_booster().get_dump()))
    for leaf in model1.get_booster().get_dump():
        print(leaf)
    
    print("-+-" * 25)
    params2 = {'tree_method': 'hist', "n_estimators": 3}
    model2 = XGBClassifier(**params2)
    model2.fit(X_test, y_test, xgb_model=model1.get_booster())
    print(len(model2.get_booster().get_dump()))
    for leaf in model2.get_booster().get_dump():
        print(leaf)
    
    print("-+-" * 25)
    params3 = {'tree_method': 'exact', "n_estimators": 3}
    params3["updater"] = "refresh"
    params3["process_type"] = "update"
    params3["refresh_leaf"] = True
    # 则3棵树结构不变，叶节点权重改变，最终结果一共3棵树
    # 特别注意这里的num_boost_round <=原始模型的boost_nums 否则汇报错
    model3 = XGBClassifier(**params3)
    model3.fit(X_test, y_test, xgb_model=model1.get_booster())
    print(len(model3.get_booster().get_dump()))
    for leaf in model3.get_booster().get_dump():
        print(leaf)