from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.random import check_random_state
import math
import pandas as pd
import numpy as np


class ProbSmoothingStr2RspTransformer(BaseEstimator,TransformerMixin):
    """概率平滑字符转响应率Transformer

    Parameters
    ----------
    col_x: list
       特征列表.
    bin_limit : int
       类别数量限制
    smoothing: float
       平滑系数，越大平滑程度越强
    data_parts: str
       数据集切分字段.
    data_parts_use: list or None
       样本分析列表，eg:[1,2],只会分析data_parts为1和2的样本，如果为None分析整个数据集


    Attributes
    ----------
    col_new :list
        输出返回特征列表
    dict_map :dict
        特征映射关系字典
    """
    def __init__(self,
                 col_x,
                 bin_limit=20,
                 smoothing=1.0,
                 data_parts='data_parts',
                 data_parts_use=None):
        self.col_x=col_x
        self.bin_limit=bin_limit
        self.smoothing=smoothing
        self.dict_map=dict()
        self.data_parts=data_parts
        self.data_parts_use=data_parts_use
        self.col_new=col_x

    def fit(self,data,y=None):
        '''fit函数
        
        Parameters
        ----------
        data: pd.DataFrame
            数据集-数据集中需要包含col_x所有字段，如果y不为None，数据集中需要包含y字段
        
        Returns
        -------
        self:对象本身对象本身
        '''
        self.col_new=self.col_x
        
        if y is  None:
            raise TypeError('missing argument: ''y''')
        else:
            if self.data_parts_use:
                data=data.loc[data[self.data_parts].isin(self.data_parts_use),:].reset_index(drop=True).copy(deep=True)
            else:
                data=data.copy(deep=True)
                
            self.obj_cols = []
            for idx, dt in enumerate(data[self.col_x].dtypes):
                if dt == 'object' or pd.api.types.is_categorical_dtype(dt):
                    self.obj_cols.append(data[self.col_x].columns.values[idx])
            
            if len(self.obj_cols) == 0 :
                return self
            print ("self.obj_cols= ",self.obj_cols)
            # print (y.value_counts(dropna=False))
            self.sample_mean = data[y.name].mean()
            self.dict_map = {}
            # print("data.shape=",data.shape)
            data[self.obj_cols] = data[self.obj_cols].astype(str)
            for col in self.obj_cols:
                stats = data[y.name].groupby(data[col]).agg(['count', 'mean'])
                # print ("stats = ",stats)
                smoove = 1 / (1 + np.exp(-(stats['count'] - self.bin_limit) / self.smoothing))
                # print ("smoove = ",stats)
                smoothing = self.sample_mean * (1 - smoove) + stats['mean'] * smoove
                smoothing[stats['count'] == 1] = self.sample_mean
                self.dict_map[col] = smoothing

    def transform(self,data):
        '''transform函数
        
        Parameters
        ----------
        data: pd.DataFrame
            需要转换的数据集
        
        Returns
        -------
        data:pd.DataFrame
            转换后的数据集
        '''
        data=data.copy(deep=True)

        for i in data.columns.tolist():
            if i in self.dict_map.keys():
                data[i] = data[i].astype(str)
                data[i]=data[i].map(self.dict_map[i]).fillna(self.sample_mean)
                # data[i]=data[i].map(self.dict_map[i])
        return data


class OrderStr2RspTransformer(BaseEstimator,TransformerMixin):
    """顺序Target Encoding Transformer

    Parameters
    ----------
    col_x: list
       特征列表.
    order_col : None or str
       排序字段列表，默认为None，如果为None则按照样本顺序编码，如果设置为字段则按照该字段升序编码
    sigma : None or float
       扰动系数，越大正则能力越强，默认None
    smoothing: float
       平滑系数，越大平滑程度越强，默认1
    data_parts: str
       数据集切分字段.
    data_parts_use: list or None
       样本分析列表，eg:[1,2],只会分析data_parts为1和2的样本，如果为None分析整个数据集

    Attributes
    ----------
    col_new :list
        输出返回特征列表
    dict_map :dict
        特征映射关系字典
    """
    def __init__(self,
                 col_x,
                 sigma=None, 
                 smoothing=1,
                 random_state=None, 
                 data_parts='data_parts',
                 data_parts_use=None):
        self.col_x=col_x
        self.dict_map=dict()
        self.data_parts=data_parts
        self.data_parts_use=data_parts_use
        self.col_new=col_x
        self.sigma = sigma
        self.smoothing = smoothing
        self.random_state = random_state

    def fit(self,data,y=None):
        '''fit函数
        
        Parameters
        ----------
        data: pd.DataFrame
            数据集-数据集中需要包含col_x所有字段，如果y不为None，数据集中需要包含y字段
        
        Returns
        -------
        self:对象本身对象本身
        '''
        self.col_new=self.col_x
        self.dict_map = {}
        
        if y is  None:
            raise TypeError('missing argument: ''y''')
        else:
            if self.data_parts_use:
                data=data.loc[data[self.data_parts].isin(self.data_parts_use),:].reset_index(drop=True).copy(deep=True)
                
            self.obj_cols = []
            for idx, dt in enumerate(data[self.col_x].dtypes):
                if dt == 'object' or pd.api.types.is_categorical_dtype(dt):
                    self.obj_cols.append(data[self.col_x].columns.values[idx])
            
            if len(self.obj_cols) == 0 :
                return self
            
            # print (y.value_counts(dropna=False))
            
            data[self.obj_cols] = data[self.obj_cols].astype(str)
            self.dict_map = self._fit(
                data[self.obj_cols], y,
                cols=self.obj_cols
            )
            
            # X_temp = self.transform(data[self.obj_cols], y)
    
    def _fit(self, X_in, y, cols):
        X = X_in.copy(deep=True)

        self.sample_mean = y.mean()

        return {col: self._fit_column_map(X[col], y) for col in cols}

    def _fit_column_map(self, series, y):
        category = pd.Categorical(series)

        categories = category.categories
        codes = category.codes.copy()

        # Ensure codes align with the length of the series, handling NaN values
        codes = pd.Series(codes, index=series.index)
        codes[codes == -1] = len(categories)
        categories = np.append(categories, np.nan)

        return_map = pd.Series(dict([(code, category) for code, category in enumerate(categories)]))

        result = y.groupby(codes).agg(['sum', 'count'])
        return result.rename(return_map)
    
    def transform(self,data,y=None):
        '''transform函数
        
        Parameters
        ----------
        data: pd.DataFrame
            需要转换的数据集
        
        Returns
        -------
        data:pd.DataFrame
            转换后的数据集
        '''
        data=data.copy(deep=True)
        
        obj_cols = list(self.dict_map.keys() & set(data.columns.tolist()))
        
        if len(obj_cols) == 0:
            pass
        else:
            data[obj_cols]  = data[obj_cols].astype(str) 
            data = self._transform(
                data[obj_cols], y,
                mapping=self.dict_map
            )
        
        # for i in data.columns.tolist():
        #     if i in self.dict_map.keys():
        #         data[i] = data[i].astype(str)
        #         data[i]=data[i].map(self.dict_map[i]).fillna(self.sample_mean)
        return data


    def _transform(self, X_in, y, mapping=None):
        """
        The model uses a single column of floats to represent the means of the target variables.
        """
        X = X_in.copy(deep=True)
        
        random_state_ = check_random_state(self.random_state)
        
        # Prepare the data
        if y is not None:
            # Convert bools to numbers (the target must be summable)
            y = y.astype('double')

        for col, colmap in mapping.items():
            level_notunique = colmap['count'] > 1

            unique_train = colmap.index
            unseen_values = pd.Series([x for x in X_in[col].unique() if x not in unique_train], dtype=unique_train.dtype)
            is_unknown_value = X_in[col].isin(unseen_values.astype(str))
            # is_nan = X_in[col].isnull()
            """
            is_unknown_value = X_in[col].isin(unseen_values.dropna().astype(object))
            """
            # print ("!!!",y is None)
            if y is None:    # Replace level with its mean target; if level occurs only once, use global mean
                level_means = ((colmap['sum'] + self.sample_mean) / (colmap['count'] + self.smoothing)).where(level_notunique, self.sample_mean)
                X[col] = X[col].map(level_means)
            else:
                self.temp = y.groupby(X[col].astype(str)).agg(['cumsum', 'cumcount'])
                X[col] = (self.temp['cumsum'] - y + self.sample_mean) / (self.temp['cumcount'] + self.smoothing)

            if X[col].dtype.name == 'category'  or X[col].dtype.name  == 'object':
                X[col] = X[col].astype(float)
            X.loc[is_unknown_value, col] = self.sample_mean
                # X.loc[unseen_values.isnull().any(), col] = self.sample_mean

            if self.sigma is not None and y is not None:
                X[col] = X[col] * random_state_.normal(1., self.sigma, X[col].shape[0])

        return X


class TfidfStr2RspTransfomer(BaseEstimator,TransformerMixin):
    def __init__(self,
                col_x,
                bin_limit = 20,
                data_parts = 'data_parts',
                data_parts_use = None):
        self.col_x = col_x
        self.bin_limit = bin_limit
        self.data_parts = data_parts
        self.data_parts_use = data_parts_use
        self.dict_map = dict()
        self.col_new = col_x
    
    def fit(self,data,y = None):
        
        self.col_new = self.col_x
        if y is None:
            raise TypeError('missing argument: ''y''')
        else:
            if self.data_parts_use:
                data=data.loc[data[self.data_parts].isin(self.data_parts_use),:].reset_index(drop=True).copy(deep=True)
            else:
                data=data.copy(deep=True)
                
            self.obj_cols = []
            for idx, dt in enumerate(data[self.col_x].dtypes):
                if dt == 'object' or pd.api.types.is_categorical_dtype(dt):
                    self.obj_cols.append(data[self.col_x].columns.values[idx])
                    
            if len(self.obj_cols) == 0 :
                return self
            print ("self.obj_cols= ",self.obj_cols)
            self.dict_map = {}
            self.sample_mean = data[y.name].mean()
            data[self.obj_cols] = data[self.obj_cols].astype(str)
            for col in self.obj_cols:
                states = data[y.name].groupby(data[col]).agg([('sum', np.sum), ('cnt', np.size)])
                bad_num = states['sum'].sum()
                cust_cnt = data.shape[0]
                states = states[states['cnt']>= self.bin_limit].assign(
                            tf = lambda x:x['sum']/bad_num )
                states['idf'] = states['cnt'].apply(lambda x:math.log(cust_cnt/x))
                states['tfidf'] = states['tf']*states['idf']
                self.dict_map[col] = dict(zip(states.index,states['tfidf'].round(4)))
        
    def transform(self,data):
        data = data.copy(deep=True)
        
        for i in data.columns.tolist():
            if i in self.dict_map.keys():
                data[i] = data[i].astype(str)
                data[i] = data[i].map(self.dict_map[i]).fillna(self.sample_mean)
        return data