# 加载包
# 忽略warning
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
# 绘图
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = 'C:/Windows/Fonts/simsun.ttc'  # 替换为微软雅黑字体文件路径
kaiti_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = kaiti_font.get_name() # 将微软雅黑设置为绘图默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 机器学习
from sklearn.linear_model import LinearRegression, LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# 模型评价
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error, r2_score
# 标准化
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
# 网格搜索
from sklearn.model_selection import GridSearchCV
# # 神经网络
# from keras.models import Sequential
# import tensorflow as tf
# from keras.layers import LSTM, Dense, Dropout, GRU
import joblib
from WindPy import w
import datetime
import os
import matplotlib.ticker as ticker #处理坐标轴密度
import math
import re
w.start()


class DataTrain(object):
    def __init__(self):
        # 调优参数设置
        self.parameters = {'DecisionTreeRegressor': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 4, 6, 8],
                                                     'criterion': ['mse', 'mae'], 'min_samples_leaf': [10, 20, 30, 40]},
                           'DecisionTreeClassifier': {'max_depth': [10, 15, 20, 25, 30],
                                                      'min_samples_split': [2, 4, 6, 8],
                                                      'criterion': ['gini', 'entropy'],
                                                      'min_samples_leaf': [10, 20, 30, 40]},
                           'SVR': {'kernel': ['rbf', 'sigmoid'], 'C': [0.0001, 0.001, 0.01, 0.1, 1],
                                   'gamma': [0.1, 0.001, 0.001, 0.0001, 0.00001]},
                           'SVC': {'kernel': ['linear', 'rbf', 'sigmoid'], 'C': [0.01, 0.1, 1, 10],
                                   'gamma': [0.001, 0.01, 0.1, 1]},
                           'RandomForestRegressor': {'max_depth': [3, 5, 7, 10], 'n_estimators': [20, 30, 40, 50],
                                                     'min_samples_split': [2, 4, 6, 8],
                                                     'max_leaf_nodes': [30, 40, 50, 60], 'random_state': [123]},
                           'RandomForestClassifier': {'max_depth': [3, 5, 7, 10], 'n_estimators': [10, 20, 30, 40, 50],
                                                      'min_samples_split': [2, 4, 6, 8],
                                                      'max_leaf_nodes': [20, 30, 40, 50, 60], 'random_state': [123]},
                           'AdaBoostRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001],
                                                 'random_state': [123]},
                           'AdaBoostClassifier': {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001],
                                                  'random_state': [123]},
                           'XGBRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]},
                           'XGBClassifier': {'max_depth': [3, 5, 7, 10], 'n_estimators': [20, 40, 60, 80, 100],
                                             'gamma': [0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
                                             'learning_rate': [0.5, 0.1, 0.05, 0.01], 'random_state': [123]},
                           'LGBMClassifier': {'n_estimators': [30, 50, 70], 'learning_rate': [0.1, 0.01, 0.001],
                                              'max_depth': [3, 5, 7, 10], 'num_leaves': [10, 20, 30, 40],
                                              'random_state': [123]},
                           'LGBMRegressor': {'n_estimators': [30, 50, 70], 'learning_rate': [0.1, 0.01, 0.001],
                                             'max_depth': [3, 5, 7, 10], 'num_leaves': [10, 20, 30, 40],
                                             'random_state': [123]}}

        self.today = datetime.datetime.now().date()

        self.df_raw = pd.read_excel('因子库_09.xlsx')
        self.df_raw.set_index('date', inplace=True)
        #self.df_raw = self.df_raw.loc[self.df_raw.index > '2021-08-13']#永续回测需要
        self.df = pd.DataFrame(index=self.df_raw.index)
        self.df = pd.concat([self.df, self.df_raw.iloc[:, 0:63].diff(1)], axis=1)
        self.df = pd.concat([self.df, self.df_raw.iloc[:, 63:65].fillna(method='ffill').diff(1)], axis=1)
        self.df = pd.concat([self.df, self.df_raw.iloc[:, 65:]], axis=1)

    def preprocess(self, y):
        '''
        数据预处理函数，进行异常值和缺失值处理
        '''
        # 部分因子前端出现较多缺失，难以插补，将因子直接删除
        self.df = self.df.dropna(thresh=self.df.shape[1] / 2)  # 删除一行超过20各na的数据，主要涉及周末
        for col in self.df.columns:  # 删除na过多的列，主要涉及永续收益率
            if self.df[col].isnull().sum() > self.df.shape[0] / 2:
                print(col, '前端缺失，已删除。')
                self.df.drop([col], axis=1, inplace=True)
        # 将各列由object转换为数值,将异常值替换为缺失值，然后对缺失值进行线性插补
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col])
            # z-score绝对值大于5判定为异常值
            self.df[col] = self.df[col].apply(lambda x: np.nan if abs(
                (x - self.df[col].mean()) / self.df[col].std()) > 5 else x)
            self.df[col] = self.df[col].interpolate(method='linear', axis=0)  # 线性插补
        # data = data.set_index('日期')
        self.df.fillna(0, inplace=True)
        self.df_clean = self.df.dropna()
        print(self.df_clean.info())

        self.df_clean['y'] = self.df_clean[y].shift(-1).apply(lambda x: 1 if x > 0 else -1)
        self.df_clean = self.df_clean.iloc[:-1]  # 最后一天数据删除 没有标签

    def train_test_split(self, train_size):
        '''
        划分训练集和测试集
        其中train_size表示训练集数据所占比例
        '''
        y = self.df_clean['y']
        X = self.df_clean.drop('y', axis=1)

        length_train = round(len(y) * train_size)
        self.y_train = y[:length_train]
        self.y_test = y[length_train:]
        self.x_train = X.iloc[:length_train, :]
        self.x_test = X.iloc[length_train:, :]
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.x_test.shape)
        print(self.y_test.shape)

    def predict(self, model_path, y):
        '''
        model_path：模型文件路径
        y：收益率
        '''
        # 标准化
        scaler = MinMaxScaler()
        scaler = scaler.fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        self.make_dirs(model_path)
        joblib.dump(scaler, f'{model_path}scaler_{y}.pkl')

    def make_dirs(self, file_path):
        '''
        传入文件路径，判断文件路径是否存在，不存在则创建
        file_path: 文件路径
        '''
        dir_path = os.path.dirname(file_path)
        # print(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def model_evaluation(self, y_train, y_train_pred, y_test, y_test_pred, model_str):
        # 模型评价
        accuracy_train = accuracy_score(y_train, y_train_pred)
        Confusion_Matrix_train = confusion_matrix(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        Confusion_Matrix_test = confusion_matrix(y_test, y_test_pred)
        print(model_str + ' 训练集表现：')  # 模型名称
        print("准确率:", accuracy_train)  # accuracy_train = accuracy_score(y_train, y_train_pred)
        print("Confusion Matrix:")
        print(Confusion_Matrix_train)  # Confusion_Matrix_train = confusion_matrix(y_train, y_train_pred)
        print('                  ')
        print(model_str + ' 测试集表现：')
        print("准确率:", accuracy_test)
        print("Confusion Matrix:")
        print(Confusion_Matrix_test)
        print('------------------')
        return accuracy_train, accuracy_test

    def use_model_LogisticRegression(self, model_path, save_path, y):
        '''
        使用  逻辑回归  模型对处理后的数据进行预测
        model_path：模型文件路径
        save_path：预测结果保存路径
        y：收益率
        '''
        # 模型拟合
        model = LogisticRegression()
        model.fit(self.x_train, self.y_train)
        # 未存在的文件需要创建
        self.make_dirs(model_path)
        # 收益率的：改成_
        y = re.sub(r'[:：]', '_', y)
        joblib.dump(model, f'{model_path}lg_model_{y}.pkl')
        # joblib.dump(model, '.\lg_model.pkl')

        # 模型预测，输出涨跌概率
        y_train_pred = model.predict_proba(self.x_train)
        y_test_pred = model.predict_proba(self.x_test)
        y_train_predict = model.predict(self.x_train)
        y_test_predict = model.predict(self.x_test)

        # 整合日期date、到期收益率yield、跌的概率proba到dataframe中，便于回测
        train_df = pd.DataFrame(
            {'date': self.df_clean.index[:len(self.x_train)],
             'yield': self.df_raw[y].loc[self.df_clean.index[:len(self.x_train)]],
             'proba': y_train_pred[:, 0]})
        test_df = pd.DataFrame(
            {'date': self.df_clean.index[len(self.x_train):],
             'yield': self.df_raw[y].loc[self.df_clean.index[len(self.x_train):]],
             'proba': y_test_pred[:, 0]})
        df = pd.concat([train_df, test_df], axis=0)
        # 信号生成
        train_df['signal'] = train_df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        test_df['signal'] = test_df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        df['signal'] = df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        # 将df保存至excel
        self.make_dirs(save_path)
        train_df.reset_index(drop=True).to_excel(save_path + f'lo_re_train_{y}.xlsx')
        test_df.reset_index(drop=True).to_excel(save_path + f'lo_re_test_{y}.xlsx')
        df.reset_index(drop=True).to_excel(save_path + f'lo_re_{y}.xlsx')
        # 模型评价
        self.acc_train, self.acc_test = self.model_evaluation(self.y_train, y_train_predict, self.y_test,
                                                              y_test_predict, 'LogisticRegression')
        # 因子重要性
        model = joblib.load(f'{model_path}lg_model_{y}.pkl')
        feature_importance = model.coef_[0]
        feature_names = self.df_clean.columns[:-1]

        print(len(feature_importance), feature_importance, sep='\n')
        print(len(feature_names), feature_names, sep='\n')

        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        # 倒叙排序
        importance = importance_df.sort_values(by='importance', ascending=False).head(30)
        importance["importance"] = (importance["importance"] * 1000).astype(int)
        importance = importance.sort_values(["importance"])
        importance.set_index('feature', inplace=True)
        importance.plot.barh(color='r', alpha=0.7, rot=0, figsize=(8, 8))
        plt.savefig(save_path + f'lo_re_importance_{y}.png', dpi=300)
        # plt.show()

    def use_model_AdaBoostClassifier(self, model_path, save_path, y):
        '''
        使用  AdaBoost分类  模型对处理后的数据进行预测
        model_path：模型文件路径
        save_path：预测结果保存路径
        y：收益率
        '''
        # 模型拟合 AdaBoostClassifier
        model = AdaBoostClassifier()
        # 网格搜索弱分类器
        ada = GridSearchCV(estimator=model, cv=5, param_grid=self.parameters['AdaBoostClassifier'], scoring='accuracy',
                           n_jobs=2)
        ada.fit(self.x_train, self.y_train)
        best_est = ada.best_estimator_
        # 未存在的文件需要创建
        self.make_dirs(model_path)
        # joblib.dump(best_est, model_path)
        # 收益率的：改成_
        y = re.sub(r'[:：]', '_', y)
        joblib.dump(best_est, f'{model_path}ada_model_{y}.pkl')
        print("AdaBoost最优参数:", ada.best_params_)
        # 模型预测，输出涨跌概率
        y_train_pred = ada.predict_proba(self.x_train)
        y_test_pred = ada.predict_proba(self.x_test)
        y_train_predict = ada.predict(self.x_train)
        y_test_predict = ada.predict(self.x_test)
        # 整合日期date、到期收益率yield、跌的概率proba到dataframe中，便于回测
        train_df = pd.DataFrame(
            {'date': self.df_clean.index[:len(self.x_train)],
             'yield': self.df_raw[y].loc[self.df_clean.index[:len(self.x_train)]],
             'proba': y_train_pred[:, 0]})
        test_df = pd.DataFrame(
            {'date': self.df_clean.index[len(self.x_train):],
             'yield': self.df_raw[y].loc[self.df_clean.index[len(self.x_train):]],
             'proba': y_test_pred[:, 0]})
        df = pd.concat([train_df, test_df], axis=0)
        # 信号生成
        train_df['signal'] = train_df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        test_df['signal'] = test_df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        df['signal'] = df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        # 将df保存至excel
        self.make_dirs(save_path)
        train_df.reset_index(drop=True).to_excel(save_path + f'ada_cl_train_{y}.xlsx')
        test_df.reset_index(drop=True).to_excel(save_path + f'ada_cl_test_{y}.xlsx')
        df.reset_index(drop=True).to_excel(save_path + f'ada_cl_{y}.xlsx')
        # 模型评价
        self.acc_train, self.acc_test = self.model_evaluation(self.y_train, y_train_predict, self.y_test,
                                                              y_test_predict, 'AdaBoostClassifier')
        # 因子重要性排序
        model = joblib.load(f'{model_path}ada_model_{y}.pkl')
        importances_values = model.feature_importances_
        importances = pd.DataFrame(importances_values, columns=["importance"])
        feature_data = pd.DataFrame(self.df_clean.columns[:-1], columns=["feature"])
        importance = pd.concat([feature_data, importances], axis=1)
        # 倒叙排序
        importance = importance.sort_values(["importance"], ascending=False).head(30)
        importance["importance"] = (importance["importance"] * 1000).astype(int)
        importance = importance.sort_values(["importance"])
        importance.set_index('feature', inplace=True)
        importance.plot.barh(color='r', alpha=0.7, rot=0, figsize=(12, 8))
        plt.savefig(save_path + f'ada_cl_importance_{y}.png', dpi=300)
        # plt.show()

    def use_model_XGBClassifier(self, model_path, save_path, y):
        '''
        使用  XGB分类  模型对处理后的数据进行预测
        model_path：模型文件路径
        save_path：预测结果保存路径
        y：收益率
        '''
        # 模型拟合 XGBClassifier
        model = XGBClassifier()
        # 网格搜索弱分类器
        xgb = GridSearchCV(estimator=model, cv=5, param_grid=self.parameters['XGBClassifier'], scoring='accuracy',
                           n_jobs=2)
        xgb_y_train = self.y_train.replace(-1, 0)
        xgb_y_test = self.y_test.replace(-1, 0)
        xgb.fit(self.x_train, xgb_y_train)
        best_est = xgb.best_estimator_
        # 未存在的文件需要创建
        self.make_dirs(model_path)
        # joblib.dump(best_est, model_path)
        # 收益率的：改成_
        y = re.sub(r'[:：]', '_', y)
        joblib.dump(best_est, f'{model_path}xgb_model_{y}.pkl')
        print("XGBoost最优参数:", xgb.best_params_)
        # 模型预测，输出涨跌概率
        y_train_pred = xgb.predict_proba(self.x_train)
        y_test_pred = xgb.predict_proba(self.x_test)
        y_train_predict = xgb.predict(self.x_train)
        y_test_predict = xgb.predict(self.x_test)
        # 整合日期date、到期收益率yield、跌的概率proba到dataframe中，便于回测
        train_df = pd.DataFrame(
            {'date': self.df_clean.index[:len(self.x_train)],
             'yield': self.df_raw[y].loc[self.df_clean.index[:len(self.x_train)]],
             'proba': y_train_pred[:, 0]})
        test_df = pd.DataFrame(
            {'date': self.df_clean.index[len(self.x_train):],
             'yield': self.df_raw[y].loc[self.df_clean.index[len(self.x_train):]],
             'proba': y_test_pred[:, 0]})
        df = pd.concat([train_df, test_df], axis=0)
        # 信号生成
        train_df['signal'] = train_df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        test_df['signal'] = test_df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        df['signal'] = df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        # 将df保存至excel
        self.make_dirs(save_path)
        train_df.reset_index(drop=True).to_excel(save_path + f'xgb_cl_train_{y}.xlsx')
        test_df.reset_index(drop=True).to_excel(save_path + f'xgb_cl_test_{y}.xlsx')
        df.reset_index(drop=True).to_excel(save_path + f'xgb_cl_{y}.xlsx')
        # 模型评价
        self.acc_train, self.acc_test = self.model_evaluation(xgb_y_train, y_train_predict, xgb_y_test, y_test_predict,
                                                              'XGBClassifier')
        # 因子重要性排序
        model = joblib.load(f'{model_path}xgb_model_{y}.pkl')
        importances_values = model.feature_importances_
        importances = pd.DataFrame(importances_values, columns=["importance"])
        feature_data = pd.DataFrame(self.df_clean.columns[:-1], columns=["feature"])
        importance = pd.concat([feature_data, importances], axis=1)
        # 倒叙排序
        importance = importance.sort_values(["importance"], ascending=False).head(30)
        importance["importance"] = (importance["importance"] * 1000).astype(int)
        importance = importance.sort_values(["importance"])
        importance.set_index('feature', inplace=True)
        importance.plot.barh(color='r', alpha=0.7, rot=0, figsize=(12, 8))
        plt.savefig(save_path + f'xgb_cl_importance_{y}.png', dpi=300)
        # plt.show()

    def use_model_RandomForestClassifier(self, model_path, save_path, y):
        '''
        使用  随机森林  模型对处理后的数据进行预测
        model_path：模型文件路径
        save_path：预测结果保存路径
        y：收益率
        '''
        # 模型拟合
        model = RandomForestClassifier()
        rf = GridSearchCV(estimator=model, cv=5, param_grid=self.parameters['RandomForestClassifier'],
                          scoring='accuracy',
                          n_jobs=2)
        rf.fit(self.x_train, self.y_train)
        self.make_dirs(model_path)
        # joblib.dump(rf.best_estimator_, model_path)
        # 收益率的：改成_
        y = re.sub(r'[:：]', '_', y)
        joblib.dump(rf.best_estimator_, f'{model_path}rf_model_{y}.pkl')
        print("RandomForest最优参数:", rf.best_params_)
        # 模型预测，输出涨跌概率
        y_train_pred = rf.predict_proba(self.x_train)
        y_test_pred = rf.predict_proba(self.x_test)
        y_train_predict = rf.predict(self.x_train)
        y_test_predict = rf.predict(self.x_test)
        # 整合日期date、到期收益率yield、跌的概率proba到dataframe中，便于回测
        train_df = pd.DataFrame(
            {'date': self.df_clean.index[:len(self.x_train)],
             'yield': self.df_raw[y].loc[self.df_clean.index[:len(self.x_train)]],
             'proba': y_train_pred[:, 0]})
        test_df = pd.DataFrame(
            {'date': self.df_clean.index[len(self.x_train):],
             'yield': self.df_raw[y].loc[self.df_clean.index[len(self.x_train):]],
             'proba': y_test_pred[:, 0]})
        df = pd.concat([train_df, test_df], axis=0)
        # 信号生成
        train_df['signal'] = train_df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        test_df['signal'] = test_df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        df['signal'] = df['proba'].apply(lambda x: 1 if x >= 0.52 else (0 if x >= 0.48 else -1))
        # 将df保存至excel
        self.make_dirs(save_path)
        train_df.reset_index(drop=True).to_excel(save_path + f'rf_cl_train_{y}.xlsx')
        test_df.reset_index(drop=True).to_excel(save_path + f'rf_cl_test_{y}.xlsx')
        df.reset_index(drop=True).to_excel(save_path + f'rf_cl_{y}.xlsx')
        # 模型评价
        self.acc_train, self.acc_test = self.model_evaluation(self.y_train, y_train_predict, self.y_test,
                                                              y_test_predict, 'RandomForestClassifier')
        # 得到特征重要度分数
        model = joblib.load(f'{model_path}rf_model_{y}.pkl')
        importances_values = model.feature_importances_
        importances = pd.DataFrame(importances_values, columns=["importance"])
        feature_data = pd.DataFrame(self.df_clean.columns[:-1], columns=["feature"])
        importance = pd.concat([feature_data, importances], axis=1)
        # 倒叙排序
        importance = importance.sort_values(["importance"], ascending=False).head(30)
        importance["importance"] = (importance["importance"] * 1000).astype(int)
        importance = importance.sort_values(["importance"])
        importance.set_index('feature', inplace=True)
        importance.plot.barh(color='r', alpha=0.7, rot=0, figsize=(12, 8))
        plt.savefig(save_path + f'rf_cl_importance_{y}.png', dpi=300)
        # plt.show()

    # 处理信号
    # 第一个工作日发出原信号后，第二个工作日进行交易并变更持仓，本步骤将原信号转化为交易和仓位结合的信号
    # 转化后，若某日有交易信号1/0/-1，则当日执行交易买入/不动/卖出，仓位对应变化
    def tradesignal(self, data):
        tradesignaldata = data.copy()
        position = 0
        for i in tradesignaldata.index:
            if (tradesignaldata.loc[i, 'signal'] == 1) & (position == 0):
                tradesignaldata.loc[i, 'tradesignal'] = 1
                position += 1
            elif (tradesignaldata.loc[i, 'signal'] == -1) & (position == 1):
                tradesignaldata.loc[i, 'tradesignal'] = -1
                position += -1
            else:
                tradesignaldata.loc[i, 'tradesignal'] = 0
            tradesignaldata.loc[i, 'position'] = position
        tradesignaldata['tradesignal'] = tradesignaldata['tradesignal'].shift(fill_value=0)
        tradesignaldata['position'] = tradesignaldata['position'].shift(fill_value=0)
        data['tradesignal'] = tradesignaldata['tradesignal']
        data['position'] = tradesignaldata['position']
        return data

    # 执行交易
    # trade1：交易前准备
    def trade1(self, data):
        trade1data = data.copy()
        # 计算利息率
        trade1data['interest rate'] = trade1data['yield'] * trade1data['position'] * trade1data['tradesignal']
        trade1data['interest rate'] = trade1data['interest rate'].replace(0, np.nan).ffill().fillna(0) * trade1data[
            'position']
        # 设日期为index:trade1data.set_index('日期', inplace=True, drop=True)
        # 拼接日历日
        trade1data_ziranri = pd.DataFrame(
            index=pd.date_range(start=trade1data.index[0], end=trade1data.index[-1], freq="D"))
        trade1data = trade1data_ziranri.merge(trade1data, left_index=True, right_index=True, how="left")
        trade1data['yield'] = trade1data['yield'].ffill()
        trade1data['signal'] = trade1data['signal'].fillna(0)
        trade1data['tradesignal'] = trade1data['tradesignal'].fillna(0)
        trade1data['position'] = trade1data['position'].ffill()
        trade1data['interest rate'] = trade1data['interest rate'].ffill()
        return trade1data

    # trade2：执行交易各项计算，假设初始仓位为1亿元
    def trade2(self, data, term):
        trade2data = data.copy()
        # 计算资本利得
        trade2data['trading income bps'] = -(trade2data['yield'] - trade2data['yield'].shift()).fillna(0)
        # 初始仓位
        openworth = 100
        lastworth = 100
        # 计算仓位
        for i in trade2data.index:
            if trade2data.loc[i, 'position'] == 0 and trade2data.loc[i, 'tradesignal'] == 0:
                # 尚未首次开仓
                trade2data.loc[i, 'trading income'] = 0
                trade2data.loc[i, 'interest rate income'] = 0
                trade2data.loc[i, 'networth'] = openworth
            # elif tradedata.loc[i,'position'] == 1 & ((tradedata.loc[i,'tradesignal'] == 1) | (tradedata.loc[i,'tradesignal'] == 0)):
            elif trade2data.loc[i, 'position'] == 1 and trade2data.loc[i, 'tradesignal'] == 1:
                # 首次开仓
                trade2data.loc[i, 'trading income'] = 0
                trade2data.loc[i, 'interest rate income'] = openworth * trade2data.loc[i, 'interest rate'] / 36500
                dailypnl = trade2data.loc[i, 'interest rate income']
                lastworth += dailypnl
                trade2data.loc[i, 'networth'] = lastworth
            elif trade2data.loc[i, 'position'] == 1 and trade2data.loc[i, 'tradesignal'] == 0:
                # 首次开仓后持仓
                trade2data.loc[i, 'trading income'] = openworth * term * trade2data.loc[i, 'trading income bps'] / 100
                trade2data.loc[i, 'interest rate income'] = openworth * trade2data.loc[i, 'interest rate'] / 36500
                dailypnl = trade2data.loc[i, 'trading income'] + trade2data.loc[i, 'interest rate income']
                lastworth += dailypnl
                trade2data.loc[i, 'networth'] = lastworth
            # elif tradedata.loc[i,'position'] == 1 and tradedata.loc[i,'tradesignal'] == 0:
            #     #持仓
            #     networth = networth + networth*tradedata.loc[i,'interest rate']/36500 + networth*int(y[-2])*tradedata.loc[i,'trading income']/1000000
            elif trade2data.loc[i, 'position'] == 0 and trade2data.loc[i, 'tradesignal'] == -1:
                # 首次开仓持仓后平仓并更新openworth
                trade2data.loc[i, 'trading income'] = openworth * term * trade2data.loc[i, 'trading income bps'] / 100
                trade2data.loc[i, 'interest rate income'] = 0
                dailypnl = trade2data.loc[i, 'trading income']
                lastworth += dailypnl
                trade2data.loc[i, 'networth'] = lastworth
                # 一个开仓持仓平仓周期结束后更新openworth
                openworth = lastworth
                # 报错
            else:
                trade2data.loc[i, 'networth'] = 'something wrong'
        return trade2data

    # #ROA计算模块
    # def roa(self, df):
    #     return (df['networth'][-1]-df['networth'][0])/df['networth'][0]/((df.index[-1]-df.index[0])/np.timedelta64(1,'D'))*365

    # 绩效核算
    def performance(self, data, rf):
        performancedata = data.copy()
        # 计算年化ROA
        ROA = 365 * (performancedata['networth'][-1] - performancedata['networth'][0]) / (
                performancedata['networth'] * performancedata['position']).sum()
        # year_earnings = (performancedata['networth'][-1]-performancedata['networth'][0])/((performancedata.index[-1]-performancedata.index[0])/np.timedelta64(1,'D'))*365
        # zhanzi = (performancedata['networth']*performancedata['position']).sum()/((performancedata.index[-1]-performancedata.index[0])/np.timedelta64(1,'D'))
        # ROA = earings/zhanzi
        # ROA=(performancedata['networth'][-1]-performancedata['networth'][0])/performancedata['networth'][0]/((performancedata.index[-1]-performancedata.index[0])/np.timedelta64(1,'D'))*365
        # 计算开仓次数
        if performancedata['tradesignal'].sum() != 0:
            num = performancedata['tradesignal'][performancedata['tradesignal'] > 0].sum() - 1
        else:
            num = performancedata['tradesignal'][performancedata['tradesignal'] > 0].sum()
        # 计算年化收益率
        daily_pctchange = performancedata['networth'].pct_change()
        Annual_yield = daily_pctchange.mean() * 365
        # 计算年化波动率
        Annual_std = daily_pctchange.std() * math.sqrt(365)
        # 计算夏普比率
        if Annual_std == 0:
            Sharp_ratio = 'NA'
        else:
            Sharp_ratio = (Annual_yield - rf) / Annual_std
        # 最大回撤
        Max_drawdown = max(
            (np.maximum.accumulate(performancedata['networth']) - performancedata['networth']) / np.maximum.accumulate(
                performancedata['networth']))
        # 卡尔玛比率
        if Max_drawdown == 0:
            Calmar_ratio = 'NA'
        else:
            Calmar_ratio = (Annual_yield - rf) / Max_drawdown
        return {'训练集准确率': self.acc_train, '测试集准确率': self.acc_test, '年化收益率': Annual_yield,
                '年化波动率': Annual_std, '夏普比率': Sharp_ratio, '最大回撤': Max_drawdown,
                '卡尔玛比率': Calmar_ratio, 'ROA': ROA, '开仓次数': num}

    def backtest(self, plt_savepath, term, model, test_file):
        '''
        回测函数
        plt_savepath: 图表保存路径
        '''
        # 指定文件
        # file_name = r'.\result\adaboost\ada_cl_test_国开1yYTM.xlsx'
        file_name = os.path.join('result', model, test_file)

        # 读取数据
        # 数据应为三列，表头分别为date代表日期，yield代表收益率，signal代表策略发出的原信号
        # 若某日有原信号1/0/-1，则第二日执行交易买入/不动/卖出

        df = pd.read_excel(file_name)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df.loc[df['date'] > '20231230']
        df.set_index(['date'], drop=True, inplace=True)

        # 计算结果
        df = self.tradesignal(df)
        df = self.trade2(self.trade1(df), term)
        df['trading income accu'] = df['trading income'].cumsum()
        df['interest rate income accu'] = df['interest rate income'].cumsum()
        df['pnl accu'] = df['trading income accu'] + df['interest rate income accu']
        df['trading income percentage'] = df['trading income accu'] / df['pnl accu']
        df['trading income percentage'] = df['trading income percentage'].fillna(0)
        performace = self.performance(df, 0.02)
        print(performace)

        df['date'] = df.index
        df0 = df.copy()
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        plt.grid()
        title1 = '策略总体表现'
        plt.title(title1)
        line1 = ax.plot(df['date'], df['networth'], label='净值曲线(左轴、起始资金100)', color='crimson', linewidth=1,
                        alpha=0.9)
        ax_twin = ax.twinx()
        line2 = ax_twin.plot(df['date'], df['yield'], label="收益率曲线(右轴)", color='slategrey', linewidth=0.5,
                             linestyle="dashed")
        # 增加开多和开空时间点位
        long_idx = df[df['tradesignal'] == 1]['date']
        short_idx = df[df['tradesignal'] == -1]['date']
        line3 = ax_twin.plot(long_idx, df.loc[df['date'].isin(long_idx), 'yield'], '>', alpha=0.5, markersize=6,
                             color='lightcoral', label='开仓')
        line4 = ax_twin.plot(short_idx, df.loc[df['date'].isin(short_idx), 'yield'], '<', alpha=0.5, markersize=6,
                             color='springgreen', label='平仓')
        # 加图例
        lines = line1 + line2 + line3 + line4
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))  # 设置绘图坐标轴间隔
        fig.autofmt_xdate()  # 自动旋转xlabel
        # 未存在的文件需要创建
        self.make_dirs(plt_savepath)
        plt.savefig(plt_savepath + f'{model}_{title1}.png', dpi=300)
        # plt.show()

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)  # figsize=(20,10),
        plt.grid()
        title2 = '策略资本利得情况'
        plt.title(title2)
        line1 = ax.plot(df['date'], df['pnl accu'], label='总收入曲线(起始资金100)', color='crimson', linewidth=1,
                        alpha=0.9)
        line2 = ax.plot(df['date'], df['trading income accu'], label="累计资本利得曲线", color='tab:green',
                        linewidth=0.5, linestyle="dashed")
        line3 = ax.plot(df['date'], df['interest rate income accu'], label="累计利息收入曲线", color='slategrey',
                        linewidth=0.5, linestyle="dashed")
        # 加图例
        lines = line1 + line2 + line3
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))  # 设置绘图坐标轴间隔
        fig.autofmt_xdate()  # 自动旋转xlabel
        self.make_dirs(plt_savepath)
        plt.savefig(plt_savepath + f'{model}_{title2}.png', dpi=300)
        # plt.show()

        df_performace = pd.DataFrame(performace, index=[0])
        print(df_performace)
        df_performace.to_excel(plt_savepath + f'{model}_performance.xlsx', index=False)

# if __name__ == '__main__':
#     es = DataTrain()
#     label = '永续5yYTM'
#     term = 5
#     es.preprocess(label)
#     es.train_test_split(0.8)
#     es.predict('./model/',label)
#     print('Step：1 逻辑回归')
#     es.use_model_LogisticRegression('./model/', './result/logistic/', label)
#     es.backtest(f'./backtest/{label}/',term,'logistic', f'lo_re_test_{label}.xlsx')
#     print('Step：2 adaboost')
#     es.use_model_AdaBoostClassifier('./model/', './result/adaboost/', label)
#     es.backtest(f'./backtest/{label}/', term, 'adaboost', f'ada_cl_test_{label}.xlsx')
#     print('Step：3 RandomForest')
#     es.use_model_RandomForestClassifier('./model/', './result/randomforest/', label)
#     es.backtest(f'./backtest/{label}/', term, 'randomforest', f'rf_cl_test_{label}.xlsx')
#     print('Step：4 xgboost')
#     es.use_model_XGBClassifier('./model/', './result/xgboost/', label)
#     es.backtest(f'./backtest/{label}/', term, 'xgboost', f'xgb_cl_test_{label}.xlsx')
#
#     perf1 = pd.read_excel(f'./backtest/{label}/logistic_performance.xlsx')
#     perf2 = pd.read_excel(f'./backtest/{label}/adaboost_performance.xlsx')
#     perf3 = pd.read_excel(f'./backtest/{label}/randomforest_performance.xlsx')
#     perf4 = pd.read_excel(f'./backtest/{label}/xgboost_performance.xlsx')
#     perf = pd.concat([perf1,perf2,perf3,perf4]).set_index(np.array(['logistic','adaboost','randomforest','xgboost']))
#     perf = perf.reset_index().rename(columns={'index':'model'})
#     perf.to_excel(f'./backtest/{label}/performance.xlsx',index=False)

if __name__ == '__main__':
    for i in [3,5]:
        es = DataTrain()
        label = f'永续{i}yYTM'
        print(label)
        term = i
        es.preprocess(label)
        es.train_test_split(0.8)
        es.predict('./model/', label)
        print('Step：1 逻辑回归')
        es.use_model_LogisticRegression('./model/', './result/logistic/', label)
        es.backtest(f'./backtest/{label}/', term, 'logistic', f'lo_re_test_{label}.xlsx')
        print('Step：2 adaboost')
        es.use_model_AdaBoostClassifier('./model/', './result/adaboost/', label)
        es.backtest(f'./backtest/{label}/', term, 'adaboost', f'ada_cl_test_{label}.xlsx')
        print('Step：3 RandomForest')
        es.use_model_RandomForestClassifier('./model/', './result/randomforest/', label)
        es.backtest(f'./backtest/{label}/', term, 'randomforest', f'rf_cl_test_{label}.xlsx')
        print('Step：4 xgboost')
        es.use_model_XGBClassifier('./model/', './result/xgboost/', label)
        es.backtest(f'./backtest/{label}/', term, 'xgboost', f'xgb_cl_test_{label}.xlsx')

        perf1 = pd.read_excel(f'./backtest/{label}/logistic_performance.xlsx')
        perf2 = pd.read_excel(f'./backtest/{label}/adaboost_performance.xlsx')
        perf3 = pd.read_excel(f'./backtest/{label}/randomforest_performance.xlsx')
        perf4 = pd.read_excel(f'./backtest/{label}/xgboost_performance.xlsx')
        perf = pd.concat([perf1, perf2, perf3, perf4]).set_index(
            np.array(['logistic', 'adaboost', 'randomforest', 'xgboost']))
        perf = perf.reset_index().rename(columns={'index': 'model'})
        perf.to_excel(f'./backtest/{label}/performance.xlsx', index=False)









