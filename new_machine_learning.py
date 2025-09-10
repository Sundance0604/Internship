# -*- coding: utf-8 -*-
# 只训练 + 输出指标到 Excel（ACC/F1/ROC-AUC）+ 保存清晰混淆矩阵，不跑回测/夏普等
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import datetime, os, math, re, joblib

# 机器学习
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier  # 未使用，但保留导入不影响；若不想装lightgbm可删
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# 评估指标（F1、AUC）
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# === 可选字体（Windows宋体），非必须 ===
try:
    font_path = 'C:/Windows/Fonts/simsun.ttc'
    kaiti_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = kaiti_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


class DataTrain(object):
    def __init__(self):
        # 调优参数设置（保持你的原配置）
        self.parameters = {
            'RandomForestClassifier': {
                'max_depth': [3, 5, 7, 10],
                'n_estimators': [10, 20, 30, 40, 50],
                'min_samples_split': [2, 4, 6, 8],
                'max_leaf_nodes': [20, 30, 40, 50, 60],
                'random_state': [123]
            },
            'AdaBoostClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.1, 0.01, 0.001],
                'random_state': [123]
            },
            'XGBClassifier': {
                'max_depth': [3, 5, 7, 10],
                'n_estimators': [20, 40, 60, 80, 100],
                'gamma': [0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
                'learning_rate': [0.5, 0.1, 0.05, 0.01],
                'random_state': [123]
            },
        }

        self.today = datetime.datetime.now().date()

        # === 数据加载（与你原来一致） ===
        self.df_raw = pd.read_excel('因子库_09.xlsx')
        self.df_raw.set_index('date', inplace=True)

        # 构造特征表
        self.df = pd.DataFrame(index=self.df_raw.index)
        self.df = pd.concat([self.df, self.df_raw.iloc[:, 0:63].diff(1)], axis=1)
        self.df = pd.concat([self.df, self.df_raw.iloc[:, 63:65].fillna(method='ffill').diff(1)], axis=1)
        self.df = pd.concat([self.df, self.df_raw.iloc[:, 65:]], axis=1)

    def preprocess(self, y_col):
        """异常/缺失处理 + 生成标签 y ∈ {-1, 1}（下一期涨跌）"""
        # 行缺失过多删除
        self.df = self.df.dropna(thresh=self.df.shape[1] / 2)
        # 列缺失过多删除
        for col in list(self.df.columns):
            if self.df[col].isnull().sum() > self.df.shape[0] / 2:
                print(col, '前端缺失，已删除。')
                self.df.drop([col], axis=1, inplace=True)
        # 转数值 + 异常置 NaN + 线性插补
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col])
            self.df[col] = self.df[col].apply(
                lambda x: np.nan if abs((x - self.df[col].mean()) / (self.df[col].std() + 1e-12)) > 5 else x
            )
            self.df[col] = self.df[col].interpolate(method='linear', axis=0)
        self.df.fillna(0, inplace=True)
        self.df_clean = self.df.dropna()
        print(self.df_clean.info())

        # 生成二分类标签（下一期收益率 > 0 为 1，否则 -1）
        self.df_clean['y'] = self.df_clean[y_col].shift(-1).apply(lambda x: 1 if x > 0 else -1)
        self.df_clean = self.df_clean.iloc[:-1]  # 去掉末行无标签

        self.target_col = y_col  # 保存目标列名

    def train_test_split(self, train_size):
        y = self.df_clean['y']
        X = self.df_clean.drop('y', axis=1)

        n_tr = round(len(y) * train_size)
        self.y_train, self.y_test = y[:n_tr], y[n_tr:]
        self.x_train, self.x_test = X.iloc[:n_tr, :], X.iloc[n_tr:, :]

        print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)

    def fit_scaler(self, model_path, y_name):
        """MinMaxScaler 仅拟合训练集，再变换 train/test"""
        scaler = MinMaxScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        self.make_dirs(model_path)
        y_name = re.sub(r'[:：]', '_', y_name)
        joblib.dump(scaler, f'{model_path}scaler_{y_name}.pkl')

    @staticmethod
    def make_dirs(file_path):
        dir_path = os.path.dirname(file_path)
        if dir_path and (not os.path.exists(dir_path)):
            os.makedirs(dir_path)

    # === 统一评估（ACC/F1/ROC-AUC）：y_true ∈ {-1,1}, y_proba_pos ∈ [0,1], y_pred ∈ {-1,1}
    def compute_metrics(self, y_true, y_proba_pos, y_pred):
        y_true_bin = (pd.Series(y_true).values == 1).astype(int)
        acc = accuracy_score(y_true_bin, (np.array(y_pred) == 1).astype(int))
        f1 = f1_score(y_true_bin, (np.array(y_pred) == 1).astype(int))
        try:
            auc = roc_auc_score(y_true_bin, np.asarray(y_proba_pos).reshape(-1))
        except Exception:
            auc = np.nan
        return acc, f1, auc

    # === 将一条结果写入 DataFrame（便于统一输出） ===
    def append_result(self, results, model_name, dataset, best_params, acc, f1, auc):
        results.append({
            'model': model_name,
            'dataset': dataset,   # train / test
            'ACC': acc,
            'F1': f1,
            'AUC': auc,
            'best_params': str(best_params)
        })

    # === 混淆矩阵绘图（清晰） ===
    def _plot_cm(self, cm, classes, title, save_path, normalize=False):
        """
        画混淆矩阵（清晰版，支持归一化），并保存为高分辨率 PNG。
        cm: 2x2 numpy 数组（整数计数）
        classes: 类别名称列表（比如 [-1, 1]）
        title: 图标题
        save_path: 保存路径
        normalize: 是否按行归一化（显示百分比）
        """
        import numpy as np
        # 底图：float（便于色条显示）
        cm_img = cm.astype(float)
        if normalize:
            row_sums = cm_img.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_img = cm_img / row_sums

        fig, ax = plt.subplots(figsize=(5, 4), dpi=220)  # 高DPI更清晰
        im = ax.imshow(cm_img, interpolation='nearest', cmap=plt.cm.Blues)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel('Predicted label', fontsize=10)
        ax.set_ylabel('True label', fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)

        # 文本：计数图用整数 cm，归一化图用百分比
        thresh = cm_img.max() / 2.0 if cm_img.size > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if normalize:
                    val = cm_img[i, j] * 100.0
                    text = f"{val:.1f}%"
                    color = "white" if cm_img[i, j] > thresh else "black"
                else:
                    val = int(cm[i, j])
                    text = f"{val:d}"
                    color = "white" if cm_img[i, j] > thresh else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

        ax.set_ylim(len(classes) - 0.5, -0.5)  # 防止图像被裁掉
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def save_confusion_plots(self, y_true, y_pred, out_dir, model_name, dataset_tag):
        """
        计算并保存混淆矩阵图（计数 + 行归一化）。
        y_true, y_pred: 取值为 {-1, 1}
        out_dir: 输出根目录（建议传入 ./metrics/{label}/figs/ ）
        model_name: 模型名
        dataset_tag: 'train' or 'test'
        """
        from sklearn.metrics import confusion_matrix
        labels = [-1, 1]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        base = os.path.join(out_dir, f"{model_name}_{dataset_tag}_cm")
        self._plot_cm(cm, labels, f"{model_name} - {dataset_tag} (counts)", base + "_counts.png", normalize=False)
        self._plot_cm(cm, labels, f"{model_name} - {dataset_tag} (row-normalized)", base + "_norm.png", normalize=True)

    # ============ 各模型训练（仅输出指标 + 混淆矩阵） ============
    def run_Logistic(self, model_path, y_name, results):
        model = LogisticRegression(max_iter=1000)
        model.fit(self.x_train, self.y_train)

        y_tr_proba = model.predict_proba(self.x_train)[:, 1]   # P(y=1)
        y_te_proba = model.predict_proba(self.x_test)[:, 1]
        y_tr_pred = model.predict(self.x_train)
        y_te_pred = model.predict(self.x_test)

        acc_tr, f1_tr, auc_tr = self.compute_metrics(self.y_train, y_tr_proba, y_tr_pred)
        acc_te, f1_te, auc_te = self.compute_metrics(self.y_test, y_te_proba, y_te_pred)

        y_name_s = re.sub(r'[:：]', '_', y_name)
        self.make_dirs(model_path)
        joblib.dump(model, f'{model_path}lg_model_{y_name_s}.pkl')

        self.append_result(results, 'LogisticRegression', 'train', model.get_params(), acc_tr, f1_tr, auc_tr)
        self.append_result(results, 'LogisticRegression', 'test',  model.get_params(), acc_te, f1_te, auc_te)

        # === 保存混淆矩阵图 ===
        figs_dir = f'./metrics/{y_name}/figs/'
        self.save_confusion_plots(self.y_train, y_tr_pred, figs_dir, 'LogisticRegression', 'train')
        self.save_confusion_plots(self.y_test,  y_te_pred, figs_dir, 'LogisticRegression', 'test')

    def run_AdaBoost(self, model_path, y_name, results):
        ada = GridSearchCV(
            estimator=AdaBoostClassifier(),
            cv=5, param_grid=self.parameters['AdaBoostClassifier'],
            scoring='accuracy', n_jobs=2
        )
        ada.fit(self.x_train, self.y_train)
        best_est = ada.best_estimator_

        y_tr_proba = ada.predict_proba(self.x_train)[:, 1]
        y_te_proba = ada.predict_proba(self.x_test)[:, 1]
        y_tr_pred = ada.predict(self.x_train)
        y_te_pred = ada.predict(self.x_test)

        acc_tr, f1_tr, auc_tr = self.compute_metrics(self.y_train, y_tr_proba, y_tr_pred)
        acc_te, f1_te, auc_te = self.compute_metrics(self.y_test, y_te_proba, y_te_pred)

        y_name_s = re.sub(r'[:：]', '_', y_name)
        self.make_dirs(model_path)
        joblib.dump(best_est, f'{model_path}ada_model_{y_name_s}.pkl')

        self.append_result(results, 'AdaBoostClassifier', 'train', ada.best_params_, acc_tr, f1_tr, auc_tr)
        self.append_result(results, 'AdaBoostClassifier', 'test',  ada.best_params_, acc_te, f1_te, auc_te)

        # === 保存混淆矩阵图 ===
        figs_dir = f'./metrics/{y_name}/figs/'
        self.save_confusion_plots(self.y_train, y_tr_pred, figs_dir, 'AdaBoostClassifier', 'train')
        self.save_confusion_plots(self.y_test,  y_te_pred, figs_dir, 'AdaBoostClassifier', 'test')

    def run_RF(self, model_path, y_name, results):
        rf = GridSearchCV(
            estimator=RandomForestClassifier(),
            cv=5, param_grid=self.parameters['RandomForestClassifier'],
            scoring='accuracy', n_jobs=2
        )
        rf.fit(self.x_train, self.y_train)

        y_tr_proba = rf.predict_proba(self.x_train)[:, 1]
        y_te_proba = rf.predict_proba(self.x_test)[:, 1]
        y_tr_pred = rf.predict(self.x_train)
        y_te_pred = rf.predict(self.x_test)

        acc_tr, f1_tr, auc_tr = self.compute_metrics(self.y_train, y_tr_proba, y_tr_pred)
        acc_te, f1_te, auc_te = self.compute_metrics(self.y_test, y_te_proba, y_te_pred)

        y_name_s = re.sub(r'[:：]', '_', y_name)
        self.make_dirs(model_path)
        joblib.dump(rf.best_estimator_, f'{model_path}rf_model_{y_name_s}.pkl')

        self.append_result(results, 'RandomForestClassifier', 'train', rf.best_params_, acc_tr, f1_tr, auc_tr)
        self.append_result(results, 'RandomForestClassifier', 'test',  rf.best_params_, acc_te, f1_te, auc_te)

        # === 保存混淆矩阵图 ===
        figs_dir = f'./metrics/{y_name}/figs/'
        self.save_confusion_plots(self.y_train, y_tr_pred, figs_dir, 'RandomForestClassifier', 'train')
        self.save_confusion_plots(self.y_test,  y_te_pred, figs_dir, 'RandomForestClassifier', 'test')

    def run_XGB(self, model_path, y_name, results):
        # XGB 用 0/1 标签更稳妥
        xgb_y_train = self.y_train.replace(-1, 0)
        xgb_y_test  = self.y_test.replace(-1, 0)

        xgb = GridSearchCV(
            estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            cv=5, param_grid=self.parameters['XGBClassifier'],
            scoring='accuracy', n_jobs=2
        )
        xgb.fit(self.x_train, xgb_y_train)

        y_tr_proba = xgb.predict_proba(self.x_train)[:, 1]
        y_te_proba = xgb.predict_proba(self.x_test)[:, 1]
        y_tr_pred01 = xgb.predict(self.x_train)  # 0/1
        y_te_pred01 = xgb.predict(self.x_test)

        # 统一回到 -1/1 以复用 compute_metrics
        y_tr_pred = np.where(y_tr_pred01 == 1, 1, -1)
        y_te_pred = np.where(y_te_pred01 == 1, 1, -1)

        acc_tr, f1_tr, auc_tr = self.compute_metrics(self.y_train, y_tr_proba, y_tr_pred)
        acc_te, f1_te, auc_te = self.compute_metrics(self.y_test, y_te_proba, y_te_pred)

        y_name_s = re.sub(r'[:：]', '_', y_name)
        self.make_dirs(model_path)
        joblib.dump(xgb.best_estimator_, f'{model_path}xgb_model_{y_name_s}.pkl')

        self.append_result(results, 'XGBClassifier', 'train', xgb.best_params_, acc_tr, f1_tr, auc_tr)
        self.append_result(results, 'XGBClassifier', 'test',  xgb.best_params_, acc_te, f1_te, auc_te)

        # === 保存混淆矩阵图 ===
        figs_dir = f'./metrics/{y_name}/figs/'
        self.save_confusion_plots(self.y_train, y_tr_pred, figs_dir, 'XGBClassifier', 'train')
        self.save_confusion_plots(self.y_test,  y_te_pred, figs_dir, 'XGBClassifier', 'test')


if __name__ == '__main__':
    # 你原来是循环 3y/5y，这里保持一致（示例仅 5Y）
    for i in [5]:
        es = DataTrain()
        label = f'永续{i}yYTM'
        print('Target:', label)

        es.preprocess(label)
        es.train_test_split(0.8)
        es.fit_scaler('./model/', label)  # 仅缩放与保存 scaler

        results = []  # === 收集所有模型与指标 ===

        print('Step：1 LogisticRegression')
        es.run_Logistic('./model/', label, results)

        print('Step：2 AdaBoost')
        es.run_AdaBoost('./model/', label, results)

        print('Step：3 RandomForest')
        es.run_RF('./model/', label, results)

        print('Step：4 XGBoost')
        es.run_XGB('./model/', label, results)

        # === 保存到 Excel（不做任何回测/夏普） ===
        out_dir = f'./metrics/{label}/'
        os.makedirs(out_dir, exist_ok=True)
        df_out = pd.DataFrame(results, columns=['model', 'dataset', 'ACC', 'F1', 'AUC', 'best_params'])
        df_out.to_excel(os.path.join(out_dir, 'training_metrics.xlsx'), index=False)
        print(f'已输出到 {os.path.join(out_dir, "training_metrics.xlsx")}')
        print(f'混淆矩阵图片保存在 {os.path.join(out_dir, "figs")} 目录下')
