import numpy as np
import pandas as pd
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

df = pd.read_excel('基差\TF基差.xls')
df.set_index('日期', inplace=True)

df = df.diff(1)  # 全部先差分，后续再细化
df_clean = df.dropna().copy()
# 标签：下一期 Δ(基差) 是否 > 0
df_clean['value_sort'] = df_clean['基差'].shift(-1).apply(lambda x: 1 if x > 0 else 0)
df_clean = df_clean.iloc[:-1]
experiment = make_experiment(df_clean, target='value_sort')
estimator = experiment.run()
print(estimator)