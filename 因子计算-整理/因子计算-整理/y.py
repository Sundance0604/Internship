import pandas as pd
import numpy as np
import time
import data
import importlib
import os
import glob
importlib.reload(data)


def calculate_y(date_start, date_end):
    factor_all = pd.DataFrame()
    calender_sjs = load_data_calender()
    trade_date = calender_sjs[(calender_sjs >= pd.to_datetime(date_start)) & (calender_sjs <= pd.to_datetime(date_end))]
    
    data = load_data('利率利差', date_start, date_end, mode='excel')
    data = data.loc[trade_date]
    data.fillna(0)

    factor_all['未来5日涨跌'] = data['国债_10Y'].shift(-5) > data['国债_10Y']
    factor_all['未来5日涨跌'] = factor_all['未来5日涨跌'].apply(lambda x: 1 if x else -1)

    factor_all['未来1日涨跌'] = data['国债_10Y'].shift(-1) > data['国债_10Y']
    factor_all['未来1日涨跌'] = factor_all['未来1日涨跌'].apply(lambda x: 1 if x else -1)
    
    return factor_all
    

def get_factor(factor_name, date_start, date_end):
    date_start = f'{date_start // 10000}-{date_start % 10000 // 100}-{date_start % 100}'
    date_end = f'{date_end // 10000}-{date_end % 10000 // 100}-{date_end % 100}'

    if factor_name == '预测标签':
        factor = calculate_y(date_start, date_end)
        # factor = factor.reindex(pd.date_range(start=date_start, end=date_end))

    return factor

def load_data(data_name, date_start, date_end, mode='excel'):
    if mode == 'excel':   
        path = f'{DATA_PATH}/{data_name}.xlsx'
        data = pd.read_excel(path, index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.reindex(pd.date_range(pd.to_datetime(date_start), pd.to_datetime(date_end)))
        return data

def save_and_update_factor(factor, factor_name, date_start, date_end):
    if not os.path.exists(FACTOR_PATH):
        os.makedirs(FACTOR_PATH)
    path = f'{FACTOR_PATH}/{factor_name}_{date_end}.xlsx'
    if not os.path.exists(path):
        factor.index = pd.to_datetime(factor.index)
        factor.to_excel(path, index=True)
    else:
        old_factor = pd.read_excel(path, index_col=0)
        old_factor.index = pd.to_datetime(old_factor.index)
        factor.index = pd.to_datetime(factor.index)
        factor = old_factor.combine_first(factor)
        factor.to_excel(path, index=True)

    print(f"类别: {factor_name}\n保存并更新数据成功\n时间: {date_start} - {date_end}")
    print(f"当前存在数据时间范围: {factor.index.min().strftime('%Y%m%d')} - {factor.index.max().strftime('%Y%m%d')}\n保存路径: {path}\n")

def load_data_calender():
    calender_sjs = pd.read_excel(f'{DATA_PATH}/交易日期.xlsx', index_col=0)
    calender_sjs = pd.Series(calender_sjs.index)
    calender_sjs = pd.to_datetime(calender_sjs)
    return calender_sjs

def update_factor(date_end):
    date_start = 20170101
    factor_name = '预测标签'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    

DATA_PATH = 'data' 
FACTOR_PATH = 'y' 
if __name__ == "__main__":

    date_start = 20170101
    date_end = int(time.strftime('%Y%m%d'))
    # 目前无权限, 需要在您电脑上运行的代码部分 #
    factor_name = '预测标签'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    ######################################