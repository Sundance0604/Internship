from WindPy import w
import pandas as pd
import h5py
import numpy as np
import itertools
from tqdm import tqdm
import os
import time
import importlib

DATA_PATH = './data' 
# 分类别读入数据, 保存, 计算因子, 保存至本地Excel中
def get_data(data_name, date_start, date_end):
    date_start = f'{date_start // 10000}-{date_start % 10000 // 100}-{date_start % 100}'
    date_end = f'{date_end // 10000}-{date_end % 10000 // 100}-{date_end % 100}'
    data = pd.DataFrame()
    if data_name == '利率利差':
        data_col_name = ['国债_1Y', '国债_3Y', '国债_5Y', '国债_7Y', '国债_10Y', '国债_30Y', 
                         '国开债_1Y', '国开债_3Y', '国开债_5Y', '国开债_7Y', '国开债_10Y', '国开债_30Y', 
                         '企业债AAA+_6M', '企业债AAA+_1Y', '企业债AAA+_3Y', '企业债AAA+_5Y', '企业债AAA+_7Y', '企业债AAA+_10Y', '企业债AAA+_30Y', 
                         '同业存单AAA_6M', '同业存单AAA_1Y', 
                         '中短期票据AAA+_0D', '中短期票据AAA+_7D', '中短期票据AAA+_1M', '中短期票据AAA+_6M', '中短期票据AAA+_1Y', '中短期票据AAA+_3Y', '中短期票据AAA+_5Y', '中短期票据AAA+_7Y', '中短期票据AAA+_10Y', '中短期票据AAA+_30Y']
        code = ['S0059744', 'S0059746', 'S0059747', 'S0059748', 'S0059749', 'S0059752',
                'M1004263', 'M1004265', 'M1004267', 'M1004269', 'M1004271', 'M1004274',
                'S0059770', 'S0059771', 'S0059773', 'S0059774', 'S0059775', 'S0059776', 'S0059779',
                'M1010883', 'M1010885',
                'M1004164', 'M1000510', 'M1000512', 'M0067156', 'M0067158', 'M0067160', 'M0067162', 'M0067164', 'M0067166', 'M1006927']
        for col, c in zip(data_col_name, code):
            res = w.edb(c, date_start, date_end, usedf=True)[1]
            res.index = pd.to_datetime(res.index)
            if len(res) == 0:
                res = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end), columns=[col])
            else:
                res.columns = [col]
            res = res.reindex(pd.date_range(start=date_start, end=date_end))
            data = pd.concat([data, res], axis=1)
        data = data.reindex(pd.date_range(start=date_start, end=date_end))
        
        data_col_name = ['活跃券_3Y_收盘到期收益率', '活跃券_5Y_收盘到期收益率', '活跃券_7Y_收盘到期收益率', '活跃券_10Y_收盘到期收益率']
        code = ['TB3Y.WI', 'TB5Y.WI', 'TB7Y.WI', 'TB10Y.WI']
        for col, c in zip(data_col_name, code):
            res =  w.wsd(c, 'ytm_b', date_start, date_end, "returnType=1;Days=ALLDAYS", usedf=True)[1]
            res.index = pd.to_datetime(res.index)
            res.columns = [col]
            res = res.reindex(pd.date_range(start=date_start, end=date_end))
            data = pd.concat([data, res], axis=1)
        data = data.reindex(pd.date_range(start=date_start, end=date_end))
        
        data_col_name = ['R001', 'R007']
        code = ["M1001794", "M1001795"]
        for col, c in zip(data_col_name, code):
            res = w.edb(c, date_start, date_end, usedf=True)[1]
            res.index = pd.to_datetime(res.index)
            res.columns = [col]
            res = res.reindex(pd.date_range(start=date_start, end=date_end))
            data = pd.concat([data, res], axis=1)
        data = data.reindex(pd.date_range(start=date_start, end=date_end))

        data_col_name = ['FR007_IRS_9M', 'FR007_IRS_1Y', 'FR007_IRS_5Y', 'SHIROR_3M_IRS_1Y', 'SHIROR_3M_IRS_5Y']
        code = ["M0048485", "M0048486", "M0048490",
                 "M0048499", "M0075930"]
        for col, c in zip(data_col_name, code):
            res = w.edb(c, date_start, date_end, usedf=True)[1]
            res.index = pd.to_datetime(res.index)
            res.columns = [col]
            res = res.reindex(pd.date_range(start=date_start, end=date_end))
            data = pd.concat([data, res], axis=1)
        data = data.reindex(pd.date_range(start=date_start, end=date_end))
         
    elif data_name == '国债期货技术指标':
        data_col_name = ['国债期货_2Y', '国债期货_5Y', '国债期货_10Y', '国债期货_30Y']
        code = ["TS.CFE", "TF.CFE", "T.CFE", "TL.CFE"]
        field = ['open','close','high','low','volume']
        for col, c in zip(data_col_name, code):
            field_join = ','.join(field)
            res = w.wsd(c, field_join, date_start, date_end, "Days=ALLDAYS", usedf=True)[1]
            res.index = pd.to_datetime(res.index)
            res.columns = [f'{col}_{f}' for f in field]
            res = res.reindex(pd.date_range(start=date_start, end=date_end))
            data = pd.concat([data, res], axis=1)
        data = data.reindex(pd.date_range(start=date_start, end=date_end))
        
    
    elif data_name == '资金面':
        data_col_name = ['R007', 'R001', 'DR007', 'DR001', 'FR007', 'FR001', 'SHIROR_3M', 'SHIBOR_1Y']
        code = ["M1001795", "M1001794", "M1006337", "M1006336", "M1001846", "M1001845",  "M0017142", "M0017145"]
        res = w.edb(code, date_start, date_end, "", usedf=True)[1]
        for col, c in zip(data_col_name, code):
            res = w.edb(c, date_start, date_end, "", usedf=True)[1]
            res.index = pd.to_datetime(res.index)
            res.columns = [col]
            res = res.reindex(pd.date_range(start=date_start, end=date_end))
            data = pd.concat([data, res], axis=1)
        data = data.reindex(pd.date_range(start=date_start, end=date_end))
        
    elif data_name == '国债期货价差':
        data_col_name = ['国债期货_活跃_2Y', '国债期货_活跃_5Y', '国债期货_活跃_10Y', '国债期货_活跃_30Y', 
                        '国债期货_次活跃_2Y', '国债期货_次活跃_5Y', '国债期货_次活跃_10Y', '国债期货_次活跃_30Y']
        code = ["TS.CFE", "TF.CFE", "T.CFE", "TL.CFE",
                "TS_S.CFE", "TF_S.CFE", "T_S.CFE", "TL_S.CFE"]
        field = ['vwap']
        for col, c in zip(data_col_name, code):
            field_join = ','.join(field)
            res = w.wsd(c, 'vwap', date_start, date_end, "Days=ALLDAYS", usedf=True)[1]
            res.index = pd.to_datetime(res.index)
            res.columns = [f'{col}_均价' for f in field]
            res = res.reindex(pd.date_range(start=date_start, end=date_end))
            data = pd.concat([data, res], axis=1)
        data = data.reindex(pd.date_range(start=date_start, end=date_end))
    elif data_name == '股市':
        data_col_name = ['万得全A', '上证50', '中证500', '中证1000']
        code = ["881001.WI", "000016.SH","000905.SH","000852.SH"]
        field = ['close']
        for col, c in zip(data_col_name, code):
            field_join = ','.join(field)
            res = w.wsd(c, 'close', date_start, date_end, "Days=ALLDAYS", usedf=True)[1]
            res.index = pd.to_datetime(res.index)
            res.columns = [f'{col}_收盘价' for f in field]
            res = res.reindex(pd.date_range(start=date_start, end=date_end))
            data = pd.concat([data, res], axis=1)
        data = data.reindex(pd.date_range(start=date_start, end=date_end))
    elif data_name == '宏观':
        data_col_name = ['GDP不变价同比', 'GDP不变价当季值', '美元指数', '美元兑人民币即期汇率', 'MLF1Y', 'OMO007数量']
        code = ["M0039354", "M5567889", "M0000271", "M0067855", "M0329545", "M0041372"]
        for col, c in zip(data_col_name, code):
            res = w.edb(c, date_start, date_end, "Fill=Previous", usedf=True)[1]
            print(res)
            res.index = pd.to_datetime(res.index)
            print(res)
            res.columns = [col]
            res = res.reindex(pd.date_range(start=date_start, end=date_end))
            data = pd.concat([data, res], axis=1)
        data = data.reindex(pd.date_range(start=date_start, end=date_end))
    elif data_name == '交易日期':
        data = w.tdays(date_start , date_end, "", usedf=True)[1]
        data.index = pd.to_datetime(data.index)
    return data

def save_and_update_data(data, data_name, date_start, date_end):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    path = f'{DATA_PATH}/{data_name}.xlsx'
    if not os.path.exists(path):
        data.index = pd.to_datetime(data.index)
        data.to_excel(path, index=True)
    else:
        old_data = pd.read_excel(path, index_col=0)
        old_data.index = pd.to_datetime(old_data.index)
        data.index = pd.to_datetime(data.index)
        data = old_data.combine_first(data)
        data.to_excel(path, index=True)

    print(f"类别: {data_name}\n保存并更新数据成功\n时间: {date_start} - {date_end}")
    print(f"当前存在数据时间范围: {data.index.min().strftime('%Y%m%d')} - {data.index.max().strftime('%Y%m%d')}\n保存路径: {path}\n")

def update_data(date_end):
    date_start = (date_end // 100) * 100 + 1
    
    data_name = '利率利差'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end)
    
    data_name = '国债期货技术指标'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end)
    
    data_name = '资金面'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end)  
    
    data_name = '国债期货价差'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end) 
    
    data_name = '股市'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end) 
    
    data_name = '宏观'
    data = get_data(data_name, date_start-40000, date_end)
    save_and_update_data(data, data_name, date_start, date_end) 
    
    data_name = '交易日期'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end) 
def run_data():
    try:
        w.start()
        w.isconnected()
    except Exception as e:
        print(e)
    ##### 修改以下数据 #####
    # 因子保存路径
    date_start = 20170101
    date_end = int(time.strftime('%Y%m%d'))
    # 获取当前日期的 yyyymmdd 格式
    ######################
    try:
        w.start()
        w.isconnected()
    except Exception as e:
        print(e)
    # 设置工作路径为代码文件夹
    print(f"当前工作路径: {os.getcwd()}")
    # 目前无权限, 需要在您电脑上运行的代码部分 #
    data_name = '利率利差'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end)
    
    data_name = '国债期货技术指标'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end)
    ######################################
    
    data_name = '资金面'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end)  
    
    data_name = '国债期货价差'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end) 
    
    data_name = '股市'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end) 
    
    data_name = '宏观'
    data = get_data(data_name, date_start-40000, date_end)
    save_and_update_data(data, data_name, date_start, date_end) 
    
    data_name = '交易日期'
    data = get_data(data_name, date_start, date_end)
    save_and_update_data(data, data_name, date_start, date_end) 