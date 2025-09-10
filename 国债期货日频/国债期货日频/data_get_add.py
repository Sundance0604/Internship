#!/usr/bin/env python
# coding: utf-8

# In[3]:
'''
获取的基本数据，和宏观经济指标，在这里已经进行了ffill处理。
'''
from WindPy import *
import pandas as pd
import numpy as np
def fetch_wind_data():
    # 启动WindPy接口
    ret = w.start()
    if not ret.ErrorCode == 0:
        raise Exception("WindPy启动失败")
    
    # 检查是否连接成功
    ret = w.isconnected()
    if not ret:
        raise Exception("WindPy未连接")

    # 提取TF和T的初始行情数据
    TF5 = w.wsd('TF.CFE', ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'volume'], '2014-05-05', usedf=True)
    TF = TF5[1].copy()
    TF10 = w.wsd('T.CFE', ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'volume'], '2015-03-20', usedf=True)
    T = TF10[1].copy()
    TL30 = w.wsd('TL.CFE', ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'volume'], '2023-04-22', usedf=True)
    TL = TL30[1].copy()

    # 宏观数据，开始时间是2014-01-02，初始的缺失值用0填充
    ids = ["S0029657", "S0059749", "S0059744", "S0059747", "M0067855", "G0000886", "G0000889", "G0000891", 
           "G1306752", "G0006352", "G0006353", "M0000612", "M0001227", "M0074417", "M1004524", "M1004520", 
           "M0048486", "M0048488", "M0048490", "M0096868", "M0017142", "M0017141", "M0017145", "M1001854", 
           "S0181383", "S5808575", "S0031525", "M5525763", "M0041653", "M0041652",
           #新加的因子（27个）
          "M1004263","M1004267","M1004271","L4530250","U0737658","O8195887",
            "W6109272","U5267974","Y1667217","W1775339","U9659646","A0239140",
           "F2827408","Z6496161","Y4138099","M0041372","M0041374",
           "M0041378","M0329655","M1004899","M1004900","M1004902",'S0059745',
           'S0059752','M1004264','M1004274','W8696400']
    macro = w.edb(ids, beginTime="2014-01-02", ShowBlank=0, usedf=True)

    #将macro和行情数据拼接在一起
    macro_final = macro[1].copy()

    
    T= T.reset_index()
    T = T.rename(columns={'index': 'date'})
    TF= TF.reset_index()
    TF= TF.rename(columns={'index': 'date'})
    TL= TL.reset_index()
    TL= TL.rename(columns={'index': 'date'})   
    
    TF.columns = [col.lower() for col in TF.columns]
    T.columns = [col.lower() for col in T.columns]
    TL.columns = [col.lower() for col in TL.columns]


    #向前填充macro_final缺失值
    
    macro_final.replace(0, np.nan, inplace=True)  # Replace 0 with NA for proper forward fill
    macro_final.fillna(method='ffill', inplace=True)    
    macro_final = macro_final.fillna(macro_final.mean())
    macro_final= macro_final.reset_index()
    macro_final= macro_final.rename(columns={'index': 'date'})
    #分别合并TF,T
    merged_TF = pd.merge(TF, macro_final, on='date', how='left') 


    merged_T = pd.merge(T, macro_final, on='date', how='left')
    merged_TL = pd.merge(TL, macro_final, on='date', how='left')

    
    return merged_TF, merged_T, merged_TL
if __name__ == "__main__":
    # 这部分代码只有在直接运行这个 notebook 时才会执行
    TF, T, TL = fetch_wind_data()

    print(TL.tail(5))


# # 宏观指标说明及指标ID

# ![图片.png](attachment:8aedc330-ee8a-4bc4-abec-aae8abef2abc.png)
# ![图片.png](attachment:c84eb8a2-7f4f-40ce-9bf5-d7a77481295d.png)
# ![图片.png](attachment:3ec57728-93b4-4298-9b7d-33c742f61c06.png)
# ![图片.png](attachment:1d4d1220-92cd-47e1-a6e0-fdbcea0ee973.png)

# 
