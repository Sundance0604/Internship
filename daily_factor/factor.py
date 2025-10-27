# from WindPy import w
import pandas as pd
import numpy as np
import time
import data
import importlib
import os
import glob
importlib.reload(data)
DATA_PATH = 'data' 
FACTOR_PATH = 'factor' 
get_str_time = lambda x: pd.to_datetime(str(x), format='%Y%m%d').strftime('%Y-%m-%d')

def get_factor_y(factorType, date_start, date_end):
    data_all = data_dict["yield_rate_1"].load_data(date_start, date_end, mode='excel')
    data_all = data_all[['国债_10Y']]
    calender_sjs = data_calender.load_data_calender()
    trade_date = calender_sjs[(calender_sjs >= pd.to_datetime(date_start)) & (calender_sjs <= pd.to_datetime(date_end))]
    data_all = data_all.loc[trade_date]  
    data_all = data_all.reindex(pd.date_range(start=date_start, end=date_end))  
    return data_all

def calculate_factor_yield_rate(date_start, date_end):
    factor_all = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))
    
    data_all_1 = load_data('利率利差', date_start, date_end, mode='excel')
    
    factor_name = '国债_10Y-国开债_10Y' 
    factor_all[factor_name] = data_all_1['国债_10Y'] - data_all_1['国开债_10Y']
    
    factor_name = '国债_10Y-企业债AAA+_10Y'
    factor_all[factor_name] = data_all_1['国债_10Y'] - data_all_1['企业债AAA+_10Y']
    
    factor_name = '国债_10Y-中短期票据AAA+_10Y'
    factor_all[factor_name] = data_all_1['国债_10Y'] - data_all_1['中短期票据AAA+_10Y']
    
    factor_name = '国债_10Y-国债_1Y'
    factor_all[factor_name] = data_all_1['国债_10Y'] - data_all_1['国债_1Y']
    
    factor_name = '国债_10Y-国债_3Y'
    factor_all[factor_name] = data_all_1['国债_10Y'] - data_all_1['国债_3Y']
    
    factor_name = '国债_10Y-国债_5Y'
    factor_all[factor_name] = data_all_1['国债_10Y'] - data_all_1['国债_5Y']
    
    factor_name = '国债_10Y-国债_7Y'
    factor_all[factor_name] = data_all_1['国债_10Y'] - data_all_1['国债_7Y']
    
    factor_name = '国债_30Y-国债_10Y'
    factor_all[factor_name] = data_all_1['国债_30Y'] - data_all_1['国债_10Y']
    
    factor_name = '国开债_10Y-国债_10Y'
    factor_all[factor_name] = data_all_1['国开债_10Y'] - data_all_1['国债_10Y']

    factor_name = '国开债_5Y-国债_5Y'
    factor_all[factor_name] = data_all_1['国开债_5Y'] - data_all_1['国债_5Y']

    factor_name = '国开债_1Y-国债_1Y'
    factor_all[factor_name] = data_all_1['国开债_1Y'] - data_all_1['国债_1Y']

    # 信用利差
    factor_name = '中短期票据AAA+_{}-国开债{}'
    for i in ['1Y' ,'3Y', '5Y', '7Y', '10Y', '30Y']:
        factor_all[factor_name.format(i, i)] = data_all_1[f'中短期票据AAA+_{i}'] - data_all_1[f'国开债_{i}']

    factor_name = '国债_10Y-同业存单AAA_1Y'
    factor_all[factor_name] = data_all_1['国债_10Y'] - data_all_1['同业存单AAA_1Y']

    factor_name = '国开债_10Y-国开债_5Y'
    factor_all[factor_name] = data_all_1['国开债_10Y'] - data_all_1['国开债_5Y']

    factor_name = '国开债_10Y-国开债_1Y'
    factor_all[factor_name] = data_all_1['国开债_10Y'] - data_all_1['国开债_1Y']    

    factor_name = 'R007-R001'
    factor_all[factor_name] = data_all_1['R007'] - data_all_1['R001']   

    factor_name = '活跃券_10Y-活跃券_3Y'
    factor_all[factor_name] = data_all_1['活跃券_10Y_收盘到期收益率'] - data_all_1['活跃券_3Y_收盘到期收益率']   

    factor_name = '活跃券_10Y-活跃券_5Y'
    factor_all[factor_name] = data_all_1['活跃券_10Y_收盘到期收益率'] - data_all_1['活跃券_5Y_收盘到期收益率']   

    factor_name = '活跃券_10Y-活跃券_7Y'
    factor_all[factor_name] = data_all_1['活跃券_10Y_收盘到期收益率'] - data_all_1['活跃券_7Y_收盘到期收益率']   

    factor_name = '活跃券_10Y-国债_10Y'
    factor_all[factor_name] = data_all_1['活跃券_10Y_收盘到期收益率'] - data_all_1['国债_10Y']   
    
    factor_name = 'FR007_IRS_5Y-FR007_IRS_1Y'
    factor_all[factor_name] = data_all_1['FR007_IRS_5Y'] - data_all_1['FR007_IRS_1Y']   
    
    factor_name = 'FR007_IRS_1Y-FR007_IRS_9M'
    factor_all[factor_name] = data_all_1['FR007_IRS_1Y'] - data_all_1['FR007_IRS_9M']   
    
    factor_name = '国债_5Y-FR007_IRS_5Y'
    factor_all[factor_name] = data_all_1['国债_5Y'] - data_all_1['FR007_IRS_5Y']   
    
    data_all_2 = load_data('资金面', date_start, date_end, mode='excel')
    factor_name = 'SHIBOR_1Y-FR007_IRS_1Y'
    factor_all[factor_name] = data_all_2['SHIBOR_1Y'] - data_all_1['FR007_IRS_1Y']
   
    return factor_all

def calculate_factor_gzqh_technology(data, date_start, date_end):
    def calculate_momentum(data, period=10):
        return data['close'] - data['close'].shift(period)

    def calculate_cmo(data, period=10):
        delta = data['close'].diff()
        up = delta.apply(lambda x: max(x, 0))
        dn = delta.apply(lambda x: min(x, 0))
        sum_up = up.rolling(window=period, min_periods=1).sum()
        sum_dn = abs(dn.rolling(window=period, min_periods=1).sum())
        return 100 * (sum_up - sum_dn) / (sum_up + sum_dn)

    
    def calculate_kdj(data, period=9):
        low_min = data['low'].rolling(window=period, min_periods=1).min()
        high_max = data['high'].rolling(window=period, min_periods=1).max()
        rsv = (data['close'] - low_min) / (high_max - low_min) * 100
        k = [50] * len(rsv)
        d = [50] * len(rsv)
        for i in range(1, len(rsv)):
            k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]
            d[i] = (2 / 3) * d[i - 1] + (1 / 3) * k[i]
        j = [3 * k[i] - 2 * d[i] for i in range(len(k))]
        return pd.Series(j)

    def calculate_roc(data, period=10):
        n_period_ago = data['close'].shift(period)
        return (data['close'] - n_period_ago) / n_period_ago * 100

    def calculate_wr(data, period=9):
        highest_high = data['high'].rolling(window=period, min_periods=1).max()
        lowest_low = data['low'].rolling(window=period, min_periods=1).min()
        return (highest_high - data['close']) / (highest_high - lowest_low) * 100

    def calculate_sma(data, period=10):
        sma = data['close'].rolling(window=period).mean()
        return sma

    def calculate_trima(data, period=10):
        half_length = round(period / 2)
        sma = calculate_sma(data, period=half_length-1)
        sma.name = 'close'
        trima = calculate_sma(sma.to_frame(), half_length+1)
        return trima

    def calculate_obv(data):
        close_price_delta = data['close'] - data['close'].shift(1)
        close_price_sign = close_price_delta.apply(lambda x: np.sign(x))
        close_price_sign[0] = 1
        obv = close_price_sign * data['volume']
        obv = obv.cumsum()
        return obv

    def calculate_adi(data, period=10):
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        adi = clv * data['volume']
        adi = adi.cumsum()
        return adi

    def calculate_tr(data):
        tr1 = data['high'] - data['low']
        tr2 = data['high'] - data['close'].shift(1)
        tr2 = tr2.abs()
        tr2 = tr2.fillna(0)
        tr3 = data['low'] - data['close'].shift(1)
        tr3 = tr3.abs()
        tr3 = tr3.fillna(0)
        tr = pd.Series(zip(tr1, tr2, tr3)).apply(lambda x: max(x))
        return tr

    def calculate_atr(data, period=10):
        tr = calculate_tr(data)
        atr = pd.Series(np.zeros(len(tr)))
        for t in range(len(tr)):
            if t == 0:
                atr[t] = tr[t]
            else:
                atr[t] = (atr[t-1] * (period - 1) + tr[t]) / period
        return atr

    def calculate_vwap(data):
        return (data['high'] + data['low'] + data['close']) / 3

    def calculate_low2high(data):
        return data['low'] / data['high']

    def calculate_vwap2close(data):
        vwap = calculate_vwap(data)
        return vwap / data['close']

    def calculate_kmid(data):
        return (data['close'] - data['open']) / data['open']

    def calculate_klen(data):
        return (data['high'] - data['low']) / data['open']

    def calculate_kmid2(data):
        return (data['close'] - data['open']) / (data['high'] - data['low'])

    def calculate_kup2(data):
        return (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])

    def calculate_klow2(data):
        return (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])

    def calculate_ksft2(data):
        return (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low'])
    
    calender_sjs = load_data_calender()
    trade_date = calender_sjs[(calender_sjs >= pd.to_datetime(date_start)) & (calender_sjs <= pd.to_datetime(date_end))]
    data = data.loc[trade_date]
    data = data.reset_index(drop=True)
    
    first_non_null_index = data['high'].first_valid_index()
    # print(first_non_null_index)
    
    trade_date_non_null = trade_date[first_non_null_index:]
    data = data.iloc[first_non_null_index:,:]
    data = data.reset_index(drop=True)
    data = data.ffill()
    old_data = data.copy()
    periods = [5, 10, 20]
    for period in periods:
        data['SMA_{p}'.format(p=period)] = calculate_sma(old_data, period=period)
        data['TRIMA_{p}'.format(p=period)]= calculate_trima(old_data, period=period)
        data['ROC_{p}'.format(p=period)]= calculate_roc(old_data, period=period)
        data['CMO_{p}'.format(p=period)]= calculate_cmo(old_data, period=period)
        data['WR_{p}'.format(p=period)]= calculate_wr(old_data, period=period)
        data['KDJ_{p}'.format(p=period)]= calculate_kdj(old_data, period=period)
    data['MOMENTUM']= calculate_momentum(old_data)
    data['OBV']= calculate_obv(old_data)
    data['ADI']= calculate_adi(old_data)
    data['ATR']= calculate_atr(old_data)
    data['VWAP']= calculate_vwap(old_data)
    data['LOW2HIGH']= calculate_low2high(old_data)
    data['VWAP2CLOSE']= calculate_vwap2close(old_data)
    data['KMID']= calculate_kmid(old_data)
    data['KLEN']= calculate_klen(old_data)
    data['KMID2']= calculate_kmid2(old_data)
    data['KUP2']= calculate_kup2(old_data)
    data['KLOW2']= calculate_klow2(old_data)
    data['KSFT2']= calculate_ksft2(old_data)
    
    data.index = trade_date_non_null
    data = data.reindex(pd.date_range(start=date_start, end=date_end))
    # print(data.columns.to_list())
    return data

def calculate_factor_capital(date_start, date_end):
    def calculate_ma(series, period=5):
        return series.rolling(window=period).mean()
    
    def calculate_quantile(series, period=1):
        def get_quantile_year(x, year_period):
            last_year = x.index[-1] - pd.DateOffset(years=year_period)
            return x[x.index > last_year].rank(pct=True).iloc[-1] * 100 # 百分位数
        return series.rolling(window=period*365, min_periods=1).apply(lambda x: get_quantile_year(x, period)) 
    
    def calculate_std(series, period=5):
        return series.rolling(window=period).std()

    data_all = load_data('资金面', date_start, date_end, mode='excel')
    
    calender_sjs = load_data_calender()
    trade_date = calender_sjs[(calender_sjs >= pd.to_datetime(date_start)) & (calender_sjs <= pd.to_datetime(date_end))]
    
    data_all = data_all.loc[trade_date]
    factor_all = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))
    data_all = data_all.ffill()

    factors = [factor_all]
    for i, bond in enumerate(['R007', 'R001', 'DR007', 'DR001', 'FR007', 'FR001', 'SHIROR_3M', 'SHIBOR_1Y']):
        temp_name = f"{bond}_5日移动平均"
        factor = calculate_ma(data_all[bond], period=5)
        factor.name = temp_name
        factor = pd.DataFrame(factor)
        factor = factor.reindex(pd.date_range(start=date_start, end=date_end))
        factors.append(factor)
    factor_all = pd.concat(factors, axis=1)

    factors = [factor_all]
    for i, bond in enumerate(['R007', 'R001', 'DR007', 'DR001', 'FR007', 'FR001', 'SHIROR_3M', 'SHIBOR_1Y']):
        temp_name = f"{bond}_5日移动平均_近1Y历史分位数"
        temp_factor_name = f"{bond}_5日移动平均"
        factor = calculate_quantile(factor_all[temp_factor_name], period=1)
        factor.name = temp_name
        factor = pd.DataFrame(factor)
        factor = factor.reindex(pd.date_range(start=date_start, end=date_end))
        factors.append(factor)  
    factor_all = pd.concat(factors, axis=1)
    
    factors = [factor_all]
    for i, bond in enumerate(['R007', 'R001', 'DR007', 'DR001', 'FR007', 'FR001', 'SHIROR_3M', 'SHIBOR_1Y']):
        temp_name = f"{bond}_5日标准差"
        factor = calculate_std(data_all[bond], period=5)
        factor.name = temp_name
        factor = pd.DataFrame(factor)
        factor = factor.reindex(pd.date_range(start=date_start, end=date_end))
        factors.append(factor)
    factor_all = pd.concat(factors, axis=1)
    
    factors = [factor_all]
    for i, bond in enumerate(['R007', 'R001', 'DR007', 'DR001', 'FR007', 'FR001', 'SHIROR_3M', 'SHIBOR_1Y']):
        temp_name = f"{bond}_5日标准差_近1Y历史分位数"
        temp_factor_name = f"{bond}_5日标准差"
        factor = calculate_quantile(factor_all[temp_factor_name], period=1)
        factor.name = temp_name
        factor = pd.DataFrame(factor)
        factor = factor.reindex(pd.date_range(start=date_start, end=date_end))
        factors.append(factor)  
    factor_all = pd.concat(factors, axis=1)

    return factor_all

def calculate_factor_gzqh_price_diff(date_start, date_end):
    data_all_1 = load_data('国债期货价差', date_start, date_end, mode='excel')
    factor_all = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))
    
    factor_name = '国债期货_活跃_{}_均价-国债期货_次活跃_{}_均价'
    for i in ['2Y', '5Y', '10Y', '30Y']:
        factor_all[factor_name.format(i, i)] = data_all_1[f'国债期货_活跃_{i}_均价'] - data_all_1[f'国债期货_次活跃_{i}_均价']

    factor_all = factor_all.reindex(pd.date_range(start=date_start, end=date_end))

    return factor_all

def calculate_factor_stock(date_start, date_end):
    data_all = load_data('股市', date_start, date_end, mode='excel')
    data_all = data_all.reindex(pd.date_range(start=date_start, end=date_end))
    data_all = data_all.ffill()

    factor_all = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))
    columns = data_all.columns.to_list()
    factor_all[[f'{col}_1D涨幅' for col in columns]] = data_all[[f'{col}' for col in columns]].pct_change(periods=1)
    factor_all[[f'{col}_1M涨幅' for col in columns]] = data_all[[f'{col}' for col in columns]].pct_change(periods=30)
    
    factor_all = factor_all.reindex(pd.date_range(start=date_start, end=date_end))
    return factor_all   

def calculate_factor_macro(date_start, date_end):
    data_all = load_data('宏观', date_start, date_end, mode='excel')
    data_all = data_all.reindex(pd.date_range(start=date_start, end=date_end))
    data_all = data_all.ffill()
    factor_all = data_all.copy()
    return factor_all

def calculate_factor_gz_technology( date_start, date_end):
    data = load_data('利率利差', date_start, date_end, mode='excel')
    calender_sjs = load_data_calender()
    trade_date = calender_sjs[(calender_sjs >= pd.to_datetime(date_start)) & (calender_sjs <= pd.to_datetime(date_end))]
    data = data.loc[trade_date]
    def get_norm_mom(x):
        ma1 = x.rolling(window=10, min_periods=1).mean()
        ma2 = x.rolling(window=30, min_periods=1).mean()
        std2 = x.rolling(window=30, min_periods=1).std()
        res = (ma1 - ma2) / std2
        return res
    def get_rate(x, period):
        x = x.reindex(pd.date_range(start=date_start, end=date_end))
        x = x.ffill()
        res = x.shift(-period) / x - 1
        res = res.loc[trade_date]
        return res
    def get_macd(x):
        short_ema = x.ewm(span=12, adjust=False).mean()
        long_ema = x.ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal
    
    factor_all = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))
    category = ['国债']
    attribute = ['1Y', '5Y', '7Y', '10Y', '30Y']
    days = [5, 30, 90]
    for c in category:
        for a in attribute:
            for d in days:
                temp_name = f'{c}_{a}_收益率变动_{d}D'
                factor_all[temp_name] = get_rate(data[f'{c}_{a}'], d)

    for c in category:
        for a in attribute:
            temp_name = f'{c}_{a}_动量'
            factor_all[temp_name] = get_norm_mom(data[f'{c}_{a}'])

    for c in category:
        for a in attribute:
            temp_name = f'{c}_{a}_macd'
            factor_all[temp_name] = get_macd(data[f'{c}_{a}'])
            
    return factor_all

def calculate_factor_calender(date_start, date_end):
    calender_sjs = load_data_calender()
    trade_date = calender_sjs[(calender_sjs >= pd.to_datetime(date_start)) & (calender_sjs <= pd.to_datetime(date_end))]
    factor_all = pd.DataFrame(index=pd.date_range(start=date_start, end=date_end))
    factor_all['是否交易日'] = factor_all.index.isin(trade_date).astype(int)
    factor_all['月'] = factor_all.index.month.astype(int)
    factor_all['日'] = factor_all.index.day.astype(int)
    factor_all['是否15号'] = (factor_all.index.day == 15).astype(int)
    factor_all['是否20号'] = (factor_all.index.day == 20).astype(int)
    factor_all['星期'] = factor_all.index.weekday + 1
    return factor_all

def get_factor(factor_name, date_start, date_end):
    date_start = f'{date_start // 10000}-{date_start % 10000 // 100}-{date_start % 100}'
    date_end = f'{date_end // 10000}-{date_end % 10000 // 100}-{date_end % 100}'

    if factor_name == '利率利差':
        factor = calculate_factor_yield_rate(date_start, date_end)
         
    elif factor_name == '国债期货技术指标':
        data_all = load_data('国债期货技术指标', date_start, date_end, mode='excel')
        factor = pd.DataFrame()
        year = ['2Y', '5Y', '10Y', '30Y']
        for y in year:
            cols = [f'国债期货_{y}_open', f'国债期货_{y}_close', f'国债期货_{y}_high', f'国债期货_{y}_low', f'国债期货_{y}_volume']
            data = data_all[cols]
            data.columns=['open','close','high','low','volume']
            factor_temp = calculate_factor_gzqh_technology(data, date_start, date_end)
            factor_temp.columns = [f'国债期货_{y}_' + f.lower() for f in factor_temp.columns.to_list()]
            factor = pd.concat([factor, factor_temp], axis=1)
        factor = factor.reindex(pd.date_range(start=date_start, end=date_end))
    
    elif factor_name == '资金面':
        factor = calculate_factor_capital(date_start, date_end)
        factor = factor.reindex(pd.date_range(start=date_start, end=date_end))
        
    elif factor_name == '国债期货价差':
        factor = calculate_factor_gzqh_price_diff(date_start, date_end)
        factor = factor.reindex(pd.date_range(start=date_start, end=date_end))
        
    elif factor_name == '国债技术指标':
        factor = calculate_factor_gz_technology(date_start, date_end)
        factor = factor.reindex(pd.date_range(start=date_start, end=date_end))

    elif factor_name == '股市':
        factor = calculate_factor_stock(date_start, date_end)
        factor = factor.reindex(pd.date_range(start=date_start, end=date_end))

    elif factor_name == '宏观':
        factor = calculate_factor_macro(date_start, date_end)
        factor = factor.reindex(pd.date_range(start=date_start, end=date_end))
    
    elif factor_name == '交易日期':
        factor = calculate_factor_calender(date_start, date_end)
        factor = factor.reindex(pd.date_range(start=date_start, end=date_end))

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

def save_combine_factor(date_start, date_end):
    # 获取FACTOR_PATH下所有xlsx文件路径
    str_date_start = f'{date_start // 10000}-{date_start % 10000 // 100}-{date_start % 100}'
    str_date_end = f'{date_end // 10000}-{date_end % 10000 // 100}-{date_end % 100}'
    
    factor_files = glob.glob(f"{FACTOR_PATH}/*_{date_end}.xlsx")
    factors = [pd.read_excel(file, index_col=0) for file in factor_files]
    combined_factors = pd.concat(factors, axis=1)
    combined_factors.to_excel(f"{FACTOR_PATH}/{date_end}-因子.xlsx", index=True)
 
    calender_sjs = load_data_calender()
    trade_date = calender_sjs[(calender_sjs >= pd.to_datetime(str_date_start)) & (calender_sjs <= pd.to_datetime(str_date_end))]
    combined_factors = combined_factors.loc[trade_date]
    combined_factors.to_excel(f"{FACTOR_PATH}/{date_end}-因子交易日.xlsx", index=True)
    
    # 生成因子清单
    factor_files = glob.glob(f"{FACTOR_PATH}/*_{date_end}.xlsx")
    factor_list = []
    for file in factor_files:
        df = pd.read_excel(file, index_col=0)
        factor_category = os.path.basename(file).replace(f"_{date_end}.xlsx", "")  # 去除后缀获取因子类别
        for column in df.columns:
            factor_list.append([factor_category, column])  # 因子类别和因子名称
    factor_df = pd.DataFrame(factor_list, columns=["因子类别", "因子名称"])
    factor_df.to_excel(f"{FACTOR_PATH}/{date_end}-因子清单.xlsx", index=True)
    return 

def update_factor(date_end):
    date_start = 20170101
    factor_name = '利率利差'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    
    factor_name = '国债期货技术指标'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)

    factor_name = '国债技术指标'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)

    factor_name = '资金面'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)

    factor_name = '国债期货价差'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    
    factor_name = '股市'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    
    factor_name = '宏观'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    
    factor_name = '交易日期'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    save_combine_factor(date_start, date_end)


def run_factor(date_start=20170101,date_end=int(time.strftime('%Y%m%d'))):
    if type(date_start) == str:
        date_start = int(pd.to_datetime(date_start).strftime('%Y%m%d'))
    if type(date_end) == str:
        date_end =  int(pd.to_datetime(date_end).strftime('%Y%m%d'))

    # 目前无权限, 需要在您电脑上运行的代码部分 #
    factor_name = '利率利差'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    
    factor_name = '国债技术指标'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    ######################################
    factor_name = '国债期货技术指标'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)

    factor_name = '资金面'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)

    factor_name = '国债期货价差'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    
    factor_name = '股市'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    
    factor_name = '宏观'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    
    factor_name = '交易日期'
    factor = get_factor(factor_name, date_start, date_end)
    save_and_update_factor(factor, factor_name, date_start, date_end)
    save_combine_factor(date_start, date_end)
