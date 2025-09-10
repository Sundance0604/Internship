import pandas as pd
import talib as ta
import numpy as np
from WindPy import w
import datetime
import os
# 均线策略


# 计算指标
def get_indicator(data):
    # 5日均线上穿20日均线，开多持仓22个交易日
    data['ma5'] = data['close'].rolling(window=5).mean()
    data['ma20'] = data['close'].rolling(window=20).mean()
    data['ma_signal'] = np.where((data['ma5'] > data['ma20'])&(data['ma5'].shift(1)<data['ma20'].shift(1)), 1, np.nan)
    data['ma_signal'] = data['ma_signal'].ffill(limit=21).fillna(0)

    # KDJ反转，k线下穿D线时开多，上穿D线时平多
    data['rsv'] = (data['close'] - data['low'].rolling(window=9).min()) / (data['high'].rolling(window=9).max() - data['low'].rolling(window=9).min())
    data['k'] = data['rsv'].ewm(com=2).mean()
    data['d'] = data['k'].ewm(com=2).mean()
    data['kdj_signal'] = np.where((data['k'] < data['d']), 1, 0)
    ##data['kdj_signal'] = np.where((data['k'].shift(1) > data['d'].shift(1)) & (data['k'] < data['d']), 1, 0)


    # CCI CCI低于-100开多，回到-100平多，CCI参数设为20
    data['cci'] = ta.CCI(data['high'], data['low'], data['close'])
    data['cci_signal'] = np.where((data['cci'] < -100), 1, 0)
    ##data['cci_signal'] = np.where((data['cci'].shift(1) >= -100) & (data['cci'] < -100), 1, 0)

    # RSI双均线，快线下穿慢线开多，上穿平多
    data['rsi_fast'] = ta.RSI(data['close'], timeperiod=9)
    data['rsi_slow'] = ta.RSI(data['close'], timeperiod=14)
    data['rsi_signal'] = np.where((data['rsi_fast'] < data['rsi_slow']), 1, 0)
    ##data['rsi_signal'] = np.where((data['rsi_fast'].shift(1) > data['rsi_slow'].shift(1)) & (data['rsi_fast'] < data['rsi_slow']), 1, 0)

    # 布林带 乖离率联合指标
    # 指数移动平均
    # 半衰期20个交易日
    # 过去252个交易日分别计算布林带、乖离率0.1，0.3分位数
    # 布林带指标和乖离率同时低于0.1分位数时开多，任一指标回到0.3分位数时平多    
    halflife = 20
    data['bb_indicator'] = (data['close'] - data['close'].ewm(halflife=halflife).mean())/(data['close'].ewm(halflife=halflife).std())
    data['deviation'] = (data['close'] - data['close'].ewm(halflife=halflife).mean())/data['close'].ewm(halflife=halflife).mean()
    position_series = generate_position_series(data['bb_indicator'], data['deviation'], window_size=252, upper_open=.9, lower_open=.1, upper_close=.8, lower_close=.3)
    position_series.index = data.index
    position_series = position_series[position_series>=0].reindex(data.index).fillna(0)
    data['bb_signal'] = position_series

    data['cci'] = ta.CCI(data['high'], data['low'], data['close'],)
    data['cci_反转_-80']= np.where((data['cci'] < -80), 1, 0)
    ##data['cci_反转_-80'] = np.where((data['cci'].shift(1) < -80) & (data['cci'] >= -80), 1, 0)  # 向上反转
    data['cci_反转_-10']= np.where((data['cci'] <-10), 1, 0)
    ##data['cci_反转_-80'] = np.where((data['cci'].shift(1) < -10) & (data['cci'] >= -10), 1, 0)  # 向上反转

    data['rsi_反转_-0.5'] = np.where((data['rsi_fast'] - data['rsi_slow']< -0.5), 1, 0)
    ##delta = data['rsi_fast'] - data['rsi_slow']
    ##data['rsi_反转_-0.5'] = np.where((delta.shift(1) > -0.5) & (delta < -0.5), 1, 0)

    data['cci_quantile'] = data['cci'].rolling(window=504).quantile(0.55)
    data['cci_趋势_0.55'] = np.where((data['cci'] > data['cci_quantile']), 1, 0)


    return data


# 过去252个交易日分别计算布林带、乖离率0.1，0.3，0.7，0.9分位数
# 布林带指标和乖离率同时低于0.1分位数时开多，任一指标回到0.3分位数时平多

def generate_position_series(data1, data2, window_size, upper_open, lower_open, upper_close, lower_close):
    position_series = []
    current_position = 0 
    for i in range(len(data)):
        
        if i<window_size:
            position_series.append(0)
            continue
        rolling_data1 = data1.iloc[i-window_size:i].rolling(window_size)
        #去掉rolling window_size
        rolling_data2 = data2.iloc[i-window_size:i].rolling(window_size)
        #去掉rolling window_size
        lower_band1 = rolling_data1.quantile(lower_open).iloc[-1]
        lower_band_close1 = rolling_data1.quantile(lower_close).iloc[-1]
        lower_band2 = rolling_data2.quantile(lower_open).iloc[-1]
        lower_band_close2 = rolling_data2.quantile(lower_close).iloc[-1]

        if current_position == 1:
            if (data1.iloc[i] >= lower_band_close1) or (data2.iloc[i] >= lower_band_close2):
                current_position = 0 

        if current_position == 0:
            if (data1.iloc[i] < lower_band1) and (data2.iloc[i] < lower_band2):
                current_position = 1  

        
        position_series.append(current_position)
    
    return pd.Series(position_series)

'''def generate_position_series(bb, dev, window_size=252, lower_open=0.1, upper_close=0.3):
    position = []
    holding = False
    for i in range(len(bb)):
        if i < window_size:
            position.append(0)
            continue
        bb_window = bb[i - window_size:i]
        dev_window = dev[i - window_size:i]
        bb_q10 = bb_window.quantile(lower_open)
        dev_q10 = dev_window.quantile(lower_open)
        bb_q30 = bb_window.quantile(upper_close)
        dev_q30 = dev_window.quantile(upper_close)

        if not holding and bb[i] < bb_q10 and dev[i] < dev_q10:
            holding = True
            position.append(1)
        elif holding and (bb[i] > bb_q30 or dev[i] > dev_q30):
            holding = False
            position.append(0)
        else:
            position.append(int(holding))
    return pd.Series(position, index=bb.index)
'''

# download data
def get_data(date):
    end_date = pd.to_datetime(date)
    start_date = end_date - pd.Timedelta(days=365*3)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    w.start()
    data = w.wsd('T.CFE', "high,open,low,close,volume,amt", start_date, end_date, "unit=1;PriceAdj=F")
    data = pd.DataFrame(data.Data, index=data.Fields, columns=data.Times).T
    data.columns = [i.lower() for i in data.columns]
    data = data.dropna()
    return data


if __name__ == '__main__':
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    # print(current_dir_path)
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    data = get_data(date)
    # get_indicator(data).to_excel(f'{current_dir_path}/test.xlsx')
    data = get_indicator(data)[['ma_signal', 'kdj_signal', 'cci_signal', 'rsi_signal', 'bb_signal', 'cci_反转_-80', 'cci_反转_-10', 'rsi_反转_-0.5', 'cci_趋势_0.55']]
    data.columns = ['双均线', 'KDJ', 'CCI', 'RSI', '布林带+乖离率', 'CCI_反转_-80', 'CCI_反转_-10', 'RSI_反转_-0.5', 'CCI_趋势_0.55']
    data.to_excel(f'{current_dir_path}/position.xlsx')
    print(data.dropna().iloc[-1])
    x = input('press enter to exit') 