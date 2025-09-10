#!/usr/bin/env python
# coding: utf-8

# In[7]:

'''
添加的都是技术指标
'''
import pandas as pd
import numpy as np
#from data_get import fetch_wind_data
def tech_con(data):
    #data = data.dropna()
    def calculate_momentum(data, period=10):
        return data['close'] - data['close'].shift(period)

    def calculate_cmo(data, period=10):
        delta = data['close'].diff()
        up = delta.apply(lambda x: max(x, 0))
        dn = delta.apply(lambda x: min(x, 0))
        sum_up = up.rolling(window=period, min_periods=1).sum()
        sum_dn = abs(dn.rolling(window=period, min_periods=1).sum())
        return 100 * (sum_up - sum_dn) / (sum_up + sum_dn)

    def calculate_apo(data, short_period=10, long_period=20):
        ma_short = data['close'].rolling(window=short_period).mean()
        ma_long = data['close'].rolling(window=long_period).mean()
        return ma_short - ma_long

    def calculate_macd(data, short_period=10, long_period=20, signal_period=9):
        def calculate_ema(data, period):
            ema = [0] * len(data)
            multiplier = 2 / (period + 1)
            for i in range(len(data)):
                if i == 0:
                    ema[i] = data[i]
                else:
                    ema[i] = ((data[i] - ema[i - 1]) * multiplier) + ema[i - 1]
            return ema

        ema_short = calculate_ema(data['close'].tolist(), short_period)
        ema_long = calculate_ema(data['close'].tolist(), long_period)
        dif = [s - l for s, l in zip(ema_short, ema_long)]
        dea = calculate_ema(dif, signal_period)
        hist = [2 * (d - e) for d, e in zip(dif, dea)]
        return hist

    def calculate_dmi(data, period=10):
        high_diff = data['high'].diff()
        low_diff = -data['low'].diff()
        plus_dm = high_diff.where((high_diff > 0) & (high_diff > low_diff), 0.0)
        minus_dm = low_diff.where((low_diff > 0) & (low_diff > high_diff), 0.0)
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift(1))
        tr3 = abs(data['low'] - data['close'].shift(1))
        tr = tr1.combine(tr2, max).combine(tr3, max)
        tr_sum = tr.rolling(window=period, min_periods=1).sum()
        plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).sum() / tr_sum)
        minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).sum() / tr_sum)
        dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period, min_periods=1).mean()
        return plus_di, minus_di, adx

    def calculate_cci(data, period=10):
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = tp.rolling(window=period, min_periods=1).mean()
        mean_deviation = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return (tp - sma_tp) / (0.015 * mean_deviation)

    def calculate_bopa(data, period=20):
        def calculate_bop(data):
            return (data['open'] - data['close']) / (data['high'] - data['low'])

        bop_values = calculate_bop(data)
        return bop_values.rolling(window=period).mean()

    def calculate_mfi(data, period=10):
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        raw_money_flow = typical_price * data['volume']
        positive_flow = []
        negative_flow = []
        for i in range(1, len(data)):
            if typical_price[i] > typical_price[i - 1]:
                positive_flow.append(raw_money_flow[i])
                negative_flow.append(0)
            elif typical_price[i] < typical_price[i - 1]:
                positive_flow.append(0)
                negative_flow.append(raw_money_flow[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        positive_flow = pd.Series(positive_flow).rolling(window=period).sum()
        negative_flow = pd.Series(negative_flow).rolling(window=period).sum()
        money_flow_ratio = positive_flow / negative_flow
        mfi_value = 100 - (100 / (1 + money_flow_ratio))
        mfi_value = pd.Series([None] + mfi_value.tolist())
        return mfi_value

    def calculate_aroon(data, period=25):
        high = data['high'].rolling(window=period, min_periods=1).max()
        low = data['low'].rolling(window=period, min_periods=1).min()
        aroon_up = 100 * ((period - (period - (data['high'].rolling(window=period).apply(lambda x: list(x).index(max(x)), raw=True)))) / period)
        aroon_down = 100 * ((period - (period - (data['low'].rolling(window=period).apply(lambda x: list(x).index(min(x)), raw=True)))) / period)
        return aroon_up, aroon_down

    def calculate_bbi(data):
        ma3 = data['close'].rolling(window=3).mean()
        ma6 = data['close'].rolling(window=6).mean()
        ma12 = data['close'].rolling(window=12).mean()
        ma24 = data['close'].rolling(window=24).mean()
        return (ma3 + ma6 + ma12 + ma24) / 4

    def calculate_sar(data, initial_af=0.02, step_af=0.02, max_af=0.2, accel_period=5):
        sar = [0] * len(data)
        af = initial_af
        uptrend = True
        ep = data['low'][0] if uptrend else data['high'][0]
        sar[0] = data['low'][0] if uptrend else data['high'][0]
        for i in range(1, len(data)):
            prev_sar = sar[i - 1]
            prev_ep = ep
            prev_af = af
            if i >= 2 * accel_period:
                cond1 = (i >= accel_period and
                     data['close'][i] > max(data['close'][i - accel_period], data['close'][i - 2 * accel_period]) and
                     data['high'][i] > data['high'][i - 1])
                cond2 = (i >= accel_period and
                     data['close'][i] < min(data['close'][i - accel_period], data['close'][i - 2 * accel_period]) and
                     data['low'][i] < data['low'][i - 1])
            else:
                cond1 = cond2 = False
            if uptrend:
                if cond1:
                    af = min(max_af, prev_af + step_af)
                else:
                    af = 0.02
            else:
                if cond2:
                    af = min(max_af, prev_af + step_af)
                else:
                    af = 0.02
            if uptrend:
                ep = max(prev_ep, data['high'][i])
            else:
                ep = min(prev_ep, data['low'][i])
            sar[i] = prev_sar + prev_af * (ep - prev_sar)
            if uptrend:
                if data['low'][i] < sar[i]:
                    uptrend = False
                    sar[i] = prev_ep
                    ep = data['low'][i]
                    af = initial_af
                else:
                    ep = max(ep, data['high'][i])
            else:
                if data['high'][i] > sar[i]:
                    uptrend = True
                    sar[i] = prev_ep
                    ep = data['high'][i]
                    af = initial_af
                else:
                    ep = min(ep, data['low'][i])
            if i >= accel_period:
                if uptrend:
                    sar[i] = min(sar[i], min(data['low'][i - accel_period:i + 1]))
                else:
                    sar[i] = max(sar[i], max(data['high'][i - accel_period:i + 1]))
        return pd.Series(sar)

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

    def calculate_rsi(data, period=10):
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_roc(data, period=10):
        n_period_ago = data['close'].shift(period)
        return (data['close'] - n_period_ago) / n_period_ago * 100

    def calculate_bias(data, period=10):
        moving_average = data['close'].rolling(window=period, min_periods=1).mean()
        return (data['close'] - moving_average) / moving_average * 100

    def calculate_osc(data, period=10):
        moving_average = data['close'].rolling(window=period, min_periods=1).mean()
        return data['close'] - moving_average

    def calculate_wr(data, period=9):
        highest_high = data['high'].rolling(window=period, min_periods=1).max()
        lowest_low = data['low'].rolling(window=period, min_periods=1).min()
        return (highest_high - data['close']) / (highest_high - lowest_low) * 100

    data['mom'] = calculate_momentum(data, period=10)
    data['cmo'] = calculate_cmo(data, period=10)
    data['apo'] = calculate_apo(data, short_period=10, long_period=20)
    data['macd'] = calculate_macd(data)
    data['plus_di'], data['minus_di'], data['adx'] = calculate_dmi(data, period=10)
    data['cci'] = calculate_cci(data, period=10)
    data['bopa'] = calculate_bopa(data, period=20)
    data['mfi'] = calculate_mfi(data, period=10)
    data['aroon_up'], data['aroon_down'] = calculate_aroon(data)
    data['aroon'] = data['aroon_up'] - data['aroon_down']
    data['bbi'] = calculate_bbi(data)
    data['sar'] = calculate_sar(data)
    data['kdj'] = calculate_kdj(data, period=9)
    data['rsi'] = calculate_rsi(data, period=10)
    data['roc'] = calculate_roc(data, period=10)
    data['bias'] = calculate_bias(data, period=10)
    data['osc'] = calculate_osc(data, period=10)
    data['wr'] = calculate_wr(data, period=9)

    # Additional indicators and calculations
    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
    data['low2high'] = data['low'] / data['high']
    data['vwap2close'] = data['vwap'] / data['close']
    data['kmid'] = (data['close'] - data['open']) / data['open']
    data['klen'] = (data['high'] - data['low']) / data['open']
    data['kmid2'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['kup'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['open']
    data['kup2'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])
    data['klow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['open']
    data['klow2'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])
    data['ksft'] = (2 * data['close'] - data['high'] - data['low']) / data['open']
    data['ksft2'] = (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low'])

    for window in [3, 5, 14, 28]:
        data[f'ma_{window}'] = data['close'].rolling(window=window, min_periods=1).mean()
        data[f'volatility_{window}'] = data['close'].rolling(window=window, min_periods=1).std()

    data['y'] = (data['close']-data['open'])/data['open']*100
    data['y_p'] = data['y'].shift(-1)
    data['R1'] = data['y']
    data['R3'] = data['y'].rolling(3).sum()
    data['R5'] = data['y'].rolling(5).sum()
    #1Y国债收益率变化
    data['1Y_return'] = data['S0059744'].pct_change() * 100
    #2Y国债收益率变化
    data['2Y_return'] = data['S0059745'].pct_change() * 100
    #5Y国债收益率变化
    data['5Y_return'] = data['S0059747'].pct_change() * 100
    #10Y国债收益率变化
    data['10Y_return'] = data["S0059749"].pct_change() * 100
    #30Y国债收益率变化
    data['30Y_return'] = data["S0059752"].pct_change() * 100
    #10Y国债活跃收益率变化
    data['10Yactive_return'] = data["W8696400"].pct_change() * 100
    #10Y/1Y
    data['10Y/1Y'] = data['S0059749']/data['S0059744']
    #5Y/1Y
    data['5Y/1Y'] = data['S0059747']/data['S0059744']
    
    del data['volume']
    del data['open']
    del data['high']
    del data['low']
    del data['close']
    #data=data.dropna()

    return data


# In[15]:


if __name__ == "__main__":
    TF, T ,Tindex= fetch_wind_data()
    #Tindex=tech_con(Tindex)
    #Tindex
    




# In[ ]:




