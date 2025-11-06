import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import time
from datetime import datetime
today = datetime.now().strftime("%Y-%m-%d")
from WindPy import w
import requests
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
w.start()
# df = w.wsd("AU.SHF", "close", "2010-01-01", "2025-06-22", "PriceAdj=F")
# df = pd.DataFrame(df.Data, columns=df.Times, index=["close"]).T
# df
# Windows系统下的中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def backtest_strategy(close, signals, start_date, end_date):
    """
    Backtests the strategy using the provided signals.
    """
    # Filter the data based on the specified date range
    close = close.loc[start_date:end_date].copy()
    signals = signals.loc[start_date:end_date].copy()
    signals = signals.reindex(close.index).ffill()

    # Calculate the daily returns of the close price
    daily_returns = close.diff().fillna(0)

    # Calculate the strategy returns based on the signals
    strategy_returns = daily_returns * signals.shift(1).fillna(0)

    # Calculate the cumulative returns of the strategy
    cumulative_returns = (strategy_returns).cumsum() 

    return cumulative_returns

def cal_performance_metrics(cumulative_returns):
    """
    Calculates the performance metrics of the strategy.
    """
    # Calculate the annualized return
    annualized_return = (cumulative_returns.iloc[-1]) / (len(cumulative_returns))*252

    # Calculate the annualized volatility
    annualized_volatility = cumulative_returns.diff().std() * np.sqrt(252)

    # Calculate the Sharpe ratio
    sharpe_ratio = annualized_return / annualized_volatility

    # Calculate the maximum drawdown
    cumulative_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cumulative_max) 
    max_drawdown = drawdown.min()

    return annualized_return, annualized_volatility, sharpe_ratio, max_drawdown

def plot_pnl_curve(cumulative_returns, signals, verbose=True):
    """
    Plots the PnL curve of the strategy.
    add buy and sell signals to the plot to show the strategy
    """
    # Plot the PnL curve
    plt.figure(figsize=(10, 6))
    signals = signals.reindex(cumulative_returns.index).copy()
    plt.plot(cumulative_returns, label='Cumulative Returns')

    # Add buy and sell signals to the plot to show the strategy
    # buy_signals = signals[(signals == 1)&(signals.shift(1).fillna(0)!= 1)]
    # sell_signals = signals[(signals == -1)&(signals.shift(1).fillna(0)!= -1)]
    # plt.scatter(buy_signals.index, cumulative_returns[buy_signals.index], marker='^', color='r', label='Buy')
    # plt.scatter(sell_signals.index, cumulative_returns[sell_signals.index], marker='v', color='g', label='Sell')

    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('PnL Curve')
    plt.legend()
    plt.grid(True)
    if verbose ==  False:
        plt.close()

    # return figure
    # Convert the plot to a figure object
    return plt # Return the figure for further processing or saving if needed.

def backtest_process(close, signals, start_date, end_date, verbose=True):
    """
    Backtests the strategy using the provided signals.
    return pnl, performance metrics, and plot the pnl curve
    """
    # Backtest the strategy using the provided signals
    cumulative_returns = backtest_strategy(close, signals, start_date, end_date)

    # Calculate the performance metrics of the strategy
    annualized_return, annualized_volatility, sharpe_ratio, max_drawdown = cal_performance_metrics(cumulative_returns)
    sortino_ratio = annualized_return / max_drawdown
    performance_metrics = {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Sortino Ratio': sortino_ratio
    }
    # Plot the PnL curve of the strategy
    plot = plot_pnl_curve(cumulative_returns, signals, verbose)


    return cumulative_returns, performance_metrics, plot

def compare_strategy_vs_benchmark(close, strategy_signals, start_date='2025-01-01', end_date=None):
    """
    比较策略和基准从2022年至今的表现
    基准使用long only仓位全都设置为1
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 创建基准信号（long only，仓位全为1）
    benchmark_signals = pd.Series(1, index=close.index)
    
    # 确保信号和收盘价对齐
    strategy_signals = strategy_signals.reindex(close.index).fillna(0)
    benchmark_signals = benchmark_signals.reindex(close.index).fillna(1)
    
    # 过滤指定日期范围
    close_filtered = close.loc[start_date:end_date].copy()
    strategy_signals_filtered = strategy_signals.loc[start_date:end_date].copy()
    benchmark_signals_filtered = benchmark_signals.loc[start_date:end_date].copy()
    
    # 计算策略表现
    strategy_cum_returns, strategy_metrics, strategy_plot = backtest_process(
        close_filtered, strategy_signals_filtered, start_date, end_date, verbose=False
    )
    
    # 计算基准表现
    benchmark_cum_returns, benchmark_metrics, benchmark_plot = backtest_process(
        close_filtered, benchmark_signals_filtered, start_date, end_date, verbose=False
    )
    
    # 创建对比图表
    plt.figure(figsize=(15, 10))
    
    # 子图1：累计收益对比
    plt.subplot(2, 2, 1)
    plt.plot(strategy_cum_returns.index, strategy_cum_returns, label='策略收益', color='blue', linewidth=2)
    plt.plot(benchmark_cum_returns.index, benchmark_cum_returns, label='基准收益(long only)', color='red', linewidth=2)
    plt.title('策略 vs 基准累计收益对比 (2022年至今)', fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('累计收益')
    plt.legend()
    plt.grid(True)
    
    # 子图2：相对表现
    relative_performance = strategy_cum_returns - benchmark_cum_returns
    plt.subplot(2, 2, 2)
    plt.plot(relative_performance.index, relative_performance, color='green', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('策略相对基准的超额收益', fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('超额收益')
    plt.grid(True)
    
    # 子图3：回撤对比
    plt.subplot(2, 2, 3)
    strategy_drawdown = (strategy_cum_returns - strategy_cum_returns.cummax())
    benchmark_drawdown = (benchmark_cum_returns - benchmark_cum_returns.cummax())
    plt.plot(strategy_drawdown.index, strategy_drawdown, label='策略回撤', color='blue', alpha=0.7)
    plt.plot(benchmark_drawdown.index, benchmark_drawdown, label='基准回撤', color='red', alpha=0.7)
    plt.title('回撤对比', fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('回撤')
    plt.legend()
    plt.grid(True)
    
    # 子图4：月度收益对比
    strategy_monthly = strategy_cum_returns.resample('M').last().pct_change().dropna()
    benchmark_monthly = benchmark_cum_returns.resample('M').last().pct_change().dropna()
    plt.subplot(2, 2, 4)
    width = 0.35
    x = range(len(strategy_monthly))
    plt.bar([i - width/2 for i in x], strategy_monthly.values, width, label='策略月度收益', color='blue', alpha=0.7)
    plt.bar([i + width/2 for i in x], benchmark_monthly.values, width, label='基准月度收益', color='red', alpha=0.7)
    plt.title('月度收益对比', fontsize=14, fontweight='bold')
    plt.xlabel('月份')
    plt.ylabel('月度收益')
    plt.legend()
    plt.xticks(x, [d.strftime('%Y-%m') for d in strategy_monthly.index], rotation=45, ha='right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 创建详细对比表格
    comparison_df = pd.DataFrame({
        '指标': ['年化收益率', '年化波动率', '夏普比率', '最大回撤', '索提诺比率', '累计收益'],
        '策略': [
            strategy_metrics['Annualized Return'],
            strategy_metrics['Annualized Volatility'],
            strategy_metrics['Sharpe Ratio'],
            strategy_metrics['Max Drawdown'],
            strategy_metrics['Sortino Ratio'],
            strategy_cum_returns.iloc[-1]
        ],
        '基准(long only)': [
            benchmark_metrics['Annualized Return'],
            benchmark_metrics['Annualized Volatility'],
            benchmark_metrics['Sharpe Ratio'],
            benchmark_metrics['Max Drawdown'],
            benchmark_metrics['Sortino Ratio'],
            benchmark_cum_returns.iloc[-1]
        ],
        '差值(策略-基准)': [
            strategy_metrics['Annualized Return'] - benchmark_metrics['Annualized Return'],
            strategy_metrics['Annualized Volatility'] - benchmark_metrics['Annualized Volatility'],
            strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio'],
            strategy_metrics['Max Drawdown'] - benchmark_metrics['Max Drawdown'],
            strategy_metrics['Sortino Ratio'] - benchmark_metrics['Sortino Ratio'],
            strategy_cum_returns.iloc[-1] - benchmark_cum_returns.iloc[-1]
        ]
    })
    
    # 计算胜率
    strategy_daily = strategy_cum_returns.diff().dropna()
    benchmark_daily = benchmark_cum_returns.diff().dropna()
    win_rate = (strategy_daily > benchmark_daily).mean()
    
    # 计算年化超额收益
    annual_excess_return = strategy_metrics['Annualized Return'] - benchmark_metrics['Annualized Return']
    
    print("=== 策略 vs 基准表现对比 (2022年至今) ===")
    print(comparison_df.to_string(index=False))
    print(f"\n胜率（策略日收益 > 基准日收益）：{win_rate:.2%}")
    print(f"年化超额收益：{annual_excess_return:.2%}")
    
    return {
        'strategy_cum_returns': strategy_cum_returns,
        'benchmark_cum_returns': benchmark_cum_returns,
        'strategy_metrics': strategy_metrics,
        'benchmark_metrics': benchmark_metrics,
        'comparison_df': comparison_df,
        'win_rate': win_rate,
        'annual_excess_return': annual_excess_return,
        'relative_performance': relative_performance
    }

# 使用示例：
# 假设您已经有了策略信号signal和close_df['close']
# comparison_result = compare_strategy_vs_benchmark(close_df['close'], signal)

if __name__ == '__main__':


    renamed_indicators = [
        # 期货部分
        "fut_total",
        "fut_pos_prod_long", "fut_pos_prod_short",
        "fut_pos_sd_long", "fut_pos_sd_short", "fut_pos_sd_spread",
        "fut_pos_mm_long", "fut_pos_mm_short", "fut_pos_mm_spread",
        "fut_pos_or_long", "fut_pos_or_short", "fut_pos_or_spread",
        "fut_pos_nr_long", "fut_pos_nr_short",
        
        # 期货环比
        "fut_total_wow",
        "fut_pos_prod_long_wow", "fut_pos_prod_short_wow",
        "fut_pos_sd_long_wow", "fut_pos_sd_short_wow", "fut_pos_sd_spread_wow",
        "fut_pos_mm_long_wow", "fut_pos_mm_short_wow", "fut_pos_mm_spread_wow",
        "fut_pos_or_long_wow", "fut_pos_or_short_wow", "fut_pos_or_spread_wow",
        "fut_pos_nr_long_wow", "fut_pos_nr_short_wow",
        
        # 期货占比
        "fut_pct_prod_long", "fut_pct_prod_short",
        "fut_pct_sd_long", "fut_pct_sd_short", "fut_pct_sd_spread",
        "fut_pct_mm_long", "fut_pct_mm_short", "fut_pct_mm_spread",
        "fut_pct_or_long", "fut_pct_or_short", "fut_pct_or_spread",
        "fut_pct_nr_long", "fut_pct_nr_short",
        
        # 期货交易商数量
        "fut_traders",
        "fut_traders_prod_long", "fut_traders_prod_short",
        "fut_traders_sd_long", "fut_traders_sd_short", "fut_traders_sd_spread",
        "fut_traders_mm_long", "fut_traders_mm_short", "fut_traders_mm_spread",
        "fut_traders_or_long", "fut_traders_or_short", "fut_traders_or_spread",
        
        # 期货和期权部分
        "f_o_total",
        "f_o_pos_prod_long", "f_o_pos_prod_short",
        "f_o_pos_sd_long", "f_o_pos_sd_short", "f_o_pos_sd_spread",
        "f_o_pos_mm_long", "f_o_pos_mm_short", "f_o_pos_mm_spread",
        "f_o_pos_or_long", "f_o_pos_or_short", "f_o_pos_or_spread",
        "f_o_pos_nr_long", "f_o_pos_nr_short",
        
        # 期货和期权环比
        "f_o_total_wow",
        "f_o_pos_prod_long_wow", "f_o_pos_prod_short_wow",
        "f_o_pos_sd_long_wow", "f_o_pos_sd_short_wow", "f_o_pos_sd_spread_wow",
        "f_o_pos_mm_long_wow", "f_o_pos_mm_short_wow", "f_o_pos_mm_spread_wow",
        "f_o_pos_or_long_wow", "f_o_pos_or_short_wow", "f_o_pos_or_spread_wow",
        "f_o_pos_nr_long_wow", "f_o_pos_nr_short_wow",
        
        # 期货和期权占比
        "f_o_pct_prod_long", "f_o_pct_prod_short",
        "f_o_pct_sd_long", "f_o_pct_sd_short", "f_o_pct_sd_spread",
        "f_o_pct_mm_long", "f_o_pct_mm_short", "f_o_pct_mm_spread",
        "f_o_pct_or_long", "f_o_pct_or_short", "f_o_pct_or_spread",
        "f_o_pct_nr_long", "f_o_pct_nr_short",
        
        # 期货和期权交易商数量
        "f_o_traders",
        "f_o_traders_prod_long", "f_o_traders_prod_short",
        "f_o_traders_sd_long", "f_o_traders_sd_short", "f_o_traders_sd_spread",
        "f_o_traders_mm_long", "f_o_traders_mm_short", "f_o_traders_mm_spread",
        "f_o_traders_or_long", "f_o_traders_or_short", "f_o_traders_or_spread"
    ]

    # 加载cftc因子
    cftc_list = ['S0240428', 'S0240429', 'S0240430', 'S0240431', 'S0240432', 'S0240433', 
    'S0240434', 'S0240435', 'S0240436', 'S0240437', 'S0240438', 'S0240439', 'S0240440', 
    'S0240441', 'S0240442', 'S0240443', 'S0240444', 'S0240445', 'S0240446', 'S0240447', 
    'S0240448', 'S0240449', 'S0240450', 'S0240451', 'S0240452', 'S0240453', 'S0240454', 
    'S0240455', 'S0240456', 'S0240457', 'S0240458', 'S0240459', 'S0240460', 'S0240461', 
    'S0240462', 'S0240463', 'S0240464', 'S0240465', 'S0240466', 'S0240467', 'S0240468', 
    'S0240469', 'S0240470', 'S0240471', 'S0240472', 'S0240473', 'S0240474', 'S0240475', 
    'S0240476', 'S0240477', 'S0240478', 'S0240479', 'S0240480', 'S0241912', 'S0241913', 
    'S0241914', 'S0241915', 'S0241916', 'S0241917', 'S0241918', 'S0241919', 'S0241920', 
    'S0241921', 'S0241922', 'S0241923', 'S0241924', 'S0241925', 'S0241926', 'S0241927', 
    'S0241928', 'S0241929', 'S0241930', 'S0241931', 'S0241932', 'S0241933', 'S0241934', 
    'S0241935', 'S0241936', 'S0241937', 'S0241938', 'S0241939', 'S0241940', 'S0241941', 
    'S0241942', 'S0241943', 'S0241944', 'S0241945', 'S0241946', 'S0241947', 'S0241948', 
    'S0241949', 'S0241950', 'S0241951', 'S0241952', 'S0241953', 'S0241954', 'S0241955', 
    'S0241956', 'S0241957', 'S0241958', 'S0241959', 'S0241960', 'S0241961', 'S0241962', 
    'S0241963', 'S0241964']

    cftc_df = w.edb(",".join(cftc_list), "2006-01-01", today,"Fill=Previous")
    cftc_df = pd.DataFrame(cftc_df.Data, columns=cftc_df.Times, index=cftc_list).T
    cftc_df.index = pd.to_datetime(cftc_df.index)
    cftc_df = cftc_df.loc['2006-01-01':]
    close_df = w.wsd("AU.SHF", "close", "2010-01-01", today, "PriceAdj=F")
    close_df = pd.DataFrame(close_df.Data, columns=close_df.Times, index=["close"]).T
    close_df.index = pd.to_datetime(close_df.index)
    # 将cftc 的 index改为下一个周一
    # 计算每个日期对应的下周一
    cftc_df.index = pd.DatetimeIndex(cftc_df.index)
    cftc_df.index = cftc_df.index + pd.offsets.Week(weekday=0)
    bd_list = pd.date_range(start='2006-01-01', end='2026-06-25', freq='B')
    cftc_df.columns = renamed_indicators
    wow_list = [i for i in cftc_df.columns if 'wow' in i]


    cftc_df = cftc_df.resample('D').ffill().reindex(index=close_df.index).ffill().dropna(how='all')
    cftc_df[wow_list] = cftc_df[wow_list].rolling(20).sum()
    # net position
    cftc_df['fut_pos_sd_net'] = cftc_df['fut_pos_sd_long'] - cftc_df['fut_pos_sd_short']
    cftc_df['fut_pos_mm_net'] = cftc_df['fut_pos_mm_long'] - cftc_df['fut_pos_mm_short']
    cftc_df['fut_pos_prod_net'] = cftc_df['fut_pos_prod_long'] - cftc_df['fut_pos_prod_short']
    cftc_df['f_o_pos_prod_net'] = cftc_df['f_o_pos_prod_long'] - cftc_df['f_o_pos_prod_short']
    cftc_df['f_o_pos_mm_net'] = cftc_df['f_o_pos_mm_long'] - cftc_df['f_o_pos_mm_short']
    cftc_df['f_o_pos_sd_net'] = cftc_df['f_o_pos_sd_long'] - cftc_df['f_o_pos_sd_short']
    cftc_df['fut_pos_sd_net_wow'] = cftc_df['fut_pos_sd_long_wow'] - cftc_df['fut_pos_sd_short_wow']
    cftc_df['fut_pos_mm_net_wow'] = cftc_df['fut_pos_mm_long_wow'] - cftc_df['fut_pos_mm_short_wow']
    cftc_df['fut_pos_prod_net_wow'] = cftc_df['fut_pos_prod_long_wow'] - cftc_df['fut_pos_prod_short_wow']
    cftc_df['f_o_pos_sd_net_wow'] = cftc_df['f_o_pos_sd_long_wow'] - cftc_df['f_o_pos_sd_short_wow']
    cftc_df['f_o_pos_mm_net_wow'] = cftc_df['f_o_pos_mm_long_wow'] - cftc_df['f_o_pos_mm_short_wow']
    cftc_df['f_o_pos_prod_net_wow'] = cftc_df['f_o_pos_prod_long_wow'] - cftc_df['f_o_pos_prod_short_wow']


    # column_list = [i for i in cftc_df.columns if 'or' not in i and 'nr' not in i]
    column_list = [i for i in cftc_df.columns if 'nr' not in i]

    cftc_df = cftc_df[column_list]

    def quantile_strategy(data, q, window):
        # 计算每个窗口的分位数
        quantiles = data.rolling(window=window).quantile(1-q)
        quantiles2 = data.rolling(window=window).quantile(q)

        # 大于分位数为1，小于1-q分位数为-1，否则为0
        # signals = np.where(data > quantiles, 1, np.where(data < quantiles, -1, 0))
        signals = np.where(data > quantiles, 1, 0)
        signals = pd.Series(signals, index=data.index)
        return signals


    start_date = '20200101'
    end_date = '20241231'
    q_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    window_list= [252,252*2,252*3,252*4]
    direction = ['正向', '反向']
    performance_matrix = []
    for indi_ in cftc_df.columns:
        temp = cftc_df[indi_]
        for q in q_list:
            for window in window_list:
                for direction_ in direction:
                    if direction_ == '正向':
                        s1 = quantile_strategy(temp, q, window)
                    else:
                        s1 = quantile_strategy(-1*temp, q, window)
                    pnl, p_matrix, plot = backtest_process(close_df['close'], s1, start_date, end_date, verbose=False)
                    p_matrix['q'] = q
                    p_matrix['window'] = window
                    p_matrix['indi'] = indi_
                    p_matrix['signal'] = s1
                    p_matrix['direction'] = direction_
                    performance_matrix.append(p_matrix)
    benchmark_signal = np.ones(len(close_df['close']))
    benchmark_signal = pd.Series(benchmark_signal, index=close_df['close'].index)
    pnl, p_matrix, plot = backtest_process(close_df['close'], benchmark_signal, start_date, end_date, verbose=False)
    p_matrix['q'] = 'benchmark'
    p_matrix['window'] = 'benchmark'
    p_matrix['indi'] = 'benchmark'
    p_matrix['signal'] = benchmark_signal
    performance_matrix.append(p_matrix)
    # 两列信号，分仓，降息，白银，美元等宏观，CPI，汇率
    import pickle
    performance_matrix = pd.DataFrame(performance_matrix)
    # performance_matrix.to_pickle('performance_matrix.pkl')
    import pickle
    # performance_matrix = pd.read_pickle('performance_matrix.pkl')
    close_df
    performance_matrix.sort_values('Sharpe Ratio',ascending=False).head(10)
    # erformance_matrix[performance_matrix['Sharpe Ratio']>benchmark_sharpe_ratio].sort_values(by='Sharpe Ratio',ascending=False)
    benchmark_sharpe_ratio = performance_matrix[performance_matrix['indi'] == 'benchmark']['Sharpe Ratio'].values[0]
    factor_performance_matrix = performance_matrix[performance_matrix['Sharpe Ratio']>benchmark_sharpe_ratio].sort_values(by='Sharpe Ratio',ascending=False)

    factor_performance_matrix['count'] = factor_performance_matrix.groupby(['indi'])['q'].transform('count').values
    factor_performance_matrix = factor_performance_matrix[factor_performance_matrix['count']>=10]
    factor_performance_matrix['institute'] = np.where(factor_performance_matrix['indi'].str.contains('sd'), 'sd', 'or')
    factor_performance_matrix['institute'] = np.where(factor_performance_matrix['indi'].str.contains('mm'), 'mm', factor_performance_matrix['institute'])
    factor_performance_matrix['institute'] = np.where(factor_performance_matrix['indi'].str.contains('prod'), 'prod', factor_performance_matrix['institute'])
    factor_performance_matrix['institute'] = np.where(factor_performance_matrix['indi'].str.contains('total'), 'total', factor_performance_matrix['institute'])
    factor_performance_matrix = factor_performance_matrix[~(factor_performance_matrix['institute'].isin(['total', 'or']))]

    # long short net or spread
    factor_performance_matrix['ls'] = np.where(factor_performance_matrix['indi'].str.contains('long'),'long', 'nan')
    factor_performance_matrix['ls'] = np.where(factor_performance_matrix['indi'].str.contains('short'),'short', factor_performance_matrix['ls'])
    factor_performance_matrix['ls'] = np.where(factor_performance_matrix['indi'].str.contains('spread'),'spread', factor_performance_matrix['ls'])
    factor_performance_matrix['ls'] = np.where(factor_performance_matrix['indi'].str.contains('net'),'net', factor_performance_matrix['ls'])
    factor_performance_matrix = factor_performance_matrix[factor_performance_matrix['ls'] != 'spread']
    factor_performance_matrix['wow'] = np.where(factor_performance_matrix['indi'].str.contains('wow'), 'wow', 'pos')


    factor_performance_matrix = factor_performance_matrix.groupby(['institute', 'ls', 'wow']).first().reset_index().sort_values(by='Sharpe Ratio',ascending=False)
    temp = factor_performance_matrix.copy()
    temp = []
    while len(factor_performance_matrix)>0:
        temp.append(factor_performance_matrix.iloc[0])
        factor_performance_matrix['corr'] = factor_performance_matrix['signal'].apply(lambda x: np.corrcoef(x.loc[start_date:end_date],factor_performance_matrix.iloc[0]['signal'].loc[start_date:end_date])[0,1])
        factor_performance_matrix = factor_performance_matrix[factor_performance_matrix['corr']<0.3]
    temp = pd.concat(temp,axis=1).T
    # temp = temp.head(10)
    temp.to_excel(f'temp_{start_date}_{end_date}.xlsx',index=False)
    signal = temp['signal'].sum()/len(temp['signal'])
    signal = pd.Series(np.where(signal.values>=2/3,1,np.where(signal.values>=1/3, 0.5, 0)), index=signal.index).reindex(signal.index).fillna(0)
    # signal = 
    # signal = temp.iloc[-4]['signal']
    backtest_process(close_df['close'], signal, end_date, '20250630')
    signal.to_excel(f'signal_{start_date}_{end_date}.xlsx')

    t = signal.index[-1]

    #########################邮件发送##############################
    # content = t.strftime('%Y%m%d')+"黄金cftc持仓最新信号为："+str(signal.iloc[-1])+', 前一日信号为'+str(signal.iloc[-2])+'； <br/> '


    # data = {"to": ["zhuyufan_fi@chinastock.com.cn"], "subject": "黄金cftc持仓"+t.strftime('%Y%m%d'), "body": content}
    # headers = {'Content-Type':'application/json',"X-API-Key": 'fjtgyxa'}
    # response = requests.post('http://10.4.111.86:8080/send-email', json=data, headers = headers)

        # 获取最近一个月的信号数据
    # import requests
    # from datetime import datetime, timedelta
    
    # # 获取最近一个月的数据
    # end_date = signal.index[-1]
    # start_date = end_date - timedelta(days=30)
    # recent_signal = signal.loc[start_date:end_date].copy()
    
    # # 按日期倒序排列
    # recent_signal = recent_signal.sort_index(ascending=False)
    
    # # 创建HTML表格
    # html_table = '<table border="1" style="border-collapse: collapse; width: 100%;">'
    # html_table += '<thead><tr><th style="padding: 8px; background-color: #f2f2f2;">日期</th><th style="padding: 8px; background-color: #f2f2f2;">信号值</th></tr></thead>'
    # html_table += '<tbody>'
    
    # for date, value in recent_signal.items():        
    #     html_table += f'<tr><td style="padding: 8px; text-align: center;">{date.strftime("%Y-%m-%d")}</td><td style="padding: 8px; text-align: center;">{value}</td></tr>'
    
    # html_table += '</tbody></table>'
    
    # # 创建邮件内容
    # current_date = signal.index[-1].strftime('%Y%m%d')
    # latest_signal = signal.iloc[-1]
    # prev_signal = signal.iloc[-2]
    
    # # 信号描述
    # latest_desc = {
    #     0: "空仓(0)",
    #     0.5: "半仓(0.5)",
    #     1: "满仓(1)"
    # }.get(latest_signal, str(latest_signal))
    
    # prev_desc = {
    #     0: "空仓(0)",
    #     0.5: "半仓(0.5)",
    #     1: "满仓(1)"
    # }.get(prev_signal, str(prev_signal))
    
    # content = f"""
    # <h2>黄金CFTC持仓信号报告 - {current_date}</h2>
    # <p><strong>最新信号：</strong>{latest_desc}</p>
    # <p><strong>前一日信号：</strong>{prev_desc}</p>
    
    # <h3>最近30天信号明细（按日期倒序排列）：</h3>
    # {html_table}
    # """
    
    # # 发送邮件
    # data = {
    #     "to": ["zhuyufan_fi@chinastock.com.cn", "renhongbo_zb@chinastock.com.cn", "pengpeng_fi@chinastock.com.cn"], 
    #     "subject": f"黄金CFTC持仓信号-{current_date}", 
    #     "body": content,
    #     "html": True  # 使用HTML格式
    # }
    
    # headers = {'Content-Type': 'application/json', "X-API-Key": 'fjtgyxa'}
    
    # try:
    #     # response = requests.post('http://10.4.111.86:8080/send-email', json=data, headers=headers)
    #     if response.status_code == 200:
    #         print("邮件发送成功！")
    #     else:
    #         print(f"邮件发送失败，状态码：{response.status_code}")
    # except Exception as e:
        # print(f"发送邮件时出错：{str(e)}")