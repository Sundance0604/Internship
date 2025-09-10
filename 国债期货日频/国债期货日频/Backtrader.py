import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

class StrategyBacktester:
    """
    A class to backtest a trading strategy based on historical market data.

    This class allows the user to define trading signals, apply them to historical data, 
    and calculate the resulting strategy's performance. It also supports slippage handling 
    and offers methods to visualize the results and evaluate performance.

    Attributes:
        df (pd.DataFrame): The market data to be used for backtesting, filtered by the specified date range.
        signal_column (str): The column name containing the trading signals.
        price_column (str): The column name containing the price data (default: 'close').
        initial_capital (float): The starting capital for the backtest (default: 1e9).
        rf (float): The risk-free rate used for Sharpe ratio calculation (default: 0.0).
        slippage (float): The slippage rate applied to the price (default: 0.0).
        result (pd.DataFrame): The result of the backtest including daily profits, cumulative returns, etc.
    
    Methods:
        __init__(data, start_date, end_date, signal_column, price_column, initial_capital, rf, slippage):
            Initializes the backtester with market data, trading signals, slippage, and other configuration settings.
        
        run_backtest():
            Executes the backtest by simulating trades based on the trading signals.
        
        plot_price_with_signals():
            Plots the market price along with the entry/exit signals and market regime background.
        
        plot_pnl_curve():
            Plots the cumulative return, drawdown, and other performance metrics.
        
        evaluate_performance():
            Evaluates the backtest performance by calculating key metrics such as annual return, Sharpe ratio, 
            maximum drawdown, profit factor, etc., and prints the results.
    """

    
    def __init__(self, data, start_year=None, end_year=None, signal_column='combo_signal',
                 price_column='close', initial_capital=1e9, rf=0., slippage=0., daily_stop_loss_rate=None,weekly_stop_loss=None):
        self.df = data.copy()[(data.date.dt.year >= start_year) & (data.date.dt.year <= end_year)].reset_index(drop=True)
        self.signal_column = signal_column
        self.price_column = price_column
        self.initial_capital = initial_capital
        self.rf = rf
        self.slippage = slippage
        self.daily_stop_loss_rate = daily_stop_loss_rate  # 新增参数
        self.weekly_stop_loss = weekly_stop_loss  # 新增参数
        self.result = None

    def run_backtest(self):
        df = self.df.copy()
        price_col = self.price_column
        signal_col = self.signal_column

        #-----------------------BenchMark---------------------------
        df['daily_profit'] = 0.0
        if price_col == 'close':
            for i in range(1, len(df)):
                today_price = df[price_col].iloc[i]
                yesterday_price = df['open'].iloc[i]
                buy_price = yesterday_price * (1 + self.slippage)
                sell_price = today_price * (1 - self.slippage)
                price_diff = sell_price - buy_price
                holdings = self.initial_capital // buy_price
                df.at[i, 'daily_profit'] = holdings * price_diff
        elif price_col == 'open':
            for i in range(2, len(df)):
                today_price = df[price_col].iloc[i]
                yesterday_price = df['open'].iloc[i-1]
                buy_price = yesterday_price * (1 + self.slippage)
                sell_price = today_price * (1 - self.slippage)
                price_diff = sell_price - buy_price
                holdings = self.initial_capital // buy_price
                df.at[i, 'daily_profit'] = holdings * price_diff

        df['return_bah'] = df['daily_profit'].cumsum() / self.initial_capital + 1

        #-----------------------Strategy---------------------------
        df['position'] = df[signal_col]
        df['daily_profit_s'] = 0.0
        # 初始化累计止损追踪变量
        rolling_loss = [0]*5  
        # 初始为0的5日盈亏记录
        stop_triggered = False

        if price_col == 'close':
            for i in range(1, len(df)):
                today_price = df[price_col].iloc[i]
                yesterday_price = df['open'].iloc[i]
                buy_price = yesterday_price * (1 + self.slippage)
                sell_price = today_price * (1 - self.slippage)
                price_diff = sell_price - buy_price
                holdings = self.initial_capital // buy_price
                daily_return = sell_price / buy_price - 1

                if self.daily_stop_loss_rate is not None and (df['low'].iloc[i]/yesterday_price-1) <= -abs(self.daily_stop_loss_rate):
                    pnl = holdings * yesterday_price * (-abs(self.daily_stop_loss_rate)) * df.at[i-1,'position'] 
                else:
                    pnl = holdings * price_diff * df.at[i-1,'position']
                # 更新 daily_profit_s
                df.at[i, 'daily_profit_s'] = pnl
                if self.weekly_stop_loss!=None:
                    # --- 累计止损判断 ---
                    rolling_loss.append(pnl / holdings)  # 以每手盈亏为单位
                    if len(rolling_loss) > 5:
                        rolling_loss.pop(0)

                    # 判断过去5天是否累计亏损超过0.5元
                    if sum(rolling_loss) <= -abs(self.weekly_stop_loss):
                        stop_triggered = True
                    else:
                        stop_triggered = False

                    # 如果触发累计止损，强制清仓
                    if stop_triggered:
                        df.at[i, 'position'] = 0  # 当天仓位强制为 0

        elif price_col == 'open':
            for i in range(2, len(df)):
                today_price = df[price_col].iloc[i]
                yesterday_price = df['open'].iloc[i-1]
                buy_price = yesterday_price * (1 + self.slippage)
                sell_price = today_price * (1 - self.slippage)
                price_diff = sell_price - buy_price
                holdings = self.initial_capital // buy_price
                daily_return = sell_price / buy_price - 1

                # 获取持仓
                position = df.at[i-2, 'position']

                # 计算是否触发**日内止损**
                if self.daily_stop_loss_rate is not None and (df['low'].iloc[i-1]/yesterday_price - 1) <= -abs(self.daily_stop_loss_rate):
                    pnl = holdings * yesterday_price * (-abs(self.daily_stop_loss_rate)) * position
                else:
                    pnl = holdings * price_diff * position

                # 更新 daily_profit_s
                df.at[i, 'daily_profit_s'] = pnl

                if self.weekly_stop_loss!=None:
                    # --- 累计止损判断 ---
                    rolling_loss.append(pnl / holdings)  # 以每手盈亏为单位
                    if len(rolling_loss) > 5:
                        rolling_loss.pop(0)

                    # 判断过去5天是否累计亏损超过0.5元
                    if sum(rolling_loss) <= -abs(self.weekly_stop_loss):
                        stop_triggered = True
                    else:
                        stop_triggered = False

                    # 如果触发累计止损，强制清仓
                    if stop_triggered:
                        df.at[i, 'position'] = 0  # 当天仓位强制为 0

        df['return_strategy'] = df['daily_profit_s'].cumsum() / self.initial_capital + 1
        self.result = df

        return df


    def plot_price_with_signals(self):
        df = self.result.reset_index(drop=True)
        open_locs = df.index[(df[self.signal_column] > 0)]
        close_locs = df.index[(df[self.signal_column] < 0)]

        plt.figure(figsize=(14, 4))
        plt.plot(df.date, df['open'], label='Open Price', color='black', linewidth=2)
        plt.scatter(df.date.loc[open_locs], df.loc[open_locs, 'open'], color='C3', marker='^', label='Long', zorder=3, alpha=0.7)
        plt.scatter(df.date.loc[close_locs], df.loc[close_locs, 'open'], color='C2', marker='v', label='Short', zorder=3, alpha=0.7)

        # ====== 背景绘制市场风格区域 ======
        if 'market_state_v2' in df.columns:
            colors = {
                'bull_volatile': '#e0f7fa',
                'bull_stable': '#e8f5e9',
                'bear_volatile': '#ffebee',
                'bear_stable': '#f3e5f5',
                'unknown': '#eeeeee'
            }

            last_state = None
            start_idx = 0

            for i in range(len(df)):
                current_state = df.loc[i, 'market_state_v2']
                if current_state != last_state:
                    if last_state is not None:
                        start_date = df.loc[start_idx, 'date']
                        end_date = df.loc[i, 'date']
                        plt.axvspan(start_date, end_date, color=colors.get(last_state, '#eeeeee'), alpha=1)
                    start_idx = i
                    last_state = current_state

            # 为最后一段也加上颜色
            if last_state is not None:
                plt.axvspan(df.loc[start_idx, 'date'], df.loc[len(df)-1, 'date'], color=colors.get(last_state, '#eeeeee'), alpha=1)

            # 图例：添加 market state 色块说明
            patches = [Patch(color=col, alpha=1, label=label) for label, col in colors.items()]
            plt.legend(handles=patches + plt.gca().get_legend_handles_labels()[0])

        # ====== 其他图形设置 ======
        plt.title('Price with Entry/Exit Signals and Market Regimes', weight='bold', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('open'.capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_pnl_curve(self):
        df = self.result.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 计算买入持有累计收益
        bah_cum = df['return_bah']
        #bah_cum = (df[self.price_column]/df[self.price_column].shift(1)).fillna(1).cumprod()
        strategy_cum = df['return_strategy']
        
        # 计算回撤
        peak_bah = bah_cum.cummax()
        drawdown_bah = -(peak_bah - bah_cum) / peak_bah * 100

        peak = strategy_cum.cummax()
        drawdown_pct = -(peak - strategy_cum) / peak * 100  # 转换为百分比
        
        #-------------策略收益图--------------
        # 创建图表和轴
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax2 = ax1.twinx()  # 共享x轴的右侧轴
        
        # 绘制策略、买入持有和超额收益
        ax1.plot(bah_cum, label='Benchmark', color='C7', linestyle='--', alpha=0.7)
        ax1.plot(strategy_cum, label='Strategy', color='C3', lw=2)
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 绘制回撤区域（右侧轴）
        ax2.fill_between(df.index, drawdown_pct, 0, color='C2', alpha=0.3, label='Drawdown')
        ax2.fill_between(df.index, drawdown_bah, 0, color='C7', alpha=0.2, label='Drawdown Benchmark')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend(loc='upper left', bbox_to_anchor=(0.01, 0.85))
        ax2.set_ylim(drawdown_pct.min() * 1.1, 0)  # 设置回撤轴范围

        # 设置标题和布局
        plt.title('Strategy Performance: Cumulative Return, Drawdown', weight='bold', fontsize=14, pad=20)
        fig.tight_layout()
        plt.show()

        #------------每日收益-------------
        fig2 = plt.figure(figsize=(14,3))

        # 计算每日收益
        excess_return = df['return_strategy'] - df['return_bah']

        plt.plot(df.index,excess_return, label='Excess Return', color='C3', alpha=0.8)
        plt.title('Excess Return', weight='bold', fontsize=14, pad=20)
        plt.xlabel('Date') 
        plt.ylabel('Return')
        plt.grid(True, linestyle='--',alpha=0.5)
        plt.show()

        # ------------ 年度收益率对比直方图 -------------
        # 计算年收益率
        df['year'] = df.index.year
        strategy_annual_return = df.groupby('year')['daily_profit_s'].apply(lambda x: x.sum() / self.initial_capital)
        bah_annual_return = df.groupby('year')['daily_profit'].apply(lambda x: x.sum() / self.initial_capital)

        # 绘制年度收益率对比图
        # 创建图形
        fig4, ax = plt.subplots(figsize=(14, 4))

        # 绘制柱状图
        bars_strategy = ax.bar(strategy_annual_return.index - 0.2, strategy_annual_return, width=0.4, label='Strategy Annual Return', color='C3',alpha=0.7)
        bars_benchmark = ax.bar(bah_annual_return.index + 0.2, bah_annual_return, width=0.4, label='Benchmark Annual Return', color='C7',alpha=0.7)

        # 在每个柱子顶部标注数字
        for bar in bars_strategy:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2%}',weight='bold', ha='center', va='bottom', fontsize=10, color='C3')

        for bar in bars_benchmark:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2%}',weight='bold', ha='center', va='bottom', fontsize=10, color='C7')

        # 设置标题和标签
        ax.set_title('Annual Return Comparison: Strategy vs Benchmark', fontsize=14, weight='bold', color='black', pad=20)
        ax.set_xlabel('Year', fontsize=12, color='black', labelpad=10)
        ax.set_ylabel('Annual Return', fontsize=12, color='black', labelpad=10)

        # 设置x轴为年份整数
        ax.set_xticks(strategy_annual_return.index)
        ax.set_xticklabels(strategy_annual_return.index, fontsize=12, color='black')

        # 设置y轴
        ax.set_ylim(min(strategy_annual_return.min(), bah_annual_return.min()) * 1.1, max(strategy_annual_return.max(), bah_annual_return.max()) * 1.1)
        ax.tick_params(axis='y', labelsize=12)

        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.6)

        # 添加图例
        ax.legend(loc='upper left', fontsize=12)

        # 美化边框和去除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 显示图形
        plt.tight_layout()
        plt.show()

        #-------------收益率分布图---------------
        # 创建图形
        fig3 = plt.figure(figsize=(14, 4))  # 增加高度，使得图表更清晰

        # 绘制直方图
        plt.hist(df['daily_profit_s'] / self.initial_capital, bins=len(df)//10, range=(-0.01, 0.01), color='C3', alpha=0.5, label='Strategy Return')
        plt.hist(df['daily_profit'] / self.initial_capital,bins=len(df)//10,range=(-0.01, 0.01),color='C7',alpha=0.3, label='Benchmark Return',stacked=True)

        # 添加标题与标签
        plt.title('Return Distribution', fontsize=14, weight='bold', color='black', pad=20)
        plt.xlabel('Daily Return', fontsize=12, color='black', labelpad=10)
        plt.ylabel('Frequency', fontsize=12, color='black', labelpad=10)

        # 设置网格
        plt.grid(True, linestyle='--', alpha=0.6)

        # 美化x轴与y轴刻度
        plt.xticks(fontsize=12, color='black')
        plt.yticks(fontsize=12, color='black')

        # 添加边框（美化）
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.legend(loc='upper left', fontsize=12)

        # 显示图形
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_heatmap_by_year(self):
        df = self.result.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # 计算每笔收益占初始资金的比例
        df['daily_return'] = df['daily_profit_s'] / self.initial_capital

        # 按月汇总收益率（非累计）
        monthly_returns = df.groupby(['year', 'month'])['daily_return'].sum().unstack().fillna(0)

        plt.figure(figsize=(14, 6))
        sns.heatmap(monthly_returns.applymap(lambda x: x * 100),  # 转为百分比显示
                    annot=True, fmt=".1f", cmap="RdGy_r", center=0,
                    linewidths=0.5, linecolor='gray', cbar_kws={"label": "Monthly Return (%)"})
        
        plt.title("Monthly Strategy Returns (%)", fontsize=14, weight='bold')
        plt.ylabel("Year", fontsize=12, color='black', labelpad=10)
        plt.xlabel("Month", fontsize=12, color='black', labelpad=10)
        plt.tight_layout()
        plt.show()


    def summarize_by_year_month_market_state(self):
        df = self.result.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        summary = []

        grouped = df.groupby(['year', 'month', 'market_state_v2'])

        for (year, month, state), group in grouped:
            if len(group) < 2:
                continue

            returns = group['return_strategy'].pct_change().dropna()
            duration = len(group)
            total_return = (group['return_strategy'].iloc[-1] / group['return_strategy'].iloc[0]) - 1
            annualized_return = (1 + total_return) ** (252 / duration) - 1 if duration > 0 else 0
            drawdown = group['return_strategy'] / group['return_strategy'].cummax() - 1
            max_drawdown = drawdown.min()
            daily_profit = group['daily_profit_s']
            win_rate = (daily_profit > 0).sum() / len(daily_profit[daily_profit != 0]) if len(daily_profit[daily_profit != 0]) > 0 else 0

            summary.append({
                'Year': year,
                'Month': month,
                'Market State': state,
                'Duration': duration,
                'Return': total_return,
                'Annualized Return': annualized_return,
                'Max Drawdown': max_drawdown,
                'Win Rate': win_rate
            })

        summary_df = pd.DataFrame(summary)
        return summary_df

    def evaluate_performance(self):
        df = self.result.reset_index(drop=True)
        daily_profit = df['daily_profit_s'].dropna()
        win_rate = (daily_profit > 0).sum() / len(daily_profit[daily_profit != 0])
        daily_return = daily_profit / self.initial_capital

        if len(df) == 0:
            print("Error: 策略无交易或数据缺失，无法评估绩效。")
            return

        total_days = len(df['return_strategy'])
        final_nav = df['return_strategy'].iloc[-1]

        annual_return = final_nav ** (252 / total_days) - 1
        annual_vol = daily_return.std() * np.sqrt(252)
        sharpe = (annual_return - self.rf) / annual_vol if annual_vol != 0 else np.nan

        peak = df['return_strategy'].cummax()
        drawdown = (df['return_strategy'] - peak) / peak
        max_drawdown = drawdown.min()
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

        # 计算盈亏比 (Profit Factor)
        profit_trades = daily_profit[daily_profit > 0].sum()
        loss_trades = abs(daily_profit[daily_profit < 0].sum())
        profit_factor = profit_trades / loss_trades if loss_trades != 0 else np.nan
        
        # --------- 策略绩效指标图表 ---------
        performance_metrics = {
            'Cummulative Return': final_nav-1,
            'Annual Return': annual_return,
            'Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor
        }
        performance_metrics = pd.DataFrame(performance_metrics,index=['value'])
        return performance_metrics
        

