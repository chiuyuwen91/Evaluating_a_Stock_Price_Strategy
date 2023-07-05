import marketsimcode as ms
import StrategyLearner as sl
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def author():
        return 'ychiu60'

def trades(prices, symbol):
    buy_2000 = prices[symbol] == 2000
    buy_1000 = prices[symbol] == 1000
    sell_2000 = prices[symbol] == -2000
    sell_1000 = prices[symbol] == -1000

    trades = pd.DataFrame(columns=['Date', symbol, 'Order', 'Shares'])
    trades = trades.append(pd.DataFrame({'Date': prices[buy_2000].index, 'Symbol': symbol, 'Order': 'BUY', 'Shares': 2000}), ignore_index=True)
    trades = trades.append(pd.DataFrame({'Date': prices[buy_1000].index, 'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000}), ignore_index=True)
    trades = trades.append(pd.DataFrame({'Date': prices[sell_2000].index, 'Symbol': symbol, 'Order': 'SELL', 'Shares': 2000}), ignore_index=True)
    trades = trades.append(pd.DataFrame({'Date': prices[sell_1000].index, 'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000}), ignore_index=True)
    trades.set_index('Date', inplace=True)

    return trades


def plot_diff_impacts():
    np.random.seed(903847133)
    impact1 = 0.0
    impact2 = 0.005
    impact3 = 0.01
    SLearner1 = sl.StrategyLearner(verbose=False, impact=impact1)
    SLearner1.add_evidence('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    sl_df_trades1 = SLearner1.testPolicy('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    sl_df_trades1 = trades(sl_df_trades1, 'JPM')
    sl_portvals1 = ms.compute_portvals(sl_df_trades1, 100000, 9.95, impact1)
    sl_portvals1 = sl_portvals1 / sl_portvals1.iloc[0]
    sl_portvals1 = pd.DataFrame(sl_portvals1)

    SLearner2 = sl.StrategyLearner(verbose=False, impact=impact2)
    SLearner2.add_evidence('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    sl_df_trades2 = SLearner2.testPolicy('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    sl_df_trades2 = trades(sl_df_trades2, 'JPM')
    sl_portvals2 = ms.compute_portvals(sl_df_trades2, 100000, 9.95, impact2)
    sl_portvals2 = sl_portvals2 / sl_portvals2.iloc[0]
    sl_portvals2 = pd.DataFrame(sl_portvals2)

    SLearner3 = sl.StrategyLearner(verbose=False, impact=impact3)
    SLearner3.add_evidence('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    sl_df_trades3 = SLearner1.testPolicy('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    sl_df_trades3 = trades(sl_df_trades3, 'JPM')
    sl_portvals3 = ms.compute_portvals(sl_df_trades3, 100000, 9.95, impact3)
    sl_portvals3 = sl_portvals3 / sl_portvals3.iloc[0]
    sl_portvals3 = pd.DataFrame(sl_portvals3)

    plt.figure()
    plt.plot(sl_portvals1, 'r', label='Strategy Learner (Impact:0.0)')
    plt.plot(sl_portvals2, 'purple', label='Strategy Learner (Impact:0.005)')
    plt.plot(sl_portvals3, 'g', label='Strategy Learner (Impact:0.01)')
    plt.title('Strategy Learner(Impact:0.0 vs. Impact:0.005 vs. Impact:0.01')
    plt.grid(True)
    plt.xlabel('Date')
    plt.xticks(rotation=10)
    plt.ylabel('Normalized Value')
    plt.yticks()
    plt.legend(['Strategy Learner(Impact:0.0)', 'Strategy Learner(Impact:0.005)', 'Strategy Learner(Impact:0.01)'], loc='best')

    plt.savefig('ex2.png')
    plt.clf()

def table_diff_impacts():
    np.random.seed(903847133)
    impact1 = 0.0
    impact2 = 0.005
    impact3 = 0.01

    Slearner1 = sl.StrategyLearner(verbose=False, impact=impact1)
    Slearner1.add_evidence('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    df_trades1 = Slearner1.testPolicy('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    df_trades1 = trades(df_trades1, 'JPM')
    sl_portvals1 = ms.compute_portvals(df_trades1, 100000, 0, impact1)
    sl_portvals1 = sl_portvals1 / sl_portvals1.iloc[0]

    Slearner2 = sl.StrategyLearner(verbose=False, impact=impact2)
    Slearner2.add_evidence('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    df_trades2 = Slearner2.testPolicy('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    df_trades2 = trades(df_trades2, 'JPM')
    sl_portvals2 = ms.compute_portvals(df_trades2, 100000, 0, impact2)
    sl_portvals2 = sl_portvals2 / sl_portvals2.iloc[0]

    Slearner3 = sl.StrategyLearner(verbose=False, impact=impact3)
    Slearner3.add_evidence('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    df_trades3 = Slearner3.testPolicy('JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    df_trades3 = trades(df_trades3, 'JPM')
    sl_portvals3 = ms.compute_portvals(df_trades3, 100000, 0, impact3)
    sl_portvals3 = sl_portvals3 / sl_portvals3.iloc[0]


    cumulative_returns_slr1, average_daily_return_slr1, std_daily_return_slr1 = metrics(
        sl_portvals1)
    cumulative_returns_slr2, average_daily_return_slr2, std_daily_return_slr2 = metrics(
        sl_portvals2)
    cumulative_returns_slr3, average_daily_return_slr3, std_daily_return_slr3 = metrics(
        sl_portvals3)

    cumulative_returns_slr1 = round(cumulative_returns_slr1,6)
    average_daily_return_slr1 = round(average_daily_return_slr1,6)
    std_daily_return_slr1 = round(std_daily_return_slr1,6)
    cumulative_returns_slr2 = round(cumulative_returns_slr2,6)
    average_daily_return_slr2 = round(average_daily_return_slr2,6)
    std_daily_return_slr2 = round(std_daily_return_slr2,6)
    cumulative_returns_slr3 = round(cumulative_returns_slr3,6)
    average_daily_return_slr3 = round(average_daily_return_slr3,6)
    std_daily_return_slr3 = round(std_daily_return_slr3,6)

    pre = {'Strategy Learner (Impact:0.0)': [cumulative_returns_slr1, average_daily_return_slr1, std_daily_return_slr1],
           'Strategy Learner (Impact:0.005)': [cumulative_returns_slr2, average_daily_return_slr2, std_daily_return_slr2],
           'Strategy Learner (Impact:0.01)': [cumulative_returns_slr3, average_daily_return_slr3, std_daily_return_slr3]}

    df = pd.DataFrame(data=pre)
    df = df.rename(index={0: 'cummulative_returns', 1: 'average_daily_return', 2:'std_daily_return'})

    df.to_csv('ex2.csv')

def metrics(port_val):
    cum_ret = (port_val[-1] / port_val[0]) - 1
    daily_rets = (port_val / port_val.shift(1)) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    return cum_ret, adr, sddr

if __name__ == "__main__":
    table_diff_impacts()
    plot_diff_impacts()
