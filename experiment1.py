import marketsimcode as ms
import ManualStrategy as msgy
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

def plots_in_sample(symbol):
    np.random.seed(903847133)
    msg = msgy.ManualStrategy(symbol)
    df_trades = msg.testPolicy(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    portvals = ms.compute_portvals(df_trades, 100000, 9.95, 0.005)
    portvals = portvals / portvals.iloc[0]
    portvals = pd.DataFrame(portvals)

    bm_df_trades = msg.benchmark(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    bm_portvals = ms.compute_portvals(bm_df_trades, 100000, 9.95, 0.005)
    bm_portvals = bm_portvals / bm_portvals.iloc[0]
    bm_portvals = pd.DataFrame(bm_portvals)

    SLearner = sl.StrategyLearner(verbose=False, impact=0.005)
    SLearner.add_evidence(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    sl_df_trades = SLearner.testPolicy(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    sl_df_trades = trades(sl_df_trades, symbol)
    sl_portvals = ms.compute_portvals(sl_df_trades, 100000, 9.95, 0.005)
    sl_portvals = sl_portvals / sl_portvals.iloc[0]
    sl_portvals = pd.DataFrame(sl_portvals)

    plt.figure()
    plt.plot(portvals, 'r', label='Manual Strategy')
    plt.plot(bm_portvals, 'purple', label='Benchmark')
    plt.plot(sl_portvals, 'g', label='Strategy Learner')
    plt.title('Manual Strategy vs. Benchmark vs. Strategy Learner(In-Sample)')
    plt.grid(True)
    plt.xlabel('Date')
    plt.xticks(rotation=10)
    plt.ylabel('Normalized Value')
    plt.yticks()
    plt.legend(['Manual Strategy', 'Benchmark', 'Strategy Learner'], loc='best')

    plt.savefig('ex1_in.png')
    plt.clf()

def plots_out_of_sample(symbol):
    np.random.seed(903847133)
    msg = msgy.ManualStrategy(symbol)
    df_trades = msg.testPolicy(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
    portvals = ms.compute_portvals(df_trades, 100000, 9.95, 0.005)
    portvals = portvals / portvals.iloc[0]
    portvals = pd.DataFrame(portvals)

    bm_df_trades = msg.benchmark(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
    bm_portvals = ms.compute_portvals(bm_df_trades, 100000, 9.95, 0.005)
    bm_portvals = bm_portvals / bm_portvals.iloc[0]
    bm_portvals = pd.DataFrame(bm_portvals)

    SLearner = sl.StrategyLearner(verbose=False, impact=0.005)
    SLearner.add_evidence(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
    sl_df_trades = SLearner.testPolicy(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
    sl_df_trades = trades(sl_df_trades, symbol)
    sl_portvals = ms.compute_portvals(sl_df_trades, 100000, 9.95, 0.005)
    sl_portvals = sl_portvals / sl_portvals.iloc[0]
    sl_portvals = pd.DataFrame(sl_portvals)

    plt.figure()
    plt.plot(portvals, 'r', label='Manual Strategy')
    plt.plot(bm_portvals, 'purple', label='Benchmark')
    plt.plot(sl_portvals, 'g', label='Strategy Learner')
    plt.title('Manual Strategy vs. Benchmark vs. Strategy Learner(Out-Of-Sample)')
    plt.grid(True)
    plt.xlabel('Date')
    plt.xticks(rotation=10)
    plt.ylabel('Normalized Value')
    plt.yticks()
    plt.legend(['Manual Strategy', 'Benchmark', 'Strategy Learner'], loc='best')

    plt.savefig('ex1_out.png')
    plt.clf()

def portfolio_stats(port_val):
    cum_ret = (port_val[-1] / port_val[0]) - 1
    daily_rets = (port_val / port_val.shift(1)) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    return cum_ret, adr, sddr

def table_in_sample(symbol):
    msg = msgy.ManualStrategy(symbol)
    slr = sl.StrategyLearner(verbose=False, impact=0.005)
    df_trade_in = msg.testPolicy(symbol, dt.datetime(2008,1,1), dt.datetime(2009,12,31), 100000)
    portvals_in = ms.compute_portvals(df_trade_in, start_val=100000, commission=9.95, impact=0.005)
    portvals_in = portvals_in / portvals_in.iloc[0]

    bm_df_trades_in = msg.benchmark(symbol, dt.datetime(2008,1,1), dt.datetime(2009,12,31), 100000)
    bm_portvals_in = ms.compute_portvals(bm_df_trades_in, 100000, 9.95, 0.005)
    bm_portvals_in = bm_portvals_in / bm_portvals_in.iloc[0]

    slr.add_evidence(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    sl_df_trades = slr.testPolicy(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
    sl_df_trades = trades(sl_df_trades, symbol)
    sl_portvals_in = ms.compute_portvals(sl_df_trades, 100000, 9.95, 0.005)
    sl_portvals_in = sl_portvals_in / sl_portvals_in.iloc[0]

    cumulative_returns_opt_in, average_daily_return_opt_in, std_daily_return_opt_in = portfolio_stats(
        portvals_in)
    cumulative_returns_benc_in, average_daily_return_benc_in, std_daily_return_benc_in = portfolio_stats(
        bm_portvals_in)
    cumulative_returns_slr_in, average_daily_return_slr_in, std_daily_return_slr_in = portfolio_stats(
        sl_portvals_in)

    cummulative_returns_opt_in = round(cumulative_returns_opt_in,6)
    average_daily_return_opt_in = round(average_daily_return_opt_in,6)
    std_daily_return_opt_in = round(std_daily_return_opt_in,6)
    cummulative_returns_benc_in = round(cumulative_returns_benc_in,6)
    average_daily_return_benc_in = round(average_daily_return_benc_in,6)
    std_daily_return_benc_in = round(std_daily_return_benc_in,6)
    cummulative_returns_slr_in = round(cumulative_returns_slr_in,6)
    average_daily_return_slr_in = round(average_daily_return_slr_in,6)
    std_daily_return_slr_in = round(std_daily_return_slr_in,6)


    pre = {'Manual Strategy (In Sample)': [cummulative_returns_opt_in, average_daily_return_opt_in, std_daily_return_opt_in],
           'Benchmark (In Sample)': [cummulative_returns_benc_in, average_daily_return_benc_in, std_daily_return_benc_in],
           'Strategy Learner (In Sample)': [cummulative_returns_slr_in, average_daily_return_slr_in, std_daily_return_slr_in]}
    df = pd.DataFrame(data=pre)
    df = df.rename(index={0: 'cummulative_returns', 1: 'average_daily_return', 2:'std_daily_return'})

    df.to_csv('ex_in.csv')

def table_out_of_sample(symbol):
    msg = msgy.ManualStrategy(symbol)
    slr = sl.StrategyLearner(verbose=False, impact=0.005)
    df_trades_out = msg.testPolicy(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
    portvals_out = ms.compute_portvals(df_trades_out, start_val=100000, commission=9.95, impact=0.005)
    portvals_out = portvals_out / portvals_out.iloc[0]

    bm_trade_out = msg.benchmark(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
    bm_portvals_out = ms.compute_portvals(bm_trade_out, 100000, 9.95, 0.005)
    bm_portvals_out = bm_portvals_out / bm_portvals_out.iloc[0]

    slr.add_evidence(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
    sl_df_trades = slr.testPolicy(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
    sl_df_trades = trades(sl_df_trades, symbol)
    sl_portvals_out = ms.compute_portvals(sl_df_trades, 100000, 9.95, 0.005)
    sl_portvals_out = sl_portvals_out / sl_portvals_out.iloc[0]

    cumulative_returns_opt_out, average_daily_return_opt_out, std_daily_return_opt_out = portfolio_stats(
        portvals_out)
    cumulative_returns_benc_out, average_daily_return_benc_out, std_daily_return_benc_out = portfolio_stats(
        bm_portvals_out)
    cumulative_returns_slr_out, average_daily_return_slr_out, std_daily_return_slr_out = portfolio_stats(
        sl_portvals_out)

    cumulative_returns_opt_out = round(cumulative_returns_opt_out,6)
    average_daily_return_opt_out = round(average_daily_return_opt_out,6)
    std_daily_return_opt_out = round(std_daily_return_opt_out,6)
    cumulative_returns_benc_out = round(cumulative_returns_benc_out,6)
    average_daily_return_benc_out = round(average_daily_return_benc_out,6)
    std_daily_return_benc_out = round(std_daily_return_benc_out,6)
    cumulative_returns_slr_out = round(cumulative_returns_slr_out,6)
    average_daily_return_slr_out = round(average_daily_return_slr_out,6)
    std_daily_return_slr_out = round(std_daily_return_slr_out,6)

    pre = {'Manual Strategy (Out Of Sample)': [cumulative_returns_opt_out, average_daily_return_opt_out, std_daily_return_opt_out],
           'Benchmark (Out Of Sample)': [cumulative_returns_benc_out, average_daily_return_benc_out, std_daily_return_benc_out],
           'Strategy Learner (Out Of Sample)': [cumulative_returns_slr_out, average_daily_return_slr_out, std_daily_return_slr_out]}

    df = pd.DataFrame(data=pre)
    df = df.rename(index={0: 'cummulative_returns', 1: 'average_daily_return', 2:'std_daily_return'})

    df.to_csv('ex_out.csv')

if __name__ == "__main__":
    plots_in_sample(symbol='JPM')
    plots_out_of_sample(symbol='JPM')
    table_in_sample(symbol='JPM')
    table_out_of_sample(symbol='JPM')
