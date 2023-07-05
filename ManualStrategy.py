
import util as ut
import marketsimcode as ms
import indicators as inds

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ManualStrategy(object):

    def __init__(self, symbol, verbose=False, impact=0.005, commission=9.95):
        self.symbol = symbol
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def author(self):
        return 'ychiu60'


    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_all = prices_all.ffill().bfill()
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        price = prices / prices.iloc[0]
        # price = price.sort_index(axis=0)

        # df_price = df/ df.iloc[0]
        # df_price = df_price.sort_index(axis=0)

        lookback = 20
        momentum = inds.get_momentum(prices, lookback)
        sma = inds.get_sma_ratio(prices, lookback)
        top_band, bottom_band, percentage = inds.get_Bollinger_Bands(prices, lookback)

        trading_date = price.index
        df_trades = pd.DataFrame(index=trading_date, columns=['Symbol', 'Order', 'Shares'])
        df_trades.Symbol = 'JPM'
        df_trades.Order = np.NaN
        df_trades.Shares = 1000
        curr_position = 0

        for i in range(df_trades.shape[0] - 1):
            momentum_price_today = momentum.iloc[i+1, 0]
            momentum_price_yesterday = momentum.iloc[i, 0]
            sma_ratio = sma.iloc[i, 0]
            percentage_today = percentage.iloc[i, 0]
            momentum_cal = (momentum_price_today - momentum_price_yesterday) / momentum_price_yesterday

            if (sma_ratio < 0.95) & (momentum_cal < -0.2) | \
                    (percentage_today < 0.2):
                if curr_position <= 0:
                    curr_position += 1000
                    df_trades.iloc[i, 2] += curr_position
                    df_trades.iloc[i, 1] = 'BUY'
            elif (sma_ratio > 1.1) & (momentum_cal > 0.2) | \
                    (percentage_today > 0.8):
                if curr_position >= 0:
                    curr_position -= 1000
                    df_trades.iloc[i, 2] -= curr_position
                    df_trades.iloc[i, 1] = 'SELL'

            elif (percentage_today < 0.2) & (sma_ratio < 0) & (momentum_cal < -1):
                if curr_position <= 0:
                    curr_position += 0
                    df_trades.iloc[i, 2] += curr_position
                    df_trades.iloc[i, 1] = 'BUY'
            elif (percentage_today > 0.8) & (sma_ratio > 0) & (momentum_cal > 1):
                if curr_position >= 0:
                    curr_position -= 0
                    df_trades.iloc[i, 2] -= curr_position
                    df_trades.iloc[i, 1] = 'SELL'

        df_trades.dropna(inplace=True)
        return df_trades


    def benchmark(self, symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        df_bm = self.testPolicy(symbol, sd, ed, sv)
        df_bm.iloc[0] = [symbol, 'BUY', 1000]
        for i in range(1, len(df_bm.index)):
            df_bm.iloc[i, 0] = symbol
            df_bm.iloc[i, 1] = 'HOLD'
            df_bm.iloc[i, 2] = 0

        df_bm.dropna(inplace=True)
        return df_bm

    def portfolio_stats(self, port_val):
        cum_ret = (port_val[-1] / port_val[0]) - 1
        daily_rets = (port_val / port_val.shift(1)) - 1
        adr = daily_rets.mean()
        sddr = daily_rets.std()
        return cum_ret, adr, sddr


    def plot_in_sample(self, symbol):
        np.random.seed(903847133)
        df_trades = self.testPolicy(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
        portvals = ms.compute_portvals(df_trades, 100000, 9.95, 0.005)
        portvals = portvals / portvals.iloc[0]
        portvals = pd.DataFrame(portvals)

        bm_trades = self.benchmark(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)
        bm_portvals = ms.compute_portvals(bm_trades, 100000, 9.95, 0.005)
        bm_portvals = bm_portvals / bm_portvals.iloc[0]
        bm_portvals = pd.DataFrame(bm_portvals)

        plt.figure()
        plt.plot(portvals, 'r', label='Manual Strategy')
        plt.plot(bm_portvals, 'purple', label='Benchmark')
        plt.title('Manual Strategy vs. Benchmark (In-Sample)')
        plt.grid(True)
        plt.xlabel('Date')
        plt.xticks(rotation=10)
        plt.ylabel('Normalized Value')
        plt.yticks()
        plt.legend(['Manual Strategy', 'Benchmark'], loc='best')

        for index, n in df_trades.iterrows():
            if n.Order == 'BUY':
                plt.axvline(index, color='black', linewidth=0.3, linestyle='--')
            else:
                plt.axvline(index, color='blue', linewidth=0.3, linestyle='--')

        plt.savefig('insample.png')
        plt.clf()

    def plot_out_of_sample(self, symbol):
        np.random.seed(903847133)
        df_trades = self.testPolicy(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
        portvals = ms.compute_portvals(df_trades, 100000, 9.95, 0.005)
        portvals = portvals / portvals.iloc[0]
        portvals = pd.DataFrame(portvals)

        bm_trades = self.benchmark(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
        bm_portvals = ms.compute_portvals(bm_trades, 100000, 9.95, 0.005)
        bm_portvals = bm_portvals / bm_portvals.iloc[0]
        bm_portvals = pd.DataFrame(bm_portvals)

        plt.figure()
        plt.plot(portvals, 'r', label="Manual Strategy")
        plt.plot(bm_portvals, 'purple', label="Benchmark")
        plt.title('Manual Strategy vs. Benchmark (Out-Of-Sample)')
        plt.grid(True)
        plt.xlabel('Date')
        plt.xticks(rotation=10)
        plt.ylabel('Normalized Value')
        plt.yticks()
        plt.legend(['Manual Strategy', 'Benchmark'], loc='best')

        for index, row in df_trades.iterrows():
            if row.Order == 'BUY':
                plt.axvline(index, color='black', linewidth=0.3, linestyle='--')
            else:
                plt.axvline(index, color='blue', linewidth=0.3, linestyle='--')


        plt.savefig('outofsample.png')
        plt.clf()

    def table(self, symbol):
        df_trade_in = self.testPolicy(symbol, dt.datetime(2008,1,1), dt.datetime(2009,12,31), 100000)
        portvals_in = ms.compute_portvals(df_trade_in, start_val=100000, commission=9.95, impact=0.005)
        portvals_in = portvals_in / portvals_in.iloc[0]

        bm_trade_in = self.benchmark(symbol, dt.datetime(2008,1,1), dt.datetime(2009,12,31), 100000)
        bm_portvals_in = ms.compute_portvals(bm_trade_in, 100000, 9.95, 0.005)
        bm_portvals_in = bm_portvals_in / bm_portvals_in.iloc[0]

        cummulative_returns_opt_in, average_daily_return_opt_in, std_daily_return_opt_in = self.portfolio_stats(
            portvals_in)
        cummulative_returns_benc_in, average_daily_return_benc_in, std_daily_return_benc_in = self.portfolio_stats(
            bm_portvals_in)

        cummulative_returns_opt_in = round(cummulative_returns_opt_in,6)
        average_daily_return_opt_in = round(average_daily_return_opt_in,6)
        std_daily_return_opt_in = round(std_daily_return_opt_in,6)
        cummulative_returns_benc_in = round(cummulative_returns_benc_in,6)
        average_daily_return_benc_in = round(average_daily_return_benc_in,6)
        std_daily_return_benc_in = round(std_daily_return_benc_in,6)

        df_trades_out = self.testPolicy(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
        portvals_out = ms.compute_portvals(df_trades_out, start_val=100000, commission=9.95, impact=0.005)
        portvals_out = portvals_out / portvals_out.iloc[0]

        bm_portvals_out = self.benchmark(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 100000)
        bm_portvals_out = ms.compute_portvals(bm_portvals_out, 100000, 9.95, 0.005)
        bm_portvals_out = bm_portvals_out / bm_portvals_out.iloc[0]

        cummulative_returns_opt_out, average_daily_return_opt_out, std_daily_return_opt_out = self.portfolio_stats(
            portvals_out)
        cummulative_returns_benc_out, average_daily_return_benc_out, std_daily_return_benc_out = self.portfolio_stats(
            bm_portvals_out)

        cummulative_returns_opt_out = round(cummulative_returns_opt_out,6)
        average_daily_return_opt_out = round(average_daily_return_opt_out,6)
        std_daily_return_opt_out = round(std_daily_return_opt_out,6)
        cummulative_returns_benc_out = round(cummulative_returns_benc_out,6)
        average_daily_return_benc_out = round(average_daily_return_benc_out,6)
        std_daily_return_benc_out = round(std_daily_return_benc_out,6)

        pre = {'Manual Strategy (In Sample)': [cummulative_returns_opt_in, average_daily_return_opt_in, std_daily_return_opt_in],
               'Benchmark (In Sample)': [cummulative_returns_benc_in, average_daily_return_benc_in, std_daily_return_benc_in],
               'Manual Strategy (Out of Sample)': [cummulative_returns_opt_out, average_daily_return_opt_out, std_daily_return_opt_out],
               'Benchmark (Out Of Sample)': [cummulative_returns_benc_out, average_daily_return_benc_out, std_daily_return_benc_out]}
        df = pd.DataFrame(data=pre)
        df = df.rename(index={0: 'cummulative_returns', 1: 'average_daily_return', 2:'std_daily_return'})

        df.to_csv('table.csv')
if __name__ == "__main__":
    msg = ManualStrategy(symbol='JPM')
    msg.plot_in_sample(symbol='JPM')
    msg.plot_out_of_sample(symbol='JPM')
    msg.table(symbol='JPM')

