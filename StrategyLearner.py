import RTLearner as rtl
import BagLearner as bl

import datetime as dt
import numpy as np
import pandas as pd
import util as ut

import ManualStrategy as msg
import marketsimcode as ms
import indicators as inds

class StrategyLearner(object):

    def __init__(self, verbose = False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = bl.BagLearner(learner=rtl.RTLearner, bags=20, kwargs={"leaf_size": 5}, boost=False, verbose=False)

    def author(self):
        return 'ychiu60'


    def add_evidence(self, symbol = "AAPL", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_all = prices_all.ffill().bfill()
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        # price = prices / prices.iloc[0]
        # price = price.sort_index(axis=0)

        lookback = 15
        momentum = inds.get_momentum(prices, lookback)
        momentum = momentum.rename(columns={symbol: 'momentum'})
        sma = inds.get_sma_ratio(prices, lookback)
        sma = sma.rename(columns={symbol: 'sma'})
        top_band, bottom_band, percentage = inds.get_Bollinger_Bands(prices, lookback)
        percentage = percentage.rename(columns={symbol: 'percentage'})

        data_all = pd.concat([momentum, sma, percentage], axis=1)
        data_all = data_all.ffill().bfill()
        x_train = data_all[:-10].values


        y_train=[]
        for i in range(prices[10:].shape[0]):
            YBUY = self.impact
            YSELL = self.impact
            ret = (prices.iloc[i+10] / prices.iloc[i]) - 1
            if float(ret) > YBUY:
                y_train.append(1)
            elif float(ret) < YSELL:
                y_train.append(-1)
            else:
                y_train.append(0)

        y_train = np.array(y_train)
        print(y_train)
        self.learner.add_evidence(x_train, y_train)


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_all = prices_all.ffill().bfill()
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        # price = prices / prices.iloc[0]
        # price = price.sort_index(axis=0)


        lookback = 15
        momentum = inds.get_momentum(prices, lookback)
        momentum = momentum.rename(columns={symbol: 'momentum'})
        sma = inds.get_sma_ratio(prices, lookback)
        sma = sma.rename(columns={symbol: 'sma'})
        top_band, bottom_band, percentage = inds.get_Bollinger_Bands(prices, lookback)
        percentage = percentage.rename(columns={symbol: 'percentage'})

        data_all = pd.concat([momentum, sma, percentage], axis=1)
        data_all = data_all.ffill().bfill()

        shares = 0
        orders = []

        for _, data in data_all.iterrows():
            x_test = data.values.reshape(1, -1)
            y_test = self.learner.query(x_test)

            if shares == 0:
                if y_test > 0:
                    shares = 1000
                    orders.append((data.name, 'BUY', 1000))
                elif y_test < 0:
                    shares = -1000
                    orders.append((data.name, 'SELL', -1000))
            elif shares > 0 and y_test < 0:
                shares = -1000
                orders.append((data.name, 'SELL', -2000))
            elif shares < 0 and y_test > 0:
                shares = 1000
                orders.append((data.name, 'BUY', 2000))
            else:
                orders.append((data.name, 'HOLD', 0))

        # df_trades = pd.DataFrame(orders, columns=[symbol, "Order", "Shares"])
        df_trades = pd.DataFrame(orders, columns=['Dates', 'Order', symbol])
        df_trades.set_index('Dates', inplace=True)
        df_trades = df_trades.drop('Order', axis=1)
        # print(df_trades)
        return df_trades

if __name__=="__main__":
    print ("One does not simply think up a strategy")
    sl = StrategyLearner()
    sl.add_evidence(symbol='JPM',sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    sl.testPolicy(symbol='JPM',sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000)
