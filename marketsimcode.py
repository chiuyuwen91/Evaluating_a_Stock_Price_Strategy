
import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data


def compute_portvals(orders_df, start_val=100000, commission=0, impact=0):
    orders_df.sort_index(inplace=True)
    symbols = list(orders_df.Symbol.unique())
    # symbols = orders_df.columns
    start_date, end_date = get_date(orders_df)
    prices = get_price(symbols, start_date, end_date)

    portvals = market_simulation(prices, orders_df, start_val, commission, impact)
    return portvals


def market_simulation(prices, orders_df, start_val, commission, impact):
    trades = prices.copy()
    trades[:] = 0

    for date, row in orders_df.iterrows():
        # date = row.Date
        order = row.Order
        symbol = row.Symbol
        shares = row.Shares

        if order == 'BUY':
            trades.loc[date, symbol] += shares
            trades.loc[date, 'Cash'] += shares * (prices.loc[date, symbol] * (1. + impact)) * -1. - commission
        else:
            trades.loc[date, symbol] += shares * -1.
            trades.loc[date, 'Cash'] += shares * (prices.loc[date, symbol] * (1. - impact)) - commission

    holdings = trades.copy().cumsum()
    holdings['Cash'] += start_val

    values = prices * holdings

    portfolio_values = values.sum(axis=1)
    return portfolio_values

def get_date(orders_df):
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()
    return start_date, end_date

def get_price(symbols, start_date, end_date):
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices.ffill().bfill()
    prices = prices[symbols]
    prices['Cash'] = 1
    return prices


def portfolio_stats(port_val):
    cum_ret = (port_val[-1] / port_val[0]) - 1
    daily_rets = (port_val / port_val.shift(1)) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    return cum_ret, adr, sddr

def author():
    return 'ychiu60'
