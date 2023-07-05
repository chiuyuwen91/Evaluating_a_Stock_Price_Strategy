from util import get_data

def author():
        return 'ychiu60'

def get_momentum(prices, lookback):
    # df = get_data([symbols], dates, False)
    # df = df.ffill().bfill()
    # # df = df.drop(columns='SPY', inplace=True)
    # price = df / df.iloc[0]
    # price = price.sort_index(axis=0)
    momentum = (prices / prices.shift(lookback)) - 1
    return momentum

def get_sma_ratio(prices, lookback):
    # df = get_data([symbols], dates, False)
    # df = df.ffill().bfill()
    # df = df.drop(columns='SPY', inplace=True)
    # price = df / df.iloc[0]

    # price = prices.sort_index(axis=0)
    sma = prices.rolling(lookback).mean()
    sma_ratio = prices / sma 

    return sma_ratio

def get_Bollinger_Bands(prices, lookback):
    # df = get_data([symbols], dates, False)
    # df = df.ffill().bfill()
    # # df = df.drop(columns=["SPY"], inplace=True)
    # price = df / df.iloc[0]
    # price = price.sort_index(axis=0)
    sma_bb = prices.rolling(lookback).mean()
    sma_std = prices.rolling(lookback).std()
    top_band = sma_bb + (2 * sma_std)
    bottom_band = sma_bb - (2 * sma_std)
    percentage = (prices - bottom_band) / (top_band - bottom_band)
    return top_band, bottom_band, percentage


