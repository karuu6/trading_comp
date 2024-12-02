import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def load_data(file):
    df = pd.read_csv('data.csv', index_col=['symbol', 'timestamp'], parse_dates=['timestamp'])
    df = df.sort_index()
    return df

def train_test_split(df, test_size=0.2):
    unique_dates = df.index.get_level_values('timestamp').unique()
    split_idx = int(len(unique_dates) * (1 - test_size))
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]

    train_df = df[df.index.get_level_values('timestamp').isin(train_dates)]
    test_df = df[df.index.get_level_values('timestamp').isin(test_dates)]
    return train_df, test_df

def calculate_beta(df, market_symbol='SPY'):
    ret = df.groupby('symbol').pct_change()['close'].dropna()

    def beta(group):
        cov = group.cov(ret.loc[market_symbol])
        var = ret.loc[market_symbol].var()
        return cov / var

    betas = ret.groupby('symbol').apply(beta).rename('Beta')
    return betas

def backtest(df, test_size=0.2):
    train, test = train_test_split(df)
    betas = calculate_beta(df)
    ret = test.groupby('symbol').pct_change()['close'].dropna()
    winner = ret.groupby('timestamp').apply(lambda day: day.idxmax())
    loser = ret.groupby('timestamp').apply(lambda day: day.idxmin())
    longs = test.groupby('symbol').shift(-1).loc[winner].dropna()
    shorts = test.groupby('symbol').shift(-1).loc[loser].dropna()
    long_ret = ((longs['close'] - longs['open']) / longs['open'])
    short_ret = (-(shorts['close'] - shorts['open']) / shorts['open'])
    long_betas = betas[long_ret.index.get_level_values('symbol')].rename('LBeta')
    long_betas.index = long_ret.index.get_level_values('timestamp')
    short_betas = -betas[short_ret.index.get_level_values('symbol')].rename('SBeta')
    short_betas.index = short_ret.index.get_level_values('timestamp')

    def reg(x):
        y = np.array([0, 1])
        X = np.array([x.tolist(), [1, 1]])
        w = np.linalg.solve(X, y)
        return pd.Series(w, index=['LWeight', 'SWeight'])

    weights = pd.concat((short_betas, long_betas), axis=1).apply(reg, axis=1)
    port_returns = long_ret.droplevel('symbol') * weights['LWeight'] + short_ret.droplevel('symbol') * weights['SWeight']
    return port_returns


if __name__ == '__main__':
    df = load_data('data.csv')
    ret = backtest(df)
    print('Mean returns:', 252 * ret.mean())
    print('Standard deviation of returns:', np.sqrt(252) * ret.std())
    print('Sharpe ratio:', np.sqrt(252) * ret.mean() / ret.std())

    (1 + ret).cumprod().plot()
    plt.show()