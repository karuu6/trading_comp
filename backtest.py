import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def load_data(file):
    """
    Load data from a CSV file, parse dates, and sort by index.

    Parameters:
    file (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded and sorted DataFrame.
    """
    df = pd.read_csv('data.csv', index_col=['symbol', 'timestamp'], parse_dates=['timestamp'])
    df = df.sort_index()
    return df

def train_test_split(df, test_size=0.2):
    """
    Split the DataFrame into training and testing sets based on timestamp.

    Parameters:
    df (pd.DataFrame): DataFrame to split.
    test_size (float): Proportion of the data to use for testing.

    Returns:
    tuple: Training and testing DataFrames.
    """
    unique_dates = df.index.get_level_values('timestamp').unique()
    split_idx = int(len(unique_dates) * (1 - test_size))
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]

    train_df = df[df.index.get_level_values('timestamp').isin(train_dates)]
    test_df = df[df.index.get_level_values('timestamp').isin(test_dates)]
    return train_df, test_df

def calculate_beta(df, market_symbol='SPY'):
    """
    Calculate the beta of each symbol relative to the market symbol.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    market_symbol (str): Symbol of the market index.

    Returns:
    pd.Series: Betas of each symbol.
    """
    ret = df.groupby('symbol').pct_change()['close'].dropna()

    def beta(group):
        cov = group.cov(ret.loc[market_symbol])
        var = ret.loc[market_symbol].var()
        return cov / var

    betas = ret.groupby('symbol').apply(beta).rename('Beta')
    return betas

def backtest(df, test_size=0.2):
    """
    Perform a backtest on the data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    test_size (float): Proportion of the data to use for testing.

    Returns:
    pd.Series: Portfolio returns.
    """
    # Split the data into training and testing sets
    train, test = train_test_split(df)
    
    # Calculate betas for each symbol
    betas = calculate_beta(df)
    
    # Calculate daily returns for each symbol
    ret = test.groupby('symbol').pct_change()['close'].dropna()
    
    # Identify the best and worst performing symbols each day
    winner = ret.groupby('timestamp').apply(lambda day: day.idxmax())
    loser = ret.groupby('timestamp').apply(lambda day: day.idxmin())
    
    # Get the next day's data for the winning and losing symbols
    longs = test.groupby('symbol').shift(-1).loc[winner].dropna()
    shorts = test.groupby('symbol').shift(-1).loc[loser].dropna()
    
    # Calculate returns for long and short positions
    long_ret = ((longs['close'] - longs['open']) / longs['open'])
    short_ret = (-(shorts['close'] - shorts['open']) / shorts['open'])
    
    # Align betas with the returns
    long_betas = betas[long_ret.index.get_level_values('symbol')].rename('LBeta')
    long_betas.index = long_ret.index.get_level_values('timestamp')
    short_betas = -betas[short_ret.index.get_level_values('symbol')].rename('SBeta')
    short_betas.index = short_ret.index.get_level_values('timestamp')

    def reg(x):
        """
        Calculate weights for long and short positions.

        Parameters:
        x (pd.Series): Series containing betas.

        Returns:
        pd.Series: Weights for long and short positions.
        """
        y = np.array([0, 1])
        X = np.array([x.tolist(), [1, 1]])
        w = np.linalg.solve(X, y)
        return pd.Series(w, index=['LWeight', 'SWeight'])

    # Calculate weights for the portfolio
    weights = pd.concat((short_betas, long_betas), axis=1).apply(reg, axis=1)
    
    # Calculate portfolio returns
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