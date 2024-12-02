from data import get_nasdaq_100_constituents, get_last_year_bars
from alpaca.trading.client import TradingClient
from backtest import calculate_beta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.requests import StockLatestBarRequest, StockLatestQuoteRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import numpy as np
import schedule
import time
import os

OPEN_ORDERS = []

def get_latest_bars(symbols, client):
    request_params = StockLatestBarRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day
    )
    return client.get_stock_latest_bar(request_params)

def generate_positions(client):
    # Get the list of NASDAQ-100 constituents and add 'SPY' to the list
    symbols = get_nasdaq_100_constituents() + ['SPY']
    # Get the latest stock bars for the symbols
    bars = get_last_year_bars(symbols, client)
    # Calculate beta values for the symbols
    betas = calculate_beta(bars)
    # Get the latest stock bars for the symbols
    latest_bars = get_latest_bars(symbols, client)

    ret = {symbol: (data.close - data.open) / data.open for symbol, data in latest_bars.items()}
    winner = max(ret, key=ret.get)
    loser = min(ret, key=ret.get)

    X = np.array([[betas[winner], -betas[loser]], [1, 1]])
    y = np.array([0, 1])
    w = np.linalg.solve(X, y)
    return {winner: w[0], loser: -w[1]}

def enter_opening_orders(positions, trading_client, market_data_client):
    value = float(trading_client.get_account().cash)

    orders = []
    for symbol, weight in positions.items():
        price_req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        price = market_data_client.get_stock_latest_quote(price_req)[symbol].bid_price
        if weight > 0:
            shares = int(value * weight / price)
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.OPG
            )
            order = trading_client.submit_order(order_data=order_req)
            orders.append((symbol, shares))
        else:
            shares = int(value * -weight / price)
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.OPG
            )
            order = trading_client.submit_order(order_data=order_req)
            orders.append((symbol, -shares))
    return orders

def enter_closing_orders(opening_orders):
    for symbol, shares in opening_orders:
        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=-shares,
            side=OrderSide.SELL if shares > 0 else OrderSide.BUY,
            time_in_force=TimeInForce.CLS
        )
        trading_client.submit_order(order_data=order_req)

def schedule_open(trading_client, market_data_client):
    global OPEN_ORDERS
    pos = generate_positions(market_data_client)
    OPEN_ORDERS = enter_opening_orders(pos, trading_client, market_data_client)

def schedule_close(trading_client):
    enter_closing_orders(OPEN_ORDERS)


if __name__ == '__main__':
    api_key = os.getenv('ALPACA_API_KEY')  # Get Alpaca API key from environment variable
    api_secret = os.getenv('ALPACA_API_SECRET')  # Get Alpaca API secret from environment variable

    # Create clients for Alpaca's API
    market_data_client = StockHistoricalDataClient(api_key, api_secret)
    trading_client = TradingClient(api_key, api_secret, paper=True)

    schedule.every().day.at('07:00', 'America/Chicago').do(schedule_open, trading_client, market_data_client)
    schedule.every().day.at('09:00', 'America/Chicago').do(schedule_close, trading_client, market_data_client)

    while True:
        schedule.run_pending()
        time.sleep(1)