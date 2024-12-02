from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import requests
import os

def get_nasdaq_100_constituents():
    url = 'https://api.nasdaq.com/api/quote/list-type/nasdaq100'

    # Need browser-like headers to get a response. Nasdaq API blocks bots
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Upgrade-Insecure-Requests': '1',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Priority': 'u=0, i',
    }
    # Send GET request to the NASDAQ API and parse the JSON response
    response = requests.get(url, headers=headers).json()
    # Extract and return the list of symbols from the response
    return [row['symbol'] for row in response['data']['data']['rows']]

# Function to get the last year's stock bars for given symbols
def get_last_year_bars(symbols, client):
    now = datetime.now().date()
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=now - timedelta(days=365),
        end=now
    )
    return client.get_stock_bars(request_params).df

if __name__ == '__main__':
    api_key = os.getenv('ALPACA_API_KEY')  # Get Alpaca API key from environment variable
    api_secret = os.getenv('ALPACA_API_SECRET')  # Get Alpaca API secret from environment variable

    # Create a client for Alpaca's historical data API
    client = StockHistoricalDataClient(api_key, api_secret)
    # Get the list of NASDAQ-100 constituents and add 'SPY' to the list
    symbols = get_nasdaq_100_constituents() + ['SPY']
    # Get the last year's stock bars for the symbols
    bars = get_last_year_bars(symbols, client)

    # Save the bars data to a CSV file
    bars.to_csv('data.csv')
