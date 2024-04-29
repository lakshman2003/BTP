import os
import pandas as pd
import numpy as np
import yfinance as yf

def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, min_periods=0, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['Close'].ewm(span=12, min_periods=0, adjust=False).mean() - df['Close'].ewm(span=26, min_periods=0, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(
        span=9, min_periods=0, adjust=False).mean()
    df['std_20'] = df['Close'].rolling(window=20).std()
    df['Bollinger_Band_Upper'] = df['SMA_20'] + (df['std_20'] * 2)
    df['Bollinger_Band_Lower'] = df['SMA_20'] - (df['std_20'] * 2)
    high14 = df['High'].rolling(window=14).max()
    low14 = df['Low'].rolling(window=14).min()
    df['%K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
    df['%D'] = df['%K'].rolling(window=3).mean()
    df['TR'] = np.maximum.reduce([df['High'] - df['Low'], abs(
        df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())])
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['Daily returns']= df['Close'].pct_change()
    df['Weekly returns'] = df['Close'].pct_change(periods = 5)  
    df['Next Week Close'] = df['Close'].shift(-5)
    df['Label'] = np.where(df['Next Week Close'] > df['Close'], 1, 0)
    df = df.drop(['Next Week Close'],axis= 1)
    return df


tickers = pd.read_csv("ind_nifty50list.csv", index_col=0)

start = '2012-01-01'
end = '2024-03-31'

merged_data = pd.DataFrame()  # Initialize an empty DataFrame

print("PARSING STARTED! ")
for count, symbol in enumerate(tickers.Symbol):
    symbol_with_exchange = symbol + '.NS'
    stock_data = download_stock_data(symbol_with_exchange, start, end)
    stock_data_with_indicators = calculate_technical_indicators(stock_data)
    stock_data_with_indicators['Ticker'] = symbol 
    if merged_data.empty:
        merged_data = stock_data_with_indicators
    else:
        merged_data = pd.concat([merged_data, stock_data_with_indicators])  # Concatenate the data frames
    print("Done ", count, " ", symbol)

merged_data.sort_values(by='Date', inplace=True)

merged_data.to_csv("Dataset\merged_stock_data.csv") 
print("PARSED!")

nifty_50_ticker = '^NSEI'
stock_data = download_stock_data(nifty_50_ticker, start, end)
stock_data_with_indicators = calculate_technical_indicators(stock_data)
stock_data_with_indicators.to_csv("Dataset/nifty50index.csv")


