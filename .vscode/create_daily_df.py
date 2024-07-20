import pandas as pd
import pandas_ta as ta
from pandas_ta.momentum import macd

def create_daily_frame(df):
    # Resample the data to daily timeframe
    daily_df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Reset the index to make 'datetime' a column
    daily_df = daily_df.reset_index()

    # Convert the datetime column to datetime format
    daily_df['datetime'] = pd.to_datetime(daily_df['datetime'])

    # Remove the nan
    daily_df = daily_df.dropna()

    # Calculate the ATR for the entire DataFrame
    daily_df['ATR'] = ta.atr(high=daily_df['High'], low=daily_df['Low'], close=daily_df['Close'], length=10)

    # Create a new column 'prevday_high' in the daily_df DataFrame
    daily_df['prevday_high'] = daily_df['High'].shift(1)

    daily_df.ta.macd(fast=12, slow=26, signal=9, min_periods=None, append=True)

    daily_df['uptrend'] = (daily_df['MACD_12_26_9'].shift(1) >= daily_df['MACDs_12_26_9'].shift(1)) & (daily_df['MACD_12_26_9'].shift(1) > daily_df['MACD_12_26_9'].shift(2))

    return daily_df
