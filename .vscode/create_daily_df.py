import pandas as pd
import pandas_ta as ta
from pandas_ta.momentum import macd
import math
import numpy as np
import os

def create_daily_frame(df, file_name):

    data_dir = os.path.join(os.getcwd(), 'data')
    
    df_value_area_levels = pd.read_csv(data_dir+'/value_area_levels_for_all_stocks_new.csv')
    df_value_area_levels = df_value_area_levels[df_value_area_levels['Stock_Name'] == file_name].reset_index(drop=True)
    df_value_area_levels.rename(columns={'VAH Prev' : 'VAH', 'POC Prev' : 'POC', 'VAL Prev':'VAL'}, inplace=True)
    df_value_area_levels['Date'] = pd.to_datetime(df_value_area_levels['Date'], format="%d-%m-%Y")
    
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

    lookup_days = 10

    # Calculate the mean of the closing price based on the previous 10 values
    daily_df['Close_avg'] = daily_df['Close'].shift(1).rolling(lookup_days).mean()

    # Volatility
    daily_df['Close_prev'] = daily_df['Close'].shift(1)
    daily_df['daily_return'] = np.log(daily_df['Close'] / daily_df['Close_prev'])
    daily_df['volatility'] = daily_df['daily_return'].shift(1).rolling(lookup_days).std()

    # Calculate the SMA of the Volume of last 10 days
    daily_df['Volume_avg'] = daily_df['Volume'].shift(1).rolling(lookup_days).mean()

    # Calculate the range - High - Low for each day
    daily_df['range'] = daily_df['High'] - daily_df['Low']

    # Calculate the Standard deviation of the range based on the previous 10 values
    daily_df['range_sd'] = daily_df['range'].shift(1).rolling(lookup_days).std()

    # Calculate the ATR for the entire DataFrame
    daily_df['ATR'] = ta.atr(high=daily_df['High'], low=daily_df['Low'], close=daily_df['Close'], length=lookup_days)

    # Calculate the Standard deviation / ATR
    daily_df['SD/ATR'] = daily_df['range_sd'] / daily_df['ATR']

    # Calculate the multiplier
    daily_df['multiplier'] = daily_df['SD/ATR'] * daily_df['volatility']

    # Calculate the ATR/Close
    daily_df['ATR/Close'] = daily_df['ATR'] / daily_df['Close_avg']

    # Create a new column 'prevday_high' in the daily_df DataFrame
    daily_df['prevday_high'] = daily_df['High'].shift(1)   

    # Create a new column 'prevday_high' in the daily_df DataFrame
    daily_df['prevday_low'] = daily_df['Low'].shift(1)

    # ADX
    daily_df.ta.adx(length=lookup_days, scalar=100, drift=1, append=True)
    daily_df['ADX_uptrend'] = (daily_df['ADX_'+str(lookup_days)].shift(1) > 25) & (daily_df['DMP_'+str(lookup_days)].shift(1) > daily_df['DMN_'+str(lookup_days)].shift(1)) & (daily_df['DMP_'+str(lookup_days)].shift(1) > 25)

    # RSI
    daily_df.ta.rsi(close='Close', length=lookup_days, scalar=100, drift=1, append=True)
    daily_df['RSI_uptrend'] = (daily_df['RSI_'+str(lookup_days)].shift(1) > 30) & (daily_df['RSI_'+str(lookup_days)].shift(1) < 60)

    # MACD
    daily_df.ta.macd(fast=12, slow=26, signal=9, min_periods=None, append=True)
    daily_df['MACD_uptrend'] = (daily_df['MACD_12_26_9'].shift(1) >= daily_df['MACDs_12_26_9'].shift(1)) & (daily_df['MACD_12_26_9'].shift(1) > daily_df['MACD_12_26_9'].shift(2)) & (daily_df['MACD_12_26_9'].shift(1) > 0)

    # MA
    daily_df.ta.ema(close='Close', length=5, append=True)
    daily_df.ta.ema(close='Close', length=20, append=True)
    daily_df.ta.ema(close='Close', length=40, append=True)
    #daily_df['EMA_uptrend'] = (daily_df['EMA_5'].shift(1) > daily_df['EMA_20'].shift(1)) & (daily_df['EMA_20'].shift(1) > daily_df['EMA_40'].shift(1))
    daily_df['EMA_uptrend'] = (daily_df['Close'].shift(1) > daily_df['EMA_20'].shift(1))


    # Count the number of uptrend signals(from MACD, ADX and EMA)
    daily_df['uptrend_signals'] = daily_df['MACD_uptrend'].apply(int) + daily_df['ADX_uptrend'].apply(int)# + daily_df['EMA_uptrend'].apply(int)
    #daily_df['uptrend_signals'] = daily_df['EMA_uptrend'].apply(int)

    #daily_df['uptrend'] = (daily_df['MACD_uptrend'] | daily_df['ADX_uptrend'] | daily_df['EMA_uptrend']) & (daily_df['Volume_avg'] > 1000000)
    daily_df['uptrend'] = (daily_df['uptrend_signals'] >= 0) & (daily_df['Volume_avg'] > 500000)
    daily_df['downtrend'] = (daily_df['uptrend_signals'] == 0) & (daily_df['Volume_avg'] > 1000000)
    
    daily_df = pd.merge(daily_df, df_value_area_levels[['Date', 'VAH', 'POC', 'VAL']], left_on='datetime', right_on='Date', how='left')
    daily_df.drop(['Date'], axis=1, inplace=True)
    daily_df['VAH_prev'] = daily_df['VAH'].shift(1)
    daily_df['POC_prev'] = daily_df['POC'].shift(1)
    daily_df['VAL_prev'] = daily_df['VAL'].shift(1)

    daily_df['High_prev'] = daily_df['High'].shift(1)
    daily_df['Low_prev'] = daily_df['Low'].shift(1)

    # Within PDVA - 1, Above PDVA below PDH - 2, Above PDH - 3, Below PDVA above PDL - 4, Below PDL - 5  
    daily_df['Open_Location'] = np.where((daily_df['Open'] <= daily_df['VAH_prev']) & (daily_df['Open'] >= daily_df['VAL_prev']), 1, 
                                            np.where((daily_df['Open'] > daily_df['VAH_prev']) & (daily_df['Open'] <= daily_df['High_prev']), 2, 
                                            np.where((daily_df['Open'] > daily_df['High_prev']), 3, 
                                            np.where((daily_df['Open'] < daily_df['VAL_prev']) & (daily_df['Open'] >= daily_df['Low_prev']), 4, 
                                            np.where((daily_df['Open'] < daily_df['Low_prev']), 5, 6)))))

    return daily_df
