import pandas as pd
import os
import pandas_ta as ta

'''Custom Functions'''
# def is_uptrend(df, short_window=20, long_window=40, adx_window=14, macd_fast=12, macd_slow=26, macd_signal=9):
#     """
#     Returns True if the stock is in an uptrending phase, False otherwise.
#     """
#     # Calculate the short and long SMAs
#     df['short_sma'] = ta.sma(df['Close'].dropna(), length=short_window)
#     df['long_sma'] = ta.sma(df['Close'].dropna(), length=long_window)
    
#     # Calculate the RSI
#     df['rsi'] = ta.rsi(df['Close'].dropna(), length=14)
    
#     # Calculate the ADX
#     adx = ta.adx(df['High'].dropna(), df['Low'].dropna(), df['Close'].dropna(), length=adx_window)
#     df['adx'] = adx['ADX_' + str(adx_window)]
#     df['adx_pos'] = adx['DMP_' + str(adx_window)]
#     df['adx_neg'] = adx['DMN_' + str(adx_window)]
    
#     # Calculate the MACD
#     macd = ta.macd(df['Close'].dropna(), fast=macd_fast, slow=macd_slow, signal=macd_signal)
#     df['macd'] = macd['MACD_' + str(macd_fast) + '_' + str(macd_slow) + '_' + str(macd_signal)]
#     df['macd_signal'] = macd['MACDs_' + str(macd_fast) + '_' + str(macd_slow) + '_' + str(macd_signal)]
    
#     # Check if the short SMA is above the long SMA, RSI is above 50, ADX_POS is greater than ADX_NEG, and MACD is above the signal line
#     uptrend = (df['short_sma'] > df['long_sma']) & (df['rsi'] > 50) & (df['adx_pos'] > df['adx_neg']) & (df['macd'] > df['macd_signal'])
    
#     return uptrend

# Get the current working directory
current_dir = os.getcwd()

# Construct the path to the data folder
data_folder = os.path.join(current_dir, './data')

# Adjust the file name if it's different
file_path = os.path.join(data_folder, 'ACC.csv')

# Read the CSV file
df = pd.read_csv(file_path)

# Convert Date and Time columns to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

# Combine Date and Time columns into a single datetime column
df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

# Set the datetime column as the index
df = df.set_index('datetime')

# Drop the original Date and Time columns
df = df.drop(['Date', 'Time'], axis=1)

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

# Set the datetime column as the index again
daily_df = daily_df.set_index('datetime')

# Remove the nan
daily_df = daily_df.dropna()

#print(daily_df.head(100))


# Apply the scanning logic to the daily DataFrame
# daily_df['is_uptrend'] = is_uptrend(daily_df)

# daily_df[daily_df['is_uptrend'] == True].tail(200).to_csv('./data/uptrend.csv')
# print(daily_df.head(50))

import pandas as pd
import numpy as np

def identify_uptrend_periods(df, col_name, window=15, tolerance=1e-6):
    """
    Identifies the uptrend periods in the DataFrame based on the closing prices
    of the past `window` days.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the 'Close' column.
        window (int): The number of days to consider for the uptrend detection.
        tolerance (float): The tolerance value to consider closing prices as constant.
        
    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'uptrend' column
                          indicating the uptrend periods.
    """
    df[col_name] = False
    
    for i in range(window, len(df)):
        # Get the closing prices for the past `window` days
        past_prices = df['Close'].iloc[i - window:i]

        # Check if the closing prices are constant or nearly constant
        if np.ptp(past_prices) <= tolerance:
            # If constant, mark it as a non-uptrend
            df.at[df.index[i], col_name] = False
            continue
        
        # Calculate the slope of the regression line
        x = np.arange(window)
        y = past_prices.values        
        
        try:
            fitted_eq = np.polyfit(x, y, 1)
            slope = fitted_eq[0]
        except ValueError:
            # Handle the case where there are not enough unique data points
            slope = 0

        # If the slope is positive, mark it as an uptrend
        if slope > 0:
            df.at[df.index[i], col_name] = True
    
    # Handle the first `window` days
    for i in range(window):
        past_prices = df['Close'].iloc[:i + 1]
        
        # Check if the closing prices are constant or nearly constant
        if np.ptp(past_prices) <= tolerance:
            # If constant, mark it as a non-uptrend
            df.at[df.index[i], col_name] = False
            continue
        
        x = np.arange(len(past_prices))
        y = past_prices.values
        
        try:            
            fitted_eq = np.polyfit(x, y, 1)
            slope = fitted_eq[0]
        except ValueError:
            slope = 0
        
        if slope > 0:
            df.at[df.index[i], col_name] = True
    
    return df


# Apply the uptrend identification function
daily_df = identify_uptrend_periods(daily_df, 'short_uptrend', window=3)
daily_df = identify_uptrend_periods(daily_df, 'medium_uptrend', window=5)
daily_df = identify_uptrend_periods(daily_df, 'long_uptrend', window=10)


#daily_df[daily_df['uptrend'] == True].to_csv('./data/uptrend.csv')

daily_df.to_csv('./data/daily_df.csv')






