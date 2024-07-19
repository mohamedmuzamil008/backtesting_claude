import pandas as pd
import numpy as np
import os
import pandas_ta as ta
from pandas_ta.momentum import macd

# '''Custom Functions'''
def get_curtop_curbot(group):
    group = group.sort_index()  # Sort the group by datetime
    group['curtop'] = group['High'].cummax()
    group['curbot'] = group['Low'].cummin()
    return group

def calculate_first5(group):
    # Filter the group to include only the first 5 minutes (09:15 to 09:19)
    first5_group = group.between_time('09:15', '09:19')
    
    # Check if first5_group is not empty
    if not first5_group.empty:
        # Calculate first5_open, first5_high, first5_low, and first5_close
        first5_open = first5_group['Open'].iloc[0]
        first5_high = first5_group['High'].max()
        first5_low = first5_group['Low'].min()
        first5_close = first5_group['Close'].iloc[-1]
        
        first5_stats = pd.DataFrame({
            'first5_open': [first5_open],
            'first5_high': [first5_high],
            'first5_low': [first5_low],
            'first5_close': [first5_close]
        }, index=group.index[:1])  # Set the index to the first index of the group
        
        # Get the previous day's high value
        prevday_high = group['prevday_high'].iloc[0]
        
        # Calculate openconviction_5
        first5_stats['openconviction_5'] = calculate_openconviction(first5_stats, prevday_high)
        
        # Add the first5 statistics to the group DataFrame
        group = group.join(first5_stats)
        
        # Forward-fill the first5 statistics for the rest of the day
        group['first5_open'] = group['first5_open'].ffill()
        group['first5_high'] = group['first5_high'].ffill()
        group['first5_low'] = group['first5_low'].ffill()
        group['first5_close'] = group['first5_close'].ffill()
        group['openconviction_5'] = group['openconviction_5'].ffill()
    
    return group

def calculate_openconviction(first5_stats, prevday_high):
    openconviction = []
    for _, row in first5_stats.iterrows():
        buying_od_5 = (row['first5_open'] == row['first5_low']) and (row['first5_close'] >= ((row['first5_high'] - row['first5_low']) * 0.5) + row['first5_low']) and (row['first5_close'] > row['first5_open'])
        buying_otd_5 = (row['first5_open'] != row['first5_low']) and (row['first5_close'] >= (((row['first5_high'] - row['first5_low']) * 0.5) + row['first5_low'])) and (row['first5_close'] > row['first5_open']) and (pd.notna(prevday_high) and row['first5_low'] > prevday_high * 0.999)
        selling_orr_5 = (row['first5_close'] >= ((row['first5_high'] - row['first5_low']) * 0.5) + row['first5_low']) and (row['first5_open'] < row['first5_close'])
        
        if buying_od_5:
            openconviction.append(1)
        elif buying_otd_5:
            openconviction.append(2)
        elif selling_orr_5:
            openconviction.append(6)
        else:
            openconviction.append(7)
    
    return openconviction


#def calculate_initial_balance(group):
    # Filter the group to include only the first 60 minutes (09:15 to 10:14)
    ib_group = group.between_time('09:15', '10:14')
    
    # Calculate IB_high and IB_low
    ib_group['IB_high'] = ib_group['High'].cummax()
    ib_group['IB_low'] = ib_group['Low'].cummin()
    
    # Calculate IB_range
    ib_group['IB_range'] = ib_group['IB_high'] - ib_group['IB_low']
    
    # Get the previous day's ATR value
    if isinstance(daily_df.index, pd.DatetimeIndex):
        try:
            prev_day_atr = daily_df['ATR'].shift(1).loc[group.index[0]]
        except KeyError:
            # Handle missing data for the given date
            prev_day_atr = np.nan
    else:
        # Handle the case where the index is not a datetime index
        prev_day_atr = daily_df['ATR'].fillna(method='ffill').iloc[0]
    
    # Calculate ibrange/atr
    ib_group['ibrange/atr'] = ib_group['IB_range'] / prev_day_atr
    
    # Replicate the last values of IB_high, IB_low, IB_range, and ibrange/atr for the remaining times
    if not ib_group.empty:
        last_values = ib_group.loc[ib_group.index.time <= pd.Timestamp('10:14').time(), ['IB_high', 'IB_low', 'IB_range', 'ibrange/atr']].iloc[-1]
        ib_group.loc[ib_group.index.time > pd.Timestamp('10:14').time(), ['IB_high', 'IB_low', 'IB_range', 'ibrange/atr']] = last_values
    
    return ib_group

def calculate_initial_balance(group):
    # Keep all records from 09:15 to 15:29
    group = group.between_time('09:15', '15:29')
    
    # Filter the group to include only the first hour (09:15 to 10:15)
    first_hour_group = group.between_time('09:15', '10:14')

    # Reset the index of first_hour_group to ensure a unique index
    first_hour_group = first_hour_group.reset_index(drop=True)

    # Reset the index of the group DataFrame to ensure a unique index
    group = group.reset_index()

    ib_high_values = first_hour_group['High'].cummax()
    ib_low_values = first_hour_group['Low'].cummin()
    
    # Align the indices of the group DataFrame and the ib_high_values Series
    group['IB_high'] = group.join(ib_high_values.rename('IB_high'), how='left')['IB_high']
    group['IB_low'] = group.join(ib_low_values.rename('IB_low'), how='left')['IB_low']
    group['IB_range'] = group['IB_high'] - group['IB_low']
    group['ibrange/atr'] = group['IB_range'] / group['ATR']

    #print(group['IB_high'][0])
    group = group.set_index('datetime')   
    
    # Replicate the last values of IB_high, IB_low, IB_range, and ibrange/atr for the remaining times
    before_10_14 = group.loc[group.index.time <= pd.Timestamp('10:14').time(), ['IB_high', 'IB_low', 'IB_range', 'ibrange/atr']]
    if not before_10_14.empty:
        last_values = before_10_14.iloc[-1]
        #print(last_values)
        #group.loc[group.index.time > pd.Timestamp('10:14').time(), ['IB_high', 'IB_low', 'IB_range', 'ibrange/atr']] = last_values

        group.loc[group.index.time > pd.Timestamp('10:14').time(), 'IB_high'] = last_values['IB_high']
        group.loc[group.index.time > pd.Timestamp('10:14').time(), 'IB_low'] = last_values['IB_low']
        group.loc[group.index.time > pd.Timestamp('10:14').time(), 'IB_range'] = last_values['IB_range']
        group.loc[group.index.time > pd.Timestamp('10:14').time(), 'ibrange/atr'] = last_values['ibrange/atr']

    return group

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
df['date'] = df['Date'].astype(str)
df['date'] = pd.to_datetime(df['date'])

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

# Changing datetime as index to column
df = df.reset_index()

# Convert the datetime column to datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

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

# Create a MACD line and signal in the daily_df DataFrame
#daily_df['macd'], daily_df['macdsignal'], daily_df['macdhist'] = macd(close=daily_df['Close'], fast=12, slow=26, signal=9)

daily_df.ta.macd(fast=12, slow=26, signal=9, min_periods=None, append=True)

daily_df['uptrend'] = (daily_df['MACD_12_26_9'].shift(1) >= daily_df['MACDs_12_26_9'].shift(1)) & (daily_df['MACD_12_26_9'].shift(1) > daily_df['MACD_12_26_9'].shift(2))


# Create a ATR in df based on joins from the daily_df on datetime
df = df.merge(daily_df[['datetime', 'ATR', 'prevday_high', 'uptrend']], left_on='date', right_on='datetime', how='left')
df.drop(['datetime_y'], axis=1, inplace=True)
df.rename(columns={'datetime_x': 'datetime'}, inplace=True)

# Set the datetime column as the index
df = df.set_index('datetime')

# Calculate the CurrentTop (curtop) and CurrentBottom (curbot) for each timestamp within the same day
df = df.groupby(df.index.date).apply(get_curtop_curbot)

# Reset the index
df = df.reset_index(level=0, drop=True)

# Apply the calculate_initial_balance function to each day
df = df.groupby(df.index.date).apply(calculate_initial_balance)
print(df.shape)

# Reset the index
df = df.reset_index(level=0, drop=True)

# Apply the calculate_first5 function to each day
df = df.groupby(df.index.date).apply(calculate_first5)
df = df.reset_index(level=0, drop=True)
df = df.reset_index(drop=False)

df['datetime'] = pd.to_datetime(df['datetime'])
df['time'] = df['datetime'].dt.time

# Create a new column 'trade_active' and initialize it with 0
df['trade_active'] = 0

# Initializing the target and sl levels
df['buy_price'] = 99999
df['tp_price'] = 99999
df['trail_activated'] = 0
df['trail_activation_price'] = 99999
df['sl_price'] = 0
df['trail_sl_price'] = 0
df['buy'] = 0
df['sell'] = 0

# Exclude if its not uptrend
df = df[df['uptrend'] == True].reset_index(drop=True)
print(df.shape)

for i in range(1, len(df)):
    
    if i % 10000 == 0:
         print(i)

    # Buy signal calculation
    df.loc[i, 'buy'] = 1 \
    if df.loc[i, 'openconviction_5'] == 7 and \
    df.loc[i, 'ibrange/atr'] >= 1.25 and \
    df.loc[i, 'Low'] == df.loc[i, 'curbot'] and \
    df.loc[i, 'time'] > pd.Timestamp('09:19').time() and \
    df.loc[i, 'time'] <= pd.Timestamp('10:14').time() and \
    df.loc[i-1, 'trade_active'] == 0 \
    else 0

    # Trade active signal
    df.loc[i, 'trade_active'] = 1 if df.loc[i, 'buy'] == 1 else \
    0 if df.loc[i-1, 'sell'] == 1 else \
    1 if (df.loc[i, 'buy'] == 0) & (df.loc[i-1, 'trade_active'] == 1) else \
    0

    # Buy price calculation
    df.loc[i, 'buy_price'] = df.loc[i, 'curtop'] - 1.25 * df.loc[i, 'ATR'] if df.loc[i, 'buy'] == 1 else \
                         df.loc[i-1, 'buy_price'] if (df.loc[i, 'buy'] == 0) & (df.loc[i, 'trade_active'] == 1) else \
                         99999
    
    # Stop loss price calculation
    df.loc[i, 'sl_price'] = df.loc[i, 'curtop'] - 1.4 * df.loc[i, 'ATR'] if df.loc[i, 'buy'] == 1 else \
                        df.loc[i-1,'sl_price'] if (df.loc[i, 'buy'] == 0) & (df.loc[i, 'trade_active'] == 1) else \
                        0
    

    # Target price calculation
    df.loc[i, 'tp_price'] = df.loc[i, 'curbot'] + 0.75 * df.loc[i, 'ATR'] if df.loc[i, 'buy'] == 1 else \
                            df.loc[i-1, 'tp_price'] if (df.loc[i, 'buy'] == 0) & (df.loc[i, 'trade_active'] == 1) else \
                            99999
    

    # Trail activation status
    df.loc[i, 'trail_activated'] = 1 if df.loc[i, 'High'] >= df.loc[i-1, 'trail_activation_price'] and \
    df.loc[i, 'buy'] != 1 and \
    df.loc[i, 'trade_active'] == 1 else\
    0 if (df.loc[i-1, 'sell'] == 1) else\
    df.loc[i-1, 'trail_activated']

    # Trail activation price
    df.loc[i, 'trail_activation_price'] = df.loc[i, 'curtop'] - 0.95 * df.loc[i, 'ATR'] if df.loc[i, 'buy'] == 1 else \
    1.002 * df.loc[i-1, 'trail_activation_p rice'] if (df.loc[i, 'trade_active'] == 1) & (df.loc[i, 'trail_activated'] == 1) & (df.loc[i, 'High'] >= 1.002 * df.loc[i-1, 'trail_activation_price']) else \
    df.loc[i-1, 'trail_activation_price'] if (df.loc[i, 'trade_active'] == 1) else \
    99999
    
    # Trail stop loss price
    df.loc[i, 'trail_sl_price'] = df.loc[i, 'curtop'] - 1.05 * df.loc[i, 'ATR'] if df.loc[i, 'buy'] == 1 else \
    df.loc[i-1, 'trail_sl_price'] + 0.002 * df.loc[i-1, 'trail_activation_price'] if (df.loc[i, 'trade_active'] == 1) & (df.loc[i, 'trail_activated'] == 1) & (df.loc[i, 'High'] >= 1.002 * df.loc[i-1, 'trail_activation_price']) else \
    df.loc[i-1, 'trail_sl_price'] if (df.loc[i, 'trade_active'] == 1) else \
    0  

    # SL hit
    df.loc[i,'sl_hit'] = 1 if df.loc[i, 'trade_active'] == 1 and df.loc[i, 'buy'] != 1 and df.loc[i, 'Low'] <= df.loc[i,'sl_price'] else 0

    # TP hit
    df.loc[i, 'tp_hit'] = 1 if df.loc[i, 'trade_active'] == 1 and df.loc[i, 'buy'] != 1 and df.loc[i, 'High'] >= df.loc[i, 'tp_price'] else 0

    # Trail SL hit
    df.loc[i, 'trail_sl_hit'] = 1 if df.loc[i, 'trade_active'] == 1 and df.loc[i, 'buy'] != 1 and df.loc[i, 'trail_activated'] == 1 and df.loc[i, 'Low'] <= df.loc[i, 'trail_sl_price'] else 0

    # Day end reached
    df.loc[i, 'day_end_reached'] = 1 if df.loc[i, 'time'] >= pd.Timestamp('15:15').time() and df.loc[i, 'trade_active'] == 1 else 0

    # Sell signal calculation
    df.loc[i,'sell'] = 1 if df.loc[i,'sl_hit'] == 1 or df.loc[i, 'tp_hit'] == 1 or df.loc[i, 'trail_sl_hit'] == 1 or df.loc[i, 'day_end_reached'] == 1 else 0

    # Sell price calculation
    df.loc[i,'sell_price'] = df.loc[i,'sl_price'] if df.loc[i,'sl_hit'] == 1 and df.loc[i, 'sell'] == 1 else \
    df.loc[i, 'tp_price'] if df.loc[i, 'tp_hit'] == 1 and df.loc[i, 'sell'] == 1 else \
    df.loc[i, 'trail_sl_price'] if df.loc[i, 'trail_sl_hit'] == 1  and df.loc[i, 'sell'] == 1 else \
    df.loc[i, 'Close'] if df.loc[i, 'day_end_reached'] == 1 and df.loc[i, 'sell'] == 1 else 0

# Exclude records in df where the number of rows per day is less than 360
min_rows_per_day = 360
df = df.groupby(df.date).filter(lambda x: len(x) >= min_rows_per_day)

daily_df.to_csv('./data/daily_df.csv', index=False)
df.to_csv('./data/df.csv', index=False)

# Backtesting logic starts here...

df = pd.read_csv('./data/df.csv')

# Keep records only when the trade is active
df = df[(df['buy'] == 1) | (df['sell'] == 1)].reset_index(drop=True)

# Define the initial capital
df['cum_pl'] = 100000.0
df['pl'] = 0.0
df['num_trades'] = 0
df['shares_purchased'] = 0

# Margin availability
margin = 5

# Define the position size
position_sizing_pct = 0.3

for i in range(len(df)):
    
    try:
        # Number of shares purchased
        df.loc[i, 'shares_purchased'] = np.where((df.loc[i, 'buy'] == 1), round((min(df.loc[i, 'cum_pl'], 100000) * margin * position_sizing_pct) / df.loc[i, 'buy_price'], 0),
                df.loc[i-1, 'shares_purchased']
        )
    except KeyError:
        df.loc[i, 'shares_purchased'] = round((min(df.loc[i, 'cum_pl'], 100000) * margin * position_sizing_pct) / df.loc[i, 'buy_price'], 0)  # Set a default value of 0


    # Calculate the profit/loss per trade
    df.loc[i, 'pl'] = np.where((df.loc[i, 'sell'] == 1), (df.loc[i, 'sell_price'] - df.loc[i, 'buy_price']) * df.loc[i, 'shares_purchased'], 0)

    # Cumulative profit/loss calculation
    try:
        df.loc[i, 'cum_pl'] = np.where((df.loc[i, 'sell'] == 1), df.loc[i-1, 'cum_pl'] + df.loc[i, 'pl'], 
                                       np.where(i == 0, 100000, df[i-1, 'cum_pl'])
        )
    except KeyError:
        if i > 0:
            df.loc[i, 'cum_pl'] = np.where((df.loc[i, 'sell'] == 1), df.loc[i-1, 'cum_pl'] + df.loc[i, 'pl'], df.loc[i-1, 'cum_pl'])
        else:
            df.loc[i, 'cum_pl'] = 100000 

    # Calculate the number of trades
    try:
        df.loc[i, 'num_trades'] = np.where((df.loc[i,'sell'] == 1), df.loc[i-1, 'num_trades'] + 1,
                                                df.loc[i-1, 'num_trades']
        )
    except KeyError:
            df.loc[i, 'num_trades'] = 0  # Set a default value of 0
    
    # Calculate number of winning trades
    try: 
        df.loc[i, 'num_winning_trades'] = np.where((df.loc[i,'sell'] == 1) & (df.loc[i, 'pl'] > 0), df.loc[i-1, 'num_winning_trades'] + 1,
                                                df.loc[i-1, 'num_winning_trades']
        )
    except KeyError:
        df.loc[i, 'num_winning_trades'] = 0  # Set a default value of 0

    # Calculate number of losing trades
    try:
        df.loc[i, 'num_losing_trades'] = np.where((df.loc[i,'sell'] == 1) & (df.loc[i, 'pl'] <= 0), df.loc[i-1, 'num_losing_trades'] + 1,
                                                df.loc[i-1, 'num_losing_trades']
        )
    except KeyError:
        df.loc[i, 'num_losing_trades'] = 0  # Set a default value of 0

    # Calculate the Accuracy
    try:
        df.loc[i, 'accuracy'] = np.where((df.loc[i, 'sell'] == 1) & (df.loc[i, 'num_trades'] > 0), df.loc[i, 'num_winning_trades'] / df.loc[i, 'num_trades'],
                                        df.loc[i-1, 'accuracy']
        )

    except KeyError:
        df.loc[i, 'accuracy'] = 0  # Set a default value of 0

df.to_csv('./data/df_backtest.csv', index=False)









