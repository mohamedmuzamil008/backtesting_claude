import pandas as pd
import numpy as np
import os
import pandas_ta as ta

'''Custom Functions'''
def get_curtop_curbot(group):
    group = group.sort_index()  # Sort the group by datetime
    group['curtop'] = group['High'].cummax()
    group['curbot'] = group['Low'].cummin()
    return group

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

# Add the previous day's high value to the df DataFrame
# df['prevday_high'] = daily_df['High'].shift(1)

# Changing datetime as index to column
df = df.reset_index()

# Convert the datetime column to datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Reset the index to make 'datetime' a column
daily_df = daily_df.reset_index()

# Convert the datetime column to datetime format
daily_df['datetime'] = pd.to_datetime(daily_df['datetime'])

# Set the datetime column as the index again
# daily_df = daily_df.set_index('datetime')

# Remove the nan
daily_df = daily_df.dropna()

# Calculate the ATR for the entire DataFrame
daily_df['ATR'] = ta.atr(high=daily_df['High'], low=daily_df['Low'], close=daily_df['Close'], length=10)

# Create a new column 'prevday_high' in the daily_df DataFrame
daily_df['prevday_high'] = daily_df['High'].shift(1)

# Create a ATR in df based on joins from the daily_df on datetime
df = df.merge(daily_df[['datetime', 'ATR', 'prevday_high']], left_on='date', right_on='datetime', how='left')
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

# Apply the calculate_first5 function to each day
df = df.groupby(df.index.date).apply(calculate_first5)

daily_df.to_csv('./data/daily_df.csv')
df.to_csv('./data/df.csv')

#print(df['IB_high'].tail())
#df[['IB_high']].to_csv('./data/IB_high.csv')








