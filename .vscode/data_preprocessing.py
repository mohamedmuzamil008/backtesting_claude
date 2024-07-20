import pandas as pd

def data_preprocessing(df):

    # Convert Date and Time columns to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

    # Combine Date and Time columns into a single datetime column
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df['date'] = df['Date'].astype(str)
    df['date'] = pd.to_datetime(df['date'])

    # Set the datetime column as the index
    df = df.set_index('datetime')

    # Drop the original Date and Time columns
    df = df.drop(['Date', 'Time'], axis=1)

    return df