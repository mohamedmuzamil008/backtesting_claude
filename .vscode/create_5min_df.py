import pandas as pd
import pandas_ta as ta
from pandas_ta.momentum import macd
import math
import numpy as np
import os

def create_5min_frame(df, file_name):

    five_min_df = df.resample('5min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Reset the index to make 'datetime' a column
    five_min_df = five_min_df.reset_index()

    # Convert the datetime column to datetime format
    five_min_df['datetime'] = pd.to_datetime(five_min_df['datetime'])

    return five_min_df
