import multiprocessing as mp
import numpy as np
import pandas as pd
import os

def process_chunk(chunk):
    
    for i in range(0, len(chunk)):
        
        # Buy signal calculation
        try:        
            chunk.loc[i, 'buy'] = 1 \
            if chunk.loc[i, 'openconviction_5'] == 7 and \
            chunk.loc[i, 'ibrange/atr'] >= 1.25 and \
            chunk.loc[i, 'Low'] == chunk.loc[i, 'curbot'] and \
            chunk.loc[i, 'time'] > pd.Timestamp('09:19').time() and \
            chunk.loc[i, 'time'] <= pd.Timestamp('10:14').time() and \
            chunk.loc[i-1, 'trade_active'] == 0 \
            else 0
        except KeyError:
            chunk.loc[i, 'buy'] = 0

        # Trade active signal
        try:
            chunk.loc[i, 'trade_active'] = 1 if chunk.loc[i, 'buy'] == 1 else \
            0 if chunk.loc[i-1, 'sell'] == 1 else \
            1 if (chunk.loc[i, 'buy'] == 0) & (chunk.loc[i-1, 'trade_active'] == 1) else \
            0
        except KeyError:
            chunk.loc[i, 'trade_active'] = 0

        # Buy price calculation
        try:
            chunk.loc[i, 'buy_price'] = chunk.loc[i, 'curtop'] - 1.25 * chunk.loc[i, 'ATR'] if chunk.loc[i, 'buy'] == 1 else \
                                chunk.loc[i-1, 'buy_price'] if (chunk.loc[i, 'buy'] == 0) & (chunk.loc[i, 'trade_active'] == 1) else \
                                99999
        except KeyError:
            chunk.loc[i, 'buy_price'] = 99999
            
        # Stop loss price calculation
        try:
            chunk.loc[i, 'sl_price'] = chunk.loc[i, 'curtop'] - 1.4 * chunk.loc[i, 'ATR'] if chunk.loc[i, 'buy'] == 1 else \
                                chunk.loc[i-1,'sl_price'] if (chunk.loc[i, 'buy'] == 0) & (chunk.loc[i, 'trade_active'] == 1) else \
                                0     
        except KeyError:
            chunk.loc[i,'sl_price'] = 0

        # Target price calculation
        try:
            chunk.loc[i, 'tp_price'] = chunk.loc[i, 'curbot'] + 0.75 * chunk.loc[i, 'ATR'] if chunk.loc[i, 'buy'] == 1 else \
                                    chunk.loc[i-1, 'tp_price'] if (chunk.loc[i, 'buy'] == 0) & (chunk.loc[i, 'trade_active'] == 1) else \
                                    99999   
        except KeyError:
            chunk.loc[i, 'tp_price'] = 99999

        # Trail activation status
        try:
            chunk.loc[i, 'trail_activated'] = 1 if chunk.loc[i, 'High'] >= chunk.loc[i-1, 'trail_activation_price'] and \
            chunk.loc[i, 'buy'] != 1 and \
            chunk.loc[i, 'trade_active'] == 1 else\
            0 if (chunk.loc[i-1, 'sell'] == 1) else\
            chunk.loc[i-1, 'trail_activated']
        except KeyError:
            chunk.loc[i, 'trail_activated'] = 0

        # Trail activation price
        try:
            chunk.loc[i, 'trail_activation_price'] = chunk.loc[i, 'curtop'] - 0.95 * chunk.loc[i, 'ATR'] if chunk.loc[i, 'buy'] == 1 else \
            1.002 * chunk.loc[i-1, 'trail_activation_p rice'] if (chunk.loc[i, 'trade_active'] == 1) & (chunk.loc[i, 'trail_activated'] == 1) & (chunk.loc[i, 'High'] >= 1.002 * chunk.loc[i-1, 'trail_activation_price']) else \
            chunk.loc[i-1, 'trail_activation_price'] if (chunk.loc[i, 'trade_active'] == 1) else \
            99999
        except KeyError:
            chunk.loc[i, 'trail_activation_price'] = 99999
        
        # Trail stop loss price
        try:
            chunk.loc[i, 'trail_sl_price'] = chunk.loc[i, 'curtop'] - 1.05 * chunk.loc[i, 'ATR'] if chunk.loc[i, 'buy'] == 1 else \
            chunk.loc[i-1, 'trail_sl_price'] + 0.002 * chunk.loc[i-1, 'trail_activation_price'] if (chunk.loc[i, 'trade_active'] == 1) & (chunk.loc[i, 'trail_activated'] == 1) & (chunk.loc[i, 'High'] >= 1.002 * chunk.loc[i-1, 'trail_activation_price']) else \
            chunk.loc[i-1, 'trail_sl_price'] if (chunk.loc[i, 'trade_active'] == 1) else \
            0  
        except KeyError:
            chunk.loc[i, 'trail_sl_price'] = 0

        # SL hit
        chunk.loc[i,'sl_hit'] = 1 if chunk.loc[i, 'trade_active'] == 1 and chunk.loc[i, 'buy'] != 1 and chunk.loc[i, 'Low'] <= chunk.loc[i,'sl_price'] else 0

        # TP hit
        chunk.loc[i, 'tp_hit'] = 1 if chunk.loc[i, 'trade_active'] == 1 and chunk.loc[i, 'buy'] != 1 and chunk.loc[i, 'High'] >= chunk.loc[i, 'tp_price'] else 0

        # Trail SL hit
        chunk.loc[i, 'trail_sl_hit'] = 1 if chunk.loc[i, 'trade_active'] == 1 and chunk.loc[i, 'buy'] != 1 and chunk.loc[i, 'trail_activated'] == 1 and chunk.loc[i, 'Low'] <= chunk.loc[i, 'trail_sl_price'] else 0

        # Day end reached
        #chunk.loc[i, 'day_end_reached'] = 1 if chunk.loc[i, 'time'] >= pd.Timestamp('15:15').time() and chunk.loc[i, 'trade_active'] == 1 else 0
        chunk.loc[i, 'day_end_reached'] = 0

        # Sell signal calculation
        chunk.loc[i,'sell'] = 1 if chunk.loc[i,'sl_hit'] == 1 or chunk.loc[i, 'tp_hit'] == 1 or chunk.loc[i, 'trail_sl_hit'] == 1 or chunk.loc[i, 'day_end_reached'] == 1 else 0

        # Sell price calculation
        chunk.loc[i,'sell_price'] = chunk.loc[i,'sl_price'] if chunk.loc[i,'sl_hit'] == 1 and chunk.loc[i, 'sell'] == 1 else \
        chunk.loc[i, 'tp_price'] if chunk.loc[i, 'tp_hit'] == 1 and chunk.loc[i, 'sell'] == 1 else \
        chunk.loc[i, 'trail_sl_price'] if chunk.loc[i, 'trail_sl_hit'] == 1  and chunk.loc[i, 'sell'] == 1 else \
        chunk.loc[i, 'Close'] if chunk.loc[i, 'day_end_reached'] == 1 and chunk.loc[i, 'sell'] == 1 else 0

    return chunk

def parallel_processing(df):
    num_cores = mp.cpu_count()
    chunk_size = len(df) // num_cores
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

    with mp.Pool(num_cores) as pool:
        results = pool.map(process_chunk, chunks)

    return pd.concat(results)

def generate_strategy5_signals(df, daily_df, results_dir, file_name):

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

    df = parallel_processing(df)       
        
    # Exclude records in df where the number of rows per day is less than 360
    min_rows_per_day = 360
    df = df.groupby(df.date).filter(lambda x: len(x) >= min_rows_per_day)
    
    # Save the processed data
    daily_processed_file = os.path.join(results_dir, f"df_daily_{file_name}.csv")
    daily_df.to_csv(daily_processed_file, index=False)
    processed_file = os.path.join(results_dir, f"df_{file_name}.csv")
    df.to_csv(processed_file, index=False)