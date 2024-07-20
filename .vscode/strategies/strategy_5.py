import pandas as pd
import numpy as np
import os

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
        1.002 * df.loc[i-1, 'trail_activation_price'] if (df.loc[i, 'trade_active'] == 1) & (df.loc[i, 'trail_activated'] == 1) & (df.loc[i, 'High'] >= 1.002 * df.loc[i-1, 'trail_activation_price']) else \
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
    
    # Save the processed data
    daily_processed_file = os.path.join(results_dir, f"df_daily_{file_name}.csv")
    daily_df.to_csv(daily_processed_file, index=False)
    processed_file = os.path.join(results_dir, f"df_{file_name}.csv")
    df.to_csv(processed_file, index=False)