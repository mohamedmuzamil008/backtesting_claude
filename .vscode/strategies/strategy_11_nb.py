import numpy as np
import numba as nb
import os
from flask import current_app

@nb.jit(nopython=True)
def calculate_signals(openconviction_5, openconviction_15, first1_open, first1_close, sd_atr, volatility, multiplier, low, curbot, time, curtop, atr, high, close, open, entry_param, sl_param, target_param, trail_activation_param, trail_sl_param, uptrend, downtrend, vah, poc, val, openlocation, ib_high, ib_low, ib_range, ibrange_atr):
    n = len(openconviction_5)
    buy = np.zeros(n, dtype=np.int32)
    sell = np.zeros(n, dtype=np.int32)
    trade_active = np.zeros(n, dtype=np.int32)
    buy_price = np.full(n, 99999.0)
    sl_price = np.zeros(n, dtype=np.float64)
    tp_price = np.full(n, 99999.0)
    trail_activated = np.zeros(n, dtype=np.int32)
    trail_activation_price = np.full(n, 99999.0)
    trail_sl_price = np.zeros(n, dtype=np.float64)
    sl_hit = np.zeros(n, dtype=np.int32)
    tp_hit = np.zeros(n, dtype=np.int32)
    trail_sl_hit = np.zeros(n, dtype=np.int32)
    day_end_reached = np.zeros(n, dtype=np.int32)
    sell_price = np.zeros(n, dtype=np.float64)  
    trades_taken_today = np.zeros(n, dtype=np.int32)  
    
    for i in range(1, n):

        # Trades taken for the day
        trades_taken_today[i] = 1 if buy[i-1] == 1 else(
                                0 if time[i] == 555 else trades_taken_today[i-1])

        # Buy signal calculation
        buy[i] = 1 if ((openlocation[i] == 2 or openlocation[i] == 3) and
                       (openconviction_15[i] == 1 or openconviction_15[i] == 2) and                        
                       uptrend[i] == True and
                       ibrange_atr[i] <= 0.5 and 
                       close[i] > ib_high[i] and
                       #((ibrange_atr[i] <= 0.5 and close[i] < (0.2 * ib_range[i]) + ib_low[i]) or  (ibrange_atr[i] > 0.5 and close[i] < (0.3 * ib_range[i]) + ib_low[i]) ) and
                       #multiplier[i] >= 0.0022 and multiplier[i] <= 0.0070 and
                       #volatility[i] >= 0.012 and 
                       #(time[i] + 1) % 5 == 0 and  
                       time[i] > 614 and time[i] <= 855 and  #42300
                       #time[i] > 36900 and time[i] <= 51300 and  #42300
                       trades_taken_today[i] == 0 and 
                       trade_active[i-1] == 0) else 0

        # Trade active signal
        trade_active[i] = 1 if buy[i] == 1 else (
                          0 if sell[i-1] == 1 else (
                          1 if (buy[i] == 0) and (trade_active[i-1] == 1) else 0))

        
        buy_price[i] = entry_param[i] if buy[i] == 1 else (
                      buy_price[i-1] if (buy[i] == 0) and (trade_active[i] == 1) else 99999)
        

        # Stop loss price calculation        
        sl_price[i] = sl_param[i] if buy[i] == 1 else (
                            sl_price[i-1] if (buy[i] == 0) and (trade_active[i] == 1) else 0)
        

        # Target price calculation
        tp_price[i] = target_param[i] if buy[i] == 1 else (
                                tp_price[i-1] if (buy[i] == 0) and (trade_active[i] == 1) else 99999)
        
        # Trail activation status
        trail_activated[i] = 1 if (high[i] >= trail_activation_price[i-1] and buy[i] != 1 and trade_active[i] == 1) else (
                             0 if sell[i-1] == 1 else trail_activated[i-1])
        
        # Trail activation price
        trail_activation_price[i] = trail_activation_param[i] if buy[i] == 1 else (
        1.002 * trail_activation_price[i-1] if (trade_active[i] == 1) and (trail_activated[i] == 1) and (high[i] >= 1.002 * trail_activation_price[i-1]) else (
        trail_activation_price[i-1] if (trade_active[i] == 1) else 99999))
        

        # Trail stop loss price
        trail_sl_price[i] = trail_sl_param[i] if buy[i] == 1 else (
        trail_sl_price[i-1] + 0.002 * trail_activation_price[i-1] if (trade_active[i] == 1) and (trail_activated[i] == 1) and (high[i] >= 1.002 * trail_activation_price[i-1]) else (
        trail_sl_price[i-1] if (trade_active[i] == 1) else 0))
        
        # SL hit
        sl_hit[i] = 1 if trade_active[i] == 1 and buy[i] != 1 and low[i] <= sl_price[i] else 0

        # TP hit
        tp_hit[i] = 1 if trade_active[i] == 1 and buy[i] != 1 and high[i] >= tp_price[i] else 0

        # Trail SL hit
        trail_sl_hit[i] = 1 if trade_active[i] == 1 and buy[i] != 1 and trail_activated[i] == 1 and low[i] <= trail_sl_price[i] else 0

        # Day end reached
        day_end_reached[i] = 1 if time[i] >= 915 and trade_active[i] == 1 else 0

        # Sell signal calculation        
        sell[i] = 1 if sl_hit[i] == 1 or tp_hit[i] == 1 or trail_sl_hit[i] == 1 or day_end_reached[i] == 1 else 0

        # Sell price calculation
        sell_price[i] = sl_price[i] if sl_hit[i] == 1 and sell[i] == 1 else (
            tp_price[i] if tp_hit[i] == 1 and sell[i] == 1 else (
            trail_sl_price[i] if trail_sl_hit[i] == 1 and sell[i] == 1 else (
            close[i] if day_end_reached[i] == 1 and sell[i] == 1 else 0
        )))

    return trades_taken_today, buy, trade_active, buy_price, sl_price, tp_price, trail_activated, trail_activation_price, trail_sl_price, \
            sl_hit, tp_hit, trail_sl_hit, day_end_reached, sell, sell_price

def generate_strategy11_signals(df, daily_df, results_dir, file_name):

    # Exclude if its not uptrend
    #df = df[df['uptrend'] == True].reset_index(drop=True)

    df['time_seconds'] = df['time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    df['time_mins'] = df['time'].apply(lambda x: x.hour * 60 + x.minute)

    df['entry_param'] = np.where(df['uptrend'] == True, df['Close'], df['ATR'] + ((1.8 + df['SD/ATR'])  * df['range_sd']))
    df['sl_param'] = np.where(df['uptrend'] == True, np.where(df['ibrange/atr'] <= 0.5, df['IB_low']  + 0.4 * df['IB_range'], df['IB_low'] - 0.25 * df['IB_range']), df['entry_param'] + ((0.4 + df['SD/ATR'])  * df['range_sd']))
    df['target_param'] = np.where(df['uptrend'] == True, np.where(df['ibrange/atr'] <= 0.5, df['IB_high'] + 0.5 * df['IB_range'], df['IB_low'] + 0.65 * df['IB_range']), df['entry_param'] - ((0.7 + df['SD/ATR'])  * df['range_sd']))
    df['trail_activation_param'] = np.where(df['uptrend'] == True, np.where(df['ibrange/atr'] <= 0.5, df['IB_high'] + 0.5 * df['IB_range'], df['IB_low'] + 0.65 * df['IB_range']), df['entry_param'] - ((0.5 + df['SD/ATR']) * df['range_sd']))
    df['trail_sl_param'] = np.where(df['uptrend'] == True, np.where(df['ibrange/atr'] <= 0.5, df['IB_high'] + 0.4 * df['IB_range'], df['IB_low'] + 0.45 * df['IB_range']), df['entry_param'] - ((0.3 + df['SD/ATR']) * df['range_sd']))
    

    # Use the function
    trades_taken_today, buy, trade_active, buy_price, sl_price, tp_price, trail_activated, trail_activation_price, trail_sl_price, \
            sl_hit, tp_hit, trail_sl_hit, day_end_reached, sell, sell_price = calculate_signals(
        df['openconviction_5'].values, df['openconviction_15'].values, df['first1_open'].values, df['first1_close'].values, df['SD/ATR'].values, df['volatility'].values, df['multiplier'].values, df['Low'].values, df['curbot'].values,
        df['time_mins'].values, df['curtop'].values, df['ATR'].values, df['High'].values, df['Close'].values, df['Open'].values,
        df['entry_param'].values, df['sl_param'].values, df['target_param'].values, df['trail_activation_param'].values,
        df['trail_sl_param'].values, df['uptrend'].values, df['downtrend'].values, df['VAH_prev'].values, df['POC_prev'].values, df['VAL_prev'].values, df['Open_Location'].values,
        df['IB_high'].values, df['IB_low'].values, df['IB_range'].values, df['ibrange/atr'].values
    )

    # Assign results back to DataFrame
    df['trades_taken_today'] = trades_taken_today
    df['buy'] = buy
    df['trade_active'] = trade_active
    df['buy_price'] = buy_price
    df['sl_price'] = sl_price
    df['tp_price'] = tp_price
    df['trail_activated'] = trail_activated
    df['trail_activation_price'] = trail_activation_price
    df['trail_sl_price'] = trail_sl_price
    df['sl_hit'] = sl_hit
    df['tp_hit'] = tp_hit
    df['trail_sl_hit'] = trail_sl_hit
    df['day_end_reached'] = day_end_reached
    df['sell'] = sell
    df['sell_price'] = sell_price

    # Exclude records in df where the number of rows per day is less than 360
    min_rows_per_day = 360
    df = df.groupby(df.date).filter(lambda x: len(x) >= min_rows_per_day)
    
    # Save the processed data
    daily_processed_file = os.path.join(results_dir, f"df_daily_{file_name}.csv")
    daily_df.to_csv(daily_processed_file, index=False)
    processed_file = os.path.join(results_dir, f"df_{file_name}.csv")
    df.to_csv(processed_file, index=False)

    current_app.result_df = df