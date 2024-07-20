import numpy as np
import numba as nb
import os

@nb.jit(nopython=True)
def calculate_signals(openconviction_5, ibrange_atr, low, curbot, time, curtop, atr, high, close):
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
                                0 if time[i] == 33300 else trades_taken_today[i-1])

        # Buy signal calculation
        buy[i] = 1 if (openconviction_5[i] == 7 and 
                       ibrange_atr[i] >= 1.25 and 
                       low[i] == curbot[i] and 
                       time[i] > 33540 and time[i] <= 36840 and 
                       trades_taken_today[i] == 0 and 
                       trade_active[i-1] == 0) else 0

        # Trade active signal
        trade_active[i] = 1 if buy[i] == 1 else (
                          0 if sell[i-1] == 1 else (
                          1 if (buy[i] == 0) and (trade_active[i-1] == 1) else 0))

        # Buy price calculation
        buy_price[i] = curtop[i] - 1.25 * atr[i] if buy[i] == 1 else (
                       buy_price[i-1] if (buy[i] == 0) and (trade_active[i] == 1) else 99999)

        # Stop loss price calculation
        sl_price[i] = curtop[i] - 1.4 * atr[i] if buy[i] == 1 else (
                            sl_price[i-1] if (buy[i] == 0) and (trade_active[i] == 1) else 0)
        
        # Target price calculation
        tp_price[i] = curbot[i] + 0.75 * atr[i] if buy[i] == 1 else (
                                tp_price[i-1] if (buy[i] == 0) and (trade_active[i] == 1) else 99999)
        
        # Trail activation status
        trail_activated[i] = 1 if (high[i] >= trail_activation_price[i-1] and buy[i] != 1 and trade_active[i] == 1) else (
                             0 if sell[i-1] == 1 else trail_activated[i-1])
        
        # Trail activation price
        trail_activation_price[i] = curtop[i] - 0.95 * atr[i] if buy[i] == 1 else (
        1.002 * trail_activation_price[i-1] if (trade_active[i] == 1) and (trail_activated[i] == 1) and (high[i] >= 1.002 * trail_activation_price[i-1]) else (
        trail_activation_price[i-1] if (trade_active[i] == 1) else 99999))

        # Trail stop loss price
        trail_sl_price[i] = curtop[i] - 1.05 * atr[i] if buy[i] == 1 else (
        trail_sl_price[i-1] + 0.002 * trail_activation_price[i-1] if (trade_active[i] == 1) and (trail_activated[i] == 1) and (high[i] >= 1.002 * trail_activation_price[i-1]) else (
        trail_sl_price[i-1] if (trade_active[i] == 1) else 0))

        # SL hit
        sl_hit[i] = 1 if trade_active[i] == 1 and buy[i] != 1 and low[i] <= sl_price[i] else 0

        # TP hit
        tp_hit[i] = 1 if trade_active[i] == 1 and buy[i] != 1 and high[i] >= tp_price[i] else 0

        # Trail SL hit
        trail_sl_hit[i] = 1 if trade_active[i] == 1 and buy[i] != 1 and trail_activated[i] == 1 and low[i] <= trail_sl_price[i] else 0

        # Day end reached
        day_end_reached[i] = 1 if time[i] >= 54900 and trade_active[i] == 1 else 0

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

def generate_strategy5_signals(df, daily_df, results_dir, file_name):

    # Exclude if its not uptrend
    df = df[df['uptrend'] == True].reset_index(drop=True)

    df['time_seconds'] = df['time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

    # Use the function
    trades_taken_today, buy, trade_active, buy_price, sl_price, tp_price, trail_activated, trail_activation_price, trail_sl_price, \
            sl_hit, tp_hit, trail_sl_hit, day_end_reached, sell, sell_price = calculate_signals(
        df['openconviction_5'].values, df['ibrange/atr'].values, df['Low'].values, df['curbot'].values,
        df['time_seconds'].values, df['curtop'].values, df['ATR'].values, df['High'].values, df['Close'].values
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