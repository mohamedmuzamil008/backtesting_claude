import pandas as pd
import numpy as np
import os

def calculate_pl(df, initial_capital, margin=5, position_sizing_pct=30, commission=100, results_dir=None, file_name=None):

    # Keep records only when the trade is active
    df = df[(df['buy'] == 1) | (df['sell'] == 1)].reset_index(drop=True)

    df['cum_pl'] = initial_capital
    df['pl'] = 0.0
    df['num_trades'] = 0
    df['shares_purchased'] = 0      

    for i in range(len(df)):        
        try:
            # Number of shares purchased
            df.loc[i, 'shares_purchased'] = np.where((df.loc[i, 'buy'] == 1), round((min(df.loc[i, 'cum_pl'], initial_capital) * margin * position_sizing_pct) / df.loc[i, 'buy_price'], 0),
                    df.loc[i-1, 'shares_purchased']
            )
        except KeyError:
            df.loc[i, 'shares_purchased'] = round((min(df.loc[i, 'cum_pl'], initial_capital) * margin * position_sizing_pct) / df.loc[i, 'buy_price'], 0)  # Set a default value of 0


        # Calculate the profit/loss per trade
        df.loc[i, 'pl'] = np.where((df.loc[i, 'sell'] == 1), ((df.loc[i, 'sell_price'] - df.loc[i, 'buy_price']) * df.loc[i, 'shares_purchased']) - commission, 0)

        # Cumulative profit/loss calculation
        try:
            df.loc[i, 'cum_pl'] = np.where((df.loc[i, 'sell'] == 1), df.loc[i-1, 'cum_pl'] + df.loc[i, 'pl'], 
                                        np.where(i == 0, initial_capital, df[i-1, 'cum_pl'])
            )
        except KeyError:
            if i > 0:
                df.loc[i, 'cum_pl'] = np.where((df.loc[i, 'sell'] == 1), df.loc[i-1, 'cum_pl'] + df.loc[i, 'pl'], df.loc[i-1, 'cum_pl'])
            else:
                df.loc[i, 'cum_pl'] = initial_capital 
        
        # Calculate the running maximum
        try:
            df.loc[i, 'running_max'] = np.where((df.loc[i, 'sell'] == 1) & (df.loc[i, 'cum_pl'] > df.loc[i-1, 'running_max']), df.loc[i, 'cum_pl'], df.loc[i-1, 'running_max'])
        except KeyError:
            df.loc[i, 'running_max'] = df.loc[i, 'cum_pl']  

        # Calculate the drawdown
        try:
            df.loc[i, 'drawdown'] = np.where((df.loc[i, 'sell'] == 1), min(df.loc[i, 'cum_pl'] - df.loc[i, 'running_max'], 0), df.loc[i-1, 'drawdown'])
        except KeyError:
            df.loc[i, 'drawdown'] = 0  # Set a default value of 0

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

    # Save the processed data or pass it to a template for display
    processed_backtest_file = os.path.join(results_dir, f"df_backtest_{file_name}.csv")
    df.to_csv(processed_backtest_file, index=False)

    total_profit_loss = df['cum_pl'].iloc[-1] - initial_capital
    win_rate = df['num_winning_trades'].iloc[-1] / df['num_trades'].iloc[-1] * 100
    sharpe_ratio = (df['cum_pl'].iloc[-1] - initial_capital) / (df['cum_pl'].iloc[-1] / df['num_trades'].iloc[-1]) ** 0.5
    num_trades = df['num_trades'].iloc[-1]
    max_dd = df['drawdown'].min()
    avg_profit = total_profit_loss / num_trades

    df_sell = df[df['sell'] == 1].reset_index(drop=True)
    
    trade_log = []
    # Create trade_log from df_sell as per the example data above
    for i in range(len(df_sell)):
        trade_log.append({
            "date": df_sell.loc[i, 'date'],
            "entry": round(df_sell.loc[i, 'buy_price'],2),
            "exit": round(df_sell.loc[i,'sell_price'],2),
            "pnl": round(df_sell.loc[i, 'pl'],2),
            "cum_pnl": round(df_sell.loc[i, 'cum_pl'],2)
        })

    cumulative_pl = {
        "dates": [],
        "values": []
    }
    # Create a cumulative p&l data(cum_pl) from df_sell as per the example data above
    for i in range(len(df_sell)):
        cumulative_pl['dates'].append(df_sell.loc[i, 'date'])
        cumulative_pl['values'].append(df_sell.loc[i, 'cum_pl'])

    return total_profit_loss, win_rate, sharpe_ratio, num_trades, max_dd, avg_profit, trade_log, cumulative_pl