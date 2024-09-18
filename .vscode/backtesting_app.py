# Create a flask app to backtest a trading strategy

# Flask app first input page where the user uploads a CSV file
from flask import Flask, current_app, render_template, request, Response, redirect, url_for, stream_with_context, jsonify, session
import os
import json
import pandas as pd
import numpy as np
import os
import pandas_ta as ta
from pandas_ta.momentum import macd

#from strategies.strategy_5 import generate_strategy5_signals
from strategies.strategy_5_nb import generate_strategy5_signals
from strategies.strategy_5_original import generate_strategy5_original_signals
from strategies.strategy_6_nb import generate_strategy6_signals
from strategies.strategy_7_nb import generate_strategy7_signals
from strategies.strategy_8_nb import generate_strategy8_signals
from strategies.strategy_9_nb import generate_strategy9_signals
from strategies.strategy_10_nb import generate_strategy10_signals
from strategies.strategy_11_nb import generate_strategy11_signals

from create_daily_df import create_daily_frame
from create_5min_df import create_5min_frame
from pl_calculation import calculate_pl
from data_preprocessing import data_preprocessing

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

def calculate_first15(group):
    # Filter the group to include only the first 15 minutes (09:15 to 09:29)
    first15_group = group.between_time('09:15', '09:29')
    
    # Check if first15_group is not empty
    if not first15_group.empty:
        # Calculate first15_open, first15_high, first15_low, and first15_close
        first15_open = first15_group['Open'].iloc[0]
        first15_high = first15_group['High'].max()
        first15_low = first15_group['Low'].min()
        first15_close = first15_group['Close'].iloc[-1]
        
        first15_stats = pd.DataFrame({
            'first15_open': [first15_open],
            'first15_high': [first15_high],
            'first15_low': [first15_low],
            'first15_close': [first15_close]
        }, index=group.index[:1])  # Set the index to the first index of the group
        
        # Get the previous day's high value
        prevday_high = group['prevday_high'].iloc[0]
        
        # Calculate openconviction_15
        first15_stats['openconviction_15'] = calculate_openconviction_15(first15_stats)
        
        # Add the first5 statistics to the group DataFrame
        group = group.join(first15_stats)
        
        # Forward-fill the first5 statistics for the rest of the day
        group['first15_open'] = group['first15_open'].ffill()
        group['first15_high'] = group['first15_high'].ffill()
        group['first15_low'] = group['first15_low'].ffill()
        group['first15_close'] = group['first15_close'].ffill()
        group['openconviction_15'] = group['openconviction_15'].ffill()
    
    return group


def calculate_first1(group):
    # Filter the group to include only the first 5 minutes (09:15 to 09:19)
    first1_group = group.between_time('09:15', '09:15')
    
    # Check if first5_group is not empty
    if not first1_group.empty:
        # Calculate first5_open, first5_high, first5_low, and first5_close
        first1_open = first1_group['Open'].iloc[0]
        first1_high = first1_group['High'].max()
        first1_low = first1_group['Low'].min()
        first1_close = first1_group['Close'].iloc[-1]
        
        first1_stats = pd.DataFrame({
            'first1_open': [first1_open],
            'first1_high': [first1_high],
            'first1_low': [first1_low],
            'first1_close': [first1_close]
        }, index=group.index[:1])  # Set the index to the first index of the group        
                
        # Add the first5 statistics to the group DataFrame
        group = group.join(first1_stats)
        
        # Forward-fill the first5 statistics for the rest of the day
        group['first1_open'] = group['first1_open'].ffill()
        group['first1_high'] = group['first1_high'].ffill()
        group['first1_low'] = group['first1_low'].ffill()
        group['first1_close'] = group['first1_close'].ffill()
    
    return group

def calculate_openconviction(first5_stats, prevday_high):
    openconviction = []
    for _, row in first5_stats.iterrows():
        buying_od_5 = (row['first5_open'] == row['first5_low']) and (row['first5_close'] >= ((row['first5_high'] - row['first5_low']) * 0.5) + row['first5_low']) and (row['first5_close'] > row['first5_open'])
        #buying_otd_5 = (row['first5_open'] != row['first5_low']) and (row['first5_close'] >= (((row['first5_high'] - row['first5_low']) * 0.7) + row['first5_low'])) and (row['first5_close'] > row['first5_open']) and (pd.notna(prevday_high) and row['first5_low'] > prevday_high * 0.999)
        buying_otd_5 = (row['first5_open'] != row['first5_low']) and (row['first5_close'] >= (((row['first5_high'] - row['first5_low']) * 0.7) + row['first5_low'])) and (row['first5_close'] > row['first5_open']) and (row['first5_open'] <= (((row['first5_high'] - row['first5_low']) * 0.2) + row['first5_low']))
        #selling_orr_5 = (row['first5_close'] >= ((row['first5_high'] - row['first5_low']) * 0.5) + row['first5_low']) and (row['first5_open'] < row['first5_close'])
        buying_orr_5 = (row['first5_close'] <= row['first5_open'])

        if buying_od_5:
            openconviction.append(1)
        elif buying_otd_5:
            openconviction.append(2)
        #elif selling_orr_5:
        elif buying_orr_5:
            openconviction.append(6)
        else:
            openconviction.append(7)
    
    return openconviction

def calculate_openconviction_15(first15_stats):
    openconviction_15 = []
    for _, row in first15_stats.iterrows():
        buying_od_15 = (row['first15_open'] == row['first15_low']) and (row['first15_close'] >= ((row['first15_high'] - row['first15_low']) * 0.5) + row['first15_low']) and (row['first15_close'] > row['first15_open'])
        buying_otd_15 = (row['first15_open'] != row['first15_low']) and (row['first15_close'] >= (((row['first15_high'] - row['first15_low']) * 0.7) + row['first15_low'])) and (row['first15_close'] > row['first15_open']) and (row['first15_open'] <= (((row['first15_high'] - row['first15_low']) * 0.2) + row['first15_low']))
        buying_orr_15 = (row['first15_close'] <= row['first15_open'])

        if buying_od_15:
            openconviction_15.append(1)
        elif buying_otd_15:
            openconviction_15.append(2)
        elif buying_orr_15:
            openconviction_15.append(6)
        else:
            openconviction_15.append(7)
    
    return openconviction_15

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
    group['ibtype'] = np.where(group['ibrange/atr'] <= 0.33, "Small_IB",
                        np.where(group['ibrange/atr'] <= 0.5, "Normal_IB",
                            np.where(group['ibrange/atr'] <= 0.8, "Wide_IB", "Very_Wide_IB"                                
                                )))

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

def fetch_ohlc_data(date):

    df = current_app.result_df

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['date'] = df['date'].astype(str)  

    df['time'] = df['time'].astype(str)

    df = df[df['date'] == date].reset_index(drop=True)    
    print(df.shape)
    ohlc_data = df[['Open', 'High', 'Low', 'Close', 'time', 'buy_price', 'sl_price', 'trail_activation_price', 'trail_sl_price', 'tp_price', 'sell_price', 'buy', 'sell']].rename(columns={'Open': 'open', 'High':'high', 'Low':'low', 'Close':'close', 'time':'timestamp', 'tp_price':'target_price'}).to_dict('records')

    return ohlc_data

app = Flask(__name__)
app.secret_key = 'App$ecretKey@98765#'  

@app.route('/process')
def process_data():
    selected_strategy = session.get('selected_strategy')
    def generate():
        # Reading the uploaded CSV file
        uploads_dir = os.path.join(os.getcwd(), 'uploads')        
        files = os.listdir(uploads_dir)
        if not files:
            return "No files uploaded yet."        
        latest_file = max([os.path.join(uploads_dir, f) for f in files], key=os.path.getmtime)
        file_name = latest_file.split('/')[-1].split('.')[0].split("\\")[-1]        
        results_dir = os.path.join(os.getcwd(), f"results\{selected_strategy}\{file_name}")
        df = pd.read_csv(latest_file)
        print(file_name)
        print(selected_strategy)
        
        total_steps = 6  # Adjust based on your processing steps        
        
        '''Data preprocessing steps here...'''
        df = data_preprocessing(df)
        
        yield "data:" + json.dumps({"progress": 10, "status": "Creating daily timeframe"}) + "\n\n"
        
        daily_df = create_daily_frame(df, file_name) 

        ## Comment this block if you don't want 5min timeframe 
        # five_min_df = create_5min_frame(df, file_name)         
        # five_min_df['date'] = five_min_df['datetime'].dt.date
        # five_min_df['date'] = pd.to_datetime(five_min_df['date'])
        # five_min_df = five_min_df.set_index('datetime')   
        # df = five_min_df  
        
        df = df.reset_index()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Calcaulte ATR in df based on joins from the daily_df on datetime
        df = df.merge(daily_df[['datetime', 'Close_avg', 'ATR', 'range_sd', 'volatility', 'SD/ATR', 'multiplier', 'ATR/Close', 'prevday_high', 'prevday_low', 'uptrend', 'downtrend', 'MACD_uptrend', 'ADX_uptrend', 'RSI_uptrend', 'EMA_uptrend',  'VAH', 'VAH_prev', 'POC', 'POC_prev', 'VAL', 'VAL_prev', 'Open_Location']], left_on='date', right_on='datetime', how='left')
        df.drop(['datetime_y'], axis=1, inplace=True)
        df.rename(columns={'datetime_x': 'datetime'}, inplace=True)
        df = df.set_index('datetime')
        
        yield "data:" + json.dumps({"progress": 20, "status": "Calculating CurTop and CurBot"}) + "\n\n"
        
        # Calculate the CurrentTop (curtop) and CurrentBottom (curbot) for each timestamp within the same day
        df = df.groupby(df.index.date).apply(get_curtop_curbot)
        df = df.reset_index(level=0, drop=True)

        yield "data:" + json.dumps({"progress": 30, "status": "Calculating Initial Balance"}) + "\n\n"
        
        # Calculate_initial_balance function to each day
        df = df.groupby(df.index.date).apply(calculate_initial_balance)
        df = df.reset_index(level=0, drop=True)        

        # Apply the calculate_first5 function to each day
        df = df.groupby(df.index.date).apply(calculate_first5)
        df = df.reset_index(level=0, drop=True)

        # Apply the calculate_first15 function to each day
        df = df.groupby(df.index.date).apply(calculate_first15)
        df = df.reset_index(level=0, drop=True)

        # Apply the calculate_first1 function to each day
        df = df.groupby(df.index.date).apply(calculate_first1)
        df = df.reset_index(level=0, drop=True)
        df = df.reset_index(drop=False)
        
        yield "data:" + json.dumps({"progress": 70, "status": "Calculating Buy and Sell signals"}) + "\n\n"
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['time'] = df['datetime'].dt.time
        
        if selected_strategy == 'Strategy 5':
            generate_strategy5_signals(df, daily_df, results_dir, file_name)  
        elif selected_strategy == 'Strategy 5 original':
            generate_strategy5_original_signals(df, daily_df, results_dir, file_name)
        elif selected_strategy == 'Strategy 6':
            generate_strategy6_signals(df, daily_df, results_dir, file_name)
        elif selected_strategy == 'Strategy 7':
            generate_strategy7_signals(df, daily_df, results_dir, file_name)
        elif selected_strategy == 'Strategy 8':
            generate_strategy8_signals(df, daily_df, results_dir, file_name) 
        elif selected_strategy == 'Strategy 9':
            generate_strategy9_signals(df, daily_df, results_dir, file_name)  
        elif selected_strategy == 'Strategy 10':
            generate_strategy10_signals(df, daily_df, results_dir, file_name)
        elif selected_strategy == 'Strategy 11':
            generate_strategy11_signals(df, daily_df, results_dir, file_name)         


        yield "data:" + json.dumps({"progress": 100, "status": "Processing complete"}) + "\n\n"
                    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and file.filename.endswith('.csv'):
            filename = file.filename

            # Get the selected strategy
            selected_strategy = request.form.get('strategy')  
            session['selected_strategy'] = selected_strategy 

            uploads_dir = os.path.join(os.getcwd(), 'uploads')
            results_dir = os.path.join(os.getcwd(), f"results\{selected_strategy}\{filename.split('.')[0]}")
            os.makedirs(uploads_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            file.save(os.path.join('uploads', filename))
            session['uploaded_file'] = filename.split('.')[0]

                 

            return redirect(url_for('show_processing'))
    return render_template('upload.html')

@app.route('/processing')
def show_processing():
    return render_template('processing.html')

@app.route('/results')
def results():
    strategy_name = session.get('selected_strategy', 'Unknown Strategy')
    file_name = session.get('uploaded_file', 'Unknown File') 
    return render_template('results.html', strategy_name=strategy_name, file_name=file_name)

@app.route('/calculate_results', methods=['POST'])
def calculate_results():
    initial_capital = float(request.form['initialCapital'])
    margin = float(request.form['margin'])
    position_size = float(request.form['positionSize'])
    commission = 100
    position_sizing_pct = position_size/100
    
    '''P&L logic starts here...'''
    # Get the results file
    selected_strategy = session.get('selected_strategy', 'Unknown Strategy')
    uploads_dir = os.path.join(os.getcwd(), 'uploads')         
    files = os.listdir(uploads_dir)
    if not files:
        return "No files uploaded yet."    
    latest_file = max([os.path.join(uploads_dir, f) for f in files], key=os.path.getmtime)
    file_name = latest_file.split('/')[-1].split('.')[0].split("\\")[-1]  
    results_dir = os.path.join(os.getcwd(), f"results\{selected_strategy}\{file_name}")

    df = current_app.result_df
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['date'] = df['date'].astype(str)    

    total_profit_loss, win_rate, sharpe_ratio, num_trades, max_dd, avg_profit, trade_log, cumulative_pl  = calculate_pl(df, initial_capital, margin, position_sizing_pct, commission, results_dir, file_name)
    
    return jsonify({
        'totalProfitLoss': f"Rs.{total_profit_loss:.2f}",
        'winRate': f"{win_rate:.2f}%",
        'sharpeRatio': f"{sharpe_ratio:.2f}",
        'numTrades': f"{num_trades:.2f}",
        'maxDrawdown': f"Rs.{max_dd:.2f}",
        'avgProfit': f"Rs.{avg_profit:.2f}",
        'tradeLog': trade_log,
        'cumulativePL': cumulative_pl

    })

@app.route('/chart/<date>')
def chart_screen(date):
    # Fetch OHLC data for the specified date
    ohlc_data = fetch_ohlc_data(date)
    return render_template('chart_screen.html', date=date, ohlc_data=ohlc_data)

if __name__ == '__main__':
    app.run(debug=True)

