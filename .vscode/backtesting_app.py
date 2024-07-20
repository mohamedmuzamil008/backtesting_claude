# Create a flask app to backtest a trading strategy

# Flask app first input page where the user uploads a CSV file
from flask import Flask, render_template, request, Response, redirect, url_for, stream_with_context, jsonify, session
import os
import json
import pandas as pd
import numpy as np
import os
import pandas_ta as ta
from pandas_ta.momentum import macd

#from strategies.strategy_5 import generate_strategy5_signals
from strategies.strategy_5_nb import generate_strategy5_signals
from create_daily_df import create_daily_frame
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
        results_dir = os.path.join(os.getcwd(), f"results\{file_name}")
        df = pd.read_csv(latest_file)
        print(file_name)
        print(selected_strategy)
        
        total_steps = 6  # Adjust based on your processing steps        
        
        '''Data preprocessing steps here...'''
        df = data_preprocessing(df)
        
        yield "data:" + json.dumps({"progress": 10, "status": "Creating daily timeframe"}) + "\n\n"
        
        daily_df = create_daily_frame(df)        
        
        df = df.reset_index()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Calcaulte ATR in df based on joins from the daily_df on datetime
        df = df.merge(daily_df[['datetime', 'ATR', 'prevday_high', 'uptrend']], left_on='date', right_on='datetime', how='left')
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
        df = df.reset_index(drop=False)

        yield "data:" + json.dumps({"progress": 70, "status": "Calculating Buy and Sell signals"}) + "\n\n"
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['time'] = df['datetime'].dt.time
        
        if selected_strategy == 'Strategy 5':
            generate_strategy5_signals(df, daily_df, results_dir, file_name)         

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
            uploads_dir = os.path.join(os.getcwd(), 'uploads')
            results_dir = os.path.join(os.getcwd(), f"results\{filename.split('.')[0]}")
            os.makedirs(uploads_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            file.save(os.path.join('uploads', filename))
            session['uploaded_file'] = filename.split('.')[0]

            # Get the selected strategy
            selected_strategy = request.form.get('strategy')  
            session['selected_strategy'] = selected_strategy      

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
    uploads_dir = os.path.join(os.getcwd(), 'uploads')         
    files = os.listdir(uploads_dir)
    if not files:
        return "No files uploaded yet."    
    latest_file = max([os.path.join(uploads_dir, f) for f in files], key=os.path.getmtime)
    file_name = latest_file.split('/')[-1].split('.')[0].split("\\")[-1]  
    results_dir = os.path.join(os.getcwd(), f"results\{file_name}")
    processed_file = os.path.join(results_dir, f"df_{file_name}.csv")
    df = pd.read_csv(processed_file)

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

if __name__ == '__main__':
    app.run(debug=True)

