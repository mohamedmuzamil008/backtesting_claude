import pandas as pd
import os

# master file
master = pd.read_csv("E:/App Dev/GitHub/backtesting_claude/data/value_area_levels_for_all_stocks_new_2.csv")
master = master[['Stock_Name', 'Date', 'VAH Prev', 'VAL Prev']]

# Specify the folder path
folder_path = 'E:/Amibroker/Historical Data/Combined_for_all_years_upto Feb 2024_with_value_areas'

# Get a list of all .csv files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read the files one by one
for file_name in csv_files:
#for file_name in ['BPCL.csv']:
    print(file_name)
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Date'] = df['Date'].dt.strftime('%d-%m-%Y') 
    
    # Retrieve the VAH Prev and VAL Prev values from the master file based on Date and Stock_Name
    df = pd.merge(df, master, how='left', on=['Stock_Name', 'Date'])

    df.to_csv(file_path,index=False)