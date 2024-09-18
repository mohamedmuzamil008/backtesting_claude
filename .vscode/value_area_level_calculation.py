# Databricks notebook source
# DBTITLE 1,Importing the necessary libraries
import pandas as pd
import numpy as np
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id
from pyspark_assert import assert_frame_equal
import math
import functools
import datetime
import gc

# COMMAND ----------

# DBTITLE 1,Creating new JDBC connection
jdbcHostname = "algoploy-sql-server.database.windows.net"
jdbcPort = 1433
jdbcDatabase = "market_data"
jdbcTable = "[dbo].[historical_master]"
sqlUserName = "algoploy"
sqlPassword = "sqlserver@92"

jdbcUrl = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,jdbcDatabase)
connectionProperties = {
    "driver":"com.microsoft.sqlserver.jdbc.SQLServerDriver",
    "authentication" : "SqlPassword",
    "userName" : sqlUserName,
    "password" : sqlPassword
}

# COMMAND ----------

# DBTITLE 1,Custom Functions
def find_indexes(input_array, mid_index):
    lower_side = input_array[:mid_index][::-1]
    higher_side = input_array[mid_index + 1:]

    total = input_array[mid_index]
    overall_sum = sum(input_array)
    idx_low = 0
    idx_high = 0

    while total < int(overall_sum * 0.7):
        next_low = lower_side[idx_low] if idx_low < len(lower_side) else float('-inf')
        next_high = higher_side[idx_high] if idx_high < len(higher_side) else float('-inf')

        if next_low >= next_high:
            total += next_low
            idx_low += 1
        else:
            total += next_high
            idx_high += 1

    up_index = mid_index + idx_high + 1  # Adjust indices to get the final indexes
    down_index = mid_index - idx_low
    
    return up_index, down_index


@udf("array<float>")
def rep_(j_array, baseY, Den):
    return [baseY + (i * Den) for i in (range(j_array + 1))]

def calculateExtremes(input_array):
    extremeCount = 0
    for i in input_array:
        if(i == 1 or i == 0):
            extremeCount = extremeCount + 1
        else:
            break
    return extremeCount


def extremePresentCheck(buyExtremePresent, shortExtremePresent, netExtremeCount):
    # 1-> Buy 2-> Short 3-> Neutral 4-> Absent
    if buyExtremePresent == 0 and shortExtremePresent == 0:
        return 4
    elif (buyExtremePresent == 1 and shortExtremePresent == 0) or (buyExtremePresent == 1 and shortExtremePresent == 1 and netExtremeCount >= 3):
        return 1
    elif (buyExtremePresent == 0 and shortExtremePresent == 1) or (buyExtremePresent == 1 and shortExtremePresent == 1 and netExtremeCount <= -3):
        return 2
    elif (buyExtremePresent == 1 and shortExtremePresent == 1):
        return 3 

def rangeExtensionCheck(buyRangeExtension, shortRangeExtension):
    # 1-> Buying 2-> Selling 3-> Neutral 4-> Absent
    if buyRangeExtension == 0 and shortRangeExtension == 0:
        return 4
    elif buyRangeExtension == 1 and shortRangeExtension == 1:
        return 3
    elif buyRangeExtension == 1 and shortRangeExtension == 0:
        return 1
    elif buyRangeExtension == 0 and shortRangeExtension == 1:
        return 2
    
def IBTypeCheck(IBRange, ATR):
    # 1-> Small 2-> Normal 3-> Wide 4-> Very Wide
    if IBRange <= 0.33 * ATR:
        return 1
    elif IBRange > 0.33 * ATR and IBRange <= 0.5 * ATR:
        return 2
    elif IBRange > 0.5 * ATR and IBRange <= 0.8 * ATR:
        return 3
    if IBRange > 0.8 * ATR:
        return 4
    
def tpoCountCheck(buyTPOCount, shortTPOCount):
    # 1-> Buying TPO 2-> Selling TPO 3-> Neutral TPO
    if buyTPOCount == shortTPOCount:
        return 3
    elif buyTPOCount > shortTPOCount:
        return 1
    elif buyTPOCount < shortTPOCount:
        return 2
    
def marketSentimentCheck(extremePresent, rangeExtension, tpoCount, valueShift):
    # 1-> Positive 2-> Slightly Positive 3-> Negative 4-> Slightly Negative 5-> Neutral
    if valueShift == 1 and rangeExtension == 1 and extremePresent == 1:
        return 1
    elif valueShift == 1 and rangeExtension == 4 and extremePresent == 4 and tpoCount == 1:
        return 2
    elif valueShift == 1 and rangeExtension == 1 and extremePresent != 1:
        return 2
    elif valueShift == 1 and rangeExtension != 1 and extremePresent == 1:
        return 2
    elif valueShift != 1 and rangeExtension == 1 and extremePresent == 1:
        return 2
    elif valueShift == 2 and rangeExtension == 2 and extremePresent == 2:
        return 3
    elif valueShift == 2 and rangeExtension == 4 and extremePresent == 4 and tpoCount == 2:
        return 4
    elif valueShift == 2 and rangeExtension == 2 and extremePresent != 2:
        return 4
    elif valueShift == 2 and rangeExtension != 2 and extremePresent == 2:
        return 4
    elif valueShift != 2 and rangeExtension == 2 and extremePresent == 2:
        return 4
    else:
        return 5
    
def openLocationCheck(todayOpen, yestHigh, yestLow, vah_prev, val_prev):
    # Within PDVA - 1, Above PDVA below PDH - 2, Above PDH - 3, Below PDVA above PDL - 4, Below PDL - 5  
    if todayOpen <= vah_prev and todayOpen >= val_prev:
        return 1
    elif todayOpen > vah_prev and todayOpen <= yestHigh:
        return 2
    elif todayOpen > yestHigh:
        return 3
    elif todayOpen < val_prev and todayOpen >= yestLow:
        return 4
    elif todayOpen < yestLow:
        return 5
    
def openConviction_5minCheck(Buying_OD_5, Buying_OTD_5, Buying_ORR_5, Selling_OD_5, Selling_OTD_5, Selling_ORR_5):
    # Buying_OD - 1, Buying_OTD - 2, Buying_ORR - 3, Selling_OD - 4, Selling_OTD - 5, Selling_ORR - 6, Others - 7
    if Buying_OD_5 == 1:
        return 1
    elif Buying_OTD_5 == 1:
        return 2
    elif Selling_OD_5 == 1:
        return 4
    elif Selling_OTD_5 == 1:
        return 5
    elif Buying_ORR_5 == 1:
        return 3
    elif Selling_ORR_5 == 1:
        return 6
    else:
        return 7

def openConviction_15minCheck(Buying_OD_15, Buying_OTD_15, Buying_ORR_15, Selling_OD_15, Selling_OTD_15, Selling_ORR_15):
    # Buying_OD - 1, Buying_OTD - 2, Buying_ORR - 3, Selling_OD - 4, Selling_OTD - 5, Selling_ORR - 6, Others - 7
    if Buying_OD_15 == 1:
        return 1
    elif Buying_OTD_15 == 1:
        return 2
    elif Selling_OD_15 == 1:
        return 4
    elif Selling_OTD_15 == 1:
        return 5
    elif Buying_ORR_15 == 1:
        return 3
    elif Selling_ORR_15 == 1:
        return 6
    else:
        return 7
    
def openConviction_30minCheck(StrongUp, ModerateUp, StrongDown, ModerateDown):
    # Strong Up - 1, Moderate Up - 2, Strong Down - 3, Moderate Down - 4, Other - 5
    if StrongUp == 1:
        return 1
    elif ModerateUp == 1:
        return 2
    elif StrongDown == 1:
        return 3
    elif ModerateDown == 1:
        return 4
    else:
        return 5
    
def openConviction_5min_2Check(Buying_OD_5_2, Buying_OTD_5_2, Buying_ORR_5_2, Selling_OD_5_2, Selling_OTD_5_2, Selling_ORR_5_2):
    if Buying_OD_5_2 == 1:
        return 1
    elif Buying_OTD_5_2 == 1:
        return 2
    elif Selling_OD_5_2 == 1:
        return 4
    elif Selling_OTD_5_2 == 1:
        return 5
    elif Buying_ORR_5_2 == 1:
        return 3
    elif Selling_ORR_5_2 == 1:
        return 6
    else:
        return 7

def unionAll(dfs):
    return functools.reduce(lambda df1,df2: df1.union(df2.select(df1.columns)), dfs)

# COMMAND ----------

# DBTITLE 1,Reading the input tables
df = spark.read.jdbc(url=jdbcUrl,table=jdbcTable,properties = connectionProperties)
column_list = ["Stock_Name","Date"]

w = Window.partitionBy(*column_list).orderBy("Timestamp").rowsBetween(Window.unboundedPreceding, Window.currentRow)
df = df.withColumn("CurTop", F.max("High").over(w))
df = df.withColumn("CurBot", F.min("Low").over(w))

df.display()

# COMMAND ----------

df_daily = spark.read.jdbc(url=jdbcUrl,table="[dbo].[historical_daily]",properties = connectionProperties)
df_daily = df_daily.orderBy("Stock_Name", "Date", ascending=[True, True])
df_daily.display()

# COMMAND ----------

df_15min = spark.read.jdbc(url=jdbcUrl,table="[dbo].[historical_15min]",properties = connectionProperties)
#df_15min.display()

# COMMAND ----------

df_5min = spark.read.jdbc(url=jdbcUrl,table="[dbo].[historical_5min]",properties = connectionProperties)
#df_5min.display()

# COMMAND ----------

# DBTITLE 1,Reading the output table to check if there are already available data
df_ref_levels = spark.read.jdbc(url=jdbcUrl,table="[dbo].[historical_daily_reference_levels]",properties = connectionProperties)
df_ref_levels_to_be_appended = df_ref_levels.filter(df_ref_levels.Stock_Name == "NA")
df_ref_levels.display()

# COMMAND ----------

# Exclude those stock_name-date combination in the main tables


# COMMAND ----------

# DBTITLE 1,Creating a basic output table
df_output = df_daily
df_output = df_output.drop(*['Volume', 'Timestamp'])
#df_output.display()

# COMMAND ----------

# DBTITLE 1,Calculating Den
w = Window.partitionBy("Stock_Name").orderBy("Date")
df_output = df_output.withColumn("Previous_High", F.lag("High").over(w)).withColumn("Previous_Low", F.lag("Low").over(w))
df_output = df_output.withColumn("High_of_Two",F.greatest(*["High","Previous_High"])).withColumn("Low_of_Two",F.least(*["Low","Previous_Low"]))
df_output = df_output.withColumn("Den", F.round(F.greatest(F.floor(F.round(df_output.High_of_Two - df_output.Low_of_Two, 2) * 0.3) * 0.05, F.lit(0.05)), 2))
df_output = df_output.withColumn("DayRange", F.round(df_output.High - df_output.Low, 2))\
            .withColumn("baseY", F.round(F.floor(df_output.Low / df_output.Den) * df_output.Den, 2))\
            .withColumn("maxY", F.round(F.ceil(df_output.High / df_output.Den) * df_output.Den, 2))
df_output = df_output.withColumn("todayRange", F.round((df_output.maxY - df_output.baseY) / df_output.Den, 0))
w = Window.partitionBy("Stock_Name").orderBy("Date").rowsBetween(-9, 0)
df_output = df_output.withColumn("ATR", F.round(F.mean(df_output.DayRange).over(w), 2))
df_output.display()

# COMMAND ----------

# DBTITLE 1,Validations
#baseY <= Low
print(df_output.filter(df_output.baseY > df_output.Low).count() == 0)

#maxY >= High
print(df_output.filter(df_output.maxY < df_output.High).count() == 0)

#baseY + todayRange * Den == maxY
print(df_output.filter(df_output.maxY != F.round(df_output.baseY + (df_output.todayRange * df_output.Den), 2)).count() == 0)

# COMMAND ----------

# Retrieving the stock name and date from daily table and check it in the reference table
#df_daily_iter = df_daily.rdd.map(lambda x: (x.Stock_Name, x.Date))
df_daily = df_daily.filter(df_daily.Stock_Name == "ACC")
#df_daily_pd = df_daily.select(df_daily.Stock_Name, df_daily.Date).toPandas()
#arr_Stock = df_daily_pd['Stock_Name'].to_numpy()
#arr_Date = df_daily_pd['Date'].to_numpy()
startDate = datetime.datetime(2017,1,1)
endDate = datetime.datetime(2023,12,29)
date_value_prev = datetime.datetime(2017,1,1)

for i in range((endDate - startDate).days):
    #print(df_daily_iter.collect()[i][1])
    #symbol = df_daily_iter.collect()[i][0]
    #date_value = df_daily_iter.collect()[i][1].strftime("%Y-%m-%d")
    #date_value_prev = df_daily_iter.collect()[i][1].strftime("%Y-%m-%d")
    #print(df_daily_pd.values[i, 1])
    #symbol = df_daily_pd.values[i, 0]
    #date_value = df_daily_pd.values[i, 1].strftime("%Y-%m-%d")
    #date_value_prev = df_daily_pd.values[i, 1].strftime("%Y-%m-%d")
    #print(arr_Date[i])
    #symbol = arr_Stock[i]
    #date_value = arr_Date[i].strftime("%Y-%m-%d")
    #date_value_prev = arr_Date[i].strftime("%Y-%m-%d")
    symbol = "ACC"
    startDate += datetime.timedelta(days=1)
    date_value = startDate.strftime("%Y-%m-%d")
    #date_value_prev = startDate.strftime("%Y-%m-%d")
    try:
        dummy_val = df_daily.filter((df_daily.Stock_Name == symbol) & (df_daily.Date == date_value)).collect()[0]
    except:
        continue
    prev_day_filter = 0
    #print(i)
    print(date_value)
    print(date_value_prev)
    
    #Continue if the stock_date is already present in reference table
    if df_ref_levels.filter((df_ref_levels.Stock_Name == symbol) & (df_ref_levels.Date == date_value)).count() != 0:
        print("Data already present for this date! Skipping!!")
        date_value_prev = date_value
        continue
    
    # Retrieve the previous data reference levels
    if i != 0:
        try:
            #date_value_prev = arr_Date[i-1].strftime("%Y-%m-%d")
            prev_day_filter = df_ref_levels.filter((df_ref_levels.Stock_Name == symbol) & (df_ref_levels.Date == date_value_prev)).collect()[0]
        except:
            prev_day_filter = df_ref_levels.collect()[0]
    
    # Filter for the current day
    day_filter = df_output.filter((df_output.Stock_Name == symbol) & (df_output.Date == date_value)).collect()[0]  
    todayRange = int(getattr(day_filter, "todayRange"))
    todayOpen = getattr(day_filter, "Open")
    yestHigh = getattr(day_filter, "Previous_High")
    yestLow = getattr(day_filter, "Previous_Low")
    baseY = getattr(day_filter, "baseY")
    Den = getattr(day_filter, "Den")
    ATR = getattr(day_filter, "ATR")

    '''Calculating the Market Profile - TPO Prints'''
    data_list = [(todayRange,)]
    j_array = spark.createDataFrame(data_list,StructType([ StructField("myInt", IntegerType(), True)]))
    j_array = j_array.withColumn("baseY", F.lit(baseY)).withColumn("Den", F.lit(Den))
    j_array = j_array.withColumn("myArr", rep_("myInt", "baseY", "Den"))
    
    # Getting the high and low for each of the 30min period    
    df_15min_temp = df_15min.filter((df_15min.Stock_Name == symbol) & (df_15min.Date == date_value)).sort(df_15min.Timestamp.asc()).select(df_15min.High, df_15min.Low, df_15min.Open, df_15min.Close, df_15min.Timestamp)
    
    #print("df_15min_count: " + str(df_15min_temp.count()))
    if df_15min_temp.count() == 25:
        a_period = df_15min_temp.collect()[0:2]
        a_high = [ ele.__getattr__("High") for ele in a_period]
        a_high.sort(reverse=True)
        a_high = list(map(a_high.__getitem__, [0]))[0]
        a_low = [ ele.__getattr__("Low") for ele in a_period]
        a_low.sort()
        a_low = list(map(a_low.__getitem__, [0]))[0]
    
        b_period = df_15min_temp.collect()[2:4]
        b_high = [ ele.__getattr__("High") for ele in b_period]
        b_high.sort(reverse=True)
        b_high = list(map(b_high.__getitem__, [0]))[0]
        b_low = [ ele.__getattr__("Low") for ele in b_period]
        b_low.sort()
        b_low = list(map(b_low.__getitem__, [0]))[0]
    
        c_period = df_15min_temp.collect()[4:6]
        c_high = [ ele.__getattr__("High") for ele in c_period]
        c_high.sort(reverse=True)
        c_high = list(map(c_high.__getitem__, [0]))[0]
        c_low = [ ele.__getattr__("Low") for ele in c_period]
        c_low.sort()
        c_low = list(map(c_low.__getitem__, [0]))[0]
    
        d_period = df_15min_temp.collect()[6:8]
        d_high = [ ele.__getattr__("High") for ele in d_period]
        d_high.sort(reverse=True)
        d_high = list(map(d_high.__getitem__, [0]))[0]
        d_low = [ ele.__getattr__("Low") for ele in d_period]
        d_low.sort()
        d_low = list(map(d_low.__getitem__, [0]))[0]
    
        e_period = df_15min_temp.collect()[8:10]
        e_high = [ ele.__getattr__("High") for ele in e_period]
        e_high.sort(reverse=True)
        e_high = list(map(e_high.__getitem__, [0]))[0]
        e_low = [ ele.__getattr__("Low") for ele in e_period]
        e_low.sort()
        e_low = list(map(e_low.__getitem__, [0]))[0]
    
        f_period = df_15min_temp.collect()[10:12]
        f_high = [ ele.__getattr__("High") for ele in f_period]
        f_high.sort(reverse=True)
        f_high = list(map(f_high.__getitem__, [0]))[0]
        f_low = [ ele.__getattr__("Low") for ele in f_period]
        f_low.sort()
        f_low = list(map(f_low.__getitem__, [0]))[0]
    
        g_period = df_15min_temp.collect()[12:14]
        g_high = [ ele.__getattr__("High") for ele in g_period]
        g_high.sort(reverse=True)
        g_high = list(map(g_high.__getitem__, [0]))[0]
        g_low = [ ele.__getattr__("Low") for ele in g_period]
        g_low.sort()
        g_low = list(map(g_low.__getitem__, [0]))[0]
    
        h_period = df_15min_temp.collect()[14:16]
        h_high = [ ele.__getattr__("High") for ele in h_period]
        h_high.sort(reverse=True)
        h_high = list(map(h_high.__getitem__, [0]))[0]
        h_low = [ ele.__getattr__("Low") for ele in h_period]
        h_low.sort()
        h_low = list(map(h_low.__getitem__, [0]))[0]
    
        i_period = df_15min_temp.collect()[16:18]
        i_high = [ ele.__getattr__("High") for ele in i_period]
        i_high.sort(reverse=True)
        i_high = list(map(i_high.__getitem__, [0]))[0]
        i_low = [ ele.__getattr__("Low") for ele in i_period]
        i_low.sort()
        i_low = list(map(i_low.__getitem__, [0]))[0]
    
        j_period = df_15min_temp.collect()[18:20]
        j_high = [ ele.__getattr__("High") for ele in j_period]
        j_high.sort(reverse=True)
        j_high = list(map(j_high.__getitem__, [0]))[0]
        j_low = [ ele.__getattr__("Low") for ele in j_period]
        j_low.sort()
        j_low = list(map(j_low.__getitem__, [0]))[0]
    
        k_period = df_15min_temp.collect()[20:22]
        k_high = [ ele.__getattr__("High") for ele in k_period]
        k_high.sort(reverse=True)
        k_high = list(map(k_high.__getitem__, [0]))[0]
        k_low = [ ele.__getattr__("Low") for ele in k_period]
        k_low.sort()
        k_low = list(map(k_low.__getitem__, [0]))[0]
    
        l_period = df_15min_temp.collect()[22:24]
        l_high = [ ele.__getattr__("High") for ele in l_period]
        l_high.sort(reverse=True)
        l_high = list(map(l_high.__getitem__, [0]))[0]
        l_low = [ ele.__getattr__("Low") for ele in l_period]
        l_low.sort()
        l_low = list(map(l_low.__getitem__, [0]))[0]
    
        m_period = df_15min_temp.collect()[24:25]
        m_high = [ ele.__getattr__("High") for ele in m_period]
        m_high.sort(reverse=True)
        m_high = list(map(m_high.__getitem__, [0]))[0]
        m_low = [ ele.__getattr__("Low") for ele in m_period]
        m_low.sort()
        m_low = list(map(m_low.__getitem__, [0]))[0]   
    else:
        continue
        a_high = df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 09:44:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        a_low = df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 09:44:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        b_high = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 09:45:00") & (df_15min_temp.Timestamp <= date_value +" 10:14:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        b_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 09:45:00") & (df_15min_temp.Timestamp <= date_value +" 10:14:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        c_high = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 10:15:00") & (df_15min_temp.Timestamp <= date_value +" 10:44:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        c_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 10:15:00") & (df_15min_temp.Timestamp <= date_value +" 10:44:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        d_high = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 10:45:00") & (df_15min_temp.Timestamp <= date_value +" 11:14:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        d_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 10:45:00") & (df_15min_temp.Timestamp <= date_value +" 11:14:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        e_high = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 11:15:00") & (df_15min_temp.Timestamp <= date_value +" 11:44:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        e_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 11:15:00") & (df_15min_temp.Timestamp <= date_value +" 11:44:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        f_high = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 11:45:00") & (df_15min_temp.Timestamp <= date_value +" 12:14:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        f_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 11:45:00") & (df_15min_temp.Timestamp <= date_value +" 12:14:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        g_high = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 12:15:00") & (df_15min_temp.Timestamp <= date_value +" 12:44:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        g_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 12:15:00") & (df_15min_temp.Timestamp <= date_value +" 12:44:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        h_high = df_15min_temp.filter( (df_15min_temp.Timestamp >= date_value +" 12:45:00") & (df_15min_temp.Timestamp <= date_value +" 13:14:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        h_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 12:45:00") & (df_15min_temp.Timestamp <= date_value +" 13:14:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        i_high = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 13:15:00") & (df_15min_temp.Timestamp <= date_value +" 13:44:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        i_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 13:15:00") & (df_15min_temp.Timestamp <= date_value +" 13:44:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        j_high = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 13:45:00") & (df_15min_temp.Timestamp <= date_value +" 14:14:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        j_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 13:45:00") & (df_15min_temp.Timestamp <= date_value +" 14:14:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        k_high = df_15min_temp.filter( (df_15min_temp.Timestamp >= date_value +" 14:15:00") & (df_15min_temp.Timestamp <= date_value +" 14:44:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        k_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 14:15:00") & (df_15min_temp.Timestamp <= date_value +" 14:44:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        l_high = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 14:45:00") & (df_15min_temp.Timestamp <= date_value +" 15:14:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        l_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 14:45:00") & (df_15min_temp.Timestamp <= date_value +" 15:14:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
    
        m_high = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 15:15:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High
        m_low = df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 15:15:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low
        
    # Creating the J bins
    j_bin_list = j_array.collect()[0][3]
    j_bin_list = [ '%.2f' % elem for elem in j_bin_list ]
    j_bin_list = list(map(float, j_bin_list))
    j_bins = np.array(j_bin_list)
    
    # Getting all the elements for each 30min period and putting the elements in j bins
    j_idx = list(range(todayRange + 1))
    x_j = list(range(todayRange + 1))
    
    a_index = np.array([a_low, a_high])
    b_index = np.array([b_low, b_high])
    c_index = np.array([c_low, c_high])
    d_index = np.array([d_low, d_high])
    e_index = np.array([e_low, e_high])
    f_index = np.array([f_low, f_high])
    g_index = np.array([g_low, g_high])
    h_index = np.array([h_low, h_high])
    i_index = np.array([i_low, i_high])
    j_index = np.array([j_low, j_high])
    k_index = np.array([k_low, k_high])
    l_index = np.array([l_low, l_high])
    m_index = np.array([m_low, m_high])
    
    a_index = np.digitize(a_index, j_bins, right=True)
    a_index = list(range(a_index[0], a_index[1] + 1))
    
    b_index = np.digitize(b_index, j_bins, right=True)
    b_index = list(range(b_index[0], b_index[1] + 1))
    
    c_index = np.digitize(c_index, j_bins, right=True)
    c_index = list(range(c_index[0], c_index[1] + 1))
    
    d_index = np.digitize(d_index, j_bins, right=True)
    d_index = list(range(d_index[0], d_index[1] + 1))
    
    e_index = np.digitize(e_index, j_bins, right=True)
    e_index = list(range(e_index[0], e_index[1] + 1))
    
    f_index = np.digitize(f_index, j_bins, right=True)
    f_index = list(range(f_index[0], f_index[1] + 1))
    
    g_index = np.digitize(g_index, j_bins, right=True)
    g_index = list(range(g_index[0], g_index[1] + 1))
    
    h_index = np.digitize(h_index, j_bins, right=True)
    h_index = list(range(h_index[0], h_index[1] + 1))
    
    i_index = np.digitize(i_index, j_bins, right=True)
    i_index = list(range(i_index[0], i_index[1] + 1))
    
    j_index = np.digitize(j_index, j_bins, right=True)
    j_index = list(range(j_index[0], j_index[1] + 1))
    
    k_index = np.digitize(k_index, j_bins, right=True)
    k_index = list(range(k_index[0], k_index[1] + 1))
    
    l_index = np.digitize(l_index, j_bins, right=True)
    l_index = list(range(l_index[0], l_index[1] + 1))
    
    m_index = np.digitize(m_index, j_bins, right=True)
    m_index = list(range(m_index[0], m_index[1] + 1))
    
    # Calculating the sum of elements in each J bins
    # Initialize the output array with zeros
    x_j = [0] * len(j_idx)
    
    # Count occurrences in a_index
    for index in a_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in b_index
    for index in b_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in c_index
    for index in c_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in d_index
    for index in d_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in e_index
    for index in e_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in f_index
    for index in f_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in g_index
    for index in g_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in h_index
    for index in h_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in i_index
    for index in i_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in j_index
    for index in j_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in k_index
    for index in k_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in l_index
    for index in l_index:
        if index in j_idx:
            x_j[index] += 1
    
    # Count occurrences in m_index
    for index in m_index:
        if index in j_idx:
            x_j[index] += 1
    
    x_j = x_j[1:]
    #print("Output Array:", x_j)
    #print("Output Array created!")
    
    '''Validation'''
    #print("Validation: " + str(len(x_j) == todayRange))
    
    '''Calculate the VAH, VAL and POC'''
    # Calculate the half price index
    halfpriceIndex = int(todayRange/2)
    
    # Calculate sum of lower and upper count
    lowerCount = sum(x_j[0:halfpriceIndex])
    upperCount = sum(x_j[halfpriceIndex:])
    tpoHighorLow = 1 if lowerCount > upperCount else 2 if lowerCount < upperCount else -1
    
    # Calculate the POC
    maxTPOCounter = x_j.count(max(x_j))
    maxTPOArray = [i for i in range(len(x_j)) if x_j[i] == max(x_j)]
    
    if maxTPOCounter == 1:
        maxJ = maxTPOArray[0]
    else:
        maxTPOdistfromMidArray = [abs(x - halfpriceIndex) for x in maxTPOArray]
        minDistCount = maxTPOdistfromMidArray.count(min(maxTPOdistfromMidArray))
        minDistArray = [maxTPOArray[i] for i in [i for i in range(len(maxTPOdistfromMidArray)) if maxTPOdistfromMidArray[i] == min(maxTPOdistfromMidArray)]]
        if minDistCount == 1:
            maxJ = minDistArray[0]
        else:
            maxJ = minDistArray[0] if tpoHighorLow == 1 or tpoHighorLow == -1 else minDistArray[1]
    
    # Calculate the VAH and VAL
    up_idx, down_idx = find_indexes(x_j, maxJ)
    vah = round(baseY + (up_idx * Den),2)
    val = round(baseY + (down_idx * Den),2)
    poc = round(baseY + (maxJ * Den),2)
    
    #print(f"VAL: {val}, POC: {poc}, VAH:{vah}, maxJ:{maxJ}, up:{up_idx}, dn:{down_idx}, base:{baseY}, Den:{Den}")
    
    '''Calculate IBH, IBL and IBType'''
    # First 60mins
    if df_15min_temp.count() == 25:
        min60_filter = df_15min_temp.collect()[0:4]
        first60minHigh = [ ele.__getattr__("High") for ele in min60_filter]
        first60minHigh.sort(reverse=True)
        first60minHigh = round(list(map(first60minHigh.__getitem__, [0]))[0], 2)
        first60minLow = [ ele.__getattr__("Low") for ele in min60_filter]
        first60minLow.sort()
        first60minLow = round(list(map(first60minLow.__getitem__, [0]))[0], 2)
        first60minOpen = [ ele.__getattr__("Open") for ele in min60_filter]    
        first60minOpen = round(list(map(first60minOpen.__getitem__, [0]))[0], 2)
        first60minClose = [ ele.__getattr__("Close") for ele in min60_filter]    
        first60minClose = round(list(map(first60minClose.__getitem__, [3]))[0], 2)
    else:
        continue
        first60minHigh = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 10:14:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High, 2)
        first60minLow = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 10:14:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low, 2)
        first60minOpen = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 10:14:00")).sort(df_15min_temp.Timestamp.asc()).select(df_15min_temp.Open).collect()[0].Open, 2)
        first60minClose = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 10:14:00")).sort(df_15min_temp.Timestamp.desc()).select(df_15min_temp.Close).collect()[0].Close, 2)
    
    IBH = first60minHigh
    IBL = first60minLow
    IBRange = round(IBH - IBL, 2)
    
    IBTarget0High = IBH + (IBRange * 0.25)
    IBTarget0Low = IBL - (IBRange * 0.25)
    IBTarget1High = IBH + (IBRange * 0.5)
    IBTarget1Low = IBL - (IBRange * 0.5)
    IBTarget15High = IBH + (IBRange * 0.6)
    IBTarget15Low = IBL - (IBRange * 0.6)
    IBTarget2High = IBH + (IBRange * 0.8)
    IBTarget2Low = IBL - (IBRange * 0.8)
    IBTarget3High = IBH + (IBRange * 1)
    IBTarget3Low = IBL - (IBRange * 1)
    IBTarget4High = IBH + (IBRange * 1.2)
    IBTarget4Low = IBL - (IBRange * 1.2)
    
    IBType = IBTypeCheck(IBRange, ATR)
    #print("IB levels calculated!")
    
    '''Calculate First5, First15, First30, Second15, Second30 and Day OHLC'''
    dayHigh = round(getattr(day_filter, "High"), 2)
    dayLow =  round(getattr(day_filter, "Low"), 2)
    dayOpen = round(getattr(day_filter, "Open"), 2)
    dayClose = round(getattr(day_filter, "Close"), 2)
    
    df_5min_temp = df_5min.filter((df_5min.Stock_Name == symbol) & (df_5min.Date == date_value)).sort(df_5min.Timestamp.asc())
    
    # First 5mins
    if df_5min_temp.count() == 75:
        min5_filter = df_5min_temp.collect()[0]
        first5minHigh = round(getattr(min5_filter, "High"), 2)
        first5minLow = round(getattr(min5_filter, "Low"), 2)
        first5minOpen = round(getattr(min5_filter, "Open"), 2)
        first5minClose = round(getattr(min5_filter, "Close"), 2)
    else:
        first5minHigh = round(df_5min_temp.filter((df_5min_temp.Timestamp <= date_value +" 09:19:00")).select(df_5min_temp.High).collect()[0].High, 2)
        first5minLow = round(df_5min_temp.filter((df_5min_temp.Timestamp <= date_value +" 09:19:00")).select(df_5min_temp.Low).collect()[0].Low, 2)
        first5minOpen = round(df_5min_temp.filter((df_5min_temp.Timestamp <= date_value +" 09:19:00")).select(df_5min_temp.Open).collect()[0].Open, 2)
        first5minClose = round(df_5min_temp.filter((df_5min_temp.Timestamp <= date_value +" 09:19:00")).select(df_5min_temp.Close).collect()[0].Close, 2)
    
    # First 15mins
    if df_15min_temp.count() == 25:
        min15_filter = df_15min_temp.collect()[0]
        first15minHigh = round(getattr(min15_filter, "High"), 2)
        first15minLow = round(getattr(min15_filter, "Low"), 2)
        first15minOpen = round(getattr(min15_filter, "Open"), 2)
        first15minClose = round(getattr(min15_filter, "Close"), 2)
    else:
        continue
        first15minHigh = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 09:29:00")).select(df_15min_temp.High).collect()[0].High, 2)
        first15minLow = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 09:29:00")).select(df_15min_temp.Low).collect()[0].Low, 2)
        first15minOpen = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 09:29:00")).select(df_15min_temp.Open).collect()[0].Open, 2)
        first15minClose = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 09:29:00")).select(df_15min_temp.Close).collect()[0].Close, 2)
    
    # First 30mins
    if df_15min_temp.count() == 25:
        min30_filter = df_15min_temp.collect()[0:2]
        first30minHigh = [ ele.__getattr__("High") for ele in min30_filter]
        first30minHigh.sort(reverse=True)
        first30minHigh = round(list(map(first30minHigh.__getitem__, [0]))[0], 2)
        first30minLow = [ ele.__getattr__("Low") for ele in min30_filter]
        first30minLow.sort()
        first30minLow = round(list(map(first30minLow.__getitem__, [0]))[0], 2)
        first30minOpen = [ ele.__getattr__("Open") for ele in min30_filter]
        #first30minOpen.sort()
        first30minOpen = round(list(map(first30minOpen.__getitem__, [0]))[0], 2)
        first30minClose = [ ele.__getattr__("Close") for ele in min30_filter]
        #first30minClose.sort()
        first30minClose = round(list(map(first30minClose.__getitem__, [1]))[0], 2)
    else:
        continue
        first30minHigh = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 09:44:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High, 2)
        first30minLow = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 09:44:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low, 2)
        first30minOpen = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 09:44:00")).sort(df_15min_temp.Timestamp.asc()).select(df_15min_temp.Open).collect()[0].Open, 2)
        first30minClose = round(df_15min_temp.filter((df_15min_temp.Timestamp <= date_value +" 09:44:00")).sort(df_15min_temp.Timestamp.desc()).select(df_15min_temp.Close).collect()[0].Close, 2)
    
    # Second 15mins
    if df_15min_temp.count() == 25:
        min15_2_filter = df_15min_temp.collect()[1]
        second15minHigh = round(getattr(min15_2_filter, "High"), 2)
        second15minLow = round(getattr(min15_2_filter, "Low"), 2)
        second15minOpen = round(getattr(min15_2_filter, "Open"), 2)
        second15minClose = round(getattr(min15_2_filter, "Close"), 2)
    else:
        continue
        second15minHigh = round(df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 09:30:00") & (df_15min_temp.Timestamp <= date_value +" 09:44:00")).select(df_15min_temp.High).collect()[0].High, 2)
        second15minLow = round(df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 09:30:00") & (df_15min_temp.Timestamp <= date_value +" 09:44:00")).select(df_15min_temp.Low).collect()[0].Low, 2)
        second15minOpen = round(df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 09:30:00") & (df_15min_temp.Timestamp <= date_value +" 09:44:00")).select(df_15min_temp.Open).collect()[0].Open, 2)
        second15minClose = round(df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 09:30:00") & (df_15min_temp.Timestamp <= date_value +" 09:44:00")).select(df_15min_temp.Close).collect()[0].Close, 2)
    
    # Second 30mins
    if df_15min_temp.count() == 25:
        min30_2_filter = df_15min_temp.collect()[2:4]
        second30minHigh = [ ele.__getattr__("High") for ele in min30_2_filter]
        second30minHigh.sort(reverse=True)
        second30minHigh = round(list(map(second30minHigh.__getitem__, [0]))[0], 2)
        second30minLow = [ ele.__getattr__("Low") for ele in min30_2_filter]
        second30minLow.sort()
        second30minLow = round(list(map(second30minLow.__getitem__, [0]))[0], 2)
        second30minOpen = [ ele.__getattr__("Open") for ele in min30_2_filter]
        #second30minOpen.sort()
        second30minOpen = round(list(map(second30minOpen.__getitem__, [0]))[0], 2)
        second30minClose = [ ele.__getattr__("Close") for ele in min30_2_filter]
        #second30minClose.sort()
        second30minClose = round(list(map(second30minClose.__getitem__, [1]))[0], 2)
    else:
        continue
        second30minHigh = round(df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 09:45:00") & (df_15min_temp.Timestamp <= date_value +" 10:14:00")).select(df_15min_temp.High).sort(df_15min_temp.High.desc()).collect()[0].High, 2)
        second30minLow = round(df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 09:45:00") & (df_15min_temp.Timestamp <= date_value +" 10:14:00")).select(df_15min_temp.Low).sort(df_15min_temp.Low.asc()).collect()[0].Low, 2)
        second30minOpen = round(df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 09:45:00") & (df_15min_temp.Timestamp <= date_value +" 10:14:00")).sort(df_15min_temp.Timestamp.asc()).select(df_15min_temp.Open).collect()[0].Open, 2)
        second30minClose = round(df_15min_temp.filter((df_15min_temp.Timestamp >= date_value +" 09:45:00") & (df_15min_temp.Timestamp <= date_value +" 10:14:00")).sort(df_15min_temp.Timestamp.desc()).select(df_15min_temp.Close).collect()[0].Close, 2)
    #print("Day levels calculated!")
    
    '''Calculate the Extremes, RE, Value Shift and Market Sentiment'''
    # Extremes
    buyExtremeCount = calculateExtremes(x_j)
    shortExtremeCount = calculateExtremes(list(reversed(x_j)))
    netExtremeCount = buyExtremeCount - shortExtremeCount
    
    buyExtremePresent = 1 if buyExtremeCount >= 2 else 0
    shortExtremePresent = 1 if shortExtremeCount >= 2 else 0  
    extremePresent = extremePresentCheck(buyExtremePresent, shortExtremePresent, netExtremeCount)
    
    # Range Extension
    buyRangeExtension = 1 if dayHigh >= IBTarget1High else 0
    shortRangeExtension = 1 if dayLow <= IBTarget1Low else 0
    rangeExtension = rangeExtensionCheck(buyRangeExtension, shortRangeExtension)
    
    # TPO Count
    buyTPOCount = sum(x_j[:maxJ])
    shortTPOCount = sum(x_j[maxJ + 1:])
    tpoCount = tpoCountCheck(buyTPOCount, shortTPOCount)
    
    # Value Shift
    # 1 -> Positive, 2 -> Negative, 3 -> Neutral
    vah = round(baseY + (up_idx * Den),2)
    val = round(baseY + (down_idx * Den),2)
    poc = round(baseY + (maxJ * Den),2)
    
    if i != 0:
        vah_prev = getattr(prev_day_filter, "VAH")
        val_prev = getattr(prev_day_filter, "VAL")
        poc_prev = getattr(prev_day_filter, "POC")
    else:
        vah_prev = 999999
        val_prev = 0
        poc_prev = 999999
        
    valueShift = 1 if (vah >= vah_prev * 1.005 or val >= val_prev * 1.005) and poc > poc_prev else 2 if (val <= val_prev * 0.995 or vah <= vah_prev * 0.995) and poc < poc_prev else 3
    
    # Market Sentiment
    marketSentiment = marketSentimentCheck(extremePresent, rangeExtension, tpoCount, valueShift)
    #print("MP levels calculated!")
    
    '''Open Location and Open Conviction'''
    openLocation = openLocationCheck(todayOpen, yestHigh, yestLow, vah_prev, val_prev)
    
    # Open Conviction - 5mins
    Buying_OD_5 = first5minOpen == first5minLow and first5minClose >= ((first5minHigh - first5minLow) * 0.7) + first5minLow and first5minClose > first5minOpen
    Buying_OTD_5 = first5minOpen != first5minLow and first5minClose >= ((first5minHigh - first5minLow) * 0.7) + first5minLow and first5minClose > first5minOpen and first5minLow > yestHigh * 0.999
    Buying_ORR_5 = first5minClose <= ((first5minHigh - first5minLow) * 0.3) + first5minLow and first5minOpen > first5minClose
    Selling_OD_5 = first5minOpen == first5minHigh and first5minClose <= (first5minHigh - ((first5minHigh - first5minLow) * 0.7)) and first5minClose < first5minOpen 
    Selling_OTD_5 = first5minOpen != first5minHigh and first5minClose <= (first5minHigh - ((first5minHigh - first5minLow) * 0.7)) and first5minClose < first5minOpen and first5minHigh < yestLow * 1.001
    Selling_ORR_5 = first5minClose >= ((first5minHigh - first5minLow) * 0.7) + first5minLow and first5minOpen < first5minClose
    openConviction_5 = openConviction_5minCheck(Buying_OD_5, Buying_OTD_5, Buying_ORR_5, Selling_OD_5, Selling_OTD_5, Selling_ORR_5)
    
    # Open Conviction - 15mins
    Buying_OD_15 = first15minOpen == first15minLow and first15minClose >= ((first15minHigh - first15minLow) * 0.7) + first15minLow and first15minClose > first15minOpen
    Buying_OTD_15 = first15minOpen != first15minLow and first15minClose >= ((first15minHigh - first15minLow) * 0.7) + first15minLow and first15minClose > first15minOpen and first15minLow > yestHigh
    Buying_ORR_15 = first15minOpen > first15minClose or (first15minClose <= ((first15minHigh - first15minLow) * 0.7) + first15minLow and first15minClose > first15minOpen) or first15minLow <= yestHigh
    Selling_OD_15 = first15minOpen == first15minHigh and first15minClose <= (first15minHigh - ((first15minHigh - first15minLow) * 0.7)) and first15minClose < first15minOpen 
    Selling_OTD_15 = first15minOpen != first15minHigh and first15minClose <= (first15minHigh - ((first15minHigh - first15minLow) * 0.7)) and first15minClose < first15minOpen and first15minHigh < yestLow
    Selling_ORR_15 = first15minOpen < first15minClose or (first15minClose >= (first15minHigh - ((first15minHigh - first15minLow) * 0.7)) and first15minClose < first15minOpen) or first15minHigh >= yestLow
    openConviction_15 = openConviction_15minCheck(Buying_OD_15, Buying_OTD_15, Buying_ORR_15, Selling_OD_15, Selling_OTD_15, Selling_ORR_15)
    
    # Open Conviction - 30mins
    Uncertain = (first15minClose > first15minOpen and second15minOpen > second15minClose) or (first15minClose < first15minOpen and second15minOpen < second15minClose)
    StrongUp = (first30minOpen == first30minLow and first30minClose >= ((first30minHigh - first30minLow) * 0.7) + first30minLow and first30minClose > first30minOpen and second30minLow >= (first30minHigh - first30minLow) * 0.5 + first30minLow) or (first30minOpen == first30minLow and first15minClose > first15minOpen and second15minClose > second15minOpen and second30minLow >= (first30minHigh - first30minLow) * 0.5 + first30minLow)
    ModerateUp = (first30minOpen <= ((first30minHigh - first30minLow) * 0.2) + first30minLow and first30minClose >= ((first30minHigh - first30minLow) * 0.7) + first30minLow and first30minClose > first30minOpen and second30minLow >= (first30minHigh - first30minLow) * 0.5 + first30minLow) or (first30minOpen <= ((first30minHigh - first30minLow) * 0.2) + first30minLow and first15minClose > first15minOpen and second15minClose > second15minOpen and second30minLow >= (first30minHigh - first30minLow) * 0.5 + first30minLow)
    StrongDown = (first30minOpen == first30minHigh and first30minClose <= first30minHigh - ((first30minHigh - first30minLow) * 0.7) and first30minClose < first30minOpen and second30minHigh <= first30minHigh - ((first30minHigh - first30minLow) * 0.5)) or (first30minOpen == first30minHigh and first15minClose < first15minOpen and second15minClose < second15minOpen and second30minHigh <= first30minHigh - ((first30minHigh - first30minLow) * 0.5))
    ModerateDown = (first30minOpen >= first30minHigh - ((first30minHigh - first30minLow) * 0.2) and first30minClose <= first30minHigh - ((first30minHigh - first30minLow) * 0.7) and first30minClose < first30minOpen and second30minHigh <= first30minHigh - ((first30minHigh - first30minLow) * 0.5)) or (first30minOpen >= first30minHigh - ((first30minHigh - first30minLow) * 0.2) and first15minClose < first15minOpen and second15minClose < second15minOpen and second30minHigh <= first30minHigh - ((first30minHigh - first30minLow) * 0.5))
    openConviction_30 = openConviction_30minCheck(StrongUp, ModerateUp, StrongDown, ModerateDown)
    
    # Open Conviction - 5mins_2
    Buying_OD_5_2 = first5minOpen == first5minLow and first5minClose >= ((first5minHigh - first5minLow) * 0.7) + first5minLow and first5minClose > first5minOpen
    Buying_OTD_5_2 = first5minOpen != first5minLow and first5minClose >= ((first5minHigh - first5minLow) * 0.7) + first5minLow and first5minClose > first5minOpen and first5minLow > vah_prev * 0.999
    Buying_ORR_5_2 = first5minClose <= ((first5minHigh - first5minLow) * 0.3) + first5minLow and first5minOpen > first5minClose
    Selling_OD_5_2 = first5minOpen == first5minHigh and first5minClose <= first5minHigh - ((first5minHigh - first5minLow) * 0.7) and first5minClose < first5minOpen 
    Selling_OTD_5_2 = first5minOpen != first5minHigh and first5minClose <= first5minHigh - ((first5minHigh - first5minLow) * 0.7) and first5minClose < first5minOpen and first5minHigh < val_prev * 1.001
    Selling_ORR_5_2 = first5minClose >= ((first5minHigh - first5minLow) * 0.7) + first5minLow and first5minOpen < first5minClose
    openConviction_5_2 = openConviction_5min_2Check(Buying_OD_5_2, Buying_OTD_5_2, Buying_ORR_5_2, Selling_OD_5_2, Selling_OTD_5_2, Selling_ORR_5_2)
    #print("Open convictions calculated!")
    
    '''Insert into reference table'''
    df_output_row = spark.createDataFrame([
        Row(Stock_Name=symbol, Date=date_value, Den=Den, 
            VAH=vah, VAL=val, POC=poc,
            IBH=IBH , IBL=IBL, IBType=IBType, OpenLocation=openLocation,
            OpenConviction_5=openConviction_5, OpenConviction_15=openConviction_15,
            OpenConviction_30=openConviction_30, OpenConviction_5_2=openConviction_5_2,
            First5_Open=first5minOpen, First5_High=first5minHigh, First5_Low=first5minLow,First5_Close=first5minClose,
            First15_Open=first15minOpen, First15_High=first15minHigh, First15_Low=first15minLow, First15_Close=first15minClose, 
            First30_Open=first30minOpen, First30_High=first30minHigh ,First30_Low=first30minLow , First30_Close=first30minClose,
            Second15_Open=second15minOpen ,Second15_High=second15minHigh, Second15_Low=second15minLow ,Second15_Close=second15minClose ,
            Second30_Open=second30minOpen, Second30_High=second30minHigh ,Second30_Low=second30minLow ,Second30_Close=second30minClose ,
            SP_Present=0,
            Extreme_Buy_Present=buyExtremePresent, Extreme_Buy_Count=buyExtremeCount,Extreme_Short_Present=shortExtremePresent, Extreme_Short_Count=shortExtremeCount,Extreme_Present=extremePresent, Extreme_Count=netExtremeCount,
            RE_Present=rangeExtension, TPO_Buy_Count=buyTPOCount, TPO_Short_Count=shortTPOCount, TPO_Count=tpoCount, 
            Value_Shift=valueShift, Market_Sentiment=marketSentiment, DayRange=todayRange,
            ATR=ATR)  ])    
    df_ref_levels = df_output_row
    date_value_prev = date_value
    #df_ref_levels = unionAll([df_ref_levels, df_output_row])
    #df_ref_levels = df_ref_levels.filter((df_ref_levels.Stock_Name == symbol) & (df_ref_levels.Date == date_value))
    df_ref_levels_to_be_appended = unionAll([df_ref_levels_to_be_appended, df_output_row])
    #print("Row Finished!")
    #print("df_ref_levels: " + str(df_ref_levels.count()))
    #print("df_ref_levels_to_be_appended: " + str(df_ref_levels_to_be_appended.count()))
    #print("df_15min_temp: " + str(df_15min_temp.count()))
    #print("day_filter: " + str(len(day_filter)))
    #print("prev_day_filter: " + str(len(prev_day_filter)))
    #print("j_array: " + str(j_array.count()))
    #print("x_j: " + str(len(x_j)))
    #print("j_bins: " + str(len(j_bins)))
    x_j.clear()
    j_bin_list.clear()
    j_idx.clear()
    data_list.clear()
    a_period.clear()
    b_period.clear()
    c_period.clear()
    d_period.clear()
    e_period.clear()
    f_period.clear()
    g_period.clear()
    h_period.clear()
    i_period.clear()
    j_period.clear()
    k_period.clear()
    l_period.clear()
    m_period.clear()
        
    '''Write to the reference levels table and flush the append table'''
    if df_ref_levels_to_be_appended.count() == 10 or i == (endDate - startDate).days - 1:  
        df_ref_levels_to_be_appended.write.jdbc(url=jdbcUrl,table="[dbo].[historical_daily_reference_levels]",properties = connectionProperties,mode="append")    
        df_ref_levels_to_be_appended = df_ref_levels_to_be_appended.filter(df_ref_levels_to_be_appended.Stock_Name == "NA")
    gc.collect()
