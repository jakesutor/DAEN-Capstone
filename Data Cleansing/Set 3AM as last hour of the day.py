# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:55:16 2020

@author: jakes
"""
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import os
os.chdir(r'C:\Users\jakes\Downloads')

# Load in data
data = pd.read_csv('daily_&_cumulative_combined_final_data.csv')
data.columns
data = data[['Marketing_Airline_Network', 'ACT_ARR_DATE',
       'ACT_ARR_HOUR', 'ArrDelayMinutes', 'DepDelayMinutes',
       'CumulativeArrDelayMinutes', 'CumulativeDepDelayMinutes', 'Weekday']]

# Convert date to datetime 
data['ACT_ARR_DATE'] = pd.to_datetime(data['ACT_ARR_DATE'])

# Add variable to find four hours prior (resetting 3AM to be equal to 11PM)
data['Date_Time'] = data['ACT_ARR_DATE'] + pd.to_timedelta(data['ACT_ARR_HOUR'], unit='h')
four_hours = pd.Timedelta(hours=4)
data['Date'] = (data["Date_Time"] - four_hours)
data['Hour'] = data['Date'].dt.hour
data['Date'] = data['Date'].dt.date

# Group by new dates
cumulative_dep_cancelled = crs_dep.groupby(['Marketing_Airline_Network','CRS_DEP_DATE','CRS_DEP_HOUR'])['ArrDelayMinutes'].sum().unstack(2).stack().groupby(level=[0,1]).cumsum()  # ARR_DELAY_NEW is arrival delays that are positive

dep_df = data.groupby(['Marketing_Airline_Network','Date','Hour'])['DepDelayMinutes'].sum().unstack(2).stack().groupby(level=[0,1]).cumsum()
arr_df = data.groupby(['Marketing_Airline_Network','Date','Hour'])['ArrDelayMinutes'].sum().unstack(2).stack().groupby(level=[0,1]).cumsum()

# sort based on date and hour:
dep_df = dep_df.sort_index(level=(0,1))
arr_df = arr_df.sort_index(level=(0,1))
dep_df = dep_df.reset_index()
arr_df = arr_df.reset_index()


final_df = pd.merge(arr_df, dep_df,  how='left', left_on=['Marketing_Airline_Network','Date','Hour'], right_on = ['Marketing_Airline_Network','Date','Hour'])
final_df.columns = ['Marketing_Airline_Network','Date','Hour','ArrDelay3AM','DepDelay3AM']

result = pd.merge(data, final_df,  how='left', left_on=['Marketing_Airline_Network','Date','Hour'], right_on = ['Marketing_Airline_Network','Date','Hour'])

result.head(30)


# drop records for years 2017 which were created because of resetting the day
result['Date'] = result['Date'].astype(str)
result = result[result['Date']!='2017-12-31']
result.head(20)

# Select desired variables
result.reset_index()
result = result[['Marketing_Airline_Network', 'Date','Hour','ACT_ARR_DATE',
       'ACT_ARR_HOUR', 'ArrDelayMinutes', 'DepDelayMinutes',
       'CumulativeArrDelayMinutes', 'CumulativeDepDelayMinutes', 'ArrDelay3AM','DepDelay3AM',
       'Weekday']]

# Write CSV
result.to_csv('updated_final_data.csv')

# WN only
result_WN = result[result['Marketing_Airline_Network']=='WN']
result_WN.to_csv('updated_final_data_WN.csv')



