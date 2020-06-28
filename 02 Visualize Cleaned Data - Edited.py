#This is the the THIRD FILE to run
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime as dt
import pytz
import os
os.chdir(r'C:\Users\jakes\Downloads')

#read csv files
act_dep = pd.read_csv('act_dep.csv')
act_arr = pd.read_csv('act_arr.csv')

# drop records for years 2017 and 2020 which were created because of changing times to EST
act_dep.drop(act_dep[act_dep['ACT_DEP_DATE'] == '2017-12-31'].index, inplace=True)
act_dep.drop(act_dep[act_dep['ACT_DEP_DATE'] == '2020-03-01'].index, inplace=True)
act_arr.drop(act_arr[act_arr['ACT_ARR_DATE'] == '2017-12-31'].index, inplace=True)
act_arr.drop(act_arr[act_arr['ACT_ARR_DATE'] == '2020-03-01'].index, inplace=True)


dep_delay = act_dep.groupby(['Marketing_Airline_Network','ACT_DEP_DATE','ACT_DEP_HOUR'])['DepDelayMinutes'].sum().unstack(2).stack()  # ARR_DELAY_NEW is arrival delays that are positive
arr_delay = act_arr.groupby(['Marketing_Airline_Network','ACT_ARR_DATE','ACT_ARR_HOUR'])['ArrDelayMinutes'].sum().unstack(2).stack()  # ARR_DELAY_NEW is arrival delays that are positive

# cumulative delays by airline, date, and hour
dep_delay = act_dep.groupby(['Marketing_Airline_Network','ACT_DEP_DATE','ACT_DEP_HOUR'])['DepDelayMinutes'].sum().unstack(2).stack().groupby(level=[0,1]).cumsum()  # ARR_DELAY_NEW is arrival delays that are positive
arr_delay = act_arr.groupby(['Marketing_Airline_Network','ACT_ARR_DATE','ACT_ARR_HOUR'])['ArrDelayMinutes'].sum().unstack(2).stack().groupby(level=[0,1]).cumsum()  # ARR_DELAY_NEW is arrival delays that are positive

# sort based on date and hour:
dep_delay = dep_delay.sort_index(level=(0,1))
arr_delay = arr_delay.sort_index(level=(0,1))
dep_delay = dep_delay.reset_index()
arr_delay = arr_delay.reset_index()

# Merge the arrival and departure delays datasets using a "LEFT JOIN"
final_df = pd.merge(arr_delay, dep_delay,  how='left', left_on=['Marketing_Airline_Network','ACT_ARR_DATE','ACT_ARR_HOUR'], right_on = ['Marketing_Airline_Network','ACT_DEP_DATE','ACT_DEP_HOUR'])
final_df = final_df[['Marketing_Airline_Network','ACT_ARR_DATE','ACT_ARR_HOUR','0_x','0_y']]


# Rename columns
final_df.columns = ['Marketing_Airline_Network','ACT_ARR_DATE','ACT_ARR_HOUR','ArrDelayMinutes','DepDelayMinutes']

# For any NAs, fill in with the previous value if the criteria match
final_df = final_df.fillna(method='ffill')

# Change data type to datetime
final_df['ACT_ARR_DATE'] = pd.to_datetime(final_df['ACT_ARR_DATE'])

# Create variable for the day of the week
final_df['Weekday'] = final_df['ACT_ARR_DATE'].dt.dayofweek

# Create new csv with final data
final_df.to_csv('final_data.csv')

# Create new csv with WN data
final_df_WN = final_df[final_df['Marketing_Airline_Network']=='WN']
final_df_WN.to_csv('final_data_WN.csv')


############## Previous Group's Visualizations ##########################
# plot delay for a sample day:
dep_delay['2020-01-10'].plot(label='Departure Delay')
arr_delay['2020-01-10'].plot(label='Arrival Delay')
plt.xlabel('Time',fontsize=12)
plt.ylabel("Total Delay (minutes)",fontsize=12)
plt.title('US Domestic Flights Delay for Jun 10, 2019')
plt.legend()
plt.xticks(range(24),range(24))


# plot delay for Jan 1st 2020:
arr_delay['2020-01-01'].plot(label='1-Jan')
plt.xlabel("ACTUAL ARRIVE Time",fontsize=12)
plt.ylabel("Total Delay (minutes)",fontsize=12)
plt.legend()
plt.xticks(range(24),range(24))

# plot delay for Jan 2nd 2020:
arr_delay['2020-01-02'].plot(label='2-Jan')
plt.xlabel("CRS Departure Time",fontsize=12)
plt.ylabel("Total Delay (minutes)",fontsize=12)
plt.legend()
plt.xticks(range(24),range(24))

# plot delay for Jan 3rd 2020:
arr_delay['2020-01-03'].plot(label='3-Jan')
plt.xlabel("CRS Departure Time",fontsize=12)
plt.ylabel("Hourly Arrival Delay (minutes)",fontsize=12)
plt.legend()
plt.xticks(range(24),range(24))

# delay for day in January 2020:
fig = plt.figure(figsize=(13,7))
for day in arr_delay.index.get_level_values(0).unique():
    arr_delay[day].plot()
plt.xlabel("HOUR",fontsize=12)
plt.ylabel("Hourly Arrival Delay (minutes)",fontsize=12)
plt.xticks(range(24),range(24))
plt.title("Arrival Delay for All Days in 2020")

fig = plt.figure()
for day in dep_delay.index.get_level_values(0).unique():
    dep_delay[day].plot()
plt.xlabel("ACTUAL DEPARTURE TIME",fontsize=12)
plt.ylabel("Total Delay (minutes)",fontsize=12)
plt.xticks(range(24),range(24))
plt.title("All days in 2020")

arr_delay.plot()
plt.xlabel("CRS Departure Time",fontsize=12)
plt.ylabel("Total Delay (minutes)",fontsize=12)
plt.xticks(rotation='vertical')
