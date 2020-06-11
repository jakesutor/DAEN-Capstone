#This is the the THIRD FILE to run
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime as dt
import pytz

#read csv files
act_dep = pd.read_csv('act_dep.csv')
act_arr = pd.read_csv('act_arr.csv')

# drop records for years 2017 and 2020 which were created because of changing times to EST
act_dep.drop(act_dep[act_dep['ACT_DEP_DATE'] == '2017-12-31'].index, inplace=True)
act_dep.drop(act_dep[act_dep['ACT_DEP_DATE'] == '2020-01-01'].index, inplace=True)
act_arr.drop(act_arr[act_arr['ACT_ARR_DATE'] == '2017-12-31'].index, inplace=True)
act_arr.drop(act_arr[act_arr['ACT_ARR_DATE'] == '2020-01-01'].index, inplace=True)


dep_delay = act_dep.groupby(['ACT_DEP_DATE','ACT_DEP_HOUR'])['DepDelayMinutes'].sum().unstack(1).stack()  # ARR_DELAY_NEW is arrival delays that are positive
arr_delay = act_arr.groupby(['ACT_ARR_DATE','ACT_ARR_HOUR'])['ArrDelayMinutes'].sum().unstack(1).stack()  # ARR_DELAY_NEW is arrival delays that are positive

# cumulative
dep_delay = act_dep.groupby(['ACT_DEP_DATE','ACT_DEP_HOUR'])['DepDelayMinutes'].sum().unstack(1).stack().groupby(level=[0]).cumsum()  # ARR_DELAY_NEW is arrival delays that are positive
arr_delay = act_arr.groupby(['ACT_ARR_DATE','ACT_ARR_HOUR'])['ArrDelayMinutes'].sum().unstack(1).stack().groupby(level=[0]).cumsum()  # ARR_DELAY_NEW is arrival delays that are positive

# sort based on date and hour:
dep_delay = dep_delay.sort_index(level=(0,1))
arr_delay = arr_delay.sort_index(level=(0,1))



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
