# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 21:07:56 2020

@author: jakes
"""

#Import necessary libraries
import pandas as pd
import numpy as np
import datetime as dt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from timezonefinder import TimezoneFinder
import pytz
from tqdm import tqdm
import os
os.chdir(r'C:\Users\jakes\Downloads')

#Read in dataset for all flights
#df = pd.read_csv('On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2019_10.csv')  
df = pd.read_csv('data.csv',encoding= 'unicode_escape',error_bad_lines=False)
#Separate out only flights that were considered "Delayed" by the BTS
df2 = df[df["ARR_DELAY_NEW"]>=15]

#Extract the airport codes and arrival delays for all delayed flights in the dataset
airports = df2[["DEST","ARR_DELAY_NEW"]]

#Calculate the mean arrival delay of delayed flights for each unique airport
means=airports.groupby('DEST',as_index=False)['ARR_DELAY_NEW'].mean()
#Round the mean arrival delay to the nearest integer to match 
means.ARR_DELAY_NEW=means.ARR_DELAY_NEW.astype(float).round()
means.head()

#read csv files for all cancelled flights
data = pd.read_csv('cancelled_raw_data.csv')
data.head()
data = data[['FL_DATE','MKT_UNIQUE_CARRIER','ORIGIN','ORIGIN_CITY_NAME','DEST','DEST_CITY_NAME','CRS_DEP_TIME', 'CRS_ARR_TIME','CANCELLED']]


#Fill in the missing arrival delay values for the cancelled flights
# Create empty attribute ARR_DELAY_NEW
data['ARR_DELAY_NEW'] = 0
data['ARR_DELAY_NEW']=data['DEST'].map(dict(zip(means['DEST'],means['ARR_DELAY_NEW']))).fillna(data.ARR_DELAY_NEW)
#Verify it filled in correctly
data.ARR_DELAY_NEW

data.head()


# Use the same methodology to change time to EST
def get_timezone_origin(city):
    geolocator = Nominatim(user_agent='CATSR',timeout=10)
    locfinder = RateLimiter(geolocator.geocode, min_delay_seconds=0)
    location = locfinder(city['ORIGIN_CITY_NAME'])
    #location = geolocator.geocode(city['CITY_NAME'])
    if location != None:
        tf = TimezoneFinder()
        time_zone = tf.timezone_at(lng=location.longitude, lat=location.latitude)
    else:
        time_zone = np.nan
    return time_zone

def get_timezone_dest(city):
    geolocator = Nominatim(user_agent='CATSR',timeout=10)
    locfinder = RateLimiter(geolocator.geocode, min_delay_seconds=0)
    location = locfinder(city['DEST_CITY_NAME'])
    #location = geolocator.geocode(city['CITY_NAME'])
    if location != None:
        tf = TimezoneFinder()
        time_zone = tf.timezone_at(lng=location.longitude, lat=location.latitude)
    else:
        time_zone = np.nan
    return time_zone

def time_change_crs_dep(data):
    #change time to standard format first
    hour = str(int(data['CRS_DEP_TIME']//100))
    if len(hour) < 2:
        hour = str(0) + hour
    mint = str(int(data['CRS_DEP_TIME']%100))
    if len(mint) < 2:
        mint = str(0) + mint
    naive_datetime = dt.datetime.strptime(data['FL_DATE'] + ' ' + hour + mint, "%Y-%m-%d %H%M") #"%Y-%m-%d %H%M"
    #change the local times (CRS_DEP_TIME) to UTC
    local_time = pytz.timezone(data['ORIGIN_CITY_TIMEZONE'])
    local_datetime = local_time.localize(naive_datetime, is_dst=True)
    eastern = pytz.timezone('US/Eastern')
    eastern_datetime = local_datetime.astimezone(eastern)
    return eastern_datetime

def time_change_crs_arr(data):
    #change time to standard format first
    hour = str(int(data['CRS_ARR_TIME']//100))
    if len(hour) < 2:
        hour = str(0) + hour
    if hour == '24':
        hour = '23'
        mint = '59'
    else:
        mint = str(int(data['CRS_ARR_TIME']%100))
        if len(mint) < 2:
            mint = str(0) + mint
    naive_datetime = dt.datetime.strptime(data['FL_DATE'] + ' ' + hour + mint, "%Y-%m-%d %H%M")
    #change the local times (CRS_DEP_TIME) to UTC
    local_time = pytz.timezone(data['DEST_CITY_TIMEZONE'])
    local_datetime = local_time.localize(naive_datetime, is_dst=True)
    eastern = pytz.timezone('US/Eastern')
    eastern_datetime = local_datetime.astimezone(eastern)
    if (eastern_datetime < data['CRS_DEP_DATETIME_EST']):
        eastern_datetime = eastern_datetime + dt.timedelta(days=1)
    return eastern_datetime

# changing local times using the above functions
tqdm.pandas()
origin_city_timezone = pd.DataFrame(data['ORIGIN_CITY_NAME'].unique(),columns=['ORIGIN_CITY_NAME'])
origin_city_timezone['ORIGIN_CITY_TIMEZONE'] = origin_city_timezone.progress_apply(get_timezone_origin,axis=1)
dest_city_timezone = pd.DataFrame(data['DEST_CITY_NAME'].unique(),columns=['DEST_CITY_NAME'])
dest_city_timezone['DEST_CITY_TIMEZONE'] = dest_city_timezone.progress_apply(get_timezone_dest,axis=1)

# combine ORIGIN_CITY_TIMEZONE and DEST_CITY_TIMEZONE and data
data = pd.merge(data,origin_city_timezone, on = ['ORIGIN_CITY_NAME'], how='left')
data = pd.merge(data,dest_city_timezone, on = ['DEST_CITY_NAME'], how='left')
len(data.index)
#288175
# drop na values in ORIGIN_CITY_NAME and DEST_CITY_TIMEZONE
data = data.dropna(subset=['ORIGIN_CITY_TIMEZONE','DEST_CITY_TIMEZONE'])
len(data.index)
#286292

# Apply the above functions
tqdm.pandas()
data['CRS_DEP_DATETIME_EST'] = data.progress_apply(time_change_crs_dep,axis=1)
data['CRS_ARR_DATETIME_EST'] = data.progress_apply(time_change_crs_arr,axis=1)

# EoDTD
# add the CRS departure time of the flights to the data (0: CRS_DEP_TIME is 00:00-00:59,..., 23:CRS_DEP_TIME is 23:00-23:59)
def get_hour_crs_dep(data):
    return data['CRS_DEP_DATETIME_EST'].hour

def get_hour_crs_arr(data):
    return data['CRS_ARR_DATETIME_EST'].hour

data['CRS_DEP_HOUR'] = data.progress_apply(get_hour_crs_dep,axis=1)
data['CRS_ARR_HOUR'] = data.progress_apply(get_hour_crs_arr,axis=1)

def get_date_crs_dep(data):
    return data['CRS_DEP_DATETIME_EST'].strftime("%Y-%m-%d")


def get_date_crs_arr(data):
    return data['CRS_ARR_DATETIME_EST'].strftime("%Y-%m-%d")

data['CRS_DEP_DATE'] = data.progress_apply(get_date_crs_dep,axis=1)
data['CRS_ARR_DATE'] = data.progress_apply(get_date_crs_arr,axis=1)

# Create dataframe with the desired column names
data = data[['CRS_DEP_DATETIME_EST','CRS_DEP_DATE','CRS_DEP_HOUR','CRS_ARR_DATETIME_EST','CRS_ARR_DATE','CRS_ARR_HOUR','MKT_UNIQUE_CARRIER','ORIGIN','ORIGIN_CITY_NAME','DEST','DEST_CITY_NAME','CANCELLED','ARR_DELAY_NEW']]
data.columns = ['CRS_DEP_DATETIME_EST','CRS_DEP_DATE','CRS_DEP_HOUR','CRS_ARR_DATETIME_EST','CRS_ARR_DATE','CRS_ARR_HOUR','Marketing_Airline_Network','Origin','OriginCityName','Dest','DestCityName','CANCELLED','ArrDelayMinutes']

# Create csv from dataframe
data.to_csv('cleaned_cancelled_data.csv')

# Divide into departure and arrival datasets
crs_dep = data[['CRS_DEP_DATETIME_EST','CRS_DEP_DATE','CRS_DEP_HOUR','Marketing_Airline_Network','Origin','OriginCityName','Dest','DestCityName','CANCELLED','ArrDelayMinutes']]
crs_arr = data[['CRS_ARR_DATETIME_EST','CRS_ARR_DATE','CRS_ARR_HOUR','Marketing_Airline_Network','Origin','OriginCityName','Dest','DestCityName','CANCELLED','ArrDelayMinutes']]

#makes csv of each feature
crs_dep.to_csv('crs_dep_cancelled.csv')
crs_arr.to_csv('crs_arr_cancelled.csv')

#read csv files
crs_dep = pd.read_csv('crs_dep_cancelled.csv')
crs_arr = pd.read_csv('crs_arr_cancelled.csv')

# drop records for years 2017 and 2020 which were created because of changing times to EST
crs_dep.drop(crs_dep[crs_dep['CRS_DEP_DATE'] == '2017-12-31'].index, inplace=True)
crs_dep.drop(crs_dep[crs_dep['CRS_DEP_DATE'] == '2020-03-01'].index, inplace=True)
crs_arr.drop(crs_arr[crs_arr['CRS_ARR_DATE'] == '2017-12-31'].index, inplace=True)
crs_arr.drop(crs_arr[crs_arr['CRS_ARR_DATE'] == '2020-03-01'].index, inplace=True)


dep_cancelled = crs_dep.groupby(['Marketing_Airline_Network','CRS_DEP_DATE','CRS_DEP_HOUR'])['ArrDelayMinutes'].sum().unstack(2).stack()  # ARR_DELAY_NEW is arrival delays that are positive
arr_cancelled = crs_arr.groupby(['Marketing_Airline_Network','CRS_ARR_DATE','CRS_ARR_HOUR'])['ArrDelayMinutes'].sum().unstack(2).stack()  # ARR_DELAY_NEW is arrival delays that are positive

# cumulative count of cancelled flights
dep_cancelled = crs_dep.groupby(['Marketing_Airline_Network','CRS_DEP_DATE','CRS_DEP_HOUR'])['ArrDelayMinutes'].sum().unstack(2).stack().groupby(level=[0,1]).cumsum()  # ARR_DELAY_NEW is arrival delays that are positive
arr_cancelled = crs_arr.groupby(['Marketing_Airline_Network','CRS_ARR_DATE','CRS_ARR_HOUR'])['ArrDelayMinutes'].sum().unstack(2).stack().groupby(level=[0,1]).cumsum()  # ARR_DELAY_NEW is arrival delays that are positive

# sort based on date and hour:
dep_cancelled = dep_cancelled.sort_index(level=(0,1))
arr_cancelled = arr_cancelled.sort_index(level=(0,1))
dep_cancelled = dep_cancelled.reset_index()
arr_cancelled = arr_cancelled.reset_index()

# Merge datasets together
final_df = pd.merge(arr_cancelled, dep_cancelled,  how='left', left_on=['Marketing_Airline_Network','CRS_ARR_DATE','CRS_ARR_HOUR'], right_on = ['Marketing_Airline_Network','CRS_DEP_DATE','CRS_DEP_HOUR'])
final_df = final_df[['Marketing_Airline_Network','CRS_ARR_DATE','CRS_ARR_HOUR','0_x','0_y']]

final_df.columns = ['Marketing_Airline_Network','CRS_ARR_DATE','CRS_ARR_HOUR','ArrDelayCANCELLED','DepDelayCANCELLED']
# Fill in NAs using the same methodology
final_df = final_df.fillna(method='ffill')
final_df['CRS_ARR_DATE'] = pd.to_datetime(final_df['CRS_ARR_DATE'])
final_df['Weekday'] = final_df['CRS_ARR_DATE'].dt.dayofweek # Add weekday variable
final_df['DepDelayCANCELLED'] = round(final_df['DepDelayCANCELLED'],0)
final_df['ArrDelayCANCELLED'] = round(final_df['ArrDelayCANCELLED'],0)
final_df.to_csv('final_data_cancelled.csv')

final_df_WN = final_df[final_df['Marketing_Airline_Network']=='WN']
final_df_WN.to_csv('final_data_cancelled_WN.csv')


