#This is the SECOND script to run
#This script converts the time zone
#import libraries
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


#grabs the time zone of origin and destination
###### Below are the functions that we will call later ######
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

def time_change_act_dep(data):
    #change time to standard format first
    hour = str(int(data['DEP_TIME']//100))
    if len(hour) < 2:
        hour = str(0) + hour
    if hour == '24':
        hour = '23'
        mint = '59'
    else:
        mint = str(int(data['DEP_TIME']%100))
        if len(mint) < 2:
            mint = str(0) + mint
    if (data['DEP_TIME'] >= data['CRS_DEP_TIME'] and data['DEP_DELAY_NEW'] <= 1440):
        naive_datetime = dt.datetime.strptime(data['FL_DATE'] + ' ' + hour + mint, "%Y-%m-%d %H%M") #"%Y-%m-%d %H%M"
    elif (data['DEP_TIME'] < data['CRS_DEP_TIME'] and data['DEP_DELAY_NEW'] == 0):
        naive_datetime = dt.datetime.strptime(data['FL_DATE'] + ' ' + hour + mint, "%Y-%m-%d %H%M")
    else:
        naive_datetime = dt.datetime.strptime(data['FL_DATE'] + ' ' + hour + mint, "%Y-%m-%d %H%M") + dt.timedelta(days=1)
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

def time_change_act_arr(data):
    #change time to standard format first
    hour = str(int(data['ARR_TIME']//100))
    if len(hour) < 2:
        hour = str(0) + hour
    if hour == '24':
        hour = '23'
        mint = '59'
    else:
        mint = str(int(data['ARR_TIME']%100))
        if len(mint) < 2:
            mint = str(0) + mint
    naive_datetime = dt.datetime.strptime(data['FL_DATE'] + ' ' + hour + mint, "%Y-%m-%d %H%M")
    #change the local times (CRS_DEP_TIME) to UTC
    local_time = pytz.timezone(data['DEST_CITY_TIMEZONE'])
    local_datetime = local_time.localize(naive_datetime, is_dst=True)
    eastern = pytz.timezone('US/Eastern')
    eastern_datetime = local_datetime.astimezone(eastern)
    if (eastern_datetime < data['CRS_ARR_DATETIME_EST'] and data['ARR_DELAY_NEW'] > 0):
        eastern_datetime = eastern_datetime + dt.timedelta(days=1)
    return eastern_datetime

# loading data that was combined in first file
# Unicode_escape and bad lines address any issues with file conversion due to the size of the file
data = pd.read_csv('data.csv',encoding= 'unicode_escape',error_bad_lines=False)

# cleaning data - remove outliers that are sitting and taxiing for a long time
data = data[(data['TAXI_IN'] <= 200) & (data['TAXI_OUT'] <= 200) &
            ((data['WHEELS_ON'] - data['WHEELS_OFF']) -
             (data['CRS_ARR_TIME'] - data['CRS_DEP_TIME'])<= 100)]
len(data.index) # To find the number of rows after removing these outliers

# changing local times
tqdm.pandas() # This allows us to use the progress_apply function
origin_city_timezone = pd.DataFrame(data['ORIGIN_CITY_NAME'].unique(),columns=['ORIGIN_CITY_NAME'])
origin_city_timezone['ORIGIN_CITY_TIMEZONE'] = origin_city_timezone.progress_apply(get_timezone_origin,axis=1)
dest_city_timezone = pd.DataFrame(data['DEST_CITY_NAME'].unique(),columns=['DEST_CITY_NAME'])
dest_city_timezone['DEST_CITY_TIMEZONE'] = dest_city_timezone.progress_apply(get_timezone_dest,axis=1)

# combine ORIGIN_CITY_TIMEZONE and DEST_CITY_TIMEZONE and data
data = pd.merge(data,origin_city_timezone, on = ['ORIGIN_CITY_NAME'], how='left')
data = pd.merge(data,dest_city_timezone, on = ['DEST_CITY_NAME'], how='left')

# drop na values in ORIGIN_CITY_NAME and DEST_CITY_TIMEZONE
data = data.dropna(subset=['ORIGIN_CITY_TIMEZONE','DEST_CITY_TIMEZONE'])
len(data.index)
data['DEP_TIME'] = data['DEP_TIME'].astype('int64') 

# chage data and time to Eastern Time
tqdm.pandas()
data['CRS_DEP_DATETIME_EST'] = data.progress_apply(time_change_crs_dep,axis=1)
data['ACT_DEP_DATETIME_EST'] = data.progress_apply(time_change_act_dep,axis=1)
data['CRS_ARR_DATETIME_EST'] = data.progress_apply(time_change_crs_arr,axis=1)
data['ACT_ARR_DATETIME_EST'] = data.progress_apply(time_change_act_arr,axis=1)

# EoDTD 
# add the CRS departure time of the flights to the data (0: CRS_DEP_TIME is 00:00-00:59,..., 23:CRS_DEP_TIME is 23:00-23:59)
def get_hour_crs_dep(data):
    return data['CRS_DEP_DATETIME_EST'].hour

def get_hour_act_dep(data):
    return data['ACT_DEP_DATETIME_EST'].hour

def get_hour_crs_arr(data):
    return data['CRS_ARR_DATETIME_EST'].hour

def get_hour_act_arr(data):
    return data['ACT_ARR_DATETIME_EST'].hour

# Apply the above functions
data['CRS_DEP_HOUR'] = data.progress_apply(get_hour_crs_dep,axis=1)
data['ACT_DEP_HOUR'] = data.progress_apply(get_hour_act_dep,axis=1)
data['CRS_ARR_HOUR'] = data.progress_apply(get_hour_crs_arr,axis=1)
data['ACT_ARR_HOUR'] = data.progress_apply(get_hour_act_arr,axis=1)

# Memory Error - had to split up the data to clean in next file. 
# By splitting up the data we will not need to use as much memory when we read these files in
data.iloc[:500000,].head()
len(data.index) # First we need to have an idea of how many total rows there are
data_1 = data.iloc[:5000000,]
data_1.to_csv('data_cleaned.csv')

data_2 = data.iloc[5000001:10000000,]
data_2.to_csv('data_2_cleaned.csv')

data_3 = data.iloc[10000001:,]
data_3.to_csv('data_3_cleaned.csv')
