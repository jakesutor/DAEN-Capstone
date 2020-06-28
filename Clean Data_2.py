# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:00:58 2020

@author: jakes
"""
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


# Using the split up data, we will now combine the data again and run the last few functions
data_1 = pd.read_csv('data_cleaned.csv')
data_2 = pd.read_csv('data_2_cleaned.csv')
data_3 = pd.read_csv('data_3_cleaned.csv')

frames=[data_1,data_2,data_3]
data = pd.concat(frames,ignore_index=True)

data = data[['FL_DATE','MKT_UNIQUE_CARRIER','ORIGIN','ORIGIN_CITY_NAME','DEST','DEST_CITY_NAME','CRS_DEP_TIME','DEP_TIME','DEP_DELAY_NEW','TAXI_OUT','WHEELS_OFF','WHEELS_ON','TAXI_IN','CRS_ARR_TIME','ARR_TIME','ARR_DELAY_NEW','CANCELLED','ORIGIN_CITY_TIMEZONE','DEST_CITY_TIMEZONE','CRS_DEP_DATETIME_EST','ACT_DEP_DATETIME_EST','CRS_ARR_DATETIME_EST','ACT_ARR_DATETIME_EST','CRS_DEP_HOUR','ACT_DEP_HOUR','CRS_ARR_HOUR','ACT_ARR_HOUR']]

data['CRS_DEP_DATETIME_EST'].head()

# Find the date from the time by taking the first 10 characters
data['CRS_DEP_DATE'] = data['CRS_DEP_DATETIME_EST'].str[:10]
data['ACT_DEP_DATE'] = data['ACT_DEP_DATETIME_EST'].str[:10]
data['CRS_ARR_DATE'] = data['CRS_ARR_DATETIME_EST'].str[:10]
data['ACT_ARR_DATE'] = data['ACT_ARR_DATETIME_EST'].str[:10]


#select features
data = data[['CRS_DEP_DATETIME_EST','CRS_DEP_DATE','CRS_DEP_HOUR','ACT_DEP_DATETIME_EST','ACT_DEP_DATE','ACT_DEP_HOUR','CRS_ARR_DATETIME_EST','CRS_ARR_DATE','CRS_ARR_HOUR','ACT_ARR_DATETIME_EST','ACT_ARR_DATE','ACT_ARR_HOUR','MKT_UNIQUE_CARRIER','ORIGIN','ORIGIN_CITY_NAME','DEST','DEST_CITY_NAME','ARR_DELAY_NEW','DEP_DELAY_NEW']]
data.columns = ['CRS_DEP_DATETIME_EST','CRS_DEP_DATE','CRS_DEP_HOUR','ACT_DEP_DATETIME_EST','ACT_DEP_DATE','ACT_DEP_HOUR','CRS_ARR_DATETIME_EST','CRS_ARR_DATE','CRS_ARR_HOUR','ACT_ARR_DATETIME_EST','ACT_ARR_DATE','ACT_ARR_HOUR','Marketing_Airline_Network','Origin','OriginCityName','Dest','DestCityName','ArrDelayMinutes','DepDelayMinutes']
data = data.dropna() # There are no NAs left already by this point

# Create cleaned final dataset
data.to_csv('data_cleaned_final.csv')

