# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:10:04 2020

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

data = pd.read_csv('data_cleaned_final.csv')

crs_dep = data[['CRS_DEP_DATETIME_EST','CRS_DEP_DATE','CRS_DEP_HOUR','Marketing_Airline_Network','Origin','OriginCityName','Dest','DestCityName','ArrDelayMinutes','DepDelayMinutes']]
act_dep = data[['ACT_DEP_DATETIME_EST','ACT_DEP_DATE','ACT_DEP_HOUR','Marketing_Airline_Network','Origin','OriginCityName','Dest','DestCityName','ArrDelayMinutes','DepDelayMinutes']]
crs_arr = data[['CRS_ARR_DATETIME_EST','CRS_ARR_DATE','CRS_ARR_HOUR','Marketing_Airline_Network','Origin','OriginCityName','Dest','DestCityName','ArrDelayMinutes','DepDelayMinutes']]
act_arr = data[['ACT_ARR_DATETIME_EST','ACT_ARR_DATE','ACT_ARR_HOUR','Marketing_Airline_Network','Origin','OriginCityName','Dest','DestCityName','ArrDelayMinutes','DepDelayMinutes']]

#makes csv of each feature
crs_dep.to_csv('crs_dep.csv')
act_dep.to_csv('act_dep.csv')
crs_arr.to_csv('crs_arr.csv')
act_arr.to_csv('act_arr.csv')

act_arr.head()

