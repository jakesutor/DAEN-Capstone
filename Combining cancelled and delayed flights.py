# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:02:42 2020

@author: jakes
"""
import pandas as pd
import numpy as np
import datetime as dt
import os
os.chdir(r'C:\Users\jakes\Downloads')

# Load in data
data = pd.read_csv('final_data.csv')

# Load in cancelled data
cancelled_data = pd.read_csv('final_data_cancelled.csv')
cancelled_data.columns
##########################################################################
################# Make each cancelled flight equal 15 min delay ##########
##########################################################################
cancelled_data['ArrDelayCANCELLED'] = cancelled_data['ArrDelayCANCELLED']*15

cancelled_data['DepDelayCANCELLED'] = cancelled_data['DepDelayCANCELLED']*15


# Merge datasets
final_df = pd.merge(data, cancelled_data,  how='left', left_on=['Marketing_Airline_Network','ACT_ARR_DATE','ACT_ARR_HOUR'], right_on = ['Marketing_Airline_Network','CRS_ARR_DATE','CRS_ARR_HOUR'])
final_df = final_df[['Marketing_Airline_Network','ACT_ARR_DATE','ACT_ARR_HOUR','ArrDelayMinutes','DepDelayMinutes','ArrDelayCANCELLED','DepDelayCANCELLED','Weekday_x']]
final_df.columns = ['Marketing_Airline_Network','ACT_ARR_DATE','ACT_ARR_HOUR','ArrDelayMinutes','DepDelayMinutes','ArrDelayCANCELLED','DepDelayCANCELLED','Weekday']
final_df = final_df.fillna(method='ffill')

# The first few Dep delay cancelled values are still NA since none are before it so we'll set these to 0
final_df = final_df.fillna(0)

# Add the datasets together
final_df['ArrDelayMinutes'] = final_df['ArrDelayMinutes'] + final_df['ArrDelayCANCELLED']
final_df['DepDelayMinutes'] = final_df['DepDelayMinutes'] + final_df['DepDelayCANCELLED']

final_df = final_df[['Marketing_Airline_Network','ACT_ARR_DATE','ACT_ARR_HOUR','ArrDelayMinutes','DepDelayMinutes','Weekday']]

# Write CSV
final_df.to_csv('combined_final_data.csv')

# WN only
final_df_WN = final_df[final_df['Marketing_Airline_Network']=='WN']
final_df_WN.to_csv('combined_final_data_WN.csv')



