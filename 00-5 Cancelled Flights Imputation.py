# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:54:47 2020

@author: Chas Cantrell
"""
#Import necessary libraries
import pandas as pd
import os
os.chdir(r'D:\DAEN690\Datasets 690')

#Read in dataset for all flights
#df = pd.read_csv('On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2019_10.csv')  
df = pd.read_csv('data.csv')

#Separate out only flights that were considered "Delayed" by the BTS
df2 = df[df["ARR_DEL15"]==1]
#Extract the airport codes and arrival delays for all delayed flights in the dataset
airports = df2[["DEST","ARR_DELAY"]]

#Calculate the mean arrival delay of delayed flights for each unique airport
means=airports.groupby('DEST',as_index=False)['ARR_DELAY'].mean()
#Round the mean arrival delay to the nearest integer to match 
means.ARR_DELAY=means.ARR_DELAY.astype(float).round()
means


#Separate out cancelled flights from non-cancelled flights
#Tried doing this all at once but it simply arrival delay for all flights, even the non-cancelled ones
cancelled = df[df["CANCELLED"]==1]
cancelled.ARR_DELAY
#Create a dataframe for non-cancelled flights
noncanc = df[df["CANCELLED"]==0]

#Fill in the missing arrival delay values for the cancelled flights
cancelled['ARR_DELAY']=cancelled['DEST'].map(dict(zip(means['DEST'],means['ARR_DELAY']))).fillna(cancelled.ARR_DELAY)
#Verify it filled in correctly
cancelled.ARR_DELAY
#noncanc.ARR_DELAY
#noncanc

#Add the cancelled flights back on to the non-cancelled flights
noncanc=noncanc.append(cancelled,ignore_index=True)
noncanc
noncanc.to_csv('data.csv')


