# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:45:52 2020

@author: jakes
"""

### This file creates a combined line chart for all ten airlines' delays ###

#import libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import holidays
import datetime as dt


# load data
data_full = pd.read_csv('final_final_data.csv')
airlines_cv = pd.read_csv('airlines.csv')
airlines_cv['Marketing_Airline_Network'] = pd.Categorical(airlines_cv['Marketing_Airline_Network'], ["AS", "HA","DL","NK","AA","F9","B6","WN","UA","G4"])
airlines_cv = airlines_cv.sort_values("Marketing_Airline_Network")

# Convert hours to EST (i.e. 0 = 4AM EST)
data_full['Hour'] = data_full['Hour'] + 4
data_full['Hour'] = data_full['Hour'].astype(str)

data_full.loc[data_full['Hour']=='24','Hour'] = '0'
data_full.loc[data_full['Hour']=='25','Hour'] = '1'
data_full.loc[data_full['Hour']=='26','Hour'] = '2'
data_full.loc[data_full['Hour']=='27','Hour'] = '3'

# Set up major plot
figure, axes = plt.subplots(nrows=3, ncols=4,sharex=True,sharey=True,figsize=(24,18))
figure.suptitle('Cumulative Delays by Hour for Each Airline', fontsize=24)

# Set initial location of plot and labels
l = -1
r = 0
labels = ['4','','','','8','','','','12','','','','16','','','','20','','','','0','','','3']

# Iterate through airlines and combine as subplots
for j in airlines_cv['Marketing_Airline_Network'].unique():
    airline = j
    l = l+1
    if l < 4:
        l = l
    else:
        l = 0
        if r < 2:
            r = r+1
        else:
            l = l+1
            
    print(l,r)
    airline_name = airlines_cv[airlines_cv['Marketing_Airline_Network']==airline]['Airline'].reset_index()
    airline_name = airline_name['Airline'][0]

    data = data_full[data_full['Marketing_Airline_Network']==airline]
    
    data = data[['Date','Hour','ArrDelay3AM']]
    
    cutoff = max(data['ArrDelay3AM'])*0.7
    
    data1 = pd.DataFrame({'Date':[],'Hour':[],'ArrDelay3AM':[]})
    data2 = pd.DataFrame({'Date':[],'Hour':[],'ArrDelay3AM':[]})
    for i in data.Date.unique():
        x = data[data.Date==i]['Hour']
        y = data[data.Date==i]['ArrDelay3AM']
        if max(y) < cutoff:
            data1 = data1.append(data[data['Date']==i]) # to plot low days as gray
        else:
            data2 = data2.append(data[data['Date']==i]) # to plot meltdown days in blue
    frames = [data1,data2]
    df = pd.concat(frames,ignore_index=True)
    
    for i in df.Date.unique():
        x = df[df.Date==i]['Hour']
        y = df[df.Date==i]['ArrDelay3AM']
        if max(y) < cutoff:
            axes[r, l].plot(x, y, color='gray')
        else:
            axes[r, l].plot(x,y,color='blue')
    axes[r, l].set_title('%s' % airline_name,fontdict = dict(fontsize=20))
    axes[r, l].set_xticklabels(labels)
    axes[r, l].tick_params(labelsize=14)


figure.add_subplot(111, frame_on=False)

figure.tight_layout()
figure.subplots_adjust(top=0.92, left=0.1)
figure.text(0.04, 0.5, 'Cumulative Delays (in minutes)', va='center', rotation='vertical',fontsize=20)

plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel("Hour (EST)", fontsize=20)

# Save or show plot
plt.savefig('cumulative_delays.png')
#plt.show()



