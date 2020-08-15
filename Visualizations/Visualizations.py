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

os.chdir(r'C:\Users\jakes\Downloads')


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


######################## Or run airlines individually #######################

airline = 'HA'
labels = ['4','','','','8','','','','12','','','','16','','','','20','','','','0','','','3']

# Iterate through airlines and combine as subplots
airline_name = airlines_cv[airlines_cv['Marketing_Airline_Network']==airline]['Airline'].reset_index()
airline_name = airline_name['Airline'][0]

data = data_full[data_full['Marketing_Airline_Network']==airline]

data = data[['Date','Hour','ArrDelay3AM']]

cutoff = max(data['ArrDelay3AM'])*0.33

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
        plt.plot(x, y, color='gray')
    else:
        plt.plot(x,y,color='blue')

plt.xlabel("Hour (EST)", fontsize=12)
plt.ylabel('Cumulative Delays (in minutes)',fontsize=12)
plt.title('%s Delays by Hour' % airline_name, fontsize=12)
plt.xticks(np.arange(24),labels)

plt.show()



#################### Correcting old visualizations ########################

updated_data = pd.read_csv('testing_cumulative_data.csv')
data2 = updated_data[updated_data['Delay Type'] == 'ArrDelay3AM']
plt.figure(figsize=(12,8))
plt.scatter(data2['8'],data2['23'],facecolors='none', edgecolors='black')
plt.xlabel('Cumulative Delays at Noon (in minutes)', fontsize=14)
plt.ylabel('Cumulative Delays at 3:00 AM (in minutes)',fontsize=14)
plt.title('Arrival Delays Noon vs 3:00 AM', fontsize=16)
plt.show()

# For just WN
data3 = data2[data2['Marketing_Airline_Network']=='WN']
plt.figure(figsize=(12,8))
plt.scatter(data3['8'],data3['23'],facecolors='none', edgecolors='black')
plt.xlabel('Cumulative Delays at Noon (in minutes)', fontsize=14)
plt.ylabel('Cumulative Delays at 3:00 AM (in minutes)',fontsize=14)
plt.title('Southwest Airlines\nArrival Delays Noon vs 3:00 AM', fontsize=16)
plt.show()


# For just WN by day

plt.figure(figsize=(12,8))
plt.scatter(data3['8'],data3['23'],facecolors='none', edgecolors='black')
plt.xlabel('Cumulative Delays at Noon (in minutes)', fontsize=14)
plt.ylabel('Cumulative Delays at 3:00 AM (in minutes)',fontsize=14)
plt.title('Southwest Airlines\nArrival Delays Noon vs 3:00 AM', fontsize=16)
plt.show()

data3.loc[data3['Weekday']==0,'Weekday']='Monday'
data3.loc[data3['Weekday']==1,'Weekday']='Tuesday'
data3.loc[data3['Weekday']==2,'Weekday']='Wednesday'
data3.loc[data3['Weekday']==3,'Weekday']='Thursday'
data3.loc[data3['Weekday']==4,'Weekday']='Friday'
data3.loc[data3['Weekday']==5,'Weekday']='Saturday'
data3.loc[data3['Weekday']==6,'Weekday']='Sunday'


# Set up major plot
figure, axes = plt.subplots(nrows=2, ncols=4,sharex=True,sharey=True,figsize=(20,10))
figure.suptitle('Cumulative Delays by Day for Southwest Airlines', fontsize=24)

# Set initial location of plot and labels
l = -1
r = 0
#labels = ['4','','','','8','','','','12','','','','16','','','','20','','','','0','','','3']


# Iterate through airlines and combine as subplots
for j in data3['Weekday'].unique():
    l = l+1
    if l < 4:
        l = l
    else:
        l = 0
        if r < 1:
            r = r+1
        else:
            l = l+1
            
    print(l,r)

    data = data3[data3['Weekday']==j]
        
    x = data['8']
    y = data['23']
    axes[r, l].scatter(x, y)    

    axes[r, l].set_title('%s' % j,fontdict = dict(fontsize=14))
#    axes[r, l].set_xticklabels(labels)
#    axes[r, l].tick_params(labelsize=14)


figure.add_subplot(111, frame_on=False)

figure.tight_layout()
figure.subplots_adjust(top=0.92, left=0.1)
figure.text(0.04, 0.5, 'Cumulative Delays at 3:00 am (in mins)', va='center', rotation='vertical',fontsize=16)

plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel('Cumulative Delays at Noon (in mins)', fontsize=16)

# Save or show plot
#plt.savefig('cumulative_delays.png')
plt.show()


