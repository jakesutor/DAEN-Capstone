# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:45:52 2020

@author: jakes
"""

#import libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import holidays
import datetime as dt
#%matplotlib qt
#%matplotlib inline


# load data
data = pd.read_csv('final_final_data.csv')
data = data[data['Marketing_Airline_Network']=='NK']

data = data[['Date','Hour','ArrDelay3AM']]

cutoff = 20000

data1 = pd.DataFrame({'Date':[],'Hour':[],'ArrDelay3AM':[]})
data2 = pd.DataFrame({'Date':[],'Hour':[],'ArrDelay3AM':[]})
for i in data.Date.unique():
    x = data[data.Date==i]['Hour']
    y = data[data.Date==i]['ArrDelay3AM']
    if max(y) < cutoff:
#        d = df[df.Date==i]
        data1 = data1.append(data[data['Date']==i])
    else:
#        d = df[df.Date==i]
        data2 = data2.append(data[data['Date']==i])
frames = [data1,data2]
df = pd.concat(frames,ignore_index=True)

plt.figure(figsize=(12,8))
for i in df.Date.unique():
    x = df[df.Date==i]['Hour']
    y = df[df.Date==i]['ArrDelay3AM']
    if max(y) < cutoff:
        plt.plot(x, y, color='gray')
    else:
        plt.plot(x,y,color='blue')
plt.title('Cumulative Delays by Hour (Spirit Airlines)')
plt.xlabel('Hour')
plt.ylabel('Cumulative Delay (in minutes)')
plt.show()




