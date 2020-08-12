#!/usr/bin/env python
# coding: utf-8

# In[2]:


#This is the FIRST script to run
#import proper libraries
import pandas as pd
import zipfile
import os
#os.chdir(r'C:\Users\jakes\Documents\DAEN')

#opens zip
#zf = zipfile.ZipFile('Datasets 690.zip')


# In[3]:


import boto3
from sagemaker import get_execution_role

bucket='jet-lag-data-6.21.20'

df = {}
i = 1
#adds all the different datasets from 2018 to 2020 into one csv file
for year in range(2018,2020):
    for month in range(1,13):
        data_key = '%s_%s.csv'%(year,month)
        data_location = 's3://{}/{}'.format(bucket, data_key)
        df[i] = pd.read_csv(data_location)
        df[i] = df[i].dropna(subset=['DEP_DELAY_NEW','ARR_DELAY_NEW','CRS_DEP_TIME','DEP_TIME','CRS_ARR_TIME','ARR_TIME','FL_DATE','ORIGIN_CITY_NAME'])
        i += 1

frames = [df[i] for i in range(1,25)]
df = pd.concat(frames,ignore_index=True)


# In[3]:


import boto3
from sagemaker import get_execution_role

bucket='jet-lag-data-6.21.20'

df = {}
i = 1
#adds all the different datasets from 2018 to 2020 into one csv file
for year in range(2018,2019):
    for month in range(1,13):
        data_key = '%s_%s.csv'%(year,month)
        data_location = 's3://{}/{}'.format(bucket, data_key)
        df[i] = pd.read_csv(data_location)
        i += 1

frames = [df[i] for i in range(1,13)]
df = pd.concat(frames,ignore_index=True)
#


# In[5]:


max(df['FL_DATE'])


# In[6]:


dataframe = df[df['CANCELLED']==1]
dataframe.head()


# In[7]:


df = {}
i = 1
#adds all the different datasets from 2018 to 2020 into one csv file
for year in range(2019,2020):
    for month in range(1,13):
        data_key = '%s_%s.csv'%(year,month)
        data_location = 's3://{}/{}'.format(bucket, data_key)
        df[i] = pd.read_csv(data_location)
        i += 1

frames = [df[i] for i in range(1,13)]
df = pd.concat(frames,ignore_index=True)


# In[8]:


df = df[df['CANCELLED']==1]
df.head()


# In[9]:


frames = [dataframe,df]
df = pd.concat(frames,ignore_index=True)
df.head()


# In[10]:


df.to_csv('cancelled_raw_data.csv')


# In[3]:


# See top data
df.head(10)


# In[4]:


# Describe data
#df.describe()

# Drop last column
max(df.FL_DATE)


# In[4]:


data_key_2020_1 = '2020_1.csv'
data_location_2020_1 = 's3://{}/{}'.format(bucket, data_key_2020_1)
data_key_2020_2 = '2020_2.csv'
data_location_2020_2 = 's3://{}/{}'.format(bucket, data_key_2020_2)


df1 = pd.read_csv(data_location_2020_1)
first_1 = len(df1.index)
df1 = df1.dropna(subset=['DEP_DELAY_NEW','ARR_DELAY_NEW','CRS_DEP_TIME','DEP_TIME','CRS_ARR_TIME','ARR_TIME','FL_DATE','ORIGIN_CITY_NAME'])
dropped_1 = len(df1.index)

df2 = pd.read_csv(data_location_2020_2)
first_2 = len(df2.index)
df2 = df2.dropna(subset=['DEP_DELAY_NEW','ARR_DELAY_NEW','CRS_DEP_TIME','DEP_TIME','CRS_ARR_TIME','ARR_TIME','FL_DATE','ORIGIN_CITY_NAME'])
dropped_2 = len(df2.index)


# In[11]:


first_1


# In[12]:


dropped_1


# In[13]:


first_2


# In[14]:


dropped_2


# In[5]:


frames=[df,df1,df2]
df = pd.concat(frames, ignore_index=True)


# In[6]:


df.head()


# In[7]:


max(df.FL_DATE)


# In[8]:


total_rows = len(df.index)
total_rows
#dropped_nas = len(df.dropna(subset=['DEP_DELAY_NEW','ARR_DELAY_NEW','CRS_DEP_TIME','DEP_TIME','CRS_ARR_TIME','ARR_TIME','FL_DATE','ORIGIN_CITY_NAME']))


# In[9]:


#dataframe into a csv
df.to_csv('data.csv')


# In[ ]:




