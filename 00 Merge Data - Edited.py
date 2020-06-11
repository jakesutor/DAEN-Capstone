#This is the FIRST script to run
#import proper libraries
import pandas as pd
import zipfile
import os
os.chdir(r'C:\Users\jakes\Documents\DAEN')

#opens zip
zf = zipfile.ZipFile('Datasets 690.zip')
df = {}
#i = 1
#adds all the different datasets from 2018 to 2020 into one csv file
#for year in range(2018,2020):
#    for month in range(1,2):
#        df[i] = pd.read_csv(zf.open('On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2020_%s.csv'%(month)))
#        i += 1

df1 = pd.read_csv(zf.open('On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2020_1.csv'))
df1 = df1.dropna(subset=['DEP_DELAY_NEW','ARR_DELAY_NEW','CRS_DEP_TIME','DEP_TIME','CRS_ARR_TIME','ARR_TIME','FL_DATE','ORIGIN_CITY_NAME'])

df2 = pd.read_csv(zf.open('On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2020_2.csv'))
df2 = df2.dropna(subset=['DEP_DELAY_NEW','ARR_DELAY_NEW','CRS_DEP_TIME','DEP_TIME','CRS_ARR_TIME','ARR_TIME','FL_DATE','ORIGIN_CITY_NAME'])

#frames = [df[i] for i in range(1,2)]

#df = pd.concat(frames,ignore_index=True)
result = df1.append(df2, ignore_index=True)
#dataframe into a csv
result.to_csv('data.csv')
