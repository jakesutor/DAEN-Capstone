#This is the FIRST script to run
#import proper libraries
import pandas as pd
import zipfile
import os
os.chdir(r'C:\Users\jakes\Documents\DAEN')

#opens zip
zf = zipfile.ZipFile('Datasets 690.zip')
df = {}
i = 1
#adds all the different datasets from 2018 to 2020 into one csv file
for year in range(2018,2019):
    for month in range(1,13):
        df[i] = pd.read_csv(zf.open('On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)%s%s.csv'%(year,month)))
        i += 1

frames = [df[i] for i in range(1,25)]

df = pd.concat(frames,ignore_index=True)
df.columns = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE','MKT_UNIQUE_CARRIER','BRANDED_CODE_SHARE','MKT_CARRIER_AIRLINE_ID','MKT_CARRIER','MKT_CARRIER_FL_NUM','SCH_OP_UNIQUE_CARRIER','SCH_OP_CARRIER_AIRLINE_ID','SCH_OP_CARRIER','SCH_OP_CARRIER_FL_NUM','OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','OP_CARRIER','TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','ORIGIN_CITY_MARKET_ID','ORIGIN','ORIGIN_CITY_NAME','ORIGIN_STATE_ABR','ORIGIN_STATE_FIPS','ORIGIN_STATE_NM','ORIGIN_WAC','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','DEST_CITY_MARKET_ID','DEST','DEST_CITY_NAME','DEST_STATE_ABR','DEST_STATE_FIPS','DEST_STATE_NM','DEST_WAC','CRS_DEP_TIME','DEP_TIME','DEP_DELAY','DEP_DELAY_NEW','DEP_DEL15','DEP_DELAY_GROUP','DEP_TIME_BLK','TAXI_OUT','WHEELS_OFF','WHEELS_ON','TAXI_IN','CRS_ARR_TIME','ARR_TIME','ARR_DELAY','ARR_DELAY_NEW','ARR_DEL15','ARR_DELAY_GROUP','ARR_TIME_BLK','CANCELLED','CANCELLATION_CODE','DIVERTED','CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME','AIR_TIME','FLIGHTS','DISTANCE','DISTANCE_GROUP','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','FIRST_DEP_TIME','TOTAL_ADD_GTIME','LONGEST_ADD_GTIME','DIV_AIRPORT_LANDINGS','DIV_REACHED_DEST','DIV_ACTUAL_ELAPSED_TIME','DIV_ARR_DELAY','DIV_DISTANCE','DIV1_AIRPORT','DIV1_AIRPORT_ID','DIV1_AIRPORT_SEQ_ID','DIV1_WHEELS_ON','DIV1_TOTAL_GTIME','DIV1_LONGEST_GTIME','DIV1_WHEELS_OFF','DIV1_TAIL_NUM','DIV2_AIRPORT','DIV2_AIRPORT_ID','DIV2_AIRPORT_SEQ_ID','DIV2_WHEELS_ON','DIV2_TOTAL_GTIME','DIV2_LONGEST_GTIME','DIV2_WHEELS_OFF','DIV2_TAIL_NUM','DIV3_AIRPORT','DIV3_AIRPORT_ID','DIV3_AIRPORT_SEQ_ID','DIV3_WHEELS_ON','DIV3_TOTAL_GTIME','DIV3_LONGEST_GTIME','DIV3_WHEELS_OFF','DIV3_TAIL_NUM','DIV4_AIRPORT','DIV4_AIRPORT_ID','DIV4_AIRPORT_SEQ_ID','DIV4_WHEELS_ON','DIV4_TOTAL_GTIME','DIV4_LONGEST_GTIME','DIV4_WHEELS_OFF','DIV4_TAIL_NUM','DIV5_AIRPORT','DIV5_AIRPORT_ID','DIV5_AIRPORT_SEQ_ID','DIV5_WHEELS_ON','DIV5_TOTAL_GTIME','DIV5_LONGEST_GTIME','DIV5_WHEELS_OFF','DIV5_TAIL_NUM', 'DUP','Unnamed']
df = df.dropna(subset=['DEP_DELAY_NEW','ARR_DELAY_NEW','CRS_DEP_TIME','DEP_TIME','CRS_ARR_TIME','ARR_TIME','FL_DATE','ORIGIN_CITY_NAME'])
df = df[['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE','MKT_UNIQUE_CARRIER','BRANDED_CODE_SHARE','MKT_CARRIER_AIRLINE_ID','MKT_CARRIER','MKT_CARRIER_FL_NUM','SCH_OP_UNIQUE_CARRIER','SCH_OP_CARRIER_AIRLINE_ID','SCH_OP_CARRIER','SCH_OP_CARRIER_FL_NUM','OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','OP_CARRIER','TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','ORIGIN_CITY_MARKET_ID','ORIGIN','ORIGIN_CITY_NAME','ORIGIN_STATE_ABR','ORIGIN_STATE_FIPS','ORIGIN_STATE_NM','ORIGIN_WAC','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','DEST_CITY_MARKET_ID','DEST','DEST_CITY_NAME','DEST_STATE_ABR','DEST_STATE_FIPS','DEST_STATE_NM','DEST_WAC','CRS_DEP_TIME','DEP_TIME','DEP_DELAY','DEP_DELAY_NEW','DEP_DEL15','DEP_DELAY_GROUP','DEP_TIME_BLK','TAXI_OUT','WHEELS_OFF','WHEELS_ON','TAXI_IN','CRS_ARR_TIME','ARR_TIME','ARR_DELAY','ARR_DELAY_NEW','ARR_DEL15','ARR_DELAY_GROUP','ARR_TIME_BLK','CANCELLED','CANCELLATION_CODE','DIVERTED','DUP','CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME','AIR_TIME','FLIGHTS','DISTANCE','DISTANCE_GROUP','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','FIRST_DEP_TIME','TOTAL_ADD_GTIME','LONGEST_ADD_GTIME','DIV_AIRPORT_LANDINGS','DIV_REACHED_DEST','DIV_ACTUAL_ELAPSED_TIME','DIV_ARR_DELAY','DIV_DISTANCE','DIV1_AIRPORT','DIV1_AIRPORT_ID','DIV1_AIRPORT_SEQ_ID','DIV1_WHEELS_ON','DIV1_TOTAL_GTIME','DIV1_LONGEST_GTIME','DIV1_WHEELS_OFF','DIV1_TAIL_NUM','DIV2_AIRPORT','DIV2_AIRPORT_ID','DIV2_AIRPORT_SEQ_ID','DIV2_WHEELS_ON','DIV2_TOTAL_GTIME','DIV2_LONGEST_GTIME','DIV2_WHEELS_OFF','DIV2_TAIL_NUM','DIV3_AIRPORT','DIV3_AIRPORT_ID','DIV3_AIRPORT_SEQ_ID','DIV3_WHEELS_ON','DIV3_TOTAL_GTIME','DIV3_LONGEST_GTIME','DIV3_WHEELS_OFF','DIV3_TAIL_NUM','DIV4_AIRPORT','DIV4_AIRPORT_ID','DIV4_AIRPORT_SEQ_ID','DIV4_WHEELS_ON','DIV4_TOTAL_GTIME','DIV4_LONGEST_GTIME','DIV4_WHEELS_OFF','DIV4_TAIL_NUM','DIV5_AIRPORT','DIV5_AIRPORT_ID','DIV5_AIRPORT_SEQ_ID','DIV5_WHEELS_ON','DIV5_TOTAL_GTIME','DIV5_LONGEST_GTIME','DIV5_WHEELS_OFF','DIV5_TAIL_NUM','Unnamed']]



df1 = pd.read_csv(zf.open('On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2020_1.csv'))
df1.columns = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE','MKT_UNIQUE_CARRIER','BRANDED_CODE_SHARE','MKT_CARRIER_AIRLINE_ID','MKT_CARRIER','MKT_CARRIER_FL_NUM','SCH_OP_UNIQUE_CARRIER','SCH_OP_CARRIER_AIRLINE_ID','SCH_OP_CARRIER','SCH_OP_CARRIER_FL_NUM','OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','OP_CARRIER','TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','ORIGIN_CITY_MARKET_ID','ORIGIN','ORIGIN_CITY_NAME','ORIGIN_STATE_ABR','ORIGIN_STATE_FIPS','ORIGIN_STATE_NM','ORIGIN_WAC','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','DEST_CITY_MARKET_ID','DEST','DEST_CITY_NAME','DEST_STATE_ABR','DEST_STATE_FIPS','DEST_STATE_NM','DEST_WAC','CRS_DEP_TIME','DEP_TIME','DEP_DELAY','DEP_DELAY_NEW','DEP_DEL15','DEP_DELAY_GROUP','DEP_TIME_BLK','TAXI_OUT','WHEELS_OFF','WHEELS_ON','TAXI_IN','CRS_ARR_TIME','ARR_TIME','ARR_DELAY','ARR_DELAY_NEW','ARR_DEL15','ARR_DELAY_GROUP','ARR_TIME_BLK','CANCELLED','CANCELLATION_CODE','DIVERTED','DUP','CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME','AIR_TIME','FLIGHTS','DISTANCE','DISTANCE_GROUP','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','FIRST_DEP_TIME','TOTAL_ADD_GTIME','LONGEST_ADD_GTIME','DIV_AIRPORT_LANDINGS','DIV_REACHED_DEST','DIV_ACTUAL_ELAPSED_TIME','DIV_ARR_DELAY','DIV_DISTANCE','DIV1_AIRPORT','DIV1_AIRPORT_ID','DIV1_AIRPORT_SEQ_ID','DIV1_WHEELS_ON','DIV1_TOTAL_GTIME','DIV1_LONGEST_GTIME','DIV1_WHEELS_OFF','DIV1_TAIL_NUM','DIV2_AIRPORT','DIV2_AIRPORT_ID','DIV2_AIRPORT_SEQ_ID','DIV2_WHEELS_ON','DIV2_TOTAL_GTIME','DIV2_LONGEST_GTIME','DIV2_WHEELS_OFF','DIV2_TAIL_NUM','DIV3_AIRPORT','DIV3_AIRPORT_ID','DIV3_AIRPORT_SEQ_ID','DIV3_WHEELS_ON','DIV3_TOTAL_GTIME','DIV3_LONGEST_GTIME','DIV3_WHEELS_OFF','DIV3_TAIL_NUM','DIV4_AIRPORT','DIV4_AIRPORT_ID','DIV4_AIRPORT_SEQ_ID','DIV4_WHEELS_ON','DIV4_TOTAL_GTIME','DIV4_LONGEST_GTIME','DIV4_WHEELS_OFF','DIV4_TAIL_NUM','DIV5_AIRPORT','DIV5_AIRPORT_ID','DIV5_AIRPORT_SEQ_ID','DIV5_WHEELS_ON','DIV5_TOTAL_GTIME','DIV5_LONGEST_GTIME','DIV5_WHEELS_OFF','DIV5_TAIL_NUM','Unnamed']
df1 = df1.dropna(subset=['DEP_DELAY_NEW','ARR_DELAY_NEW','CRS_DEP_TIME','DEP_TIME','CRS_ARR_TIME','ARR_TIME','FL_DATE','ORIGIN_CITY_NAME'])

df2 = pd.read_csv(zf.open('On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2020_2.csv'))
df2.columns = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE','MKT_UNIQUE_CARRIER','BRANDED_CODE_SHARE','MKT_CARRIER_AIRLINE_ID','MKT_CARRIER','MKT_CARRIER_FL_NUM','SCH_OP_UNIQUE_CARRIER','SCH_OP_CARRIER_AIRLINE_ID','SCH_OP_CARRIER','SCH_OP_CARRIER_FL_NUM','OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','OP_CARRIER','TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','ORIGIN_CITY_MARKET_ID','ORIGIN','ORIGIN_CITY_NAME','ORIGIN_STATE_ABR','ORIGIN_STATE_FIPS','ORIGIN_STATE_NM','ORIGIN_WAC','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','DEST_CITY_MARKET_ID','DEST','DEST_CITY_NAME','DEST_STATE_ABR','DEST_STATE_FIPS','DEST_STATE_NM','DEST_WAC','CRS_DEP_TIME','DEP_TIME','DEP_DELAY','DEP_DELAY_NEW','DEP_DEL15','DEP_DELAY_GROUP','DEP_TIME_BLK','TAXI_OUT','WHEELS_OFF','WHEELS_ON','TAXI_IN','CRS_ARR_TIME','ARR_TIME','ARR_DELAY','ARR_DELAY_NEW','ARR_DEL15','ARR_DELAY_GROUP','ARR_TIME_BLK','CANCELLED','CANCELLATION_CODE','DIVERTED','DUP','CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME','AIR_TIME','FLIGHTS','DISTANCE','DISTANCE_GROUP','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','FIRST_DEP_TIME','TOTAL_ADD_GTIME','LONGEST_ADD_GTIME','DIV_AIRPORT_LANDINGS','DIV_REACHED_DEST','DIV_ACTUAL_ELAPSED_TIME','DIV_ARR_DELAY','DIV_DISTANCE','DIV1_AIRPORT','DIV1_AIRPORT_ID','DIV1_AIRPORT_SEQ_ID','DIV1_WHEELS_ON','DIV1_TOTAL_GTIME','DIV1_LONGEST_GTIME','DIV1_WHEELS_OFF','DIV1_TAIL_NUM','DIV2_AIRPORT','DIV2_AIRPORT_ID','DIV2_AIRPORT_SEQ_ID','DIV2_WHEELS_ON','DIV2_TOTAL_GTIME','DIV2_LONGEST_GTIME','DIV2_WHEELS_OFF','DIV2_TAIL_NUM','DIV3_AIRPORT','DIV3_AIRPORT_ID','DIV3_AIRPORT_SEQ_ID','DIV3_WHEELS_ON','DIV3_TOTAL_GTIME','DIV3_LONGEST_GTIME','DIV3_WHEELS_OFF','DIV3_TAIL_NUM','DIV4_AIRPORT','DIV4_AIRPORT_ID','DIV4_AIRPORT_SEQ_ID','DIV4_WHEELS_ON','DIV4_TOTAL_GTIME','DIV4_LONGEST_GTIME','DIV4_WHEELS_OFF','DIV4_TAIL_NUM','DIV5_AIRPORT','DIV5_AIRPORT_ID','DIV5_AIRPORT_SEQ_ID','DIV5_WHEELS_ON','DIV5_TOTAL_GTIME','DIV5_LONGEST_GTIME','DIV5_WHEELS_OFF','DIV5_TAIL_NUM','Unnamed']
df2 = df2.dropna(subset=['DEP_DELAY_NEW','ARR_DELAY_NEW','CRS_DEP_TIME','DEP_TIME','CRS_ARR_TIME','ARR_TIME','FL_DATE','ORIGIN_CITY_NAME'])


df = df.append(df1, ignore_index=True)
df = df.append(df2, ignore_index=True)

#dataframe into a csv
df.to_csv('data.csv')
