# DAEN-Capstone
Capstone Project - Airline Delays Analysis

### Problem Statement
Within each airline, air traffic managers and airline dispatchers have the authority to make adjustments at noon to avoid suffering excessive delays throughout the remainder of the day. These airline employees need a prediction of cumulative end of day flight delays around approximately noon in order to properly determine whether or not operational adjustments are necessary to avoid suffering an operation “Meltdown”.  Each of the 10 different major United States airlines with reported flights since January 2018 will be considered. Initially, our project will focus on modeling flight delays for Southwest Airlines, using the cumulative delays at noon as the baseline to predict the total delays at the end of a day. Given the volume of available data and number of flights, we will use available data since January 2018 and train our model using variables including the number and times of these delays, as well as the causes of these delays. 

### Contents
#### Data Cleansing
Initial iteration of cleansing code provided by Spring Semester GMU DAEN 690 group. Please see their initial project and code at https://github.com/teamatash/CATSR-Flight-Prediction-BTS.
We added January-February 2020 data to their datasets. All raw data can be found at https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=237. Refer to the 'Download raw data explanation' file for details regarding which variables were utilized.

Once the files have been downloaded, first utilize the files within the 'Data Cleansing' folder to prepare and combine the data. Run files according to the order of their number.

1.) Merge Data - first, Merge Data file (removes all cancelled flights as part of the cleansing), then Cancelled Flights Imputation

2.) Clean Data - three individual files for cleaning data

3.) Calculate Cumulative Delays - calculates cumulative delays at each hour

4.) Calculate Cancelled Flights Cumulative Delays - calculates cumulative delays at each hour for cancelled flights

5.) Combining cancelled and delayed flights - needed in order to combine the cancelled flights and other flights

6.) Set 3AM as last hour of the day - rather than having midnight as the beginning of the day, set 3AM as our target time

#### Visualizations

Creates histograms and time series for each airline

#### Initial Analysis

1.) KNN Classification - used to label data for further analysis

2.) Regression attempt - testing linear regression analysis

#### Algorithms

1.) LSTM - includes both GridSearch version and version without GridSearch

2.) XGBoost

3.) SVM - includes a separate file for creating the confusion matrices to match the format of the confusion matrices of the LSTM and XGBoost models; also includes separate files for each airline


#### Data Files

Can be found in the Data folder with other files in the Old Data folder.




