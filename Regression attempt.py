# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 19:56:56 2020

@author: jakes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn                   import metrics, preprocessing
from sklearn.feature_selection import RFECV
from sklearn.model_selection   import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics           import accuracy_score, mean_squared_error, confusion_matrix, classification_report, precision_recall_curve, roc_curve
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.preprocessing     import StandardScaler

import os
os.chdir(r'C:\Users\jakes\Downloads')

# Load in data
data = pd.read_csv('updated_final_data.csv')
data = data[data['Marketing_Airline_Network']=='WN']
data.head(20)

df = data[['Marketing_Airline_Network','Date','Hour','ArrDelay3AM']]
df.head()

df = (df.set_index(['Marketing_Airline_Network','Date','Hour'])
        .rename_axis('Delay Type', axis=1)
        .stack()
        .unstack(2)
        .reset_index()
        .rename_axis(None, axis=1))


df_new = df.drop(columns=['Delay Type'], axis=1)
df_new.head()

# Create variable for the day of the week
df_new['Date'] = pd.to_datetime(df_new['Date'])
df_new['Weekday'] = df_new['Date'].dt.dayofweek
cols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

df_new[0] = df_new[0].fillna(0)
for i in range(1,24):
    df_new[i] = df_new[i].fillna(df_new[(i-1)])
    i = i + 1
df_new.head()
df_new.isna().sum()




df_new[cols] = df_new[cols].astype(int)

# Load in data

# LOGISTIC REGRESSION
df_new = df_new[cols].applymap(lambda x: np.log(x+1))

top_features = [0,1,2,3,4,5,6,7,8]
y = df_new[23]
x = df_new[top_features]




#split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))


# Plot outputs
plt.figure(figsize=(12,8))
plt.scatter(y_test, y_pred,  color='black')
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"r--")
plt.title("Predicted vs Actual End of Day (EOD) Delays\nLog Transformation", fontsize=14)
plt.xlabel("Actual EOD Delays (cumulative minutes)", fontsize=14)
plt.ylabel("Predicted EOD Delays (cumulative minutes)", fontsize=14)

plt.show()





############################################################################
############################################################################
df_new_WN = df_new[df_new['Marketing_Airline_Network']=='WN']
test = df_new_WN[23]

# apply log(x+1) element-wise to a subset of columns
to_log = [23]
df_log = df_new_WN[to_log].applymap(lambda x: np.log(x+1))


plt.hist(df_log, bins=100)
plt.show()

#create the model
logisreg = LogisticRegression()
model_res = logisreg.fit(x_train,y_train)
logisreg.fit(x_train,y_train)

#evaluate model with test split
y_pred = logisreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisreg.score(x_test, y_test)))

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print(classification_report(y_test, y_pred))

