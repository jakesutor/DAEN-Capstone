# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 21:18:19 2020

@author: jakes
"""

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
os.chdir(r'C:\Users\jakes\Downloads')

# Load in data
data = pd.read_csv('combined_final_data_WN.csv')
data.head()

df = data[['ACT_ARR_DATE','ACT_ARR_HOUR','ArrDelayMinutes','DepDelayMinutes','Weekday']]
df.head()

df = (df.set_index(['ACT_ARR_DATE','ACT_ARR_HOUR'])
        .rename_axis('ArrDelayMinutes', axis=1)
        .stack()
        .unstack(1)
        .reset_index()
        .rename_axis(None, axis=1))
df_new = df[df['ArrDelayMinutes']=='ArrDelayMinutes']
df_new = df_new.drop(columns=['ArrDelayMinutes'], axis=1)
df_new.head()

# Create variable for the day of the week
df_new['ACT_ARR_DATE'] = pd.to_datetime(df_new['ACT_ARR_DATE'])
df_new['Weekday'] = df_new['ACT_ARR_DATE'].dt.dayofweek
cols = [0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

df_new[2] = df_new[2].fillna(df_new[1])
df_new[3] = df_new[3].fillna(df_new[2])
df_new[4] = df_new[4].fillna(df_new[3])
df_new[5] = df_new[5].fillna(df_new[4])
df_new[6] = df_new[6].fillna(df_new[5])
df_new.isna().sum()


df_new[cols] = df_new[cols].astype(int)

from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


  
df = DataFrame(df_new,columns=[12,23])
df = df.reset_index()
df = df[[12,23]]  
kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
c= kmeans.labels_.astype(float)

cluster = pd.DataFrame(c)
cluster[0] = cluster[0].astype(int)

df = df.join(cluster)
df.columns = ['Noon Delays','EOD Delays', 'Classification']

plt.scatter(df['Noon Delays'], df['EOD Delays'], c= c, s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.title("Southwest Airlines Cumulative Delays (in Minutes) \nNoon vs. Eleven PM")
plt.xlabel("Cumulative Delay at Noon")
plt.ylabel("Cumulative Delay at 11:00pm")
plt.show()

from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(df['Noon Delays'], df['Classification'], test_size=0.3) # 70% training and 30% test
X_train= X_train.values.reshape(-1, 1)
y_train= y_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1,1)

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




