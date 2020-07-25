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
data = pd.read_csv('updated_final_data.csv')
data.head(20)

df = data[['Marketing_Airline_Network','Date','Hour','ArrDelayMinutes', 'DepDelayMinutes',
           'ArrDelay3AM','DepDelay3AM']]
df.head()

df = (df.set_index(['Marketing_Airline_Network','Date','Hour'])
        .rename_axis('Delay Type', axis=1)
        .stack()
        .unstack(2)
        .reset_index()
        .rename_axis(None, axis=1))

df_total = df.copy()

df_new = df[df['Delay Type']=='ArrDelay3AM']
df_new = df_new.drop(columns=['Delay Type'], axis=1)
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

# Create variable for the day of the week - total dataset
df_total['Date'] = pd.to_datetime(df_total['Date'])
df_total['Weekday'] = df_total['Date'].dt.dayofweek

cols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

df_total[df_total['Delay Type']=='ArrDelayMinutes']=df_total[df_total['Delay Type']=='ArrDelayMinutes'].fillna(0)
df_total[df_total['Delay Type']=='DepDelayMinutes']=df_total[df_total['Delay Type']=='DepDelayMinutes'].fillna(0)


df_total[0] = df_total[0].fillna(0)
for i in range(1,24):
    df_total[i] = df_total[i].fillna(df_total[(i-1)])
    i = i + 1
df_total.head()
df_total.isna().sum()


df_total[cols] = df_total[cols].astype(int)
df_total.to_csv('testing_cumulative_data.csv')


df_total_new = df_total.copy()
df_total_new = df_total_new.drop(columns='Weekday')
df_total_new = df_total_new.set_index(['Marketing_Airline_Network','Date','Delay Type'])
df_total_new = df_total_new.stack()
df_total_new.head()


df_total_new = df_total_new.reset_index()


df_total_new_2 = df_total_new.pivot_table(index=['Marketing_Airline_Network','Date','level_3'], columns='Delay Type', values=0)
df_total_new_2 = df_total_new_2.reset_index()
df_total_new_2.columns = ['Marketing_Airline_Network','Date','Hour','ArrDelay3AM', 'ArrDelayMinutes', 'DepDelay3AM', 'DepDelayMinutes']
df_total_new_2.head()


# Create variable for the day of the week - total dataset
df_total_new_2['Date'] = pd.to_datetime(df_total_new_2['Date'])
df_total_new_2['Weekday'] = df_total_new_2['Date'].dt.dayofweek
df_total_new_2['EST Hour'] = df_total_new_2['Hour'] + 4

df_total_new_2['EST Hour'].loc[df_total_new_2['EST Hour']==24] = 0
df_total_new_2['EST Hour'].loc[df_total_new_2['EST Hour']==25] = 1
df_total_new_2['EST Hour'].loc[df_total_new_2['EST Hour']==26] = 2
df_total_new_2['EST Hour'].loc[df_total_new_2['EST Hour']==27] = 3

df_total_new_2.head()

df_total_new_2.to_csv('final_final_data.csv')


#######################################
###### START HERE MOVING FORWARD ######

df = pd.read_csv('testing_cumulative_data_WN_labeled_equalized.csv')
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = df[df['Marketing_Airline_Network']=="WN"]
df = df[df['Delay Type']=='ArrDelay3AM']

cols = ['0','1','2','3','4','5','6','7','8','23']
df[cols] = df[cols].astype(int)


df = DataFrame(df,columns=cols)
df = df.reset_index()
df.head()
df = df[cols]
#df = df[[9,23]]  
kmeans = KMeans(n_clusters=5).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
c= kmeans.labels_.astype(float)

cluster = pd.DataFrame(c)
cluster[0] = cluster[0].astype(int)

df = df.join(cluster)
df.columns = [0,1,2,3,4,5,6,7,8,'EOD (3AM EST) Delays', 'Classification']
colors = np.where(df["Classification"]==0,'b','-')
colors[df["Classification"]==1] = 'g'
colors[df["Classification"]==2] = 'r'
colors[df["Classification"]==3] = 'c'
colors[df["Classification"]==4] = 'y'

### Create visualization
colors=['b','g','r','c','y']
plt.figure(figsize=(10,12))

lo = plt.scatter(df[df['Classification']==2][8], df[df['Classification']==2]['EOD (3AM EST) Delays'], s=10,color=colors[2])
l  = plt.scatter(df[df['Classification']==0][8], df[df['Classification']==0]['EOD (3AM EST) Delays'], s=10,color=colors[0])
a  = plt.scatter(df[df['Classification']==1][8], df[df['Classification']==1]['EOD (3AM EST) Delays'], s=10,color=colors[1])
h  = plt.scatter(df[df['Classification']==3][8], df[df['Classification']==3]['EOD (3AM EST) Delays'], s=10,color=colors[3])
hh = plt.scatter(df[df['Classification']==4][8], df[df['Classification']==4]['EOD (3AM EST) Delays'], s=10,color=colors[4])
#plt.scatter(centroids[:, 8],centroids[:, 9], c='red', s=50)

plt.legend((a, h, l, hh, lo),
           ('Great', 'Good', 'Normal', 'Bad','Meltdown'),
           scatterpoints=1,
           loc='center right',
           ncol=1,
           fontsize=12)
plt.title("Southwest Airlines Cumulative Delays (in Minutes) \nNoon vs. 3AM EST",fontsize=14)
plt.xlabel("Cumulative Delay at Noon", fontsize=14)
plt.ylabel("Cumulative Delay at 3AM EST", fontsize=14)
plt.show()



from sklearn.model_selection import train_test_split
cols = [0,1,2,3,4,5,6,7,8]
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(df[cols], df['Classification'], test_size=0.3) # 70% training and 30% test
#X_train= X_train.values.reshape(-1, 1)
#y_train= y_train.values.reshape(-1, 1)
#X_test = X_test.values.reshape(-1, 1)
#y_test = y_test.values.reshape(-1,1)

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

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

import itertools
#Define the function used to create a Confusion Matrix plot
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

#Create labels that correspond with our respective cluster labels
labsWN=['Great', 'Good','Normal','Bad','Meltdown']

#Plot non-normalized confusion matrix to show counts of predicted vs. actual clusters
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labsWN,
                      title='Confusion matrix, without normalization')

#Plot normalized confusion matrix to show percentage of classifications in predicted vs. actual clusters
plt.figure()
plt.figure(figsize=(11,7))
plot_confusion_matrix(cnf_matrix, classes=labsWN, normalize=True,
                      title='Southwest Airlines KNN Model \nNormalized Confusion Matrix')

plt.show()
