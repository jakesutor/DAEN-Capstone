# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:18:34 2020

@author: jakes
"""

#This is the SIXTH FILE to run
#import libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import holidays
import datetime as dt
from datetime import timedelta  

#%matplotlib qt
#%matplotlib inline
os.chdir(r'C:\Users\jakes\Downloads')

tf.enable_eager_execution()

# parameters of LSTM
TRAIN_SPLIT = 364*24 #14016 #don't cahnge this number
BATCH_SIZE = 24 #256
BUFFER_SIZE = 50000
EVALUATION_INTERVAL = 364/BATCH_SIZE #200
EPOCHS = 200
accuracy_threshold = 0.25

# past history valid options: 24-future_target,2(24)-future_target,3(24)-future_target,...
past_history = 8
# future target valid options: 12,11,10,...
future_target = 24 - past_history
# note that 'past history' + 'future target' must be a multiple of 24, i.e. 24, 48, 72, ...

# this is not relevant in our problem, it is always 1
STEP = 1 

# load data
data = pd.read_csv('final_final_data.csv')

# select an airline (for example: AA)
data = data[data['Marketing_Airline_Network'] == 'WN']
data.columns

# total delay for all airlines
#data1 = data.groupby(['Date','Hour','Weekday'])['ArrDelayMinutes'].sum().unstack(1,0).stack().reset_index(name='ArrDelay3AM')
#data2 = data.groupby(['Date','Hour','Weekday'])['DepDelayMinutes'].sum().unstack(1,0).stack().reset_index(name='DepDelay3AM')
#df = pd.merge(data1,data2,on=['Date','Hour','Weekday'])
df = data[['Date','Hour','Weekday','ArrDelay3AM','DepDelay3AM']].copy()

df['season'] = pd.to_datetime(df['Date']).dt.quarter
dates = df['Date'].values
holiday = np.empty(dates.shape[0])
for i in range(0,dates.shape[0]):
    if dt.datetime.strptime(dates[i],'%Y-%m-%d') in holidays.US():
        holiday[i] = 1
    else:
        holiday[i] = 0
df['holiday'] = holiday

df = df[df['Date'] != '2018-03-11']
#df = df[df['Date'] != '2019-03-10']


# don't change the seed so that we can compare the results with each other
#tf.random.set_seed(13)
tf.set_random_seed(13)

#creating time steps
def create_time_steps(length):
  return list(range(-length, 0))

#def for plotting
def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.ylabel('Total Delay (min)')
  plt.xlabel('Time-Step')
  return plt

#def for baseline
def baseline(history):
  return np.mean(history)


######## multivariate
features_considered = ['ArrDelay3AM','DepDelay3AM','Weekday','season','holiday']
features = df[features_considered]
features.index = df[['Date','Hour']]
features.head()

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

for i in range(0,2):
    dataset[:,i] = (dataset[:,i]-data_mean[i])/data_std[i]

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset)

  for i in range(start_index, end_index, 24):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size-1]) #added -1
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

#def for plotting the error
def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.xlabel('Epoch')
  plt.ylabel('Mean Absolute Error')
  plt.title(title)
  plt.legend()

  plt.show()

#multivariate_data(dataset, target, start_index, end_index, history_size,target_size, step, single_step=False)
#preparing the dataset
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('Target delay to predict : {}'.format(y_train_multi[0].shape))

#train
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

#validation
val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

#definition for multi step plot
def multi_step_plot(history, true_future, prediction):
  #plt.figure(figsize=(12, 6))
  plt.figure(figsize=(8, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 0]*data_std[0]+data_mean[0]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future)*data_std[0]+data_mean[0],color='black',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction)*data_std[0]+data_mean[0], color='red', ls='dashed',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.xlabel('Time of Day')
  plt.xticks(range(-past_history+2,future_target,5),range(2,24,5))
  plt.ylabel('Cumulative Delay (Minute)')
  plt.show()
  

for x, y in train_data_multi.take(1):
  multi_step_plot(x[0], y[0], np.array([0]))

#Building the LSTM model
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(25))
multi_step_model.add(tf.keras.layers.Dense(future_target))

multi_step_model.compile(optimizer='adam', loss='mean_squared_error')

for x, y in val_data_multi.take(1):
  print (multi_step_model.predict(x).shape)

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=18)

# plot training and validation loss
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

# show sample results
#rmse
rmse = np.sqrt(multi_step_model.evaluate(x_val_multi,y_val_multi))
print('RMSE: %s' % rmse)
nrmse = rmse*data_std[0]
print('NRMSE: %s' % nrmse)

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(1)

#plotting the sample predictions
for x, y in val_data_multi.take(10):
  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])

pred_data=pd.DataFrame([])
est_date = dt.date(2018, 12, 31)

for x, y in val_data_multi.take(365):
    true_val = y[:,15]*data_std[0]+data_mean[0]
    prediction = np.array(multi_step_model.predict(x)[:,15]*data_std[0]+data_mean[0])
    pred_data = pred_data.append(pd.DataFrame({'Date':est_date,'True Value':true_val,'Predicted Value':prediction}, index=[0]),ignore_index=True)
    est_date = est_date + timedelta(days=1) 
pred_data['Predicted Value'] = round(pred_data['Predicted Value'])

print(pred_data)


plt.figure(figsize=(12,8))
meltdown = plt.scatter(pred_data[pred_data['True Value']>=80000]['True Value'],pred_data[pred_data['True Value']>=80000]['Predicted Value'],color='blue')
normal = plt.scatter(pred_data[pred_data['True Value']<80000]['True Value'],pred_data[pred_data['True Value']<80000]['Predicted Value'],color='gray')
plt.axhline(y=80000, color='r', linestyle='-')
plt.legend((meltdown,normal),
           ('Meltdown', 'Normal'),
           scatterpoints=1,
           loc='top left',
           ncol=1,
           fontsize=12)
plt.title('LSTM Predictions',fontsize=16)
plt.xlabel('True Values',fontsize=14)
plt.ylabel('Predicted Values',fontsize=14)
plt.ylim(ymin=0)  
plt.xlim(xmin=0) 
plt.show()


labels = pd.read_csv('testing_cumulative_data_WN_labeled_equalized.csv')
labels['Date'] = pd.to_datetime(labels['Date'])

cluster = pred_data.copy()
cluster['Predicted Cluster'] = ''
cluster['True Cluster'] = ''

cluster['True Cluster']=cluster['Date'].map(dict(zip(labels['Date'],labels['Cluster'])))
cluster['True Value'].groupby(cluster['True Cluster']).describe()

cluster.loc[cluster['Predicted Value']<=24333,'Predicted Cluster']='Great'
cluster.loc[cluster['Predicted Value']>24333,'Predicted Cluster']='Good'
cluster.loc[cluster['Predicted Value']>38890,'Predicted Cluster']='Normal'
cluster.loc[cluster['Predicted Value']>55396,'Predicted Cluster']='Bad'
cluster.loc[cluster['Predicted Value']>78910,'Predicted Cluster']='Meltdown'




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

y_test = cluster['True Cluster']
y_pred = cluster['Predicted Cluster']

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


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
                      title='Southwest Airlines Classified LSTM Model \nNormalized Confusion Matrix')

plt.show()





