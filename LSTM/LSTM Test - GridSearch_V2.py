# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:18:34 2020

@author: jakes
"""

# LSTM Analysis
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
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import six
from keras.wrappers.scikit_learn import KerasRegressor
import itertools
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Set working directory

# Allow TensorFlow to use old functions
tf.executing_eagerly()

# Read in data
airlines_cv = pd.read_csv('airlines.csv') # For mapping airlines to abbreviations
airlines_cv['Marketing_Airline_Network'] = pd.Categorical(airlines_cv['Marketing_Airline_Network'], ["AS", "HA","DL","NK","AA","F9","B6","WN","UA","G4"])
airlines_cv = airlines_cv.sort_values("Marketing_Airline_Network")

full_data = pd.read_csv('final_final_data.csv') # For raw data with cumulative delays

airline = 'HA'

# Create a function to run entire LSTM algorithm
def run_lstm(airline):
    # parameters of LSTM
    TRAIN_SPLIT = 364*24 
    BATCH_SIZE = 24 
    BUFFER_SIZE = 5000
    EVALUATION_INTERVAL = 364/BATCH_SIZE 
    EPOCHS = 200
    accuracy_threshold = 0.25
    
    n_iter_search = 16 # Number of parameter settings that are sampled.
    
    # Parameters to evaluate in order to find best parameters for each model
    optimizers = ['rmsprop', 'adam', 'adadelta'] 
    init = ['glorot_uniform', 'normal', 'uniform']
    EPOCHS = np.array([100, 200, 500])
    param_grid = dict(optimizer=optimizers, nb_epoch=EPOCHS, init=init)
    
    
    # past history valid options: 24-future_target,2(24)-future_target,3(24)-future_target,...
    past_history = 8
    # future target valid options: 12,11,10,...
    future_target = 24 - past_history
    # note that 'past history' + 'future target' must be a multiple of 24, i.e. 24, 48, 72, ...
    
    # this is not relevant in our problem, it is always 1 (to make a prediction for each hour)
    STEP = 1 
    
    # Map airline to airline name
    airline_name = airlines_cv[airlines_cv['Marketing_Airline_Network']==airline]['Airline'].reset_index()
    airline_name = airline_name['Airline'][0]
    
    # select an airline from the dataset (for example: AA)
    data = full_data[full_data['Marketing_Airline_Network'] == airline]
    data.columns
    
    # total delay for all airlines
    df = data[['Date','Hour','Weekday','ArrDelay3AM','DepDelay3AM']].copy()
    
    # Create variables for seasons and holidays
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
    tf.random.set_seed(13)
    #tf.set_random_seed(13) # use this instead depending on version of TensorFlow
    
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
    
    
    #definition for multi step plot - this shows the predictions for an individual day
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
    
    #train
    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    #validation
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
    
    # Create model for gridsearch analysis of different parameters
    def create_model(BUFFER_SIZE=BUFFER_SIZE,optimizer='rmsprop', init='glorot_uniform'):
        
        #Building the LSTM model
        multi_step_model = tf.keras.models.Sequential()
        multi_step_model.add(tf.keras.layers.LSTM(32,
                                                  return_sequences=True,
                                                  input_shape=x_train_multi.shape[-2:]))
        multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
        multi_step_model.add(tf.keras.layers.Dense(25))
        multi_step_model.add(tf.keras.layers.Dense(future_target))
        
        multi_step_model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=["mse","mae"])
        return multi_step_model
    
    # Gridsearch analysis to evaluate each of the parameters
    multi_step_model = KerasRegressor(build_fn=create_model)
    random_search = RandomizedSearchCV(estimator=multi_step_model, 
                                       param_distributions=param_grid,
                                       n_iter=n_iter_search)
    random_search.fit(x_train_multi, y_train_multi)
    print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
    # Create dataframe with best parameters
    parameters = pd.DataFrame(random_search.best_params_, index=[0])
    parameters['Airline']=airline
    parameters = parameters[['Airline','optimizer','nb_epoch','init']]
    optimizer = parameters['optimizer'][0]
    EPOCHS = parameters['nb_epoch'][0]
    init = parameters['init'][0]
    
    
    # Table of best parameters and performance
    def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                         header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                         bbox=[0, 0, 1, 1], header_columns=0,
                         ax=None, **kwargs):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            plt.title('%s Best Parameters & Performance' % airline_name,fontdict=dict(fontsize=16,fontweight='bold'),loc='center')
            ax.axis('off')
    
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, loc='center',cellLoc='center',**kwargs)
    
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)
    
        for k, cell in  six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
        plt.savefig('%s_parameters_performance.png' % airline,bbox_inches='tight')
        return ax
    
    
    
    #Building the LSTM model using best model parameters
    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(32,
                                              return_sequences=True,
                                              input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    multi_step_model.add(tf.keras.layers.Dense(25))
    multi_step_model.add(tf.keras.layers.Dense(future_target))
    
    multi_step_model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=["mse","mae"])
    
    
    
    
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
    
    parameters['RMSE'] = rmse[0].round(3)
    parameters['NRMSE'] = nrmse[0].round(3)
    
    
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(1)
    
    #plotting the sample predictions
    for x, y in val_data_multi.take(1):
      multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
    
    pred_data=pd.DataFrame([])
    est_date = dt.date(2019, 1, 1)
    # Consolidate true and predictions into a single dataframe
    for x, y in val_data_multi.take(365):
        true_val = y[:,15]*data_std[0]+data_mean[0]
        prediction = np.array(multi_step_model.predict(x)[:,15]*data_std[0]+data_mean[0])
        pred_data = pred_data.append(pd.DataFrame({'Date':est_date,'True Value':true_val,'Predicted Value':prediction}, index=[0]),ignore_index=True)
        est_date = est_date + timedelta(days=1) 
    pred_data['Predicted Value'] = round(pred_data['Predicted Value'])
    
    print(pred_data)
    
    # Begin labeling data with old labels to test accuracy
    labels = pd.read_csv('testing_cumulative_data_%s_labeled.csv' % airline)
    labels['Date'] = pd.to_datetime(labels['Date'])
    
    cluster = pred_data.copy()
    cluster['Predicted Cluster'] = ''
    cluster['True Cluster'] = ''
    
    cluster['True Cluster']=cluster['Date'].map(dict(zip(labels['Date'],labels['Cluster_Num'])))
    
    # Find cutoffs in order to label data based on predictions and compare with actual clusters
    cutoffs = pd.DataFrame([])
    for i in cluster['True Cluster'].unique():
        data = cluster[cluster['True Cluster']==i]
        val = data['True Value'].min()
        cutoffs = cutoffs.append(pd.DataFrame({'Cluster':i,'Cutoffs':val}, index=[0]),ignore_index=True)
    cutoffs = cutoffs.sort_values(by=['Cluster']).reset_index()
    cutoffs = cutoffs[['Cluster','Cutoffs']]
    
    for i in cutoffs['Cluster'].unique():
        cluster.loc[cluster['Predicted Value']>cutoffs['Cutoffs'][i],'Predicted Cluster']=i
    cluster.loc[cluster['Predicted Value']<=cutoffs['Cutoffs'][1],'Predicted Cluster']=0
    
    meltdown_cutoff = max(cutoffs['Cutoffs'])
    # Plot scatter of predictions
    fig,ax = plt.subplots(figsize=(12,8))
    meltdown = plt.scatter(pred_data[pred_data['True Value']>=meltdown_cutoff]['True Value'],pred_data[pred_data['True Value']>=meltdown_cutoff]['Predicted Value'],color='blue')
    normal = plt.scatter(pred_data[pred_data['True Value']<meltdown_cutoff]['True Value'],pred_data[pred_data['True Value']<meltdown_cutoff]['Predicted Value'],color='gray')
    plt.axhline(y=meltdown_cutoff, color='r', linestyle='-')
    plt.legend((meltdown,normal),
               ('Meltdown', 'Normal'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=12)
    plt.title('LSTM Predictions',fontsize=16)
    plt.xlabel('True Values',fontsize=14)
    plt.ylabel('Predicted Values',fontsize=14)
    plt.ylim(ymin=0)  
    plt.xlim(xmin=0) 
    plt.axis('square')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    plt.savefig('%s_scatter_plot.png' % airline)
    
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
        plt.title(title, fontdict = dict(fontsize=24))
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize = 18, rotation=45)
        plt.yticks(tick_marks, classes, fontsize = 18)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     fontsize = 20,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.grid(b=None)    
        plt.tight_layout()
        plt.ylabel('True label', fontsize = 18)
        plt.xlabel('Predicted label', fontsize = 18)
    
    y_test = cluster['True Cluster']
    y_pred = cluster['Predicted Cluster']
    
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    parameters['Cluster Acc.']=metrics.accuracy_score(y_test,y_pred).round(3)
    
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    classificationReport = classification_report(y_test, y_pred)
    cr_lines = classificationReport.split('/n')
    cr_aveTotal = cr_lines[len(cr_lines) - 2].split()
    ave_recall = float(cr_aveTotal[len(cr_aveTotal) - 3])

    parameters['Cluster Recall'] = ave_recall

    def plot_classification_report(cr, title='Classification Report ', with_avg_total=False, cmap=plt.cm.Blues):
        lines = cr.split('\n')
        classes = []
        plotMat = []
        for line in lines[2 : (len(lines) - 3)]:         #print(line)
            t = line.split()         # print(t)         
            if(len(t)==0):      
                break
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            print(v)
            plotMat.append(v)
        if with_avg_total:
            aveTotal = lines[len(lines) - 2].split()
            classes.append('avg/total')
            vAveTotal = [float(x) for x in aveTotal[2:len(aveTotal) - 1]]
            plotMat.append(vAveTotal)

        plt.figure()
        plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=16)
        plt.colorbar()
        x_tick_marks = np.arange(3)
        y_tick_marks = np.arange(len(classes))
        plt.xticks(x_tick_marks, ['Precision', 'Recall', 'F1-Score'])
        plt.yticks(y_tick_marks, classes)
        plt.grid(b=None)
        plt.tight_layout()
        plt.ylabel('Classes', fontsize=14)
        plt.xlabel('Measures', fontsize=14)
        plt.savefig('%s_classif_report.png' % airline, bbox_inches='tight')
    
    plot_classification_report(classificationReport, with_avg_total=True)
    
    
    
    #Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    #Create labels that correspond with our respective cluster labels
    if max(cluster['True Cluster'])==2:
        labs=['Good','Normal','Meltdown']
    if max(cluster['True Cluster'])==3:
        labs=['Good','Normal','Bad','Meltdown']
    if max(cluster['True Cluster'])==4:
        labs=['Great', 'Good','Normal','Bad','Meltdown']
    if max(cluster['True Cluster'])==5:
        labs=['Great', 'Good','Normal','Bad','Very Bad','Meltdown']
    
    #Plot non-normalized confusion matrix to show counts of predicted vs. actual clusters
    plt.figure()
    plt.grid(b=None)
    plot_confusion_matrix(cnf_matrix, classes=labs,
                          title='Confusion matrix, without normalization')
    plt.savefig('%s_confusion_matrix_count.png' % airline)
    
    #Plot normalized confusion matrix to show percentage of classifications in predicted vs. actual clusters
    plt.figure()
    plt.figure(figsize=(11,7))
    plot_confusion_matrix(cnf_matrix, classes=labs, normalize=True,
                          title='%s LSTM Model \nNormalized Confusion Matrix' % airline_name)
    plt.grid(b=None)
    plt.savefig('%s_confusion_matrix.png' % airline)
 
    plt.figure()       
    render_mpl_table(parameters, header_columns=0, col_width=2.0)


# To run all airlines in dataset
for i in airlines_cv['Marketing_Airline_Network'].unique():
    print(i)
    run_lstm(airline=i)

# Or select an individual airline and run one at a time
airline = 'AA'
run_lstm(airline=airline)
