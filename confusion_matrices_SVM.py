# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 23:10:31 2020

@author: jakes
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import six
import itertools
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Read in data
airlines_cv = pd.read_csv('airlines.csv')

airline = 'AS'
airline_name = 'Alaska Airlines'

# Create table with parameters and performance
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    airline = data['Airline'][0]
    airline_name = airlines_cv[airlines_cv['Marketing_Airline_Network']==airline]['Airline'].reset_index()
    airline_name = airline_name['Airline'][0]
    
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        plt.title('%s SVM Performance' % airline_name,fontdict=dict(fontsize=16,fontweight='bold'),loc='center')
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
    plt.savefig('%s_SVM_performance.png' % airline,bbox_inches='tight')
    return ax

# Create classification chart for each airline
def create_graphs(airline):
    classifications = pd.read_csv("predicted-actual-%s.csv" % airline)
    airline_name = airlines_cv[airlines_cv['Marketing_Airline_Network']==airline]['Airline'].reset_index()
    airline_name = airline_name['Airline'][0]
    # Set cluster classifications based on number of clusters
    if len(classifications['Actual'].unique()) == 3:
        classifications.loc[classifications['Actual']=='Good','Actual'] = 0
        classifications.loc[classifications['Actual']=='Normal','Actual'] = 1
        classifications.loc[classifications['Actual']=='Meltdown','Actual'] = 2
        classifications.loc[classifications['Predicted']=='Good','Predicted'] = 0
        classifications.loc[classifications['Predicted']=='Normal','Predicted'] = 1
        classifications.loc[classifications['Predicted']=='Meltdown','Predicted'] = 2
    if len(classifications['Actual'].unique()) == 4:
        classifications.loc[classifications['Actual']=='Good','Actual'] = 0
        classifications.loc[classifications['Actual']=='Normal','Actual'] = 1
        classifications.loc[classifications['Actual']=='Bad','Actual'] = 2
        classifications.loc[classifications['Actual']=='Meltdown','Actual'] = 3
        classifications.loc[classifications['Predicted']=='Good','Predicted'] = 0
        classifications.loc[classifications['Predicted']=='Normal','Predicted'] = 1
        classifications.loc[classifications['Predicted']=='Bad','Predicted'] = 2
        classifications.loc[classifications['Predicted']=='Meltdown','Predicted'] = 3
    if len(classifications['Actual'].unique()) == 5:
        classifications.loc[classifications['Actual']=='Great','Actual'] = 0
        classifications.loc[classifications['Actual']=='Good','Actual'] = 1
        classifications.loc[classifications['Actual']=='Normal','Actual'] = 2
        classifications.loc[classifications['Actual']=='Bad','Actual'] = 3
        classifications.loc[classifications['Actual']=='Meltdown','Actual'] = 4
        classifications.loc[classifications['Predicted']=='Great','Predicted'] = 0
        classifications.loc[classifications['Predicted']=='Good','Predicted'] = 1
        classifications.loc[classifications['Predicted']=='Normal','Predicted'] = 2
        classifications.loc[classifications['Predicted']=='Bad','Predicted'] = 3
        classifications.loc[classifications['Predicted']=='Meltdown','Predicted'] = 4
    if len(classifications['Actual'].unique()) == 6:
        classifications.loc[classifications['Actual']=='Great','Actual'] = 0
        classifications.loc[classifications['Actual']=='Good','Actual'] = 1
        classifications.loc[classifications['Actual']=='Normal','Actual'] = 2
        classifications.loc[classifications['Actual']=='Bad','Actual'] = 3
        classifications.loc[classifications['Actual']=='Very Bad','Actual'] = 4
        classifications.loc[classifications['Actual']=='Meltdown','Actual'] = 5
        classifications.loc[classifications['Predicted']=='Great','Predicted'] = 0
        classifications.loc[classifications['Predicted']=='Good','Predicted'] = 1
        classifications.loc[classifications['Predicted']=='Normal','Predicted'] = 2
        classifications.loc[classifications['Predicted']=='Bad','Predicted'] = 3
        classifications.loc[classifications['Predicted']=='Very Bad','Predicted'] = 4
        classifications.loc[classifications['Predicted']=='Meltdown','Predicted'] = 5
    
    
    y_pred = classifications['Predicted']
    y_test = classifications['Actual']
    print(classification_report(y_test, y_pred))
    
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
        plt.grid(b=None)    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    #Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    #Create labels that correspond with our respective cluster labels
    if max(y_pred)==2:
        labs=['Good','Normal','Meltdown']
    if max(y_pred)==3:
        labs=['Good','Normal','Bad','Meltdown']
    if max(y_pred)==4:
        labs=['Great', 'Good','Normal','Bad','Meltdown']
    if max(y_pred)==5:
        labs=['Great', 'Good','Normal','Bad','Very Bad','Meltdown']
    
    #Plot non-normalized confusion matrix to show counts of predicted vs. actual clusters
    plt.figure()
    plt.grid(b=None)
    plot_confusion_matrix(cnf_matrix, classes=labs,
                          title='Confusion matrix, without normalization')
    plt.savefig('%s_SVM_confusion_matrix_count.png' % airline)
    
    
    #Plot normalized confusion matrix to show percentage of classifications in predicted vs. actual clusters
    plt.figure()
    plt.figure(figsize=(11,7))
    plot_confusion_matrix(cnf_matrix, classes=labs, normalize=True,
                          title='%s SVM Model \nNormalized Confusion Matrix' % airline_name)
    plt.grid(b=None)
    plt.savefig('%s_SVM_confusion_matrix.png' % airline)
    
    parameters = pd.DataFrame(index=[0])
    parameters['Airline']=airline
    parameters['Cluster Acc.']=metrics.accuracy_score(y_test,y_pred).round(3)
    classificationReport = classification_report(y_test, y_pred)
    cr_lines = classificationReport.split('/n')
    cr_aveTotal = cr_lines[len(cr_lines) - 2].split()
    ave_recall = float(cr_aveTotal[len(cr_aveTotal) - 3])

    parameters['Cluster Recall'] = ave_recall

    render_mpl_table(parameters, header_columns=0, col_width=2.0)

# iterate through airlines
for i in airlines_cv['Marketing_Airline_Network'].unique():
    print(i)
    create_graphs(airline=i)
