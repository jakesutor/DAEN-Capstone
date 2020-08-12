# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:38:44 2020

@author: Chas Cantrell
"""
#Prior to beginning the modeling steps of this project, a few data cleansing steps were taken to ensure
#the data would be suited for a multiclass classification model. First, the clusters calculated from the
#K-Means Clustering process (the process carried out in the R-files titled "Visualizations_XX.R" where
#"XX" represents an airline's IATA code) were re-numbered in Microsoft Excel using and "IF" statement
#to ensure that the cluster numbers were increasing numerically as the delay severity increased. Second, 
#these cluster numbers were then assigned to each date within the dataset using Microsoft Excel's 
#"VLOOKUP" command. Finally, upon recommendations from the team's sponsor and experimentation with the
#data and XGBoosting models, each respective airline's dataset was equalized so that it contained clusters
#that were of relatively equal size (roughly within 30 days worth of datapoints of each other). This was
#simply done by replicating datapoints within the dataset using a simply "copy-and-paste" method. The
#"cluster-equalized" dataset for this file's respective airline is what is used throughout this file.

#This file models for Southwest Airlines (IATA code "WN")

#Import packages and necessary functions
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import itertools
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import train_test_split

#read dataset into Python
WN=pd.read_csv("D:/DAEN690/New Data 07-06-20/testing_cumulative_data_WN_labeled_train.csv")
#verify data read in correctly
WN
#view dimensions of dataset
WN.shape

#split off predictor variables
X=WN[["0","1","2","3","4","5","6","7","8"]]
#verify this extraction was done correctly
X
#split off target variable
y=WN["Cluster_Num"]
#verify this extraction was done correctly
y

#split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=71)

#convert training and test sets into DMatrices 
#XGBoost likes DMatrices b/c it's a structure optimized for memory efficiency and training speed
dtrain=xgb.DMatrix(X_train, label=y_train)
dtest=xgb.DMatrix(X_test, label=y_test)


#Define the initial parameter sets. We will use a set of parameters that seem reasonable to start with
#but we will optimize these parameters through the next few steps.
params = {
    # Parameters that we are going to tune.
    'max_depth':6, #the maximum depth of a tree within the model. Larger values make the model more complex
    'min_child_weight': 1, #the minimum sum of instance weight needed in a tree's "child". 
    'eta':.3, #step-size shrinkage used to prevent overfitting of the model
    'subsample': 1, #how much of the training data is sampled prior to growing the model's trees
    'colsample_bytree': 1, #subsample of ratio of columns used when constructing each tree
    # Other parameters
    'objective':'multi:softprob', #the model learning objective. Since we are doing a multiclass classification, we can use either "multi:softprob" or "multi:softmax"
    'num_class': 5 #the number of clusters in your dataset.
}

params['eval_metric'] = "mlogloss" #the evaluation metric for the model. What determines whether the model is good or not. We are using Multiclass Logloss, though Multiclass Error ("merror") is another option

num_boost_round = 999 #The initial number of trees to build. While seemingly large, this value will work together with "early_stopping_rounds" found in the model to find the optimal number of rounds before the model stops improving

#Build the intial XGBoost model using the initial set of parameters
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)

#Perform a cross-validation on the model to see what the best mlogloss is when using our inital parameters
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mlogloss'},
    early_stopping_rounds=10
)

#Print the results of the cross-validation to see how each iteration of the model performed
cv_results

#Extract the minimum mlogloss value from the test dataset when using the initial set of parameters 
cv_results['test-mlogloss-mean'].min()

#Now we are going to begin optimizing the parameters found in the "params" set above. We will do so using grid-search

#Begin with optimizing "max_depth" and "min_child_weight". These parameters control the complexity of the model's trees so it is important to tune them together
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(3,10)
    for min_child_weight in range(1,6)
]

#Create "min_mll" to store the minimum mlogloss value found during the optimization process below
min_mll = float("Inf")
#Create an empty "best_params" vector to store the two optimal values that will be found during the for loop below
best_params = None

#Perform the grid search to find the optimal values for "max_depth" and "min_child_weight"
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mlogloss'},
        early_stopping_rounds=10
    )
    # Update best MAE
    mean_mll = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print("\tMLogLoss {} for {} rounds".format(mean_mll, boost_rounds))
    if mean_mll < min_mll:
        min_mll = mean_mll
        best_params = (max_depth,min_child_weight)
        
#Print the optimal values for these two parameters and their respective mlogloss to track how the model improves        
print("Best params: {}, {}, MLogLoss: {}".format(best_params[0], best_params[1], min_mll))

#Update our previous parameter set with the new values for "max_depth" and "min_child_weight"
params = {
    # Parameters that we are going to tune.
    'max_depth':9,
    'min_child_weight': 2,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'multi:softprob',
    'num_class': 5,
    'eval_metric':'mlogloss'
}

#Repeat the grid search process, this time optimizing "subsample" and "colsample_bytree"
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(5,10)]
    for colsample in [i/10. for i in range(5,10)]
]

min_mll = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mlogloss'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_mll = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print("\tMLogLoss {} for {} rounds".format(mean_mll, boost_rounds))
    if mean_mll < min_mll:
        min_mll = mean_mll
        best_params = (subsample,colsample)
print("Best params: {}, {}, MLogLoss: {}".format(best_params[0], best_params[1], min_mll))


#Update our previous parameter set with the new values for "subsample" and "colsample_bytree"
params = {
    # Parameters that we are going to tune.
    'max_depth':9,
    'min_child_weight': 2,
    'eta':.3,
    'subsample': 0.9,
    'colsample_bytree': 0.6,
    # Other parameters
    'objective':'multi:softprob',
    'num_class': 5,
    'eval_metric':'mlogloss'
}

#Repeat the grid search process, this time optimizing "eta" 
min_mll = float("Inf")
best_params = None
for eta in [.3, .2, .1, .05, .01]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta 
    %time cv_results = xgb.cv(params,dtrain,num_boost_round=num_boost_round,seed=42,nfold=5, metrics=['mlogloss'],early_stopping_rounds=10)
    # Update best score
    mean_mll = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print("\tMLogLoss {} for {} rounds\n".format(mean_mll, boost_rounds))
    if mean_mll < min_mll:
        min_mll = mean_mll
        best_params = eta
print("Best params: {}, MLogLoss: {}".format(best_params, min_mll))

#Update our previous parameter set with the new value for "eta"
params = {
    # Parameters that we are going to tune.
    'max_depth':9,
    'min_child_weight': 2,
    'eta':0.01,
    'subsample': 0.9,
    'colsample_bytree': 0.6,
    # Other parameters
    'objective':'multi:softprob',
    'num_class': 5,
    'eval_metric':'mlogloss'
}

#Now we will build our "optimal" XGBoost model. Start by defining the number of boosting rounds we will use.
num_round=200

#Train the XGBoost model, passing through the parameters, training dataset, and number of rounds
bst = xgb.train(params, dtrain, num_round)

#read dataset into Python
WNtest=pd.read_csv("D:/DAEN690/New Data 07-06-20/testing_cumulative_data_WN_labeled_test.csv")
#verify data read in correctly
WNtest
#view dimensions of dataset
WNtest.shape

#split off predictor variables
Xtest=WNtest[["0","1","2","3","4","5","6","7","8"]]
#verify this extraction was done correctly
Xtest
#split off target variable
ytest=WNtest["Cluster_Num"]
#verify this extraction was done correctly
ytest

#convert training and test sets into DMatrices 
#XGBoost likes DMatrices b/c it's a structure optimized for memory efficiency and training speed
dtest=xgb.DMatrix(Xtest, label=ytest)


#Make predictions on the hold-out test dataset for delay severity
preds = bst.predict(dtest)
#Print the predictions to see the probabilities that a day is labeled as each cluster
preds
#Label each day according to which cluster they are most likely apart of.
best_preds = np.asarray([np.argmax(line) for line in preds])
#Print these predicted cluster labels
best_preds

#Print the precision score for the model. That is, how many points were correctly classified
print( "Numpy array precision:", precision_score(ytest, best_preds, average='macro'))

#Create some labels for each cluster 
labelsWN=[0,1,2,3,4]

#Create a confusion matrix to see how the model performed in classifying for each cluster
cmtx = pd.DataFrame(
    confusion_matrix(ytest, best_preds, labels=labelsWN), 
    index=['Great', 'Good','Normal','Bad','Meltdown'], 
    columns=['Great', 'Good','Normal','Bad','Meltdown']
)
print(cmtx)


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
cnf_matrix = confusion_matrix(ytest, best_preds)
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
                      title='Southwest Airlines XGBoost Model \nNormalized Confusion Matrix')

plt.show()
