#install.packages("e1071")
library("e1071")
#install.packages("dplyr",dependencies=TRUE)
library(dplyr)
library(tidyverse)  
library(cluster)    
#install.packages("factoextra")
library(factoextra)
library(gridExtra)
library(ggplot2)
#install.packages("ggrepel")
library(ggrepel)
#install.packages("caTools")
library(caTools)
#install.packages("tidyr")
library(tidyr)
#install.packages("broom")
library(broom)
#install.packages("caret")
library(caret)
#install.packages("yardstick")
library(yardstick)
#install.packages("cvms")
library(cvms)
library(broom)    
library(tibble)

#Set the working directory equal to the folder address in which you keep the dataset
setwd("C:/Users/jakes/Downloads")

# Load in the airline specific data 
data = read.csv("testing_cumulative_data.csv")
data = data[data$Delay.Type=="ArrDelay3AM",]

dataAS = data[data$Marketing_Airline_Network=="AS",]

#attach(dataAS)
set.seed(321)

#Split the data into a training & test set (30/70)
sample <- sample.split(dataAS, SplitRatio = .7)
train <- subset(dataAS, sample == TRUE)
test <- subset(dataAS, sample == FALSE)

write.csv(train, "train_AS_3_clusters.csv")

trainUpdated = read.csv("updated_train_AS.csv")


### Will eventually need to do x&y twice... one for test set, one for training set
#x = predictors
x_train <- trainUpdated[,0:8]
x_test <- test[,0:8]

y_train <- trainUpdated[,"Cluster"]
y_test <- test[,"Cluster"]

model <- svm(x_train, y_train, probability = TRUE)  ###Likely to be "x-train" & "y-train"
pred_prob <- predict(model, x_test, decision.values = TRUE, probability = TRUE) ###Likely to be "x-test"

### y-test will be used as a comparison to the predicted clusters to evaluate quality of predictions

###Add in cluster label next to each prediction, i.e. "predicted cluster"

pred = attr(pred_prob, "probabilities")


write.csv(pred, "AS_SVM Predictions_First8hrPreds.csv")
write.csv(y_test, "Y Test Data AS.csv")


########################### START HERE 

### consider confusion matrices for initial visual
classificationsAS = read.csv("predicted-actual-AS.csv")

predicted=classificationsAS$Predicted
actual=classificationsAS$Actual

# Confusion Matrix 
# cm <- confusionMatrix(predicted, actual, positive = NULL, mode = "prec_recall")
cm <- confusionMatrix(predicted, actual, positive = NULL, mode = "everything")

recall(cm$table)

# https://ragrawal.wordpress.com/2011/05/16/visualizing-confusion-matrix-in-r/
df <- data.frame(cm$table)

tile <- ggplot() +
  geom_tile(aes(x=Reference, y=Prediction,fill=Freq),data=df, color="black",size=0.1) +
  labs(x="Actual",y="Predicted")
tile = tile + 
  geom_text(aes(x=Reference,y=Prediction, label=sprintf("%.1f", Freq)),data=df, size=3, colour="black") +
  scale_fill_gradient(low="white",high="light blue")
tile = tile + 
  geom_tile(aes(x=Reference,y=Prediction),data=subset(df, as.character(Reference)==as.character(Prediction)), color="black",size=0.3, fill="black", alpha=0) 

#render
tile




