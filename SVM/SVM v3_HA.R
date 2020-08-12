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

# Load in the airline specific data 
dataHA = read.csv("testing_cumulative_data_HA_original clusters.csv")

attach(dataHA)
set.seed(321)

#Split the data into a training & test set (30/70)
sample <- sample.split(dataHA, SplitRatio = .7)
train <- subset(dataHA, sample == TRUE)
test <- subset(dataHA, sample == FALSE)

write.csv(train, "train_HA_original clusters.csv")

trainUpdated = read.csv("updated_train_HA.csv")


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


write.csv(pred, "HA_SVM Predictions_First8hrPreds.csv")
write.csv(y_test, "Y Test Data_HA.csv")

### consider confusion matrices for initial visual
classificationsHA = read.csv("predicted-actual-HA.csv")

predicted=classificationsHA$Predicted
actual=classificationsHA$Actual

# Confusion Matrix 
# cm <- confusionMatrix(predicted, actual, positive = NULL, mode = "prec_recall")
cm <- confusionMatrix(predicted, actual, positive = NULL, mode = "everything")



# https://ragrawal.wordpress.com/2011/05/16/visualizing-confusion-matrix-in-r/
df <- data.frame(cm$table)

tile <- ggplot() +
  geom_tile(aes(x=Reference, y=Prediction,fill=Percent),data=df, color="black",size=0.1) +
  labs(x="Actual",y="Predicted")
tile = tile + 
  geom_text(aes(x=Reference,y=Prediction, label=sprintf("%.1f", Freq)),data=df, size=3, colour="black") +
  scale_fill_gradient(low="white",high="light blue")
tile = tile + 
  geom_tile(aes(x=Reference,y=Prediction),data=subset(df, as.character(Reference)==as.character(Prediction)), color="black",size=0.3, fill="black", alpha=0) 

#render
tile




