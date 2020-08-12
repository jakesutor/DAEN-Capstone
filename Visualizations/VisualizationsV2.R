#Set the working directory equal to the folder address in which you keep the dataset
setwd("D:/DAEN690/New Data 07-06-20")

#load the necessary packages
library(dplyr)
library(tidyverse)  
library(cluster)    
library(factoextra)
library(gridExtra)
library(ggplot2)
library(ggrepel)

#load the dataset
data=read.csv("final_final_data.csv")
#tail(data) #available as a check to make sure data was loaded correctly

#filter out for your specific airline via filtering for its respective IATA code
#the IATA codes can be found in the file "Codes.txt"
dataXX=data %>% filter(data$Marketing_Airline_Network=="XX")
#dataWN=data %>% filter(data$Marketing_Airline_Network=="WN")

#filter out the data to get only rows for 3:00 AM EST (11:00 PM PST from previous day)
three=dataWN %>% filter(dataWN$Hour==23)
#tail(three) #same as above, another check to make sure data was loaded correctly

#set R to allow for plotting side by side
par(mfrow=c(1,2))

#build histogram using 100 bins
hist(three$ArrDelay3AM,breaks=100, main="Histogram of Southwest Airlines \n3:00am EST Cumulative Arrival Delays \n(Bins=100)", xlab="Cumulative Arrival Delay")

#build histogram using 75 bins
hist(three$ArrDelay3AM,breaks=75, main="Histogram of Southwest Airlines \n3:00am EST Cumulative Arrival Delays \n(Bins=100)", xlab="Cumulative Arrival Delay")

#clear plot area and reset R to allow for only one plot to appear in the Plots window
dev.off()

#Extract the Arrival Delays at 3:00 AM from each day as a single data frame
threeADs=three$ArrDelay3AM

#set seed for repeatability
set.seed(71)

#create function to compute total within-cluster sum of square 
wss <- function(k) {
  kmeans(threeADs, k, nstart = 10 )$tot.withinss
}

#set test values to test each of k = 1 through k = 15
k.values <- 1:15

#compute wss for 2-15 clusters
wss_values <- map_dbl(k.values, wss)

#create Clusters vs. WSS plot
plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares",
     main="Number of Clusters vs. Within-Clusters Sum of Squares")

#create a k-means clustering algorithm for the data using k=3 clusters
threeCluster <- kmeans(threeADs, 3, nstart = 20)
#threeCluster #summary of algorithm results

#filter out the data to get only rows for 12:00 PM EST (8:00 AM PST)
noon=dataWN %>% filter(dataWN$Hour==8)

#extract cluster labels for each day
Clusters <- as.factor(threeCluster$cluster)

#set colors for scatterplot
cols <- c("1" = "blue", "2" = "red", "3" = "green")

#create scatterplot of noon vs. 3:00 am
ggplot(three, aes(noon$ArrDelay3AM, three$ArrDelay3AM , color = Clusters)) + geom_point() +
  ggtitle("Southwest Airlines Cumulative Delays (in Minutes) \nNoon vs. Eleven PM")+theme(plot.title = element_text(hjust = 0.5)) +
  xlab("Cumulative Delay at Noon") +
  ylab("Cumulative Delay at 11:00 pm")+
  scale_colour_manual(values = cols, breaks=c("3","1","2"),labels=c("Good", "Normal", "Meltdown"))

#extract the cluster values from the algorithm summary
threeCluster$cluster

#first=as.Date.default()

#create a data frame with each date and its corresponding cluster value
dateClus=cbind.data.frame(three$Date,threeCluster$cluster)
#tail(dateClus) #check to ensure formatting is correct

#create the file with all data from your specific airline
write.csv(dataWN,"final_final_data_WN.csv")

#create the file with each airline's dates and cluster values
write.csv(dateClus,"ClustersByDateWN.csv")

#manually add the clusters to each hour of its respective day using VLOOKUP in Excel
