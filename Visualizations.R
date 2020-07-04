setwd("D:/DAEN690/New Data 06-29-20")
library(dplyr)
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra)
library(gridExtra)
library(ggplot2)
library(ggrepel)
data=read.csv("combined_final_data_WN.csv")
#tail(data)

eleven=data %>% filter(data$ACT_ARR_HOUR==23)
#tail(eleven)

par(mfrow=c(1,2))
hist(eleven$ArrDelayMinutes,breaks=100, main="Histogram of Southwest Airlines \n11:00pm Cumulative Arrival Delays \n(Bins=100)", xlab="Cumulative Arrival Dealy")
hist(eleven$ArrDelayMinutes,breaks=75,, main="Histogram of Southwest Airlines \n11:00pm Cumulative Arrival Delays \n(Bins=75)", xlab="Cumulative Arrival Dealy")
?hist
dev.off()

elevenADs=eleven$ArrDelayMinutes


set.seed(71)
# function to compute total within-cluster sum of square 
wss <- function(k) {
  kmeans(elevenADs, k, nstart = 10 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values <- 1:15

# extract wss for 2-15 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

# k.values <- 3:15
# 
# # extract wss for 2-15 clusters
# wss_values <- map_dbl(k.values, wss)
# 
# plot(k.values, wss_values,
#      type="b", pch = 19, frame = FALSE, 
#      xlab="Number of clusters K",
#      ylab="Total within-clusters sum of squares")


elevenCluster <- kmeans(elevenADs, 3, nstart = 20)
elevenCluster

noon=data %>% filter(data$ACT_ARR_HOUR==12)

Clusters <- as.factor(elevenCluster$cluster, levels = c("2", "1", "3"))

# geom_text(data=subset(DefDF[DefDF$ShotsCont >= 150,], 
#                       ShotsNear >= 700 | OppPPP <= .9 | OppPPP >= 1.1), 
#           aes(label = DefenderList), size = 4, color = "black", vjust = -0.5, check_overlap = TRUE)

cluslab <- levels(Clusters)
cluslab = c("2", "1", "3")

cluslab <- factor(elevenCluster$cluster, levels = c("2", "1", "3"))


cols <- c("1" = "blue", "2" = "green", "3" = "red")


ggplot(eleven, aes(noon$ArrDelayMinutes, eleven$ArrDelayMinutes , color = Clusters)) + geom_point() +
ggtitle("Southwest Airlines Cumulative Delays (in Minutes) \nNoon vs. Eleven PM")+theme(plot.title = element_text(hjust = 0.5)) +
xlab("Cumulative Delay at Noon") +
ylab("Cumulative Delay at 11:00 pm")+
  # scale_colour_discrete(name  ="Severity",
  #                       breaks=c("2", "1", "3"),
  #                       labels=c("Good", "Medium", "Meltdown"))+
  scale_colour_manual(values = cols, breaks=c("2","1","3"),labels=c("Good", "Medium", "Meltdown"))

elevenCluster$cluster

first=as.Date.default()

dateClus=cbind.data.frame(eleven$ACT_ARR_DATE,elevenCluster$cluster)
dateClus

write.csv(dateClus,"ClustersByDate.csv")
