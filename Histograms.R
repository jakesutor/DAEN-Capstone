#Set the working directory equal to the folder address in which you keep the dataset
setwd("C:/Users/jakes/Downloads")

#load the necessary packages
library(dplyr)
library(tidyverse)  
library(cluster)    
library(factoextra)
library(gridExtra)
library(ggplot2)
library(ggrepel)
library(tidyr)
library(egg)
library(patchwork)
library(cowplot)
library(grid)
library(gridExtra)

#load the dataset
#data=read.csv("final_final_data.csv")
#tail(data) #available as a check to make sure data was loaded correctly
airlines_cv = read.csv('airlines.csv')
airlines_cv$Marketing_Airline_Network <- factor(airlines_cv$Marketing_Airline_Network, levels = c("AS", "HA","DL","NK","AA","F9","B6","WN","UA","G4"))
airlines_cv = airlines_cv[order(airlines_cv$Marketing_Airline_Network),]

#i = 'AA'
#data_file = read.csv(file_name)

#par(mfrow=c(3,4))
options(scipen=100000)
p <- list()
for (i in airlines_cv$Marketing_Airline_Network){
  airline_name = airlines_cv$Airline[airlines_cv$Marketing_Airline_Network==i]
  file_name = sprintf('testing_cumulative_data_%s_labeled.csv', i)
  data_file = read.csv(file_name)
  my_vars = c('Date','X23','Cluster')
  data_file=data_file[my_vars]
  
  num_clusters = length(unique(data_file$Cluster))
    if(num_clusters == 3){
      data_file$Cluster <- factor(data_file$Cluster,levels = c("Good", "Normal","Meltdown"));
      cols <- c("Good" = "green", "Normal" = "blue", "Meltdown" = "red");
    }else if (num_clusters == 4){
      data_file$Cluster <- factor(data_file$Cluster,levels = c("Good", "Normal","Bad","Meltdown"))
      cols <- c("Good" = "green", "Normal" = "blue", "Bad" = "orange","Meltdown" = "red");
    }else if (num_clusters == 5){
      data_file$Cluster <- factor(data_file$Cluster,levels = c("Great","Good", "Normal","Bad","Meltdown"))
      cols <- c("Great" = "darkgreen","Good" = "green", "Normal" = "blue", "Bad" = "orange","Meltdown" = "red");
    }else{
      data_file$Cluster <- factor(data_file$Cluster,levels = c("Great","Good", "Normal","Bad","Very Bad","Meltdown"))
      cols <- c("Great" = "darkgreen","Good" = "green", "Normal" = "blue", "Bad" = "orange","Very Bad" = "darkorange3","Meltdown" = "red");
    }

  
  # Change histogram plot line colors by groups
  if (i !='G4'){
    p[[i]] <- ggplot(data_file, aes(x=X23, fill=Cluster,color=Cluster)) +
      geom_histogram(alpha=0.5,position="identity", bins=100, show.legend = FALSE) +
      scale_color_manual(values=cols)+
      scale_fill_manual(values=cols) + 
      ylim(0,80) +
      labs(title=airline_name,x="Cumulative Delays (in minutes)", y = "Count", col="Cluster",fill="Cluster")
  } else{
    p[[i]] <- ggplot(data_file, aes(x=X23, fill=Cluster,color=Cluster)) +
      geom_histogram(alpha=0.5,position="identity", bins=100) +
      scale_color_manual(values=cols)+
      scale_fill_manual(values=cols) + 
      ylim(0,80) +
      labs(title=airline_name,x="Cumulative Delays (in minutes)", y = "Count", col="Cluster",fill="Cluster")
  }
}

plot = plot_grid(p[['AS']]+ theme(axis.title.y = element_blank(),
                                  axis.title.x = element_blank()), 
                   p[['HA']] + theme(axis.text.y = element_blank(),
                                    axis.title.y = element_blank(),
                                    axis.title.x = element_blank() ), 
                   p[['DL']] + theme(axis.text.y = element_blank(),
                                   axis.title.y = element_blank(),
                                   axis.title.x = element_blank() ),
                   p[['NK']] + theme(axis.text.y = element_blank(),
                                     axis.title.y = element_blank(),
                                     axis.title.x = element_blank() ), 
                   nrow = 1)
second_row <- plot_grid(
  p[['AA']]+ theme(axis.title.y = element_blank(),
                   axis.title.x = element_blank()),
  p[['F9']] + theme(axis.text.y = element_blank(),
                    axis.title.y = element_blank(),
                    axis.title.x = element_blank() ), 
  p[['B6']] + theme(axis.text.y = element_blank(),
                    axis.title.y = element_blank(),
                    axis.title.x = element_blank() ),
  p[['WN']] + theme(axis.text.y = element_blank(),
                    axis.title.y = element_blank(),
                    axis.title.x = element_blank() ), 
  nrow = 1
)

bottom_row <- plot_grid(
  p[['UA']]+ theme(axis.title.y = element_blank(),
                   axis.title.x = element_blank()),
  p[['G4']] + theme(axis.text.y = element_blank(),
                    axis.title.y = element_blank(),
                    axis.title.x = element_blank() )
  )

#do.call(grid.arrange,c(p, ncol=4))

y.grob <- textGrob("Count", 
                   gp=gpar(fontsize=14), rot=90)

x.grob <- textGrob("Cumulative Delays (in minutes)", 
                   gp=gpar(fontsize=14))

#add to plot

grid.arrange(arrangeGrob(plot, second_row, bottom_row, ncol = 1, left = y.grob, bottom = x.grob))


