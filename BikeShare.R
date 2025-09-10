library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(DataExplorer)

#Load in and clean data
training_data <- vroom("C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/bike-sharing-demand/train.csv")
training_data$weather <- as.factor(training_data$weather)

#EDA
plot_correlation(training_data)

#Plots
scatter_humidity <- ggplot(data = training_data, aes(x= humidity, y=count)) +
  geom_point()
barplot_weather <- ggplot(data = training_data, aes(x= humidity, y=count)) +
  geom_bar()
