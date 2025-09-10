library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(DataExplorer)
library(dplyr)
library(patchwork)

#Load in and clean data
training_data <- vroom("C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/bike-sharing-demand/train.csv")
training_data$weather <- as.factor(training_data$weather)
training_data$season <- as.factor(training_data$season)
training_data <- training_data %>%
  select(-casual, -registered)

test_data <- vroom("C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/bike-sharing-demand/test.csv")
test_data$weather <- as.factor(test_data$weather)
test_data$season <- as.factor(test_data$season)

#EDA
glimpse(training_data)
plot_correlation(training_data)
plot_missing(training_data)

#Plots
scatter_humidity <- ggplot(data = training_data, aes(x= humidity, y=count)) +
  geom_point()
barplot_weather <- ggplot(data = training_data, aes(x= weather)) +
  geom_bar()
scatter_temp <- ggplot(data = training_data, aes(x= temp, y=count, color = weather)) +
  geom_point() +
  geom_smooth()
scatter_windspeed <- ggplot(data = training_data, aes(x= windspeed, y = count)) +
  geom_point()

(scatter_humidity + barplot_weather)/(scatter_temp + scatter_windspeed)

#Predictions
my_linear_model <- linear_reg()%>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula = count ~ .-datetime, data = training_data)

bike_predictions <- predict(my_linear_model, new_data =test_data)
bike_predictions