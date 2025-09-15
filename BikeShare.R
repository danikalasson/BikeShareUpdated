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

#Data Mutation and feature engineering
my_recipe <- recipe(Response~..., data=training_data) %>%
  step_mutate(count = log(count))#Turn response var to be log count
  step_mutate(weather=ifselse(4 ==3,weather)) %>% #Change weather factor
  step_mutate(weather=as.factor(weather)) %>% #Create a new variable
  step_mutate(season = as.factor(season)) %>% #Create polynomial expansion of var
  step_date(datetime, features=c("hour")) %>% # gets hour of week
  step_time(datetime, features=c("dow"))#create day variable
  
prepped_recipe <- prep(my_recipe)
baked_train <- bake(prepped_recipe, new_data = NULL)
baked_test <- bake(prepped_recipe, new_data = test_data)
  

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

bike_workflow <- workflow() %>%
  add_model(my_linear_model) %>%
  add_recipe(my_recipe)

# Fit model
bike_fit <- fit(bike_workflow, data = training_data)

# Predict on test data (log scale first)
bike_predictions <- predict(bike_fit, new_data = test_data)

# Back-transform log(count) -> count
bike_predictions <- bike_predictions %>%
  mutate(count = exp(.pred)) %>%
  select(count)

#Kaggle Format
kaggle_submission <- bike_predictions %>%
bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write out the file9
vroom_write(x=kaggle_submission, file="C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/LinearPreds.csv", delim=",")
