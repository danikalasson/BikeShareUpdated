library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(DataExplorer)
library(dplyr)
library(patchwork)
library(glmnet)
library(rpart)

#Load in and clean data
training_data <- vroom("C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/bike-sharing-demand/train.csv")
training_data$weather <- as.factor(training_data$weather)
training_data$season <- as.factor(training_data$season)
training_data <- training_data %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

test_data <- vroom("C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/bike-sharing-demand/test.csv")
test_data$weather <- as.factor(test_data$weather)
test_data$season <- as.factor(test_data$season)

# Recipe
my_recipe <- recipe(count ~ ., data = training_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(season = as.factor(season)) %>%
  step_time(datetime, features = c("hour")) %>%
  step_date(datetime, features = c("dow")) %>%
  step_rm(datetime)%>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet13
bake(prepped_recipe, new_data=training_data)  

#Penalized Regression Section#################################################################################################
#Penalized Regression Try 1
preg_model1 <- linear_reg(penalty=0.1, mixture=0) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R
preg_wf1 <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model1) %>%
  fit(data=training_data)
predict(preg_wf1, new_data=test_data)
  
# Predict with penalized regression workflow
preg_predictions1 <- predict(preg_wf1, new_data = test_data) %>%
  mutate(count = exp(.pred)) %>%   # back-transform log scale
  select(count)

#Kaggle Format
kaggle_submission_preg1 <- preg_predictions1 %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, count) %>% #Just keep datetime and prediction variables
  rename(count=count) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>%#pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))
## Write out the file
vroom_write(x=kaggle_submission_preg1, file="C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/PenalizedPreds.csv", delim=",")


#Cross-Validation Work###################################################################################
L <- 5
K <- 3
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model)

## Grid of values to tune over13
grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = L) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(training_data, v = K, repeats=1)

## Run the CV
CV_results <- preg_wf %>%
tune_grid(resamples=folds,
          grid=grid_of_tuning_params,
          metrics=metric_set(rmse, mae)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best(metric="rmse")

## Finalize the Workflow & fit it
final_wf <-preg_wf %>%
finalize_workflow(bestTune) %>%
fit(data=training_data)

## Predict
final_preds <- final_wf %>%
predict(new_data = test_data) %>%
  mutate(count = exp(.pred)) %>%   # back-transform log scale
  select(count)

#Kaggle Format
kaggle_submission_final_wf<- final_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, count) %>% #Just keep datetime and prediction variables
  rename(count=count) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>%#pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))
## Write out the file
vroom_write(x=kaggle_submission_preg1, file="C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/final_wf.csv", delim=",")


#Regression Trees############################################################################
tree_recipe <- recipe(count ~ ., data = training_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(season = as.factor(season)) %>%
  step_time(datetime, features = c("hour")) %>%
  step_date(datetime, features = c("dow")) %>%
  step_rm(datetime)%>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # What R function to use
  set_mode("regression")

## Set Workflow
tree_wf <- workflow() %>%
  add_recipe(tree_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(tree_depth(),
                                      cost_complexity(),
                                      min_n(),
                                      levels=L) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(training_data, v = K, repeats=1)

## Run the CV
CV_results <- tree_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae)) #Or leave metrics NULL


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric="rmse")

## Finalize the Workflow & fit it
final_tree_wf <-tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=training_data)

## Predict
final_tree_preds <- final_tree_wf %>%
  predict(new_data = test_data) %>%
  mutate(count = exp(.pred)) %>%   # back-transform log scale
  select(count)

#Kaggle Format
kaggle_submission_tree_wf<- final_tree_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, count) %>% #Just keep datetime and prediction variables
  rename(count=count) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>%#pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))
## Write out the file
vroom_write(x=kaggle_submission_tree_wf, file="C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/final_tree_wf.csv", delim=",")


#Random Forests###################################################################################
library(ranger)
L<-3
K<-3
maxNumXs <- 4
forest_recipe <- recipe(count ~ ., data = training_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(season = as.factor(season)) %>%
  step_time(datetime, features = c("hour")) %>%
  step_date(datetime, features = c("dow")) %>%
  step_rm(datetime)%>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=1000) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

## Create a workflow with model & recipe
forest_wf <- workflow() %>%
  add_recipe(forest_recipe) %>%
  add_model(my_mod)
## Set up grid of tuning values
mygrid <- grid_regular(mtry(range = c(1, maxNumXs)),
                       min_n(),
                       levels = L)
## Set up K-fold CV
folds <- vfold_cv(training_data, v = K, repeats=1)

## Run the CV
CV_results <- forest_wf %>%
  tune_grid(resamples=folds,
            grid=mygrid,
            metrics=metric_set(rmse, mae)) #Or leave metrics NULL


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric="rmse")
## Finalize workflow and predict
final_forest_wf <-forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=training_data)

## Predict
final_forest_preds <- final_forest_wf %>%
  predict(new_data = test_data) %>%
  mutate(count = exp(.pred)) %>%   # back-transform log scale
  select(count)

#Kaggle Format
kaggle_submission_forest_wf<- final_forest_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, count) %>% #Just keep datetime and prediction variables
  rename(count=count) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>%#pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))
## Write out the file
vroom_write(x=kaggle_submission_forest_wf, file="C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/final_forest.csv", delim=",")


#Boosted Trees and BART####################################################################
library(bonsai)
library(lightgbm)
L<-3
K<-3
maxNumXs <- 4
boost_recipe <- recipe(count ~ ., data = training_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(season = as.factor(season)) %>%
  step_time(datetime, features = c("hour")) %>%
  step_date(datetime, features = c("month")) %>%
  step_date(datetime, features = c("dow")) %>%
  step_rm(datetime)%>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("regression")

bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("regression")


## CV tune, finalize and predict here and save results
## Create a workflow with model & recipe
boost_wf <- workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_model)
## Set up grid of tuning values
# Define parameter ranges
#boost_params <- parameters(
#  tree_depth(),
#  trees(),
#  learn_rate()
#)

# Build a grid with L levels for each
#mygrid <- grid_regular(boost_params, levels = L)
## Set up K-fold CV
folds <- vfold_cv(training_data, v = K, repeats=1)

#Chat Code##################
boost_params <- parameters(
  tree_depth(range = c(3L, 8L)),
  trees(range = c(100L, 500L)),
  learn_rate(range = c(-2, -0.5)) # log10 scale
)

mygrid <- grid_random(boost_params, size = 100)

CV_results <- boost_wf %>%
  tune_grid(
    resamples = folds,
    grid = mygrid,
    metrics = metric_set(rmse, mae),
    control = control_grid(verbose = TRUE, allow_par = TRUE)
  )
#############################
## Run the CV
#CV_results <- boost_wf %>%
#  tune_grid(resamples=folds,
#            grid=mygrid,
#            metrics=metric_set(rmse, mae)) #Or leave metrics NULL


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric="rmse")
## Finalize workflow and predict
final_boost_wf <-boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=training_data)

## Predict
final_boost_preds <- final_boost_wf %>%
  predict(new_data = test_data) %>%
  mutate(count = exp(.pred)) %>%   # back-transform log scale
  select(count)

#Kaggle Format
kaggle_submission_boost_wf<- final_boost_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, count) %>% #Just keep datetime and prediction variables
  rename(count=count) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>%#pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))
## Write out the file
vroom_write(x=kaggle_submission_boost_wf, file="C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/final_boost.csv", delim=",")


#Stacking##########################################################################################################
## Libraries
library(agua) #Install if necessary
stack_recipe <- recipe(count ~ ., data = training_data) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(season = as.factor(season)) %>%
  step_time(datetime, features = c("hour")) %>%
  step_date(datetime, features = c("month")) %>%
  step_date(datetime, features = c("dow")) %>%
  step_rm(datetime)%>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

## Initialize an h2o session
h2o::h2o.init()

## Define the model
## max_runtime_secs = how long to let h2o.ai run
## max_models = how many models to stack
auto_model <- auto_ml() %>%
set_engine("h2o", max_runtime_secs=600, max_models=20) %>%
set_mode("regression")

## Combine into Workflow
automl_wf <- workflow() %>%
add_recipe(stack_recipe) %>%
add_model(auto_model) %>%
fit(data=training_data)

## Predict
final_stack_preds <- automl_wf %>%
  predict(new_data = test_data) %>%
  mutate(count = exp(.pred)) %>%   # back-transform log scale
  select(count)

#Kaggle Format
kaggle_submission_stack_wf<- final_stack_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, count) %>% #Just keep datetime and prediction variables
  rename(count=count) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>%#pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))
## Write out the file
vroom_write(x=kaggle_submission_stack_wf, file="C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/final_stack.csv", delim=",")




#Old Stuff Predictions################################################################
# Workflow
bike_workflow <- workflow() %>%
  add_model(linear_reg() %>% set_engine("lm")) %>%
  add_recipe(my_recipe)

# Fit model
bike_fit <- fit(bike_workflow, data = training_data)

# Predict on test data
bike_predictions <- predict(bike_fit, new_data = test_data)

# Back-transform
bike_predictions <- bike_predictions %>%
  mutate(count = exp(.pred)) %>%  # back-transform log scale
  select(count)
# Define model spec (not fitted yet)
my_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")


#Kaggle Format
kaggle_submission <- bike_predictions %>%
bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, count) %>% #Just keep datetime and prediction variables
  rename(count=count) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) #pointwise max of (0, prediction)
## Write out the file9
vroom_write(x=kaggle_submission, file="C:/Users/lasso/OneDrive/Documents/Fall 2025/Stat 348/BikeShareUpdated/LinearPreds.csv", delim=",")








#EDA######################################################################################################
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