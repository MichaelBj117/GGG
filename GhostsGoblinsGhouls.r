# Useful libraries for this analysis
library(vroom)
library(recipes)
library(tidyverse)
library(tidymodels)
library(ggmosaic)
library(skimr)
library(dplyr)
library(DataExplorer)
library(corrplot)
library(embed)
library(discrim)
library(kernlab)
library(themis)

# Read in test and training data
gggTrain <- vroom("~/Documents/STAT 348/Ghosts-Goblins-Ghouls-Kaggle/ghouls-goblins-and-ghosts-boo/train.csv.zip")
gggTest <- vroom("~/Documents/STAT 348/Ghosts-Goblins-Ghouls-Kaggle/ghouls-goblins-and-ghosts-boo/test.csv.zip")
gggTrain$type <- as.factor(gggTrain$type)

## EDA ##
head(gggTrain)
skim(gggTrain)
glimpse(gggTrain)
ggplot(data=gggTrain, aes(x = type, y = bone_length)) +
  geom_boxplot()
ggplot(data = gggTrain) +
  geom_mosaic(aes(x = product(color), fill = type))

## Recipe ##
ggg_recipe <- recipe(type ~ ., data = gggTrain) |>
  update_role(id, new_role = "ID") |>
  step_rm(id) |>
  step_unknown(all_nominal_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())
  
  
## Naive Bayes ##
nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) |>
  set_mode("classification") |>
  set_engine("klaR")

nb_wf <- workflow() |>
  add_recipe(ggg_recipe) |>
  add_model(nb_model)

nb_grid_of_tuning_params <- grid_regular(
  Laplace(range = c(0.001,2)),
  smoothness(range = c(0.001,2)),
  levels = 5
)

folds <- vfold_cv(gggTrain, v = 5)

nb_CV_results <- nb_wf |>
  tune_grid(
    resamples = folds,
    grid = nb_grid_of_tuning_params,
    metrics = metric_set(accuracy))

nb_bestTune <- nb_CV_results |>
  select_best(metric = "accuracy")

nb_final_wf <- nb_wf |>
  finalize_workflow(nb_bestTune) |>
  fit(data = gggTrain)

nb_class_preds <- predict(nb_final_wf, 
                          new_data = gggTest, 
                          type = "class")

submission <- gggTest |>
  select(id) |>
  bind_cols(nb_class_preds) |>
  rename(type = .pred_class)

vroom_write(submission,
            file = "~/Documents/STAT 348/Ghosts-Goblins-Ghouls-Kaggle/nb_5.csv",
            delim = ",")


## SVMS ##

# SVM poly
svm_poly_model <- svm_poly(degree = tune(), cost = tune(), scale = FALSE) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_poly_wf <- 
  workflow() %>%
  add_model(svm_poly_model) %>%
  add_recipe(ggg_recipe)

poly_grid_of_tuning_params <- grid_regular(
  degree(range = c(1,2)),
  cost(range = c(0.01,1))
)

folds <- vfold_cv(gggTrain, v = 5)

poly_CV_results <- svm_poly_wf |>
  tune_grid(
    resamples = folds,
    grid = poly_grid_of_tuning_params,
    metrics = metric_set(accuracy))

poly_bestTune <- poly_CV_results |>
  select_best(metric = "accuracy")

poly_final_wf <- svm_poly_wf |>
  finalize_workflow(poly_bestTune) |>
  fit(data = gggTrain)

poly_class_preds <- predict(poly_final_wf, 
                          new_data = gggTest, 
                          type = "class")

submission <- gggTest |>
  select(id) |>
  bind_cols(poly_class_preds) |>
  rename(type = .pred_class)

vroom_write(submission,
            file = "~/Documents/STAT 348/Ghosts-Goblins-Ghouls-Kaggle/svm_poly_2.csv",
            delim = ",")

# SVM Radial
svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

radial_wf <- workflow() |>
  add_recipe(ggg_recipe) |>
  add_model(svmRadial)

rbf_grid_of_tuning_params <- grid_regular(
  rbf_sigma(range = c(0.01, 0.1)),
  cost(range = c(0.01, 1)),
  levels = 5
)

folds <- vfold_cv(gggTrain, v = 5)

rbf_CV_results <- radial_wf |>
  tune_grid(
    resamples = folds,
    grid = rbf_grid_of_tuning_params,
    metrics = metric_set(accuracy))

rbf_bestTune <- rbf_CV_results |>
  select_best(metric = "accuracy")

rbf_final_wf <- radial_wf |>
  finalize_workflow(rbf_bestTune) |>
  fit(data = gggTrain)

rbf_class_preds <- predict(rbf_final_wf, 
                            new_data = gggTest, 
                            type = "class")

submission_rbf <- gggTest |>
  select(id) |>
  bind_cols(rbf_class_preds) |>
  rename(type = .pred_class)

vroom_write(submission_rbf,
            file = "~/Documents/STAT 348/Ghosts-Goblins-Ghouls-Kaggle/svm_rbf.csv",
            delim = ",")

# SVM Linear
svmLinear <- svm_linear(cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

linear_wf <- workflow() |>
  add_recipe(ggg_recipe) |>
  add_model(svmLinear)

linear_grid_of_tuning_params <- grid_regular(
  cost(range = c(0.01, 1)),
  levels = 5
)

folds <- vfold_cv(gggTrain, v = 5)

linear_CV_results <- linear_wf |>
  tune_grid(
    resamples = folds,
    grid = linear_grid_of_tuning_params,
    metrics = metric_set(accuracy))

linear_bestTune <- linear_CV_results |>
  select_best(metric = "accuracy")

linear_final_wf <- linear_wf |>
  finalize_workflow(linear_bestTune) |>
  fit(data = gggTrain)

linear_class_preds <- predict(linear_final_wf, 
                           new_data = gggTest, 
                           type = "class")

submission_linear <- gggTest |>
  select(id) |>
  bind_cols(linear_class_preds) |>
  rename(type = .pred_class)

vroom_write(submission_linear,
            file = "~/Documents/STAT 348/Ghosts-Goblins-Ghouls-Kaggle/svm_linear.csv",
            delim = ",")
