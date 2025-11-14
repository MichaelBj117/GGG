library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(glmnet)
library(rpart)
library(embed)
library(kernlab)


train_data <- vroom("GitHub/GGG/train.csv")
test_data <- vroom("GitHub/GGG/test.csv")

ggplot(data = train_data, aes(x = bone_length, y = hair_length)) +
  geom_point()


ggg_recipe <- recipe(type ~ ., data = train_data) |>
  step_rm(id) |>
  step_unknown(all_nominal_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(GGG_recipe)
baked_data <- bake(prepped_recipe, new_data=train_data)
ncol(baked_data)

svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmL_workflow <- workflow() %>% 
  add_recipe(GGG_recipe) %>%
  add_model(svmLinear) 

tuning_grid <- grid_regular(cost(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- svmL_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_workflow <- svmL_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

GGG_predictions <- predict(final_workflow,
                              new_data=test_data,
                              type="class") # "class" or "prob"

kaggle_sub <- bind_cols(test_data$id, GGG_predictions[1]) %>% 
  rename(type = .pred_class) %>% 
  rename(id = ...1)

vroom_write(kaggle_sub,
            file="GitHub/GGG/SVML3.csv", delim=",")
