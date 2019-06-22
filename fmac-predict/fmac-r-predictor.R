library(tidyverse)
library(parsnip)
library(rsample)
library(yardstick)

r_predictor = function(df, state_1_predictor, state_2_predictor, state_predicted) {

  df_split <- 
    df %>%
    rsample::initial_time_split(prop = 0.8)
  
  train_data <- training(df_split)
  test_data  <- testing(df_split)
  
  linear_lm <-
    linear_reg(mode = "regression") %>%
    set_engine("lm") %>%
    fit(state_predicted ~ state_1_predictor, state_2_predictor, data = train_data)
  
  results <- linear_lm %>% 
    predict(new_data = test_data) %>%
    bind_cols(test_data %>% select(state_predicted),
              test_data %>% ungroup() %>% select(date))
    
  return(results)
}
