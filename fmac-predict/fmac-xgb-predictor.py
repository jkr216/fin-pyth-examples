import pandas as pd
import numpy as np
import sklearn
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def predict(df, state_1_predictor, state_2_predictor, state_predicted):
    X = df[[state_1_predictor, state_2_predictor]]
    y = df[state_predicted]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results = pd.DataFrame({'predicted':preds, 'actual':y_test})
    nrow = y_test.shape[0]
    results['date'] = df[-nrow:]['date'] 
    return(results)
