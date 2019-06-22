import pandas as pd
import numpy as np
import sklearn
import datetime
import janitor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

freddie = pd.read_excel("http://www.freddiemac.com/fmac-resources/research/docs/State_and_US_SA.xls", skiprows = 5, nrows = 532)

freddie['date'] = freddie['Month'].apply(lambda x:datetime.datetime.strptime(x,"%YM%m"))

freddie = janitor.clean_names(freddie)

freddie.drop(['month'], axis=1, inplace = True)

freddie = freddie[freddie['date'] > '1989-12-31']

def predict(df, state_1_predictor, state_2_predictor, state_predicted, test_size):
    X = df[[state_1_predictor, state_2_predictor]]
    y = df[state_predicted]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, shuffle = False)
    model = LinearRegression()
    model.fit(X_train, y_train) 
    preds = model.predict(X_test)
    results = pd.DataFrame({'predicted':preds, 'actual':y_test})
    nrow = y_test.shape[0]
    results['date'] = df[-nrow:]['date'] 
    return(results)
