# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:48:34 2017
@author: Igor Busquets LML
"""

#Random Forest Regression Model
#Non linear and  Non continuos regression model
# random forest is a n_estimators team of regression trees

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set #Data set is too small

# Feature Scaling - not required

#Creating the regressor
from sklearn.ensemble import RandomForestRegressor
# n_estimators = number of trees, usually starts with 500 but can have less for start
#random_state keeps the results the same of the tutorial
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)
# Fitting the Regression Model to the dataset
 
# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()