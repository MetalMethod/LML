# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:32:05 2017

@author: Igor Busquets LML
"""

#Polynomial Regression Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split

#### Dataset Preprocessing

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#X must have 1:2 so it is considered a matrix, not a vector
X = dataset.iloc[:, 1:2].values
#y showud be a vector of salaries
y = dataset.iloc[:, 2].values

##### NO TRAINING SET / TEST SET SPLIT
# not enought data to split intro training / test
# a very acurate prediction is nrequired, so we need as much as a trining set possible.
# Feature Scaling

##### No feature scaling required, the lybrary does the job

### Build a Linear regression model for comparition with Polynomial
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()

#Fit the linear regressor to X and y
linear_regressor.fit(X, y)

### Build a Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4)
#exponencial matrix based in X
# creates a matrix with X in the midle and X squared on the right, and a column of ones in the left
X_poly = poly_regressor.fit_transform(X)
#second linear regressor to fit the X_poly and y
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)

##### Visualizes the linear regression results
# plot the salaries
plt.scatter(X, y, color = "red")
#predict using linear regressor (blue fit line)
plt.plot(X, linear_regressor.predict(X), color = "blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

##### Visualizing the Polynnomial Regressor fit curve
# plot the salaries

### Add a degree to polynomial
#change degree param to 3 and later 4 when creating the poly_regressor object
#the predictions that wiill match the pointing with more degrees.

######increases resolution of plot
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = "red")
#predict lienar regression using polynomial matrix of features(blue fit curve)
plt.plot(X_grid, linear_regressor_2.predict(poly_regressor.fit_transform(X_grid)), color = "blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Final prediction Linear Regression
# the target is predict position 6.5 from matrix X
linear_regressor.predict(6.5)

#Final prediction Polynomial Regression
linear_regressor_2.predict(poly_regressor.fit_transform(6.5))