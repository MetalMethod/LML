# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:07:50 2017
@author: Igor Busquets LML
"""

#Multiple Linear Regression

# Importing the libraries
import numpy
import matplotlib.pyplot as plt
import pandas

###### Data Preprocessing
# Importing the dataset
dataset = pandas.read_csv('50_Startups.csv')
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


