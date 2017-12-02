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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


###### Data Preprocessing
# Importing the dataset
dataset = pandas.read_csv('50_Startups.csv')
#matrix of features - dependent variables
# columns 0 to 3
X = dataset.iloc[:, :-1].values

#independent variable 
y = dataset.iloc[:, 4].values


#ENCODE CATEGORICAL DATA - Encode data so there are no string columns
label_encoder_X = LabelEncoder()
# encode the first column
X[:,3] = label_encoder_X.fit_transform(X[:, 3])
# transform the encoded column into 3 different columns populated with 1 and 0 only
one_hot_encoder = OneHotEncoder(categorical_features = [3])
X = one_hot_encoder.fit_transform(X).toarray() 

#AVOIDING DUMMY TRAP
#remove the first column of X, one of the dummy encoded 
X = X[:, 1:]

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



