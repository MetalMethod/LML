# -*- coding: utf-8 -*-
"""
Igor Busquets
Learning Machine Learning
Data preprocessing template
"""

#Import libraries
import numpy
import matplotlib
import pandas
from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split #model_selection replaces cross_validation that is deprecated
from sklearn.preprocessing import StandardScaler

#Import dataset
dataset = pandas.read_csv("Data.csv")

#Matrix of features X and y
#X are the dependent variables, Y independent variables
#Add all the rows and all the columns except the last one ( -1)
X = dataset.iloc[:, :-1].values
#Create the dependent variables vector - last column only
Y = dataset.iloc[:, 3].values

#MISSING DATA
#Replace missing data with the mean of the column field
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#fit imputer to data vector X, all the rows of columns 2 and 3(index 1 and 3excluded) of table data
imputer = imputer.fit(X[:, 1:3])
#Replace missing data by the mean
X[:, 1:3] = imputer.transform(X[:, 1:3])

#ENCODE CATEGORICAL DATA - Encode data so there are no strin g columns
label_encoder_X = LabelEncoder()
# encode the first column
X[:,0] = label_encoder_X.fit_transform(X[:, 0])
# transform the encoded column into 3 different columns populated with 1 and 0 only
one_hot_encoder = OneHotEncoder(categorical_features = [0])
X = one_hot_encoder.fit_transform(X).toarray() 

#label encoder only for column 4
label_encoder_Y = LabelEncoder()
# 0 to No, 1 to Yes
Y = label_encoder_Y.fit_transform(Y)

#SPLIT TRAINING / TEST SET
#Separate the data set into training and test
#split 20 percento of dataset to testset
#random state makes the result be equal to the tutorial'
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#FEATURE SCALING - sets all values to same scale, that is from -1 to 1
#Euclidean distance of values must be close to each
#Scaling can be Standardisation and Normalization.
scaler_X = StandardScaler()
#Apply fit to data then later transform
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)










