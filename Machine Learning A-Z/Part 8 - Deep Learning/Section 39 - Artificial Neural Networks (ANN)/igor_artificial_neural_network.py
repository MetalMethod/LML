"""
Igor Busquets LML
"""

#Artifficial Neural Networks

################################################################
#what I think is needed:

# Data pre processing
#feature scaling
# a array of features X
# add Random weights
# Activation Function (relu or sigmoid)
# output to ^y with sigmoid
# feed hidden layers and repeat until reaching output y

# calculate Error and Cost function
# backpropagation and update weights

# run again for all the rows until Epoch.
################################################################

# DEPENDENCIES
import tensorflow as af
import theano as th
import keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# DATA PREPROCESSING
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Feature Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Enconding Country Column
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# Enconding Gender Column
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#create dummy variables
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#Avoid dummy variables trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ARTIFICIAL NEURAL NETWORK IMPLEMENTATION
#Sequencial - Initializes the Neural Network
from keras.models import Sequential
#Dense - Build the layers of the ANN
from keras.layers import Dense

#Initializes th ANN
classifier = Sequential()

#Add input and first hidden layers
#theres 11 inputs X and only 1 output y
#output_dim = number of nodes in the layer  = avg of input x and output y so avg(1,11)=6
#input_dim is compulsory to first layer onlym, because it doesnt know how many X it will recieve
classifier.add(Dense(output_dim = 6, init='uniform', activation = 'relu', input_dim = 11))

#Add second hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation = 'relu'))

#Add the output layer
classifier.add(Dense(output_dim = 1, init='uniform', activation = 'sigmoid'))

#Complle the ANN = Apply Stochastic Gradient Descent
#adam = type of Stochastic Gradient Descent
# loss = same as logistic regression - logaritic loss 
#metrics = crteria for evaluate model, criteria is accuracy...a array is expected
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )


#Initializes random weights close to 0

#Input layer will have 11 input nodes

#Activation function Relu

#Output function sigmoid















# PREDICTIONS
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
