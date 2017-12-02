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

########Fiting the mulitple linear regressor to the training dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fit to the dataset
regressor.fit(X_train, y_train)


##### VECTOR PREDICTIONS
y_pred = regressor.predict(X_test)

####### OPTIMIZATION
import statsmodels.formula.api as sn
#add constant b0 from equasion of multiple regression to matrix of features X
# add a rray of 50 lines and one column
#ones(returns a  matrix of one column of 1 ) - first parameter is shape, that in this case is 
#a matrix, so it requires its size of 50 lines and 1 column
#requires cast to int
#append aad a column to the end so the trick is to add X
X = numpy.append(arr = numpy.ones(([50, 1])).astype(int), values = X,  axis = 1 )

###Backward elimination
#Optimal matrix of features, only contain dependeble with high impact
#remove the not relevant statiscaly
#initialize the result matrix with its columns indexes
X_opt = X[:, [0,1,2,3,4,5]]

## STEP 2 FIT the regressor to the X_opt
regressor_OLS = sn.OLS(endog = y, exog = X_opt).fit()

#STEP 3 look for y_pred with highest P value
regressor_OLS.summary()

#STEP 4 look for highest P value (wich is bad) and remove it
X_opt = X[:, [0,1,3,4,5]]

#STEP 5 FIT model without this highest P value variable
regressor_OLS = sn.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#HOMEWORK
# remove all columns with P > 0.05
#STEP 3 and 4 look for highest P value (wich is bad) and remove it
X_opt = X[:, [0,3,4,5]]

#STEP 5 FIT model without this highest P value variable
regressor_OLS = sn.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#STEP 3 and 4 look for highest P value (wich is bad) and remove it
X_opt = X[:, [0,3,5]]

#STEP 5 FIT model without this highest P value variable
regressor_OLS = sn.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#STEP 3 and 4 look for highest P value (wich is bad) and remove it
X_opt = X[:, [0,3]]

#STEP 5 FIT model without this highest P value variable
regressor_OLS = sn.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

## trying of prediction
#y_pred_OLS = regressor_OLS.predict(X_test)

#######predictions vs testset
######coordinates of observation
#plt.scatter(X_test, X_test, color = 'red')

#######blue regression line 
#####must be X_train so it compares to the regression
#plt.plot(X_train, X_opt, color = 'blue')
#plt.title('Possibel Profit vs investments (Test set)')
#plt.xlabel('Investments')
#plt.ylabel('Profit')
#plt.show()