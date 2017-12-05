'''
Igor Busquets LML 28/11/2017
Simple Linear Regression 
my very first machine learing model

'''

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
####DATA PREPROCESSING
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

#Matrix of Features X independent variable is the Salary
X = dataset.iloc[:, :-1].values

#Vector Y - dependent variable Experience
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set

#test_size = 1/3 = 10 in total for testset
#random_state = 0 so the results are the same from the tutorial
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
#libs does the job and data preprocessing is finished
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#####FIT LINEAR REGRESSION TO TRAINING SET
#The models learns the correlation from the training set
#from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

######Predicting the testset results
#vector of predict 
#value of predicted salaries
#X is matrix of features
#y_pred is convention for naming the future desirable(independent variable) value
# y_pred are predictions of the TEST TEST
y_pred = regressor.predict(X_test)

#######Visualizing the training results
# import matplotlib.pyplot as plt
#x experience
#y salary
# observation red
# regression blue
plt.scatter(X_train, y_train, color = 'red')

#predictions salaries of the trainingset
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#predictions salaries of the testset
#coordinates of observation
plt.scatter(X_test, y_test, color = 'red')

#blue regression line 
#must be X_train so it compares to the regression
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()






