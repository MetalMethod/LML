'''
@author: Igor Buquets LML

# Support Vector Regression Model
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

############# DATA PREPROCESSING
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set 
#not required, the dataset is too small

### Feature Scaling IS required because SVR Class does not do it
#from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
#the reshape WAS NOT IN TUTORIAL, required for matching the size of feature to scale
y = sc_y.fit_transform(y.reshape(-1,1))

############# Creating the regressor
#from sklearn.svm import SVR

#param C is penalty
#param kernel is the default kernel
regressor = SVR(kernel = 'rbf')

#############  Fitting the SVR regressor to the dataset
regressor.fit(X, y)

############## Predicting a new result
#apply the predict to the scale feature
#first param must be a array, so the value 6.5 is not enough
#numpy array creats a array with desired value
# also requires to (inverse) scale back to see the results in original scale
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

##### Visualising the SVR Regression results (for higher resolution and smoother curve)
#HIGH RESOLUTION CURVE
X_grid = np.arange(min(X), max(X), 0.1)
#turn X_grid from a vector to a matrix - plot requires a matrix
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

'''
#LOW RESOLUTION CURVE
# Visualising the Regression results (for higher resolution and smoother curve)
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
'''
