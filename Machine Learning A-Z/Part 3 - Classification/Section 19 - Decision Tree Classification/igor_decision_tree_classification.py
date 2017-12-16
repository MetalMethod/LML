"""
Igor Busquets LML
"""

#Decision Tree Clasification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
#######DATA PREPOCESSING

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

#Matrix of features: select the right field columns
X = dataset.iloc[:, [2, 3]].values

#dependent variable
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
# does not require scaling
#from sklearn.preprocessing import StandardScaler
#REQUIRED ONLY BECAUSE OF PLOTTING RESOLUTION
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#####Creatin decision tree classifier
from sklearn.tree import DecisionTreeClassifier
#Decision tree based on entropy
#each node is the more homogenous as possible
#fully homogenous info
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


########Prediction X_test set using classifier
# vector of each prediction
y_pred = classifier.predict(X_test)

#### EVALUATING THE MODEL
#Making Confusion Matrix - correct predictios + incorrect predictions
#from sklearn.metrics import confusion_matrix
#params : y_true: value of y for true - y_pred = predictions
cm = confusion_matrix(y_test ,y_pred)
#cm is a confusion matrix, based on the test set, both 65 and 24 are the correct predicts, the other lower (11) values are the wrong ones


####### Visualizing the test dataset
#users are not linear distributed but he classifier is so the separator is a line and wont be 100% perfect
#green is purchase, red is not. Points are real users
from matplotlib.colors import ListedColormap
# data set aliases so replacing datasets becames easy
x_set, y_set = X_test, y_test

#prepare the grid: each pixel is a user of the example
# -1 minimal values of X and +1 maximum values of X-....same salary ... defines the range of pixels in the range
# 0.01 is the  resolution of the pixel points, smaller is more pixels
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))

#apply the classifier to all pixel of the region points
#draw the regions of separation of prediction areas classifying each pixel
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
            alpha = 0.55, cmap = ListedColormap(('red', 'green')))

#defines the limits of areas
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

#draw the real data points (red and green point)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Decision Tree Classifier (testset)')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
