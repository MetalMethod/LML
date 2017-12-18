"""
Created on Mon Dec 18 19:56:31 2017 @ Mouse´s house
Igor Busquets LML
"""

#Hierarchical CLustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch

#Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')

#matrix of selected features = 2 last columns
X = dataset.iloc[:, [3,4]].values

#####DENDROGRAM
#Finding the optimal number of clusters
#as it is hierarchical clustering, using Plot the Dendogram
#import scipy.cluster.hierarchy as sch
#'ward' is a method to tries the minimize the variance within each cluster
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()
#analysis of dendrogram
# find the higher vertical line that does not have a horizontal line crossing it
# count the number of vertical lines in that distance = number of optimal clustering
# in this case is 5

###### Fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
#create the algorithm object
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
#fitting
y_hc = hc.fit_predict(X)
#y_hc vector shows wich row entry has each cluster number 

######Visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'TARGET')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')

plt.title('Clusters of clients (Hierarchical)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
